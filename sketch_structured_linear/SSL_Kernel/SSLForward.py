import torch
import triton
import triton.language as tl
from triton.runtime import driver
from typing import NamedTuple
import argparse

A = 11211
B = 11311
C = 11411
D = 11511

device_index = 0
benchmarking = False
default_vec = 8


def matmul(input: torch.tensor, weight: torch.tensor, log2_redn_factor: int):
    redn_factor = 2**log2_redn_factor
    assert (weight.shape[0] >= input.shape[1] / redn_factor)
    bias = torch.randn(weight.shape[0], device='cuda', dtype=torch.float16)
    return ssl_forward_tl(input, weight, bias, input.shape[0], input.shape[1], weight.shape[1], redn_factor, A, B, C, D, allow_autotune=True)


def ssl_forward_tl(input: torch.tensor, weight: torch.tensor, bias: torch.tensor,
                   M: int, K: int, N: int,
                   redn_factor: int,
                   R3: int, R2: int, R1: int, R0: int,
                   BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32,
                   allow_tf32: bool = False, allow_autotune: bool = False, VEC: int = default_vec, BIAS: bool = True,
                   ) -> torch.tensor:
    '''
      Compute input_tensor x weight and return an output tensor

      Args:
        input (Tensor): A MxK tensor
        weight (Tensor): A KxN tensor
        bias (Tensor): A 1xN tensor
        M, K, N, H (int): Matrix dimensions
        R3, R2, R1, R0 (int): Random numbers
        allow_tf32 (bool): If tensor core is allowed
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M: Matrix tiling parameters for performance tunning

      Returns:
        output (Tensor): A MxN tensor
    '''
    # allocates output
    output = torch.zeros((M, N), device=input.device, dtype=input.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M'])
            * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

    if allow_autotune:
        ssl_forward_kernel_tune[grid](
            input, weight, bias, output,
            M, N, K, K // redn_factor,
            input.stride(0), input.stride(1),
            # stored in column major format / transpose
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            allow_tf32=allow_tf32,
            R3=R3, R2=R2, R1=R1, R0=R0,
            GROUP_SIZE_M=1,
            VEC=VEC,
            BIAS=BIAS,
            redn_factor=redn_factor
        )
    else:
        ssl_forward_kernel_notune[grid](
            input, weight, bias, output,
            M, N, K, K // redn_factor,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            allow_tf32=allow_tf32,
            R3=R3, R2=R2, R1=R1, R0=R0,
            num_stages=4,
            num_warps=4,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=1,
            VEC=VEC,
            BIAS=BIAS,
            EVEN_K=(K % (BLOCK_SIZE_K * redn_factor) == 0),
            redn_factor=redn_factor,
        )
    return output


def _generate_configs():
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    cap = capability[0] * 10 + capability[1]
    if cap == 90:
        BLOCK_SIZE_M = [32, 64, 128, 256]
        BLOCK_SIZE_N = [32, 64, 128, 256]
        BLOCK_SIZE_K = [16, 32, 64]
    elif cap == 89:
        BLOCK_SIZE_M = [16, 32, 64, 128]
        BLOCK_SIZE_N = [16, 32, 64, 128]
        BLOCK_SIZE_K = [16, 32, 64]
    else:
        BLOCK_SIZE_M = [16, 32, 64, 128, 256]
        BLOCK_SIZE_N = [16, 32, 64, 128, 256]
        BLOCK_SIZE_K = [16, 32, 64]
    configs = []
    for bm in BLOCK_SIZE_M:
        for bn in BLOCK_SIZE_N:
            for bk in BLOCK_SIZE_K:
                for stages in [3, 4]:
                    for warps in [4, 8]:
                        configs.append(triton.Config({'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_K': bk, 'BLOCK_SIZE_N': bn}, num_stages=stages, num_warps=warps))
    return configs


def _early_config_prune(configs, named_args, **kwargs):
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    cap = capability[0] * 10 + capability[1]
    max_shared_mem = driver.active.utils.get_device_properties(device)["max_shared_mem"]
    pruned_configs = []
    K = named_args['K']
    element_size = named_args['a_ptr'].element_size()
    redn_factor = kwargs['redn_factor']
    for config in configs:
        BLOCK_SIZE_K = config.kwargs['BLOCK_SIZE_K']
        BLOCK_SIZE_M = config.kwargs['BLOCK_SIZE_M']
        BLOCK_SIZE_N = config.kwargs['BLOCK_SIZE_N']
        num_stages = config.num_stages
        if K % (BLOCK_SIZE_K * redn_factor) != 0:
            continue
        if (K < 1024 and (config.kwargs['BLOCK_SIZE_K'] > 32 and config.num_stages > 3)) or (K >= 1024 and config.num_stages < 4):
            continue
        if config.kwargs['BLOCK_SIZE_M'] == 256 and config.kwargs['BLOCK_SIZE_N'] == 256:
            # avoid large block sizes
            continue
        if config.kwargs['BLOCK_SIZE_K'] == 16 and K % (32 * kwargs['redn_factor']) == 0:
            # small block size k is only considered when larger block size is not possible
            continue
        if cap >= 80:
            if redn_factor >= 8:
                # inner pipeline = 4
                estimated_shared_mem = (4 * BLOCK_SIZE_K * BLOCK_SIZE_M + num_stages * BLOCK_SIZE_K * BLOCK_SIZE_N) * element_size
            elif redn_factor >= 4:
                # inner pipeline = 3
                estimated_shared_mem = (3 * BLOCK_SIZE_K * BLOCK_SIZE_M + num_stages * BLOCK_SIZE_K * BLOCK_SIZE_N) * element_size
            elif redn_factor >= 2:
                estimated_shared_mem = (2 * BLOCK_SIZE_K * BLOCK_SIZE_M + num_stages * BLOCK_SIZE_K * BLOCK_SIZE_N) * element_size
            else:
                estimated_shared_mem = (num_stages * BLOCK_SIZE_K * BLOCK_SIZE_M + num_stages * BLOCK_SIZE_K * BLOCK_SIZE_N) * element_size
        else:
            estimated_shared_mem = num_stages * (BLOCK_SIZE_K * BLOCK_SIZE_M + BLOCK_SIZE_K * BLOCK_SIZE_N) * element_size
        if estimated_shared_mem > max_shared_mem:
            continue
        pruned_configs.append(config)
    return pruned_configs


def _metadata_fn(
    grid: tuple,
    metadata: NamedTuple,
    args: dict
):
    bm = args['BLOCK_SIZE_M']
    bn = args['BLOCK_SIZE_N']
    bk = args['BLOCK_SIZE_K']
    num_stages = metadata.num_stages
    num_warps = metadata.num_warps
    return {"name": f"ssl_bm:{bm},bn:{bn},bk:{bk},stages:{num_stages},warps:{num_warps}"}

@triton.autotune(
    configs=_generate_configs(), key=['M', 'N', 'K_RED'],
    prune_configs_by={
        'early_config_prune': _early_config_prune
    },
    warmup=1,
    rep=10,
)
@triton.heuristics({
    "EVEN_K": lambda META: META['K_RED'] % META['BLOCK_SIZE_K'] == 0,
})
@triton.jit(launch_metadata=_metadata_fn)
def ssl_forward_kernel_pretune(
    # Pointers to matrices
    a_ptr, b_ptr, o_ptr, c_ptr,
    # Best block_sizes 
    m_blk_ptr, k_blk_ptr, n_blk_ptr, 
    # Matrix dimensions
    M, N, K, K_RED,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    allow_tf32: tl.constexpr,
    # Random numbers
    R3: tl.constexpr, R2: tl.constexpr, R1: tl.constexpr, R0: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, VEC: tl.constexpr, EVEN_K: tl.constexpr,
    BIAS: tl.constexpr,
    redn_factor: tl.constexpr,
):
    ssl_forward_core(redn_factor=redn_factor, a_ptr=a_ptr, b_ptr=b_ptr, o_ptr=o_ptr, c_ptr=c_ptr, M=M, N=N, K=K,
                     stride_am=stride_am, stride_ak=stride_ak,
                     stride_bk=stride_bk, stride_bn=stride_bn,
                     stride_cm=stride_cm, stride_cn=stride_cn,
                     allow_tf32=allow_tf32,
                     R3=R3, R2=R2, R1=R1, R0=R0,
                     BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                     GROUP_SIZE_M=GROUP_SIZE_M, VEC=VEC, EVEN_K=EVEN_K, BIAS=BIAS)
    
    tl.store(m_blk_ptr, BLOCK_SIZE_M)
    tl.store(k_blk_ptr, BLOCK_SIZE_K)
    tl.store(n_blk_ptr, BLOCK_SIZE_N)



@triton.autotune(
    configs=_generate_configs(), key=['M', 'N', 'K_RED'],
    prune_configs_by={
        'early_config_prune': _early_config_prune
    },
    warmup=1,
    rep=10,
)
@triton.heuristics({
    "EVEN_K": lambda META: META['K_RED'] % META['BLOCK_SIZE_K'] == 0,
})
@triton.jit(launch_metadata=_metadata_fn)
def ssl_forward_kernel_tune(
    # Pointers to matrices
    a_ptr, b_ptr, o_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, K_RED,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    allow_tf32: tl.constexpr,
    # Random numbers
    R3: tl.constexpr, R2: tl.constexpr, R1: tl.constexpr, R0: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, VEC: tl.constexpr, EVEN_K: tl.constexpr,
    BIAS: tl.constexpr,
    redn_factor: tl.constexpr,
):
    ssl_forward_core(redn_factor=redn_factor, a_ptr=a_ptr, b_ptr=b_ptr, o_ptr=o_ptr, c_ptr=c_ptr, M=M, N=N, K=K,
                     stride_am=stride_am, stride_ak=stride_ak,
                     stride_bk=stride_bk, stride_bn=stride_bn,
                     stride_cm=stride_cm, stride_cn=stride_cn,
                     allow_tf32=allow_tf32,
                     R3=R3, R2=R2, R1=R1, R0=R0,
                     BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                     GROUP_SIZE_M=GROUP_SIZE_M, VEC=VEC, EVEN_K=EVEN_K, BIAS=BIAS)


@triton.jit(launch_metadata=_metadata_fn)
def ssl_forward_kernel_notune(
    # Pointers to matrices
    a_ptr, b_ptr, o_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, K_RED,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    allow_tf32: tl.constexpr,
    # Random numbers
    R3: tl.constexpr, R2: tl.constexpr, R1: tl.constexpr, R0: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, VEC: tl.constexpr, EVEN_K: tl.constexpr,
    BIAS: tl.constexpr,
    redn_factor: tl.constexpr,
):
    ssl_forward_core(redn_factor=redn_factor, a_ptr=a_ptr, b_ptr=b_ptr, o_ptr=o_ptr, c_ptr=c_ptr, M=M, N=N, K=K,
                     stride_am=stride_am, stride_ak=stride_ak,
                     stride_bk=stride_bk, stride_bn=stride_bn,
                     stride_cm=stride_cm, stride_cn=stride_cn,
                     allow_tf32=allow_tf32,
                     R3=R3, R2=R2, R1=R1, R0=R0,
                     BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                     GROUP_SIZE_M=GROUP_SIZE_M, VEC=VEC, EVEN_K=EVEN_K, BIAS=BIAS)


@triton.jit
def ssl_acc_a(
    offs_k, IDX, IDX1,
    stride_ak, ck, k, a_ptrs,
    VEC: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, redn_factor: tl.constexpr, EVEN_K: tl.constexpr, K: tl.constexpr
):
    offset = ((((offs_k + (IDX + IDX1) * VEC)) % BLOCK_SIZE_K) * stride_ak)[None, :]
    if EVEN_K:
        a = tl.load(a_ptrs + offset)
    else:
        a = tl.load(a_ptrs + offset, mask=offset < K - (
            k * redn_factor + ck) * BLOCK_SIZE_K, other=0.0)
    return a

@triton.jit
def ssl_pipeline_a(
    offs_k, IDX, IDX1,
    stride_ak, k, a_ptrs, a_ptr,
    R0: tl.constexpr, VEC: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, redn_factor: tl.constexpr, EVEN_K: tl.constexpr, K: tl.constexpr, num_stages: tl.constexpr
):
    a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=a_ptr.dtype.element_ty)
    for ck in tl.range(0, redn_factor, num_stages=num_stages):
        IDX1 += R0
        offset = ((((offs_k + (IDX + IDX1) * VEC)) % BLOCK_SIZE_K) * stride_ak)[None, :]
        if EVEN_K:
            a += tl.load(a_ptrs + offset)
        else:
            a += tl.load(a_ptrs + offset, mask=offset < K - (
                k * redn_factor + ck) * BLOCK_SIZE_K, other=0.0)
        a_ptrs += BLOCK_SIZE_K * stride_ak
    return a


@triton.jit
def ssl_forward_core(
    redn_factor: tl.constexpr,
    # Pointers to matrices
    a_ptr, b_ptr, o_ptr, c_ptr,
    # Matrix dimensions
    M: int, N: int, K: int,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    allow_tf32: tl.constexpr,
    # Random numbers
    R3: int, R2: int, R1: int, R0: int,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, VEC: tl.constexpr, EVEN_K: tl.constexpr, BIAS: tl.constexpr
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk
                      + offs_bn[None, :] * stride_bn)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    IDX = (R3 + R2 * pid_n)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * redn_factor)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        IDX += R1
        IDX1 = 0

        if redn_factor == 1:
            IDX1 += R0
            a = ssl_acc_a(offs_k, 0, 0, stride_ak, 0, k, a_ptrs, VEC, BLOCK_SIZE_K, redn_factor, EVEN_K, K)
            a_ptrs += BLOCK_SIZE_K * stride_ak
        elif redn_factor == 2:
            IDX1 += R0
            a0 = ssl_acc_a(offs_k, IDX, IDX1, stride_ak, 0, k, a_ptrs, VEC, BLOCK_SIZE_K, redn_factor, EVEN_K, K)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            IDX1 += R0
            a1 = ssl_acc_a(offs_k, IDX, IDX1, stride_ak, 1, k, a_ptrs, VEC, BLOCK_SIZE_K, redn_factor, EVEN_K, K)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            a = a0 + a1
        elif redn_factor == 4:
            IDX1 += R0
            a0 = ssl_acc_a(offs_k, IDX, IDX1, stride_ak, 0, k, a_ptrs, VEC, BLOCK_SIZE_K, redn_factor, EVEN_K, K)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            IDX1 += R0
            a1 = ssl_acc_a(offs_k, IDX, IDX1, stride_ak, 1, k, a_ptrs, VEC, BLOCK_SIZE_K, redn_factor, EVEN_K, K)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            IDX1 += R0
            a2 = ssl_acc_a(offs_k, IDX, IDX1, stride_ak, 2, k, a_ptrs, VEC, BLOCK_SIZE_K, redn_factor, EVEN_K, K)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            IDX1 += R0
            a3 = ssl_acc_a(offs_k, IDX, IDX1, stride_ak, 3, k, a_ptrs, VEC, BLOCK_SIZE_K, redn_factor, EVEN_K, K)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            a = a0 + a1 + a2 + a3
        else:
            a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
            for ck in range(0, redn_factor):
                IDX1 += R0
                offset = ((((offs_k + (IDX + IDX1) * VEC)) % BLOCK_SIZE_K) * stride_ak)[None, :]
                if EVEN_K:
                    a += tl.load(a_ptrs + offset)
                else:
                    a += tl.load(a_ptrs + offset, mask=offset < K - (
                        k * redn_factor + ck) * BLOCK_SIZE_K, other=0.0)
                a_ptrs += BLOCK_SIZE_K * stride_ak
        if EVEN_K:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < (K // redn_factor) - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.

        accumulator += tl.dot(a, b, allow_tf32=allow_tf32, out_dtype=tl.float32)
        # Advance the ptrs to the next K block.
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if BIAS:
        bias = tl.load(o_ptr + offs_bn)
        accumulator = accumulator + bias[None, :]
    c = accumulator.to(c_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# unit tests

def unit_test_1():
    torch.manual_seed(0)
    S = 32
    a = torch.full((S, S), 1 / 100, device='cuda', dtype=torch.float16)
    b = torch.arange((S * S / 2), device='cuda',
                     dtype=torch.float16).reshape(int(S / 2), S).T
    b_full = matmul(torch.eye(S, device='cuda', dtype=torch.float16), b, 1)
    triton_output = matmul(a, b, 1)
    torch_output = torch.matmul(a, b_full)
    import matplotlib.pyplot as plt
    import numpy as np
    torch.save(b_full, "freq_b.pt")
    torch.save(triton_output, "triton_output.pt")
    torch.save(torch_output, "torch_output.pt")

    plt.imshow(np.array(b.T.cpu()))
    plt.show()
    plt.imshow(np.array(b_full.cpu()))
    plt.show()
    print(f"b_full={b_full}")
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-8):
        print("✅ Test 1 Triton and Torch match")
    else:
        print("❌ Test 1 Triton and Torch differ")


def unit_test_2():
    torch.manual_seed(0)
    S = 64 * 4
    a = torch.randn((S, S), device='cuda', dtype=torch.float16)
    # b = torch.arange(0, 1024, 1024/(S*S/2), device='cuda', dtype=torch.float16).reshape(int(S/2),S).T
    b = torch.randn((S, 64 * 2), device='cuda', dtype=torch.float16)
    b_full = matmul(torch.eye(S, device='cuda', dtype=torch.float16), b, 1)
    triton_output = matmul(a, b, 1)
    torch_output = torch.matmul(a, b_full)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

    print(torch.sum(b_full[:64, :] == b.T[:64, :]) / 64 / b_full.shape[1])
    print(torch.sum(b_full[64:128, :] == b.T[:64, :]) / 64 / b_full.shape[1])
    print(torch.sum(b_full[128:192, :]
          == b.T[64:128, :]) / 64 / b_full.shape[1])
    print(torch.sum(b_full[192:256, :]
          == b.T[64:128, :]) / 64 / b_full.shape[1])

    print(torch.sum(b_full[:64, :]) == torch.sum(b.T[:64, :]))
    print(torch.sum(b_full[64:128, :]) == torch.sum(b.T[:64, :]))
    print(torch.sum(b_full[128:192, :]) == torch.sum(b.T[64:128, :]))
    print(torch.sum(b_full[192:256, :]) == torch.sum(b.T[64:128, :]))


# unit_test_2()
# unit_test_1()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # Argument names to use as an x-axis for the plot
        x_names=['M', 'N', 'K'],
        # Different possible values for `x_name`
        x_vals=[(8192, 768, 3072), (8192, 3072, 768), (8192, 2304, 768)],
        # Argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # Possible values for `line_arg`
        line_vals=['cublas', 'cublas2', 'cublas4', 'cublas16', 'triton0', 'triton1', 'triton2', 'triton4'],
        # Label name for the lines
        line_names=['cuBLAS', 'cublas2', 'cublas4', 'cublas16', 'roast-comp,c=1', 'roast-comp,c=2',
                    'roast-comp,c=4', 'roast-comp,c=16'],
        # Line styles
        styles=[('green', '-'), ('green', '-'), ('green', '-'), ('green', '-'), ('blue', '-'), ('black', '-'),
                ('red', '-'), ('yellow', '-')],
        ylabel="ms",  # Label name for the y-axis
        # Name for the plot, used also as a file name for saving the plot.
        plot_name="matmul-performance",
        args={},
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)

    def bench(fn, *args):
        if benchmarking:
            return triton.testing.do_bench_cudagraph(lambda: fn(*args))
        else:
            fn(*args)
            return (0, 0, 0,)

    with proton.scope(f"M:{M},N:{N},K:{K}"):
        with proton.scope(provider):
            if provider == 'cublas':
                ms = bench(torch.matmul, a, b.T)
            if provider == 'cublas2':
                a = torch.randn((M, int(K / 2)), device='cuda', dtype=torch.float16)
                b = torch.randn((N, int(K / 2)), device='cuda', dtype=torch.float16)
                ms = bench(torch.matmul, a, b.T)
            if provider == 'cublas4':
                a = torch.randn((M, int(K / 4)), device='cuda', dtype=torch.float16)
                b = torch.randn((N, int(K / 4)), device='cuda', dtype=torch.float16)
                ms = bench(torch.matmul, a, b.T)
            if provider == 'cublas16':
                a = torch.randn((M, int(K / 16)), device='cuda', dtype=torch.float16)
                b = torch.randn((N, int(K / 16)), device='cuda', dtype=torch.float16)
                ms = bench(torch.matmul, a, b.T)
            if provider == 'triton0':
                ms = bench(matmul, a, b.T, 0)
            if provider == 'triton1':
                b = torch.randn((N, int(K / 2)), device='cuda',
                                dtype=torch.float16).cuda()
                ms = bench(matmul, a, b.T, 1)
            if provider == 'triton2':
                b = torch.randn((N, int(K / 4)), device='cuda',
                                dtype=torch.float16).cuda()
                ms = bench(matmul, a, b.T, 2)
            if provider == 'triton4':
                b = torch.randn((N, int(K / 16)), device='cuda',
                                dtype=torch.float16).cuda()
                ms = bench(matmul, a, b.T, 4)
    return ms


if __name__ == "__main__":
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    torch.cuda.set_device(device_index)
    # Init a parser with a profile option
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    import triton.profiler as proton
    benchmark.run(show_plots=True, print_data=True, save_path=".")
    if args.profile:
        benchmarking = False
        proton.start(name="ssl_forward", hook="triton")
        for _ in range(5):
            benchmark.run(show_plots=False, print_data=False, save_path=".")
        proton.finalize()
