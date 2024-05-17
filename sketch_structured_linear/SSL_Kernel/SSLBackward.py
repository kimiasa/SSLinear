from typing import Tuple
import torch
import triton
import triton.language as tl
#import SSLForward
import functools

A = 11211
B = 11311
C = 11411
D = 11511

allow_backward_autotune = True
override_block_size_k = True
default_vec_width = 8


def matmul(input: torch.tensor, weight: torch.tensor, output_grad: torch.tensor, log2_redn_factor: int):
    redn_factor = 2**log2_redn_factor
    assert (weight.shape[1] >= input.shape[1] / redn_factor)
    return ssl_backward_tl(input, weight, output_grad, input.shape[0], input.shape[1], weight.shape[0], redn_factor, A, B, C, D)


def torch_matmul(input: torch.tensor, full_weight: torch.tensor, output_grad: torch.tensor):
    return torch.matmul(output_grad, full_weight), torch.matmul(input.T, output_grad)


def ssl_backward_tl(input: torch.tensor, weight: torch.tensor, output_grad: torch.tensor,
                    M: int, K: int, N: int, redn_factor: int,
                    R3: int, R2: int, R1: int, R0: int,
                    allow_tf32: bool = True,
                    BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32,
                    VEC: int = default_vec_width) -> Tuple[torch.tensor, torch.tensor]:
    input_grad = ssl_backward_input_grad_tl(output_grad, weight, M, K, N, redn_factor, R3, R2, R1, R0, allow_tf32=allow_tf32,
                                            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                            VEC=VEC)
    weight_grad = ssl_backward_weight_grad_tl(input, output_grad, M, K, N, redn_factor, R3, R2, R1, R0, allow_tf32=allow_tf32,
                                              BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
                                              VEC=VEC)
    return input_grad, weight_grad


def _early_config_prune(configs, named_args, is_weight, **kwargs):
    pruned_configs = []
    FIXED_BLOCK_SIZE_K = 16 if 'FIXED_BLOCK_SIZE_K' not in kwargs else kwargs["FIXED_BLOCK_SIZE_K"]
    FIXED_BLOCK_SIZE_M = 16 if 'FIXED_BLOCK_SIZE_M' not in kwargs else kwargs["FIXED_BLOCK_SIZE_M"]
    FIXED_BLOCK_SIZE_N = 16 if 'FIXED_BLOCK_SIZE_N' not in kwargs else kwargs["FIXED_BLOCK_SIZE_N"]
    K = named_args['K']
    if not allow_backward_autotune:
        pruned_configs.append(triton.Config({'BLOCK_SIZE_M': FIXED_BLOCK_SIZE_M, 'BLOCK_SIZE_K': FIXED_BLOCK_SIZE_K, 'BLOCK_SIZE_N': FIXED_BLOCK_SIZE_N}, num_stages=4, num_warps=4))
    else:
        redn_factor = 1 if 'redn_factor' not in kwargs else kwargs["redn_factor"]
        for config in configs:
            BLOCK_SIZE_K = config.kwargs['BLOCK_SIZE_K']
            BLOCK_SIZE_M = config.kwargs['BLOCK_SIZE_M']
            BLOCK_SIZE_N = config.kwargs['BLOCK_SIZE_N']
            if not override_block_size_k and BLOCK_SIZE_K != FIXED_BLOCK_SIZE_K:
                # Skip this config because it is not the same as the one used in the forward pass
                # If override_block_size_k is True, it means we autotune regardless of the forward pass
                continue
            if K % (BLOCK_SIZE_K * redn_factor) != 0:
                # Skip this config because non-divisible K's will result in incorrect results
                continue
            # Skip inefficient configurations
            if is_weight:
                if BLOCK_SIZE_M == 128 or BLOCK_SIZE_M == 256 or BLOCK_SIZE_N == 32:
                    continue
            else:
                if BLOCK_SIZE_N == 128 or BLOCK_SIZE_N == 256 or BLOCK_SIZE_M == 32:
                    continue
            pruned_configs.append(config)
    if len(pruned_configs) == 0:
        if K % (32 * redn_factor) == 0:
            pruned_configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4))
        else:
            pruned_configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 16, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4))
    return pruned_configs


def _generate_configs():
    BLOCK_SIZE_M = [32, 64, 128, 256]
    BLOCK_SIZE_N = [32, 64, 128, 256]
    BLOCK_SIZE_K = [32, 64, 128, 256]
    configs = []
    for bm in BLOCK_SIZE_M:
        for bn in BLOCK_SIZE_N:
            for bk in BLOCK_SIZE_K:
                for stages in [3, 4]:
                    for warps in [4, 8]:
                        configs.append(triton.Config({'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_K': bk, 'BLOCK_SIZE_N': bn}, num_stages=stages, num_warps=warps))
    return configs


def ssl_backward_weight_grad_tl(input: torch.tensor, output_grad: torch.tensor,
                                M: int, K: int, N: int, redn_factor: int,
                                R3: int, R2: int, R1: int, R0: int,
                                allow_tf32: bool = True,
                                BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64, BLOCK_SIZE_K: int = 32,
                                VEC: int = 2) -> torch.tensor:
    '''
        Compute output_grad^T * input and return a weight_grad tensor

        Args:
            input (Tensor): A MxK tensor
            output_grad (Tensor): A MxN tensor
            M, K, N (int): Matrix dimensions
            redn_factor: Reduction factor
            R3, R2, R1, R0 (int): Random numbers
            allow_tf32 (bool): If tensor core is allowed
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Matrix tiling parameters for performance tunning
            VEC: Vectorization units

        Returns:
            weight_grad (Tensor): A N x (K // redn_factor) tensor
    '''

    # allocates output
    K_RED = K // redn_factor
    hashed_weight_grad = torch.zeros(
        (N, K_RED), device=output_grad.device, dtype=output_grad.dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            triton.cdiv(N, META['BLOCK_SIZE_N'])
            * triton.cdiv(K, META['BLOCK_SIZE_K']),
        )

    ssl_backward_weight_grad_kernel[grid](
        output_grad, input, hashed_weight_grad,
        M, N, K, K_RED,
        redn_factor,
        output_grad.stride(1), output_grad.stride(0),
        input.stride(0), input.stride(1),
        hashed_weight_grad.stride(0), hashed_weight_grad.stride(1),
        R3=R3, R2=R2, R1=R1, R0=R0,
        allow_tf32=allow_tf32,
        FIXED_BLOCK_SIZE_K=BLOCK_SIZE_K,
        FIXED_BLOCK_SIZE_M=BLOCK_SIZE_M,
        FIXED_BLOCK_SIZE_N=BLOCK_SIZE_N,
        VEC=VEC
    )

    return hashed_weight_grad


@triton.autotune(
    configs=_generate_configs(),
    reset_to_zero=['c_ptr'],
    # VEC_INT is not used but has to be a key to trigger recompilation
    # Every constant that is related to the computation correctness has to be a key
    key=['M', 'N', 'K_RED'],
    prune_configs_by={
        'early_config_prune': functools.partial(_early_config_prune, is_weight=True)
    },
    warmup=1,
    rep=10,
)
@triton.heuristics({
    "EVEN_N": lambda META: META['N'] % META['BLOCK_SIZE_N'] == 0,
    "EVEN_M": lambda META: META['M'] % META['BLOCK_SIZE_M'] == 0,
})
@triton.jit
def ssl_backward_weight_grad_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, K_RED,
    # Reduction factor
    redn_factor,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_an, stride_am,
    stride_bm, stride_bk,
    stride_cn, stride_ck,
    # Random numbers
    R3: tl.constexpr, R2: tl.constexpr, R1: tl.constexpr, R0: tl.constexpr,
    allow_tf32: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    FIXED_BLOCK_SIZE_M: tl.constexpr,
    FIXED_BLOCK_SIZE_N: tl.constexpr,
    FIXED_BLOCK_SIZE_K: tl.constexpr,
    VEC: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A^T x B.
    A has shape (M, N), B has shape (N, K) and C has shape (N, K//redn_factor)
    """
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    pid_k = pid % num_pid_k
    pid_n = pid // num_pid_k
    pid_k_red = pid_k // redn_factor
    pid_r = pid_k % redn_factor

    # [BLOCK_SIZE_N, BLOCK_SIZE_M]
    offs_an = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = a_ptr + offs_an[:, None] * \
        stride_an + offs_am[None, :] * stride_am

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    offs_bm = tl.arange(0, BLOCK_SIZE_M)
    offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = b_ptr + offs_bm[:, None] * \
        stride_bm + offs_bk[None, :] * stride_bk

    # [BLOCK_SIZE_N, BLOCK_SIZE_K]
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    for _ in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        if EVEN_M and EVEN_N:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a_mask = (offs_an[:, None] < N) & (offs_am[None, :] < M)
            b_mask = (offs_bm[:, None] < M) & (offs_bk[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        # We accumulate along the M dimension
        accumulator += tl.dot(a, b, allow_tf32=allow_tf32)
        # Advance the ptrs to the next M block
        a_ptrs += BLOCK_SIZE_M * stride_am
        b_ptrs += BLOCK_SIZE_M * stride_bm

    c = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    # TODO check corner cases and masking
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    IDX = R3 + R2 * pid_n + R1 + R1 * pid_k_red
    IDX1 = R0 + R0 * pid_r

    offs_ck = pid_k_red * BLOCK_SIZE_K + ((tl.arange(0, BLOCK_SIZE_K) + (BLOCK_SIZE_K - (((IDX + IDX1) * VEC)) % BLOCK_SIZE_K))) % BLOCK_SIZE_K

    if EVEN_N:
        c_ptrs = c_ptr + (offs_cn[:, None] * stride_cn) + (offs_ck[None, :] * stride_ck)
        tl.atomic_add(c_ptrs, c)
    else:
        c_ptrs = c_ptr + (offs_cn[:, None] * stride_cn) + (offs_ck[None, :] * stride_ck)
        c_mask = (offs_cn[:, None] < N) & (offs_ck[None, :] < K_RED)
        tl.atomic_add(c_ptrs, c, mask=c_mask)


#################### Input grad computing kernel ####################


def ssl_backward_input_grad_tl(output_grad: torch.tensor, weight: torch.tensor,
                               M: int, K: int, N: int, redn_factor: int,
                               R3: int, R2: int, R1: int, R0: int,
                               allow_tf32: bool = True,
                               BLOCK_SIZE_M: int = 64,
                               BLOCK_SIZE_N: int = 64,
                               BLOCK_SIZE_K: int = 32,
                               VEC: int = 2) -> torch.tensor:
    '''
        Compute output_grad x weight and return an input_grad tensor

        Args:
            output_grad (Tensor): A MxN tensor
            weight (Tensor): A NxK tensor
            M, K, N (int): matrix dimensions
            redn_factor: Reduction factor
            R3, R2, R1, R0 (int): random numbers
            allow_tf32 (bool): If tensor core is allowed
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Matrix tiling parameters for performance tunning
            VEC: Vectorization units


        Returns:
            input_grad (Tensor): A MxK tensor
    '''
    # Allocates output
    input_grad = torch.empty(
        (M, K), device=output_grad.device, dtype=output_grad.dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M'])
            * triton.cdiv(K, META['BLOCK_SIZE_K']),
        )

    ssl_backward_input_grad_kernel[grid](
        output_grad, weight, input_grad,
        M, N, K, K // redn_factor,
        output_grad.stride(0), output_grad.stride(1),
        weight.stride(0), weight.stride(1),
        input_grad.stride(0), input_grad.stride(1),
        R3=R3, R2=R2, R1=R1, R0=R0,
        FIXED_BLOCK_SIZE_N=BLOCK_SIZE_N,
        FIXED_BLOCK_SIZE_M=BLOCK_SIZE_M,
        FIXED_BLOCK_SIZE_K=BLOCK_SIZE_K,
        allow_tf32=allow_tf32,
        VEC=VEC,
        redn_factor=redn_factor
    )
    return input_grad


@triton.autotune(
    configs=_generate_configs(),
    # VEC_INT is not used but has to be a key to trigger recompilation
    # Every constant that is related to the computation correctness has to be a key
    key=['M', 'N', 'K_RED'],
    prune_configs_by={
        'early_config_prune': functools.partial(_early_config_prune, is_weight=False)
    },
    warmup=1,
    rep=10,
)
@triton.heuristics({
    "EVEN_N": lambda META: META['N'] % META['BLOCK_SIZE_N'] == 0,
    "EVEN_M": lambda META: META['M'] % META['BLOCK_SIZE_M'] == 0,
})
@triton.jit
def ssl_backward_input_grad_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K, K_RED,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension.
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    # Random numbers
    R3: tl.constexpr, R2: tl.constexpr, R1: tl.constexpr, R0: tl.constexpr,
    # Meta-parameters
    FIXED_BLOCK_SIZE_M: tl.constexpr, FIXED_BLOCK_SIZE_N: tl.constexpr, FIXED_BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # Reduction factor
    redn_factor: tl.constexpr,
    # Vectorization units
    VEC: tl.constexpr,
    # Tensor core
    allow_tf32: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """Kernel for computing the matmul C = (A x B'^T) or (A x B)
    A has shape (M, N), B has shape (N, (K // redn_factor)) and C has shape (M, K)
    """
    pid = tl.program_id(axis=0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k
    pid_k_red = pid_k // redn_factor
    pid_r = pid_k % redn_factor

    # [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_an = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_an[None, :] * stride_an

    # [BLOCK_SIZE_N, BLOCK_SIZE_K]
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    b_ptrs = b_ptr + pid_k_red * BLOCK_SIZE_K + offs_bn[:, None] * stride_bn

    IDX1 = R0 + R0 * pid_r
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # Recompute IDX is better than use it as a loop carried variable
        IDX = R3 + R2 * n + R1 + R1 * pid_k_red
        offs_bk = ((offs_k + (BLOCK_SIZE_K - (((IDX + IDX1) * VEC)) % BLOCK_SIZE_K))) % BLOCK_SIZE_K
        offs_bk = tl.max_constancy(offs_bk, VEC)

        if EVEN_N and EVEN_M:
            a = tl.load(a_ptrs)
        else:
            offs_an = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            a_mask = (offs_am[:, None] < M) & (offs_an[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b = tl.load(b_ptrs + offs_bk[None, :] * stride_bk)
        # We accumulate along the N dimension
        accumulator += tl.dot(a, b, allow_tf32=allow_tf32)
        # Advance the ptrs to the next N block
        a_ptrs += BLOCK_SIZE_N * stride_an
        b_ptrs += BLOCK_SIZE_N * stride_bn

    c = accumulator.to(tl.float16)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C
    # [BLOCK_SIZE_M, BLOCK_SIZE_K]
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_ck * offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptrs, c, mask=c_mask)


# unit tests

def unit_test_1():
    print("test 1 backward:")
    torch.manual_seed(0)
    S = 32
    a = torch.randn((S, S), device='cuda', dtype=torch.float16)
    b = torch.arange((S * S / 2), device='cuda',
                     dtype=torch.float16).reshape(int(S / 2), S).T
    grad = torch.randn((S, S), device='cuda', dtype=torch.float16)
    b_full = 0 #SSLForward.matmul(torch.eye(S, device='cuda', dtype=torch.float16), b, 1)
    triton_input_grad, triton_weight_grad = matmul(torch.eye(S, device='cuda', dtype=torch.float16), b, torch.eye(S, device='cuda', dtype=torch.float16), 1)
    torch_input_grad, torch_weight_grad = torch_matmul(torch.eye(S, device='cuda', dtype=torch.float16), b_full, torch.eye(S, device='cuda', dtype=torch.float16))
    print(triton_input_grad.shape, triton_weight_grad.shape, torch_input_grad.shape, torch_weight_grad.shape)
    import matplotlib.pyplot as plt
    import numpy as np
    torch.save(triton_input_grad, "triton_input_grad.pt")
    torch.save(triton_weight_grad, "triton_weight_grad.pt")
    # torch.save(torch_output, "torch_output.pt")

    plt.imshow(np.array(b.T.cpu()))
    plt.show()
    plt.imshow(np.array(b_full.cpu()))
    plt.show()
    print(f"triton_input_grad={triton_input_grad}")
    print(f"torch_input_grad={torch_input_grad}")
    if torch.allclose(triton_input_grad, torch_input_grad, atol=1e-2, rtol=0):
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
        line_vals=['cublas', 'triton0', 'triton1', 'triton2', 'triton4'],
        # Label name for the lines
        line_names=['cuBLAS', 'roast-comp,c=1', 'roast-comp,c=2',
                    'roast-comp,c=4', 'roast-comp,c=16'],
        # Line styles
        styles=[('green', '-'), ('blue', '-'), ('black', '-'),
                ('red', '-'), ('yellow', '-')],
        ylabel="ms",  # Label name for the y-axis
        # Name for the plot, used also as a file name for saving the plot.
        plot_name="matmul-performance",
        args={},
    ))
def benchmark(M, N, K, provider):

    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)
    c = torch.randn((M, N), device='cuda', dtype=torch.float16)
    if provider == 'cublas':
        ms = triton.testing.do_bench_cudagraph(lambda: torch_matmul(a, b, c))
    if provider == 'triton0':
        ms = triton.testing.do_bench_cudagraph(lambda: matmul(a, b, c, 0))
    if provider == 'triton1':
        b = torch.randn((N, int(K / 2)), device='cuda',
                        dtype=torch.float16).cuda()
        ms = triton.testing.do_bench_cudagraph(lambda: matmul(a, b, c, 1))
    if provider == 'triton2':
        b = torch.randn((N, int(K / 4)), device='cuda',
                        dtype=torch.float16).cuda()
        ms = triton.testing.do_bench_cudagraph(lambda: matmul(a, b, c, 2))
    if provider == 'triton4':
        b = torch.randn((N, int(K / 16)), device='cuda',
                        dtype=torch.float16).cuda()
        ms = triton.testing.do_bench_cudagraph(lambda: matmul(a, b, c, 4))
    return ms


if __name__ == '__main__':
    benchmark.run(show_plots=True, print_data=True, save_path=".")
