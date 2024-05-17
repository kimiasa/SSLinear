import torch
import SSLForward as fwd
import SSLBackward as bwd


A = 11211
B = 11311
C = 11411
D = 11511

device_index = 0
seed = 42

# TOOD: fix torch_get_full_weight_idx iotb compatible with get_hashed_idx


def get_hashed_idx(redn_factor: int,
                   K: int, N: int,
                   R3: int, R2: int, R1: int, R0: int,
                   BLOCK_SIZE_N: int, BLOCK_SIZE_K: int, VEC: int
                   ):
    stride_bn = K // redn_factor
    stride_bk = 1
    idx = torch.zeros((N, K), dtype=torch.float16).long()
    for pid_n in range((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N):
        for k in range(0, (K + BLOCK_SIZE_K * redn_factor - 1) // (BLOCK_SIZE_K * redn_factor)):
            for ck in range(0, redn_factor):
                block = (k * BLOCK_SIZE_K + (torch.arange(BLOCK_SIZE_K).long() +
                                            (BLOCK_SIZE_K - (((R2 * pid_n + R1 * (k + 1) + R0 * (ck + 1) + R3) * VEC) % BLOCK_SIZE_K))) % BLOCK_SIZE_K)[None, :] * stride_bk + \
                                            (pid_n * BLOCK_SIZE_N + torch.arange(BLOCK_SIZE_N)[:, None]) * stride_bn
                off = k * BLOCK_SIZE_K * redn_factor + ck * BLOCK_SIZE_K
                idx[pid_n * BLOCK_SIZE_N:(pid_n + 1) * BLOCK_SIZE_N, off:off + BLOCK_SIZE_K] = block
    return idx


def test_forward():
    def torch_get_full_weight_idx(hashed_weight: torch.tensor) -> torch.tensor:
        weight = torch.empty(
            (N, K), device=hashed_weight.device, dtype=hashed_weight.dtype)
        for k in range(K // (BLOCK_SIZE_K * redn_factor)):
            for n in range(N // BLOCK_SIZE_N):
                for ck in range(0, redn_factor):
                    hashed_idx = (torch.arange(0, BLOCK_SIZE_K) + (BLOCK_SIZE_K - (((R2 * n + R1 * (k + 1) + R0 * (ck + 1) + R3) * VEC) % BLOCK_SIZE_K))) % BLOCK_SIZE_K
                    offset = k * BLOCK_SIZE_K
                    weight_k_slice = slice((k * redn_factor + ck) * BLOCK_SIZE_K, (k * redn_factor + ck + 1) * BLOCK_SIZE_K)
                    weight_n_slice = slice(n * BLOCK_SIZE_N, (n + 1) * BLOCK_SIZE_N)
                    weight[weight_n_slice, weight_k_slice] = hashed_weight[weight_n_slice, offset + hashed_idx]
        return weight

    M = 64
    K = 64
    N = 64
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 64
    VEC = 2

    redn_factor = 1

    input = torch.eye(M, device='cuda', dtype=torch.float16)
    hashed_weight = torch.arange((N * int(K // redn_factor)), device='cuda', dtype=torch.float16).reshape(N, int(K // redn_factor))

    R3, R2, R1, R0 = A, B, C, D

    torch_output = torch.mm(input, torch_get_full_weight_idx(hashed_weight).T)

    # Disable tf32 in testing
    tl_output = fwd.ssl_forward_tl(input, hashed_weight.T, input.shape[0], input.shape[1], hashed_weight.shape[0], redn_factor, R3, R2, R1, R0,
                                   BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, VEC=VEC,
                                   allow_autotune=False)

    assert (torch.allclose(tl_output, torch_output, rtol=1e-3) is True)


def test_backward_weight():
    def torch_get_weight(input: torch.tensor, output: torch.tensor) -> torch.tensor:
        hashed_weight = torch.zeros((N, int(K // redn_factor)), device=output.device, dtype=output.dtype)
        weight = torch.mm(output.permute(1, 0), input)
        for k in range(K // BLOCK_SIZE_K):
            for n in range(N // BLOCK_SIZE_N):
                IDX = R3 + (R2 * n) + R1 + (R1 * (k // redn_factor))
                IDX1 = R0 + (R0 * (k % redn_factor))

                hashed_idx = (torch.arange(0, BLOCK_SIZE_K) + (BLOCK_SIZE_K - (((IDX + IDX1) * VEC) % BLOCK_SIZE_K))) % BLOCK_SIZE_K
                offset = (k // redn_factor) * BLOCK_SIZE_K

                weight_k_slice = slice(k * BLOCK_SIZE_K, (k + 1) * BLOCK_SIZE_K)
                weight_n_slice = slice(n * BLOCK_SIZE_N, (n + 1) * BLOCK_SIZE_N)
                hashed_weight[weight_n_slice, offset + hashed_idx] += weight[weight_n_slice, weight_k_slice]
        return hashed_weight

    bwd.allow_backward_autotune = False

    M = 64
    K = 64
    N = 64
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 64
    VEC = 2

    redn_factor = 2

    input = torch.rand((M, K), device='cuda', dtype=torch.float16)
    output = torch.rand((M, N), device='cuda', dtype=torch.float16)

    R3, R2, R1, R0 = A, B, C, D

    torch.backends.cuda.matmul.allow_tf32 = False
    torch_weight = torch_get_weight(input, output)

    # Disable tf32 in testing
    tl_weight = bwd.ssl_backward_weight_grad_tl(
        input, output, M, K, N, redn_factor, R3, R2, R1, R0, allow_tf32=False,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N, VEC=VEC)

    assert (torch.allclose(tl_weight, torch_weight, rtol=1e-3) is True)

    # for full tensors testcase
    # assert torch.unique(torch_weight) == torch.unique(tl_weight)


def test_backward_input():
    def torch_get_full_weight_idx(hashed_weight: torch.tensor) -> torch.tensor:
        weight = torch.empty(
            (N, K), device=hashed_weight.device, dtype=hashed_weight.dtype)
        for k in range(K // (BLOCK_SIZE_K * redn_factor)):
            for n in range(N // BLOCK_SIZE_N):
                for ck in range(0, redn_factor):
                    hashed_idx = (torch.arange(0, BLOCK_SIZE_K) + (BLOCK_SIZE_K - (((R2 * n + R1 * (k + 1) + R0 * (ck + 1) + R3) * VEC) % BLOCK_SIZE_K))) % BLOCK_SIZE_K
                    offset = k * BLOCK_SIZE_K
                    weight_k_slice = slice((k * redn_factor + ck) * BLOCK_SIZE_K, (k * redn_factor + ck + 1) * BLOCK_SIZE_K)
                    weight_n_slice = slice(n * BLOCK_SIZE_N, (n + 1) * BLOCK_SIZE_N)
                    weight[weight_n_slice, weight_k_slice] = hashed_weight[weight_n_slice, offset + hashed_idx]
        return weight

    bwd.allow_backward_autotune = False

    M = 64
    K = 32
    N = 64
    BLOCK_SIZE_K = 16
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 64
    VEC = 2

    redn_factor = 2

    output = torch.eye(M, device='cuda', dtype=torch.float16)
    hashed_weight = torch.arange((N * int(K // redn_factor)), device='cuda', dtype=torch.float16).reshape(int(K // redn_factor), (N)).T

    # input = torch.rand((M, K), device='cuda', dtype=torch.float16)
    # hashed_weight = torch.rand((N, int(K//redn_factor)), device='cuda', dtype=torch.float16)

    R3, R2, R1, R0 = A, B, C, D

    torch.backends.cuda.matmul.allow_tf32 = False

    # [M, N] x [K, N]^T
    torch_input = torch.mm(output, torch_get_full_weight_idx(hashed_weight))

    # Disable tf32 in testing
    tl_input = bwd.ssl_backward_input_grad_tl(
        output, hashed_weight, M, K, N, redn_factor, R3, R2, R1, R0, allow_tf32=False,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, VEC=VEC)

    assert (torch.allclose(tl_input, torch_input, rtol=1e-3) is True)


def test_backward_autograd():
    def torch_get_full_weight_idx(hashed_weight: torch.tensor) -> torch.tensor:
        weight = torch.empty(
            (N, K), device=hashed_weight.device, dtype=hashed_weight.dtype)
        for k in range(K // (BLOCK_SIZE_K * redn_factor)):
            for n in range(N // BLOCK_SIZE_N):
                for ck in range(0, redn_factor):
                    hashed_idx = (torch.arange(0, BLOCK_SIZE_K) + (BLOCK_SIZE_K - (((R2 * n + R1 * (k + 1) + R0 * (ck + 1) + R3) * VEC) % BLOCK_SIZE_K))) % BLOCK_SIZE_K
                    offset = k * BLOCK_SIZE_K
                    weight_k_slice = slice((k * redn_factor + ck) * BLOCK_SIZE_K, (k * redn_factor + ck + 1) * BLOCK_SIZE_K)
                    weight_n_slice = slice(n * BLOCK_SIZE_N, (n + 1) * BLOCK_SIZE_N)
                    weight[weight_n_slice, weight_k_slice] = hashed_weight[weight_n_slice, offset + hashed_idx]
        return weight

    bwd.allow_backward_autotune = False

    M = 64
    K = 32
    N = 64
    BLOCK_SIZE_K = 16
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 64
    VEC = 2

    redn_factor = 2

    # input = torch.eye(M, device='cuda', dtype=torch.float16)
    # hashed_weight = torch.arange((N * int(K//redn_factor)), device='cuda', dtype=torch.float16).reshape(int(K//redn_factor), (N)).T

    input = torch.rand((M, K), device='cuda', dtype=torch.float16)
    hashed_weight = torch.rand((N, int(K // redn_factor)), device='cuda', dtype=torch.float16)

    R3, R2, R1, R0 = A, B, C, D

    torch.backends.cuda.matmul.allow_tf32 = False

    idx = get_hashed_idx(redn_factor=redn_factor,
                         K=K, N=N,
                         R3=R3, R2=R2, R1=R1, R0=R0,
                         BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, VEC=VEC
                         )

    auto_grad_input = torch.nn.parameter.Parameter(data=input, requires_grad=True)
    auto_grad_weight = torch.nn.parameter.Parameter(data=hashed_weight, requires_grad=True)

    idx = idx.to(hashed_weight.device)
    torch_idx = torch.nn.parameter.Parameter(data=idx, requires_grad=False)

    out = torch.matmul(auto_grad_input, auto_grad_weight.flatten()[torch_idx].T)

    out.retain_grad()

    loss = out.sum()

    loss.backward()

    out_grad, torch_weight_grad, torch_input_grad = out.grad, auto_grad_weight.grad, auto_grad_input.grad

    tl_output = fwd.ssl_forward_tl(input, hashed_weight.T, M, K, N, redn_factor, R3, R2, R1, R0,
                                   BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, VEC=VEC,
                                   allow_autotune=False)

    torch.testing.assert_close(out, tl_output, rtol=1e-3, atol=1e-3)

    # Disable tf32 in testing
    tl_weight_grad = bwd.ssl_backward_weight_grad_tl(
        input, out_grad, M, K, N, redn_factor, R3, R2, R1, R0, allow_tf32=False,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N, VEC=VEC)

    tl_input_grad = bwd.ssl_backward_input_grad_tl(
        out_grad, hashed_weight, M, K, N, redn_factor, R3, R2, R1, R0, allow_tf32=False,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, VEC=VEC)

    count = torch.zeros(N * int(K // redn_factor),
                        dtype=torch.float16, device=torch_weight_grad.device)
    count.scatter_add_(0, torch_idx.T.reshape(-1), torch.ones_like(torch_idx.T,
                                                                   device=torch_weight_grad.device, dtype=torch.float16).reshape(-1))

    torch.testing.assert_close(torch_weight_grad, tl_weight_grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(torch_input_grad, tl_input_grad, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":

    torch.cuda.set_device(device_index)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    test_forward()
    test_backward_weight()
    test_backward_input()
    test_backward_autograd()
