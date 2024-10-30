
import torch
from torch.nn.parameter import Parameter

try:
    from .SSL import SSL, BLOCK_K_SIZE_MIN  
    from . import SSLFunction as SSF
except:
    from SSL import SSL, BLOCK_K_SIZE_MIN 
    import SSLFunction as SSF

def test_sslinear():
    import numpy as np

    # quick test
    M = 32768
    K = 1024
    N = 1024
    input = torch.randn((M,K)).cuda().half()
    torch.manual_seed(12)
    orig = torch.nn.Linear(K,512).cuda().half()
    torch.manual_seed(12)
    mod = SSL(K,512, redn_factor=1, batch_size=M).cuda().half()
    print("weight close=", torch.allclose(mod.weight, orig.weight, atol=1e-1, rtol=1e-5))
    print("bias close=", torch.allclose(mod.bias, orig.bias, atol=1e-1, rtol=1e-5))

    x = orig(input)
    y = mod(input)
    print(x)
    print(y)
    print("fwd close=", torch.allclose(x, y, atol=1e-1, rtol=1e-5))

    input = torch.randn((M,K)).cuda().half()
    orig = torch.nn.Linear(K,N).cuda().half()
    # quick time test
    times = []
    for i in range(12):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = orig(input)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    print("orig", times)
    print("orig", np.mean(times[3:]), "+-", np.std(times[3:]))

    for fac in [1,2,4,8]:
        mod = SSL(K,N,redn_factor=fac,batch_size=M).cuda().half()
        print("before autotune: ", mod.BLOCK_SIZE_K, mod.BLOCK_SIZE_N, mod.BLOCK_SIZE_M)
        mod.autotune()
        print("after autotune: ", mod.BLOCK_SIZE_K, mod.BLOCK_SIZE_N, mod.BLOCK_SIZE_M)
        times = []
        for i in range(12):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output = mod(input)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        print(fac, times)
        print(fac, np.mean(times[3:]), "+-", np.std(times[3:]))


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

def test_bwd():

    import numpy as np

    standard = [(128, 512, 512)]
    not_even_k = []

    x = [(64,64,64), (128,512,512)]
    for M,K,N in standard + not_even_k:
        for F in [2, 4,  8]:
            mod = SSL(K, N, redn_factor=F, bias=False, batch_size=M).cuda().half()
        
            input1 = torch.nn.Parameter(torch.randn((M,K)).cuda().half(), requires_grad=True)
            input1.retain_grad()
            
            input2 = torch.nn.Parameter(input1.data.clone(), requires_grad=True)
            input2.retain_grad()
        
            # triton 
            triton_output = mod(input1)
            triton_loss = torch.sum(triton_output)
            triton_loss.backward()
            triton_input_grad = input1.grad.clone()
            triton_weight_grad = mod.weight.grad.clone()
        
            triton_pack = [triton_loss, triton_input_grad, triton_weight_grad]
            print(mod.BLOCK_SIZE_M)

            if mod.redn_factor > 1:       
                random_numbers = mod.random_numbers
                R3, R2, R1, R0 = random_numbers[3].item(), random_numbers[2].item(
                        ), random_numbers[1].item(), random_numbers[0].item()
                idx = SSL.get_idx(M, K, N, block_m=mod.BLOCK_SIZE_M, block_k=mod.BLOCK_SIZE_K, block_n=mod.BLOCK_SIZE_N, R3=R3, R2=R2, R1=R1, R0=R0, reduction_factor=mod.redn_factor, device=mod.weight.device)
                full_weight = mod.weight.view(-1)[idx].data.clone()
                weight = torch.nn.Parameter(full_weight, requires_grad=True)
            else: 
                full_weight = mod.weight.data.clone().T
                weight = torch.nn.Parameter(full_weight, requires_grad=True)


            torch_output = torch.matmul(input2, weight)
            torch_output.retain_grad()
            torch_loss = torch.sum(torch_output)
            torch_loss.backward()
            torch_input_grad = input2.grad.clone()
            torch_weight_grad = weight.grad.clone()

            if mod.redn_factor > 1:
                weight_grad = torch.zeros(mod.weight.numel(), device=input2.device, dtype=input2.dtype)
                weight_grad.scatter_add_(0, idx.view(-1), torch_weight_grad.reshape(-1))
                c_torch_weight_grad = weight_grad.reshape(*mod.weight.shape)
            else:
                c_torch_weight_grad = torch_weight_grad.T
        
            torch_pack = [torch_loss, torch_input_grad, c_torch_weight_grad]
            print(torch_loss, triton_loss)
            print(M, K, N, F,
                    torch.allclose(triton_pack[0], torch_pack[0], atol=1e-2, rtol=1e-5),
                    torch.allclose(triton_pack[1], torch_pack[1], atol=1e-2, rtol=1e-5),
                    torch.allclose(triton_pack[2], torch_pack[2], atol=1e-2, rtol=1e-5),
                )

if __name__ == '__main__':
    M = 1024
    K = 1024
    N = 1024
    F = 2
    mod = SSL(K, N, redn_factor=F, bias=False, batch_size=M).cuda().half()
    input1 = torch.nn.Parameter(torch.randn((M,K)).cuda().half(), requires_grad=True)
    SSF.controls['triton_allow_autotune'] = False
    #test_sslinear()
    test_bwd()
