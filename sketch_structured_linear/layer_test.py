
import torch
from torch.nn.parameter import Parameter

try:
    from .SSL import SSL 
    from . import SSLFunction as SSF
except:
    from SSL import SSL 
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

if __name__ == '__main__':
    M = 1024
    K = 1024
    N = 1024
    F = 2
    mod = SSL(K, N, redn_factor=F, bias=False, batch_size=M).cuda().half()
    input1 = torch.nn.Parameter(torch.randn((M,K)).cuda().half(), requires_grad=True)
    SSF.controls['triton_allow_autotune'] = False
    test_sslinear()
