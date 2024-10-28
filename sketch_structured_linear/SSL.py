import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .SSLFunction import ssl_linear
from .SSL_Kernel.SSLForward import ssl_forward_kernel_pretune, default_vec
from .Hasher import *

from .block_sizes import BLOCK_SIZE_K as BLOCK_K_SIZE_MIN

BLOCK_K_SIZE_MIN = 32

class SSL(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 redn_factor: int = 2, seed: int = 1024,
                 device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.BLOCK_SIZE_M, self.BLOCK_SIZE_K, self.BLOCK_SIZE_N  = (kwargs.get('BLOCK_SIZE_M', BLOCK_K_SIZE_MIN), kwargs.get('BLOCK_SIZE_K', BLOCK_K_SIZE_MIN), kwargs.get('BLOCK_SIZE_N', BLOCK_K_SIZE_MIN))

        assert redn_factor > 0 and (redn_factor & (redn_factor - 1)) == 0 # reduction factor should be power of 2
        self.redn_factor = redn_factor
        
        self.out_features = out_features  
        self.in_features = in_features    
       
        self.red_in_features = (in_features  // redn_factor + self.BLOCK_SIZE_K - 1) // self.BLOCK_SIZE_K * self.BLOCK_SIZE_K 

        self.seed = seed + kwargs.get('layer_idx', 0)

        self.weight = nn.Parameter(torch.zeros((out_features, self.red_in_features), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.random_numbers = HasherFactory.get("uhash", self.seed).random_numbers.to('cpu')

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). 
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #if self.bias is not None:
        #    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #    init.uniform_(self.bias, -bound, bound)
            

    def forward(self, x):
        original_shape = x.shape
        
        x = self.preprocess(x, original_shape)
        x = ssl_linear(x, self.weight, self.bias, self.random_numbers, self.redn_factor, self.BLOCK_SIZE_M, self.BLOCK_SIZE_N, self.BLOCK_SIZE_K)
        x = self.postprocess(x, original_shape)
        
        return x
    
    def preprocess(self, x, shape):
        dim_gt_2 = len(shape) > 2 
        if (dim_gt_2):
            x = x.reshape(-1, shape[-1]).contiguous()
        return x
    
    def postprocess(self, x, shape):
        dim_gt_2 = len(shape) > 2
        if (dim_gt_2):
            x = x.view(*shape[:-1], x.shape[-1]).contiguous()
        return x

    #def get_idx()

    def autotune(self):

        device = torch.device('cuda') if torch.cuda.is_available() else (_ for _ in ()).throw(RuntimeError("CUDA is not available. Please run on a CUDA-capable GPU."))
        
        self.best_configs = {}
        
        batch_size = kwargs.get('batch_size', 32)
        sample_input = torch.randn(batch_size, self.in_features, device=device)
        output = torch.empty(batch_size, self.out_features, device=device)

        block_m, block_k, block_n = (torch.zeros((1,), device=device), 
                                    torch.zeros((1,), device=device), 
                                    torch.zeros((1,), device=device), 
                                    )

        M, K = sample_input.shape
        N = self.out_features

        assert(self.random_numbers.numel() == 4)
        R3, R2, R1, R0 = self.random_numbers[3].item(), self.random_numbers[2].item(), self.random_numbers[1].item(), self.random_numbers[0].item()

        def grid(META):
            return (
                triton.cdiv(M, META['BLOCK_SIZE_M'])
                * triton.cdiv(N, META['BLOCK_SIZE_N']),
            )

        ssl_forward_kernel_pretune[grid](
            sample_input, self.weight, self.bias, output,
            block_m, block_k, block_n 
            M, N, K, K // self.redn_factor,
            sample_input.stride(0), sample_input.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1),
            allow_tf32=False,
            R3=R3, R2=R2, R1=R1, R0=R0,
            GROUP_SIZE_M=1,
            VEC=default_vec,
            redn_factor=self.redn_factor
        )

        self.BLOCK_SIZE_M = block_m
        self.BLOCK_SIZE_K = block_k
        self.BLOCK_SIZE_N = block_n


    
    
    @property
    def saving(self):
        return (self.weight.numel() / (self.in_features * self.out_features))

    def __repr__(self):        
        return "SketchStructuredLinear(in={}, out={}, compression={}, seed={}, saving={})".format(self.in_features, self.out_features, self.redn_factor, self.seed, self.saving)
    
    
    
