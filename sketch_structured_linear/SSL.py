import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .SSLFunction import ssl_linear
from .Mapper_v2 import *

from .block_sizes import BLOCK_SIZE_K as BLOCK_K_SIZE_MIN

#BLOCK_K_SIZE_MIN = 32

class SSL(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 redn_factor: int = 2, seed: int = 1024,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        assert redn_factor > 0 and (redn_factor & (redn_factor - 1)) == 0 # reduction factor should be power of 2
        self.redn_factor = redn_factor
        
        self.out_features = out_features  
        self.in_features = in_features    
       
        self.red_in_features = (in_features  // redn_factor + BLOCK_K_SIZE_MIN - 1) // BLOCK_K_SIZE_MIN * BLOCK_K_SIZE_MIN 

        self.seed = seed

        self.weight = nn.Parameter(torch.zeros((out_features, self.red_in_features), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.hasher = HasherFactory.get("uhash", self.seed)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). 
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            

    def forward(self, x):
        original_shape = x.shape
        
        x = self.preprocess(x, original_shape)
        x = ssl_linear(x, self.weight, self.bias, self.hasher.random_numbers, self.redn_factor)
        
        ## fused
        if self.bias is not None:
            x = x + self.bias
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
    
    
    @property
    def saving(self):
        return (self.weight.numel() / (self.in_features * self.out_features))

    def __repr__(self):        
        return "SketchStructuredLinear(in={}, out={}, compression={}, seed={}, saving={})".format(self.in_features, self.out_features, self.redn_factor, self.seed, self.saving)
    
    
    