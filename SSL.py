import torch
import torch.nn as nn
from math import sqrt

try:
    from .SketchStructuredLinear import SketchStructuredLinear
    from .SSLFunction import SketchStructuredLinearFunction
    from .Mapper_v2 import *
except:
    from SketchStructuredLinear import SketchStructuredLinear
    from SSLFunction import SketchStructuredLinearFunction
    from Mapper_v2 import *


class SSL(SketchStructuredLinear):
    def __init__(self, *args, bias, redn_factor=1, seed=1024):
        super().__init__(*args)
        self.redn_factor = redn_factor
        self.seed = seed

        assert redn_factor > 0 and (redn_factor & (redn_factor - 1)) == 0 # reduction factor should be power of 2
        self.redn_factor = redn_factor
        
        self.weight = nn.Parameter(torch.zeros((out_features, int(in_features // redn_factor)), dtype=torch.float), requires_grad=True)
        
        self.init_scale = 1/sqrt(in_features)
        nn.init.uniform_(self.weight.data, a=-
                            self.init_scale, b=self.init_scale)

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                self.odim, dtype=torch.float), requires_grad=True)
            
        self.hasher = HasherFactory.get(self.seed)

    def forward_matmul(self, x):
        x = SketchStructuredLinearFunction(self.preprocess(x, x.shape), self.weight, self.hasher.random_numbers, self.redn_factor)
        if self.bias is not None:
            x = x + self.bias
        return self.postprocess(x, x.shape)
    
    def preprocess(self, x, shape):
        dim_gt_2 = len(shape) > 2 
        if (dim_gt_2):
            x = x.reshape(-1, shape[-1]).contiguous()
        return x
    
    def postprocess(self, x, shape):
        dim_gt_2 = len(shape) > 2
        if (dim_gt_2):
            x = x.view(*shape[:-1], x.shape[-1])
        return x

    @property
    def saving(self):
        return (self.WHelper.weight.numel() / (self.in_features * self.out_features))
    