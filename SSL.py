from math import sqrt

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import time

from .SSLFunction import SketchStructuredLinearFunction
from .SSLFunction import ssl_linear
from .Mapper_v2 import *


class SSL(nn.Module):
    def __init__(self, in_features, out_features, bias=None, redn_factor=1, seed=1024):
        super(SSL, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.redn_factor = redn_factor
        self.seed = seed

        assert redn_factor > 0 and (redn_factor & (redn_factor - 1)) == 0 # reduction factor should be power of 2
        self.redn_factor = redn_factor
        
        self.weight = nn.Parameter(torch.zeros((out_features, int(in_features // redn_factor)), dtype=torch.float), requires_grad=True)
        
        self.init_scale = 1/sqrt(in_features)
        nn.init.uniform_(self.weight.data, a=-
                            self.init_scale, b=self.init_scale)

        #self.bias = None
        #if bias is None:
        #    self.bias = nn.Parameter(torch.zeros(
        #        self.out_features, dtype=torch.float), requires_grad=True)
            
            
        self.hasher = HasherFactory.get("uhash", self.seed)
        self.hasher = HasherFactory.get("uhash", self.seed)

    def forward(self, x):
        original_shape = x.shape
        
        x = self.preprocess(x, original_shape)
        x = ssl_linear(x, self.weight, self.hasher.random_numbers, self.redn_factor)
        
        ## fused
        #if self.bias is not None:
        #    x = x + self.bias
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
            x = x.view(*shape[:-1], x.shape[-1]).contiguous()
        return x
    
    
    @property
    def saving(self):
        return (self.weight.numel() / (self.in_features * self.out_features))

    def __repr__(self):        
        return "SketchStructuredLinear(in={}, out={}, compression={}, seed={}, saving={})".format(self.in_features, self.out_features, self.redn_factor, self.seed, self.saving)
    
        return (self.weight.numel() / (self.in_features * self.out_features))

    def __repr__(self):        
        return "SketchStructuredLinear(in={}, out={}, compression={}, seed={}, saving={})".format(self.in_features, self.out_features, self.redn_factor, self.seed, self.saving)
    
    