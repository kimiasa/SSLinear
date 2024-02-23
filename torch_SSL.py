import torch
import torch.nn as nn
from math import sqrt

try:
    from .SketchStructuredLinear import SketchStructuredLinear
    from .Mapper_v2 import *
except:
    from SketchStructuredLinear import SketchStructuredLinear 
    from Mapper_v2 import *

    

class TH_SSLTranform(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 redn_factor=1,
                 seed=2023,
                 mapper_args=None):
        super(TH_SSLTranform, self).__init__()
        
        W_shape = (out_features, in_features)
        assert redn_factor > 0 and (redn_factor & (redn_factor - 1)) == 0 # reduction factor should be power of 2
        self.redn_factor = redn_factor
        self.seed = seed
        self.wsize = int(out_features * int(in_features // redn_factor))
        self.weight = nn.Parameter(torch.zeros(self.wsize, dtype=torch.float), requires_grad=True)
        
        self.init_scale = 1/sqrt(in_features)
        nn.init.uniform_(self.weight.data, a=-
                            self.init_scale, b=self.init_scale)
        self.in_features = in_features
        self.out_features = out_features

        gen = torch.Generator()
        gen.manual_seed(seed)

        assert(mapper_args is not None)
        mapper = MapperFactory.get(**mapper_args)
        idx = mapper.get_idx(w_shape=W_shape, target_size=self.wsize, **mapper_args)
        self.register_buffer('IDX', idx)

        if idx is None:
            raise NotImplementedError
        
            
        self.register_buffer('G', torch.randint(0, 2, size=W_shape, dtype=torch.float, generator=gen)*2 - 1)


    def forward(self):
        W = torch.mul(self.weight[self.IDX], self.G)
        return W        

    def grad_comp_to_orig(self, grad):  # grad of compressed to original
        return torch.mul(grad[self.IDX], self.G)

     
    def grad_orig_to_comp(self, grad): # original gradient to compressed gradient
        out_grad = torch.zeros(
            self.wsize, dtype=torch.float, device=grad.device)
        out_grad.scatter_add_(0, self.IDX.reshape(-1),
                              (torch.mul(grad, self.G)).reshape(-1))

        count = torch.zeros(self.wsize, dtype=torch.float, device=grad.device)
        count.scatter_add_(0, self.IDX.reshape(-1), torch.ones_like(self.IDX,
                           device=grad.device, dtype=torch.float).reshape(-1))
        return (out_grad, count)

    def wt_comp_to_orig(self, wt):
        return torch.mul(wt[self.IDX], self.G)

    def wt_orig_to_comp(self, wt):
        out_wt = torch.zeros(self.wsize, dtype=torch.float, device=wt.device)
        out_wt.scatter_add_(0, self.IDX.reshape(-1),
                            (torch.mul(wt, self.G)).reshape(-1))

        count = torch.zeros(self.wsize, dtype=torch.float, device=wt.device)
        count.scatter_add_(0, self.IDX.reshape(-1), torch.ones_like(self.IDX,
                           device=wt.device, dtype=torch.float).reshape(-1)) + 1e-3
        return (out_wt, count)


class TH_SSL(SketchStructuredLinear):
    def __init__(self, *args, bias, redn_factor=1, seed=1024, mapper_args=None):
        super().__init__(*args)
        self.redn_factor = redn_factor
        self.seed = seed

        if mapper_args is not None:
            mapper_args['mode'] = "mlp"
            mapper_args['seed'] = seed

        self.WHelper = TH_SSLTranform(
                        self.in_features, self.out_features, redn_factor,
                        seed=seed, mapper_args=mapper_args)

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                self.odim, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        W = self.WHelper() 
        x = nn.functional.linear(x, W, self.bias)
        return x

    @property
    def saving(self):
        return (self.WHelper.weight.numel() / (self.in_features * self.out_features))
    
    def __repr__(self):
        return "FakeRoastLinear(in={}, out={},, matrix_mode={}, seed={})".format(self.idim, self.odim, self.matrix_mode, self.seed)
