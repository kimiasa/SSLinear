import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import pdb
import mmh3
#try:
from .Mapper_v2 import *
#except:
#    from Mapper_v2 import *

class SketchStructuredLinearTranform(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 init_scale=None,
                 redn_factor=1,
                 matrix_mode="mapper",# use mapper for using mappers
                 seed=2023,
                 mapper_args=None):
        super(SketchStructuredLinearTranform, self).__init__()
        
        W_shape = (out_features, in_features)
        assert redn_factor > 0 and (redn_factor & (redn_factor - 1)) == 0 # factor should be power of 2
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

        if matrix_mode == "mapper":
            assert(mapper_args is not None)
            mapper = MapperFactory.get(**mapper_args)
            idx = mapper.get_idx(w_shape=W_shape, target_size=self.wsize, **mapper_args)
            self.IDX = nn.Parameter(idx, requires_grad=False)
        elif matrix_mode == "random":
            # making it consistent for power of 2 compression
            assert(N > self.wsize)
            self.IDX = nn.Parameter(torch.randint(0, N, size=W_shape, dtype=torch.int64, generator=gen) % self.wsize, requires_grad=False)
        else:
            raise NotImplementedError
        self.G = nn.Parameter(torch.randint(0, 2, size=W_shape, dtype=torch.float, generator=gen)*2 - 1, requires_grad=False)

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


class SketchStructuredLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, redn_factor=1, matrix_mode="mapper", seed=1024, mapper_args=None):
        super(SketchStructuredLinear, self).__init__()
        self.W_shape = (out_features, in_features)
        self.idim = in_features
        self.odim = out_features
        self.redn_factor = redn_factor
        self.matrix_mode = matrix_mode
        self.seed = seed

        init_scale = 1/sqrt(self.idim)

        if mapper_args is not None:
            mapper_args["mode"] = "mlp"
            mapper_args['seed'] = seed

        self.WHelper = SketchStructuredLinearTranform(
                        self.idim, self.odim, init_scale, redn_factor,
                        matrix_mode=matrix_mode,
                        seed=seed, mapper_args=mapper_args)

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                self.odim, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        W = self.WHelper() 
        x = nn.functional.linear(x, W, self.bias)
        return x

    def __repr__(self):
        return "FakeRoastLinear(in={}, out={},, matrix_mode={}, seed={})".format(self.idim, self.odim, self.matrix_mode, self.seed)
