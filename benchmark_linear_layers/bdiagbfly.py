import torch
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Linear, init

from einops import rearrange

# generated from authors code 
# https://github.com/HazyResearch/fly/blob/master/src/models/layers/




def low_rank_project(M, rank):
    """Supports batches of matrices as well.
    """

    U, S, Vt = torch.linalg.svd(M)
    S_sqrt = S[..., :rank].sqrt()
    U = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
    return U, Vt


def blockdiag_butterfly_multiply_einsum(x, w1_bfly, w2_bfly, b2):
    """
    Arguments:
        x: (batch, n)
        w1_bfly: (k, (j * b1), i), where k = n / i
        w2_bfly: (j, (l * b2), (k b1))
    Outputs:
        out: (batch, m), where m = l * j * b2
    """
    batch, n = x.shape
    k, jb1, i = w1_bfly.shape
    j, lb2, kb1 = w2_bfly.shape
    b1 = jb1 // j
    assert jb1 == j * b1
    assert kb1 == k * b1
    assert k * i == n

    x_reshaped = rearrange(x, 'b (k i) -> b k i', k=k)
    w1_bfly = rearrange(w1_bfly, 'k (j b1) i -> k j b1 i', b1=b1)
    w2_bfly = rearrange(w2_bfly, 'j (l b2) (k b1) -> j l b2 k b1', b1=b1, b2=b2)
    # torch.einsum doesn't support indices named b1 or b2, so we map b1 -> y, b2 -> z
    out = torch.einsum('b k i, k j y i, j l z k y -> b l j z', x_reshaped, w1_bfly, w2_bfly)
    return rearrange(out, 'b l j b2 -> b (l j b2)')


def blockdiag_butterfly_project_einsum(M, nblocks1, nblocks2, b1, b2):
    """
    Arguments:
        M: (m, n)
    Outputs:
        w1_bfly: (nblocks1, nblocks2, i)
        w2_bfly: (nblocks2, l, nblocks1)
    """
    m, n = M.shape
    k, j = nblocks1, nblocks2
    M_permuted_batched = rearrange(M, '(l j b2) (k i) -> k j (l b2) i', k=nblocks1, j=nblocks2,
                                   b2=b2)
    U, Vt = low_rank_project(M_permuted_batched, rank=b1)
    w1_bfly = rearrange(Vt, 'k j b1 i -> k (j b1) i')
    w2_bfly = rearrange(U, 'k j lb2 b1 -> j lb2 (k b1)')
    return w1_bfly, w2_bfly


class BlockdiagButterflyLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, nblocks1: int = 4, nblocks2: int = 4,
                 b1: int = 48, b2: int = 1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks1 = nblocks1
        self.nblocks2 = nblocks2

        m, n = out_features, in_features
        i = n//nblocks1
        l = m//nblocks2
        assert n == i * nblocks1
        assert m == l * nblocks2
        self.w1_bfly = Parameter(torch.empty((nblocks1, nblocks2*b1, i), **factory_kwargs))
        self.w2_bfly = Parameter(torch.empty((nblocks2, l, nblocks1*b1), **factory_kwargs))
        self.b1 = b1
        self.b2 = b2
        self.saving = ((torch.numel(self.w1_bfly)+torch.numel(self.w2_bfly)))/(self.in_features*self.out_features)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def init_factors(self, weight):
        self.w1_bfly.data, self.w2_bfly.data = blockdiag_butterfly_project_einsum(weight, nblocks1=self.nblocks1,
                                                nblocks2=self.nblocks2, b1=self.b1, b2=self.b2)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.w1_bfly, a=math.sqrt(5))
        init.kaiming_uniform_(self.w2_bfly, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w1_bfly)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def preprocess(self, x):
        return x.reshape(-1, x.shape[-1])

    def postprocess(self, output, x_shape):
        batch_shape = x_shape[:-1]
        return output.reshape(batch_shape + (output.shape[-1],))

    def forward(self, input: Tensor) -> Tensor:
        x_shape = input.shape
        output = blockdiag_butterfly_multiply_einsum(self.preprocess(input), self.w1_bfly, self.w2_bfly, self.b2)
        output = self.postprocess(output, x_shape)
        return (output + self.bias) if self.bias is not None else output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    def get_from_linear(weight, in_dim, out_dim, bias, seed, red_fac):
        mod  = BlockdiagButterflyLinear(in_dim, 
                             out_dim,
                             bias=bias is not None,
                             nblocks1=int(red_fac*2), nblocks2=int(red_fac*2)) # *2 since we want to keep the parameters same
        mod.init_factors(weight.T)
        return mod



def ConvertBlockdiagButterfly(model, size_limit, red_fac, init_seed):
    _diag("head", model, size_limit=size_limit, red_fac=red_fac, init_seed=init_seed)
    return model

def _diag(name, pytorch_model, size_limit, red_fac, init_seed):
    seed = init_seed * 1024
    for attr in dir(pytorch_model):
        target_attr = getattr(pytorch_model, attr)
        #print(name, "->", attr, "type:", type(target_attr))
        if type(target_attr) in[torch.nn.Linear, torch.nn.modules.Linear]:
            seed = seed + 1

            if 'do_not_roast' in dir(target_attr) and target_attr.do_not_roast:
                print("ignored since set")
                continue
            if target_attr.in_features < size_limit or target_attr.out_features < size_limit:
                print("ignored due to scale", target_attr)
                continue
            new_attr =  BlockdiagButterflyLinear.get_from_linear(target_attr.weight, target_attr.in_features, 
                                       target_attr.out_features,
                                       target_attr.bias,
                                       seed,
                                       red_fac)
            print("replaced", target_attr)
            setattr(pytorch_model, attr, new_attr)
        
    for name, immediate_child_module in  pytorch_model.named_children():
        target_attr = immediate_child_module

        if type(immediate_child_module) in [torch.nn.modules.Linear , torch.nn.modules.linear.Linear]:
            seed = seed + 1

            if 'do_not_roast' in dir(target_attr) and target_attr.do_not_roast:
                print("ignored since set")
            elif target_attr.in_features < size_limit or target_attr.out_features < size_limit:
                print("ignored due to scale", target_attr)
            else:

                new_attr =  BlockdiagButterflyLinear.get_from_linear(target_attr.weight, target_attr.in_features, 
                                       target_attr.out_features,
                                       target_attr.bias,
                                       seed,
                                       red_fac)
                print("replaced", target_attr)
                setattr(pytorch_model, name, new_attr)
        init_seed = init_seed + 1
        _diag(name, immediate_child_module, size_limit, red_fac, init_seed)
