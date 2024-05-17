import torch
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange

import numpy as np

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


def factors(n):
    return [(i, n // i) for i in range(1, math.floor(math.sqrt(n)) + 1) if n % i == 0]


def blockdiag_butterfly_project(M, sizes=None):
    """Only works for square matrices for now
    """
    m, n = M.shape
    if m != n:
        raise NotImplementedError('Only support square matrices')
    if sizes is None:
        # Find the factors that are closest to sqrt(n)
        sizes = factors(n)[-1]
        # Larger factor first is probably more efficient, idk
        sizes = (sizes[1], sizes[0])
    assert n == sizes[0] * sizes[1]
    M_permuted_batched = rearrange(M, '(p k) (r s) -> k r p s', k=sizes[1], r=sizes[0])
    U, Vt = low_rank_project(M_permuted_batched, rank=1)
    w1_bfly = rearrange(Vt, 'k r 1 s -> r k s')
    w2_bfly = rearrange(U, 'k r s 1 -> k s r')
    return w1_bfly, w2_bfly

class BlockdiagButterflyMultiply(torch.autograd.Function):

    """
    // taken from the original authors repo  

    This is a faster implementation, with careful memory copies for the fastest
    bmm performance.
    The backward pass is also written manually with careful memory copies.
    Arguments:
        x: (batch, n)
        w1_bfly: (k, q, p), where k = n / p
        w2_bfly: (l, s, r), where l = k * q / r = n * q / (p * r)
    Outputs:
        out: (batch, m), where m = l * s = n * s * q / (p * r)
    """

    @staticmethod
    #@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, w1_bfly, w2_bfly):
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        assert k * p == n
        assert l * r == k * q
        x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
        out1 = torch.empty(batch_dim, k, q, device=x.device, dtype=x.dtype).transpose(0, 1)
        out1 = torch.bmm(x_reshaped, w1_bfly.transpose(-1, -2), out=out1)
        out1 = out1.transpose(0, 1).reshape(batch_dim, r, l).transpose(-1, -2).contiguous().transpose(0, 1)
        out2 = torch.empty(batch_dim, l, s, device=x.device, dtype=x.dtype).transpose(0, 1)
        out2 = torch.bmm(out1, w2_bfly.transpose(-1, -2), out=out2)
        out2 = out2.permute(1, 2, 0).reshape(*batch_shape, s * l)
        ctx.save_for_backward(x, w1_bfly, w2_bfly, out1)
        return out2

    @staticmethod
    #@torch.cuda.amp.custom_bwd
    def backward(ctx, dout):
        x, w1_bfly, w2_bfly, out1 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        k, q, p = w1_bfly.shape
        l, s, r = w2_bfly.shape
        # assert k * p == n
        # assert l * r == k * q
        dx, dw1_bfly, dw2_bfly = None, None, None
        # dout_reshaped = dout.reshape(batch_dim, sqrtn, sqrtn).permute(2, 1, 0).contiguous()
        dout_reshaped = dout.reshape(batch_dim, s, l).transpose(-1, -2).contiguous()
        dout_reshaped = dout_reshaped.transpose(0, 1)
        if ctx.needs_input_grad[2]:
            # dw2_bfly = torch.empty(l, s, r, device=w2_bfly.device, dtype=w2_bfly.dtype)
            # dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1, out=dw2_bfly)
            dw2_bfly = torch.bmm(dout_reshaped.transpose(-1, -2), out1.conj())
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[0]:
            dout1 = torch.empty(batch_dim, l, r, device=x.device, dtype=x.dtype).transpose(0, 1)
            dout1 = torch.bmm(dout_reshaped, w2_bfly.conj(), out=dout1)
            dout1 = dout1.transpose(0, 1).transpose(-1, -2).contiguous().reshape(batch_dim, k, q).transpose(0, 1)
            # dout1 = dout1.permute(1, 2, 0).contiguous().transpose(0, 1)
            if ctx.needs_input_grad[0]:
                dx = torch.empty(batch_dim, k, p, device=x.device, dtype=x.dtype)
                dx = torch.bmm(dout1, w1_bfly.conj(), out=dx.transpose(0, 1)).transpose(0, 1).reshape(*batch_shape, n)
            if ctx.needs_input_grad[1]:
                x_reshaped = x.reshape(batch_dim, k, p).transpose(0, 1)
                dw1_bfly = torch.bmm(dout1.transpose(-1, -2), x_reshaped.conj())
        return dx, dw1_bfly, dw2_bfly


blockdiag_butterfly_multiply = BlockdiagButterflyMultiply.apply



class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        """Subclasses should call reset_parameters
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Subclasses may override {in,out}_features_extended
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = out_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        self.set_weights_from_dense_init(dense_init_fn_=partial(init.kaiming_uniform_, a=math.sqrt(5)))
        self.reset_parameters_bias()

    def set_weights_from_dense_init(self, dense_init_fn_):
        raise NotImplementedError

    def reset_parameters_bias(self):
        if self.bias is not None:
            fan_in = self.bias.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    @property
    def saving(self):
        raise NotImplementedError

    def convert_to_dense_weight(self):
        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}
        dense_weight = self.forward_matmul(torch.eye(self.in_features, **factory_kwargs)).T
        return dense_weight

    def preprocess(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            x = F.pad(x, (0, self.in_features_extended - in_features))
        return x

    def postprocess(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        # Convert bias to output.dtype in case of AMP, otherwise bias and activation will be in FP32
        return (output + self.bias.to(dtype=output.dtype)) if self.bias is not None else output

class MonarchLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, **kwargs):
        super().__init__(*args, **kwargs)
        in_blksz = int(math.ceil(self.in_features / nblocks))
        out_blksz = int(math.ceil(self.out_features / nblocks))
        self.in_features_extended = in_blksz * nblocks
        self.out_features_extended = out_blksz * nblocks
        self.nblocks=nblocks
        if self.in_features_extended < self.out_features_extended:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, in_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
        else:
            self.blkdiag1 = nn.Parameter(torch.empty(nblocks, out_blksz, in_blksz))
            self.blkdiag2 = nn.Parameter(torch.empty(nblocks, out_blksz, out_blksz))
        self.reset_parameters()
        print(f'Linear class {self.__class__}: saving={self.saving}')

    def reset_parameters(self) -> None:
        # Mimic init.kaiming_uniform: https://github.com/pytorch/pytorch/blob/24087d07ca7ffa244575d259711dd7c99245a67a/torch/nn/init.py#L360
        for blkdiag in [self.blkdiag1, self.blkdiag2]:
            fan_in = blkdiag.shape[-1]
            gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                blkdiag.uniform_(-bound, bound)
        self.reset_parameters_bias()

    @property
    def saving(self):
        return ((self.blkdiag1.numel() + self.blkdiag2.numel())
                / (self.in_features * self.out_features))

    def forward_matmul(self, x):
        output = blockdiag_butterfly_multiply(self.preprocess(x), self.blkdiag1, self.blkdiag2)
        return self.postprocess(output)

    def __repr__(self):
        return f'MonarchLinear({self.in_features},{self.out_features},nblocks={self.nblocks})'

    def get_from_linear(weight, in_dim, out_dim, bias, seed, red_fac):
        mod  = MonarchLinear(in_dim, 
                             out_dim,
                             bias=bias is not None,
                             nblocks=int(red_fac*2)) # *2 since we want to keep the parameters same

        return mod
        raise NotImplementedError
        wbfly1, wbfly2 = blockdiag_butterfly_project(weight.T, mod.blkdiag1.shape) # check weight needs to be transposed or not
        mod.bias.data[:] = bias
        mod.blkdiag1.data[:,:,:] = wbfly1
        mod.blkdiag2.data[:,:,:] = wbfly2
        return mod
        
        



def ConvertMonarch(model, size_limit, red_fac, init_seed):
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
            new_attr = MonarchLinear.get_from_linear(target_attr.weight, target_attr.in_features, 
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

                new_attr = MonarchLinear.get_from_linear(target_attr.weight, target_attr.in_features, 
                                       target_attr.out_features,
                                       target_attr.bias,
                                       seed,
                                       red_fac)
                print("replaced", target_attr)
                setattr(pytorch_model, name, new_attr)
        init_seed = init_seed + 1
        _diag(name, immediate_child_module, size_limit, red_fac, init_seed)
