import torch
from torch import nn
from pytorch_block_sparse import BlockSparseLinear
import numpy as np

class LowRankLinear(nn.Module):
    def __init__(self, input, output, compression, bias, dtype=torch.float):
        super(LowRankLinear, self).__init__()
        self.idim = input
        self.odim = output
        self.compression = compression
        self.intermediate_dim = int(((input * output) * compression) / (input + output))
        
        # power of 2
        self.intermediate_dim = int(2** np.round(np.log2(self.intermediate_dim)))
        if self.intermediate_dim > self.idim:
            self.intermediate_dim = self.intermediate_dim // 2
        
        assert(self.intermediate_dim > 0)
        self.w1 = nn.Parameter(torch.zeros((self.idim, self.intermediate_dim), dtype=dtype), requires_grad=True)
        self.w2 = nn.Parameter(torch.zeros((self.intermediate_dim, self.odim), dtype=dtype), requires_grad=True)
        nn.init.normal_(self.w1.data)
        nn.init.normal_(self.w2.data)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                self.odim, dtype=dtype), requires_grad=True)
        self.init_weight()

    def init_weight(self):
        temp = torch.zeros(self.odim, self.idim)
        torch.nn.init.xavier_uniform_(temp)
        b, c = self.wt_orig_to_comp(temp)
        self.w1.data[:,:] = b
        self.w2.data[:,:] = c
        

    def forward(self, x):
        #W = torch.matmul(self.w1, self.w2)
        #x = torch.matmul(x, W)
        x = torch.matmul(torch.matmul(x, self.w1), self.w2)
        if self.bias is not None:
            x = x + self.bias
        return x

    def __repr__(self):
        return "LowRankLinear(in={}, int={}, out={})".format(self.idim, self.intermediate_dim, self.odim)

    def wt_orig_to_comp(self, W):
        # orig is (out,in)
        U,S,Vh = torch.svd_lowrank(W.T, q=self.intermediate_dim, niter=10)
        B = torch.matmul(U, torch.sqrt(torch.diag(S)))
        C = torch.matmul(Vh, torch.sqrt(torch.diag(S)))
        return B, C.T

    def wt_comp_to_orig(self, B, C):
        # orig is out,in
        return torch.matmul(B,C).T



def ConvertLowRank(model, size_limit, red_fac, init_seed):
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
            new_attr = LowRankLinear(target_attr.in_features, 
                                       target_attr.out_features,
                                       bias=target_attr.bias is not None,
                                       compression=1.0/red_fac)
            print("replaced", target_attr, "by", new_attr)
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

                new_attr = LowRankLinear(target_attr.in_features, 
                                       target_attr.out_features,
                                       bias=target_attr.bias is not None,
                                       compression=1.0/red_fac)
                print("replaced", target_attr)
                setattr(pytorch_model, name, new_attr)
        init_seed = init_seed + 1
        _diag(name, immediate_child_module, size_limit, red_fac, init_seed)
