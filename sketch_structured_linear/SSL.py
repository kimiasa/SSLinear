import math
import triton
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

try:
    from .SSLFunction import ssl_linear
    from .SSL_Kernel.SSLForward import ssl_forward_kernel_pretune, default_vec
except:
    from SSLFunction import ssl_linear
    from SSL_Kernel.SSLForward import ssl_forward_kernel_pretune, default_vec

from functools import lru_cache


BLOCK_K_SIZE_MIN = 32

class SSL(nn.Module):
    P = 45007
    R = 4

    '''
        Args:
            P (int): The prime number used in the hash function
            R (int): Number of random numbers
    '''

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
        self.batch_size = kwargs.get('batch_size', 256)

        self.red_in_features = (in_features  // redn_factor + self.BLOCK_SIZE_K - 1) // self.BLOCK_SIZE_K * self.BLOCK_SIZE_K 

        self.seed = seed + kwargs.get('layer_idx', 0)
        
        # random numbers are always on the CPU
        self.random_numbers = self._generate_random_numbers(seed)

        # weight
        self.weight = nn.Parameter(torch.zeros((out_features, self.red_in_features), dtype=dtype), requires_grad = True)
        torch.nn.init.xavier_uniform_(self.weight)
        # bias term
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        #self.initialize_parameters()


    def initialize_parameters(self) -> None:
        full_weight = torch.zeros((self.out_features, self.in_features), device=self.weight.device, dtype=self.weight.dtype)
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(full_weight, -bound, bound)
        IDX = SSL.get_idx(-1, self.in_features, self.out_features, block_m=self.BLOCK_SIZE_M, block_k=self.BLOCK_SIZE_K, block_n=self.BLOCK_SIZE_N, 
                        R3=self.random_numbers[3].item(), R2=self.random_numbers[2].item(), R1=self.random_numbers[1].item(), R0=self.random_numbers[0].item(), 
                        reduction_factor=self.redn_factor, device=self.weight.device) # K x N
        IDX = torch.transpose(IDX, 0, 1).contiguous()
        comp_weight = torch.zeros_like(self.weight.data).view(-1)
        comp_ct = torch.zeros_like(self.weight.data).view(-1)
        ones = torch.ones_like(full_weight).view(-1)
        full_weight = full_weight.view(-1)
        comp_weight.scatter_add_(0, IDX.view(-1), full_weight)
        comp_ct.scatter_add_(0, IDX.view(-1), ones)
        comp_weight = comp_weight / (1e-6 + comp_ct)
        comp_weight = comp_weight.view(*self.weight.shape)
        self.weight.data[:,:] = comp_weight
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
            

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

    def _generate_random_numbers(self, seed: int):
        torch.manual_seed(seed)
        x = torch.randint(0, SSL.P, (SSL.R - 1,)).type(
            torch.int32).requires_grad_(False)
        x = torch.cat([torch.tensor([SSL.P], dtype=torch.int32), x])
        return x.requires_grad_(False).cpu()

    def autotune(self):

        device = torch.device('cuda') if torch.cuda.is_available() else (_ for _ in ()).throw(RuntimeError("CUDA is not available. Please run on a CUDA-capable GPU."))
                
        sample_input = torch.randn(self.batch_size, self.in_features, device=device, dtype=self.weight.dtype)
        output = torch.empty(self.batch_size, self.out_features, device=device, dtype=self.weight.dtype)

        block_m, block_k, block_n = (torch.zeros((1,), device=device, dtype=torch.int), 
                                    torch.zeros((1,), device=device, dtype=torch.int), 
                                    torch.zeros((1,), device=device, dtype=torch.int), 
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
            block_m, block_k, block_n, 
            M, N, K, K // self.redn_factor,
            sample_input.stride(0), sample_input.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1),
            allow_tf32=False,
            R3=R3, R2=R2, R1=R1, R0=R0,
            GROUP_SIZE_M=1,
            VEC=default_vec,
            BIAS=self.bias is not None,
            redn_factor=self.redn_factor
        )

        self.BLOCK_SIZE_M = block_m.item()
        self.BLOCK_SIZE_K = block_k.item()
        self.BLOCK_SIZE_N = block_n.item()

    @staticmethod
    def get_idx(M, K, N, block_m, block_k, block_n, R3,R2,R1,R0, reduction_factor, device):
        # weight shape is N,K || but we will start with how it is viewed in sslforward
        if reduction_factor == 1: 
            return torch.arange(K * N, device=device).reshape(N, K).T
        red_input_dim = (K  // reduction_factor + block_k - 1) // block_k * block_k   # keep it multiple of block size k 
        IDX = torch.arange(N*red_input_dim, device=device).long().reshape(N, red_input_dim)
        IDXT = IDX.T
        FullIDX = torch.zeros((K, N), device=device, dtype=torch.long)
        for i in range((K+block_k -1)//block_k):
            for j in range((N+block_k-1)//block_n):
                it = i // reduction_factor
                itin = i % reduction_factor
                IDX = (R3 + R2 * j + R1*(it+1))
                IDX1 = R0*(itin+1)
                VEC = default_vec
                offset = block_k - ((IDX + IDX1) * VEC) % block_k
                locs_k = (offset + torch.arange(block_k, device=device).long() ) % block_k
                block = IDXT[it*block_k:(it+1)*block_k,j*block_n:(j+1)*block_n][locs_k]
                kl, nl = FullIDX[i*block_k:(i+1)*block_k,j*block_n:(j+1)*block_n].shape
                FullIDX[i*block_k:(i+1)*block_k,j*block_n:(j+1)*block_n] = block[:kl,:nl]
        return FullIDX
    
    
    @property
    def saving(self):
        return (self.weight.numel() / (self.in_features * self.out_features))

    def __repr__(self):        
        return "SketchStructuredLinear(in={}, out={}, compression={}, seed={}, saving={})".format(self.in_features, self.out_features, self.redn_factor, self.seed, self.saving)
    
    
    
