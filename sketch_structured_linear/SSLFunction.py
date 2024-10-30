from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

try:
    from .SSL_Kernel.SSLForward import *
    from .SSL_Kernel.SSLBackward import *
except:
    from SSL_Kernel.SSLForward import *
    from SSL_Kernel.SSLBackward import *


import time

controls = {}
controls['triton_allow_tf32'] = False
controls['triton_allow_autotune'] = False


class SketchStructuredLinearFunction(torch.autograd.Function):
             
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input: torch.tensor, weight: torch.tensor, bias: torch.tensor,
                random_numbers: torch.tensor, redn_factor: int,
                BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 128, BLOCK_SIZE_K: int = 32) -> torch.tensor:  


        '''
        Args:
            input (Tensor): (batch_size, in_features) (M, K)
            hashed_weight (Tensor): (N, cK), the compressed weight matrix for layer, c is compression factor
            random_numbers (Tensor): (4), hash_function: (R0 * (1+c_index) + R1 * (1+k_index) + R2 * n_index  + R3)
            out_features (int): N
            redn_factor (int): The factor of 2 to determine compression
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: tuned block sizes

        Returns:
            output (Tensor): (batch_size, out_features)
        '''

        assert(random_numbers.numel() == 4)
        R3, R2, R1, R0 = random_numbers[3].item(), random_numbers[2].item(), random_numbers[1].item(), random_numbers[0].item()

        batch_size, in_features, out_features= input.shape[0], input.shape[1], weight.shape[0]

        output = ssl_forward_tl(input, weight.T, bias, batch_size, in_features, out_features, redn_factor, R3, R2, R1, R0, 
                                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, BIAS=bias is not None,
                                allow_tf32=controls['triton_allow_tf32'], allow_autotune=controls['triton_allow_autotune'])
        
        ctx.save_for_backward(input, weight, bias, random_numbers)
        ctx.redn_factor = redn_factor
        ctx.block_sizes = (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K) 

        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        
        input, weight, bias, random_numbers = ctx.saved_tensors
        
        assert(random_numbers.numel() == 4)
        R3, R2, R1, R0 = random_numbers[3].item(), random_numbers[2].item(
        ), random_numbers[1].item(), random_numbers[0].item()

        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = ctx.block_sizes
        
        redn_factor = ctx.redn_factor
        M, K, N= input.shape[0], input.shape[1], weight.shape[0]

        if redn_factor <= 1:
            input_grad = torch.matmul(grad, weight)
            weight_grad = torch.matmul(input.T, grad).T

        else:
            input_grad, weight_grad = ssl_backward_tl(input.contiguous(), weight, grad.contiguous(), M, K, N, redn_factor, R3, R2, R1,
                                                        R0, allow_tf32=controls['triton_allow_tf32'],
                                                        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K)
        
        bias_grad = None
        if ctx.needs_input_grad[2] and bias is not None:
            bias_grad = grad.sum(dim=0)

        return input_grad, weight_grad, bias_grad, None, None, None, None, None 
    
ssl_linear = SketchStructuredLinearFunction.apply

    
