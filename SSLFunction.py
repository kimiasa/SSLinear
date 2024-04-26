from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

from .SSL_Kernel.sketch_structured_linear.impl.SSLForward import *
from .SSL_Kernel.sketch_structured_linear.impl.SSLBackward import *

import time


controls = {}
controls['triton_allow_tf32'] = False
controls['triton_allow_autotune'] = False

class SketchStructuredLinearFunction(torch.autograd.Function):
             
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input: torch.tensor, weight: torch.tensor,
                random_numbers: torch.tensor, redn_factor: int) -> torch.tensor:  

        ctx._fwd_used_autocast = torch.is_autocast_enabled()

        '''
        Args:
            input (Tensor): (batch_size, in_features) (M, K)
            hashed_weight (Tensor): (N, cK), the compressed weight matrix for layer, c is compression factor
            random_numbers (Tensor): (4), hash_function: (R0 * (1+c_index) + R1 * (1+k_index) + R2 * n_index  + R3)
            out_features (int): N
            redn_factor (int): The factor of 2 to determine compression

        Returns:
            output (Tensor): (batch_size, out_features)
        '''

        assert(random_numbers.numel() == 4)
        R3, R2, R1, R0 = random_numbers[3].item(), random_numbers[2].item(), random_numbers[1].item(), random_numbers[0].item()

        batch_size, in_features, out_features= input.shape[0], input.shape[1], weight.shape[0]

        output = ssl_forward_tl(input, weight.T.contiguous(), batch_size, in_features, out_features, redn_factor, R3, R2, R1, R0,
                                      allow_tf32=controls['triton_allow_tf32'], allow_autotune=controls['triton_allow_autotune'])
        
        ctx.save_for_backward(input, weight, random_numbers)
        ctx.redn_factor = redn_factor

        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):

        
        with torch.cuda.amp.autocast(ctx._fwd_used_autocast):
        
            input, weight, random_numbers = ctx.saved_tensors
            
            assert(random_numbers.numel() == 4)
            R3, R2, R1, R0 = random_numbers[3].item(), random_numbers[2].item(
            ), random_numbers[1].item(), random_numbers[0].item()

            
            redn_factor = ctx.redn_factor
            M, K, N= input.shape[0], input.shape[1], weight.shape[0]

            input_grad, weight_grad = ssl_backward_tl(input.contiguous(), weight, grad.contiguous(), M, K, N, redn_factor, R3, R2, R1,
                                                        R0, allow_tf32=controls['triton_allow_tf32'])
        
            #bias_grad = None
            #if ctx.needs_input_grad[2]:
            #    bias_grad = grad.sum(0)

                                          
            return input_grad, weight_grad.T.contiguous(), None, None 
    
ssl_linear = SketchStructuredLinearFunction.apply

    