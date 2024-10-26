import torch
import torch.nn as nn
from typing import List, Union, Optional
from transformers import PreTrainedModel
from sketch_structured_linear.SSL import SSL

def get_from_linear(weight, in_dim, out_dim, bias, seed, red_fac):
        mod = SSL(in_dim, out_dim, redn_factor=red_fac, seed=seed, bias=(bias is not None))
        mod.bias.data[:] = bias.data[:]
        # check if M,N,K in config or cache. 
        # If yes, get the block values
        # If no, run autotune
        # BLOCK_K, BLOCK_M, BLOCK_N = autotune(M,N,K)
        # save results to cache or config.

        if red_fac > 1:
            random_numbers = mod.random_numbers
            R3, R2, R1, R0 = random_numbers[3].item(), random_numbers[2].item(
                    ), random_numbers[1].item(), random_numbers[0].item()
            IDX = get_idx(-1, in_dim, out_dim, block_m=BLOCK_K_SIZE_MIN, block_k=BLOCK_K_SIZE_MIN, block_n=BLOCK_K_SIZE_MIN, R3=R3,R2=R2,R1=R1,R0=R0, reduction_factor=red_fac, device=weight.device) # K x N
            IDX = torch.transpose(IDX, 0, 1)
            IDX = IDX.contiguous()
            comp_weight = torch.zeros_like(mod.weight.data).view(-1)
            comp_ct = torch.zeros_like(mod.weight.data).view(-1)
            ones = torch.ones_like(weight).view(-1)
            weight = weight.view(-1)
            comp_weight.scatter_add_(0, IDX.view(-1), weight)
            comp_ct.scatter_add_(0, IDX.view(-1), ones)
            comp_weight = comp_weight / (1e-6 + comp_ct)
            comp_weight = comp_weight.view(*mod.weight.shape)
            mod.weight.data[:,:] = comp_weight
        else:
            mod.weight.data[:,:] = weight

        return mod
        

def is_linear_layer(module: nn.Module) -> bool:
    """Check if the module is a linear layer that should be converted."""
    return isinstance(module, nn.Linear)

def get_module_path(model: nn.Module, target_module: nn.Module) -> Optional[str]:
    """Get the path to a specific module in the model."""
    for name, module in model.named_modules():
        if module is target_module:
            return name
    return None

def convert_to_ss_linear(
    model: PreTrainedModel,
    reduction_factor: int,
    layer_indices: Optional[List[int]] = None,
    skip_attention: Optional[bool] = False,
    init_seed: Optional[int] = 42,
    skip_pattern: Optional[List[str]] = None
) -> PreTrainedModel:
    """
    Convert a HuggingFace model's linear layers to SSLinear layers.
    
    Args:
        model: HuggingFace model to convert
        reduction_factor: Reduction factor for SSLinear layers
        layer_indices: List of layer indices to skip (model specific)
        skip_attention: If True, does not convert attention layers
        init_seed: Initial seed for random number generation
        skip_pattern: List of strings to match against layer names to skip
        
    Returns:
        Converted model
    """
    if skip_pattern is None:
        skip_pattern = []
    
    # Keep track of current seed
    current_seed = init_seed
    
    def should_skip_module(module_path: str) -> bool:
        """Determine if a module should be skipped based on its path."""
        # Skip if path contains any of the skip patterns
        if any(pattern in module_path for pattern in skip_pattern):
            return True
            
        # If layer_indices is specified, check if current layer should be skipped
        if layer_indices is not None:
            # Extract layer index from path if possible
            try:
                for idx in layer_indices:
                    if f"layer.{idx}." in module_path:
                        return True
            except:
                pass
            
        # If skip_attention, do not convert attention layers
        if skip_attention:
            return "attention" in module_path
            
        return False

    def convert_module(module: nn.Module, module_path: str) -> None:
        """Recursively convert linear layers to SSLinear."""
        nonlocal current_seed
        
        for name, child in module.named_children():
            child_path = f"{module_path}.{name}" if module_path else name
            
            if is_linear_layer(child) and not should_skip_module(child_path):
                # Convert to SSLinear
                new_layer = SSLinear.get_from_linear(
                    weight=child.weight.data,
                    in_dim=child.in_features,
                    out_dim=child.out_features,
                    bias=child.bias,
                    seed=current_seed,
                    red_fac=reduction_factor
                )
                current_seed += 1
                
                # Replace the old layer with the new one
                setattr(module, name, new_layer)
            else:
                # Recurse into child modules
                convert_module(child, child_path)

    # Start conversion from the root
    convert_module(model, "")
    
    return model

def convert_specific_architecture(
    model: PreTrainedModel,
    architecture_type: str,
    reduction_factor: int,
    layer_indices: Optional[List[int]] = None,
    skip_attention: Optional[bool] = False,
    init_seed: Optional[int] = 42
) -> PreTrainedModel:
    """
    Convert specific model architectures with predefined settings.
    
    Args:
        model: HuggingFace model to convert
        architecture_type: Type of architecture ('bert', 'gpt', 't5', etc.)
        reduction_factor: Reduction factor for SSLinear layers
        layer_indices: List of layer indices to skip
        skip_attention: If True, does not convert attention layers
        init_seed: Initial seed for random number generation
        
    Returns:
        Converted model
    """
    architecture_configs = {
        'bert': {
            'skip_pattern': ['pooler', 'embeddings'],
            'base_path': 'bert.encoder'
        },
        'gpt2': {
            'skip_pattern': ['wte', 'wpe'],
            'base_path': 'h'
        },
        't5': {
            'skip_pattern': ['embed_tokens', 'final_layer_norm'],
            'base_path': 'encoder.block'
        }
        # Add more architectures as needed
    }
    
    if architecture_type.lower() not in architecture_configs:
        raise ValueError(f"Unsupported architecture: {architecture_type}")
    
    config = architecture_configs[architecture_type.lower()]
    
    return convert_to_ss_linear(
        model=model,
        reduction_factor=reduction_factor,
        layer_indices=layer_indices,
        skip_attention=skip_attention,
        init_seed=init_seed,
        skip_pattern=config['skip_pattern']
    )

# Example usage:
"""
# Generic conversion
model = convert_to_ss_linear(
    model,
    reduction_factor=8,
    layer_indices [1,2,3,4,5],
    skip_attention=False,
    init_seed=42,
    skip_pattern=['pooler', 'embeddings']
)

# Architecture-specific conversion
model = convert_specific_architecture(
    model,
    architecture_type='bert',
    reduction_factor=8,
    layer_indices [1,2,3,4,5],
    skip_attention=True,
    init_seed=42
)
"""
