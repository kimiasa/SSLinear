import torch
from pytorch_block_sparse import BlockSparseLinear

def ConvertBlockSparse(model, size_limit, red_fac, init_seed):
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
            elif target_attr.in_features < size_limit or target_attr.out_features < size_limit:
                print("ignored due to scale", target_attr)
                continue
            new_attr = BlockSparseLinear(target_attr.in_features, 
                                       target_attr.out_features,
                                       bias=target_attr.bias is not None,
                                       density=1.0/red_fac,
                                       torch_nn_linear = target_attr)
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

                new_attr = BlockSparseLinear(target_attr.in_features, 
                                       target_attr.out_features,
                                       bias = target_attr.bias is not None,
                                       density=1.0/red_fac,
                                       torch_nn_linear = target_attr)
                print("replaced", target_attr)
                setattr(pytorch_model, name, new_attr)
        init_seed = init_seed + 1
        _diag(name, immediate_child_module, size_limit, red_fac, init_seed)
