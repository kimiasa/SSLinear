import torch
from sketch_structured_linear.SSL import SSL
from benchmark_linear_layers.monarch import MonarchLinear
from benchmark_linear_layers.bdiagbfly import BlockdiagButterflyLinear
from benchmark_linear_layers.lowrank import LowRankLinear
from torch import nn
from triton.testing import do_bench_cudagraph
from experiments.utils import save_to_csv

import numpy as np
import pandas as pd
import argparse

device_index = 0
seed = 1111
default_dtype = torch.float16
data_output_dir = "./output.csv"

layer_types= [
    torch.nn.Linear,
    SSL,
    LowRankLinear,
    MonarchLinear,
    BlockdiagButterflyLinear,
    #BlockSparseLinear
]

shapes = [(2**n, 2**n) for n in range(9,15)]
batch_sizes = [2**n for n in range(15, 16)]

shapes_of_interest = [
    (1024, 1024, 1024),
    (16384, 1024, 1024),
    (32768, 1024, 1024),
    (36864, 768, 768),
    (36864, 768, 3072),
    (36864, 3072, 768),
    (12288, 768, 2304),
    (12288, 768, 50256),
]
reduction_factors = [1, 2, 4, 8, 16]

def count_parameters(model):    
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    # print(f"Total Trainable Params: {total_params}")
    return total_params

def time_random_in_forward_proton(model, input_shape, batch_size, generate_grad=False, warmup=True, repetitions=25):
    import triton.profiler as proton

    timings = []
    full_shape = (batch_size, *input_shape)
    if warmup:
        for _ in range(2):
            x = torch.rand(*full_shape, dtype=default_dtype)
            _ = model(x)

    session_id = proton.start(name="profile_name", context="python")
    proton.deactivate(session_id)
    for _ in range(repetitions):
        x = torch.rand(*full_shape, dtype=default_dtype)

        if generate_grad:
            _ = model(x)

        else:
            with torch.no_grad():
                proton.activate(session_id)
                _ = model(x)
                proton.deactivate(session_id)

    return np.sum(timings) / repetitions, np.std(timings) / repetitions

def time_random_in_forward_cuda_event(model: nn.Module, input_shape, batch_size, generate_grad=False, warmup=True, repetitions=25):
    timings = []
    full_shape = (batch_size, *input_shape)
    x = torch.rand(*full_shape, dtype=default_dtype)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    return do_bench_cudagraph(lambda: model(x))

def profile_all_layers(batch_size, shape):
    df = pd.DataFrame()
    for layer_type in layer_types:
        if layer_type == SSL:
            models = [(SSL(*shape, redn_factor=r, bias=True, dtype=default_dtype), f'SSL{r}x') for r in reduction_factors]
        elif layer_type == LowRankLinear:
            models = [(LowRankLinear(*shape, compression=1.0/r, bias=True, dtype=default_dtype), f'LowRankLinear1/{r}x') for r in reduction_factors]
        elif layer_type == nn.Linear:
            models = [(nn.Linear(*shape, device='cuda', bias=True, dtype=default_dtype), 'nnLinear')]
        elif layer_type == MonarchLinear:
            models = [(MonarchLinear(*shape, nblocks=r*2, bias=True, dtype=default_dtype), f'Monarch{r*2}b') for r in reduction_factors]
        else:
            models= [(layer_type(*shape, device='cuda', bias=True, dtype=default_dtype), f'{layer_type.__name__}')]
        for model, label in models:
            print(f'Starting {label} shape:{shape}')
            avg_time = time_random_in_forward_cuda_event(
                model = model,
                input_shape = (shape[0],),
                batch_size = batch_size,
                repetitions = 100
            )

            num_params = count_parameters(model)
            new_data = {
                'model': [label],
                'shape': [(batch_size, shape[0], shape[1])],
                'num_params': num_params,
                'avg_time_ms': [avg_time],
            }
            print(f'{label} shape:{shape} num_params: {num_params} {avg_time:3f} ms')
            df = pd.concat([df, pd.DataFrame(new_data)])

    return df
        
def profile_square(filepath):
    for batch_size in batch_sizes:
        for shape in shapes:
            save_to_csv(profile_all_layers(batch_size, shape), filepath)

def profile_target(filepath):
    for shape in shapes_of_interest:
        print(shape)
        save_to_csv(profile_all_layers(shape[0], (shape[1], shape[2])), filepath)

def main(target, square, filepath):
    if target:
        profile_target(filepath)
    if square:
        profile_square(filepath)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_default_device('cuda')
    torch.cuda.set_device(device_index)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process a file path.")

    # Add an optional argument for the file path
    parser.add_argument('-f', '--file', type=str, help='The path to the output csv file', default=data_output_dir)
    parser.add_argument('--square', action='store_true', help='A flag for benchmarking square shapes')
    parser.add_argument('--paper', action='store_true', help='A flag for benchmarking shapes of interest for the paper')
    parser.add_argument('--proton', action='store_true', help='A flag for using proton instead of cuda events')
    parser.add_argument('-d', '--dtype', type=str, default='float16')

    # Parse the command-line arguments
    args = parser.parse_args()

    default_dtype = getattr(torch, args.dtype)
    torch.set_default_dtype = default_dtype
    # Access the file path from the parsed arguments
    file_path = args.file

    # Check if the file path is provided
    if file_path:
        print(f"The provided file path is: {file_path}")
    else:
        print("No file path provided.")
    main(args.paper, args.square, file_path)

