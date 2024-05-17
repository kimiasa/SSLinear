import torch
from sketch_structured_linear.SSL import SSL
from benchmark_linear_layers.monarch import MonarchLinear
from benchmark_linear_layers.bdiagbfly import BlockdiagButterflyLinear
from benchmark_linear_layers.lowrank import LowRankLinear
from torch import nn
from experiments.utils import save_to_csv

import numpy as np
import pandas as pd
import argparse

device_index = 0
seed = 1111
default_dtype = torch.float16
data_output_dir = "./results/"

layer_types= [
    # torch.nn.Linear,
    SSL,
    # MonarchLinear,
    # BlockdiagButterflyLinear,
    # LowRankLinear,
    # BlockSparseLinear
]

shapes = [(2**n, 2**n) for n in range(9,15)]
batch_sizes = [8, 16, 32] + [2**n for n in range(7, 16)]
# Skipping 8 due to compiler errors during autotune
reduction_factors = [2, 4]


def time_random_in_forward_cuda_event(model: nn.Module, input_shape, batch_size, generate_grad=False, warmup=True, repetitions=25):
    timings = []
    full_shape = (batch_size, *input_shape)
    if warmup:
        for _ in range(2):
            x = torch.rand(*full_shape)
            _ = model(x)
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repetitions):
        x = torch.rand(*full_shape)

        if generate_grad:
            starter.record()
            _ = model(x)
            ender.record()

        else:
            with torch.no_grad():
                starter.record()
                _ = model(x)
                ender.record()

        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))
    return np.sum(timings) / repetitions, np.std(timings) / repetitions

def main(filepath):
    for batch_size in batch_sizes:
        for shape in shapes:
            for layer_type in layer_types:
                if layer_type == SSL:
                    models = [(SSL(*shape, redn_factor=r, bias=True), f'SSL{r}x') for r in reduction_factors]
                elif layer_type == LowRankLinear:
                    models = [(LowRankLinear(*shape, compression=0.5, bias=True), 'LowRankLinear0.5x')]
                elif layer_type == nn.Linear:
                    models = [(nn.Linear(*shape, device='cuda', bias=True), 'nnLinear')]
                else:
                    models= [(layer_type(*shape, device='cuda', bias=True), f'{layer_type.__name__}')]
                for model, label in models:
                    print(f'Starting {label} shape:{shape} batch:{batch_size}')
                    avg_time, std_dev = time_random_in_forward_cuda_event(
                        model = model,
                        input_shape = (shape[1],),
                        batch_size = batch_size,
                        repetitions = 100
                    )
                    new_data = {
                        'model': [label],
                        'shape': [shape],
                        'batch': [batch_size],
                        'avg_time_ms': [avg_time],
                        'std_dev_ms': [std_dev],
                    }
                    new_df = pd.DataFrame(new_data)
                    save_to_csv(new_df, filepath)
                    print(f'{label} shape:{shape} batch:{batch_size} {avg_time:3f} ms +- {std_dev:2f}')

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_default_device(f'cuda')
    torch.set_default_dtype = default_dtype
    torch.cuda.set_device(device_index)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process a file path.")

    # Add an optional argument for the file path
    parser.add_argument('-f', '--file', type=str, help='The path to the output csv file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the file path from the parsed arguments
    file_path = args.file

    # Check if the file path is provided
    if file_path:
        print(f"The provided file path is: {file_path}")
    else:
        print("No file path provided.")
    main(file_path)