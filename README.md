

# SSL Linear 



## Installing
```sh
conda create -n sslinear
conda activate sslinear
conda install numpy pandas
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install triton
INSTALL_BENCHMARK_DEPENDENCIES=1 pip install -e .
```
Where `INSTALL_BENCHMARK_DEPENDENCIES` can be optionally set to 0 if you don't plan on running against the benchmarks.

## Using the PyTorch layers
```
from sketch_structured_linear.SSL import SSL
layer = SSL(in_dim, out_dim, reduction_factor, dtype)
```

## Benchmarking

### Benchmarking linear layer latency of various shapes
Saves results to `results.csv`
```sh
python experiments/benchmark_linear.py -f ./outfile.csv [--square] [--paper] [-d float16 or float32]
```
where `--square` will run against square shapes and `--paper` will run against GPT2 and Llama shapes.
