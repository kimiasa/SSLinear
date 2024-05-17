

# SSL Linear 

<img width="605" alt="Screen Shot 2024-03-22 at 4 37 49 PM" src="https://github.com/kimiasa/SSLinear/assets/98286289/6e090a92-af37-4f84-bc36-bed86a787223">


## Installing

### With Proton (TODO)
To use proton for profiling, install triton from source before installing the package
```
conda create -n sslinear
conda activate sslinear
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

#2 install c++ compilers locally via conda. 
conda install -c conda-forge cxx-compiler

#3 you can check this has the GLIBC you want
strings ~/anaconda3/envs/sslinear/lib/libstdc++.so | grep GLIBCXX

#4 compile triton with correct bin and lib paths 
pip install ninja cmake wheel
PATH=~/anaconda3/envs/sslinear/bin/ LD_LIBRARY_PATH=~/anaconda3/envs/sslinear/lib/ pip install -e .
```

### Without proton/after proton installation
```sh
INSTALL_BENCHMARK_DEPENDENCIES=1 pip install -e .
```
Where `INSTALL_BENCHMARK_DEPENDENCIES` can be optionally set to 0 if you don't plan on running against the benchmarks.

## Benchmarking

### Benchmarking linear layer of various shapes
Saves results to `results.csv`
```sh
python experiments/benchmark_linear.py -f ./outfile.csv
```
