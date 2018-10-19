#!/bin/bash

set -e

libName=libgfoidl-Stochastics-gpu.so

# https://docs.nvidia.com/cuda/pdf/CUDA_Compiler_Driver_NVCC.pdf
nvcc    -std=c++14 \
        --disable-warnings \
        -Wno-deprecated-declarations \
        -o "$libName" \
        -O3 \
        --ptxas-options=-v \
        --machine 64 \
        -x cu \
        -cudart static \
        -shared -rdc=true \
        -Xcompiler -fPIC,-fvisibility=hidden  \
        -gencode=arch=compute_60,code=sm_60 \
        kernel_utils.cu kernel.cu gpu_core.cu

nm -DC "$libName" | grep " T "
