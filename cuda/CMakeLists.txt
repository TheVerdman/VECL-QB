cmake_minimum_required(VERSION 3.26)
project(vecl_qb_cuda_placeholder LANGUAGES CXX)

add_library(vecl_qb_cuda_placeholder INTERFACE)
target_compile_features(vecl_qb_cuda_placeholder INTERFACE cxx_std_17)

# v0 is an ABI/API scaffold only. The Python-facing function delegates to the
# CPU oracle in vecl.sparse.cuda_like until CUDA kernels are implemented and
# differentially tested.
