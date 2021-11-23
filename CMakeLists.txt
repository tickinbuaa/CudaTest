cmake_minimum_required(VERSION 3.10)
project(CudaTest CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(CudaTest main.cu)

target_compile_options(CudaTest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -arch compute_75
        -code sm_75
        -m 64
        >)

set_target_properties(CudaTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
