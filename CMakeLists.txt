cmake_minimum_required(VERSION 3.10)
project(CudaTest CUDA)

set(CMAKE_CUDA_STANDARD 11)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_executable(CudaTest main.cu)

