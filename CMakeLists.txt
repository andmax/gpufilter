cmake_minimum_required(VERSION 2.6.0 FATAL_ERROR)
project(gpufilter CXX C)

find_package(CUDA REQUIRED)

if(NOT CUDA_SDK_ROOT_DIR)
    message(FATAL "Please set environment variable NVSDKCUDA_ROOT to the
    SDK base directory (with 'common','lib','bin', etc.), for instance
    '/usr/local/cuda/NVIDIA_GPU_Computing_SDK/C'")
endif()

set(CUDA_NVCC_FLAGS
  -gencode arch=compute_20,code=sm_20
  --ptxas-options=-v
  -ftz=true -prec-div=false -prec-sqrt=false
)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(OPENCV_DIR /usr/include/opencv)

include_directories(${INCLUDE_DIR})

add_subdirectory(src)
