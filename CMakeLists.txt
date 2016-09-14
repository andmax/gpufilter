cmake_minimum_required(VERSION 2.6.0 FATAL_ERROR)
project(boundary CXX C)

if(NOT CUDA_SDK_ROOT_DIR)
  set(CUDA_SDK_ROOT_DIR /usr/local/cuda-7.0/samples)
endif()
if(NOT CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-7.0)
endif()

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS}
#  -gencode arch=compute_30,code=sm_30
#  -gencode arch=compute_20,code=sm_21
  --ptxas-options=-v -O3
  -ftz=true -prec-div=false -prec-sqrt=false
  -use_fast_math #-maxrregcount=255
  -arch=sm_${USE_SM}
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_35)

include_directories(lib)

add_subdirectory(lib)
add_subdirectory(src)
