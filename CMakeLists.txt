cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)
project(gpufilter CXX C)

if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

if(NOT CUDA_SDK_ROOT_DIR)
  set(CUDA_SDK_ROOT_DIR /usr/local/cuda/samples)
endif()
if(NOT CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
endif()

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS}
#  -gencode arch=compute_30,code=sm_30
#  -gencode arch=compute_20,code=sm_21
#  -maxrregcount=255
  -arch=sm_${USE_SM}
  --ptxas-options=-v -O3 -ftz=true
  -prec-div=false -prec-sqrt=false -use_fast_math -std=c++11
)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_35) # Kepler
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_61) # Pascal
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_37) # Kepler K80
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_60) # Tesla P100
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_75) # Titan RTX

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(lib)

add_subdirectory(lib)
add_subdirectory(src)
