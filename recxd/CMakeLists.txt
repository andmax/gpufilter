cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
project(recxd CXX)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS
  --ptxas-options=-v
  -O3 -ftz=true -prec-div=false
  -prec-sqrt=false -use_fast_math
  -arch=sm_60
  -std=c++11
)

set(CMAKE_CXX_STANDARD 11)

set(GPUFILTER_DIR ~/gpufilter)

include_directories(${GPUFILTER_DIR}/lib ${GPUFILTER_DIR}/src)

include_directories(~/cub/)

link_directories(${GPUFILTER_DIR}/build/lib/util)

add_subdirectory(src)

add_subdirectory(thrustps)

#"/usr/local/cuda/bin/nvcc"   -o bin/cubps cubps.cu \
#-Xptxas -v -Xcudafe -#  -Xcompiler -ffloat-store -m64 -I../../cub/ -I../../cub/test \
#-I../../gpufilter/lib -L../../gpufilter/build/lib/util -lutil  -O3
