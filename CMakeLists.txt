cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(gpufilter CXX C)

find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)

set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS}
  -gencode arch=compute_30,code=sm_30
  --ptxas-options=-v
  -ftz=true -prec-div=false -prec-sqrt=false
)

# Set this variable to true if you want to watch compilation
set(CMAKE_VERBOSE_MAKEFILE FALSE)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

include_directories(${INCLUDE_DIR})

add_subdirectory(src)
