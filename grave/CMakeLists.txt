cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(gpufilter CXX C)

if(NOT CUDA_SDK_ROOT_DIR) # it is ugly, I know
  set(CUDA_SDK_ROOT_DIR /usr/local/cuda/samples)
endif()
if(NOT CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
endif()

find_package(CUDA REQUIRED)

#find_package(OpenCV REQUIRED)

if(NOT USE_SM)
    #set(USE_SM 30) #using SM_35 in TITAN was slower than SM_30
    #set(USE_SM 35) #..it does not seem to be the case anymore..
    set(USE_SM 61) # Pascal
endif()

set(CUDA_NVCC_FLAGS
#  ${CUDA_NVCC_FLAGS}
#  -gencode arch=compute_20,code=sm_21
  --ptxas-options=-v -O3
  -ftz=true -prec-div=false -prec-sqrt=false
  -use_fast_math #-maxrregcount=255
  -arch=sm_${USE_SM}
)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Set this variable to true if you want to watch compilation
set(CMAKE_VERBOSE_MAKEFILE FALSE)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

include_directories(${INCLUDE_DIR})

add_subdirectory(src)
