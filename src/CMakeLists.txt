
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_ROOT_DIR}/common/inc)
link_directories(${CUDA_SDK_ROOT_DIR}/lib)

cuda_compile(CUDA_GPUFILTER gpufilter.cu)
add_library(gpufilter timer.cpp ${CUDA_GPUFILTER})

add_executable(ex1 example_sat1.cc)
#add_executable(ex2 example_r1.cc)

cuda_add_executable(ex3 example_r2.cc)
target_link_libraries(ex3 gpufilter)

include_directories(${OPENCV_DIR})

add_executable(recursive_cpu app_recursive_cpu.cc)
target_link_libraries(recursive_cpu cv highgui)
