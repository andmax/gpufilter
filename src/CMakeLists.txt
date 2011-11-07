
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_ROOT_DIR}/common/inc)
link_directories(${CUDA_SDK_ROOT_DIR}/lib)

cuda_compile(CUDA_SOURCES gpufilter.cu sat.cu)
add_library(gpufilter timer.cpp ${CUDA_SOURCES})

add_executable(ex1 example_sat1.cc)

cuda_add_executable(ex2 example_sat2.cc)
target_link_libraries(ex2 gpufilter)

#add_executable(ex3 example_r1.cc)

cuda_add_executable(ex4 example_r2.cc)
target_link_libraries(ex4 gpufilter)

include_directories(${OPENCV_DIR})

add_executable(recursive_cpu app_recursive_cpu.cc)
target_link_libraries(recursive_cpu cv highgui)
