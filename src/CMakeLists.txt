
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_ROOT_DIR}/common/inc)
link_directories(${CUDA_SDK_ROOT_DIR}/lib)

cuda_compile(CUDA_SOURCES gpufilter.cu sat.cu)
add_library(gpufilter timer.cpp gpudefs.cpp ${CUDA_SOURCES})

add_executable(ex1 example_sat1.cc)

cuda_add_executable(ex2 example_sat2.cc)
target_link_libraries(ex2 gpufilter)

cuda_add_executable(ex3 example_sat3.cc)
target_link_libraries(ex3 gpufilter)

add_executable(ex4 example_r1.cc)

cuda_add_executable(ex5 example_r2.cc)
target_link_libraries(ex5 gpufilter)

cuda_add_executable(ex6 example_r3.cc)
target_link_libraries(ex6 gpufilter)

cuda_add_executable(ex7 example_gauss.cc)
target_link_libraries(ex7 gpufilter)

include_directories(${OPENCV_DIR})

cuda_add_executable(recursive app_recursive.cc)
target_link_libraries(recursive cv highgui gpufilter)
