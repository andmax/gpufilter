
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_ROOT_DIR}/common/inc)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib)

cuda_compile(CUDA_SOURCES gpudefs.cu)
cuda_add_library(gpufilter timer.cpp defs.cpp ${CUDA_SOURCES})

add_executable(ex1 example_sat1.cc)
target_link_libraries(ex1 gpufilter)

cuda_add_executable(ex2 example_sat2.cc gpufilter.cu)
target_link_libraries(ex2 gpufilter)

cuda_add_executable(ex3 example_sat3.cc gpufilter.cu)
target_link_libraries(ex3 gpufilter)

add_executable(ex4 example_r1.cc)
target_link_libraries(ex4 gpufilter)

cuda_add_executable(ex5 example_r2.cc gpufilter.cu)
target_link_libraries(ex5 gpufilter)

cuda_add_executable(ex6 example_r3.cc gpufilter.cu)
target_link_libraries(ex6 gpufilter)

cuda_add_executable(ex7 example_r4.cc gpufilter.cu)
target_link_libraries(ex7 gpufilter)

cuda_add_executable(ex8 example_r5.cu gpufilter.cu)
target_link_libraries(ex8 gpufilter)

cuda_add_executable(exB example_bspline.cc gpufilter.cu)
target_link_libraries(exB gpufilter)

cuda_add_executable(exG example_gauss.cc gpufilter.cu)
target_link_libraries(exG gpufilter)

add_executable(gaussian gaussian.cc)
target_link_libraries(gaussian gpufilter)

include_directories(${OpenCV_DIR})

cuda_add_executable(recursive app_recursive.cc)
target_link_libraries(recursive ${OpenCV_LIBS} gpufilter)

cuda_add_executable(alg5 run_alg5.cu)
target_link_libraries(alg5 gpufilter)

cuda_add_executable(alg5f4 run_alg5f4.cu)
target_link_libraries(alg5f4 gpufilter)

include_directories(${CUDA_SDK_ROOT_DIR}/common/inc)

set(CUDA_BUILD_CUBIN on)

macro(add_cuda_exec name)
  cuda_add_executable(${name} ${name}.cu)
  target_link_libraries(${name} util)
endmacro()

macro(add_cuda_exec_r name r)
  add_definitions(-DORDER=${r})
  cuda_add_executable(${name}_${r} ${name}.cu)
  target_link_libraries(${name}_${r} util)
  remove_definitions(-DORDER=${r})
endmacro()

add_cuda_exec(alg5f4)
add_cuda_exec(alg5varc)
add_cuda_exec(sat)

add_cuda_exec_r(alg3 1)
add_cuda_exec_r(alg3 2)
add_cuda_exec_r(alg3 3)
add_cuda_exec_r(alg3 4)
add_cuda_exec_r(alg3 5)

add_cuda_exec_r(alg4 1)
add_cuda_exec_r(alg4 2)
add_cuda_exec_r(alg4 3)
add_cuda_exec_r(alg4 4)
add_cuda_exec_r(alg4 5)

add_cuda_exec_r(alg5 1)
add_cuda_exec_r(alg5 2)
add_cuda_exec_r(alg5 3)
add_cuda_exec_r(alg5 4)
add_cuda_exec_r(alg5 5)

add_cuda_exec_r(alg6 1)
add_cuda_exec_r(alg6 2)
add_cuda_exec_r(alg6 3)
add_cuda_exec_r(alg6 4)
add_cuda_exec_r(alg6 5)
