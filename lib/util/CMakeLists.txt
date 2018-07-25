include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_SDK_ROOT_DIR}/common/inc)

add_library(util gaussian.cpp image.cpp timer.cpp ${CUDA_UTILS})
target_link_libraries(util)
