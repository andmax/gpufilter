
add_executable(ex1 example_sat.cc)

include_directories(${OPENCV_DIR})

add_executable(recursive_cpu app_recursive_cpu.cc)
target_link_libraries(recursive_cpu cv highgui)
