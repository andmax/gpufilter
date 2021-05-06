
cuda_add_executable(thrustps thrustps.cu)
target_link_libraries(thrustps util)
set_property(TARGET thrustps PROPERTY CXX_STANDARD 11)
