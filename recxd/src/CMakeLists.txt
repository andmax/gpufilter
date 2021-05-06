
add_executable(gen_random_samples gen_random_samples.cpp)

add_executable(gen_random_samples_double gen_random_samples.cpp)
target_compile_definitions(gen_random_samples_double PRIVATE DOUBLE)

cuda_add_executable(memcpy memcpy.cu)
target_link_libraries(memcpy util)

cuda_add_executable(memcpy_double memcpy.cu OPTIONS -DDOUBLE)
target_link_libraries(memcpy_double util)

cuda_add_executable(paper_code paper.cu)
target_link_libraries(paper_code util)

cuda_add_executable(alg6i1_1 alg6.cu OPTIONS -DALG6I1)
target_link_libraries(alg6i1_1 util)

cuda_add_executable(alg6i2_1 alg6.cu OPTIONS -DALG6I2)
target_link_libraries(alg6i2_1 util)

cuda_add_executable(alg_rd alg_rd.cu)
target_link_libraries(alg_rd util)

cuda_add_executable(alg_rd_smem alg_rd.cu OPTIONS -DUSE_SMEM)
target_link_libraries(alg_rd_smem util)

cuda_add_executable(alg3_1d_fwd alg3_1d.cu OPTIONS -DFWD_ONLY)
target_link_libraries(alg3_1d_fwd util)

cuda_add_executable(alg3_1d alg3_1d.cu)
target_link_libraries(alg3_1d util)

cuda_add_executable(alg3_1d_step2 alg3_1d.cu OPTIONS -DDO_STEP2)
target_link_libraries(alg3_1d_step2 util)

cuda_add_executable(alg3_3d alg3_3d.cu)
target_link_libraries(alg3_3d util)

