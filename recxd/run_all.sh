#!/bin/bash

set -x

#python run.py ../gpufilter/build/src/alg3_1 1000 0 0 > alg3_1.out

#python run.py ../gpufilter/build/src/alg6_1 1000 0 0 > alg6_1.out

#python run.py build/src/alg6i1_1 1000 0 0 > alg6i1_1.out

#python run.py build/src/alg6i2_1 1000 0 0 > alg6i2_1.out

#python run.py build/src/alg_rd 1000 bin/random_array.bin > alg_rd.out

#python run.py build/src/alg_rd_smem 1000 bin/random_array.bin > alg_smem_rd.out

#python run.py build/src/memcpy 1000 bin/random_array.bin > memcpy_fwd_rev.out

#python run.py build/src/alg3_1d 1000 bin/random_array.bin > alg3_fwd_rev.out

python run.py build/src/alg3_1d_step2 1000 bin/random_array.bin > alg3_step2_fwd_rev.out

python run.py build/src/paper_code 1000 bin/random_array.bin > alg3_paper_fwd_rev.out

#python run.py plr/bin/plr1_fwd_rev 1000 > plr1_fwd_rev.out

#python run.py cub/bin/cubrf 1000 0.719010 > cub_fwd_rev.out

#python run.py build/src/memcpy_double 1000 bin/random_array_double.bin > memcpy_fwd_ps.out

#python run.py build/src/alg3_1d_fwd 1000 bin/random_array_double.bin > alg3_fwd_ps.out

#python run.py build/thrustps/thrustps 1000 bin/random_array_double.bin > thrust_fwd_ps.out

#python run.py cub/bin/cubps 1000 bin/random_array_double.bin > cub_fwd_ps.out

#python run.py plr/bin/plrps 1000 bin/random_array_double.bin > plr1_fwd_ps.out

#python run.py sam/testSAM 1000 > sam_fwd_ps.out

#python run.py build/src/alg3_3d 1000 bin/random_array.bin > alg3_3d.out

