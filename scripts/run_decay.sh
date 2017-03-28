#!/bin/bash

set -x

#sh run6b_1.sh results/alg6bb_1
#sh run5vb.sh results/alg5vb_1

python Btob.py results/alg6cB_1 results/alg6cb_1
python Btob.py results/alg6pB_1 results/alg6pb_1
python Btob.py results/alg6eB_1 results/alg6eb_1
