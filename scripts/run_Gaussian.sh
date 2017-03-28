#!/bin/bash

set -x

sh run6b_3.sh results/alg6bG_3
sh runs.sh alg6_3 results/alg6cG_3 1 0
sh runs.sh alg6_3 results/alg6pG_3 2 0
sh runs.sh alg6_3 results/alg6eG_3 3 0
sh run_gauss_fft.sh results/gauss_fft 1
sh run_gauss_dir.sh results/gauss_dir 1
sh runs.sh alg5f4 results/alg5f4_3
