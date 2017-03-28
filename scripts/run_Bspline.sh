#!/bin/bash

set -x

sh runs.sh alg6_1 results/alg6bB_1 1 1
sh runs.sh alg6_1 results/alg6cB_1 1 0
sh runs.sh alg6_1 results/alg6pB_1 2 0
sh runs.sh alg6_1 results/alg6eB_1 3 0
sh runs.sh alg5varc results/alg5vB_1 1
sh runs.sh alg5orig_1 results/alg5oB_1 0 0
