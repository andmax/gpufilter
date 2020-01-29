#!/bin/bash

set -x

for a in $(seq 3 6); do
    for r in $(seq 1 5); do
        echo alg${a}_${r}
        sh runs.sh alg${a}_${r} results/alg${a}_${r} 0 0
    done
done
