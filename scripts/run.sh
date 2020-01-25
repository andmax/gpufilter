#!/bin/bash

set -x

for i in $(seq 1 128);
do
    for j in $(seq 1 1);
    do
        echo -n "$(($i * 64)) $(($i * 64)) "
        ../build/src/$1 $(($i * 64)) $(($i * 64)) 5 $2 $3
    done
done
