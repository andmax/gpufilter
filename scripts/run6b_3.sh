#!/bin/bash

set -x

rm -f $1 $1-tmp

for i in $(seq 1 128);
do
    for j in $(seq 1 8);
    do
        echo -n "$(($i * 64)) $(($i * 64)) " >> $1-tmp
        ../build/src/alg6_3 $(($i * 64)) $(($i * 64)) 1000 1 $i >> $1-tmp
    done
done

python resavg.py $1-tmp $1

rm -f $1-tmp

