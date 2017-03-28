#!/bin/bash

set -x

rm -f $2 $2-tmp

for i in $(seq 1 128);
do
    for j in $(seq 1 8);
    do
        echo -n "$(($i * 64)) $(($i * 64))" >> $2-tmp
        ../build/src/$1 $(($i * 64)) $(($i * 64)) $3 >> $2-tmp
        echo >> $2-tmp
    done
done

python resavg.py $2-tmp $2

rm -f $2-tmp

