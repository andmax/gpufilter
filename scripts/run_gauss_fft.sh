#!/bin/bash

set -x

rm -f $1 $1-tmp
wh=64

for i in $(seq 1 8);
do
    for j in $(seq 1 8);
    do
        echo -n "$wh $wh" >> $1-tmp
        ../build/src/gauss_fft $wh $wh $2 >> $1-tmp
        echo >> $1-tmp
    done
    wh=$(($wh*2))
done

python resavg.py $1-tmp $1

rm -f $1-tmp
