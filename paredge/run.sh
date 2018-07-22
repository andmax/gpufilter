#!/bin/bash

for i in $(seq 1 64);
do
    echo -n "$(($i * 64)) $(($i * 64)) "
    $1 -width=$(($i * 64)) -height=$(($i * 64))
    echo
done
