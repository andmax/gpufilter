#!/bin/bash

set -x

rm -f $2 $2-tmp

sh run.sh $1 $3 $4 > $2-tmp

python resavg.py $2-tmp $2

rm -f $2-tmp

