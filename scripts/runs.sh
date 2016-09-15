#!/bin/bash

set -x

rm -f $2 $2-tmp

sh run.sh $1 > $2-tmp

python resavg.py $2-tmp $2
