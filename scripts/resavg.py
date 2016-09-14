#!/usr/bin/env python

import sys
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print 'Usage:', sys.argv[0], 'res.txt resavg.txt'
        sys.exit(1)
    print 'Reading', sys.argv[1]
    df = pd.read_csv(sys.argv[1], sep=' ', header=None)
    df.drop(3, axis=1, inplace=True) # drop maximum-error column
    df.drop(4, axis=1, inplace=True) # drop maximum-relative-error column
    print 'Writing', sys.argv[2]
    df.groupby([0, 1]).mean().to_csv(sys.argv[2], sep=' ',header=None)
    print 'Done!'
    sys.exit(0)
