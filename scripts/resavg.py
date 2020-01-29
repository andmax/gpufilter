#!/usr/bin/env python

import sys
import pandas as pd

if __name__ == "__main__":
    gbc = [0, 1] # group-by columns (default 0 and 1)
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print('Usage: {} res.txt resavg.txt [group-by cols={}]'.format(
            sys.argv[0], gbc))
        sys.exit(1)
    if len(sys.argv) == 4: gbc = [int(sys.argv[3])]
    if len(sys.argv) == 5: gbc = [int(sys.argv[3]), int(sys.argv[4])]
    print('Reading {}'.format(sys.argv[1]))
    df = pd.read_csv(sys.argv[1], sep=' ', header=None)
    print('Writing {}'.format(sys.argv[2]))
    df.groupby(gbc).mean().to_csv(sys.argv[2], sep=' ', header=None)
    print('Done!')
    sys.exit(0)
