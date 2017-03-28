#!/usr/bin/env python

import sys
import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: {} in_B out_b'.format(sys.argv[0]))
        sys.exit(1)
    print('Reading {}'.format(sys.argv[1]))
    df_in = pd.read_csv(sys.argv[1], sep=' ', header=None)
    v = df_in[df_in[0] == 2048][2].values[0]
    df_out = pd.DataFrame({0: [1, 32], 1: [v, v]})
    print('Writing {}'.format(sys.argv[2]))
    df_out.to_csv(sys.argv[2], sep=' ', header=False, index=False)
    print 'Done!'
    sys.exit(0)
