#!/usr/bin/env python

from __future__ import print_function

import sys
import subprocess


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: {} cmd-and-args-to-run'.format(sys.argv[0]))
        sys.exit(1)
    
    cmd = sys.argv[1:]

    if 'gpufilter' in cmd[0]:
        dim = '2D'
    elif '3d' in cmd[0]:
        dim = '3D'
    else:
        dim = '1D'

    if dim == '1D':
        cmd = [cmd[0], None] + cmd[1:]
    
        for i in range(14, 31): # max is 31 or (2**30)
            cmd[1] = '{}'.format(2**i)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output, _ = process.communicate()
            output = output.decode('utf-8')
            print('{} {}'.format(cmd[1], output[:-1]))
            sys.stdout.flush()

    elif dim == '2D':
        cmd = [cmd[0], None, None] + cmd[1:]

        if 'alg3' in cmd[0]: # 1D from 2D
            l_hw = [ str(hw) for hw in range(64, 8192+1, 64) ]
        else: # 2D in fact
            l_hw = [ str(2**hw) for hw in range(6, 14) ]

        for hw in l_hw:
            cmd[1:3] = [hw, hw]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            output, _ = process.communicate()
            output = output.decode('utf-8')
            print('{} {} {}'.format(cmd[1], cmd[2], output[:-1]))
            sys.stdout.flush()

    elif dim == '3D':
        cmd = [cmd[0], None, None, None] + cmd[1:]
    
        for d in range(3, 9):
            for h in range(9, 13):
                for w in range(9, 13):
                    cmd[1:4] = [str(2**w), str(2**h), str(2**d)]
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                    output, _ = process.communicate()
                    output = output.decode('utf-8')
                    print('{} {} {} {}'.format(
                        cmd[1], cmd[2], cmd[3], output[:-1]))
                    sys.stdout.flush()

