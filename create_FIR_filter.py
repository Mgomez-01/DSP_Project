# -*- coding: utf-8 -*-
"""
@author: Benjamin Hayes
@perfector: Miguel Gomez
Created on Wed Nov 22 17:35:52 2023
"""

from util.FIR_filter import FIRFilter
import argparse
import sys
# =============================================================================
# Creating an FIR filter
# =============================================================================


def main(N=10000, fmin=3, fmax=7):
    fir_filter = FIRFilter(N, fmin, fmax, 9)
    fir_filter.plot_filter()
    input("Press [enter] to continue.")


if __name__ == "__main__":
    # print("calling main")
    parser = argparse.ArgumentParser(description='Creates an FIR filter with len N, fmin, and fmax')

    # Add options
    parser.add_argument('-D', '--debug', action='store_true', help='Enable debugging')
    parser.add_argument('-N', '--num', type=int, help='Specify a length for the filter <= 30')
    parser.add_argument('-F', '--freqs', nargs='+', help='Specify a two element list of frequencies for min and max')
    args = parser.parse_args()
    
    if args.num is None:
        print('min N is 32')
        N = 32
    else:
        print(args.num)
        N = args.num
        if args.num < 32:
            N = 32
    if args.freqs is None:
        print('freq min and max are required')
        sys.exit()
    if args.freqs is not [] and len(args.freqs) < 2:
        print('supply a min and a max. must have two')
        sys.exit(1)
    else:
        freqs = [int(a) for a in args.freqs]
        fmax = freqs[1]
        fmin = freqs[0]
        if (fmax - fmin) < 0:
            print('First frequency should be min, max should follow')
            sys.exit(1)
        else:
            main(N, fmin, fmax)

# call like this python3 create_FIR_filter.py -N 100 -F 5 25
