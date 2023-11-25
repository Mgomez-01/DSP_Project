# -*- coding: utf-8 -*-
"""
@author: Benjamin Hayes
@perfector: Miguel Gomez
Created on Wed Nov 22 17:35:52 2023
"""

import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from numpy import zeros, append
import numpy as np
import argparse
import sys
# =============================================================================
# Creating an FIR filter
# =============================================================================


def main(N=10000, fmin=3, fmax=7):
    # Filter Characteristics
    # N = 10000      # Length of Filter
    # fmin = 3    # Minimum frequency
    # fmax = 7    # Maximum frequency

    # Create Arrays
    H = zeros(N)   # N-point Frequency Response H(w)
    w = zeros(N)   # N-point omega array (Only used for plotting against H(w) )

    # Create FIR Filter in Frequency Domain
    K = -1  # Integer used to keep track of position in array
    K = np.arange(N)
    k = np.arange(-int(N/2), int(N/2))
    w = k / N

    # Implementing the square-wave style filter
    H = np.where((np.abs(k) >= fmin) & (np.abs(k) <= fmax), 1, 0)
    # Convert Frequency domain H(w) to time domain to obtain FIR
    h = fftshift(fft(fftshift(H)))
    # Create position array from 0 to N-1 for plotting
    pos = np.arange(N)

    # =============================================================================
    # Padding
    #
    # This section is just for demonstrating the frequency response of the FIR
    # filter we just created.
    # =============================================================================

    # Add Padding
    P = 9*N
    NP = N+P

    # Create Padded Arrays
    h_pad = append(h, zeros(P))
    H_pad = fftshift(fft(h_pad))/N

    w_pad = zeros(NP)

    K = -1
    # Assuming NP is defined and w_pad is a numpy array of the appropriate size
    k = np.arange(-NP/2, NP/2)  # Creates an array from -NP/2 to NP/2-1
    w_pad = k / NP

    # =============================================================================
    # Implement Hamming Window
    # =============================================================================
    # Assuming N is defined and h is a numpy array of length N
    N = len(h)  # or some predefined value
    n = np.arange(N)  # creates an array [0, 1, ..., N-1]
    h_ham = h * 0.5 * (1 + np.cos(2 * np.pi * (n - N / 2) / N))

    h_ham_pad = append(h_ham, zeros(P))
    H_ham_pad = fftshift(fft(h_ham_pad))/N


    # MatPlotLib plotting
    fig1 = plt.figure()            # Copy pasta plot template from stack overflow
    plot1 = fig1.add_subplot(211)   # IDK what this shit does but it works
    plt.scatter(w,  H.real,  c='b',  s=150)
    plt.plot(w_pad,  abs(H_pad),  'r')
    plt.plot(w_pad,  abs(H_ham_pad),  'black')
    # plt.vlines(w,  0,  H,  'b')
    plt.xlabel('Frequency (Ratio of Fs)', {'size': 15, 'weight': 'bold'})
    plt.ylabel('Magnitude', {'size': 15, 'weight': 'bold'})
    plt.title('Frequency Response of FIR Filter', {'size': 15, 'weight': 'bold'})
    plt.legend(['Ideal', 'Actual', 'Hamming'],  prop={'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    fig1.set_size_inches(22, 16)
    plt.ticklabel_format(useOffset=False) # Useful to make plots easier to read

    plot2 = fig1.add_subplot(212)   # IDK what this shit does but it works
    plt.vlines(pos,  0,  h.imag,  'r')
    plt.vlines(pos,  0,  h.real,  'b')
    plt.scatter(pos,  h.real,  c='b',  s=150)
    plt.scatter(pos,  h.imag,  c='r',  s=150)
    plt.xlabel('Position', {'size': 15, 'weight': 'bold'})
    plt.ylabel('Value (Unscaled)', {'size': 15, 'weight': 'bold'})
    plt.title('Time Domain FIR Filter', {'size': 15, 'weight': 'bold'})
    plt.legend(['Real',  'Imag'],  prop={'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    fig1.set_size_inches(22, 16)
    plt.ticklabel_format(useOffset=False) # Useful to make plots easier to read
    plt.show(block=False)
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
