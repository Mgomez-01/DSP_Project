# =============================================================================
# Example for how to use the FIR_filter class to create a band-pass filter
#
# Written by Benjamin Hayes
# =============================================================================

from util.FIR_filter import FIRFilter
import numpy as np
import matplotlib.pyplot as plt



L = 50      # Length of filter
fmin = 0.15 # Minimum frequency to pass (Hz)
fmax = 0.35 # Maximum frequency to pass (Hz)
padFac = 9  # Padding factor
fs = 1      # Sampling Rate (Hz)

# Create FIR filter object f1
f1 = FIRFilter(L,fmin,fmax,padFac,fs)
f1.plot_filter1()   # Plot the filter