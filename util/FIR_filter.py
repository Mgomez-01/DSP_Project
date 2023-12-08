# =============================================================================
# FIR_Filter Class
#
# This class is used to create a bandpass filter object
# The filter object constructs an FIR filter from the given inputs:
#   
#   N               --> Number of points constructing the FIR filter
#   fmin            --> Minimum frequency to be passed
#   fmax            --> Maximum frequency of bandpass region
#   padding_factor  --> Determines how many zeros are appended to the time
#                       domain FIR filter when plotting the frequency response
#   fs              --> Sampling frequency
#   passing         --> Boolean value determines if filter will be band pass or
#                       a band-reject filter
# =============================================================================

import numpy as np
from numpy import zeros, append
from numpy.fft import fftshift, fft
import matplotlib.pyplot as plt

class FIRFilter:
    def __init__(self, N=10000, fmin=3, fmax=7, padding_factor=9, fs=8000, passing=True):
        self.N = N
        self.padding_factor = padding_factor
        self.fs = fs  # Sampling rate
        self.H = zeros(N)   # H[k], The specified frequency domain used to create h[n]
        self.w = zeros(N)   # frequency array, used for plotting H[k] in the frequency domain
        self.pos = np.arange(N)     # array of "n" values, used for plotting h[n] in the time domain
        self.fmin = fmin*self.N/self.fs     # Minimum frequency of pass-band region
        self.fmax = fmax*self.N/self.fs     # Maximum frequency of pass-band region
        self.passing = passing  # Boolean value, turns the filter from band-pass to band-reject if false
        
        self.h = None   # array for storing time domain FIR filter
        self.h_pad = None   # Time domain FIR filter padded with extra zeros, used to plot FIR frequency response
        self.H_pad = None   # Frequency doman of h_pad, the padded time domain FIR filter
        self.w_pad = None   # Frequency array for plotting against the padded FIR filter frequency response
        self.h_ham = None   # FIR filter h[n], but modified with a hamming window
        self.H_ham_pad = None   # Frequency response of the FIR filter modified with a hamming window
        
        self.create_filter()
        self.apply_padding()
        self.apply_hamming_window()

    # Creates the FIR filter by specifying specific values in the frequency
    #   domain, if "k" is within the desired passband, H[k]=1, else, H[k]=0
    #   After H[k] is specified, the FIR filter is obtained by taking the fft
    #   of H[k]. Both H[k] and h[n] are of length N.
    def create_filter(self):
        k = np.arange(-int(self.N/2), int(self.N/2))    # Create array of k values of length N
        self.w = k * self.fs / self.N  # Create frequency array from -Fs/2 to Fs/2
        # If "k" is within the passband, set H[k] to 1, else 0
        if self.passing:
            self.H = np.where((np.abs(k) >= self.fmin) & (np.abs(k) <= self.fmax), 1, 0)
        # If passing is false, do the opposite, this turns the bandpass filter to band-reject
        else:
            self.H = np.where((np.abs(k) >= self.fmin) & (np.abs(k) <= self.fmax), 0, 1)
        # Take the FFT of the specified frequency response H[k] to create h[n], the FIR filter in the time domain
        self.h = fftshift(fft(fftshift(self.H)))

    # The FIR fitler is only N points long, therefore, to find the true frequency response, we need
    #   to find the effect of convoluting the FIR filter with a signal much longer than N points.
    #   To do this, we can append many zeros to the time domain FIR filter h[n], and take the fft of
    #   that padded filter to find the true frequency response. NP is the lgnth of the padded FIR
    #
    #   NP = N + padding_factor*N
    #
    #   If padding_factor = 9, the padded signal will be length 10*N
    def apply_padding(self):
        NP = self.N + self.padding_factor * self.N  # Find length of padded signal
        self.h_pad = append(self.h, zeros(self.padding_factor * self.N))  # Append zeros
        self.H_pad = fftshift(fft(self.h_pad)) / self.N  # Take FFT of padded time domain FIR filter
        k = np.arange(-NP/2, NP/2)  # Create frequency array to plot padded frequency response against
        self.w_pad = k * self.fs / NP # Scale frequency response to the sample rate

    # This function applies a hamming window to h[n] by multiplying h[n] by the 
    #   hamming window funciton in the time domain
    def apply_hamming_window(self):
        # Apply hamming window
        self.h_ham = self.h * 0.5 * (1 + np.cos(2 * np.pi * (self.pos - self.N / 2) / self.N))
        # Add padding for plotting frequency response
        self.h_ham_pad = append(self.h_ham, zeros(self.padding_factor * self.N))
        # Find frequency response by taking fft of padded hamming windowed FIR filter
        self.H_ham_pad = fftshift(fft(self.h_ham_pad)) / self.N

    # This functions applies the created FIR filter to an input signal.
    #   This is just a convolution but scales the output by N, the length of the FIR
    #   filter, to maintain proper scaling
    def process(self, input_data):
        return np.convolve(input_data, self.h_ham)/self.N

    # One of two plotting fucntions, for a general view of the performance of a
    #   created filter, I would personally use the other plotting function
    def plot_filter(self, fig, row, col, pos):
        # Frequency Response Plot
        ax1 = fig.add_subplot(row, col, pos)
        # ax1.scatter(self.w, self.H.real, c='b', s=150)
        # ax1.plot(self.w_pad, abs(self.H_pad), 'r')
        ax1.plot(self.w_pad, abs(self.H_ham_pad), 'black')
        #ax1.set_xlim(0, .5)
        ax1.set_title(f'Frequency Response of Octave Filter {pos}', fontsize=10, fontweight='bold')
        if pos == row-1:
            # ax1.legend(['Hamming'], prop={'size': 5})
            ax1.set_xlabel('Frequency [Hz]', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Magnitude', fontsize=10, fontweight='bold')
        ax1.set_xlim([0, self.fs/2])
        ax1.grid(True)

    # Plotting function, Use this to check the filter created performs as intended
    def plot_filter1(self):
        # MatPlotLib plotting
        fig = plt.figure(figsize=(22, 16))

        # Frequency Response Plot
        ax1 = fig.add_subplot(211)
        ax1.scatter(self.w, self.H.real, c='b', s=150)
        ax1.plot(self.w_pad, abs(self.H_pad), 'r')
        ax1.plot(self.w_pad, abs(self.H_ham_pad), 'black')
        ax1.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Magnitude', fontsize=10, fontweight='bold')
        ax1.set_title('Frequency Response of FIR Filter', fontsize=10, fontweight='bold')
        ax1.legend(['Ideal', 'Actual', 'Hamming'], prop={'size': 15})
        ax1.tick_params(axis='both', labelsize=15)
        ax1.grid(True)

        # Time Domain Plot
        ax2 = fig.add_subplot(212)
        ax2.vlines(self.pos, 0, self.h.real, 'b')
        ax2.scatter(self.pos, self.h.real, c='b', s=150)
        # ax2.scatter(self.pos, self.h.imag, c='r', s=150)
        ax2.set_xlabel('Position', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Value (Unscaled)', fontsize=10, fontweight='bold')
        ax2.set_title('Time Domain FIR Filter', fontsize=10, fontweight='bold')
        # ax2.legend(['Real', 'Imag'], prop={'size': 15})
        ax2.tick_params(axis='both', labelsize=15)
        ax2.grid(True)

        plt.show(block=False)

