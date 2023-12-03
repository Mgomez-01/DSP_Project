import numpy as np
from numpy import zeros, append
from numpy.fft import fftshift, fft
import matplotlib.pyplot as plt


class FIRFilter:
    def __init__(self, N=10000, fmin=3, fmax=7, padding_factor=9, fs=8000):
        self.N = N
        self.fmin = fmin
        self.fmax = fmax
        self.padding_factor = padding_factor
        self.fs = fs  # Sampling rate
        
        self.H = zeros(N)
        self.w = zeros(N)
        self.pos = np.arange(N)
        self.h = None
        self.h_pad = None
        self.H_pad = None
        self.w_pad = None
        self.h_ham = None
        self.H_ham_pad = None

        self.create_filter()
        self.apply_padding()
        self.apply_hamming_window()
        self.buffer_length = 1024
        self.buffer = np.zeros(self.buffer_length + len(self.h_ham) - 1)


    def create_filter(self):
        k = np.arange(-int(self.N/2), int(self.N/2))
        self.w = k * self.fs / self.N  # Adjusted to use actual frequency values
        self.H = np.where((np.abs(k) >= self.fmin) & (np.abs(k) <= self.fmax), 1, 0)
        self.h = fftshift(fft(fftshift(self.H)))

    def apply_padding(self):
        NP = self.N + self.padding_factor * self.N
        self.h_pad = append(self.h, zeros(self.padding_factor * self.N))
        self.H_pad = fftshift(fft(self.h_pad)) / self.N
        k = np.arange(-NP/2, NP/2)
        self.w_pad = k * self.fs / NP

    def apply_hamming_window(self):
        self.h_ham = self.h * 0.5 * (1 + np.cos(2 * np.pi * (self.pos - self.N / 2) / self.N))
    # =============================================================================
    #         Below is the hamming function from the assignment, It doesn't quite work so we're using the one above,
    #         The line above also works for a wide bandwidth passband instead of at a single frequency point "wc"
    # =============================================================================
        # self.h_ham = self.h*(0.54-0.46*np.cos(2*np.pi*self.pos/(self.N-1)))*np.cos((self.fmin+self.fmax)/2*(self.pos-(self.N-1)/2))
        self.h_ham_pad = append(self.h_ham, zeros(self.padding_factor * self.N))
        self.H_ham_pad = fftshift(fft(self.h_ham_pad)) / self.N

    def process(self, input_data):
        # Append input data to buffer
        self.buffer_length = len(input_data)
        self.buffer = np.zeros(self.buffer_length + len(self.h_ham) - 1)
        self.buffer[:-len(input_data)] = self.buffer[len(input_data):]
        self.buffer[-len(input_data):] = input_data

        # Perform FFT-based convolution
        input_fft = np.fft.fft(self.buffer)
        filter_fft = np.fft.fft(self.h_ham, n=len(self.buffer))
        result_fft = input_fft * filter_fft

        # Inverse FFT to get the time domain signal
        result = np.fft.ifft(result_fft)

        # Return the relevant part of the result
        return result[-len(input_data):].real  # Returning only the real part

    def plot_filter(self, fig, row, col, pos):
        # Frequency Response Plot
        ax1 = fig.add_subplot(row, col, pos)
        # ax1.scatter(self.w, self.H.real, c='b', s=150)
        # ax1.plot(self.w_pad, abs(self.H_pad), 'r')
        ax1.plot(self.w_pad, abs(self.H_ham_pad), 'black')
        ax1.set_xlim(0, .5)
        ax1.set_title(f'Frequency Response of Octave Filter {pos}', fontsize=5, fontweight='bold')
        if pos == row-1:            
            ax1.legend(['Hamming'], prop={'size': 5})
            ax1.set_xlabel('Frequency (Ratio of Fs)', fontsize=5, fontweight='bold')
            ax1.set_ylabel('Magnitude', fontsize=5, fontweight='bold')
        ax1.grid(True)
