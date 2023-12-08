import numpy as np
from numpy import zeros, append
from numpy.fft import fftshift, fft
import matplotlib.pyplot as plt


class FIRFilter:
    def __init__(self, N=10000, fmin=3, fmax=7, padding_factor=9, fs=8000, passing=True):
        self.N = N
        self.padding_factor = padding_factor
        self.fs = fs  # Sampling rate
        self.H = zeros(N)
        self.w = zeros(N)
        self.pos = np.arange(N)
        self.fmin = fmin*self.N/self.fs
        self.fmax = fmax*self.N/self.fs
        self.passing = passing
        
        self.h = None
        self.h_pad = None
        self.H_pad = None
        self.w_pad = None
        self.h_ham = None
        self.H_ham_pad = None
        
        self.create_filter()
        self.apply_padding()
        self.apply_hamming_window()


    def create_filter(self):
        k = np.arange(-int(self.N/2), int(self.N/2))
        self.w = k * self.fs / self.N  # Adjusted to use actual frequency values
        if self.passing:
            self.H = np.where((np.abs(k) >= self.fmin) & (np.abs(k) <= self.fmax), 1, 0)
        else:
            self.H = np.where((np.abs(k) >= self.fmin) & (np.abs(k) <= self.fmax), 0, 1)

        self.h = fftshift(fft(fftshift(self.H)))

    def apply_padding(self):
        NP = self.N + self.padding_factor * self.N
        self.h_pad = append(self.h, zeros(self.padding_factor * self.N))
        self.H_pad = fftshift(fft(self.h_pad)) / self.N
        k = np.arange(-NP/2, NP/2)
        self.w_pad = k * self.fs / NP

    def apply_hamming_window(self):
        self.h_ham = self.h * 0.5 * (1 + np.cos(2 * np.pi * (self.pos - self.N / 2) / self.N))
        self.h_ham_pad = append(self.h_ham, zeros(self.padding_factor * self.N))
        self.H_ham_pad = fftshift(fft(self.h_ham_pad)) / self.N

    def process(self, input_data):
        return np.convolve(input_data, self.h_ham)/self.N

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
        # ax2.vlines(self.pos, 0, self.h.imag, 'r')
        ax2.scatter(self.pos, self.h.real, c='b', s=150)
        # ax2.scatter(self.pos, self.h.imag, c='r', s=150)
        ax2.set_xlabel('Position', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Value (Unscaled)', fontsize=10, fontweight='bold')
        ax2.set_title('Time Domain FIR Filter', fontsize=10, fontweight='bold')
        # ax2.legend(['Real', 'Imag'], prop={'size': 15})
        ax2.tick_params(axis='both', labelsize=15)
        ax2.grid(True)

        plt.show(block=False)

