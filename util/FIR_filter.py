import numpy as np
from numpy import zeros, append
from numpy.fft import fftshift, fft
import matplotlib.pyplot as plt


class FIRFilter:
    def __init__(self, N=100, fmin=1000, fmax=2000, padding_factor=9, fs=8000):
        self.N = N
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self._kmax = self.fmax/fs*N
        self._kmin = self.fmin/fs*N
        self.padding_factor = padding_factor

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

    def create_filter(self):
        k = np.arange(-int(self.N/2), int(self.N/2))
        self.w = k / self.N * self.fs
        self.H = np.where((np.abs(k) >= self._kmin) & (np.abs(k) <= self._kmax), 1, 0)
        self.h = fftshift(fft(fftshift(self.H)))

    def apply_padding(self):
        NP = self.N + self.padding_factor * self.N
        self.h_pad = append(self.h, zeros(self.padding_factor * self.N))
        self.H_pad = fftshift(fft(self.h_pad)) / self.N
        k = np.arange(-NP/2, NP/2)
        self.w_pad = k / NP * self.fs

    def apply_hamming_window(self):
        self.h_ham = self.h * 0.5 * (1 + np.cos(2 * np.pi * (self.pos - self.N / 2) / self.N))
# =============================================================================
#         Below is the hamming function from the assignment, It doesn't quite work so we're using the one above,
#         The line above also works for a wide bandwidth passband instead of at a single frequency point "wc"
# =============================================================================
        # self.h_ham = self.h*(0.54-0.46*np.cos(2*np.pi*self.pos/(self.N-1)))*np.cos((self.fmin+self.fmax)/2*(self.pos-(self.N-1)/2))
        self.h_ham_pad = append(self.h_ham, zeros(self.padding_factor * self.N))
        self.H_ham_pad = fftshift(fft(self.h_ham_pad)) / self.N

    def plot_filter(self):
        # MatPlotLib plotting
        fig = plt.figure(figsize=(22, 16))

        # Frequency Response Plot
        ax1 = fig.add_subplot(211)
        ax1.scatter(self.w, self.H.real, c='b', s=150)
        # ax1.plot(self.w_pad, abs(self.H_pad), 'r')
        # ax1.plot(self.w_pad, abs(self.H_ham_pad), 'black')
        ax1.set_xlabel('Frequency (Hz)', fontsize=15, fontweight='bold')
        ax1.set_ylabel('Magnitude', fontsize=15, fontweight='bold')
        ax1.set_title('Frequency Response of FIR Filter', fontsize=15, fontweight='bold')
        ax1.legend(['Ideal', 'Actual', 'Hamming'], prop={'size': 15})
        ax1.tick_params(axis='both', labelsize=15)
        ax1.grid(True)

        # Time Domain Plot
        ax2 = fig.add_subplot(212)
        ax2.vlines(self.pos, 0, self.h.real, 'b')
        # ax2.vlines(self.pos, 0, self.h.imag, 'r')
        ax2.scatter(self.pos, self.h.real, c='b', s=150)
        # ax2.scatter(self.pos, self.h.imag, c='r', s=150)
        ax2.set_xlabel('Position', fontsize=15, fontweight='bold')
        ax2.set_ylabel('Value (Unscaled)', fontsize=15, fontweight='bold')
        ax2.set_title('Time Domain FIR Filter', fontsize=15, fontweight='bold')
        # ax2.legend(['Real', 'Imag'], prop={'size': 15})
        ax2.tick_params(axis='both', labelsize=15)
        ax2.grid(True)

        plt.show(block=False)
        
    def plot_filter2(self):
        font = 30
        
        # MatPlotLib plotting
        fig = plt.figure(figsize=(22, 16))

        # Frequency Response Plot
        ax1 = fig.add_subplot(111)
        ax1.scatter(self.w, self.H.real, c='b', s=150)
        # ax1.plot(self.w, self.H.real, c='b')
        ax1.plot(self.w_pad, abs(self.H_pad), 'r')
        ax1.plot(self.w_pad, abs(self.H_ham_pad), 'black', linewidth=5)
        ax1.set_xlabel('Frequency (Hz)', fontsize=font, fontweight='bold')
        ax1.set_ylabel('Magnitude', fontsize=font, fontweight='bold')
        ax1.set_title('Frequency Response of FIR Filter', fontsize=font+10, fontweight='bold')
        ax1.legend(['Original Points', 'Rectangle Window', 'Hamming Window'], prop={'size': font})
        ax1.tick_params(axis='both', labelsize=font)
        ax1.grid(True)
        
        plt.show(block=False)

        # MatPlotLib plotting
        fig = plt.figure(figsize=(22, 16))
        # Time Domain Plot
        ax2 = fig.add_subplot(111)
        ax2.vlines(self.pos, 0, self.h_ham.real, 'b')
        # ax2.vlines(self.pos, 0, self.h.imag, 'r')
        ax2.scatter(self.pos, self.h_ham.real, c='b', s=150)
        # ax2.scatter(self.pos, self.h.imag, c='r', s=150)
        ax2.set_xlabel('Position', fontsize=font, fontweight='bold')
        ax2.set_ylabel('Value (Unscaled)', fontsize=font, fontweight='bold')
        ax2.set_title('Time Domain FIR Filter', fontsize=font+10, fontweight='bold')
        # ax2.legend(['Real', 'Imag'], prop={'size': 15})
        ax2.tick_params(axis='both', labelsize=font)
        ax2.grid(True)

        plt.show(block=False)
# Example of using the FIRFilter class
# fir_filter = FIRFilter(N=10000, fmin=3, fmax=7, padding_factor=9)
