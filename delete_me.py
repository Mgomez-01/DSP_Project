# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:13:28 2023

@author: benja
"""

from util.OctaveBandFilt import OctaveBandFilter, ourOctaveBandFilter
from util.FIR_filter import FIRFilter
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from numpy import sin,cos,e,pi
from numpy.fft import fftshift, fft, ifft


L = 100
f1 = FIRFilter(L,0.11,0.14,9,1)
# f1 = FIRFilter(L,0.11,0.3,9,1)
f1.plot_filter2()

N = 201
x = [0]*N
pos = [0]*N
omega = [0]*N
for n in range(N):
    x[n] = 2 + 2*cos(2*pi*n*0.05 + pi/3) + cos(2*pi*n*0.125 - pi/3)
    pos[n] = n
    omega[n] = (n-int(N/2))/N

X = fftshift(fft(fftshift(x)))/N
# Y = X*f1.H_ham_pad
# y = fftshift(ifft(fftshift(Y)))*N


y2 = np.convolve(x,f1.h_ham)/L
pos2 = [0]*len(y2)
for n in range(len(y2)):
    pos2[n] = n

# y2 = y2[50:225]
# pos2 = pos2[50:225]
    
Y2 = fftshift(fft(fftshift(y2)))/len(y2)
N2 = len(Y2)
w2 = [0]*N2
for n in range(N2):
    w2[n] = (n-int(N2/2))/N2

font = 30

# MatPlotLib plotting
fig = plt.figure(figsize=(22, 16))

# Frequency Response Plot
ax1 = fig.add_subplot(111)
ax1.scatter(pos, x, c='b', s=150)
ax1.plot(pos, x, 'b', linewidth=5)
# ax1.scatter(pos, y, c='r', s=150)
# ax1.plot(pos, y, 'r', linewidth=5)
ax1.scatter(pos2, y2, c='g', s=150)
ax1.plot(pos2, y2, 'g', linewidth=5)
# ax1.scatter(pos, y, c='r', s=150)
# ax1.plot(pos, y, 'r', linewidth=5)
ax1.set_xlabel('Position', fontsize=font, fontweight='bold')
ax1.set_ylabel('Value (Unscaled)', fontsize=font, fontweight='bold')
ax1.set_title('Frequency Response of FIR Filter', fontsize=font+10, fontweight='bold')
# ax1.legend(['Original Points', 'Rectangle Window', 'Hamming Window'], prop={'size': font})
ax1.tick_params(axis='both', labelsize=font)
ax1.grid(True)

plt.show(block=False)

# MatPlotLib plotting
fig = plt.figure(figsize=(22, 16))
# Time Domain Plot
ax2 = fig.add_subplot(111)
ax2.plot(f1.w_pad, abs(f1.H_ham_pad), 'black')
ax2.plot(omega,abs(X),'blue')
ax2.plot(w2, abs(Y2), 'red')
# ax2.vlines(self.pos, 0, self.h.imag, 'r')
# ax2.scatter(self.pos, self.h_ham.real, c='b', s=150)
# ax2.scatter(self.pos, self.h.imag, c='r', s=150)
ax2.set_xlabel('Position', fontsize=font, fontweight='bold')
ax2.set_ylabel('Value (Unscaled)', fontsize=font, fontweight='bold')
ax2.set_title('Time Domain FIR Filter', fontsize=font+10, fontweight='bold')
# ax2.legend(['Real', 'Imag'], prop={'size': 15})
ax2.tick_params(axis='both', labelsize=font)
ax2.grid(True)

plt.show(block=False)


