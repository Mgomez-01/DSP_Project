# -*- coding: utf-8 -*-
"""
@author: Benjamin Hayes

Created on Wed Nov 22 17:35:52 2023
"""

import matplotlib.pyplot as plt
from numpy.fft import fft,fftshift
from numpy import zeros,append,cos,pi

# =============================================================================
# Creating an FIR filter
# =============================================================================

# Filter Characteristics
N = 32      # Length of Filter
fmin = 3    # Minimum frequency
fmax = 7    # Maximum frequency

# Create Arrays
H = zeros(N)   # N-point Frequency Response H(w)
w = zeros(N)   # N-point omega array (Only used for plotting against H(w) )

# Create FIR Filter in Frequency Domain
K = -1  # Integer used to keep track of position in array
for k in range(int(-N/2),int(N/2)):
    K = K + 1   # Increment the array posiiton-counter
    w[K] = k/N  # Save the frequency to omega array
    
    # Implement simple square-wave style filter
    if abs(k) >= fmin and abs(k) <= fmax:
        H[K] = 1
    else:
        H[K] = 0

# Convert Frequency domain H(w) to time domain to obtain FIR
h = fftshift(fft(fftshift(H)))
# Create position array from 0 to N-1 for plotting
pos = [0]*N
for n in range(N):
    pos[n] = n
        
# =============================================================================
# Padding
#
# This section is just for demonstrating the frequency response of the FIR
# filter we just created.
# =============================================================================

# Add Padding
P = 1000-N
NP = N+P

# Create Padded Arrays
h_pad = append(h,zeros(P))
H_pad = fftshift(fft(h_pad))/N

w_pad = zeros(NP)

K = -1
for k in range(int(-NP/2),int(NP/2)):
    K = K + 1
    w_pad[K] = k/NP
    
# =============================================================================
# Implement Hamming Window
# =============================================================================

h_ham = zeros(N)
for n in range(N):
    h_ham[n] = h[n]*0.5*(1 + cos(2*pi*(n-N/2)/N))

h_ham_pad = append(h_ham,zeros(P))
H_ham_pad = fftshift(fft(h_ham_pad))/N


# MatPlotLib plotting
fig1 = plt.figure()            # Copy pasta plot template from stack overflow
plot1 = fig1.add_subplot(111)   # IDK what this shit does but it works
plt.scatter(w, H.real, c='b', s=150)
plt.plot(w_pad, abs(H_pad), 'r')
plt.plot(w_pad, abs(H_ham_pad), 'black')
# plt.vlines(w, 0, H, 'b')
plt.xlabel('Frequency (Ratio of Fs)',{'size':30,'weight':'bold'})
plt.ylabel('Magnitude',{'size':30,'weight':'bold'})
plt.title('Frequency Response of FIR Filter',{'size':30,'weight':'bold'})
plt.legend(['Ideal','Actual','Hamming'], prop={'size':25})
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
fig1.set_size_inches(22,16)
plt.ticklabel_format(useOffset=False) # Useful to make plots easier to read

# MatPlotLib plotting
fig1 = plt.figure()            # Copy pasta plot template from stack overflow
plot2 = fig1.add_subplot(111)   # IDK what this shit does but it works
plt.vlines(pos, 0, h.real, 'b')
plt.vlines(pos, 0, h.imag, 'r')
plt.scatter(pos, h.real, c='b', s=150)
plt.scatter(pos, h.imag, c='r', s=150)
plt.xlabel('Position',{'size':30,'weight':'bold'})
plt.ylabel('Value (Unscaled)',{'size':30,'weight':'bold'})
plt.title('Time Domain FIR Filter',{'size':30,'weight':'bold'})
plt.legend(['Real', 'Imag'], prop={'size':25})
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
fig1.set_size_inches(22,16)
plt.ticklabel_format(useOffset=False) # Useful to make plots easier to read