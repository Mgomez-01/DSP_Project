import wave
import numpy as np
import pyaudio
from util.FIR_filter import FIRFilter
import matplotlib.pyplot as plt
import time
import sys
from queue import Queue
import threading


def calculate_octave_ranges(base_freq, num_octaves, sample_rate):
    octave_ranges = []
    for i in range(num_octaves+1):
        fmin = base_freq * (2 ** i)  # Minimum frequency for the octave
        fmax = base_freq * (2 ** ((i*12+11)/12))  # Maximum frequency for the octave

        # Normalize the frequencies
        fmin_normalized = fmin
        fmax_normalized = fmax

        octave_ranges.append((fmin_normalized, fmax_normalized))
    return octave_ranges



# Sample rate and base frequency
base_freq = 16.35159783*2  # Starting frequency of the first octave
# Calculate octave ranges
num_octaves = 7



# Replace 'your_file.wav' with the path to your WAV file
filename = sys.argv[1]
# Open the WAV file
with wave.open(filename, 'rb') as wav_file:
    # Extract audio parameters
    n_channels, sampwidth, framerate, n_frames, comptype, compname = wav_file.getparams()
    print(wav_file.getparams())
    
    # Read audio frames
    frames = wav_file.readframes(n_frames)

    # Convert frames to numpy array
    if sampwidth == 1:
        dtype = np.uint8  # 8-bit audio
    elif sampwidth == 2:
        dtype = np.int16  # 16-bit audio
    # For 24-bit and 32-bit, using int32 is more common
    elif sampwidth == 3 or sampwidth == 4:
        dtype = np.int32  # 24-bit or 32-bit audio
    else:
        raise ValueError("Unsupported sample width")

    audio_data = np.frombuffer(frames, dtype=dtype)
    print(audio_data.shape)
    # Correct reshaping for stereo audio
    if n_channels == 2:
        # Reshape and then transpose
        audio_data = np.reshape(audio_data, (-1, n_channels)).T
        print(f"reshaped: {audio_data.shape}")

audio_data = audio_data[1, 0:len(audio_data[0])//2]
sample_rate = framerate
N = 1000
octave_ranges = calculate_octave_ranges(base_freq, num_octaves, sample_rate)
filters = [FIRFilter(N, fmin=fmin, fmax=fmax, padding_factor=1, fs=sample_rate) for fmin, fmax in octave_ranges]

data = {}
for i, filt in enumerate(filters):
    data[i] = filt.process(audio_data)/np.max(audio_data)

# audio_fft = np.fft.fft(audio_data)
# fig1, ax1 = plt.subplots()
# fig1.add_subplot(1+len(filters)//2, 2, 1)
# plt.semilogx(audio_fft)
# ax1.set_xlim([0, sample_rate])
# plt.show(block=False)
# # Now `audio_data` is a numpy array with the WAV file's audio data
# fig1.add_subplot(2 + len(filters)//2, 2, 2)
# plt.semilogx(audio_data, marker='*', linewidth=0.01, markersize=2)
# for i, x in enumerate(data):
#     fig1.add_subplot(2 + len(filters)//2, 2, 2 + i)
#     plt.semilogx(data[i])
# plt.show(block=False)
# input()


# Calculate the FFT and frequencies
audio_fft = np.fft.fft(audio_data)
audio_fft = audio_fft/np.max(audio_fft)
freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)

# Keep only the positive frequencies
pos_mask = freqs >= 0
freqs = freqs[pos_mask]
audio_fft = audio_fft[pos_mask]

# Prepare subplots
n_rows = 1 + len(filters) // 2
fig, axs = plt.subplots(n_rows, 2, figsize=(10, 2 * n_rows))

# Plot FFT
axs[0, 0].plot(freqs, np.abs(audio_fft))
axs[0, 0].set_xlim([0, 10000])  # Usually, we are interested in 20 Hz to Nyquist
axs[0, 0].set_title('FFT of Audio Signal')

# Plot original audio data (if it makes sense)
axs[0, 1].plot(audio_data)
axs[0, 1].set_title('Audio Signal')

# Plot filter outputs
for i, x in enumerate(data):
    row = (2 + i) // 2
    col = (2 + i) % 2
    axs[row, col].plot(data[i])
    axs[row, col].set_title(f'Filter {i+1} Output')

# Adjust layout and show plot
plt.tight_layout()
plt.show()
