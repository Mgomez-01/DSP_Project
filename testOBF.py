from util.OctaveBandFilt import OctaveBandFilter, ourOctaveBandFilter
from util.FIR_filter import FIRFilter
import numpy as np
import matplotlib.pyplot as plt
from rich import print
import math

fontsize = 10

def calculate_grid_dimensions(n):
    columns = round(math.sqrt(n))
    rows = math.ceil(n / columns)
    return rows, columns

def calculate_octave_ranges(base_freq, num_octaves, sample_rate, N):
    octave_ranges = []
    for i in range(num_octaves+1):
        fmin = base_freq * (2 ** i)  # Minimum frequency for the octave
        if N < 500:
            fmax = base_freq * (2 ** (i+(11/12)))
        else:
            fmax = base_freq * (2 ** (i+1)) # Maximum frequency for the octave
       
        # Normalize the frequencies
        fmin_normalized = fmin
        fmax_normalized = fmax

        octave_ranges.append((fmin_normalized, fmax_normalized))
    return octave_ranges


# Create a set of filters
N = 600
sampling_frequency = 8000  # Sampling frequency in Hz
# Sample rate and base frequency
base_freq = 32.703  # Starting frequency of the first octave
# Calculate octave ranges
num_octaves = 6
# for the filtering
octave_ranges = calculate_octave_ranges(base_freq, num_octaves, sampling_frequency, 2*N)
# for the midpoints
octave_ranges1 = calculate_octave_ranges(base_freq, num_octaves, sampling_frequency, N/2)


filters = [FIRFilter(N, fmin=fmin, fmax=fmax, padding_factor=9, fs=sampling_frequency) for fmin, fmax in octave_ranges]
num_filters = len(filters)
#filters = [FIRFilter(N, fmin=fmin*N, fmax=fmax*N, padding_factor=9) for fmin, fmax in octave_ranges]
print(f"octave ranges: {octave_ranges}")
# Calculate grid size
total_subplots = num_filters
rows, cols = calculate_grid_dimensions(total_subplots)
print(f"rows: {rows}")
print(f"cols: {cols}")
# Create a figure for plotting
fig = plt.figure(figsize=(10, 10))

mids = [(b+a)/2 for a, b in octave_ranges]
mids1 = [(b+a)/2 for a, b in octave_ranges1]
print(f"midpoints_overlap: {mids}")
print(f"midpoints_no_overlap: {mids1}")
# Plot each filter's response
for i, filter in enumerate(filters):
    # Calculate position
    position = i * 2 + 1  # Position for frequency response plot
    x = mids1[i]
    filter.plot_filter(fig, num_filters+1, 1, i+1)
    plt.axvline(x=x, ymin=0, ymax=1, color='r', linestyle='--')
    if i == 0:
        plt.legend(['octave window', 'center'])
    plt.xticks([x])
    plt.yticks([0, .5, 1])
    ax = plt.gca()
    ax.set_xlim([0, filter.fs/2])

plt.tight_layout()
plt.show(block=False)
plt.savefig("images/post_lock_freq_data.png", dpi=200, transparent=False)
plt.close()

num_octaves = 6
octave_ranges = calculate_octave_ranges(base_freq, num_octaves, sampling_frequency, N)
filters = [FIRFilter(N, fmin=fmin, fmax=fmax, padding_factor=9, fs=sampling_frequency) for fmin, fmax in octave_ranges]
num_filters = len(filters)
# Generate the signal 5.2
fs = sampling_frequency
t1 = np.linspace(0, 0.25, int(fs*0.25), endpoint=False)
t2 = np.linspace(0.3, 0.55, int(fs*0.25), endpoint=False)
t3 = np.linspace(0.6, 0.85, int(fs*0.25), endpoint=False)
t_end = np.linspace(0.85, 1, int(fs*0.15), endpoint=False)

x1 = np.cos(2*np.pi*220*t1)
x2 = np.cos(2*np.pi*880*t2)
x3 = np.cos(2*np.pi*440*t3) + np.cos(2*np.pi*1760*t3)

zero_padding = np.zeros(int(fs*0.05))
zeros_end = np.zeros(int(fs*0.15))
x = np.concatenate((x1, zero_padding, x2, zero_padding, x3, zeros_end))
t = np.linspace(0, 1, len(x), endpoint=False)
# Filter and plot the signal
fig = plt.figure(figsize=(10, 10))
plt.title('Filtered Signal with Filters', fontsize=fontsize+10, fontweight='bold')
plt.subplot(len(filters)+1, 1, 1)
plt.plot(t, x)
plt.ylabel('Input Signal x', fontsize=fontsize, fontweight='bold')
ax = plt.gca()
ax.set_xlim([0, 1])

for i, filter in enumerate(filters):
    filtered_x = filter.process(x)
    t = np.linspace(0, 1, len(filtered_x), endpoint=False)
    filtered_sig = filtered_x
    plt.subplot(len(filters)+1, 1, i+2)
    plt.plot(t, np.real(filtered_sig))
    ax = plt.gca()
    ax.set_ylim([-1.100, 1.100])
    ax.set_xlim([0, 1])
    plt.tight_layout(pad=2.0)
    plt.ylabel('Amplitude', fontsize=fontsize, fontweight='bold') if i == (len(filters)//2) else None


plt.xlabel('Sample t[s]', fontsize=fontsize, fontweight='bold')
plt.show(block=False)
plt.savefig("images/post_lock_time_data.png", dpi=200, transparent=False)
plt.close()


fig = plt.figure(figsize=(22, 16))
x_fftd = np.fft.fft(x)

# Plot each filter's response
for i, filter in enumerate(filters):
    # Calculate position
    position = i * 2 + 1  # Position for frequency response plot
    x = mids[i]
    # Determine the width of the rectangle from the octave range
    fmin, fmax = octave_ranges[i]
    rect_width = fmax - fmin
    filter.plot_filter(fig, num_filters+1, 1, i+1)
    plt.axvline(x=x, ymin=0, ymax=1, color='r', linestyle='--')
    
    # Adding the shaded rectangle
    plt.axvspan(x - rect_width/2, x + rect_width/2, ymin=0, ymax=1, alpha=0.3, color='blue')

    plt.plot(x_fftd/max(x_fftd))
    plt.legend(['octave window', 'center', 'filter pass band', 'signal_fft'], fontsize=10, loc='upper right', bbox_to_anchor=(1, 1)) if i == 1 else None
    plt.xticks([x])
    ax = plt.gca()
    ax.set_xlim([0, filter.fs/2])
    ax.set_ylabel("Octave Detected", fontsize=10) if i >= 2 and i < 6 and i != 3 else ax.set_ylabel(" ", fontsize=10)
    plt.tight_layout(pad=2.0)


plt.show(block=False)
plt.savefig("images/post_lock_fft_freqdata.png", dpi=200, transparent=False)
plt.close()
