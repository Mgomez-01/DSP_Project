from util.OctaveBandFilt import OctaveBandFilter, ourOctaveBandFilter
from util.FIR_filter import FIRFilter
import numpy as np
import matplotlib.pyplot as plt
from rich import print


def windowed_sinc(filter_length, low_freq, high_freq):
    n = np.arange(filter_length)
    mid = (filter_length - 1) / 2
    window = np.hamming(filter_length)

    # Debugging: Print frequency values
    print("low_freq:", low_freq)
    print("high_freq:", high_freq)

    sinc_low = np.sinc(2 * low_freq * (n - mid))
    sinc_high = np.sinc(2 * high_freq * (n - mid))
    bp_filter = sinc_high - sinc_low

    # Debugging: Print sinc function outputs
    print("sinc_low sample values:", sinc_low[:5])
    print("sinc_high sample values:", sinc_high[:5])

    bp_filter *= window
    bp_filter_sum = np.sum(bp_filter)

    # Debugging: Check for zero sum
    print("bp_filter sum after window:", bp_filter_sum)
    if bp_filter_sum == 0:
        bp_filter_sum = 1  # To avoid division by zero

    bp_filter /= bp_filter_sum

    # Debugging: Final check
    print("Final bp_filter sample values:", bp_filter[:5])
    return bp_filter


# Ben waz here
# Ben waz not here
# Ben waz here
#Chase was here

# Example usage
filter_length = 150  # A typical length for FIR filters
center_frequency = 220  # Center frequency in Hz
sampling_frequency = 12000  # Sampling frequency in Hz

# Recreating the OctaveBandFilter instance with the updated class definition
octave_filter = OctaveBandFilter(filter_length, center_frequency, sampling_frequency)
octave_filter.calculate_coefficients()


# Recreating the OctaveBandFilter instance with the updated class definition
ourOctave_filter = ourOctaveBandFilter(filter_length, center_frequency, sampling_frequency, windowed_sinc)
ourOctave_filter.calculate_coefficients()
print("Coefficients:", ourOctave_filter.coefficients)
# Generating a test signal - a simple sine wave at the center frequency
test_signal = np.sin(2 * np.pi * center_frequency / sampling_frequency * np.arange(0, sampling_frequency))

# Applying the filter to the test signal
filtered_signal = octave_filter.apply_filter(test_signal)
# Applying the filter to the test signal
ourFiltered_signal = ourOctave_filter.apply_filter(test_signal)


# print(f"filter_length: {filter_length}")
# print(f"center_frequency: {center_frequency}")
# print(f"sampling_frequency: {sampling_frequency}")
# plt.figure(figsize=(10, 10))
# plt.plot(range(0, len(test_signal)), test_signal, label="test_signal", color="blue")
# plt.plot(range(0, len(filtered_signal)), filtered_signal, label="filtered_signal", color="red", linestyle="dashed")
# plt.xlim(0, 100)
# plt.title('Test Signal vs Filtered Signal')
# plt.xlabel('Time')
# plt.ylabel('Signal Amplitude')

# plt.show()


def generate_fm_signal(sampling_frequency, duration, start_freq, end_freq):
    """
    Generate a frequency modulated (FM) signal.
    """
    t = np.arange(0, duration, 1/sampling_frequency)  # Time vector
    instantaneous_frequency = np.linspace(start_freq, end_freq, len(t))
    phase = 2 * np.pi * np.cumsum(instantaneous_frequency) / sampling_frequency
    signal = np.sin(phase)
    return signal

# Parameters for the OctaveBandFilter
filter_length = 150
center_frequency = 440  # Center frequency in Hz
sampling_frequency = 8000  # Sampling frequency in Hz
# Lambda function for windowed sinc



# Creating the filter instance and calculating coefficients
Octave_filter = OctaveBandFilter(filter_length, center_frequency, sampling_frequency)
Octave_filter.calculate_coefficients()
# Creating the filter instance and calculating coefficients
ourOctave_filter = ourOctaveBandFilter(filter_length, center_frequency, sampling_frequency, windowed_sinc)
ourOctave_filter.calculate_coefficients()

# Generating an FM signal
duration = 5  # 1 second duration
start_freq = 1  # Starting frequency of 0 Hz
end_freq = 4 * ourOctave_filter.center_frequency  # Ending at twice the center frequency of the filter
fm_signal = generate_fm_signal(sampling_frequency, duration, start_freq, end_freq)

# Filtering the FM signal
filtered_fm_signal = Octave_filter.apply_filter(fm_signal)
# Filtering the FM signal
ourFiltered_fm_signal = ourOctave_filter.apply_filter(fm_signal)

# Plotting the original and filtered signals
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(fm_signal)
plt.title('Original Frequency Modulated Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(np.abs(ourFiltered_fm_signal))
plt.title('My Filtered Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')


plt.subplot(3, 1, 3)
plt.plot(np.abs(filtered_fm_signal))
plt.title('Scipy Filtered Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show(block=False)
fir_filter = [None]*10
for i in range(0, 10):
    fir_filter[i] = FIRFilter(N=32, fmin=i + 1, fmax=i + 5, padding_factor=25)
    fir_filter[i].plot_filter()
print(fir_filter)
input()
