from util.OctaveBandFilt import OctaveBandFilter
import numpy as np
from rich import print
import matplotlib.pyplot as plt

# Example usage
filter_length = 100  # A typical length for FIR filters
center_frequency = 1000  # Center frequency in Hz
sampling_frequency = 8000  # Sampling frequency in Hz

# Recreating the OctaveBandFilter instance with the updated class definition
octave_filter = OctaveBandFilter(filter_length, center_frequency, sampling_frequency)
octave_filter.calculate_coefficients()

# Generating a test signal - a simple sine wave at the center frequency
test_signal = np.sin(2 * np.pi * center_frequency / sampling_frequency * np.arange(0, sampling_frequency))

# Applying the filter to the test signal
filtered_signal = octave_filter.apply_filter(test_signal)


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
filter_length = 101
center_frequency = 1000  # Center frequency in Hz
sampling_frequency = 8000  # Sampling frequency in Hz

# Creating the filter instance and calculating coefficients
octave_filter = OctaveBandFilter(filter_length, center_frequency, sampling_frequency)
octave_filter.calculate_coefficients()

# Generating an FM signal
duration = 1  # 1 second duration
start_freq = 0  # Starting frequency of 0 Hz
end_freq = 2 * octave_filter.center_frequency  # Ending at twice the center frequency of the filter
fm_signal = generate_fm_signal(sampling_frequency, duration, start_freq, end_freq)

# Filtering the FM signal
filtered_fm_signal = octave_filter.apply_filter(fm_signal)

# Plotting the original and filtered signals
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(fm_signal)
plt.title('Original Frequency Modulated Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(filtered_fm_signal)
plt.title('Filtered Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
