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


print(f"filter_length: {filter_length}")
print(f"center_frequency: {center_frequency}")
print(f"sampling_frequency: {sampling_frequency}")
plt.figure(figsize=(10, 10))
plt.plot(range(0, len(test_signal)), test_signal, label="test_signal", color="blue")
plt.plot(range(0, len(filtered_signal)), filtered_signal, label="filtered_signal", color="red", linestyle="dashed")
plt.xlim(0, 100)
plt.title('Test Signal vs Filtered Signal')
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')

plt.show()
