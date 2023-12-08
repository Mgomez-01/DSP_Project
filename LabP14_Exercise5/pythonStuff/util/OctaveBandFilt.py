from scipy.signal import lfilter
from scipy.signal import firwin
import numpy as np


class OctaveBandFilter:
    def __init__(self, filter_length, center_frequency, sampling_frequency):
        self.filter_length = filter_length
        self.center_frequency = center_frequency
        self.sampling_frequency = sampling_frequency
        self.coefficients = None

    def calculate_coefficients(self):
        lower_freq = self.center_frequency / np.sqrt(2)
        upper_freq = self.center_frequency * np.sqrt(2)
        nyquist = 0.5 * self.sampling_frequency
        lower_freq /= nyquist
        upper_freq /= nyquist
        self.coefficients = firwin(self.filter_length, [lower_freq, upper_freq], pass_zero=False)
        return self.coefficients

    def apply_filter(self, signal):
        # Ensure the filter coefficients have been calculated
        if self.coefficients is None:
            raise ValueError("Filter coefficients have not been calculated. Call calculate_coefficients() first.")

        # Apply the filter to the signal
        filtered_signal = lfilter(self.coefficients, 1.0, signal)
        return filtered_signal


class ourOctaveBandFilter:
    def __init__(self, filter_length, center_frequency, sampling_frequency, windowed_sinc_func):
        self.filter_length = filter_length
        self.center_frequency = center_frequency
        self.sampling_frequency = sampling_frequency
        self.windowed_sinc_func = windowed_sinc_func
        self.coefficients = None

    def calculate_coefficients(self):
        # Calculate lower and upper frequencies of the octave band
        lower_freq = self.center_frequency / np.sqrt(2)
        upper_freq = self.center_frequency * np.sqrt(2)
        nyquist = 0.5 * self.sampling_frequency
        
        # Normalize frequencies
        lower_freq /= nyquist
        upper_freq /= nyquist

        # Generate coefficients using a windowed-sinc filter
        self.coefficients = self.windowed_sinc_func(self.filter_length, lower_freq, upper_freq)
        return self.coefficients

    def apply_filter(self, signal):
        # Ensure the filter coefficients have been calculated
        if self.coefficients is None:
            raise ValueError("Filter coefficients have not been calculated. Call calculate_coefficients() first.")
        # Apply the filter to the signal
        filtered_signal = np.convolve(signal, self.coefficients, mode='same')
        return filtered_signal

# Example usage
# filter = OctaveBandFilter(filter_length=101, center_frequency=1000, sampling_frequency=8000)
# filter.calculate_coefficients()
# filtered_signal = filter.apply_filter(your_input_signal)
