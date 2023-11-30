import pyaudio
import numpy as np
from util.FIR_filter import FIRFilter
import matplotlib.pyplot as plt
import time
import sys
from queue import Queue
# Create a queue for thread-safe data transfer
data_queue = Queue()


def calculate_octave_ranges(base_freq, num_octaves, sample_rate):
    octave_ranges = []
    for i in range(num_octaves):
        fmin = base_freq * (2 ** i)  # Minimum frequency for the octave
        fmax = base_freq * (2 ** (i + 1))  # Maximum frequency for the octave

        # Normalize the frequencies
        fmin_normalized = fmin / sample_rate
        fmax_normalized = fmax / sample_rate

        octave_ranges.append((fmin_normalized, fmax_normalized))
    return octave_ranges


# Initialize PyAudio
pa = pyaudio.PyAudio()
devices = pa.get_device_count()
print(f"num devices: {devices}")
if devices < 1:
    print(f"num of devices is {devices}. need at least one device")
    sys.exit(1)
sample_rate = 44100  # Adjust as needed
mic_index = -1
for i in range(pa.get_device_count()):
    dev = pa.get_device_info_by_index(i)
    print(f"Device {i}: {dev['name']}")
    # add a device name from the list that you have
    if 'Razer Kiyo' in dev['name']:
        print(f"razr cam has mic, using index {i}")
        mic_index = i
        sample_rate = 16000  # for razer kiyo cam
        break

if mic_index == -1:
    print("no mic detected...\nExiting")
    sys.exit(1)

# Sample rate and base frequency
base_freq = 27.5  # Starting frequency of the first octave
# Calculate octave ranges
octave_ranges = calculate_octave_ranges(base_freq, 7, sample_rate)
N = 1000
# Instantiate FIR filters for each octave
filters = [FIRFilter(N, fmin=fmin * N, fmax=fmax * N) for fmin, fmax in octave_ranges]
print(f"octave ranges: {octave_ranges}")

# Define a decay time in seconds (adjust as needed)
decay_time = 0.5  # half a second, for example

# Keep track of the last time a note was detected for each filter
last_detection_times = [0] * len(filters)


# Initialize the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
octave_indices = np.arange(1, 8)  # Octave indices (1-7)
octave_values = np.zeros(7)  # Initial octave values
stem_container = ax.stem(octave_indices, octave_values)
ax.set_ylim(0, 1.5)  # Set the limits of the y-axis


def update_plot(stem_container, octave_values):
    
    try:
        # Update the heights of the stems
        stem_container[0].set_ydata(octave_values)
        plt.draw()
        plt.pause(0.01)  # Pause to update the plot
    except ValueError as e:
        print(f"error in update from remove. \n Error: {e}")


# Stream callback function
def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)

    # Process through each FIR filter
    filtered_data = [filter.process(audio_data) for filter in filters]
    current_time = time.time()
    # Analyze the filtered data to update octave_values
    for i, data in enumerate(filtered_data):
        if np.max(np.abs(data)) > 50:  # Threshold to detect note presence
            octave_values[i] = 1
            last_detection_times[i] = current_time
            print(f"max data on filter {i}: {np.max(np.abs(data))}")
        elif current_time - last_detection_times[i] > decay_time:
            octave_values[i] = 0

    # Instead of updating the plot, put the data in the queue
    data_queue.put(octave_values.copy())
    return (in_data, pyaudio.paContinue)


# Open stream
stream = pa.open(format=pyaudio.paFloat32,
                 channels=1,
                 rate=16000,
                 input=True,
                 input_device_index=mic_index,
                 stream_callback=callback)

stream.start_stream()

# Main loop
try:
    while stream.is_active():
        if not data_queue.empty():
            octave_values = data_queue.get()
            update_plot(stem_container, octave_values)
        time.sleep(.1)  # Adjust sleep time as needed
except KeyboardInterrupt:
    # Handle exit gracefully
    print("ctrl-C pressed\nExiting")
    stream.stop_stream()
    stream.close()
    pa.terminate()
