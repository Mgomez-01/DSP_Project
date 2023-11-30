import pyaudio
import numpy as np
from util.FIR_filter import FIRFilter
import matplotlib.pyplot as plt
import time
import sys
from queue import Queue
import threading
# Create a queue for thread-safe data transfer
data_queue = Queue()


def plot_update_thread():
    while True:
        if not data_queue.empty():
            octave_values = data_queue.get()
            update_plot(stem_container, octave_values)
        time.sleep(0.1)  # Adjust sleep time as needed


# # Start the plot update thread
# plot_thread = threading.Thread(target=plot_update_thread, daemon=True)
# plot_thread.start()


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
octave_ranges = calculate_octave_ranges(base_freq, 7, 3500)
N = 10000
# Instantiate FIR filters for each octave
filters = [FIRFilter(N, fmin=fmin*N, fmax=fmax*N, padding_factor=9) for fmin, fmax in octave_ranges]
print(f"octave ranges: {octave_ranges}")
#sys.exit(1)
# Define a decay time in seconds (adjust as needed)
decay_time = 0.25  # half a second, for example

# Keep track of the last time a note was detected for each filter
last_detection_times = [0] * len(filters)


# Initialize the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
octave_indices = np.arange(1, 8)  # Octave indices (1-7)
octave_values = np.zeros(7)  # Initial octave values
stem_container = ax.stem(octave_indices, octave_values)
ax.set_ylim(0, 1.5)  # Set the limits of the y-axis
ax.set_xlim(-2, 25)  # Set the limits of the y-axis


def update_plot():
    if not data_queue.empty():
        octave_values = data_queue.get()
        stem_container[0].set_ydata(octave_values)
        plt.draw()
        plt.pause(0.1)  # Adjust sleep time as needed


on_threshold = 25  # Threshold to turn the indicator on
off_threshold = 15  # Threshold to turn the indicator off
is_note_detected = [False] * 7  # State for each octave


#fig2, ax2 = plt.subplots()
def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    filtered_data = [filter.process(audio_data) for filter in filters]
    # Plot a segment of audio data
    plt.cla()
    plt.plot(np.abs(filtered_data))
    #ax2.set_ylim(-1, 1)  # Set the limits of the y-axis
    #ax2.set_xlim(-25, 25)  # Set the limits of the y-axis
    plt.show()
    for i, data in enumerate(filtered_data):
        if is_note_detected[i]:
            if np.max(np.abs(data)) < off_threshold:
                print(f"max data on filter {i} off: {np.max(np.abs(data))}")
                is_note_detected[i] = False
                octave_values[i] = 0
        else:
            if np.max(np.abs(data)) > on_threshold:
                print(f"max data on filter {i} on: {np.max(np.abs(data))}")
                is_note_detected[i] = True
                octave_values[i] = 1

    data_queue.put(octave_values.copy())
    return (in_data, pyaudio.paContinue)


# # Stream callback function
# def callback(in_data, frame_count, time_info, status):
#     audio_data = np.frombuffer(in_data, dtype=np.float32)

#     # Process through each FIR filter
#     filtered_data = [filter.process(audio_data) for filter in filters]
#     current_time = time.time()
#     # Analyze the filtered data to update octave_values
#     for i, data in enumerate(filtered_data):
#         if np.max(np.abs(data)) > 20:  # Threshold to detect note presence
#             octave_values[i] = 1
#             last_detection_times[i] = current_time
#             print(f"max data on filter {i}: {np.max(np.abs(data))}")
#         elif current_time - last_detection_times[i] > decay_time:
#             octave_values[i] = 0

#     # Instead of updating the plot, put the data in the queue
#     data_queue.put(octave_values.copy())
#     return (in_data, pyaudio.paContinue)


# Function to handle audio processing
def audio_thread():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paFloat32,
                     channels=1,
                     rate=sample_rate,
                     input=True,
                     input_device_index=mic_index,
                     stream_callback=callback)

    stream.start_stream()
    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()
    pa.terminate()


# Start the audio processing in a separate thread
audio_proc_thread = threading.Thread(target=audio_thread)
audio_proc_thread.start()


# Main loop
try:
    while True:
        update_plot()
        # if not data_queue.empty():
        #     octave_values = data_queue.get()
        #     update_plot(stem_container, octave_values)
        time.sleep(.15)  # Adjust sleep time as needed
except KeyboardInterrupt:
    # Handle exit gracefully
    print("ctrl-C pressed\nExiting")
    

# Wait for the audio thread to finish
audio_proc_thread.join()
