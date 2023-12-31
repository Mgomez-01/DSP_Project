import pyaudio
import numpy as np
from util.FIR_filter import FIRFilter
import matplotlib.pyplot as plt
import time
import sys
from queue import Queue
import threading
# Create a queue for thread-safe data transfer
octave_data_queue = Queue()
filtered_data_queue = Queue()


def plot_update_thread():
    while True:
        if not octave_data_queue.empty():
            octave_values = octave_data_queue.get()
            update_plot(stem_container, octave_values)
        time.sleep(0.1)  # Adjust sleep time as needed


# # Start the plot update thread
# plot_thread = threading.Thread(target=plot_update_thread, daemon=True)
# plot_thread.start()


def calculate_note_ranges(base_freq, num_notes, sample_rate):
    note_ranges = []
    for i in range(num_notes):
        fmin = base_freq * (2 ** i)  # Minimum frequency for the note
        fmax = base_freq * (2 ** (i/12))  # Maximum frequency for the octave

        # Normalize the frequencies
        fmin_normalized = fmin
        fmax_normalized = fmax

        note_ranges.append((fmin_normalized, fmax_normalized))
    return note_ranges


def calculate_octave_ranges(base_freq, num_octaves, sample_rate):
    octave_ranges = []
    for i in range(num_octaves+1):
        fmin = base_freq * (2 ** i)  # Minimum frequency for the octave
        fmax = base_freq * (2 ** (i+1))  # Maximum frequency for the octave

        # Normalize the frequencies
        fmin_normalized = fmin
        fmax_normalized = fmax

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
this_device = None
for i in range(pa.get_device_count()):
    dev = pa.get_device_info_by_index(i)
    print(f"Device {i}: {dev['name']}, Channels: {dev['maxInputChannels']}")
    for key in dev.keys():
        print(f"key: {key}val: {dev[key]}")
    # add a device name from the list that you have
    if 'Razer Kiyo' in dev['name']:
        print(f"razr cam has mic, using index {i}")
        mic_index = i
        sample_rate = 48000  # for razer kiyo cam
        on_threshold = .3  # Threshold to turn the indicator on
        off_threshold = .1  # Threshold to turn the indicator off
        this_device = dev
        break
    if 'default' in dev['name']:
        mic_index = i
        print(f"sydef mic, using index {mic_index}")
        sample_rate = 48000  # for default mic in most systems
        on_threshold = .3  # Threshold to turn the indicator on
        off_threshold = .1  # Threshold to turn the indicator off
        this_device = dev
        break

if mic_index == -1:
    print("no mic detected...\nExiting")
    sys.exit(1)

# Sample rate and base frequency
base_freq = 16.35159783  # Starting frequency of the first octave
# Calculate octave ranges
num_octaves = 7
octave_ranges = calculate_octave_ranges(base_freq, num_octaves, sample_rate)
num_notes = 88
note_ranges = calculate_note_ranges(base_freq, num_notes, sample_rate)
N = 1000
# Instantiate FIR filters for each octave
filters = [FIRFilter(N, fmin=fmin, fmax=fmax, padding_factor=2, fs=8000) for fmin, fmax in octave_ranges]
fig = plt.figure(figsize=(22, 22))
for i, filter in enumerate(filters):
    filter.plot_filter(fig, len(filters)+1, 1, i+1)
    plt.tight_layout(pad=2)
    
plt.show()
input()
#sys.exit(1)

#filters = [FIRFilter(N, fmin=fmin*N, fmax=fmax*N, padding_factor=9) for fmin, fmax in octave_ranges]
print(f"octave ranges: {octave_ranges}")
#print(f"note ranges: {note_ranges}")
#sys.exit(1)
# Define a decay time in seconds (adjust as needed)
decay_time = 0.01  # half a second, for example

# Keep track of the last time a note was detected for each filter
last_detection_times = [0] * len(filters)

buffer_len = 1024<<2
stem_buffer = np.arange(1, buffer_len+1)
# Initialize the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
octave_indices = np.arange(0, num_octaves+1)  # Octave indices
octave_values = np.zeros(num_octaves+1)
print(f"len(octave_values): {len(octave_values)}")
print(f"len(octave_indices): {len(octave_indices)}")
filtered_data_vals = np.zeros(buffer_len)# Initial octave values
stem_container = ax.stem(octave_indices, octave_values)
stem_container2 = ax2.stem(stem_buffer, filtered_data_vals)
#ax.set_ylim(0, 1.5)  # Set the limits of the y-axis
#ax.set_xlim(-2, 9)  # Set the limits of the y-axis
ax.set_ylim([0, 2])
ax.set_xlim([0, 10])
ax2.set_ylim([-1, 2])
ax2.set_xlim([0, buffer_len])
plt.show(block=False)
plt.pause(.01)


def update_plot():
    if not octave_data_queue.empty() and not filtered_data_queue.empty():
        octave_values = octave_data_queue.get()
        filtered_data_vals = filtered_data_queue.get()

        # Update octave stem plot
        stem_container.markerline.set_ydata(octave_values)
        #print(f"octave_indices: {octave_indices} \noctave_values: {octave_values}")
        stem_container.stemlines.set_segments([[[x, 0], [x, y]] for x, y in zip(octave_indices, octave_values)])

        # Update filtered data stem plot
        stem_container2.markerline.set_ydata(filtered_data_vals)
        stem_container2.stemlines.set_segments([[[x, 0], [x, y]] for x, y in zip(stem_buffer, filtered_data_vals)])

        ax.draw_artist(ax.patch)
        ax2.draw_artist(ax2.patch)
        ax.draw_artist(stem_container.markerline)
        ax.draw_artist(stem_container.stemlines)
        ax2.draw_artist(stem_container2.markerline)
        ax2.draw_artist(stem_container2.stemlines)
        fig.canvas.update()
        fig.canvas.flush_events()
        fig2.canvas.update()
        fig2.canvas.flush_events()


# def update_plot():
#     if not octave_data_queue.empty() and not filtered_data_queue.empty():
#         #print("updating plot")
#         octave_values = octave_data_queue.get()
#         filtered_data_vals = filtered_data_queue.get()

#         # Update the octave stem plot
#         stem_container.markerline.set_ydata(octave_values)
#         stem_container.stemlines.set_segments([[[i, 0], [i, y]] for i, y in enumerate(octave_values)])

#         # Update the filtered data stem plot
#         stem_container2.markerline.set_ydata(filtered_data_vals)
#         stem_container2.stemlines.set_segments([[[i, 0], [i, y]] for i, y in enumerate(filtered_data_vals)])

#         # Redraw the canvas
#         fig.canvas.draw()
#         plt.pause(0.1)  # Adjust pause time as needed


# def update_plot():
#     if not octave_data_queue.empty():
#         octave_values = octave_data_queue.get()
#         filtered_data_vals = filtered_data_queue.get()
#         ax.set_ylim([0, 10])
#         ax.set_xlim([0, 10])
#         stem_container[0].set_ydata(octave_values)
#         stem_container2[0].set_ydata(filtered_data_vals)
#         #print(f"data from stem: {octave_values}")
#         fig.canvas.draw()
#         plt.pause(1)  # Adjust sleep time as needed


is_note_detected = [False] * num_notes  # State for each octave


#fig2, ax2 = plt.subplots()
def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    filtered_data = [filter.process(audio_data) for filter in filters]
    #print(f" audio_data len = {len(audio_data)}")
    # Check if the length of audio_data is less than the buffer length
    if len(audio_data) < buffer_len:
        # Pad audio_data to make its length equal to buffer_len
        audio_data = np.pad(audio_data, (0, buffer_len - len(audio_data)), 'constant')

    # Rest of the processing remains the same
    filtered_data = [filter.process(audio_data) for filter in filters]
    # Summing the first 1024 elements of each array in filtered_data
    filtered_data_vals = audio_data
    # np.sum([arr[0:1023] for arr in filtered_data],axis=0)
    # Plot a segment of audio data
    for i, data in enumerate(filtered_data):
        max_abs_val = np.max(np.abs(data.real))
        if max_abs_val == 0 or np.isnan(max_abs_val):
            normed_data = data.real  # No normalization if max is 0 or nan
        else:
            normed_data = data.real / max_abs_val
        current_time = time.time()
        if i >= 6:
            break
        if is_note_detected[i]:
            print(f"i: {i}, normed_data max: {np.max(np.abs(normed_data.real))}, is_note_detected: {is_note_detected[i]}")
            if (np.max(normed_data.real) < off_threshold):
                print(f"last_detection_times[i]: {last_detection_times[i]}")
                is_note_detected[i] = False
                octave_values[i] = 0
        else:
            print(f"i: {i}, normed_data max: {np.max(np.abs(normed_data.real))}, is_note_detected: {is_note_detected[i]}")
            if np.max(np.abs(normed_data.real)) > on_threshold:
                last_detection_times[i] = current_time
                is_note_detected[i] = True
                octave_values[i] = 1
    octave_data_queue.put(octave_values.copy())
    filtered_data_queue.put(filtered_data_vals.copy())
    return (in_data, pyaudio.paContinue)


fig_test, ax_test = plt.subplots(figsize=(22,22))
ax_test.set_ylim([-1, 1])
plt.xlabel('Time')  # Label for x-axis
plt.ylabel('Amplitude')  # Label for y-axis
plt.title('all data Filtered Data Over Time')  # Title of the plot
gain = 2
def callback_notime(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    filtered_data = [filter.process(audio_data) for filter in filters]
    time = np.arange(len(filtered_data[0]))
    ax_test.clear()
    ax_test.set_ylim([-1, 1])
    plt.xlabel('Time')  # Label for x-axis
    plt.ylabel('Amplitude')  # Label for y-axis
    plt.title('all data Filtered Data Over Time')  # Title of the plot
    for data in filtered_data:
        plt.plot(time, data)
    plt.show(block=False)

    # Check if the length of audio_data is less than the buffer length
    if len(audio_data) < buffer_len:
        # Pad audio_data to make its length equal to buffer_len
        audio_data = np.pad(audio_data, (0, buffer_len - len(audio_data)), 'constant')

    #filtered_data_vals = audio_data
    filtered_data_vals = audio_data
    for i, data in enumerate(filtered_data):
        max_abs_val = np.max((data.real))
        min_abs_val = np.min((data.real))
        if max_abs_val == 0 or np.isnan(max_abs_val):
            max_abs_val = 1
            normed_data = data.real/max_abs_val  # No normalization if max is 0 or nan
        elif max_abs_val <= .5:
            normed_data = 2*data.real/max_abs_val
        else:
            normed_data = data.real/max_abs_val

        if min_abs_val == 0:
            power = 0
        else:
            power = 100*(max_abs_val - min_abs_val)

        on_threshold = 50  # Threshold to turn the indicator on

        # Use threshold to set octave_values to 1 or 0
        #print(f"max: {max_abs_val}")
        #print(f"min: {min_abs_val}")
        if power > on_threshold:
            is_note_detected[i] = True
            octave_values[i] = 1
        else:
            is_note_detected[i] = False
            #print(f"i: {i}")
            #print(f"len(filters): {len(filtered_data)}")
            #print(f"len(octave_values): {len(octave_values)}")
            
            octave_values[i] = 0

    octave_data_queue.put(octave_values.copy())
    filtered_data_queue.put(filtered_data_vals.copy())
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
#             #print(f"max data on filter {i}: {np.max(np.abs(data))}")
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
                     stream_callback=callback_notime,
                     frames_per_buffer=buffer_len)

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
        time.sleep(.01)  # Adjust sleep time as needed
except KeyboardInterrupt:
    # Handle exit gracefully
    print("ctrl-C pressed\nExiting")


# Wait for the audio thread to finish
audio_proc_thread.join()
