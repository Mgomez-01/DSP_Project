from util.FIR_filter import FIRFilter
import numpy as np
import matplotlib.pyplot as plt

# Reference frequency for A0
f_ref_C1 = 32.70  # Frequency of C1 in Hz
# Notes in an octave starting from C
# notes = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']

# # Midpoint frequency made using geometric mean
# midpoint_frequencies = [
#     46.2493028389542,
#     92.4986056779085,
#     184.997211355817,
#     369.994422711634,
#     739.988845423269,
#     1479.97769084654,
#     2959.95538169308
# ]

# midpoint_frequencies = np.array(midpoint_frequencies)
# # Calculate the number of semitones n from A0 to the midpoint frequency
# n = (12 * np.log2(midpoint_frequencies / f_ref_C1))

# print(f"steps from reference C1 note: {np.floor(n)}")

# # Given semitone positions starting from C1
# semitone_positions = np.floor(n)
# semitone_positions = [int(index) for index in semitone_positions]

# # Mapping semitone positions to notes within an octave
# note_names = [notes[position % 12] for position in semitone_positions]

# print(f"note_names: {note_names}")
filter = FIRFilter(N=100, fmin=1000, fmax=2000, padding_factor=1, fs=8000, passing=False)

filter.plot_filter1()
fig = plt.figure(figsize=(22,22))


fs = 8000
t1 = np.linspace(0, 0.25, int(fs*0.25), endpoint=False)
t2 = np.linspace(0.3, 0.55, int(fs*0.25), endpoint=False)
t3 = np.linspace(0.6, 0.85, int(fs*0.25), endpoint=False)

x1 = np.cos(2*np.pi*220*t1)
x2 = np.cos(2*np.pi*440*t2)
x3 = np.cos(2*np.pi*880*t3) + np.cos(2*np.pi*1760*t3)

zero_padding = np.zeros(int(fs*0.05))
x = np.concatenate((x1, zero_padding, x2, zero_padding, x3))




plt.plot()
input()
