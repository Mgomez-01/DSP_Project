import numpy as np

# Reference frequency for A0
f_ref_C1 = 32.70  # Frequency of C1 in Hz
# Notes in an octave starting from C
notes = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']

# Midpoint frequency made using geometric mean
midpoint_frequencies = [
    46.2493028389542,
    92.4986056779085,
    184.997211355817,
    369.994422711634,
    739.988845423269,
    1479.97769084654,
    2959.95538169308
]

midpoint_frequencies = np.array(midpoint_frequencies)
# Calculate the number of semitones n from A0 to the midpoint frequency
n = (12 * np.log2(midpoint_frequencies / f_ref_C1))

print(f"steps from reference C1 note: {np.floor(n)}")

# Given semitone positions starting from C1
semitone_positions = np.floor(n)
semitone_positions = [int(index) for index in semitone_positions]

# Mapping semitone positions to notes within an octave
note_names = [notes[position % 12] for position in semitone_positions]

print(f"note_names: {note_names}")
