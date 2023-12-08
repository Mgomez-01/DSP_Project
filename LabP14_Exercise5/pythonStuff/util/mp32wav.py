from pydub import AudioSegment
import sys
# Replace 'path_to_mp3' and 'path_to_wav' with your file paths
path_to_mp3 = sys.argv[1]
path_to_wav = sys.argv[2]

# Load the mp3 file
audio = AudioSegment.from_mp3(path_to_mp3)

# Export as wav
audio.export(path_to_wav, format="wav")
