import pyaudio
import wave
import sys

FORMAT = pyaudio.paInt32
CHANNELS = 1
RATE = 48000
CHUNK = 2**11
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"

audio = pyaudio.PyAudio()
MIC_INDEX = 0
while True:
    try:
        # Start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            input_device_index=MIC_INDEX,
                            frames_per_buffer=CHUNK)
        print("Recording...")
        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("Finished Recording.")
        break

    except Exception as e:
        print(f"error {e}")
        MIC_INDEX += 1
        if MIC_INDEX > 5:
            sys.exit(1)

print(f"working mic {MIC_INDEX}")
# Stop Recording
stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
