from fastapi import FastAPI, WebSocket
import pyaudio
import numpy as np
import asyncio
import sys
import logging
import os

LOG = logging.getLogger(__name__)

app = FastAPI()

# Retrieve settings from environment variables with defaults
FORMAT = getattr(pyaudio, os.getenv("AUDIO_FORMAT", "paInt32"))
CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
RATE = int(os.getenv("AUDIO_RATE", "48000"))
FRAMES_PER_BUFFER = int(os.getenv("AUDIO_FRAMES_PER_BUFFER", "2048"))
MIC_INDEX = int(os.getenv("MIC_INDEX", "0"))

# Initialize PyAudio
audio = pyaudio.PyAudio()

if MIC_INDEX == -1:
    LOG.info("no mic detected...\nExiting")
    sys.exit(1)


# Function to capture audio data
def stream_audio(websocket):
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=MIC_INDEX,
                        frames_per_buffer=FRAMES_PER_BUFFER)

    LOG.info("stream created")

    while True:
        try:
            data = np.frombuffer(stream.read(FRAMES_PER_BUFFER), dtype=np.float32).tolist()
            asyncio.run(websocket.send_json({"data": data}))
            LOG.info(f"Data stream: {data}")
        except Exception as e:
            LOG.info(f"Error in audio stream: {e}")
            break

    stream.stop_stream()
    stream.close()


@app.websocket("/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    LOG.info("Socket data accepted, streaming")
    stream_audio(websocket)


@app.on_event("shutdown")
def shutdown_event():
    LOG.info("shutdown event.")
    audio.terminate()
