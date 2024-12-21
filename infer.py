import threading
import queue
import time

import pyaudio
import webrtcvad
import numpy as np

from faster_whisper import WhisperModel
import torch

# Configuration Parameters
MODEL_SIZE = "large-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

FORMAT = pyaudio.paInt16  # 16-bit PCM
CHANNELS = 1              # Mono audio
RATE = 16000              # 16kHz sampling rate for VAD
FRAME_DURATION = 30       # Frame size in ms for VAD
CHUNK = int(RATE * FRAME_DURATION / 1000)  # Number of samples per frame

# Initialize the Whisper model
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(1)  # 0: least aggressive, 3: most aggressive

# Queue to hold audio frames
audio_queue = queue.Queue()

# Function to capture audio from the microphone
def audio_capture(stream, audio_q):
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_q.put(data)
    except Exception as e:
        print(f"Error capturing audio: {e}")

# Function to process audio frames and perform VAD
def vad_processor(audio_q, segments_q):
    frames = []
    in_speech = False
    speech_start = 0
    frame_duration_sec = FRAME_DURATION / 1000.0
    total_frames = 0

    while True:
        frame = audio_q.get()
        if frame is None:
            break

        is_speech = vad.is_speech(frame, RATE)

        if is_speech:
            if not in_speech:
                in_speech = True
                speech_start = time.time()
            frames.append(frame)
        else:
            if in_speech:
                in_speech = False
                speech_end = time.time()
                # Combine frames
                audio_data = b"".join(frames)
                segments_q.put(audio_data)
                frames = []

        total_frames += 1

# Function to transcribe audio segments
def transcribe_segments(segments_q):
    while True:
        audio_data = segments_q.get()
        if audio_data is None:
            break

        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe using faster_whisper
        segments, info = model.transcribe(audio_np, beam_size=5, language=None)

        if info.language_probability < 0.9:
            continue

        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

def main():
    # Open the microphone stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening...")

    # Queues for audio frames and speech segments
    segments_queue = queue.Queue()

    # Start audio capture thread
    capture_thread = threading.Thread(target=audio_capture, args=(stream, audio_queue))
    capture_thread.daemon = True
    capture_thread.start()

    # Start VAD processing thread
    vad_thread = threading.Thread(target=vad_processor, args=(audio_queue, segments_queue))
    vad_thread.daemon = True
    vad_thread.start()

    # Start transcription thread
    transcription_thread = threading.Thread(target=transcribe_segments, args=(segments_queue,))
    transcription_thread.daemon = True
    transcription_thread.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Signal threads to exit
        audio_queue.put(None)
        segments_queue.put(None)

if __name__ == "__main__":
    main()
