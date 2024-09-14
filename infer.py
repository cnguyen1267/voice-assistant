from faster_whisper import WhisperModel
import torch

model_size = "large-v3"

if torch.cuda.is_available():
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
else:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe(file: str):
    segments, info = model.transcribe(file, beam_size=5)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

if __name__ == "__main__":
    transcribe("sample.mp3")
