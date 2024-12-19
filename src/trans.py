from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
import os

# Файл для збереження результатів
output_file = "transcription_only.txt"

# Очищаємо файл перед записом (якщо він вже існує)
with open(output_file, "w") as f:
    f.write("Результати транскрипції:\n\n")

# Завантаження моделі Whisper для розпізнавання тексту
whisper_model = whisper.load_model("turbo")

mp3_file = os.path.join(os.path.dirname(__file__), "exmpl.mp3")

if os.path.exists(mp3_file):
    print(f"Файл {mp3_file} знайдено")
else:
    print(f"Файл {mp3_file} не знайдено")
    
audio = whisper.load_audio(mp3_file)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(whisper_model.device)

# detect the spoken language
_, probs = whisper_model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(whisper_model, mel, options)

# print the recognized text
print(result.text)


with open(output_file, "a") as f:
    f.write(result.text)
