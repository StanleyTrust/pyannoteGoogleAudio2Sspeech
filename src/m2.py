from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.effects import normalize
import whisper
import os
import noisereduce as nr
import librosa
import soundfile as sf

# Файл для збереження результатів
output_file = "transcription.txt"

# Очищаємо файл перед записом (якщо він вже існує)
with open(output_file, "w") as f:
    f.write("Результати транскрипції:\n\n")

# Вирізання сегменту аудіо для обробки
def extract_audio_segment(input_file, start_time, end_time, output_file):
    audio = AudioSegment.from_file(input_file)
    segment = audio[start_time * 1000:end_time * 1000]  # Час у мілісекундах
    segment.export(output_file, format="wav")
    return output_file

print('imports passed')
token = "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"
# Завантаження моделі pyannote для діаризації
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                     use_auth_token=token)

# Завантаження моделі Whisper для розпізнавання тексту
whisper_model = whisper.load_model("medium")

mp3_file = os.path.join(os.path.dirname(__file__), "exmpl.mp3")

if os.path.exists(mp3_file):
    print(f"Файл {mp3_file} знайдено")
else:
    print(f"Файл {mp3_file} не знайдено")


# Виконання діаризації
diarization = pipeline(mp3_file)

with open("transcribe_db", "w") as f:
    f.write(whisper_model.transcribe(mp3_file)["text"])

# Розпізнавання тексту для кожного сегмента
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # Вирізаємо сегмент аудіо для конкретного мовця
    start_time = turn.start
    end_time = turn.end

        # Вирізаємо сегмент
    segment_file = f"temp_segment_{start_time:.1f}_{end_time:.1f}.wav"
    extract_audio_segment(mp3_file, start_time, end_time, segment_file)
    
    # Виконуємо розпізнавання тексту Whisper'ом
    result = whisper_model.transcribe(segment_file)
    transcript = result["text"]
    
    # Форматований результат
    result_line = f"{start_time:.1f}s - {end_time:.1f}s ({speaker}): {transcript}\n"

    # Запис у файл
    with open(output_file, "a") as f:
        f.write(result_line)
    
    print(result_line)

    # Видаляємо тимчасовий файл сегмента
    if os.path.exists(segment_file):
        os.remove(segment_file)

    
print(f"процесс закінчено")