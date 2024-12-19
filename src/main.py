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
    segment = segment.set_channels(1)  # Преобразуем в моно
    segment = segment.set_frame_rate(16000)  # Устанавливаем частоту 16000 Гц
    segment.export(output_file, format="wav")
    return output_file

print('imports passed')
token = "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"
# Завантаження моделі pyannote для діаризації
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                     use_auth_token=token)

# Завантаження моделі Whisper для розпізнавання тексту
whisper_model = whisper.load_model("turbo")

mp3_file = os.path.join(os.path.dirname(__file__), "exmpl.mp3")

if os.path.exists(mp3_file):
    print(f"Файл {mp3_file} знайдено")
else:
    print(f"Файл {mp3_file} не знайдено")


# # Завантажуємо MP3-файл
# audio = AudioSegment.from_file(mp3_file, format="mp3")

# # Аудіофайл
# audio_file = "exmpl.wav"

# # Конвертуємо в WAV і зберігаємо
# audio.export(audio_file, format="wav")

# reduced_file = "reduced.wav"

# y, sr = librosa.load(audio_file, sr=None)
# reduced_noise = nr.reduce_noise(y=y, sr=sr)
# sf.write(reduced_file, reduced_noise, sr)

# normalized_file = "normalized.wav"

# audio = AudioSegment.from_file(reduced_file)
# normalized_audio = normalize(audio)
# normalized_audio.export(normalized_file, format="wav")

# print(f" - видалено шум + нормалізовано")


# Виконання діаризації
diarization = pipeline(mp3_file)

# Розпізнавання тексту для кожного сегмента
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # Вирізаємо сегмент аудіо для конкретного мовця
    start_time = turn.start
    end_time = turn.end

    # Вирізаємо сегмент
    segment_file = f"temp_segment_{start_time:.1f}_{end_time:.1f}_({speaker}).wav"
    extract_audio_segment(mp3_file, start_time, end_time, segment_file)
    
    # Виконуємо розпізнавання тексту Whisper'ом
    audio = whisper.load_audio(segment_file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=whisper_model.dims.n_mels).to(whisper_model.device)

    print("Shape of mel spectrogram:", mel.shape)
    print("whisper_model.dims.n_mels:", whisper_model.dims.n_mels)

    # detect the spoken language
    _, probs = whisper_model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(whisper_model, mel, options)
    transcript = result.text
    
    # Форматований результат
    result_line = f"{start_time:.1f}s - {end_time:.1f}s ({speaker}): {transcript}\n"

    # Запис у файл
    with open(output_file, "a") as f:
        f.write(result_line)
    
    # Видаляємо тимчасовий файл сегмента
    # if os.path.exists(segment_file):
    #     os.remove(segment_file)