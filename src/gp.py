from pydub import AudioSegment
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from datetime import datetime
from pyannote.audio import Pipeline
from pydub.effects import normalize
import whisper
import os
import noisereduce as nr
import librosa
import soundfile as sf

print(f"imports")

def cur_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Файл для збереження результатів
output_file = "trans_pyannote_google.txt"
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# Очищаємо файл перед записом (якщо він вже існує)
with open(output_file, "a") as f:
    f.write(cur_time() + "Результати транскрипції:\n\n")

print(f"output file created")
PROJECT_ID = "united-helix-444614-e4"
#/root/.config/gcloud/application_default_credentials.json
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "application_default_credentials.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

mp3_file = os.path.join(os.path.dirname(__file__), "exmpl.mp3")

if os.path.exists(mp3_file):
    print(f"Файл {mp3_file} знайдено")
else:
    print(f"Файл {mp3_file} не знайдено")

# # Завантажуємо MP3-файл
audio = AudioSegment.from_file(mp3_file, format="mp3")
audio = audio.set_channels(1)  # Преобразуем в моно
#audio = audio.set_frame_rate(8000)

# # Аудіофайл
audio_file = "exmpl.wav"

# # Конвертуємо в WAV і зберігаємо
audio.export(audio_file, format="wav")

print(f"wav file created")

# Вирізання сегменту аудіо для обробки
def extract_audio_segment(input_file, start_time, end_time, output_file):
    audio = AudioSegment.from_file(input_file)
    audio = audio[start_time * 1000:end_time * 1000]  # Час у мілісекундах
    audio = audio.set_channels(1)  # Преобразуем в моно
    audio.export(output_file, format="wav")
    return output_file

def transcribe_word_level_confidence_v2(
    audio_file: str,
) -> str:
    # Instantiates a client
    client = SpeechClient()
    PROJECT_ID = "united-helix-444614-e4"

    # Reads a file as bytes
    with open(audio_file, "rb") as file:
        audio_content = file.read()
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["uk-UA", "ru-RU"],
        model="long",
        features=cloud_speech.RecognitionFeatures(
            enable_word_confidence=True,
        ),
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
        config=config,
        content=audio_content,
    )

    # Transcribes the audio into text
    response = client.recognize(request=request)

    
    result = response.results[-1] if response.results else False

    if result and result.alternatives:
        return result.alternatives[-1].transcript
    else:
        return ''

    

# Завантаження моделі pyannote для діаризації
token = "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"
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                     use_auth_token=token)

with open(output_file, "a") as f:
    f.write(f"diarization started: " + cur_time())
print(f"diarization started: " + cur_time())
# Виконання діаризації
diarization = pipeline(audio_file)
with open(output_file, "a") as f:
    f.write(f"diarization end: " + cur_time())
print(f"diarization end: " + cur_time())

# Розпізнавання тексту для кожного сегмента
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # Вирізаємо сегмент аудіо для конкретного мовця
    start_time = turn.start
    end_time = turn.end

        # Вирізаємо сегмент
    segment_file = f"temp_segment_{start_time:.1f}_{end_time:.1f}.wav"
    extract_audio_segment(audio_file, start_time, end_time, segment_file)
    
    # Виконуємо розпізнавання тексту Whisper'ом
    transcript = transcribe_word_level_confidence_v2(segment_file)
    
    if transcript == '':
        continue
    # Форматований результат
    result_line = f"{start_time:.1f}s - {end_time:.1f}s ({speaker}): {transcript}\n"

    # Запис у файл
    with open(output_file, "a") as f:
        f.write(result_line)
    
    print(result_line)

    # Видаляємо тимчасовий файл сегмента
    if os.path.exists(segment_file):
        os.remove(segment_file)

if os.path.exists(audio_file):
    os.remove(audio_file)

print(f"процесс закінчено")