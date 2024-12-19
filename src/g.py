from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment


#/root/.config/gcloud/application_default_credentials.json
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/.config/gcloud/application_default_credentials.json"

client = speech.SpeechClient()

output_file = "transcription_g.txt"

mp3_file = "src/exmpl.mp3"

# # Завантажуємо MP3-файл
audio = AudioSegment.from_file(mp3_file, format="mp3")
audio = audio.set_channels(1)  # Преобразуем в моно
#audio = audio.set_frame_rate(8000)

# # Аудіофайл
audio_file = "exmpl.wav"

# # Конвертуємо в WAV і зберігаємо
audio.export(audio_file, format="wav")

speech_file = audio_file

with open(speech_file, "rb") as audio_file:
    content = audio_file.read()

audio = speech.RecognitionAudio(content=content)

diarization_config = speech.SpeakerDiarizationConfig(
    enable_speaker_diarization=True,
    min_speaker_count=2,
    max_speaker_count=10,
)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=48000,
    language_code="uk-UA",
    diarization_config=diarization_config,
)

print("Waiting for operation to complete...")
response = client.recognize(config=config, audio=audio)

with open(output_file, "a") as f:
    f.write(str(response))



# The transcript within each result is separate and sequential per result.
# However, the words list within an alternative includes all the words
# from all the results thus far. Thus, to get all the words with speaker
# tags, you only have to take the words list from the last result:
result = response.results[-1]

words_info = result.alternatives[0].words

# Printing out the output:
for word_info in words_info:
    print(f"word: '{word_info.word}', speaker_tag: {word_info.speaker_tag}")

# return result