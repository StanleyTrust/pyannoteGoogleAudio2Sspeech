from pydub import AudioSegment
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from datetime import datetime


#/root/.config/gcloud/application_default_credentials.json
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/.config/gcloud/application_default_credentials.json"

output_file = "transcription_g2.txt"

mp3_file = "src/exmpl.mp3"

# # Завантажуємо MP3-файл
audio = AudioSegment.from_file(mp3_file, format="mp3")
audio = audio.set_channels(1)  # Преобразуем в моно
#audio = audio.set_frame_rate(8000)

# # Аудіофайл
audio_file = "exmpl.wav"

# # Конвертуємо в WAV і зберігаємо
audio.export(audio_file, format="wav")

PROJECT_ID = "united-helix-444614-e4"


def transcribe_word_level_confidence_v2(
    audio_file: str,
) -> cloud_speech.RecognizeResponse:
    """Transcribes a local audio file into text with word-level confidence.
    Args:
        audio_file (str): Path to the local audio file to be transcribed.
            Example: "resources/audio.wav"
    Returns:
        cloud_speech.RecognizeResponse: The response containing the
            transcription results with word-level confidence.
    """
    # Instantiates a client
    client = SpeechClient()

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

    for result in response.results:
        print(f"Transcript: {result.alternatives[0].transcript}")

    return response
current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(output_file, "a") as f:
    f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + str(transcribe_word_level_confidence_v2(audio_file)))