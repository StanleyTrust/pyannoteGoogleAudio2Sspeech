from speechmatics.models import ConnectionSettings, BatchTranscriptionConfig
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError
from datetime import datetime

API_KEY = "2530UB4SIYLe3CweZYcS01c3sjOKRiyb"
PATH_TO_FILE = "src/exmpl.mp3"
LANGUAGE = "auto"

def cur_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print('start: ' + cur_time())
connection_settings = ConnectionSettings(url="https://asr.api.speechmatics.com/v2", auth_token=API_KEY, semaphore_timeout_seconds=500, ping_timeout_seconds=500)  # Timeout in seconds

# Define transcription parameters
conf = {
    "type": "transcription",
    "transcription_config": {
        "language": LANGUAGE,
        "diarization": "speaker"
    },
    "language_identification_config": {
        "expected_languages": ["uk", "ru", "en"],
        "low_confidence_action": "use_default_language",
        "default_language": "uk"
    }
}

# Open the client using a context manager
with BatchClient(connection_settings) as client:
    try:
        job_id = client.submit_job(
            audio=PATH_TO_FILE,
            transcription_config=conf,
        )
        print(f'job {job_id} submitted successfully, waiting for transcript')

        # Note that in production, you should set up notifications instead of polling.
        # Notifications are described here: https://docs.speechmatics.com/features-other/notifications
        transcript = client.wait_for_completion(job_id, transcription_format='txt')
        # To see the full output, try setting transcription_format='json-v2'.
        print(transcript)
    except HTTPStatusError as e:
        if e.response.status_code == 401:
            print('Invalid API key - Check your API_KEY at the top of the code!')
        elif e.response.status_code == 400:
            print(e.response.json()['detail'])
        else:
            raise e     
print('finished: ' + cur_time())
