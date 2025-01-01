# main.py (python example)

import os
from datetime import datetime
import json  # Добавляем модуль JSON для работы с текстом

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

# Path to the audio file
AUDIO_FILE = "src/exmpl.mp3"

# Файл для збереження результатів
output_file = "res.json"

def cur_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\n'

def pr_cur_time():
    print('time: ' + cur_time() + '\n')

def main():
    try:
        pr_cur_time()
        # STEP 1 Create a Deepgram client using the API key
        deepgram = DeepgramClient('492e5af9fb3d428f20cc4d75097fe55cb95361e5')

        with open(AUDIO_FILE, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        #STEP 2: Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-2",
            language="uk",
            smart_format=True,
            punctuate=True,
            diarize=True,
        )
        
        # PrerecordedOptions(
        #     model="nova-2",
        #     smart_format=True,
        # )

        # STEP 3: Call the transcribe_file method with the text payload and options

        pr_cur_time()

        response = deepgram.listen.rest.v("1").transcribe_file(payload, options,  timeout=300)

        channels = response.results.channels
        channel = channels[0] if channels else None
        alternatives = channel.alternatives if channel else None
        alternative = alternatives[0] if alternatives else None

        if alternative:
            print(alternative.paragraphs.transcript)
        else:
            print('not found')

        

        response_json = json.dumps(response.to_dict(), indent=4, ensure_ascii=False)

        # with open(output_file, "w") as f:
        #     f.write(response_json)

        # STEP 4: Print the response
        print(f"Success!!!" + cur_time())

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()