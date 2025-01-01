FROM python:3.10

WORKDIR /app

RUN apt update \
    && apt install -y ffmpeg build-essential python3-dev libsndfile1


# Устанавливаем зависимости
RUN pip install --upgrade pip \
    && pip install setuptools transformers accelerate>=0.26.0 datasets pydub \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install pyannote.audio noisereduce librosa soundfile \
    && pip install deepgram-sdk python-dotenv



# Копируем весь проект в контейнер
COPY . /app/