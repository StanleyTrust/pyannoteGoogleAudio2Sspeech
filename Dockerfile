FROM python:3.10

WORKDIR /app

RUN apt update \
    && apt install -y ffmpeg build-essential python3-dev libsndfile1


# Устанавливаем зависимости
RUN pip install --upgrade pip \
    && pip install setuptools transformers accelerate>=0.26.0 datasets pydub \
    && pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install pyannote.audio noisereduce librosa soundfile \
    && pip install --upgrade google-cloud-speech



# Копируем весь проект в контейнер
COPY . /app/