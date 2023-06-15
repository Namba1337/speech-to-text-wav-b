import requests
import soundfile as sf
import scipy.signal as sps
from vosk import Model, KaldiRecognizer
import numpy as np
import json

# Загрузка модели распознавания для казахского языка
model = Model("vosk-model-kz-0.15")

# Параметры распознавания
sample_rate = 16000

# Ввод URL звукового файла WAV
url = input("Введите URL звукового файла WAV: ")

# Загрузка звукового файла
response = requests.get(url)
audio_path = "audio.wav"
with open(audio_path, "wb") as file:
    file.write(response.content)

# Чтение звуковых данных
audio_data, sr = sf.read(audio_path)

# Проверка параметров звука
if sr != sample_rate:
    # Преобразование частоты дискретизации
    audio_data = sps.resample(audio_data, int(len(audio_data) * sample_rate / sr))

# Преобразование битности
audio_data = (audio_data * 32768).astype(np.int16)

# Инициализация распознавателя
rec = KaldiRecognizer(model, sample_rate)

# Распознавание речи
rec.AcceptWaveform(audio_data.tobytes())

# Получение результата распознавания
result = json.loads(rec.FinalResult())

# Вывод распознанного текста
transcription = result["text"]
print("Текст распознавания:")
print(transcription)