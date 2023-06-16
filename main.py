import requests
import soundfile as sf
import scipy.signal as sps
from vosk import Model, KaldiRecognizer
import numpy as np
import json

import logging
from config import TOKEN_API
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware


logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN_API)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Отправь мне URL аудиофайла (формат WAV), и я распознаю речь в нем.")

@dp.message_handler(regexp=r'https?://.*\.(wav)')  # проверка на соответствие URL и расширения файла
async def process_audio_url(message: types.Message):
    audio_url = message.text
    await message.reply("Загружаю и обрабатываю аудиофайл...")

    try:
        result_text = transcribe_audio_file(audio_url)  # вызов функции распознавания речи
        if result_text:
            await message.reply(f"Распознанный текст:\n\n{result_text}")
        else:
            await message.reply("Не удалось распознать речь в аудиофайле.")
    except Exception as e:
        await message.reply(f"Произошла ошибка при обработке аудиофайла. Пожалуйста, проверьте формат и кодировку файла и повторите попытку. Ошибка: {e}")

@dp.message_handler()
async def echo(message: types.Message):
    await message.reply("Пожалуйста, отправьте действительный URL аудиофайла (формат WAV).")

def transcribe_audio_file(audio_url):
    # Загрузка модели распознавания для казахского языка
    model = Model("vosk-model-kz-0.15")

    # Параметры распознавания
    sample_rate = 16000

    # Загрузка звукового файла
    response = requests.get(audio_url)
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

    return transcription

if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)