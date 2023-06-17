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

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я бот, который преобразует речь в текст по ссылкам с файлами формата .wav\n\nДля помощи в использовании набери /help")

@dp.message_handler(commands=['help'])
async def send_welcome(message: types.Message):
    await message.reply("Доступные модели:\n- kz: казахская модель\n- en: английская модель\n- hi: индийская модель\n- enin: анг-инди модель\n\nПример запроса: http://example.com/audio.wav en")

@dp.message_handler(regexp=r'https?://.*\.(wav)\s+(kz|en|hi|enin)')  # проверка на соответствие URL, расширения файла и указания модели
async def process_audio_url(message: types.Message):
    audio_info = message.text.split()  # разделение текста на URL и модель
    audio_url = audio_info[0]
    model_name = audio_info[1]
    await message.reply("Загружаю и обрабатываю аудиофайл...")

    try:
        result_text = transcribe_audio_file(audio_url, model_name)  # вызов функции распознавания речи с указанием выбранной модели
        if result_text:
            await message.reply(f"Распознанный текст:\n\n{result_text}")
        else:
            await message.reply("Не удалось распознать речь в аудиофайле.")
    except Exception as e:
        await message.reply(f"Произошла ошибка при обработке аудиофайла. Пожалуйста, проверьте формат и кодировку файла и повторите попытку. Ошибка: {e}")

@dp.message_handler()
async def echo(message: types.Message):
    await message.reply("Пожалуйста, отправьте действительный URL аудиофайла (формат WAV).")

def transcribe_audio_file(audio_url, model_name):
    # Загрузка модели распознавания
    if model_name == "kz":
        model = Model("vosk-model-kz-0.15")
    elif model_name == "en":
        model = Model("vosk-model-en-us-0.22-lgraph")
    elif model_name == "hi":
        model = Model("vosk-model-small-hi-0.22")
    elif model_name == "enin":
        model = Model("vosk-model-small-en-in-0.4")

    sample_rate = 16000 # Параметры распознавания

    # Загрузка звукового файла
    response = requests.get(audio_url)
    audio_path = "audio.wav"
    with open(audio_path, "wb") as file:
        file.write(response.content)

    audio_data, sr = sf.read(audio_path)  # Чтение звуковых данных

    # Проверка параметров звука
    if sr != sample_rate:
        audio_data = sps.resample(audio_data, int(len(audio_data) * sample_rate / sr)) # Преобразование частоты дискретизации

    audio_data = (audio_data * 32768).astype(np.int16) # Преобразование битности
    rec = KaldiRecognizer(model, sample_rate) # Инициализация распознавателя
    rec.AcceptWaveform(audio_data.tobytes())  # Распознавание речи
    result = json.loads(rec.FinalResult()) # Получение результата распознавания
    transcription = result["text"] # Вывод распознанного текста

    return transcription

if __name__ == '__main__':
    from aiogram import executor
    executor.start_polling(dp, skip_updates=True)