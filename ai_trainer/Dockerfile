# Dockerfile для AI Trainer Platform
FROM python:3.11-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Установка системных зависимостей для компиляции пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем все зависимости из requirements.txt
# Для CPU версии PyTorch (если нужен GPU, см. комментарий ниже)
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Открываем порт для Gradio (по умолчанию 7860)
EXPOSE 7860

# Переменная окружения для хоста Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Запуск приложения
CMD ["python", "main.py"]

# ============================================================
# ПРИМЕЧАНИЕ ДЛЯ GPU ПОДДЕРЖКИ:
# Если у вас есть NVIDIA GPU и вы хотите использовать CUDA:
# 1. Используйте базовый образ с CUDA: 
#    FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
# 2. Установите NVIDIA Container Toolkit на хост-машине
# 3. Запускайте контейнер с флагом --gpus all
# ============================================================
