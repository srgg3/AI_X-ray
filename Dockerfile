# Фиксация базового образа с поддержкой CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Установка Python и необходимых системных библиотек
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копирование и установка зависимостей
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Копирование исходного кода и директории с моделями
COPY double_bobble.py api.py ./
COPY models/ ./models/

# Открытие порта для API
EXPOSE 8000

# Запуск FastAPI сервера
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]