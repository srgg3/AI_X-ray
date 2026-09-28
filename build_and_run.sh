#!/bin/bash
set -e

echo "Сборка Docker-образа..."
docker build -t dxa-pipeline-api .

echo "Запуск контейнера..."
# Флаг --gpus all пробрасывает GPU в контейнер
# Директории монтируются (через -v) для локального доступа к данным без передачи во внешние системы
docker run -d --name dxa-api-container \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/dataset:/app/dataset \
  -v $(pwd)/output:/app/output \
  dxa-pipeline-api

echo "API запущено и доступно на порту 8000."
echo "Проверка состояния: curl http://localhost:8000/health"
echo "Пример вызова пакетной обработки:"
echo 'curl -X POST "http://localhost:8000/api/v1/process_batch" -H "Content-Type: application/json" -d "{\"input_dir\": \"/app/dataset\", \"output_csv\": \"/app/output/submission.csv\"}"'