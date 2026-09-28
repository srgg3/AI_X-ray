import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
import pandas as pd

# Импорт вашего пайплайна и вспомогательных функций
from double_bobble import DXAPipeline, load_pytorch_model, find_dicoms, MODEL_FILES, OUTPUT_COLUMNS

app = FastAPI(title="DXA Batch Processing API")

# Глобальные переменные для хранения состояния
pipeline = None
device = None

class BatchRequest(BaseModel):
    input_dir: str
    labels_path: str = None
    output_csv: str = "final_submission.csv"

class BatchResponse(BaseModel):
    message: str
    task_status: str
    processed_files: int = 0
    processing_time_seconds: float = 0.0
    output_file: str = ""

@app.on_event("startup")
def load_models():
    global pipeline, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Инициализация API. Устройство: {device}")
    
    bundles = {}
    for name, path in MODEL_FILES.items():
        # Пути к моделям внутри контейнера
        container_path = Path("/app") / path
        model_bundle = load_pytorch_model(container_path, device)
        if model_bundle:
            bundles[name] = model_bundle
            
    pipeline = DXAPipeline(device, bundles)
    print("Модели успешно загружены в память.")

@app.post("/api/v1/process_batch", response_model=BatchResponse)
async def process_batch(request: BatchRequest):
    """
    API для пакетной обработки тестового набора данных.
    """
    input_path = Path(request.input_dir)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Указанная директория input_dir не найдена")

    try:
        files = find_dicoms(input_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска DICOM файлов: {e}")

    if not files:
        raise HTTPException(status_code=404, detail="DICOM файлы не найдены в указанной директории")

    labels_df = None
    if request.labels_path and Path(request.labels_path).exists():
        labels_df = pd.read_excel(request.labels_path)

    results = []
    total_start = time.perf_counter()

    for path in files:
        row = pipeline.process_file(path, labels_df)
        results.append(row)

    # Формирование и сохранение результатов
    result_df = pd.DataFrame(results)
    for col in OUTPUT_COLUMNS:
        if col not in result_df.columns:
            result_df[col] = ""
    result_df = result_df[OUTPUT_COLUMNS]
    
    output_file_path = Path(request.output_csv)
    result_df.to_csv(output_file_path, index=False, encoding="utf-8-sig", sep=";")

    total_time = time.perf_counter() - total_start

    return BatchResponse(
        message="Пакетная обработка успешно завершена",
        task_status="success",
        processed_files=len(files),
        processing_time_seconds=round(total_time, 2),
        output_file=str(output_file_path)
    )

@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(device)}