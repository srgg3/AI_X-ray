import os
import time
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torchvision import models, transforms

# Отключаем спам от pydicom
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

MODEL_DIR = Path("models")
OUTPUT_COLUMNS = [
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "violation_type",
    "processing_status",
    "time_of_processing",
]

MODEL_FILES = {
    "spine_quality": MODEL_DIR / "spine_resnet18_best.pt",
    "hip_quality": MODEL_DIR / "hip_resnet18_best.pt",
    "spine_artifacts": MODEL_DIR / "spine_artifacts_resnet18_best.pt",
    "hip_position_rotation": MODEL_DIR / "hip_position_rotation_resnet18_best.pt",
}


class DXAPipeline:
    def __init__(self, device, models_dict=None):
        self.device = device
        self.bundles = models_dict or {}

    def get_dcm_value(self, dcm, tag_name, default=np.nan):
        """Безопасное извлечение значения тега."""
        if tag_name in dcm:
            val = dcm[tag_name].value
            if val is None or str(val).strip() == '':
                return default
            return val
        return default

    def preprocess_image(self, dcm, image_size):
        """Продвинутая предобработка из inference.py + dicom_loader"""
        image = dcm.pixel_array.astype(np.float32)

        # Обработка многокадровости (если вдруг 3D матрица)
        if len(image.shape) == 3:
            image = image[0]  # Берем первый фрейм

        # Инверсия, если необходимо
        if self.get_dcm_value(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
            image = image.max() - image

        # Нормализация по перцентилям (отрезаем шумы детекторов)
        lo, hi = np.percentile(image, [1, 99])
        if hi <= lo:
            lo, hi = float(image.min()), float(image.max())

        image = np.clip(image, lo, hi)
        if hi > lo:
            image = (image - lo) / (hi - lo)
        else:
            image = np.zeros_like(image)

        image = (image * 255).astype(np.uint8)

        # PyTorch трансформации
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)

    @torch.no_grad()
    def predict_probability(self, bundle_name, dcm):
        if bundle_name not in self.bundles:
            return 0.0  # Если модель не загружена, возвращаем 0

        bundle = self.bundles[bundle_name]
        x = self.preprocess_image(dcm, bundle["image_size"]).to(self.device)
        logit = bundle["model"](x).squeeze()
        return float(torch.sigmoid(logit).cpu().item())

    def detect_region_smart(self, dcm, study_uid, labels_df):
        """
        Умная детекция региона.
        Сначала ищем в разметке. Если нет - используем пропорции снимка.
        """
        # 1. Попытка найти в разметке
        if labels_df is not None and not labels_df.empty:
            match = labels_df[labels_df['study'] == study_uid]
            if not match.empty:
                # В экселе: если "Позвоночник" заполнен (не NaN), значит это поясница.
                # (Исходя из структуры, 0 или 1 означает наличие исследования)
                spine_val = match['Позвоночник'].values[0]
                if pd.notna(spine_val):
                    return "spine"
                else:
                    return "hip"

        # 2. Эвристика по соотношению сторон (более надежно, чем точные пиксели)
        # Поясница обычно имеет соотношение близкое к квадрату, бедро - более вытянутое.
        rows = int(self.get_dcm_value(dcm, "Rows", 0))
        cols = int(self.get_dcm_value(dcm, "Columns", 0))

        if cols == 0 or rows == 0:
            return "unknown"

        ratio = cols / rows
        # Эти пороги можно подстроить. Если ширина почти равна высоте -> позвоночник
        if ratio >= 1.0:
            return "spine"
        elif ratio < 1.0:
            return "hip"

        return "unknown"

    def process_file(self, path: Path, labels_df=None):
        start_time = time.perf_counter()

        base_result = {
            "path_to_study": str(path),
            "study_uid": "",
            "image_uid": "",
            "anatomical_region": "unknown",
            "quality_class": "",
            "violation_type": "",
            "processing_status": "",
            "time_of_processing": 0.0,  # ТЗ: Float, время выполнения
        }

        try:
            dcm = pydicom.dcmread(path)
            base_result["study_uid"] = str(self.get_dcm_value(dcm, "StudyInstanceUID", "Unknown_Study"))
            base_result["image_uid"] = str(self.get_dcm_value(dcm, "SOPInstanceUID", "Unknown_Image"))

            region = self.detect_region_smart(dcm, base_result["study_uid"], labels_df)
            base_result["anatomical_region"] = region

            if region == "unknown":
                base_result["processing_status"] = "Failure: unsupported anatomy"
                base_result["violation_type"] = "not_assessed"
                base_result["time_of_processing"] = round(time.perf_counter() - start_time, 4)
                return base_result

            # Инференс моделей
            if region == "spine":
                q = self.predict_probability("spine_quality", dcm)
                v = self.predict_probability("spine_artifacts", dcm)

                # Если моделей нет (тестовый запуск), ставим пороги по дефолту
                q_thr = self.bundles.get("spine_quality", {}).get("threshold", 0.5)
                v_thr = self.bundles.get("spine_artifacts", {}).get("threshold", 0.5)

                quality_class = int(q >= q_thr)
                if quality_class == 0:
                    violation = "no_violation"
                elif v >= v_thr:
                    violation = "artifacts"
                else:
                    violation = "quality_violation_unspecified"
            else:  # hip
                q = self.predict_probability("hip_quality", dcm)
                v = self.predict_probability("hip_position_rotation", dcm)

                q_thr = self.bundles.get("hip_quality", {}).get("threshold", 0.5)
                v_thr = self.bundles.get("hip_position_rotation", {}).get("threshold", 0.5)

                quality_class = int(q >= q_thr)
                if quality_class == 0:
                    violation = "no_violation"
                elif v >= v_thr:
                    violation = "position_rotation"
                else:
                    violation = "quality_violation_unspecified"

            base_result["quality_class"] = quality_class
            base_result["violation_type"] = violation
            base_result["processing_status"] = "Success"

        except Exception as e:
            base_result["processing_status"] = f"Failure: {type(e).__name__}: {e}"
            base_result["violation_type"] = "not_assessed"

        base_result["time_of_processing"] = round(time.perf_counter() - start_time, 4)
        return base_result


def load_pytorch_model(path, device):
    """Функция загрузки из оригинального inference.py"""
    if not path.exists():
        print(f"ВНИМАНИЕ: Не найдена модель: {path}. Инференс будет пропущен.")
        return None

    checkpoint = torch.load(path, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        image_size = int(checkpoint.get("image_size", 224))
        threshold = float(checkpoint.get("threshold", 0.5))
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        image_size = int(checkpoint.get("image_size", 224))
        threshold = float(checkpoint.get("threshold", 0.5))
    else:
        state_dict = checkpoint
        image_size = 224
        threshold = 0.5

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {"model": model, "image_size": image_size, "threshold": threshold}


def find_dicoms(input_path):
    input_path = Path(input_path)
    if input_path.is_file() and input_path.suffix.lower() == ".dcm":
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Путь не существует: {input_path}")

    # Ищем файлы рекурсивно
    dcm = sorted(p for p in input_path.rglob("*.dcm") if p.is_file())
    if dcm: return dcm
    # Если расширения нет, берем все файлы, кроме скрытых и логов
    return sorted(p for p in input_path.rglob("*") if
                  p.is_file() and not p.name.startswith('.') and p.suffix not in ['.csv', '.xlsx', '.py'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DXA Unified Pipeline")
    #НАДО ПУТЬ ПОМЕНЯТЬ
    parser.add_argument("--input", default="dataset", help="Папка с DICOM")
    parser.add_argument("--labels", default="разметка.xlsx", help="Excel файл с разметкой")
    parser.add_argument("--output", default="final_submission.csv", help="CSV результата")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Загрузка разметки
    labels_df = None
    if Path(args.labels).exists():
        labels_df = pd.read_excel(args.labels)
        print(f"Загружена разметка: {len(labels_df)} записей.")
    else:
        print("Файл разметки не найден, используем эвристику.")

    # Загрузка моделей (если их нет в папке models, скрипт не упадет, а просто проставит классы по нулям)
    bundles = {}
    for name, path in MODEL_FILES.items():
        model_bundle = load_pytorch_model(path, device)
        if model_bundle:
            bundles[name] = model_bundle

    # Инициализация пайплайна
    pipeline = DXAPipeline(device, bundles)
    files = find_dicoms(args.input)
    print(f"Найдено DICOM файлов: {len(files)}")

    results = []
    total_start = time.perf_counter()

    for i, path in enumerate(files, 1):
        row = pipeline.process_file(path, labels_df)
        results.append(row)

        # Вывод прогресса
        print(f"[{i}/{len(files)}] {path.name} | {row['anatomical_region']} | "
              f"Q:{row['quality_class']} | {row['violation_type']} | "
              f"{row['processing_status']} | {row['time_of_processing']}s")

    # Сохранение по формату ТЗ
    result_df = pd.DataFrame(results)
    for col in OUTPUT_COLUMNS:
        if col not in result_df.columns:
            result_df[col] = ""
    result_df = result_df[OUTPUT_COLUMNS]

    result_df.to_csv(args.output, index=False, encoding="utf-8-sig", sep=";")  # Разделитель ; лучше для русского экселя

    print("\n=== ГОТОВО ===")
    print(f"Файл сохранен: {args.output}")
    print(f"Время выполнения: {time.perf_counter() - total_start:.2f} сек.")