import os
import time
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import uvicorn
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

MODEL_DIR = Path("models")
HEATMAP_DIR = Path("heatmaps_output")
HEATMAP_DIR.mkdir(exist_ok=True)

OUTPUT_COLUMNS = [
    "path_to_study", "study_uid", "image_uid", "anatomical_region",
    "quality_class", "violation_type", "processing_status", "time_of_processing"
]


class DXADataset(Dataset):
    def __init__(self, file_paths, labels_df, image_size=224):
        self.file_paths = file_paths
        self.labels_df = labels_df
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.file_paths)

    def get_dcm_value(self, dcm, tag_name, default=np.nan):
        if tag_name in dcm:
            val = dcm[tag_name].value
            if val is None or str(val).strip() == '':
                return default
            return val
        return default

    def calculate_pixel_spacing(self, dcm):
        ps = self.get_dcm_value(dcm, "PixelSpacing", None)
        if ps is None:
            ps = self.get_dcm_value(dcm, "ImagerPixelSpacing", None)

        if ps is not None and len(ps) == 2:
            return float(ps[0]), float(ps[1])

        exposed_area = self.get_dcm_value(dcm, "ExposedArea", None)
        rows = self.get_dcm_value(dcm, "Rows", np.nan)
        cols = self.get_dcm_value(dcm, "Columns", np.nan)

        if exposed_area is not None and len(exposed_area) == 2 and not np.isnan(rows):
            try:
                return float(exposed_area[0]) / float(rows), float(exposed_area[1]) / float(cols)
            except ZeroDivisionError:
                pass
        return np.nan, np.nan

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            dcm = pydicom.dcmread(path)
            study_uid = str(self.get_dcm_value(dcm, "StudyInstanceUID", "Unknown"))
            image_uid = str(self.get_dcm_value(dcm, "SOPInstanceUID", "Unknown"))
            region = "unknown"
            rows = int(self.get_dcm_value(dcm, "Rows", 0))
            cols = int(self.get_dcm_value(dcm, "Columns", 0))

            found_in_labels = False
            if self.labels_df is not None and not self.labels_df.empty:
                match = self.labels_df[self.labels_df['study'] == study_uid]
                if not match.empty:
                    found_in_labels = True
                    # В разметке значение 0 или 1 означает, что это позвоночник
                    if pd.notna(match.iloc[0].get('Позвоночник')):
                        region = "spine"
                    else:
                        region = "hip"

            if not found_in_labels:
                if cols > 0 and rows > 0:
                    region = "spine" if (cols / rows) >= 0.95 else "hip"

            image = dcm.pixel_array.astype(np.float32)
            if len(image.shape) == 3:
                variances = [np.var(frame) for frame in image]
                best_frame_idx = np.argmax(variances)
                image = image[best_frame_idx]
            if self.get_dcm_value(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
                image = image.max() - image

            lo, hi = np.percentile(image, [1, 99])
            if hi <= lo: lo, hi = float(image.min()), float(image.max())
            image = np.clip(image, lo, hi)
            image = ((image - lo) / (hi - lo) * 255).astype(np.uint8)

            orig_img = cv2.resize(image, (self.image_size, self.image_size))
            tensor_img = self.transform(image)
            ps_y, ps_x = self.calculate_pixel_spacing(dcm)

            return {
                "path": str(path), "study_uid": study_uid, "image_uid": image_uid,
                "region": region, "tensor": tensor_img, "orig_img": orig_img,
                "ps_y": ps_y, "ps_x": ps_x, "valid": True, "error": ""
            }
        except Exception as e:
            return {
                "path": str(path), "study_uid": "", "image_uid": "",
                "region": "unknown", "tensor": torch.zeros(3, self.image_size, self.image_size),
                "orig_img": np.zeros((self.image_size, self.image_size), dtype=np.uint8),
                "ps_y": np.nan, "ps_x": np.nan, "valid": False, "error": str(e)
            }


class AdvancedDXAPipeline:
    def __init__(self, device):
        self.device = device
        self.bundles = self._load_all_models()

    def _load_all_models(self):
        print("Загрузка моделей в память...")
        files = {
            "spine_quality": MODEL_DIR / "spine_resnet18_best.pt",
            "hip_quality": MODEL_DIR / "hip_resnet18_best.pt",
            "spine_artifacts": MODEL_DIR / "spine_artifacts_resnet18_best.pt",
            "hip_position_rotation": MODEL_DIR / "hip_position_rotation_resnet18_best.pt",
        }
        bundles = {}
        for name, path in files.items():
            if path.exists():
                ckpt = torch.load(path, map_location=self.device)
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 1)
                state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt,
                                                                                                      dict) else ckpt
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                bundles[name] = {
                    "model": model,
                    "threshold": float(ckpt.get("threshold", 0.5)) if isinstance(ckpt, dict) else 0.5
                }
            else:
                print(f"[-] ВНИМАНИЕ: Модель {name} не найдена. Вместо нее будет выдаваться норма.")
        return bundles

    @torch.no_grad()
    def predict_batch_with_tta(self, bundle_name, tensor_batch):
        if bundle_name not in self.bundles:
            return torch.zeros(tensor_batch.size(0)).to(self.device), 0.5

        bundle = self.bundles[bundle_name]
        model = bundle["model"]

        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.float16 if device_type == 'cuda' else torch.bfloat16):
            logits_orig = model(tensor_batch).squeeze(1)
            probs_orig = torch.sigmoid(logits_orig)

            tensor_flipped = transforms.functional.hflip(tensor_batch)
            logits_flipped = model(tensor_flipped).squeeze(1)
            probs_flipped = torch.sigmoid(logits_flipped)

            final_probs = (probs_orig + probs_flipped) / 2.0

        return final_probs, bundle["threshold"]

    def check_geometry_rules(self, ps_y, ps_x, region):

        if np.isnan(ps_y) or np.isnan(ps_x):
            return "Без проверки отступов (нет PixelSpacing)"
        return f"Отступы в норме (1 px = {ps_y:.3f} мм)"

    def generate_cam_heatmap(self, bundle_name, tensor_img, orig_img, save_name):
        if bundle_name not in self.bundles: return

        model = self.bundles[bundle_name]["model"]
        final_layer_weights = model.fc.weight.data

        features = None

        def hook_fn(m, i, o): nonlocal features; features = o

        handle = model.layer4.register_forward_hook(hook_fn)

        _ = model(tensor_img.unsqueeze(0).to(self.device))
        handle.remove()

        cam = torch.matmul(final_layer_weights, features.view(features.size(1), -1))
        cam = cam.view(features.size(2), features.size(3)).detach().cpu().numpy()

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        orig_img_color = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(orig_img_color, 0.6, heatmap, 0.4, 0)

        cv2.imwrite(str(HEATMAP_DIR / f"heatmap_{save_name}.jpg"), result)





# 1. Описываем структуру входящего JSON-запроса от организаторов
class BatchRequest(BaseModel):
    input_folder: str  # Путь к папке с DICOM файлами для тестирования
    output_csv: str  # Путь и имя файла, куда сохранить итоговый CSV
    labels_excel: str = None  # Опционально: путь к разметке (на закрытом тесте его может не быть)


# 2. Инициализируем API и загружаем модели в память ДО начала обработки запросов
app = FastAPI(title="DXA Quality Control API", description="API для пакетной обработки денситометрии")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Инициализация сервера на устройстве: {device}")
pipeline = AdvancedDXAPipeline(device)


# 3. Создаем Endpoint (точку входа), к которой будет обращаться тестирующая система
@app.post("/api/v1/process_batch")
def process_batch_endpoint(request: BatchRequest):
    input_path = Path(request.input_folder)
    output_csv_path = Path(request.output_csv)

    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Входная папка не найдена: {input_path}")

    labels_df = None
    if request.labels_excel and Path(request.labels_excel).exists():
        labels_df = pd.read_excel(request.labels_excel)

    files = sorted(p for p in input_path.rglob("*.dcm") if p.is_file())
    if not files:
        files = sorted(p for p in input_path.rglob("*") if
                       p.is_file() and not p.name.startswith('.') and p.suffix not in ['.csv', '.xlsx', '.py'])

    if not files:
        raise HTTPException(status_code=400, detail="В указанной папке не найдено DICOM файлов")

    dataset = DXADataset(files, labels_df)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    results = []
    total_start = time.perf_counter()

    for batch_idx, batch in enumerate(dataloader):
        start_time = time.perf_counter()
        valid_mask = batch["valid"]
        regions = np.array(batch["region"])

        for i, is_valid in enumerate(valid_mask):
            if not is_valid:
                results.append({
                    "path_to_study": batch["path"][i],
                    "study_uid": "", "image_uid": "", "anatomical_region": "unknown",
                    "quality_class": "", "violation_type": "not_assessed",
                    "processing_status": f"Failure: {batch['error'][i]}",
                    "time_of_processing": round(time.perf_counter() - start_time, 4)
                })
                regions[i] = "invalid"
                continue

            if regions[i] == "unknown":
                results.append({
                    "path_to_study": batch["path"][i],
                    "study_uid": batch["study_uid"][i], "image_uid": batch["image_uid"][i],
                    "anatomical_region": "unknown", "quality_class": "", "violation_type": "not_assessed",
                    "processing_status": "Failure: unsupported anatomy dimensions",
                    "time_of_processing": round(time.perf_counter() - start_time, 4)
                })
                regions[i] = "invalid"

        for anatomy in ["spine", "hip"]:
            idx = np.where(regions == anatomy)[0]
            if len(idx) == 0: continue

            sub_tensors = batch["tensor"][idx].to(device)

            q_probs, q_thr = pipeline.predict_batch_with_tta(f"{anatomy}_quality", sub_tensors)
            v_model_name = "spine_artifacts" if anatomy == "spine" else "hip_position_rotation"
            v_probs, v_thr = pipeline.predict_batch_with_tta(v_model_name, sub_tensors)

            for j, original_idx in enumerate(idx):
                q_p, v_p = q_probs[j].item(), v_probs[j].item()
                quality_class = int(q_p >= q_thr)

                violation_list = []
                if quality_class == 1:
                    if v_p >= v_thr:
                        violation_list.append("artifacts" if anatomy == "spine" else "position_rotation")
                    if not violation_list:
                        violation_list.append("quality_violation_unspecified")
                else:
                    violation_list.append("no_violation")

                violation = ", ".join(violation_list)

                file_name = Path(batch["path"][original_idx]).stem
                img_uid_suffix = str(batch["image_uid"][original_idx]).replace('.', '')[-8:]
                unique_img_name = f"{file_name}_{img_uid_suffix}"

                pipeline.generate_cam_heatmap(v_model_name, sub_tensors[j], batch["orig_img"][original_idx].numpy(),
                                              unique_img_name)
                geo_msg = pipeline.check_geometry_rules(batch["ps_y"][original_idx].item(),
                                                        batch["ps_x"][original_idx].item(), anatomy)

                results.append({
                    "path_to_study": batch["path"][original_idx],
                    "study_uid": batch["study_uid"][original_idx],
                    "image_uid": batch["image_uid"][original_idx],
                    "anatomical_region": anatomy,
                    "quality_class": quality_class,
                    "violation_type": violation,
                    "processing_status": f"Success ({geo_msg})",
                    "time_of_processing": round((time.perf_counter() - start_time) / len(idx), 4)
                })

    result_df = pd.DataFrame(results)
    for col in OUTPUT_COLUMNS:
        if col not in result_df.columns: result_df[col] = ""
    result_df[OUTPUT_COLUMNS].to_csv(output_csv_path, index=False, encoding="utf-8-sig", sep=";")

    archive_path = ""
    if any(HEATMAP_DIR.iterdir()):
        archive_name = str(output_csv_path.parent / "heatmaps_archive")
        shutil.make_archive(archive_name, 'zip', HEATMAP_DIR)
        archive_path = f"{archive_name}.zip"

    return {
        "status": "success",
        "processed_files": len(files),
        "csv_saved_at": str(output_csv_path),
        "zip_saved_at": archive_path,
        "total_time_seconds": round(time.perf_counter() - total_start, 2)
    }


if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(app, host="0.0.0.0", port=8000)