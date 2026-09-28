import os
import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Отключаем спам от pydicom про невалидные UID в датасете
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")


class DicomDataLoader:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def get_dcm_value(self, dcm, tag_name, default=np.nan):
        """Безопасное извлечение значения тега."""
        if tag_name in dcm:
            val = dcm[tag_name].value
            # Если тег есть, но он пустой (None или пустая строка)
            if val is None or str(val).strip() == '':
                return default
            return val
        return default

    def metadata_load(self, dicom_path: Path) -> dict:
        """Сбор расширенных метаданных и расчет недостающей геометрии."""
        try:
            dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)

            # Базовые идентификаторы
            study_uid = self.get_dcm_value(dcm, "StudyInstanceUID", "Unknown_Study")
            image_uid = self.get_dcm_value(dcm, "SOPInstanceUID", "Unknown_Image")
            body_part = str(self.get_dcm_value(dcm, "BodyPartExamined", "Unknown"))
            view_position = str(self.get_dcm_value(dcm, "ViewPosition", "Unknown"))

            # Технические параметры изображения
            rows = self.get_dcm_value(dcm, "Rows", np.nan)
            columns = self.get_dcm_value(dcm, "Columns", np.nan)
            bits_allocated = self.get_dcm_value(dcm, "BitsAllocated", np.nan)
            bits_stored = self.get_dcm_value(dcm, "BitsStored", np.nan)
            photo_interp = str(self.get_dcm_value(dcm, "PhotometricInterpretation", "Unknown"))

            # Параметры сканирования DXA
            num_frames = int(self.get_dcm_value(dcm, "NumberOfFrames", 1))
            total_exposures = self.get_dcm_value(dcm, "TotalNumberOfExposures", np.nan)
            exposed_area = self.get_dcm_value(dcm, "ExposedArea", None)

            # Умный поиск и расчет Pixel Spacing
            ps_y, ps_x = np.nan, np.nan
            spacing_source = "Missing"

            pixel_spacing = self.get_dcm_value(dcm, "PixelSpacing", None)
            imager_spacing = self.get_dcm_value(dcm, "ImagerPixelSpacing", None)

            if pixel_spacing is not None and len(pixel_spacing) == 2:
                ps_y, ps_x = float(pixel_spacing[0]), float(pixel_spacing[1])
                spacing_source = "PixelSpacing"
            elif imager_spacing is not None and len(imager_spacing) == 2:
                ps_y, ps_x = float(imager_spacing[0]), float(imager_spacing[1])
                spacing_source = "ImagerPixelSpacing"
            elif exposed_area is not None and len(exposed_area) == 2 and not np.isnan(rows) and not np.isnan(columns):
                # Расчет: размер области в мм делим на количество пикселей
                try:
                    ps_y = float(exposed_area[0]) / float(rows)
                    ps_x = float(exposed_area[1]) / float(columns)
                    spacing_source = "Calculated_from_ExposedArea"
                except ZeroDivisionError:
                    pass

            return {
                "path_to_study": str(dicom_path),
                "study_uid": str(study_uid),
                "image_uid": str(image_uid),
                "body_part": body_part,
                "view_position": view_position,
                "rows": rows,
                "columns": columns,
                "pixel_spacing_y": ps_y,
                "pixel_spacing_x": ps_x,
                "spacing_source": spacing_source,  # Откуда взяли масштаб
                "bits_allocated": bits_allocated,
                "bits_stored": bits_stored,
                "photometric_interp": photo_interp,
                "num_frames": num_frames,
                "total_exposures": total_exposures,
                "processing_status": "Success",
                "error": None
            }
        except Exception as e:
            return {
                "path_to_study": str(dicom_path),
                "processing_status": "Failure",
                "error": str(e)
            }

    def pixel_model(self, dicom_path: Path):
        """Извлекает пиксельную матрицу для CV моделей"""
        try:
            dcm = pydicom.dcmread(dicom_path)
            img_array = dcm.pixel_array

            # Нормализация для визуализации и нейросетей
            img_array = img_array.astype(np.float32)
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-8)

            num_frames = int(self.get_dcm_value(dcm, "NumberOfFrames", 1))
            return img_array, num_frames
        except Exception as e:
            print(f"Ошибка чтения пикселей {dicom_path}: {e}")
            return None, 0

    def build_eda(self) -> pd.DataFrame:
        """Собирает таблицу из метаданных"""
        records = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.lower().endswith('.dcm'):
                    file_path = Path(root) / file
                    meta = self.metadata_load(file_path)
                    records.append(meta)

        df = pd.DataFrame(records)
        return df

    def visualize_sample(self, dicom_path: Path):
        """Визуализация снимка"""
        img_array, num_frames = self.pixel_model(dicom_path)
        if img_array is None:
            return

        if num_frames > 1 and len(img_array.shape) == 3:
            fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
            for i in range(num_frames):
                axes[i].imshow(img_array[i], cmap='gray')
                axes[i].set_title(f"Frame {i + 1}")
                axes[i].axis('off')
            plt.show()
        else:
            plt.figure(figsize=(8, 8))
            plt.imshow(img_array, cmap='gray')
            plt.title("DICOM Image")
            plt.axis('off')
            plt.show()


# 1. Запуск
# Обязательно укажите ваш путь к разархивированной папке
loader = DicomDataLoader(r"/Users/whynot/PycharmProjects/lct/dataset")

# 2. Сбор и сохранение
df_metadata = loader.build_eda()
df_metadata.to_csv("dicom_metadata_extended.csv", index=False, encoding="utf-8")
print(f"Собрано {len(df_metadata)} записей. Датафрейм сохранен.")

# Выведем статистику по источникам масштаба пикселей
print("\nСтатистика поиска размера пикселей (Pixel Spacing):")
print(df_metadata['spacing_source'].value_counts())

# 3. Визуализация
if not df_metadata[df_metadata['processing_status'] == 'Success'].empty:
    sample_path = Path(df_metadata[df_metadata['processing_status'] == 'Success'].iloc[0]['path_to_study'])
    loader.visualize_sample(sample_path)