"""
Конфигурация проекта TaiChi Pose Detector
"""
import os

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'input_videos')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_videos')

# Создаём папки если не существуют
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Настройки отрисовки скелета
POSE_SETTINGS = {
    'line_thickness': 1,          # Толщина линий
    'point_radius': 1,            # Размер точек суставов
    'line_color': (200, 150, 50), # Цвет линий (BGR)
    'point_color': (255, 200, 100) # Цвет точек (BGR)
}

# Настройки детекции
POSE_DETECTION_SETTINGS = {
    'model_complexity': 1,        # 0=быстро, 1=баланс, 2=точно
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}