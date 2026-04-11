"""
Рендеринг текста с поддержкой кириллицы
OpenCV не поддерживает Unicode, используем PIL
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def draw_text_cv2(image, text, position, font_size=20, color=(255, 255, 255), bg_color=None):
    """
    Рисует текст с поддержкой кириллицы на изображении

    Args:
        image: Кадр (BGR формат numpy array)
        text: Текст (поддерживает кириллицу)
        position: (x, y) координаты левого верхнего угла
        font_size: Размер шрифта в пикселях
        color: Цвет текста (R, G, B)
        bg_color: Цвет фона (R, G, B) или None для прозрачного

    Returns:
        image: Изображение с текстом
    """
    # Конвертируем BGR -> RGB для PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    # Попытка загрузить шрифт с поддержкой кириллицы
    font = None
    font_paths = [
        "arial.ttf",
        "Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/seguiemj.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]

    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue

    # Если не нашли - используем шрифт по умолчанию
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # Рисуем фон если указан
    if bg_color:
        # Примерная ширина текста
        bbox = draw.textbbox(position, text, font=font)
        padding = 5
        draw.rectangle(
            [position[0] - padding, position[1] - padding,
             bbox[2] + padding, bbox[3] + padding],
            fill=bg_color
        )

    # Рисуем текст
    draw.text(position, text, font=font, fill=color)

    # Конвертируем обратно в BGR для OpenCV
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image_bgr