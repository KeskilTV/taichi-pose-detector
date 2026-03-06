"""
Модуль для обработки видео файлов
"""
import cv2
from config import OUTPUT_DIR
import os


class VideoProcessor:
    """Класс для чтения и записи видео"""

    def __init__(self, input_path, output_filename):
        self.input_path = input_path
        self.output_path = os.path.join(OUTPUT_DIR, output_filename)
        self.cap = None
        self.out = None
        self.fps = 0
        self.width = 0
        self.height = 0

    def open(self):
        """Открывает видео файл"""
        self.cap = cv2.VideoCapture(self.input_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {self.input_path}")

        # Получаем параметры видео
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Настраиваем запись
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

        print(f"✓ Видео открыто: {self.width}x{self.height}, {self.fps} FPS")
        print(f"✓ Выходной файл: {self.output_path}")

    def process(self, frame_callback):
        """
        Обрабатывает видео кадр за кадром

        Args:
            frame_callback: Функция которая принимает кадр и возвращает обработанный кадр
        """
        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Обрабатываем кадр через callback функцию
            processed_frame = frame_callback(frame)

            # Записываем результат
            self.out.write(processed_frame)

            # Прогресс
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"→ Обработано: {frame_count}/{total_frames} ({progress:.1f}%)")

        print(f"✓ Обработка завершена: {frame_count} кадров")

    def close(self):
        """Закрывает файлы"""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()