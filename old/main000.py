"""
TaiChi Pose Detector - Главная точка входа
Обнаружение позы и отрисовка скелета на видео
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from src.pose_detector import PoseDetector
from src.video_processor import VideoProcessor
from config import INPUT_DIR, OUTPUT_DIR
import os
from datetime import datetime


def select_video_file():
    """Открывает диалог выбора видео файла"""
    root = tk.Tk()
    root.withdraw()  # Скрываем главное окно

    file_path = filedialog.askopenfilename(
        title="Выберите видео файл",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ],
        initialdir=INPUT_DIR
    )

    root.destroy()
    return file_path


def generate_output_filename(input_path):
    """Генерирует имя выходного файла с датой и временем"""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_skeleton_{timestamp}.mp4"


def main():
    """Основная функция программы"""
    print("=" * 50)
    print("🥋 TaiChi Pose Detector")
    print("=" * 50)

    # Выбор файла
    input_path = select_video_file()

    if not input_path:
        print("✗ Файл не выбран. Выход.")
        return

    print(f"✓ Выбран файл: {input_path}")

    # Инициализация компонентов
    pose_detector = PoseDetector()
    output_filename = generate_output_filename(input_path)
    video_processor = VideoProcessor(input_path, output_filename)

    try:
        # Открытие видео
        video_processor.open()

        # Функция обработки одного кадра
        def process_frame(frame):
            processed_frame, _ = pose_detector.process_frame(frame)
            return processed_frame

        # Обработка всего видео
        print("\n⏳ Начало обработки...")
        video_processor.process(process_frame)

        print("\n" + "=" * 50)
        print(f"✅ Готово! Видео сохранено:")
        print(f"   {video_processor.output_path}")
        print("=" * 50)

        # Предложение открыть папку
        open_folder = input("\nОткрыть папку с результатом? (y/n): ").lower()
        if open_folder == 'y':
            os.startfile(OUTPUT_DIR)  # Windows
            # Для macOS используйте: os.system(f'open "{OUTPUT_DIR}"')
            # Для Linux: os.system(f'xdg-open "{OUTPUT_DIR}"')

    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        messagebox.showerror("Ошибка", str(e))

    finally:
        # Очистка ресурсов
        video_processor.close()
        pose_detector.close()


if __name__ == "__main__":
    main()