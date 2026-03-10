"""
🥋 TaiChi Pose Detector
Главный файл запуска программы

Режимы работы:
  1. 2D скелет на видео (отрисовка поверх оригинала)
  2. 3D анимация скелета (вращающаяся модель)
  3. Оба режима одновременно

Автор: Васильев Кэскил
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
from datetime import datetime

# Создаём папки для видео
os.makedirs('input_videos', exist_ok=True)
os.makedirs('output_videos', exist_ok=True)


# ============================================================
# КЛАСС 1: Обработка видео и 2D скелет
# ============================================================

class VideoProcessor2D:
    """Обработка видео с отрисовкой 2D скелета поверх кадра"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """Обрабатывает один кадр и рисует скелет"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 200, 100),
                    thickness=1,
                    circle_radius=1
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(200, 150, 50),
                    thickness=1
                )
            )

        return image_bgr

    def process_video(self, input_path, output_path):
        """Обрабатывает всё видео"""
        cap = cv2.VideoCapture(input_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 Видео: {width}x{height}, {fps} FPS, {total_frames} кадров")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            out.write(processed_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  → Обработано: {frame_count}/{total_frames} ({progress:.1f}%)")

        cap.release()
        out.release()
        print(f"✓ 2D видео сохранено: {output_path}")

    def close(self):
        self.pose.close()


# ============================================================
# КЛАСС 2: Извлечение поз и 3D анимация
# ============================================================

class PoseExtractor3D:
    """Извлечение 3D координат и создание 3D анимации"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        self.connections = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
            (3, 7), (6, 8), (9, 10),
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32)
        ]

    def extract_poses_from_video(self, video_path):
        """Извлекает все 3D позы из видео"""
        cap = cv2.VideoCapture(video_path)
        poses = []

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])
                poses.append(np.array(landmarks))

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  → Извлечено поз: {len(poses)}")

        cap.release()
        print(f"✓ Всего извлечено: {len(poses)} поз")
        return poses

    def animate_poses_3d(self, poses, output_path, fps=20):
        """Создаёт 3D анимацию"""
        if len(poses) == 0:
            print("✗ Нет данных для анимации")
            return False

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        all_points = np.vstack(poses)
        margin = 0.1
        ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

        def update(frame):
            ax.clear()
            pose = poses[frame]

            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='red', s=50)

            for i, j in self.connections:
                ax.plot([pose[i, 0], pose[j, 0]],
                       [pose[i, 1], pose[j, 1]],
                       [pose[i, 2], pose[j, 2]], 'b-', linewidth=2)

            ax.set_title(f'Frame {frame + 1} / {len(poses)}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
            ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
            ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

            return ax,

        print("🎬 Создание 3D анимации...")
        ani = animation.FuncAnimation(fig, update, frames=len(poses),
                                       interval=1000/fps, blit=False)

        try:
            ani.save(output_path, writer='ffmpeg', fps=fps)
            print(f"✓ 3D видео сохранено: {output_path}")
            plt.close()
            return True
        except Exception as e:
            print(f"⚠ Ошибка MP4: {e}")
            try:
                html_path = output_path.replace('.mp4', '.html')
                ani.save(html_path, writer='html')
                print(f"✓ 3D HTML сохранён: {html_path}")
                print("  Откройте в браузере!")
                plt.close()
                return True
            except Exception as e2:
                print(f"⚠ Ошибка HTML: {e2}")
                plt.close()
                return False

    def close(self):
        self.pose.close()


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def list_video_files(folder='input_videos'):
    """Возвращает список видео файлов в папке"""
    if not os.path.exists(folder):
        os.makedirs(folder)
        return []

    files = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    return files


def generate_output_filename(input_path, suffix):
    """Генерирует имя выходного файла с датой и временем"""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join('output_videos', f"{base_name}_{suffix}_{timestamp}.mp4")


# ============================================================
# РЕЖИМЫ РАБОТЫ
# ============================================================

def mode_2d_skeleton(input_path):
    """Режим 1: 2D скелет на видео"""
    print("\n" + "=" * 60)
    print("🎬 Режим 1: 2D скелет на видео")
    print("=" * 60)

    processor = VideoProcessor2D()
    output_path = generate_output_filename(input_path, 'skeleton_2d')

    try:
        processor.process_video(input_path, output_path)
        print(f"\n✅ Готово! Файл: {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        return False
    finally:
        processor.close()


def mode_3d_animation(input_path):
    """Режим 2: 3D анимация скелета"""
    print("\n" + "=" * 60)
    print("🎭 Режим 2: 3D анимация скелета")
    print("=" * 60)

    extractor = PoseExtractor3D()
    output_path = generate_output_filename(input_path, 'skeleton_3d')

    try:
        poses = extractor.extract_poses_from_video(input_path)

        if len(poses) == 0:
            print("✗ Позы не обнаружены!")
            print("  Проверьте: человек виден в полный рост, хорошее освещение")
            return False

        success = extractor.animate_poses_3d(poses, output_path, fps=20)

        if success:
            print(f"\n✅ Готово! Файл: {output_path}")
            return True
        else:
            print("\n⚠ Не удалось сохранить анимацию")
            return False

    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        return False
    finally:
        extractor.close()


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    """Основная функция программы"""
    print("=" * 60)
    print("🥋 TaiChi Pose Detector")
    print("   Анализ движений Тайцзи с компьютерным зрением")
    print("=" * 60)
    print(f"\n👤 Автор: Васильев Кэскил")
    print(f"📧 Email: zrengemlab@gmail.com")
    print()

    # Поиск видео файлов
    video_files = list_video_files('input_videos')

    if not video_files:
        print("⚠ В папке input_videos нет видео файлов!")
        print("\n💡 Как добавить видео:")
        print("  1. Скопируйте видео файл в папку: input_videos/")
        print("  2. Или укажите полный путь к видео при запуске")
        print("\nПример:")
        print("  python main.py C:/Videos/my_video.mp4")
        print()

        # Предлагаем ввести путь вручную
        manual_path = input("Или введите полный путь к видео (Enter для выхода): ").strip()

        if not manual_path or not os.path.exists(manual_path):
            print("✗ Файл не найден. Выход.")
            return

        video_files = [manual_path]
        print(f"✓ Используется: {manual_path}")

    # Показать найденные файлы
    print("📁 Найденные видео файлы:")
    for i, file in enumerate(video_files, 1):
        print(f"  {i}. {file}")
    print()

    # Выбор файла
    if len(video_files) == 1:
        selected_file = video_files[0]
        print(f"✓ Выбран файл: {selected_file}")
    else:
        while True:
            try:
                choice = input(f"Выберите номер файла (1-{len(video_files)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(video_files):
                    selected_file = video_files[idx]
                    print(f"✓ Выбран файл: {selected_file}")
                    break
                else:
                    print("⚠ Неверный номер")
            except ValueError:
                print("⚠ Введите число")

    # Полный путь
    if os.path.exists(selected_file):
        input_path = selected_file
    else:
        input_path = os.path.join('input_videos', selected_file)

    print()

    # Выбор режима
    print("Выберите режим работы:")
    print("  1. 2D скелет на видео (отрисовка поверх оригинала)")
    print("  2. 3D анимация скелета (вращающаяся 3D модель)")
    print("  3. Оба режима одновременно")
    print()

    mode_choice = input("Ваш выбор (1/2/3): ").strip()

    if mode_choice not in ['1', '2', '3']:
        print("✗ Неверный выбор. Выход.")
        return

    print(f"\n✓ Режим: {'2D + 3D' if mode_choice == '3' else '2D' if mode_choice == '1' else '3D'}")
    print()

    # Выполнение
    success = False

    if mode_choice == '1':
        success = mode_2d_skeleton(input_path)
    elif mode_choice == '2':
        success = mode_3d_animation(input_path)
    elif mode_choice == '3':
        success1 = mode_2d_skeleton(input_path)
        print()
        success2 = mode_3d_animation(input_path)
        success = success1 and success2

    # Финал
    print("\n" + "=" * 60)
    if success:
        print("✅ Все операции завершены успешно!")
        print(f"📁 Результаты в папке: output_videos/")

        open_folder = input("\n📂 Открыть папку с результатами? (y/n): ").lower()
        if open_folder == 'y':
            os.startfile('output_videos')
    else:
        print("⚠ Завершено с ошибками")
    print("=" * 60)


if __name__ == "__main__":
    main()