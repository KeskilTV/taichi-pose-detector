"""
Движок сравнения движений Тайцзи
ИСПРАВЛЕННАЯ ВЕРСИЯ: синхронизация видео + поз по DTW пути
С поддержкой ручной обрезки кадров начала
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from src.segment_detector import SegmentDetector
from src.dtw_aligner import DTWAligner
from src.text_renderer import draw_text_cv2


class ComparisonEngine:
    """Основной класс для сравнения двух видео"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.connections = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
            (3, 7), (6, 8), (9, 10),
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32)
        ]

        self.segment_detector = SegmentDetector(min_segment_length=150)
        self.dtw_aligner = DTWAligner(step=5, window_size=50)

    def extract_poses(self, video_path):
        """Извлекает все позы из видео"""
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
                    landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                poses.append(np.array(landmarks))
            else:
                poses.append(None)

            frame_count += 1

        cap.release()
        return poses

    def get_frame_at_index(self, cap, target_index, current_index, current_frame):
        """
        Получает кадр по индексу (перемотка видео)

        Args:
            cap: VideoCapture объект
            target_index: Нужный номер кадра
            current_index: Текущая позиция
            current_frame: Текущий кадр (кэш)

        Returns:
            frame, new_index
        """
        if target_index == current_index:
            return current_frame, current_index

        # Перемотка к нужному кадру
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ret, frame = cap.read()

        if ret:
            return frame, target_index
        else:
            return current_frame, current_index

    def find_movement_start(self, poses, threshold=0.02, min_frames=30):
        """
        Находит кадр, где начинается движение

        Args:
            poses: Список поз
            threshold: Порог изменения позы для детекции движения
            min_frames: Минимальное количество кадров движения

        Returns:
            start_frame: Номер кадра начала движения
        """
        for i in range(1, len(poses)):
            if poses[i] is None or poses[i - 1] is None:
                continue

            # Изменение позы
            dist = 0
            count = 0
            for j in range(len(poses[i])):
                if poses[i][j][3] > 0.5 and poses[i - 1][j][3] > 0.5:
                    dist += np.linalg.norm(poses[i][j][:2] - poses[i - 1][j][:2])
                    count += 1

            if count > 0 and (dist / count) > threshold:
                # Проверяем, что движение продолжается
                if i + min_frames < len(poses):
                    return i

        return 0  # Движение не найдено, начинаем с 0

    def trim_poses(self, poses, start_frame):
        """
        Обрезает позы с начала

        Args:
            poses: Список поз
            start_frame: Кадр начала

        Returns:
            trimmed_poses: Обрезанный список поз
        """
        if start_frame == 0:
            return poses
        print(f"  ✂️ Обрезка поз: {start_frame} кадров удалено")
        return poses[start_frame:]

    def calculate_balance(self, pose):
        """
        Расчет баланса (центр тяжести)

        Args:
            pose: Данные позы (33 точки)

        Returns:
            balance_score: Оценка баланса (0-1, где 1 = идеально)
        """
        if pose is None:
            return 0.0

        left_hip = pose[23][:2]
        right_hip = pose[24][:2]
        center_hip = (left_hip + right_hip) / 2

        left_ankle = pose[27][:2]
        right_ankle = pose[28][:2]
        center_feet = (left_ankle + right_ankle) / 2

        deviation = np.linalg.norm(center_hip - center_feet)
        balance_score = max(0, 1 - deviation * 2)
        return balance_score

    def calculate_pose_similarity(self, pose1, pose2):
        """
        Сравнение двух поз (процент совпадения)

        Args:
            pose1: Поза 1
            pose2: Поза 2

        Returns:
            similarity: Процент совпадения (0-100)
        """
        if pose1 is None or pose2 is None:
            return 0.0

        valid_points = []
        for i in range(len(pose1)):
            if pose1[i][3] > 0.5 and pose2[i][3] > 0.5:
                valid_points.append(i)

        if len(valid_points) == 0:
            return 0.0

        total_distance = 0
        for i in valid_points:
            dist = np.linalg.norm(pose1[i][:2] - pose2[i][:2])
            total_distance += dist

        avg_distance = total_distance / len(valid_points)
        similarity = max(0, 100 - avg_distance * 100)
        return similarity

    def draw_skeleton(self, frame, pose, color=(0, 255, 0), thickness=2):
        """
        Рисует скелет на кадре

        Args:
            frame: Кадр изображения
            pose: Данные позы
            color: Цвет скелета (BGR)
            thickness: Толщина линий

        Returns:
            image: Кадр с нарисованным скелетом
        """
        if pose is None:
            return frame

        image = frame.copy()

        for i, landmark in enumerate(pose):
            if landmark[3] > 0.5:
                x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
                cv2.circle(image, (x, y), 4, color, -1)

        for i, j in self.connections:
            if pose[i][3] > 0.5 and pose[j][3] > 0.5:
                x1, y1 = int(pose[i][0] * frame.shape[1]), int(pose[i][1] * frame.shape[0])
                x2, y2 = int(pose[j][0] * frame.shape[1]), int(pose[j][1] * frame.shape[0])
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)

        return image

    def create_comparison_video(self, master_path, student_path, output_path,
                                poses_master, poses_student, segments_master=None,
                                dtw_path=None, start_offset_master=0, start_offset_student=0):
        """
        Создает видео с сравнением (split-screen) + DTW синхронизация видео и поз

        Args:
            master_path: Путь к видео мастера
            student_path: Путь к видео ученика
            output_path: Путь для сохранения результата
            poses_master: Список поз мастера
            poses_student: Список поз ученика (выровненных по DTW)
            segments_master: Сегменты (формы) Тайцзи
            dtw_path: Путь DTW [(m_idx, s_idx), ...]
            start_offset_master: Начальный кадр мастера (для отображения в видео)
            start_offset_student: Начальный кадр ученика (для отображения в видео)

        Returns:
            output_path: Путь к сохранённому видео
            segments_master: Сегменты форм
            dtw_path: Путь DTW
        """
        cap_master = cv2.VideoCapture(master_path)
        cap_student = cv2.VideoCapture(student_path)

        width_m = int(cap_master.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_m = int(cap_master.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_m = int(cap_master.get(cv2.CAP_PROP_FPS))

        width_s = int(cap_student.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_s = int(cap_student.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_height = min(height_m, height_s)
        width_m_new = int(width_m * target_height / height_m)
        width_s_new = int(width_s * target_height / height_s)

        output_width = width_m_new + width_s_new
        output_height = target_height

        print(f"📐 Мастер: {width_m}x{height_m} → {width_m_new}x{target_height}")
        print(f"📐 Ученик: {width_s}x{height_s} → {width_s_new}x{target_height}")
        print(f"📐 Итог: {output_width}x{output_height}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps_m, (output_width, output_height))

        # Детекция сегментов если не переданы
        if segments_master is None:
            segments_master = self.segment_detector.detect_segments(poses_master)
            print(f"📍 Найдено сегментов (форм): {len(segments_master)}")
            for i, seg in enumerate(segments_master):
                print(
                    f"   {i + 1}. {seg['name']}: кадры {seg['start']}-{seg['end']} ({seg['end'] - seg['start']} кадров)")

        # DTW путь
        if dtw_path is None:
            print("⏳ Вычисление DTW пути...")
            _, dtw_path, _, _ = self.dtw_aligner.align_poses(poses_master, poses_student)

        print(f"🔗 DTW путь: {len(dtw_path)} пар кадров")

        # Переменные для перемотки видео
        curr_m_idx = 0
        curr_s_idx = 0
        frame_master = None
        frame_student = None

        # Обработка по DTW пути
        for out_frame_idx, (m_idx, s_idx) in enumerate(dtw_path):
            # Получаем кадры по индексам DTW (с учётом обрезки)
            actual_m_idx = m_idx + start_offset_master
            actual_s_idx = s_idx + start_offset_student

            frame_master, curr_m_idx = self.get_frame_at_index(
                cap_master, actual_m_idx, curr_m_idx, frame_master
            )
            frame_student, curr_s_idx = self.get_frame_at_index(
                cap_student, actual_s_idx, curr_s_idx, frame_student
            )

            if frame_master is None or frame_student is None:
                continue

            # Изменяем размер
            frame_master = cv2.resize(frame_master, (width_m_new, target_height))
            frame_student = cv2.resize(frame_student, (width_s_new, target_height))

            # Получаем позы (из обрезанных данных)
            pose_m = poses_master[m_idx] if m_idx < len(poses_master) else None
            pose_s = poses_student[s_idx] if s_idx < len(poses_student) else None

            # Рисуем скелеты
            frame_master_vis = self.draw_skeleton(frame_master, pose_m, color=(0, 255, 0))
            frame_student_vis = self.draw_skeleton(frame_student, pose_s, color=(0, 0, 255))

            # Название формы (по индексу в обрезанных позах)
            current_form = self.segment_detector.get_current_form(segments_master, m_idx)

            # Чёрный фон для текста
            cv2.rectangle(frame_master_vis, (0, 0), (frame_master_vis.shape[1], 100), (0, 0, 0), -1)
            cv2.rectangle(frame_student_vis, (0, 0), (frame_student_vis.shape[1], 100), (0, 0, 0), -1)

            # Текст названия формы (кириллица через PIL)
            frame_master_vis = draw_text_cv2(
                frame_master_vis,
                current_form,
                (10, 35),
                font_size=18,
                color=(255, 255, 255)
            )
            frame_student_vis = draw_text_cv2(
                frame_student_vis,
                current_form,
                (10, 35),
                font_size=18,
                color=(255, 255, 255)
            )

            # Текст MASTER / STUDENT
            frame_master_vis = draw_text_cv2(
                frame_master_vis,
                "MASTER",
                (10, 70),
                font_size=18,
                color=(0, 255, 0)
            )
            frame_student_vis = draw_text_cv2(
                frame_student_vis,
                "STUDENT",
                (10, 70),
                font_size=18,
                color=(0, 0, 255)
            )

            # Прогресс бар сегмента
            if segments_master:
                for seg in segments_master:
                    if seg['start'] <= m_idx <= seg['end']:
                        progress = (m_idx - seg['start']) / max(1, seg['end'] - seg['start'])
                        bar_width = int(frame_master_vis.shape[1] * progress)
                        cv2.rectangle(frame_master_vis, (10, 80), (10 + bar_width, 90),
                                      (0, 255, 0), -1)
                        cv2.rectangle(frame_student_vis, (10, 80), (10 + bar_width, 90),
                                      (0, 0, 255), -1)
                        break

            # Номер кадра (оригинальный, с учётом смещения)
            frame_master_vis = draw_text_cv2(
                frame_master_vis,
                f"Frame: {actual_m_idx}",
                (frame_master_vis.shape[1] - 150, 35),
                font_size=14,
                color=(200, 200, 200)
            )
            frame_student_vis = draw_text_cv2(
                frame_student_vis,
                f"Frame: {actual_s_idx}",
                (frame_student_vis.shape[1] - 150, 35),
                font_size=14,
                color=(200, 200, 200)
            )

            # Объединяем кадры
            combined = np.hstack([frame_master_vis, frame_student_vis])
            out.write(combined)

            if out_frame_idx % 30 == 0:
                print(f"  → Обработано: {out_frame_idx}/{len(dtw_path)} (M:{actual_m_idx}, S:{actual_s_idx})")

        cap_master.release()
        cap_student.release()
        out.release()

        print(f"✓ Видео сохранено: {output_path}")
        return output_path, segments_master, dtw_path

    def close(self):
        """Освобождает ресурсы MediaPipe"""
        self.pose.close()