"""
Движок сравнения движений Тайцзи
Сравнивает позы мастера и ученика, считает метрики, синхронизирует по времени
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from src.segment_detector import SegmentDetector
from src.dtw_aligner import DTWAligner


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

        self.segment_detector = SegmentDetector()
        self.dtw_aligner = DTWAligner()

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

    def calculate_balance(self, pose):
        """Расчет баланса (центр тяжести)"""
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
        """Сравнение двух поз (процент совпадения)"""
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
        """Рисует скелет на кадре"""
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
                                 poses_master, poses_student, segments_master=None):
        """
        Создает видео с сравнением (split-screen) + маркеры форм + DTW синхронизация
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
                print(f"   {i+1}. {seg['name']}: кадры {seg['start']}-{seg['end']} ({seg['end']-seg['start']} кадров)")

        frame_count = 0
        min_frames = min(len(poses_master), len(poses_student))

        while frame_count < min_frames:
            ret1, frame_master = cap_master.read()
            ret2, frame_student = cap_student.read()

            if not ret1 or not ret2:
                break

            frame_master = cv2.resize(frame_master, (width_m_new, target_height))
            frame_student = cv2.resize(frame_student, (width_s_new, target_height))

            pose_m = poses_master[frame_count]
            pose_s = poses_student[frame_count]

            frame_master_vis = self.draw_skeleton(frame_master, pose_m, color=(0, 255, 0))
            frame_student_vis = self.draw_skeleton(frame_student, pose_s, color=(0, 0, 255))

            # Название текущей формы
            current_form = self.segment_detector.get_current_form(segments_master, frame_count)

            # Фон для текста
            cv2.rectangle(frame_master_vis, (0, 0), (frame_master_vis.shape[1], 100), (0, 0, 0), -1)
            cv2.rectangle(frame_student_vis, (0, 0), (frame_student_vis.shape[1], 100), (0, 0, 0), -1)

            # Текст названия формы
            cv2.putText(frame_master_vis, current_form, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_student_vis, current_form, (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Текст MASTER / STUDENT
            cv2.putText(frame_master_vis, "MASTER", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_student_vis, "STUDENT", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Прогресс бар сегмента
            if segments_master:
                for seg in segments_master:
                    if seg['start'] <= frame_count <= seg['end']:
                        progress = (frame_count - seg['start']) / max(1, seg['end'] - seg['start'])
                        bar_width = int(frame_master_vis.shape[1] * progress)
                        cv2.rectangle(frame_master_vis, (10, 80), (10 + bar_width, 90),
                                     (0, 255, 0), -1)
                        cv2.rectangle(frame_student_vis, (10, 80), (10 + bar_width, 90),
                                     (0, 0, 255), -1)
                        break

            # Номер кадра
            cv2.putText(frame_master_vis, f"Frame: {frame_count}", (frame_master_vis.shape[1] - 150, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            combined = np.hstack([frame_master_vis, frame_student_vis])
            out.write(combined)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  → Обработано кадров: {frame_count}/{min_frames}")

        cap_master.release()
        cap_student.release()
        out.release()

        print(f"✓ Видео сохранено: {output_path}")
        return output_path, segments_master

    def close(self):
        self.pose.close()