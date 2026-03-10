"""
Модуль для извлечения координат позы
"""
import mediapipe as mp
import numpy as np


class PoseExtractor:
    """Извлечение 3D координат позы из видео"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Названия точек (33 landmarks)
        self.landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky', 'right_pinky',
            'left_index', 'right_index',
            'left_thumb', 'right_thumb',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]

    def extract_frame_pose(self, frame):
        """
        Извлекает позу из одного кадра

        Returns:
            dict: {landmark_name: {'x': x, 'y': y, 'z': z, 'visibility': v}}
            или None если поза не обнаружена
        """
        # Конвертация в RGB
        image_rgb = mp.solutions.image_utils.image_to_numpy(frame)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        # Извлечение всех точек
        pose_dict = {}
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            pose_dict[self.landmark_names[i]] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }

        return pose_dict

    def extract_to_numpy(self, pose_dict):
        """
        Конвертирует словарь в numpy массив

        Returns:
            np.array: shape (33, 4) [x, y, z, visibility]
        """
        if not pose_dict:
            return None

        coords = []
        for name in self.landmark_names:
            point = pose_dict[name]
            coords.append([point['x'], point['y'], point['z'], point['visibility']])

        return np.array(coords)

    def close(self):
        self.pose.close()