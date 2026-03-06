"""
Модуль для обнаружения позы с помощью MediaPipe
"""
import mediapipe as mp
import cv2
from config import POSE_SETTINGS, POSE_DETECTION_SETTINGS


class PoseDetector:
    """Класс для обнаружения и отрисовки позы человека"""

    def __init__(self):
        # Инициализация MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=POSE_DETECTION_SETTINGS['model_complexity'],
            model_complexity=POSE_DETECTION_SETTINGS['model_complexity'],
            min_detection_confidence=POSE_DETECTION_SETTINGS['min_detection_confidence'],
            min_tracking_confidence=POSE_DETECTION_SETTINGS['min_tracking_confidence']
        )

    def process_frame(self, frame):
        """
        Обрабатывает один кадр и возвращает изображение с нарисованным скелетом

        Args:
            frame: Кадр изображения (BGR формат)

        Returns:
            image_with_pose: Кадр с нарисованным скелетом
            pose_landmarks: Данные о позе (или None если не обнаружена)
        """
        # Конвертируем BGR -> RGB для MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Detect Pose
        results = self.pose.process(image_rgb)

        # Возвращаем обратно в BGR
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Отрисовка скелета если поза обнаружена
        if results.pose_landmarks:
            self._draw_pose(image_bgr, results.pose_landmarks)

        return image_bgr, results.pose_landmarks

    def _draw_pose(self, image, landmarks):
        """Рисует скелет на изображении"""
        self.mp_drawing.draw_landmarks(
            image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=POSE_SETTINGS['point_color'],
                thickness=POSE_SETTINGS['line_thickness'],
                circle_radius=POSE_SETTINGS['point_radius']
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=POSE_SETTINGS['line_color'],
                thickness=POSE_SETTINGS['line_thickness']
            )
        )

    def get_landmark_coordinates(self, landmarks, landmark_id):
        """
        Получает координаты конкретной точки скелета

        Args:
            landmarks: Данные о позе от MediaPipe
            landmark_id: ID точки (например, mp_pose.PoseLandmark.LEFT_WRIST)

        Returns:
            tuple: (x, y, z) координаты или None
        """
        if landmarks:
            landmark = landmarks.landmark[landmark_id]
            return (landmark.x, landmark.y, landmark.z)
        return None

    def close(self):
        """Освобождает ресурсы"""
        self.pose.close()