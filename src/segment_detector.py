"""
Детектор сегментов (форм) Тайцзи
Разбивает движение на смысловые фрагменты по ключевым позам
"""
import numpy as np


class SegmentDetector:
    """Автоматическое разбиение видео на формы Тайцзи"""

    def __init__(self):
        # Названия 24 форм Пекинского стиля (пример)
        self.form_names = [
            "1. Начало (Qi Shi)",
            "2. Размахнуть крыльями",
            "3. Облачные руки (Левая)",
            "4. Облачные руки (Правая)",
            "5. Одиночный удар",
            "6. Золотой петух стоит на одной ноге",
            "7. Удар пяткой",
            "8. Закрыть форму (Shou Shi)"
        ]

        # Ключевые позы для каждой формы (углы для детекции)
        # В реальной версии нужно настроить с мастером
        self.key_poses = {
            'start': {'left_wrist_y': 0.3, 'right_wrist_y': 0.3},  # Руки внизу
            'cloud_hands': {'left_wrist_y': 0.5, 'right_wrist_y': 0.5},  # Руки на уровне груди
            'single_whip': {'left_wrist_x': 0.2, 'right_wrist_x': 0.8},  # Руки в стороны
        }

    def detect_segments(self, poses):
        """
        Находит границы сегментов по изменению позы

        Args:
            poses: список поз (np.array или None)

        Returns:
            segments: список [{'start': int, 'end': int, 'name': str}]
        """
        if not poses or len(poses) < 30:
            return []

        segments = []
        segment_start = 0

        # Вычисляем "скорость изменения позы" для каждого кадра
        pose_changes = []
        for i in range(1, len(poses)):
            if poses[i] is None or poses[i - 1] is None:
                pose_changes.append(0)
                continue

            # Расстояние между позами (все видимые точки)
            dist = 0
            count = 0
            for j in range(len(poses[i])):
                if poses[i][j][3] > 0.5 and poses[i - 1][j][3] > 0.5:
                    dist += np.linalg.norm(poses[i][j][:2] - poses[i - 1][j][:2])
                    count += 1

            avg_change = dist / count if count > 0 else 0
            pose_changes.append(avg_change)

        # Ищем пики изменений (границы между формами)
        # Там где движение резко меняется - вероятно новая форма
        threshold = np.percentile(pose_changes, 80)  # Top 20% изменений

        for i, change in enumerate(pose_changes):
            if change > threshold and (i - segment_start) > 30:  # Минимум 30 кадров на форму
                segment_end = i

                # Определяем название формы (упрощённо)
                form_idx = len(segments) % len(self.form_names)
                form_name = self.form_names[form_idx]

                segments.append({
                    'start': segment_start,
                    'end': segment_end,
                    'name': form_name
                })

                segment_start = i

        # Последний сегмент
        if segment_start < len(poses) - 30:
            form_idx = len(segments) % len(self.form_names)
            segments.append({
                'start': segment_start,
                'end': len(poses) - 1,
                'name': self.form_names[form_idx]
            })

        return segments

    def get_current_form(self, segments, frame_num):
        """Возвращает текущую форму для данного кадра"""
        for seg in segments:
            if seg['start'] <= frame_num <= seg['end']:
                return seg['name']
        return "Не определено"