"""
Детектор сегментов (форм) Тайцзи
Оптимизированная версия с фильтрацией и кириллицей
"""
import numpy as np


class SegmentDetector:
    """Автоматическое разбиение видео на формы Тайцзи"""

    def __init__(self, min_segment_length=150, merge_threshold=0.3):
        """
        Args:
            min_segment_length: Минимальная длина сегмента в кадрах
            merge_threshold: Порог для слияния соседних одинаковых форм
        """
        # Названия форм (кириллица теперь работает через PIL)
        self.form_names = [
            "1. Начало (Qi Shi)",
            "2. Размахнуть крыльями",
            "3. Облачные руки (Левая)",
            "4. Облачные руки (Правая)",
            "5. Одиночный удар",
            "6. Золотой петух",
            "7. Удар пяткой",
            "8. Закрыть форму (Shou Shi)"
        ]
        self.min_segment_length = min_segment_length
        self.merge_threshold = merge_threshold

    def detect_segments(self, poses):
        """Находит границы сегментов с фильтрацией"""
        if not poses or len(poses) < 60:
            return []

        segments = []
        segment_start = 0

        # Вычисляем изменения позы
        pose_changes = []
        for i in range(1, len(poses)):
            if poses[i] is None or poses[i-1] is None:
                pose_changes.append(0)
                continue

            dist = 0
            count = 0
            for j in range(len(poses[i])):
                if poses[i][j][3] > 0.5 and poses[i-1][j][3] > 0.5:
                    dist += np.linalg.norm(poses[i][j][:2] - poses[i-1][j][:2])
                    count += 1

            avg_change = dist / count if count > 0 else 0
            pose_changes.append(avg_change)

        # Более высокий порог (top 10% вместо 20%)
        threshold = np.percentile(pose_changes, 90)

        for i, change in enumerate(pose_changes):
            # Увеличенная минимальная длина
            if change > threshold and (i - segment_start) >= self.min_segment_length:
                segment_end = i
                form_idx = len(segments) % len(self.form_names)
                form_name = self.form_names[form_idx]

                segments.append({
                    'start': segment_start,
                    'end': segment_end,
                    'name': form_name
                })

                segment_start = i

        # Последний сегмент
        if segment_start < len(poses) - self.min_segment_length:
            form_idx = len(segments) % len(self.form_names)
            segments.append({
                'start': segment_start,
                'end': len(poses) - 1,
                'name': self.form_names[form_idx]
            })

        # Слияние соседних одинаковых форм
        segments = self._merge_similar_segments(segments)

        print(f"📍 Найдено форм после фильтрации: {len(segments)}")

        return segments

    def _merge_similar_segments(self, segments):
        """Сливает соседние сегменты с одинаковыми названиями"""
        if len(segments) <= 1:
            return segments

        merged = [segments[0]]

        for seg in segments[1:]:
            if seg['name'] == merged[-1]['name']:
                # Сливаем
                merged[-1]['end'] = seg['end']
            else:
                merged.append(seg)

        return merged

    def get_current_form(self, segments, frame_num):
        """Возвращает текущую форму для данного кадра"""
        if not segments:
            return "Форма: Не определено"

        for seg in segments:
            if seg['start'] <= frame_num <= seg['end']:
                return seg['name']

        return "Форма: Не определено"