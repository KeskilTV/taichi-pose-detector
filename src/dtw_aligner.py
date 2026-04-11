"""
DTW (Dynamic Time Warping) для синхронизации видео
Оптимизированная версия с прореживанием и ограниченным окном
"""
import numpy as np
from scipy.spatial.distance import euclidean


class DTWAligner:
    """Синхронизация двух последовательностей поз по времени"""

    def __init__(self, step=5, window_size=50):
        """
        Args:
            step: Прореживание (каждый N-й кадр) - ускоряет в N раз
            window_size: Ограничение окна DTW - ускоряет ещё в 10-20 раз
        """
        self.step = step
        self.window_size = window_size

    def compute_dtw(self, poses1, poses2):
        """DTW с ограниченным окном (Sakoe-Chiba band)"""
        n = len(poses1)
        m = len(poses2)

        if n == 0 or m == 0:
            return [(0, 0)]

        # Матрица расстояний
        dist_matrix = np.full((n, m), np.inf)

        for i in range(n):
            # Ограничиваем поиск окном (ускорение!)
            j_min = max(0, i - self.window_size)
            j_max = min(m, i + self.window_size)

            for j in range(j_min, j_max):
                if poses1[i] is None or poses2[j] is None:
                    dist_matrix[i, j] = 1.0
                else:
                    d = 0
                    count = 0
                    for k in range(min(len(poses1[i]), len(poses2[j]))):
                        if poses1[i][k][3] > 0.5 and poses2[j][k][3] > 0.5:
                            d += euclidean(poses1[i][k][:2], poses2[j][k][:2])
                            count += 1
                    dist_matrix[i, j] = d / count if count > 0 else 1.0

        # DTW матрица накопленных расстояний
        dtw_matrix = np.full((n, m), np.inf)
        dtw_matrix[0, 0] = dist_matrix[0, 0]

        for i in range(n):
            j_min = max(0, i - self.window_size)
            j_max = min(m, i + self.window_size)

            for j in range(j_min, j_max):
                if i == 0 and j == 0:
                    continue

                candidates = []
                if i > 0 and j >= j_min:
                    candidates.append(dtw_matrix[i-1, j])
                if j > 0 and i >= j_min:
                    candidates.append(dtw_matrix[i, j-1])
                if i > 0 and j > 0:
                    candidates.append(dtw_matrix[i-1, j-1])

                if candidates:
                    dtw_matrix[i, j] = dist_matrix[i, j] + min(candidates)

        # Обратный путь (от конца к началу)
        path = []
        i, j = n - 1, min(m - 1, n - 1 + self.window_size)
        j = min(j, m - 1)
        path.append((i, j))

        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                prev = min(
                    dtw_matrix[i-1, j] if i > 0 and j >= max(0, i-1-self.window_size) else np.inf,
                    dtw_matrix[i, j-1] if j > 0 and i >= max(0, j-1-self.window_size) else np.inf,
                    dtw_matrix[i-1, j-1] if i > 0 and j > 0 else np.inf
                )
                if prev == dtw_matrix[i-1, j-1]:
                    i -= 1
                    j -= 1
                elif prev == dtw_matrix[i-1, j]:
                    i -= 1
                else:
                    j -= 1
            path.append((i, j))

        path.reverse()
        return path

    def align_poses(self, poses_master, poses_student):
        """Синхронизирует позы с прореживанием"""
        if len(poses_master) < 10 or len(poses_student) < 10:
            print("⚠️ Слишком мало кадров для DTW")
            return poses_student[:len(poses_master)], [], [], []

        # Прореживаем (каждый 5-й кадр)
        poses_master_sampled = poses_master[::self.step]
        poses_student_sampled = poses_student[::self.step]

        print(f"🔧 DTW: {len(poses_master)} → {len(poses_master_sampled)} кадров (step={self.step})")

        # DTW на прореженных данных
        path_sampled = self.compute_dtw(poses_master_sampled, poses_student_sampled)

        # Масштабируем путь обратно к полным кадрам
        path = []
        for m_idx, s_idx in path_sampled:
            m_full = m_idx * self.step
            s_full = s_idx * self.step
            path.append((m_full, s_full))

        # Создаём выровненные позы ученика
        aligned_student = []
        for m_idx, s_idx in path:
            if m_idx < len(poses_master):
                s_clamped = min(s_idx, len(poses_student) - 1)
                aligned_student.append(poses_student[s_clamped])

        print(f"✓ DTW путь: {len(path)} соответствий")

        return aligned_student, path, [p[0] for p in path], [p[1] for p in path]