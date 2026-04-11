"""
DTW (Dynamic Time Warping) для синхронизации видео
Растягивает/сжимает время ученика под мастера
"""
import numpy as np
from scipy.spatial.distance import euclidean


class DTWAligner:
    """Синхронизация двух последовательностей поз по времени"""

    def __init__(self):
        pass

    def compute_dtw(self, poses1, poses2):
        """
        Вычисляет DTW матрицу и оптимальный путь

        Returns:
            path: список пар индексов [(i1, i2), (i1, i2), ...]
        """
        n = len(poses1)
        m = len(poses2)

        # Матрица расстояний
        dist_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                if poses1[i] is None or poses2[j] is None:
                    dist_matrix[i, j] = 1.0  # Максимальное расстояние
                else:
                    # Расстояние между позами (только x, y)
                    d = 0
                    count = 0
                    for k in range(len(poses1[i])):
                        if poses1[i][k][3] > 0.5 and poses2[j][k][3] > 0.5:
                            d += euclidean(poses1[i][k][:2], poses2[j][k][:2])
                            count += 1
                    dist_matrix[i, j] = d / count if count > 0 else 1.0

        # DTW матрица накопленных расстояний
        dtw_matrix = np.full((n, m), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(n):
            for j in range(m):
                cost = dist_matrix[i, j]
                if i > 0 or j > 0:
                    cost += min(
                        dtw_matrix[i - 1, j] if i > 0 else np.inf,
                        dtw_matrix[i, j - 1] if j > 0 else np.inf,
                        dtw_matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
                    )
                dtw_matrix[i, j] = cost

        # Обратный путь (от конца к началу)
        path = []
        i, j = n - 1, m - 1
        path.append((i, j))

        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                prev = min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1]
                )
                if prev == dtw_matrix[i - 1, j - 1]:
                    i -= 1
                    j -= 1
                elif prev == dtw_matrix[i - 1, j]:
                    i -= 1
                else:
                    j -= 1
            path.append((i, j))

        path.reverse()
        return path

    def align_poses(self, poses_master, poses_student):
        """
        Синхронизирует позы ученика с мастером

        Returns:
            aligned_student: позы ученика, растянутые под мастера
            alignment_path: DTW путь для визуализации
        """
        path = self.compute_dtw(poses_master, poses_student)

        # Для каждого кадра мастера находим соответствующий кадр ученика
        aligned_student = []
        master_indices = []
        student_indices = []

        for m_idx, s_idx in path:
            aligned_student.append(poses_student[s_idx] if s_idx < len(poses_student) else None)
            master_indices.append(m_idx)
            student_indices.append(s_idx)

        return aligned_student, path, master_indices, student_indices