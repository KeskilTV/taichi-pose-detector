"""
3D визуализация скелета
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class Skeleton3D:
    """3D визуализация скелета MediaPipe"""

    def __init__(self):
        # Соединения между точками (ребра скелета)
        self.connections = [
            # Голова
            ('nose', 'left_eye_inner'),
            ('left_eye_inner', 'left_eye'),
            ('left_eye', 'left_eye_outer'),
            ('nose', 'right_eye_inner'),
            ('right_eye_inner', 'right_eye'),
            ('right_eye', 'right_eye_outer'),
            ('left_ear', 'left_eye_outer'),
            ('right_ear', 'right_eye_outer'),
            ('mouth_left', 'nose'),
            ('mouth_right', 'nose'),

            # Тело
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),

            # Левая рука
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('left_wrist', 'left_pinky'),
            ('left_wrist', 'left_index'),
            ('left_wrist', 'left_thumb'),

            # Правая рука
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('right_wrist', 'right_pinky'),
            ('right_wrist', 'right_index'),
            ('right_wrist', 'right_thumb'),

            # Левая нога
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('left_ankle', 'left_heel'),
            ('left_ankle', 'left_foot_index'),

            # Правая нога
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('right_ankle', 'right_heel'),
            ('right_ankle', 'right_foot_index'),
        ]

        # Названия точек
        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]

    def plot_pose(self, pose_data, ax=None):
        """
        Рисует одну позу в 3D

        Args:
            pose_data: dict или np.array с координатами
            ax: matplotlib 3D axis
        """
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Конвертация в numpy если dict
        if isinstance(pose_data, dict):
            coords = np.array([
                [pose_data[name]['x'], pose_data[name]['y'], pose_data[name]['z']]
                for name in self.landmark_names
            ])
        else:
            coords = pose_data[:, :3]  # Берем только x, y, z

        # Рисуем точки
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c='red', s=20, alpha=0.8)

        # Рисуем соединения
        for name1, name2 in self.connections:
            idx1 = self.landmark_names.index(name1)
            idx2 = self.landmark_names.index(name2)

            ax.plot(
                [coords[idx1, 0], coords[idx2, 0]],
                [coords[idx1, 1], coords[idx2, 1]],
                [coords[idx1, 2], coords[idx2, 2]],
                'b-', linewidth=2, alpha=0.6
            )

        # Настройка осей
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])

        return ax

    def animate_sequence(self, pose_sequence, output_path='skeleton_3d.mp4'):
        """
        Создает 3D анимацию последовательности поз

        Args:
            pose_sequence: список поз (dict или np.array)
            output_path: путь для сохранения видео
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            ax.clear()
            self.plot_pose(pose_sequence[frame], ax)
            ax.set_title(f'Frame {frame}')
            return ax,

        ani = animation.FuncAnimation(
            fig, update, frames=len(pose_sequence),
            interval=50, blit=False
        )

        # Сохранение
        ani.save(output_path, writer='ffmpeg', fps=20)
        plt.close()
        print(f"✓ 3D анимация сохранена: {output_path}")