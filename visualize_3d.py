"""
3D визуализация позы из MediaPipe
"""
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class Pose3DVisualizer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # 33 точки MediaPipe Pose
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

        # Соединения (ребра)
        self.connections = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
            (3, 7), (6, 8), (9, 10),
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32)
        ]

    def extract_pose_from_frame(self, frame):
        """Извлекает 3D координаты из кадра"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        return np.array(landmarks)

    def extract_poses_from_video(self, video_path):
        """Извлекает все позы из видео"""
        cap = cv2.VideoCapture(video_path)
        poses = []

        print(f"📹 Обработка видео: {video_path}")
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pose = self.extract_pose_from_frame(frame)
            if pose is not None:
                poses.append(pose)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  Кадров обработано: {frame_count}")

        cap.release()
        print(f"✓ Извлечено {len(poses)} поз")
        return poses

    def plot_single_pose_3d(self, pose, frame_num=0):
        """Рисует одну позу в 3D"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Точки
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='red', s=50)

        # Линии
        for i, j in self.connections:
            ax.plot([pose[i, 0], pose[j, 0]],
                    [pose[i, 1], pose[j, 1]],
                    [pose[i, 2], pose[j, 2]], 'b-', linewidth=2)

        ax.set_title(f'Frame {frame_num}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def animate_poses_3d(self, poses, output_path='output_3d.mp4', fps=20):
        """Создает 3D анимацию"""
        if len(poses) == 0:
            print("✗ Нет данных для анимации")
            return

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Фиксированные границы
        all_points = np.vstack(poses)
        margin = 0.1
        ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

        def update(frame):
            ax.clear()

            pose = poses[frame]

            # Точки
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='red', s=50)

            # Линии
            for i, j in self.connections:
                ax.plot([pose[i, 0], pose[j, 0]],
                        [pose[i, 1], pose[j, 1]],
                        [pose[i, 2], pose[j, 2]], 'b-', linewidth=2)

            ax.set_title(f'Frame {frame + 1} / {len(poses)}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Те же границы
            ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
            ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
            ax.set_zlim(all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

            return ax,

        print("🎬 Создание анимации...")
        ani = animation.FuncAnimation(fig, update, frames=len(poses),
                                      interval=1000 / fps, blit=False)

        try:
            ani.save(output_path, writer='ffmpeg', fps=fps)
            print(f"✓ Видео сохранено: {output_path}")
        except Exception as e:
            print(f"⚠ Ошибка сохранения MP4: {e}")
            print("💡 Попробуйте сохранить как HTML...")
            html_path = output_path.replace('.mp4', '.html')
            ani.save(html_path, writer='html')
            print(f"✓ HTML сохранен: {html_path}")

        plt.close()


def main():
    import sys

    if len(sys.argv) < 2:
        print("Использование: python visualize_3d.py <video_file>")
        print("Пример: python visualize_3d.py input_videos/master.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    visualizer = Pose3DVisualizer()

    # Извлечение поз
    poses = visualizer.extract_poses_from_video(video_path)

    if len(poses) > 0:
        # Показать первую позу
        print("\n📊 Показ первой позы...")
        visualizer.plot_single_pose_3d(poses[0], frame_num=0)

        # Создать анимацию
        print("\n🎬 Создание 3D анимации...")
        visualizer.animate_poses_3d(poses, 'output_videos/skeleton_3d.mp4', fps=20)


if __name__ == "__main__":
    main()