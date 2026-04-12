"""
Диагностика видео: поиск начала движения
"""
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


def analyze_video_motion(video_path, name):
    """Анализирует движение в видео"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"\n📹 {name}: {total_frames} кадров, {fps} FPS, {total_frames / fps:.1f} сек")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.3)

    motion_scores = []
    prev_pose = None
    first_motion_frame = None

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            curr_pose = np.array([[l.x, l.y] for l in results.pose_landmarks.landmark])

            if prev_pose is not None:
                # Расстояние между позами
                motion = np.mean(np.linalg.norm(curr_pose - prev_pose, axis=1))
                motion_scores.append(motion)

                # Поиск первого движения
                if first_motion_frame is None and motion > 0.02:
                    first_motion_frame = frame_count
                    print(f"  🏁 Начало движения: кадр {frame_count} ({frame_count / fps:.1f} сек)")
            else:
                motion_scores.append(0)

            prev_pose = curr_pose
        else:
            motion_scores.append(0)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  → Обработано: {frame_count}/{total_frames}")

    cap.release()
    pose.close()

    # Визуализация
    plt.figure(figsize=(15, 4))
    plt.plot(motion_scores[:500], linewidth=1)  # Первые 500 кадров
    plt.axhline(y=0.02, color='r', linestyle='--', label='Порог движения')
    if first_motion_frame and first_motion_frame < 500:
        plt.axvline(x=first_motion_frame, color='g', linestyle='-', label='Старт движения')
    plt.title(f'{name} - Активность движения (первые 500 кадров)')
    plt.xlabel('Кадр')
    plt.ylabel('Изменение позы')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'debug_{name.replace(" ", "_")}.png', dpi=150)
    print(f"  📊 График сохранён: debug_{name.replace(' ', '_')}.png")

    return first_motion_frame, len(motion_scores)


# Запуск
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Диагностика видео")
    print("=" * 60)

    master_start, master_len = analyze_video_motion('input_videos/master_video01.mp4', 'MASTER')
    student_start, student_len = analyze_video_motion('input_videos/student_video.mp4', 'STUDENT')

    print("\n" + "=" * 60)
    print("📋 ИТОГИ:")
    print(f"  Мастер: начало движения на кадре {master_start}")
    print(f"  Ученик: начало движения на кадре {student_start}")
    print(f"  Разница: {abs(master_start - student_start)} кадров")

    if master_start and student_start:
        offset = abs(master_start - student_start)
        if offset > 30:
            print(f"\n⚠️ ПРЕДУПРЕЖДЕНИЕ: Большая разница в начале ({offset} кадров)!")
            print("  Рекомендация: Обрезать видео до начала движения")
    print("=" * 60)