import cv2
import mediapipe as mp
import os

# Автопоиск видео в папке
input_dir = 'input_videos'
video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

if not video_files:
    print(f"✗ В папке {input_dir} нет видео файлов!")
    print("Положите туда видео и попробуйте снова")
    exit()

print(f"Найдено видео: {video_files}")
video_path = os.path.join(input_dir, video_files[0])
print(f"Используем: {video_path}\n")

# Проверка
cap = cv2.VideoCapture(video_path)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

poses_found = 0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        poses_found += 1

    frame_count += 1

    if frame_count % 30 == 0:
        print(f"Кадр {frame_count}: найдено поз = {poses_found}")

    if frame_count >= 150:  # Первые 5 секунд
        break

cap.release()
pose.close()

print(f"\n✓ Итого: {poses_found}/{frame_count} кадров с позами")

if poses_found > 0:
    print("🎉 Видео подходит! Запускайте visualize_3d.py")
else:
    print("⚠ Видео НЕ подходит. Попробуйте другое.")