"""
🥋 TaiChi Comparison Tool
Веб-интерфейс для сравнения движений мастера и ученика

Запуск:
    streamlit run app.py

Автор: Васильев Кэскил
Email: zrengemlab@gmail.com
"""

import streamlit as st
import os
import tempfile
from datetime import datetime
from src.comparison_engine import ComparisonEngine
from src.dtw_aligner import DTWAligner
import numpy as np

# Настройка страницы
st.set_page_config(
    page_title="TaiChi Comparison Tool",
    page_icon="🥋",
    layout="wide"
)

# Заголовок
st.title("🥋 TaiChi Comparison Tool")
st.markdown("**Сравнение движений мастера и ученика с анализом позы и форм Тайцзи**")
st.markdown("---")

# Боковая панель
st.sidebar.header("⚙️ Настройки")
st.sidebar.markdown(f"**Автор:** Васильев Кэскил")
st.sidebar.markdown(f"**Версия:** 0.3.0 MVP")
st.sidebar.markdown(f"**Ветка:** `feature/comparison-mvp`")

# Создание папок
os.makedirs('input_videos', exist_ok=True)
os.makedirs('output_videos', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Загрузка видео
st.header("📁 Загрузка видео")

col1, col2 = st.columns(2)

master_path = None
student_path = None

with col1:
    master_file = st.file_uploader(
        "📹 Видео мастера",
        type=['mp4', 'avi', 'mov'],
        key='master'
    )
    if master_file:
        st.video(master_file)
        temp_master = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_master.write(master_file.read())
        master_path = temp_master.name
        st.success(f"✓ Загружено: {master_file.name}")

with col2:
    student_file = st.file_uploader(
        "📹 Видео ученика",
        type=['mp4', 'avi', 'mov'],
        key='student'
    )
    if student_file:
        st.video(student_file)
        temp_student = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_student.write(student_file.read())
        student_path = temp_student.name
        st.success(f"✓ Загружено: {student_file.name}")

# Кнопка запуска анализа
st.markdown("---")
st.header("🔍 Анализ")

if st.button("🚀 Запустить сравнение", type='primary', disabled=not (master_path and student_path)):
    st.info("⏳ Обработка видео... (это займёт некоторое время)")

    try:
        engine = ComparisonEngine()
        dtw_aligner = DTWAligner()

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Шаг 1: Извлечение поз мастера
        status_text.text("Извлечение поз мастера...")
        poses_master = engine.extract_poses(master_path)
        progress_bar.progress(20)

        # Шаг 2: Извлечение поз ученика
        status_text.text("Извлечение поз ученика...")
        poses_student = engine.extract_poses(student_path)
        progress_bar.progress(40)

        # Шаг 3: DTW синхронизация
        status_text.text("Синхронизация скорости (DTW)...")
        poses_student_aligned, dtw_path, master_idx, student_idx = dtw_aligner.align_poses(
            poses_master, poses_student
        )
        progress_bar.progress(55)

        # Шаг 4: Детекция сегментов
        status_text.text("Разбиение на формы Тайцзи...")
        segments_master = engine.segment_detector.detect_segments(poses_master)
        progress_bar.progress(65)

        # Показываем найденные сегменты
        if segments_master:
            with st.expander("📍 Найденные формы Тайцзи (развернуть для деталей)"):
                for i, seg in enumerate(segments_master):
                    st.write(f"**{i+1}. {seg['name']}** — кадры {seg['start']}–{seg['end']} ({seg['end']-seg['start']} кадров)")

        # Шаг 5: Создание видео сравнения
        status_text.text("Создание видео сравнения...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_videos/comparison_{timestamp}.mp4"

        output_path, segments = engine.create_comparison_video(
            master_path, student_path, output_path,
            poses_master, poses_student_aligned,
            segments_master=segments_master
        )
        progress_bar.progress(85)

        # Шаг 6: Расчет метрик
        status_text.text("Расчет метрик...")

        similarities = []
        balances_master = []
        balances_student = []

        for i in range(min(len(poses_master), len(poses_student_aligned))):
            sim = engine.calculate_pose_similarity(poses_master[i], poses_student_aligned[i])
            similarities.append(sim)

            if poses_master[i] is not None:
                balances_master.append(engine.calculate_balance(poses_master[i]))
            if poses_student_aligned[i] is not None:
                balances_student.append(engine.calculate_balance(poses_student_aligned[i]))

        avg_similarity = np.mean(similarities) if similarities else 0
        avg_balance_master = np.mean(balances_master) if balances_master else 0
        avg_balance_student = np.mean(balances_student) if balances_student else 0

        progress_bar.progress(100)
        status_text.text("✅ Готово!")

        st.success("✅ Анализ завершён!")

        # Результаты
        st.header("📊 Результаты")

        col1, col2, col3 = st.columns(3)
        col1.metric("Точность совпадения", f"{avg_similarity:.1f}%", delta=f"{avg_similarity - 50:.1f}%")
        col2.metric("Баланс (Мастер)", f"{avg_balance_master * 100:.1f}%", delta="stable")
        col3.metric("Баланс (Ученик)", f"{avg_balance_student * 100:.1f}%", delta=f"{(avg_balance_student - avg_balance_master) * 100:.1f}%")

        # Видео сравнения
        st.header("📹 Видео сравнения")
        if os.path.exists(output_path):
            st.video(output_path)
            st.success(f"Файл сохранён: {output_path}")

            # Кнопка скачивания
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 Скачать видео",
                    data=f,
                    file_name=os.path.basename(output_path),
                    mime='video/mp4'
                )
        else:
            st.error("Не удалось создать видео")

        # Информация о сегментах
        if segments_master:
            st.header("📋 Детали разбиения на формы")
            st.write(f"Всего обнаружено форм: **{len(segments_master)}**")
            st.write("Формы отображаются на видео в верхнем левом углу с прогресс-баром.")

        engine.close()

    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Информация
st.markdown("---")
st.markdown("""
### ℹ️ О проекте

Этот инструмент сравнивает движения Тайцзи между мастером и учеником используя:
- **Computer Vision** (MediaPipe) для распознавания поз
- **DTW алгоритм** для синхронизации по времени
- **Анализ баланса** для оценки устойчивости
- **Детекция форм** для разбиения на смысловые сегменты

**Ветка:** `feature/comparison-mvp`  
**Статус:** В разработке 🚧  
**Автор:** Васильев Кэскил
""")