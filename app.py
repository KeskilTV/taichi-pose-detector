"""
🥋 TaiChi Comparison Tool
Веб-интерфейс для сравнения движений мастера и ученика
Версия 0.5.0: Ручная настройка кадров начала

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
import matplotlib.pyplot as plt
import cv2

# Настройка страницы
st.set_page_config(
    page_title="TaiChi Comparison Tool",
    page_icon="🥋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("🥋 TaiChi Comparison Tool")
st.markdown("**Сравнение движений Тайцзи с синхронизацией и анализом форм**")
st.markdown("---")

# Боковая панель
st.sidebar.header("⚙️ Настройки")
st.sidebar.markdown(f"**Автор:** Васильев Кэскил")
st.sidebar.markdown(f"**Версия:** 0.5.0 MVP")
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
master_total_frames = 0
student_total_frames = 0

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

        # Получаем количество кадров
        cap = cv2.VideoCapture(master_path)
        master_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        master_fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        st.success(f"✓ Загружено: {master_file.name}")
        st.info(f"📊 {master_total_frames} кадров ({master_total_frames/master_fps:.1f} сек)")

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

        # Получаем количество кадров
        cap = cv2.VideoCapture(student_path)
        student_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        student_fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        st.success(f"✓ Загружено: {student_file.name}")
        st.info(f"📊 {student_total_frames} кадров ({student_total_frames/student_fps:.1f} сек)")

# Если видео загружены - показываем настройку кадров начала
if master_path and student_path:
    st.markdown("---")
    st.header("✂️ Настройка начала движения")

    # Инициализация session_state для ползунков
    if 'master_start_frame' not in st.session_state:
        st.session_state.master_start_frame = 0
    if 'student_start_frame' not in st.session_state:
        st.session_state.student_start_frame = 0

    # Автоматическое определение (только один раз при загрузке видео)
    if 'auto_starts_detected' not in st.session_state:
        engine = ComparisonEngine()
        poses_master_test = engine.extract_poses(master_path)
        poses_student_test = engine.extract_poses(student_path)

        st.session_state.auto_master_start = engine.find_movement_start(poses_master_test, threshold=0.02)
        st.session_state.auto_student_start = engine.find_movement_start(poses_student_test, threshold=0.02)
        st.session_state.auto_starts_detected = True

        # Устанавливаем начальные значения
        st.session_state.master_start_frame = st.session_state.auto_master_start
        st.session_state.student_start_frame = st.session_state.auto_student_start

    st.info(
        f"🤖 Автоматически определено: Мастер = кадр {st.session_state.auto_master_start}, Ученик = кадр {st.session_state.auto_student_start}")

    # Ручная настройка через ползунки
    col1, col2 = st.columns(2)

    with col1:
        master_start_frame = st.slider(
            "📹 Мастер: начать с кадра",
            min_value=0,
            max_value=master_total_frames - 1,
            value=st.session_state.master_start_frame,
            step=1,
            key='master_slider'
        )
        st.session_state.master_start_frame = master_start_frame

    with col2:
        student_start_frame = st.slider(
            "📹 Ученик: начать с кадра",
            min_value=0,
            max_value=student_total_frames - 1,
            value=st.session_state.student_start_frame,
            step=1,
            key='student_slider'
        )
        st.session_state.student_start_frame = student_start_frame

    # Предпросмотр кадров
    st.subheader("🔍 Предпросмотр выбранных кадров")

    col1, col2 = st.columns(2)

    with col1:
        cap = cv2.VideoCapture(master_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.master_start_frame)
        ret, frame = cap.read()
        if ret:
            st.image(frame, caption=f"Мастер - кадр {st.session_state.master_start_frame}", channels="BGR")
        else:
            st.error("Не удалось загрузить кадр")
        cap.release()

    with col2:
        cap = cv2.VideoCapture(student_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.student_start_frame)
        ret, frame = cap.read()
        if ret:
            st.image(frame, caption=f"Ученик - кадр {st.session_state.student_start_frame}", channels="BGR")
        else:
            st.error("Не удалось загрузить кадр")
        cap.release()

    # Предупреждение если кадры слишком близки к концу
    if master_total_frames - st.session_state.master_start_frame < 100:
        st.warning(
            f"⚠️ Мастер: осталось только {master_total_frames - st.session_state.master_start_frame} кадров после начала")
    if student_total_frames - st.session_state.student_start_frame < 100:
        st.warning(
            f"⚠️ Ученик: осталось только {student_total_frames - st.session_state.student_start_frame} кадров после начала")
    # Кнопка для обновления предпросмотра (чтобы не грузить постоянно)
    st.subheader("🔍 Предпросмотр выбранных кадров")

    if st.button("🔄 Обновить предпросмотр", key='update_preview_btn'):
        with st.spinner("Загрузка кадров..."):
            col1, col2 = st.columns(2)

            with col1:
                cap = cv2.VideoCapture(master_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.master_start_frame)
                ret, frame = cap.read()
                if ret:
                    st.image(frame, caption=f"Мастер - кадр {st.session_state.master_start_frame}", channels="BGR")
                else:
                    st.error("Не удалось загрузить кадр")
                cap.release()

            with col2:
                cap = cv2.VideoCapture(student_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.student_start_frame)
                ret, frame = cap.read()
                if ret:
                    st.image(frame, caption=f"Ученик - кадр {st.session_state.student_start_frame}", channels="BGR")
                else:
                    st.error("Не удалось загрузить кадр")
                cap.release()
    else:
        # Показываем предпросмотр по умолчанию (первые кадры)
        col1, col2 = st.columns(2)

        with col1:
            cap = cv2.VideoCapture(master_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.master_start_frame)
            ret, frame = cap.read()
            if ret:
                st.image(frame, caption=f"Мастер - кадр {st.session_state.master_start_frame}", channels="BGR")
            cap.release()

        with col2:
            cap = cv2.VideoCapture(student_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.student_start_frame)
            ret, frame = cap.read()
            if ret:
                st.image(frame, caption=f"Ученик - кадр {st.session_state.student_start_frame}", channels="BGR")
            cap.release()

    # Предупреждение если кадры слишком близки к концу
    if master_total_frames - st.session_state.master_start_frame < 100:
        st.warning(
            f"⚠️ Мастер: осталось только {master_total_frames - st.session_state.master_start_frame} кадров после начала")
    if student_total_frames - st.session_state.student_start_frame < 100:
        st.warning(
            f"⚠️ Ученик: осталось только {student_total_frames - st.session_state.student_start_frame} кадров после начала")
# Кнопка запуска анализа
st.markdown("---")
st.header("🔍 Анализ")

if st.button("🚀 Запустить сравнение", type='primary', disabled=not (master_path and student_path)):
    st.info("⏳ Обработка видео... (это займёт некоторое время)")

    try:
        engine = ComparisonEngine()
        dtw_aligner = DTWAligner(step=5, window_size=50)

        progress_bar = st.progress(0)
        status_text = st.empty()

        # ==========================================
        # Шаг 1: Извлечение поз
        # ==========================================
        status_text.text("Извлечение поз мастера...")
        poses_master = engine.extract_poses(master_path)
        progress_bar.progress(15)

        status_text.text("Извлечение поз ученика...")
        poses_student = engine.extract_poses(student_path)
        progress_bar.progress(30)

        # ==========================================
        # Шаг 2: Обрезка по выбранным кадрам
        # ==========================================
        status_text.text(f"Обрезка видео (Мастер: {master_start_frame}, Ученик: {student_start_frame})...")

        st.info(f"✂️ Мастер: обрезано {master_start_frame} кадров | Ученик: обрезано {student_start_frame} кадров")

        # Обрезаем позы
        poses_master_trimmed = engine.trim_poses(poses_master, master_start_frame)
        poses_student_trimmed = engine.trim_poses(poses_student, student_start_frame)

        progress_bar.progress(45)

        # ==========================================
        # Шаг 3: DTW синхронизация
        # ==========================================
        status_text.text("Синхронизация скорости (DTW)...")

        poses_student_aligned, dtw_path, master_idx_list, student_idx_list = dtw_aligner.align_poses(
            poses_master_trimmed, poses_student_trimmed
        )

        progress_bar.progress(60)

        # ==========================================
        # Шаг 4: Детекция сегментов (форм)
        # ==========================================
        status_text.text("Разбиение на формы Тайцзи...")
        segments_master = engine.segment_detector.detect_segments(poses_master_trimmed)
        progress_bar.progress(70)

        # Показываем найденные сегменты
        if segments_master:
            with st.expander(f"📍 Найденные формы Тайцзи ({len(segments_master)})"):
                for i, seg in enumerate(segments_master):
                    st.write(f"**{i+1}. {seg['name']}** — кадры {seg['start']}–{seg['end']} ({seg['end']-seg['start']} кадров)")

        # ==========================================
        # Шаг 5: Визуализация DTW пути
        # ==========================================
        status_text.text("Анализ синхронизации...")

        with st.expander("📈 График DTW синхронизации (развернуть)"):
            if dtw_path and len(dtw_path) > 10:
                sample_step = max(1, len(dtw_path) // 200)
                m_indices = [p[0] for p in dtw_path[::sample_step]]
                s_indices = [p[1] for p in dtw_path[::sample_step]]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(m_indices, s_indices, 'b-', linewidth=1.5, label='DTW путь')
                ax.plot([0, max(m_indices)], [0, max(s_indices)], 'r--', alpha=0.5, label='Идеальная синхронность')
                ax.set_xlabel('Кадр мастера', fontsize=12)
                ax.set_ylabel('Кадр ученика', fontsize=12)
                ax.set_title('Синхронизация видео (DTW Path)', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                st.caption("""
                **Как читать график:**
                - 🔵 **Синяя линия** — реальная синхронизация.
                - 🔴 **Красная пунктирная** — идеальное совпадение скоростей.
                - 📈 **Линия выше** — ученик медленнее мастера.
                - 📉 **Линия ниже** — ученик быстрее мастера.
                """)

        progress_bar.progress(80)

        # ==========================================
        # Шаг 6: Создание видео сравнения
        # ==========================================
        status_text.text("Создание видео сравнения...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_videos/comparison_{timestamp}.mp4"

        output_path, segments, dtw_path_saved = engine.create_comparison_video(
            master_path, student_path, output_path,
            poses_master_trimmed,
            poses_student_aligned,
            segments_master=segments_master,
            dtw_path=dtw_path,
            start_offset_master=master_start_frame,
            start_offset_student=student_start_frame
        )
        progress_bar.progress(95)

        # ==========================================
        # Шаг 7: Расчет метрик
        # ==========================================
        status_text.text("Расчет метрик...")

        similarities = []
        balances_master = []
        balances_student = []

        for i in range(min(len(poses_master_trimmed), len(poses_student_aligned))):
            sim = engine.calculate_pose_similarity(poses_master_trimmed[i], poses_student_aligned[i])
            similarities.append(sim)

            if poses_master_trimmed[i] is not None:
                balances_master.append(engine.calculate_balance(poses_master_trimmed[i]))
            if poses_student_aligned[i] is not None:
                balances_student.append(engine.calculate_balance(poses_student_aligned[i]))

        avg_similarity = np.mean(similarities) if similarities else 0
        avg_balance_master = np.mean(balances_master) if balances_master else 0
        avg_balance_student = np.mean(balances_student) if balances_student else 0

        progress_bar.progress(100)
        status_text.text("✅ Готово!")

        st.success("✅ Анализ завершён!")

        # ==========================================
        # Результаты
        # ==========================================
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

            with open(output_path, 'rb') as f:
                st.download_button(
                    label="📥 Скачать видео",
                    data=f,
                    file_name=os.path.basename(output_path),
                    mime='video/mp4'
                )
        else:
            st.error("Не удалось создать видео")

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
- **Ручная настройка** кадров начала для точного контроля
- **Детекция форм** для разбиения на смысловые сегменты

**Ветка:** `feature/comparison-mvp`  
**Статус:** В разработке 🚧  
**Автор:** Васильев Кэскил
""")