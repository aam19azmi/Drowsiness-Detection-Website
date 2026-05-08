import streamlit as st
import streamlit.components.v1 as components
import cv2
import tempfile
import os
import time
import pytesseract
import re
import difflib 
import zipfile 
import math
import pandas as pd
import numpy as np
import mediapipe as mp
from PIL import Image
from datetime import date
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as ExcelImage

# ==========================================
# 1. KONFIGURASI AI & MEDIAPIPE
# ==========================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Indeks Landmark Penting
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178] 

LEFT_IRIS_CENTER = 473; RIGHT_IRIS_CENTER = 468
LEFT_IRIS_LEFT = 476; LEFT_IRIS_RIGHT = 474
RIGHT_IRIS_LEFT = 471; RIGHT_IRIS_RIGHT = 469
NOSE_TIP = 1 

# Threshold Standar (Biometrik)
EAR_LOW = 0.20
EAR_HIGH = 0.25
EAR_P80 = 0.21
MICROSLEEP_THRESH = 0.5
MAR_THRESH = 0.50
YAWN_MIN_TIME = 1.5
SACCADE_THRESHOLD = 0.005
GAZE_CENTER_MARGIN = [0.35, 0.65]
NOD_PITCH_THRESH = -15.0

# Titik Face 3D untuk Pose Estimation
FACE_3D = np.array([
    [0.0, 0.0, 0.0],            
    [0.0, -330.0, -65.0],       
    [-225.0, 170.0, -135.0],    
    [225.0, 170.0, -135.0],     
    [-150.0, -150.0, -125.0],   
    [150.0, -150.0, -125.0]     
], dtype=np.float64)

# ==========================================
# 2. FUNGSI MATEMATIKA (Kalkulasi Jarak)
# ==========================================
def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calculate_ear(landmarks, frame_w, frame_h, idxs):
    points = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in idxs]
    v1 = euclidean_distance(points[1], points[5])
    v2 = euclidean_distance(points[2], points[4])
    h1 = euclidean_distance(points[0], points[3])
    return (v1 + v2) / (2.0 * h1), points

def calculate_mar(landmarks, frame_w, frame_h, idxs):
    points = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in idxs]
    v_dist = euclidean_distance(points[2], points[6]) 
    h_dist = euclidean_distance(points[0], points[4]) 
    return v_dist / h_dist if h_dist != 0 else 0, points

def calculate_gaze_ratio(landmarks, frame_w, frame_h):
    l_inner = (landmarks[133].x * frame_w, landmarks[133].y * frame_h)
    l_outer = (landmarks[33].x * frame_w, landmarks[33].y * frame_h)
    l_iris = (landmarks[LEFT_IRIS_CENTER].x * frame_w, landmarks[LEFT_IRIS_CENTER].y * frame_h)
    jarak_total = euclidean_distance(l_inner, l_outer)
    jarak_iris_ke_dalam = euclidean_distance(l_iris, l_inner)
    if jarak_total == 0: return 0.5
    return jarak_iris_ke_dalam / jarak_total

# ==========================================
# 3. FUNGSI CORE ANALYSIS (Deteksi & Export)
# ==========================================
def run_live_analysis(video_path, id_kerja="-", nama_pegawai="Unknown", jenis_kelamin="-", usia=0, status_shift="Tidak Diketahui", is_live=False, crop_start=0.0, crop_end=0.0):
    cap = cv2.VideoCapture(0 if is_live else video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- MENGHITUNG TOTAL DURASI VIDEO (UNTUK LOGIKA POTONG AKHIR) ---
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_duration = total_frames / fps if fps > 0 else 0
    absolute_stop_time = total_duration - crop_end if not is_live else 0.0

    video_out_name = f"Rekaman_{nama_pegawai}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_out_name, fourcc, fps, (width, height))

    cam_matrix = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    state = 0; t1 = t2 = t3 = t4 = 0.0
    blinks = []; p80_frames = 0
    frame_count = 0
    
    saccade_amplitudes = []; total_bursts = total_yawns = 0
    center_gaze_frames = 0; max_yawn_mar = 0.0
    brightness_levels = []; yaw_angles = []; pitch_angles = []
    is_yawning = False; yawn_start_time = 0.0
    prev_pitch = 0; prev_rel_l_iris = None
    total_head_nods = 0

    thumbnail_saved = False
    temp_img_path = f"web_thumbnail_{nama_pegawai}.jpg" 
    
    frames_analyzed = 1 

    if not is_live and crop_start > 0.0:
        cap.set(cv2.CAP_PROP_POS_MSEC, crop_start * 1000)
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    start_time_live = time.time()

    while cap.isOpened():
        if is_live:
            elapsed_time = time.time() - start_time_live
            if elapsed_time >= 180.0: break 
            
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        time_sec = frame_count / fps

        # --- BATAS CROP END VIDEO BERDASARKAN DURASI ASLI ---
        if not is_live and crop_end > 0.0 and time_sec >= absolute_stop_time:
            break
        
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, _, _ = cv2.split(lab)
        brightness_levels.append(np.mean(l_channel))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            if not thumbnail_saved and frame_count > 15: 
                cv2.imwrite(temp_img_path, cv2.resize(frame, (150, 100)))
                thumbnail_saved = True

            landmarks = results.multi_face_landmarks[0].landmark

            mp_drawing.draw_landmarks(
                image=rgb_frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=rgb_frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=rgb_frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

            face_2d = np.array([
                (landmarks[1].x * width, landmarks[1].y * height),
                (landmarks[152].x * width, landmarks[152].y * height),
                (landmarks[33].x * width, landmarks[33].y * height),
                (landmarks[263].x * width, landmarks[263].y * height),
                (landmarks[61].x * width, landmarks[61].y * height),
                (landmarks[291].x * width, landmarks[291].y * height)
            ], dtype=np.float64)

            success_pnp, rot_vec, trans_vec = cv2.solvePnP(FACE_3D, face_2d, cam_matrix, dist_coeffs)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            
            pitch = angles[0] * 360
            yaw = angles[1] * 360
            yaw_angles.append(yaw); pitch_angles.append(pitch)

            if pitch < NOD_PITCH_THRESH and prev_pitch >= NOD_PITCH_THRESH: total_head_nods += 1
            prev_pitch = pitch

            gaze_ratio = calculate_gaze_ratio(landmarks, width, height)
            if GAZE_CENTER_MARGIN[0] < gaze_ratio < GAZE_CENTER_MARGIN[1]: center_gaze_frames += 1

            nose = landmarks[NOSE_TIP]
            l_iris = landmarks[LEFT_IRIS_CENTER]
            rel_l_x, rel_l_y = (l_iris.x - nose.x), (l_iris.y - nose.y)
            
            if prev_rel_l_iris is not None and state == 0:
                dist_l = math.hypot(rel_l_x - prev_rel_l_iris[0], rel_l_y - prev_rel_l_iris[1])
                if dist_l > SACCADE_THRESHOLD: saccade_amplitudes.append(dist_l * 1000)
            prev_rel_l_iris = (rel_l_x, rel_l_y)

            mar_val, _ = calculate_mar(landmarks, width, height, MOUTH)
            if mar_val > MAR_THRESH:
                max_yawn_mar = max(max_yawn_mar, mar_val)
                if not is_yawning: is_yawning, yawn_start_time = True, time_sec
            else:
                if is_yawning and (time_sec - yawn_start_time) > YAWN_MIN_TIME: total_yawns += 1
                is_yawning = False

            l_ear, _ = calculate_ear(landmarks, width, height, LEFT_EYE)
            r_ear, _ = calculate_ear(landmarks, width, height, RIGHT_EYE)
            ear_avg = (l_ear + r_ear) / 2.0

            if ear_avg <= EAR_P80: p80_frames += 1

            if state == 0 and ear_avg < EAR_HIGH: t1, state = time_sec, 1
            elif state == 1:
                if ear_avg <= EAR_LOW: t2, state = time_sec, 2
                elif ear_avg >= EAR_HIGH: state = 0
            elif state == 2:
                if ear_avg > EAR_LOW: t3, state = time_sec, 3
            elif state == 3:
                if ear_avg >= EAR_HIGH:
                    t4 = time_sec
                    interval = (t1 - blinks[-1]['t4']) if len(blinks) > 0 else 0
                    if 0 < interval < 1.0: total_bursts += 1
                    blinks.append({
                        "t1": t1, "t2": t2, "t3": t3, "t4": t4, 
                        "closed": t3 - t2, "closing": t2 - t1, "reopen": t4 - t3, 
                        "duration": t4 - t1, "interval": interval, 
                        "ms_flag": 1 if (t3 - t2) >= MICROSLEEP_THRESH else 0
                    })
                    state = 0
                elif ear_avg <= EAR_LOW: state = 2

        frames_analyzed = frame_count if (crop_start == 0.0 or is_live) else (frame_count - int(crop_start * fps))
        if frames_analyzed <= 0: frames_analyzed = 1 
        
        perclos_live = (p80_frames / frames_analyzed) * 100 if frames_analyzed > 0 else 0
        freq_kedipan = len(blinks)
        mcd_live = sum(b['closed'] for b in blinks) / freq_kedipan if freq_kedipan > 0 else 0
        avg_blink_dur = sum(b['duration'] for b in blinks) / freq_kedipan if freq_kedipan > 0 else 0
        total_ms_live = sum(b['ms_flag'] for b in blinks)
        
        if perclos_live >= 12 or total_ms_live > 0: status = "4 - BAHAYA (KRITIS)"
        elif mcd_live >= 0.4 or perclos_live >= 10: status = "3 - LELAH (Risiko Tinggi)"
        elif mcd_live >= 0.25 or perclos_live >= 5: status = "2 - KURANG FIT"
        else: status = "1 - FIT (Aman)"

        bgr_out = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR) 
        
        cv2.rectangle(bgr_out, (10, 10), (320, 240), (0, 0, 0), -1) 
        cv2.addWeighted(bgr_out, 0.6, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), 0.4, 0, bgr_out) 

        warna_teks = (0, 255, 0) if "FIT" in status else (0, 0, 255) 
        
        cv2.putText(bgr_out, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna_teks, 2)
        cv2.putText(bgr_out, f"PERCLOS: {perclos_live:.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bgr_out, f"MCD: {mcd_live:.3f}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bgr_out, f"Kedipan: {freq_kedipan}x", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bgr_out, f"Blink Dur: {avg_blink_dur:.3f}s", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        warna_ms = (0, 0, 255) if total_ms_live > 0 else (255, 255, 255)
        cv2.putText(bgr_out, f"Microsleep: {total_ms_live}x", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_ms, 1)

        if total_ms_live > 0:
            cv2.putText(bgr_out, f"!!! MICROSLEEP DETECTED !!!", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        out_video.write(bgr_out) 

        yield rgb_frame, status, perclos_live, mcd_live, freq_kedipan, avg_blink_dur, total_ms_live

    cap.release()
    out_video.release() 

    if not thumbnail_saved:
        cv2.imwrite(temp_img_path, np.zeros((100, 150, 3), dtype=np.uint8))

    excel_name = f"Laporan_{nama_pegawai}.xlsx"
    
    avg_saccade_amp = sum(saccade_amplitudes) / len(saccade_amplitudes) if len(saccade_amplitudes) > 0 else 0
    gaze_center_percent = (center_gaze_frames / frames_analyzed) * 100 if frames_analyzed > 0 else 0
    avg_brightness_final = sum(brightness_levels) / len(brightness_levels) if len(brightness_levels) > 0 else 0
    avg_yaw_final = sum(yaw_angles) / len(yaw_angles) if len(yaw_angles) > 0 else 0
    
    output_rows = []
    output_rows.append(["ID Kerja", "Nama", "Jenis Kelamin", "Usia", "Screenshot", "Waktu Mulai Tutup", "Waktu Tertutup", "Waktu Mulai Buka", 
                        "Waktu Buka Penuh", "Closed Phase", "Closing Phase", "Reopening Phase", 
                        "Blink Duration", "Blink Interval", "Microsleep Event", 
                        "MCD (Kesimpulan)", "Blink Burst", "PERCLOS P80", "Saccade Amp", 
                        "Tunnel Vision", "Head Nods", "Total Yawn", "Max Yawn MAR", 
                        "Cahaya Rata-rata", "Sudut Yaw Kamera", "STATUS KLASIFIKASI", "Status Shift"])
                        
    output_rows.append(["[-]", "[-]", "[-]", "[Tahun]", "[-]", "[s]", "[s]", "[s]", 
                        "[s]", "[s]", "[s]", "[s]", 
                        "[s]", "[s]", "[0/1]", 
                        "[s]", "[Kali]", "[%]", "[px/°]", 
                        "[%]", "[Kali]", "[Kali]", "[Rasio]", 
                        "[Lux Px]", "[Derajat]", "[Kesimpulan Sistem]", "[-]"])
    
    if freq_kedipan == 0:
        output_rows.append([
            id_kerja, nama_pegawai, jenis_kelamin, usia, "", 0,0,0,0, 0,0,0, 0,0, 0, 
            0, total_bursts, round(perclos_live, 2), round(avg_saccade_amp, 2), 
            round(gaze_center_percent, 2), total_head_nods, total_yawns, round(max_yawn_mar, 2), 
            round(avg_brightness_final, 1), round(avg_yaw_final, 1), status, status_shift
        ])
    else:
        for idx, b in enumerate(blinks):
            output_rows.append([
                id_kerja if idx == 0 else "",
                nama_pegawai if idx == 0 else "", 
                jenis_kelamin if idx == 0 else "",
                usia if idx == 0 else "",
                "", 
                round(b['t1'], 3), round(b['t2'], 3), round(b['t3'], 3), round(b['t4'], 3),
                round(b['closed'], 3), round(b['closing'], 3), round(b['reopen'], 3), 
                round(b['duration'], 3), round(b['interval'], 3), b['ms_flag'], 
                
                round(mcd_live, 3) if idx == 0 else "",
                total_bursts if idx == 0 else "",
                round(perclos_live, 2) if idx == 0 else "",
                round(avg_saccade_amp, 2) if idx == 0 else "",
                round(gaze_center_percent, 2) if idx == 0 else "",
                total_head_nods if idx == 0 else "",
                total_yawns if idx == 0 else "",
                round(max_yawn_mar, 2) if idx == 0 else "",
                round(avg_brightness_final, 1) if idx == 0 else "",
                round(avg_yaw_final, 1) if idx == 0 else "",
                status if idx == 0 else "",
                status_shift if idx == 0 else ""
            ])

    df = pd.DataFrame(output_rows)
    df.to_excel(excel_name, index=False, header=False)

    if os.path.exists(temp_img_path):
        wb = load_workbook(excel_name)
        ws = wb.active
        ws.column_dimensions['E'].width = 25 
        ws.row_dimensions[3].height = 80 
        img = ExcelImage(temp_img_path)
        ws.add_image(img, 'E3')
        wb.save(excel_name)
        os.remove(temp_img_path)
        
    yield "DONE", excel_name, video_out_name, 0, 0, 0, 0


# ==========================================
# 4. DASHBOARD UTAMA (STREAMLIT ANTARMUKA)
# ==========================================
# Konfigurasi Tesseract Multi-Platform
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(layout="wide", page_title="Dashboard K3 Tambang")

# PENGATURAN STATE UNTUK PINDAH HALAMAN
if "halaman" not in st.session_state:
    st.session_state.halaman = "beranda"

def pindah_halaman(nama_halaman):
    st.session_state.halaman = nama_halaman

# HALAMAN 1: BERANDA / MENU UTAMA
if st.session_state.halaman == "beranda":
    st.title("🚜 Sistem Pemantauan FFD (Fitness-for-Duty)")
    st.markdown("### Selamat Datang di Dashboard Evaluasi Kelelahan Operator")
    st.markdown("Silakan pilih modul pengujian yang ingin dilakukan:")
    
    st.markdown("<br><br>", unsafe_allow_html=True) 
    
    col_menu1, col_menu2 = st.columns(2)
    
    with col_menu1:
        st.info("👁️ **Analisis FFD (MediaPipe)**\n\nPengujian tingkat kewaspadaan berbasis deteksi biometrik wajah (PERCLOS, MCD, Kedipan, Microsleep).")
        if st.button("Masuk ke Modul Analisis FFD", type="primary", use_container_width=True):
            pindah_halaman("analisis")
            st.rerun()
            
    with col_menu2:
        st.warning("⏱️ **PVT Task (Vigilance)**\n\nPengujian waktu reaksi (Reaction Time) untuk menilai kelelahan kognitif dan kewaspadaan motorik pekerja.")
        st.link_button("Buka Modul PVT Task", "https://pvt-v3.vercel.app/auth", use_container_width=True)

# HALAMAN 2: MODUL ANALISIS (UPLOAD & LIVE)
elif st.session_state.halaman == "analisis":
    if st.button("⬅️ Kembali ke Menu Utama"):
        pindah_halaman("beranda")
        st.rerun()
        
    st.title("🖥️ Modul Analisis FFD Operator")

    tab1, tab2 = st.tabs(["📂 Upload Video", "📷 Live Kamera"])

    # ------------------------------------------
    # TAB 1: UPLOAD VIDEO
    # ------------------------------------------
    with tab1:
        st.markdown("### Uji Coba via Rekaman Video")
        
        col_input1, col_input2 = st.columns(2)
        with col_input1:
            id_kerja_up = st.text_input("ID Kerja", key="id_up")
            nama_upload = st.text_input("Nama Pegawai", key="nama_up")
        with col_input2:
            jenis_kelamin_up = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], key="jk_up")
            tgl_lahir_up = st.date_input("Tanggal Lahir", min_value=date(1950, 1, 1), max_value=date.today(), key="tgl_up")
            usia_up = date.today().year - tgl_lahir_up.year - ((date.today().month, date.today().day) < (tgl_lahir_up.month, tgl_lahir_up.day))
            st.caption(f"Usia saat ini: {usia_up} Tahun")

        status_upload = st.selectbox("Status Pengujian", ["Pre-Shift", "Post-Shift", "Fatigue testing"], key="stat1")
        uploaded_file = st.file_uploader("Upload video evaluasi", type=['mp4', 'avi', 'mov'])
        
        st.markdown("#### ✂️ Pengaturan Pemotongan Video (Crop)")
        crop_option = st.radio("Opsi Analisis:", ["Analisis Penuh (Tanpa Potong)", "Potong Awal & Akhir"], horizontal=True)
        crop_start_sec = 0.0
        crop_end_sec = 0.0
        
        if crop_option == "Potong Awal & Akhir":
            c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
            val_start = c1.number_input("Buang/Potong bagian AWAL:", min_value=0.0, value=0.0, step=1.0)
            unit_start = c2.selectbox("Satuan Awal", ["Detik", "Menit"], key="unit_start")
            
            val_end = c3.number_input("Buang/Potong bagian AKHIR:", min_value=0.0, value=0.0, step=1.0)
            unit_end = c4.selectbox("Satuan Akhir", ["Detik", "Menit"], key="unit_end")
            
            crop_start_sec = val_start if unit_start == "Detik" else val_start * 60.0
            crop_end_sec = val_end if unit_end == "Detik" else val_end * 60.0

        if uploaded_file is not None:
            if st.button("▶ Mulai Analisis Video", type="primary", use_container_width=True):
                col_vid, col_stat = st.columns([2, 1])
                with col_vid: vid_ph = st.empty()
                with col_stat:
                    st.subheader("Status Operator")
                    status_ui = st.empty()
                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    m4, m5, m6 = st.columns(3)
                    perclos_ui = m1.empty(); mcd_ui = m2.empty(); blink_ui = m3.empty()
                    dur_ui = m4.empty(); ms_ui = m5.empty()
                
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                tfile.close()

                excel_result = ""
                video_result = ""
                for frame, status, perclos, mcd, blinks, blink_dur, total_ms in run_live_analysis(
                    video_path, id_kerja_up, nama_upload, jenis_kelamin_up, usia_up, status_upload, False, crop_start_sec, crop_end_sec):
                    
                    if isinstance(frame, str) and frame == "DONE":
                        excel_result = status 
                        video_result = perclos 
                        break
                    
                    vid_ph.image(frame, channels="RGB", use_container_width=True)
                    if "BAHAYA" in status: status_ui.error(f"🚨 **{status}**")
                    elif "LELAH" in status: status_ui.warning(f"⚠️ **{status}**")
                    elif "KURANG FIT" in status: status_ui.info(f"👀 **{status}**")
                    else: status_ui.success(f"✅ **{status}**")
                        
                    perclos_ui.metric("PERCLOS", f"{perclos:.1f}%")
                    mcd_ui.metric("MCD", f"{mcd:.3f}s")
                    blink_ui.metric("Kedipan", f"{blinks}x")
                    dur_ui.metric("Durasi", f"{blink_dur:.3f}s")
                    ms_ui.metric("Microsleep", f"{total_ms}x")

                st.success("✅ Analisis Selesai!")
                
                zip_name_up = f"Hasil_Upload_{nama_upload}.zip"
                with zipfile.ZipFile(zip_name_up, 'w') as zipf:
                    if os.path.exists(excel_result): zipf.write(excel_result)
                    if os.path.exists(video_result): zipf.write(video_result)
                    
                if os.path.exists(zip_name_up):
                    with open(zip_name_up, "rb") as f:
                        st.download_button("📦 Download Paket Bukti (ZIP)", data=f, file_name=zip_name_up, mime="application/zip")
                        
                os.remove(video_path)

    # ------------------------------------------
    # TAB 2: LIVE KAMERA
    # ------------------------------------------
    with tab2:
        st.markdown("### Uji Coba Langsung (3 Menit)")
        
        col_live1, col_live2 = st.columns(2)
        with col_live1:
            id_kerja_live = st.text_input("ID Kerja", key="id_live")
            metode_input = st.radio("Pilih Metode Input Nama:", ["Ketik Manual", "Scan KTP/ID Card"])
            nama_live = "" 
            if metode_input == "Ketik Manual":
                nama_live = st.text_input("Masukkan Nama Pegawai", key="nama_live")
            else:
                foto_ktp = st.camera_input("Jepret ID Card")
                if foto_ktp is not None:
                    img_pil = Image.open(foto_ktp)
                    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    with st.spinner("OCR NLP sedang bekerja..."):
                        teks_hasil = pytesseract.image_to_string(thresh, config='--psm 6')
                        baris_teks = teks_hasil.split('\n')
                        nama_ditemukan = ""
                        for baris in baris_teks:
                            kata_kata = baris.split()
                            if not kata_kata: continue
                            target_kata = ['nama', 'name']
                            kemiripan = difflib.get_close_matches(kata_kata[0].lower(), target_kata, n=1, cutoff=0.6)
                            if kemiripan:
                                teks_setelah_keyword = baris[len(kata_kata[0]):].strip()
                                teks_bersih = re.sub(r'^[:;|=]\s*', '', teks_setelah_keyword).strip()
                                if len(teks_bersih) > 2:
                                    nama_ditemukan = teks_bersih
                                    break
                        if nama_ditemukan:
                            st.success(f"NLP Mendeteksi: **{nama_ditemukan}**")
                            nama_live = st.text_input("Konfirmasi Nama:", value=nama_ditemukan)
                        else:
                            st.warning("Nama tidak terdeteksi. Silakan ketik manual.")
                            nama_live = st.text_input("Ketik Nama Manual:")
                            
        with col_live2:
            jenis_kelamin_live = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], key="jk_live")
            tgl_lahir_live = st.date_input("Tanggal Lahir", min_value=date(1950, 1, 1), max_value=date.today(), key="tgl_live")
            usia_live = date.today().year - tgl_lahir_live.year - ((date.today().month, date.today().day) < (tgl_lahir_live.month, tgl_lahir_live.day))
            st.caption(f"Usia saat ini: {usia_live} Tahun")
            status_live = st.selectbox("Status Pengujian", ["Pre-Shift", "Post-Shift", "Fatigue testing"], key="stat2")
        
        if st.button("🔴 Mulai Tes FFD 3 Menit", type="primary", use_container_width=True) and nama_live != "":
            yt_ph = st.empty()
            with yt_ph:
                components.html(
                    """<iframe width="100%" height="400" src="https://www.youtube.com/embed/Se5NjX-cM5I?si=__OUHuj-V2w_wi4J&autoplay=1" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>""",
                    height=400
                )
            
            st.markdown("---")
            p_text = st.empty()
            p_bar = st.progress(0)
            
            col_vid2, col_stat2 = st.columns([2, 1])
            with col_vid2: vid_ph2 = st.empty()
            with col_stat2:
                st.subheader("Monitoring")
                status_ui2 = st.empty()
                st.markdown("---")
                m7, m8, m9 = st.columns(3)
                m10, m11, m12 = st.columns(3)
                perclos_ui2 = m7.empty(); mcd_ui2 = m8.empty(); blink_ui2 = m9.empty()
                dur_ui2 = m10.empty(); ms_ui2 = m11.empty()

            waktu_mulai = time.time()
            excel_result_live = ""
            video_result_live = ""
            
            for frame, status, perclos, mcd, blinks, blink_dur, total_ms in run_live_analysis(
                "", id_kerja_live, nama_live, jenis_kelamin_live, usia_live, status_live, True):
                
                if isinstance(frame, str) and frame == "DONE":
                    excel_result_live = status
                    video_result_live = perclos 
                    break
                    
                elapsed = time.time() - waktu_mulai
                p_bar.progress(min(elapsed / 180.0, 1.0))
                p_text.markdown(f"**Progress: {int(elapsed)} / 180 Detik**")
                
                vid_ph2.image(frame, channels="RGB", use_container_width=True)
                
                if "BAHAYA" in status: status_ui2.error(f"🚨 **{status}**")
                elif "LELAH" in status: status_ui2.warning(f"⚠️ **{status}**")
                elif "KURANG FIT" in status: status_ui2.info(f"👀 **{status}**")
                else: status_ui2.success(f"✅ **{status}**")
                    
                perclos_ui2.metric("PERCLOS", f"{perclos:.1f}%")
                mcd_ui2.metric("MCD", f"{mcd:.3f}s")
                blink_ui2.metric("Kedipan", f"{blinks}x")
                dur_ui2.metric("Durasi", f"{blink_dur:.3f}s")
                
                if total_ms > 0: ms_ui2.error(f"⚠️ {total_ms}x")
                else: ms_ui2.metric("Microsleep", f"{total_ms}x")
                
            p_bar.progress(1.0)
            yt_ph.empty() 
            st.success("✅ Sesi 3 Menit Selesai! Bukti Laporan & Video telah dibuat.")
            
            zip_name_live = f"Bukti_FFD_{nama_live}.zip"
            with zipfile.ZipFile(zip_name_live, 'w') as zipf:
                if os.path.exists(excel_result_live): zipf.write(excel_result_live)
                if os.path.exists(video_result_live): zipf.write(video_result_live)
                
            if os.path.exists(zip_name_live):
                with open(zip_name_live, "rb") as f:
                    st.download_button("📦 Download Paket Bukti (ZIP)", data=f, file_name=zip_name_live, mime="application/zip")
