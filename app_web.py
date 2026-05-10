import streamlit as st
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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av

# ==========================================
# 1. KONFIGURASI AI & MEDIAPIPE
# ==========================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH = [13, 14, 78, 308, 81, 82, 311, 312, 178, 402, 317, 318, 87, 88, 95, 324]
LEFT_IRIS_CENTER = 468
NOSE_TIP = 1

EAR_LOW = 0.20
EAR_HIGH = 0.25
EAR_P80 = 0.21
MICROSLEEP_THRESH = 0.5
MAR_THRESH = 0.50
YAWN_MIN_TIME = 1.5
SACCADE_THRESHOLD = 0.005
GAZE_CENTER_MARGIN = [0.35, 0.65]
NOD_PITCH_THRESH = -15.0

FACE_3D = np.array([
    [0.0, 0.0, 0.0],            
    [0.0, -330.0, -65.0],       
    [-225.0, 170.0, -135.0],    
    [225.0, 170.0, -135.0],     
    [-150.0, -150.0, -125.0],   
    [150.0, -150.0, -125.0]     
], dtype=np.float64)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ==========================================
# 2. FUNGSI MATEMATIKA & EXCEL
# ==========================================
def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calculate_ear(landmarks, frame_w, frame_h, idxs):
    points = [np.array([landmarks[i].x * frame_w, landmarks[i].y * frame_h]) for i in idxs]
    v1 = np.linalg.norm(points[12] - points[4])
    v2 = np.linalg.norm(points[11] - points[5])
    v3 = np.linalg.norm(points[13] - points[3])
    horiz = np.linalg.norm(points[0] - points[8])
    ear = (v1 + v2 + v3) / (3.0 * horiz) if horiz != 0 else 0
    return ear, points

def calculate_mar(landmarks, frame_w, frame_h, idxs):
    points = [np.array([landmarks[i].x * frame_w, landmarks[i].y * frame_h]) for i in idxs]
    vert = np.linalg.norm(points[0] - points[1])
    horiz = np.linalg.norm(points[2] - points[3])
    return vert / horiz if horiz != 0 else 0, points

def calculate_gaze_ratio(landmarks, frame_w, frame_h):
    return landmarks[LEFT_IRIS_CENTER].x

def generate_excel_from_vp(vp, excel_name, id_kerja, nama_pegawai, jenis_kelamin, usia, status_shift):
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
    
    if vp.freq_kedipan == 0:
        output_rows.append([
            id_kerja, nama_pegawai, jenis_kelamin, usia, "", 0,0,0,0, 0,0,0, 0,0, 0, 
            0, vp.total_bursts, round(vp.perclos_live, 2), 0, 
            0, 0, 0, 0, 0, 0, vp.status, status_shift
        ])
    else:
        for idx, b in enumerate(vp.blinks):
            output_rows.append([
                id_kerja if idx == 0 else "", nama_pegawai if idx == 0 else "", 
                jenis_kelamin if idx == 0 else "", usia if idx == 0 else "", "", 
                round(b['t1'], 3), round(b['t2'], 3), round(b['t3'], 3), round(b['t4'], 3),
                round(b['closed'], 3), round(b['closing'], 3), round(b['reopen'], 3), 
                round(b['duration'], 3), round(b['interval'], 3), b['ms_flag'], 
                round(vp.mcd_live, 3) if idx == 0 else "", vp.total_bursts if idx == 0 else "",
                round(vp.perclos_live, 2) if idx == 0 else "", 0 if idx == 0 else "",
                0 if idx == 0 else "", 0 if idx == 0 else "", 0 if idx == 0 else "",
                0 if idx == 0 else "", 0 if idx == 0 else "", 0 if idx == 0 else "",
                vp.status if idx == 0 else "", status_shift if idx == 0 else ""
            ])
    df = pd.DataFrame(output_rows)
    df.to_excel(excel_name, index=False, header=False)
    return excel_name

# ==========================================
# 3. CLASS WEBRTC (Processor Video Cloud)
# ==========================================
class FFDWebRTCProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.state = 0
        self.t1 = 0.0; self.t2 = 0.0; self.t3 = 0.0; self.t4 = 0.0
        self.blinks = []
        self.p80_frames = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.total_bursts = 0
        
        # Variabel Logika CROP (Jeda Analisis)
        self.crop_start = 0.0
        self.crop_end = 0.0
        self.max_duration = 180.0
        
        self.perclos_live = 0.0
        self.mcd_live = 0.0
        self.freq_kedipan = 0
        self.avg_blink_dur = 0.0
        self.total_ms_live = 0
        self.status = "1 - FIT (Aman)"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width, _ = img.shape
        bgr_out = img.copy()
        
        elapsed_total = time.time() - self.start_time

        # --- LOGIKA FITUR CROP (JEDA) LIVE ---
        if elapsed_total < self.crop_start:
            # Masa Tunggu (Crop Awal)
            cv2.putText(bgr_out, f"MEMULAI ANALISIS DALAM: {int(self.crop_start - elapsed_total)}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return av.VideoFrame.from_ndarray(bgr_out, format="bgr24")
            
        if elapsed_total > (self.max_duration - self.crop_end):
            # Masa Berhenti Dini (Crop Akhir)
            cv2.putText(bgr_out, "ANALISIS SELESAI (Crop Akhir Aktif)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return av.VideoFrame.from_ndarray(bgr_out, format="bgr24")
        # -------------------------------------

        self.frame_count += 1
        time_sec = elapsed_total - self.crop_start # Hitung murni waktu setelah crop
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            mp_drawing.draw_landmarks(
                image=bgr_out,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            l_ear, _ = calculate_ear(landmarks, width, height, LEFT_EYE)
            r_ear, _ = calculate_ear(landmarks, width, height, RIGHT_EYE)
            ear_avg = (l_ear + r_ear) / 2.0

            if ear_avg <= EAR_P80: self.p80_frames += 1

            if self.state == 0 and ear_avg < EAR_HIGH: 
                self.t1, self.state = time_sec, 1
            elif self.state == 1:
                if ear_avg <= EAR_LOW: self.t2, self.state = time_sec, 2
                elif ear_avg >= EAR_HIGH: self.state = 0
            elif self.state == 2:
                if ear_avg > EAR_LOW: self.t3, self.state = time_sec, 3
            elif self.state == 3:
                if ear_avg >= EAR_HIGH:
                    self.t4 = time_sec
                    interval = (self.t1 - self.blinks[-1]['t4']) if len(self.blinks) > 0 else 0
                    if 0 < interval < 1.0: self.total_bursts += 1
                    self.blinks.append({
                        "t1": self.t1, "t2": self.t2, "t3": self.t3, "t4": self.t4, 
                        "closed": self.t3 - self.t2, "closing": self.t2 - self.t1, 
                        "reopen": self.t4 - self.t3, "duration": self.t4 - self.t1, 
                        "interval": interval, "ms_flag": 1 if (self.t3 - self.t2) >= MICROSLEEP_THRESH else 0
                    })
                    self.state = 0
                elif ear_avg <= EAR_LOW: self.state = 2

        frames_analyzed = self.frame_count if self.frame_count > 0 else 1
        self.perclos_live = (self.p80_frames / frames_analyzed) * 100
        self.freq_kedipan = len(self.blinks)
        self.mcd_live = sum(b['closed'] for b in self.blinks) / self.freq_kedipan if self.freq_kedipan > 0 else 0
        self.avg_blink_dur = sum(b['duration'] for b in self.blinks) / self.freq_kedipan if self.freq_kedipan > 0 else 0
        self.total_ms_live = sum(b['ms_flag'] for b in self.blinks)

        self.status = "1 - FIT (Aman)"
        if self.perclos_live >= 12 or self.total_ms_live > 0: self.status = "4 - BAHAYA (KRITIS)"
        elif self.mcd_live >= 0.4 or self.perclos_live >= 10: self.status = "3 - LELAH (Risiko Tinggi)"
        elif self.mcd_live >= 0.25 or self.perclos_live >= 5: self.status = "2 - KURANG FIT"

        cv2.rectangle(bgr_out, (10, 10), (320, 240), (0, 0, 0), -1) 
        cv2.addWeighted(bgr_out, 0.6, img, 0.4, 0, bgr_out) 
        
        warna_teks = (0, 255, 0) if "FIT" in self.status else (0, 0, 255) 
        cv2.putText(bgr_out, f"Status: {self.status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna_teks, 2)
        cv2.putText(bgr_out, f"PERCLOS: {self.perclos_live:.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bgr_out, f"MCD: {self.mcd_live:.3f}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bgr_out, f"Kedipan: {self.freq_kedipan}x", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bgr_out, f"Blink Dur: {self.avg_blink_dur:.3f}s", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        warna_ms = (0, 0, 255) if self.total_ms_live > 0 else (255, 255, 255)
        cv2.putText(bgr_out, f"Microsleep: {self.total_ms_live}x", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_ms, 1)

        if self.total_ms_live > 0:
            cv2.putText(bgr_out, f"!!! MICROSLEEP DETECTED !!!", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(bgr_out, format="bgr24")

# ==========================================
# 4. FUNGSI UPLOAD VIDEO (Optimasi RAM & Jaringan)
# ==========================================
def run_upload_analysis(video_path, id_kerja="-", nama_pegawai="Unknown", jenis_kelamin="-", usia=0, status_shift="Tidak Diketahui", crop_start=0.0, crop_end=0.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # OPTIMASI: Turunkan resolusi video ke 480p di awal untuk mencegah RAM meledak
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = 640
    height = int(orig_height * (640.0 / orig_width)) if orig_width > 0 else 480
    
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_duration = total_frames / fps if fps > 0 else 0
    absolute_stop_time = total_duration - crop_end

    video_out_name = f"Rekaman_{nama_pegawai}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_out_name, fourcc, fps, (width, height))

    state = 0; t1 = t2 = t3 = t4 = 0.0
    blinks = []; p80_frames = 0
    frame_count = 0
    total_bursts = total_yawns = 0
    is_yawning = False; yawn_start_time = 0.0

    thumbnail_saved = False
    temp_img_path = f"web_thumbnail_{nama_pegawai}.jpg" 
    face_mesh_up = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    if crop_start > 0.0:
        cap.set(cv2.CAP_PROP_POS_MSEC, crop_start * 1000)
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # MENCEGAH HEALTH CHECK TIMEOUT SERVER STREAMLIT
        time.sleep(0.001)
        
        frame_count += 1
        time_sec = frame_count / fps
        if crop_end > 0.0 and time_sec >= absolute_stop_time: break

        # Resize untuk meringankan CPU
        frame = cv2.resize(frame, (width, height))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh_up.process(rgb_frame)

        if results.multi_face_landmarks:
            if not thumbnail_saved and frame_count > 15: 
                cv2.imwrite(temp_img_path, cv2.resize(frame, (150, 100)))
                thumbnail_saved = True

            landmarks = results.multi_face_landmarks[0].landmark
            mp_drawing.draw_landmarks(rgb_frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION, None, mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mar_val, _ = calculate_mar(landmarks, width, height, MOUTH)
            if mar_val > MAR_THRESH:
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
                        "t1": t1, "t2": t2, "t3": t3, "t4": t4, "closed": t3 - t2, "closing": t2 - t1, 
                        "reopen": t4 - t3, "duration": t4 - t1, "interval": interval, 
                        "ms_flag": 1 if (t3 - t2) >= MICROSLEEP_THRESH else 0
                    })
                    state = 0
                elif ear_avg <= EAR_LOW: state = 2

        frames_analyzed = frame_count - int(crop_start * fps)
        if frames_analyzed <= 0: frames_analyzed = 1 
        
        perclos_live = (p80_frames / frames_analyzed) * 100
        freq_kedipan = len(blinks)
        mcd_live = sum(b['closed'] for b in blinks) / freq_kedipan if freq_kedipan > 0 else 0
        avg_blink_dur = sum(b['duration'] for b in blinks) / freq_kedipan if freq_kedipan > 0 else 0
        total_ms_live = sum(b['ms_flag'] for b in blinks)
        
        status = "1 - FIT (Aman)"
        if perclos_live >= 12 or total_ms_live > 0: status = "4 - BAHAYA (KRITIS)"
        elif mcd_live >= 0.4 or perclos_live >= 10: status = "3 - LELAH (Risiko Tinggi)"
        elif mcd_live >= 0.25 or perclos_live >= 5: status = "2 - KURANG FIT"

        bgr_out = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR) 
        cv2.rectangle(bgr_out, (10, 10), (320, 240), (0, 0, 0), -1) 
        cv2.addWeighted(bgr_out, 0.6, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR), 0.4, 0, bgr_out) 
        
        warna_teks = (0, 255, 0) if "FIT" in status else (0, 0, 255) 
        cv2.putText(bgr_out, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna_teks, 2)
        cv2.putText(bgr_out, f"PERCLOS: {perclos_live:.1f}%", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(bgr_out, f"MCD: {mcd_live:.3f}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out_video.write(bgr_out) 
        
        # OPTIMASI UI: Mengirimkan (yield) frame ke UI secara sepotong-sepotong (tiap 5 frame)
        if frame_count % 5 == 0:
            yield rgb_frame, status, perclos_live, mcd_live, freq_kedipan, avg_blink_dur, total_ms_live

    cap.release()
    out_video.release() 

    if not thumbnail_saved:
        cv2.imwrite(temp_img_path, np.zeros((100, 150, 3), dtype=np.uint8))

    excel_name = f"Laporan_{nama_pegawai}.xlsx"
    output_rows = []
    output_rows.append(["ID Kerja", "Nama", "Jenis Kelamin", "Usia", "Screenshot", "Waktu Mulai Tutup", "Waktu Tertutup", "Waktu Mulai Buka", "Waktu Buka Penuh", "Closed Phase", "Closing Phase", "Reopening Phase", "Blink Duration", "Blink Interval", "Microsleep Event", "MCD (Kesimpulan)", "Blink Burst", "PERCLOS P80", "Total Yawn", "STATUS KLASIFIKASI", "Status Shift"])
    output_rows.append(["[-]", "[-]", "[-]", "[Tahun]", "[-]", "[s]", "[s]", "[s]", "[s]", "[s]", "[s]", "[s]", "[s]", "[s]", "[0/1]", "[s]", "[Kali]", "[%]", "[Kali]", "[Kesimpulan Sistem]", "[-]"])
    
    if freq_kedipan == 0:
        output_rows.append([id_kerja, nama_pegawai, jenis_kelamin, usia, "", 0,0,0,0, 0,0,0, 0,0, 0, 0, total_bursts, round(perclos_live, 2), total_yawns, status, status_shift])
    else:
        for idx, b in enumerate(blinks):
            output_rows.append([
                id_kerja if idx == 0 else "", nama_pegawai if idx == 0 else "", jenis_kelamin if idx == 0 else "", usia if idx == 0 else "", "", 
                round(b['t1'], 3), round(b['t2'], 3), round(b['t3'], 3), round(b['t4'], 3), round(b['closed'], 3), round(b['closing'], 3), round(b['reopen'], 3), 
                round(b['duration'], 3), round(b['interval'], 3), b['ms_flag'], round(mcd_live, 3) if idx == 0 else "", total_bursts if idx == 0 else "", round(perclos_live, 2) if idx == 0 else "", total_yawns if idx == 0 else "", status if idx == 0 else "", status_shift if idx == 0 else ""
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
# 5. DASHBOARD UTAMA (STREAMLIT ANTARMUKA)
# ==========================================
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(layout="wide", page_title="Dashboard K3 Tambang")

if "halaman" not in st.session_state:
    st.session_state.halaman = "beranda"

def pindah_halaman(nama_halaman):
    st.session_state.halaman = nama_halaman

if st.session_state.halaman == "beranda":
    st.title("🚜 Sistem Pemantauan FFD (Fitness-for-Duty)")
    st.markdown("### Selamat Datang di Dashboard Evaluasi Kelelahan Operator")
    col_menu1, col_menu2 = st.columns(2)
    
    with col_menu1:
        st.info("👁️ **Analisis FFD (MediaPipe)**\n\nPengujian tingkat kewaspadaan berbasis deteksi biometrik wajah.")
        if st.button("Masuk ke Modul Analisis FFD", type="primary", width="stretch"):
            pindah_halaman("analisis")
            st.rerun()
            
    with col_menu2:
        st.warning("⏱️ **PVT Task (Vigilance)**\n\nPengujian waktu reaksi (Reaction Time) kognitif.")
        st.link_button("Buka Modul PVT Task", "https://pvt-v3.vercel.app/auth", width="stretch")

elif st.session_state.halaman == "analisis":
    if st.button("⬅️ Kembali ke Menu Utama"):
        pindah_halaman("beranda")
        st.rerun()
        
    st.title("🖥️ Modul Analisis FFD Operator")
    tab1, tab2 = st.tabs(["📂 Upload Video (Ekspor Data)", "🌐 Live Kamera (WebRTC)"])

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
        c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
        val_start = c1.number_input("Buang/Potong bagian AWAL:", min_value=0.0, value=0.0, step=1.0)
        unit_start = c2.selectbox("Satuan Awal", ["Detik", "Menit"], key="unit_start")
        val_end = c3.number_input("Buang/Potong bagian AKHIR:", min_value=0.0, value=0.0, step=1.0)
        unit_end = c4.selectbox("Satuan Akhir", ["Detik", "Menit"], key="unit_end")
        crop_start_sec = val_start if unit_start == "Detik" else val_start * 60.0
        crop_end_sec = val_end if unit_end == "Detik" else val_end * 60.0

        if uploaded_file is not None:
            if st.button("▶ Mulai Analisis Video & Buat Laporan", type="primary", width="stretch"):
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
                excel_result = ""; video_result = ""
                
                for frame, status, perclos, mcd, blinks, blink_dur, total_ms in run_upload_analysis(tfile.name, id_kerja_up, nama_upload, jenis_kelamin_up, usia_up, status_upload, crop_start_sec, crop_end_sec):
                    if isinstance(frame, str) and frame == "DONE": 
                        excel_result = status; video_result = perclos; break
                        
                    # Menggambar di Web hanya terjadi sebagian kali (Lebih Cepat!)
                    vid_ph.image(frame, channels="RGB", width="stretch")
                    if "BAHAYA" in status: status_ui.error(f"🚨 **{status}**")
                    elif "LELAH" in status: status_ui.warning(f"⚠️ **{status}**")
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
                
                # PEMBERSIHAN MEMORI
                os.remove(tfile.name)
                if os.path.exists(excel_result): os.remove(excel_result)
                if os.path.exists(video_result): os.remove(video_result)

    # ------------------------------------------
    # TAB 2: LIVE KAMERA WEBRTC
    # ------------------------------------------
    with tab2:
        st.markdown("### Uji Coba Langsung (3 Menit) via WebRTC")
        
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
                                teks_bersih = re.sub(r'^[:;|=]\s*', '', baris[len(kata_kata[0]):].strip()).strip()
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

        st.markdown("#### ✂️ Pengaturan Jeda (Crop) Kamera Live")
        c1_L, c2_L, c3_L, c4_L = st.columns([2, 1, 2, 1])
        val_start_L = c1_L.number_input("Jeda Awal (Jangan hitung di awal):", min_value=0.0, value=0.0, step=1.0, key="val_start_L")
        unit_start_L = c2_L.selectbox("Satuan Awal", ["Detik", "Menit"], key="unit_start_L")
        val_end_L = c3_L.number_input("Jeda Akhir (Berhenti lebih cepat):", min_value=0.0, value=0.0, step=1.0, key="val_end_L")
        unit_end_L = c4_L.selectbox("Satuan Akhir", ["Detik", "Menit"], key="unit_end_L")
        
        crop_start_sec_live = val_start_L if unit_start_L == "Detik" else val_start_L * 60.0
        crop_end_sec_live = val_end_L if unit_end_L == "Detik" else val_end_L * 60.0

        st.markdown("---")
        st.info("💡 **Langkah:** Pastikan data Anda terisi. Klik tombol **'START'** di bawah ini untuk mengizinkan kamera dan memulai waktu tes 3 Menit.")

        if "live_test_completed" not in st.session_state:
            st.session_state.live_test_completed = False
            st.session_state.live_zip_path = ""

        yt_ph = st.empty()
        p_text = st.empty()
        p_bar = st.empty()
        
        col_vid2, col_stat2 = st.columns([2, 1])
        with col_vid2:
            ctx = webrtc_streamer(
                key="ffd-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=FFDWebRTCProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
        with col_stat2:
            st.subheader("Monitoring")
            status_ui2 = st.empty()
            st.markdown("---")
            m7, m8, m9 = st.columns(3)
            m10, m11, m12 = st.columns(3)
            perclos_ui2 = m7.empty(); mcd_ui2 = m8.empty(); blink_ui2 = m9.empty()
            dur_ui2 = m10.empty(); ms_ui2 = m11.empty()

        if ctx.state.playing and nama_live != "":
            # Transfer konfigurasi crop ke dalam Processor Kamera
            if ctx.video_processor:
                ctx.video_processor.crop_start = crop_start_sec_live
                ctx.video_processor.crop_end = crop_end_sec_live
                
            with yt_ph:
                st.markdown(
                    """<iframe width="100%" height="400" src="https://www.youtube.com/embed/Se5NjX-cM5I?si=__OUHuj-V2w_wi4J&autoplay=1" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>""",
                    unsafe_allow_html=True
                )
            p_bar_elem = p_bar.progress(0)
            
            if "webrtc_start_time" not in st.session_state or st.session_state.webrtc_start_time == 0:
                st.session_state.webrtc_start_time = time.time()
                st.session_state.live_test_completed = False
                
            while ctx.state.playing:
                elapsed = time.time() - st.session_state.webrtc_start_time
                if elapsed >= 180.0:
                    if not st.session_state.live_test_completed:
                        st.session_state.live_test_completed = True
                        excel_file = generate_excel_from_vp(ctx.video_processor, f"Laporan_Live_{nama_live}.xlsx", id_kerja_live, nama_live, jenis_kelamin_live, usia_live, status_live)
                        zip_name = f"Bukti_FFD_{nama_live}.zip"
                        with zipfile.ZipFile(zip_name, 'w') as zipf: zipf.write(excel_file)
                        st.session_state.live_zip_path = zip_name
                    break
                    
                p_bar_elem.progress(min(elapsed / 180.0, 1.0))
                p_text.markdown(f"**Progress: {int(elapsed)} / 180 Detik**")
                
                vp = ctx.video_processor
                if vp:
                    status = vp.status
                    if "BAHAYA" in status: status_ui2.error(f"🚨 **{status}**")
                    elif "LELAH" in status: status_ui2.warning(f"⚠️ **{status}**")
                    elif "KURANG FIT" in status: status_ui2.info(f"👀 **{status}**")
                    else: status_ui2.success(f"✅ **{status}**")
                        
                    perclos_ui2.metric("PERCLOS", f"{vp.perclos_live:.1f}%")
                    mcd_ui2.metric("MCD", f"{vp.mcd_live:.3f}s")
                    blink_ui2.metric("Kedipan", f"{vp.freq_kedipan}x")
                    dur_ui2.metric("Durasi", f"{vp.avg_blink_dur:.3f}s")
                    if vp.total_ms_live > 0: ms_ui2.error(f"⚠️ {vp.total_ms_live}x")
                    else: ms_ui2.metric("Microsleep", f"{vp.total_ms_live}x")
                    
                time.sleep(0.5)

        if st.session_state.get("live_test_completed"):
            p_bar.progress(1.0)
            yt_ph.empty() 
            st.success("✅ Sesi 3 Menit Selesai! Bukti Laporan (Excel) telah dibuat. Silakan klik STOP kamera.")
            
        if st.session_state.get("live_test_completed") and st.session_state.get("live_zip_path"):
            with open(st.session_state.live_zip_path, "rb") as f:
                st.download_button("📦 Download Laporan (ZIP)", data=f, file_name=st.session_state.live_zip_path, mime="application/zip")

        if not ctx.state.playing:
            st.session_state.webrtc_start_time = 0
