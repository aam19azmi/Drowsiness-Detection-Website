# 🚜 Dashboard FFD (Fitness-for-Duty) Operator Tambang

Aplikasi berbasis Web AI yang dirancang untuk mengevaluasi kesiapan kerja (Fitness-for-Duty) operator alat berat secara real-time. Sistem ini menggunakan teknologi **Computer Vision** dan **MediaPipe Face Mesh** untuk mendeteksi indikator kelelahan (fatigue) secara objektif dan tervalidasi.

---

## 🌟 Fitur Utama

- **Analisis Biometrik Real-Time**: Menghitung parameter klinis kelelahan seperti:
  - **PERCLOS (Percentage of Eye Closure)**: Standar emas deteksi kantuk.
  - **MCD (Mean Closure Duration)**: Rata-rata durasi mata tertutup.
  - **Blink Frequency**: Frekuensi kedipan per menit.
  - **Microsleep Detection**: Peringatan instan jika mata tertutup >0.5 detik.
- **Smart Video Cropping**: Fitur untuk memotong durasi awal dan akhir video secara otomatis untuk efisiensi analisis.
- **HUD (Heads-Up Display) Video**: Rekaman hasil deteksi dilengkapi dengan jaring wajah (*face mesh*) dan parameter biometrik yang menempel pada video.
- **Otomasi Laporan Excel**: Ekspor data mentah per-kedipan beserta screenshot wajah operator ke dalam file .xlsx yang rapi.
- **Integrasi PVT Task**: Tautan langsung ke pengujian *Psychomotor Vigilance Task* untuk validasi kognitif.
- **OCR ID Card Scan**: Input nama otomatis menggunakan pemindaian KTP/ID Card berbasis Tesseract OCR.

---

## 🛠️ Teknologi yang Digunakan

- **Python 3.9+**
- **Streamlit**: Framework antarmuka web.
- **OpenCV**: Pemrosesan video dan gambar.
- **MediaPipe**: Pelacakan titik landmark wajah (Face Mesh).
- **Pandas & Openpyxl**: Manajemen data dan laporan Excel.
- **Tesseract OCR**: Pengenalan teks pada ID Card.

---

## 🚀 Cara Menjalankan Secara Lokal

1. **Clone Repositori**

```bash
git clone [https://github.com/aam19azmi/Drowsiness-Detection-Website.git](https://github.com/aam19azmi/Drowsiness-Detection-Website.git)
cd Drowsiness-Detection-Website
```

2. **Instalasi Library**

**Pastikan Anda sudah menginstal Tesseract OCR di PC Anda, lalu jalankan:**

`pip install -r requirements.txt`

3. Jalankan Aplikasi

`streamlit run app_web4.py`


## ☁️ Deployment ke Streamlit Cloud

Sistem ini siap dihosting secara gratis di  **Streamlit Community Cloud** :

1. Upload file `app_web4.py`, `requirements.txt`, `packages.txt`, dan folder `.streamlit` ke GitHub.
2. Masuk ke [share.streamlit.io](https://share.streamlit.io/).
3. Hubungkan repositori GitHub Anda.
4. Set *Main file path* ke `app_web4.py`.
5. Klik  **Deploy** .

## 📁 Struktur Proyek

**Plaintext**

```
├── .streamlit/
│   └── config.toml         # Konfigurasi server (Max Upload Size)
├── app_web.py             # File aplikasi utama (Logika AI + UI)
├── requirements.txt        # Daftar library Python
├── packages.txt            # Daftar dependensi sistem Linux (Tesseract)
└── README.md               # Dokumentasi proyek
```

---

## 📝 Catatan Penting

* **Pencahayaan** : Pastikan wajah operator mendapatkan cahaya yang cukup untuk akurasi *Face Mesh* yang maksimal.
* **Privasi** : Hasil rekaman video dan laporan Excel hanya disimpan sementara di sesi server dan akan terhapus saat sesi berakhir jika tidak diunduh.
