# ============================================================
# 🎧 STREAMLIT APP — Prediksi Suara Buka/Tutup (Nadia & Vanisa)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import soundfile as sf

# ============================================================
# Konfigurasi Streamlit
# ============================================================
st.set_page_config(
    page_title="Prediksi Suara Buka/Tutup",
    page_icon="🎵",
    layout="centered",
)

st.title("🎧 Prediksi Suara Buka/Tutup")
st.markdown(
    "Aplikasi ini mendeteksi siapa yang berbicara dan apakah suaranya **Buka** atau **Tutup**. "
    "Kamu bisa mengatur **threshold kepercayaan** dan mengunci hanya untuk **Nadia & Vanisa**."
)

# ============================================================
# Load Model & Scaler
# ============================================================
@st.cache_resource
def load_model_scaler():
    model = joblib.load("best_audio_model.pkl")
    scaler = joblib.load("scaler_audio.pkl")
    return model, scaler

model, scaler = load_model_scaler()

# ============================================================
# Fungsi Ekstraksi Fitur
# ============================================================
def zero_crossing_rate(y):
    return np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)[0]

def rms(signal):
    return np.sqrt(np.mean(signal**2))

def spectral_centroid(y, sr):
    return np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

def spectral_bandwidth(y, sr):
    return np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

def spectral_contrast(y, sr):
    return np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

def mfcc_features(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    features = [
        zero_crossing_rate(y),
        rms(y),
        spectral_centroid(y, sr),
        spectral_bandwidth(y, sr),
        spectral_contrast(y, sr)
    ]
    mfccs = mfcc_features(y, sr)
    features.extend(mfccs.tolist())
    return np.array(features).reshape(1, -1), y, sr

# ============================================================
# Fungsi Prediksi
# ============================================================
def predict_speaker(file_path, threshold=0.7, lock_speakers=True):
    features, y, sr = extract_features(file_path)

    if features.shape[1] != scaler.n_features_in_:
        return None, None, None, None, y, sr, "❌ Jumlah fitur tidak cocok dengan scaler."

    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    labels = model.classes_

    max_prob = np.max(probs)
    pred_label = labels[np.argmax(probs)]
    speaker_name = pred_label.split("_")[0].lower()

    allowed = ["nadia", "vanisa"]

    # 🔒 Kunci speaker & threshold
    if (lock_speakers and speaker_name not in allowed) or (max_prob < threshold):
        speaker_name = "Unknown"
        status = "Tidak diketahui"
    else:
        status = pred_label.split("_")[1].capitalize() if "_" in pred_label else "-"

    return speaker_name.capitalize(), status, max_prob, probs, labels, y, sr, None

# ============================================================
# UI Streamlit
# ============================================================

col1, col2 = st.columns(2)
threshold = col1.slider("🎚️ Threshold Kepercayaan", 0.0, 1.0, 0.7, 0.01)
lock_speakers = col2.checkbox("🔒 Kunci hanya untuk Nadia & Vanisa", value=True)

st.markdown("### 🎙️ Pilih Input Suara")
mode = st.radio("Pilih metode input:", ["Upload File", "Rekam Langsung"], horizontal=True)

audio_data = None

# ============ Mode Upload File ============
if mode == "Upload File":
    uploaded_file = st.file_uploader("📂 Upload file audio (.wav)", type=["wav"])
    if uploaded_file is not None:
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        audio_data = temp_path
        st.audio(temp_path, format="audio/wav")

# ============ Mode Rekam Langsung ============
elif mode == "Rekam Langsung":
    st.markdown(
        """
        🎤 Klik tombol di bawah untuk merekam suara (maksimal ±10 detik).
        Setelah selesai, simpan dan kirim hasilnya sebagai file `.wav`.
        """
    )
    # Versi simple pakai elemen audio HTML
    audio_bytes = st.audio_input("🎙️ Rekam suara kamu di sini")
    if audio_bytes:
        temp_path = "recorded_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(audio_bytes.read())
        audio_data = temp_path
        st.success("✅ Suara berhasil direkam.")
        st.audio(temp_path, format="audio/wav")

# ============================================================
# Proses Prediksi
# ============================================================
if audio_data:
    with st.spinner("⏳ Menganalisis audio..."):
        speaker, status, prob, probs, labels, y, sr, err = predict_speaker(
            audio_data, threshold=threshold, lock_speakers=lock_speakers
        )

    if err:
        st.error(err)
    else:
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")
        col1, col2 = st.columns(2)
        col1.metric("Speaker", speaker)
        col2.metric("Status", status)
        st.metric("Confidence", f"{prob*100:.2f}%")

        # 📊 Probabilitas
        prob_df = pd.DataFrame({
            "Kelas": labels,
            "Probabilitas (%)": [round(float(p)*100, 2) for p in probs]
        }).sort_values("Probabilitas (%)", ascending=False)

        st.markdown("#### 📊 Probabilitas Tiap Kelas")
        st.table(prob_df)

        # 🎵 Visualisasi Audio
        st.subheader("📈 Waveform Audio")
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform Audio")
        st.pyplot(fig)

        st.subheader("🎛️ Mel Spectrogram")
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("Mel Spectrogram")
        st.pyplot(fig)

        # 📉 Barplot
        st.subheader("📉 Distribusi Probabilitas")
        plt.figure(figsize=(6, 4))
        sns.barplot(x="Kelas", y="Probabilitas (%)", data=prob_df)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        st.pyplot(plt)

    # Hapus file sementara
    if os.path.exists(audio_data):
        os.remove(audio_data)
else:
    st.info("📂 Silakan upload atau rekam suara terlebih dahulu untuk memulai prediksi.")
