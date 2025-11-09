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
from streamlit_mic_recorder import mic_recorder

# ============================================================
# Konfigurasi Streamlit
# ============================================================
st.set_page_config(
    page_title="Prediksi Suara Buka/Tutup",
    page_icon="🎵",
    layout="centered",
)

st.title("🎧 Prediksi Suara Buka/Tutup")
st.markdown("""
Aplikasi ini memprediksi **siapa yang berbicara (Nadia/Vanisa)** dan status suaranya (**Buka/Tutup**).  
Kamu bisa **atur threshold** dan juga **rekam langsung dari microphone** 🎙️
""")

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

    if (lock_speakers and speaker_name not in allowed) or (max_prob < threshold):
        speaker_name = "Unknown"
        status = "Tidak diketahui"
    else:
        status = pred_label.split("_")[1].capitalize() if "_" in pred_label else "-"

    return speaker_name.capitalize(), status, max_prob, probs, labels, y, sr, None

# ============================================================
# UI — Upload / Rekam Audio
# ============================================================
col1, col2 = st.columns(2)
threshold = col1.slider("🎚️ Threshold Kepercayaan", 0.0, 1.0, 0.7, 0.01)
lock_speakers = col2.checkbox("🔒 Kunci hanya untuk Nadia & Vanisa", value=True)

st.markdown("### 🎙️ Input Suara")
rec_option = st.radio("Pilih metode input:", ["Upload file", "Rekam langsung"])

audio_bytes = None

if rec_option == "Upload file":
    uploaded_file = st.file_uploader("📂 Upload file audio (.wav)", type=["wav"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format="audio/wav")

else:
    st.info("Tekan tombol di bawah untuk merekam suara (maks 5 detik).")
    audio_bytes = mic_recorder(start_prompt="🎤 Mulai Rekam", stop_prompt="⏹️ Berhenti Rekam", just_once=True)
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")

# ============================================================
# Prediksi & Visualisasi
# ============================================================
if audio_bytes:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    with st.spinner("⏳ Menganalisis audio..."):
        speaker, status, prob, probs, labels, y, sr, err = predict_speaker(
            temp_path, threshold=threshold, lock_speakers=lock_speakers
        )

    if err:
        st.error(err)
    else:
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")

        # Warna confidence
        if prob < 0.5:
            color = "🔴"
        elif prob < 0.7:
            color = "🟡"
        else:
            color = "🟢"

        col1, col2 = st.columns(2)
        col1.metric("Speaker", speaker)
        col2.metric("Status", status)
        st.metric("Confidence", f"{prob*100:.2f}% {color}")

        # Tabel Probabilitas
        prob_df = pd.DataFrame({
            "Kelas": labels,
            "Probabilitas (%)": [round(float(p)*100, 2) for p in probs]
        }).sort_values("Probabilitas (%)", ascending=False)
        st.markdown("#### 📊 Probabilitas Tiap Kelas")
        st.dataframe(prob_df, use_container_width=True)

        # Waveform
        st.subheader("📈 Waveform Audio")
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform Audio")
        st.pyplot(fig)

        # Spectrogram
        st.subheader("🎛️ Mel Spectrogram")
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("Mel Spectrogram")
        st.pyplot(fig)

        # Probabilitas Barplot
        st.subheader("📉 Distribusi Probabilitas")
        plt.figure(figsize=(6, 4))
        sns.barplot(x="Kelas", y="Probabilitas (%)", data=prob_df)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        st.pyplot(plt)

        # Analisis threshold otomatis
        st.subheader("📈 Analisis Pengaruh Threshold")
        th_values = np.arange(0.3, 0.95, 0.05)
        decisions = ["Dikenal" if prob >= t else "Tidak dikenal" for t in th_values]
        df_thresh = pd.DataFrame({"Threshold": th_values, "Status": decisions})
        st.line_chart(df_thresh.set_index("Threshold"))

    os.remove(temp_path)
else:
    st.info("🎧 Silakan upload atau rekam audio terlebih dahulu.")
