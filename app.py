# ============================================================
# 🎧 STREAMLIT APP — Prediksi Suara Buka/Tutup (Nadia & Vanisa Only)
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
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av

# ============================================================
# Konfigurasi Streamlit
# ============================================================
st.set_page_config(
    page_title="Prediksi Suara Buka/Tutup (Nadia & Vanisa Only)",
    page_icon="🎵",
    layout="centered",
)

# ============================================================
# Load model & scaler
# ============================================================
@st.cache_resource
def load_model_scaler():
    model = joblib.load("best_audio_model.pkl")
    scaler = joblib.load("scaler_audio.pkl")
    return model, scaler

model, scaler = load_model_scaler()

# ============================================================
# Fungsi ekstraksi fitur
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
        spectral_contrast(y, sr),
    ]
    mfccs = mfcc_features(y, sr)
    features.extend(mfccs.tolist())
    return np.array(features).reshape(1, -1), y, sr

def extract_features_from_array(y, sr):
    features = [
        zero_crossing_rate(y),
        rms(y),
        spectral_centroid(y, sr),
        spectral_bandwidth(y, sr),
        spectral_contrast(y, sr),
    ]
    mfccs = mfcc_features(y, sr)
    features.extend(mfccs.tolist())
    return np.array(features).reshape(1, -1)

# ============================================================
# Prediksi
# ============================================================
def predict_audio_file(file_path, threshold=0.6):
    features, y, sr = extract_features(file_path)
    features_scaled = scaler.transform(features)
    return predict_features(features_scaled, y, sr, threshold)

def predict_audio_array(y, sr, threshold=0.6):
    features_scaled = scaler.transform(extract_features_from_array(y, sr))
    return predict_features(features_scaled, y, sr, threshold)

def predict_features(features_scaled, y, sr, threshold=0.6):
    probs = model.predict_proba(features_scaled)[0]
    labels = model.classes_
    idx_top = np.argmax(probs)
    pred_label = labels[idx_top]
    pred_prob = probs[idx_top]

    parts = pred_label.lower().split("_")
    speaker = parts[0] if len(parts) > 0 else "unknown"
    status = parts[1].capitalize() if len(parts) > 1 else "-"

    allowed = ["nadia", "vanisa"]
    if speaker not in allowed or pred_prob < threshold:
        speaker = "Unknown"
        status = "Tidak diketahui"

    return speaker.capitalize(), status, pred_prob, probs, labels, y, sr

# ============================================================
# WebRTC Audio Processor
# ============================================================
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.audio_buffer.append(audio)
        return frame

# ============================================================
# UI Streamlit
# ============================================================
st.title("🎧 Prediksi Suara Buka/Tutup")
st.markdown(
    """
    <p style="font-size:16px;">Aplikasi ini hanya menerima suara dari <b>Nadia</b> dan <b>Vanisa</b>.<br>
    Jika suara lain terdeteksi, hasil akan menjadi <b>Unknown</b>.</p>
    """,
    unsafe_allow_html=True,
)

# 🧩 Slider threshold
st.sidebar.header("⚙️ Pengaturan Model")
threshold = st.sidebar.slider(
    "Ambang Confidence (Threshold)",
    min_value=0.3,
    max_value=0.9,
    value=0.6,
    step=0.05,
)

# Pilih metode input
st.markdown("## 🔹 Pilih Metode Input Audio")
input_method = st.radio("Metode:", ("Upload File", "Rekam WebRTC"))

# ============================================================
# Upload File
# ============================================================
if input_method == "Upload File":
    uploaded_file = st.file_uploader("🎵 Upload file audio (.wav)", type=["wav"])
    if uploaded_file is not None:
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(temp_path, format="audio/wav")

        with st.spinner("⏳ Memproses audio..."):
            speaker, status, prob, probs, labels, y, sr = predict_audio_file(temp_path, threshold)

        # Tampilkan hasil sama seperti versi awal
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")
        col1, col2 = st.columns(2)
        col1.metric("Speaker", speaker)
        col2.metric("Status Suara", status)
        st.metric("Confidence (%)", f"{prob*100:.2f}%")

        # Tabel probabilitas
        prob_df = pd.DataFrame({
            "Kelas": labels,
            "Probabilitas (%)": [round(float(p)*100,2) for p in probs]
        }).sort_values("Probabilitas (%)", ascending=False)
        st.markdown("#### 📊 Probabilitas Tiap Kelas")
        st.table(prob_df.style.set_table_styles([
            {"selector": "th", "props": [("text-align", "center"), ("font-weight","bold")]},
            {"selector": "td", "props": [("text-align","center")]}
        ]))

        # Waveform
        st.subheader("📈 Waveform Audio")
        fig, ax = plt.subplots(figsize=(8,3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform Audio", fontsize=12)
        st.pyplot(fig)

        # Mel Spectrogram
        st.subheader("🎛️ Mel Spectrogram")
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(10,4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("Mel Spectrogram", fontsize=12)
        st.pyplot(fig)

        # Distribusi probabilitas
        st.subheader("📉 Distribusi Probabilitas")
        plt.figure(figsize=(6,4))
        sns.barplot(x="Kelas", y="Probabilitas (%)", data=prob_df)
        plt.xticks(rotation=45)
        plt.ylim(0,100)
        plt.tight_layout()
        st.pyplot(plt)

        os.remove(temp_path)

# ============================================================
# Rekam WebRTC
# ============================================================
elif input_method == "Rekam WebRTC":
    webrtc_ctx = webrtc_streamer(
        key="audio",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True
    )

    if webrtc_ctx.state.playing and webrtc_ctx.audio_processor:
        if st.button("🛑 Prediksi dari Rekaman"):
            audio_data = np.concatenate(webrtc_ctx.audio_processor.audio_buffer, axis=0)
            y = audio_data.astype(np.float32)
            sr = 44100

            speaker, status, prob, probs, labels, y_proc, sr_proc = predict_audio_array(y, sr, threshold)

            # Tampilkan hasil persis sama dengan versi upload
            st.markdown("---")
            st.subheader("🎯 Hasil Prediksi")
            col1, col2 = st.columns(2)
            col1.metric("Speaker", speaker)
            col2.metric("Status Suara", status)
            st.metric("Confidence (%)", f"{prob*100:.2f}%")

            # Tabel probabilitas
            prob_df = pd.DataFrame({
                "Kelas": labels,
                "Probabilitas (%)": [round(float(p)*100,2) for p in probs]
            }).sort_values("Probabilitas (%)", ascending=False)
            st.markdown("#### 📊 Probabilitas Tiap Kelas")
            st.table(prob_df.style.set_table_styles([
                {"selector": "th", "props": [("text-align", "center"), ("font-weight","bold")]},
                {"selector": "td", "props": [("text-align","center")]}
            ]))

            # Waveform
            st.subheader("📈 Waveform Audio")
            fig, ax = plt.subplots(figsize=(8,3))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title("Waveform Audio", fontsize=12)
            st.pyplot(fig)

            # Mel Spectrogram
            st.subheader("🎛️ Mel Spectrogram")
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots(figsize=(10,4))
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title("Mel Spectrogram", fontsize=12)
            st.pyplot(fig)

            # Distribusi probabilitas
            st.subheader("📉 Distribusi Probabilitas")
            plt.figure(figsize=(6,4))
            sns.barplot(x="Kelas", y="Probabilitas (%)", data=prob_df)
            plt.xticks(rotation=45)
            plt.ylim(0,100)
            plt.tight_layout()
            st.pyplot(plt)
