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
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
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
    "Aplikasi ini memprediksi siapa yang berbicara dan apakah suaranya **Buka** atau **Tutup**. "
    "Kamu bisa **upload file audio**, **rekam suara langsung**, atau **gunakan contoh bawaan** 🎙️."
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
def zero_crossing_rate(y): return np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)[0]
def rms(signal): return np.sqrt(np.mean(signal**2))
def spectral_centroid(y, sr): return np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
def spectral_bandwidth(y, sr): return np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
def spectral_contrast(y, sr): return np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
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
# Pilihan Mode Input
# ============================================================
st.markdown("## 🗂️ Pilih Sumber Suara")
mode = st.radio("Pilih metode input:", ["📁 Upload File (.wav)", "🎙️ Rekam Langsung", "🎵 Gunakan File Contoh"])

col1, col2 = st.columns(2)
threshold = col1.slider("🎚️ Threshold Kepercayaan", 0.0, 1.0, 0.7, 0.01)
lock_speakers = col2.checkbox("🔒 Kunci hanya untuk Nadia & Vanisa", value=True)

audio_path = None

# ============================================================
# Mode 1: Upload File
# ============================================================
if mode == "📁 Upload File (.wav)":
    uploaded_file = st.file_uploader("🎵 Upload file audio", type=["wav"])
    if uploaded_file:
        audio_path = "temp_uploaded.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())
        st.audio(audio_path, format="audio/wav")
        st.success("✅ File berhasil diupload!")

# ============================================================
# Mode 2: Rekam Langsung
# ============================================================
elif mode == "🎙️ Rekam Langsung":
    st.info("Klik **Start** untuk mulai merekam suara kamu 🎙️ (minimal 2 detik).")

    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []
        def recv_audio_frame(self, frame: av.AudioFrame):
            audio = frame.to_ndarray()
            self.frames.append(audio)
            return frame
        def get_audio(self):
            if not self.frames:
                return None
            return np.concatenate(self.frames, axis=1)

    webrtc_ctx = webrtc_streamer(
        key="recorder",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        audio_processor_factory=AudioProcessor
    )

    if webrtc_ctx.audio_receiver:
        if st.button("🔴 Stop & Simpan Rekaman"):
            audio = webrtc_ctx.audio_processor.get_audio()
            if audio is not None:
                audio_path = "recorded_audio.wav"
                sf.write(audio_path, audio.T, 16000)
                st.success("✅ Rekaman berhasil disimpan!")
                st.audio(audio_path, format="audio/wav")

# ============================================================
# Mode 3: File Contoh
# ============================================================
elif mode == "🎵 Gunakan File Contoh":
    sample_dir = "clean_audio"
    if os.path.exists(sample_dir):
        speaker_opt = st.selectbox("Pilih contoh suara:", [
            "Nadia_Buka/1.wav",
            "Nadia_Tutup/1.wav",
            "Vanisa_Buka/1.wav",
            "Vanisa_Tutup/1.wav",
        ])
        audio_path = os.path.join(sample_dir, speaker_opt)
        st.audio(audio_path, format="audio/wav")
    else:
        st.error("❌ Folder `clean_audio` tidak ditemukan. Pastikan contoh audio tersedia.")

# ============================================================
# Prediksi
# ============================================================
if audio_path is not None and os.path.exists(audio_path):
    with st.spinner("⏳ Menganalisis audio..."):
        speaker, status, prob, probs, labels, y, sr, err = predict_speaker(
            audio_path, threshold=threshold, lock_speakers=lock_speakers
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

        # Probabilitas
        prob_df = pd.DataFrame({
            "Kelas": labels,
            "Probabilitas (%)": [round(float(p)*100, 2) for p in probs]
        }).sort_values("Probabilitas (%)", ascending=False)

        st.markdown("#### 📊 Probabilitas Tiap Kelas")
        st.dataframe(prob_df, use_container_width=True)

        # Visualisasi
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

        st.subheader("📉 Distribusi Probabilitas")
        plt.figure(figsize=(6, 4))
        sns.barplot(x="Kelas", y="Probabilitas (%)", data=prob_df)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        st.pyplot(plt)

else:
    st.info("📂 Silakan pilih atau rekam file audio terlebih dahulu.")
