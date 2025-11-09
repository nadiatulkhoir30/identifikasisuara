# ==================================================
# 🎤 Aplikasi Identifikasi Suara (Speaker Recognition)
# ==================================================
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import tempfile
from pydub import AudioSegment

# ==================================================
# 🔹 Load Model & Scaler
# ==================================================
@st.cache_resource
def load_model_scaler():
    model = joblib.load("best_audio_model.pkl")   # pastikan namanya sesuai
    scaler = joblib.load("scaler_audio.pkl")      # pastikan namanya sesuai
    return model, scaler

model, scaler = load_model_scaler()

# ==================================================
# 🔹 Fungsi Ekstraksi Fitur
# ==================================================
def extract_features(file_path):
    """Ekstraksi fitur utama dari file audio"""
    y, sr = librosa.load(file_path, sr=22050)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    features = np.hstack([zcr, rms, centroid, bandwidth, contrast, mfcc])
    return np.array(features).reshape(1, -1)

# ==================================================
# 🔹 Halaman Streamlit
# ==================================================
st.set_page_config(page_title="🎤 Identifikasi Suara", layout="wide")

st.title("🎶 Identifikasi Suara (Speaker Recognition)")
st.write("Unggah suara untuk mendeteksi apakah berasal dari **Nadia** atau **Vanisa** 🎧")

uploaded_file = st.file_uploader("🎵 Unggah file audio (.wav / .m4a)", type=["wav", "m4a"])

THRESHOLD = 0.65  # ambang minimal kepercayaan

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        if uploaded_file.name.endswith(".m4a"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_m4a:
                tmp_m4a.write(uploaded_file.read())
                audio = AudioSegment.from_file(tmp_m4a.name, format="m4a")
                audio.export(tmp_wav.name, format="wav")
        else:
            tmp_wav.write(uploaded_file.read())

        # Tampilkan audio
        st.audio(tmp_wav.name, format="audio/wav")

        # ==================================================
        # 🔹 Ekstraksi dan Prediksi
        # ==================================================
        with st.spinner("🎧 Sedang memproses audio..."):
            features = extract_features(tmp_wav.name)
            scaled = scaler.transform(features)
            probs = model.predict_proba(scaled)[0]
            max_prob = np.max(probs)
            pred_class = model.classes_[np.argmax(probs)]

        # ==================================================
        # 🔹 Logika Threshold
        # ==================================================
        if max_prob < THRESHOLD:
            speaker = "Unknown"
            status_suara = "Tidak diketahui"
        else:
            if "Nadia" in pred_class:
                speaker = "Nadia"
            elif "Vanisa" in pred_class:
                speaker = "Vanisa"
            else:
                speaker = "Unknown"
            if "Buka" in pred_class:
                status_suara = "Buka"
            elif "Tutup" in pred_class:
                status_suara = "Tutup"
            else:
                status_suara = "Tidak diketahui"

        # ==================================================
        # 🔹 Tampilkan Hasil
        # ==================================================
        st.success("✅ Analisis selesai!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🧑 Speaker", speaker)
        with col2:
            st.metric("🔊 Status Suara", status_suara)
        with col3:
            st.metric("📈 Confidence", f"{max_prob*100:.2f}%")

        # ==================================================
        # 🔹 Visualisasi Spektrogram
        # ==================================================
        st.subheader("🎨 Spektrogram Audio")
        y, sr = librosa.load(tmp_wav.name, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set(title="Mel-frequency Spectrogram")
        st.pyplot(fig)

        # ==================================================
        # 🔹 Debug Info (opsional)
        # ==================================================
        with st.expander("🧩 Debug Info"):
            st.write("Model classes:", model.classes_)
            st.write("Probabilities:", dict(zip(model.classes_, np.round(probs, 2))))

