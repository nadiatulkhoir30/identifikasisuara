# ==================================================
# 🎧 STREAMLIT APP — Prediksi Suara (Nadia & Vanisa)
# ==================================================
import os
import numpy as np
import pandas as pd
import librosa
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import librosa.display

# ==================================================
# 🔹 Konfigurasi Streamlit
# ==================================================
st.set_page_config(page_title="Prediksi Suara Buka/Tutup", page_icon="🎵", layout="centered")

st.title("🎧 Prediksi Suara Buka/Tutup")
st.markdown("Hanya menerima suara dari **Nadia** dan **Vanisa**. Speaker lain otomatis jadi *Unknown*.")

# ==================================================
# 🔹 Load model dan scaler
# ==================================================
@st.cache_resource
def load_model_scaler():
    model = joblib.load("best_audio_model.pkl")
    scaler = joblib.load("scaler_audio.pkl")
    return model, scaler

model, scaler = load_model_scaler()

# ==================================================
# 🔹 Fungsi ekstraksi fitur
# ==================================================
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

# ==================================================
# 🔹 Fungsi Prediksi (pakai threshold user)
# ==================================================
def predict_speaker(file_path, threshold=0.7):
    features, y, sr = extract_features(file_path)

    # cek dimensi fitur
    if features.shape[1] != scaler.n_features_in_:
        return "Error", "Jumlah fitur tidak cocok", 0, None, None, y, sr

    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    max_prob = np.max(probs)
    pred_label = model.classes_[np.argmax(probs)]

    # parsing label
    parts = pred_label.lower().split("_")
    speaker = parts[0] if len(parts) > 0 else "unknown"
    status = parts[1].capitalize() if len(parts) > 1 else "-"

    # jika confidence < threshold → unknown
    if max_prob < threshold:
        speaker = "Unknown"
        status = "Tidak diketahui"

    return speaker.capitalize(), status, max_prob, probs, model.classes_, y, sr

# ==================================================
# 🔹 UI Upload & Prediksi
# ==================================================
threshold = st.sidebar.slider("⚙️ Threshold Confidence", 0.3, 0.9, 0.7, 0.05)

uploaded_file = st.file_uploader("🎵 Upload file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_path, format="audio/wav")

    with st.spinner("⏳ Menganalisis audio..."):
        speaker, status, prob, probs, labels, y, sr = predict_speaker(temp_path, threshold)

    st.markdown("---")
    st.subheader("🎯 Hasil Prediksi")

    col1, col2 = st.columns(2)
    col1.metric("Speaker", speaker)
    col2.metric("Status Suara", status)
    st.metric("Confidence (%)", f"{prob*100:.2f}%")

    # probabilitas tiap kelas
    if probs is not None:
        prob_df = pd.DataFrame({
            "Kelas": labels,
            "Probabilitas (%)": [round(float(p)*100, 2) for p in probs]
        }).sort_values("Probabilitas (%)", ascending=False)

        st.markdown("#### 📊 Probabilitas Tiap Kelas")
        st.dataframe(prob_df, use_container_width=True)

        # waveform
        st.subheader("📈 Waveform Audio")
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform Audio", fontsize=12)
        st.pyplot(fig)

        # mel spectrogram
        st.subheader("🎛️ Mel Spectrogram")
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("Mel Spectrogram", fontsize=12)
        st.pyplot(fig)

    os.remove(temp_path)
else:
    st.info("📂 Silakan upload file audio untuk memulai prediksi.")
