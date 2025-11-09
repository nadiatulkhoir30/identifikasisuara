# ==================================================
# 🔹 IMPORT LIBRARY
# ==================================================
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
import os

# ==================================================
# 🔹 LOAD MODEL & SCALER
# ==================================================
model = joblib.load("best_audio_model.pkl")
scaler = joblib.load("scaler_audio.pkl")

# ==================================================
# 🔹 EKSTRAKSI FITUR AUDIO (SAMA PERSIS DENGAN PY)
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
    """Ekstraksi fitur untuk satu file audio"""
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
    return np.array(features).reshape(1, -1)

# ==================================================
# 🔹 PREDIKSI SPEAKER DENGAN THRESHOLD
# ==================================================
def predict_speaker(file_path, threshold=0.7):
    features = extract_features(file_path)

    if features.shape[1] != scaler.n_features_in_:
        return None, None, None, None, f"❌ Jumlah fitur {features.shape[1]} tidak cocok dengan scaler ({scaler.n_features_in_})"

    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    max_prob = np.max(probs)
    pred_label = model.classes_[np.argmax(probs)]

    if max_prob < threshold:
        return "Tidak dikenal", "-", max_prob, probs, None
    else:
        return pred_label, "Dikenali", max_prob, probs, None

# ==================================================
# 🔹 STREAMLIT UI
# ==================================================
st.set_page_config(page_title="🎙️ Speaker Identification", layout="centered")
st.title("🎧 Sistem Identifikasi Suara")

st.markdown("Unggah file `.wav` untuk dikenali oleh sistem.")

# Threshold kontrol manual
threshold = st.sidebar.slider("🔧 Threshold Kepercayaan", 0.1, 0.95, 0.7, 0.05)
st.sidebar.write(f"Threshold saat ini: **{threshold:.2f}**")

# Upload file
uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_path)

    with st.spinner("🔍 Menganalisis suara..."):
        speaker, status, conf, probs, err = predict_speaker(temp_path, threshold)

    if err:
        st.error(err)
    elif speaker == "Tidak dikenal":
        st.warning(f"⚠️ Speaker tidak dikenali (Confidence: {conf:.2f})")
    else:
        st.success(f"✅ Speaker terdeteksi: **{speaker}** (Confidence: {conf:.2f})")

    # Tabel probabilitas semua kelas
    if probs is not None:
        df_probs = pd.DataFrame({
            "Label": model.classes_,
            "Probabilitas": probs
        }).sort_values("Probabilitas", ascending=False)
        st.subheader("📊 Probabilitas Setiap Kelas")
        st.dataframe(df_probs.reset_index(drop=True))

    # Hapus file sementara
    os.remove(temp_path)
