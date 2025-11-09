import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa
import os

# ==================================================
# 🔹 1. LOAD MODEL DAN SCALER
# ==================================================
@st.cache_resource
def load_model_scaler():
    model = joblib.load("best_audio_model.pkl")
    scaler = joblib.load("scaler_audio.pkl")
    return model, scaler

model, scaler = load_model_scaler()

# ==================================================
# 🔹 2. FUNGSI EKSTRAKSI FITUR DARI AUDIO
# ==================================================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Ekstraksi fitur dasar (bisa disesuaikan dengan fitur pelatihanmu)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)

    features = np.hstack([chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr, mfcc])
    return features

# ==================================================
# 🔹 3. UI STREAMLIT
# ==================================================
st.title("🎵 Prediksi Kelas Audio")
st.write("Unggah file audio (format: .wav, .mp3, .m4a) untuk diprediksi menggunakan model yang sudah dilatih.")

uploaded_file = st.file_uploader("Pilih file audio", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(file_path)

    # Ekstraksi fitur
    features = extract_features(file_path)
    X_new = np.array([features])

    # Normalisasi fitur baru (pakai scaler training)
    X_new_scaled = scaler.transform(X_new)

    # Prediksi
    prediction = model.predict(X_new_scaled)
    st.success(f"🎯 Prediksi kelas audio: **{prediction[0]}**")

    # Hapus file sementara
    os.remove(file_path)
