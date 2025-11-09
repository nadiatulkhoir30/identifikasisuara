# =========================================
# APP STREAMLIT: Prediksi Suara Buka / Tutup Interaktif
# =========================================

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

# ===============================
# Konfigurasi tampilan Streamlit
# ===============================
st.set_page_config(
    page_title="Prediksi Suara Buka/Tutup",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ===============================
# Load model dan scaler
# ===============================
@st.cache_resource
def load_model_scaler():
    model = joblib.load("best_audio_model.pkl")
    scaler = joblib.load("scaler_audio.pkl")
    return model, scaler

model, scaler = load_model_scaler()

# ===============================
# Fungsi ekstraksi fitur
# ===============================
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
    y = librosa.util.normalize(y)
    # Samakan durasi
    target_length = 2 * sr
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]

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

# ===============================
# Fungsi prediksi
# ===============================
def predict_audio(file_path, threshold):
    features, y, sr = extract_features(file_path)
    if features.shape[1] != scaler.n_features_in_:
        st.error(f"Jumlah fitur ({features.shape[1]}) tidak cocok dengan scaler ({scaler.n_features_in_})")
        return None, None, None, None
    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    pred_label = model.classes_[np.argmax(probs)] if np.max(probs) >= threshold else "Tidak dikenal"
    return pred_label, probs, y, sr

# ===============================
# Header aplikasi
# ===============================
st.markdown("""
<div style="text-align:center">
<h1>🎧 Prediksi Suara <span style="color:#1E90FF;">Buka</span> / <span style="color:#FF6347;">Tutup</span></h1>
<p style="font-size:16px;">Upload file audio atau rekam langsung untuk mendeteksi suara Buka / Tutup.</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# Sidebar untuk threshold
# ===============================
threshold = st.sidebar.slider("Threshold Confidence", 0.0, 1.0, 0.7, 0.05)

# ===============================
# Upload audio
# ===============================
uploaded_file = st.file_uploader("🎵 Pilih file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.audio(uploaded_file, format="audio/wav")
    st.info("🎶 Sedang memproses audio...")
    
    pred_label, probs, y, sr = predict_audio(temp_path, threshold)
    
    if pred_label:
        st.subheader("🎯 Hasil Prediksi")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Prediksi", value=pred_label.upper())
        with col2:
            st.metric(label="Confidence Tertinggi (%)", value=f"{max(probs)*100:.2f}%")
        
        # Waveform
        fig, ax = plt.subplots(figsize=(8,2.5))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform Audio")
        st.pyplot(fig)
        
        # Spectrogram
        st.subheader("📊 Spectrogram")
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10,4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        st.pyplot(plt)
        
        # Bar plot probabilitas
        st.subheader("📊 Probabilitas Tiap Kelas")
        prob_df = pd.DataFrame({
            "Kelas": model.classes_,
            "Probabilitas (%)": [round(float(p)*100,2) for p in probs]
        })
        plt.figure(figsize=(6,4))
        sns.barplot(x="Kelas", y="Probabilitas (%)", data=prob_df)
        plt.ylim(0,100)
        plt.title("Probabilitas Prediksi")
        st.pyplot(plt)
        
    os.remove(temp_path)
else:
    st.warning("📂 Silakan upload file audio terlebih dahulu untuk melakukan prediksi.")
