# ==================================================
# 🔹 IMPORT LIBRARY
# ==================================================
import os
import numpy as np
import pandas as pd
import librosa
import joblib
import streamlit as st
import soundfile as sf
import matplotlib.pyplot as plt

# ==================================================
# 🔹 LOAD MODEL DAN SCALER
# ==================================================
model = joblib.load("best_audio_model.pkl")
scaler = joblib.load("scaler_audio.pkl")

# ==================================================
# 🔹 FUNGSI EKSTRAKSI FITUR
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
# 🔹 FUNGSI PREDIKSI DENGAN THRESHOLD
# ==================================================
def predict_speaker(file_path, threshold=0.7):
    features, y, sr = extract_features(file_path)
    
    if features.shape[1] != scaler.n_features_in_:
        st.error(f"Jumlah fitur tidak cocok ({features.shape[1]} != {scaler.n_features_in_})")
        return None, None, None, None, y, sr

    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    max_prob = np.max(probs)
    pred_label = model.classes_[np.argmax(probs)]

    if max_prob < threshold:
        return "Unknown", "Tidak diketahui", probs, model.classes_, y, sr
    else:
        return pred_label, f"{max_prob:.2f}", probs, model.classes_, y, sr

# ==================================================
# 🔹 STREAMLIT UI
# ==================================================
st.set_page_config(page_title="🔊 Voice Recognition", layout="wide")
st.title("🎙️ Voice Recognition - Speaker Identification")

uploaded_file = st.file_uploader("Unggah file audio (.wav)", type=["wav"])
threshold = st.slider("Atur Threshold Kepercayaan", 0.0, 1.0, 0.7, 0.01)

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("🔍 Sedang menganalisis suara..."):
        speaker, confidence, probs, labels, y, sr = predict_speaker(temp_path, threshold)

    if probs is not None:
        # ==================================================
        # 🔹 TAMPILKAN HASIL PREDIKSI
        # ==================================================
        st.subheader("🎯 Hasil Prediksi")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Speaker", speaker)
        with col2:
            if speaker == "Unknown":
                st.metric("Status Suara", "❌ Tidak dikenali")
            else:
                st.metric("Status Suara", "✅ Dikenali")

        # ==================================================
        # 🔹 PROBABILITAS PER KELAS
        # ==================================================
        st.markdown("#### 📊 Probabilitas Tiap Kelas")
        prob_df = pd.DataFrame({
            "Kelas": labels,
            "Probabilitas (%)": (probs * 100).round(2)
        })
        prob_df["Status"] = np.where(
            prob_df["Kelas"] == labels[np.argmax(probs)],
            "🎯 Tertinggi",
            ""
        )
        prob_df = prob_df.sort_values("Probabilitas (%)", ascending=False).reset_index(drop=True)

        # Warna otomatis berdasarkan nilai probabilitas
        def color_confidence(val):
            color = ""
            if val >= 70:
                color = "#22c55e"  # hijau
            elif val >= 40:
                color = "#eab308"  # kuning
            else:
                color = "#ef4444"  # merah
            return f"color: {color}; font-weight: bold;"

        st.dataframe(
            prob_df.style.applymap(color_confidence, subset=["Probabilitas (%)"]),
            use_container_width=True,
            height=240
        )

        # ==================================================
        # 🔹 PROGRESS BAR UNTUK TIAP KELAS
        # ==================================================
        st.markdown("#### 📈 Confidence per Kelas")
        for _, row in prob_df.iterrows():
            st.write(f"**{row['Kelas']}** — {row['Probabilitas (%)']}%")
            st.progress(int(row["Probabilitas (%)"]))

        # ==================================================
        # 🔹 WAVEFORM AUDIO
        # ==================================================
        st.markdown("#### 🎵 Waveform Audio")
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#2563eb")
        ax.set_title("Visualisasi Gelombang Suara")
        ax.set_xlabel("Waktu (detik)")
        ax.set_ylabel("Amplitudo")
        st.pyplot(fig)

        # ==================================================
        # 🔹 PEMUTAR AUDIO
        # ==================================================
        st.audio(temp_path, format="audio/wav")

        # Hapus file sementara
        os.remove(temp_path)
