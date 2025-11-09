# ============================================================
# 🎧 STREAMLIT APP — Prediksi Speaker Buka/Tutup (Nadia & Vanisa Only)
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

# ============================================================
# Konfigurasi Streamlit
# ============================================================
st.set_page_config(
    page_title="Prediksi Suara Buka/Tutup",
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
# Fungsi Prediksi (Whitelist: hanya Nadia & Vanisa)
# ============================================================
def predict_audio(file_path, threshold=0.6, force_accept=False):
    features, y, sr = extract_features(file_path)
    features_scaled = scaler.transform(features)

    probs = model.predict_proba(features_scaled)[0]
    class_labels = model.classes_

    # ✅ Daftar speaker yang diizinkan
    allowed_speakers = ["nadia", "vanisa"]

    # Agregasi probabilitas per speaker
    speaker_probs = {}
    for label, prob in zip(class_labels, probs):
        spk = label.split("_")[0].strip().lower()
        speaker_probs.setdefault(spk, []).append(prob)

    speaker_avg = {spk: np.mean(vals) for spk, vals in speaker_probs.items()}
    speaker_sorted = sorted(speaker_avg.items(), key=lambda x: x[1], reverse=True)

    if not speaker_sorted:
        return "Unknown", "Tidak diketahui", 0.0, probs, features_scaled, y, sr, {}

    top_speaker, top_speaker_prob = speaker_sorted[0]

    # Ambil semua label milik speaker tsb
    speaker_related = [
        (lbl, pr)
        for lbl, pr in zip(class_labels, probs)
        if lbl.lower().startswith(top_speaker)
    ]

    if not speaker_related:
        best_label, best_prob = "unknown_tidakdiketahui", 0.0
        speaker_name, status = "Unknown", "Tidak diketahui"
    else:
        best_label, best_prob = max(speaker_related, key=lambda x: x[1])
        speaker_name, status = best_label.split("_", 1)
        status = status.capitalize()

    # 🔒 Hanya terima speaker yang diizinkan
    if speaker_name.lower() not in allowed_speakers:
        speaker_name = "Unknown"
        status = "Tidak diketahui"
        top_speaker_prob = 0.0

    # 🔻 Threshold check
    if top_speaker_prob < threshold and not force_accept:
        speaker_name = "Unknown"
        status = "Tidak diketahui"

    debug = {
        "speaker_avg": speaker_avg,
        "speaker_sorted": speaker_sorted,
        "speaker_related": speaker_related,
        "best_label": best_label,
        "best_prob": best_prob
    }

    return (
        speaker_name.capitalize(),
        status,
        top_speaker_prob,
        probs,
        features_scaled,
        y,
        sr,
        debug
    )

# ============================================================
# Tampilan UI Streamlit
# ============================================================
st.title("🎧 Prediksi Suara Buka/Tutup (Nadia & Vanisa Only)")
st.write("Upload file audio `.wav` untuk mendeteksi siapa speakernya dan status suaranya (Buka/Tutup).")

uploaded_file = st.file_uploader("🎵 Pilih file audio", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_path, format="audio/wav")

    with st.spinner("⏳ Sedang memproses audio..."):
        speaker, status, prob, probs, features_scaled, y, sr, debug = predict_audio(temp_path)

    st.subheader("🎯 Hasil Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Speaker", speaker)
    with col2:
        st.metric("Status", status)
    st.metric("Confidence", f"{prob*100:.2f}%")

    # Probabilitas per kelas
    prob_df = pd.DataFrame({
        "Kelas": model.classes_,
        "Probabilitas (%)": [round(float(p)*100, 2) for p in probs]
    }).sort_values("Probabilitas (%)", ascending=False)

    st.write("### 🔍 Distribusi Probabilitas")
    st.dataframe(prob_df, use_container_width=True)

    # Waveform
    st.subheader("📈 Waveform Audio")
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform Audio")
    ax.set_xlabel("Waktu (detik)")
    ax.set_ylabel("Amplitudo")
    st.pyplot(fig)

    # Mel Spectrogram
    st.subheader("🎛️ Mel Spectrogram")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("Mel Spectrogram")
    st.pyplot(fig)

    # Barplot Probabilitas
    st.subheader("📊 Distribusi Probabilitas Kelas")
    plt.figure(figsize=(6, 4))
    sns.barplot(x="Kelas", y="Probabilitas (%)", data=prob_df)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    st.pyplot(plt)

    with st.expander("🧠 Debug Info"):
        st.write(debug)

    os.remove(temp_path)
else:
    st.info("📂 Silakan upload file audio untuk memulai prediksi.")
