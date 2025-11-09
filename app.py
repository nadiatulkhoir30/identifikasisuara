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
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile

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

# ============================================================
# Prediksi (Locked Speaker: Nadia & Vanisa)
# ============================================================
def predict_audio(file_path, threshold=0.6):
    features, y, sr = extract_features(file_path)
    features_scaled = scaler.transform(features)

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
# Fungsi rekam audio dari mic
# ============================================================
def record_audio(duration=3, fs=22050):
    st.info(f"🎙️ Rekam selama {duration} detik...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(tmp_file.name, fs, recording)
    return tmp_file.name

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

st.sidebar.header("⚙️ Pengaturan Model")
threshold = st.sidebar.slider(
    "Ambang Confidence (Threshold)",
    min_value=0.3,
    max_value=0.9,
    value=0.6,
    step=0.05,
    help="Semakin tinggi nilainya, semakin ketat sistem dalam mengenali suara.",
)

# Pilihan input: Upload atau Rekam
input_mode = st.radio("Pilih metode input audio:", ("Upload File", "Rekam Suara"))

if input_mode == "Upload File":
    uploaded_file = st.file_uploader("🎵 Upload file audio (.wav)", type=["wav"])
    if uploaded_file is not None:
        temp_path = "temp_audio.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
else:
    if st.button("🎤 Mulai Rekam"):
        temp_path = record_audio(duration=3)  # durasi rekam 3 detik
        st.audio(temp_path, format="audio/wav")

# Prediksi jika file ada
if 'temp_path' in locals() and os.path.exists(temp_path):
    st.audio(temp_path, format="audio/wav")
    with st.spinner("⏳ Memproses audio..."):
        speaker, status, prob, probs, labels, y, sr = predict_audio(temp_path, threshold)

    st.markdown("---")
    st.subheader("🎯 Hasil Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Speaker", speaker)
    with col2:
        st.metric("Status Suara", status)

    st.metric("Confidence (%)", f"{prob*100:.2f}%")

    prob_df = pd.DataFrame({
        "Kelas": labels,
        "Probabilitas (%)": [round(float(p)*100, 2) for p in probs]
    }).sort_values("Probabilitas (%)", ascending=False)

    st.markdown("#### 📊 Probabilitas Tiap Kelas")
    st.table(
        prob_df.style.set_table_styles([
            {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]},
            {"selector": "td", "props": [("text-align", "center")]}
        ])
    )

    st.subheader("📈 Waveform Audio")
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform Audio", fontsize=12)
    st.pyplot(fig)

    st.subheader("🎛️ Mel Spectrogram")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("Mel Spectrogram", fontsize=12)
    st.pyplot(fig)

    st.subheader("📉 Distribusi Probabilitas")
    plt.figure(figsize=(6, 4))
    sns.barplot(x="Kelas", y="Probabilitas (%)", data=prob_df)
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    st.pyplot(plt)

    os.remove(temp_path)
else:
    st.info("📂 Silakan upload atau rekam audio terlebih dahulu.")
