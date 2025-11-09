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

from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

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
def predict_audio(file_path=None, y=None, sr=22050, threshold=0.6):
    if file_path:
        features, y, sr = extract_features(file_path)
    else:
        # Jika input berupa array y langsung dari voice
        features = [
            zero_crossing_rate(y),
            rms(y),
            spectral_centroid(y, sr),
            spectral_bandwidth(y, sr),
            spectral_contrast(y, sr),
        ]
        mfccs = mfcc_features(y, sr)
        features.extend(mfccs.tolist())
        features = np.array(features).reshape(1, -1)

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
# UI Streamlit
# ============================================================
st.title("🎧 Prediksi Suara Buka/Tutup")
st.markdown("""
<p style="font-size:16px;">Aplikasi ini hanya menerima suara dari <b>Nadia</b> dan <b>Vanisa</b>.<br>
Jika suara lain terdeteksi, hasil akan menjadi <b>Unknown</b>.</p>
""", unsafe_allow_html=True)

# 🧩 Slider threshold
st.sidebar.header("⚙️ Pengaturan Model")
threshold = st.sidebar.slider(
    "Ambang Confidence (Threshold)",
    min_value=0.3,
    max_value=0.9,
    value=0.6,
    step=0.05,
    help="Semakin tinggi nilainya, semakin ketat sistem dalam mengenali suara.",
)

# ============================================================
# Upload file audio
# ============================================================
uploaded_file = st.file_uploader("🎵 Upload file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_path, format="audio/wav")

    with st.spinner("⏳ Memproses audio..."):
        speaker, status, prob, probs, labels, y, sr = predict_audio(file_path=temp_path, threshold=threshold)

    os.remove(temp_path)

# ============================================================
# Rekam suara langsung (voice)
# ============================================================
st.markdown("### 🎤 Rekam Suara Langsung")
webrtc_ctx = webrtc_streamer(
    key="audio-predictor",
    mode=WebRtcMode.SENDONLY,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
    ),
    audio_receiver_size=1024,
)

if webrtc_ctx.audio_receiver:
    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
    if audio_frames:
        audio_data = np.concatenate([f.to_ndarray() for f in audio_frames], axis=0)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        y = audio_data.astype(np.float32)
        sr = 44100

# ============================================================
# Tampilkan hasil prediksi (sama seperti versi upload)
# ============================================================
if uploaded_file is not None or (webrtc_ctx.audio_receiver and y is not None):
    speaker, status, prob, probs, labels, y_proc, sr_proc = predict_audio(y=y, sr=sr, threshold=threshold)

    st.markdown("---")
    st.subheader("🎯 Hasil Prediksi")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Speaker", speaker)
    with col2:
        st.metric("Status Suara", status)

    st.metric("Confidence (%)", f"{prob*100:.2f}%")

    # Tabel Probabilitas
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

    # Visualisasi Audio
    st.subheader("📈 Waveform Audio")
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(y_proc, sr=sr_proc, ax=ax)
    ax.set_title("Waveform Audio", fontsize=12)
    st.pyplot(fig)

    st.subheader("🎛️ Mel Spectrogram")
    S = librosa.feature.melspectrogram(y=y_proc, sr=sr_proc)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr_proc, x_axis='time', y_axis='mel', ax=ax)
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
else:
    st.info("📂 Silakan upload file audio atau rekam suara terlebih dahulu.")
