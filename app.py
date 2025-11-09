# =========================================
# APP STREAMLIT: Prediksi Suara Buka/Tutup + Speaker (Fix Bias & Agregasi)
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

# =========================================
# Konfigurasi Tampilan Streamlit
# =========================================
st.set_page_config(
    page_title="Prediksi Suara Buka/Tutup & Speaker",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =========================================
# Load model dan scaler
# =========================================
@st.cache_resource
def load_model_scaler():
    try:
        model = joblib.load("best_audio_model.pkl")
        scaler = joblib.load("scaler_audio.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau scaler: {e}")
        st.stop()

model, scaler = load_model_scaler()

# =========================================
# FUNGSI EKSTRAKSI FITUR (identik dengan training)
# =========================================
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

# =========================================
# FUNGSI PREDIKSI (versi agregasi speaker)
# =========================================
def predict_audio(file_path, threshold=0.6, force_accept=False):
    features, y, sr = extract_features(file_path)
    features_scaled = scaler.transform(features)

    probs = model.predict_proba(features_scaled)[0]
    class_labels = model.classes_

    # ----- Agregasi probabilitas per speaker -----
    speaker_probs = {}
    for label, prob in zip(class_labels, probs):
        spk = label.split("_")[0].lower()
        speaker_probs.setdefault(spk, []).append(prob)

    speaker_avg = {spk: np.mean(vals) for spk, vals in speaker_probs.items()}
    speaker_sorted = sorted(speaker_avg.items(), key=lambda x: x[1], reverse=True)
    top_speaker, top_speaker_prob = speaker_sorted[0]

    # Ambil status (buka/tutup) dengan probabilitas tertinggi untuk speaker tsb
    speaker_related = [(lbl, pr) for lbl, pr in zip(class_labels, probs) if lbl.startswith(top_speaker)]
    best_label, best_prob = max(speaker_related, key=lambda x: x[1])

    # Parse label
    speaker_name, status = best_label.split("_", 1)
    status = status.capitalize()

    # Threshold handling
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

    return speaker_name.capitalize(), status, top_speaker_prob, probs, features_scaled, y, sr, debug


# =========================================
# UI: Sidebar
# =========================================
st.sidebar.header("⚙️ Pengaturan")
threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.6, 0.01)
force_accept = st.sidebar.checkbox("Force accept prediction even if < threshold", False)

# =========================================
# UI: Header
# =========================================
st.markdown(
    """
    <div style="text-align:center">
        <h1>🎧 Prediksi Suara Buka/Tutup & Speaker</h1>
        <p style="font-size:17px;">Upload file audio (.wav) untuk mendeteksi <b>siapa speakernya</b> dan <b>apakah suaranya Buka atau Tutup</b>.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Info model
known_speakers = sorted(set([c.split("_")[0].capitalize() for c in model.classes_]))
st.info(f"Derived known speakers: {', '.join(known_speakers)}")

# =========================================
# Upload File Audio
# =========================================
uploaded_file = st.file_uploader("Upload .wav", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(temp_path, format="audio/wav")
        st.info("⏳ Processing...")

        # Prediksi
        speaker, status, top_prob, probs, features_scaled, y, sr, debug = predict_audio(
            temp_path, threshold, force_accept
        )

        # =========================================
        # Hasil Prediksi
        # =========================================
        st.markdown("### 🎯 Hasil Prediksi")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Speaker", speaker)
        with col2:
            st.metric("Status", status)
        st.metric("Confidence (%)", f"{top_prob*100:.2f}%")

        # =========================================
        # Probabilitas Kelas
        # =========================================
        prob_df = pd.DataFrame({
            "Kelas": model.classes_,
            "Prob (%)": [round(p * 100, 2) for p in probs]
        }).sort_values("Prob (%)", ascending=False)
        st.markdown("#### 📊 Probabilitas tiap kelas (desc):")
        st.dataframe(prob_df, use_container_width=True)

        # =========================================
        # Agregasi Probabilitas per Speaker
        # =========================================
        st.markdown("#### 🧩 Probabilitas rata-rata per speaker:")
        speaker_df = pd.DataFrame(debug["speaker_sorted"], columns=["Speaker", "Prob (%)"])
        speaker_df["Prob (%)"] = speaker_df["Prob (%)"] * 100
        st.dataframe(speaker_df, use_container_width=True)

        # Bar chart per speaker
        plt.figure(figsize=(6, 3))
        sns.barplot(x="Speaker", y="Prob (%)", data=speaker_df)
        plt.ylim(0, 100)
        plt.title("Distribusi Probabilitas per Speaker")
        st.pyplot(plt)

        # =========================================
        # Visualisasi Audio
        # =========================================
        st.markdown("### 🎵 Visualisasi Audio")
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform Audio")
        st.pyplot(fig)

        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("Mel Spectrogram")
        st.pyplot(fig)

        # =========================================
        # Debug Info
        # =========================================
        with st.expander("🧠 Debug Info"):
            st.write("Fitur hasil ekstraksi (18 dimensi):")
            st.dataframe(pd.DataFrame(features_scaled, columns=[f"feat_{i+1}" for i in range(features_scaled.shape[1])]))
            st.write("Rincian speaker prob:", debug["speaker_avg"])

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    st.warning("📂 Silakan upload file audio untuk melakukan prediksi.")
