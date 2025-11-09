# =========================================
# APP STREAMLIT: Prediksi Suara Buka/Tutup + Speaker (Final & Stabil)
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
    initial_sidebar_state="collapsed"
)

# =========================================
# Load Model dan Scaler
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

# Ambil daftar speaker otomatis dari model
known_speakers = sorted(set([cls.split("_")[0].lower() for cls in model.classes_]))

# =========================================
# Fungsi Ekstraksi Fitur (sesuai model training)
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

def extract_features_streamlit(file_path):
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
# Fungsi Prediksi Audio
# =========================================
def predict_audio(file_path, threshold=0.6):
    features, y, sr = extract_features_streamlit(file_path)

    if features.shape[1] != scaler.n_features_in_:
        st.error(f"❌ Jumlah fitur tidak cocok dengan scaler: {features.shape[1]} vs {scaler.n_features_in_}")
        st.stop()

    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    max_prob = np.max(probs)
    pred_label = model.classes_[np.argmax(probs)]

    # Pisah nama speaker dan status
    if "_" in pred_label:
        speaker_name, status = pred_label.split("_")
    else:
        speaker_name, status = pred_label, "-"

    # Logika Unknown: di luar daftar known_speakers ATAU confidence rendah
    if speaker_name.lower() not in known_speakers or max_prob < threshold:
        speaker = "Unknown"
        status = "Tidak diketahui"
    else:
        speaker = speaker_name.capitalize()
        status = status.capitalize()

    return speaker, status, max_prob, probs, features_scaled, y, sr


# =========================================
# UI: Header
# =========================================
st.markdown(
    f"""
    <div style="text-align:center">
        <h1>🎧 Prediksi Suara Buka/Tutup & Speaker</h1>
        <p style="font-size:17px;">Upload file audio (.wav) untuk mendeteksi siapa speaker dan apakah suaranya <b>Buka</b> atau <b>Tutup</b>.</p>
        <p style="color:gray; font-size:14px;">Speaker yang dikenali model: <b>{", ".join([s.capitalize() for s in known_speakers])}</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================
# Upload File
# =========================================
uploaded_file = st.file_uploader("🎵 Pilih file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(temp_path, format="audio/wav")
        st.info("🎶 File audio berhasil diunggah. Sedang diproses...")

        # Prediksi
        speaker, status, max_prob, probs, features_scaled, y, sr = predict_audio(temp_path, threshold=0.6)

        # =========================================
        # Hasil Prediksi
        # =========================================
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Speaker", value=f"{speaker}")
        with col2:
            st.metric(label="Status Suara", value=f"{status}")

        st.metric(label="Confidence (%)", value=f"{max_prob*100:.2f}%")

        # Probabilitas per kelas
        prob_df = pd.DataFrame({
            "Kelas": model.classes_,
            "Probabilitas (%)": [round(float(p)*100, 2) for p in probs]
        }).sort_values("Probabilitas (%)", ascending=False)

        st.markdown("#### 🔍 Probabilitas Tiap Kelas:")
        st.dataframe(prob_df, use_container_width=True)

        # =========================================
        # Visualisasi Audio
        # =========================================
        st.subheader("📈 Waveform Audio")
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform Audio")
        ax.set_xlabel("Waktu (detik)")
        ax.set_ylabel("Amplitudo")
        st.pyplot(fig)

        st.subheader("📊 Mel Spectrogram")
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("Mel Spectrogram")
        st.pyplot(fig)

        st.subheader("📊 Distribusi Probabilitas Prediksi")
        plt.figure(figsize=(6, 4))
        sns.barplot(x="Kelas", y="Probabilitas (%)", data=prob_df)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.tight_layout()
        st.pyplot(plt)

        with st.expander("🧠 Debug Info"):
            st.write("Fitur hasil ekstraksi (18 dimensi):")
            st.dataframe(pd.DataFrame(features_scaled, columns=[f'feat_{i+1}' for i in range(features_scaled.shape[1])]))
            st.write("Probabilitas mentah:", probs)

        st.markdown(
            """
            <div style="text-align:center; color:gray; font-size:13px;">
            Model menggunakan fitur audio (ZCR, RMS, Spectral, MFCC).<br>
            Threshold digunakan untuk menandai suara asing (Unknown).<br>
            Jika confidence rendah, hasil bisa dianggap tidak pasti.
            </div>
            """,
            unsafe_allow_html=True
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    st.warning("📂 Silakan upload file audio (.wav) untuk memulai prediksi.")
