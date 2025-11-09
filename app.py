# =========================================
# APP STREAMLIT: Prediksi Suara Buka/Tutup + Speaker (Fixed)
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

# ===============================
# Konfigurasi Tampilan Streamlit
# ===============================
st.set_page_config(
    page_title="Prediksi Suara Buka/Tutup & Speaker",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ===============================
# Load model dan scaler
# ===============================
@st.cache_resource
def load_model_scaler():
    try:
        model = joblib.load("best_audio_model.pkl")  # model closed-set speaker + Buka/Tutup
        scaler = joblib.load("scaler_audio.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau scaler: {e}")
        st.stop()

model, scaler = load_model_scaler()

# ===============================
# Fungsi bantu ekstraksi fitur
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

# ===============================
# Fungsi ekstraksi fitur Streamlit (18 fitur)
# ===============================
def extract_features_streamlit(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    y = librosa.util.normalize(y)
    
    # Samakan durasi audio (2 detik)
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
    features.extend(mfccs.tolist())  # total 18 fitur

    return np.array(features).reshape(1, -1), y, sr

# ===============================
# Fungsi prediksi speaker + status + unknown detection
# ===============================
def predict_audio(file_path, threshold=0.75):
    features, y, sr = extract_features_streamlit(file_path)
    features_scaled = scaler.transform(features)

    probs = model.predict_proba(features_scaled)[0]
    max_prob = np.max(probs)
    pred_label = model.classes_[np.argmax(probs)]  # e.g. "NADIA_BUKA"

    if max_prob < threshold:
        speaker = "Unknown"
        status = "Tidak diketahui"
    else:
        # Pisahkan label sesuai format model: SPEAKER_STATUS
        if "_" in pred_label:
            parts = pred_label.split("_")
            speaker = parts[0].capitalize()
            status = parts[1].capitalize()
        else:
            speaker = pred_label.capitalize()
            status = "Tidak diketahui"

    return speaker, status, max_prob, probs, features_scaled, y, sr

# ===============================
# Header aplikasi UI
# ===============================
st.markdown(
    """
    <div style="text-align:center">
        <h1>🎧 Prediksi Suara Buka/Tutup & Speaker</h1>
        <p style="font-size:17px;">Upload file audio (.wav) untuk mendeteksi siapa speaker dan apakah suaranya <b>Buka</b> atau <b>Tutup</b>.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# Upload audio
# ===============================
uploaded_file = st.file_uploader("🎵 Pilih file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        st.audio(uploaded_file, format="audio/wav")
        st.info("🎶 File audio berhasil diunggah. Sedang diproses...")

        # Prediksi
        speaker, status, max_prob, probabilities, features_scaled, y, sr = predict_audio(temp_path)

        # ===============================
        # Tampilan hasil prediksi
        # ===============================
        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Speaker", value=f"{speaker}")
        with col2:
            st.metric(label="Status Suara", value=f"{status}")
        
        st.metric(label="Confidence Tertinggi (%)", value=f"{max_prob*100:.2f}%")

        # Probabilitas tiap kelas
        prob_df = pd.DataFrame({
            "Kelas": model.classes_,
            "Probabilitas (%)": [round(float(p)*100, 2) for p in probabilities]
        })
        st.markdown("#### 🔍 Probabilitas Tiap Kelas:")
        st.dataframe(prob_df, use_container_width=True)

        # Waveform
        fig, ax = plt.subplots(figsize=(8, 2.5))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform Audio")
        ax.set_xlabel("Waktu (detik)")
        ax.set_ylabel("Amplitudo")
        st.pyplot(fig)

        # Spectrogram
        st.subheader("📊 Spectrogram")
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        st.pyplot(plt)

        # Bar plot probabilitas
        st.subheader("📊 Probabilitas Model")
        plt.figure(figsize=(6, 4))
        sns.barplot(x="Kelas", y="Probabilitas (%)", data=prob_df)
        plt.ylim(0, 100)
        plt.title("Probabilitas Prediksi")
        plt.tight_layout()
        st.pyplot(plt)

        # Debug Info
        with st.expander("🧠 Debug Info (cek fitur dan nilai):"):
            st.write("Fitur setelah normalisasi (18 dimensi):")
            st.dataframe(pd.DataFrame(features_scaled, columns=[f'feat_{i+1}' for i in range(18)]))
            st.write("Probabilitas mentah:", probabilities)

        # Footer
        st.markdown(
            """
            <div style="text-align:center; color:gray; font-size:13px;">
            Model menggunakan fitur audio (ZCR, RMS, Spectral, MFCC).<br>
            Threshold digunakan untuk deteksi suara asing.
            </div>
            """,
            unsafe_allow_html=True
        )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    st.warning("📂 Silakan upload file audio terlebih dahulu untuk melakukan prediksi.")
