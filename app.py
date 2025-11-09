# ==================================================
# 🎵 APLIKASI PREDIKSI SPEAKER MENGGUNAKAN STREAMLIT
# ==================================================
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
import os
import matplotlib.pyplot as plt
import librosa.display

# ==================================================
# 🔹 LOAD MODEL & SCALER
# ==================================================
@st.cache_resource
def load_model_scaler():
    model = joblib.load("best_audio_model.pkl")
    scaler = joblib.load("scaler_audio.pkl")
    return model, scaler

model, scaler = load_model_scaler()

# ==================================================
# 🔹 FUNGSI EKSTRAKSI FITUR AUDIO
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
    """Ekstraksi fitur dari file audio"""
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
# 🔹 FUNGSI PREDIKSI SPEAKER
# ==================================================
def predict_speaker(file_path, threshold=0.7):
    features, y, sr = extract_features(file_path)
    
    if features.shape[1] != scaler.n_features_in_:
        return None, None, None, f"❌ Jumlah fitur {features.shape[1]} tidak cocok dengan scaler ({scaler.n_features_in_})"

    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    max_prob = np.max(probs)
    pred_label = model.classes_[np.argmax(probs)]

    if max_prob < threshold:
        result = "❌ Speaker tidak dikenal"
    else:
        result = f"✅ Prediksi Speaker: {pred_label}"

    return result, max_prob, (y, sr), None

# ==================================================
# 🌟 TAMPILAN STREAMLIT
# ==================================================
st.set_page_config(page_title="Prediksi Speaker Audio", page_icon="🎧", layout="wide")

st.title("🎙️ Aplikasi Prediksi Speaker Berdasarkan Suara")
st.markdown("Unggah file audio (.wav, .mp3, .m4a) untuk mengetahui siapa pembicaranya berdasarkan model yang sudah dilatih.")

uploaded_file = st.file_uploader("📂 Unggah File Audio", type=["wav", "mp3", "m4a"])

threshold = st.slider("🔽 Atur Threshold Kepercayaan", 0.0, 1.0, 0.7, 0.01)

if uploaded_file is not None:
    # Simpan file sementara
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(temp_path, format="audio/wav")
    
    with st.spinner("⏳ Memproses audio dan mengekstrak fitur..."):
        result, confidence, (y, sr), error = predict_speaker(temp_path, threshold)

    if error:
        st.error(error)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Hasil Prediksi")
            st.success(result)
            st.write(f"**Confidence:** {confidence:.2f}")
            if confidence < threshold:
                st.warning("Confidence di bawah threshold — kemungkinan suara baru atau tidak dikenal.")
        
        with col2:
            st.subheader("🎵 Visualisasi Audio")
            fig, ax = plt.subplots(figsize=(8, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax, color="purple")
            ax.set(title="Gelombang Suara")
            st.pyplot(fig)

        # Ekstraksi fitur detail
        features, _, _ = extract_features(temp_path)
        feature_names = [
            "Zero Crossing Rate", "RMS", "Spectral Centroid",
            "Spectral Bandwidth", "Spectral Contrast"
        ] + [f"MFCC-{i+1}" for i in range(13)]

        df_features = pd.DataFrame(features, columns=feature_names)
        st.subheader("🔍 Detail Fitur Audio")
        st.dataframe(df_features.T.rename(columns={0: "Nilai"}), use_container_width=True)

    # Hapus file sementara
    os.remove(temp_path)
else:
    st.info("📁 Silakan unggah file audio terlebih dahulu untuk melakukan prediksi.")
