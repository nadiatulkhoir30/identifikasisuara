# ==================================================
# 🔹 IMPORT LIBRARY
# ==================================================
import streamlit as st
import numpy as np
import librosa
import joblib
import os
import tempfile

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
    """Ekstraksi fitur untuk satu file audio"""
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
    return np.array(features).reshape(1, -1)

# ==================================================
# 🔹 LOAD MODEL & SCALER
# ==================================================
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("best_audio_model.pkl")
    scaler = joblib.load("scaler_audio.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# ==================================================
# 🔹 PREDIKSI SPEAKER DENGAN THRESHOLD
# ==================================================
def predict_speaker(file_path, threshold=0.7):
    features = extract_features(file_path)

    if features.shape[1] != scaler.n_features_in_:
        st.error(f"Jumlah fitur {features.shape[1]} tidak cocok dengan scaler ({scaler.n_features_in_})")
        return None, None

    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    max_prob = np.max(probs)
    pred_label = model.classes_[np.argmax(probs)]

    if max_prob < threshold:
        return "Tidak dikenal", max_prob
    else:
        return pred_label, max_prob

# ==================================================
# 🔹 STREAMLIT APP
# ==================================================
st.title("🎙️ Voice Recognition App")
st.write("Unggah file audio untuk mendeteksi siapa speaker-nya.")

# Upload file audio
uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav", "mp3", "ogg"])

threshold = st.slider("Threshold Kepercayaan", 0.0, 1.0, 0.7, 0.05)

if uploaded_file is not None:
    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Tampilkan audio player
    st.audio(uploaded_file, format="audio/wav")

    # Tombol prediksi
    if st.button("🔍 Prediksi Speaker"):
        with st.spinner("Menganalisis audio..."):
            pred_label, confidence = predict_speaker(temp_path, threshold)

        if pred_label is None:
            st.error("Gagal melakukan prediksi. Periksa log atau model.")
        elif pred_label == "Tidak dikenal":
            st.warning(f"❌ Speaker tidak dikenal (confidence: {confidence:.2f})")
        else:
            st.success(f"✅ Prediksi Speaker: **{pred_label}** (confidence: {confidence:.2f})")

        # Hapus file sementara
        os.remove(temp_path)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit & Librosa")
