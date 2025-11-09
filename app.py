# ==================================================
# 🔹 IMPORT LIBRARY
# ==================================================
import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# ==================================================
# 🔹 FUNGSIONAL EKSTRAKSI FITUR AUDIO
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
    return np.array(features).reshape(1, -1)

# ==================================================
# 🔹 LOAD MODEL & SCALER
# ==================================================
model = joblib.load("best_audio_model.pkl")
scaler = joblib.load("scaler_audio.pkl")

# ==================================================
# 🔹 PREDIKSI SPEAKER
# ==================================================
def predict_speaker(file_path, threshold=0.7):
    features = extract_features(file_path)
    if features.shape[1] != scaler.n_features_in_:
        return f"❌ Jumlah fitur {features.shape[1]} tidak cocok dengan scaler ({scaler.n_features_in_})"
    features_scaled = scaler.transform(features)
    probs = model.predict_proba(features_scaled)[0]
    max_prob = np.max(probs)
    pred_label = model.classes_[np.argmax(probs)]
    if max_prob < threshold:
        return f"❌ Speaker tidak dikenal (Confidence tertinggi: {max_prob:.2f})"
    else:
        return f"✅ Prediksi Speaker: {pred_label} (Confidence: {max_prob:.2f})"

# ==================================================
# 🔹 STREAMLIT APP
# ==================================================
st.title("🔊 Speaker Recognition")
st.markdown("Upload file audio atau rekam langsung menggunakan mikrofon.")

# ----------------------
# Upload audio
# ----------------------
uploaded_file = st.file_uploader("Pilih file audio (wav)", type=["wav"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
    result = predict_speaker(file_path)
    st.success(result)

# ----------------------
# Rekam audio via browser
# ----------------------
st.markdown("**Atau rekam audio langsung di browser:**")
webrtc_ctx = webrtc_streamer(
    key="speaker-recorder",
    mode=WebRtcMode.SENDONLY,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False}
    ),
)

if webrtc_ctx.audio_receiver:
    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
    if audio_frames:
        # Gabungkan semua frame
        audio_data = np.hstack([f.to_ndarray()[:, 0] for f in audio_frames])
        # Simpan sementara
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        librosa.output.write_wav(temp_file.name, audio_data.astype(np.float32), sr=22050)
        result = predict_speaker(temp_file.name)
        st.success(result)
