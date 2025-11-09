import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib
import tempfile
from pydub import AudioSegment

# ------------------------------
# Load model dan scaler
# ------------------------------
@st.cache_resource
def load_model_scaler():
    model = joblib.load("model_knn_regression.pkl")
    scaler = joblib.load("scaler1.pkl")
    return model, scaler

model, scaler = load_model_scaler()

# ------------------------------
# Fungsi ekstraksi fitur audio
# ------------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    stft = np.abs(librosa.stft(y))
    features = np.array([
        np.mean(librosa.feature.chroma_stft(S=stft, sr=sr)),
        np.mean(librosa.feature.rms(y=y)),
        np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    ])
    return features

# ------------------------------
# Halaman Streamlit
# ------------------------------
st.title("🎵 Aplikasi Prediksi Audio (KNN Regression)")

uploaded_file = st.file_uploader("Unggah file audio (.m4a / .wav)", type=["m4a", "wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        if uploaded_file.name.endswith(".m4a"):
            # Konversi m4a ke wav
            with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as tmp_m4a:
                tmp_m4a.write(uploaded_file.read())
                audio = AudioSegment.from_file(tmp_m4a.name, format="m4a")
                audio.export(tmp_wav.name, format="wav")
        else:
            tmp_wav.write(uploaded_file.read())
        
        st.audio(tmp_wav.name, format="audio/wav")
        st.success("✅ File audio berhasil diproses")

        # Ekstraksi fitur
        features = extract_features(tmp_wav.name)
        st.write("📊 Fitur audio hasil ekstraksi:")
        st.write(features)

        # Normalisasi
        scaled_features = scaler.transform([features])

        # Prediksi
        prediction = model.predict(scaled_features)[0]
        st.subheader(f"🎯 Hasil Prediksi: {prediction:.6f}")

        # ------------------------------
        # Visualisasi Spektrogram
        # ------------------------------
        st.subheader("🎨 Spektrogram Audio")
        y, sr = librosa.load(tmp_wav.name, sr=44100)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(8, 3))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        st.pyplot(fig)
