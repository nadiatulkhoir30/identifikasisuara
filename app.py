# ==================================================
# 🎧 APLIKASI IDENTIFIKASI SUARA DENGAN STREAMLIT
# ==================================================
import streamlit as st
import numpy as np
import librosa
import joblib
import os
import tempfile

# ==================================================
# 🔹 FUNGSI EKSTRAKSI FITUR (HARUS SESUAI TRAINING)
# ==================================================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    # Hitung fitur utama
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rms = np.sqrt(np.mean(y ** 2))
    sc = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    sb = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    scon = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # Gabungkan ke satu vektor fitur
    features = np.hstack([zcr, rms, sc, sb, scon, mfccs])
    return features.reshape(1, -1)  # bentuk [1, 18]

# ==================================================
# 🔹 LOAD MODEL & SCALER
# ==================================================
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("best_audio_model.pkl")
    scaler = joblib.load("scaler_audio.pkl")  # dari tahap normalisasi
    return model, scaler

model, scaler = load_model_and_scaler()

# ==================================================
# 🔹 TAMPILAN APLIKASI
# ==================================================
st.title("🎙️ Aplikasi Identifikasi Suara")
st.write("Unggah file audio (.wav) untuk mengenali siapa yang berbicara atau jenis suara yang terdeteksi.")

# Upload file
uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Tampilkan nama file
    st.audio(uploaded_file, format='audio/wav')
    st.write(f"📁 File berhasil diunggah: `{uploaded_file.name}`")

    # ==================================================
    # 🔹 EKSTRAKSI FITUR DAN PREDIKSI
    # ==================================================
    try:
        with st.spinner("🔍 Sedang mengekstraksi fitur dan memprediksi..."):
            features = extract_features(file_path)
            X_new_scaled = scaler.transform(features)
            pred = model.predict(X_new_scaled)[0]
            probas = model.predict_proba(X_new_scaled)[0]

        # ==================================================
        # 🔹 HASIL PREDIKSI
        # ==================================================
        st.success(f"✅ Prediksi: **{pred}**")
        st.write("### Probabilitas Kelas:")
        for label, prob in zip(model.classes_, probas):
            st.write(f"- {label}: **{prob*100:.2f}%**")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")

else:
    st.info("Silakan unggah file audio terlebih dahulu.")

# ==================================================
# 🔹 INFORMASI TAMBAHAN
# ==================================================
st.markdown("---")
st.caption("Dibuat oleh **Nadiatul Khoir** — Universitas Trunojoyo Madura 💙")
