# ==================================================
# 🎙️ APLIKASI IDENTIFIKASI SUARA TERBATAS (2 SPEAKER)
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
    return features.reshape(1, -1)

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
# 🔹 SPEAKER YANG DIIZINKAN
# ==================================================
allowed_speakers = ["Nadia", "Vanisa"]  # ubah sesuai nama folder training kamu
confidence_threshold = 0.75  # minimal confidence agar diakui

# ==================================================
# 🔹 TAMPILAN APLIKASI
# ==================================================
st.title("🎙️ Aplikasi Identifikasi Suara (2 Speaker Tertentu)")
st.write("Unggah file audio (.wav) untuk mengenali apakah suara termasuk salah satu speaker yang dikenal.")

# Upload file
uploaded_file = st.file_uploader("📂 Unggah file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    # Simpan file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.audio(uploaded_file, format='audio/wav')
    st.write(f"📁 File berhasil diunggah: `{uploaded_file.name}`")

    # ==================================================
    # 🔹 EKSTRAKSI FITUR & PREDIKSI
    # ==================================================
    try:
        with st.spinner("🔍 Menganalisis suara..."):
            features = extract_features(file_path)
            X_new_scaled = scaler.transform(features)
            probas = model.predict_proba(X_new_scaled)[0]
            pred = model.classes_[np.argmax(probas)]
            confidence = np.max(probas)

        # ==================================================
        # 🔹 LOGIKA PEMBATASAN SPEAKER
        # ==================================================
        recognized = False
        for speaker in allowed_speakers:
            if speaker.lower() in pred.lower() and confidence >= confidence_threshold:
                st.success(f"✅ Speaker terdeteksi: **{speaker}** (Confidence: {confidence:.2f})")
                recognized = True
                break

        if not recognized:
            st.error(f"❌ Speaker tidak dikenal atau confidence rendah ({confidence:.2f})")
            st.info("Hanya suara dari speaker yang telah terdaftar yang bisa dikenali.")

        # ==================================================
        # 🔹 TAMPILKAN PROBABILITAS TIAP KELAS
        # ==================================================
        st.write("### 🔢 Probabilitas Kelas:")
        for label, prob in zip(model.classes_, probas):
            st.write(f"- {label}: **{prob*100:.2f}%**")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")

else:
    st.info("Silakan unggah file audio terlebih dahulu.")

# ==================================================
# 🔹 FOOTER
# ==================================================
st.markdown("---")
st.caption("Dibuat oleh **Nadiatul Khoir** — Universitas Trunojoyo Madura 💙")
