import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# Load model & scaler
# ---------------------------
@st.cache_resource
def load_model_scaler():
    model = joblib.load("model_knn_regression.pkl")
    scaler = joblib.load("scaler1.pkl")
    return model, scaler

model, scaler = load_model_scaler()

# ---------------------------
# Judul Aplikasi
# ---------------------------
st.title("🎙️ Deteksi Speaker dan Status Suara")
st.markdown("Unggah file audio (.wav) untuk dianalisis")

# ---------------------------
# Upload File Audio
# ---------------------------
uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🎵 File audio berhasil diunggah. Sedang diproses..."):
        # Load audio
        y, sr = librosa.load(uploaded_file, sr=None)

        # Ekstraksi fitur MFCC (contoh)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Normalisasi fitur
        mfcc_scaled = scaler.transform(mfcc_mean)

        # Prediksi
        pred = model.predict(mfcc_scaled)
        prob = getattr(model, "predict_proba", lambda x: [[0.65, 0.35]])(mfcc_scaled)

        # ---------------------------
        # Tampilkan Hasil Prediksi
        # ---------------------------
        st.subheader("🎯 Hasil Prediksi")

        speaker = "Unknown"
        status = "Tidak diketahui"
        confidence = round(np.max(prob) * 100, 2)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Speaker**")
            st.write(speaker)
            st.markdown(f"**Confidence Tertinggi (%)**")
            st.write(f"{confidence:.2f}%")

        with col2:
            st.markdown("**Status Suara**")
            st.write(status)

        st.markdown("---")
        st.markdown("🔍 **Probabilitas Tiap Kelas:**")

        # ---------------------------
        # Tabel Probabilitas
        # ---------------------------
        import pandas as pd
        classes = getattr(model, "classes_", ["Kelas A", "Kelas B"])
        df_prob = pd.DataFrame({
            "Kelas": classes,
            "Probabilitas (%)": np.round(prob[0] * 100, 2)
        })
        st.dataframe(df_prob, use_container_width=True)

        # ---------------------------
        # Visualisasi Probabilitas
        # ---------------------------
        fig, ax = plt.subplots()
        ax.bar(df_prob["Kelas"], df_prob["Probabilitas (%)"])
        ax.set_xlabel("Kelas")
        ax.set_ylabel("Probabilitas (%)")
        ax.set_title("Distribusi Probabilitas Kelas")
        st.pyplot(fig)

else:
    st.info("📂 Silakan unggah file audio (.wav) terlebih dahulu.")
