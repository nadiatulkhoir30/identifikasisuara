# =========================================
# APP STREAMLIT: Prediksi Suara Buka / Tutup (Versi Stabil & Lengkap)
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
    page_title="Prediksi Suara Buka/Tutup",
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
        model = joblib.load("rf_model.pkl")  # Ganti sesuai nama model
        scaler = joblib.load("scaler.pkl")   # Ganti sesuai nama scaler
        return model, scaler
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau scaler: {e}")
        st.stop()

model, scaler = load_model_scaler()

# ===============================
# Fungsi Ekstraksi Fitur
# ===============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    y = librosa.util.normalize(y)

    # Samakan durasi audio (misal 2 detik)
    target_length = 2 * sr
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]

    # Fitur audio
    duration = librosa.get_duration(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms_val = np.mean(librosa.feature.rms(y=y))

    features = {
        "duration_sec": duration,
        "chroma_mean": np.mean(chroma),
        "zero_crossing_rate": zcr,
        "rms": rms_val
    }

    # Pilih beberapa MFCC penting (mirip training)
    for i in [4, 8, 9, 10, 11, 13]:
        if i <= mfcc.shape[0]:
            features[f"mfcc_{i}"] = np.mean(mfcc[i-1])

    return features, y, sr

# ===============================
# Fungsi Prediksi
# ===============================
def predict_audio(file_path):
    features, y, sr = extract_features(file_path)
    features_df = pd.DataFrame([features])

    # Pastikan fitur sesuai scaler
    if features_df.shape[1] != scaler.n_features_in_:
        st.error(f"Jumlah fitur ({features_df.shape[1]}) tidak cocok dengan scaler ({scaler.n_features_in_})")
        return None, None, None, None, None

    # Normalisasi
    features_scaled = scaler.transform(features_df)

    # Prediksi
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    return prediction, probabilities, features_df, y, sr

# ===============================
# Header aplikasi
# ===============================
st.markdown(
    """
    <div style="text-align:center">
        <h1>🎧 Prediksi Suara <span style="color:#1E90FF;">Buka</span> / <span style="color:#FF6347;">Tutup</span></h1>
        <p style="font-size:17px;">Upload file audio kamu (.wav) untuk mendeteksi apakah suaranya <b>Buka</b> atau <b>Tutup</b>.</p>
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
        # Simpan sementara
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(uploaded_file, format="audio/wav")
        st.info("🎶 File audio berhasil diunggah. Sedang diproses...")

        # Prediksi
        prediction, probabilities, features_df, y, sr = predict_audio(temp_path)

        if prediction is not None:
            # ===============================
            # Tampilan hasil prediksi
            # ===============================
            st.markdown("---")
            st.subheader("🎯 Hasil Prediksi")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Prediksi", value=f"{prediction.upper()}")
            with col2:
                st.metric(label="Kepercayaan Model (%)", value=f"{max(probabilities)*100:.2f}%")

            if prediction.lower() == "buka":
                st.success("✅ Suara ini terdeteksi sebagai **BUKA** 🔊")
            else:
                st.error("🔒 Suara ini terdeteksi sebagai **TUTUP** 🤫")

            # ===============================
            # Probabilitas tiap kelas
            # ===============================
            prob_df = pd.DataFrame({
                "Kelas": model.classes_,
                "Probabilitas (%)": [round(float(p)*100, 2) for p in probabilities]
            })
            st.markdown("#### 🔍 Probabilitas Tiap Kelas:")
            st.dataframe(prob_df, use_container_width=True)

            # ===============================
            # Waveform
            # ===============================
            fig, ax = plt.subplots(figsize=(8, 2.5))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title("Waveform Audio")
            ax.set_xlabel("Waktu (detik)")
            ax.set_ylabel("Amplitudo")
            st.pyplot(fig)

            # ===============================
            # Spectrogram
            # ===============================
            st.subheader("📊 Spectrogram")
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            st.pyplot(plt)

            # ===============================
            # Bar plot probabilitas
            # ===============================
            st.subheader("📊 Probabilitas Model")
            plt.figure(figsize=(6, 4))
            sns.barplot(x="Kelas", y="Probabilitas (%)", data=prob_df)
            plt.ylim(0, 100)
            plt.title("Probabilitas Prediksi")
            plt.tight_layout()
            st.pyplot(plt)

            # ===============================
            # Debug Info
            # ===============================
            with st.expander("🧠 Debug Info (cek fitur dan nilai):"):
                st.write("Fitur sebelum normalisasi:")
                st.dataframe(features_df)
                st.write("Fitur setelah normalisasi:")
                st.dataframe(pd.DataFrame(features_scaled, columns=features_df.columns))
                st.write("Probabilitas mentah:", probabilities)

            # ===============================
            # Catatan Footer
            # ===============================
            st.markdown(
                """
                <div style="text-align:center; color:gray; font-size:13px;">
                Model menggunakan fitur audio (MFCC, RMS, ZCR, Chroma).<br>
                Pastikan file audio mirip dengan data training untuk hasil akurat.
                </div>
                """,
                unsafe_allow_html=True
            )

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    st.warning("📂 Silakan upload file audio terlebih dahulu untuk melakukan prediksi.")
