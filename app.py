# =========================================
# APP STREAMLIT: Prediksi Suara Buka/Tutup & Speaker (Debug + Robust)
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

st.set_page_config(page_title="Prediksi Suara (Debug)", page_icon="🎵", layout="centered")

# -------------------------
# Load model & scaler
# -------------------------
@st.cache_resource
def load_model_scaler():
    try:
        model = joblib.load("best_audio_model.pkl")
        scaler = joblib.load("scaler_audio.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model/scaler: {e}")
        st.stop()

model, scaler = load_model_scaler()

# derive known speakers from model classes
model_classes = list(model.classes_)
derived_known_speakers = sorted(set([c.split("_")[0].lower() for c in model_classes]))

# -------------------------
# Feature funcs (same as training)
# -------------------------
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
    # load with fixed sr and mono to match training pipeline
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    # compute features exactly as training
    feats = [
        zero_crossing_rate(y),
        rms(y),
        spectral_centroid(y, sr),
        spectral_bandwidth(y, sr),
        spectral_contrast(y, sr)
    ]
    mfccs = mfcc_features(y, sr)
    feats.extend(mfccs.tolist())
    return np.array(feats).reshape(1, -1), y, sr

# -------------------------
# Prediction function (debug-friendly)
# -------------------------
def predict_audio_debug(file_path, threshold=0.6, force_accept=False):
    features, y, sr = extract_features(file_path)

    # debug info container
    debug = {"features_raw": features.flatten().tolist(), "features_shape": features.shape}

    # validate with scaler
    n_expected = getattr(scaler, "n_features_in_", None)
    debug["scaler_n_features_in_"] = n_expected
    if n_expected is not None and features.shape[1] != n_expected:
        raise ValueError(f"Jumlah fitur ({features.shape[1]}) tidak cocok dengan scaler ({n_expected}).")

    # scale
    features_scaled = scaler.transform(features)
    debug["features_scaled"] = features_scaled.flatten().tolist()

    # predict proba
    probs = model.predict_proba(features_scaled)[0]
    idx_sorted = np.argsort(probs)[::-1]
    top_indices = idx_sorted[:5]
    top_preds = [(model.classes_[i], float(probs[i])) for i in top_indices]

    max_prob = float(np.max(probs))
    pred_idx = int(np.argmax(probs))
    pred_label = model.classes_[pred_idx]

    # parse label
    if "_" in pred_label:
        speaker_name, status = pred_label.split("_", 1)
    else:
        speaker_name, status = pred_label, "-"

    # derive known speakers from model (to be safe)
    known_speakers = derived_known_speakers

    # decision logic
    reason = None
    if speaker_name.lower() not in known_speakers:
        reason = f"predicted speaker '{speaker_name}' tidak ada di daftar known_speakers"
        final_speaker = "Unknown"
        final_status = "Tidak diketahui"
    elif max_prob < threshold and not force_accept:
        reason = f"confidence {max_prob:.3f} < threshold {threshold:.3f}"
        final_speaker = "Unknown"
        final_status = "Tidak diketahui"
    else:
        final_speaker = speaker_name.capitalize()
        final_status = status.capitalize() if status else "-"

    # fill debug
    debug.update({
        "model_classes": model_classes,
        "known_speakers_derived": known_speakers,
        "pred_label": pred_label,
        "pred_prob": float(probs[pred_idx]),
        "max_prob": max_prob,
        "top_predictions": [(p, round(pr, 4)) for p, pr in top_preds],
        "decision_reason": reason,
        "final_speaker": final_speaker,
        "final_status": final_status
    })

    return final_speaker, final_status, max_prob, probs, features_scaled, y, sr, debug

# -------------------------
# UI
# -------------------------
st.title("🎧 Prediksi Suara (Debug & Robust)")
st.markdown(f"**Model classes (exact):** `{model_classes}`")
st.markdown(f"**Derived known speakers:** {', '.join([s.capitalize() for s in derived_known_speakers])}")

threshold = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
force_accept = st.sidebar.checkbox("Force accept prediction even if < threshold", value=False)

uploaded_file = st.file_uploader("Upload .wav", type=["wav"])
if uploaded_file is not None:
    tmp = "tmp_audio_uploaded.wav"
    with open(tmp, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(tmp, format="audio/wav")
    st.info("Processing...")

    try:
        speaker, status, max_prob, probs, features_scaled, y, sr, debug = predict_audio_debug(tmp, threshold=threshold, force_accept=force_accept)

        # Results
        st.subheader("Hasil Prediksi")
        st.metric("Speaker", speaker)
        st.metric("Status", status)
        st.metric("Confidence (%)", f"{max_prob*100:.2f}%")

        # Probabilities table
        prob_df = pd.DataFrame({
            "Kelas": model.classes_,
            "Prob": [float(p) for p in probs]
        }).sort_values("Prob", ascending=False)
        prob_df["Prob (%)"] = (prob_df["Prob"] * 100).round(2)
        st.markdown("#### Probabilitas tiap kelas (desc):")
        st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)

        # show top-3
        st.markdown("**Top predictions:**")
        topn = prob_df.head(5)
        for i, row in topn.iterrows():
            st.write(f"{i+1}. {row['Kelas']} — {row['Prob (%)']}%")

        # Waveform & spectrogram
        st.subheader("Waveform")
        fig1, ax1 = plt.subplots(figsize=(8, 2.5))
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title("Waveform")
        st.pyplot(fig1)

        st.subheader("Mel Spectrogram")
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax2)
        fig2.colorbar(img, ax=ax2, format="%+2.f dB")
        st.pyplot(fig2)

        # Debug panel
        with st.expander("🧾 Debug detail (features, decision, top preds)"):
            st.write("Decision debug:", debug["decision_reason"])
            st.write("Final speaker:", debug["final_speaker"])
            st.write("Final status:", debug["final_status"])
            st.write("Model classes (exact):", debug["model_classes"])
            st.write("Derived known speakers:", debug["known_speakers_derived"])
            st.write("Top predictions (label, prob):", debug["top_predictions"])
            st.markdown("Features (raw):")
            st.dataframe(pd.DataFrame([debug["features_raw"]], columns=[f"feat_{i+1}" for i in range(len(debug["features_raw"]))]))
            st.markdown("Features (scaled):")
            st.dataframe(pd.DataFrame(debug["features_scaled"], columns=[f"feat_{i+1}" for i in range(len(debug["features_scaled"]))]))

        # Explain why unknown if Unknown
        if speaker == "Unknown":
            st.warning("Hasil adalah `Unknown`. Periksa detail debug: kemungkinan nama kelas tidak cocok atau confidence < threshold.")
            st.info("Jika kamu yakin ini Vanisa dan confidence sedikit di bawah threshold, aktifkan 'Force accept' di sidebar untuk menerima prediksi.")

    except Exception as e:
        st.error(f"Error saat memproses audio: {e}")

    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
else:
    st.info("Silakan upload file .wav untuk diuji.")
