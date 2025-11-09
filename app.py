# app_debug_compare.py
import os
import numpy as np
import pandas as pd
import librosa
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import librosa.display
import seaborn as sns

st.set_page_config(page_title="Audio Predict Debug", layout="wide")

st.title("🔍 Debug Prediksi Suara — Compare offline vs Streamlit")

# -------------------------
# Load model & scaler (cached)
# -------------------------
@st.cache_resource
def load_model_scaler():
    model = joblib.load("best_audio_model.pkl")
    scaler = joblib.load("scaler_audio.pkl")
    return model, scaler

model, scaler = load_model_scaler()

st.sidebar.header("Pengaturan")
threshold = st.sidebar.slider("Threshold (confidence)", 0.0, 1.0, 0.7, 0.01)
force_accept = st.sidebar.checkbox("Force accept even if below threshold", value=False)
topn = st.sidebar.slider("Show top-N probs", 1, min(10, len(model.classes_)), 5)

# Show model info
st.sidebar.markdown("**Model classes (exact):**")
st.sidebar.code(str(list(model.classes_)))
st.sidebar.markdown(f"scaler.n_features_in_: `{getattr(scaler, 'n_features_in_', None)}`")
if hasattr(scaler, "mean_"):
    st.sidebar.markdown("scaler.mean_ (first 6 vals):")
    st.sidebar.code(np.array2string(scaler.mean_[:6], precision=4, separator=", "))

# -------------------------
# Feature funcs (same as your offline script)
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

def extract_features_offline(file_path):
    """Matches your offline script: returns features array only"""
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

# Reimplement your offline predict_speaker (returns details rather than prints)
def predict_speaker_offline(file_path, threshold=0.7):
    feats, y, sr = extract_features_offline(file_path)
    if feats.shape[1] != scaler.n_features_in_:
        return {
            "error": f"FEATURE_DIM_MISMATCH: {feats.shape[1]} vs scaler.n_features_in_={scaler.n_features_in_}"
        }
    feats_scaled = scaler.transform(feats)
    probs = model.predict_proba(feats_scaled)[0]
    max_prob = float(np.max(probs))
    pred_label = model.classes_[np.argmax(probs)]
    known = max_prob >= threshold
    return {
        "features_raw": feats.flatten().tolist(),
        "features_scaled": feats_scaled.flatten().tolist(),
        "probs": probs.tolist(),
        "max_prob": max_prob,
        "pred_label": pred_label,
        "accepted_by_threshold": bool(known),
        "y": y,
        "sr": sr
    }

# -------------------------
# Streamlit Predict function (same extraction, returns rich debug)
# -------------------------
def predict_streamlit(file_path, threshold=0.7, force_accept=False):
    feats, y, sr = extract_features_offline(file_path)  # same extraction
    if feats.shape[1] != scaler.n_features_in_:
        return {"error": f"FEATURE_DIM_MISMATCH: {feats.shape[1]} vs scaler.n_features_in_={scaler.n_features_in_}"}
    feats_scaled = scaler.transform(feats)
    probs = model.predict_proba(feats_scaled)[0]
    probs = np.array(probs, dtype=float)
    idx_sort = np.argsort(probs)[::-1]
    top_idx = idx_sort[:topn]
    top_preds = [(model.classes_[i], float(probs[i])) for i in top_idx]

    # per-speaker aggregation
    speaker_map = {}
    for lbl, p in zip(model.classes_, probs):
        sp = lbl.split("_")[0].strip().lower()
        speaker_map.setdefault(sp, []).append(float(p))
    speaker_avg = {k: float(np.mean(v)) for k, v in speaker_map.items()}
    speaker_sorted = sorted(speaker_avg.items(), key=lambda x: x[1], reverse=True)

    # decide top label & speaker
    idx_top = int(np.argmax(probs))
    pred_label = model.classes_[idx_top]
    pred_prob = float(probs[idx_top])
    parts = pred_label.lower().split("_")
    speaker = parts[0] if len(parts) > 0 else "unknown"
    status = parts[1].capitalize() if len(parts) > 1 else "-"

    # final decision: by threshold (but show both)
    accepted = (pred_prob >= threshold) or force_accept
    return {
        "features_raw": feats.flatten().tolist(),
        "features_scaled": feats_scaled.flatten().tolist(),
        "probs": probs.tolist(),
        "top_preds": top_preds,
        "pred_label": pred_label,
        "pred_prob": pred_prob,
        "speaker": speaker,
        "status": status,
        "accepted": bool(accepted),
        "speaker_avg": speaker_avg,
        "speaker_sorted": speaker_sorted,
        "y": y,
        "sr": sr
    }

# -------------------------
# UI: upload file
# -------------------------
uploaded = st.file_uploader("Upload .wav to inspect", type=["wav"])
if not uploaded:
    st.info("Upload file .wav yang ingin diuji (mis. Vanisa/Tutup sample).")
    st.stop()

# Save temp
tmp = "tmp_debug_audio.wav"
with open(tmp, "wb") as f:
    f.write(uploaded.read())

st.audio(tmp, format="audio/wav")

# Buttons to run both functions
colA, colB = st.columns(2)
with colA:
    if st.button("Run offline predict_speaker (reference)"):
        res_off = predict_speaker_offline(tmp, threshold=threshold)
        st.subheader("📌 Offline script result")
        if "error" in res_off:
            st.error(res_off["error"])
        else:
            st.write("Pred label:", res_off["pred_label"])
            st.write("Max prob:", res_off["max_prob"])
            st.write("Accepted by threshold:", res_off["accepted_by_threshold"])
            st.markdown("Top probs (offline):")
            df_off = pd.DataFrame({
                "Kelas": list(model.classes_),
                "Prob": [round(float(x)*100,2) for x in res_off["probs"]]
            }).sort_values("Prob", ascending=False)
            st.table(df_off.head(topn))
            # show features raw & scaled
            st.markdown("Features (raw)[:20]:")
            st.code(np.array2string(np.array(res_off["features_raw"])[:20], precision=6, separator=", "))
            st.markdown("Features (scaled)[:20]:")
            st.code(np.array2string(np.array(res_off["features_scaled"])[:20], precision=6, separator=", "))

with colB:
    if st.button("Run Streamlit predict (same pipeline)"):
        res_stream = predict_streamlit(tmp, threshold=threshold, force_accept=force_accept)
        st.subheader("📌 Streamlit pipeline result")
        if "error" in res_stream:
            st.error(res_stream["error"])
        else:
            st.write("Pred label:", res_stream["pred_label"])
            st.write("Pred prob:", round(res_stream["pred_prob"],4))
            st.write("Final accepted (threshold/force):", res_stream["accepted"])
            st.markdown("Top predictions (stream):")
            df_s = pd.DataFrame(res_stream["top_preds"], columns=["Kelas","Prob"]).assign(ProbPct=lambda d: (d["Prob"]*100).round(2))
            st.table(df_s)
            st.markdown("Speaker avg (per speaker):")
            st.table(pd.DataFrame.from_dict(res_stream["speaker_sorted"]).rename(columns={0:"Speaker",1:"AvgProb"}).assign(AvgProbPct=lambda d: (d["AvgProb"]*100).round(2)))
            st.markdown("Features (raw)[:20]:")
            st.code(np.array2string(np.array(res_stream["features_raw"])[:20], precision=6, separator=", "))
            st.markdown("Features (scaled)[:20]:")
            st.code(np.array2string(np.array(res_stream["features_scaled"])[:20], precision=6, separator=", "))

# Always show waveform and spectrogram for eyeballing
st.subheader("Waveform & Spectrogram")
y, sr = librosa.load(tmp, sr=22050)
fig, ax = plt.subplots(2,1, figsize=(10,5))
librosa.display.waveshow(y, sr=sr, ax=ax[0])
ax[0].set(title="Waveform")
S = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel', ax=ax[1])
ax[1].set(title="Mel spectrogram")
st.pyplot(fig)

# remove temp file when done
# (we don't auto-delete so user can re-run quickly; uncomment if you prefer deletion)
# os.remove(tmp)
