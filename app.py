# ==================================================
# 🎯 PREDIKSI SPEAKER BARU DENGAN DETEKSI UNKNOWN
# ==================================================

import librosa
import numpy as np
import joblib

# -------------------------------
# 1️⃣ Load model & scaler
# -------------------------------
model = joblib.load("best_audio_model.pkl")
scaler = joblib.load("scaler_audio.pkl")

# -------------------------------
# 2️⃣ Fungsi ekstraksi fitur audio
# -------------------------------
def extract_features(file_path, n_mfcc=26):
    """
    Ekstrak fitur dari file audio:
    - zcr, rms, spectral centroid, spectral bandwidth, spectral contrast
    - MFCC (default 26)
    """
    y, sr = librosa.load(file_path, sr=22050)
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)[0]
    rms_val = np.sqrt(np.mean(y**2))
    sc = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    sb = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    scon = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
    
    features = np.array([zcr, rms_val, sc, sb, scon] + mfccs.tolist()).reshape(1, -1)
    return features

# -------------------------------
# 3️⃣ Fungsi prediksi speaker dengan threshold
# -------------------------------
def predict_speaker(file_path, threshold=0.6):
    """
    Prediksi speaker, jika probabilitas tertinggi < threshold, maka Unknown
    """
    # Ekstrak fitur
    features_new = extract_features(file_path, n_mfcc=26)
    
    # Normalisasi
    features_scaled = scaler.transform(features_new)
    
    # Prediksi probabilitas
    probs = model.predict_proba(features_scaled)[0]
    class_idx = np.argmax(probs)
    prob_max = probs[class_idx]
    
    # Tentukan label
    if prob_max < threshold:
        return "Unknown", prob_max
    else:
        return model.classes_[class_idx], prob_max

# -------------------------------
# 4️⃣ Contoh penggunaan
# -------------------------------
file_test = "clean_audio/Nadia/buka/buka1.wav"  # ganti sesuai file audio baru
speaker, confidence = predict_speaker(file_test, threshold=0.6)

print(f"Prediksi Speaker: {speaker}")
print(f"Confidence: {confidence:.2f}")
