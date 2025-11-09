import librosa
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# Load model dan scaler
# -------------------------------
best_model = joblib.load("best_audio_model.pkl")
scaler = joblib.load("scaler_audio.pkl")

# -------------------------------
# Daftar orang yang diizinkan
# -------------------------------
allowed_people = ["Nadia", "Vanisa"]  # hanya orang ini
confidence_threshold = 0.6  # threshold minimal confidence

# -------------------------------
# Fungsi ekstraksi fitur
# -------------------------------
def extract_features(file_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=sr)
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)[0]
    rms = np.sqrt(np.mean(y**2))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), axis=1)
    
    features = [zcr, rms, spectral_centroid, spectral_bandwidth, spectral_contrast] + mfccs.tolist()
    return np.array(features).reshape(1, -1)

# -------------------------------
# Fungsi prediksi dengan filter
# -------------------------------
def predict_audio(file_path):
    features = extract_features(file_path)
    features_scaled = scaler.transform(features)
    
    pred_label = best_model.predict(features_scaled)[0]
    pred_prob = best_model.predict_proba(features_scaled).max() if hasattr(best_model, "predict_proba") else np.nan
    
    # Ambil nama orang dari label (misal "Nadia_buka")
    person_name = pred_label.split("_")[0]
    
    # Filter: hanya allowed_people dan confidence > threshold
    if person_name not in allowed_people or pred_prob < confidence_threshold:
        pred_label = "Tidak dikenal"
    
    return {"file": file_path, "pred_label": pred_label, "confidence": pred_prob}

# -------------------------------
# Contoh file audio
# -------------------------------
audio_files = [
    "clean_audio/Nadia/buka/buka 1.wav",
    "clean_audio/Nadia/tutup/tutup1.wav",
    "clean_audio/ulva_tutup.wav"  # contoh orang lain
]

# -------------------------------
# Prediksi semua file
# -------------------------------
results = [predict_audio(f) for f in audio_files]
df_results = pd.DataFrame(results)
print("✅ Hasil Prediksi Audio Baru:")
print(df_results)

# Simpan ke CSV
df_results.to_csv("prediksi_audio_filtered.csv", index=False)
print("✅ Hasil prediksi disimpan ke 'prediksi_audio_filtered.csv'")
