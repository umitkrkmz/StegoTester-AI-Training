# src/final_detector.py
# The Master Brain of StegoTester AI
# Combines Rule-Based Math (for Audio LSB) and Machine Learning (for Image & Audio Echo).

import joblib
import numpy as np
import soundfile as sf
import cv2
import pandas as pd
import librosa
from scipy.stats import entropy, skew, kurtosis
from pathlib import Path

# --- LOAD MODELS ---
print("Loading AI Models...")
try:
    # Ensure you have trained models in the 'models/' directory
    IMG_MODEL = joblib.load("models/stego_model_image.pkl")
    AUD_MODEL = joblib.load("models/stego_model_audio.pkl")
    
    # Optional: Load scalers if your model requires normalization (e.g., SVM/NeuralNet)
    # IMG_SCALER = joblib.load("models/scaler_image.pkl") 
    # AUD_SCALER = joblib.load("models/scaler_audio.pkl")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Critical Error: Models not found! ({e})")
    print("   Please run 'train_advanced.py' first to generate models.")
    exit()

# --- CONFIGURATION ---
# Threshold for Rule-Based Audio LSB Detection
# Calculated based on statistical analysis of clean vs stego samples.
AUDIO_LSB_THRESHOLD = 0.455 

def extract_image_features(img_path):
    """ Extracts statistical features from an image for the ML model. """
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        
        # 1. LSB Analysis
        lsb = img & 1
        counts = np.bincount(lsb.flatten(), minlength=2)
        prob = counts / (np.sum(counts) + 1e-10)
        lsb_ent = entropy(prob, base=2)
        
        # 2. Adjacent Pixel Difference (APD)
        diff_h = np.abs(img[:, :-1] - img[:, 1:])
        diff_v = np.abs(img[:-1, :] - img[1:, :])
        apd = (np.mean(diff_h) + np.mean(diff_v)) / 2.0
        
        # 3. Edge Density
        edges = cv2.Canny(img, 100, 200)
        edge_dens = np.mean(edges) / 255.0
        
        # Return as DataFrame (must match training columns)
        features = pd.DataFrame([{
            'img_mean': np.mean(img),
            'img_std': np.std(img),
            'img_lsb_entropy': lsb_ent,
            'img_apd_mean': apd,
            'img_edge_density': edge_dens,
            'img_skew': skew(img.flatten())
        }])
        return features
    except:
        return None

def extract_audio_features_for_ml(aud_path):
    """ Extracts spectral features for the Audio ML model (Echo/SSIS detection). """
    try:
        # 1. Bit-Level Analysis
        data_int, _ = sf.read(str(aud_path), dtype='int16')
        if len(data_int.shape) > 1: data_int = np.mean(data_int, axis=1).astype(np.int16)
        
        lsb = data_int & 1
        counts = np.bincount(lsb.flatten(), minlength=2)
        prob = counts / (np.sum(counts) + 1e-10)
        lsb_ent = entropy(prob, base=2)
        
        # LSB Correlation
        lsb_corr = 0
        if len(lsb) > 1:
            c = np.corrcoef(lsb[:-1], lsb[1:])[0, 1]
            lsb_corr = 0 if np.isnan(c) else c
            
        # Transition Rate
        trans = np.sum(np.abs(np.diff(lsb))) / len(lsb)
        
        # 2. Spectral Analysis (Librosa)
        y, sr = librosa.load(str(aud_path), sr=22050)
        if len(y) < 512: y = np.pad(y, (0, 512 - len(y)))
        
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
        flat = np.mean(librosa.feature.spectral_flatness(y=y))
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        
        features = pd.DataFrame([{
            'aud_lsb_entropy': lsb_ent,
            'aud_lsb_corr': lsb_corr,
            'aud_lsb_trans': trans,
            'aud_kurtosis': kurtosis(data_int),
            'aud_spec_flatness': flat,
            'aud_mfcc_mean': mfcc,
            'aud_spec_cent_mean': cent,
            'aud_rms_mean': rms
        }])
        return features
    except:
        return None

def analyze_file(file_path):
    """ Main analysis function: Determines file type and routes to appropriate detector. """
    file_path = Path(file_path)
    if not file_path.exists():
        print("Error: File not found.")
        return

    ext = file_path.suffix.lower()
    
    # === IMAGE ANALYSIS ===
    if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        print(f"\nANALYZING IMAGE: {file_path.name}")
        feat = extract_image_features(file_path)
        
        if feat is not None:
            # AI Decision
            prediction = IMG_MODEL.predict(feat)[0]
            probability = IMG_MODEL.predict_proba(feat)[0][1] # Probability of being Stego
            
            result = "STEGO DETECTED" if prediction == 1 else "CLEAN"
            print(f"   Result: {result}")
            print(f"   Confidence: {probability*100:.2f}%")
            if prediction == 1:
                print("   Reason: Anomalies detected in pixel correlation and LSB patterns.")
        else:
            print("   Error: Could not extract features.")

    # === AUDIO ANALYSIS (HYBRID ENGINE) ===
    elif ext in ['.wav', '.flac']:
        print(f"\nANALYZING AUDIO: {file_path.name}")
        
        # PHASE 1: Mathematical Rule (The Gatekeeper)
        # Checks for high-frequency bit transitions typical of LSB steganography.
        try:
            data_int, _ = sf.read(str(file_path), dtype='int16')
            if len(data_int.shape) > 1: data_int = data_int[:, 0]
            lsb = data_int & 1
            trans_rate = np.sum(np.abs(np.diff(lsb))) / (len(lsb) - 1)
            
            print(f"   Bit Transition Rate: {trans_rate:.5f}")
            
            if trans_rate > AUDIO_LSB_THRESHOLD:
                print("   Result: STEGO DETECTED (LSB)")
                print("   Method: Mathematical Rule-Based Detection")
                print(f"   Reason: Bit transition rate exceeds natural threshold ({AUDIO_LSB_THRESHOLD}).")
                return # Stop here, no need for ML.
                
        except Exception as e:
            print(f"   Warning: Bit analysis failed ({e}). Proceeding to AI...")

        # PHASE 2: AI Analysis (The Detective)
        # Checks for Echo Hiding, Spread Spectrum, or subtle anomalies.
        print("   LSB check passed. Consulting AI for advanced concealment (Echo, etc.)...")
        feat = extract_audio_features_for_ml(file_path)
        
        if feat is not None:
            prediction = AUD_MODEL.predict(feat)[0]
            probability = AUD_MODEL.predict_proba(feat)[0][1]
            
            result = "STEGO DETECTED (Complex)" if prediction == 1 else "CLEAN"
            print(f"   Result: {result}")
            print(f"   AI Confidence: {probability*100:.2f}%")
        else:
            print("   Error: Could not extract ML features.")
            
    else:
        print("Error: Unsupported file format.")

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    print("--- STEGOTESTER AI DIAGNOSTIC TOOL ---")
    # Example usage:
    # analyze_file("path/to/your/test_file.wav")
    print("System ready. Import 'analyze_file' to use.")