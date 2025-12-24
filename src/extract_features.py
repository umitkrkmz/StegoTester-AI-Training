# src/extract_features.py (FINAL - AUTOCORRELATION BOOST)
# Adds LSB Correlation metrics to distinguish "Natural Noise" from "Stego Noise".

import os
import cv2
import numpy as np
import soundfile as sf
import pandas as pd
import librosa
from scipy.stats import entropy, skew, kurtosis, pearsonr
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
DATASET_MAP = Path("dataset/dataset_map.csv")
OUTPUT_CSV = Path("dataset/features.csv")

def calculate_image_features(img_path):
    """ Image Features (Unchanged - It works well) """
    features = {}
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: return None

        features['img_mean'] = np.mean(img)
        features['img_std'] = np.std(img)
        features['img_skew'] = skew(img.flatten())
        
        lsb_plane = img & 1
        counts = np.bincount(lsb_plane.flatten(), minlength=2)
        prob = counts / (np.sum(counts) + 1e-10)
        features['img_lsb_entropy'] = entropy(prob, base=2)
        
        diff_h = np.abs(img[:, :-1] - img[:, 1:])
        diff_v = np.abs(img[:-1, :] - img[1:, :])
        features['img_apd_mean'] = (np.mean(diff_h) + np.mean(diff_v)) / 2.0
        
        edges = cv2.Canny(img, 100, 200)
        features['img_edge_density'] = np.mean(edges) / 255.0

        return features
    except Exception as e:
        print(f"⚠️ Image Error: {e}")
        return None

def calculate_audio_features(aud_path):
    """ 
    HYBRID AUDIO ANALYSIS V3 - CORRELATION
    Added 'aud_lsb_corr' to detect randomness continuity.
    """
    features = {}
    try:
        # --- PART 1: RAW BIT ANALYSIS ---
        data_int, samplerate = sf.read(str(aud_path), dtype='int16')
        
        if len(data_int.shape) > 1:
            data_int = np.mean(data_int, axis=1).astype(np.int16)
            
        # 1.1 LSB Extraction
        lsb = data_int & 1
        
        # 1.2 LSB Autocorrelation (THE GAME CHANGER)
        # Natural audio LSB has slight correlation. Stego LSB has near 0.
        if len(lsb) > 1:
            # Compare bit sequence with shifted version of itself
            corr = np.corrcoef(lsb[:-1], lsb[1:])[0, 1]
            features['aud_lsb_corr'] = 0 if np.isnan(corr) else corr
        else:
            features['aud_lsb_corr'] = 0

        # 1.3 Transition Rate (How often 0->1 or 1->0)
        # Stego maximizes this (randomness). Natural audio might be lower.
        transitions = np.sum(np.abs(np.diff(lsb)))
        features['aud_lsb_trans'] = transitions / len(lsb)

        # 1.4 Standard LSB Entropy
        counts = np.bincount(lsb.flatten(), minlength=2)
        prob = counts / (np.sum(counts) + 1e-10)
        features['aud_lsb_entropy'] = entropy(prob, base=2)
        
        features['aud_kurtosis'] = kurtosis(data_int)
        
        # --- PART 2: SPECTRAL ANALYSIS ---
        y, sr = librosa.load(str(aud_path), sr=22050)
        if len(y) < 512: y = np.pad(y, (0, 512 - len(y)))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['aud_mfcc_mean'] = np.mean(mfcc)
        features['aud_spec_flatness'] = np.mean(librosa.feature.spectral_flatness(y=y))
        features['aud_spec_cent_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['aud_rms_mean'] = np.mean(librosa.feature.rms(y=y))

        return features
    except Exception as e:
        print(f"Audio Error ({Path(aud_path).name}): {e}")
        return None

def main():
    print("Starting V3 Feature Extraction (Correlation Boost)...")
    
    if not DATASET_MAP.exists():
        print("Error: dataset_map.csv not found!")
        return

    df = pd.read_csv(DATASET_MAP)
    extracted_data = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting"):
        filepath = Path(row['filepath'])
        filetype = row['type']
        label = row['label']
        
        if not filepath.exists(): continue
        
        feat = None
        if 'image' in str(filetype):
            feat = calculate_image_features(filepath)
            if feat:
                # Zero out audio features
                for k in ['aud_lsb_entropy', 'aud_lsb_corr', 'aud_lsb_trans', 'aud_kurtosis', 
                          'aud_mfcc_mean', 'aud_spec_flatness', 'aud_spec_cent_mean', 'aud_rms_mean']:
                    feat[k] = 0
                    
        elif 'audio' in str(filetype):
            feat = calculate_audio_features(filepath)
            if feat:
                # Zero out image features
                for k in ['img_mean', 'img_std', 'img_skew', 'img_lsb_entropy', 
                          'img_apd_mean', 'img_edge_density']:
                    feat[k] = 0
                filetype = 'audio'

        if feat:
            feat['label'] = label
            feat['type'] = filetype
            feat['filename'] = row['filename']
            extracted_data.append(feat)

    if extracted_data:
        df_feat = pd.DataFrame(extracted_data)
        df_feat = df_feat.fillna(0)
        df_feat.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Extraction Complete! Saved to: {OUTPUT_CSV}")
    else:
        print("⚠️ No features extracted.")

if __name__ == "__main__":
    main()