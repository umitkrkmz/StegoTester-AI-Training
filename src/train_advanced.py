# src/train_advanced.py
# "Model Arena" - Trains multiple algorithms and picks the best one.
# Contenders: Random Forest, Gradient Boosting, SVM, MLP (Neural Net), KNN.

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- CONTENDERS (Modeller) ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# --- CONFIG ---
FEATURES_CSV = Path("dataset/features.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Define feature columns (Must match extract_features.py output)
IMG_FEATURES = ['img_mean', 'img_std', 'img_lsb_entropy', 'img_apd_mean', 'img_edge_density', 'img_skew']
# Hybrid Audio Features
AUD_FEATURES = [
    'aud_lsb_entropy', 
    'aud_lsb_corr',    # <--- YENİ (Game Changer)
    'aud_lsb_trans',   # <--- YENİ
    'aud_kurtosis', 
    'aud_spec_flatness', 
    'aud_mfcc_mean', 
    'aud_spec_cent_mean', 
    'aud_rms_mean'
]

def get_models():
    """Returns a dictionary of models to test."""
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradBoosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42), # Probability needed for confidence score
        "NeuralNet": MLPClassifier(max_iter=500, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

def train_and_evaluate(df, model_type, feature_cols):
    print(f"\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")
    print(f"🥊 ARENA: Finding Best Model for {model_type.upper()}")
    print(f"▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")
    
    # Filter Data
    subset = df[df['type'].astype(str).str.contains(model_type)].copy()
    if len(subset) == 0:
        print(f"⚠️ No data found for {model_type}!")
        return

    X = subset[feature_cols]
    y = subset['label']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- CRITICAL: SCALING ---
    # SVM and Neural Networks fail without scaling. RF doesn't care, but it doesn't hurt.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler (We need this for the UI app later!)
    scaler_path = MODELS_DIR / f"scaler_{model_type}.pkl"
    joblib.dump(scaler, scaler_path)
    
    best_name = None
    best_score = 0.0
    best_model = None

    models = get_models()
    
    for name, clf in models.items():
        print(f"⏳ Training {name}...", end=" ")
        try:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"-> Accuracy: %{acc*100:.2f}")
            
            if acc > best_score:
                best_score = acc
                best_name = name
                best_model = clf
        except Exception as e:
            print(f"❌ Failed: {e}")

    print("-" * 40)
    print(f"🏆 WINNER for {model_type.upper()}: {best_name} with %{best_score*100:.2f}")
    
    # Detailed Report for the Winner
    print(f"\n📊 Detailed Report for {best_name}:")
    y_pred_best = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_best, target_names=['Clean', 'Stego']))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_best)}")

    # Save the Champion
    save_path = MODELS_DIR / f"stego_model_{model_type}.pkl"
    joblib.dump(best_model, save_path)
    print(f"💾 Saved Best Model to: {save_path}")

def main():
    if not FEATURES_CSV.exists():
        print("❌ features.csv not found!")
        return

    # Load Data & Handle NaN
    df = pd.read_csv(FEATURES_CSV)
    df = df.fillna(0) 

    # 1. Run Arena for IMAGE
    train_and_evaluate(df, 'image', IMG_FEATURES)

    # 2. Run Arena for AUDIO
    train_and_evaluate(df, 'audio', AUD_FEATURES)

if __name__ == "__main__":
    main()