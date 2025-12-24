# src/slice_audio.py
# Helper script to slice audio files into smaller chunks for Data Augmentation.

import os
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
INPUT_DIR = Path("assets/clean_audio")
CHUNK_DURATION = 3.0 # Seconds

def slice_audio_files():
    print(f"🔪 Slicing audio files in {INPUT_DIR} into {CHUNK_DURATION}s chunks...")
    
    files = list(INPUT_DIR.glob("*.wav")) + list(INPUT_DIR.glob("*.flac"))
    
    for file_path in tqdm(files):
        try:
            data, sr = sf.read(str(file_path))
            
            # Calculate chunk size in samples
            chunk_samples = int(sr * CHUNK_DURATION)
            total_samples = len(data)
            
            # Skip if file is too short
            if total_samples < chunk_samples:
                continue
            
            # Slice and save
            num_chunks = total_samples // chunk_samples
            
            for i in range(num_chunks):
                start = i * chunk_samples
                end = start + chunk_samples
                chunk_data = data[start:end]
                
                # New filename: original_chunk0.wav, original_chunk1.wav
                new_filename = f"{file_path.stem}_chunk{i}{file_path.suffix}"
                sf.write(str(INPUT_DIR / new_filename), chunk_data, sr)
            
            # Optional: Remove the original large file to prevent duplication
            os.remove(file_path)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print("✅ Slicing Complete! Dataset multiplied.")

if __name__ == "__main__":
    slice_audio_files()