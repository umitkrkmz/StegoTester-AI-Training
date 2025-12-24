# main_generator.py
# StegoTester AI - Dataset Generation Factory
# Orchestrates the creation of synthetic steganography datasets.

import os
import csv
import shutil
from pathlib import Path
from src.generators import StegoGenerator
from tqdm import tqdm

# --- CONFIGURATION ---
ASSETS_DIR = Path("assets")
DATASET_DIR = Path("dataset")
CSV_PATH = DATASET_DIR / "dataset_map.csv"

# Supported Formats
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
AUD_EXTS = {".wav", ".flac"}

def main():
    # 1. Clean up previous dataset
    if DATASET_DIR.exists():
        try:
            shutil.rmtree(DATASET_DIR)
            print("Previous dataset cleaned.")
        except PermissionError:
            print("Warning: Could not delete some files (locked). Overwriting...")
    
    # 2. Create Directory Structure
    # Targets: dataset/clean_images/clean, dataset/clean_images/stego, etc.
    target_types = ["clean_images", "clean_audio"] 
    
    for dtype in target_types:
        for label in ["clean", "stego"]:
            (DATASET_DIR / dtype / label).mkdir(parents=True, exist_ok=True)

    gen = StegoGenerator(output_dir=DATASET_DIR)
    
    # Initialize CSV Map
    csv_file = open(CSV_PATH, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "type", "label", "method", "filepath"])

    print("Steganography Dataset Generation Started...")

    # ==========================
    # IMAGE PROCESSING
    # ==========================
    source_img_dir = ASSETS_DIR / "clean_images"
    img_files = [f for f in source_img_dir.glob("*") if f.suffix.lower() in IMG_EXTS]
    
    print(f"\nFound {len(img_files)} Clean Images. Generating variations...")
    
    for img_path in tqdm(img_files, desc="Processing Images"):
        # A) Save Clean Original (Label: 0)
        clean_dest = DATASET_DIR / "clean_images" / "clean" / img_path.name
        shutil.copy(img_path, clean_dest)
        writer.writerow([clean_dest.name, "image", 0, "original", str(clean_dest)])

        # B) Generate Stego Variations (Label: 1)
        
        # 1. LSB
        stego_path = gen.gen_image_lsb(img_path)
        if stego_path: writer.writerow([Path(stego_path).name, "image", 1, "lsb", stego_path])

        # 2. DCT
        stego_path = gen.gen_image_dct(img_path)
        if stego_path: writer.writerow([Path(stego_path).name, "image", 1, "dct", stego_path])

        # 3. Spread Spectrum
        stego_path = gen.gen_image_spread(img_path)
        if stego_path: writer.writerow([Path(stego_path).name, "image", 1, "ssis", stego_path])

    # ==========================
    # AUDIO PROCESSING
    # ==========================
    source_aud_dir = ASSETS_DIR / "clean_audio"
    aud_files = [f for f in source_aud_dir.glob("*") if f.suffix.lower() in AUD_EXTS]
    
    print(f"\nFound {len(aud_files)} Clean Audio Files. Generating variations...")
    
    for aud_path in tqdm(aud_files, desc="Processing Audio"):
        # A) Save Clean Original (Label: 0)
        clean_dest = DATASET_DIR / "clean_audio" / "clean" / aud_path.name
        shutil.copy(aud_path, clean_dest)
        writer.writerow([clean_dest.name, "audio", 0, "original", str(clean_dest)])

        # B) Generate Stego Variations (Label: 1)

        # 1. LSB
        stego_path = gen.gen_audio_lsb(aud_path)
        if stego_path: writer.writerow([Path(stego_path).name, "audio", 1, "lsb", stego_path])

        # 2. Echo Hiding
        stego_path = gen.gen_audio_echo(aud_path)
        if stego_path: writer.writerow([Path(stego_path).name, "audio", 1, "echo", stego_path])

    csv_file.close()
    print(f"\nAll Tasks Completed!")
    print(f"Dataset Location: {DATASET_DIR}")
    print(f"Map File: {CSV_PATH}")

if __name__ == "__main__":
    main()