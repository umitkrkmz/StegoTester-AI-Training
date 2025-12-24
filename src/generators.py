# src/generators.py
# Robust Steganography Generator Module (AGGRESSIVE MODE)
# Increased payload intensity to help AI detection.

import os
import cv2
import numpy as np
import soundfile as sf
from pathlib import Path

class StegoGenerator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================
    # IMAGE GENERATORS (Unchanged - They work well)
    # ==========================

    def _load_and_normalize(self, img_path):
        try:
            img = cv2.imread(str(img_path))
            if img is None: return None
            
            # Ensure even dimensions
            h, w = img.shape[:2]
            h_new = h - (h % 2)
            w_new = w - (w % 2)
            if h_new != h or w_new != w: img = img[:h_new, :w_new]

            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
            return img
        except Exception as e:
            print(f"Load Error ({Path(img_path).name}): {e}")
            return None

    def gen_image_lsb(self, img_path):
        img = self._load_and_normalize(img_path)
        if img is None: return None
        try:
            noise = np.random.randint(0, 2, img.shape, dtype=np.uint8)
            img_stego = (img & 0xFE) | noise
            save_name = f"stego_lsb_{Path(img_path).stem}.png"
            save_path = self.output_dir / "clean_images" / "stego" / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), img_stego)
            return str(save_path)
        except Exception as e: return None

    def gen_image_dct(self, img_path, scaling=0.01):
        img = self._load_and_normalize(img_path)
        if img is None: return None
        try:
            img_float = (img.astype(np.float32) / 255.0).astype(np.float32)
            img_yuv = cv2.cvtColor(img_float, cv2.COLOR_BGR2YCrCb)
            y, u, v = img_yuv[:, :, 0], img_yuv[:, :, 1], img_yuv[:, :, 2]
            dct_coef = cv2.dct(y)
            noise = np.random.normal(0, scaling, dct_coef.shape).astype(np.float32)
            dct_stego = dct_coef + noise
            y_stego = cv2.idct(dct_stego)
            merged_yuv = np.stack([y_stego, u, v], axis=2)
            rgb = cv2.cvtColor(merged_yuv, cv2.COLOR_YCrCb2BGR) * 255.0
            img_stego = np.clip(rgb, 0, 255).astype(np.uint8)
            save_name = f"stego_dct_{Path(img_path).stem}.png"
            save_path = self.output_dir / "clean_images" / "stego" / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), img_stego)
            return str(save_path)
        except Exception as e: return None

    def gen_image_spread(self, img_path, alpha=0.005):
        img = self._load_and_normalize(img_path)
        if img is None: return None
        try:
            row, col, ch = img.shape
            gauss = np.random.normal(0, 5, (row, col, ch))
            noisy = img + img * alpha * gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            save_name = f"stego_ssis_{Path(img_path).stem}.png"
            save_path = self.output_dir / "clean_images" / "stego" / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), noisy)
            return str(save_path)
        except Exception as e: return None

    # ==========================
    # AUDIO GENERATORS (AGGRESSIVE MODE V2)
    # ==========================

    def gen_audio_lsb(self, aud_path):
        """
        MODIFIED: Changes last 2 BITS (Bitmask 3) instead of 1.
        This increases entropy difference significantly.
        """
        try:
            data, samplerate = sf.read(str(aud_path), dtype='int16')
            
            # Generate 2-bit noise (Values 0, 1, 2, 3)
            if len(data.shape) > 1: 
                noise = np.random.randint(0, 4, data.shape, dtype='int16')
            else: 
                noise = np.random.randint(0, 4, len(data), dtype='int16')

            # Clear last 2 bits (~3) and add noise
            stego_data = (data & ~3) | noise
            
            save_name = f"stego_lsb_{Path(aud_path).name}"
            save_path = self.output_dir / "clean_audio" / "stego" / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Force PCM_16 to prevent float conversion data loss
            sf.write(str(save_path), stego_data, samplerate, subtype='PCM_16')
            return str(save_path)
        except Exception as e:
            print(f"⚠️ Error Audio LSB ({Path(aud_path).name}): {e}")
            return None

    def gen_audio_echo(self, aud_path, delay_ms=50, decay=0.6):
        """
        MODIFIED: Stronger Echo (50ms delay, 0.6 decay)
        """
        try:
            data, samplerate = sf.read(str(aud_path))
            delay_samples = int(samplerate * (delay_ms / 1000.0))
            if len(data) <= delay_samples: return None
            
            echo_signal = np.zeros_like(data)
            echo_signal[delay_samples:] = data[:-delay_samples]
            
            stego_data = data + (decay * echo_signal)
            
            max_val = np.max(np.abs(stego_data))
            if max_val > 0.99: stego_data = stego_data / max_val * 0.99
            
            save_name = f"stego_echo_{Path(aud_path).name}"
            save_path = self.output_dir / "clean_audio" / "stego" / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(save_path), stego_data, samplerate)
            return str(save_path)
        except Exception as e:
            print(f"Error Audio Echo ({Path(aud_path).name}): {e}")
            return None