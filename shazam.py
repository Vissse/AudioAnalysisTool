#!/usr/bin/env python3

########################################
# IMPORTS
########################################

import subprocess
import argparse
import json
import os
import shutil
import time
from pathlib import Path
import math
import csv

import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from tqdm import tqdm

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

########################################
# CONFIG
########################################

PEAK_THRESHOLD_DB = 10      
SPECTROGRAM_TILE_SEC = 10   
FIG_SIZE = (13, 5) 

FAN_VALUE = 15              
MIN_HASH_TIME_DELTA = 0     
MAX_HASH_TIME_DELTA = 200   

DEFAULT_SPECTROGRAM_BASE = Path("G:/SHAZAM/SPECTROGRAM")
DEFAULT_OUTPUT_BASE = Path("G:/SHAZAM/OUTPUT")

########################################
# 1. VIZUALIZACE (DLAŽDICE)
########################################

def save_tiled_spectrogram(mel_db, sr, hop_length, media_id, output_base, tile_duration=10):
    media_dir = Path(output_base) / media_id
    if media_dir.exists():
        shutil.rmtree(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)

    n_mels, n_frames = mel_db.shape
    frames_per_tile = int(tile_duration * sr / hop_length)
    total_tiles = math.ceil(n_frames / frames_per_tile)

    for i in range(total_tiles):
        start_f = i * frames_per_tile
        end_f = min((i + 1) * frames_per_tile, n_frames)
        
        if start_f >= end_f: break

        chunk = mel_db[:, start_f:end_f]
        
        start_time = (start_f * hop_length) / sr
        end_time = (end_f * hop_length) / sr

        fig = plt.figure(figsize=FIG_SIZE)
        ax = plt.gca()
        
        img = ax.imshow(chunk, aspect='auto', origin='lower', cmap='magma', 
                        vmin=-80, vmax=0,
                        extent=[start_time, end_time, 0, n_mels]) 
        
        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Amplitude (dB)')

        ticks_x = np.linspace(start_time, end_time, num=6)
        ax.set_xticks(ticks_x)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks_x])

        ticks_y = np.linspace(0, n_mels, num=5)
        ax.set_yticks(ticks_y)
        ax.set_yticklabels([f"{int(t)}" for t in ticks_y])

        plt.title(f"{media_id} | {start_time:.1f}s - {end_time:.1f}s")
        plt.ylabel("Mel Band Index")
        plt.xlabel("Time (seconds)")
        
        plt.tight_layout()
        out_path = media_dir / f"{media_id}_{i+1:03d}.png"
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()

########################################
# 2. AUDIO LOADING
########################################

def load_audio(path, target_sr=22050):
    path = str(path)
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y, sr
    except:
        try:
            cmd = ["ffmpeg", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "pipe:1"]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            y = np.frombuffer(out, dtype=np.float32)
            return y, target_sr
        except Exception as e:
            print(f"[ERR] Nepodařilo se načíst {path}: {e}")
            return None, None

def compute_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(S)**2
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=80, fmax=8000)
    mel_S = np.dot(mel_basis, S_mag)
    mel_db = librosa.power_to_db(mel_S + 1e-9, ref=np.max)
    return mel_db

########################################
# 3. PEAK DETECTION & HASHING
########################################

def get_2d_peaks(mel_db, amp_min=PEAK_THRESHOLD_DB):
    struct_elem = np.ones((10, 10), dtype=bool)
    local_max = maximum_filter(mel_db, footprint=struct_elem)
    background_threshold = mel_db.mean() + amp_min 
    detected_peaks = (mel_db == local_max) & (mel_db > background_threshold)
    freq_indices, time_indices = np.where(detected_peaks)
    peaks = list(zip(freq_indices, time_indices))
    peaks.sort(key=lambda x: x[1])
    return peaks

def generate_hashes(peaks, media_id):
    hashes = []
    for i in range(len(peaks)):
        freq1, time1 = peaks[i]
        for j in range(1, FAN_VALUE + 1):
            if (i + j) < len(peaks):
                freq2, time2 = peaks[i + j]
                time_delta = time2 - time1
                if MIN_HASH_TIME_DELTA <= time_delta <= MAX_HASH_TIME_DELTA:
                    h_key = f"{freq1}:{freq2}:{time_delta}"
                    hashes.append({
                        "hash": h_key,
                        "offset": int(time1),
                        "id": media_id
                    })
    return hashes

########################################
# 4. PROCESSING LOGIC
########################################

def process_file(path, output_dir, spectrogram_dir, args):
    """
    Vrací slovník (metadata) o zpracovaném souboru, nebo None při chybě.
    """
    start_time_proc = time.time()
    media_id = path.stem.replace(" ", "_")
    
    file_size_mb = path.stat().st_size / (1024 * 1024)

    y, sr = load_audio(path, target_sr=args.sr)
    if y is None: return None

    mel_db = compute_mel_spectrogram(y, sr, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels)
    n_mels_total, n_frames_total = mel_db.shape
    
    duration_sec = (n_frames_total * args.hop_length) / sr

    # 1. Vizualizace (Dlaždice)
    save_tiled_spectrogram(mel_db, sr, args.hop_length, media_id, spectrogram_dir, tile_duration=SPECTROGRAM_TILE_SEC)
    
    # 2. Detekce vrcholů
    peaks = get_2d_peaks(mel_db, amp_min=PEAK_THRESHOLD_DB)
    
    # 3. Debug Vizualizace Peaků (pouze prvních 30s)
    if len(peaks) > 0:
        fig = plt.figure(figsize=FIG_SIZE)
        ax = plt.gca()
        
        debug_duration_target = 30
        limit_frames = min(n_frames_total, int(debug_duration_target * sr / args.hop_length))
        actual_duration_sec = limit_frames * args.hop_length / sr

        region = mel_db[:, :limit_frames]
        # extent=[..., 0, n_mels_total] zajistí, že data sahají až k 128
        img = ax.imshow(region, aspect='auto', origin='lower', cmap='gray_r',
                    extent=[0, actual_duration_sec, 0, n_mels_total])
        
        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Amplitude (dB)')

        region_peaks = [p for p in peaks if p[1] < limit_frames]
        if region_peaks:
            py, px_frames = zip(*region_peaks)
            px_seconds = [f * args.hop_length / sr for f in px_frames]
            ax.scatter(px_seconds, py, c='red', s=5, alpha=0.7)
        
        # --- OSY X ---
        ticks_x = np.linspace(0, actual_duration_sec, num=6)
        ax.set_xticks(ticks_x)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks_x])

        # --- OSY Y (Opraveno: Vynucení 0 až 128) ---
        # Vygeneruje 5 bodů: 0, 32, 64, 96, 128
        ticks_y = np.linspace(0, n_mels_total, num=5)
        ax.set_yticks(ticks_y)
        ax.set_yticklabels([f"{int(t)}" for t in ticks_y])

        # POPISKY OS
        plt.ylabel("Mel Band Index")
        plt.xlabel("Time (seconds)")

        plt.title(f"Peaks Debug - {media_id}")
        plt.tight_layout()
        
        debug_path = Path(spectrogram_dir) / media_id / "debug_peaks.png"
        plt.savefig(debug_path, dpi=100, bbox_inches='tight')
        plt.close()

    # 4. Generování hashů
    fingerprints = generate_hashes(peaks, media_id)
    
    output_path = Path(output_dir) / f"{media_id}_fingerprints.json"
    with open(output_path, 'w') as f:
        json.dump({
            "media_id": media_id,
            "sr": sr,
            "hop_length": args.hop_length,
            "total_hashes": len(fingerprints),
            "fingerprints": fingerprints
        }, f)
        
    proc_time = time.time() - start_time_proc

    # --- METADATA ---
    metadata = {
        "media_id": media_id,
        "filename": path.name,
        "format": path.suffix.lower(),
        "duration_sec": round(duration_sec, 2),
        "file_size_mb": round(file_size_mb, 2),
        "sample_rate": sr,
        "peaks_count": len(peaks),
        "hashes_count": len(fingerprints),
        "processing_time_sec": round(proc_time, 2),
        "output_json": str(output_path.name)
    }

    return metadata

def save_dataset_summary(dataset, output_dir):
    out_dir = Path(output_dir)
    
    json_path = out_dir / "_dataset_metadata.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    if len(dataset) > 0:
        csv_path = out_dir / "_dataset_metadata.csv"
        keys = dataset[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(dataset)
            
    print(f"\n[DATASET] Uložena metadata datasetu do:\n -> {json_path}\n -> {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./INPUT")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--spectrograms", default=str(DEFAULT_SPECTROGRAM_BASE))
    
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=128)
    
    args = parser.parse_args()
    
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    spec_dir = Path(args.spectrograms)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    
    files = [f for f in in_dir.rglob("*") if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']]
    
    print(f"Startuji zpracování {len(files)} souborů...")
    
    dataset_registry = []
    total_hashes = 0
    
    for f in tqdm(files):
        meta = process_file(f, out_dir, spec_dir, args)
        if meta:
            dataset_registry.append(meta)
            total_hashes += meta["hashes_count"]
            
    if dataset_registry:
        save_dataset_summary(dataset_registry, out_dir)
            
    print("="*40)
    print(f"HOTOVO. Celkem vygenerováno {total_hashes} hashů.")

if __name__ == "__main__":
    main()