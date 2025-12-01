#!/usr/bin/env python3

import subprocess
import argparse
import os
import shutil
import time
import sqlite3
import math
import csv
from pathlib import Path

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
DEFAULT_DB_PATH = Path("G:/SHAZAM/indexer.db")

########################################
# 1. DATABASE HANDLER
########################################

class SoundDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup()

    def setup(self):
        # Tabulka pro zvuky (metadata)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                media_id TEXT,
                duration REAL,
                sample_rate INTEGER,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tabulka pro hashe (fingerprints)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS hashes (
                hash TEXT NOT NULL,
                sound_id INTEGER NOT NULL,
                offset INTEGER NOT NULL,
                FOREIGN KEY(sound_id) REFERENCES sounds(id)
            )
        """)

        # INDEX pro rychlé vyhledávání
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_hash ON hashes(hash);
        """)
        self.conn.commit()

    def insert_sound(self, filename, media_id, duration, sr):
        try:
            self.cursor.execute("""
                INSERT INTO sounds (filename, media_id, duration, sample_rate)
                VALUES (?, ?, ?, ?)
            """, (filename, media_id, duration, sr))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            print(f"[DB] Soubor {filename} již v databázi existuje.")
            return None

    def insert_hashes(self, sound_id, hash_list):
        if not hash_list:
            return
        # Data: (hash, sound_id, offset)
        data_to_insert = [(h, sound_id, o) for (h, o) in hash_list]
        self.cursor.executemany("""
            INSERT INTO hashes (hash, sound_id, offset) VALUES (?, ?, ?)
        """, data_to_insert)
        self.conn.commit()

    def close(self):
        self.conn.close()

########################################
# 2. VIZUALIZACE
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
        # Zde ponecháváme aspect='auto', aby se tile roztáhl, jak je zvykem u dlaždic
        img = ax.imshow(chunk, aspect='auto', origin='lower', cmap='magma', 
                        vmin=-80, vmax=0,
                        extent=[start_time, end_time, 0, n_mels]) 
        
        plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04).set_label('Amplitude (dB)')
        
        ticks_x = np.linspace(start_time, end_time, num=6)
        ax.set_xticks(ticks_x)
        ax.set_xticklabels([f"{t:.1f}" for t in ticks_x])

        ticks_y = np.linspace(0, n_mels, num=5)
        ax.set_yticks(ticks_y)
        ax.set_yticklabels([f"{int(t)}" for t in ticks_y])

        plt.title(f"{media_id} | Tile {i+1}")
        plt.ylabel("Mel Band Index")
        plt.xlabel("Time (seconds)")
        plt.tight_layout()
        
        out_path = media_dir / f"{media_id}_tile_{i+1:03d}.png"
        plt.savefig(out_path, dpi=100, bbox_inches='tight')
        plt.close()

def save_fixed_spectrogram(mel_db, sr, hop_length, media_id, spectrogram_dir, fixed_duration=10):
    """
    ### NOVÁ FUNKCE ###
    Vytvoří spektogram, který má VŽDY šířku odpovídající 'fixed_duration' (např. 10s).
    Pokud je zvuk kratší, zbytek se doplní černou (-80dB).
    Pokud je zvuk delší, ořízne se na prvních 10s.
    """
    media_dir = Path(spectrogram_dir) / media_id
    media_dir.mkdir(parents=True, exist_ok=True) # Pro jistotu, kdyby neexistovalo

    n_mels, n_frames = mel_db.shape
    
    # Kolik framů odpovídá přesně 10 sekundám?
    target_frames = int(fixed_duration * sr / hop_length)
    
    # Vytvoříme prázdné plátno (canvas) vyplněné hodnotou ticha (-80 dB)
    padded_mel = np.full((n_mels, target_frames), -80.0)
    
    # Zkopírujeme data: vezmeme buď vše (pokud je kratší), nebo jen začátek (pokud je delší)
    frames_to_copy = min(n_frames, target_frames)
    padded_mel[:, :frames_to_copy] = mel_db[:, :frames_to_copy]

    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()
    
    # Vykreslíme padded_mel. Extent je nyní fixní 0 až 10s.
    img = ax.imshow(padded_mel, aspect='auto', origin='lower', cmap='magma', 
                    vmin=-80, vmax=0,
                    extent=[0, fixed_duration, 0, n_mels]) 
    
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04).set_label('Amplitude (dB)')
    
    # Osy X nastavíme fixně na 0-10s
    ticks_x = np.linspace(0, fixed_duration, num=11) # po 1 sekundě
    ax.set_xticks(ticks_x)
    ax.set_xticklabels([f"{int(t)}" for t in ticks_x])

    ticks_y = np.linspace(0, n_mels, num=5)
    ax.set_yticks(ticks_y)
    ax.set_yticklabels([f"{int(t)}" for t in ticks_y])

    plt.title(f"{media_id} | FIXED {fixed_duration}s VIEW")
    plt.ylabel("Mel Band Index")
    plt.xlabel("Time (seconds)")
    plt.tight_layout()
    
    # Uložíme jako speciální soubor, např. _fixed_10s.png
    out_path = media_dir / f"{media_id}_FIXED_10s.png"
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()

def save_debug_peaks(mel_db, peaks, sr, hop_length, media_id, spectrogram_dir):
    """
    Uloží vizualizaci s červenými tečkami (debug peaks).
    """
    n_mels_total, n_frames_total = mel_db.shape
    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()
    
    debug_duration_target = 30
    limit_frames = min(n_frames_total, int(debug_duration_target * sr / hop_length))
    actual_duration_sec = limit_frames * hop_length / sr

    region = mel_db[:, :limit_frames]
    
    img = ax.imshow(region, aspect='auto', origin='lower', cmap='gray_r',
                    extent=[0, actual_duration_sec, 0, n_mels_total])
    
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Amplitude (dB)')

    region_peaks = [p for p in peaks if p[1] < limit_frames]
    if region_peaks:
        py, px_frames = zip(*region_peaks)
        px_seconds = [f * hop_length / sr for f in px_frames]
        ax.scatter(px_seconds, py, c='red', s=5, alpha=0.7)
    
    ticks_x = np.linspace(0, actual_duration_sec, num=6)
    ax.set_xticks(ticks_x)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks_x])

    ticks_y = np.linspace(0, n_mels_total, num=5)
    ax.set_yticks(ticks_y)
    ax.set_yticklabels([f"{int(t)}" for t in ticks_y])

    plt.ylabel("Mel Band Index")
    plt.xlabel("Time (seconds)")
    plt.title(f"Peaks Debug - {media_id}")
    plt.tight_layout()
    
    debug_path = Path(spectrogram_dir) / media_id / "debug_peaks.png"
    plt.savefig(debug_path, dpi=100, bbox_inches='tight')
    plt.close()

########################################
# 3. DSP & HASHING
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

def get_2d_peaks(mel_db, amp_min=PEAK_THRESHOLD_DB):
    struct_elem = np.ones((10, 10), dtype=bool)
    local_max = maximum_filter(mel_db, footprint=struct_elem)
    background_threshold = mel_db.mean() + amp_min 
    detected_peaks = (mel_db == local_max) & (mel_db > background_threshold)
    freq_indices, time_indices = np.where(detected_peaks)
    peaks = list(zip(freq_indices, time_indices))
    peaks.sort(key=lambda x: x[1])
    return peaks

def generate_hashes(peaks):
    """
    Vrací seznam (hash_key, time_offset)
    """
    hashes = []
    for i in range(len(peaks)):
        freq1, time1 = peaks[i]
        for j in range(1, FAN_VALUE + 1):
            if (i + j) < len(peaks):
                freq2, time2 = peaks[i + j]
                time_delta = time2 - time1
                if MIN_HASH_TIME_DELTA <= time_delta <= MAX_HASH_TIME_DELTA:
                    h_key = f"{freq1}:{freq2}:{time_delta}"
                    hashes.append((h_key, int(time1)))
    return hashes

########################################
# 4. MAIN PROCESSING
########################################

def process_file_to_db(path, db_handler, spectrogram_dir, args):
    start_time_proc = time.time()
    filename = path.name
    media_id = path.stem.replace(" ", "_")
    
    # 1. Load Audio
    y, sr = load_audio(path, target_sr=args.sr)
    if y is None: return None

    # 2. Spectrogram
    mel_db = compute_mel_spectrogram(y, sr, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels)
    n_frames_total = mel_db.shape[1]
    duration_sec = (n_frames_total * args.hop_length) / sr

    # 3. Uložení Metadata do DB
    sound_db_id = db_handler.insert_sound(filename, media_id, duration_sec, sr)
    if sound_db_id is None:
        return None

    # 4a. Vizualizace (Klasické dlaždice - zachováno)
    save_tiled_spectrogram(mel_db, sr, args.hop_length, media_id, spectrogram_dir, tile_duration=SPECTROGRAM_TILE_SEC)
    
    # 4b. Vizualizace (NOVÉ: Fixní 10s pohled pro porovnání)
    # Tato funkce vytvoří obrázek, který má vždy osu X 0-10s. Krátké soubory se neroztáhnou.
    save_fixed_spectrogram(mel_db, sr, args.hop_length, media_id, spectrogram_dir, fixed_duration=10)

    # 5. Peaks & Debug Vizualizace (Zachováno na přání)
    peaks = get_2d_peaks(mel_db, amp_min=PEAK_THRESHOLD_DB)
    if len(peaks) > 0:
        save_debug_peaks(mel_db, peaks, sr, args.hop_length, media_id, spectrogram_dir)
    
    # 6. Hashes -> DB
    fingerprints = generate_hashes(peaks)
    db_handler.insert_hashes(sound_db_id, fingerprints)
    
    return {
        "media_id": media_id,
        "hashes": len(fingerprints)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./INPUT")
    parser.add_argument("--db_path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--spectrograms", default=str(DEFAULT_SPECTROGRAM_BASE))
    
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=128)
    
    args = parser.parse_args()
    
    in_dir = Path(args.input)
    spec_dir = Path(args.spectrograms)
    db_path = Path(args.db_path)
    
    spec_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Připojuji k databázi: {db_path}")
    db = SoundDatabase(db_path)

    files = [f for f in in_dir.rglob("*") if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']]
    print(f"Startuji indexaci {len(files)} souborů...")
    
    total_hashes = 0
    for f in tqdm(files):
        stats = process_file_to_db(f, db, spec_dir, args)
        if stats:
            total_hashes += stats["hashes"]
            
    db.close()
    print("="*40)
    print(f"HOTOVO. Uloženo hashů: {total_hashes}")

if __name__ == "__main__":
    main()