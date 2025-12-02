# indexer.py
#!/usr/bin/env python3

import argparse
import shutil
import time
import sqlite3
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# IMPORT Z AUDIO CORE
from audio_core import (
    load_audio, compute_mel_spectrogram, get_2d_peaks, generate_hashes,
    PEAK_THRESHOLD_DB
)

# KONFIGURACE POUZE PRO INDEXER
SPECTROGRAM_TILE_SEC = 10   
FIG_SIZE = (13, 5)
DEFAULT_SPECTROGRAM_BASE = Path("G:/SHAZAM/SPECTROGRAM")
DEFAULT_DB_PATH = Path("G:/SHAZAM/indexer.db")

class SoundDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup()

    def setup(self):
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
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS hashes (
                hash TEXT NOT NULL,
                sound_id INTEGER NOT NULL,
                offset INTEGER NOT NULL,
                FOREIGN KEY(sound_id) REFERENCES sounds(id)
            )
        """)
        # DŮLEŽITÉ: Index pro rychlé hledání
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON hashes(hash);")
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
        if not hash_list: return
        data_to_insert = [(h, sound_id, o) for (h, o) in hash_list]
        self.cursor.executemany("""
            INSERT INTO hashes (hash, sound_id, offset) VALUES (?, ?, ?)
        """, data_to_insert)
        self.conn.commit()

    def close(self):
        self.conn.close()

# --- VIZUALIZACE ---
def save_tiled_spectrogram(mel_db, sr, hop_length, media_id, output_base, tile_duration=10):
    media_dir = Path(output_base) / media_id
    if media_dir.exists(): shutil.rmtree(media_dir)
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
        plt.title(f"{media_id} | {start_time:.1f}s - {end_time:.1f}s")
        plt.tight_layout()
        plt.savefig(media_dir / f"{media_id}_{i+1:03d}.png", dpi=100, bbox_inches='tight')
        plt.close()

def save_debug_peaks(mel_db, peaks, sr, hop_length, media_id, spectrogram_dir):
    n_mels_total, n_frames_total = mel_db.shape
    fig = plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()
    
    # Zobraz jen prvních 30 sekund pro debug
    limit_frames = min(n_frames_total, int(30 * sr / hop_length))
    actual_duration_sec = limit_frames * hop_length / sr
    region = mel_db[:, :limit_frames]
    
    img = ax.imshow(region, aspect='auto', origin='lower', cmap='gray_r',
                    extent=[0, actual_duration_sec, 0, n_mels_total])
    
    region_peaks = [p for p in peaks if p[1] < limit_frames]
    if region_peaks:
        py, px_frames = zip(*region_peaks)
        px_seconds = [f * hop_length / sr for f in px_frames]
        ax.scatter(px_seconds, py, c='red', s=5, alpha=0.7)
    
    plt.title(f"Peaks Debug (Threshold={PEAK_THRESHOLD_DB}) - {media_id}")
    plt.tight_layout()
    plt.savefig(Path(spectrogram_dir) / media_id / "debug_peaks.png", dpi=100, bbox_inches='tight')
    plt.close()

def process_file_to_db(path, db_handler, spectrogram_dir, args):
    filename = path.name
    media_id = path.stem.replace(" ", "_")
    
    y, sr = load_audio(path, target_sr=args.sr)
    if y is None: return None

    mel_db = compute_mel_spectrogram(y, sr, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels)
    duration_sec = (mel_db.shape[1] * args.hop_length) / sr

    sound_db_id = db_handler.insert_sound(filename, media_id, duration_sec, sr)
    if sound_db_id is None: return None

    save_tiled_spectrogram(mel_db, sr, args.hop_length, media_id, spectrogram_dir, tile_duration=SPECTROGRAM_TILE_SEC)
    
    peaks = get_2d_peaks(mel_db, amp_min=PEAK_THRESHOLD_DB)
    if len(peaks) > 0:
        save_debug_peaks(mel_db, peaks, sr, args.hop_length, media_id, spectrogram_dir)
    
    fingerprints = generate_hashes(peaks)
    db_handler.insert_hashes(sound_db_id, fingerprints)
    
    return {"media_id": media_id, "hashes": len(fingerprints)}

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