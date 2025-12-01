#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from pathlib import Path
from collections import defaultdict, Counter
import time

########################################
# 1. KONFIGURACE (MUSÍ BÝT STEJNÁ JAKO U GENERÁTORU!)
########################################
PEAK_THRESHOLD_DB = 10
FAN_VALUE = 15
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

# Parametry spektrogramu
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

########################################
# 2. FUNKCE PRO EXTRAKCI (Zkopírováno z generátoru)
########################################

def load_audio(path):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        return y
    except Exception as e:
        print(f"[ERR] Chyba načítání {path}: {e}")
        return None

def compute_spectrogram_and_peaks(y):
    # 1. Mel Spektrogram
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag = np.abs(S)**2
    mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=80, fmax=8000)
    mel_db = librosa.power_to_db(np.dot(mel_basis, S_mag) + 1e-9, ref=np.max)

    # 2. Peak Detection
    struct_elem = np.ones((10, 10), dtype=bool)
    local_max = maximum_filter(mel_db, footprint=struct_elem)
    background = mel_db.mean() + PEAK_THRESHOLD_DB
    detected = (mel_db == local_max) & (mel_db > background)
    
    peaks = list(zip(*np.where(detected))) # (freq, time)
    peaks.sort(key=lambda x: x[1])
    return peaks

def generate_query_hashes(peaks):
    hashes = [] # List of (hash, time_offset)
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
# 3. DATABÁZE (NAČTENÍ JSONŮ)
########################################

def load_database(json_dir):
    print(f"[DB] Načítám databázi z: {json_dir}")
    database = defaultdict(list)
    songs_info = {} # id -> info

    json_files = list(Path(json_dir).glob("*_fingerprints.json"))
    
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            media_id = data['media_id']
            songs_info[media_id] = data
            
            # Indexování: Hash -> (SongID, AbsolutníČas)
            for item in data['fingerprints']:
                h = item['hash']
                offset = item['offset']
                database[h].append( (media_id, offset) )
    
    print(f"[DB] Načteno {len(songs_info)} skladeb, {len(database)} unikátních hashů.")
    return database, songs_info

########################################
# 4. VYHLEDÁVACÍ JÁDRO (MATCHING)
########################################

def find_matches(query_hashes, database):
    # Mapa: SongID -> Seznam časových posunů (Deltas)
    # Příklad: 'zaznam_1': [500, 501, 500, 500, 1200, 500...]
    matches_per_song = defaultdict(list)

    for q_hash, q_offset in query_hashes:
        if q_hash in database:
            for db_song_id, db_offset in database[q_hash]:
                # KLÍČOVÝ VÝPOČET:
                # Kde v originále začíná tento zvuk, pokud by to byla shoda?
                offset_delta = db_offset - q_offset
                matches_per_song[db_song_id].append(offset_delta)
    
    return matches_per_song

def score_matches(matches_per_song):
    results = []
    
    for song_id, deltas in matches_per_song.items():
        # Najdeme nejčastější časový posun (Histogram)
        # Counter nám řekne: posun 500 se vyskytl 50x, posun 1200 jen 1x.
        c = Counter(deltas)
        best_delta, score = c.most_common(1)[0]
        
        # Score = kolik hashů "hlasovalo" pro tento časový začátek
        results.append({
            "song_id": song_id,
            "score": score,
            "start_frame": best_delta,
            "start_time_sec": (best_delta * HOP_LENGTH) / SR
        })
        
    # Seřadíme podle skóre (nejvyšší první)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

########################################
# MAIN
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_file", help="Cesta k audio souboru, který hledáme")
    parser.add_argument("--db_dir", default="G:/SHAZAM/OUTPUT", help="Složka s JSON fingerprinty")
    args = parser.parse_args()

    # 1. Načíst DB do paměti
    db, songs_meta = load_database(args.db_dir)
    if not db:
        print("[ERR] Prázdná databáze.")
        return

    # 2. Zpracovat dotaz (Query)
    print(f"[QUERY] Zpracovávám: {args.query_file}")
    y = load_audio(args.query_file)
    if y is None: return

    peaks = compute_spectrogram_and_peaks(y)
    query_hashes = generate_query_hashes(peaks)
    print(f"[QUERY] Vygenerováno {len(query_hashes)} hashů.")

    # 3. Najít shody
    t0 = time.time()
    raw_matches = find_matches(query_hashes, db)
    results = score_matches(raw_matches)
    dt = time.time() - t0

    # 4. Výpis výsledků
    print(f"\n=== VÝSLEDKY HLEDÁNÍ ({dt:.3f}s) ===")
    
    if not results:
        print("Žádná shoda nenalezena.")
        return

    # Zobrazíme TOP 3 výsledky
    for i, res in enumerate(results[:3]):
        song_id = res['song_id']
        # Výpočet jistoty (kolik % z dotazu se našlo)
        confidence = (res['score'] / len(query_hashes)) * 100
        
        print(f"{i+1}. {song_id}")
        print(f"   Čas v originále: {res['start_time_sec']:.2f}s")
        print(f"   Skóre shody:     {res['score']} (Confidence: {confidence:.1f}%)")
        print("-" * 30)

if __name__ == "__main__":
    main()