#!/usr/bin/env python3

import sqlite3
import argparse
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
import time
from pathlib import Path
from collections import defaultdict
import warnings
import traceback

warnings.filterwarnings("ignore")

########################################
# CONFIG
########################################

PEAK_THRESHOLD_DB = 10
FAN_VALUE = 15
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

DEFAULT_DB_PATH = Path("G:/SHAZAM/indexer.db")
DEFAULT_INPUT_DIR = Path("MATCHER") 

########################################
# 1. DSP FUNKCE
########################################

def load_audio_chunk(path, duration=None):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True, duration=duration)
        return y
    except Exception as e:
        # print(f"   [DEBUG] Chyba: {e}") 
        return None

def compute_mel_spectrogram(y):
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_mag = np.abs(S)**2
    mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=80, fmax=8000)
    mel_S = np.dot(mel_basis, S_mag)
    mel_db = librosa.power_to_db(mel_S + 1e-9, ref=np.max)
    return mel_db

def get_2d_peaks(mel_db):
    struct_elem = np.ones((10, 10), dtype=bool)
    local_max = maximum_filter(mel_db, footprint=struct_elem)
    background_threshold = mel_db.mean() + PEAK_THRESHOLD_DB
    detected_peaks = (mel_db == local_max) & (mel_db > background_threshold)
    freq_indices, time_indices = np.where(detected_peaks)
    peaks = list(zip(freq_indices, time_indices))
    peaks.sort(key=lambda x: x[1])
    return peaks

def generate_hashes(peaks):
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
# 2. MATCHING JÁDRO
########################################

def query_database(hashes, db_path):
    """Vrátí všechny surové shody z celé databáze (INPUT složky)."""
    if not hashes: return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    hash_list = [h[0] for h in hashes]
    
    sample_hash_offsets = defaultdict(list)
    for h, offset in hashes:
        sample_hash_offsets[h].append(offset)

    BATCH_SIZE = 900
    matches = [] 

    # Hledáme shody ve všech nahrávkách najednou
    for i in range(0, len(hash_list), BATCH_SIZE):
        batch = hash_list[i:i+BATCH_SIZE]
        placeholders = ','.join(['?'] * len(batch))
        
        sql = f"""
            SELECT h.hash, h.sound_id, h.offset, s.media_id 
            FROM hashes h
            JOIN sounds s ON h.sound_id = s.id
            WHERE h.hash IN ({placeholders})
        """
        
        cursor.execute(sql, batch)
        results = cursor.fetchall()
        
        for r_hash, r_sound_id, r_db_offset, r_media_id in results:
            if r_hash in sample_hash_offsets:
                for sample_offset in sample_hash_offsets[r_hash]:
                    matches.append({
                        'sound_id': r_sound_id,
                        'media_id': r_media_id,
                        'db_offset': r_db_offset,
                        'sample_offset': sample_offset
                    })
    conn.close()
    return matches

def align_matches(matches):
    """Seřadí kandidáty podle kvality časového zákrytu."""
    # sound_id -> { diff -> count }
    candidates = defaultdict(lambda: defaultdict(int))
    candidates_info = {} 

    # Procházíme shody ze všech souborů z INPUTu
    for m in matches:
        diff = m['db_offset'] - m['sample_offset']
        sid = m['sound_id']
        candidates[sid][diff] += 1
        candidates_info[sid] = m['media_id']

    results = []
    for sid, diffs in candidates.items():
        # Pro každý soubor najdeme nejlepší časový posun
        best_diff, score = max(diffs.items(), key=lambda x: x[1])
        results.append({
            'sound_id': sid,
            'media_id': candidates_info[sid],
            'score': score,
            'offset_seconds': (best_diff * HOP_LENGTH) / SR
        })

    # Seřadíme sestupně podle skóre
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

########################################
# 3. ZPRACOVÁNÍ
########################################

def process_single_file(filepath, db_path, limit_sec=None):
    print(f"\n🎧 ANALÝZA: {filepath.name}")
    t0 = time.time()
    
    y = load_audio_chunk(filepath, duration=limit_sec)
    if y is None:
        print("   ❌ Chyba: Soubor nelze přečíst.")
        return "error"

    # Generování hashů
    mel = compute_mel_spectrogram(y)
    peaks = get_2d_peaks(mel)
    sample_hashes = generate_hashes(peaks)
    
    if len(sample_hashes) == 0:
        print("   ❌ Chyba: Žádný signál (ticho).")
        return "no_signal"

    # Dotaz na celou databázi
    raw_matches = query_database(sample_hashes, db_path)
    
    # Seřazení výsledků
    final_results = align_matches(raw_matches)
    dt = time.time() - t0

    if not final_results:
        print(f"   ❌ Žádná shoda v databázi. ({dt:.2f}s)")
        return "no_match"

    # --- VÝPIS ŽEBŘÍČKU (RANKING) ---
    # Zde vidíš srovnání se všemi kandidáty z INPUTu
    
    top_match = final_results[0]
    top_confidence = (top_match['score'] / len(sample_hashes)) * 100
    
    # Kolik výsledků zobrazit? (Např. top 5 nebo ty co mají aspoň malé skóre)
    print(f"   ⏱️  Zpracováno za {dt:.2f}s | Celkem hashů v dotazu: {len(sample_hashes)}")
    print(f"   📊 NALEZENÍ KANDIDÁTI (z celé DB):")
    
    matched_any = False
    
    # Zobrazíme max 5 nejlepších, pokud mají skóre > 2
    for i, res in enumerate(final_results[:5]):
        score = res['score']
        conf = (score / len(sample_hashes)) * 100
        
        # Ikonka podle síly shody
        if conf > 5.0:
            icon = "✅"
            matched_any = True
        elif conf > 1.0:
            icon = "❓"
        else:
            icon = "  " # Slabá shoda
            
        # Pokud je skóre příliš malé, přestaň vypisovat (šum)
        if score < 2: 
            break
            
        print(f"      {i+1}. {icon} {res['media_id']:<30} | Skóre: {score:>4} | Jistota: {conf:>5.1f}% | Čas: {res['offset_seconds']:>6.2f}s")

    if not matched_any:
        return "no_match"
    return "match"

########################################
# 4. MAIN
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR), help="Složka s nahrávkami k porovnání")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Vytvořena složka: {input_dir}. Vložte soubory.")
        return

    files = [f for f in input_dir.rglob("*") if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']]
    
    if not files:
        print(f"[INFO] Složka {input_dir} je prázdná.")
        return

    print(f"=== HROMADNÉ SROVNÁNÍ ({len(files)} souborů) ===")
    print(f"Databáze (INPUT): {args.db}")
    print("=" * 60)
    
    stats = defaultdict(int)
    
    for f in files:
        try:
            res = process_single_file(f, args.db, args.limit)
            stats[res] += 1
            print("-" * 60)
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            stats["error"] += 1

    print("\n=== SOUHRN ===")
    print(f"Počet úspěšných identifikací: {stats['match']}")
    print(f"Nenalezeno / Slabá shoda:     {stats['no_match']}")
    print(f"Chyby souborů:                {stats['error'] + stats['no_signal']}")

if __name__ == "__main__":
    main()