#!/usr/bin/env python3

import numpy as np
import librosa
import soundfile as sf
import os
from pathlib import Path
import warnings

# Potlačení warningů
warnings.filterwarnings("ignore")

########################################
#  🛠️  UŽIVATELSKÉ NASTAVENÍ
########################################

QUERY_PATH = "G:/SHAZAM/MATCHER/Zaznam_1.m4a"   # Tvoje nahrávka
INPUT_DIR = "INPUT"                # Složka s originály
OUTPUT_DIR = "VYSLEDKY_S_CASEM"    # Kam to uložit
GENERATE_ALL = True                # True = vygenerovat wav pro všechny soubory

########################################
# CONFIG
########################################
SR = 16000
N_MFCC = 13
HOP_LENGTH = 512

def extract_features(path):
    print(".", end="", flush=True)
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y) < 1024: return None, None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        # Normalizace
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-9)
        
        return y, mfcc
    except Exception:
        return None, None

def find_best_snippet(query_mfcc, ref_mfcc):
    """Vrátí skóre a časové okno nejlepší shody."""
    n_query = query_mfcc.shape[1]
    n_ref = ref_mfcc.shape[1]
    
    if n_query > n_ref:
        return float('inf'), 0, 0 

    try:
        D, wp = librosa.sequence.dtw(X=query_mfcc, Y=ref_mfcc, metric='cosine', subseq=True)
        
        min_cost_idx = np.argmin(D[-1, :])
        min_cost = D[-1, min_cost_idx]
        
        path_end = wp[0] 
        path_start = wp[-1] 
        
        ref_start_frame = path_start[1]
        ref_end_frame = path_end[1]
        
        normalized_score = min_cost / len(wp)
        return normalized_score, ref_start_frame, ref_end_frame
        
    except Exception:
        return float('inf'), 0, 0

def save_stereo_result(query_y, ref_y, start_f, end_f, out_path):
    start_sample = start_f * HOP_LENGTH
    end_sample = end_f * HOP_LENGTH
    
    ref_segment = ref_y[start_sample:end_sample]
    min_len = min(len(query_y), len(ref_segment))
    if min_len == 0: return
    
    stereo = np.vstack((query_y[:min_len], ref_segment[:min_len])).T
    sf.write(out_path, stereo, SR)

def main():
    print("="*80)
    print(f"      HLEDAČ PODOBNOSTI (S ČASOVÝM RAZÍTKEM)")
    print("="*80)

    # 1. Načtení Query
    q_path = Path(QUERY_PATH)
    if not q_path.exists():
        print(f"❌ CHYBA: Soubor '{QUERY_PATH}' neexistuje.")
        return
    
    print(f"🎤 Query: {q_path.name}")
    y_query, mfcc_query = extract_features(q_path)
    if y_query is None: return
    print(" ✅ Načteno")

    # 2. Načtení Inputů
    in_dir = Path(INPUT_DIR)
    files = [f for f in in_dir.rglob("*") if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']]
    
    if not files:
        print("❌ Žádné soubory v INPUT.")
        return

    print(f"🔍 Analyzuji {len(files)} souborů...")
    
    results = []
    
    for f in files:
        y_ref, mfcc_ref = extract_features(f)
        if y_ref is None: continue
        
        score, start_f, end_f = find_best_snippet(mfcc_query, mfcc_ref)
        
        if score != float('inf'):
            results.append({
                'name': f.name,
                'score': score,
                'start_f': start_f,
                'end_f': end_f,
                'y_ref': y_ref
            })

    print("\n✅ Analýza dokončena. Generuji soubory...")

    # 3. Seřazení a výstup
    results.sort(key=lambda x: x['score'])
    
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    # Hlavička tabulky
    print("-" * 90)
    print(f"{'#':<3} | {'SKÓRE':<8} | {'ČAS (Od - Do)':<20} | {'SOUBOR'}")
    print("-" * 90)

    for i, res in enumerate(results):
        # Výpočet času
        t_start = (res['start_f'] * HOP_LENGTH) / SR
        t_end = (res['end_f'] * HOP_LENGTH) / SR
        
        # Formátování času pro výpis (např. 12.5s - 18.2s)
        time_str = f"{t_start:.1f}s - {t_end:.1f}s"
        
        # Formátování názvu souboru (Čas dáme přímo do názvu)
        # RANK_01_TIME_12.5s_nazev.wav
        safe_name = Path(res['name']).stem.replace(" ", "_")
        out_filename = out_dir / f"RANK_{i+1:02d}_TIME_{t_start:.1f}s_{safe_name}.wav"
        
        # Výpis do konzole
        print(f"{i+1:<3} | {res['score']:<8.4f} | {time_str:<20} | {res['name']}")

        # Uložení audia
        if GENERATE_ALL:
            save_stereo_result(y_query, res['y_ref'], res['start_f'], res['end_f'], out_filename)

    print("-" * 90)
    print(f"💾 Všechny wav soubory jsou ve složce: '{OUTPUT_DIR}'")
    print(f"👀 Všimni si, že názvy souborů obsahují 'TIME_xx.xs', což je čas začátku shody.")

if __name__ == "__main__":
    main()