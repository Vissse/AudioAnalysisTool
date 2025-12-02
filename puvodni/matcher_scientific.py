#!/usr/bin/env python3

import numpy as np
import librosa
import soundfile as sf
import os
from pathlib import Path
import warnings
from scipy.stats import zscore

warnings.filterwarnings("ignore")

########################################
#  🛠️  UŽIVATELSKÉ NASTAVENÍ
########################################

QUERY_PATH = "G:/SHAZAM/MATCHER/Zaznam_3.mp3"
INPUT_DIR = "INPUT"
OUTPUT_DIR = "VYSLEDKY_SCIENTIFIC"

# VĚDECKÉ NASTAVENÍ:
# 'mfcc'   = Pro mluvené slovo (řeší barvu hlasu, fonémy)
# 'chroma' = Pro zpěv/hudbu (řeší melodii, ignoruje barvu hlasu - dle literatury Müller et al.)
FEATURE_MODE = 'mfcc' 

########################################
# CONFIG
########################################
SR = 22050
HOP_LENGTH = 512

def extract_features(path, mode='mfcc'):
    print(".", end="", flush=True)
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y) < 2048: return None, None

        if mode == 'mfcc':
            # MFCC: Klasika pro řeč
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
        elif mode == 'chroma':
            # Chroma CENS: State-of-the-art pro Cover Song Identification
            # CENS (Chroma Energy Normalized Statistics) vyhlazuje lokální odchylky
            features = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP_LENGTH)

        # Normalizace (Z-score normalization po dimenzích)
        # Toto je klíčové pro Cosine distance
        features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-9)
        
        return y, features
    except Exception:
        return None, None

def find_best_subsequence_dtw(query_feat, ref_feat):
    """
    Subsequence DTW (Dynamic Time Warping)
    Vrací normalizovanou vzdálenost (nižší = lepší).
    """
    n_query = query_feat.shape[1]
    n_ref = ref_feat.shape[1]
    
    if n_query > n_ref:
        return float('inf'), 0, 0 

    try:
        # Cosine metrika je standard pro porovnávání vektorů v MIR
        D, wp = librosa.sequence.dtw(X=query_feat, Y=ref_feat, metric='cosine', subseq=True)
        
        min_cost_idx = np.argmin(D[-1, :])
        min_cost = D[-1, min_cost_idx]
        
        path_start = wp[-1] 
        path_end = wp[0] 
        
        ref_start_frame = path_start[1]
        ref_end_frame = path_end[1]
        
        # Normalizace délkou cesty
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

def calculate_statistics(results):
    """
    Aplikuje statistickou analýzu na výsledky.
    """
    if not results: return []
    
    # Extrahuje všechna skóre
    scores = np.array([r['raw_score'] for r in results])
    
    # Vypočítáme průměr a směrodatnou odchylku
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    processed_results = []
    for r in results:
        score = r['raw_score']
        
        # Z-Score: Kolik sigma je toto skóre od průměru?
        # Protože u DTW je MENŠÍ skóre lepší, otočíme znaménko.
        # Kladné Z-Score zde znamená "o kolik je to lepší než průměr".
        if std_score > 0:
            z_val = (mean_score - score) / std_score
        else:
            z_val = 0.0
            
        r['z_score'] = z_val
        processed_results.append(r)
        
    # Seřadíme podle raw_score (nejnižší první)
    processed_results.sort(key=lambda x: x['raw_score'])
    return processed_results

def main():
    print("="*80)
    print(f"      VĚDECKÝ SROVNÁVAČ (Metoda: {FEATURE_MODE.upper()} + DTW)")
    print("="*80)

    # 1. Načtení Query
    q_path = Path(QUERY_PATH)
    if not q_path.exists():
        print(f"❌ CHYBA: '{QUERY_PATH}' neexistuje.")
        return
    
    print(f"🎤 Query: {q_path.name}")
    y_query, feat_query = extract_features(q_path, mode=FEATURE_MODE)
    if y_query is None: return
    print(" ✅ Načteno")

    # 2. Načtení DB
    in_dir = Path(INPUT_DIR)
    files = [f for f in in_dir.rglob("*") if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']]
    
    if not files:
        print("❌ Žádné soubory v INPUT.")
        return

    print(f"🔍 Počítám DTW distance pro {len(files)} souborů...")
    
    raw_results = []
    
    for f in files:
        y_ref, feat_ref = extract_features(f, mode=FEATURE_MODE)
        if y_ref is None: continue
        
        score, start_f, end_f = find_best_subsequence_dtw(feat_query, feat_ref)
        
        if score != float('inf'):
            raw_results.append({
                'name': f.name,
                'raw_score': score,
                'start_f': start_f,
                'end_f': end_f,
                'y_ref': y_ref
            })

    print("\n✅ Výpočet hotov. Aplikuji statistickou analýzu...")

    # 3. Statistika (Z-Score)
    final_results = calculate_statistics(raw_results)
    
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(exist_ok=True)

    # Hlavička
    print("-" * 100)
    print(f"{'#':<3} | {'RAW SKÓRE':<10} | {'Z-SCORE':<8} | {'CONFIDENCE':<12} | {'SOUBOR'}")
    print(f"{'':<3} | {'(Méně=Lépe)':<10} | {'(Více=Lépe)':<8} | {'Verdikt':<12} | ")
    print("-" * 100)

    for i, res in enumerate(final_results):
        z = res['z_score']
        raw = res['raw_score']
        
        # Interpretace Z-Score podle statistiky
        # Z > 3.0 = Extrémní anomálie (tohle je určitě ono)
        # Z > 2.0 = Významná odchylka
        if z > 3.0:
            verdict = "🟢 JISTOTA"
        elif z > 1.5:
            verdict = "🟢 SHODA"
        elif z > 0.5:
            verdict = "🟡 MOŽNÁ"
        else:
            verdict = "🔴 ŠUM"

        print(f"{i+1:<3} | {raw:<10.4f} | {z:<8.2f} | {verdict:<12} | {res['name']}")

        # Uložení audia pro TOP 5
        if i < 5:
            t_start = (res['start_f'] * HOP_LENGTH) / SR
            safe_name = Path(res['name']).stem.replace(" ", "_")
            out_filename = out_dir / f"RANK_{i+1:02d}_Z_{z:.1f}_{safe_name}.wav"
            save_stereo_result(y_query, res['y_ref'], res['start_f'], res['end_f'], out_filename)

    print("-" * 100)
    print(f"💾 Výsledky (TOP 5) uloženy do: '{OUTPUT_DIR}'")
    print(f"💡 TIP: Sleduj sloupec Z-SCORE. Pokud je > 2.0, je to statisticky významná shoda.")

if __name__ == "__main__":
    main()