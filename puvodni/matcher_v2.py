#!/usr/bin/env python3

import argparse
import numpy as np
import librosa
import os
from pathlib import Path
from tqdm import tqdm
import warnings

# Potlačení warningů
warnings.filterwarnings("ignore")

########################################
# CONFIG
########################################

# MFCC jsou lepší pro hlas/slova. Chroma je lepší pro hudbu.
# Pro "stejná slova" doporučuji MFCC.
FEATURE_TYPE = 'mfcc'  # 'mfcc' nebo 'chroma'
SR = 16000             # Stačí nižší SR pro hlas/obsah
N_MFCC = 13            # Počet koeficientů (13 je standard pro řeč)
HOP_LENGTH = 512       

########################################
# 1. FEATURE EXTRACTION
########################################

def load_and_extract_features(path):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        
        # Odstranění ticha na začátku a konci (důležité pro srovnání!)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y) == 0:
            return None

        if FEATURE_TYPE == 'mfcc':
            # MFCC - skvělé pro řeč a obecné zvuky
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
            # Normalizace (aby nezáleželo na hlasitosti)
            features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-9)
        else:
            # Chroma - skvělé pro harmonickou hudbu
            features = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP_LENGTH)
            
        return features
    except Exception as e:
        print(f"[ERR] {path.name}: {e}")
        return None

########################################
# 2. VÝPOČET PODOBNOSTI (DTW)
########################################

def compute_dtw_distance(feat1, feat2):
    """
    Vypočítá DTW vzdálenost mezi dvěma sekvencemi.
    Nižší číslo = větší shoda.
    """
    # Používáme librosa DTW. Vrací D (cost matrix) a wp (path).
    # Zajímá nás poslední hodnota v cost matrix normalizovaná délkou cesty.
    try:
        D, wp = librosa.sequence.dtw(X=feat1, Y=feat2, metric='euclidean')
        
        # Normalizovaná vzdálenost (aby nezáleželo na délce nahrávky)
        # Poslední prvek D[-1, -1] je celková cena. Dělíme délkou cesty.
        cost = D[-1, -1] / len(wp)
        return cost
    except Exception:
        return float('inf')

########################################
# 3. MAIN
########################################

def main():
    parser = argparse.ArgumentParser(description="Similarity Matcher (DTW)")
    parser.add_argument("--input_dir", default="INPUT", help="Složka se známými zvuky (Databáze)")
    parser.add_argument("--query_dir", default="MATCHER", help="Složka s nahrávkami k identifikaci")
    
    args = parser.parse_args()
    
    ref_dir = Path(args.input_dir)
    query_dir = Path(args.query_dir)

    # 1. Načtení referenčních souborů (INPUT) do paměti
    print(f"--- Načítám referenční zvuky z {ref_dir} ---")
    references = []
    files = [f for f in ref_dir.rglob("*") if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']]
    
    if not files:
        print("❌ Složka INPUT je prázdná!")
        return

    for f in tqdm(files):
        feats = load_and_extract_features(f)
        if feats is not None:
            references.append({
                'name': f.name,
                'features': feats
            })
            
    print(f"✅ Načteno {len(references)} referenčních vzorků.")
    print("=" * 60)

    # 2. Porovnávání souborů z MATCHER
    queries = [f for f in query_dir.rglob("*") if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']]
    
    if not queries:
        print("❌ Složka MATCHER je prázdná!")
        return

    for q_file in queries:
        print(f"\n🎧 ANALÝZA: {q_file.name}")
        q_feats = load_and_extract_features(q_file)
        
        if q_feats is None:
            continue

        results = []
        
        # Porovnání Query se všemi v References
        # (Tohle může chvíli trvat, pokud je INPUT obrovský)
        for ref in references:
            dist = compute_dtw_distance(q_feats, ref['features'])
            results.append({
                'name': ref['name'],
                'distance': dist
            })
            
        # Seřazení podle vzdálenosti (nejmenší vzdálenost = nejlepší shoda)
        results.sort(key=lambda x: x['distance'])
        
        # Výpis TOP 5
        print(f"   📊 ŽEBŘÍČEK PODOBNOSTI (Nižší skóre = lepší):")
        for i, res in enumerate(results[:5]):
            score = res['distance']
            
            # Heuristika pro barvičky (záleží na datech, nutno vyzkoušet)
            if score < 25.0:    icon = "✅" # Velmi podobné
            elif score < 40.0:  icon = "❓" # Možná
            else:               icon = "  " # Asi ne
            
            print(f"      {i+1}. {icon} {res['name']:<30} | Distance: {score:.2f}")

if __name__ == "__main__":
    main()