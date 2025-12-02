#!/usr/bin/env python3

import numpy as np
import librosa
import soundfile as sf
import os
from pathlib import Path
import warnings

# Potlačení chybových hlášek, aby byl výstup čitelný
warnings.filterwarnings("ignore")

########################################
#  🛠️  UŽIVATELSKÉ NASTAVENÍ (UPRAV ZDE)
########################################

# 1. Jak se jmenuje tvá nahrávka? (Zadej celou cestu nebo jen název)
QUERY_PATH = "G:/SHAZAM/MATCHER/Zaznam_3.mp3"

# 2. Kde jsou originály? (Název složky)
INPUT_DIR = "INPUT"

# 3. Jak se má jmenovat výsledek?
OUTPUT_FILE = "VYSLEDEK_SROVNANI.wav"

########################################
# KONFIGURACE (Neměnit, pokud nevíš co děláš)
########################################
SR = 16000          # 16kHz stačí pro hlas
N_MFCC = 13         # Počet rysů pro analýzu hlasu
HOP_LENGTH = 512    # Přesnost (menší číslo = větší přesnost, ale pomalejší)

def extract_features(path):
    print(f"   Načítám: {path.name} ... ", end="")
    try:
        if not path.exists():
            print("❌ CHYBA: Soubor neexistuje!")
            return None, None

        # Načtení audia
        y, sr = librosa.load(path, sr=SR, mono=True)
        
        # Oříznutí ticha
        y, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y) < 2048:
            print("⚠️ Moc krátké!")
            return None, None

        # Výpočet MFCC (charakteristika zvuku)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        # Normalizace
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-9)
        
        print("✅ OK")
        return y, mfcc
    except Exception as e:
        print(f"❌ CHYBA: {e}")
        return None, None

def find_best_snippet(query_mfcc, ref_mfcc):
    """
    Najde místo v 'ref_mfcc', které se nejvíce podobá 'query_mfcc'.
    """
    n_query = query_mfcc.shape[1]
    n_ref = ref_mfcc.shape[1]
    
    if n_query > n_ref:
        return float('inf'), 0, 0 # Query je delší než originál

    try:
        # Subsequence DTW - najde nejlepší pod-úsek
        # metric='cosine' ignoruje hlasitost, řeší jen barvu zvuku
        D, wp = librosa.sequence.dtw(X=query_mfcc, Y=ref_mfcc, metric='cosine', subseq=True)
        
        # Kde končí nejlepší shoda? (Minimum v posledním řádku)
        min_cost_idx = np.argmin(D[-1, :])
        min_cost = D[-1, min_cost_idx]
        
        # Kde shoda začíná?
        # Najdeme v cestě (wp) bod, který odpovídá našemu konci
        path_end = wp[0] 
        path_start = wp[-1] 
        
        ref_start_frame = path_start[1]
        ref_end_frame = path_end[1]
        
        # Normalizované skóre (průměrná chyba na jeden krok)
        normalized_score = min_cost / len(wp)
        
        return normalized_score, ref_start_frame, ref_end_frame
        
    except Exception:
        return float('inf'), 0, 0

def save_stereo_result(query_y, ref_y, start_f, end_f, filename):
    start_sample = start_f * HOP_LENGTH
    end_sample = end_f * HOP_LENGTH
    
    # Vystřihneme kus z originálu
    ref_segment = ref_y[start_sample:end_sample]
    
    # Srovnáme délky (pro stereo soubor musí být stejné)
    min_len = min(len(query_y), len(ref_segment))
    if min_len == 0: return
    
    out_q = query_y[:min_len]       # Levé ucho (Ty)
    out_r = ref_segment[:min_len]   # Pravé ucho (Originál)
    
    # Uložení
    stereo = np.vstack((out_q, out_r)).T
    sf.write(filename, stereo, SR)

def main():
    print("="*40)
    print("      HLEDAČ PODOBNOSTI (SIMPLE)")
    print("="*40)

    # 1. Kontrola Query
    q_path = Path(QUERY_PATH)
    if not q_path.exists():
        print(f"\n❌ CHYBA: Soubor '{QUERY_PATH}' nebyl nalezen!")
        print("   -> Zkontroluj řádek 'QUERY_PATH' ve skriptu.")
        return

    print(f"🎤 Tvoje nahrávka: {q_path.name}")
    y_query, mfcc_query = extract_features(q_path)
    if y_query is None: return

    # 2. Kontrola Input složky
    in_dir = Path(INPUT_DIR)
    if not in_dir.exists():
        print(f"\n❌ CHYBA: Složka '{INPUT_DIR}' neexistuje.")
        return
        
    files = [f for f in in_dir.rglob("*") if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']]
    if not files:
        print(f"\n❌ Složka '{INPUT_DIR}' je prázdná.")
        return

    print(f"\n🔍 Prohledávám {len(files)} souborů v '{INPUT_DIR}'...")
    print("-" * 40)
    
    results = []

    # 3. Hlavní smyčka
    for f in files:
        # Načteme originál
        y_ref, mfcc_ref = extract_features(f)
        if y_ref is None: continue
        
        # Najdeme shodu
        score, start_f, end_f = find_best_snippet(mfcc_query, mfcc_ref)
        
        # Převedeme na sekundy
        t_start = (start_f * HOP_LENGTH) / SR
        t_end = (end_f * HOP_LENGTH) / SR
        
        print(f"      -> Rozdíl: {score:.4f} (Čas: {t_start:.1f}s - {t_end:.1f}s)")
        
        results.append({
            'name': f.name,
            'score': score,
            'y_ref': y_ref,
            'start_f': start_f,
            'end_f': end_f
        })

    # 4. Vyhodnocení
    if not results:
        print("\n❌ Nic se nepodařilo porovnat.")
        return

    # Seřadíme podle nejmenšího rozdílu (nejlepší shoda)
    results.sort(key=lambda x: x['score'])
    winner = results[0]

    print("\n" + "="*40)
    print("      🏆 NEJLEPŠÍ VÝSLEDEK")
    print("="*40)
    print(f"Soubor: {winner['name']}")
    print(f"Skóre:  {winner['score']:.4f} (Méně je lépe)")
    
    t_start = (winner['start_f'] * HOP_LENGTH) / SR
    t_end = (winner['end_f'] * HOP_LENGTH) / SR
    print(f"Místo:  {t_start:.2f}s až {t_end:.2f}s")
    
    # Uložení
    save_stereo_result(y_query, winner['y_ref'], winner['start_f'], winner['end_f'], OUTPUT_FILE)
    print(f"\n💾 Uloženo do souboru: {OUTPUT_FILE}")
    print("🎧 Nasaď si sluchátka a pusť si to!")
    print("   Levé ucho  = Tvoje nahrávka")
    print("   Pravé ucho = Nalezená část z originálu")

if __name__ == "__main__":
    main()