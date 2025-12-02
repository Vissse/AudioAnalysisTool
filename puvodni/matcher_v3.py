#!/usr/bin/env python3

import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

########################################
# CONFIG
########################################

SR = 16000             
N_MFCC = 13            
HOP_LENGTH = 512       

# Jak moc detailní má být graf?
FIG_SIZE = (10, 8)

########################################
# FUNKCE
########################################

def load_features(path):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        # Oříznutí ticha (důležité pro vizualizaci, aby graf nezačínal dlouhým prázdnem)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y) == 0: return None, None

        # MFCC Features (Tvar zvuku)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        
        # Normalizace
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-9)
        
        return mfcc, len(y) / sr # Vracíme i délku v sekundách
    except Exception as e:
        print(f"[ERR] {path.name}: {e}")
        return None, None

def plot_alignment(query_mfcc, ref_mfcc, query_name, ref_name, save_path):
    """
    Vypočítá DTW a vykreslí cestu zarovnání.
    """
    # Výpočet DTW
    # metric='cosine' je často lepší pro srovnání barvy hlasu než euclidean
    D, wp = librosa.sequence.dtw(X=query_mfcc, Y=ref_mfcc, metric='cosine')
    
    # Výpočet průměrné vzdálenosti (Skóre podobnosti)
    # Menší = lepší
    score = D[-1, -1] / len(wp)

    # --- VYKRESLENÍ GRAFU ---
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)

    # Vykreslení matice vzdáleností (heatmapa)
    # Tmavá místa = vysoká shoda, Světlá = rozdíl
    librosa.display.specshow(D, x_axis='frames', y_axis='frames', 
                             cmap='gray_r', ax=ax)
    
    # Vykreslení Optimální Cesty (Bílá/Červená čára)
    # wp[:, 1] je osa X (Query), wp[:, 0] je osa Y (Reference)
    ax.plot(wp[:, 1], wp[:, 0], label='Shoda', color='r', linewidth=2)

    # Popisky os
    ax.set_title(f"Zarovnání: {query_name} vs {ref_name}\nDistance Score: {score:.3f} (Méně je lépe)")
    ax.set_xlabel(f"Tvoje nahrávka: {query_name} (framy)")
    ax.set_ylabel(f"Originál: {ref_name} (framy)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return score

########################################
# MAIN
########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="INPUT", help="Složka originálů")
    parser.add_argument("--query_dir", default="MATCHER", help="Složka nahrávek")
    args = parser.parse_args()
    
    ref_dir = Path(args.input_dir)
    query_dir = Path(args.query_dir)
    
    # 1. Načtení referencí
    print("--- Načítám originály ---")
    refs = []
    for f in ref_dir.rglob("*"):
        if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']:
            feat, dur = load_features(f)
            if feat is not None:
                refs.append({'name': f.name, 'feat': feat, 'dur': dur})

    if not refs:
        print("Složka INPUT je prázdná.")
        return

    # 2. Zpracování Query
    queries = [f for f in query_dir.rglob("*") if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']]
    
    for q_file in queries:
        print(f"\n🎧 Zpracovávám: {q_file.name}")
        q_feat, q_dur = load_features(q_file)
        if q_feat is None: continue
        
        results = []

        # Porovnání se všemi originály
        for ref in tqdm(refs, desc="Porovnávám"):
            # Generujeme název obrázku
            out_img = query_dir / f"ALIGN_{q_file.stem}_VS_{ref['name']}.png"
            
            # Spustit výpočet a kreslení
            score = plot_alignment(q_feat, ref['feat'], q_file.name, ref['name'], out_img)
            
            results.append({
                'name': ref['name'],
                'score': score,
                'img': out_img.name
            })
            
        # Seřadit podle skóre
        results.sort(key=lambda x: x['score'])
        
        print(f"   📊 VÝSLEDKY PRO {q_file.name}:")
        for i, res in enumerate(results[:3]): # Top 3
            print(f"      {i+1}. {res['name']}")
            print(f"         Skóre: {res['score']:.4f}")
            print(f"         Graf:  {res['img']}")

if __name__ == "__main__":
    main()