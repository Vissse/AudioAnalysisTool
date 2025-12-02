#!/usr/bin/env python3

import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")

########################################
# CONFIG
########################################
SR = 16000
N_MFCC = 13
HOP_LENGTH = 512

def warp_audio(query_audio, wp, ref_len_samples, hop_length):
    """
    Klíčová funkce: "Pokřiví" (Warp) nahrávku query tak, aby časově
    odpovídala referenci podle cesty DTW (wp).
    """
    # wp[:, 0] jsou indexy v Query (to co chceme měnit)
    # wp[:, 1] jsou indexy v Referenci (cílový čas)
    
    # Obrátíme cestu, aby šla od začátku do konce (librosa ji vrací pozpátku)
    wp = wp[::-1]
    
    # Převod z rámců (frames) na vzorky (samples)
    query_indices = wp[:, 0] * hop_length
    ref_indices = wp[:, 1] * hop_length
    
    # Vytvoříme interpolační funkci:
    # "Pro tento čas v originále (ref), jaký čas mám vzít z tvé nahrávky (query)?"
    # kind='linear' zajistí plynulý přechod (zrychlení/zpomalení)
    interpolator = interp1d(ref_indices, query_indices, kind='linear', fill_value="extrapolate")
    
    # Vytvoříme časovou osu pro nový (warped) zvuk (podle délky originálu)
    target_times = np.arange(ref_len_samples)
    
    # Zjistíme, které vzorky z tvé nahrávky máme vzít
    sample_map = interpolator(target_times)
    
    # Zajistíme, abychom nečetli mimo rozsah pole
    sample_map = np.clip(sample_map, 0, len(query_audio) - 1).astype(int)
    
    # Vytvoříme nové audio seřazením vzorků podle mapy
    warped_query = query_audio[sample_map]
    
    return warped_query

def process_pair(query_path, ref_path, output_path):
    print(f"🔄 Zpracovávám:")
    print(f"   Query (Ty):     {query_path}")
    print(f"   Ref (Originál): {ref_path}")

    # 1. Načtení audia
    y_query, _ = librosa.load(query_path, sr=SR)
    y_ref, _ = librosa.load(ref_path, sr=SR)
    
    # Trim ticha (pro lepší synchronizaci)
    y_query, _ = librosa.effects.trim(y_query, top_db=20)
    y_ref, _ = librosa.effects.trim(y_ref, top_db=20)

    # 2. Výpočet features (MFCC)
    mfcc_query = librosa.feature.mfcc(y=y_query, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    
    # Normalizace
    mfcc_query = (mfcc_query - np.mean(mfcc_query, axis=1, keepdims=True)) / (np.std(mfcc_query, axis=1, keepdims=True) + 1e-9)
    mfcc_ref = (mfcc_ref - np.mean(mfcc_ref, axis=1, keepdims=True)) / (np.std(mfcc_ref, axis=1, keepdims=True) + 1e-9)

    # 3. DTW (Získání cesty zarovnání)
    # metric='cosine' je lepší pro barvu hlasu
    D, wp = librosa.sequence.dtw(X=mfcc_query, Y=mfcc_ref, metric='cosine')
    
    # Skóre (jen pro info)
    score = D[-1, -1] / len(wp)
    print(f"   📊 DTW Distance: {score:.4f}")

    # 4. Audio Warping
    print("   🔨 Ohýbám čas (Time Warping)...")
    y_query_warped = warp_audio(y_query, wp, len(y_ref), HOP_LENGTH)

    # 5. Vytvoření Stereo Mixu
    # Ujistíme se, že oba kanály mají stejnou délku
    min_len = min(len(y_ref), len(y_query_warped))
    y_ref = y_ref[:min_len]
    y_query_warped = y_query_warped[:min_len]
    
    # Normalizace hlasitosti (aby jedno neřvalo víc než druhé)
    y_ref = librosa.util.normalize(y_ref) * 0.8
    y_query_warped = librosa.util.normalize(y_query_warped) * 0.8

    # Stack do sterea (Left=Ref, Right=WarpedQuery)
    stereo_output = np.vstack((y_ref, y_query_warped)).T
    
    # 6. Uložení
    sf.write(output_path, stereo_output, SR)
    print(f"   💾 Uloženo do: {output_path}")
    print("   🎧 (Nasaď si sluchátka: Levé=Originál, Pravé=Ty)")

def main():
    parser = argparse.ArgumentParser(description="Poslechová analýza shody")
    parser.add_argument("query_file", help="Cesta k tvé nahrávce")
    parser.add_argument("ref_file", help="Cesta k originálu (se kterým to chceš srovnat)")
    parser.add_argument("--out", default="comparison.wav", help="Výstupní soubor")
    
    args = parser.parse_args()
    
    try:
        process_pair(args.query_file, args.ref_file, args.out)
    except Exception as e:
        print(f"❌ Chyba: {e}")

if __name__ == "__main__":
    main()