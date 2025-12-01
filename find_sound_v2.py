import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from skimage.feature import match_template
import os

# ==========================================
# NASTAVENÍ
# ==========================================
DLOUHA_NAHRAVKA = "G:/SHAZAM/INPUT/zkouska.mp3"
HLEDANY_VZOREK = "G:\SHAZAM\INPUT\slovo_noviny.mp3"
VYSTUPNI_KONTROLA = "nalezeno_final.wav"

# Práh citlivosti (0.0 až 1.0). 
# Pokud to najde nesmysl, zvyšte na 0.6 nebo 0.7
# Pokud nenajde nic, snižte na 0.4
PRAH = 0.7 
# ==========================================

def compute_normalized_spectrogram(y, sr):
    """
    Vytvoří spektrogram, který je odolný vůči změně hlasitosti.
    """
    # 1. Mel Spektrogram (zaměření na lidský hlas 150-8000Hz)
    # n_mels=80 dává dost detailů pro rozlišení hlásek
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, 
        n_mels=80, fmin=150, fmax=8000
    )
    
    # 2. Logaritmická škála (dB) - jako lidské ucho
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 3. Z-Score Normalizace (KLÍČOVÝ KROK)
    # Od každého bodu odečteme průměr a vydělíme odchylkou.
    # Tím zajistíme, že extrémně hlasité tlesknutí bude mít stejnou "váhu" 
    # jako tiché slovo. Obě budou mít hodnoty cca od -3 do +3.
    norm_S = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-6)
    
    return norm_S

def main():
    print("1. Načítám audio...")
    try:
        target_sr = 16000 
        y_long, _ = librosa.load(DLOUHA_NAHRAVKA, sr=target_sr, mono=True)
        y_snippet, _ = librosa.load(HLEDANY_VZOREK, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Chyba: {e}")
        return

    # Ořezání ticha u vzorku
    y_snippet, _ = librosa.effects.trim(y_snippet, top_db=20)

    print("2. Vypočítávám normalizované spektrogramy...")
    # Tady se děje to kouzlo, které ignoruje hlasitost tleskání
    S_long = compute_normalized_spectrogram(y_long, target_sr)
    S_snippet = compute_normalized_spectrogram(y_snippet, target_sr)

    print("3. Porovnávám vzory (Match Template)...")
    # match_template automaticky dělá "Normalized Cross-Correlation"
    # To znamená, že hledá SHODU TVARU, ne shodu energie.
    result = match_template(S_long, S_snippet)
    
    # Výsledek je 2D matice, ale nás zajímá jen časová osa (max přes frekvence)
    # Protože match_template vrací jeden bod za celý blok, musíme vzít jen 1. řádek nebo max
    score_curve = np.max(result, axis=0) # Axis 0 nebo 1 podle orientace librosa (librosa má čas na ose 1)
    
    # Librosa má čas na ose 1 (sloupce). match_template vrací mapu (výška_mapy, šířka_mapy).
    # Protože výška vzorku a hledaného obrazu je stejná (n_mels), výška výsledku je 1.
    score_curve = result.flatten()

    # Najít nejvyšší shodu
    peak_idx = np.argmax(score_curve)
    max_score = score_curve[peak_idx]
    
    # Převod na čas
    hop_length = 512
    time_seconds = (peak_idx * hop_length) / target_sr
    
    m, s = divmod(time_seconds, 60)

    print("\n" + "="*40)
    print(f" VÝSLEDEK: {int(m):02d}:{s:05.2f}")
    print(f" Jistota shody: {max_score:.2f} (Max je 1.0)")
    print("="*40)

    if max_score < PRAH:
        print("VAROVÁNÍ: Jistota je velmi nízká. Možná to tam není, nebo je to příliš odlišné.")

    # --- ULOŽENÍ ---
    start_sample = int(peak_idx * hop_length)
    end_sample = start_sample + len(y_snippet)
    if end_sample > len(y_long): end_sample = len(y_long)
    
    sf.write(VYSTUPNI_KONTROLA, y_long[start_sample:end_sample], target_sr)
    print(f"Uloženo do: {VYSTUPNI_KONTROLA} (zkontrolujte poslech)")

    # GRAF
    plt.figure(figsize=(12, 6))
    times = np.linspace(0, (len(score_curve) * hop_length) / target_sr, len(score_curve))
    plt.plot(times, score_curve, label='Podobnost')
    plt.plot(time_seconds, max_score, 'ro', label='Nalezeno')
    plt.axhline(PRAH, color='r', linestyle='--', alpha=0.5, label='Práh')
    plt.title(f"Spektrální hledání - Čas: {int(m):02d}:{s:05.2f}")
    plt.xlabel("Čas (s)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()