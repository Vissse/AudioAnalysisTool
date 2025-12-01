import numpy as np
import librosa
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt
import os

# ==========================================
# NASTAVENÍ
# ==========================================
DLOUHA_NAHRAVKA = "G:/SHAZAM/INPUT/test.mp3"
HLEDANY_VZOREK = "G:\SHAZAM\INPUT\slovo_veci.mp3"

# Kam uložit výstřižek pro kontrolu
VYSTUPNI_KONTROLA = "nalezeno_check.wav"
ZOBRAZIT_GRAF = True
# ==========================================

def get_loudness_envelope(y, hop_length):
    """
    Vypočítá křivku hlasitosti (RMS Energy).
    Tím ignorujeme barvu hlasu a soustředíme se na rytmus slabik.
    """
    # frame_length = délka okna pro průměrování energie
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    
    # Normalizace (aby nezáleželo na tom, jak nahlas je nahrávka celkově)
    rms = rms - np.mean(rms)
    if np.std(rms) > 0:
        rms = rms / np.std(rms)
        
    return rms

def main():
    print("1. Načítám audio...")
    try:
        # Použijeme nižší SR, pro tvar slova bohatě stačí
        target_sr = 16000 
        y_long, _ = librosa.load(DLOUHA_NAHRAVKA, sr=target_sr, mono=True)
        y_snippet, _ = librosa.load(HLEDANY_VZOREK, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Chyba: {e}")
        return

    # Ořezání ticha u vzorku (aby začínal hned slovem)
    y_snippet, _ = librosa.effects.trim(y_snippet, top_db=20)

    print("2. Vypočítávám energetické křivky (tvary slov)...")
    # hop_length určuje přesnost. 256 samples při 16kHz = cca 16ms přesnost
    hop = 256
    
    env_long = get_loudness_envelope(y_long, hop)
    env_snippet = get_loudness_envelope(y_snippet, hop)

    print("3. Hledám shodu v rytmu...")
    # Korelace nad obálkami (ne nad surovým zvukem)
    correlation = scipy.signal.correlate(env_long, env_snippet, mode='valid')
    
    # Najít nejlepší shodu
    peak_frame = np.argmax(correlation)
    
    # Převod na čas
    samples_offset = peak_frame * hop
    time_seconds = samples_offset / target_sr
    
    m, s = divmod(time_seconds, 60)
    
    print("\n" + "="*40)
    print(f" NEJLÉPE PASUJÍCÍ MÍSTO: {int(m):02d}:{s:05.2f}")
    print("="*40)

    # --- ULOŽENÍ VÝSTŘIŽKU PRO KONTROLU ---
    print(f"Ukládám nalezenou část do souboru '{VYSTUPNI_KONTROLA}'...")
    
    # Vypočítáme délku vzorku v samplech
    snippet_len_samples = len(y_snippet)
    
    # Vystřihneme z dlouhé nahrávky
    start_sample = samples_offset
    end_sample = start_sample + snippet_len_samples
    
    # Ošetření konců
    if end_sample > len(y_long): end_sample = len(y_long)
    
    found_audio = y_long[start_sample:end_sample]
    
    # Uložíme
    sf.write(VYSTUPNI_KONTROLA, found_audio, target_sr)
    print("-> Hotovo. Poslechněte si tento soubor, zda je to správně.")

    if ZOBRAZIT_GRAF:
        plt.figure(figsize=(12, 6))
        # Časová osa pro korelaci
        frames_time = np.linspace(0, (len(correlation)*hop)/target_sr, len(correlation))
        
        plt.plot(frames_time, correlation, label='Shoda tvaru (Rytmus)')
        plt.plot(time_seconds, correlation[peak_frame], 'ro', markersize=10, label='Nalezeno')
        
        plt.title(f"Hledání podle tvaru hlasitosti - Výsledek: {int(m):02d}:{s:05.2f}")
        plt.xlabel("Čas (sekundy)")
        plt.ylabel("Podobnost")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()

if __name__ == "__main__":
    main()