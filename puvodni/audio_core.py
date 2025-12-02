# audio_core.py
import numpy as np
import librosa
import subprocess
from scipy.ndimage import maximum_filter

########################################
# CONFIG (Optimalizováno pro hlas/zvuk)
########################################

# Sníženo z 10 na 5 pro zachycení tichých částí řeči
PEAK_THRESHOLD_DB = 5       

# Konstanty pro hashing
FAN_VALUE = 15              
MIN_HASH_TIME_DELTA = 0     
MAX_HASH_TIME_DELTA = 200   

########################################
# DSP FUNCTIONS
########################################

def load_audio(path, target_sr=22050):
    path = str(path)
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
    except:
        # Fallback na ffmpeg pokud librosa selže
        try:
            cmd = ["ffmpeg", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "pipe:1"]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            y = np.frombuffer(out, dtype=np.float32)
            sr = target_sr
        except Exception as e:
            print(f"[ERR] Nepodařilo se načíst {path}: {e}")
            return None, None

    # --- OPTIMALIZACE: NORMALIZACE HLASITOSTI ---
    # Toto je kritické pro tiché nahrávky řeči
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    
    return y, sr

def compute_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(S)**2
    
    # --- OPTIMALIZACE: BANDPASS FILTER (150Hz - 4000Hz) ---
    # Odstraní hluky místnosti (<150) a syčení (>4000), soustředí se na hlas
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=150, fmax=4000)
    
    mel_S = np.dot(mel_basis, S_mag)
    mel_db = librosa.power_to_db(mel_S + 1e-9, ref=np.max)
    return mel_db

def get_2d_peaks(mel_db, amp_min=PEAK_THRESHOLD_DB):
    """
    Hledá lokální maxima (peaks).
    Upraveno pro řeč: menší okno a medián pro pozadí.
    """
    # --- OPTIMALIZACE: MENŠÍ OKNO (5x5) ---
    # Řeč se mění rychleji než hudba, potřebujeme detailnější mřížku
    struct_elem = np.ones((5, 5), dtype=bool)
    
    local_max = maximum_filter(mel_db, footprint=struct_elem)
    
    # Použijeme medián místo průměru (lépe odolává pauzám v řeči)
    background_threshold = np.median(mel_db) + amp_min 
    
    detected_peaks = (mel_db == local_max) & (mel_db > background_threshold)
    
    freq_indices, time_indices = np.where(detected_peaks)
    peaks = list(zip(freq_indices, time_indices))
    peaks.sort(key=lambda x: x[1])
    return peaks

def generate_hashes(peaks):
    """
    Vytváří otisky (fingerprints) z bodů.
    Vrací seznam (hash_key, time_offset)
    """
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