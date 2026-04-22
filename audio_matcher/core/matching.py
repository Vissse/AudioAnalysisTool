import numpy as np
import librosa
from scipy.spatial.distance import cdist
from skimage.feature import match_template

from config import APP_CFG, MODEL_CFG

def compute_dtw(
    query_feat: np.ndarray, 
    ref_feat: np.ndarray, 
    excluded_ranges: list[tuple[float, float]] | None = None
) -> tuple[float, np.ndarray, float, float, np.ndarray]:
    
    if excluded_ranges is None:
        excluded_ranges = []

    N = query_feat.shape[1]
    M = ref_feat.shape[1]

    # NOVÁ OCHRANNÁ POJISTKA
    # Fyzicky nelze hledat delší vzorek v kratším korpusu.
    # Zabrání to tichému převrácení matic funkcí numpy.convolve!
    if N > M:
        return float('inf'), np.array([]), 0.0, 0.0, np.empty((N, M))

    # Vektorizovaný výpočet kompletní matice vzdáleností v C (SciPy cdist je extrémně rychlý)
    cost_matrix = cdist(query_feat.T, ref_feat.T, metric=MODEL_CFG.dtw_metric)
    
    # Aplikace penalizace pro již nalezené/vyloučené úseky
    pad = max(10, APP_CFG.exclusion_padding_frames) 
    for start_f, end_f in excluded_ranges:
        s = max(0, int(start_f) - pad)
        e = min(M, int(end_f) + pad)
        # Penalizace se přičítá k existující ceně (+=)
        cost_matrix[:, s:e] += MODEL_CFG.dtw_penalty_value  
        
    # =========================================================================
    # OPTIMALIZACE 1: Lower Bounding Heuristics (Rychlé zamítnutí)
    # Spočítáme klouzavý průměr minimálních vzdáleností (vektorová konvoluce).
    # Toto nahrazuje drahé DP levným odhadem shody podél diagonály.
    # =========================================================================
    col_mins = np.min(cost_matrix, axis=0)
    window = np.ones(N) / N
    # Konvoluce nám dá vyhlazené "skóre" pro každý možný startovní frame
    rolling_bounds = np.convolve(col_mins, window, mode='valid')

    # Filtrování kandidátů: Zajímá nás jen to, co má šanci překonat náš práh
    # Filtrování kandidátů: Zajímá nás jen to, co má šanci překonat náš práh
    candidate_starts = np.where(rolling_bounds < APP_CFG.dtw_early_stop_threshold)[0]

    # OPRAVA: Pokud heuristika kvůli příliš přísnému prahu nic nenašla,
    # nesmíme slovo tiše zahodit (vrátit nekonečno). 
    # Místo toho vezmeme "to nejlepší, co se dalo najít" a necháme ho projít.
    if len(candidate_starts) == 0:
        # Najde absolutní minimum i přes to, že je nad prahem
        best_fallback = int(np.argmin(rolling_bounds))
        candidate_starts = np.array([best_fallback])

    # =========================================================================
    # OPTIMALIZACE 2: Omezení vyhledávacího prostoru (Sakoe-Chiba princip)
    # Vezmeme nejlepšího kandidáta z heuristiky a plné DP spustíme JEN kolem něj.
    # =========================================================================
    for _ in range(APP_CFG.dtw_max_retries):
        
        # Najdeme lokální minimum v našem odhadu
        best_start = candidate_starts[np.argmin(rolling_bounds[candidate_starts])]
        
        # Vyřízneme matici: Hledáme jen v úzkém okně kolem kandidáta (Boundary Tolerance)
        margin = int(N * MODEL_CFG.dtw_boundary_tolerance)
        search_start = max(0, best_start - margin)
        search_end = min(M, best_start + N + margin)
        
        local_cost_matrix = cost_matrix[:, search_start:search_end]
        
        # Plné DP (Dynamické programování) nyní běží např. na matici 30x40 místo 30x9000!
        D, wp = librosa.sequence.dtw(C=local_cost_matrix, subseq=True)
        
        min_cost_idx = np.argmin(D[-1, :])
        min_cost = float(D[-1, min_cost_idx])
        
        # Přepočet souřadnic lokální cesty na globální pozice v audiu
        path_in_ref = wp[:, 1]
        start_frame = float(np.min(path_in_ref)) + search_start
        end_frame = float(np.max(path_in_ref)) + search_start
        
        ratio = (end_frame - start_frame) / max(1, N)
        
        # Ochrana proti zdegenerovaným cestám (nalepení na hranu)
        path_length = end_frame - start_frame
        min_allowed_length = max(3, N * 0.25) 
        
        # Kontrola, zda slovo nebylo mluveno příliš rychle/pomalu A nemá nulovou délku
        if APP_CFG.min_speed_ratio <= ratio <= APP_CFG.max_speed_ratio and path_length >= min_allowed_length:
            return min_cost / len(wp), wp, start_frame, end_frame, cost_matrix
            
        # Pokud se ohyb nevešel do tolerancí (nebo je zdegenerovaný), penalizujeme tuto oblast
        rolling_bounds[max(0, best_start-margin) : min(len(rolling_bounds), best_start+margin)] = float('inf')
        candidate_starts = np.where(rolling_bounds < APP_CFG.dtw_early_stop_threshold)[0]
        
        if len(candidate_starts) == 0:
            break

    return float('inf'), np.array([]), 0.0, 0.0, cost_matrix


def compute_pattern_matching(
    query_spec: np.ndarray, 
    ref_spec: np.ndarray, 
    excluded_ranges: list[tuple[float, float]] | None = None
) -> tuple[float, np.ndarray, float, float]:
    
    if excluded_ranges is None:
        excluded_ranges = []

    result = match_template(ref_spec, query_spec).flatten()
    
    pad = max(10, APP_CFG.exclusion_padding_frames)
    for start_f, end_f in excluded_ranges:
        s = max(0, int(start_f) - pad)
        e = min(len(result), int(end_f) + pad)
        result[s:e] = -1.0  
        
    peak_idx = int(np.argmax(result))
    max_score = float(result[peak_idx])
    
    return max_score, result, float(peak_idx), float(peak_idx + query_spec.shape[1])