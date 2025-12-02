# matcher.py
#!/usr/bin/env python3

import sqlite3
import argparse
import time
from collections import defaultdict
from pathlib import Path

# IMPORT Z AUDIO CORE (stejná logika jako indexer)
from audio_core import (
    load_audio, compute_mel_spectrogram, get_2d_peaks, generate_hashes, 
    PEAK_THRESHOLD_DB
)

class SoundMatcher:
    def __init__(self, db_path):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Databáze nenalezena: {db_path}")
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def find_matches(self, hashes):
        """
        Hledá shody a vrací seřazené výsledky podle skóre.
        """
        t_start = time.time()
        
        # 1. Extrakce hash klíčů
        mapper = {} 
        for h, t in hashes:
            if h not in mapper: mapper[h] = []
            mapper[h].append(t)
        
        keys = list(mapper.keys())
        if not keys: return []

        print(f" -> Dotazuji se DB na {len(keys)} unikátních hashů...")
        
        # 2. Hledání v DB (Chunking pro velké množství hashů)
        CHUNK_SIZE = 900
        results = []
        for i in range(0, len(keys), CHUNK_SIZE):
            chunk = keys[i:i + CHUNK_SIZE]
            placeholders = ',' .join('?' for _ in chunk)
            query = f"""
                SELECT h.hash, h.sound_id, h.offset 
                FROM hashes h
                WHERE h.hash IN ({placeholders})
            """
            self.cursor.execute(query, chunk)
            results.extend(self.cursor.fetchall())
            
        print(f" -> Nalezeno {len(results)} hrubých shod v DB ({time.time()-t_start:.2f}s)")

        # 3. Time Alignment (Hledání konstantního posunu)
        matches_counter = defaultdict(lambda: defaultdict(int))
        
        for db_hash, sound_id, db_offset in results:
            if db_hash in mapper:
                for sample_offset in mapper[db_hash]:
                    # diff = Kde je to v DB - Kde je to v ukázce
                    diff = db_offset - sample_offset
                    matches_counter[sound_id][diff] += 1

        # 4. Agregace výsledků
        final_results = []
        unique_sound_ids = list(matches_counter.keys())
        
        # Načteme jména souborů
        names_map = {}
        if unique_sound_ids:
            q_ids = ','.join('?' for _ in unique_sound_ids)
            self.cursor.execute(f"SELECT id, media_id FROM sounds WHERE id IN ({q_ids})", unique_sound_ids)
            for sid, name in self.cursor.fetchall():
                names_map[sid] = name

        for sound_id, diffs in matches_counter.items():
            best_diff, score = max(diffs.items(), key=lambda x: x[1])
            final_results.append({
                "media_id": names_map.get(sound_id, "Unknown"),
                "score": score,
                "offset_frames": best_diff,
                "sound_id": sound_id
            })

        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_file", help="Soubor k vyhledání")
    parser.add_argument("--db_path", default="G:/SHAZAM/indexer.db")
    
    # Parametry DSP (Musí být shodné s indexerem!)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=128)
    
    args = parser.parse_args()
    
    print(f"=== AUDIO MATCHER ===")
    print(f"Analyzuji: {args.query_file}")
    
    # 1. Zpracování zvuku
    y, sr = load_audio(args.query_file, target_sr=args.sr)
    if y is None: return

    mel_db = compute_mel_spectrogram(y, sr, n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels)
    
    # Používá optimalizované parametry z audio_core
    peaks = get_2d_peaks(mel_db, amp_min=PEAK_THRESHOLD_DB)
    hashes = generate_hashes(peaks)
    
    print(f"Vygenerováno {len(hashes)} otisků (Peaks: {len(peaks)}). Hledám...")
    
    # 2. Hledání
    try:
        matcher = SoundMatcher(args.db_path)
        matches = matcher.find_matches(hashes)
    except Exception as e:
        print(f"Chyba: {e}")
        return

    print("\n=== VÝSLEDKY ===")
    if not matches:
        print("Žádná shoda.")
    else:
        for m in matches[:5]:
            offset_sec = (m['offset_frames'] * args.hop_length) / sr
            print(f"MEDIA: {m['media_id']}")
            print(f"  -> Skóre: {m['score']} (shodných bodů)")
            print(f"  -> Čas v originále: {offset_sec:.2f} s")
            print("-" * 30)

if __name__ == "__main__":
    main()