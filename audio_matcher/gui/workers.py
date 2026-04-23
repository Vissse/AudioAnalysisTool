import os
import json
import string
import librosa
import difflib
import h5py
import numpy as np
import soundfile as sf
import traceback
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from config import AUDIO_CFG, APP_CFG, MODEL_CFG
from data_types import MatchResult, TimeRange
from core.audio_utils import (
    load_and_prep, get_display_spectrogram, get_mfcc_features, 
    get_math_spectrogram, get_wav2vec_features, refine_boundaries_vad,
    stream_audio_chunks
)
from core.matching import compute_dtw, compute_pattern_matching
from core.model_manager import ModelManager


class SingleAnalysisWorker(QThread):
    finished = pyqtSignal(MatchResult)
    progress = pyqtSignal(int)

    def __init__(self, long_path: str, query_path: str, method: str, excluded: list[TimeRange]):
        super().__init__()
        self.long_path = long_path
        self.query_path = query_path
        self.method = method
        self.excluded = excluded

    def run(self):
        try:
            self.progress.emit(10)
            y_long, sr = load_and_prep(self.long_path)
            y_query, _ = load_and_prep(self.query_path)

            if y_query is None or y_long is None:
                raise ValueError("Nelze načíst audio soubory.")

            self.progress.emit(30)
            vis_spec_long = get_display_spectrogram(y_long, sr)
            vis_spec_query = get_display_spectrogram(y_query, sr)
            
            result = MatchResult(
                score=0.0, start_f=0.0, end_f=0.0, method=self.method,
                y_long=y_long, y_query=y_query, sr=sr,
                vis_spec_query=vis_spec_query, vis_spec_long=vis_spec_long
            )

            self.progress.emit(50)
            
            if self.method == 'whisper':
                whisper_model = ModelManager().get_whisper()
                self.progress.emit(60)
                
                y_long_16k = librosa.resample(y_long, orig_sr=sr, target_sr=MODEL_CFG.model_target_sr) if sr != MODEL_CFG.model_target_sr else y_long
                y_query_16k = librosa.resample(y_query, orig_sr=sr, target_sr=MODEL_CFG.model_target_sr) if sr != MODEL_CFG.model_target_sr else y_query
                
                # --- HLAVNÍ OPRAVA: OCHRANA PROTI PÁDU WHISPERU NA VZORKU ---
                try:
                    query_res = whisper_model.transcribe(y_query_16k, language=MODEL_CFG.whisper_language)
                    target_text = query_res.get("text", "").strip().lower().translate(str.maketrans('', '', string.punctuation))
                except Exception:
                    target_text = ""
                    
                if not target_text: 
                    raise ValueError("Whisper nerozpoznal ve vzorku žádná slova (audio je buď zašuměné, tiché, nebo moc krátké).")
                
                self.progress.emit(75)
                    
                # OCHRANA PROTI PÁDU WHISPERU NA DLOUHÉM AUDIU
                try:
                    long_res = whisper_model.transcribe(y_long_16k, language=MODEL_CFG.whisper_language, word_timestamps=True)
                except Exception:
                    long_res = {"segments": []}
                
                occurrences = []
                for seg in (long_res.get("segments") or []):
                    for w in (seg.get("words") or []):
                        raw_word = w.get("word") or ""
                        if not raw_word: 
                            continue
                            
                        clean_w = raw_word.strip().lower().translate(str.maketrans('', '', string.punctuation))
                        if clean_w == target_text:
                            occurrences.append(w)
                
                idx = len(self.excluded)
                if idx < len(occurrences):
                    match = occurrences[idx]
                    result.start_f = match["start"] * sr / AUDIO_CFG.hop_length
                    result.end_f = match["end"] * sr / AUDIO_CFG.hop_length
                    result.score = 0.0
                    result.whisper_text = target_text
                else:
                    raise ValueError(f"Slovo '{target_text}' nebylo v nahrávce (již) nalezeno.")

            elif self.method == 'wav2vec':
                feat_long = get_wav2vec_features(y_long, sr)
                feat_query = get_wav2vec_features(y_query, sr)
                self.progress.emit(70)
                factor = 0.02 * sr / AUDIO_CFG.hop_length
                w2v_excl = [(s / factor, e / factor) for s, e in self.excluded]
                score, _, start_w2v, end_w2v, _ = compute_dtw(feat_query, feat_long, w2v_excl)
                result.score, result.start_f, result.end_f = score, start_w2v * factor, end_w2v * factor

            elif self.method == 'dtw':
                feat_long = get_mfcc_features(y_long, sr)
                feat_query = get_mfcc_features(y_query, sr)
                self.progress.emit(70)
                score, _, start_f, end_f, _ = compute_dtw(feat_query, feat_long, self.excluded)
                result.score, result.start_f, result.end_f = score, start_f, end_f

            elif self.method == 'pattern':
                math_spec_long = get_math_spectrogram(y_long, sr)
                math_spec_query = get_math_spectrogram(y_query, sr)
                self.progress.emit(70)
                score, _, start_f, end_f = compute_pattern_matching(math_spec_query, math_spec_long, self.excluded)
                result.score, result.start_f, result.end_f = score, start_f, end_f
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            # SPRÁVNĚ UMÍSTĚNÝ VÝPIS CHYBY DO APLIKACE
            import traceback
            full_traceback = traceback.format_exc()
            self.finished.emit(MatchResult(score=0, start_f=0, end_f=0, method=self.method, error=full_traceback))


class CorpusScannerWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object) 

    def __init__(self, long_path: str, db_file: str, method: str):
        super().__init__()
        self.long_path = long_path
        self.db_file = db_file
        self.method = method
        
        # --- Proměnné pro pauzu ---
        self.mutex = QMutex()
        self.pause_cond = QWaitCondition()
        self._is_paused = False

    def toggle_pause(self):
        self.mutex.lock()
        self._is_paused = not self._is_paused
        status = self._is_paused
        if not self._is_paused:
            self.pause_cond.wakeAll() # Probudit vlákno
        self.mutex.unlock()
        return status

    def check_pause(self):
        """Bezpečně uspí vlákno, pokud je zapnutá pauza."""
        self.mutex.lock()
        if self._is_paused:
            self.pause_cond.wait(self.mutex)
        self.mutex.unlock()

    def run(self):
        db_file_handle = None
        try:
            if self.method != 'whisper':
                self.progress.emit(5, "Připojuji HDF5 databázi...")
                db_file_handle = h5py.File(self.db_file, 'r')
                
            self.progress.emit(15, "Zpracovávám dlouhý korpus...")
            y_long, sr = load_and_prep(self.long_path)
            vis_spec_long = get_display_spectrogram(y_long, sr)
            
            results = []
            
            if self.method.startswith('whisper'):
                self.check_pause() # Kontrola před těžkým výpočtem
                
                whisper_model = ModelManager().get_whisper()
                self.progress.emit(20, "Přepisuji audio...")
                y_long_16k = librosa.resample(y_long, orig_sr=sr, target_sr=16000)
                
                try:
                    long_res = whisper_model.transcribe(y_long_16k, language=MODEL_CFG.whisper_language, word_timestamps=True)
                except Exception:
                    long_res = {"segments": []}
                
                filename_lower = os.path.basename(self.long_path).lower()
                duration_sec = len(y_long) / sr
                
                if "sample" in filename_lower or duration_sec <= 60.0:
                    raw_text = long_res.get("text", "").strip()
                else:
                    raw_text = "[Náhled odepřen: Nahrávka přesahuje 1 minutu a není typu 'sample']"
                
                self.whisper_raw_text = raw_text
                db_words_list = list(db_file_handle.keys()) if db_file_handle else []

                for seg in (long_res.get("segments") or []):
                    for w in (seg.get("words") or []):
                        self.check_pause() # <--- KONTROLA PAUZY PŘI HLEDÁNÍ SLOV
                        
                        raw_word = w.get("word") or ""
                        if not raw_word: 
                            continue
                            
                        clean_w = raw_word.strip().lower().translate(str.maketrans('', '', string.punctuation))
                        if not clean_w: continue
                        
                        whisper_prob = w.get("probability", 0.0)
                        if whisper_prob < APP_CFG.whisper_min_confidence: continue

                        start_f = w["start"] * sr / AUDIO_CFG.hop_length
                        end_f = w["end"] * sr / AUDIO_CFG.hop_length
                        score = whisper_prob

                        if self.method == 'whisper_hybrid':
                            matches = difflib.get_close_matches(clean_w, db_words_list, n=1, cutoff=0.8)
                            if not matches: 
                                continue 
                            
                            db_word = matches[0] 
                            pad_sec = 0.5
                            start_sample = max(0, int((w["start"] - pad_sec) * sr))
                            end_sample = min(len(y_long), int((w["end"] + pad_sec) * sr))
                            
                            y_slice = y_long[start_sample:end_sample]
                            slice_feat = get_mfcc_features(y_slice, sr)
                            feat_ref = db_file_handle[db_word]['mfcc'][:]
                            
                            dtw_score, _, _, _, _ = compute_dtw(feat_ref, slice_feat)
                            
                            if dtw_score == float('inf') or dtw_score > APP_CFG.dtw_early_stop_threshold:
                                continue
                                
                            clean_w, score = db_word, dtw_score
                            start_f = w["start"] * sr / AUDIO_CFG.hop_length
                            end_f = w["end"] * sr / AUDIO_CFG.hop_length

                        results.append({"word": clean_w, "score": score, "start_f": start_f, "end_f": end_f})
            else:
                long_feat = None
                if self.method == 'dtw': long_feat = get_mfcc_features(y_long, sr)
                elif self.method == 'pattern': long_feat = get_math_spectrogram(y_long, sr)
                elif self.method == 'wav2vec': long_feat = get_wav2vec_features(y_long, sr)
                
                total = len(db_file_handle.keys())
                for i, word in enumerate(db_file_handle.keys()):
                    self.check_pause() # <--- KONTROLA PAUZY PŘI HLEDÁNÍ V DB
                    
                    self.progress.emit(15 + int((i/total)*80), f"Hledám: {word}")
                    feats_group = db_file_handle[word]
                    
                    if self.method == 'wav2vec':
                        if 'wav2vec' not in feats_group: continue
                        feat_ref = feats_group['wav2vec'][:] 
                        score, _, start_f, end_f, _ = compute_dtw(feat_ref, long_feat)
                        
                        if score != float('inf') and score <= APP_CFG.dtw_early_stop_threshold:
                            factor = 0.02 * sr / AUDIO_CFG.hop_length
                            results.append({"word": word, "score": score, "start_f": start_f * factor, "end_f": end_f * factor})
                            
                    elif self.method == 'dtw': 
                        feat_ref = feats_group['mfcc'][:] 
                        score, _, start_f, end_f, _ = compute_dtw(feat_ref, long_feat)
                        
                        if score != float('inf') and score <= APP_CFG.dtw_early_stop_threshold:
                            results.append({"word": word, "score": score, "start_f": start_f, "end_f": end_f})
                            
                    else: 
                        feat_ref = feats_group['math_spec'][:]
                        score, _, start_f, end_f = compute_pattern_matching(feat_ref, long_feat)
                        
                        if score > 0.6:  
                            results.append({"word": word, "score": score, "start_f": start_f, "end_f": end_f})
                            
            results.sort(key=lambda x: x.get('score', 0.0), reverse=(self.method == 'pattern'))

            whisper_text_to_emit = getattr(self, 'whisper_raw_text', "") if self.method.startswith('whisper') else ""
            
            self.progress.emit(100, "Hotovo!")
            self.finished.emit({
                "type": "success", 
                "results": results, 
                "y_long": y_long, 
                "sr": sr, 
                "vis_spec_long": vis_spec_long,
                "whisper_text": whisper_text_to_emit
            })
            
        except Exception as e: 
            import traceback
            trace_str = traceback.format_exc()
            self.finished.emit({"type": "error", "msg": trace_str})
        finally:
            if db_file_handle: db_file_handle.close()


class BatchEvaluationWorker(QThread):
    progress = pyqtSignal(int, str)
    log_msg = pyqtSignal(str)          
    result_row = pyqtSignal(dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, long_path: str, query_path: str, gt_path: str, method: str, threshold: float):
        super().__init__()
        self.long_path = long_path
        self.query_path = query_path   
        self.gt_path = gt_path
        self.method = method
        self.threshold = threshold
    
    def toggle_pause(self):
        self.mutex.lock()
        self._is_paused = not self._is_paused
        status = self._is_paused
        if not self._is_paused:
            self.pause_cond.wakeAll() # Probudit vlákno
        self.mutex.unlock()
        return status

    def check_pause(self):
        """Pomocná metoda, kterou voláme uvnitř smyček."""
        self.mutex.lock()
        if self._is_paused:
            self.pause_cond.wait(self.mutex)
        self.mutex.unlock()

    def run(self):
        try:
            self.log_msg.emit("=== START EXHAUSTIVNÍ EVALUACE ===")
            with open(self.gt_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)

            info = sf.info(self.long_path)
            sr_stream = info.samplerate
            audio_length_hours = (info.frames / sr_stream) / 3600.0
            current_file_name = os.path.basename(self.long_path)

            target_word = os.path.splitext(os.path.basename(self.query_path))[0].lower().translate(str.maketrans('', '', string.punctuation)).strip()
            self.log_msg.emit(f"Hledám: '{target_word}'")
            
            y_q, sr_q = load_and_prep(self.query_path)
            if y_q is None: raise ValueError("Vzorek nelze načíst.")
                
            q_feat = None
            if self.method == 'wav2vec': q_feat = get_wav2vec_features(y_q, sr_q)
            elif self.method == 'dtw': q_feat = get_mfcc_features(y_q, sr_q)
            elif self.method != 'whisper': q_feat = get_math_spectrogram(y_q, sr_q)

            app_results_raw = []
            
            if self.method == 'whisper':
                self.log_msg.emit("Analyzuji nahrávku)...")
                whisper_model = ModelManager().get_whisper()
                
                chunk_duration, overlap = 120, 5
                chunk_gen = stream_audio_chunks(self.long_path, chunk_duration_sec=chunk_duration, overlap_sec=overlap, target_sr=16000)
                total_chunks = max(1, int((info.frames / sr_stream) / (chunk_duration - overlap)))
                
                current_chunk = 0
                for chunk_y, sr_chunk, global_offset_sec in chunk_gen:
                    current_chunk += 1
                    self.progress.emit(10 + int((current_chunk / total_chunks) * 70), f"Whisper Blok {current_chunk}/{total_chunks}...")
                    
                    chunk_16k = librosa.resample(chunk_y, orig_sr=sr_chunk, target_sr=16000) if sr_chunk != 16000 else chunk_y
                    
                    try:
                        res = whisper_model.transcribe(chunk_16k, language=MODEL_CFG.whisper_language, word_timestamps=True)
                    except Exception:
                        res = {"segments": []}
                    
                    for seg in (res.get("segments") or []):
                        for w in (res.get("words") or []):
                            raw_word = w.get("word") or ""
                            if not raw_word: 
                                continue
                                
                            clean_w = raw_word.strip().lower().translate(str.maketrans('', '', string.punctuation))
                            abs_start = global_offset_sec + w.get("start", 0.0)
                            
                            if clean_w == target_word:
                                app_results_raw.append({'start_s': abs_start, 'score': 0.0})
                                
                self.progress.emit(90, "Analýza dokončena.")

            else:
                chunk_duration, overlap = 120, 5
                chunk_gen = stream_audio_chunks(self.long_path, chunk_duration_sec=chunk_duration, overlap_sec=overlap, target_sr=sr_q)
                total_chunks = max(1, int((info.frames / sr_stream) / (chunk_duration - overlap)))
                
                current_chunk = 0
                for chunk_y, sr_chunk, global_offset_sec in chunk_gen:
                    current_chunk += 1
                    self.progress.emit(10 + int((current_chunk / total_chunks) * 70), f"Blok {current_chunk}/{total_chunks}...")

                    if self.method == 'wav2vec': chunk_feat = get_wav2vec_features(chunk_y, sr_chunk)
                    elif self.method == 'dtw': chunk_feat = get_mfcc_features(chunk_y, sr_chunk)
                    else: chunk_feat = get_math_spectrogram(chunk_y, sr_chunk)

                    excluded = []
                    while True:
                        if self.method == 'pattern':
                            score, _, start_f, end_f = compute_pattern_matching(q_feat, chunk_feat, excluded)
                        else:
                            v_excl = excluded
                            if self.method == 'wav2vec':
                                fct = 0.02 * sr_chunk / AUDIO_CFG.hop_length
                                v_excl = [(s / fct, e / fct) for s, e in excluded]
                            score, _, s_f, e_f, _ = compute_dtw(q_feat, chunk_feat, v_excl)
                            if self.method == 'wav2vec': start_f, end_f = s_f * fct, e_f * fct
                            else: start_f, end_f = s_f, e_f

                        if score > self.threshold or score == float('inf'): break
                        app_results_raw.append({
                            'start_s': global_offset_sec + (start_f * AUDIO_CFG.hop_length / sr_chunk),
                            'score': score
                        })
                        excluded.append((start_f, end_f))

            app_results_raw.sort(key=lambda x: x['start_s'])
            app_results = []
            for res in app_results_raw:
                if not app_results or abs(res['start_s'] - app_results[-1]['start_s']) >= 1.0:
                    app_results.append(res)
                elif res['score'] < app_results[-1]['score']:
                    app_results[-1] = res

            positives_gt = []
            for w in gt_data:
                word_text = w.get('word') or w.get('orthographic') or w.get('text')
                source_file = w.get('source_file') or w.get('filename')
                if word_text and word_text.lower().strip() == target_word:
                    if source_file and source_file != current_file_name: continue
                    t = w.get('start') or w.get('start_time') or w.get('start_s') or w.get('start_f')
                    w['start_parsed'] = float(t) if t is not None else 0.0
                    positives_gt.append(w)
            
            total_targets = len(positives_gt)
            matched_gt, scored_predictions = set(), [] 
            
            for res in app_results:
                is_tp = False
                for i, gt in enumerate(positives_gt):
                    if i not in matched_gt and abs(res['start_s'] - gt['start_parsed']) <= 0.5:
                        is_tp, _ = True, matched_gt.add(i)
                        break
                scored_predictions.append((res['score'], is_tp))
            
            scored_predictions.sort(key=lambda x: x[0])
            det_curve_data = []
            for idx in range(len(scored_predictions)):
                curr = scored_predictions[:idx + 1]
                tp = sum(1 for _, ok in curr if ok)
                fp = len(curr) - tp
                miss = (total_targets - tp) / total_targets if total_targets > 0 else 0.0
                det_curve_data.append({'threshold': curr[-1][0], 'miss': miss, 'fa_h': fp / audio_length_hours if audio_length_hours > 0 else 0.0, 'tp': tp, 'fp': fp})

            op = min(det_curve_data, key=lambda x: abs(x['fa_h'] - 1.0)) if det_curve_data else {'threshold': 0, 'miss': 1.0, 'fa_h': 0, 'tp': 0, 'fp': 0}
            self.log_msg.emit(f"=== VÝSLEDEK ===\nZásahy: {op['tp']} | Falešné (FP): {op['fp']} | Miss (FN): {op['miss']*100:.1f}%")

            self.result_row.emit({
                "word": target_word, "gt_count": total_targets, "found_count": len(app_results),
                "frr": op['miss'], "fa_h": op['fa_h'], "f1": 0.0, 
                "tp": op['tp'], "fp": op['fp'],
                "found_times": [round(res['start_s'], 2) for res in app_results],
                "gt_times": [round(gt['start_parsed'], 2) for gt in positives_gt]
            })
            self.progress.emit(100, "Hotovo!"); self.finished.emit()

        except Exception as e:
            import traceback
            trace_str = traceback.format_exc()
            self.log_msg.emit(f"❌ KRITICKÁ CHYBA:\n{trace_str}")
            self.error.emit(str(e))
            self.finished.emit()