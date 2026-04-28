# ==============================================================================
# audio_matcher/core/audio_utils.py
# ==============================================================================
import librosa
import numpy as np
import soundfile as sf
import torch
from config import AUDIO_CFG, MODEL_CFG
from data_types import AudioArray, Spectrogram, FeatureMatrix
from core.model_manager import ModelManager

def stream_audio_chunks(filepath: str, chunk_duration_sec: int = 300, overlap_sec: int = 5, target_sr: int = 16000):
    info = sf.info(filepath)
    orig_sr = info.samplerate
    
    chunk_frames = int(chunk_duration_sec * orig_sr)
    overlap_frames = int(overlap_sec * orig_sr)
    step_frames = chunk_frames - overlap_frames
    
    current_frame = 0
    
    for block in sf.blocks(filepath, blocksize=chunk_frames, overlap=overlap_frames, always_2d=False, fill_value=0.0):
        if block.ndim > 1:
            block = np.mean(block, axis=1)
            
        block = block.astype(np.float32)
        
        if orig_sr != target_sr:
            block = librosa.resample(block, orig_sr=orig_sr, target_sr=target_sr)
            
        global_start_sec = current_frame / orig_sr
        yield block, target_sr, global_start_sec
        
        current_frame += step_frames

def load_and_prep(file_path):
    target_sr = 22050
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    y = librosa.util.normalize(y)
    y = smart_trim_vad(y, sr) 
    return y, sr

def refine_boundaries_vad(y: AudioArray, sr: int, start_sec: float, end_sec: float, search_window_sec: float) -> tuple[float, float]:
    window_start_samp = max(0, int((start_sec - search_window_sec) * sr))
    window_end_samp = min(len(y), int((end_sec + search_window_sec) * sr))
    y_window = y[window_start_samp:window_end_samp]

    if len(y_window) == 0: return start_sec, end_sec

    target_sr = 16000
    y_window_16k = librosa.resample(y_window, orig_sr=sr, target_sr=target_sr) if sr != target_sr else y_window
    
    device = ModelManager().get_device()
    tensor_16k = torch.from_numpy(y_window_16k).float().to(device)

    if len(tensor_16k) < 512: return start_sec, end_sec

    model, utils = ModelManager().get_silero_vad()
    get_speech_timestamps = utils[0]
    
    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(
            tensor_16k, model, sampling_rate=target_sr, threshold=MODEL_CFG.silero_vad_threshold
        )

    if not speech_timestamps: return start_sec, end_sec

    vad_start_16k = speech_timestamps[0]['start']
    vad_end_16k = speech_timestamps[-1]['end']

    vad_start_sec = (window_start_samp / sr) + (vad_start_16k / target_sr)
    vad_end_sec = (window_start_samp / sr) + (vad_end_16k / target_sr)
    return vad_start_sec, vad_end_sec

def get_mfcc_features(y: AudioArray, sr: int) -> FeatureMatrix:
    min_length = 9 * AUDIO_CFG.hop_length
    if len(y) < min_length:
        pad_len = min_length - len(y)
        dither = np.random.normal(0, 1e-4, pad_len).astype(np.float32)
        y = np.concatenate([y, dither])

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=AUDIO_CFG.n_mfcc, n_fft=AUDIO_CFG.n_fft,
        hop_length=AUDIO_CFG.hop_length, n_mels=AUDIO_CFG.n_mels, fmin=AUDIO_CFG.fmin
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    return (combined - np.mean(combined, axis=1, keepdims=True)) / (np.std(combined, axis=1, keepdims=True) + 1e-9)

def get_display_spectrogram(y: AudioArray, sr: int) -> Spectrogram:
    s = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=AUDIO_CFG.n_fft, hop_length=AUDIO_CFG.hop_length,
        n_mels=AUDIO_CFG.n_mels, fmin=AUDIO_CFG.fmin
    )
    return librosa.power_to_db(s, ref=np.max)

def get_math_spectrogram(y: AudioArray, sr: int) -> Spectrogram:
    S_db = get_display_spectrogram(y, sr)
    return (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-6)

def get_wav2vec_features(y: AudioArray, sr: int) -> FeatureMatrix:
    processor, model = ModelManager().get_wav2vec()
    target_sr = MODEL_CFG.model_target_sr
    
    y_16k = librosa.resample(y, orig_sr=sr, target_sr=target_sr) if sr != target_sr else y
    
    if len(y_16k) < 512:
        pad_len = 512 - len(y_16k)
        dither = np.random.normal(0, 1e-4, pad_len).astype(np.float32)
        y_16k = np.concatenate([y_16k, dither])
    
    chunk_samples = int(MODEL_CFG.chunk_length_sec * target_sr)
    overlap_samples = int(MODEL_CFG.chunk_overlap_sec * target_sr)
    step_samples = chunk_samples - overlap_samples
    
    all_features = []
    device = ModelManager().get_device()
    
    for start_idx in range(0, len(y_16k), step_samples):
        end_idx = min(start_idx + chunk_samples, len(y_16k))
        chunk = y_16k[start_idx:end_idx]
        
        if len(chunk) < 512: 
            break
            
        inputs = processor(chunk, sampling_rate=target_sr, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        feats = outputs.last_hidden_state.squeeze(0).cpu().numpy().T
        
        frames_per_sec = feats.shape[1] / (len(chunk) / target_sr)
        
        if end_idx < len(y_16k):
            keep_frames = int((step_samples / target_sr) * frames_per_sec)
            feats = feats[:, :keep_frames]
            
        all_features.append(feats)
        if end_idx == len(y_16k): break
            
    if not all_features: return np.empty((0, 0))
        
    features_combined = np.hstack(all_features)
    return (features_combined - np.mean(features_combined, axis=1, keepdims=True)) / (np.std(features_combined, axis=1, keepdims=True) + 1e-9)

def smart_trim_vad(y: np.ndarray, sr: int) -> np.ndarray:
    silero_model, silero_utils = ModelManager().get_silero_vad()
    get_speech_timestamps = silero_utils[0]

    if silero_model is None:
        y_trimmed, _ = librosa.effects.trim(y, top_db=AUDIO_CFG.trim_top_db)
        return y_trimmed

    if sr != 16000:
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
    else:
        y_16k = y
        
    device = "cuda" if torch.cuda.is_available() and MODEL_CFG.device == "cuda" else "cpu"
    wav_tensor = torch.from_numpy(y_16k).float().to(device)

    speech_timestamps = get_speech_timestamps(
        wav_tensor,
        silero_model,
        threshold=MODEL_CFG.silero_vad_threshold,
        sampling_rate=16000
    )

    if not speech_timestamps:
        y_trimmed, _ = librosa.effects.trim(y, top_db=AUDIO_CFG.trim_top_db)
        return y_trimmed

    ratio = sr / 16000
    
    first_start_idx = int(speech_timestamps[0]['start'] * ratio)
    last_end_idx = int(speech_timestamps[-1]['end'] * ratio)

    first_start_idx = max(0, first_start_idx)
    last_end_idx = min(len(y), last_end_idx)

    return y[first_start_idx:last_end_idx]
