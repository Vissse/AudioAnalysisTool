#!/usr/bin/env python3

########################################
# IMPORTS
########################################

import subprocess
import tempfile

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import PCA
import sklearn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", message=".*librosa.core.audio.__audioread_load*")

########################################
# CONFIG / DEFAULT PATHS
########################################

DEFAULT_SPECTROGRAM_BASE = Path("G:/SHAZAM/SPECTOGRAM")
DEFAULT_OUTPUT_BASE = Path("G:/SHAZAM/OUTPUT")

########################################
# VISUALIZATION HELPERS
########################################


def plot_3d_spectrogram(mel_db, times, sr, output_path=None, cmap="magma"):
    n_mels = mel_db.shape[0]
    X, Y = np.meshgrid(times, np.arange(n_mels))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Color surface according to mel_db values
    surf = ax.plot_surface(
        X, Y, mel_db,
        cmap=cmap,
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel band index")
    ax.set_zlabel("Amplitude (dB)")
    plt.tight_layout()

    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, label="Amplitude (dB)")

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"[INFO] 3D mel-spectrogram saved: {output_path}")

    plt.close(fig)


def save_2d_mel_spectrogram(mel_db, times, media_id, spectrogram_base, cmap='magma'):

    media_dir = Path(spectrogram_base) / media_id
    media_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_db, aspect='auto', origin='lower',
               extent=[times[0], times[-1], 0, mel_db.shape[0]],
               cmap=cmap)
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel band')
    plt.title(f"Mel Spectrogram: {media_id}")
    plt.tight_layout()

    png_path = media_dir / f"{media_id}_mel_spectrogram.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"[INFO] 2D mel-spectrogram saved: {png_path}")


########################################
# BASIC UTILITIES
########################################


def seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def seconds_from_frame(frame_idx, sr, hop_length):
    return (frame_idx * hop_length) / sr


########################################
# STEP 1: AUDIO LOADING, RESAMPLING, DOWNMIX
########################################

def resample_and_downmix(y, orig_sr, target_sr=22050):
    if orig_sr != target_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    if getattr(y, 'ndim', 1) > 1:
        # librosa may return shape (n_channels, n_samples) or (n_samples,) depending on loader
        if y.shape[0] > 1:
            y = np.mean(y, axis=0)
        else:
            y = y[0]
    return y, target_sr


########################################
# STEP 2: MEL SPECTROGRAM
########################################

def compute_mel_spectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=64, fmin=20, fmax=None):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    S_power = np.abs(S)**2
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_S = np.dot(mel_basis, S_power)
    mel_db = librosa.power_to_db(mel_S, ref=np.max)
    times = librosa.frames_to_time(np.arange(mel_db.shape[1]), sr=sr, hop_length=hop_length)
    return mel_db, times


########################################
# STEP 3: SLIDING WINDOW
########################################

def sliding_window_time_averages(mel_db, times, window_frames=32, stride_frames=4):
    n_mels, n_frames = mel_db.shape
    vectors = []
    start_frame_indices = []

    for start in range(0, max(1, n_frames - window_frames + 1), stride_frames):
        window = mel_db[:, start:start + window_frames]
        avg = np.mean(window, axis=1)
        vectors.append(avg)
        start_frame_indices.append(start)

    if n_frames < window_frames and n_frames > 0:
        avg = np.mean(mel_db, axis=1)
        if len(vectors) == 0:
            vectors.append(avg)
            start_frame_indices.append(0)

    return np.array(vectors, dtype=np.float32), np.array(start_frame_indices, dtype=int)


########################################
# STEP 4: DELTAS & PCA
########################################

def amplitude_delta(vecs):
    return np.diff(vecs, axis=1)


def fit_normalizers_and_pca(all_vectors, n_components=32):
    scaler_mel = StandardScaler(with_mean=True, with_std=True)
    mel_std = scaler_mel.fit_transform(all_vectors)

    # compute deltas on original (not standardized) vectors then fit scaler
    deltas = np.diff(all_vectors, axis=1)
    scaler_delta = StandardScaler(with_mean=True, with_std=True)
    delta_std = scaler_delta.fit_transform(deltas)

    combined = np.concatenate([mel_std, delta_std], axis=1)

    pca = PCA(n_components=n_components, svd_solver='randomized')
    pca_transformed = pca.fit_transform(combined)

    return scaler_mel, scaler_delta, pca


########################################
# STEP 5: PASS1
########################################

    ############################################################
    # 1) UNIVERSAL AUDIO LOADER 
    ############################################################

def load_any_audio(path, target_sr=None):
    """
    Universal loader:
    1) Try librosa
    2) Try FFmpeg → WAV fallback
    3) Try FFmpeg raw PCM → streaming loader (fastest)
    """
    path = str(path)

    # 1) Try librosa first
    try:
        y, sr = librosa.load(path, sr=target_sr, mono=False)
        return y, sr
    except:
        pass

    # 2) Try FFmpeg → WAV temp file
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp_wav = tmp.name

            cmd = [
                "ffmpeg", "-y", "-i", path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(target_sr if target_sr else 44100),
                "-ac", "1",
                tmp_wav
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            y, sr = sf.read(tmp_wav)
            return y.astype(np.float32), sr
    except:
        pass

    # 3) FFmpeg → raw PCM 32‑bit float streaming
    try:
        sr = target_sr if target_sr else 44100
        cmd = [
            "ffmpeg", "-i", path,
            "-f", "f32le",
            "-ac", "1",
            "-ar", str(sr),
            "pipe:1"
        ]
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        y = np.frombuffer(raw, dtype=np.float32)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"FFmpeg failed to decode {path}: {e}")


    ############################################################
    # 2) gather_time_averaged_vectors (any format)
    ############################################################

def gather_time_averaged_vectors(input_dir, sr, n_fft, hop_length, n_mels,
                                  window_frames, stride_frames, fmin, fmax):
    input_dir = Path(input_dir)

    # Take ALL files (any extension)
    media_files = [p for p in input_dir.rglob("*") if p.is_file()]

    all_vectors = []
    media_index = []

    for p in tqdm(media_files, desc="PASS1: extracting time‑averaged mel vectors"):
        try:
            y, orig_sr = load_any_audio(p, target_sr=None)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
            continue

        if y is None or len(y) == 0:
            continue

        # Ensure mono
        if getattr(y, 'ndim', 1) > 1:
            y = y.mean(axis=0)

        # Resample if needed
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

        # Compute mel
        mel_db, times = compute_mel_spectrogram(
            y, sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=fmin, fmax=fmax
        )

        vecs, start_frames = sliding_window_time_averages(
            mel_db, times, window_frames=window_frames, stride_frames=stride_frames
        )

        start_times = [seconds_from_frame(int(sf), sr, hop_length) for sf in start_frames]

        media_id = str(p.relative_to(input_dir)).replace(os.sep, "_")

        media_index.append({
            "media_id": media_id,
            "path": str(p),
            "n_vectors": int(vecs.shape[0]),
            "start_times": start_times,
            "duration_sec": float(len(y) / sr)
        })

        if vecs.size > 0:
            all_vectors.append(vecs)

    if len(all_vectors) == 0:
        return None, media_index

    all_vectors = np.vstack(all_vectors)
    return all_vectors, media_index


########################################
# STEP 6: PASS2
########################################

def save_human_readable_fingerprints(base_path, media_id, transformed, meta):
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    fp_json = base_path / f"{media_id}.fingerprints.json"
    with open(fp_json, "w", encoding="utf8") as f:
        json.dump({
            "media_id": media_id,
            "start_times": meta.get("start_times", []),
            "fingerprints": transformed.tolist()
        }, f, indent=2)

    fp_txt = base_path / f"{media_id}.fingerprints.txt"
    with open(fp_txt, "w", encoding="utf8") as f:
        for row in transformed:
            f.write("[" + ", ".join(f"{x:.6f}" for x in row) + "]")

    print(f"[INFO] Human-readable fingerprints saved: {fp_json}, {fp_txt}")


def transform_media_and_save(input_dir, output_dir, media_index, sr, n_fft, hop_length, n_mels,
                             window_frames, stride_frames, fmin, fmax,
                             scaler_mel, scaler_delta, pca, spectrogram_base=DEFAULT_SPECTROGRAM_BASE):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for meta in tqdm(media_index, desc="PASS2: transforming & saving fingerprints"):
        media_path = Path(meta['path'])
        media_id = meta['media_id']
        try:
            y, orig_sr = librosa.load(str(media_path), sr=None, mono=False)
        except Exception as e:
            print(f"[WARN] Failed to load {media_path}: {e}")
            continue

        if getattr(y, 'ndim', 1) > 1:
            y = np.mean(y, axis=0) if y.shape[0] > 1 else y[0]
        y, _ = resample_and_downmix(y, orig_sr, target_sr=sr)

        mel_db, times = compute_mel_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length,
                                                n_mels=n_mels, fmin=fmin, fmax=fmax)

        save_2d_mel_spectrogram(mel_db, times, media_id, spectrogram_base)

        media_spectro_dir = Path(spectrogram_base) / media_id
        media_spectro_dir.mkdir(parents=True, exist_ok=True)
        plot_3d_spectrogram(
            mel_db,
            times,
            sr,
            output_path=media_spectro_dir / f"{media_id}_mel_3d.png"
        )

        vecs, start_frames = sliding_window_time_averages(mel_db, times,
                                                          window_frames=window_frames,
                                                          stride_frames=stride_frames)

        if vecs.size == 0:
            save_human_readable_fingerprints(output_dir, media_id,
                                             np.zeros((0, pca.n_components_)), meta)
            continue

        mel_std = scaler_mel.transform(vecs)
        delta_std = scaler_delta.transform(np.diff(vecs, axis=1))
        combined = np.concatenate([mel_std, delta_std], axis=1)

        transformed = pca.transform(combined)

        save_human_readable_fingerprints(output_dir, media_id, transformed, meta)


########################################
# MAIN + CLI
########################################

def main():
    parser = argparse.ArgumentParser(description="Fingerprint pipeline implementing resample->mel->window->avg->standardize->delta->PCA")
    parser.add_argument("--input", required=False, default="./INPUT", help="Input folder path")
    parser.add_argument("--output-json", required=False, default=str(DEFAULT_OUTPUT_BASE / "dataset.json"), help="Output dataset manifest JSON")
    parser.add_argument("--fingerprints-dir", required=False, default=str(DEFAULT_OUTPUT_BASE), help="Directory to save per-media fingerprints and metadata")
    parser.add_argument("--spectrogram-base", required=False, default=str(DEFAULT_SPECTROGRAM_BASE), help="Base directory to save per-audio spectrogram folders")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--window_frames", type=int, default=32, help="Number of mel frames per sliding window")
    parser.add_argument("--stride_frames", type=int, default=4, help="Stride (in frames) for sliding window")
    parser.add_argument("--pca_components", type=int, default=32)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--fmax", type=float, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    fingerprints_dir = Path(args.fingerprints_dir)
    fingerprints_dir.mkdir(parents=True, exist_ok=True)

    spectrogram_base = Path(args.spectrogram_base)
    spectrogram_base.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    dataset = {"name": "proposed-fingerprint-dataset", "timestamp": now, "version": "v0.1", "mediaset": [], "cases": []}

    all_vectors, media_index = gather_time_averaged_vectors(
        input_dir,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        fmin=args.fmin,
        fmax=args.fmax
    )

    if media_index is None or len(media_index) == 0:
        print("[ERROR] No media files found in input folder.")
        return

    for m in media_index:
        dataset['mediaset'].append({"id": m['media_id'], "filename": m['path']})

    if all_vectors is None:
        for m in media_index:
            np.save(fingerprints_dir / f"{m['media_id']}.fingerprints.npy", np.zeros((0, args.pca_components), dtype=np.float16))
            with open(fingerprints_dir / f"{m['media_id']}.meta.json", 'w', encoding='utf8') as f:
                json.dump(m, f, indent=2)
        with open(args.output_json, 'w', encoding='utf8') as f:
            json.dump(dataset, f, indent=2)
        print("[INFO] No feature vectors extracted. Exiting.")
        return

    print("[INFO] PASS1 complete — fitting scalers and PCA on all extracted vectors...")
    scaler_mel, scaler_delta, pca = fit_normalizers_and_pca(all_vectors, n_components=args.pca_components)

    print("[INFO] PASS2 — transforming each media and saving fingerprints...")
    transform_media_and_save(
        input_dir=input_dir,
        output_dir=fingerprints_dir,
        media_index=media_index,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        fmin=args.fmin,
        fmax=args.fmax,
        scaler_mel=scaler_mel,
        scaler_delta=scaler_delta,
        pca=pca,
        spectrogram_base=spectrogram_base
    )

    for m in media_index:
        fp_path = str((fingerprints_dir / f"{m['media_id']}.fingerprints.npy").absolute())
        try:
            arr = np.load(fp_path)
            n_hashes = int(arr.shape[0])
        except Exception:
            n_hashes = 0
        case_item = {
            "name": f"case_{m['media_id']}",
            "description": "generated fingerprints",
            "sound": {"id": f"s_{m['media_id']}", "filename": m['path'], "transcription": None},
            "results": [{
                "source-id": m['media_id'],
                "start-time": "00:00:00",
                "end-time": seconds_to_hms(m['duration_sec']),
                "n_hashes": n_hashes
            }]
        }
        dataset['cases'].append(case_item)

    with open(args.output_json, 'w', encoding='utf8') as f:
        json.dump(dataset, f, indent=2)

    print(f"[INFO] Done. Dataset manifest written to {args.output_json}")
    print(f"[INFO] Fingerprints saved into {fingerprints_dir}")
    print(f"[INFO] Spectrogram folders saved into {spectrogram_base}")


if __name__ == '__main__':
    main()
