from __future__ import annotations

import io
import os
import random
from typing import Optional

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/resp_ai_numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/resp_ai_mplconfig")

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from resp_ai.config import AudioConfig

matplotlib.use("Agg")


def trim_normalize_signal(signal: np.ndarray, config: AudioConfig) -> np.ndarray:
    signal, _ = librosa.effects.trim(signal, top_db=config.trim_top_db)
    if len(signal) == 0:
        signal = np.zeros(1, dtype=np.float32)
    signal = librosa.util.normalize(signal)
    return signal.astype(np.float32)


def load_trimmed_audio(path: str, config: AudioConfig) -> np.ndarray:
    signal, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    return trim_normalize_signal(signal, config)


def load_audio(path: str, config: AudioConfig) -> np.ndarray:
    signal = load_trimmed_audio(path, config)
    return fit_audio_length(signal, config.target_samples)


def load_audio_bytes(blob: bytes, config: AudioConfig) -> np.ndarray:
    signal, sr = sf.read(io.BytesIO(blob), dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    signal = librosa.resample(signal, orig_sr=sr, target_sr=config.sample_rate)
    signal = trim_normalize_signal(signal, config)
    return fit_audio_length(signal, config.target_samples)


def prepare_signal(signal: np.ndarray, config: AudioConfig) -> np.ndarray:
    signal = trim_normalize_signal(signal, config)
    return fit_audio_length(signal, config.target_samples)


def fit_audio_length(signal: np.ndarray, target_samples: int) -> np.ndarray:
    if len(signal) == target_samples:
        return signal

    if len(signal) > target_samples:
        start = select_best_window_start(signal, target_samples)
        return signal[start:start + target_samples]

    pad_total = target_samples - len(signal)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(signal, (pad_left, pad_right), mode="constant")


def select_best_window_start(signal: np.ndarray, target_samples: int) -> int:
    if len(signal) <= target_samples:
        return 0

    stride = max(target_samples // 4, 1)
    candidate_starts = list(range(0, len(signal) - target_samples + 1, stride))
    last_start = len(signal) - target_samples
    if candidate_starts[-1] != last_start:
        candidate_starts.append(last_start)

    best_start = 0
    best_score = -1.0
    for start in candidate_starts:
        window = signal[start:start + target_samples]
        energy = float(np.sqrt(np.mean(np.square(window))) + np.max(np.abs(window)) * 0.25)
        if energy > best_score:
            best_score = energy
            best_start = start
    return best_start


def generate_window_starts(
    signal_length: int,
    target_samples: int,
    overlap: float = 0.5,
    max_windows: Optional[int] = None,
) -> list[int]:
    if signal_length <= target_samples:
        return [0]

    stride = max(int(target_samples * (1.0 - overlap)), 1)
    starts = list(range(0, signal_length - target_samples + 1, stride))
    last_start = signal_length - target_samples
    if starts[-1] != last_start:
        starts.append(last_start)

    if max_windows is not None and len(starts) > max_windows:
        indices = np.linspace(0, len(starts) - 1, num=max_windows, dtype=int)
        starts = [starts[index] for index in indices]

    return sorted(set(starts))


def extract_window_batch_from_path(
    path: str,
    config: AudioConfig,
    overlap: float = 0.5,
    max_windows: Optional[int] = 5,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    signal = load_trimmed_audio(path, config)
    starts = generate_window_starts(len(signal), config.target_samples, overlap=overlap, max_windows=max_windows)

    images: list[np.ndarray] = []
    metadata: list[dict[str, float | int]] = []
    for window_index, start in enumerate(starts):
        window = fit_audio_length(signal[start:start + config.target_samples], config.target_samples)
        images.append(compute_logmel_image(window, config))
        metadata.append(
            {
                "window_index": window_index,
                "start_sec": round(start / max(config.sample_rate, 1), 4),
                "duration_sec": round(config.target_samples / max(config.sample_rate, 1), 4),
            }
        )

    return np.stack(images, axis=0).astype(np.float32), metadata


def _time_shift(signal: np.ndarray, max_fraction: float) -> np.ndarray:
    if len(signal) == 0 or max_fraction <= 0:
        return signal

    max_shift = max(int(len(signal) * max_fraction), 1)
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return signal

    shifted = np.zeros_like(signal)
    if shift > 0:
        shifted[shift:] = signal[:-shift]
    else:
        shifted[:shift] = signal[-shift:]
    return shifted


def augment_audio(signal: np.ndarray, sample_rate: int, strength: float = 1.0, num_ops: int = 1) -> np.ndarray:
    augmented = signal.copy()
    strength = float(np.clip(strength, 0.6, 1.5))
    num_ops = max(int(num_ops), 1)

    operations = ["noise", "stretch", "pitch", "gain", "shift"]
    chosen = random.sample(operations, k=min(num_ops, len(operations)))

    for operation in chosen:
        if operation == "noise":
            augmented = augmented + np.random.uniform(0.001, 0.004) * strength * np.random.randn(len(augmented))
        elif operation == "stretch":
            stretch_margin = 0.04 + (0.04 * (strength - 0.6))
            rate = np.random.uniform(1.0 - stretch_margin, 1.0 + stretch_margin)
            augmented = librosa.effects.time_stretch(augmented, rate=rate)
        elif operation == "pitch":
            max_steps = 0.35 + (0.55 * (strength - 0.6))
            steps = np.random.uniform(-max_steps, max_steps)
            augmented = librosa.effects.pitch_shift(augmented, sr=sample_rate, n_steps=steps)
        elif operation == "gain":
            gain_margin = 0.08 + (0.10 * (strength - 0.6))
            augmented = augmented * np.random.uniform(1.0 - gain_margin, 1.0 + gain_margin)
        elif operation == "shift":
            shift_fraction = 0.02 + (0.06 * (strength - 0.6))
            augmented = _time_shift(augmented, max_fraction=shift_fraction)

    augmented = np.clip(augmented, -1.0, 1.0)
    return augmented


def compute_logmel_image(signal: np.ndarray, config: AudioConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    channels = [mel_db]
    if config.use_deltas:
        channels.append(librosa.feature.delta(mel_db))
        channels.append(librosa.feature.delta(mel_db, order=2))

    stacked = np.stack([normalize_image(channel) for channel in channels], axis=-1)
    if stacked.shape[-1] == 1:
        stacked = np.repeat(stacked, 3, axis=-1)
    return stacked.astype(np.float32)


def normalize_image(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    array -= array.min()
    denom = array.max() + 1e-6
    array /= denom
    return array


def preprocess_path(
    path: str,
    config: AudioConfig,
    training: bool = False,
    augmentation_strength: float = 1.0,
    augmentation_ops: int = 1,
) -> np.ndarray:
    signal = load_audio(path, config)
    if training:
        signal = fit_audio_length(
            augment_audio(signal, config.sample_rate, strength=augmentation_strength, num_ops=augmentation_ops),
            config.target_samples,
        )
    return compute_logmel_image(signal, config)


def preprocess_bytes(blob: bytes, config: AudioConfig) -> np.ndarray:
    signal = load_audio_bytes(blob, config)
    return compute_logmel_image(signal, config)


def save_preprocessed_clip(src_path: str, dst_path: str, config: AudioConfig) -> dict:
    signal, _ = librosa.load(src_path, sr=config.sample_rate, mono=True)
    trimmed, _ = librosa.effects.trim(signal, top_db=config.trim_top_db)
    original_duration = len(signal) / max(config.sample_rate, 1)
    trimmed_duration = len(trimmed) / max(config.sample_rate, 1)
    prepared = prepare_signal(signal, config)
    sf.write(dst_path, prepared, config.sample_rate)
    return {
        "original_duration_sec": round(original_duration, 4),
        "trimmed_duration_sec": round(trimmed_duration, 4),
        "processed_duration_sec": round(len(prepared) / max(config.sample_rate, 1), 4),
    }


def render_feature_heatmap(image: np.ndarray, title: Optional[str] = None) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 3), dpi=140)
    ax.imshow(image[:, :, 0], origin="lower", aspect="auto", cmap="magma")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel Bin")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()
