from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioFeatures:
    rms: float
    zcr: float
    spectral_centroid_hz: float
    voicing: bool


def pcm16_bytes_to_float32(mono_pcm16: bytes) -> np.ndarray:
    """Convert little-endian PCM16 mono bytes to float32 in [-1, 1]."""
    x = np.frombuffer(mono_pcm16, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return x
    return x / 32768.0


@lru_cache(maxsize=64)
def _hann_window(n: int) -> np.ndarray:
    # Cache small windows by frame size to avoid per-frame allocation.
    return np.hanning(n).astype(np.float32)


def compute_features(frame: np.ndarray, sample_rate: int) -> AudioFeatures:
    """Compute a small, low-latency feature ensemble.

    `frame` is expected to be mono float32/float64 in [-1, 1].
    """
    if frame.size == 0:
        return AudioFeatures(rms=0.0, zcr=0.0, spectral_centroid_hz=0.0, voicing=False)

    x = np.asarray(frame)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)

    # Remove DC to stabilize spectral features on small frames.
    x = x - float(np.mean(x))

    # RMS energy
    mean_square = float(np.mean(x * x))
    if mean_square <= 1e-12:
        return AudioFeatures(rms=0.0, zcr=0.0, spectral_centroid_hz=0.0, voicing=False)
    rms = float(np.sqrt(mean_square))

    # Zero-crossing rate (normalized by samples)
    signs = np.signbit(x)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if x.size >= 2 else 0.0

    # Spectral centroid
    # Windowing reduces leakage so centroid is meaningful even on 5-20ms frames.
    w = _hann_window(int(x.size))
    X = np.fft.rfft(x * w)
    mag2 = (np.abs(X).astype(np.float32, copy=False)) ** 2
    mag_sum = float(np.sum(mag2))
    if mag_sum <= 1e-12:
        centroid_hz = 0.0
    else:
        freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate)).astype(np.float32, copy=False)
        centroid_hz = float(np.sum(freqs * mag2) / mag_sum)

    # A very cheap voicing heuristic:
    # - voiced speech tends to have lower ZCR
    # - and non-trivial energy
    voicing = bool((rms >= 0.01) and (zcr <= 0.20) and (centroid_hz <= 3500.0))

    return AudioFeatures(rms=rms, zcr=zcr, spectral_centroid_hz=centroid_hz, voicing=voicing)
