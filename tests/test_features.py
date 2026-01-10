from __future__ import annotations

import numpy as np

from sorch.audio.features import compute_features


def test_features_voiced_sine():
    sr = 48000
    n = 480  # 10ms
    t = np.arange(n, dtype=np.float32) / float(sr)
    frame = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    feats = compute_features(frame, sr)

    assert feats.rms > 0.05
    assert feats.zcr < 0.2
    assert 100.0 <= feats.spectral_centroid_hz <= 1000.0
    assert feats.voicing is True


def test_features_silence():
    sr = 48000
    frame = np.zeros(480, dtype=np.float32)
    feats = compute_features(frame, sr)

    assert feats.rms == 0.0
    assert feats.voicing is False
