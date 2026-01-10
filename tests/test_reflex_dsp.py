from __future__ import annotations

from sorch.audio.features import AudioFeatures
from sorch.reflex.dsp_reflex import ReflexConfig, ReflexDSP


def test_holdoff_blocks_retrigger():
    cfg = ReflexConfig(dt_ms=10.0, holdoff_ms=300.0, alpha=0.01)
    r = ReflexDSP(cfg)

    loud_voiced = AudioFeatures(rms=0.5, zcr=0.05, spectral_centroid_hz=500.0, voicing=True)

    t0 = 1_000_000_000
    d1 = r.update(loud_voiced, t0, loopback_active=False)
    assert d1.stop_signal is True

    # within holdoff
    t1 = t0 + int(100 * 1e6)
    d2 = r.update(loud_voiced, t1, loopback_active=False)
    assert d2.stop_signal is False
    assert d2.in_holdoff is True

    # after holdoff
    t2 = t0 + int(400 * 1e6)
    d3 = r.update(loud_voiced, t2, loopback_active=False)
    assert d3.stop_signal is True


def test_loopback_suppresses_stop():
    cfg = ReflexConfig(dt_ms=10.0, holdoff_ms=0.0, alpha=0.01)
    r = ReflexDSP(cfg)

    loud_voiced = AudioFeatures(rms=0.5, zcr=0.05, spectral_centroid_hz=500.0, voicing=True)
    d = r.update(loud_voiced, 1_000_000_000, loopback_active=True)
    assert d.stop_signal is False
    assert d.reason == "loopback_suppressed"


def test_noise_ema_updates_on_non_speech_only():
    cfg = ReflexConfig(dt_ms=10.0, holdoff_ms=0.0, alpha=0.05, noise_ema_tau_s=1.0)
    r = ReflexDSP(cfg)

    # non-speech (unvoiced, quiet)
    quiet = AudioFeatures(rms=0.01, zcr=0.4, spectral_centroid_hz=6000.0, voicing=False)
    r.update(quiet, 1_000_000_000, loopback_active=False)
    mu1 = r.state.mu_noise

    # voiced should not update noise floor
    voiced = AudioFeatures(rms=0.02, zcr=0.05, spectral_centroid_hz=800.0, voicing=True)
    r.update(voiced, 1_010_000_000, loopback_active=False)
    mu2 = r.state.mu_noise

    assert mu1 > 0.0
    assert mu2 == mu1
