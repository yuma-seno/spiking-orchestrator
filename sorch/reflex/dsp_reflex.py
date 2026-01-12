from __future__ import annotations

import math
from dataclasses import dataclass

from sorch.audio.features import AudioFeatures


@dataclass(slots=True)
class ReflexConfig:
    # Timing
    dt_ms: float = 10.0
    holdoff_ms: float = 300.0

    # Adaptive thresholding
    noise_ema_tau_s: float = 1.0
    alpha: float = 0.03  # additive margin over estimated noise floor

    # Ensemble gates
    max_zcr: float = 0.25
    min_centroid_hz: float = 80.0
    max_centroid_hz: float = 5000.0


@dataclass(slots=True)
class ReflexState:
    mu_noise: float = 0.0
    holdoff_until_ns: int = 0


@dataclass(frozen=True, slots=True)
class ReflexDecision:
    stop_signal: bool
    interrupt_cue: bool
    cue_strength: float
    threshold: float
    in_holdoff: bool
    reason: str


class ReflexDSP:
    """Classical DSP-first reflex logic.

    Responsibilities (Phase 1):
    - adaptive energy threshold (EMA noise floor)
    - hold-off to prevent chattering
    - loopback/echo suppression hook
    """

    def __init__(self, config: ReflexConfig | None = None, state: ReflexState | None = None):
        self.config = config or ReflexConfig()
        self.state = state or ReflexState()

        dt_s = max(float(self.config.dt_ms) / 1000.0, 1e-6)
        tau = max(float(self.config.noise_ema_tau_s), 1e-3)
        # Equivalent to an exponential moving average with time constant tau.
        self._gamma = 1.0 - math.exp(-dt_s / tau)

    def update(
        self,
        features: AudioFeatures,
        now_ns: int,
        *,
        loopback_active: bool = False,
    ) -> ReflexDecision:
        in_holdoff = now_ns < int(self.state.holdoff_until_ns)
        threshold = float(self.state.mu_noise + self.config.alpha)

        centroid_ok = (self.config.min_centroid_hz <= features.spectral_centroid_hz <= self.config.max_centroid_hz)
        zcr_ok = (features.zcr <= self.config.max_zcr)
        over_threshold = (features.rms > threshold)

        # Soft Interrupt is an observation cue, not a stop command.
        # Keep it independent from holdoff/stop gating so downstream policy can learn.
        interrupt_cue = bool((not loopback_active) and over_threshold and (features.voicing or (zcr_ok and centroid_ok)))
        cue_strength = float(max(0.0, features.rms - threshold))

        # Estimate noise floor only on (likely) non-speech frames.
        likely_non_speech = (not features.voicing) and (features.rms <= threshold)
        if likely_non_speech:
            self.state.mu_noise = float((1.0 - self._gamma) * self.state.mu_noise + self._gamma * features.rms)
            threshold = float(self.state.mu_noise + self.config.alpha)

        if in_holdoff:
            return ReflexDecision(
                stop_signal=False,
                interrupt_cue=interrupt_cue,
                cue_strength=cue_strength,
                threshold=threshold,
                in_holdoff=True,
                reason="holdoff",
            )

        if loopback_active:
            return ReflexDecision(
                stop_signal=False,
                interrupt_cue=False,
                cue_strength=0.0,
                threshold=threshold,
                in_holdoff=False,
                reason="loopback_suppressed",
            )

        if over_threshold and features.voicing and zcr_ok and centroid_ok:
            holdoff_ns = int(self.config.holdoff_ms * 1_000_000)
            self.state.holdoff_until_ns = int(now_ns + holdoff_ns)
            return ReflexDecision(
                stop_signal=True,
                interrupt_cue=interrupt_cue,
                cue_strength=cue_strength,
                threshold=threshold,
                in_holdoff=False,
                reason="stop",
            )

        return ReflexDecision(
            stop_signal=False,
            interrupt_cue=interrupt_cue,
            cue_strength=cue_strength,
            threshold=threshold,
            in_holdoff=False,
            reason="no_trigger",
        )
