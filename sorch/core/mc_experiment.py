from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sorch.core.memory_capacity import MemoryCapacityResult, memory_capacity
from sorch.core.reservoir_stp import ReservoirConfig, STPReservoir


@dataclass(frozen=True, slots=True)
class MCExperiment:
    cfg: ReservoirConfig
    steps: int
    washout: int
    max_delay: int
    ridge: float = 1e-3
    seed: int = 0


def generate_uniform_input(*, seed: int, steps: int, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.uniform(low=float(low), high=float(high), size=int(steps)).astype(np.float32)


def _generate_on_off_gate(
    *,
    rng: np.random.Generator,
    steps: int,
    on_min: int,
    on_max: int,
    off_min: int,
    off_max: int,
    start: str = "on",
) -> np.ndarray:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if on_min <= 0 or on_max <= 0 or off_min <= 0 or off_max <= 0:
        raise ValueError("durations must be > 0")
    if on_min > on_max:
        raise ValueError("on_min must be <= on_max")
    if off_min > off_max:
        raise ValueError("off_min must be <= off_max")
    if start not in {"on", "off"}:
        raise ValueError("start must be 'on' or 'off'")

    gate = np.zeros(int(steps), dtype=np.float32)
    is_on = start == "on"
    i = 0
    while i < steps:
        if is_on:
            dur = int(rng.integers(int(on_min), int(on_max) + 1))
            gate[i : min(steps, i + dur)] = 1.0
        else:
            dur = int(rng.integers(int(off_min), int(off_max) + 1))
        i += dur
        is_on = not is_on
    return gate


def generate_input(
    *,
    seed: int,
    steps: int,
    mode: str = "uniform",
    input_bias: float = 0.0,
    low: float = -1.0,
    high: float = 1.0,
    std: float = 0.5,
    clip: float = 1.0,
    tempo_on_min: int = 20,
    tempo_on_max: int = 200,
    tempo_off_min: int = 20,
    tempo_off_max: int = 200,
    tempo_amp: float = 1.0,
) -> np.ndarray:
    """Generate 1D input signal u(t) for MC experiments.

    Modes:
      - uniform: U(low, high)
      - gaussian: N(0, std)
      - burst: gaussian * gate (on/off)
      - convo: bursty input intended to mimic "talk/silence" tempo

    Notes:
      - This function is deterministic for a given seed.
      - If clip > 0, values are clipped into [-clip, +clip].
      - input_bias is added after generation (before optional casting).
    """
    if steps <= 0:
        raise ValueError("steps must be > 0")

    rng = np.random.default_rng(int(seed))
    mode = str(mode)

    if mode == "uniform":
        u = rng.uniform(low=float(low), high=float(high), size=int(steps)).astype(np.float32)
    elif mode == "gaussian":
        u = rng.normal(loc=0.0, scale=float(std), size=int(steps)).astype(np.float32)
    elif mode in {"burst", "convo"}:
        gate = _generate_on_off_gate(
            rng=rng,
            steps=int(steps),
            on_min=int(tempo_on_min),
            on_max=int(tempo_on_max),
            off_min=int(tempo_off_min),
            off_max=int(tempo_off_max),
            start="on",
        )
        base = rng.normal(loc=0.0, scale=float(std), size=int(steps)).astype(np.float32)
        u = (float(tempo_amp) * gate * base).astype(np.float32)
    else:
        raise ValueError(f"unknown input mode: {mode}")

    if float(input_bias) != 0.0:
        u = (u + float(input_bias)).astype(np.float32)
    if float(clip) > 0:
        u = np.clip(u, -float(clip), float(clip)).astype(np.float32)
    return u


def run_mc_experiment(exp: MCExperiment) -> MemoryCapacityResult:
    u = generate_input(seed=exp.seed, steps=exp.steps, mode="uniform")
    reservoir = STPReservoir(exp.cfg)
    states = reservoir.run(u)
    return memory_capacity(states, u, washout=int(exp.washout), max_delay=int(exp.max_delay), ridge=float(exp.ridge))
