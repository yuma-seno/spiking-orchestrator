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


def run_mc_experiment(exp: MCExperiment) -> MemoryCapacityResult:
    u = generate_uniform_input(seed=exp.seed, steps=exp.steps)
    reservoir = STPReservoir(exp.cfg)
    states = reservoir.run(u)
    return memory_capacity(states, u, washout=int(exp.washout), max_delay=int(exp.max_delay), ridge=float(exp.ridge))
