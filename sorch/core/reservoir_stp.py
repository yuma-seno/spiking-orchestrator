from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from sorch.core.stp_lif_node import STPConfig, STP_LIFNode


@dataclass(frozen=True, slots=True)
class ReservoirConfig:
    n: int = 500
    input_scale: float = 0.5
    w_scale: float = 1.0
    sparsity: float = 0.1
    seed: int = 0

    # Recurrence source
    recurrence_source: str = "spike"  # 'spike' or 'v'

    # Simulation
    dt: float = 0.001
    tau: float = 0.02

    # Neuron
    v_threshold: float = 1.0
    v_reset: float = 0.0

    # Constant bias current added to every neuron each step.
    dc_bias: float = 0.0

    # STP
    stp: STPConfig = field(default_factory=STPConfig)


class STPReservoir:
    """Simple STP-gated reservoir for Phase 2 experiments.

    This is a pragmatic research scaffold:
    - input: scalar u(t)
    - state: spike vector s(t) in R^N
    - recurrence: random sparse weights on spikes

    For stable MC experiments, this keeps everything on CPU and uses float32.
    """

    def __init__(self, cfg: ReservoirConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)

        # Input weights
        self.w_in = torch.from_numpy(
            (cfg.input_scale * rng.standard_normal((cfg.n,), dtype=np.float32))
        ).to(dtype=torch.float32)

        # Recurrent weights (sparse mask)
        mask = rng.random((cfg.n, cfg.n)) < float(cfg.sparsity)
        w = rng.standard_normal((cfg.n, cfg.n), dtype=np.float32)
        w = w * mask.astype(np.float32)
        # Scale by 1/sqrt(N*sparsity) for rough variance control.
        denom = np.sqrt(max(1.0, cfg.n * max(cfg.sparsity, 1e-6)))
        w = (cfg.w_scale / denom) * w
        self.w_rec = torch.from_numpy(w).to(dtype=torch.float32)

        self.node = STP_LIFNode(
            dt=cfg.dt,
            tau=cfg.tau,
            stp=cfg.stp,
            v_threshold=float(cfg.v_threshold),
            v_reset=float(cfg.v_reset),
            dc_bias=float(cfg.dc_bias),
        )

        self._s = torch.zeros((cfg.n,), dtype=torch.float32)
        self._v = torch.zeros((cfg.n,), dtype=torch.float32)
        self.last_spike_rate: float = 0.0

    def reset(self):
        self.node.reset()
        self._s = torch.zeros((self.cfg.n,), dtype=torch.float32)
        self._v = torch.zeros((self.cfg.n,), dtype=torch.float32)
        self.last_spike_rate = 0.0

    @torch.no_grad()
    def step(self, u_t: float) -> torch.Tensor:
        # Current = input + recurrent
        inp = (self.w_in * float(u_t)).to(dtype=torch.float32)
        if self.cfg.recurrence_source == "v":
            rec_state = self._v
        else:
            rec_state = self._s
        rec = (self.w_rec @ rec_state).to(dtype=torch.float32)
        x = inp + rec
        s = self.node(x)
        self._s = s.to(torch.float32)

        assert self.node.v is not None
        self._v = self.node.v.to(torch.float32)
        # For MC readout, membrane potential is a denser, more informative state.
        return self._v

    @torch.no_grad()
    def run(self, u: np.ndarray) -> np.ndarray:
        self.reset()
        u = np.asarray(u).reshape(-1)
        states = np.zeros((u.size, self.cfg.n), dtype=np.float32)
        total_spikes = 0.0
        for t in range(u.size):
            s = self.step(float(u[t]))
            states[t, :] = s.cpu().numpy()
            total_spikes += float(self._s.sum().cpu().item())
        self.last_spike_rate = total_spikes / float(max(1, u.size * self.cfg.n))
        return states
