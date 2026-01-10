from __future__ import annotations

from dataclasses import dataclass

import torch
import spikingjelly.activation_based.neuron as neuron
import spikingjelly.activation_based.surrogate as surrogate


@dataclass(slots=True)
class STPConfig:
    """Tsodyks–Markram short-term plasticity parameters.

    Notes:
    - u, x are per-neuron (or per-feature) states.
    - Between spikes: u relaxes to U, x relaxes to 1.
    - On spike: u increases, x decreases.
    - Effective synaptic efficacy is u * x.
    """

    U: float = 0.2
    tau_F: float = 0.2  # seconds
    tau_D: float = 1.0  # seconds


class STP_LIFNode(neuron.BaseNode):
    """LIF neuron with Tsodyks–Markram STP gating on input current.

    This implements Phase 2 Step 2.1 from the spec: a custom neuron derived from
    SpikingJelly `BaseNode` that holds (u, x) as internal states.

    Forward contract:
    - Input `x` is a current / pre-activation tensor.
    - We compute `x_eff = x * (u * r)` where `r` is the available resources state.
    - Then we run a standard LIF membrane update + spike generation.

    The STP update is driven by the neuron spike (post-synaptic). This is a
    pragmatic choice for Phase 2 prototyping; for synapse-level STP you would
    typically drive STP with pre-synaptic spikes.
    """

    def __init__(
        self,
        *,
        dt: float = 0.001,
        tau: float = 0.02,
        decay_input: bool = True,
        stp: STPConfig | None = None,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_function=surrogate.Sigmoid(alpha=4.0, spiking=True),
        detach_reset: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        store_v_seq: bool = False,
    ):
        super().__init__(
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )

        self.dt = float(dt)
        self.tau = float(tau)
        self.decay_input = bool(decay_input)

        self.stp = stp or STPConfig()
        if not (0.0 < self.stp.U <= 1.0):
            raise ValueError("STPConfig.U must be in (0, 1]")
        if self.stp.tau_F <= 0 or self.stp.tau_D <= 0:
            raise ValueError("STP time constants must be positive")

        # Internal states (initialized lazily to match input shape/device).
        self.u: torch.Tensor | None = None
        self.r: torch.Tensor | None = None  # resources, called x in some papers

        # Membrane potential (SpikingJelly expects `self.v`).
        self.v: torch.Tensor | None = None

    def reset(self):
        super().reset()
        self.u = None
        self.r = None
        self.v = None

    def _init_states(self, like: torch.Tensor):
        if self.u is None or self.r is None:
            self.u = torch.full_like(like, float(self.stp.U))
            self.r = torch.ones_like(like)
        if self.v is None:
            self.v = torch.full_like(like, float(self.v_reset))

    def _stp_relax(self):
        assert self.u is not None and self.r is not None
        # Discrete-time exponential relaxation.
        aF = self.dt / self.stp.tau_F
        aD = self.dt / self.stp.tau_D
        self.u = self.u + (float(self.stp.U) - self.u) * aF
        self.r = self.r + (1.0 - self.r) * aD

    def _stp_on_spike(self, spike: torch.Tensor):
        assert self.u is not None and self.r is not None
        U = float(self.stp.U)
        # u <- u + U(1-u)
        self.u = self.u + (U * (1.0 - self.u)) * spike
        # r <- r * (1 - u)
        self.r = self.r * (1.0 - self.u * spike)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode != "s":
            raise NotImplementedError("Phase 2 prototype supports step_mode='s' only")

        self._init_states(x)
        assert self.u is not None and self.r is not None and self.v is not None

        # Relax encourages stable long runs.
        self._stp_relax()

        # Gate the input by STP efficacy.
        efficacy = (self.u * self.r).clamp(min=0.0, max=1.0)
        x_eff = x * efficacy

        # LIF membrane update.
        if self.decay_input:
            dv = (-(self.v - float(self.v_reset)) + x_eff) * (self.dt / self.tau)
        else:
            dv = (-(self.v - float(self.v_reset))) * (self.dt / self.tau) + x_eff

        self.v = self.v + dv

        # Spike & reset.
        spike = self.surrogate_function(self.v - float(self.v_threshold))
        if self.detach_reset:
            spike_detached = spike.detach()
        else:
            spike_detached = spike

        if self.v_reset is None:
            # Soft reset
            self.v = self.v - spike_detached * float(self.v_threshold)
        else:
            # Hard reset
            self.v = (1.0 - spike_detached) * self.v + spike_detached * float(self.v_reset)

        # Drive STP with spikes.
        self._stp_on_spike(spike_detached)

        return spike
