from __future__ import annotations

import numpy as np

from sorch.core.reservoir_stp import ReservoirConfig, STPReservoir
from sorch.core.stp_lif_node import STPConfig


def test_run_traces_shapes_and_ranges() -> None:
    cfg = ReservoirConfig(
        n=16,
        input_scale=0.5,
        w_scale=1.0,
        sparsity=0.2,
        seed=0,
        recurrence_source="spike",
        v_threshold=0.25,
        v_reset=0.0,
        dc_bias=0.25,
        stp=STPConfig(U=0.2, tau_F=0.2, tau_D=1.0),
    )
    res = STPReservoir(cfg)
    u = np.linspace(-1.0, 1.0, 200, dtype=np.float32)

    traces = res.run_traces(u, record=("v", "spike", "u", "r", "eff"))

    assert set(traces.keys()) == {"v", "spike", "u", "r", "eff"}
    for k, x in traces.items():
        assert x.shape == (u.size, cfg.n), k
        assert x.dtype == np.float32, k

    assert 0.0 <= res.last_spike_rate <= 1.0

    eff = traces["eff"]
    assert float(eff.min()) >= 0.0
    assert float(eff.max()) <= 1.0
