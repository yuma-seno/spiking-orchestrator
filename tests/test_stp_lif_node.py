from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
spikingjelly = pytest.importorskip("spikingjelly")

from sorch.core.stp_lif_node import STPConfig, STP_LIFNode


def test_stp_states_initialize_and_update():
    node = STP_LIFNode(
        dt=0.001,
        tau=0.02,
        v_threshold=0.5,
        stp=STPConfig(U=0.2, tau_F=0.2, tau_D=1.0),
    )

    x = torch.zeros(4)
    s0 = node(x)
    assert s0.shape == x.shape
    assert node.u is not None and node.r is not None and node.v is not None

    # Drive spiking with strong input for multiple steps
    x_hi = torch.full((4,), 10.0)
    spikes = []
    for _ in range(50):
        spikes.append(node(x_hi))
    s1 = torch.stack(spikes).sum(dim=0)
    assert torch.any(s1 > 0)

    u1 = node.u.clone()
    r1 = node.r.clone()

    # Another step should relax a bit + apply spike effects again
    s2 = node(x_hi)
    assert node.u.shape == x.shape
    assert node.r.shape == x.shape

    # u should stay within [0,1], r within [0,1]
    assert torch.all((node.u >= 0) & (node.u <= 1.0))
    assert torch.all((node.r >= 0) & (node.r <= 1.0))

    # Not a strict monotonicity test (depends on spike), but states should be finite.
    assert torch.isfinite(u1).all() and torch.isfinite(r1).all()
    assert torch.isfinite(node.u).all() and torch.isfinite(node.r).all()
