from __future__ import annotations

import numpy as np

from sorch.core.memory_capacity import memory_capacity


def test_memory_capacity_smoke_positive():
    rng = np.random.default_rng(0)
    T = 2000
    u = rng.uniform(-1.0, 1.0, size=T).astype(np.float32)

    # Synthetic states that contain delayed copies of u (easy MC)
    # x0(t)=u(t-1), x1(t)=u(t-2), ...
    max_delay = 20
    X = np.zeros((T, max_delay), dtype=np.float32)
    for k in range(1, max_delay + 1):
        X[k:, k - 1] = u[:-k]

    res = memory_capacity(X, u, washout=50, max_delay=max_delay, ridge=1e-6)
    assert res.mc > 10.0
    assert len(res.r2_by_delay) == max_delay
