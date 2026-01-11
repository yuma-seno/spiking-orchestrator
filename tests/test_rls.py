from __future__ import annotations

import numpy as np

from sorch.core.rls import RLS


def test_rls_learns_linear_mapping_smoke():
    rng = np.random.default_rng(0)
    D = 5
    K = 2

    true_W = rng.standard_normal((D, K))

    rls = RLS(dim=D, out_dim=K, lam=0.99, delta=1.0)

    # online training
    for _ in range(2000):
        x = rng.standard_normal((D,))
        y = x @ true_W
        rls.update(x, y)

    # evaluate
    X = rng.standard_normal((200, D))
    Y = X @ true_W
    Y_hat = np.stack([rls.predict(x) for x in X], axis=0)

    mse = float(np.mean((Y - Y_hat) ** 2))
    assert mse < 1e-2
