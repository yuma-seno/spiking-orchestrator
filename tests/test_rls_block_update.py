from __future__ import annotations

import numpy as np

from sorch.core.rls import RLS


def test_rls_block_update_matches_sequential_for_lam1():
    rng = np.random.default_rng(0)
    T = 200
    D = 8
    K = 3

    X = rng.standard_normal((T, D))
    true_W = rng.standard_normal((D, K))
    Y = X @ true_W

    # Sequential RLS
    rls_seq = RLS(dim=D, out_dim=K, lam=1.0, delta=1.0)
    for t in range(T):
        rls_seq.update(X[t], Y[t])

    # Block RLS (same data, one big block)
    rls_blk = RLS(dim=D, out_dim=K, lam=1.0, delta=1.0)
    rls_blk.update_block(X, Y)

    # They won't be bit-identical, but should be close.
    denom = float(np.linalg.norm(rls_seq.W) + 1e-12)
    rel = float(np.linalg.norm(rls_blk.W - rls_seq.W) / denom)
    assert rel < 1e-6
