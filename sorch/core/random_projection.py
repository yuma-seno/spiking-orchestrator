from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RandomProjection:
    """Fixed random projection matrix.

    Uses a dense Gaussian matrix scaled by 1/sqrt(out_dim) (Johnson–Lindenstrauss style).
    """

    in_dim: int
    out_dim: int
    seed: int = 0

    def matrix(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        W = rng.standard_normal((self.in_dim, self.out_dim), dtype=np.float32)
        W /= np.sqrt(float(self.out_dim))
        return W

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.shape[-1] != self.in_dim:
            raise ValueError("last dim mismatch")
        return X @ self.matrix()
