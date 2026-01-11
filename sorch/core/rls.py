from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class RLS:
    """Recursive Least Squares (RLS) for online linear regression.

    Learns weights W such that y ~= x @ W.

    - x: [D]
    - y: [K]
    - W: [D, K]

    Uses forgetting factor lambda (0<lambda<=1) and Tikhonov regularization delta.

    References:
      Standard RLS update with covariance matrix P.
    """

    dim: int
    out_dim: int = 1
    lam: float = 0.995
    delta: float = 1.0
    dtype: type = np.float64

    W: np.ndarray = field(init=False, repr=False)
    P: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be > 0")
        if self.out_dim <= 0:
            raise ValueError("out_dim must be > 0")
        if not (0.0 < float(self.lam) <= 1.0):
            raise ValueError("lam must be in (0, 1]")
        if float(self.delta) <= 0.0:
            raise ValueError("delta must be > 0")

        self.W = np.zeros((int(self.dim), int(self.out_dim)), dtype=self.dtype)
        # P starts as (1/delta) * I
        self.P = (1.0 / float(self.delta)) * np.eye(int(self.dim), dtype=self.dtype)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=self.dtype).reshape(-1)
        if x.shape[0] != int(self.dim):
            raise ValueError("x dim mismatch")
        return x @ self.W

    def update(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """One-step RLS update. Returns prediction error (y - y_hat)."""
        x = np.asarray(x, dtype=self.dtype).reshape(-1, 1)  # [D,1]
        if x.shape[0] != int(self.dim):
            raise ValueError("x dim mismatch")

        y = np.asarray(y, dtype=self.dtype).reshape(1, -1)  # [1,K]
        if y.shape[1] != int(self.out_dim):
            raise ValueError("y dim mismatch")

        y_hat = (x.T @ self.W).reshape(1, -1)  # [1,K]
        err = y - y_hat  # [1,K]

        Px = self.P @ x  # [D,1]
        denom = float(self.lam) + float((x.T @ Px).item())
        if denom <= 1e-12:
            # Extremely ill-conditioned step; skip update
            return err.reshape(-1)

        k = Px / denom  # [D,1]

        # Update weights
        self.W = self.W + (k @ err)  # [D,K]

        # Joseph-form-like update for numerical stability
        self.P = (self.P - (k @ x.T @ self.P)) / float(self.lam)
        return err.reshape(-1)
