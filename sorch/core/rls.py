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

    def update_block(self, X: np.ndarray, Y: np.ndarray, *, lam: float | None = None) -> np.ndarray:
        """Multi-sample RLS update (block update).

        Updates W,P using a block of samples:
        - X: [B, D]
        - Y: [B, K]

        This is useful when you want to update less frequently without subsampling.
        """
        X = np.asarray(X, dtype=self.dtype)
        if X.ndim != 2:
            raise ValueError("X must be 2D [B, D]")
        if X.shape[1] != int(self.dim):
            raise ValueError("X dim mismatch")

        Y = np.asarray(Y, dtype=self.dtype)
        if Y.ndim != 2:
            raise ValueError("Y must be 2D [B, K]")
        if Y.shape[0] != X.shape[0]:
            raise ValueError("X and Y must have same B")
        if Y.shape[1] != int(self.out_dim):
            raise ValueError("Y dim mismatch")

        B = int(X.shape[0])
        if B <= 0:
            return np.zeros((0, int(self.out_dim)), dtype=self.dtype)

        lam_eff = float(self.lam) if lam is None else float(lam)
        if not (0.0 < lam_eff <= 1.0):
            raise ValueError("lam must be in (0, 1]")

        # Prediction error for the whole block.
        E = Y - (X @ self.W)  # [B,K]

        # P X^T  (shape: [D,B])
        PXT = self.P @ X.T
        # S = lam I + X P X^T  (shape: [B,B])
        S = lam_eff * np.eye(B, dtype=self.dtype) + (X @ PXT)

        try:
            # tmp_E = S^{-1} E  (shape: [B,K])
            tmp_E = np.linalg.solve(S, E)
        except np.linalg.LinAlgError:
            # Ill-conditioned block; skip update.
            return E

        # W <- W + P X^T S^{-1} (Y - XW)
        self.W = self.W + (PXT @ tmp_E)

        # P <- (P - P X^T S^{-1} X P) / lam
        XP = X @ self.P  # [B,D]
        try:
            tmp_XP = np.linalg.solve(S, XP)  # [B,D]
        except np.linalg.LinAlgError:
            return E
        self.P = (self.P - (PXT @ tmp_XP)) / lam_eff

        return E
