from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MemoryCapacityResult:
    max_delay: int
    mc: float
    r2_by_delay: list[float]


def ridge_fit(X: np.ndarray, Y: np.ndarray, *, ridge: float = 1e-3) -> np.ndarray:
    """Fit W for Y ≈ X @ W with Tikhonov regularization.

    X: [T, D]
    Y: [T, K]
    returns W: [D, K]
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same T")

    # Solve (X^T X + ridge I) W = X^T Y
    XtX = X.T @ X
    D = XtX.shape[0]
    XtX = XtX + ridge * np.eye(D, dtype=XtX.dtype)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY)
    return W


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination for 1D arrays."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return 0.0
    var = float(np.var(y_true))
    if var <= 1e-12:
        return 0.0
    mse = float(np.mean((y_true - y_pred) ** 2))
    return max(0.0, 1.0 - mse / var)


def memory_capacity(
    states: np.ndarray,
    input_signal: np.ndarray,
    *,
    washout: int,
    max_delay: int,
    ridge: float = 1e-3,
) -> MemoryCapacityResult:
    """Compute memory capacity (MC) from reservoir states.

    Standard recipe:
    - drive reservoir with scalar input u(t)
    - collect state vector x(t)
    - train linear readout to reconstruct u(t-k) for k=1..max_delay
    - MC = sum_k R^2_k

    states: [T, N]
    input_signal: [T]
    """
    X = np.asarray(states)
    u = np.asarray(input_signal).reshape(-1)
    if X.ndim != 2:
        raise ValueError("states must be [T, N]")
    if u.ndim != 1:
        raise ValueError("input_signal must be [T]")
    if X.shape[0] != u.shape[0]:
        raise ValueError("states and input_signal must have same T")

    T = X.shape[0]
    if washout < 0 or washout >= T:
        raise ValueError("invalid washout")
    if max_delay <= 0:
        raise ValueError("max_delay must be positive")

    # Align so targets exist for all delays.
    start = washout + max_delay
    if start >= T:
        raise ValueError("T too small for washout+max_delay")

    X_eff = X[start:, :]
    # Add bias
    X_eff = np.concatenate([X_eff, np.ones((X_eff.shape[0], 1), dtype=X_eff.dtype)], axis=1)

    Y = np.stack([u[start - k : T - k] for k in range(1, max_delay + 1)], axis=1)

    W = ridge_fit(X_eff, Y, ridge=ridge)
    Y_hat = X_eff @ W

    r2_by_delay = [r2_score(Y[:, k], Y_hat[:, k]) for k in range(max_delay)]
    mc = float(np.sum(r2_by_delay))
    return MemoryCapacityResult(max_delay=max_delay, mc=mc, r2_by_delay=r2_by_delay)
