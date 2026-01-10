from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LatencySummary:
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float


def summarize_latencies_ns(latencies_ns: list[int]) -> LatencySummary:
    if not latencies_ns:
        return LatencySummary(n=0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, mean_ms=0.0, std_ms=0.0)

    arr = np.asarray(latencies_ns, dtype=np.float64)
    ms = arr / 1e6
    p50, p95, p99 = np.quantile(ms, [0.50, 0.95, 0.99]).tolist()
    return LatencySummary(
        n=int(ms.size),
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
        mean_ms=float(np.mean(ms)),
        std_ms=float(np.std(ms)),
    )
