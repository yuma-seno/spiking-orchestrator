from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ControlVecHandle:
    """Pickleable handle for a shared, low-dimensional control vector.

    Intended use:
    - Producer updates `u` (and optional timestamps) first, then increments `seq`.
    - Consumers detect new commands by observing `seq` change.

    This is designed to keep Core→Motor I/O minimal and low-latency.
    """

    dim: int
    seq: Any
    t_ns: Any
    u: Any


class ControlVec:
    def __init__(self, handle: ControlVecHandle):
        self._h = handle

    @staticmethod
    def create(ctx, *, dim: int) -> ControlVec:
        import ctypes

        d = int(dim)
        if d <= 0:
            raise ValueError("dim must be >= 1")

        # RawArray/RawValue are lock-free-ish (no synchronization).
        # We rely on seq monotonicity and the producer write order.
        h = ControlVecHandle(
            dim=d,
            seq=ctx.RawValue(ctypes.c_uint64, 0),
            t_ns=ctx.RawValue(ctypes.c_uint64, 0),
            u=ctx.RawArray(ctypes.c_double, d),
        )
        return ControlVec(h)

    def handle(self) -> ControlVecHandle:
        return self._h

    @property
    def dim(self) -> int:
        return int(self._h.dim)

    def publish(self, u: Iterable[float], *, t_ns: int) -> int:
        vals = list(u)
        if len(vals) != int(self._h.dim):
            raise ValueError(f"u length mismatch: expected {int(self._h.dim)}, got {len(vals)}")

        for i, v in enumerate(vals):
            self._h.u[i] = float(v)
        self._h.t_ns.value = int(t_ns)
        self._h.seq.value = int(self._h.seq.value) + 1
        return int(self._h.seq.value)

    def snapshot(self) -> tuple[int, int, list[float]]:
        seq = int(self._h.seq.value)
        t_ns = int(self._h.t_ns.value)
        u = [float(self._h.u[i]) for i in range(int(self._h.dim))]
        return (seq, t_ns, u)
