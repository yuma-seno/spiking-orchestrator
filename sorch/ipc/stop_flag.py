from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StopFlagHandle:
    """Pickleable handle for a shared, lock-free-ish stop flag.

    Fields are sharedctypes RawValue instances (lock-free) and are intended to be
    passed to child processes via multiprocessing spawn.

    The producer should update timestamps first, then increment `seq`.
    Consumers detect a new stop by observing `seq` change.
    """

    seq: Any
    t_read_done_ns: Any
    t_feat_done_ns: Any
    t_decision_ns: Any


class StopFlag:
    def __init__(self, handle: StopFlagHandle):
        self._h = handle

    @staticmethod
    def create(ctx) -> StopFlag:
        import ctypes

        h = StopFlagHandle(
            seq=ctx.RawValue(ctypes.c_uint64, 0),
            t_read_done_ns=ctx.RawValue(ctypes.c_uint64, 0),
            t_feat_done_ns=ctx.RawValue(ctypes.c_uint64, 0),
            t_decision_ns=ctx.RawValue(ctypes.c_uint64, 0),
        )
        return StopFlag(h)

    def handle(self) -> StopFlagHandle:
        return self._h

    def publish(self, *, t_read_done_ns: int, t_feat_done_ns: int, t_decision_ns: int) -> int:
        self._h.t_read_done_ns.value = int(t_read_done_ns)
        self._h.t_feat_done_ns.value = int(t_feat_done_ns)
        self._h.t_decision_ns.value = int(t_decision_ns)
        self._h.seq.value = int(self._h.seq.value) + 1
        return int(self._h.seq.value)

    def snapshot(self) -> tuple[int, int, int, int]:
        return (
            int(self._h.seq.value),
            int(self._h.t_read_done_ns.value),
            int(self._h.t_feat_done_ns.value),
            int(self._h.t_decision_ns.value),
        )
