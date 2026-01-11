from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.sharedctypes import RawValue
import ctypes


@dataclass(frozen=True, slots=True)
class SpscRingHandle:
    """Pickleable handle to attach to an existing shared ring buffer."""

    shm_name: str
    capacity: int
    item_shape: tuple[int, ...]
    dtype_str: str
    write_idx: Any
    read_idx: Any


class SharedSpscRing:
    """A small SPSC ring buffer backed by shared memory.

    Notes:
    - Intended for 1 producer process and 1 consumer process.
    - Indices are stored in shared memory (RawValue) without locks.
    - This is a pragmatic implementation for low-latency IPC in Phase 3.
    """

    def __init__(
        self,
        shm: SharedMemory,
        capacity: int,
        item_shape: tuple[int, ...],
        dtype: np.dtype,
        write_idx: Any,
        read_idx: Any,
        *,
        owner: bool,
    ) -> None:
        self._shm = shm
        self._capacity = int(capacity)
        self._item_shape = tuple(int(x) for x in item_shape)
        self._dtype = np.dtype(dtype)
        self._write_idx = write_idx
        self._read_idx = read_idx
        self._owner = bool(owner)

        self._buf = np.ndarray(
            (self._capacity, *self._item_shape), dtype=self._dtype, buffer=self._shm.buf
        )

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def item_shape(self) -> tuple[int, ...]:
        return self._item_shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def handle(self) -> SpscRingHandle:
        return SpscRingHandle(
            shm_name=self._shm.name,
            capacity=self._capacity,
            item_shape=self._item_shape,
            dtype_str=self._dtype.str,
            write_idx=self._write_idx,
            read_idx=self._read_idx,
        )

    @staticmethod
    def create(capacity: int, item_shape: tuple[int, ...], dtype: Any = np.float64) -> SharedSpscRing:
        cap = int(capacity)
        if cap <= 1:
            raise ValueError("capacity must be >= 2")
        shape = tuple(int(x) for x in item_shape)
        if any(x <= 0 for x in shape):
            raise ValueError("item_shape must be positive")

        dt = np.dtype(dtype)
        nbytes = int(np.prod((cap, *shape))) * dt.itemsize
        shm = SharedMemory(create=True, size=nbytes)

        # RawValue: shared memory scalar without a lock.
        # c_uint64 is enough for long runs.
        write_idx = RawValue(ctypes.c_uint64, 0)
        read_idx = RawValue(ctypes.c_uint64, 0)

        ring = SharedSpscRing(
            shm=shm,
            capacity=cap,
            item_shape=shape,
            dtype=dt,
            write_idx=write_idx,
            read_idx=read_idx,
            owner=True,
        )
        ring._buf.fill(0)
        return ring

    @staticmethod
    def attach(handle: SpscRingHandle) -> SharedSpscRing:
        shm = SharedMemory(name=handle.shm_name, create=False)
        dt = np.dtype(handle.dtype_str)
        return SharedSpscRing(
            shm=shm,
            capacity=int(handle.capacity),
            item_shape=tuple(handle.item_shape),
            dtype=dt,
            write_idx=handle.write_idx,
            read_idx=handle.read_idx,
            owner=False,
        )

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        if self._owner:
            self._shm.unlink()

    def __enter__(self) -> SharedSpscRing:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()
        self.unlink()

    def _w(self) -> int:
        return int(self._write_idx.value)

    def _r(self) -> int:
        return int(self._read_idx.value)

    def size_approx(self) -> int:
        return max(0, self._w() - self._r())

    def push(self, item: np.ndarray) -> bool:
        w = self._w()
        r = self._r()
        if (w - r) >= self._capacity:
            return False

        idx = w % self._capacity
        arr = np.asarray(item, dtype=self._dtype)
        if arr.shape != self._item_shape:
            raise ValueError(f"item shape mismatch: expected {self._item_shape}, got {arr.shape}")

        self._buf[idx, ...] = arr
        self._write_idx.value = w + 1
        return True

    def push_blocking(self, item: np.ndarray, *, timeout_s: float = 0.05, sleep_s: float = 0.0002) -> None:
        t0 = time.perf_counter()
        while not self.push(item):
            if (time.perf_counter() - t0) >= timeout_s:
                raise TimeoutError("ring buffer full")
            time.sleep(sleep_s)

    def pop(self) -> np.ndarray | None:
        r = self._r()
        w = self._w()
        if r >= w:
            return None

        idx = r % self._capacity
        out = self._buf[idx, ...].copy()
        self._read_idx.value = r + 1
        return out

    def pop_blocking(self, *, timeout_s: float = 0.05, sleep_s: float = 0.0002) -> np.ndarray:
        t0 = time.perf_counter()
        while True:
            item = self.pop()
            if item is not None:
                return item
            if (time.perf_counter() - t0) >= timeout_s:
                raise TimeoutError("ring buffer empty")
            time.sleep(sleep_s)
