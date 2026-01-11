from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from sorch.ipc.spsc_ring import SharedSpscRing


def test_spsc_ring_push_pop_single_process() -> None:
    rb = SharedSpscRing.create(capacity=8, item_shape=(4,), dtype=np.float64)
    try:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        assert rb.push(x)
        y = rb.pop()
        assert y is not None
        assert np.allclose(y, x)
        assert rb.pop() is None
    finally:
        rb.close()
        rb.unlink()


def _producer(handle, n: int, shutdown_event) -> None:  # noqa: ANN001
    rb = SharedSpscRing.attach(handle)
    try:
        for i in range(n):
            msg = np.array([float(i)], dtype=np.float64)
            while (not shutdown_event.is_set()) and (not rb.push(msg)):
                time.sleep(0.0002)
    finally:
        rb.close()


def _consumer(handle, n: int, out_q, shutdown_event) -> None:  # noqa: ANN001
    rb = SharedSpscRing.attach(handle)
    try:
        got = []
        while (not shutdown_event.is_set()) and (len(got) < n):
            msg = rb.pop()
            if msg is None:
                time.sleep(0.0002)
                continue
            got.append(int(msg[0]))
        out_q.put(got)
    finally:
        rb.close()


def test_spsc_ring_multiprocess_order() -> None:
    ctx = mp.get_context("spawn")
    shutdown_event = ctx.Event()

    rb = SharedSpscRing.create(capacity=64, item_shape=(1,), dtype=np.float64)
    out_q = ctx.Queue()

    n = 200
    try:
        p1 = ctx.Process(target=_producer, kwargs=dict(handle=rb.handle(), n=n, shutdown_event=shutdown_event))
        p2 = ctx.Process(target=_consumer, kwargs=dict(handle=rb.handle(), n=n, out_q=out_q, shutdown_event=shutdown_event))
        p1.start()
        p2.start()

        got = out_q.get(timeout=5.0)
        assert got == list(range(n))
    finally:
        shutdown_event.set()
        for p in (p1, p2):
            try:
                p.join(timeout=2.0)
            except Exception:
                pass
        rb.close()
        rb.unlink()
