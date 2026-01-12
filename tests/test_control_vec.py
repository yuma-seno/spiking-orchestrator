from __future__ import annotations

import multiprocessing as mp
import time

from sorch.ipc.control_vec import ControlVec


def _producer(handle, n: int, err_q):  # noqa: ANN001
    try:
        cv = ControlVec(handle)
        for i in range(n):
            u = [float(i), float(i + 0.25), float(i + 0.5)]
            cv.publish(u, t_ns=i + 10)
            time.sleep(0.001)
    except Exception as e:  # noqa: BLE001
        err_q.put(f"producer error: {type(e).__name__}: {e}")


def _consumer(handle, n: int, out_q, err_q):  # noqa: ANN001
    try:
        cv = ControlVec(handle)
        last_seq = 0
        got = []
        t0 = time.perf_counter()
        while (len(got) < n) and ((time.perf_counter() - t0) < 5.0):
            seq, t_ns, u = cv.snapshot()
            if seq != last_seq:
                got.append((seq, t_ns, u))
                last_seq = seq
            time.sleep(0.0002)
        out_q.put(got)
    except Exception as e:  # noqa: BLE001
        err_q.put(f"consumer error: {type(e).__name__}: {e}")


def test_control_vec_multiprocess_monotonic() -> None:
    ctx = mp.get_context("spawn")
    cv = ControlVec.create(ctx, dim=3)

    out_q = ctx.Queue()
    err_q = ctx.Queue()
    n = 5
    p1 = ctx.Process(target=_producer, kwargs=dict(handle=cv.handle(), n=n, err_q=err_q))
    p2 = ctx.Process(target=_consumer, kwargs=dict(handle=cv.handle(), n=n, out_q=out_q, err_q=err_q))
    p1.start()
    p2.start()

    try:
        got = out_q.get(timeout=5.0)
    except Exception as e:  # noqa: BLE001
        extra = []
        try:
            extra.append(err_q.get_nowait())
        except Exception:
            pass
        raise AssertionError(
            f"no output (timeout). exitcodes: producer={p1.exitcode} consumer={p2.exitcode}. extra={extra}. err={e}"
        ) from e
    finally:
        p1.join(timeout=2.0)
        p2.join(timeout=2.0)
        if p1.is_alive():
            p1.terminate()
        if p2.is_alive():
            p2.terminate()

    assert [x[0] for x in got] == list(range(1, n + 1))
    assert got[0][1] == 10
    assert got[0][2] == [0.0, 0.25, 0.5]
