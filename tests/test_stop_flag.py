from __future__ import annotations

import multiprocessing as mp
import time

from sorch.ipc.stop_flag import StopFlag


def _producer(handle, n: int, err_q):  # noqa: ANN001
    try:
        sf = StopFlag(handle)
        for i in range(n):
            sf.publish(t_read_done_ns=i + 1, t_feat_done_ns=i + 2, t_decision_ns=i + 3)
            time.sleep(0.001)
    except Exception as e:  # noqa: BLE001
        err_q.put(f"producer error: {type(e).__name__}: {e}")


def _consumer(handle, n: int, out_q, err_q):  # noqa: ANN001
    try:
        sf = StopFlag(handle)
        last_seq = 0
        got = []
        t0 = time.perf_counter()
        while (len(got) < n) and ((time.perf_counter() - t0) < 5.0):
            seq, t_read, t_feat, t_decision = sf.snapshot()
            if seq != last_seq:
                got.append((seq, t_read, t_feat, t_decision))
                last_seq = seq
            time.sleep(0.0002)
        out_q.put(got)
    except Exception as e:  # noqa: BLE001
        err_q.put(f"consumer error: {type(e).__name__}: {e}")


def test_stop_flag_multiprocess_monotonic() -> None:
    ctx = mp.get_context("spawn")
    sf = StopFlag.create(ctx)

    out_q = ctx.Queue()
    err_q = ctx.Queue()
    n = 5
    p1 = ctx.Process(target=_producer, kwargs=dict(handle=sf.handle(), n=n, err_q=err_q))
    p2 = ctx.Process(target=_consumer, kwargs=dict(handle=sf.handle(), n=n, out_q=out_q, err_q=err_q))
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
    # Ensure timestamps correspond to producer writes.
    assert got[0][1:] == (1, 2, 3)
