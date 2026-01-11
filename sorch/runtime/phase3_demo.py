from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sorch.audio.features import AudioFeatures, compute_features
from sorch.ipc.spsc_ring import SharedSpscRing, SpscRingHandle
from sorch.reflex.dsp_reflex import ReflexConfig, ReflexDSP
from sorch.utils.metrics import summarize_latencies_ns


@dataclass(frozen=True, slots=True)
class Phase3StopEvent:
    seq: int
    t_read_done_ns: int
    t_feat_done_ns: int
    t_decision_ns: int
    t_stop_done_ns: int
    latency_ns: int
    rms: float
    zcr: float
    centroid_hz: float
    voicing: bool
    threshold: float
    reason: str


def _audio_proc(
    *,
    feat_ring: SpscRingHandle,
    shutdown_event,
    seconds: float,
    sample_rate: int,
    frames: int,
) -> None:
    ring = SharedSpscRing.attach(feat_ring)

    t_end = time.perf_counter() + float(seconds)
    seq = 0
    dropped = 0

    dt_s = float(frames) / float(sample_rate)

    try:
        while (not shutdown_event.is_set()) and (time.perf_counter() < t_end):
            # synthetic: mostly silence with occasional voiced burst
            if (seq % 200) == 120:
                t = np.arange(frames, dtype=np.float32) / float(sample_rate)
                frame = 0.2 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
            else:
                frame = (0.001 * np.random.randn(frames)).astype(np.float32)

            t_read_done_ns = time.perf_counter_ns()
            feats = compute_features(frame, sample_rate)
            t_feat_done_ns = time.perf_counter_ns()

            msg = np.array(
                [
                    float(t_read_done_ns),
                    float(t_feat_done_ns),
                    float(feats.rms),
                    float(feats.zcr),
                    float(feats.spectral_centroid_hz),
                    1.0 if feats.voicing else 0.0,
                ],
                dtype=np.float64,
            )

            if not ring.push(msg):
                dropped += 1

            seq += 1
            time.sleep(dt_s)
    finally:
        ring.close()


def _core_proc(
    *,
    feat_ring: SpscRingHandle,
    stop_ring: SpscRingHandle,
    shutdown_event,
    holdoff_ms: float,
    dt_ms: float,
) -> None:
    feat_rb = SharedSpscRing.attach(feat_ring)
    stop_rb = SharedSpscRing.attach(stop_ring)

    reflex = ReflexDSP(
        ReflexConfig(
            dt_ms=float(dt_ms),
            holdoff_ms=float(holdoff_ms),
            noise_ema_tau_s=1.0,
            alpha=0.03,
        )
    )

    try:
        while not shutdown_event.is_set():
            msg = feat_rb.pop()
            if msg is None:
                time.sleep(0.0002)
                continue

            t_read_done_ns = int(msg[0])
            t_feat_done_ns = int(msg[1])
            feats = AudioFeatures(
                rms=float(msg[2]),
                zcr=float(msg[3]),
                spectral_centroid_hz=float(msg[4]),
                voicing=bool(msg[5] >= 0.5),
            )

            decision = reflex.update(feats, t_feat_done_ns)
            t_decision_ns = time.perf_counter_ns()

            if decision.stop_signal:
                stop_msg = np.array(
                    [
                        float(t_read_done_ns),
                        float(t_feat_done_ns),
                        float(t_decision_ns),
                        float(feats.rms),
                        float(feats.zcr),
                        float(feats.spectral_centroid_hz),
                        1.0 if feats.voicing else 0.0,
                        float(decision.threshold),
                    ],
                    dtype=np.float64,
                )
                # best-effort: if motor is lagging, we drop old stop signals
                if not stop_rb.push(stop_msg):
                    _ = stop_rb.pop()
                    _ = stop_rb.push(stop_msg)

    finally:
        feat_rb.close()
        stop_rb.close()


def _motor_proc(*, stop_ring: SpscRingHandle, shutdown_event, out_jsonl: str) -> None:
    stop_rb = SharedSpscRing.attach(stop_ring)

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seq = 0

    try:
        with out_path.open("w", encoding="utf-8") as f:
            while not shutdown_event.is_set():
                msg = stop_rb.pop()
                if msg is None:
                    time.sleep(0.0002)
                    continue

                t_stop_done_ns = time.perf_counter_ns()
                seq += 1

                ev = Phase3StopEvent(
                    seq=seq,
                    t_read_done_ns=int(msg[0]),
                    t_feat_done_ns=int(msg[1]),
                    t_decision_ns=int(msg[2]),
                    t_stop_done_ns=t_stop_done_ns,
                    latency_ns=int(t_stop_done_ns - int(msg[0])),
                    rms=float(msg[3]),
                    zcr=float(msg[4]),
                    centroid_hz=float(msg[5]),
                    voicing=bool(msg[6] >= 0.5),
                    threshold=float(msg[7]),
                    reason="stop_signal",
                )
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
                f.flush()
    finally:
        stop_rb.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="SORCH Phase3 multiprocess dry-run demo")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--holdoff-ms", type=float, default=300.0)
    ap.add_argument("--output", type=str, default="outputs/phase3/stop_events.jsonl")
    args = ap.parse_args()

    dt_ms = (int(args.frames) / int(args.sample_rate)) * 1000.0

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    shutdown_event = ctx.Event()

    feat_rb = SharedSpscRing.create(capacity=2048, item_shape=(6,), dtype=np.float64)
    stop_rb = SharedSpscRing.create(capacity=256, item_shape=(8,), dtype=np.float64)

    try:
        p_audio = ctx.Process(
            target=_audio_proc,
            kwargs=dict(
                feat_ring=feat_rb.handle(),
                shutdown_event=shutdown_event,
                seconds=float(args.seconds),
                sample_rate=int(args.sample_rate),
                frames=int(args.frames),
            ),
            name="audio",
            daemon=True,
        )
        p_core = ctx.Process(
            target=_core_proc,
            kwargs=dict(
                feat_ring=feat_rb.handle(),
                stop_ring=stop_rb.handle(),
                shutdown_event=shutdown_event,
                holdoff_ms=float(args.holdoff_ms),
                dt_ms=float(dt_ms),
            ),
            name="core",
            daemon=True,
        )
        p_motor = ctx.Process(
            target=_motor_proc,
            kwargs=dict(
                stop_ring=stop_rb.handle(),
                shutdown_event=shutdown_event,
                out_jsonl=str(args.output),
            ),
            name="motor",
            daemon=True,
        )

        p_audio.start()
        p_core.start()
        p_motor.start()

        time.sleep(float(args.seconds) + 0.5)

    finally:
        shutdown_event.set()
        for p in (p_audio, p_core, p_motor):
            try:
                p.join(timeout=2.0)
            except Exception:
                pass

        feat_rb.close()
        stop_rb.close()
        feat_rb.unlink()
        stop_rb.unlink()

    out_path = Path(args.output)
    if out_path.exists():
        latencies: list[int] = []
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                latencies.append(int(ev.get("latency_ns", 0)))

        summary = summarize_latencies_ns(latencies)
        print("Phase3 stop latency summary (audio->motor):", summary)
        print(f"events={len(latencies)} output={out_path}")
    else:
        print("No events were written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
