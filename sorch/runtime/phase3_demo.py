from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sorch.audio.features import AudioFeatures, compute_features
from sorch.ipc.spsc_ring import SharedSpscRing, SpscRingHandle
from sorch.ipc.stop_flag import StopFlagHandle
from sorch.reflex.dsp_reflex import ReflexConfig, ReflexDSP
from sorch.utils.metrics import summarize_latencies_ns


@dataclass(frozen=True, slots=True)
class Phase3StopEvent:
    seq: int
    t_read_done_ns: int
    t_feat_done_ns: int
    t_decision_ns: int
    t_stop_done_ns: int
    t_stream_inactive_ns: int
    t_stream_closed_ns: int
    t_stream_reopened_ns: int
    latency_ns: int
    latency_physical_ns: int
    rms: float
    zcr: float
    centroid_hz: float
    voicing: bool
    threshold: float
    reason: str
    stop_mode: str
    error: str


def _open_pyaudio():
    # PortAudio tends to try JACK first; in containers that often spews logs.
    os.environ.setdefault("JACK_NO_START_SERVER", "1")
    try:
        import pyaudio  # type: ignore

        return pyaudio
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "PyAudio の import に失敗しました。requirements のインストール完了と、"
            "システム側の PortAudio 依存（portaudio19-dev 等）が揃っているか確認してください。"
        ) from e


def _audio_proc(
    *,
    feat_ring: SpscRingHandle,
    shutdown_event,
    seconds: float,
    sample_rate: int,
    frames: int,
    input_device: int | None,
    input_channels: int,
    dry_run: bool,
) -> None:
    ring = SharedSpscRing.attach(feat_ring)

    t_end = time.perf_counter() + float(seconds)

    try:
        if dry_run:
            seq = 0
            dt_s = float(frames) / float(sample_rate)
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
                _ = ring.push(msg)

                seq += 1
                time.sleep(dt_s)
            return

        pyaudio = _open_pyaudio()
        pa = pyaudio.PyAudio()
        try:
            input_stream = pa.open(
                format=pyaudio.paInt16,
                channels=int(input_channels),
                rate=int(sample_rate),
                input=True,
                frames_per_buffer=int(frames),
                input_device_index=input_device,
                start=True,
            )
        except Exception as e:  # noqa: BLE001
            pa.terminate()
            raise RuntimeError(
                "オーディオ入力（マイク）のオープンに失敗しました。\n"
                "- まず `--list-devices` で index を確認\n"
                "- コンテナ内で /dev/snd が見えているか確認\n"
            ) from e

        try:
            while (not shutdown_event.is_set()) and (time.perf_counter() < t_end):
                data = input_stream.read(int(frames), exception_on_overflow=False)
                t_read_done_ns = time.perf_counter_ns()

                x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                if int(input_channels) > 1:
                    x = x.reshape(-1, int(input_channels)).mean(axis=1)
                frame = x / 32768.0

                feats = compute_features(frame, int(sample_rate))
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
                # best-effort: if core lags, drop newest frames
                _ = ring.push(msg)

        finally:
            try:
                input_stream.stop_stream()
            except Exception:
                pass
            try:
                input_stream.close()
            except Exception:
                pass
            pa.terminate()
    finally:
        ring.close()


def _core_proc(
    *,
    feat_ring: SpscRingHandle,
    stop_ring: SpscRingHandle,
    stop_flag: StopFlagHandle,
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

    from sorch.ipc.stop_flag import StopFlag

    sf = StopFlag(stop_flag)

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
                # Priority path: publish stop flag first.
                sf.publish(t_read_done_ns=t_read_done_ns, t_feat_done_ns=t_feat_done_ns, t_decision_ns=t_decision_ns)
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


def _motor_proc(
    *,
    stop_ring: SpscRingHandle,
    stop_flag: StopFlagHandle,
    shutdown_event,
    out_jsonl: str,
    seconds: float,
    sample_rate: int,
    frames: int,
    output_device: int | None,
    output_channels: int,
    output_mode: str,
    output_gain: float,
    tone_hz: float,
    stop_mode: str,
    resume_after_ms: float,
    dry_run: bool,
) -> None:
    stop_rb = SharedSpscRing.attach(stop_ring)

    from sorch.ipc.stop_flag import StopFlag

    sf = StopFlag(stop_flag)

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seq = 0

    output_stream = None
    pa = None
    out_buf = b""

    # Callback-controlled output is preferred to reduce stop latency.
    output_enabled = True
    last_seen_seq = 0
    pending_stop_event: Phase3StopEvent | None = None
    pending_stop_event_wait_physical = False
    callback_should_complete = False
    resume_at_ns = 0

    t_end = time.perf_counter() + float(seconds)

    try:
        if not dry_run:
            pyaudio = _open_pyaudio()
            pa = pyaudio.PyAudio()

            out_frames = int(frames)
            gain = float(max(0.0, min(1.0, float(output_gain))))
            ch = int(output_channels)

            if output_mode == "silence":
                out_buf = (np.zeros((out_frames, ch), dtype=np.float32)).tobytes()
            elif output_mode == "noise":
                out_buf = (gain * np.random.randn(out_frames, ch).astype(np.float32)).tobytes()
            elif output_mode == "tone":
                t = np.arange(out_frames, dtype=np.float32) / float(sample_rate)
                wave = gain * np.sin(2.0 * np.pi * float(tone_hz) * t).astype(np.float32)
                if ch == 1:
                    out_buf = wave.tobytes()
                else:
                    out_buf = np.repeat(wave[:, None], repeats=ch, axis=1).tobytes()
            else:
                out_buf = b""

            if output_mode != "off":
                try:
                    if str(stop_mode) not in {"mute", "hard"}:
                        raise ValueError("stop_mode must be 'mute' or 'hard'")

                    def callback(in_data, frame_count, time_info, status_flags):  # noqa: ANN001
                        nonlocal output_enabled, last_seen_seq, pending_stop_event, pending_stop_event_wait_physical
                        nonlocal callback_should_complete
                        nonlocal resume_at_ns

                        seq, t_read_done_ns, t_feat_done_ns, t_decision_ns = sf.snapshot()
                        if seq != last_seen_seq:
                            last_seen_seq = seq
                            # Stop path: silence output immediately.
                            output_enabled = False
                            t_stop_done_ns = time.perf_counter_ns()
                            pending_stop_event_wait_physical = bool(str(stop_mode) == "hard")
                            callback_should_complete = bool(str(stop_mode) == "hard")
                            if (str(stop_mode) == "mute") and float(resume_after_ms) > 0:
                                resume_at_ns = t_stop_done_ns + int(float(resume_after_ms) * 1e6)
                            pending_stop_event = Phase3StopEvent(
                                seq=int(seq),
                                t_read_done_ns=int(t_read_done_ns),
                                t_feat_done_ns=int(t_feat_done_ns),
                                t_decision_ns=int(t_decision_ns),
                                t_stop_done_ns=int(t_stop_done_ns),
                                t_stream_inactive_ns=0,
                                t_stream_closed_ns=0,
                                t_stream_reopened_ns=0,
                                latency_ns=int(t_stop_done_ns - int(t_read_done_ns)),
                                latency_physical_ns=0,
                                rms=0.0,
                                zcr=0.0,
                                centroid_hz=0.0,
                                voicing=False,
                                threshold=0.0,
                                reason="stop_flag",
                                stop_mode=str(stop_mode),
                                error="",
                            )

                        # Physical hard stop: request stream termination from callback.
                        if callback_should_complete:
                            callback_should_complete = False
                            return (b"\x00" * (frame_count * ch * 4), pyaudio.paComplete)

                        if not output_enabled:
                            return (b"\x00" * (frame_count * ch * 4), pyaudio.paContinue)

                        if out_buf:
                            # out_buf is precomputed for the configured frame size; for safety,
                            # match frame_count exactly by slicing.
                            need = frame_count * ch * 4
                            if len(out_buf) >= need:
                                return (out_buf[:need], pyaudio.paContinue)

                        return (b"\x00" * (frame_count * ch * 4), pyaudio.paContinue)

                    output_stream = pa.open(
                        format=pyaudio.paFloat32,
                        channels=ch,
                        rate=int(sample_rate),
                        output=True,
                        frames_per_buffer=out_frames,
                        output_device_index=output_device,
                        stream_callback=callback,
                        start=True,
                    )
                except Exception as e:  # noqa: BLE001
                    pa.terminate()
                    raise RuntimeError(
                        "オーディオ出力（スピーカー）のオープンに失敗しました。\n"
                        "- まず `--list-devices` で index を確認\n"
                        "- コンテナ内で /dev/snd が見えているか確認\n"
                    ) from e

                # In callback mode, no explicit priming is needed.

        with out_path.open("w", encoding="utf-8") as f:
            while (not shutdown_event.is_set()) and (time.perf_counter() < t_end):
                # If we got a stop event from the callback, persist it.
                if pending_stop_event is not None:
                    if not pending_stop_event_wait_physical:
                        f.write(json.dumps(asdict(pending_stop_event), ensure_ascii=False) + "\n")
                        f.flush()
                        pending_stop_event = None
                    else:
                        # For hard-stop mode, wait until stream actually becomes inactive/closed.
                        if output_stream is None:
                            pending_stop_event = Phase3StopEvent(
                                **{
                                    **asdict(pending_stop_event),
                                    "error": "hard_stop_failed: output_stream is None",
                                }
                            )
                            f.write(json.dumps(asdict(pending_stop_event), ensure_ascii=False) + "\n")
                            f.flush()
                            pending_stop_event = None
                            pending_stop_event_wait_physical = False
                        else:
                            try:
                                if not output_stream.is_active():
                                    t_inactive_ns = time.perf_counter_ns()
                                    t_closed_ns = 0
                                    try:
                                        output_stream.close()
                                        output_stream = None
                                        t_closed_ns = time.perf_counter_ns()
                                    except Exception as e:  # noqa: BLE001
                                        pending_stop_event = Phase3StopEvent(
                                            **{
                                                **asdict(pending_stop_event),
                                                "t_stream_inactive_ns": int(t_inactive_ns),
                                                "t_stream_closed_ns": int(t_closed_ns),
                                                "latency_physical_ns": int(t_inactive_ns - int(pending_stop_event.t_read_done_ns)),
                                                "error": f"close_failed: {type(e).__name__}: {e}",
                                            }
                                        )

                                    if output_stream is None:
                                        pending_stop_event = Phase3StopEvent(
                                            **{
                                                **asdict(pending_stop_event),
                                                "t_stream_inactive_ns": int(t_inactive_ns),
                                                "t_stream_closed_ns": int(t_closed_ns),
                                                "latency_physical_ns": int(t_inactive_ns - int(pending_stop_event.t_read_done_ns)),
                                            }
                                        )

                                    f.write(json.dumps(asdict(pending_stop_event), ensure_ascii=False) + "\n")
                                    f.flush()
                                    pending_stop_event = None
                                    pending_stop_event_wait_physical = False

                                    if float(resume_after_ms) > 0:
                                        resume_at_ns = time.perf_counter_ns() + int(float(resume_after_ms) * 1e6)
                            except Exception as e:  # noqa: BLE001
                                pending_stop_event = Phase3StopEvent(
                                    **{
                                        **asdict(pending_stop_event),
                                        "error": f"hard_stop_wait_failed: {type(e).__name__}: {e}",
                                    }
                                )
                                f.write(json.dumps(asdict(pending_stop_event), ensure_ascii=False) + "\n")
                                f.flush()
                                pending_stop_event = None
                                pending_stop_event_wait_physical = False

                # Resume output after cooldown (demo convenience).
                if (not output_enabled) and resume_at_ns and time.perf_counter_ns() >= int(resume_at_ns):
                    output_enabled = True
                    resume_at_ns = 0

                # Re-open output stream after a hard stop.
                if (
                    (not dry_run)
                    and (str(stop_mode) == "hard")
                    and (output_mode != "off")
                    and output_enabled
                    and (output_stream is None)
                    and (pa is not None)
                ):
                    try:
                        # Sync last_seen_seq so we don't immediately re-trigger on an old stop.
                        last_seen_seq = int(sf.snapshot()[0])
                        output_stream = pa.open(
                            format=pyaudio.paFloat32,
                            channels=ch,
                            rate=int(sample_rate),
                            output=True,
                            frames_per_buffer=out_frames,
                            output_device_index=output_device,
                            stream_callback=callback,
                            start=True,
                        )
                        # Record reopen timestamp on the next stop event (best-effort).
                    except Exception:
                        # If reopen fails, disable output for the remainder of the demo.
                        output_enabled = False
                        output_stream = None

                msg = stop_rb.pop()
                if msg is None:
                    time.sleep(0.0002)
                    continue

                # Keep the ring-based events too (they include threshold and features).
                t_stop_done_ns = time.perf_counter_ns()
                seq += 1
                ev = Phase3StopEvent(
                    seq=seq,
                    t_read_done_ns=int(msg[0]),
                    t_feat_done_ns=int(msg[1]),
                    t_decision_ns=int(msg[2]),
                    t_stop_done_ns=t_stop_done_ns,
                    t_stream_inactive_ns=0,
                    t_stream_closed_ns=0,
                    t_stream_reopened_ns=0,
                    latency_ns=int(t_stop_done_ns - int(msg[0])),
                    latency_physical_ns=0,
                    rms=float(msg[3]),
                    zcr=float(msg[4]),
                    centroid_hz=float(msg[5]),
                    voicing=bool(msg[6] >= 0.5),
                    threshold=float(msg[7]),
                    reason="stop_signal",
                    stop_mode=str(stop_mode),
                    error="",
                )
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
                f.flush()
    finally:
        try:
            if output_stream is not None:
                try:
                    output_stream.stop_stream()
                except Exception:
                    pass
                try:
                    output_stream.close()
                except Exception:
                    pass
        finally:
            if pa is not None:
                pa.terminate()
        stop_rb.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="SORCH Phase3 multiprocess demo (mic->core->motor stop)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--input-device", type=int, default=None, help="PyAudio input device index")
    ap.add_argument("--output-device", type=int, default=None, help="PyAudio output device index")
    ap.add_argument("--input-channels", type=int, default=1)
    ap.add_argument("--output-channels", type=int, default=1)
    ap.add_argument(
        "--output-mode",
        type=str,
        default="silence",
        choices=["silence", "noise", "tone", "off"],
        help="speaker output content: silence/noise/tone/off",
    )
    ap.add_argument("--output-gain", type=float, default=0.0, help="0.0..1.0 (use small values)")
    ap.add_argument("--tone-hz", type=float, default=220.0)
    ap.add_argument(
        "--stop-mode",
        type=str,
        default="mute",
        choices=["mute", "hard"],
        help="stop behavior: mute (silence only) or hard (stop/close/reopen output stream)",
    )
    ap.add_argument("--resume-after-ms", type=float, default=500.0)
    ap.add_argument("--holdoff-ms", type=float, default=300.0)
    ap.add_argument("--output", type=str, default="outputs/phase3/stop_events.jsonl")
    ap.add_argument("--list-devices", action="store_true", help="print PyAudio device list and exit")
    ap.add_argument("--dry-run", action="store_true", help="no audio devices; synthetic input and no output")
    args = ap.parse_args()

    if not args.dry_run and args.list_devices:
        pyaudio = _open_pyaudio()
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                name = info.get("name", "")
                max_in = int(info.get("maxInputChannels", 0))
                max_out = int(info.get("maxOutputChannels", 0))
                default_sr = info.get("defaultSampleRate", "?")
                host_api = info.get("hostApi", "?")
                print(f"[{i}] in={max_in} out={max_out} sr={default_sr} hostApi={host_api} name={name}")
        finally:
            pa.terminate()
        return 0

    dt_ms = (int(args.frames) / int(args.sample_rate)) * 1000.0

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    shutdown_event = ctx.Event()

    feat_rb = SharedSpscRing.create(capacity=2048, item_shape=(6,), dtype=np.float64)
    stop_rb = SharedSpscRing.create(capacity=256, item_shape=(8,), dtype=np.float64)

    from sorch.ipc.stop_flag import StopFlag

    stop_flag = StopFlag.create(ctx)

    try:
        p_audio = ctx.Process(
            target=_audio_proc,
            kwargs=dict(
                feat_ring=feat_rb.handle(),
                shutdown_event=shutdown_event,
                seconds=float(args.seconds),
                sample_rate=int(args.sample_rate),
                frames=int(args.frames),
                input_device=args.input_device,
                input_channels=int(args.input_channels),
                dry_run=bool(args.dry_run),
            ),
            name="audio",
            daemon=True,
        )
        p_core = ctx.Process(
            target=_core_proc,
            kwargs=dict(
                feat_ring=feat_rb.handle(),
                stop_ring=stop_rb.handle(),
                stop_flag=stop_flag.handle(),
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
                stop_flag=stop_flag.handle(),
                shutdown_event=shutdown_event,
                out_jsonl=str(args.output),
                seconds=float(args.seconds),
                sample_rate=int(args.sample_rate),
                frames=int(args.frames),
                output_device=args.output_device,
                output_channels=int(args.output_channels),
                output_mode=str(args.output_mode),
                output_gain=float(args.output_gain),
                tone_hz=float(args.tone_hz),
                stop_mode=str(args.stop_mode),
                resume_after_ms=float(args.resume_after_ms),
                dry_run=bool(args.dry_run),
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
        latencies_by_reason: dict[str, list[int]] = {}
        physical_latencies_by_reason: dict[str, list[int]] = {}
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                reason = str(ev.get("reason", "unknown"))
                lat = int(ev.get("latency_ns", 0) or 0)
                latencies_by_reason.setdefault(reason, []).append(lat)

                physical_lat = int(ev.get("latency_physical_ns", 0) or 0)
                if physical_lat > 0:
                    physical_latencies_by_reason.setdefault(reason, []).append(physical_lat)

        total = sum(len(v) for v in latencies_by_reason.values())
        for reason, lats in sorted(latencies_by_reason.items(), key=lambda kv: kv[0]):
            summary = summarize_latencies_ns(lats)
            print(f"Phase3 stop latency summary detect [{reason}]:", summary)

        for reason, lats in sorted(physical_latencies_by_reason.items(), key=lambda kv: kv[0]):
            summary = summarize_latencies_ns(lats)
            print(f"Phase3 stop latency summary physical [{reason}]:", summary)

        print(f"events={total} output={out_path}")
    else:
        print("No events were written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
