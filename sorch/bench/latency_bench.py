from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sorch.audio.features import compute_features
from sorch.reflex.dsp_reflex import ReflexConfig, ReflexDSP
from sorch.utils.metrics import summarize_latencies_ns


@dataclass(frozen=True, slots=True)
class StopEvent:
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
    interrupt_cue: bool
    cue_strength: float
    threshold: float
    reason: str
    stop_mode: str


def _meta_path(out_path: Path) -> Path:
    # Keep the JSONL schema focused on per-event data; write run config separately.
    return out_path.with_suffix(out_path.suffix + ".meta.json")


def _open_pyaudio():
    # PortAudio tends to try JACK first; in containers that often spews logs.
    os.environ.setdefault("JACK_NO_START_SERVER", "1")
    try:
        import pyaudio  # type: ignore

        return pyaudio
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "PyAudio の import に失敗しました。Dockerfile では portaudio19-dev を入れているので、"
            "requirements のインストールが完了しているか確認してください。"
        ) from e


def main() -> int:
    ap = argparse.ArgumentParser(description="SORCH Phase1 latency benchmark (mic->reflex->stop)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--sample-rate", type=int, default=48000, help="legacy: used for both input/output unless overridden")
    ap.add_argument("--input-sample-rate", type=int, default=None)
    ap.add_argument("--output-sample-rate", type=int, default=None)
    ap.add_argument("--frames", type=int, default=128, help="frames per buffer (e.g., 64-256)")
    ap.add_argument("--list-devices", action="store_true", help="print PyAudio device list and exit")
    ap.add_argument("--input-device", type=int, default=None, help="PyAudio input device index")
    ap.add_argument("--output-device", type=int, default=None, help="PyAudio output device index")
    ap.add_argument("--input-channels", type=int, default=1)
    ap.add_argument("--output-channels", type=int, default=1)
    ap.add_argument("--holdoff-ms", type=float, default=300.0)
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--noise-tau-s", type=float, default=1.0)
    ap.add_argument("--output", type=str, default="outputs/phase1/latency/latency_events.jsonl")
    ap.add_argument("--resume-after-ms", type=float, default=500.0, help="restart output after stop")
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
        default="stop_stream",
        choices=["stop_stream", "close_reopen"],
        help="how to stop output on trigger (close_reopen approximates physical stop)",
    )
    ap.add_argument("--dry-run", action="store_true", help="no audio devices; synthetic input")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_sr = int(args.input_sample_rate or args.sample_rate)
    output_sr = int(args.output_sample_rate or args.sample_rate)
    input_channels = int(args.input_channels)
    output_channels = int(args.output_channels)

    if input_channels not in (1, 2):
        raise SystemExit("--input-channels は 1 または 2 を指定してください")
    if output_channels not in (1, 2):
        raise SystemExit("--output-channels は 1 または 2 を指定してください")

    dt_ms = (args.frames / input_sr) * 1000.0

    meta = {
        "tool": "sorch.bench.latency_bench",
        "timestamp_unix_s": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "args": vars(args),
        "derived": {
            "input_sr": input_sr,
            "output_sr": output_sr,
            "input_channels": input_channels,
            "output_channels": output_channels,
            "dt_ms": dt_ms,
        },
    }
    _meta_path(out_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reflex = ReflexDSP(
        ReflexConfig(
            dt_ms=dt_ms,
            holdoff_ms=float(args.holdoff_ms),
            noise_ema_tau_s=float(args.noise_tau_s),
            alpha=float(args.alpha),
        )
    )

    stop_events: list[StopEvent] = []

    if args.dry_run:
        t_end = time.perf_counter() + float(args.seconds)
        seq = 0
        while time.perf_counter() < t_end:
            # synthetic: mostly silence with occasional voiced burst
            if (seq % 200) == 120:
                t = np.arange(args.frames, dtype=np.float32) / float(args.sample_rate)
                frame = 0.2 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
            else:
                frame = (0.001 * np.random.randn(args.frames)).astype(np.float32)

            t_read_done_ns = time.perf_counter_ns()
            feats = compute_features(frame, input_sr)
            t_feat_done_ns = time.perf_counter_ns()
            decision = reflex.update(feats, t_feat_done_ns)
            t_decision_ns = time.perf_counter_ns()

            if decision.stop_signal:
                t_stop_done_ns = time.perf_counter_ns()
                t_stream_inactive_ns = 0
                t_stream_closed_ns = 0
                t_stream_reopened_ns = 0
                latency_ns = t_stop_done_ns - t_read_done_ns
                stop_events.append(
                    StopEvent(
                        seq=len(stop_events) + 1,
                        t_read_done_ns=t_read_done_ns,
                        t_feat_done_ns=t_feat_done_ns,
                        t_decision_ns=t_decision_ns,
                        t_stop_done_ns=t_stop_done_ns,
                        t_stream_inactive_ns=t_stream_inactive_ns,
                        t_stream_closed_ns=t_stream_closed_ns,
                        t_stream_reopened_ns=t_stream_reopened_ns,
                        latency_ns=latency_ns,
                        latency_physical_ns=0,
                        rms=feats.rms,
                        zcr=feats.zcr,
                        centroid_hz=feats.spectral_centroid_hz,
                        voicing=feats.voicing,
                        interrupt_cue=decision.interrupt_cue,
                        cue_strength=decision.cue_strength,
                        threshold=decision.threshold,
                        reason=decision.reason,
                        stop_mode=str(args.stop_mode),
                    )
                )
            seq += 1

        with out_path.open("w", encoding="utf-8") as f:
            for ev in stop_events:
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")

        summary = summarize_latencies_ns([e.latency_ns for e in stop_events])
        print("DRY-RUN summary:", summary)
        return 0

    pyaudio = _open_pyaudio()
    pa = pyaudio.PyAudio()

    if args.list_devices:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            name = info.get("name", "")
            max_in = int(info.get("maxInputChannels", 0))
            max_out = int(info.get("maxOutputChannels", 0))
            default_sr = info.get("defaultSampleRate", "?")
            host_api = info.get("hostApi", "?")
            print(f"[{i}] in={max_in} out={max_out} sr={default_sr} hostApi={host_api} name={name}")
        pa.terminate()
        return 0

    output_stream = None
    input_stream = None

    out_frames = args.frames
    gain = float(max(0.0, min(1.0, args.output_gain)))

    if args.output_mode == "silence":
        out_buf = (np.zeros((out_frames, output_channels), dtype=np.float32)).tobytes()
    elif args.output_mode == "noise":
        out_buf = (gain * np.random.randn(out_frames, output_channels).astype(np.float32)).tobytes()
    elif args.output_mode == "tone":
        t = np.arange(out_frames, dtype=np.float32) / float(output_sr)
        wave = gain * np.sin(2.0 * np.pi * float(args.tone_hz) * t).astype(np.float32)
        if output_channels == 1:
            out_buf = wave.tobytes()
        else:
            out_buf = np.repeat(wave[:, None], repeats=output_channels, axis=1).tobytes()
    else:
        out_buf = b""

    try:
        if args.output_mode != "off":
            output_stream = pa.open(
                format=pyaudio.paFloat32,
                channels=output_channels,
                rate=output_sr,
                output=True,
                frames_per_buffer=out_frames,
                output_device_index=args.output_device,
                start=True,
            )

        input_stream = pa.open(
            format=pyaudio.paInt16,
            channels=input_channels,
            rate=input_sr,
            input=True,
            frames_per_buffer=args.frames,
            input_device_index=args.input_device,
            start=True,
        )
    except Exception as e:  # noqa: BLE001
        pa.terminate()
        raise RuntimeError(
            "オーディオデバイスのオープンに失敗しました。\n"
            "- まず `python -m sorch.bench.latency_bench --list-devices` で index を確認\n"
            "- コンテナ内で /dev/snd が見えているか確認（無い場合はホストの音声デバイス共有設定が必要）\n"
            "- PulseAudio/pipewire 利用時は PULSE_SERVER が渡っているか確認\n"
        ) from e

    t_end = time.perf_counter() + float(args.seconds)
    last_resume_ns = 0
    pending_reopen_event_idx: int | None = None

    try:
        if output_stream is not None:
            # Prime output
            for _ in range(10):
                output_stream.write(out_buf)

        while time.perf_counter() < t_end:
            if output_stream is not None:
                # Keep output running unless stopped.
                if output_stream.is_active():
                    output_stream.write(out_buf)
                else:
                    # resume after cooldown
                    if last_resume_ns and (time.perf_counter_ns() - last_resume_ns) >= int(args.resume_after_ms * 1e6):
                        if args.stop_mode == "close_reopen":
                            output_stream = pa.open(
                                format=pyaudio.paFloat32,
                                channels=output_channels,
                                rate=output_sr,
                                output=True,
                                frames_per_buffer=out_frames,
                                output_device_index=args.output_device,
                                start=True,
                            )
                        else:
                            output_stream.start_stream()

                        t_reopen_ns = time.perf_counter_ns()
                        if pending_reopen_event_idx is not None:
                            ev = stop_events[pending_reopen_event_idx]
                            stop_events[pending_reopen_event_idx] = StopEvent(
                                **{
                                    **asdict(ev),
                                    "t_stream_reopened_ns": int(t_reopen_ns),
                                }
                            )
                            pending_reopen_event_idx = None

                        last_resume_ns = 0

            assert input_stream is not None
            data = input_stream.read(args.frames, exception_on_overflow=False)
            t_read_done_ns = time.perf_counter_ns()

            # int16 interleaved -> mono float32
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            if input_channels > 1:
                x = x.reshape(-1, input_channels).mean(axis=1)
            frame = x / 32768.0
            feats = compute_features(frame, input_sr)
            t_feat_done_ns = time.perf_counter_ns()

            decision = reflex.update(feats, t_feat_done_ns)
            t_decision_ns = time.perf_counter_ns()

            if decision.stop_signal:
                t_stream_inactive_ns = 0
                t_stream_closed_ns = 0
                t_stream_reopened_ns = 0

                if output_stream is not None and output_stream.is_active():
                    if args.stop_mode == "close_reopen":
                        output_stream.stop_stream()
                        t_stop_done_ns = time.perf_counter_ns()
                        t_stream_inactive_ns = t_stop_done_ns

                        output_stream.close()
                        t_stream_closed_ns = time.perf_counter_ns()
                        output_stream = None
                        last_resume_ns = t_stream_closed_ns
                    else:
                        output_stream.stop_stream()
                        t_stop_done_ns = time.perf_counter_ns()
                        t_stream_inactive_ns = t_stop_done_ns
                        last_resume_ns = t_stop_done_ns
                else:
                    # No output stream: approximate stop timestamp right after decision.
                    t_stop_done_ns = time.perf_counter_ns()

                latency_ns = t_stop_done_ns - t_read_done_ns
                latency_physical_ns = 0
                if t_stream_closed_ns:
                    latency_physical_ns = int(t_stream_closed_ns - t_read_done_ns)
                elif t_stream_inactive_ns:
                    latency_physical_ns = int(t_stream_inactive_ns - t_read_done_ns)

                stop_events.append(
                    StopEvent(
                        seq=len(stop_events) + 1,
                        t_read_done_ns=t_read_done_ns,
                        t_feat_done_ns=t_feat_done_ns,
                        t_decision_ns=t_decision_ns,
                        t_stop_done_ns=t_stop_done_ns,
                        t_stream_inactive_ns=t_stream_inactive_ns,
                        t_stream_closed_ns=t_stream_closed_ns,
                        t_stream_reopened_ns=t_stream_reopened_ns,
                        latency_ns=latency_ns,
                        latency_physical_ns=latency_physical_ns,
                        rms=feats.rms,
                        zcr=feats.zcr,
                        centroid_hz=feats.spectral_centroid_hz,
                        voicing=feats.voicing,
                        interrupt_cue=decision.interrupt_cue,
                        cue_strength=decision.cue_strength,
                        threshold=decision.threshold,
                        reason=decision.reason,
                        stop_mode=str(args.stop_mode),
                    )
                )

                if args.stop_mode == "close_reopen":
                    pending_reopen_event_idx = len(stop_events) - 1

    finally:
        with out_path.open("w", encoding="utf-8") as f:
            for ev in stop_events:
                f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")

        if input_stream is not None:
            try:
                input_stream.stop_stream()
                input_stream.close()
            except Exception:
                pass

        if output_stream is not None:
            try:
                output_stream.stop_stream()
                output_stream.close()
            except Exception:
                pass

        pa.terminate()

    summary = summarize_latencies_ns([e.latency_ns for e in stop_events])
    print("Stop events:", summary)
    print("Wrote:", os.fspath(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
