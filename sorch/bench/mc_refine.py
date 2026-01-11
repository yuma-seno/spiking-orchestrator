from __future__ import annotations

import argparse
import csv
import itertools
import math
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sorch.core.mc_experiment import generate_input
from sorch.core.memory_capacity import memory_capacity
from sorch.core.reservoir_stp import ReservoirConfig, STPReservoir
from sorch.core.stp_lif_node import STPConfig


def _parse_list_or_range(spec: str) -> list[float]:
    s = spec.strip()
    if not s:
        raise ValueError("empty spec")

    if "," in s:
        out = [float(x.strip()) for x in s.split(",") if x.strip()]
        if not out:
            raise ValueError("no values")
        return out

    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) != 3:
            raise ValueError("range must be start:stop:step")
        start, stop, step = map(float, parts)
        if step == 0:
            raise ValueError("step must be non-zero")
        n = int(math.floor((stop - start) / step))
        values = [start + i * step for i in range(n + 1)]
        if step > 0:
            values = [v for v in values if v <= stop + 1e-9]
        else:
            values = [v for v in values if v >= stop - 1e-9]
        if not values:
            raise ValueError("range produced no values")
        return values

    return [float(s)]


def _refine_around(center: float, *, step: float, span: float, bounds: tuple[float, float] | None = None) -> list[float]:
    """Create a refined 1D grid around `center`.

    Returns sorted unique values.
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    if span < 0:
        raise ValueError("span must be >= 0")

    start = center - span
    stop = center + span

    if bounds is not None:
        lo, hi = bounds
        start = max(start, lo)
        stop = min(stop, hi)

    # Build inclusive grid.
    n = int(math.floor((stop - start) / step))
    vals = [start + i * step for i in range(n + 1)]
    # Ensure center included (within tolerance)
    if all(abs(v - center) > 1e-9 for v in vals):
        vals.append(center)

    # Round to stable decimals to avoid dupes due to float error.
    vals = sorted({round(v, 9) for v in vals})
    return vals


@dataclass(frozen=True, slots=True)
class SearchRow:
    stage: str
    tauF_ms: float
    tauD_ms: float
    w_scale: float
    U: float
    seed: int
    n: int
    steps: int
    washout: int
    max_delay: int
    ridge: float
    input_mode: str
    input_params: str
    input_bias: float
    sparsity: float
    input_scale: float
    recurrence_source: str
    v_threshold: float
    v_reset: float
    spike_rate: float
    dc_bias: float
    mc: float


def _run_grid(
    *,
    stage: str,
    tauF_list: list[float],
    tauD_list: list[float],
    w_list: list[float],
    args: argparse.Namespace,
) -> list[SearchRow]:
    rows: list[SearchRow] = []
    warned_no_spikes = False

    for rep in range(int(args.repeats)):
        seed = int(args.seed) + rep
        u = generate_input(
            seed=seed,
            steps=int(args.steps),
            mode=str(args.input_mode),
            input_bias=float(args.input_bias),
            low=float(args.input_low),
            high=float(args.input_high),
            std=float(args.input_std),
            clip=float(args.input_clip),
            tempo_on_min=int(args.tempo_on_min),
            tempo_on_max=int(args.tempo_on_max),
            tempo_off_min=int(args.tempo_off_min),
            tempo_off_max=int(args.tempo_off_max),
            tempo_amp=float(args.tempo_amp),
        )

        input_params = json.dumps(
            {
                "mode": str(args.input_mode),
                "low": float(args.input_low),
                "high": float(args.input_high),
                "std": float(args.input_std),
                "clip": float(args.input_clip),
                "tempo_on_min": int(args.tempo_on_min),
                "tempo_on_max": int(args.tempo_on_max),
                "tempo_off_min": int(args.tempo_off_min),
                "tempo_off_max": int(args.tempo_off_max),
                "tempo_amp": float(args.tempo_amp),
            },
            sort_keys=True,
            ensure_ascii=False,
        )

        for tauF_ms, tauD_ms, w_scale in itertools.product(tauF_list, tauD_list, w_list):
            stp = STPConfig(U=float(args.U), tau_F=float(tauF_ms) / 1000.0, tau_D=float(tauD_ms) / 1000.0)
            cfg = ReservoirConfig(
                n=int(args.n),
                input_scale=float(args.input_scale),
                w_scale=float(w_scale),
                sparsity=float(args.sparsity),
                seed=seed,
                stp=stp,
                recurrence_source=str(args.recurrence_source),
                v_threshold=float(args.v_threshold),
                v_reset=float(args.v_reset),
                dc_bias=float(args.dc_bias),
            )

            res = STPReservoir(cfg)
            states = res.run(u)
            spike_rate = float(res.last_spike_rate)

            if (not warned_no_spikes) and spike_rate <= 0.0 and str(args.recurrence_source) == "spike":
                print(
                    "WARNING: spike_rate=0. STP (tauF/tauD/U) won't affect dynamics when recurrence_source='spike'. "
                    "Try --preset convo_spiking, or lower --v-threshold / raise --dc-bias / raise --input-scale, "
                    "or switch --recurrence-source v."
                )
                warned_no_spikes = True
            mc_res = memory_capacity(
                states,
                u,
                washout=int(args.washout),
                max_delay=int(args.max_delay),
                ridge=float(args.ridge),
            )

            row = SearchRow(
                stage=stage,
                tauF_ms=float(tauF_ms),
                tauD_ms=float(tauD_ms),
                w_scale=float(w_scale),
                U=float(args.U),
                seed=seed,
                n=int(args.n),
                steps=int(args.steps),
                washout=int(args.washout),
                max_delay=int(args.max_delay),
                ridge=float(args.ridge),
                input_mode=str(args.input_mode),
                input_params=input_params,
                input_bias=float(args.input_bias),
                sparsity=float(args.sparsity),
                input_scale=float(args.input_scale),
                recurrence_source=str(args.recurrence_source),
                v_threshold=float(args.v_threshold),
                v_reset=float(args.v_reset),
                spike_rate=spike_rate,
                dc_bias=float(args.dc_bias),
                mc=float(mc_res.mc),
            )
            rows.append(row)
            print(
                f"{stage} seed={seed} tauF={tauF_ms:.0f}ms tauD={tauD_ms:.0f}ms w={w_scale:.3f} -> MC={row.mc:.3f}"
            )
            if getattr(args, "log_spike_rate", False):
                print(f"  spike_rate={spike_rate:.6f}")

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="SORCH Phase2: MC 2-stage refine (coarse -> fine)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--washout", type=int, default=800)
    ap.add_argument("--max-delay", type=int, default=200)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=["none", "convo", "convo_spiking"],
        help="convenience preset (overrides input-related args; *_spiking also sets neuron defaults)",
    )
    ap.add_argument(
        "--input-mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian", "burst", "convo"],
        help="input signal mode for u(t)",
    )
    ap.add_argument("--input-bias", type=float, default=0.0, help="added to u(t) to shift mean")
    ap.add_argument("--input-low", type=float, default=-1.0, help="uniform: low")
    ap.add_argument("--input-high", type=float, default=1.0, help="uniform: high")
    ap.add_argument("--input-std", type=float, default=0.5, help="gaussian/burst/convo: std")
    ap.add_argument("--input-clip", type=float, default=1.0, help="clip abs(u) if > 0")
    ap.add_argument("--tempo-on-min", type=int, default=20, help="burst/convo: on duration min (steps)")
    ap.add_argument("--tempo-on-max", type=int, default=200, help="burst/convo: on duration max (steps)")
    ap.add_argument("--tempo-off-min", type=int, default=20, help="burst/convo: off duration min (steps)")
    ap.add_argument("--tempo-off-max", type=int, default=200, help="burst/convo: off duration max (steps)")
    ap.add_argument("--tempo-amp", type=float, default=1.0, help="burst/convo: amplitude multiplier")

    ap.add_argument("--U", type=float, default=0.2)

    ap.add_argument("--coarse-tauF-ms", type=str, default="100:800:100")
    ap.add_argument("--coarse-tauD-ms", type=str, default="500:3000:500")
    ap.add_argument("--coarse-w-scale", type=str, default="0.8:1.1:0.1")

    # Fine grid settings (around each top candidate)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--fine-tauF-step-ms", type=float, default=50.0)
    ap.add_argument("--fine-tauF-span-ms", type=float, default=100.0)
    ap.add_argument("--fine-tauD-step-ms", type=float, default=250.0)
    ap.add_argument("--fine-tauD-span-ms", type=float, default=500.0)
    ap.add_argument("--fine-w-step", type=float, default=0.05)
    ap.add_argument("--fine-w-span", type=float, default=0.10)

    # Hard bounds (optional)
    ap.add_argument("--tauF-bounds-ms", type=str, default="100,800")
    ap.add_argument("--tauD-bounds-ms", type=str, default="500,3000")
    ap.add_argument("--w-bounds", type=str, default="0.8,1.1")

    ap.add_argument("--sparsity", type=float, default=0.1)
    ap.add_argument("--input-scale", type=float, default=0.5)
    ap.add_argument("--recurrence-source", type=str, default="spike", choices=["spike", "v"])
    ap.add_argument("--v-threshold", type=float, default=1.0)
    ap.add_argument("--v-reset", type=float, default=0.0)
    ap.add_argument("--dc-bias", type=float, default=0.0, help="constant bias current per neuron")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--log-spike-rate", action="store_true")

    ap.add_argument("--out", type=str, default="outputs/phase2/mc/runs/phase2_mc_refine.csv")
    args = ap.parse_args()

    if str(args.preset) in {"convo", "convo_spiking"}:
        args.input_mode = "convo"
        args.input_std = 0.6
        args.input_clip = 1.0
        args.tempo_on_min = 50
        args.tempo_on_max = 250
        args.tempo_off_min = 80
        args.tempo_off_max = 400
        args.tempo_amp = 1.0
    if str(args.preset) == "convo_spiking":
        args.recurrence_source = "spike"
        args.v_threshold = 0.25
        args.dc_bias = 0.25

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coarse_tauF = _parse_list_or_range(args.coarse_tauF_ms)
    coarse_tauD = _parse_list_or_range(args.coarse_tauD_ms)
    coarse_w = _parse_list_or_range(args.coarse_w_scale)

    tauF_lo, tauF_hi = map(float, args.tauF_bounds_ms.split(","))
    tauD_lo, tauD_hi = map(float, args.tauD_bounds_ms.split(","))
    w_lo, w_hi = map(float, args.w_bounds.split(","))

    # Stage 1: coarse
    rows_all: list[SearchRow] = []
    rows_coarse = _run_grid(stage="coarse", tauF_list=coarse_tauF, tauD_list=coarse_tauD, w_list=coarse_w, args=args)
    rows_all.extend(rows_coarse)

    # Select top-k unique candidates by parameter tuple (ignoring seed)
    rows_sorted = sorted(rows_coarse, key=lambda r: r.mc, reverse=True)
    seen: set[tuple[float, float, float]] = set()
    top: list[SearchRow] = []
    for r in rows_sorted:
        key = (r.tauF_ms, r.tauD_ms, r.w_scale)
        if key in seen:
            continue
        seen.add(key)
        top.append(r)
        if len(top) >= int(args.topk):
            break

    # Stage 2: fine around top candidates
    fine_tauF_vals: set[float] = set()
    fine_tauD_vals: set[float] = set()
    fine_w_vals: set[float] = set()

    for r in top:
        for v in _refine_around(
            r.tauF_ms,
            step=float(args.fine_tauF_step_ms),
            span=float(args.fine_tauF_span_ms),
            bounds=(tauF_lo, tauF_hi),
        ):
            fine_tauF_vals.add(v)
        for v in _refine_around(
            r.tauD_ms,
            step=float(args.fine_tauD_step_ms),
            span=float(args.fine_tauD_span_ms),
            bounds=(tauD_lo, tauD_hi),
        ):
            fine_tauD_vals.add(v)
        for v in _refine_around(
            r.w_scale,
            step=float(args.fine_w_step),
            span=float(args.fine_w_span),
            bounds=(w_lo, w_hi),
        ):
            fine_w_vals.add(v)

    fine_tauF = sorted(fine_tauF_vals)
    fine_tauD = sorted(fine_tauD_vals)
    fine_w = sorted(fine_w_vals)

    print(
        f"\nRefine grid sizes: tauF={len(fine_tauF)} tauD={len(fine_tauD)} w={len(fine_w)} (from topk={len(top)})"
    )

    rows_fine = _run_grid(stage="fine", tauF_list=fine_tauF, tauD_list=fine_tauD, w_list=fine_w, args=args)
    rows_all.extend(rows_fine)

    # Append all rows to CSV
    fieldnames = list(SearchRow.__annotations__.keys())
    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows_all:
            w.writerow({k: getattr(r, k) for k in fieldnames})

    best = max(rows_all, key=lambda r: r.mc)
    print(
        f"\nBest: MC={best.mc:.3f} stage={best.stage} seed={best.seed} tauF={best.tauF_ms:.0f}ms tauD={best.tauD_ms:.0f}ms w={best.w_scale:.3f}"
    )
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
