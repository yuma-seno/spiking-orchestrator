from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sorch.core.mc_experiment import generate_input
from sorch.core.memory_capacity import memory_capacity
from sorch.core.reservoir_stp import ReservoirConfig, STPReservoir
from sorch.core.stp_lif_node import STPConfig


def main() -> int:
    ap = argparse.ArgumentParser(description="SORCH Phase2: Memory Capacity sweep")
    ap.add_argument("--n", type=int, default=200, help="neurons (use 500-1000 for full runs)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--washout", type=int, default=500)
    ap.add_argument("--max-delay", type=int, default=200)
    ap.add_argument("--ridge", type=float, default=1e-3)

    ap.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=["none", "convo"],
        help="convenience preset (overrides input-related args)",
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

    ap.add_argument("--tauF-ms", type=float, default=200.0)
    ap.add_argument("--tauD-ms", type=float, default=1000.0)
    ap.add_argument("--U", type=float, default=0.2)

    ap.add_argument("--w-scale", type=float, default=1.0)
    ap.add_argument("--sparsity", type=float, default=0.1)
    ap.add_argument("--input-scale", type=float, default=0.5)
    ap.add_argument("--recurrence-source", type=str, default="spike", choices=["spike", "v"])
    ap.add_argument("--v-threshold", type=float, default=1.0)
    ap.add_argument("--v-reset", type=float, default=0.0)
    ap.add_argument("--dc-bias", type=float, default=0.0, help="constant bias current per neuron")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/phase2_mc.csv")
    ap.add_argument("--log-spike-rate", action="store_true")

    args = ap.parse_args()

    if str(args.preset) == "convo":
        args.input_mode = "convo"
        args.input_std = 0.6
        args.input_clip = 1.0
        args.tempo_on_min = 50
        args.tempo_on_max = 250
        args.tempo_off_min = 80
        args.tempo_off_max = 400
        args.tempo_amp = 1.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    u = generate_input(
        seed=int(args.seed),
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

    stp = STPConfig(U=float(args.U), tau_F=float(args.tauF_ms) / 1000.0, tau_D=float(args.tauD_ms) / 1000.0)
    cfg = ReservoirConfig(
        n=int(args.n),
        input_scale=float(args.input_scale),
        w_scale=float(args.w_scale),
        sparsity=float(args.sparsity),
        seed=int(args.seed),
        stp=stp,
        recurrence_source=str(args.recurrence_source),
        v_threshold=float(args.v_threshold),
        v_reset=float(args.v_reset),
        dc_bias=float(args.dc_bias),
    )

    res = STPReservoir(cfg)
    states = res.run(u)
    spike_rate = float(res.last_spike_rate)

    mc_res = memory_capacity(states, u, washout=int(args.washout), max_delay=int(args.max_delay), ridge=float(args.ridge))

    row = {
        "n": cfg.n,
        "steps": int(args.steps),
        "washout": int(args.washout),
        "max_delay": int(args.max_delay),
        "ridge": float(args.ridge),
        "input_mode": str(args.input_mode),
        "input_params": input_params,
        "input_bias": float(args.input_bias),
        "U": stp.U,
        "tauF_ms": float(args.tauF_ms),
        "tauD_ms": float(args.tauD_ms),
        "w_scale": cfg.w_scale,
        "sparsity": cfg.sparsity,
        "input_scale": cfg.input_scale,
        "recurrence_source": cfg.recurrence_source,
        "v_threshold": cfg.v_threshold,
        "v_reset": cfg.v_reset,
        "seed": cfg.seed,
        "spike_rate": spike_rate,
        "dc_bias": cfg.dc_bias,
        "mc": mc_res.mc,
    }

    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print("MC:", mc_res.mc)
    if args.log_spike_rate:
        print("Spike rate:", spike_rate)
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
