from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

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

    ap.add_argument("--input-bias", type=float, default=0.0, help="added to u(t) to shift mean")

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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    u = rng.uniform(low=-1.0, high=1.0, size=int(args.steps)).astype(np.float32)
    if float(args.input_bias) != 0.0:
        u = (u + float(args.input_bias)).astype(np.float32)

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
