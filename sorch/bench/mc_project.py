from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from sorch.core.mc_experiment import generate_input
from sorch.core.memory_capacity import memory_capacity
from sorch.core.random_projection import RandomProjection
from sorch.core.reservoir_stp import ReservoirConfig, STPReservoir
from sorch.core.stp_lif_node import STPConfig


def _parse_list_or_range(spec: str) -> list[int]:
    s = spec.strip()
    if not s:
        raise ValueError("empty spec")

    if "," in s:
        out = [int(x.strip()) for x in s.split(",") if x.strip()]
        if not out:
            raise ValueError("no values")
        return out

    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) != 3:
            raise ValueError("range must be start:stop:step")
        start, stop, step = map(int, parts)
        if step == 0:
            raise ValueError("step must be non-zero")
        n = int(math.floor((stop - start) / step))
        values = [start + i * step for i in range(n + 1)]
        if step > 0:
            values = [v for v in values if v <= stop]
        else:
            values = [v for v in values if v >= stop]
        if not values:
            raise ValueError("range produced no values")
        return values

    return [int(s)]


@dataclass(frozen=True, slots=True)
class Row:
    n: int
    steps: int
    washout: int
    max_delay: int
    ridge: float
    seed: int
    input_mode: str
    input_params: str
    input_bias: float
    U: float
    tauF_ms: float
    tauD_ms: float
    w_scale: float
    sparsity: float
    input_scale: float
    recurrence_source: str
    v_threshold: float
    v_reset: float
    dc_bias: float
    spike_rate: float
    proj_out_dim: int
    proj_seed: int
    mc: float


def main() -> int:
    ap = argparse.ArgumentParser(description="SORCH Phase2: MC with random projection")

    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--washout", type=int, default=800)
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
    )
    ap.add_argument("--input-bias", type=float, default=0.0)
    ap.add_argument("--input-low", type=float, default=-1.0)
    ap.add_argument("--input-high", type=float, default=1.0)
    ap.add_argument("--input-std", type=float, default=0.5)
    ap.add_argument("--input-clip", type=float, default=1.0)
    ap.add_argument("--tempo-on-min", type=int, default=20)
    ap.add_argument("--tempo-on-max", type=int, default=200)
    ap.add_argument("--tempo-off-min", type=int, default=20)
    ap.add_argument("--tempo-off-max", type=int, default=200)
    ap.add_argument("--tempo-amp", type=float, default=1.0)

    ap.add_argument("--U", type=float, default=0.2)
    ap.add_argument("--tauF-ms", type=float, default=200.0)
    ap.add_argument("--tauD-ms", type=float, default=1000.0)

    ap.add_argument("--w-scale", type=float, default=1.0)
    ap.add_argument("--sparsity", type=float, default=0.1)
    ap.add_argument("--input-scale", type=float, default=0.5)
    ap.add_argument("--recurrence-source", type=str, default="spike", choices=["spike", "v"])
    ap.add_argument("--v-threshold", type=float, default=1.0)
    ap.add_argument("--v-reset", type=float, default=0.0)
    ap.add_argument("--dc-bias", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=0, help="reservoir+input seed")

    ap.add_argument(
        "--proj-dims",
        type=str,
        default="0,50,100,200",
        help="projection output dims. include 0 to mean 'no projection'",
    )
    ap.add_argument("--proj-seed", type=int, default=0)

    ap.add_argument("--out", type=str, default="outputs/phase2/mc/runs/phase2_mc_project.csv")
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

    proj_dims = _parse_list_or_range(args.proj_dims)
    if any(d < 0 for d in proj_dims):
        raise ValueError("proj dims must be >= 0")

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

    reservoir = STPReservoir(cfg)
    states_full = reservoir.run(u)
    spike_rate = float(reservoir.last_spike_rate)

    rows: list[Row] = []
    for out_dim in proj_dims:
        if int(out_dim) == 0:
            states = states_full
        else:
            rp = RandomProjection(in_dim=int(cfg.n), out_dim=int(out_dim), seed=int(args.proj_seed))
            states = rp.project(states_full)

        mc_res = memory_capacity(
            states,
            u,
            washout=int(args.washout),
            max_delay=int(args.max_delay),
            ridge=float(args.ridge),
        )

        rows.append(
            Row(
                n=int(cfg.n),
                steps=int(args.steps),
                washout=int(args.washout),
                max_delay=int(args.max_delay),
                ridge=float(args.ridge),
                seed=int(args.seed),
                input_mode=str(args.input_mode),
                input_params=input_params,
                input_bias=float(args.input_bias),
                U=float(stp.U),
                tauF_ms=float(args.tauF_ms),
                tauD_ms=float(args.tauD_ms),
                w_scale=float(cfg.w_scale),
                sparsity=float(cfg.sparsity),
                input_scale=float(cfg.input_scale),
                recurrence_source=str(cfg.recurrence_source),
                v_threshold=float(cfg.v_threshold),
                v_reset=float(cfg.v_reset),
                dc_bias=float(cfg.dc_bias),
                spike_rate=spike_rate,
                proj_out_dim=int(out_dim),
                proj_seed=int(args.proj_seed),
                mc=float(mc_res.mc),
            )
        )

        tag = "full" if int(out_dim) == 0 else f"rp{int(out_dim)}"
        print(f"{tag}: MC={mc_res.mc:.3f}")

    fieldnames = list(Row.__annotations__.keys())
    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fieldnames})

    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
