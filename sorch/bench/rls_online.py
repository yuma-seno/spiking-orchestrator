from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sorch.core.mc_experiment import generate_input
from sorch.core.memory_capacity import r2_score
from sorch.core.random_projection import RandomProjection
from sorch.core.reservoir_stp import ReservoirConfig, STPReservoir
from sorch.core.rls import RLS
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


def _parse_state_mode(mode: str) -> tuple[str, tuple[str, ...]]:
    allowed = {"v", "spike", "u", "r", "eff"}
    aliases = {"s": "spike"}

    raw = str(mode).strip()
    if not raw:
        raise ValueError("empty state mode")

    tokens = [t.strip() for t in raw.replace(",", "+").split("+") if t.strip()]
    tokens = [aliases.get(t, t) for t in tokens]
    unknown = [t for t in tokens if t not in allowed]
    if unknown:
        raise ValueError(f"unknown state token(s): {unknown}. allowed={sorted(allowed)}")

    seen: set[str] = set()
    comps: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        comps.append(t)
        seen.add(t)

    canonical = "+".join(comps)
    return canonical, tuple(comps)


def _zscore_states(states: np.ndarray, *, start: int) -> np.ndarray:
    X = np.asarray(states, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("states must be 2D")
    if start < 0 or start >= X.shape[0]:
        raise ValueError("invalid zscore start")

    ref = X[start:, :]
    mean = ref.mean(axis=0, keepdims=True)
    std = ref.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std


@dataclass(frozen=True, slots=True)
class Row:
    n: int
    steps: int
    washout: int
    delays: str
    update_every: int
    lam: float
    delta: float
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
    state_mode: str
    state_dim: int
    proj_out_dim: int
    proj_seed: int
    mc_online: float
    r2_by_delay_json: str


def main() -> int:
    ap = argparse.ArgumentParser(description="SORCH Phase2: Online RLS readout on reservoir states")

    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--washout", type=int, default=800)

    ap.add_argument(
        "--delays",
        type=str,
        default="1,5,10,20,40,80,120",
        help="delays (in steps) to reconstruct, e.g. '1,10,20' or '1:120:1'",
    )

    ap.add_argument(
        "--update-every",
        type=int,
        default=10,
        help="RLS update interval in steps (e.g., 10 means ~10ms if dt=1ms)",
    )
    ap.add_argument("--lam", type=float, default=0.995, help="forgetting factor (0<lam<=1)")
    ap.add_argument("--delta", type=float, default=1.0, help="Tikhonov regularization (>0)")

    ap.add_argument(
        "--preset",
        type=str,
        default="convo_spiking",
        choices=["none", "convo", "convo_spiking"],
        help="convenience preset (overrides input-related args; *_spiking also sets neuron defaults)",
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
        "--state-mode",
        type=str,
        default="v+spike",
        help="'+'-joined list among {v, spike, u, r, eff}. Examples: 'v', 'v+spike'",
    )
    ap.add_argument(
        "--state-zscore",
        action="store_true",
        help="z-score states per feature using the post-(washout+max_delay) window",
    )

    ap.add_argument("--proj-out-dim", type=int, default=200, help="0 means 'no projection'")
    ap.add_argument("--proj-seed", type=int, default=0)

    ap.add_argument("--out", type=str, default="outputs/phase2/rls/runs/phase2_rls_online.csv")
    args = ap.parse_args()

    delays = _parse_list_or_range(args.delays)
    delays = sorted({int(d) for d in delays if int(d) > 0})
    if not delays:
        raise ValueError("delays must include at least one positive integer")

    if int(args.update_every) <= 0:
        raise ValueError("update-every must be > 0")

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

    state_mode, comps = _parse_state_mode(args.state_mode)
    if set(comps).issubset({"v"}):
        traces: dict[str, np.ndarray] = {"v": reservoir.run(u)}
    else:
        order = [k for k in ("v", "spike", "u", "r", "eff") if k in set(comps)]
        traces = reservoir.run_traces(u, record=tuple(order))

    spike_rate = float(reservoir.last_spike_rate)
    if spike_rate <= 0.0 and str(args.recurrence_source) == "spike":
        print(
            "WARNING: spike_rate=0. STP (tauF/tauD/U) won't affect dynamics when recurrence_source='spike'. "
            "Try --preset convo_spiking, or lower --v-threshold / raise --dc-bias / raise --input-scale, "
            "or switch --recurrence-source v."
        )

    blocks = [np.asarray(traces[c], dtype=np.float32) for c in comps]
    states_full = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)
    state_dim = int(states_full.shape[1])

    start = int(args.washout) + int(max(delays))
    if start >= int(args.steps):
        raise ValueError("T too small for washout+max_delay")

    if bool(args.state_zscore):
        states_full = _zscore_states(states_full, start=start)

    if int(args.proj_out_dim) == 0:
        states = states_full
    else:
        rp = RandomProjection(in_dim=int(state_dim), out_dim=int(args.proj_out_dim), seed=int(args.proj_seed))
        states = rp.project(states_full)

    in_dim = int(states.shape[1])

    # Add bias term to x.
    rls = RLS(dim=in_dim + 1, out_dim=len(delays), lam=float(args.lam), delta=float(args.delta), dtype=np.float64)

    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    for t in range(start, int(args.steps)):
        x = np.asarray(states[t, :], dtype=np.float64)
        x_aug = np.concatenate([x, np.array([1.0], dtype=np.float64)], axis=0)

        y_t = np.array([u[t - d] for d in delays], dtype=np.float64)
        y_hat = rls.predict(x_aug).reshape(-1)

        y_true.append(y_t)
        y_pred.append(y_hat)

        if (t - start) % int(args.update_every) == 0:
            rls.update(x_aug, y_t)

    Y = np.stack(y_true, axis=0)
    Yh = np.stack(y_pred, axis=0)

    r2_by_delay: dict[int, float] = {}
    for i, d in enumerate(delays):
        r2_by_delay[int(d)] = float(r2_score(Y[:, i], Yh[:, i]))

    mc_online = float(sum(max(0.0, v) for v in r2_by_delay.values()))

    row = Row(
        n=int(cfg.n),
        steps=int(args.steps),
        washout=int(args.washout),
        delays=str(args.delays),
        update_every=int(args.update_every),
        lam=float(args.lam),
        delta=float(args.delta),
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
        spike_rate=float(spike_rate),
        state_mode=str(state_mode),
        state_dim=int(state_dim),
        proj_out_dim=int(args.proj_out_dim),
        proj_seed=int(args.proj_seed),
        mc_online=float(mc_online),
        r2_by_delay_json=json.dumps(r2_by_delay, ensure_ascii=False, sort_keys=True),
    )

    fieldnames = list(Row.__annotations__.keys())
    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: getattr(row, k) for k in fieldnames})

    print(f"{state_mode} online MC={mc_online:.3f} (delays={len(delays)}, update_every={int(args.update_every)})")
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
