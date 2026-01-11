from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Row:
    n: int
    steps: int
    washout: int
    delays: str
    update_every: int
    update_mode: str
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


def _get(d: dict[str, str], key: str, default: str = "") -> str:
    return d.get(key, default)


def _load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Row(
                    n=int(float(_get(r, "n", "0"))),
                    steps=int(float(_get(r, "steps", "0"))),
                    washout=int(float(_get(r, "washout", "0"))),
                    delays=_get(r, "delays", ""),
                    update_every=int(float(_get(r, "update_every", "0"))),
                    update_mode=_get(r, "update_mode", ""),
                    lam=float(_get(r, "lam", "0")),
                    delta=float(_get(r, "delta", "0")),
                    seed=int(float(_get(r, "seed", "0"))),
                    input_mode=_get(r, "input_mode", ""),
                    input_params=_get(r, "input_params", ""),
                    input_bias=float(_get(r, "input_bias", "0")),
                    U=float(_get(r, "U", "0")),
                    tauF_ms=float(_get(r, "tauF_ms", "0")),
                    tauD_ms=float(_get(r, "tauD_ms", "0")),
                    w_scale=float(_get(r, "w_scale", "0")),
                    sparsity=float(_get(r, "sparsity", "0")),
                    input_scale=float(_get(r, "input_scale", "0")),
                    recurrence_source=_get(r, "recurrence_source", ""),
                    v_threshold=float(_get(r, "v_threshold", "0")),
                    v_reset=float(_get(r, "v_reset", "0")),
                    dc_bias=float(_get(r, "dc_bias", "0")),
                    spike_rate=float(_get(r, "spike_rate", "nan")),
                    state_mode=_get(r, "state_mode", ""),
                    state_dim=int(float(_get(r, "state_dim", "0"))),
                    proj_out_dim=int(float(_get(r, "proj_out_dim", "0"))),
                    proj_seed=int(float(_get(r, "proj_seed", "0"))),
                    mc_online=float(_get(r, "mc_online", "0")),
                    r2_by_delay_json=_get(r, "r2_by_delay_json", "{}"),
                )
            )
    return rows


def _parse_r2_map(s: str) -> dict[int, float]:
    try:
        d = json.loads(s)
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}

    out: dict[int, float] = {}
    for k, v in d.items():
        try:
            kk = int(k)
            vv = float(v)
        except Exception:
            continue
        out[kk] = vv
    return out


def generate_report_markdown(csv_path: Path) -> str:
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError("no rows")

    uniq_n = sorted({r.n for r in rows})
    uniq_steps = sorted({r.steps for r in rows})
    uniq_washout = sorted({r.washout for r in rows})
    uniq_delays = sorted({r.delays for r in rows})
    uniq_update_every = sorted({r.update_every for r in rows})
    uniq_update_mode = sorted({r.update_mode for r in rows})
    uniq_lam = sorted({r.lam for r in rows})
    uniq_delta = sorted({r.delta for r in rows})
    uniq_proj_out_dim = sorted({r.proj_out_dim for r in rows})
    uniq_state_mode = sorted({r.state_mode for r in rows})

    # Grouping key (ignore seed)
    key_type = tuple[Any, ...]
    buckets_mc: dict[key_type, list[float]] = {}
    buckets_r2: dict[key_type, list[dict[int, float]]] = {}
    buckets_sr: dict[key_type, list[float]] = {}

    for r in rows:
        key: key_type = (
            r.delays,
            r.update_every,
            r.update_mode,
            r.lam,
            r.delta,
            r.n,
            r.steps,
            r.washout,
            r.input_mode,
            r.input_params,
            r.input_bias,
            r.U,
            r.tauF_ms,
            r.tauD_ms,
            r.w_scale,
            r.sparsity,
            r.input_scale,
            r.recurrence_source,
            r.v_threshold,
            r.v_reset,
            r.dc_bias,
            r.state_mode,
            r.state_dim,
            r.proj_out_dim,
            r.proj_seed,
        )
        buckets_mc.setdefault(key, []).append(r.mc_online)
        buckets_r2.setdefault(key, []).append(_parse_r2_map(r.r2_by_delay_json))
        if not math.isnan(r.spike_rate):
            buckets_sr.setdefault(key, []).append(r.spike_rate)

    scored: list[tuple[float, float, int, key_type]] = []
    for key, mcs in buckets_mc.items():
        arr = np.asarray(mcs, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
        scored.append((mean, std, int(arr.size), key))

    scored.sort(key=lambda t: t[0], reverse=True)

    def _fmt_list(xs: list[float | int | str]) -> str:
        if len(xs) == 1:
            return str(xs[0])
        if len(xs) <= 6:
            return ", ".join(str(x) for x in xs)
        return f"{xs[0]}..{xs[-1]} (n={len(xs)})"

    lines: list[str] = []
    lines.append("# Phase 2: Online RLS レポート")
    lines.append("")
    lines.append("このレポートは、Phase 2 Step 2.4（オンライン学習）の結果CSVを集計して、")
    lines.append("**RLS更新が安定しているか（mc_online / delay別R^2）** を見るためのものです。")
    lines.append("")
    lines.append(f"- 入力CSV: {csv_path.as_posix()}")
    lines.append(f"- 総試行数: {len(rows)}")
    lines.append(f"- ユニーク条件数: {len(scored)}")
    lines.append("")

    lines.append("## 今回の条件（共通設定の一覧）")
    lines.append("")
    lines.append(f"- n: {_fmt_list(uniq_n)}")
    lines.append(f"- steps: {_fmt_list(uniq_steps)}")
    lines.append(f"- washout: {_fmt_list(uniq_washout)}")
    lines.append(f"- delays: {_fmt_list(uniq_delays)}")
    lines.append(f"- update_every: {_fmt_list(uniq_update_every)}")
    lines.append(f"- update_mode: {_fmt_list(uniq_update_mode)}")
    lines.append(f"- lam: {_fmt_list([f'{x:.6g}' for x in uniq_lam])}")
    lines.append(f"- delta: {_fmt_list([f'{x:.6g}' for x in uniq_delta])}")
    lines.append(f"- state_mode: {_fmt_list(uniq_state_mode)}")
    lines.append(f"- proj_out_dim (0=full): {_fmt_list(uniq_proj_out_dim)}")
    lines.append("")

    lines.append("## Top candidates (mean mc_online)")
    lines.append("")
    lines.append("表の見方:")
    lines.append("- **mc_online**: delay別のR^2を足し合わせた指標（負のR^2は0扱い）")
    lines.append("- **std**: ばらつき（小さいほど安定）")
    lines.append("- **n_runs**: 平均を取った回数（seed数）")
    lines.append("")

    cols = [
        "rank",
        "mc_online(mean)",
        "std",
        "n_runs",
        "delays",
        "update_every",
        "update_mode",
        "lam",
        "delta",
        "state_mode",
        "proj_out_dim",
        "spike_rate(mean)",
    ]
    aligns = ["---:", "---:", "---:", "---:", ":--", "---:", ":--", "---:", "---:", ":--", "---:", "---:"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(aligns) + "|")

    topk = min(10, len(scored))
    for i in range(topk):
        mean, std, n_runs, key = scored[i]
        (
            delays_s,
            update_every,
            update_mode,
            lam,
            delta,
            _n,
            _steps,
            _washout,
            *_rest,
            state_mode,
            _state_dim,
            proj_out_dim,
            _proj_seed,
        ) = key

        srs = buckets_sr.get(key, [])
        sr_mean = float(np.mean(srs)) if srs else float("nan")

        lines.append(
            "| "
            + " | ".join(
                [
                    str(i + 1),
                    f"{mean:.4f}",
                    f"{std:.4f}",
                    str(n_runs),
                    str(delays_s),
                    str(update_every),
                    str(update_mode),
                    f"{float(lam):.6g}",
                    f"{float(delta):.6g}",
                    str(state_mode),
                    str(int(proj_out_dim)),
                    (f"{sr_mean:.6f}" if not math.isnan(sr_mean) else ""),
                ]
            )
            + " |"
        )

    # Also show delay-wise mean/std for the best condition
    lines.append("")
    lines.append("## Best condition: delay別 R^2")
    lines.append("")
    if scored:
        best_key = scored[0][3]
        maps = buckets_r2.get(best_key, [])
        if maps:
            all_delays = sorted({d for m in maps for d in m.keys()})
            if all_delays:
                lines.append("| delay | R^2(mean) | R^2(std) |")
                lines.append("|---:|---:|---:|")
                for d in all_delays:
                    vals = [m.get(d, float("nan")) for m in maps]
                    vals = [v for v in vals if not math.isnan(v)]
                    if not vals:
                        continue
                    arr = np.asarray(vals, dtype=np.float64)
                    mean = float(arr.mean())
                    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
                    lines.append(f"| {d} | {mean:.4f} | {std:.4f} |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate markdown report for RLS online CSV")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = generate_report_markdown(csv_path)
    out_path.write_text(md, encoding="utf-8")
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
