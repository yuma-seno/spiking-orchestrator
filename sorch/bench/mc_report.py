from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Row:
    stage: str | None
    tauF_ms: float
    tauD_ms: float
    w_scale: float
    U: float
    recurrence_source: str
    v_threshold: float
    seed: int
    n: int
    steps: int
    washout: int
    max_delay: int
    ridge: float
    sparsity: float
    input_scale: float
    input_bias: float
    dc_bias: float
    spike_rate: float | None
    mc: float


def _get(d: dict[str, str], key: str, default: str = "") -> str:
    return d.get(key, default)


def _load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            stage = _get(r, "stage") or None
            spike_rate_s = _get(r, "spike_rate", "")
            spike_rate = float(spike_rate_s) if spike_rate_s != "" else None
            rows.append(
                Row(
                    stage=stage,
                    tauF_ms=float(_get(r, "tauF_ms", "0")),
                    tauD_ms=float(_get(r, "tauD_ms", "0")),
                    w_scale=float(_get(r, "w_scale", "0")),
                    U=float(_get(r, "U", "0")),
                    recurrence_source=_get(r, "recurrence_source", ""),
                    v_threshold=float(_get(r, "v_threshold", "0")),
                    seed=int(float(_get(r, "seed", "0"))),
                    n=int(float(_get(r, "n", "0"))),
                    steps=int(float(_get(r, "steps", "0"))),
                    washout=int(float(_get(r, "washout", "0"))),
                    max_delay=int(float(_get(r, "max_delay", "0"))),
                    ridge=float(_get(r, "ridge", "0")),
                    sparsity=float(_get(r, "sparsity", "0")),
                    input_scale=float(_get(r, "input_scale", "0")),
                    input_bias=float(_get(r, "input_bias", "0")),
                    dc_bias=float(_get(r, "dc_bias", "0")),
                    spike_rate=spike_rate,
                    mc=float(_get(r, "mc", "0")),
                )
            )
    return rows


def generate_report_markdown(csv_path: Path, topk: int = 10) -> str:
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError("no rows")

    # Common settings (help non-engineers understand what's being compared)
    uniq_n = sorted({r.n for r in rows})
    uniq_steps = sorted({r.steps for r in rows})
    uniq_washout = sorted({r.washout for r in rows})
    uniq_max_delay = sorted({r.max_delay for r in rows})
    uniq_ridge = sorted({r.ridge for r in rows})
    uniq_sparsity = sorted({r.sparsity for r in rows})

    has_spike_rate = any(r.spike_rate is not None and not math.isnan(r.spike_rate) for r in rows)

    # Aggregate by parameter tuple (ignoring seed/stage)
    buckets_mc: dict[tuple[float, float, float, float, str, float, float, float, float], list[float]] = {}
    buckets_sr: dict[tuple[float, float, float, float, str, float, float, float, float], list[float]] = {}
    for r in rows:
        key = (
            r.tauF_ms,
            r.tauD_ms,
            r.w_scale,
            r.U,
            r.recurrence_source,
            r.v_threshold,
            r.input_scale,
            r.input_bias,
            r.dc_bias,
        )
        buckets_mc.setdefault(key, []).append(r.mc)
        if r.spike_rate is not None and not math.isnan(r.spike_rate):
            buckets_sr.setdefault(key, []).append(r.spike_rate)

    scored: list[
        tuple[float, float, int, float | None, tuple[float, float, float, float, str, float, float, float, float]]
    ] = []
    for key, mcs in buckets_mc.items():
        mean = sum(mcs) / len(mcs)
        # naive std
        var = sum((x - mean) ** 2 for x in mcs) / max(1, len(mcs) - 1)
        std = var**0.5

        mean_sr: float | None = None
        if has_spike_rate:
            srs = buckets_sr.get(key, [])
            if srs:
                mean_sr = sum(srs) / len(srs)

        scored.append((mean, std, len(mcs), mean_sr, key))

    scored.sort(key=lambda t: t[0], reverse=True)

    lines: list[str] = []
    lines.append("# Phase 2: Memory Capacity (MC) レポート")
    lines.append("")
    lines.append("このレポートは、Phase 2 Step 2.2（MC最大化探索）の結果CSVを集計して、")
    lines.append("**良さそうなパラメータ候補（上位）**を見つけるための一覧です。")
    lines.append("")
    lines.append(f"- 入力CSV: {csv_path.as_posix()}")
    lines.append(f"- 総試行数: {len(rows)}")
    lines.append(f"- ユニーク条件数: {len(scored)}")
    lines.append("")

    lines.append("## これは何の数値？（MCの超ざっくり説明）")
    lines.append("")
    lines.append("- **MC（Memory Capacity）** は『少し前の入力を、今のリザーバ状態からどれだけ復元できるか』のスコアです。")
    lines.append("- 数値は **大きいほど短期記憶が強い** と解釈します（ただし比較は同じ設定同士で行います）。")
    lines.append("- 例え話/図解で掴みたい場合: docs/ガイド.md の「MC」節")
    lines.append("")

    lines.append("## 今回の実験条件（共通設定）")
    lines.append("")

    def _fmt_list(xs: list[float | int]) -> str:
        if len(xs) == 1:
            return str(xs[0])
        if len(xs) <= 6:
            return ", ".join(str(x) for x in xs)
        return f"{xs[0]}..{xs[-1]} (n={len(xs)})"

    lines.append(f"- ニューロン数 n: {_fmt_list(uniq_n)}")
    lines.append(f"- ステップ数 steps: {_fmt_list(uniq_steps)}（長いほど時間がかかる）")
    lines.append(f"- washout: {_fmt_list(uniq_washout)}（最初の過渡を捨てる）")
    lines.append(f"- max_delay: {_fmt_list(uniq_max_delay)}（何ステップ前まで復元するか）")
    lines.append(f"- ridge（回帰の正則化）: {_fmt_list(uniq_ridge)}")
    lines.append(f"- sparsity（結合の疎さ）: {_fmt_list(uniq_sparsity)}")
    lines.append("")

    if has_spike_rate:
        sr_all = [r.spike_rate for r in rows if r.spike_rate is not None and not math.isnan(r.spike_rate)]
        sr_mean = (sum(sr_all) / len(sr_all)) if sr_all else 0.0
        lines.append("## 発火率のチェック（任意だが重要）")
        lines.append("")
        lines.append(
            "- spike_rate は『1ステップ×1ニューロンあたりの平均発火率』です。0に近いと“動いていない”可能性があります。"
        )
        lines.append(f"- 全体の平均 spike_rate: {sr_mean:.6f}")
        lines.append("")

    lines.append("## Top candidates (mean MC)")
    lines.append("")
    lines.append("表の見方:")
    lines.append("- **mean MC**: 同じ条件を複数回（seed/repeats）走らせた平均")
    lines.append("- **std**: ばらつき（小さいほど安定）")
    lines.append("- **n_runs**: 平均を取った回数")
    if has_spike_rate:
        lines.append("- **mean spike_rate**: 同じ条件の平均発火率（0に近い場合は要注意）")
    lines.append("")

    if has_spike_rate:
        lines.append(
            "| rank | mean MC | std | n_runs | mean spike_rate | tauF_ms | tauD_ms | w_scale | U | rec | v_th | in_scale | in_bias | dc_bias |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|---:|---:|---:|---:|")
    else:
        lines.append(
            "| rank | mean MC | std | n_runs | tauF_ms | tauD_ms | w_scale | U | rec | v_th | in_scale | in_bias | dc_bias |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|:--|---:|---:|---:|---:|")

    topk = min(int(topk), len(scored))
    for i in range(topk):
        mean, std, n_runs, mean_sr, key = scored[i]
        tauF_ms, tauD_ms, w_scale, U, rec, v_th, in_scale, in_bias, dc_bias = key
        if has_spike_rate:
            sr_s = "" if mean_sr is None else f"{mean_sr:.6f}"
            lines.append(
                f"| {i+1} | {mean:.4f} | {std:.4f} | {n_runs} | {sr_s} | {tauF_ms:.0f} | {tauD_ms:.0f} | {w_scale:.3f} | {U:.3f} | {rec} | {v_th:.3f} | {in_scale:.3f} | {in_bias:.3f} | {dc_bias:.3f} |"
            )
        else:
            lines.append(
                f"| {i+1} | {mean:.4f} | {std:.4f} | {n_runs} | {tauF_ms:.0f} | {tauD_ms:.0f} | {w_scale:.3f} | {U:.3f} | {rec} | {v_th:.3f} | {in_scale:.3f} | {in_bias:.3f} | {dc_bias:.3f} |"
            )

    lines.append("")
    best_mean, best_std, best_n, best_sr, best_key = scored[0]
    tauF_ms, tauD_ms, w_scale, U, rec, v_th, in_scale, in_bias, dc_bias = best_key
    lines.append("## Best")
    lines.append("")
    lines.append(f"- mean MC={best_mean:.4f} (std={best_std:.4f}, n_runs={best_n})")
    if has_spike_rate and best_sr is not None:
        lines.append(f"- mean spike_rate={best_sr:.6f}")
    lines.append(f"- tauF={tauF_ms:.0f}ms, tauD={tauD_ms:.0f}ms, w_scale={w_scale:.3f}, U={U:.3f}")
    lines.append(f"- rec={rec}, v_threshold={v_th:.3f}")
    lines.append(f"- input_scale={in_scale:.3f}, input_bias={in_bias:.3f}, dc_bias={dc_bias:.3f}")

    lines.append("")
    lines.append("## パラメータの意味（超ざっくり）")
    lines.append("")
    lines.append("- tauF_ms / tauD_ms: STP（短期可塑性）の時定数。値で記憶の残り方が変わる")
    lines.append("- w_scale: リカレント結合の強さ（大きいほど影響が強いが、暴れやすいこともある）")
    lines.append("- U: STPの効きやすさ（大きいほど入力が通りやすい）")
    lines.append("- rec: 再帰に使う信号（spike or v）。spikeはSNNっぽいが死にやすいことがある")
    lines.append("- v_th: 発火しやすさ（小さいほど発火しやすい）")
    lines.append("- input_scale/input_bias: 入力の強さ/平均シフト")
    lines.append("- dc_bias: 一定のバイアス電流（発火を立ち上げるために使う）")

    lines.append("")
    lines.append("## 注意（比較のしかた）")
    lines.append("")
    lines.append("- MCは“絶対値”というより、**同じ共通設定の中での相対比較**が安全です。")
    lines.append("- stdが大きい候補は、seed次第で性能が落ちる可能性があるため、追加のrepeatsで確認してください。")
    if has_spike_rate:
        lines.append("- mean spike_rate が0に近い候補は“動いていない”可能性があるため、まず発火率を上げてから比較してください。")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Phase2 MC CSV and produce a markdown report")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out", type=str, default="outputs/phase2_mc_report.md")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = generate_report_markdown(csv_path, topk=int(args.topk))
    out_path.write_text(md, encoding="utf-8")
    print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
