# Phase 2: MC（Memory Capacity）探索 手順

この手順は [docs/spec/仕様書.md](../spec/%E4%BB%95%E6%A7%98%E6%9B%B8.md) の Phase 2 / Step 2.2 に対応します。

目的:
- STPリザーバの短期記憶性能を **MC** として数値化し、探索で良い設定を見つける

用語:
- MC / STP / Readout: [docs/project/用語集.md](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md)

## 実行（例）

スモーク（まず動作確認）:
- `python -m sorch.bench.mc_sweep --n 50 --steps 2000 --washout 200 --max-delay 50 --tauF-ms 200 --tauD-ms 1000 --w-scale 1.0 --out outputs/phase2/mc/runs/phase2_mc_smoke.csv`

単発:
- `python -m sorch.bench.mc_sweep --n 200 --steps 5000 --washout 500 --max-delay 200 --tauF-ms 200 --tauD-ms 1000 --w-scale 1.0 --out outputs/phase2/mc/runs/phase2_mc.csv`

探索（レンジ指定）:
- `python -m sorch.bench.mc_search --n 200 --steps 8000 --washout 800 --max-delay 200 \
    --tauF-ms 100:800:100 --tauD-ms 500:3000:500 --w-scale 0.8:1.1:0.1 \
  --out outputs/phase2/mc/runs/phase2_mc_grid.csv`

レポート化:
- `python -m sorch.bench.mc_report --csv outputs/phase2/mc/runs/phase2_mc_grid.csv --out outputs/phase2/mc/reports/phase2_mc_grid_report.md`

## 出力
- CSV: `outputs/phase2/mc/runs/`
- レポート: `outputs/phase2/mc/reports/`

