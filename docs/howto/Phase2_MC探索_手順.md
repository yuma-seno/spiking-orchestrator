# Phase 2: MC（Memory Capacity）探索 手順

この手順は、仕様書の Phase 2 / Step 2.2 に相当する「短期記憶（STP）の強さ」を MC で測り、
パラメータ探索で良い候補を見つけるためのものです。

## 目的
- STPリザーバの **Memory Capacity (MC)** を計測する
- $\tau_F, \tau_D, W_{scale}$ を探索し、短期記憶性能を最大化する

## 実行（単発）
スモール設定（まず動作確認）:
- `python -m sorch.bench.mc_sweep --n 200 --steps 5000 --washout 500 --max-delay 200 --tauF-ms 200 --tauD-ms 1000 --w-scale 1.0 --out outputs/phase2/mc/runs/phase2_mc.csv`

より短いスモーク（まずMCが0張り付きでないことを確認）:
- `python -m sorch.bench.mc_sweep --n 50 --steps 2000 --washout 200 --max-delay 50 --tauF-ms 200 --tauD-ms 1000 --w-scale 1.0 --out outputs/phase2/mc/runs/phase2_mc_smoke.csv`

## パラメータ探索の進め方（おすすめ）
- 粗い探索 → 良かった近傍を細かく探索
- $W_{scale}$ は 0.8–1.1 を中心に調整

### グリッド探索（まとめて回す）
- `python -m sorch.bench.mc_search --n 200 --steps 8000 --washout 800 --max-delay 200 \
    --tauF-ms 100:800:100 --tauD-ms 500:3000:500 --w-scale 0.8:1.1:0.1 \
  --out outputs/phase2/mc/runs/phase2_mc_grid.csv`

指定形式:
- `--tauF-ms` / `--tauD-ms` / `--w-scale`
  - カンマ区切り: `100,200,400`
  - レンジ: `start:stop:step`（例 `100:800:100`）

### 2段階探索（粗→細）
粗いグリッドで上位候補を取り、その近傍を自動で細かく再探索します。

- スモーク:
  - `python -m sorch.bench.mc_refine --n 50 --steps 1500 --washout 200 --max-delay 50 \
      --coarse-tauF-ms 200,400 --coarse-tauD-ms 1000,2000 --coarse-w-scale 1.0 \
  --topk 2 --out outputs/phase2/mc/runs/phase2_mc_refine_smoke.csv`

- 例（大規模・repeatsで平均化）:
  - `python -m sorch.bench.mc_refine --n 600 --steps 16000 --washout 1600 --max-delay 400 --repeats 5 \
      --U 0.8 --recurrence-source spike --v-threshold 0.5 --dc-bias 0.3 \
      --coarse-tauF-ms 100:800:100 --coarse-tauD-ms 500:3000:500 --coarse-w-scale 0.8:1.1:0.1 \
      --topk 8 \
      --fine-tauF-step-ms 50 --fine-tauF-span-ms 100 \
      --fine-tauD-step-ms 250 --fine-tauD-span-ms 500 \
      --fine-w-step 0.05 --fine-w-span 0.10 \
  --out outputs/phase2/mc/runs/phase2_mc_refine_large.csv`

## 出力（どこに何が出る？）
- 生データ（CSV）: [outputs/phase2/mc/runs/](../../outputs/phase2/mc/runs/)
- レポート（Markdown）: [outputs/phase2/mc/reports/](../../outputs/phase2/mc/reports/)

## 結果の集計（Top候補レポート）
- 1本のCSVを集計:
  - `python -m sorch.bench.mc_report --csv outputs/phase2/mc/runs/phase2_mc_grid.csv --out outputs/phase2/mc/reports/phase2_mc_grid_report.md`

- まとめて再生成（おすすめ・丁寧版の統一）:
  - `python -m sorch.bench.mc_report_all --overwrite --topk 15`

## 注意（よくあるつまずき）
- `spike_rate=0`（発火しない）と、STPパラメータ差がMCに反映されにくい
  - 対策: `--v-threshold` を下げる / `--input-bias` を付ける / `--dc-bias` で定常電流を入れる
