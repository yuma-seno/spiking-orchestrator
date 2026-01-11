# Phase 2: MC（Memory Capacity）探索 手順

この手順は、仕様書の Phase 2 / Step 2.2 に相当する「短期記憶（STP）の強さ」を MC で測り、
パラメータ探索で良い候補を見つけるためのものです。

## 目的
- STPリザーバの **Memory Capacity (MC)** を計測する
- $\tau_F, \tau_D, W_{scale}$ を探索し、短期記憶性能を最大化する

用語:
- STP: [用語集](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md#glossary-stp)
- MC: [用語集](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md#glossary-mc)

## 実行（単発）
スモール設定（まず動作確認）:
- `python -m sorch.bench.mc_sweep --n 200 --steps 5000 --washout 500 --max-delay 200 --tauF-ms 200 --tauD-ms 1000 --w-scale 1.0 --out outputs/phase2/mc/runs/phase2_mc.csv`

convo入力（疑似会話テンポ）で、かつ **発火を出したい** とき（STP差を見やすくする）:
- `python -m sorch.bench.mc_sweep --preset convo_spiking --n 200 --steps 5000 --washout 500 --max-delay 200 --tauF-ms 200 --tauD-ms 1000 --w-scale 1.0 --out outputs/phase2/mc/runs/phase2_mc_convo_spiking.csv`

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

---

## Step 2.3（Readout/次元削減）: 状態ベクトル設計（state_mode）

`mc_sweep` / `mc_search` / `mc_refine` は基本的に状態として膜電位 `v` を使いますが、
Readoutに渡す状態ベクトルを **spike** や **STP内部状態（u/r/eff）** まで含めると、
MCが大きく伸びることがあります（その分、次元が増えるため Random Projection とセットで検討します）。

先に用語だけ押さえる（おすすめ）:
- state_mode（状態ベクトルの選び方）: [用語集](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md#glossary-state-mode)
- 状態ベクトル（state vector）: [用語集](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md#glossary-state-vector)
- Random Projection: [用語集](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md#glossary-random-projection)
- washout: [用語集](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md#glossary-washout)
- max_delay: [用語集](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md#glossary-max-delay)
- ridge: [用語集](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md#glossary-ridge)
- spike_rate: [用語集](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md#glossary-spike-rate)

### 状態ベクトルを切り替えて比較する（おすすめ）
- スモーク（seed=0、投影なし）:
  - `python -m sorch.bench.mc_project --preset convo_spiking --n 120 --steps 3000 --washout 300 --max-delay 120 --seed 0 \
      --proj-dims 0 \
      --state-modes 'v,spike,v+spike,v+u+r+eff,v+spike+u+r+eff' \
    --out outputs/phase2/mc/runs/phase2_mc_state_modes_smoke.csv`

- repeats（seed=0..4の例）:
  - `for s in 0 1 2 3 4; do python -m sorch.bench.mc_project --preset convo_spiking --n 120 --steps 3000 --washout 300 --max-delay 120 --seed $s \
      --proj-dims 0 \
      --state-modes 'v,spike,v+spike,v+u+r+eff,v+spike+u+r+eff' \
    --out outputs/phase2/mc/runs/phase2_mc_state_modes_smoke.csv; done`

### 次元削減とセットで確認する（例）
状態次元が大きい（例: `v+spike+u+r+eff` は `5*n`）ときは、
投影しても性能が維持できるかを確認します。

見方（超ざっくり）:
- `state_mode` を足すほど次元（=計算コスト）は増えるが、MCは伸びやすい
- `proj_out_dim` を小さくするほど軽くなるが、MCは落ちやすい
- まず「投影なし（0）」で state_mode を決め、次に投影でどこまで落ちるかを見る

- 例（投影なし=0 と out_dim=200 を比較）:
  - `python -m sorch.bench.mc_project --preset convo_spiking --n 120 --steps 3000 --washout 300 --max-delay 120 --seed 0 \
      --proj-dims 0,200 \
      --state-modes 'v+spike+u+r+eff' \
    --out outputs/phase2/mc/runs/phase2_mc_state_mode_project.csv`

### どれを採用する？（判断の目安）
まずは **「投影あり（軽い）前提」**か **「投影なし（強い）前提」**かを決めるのがおすすめです。

- 投影なし（`proj_out_dim=0`）で最強を狙う → `v+spike+u+r+eff` が強い傾向
- `proj_out_dim=200` で“現実に軽くする” → `v+spike` が強い傾向（state_dimが小さく、圧縮率が低い）

参考（repeats5レポート）:
- state_mode単体比較: [outputs/phase2/mc/reports/phase2_mc_state_modes_smoke_report.md](../../outputs/phase2/mc/reports/phase2_mc_state_modes_smoke_report.md)
- state_mode×投影（0/100/200）比較: [outputs/phase2/mc/reports/phase2_mc_state_modes_project_repeats5_report.md](../../outputs/phase2/mc/reports/phase2_mc_state_modes_project_repeats5_report.md)

注意:
- `proj_out_dim=100` は今回の条件では大きく劣化しやすい
- `--state-zscore` は実験用（今回の条件ではfull側のばらつきが大きく、採用は見送り）

### レポート化
- `python -m sorch.bench.mc_report --csv outputs/phase2/mc/runs/phase2_mc_state_modes_smoke.csv \
    --out outputs/phase2/mc/reports/phase2_mc_state_modes_smoke_report.md`

## 注意（よくあるつまずき）
- `spike_rate=0`（発火しない）と、STPパラメータ差がMCに反映されにくい
  - 用語: spike_rate（[用語集](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md#glossary-spike-rate)）
  - まずは `--preset convo_spiking` を試す（`v_threshold=0.25`, `dc_bias=0.25` を自動セット）
  - それでも0なら: `--v-threshold 0.25` 付近まで下げる / `--dc-bias 0.25` 付近まで上げる / `--input-scale` を上げる
  - どうしてもスパイクが死にやすいとき: `--recurrence-source v`（スパイクではなく膜電位で再帰させる）

---

## Step 2.4（オンライン学習）: RLSでreadoutを更新する

ここでは、Readout重みを **RLS**（[用語集](../project/%E7%94%A8%E8%AA%9E%E9%9B%86.md#glossary-rls)）でオンライン更新し、
「少数の遅延（delays）について過去入力の復元が維持できるか」をスモークで確認します。

前提:
- Step 2.3 の暫定推奨（軽さ優先）: `state_mode=v+spike` + `proj_out_dim=200`

### スモーク（seed=0）
- `python -m sorch.bench.rls_online --preset convo_spiking --n 120 --steps 3000 --washout 300 --seed 0 \
    --state-mode v+spike --proj-out-dim 200 \
    --delays 1,5,10,20,40,80,120 \
  --out outputs/phase2/rls/runs/phase2_rls_online_smoke.csv`

### repeats（seed=0..4 の例）
- `for s in 0 1 2 3 4; do python -m sorch.bench.rls_online --preset convo_spiking --n 120 --steps 3000 --washout 300 --seed $s \
    --state-mode v+spike --proj-out-dim 200 \
    --delays 1,5,10,20,40,80,120 \
  --out outputs/phase2/rls/runs/phase2_rls_online_repeats5.csv; done`

補足:
- `rls_online` のデフォルト（推奨）: `--update-mode block --update-every 10 --lam 1.0 --delta 0.01`
  - `update_every>1` のときは `--update-mode block` を使う（サンプルを捨てずにまとめて更新する）
  - `--update-mode sample` は比較用（N間引き更新）

### 更新間隔（10–50ms）を試す
- `for u in 10 20 50; do python -m sorch.bench.rls_online --preset convo_spiking --n 120 --steps 3000 --washout 300 --seed 0 \
    --state-mode v+spike --proj-out-dim 200 \
    --update-mode block --update-every $u --lam 1.0 --delta 0.01 \
  --out outputs/phase2/rls/runs/phase2_rls_online_block_sweep_smoke.csv; done`

### レポート化
- `python -m sorch.bench.rls_report --csv outputs/phase2/rls/runs/phase2_rls_online_repeats5.csv \
    --out outputs/phase2/rls/reports/phase2_rls_online_repeats5_report.md`

### 出力の見方（ざっくり）
- `mc_online`: delay別のR^2を足し合わせた指標（負のR^2は0扱い）
- `r2_by_delay_json`: delayごとのR^2（JSON）

出力場所:
- 生データ（CSV）: [outputs/phase2/rls/runs/](../../outputs/phase2/rls/runs/)

