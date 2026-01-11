# Phase 2: Memory Capacity (MC) レポート

このレポートは、Phase 2 Step 2.2（MC最大化探索）の結果CSVを集計して、
**良さそうなパラメータ候補（上位）**を見つけるための一覧です。

- 入力CSV: outputs/phase2/mc/runs/phase2_mc_spike_rate_smoke.csv
- 総試行数: 1
- ユニーク条件数: 1

## これは何の数値？（MCの超ざっくり説明）

- **MC（Memory Capacity）** は『少し前の入力を、今のリザーバ状態からどれだけ復元できるか』のスコアです。
- 数値は **大きいほど短期記憶が強い** と解釈します（ただし比較は同じ設定同士で行います）。
- 例え話/図解で掴みたい場合: docs/ガイド.md の「MC」節

## 今回の実験条件（共通設定）

- ニューロン数 n: 80
- ステップ数 steps: 2000（長いほど時間がかかる）
- washout: 200（最初の過渡を捨てる）
- max_delay: 80（何ステップ前まで復元するか）
- ridge（回帰の正則化）: 0.001
- sparsity（結合の疎さ）: 0.1

## 発火率のチェック（任意だが重要）

- spike_rate は『1ステップ×1ニューロンあたりの平均発火率』です。0に近いと“動いていない”可能性があります。
- 全体の平均 spike_rate: 0.000000

## Top candidates (mean MC)

表の見方:
- **mean MC**: 同じ条件を複数回（seed/repeats）走らせた平均
- **std**: ばらつき（小さいほど安定）
- **n_runs**: 平均を取った回数
- **mean spike_rate**: 同じ条件の平均発火率（0に近い場合は要注意）

| rank | mean MC | std | n_runs | mean spike_rate | tauF_ms | tauD_ms | w_scale | U | rec | v_th | in_scale | in_bias | dc_bias |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|---:|---:|---:|---:|
| 1 | 1.2146 | 0.0000 | 1 | 0.000000 | 100 | 500 | 1.100 | 0.200 | spike | 0.500 | 1.500 | 0.000 | 0.000 |

## Best

- mean MC=1.2146 (std=0.0000, n_runs=1)
- mean spike_rate=0.000000
- tauF=100ms, tauD=500ms, w_scale=1.100, U=0.200
- rec=spike, v_threshold=0.500
- input_scale=1.500, input_bias=0.000, dc_bias=0.000

## パラメータの意味（超ざっくり）

- tauF_ms / tauD_ms: STP（短期可塑性）の時定数。値で記憶の残り方が変わる
- w_scale: リカレント結合の強さ（大きいほど影響が強いが、暴れやすいこともある）
- U: STPの効きやすさ（大きいほど入力が通りやすい）
- rec: 再帰に使う信号（spike or v）。spikeはSNNっぽいが死にやすいことがある
- v_th: 発火しやすさ（小さいほど発火しやすい）
- input_scale/input_bias: 入力の強さ/平均シフト
- dc_bias: 一定のバイアス電流（発火を立ち上げるために使う）

## 注意（比較のしかた）

- MCは“絶対値”というより、**同じ共通設定の中での相対比較**が安全です。
- stdが大きい候補は、seed次第で性能が落ちる可能性があるため、追加のrepeatsで確認してください。
- mean spike_rate が0に近い候補は“動いていない”可能性があるため、まず発火率を上げてから比較してください。
