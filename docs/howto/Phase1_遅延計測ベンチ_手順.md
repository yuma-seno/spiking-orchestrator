# Phase 1: Stop遅延 計測ベンチ 手順

この手順は [docs/spec/仕様書.md](../spec/%E4%BB%95%E6%A7%98%E6%9B%B8.md) の Phase 1 / Step 1.0 に対応します。

目的:
- stop（停止）までの遅延を **Median / p95 / p99** で記録できる状態にする
- 以後の変更で遅延が退行していないことを回帰で担保できるようにする

## 前提（安全）
- 実機I/Oを扱う場合は、まず無音出力やドライランで確認する
- Hard Stop は学習に依存させず、最優先で成立させる（別経路）

## 実行（例）

- デバイス一覧
  - `python -m sorch.bench.latency_bench --list-devices`

- ドライラン（オーディオ無し）
  - `python -m sorch.bench.latency_bench --dry-run --seconds 5`

- 実機（安全に無音出力）
  - `python -m sorch.bench.latency_bench --seconds 20 --sample-rate 48000 --frames 128 --output-mode silence`

- 実機（より物理寄りの停止：close/reopenを含めて記録）
  - `python -m sorch.bench.latency_bench --seconds 20 --sample-rate 48000 --frames 128 --output-mode silence --stop-mode close_reopen`

## 出力
- JSONL（stopイベント）: `outputs/phase1/latency/`
- 集計レポート: `outputs/phase1/report/`

補足:
- JSONL と同名の `*.meta.json` に、計測条件（frames/sample_rate等）を保存します。

## 判定
- 平均ではなく **p95/p99** を重視する（たまに遅いケースが体験を壊すため）
