# Phase 1: 遅延計測ベンチ（Stop Latency）手順

この手順は、仕様書の Phase 1 / Step 1.0 に相当する「止めるまでの遅延」を **まず測れる状態にする**ためのものです。

## 目的
- マイク入力 → 判定 → 停止処理（ソフトウェア側）までの遅延を、Median / p95 / p99 で記録する
- 生成物（ログ）を `outputs/phase1/latency/` に保存し、条件と一緒に残す

## 先に確認すること（コンテナ環境）
- `/dev/snd` がコンテナに渡っている
- 音声デバイス権限の都合で、`audio` グループが必要な場合がある
  - 一時回避例: `sudo -n -u vscode -g audio python -m sorch.bench.latency_bench ...`

## 実行
- デバイス一覧
  - `python -m sorch.bench.latency_bench --list-devices`

- ドライラン（オーディオ無し）
  - `python -m sorch.bench.latency_bench --dry-run --seconds 5`

- 実機（安全に無音出力）
  - `python -m sorch.bench.latency_bench --seconds 20 --sample-rate 48000 --frames 128 --output-mode silence`
  - 入出力デバイスを固定する場合:
    - `python -m sorch.bench.latency_bench --seconds 20 --frames 128 --input-device <idx> --output-device <idx> --output-mode silence --output outputs/phase1/latency/latency_events.jsonl`

## 重要パラメータ（まず触る）
- `--frames`: 64〜256 を中心に探索（小さいほど低遅延だが不安定になりやすい）
- `--alpha`: 適応閾値のマージン（環境ノイズ次第で調整）
- `--holdoff-ms`: チャタリング防止（200〜500ms目安）

## 出力（どこに何が出る？）
- JSONL（stopイベント単位）: [outputs/phase1/latency/](../../outputs/phase1/latency/)
- 実測結果まとめ（レポート）: [outputs/phase1/report/Phase1_測定結果.md](../../outputs/phase1/report/Phase1_%E6%B8%AC%E5%AE%9A%E7%B5%90%E6%9E%9C.md)

## 判定（読み方）
- **Median**: 典型的な遅延
- **p95/p99**: たまに遅いケース（体験劣化に直結するため重視）
- 目安（仕様書）: Median < 50ms, p99 < 100ms

## 注意
この測定は「ソフトウェア処理としての停止の速さ」です。
体感の停止は、音声バッファやTTSなど別要因も効くため、次はStep 1.2（本当の停止）へ進みます。
