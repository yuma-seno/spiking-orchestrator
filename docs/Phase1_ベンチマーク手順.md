# Phase 1: 遅延計測ベンチマーク手順

## 目的
仕様書 Step 1.0 の「マイク入力 → 閾値判定 → スピーカー停止命令」を最小構成で実測し、
Median / p95 / p99 の遅延とジッタを把握する。

※ Phase 1 のReflex（割り込み検知）はDSP-firstの暫定実装であり、後続フェーズでSNN版Reflex Circuitへ置き換える前提。

## 先に確認すること
- コンテナがホストの音声デバイスにアクセスできること（/dev/snd がマウントされている）
- 可能ならホスト側で低遅延設定（JACK/ALSAのバッファ、PulseAudio/pipewireのレイテンシ）を確認

### よくある落とし穴（コンテナ）
- `/dev/snd/*` が `root:audio` で `crw-rw----` になっている場合、ユーザーが `audio` グループに入っていないと入出力が開けません。
  - 一時的に回避: `sudo -n -u vscode -g audio python -m sorch.bench.latency_bench ...`
  - 恒久対応: Dev Container をRebuild（Dockerfileで `audio` グループ追加済み）
- ALSA/JACK の警告ログが大量に出ることがありますが、デバイスが開けていれば致命ではありません。

## 実行
- デバイス一覧
  - `python -m sorch.bench.latency_bench --list-devices`

- ドライラン（オーディオ無し）
  - `python -m sorch.bench.latency_bench --dry-run --seconds 5`

- 実機（PyAudio / /dev/snd が必要）
  - まずは無音出力（ビープ対策）: `python -m sorch.bench.latency_bench --seconds 20 --sample-rate 48000 --frames 128 --output-mode silence`
  - 入出力デバイスを固定: `python -m sorch.bench.latency_bench --input-device <idx> --output-device <idx> ...`
  - 入出力でサンプルレートが違う場合: `--input-sample-rate 48000 --output-sample-rate 44100` のように分けて指定

### 重要パラメータ
- `--frames`: 64〜256 を中心に探索（小さいほど低遅延だがXRUN/overrunしやすい）
- `--alpha`: 適応閾値のマージン（環境ノイズ次第で調整）
- `--holdoff-ms`: チャタリング防止（200〜500ms目安）

## 出力
- JSONL: `outputs/latency_events.jsonl`
  - stopイベントごとに `t_read_done_ns → t_stop_done_ns` の遅延を保存

## 判定
- 目標: Median < 50ms, p99 < 100ms
- 未達の場合は
  - フレームサイズ/サンプルレートの再検討
  - 入出力API（ALSA/JACK/Exclusive相当）の再検討
  - PythonホットパスのCython/Numba化 or C++/Rust移行
