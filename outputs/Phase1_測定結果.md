# Phase 1 実測結果サマリー

## 環境
- コンテナ: python:3.12-slim (Debian trixie)
- 音声API: PyAudio + ALSA (hw直接)
- 入力: device 4 (Anker PowerConf C200: 48kHz stereo → monoダウンミックス)
- 出力: device 0 (HD-Audio Generic HDMI: 44.1kHz stereo, 無音出力)
- 実装: Python 3.12 + numpy/scipy (Cython/Numba最適化なし)

## 計測結果 (2026-01-10, alpha=0.015)

### frames=128 (約2.67ms @ 48kHz)
- **Stop events**: 18回（20秒間）
- **Median**: 0.24 ms
- **p95**: 0.38 ms
- **p99**: 0.39 ms
- **Mean ± Std**: 0.26 ± 0.07 ms

**判定**: ✅ 目標（Median < 50ms, p99 < 100ms）を大幅クリア

## 所見
- Python純粋実装のまま、**sub-ms（1ms未満）**の停止遅延を達成した。
- ALSA hw直接アクセス（dmix/PulseAudio経由なし）が功を奏している可能性。
- 仕様書の「50ms未達時は C++/Rust移行」の条件には当たらない。
- 次はフレームサイズ（64/256）比較と、誤検知（FP）の実地チューニングへ進む。

## 再現コマンド
```bash
sudo -u vscode -g audio python -m sorch.bench.latency_bench \
  --seconds 20 \
  --frames 128 \
  --input-device 4 \
  --output-device 0 \
  --input-sample-rate 48000 \
  --output-sample-rate 44100 \
  --input-channels 2 \
  --output-channels 2 \
  --output-mode silence \
  --alpha 0.015 \
  --output outputs/latency_events_in4_out0_alpha015.jsonl
```
