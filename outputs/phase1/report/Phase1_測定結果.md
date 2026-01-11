# Phase 1 実測結果サマリー（非エンジニア向け解説つき）

このレポートは、仕様書 Phase 1 Step 1.0「止める（Stop）までの遅延」を**まず数字で把握する**ためのものです。

## まず結論（今回の結果）
- stop処理（ソフト側）が非常に速く、今回の条件では **p99でも1ms未満**でした。
- これは「止める判断をして停止命令を出す部分」がボトルネックではないことを示します。

## ここでいう“遅延”は何？（重要）
このベンチが測っているのは、ざっくり言うと
1) マイクから1フレーム分の音を読み込めた直後 →
2) 特徴量計算と判定 →
3) 出力ストリームを stop する（または stop 相当の処理）

までの **ソフトウェア処理時間** です。

注意:
- 「音が鳴ってから耳に入って止まるまで」の体感は、
  - OS/ドライバのバッファリング
  - フレームサイズ（`--frames`）
  - サンプルレート
 などにも強く影響されます。
- つまり、この数値が小さいのは良いニュースですが、**これだけでエンドツーエンドが30–50ms達成と言い切るものではありません**。
  次の段階で、バッファやTTSを含めた測定（仕様書 Step 1.2）につなげます。

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
- Python純粋実装のまま、**sub-ms（1ms未満）**の停止処理遅延を達成した。
- ALSA hw直接アクセス（dmix/PulseAudio経由なし）が功を奏している可能性がある。
- 仕様書の「50ms未達時は C++/Rust移行」の条件には当たらない（少なくともソフト側は十分速い）。

## この結果をどう使う？（次の一手）
- `--frames` を変えて比較（例: 64 / 256）し、安定性（XRUNや取りこぼし）とのトレードオフを見る
- Phase 1 Step 1.2 に向けて、TTS再生キューのクリア等の“本当の停止”を入れ、同様に実測する

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
  --output outputs/phase1/latency/latency_events_in4_out0_alpha015.jsonl
```

## 生成されたログはどこ？
- JSONL（stopイベント単位）: [outputs/phase1/latency/](../latency/)
