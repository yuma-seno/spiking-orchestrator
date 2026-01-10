# Phase 2: 記憶回路と文脈制御（設計メモ）

## 目的（仕様書 Phase 2）
- 会話の「間」と「勢い」を短期記憶として表現する
- STP（短期可塑性）を核にしたリザーバを構築し、readoutでアクションを決める

## Step 2.1（実装済み）: STPニューロン
- 実装: [sorch/core/stp_lif_node.py](sorch/core/stp_lif_node.py)
- Tsodyks–Markramの u（促通）/ r（資源）を内部状態として保持
- 入力電流に `efficacy = u*r` を掛けてからLIFへ投入

## Step 2.2: Memory Capacity（MC）探索
- 目的: リザーバが過去入力をどれだけ保持できるかを測る
- 探索パラメータ（仕様書）
  - $\tau_F$ = 100–800ms
  - $\tau_D$ = 500–3000ms
  - $W_{scale}$ = 0.8–1.1
- 出力: MCスコア vs パラメータの表（JSON/CSV）

## Step 2.3: 次元削減
- PCAは避け、Random Projectionで低次元へ圧縮

## Step 2.4: RLS（オンライン学習）
- 忘却係数 $\lambda$ = 0.99–0.999
- 正則化（Tikhonov）を入れて数値安定化
- 更新間隔: 10–50ms を目安

## テスト方針
- 数学・状態更新はpytestでユニットテスト
- MC探索はスクリプト化し、短時間（数十秒）で最低限回るスモークを用意
