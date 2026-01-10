# Spiking Orchestrator (SORCH): SNN駆動型・自律対話エージェント

## 概要

Spiking Orchestrator (SORCH) は、スパイキングニューラルネットワーク（SNN）を基盤とした自律対話エージェントシステムです。単に言葉を生成するチャットボットではなく、人間のような**「会話の間（ま）」「テンポ」「割り込みへの反応」**を備えた、生物的な実在感のあるAIエージェントを構築します。

## 詳細仕様

本システムの詳細な実装仕様、アーキテクチャ、および使用方法については、プロジェクトの仕様書をご確認ください。

## Phase 1（遅延計測ベンチ）

- 手順: [docs/Phase1_ベンチマーク手順.md](docs/Phase1_%E3%83%99%E3%83%B3%E3%83%81%E3%83%9E%E3%83%BC%E3%82%AF%E6%89%8B%E9%A0%86.md)
- チェックリスト: [docs/チェックリスト_Phase1.md](docs/%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%88_Phase1.md)

### 実行例

- `python -m sorch.bench.latency_bench --dry-run --seconds 5`
- `python -m sorch.bench.latency_bench --seconds 20 --sample-rate 48000 --frames 128`

## 作業一覧

- チェックリスト: [docs/作業一覧_チェックリスト.md](docs/%E4%BD%9C%E6%A5%AD%E4%B8%80%E8%A6%A7_%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%88.md)

## Phase 2（記憶回路/STP）

- 設計メモ: [docs/Phase2_設計メモ.md](docs/Phase2_%E8%A8%AD%E8%A8%88%E3%83%A1%E3%83%A2.md)
