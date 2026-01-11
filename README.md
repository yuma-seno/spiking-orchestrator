# Spiking Orchestrator (SORCH)

SORCHは、音声対話で重要な **割り込み停止（Stop）** を素早く・安全に行い、
その上で **短期記憶（STP）** を使って会話の「間」「テンポ」に適応していくための実験用リポジトリです。

## まず読む（入口）
- ガイド（この1本だけ追えばOK）: [docs/ガイド.md](docs/%E3%82%AC%E3%82%A4%E3%83%89.md)
- 仕様書（ゴール/KPI/ロードマップ）: [docs/spec/仕様書.md](docs/spec/%E4%BB%95%E6%A7%98%E6%9B%B8.md)

## すぐ動かす（最小）
- テスト: `pytest -q`
- 遅延計測（ドライラン）: `python -m sorch.bench.latency_bench --dry-run --seconds 5`
- MC（スモーク）: `python -m sorch.bench.mc_sweep --n 50 --steps 2000 --washout 200 --max-delay 50 --tauF-ms 200 --tauD-ms 1000 --w-scale 1.0 --out outputs/phase2/mc/runs/phase2_mc_smoke.csv`

## 手順書（必要なものだけ）
- Phase 1 遅延計測ベンチ: [docs/howto/Phase1_遅延計測ベンチ_手順.md](docs/howto/Phase1_%E9%81%85%E5%BB%B6%E8%A8%88%E6%B8%AC%E3%83%99%E3%83%B3%E3%83%81_%E6%89%8B%E9%A0%86.md)
- Phase 2 MC探索: [docs/howto/Phase2_MC探索_手順.md](docs/howto/Phase2_MC%E6%8E%A2%E7%B4%A2_%E6%89%8B%E9%A0%86.md)

## 成果物（どこに出る？）
生成物（ログ/CSV/レポート）は原則として outputs/ 配下に保存します。

- 遅延計測ログ（JSONL）: [outputs/phase1/latency/](outputs/phase1/latency/)
- 遅延計測レポート: [outputs/phase1/report/Phase1_測定結果.md](outputs/phase1/report/Phase1_%E6%B8%AC%E5%AE%9A%E7%B5%90%E6%9E%9C.md)
- MC生データ（CSV）: [outputs/phase2/mc/runs/](outputs/phase2/mc/runs/)
- MCレポート（Markdown）: [outputs/phase2/mc/reports/](outputs/phase2/mc/reports/)

## 作業管理
- 作業一覧（チェックリスト）: [docs/project/作業一覧_チェックリスト.md](docs/project/%E4%BD%9C%E6%A5%AD%E4%B8%80%E8%A6%A7_%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%88.md)
