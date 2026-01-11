from __future__ import annotations

import argparse
from pathlib import Path

from sorch.bench.mc_report import generate_report_markdown


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate markdown MC reports for all CSVs under outputs/phase2/mc/runs"
    )
    ap.add_argument("--runs-dir", type=str, default="outputs/phase2/mc/runs")
    ap.add_argument("--reports-dir", type=str, default="outputs/phase2/mc/reports")
    ap.add_argument("--glob", type=str, default="*.csv")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(runs_dir.glob(args.glob))
    if not csv_paths:
        raise SystemExit(f"no csv found under: {runs_dir}")

    wrote = 0
    skipped = 0
    for csv_path in csv_paths:
        out_path = reports_dir / f"{csv_path.stem}_report.md"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        if args.dry_run:
            print("Would write:", out_path)
            continue

        md = generate_report_markdown(csv_path, topk=int(args.topk))
        out_path.write_text(md, encoding="utf-8")
        print("Wrote:", out_path)
        wrote += 1

    print(f"Done. wrote={wrote} skipped={skipped} total={len(csv_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
