#!/usr/bin/env python3
"""CLI entrypoint for `F10` NMR-STAR relaxation loop extraction (`p17_empirical_preregistration.json`).

Provenance:
- Original clean-room Python 3 stdlib CLI wrapper delegating to `scripts/p17_adapters/w3_adapters.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.p17_adapters.w3_adapters import evaluate_f10_modelfree, export_f10_modelfree_tsvs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F10 NMR-STAR -> nmrassign modelfree TSV adapter & evaluator")
    ap.add_argument("--star", required=True, help="Input NMR-STAR v3 file (e.g. bmr6470_3.str)")
    ap.add_argument("--out-dir", help="Output directory for modelfree relaxation TSVs")
    ap.add_argument("--prefix", default="ubq", help="Output filename prefix (default: ubq)")
    ap.add_argument("--accession", default="BMRB:6470", help="Dataset accession ID")
    ap.add_argument("--only-600-mhz", action="store_true", help="Emit only disambiguated 600 MHz saveframes")
    ap.add_argument("--eval-modelfree-tsv", help="Post-fit nmrassign modelfree --out TSV to evaluate")
    ap.add_argument("--eval-stdout-file", help="Optional captured stdout from nmrassign modelfree")
    ap.add_argument("--exit-code", type=int, default=0, help="CLI exit code of nmrassign modelfree")
    args = ap.parse_args(argv)

    if args.eval_modelfree_tsv:
        mf_tsv = Path(args.eval_modelfree_tsv).read_text(encoding="utf-8")
        star_txt = Path(args.star).read_text(encoding="utf-8", errors="replace")
        stdout_txt = Path(args.eval_stdout_file).read_text(encoding="utf-8") if args.eval_stdout_file else ""
        res = evaluate_f10_modelfree(
            mf_tsv,
            star_txt,
            stdout_text=stdout_txt,
            exit_code=args.exit_code,
            accession_id=args.accession,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0 if res["verdict"] == "PASS" else 1

    if not args.out_dir:
        ap.error("--out-dir is required when extracting relaxation TSVs")
    summary = export_f10_modelfree_tsvs(
        args.star,
        args.out_dir,
        prefix=args.prefix,
        accession_id=args.accession,
        only_600_mhz=args.only_600_mhz,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
