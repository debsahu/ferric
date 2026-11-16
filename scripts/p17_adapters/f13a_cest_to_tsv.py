#!/usr/bin/env python3
"""CLI entrypoint for `F13a` CEST saturation profile -> 11-column TSV adapter (`p17_empirical_preregistration.json`).

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

from scripts.p17_adapters.w3_adapters import convert_f13a_cest_to_tsv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F13a plain-text/zip CEST saturation profile -> 11-col TSV")
    ap.add_argument("--in", dest="src", required=True, help="Input CEST.zip, profile directory, or table")
    ap.add_argument("--out", required=True, help="Output 11-column TSV path for nmrassign cest")
    ap.add_argument("--larmor-mhz", type=float, default=None, help="Optional heteronuclear Larmor MHz override")
    args = ap.parse_args(argv)

    summary = convert_f13a_cest_to_tsv(args.src, args.out, larmor_mhz_override=args.larmor_mhz)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
