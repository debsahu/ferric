#!/usr/bin/env python3
"""CLI entrypoint for `F13b` CPMG R2,eff & raw intensity plane -> TSV adapter (`p17_empirical_preregistration.json`).

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

from scripts.p17_adapters.w3_adapters import convert_f13b_cpmg_to_tsv, convert_f13b_raw_planes_to_tsv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F13b CPMG dispersion / raw intensity planes -> TSV")
    ap.add_argument("--in", dest="src", required=True, help="Input CPMG.zip, directory, or raw intensity file")
    ap.add_argument("--out", required=True, help="Output TSV path for nmrassign cpmg")
    ap.add_argument("--raw-planes", action="store_true", help="Reshape raw CPMG intensity plane table")
    ap.add_argument("--larmor-mhz", type=float, default=None, help="Heteronuclear Larmor MHz (required with --raw-planes)")
    ap.add_argument("--t-relax-s", type=float, default=None, help="Constant-time T_relax in s (required with --raw-planes)")
    ap.add_argument("--ncyc-columns", action="store_true", help="Interpret raw-plane header numbers as n_cyc = nu_cpmg * T_relax")
    args = ap.parse_args(argv)

    if args.raw_planes:
        if args.larmor_mhz is None or args.t_relax_s is None:
            ap.error("--raw-planes requires both --larmor-mhz and --t-relax-s")
        raw_txt = Path(args.src).read_text(encoding="utf-8", errors="replace")
        tsv = convert_f13b_raw_planes_to_tsv(
            raw_txt,
            larmor_mhz=args.larmor_mhz,
            t_relax_s=args.t_relax_s,
            ncyc_values_ARE_cycles=args.ncyc_columns,
        )
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(tsv, encoding="utf-8")
        print(json.dumps({"source": args.src, "out_tsv": str(out_p), "mode": "raw_planes"}, indent=2))
        return 0

    summary = convert_f13b_cpmg_to_tsv(args.src, args.out, larmor_mhz_override=args.larmor_mhz)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
