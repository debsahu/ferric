#!/usr/bin/env python3
"""CLI entrypoint for `F16` pseudo-2D Bruker/Varian DOSY plane integrator (`p17_empirical_preregistration.json`).

Provenance:
- Original clean-room Python 3 stdlib CLI wrapper delegating to `scripts/p17_adapters/w4_adapters.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.p17_adapters.w4_adapters import integrate_pseudo2d_dosy_planes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F16 locked-g0 F2 +-2*FWHM box DOSY plane integrator -> decay.tsv")
    ap.add_argument("--exp-dir", required=True, help="Bruker (ser+difflist+acqus) or Varian (fid+procpar) DOSY directory")
    ap.add_argument("--out", required=True, help="Output Case-C decay.tsv path for nmrassign dosy")
    ap.add_argument("--nmrassign-bin", default="./target/release/nmrassign", help="Compiled nmrassign binary path")
    args = ap.parse_args(argv)

    summary = integrate_pseudo2d_dosy_planes(
        args.exp_dir,
        args.out,
        nmrassign_bin=args.nmrassign_bin,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
