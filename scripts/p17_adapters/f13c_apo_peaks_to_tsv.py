#!/usr/bin/env python3
"""CLI entrypoint for `F13c` unassigned apo 2D HSQC peaks -> 4-column TSV adapter (`p17_empirical_preregistration.json`).

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

from scripts.p17_adapters.w3_adapters import reshape_picked_apo_peaks_to_4col_tsv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F13c nmrassign pick --hsqc output -> 4-col apo_peaks.tsv")
    ap.add_argument("--in", dest="src", required=True, help="Input raw_apo_peaks.tsv from nmrassign pick --hsqc")
    ap.add_argument("--out", required=True, help="Output 4-column apo_peaks.tsv (seq_id, res_name, h_ppm, n_ppm)")
    args = ap.parse_args(argv)

    src_p = Path(args.src)
    tsv_4col = reshape_picked_apo_peaks_to_4col_tsv(
        src_p.read_text(encoding="utf-8"),
        source_label=str(src_p),
    )
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(tsv_4col, encoding="utf-8")
    n_peaks = max(0, len(tsv_4col.strip().splitlines()) - 1)
    print(json.dumps({"source": str(src_p), "out_tsv": str(out_p), "n_apo_peaks": n_peaks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
