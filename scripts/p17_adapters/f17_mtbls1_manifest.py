#!/usr/bin/env python3
"""CLI entrypoint for `F17` MetaboLights MTBLS1 ISA-Tab sample/assay manifest builder (`p17_empirical_preregistration.json`).

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

from scripts.p17_adapters.w4_adapters import build_f17_mtbls1_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F17 MTBLS1 ISA-Tab 1-to-1 .zip cohort manifest builder")
    ap.add_argument("--samples-sheet", required=True, help="Path to s_MTBLS1.txt")
    ap.add_argument("--assay-sheet", required=True, help="Path to a_MTBLS1_metabolite_profiling_NMR_spectroscopy.txt")
    ap.add_argument("--spectra-dir", default="f17_spectra", help="Directory holding processed 1D sample spectra")
    ap.add_argument("--ext", default=".tsv", help="Processed spectrum filename extension (default: .tsv)")
    ap.add_argument("--out", required=True, help="Output 3-column mtbls1_manifest.tsv path")
    args = ap.parse_args(argv)

    s_txt = Path(args.samples_sheet).read_text(encoding="utf-8", errors="replace")
    a_txt = Path(args.assay_sheet).read_text(encoding="utf-8", errors="replace")
    res = build_f17_mtbls1_manifest(
        s_txt,
        a_txt,
        spectra_dir=args.spectra_dir,
        spectrum_ext=args.ext,
        out_manifest_path=args.out,
    )
    print(
        json.dumps(
            {
                "n_joined_samples": res["n_joined_samples"],
                "group_counts": res["group_counts"],
                "ignored_nmrml_rows": res["ignored_nmrml_rows"],
                "out_manifest": args.out,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
