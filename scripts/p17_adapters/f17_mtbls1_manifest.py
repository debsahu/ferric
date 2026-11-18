#!/usr/bin/env python3
"""CLI entrypoint & ISA-Tab manifest builder for `F17`/`F29` MetaboLights `MTBLS1` cohort.

Provenance:
- Clean-room Python 3 stdlib ISA-Tab sample (`s_MTBLS1.txt`) + assay
  (`a_MTBLS1_metabolite_profiling_NMR_spectroscopy.txt`) 1-to-1 `.zip` joiner supporting both
  `"Free Induction Decay Data File"` and `"Raw Spectral Data File"` ISA-Tab columns (`F29`),
  ignoring `.nmrML` rows and raising on conflicting sample/archive mappings.
- Strictly offline and AST-firewall compliant (`scripts/p17_adapters/selftest.py`): zero network
  modules and zero raw float literals outside `{0.0, 1.0}`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Union


ISA_SPECTRAL_ZIP_COLUMNS: tuple[str, ...] = (
    "Free Induction Decay Data File",
    "Raw Spectral Data File",
)


def _read_isatab_tsv(text: str) -> List[Dict[str, str]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    reader = csv.DictReader(lines, delimiter="\t", quotechar='"')
    out: List[Dict[str, str]] = []
    for row in reader:
        cleaned = {
            (k.strip().strip('"') if k else ""): (v.strip().strip('"') if v else "")
            for k, v in row.items()
        }
        out.append(cleaned)
    return out


def build_mtbls1_isatab_manifest(
    s_mtbls1_text: str,
    a_mtbls1_text: str,
    *,
    spectra_dir: Union[Path, str] = "f17_spectra",
    spectrum_ext: str = ".tsv",
    out_manifest_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Deterministically join `s_MTBLS1.txt` and `a_MTBLS1_*.txt` (`.zip` ONLY, ignoring `.nmrML`).

    Accepts `.zip` archive entries from either `"Free Induction Decay Data File"` or
    `"Raw Spectral Data File"` (as well as any fallback `*Data File*` column ending in `.zip`),
    enforcing a strict 1-to-1 `Sample Name -> <archive>.zip` mapping.
    """
    s_rows = _read_isatab_tsv(s_mtbls1_text)
    a_rows = _read_isatab_tsv(a_mtbls1_text)

    sample_to_group: Dict[str, str] = {}
    for r in s_rows:
        sname = r.get("Sample Name", "").strip()
        fval = ""
        for k, v in r.items():
            kl = k.lower()
            if "factor value" in kl and "metabolic syndrome" in kl:
                fval = v.strip()
                break
        if sname and fval:
            if sname in sample_to_group and sample_to_group[sname] != fval:
                raise ValueError(
                    f"conflicting Factor Value[Metabolic syndrome] for sample {sname!r}"
                )
            sample_to_group[sname] = fval

    sample_to_zip: Dict[str, str] = {}
    ignored_nmrml = 0
    for r in a_rows:
        sname = r.get("Sample Name", "").strip()
        if not sname:
            continue

        candidates: List[str] = []
        for col_name in ISA_SPECTRAL_ZIP_COLUMNS:
            val = r.get(col_name, "").strip()
            if val:
                candidates.append(val)

        # Fallback scan for any other `*Data File*` column carrying a `.zip` or `.nmrML` entry
        if not candidates:
            for k, v in r.items():
                kl = k.lower()
                vs = v.strip()
                if vs and ("spectral data file" in kl or "induction decay data file" in kl):
                    candidates.append(vs)

        saw_zip_in_row = False
        saw_nmrml_in_row = False
        for raw_file in candidates:
            low = raw_file.lower()
            if low.endswith(".nmrml"):
                saw_nmrml_in_row = True
                continue
            if not low.endswith(".zip"):
                continue
            saw_zip_in_row = True
            zip_base = Path(raw_file).name
            if sname in sample_to_zip and sample_to_zip[sname] != zip_base:
                raise ValueError(
                    f"non-1-to-1 ISA-Tab .zip mapping for {sname!r}: "
                    f"{sample_to_zip[sname]} vs {zip_base}"
                )
            sample_to_zip[sname] = zip_base

        if saw_nmrml_in_row and not saw_zip_in_row:
            ignored_nmrml += 1

    joined_ids = sorted(sid for sid in sample_to_group if sid in sample_to_zip)
    if not joined_ids:
        raise ValueError(
            "1-to-1 join between s_MTBLS1.txt and a_MTBLS1 (.zip) produced 0 samples"
        )

    spec_root = Path(spectra_dir)
    manifest_lines = ["sample_id\tgroup\tspectrum_path"]
    records: List[Dict[str, str]] = []
    group_counts: Dict[str, int] = {}
    for sid in joined_ids:
        grp = sample_to_group[sid]
        group_counts[grp] = group_counts.get(grp, 0) + 1
        sp_path = (spec_root / f"{sid}{spectrum_ext}").as_posix()
        manifest_lines.append(f"{sid}\t{grp}\t{sp_path}")
        records.append(
            {
                "sample_id": sid,
                "group": grp,
                "raw_zip": sample_to_zip[sid],
                "spectrum_path": sp_path,
            }
        )

    manifest_tsv = "\n".join(manifest_lines) + "\n"
    if out_manifest_path is not None:
        op = Path(out_manifest_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(manifest_tsv, encoding="utf-8")

    return {
        "n_joined_samples": len(records),
        "group_counts": group_counts,
        "ignored_nmrml_rows": ignored_nmrml,
        "records": records,
        "manifest_tsv": manifest_tsv,
    }


def run_self_test() -> int:
    s_txt = (
        '"Source Name"\t"Sample Name"\t"Factor Value[Metabolic syndrome]"\n'
        '"src1"\t"ADG10003u_007"\t"Control"\n'
        '"src2"\t"ADG10003u_008"\t"Type 2 diabetes mellitus"\n'
        '"src3"\t"ADG10003u_009"\t"Control"\n'
        '"src4"\t"ADG10003u_010"\t"Type 2 diabetes mellitus"\n'
    )

    # Case 1: Real MTBLS1 layout where `.zip` lives in `"Free Induction Decay Data File"` and
    # `"Raw Spectral Data File"` is empty or holds `.nmrML` rows!
    a_fid_col = (
        '"Sample Name"\t"Free Induction Decay Data File"\t"Raw Spectral Data File"\n'
        '"ADG10003u_007"\t"FILES/ADG10003u_007.zip"\t"ADG10003u_007.nmrML"\n'
        '"ADG10003u_008"\t"FILES/ADG10003u_008.zip"\t""\n'
        '"ADG10003u_009"\t""\t"ADG10003u_009.nmrML"\n'
        '"ADG10003u_010"\t""\t"FILES/ADG10003u_010.zip"\n'
    )
    with tempfile.TemporaryDirectory(prefix="f29_mtbls1_selftest_") as td:
        out_p = Path(td) / "mtbls1_manifest.tsv"
        res = build_mtbls1_isatab_manifest(
            s_txt,
            a_fid_col,
            spectra_dir="f17_spectra",
            spectrum_ext=".tsv",
            out_manifest_path=out_p,
        )
        assert res["n_joined_samples"] == 3, f"expected 3 joined samples, got {res}"
        assert res["ignored_nmrml_rows"] == 1, f"expected 1 nmrML-only row ignored, got {res}"
        assert res["group_counts"] == {
            "Control": 1,
            "Type 2 diabetes mellitus": 2,
        }
        lines = out_p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4
        assert lines[0] == "sample_id\tgroup\tspectrum_path"
        assert "ADG10003u_007\tControl\tf17_spectra/ADG10003u_007.tsv" in lines
        assert (
            "ADG10003u_008\tType 2 diabetes mellitus\tf17_spectra/ADG10003u_008.tsv"
            in lines
        )
        assert (
            "ADG10003u_010\tType 2 diabetes mellitus\tf17_spectra/ADG10003u_010.tsv"
            in lines
        )

    # Case 2: Conflicting `.zip` archives across columns or duplicate rows must raise ValueError
    a_conflict = (
        '"Sample Name"\t"Free Induction Decay Data File"\t"Raw Spectral Data File"\n'
        '"ADG10003u_007"\t"ADG10003u_007_a.zip"\t"ADG10003u_007_b.zip"\n'
    )
    raised = False
    try:
        build_mtbls1_isatab_manifest(s_txt, a_conflict)
    except ValueError:
        raised = True
    assert raised, "expected ValueError on conflicting .zip filenames for one Sample Name"

    print(
        json.dumps(
            {
                "self_test": "f17_mtbls1_manifest",
                "accepted_columns": list(ISA_SPECTRAL_ZIP_COLUMNS),
                "passed": True,
            }
        )
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="F17/F29 MTBLS1 ISA-Tab 1-to-1 .zip cohort manifest builder"
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic unit self-test covering both ISA-Tab columns.",
    )
    ap.add_argument("--samples-sheet", help="Path to s_MTBLS1.txt")
    ap.add_argument(
        "--assay-sheet",
        help="Path to a_MTBLS1_metabolite_profiling_NMR_spectroscopy.txt",
    )
    ap.add_argument(
        "--spectra-dir",
        default="f17_spectra",
        help="Directory holding processed 1D sample spectra",
    )
    ap.add_argument(
        "--ext",
        default=".tsv",
        help="Processed spectrum filename extension (default: .tsv)",
    )
    ap.add_argument("--out", help="Output 3-column mtbls1_manifest.tsv path")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()
    if not args.samples_sheet or not args.assay_sheet or not args.out:
        ap.error("--samples-sheet, --assay-sheet, and --out are required unless --self-test is set")

    s_txt = Path(args.samples_sheet).read_text(encoding="utf-8", errors="replace")
    a_txt = Path(args.assay_sheet).read_text(encoding="utf-8", errors="replace")
    res = build_mtbls1_isatab_manifest(
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
