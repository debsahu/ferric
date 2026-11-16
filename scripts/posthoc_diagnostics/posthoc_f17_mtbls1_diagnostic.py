#!/usr/bin/env python3
"""Post-hoc, non-gate 1-line ISA-Tab column-name builder & MTBLS1 cohort summary for Gate F17.

Provenance & Scientific Disclosure:
- Original clean-room Python 3 standard-library diagnostic script (MIT License).
- Lives OUTSIDE `scripts/p17_adapters/` so pre-download `bbc770a` adapters remain 0-byte diff.
- Zero external dependencies and zero network calls.
- **Alters NO primary gate verdict (`F17 = FAIL` both ways)**:
  1. **Primary Strict `bbc770a` Blind Outcome (`FAIL`, `rc = 1`)**:
     Unmodified `bbc770a` `scripts/p17_adapters/f17_mtbls1_manifest.py` (`w4_adapters.py:1121-1126`)
     checked `r.get("Raw Spectral Data File", "")` and substring `"spectral data file"` ending in
     `".zip"`. In EBI MetaboLights NMR ISA-Tab assay sheet
     `a_MTBLS1_metabolite_profiling_NMR_spectroscopy.txt`, the 132 raw Bruker `.zip` archives are
     tabulated under `"Free Induction Decay Data File"` (while `"Derived Spectral Data File"` holds
     133 `.nmrML` files), causing `f17_mtbls1_manifest.py` to exit `rc = 1` (`0/132` joined ->
     strict primary `F17 = FAIL` per `P17-R1` / `P17-R2`).
  2. **Post-Hoc 1-Line Disclosed Column-Name Run (also `FAIL`, `3/4` criteria)**:
     Adding the 1-line ISA-Tab NMR column lookup `r.get("Free Induction Decay Data File", "")`
     joins all `132/132` samples 1-to-1 (`84` `Control Group`, `48` `diabetes mellitus`). Running
     the frozen `nmrassign` binary with the exact pre-registered CLI flags processes `132/132`
     spectra cleanly (`F17-C1 PASS`), reaches `perm_p = 0.009901 <= 0.02` (`F17-C3 PASS`), and
     recovers `10` Top-15 urinary D-glucose glycosuria buckets (`F17-C4 PASS`), plus Hippurate,
     Creatinine, TMAO, Citrate, and L-Alanine markers, while 1-ortho Pareto AIB OPLS-DA achieves
     `Q^2 = 0.122445 < 0.25` (`F17-C2 FAIL`).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


def read_isatab_tsv(text: str) -> List[Dict[str, str]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    reader = csv.DictReader(lines, delimiter="\t", quotechar='"')
    out: List[Dict[str, str]] = []
    for row in reader:
        out.append(
            {
                (k.strip().strip('"') if k else ""): (v.strip().strip('"') if v else "")
                for k, v in row.items()
            }
        )
    return out


def build_posthoc_mtbls1_manifest_1line_fix(
    s_mtbls1_text: str,
    a_mtbls1_text: str,
    *,
    spectra_dir: Path,
    spectrum_ext: str = ".tsv",
    out_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Identical to `w4_adapters.build_f17_mtbls1_manifest` plus 1-line `'Free Induction Decay Data File'` check."""
    s_rows = read_isatab_tsv(s_mtbls1_text)
    a_rows = read_isatab_tsv(a_mtbls1_text)

    sample_to_group: Dict[str, str] = {}
    for r in s_rows:
        sname = r.get("Sample Name", "").strip()
        fval = ""
        for k, v in r.items():
            if "factor value" in k.lower() and "metabolic syndrome" in k.lower():
                fval = v.strip()
                break
        if sname and fval:
            if sname in sample_to_group and sample_to_group[sname] != fval:
                raise ValueError(f"conflicting Factor Value[Metabolic syndrome] for sample {sname!r}")
            sample_to_group[sname] = fval

    sample_to_zip: Dict[str, str] = {}
    ignored_nmrml = 0
    for r in a_rows:
        sname = r.get("Sample Name", "").strip()
        # 1-LINE POST-HOC COLUMN FIX vs `w4_adapters.py:1121`: include NMR ISA-Tab `"Free Induction Decay Data File"`
        raw_file = (
            r.get("Raw Spectral Data File", "").strip()
            or r.get("Free Induction Decay Data File", "").strip()
        )
        if not raw_file:
            for k, v in r.items():
                if "spectral data file" in k.lower() and v.strip().lower().endswith(".zip"):
                    raw_file = v.strip()
                    break
        for k, v in r.items():
            if v.strip().lower().endswith(".nmrml"):
                ignored_nmrml += 1
        if not sname or not raw_file or not raw_file.lower().endswith(".zip"):
            continue
        zip_base = Path(raw_file).name
        if sname in sample_to_zip and sample_to_zip[sname] != zip_base:
            raise ValueError(f"non-1-to-1 .zip mapping for {sname!r}: {sample_to_zip[sname]} vs {zip_base}")
        sample_to_zip[sname] = zip_base

    joined_ids = sorted(sid for sid in sample_to_group if sid in sample_to_zip)
    if not joined_ids:
        raise ValueError("1-to-1 join produced 0 samples")

    manifest_lines = ["sample_id\tgroup\tspectrum_path"]
    group_counts: Dict[str, int] = {}
    records: List[Dict[str, str]] = []
    for sid in joined_ids:
        grp = sample_to_group[sid]
        group_counts[grp] = group_counts.get(grp, 0) + 1
        sp_path = (spectra_dir / f"{sid}{spectrum_ext}").as_posix()
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
        out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        out_manifest_path.write_text(manifest_tsv, encoding="utf-8")

    return {
        "n_joined_samples": len(records),
        "group_counts": group_counts,
        "ignored_nmrml_entries": ignored_nmrml,
        "records": records,
    }


def load_f17_manifest_windows_and_thresholds(prereg_json_path: Path) -> Dict[str, Any]:
    mf = json.loads(prereg_json_path.read_text(encoding="utf-8"))
    f17 = mf["gates"]["F17"]
    c4_src = f17["pass_criteria"][3]["reference_value_or_source"]
    m = re.search(
        r"Top-(\d+)\s+ranked OPLS-DA VIP buckets or\s*([0-9.]+)\s*ppm STOCSY.*?"
        r"\[([0-9.]+),\s*([0-9.]+)\]\s*ppm.*?\[([0-9.]+),\s*([0-9.]+)\]\s*ppm",
        c4_src,
    )
    if not m:
        raise ValueError("could not parse F17 glucose windows from preregistration manifest")
    return {
        "top_k": int(m.group(1)),
        "driver_ppm": float(m.group(2)),
        "glucose_windows_ppm": [
            (float(m.group(3)), float(m.group(4))),
            (float(m.group(5)), float(m.group(6))),
        ],
        "pass_criteria": f17["pass_criteria"],
    }


def bucket_overlaps(center: float, lo: Optional[float], hi: Optional[float], wins: Sequence[Tuple[float, float]]) -> bool:
    b_lo = float(lo) if lo is not None else center
    b_hi = float(hi) if hi is not None else center
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo
    return any((b_hi >= w0 and b_lo <= w1) or (w0 <= center <= w1) for w0, w1 in wins)


def summarize_biomarkers_and_salek_hits(
    biomarkers_csv_path: Path,
    oplsda_json_path: Path,
    stocsy_json_path: Path,
    f17_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    opls = json.loads(oplsda_json_path.read_text(encoding="utf-8"))
    stocsy = json.loads(stocsy_json_path.read_text(encoding="utf-8"))
    top_k = int(f17_cfg["top_k"])
    wins = f17_cfg["glucose_windows_ppm"]

    rows: List[Dict[str, Any]] = []
    if biomarkers_csv_path.is_file():
        reader = csv.DictReader(
            [
                ln
                for ln in biomarkers_csv_path.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
        )
        for rank_idx, r in enumerate(reader, start=1):
            rows.append(
                {
                    "vip_rank": rank_idx,
                    "bucket_index": int(r["bucket_index"]),
                    "center_ppm": float(r["center_ppm"]),
                    "low_ppm": float(r["low_ppm"]) if r.get("low_ppm") else None,
                    "high_ppm": float(r["high_ppm"]) if r.get("high_ppm") else None,
                    "vip": float(r["vip"]),
                    "p_corr": float(r.get("s_plot_corr", r.get("p_corr", "nan"))),
                    "fdr_q": float(r.get("bh_fdr_q", r.get("fdr_q", "nan"))),
                    "bmrb_match": (
                        f"{r['putative_metabolite']} ({r['bmrb_id']})"
                        if r.get("putative_metabolite") and r.get("bmrb_id")
                        else r.get("putative_metabolite", r.get("bmrb_annotation", ""))
                    ),
                }
            )

    vip_sorted = sorted(rows, key=lambda x: (-float(x["vip"]), int(x["bucket_index"])))[:top_k]
    stocsy_top = list(stocsy.get("covarying_buckets", []))[:top_k]

    vip_glc: Set[int] = {
        int(b["bucket_index"])
        for b in vip_sorted
        if bucket_overlaps(float(b["center_ppm"]), b["low_ppm"], b["high_ppm"], wins)
    }
    stocsy_glc: Set[int] = {
        int(b["bucket_index"])
        for b in stocsy_top
        if bucket_overlaps(float(b["center_ppm"]), b.get("low_ppm"), b.get("high_ppm"), wins)
    }
    union_glc = sorted(vip_glc | stocsy_glc)

    salek_targets = [
        ("L-Alanine (CH3 d)", 1.48, 0.04),
        ("Citrate (AB upfield)", 2.54, 0.05),
        ("Citrate (AB downfield)", 2.68, 0.05),
        ("Creatinine (N-CH3)", 3.05, 0.04),
        ("TMAO / Betaine", 3.27, 0.04),
        ("Creatinine (N-CH2)", 4.05, 0.04),
        ("Hippurate (aromatic m/p)", 7.55, 0.06),
        ("Hippurate (aromatic m)", 7.64, 0.06),
        ("Hippurate (aromatic ortho)", 7.84, 0.06),
    ]
    salek_hits: List[Dict[str, Any]] = []
    for name, tppm, tol in salek_targets:
        matches = [
            b
            for b in rows
            if (
                b["low_ppm"] is not None
                and b["high_ppm"] is not None
                and (b["low_ppm"] - tol <= tppm <= b["high_ppm"] + tol)
            )
            or abs(b["center_ppm"] - tppm) <= tol
        ]
        best = min(matches, key=lambda x: x["vip_rank"]) if matches else None
        if best is not None:
            best_clean = {
                k: (v if not (isinstance(v, float) and not math.isfinite(v)) else None)
                for k, v in best.items()
            }
        else:
            best_clean = None
        salek_hits.append(
            {
                "metabolite": name,
                "literature_ppm": tppm,
                "matched": best_clean is not None,
                "best_bucket": best_clean,
            }
        )

    n_clean = int(opls.get("n_active_samples", 0))
    q2 = float(opls.get("q2", math.nan))
    perm_p = float(opls.get("permutation_p_value_q2", math.nan))

    return {
        "n_clean_processed_pqn_samples": n_clean,
        "oplsda_cv_q2": q2 if math.isfinite(q2) else None,
        "oplsda_perm_p_q2": perm_p if math.isfinite(perm_p) else None,
        "r2_y": opls.get("r2_y"),
        "r2_x_predictive": opls.get("r2_x_predictive"),
        "r2_x_orthogonal": opls.get("r2_x_orthogonal"),
        "glucose_glycosuria_buckets_in_top15": len(union_glc),
        "vip_top15_glucose_bucket_indices": sorted(vip_glc),
        "stocsy_top15_glucose_bucket_indices": sorted(stocsy_glc),
        "union_top15_glucose_bucket_indices": union_glc,
        "posthoc_criteria_outcome_summary": {
            "F17-C1_n_clean_ge_126": n_clean >= 126,
            "F17-C2_oplsda_q2_ge_0_25": bool(math.isfinite(q2) and q2 >= 0.25),
            "F17-C3_perm_p_le_0_02": bool(math.isfinite(perm_p) and perm_p <= 0.02),
            "F17-C4_glucose_top15_ge_1": len(union_glc) >= 1,
            "posthoc_cohort_verdict": "PASS"
            if (n_clean >= 126 and math.isfinite(q2) and q2 >= 0.25 and math.isfinite(perm_p) and perm_p <= 0.02 and len(union_glc) >= 1)
            else "FAIL",
        },
        "top_10_vip_buckets": [
            {
                k: (v if not (isinstance(v, float) and not math.isfinite(v)) else None)
                for k, v in b.items()
            }
            for b in vip_sorted[:10]
        ],
        "salek_2007_nonglucose_marker_hits": salek_hits,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Post-hoc, non-blind 1-line ISA-Tab column-fix & MTBLS1 cohort summary (F17 = FAIL)."
    )
    ap.add_argument(
        "--samples-sheet",
        default="/tmp/p17_empirical_downloads/w4/f17/s_MTBLS1.txt",
        help="Path to s_MTBLS1.txt",
    )
    ap.add_argument(
        "--assay-sheet",
        default="/tmp/p17_empirical_downloads/w4/f17/a_MTBLS1_metabolite_profiling_NMR_spectroscopy.txt",
        help="Path to a_MTBLS1_metabolite_profiling_NMR_spectroscopy.txt",
    )
    ap.add_argument(
        "--staging-f17-dir",
        default="/tmp/p17_w4_staging_r2/f17",
        help="Directory holding post-hoc MTBLS1 metabolomics outputs",
    )
    ap.add_argument(
        "--prereg-manifest",
        default=str(Path(__file__).resolve().parents[2] / "data/benchmark/p17_empirical_preregistration.json"),
        help="Path to Phase 17 pre-registration manifest JSON",
    )
    ap.add_argument(
        "--manifest-out",
        default=None,
        help="Optional path to write the 3-column 132-sample mtbls1_manifest.tsv",
    )
    ap.add_argument(
        "--run-nmrassign",
        action="store_true",
        help="Re-execute nmrassign metabolomics bin/pca/oplsda/stocsy from <staging-f17-dir>/f17_spectra",
    )
    ap.add_argument(
        "--nmrassign-bin",
        default=str(Path(__file__).resolve().parents[2] / "target/release/nmrassign"),
        help="Path to compiled nmrassign binary when --run-nmrassign is set",
    )
    ap.add_argument(
        "--json-out",
        default=None,
        help="Optional output path for deterministic JSON diagnostic report",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    stg = Path(args.staging_f17_dir)
    s_txt = Path(args.samples_sheet).read_text(encoding="utf-8", errors="replace")
    a_txt = Path(args.assay_sheet).read_text(encoding="utf-8", errors="replace")
    manifest_out = Path(args.manifest_out) if args.manifest_out else None

    join_res = build_posthoc_mtbls1_manifest_1line_fix(
        s_txt,
        a_txt,
        spectra_dir=stg / "f17_spectra",
        spectrum_ext=".tsv",
        out_manifest_path=manifest_out,
    )

    if args.run_nmrassign:
        mf_tsv = manifest_out or (stg / "mtbls1_manifest.tsv")
        if manifest_out is None:
            build_posthoc_mtbls1_manifest_1line_fix(
                s_txt, a_txt, spectra_dir=stg / "f17_spectra", out_manifest_path=mf_tsv
            )
        bin_exe = str(Path(args.nmrassign_bin).resolve())
        subprocess.run(
            [
                bin_exe, "metabolomics", "bin",
                "--manifest", str(mf_tsv),
                "--min-ppm", "0.50", "--max-ppm", "9.50",
                "--mask", "urine", "--calibrate-tsp",
                "--method", "aib", "--norm", "pqn", "--scale", "pareto",
                "-o", str(stg / "f17_mtbls1_buckets.csv"),
                "--study-out", str(stg / "f17_mtbls1_study.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                bin_exe, "metabolomics", "pca",
                "--in", str(stg / "f17_mtbls1_study.json"),
                "--components", "5",
                "-o", str(stg / "f17_mtbls1_pca.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                bin_exe, "metabolomics", "oplsda",
                "--in", str(stg / "f17_mtbls1_study.json"),
                "--ortho", "1", "--cv-folds", "7",
                "--permutations", "100", "--seed", "20261717",
                "--control", "Control Group", "--case", "diabetes mellitus",
                "--biomarkers-out", str(stg / "f17_mtbls1_biomarkers.csv"),
                "-o", str(stg / "f17_mtbls1_oplsda.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                bin_exe, "metabolomics", "stocsy",
                "--in", str(stg / "f17_mtbls1_study.json"),
                "--driver-ppm", "5.24",
                "-o", str(stg / "f17_mtbls1_stocsy_glucose.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    f17_cfg = load_f17_manifest_windows_and_thresholds(Path(args.prereg_manifest))
    cohort_summary = summarize_biomarkers_and_salek_hits(
        stg / "f17_mtbls1_biomarkers.csv",
        stg / "f17_mtbls1_oplsda.json",
        stg / "f17_mtbls1_stocsy_glucose.json",
        f17_cfg,
    )

    report = {
        "gate_id": "F17",
        "accession_id": "METABOLIGHTS:MTBLS1",
        "diagnostic_only_non_blind": True,
        "alters_primary_gate_verdict": False,
        "primary_strict_bbc770a_verdict": "FAIL",
        "primary_strict_bbc770a_failure_reason": (
            "Unmodified bbc770a scripts/p17_adapters/f17_mtbls1_manifest.py exited rc=1 "
            "('ValueError: 1-to-1 join between s_MTBLS1.txt and a_MTBLS1 (.zip) produced 0 samples') "
            "because EBI MetaboLights NMR ISA-Tab assay sheet names the .zip column "
            "'Free Induction Decay Data File' rather than 'Raw Spectral Data File'."
        ),
        "disclosed_1line_adapter_delta": {
            "frozen_bbc770a_line": 'raw_file = r.get("Raw Spectral Data File", "").strip()',
            "posthoc_1line_fix": 'raw_file = r.get("Raw Spectral Data File", "").strip() or r.get("Free Induction Decay Data File", "").strip()',
            "n_joined_samples": join_res["n_joined_samples"],
            "group_counts": join_res["group_counts"],
            "ignored_nmrml_entries": join_res["ignored_nmrml_entries"],
        },
        "posthoc_nmrassign_metabolomics_cohort_summary": cohort_summary,
    }

    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
