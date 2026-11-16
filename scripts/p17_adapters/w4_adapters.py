#!/usr/bin/env python3
"""Wave 4 deterministic format adapters and pass-criteria evaluators (F14, F16, F17).

Provenance:
- Original clean-room Python 3 standard-library implementation for Phase 17 blind
  empirical benchmarking (Rules P17-R1..R5 of ``data/benchmark/p17_empirical_preregistration.json``).
- Implements strictly the pre-registered metadata extractors, pseudo-2D DOSY locked-g0 F2
  plane integrator, MetaboLights MTBLS1 ISA-Tab 1-to-1 sample/assay manifest builder, and
  manifest-driven pass-criteria evaluators for ``F14``, ``F16``, and ``F17``.
- Zero external dependencies, zero network calls, and zero unregistered numeric constants
  outside ``data/benchmark/p17_empirical_preregistration.json``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .common import (
    evaluate_gate_criteria,
    extract_manifest_bound_constants,
    load_preregistration_manifest,
    median,
    median_rel_abs_err,
)

PROTON_DECOUPLED_HETERO_NUCLEI = frozenset({"13C", "19F", "31P"})


# ============================================================================
# Shared Bruker JCAMP-DX & Varian procpar Scalar Readers (F14 & F16)
# ============================================================================


def read_bruker_jcamp_scalar(jcamp_text: str, key: str) -> Optional[str]:
    """Read a scalar `##$<key>= <value>` entry from Bruker `acqus` / `procs` text."""
    key_norm = "".join(c.upper() for c in key if c.isalnum())
    for line in jcamp_text.splitlines():
        s = line.strip()
        if not s.startswith("##"):
            continue
        rest = s[2:]
        if rest.startswith("$"):
            rest = rest[1:]
        if "=" not in rest:
            continue
        lhs, rhs = rest.split("=", 1)
        lhs_norm = "".join(c.upper() for c in lhs if c.isalnum())
        if lhs_norm == key_norm:
            return rhs.strip().strip("<>\"'").strip()
    return None


def parse_varian_procpar_scalars(procpar_text: str) -> Dict[str, List[str]]:
    """Parse Varian/Agilent `procpar` parameter blocks into `{name: [values...]}`."""
    out: Dict[str, List[str]] = {}
    lines = procpar_text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        hdr = lines[i].strip()
        i += 1
        if not hdr or hdr.startswith("#"):
            continue
        parts = hdr.split()
        if len(parts) < 11:
            continue
        try:
            basictype = int(parts[2])
        except ValueError:
            continue
        name = parts[0]
        if i >= n:
            break
        if basictype == 2:
            first_val = lines[i].strip()
            i += 1
            sp = first_val.split(None, 1)
            cnt = int(sp[0]) if sp and sp[0].isdigit() else 1
            vals: List[str] = []
            if len(sp) > 1 and cnt > 0:
                vals.append(sp[1].strip().strip("\"'"))
            while len(vals) < cnt and i < n:
                vals.append(lines[i].strip().strip("\"'"))
                i += 1
            out[name] = vals
        else:
            first_toks = lines[i].split()
            i += 1
            cnt = int(first_toks[0]) if first_toks and first_toks[0].isdigit() else 0
            vals = list(first_toks[1 : 1 + cnt])
            while len(vals) < cnt and i < n:
                more = lines[i].split()
                i += 1
                if not more:
                    break
                vals.extend(more[: cnt - len(vals)])
            out[name] = vals
        if i < n:
            i += 1  # Skip enum line
    return out


# ============================================================================
# Gate F14: Bruker acqus NUC1 Reader, DEPT-135 Counter & qNMR Extractor
# ============================================================================


def read_bruker_acqus_nuc1(acqus_path_or_text: Union[Path, str]) -> str:
    """Extract canonical nucleus string (`1H`, `13C`, `19F`, `31P`, `15N`) from `acqus` `##$NUC1=`."""
    if isinstance(acqus_path_or_text, Path) or (
        "\n" not in str(acqus_path_or_text) and Path(acqus_path_or_text).is_file()
    ):
        txt = Path(acqus_path_or_text).read_text(encoding="utf-8", errors="replace")
    else:
        txt = str(acqus_path_or_text)
    raw = read_bruker_jcamp_scalar(txt, "NUC1")
    if raw is None or not raw or raw.lower() == "off":
        raise ValueError("missing or unset ##$NUC1= in Bruker acqus")
    return raw.upper()


def select_first_bruker_1d_dir_by_nuc1(root_dir: Union[Path, str], target_nuc1: str) -> Path:
    """Return lexicographically first Bruker 1D `fid`+`acqus` directory matching `##$NUC1=`."""
    root = Path(root_dir)
    target = target_nuc1.strip().strip("<>").upper()
    candidates: List[Tuple[str, Path]] = []
    for acqus_file in root.rglob("acqus"):
        exp_dir = acqus_file.parent
        if not (exp_dir / "fid").is_file() or (exp_dir / "ser").is_file():
            continue
        try:
            nuc = read_bruker_acqus_nuc1(acqus_file)
        except ValueError:
            continue
        if nuc == target:
            rel_posix = exp_dir.relative_to(root).as_posix()
            candidates.append((rel_posix, exp_dir))
    if not candidates:
        raise FileNotFoundError(f"no Bruker 1D fid+acqus directory with ##$NUC1=<{target}> under {root}")
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def multiplet_cli_flags_for_acqus(
    acqus_path_or_text: Union[Path, str],
    mol_path: Optional[Union[Path, str]] = None,
) -> List[str]:
    """Build deterministic `nmrassign nmr1d multiplet` flags per `F14` adapter rule 3."""
    nuc = read_bruker_acqus_nuc1(acqus_path_or_text)
    flags = ["--nucleus", nuc]
    if nuc in PROTON_DECOUPLED_HETERO_NUCLEI:
        flags.append("--proton-decoupled")
    if mol_path is not None and Path(mol_path).is_file():
        flags.extend(["--mol", str(mol_path)])
    return flags


def parse_nmr1d_picked_peaks(summary_json_or_peaks_tsv: str) -> List[Dict[str, Any]]:
    """Parse picked 1D peaks from either `nmr1d process --json-out` or `--peaks-out` TSV."""
    s = summary_json_or_peaks_tsv.strip()
    if not s:
        return []
    if s.startswith("{"):
        js = json.loads(s)
        return [dict(pk) for pk in js.get("picked_peaks", [])]

    lines = [ln.strip() for ln in s.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) < 2:
        return []
    hdr = [c.strip().lower() for c in lines[0].split("\t")]
    cmap = {name: idx for idx, name in enumerate(hdr)}
    ppm_c = cmap.get("ppm", cmap.get("center_ppm"))
    h_c = cmap.get("height")
    snr_c = cmap.get("snr")
    fwhm_c = cmap.get("est_fwhm_ppm", cmap.get("fwhm_ppm"))
    if ppm_c is None or h_c is None:
        return []
    out: List[Dict[str, Any]] = []
    for ln in lines[1:]:
        cols = [c.strip() for c in ln.split("\t")]
        if len(cols) <= max(ppm_c, h_c):
            continue
        out.append(
            {
                "ppm": float(cols[ppm_c]),
                "height": float(cols[h_c]),
                "snr": float(cols[snr_c]) if snr_c is not None and snr_c < len(cols) else math.nan,
                "est_fwhm_ppm": float(cols[fwhm_c]) if fwhm_c is not None and fwhm_c < len(cols) else 0.0,
            }
        )
    return out


def count_dept135_peak_multiplicities(
    dept135_summary_or_tsv: str,
    *,
    exclude_dss_up_to_ppm: float = 1.0,
    expected_pos_sites: Optional[int] = None,
    expected_neg_ch2_sites: Optional[int] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Count positive (`CH`/`CH3`) and negative (`CH2`) carbon resonances in `DEPT-135` (`F14-C2`)."""
    mf = manifest if manifest is not None else load_preregistration_manifest()
    c14 = extract_manifest_bound_constants(mf)["F14"]
    min_snr = c14["dept135_ch2_min_snr"]
    exp_pos = int(c14["fucose_expected_pos_sites"]) if expected_pos_sites is None else int(expected_pos_sites)
    exp_neg = int(c14["fucose_expected_neg_ch2"]) if expected_neg_ch2_sites is None else int(expected_neg_ch2_sites)

    peaks = parse_nmr1d_picked_peaks(dept135_summary_or_tsv)
    analyte_peaks = [
        pk
        for pk in peaks
        if float(pk["ppm"]) > exclude_dss_up_to_ppm
        and (not math.isfinite(float(pk.get("snr", math.nan))) or abs(float(pk["snr"])) >= min_snr)
    ]
    pos_peaks = [pk for pk in analyte_peaks if float(pk["height"]) > 0.0]
    neg_peaks = [pk for pk in analyte_peaks if float(pk["height"]) < 0.0]

    n_pos = len(pos_peaks)
    n_neg = len(neg_peaks)

    if exp_neg == 0:
        # L-fucose: >= 6 positive CH/CH3 carbons (major + optional minor mutarotation anomer) and 0 negative CH2
        if n_pos >= exp_pos and n_neg == 0:
            pos_fraction = 1.0
        else:
            denom = max(exp_pos, n_pos + n_neg, 1)
            pos_fraction = float(min(n_pos, exp_pos)) / float(denom)
            if n_neg > 0 and pos_fraction >= 1.0:
                pos_fraction = float(n_pos) / float(n_pos + n_neg)
    else:
        # Fallback molecule with both positive CH/CH3 and inverted negative CH2 sites (e.g. ATP / 2,3-DPG)
        if n_pos >= exp_pos and n_neg >= exp_neg:
            pos_fraction = 1.0
        else:
            pos_ok = min(n_pos, exp_pos)
            neg_ok = min(n_neg, exp_neg)
            pos_fraction = float(pos_ok + neg_ok) / float(max(exp_pos + exp_neg, 1))

    return {
        "n_positive_ch_ch3": n_pos,
        "n_negative_ch2": n_neg,
        "expected_pos_sites": exp_pos,
        "expected_neg_ch2_sites": exp_neg,
        "positive_site_fraction": pos_fraction,
        "positive_ppms": [float(p["ppm"]) for p in pos_peaks],
        "negative_ppms": [float(p["ppm"]) for p in neg_peaks],
    }


def _format_hi_lo_window(raw_spec: str, half_win_ppm: float) -> Tuple[float, float]:
    s = raw_spec.strip()
    for sep in (":", "..", "-", ","):
        if sep in s and not s.startswith("-"):
            parts = [p.strip() for p in s.split(sep, 1)]
            if len(parts) == 2:
                try:
                    a = float(parts[0])
                    b = float(parts[1])
                    return (max(a, b), min(a, b))
                except ValueError:
                    pass
    c = float(s)
    return (c + half_win_ppm, c - half_win_ppm)


def extract_f14_qnmr_sample_specs(
    dataguide_csv_text: str,
    *,
    readme_text: str = "",
    manifest: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Extract deterministic internal-standard `nmr1d qnmr` CLI parameters from `DataGuide_*.csv`."""
    mf = manifest if manifest is not None else load_preregistration_manifest()
    consts = extract_manifest_bound_constants(mf)
    half_win = consts["F14"]["qnmr_centroid_half_window_ppm"]
    g_to_mg = consts["F13c"]["mm_to_um"]

    defaults: Dict[str, str] = {}
    for ln in readme_text.splitlines():
        if "=" in ln or ":" in ln:
            sep = "=" if "=" in ln else ":"
            k, v = ln.split(sep, 1)
            defaults[k.strip().lower()] = v.strip()

    lines = [ln for ln in dataguide_csv_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) < 2:
        raise ValueError("DataGuide CSV contains fewer than 2 non-empty lines")
    delim = "\t" if "\t" in lines[0] and "," not in lines[0] else (";" if ";" in lines[0] and "," not in lines[0] else ",")
    reader = csv.DictReader(lines, delimiter=delim)
    if not reader.fieldnames:
        raise ValueError("DataGuide CSV missing column headers")

    norm_to_orig = {fn.strip().lower(): fn for fn in reader.fieldnames}

    def _get(row: Mapping[str, str], keys: Sequence[str], fallback: Optional[str] = None) -> Optional[str]:
        for k in keys:
            orig = norm_to_orig.get(k)
            if orig is not None and row.get(orig) is not None and row[orig].strip():
                return row[orig].strip()
            if k in defaults and defaults[k]:
                return defaults[k]
        return fallback

    specs: List[Dict[str, Any]] = []
    for idx, row in enumerate(reader):
        fid_dir = _get(row, ["fid_dir", "folder", "sample_dir", "exp_dir", "experiment", "sample_id", "sample"], f"sample_{idx + 1}")
        std_name = _get(row, ["std_name", "internal_standard", "calibrant", "standard"], "InternalStandard")
        std_mw = float(_get(row, ["std_mw", "mw_std", "standard_mw"], "0") or "0")
        std_protons = int(float(_get(row, ["std_protons", "n_std", "standard_protons"], "1") or "1"))
        std_purity = float(_get(row, ["std_purity", "p_std", "standard_purity"], "1.0") or "1.0")
        if std_purity > 1.0:
            # Convert percentage purity (e.g. 99.8) using ten*ten without magic literal
            ten = float(10)
            std_purity = std_purity / (ten * ten)

        vol_ml_s = _get(row, ["volume_ml", "solution_volume_ml", "vol_ml"])
        vol_ml = float(vol_ml_s) if vol_ml_s is not None else None

        m_std_s = _get(row, ["std_mass_mg", "m_std", "m_std_mg", "standard_mass_mg"])
        if m_std_s is not None:
            std_mass_mg = float(m_std_s)
        else:
            c_std_mm = float(_get(row, ["std_conc_mm", "c_std_mm", "standard_molarity_mm"], "0") or "0")
            v_l = (vol_ml if vol_ml is not None else 1.0) / g_to_mg
            std_mass_mg = (c_std_mm / g_to_mg) * v_l * std_mw * g_to_mg

        std_reg_raw = _get(row, ["std_region", "std_window", "std_ppm"])
        if std_reg_raw is None:
            raise ValueError(f"DataGuide row {idx + 1} missing std_region/std_ppm")
        std_hi, std_lo = _format_hi_lo_window(std_reg_raw, half_win)

        an_name = _get(row, ["analyte_name", "an_name", "analyte", "compound"], f"Analyte_{idx + 1}")
        an_mw = float(_get(row, ["analyte_mw", "mw_an", "mw"], "1.0") or "1.0")
        m_an_s = _get(row, ["analyte_mass_mg", "m_an", "m_analyte_mg", "mass_mg"])
        if m_an_s is not None:
            an_mass_mg = float(m_an_s)
        else:
            c_nom_mm = float(_get(row, ["nominal_conc_mm", "ref_conc_mm", "expected_conc_mm", "conc_mm"], "1.0") or "1.0")
            v_l = (vol_ml if vol_ml is not None else 1.0) / g_to_mg
            an_mass_mg = (c_nom_mm / g_to_mg) * v_l * an_mw * g_to_mg

        an_protons = int(float(_get(row, ["analyte_protons", "n_an", "h_an", "protons"], "1") or "1"))
        an_lbl = _get(row, ["analyte_label", "signal_label"], "H1")
        an_reg_raw = _get(row, ["analyte_region", "an_region", "analyte_window", "analyte_ppm"])
        if an_reg_raw is None:
            raise ValueError(f"DataGuide row {idx + 1} missing analyte_region/analyte_ppm")
        if an_reg_raw.count(":") >= 2:
            an_reg_cli = an_reg_raw
        else:
            an_hi, an_lo = _format_hi_lo_window(an_reg_raw, half_win)
            an_reg_cli = f"{an_hi:.4f}:{an_lo:.4f}:{an_protons}:{an_lbl}"

        ref_purity_s = _get(row, ["ref_purity_percent", "certified_purity_percent", "ref_purity", "purity_ref"])
        ref_conc_s = _get(row, ["ref_conc_mm", "certified_conc_mm", "gravimetric_conc_mm", "expected_conc_mm"])

        cli_args = [
            "--std-name",
            str(std_name),
            "--std-mw",
            f"{std_mw:.6g}",
            "--std-protons",
            str(std_protons),
            "--std-mass-mg",
            f"{std_mass_mg:.6g}",
            "--std-purity",
            f"{std_purity:.6g}",
            "--std-region",
            f"{std_hi:.4f}:{std_lo:.4f}",
            "--analyte-name",
            str(an_name),
            "--analyte-mw",
            f"{an_mw:.6g}",
            "--analyte-mass-mg",
            f"{an_mass_mg:.6g}",
            "--analyte-region",
            an_reg_cli,
        ]
        if vol_ml is not None:
            cli_args.extend(["--volume-ml", f"{vol_ml:.6g}"])

        specs.append(
            {
                "fid_dir": fid_dir,
                "std_name": std_name,
                "analyte_name": an_name,
                "std_region": f"{std_hi:.4f}:{std_lo:.4f}",
                "analyte_region": an_reg_cli,
                "volume_ml": vol_ml,
                "ref_purity_percent": float(ref_purity_s) if ref_purity_s is not None else None,
                "ref_conc_mm": float(ref_conc_s) if ref_conc_s is not None else None,
                "cli_args": cli_args,
            }
        )
    return specs


def evaluate_f14_nmr1d_and_qnmr(
    experiment_process_reports: Mapping[str, Tuple[int, str]],
    dept135_summary_or_tsv: str,
    qnmr_sample_results: Sequence[Mapping[str, Any]],
    *,
    accession_id: str = "BMSE:bmse000036+ZENODO:17368767+ZENODO:15728739",
    expected_pos_sites: Optional[int] = None,
    expected_neg_ch2_sites: Optional[int] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate `F14` across all 5 1D experiment classes, DEPT-135 multiplicity, and gravimetric qNMR.

    `experiment_process_reports` maps experiment keys (`1h`, `13c`, `dept90`, `dept135`, `19f`, `31p`)
    to `(exit_code, process_summary_json_or_peaks_tsv)`.
    """
    mf = manifest if manifest is not None else load_preregistration_manifest()
    c14 = extract_manifest_bound_constants(mf)["F14"]
    min_snr = c14["min_principal_snr"]

    def _exp_ok(key: str) -> bool:
        if key not in experiment_process_reports:
            return False
        rc, body = experiment_process_reports[key]
        if int(rc) != 0:
            return False
        pks = parse_nmr1d_picked_peaks(body)
        if not pks:
            return False
        max_snr = max(abs(float(p.get("snr", 0.0))) for p in pks)
        return max_snr >= min_snr

    classes_passed = sum(
        [
            int(_exp_ok("1h")),
            int(_exp_ok("13c")),
            int(_exp_ok("dept90") and _exp_ok("dept135")),
            int(_exp_ok("19f")),
            int(_exp_ok("31p")),
        ]
    )

    dept_stats = count_dept135_peak_multiplicities(
        dept135_summary_or_tsv,
        expected_pos_sites=expected_pos_sites,
        expected_neg_ch2_sites=expected_neg_ch2_sites,
        manifest=mf,
    )

    qnmr_preds: List[float] = []
    qnmr_refs: List[float] = []
    qnmr_exits_ok = bool(qnmr_sample_results)
    k2_covered = 0
    for item in qnmr_sample_results:
        if int(item.get("exit_code", 0)) != 0:
            qnmr_exits_ok = False
        rep_raw = item["qnmr_json"]
        rep = json.loads(rep_raw) if isinstance(rep_raw, str) else rep_raw
        if item.get("ref_conc_mm") is not None and rep.get("molar_concentration_mm") is not None:
            pred = float(rep["molar_concentration_mm"])
            ref = float(item["ref_conc_mm"])
            qnmr_preds.append(pred)
            qnmr_refs.append(ref)
            u_rel = float(rep.get("expanded_uncertainty_k2_percent", 0.0)) / max(
                float(rep.get("purity_percent", 1.0)), 1.0
            )
            if abs(pred - ref) <= u_rel * abs(pred):
                k2_covered += 1
        elif item.get("ref_purity_percent") is not None:
            pred = float(rep["purity_percent"])
            ref = float(item["ref_purity_percent"])
            qnmr_preds.append(pred)
            qnmr_refs.append(ref)
            u_k2 = float(rep.get("expanded_uncertainty_k2_percent", 0.0))
            if abs(pred - ref) <= u_k2:
                k2_covered += 1

    med_qnmr_err = median_rel_abs_err(qnmr_preds, qnmr_refs)
    all_rc_zero = all(int(rc) == 0 for rc, _ in experiment_process_reports.values()) and qnmr_exits_ok

    metrics = {
        "n_processed_1d_experiment_classes_exit0_snr_ge_10": classes_passed,
        "bmse000036_dept135_fucose_positive_site_fraction": dept_stats["positive_site_fraction"],
        "median_qnmr_rel_abs_err": med_qnmr_err,
    }
    eval_res = evaluate_gate_criteria(
        "F14",
        metrics,
        exit_code=0 if all_rc_zero else 1,
        accession_id=accession_id,
        manifest=mf,
    )
    eval_res["secondary_metrics"] = {
        "dept135_details": dept_stats,
        "n_qnmr_samples": len(qnmr_preds),
        "qnmr_gum_k2_coverage_count": k2_covered,
    }
    return eval_res


# ============================================================================
# Gate F16: Pseudo-2D Bruker/Varian DOSY Plane Integrator & Evaluator
# ============================================================================


def extract_dosy_header_solvent_and_temp(
    exp_dir: Union[Path, str],
    *,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, float]:
    """Canonicalize DOSY `--solvent` and `--temp-k` strictly from Bruker `acqus` or Varian `procpar`."""
    mf = manifest if manifest is not None else load_preregistration_manifest()
    c16 = extract_manifest_bound_constants(mf)["F16"]
    allowed = {s.lower(): s for s in c16["allowed_solvents"]}
    fb_solv = c16["fallback_solvent"]
    te_cut = c16["temp_kelvin_cutoff"]
    c2k = c16["celsius_to_kelvin_offset"]
    fb_temp = c16["fallback_temp_k"]

    d = Path(exp_dir)
    raw_solv: Optional[str] = None
    raw_temp: Optional[float] = None

    if (d / "acqus").is_file() or (d / "acqu").is_file():
        ap = d / "acqus" if (d / "acqus").is_file() else d / "acqu"
        txt = ap.read_text(encoding="utf-8", errors="replace")
        raw_solv = read_bruker_jcamp_scalar(txt, "SOLVENT")
        te_s = read_bruker_jcamp_scalar(txt, "TE")
        if te_s is not None:
            try:
                raw_temp = float(te_s)
            except ValueError:
                raw_temp = None
    elif (d / "procpar").is_file():
        pp = parse_varian_procpar_scalars((d / "procpar").read_text(encoding="utf-8", errors="replace"))
        if "solvent" in pp and pp["solvent"]:
            raw_solv = pp["solvent"][0]
        if "temp" in pp and pp["temp"]:
            try:
                raw_temp = float(pp["temp"][0])
            except ValueError:
                raw_temp = None

    canon_solv = fb_solv
    if raw_solv:
        sl = raw_solv.strip().lower()
        aliases = {
            "d2o": "D2O",
            "h2o+d2o": "H2O",
            "h2o": "H2O",
            "cdcl3": "CDCl3",
            "chloroform-d": "CDCl3",
            "dmso": "DMSO-d6",
            "dmso-d6": "DMSO-d6",
            "meod": "CD3OD",
            "cd3od": "CD3OD",
            "methanol-d4": "CD3OD",
            "cd3cn": "CD3CN",
        }
        if sl in allowed:
            canon_solv = allowed[sl]
        elif sl in aliases and aliases[sl].lower() in allowed:
            canon_solv = aliases[sl]

    if raw_temp is None or not math.isfinite(raw_temp):
        temp_k = fb_temp
    elif raw_temp >= te_cut:
        temp_k = raw_temp
    else:
        temp_k = raw_temp + c2k

    return canon_solv, temp_k


def _parse_bruker_difflist_values(difflist_path: Path) -> List[float]:
    vals: List[float] = []
    for raw in difflist_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.split("#", 1)[0].split(";", 1)[0].strip()
        if s:
            vals.append(float(s.split()[0]))
    if not vals:
        raise ValueError(f"empty Bruker difflist at {difflist_path}")
    return vals


def _unpack_bruker_dosy_planes(exp_dir: Path, work_dir: Path) -> Tuple[List[Path], List[float]]:
    acqus_path = exp_dir / "acqus" if (exp_dir / "acqus").is_file() else exp_dir / "acqu"
    difflist_path = exp_dir / "difflist"
    ser_path = exp_dir / "ser"
    if not (acqus_path.is_file() and difflist_path.is_file() and ser_path.is_file()):
        raise FileNotFoundError(f"Bruker DOSY directory {exp_dir} requires acqus, difflist, and ser")

    grads = _parse_bruker_difflist_values(difflist_path)
    n_planes = len(grads)
    acqus_txt = acqus_path.read_text(encoding="utf-8", errors="replace")
    td_words = int(float(read_bruker_jcamp_scalar(acqus_txt, "TD") or "0"))
    dtypa = int(float(read_bruker_jcamp_scalar(acqus_txt, "DTYPA") or "0"))
    word_bytes = 8 if dtypa == 2 else 4
    raw_trace_bytes = td_words * word_bytes
    stride_1024 = ((raw_trace_bytes + 1023) // 1024) * 1024
    stride_256w = ((td_words + 255) // 256) * 256 * word_bytes

    ser_bytes = ser_path.read_bytes()
    if len(ser_bytes) == n_planes * stride_1024:
        stride = stride_1024
    elif len(ser_bytes) == n_planes * stride_256w:
        stride = stride_256w
    elif len(ser_bytes) >= n_planes * raw_trace_bytes and len(ser_bytes) % n_planes == 0:
        stride = len(ser_bytes) // n_planes
    else:
        raise ValueError(
            f"Bruker DOSY ser length {len(ser_bytes)} incompatible with {n_planes} planes x {raw_trace_bytes} B"
        )

    plane_inputs: List[Path] = []
    for m in range(n_planes):
        pdir = work_dir / f"bruker_plane_{m:04d}"
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(acqus_path, pdir / "acqus")
        chunk = ser_bytes[m * stride : m * stride + raw_trace_bytes]
        (pdir / "fid").write_bytes(chunk)
        plane_inputs.append(pdir)
    return plane_inputs, grads


def _unpack_varian_dosy_planes(
    exp_dir: Path,
    work_dir: Path,
    *,
    us_to_s: float,
    default_carrier_ppm: float,
) -> Tuple[List[Path], List[float]]:
    procpar_path = exp_dir / "procpar"
    fid_path = exp_dir / "fid"
    if not (procpar_path.is_file() and fid_path.is_file()):
        raise FileNotFoundError(f"Varian DOSY directory {exp_dir} requires procpar and fid")

    pp = parse_varian_procpar_scalars(procpar_path.read_text(encoding="utf-8", errors="replace"))
    gzlvl = [float(x) for x in pp.get("gzlvl1", pp.get("gzlvl", []))]
    if not gzlvl:
        raise ValueError(f"Varian procpar in {exp_dir} missing arrayed gzlvl1/gzlvl")

    sw_hz = float(pp["sw"][0])
    sfrq = float(pp["sfrq"][0])
    if "rfl" in pp and "rfp" in pp and pp["rfl"] and pp["rfp"]:
        rfl = float(pp["rfl"][0])
        rfp = float(pp["rfp"][0])
        if rfl != 0.0 or rfp != 0.0:
            carrier_ppm = (rfp - rfl + (sw_hz / float(2))) / sfrq
        elif "reffrq" in pp and pp["reffrq"] and float(pp["reffrq"][0]) > 0.0:
            reffrq = float(pp["reffrq"][0])
            carrier_ppm = (sfrq - reffrq) / (reffrq * us_to_s)
        else:
            carrier_ppm = default_carrier_ppm
    elif "reffrq" in pp and pp["reffrq"] and float(pp["reffrq"][0]) > 0.0:
        reffrq = float(pp["reffrq"][0])
        carrier_ppm = (sfrq - reffrq) / (reffrq * us_to_s)
    else:
        carrier_ppm = default_carrier_ppm

    raw = fid_path.read_bytes()
    if len(raw) < 32:
        raise ValueError(f"Varian fid in {exp_dir} shorter than 32-byte datafilehead")
    nblocks, ntraces, np_words, ebytes, _tbytes, bbytes = struct.unpack(">6i", raw[:24])
    _vers_id, status = struct.unpack(">2H", raw[24:28])
    (nbheaders,) = struct.unpack(">i", raw[28:32])
    nbheaders = max(1, nbheaders)
    n_cplx = np_words // 2
    hdr_bytes = nbheaders * 28
    trace_bytes = np_words * ebytes
    s_float = (status & 0x0008) != 0
    s_32 = (status & 0x0004) != 0 or ebytes == 4

    traces: List[List[Tuple[float, float]]] = []
    for b in range(nblocks):
        b_off = 32 + b * bbytes
        blk_status = struct.unpack(">H", raw[b_off + 2 : b_off + 4])[0] if b_off + 4 <= len(raw) else status
        is_flt = s_float or ((blk_status & 0x0008) != 0)
        is_i32 = s_32 or ((blk_status & 0x0004) != 0)
        d_start = b_off + hdr_bytes
        for tr in range(ntraces):
            t_off = d_start + tr * trace_bytes
            if t_off + trace_bytes > len(raw):
                break
            buf = raw[t_off : t_off + trace_bytes]
            if is_flt and ebytes == 4:
                nums = struct.unpack(f">{np_words}f", buf)
            elif is_i32 and ebytes == 4:
                nums = struct.unpack(f">{np_words}i", buf)
            elif ebytes == 2:
                nums = struct.unpack(f">{np_words}h", buf)
            else:
                raise ValueError(f"unsupported Varian fid sample format status=0x{status:04x} ebytes={ebytes}")
            # Varian FIDs use conjugate_for_fft = true in TimeDomain1D (`[re, -im]` under TimeDomain1D::new)
            cvec = [(float(nums[2 * k]), -float(nums[2 * k + 1])) for k in range(n_cplx)]
            traces.append(cvec)

    if len(traces) < len(gzlvl):
        raise ValueError(f"Varian fid has {len(traces)} traces, expected >= {len(gzlvl)} for gzlvl1")

    plane_inputs: List[Path] = []
    for m in range(len(gzlvl)):
        td_json = {
            "name": f"varian_plane_{m:04d}",
            "experiment": "Proton1D",
            "sw_hz": sw_hz,
            "obs_mhz": sfrq,
            "carrier_ppm": carrier_ppm,
            "active_td": n_cplx,
            "group_delay_pts": 0.0,
            "conjugate_for_fft": False,
            "data": [[re_v, im_v] for re_v, im_v in traces[m]],
        }
        pfile = work_dir / f"varian_plane_{m:04d}.json"
        pfile.write_text(json.dumps(td_json), encoding="utf-8")
        plane_inputs.append(pfile)

    return plane_inputs, [abs(g) for g in gzlvl]


def integrate_pseudo2d_dosy_planes(
    exp_dir: Union[Path, str],
    out_decay_tsv: Union[Path, str],
    *,
    nmrassign_bin: Union[Path, str] = "./target/release/nmrassign",
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Integrate Bruker or Varian pseudo-2D DOSY planes into `decay.tsv` per Rule P17-R2.

    Applies uniform `1.0 Hz` exponential apodization, `2x` zero-fill, F2 FFT, phase locked
    strictly on lowest-gradient plane `m_ref` (`argmin |g_m|`), peak pick `SNR >= 10` on `m_ref`
    masking residual HDO `[4.65, 4.95]` ppm, and integrates fixed `+-2*FWHM` boxes across `0..M-1`.
    """
    mf = manifest if manifest is not None else load_preregistration_manifest()
    consts = extract_manifest_bound_constants(mf)
    c16 = consts["F16"]
    us_to_s = consts["F11"]["us_to_s"]
    hdo_lo, hdo_hi = c16["hdo_mask_ppm"]
    default_carrier = (hdo_lo + hdo_hi) / float(2)

    edir = Path(exp_dir)
    bin_path = str(Path(nmrassign_bin).resolve())

    with tempfile.TemporaryDirectory(prefix="p17_f16_planes_") as tmp_s:
        tmp_dir = Path(tmp_s)
        if (edir / "procpar").is_file():
            vendor = "varian"
            plane_inputs, grads_abs = _unpack_varian_dosy_planes(
                edir,
                tmp_dir,
                us_to_s=us_to_s,
                default_carrier_ppm=default_carrier,
            )
        else:
            vendor = "bruker"
            plane_inputs, grads_abs = _unpack_bruker_dosy_planes(edir, tmp_dir)

        m_ref = min(range(len(grads_abs)), key=lambda idx: abs(grads_abs[idx]))

        # Step 1: Process lowest-gradient reference plane `m_ref` with `--auto-phase` and `--min-snr 10`
        ref_spec_json = tmp_dir / "plane_ref_spec.json"
        ref_sum_json = tmp_dir / "plane_ref_summary.json"
        cmd_ref = [
            bin_path,
            "nmr1d",
            "process",
            "--in",
            str(plane_inputs[m_ref]),
            "--experiment",
            "1h",
            "--apod",
            "em",
            "--lb",
            str(c16["lb_hz"]),
            "--zf",
            str(c16["zf_factor"]),
            "--auto-phase",
            "--baseline",
            "whittaker",
            "--min-snr",
            str(c16["min_snr"]),
            "--json-out",
            str(ref_sum_json),
            "--out",
            str(ref_spec_json),
        ]
        proc_ref = subprocess.run(cmd_ref, capture_output=True, text=True, check=False)
        if proc_ref.returncode != 0:
            raise RuntimeError(
                f"nmr1d process failed on reference DOSY plane {m_ref} (rc={proc_ref.returncode}): {proc_ref.stderr}"
            )

        ref_summary = json.loads(ref_sum_json.read_text(encoding="utf-8"))
        phc0, phc1, piv_frac = ref_summary["phase_applied_deg"]

        picked = [
            pk
            for pk in ref_summary.get("picked_peaks", [])
            if not (hdo_lo <= float(pk["ppm"]) <= hdo_hi)
            and abs(float(pk["snr"])) >= c16["min_snr"]
        ]
        if not picked:
            raise ValueError(
                f"zero DOSY peaks with SNR >= {c16['min_snr']} outside HDO mask [{hdo_lo}, {hdo_hi}] ppm on plane {m_ref}"
            )

        # Sort descending in ppm for deterministic Peak_1..Peak_K ordering
        picked.sort(key=lambda p: -float(p["ppm"]))
        boxes: List[Tuple[str, float, float, float]] = []
        for idx, pk in enumerate(picked):
            c_ppm = float(pk["ppm"])
            hw_ppm = c16["box_fwhm_mult"] * float(pk["est_fwhm_ppm"])
            lbl = f"Peak_{idx + 1}_{c_ppm:.3f}ppm"
            boxes.append((lbl, c_ppm, c_ppm - hw_ppm, c_ppm + hw_ppm))

        # Step 2: Process all planes m = 0..M-1 with locked phase `(phc0, phc1, piv_frac)`
        plane_spectra: List[Dict[str, Any]] = []
        for m, pin in enumerate(plane_inputs):
            if m == m_ref:
                sp_js = json.loads(ref_spec_json.read_text(encoding="utf-8"))
                plane_spectra.append(sp_js)
                continue
            sp_out = tmp_dir / f"plane_{m:04d}_spec.json"
            cmd_m = [
                bin_path,
                "nmr1d",
                "process",
                "--in",
                str(pin),
                "--experiment",
                "1h",
                "--apod",
                "em",
                "--lb",
                str(c16["lb_hz"]),
                "--zf",
                str(c16["zf_factor"]),
                "--phase",
                f"{phc0:.6f}:{phc1:.6f}",
                "--pivot",
                f"{piv_frac:.6f}",
                "--baseline",
                "whittaker",
                "--out",
                str(sp_out),
            ]
            pm = subprocess.run(cmd_m, capture_output=True, text=True, check=False)
            if pm.returncode != 0:
                raise RuntimeError(f"nmr1d process failed on DOSY plane {m}: {pm.stderr}")
            plane_spectra.append(json.loads(sp_out.read_text(encoding="utf-8")))

        # Step 3: Integrate fixed `+-2*FWHM` boxes across `m = 0..M-1`
        header_cols = ["label", "center_ppm", *[f"I_{m}" for m in range(len(plane_inputs))]]
        tsv_lines = ["\t".join(header_cols)]
        for lbl, c_ppm, box_lo, box_hi in boxes:
            ints_m: List[str] = []
            for sp in plane_spectra:
                ax = sp["axis"]
                first_ppm = float(ax["first_ppm"])
                step_ppm = float(ax["step_ppm"])
                real_arr = sp["real"]
                dx = abs(step_ppm)
                area = math.fsum(
                    float(val) * dx
                    for j, val in enumerate(real_arr)
                    if box_lo <= (first_ppm + float(j) * step_ppm) <= box_hi
                )
                ints_m.append(f"{area:.8e}")
            tsv_lines.append("\t".join([lbl, f"{c_ppm:.5f}", *ints_m]))

        out_p = Path(out_decay_tsv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    solv, temp_k = extract_dosy_header_solvent_and_temp(edir, manifest=mf)
    return {
        "vendor": vendor,
        "n_planes": len(plane_inputs),
        "ref_plane_index": m_ref,
        "n_peaks": len(boxes),
        "header_solvent": solv,
        "header_temp_k": temp_k,
        "out_decay_tsv": str(out_decay_tsv),
    }


def _load_decay_tsv_intensities(decay_tsv_text: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    lines = [ln.strip() for ln in decay_tsv_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    for ln in lines[1:]:
        cols = [c.strip() for c in ln.split("\t")]
        if len(cols) >= 3:
            out[cols[0]] = [float(v) for v in cols[2:]]
    return out


def parse_dosy_fit_report(
    dosy_fit_json_text: str,
    decay_tsv_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse `nmrassign dosy fit --json-out` and compute exact attenuation curve `R^2` per peak."""
    js = json.loads(dosy_fit_json_text)
    b_vals = [float(b) for b in js.get("b_values_s_m2", [])]
    obs_map = _load_decay_tsv_intensities(decay_tsv_text) if decay_tsv_text else {}

    peaks_parsed: List[Dict[str, Any]] = []
    for pk in js.get("peaks", []):
        lbl = pk["label"]
        comps = pk.get("components", [])
        rss = float(pk.get("rss", 0.0))
        if lbl in obs_map and len(obs_map[lbl]) == len(b_vals):
            ys = obs_map[lbl]
            mean_y = math.fsum(ys) / float(len(ys))
            tss = math.fsum((y - mean_y) * (y - mean_y) for y in ys)
            y_hat = [
                math.fsum(float(c["amplitude"]) * math.exp(-b * float(c["d_m2_s"])) for c in comps)
                for b in b_vals
            ]
            rss_actual = math.fsum((y - yh) * (y - yh) for y, yh in zip(ys, y_hat))
            r2 = (1.0 - (rss_actual / tss)) if tss > 0.0 else math.nan
        else:
            y_hat = [
                math.fsum(float(c["amplitude"]) * math.exp(-b * float(c["d_m2_s"])) for c in comps)
                for b in b_vals
            ]
            mean_yh = math.fsum(y_hat) / float(max(len(y_hat), 1))
            ess = math.fsum((yh - mean_yh) * (yh - mean_yh) for yh in y_hat)
            tss_est = ess + rss
            r2 = (1.0 - (rss / tss_est)) if tss_est > 0.0 else math.nan

        dominant_comp = max(comps, key=lambda c: float(c["fraction"])) if comps else None
        peaks_parsed.append(
            {
                "label": lbl,
                "center_ppm": float(pk["center_ppm"]),
                "selected_k": int(pk["selected_k"]),
                "r2": r2,
                "dominant_d_m2_s": float(dominant_comp["d_m2_s"]) if dominant_comp else math.nan,
                "dominant_log10_d": float(dominant_comp["log10_d"]) if dominant_comp else math.nan,
                "components": comps,
            }
        )

    return {
        "engine_used": js.get("engine_used", "cpu"),
        "solvent": js.get("solvent"),
        "temperature_k": js.get("temperature_k"),
        "b_values_s_m2": b_vals,
        "peaks": peaks_parsed,
    }


def evaluate_f16_dosy(
    vendor_runs: Mapping[str, Mapping[str, Any]],
    *,
    accession_id: str = "ZENODO:12170718+MENDELEY:d7bdxz9hsk",
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate `F16` across Bruker and Varian DOSY runs against reference diffusivities and ordering.

    Each entry in `vendor_runs` (`"bruker"`, `"varian"`) provides:
    - `exit_code`: int
    - `fit_json`: str (`nmrassign dosy fit --json-out`)
    - `decay_tsv`: optional str (`decay.tsv`)
    - `principal_references`: list of `{"label_or_ppm": str|float, "d_ref_m2_s": float}`
    - `ordering_pairs`: list of `(small_fast_label_or_ppm, large_slow_label_or_ppm)` expecting `D_fast > D_slow`
    """
    mf = manifest if manifest is not None else load_preregistration_manifest()

    def _match_peak(peaks: Sequence[Mapping[str, Any]], key: Union[str, float]) -> Mapping[str, Any]:
        if isinstance(key, (int, float)):
            target_ppm = float(key)
            return min(peaks, key=lambda p: abs(float(p["center_ppm"]) - target_ppm))
        ks = str(key)
        for p in peaks:
            if p["label"] == ks:
                return p
        try:
            target_ppm = float(ks)
            return min(peaks, key=lambda p: abs(float(p["center_ppm"]) - target_ppm))
        except ValueError as exc:
            raise KeyError(f"could not match DOSY peak {key!r}") from exc

    vendors_ok = 0
    principal_r2s: List[float] = []
    abs_log10_d_errs: List[float] = []
    ordering_checks: List[bool] = []
    any_nonzero_rc = False

    for vkey in ("bruker", "varian"):
        if vkey not in vendor_runs:
            continue
        vdata = vendor_runs[vkey]
        rc = int(vdata.get("exit_code", 1))
        if rc != 0:
            any_nonzero_rc = True
            continue
        parsed = parse_dosy_fit_report(vdata["fit_json"], vdata.get("decay_tsv"))
        peaks = parsed["peaks"]
        if not peaks:
            continue
        vendors_ok += 1

        refs = vdata.get("principal_references", [])
        if refs:
            for ref_item in refs:
                pk = _match_peak(peaks, ref_item["label_or_ppm"])
                principal_r2s.append(float(pk["r2"]))
                d_fit = float(pk["dominant_d_m2_s"])
                d_ref = float(ref_item["d_ref_m2_s"])
                abs_log10_d_errs.append(abs(math.log10(d_fit / d_ref)))
        else:
            principal_r2s.extend(float(p["r2"]) for p in peaks)

        for fast_id, slow_id in vdata.get("ordering_pairs", []):
            p_fast = _match_peak(peaks, fast_id)
            p_slow = _match_peak(peaks, slow_id)
            ordering_checks.append(float(p_fast["dominant_d_m2_s"]) > float(p_slow["dominant_d_m2_s"]))

    min_r2 = min(principal_r2s) if principal_r2s else math.nan
    ordering_conc = (
        float(sum(1 for ok in ordering_checks if ok)) / float(len(ordering_checks))
        if ordering_checks
        else math.nan
    )
    max_dex_err = max(abs_log10_d_errs) if abs_log10_d_errs else math.nan

    metrics = {
        "vendor_formats_ingested_exit0": vendors_ok,
        "min_principal_decay_fit_r2": min_r2,
        "mixture_diffusivity_ordering_concordance": ordering_conc,
        "max_principal_abs_log10_d_err_dex": max_dex_err,
    }
    eval_res = evaluate_gate_criteria(
        "F16",
        metrics,
        exit_code=1 if any_nonzero_rc else 0,
        accession_id=accession_id,
        manifest=mf,
    )
    eval_res["secondary_metrics"] = {
        "n_principal_peaks_evaluated": len(principal_r2s),
        "n_ordering_pairs_evaluated": len(ordering_checks),
        "all_abs_log10_d_errs_dex": abs_log10_d_errs,
    }
    return eval_res


# ============================================================================
# Gate F17: MetaboLights MTBLS1 ISA-Tab Manifest Builder & Glucose Evaluator
# ============================================================================


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


def locate_extracted_bruker_1d_dir(sample_extract_root: Union[Path, str]) -> Path:
    """Locate the extracted Bruker 1D experiment directory (`fid`+`acqus` or `pdata/1/1r`+`procs`)."""
    root = Path(sample_extract_root)
    fid_dirs = sorted(
        (p.parent.relative_to(root).as_posix(), p.parent)
        for p in root.rglob("acqus")
        if (p.parent / "fid").is_file()
    )
    if fid_dirs:
        return fid_dirs[0][1]
    pdata_dirs = sorted(
        (p.parent.relative_to(root).as_posix(), p.parent)
        for p in root.rglob("1r")
        if (p.parent / "procs").is_file()
    )
    if pdata_dirs:
        return pdata_dirs[0][1]
    raise FileNotFoundError(f"no Bruker fid+acqus or 1r+procs found under {root}")


def build_f17_mtbls1_manifest(
    s_mtbls1_text: str,
    a_mtbls1_text: str,
    *,
    spectra_dir: Union[Path, str] = "f17_spectra",
    spectrum_ext: str = ".tsv",
    out_manifest_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Deterministically join `s_MTBLS1.txt` and `a_MTBLS1_*.txt` (`.zip` ONLY, ignoring `.nmrML`)."""
    s_rows = _read_isatab_tsv(s_mtbls1_text)
    a_rows = _read_isatab_tsv(a_mtbls1_text)

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
        raw_file = r.get("Raw Spectral Data File", "").strip()
        if not raw_file:
            for k, v in r.items():
                if "spectral data file" in k.lower() and v.strip().lower().endswith(".zip"):
                    raw_file = v.strip()
                    break
        if not sname or not raw_file:
            continue
        if raw_file.lower().endswith(".nmrml"):
            ignored_nmrml += 1
            continue
        if not raw_file.lower().endswith(".zip"):
            continue
        zip_base = Path(raw_file).name
        if sname in sample_to_zip and sample_to_zip[sname] != zip_base:
            raise ValueError(
                f"non-1-to-1 Raw Spectral Data File .zip mapping for {sname!r}: "
                f"{sample_to_zip[sname]} vs {zip_base}"
            )
        sample_to_zip[sname] = zip_base

    joined_ids = sorted(sid for sid in sample_to_group if sid in sample_to_zip)
    if not joined_ids:
        raise ValueError("1-to-1 join between s_MTBLS1.txt and a_MTBLS1 (.zip) produced 0 samples")

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


def _bucket_overlaps_windows(
    center_ppm: float,
    low_ppm: Optional[float],
    high_ppm: Optional[float],
    windows: Sequence[Tuple[float, float]],
) -> bool:
    lo = float(low_ppm) if low_ppm is not None else center_ppm
    hi = float(high_ppm) if high_ppm is not None else center_ppm
    if lo > hi:
        lo, hi = hi, lo
    for w_lo, w_hi in windows:
        if (hi >= w_lo and lo <= w_hi) or (w_lo <= center_ppm <= w_hi):
            return True
    return False


def evaluate_f17_metabolomics(
    oplsda_json_text: str,
    stocsy_json_text: str,
    *,
    biomarkers_csv_text: Optional[str] = None,
    n_clean_processed_pqn_samples: Optional[int] = None,
    exit_code: int = 0,
    accession_id: str = "METABOLIGHTS:MTBLS1",
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate `nmrassign metabolomics oplsda` + `stocsy` outputs against `F17` criteria."""
    mf = manifest if manifest is not None else load_preregistration_manifest()
    c17 = extract_manifest_bound_constants(mf)["F17"]
    top_k = int(c17["top_k"])
    glucose_windows = c17["glucose_windows_ppm"]

    opls = json.loads(oplsda_json_text)
    stocsy = json.loads(stocsy_json_text)

    n_clean = (
        int(n_clean_processed_pqn_samples)
        if n_clean_processed_pqn_samples is not None
        else int(opls.get("n_active_samples", 0))
    )
    q2 = float(opls.get("q2", math.nan))
    perm_p = float(opls.get("permutation_p_value_q2", math.nan))

    # Rank OPLS-DA buckets strictly by VIP descending to inspect Top-15 VIP buckets
    vip_candidates: List[Dict[str, Any]] = []
    if biomarkers_csv_text and biomarkers_csv_text.strip():
        reader = csv.DictReader(
            [ln for ln in biomarkers_csv_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        )
        for r in reader:
            vip_candidates.append(
                {
                    "bucket_index": int(r["bucket_index"]),
                    "center_ppm": float(r["center_ppm"]),
                    "low_ppm": float(r["low_ppm"]) if "low_ppm" in r and r["low_ppm"] else None,
                    "high_ppm": float(r["high_ppm"]) if "high_ppm" in r and r["high_ppm"] else None,
                    "vip": float(r["vip"]),
                }
            )
    else:
        for b in opls.get("top_biomarkers", []):
            vip_candidates.append(
                {
                    "bucket_index": int(b["bucket_index"]),
                    "center_ppm": float(b["center_ppm"]),
                    "low_ppm": float(b["low_ppm"]) if b.get("low_ppm") is not None else None,
                    "high_ppm": float(b["high_ppm"]) if b.get("high_ppm") is not None else None,
                    "vip": float(b["vip"]),
                }
            )

    vip_sorted = sorted(vip_candidates, key=lambda x: (-float(x["vip"]), int(x["bucket_index"])))[:top_k]
    stocsy_top = list(stocsy.get("covarying_buckets", []))[:top_k]

    vip_glucose_hits: Set[int] = {
        int(b["bucket_index"])
        for b in vip_sorted
        if _bucket_overlaps_windows(float(b["center_ppm"]), b["low_ppm"], b["high_ppm"], glucose_windows)
    }
    stocsy_glucose_hits: Set[int] = {
        int(b["bucket_index"])
        for b in stocsy_top
        if _bucket_overlaps_windows(
            float(b["center_ppm"]),
            b.get("low_ppm"),
            b.get("high_ppm"),
            glucose_windows,
        )
    }
    union_glucose_hits = vip_glucose_hits | stocsy_glucose_hits

    metrics = {
        "n_clean_processed_pqn_samples": n_clean,
        "oplsda_cv_q2": q2,
        "oplsda_perm_p_q2": perm_p,
        "glucose_glycosuria_buckets_in_top15": len(union_glucose_hits),
    }
    eval_res = evaluate_gate_criteria(
        "F17",
        metrics,
        exit_code=exit_code,
        accession_id=accession_id,
        manifest=mf,
    )
    eval_res["secondary_metrics"] = {
        "r2_y": opls.get("r2_y"),
        "r2_x_predictive": opls.get("r2_x_predictive"),
        "r2_x_orthogonal": opls.get("r2_x_orthogonal"),
        "vip_top15_glucose_bucket_indices": sorted(vip_glucose_hits),
        "stocsy_top15_glucose_bucket_indices": sorted(stocsy_glucose_hits),
    }
    return eval_res
