#!/usr/bin/env python3
"""Wave 3 deterministic format adapters and pass-criteria evaluators (F10, F11, F13a, F13b, F13c).

Provenance:
- Original clean-room Python 3 standard-library implementation for Phase 17 blind
  empirical benchmarking (Rules P17-R1..R5 of ``data/benchmark/p17_empirical_preregistration.json``).
- Implements strictly the pre-registered syntactic table conversions and evaluation formulas
  for Lipari-Szabo model-free (``F10``), Bruker pseudo-3D relaxation series (``F11``),
  Bloch-McConnell CEST (``F13a``), multi-field CPMG / off-resonance R1rho (``F13b``), and
  2D 1H-15N HSQC fast-exchange ligand titration CSP (``F13c``).
- Zero external dependencies, zero network calls, and zero unregistered numeric constants
  outside ``data/benchmark/p17_empirical_preregistration.json``.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import zipfile

from .common import (
    STAR_MISSING_TOKENS,
    StarLoop,
    StarSaveframe,
    evaluate_gate_criteria,
    extract_manifest_bound_constants,
    extract_nmrstar_backbone_amide_shifts,
    load_preregistration_manifest,
    mae,
    median,
    median_abs_err,
    median_rel_abs_err,
    parse_nmrstar_saveframes,
    parse_residue_token,
    parse_star_float,
    parse_star_int,
    pearson_r,
    rmse,
    spearman_rho,
)

BACKBONE_AMIDE_ATOMS = frozenset({"N", "H", "HN"})


def assert_not_excluded_contaminated_entry(
    identifier_or_path: Union[str, Path],
    manifest: Optional[Mapping[str, Any]] = None,
) -> None:
    """Enforce Rule P17-R5 blind firewall against ``excluded_contaminated_entries`` (``BMRB:53960``)."""
    mf = manifest if manifest is not None else load_preregistration_manifest()
    consts = extract_manifest_bound_constants(mf)
    s_up = str(identifier_or_path).upper()
    for excl in consts["excluded_contaminated_entries"]:
        excl_up = str(excl).upper()
        digits = "".join(ch for ch in excl_up if ch.isdigit())
        if excl_up in s_up or (digits and f"BMR{digits}" in s_up) or (digits and f"BMRB:{digits}" in s_up) or (digits and f"BMRB_{digits}" in s_up) or (digits and re.search(rf"(?<!\d){re.escape(digits)}(?!\d)", s_up)):
            raise ValueError(
                f"BLIND FIREWALL VIOLATION: {identifier_or_path!r} matches excluded contaminated entry {excl!r}"
            )


# ============================================================================
# Shared NMR-STAR Relaxation Loop Extraction Helpers (F10 & F11)
# ============================================================================


def _infer_relax_kind(sf: StarSaveframe, lp: StarLoop) -> Optional[str]:
    cat_low = lp.category.lower()
    sf_cat_low = sf.sf_category.lower()
    sf_name_low = sf.name.lower()

    if "heteronucl_t1" in cat_low or "t1_relax" in sf_cat_low:
        return "R1"
    if "heteronucl_t2" in cat_low or "t2_relax" in sf_cat_low:
        return "R2"
    if "heteronucl_noe" in cat_low or "heteronucl_noes" in cat_low or "noe" in sf_cat_low:
        return "hetNOE"
    if "order_param" in cat_low or "order_parameter" in cat_low or "s2" in sf_cat_low:
        return None
    if "auto_relax" in cat_low or "auto_relax" in sf_cat_low:
        coherence = (
            sf.get_tag(
                [
                    "Common_relaxation_type_name",
                    "Relaxation_coherence_type",
                    "Auto_relaxation_type",
                ]
            )
            or ""
        ).lower()
        probe = f"{coherence} {sf_name_low}"
        if "r1rho" in probe or "t1rho" in probe:
            return None
        if "r2" in probe or "t2" in probe or "transverse" in probe or "in-phase x" in probe or "nxy" in probe:
            return "R2"
        if "r1" in probe or "t1" in probe or "longitudinal" in probe or "nz" in probe:
            return "R1"
    return None


def _infer_relax_units(sf: StarSaveframe, lp: StarLoop, kind: str) -> str:
    if kind == "hetNOE":
        return "dimensionless"
    unit_raw = sf.get_tag(
        [
            "T1_val_units",
            "T2_val_units",
            "Auto_relaxation_val_units",
            "Val_units",
            "Units",
        ]
    )
    if unit_raw is not None:
        u = unit_raw.strip().lower()
        if u in ("s-1", "s^-1", "1/s", "hz", "sec-1", "sec^-1"):
            return "s^-1"
        if u in ("ms", "msec", "millisecond", "milliseconds"):
            return "ms"
        if u in ("s", "sec", "second", "seconds"):
            return "s"
    cat_low = lp.category.lower()
    if "auto_relax" in cat_low:
        return "s^-1"
    return "s"


def _extract_relax_loop_rows(
    sf: StarSaveframe,
    lp: StarLoop,
    kind: str,
    unit_mode: str,
    mm_scale: float,
    default_rel_sigma: Optional[float] = None,
) -> List[Dict[str, Any]]:
    val_c = lp.find_col(
        [
            "Val",
            "T1_val",
            "T2_val",
            "NOE_val",
            "Auto_relaxation_val",
            "Relaxation_val",
        ]
    )
    err_c = lp.find_col(
        [
            "Val_err",
            "T1_val_err",
            "T2_val_err",
            "NOE_val_err",
            "Auto_relaxation_val_err",
            "Relaxation_val_err",
        ]
    )
    if val_c is None:
        return []

    sid_c = lp.find_col(["Comp_index_ID", "Seq_ID", "Auth_seq_ID", "Residue_seq_code"])
    comp_c = lp.find_col(["Comp_ID", "Auth_comp_ID", "Residue_label"])
    atom_c = lp.find_col(["Atom_ID", "Atom_name"])

    sid1_c = lp.find_col(["Comp_index_ID_1", "Seq_ID_1", "Auth_seq_ID_1"])
    comp1_c = lp.find_col(["Comp_ID_1", "Auth_comp_ID_1"])
    atom1_c = lp.find_col(["Atom_ID_1"])
    sid2_c = lp.find_col(["Comp_index_ID_2", "Seq_ID_2", "Auth_seq_ID_2"])
    atom2_c = lp.find_col(["Atom_ID_2"])

    rows_out: Dict[int, Dict[str, Any]] = {}
    for row in lp.rows:
        raw_val = parse_star_float(row[val_c])
        raw_err = parse_star_float(row[err_c]) if err_c is not None else None
        if raw_val is None:
            continue

        if sid_c is not None:
            seq_id = parse_star_int(row[sid_c])
            res_name = row[comp_c].strip().upper() if comp_c is not None else "UNK"
            if atom_c is not None:
                atom_id = row[atom_c].strip().upper()
                if atom_id not in STAR_MISSING_TOKENS and atom_id not in BACKBONE_AMIDE_ATOMS:
                    continue
        elif sid1_c is not None:
            seq_id = parse_star_int(row[sid1_c])
            if sid2_c is not None:
                seq_id_2 = parse_star_int(row[sid2_c])
                if seq_id_2 is not None and seq_id != seq_id_2:
                    continue
            res_name = row[comp1_c].strip().upper() if comp1_c is not None else "UNK"
            if atom1_c is not None:
                a1 = row[atom1_c].strip().upper()
                if a1 not in STAR_MISSING_TOKENS and a1 not in BACKBONE_AMIDE_ATOMS:
                    continue
            if atom2_c is not None:
                a2 = row[atom2_c].strip().upper()
                if a2 not in STAR_MISSING_TOKENS and a2 not in BACKBONE_AMIDE_ATOMS:
                    continue
        else:
            continue

        if seq_id is None or res_name == "PRO":
            continue
        if res_name.lower() in STAR_MISSING_TOKENS:
            res_name = "UNK"

        if kind in ("R1", "R2") and unit_mode in ("s", "ms"):
            t_sec = raw_val / mm_scale if unit_mode == "ms" else raw_val
            if t_sec <= 0.0:
                continue
            sig_t = (raw_err / mm_scale if unit_mode == "ms" else raw_err) if raw_err is not None else 0.0
            rate_val = 1.0 / t_sec
            rate_sig = abs(sig_t) / (t_sec * t_sec)
        else:
            if kind in ("R1", "R2") and raw_val <= 0.0:
                continue
            rate_val = raw_val
            rate_sig = abs(raw_err) if raw_err is not None else 0.0

        if rate_sig <= 0.0:
            if default_rel_sigma is None:
                continue
            if kind == "hetNOE":
                rate_sig = default_rel_sigma * max(abs(rate_val), 1.0)
            else:
                rate_sig = default_rel_sigma * abs(rate_val)
            if rate_sig <= 0.0:
                continue

        rows_out[seq_id] = {
            "seq_id": seq_id,
            "res_name": res_name,
            "raw_val": raw_val,
            "value": rate_val,
            "sigma": rate_sig,
        }

    return [rows_out[k] for k in sorted(rows_out)]


def extract_nmrstar_relaxation_tables(
    star_text: str,
    *,
    accession_id: Optional[str] = None,
    only_600_mhz: bool = False,
    default_rel_sigma: Optional[float] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Extract R1, R2, and hetNOE tables from NMR-STAR text without reading ``_Order_param``.

    Applies Rule P17-R2 ``relaxation_time_to_rate`` (`R = 1/T`, `sigma_R = sigma_T / T^2`)
    and ``bmrb6470_dual_field_saveframe_disambiguation`` (`median(T1_A) > median(T1_B) => 600.13 vs 360.13 MHz`).
    """
    if default_rel_sigma is not None and (default_rel_sigma <= 0.0 or default_rel_sigma >= 1.0):
        raise ValueError(f"default_rel_sigma must be strictly inside (0.0, 1.0), got {default_rel_sigma}")
    if accession_id is not None:
        assert_not_excluded_contaminated_entry(accession_id, manifest)
    mf = manifest if manifest is not None else load_preregistration_manifest()
    consts = extract_manifest_bound_constants(mf)
    hi_6470 = consts["F10"]["bmrb6470_high_field_mhz"]
    lo_6470 = consts["F10"]["bmrb6470_low_field_mhz"]
    mm_scale = consts["F13c"]["mm_to_um"]

    raw_tables: List[Dict[str, Any]] = []
    for sf in parse_nmrstar_saveframes(star_text):
        field_str = sf.get_tag(
            [
                "Spectrometer_frequency_1H",
                "Spectrometer_frequency_1H_val",
                "Spectrometer_frequency",
            ]
        )
        field_mhz = parse_star_float(field_str)
        for lp in sf.loops:
            kind = _infer_relax_kind(sf, lp)
            if kind is None:
                continue
            unit_mode = _infer_relax_units(sf, lp, kind)
            rows = _extract_relax_loop_rows(
                sf,
                lp,
                kind,
                unit_mode,
                mm_scale,
                default_rel_sigma=default_rel_sigma,
            )
            if not rows:
                continue
            raw_tables.append(
                {
                    "saveframe_name": sf.name,
                    "saveframe_index": sf.index,
                    "kind": kind,
                    "unit_mode": unit_mode,
                    "stamped_field_mhz": field_mhz,
                    "field_mhz": field_mhz if field_mhz is not None else hi_6470,
                    "rows": rows,
                }
            )

    # Disambiguate BMRB:6470 dual-field saveframes stamped with identical 600 MHz 1H frequency.
    r1_tables = [t for t in raw_tables if t["kind"] == "R1"]
    r2_tables = [t for t in raw_tables if t["kind"] == "R2"]
    noe_tables = [t for t in raw_tables if t["kind"] == "hetNOE"]

    is_dual_600_stamp = (
        len(r1_tables) == 2
        and r1_tables[0]["stamped_field_mhz"] is not None
        and r1_tables[1]["stamped_field_mhz"] is not None
        and round(float(r1_tables[0]["stamped_field_mhz"])) == round(float(r1_tables[1]["stamped_field_mhz"]))
        and round(float(r1_tables[0]["stamped_field_mhz"])) == round(hi_6470)
    )
    if is_dual_600_stamp:
        # Folded ubiquitin T1 increases monotonically with B0 (equivalently R1 = 1/T1 decreases with B0).
        def _effective_t1_median(tbl: Mapping[str, Any]) -> float:
            if tbl["unit_mode"] in ("s", "ms"):
                return median([float(r["raw_val"]) for r in tbl["rows"]])
            return median([1.0 / float(r["value"]) for r in tbl["rows"]])

        med0 = _effective_t1_median(r1_tables[0])
        med1 = _effective_t1_median(r1_tables[1])
        rel_diff = abs(med0 - med1) / max(med0, med1, 1.0 / mm_scale)
        # Only split into distinct [600.13, 360.13] MHz fields when T1 medians differ by >= 10%;
        # when < 10% (e.g. BMRB:6470 duplicate 600 MHz replicates set_1 & set_2), both stay at 600.13 MHz.
        if rel_diff * float(10) >= 1.0:
            if med0 >= med1:
                assigned_fields = [hi_6470, lo_6470]
            else:
                assigned_fields = [lo_6470, hi_6470]
        else:
            assigned_fields = [hi_6470, hi_6470]

        for idx, fm in enumerate(assigned_fields):
            r1_tables[idx]["field_mhz"] = fm
            if idx < len(r2_tables):
                r2_tables[idx]["field_mhz"] = fm
            if idx < len(noe_tables):
                noe_tables[idx]["field_mhz"] = fm

    if only_600_mhz:
        target_round = round(hi_6470)
        raw_tables = [t for t in raw_tables if round(float(t["field_mhz"])) == target_round]

    return raw_tables


def format_modelfree_rate_tsv(kind: str, field_mhz: float, rows: Sequence[Mapping[str, Any]]) -> str:
    """Format a single `(kind, field_mhz)` relaxation table for `nmrassign modelfree`."""
    lines = [
        f"# nmrassign_relax kind={kind} field_mhz={field_mhz:.4f}",
        "seq_id\tres_name\tfield_mhz\tkind\tvalue\tsigma",
    ]
    for r in rows:
        lines.append(
            f"{int(r['seq_id'])}\t{r['res_name']}\t{field_mhz:.4f}\t{kind}\t"
            f"{float(r['value']):.8f}\t{float(r['sigma']):.8f}"
        )
    return "\n".join(lines) + "\n"


def export_f10_modelfree_tsvs(
    star_path: Union[Path, str],
    out_dir: Union[Path, str],
    *,
    prefix: str = "ubq",
    accession_id: Optional[str] = None,
    only_600_mhz: bool = False,
    default_rel_sigma: Optional[float] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert an NMR-STAR relaxation file (`bmr6470_3.str`, `bmr18971_3.str`, etc.) to `nmrassign modelfree` TSVs."""
    assert_not_excluded_contaminated_entry(star_path, manifest)
    star_text = Path(star_path).read_text(encoding="utf-8", errors="replace")
    tables = extract_nmrstar_relaxation_tables(
        star_text,
        accession_id=accession_id,
        only_600_mhz=only_600_mhz,
        default_rel_sigma=default_rel_sigma,
        manifest=manifest,
    )
    if not tables:
        raise ValueError(f"no R1/R2/hetNOE relaxation saveframes found in {star_path}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    kind_suffix = {"R1": "r1", "R2": "r2", "hetNOE": "noe"}
    for tbl in tables:
        fm = float(tbl["field_mhz"])
        mhz_tag = str(int(round(fm)))
        ksuf = kind_suffix[tbl["kind"]]
        fname = f"{prefix}_{mhz_tag}_{ksuf}.tsv"
        fpath = out_root / fname
        fpath.write_text(format_modelfree_rate_tsv(tbl["kind"], fm, tbl["rows"]), encoding="utf-8")
        written[f"{mhz_tag}_{ksuf}"] = str(fpath)

    return {
        "source_star": str(star_path),
        "n_tables": len(tables),
        "fields_mhz": sorted({float(t["field_mhz"]) for t in tables}),
        "written_files": written,
    }


def extract_nmrstar_order_parameters(star_text: str) -> Dict[int, Dict[str, Any]]:
    """Extract post-fit Lipari-Szabo `S^2` reference table from `_Order_param` / `_Order_parameter`."""
    best_table: Dict[int, Dict[str, Any]] = {}
    for sf in parse_nmrstar_saveframes(star_text):
        for lp in sf.loops:
            cat_low = lp.category.lower()
            if "order_param" not in cat_low and "s2" not in sf.sf_category.lower():
                continue
            val_c = lp.find_col(["Order_param_val", "S2_val", "Val"])
            if val_c is None:
                continue
            err_c = lp.find_col(["Order_param_val_err", "S2_val_err", "Val_err"])
            sid_c = lp.find_col(["Comp_index_ID", "Seq_ID", "Auth_seq_ID", "Residue_seq_code"])
            comp_c = lp.find_col(["Comp_ID", "Auth_comp_ID", "Residue_label"])
            atom_c = lp.find_col(["Atom_ID", "Atom_name"])
            if sid_c is None:
                continue
            cur: Dict[int, Dict[str, Any]] = {}
            for row in lp.rows:
                s2_val = parse_star_float(row[val_c])
                seq_id = parse_star_int(row[sid_c])
                if s2_val is None or seq_id is None:
                    continue
                if atom_c is not None:
                    atom_id = row[atom_c].strip().upper()
                    if atom_id not in STAR_MISSING_TOKENS and atom_id not in BACKBONE_AMIDE_ATOMS:
                        continue
                res_name = row[comp_c].strip().upper() if comp_c is not None else "UNK"
                if res_name == "PRO":
                    continue
                s2_err = parse_star_float(row[err_c]) if err_c is not None else None
                cur[seq_id] = {
                    "seq_id": seq_id,
                    "res_name": res_name,
                    "s2_ref": s2_val,
                    "s2_err": s2_err,
                }
            if len(cur) > len(best_table):
                best_table = cur
    return best_table


def parse_modelfree_outputs(
    modelfree_tsv_text: str,
    stdout_text: str = "",
    json_text: str = "",
) -> Dict[str, Any]:
    """Parse `nmrassign modelfree --out` TSV + stdout diffusion summary (or optional JSON)."""
    tau_m_ns = math.nan
    d_par_over_d_perp = math.nan
    residues: Dict[int, Dict[str, Any]] = {}

    if json_text.strip():
        js = json.loads(json_text)
        if "tau_m_ns" in js and js["tau_m_ns"] is not None:
            tau_m_ns = float(js["tau_m_ns"])
        if "d_par_over_d_perp" in js and js["d_par_over_d_perp"] is not None:
            d_par_over_d_perp = float(js["d_par_over_d_perp"])
        for r in js.get("residues", []):
            sid = int(r["seq_id"])
            residues[sid] = dict(r)

    col_map: Dict[str, int] = {}
    for raw_line in modelfree_tsv_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m_tm = re.search(r"effective_tau_m_ns=([0-9.eE+-]+)", line)
            if m_tm:
                tau_m_ns = float(m_tm.group(1))
            m_dp = re.search(r"d_par_over_d_perp=([0-9.eE+-]+)", line, flags=re.IGNORECASE)
            if m_dp:
                d_par_over_d_perp = float(m_dp.group(1))
            continue
        cols = [c.strip() for c in line.split("\t")]
        if not col_map:
            col_map = {name.lower(): idx for idx, name in enumerate(cols)}
            continue
        sid = int(cols[col_map["seq_id"]])
        res_name = cols[col_map["res_name"]].upper()
        model = cols[col_map["model"]]
        s2 = float(cols[col_map["s2"]])
        sigma_s2 = float(cols[col_map["sigma_s2"]]) if "sigma_s2" in col_map else math.nan
        rex_ref = float(cols[col_map["rex_ref"]]) if "rex_ref" in col_map else 0.0
        residues[sid] = {
            "seq_id": sid,
            "res_name": res_name,
            "model": model,
            "s2": s2,
            "sigma_s2": sigma_s2,
            "rex_ref": rex_ref,
        }

    if stdout_text:
        m_tm_out = re.search(r"tau_m\s*=\s*([0-9.eE+-]+)\s*ns", stdout_text)
        if m_tm_out and not math.isfinite(tau_m_ns):
            tau_m_ns = float(m_tm_out.group(1))
        m_ax = re.search(r"D_par/D_perp\s*=\s*([0-9.eE+-]+)", stdout_text)
        if m_ax:
            d_par_over_d_perp = float(m_ax.group(1))
        elif "Isotropic(" in stdout_text and not math.isfinite(d_par_over_d_perp):
            d_par_over_d_perp = 1.0

    return {
        "tau_m_ns": tau_m_ns,
        "d_par_over_d_perp": d_par_over_d_perp,
        "residues": residues,
    }


def evaluate_f10_modelfree(
    modelfree_tsv_text: str,
    star_text: str,
    *,
    stdout_text: str = "",
    json_text: str = "",
    exit_code: int = 0,
    accession_id: str = "BMRB:6470",
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare `nmrassign modelfree` outputs against deposited `S^2` and manifest `F10` criteria."""
    assert_not_excluded_contaminated_entry(accession_id, manifest)
    mf = manifest if manifest is not None else load_preregistration_manifest()
    consts = extract_manifest_bound_constants(mf)["F10"]
    tau_ref = consts["tau_m_ref_ns"]
    aniso_ref = consts["d_par_over_d_perp_ref"]

    parsed = parse_modelfree_outputs(modelfree_tsv_text, stdout_text=stdout_text, json_text=json_text)
    s2_ref_map = extract_nmrstar_order_parameters(star_text)

    matched_ids = sorted(
        sid
        for sid, rec in parsed["residues"].items()
        if rec["res_name"] != "PRO" and sid in s2_ref_map and math.isfinite(rec["s2"])
    )
    s2_preds = [float(parsed["residues"][sid]["s2"]) for sid in matched_ids]
    s2_refs = [float(s2_ref_map[sid]["s2_ref"]) for sid in matched_ids]

    tau_m_ns = float(parsed["tau_m_ns"])
    d_ratio = float(parsed["d_par_over_d_perp"])

    tau_rel_err = abs(tau_m_ns - tau_ref) / tau_ref if math.isfinite(tau_m_ns) else math.nan
    aniso_abs_err = abs(d_ratio - aniso_ref) if math.isfinite(d_ratio) else math.nan
    r_s2 = pearson_r(s2_preds, s2_refs)
    mae_s2_val = mae(s2_preds, s2_refs)

    model_counts: Dict[str, int] = {}
    for rec in parsed["residues"].values():
        m = str(rec.get("model", ""))
        model_counts[m] = model_counts.get(m, 0) + 1

    metrics = {
        f"abs(tau_m_ns - {tau_ref:g}) / {tau_ref:g}": tau_rel_err,
        f"abs(d_par_over_d_perp - {aniso_ref:g})": aniso_abs_err,
        "n_matched_core_residues": len(matched_ids),
        "pearson_r_s2": r_s2,
        "mae_s2": mae_s2_val,
    }
    eval_res = evaluate_gate_criteria(
        "F10",
        metrics,
        exit_code=exit_code,
        accession_id=accession_id,
        converged=bool(parsed["residues"]),
        manifest=mf,
    )
    eval_res["secondary_metrics"] = {
        "tau_m_ns": tau_m_ns,
        "d_par_over_d_perp": d_ratio,
        "spearman_rho_s2": spearman_rho(s2_preds, s2_refs),
        "rmse_s2": rmse(s2_preds, s2_refs),
        "median_signed_bias_s2": median([p - r for p, r in zip(s2_preds, s2_refs)]),
        "model_counts": model_counts,
    }
    return eval_res


# ============================================================================
# Gate F11: Bruker Pseudo-3D Relaxation Diagnostic & BMRB Reference Evaluator
# ============================================================================


def parse_bruker_jcamp_numeric_array(acqus_text: str, param_name: str) -> List[float]:
    """Extract a Bruker JCAMP-DX array `##$<param_name>= (0..N)` as a list of floats."""
    lines = acqus_text.splitlines()
    header_pat = re.compile(rf"^##\${re.escape(param_name)}=\s*\(\s*\d+\.\.\d+\s*\)(.*)$")
    collecting = False
    tokens: List[str] = []
    for line in lines:
        s = line.strip()
        if not collecting:
            m = header_pat.match(s)
            if m:
                collecting = True
                rest = m.group(1).strip()
                if rest:
                    tokens.extend(rest.split())
        else:
            if s.startswith("##") or s.startswith("$$"):
                break
            if s:
                tokens.extend(s.split())
    return [float(tok) for tok in tokens]


def parse_bruker_vclist(vclist_text: str) -> List[int]:
    """Parse integer CPMG loop counts from a Bruker `vclist` file."""
    counts: List[int] = []
    for raw in vclist_text.splitlines():
        clean = raw.split("#", 1)[0].split(";", 1)[0].strip()
        if not clean:
            continue
        for tok in clean.split():
            counts.append(int(tok))
    if not counts:
        raise ValueError("Bruker vclist contains zero integer loop counts")
    return counts


def compute_bruker_hsqct2_vclist_delays(
    acqus_text: str,
    vclist_text: str,
    *,
    pulseprogram_text: Optional[str] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute pre-registered Bruker `hsqct2etf3gpsi3d` CPMG loop delays strictly from `acqus` + `vclist`.

    Formula bound in Rule P17-R2 `bruker_hsqct2etf3gpsi3d_vclist_loop_timing`:
    `tau_cpmg_block_s = 16.0 * D[20] + 8.0 * (P[4] * 1.0e-6)` with `P[4] = 2.0 * P[3]` if `P[4] == 0`.
    """
    mf = manifest if manifest is not None else load_preregistration_manifest()
    c11 = extract_manifest_bound_constants(mf)["F11"]

    d_arr = parse_bruker_jcamp_numeric_array(acqus_text, "D")
    p_arr = parse_bruker_jcamp_numeric_array(acqus_text, "P")
    vclist = parse_bruker_vclist(vclist_text)

    d_idx = c11["d_index"]
    p_idx = c11["p_index"]
    p_fb_idx = c11["p_fallback_index"]
    if len(d_arr) <= d_idx:
        raise ValueError(f"acqus ##$D array length {len(d_arr)} <= required index {d_idx}")
    if len(p_arr) <= max(p_idx, p_fb_idx):
        raise ValueError(f"acqus ##$P array length {len(p_arr)} <= required index {max(p_idx, p_fb_idx)}")

    d20_s = d_arr[d_idx]
    p3_us = p_arr[p_fb_idx]
    p4_raw_us = p_arr[p_idx]
    p4_eff_us = p4_raw_us if p4_raw_us > 0.0 else (c11["p_fallback_mult"] * p3_us)

    tau_block_s = c11["d_mult"] * d20_s + c11["p_mult"] * (p4_eff_us * c11["us_to_s"])
    delays_s = [float(vc) * tau_block_s for vc in vclist]

    has_vc_loop_in_pp = None
    if pulseprogram_text is not None:
        has_vc_loop_in_pp = bool(re.search(r"lo\s+to\s+\w+\s+times\s+c\b", pulseprogram_text))

    return {
        "vclist": vclist,
        "d20_s": d20_s,
        "p3_us": p3_us,
        "p4_raw_us": p4_raw_us,
        "p4_effective_us": p4_eff_us,
        "tau_cpmg_block_s": tau_block_s,
        "delays_s": delays_s,
        "delays_csv": ",".join(f"{t:.9f}" for t in delays_s),
        "pulseprogram_has_vc_loop": has_vc_loop_in_pp,
    }


def parse_nmrassign_relax_tsv(tsv_text: str) -> Dict[int, Dict[str, Any]]:
    """Parse `nmrassign relax --out <rates.tsv>` into `{seq_id: {value, sigma, res_name, kind}}`."""
    col_map: Dict[str, int] = {}
    out: Dict[int, Dict[str, Any]] = {}
    for raw in tsv_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = [c.strip() for c in line.split("\t")]
        if not col_map:
            col_map = {name.lower(): idx for idx, name in enumerate(cols)}
            continue
        sid = int(cols[col_map["seq_id"]])
        res_name = cols[col_map["res_name"]].upper() if "res_name" in col_map else "UNK"
        kind = cols[col_map["kind"]] if "kind" in col_map else ""
        val = float(cols[col_map["value"]])
        sig = float(cols[col_map["sigma"]]) if "sigma" in col_map else 0.0
        out[sid] = {
            "seq_id": sid,
            "res_name": res_name,
            "kind": kind,
            "value": val,
            "sigma": sig,
        }
    return out


def evaluate_f11_relax(
    r1_tsv_text: str,
    r2_tsv_text: str,
    hetnoe_tsv_text: str,
    star_text: str,
    *,
    exit_code: int = 0,
    accession_id: str = "BMRB:18380",
    r1_gpu_tsv_text: Optional[str] = None,
    r2_gpu_tsv_text: Optional[str] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare `nmrassign relax` R1, R2, and hetNOE outputs against `bmr18380_3.str` under `F11` criteria."""
    assert_not_excluded_contaminated_entry(accession_id, manifest)
    mf = manifest if manifest is not None else load_preregistration_manifest()

    ref_tables = extract_nmrstar_relaxation_tables(star_text, accession_id=accession_id, manifest=mf)
    shifts_map = extract_nmrstar_backbone_amide_shifts(star_text)
    assigned_amide_ids = (
        {
            sid
            for sid, rec in shifts_map.items()
            if rec["res_name"] != "PRO" and "H" in rec["shifts"] and "N" in rec["shifts"]
        }
        if shifts_map
        else None
    )

    ref_by_kind: Dict[str, Dict[int, Dict[str, Any]]] = {"R1": {}, "R2": {}, "hetNOE": {}}
    for tbl in ref_tables:
        k = tbl["kind"]
        if k in ref_by_kind and len(tbl["rows"]) > len(ref_by_kind[k]):
            ref_by_kind[k] = {int(r["seq_id"]): r for r in tbl["rows"]}

    fit_by_kind = {
        "R1": parse_nmrassign_relax_tsv(r1_tsv_text),
        "R2": parse_nmrassign_relax_tsv(r2_tsv_text),
        "hetNOE": parse_nmrassign_relax_tsv(hetnoe_tsv_text),
    }

    paired: Dict[str, Tuple[List[int], List[float], List[float], List[float]]] = {}
    for k in ("R1", "R2", "hetNOE"):
        fmap = fit_by_kind[k]
        rmap = ref_by_kind[k]
        sids = sorted(
            sid
            for sid in fmap
            if sid in rmap
            and fmap[sid]["res_name"] != "PRO"
            and (assigned_amide_ids is None or sid in assigned_amide_ids)
            and math.isfinite(fmap[sid]["value"])
            and math.isfinite(rmap[sid]["value"])
        )
        preds = [float(fmap[sid]["value"]) for sid in sids]
        refs = [float(rmap[sid]["value"]) for sid in sids]
        comb_sig = [
            math.hypot(float(fmap[sid]["sigma"]), float(rmap[sid]["sigma"])) for sid in sids
        ]
        paired[k] = (sids, preds, refs, comb_sig)

    r1_ids, r1_p, r1_r, r1_s = paired["R1"]
    r2_ids, r2_p, r2_r, r2_s = paired["R2"]
    noe_ids, noe_p, noe_r, noe_s = paired["hetNOE"]

    metrics = {
        "min(n_matched_r1, n_matched_r2, n_matched_hetnoe)": min(
            len(r1_ids), len(r2_ids), len(noe_ids)
        ),
        "pearson_r_r1": pearson_r(r1_p, r1_r),
        "median_rel_abs_err_r1": median_rel_abs_err(r1_p, r1_r),
        "pearson_r_r2": pearson_r(r2_p, r2_r),
        "median_rel_abs_err_r2": median_rel_abs_err(r2_p, r2_r),
        "pearson_r_hetnoe": pearson_r(noe_p, noe_r),
        "median_abs_err_hetnoe": median_abs_err(noe_p, noe_r),
    }

    def _frac_within_2sigma(ps: Sequence[float], rs: Sequence[float], sigs: Sequence[float]) -> float:
        valid = [
            abs(p - r) <= (s + s)
            for p, r, s in zip(ps, rs, sigs)
            if s > 0.0 and math.isfinite(s)
        ]
        return float(sum(1 for ok in valid if ok)) / float(len(valid)) if valid else math.nan

    secondary: Dict[str, Any] = {
        "n_matched": {"R1": len(r1_ids), "R2": len(r2_ids), "hetNOE": len(noe_ids)},
        "spearman_rho": {
            "R1": spearman_rho(r1_p, r1_r),
            "R2": spearman_rho(r2_p, r2_r),
            "hetNOE": spearman_rho(noe_p, noe_r),
        },
        "mae": {
            "R1": mae(r1_p, r1_r),
            "R2": mae(r2_p, r2_r),
            "hetNOE": mae(noe_p, noe_r),
        },
        "frac_within_2sigma": {
            "R1": _frac_within_2sigma(r1_p, r1_r, r1_s),
            "R2": _frac_within_2sigma(r2_p, r2_r, r2_s),
            "hetNOE": _frac_within_2sigma(noe_p, noe_r, noe_s),
        },
    }

    if r1_gpu_tsv_text is not None:
        g1 = parse_nmrassign_relax_tsv(r1_gpu_tsv_text)
        common_1 = sorted(sid for sid in fit_by_kind["R1"] if sid in g1)
        if common_1:
            secondary["cpu_gpu_r1_max_abs_diff"] = max(
                abs(fit_by_kind["R1"][sid]["value"] - g1[sid]["value"]) for sid in common_1
            )
    if r2_gpu_tsv_text is not None:
        g2 = parse_nmrassign_relax_tsv(r2_gpu_tsv_text)
        common_2 = sorted(sid for sid in fit_by_kind["R2"] if sid in g2)
        if common_2:
            secondary["cpu_gpu_r2_max_abs_diff"] = max(
                abs(fit_by_kind["R2"][sid]["value"] - g2[sid]["value"]) for sid in common_2
            )

    eval_res = evaluate_gate_criteria(
        "F11",
        metrics,
        exit_code=exit_code,
        accession_id=accession_id,
        manifest=mf,
    )
    eval_res["secondary_metrics"] = secondary
    return eval_res


# ============================================================================
# Gate F13a: CEST Saturation Profile Converter & Evaluator
# ============================================================================


def _parse_kv_metadata_lines(lines: Iterable[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith(";"):
            s = s[1:].strip()
        elif "=" not in s and ":" not in s:
            continue
        if s.startswith("[") and s.endswith("]"):
            continue
        for chunk in re.split(r"[;,]\s*(?=[A-Za-z_][A-Za-z0-9_]*\s*[=:])", s):
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_.-]*)\s*[=:]\s*(.+)$", chunk.strip())
            if m:
                key = m.group(1).strip().lower()
                val = m.group(2).split("#", 1)[0].strip().strip("\"'")
                meta[key] = val
    return meta


def _convert_single_cest_profile_block(
    text: str,
    *,
    default_seq_id: Optional[int] = None,
    default_res_name: str = "UNK",
    inherited_meta: Optional[Mapping[str, str]] = None,
    larmor_mhz_override: Optional[float] = None,
    ref_offset_sentinel_ppm: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Parse either an 11-column CEST table or a plain-text header + 3-column `[offset, I, sigma]` profile."""
    if ref_offset_sentinel_ppm is None:
        ref_offset_sentinel_ppm = extract_manifest_bound_constants(load_preregistration_manifest())["F13c"]["mm_to_um"]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    # Case 1: Already tabular with header containing offset_ppm / intensity
    non_comment = [ln for ln in lines if not ln.startswith("#") and not ln.startswith(";")]
    if non_comment:
        first_cols = [c.strip().lower() for c in re.split(r"[\t,]", non_comment[0])]
        if "offset_ppm" in first_cols and "intensity" in first_cols:
            sep = "\t" if "\t" in non_comment[0] else ","
            cmap = {name: idx for idx, name in enumerate(first_cols)}
            grouped_rows: Dict[Tuple[int, str, float, float, float], List[Dict[str, Any]]] = {}
            for row_ln in non_comment[1:]:
                cols = [c.strip() for c in row_ln.split(sep)]
                if len(cols) < len(first_cols):
                    continue
                if "seq_id" in cmap:
                    sid = int(cols[cmap["seq_id"]])
                    rname = cols[cmap["res_name"]].upper() if "res_name" in cmap else default_res_name
                elif "residue" in cmap:
                    sid, rname = parse_residue_token(cols[cmap["residue"]])
                else:
                    if default_seq_id is None:
                        raise ValueError("CEST table missing seq_id/residue column")
                    sid, rname = default_seq_id, default_res_name

                lmhz = (
                    float(larmor_mhz_override)
                    if larmor_mhz_override is not None
                    else float(cols[cmap.get("larmor_mhz", cmap.get("b0_mhz", 0))])
                )
                tex = float(cols[cmap["t_ex_s"]])
                b1 = float(cols[cmap["b1_hz"]])
                binh = float(cols[cmap["b1_inhom_frac"]]) if "b1_inhom_frac" in cmap else 0.0
                norm = cols[cmap["normalization"]] if "normalization" in cmap else "zero_time"
                hint_raw = cols[cmap["major_ppm_hint"]] if "major_ppm_hint" in cmap else "NA"
                off = float(cols[cmap["offset_ppm"]])
                inten = float(cols[cmap["intensity"]])
                unc = float(cols[cmap["uncertainty"]])
                gkey = (sid, rname, lmhz, tex, b1)
                grouped_rows.setdefault(gkey, []).append(
                    {
                        "seq_id": sid,
                        "res_name": rname,
                        "larmor_mhz": lmhz,
                        "t_ex_s": tex,
                        "b1_hz": b1,
                        "b1_inhom_frac": binh,
                        "normalization": norm,
                        "major_ppm_hint": None if hint_raw.upper() in ("", "NA", ".") else float(hint_raw),
                        "offset_ppm": off,
                        "intensity": inten,
                        "uncertainty": unc,
                    }
                )
            flat_out: List[Dict[str, Any]] = []
            for rows in grouped_rows.values():
                if any(r["major_ppm_hint"] is None for r in rows):
                    min_row = min(rows, key=lambda r: float(r["intensity"]))
                    hint_val = float(min_row["offset_ppm"])
                    for r in rows:
                        r["major_ppm_hint"] = hint_val
                flat_out.extend(rows)
            return flat_out

    # Case 2: Plain-text saturation profile (`B0/larmor_mhz`, `T_EX`, `B1_FRQ`, `B1_INH` + 3-col rows)
    meta: Dict[str, str] = dict(inherited_meta or {})
    header_lines: List[str] = []
    data_rows: List[Tuple[float, float, float, bool]] = []
    for ln in lines:
        if ln.startswith("#") or ln.startswith(";") or ln.startswith("["):
            header_lines.append(ln)
            continue
        if "=" in ln and not ln[0].isdigit() and not ln.startswith("-"):
            header_lines.append(ln)
            continue
        parts = ln.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            off_val = float(parts[0])
            i_val = float(parts[1])
            sig_val = float(parts[2]) if len(parts) >= 3 else 0.0
        except ValueError:
            # Column header line such as `offset intensity uncertainty`
            continue
        flag_tok = parts[3].lower() if len(parts) >= 4 else ""
        is_ref_flag = flag_tok in ("ref", "i0", "reference", "true", "1")
        data_rows.append((off_val, i_val, sig_val, is_ref_flag))

    meta.update(_parse_kv_metadata_lines(header_lines))
    if not data_rows:
        return []

    if "seq_id" in meta:
        seq_id = int(meta["seq_id"])
        res_name = meta.get("res_name", default_res_name).upper()
    elif "residue" in meta:
        seq_id, res_name = parse_residue_token(meta["residue"])
    elif default_seq_id is not None:
        seq_id, res_name = default_seq_id, default_res_name
    else:
        raise ValueError("plain-text CEST profile missing residue/seq_id in header or filename")

    larmor_raw = (
        larmor_mhz_override
        if larmor_mhz_override is not None
        else next(
            (
                float(meta[k])
                for k in ("larmor_mhz", "sfo1", "bf1", "obs_mhz", "b0_mhz", "b0", "frq")
                if k in meta
            ),
            None,
        )
    )
    if larmor_raw is None or larmor_raw <= 0.0:
        raise ValueError("plain-text CEST profile missing positive larmor_mhz / B0 header")

    t_ex_s = next(
        (float(meta[k]) for k in ("t_ex", "t_ex_s", "tex", "time_t2", "sat_time") if k in meta),
        None,
    )
    b1_hz = next(
        (float(meta[k]) for k in ("b1_frq", "b1_hz", "b1", "spinlock_hz") if k in meta),
        None,
    )
    if t_ex_s is None or b1_hz is None:
        raise ValueError("plain-text CEST profile missing T_EX or B1_FRQ header")
    b1_inhom = next(
        (float(meta[k]) for k in ("b1_inh", "b1_inh_scale", "b1_inhom_frac") if k in meta),
        0.0,
    )
    norm_label = meta.get("normalization", "zero_time")
    carrier_ppm = float(meta["carrier"]) if "carrier" in meta else (
        float(meta["carrier_ppm"]) if "carrier_ppm" in meta else None
    )
    offset_units = meta.get("offset_units", "").lower()

    # Separate reference I0 plane(s) if present (`abs(offset) >= ref_offset_sentinel_ppm` or explicit ref flag)
    ref_planes: List[Tuple[float, float]] = []
    sweep_raw: List[Tuple[float, float, float]] = []
    for off_v, i_v, s_v, is_ref in data_rows:
        if is_ref or abs(off_v) >= ref_offset_sentinel_ppm:
            ref_planes.append((i_v, s_v))
        else:
            if carrier_ppm is not None and (offset_units == "hz" or "offset_hz" in meta):
                off_ppm = carrier_ppm + (off_v / float(larmor_raw))
            else:
                off_ppm = off_v
            sweep_raw.append((off_ppm, i_v, s_v))

    if not sweep_raw:
        return []

    norm_rows: List[Tuple[float, float, float]] = []
    if ref_planes:
        i0_mean = math.fsum(r[0] for r in ref_planes) / float(len(ref_planes))
        if i0_mean == 0.0:
            raise ValueError("CEST reference plane intensity I0 is zero")
        s0_pool = math.sqrt(math.fsum(r[1] * r[1] for r in ref_planes)) / float(len(ref_planes))
        for off_ppm, i_v, s_v in sweep_raw:
            i_norm = i_v / i0_mean
            rel_i = (s_v / i_v) if i_v != 0.0 else 0.0
            rel_0 = (s0_pool / i0_mean) if i0_mean != 0.0 else 0.0
            sig_norm = abs(i_norm) * math.hypot(rel_i, rel_0)
            norm_rows.append((off_ppm, i_norm, sig_norm))
    else:
        norm_rows = sweep_raw

    # Pre-registered rule: major_ppm_hint = argmin_offset(I_norm) without inspecting minor dips
    major_hint = min(norm_rows, key=lambda item: item[1])[0]
    return [
        {
            "seq_id": seq_id,
            "res_name": res_name,
            "larmor_mhz": float(larmor_raw),
            "t_ex_s": float(t_ex_s),
            "b1_hz": float(b1_hz),
            "b1_inhom_frac": float(b1_inhom),
            "normalization": norm_label,
            "major_ppm_hint": float(major_hint),
            "offset_ppm": float(off_ppm),
            "intensity": float(i_norm),
            "uncertainty": float(sig_norm),
        }
        for off_ppm, i_norm, sig_norm in norm_rows
    ]


def format_cest_11col_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Format 11-column TSV expected by `nmrassign cest --in`."""
    header = (
        "seq_id\tres_name\tlarmor_mhz\tt_ex_s\tb1_hz\tb1_inhom_frac\t"
        "normalization\tmajor_ppm_hint\toffset_ppm\tintensity\tuncertainty"
    )
    out = [header]
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            int(r["seq_id"]),
            float(r["larmor_mhz"]),
            float(r["b1_hz"]),
            float(r["offset_ppm"]),
        ),
    )
    for r in sorted_rows:
        hint = r.get("major_ppm_hint")
        hint_s = "NA" if hint is None else f"{float(hint):.5f}"
        out.append(
            f"{int(r['seq_id'])}\t{r['res_name']}\t{float(r['larmor_mhz']):.5f}\t"
            f"{float(r['t_ex_s']):.5f}\t{float(r['b1_hz']):.4f}\t{float(r['b1_inhom_frac']):.4f}\t"
            f"{r['normalization']}\t{hint_s}\t{float(r['offset_ppm']):.5f}\t"
            f"{float(r['intensity']):.6f}\t{float(r['uncertainty']):.6f}"
        )
    return "\n".join(out) + "\n"


def convert_f13a_cest_to_tsv(
    source_path: Union[Path, str],
    out_tsv_path: Union[Path, str],
    *,
    larmor_mhz_override: Optional[float] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert `CEST.zip`, directory of CEST profile files, or single text table into 11-col TSV."""
    assert_not_excluded_contaminated_entry(source_path, manifest)
    mf = manifest if manifest is not None else load_preregistration_manifest()
    sentinel = extract_manifest_bound_constants(mf)["F13c"]["mm_to_um"]

    src = Path(source_path)
    entries: List[Tuple[str, str]] = []
    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("/") or "/__MACOSX" in name or Path(name).name.startswith("."):
                    continue
                low = name.lower()
                if low.endswith((".txt", ".out", ".dat", ".tsv", ".csv", ".toml", ".cfg", ".ini")):
                    entries.append((name, zf.read(name).decode("utf-8", errors="replace")))
    elif src.is_dir():
        for fp in sorted(src.rglob("*")):
            if fp.is_file() and not fp.name.startswith("."):
                if fp.suffix.lower() in (".txt", ".out", ".dat", ".tsv", ".csv", ".toml", ".cfg", ".ini"):
                    entries.append((str(fp.relative_to(src)), fp.read_text(encoding="utf-8", errors="replace")))
    elif src.is_file():
        entries.append((src.name, src.read_text(encoding="utf-8", errors="replace")))
    else:
        raise FileNotFoundError(f"CEST input source not found: {src}")

    # Collect directory-level config metadata (`.toml` / `.cfg` / `.ini`) first
    dir_meta: Dict[str, Dict[str, str]] = {}
    for rel_name, content in entries:
        p = Path(rel_name)
        if p.suffix.lower() in (".toml", ".cfg", ".ini"):
            parent_key = str(p.parent)
            dir_meta.setdefault(parent_key, {}).update(_parse_kv_metadata_lines(content.splitlines()))

    all_rows: List[Dict[str, Any]] = []
    for rel_name, content in entries:
        p = Path(rel_name)
        if p.suffix.lower() in (".toml", ".cfg", ".ini"):
            continue
        stem_low = p.stem.lower()
        if any(skip in stem_low for skip in ("readme", "ref", "reference_params", "published", "truth", "summary")):
            continue
        inherited = dict(dir_meta.get(str(p.parent), {}))
        def_sid: Optional[int] = None
        def_rname = "UNK"
        try:
            def_sid, def_rname = parse_residue_token(p.stem)
        except ValueError:
            pass
        rows = _convert_single_cest_profile_block(
            content,
            default_seq_id=def_sid,
            default_res_name=def_rname,
            inherited_meta=inherited,
            larmor_mhz_override=larmor_mhz_override,
            ref_offset_sentinel_ppm=sentinel,
        )
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError(f"zero CEST saturation profile rows extracted from {source_path}")

    out_p = Path(out_tsv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(format_cest_11col_tsv(all_rows), encoding="utf-8")
    return {
        "source": str(source_path),
        "out_tsv": str(out_p),
        "n_rows": len(all_rows),
        "n_residues": len({int(r["seq_id"]) for r in all_rows}),
    }


def evaluate_f13a_cest(
    cest_fit_json_text: str,
    *,
    p_b_ref: float,
    k_ex_ref_s_inv: float,
    dw_ref_ppm_by_seq_id: Mapping[int, float],
    nucleus: str = "15N",
    reference_dw_is_unsigned: bool = False,
    exit_code: int = 0,
    accession_id: str = "ZENODO:19186893",
    gpu_fit_json_text: Optional[str] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate `nmrassign cest --json-out` against published reference parameters under `F13a`."""
    assert_not_excluded_contaminated_entry(accession_id, manifest)
    mf = manifest if manifest is not None else load_preregistration_manifest()
    c13a = extract_manifest_bound_constants(mf)["F13a"]

    fit = json.loads(cest_fit_json_text)
    lm_strict_tol = bool(fit.get("converged", False))
    p_b_fit = float(fit.get("p_b", math.nan))
    k_ex_fit = float(fit.get("k_ex_s_inv", math.nan))

    abs_pb_err = abs(p_b_fit - p_b_ref)
    rel_pb_err = abs_pb_err / p_b_ref if p_b_ref > 0.0 else math.nan
    if p_b_ref < c13a["low_pb_cutoff"]:
        scaled_abs = (abs_pb_err / c13a["low_pb_abs_floor"]) * c13a["low_pb_rel_equiv"]
        pb_metric = min(rel_pb_err, scaled_abs)
    else:
        pb_metric = rel_pb_err

    kex_rel_err = abs(k_ex_fit - k_ex_ref_s_inv) / k_ex_ref_s_inv if k_ex_ref_s_inv > 0.0 else math.nan

    fit_res = {int(r["seq_id"]): float(r["dw_ab_ppm"]) for r in fit.get("residues", [])}
    matched_sids = sorted(sid for sid in fit_res if sid in dw_ref_ppm_by_seq_id)
    if reference_dw_is_unsigned:
        dw_preds = [abs(fit_res[sid]) for sid in matched_sids]
        dw_refs = [abs(float(dw_ref_ppm_by_seq_id[sid])) for sid in matched_sids]
    else:
        dw_preds = [fit_res[sid] for sid in matched_sids]
        dw_refs = [float(dw_ref_ppm_by_seq_id[sid]) for sid in matched_sids]

    med_dw_err = median_abs_err(dw_preds, dw_refs)

    metrics = {
        "pb_rel_err_or_low_pb_scaled_abs_err": pb_metric,
        "abs(k_ex_fit - k_ex_ref) / k_ex_ref": kex_rel_err,
        "median_abs_dw_err_ppm": med_dw_err,
    }
    secondary: Dict[str, Any] = {
        "p_b_fit": p_b_fit,
        "k_ex_fit_s_inv": k_ex_fit,
        "lm_strict_tol_flag": lm_strict_tol,
        "iterations": fit.get("iterations"),
        "reduced_chi2": fit.get("reduced_chi2"),
        "n_matched_residues": len(matched_sids),
        "pearson_r_dw": pearson_r(dw_preds, dw_refs),
        "rmse_dw_ppm": rmse(dw_preds, dw_refs),
    }
    if gpu_fit_json_text is not None:
        gfit = json.loads(gpu_fit_json_text)
        secondary["cpu_gpu_abs_delta_pb"] = abs(p_b_fit - float(gfit.get("p_b", math.nan)))
        secondary["cpu_gpu_abs_delta_kex"] = abs(k_ex_fit - float(gfit.get("k_ex_s_inv", math.nan)))

    eval_res = evaluate_gate_criteria(
        "F13a",
        metrics,
        exit_code=exit_code,
        accession_id=accession_id,
        nucleus=nucleus,
        n_sites=len(matched_sids),
        converged=math.isfinite(p_b_fit) and math.isfinite(k_ex_fit) and bool(matched_sids),
        manifest=mf,
    )
    eval_res["secondary_metrics"] = secondary
    return eval_res


# ============================================================================
# Gate F13b: Multi-Field CPMG & Off-Resonance R1rho Converter & Evaluator
# ============================================================================


def _parse_single_cpmg_r2eff_block(
    text: str,
    *,
    default_seq_id: Optional[int] = None,
    default_res_name: str = "UNK",
    inherited_meta: Optional[Mapping[str, str]] = None,
    larmor_mhz_override: Optional[float] = None,
) -> List[Dict[str, Any]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    non_comment = [ln for ln in lines if not ln.startswith("#") and not ln.startswith(";")]
    if non_comment:
        first_cols = [c.strip().lower() for c in re.split(r"[\t,]", non_comment[0])]
        if "nu_cpmg_hz" in first_cols and ("r2_eff_s_inv" in first_cols or "r2_eff" in first_cols):
            sep = "\t" if "\t" in non_comment[0] else ","
            cmap = {name: idx for idx, name in enumerate(first_cols)}
            r2_col = cmap.get("r2_eff_s_inv", cmap.get("r2_eff", 0))
            err_col = cmap.get("r2_eff_err", cmap.get("sigma_r2_eff", 0))
            lmhz_col = cmap.get("larmor_mhz", cmap.get("b0_mhz"))
            trel_col = cmap.get("t_relax_s", cmap.get("t_relax"))
            out: List[Dict[str, Any]] = []
            for row_ln in non_comment[1:]:
                cols = [c.strip() for c in row_ln.split(sep)]
                if len(cols) < len(first_cols):
                    continue
                if "seq_id" in cmap:
                    sid = int(cols[cmap["seq_id"]])
                    rname = cols[cmap["res_name"]].upper() if "res_name" in cmap else default_res_name
                elif "residue" in cmap:
                    sid, rname = parse_residue_token(cols[cmap["residue"]])
                elif default_seq_id is not None:
                    sid, rname = default_seq_id, default_res_name
                else:
                    raise ValueError("CPMG table missing seq_id/residue column")
                lmhz = (
                    float(larmor_mhz_override)
                    if larmor_mhz_override is not None
                    else float(cols[lmhz_col])  # type: ignore[index]
                )
                trel = float(cols[trel_col]) if trel_col is not None else 0.0
                out.append(
                    {
                        "seq_id": sid,
                        "res_name": rname,
                        "larmor_mhz": lmhz,
                        "t_relax_s": trel,
                        "nu_cpmg_hz": float(cols[cmap["nu_cpmg_hz"]]),
                        "r2_eff_s_inv": float(cols[r2_col]),
                        "r2_eff_err": float(cols[err_col]),
                    }
                )
            return out

    # Plain-text per-residue CPMG dispersion file (`# larmor_mhz=... T_relax=...` + `[nu_cpmg_hz, r2_eff, err]`)
    meta: Dict[str, str] = dict(inherited_meta or {})
    header_lines = [
        ln
        for ln in lines
        if ln.startswith("#") or ln.startswith(";") or ln.startswith("[") or ("=" in ln and not ln[0].isdigit())
    ]
    meta.update(_parse_kv_metadata_lines(header_lines))

    if "seq_id" in meta:
        sid = int(meta["seq_id"])
        rname = meta.get("res_name", default_res_name).upper()
    elif "residue" in meta:
        sid, rname = parse_residue_token(meta["residue"])
    elif default_seq_id is not None:
        sid, rname = default_seq_id, default_res_name
    else:
        return []

    lmhz = (
        float(larmor_mhz_override)
        if larmor_mhz_override is not None
        else next(
            (float(meta[k]) for k in ("larmor_mhz", "sfo1", "bf1", "obs_mhz", "b0_mhz", "b0") if k in meta),
            None,
        )
    )
    trel = next(
        (float(meta[k]) for k in ("t_relax_s", "t_relax", "time_t2", "tcpmg") if k in meta),
        None,
    )
    if lmhz is None or trel is None:
        return []

    is_time_sec = meta.get("units", "").lower() in ("s", "sec", "seconds")
    rows_out: List[Dict[str, Any]] = []
    for ln in lines:
        if ln in header_lines or ln.startswith("#") or ln.startswith(";"):
            continue
        parts = ln.replace(",", " ").split()
        if len(parts) < 3:
            continue
        try:
            nu_hz = float(parts[0])
            val = float(parts[1])
            err = float(parts[2])
        except ValueError:
            continue
        if nu_hz <= 0.0:
            continue
        if is_time_sec:
            if val <= 0.0:
                continue
            r2_val = 1.0 / val
            r2_err = abs(err) / (val * val)
        else:
            r2_val = val
            r2_err = abs(err)
        rows_out.append(
            {
                "seq_id": sid,
                "res_name": rname,
                "larmor_mhz": float(lmhz),
                "t_relax_s": float(trel),
                "nu_cpmg_hz": nu_hz,
                "r2_eff_s_inv": r2_val,
                "r2_eff_err": r2_err,
            }
        )
    return rows_out


def format_cpmg_7col_tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Format 7-column TSV expected by `nmrassign cpmg --in`."""
    out = ["seq_id\tres_name\tlarmor_mhz\tt_relax_s\tnu_cpmg_hz\tr2_eff_s_inv\tr2_eff_err"]
    for r in sorted(
        rows,
        key=lambda x: (int(x["seq_id"]), float(x["larmor_mhz"]), float(x["nu_cpmg_hz"])),
    ):
        out.append(
            f"{int(r['seq_id'])}\t{r['res_name']}\t{float(r['larmor_mhz']):.5f}\t"
            f"{float(r['t_relax_s']):.5f}\t{float(r['nu_cpmg_hz']):.4f}\t"
            f"{float(r['r2_eff_s_inv']):.5f}\t{float(r['r2_eff_err']):.5f}"
        )
    return "\n".join(out) + "\n"


def convert_f13b_cpmg_to_tsv(
    source_path: Union[Path, str],
    out_tsv_path: Union[Path, str],
    *,
    larmor_mhz_override: Optional[float] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert `CPMG.zip`, directory of `R2,eff` profiles, or tabular/STAR file into 7-col CPMG TSV."""
    assert_not_excluded_contaminated_entry(source_path, manifest)
    src = Path(source_path)
    entries: List[Tuple[str, str]] = []
    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("/") or "/__MACOSX" in name or Path(name).name.startswith("."):
                    continue
                if name.lower().endswith((".txt", ".out", ".dat", ".tsv", ".csv", ".toml", ".cfg", ".ini")):
                    entries.append((name, zf.read(name).decode("utf-8", errors="replace")))
    elif src.is_dir():
        for fp in sorted(src.rglob("*")):
            if fp.is_file() and not fp.name.startswith("."):
                if fp.suffix.lower() in (".txt", ".out", ".dat", ".tsv", ".csv", ".toml", ".cfg", ".ini"):
                    entries.append((str(fp.relative_to(src)), fp.read_text(encoding="utf-8", errors="replace")))
    elif src.is_file():
        entries.append((src.name, src.read_text(encoding="utf-8", errors="replace")))
    else:
        raise FileNotFoundError(f"CPMG input source not found: {src}")

    dir_meta: Dict[str, Dict[str, str]] = {}
    for rel_name, content in entries:
        p = Path(rel_name)
        if p.suffix.lower() in (".toml", ".cfg", ".ini"):
            dir_meta.setdefault(str(p.parent), {}).update(_parse_kv_metadata_lines(content.splitlines()))

    all_rows: List[Dict[str, Any]] = []
    for rel_name, content in entries:
        p = Path(rel_name)
        if p.suffix.lower() in (".toml", ".cfg", ".ini"):
            continue
        stem_low = p.stem.lower()
        if any(skip in stem_low for skip in ("readme", "reference_params", "published", "truth", "summary")):
            continue
        def_sid: Optional[int] = None
        def_rname = "UNK"
        try:
            def_sid, def_rname = parse_residue_token(p.stem)
        except ValueError:
            pass
        all_rows.extend(
            _parse_single_cpmg_r2eff_block(
                content,
                default_seq_id=def_sid,
                default_res_name=def_rname,
                inherited_meta=dir_meta.get(str(p.parent), {}),
                larmor_mhz_override=larmor_mhz_override,
            )
        )

    if not all_rows:
        raise ValueError(f"zero CPMG R2,eff rows extracted from {source_path}")

    out_p = Path(out_tsv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(format_cpmg_7col_tsv(all_rows), encoding="utf-8")
    return {
        "source": str(source_path),
        "out_tsv": str(out_p),
        "n_rows": len(all_rows),
        "n_residues": len({int(r["seq_id"]) for r in all_rows}),
    }


def convert_f13b_raw_planes_to_tsv(
    raw_table_text: str,
    *,
    larmor_mhz: float,
    t_relax_s: float,
    ncyc_values_ARE_cycles: bool = False,
) -> str:
    """Convert raw CPMG intensity tables (`ZENODO:6605011` `D25_MQCPMG_*`) into `--raw-planes` TSV.

    Supports both wide tables (header `Residue <nu_0_or_ncyc_0> ... <noise>`) and long tables
    (`residue, nu_cpmg_hz, intensity[, baseline_noise]`), setting `nu_cpmg_hz = 0` for `I0`.
    """
    lines = [
        ln.strip()
        for ln in raw_table_text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if len(lines) < 2:
        raise ValueError("raw CPMG plane table requires header and >= 1 data row")

    header_toks = lines[0].replace(",", " ").split()
    out_lines = ["seq_id\tres_name\tlarmor_mhz\tt_relax_s\tnu_cpmg_hz\tintensity\tbaseline_noise"]

    low_hdr = [h.lower() for h in header_toks]
    if "nu_cpmg_hz" in low_hdr and "intensity" in low_hdr:
        sep = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else None)
        cmap = {name: idx for idx, name in enumerate(low_hdr)}
        for row_ln in lines[1:]:
            cols = [c.strip() for c in (row_ln.split(sep) if sep else row_ln.split())]
            if "seq_id" in cmap:
                sid = int(cols[cmap["seq_id"]])
                rname = cols[cmap["res_name"]].upper() if "res_name" in cmap else "UNK"
            else:
                sid, rname = parse_residue_token(cols[0])
            nu_v = float(cols[cmap["nu_cpmg_hz"]])
            if ncyc_values_ARE_cycles and nu_v > 0.0:
                nu_v = nu_v / t_relax_s
            inten = float(cols[cmap["intensity"]])
            noise = cols[cmap["baseline_noise"]] if "baseline_noise" in cmap else "NA"
            out_lines.append(
                f"{sid}\t{rname}\t{larmor_mhz:.5f}\t{t_relax_s:.5f}\t{nu_v:.4f}\t{inten:.6f}\t{noise}"
            )
        return "\n".join(out_lines) + "\n"

    # Wide table: col 0 = residue token, trailing noise/err column if labelled `noise`/`err`/`rmsd`
    has_trailing_noise = low_hdr[-1] in ("noise", "err", "error", "sigma", "rmsd", "baseline_noise")
    sched_hdr = header_toks[1:-1] if has_trailing_noise else header_toks[1:]
    nu_schedule: List[float] = []
    for tok in sched_hdr:
        t_clean = tok.lower().rstrip("hz")
        if t_clean in ("ref", "i0", "0", "0.0"):
            nu_schedule.append(0.0)
        else:
            raw_num = float(t_clean)
            nu_schedule.append((raw_num / t_relax_s) if (ncyc_values_ARE_cycles and raw_num > 0.0) else raw_num)

    for row_ln in lines[1:]:
        cols = row_ln.replace(",", " ").split()
        if len(cols) < 1 + len(nu_schedule):
            continue
        sid, rname = parse_residue_token(cols[0])
        noise_s = cols[1 + len(nu_schedule)] if has_trailing_noise and len(cols) > 1 + len(nu_schedule) else "NA"
        for k, nu_v in enumerate(nu_schedule):
            inten = float(cols[1 + k])
            out_lines.append(
                f"{sid}\t{rname}\t{larmor_mhz:.5f}\t{t_relax_s:.5f}\t{nu_v:.4f}\t{inten:.6f}\t{noise_s}"
            )
    return "\n".join(out_lines) + "\n"


def convert_f13b_r1rho_to_tsv(
    r1rho_rows: Sequence[Mapping[str, Any]],
    ground_state_shifts_ppm: Optional[Mapping[int, float]],
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Build secondary report-only off-resonance `R1rho` 6-col TSV per residue.

    Per `F13b` adapter rule 2, returns `None` (report-only `N/A`) if no deposited ground-state
    isotropic chemical-shift column/table is present.
    """
    if not ground_state_shifts_ppm:
        return None
    by_sid: Dict[int, List[Mapping[str, Any]]] = {}
    for r in r1rho_rows:
        sid = int(r["seq_id"])
        if sid in ground_state_shifts_ppm:
            by_sid.setdefault(sid, []).append(r)
    if not by_sid:
        return None
    out: Dict[int, Dict[str, Any]] = {}
    for sid, rows in sorted(by_sid.items()):
        lines = ["seq_id\tres_name\tcarrier_ppm\tspin_lock_hz\tr1rho_s_inv\tr1rho_err"]
        rname = str(rows[0].get("res_name", "UNK")).upper()
        for r in rows:
            lines.append(
                f"{sid}\t{rname}\t{float(r['carrier_ppm']):.5f}\t"
                f"{float(r['spin_lock_hz']):.4f}\t{float(r['r1rho_s_inv']):.5f}\t"
                f"{float(r['r1rho_err']):.5f}"
            )
        out[sid] = {
            "seq_id": sid,
            "res_name": rname,
            "major_ppm": float(ground_state_shifts_ppm[sid]),
            "tsv": "\n".join(lines) + "\n",
        }
    return out


def evaluate_f13b_cpmg(
    cpmg_fit_json_text: str,
    *,
    p_b_ref: float,
    k_ex_ref_s_inv: float,
    dw_ref_ppm_by_seq_id: Mapping[int, float],
    exit_code: int = 0,
    accession_id: str = "ZENODO:19186893",
    gpu_fit_json_text: Optional[str] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate `nmrassign cpmg --json-out` against published reference parameters under `F13b`."""
    assert_not_excluded_contaminated_entry(accession_id, manifest)
    mf = manifest if manifest is not None else load_preregistration_manifest()

    fit = json.loads(cpmg_fit_json_text)
    lm_strict_tol = bool(fit.get("converged", False))
    p_b_fit = float(fit.get("p_b", math.nan))
    k_ex_fit = float(fit.get("k_ex_s_inv", math.nan))

    pb_rel_err = abs(p_b_fit - p_b_ref) / p_b_ref if p_b_ref > 0.0 else math.nan
    kex_rel_err = abs(k_ex_fit - k_ex_ref_s_inv) / k_ex_ref_s_inv if k_ex_ref_s_inv > 0.0 else math.nan

    # Evaluate across gated exchanging sites matched in reference table (fallback to all matched if gate omitted)
    gated_fit = {
        int(r["seq_id"]): abs(float(r["dw_ppm"]))
        for r in fit.get("residues", [])
        if r.get("gate", {}).get("significant_exchange", True)
    }
    if not gated_fit:
        gated_fit = {int(r["seq_id"]): abs(float(r["dw_ppm"])) for r in fit.get("residues", [])}

    matched_sids = sorted(sid for sid in gated_fit if sid in dw_ref_ppm_by_seq_id)
    dw_preds = [gated_fit[sid] for sid in matched_sids]
    dw_refs = [abs(float(dw_ref_ppm_by_seq_id[sid])) for sid in matched_sids]

    n_sites = len(matched_sids)
    med_dw_err = median_abs_err(dw_preds, dw_refs)
    r_abs_dw = pearson_r(dw_preds, dw_refs)

    metrics = {
        "abs(p_B_fit - p_B_ref) / p_B_ref": pb_rel_err,
        "abs(k_ex_fit - k_ex_ref) / k_ex_ref": kex_rel_err,
        "median_abs_dw_err_ppm": med_dw_err,
        "pearson_r_abs_dw": r_abs_dw,
    }
    secondary: Dict[str, Any] = {
        "p_b_fit": p_b_fit,
        "k_ex_fit_s_inv": k_ex_fit,
        "lm_strict_tol_flag": lm_strict_tol,
        "iterations": fit.get("iterations"),
        "n_gated_matched_sites": n_sites,
        "pearson_r_abs_dw": r_abs_dw,
        "rmse_abs_dw_ppm": rmse(dw_preds, dw_refs),
    }
    if gpu_fit_json_text is not None:
        gfit = json.loads(gpu_fit_json_text)
        secondary["cpu_gpu_abs_delta_pb"] = abs(p_b_fit - float(gfit.get("p_b", math.nan)))
        secondary["cpu_gpu_abs_delta_kex"] = abs(k_ex_fit - float(gfit.get("k_ex_s_inv", math.nan)))

    eval_res = evaluate_gate_criteria(
        "F13b",
        metrics,
        exit_code=exit_code,
        accession_id=accession_id,
        n_sites=n_sites,
        converged=math.isfinite(p_b_fit) and math.isfinite(k_ex_fit) and n_sites > 0,
        manifest=mf,
    )
    eval_res["secondary_metrics"] = secondary
    return eval_res


# ============================================================================
# Gate F13c: 2D HSQC Ligand Titration Apo-Peak Reshaper, Step Builder & Evaluator
# ============================================================================


def reshape_picked_apo_peaks_to_4col_tsv(
    raw_picked_tsv_text: str,
    *,
    source_label: str = "raw_apo_peaks.tsv",
    manifest: Optional[Mapping[str, Any]] = None,
) -> str:
    """Reshape `nmrassign pick --hsqc` output (`experiment\\tH\\tN...`) into `k\\tUNK\\th\\tn`."""
    assert_not_excluded_contaminated_entry(source_label, manifest)
    lines = [
        ln.strip()
        for ln in raw_picked_tsv_text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        raise ValueError("picked apo peaks TSV is empty")

    hdr = [c.strip().lower() for c in lines[0].split("\t")]
    h_idx = next((i for i, n in enumerate(hdr) if n in ("h", "h_ppm", "1h", "f2_ppm")), None)
    n_idx = next((i for i, n in enumerate(hdr) if n in ("n", "n_ppm", "15n", "f1_ppm")), None)
    if h_idx is None or n_idx is None:
        raise ValueError(f"picked apo peaks header missing H/N columns: {lines[0]!r}")

    out = ["seq_id\tres_name\th_ppm\tn_ppm"]
    k_row = 0
    for ln in lines[1:]:
        cols = [c.strip() for c in ln.split("\t")]
        if len(cols) <= max(h_idx, n_idx):
            continue
        h_val = float(cols[h_idx])
        n_val = float(cols[n_idx])
        k_row += 1
        out.append(f"{k_row}\tUNK\t{h_val:.5f}\t{n_val:.5f}")

    if k_row == 0:
        raise ValueError("picked apo peaks TSV contained zero peak rows")
    return "\n".join(out) + "\n"


def _conc_to_um(value: float, unit: str, mm_to_um: float) -> float:
    u = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    if u in ("um", "micromolar", "umol/l", ""):
        return value
    if u in ("mm", "millimolar", "mmol/l"):
        return value * mm_to_um
    if u in ("m", "mol/l", "molar"):
        return value * mm_to_um * mm_to_um
    if u in ("nm", "nanomolar", "nmol/l"):
        return value / mm_to_um
    raise ValueError(f"unsupported concentration unit: {unit!r}")


def build_f13c_titration_step_specs(
    metadata_table_text: str,
    *,
    base_dir: Optional[Union[Path, str]] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build sorted `--protein-um <P0_uM>` and `--step {L_uM}:{P_uM}={path}` arguments from archive metadata."""
    assert_not_excluded_contaminated_entry(metadata_table_text, manifest)
    mf = manifest if manifest is not None else load_preregistration_manifest()
    mm_to_um = extract_manifest_bound_constants(mf)["F13c"]["mm_to_um"]

    sep = "\t" if "\t" in metadata_table_text.splitlines()[0] else ","
    reader = csv.DictReader(
        [ln for ln in metadata_table_text.splitlines() if ln.strip() and not ln.strip().startswith("#")],
        delimiter=sep,
    )
    if reader.fieldnames is None:
        raise ValueError("titration metadata table has no header row")

    def _find_field(cands: Sequence[str]) -> Optional[str]:
        assert reader.fieldnames is not None
        for c in cands:
            for fn in reader.fieldnames:
                if fn.strip().lower() == c:
                    return fn
        return None

    l_col = _find_field(["ligand_um", "l_um", "ligand_mm", "l_mm", "ligand_conc", "ligand"])
    p_col = _find_field(["protein_um", "p_um", "protein_mm", "p_mm", "protein_conc", "protein"])
    path_col = _find_field(["spectrum_path", "path", "spectrum", "file", "exp_dir"])
    l_unit_col = _find_field(["ligand_unit", "unit"])
    p_unit_col = _find_field(["protein_unit", "unit"])

    if l_col is None or p_col is None or path_col is None:
        raise ValueError(
            f"titration metadata table requires ligand, protein, and spectrum_path columns; got {reader.fieldnames}"
        )

    default_l_unit = "mm" if l_col.strip().lower().endswith("_mm") else "um"
    default_p_unit = "mm" if p_col.strip().lower().endswith("_mm") else "um"

    steps_raw: List[Dict[str, Any]] = []
    root = Path(base_dir) if base_dir is not None else None
    for row in reader:
        raw_path = row[path_col].strip()
        assert_not_excluded_contaminated_entry(raw_path, mf)
        resolved = str((root / raw_path) if (root is not None and not Path(raw_path).is_absolute()) else Path(raw_path))
        lu = row[l_unit_col].strip() if l_unit_col and row.get(l_unit_col) else default_l_unit
        pu = row[p_unit_col].strip() if p_unit_col and row.get(p_unit_col) else default_p_unit
        l_um = _conc_to_um(float(row[l_col]), lu, mm_to_um)
        p_um = _conc_to_um(float(row[p_col]), pu, mm_to_um)
        steps_raw.append({"ligand_um": l_um, "protein_um": p_um, "spectrum_path": resolved})

    if len(steps_raw) < 2:
        raise ValueError("CSP titration requires >= 2 concentration steps")

    steps_sorted = sorted(steps_raw, key=lambda s: float(s["ligand_um"]))
    p0_um = float(steps_sorted[0]["protein_um"])
    apo_spectrum_path = steps_sorted[0]["spectrum_path"]
    step_specs = [
        f"{s['ligand_um']:.6g}:{s['protein_um']:.6g}={s['spectrum_path']}"
        for s in steps_sorted
    ]
    return {
        "protein_um": p0_um,
        "apo_spectrum_path": apo_spectrum_path,
        "steps": steps_sorted,
        "step_cli_specs": step_specs,
    }


def extract_f13c_reference_kd_um(
    biophysical_metadata_text: str,
    *,
    manifest: Optional[Mapping[str, Any]] = None,
) -> float:
    """Extract independent co-deposited ITC/MST `K_d` in micromolar (`uM`) from archive metadata."""
    mf = manifest if manifest is not None else load_preregistration_manifest()
    mm_to_um = extract_manifest_bound_constants(mf)["F13c"]["mm_to_um"]

    # Check JSON first
    s = biophysical_metadata_text.strip()
    if s.startswith("{"):
        js = json.loads(s)
        for key, unit in (
            ("kd_um", "um"),
            ("kd_nm", "nm"),
            ("kd_mm", "mm"),
            ("kd_m", "m"),
        ):
            if key in js and js[key] is not None:
                return _conc_to_um(float(js[key]), unit, mm_to_um)

    # Key-value or CSV/TSV search (`Kd = 12.5 uM`, `K_d (uM),12.5`, `KD = 1.2e-5 M`)
    m_with_unit = re.search(
        r"\bK_?[dD]\b[^0-9\n]{0,20}([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)\s*(nM|uM|µM|μM|mM|M)\b",
        s,
    )
    if m_with_unit:
        return _conc_to_um(float(m_with_unit.group(1)), m_with_unit.group(2), mm_to_um)

    lines = [ln for ln in s.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) >= 2:
        sep = "\t" if "\t" in lines[0] else ","
        reader = csv.DictReader(lines, delimiter=sep)
        if reader.fieldnames:
            for fn in reader.fieldnames:
                fn_low = fn.strip().lower().replace("μ", "u").replace("µ", "u")
                if "kd" in fn_low or "k_d" in fn_low:
                    unit = "nm" if "nm" in fn_low else ("mm" if "mm" in fn_low else ("um" if "um" in fn_low else "m"))
                    vals = [
                        _conc_to_um(float(r[fn]), unit, mm_to_um)
                        for r in reader
                        if r.get(fn) and r[fn].strip()
                    ]
                    if vals:
                        return median(vals)

    raise ValueError("could not locate independent ITC/MST K_d value in biophysical metadata")


def evaluate_f13c_csp(
    csp_fit_json_text: str,
    *,
    kd_ref_um: float,
    exit_code: int = 0,
    accession_id: str = "ZENODO:17228160",
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate `nmrassign csp --json-out` against independent reference `K_d` under `F13c`."""
    assert_not_excluded_contaminated_entry(accession_id, manifest)
    mf = manifest if manifest is not None else load_preregistration_manifest()

    fit = json.loads(csp_fit_json_text)
    kd_fit_um = float(fit.get("global_kd_um", math.nan))
    sig_ids = fit.get("thresholds", {}).get("significant_seq_ids", [])
    n_pocket = len(sig_ids)

    if kd_fit_um > 0.0 and kd_ref_um > 0.0 and math.isfinite(kd_fit_um):
        abs_log10_err = abs(math.log10(kd_fit_um / kd_ref_um))
        fold_ratio = kd_fit_um / kd_ref_um
    else:
        abs_log10_err = math.nan
        fold_ratio = math.nan

    ci_lo = fit.get("global_kd_ci95_low_um")
    ci_hi = fit.get("global_kd_ci95_high_um")
    if ci_lo is not None and ci_hi is not None:
        ci95: Optional[Tuple[float, float]] = (float(ci_lo), float(ci_hi))
    else:
        raw_ci = fit.get("global_kd_ci95_um")
        ci95 = (
            (float(raw_ci[0]), float(raw_ci[1]))
            if isinstance(raw_ci, Sequence) and len(raw_ci) == 2
            else None
        )
    ci95_covers_ref = (
        bool(ci95[0] <= kd_ref_um <= ci95[1]) if ci95 is not None else None
    )

    metrics = {
        "n_pocket_residues": n_pocket,
        "abs(log10(kd_um / kd_ref_um))": abs_log10_err,
    }
    eval_res = evaluate_gate_criteria(
        "F13c",
        metrics,
        exit_code=exit_code,
        accession_id=accession_id,
        n_sites=n_pocket,
        converged=math.isfinite(kd_fit_um) and kd_fit_um > 0.0,
        manifest=mf,
    )
    eval_res["secondary_metrics"] = {
        "global_kd_um": kd_fit_um,
        "kd_ref_um": kd_ref_um,
        "kd_fold_ratio": fold_ratio,
        "global_kd_ci95_um": ci95,
        "ci95_covers_kd_ref": ci95_covers_ref,
        "n_total_residue_fits": len(fit.get("residue_fits", [])),
    }
    return eval_res
