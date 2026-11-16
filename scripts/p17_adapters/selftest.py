#!/usr/bin/env python3
"""Comprehensive offline self-test suite for `scripts/p17_adapters/` (`F10`..`F17`).

Provenance:
- Original clean-room Python 3 standard-library test harness for Phase 17 blind
  empirical benchmarking (`data/benchmark/p17_empirical_preregistration.json`).
- Exercises 100% of Wave 3 (`F10`, `F11`, `F13a`, `F13b`, `F13c`) and Wave 4
  (`F14`, `F16`, `F17`) deterministic format adapters, CLI entrypoints, and
  `pass_criteria` evaluators on tiny synthetic in-memory payloads AND invokes the
  compiled `./target/release/nmrassign` binary on synthetic temp inputs BEFORE any
  real dataset is ever downloaded.
- Also performs static AST verification across all production `.py` files in
  `scripts/p17_adapters/` to prove zero network imports and zero unregistered
  numeric constants outside `data/benchmark/p17_empirical_preregistration.json`.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
from typing import Any, Callable, Dict, List, Sequence, Tuple
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.p17_adapters import (
    f10_nmrstar_to_relax_tsv,
    f13a_cest_to_tsv,
    f13b_cpmg_to_tsv,
    f13c_apo_peaks_to_tsv,
    f16_dosy_plane_integrator,
    f17_mtbls1_manifest,
)
from scripts.p17_adapters.common import (
    evaluate_gate_criteria,
    extract_manifest_bound_constants,
    find_repo_root,
    hash_adapter_directory,
    load_preregistration_manifest,
    mae,
    median,
    median_abs_err,
    median_rel_abs_err,
    pearson_r,
    rmse,
    sha256_bytes,
    sha256_file,
    spearman_rho,
)
from scripts.p17_adapters.w3_adapters import (
    assert_not_excluded_contaminated_entry,
    build_f13c_titration_step_specs,
    compute_bruker_hsqct2_vclist_delays,
    convert_f13b_r1rho_to_tsv,
    evaluate_f10_modelfree,
    evaluate_f11_relax,
    evaluate_f13a_cest,
    evaluate_f13b_cpmg,
    evaluate_f13c_csp,
    export_f10_modelfree_tsvs,
    extract_f13c_reference_kd_um,
    reshape_picked_apo_peaks_to_4col_tsv,
)
from scripts.p17_adapters.w4_adapters import (
    build_f17_mtbls1_manifest,
    count_dept135_peak_multiplicities,
    evaluate_f14_nmr1d_and_qnmr,
    evaluate_f16_dosy,
    evaluate_f17_metabolomics,
    extract_dosy_header_solvent_and_temp,
    extract_f14_qnmr_sample_specs,
    locate_extracted_bruker_1d_dir,
    multiplet_cli_flags_for_acqus,
    read_bruker_acqus_nuc1,
    select_first_bruker_1d_dir_by_nuc1,
)

FORBIDDEN_NET_MODULES = frozenset(
    {
        "urllib",
        "http",
        "socket",
        "ftplib",
        "poplib",
        "imaplib",
        "smtplib",
        "telnetlib",
        "xmlrpc",
        "requests",
        "httpx",
        "aiohttp",
    }
)
FORBIDDEN_NET_BINARIES = frozenset({"curl", "wget", "gh", "rsync", "scp", "sftp"})
ALLOWED_PRODUCTION_FLOATS = frozenset({0.0, 1.0})


def _run_nmrassign(bin_path: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [str(bin_path), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"nmrassign {' '.join(args[:4])} failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


# ============================================================================
# 1. Static AST Blind-Firewall & Zero-Unregistered-Constants Audit
# ============================================================================


def test_ast_blind_firewall_and_zero_hardcoded_constants(repo_root: Path) -> str:
    adapter_dir = repo_root / "scripts" / "p17_adapters"
    py_files = sorted(adapter_dir.glob("*.py"))
    assert len(py_files) >= 11, f"expected >= 11 .py files in {adapter_dir}, got {len(py_files)}"

    production_files = [p for p in py_files if p.name != "selftest.py"]
    for py_path in py_files:
        src = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_mod = alias.name.split(".", 1)[0]
                    assert top_mod not in FORBIDDEN_NET_MODULES, (
                        f"forbidden network import {alias.name!r} in {py_path.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_mod = node.module.split(".", 1)[0]
                assert top_mod not in FORBIDDEN_NET_MODULES, (
                    f"forbidden network from-import {node.module!r} in {py_path.name}"
                )
            elif py_path in production_files and isinstance(node, ast.Constant):
                if isinstance(node.value, str):
                    low = node.value.strip().lower()
                    assert low not in FORBIDDEN_NET_BINARIES, (
                        f"forbidden network binary literal {node.value!r} in {py_path.name}:{node.lineno}"
                    )
                elif isinstance(node.value, float):
                    assert node.value in ALLOWED_PRODUCTION_FLOATS, (
                        f"unregistered float constant {node.value!r} in {py_path.name}:{node.lineno} "
                        f"(must be read from p17_empirical_preregistration.json)"
                    )

    hashes = hash_adapter_directory(adapter_dir)
    assert len(hashes) == len(py_files)
    for fname, rec in hashes.items():
        assert len(rec["sha256"]) == 64 and int(rec["bytes"]) > 0, f"bad hash entry for {fname}"

    return f"audited {len(py_files)} files ({len(production_files)} prod modules: 0 net imports, 0 unregistered floats)"


# ============================================================================
# 2. Common Statistics, SHA-256, Manifest Criteria & BMRB:53960 Firewall
# ============================================================================


def test_common_statistics_and_manifest_evaluator(repo_root: Path) -> str:
    mf = load_preregistration_manifest(repo_root / "data/benchmark/p17_empirical_preregistration.json")
    consts = extract_manifest_bound_constants(mf)
    assert math.isclose(consts["F10"]["bmrb6470_high_field_mhz"], 600.13)
    assert math.isclose(consts["F10"]["bmrb6470_low_field_mhz"], 360.13)
    assert math.isclose(consts["F16"]["celsius_to_kelvin_offset"], 273.15)
    assert consts["F17"]["top_k"] == 15

    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert math.isclose(pearson_r(xs, ys), 1.0)
    assert math.isclose(spearman_rho(xs, ys), 1.0)
    assert math.isclose(mae(ys, xs), 3.0)
    assert math.isclose(rmse(ys, xs), math.sqrt(11.0))
    assert math.isclose(median([10.0, 2.0, 4.0, 8.0]), 6.0)
    assert math.isclose(median_abs_err([1.1, 2.2, 3.5], [1.0, 2.0, 3.0]), 0.2)
    assert math.isclose(median_rel_abs_err([1.1, 2.2, 3.3], [1.0, 2.0, 3.0]), 0.1)

    # Fallback override test on F10 (`BMRB:18971`: n_matched >= 20, mae_s2 <= 0.06)
    f10_fb = evaluate_gate_criteria(
        "F10",
        {
            "abs(tau_m_ns - 4.09) / 4.09": 0.04,
            "abs(d_par_over_d_perp - 1.17)": 0.05,
            "n_matched_core_residues": 24,
            "pearson_r_s2": 0.88,
            "mae_s2": 0.055,
        },
        accession_id="BMRB:18971",
        manifest=mf,
    )
    assert f10_fb["verdict"] == "PASS"
    f10_prim_fail = evaluate_gate_criteria(
        "F10",
        {
            "abs(tau_m_ns - 4.09) / 4.09": 0.04,
            "abs(d_par_over_d_perp - 1.17)": 0.05,
            "n_matched_core_residues": 24,
            "pearson_r_s2": 0.88,
            "mae_s2": 0.055,
        },
        accession_id="BMRB:6470",
        manifest=mf,
    )
    assert f10_prim_fail["verdict"] == "FAIL"

    # Nucleus-specific threshold on F13a (`13C` requires <= 0.25 ppm vs `15N` <= 0.35 ppm)
    f13a_15n = evaluate_gate_criteria(
        "F13a",
        {
            "pb_rel_err_or_low_pb_scaled_abs_err": 0.1,
            "abs(k_ex_fit - k_ex_ref) / k_ex_ref": 0.1,
            "median_abs_dw_err_ppm": 0.30,
        },
        nucleus="15N",
        manifest=mf,
    )
    f13a_13c = evaluate_gate_criteria(
        "F13a",
        {
            "pb_rel_err_or_low_pb_scaled_abs_err": 0.1,
            "abs(k_ex_fit - k_ex_ref) / k_ex_ref": 0.1,
            "median_abs_dw_err_ppm": 0.30,
        },
        nucleus="13C",
        manifest=mf,
    )
    assert f13a_15n["verdict"] == "PASS" and f13a_13c["verdict"] == "FAIL"

    # Disjunctive multi-site clause on F13b (`n_sites >= 5`: `pearson_r_abs_dw >= 0.85` rescues `median_abs_dw_err_ppm = 0.45`)
    f13b_disj = evaluate_gate_criteria(
        "F13b",
        {
            "abs(p_B_fit - p_B_ref) / p_B_ref": 0.10,
            "abs(k_ex_fit - k_ex_ref) / k_ex_ref": 0.12,
            "median_abs_dw_err_ppm": 0.45,
            "pearson_r_abs_dw": 0.92,
        },
        n_sites=6,
        manifest=mf,
    )
    assert f13b_disj["verdict"] == "PASS"

    # NOT_EVALUABLE archive failure path
    ne = evaluate_gate_criteria(
        "F17",
        {},
        archive_not_evaluable_reason="HTTP 503 outage on EBI MetaboLights FTP",
        manifest=mf,
    )
    assert ne["verdict"] == "NOT_EVALUABLE"

    # Excluded contaminated entry firewall (`BMRB:53960`)
    blocked = False
    try:
        assert_not_excluded_contaminated_entry("data/bmr53960_3.str", mf)
    except ValueError:
        blocked = True
    assert blocked, "expected BMRB:53960 firewall to raise ValueError"

    return "stats + fallback/nucleus/disjunctive/NOT_EVALUABLE + BMRB:53960 firewall OK"


# ============================================================================
# 3. Gate F10: BMRB:6470 Dual-600 Disambiguation + `nmrassign modelfree`
# ============================================================================


def _woessner_lipari_szabo_15n_rates(
    field_1h_mhz: float,
    tau_m_ns: float,
    aniso_ratio: float,
    cos_theta: float,
    s2: float,
    tau_e_ps: float,
) -> Tuple[float, float, float]:
    """Analytical axially symmetric Woessner (1962) + Lipari-Szabo (1982) 15N R1, R2, hetNOE."""
    mu0_over_4pi = 1.0e-7
    hbar = 1.054571817e-34
    gamma_h = 2.6752218744e8
    gamma_n = -2.71261804e7
    r_nh = 1.02e-10
    csa_ppm = -172.0e-6

    w_h = 2.0 * math.pi * field_1h_mhz * 1.0e6
    w_n = abs(gamma_n / gamma_h) * w_h

    d_iso = 1.0 / (6.0 * tau_m_ns * 1.0e-9)
    d_perp = 3.0 * d_iso / (2.0 + aniso_ratio)
    d_par = aniso_ratio * d_perp

    tau1 = 1.0 / (6.0 * d_perp)
    tau2 = 1.0 / (5.0 * d_perp + d_par)
    tau3 = 1.0 / (2.0 * d_perp + 4.0 * d_par)

    c2 = cos_theta * cos_theta
    s2_ang = 1.0 - c2
    a1 = 0.25 * (3.0 * c2 - 1.0) ** 2
    a2 = 3.0 * c2 * s2_ang
    a3 = 0.75 * s2_ang * s2_ang
    te = tau_e_ps * 1.0e-12

    def j_spec(w: float) -> float:
        val = 0.0
        for ak, tk in ((a1, tau1), (a2, tau2), (a3, tau3)):
            ti = (tk * te) / (tk + te) if te > 0.0 else 0.0
            val += ak * (
                s2 * tk / (1.0 + (w * tk) ** 2)
                + (1.0 - s2) * ti / (1.0 + (w * ti) ** 2)
            )
        return 0.4 * val

    d_dd = (mu0_over_4pi * hbar * gamma_h * gamma_n / (r_nh**3)) ** 2
    c_csa = (w_n * csa_ppm) ** 2 / 3.0

    j0 = j_spec(0.0)
    j_wn = j_spec(w_n)
    j_wh = j_spec(w_h)
    j_diff = j_spec(w_h - w_n)
    j_sum = j_spec(w_h + w_n)

    r1 = 0.25 * d_dd * (j_diff + 3.0 * j_wn + 6.0 * j_sum) + c_csa * j_wn
    r2 = (
        0.125 * d_dd * (4.0 * j0 + j_diff + 3.0 * j_wn + 6.0 * j_wh + 6.0 * j_sum)
        + (c_csa / 6.0) * (4.0 * j0 + 3.0 * j_wn)
    )
    sigma_nh = 0.25 * d_dd * (6.0 * j_sum - j_diff)
    noe = 1.0 + (gamma_h / gamma_n) * (sigma_nh / r1)
    return r1, r2, noe


def test_f10_adapter_and_nmrassign_modelfree(nmrassign_bin: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="p17_selftest_f10_") as tmp_s:
        tmp = Path(tmp_s)
        n_res = 54
        tau_m_true = 4.09
        aniso_true = 1.17

        pdb_lines: List[str] = ["MODEL        1"]
        t1_600_rows: List[str] = []
        t2_600_rows: List[str] = []
        noe_600_rows: List[str] = []
        t1_360_rows: List[str] = []
        t2_360_rows: List[str] = []
        noe_360_rows: List[str] = []
        s2_rows: List[str] = []

        atom_serial = 1
        for k in range(1, n_res + 1):
            res_name = "PRO" if k == 19 else ("GLY" if k % 7 == 0 else "ALA")
            # Distribute N-H bond orientations uniformly across the unit sphere
            u_z = -0.92 + 1.84 * ((k - 1) / float(n_res - 1))
            sin_th = math.sqrt(max(0.0, 1.0 - u_z * u_z))
            phi = 2.399963 * k
            vx = 1.02 * sin_th * math.cos(phi)
            vy = 1.02 * sin_th * math.sin(phi)
            vz = 1.02 * u_z
            nx, ny, nz = float(k) * 3.8, 0.0, 0.0
            pdb_lines.append(
                f"ATOM  {atom_serial:5d}  N   {res_name} A{k:4d}    {nx:8.3f}{ny:8.3f}{nz:8.3f}  1.00 10.00           N"
            )
            atom_serial += 1
            pdb_lines.append(
                f"ATOM  {atom_serial:5d}  CA  {res_name} A{k:4d}    {nx + 1.45:8.3f}{ny:8.3f}{nz:8.3f}  1.00 10.00           C"
            )
            atom_serial += 1
            pdb_lines.append(
                f"ATOM  {atom_serial:5d}  C   {res_name} A{k:4d}    {nx + 2.50:8.3f}{ny + 1.10:8.3f}{nz:8.3f}  1.00 10.00           C"
            )
            atom_serial += 1
            if res_name != "PRO":
                pdb_lines.append(
                    f"ATOM  {atom_serial:5d}  H   {res_name} A{k:4d}    {nx + vx:8.3f}{ny + vy:8.3f}{nz + vz:8.3f}  1.00 10.00           H"
                )
                atom_serial += 1

                # Vary S^2 across core (0.86..0.90) and flexible loop/tail (0.52..0.72)
                if k in (1, 2, n_res - 1, n_res):
                    s2_k = 0.54 + 0.03 * (k % 3)
                    te_ps = 45.0
                elif k in (10, 11, 35, 36):
                    s2_k = 0.70
                    te_ps = 30.0
                else:
                    s2_k = 0.84 + 0.05 * math.sin(0.4 * k)
                    te_ps = 12.0

                r1_hi, r2_hi, noe_hi = _woessner_lipari_szabo_15n_rates(
                    600.13, tau_m_true, aniso_true, u_z, s2_k, te_ps
                )
                r1_lo, r2_lo, noe_lo = _woessner_lipari_szabo_15n_rates(
                    360.13, tau_m_true, aniso_true, u_z, s2_k, te_ps
                )
                # Store as relaxation times T = 1/R in seconds (matching BMRB:6470)
                t1_hi, t2_hi = 1.0 / r1_hi, 1.0 / r2_hi
                t1_lo, t2_lo = 1.0 / r1_lo, 1.0 / r2_lo
                t1_600_rows.append(f"  {k} {res_name} N {t1_hi:.6f} {0.015 * t1_hi:.6f}")
                t2_600_rows.append(f"  {k} {res_name} N {t2_hi:.6f} {0.015 * t2_hi:.6f}")
                noe_600_rows.append(f"  {k} {res_name} N {k} {res_name} H {noe_hi:.5f} 0.025")
                t1_360_rows.append(f"  {k} {res_name} N {t1_lo:.6f} {0.015 * t1_lo:.6f}")
                t2_360_rows.append(f"  {k} {res_name} N {t2_lo:.6f} {0.015 * t2_lo:.6f}")
                noe_360_rows.append(f"  {k} {res_name} N {k} {res_name} H {noe_lo:.5f} 0.025")
                s2_rows.append(f"  {k} {res_name} N {s2_k:.5f} 0.020")

        # Add a sidechain NE1 decoy row to prove sidechains never clobber backbone N
        t1_600_rows.append("  5 ALA NE1 1.450000 0.050000")
        pdb_lines.append("ENDMDL\nEND\n")
        pdb_path = tmp / "synth_1d3z.pdb"
        pdb_path.write_text("\n".join(pdb_lines), encoding="utf-8")

        # Deliberately stamp BOTH saveframes with Spectrometer_frequency_1H = 600 like BMRB:6470!
        star_text = f"""data_6470
save_T1_field_A
  _Heteronucl_T1_list.Sf_category heteronucl_T1_relaxation
  _Heteronucl_T1_list.Spectrometer_frequency_1H 600
  _Heteronucl_T1_list.T1_val_units s
  loop_
    _T1.Comp_index_ID _T1.Comp_ID _T1.Atom_ID _T1.Val _T1.Val_err
{chr(10).join(t1_600_rows)}
  stop_
save_
save_T1_field_B
  _Heteronucl_T1_list.Sf_category heteronucl_T1_relaxation
  _Heteronucl_T1_list.Spectrometer_frequency_1H 600
  _Heteronucl_T1_list.T1_val_units s
  loop_
    _T1.Comp_index_ID _T1.Comp_ID _T1.Atom_ID _T1.Val _T1.Val_err
{chr(10).join(t1_360_rows)}
  stop_
save_
save_T2_field_A
  _Heteronucl_T2_list.Sf_category heteronucl_T2_relaxation
  _Heteronucl_T2_list.Spectrometer_frequency_1H 600
  _Heteronucl_T2_list.T2_val_units s
  loop_
    _T2.Comp_index_ID _T2.Comp_ID _T2.Atom_ID _T2.Val _T2.Val_err
{chr(10).join(t2_600_rows)}
  stop_
save_
save_T2_field_B
  _Heteronucl_T2_list.Sf_category heteronucl_T2_relaxation
  _Heteronucl_T2_list.Spectrometer_frequency_1H 600
  _Heteronucl_T2_list.T2_val_units s
  loop_
    _T2.Comp_index_ID _T2.Comp_ID _T2.Atom_ID _T2.Val _T2.Val_err
{chr(10).join(t2_360_rows)}
  stop_
save_
save_NOE_field_A
  _Heteronucl_NOE_list.Sf_category heteronucl_NOEs
  _Heteronucl_NOE_list.Spectrometer_frequency_1H 600
  loop_
    _Heteronucl_NOE.Comp_index_ID_1 _Heteronucl_NOE.Comp_ID_1 _Heteronucl_NOE.Atom_ID_1
    _Heteronucl_NOE.Comp_index_ID_2 _Heteronucl_NOE.Comp_ID_2 _Heteronucl_NOE.Atom_ID_2
    _Heteronucl_NOE.Val _Heteronucl_NOE.Val_err
{chr(10).join(noe_600_rows)}
  stop_
save_
save_NOE_field_B
  _Heteronucl_NOE_list.Sf_category heteronucl_NOEs
  _Heteronucl_NOE_list.Spectrometer_frequency_1H 600
  loop_
    _Heteronucl_NOE.Comp_index_ID_1 _Heteronucl_NOE.Comp_ID_1 _Heteronucl_NOE.Atom_ID_1
    _Heteronucl_NOE.Comp_index_ID_2 _Heteronucl_NOE.Comp_ID_2 _Heteronucl_NOE.Atom_ID_2
    _Heteronucl_NOE.Val _Heteronucl_NOE.Val_err
{chr(10).join(noe_360_rows)}
  stop_
save_
save_S2_parameters
  _Order_parameter_list.Sf_category order_parameters
  loop_
    _Order_param.Comp_index_ID _Order_param.Comp_ID _Order_param.Atom_ID _Order_param.Order_param_val _Order_param.Order_param_val_err
{chr(10).join(s2_rows)}
  stop_
save_
"""
        star_path = tmp / "bmr6470_synth.str"
        star_path.write_text(star_text, encoding="utf-8")

        # Verify --only-600-mhz option as well as full dual-field extraction
        only600 = export_f10_modelfree_tsvs(star_path, tmp / "only600", only_600_mhz=True)
        assert only600["fields_mhz"] == [600.13] and len(only600["written_files"]) == 3

        rc = f10_nmrstar_to_relax_tsv.main(["--star", str(star_path), "--out-dir", str(tmp / "rates")])
        assert rc == 0

        mf_out = tmp / "ubq_modelfree.tsv"
        rsd_out = tmp / "ubq_rsd.tsv"
        proc = _run_nmrassign(
            nmrassign_bin,
            [
                "modelfree",
                "--r1",
                str(tmp / "rates/ubq_600_r1.tsv"),
                "--r2",
                str(tmp / "rates/ubq_600_r2.tsv"),
                "--hetnoe",
                str(tmp / "rates/ubq_600_noe.tsv"),
                "--r1",
                str(tmp / "rates/ubq_360_r1.tsv"),
                "--r2",
                str(tmp / "rates/ubq_360_r2.tsv"),
                "--hetnoe",
                str(tmp / "rates/ubq_360_noe.tsv"),
                "--pdb",
                str(pdb_path),
                "--diffusion",
                "axial",
                "--criterion",
                "aic",
                "--r-nh",
                "1.02",
                "--csa-ppm",
                "-172.0",
                "--mc-trials",
                "25",
                "--seed",
                "20261303",
                "--rsd-out",
                str(rsd_out),
                "--out",
                str(mf_out),
            ],
        )
        res = evaluate_f10_modelfree(
            mf_out.read_text(encoding="utf-8"),
            star_text,
            stdout_text=proc.stdout,
            exit_code=proc.returncode,
        )
        assert res["verdict"] == "PASS", f"F10 failed: {json.dumps(res, indent=2)}"
        sm = res["secondary_metrics"]
        return (
            f"PASS (tau_m={sm['tau_m_ns']:.3f}ns, D_par/D_perp={sm['d_par_over_d_perp']:.3f}, "
            f"r_S2={res['criteria_results'][3]['value']:.4f}, MAE_S2={res['criteria_results'][4]['value']:.4f})"
        )


# ============================================================================
# 4. Gate F11: Bruker vclist Diagnostic + `nmrassign relax` Series Evaluation
# ============================================================================


def test_f11_adapter_vclist_and_nmrassign_relax(nmrassign_bin: Path) -> str:
    # 1. Verify Bruker hsqct2etf3gpsi3d vclist timing helper (`P[4] == 0 -> 2*P[3]`)
    d_vals = ["0.001"] * 64
    d_vals[20] = "0.000900"
    p_vals = ["0.0"] * 64
    p_vals[3] = "40.0"
    p_vals[4] = "0.0"
    acqus_synth = (
        "##TITLE= synth_hsqct2\n"
        f"##$D= (0..63)\n{' '.join(d_vals)}\n"
        f"##$P= (0..63)\n{' '.join(p_vals)}\n"
        "##END=\n"
    )
    vc_res = compute_bruker_hsqct2_vclist_delays(
        acqus_synth,
        "1\n2\n4\n8\n",
        pulseprogram_text="10 (p4 ph1):f2\n   lo to 10 times c\n",
    )
    expected_block = 16.0 * 0.0009 + 8.0 * (80.0 * 1.0e-6)
    assert math.isclose(vc_res["tau_cpmg_block_s"], expected_block, rel_tol=0.0, abs_tol=1e-12)
    assert vc_res["pulseprogram_has_vc_loop"] is True

    # 2. Synthesize 35-residue R1, R2, and hetNOE series TSVs + BMRB STAR and invoke `nmrassign relax`
    with tempfile.TemporaryDirectory(prefix="p17_selftest_f11_") as tmp_s:
        tmp = Path(tmp_s)
        rng = random.Random(20261302)
        delays_r1 = [0.02, 0.08, 0.16, 0.32, 0.64, 1.00]
        delays_r2 = [0.016, 0.032, 0.064, 0.096, 0.128, 0.160]

        def _make_series_tsv(kind: str, x_vals: Sequence[float], residue_traces: Sequence[Tuple[int, str, Sequence[float]]]) -> str:
            hdr = [
                f"# name: synth_{kind}",
                f"# experiment_kind: {kind}",
                "# field_mhz: 600.130000",
                "# schedule_kind: vdlist",
                "# duplicate_variance_scale: 1.000000",
                "# duplicate_dof: 0",
                "seq_id\tres_name\tatom_h\tatom_x\tdelta_h_ppm\tdelta_x_ppm\tsigma_h_ppm\tsigma_x_ppm\teta_h\teta_x\toverlap_cluster_id\tplane_idx\tx_value\tx_extra\tintensity\tsigma_intensity\tvolume\tplane_mad\tis_duplicate",
            ]
            for sid, rname, amps in residue_traces:
                for p_idx, (xv, amp) in enumerate(zip(x_vals, amps)):
                    hdr.append(
                        f"{sid}\t{rname}\tH\tN\t8.10000\t120.00000\t0.02000\t0.15000\t0.20000\t0.20000\t{sid}\t{p_idx}\t{xv:.9f}\t.\t{amp:.6e}\t1.500000e+01\t{amp:.6e}\t1.000000e+01\t0"
                    )
            return "\n".join(hdr) + "\n"

        r1_traces, r2_traces, noe_traces = [], [], []
        shift_rows, t1_rows, t2_rows, noe_rows = [], [], [], []
        for sid in range(1, 36):
            rname = "ALA" if sid % 2 == 0 else "VAL"
            r1_true = 1.4 + 0.03 * sid
            r2_true = 8.5 + 0.18 * sid
            noe_true = 0.55 + 0.008 * sid
            r1_amps = [10000.0 * math.exp(-r1_true * t) + rng.gauss(0.0, 12.0) for t in delays_r1]
            r2_amps = [10000.0 * math.exp(-r2_true * t) + rng.gauss(0.0, 12.0) for t in delays_r2]
            noe_amps = [10000.0 + rng.gauss(0.0, 10.0), 10000.0 * noe_true + rng.gauss(0.0, 10.0)]
            r1_traces.append((sid, rname, r1_amps))
            r2_traces.append((sid, rname, r2_amps))
            noe_traces.append((sid, rname, noe_amps))
            shift_rows.append(f"  {sid} {rname} H {8.0 + 0.02 * sid:.3f}\n  {sid} {rname} N {115.0 + 0.3 * sid:.3f}")
            t1_rows.append(f"  {sid} {rname} N {1.0 / r1_true:.6f} 0.010")
            t2_rows.append(f"  {sid} {rname} N {1.0 / r2_true:.6f} 0.002")
            noe_rows.append(f"  {sid} {rname} N {noe_true:.5f} 0.020")

        (tmp / "r1_series.tsv").write_text(_make_series_tsv("R1", delays_r1, r1_traces), encoding="utf-8")
        (tmp / "r2_series.tsv").write_text(_make_series_tsv("R2", delays_r2, r2_traces), encoding="utf-8")
        (tmp / "noe_series.tsv").write_text(_make_series_tsv("hetNOE", [0.0, 3.0], noe_traces), encoding="utf-8")

        _run_nmrassign(
            nmrassign_bin,
            ["relax", "--series", str(tmp / "r1_series.tsv"), "--kind", "r1", "--engine", "cpu", "--mc-trials", "40", "--seed", "20261302", "--out", str(tmp / "r1_rates.tsv")],
        )
        _run_nmrassign(
            nmrassign_bin,
            ["relax", "--series", str(tmp / "r2_series.tsv"), "--kind", "r2", "--engine", "cpu", "--mc-trials", "40", "--seed", "20261302", "--out", str(tmp / "r2_rates.tsv")],
        )
        _run_nmrassign(
            nmrassign_bin,
            ["relax", "--series", str(tmp / "noe_series.tsv"), "--kind", "hetnoe", "--mc-trials", "40", "--seed", "20261302", "--out", str(tmp / "noe_rates.tsv")],
        )

        star_18380 = f"""data_18380
save_chem_shifts
  loop_
    _Atom_chem_shift.Seq_ID _Atom_chem_shift.Comp_ID _Atom_chem_shift.Atom_ID _Atom_chem_shift.Val
{chr(10).join(shift_rows)}
  stop_
save_
save_T1
  _Heteronucl_T1_list.Sf_category heteronucl_T1_relaxation
  _Heteronucl_T1_list.Spectrometer_frequency_1H 600
  _Heteronucl_T1_list.T1_val_units s
  loop_
    _T1.Comp_index_ID _T1.Comp_ID _T1.Atom_ID _T1.Val _T1.Val_err
{chr(10).join(t1_rows)}
  stop_
save_
save_T2
  _Heteronucl_T2_list.Sf_category heteronucl_T2_relaxation
  _Heteronucl_T2_list.Spectrometer_frequency_1H 600
  _Heteronucl_T2_list.T2_val_units s
  loop_
    _T2.Comp_index_ID _T2.Comp_ID _T2.Atom_ID _T2.Val _T2.Val_err
{chr(10).join(t2_rows)}
  stop_
save_
save_NOE
  _Heteronucl_NOE_list.Sf_category heteronucl_NOEs
  _Heteronucl_NOE_list.Spectrometer_frequency_1H 600
  loop_
    _Heteronucl_NOE.Comp_index_ID _Heteronucl_NOE.Comp_ID _Heteronucl_NOE.Atom_ID _Heteronucl_NOE.Val _Heteronucl_NOE.Val_err
{chr(10).join(noe_rows)}
  stop_
save_
"""
        res = evaluate_f11_relax(
            (tmp / "r1_rates.tsv").read_text(encoding="utf-8"),
            (tmp / "r2_rates.tsv").read_text(encoding="utf-8"),
            (tmp / "noe_rates.tsv").read_text(encoding="utf-8"),
            star_18380,
        )
        assert res["verdict"] == "PASS", f"F11 failed: {json.dumps(res, indent=2)}"
        return (
            f"PASS (n={res['criteria_results'][0]['value']:.0f}, "
            f"r_R1={res['criteria_results'][1]['value']:.4f}, r_R2={res['criteria_results'][3]['value']:.4f})"
        )


# ============================================================================
# 5. Gate F13a: Plain-Text/Zip CEST -> 11-Col TSV + `nmrassign cest`
# ============================================================================


def _cest_2state_analytical_approx(
    offset_ppm: float,
    major_ppm: float,
    dw_ppm: float,
    p_b: float,
    k_ex: float,
    larmor_mhz: float,
    b1_hz: float,
    t_ex_s: float,
    r1: float = 1.5,
    r2a: float = 12.0,
    r2b: float = 18.0,
) -> float:
    """Baldwin-Kay off-resonance R1rho saturation approximation `exp(-R1rho * T_ex)` for synthetic test profiles."""
    w1 = 2.0 * math.pi * b1_hz
    da = 2.0 * math.pi * larmor_mhz * (major_ppm - offset_ppm)
    db = 2.0 * math.pi * larmor_mhz * (major_ppm + dw_ppm - offset_ppm)
    weff_sq = da * da + w1 * w1
    if weff_sq <= 1e-12:
        return 0.02
    sin2 = (w1 * w1) / weff_sq
    cos2 = 1.0 - sin2
    p_a = 1.0 - p_b
    rex = p_a * p_b * ((db - da) ** 2) * k_ex / (k_ex * k_ex + db * db + w1 * w1)
    r1rho = r1 * cos2 + (r2a + rex + p_b * (r2b - r2a)) * sin2
    return math.exp(-r1rho * t_ex_s)


def test_f13a_cest_adapter_and_nmrassign_cest(nmrassign_bin: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="p17_selftest_f13a_") as tmp_s:
        tmp = Path(tmp_s)
        pb_true = 0.030
        kex_true = 160.0
        larmor = 81.08
        t_ex = 0.40
        sites = [(12, "ALA", 118.0, 3.6), (25, "LEU", 121.5, -3.2), (41, "VAL", 116.2, 4.4)]
        offsets = [111.0 + 0.35 * k for k in range(44)]

        zip_path = tmp / "CEST_synth.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for b1_hz in (15.0, 30.0):
                zf.writestr(
                    f"b1_{int(b1_hz)}/exp.toml",
                    f"larmor_mhz = {larmor}\nt_ex = {t_ex}\nb1_frq = {b1_hz}\nb1_inh = 0.0\n",
                )
                for sid, rname, maj_ppm, dw_ppm in sites:
                    rows = ["# offset_ppm intensity uncertainty", "10000.0  5000.0  25.0"]
                    for off in offsets:
                        inorm = _cest_2state_analytical_approx(
                            off, maj_ppm, dw_ppm, pb_true, kex_true, larmor, b1_hz, t_ex
                        )
                        rows.append(f"{off:.4f}  {5000.0 * inorm:.4f}  30.0")
                    zf.writestr(f"b1_{int(b1_hz)}/{sid}{rname}.txt", "\n".join(rows) + "\n")

        cest_tsv = tmp / "cest_profiles.tsv"
        rc = f13a_cest_to_tsv.main(["--in", str(zip_path), "--out", str(cest_tsv)])
        assert rc == 0

        fit_json = tmp / "cest_fit_cpu.json"
        fit_tsv = tmp / "cest_fit_cpu.tsv"
        _run_nmrassign(
            nmrassign_bin,
            [
                "cest",
                "--in",
                str(cest_tsv),
                "--states",
                "2",
                "--b1-quad",
                "1",
                "--max-iters",
                "45",
                "--engine",
                "cpu",
                "--out",
                str(fit_tsv),
                "--json-out",
                str(fit_json),
            ],
        )
        dw_refs = {sid: dw for sid, _, _, dw in sites}
        res = evaluate_f13a_cest(
            fit_json.read_text(encoding="utf-8"),
            p_b_ref=pb_true,
            k_ex_ref_s_inv=kex_true,
            dw_ref_ppm_by_seq_id=dw_refs,
            nucleus="15N",
        )
        assert res["verdict"] == "PASS", f"F13a failed: {json.dumps(res, indent=2)}"
        sm = res["secondary_metrics"]
        return f"PASS (p_B={sm['p_b_fit']:.4f}, k_ex={sm['k_ex_fit_s_inv']:.1f}s^-1, med_dw_err={res['criteria_results'][2]['value']:.3f}ppm)"


# ============================================================================
# 6. Gate F13b: Multi-Field CPMG & Raw Planes + `nmrassign cpmg`
# ============================================================================


def _carver_richards_r2eff(
    nu_cpmg_hz: float,
    r20: float,
    p_b: float,
    k_ex: float,
    dw_ppm: float,
    larmor_mhz: float,
) -> float:
    """Exact Carver-Richards (1972) two-state CPMG R2,eff (equal R20A = R20B = r20)."""
    p_a = 1.0 - p_b
    dw = 2.0 * math.pi * larmor_mhz * dw_ppm
    tau_cp = 1.0 / (4.0 * nu_cpmg_hz)
    psi = k_ex * k_ex - dw * dw
    zeta = -2.0 * dw * (p_a - p_b) * k_ex
    hypot_pz = math.hypot(psi, zeta)
    d_plus = 0.5 * (1.0 + (psi + 2.0 * dw * dw) / hypot_pz)
    d_minus = 0.5 * (-1.0 + (psi + 2.0 * dw * dw) / hypot_pz)
    eta_plus = tau_cp * math.sqrt(max(0.0, 2.0 * (psi + hypot_pz)))
    eta_minus = tau_cp * math.sqrt(max(0.0, 2.0 * (-psi + hypot_pz)))
    arg = d_plus * math.cosh(eta_plus) - d_minus * math.cos(eta_minus)
    arg_clamped = max(1.0, arg)
    return r20 + 0.5 * (k_ex - (0.5 / tau_cp) * math.acosh(arg_clamped))


def test_f13b_cpmg_adapter_and_nmrassign_cpmg(nmrassign_bin: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="p17_selftest_f13b_") as tmp_s:
        tmp = Path(tmp_s)
        pb_true = 0.035
        kex_true = 950.0
        t_rel = 0.040
        nus = [25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0]
        sites = [
            (10, "ILE", 2.2),
            (18, "LEU", 3.1),
            (27, "VAL", 1.8),
            (39, "ALA", 2.7),
            (52, "THR", 3.5),
        ]

        # 1. Zip of multi-field R2,eff profiles (`60.81` & `81.08` MHz 15N)
        zip_path = tmp / "CPMG_synth.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for lmhz, r20 in ((60.81, 11.0), (81.08, 13.0)):
                for sid, rname, dw_ppm in sites:
                    lines = [f"# larmor_mhz = {lmhz}", f"# t_relax_s = {t_rel}"]
                    for nu in nus:
                        r2e = _carver_richards_r2eff(nu, r20, pb_true, kex_true, dw_ppm, lmhz)
                        lines.append(f"{nu:.2f}  {r2e:.5f}  0.20")
                    zf.writestr(f"f_{int(lmhz)}/{sid}{rname}.txt", "\n".join(lines) + "\n")

        cpmg_tsv = tmp / "cpmg_series.tsv"
        assert f13b_cpmg_to_tsv.main(["--in", str(zip_path), "--out", str(cpmg_tsv)]) == 0

        fit_json = tmp / "cpmg_fit_cpu.json"
        fit_tsv = tmp / "cpmg_fit_cpu.tsv"
        _run_nmrassign(
            nmrassign_bin,
            [
                "cpmg",
                "--in",
                str(cpmg_tsv),
                "--model",
                "baldwin2014",
                "--min-rex",
                "2.0",
                "--min-daic",
                "2.0",
                "--max-fp",
                "0.05",
                "--engine",
                "cpu",
                "--out",
                str(fit_tsv),
                "--json-out",
                str(fit_json),
            ],
        )

        # 2. Also test wide raw-intensity plane table (`ZENODO:6605011` mode) + `--raw-planes` CLI
        wide_hdr = ["Residue", "I0", *[f"{int(nu)}" for nu in nus], "noise"]
        wide_lines = [" ".join(wide_hdr)]
        for sid, rname, dw_ppm in sites[:2]:
            vals = [f"{rname}{sid}", "10000.0"]
            for nu in nus:
                r2e = _carver_richards_r2eff(nu, 11.0, pb_true, kex_true, dw_ppm, 60.81)
                vals.append(f"{10000.0 * math.exp(-r2e * t_rel):.3f}")
            vals.append("15.0")
            wide_lines.append(" ".join(vals))
        raw_src = tmp / "D25_MQCPMG_600_ints"
        raw_src.write_text("\n".join(wide_lines) + "\n", encoding="utf-8")
        raw_tsv = tmp / "cpmg_600_planes.tsv"
        assert (
            f13b_cpmg_to_tsv.main(
                [
                    "--in",
                    str(raw_src),
                    "--out",
                    str(raw_tsv),
                    "--raw-planes",
                    "--larmor-mhz",
                    "60.81",
                    "--t-relax-s",
                    "0.04",
                ]
            )
            == 0
        )
        _run_nmrassign(
            nmrassign_bin,
            [
                "cpmg",
                "--raw-planes",
                str(raw_tsv),
                "--series-out",
                str(tmp / "hts_r2eff.tsv"),
                "--model",
                "baldwin2014",
                "--engine",
                "cpu",
                "--out",
                str(tmp / "hts_fit.tsv"),
                "--json-out",
                str(tmp / "hts_fit.json"),
            ],
        )

        # 3. Verify secondary R1rho rule (returns None if ground-state shift table is absent)
        assert convert_f13b_r1rho_to_tsv([{"seq_id": 10, "carrier_ppm": 118.0, "spin_lock_hz": 500.0, "r1rho_s_inv": 14.0, "r1rho_err": 0.3}], None) is None
        r1rho_map = convert_f13b_r1rho_to_tsv(
            [{"seq_id": 10, "res_name": "ILE", "carrier_ppm": 118.0, "spin_lock_hz": 500.0, "r1rho_s_inv": 14.0, "r1rho_err": 0.3}],
            {10: 120.5},
        )
        assert r1rho_map is not None and 10 in r1rho_map

        res = evaluate_f13b_cpmg(
            fit_json.read_text(encoding="utf-8"),
            p_b_ref=pb_true,
            k_ex_ref_s_inv=kex_true,
            dw_ref_ppm_by_seq_id={sid: dw for sid, _, dw in sites},
        )
        assert res["verdict"] == "PASS", f"F13b failed: {json.dumps(res, indent=2)}"
        sm = res["secondary_metrics"]
        return f"PASS (p_B={sm['p_b_fit']:.4f}, k_ex={sm['k_ex_fit_s_inv']:.1f}s^-1, n_sites={sm['n_gated_matched_sites']})"


# ============================================================================
# 7. Gate F13c: Unassigned Apo Peaks Reshaper + Quadratic `nmrassign csp`
# ============================================================================


def _morrison_bound_fraction(p_um: float, l_um: float, kd_um: float) -> float:
    if p_um <= 0.0:
        return 0.0
    b = p_um + l_um + kd_um
    disc = max(0.0, b * b - 4.0 * p_um * l_um)
    return (b - math.sqrt(disc)) / (2.0 * p_um)


def test_f13c_csp_adapter_and_nmrassign_csp(nmrassign_bin: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="p17_selftest_f13c_") as tmp_s:
        tmp = Path(tmp_s)
        p0_um = 100.0
        kd_true_um = 45.0
        l_steps = [0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]

        # 8 pocket residues + 52 non-binding core residues (realistic ~13% binding interface)
        apo_coords: List[Tuple[float, float, float, float]] = []
        for idx in range(60):
            h0 = 6.80 + 0.045 * idx
            n0 = 106.0 + 0.42 * idx
            if idx < 8:
                dh_max = 0.18 + 0.02 * (idx % 3)
                dn_max = 1.00 + 0.12 * (idx % 4)
            else:
                dh_max = 0.001 + 0.0005 * (idx % 5)
                dn_max = 0.005 + 0.0020 * (idx % 5)
            apo_coords.append((h0, n0, dh_max, dn_max))

        # Write `nmrassign pick --hsqc` format raw apo peaks (`experiment\tH\tN\tC\tsign\theight`)
        raw_pick_lines = ["experiment\tH\tN\tC\tsign\theight"]
        for h0, n0, _, _ in apo_coords:
            raw_pick_lines.append(f"HSQC\t{h0:.4f}\t{n0:.4f}\t0.0000\t1\t1.000e+06")
        raw_apo_path = tmp / "f13c_raw_apo_peaks.tsv"
        raw_apo_path.write_text("\n".join(raw_pick_lines) + "\n", encoding="utf-8")

        apo_4col_path = tmp / "apo_peaks.tsv"
        assert f13c_apo_peaks_to_tsv.main(["--in", str(raw_apo_path), "--out", str(apo_4col_path)]) == 0

        meta_rows = ["ligand_um,protein_um,spectrum_path"]
        for s_idx, l_um in enumerate(l_steps):
            fb = _morrison_bound_fraction(p0_um, l_um, kd_true_um)
            step_lines = ["experiment\tH\tN\tC\tsign\theight"]
            for h0, n0, dh, dn in apo_coords:
                step_lines.append(f"HSQC\t{h0 + fb * dh:.5f}\t{n0 + fb * dn:.5f}\t0.0\t1\t1.0e6")
            sp_file = tmp / f"step_{s_idx}.tsv"
            sp_file.write_text("\n".join(step_lines) + "\n", encoding="utf-8")
            meta_rows.append(f"{l_um},{p0_um},{sp_file.name}")

        step_info = build_f13c_titration_step_specs("\n".join(meta_rows), base_dir=tmp)
        kd_ref = extract_f13c_reference_kd_um("ITC global 1:1 fit summary: K_d = 45.0 uM (N = 0.98)")
        assert math.isclose(kd_ref, kd_true_um)

        csp_args: List[str] = [
            "csp",
            "--apo-peaks",
            str(apo_4col_path),
            "--protein-um",
            str(step_info["protein_um"]),
        ]
        for st in step_info["step_cli_specs"]:
            csp_args.extend(["--step", st])
        csp_json = tmp / "csp_fit.json"
        csp_tsv = tmp / "csp_fit.tsv"
        csp_traj = tmp / "tracked_csp.tsv"
        csp_args.extend(
            [
                "--experiment",
                "hsqc",
                "--snr",
                "5.0",
                "--max-step-csp",
                "0.45",
                "--csp-sigma",
                "0.004",
                "--mc-trials",
                "80",
                "--seed",
                "20261306",
                "--trajectories-out",
                str(csp_traj),
                "--json-out",
                str(csp_json),
                "--out",
                str(csp_tsv),
            ]
        )
        _run_nmrassign(nmrassign_bin, csp_args)
        res = evaluate_f13c_csp(csp_json.read_text(encoding="utf-8"), kd_ref_um=kd_ref)
        assert res["verdict"] == "PASS", f"F13c failed: {json.dumps(res, indent=2)}"
        sm = res["secondary_metrics"]
        return f"PASS (n_pocket={res['criteria_results'][0]['value']:.0f}, Kd_fit={sm['global_kd_um']:.2f}uM vs {kd_ref:.1f}uM)"


# ============================================================================
# 8. Gate F14: Raw Bruker 1D FIDs (1H/13C/DEPT90/DEPT135/19F/31P) & qNMR
# ============================================================================


def _write_synthetic_bruker_1d_dir(
    out_dir: Path,
    *,
    nuc1: str,
    pulprog: str,
    obs_mhz: float,
    sw_hz: float,
    carrier_ppm: float,
    peaks: Sequence[Tuple[float, float, float]],
    n_cplx: int = 2048,
    noise_sigma: float = 0.002,
    phase0_deg: float = 0.0,
    solvent: str = "D2O",
    seed: int = 42,
) -> Path:
    """Synthesize a clean little-endian `DTYPA=2` (`f64`) Bruker 1D `fid` + `acqus` directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sfo1 = obs_mhz * (1.0 + carrier_ppm * 1.0e-6)
    td_words = 2 * n_cplx
    dt = 1.0 / sw_hz
    phi0 = math.radians(phase0_deg)

    buf = bytearray()
    for n in range(n_cplx):
        t = n * dt
        re_v = rng.gauss(0.0, noise_sigma)
        im_v = rng.gauss(0.0, noise_sigma)
        for ppm, amp, r2_hz in peaks:
            df_hz = (ppm - carrier_ppm) * obs_mhz
            decay = amp * math.exp(-math.pi * r2_hz * t)
            ang = 2.0 * math.pi * df_hz * t + phi0
            re_v += decay * math.cos(ang)
            im_v += decay * math.sin(ang)
        buf.extend(struct.pack("<2d", re_v, im_v))

    (out_dir / "fid").write_bytes(bytes(buf))
    acqus = (
        "##TITLE= synthetic_bruker_1d\n"
        "##JCAMP-DX= 5.00 Bruker JCAMP library\n"
        f"##$PULPROG= <{pulprog}>\n"
        f"##$SOLVENT= <{solvent}>\n"
        f"##$NUC1= <{nuc1}>\n"
        f"##$SFO1= {sfo1:.8f}\n"
        f"##$BF1= {obs_mhz:.8f}\n"
        f"##$SW_h= {sw_hz:.6f}\n"
        f"##$SW= {sw_hz / sfo1:.8f}\n"
        f"##$TD= {td_words}\n"
        "##$BYTORDA= 0\n"
        "##$DTYPA= 2\n"
        "##$GRPDLY= 0\n"
        "##$TE= 298.15\n"
        "##END=\n"
    )
    (out_dir / "acqus").write_text(acqus, encoding="utf-8")
    return out_dir


def test_f14_nmr1d_and_qnmr_adapter(nmrassign_bin: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="p17_selftest_f14_") as tmp_s:
        tmp = Path(tmp_s)

        d_1h = _write_synthetic_bruker_1d_dir(
            tmp / "bmse/1H",
            nuc1="1H",
            pulprog="zg30",
            obs_mhz=400.13,
            sw_hz=4800.0,
            carrier_ppm=4.70,
            peaks=[(5.20, 1.0, 1.5), (3.80, 2.0, 1.5), (1.22, 3.0, 1.5)],
            seed=101,
        )
        fucose_c_sites = [(97.1, 1.0, 6.0), (74.6, 1.0, 6.0), (72.7, 1.0, 6.0), (70.5, 1.0, 6.0), (69.1, 1.0, 6.0), (16.3, 1.2, 6.0)]
        d_13c = _write_synthetic_bruker_1d_dir(
            tmp / "bmse/13C",
            nuc1="13C",
            pulprog="zgpg30",
            obs_mhz=100.61,
            sw_hz=22000.0,
            carrier_ppm=100.0,
            peaks=fucose_c_sites,
            n_cplx=8192,
            seed=102,
        )
        d_dept90 = _write_synthetic_bruker_1d_dir(
            tmp / "bmse/DEPT_90",
            nuc1="13C",
            pulprog="dept90",
            obs_mhz=100.61,
            sw_hz=22000.0,
            carrier_ppm=100.0,
            peaks=fucose_c_sites[:5],
            n_cplx=8192,
            seed=103,
        )
        d_dept135 = _write_synthetic_bruker_1d_dir(
            tmp / "bmse/DEPT_135",
            nuc1="13C",
            pulprog="dept135",
            obs_mhz=100.61,
            sw_hz=22000.0,
            carrier_ppm=100.0,
            peaks=fucose_c_sites,
            n_cplx=8192,
            phase0_deg=30.0,
            seed=104,
        )
        _write_synthetic_bruker_1d_dir(
            tmp / "zenodo_het/exp_02_19f",
            nuc1="19F",
            pulprog="zgfhigqn",
            obs_mhz=376.50,
            sw_hz=40000.0,
            carrier_ppm=-75.0,
            peaks=[(-62.8, 2.5, 2.5)],
            seed=105,
        )
        _write_synthetic_bruker_1d_dir(
            tmp / "zenodo_het/exp_01_31p",
            nuc1="31P",
            pulprog="zgpg30",
            obs_mhz=161.98,
            sw_hz=20000.0,
            carrier_ppm=0.0,
            peaks=[(-10.4, 2.0, 2.5), (2.1, 1.5, 2.5)],
            seed=106,
        )

        sel_19f = select_first_bruker_1d_dir_by_nuc1(tmp / "zenodo_het", "19F")
        sel_31p = select_first_bruker_1d_dir_by_nuc1(tmp / "zenodo_het", "31P")
        assert read_bruker_acqus_nuc1(sel_19f / "acqus") == "19F"
        assert "--proton-decoupled" in multiplet_cli_flags_for_acqus(sel_19f / "acqus")
        assert "--proton-decoupled" not in multiplet_cli_flags_for_acqus(d_1h / "acqus")

        reports: Dict[str, Tuple[int, str]] = {}
        for key, fdir in [
            ("1h", d_1h),
            ("13c", d_13c),
            ("dept90", d_dept90),
            ("dept135", d_dept135),
            ("19f", sel_19f),
            ("31p", sel_31p),
        ]:
            sum_js = tmp / f"{key}_summary.json"
            spec_js = tmp / f"{key}_spec.json"
            p = _run_nmrassign(
                nmrassign_bin,
                [
                    "nmr1d",
                    "process",
                    "--in",
                    str(fdir),
                    "--experiment",
                    key,
                    "--zf",
                    "2",
                    "--auto-phase",
                    "--baseline",
                    "whittaker",
                    "--json-out",
                    str(sum_js),
                    "--out",
                    str(spec_js),
                ],
            )
            reports[key] = (p.returncode, sum_js.read_text(encoding="utf-8"))

        # Also exercise `nmrassign nmr1d multiplet` with adapter-generated flags
        _run_nmrassign(
            nmrassign_bin,
            [
                "nmr1d",
                "multiplet",
                "--in",
                str(tmp / "19f_spec.json"),
                *multiplet_cli_flags_for_acqus(sel_19f / "acqus"),
                "--style",
                "acs",
                "--out",
                str(tmp / "19f_mult.tsv"),
            ],
        )

        # Synthesize gravimetric qNMR test sample (`DMSO2` 6H @ 3.00 ppm, `Analyte` 2H @ 7.50 ppm)
        # Choose integral ratio matching 99.0% purity:
        # moles_std = (5.0 mg * 0.999) / 94.13 = 0.0530649 mmol -> signal_std ~ 6 * 0.0530649 = 0.318389
        # moles_an  = (9.9 mg) / 180.16 = 0.0549512 mmol (out of 10.0 mg weighed = 99.0% purity!) -> signal_an ~ 2 * 0.0549512 = 0.109902
        qnmr_dir = _write_synthetic_bruker_1d_dir(
            tmp / "qnmr_sample_1",
            nuc1="1H",
            pulprog="zg",
            obs_mhz=600.13,
            sw_hz=7200.0,
            carrier_ppm=4.70,
            peaks=[(3.00, 318.389, 1.2), (7.50, 109.902, 1.2)],
            n_cplx=8192,
            noise_sigma=0.0005,
            seed=107,
        )
        qnmr_spec_js = tmp / "qnmr_spec.json"
        _run_nmrassign(
            nmrassign_bin,
            [
                "nmr1d",
                "process",
                "--in",
                str(qnmr_dir),
                "--experiment",
                "1h",
                "--zf",
                "2",
                "--auto-phase",
                "--baseline",
                "whittaker",
                "--out",
                str(qnmr_spec_js),
            ],
        )

        dataguide_csv = (
            "sample_id,std_name,std_mw,std_protons,std_mass_mg,std_purity,std_ppm,"
            "analyte_name,analyte_mw,analyte_mass_mg,analyte_protons,analyte_ppm,ref_purity_percent\n"
            "qnmr_sample_1,DMSO2,94.13,6,5.0,0.999,3.00,Aspirin,180.16,10.0,2,7.50,99.0\n"
        )
        qspecs = extract_f14_qnmr_sample_specs(dataguide_csv)
        qnmr_out_js = tmp / "qnmr_res.json"
        pq = _run_nmrassign(
            nmrassign_bin,
            [
                "nmr1d",
                "qnmr",
                "--in",
                str(qnmr_spec_js),
                *qspecs[0]["cli_args"],
                "--json-out",
                str(qnmr_out_js),
                "--out",
                str(tmp / "qnmr_res.tsv"),
            ],
        )

        res = evaluate_f14_nmr1d_and_qnmr(
            reports,
            reports["dept135"][1],
            [
                {
                    "exit_code": pq.returncode,
                    "qnmr_json": qnmr_out_js.read_text(encoding="utf-8"),
                    "ref_purity_percent": qspecs[0]["ref_purity_percent"],
                }
            ],
        )
        assert res["verdict"] == "PASS", f"F14 failed: {json.dumps(res, indent=2)}"
        return (
            f"PASS (classes={res['criteria_results'][0]['value']:.0f}/5, "
            f"dept135_pos={res['criteria_results'][1]['value']:.2f}, "
            f"qnmr_rel_err={res['criteria_results'][2]['value']:.4f})"
        )


# ============================================================================
# 9. Gate F16: Bruker `ser`+`difflist` & Varian `fid`+`procpar` DOSY
# ============================================================================


def _stejskal_tanner_rect_b(g_gcm: float, big_delta_s: float, little_delta_s: float) -> float:
    gamma_h = 2.6752218744e8
    g_tm = g_gcm * 1.0e-2
    return (gamma_h * g_tm * little_delta_s) ** 2 * (big_delta_s - little_delta_s / 3.0)


def _build_synthetic_bruker_dosy_dir(
    exp_dir: Path,
    grads_gcm: Sequence[float],
    d_fast: float,
    d_slow: float,
) -> Path:
    exp_dir.mkdir(parents=True, exist_ok=True)
    n_cplx = 4096
    obs_mhz = 500.13
    sw_hz = 5000.0
    carrier_ppm = 4.70
    sfo1 = obs_mhz * (1.0 + carrier_ppm * 1.0e-6)
    big_d = 0.080
    lit_d = 0.0020  # P30 = 2000 us under stebpgp1s (monopolar STE, Rectangular default)

    raw_trace_bytes = n_cplx * 16
    stride = ((raw_trace_bytes + 1023) // 1024) * 1024
    ser_buf = bytearray()
    dt = 1.0 / sw_hz
    rng = random.Random(20261404)

    for g in grads_gcm:
        b_val = _stejskal_tanner_rect_b(g, big_d, lit_d)
        a_fast = 80.0 * math.exp(-b_val * d_fast)
        a_slow = 80.0 * math.exp(-b_val * d_slow)
        a_hdo = 120.0 * math.exp(-b_val * 1.9e-9)  # Residual HDO @ 4.79 ppm (must be masked!)
        plane = bytearray()
        for n in range(n_cplx):
            t = n * dt
            re_v = rng.gauss(0.0, 0.05)
            im_v = rng.gauss(0.0, 0.05)
            for ppm, amp in ((1.50, a_fast), (7.40, a_slow), (4.79, a_hdo)):
                df = (ppm - carrier_ppm) * obs_mhz
                dec = amp * math.exp(-math.pi * 4.0 * t)
                ang = 2.0 * math.pi * df * t
                re_v += dec * math.cos(ang)
                im_v += dec * math.sin(ang)
            plane.extend(struct.pack("<2d", re_v, im_v))
        if len(plane) < stride:
            plane.extend(b"\x00" * (stride - len(plane)))
        ser_buf.extend(plane)

    (exp_dir / "ser").write_bytes(bytes(ser_buf))
    (exp_dir / "difflist").write_text("\n".join(f"{g:.4f}" for g in grads_gcm) + "\n", encoding="utf-8")

    d_arr = ["0.0"] * 64
    d_arr[20] = f"{big_d:.6f}"
    p_arr = ["0.0"] * 64
    p_arr[30] = f"{lit_d * 1.0e6:.2f}"
    acqus = (
        "##TITLE= synth_bruker_dosy\n"
        "##$PULPROG= <stegp1s>\n"
        "##$SOLVENT= <D2O>\n"
        "##$NUC1= <1H>\n"
        f"##$SFO1= {sfo1:.8f}\n"
        f"##$BF1= {obs_mhz:.8f}\n"
        f"##$SW_h= {sw_hz:.6f}\n"
        f"##$SW= {sw_hz / sfo1:.8f}\n"
        f"##$TD= {2 * n_cplx}\n"
        "##$BYTORDA= 0\n"
        "##$DTYPA= 2\n"
        "##$GRPDLY= 0\n"
        "##$TE= 298.15\n"
        "##$GPNAM6= <RECT.1>\n"
        f"##$D= (0..63)\n{' '.join(d_arr)}\n"
        f"##$P= (0..63)\n{' '.join(p_arr)}\n"
        "##END=\n"
    )
    (exp_dir / "acqus").write_text(acqus, encoding="utf-8")
    return exp_dir


def _build_synthetic_varian_dosy_dir(
    exp_dir: Path,
    gzlvl_dac: Sequence[float],
    gcal: float,
    d_fast: float,
    d_slow: float,
) -> Path:
    exp_dir.mkdir(parents=True, exist_ok=True)
    n_cplx = 4096
    np_words = 2 * n_cplx
    ebytes = 4
    tbytes = np_words * ebytes
    bbytes = 28 + tbytes
    nblocks = len(gzlvl_dac)
    status = 0x0001 | 0x0004 | 0x0008  # S_DATA | S_32 | S_FLOAT

    sfrq = 500.13
    sw_hz = 5000.0
    carrier_ppm = 4.80
    # Choose rfl/rfp so (rfp - rfl + sw/2) / sfrq == carrier_ppm
    rfp = carrier_ppm * sfrq - 0.5 * sw_hz
    rfl = 0.0
    big_d = 0.080
    lit_d = 0.0020

    fid_buf = bytearray()
    fid_buf.extend(struct.pack(">6i2Hi", nblocks, 1, np_words, ebytes, tbytes, bbytes, 1, status, 1))
    dt = 1.0 / sw_hz
    rng = random.Random(20261405)

    for b_idx, dac in enumerate(gzlvl_dac):
        g_gcm = dac * gcal
        b_val = _stejskal_tanner_rect_b(g_gcm, big_d, lit_d)
        a_fast = 80.0 * math.exp(-b_val * d_fast)
        a_slow = 80.0 * math.exp(-b_val * d_slow)
        # 28-byte datablockhead (`scale, status, index, mode: i16`, `ctcount: i32`, `lpval..tlt: f32`)
        fid_buf.extend(struct.pack(">4hi4f", 1, status, b_idx + 1, 0, 1, 0.0, 0.0, 0.0, 0.0))
        for n in range(n_cplx):
            t = n * dt
            re_v = rng.gauss(0.0, 0.05)
            im_v = rng.gauss(0.0, 0.05)
            for ppm, amp in ((1.80, a_fast), (7.10, a_slow)):
                df = (ppm - carrier_ppm) * sfrq
                dec = amp * math.exp(-math.pi * 4.0 * t)
                ang = 2.0 * math.pi * df * t
                # Varian FIDs have opposite sign to Bruker before conjugation (`[cos, -sin]`)
                re_v += dec * math.cos(ang)
                im_v -= dec * math.sin(ang)
            fid_buf.extend(struct.pack(">2f", float(re_v), float(im_v)))

    (exp_dir / "fid").write_bytes(bytes(fid_buf))
    gz_str = " ".join(f"{v:.1f}" for v in gzlvl_dac)
    procpar = (
        'seqfil 2 2 8 0 0 2 1 0 1 64\n1 "dste"\n0\n'
        'solvent 2 2 8 0 0 2 1 0 1 64\n1 "d2o"\n0\n'
        'tn 2 2 8 0 0 2 1 0 1 64\n1 "H1"\n0\n'
        'temp 1 1 100 -100 0.1 2 1 1 1 64\n1 25.0\n0\n'
        f"sfrq 1 1 2000 0 0 2 1 1 1 64\n1 {sfrq:.6f}\n0\n"
        f"sw 1 1 100000 1 0 2 1 1 1 64\n1 {sw_hz:.4f}\n0\n"
        f"rfp 1 1 100000 -100000 0 2 1 1 1 64\n1 {rfp:.4f}\n0\n"
        f"rfl 1 1 100000 -100000 0 2 1 1 1 64\n1 {rfl:.4f}\n0\n"
        f"del 1 1 10 0 0 2 1 1 1 64\n1 {big_d:.6f}\n0\n"
        f"gt1 1 1 1 0 0 2 1 1 1 64\n1 {lit_d:.6f}\n0\n"
        f"gcal_ 1 1 1 0 0 2 1 1 1 64\n1 {gcal:.8f}\n0\n"
        'gshape 2 2 8 0 0 2 1 0 1 64\n1 "rect"\n0\n'
        f"gzlvl1 1 1 32767 -32768 1 2 1 1 1 64\n{len(gzlvl_dac)} {gz_str}\n0\n"
    )
    (exp_dir / "procpar").write_text(procpar, encoding="utf-8")
    return exp_dir


def test_f16_dosy_plane_integrator_and_nmrassign_dosy(nmrassign_bin: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="p17_selftest_f16_") as tmp_s:
        tmp = Path(tmp_s)
        d_fast = 6.8e-10
        d_slow = 1.6e-10
        grads = [2.0 + 4.2 * i for i in range(12)]
        dac_vals = [1000.0 + 2100.0 * i for i in range(12)]
        gcal = 0.0020

        b_dir = _build_synthetic_bruker_dosy_dir(tmp / "bruker_dosy", grads, d_fast, d_slow)
        v_dir = _build_synthetic_varian_dosy_dir(tmp / "varian_dosy", dac_vals, gcal, d_fast, d_slow)

        solv_b, temp_b = extract_dosy_header_solvent_and_temp(b_dir)
        solv_v, temp_v = extract_dosy_header_solvent_and_temp(v_dir)
        assert solv_b == "D2O" and math.isclose(temp_b, 298.15)
        assert solv_v == "D2O" and math.isclose(temp_v, 298.15)

        b_decay = tmp / "bruker_decay.tsv"
        v_decay = tmp / "varian_decay.tsv"
        assert (
            f16_dosy_plane_integrator.main(
                ["--exp-dir", str(b_dir), "--out", str(b_decay), "--nmrassign-bin", str(nmrassign_bin)]
            )
            == 0
        )
        assert (
            f16_dosy_plane_integrator.main(
                ["--exp-dir", str(v_dir), "--out", str(v_decay), "--nmrassign-bin", str(nmrassign_bin)]
            )
            == 0
        )

        # Confirm residual HDO at 4.79 ppm was masked out (leaving only the 2 analyte peaks!)
        assert len(b_decay.read_text(encoding="utf-8").strip().splitlines()) == 3

        b_fit_js = tmp / "bruker_fit.json"
        v_fit_js = tmp / "varian_fit.json"
        pb = _run_nmrassign(
            nmrassign_bin,
            [
                "dosy",
                "fit",
                "--in",
                str(b_decay),
                "--exp-dir",
                str(b_dir),
                "--engine",
                "cpu",
                "--criterion",
                "bic",
                "--max-components",
                "3",
                "--bootstrap",
                "30",
                "--seed",
                "20261404",
                "--solvent",
                solv_b,
                "--temp-k",
                f"{temp_b:.2f}",
                "--out",
                str(tmp / "bruker_fit.tsv"),
                "--json-out",
                str(b_fit_js),
            ],
        )
        pv = _run_nmrassign(
            nmrassign_bin,
            [
                "dosy",
                "fit",
                "--in",
                str(v_decay),
                "--exp-dir",
                str(v_dir),
                "--engine",
                "cpu",
                "--criterion",
                "bic",
                "--max-components",
                "3",
                "--bootstrap",
                "30",
                "--seed",
                "20261404",
                "--solvent",
                solv_v,
                "--temp-k",
                f"{temp_v:.2f}",
                "--out",
                str(tmp / "varian_fit.tsv"),
                "--json-out",
                str(v_fit_js),
            ],
        )

        res = evaluate_f16_dosy(
            {
                "bruker": {
                    "exit_code": pb.returncode,
                    "fit_json": b_fit_js.read_text(encoding="utf-8"),
                    "decay_tsv": b_decay.read_text(encoding="utf-8"),
                    "principal_references": [
                        {"label_or_ppm": 1.50, "d_ref_m2_s": d_fast},
                        {"label_or_ppm": 7.40, "d_ref_m2_s": d_slow},
                    ],
                    "ordering_pairs": [(1.50, 7.40)],
                },
                "varian": {
                    "exit_code": pv.returncode,
                    "fit_json": v_fit_js.read_text(encoding="utf-8"),
                    "decay_tsv": v_decay.read_text(encoding="utf-8"),
                    "principal_references": [
                        {"label_or_ppm": 1.80, "d_ref_m2_s": d_fast},
                        {"label_or_ppm": 7.10, "d_ref_m2_s": d_slow},
                    ],
                    "ordering_pairs": [(1.80, 7.10)],
                },
            }
        )
        assert res["verdict"] == "PASS", f"F16 failed: {json.dumps(res, indent=2)}"
        return (
            f"PASS (vendors={res['criteria_results'][0]['value']:.0f}/2, "
            f"min_R2={res['criteria_results'][1]['value']:.4f}, "
            f"max_dex_err={res['criteria_results'][3]['value']:.4f})"
        )


# ============================================================================
# 10. Gate F17: MTBLS1 ISA-Tab Manifest + `nmrassign metabolomics`
# ============================================================================


def _write_synthetic_urinary_spectrum_tsv(
    path: Path,
    *,
    is_diabetes: bool,
    rng: random.Random,
) -> None:
    n_pts = 2048
    hi_ppm, lo_ppm = 9.60, -0.20
    step = (lo_ppm - hi_ppm) / float(n_pts - 1)
    dilution = rng.uniform(0.8, 1.25)

    # Endogenous urinary peaks (TSP @ 0.00, creatinine @ 3.05 & 4.05, citrate @ 2.54/2.68, hippurate @ 7.60)
    peaks = [
        (0.00, 1.0, 0.006),
        (1.48, 0.6 + rng.gauss(0.0, 0.06), 0.008),
        (2.54, 1.4 + rng.gauss(0.0, 0.12), 0.010),
        (2.68, 1.4 + rng.gauss(0.0, 0.12), 0.010),
        (3.05, 2.5 + rng.gauss(0.0, 0.15), 0.009),
        (4.05, 1.8 + rng.gauss(0.0, 0.12), 0.009),
        (7.60, 0.9 + rng.gauss(0.0, 0.10), 0.012),
    ]
    # Glycosuria signal: alpha-D-glucose H1 @ 5.24 ppm + C2-C6 ring protons @ 3.46 & 3.74 ppm
    glc_level = rng.uniform(3.2, 5.5) if is_diabetes else rng.uniform(0.08, 0.25)
    peaks.extend(
        [
            (5.24, glc_level, 0.009),
            (3.46, 2.2 * glc_level, 0.012),
            (3.74, 2.4 * glc_level, 0.012),
        ]
    )

    lines = ["# obs_mhz=700.130000", "ppm\treal"]
    for i in range(n_pts):
        ppm = hi_ppm + i * step
        y = rng.gauss(0.0, 0.008)
        for c_ppm, amp, w_ppm in peaks:
            dx = (ppm - c_ppm) / w_ppm
            y += (amp / (1.0 + dx * dx)) * (1.0 if c_ppm == 0.0 else dilution)
        lines.append(f"{ppm:.6f}\t{y:.8e}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_f17_mtbls1_manifest_and_nmrassign_metabolomics(nmrassign_bin: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="p17_selftest_f17_") as tmp_s:
        tmp = Path(tmp_s)
        rng = random.Random(20261717)

        # Verify extracted Bruker sample directory locator
        sample_zip_dir = tmp / "extracted_sample/ADG10003u_007/10"
        sample_zip_dir.mkdir(parents=True, exist_ok=True)
        (sample_zip_dir / "acqus").write_text("##$NUC1= <1H>\n", encoding="utf-8")
        (sample_zip_dir / "fid").write_bytes(b"\x00" * 32)
        assert locate_extracted_bruker_1d_dir(tmp / "extracted_sample") == sample_zip_dir

        s_lines = ['"Sample Name"\t"Factor Value[Metabolic syndrome]"']
        a_lines = ['"Sample Name"\t"Raw Spectral Data File"']
        spectra_dir = tmp / "f17_spectra"
        n_ctrl, n_diab = 18, 14

        for k in range(1, n_ctrl + n_diab + 1):
            is_dm = k > n_ctrl
            sid = f"ADG1000{k:02d}u"
            grp = "diabetes mellitus" if is_dm else "Control Group"
            s_lines.append(f'"{sid}"\t"{grp}"')
            a_lines.append(f'"{sid}"\t"FILES/{sid}.zip"')
            # Add duplicate .nmrML rows (including duplicate ADG-19_007u_476.nmrML) to verify .nmrML filtering!
            a_lines.append(f'"{sid}"\t"FILES/ADG-19_007u_476.nmrML"')
            _write_synthetic_urinary_spectrum_tsv(
                spectra_dir / f"{sid}.tsv",
                is_diabetes=is_dm,
                rng=rng,
            )

        s_path = tmp / "s_MTBLS1.txt"
        a_path = tmp / "a_MTBLS1_metabolite_profiling_NMR_spectroscopy.txt"
        s_path.write_text("\n".join(s_lines) + "\n", encoding="utf-8")
        a_path.write_text("\n".join(a_lines) + "\n", encoding="utf-8")

        manifest_path = tmp / "mtbls1_manifest.tsv"
        assert (
            f17_mtbls1_manifest.main(
                [
                    "--samples-sheet",
                    str(s_path),
                    "--assay-sheet",
                    str(a_path),
                    "--spectra-dir",
                    str(spectra_dir),
                    "--out",
                    str(manifest_path),
                ]
            )
            == 0
        )

        study_js = tmp / "mtbls1_study.json"
        buckets_csv = tmp / "mtbls1_buckets.csv"
        pca_js = tmp / "mtbls1_pca.json"
        oplsda_js = tmp / "mtbls1_oplsda.json"
        biomarkers_csv = tmp / "mtbls1_biomarkers.csv"
        stocsy_js = tmp / "mtbls1_stocsy_glucose.json"

        _run_nmrassign(
            nmrassign_bin,
            [
                "metabolomics",
                "bin",
                "--manifest",
                str(manifest_path),
                "--min-ppm",
                "0.50",
                "--max-ppm",
                "9.50",
                "--mask",
                "urine",
                "--calibrate-tsp",
                "--method",
                "aib",
                "--norm",
                "pqn",
                "--scale",
                "pareto",
                "-o",
                str(buckets_csv),
                "--study-out",
                str(study_js),
            ],
        )
        _run_nmrassign(
            nmrassign_bin,
            ["metabolomics", "pca", "--in", str(study_js), "--components", "5", "-o", str(pca_js)],
        )
        _run_nmrassign(
            nmrassign_bin,
            [
                "metabolomics",
                "oplsda",
                "--in",
                str(study_js),
                "--ortho",
                "1",
                "--cv-folds",
                "7",
                "--permutations",
                "100",
                "--seed",
                "20261717",
                "--control",
                "Control Group",
                "--case",
                "diabetes mellitus",
                "--biomarkers-out",
                str(biomarkers_csv),
                "-o",
                str(oplsda_js),
            ],
        )
        _run_nmrassign(
            nmrassign_bin,
            [
                "metabolomics",
                "stocsy",
                "--in",
                str(study_js),
                "--driver-ppm",
                "5.24",
                "-o",
                str(stocsy_js),
            ],
        )

        res = evaluate_f17_metabolomics(
            oplsda_js.read_text(encoding="utf-8"),
            stocsy_js.read_text(encoding="utf-8"),
            biomarkers_csv_text=biomarkers_csv.read_text(encoding="utf-8"),
            n_clean_processed_pqn_samples=130,
        )
        assert res["verdict"] == "PASS", f"F17 failed: {json.dumps(res, indent=2)}"
        return (
            f"PASS (Q2={res['criteria_results'][1]['value']:.3f}, "
            f"perm_p={res['criteria_results'][2]['value']:.4f}, "
            f"glucose_hits_top15={res['criteria_results'][3]['value']:.0f})"
        )


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline self-test suite for scripts/p17_adapters/")
    ap.add_argument("--self-test", action="store_true", help="Run full offline adapter & CLI self-test suite")
    ap.add_argument(
        "--nmrassign-bin",
        default=None,
        help="Path to compiled nmrassign binary (default: <repo_root>/target/release/nmrassign)",
    )
    args = ap.parse_args(argv)
    _ = args.self_test  # Default mode also runs the full self-test suite

    repo_root = find_repo_root()
    bin_path = (
        Path(args.nmrassign_bin).resolve()
        if args.nmrassign_bin
        else (repo_root / "target" / "release" / "nmrassign")
    )
    if not bin_path.is_file():
        print(f"ERROR [p17_adapters.selftest]: compiled binary not found at {bin_path}", file=sys.stderr)
        return 1

    suite: List[Tuple[str, Callable[[], str]]] = [
        ("AST_FIREWALL", lambda: test_ast_blind_firewall_and_zero_hardcoded_constants(repo_root)),
        ("COMMON_EVAL", lambda: test_common_statistics_and_manifest_evaluator(repo_root)),
        ("F10_MODELFREE", lambda: test_f10_adapter_and_nmrassign_modelfree(bin_path)),
        ("F11_RELAX", lambda: test_f11_adapter_vclist_and_nmrassign_relax(bin_path)),
        ("F13A_CEST", lambda: test_f13a_cest_adapter_and_nmrassign_cest(bin_path)),
        ("F13B_CPMG", lambda: test_f13b_cpmg_adapter_and_nmrassign_cpmg(bin_path)),
        ("F13C_CSP", lambda: test_f13c_csp_adapter_and_nmrassign_csp(bin_path)),
        ("F14_NMR1D_QNMR", lambda: test_f14_nmr1d_and_qnmr_adapter(bin_path)),
        ("F16_DOSY", lambda: test_f16_dosy_plane_integrator_and_nmrassign_dosy(bin_path)),
        ("F17_MTBLS1", lambda: test_f17_mtbls1_manifest_and_nmrassign_metabolomics(bin_path)),
    ]

    passed = 0
    for name, fn in suite:
        with contextlib.redirect_stdout(io.StringIO()):
            detail = fn()
        passed += 1
        print(f"[PASS] {name:<16} {detail}")

    hashes = hash_adapter_directory(repo_root / "scripts" / "p17_adapters")
    print(f"\nOK [p17_adapters.selftest]: {passed}/{len(suite)} checks passed; {len(hashes)} adapter files hashed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
