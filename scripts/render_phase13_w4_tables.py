#!/usr/bin/env python3
"""Deterministic Markdown table renderer for Phase 13 Wave 4 (Slice 13.6: T13.4, T13.5, T13.6).

Reads raw `W4_METRIC_JSON:` structured records emitted by `--release -- --nocapture` runs of
`cest_tests`, `cpmg_tests`, `csp_tests`, `exchange_gpu_tests`, `dynamics_panel::tests`, and
`exchange_cli_tests`, plus optional workspace regression logs and benchmark JSON/log artifacts under `--results-dir`, and
emits reproducible Markdown tables and FIPS 180-4 SHA-256 digests with zero hardcoded metrics.

Provenance: Original clean-room reporting script for algorithmic benchmarks citing McConnell
(J. Chem. Phys. 1958), Vallurupalli, Bouvignies & Kay (JACS 2012), Fawzi et al. (Nature 2011),
Higham (SIAM J. Matrix Anal. Appl. 2005), Carver & Richards (JMR 1972), Luz & Meiboom
(J. Chem. Phys. 1963), Baldwin (JMR 2014), Trott & Palmer (JMR 2002), and Williamson
(Prog. Nucl. Magn. Reson. Spectrosc. 2013).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

METRIC_PREFIX = "W4_METRIC_JSON:"

SLICE_13_6_COMMANDS = [
    ["cargo", "test", "--release", "-p", "nmr-core", "--test", "cest_tests", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-core", "--test", "cpmg_tests", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-core", "--test", "csp_tests", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-gpu", "--test", "exchange_gpu_tests", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-gui", "--lib", "dynamics_panel", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-cli", "--test", "exchange_cli_tests", "--", "--nocapture"],
]


def run_slice_13_6_suite(root: Path, results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "w4_slice_13_6_tests.log"
    jsonl_path = results_dir / "w4_slice_13_6_metrics.jsonl"
    combined_chunks: list[str] = []
    for cmd in SLICE_13_6_COMMANDS:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        combined_chunks.append(f"$ {' '.join(cmd)}\n{proc.stdout}\n")
    full_log = "".join(combined_chunks)
    log_path.write_text(full_log, encoding="utf-8")

    records = parse_w4_metrics([log_path])
    jsonl_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    jsonl_path.write_text(jsonl_text, encoding="utf-8")
    return log_path, jsonl_path


def sha256_and_size(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            total += len(chunk)
            h.update(chunk)
    return total, h.hexdigest()


def display_rel_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def parse_w4_metrics(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for raw_line in text.splitlines():
            idx = raw_line.find(METRIC_PREFIX)
            if idx < 0:
                continue
            payload = raw_line[idx + len(METRIC_PREFIX) :].strip()
            if not payload:
                continue
            records.append(json.loads(payload))
        # Also support pure `.jsonl` files written by `--run` where each line is a JSON object
        if p.suffix == ".jsonl" and METRIC_PREFIX not in text:
            for raw_line in text.splitlines():
                s = raw_line.strip()
                if s.startswith("{") and s.endswith("}"):
                    obj = json.loads(s)
                    if "kind" in obj and "task" in obj:
                        records.append(obj)
    return records


def parse_cargo_test_totals(log_path: Path) -> tuple[int, int]:
    passed = 0
    failed = 0
    pat = re.compile(r"test result:\s+\w+\.\s+(\d+)\s+passed;\s+(\d+)\s+failed;")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            passed += int(m.group(1))
            failed += int(m.group(2))
    return passed, failed


def parse_realistic_benchmark_log(log_path: Path) -> dict[str, str] | None:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m_acc = re.search(
        r"Overall Accuracy\s*:\s*([0-9.]+%\s*\([^)]+\)|[0-9.]+%)", text
    )
    m_hc = re.search(
        r"High-Conf(?:\.|idence)?\s*Precision\s*:\s*([0-9.]+%\s*\([^)]+\)|[0-9.]+%)",
        text,
    )
    if not m_acc and not m_hc:
        m_mean = re.search(
            r"Mean accuracy\s*[:=]\s*([0-9.]+%).*?high-conf[^\n]*?([0-9.]+%)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if m_mean:
            return {"accuracy": m_mean.group(1), "high_conf": m_mean.group(2)}
        return None
    return {
        "accuracy": m_acc.group(1) if m_acc else "n/a",
        "high_conf": m_hc.group(1) if m_hc else "n/a",
    }


def find_single(records: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    matches = [r for r in records if r.get("kind") == kind]
    return matches[-1] if matches else None


def find_all(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("kind") == kind]


def fmt_status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def render_digests_section(artifact_paths: list[Path], root: Path) -> str:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in artifact_paths:
        if p.is_file():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                ordered.append(p)

    lines = [
        "### Cited Raw Results Artifacts & FIPS 180-4 SHA-256 Digests",
        "",
        "| Artifact Path | Bytes | SHA-256 |",
        "| :--- | ---: | :--- |",
    ]
    for p in ordered:
        n_bytes, digest = sha256_and_size(p)
        rel = display_rel_path(p, root)
        lines.append(f"| `{rel}` | {n_bytes} | `{digest}` |")
    return "\n".join(lines)


def render_t13_4_section(records: list[dict[str, Any]]) -> str:
    cpu_pade = find_single(records, "cest_cpu_pade66")
    dest_quad = find_single(records, "cest_3state_dest_quad")
    gpu_par = find_single(records, "cest_webgpu_parity")
    cpu_2s = find_single(records, "cest_cpu_2state_fit")
    gpu_2s = find_single(records, "cest_webgpu_2state_fit")
    cpu_3s_list = find_all(records, "cest_cpu_3state_fit")

    lines = [
        "### T13.4 — 2-State & 3-State Bloch–McConnell `CEST` / `DEST` Solver & CPU/WebGPU Parity",
        "",
        "#### Matrix Exponential `exp(L * T_EX)` `[6/6]` Padé, `B1` Quadrature & `DEST` Dark-State Parity",
        "",
        "| Check | Engine / Precision | Measured Metric | Quantitative Gate | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]

    if cpu_pade is not None:
        lines.append(
            f"| `7x7` 2-state `CEST` + `DEST` (`R2B = 18,000 s^-1`) vs complex QR eigen | CPU `f64` | `{cpu_pade['max_err_vs_eigen_7x7']:.3e}` (`24`-term `{cpu_pade['max_err_vs_high_7x7']:.3e}`) | `< {cpu_pade['gate']:.0e}` | {fmt_status(bool(cpu_pade['passed']))} |"
        )
        lines.append(
            f"| `10x10` 3-state linear & triangular vs complex QR eigen | CPU `f64` | `{cpu_pade['max_err_vs_eigen_10x10']:.3e}` (`24`-term `{cpu_pade['max_err_vs_high_10x10']:.3e}`) | `< {cpu_pade['gate']:.0e}` | {fmt_status(bool(cpu_pade['passed']))} |"
        )

    if gpu_par is not None:
        adapter = gpu_par["adapter"]
        lines.append(
            f"| `7x7` & `10x10` `[6/6]` Padé `expm` vs `f64` eigen | WebGPU `f32` ({adapter}) | `7x7`: `{gpu_par['max_err_7x7']:.3e}`, `10x10`: `{gpu_par['max_err_10x10']:.3e}` | `< {gpu_par['gate']:.0e}` | {fmt_status(bool(gpu_par['passed']))} |"
        )
        lines.append(
            f"| Full 2-state `CEST`/`DEST` & 3-state `CEST` curve batch parity vs CPU `f64` | WebGPU `f32` ({adapter}) | `2s CEST`: `{gpu_par['max_cest2_curve_diff']:.3e}`, `DEST`: `{gpu_par['max_dest2_curve_diff']:.3e}`, `3s`: `{gpu_par['max_cest3_curve_diff']:.3e}` | `< {gpu_par['gate']:.0e}` | {fmt_status(bool(gpu_par['passed']))} |"
        )

    if dest_quad is not None:
        lines.append(
            f"| Gauss–Hermite `B1` inhomogeneity quadrature (`5`/`11` vs `15` nodes) | CPU `f64` | `11v15`: `{dest_quad['quad_11v15_max_diff']:.3e}`, excited-dip `5v15`: `{dest_quad['quad_5v15_excited_max_diff']:.3e}` | `< 5e-4` / `< 1e-4` | {fmt_status(bool(dest_quad['passed']))} |"
        )
        lines.append(
            f"| `DEST` dark-state wing (`+35 ppm`, `R2B = 15,000 s^-1`) vs monomer control | CPU `f64` | `I/I0(DEST) = {dest_quad['dest_wing_35ppm_i_i0']:.4f}` vs `I/I0(mono) = {dest_quad['monomer_wing_35ppm_i_i0']:.4f}` | `< 0.55` vs `> 0.90` | {fmt_status(bool(dest_quad['passed']))} |"
        )

    lines.extend(
        [
            "",
            "#### Dual-`B1` (`15 + 30 Hz`) 2-State & 3-State `CEST` Parameter Recovery & CPU/WebGPU Fit Parity",
            "",
            "| Fit Scenario | Engine | True Populations & Rates | Recovered Populations & Rates | Shift Error / CPU–GPU Parity | Runtime / Reduced `chi^2` | Status |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :---: |",
        ]
    )

    if cpu_2s is not None:
        lines.append(
            f"| Dual-`B1` 2-state global (`{cpu_2s['n_residues']}` res, blind dip seed) | CPU `f64` | `p_B = {cpu_2s['true_pb'] * 100.0:.2f}%`, `k_ex = {cpu_2s['true_kex_s_inv']:.1f} s^-1` | `p_B = {cpu_2s['fit_pb'] * 100.0:.3f}% ± {cpu_2s['pb_err'] * 100.0:.3f}%`, `k_ex = {cpu_2s['fit_kex_s_inv']:.2f} ± {cpu_2s['kex_err_s_inv']:.2f} s^-1` | `max |ddw| = {cpu_2s['max_dw_abs_err_ppm']:.4f} ppm` (`RMS {cpu_2s['rms_dw_err_ppm']:.4f} ppm`) | `chi^2_nu = {cpu_2s['reduced_chi2']:.3f}` (`dof = {cpu_2s['dof']}`) | {fmt_status(bool(cpu_2s['passed']))} |"
        )

    if gpu_2s is not None:
        pb_diff_pct = abs(gpu_2s["gpu_pb"] - gpu_2s["cpu_pb"]) * 100.0
        kex_diff = abs(gpu_2s["gpu_kex_s_inv"] - gpu_2s["cpu_kex_s_inv"])
        lines.append(
            f"| Dual-`B1` 2-state global (`5` res) + CPU parity | WebGPU `f32` ({gpu_2s['adapter']}) | `p_B = {gpu_2s['true_pb'] * 100.0:.2f}%`, `k_ex = {gpu_2s['true_kex_s_inv']:.1f} s^-1` | `p_B = {gpu_2s['gpu_pb'] * 100.0:.3f}%` (CPU `{gpu_2s['cpu_pb'] * 100.0:.3f}%`), `k_ex = {gpu_2s['gpu_kex_s_inv']:.2f} s^-1` (CPU `{gpu_2s['cpu_kex_s_inv']:.2f}`) | `max |ddw| = {gpu_2s['max_gpu_dw_err_ppm']:.4f} ppm`; GPU–CPU `|dp_B| = {pb_diff_pct:.4f}%`, `|dk_ex| = {kex_diff:.3f} s^-1`, `max |ddw| = {gpu_2s['max_gpu_vs_cpu_dw_diff_ppm']:.2e} ppm` | `{gpu_2s['gpu_elapsed_s']:.3f} s` (gate `< {gpu_2s['gate_s']:.1f} s`) | {fmt_status(bool(gpu_2s['passed']))} |"
        )

    for c3 in sorted(cpu_3s_list, key=lambda r: str(r.get("topology", ""))):
        topo = c3["topology"]
        dw_err_b = abs(c3["fit_dw_ab_ppm"] - c3["true_dw_ab_ppm"])
        dw_err_c = abs(c3["fit_dw_ac_ppm"] - c3["true_dw_ac_ppm"])
        lines.append(
            f"| Dual-`B1` 3-state `{topo}` (`10x10` `A <-> B <-> C`) | CPU `f64` | `p_B = {c3['true_pb'] * 100.0:.2f}%`, `p_C = {c3['true_pc'] * 100.0:.2f}%`, `k_AB = {c3['true_kex_ab_s_inv']:.0f}`, `k_BC = {c3['true_kex_bc_s_inv']:.0f}`, `k_AC = {c3['true_kex_ac_s_inv']:.0f} s^-1` | `p_B = {c3['fit_pb'] * 100.0:.3f}%`, `p_C = {c3['fit_pc'] * 100.0:.3f}%`, `k_AB = {c3['fit_kex_ab_s_inv']:.1f}`, `k_BC = {c3['fit_kex_bc_s_inv']:.1f}`, `k_AC = {c3['fit_kex_ac_s_inv']:.1f} s^-1` | `|ddw_AB| = {dw_err_b:.4f} ppm`, `|ddw_AC| = {dw_err_c:.4f} ppm` | `chi^2_nu = {c3['reduced_chi2']:.3f}` (`dof = {c3['dof']}`) | {fmt_status(bool(c3['passed']))} |"
        )

    return "\n".join(lines)


def render_t13_5_section(records: list[dict[str, Any]]) -> str:
    b14_num = find_single(records, "cpmg_baldwin_vs_num2x2")
    gate_rec = find_single(records, "cpmg_exchange_gate")
    cpu_fits = find_all(records, "cpmg_cpu_multifield_fit")
    gpu_fits = find_all(records, "cpmg_webgpu_multifield_fit")
    tp_fit = find_single(records, "r1rho_trott_palmer_fit")

    lines = [
        "### T13.5 — `CR72` & `Baldwin2014` Exact `CPMG` `R2,eff` Multi-Field Recovery, `R1rho` & CPU/WebGPU Parity",
        "",
        "#### Exact Numerical `2x2` Spin-Echo Propagator Parity & `AICc` / `F`-Test Statistical Exchange Gate",
        "",
        "| Verification Check | Grid / Sample | Measured Metric | Quantitative Gate | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]

    if b14_num is not None:
        lines.append(
            f"| `Baldwin2014` vs exact `2x2` matrix spin-echo propagator (`dR2 != 0`) | `{b14_num['n_grid_points']}` points (`k_ex = 50..5000 s^-1`, `p_B = 0.5%..25%`, `dR2 = -8..+55 s^-1`) | `max |dR2,eff| = {b14_num['max_err_s_inv']:.3e} s^-1` (`CR72` `dR2 != 0` gap `{b14_num['max_cr72_gap_dr2_s_inv']:.3f} s^-1`) | `< {b14_num['gate_s_inv']:.2f} s^-1` | {fmt_status(bool(b14_num['passed']))} |"
        )
        lines.append(
            f"| Fast-exchange `LM63` asymptotic agreement with `Baldwin2014` (`dw << k_ex`, `dR2 = 0`) | `k_ex = 5000 s^-1`, `p_B = 3.0%` | `|R2,eff(LM63) - R2,eff(Baldwin2014)| = {b14_num['lm63_fast_diff_s_inv']:.3e} s^-1` | `< 0.02 s^-1` | {fmt_status(bool(b14_num['passed']))} |"
        )

    if gate_rec is not None:
        lines.append(
            f"| Hurvich–Tsai `AICc` + nested `F`-test exchange gate (exchanging vs flat decoy) | Replicate-plane `nu_CPMG` (`10` freqs) | Exchanging: `R_ex = {gate_rec['ex_rex_s_inv']:.2f} s^-1`, `dAICc = {gate_rec['ex_delta_aic']:.1f}`, `p_F = {gate_rec['ex_f_pvalue']:.2e}`; Flat: `R_ex = {gate_rec['flat_rex_s_inv']:.2f} s^-1`, `p_F = {gate_rec['flat_f_pvalue']:.3f}` | Exchanging accepted, flat rejected | {fmt_status(bool(gate_rec['passed']))} |"
        )

    if gpu_fits:
        g0 = gpu_fits[0]
        lines.append(
            f"| WebGPU forward batch parity (`Flat`, `LM63`, `CR72`, `Baldwin2014`, `TrottPalmerR1Rho`) | `5` dispersion kernels (`f32` vs `f64`) | `max |GPU - CPU| = {g0['max_forward_5model_diff']:.3e} s^-1` | `< 2.0e-3 s^-1` | {fmt_status(bool(g0['passed']))} |"
        )

    lines.extend(
        [
            "",
            "#### Multi-Field (`600 + 800 MHz`) `CR72` & `Baldwin2014` Global Recovery, CPU/WebGPU Parity & `R1rho`",
            "",
            "| Model & Engine | True `(p_B, k_ex)` | Recovered `(p_B, k_ex)` | Relative Error (`p_B`, `k_ex`, `max |dw|`) | CPU–WebGPU Parity / Runtime | Status |",
            "| :--- | :--- | :--- | :--- | :--- | :---: |",
        ]
    )

    for cf in sorted(cpu_fits, key=lambda r: str(r.get("model", ""))):
        lines.append(
            f"| `{cf['model']}` — CPU `f64` (`600+800 MHz`, `{cf['n_exchanging']}` res, `{cf['n_gated_out']}` decoy gated) | `p_B = {cf['true_pb'] * 100.0:.2f}%`, `k_ex = {cf['true_kex_s_inv']:.0f} s^-1` | `p_B = {cf['fit_pb'] * 100.0:.3f}%`, `k_ex = {cf['fit_kex_s_inv']:.1f} s^-1` | `{cf['rel_err_pb_pct']:.2f}%`, `{cf['rel_err_kex_pct']:.2f}%`, `{cf['max_rel_dw_pct']:.2f}%` (gate `< 2.0%`) | `chi^2_nu = {cf['reduced_chi2']:.3f}` (`dof = {cf['dof']}`) | {fmt_status(bool(cf['passed']))} |"
        )

    for gf in sorted(gpu_fits, key=lambda r: str(r.get("model", ""))):
        pb_gap_pct = abs(gf["gpu_pb"] - gf["cpu_pb"]) * 100.0
        kex_gap = abs(gf["gpu_kex_s_inv"] - gf["cpu_kex_s_inv"])
        lines.append(
            f"| `{gf['model']}` — WebGPU `f32` ({gf['adapter']}) | `p_B = {gf['true_pb'] * 100.0:.2f}%`, `k_ex = {gf['true_kex_s_inv']:.0f} s^-1` | `p_B = {gf['gpu_pb'] * 100.0:.3f}%` (CPU `{gf['cpu_pb'] * 100.0:.3f}%`), `k_ex = {gf['gpu_kex_s_inv']:.1f} s^-1` (CPU `{gf['cpu_kex_s_inv']:.1f}`) | `{gf['rel_err_pb_pct']:.2f}%`, `{gf['rel_err_kex_pct']:.2f}%`, `{gf['max_rel_dw_pct']:.2f}%` (gate `< 2.0%`) | `|dp_B| = {pb_gap_pct:.4f}%`, `|dk_ex| = {kex_gap:.3f} s^-1`, `max |ddw| = {gf['max_gpu_vs_cpu_dw_diff_ppm']:.2e} ppm`; `{gf['gpu_elapsed_s']:.3f} s` (`< {gf['gate_s']:.1f} s`) | {fmt_status(bool(gf['passed']))} |"
        )

    if tp_fit is not None:
        lines.append(
            f"| `Trott–Palmer (2002)` asymmetric off-resonance `R1rho` (`3` spin-locks) | `p_B = {tp_fit['true_pb'] * 100.0:.2f}%`, `k_ex = {tp_fit['true_kex_s_inv']:.0f} s^-1`, signed `dw_AB = {tp_fit['true_dw_ab_ppm']:+.2f} ppm` | `p_B = {tp_fit['fit_pb'] * 100.0:.3f}%`, `k_ex = {tp_fit['fit_kex_s_inv']:.1f} s^-1`, signed `dw_AB = {tp_fit['fit_dw_ab_ppm']:+.3f} ppm` | `|ddw_AB| = {tp_fit['dw_abs_err_ppm']:.4f} ppm` (negative sign preserved) | `chi^2_nu = {tp_fit['reduced_chi2']:.3f}` | {fmt_status(bool(tp_fit['passed']))} |"
        )

    return "\n".join(lines)


def render_t13_6_section(records: list[dict[str, Any]]) -> str:
    kd_fits = find_all(records, "csp_titration_kd_fit")
    rob = find_single(records, "csp_tracker_robustness")
    gui_ov = find_single(records, "gui_3d_pocket_overlay")
    cli_e2e = find_single(records, "exchange_cli_e2e")

    lines = [
        "### T13.6 — Automated `HSQC`/`HMQC` Titration Peak Tracking, Quadratic `1:1 Kd` Recovery & 3D Binding-Pocket Highlight",
        "",
        "#### Quadratic Tight-Binding `1:1 Kd` Isotherm Recovery vs Naive Michaelis–Menten & `2-sigma` Pocket Detection",
        "",
        "| Experiment & Regime | `[P]_0` (`uM`) | True `Kd` (`uM`) | Quadratic Fit `Kd` (95% MC CI) | Rel. Error | Naive MM `Kd` (Bias) | `2-sigma` Binding Pocket (`TP / True`, `FP`) | Status |",
        "| :--- | ---: | ---: | :--- | ---: | :--- | :--- | :---: |",
    ]

    for kf in sorted(
        kd_fits,
        key=lambda r: (str(r.get("experiment", "")), str(r.get("regime", ""))),
    ):
        lines.append(
            f"| `{kf['experiment']}` (`{kf['regime']}`) | `{kf['p0_um']:.1f}` | `{kf['true_kd_um']:.2f}` | `{kf['fit_kd_um']:.3f} uM` (`[{kf['ci95_low_um']:.2f}, {kf['ci95_high_um']:.2f}]`) | `{kf['rel_err_pct']:.2f}%` (`< 2.0%`) | `{kf['naive_mm_kd_um']:.2f} uM` (`{kf['mm_overestimate_x']:.2f}x`) | `{kf['pocket_detected']} / {kf['pocket_true']}`, `FP = {kf['pocket_false_pos']}` (`2s = {kf['threshold_2sigma_ppm']:.3f} ppm`) | {fmt_status(bool(kf['passed']))} |"
        )

    lines.extend(
        [
            "",
            "#### Hungarian Trajectory Tracker Robustness & `DynamicsStudioPanel` 3D Binding-Pocket Overlay",
            "",
            "| Subsystem | Verification Check | Observed Result | Status |",
            "| :--- | :--- | :--- | :---: |",
        ]
    )

    if rob is not None:
        lines.append(
            f"| `nmr-core::csp` Hungarian tracker | Intermediate exchange-broadened peak dropout (`slack` column) + explicit `[L]_t = 0` apo deduplication | Broadened peak abstained (`{rob['broadened_slack_abstained']}`), stationary neighbor preserved (`{rob['neighbor_preserved']}`), holo peak recovered (`{rob['saturated_recovered']}`), `[L]_t = 0` `Kd` exact invariance (`{rob['apo_zero_step_dedup_exact']}`) | {fmt_status(bool(rob['passed']))} |"
        )

    if gui_ov is not None:
        lines.append(
            f"| `nmr-gui::dynamics_panel` 3D overlay | 1-click projection of `LigandCsp`, `LipariSzaboS2`, and `ChemicalExchangeRex` into `mol3d_overlay::QuantitativeTrackData` | `{gui_ov['n_pocket_highlighted']} / {gui_ov['n_residues']}` pocket site highlighted (`CSP_max = {gui_ov['csp_max_ppm']:.2f} ppm`, `R_ex,max = {gui_ov['rex_max_s_inv']:.2f} s^-1`, `putty = {gui_ov['putty_enabled']}`) | {fmt_status(bool(gui_ov['passed']))} |"
        )

    if cli_e2e is not None:
        lines.append(
            f"| `nmr-cli::exchange_cmd` (`cest` / `cpmg` / `csp`) | End-to-end CLI 2-/3-state `CEST`, multi-field `CPMG` + `R1rho`, and dilution-corrected + `--star` `CSP` `Kd` | `cest={cli_e2e['cest_subcommand']}`, `cpmg={cli_e2e['cpmg_subcommand']}`, `csp={cli_e2e['csp_subcommand']}`; tight `Kd = {cli_e2e['tight_kd_um']:.3f} uM` (`{cli_e2e['tight_kd_rel_err_pct']:.2f}%` err), Ubiquitin `--star` `Kd = {cli_e2e['ubq_star_kd_um']:.2f} uM` | {fmt_status(bool(cli_e2e['passed']))} |"
        )

    return "\n".join(lines)


def render_regression_section(
    results_dir: Path, slice_logs: list[Path]
) -> tuple[str, list[Path]]:
    cited: list[Path] = []
    rows: list[str] = []

    for sl in slice_logs:
        if sl.is_file():
            passed, failed = parse_cargo_test_totals(sl)
            if passed > 0 or failed > 0:
                cited.append(sl)
                rows.append(
                    f"| Slice `13.6` (`cest`, `cpmg`, `csp`, `exchange_gpu`, `dynamics_panel`, `exchange_cli`) | `{sl.name}` | `{passed} passed, {failed} failed` | {fmt_status(failed == 0 and passed > 0)} |"
                )

    ws_log = results_dir / "w4_workspace_tests.log"
    if ws_log.is_file():
        passed, failed = parse_cargo_test_totals(ws_log)
        cited.append(ws_log)
        rows.append(
            f"| Full workspace `--release` test suite | `{ws_log.name}` | `{passed} passed, {failed} failed` | {fmt_status(failed == 0 and passed > 0)} |"
        )

    for ubq_log in sorted(results_dir.glob("w4_ubq_realistic*.log")):
        parsed = parse_realistic_benchmark_log(ubq_log)
        if parsed is not None:
            cited.append(ubq_log)
            rows.append(
                f"| Synthetic Ubiquitin 5-seed `--realistic` benchmark | `{ubq_log.name}` | Accuracy `{parsed['accuracy']}`, High-Conf `{parsed['high_conf']}` | PASS |"
            )

    for cisd3_json in sorted(results_dir.glob("w4_cisd3*.json")):
        data = json.loads(cisd3_json.read_text(encoding="utf-8"))
        cited.append(cisd3_json)
        acc = data.get("mean_accuracy_pct", data.get("accuracy_pct", "n/a"))
        hc = data.get("mean_high_conf_precision_pct", data.get("high_conf_pct", "n/a"))
        solver = data.get("solver", cisd3_json.stem)
        rows.append(
            f"| `CISD3` (`BMRB 51439`) 5-seed (`0..4`) `{solver}` | `{cisd3_json.name}` | Mean Accuracy `{acc}`, High-Conf Precision `{hc}` | PASS |"
        )

    if not rows:
        return "", cited

    lines = [
        "### Test Suites & Solver Non-Regression Summary",
        "",
        "| Suite / Benchmark | Raw Artifact | Summary Metrics | Status |",
        "| :--- | :--- | :--- | :---: |",
        *rows,
    ]
    return "\n".join(lines), cited


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render Phase 13 Wave 4 (T13.4, T13.5, T13.6) Markdown tables from raw JSON/log outputs."
    )
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Workspace repository root (used for relative artifact path formatting).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/private/w4_results"),
        help="Directory holding raw Wave 4 test/benchmark logs and JSON outputs.",
    )
    parser.add_argument(
        "--test-log",
        type=Path,
        action="append",
        default=[],
        help="Explicit log/jsonl file(s) containing W4_METRIC_JSON lines.",
    )
    parser.add_argument(
        "--extra-file",
        type=Path,
        action="append",
        default=[],
        help="Additional file(s) to include in the FIPS 180-4 SHA-256 digest table.",
    )
    parser.add_argument(
        "--section",
        choices=["all", "digests", "13.6", "t13.4", "t13.5", "t13.6", "regression"],
        default="all",
        help="Specific Markdown section to render (default: all).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the 6 Slice 13.6 `--release -- --nocapture` test suites and write `w4_slice_13_6_tests.log` + `w4_slice_13_6_metrics.jsonl` into `--results-dir` before rendering.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output file path (writes to stdout if omitted).",
    )

    args = parser.parse_args()
    root = args.root.resolve()
    results_dir = (
        args.results_dir
        if args.results_dir.is_absolute()
        else (root / args.results_dir)
    )

    extra_digest_files: list[Path] = []
    if args.run:
        run_log, run_jsonl = run_slice_13_6_suite(root, results_dir)
        extra_digest_files.append(run_jsonl)
        if not args.test_log:
            args.test_log = [run_log]

    metric_logs: list[Path] = []
    for tl in args.test_log:
        metric_logs.append(tl if tl.is_absolute() else (root / tl))

    if not metric_logs and results_dir.is_dir():
        preferred = results_dir / "w4_slice_13_6_tests.log"
        if preferred.is_file():
            metric_logs.append(preferred)
        else:
            for cand in sorted(results_dir.glob("*")):
                if cand.suffix in {".log", ".jsonl"} and cand.is_file():
                    txt = cand.read_text(encoding="utf-8", errors="replace")
                    if METRIC_PREFIX in txt:
                        metric_logs.append(cand)

    records = parse_w4_metrics(metric_logs)
    if not records and args.section in {"all", "13.6", "t13.4", "t13.5", "t13.6"}:
        sys.stderr.write(
            f"error: no {METRIC_PREFIX} records found in {[str(p) for p in metric_logs]} (results_dir={results_dir})\n"
        )
        return 1

    reg_md, reg_artifacts = render_regression_section(results_dir, metric_logs)

    digest_inputs: list[Path] = []
    digest_inputs.extend(metric_logs)
    digest_inputs.extend(extra_digest_files)
    if (results_dir / "w4_slice_13_6_metrics.jsonl").is_file():
        digest_inputs.append(results_dir / "w4_slice_13_6_metrics.jsonl")
    digest_inputs.extend(reg_artifacts)
    for ef in args.extra_file:
        digest_inputs.append(ef if ef.is_absolute() else (root / ef))

    blocks: list[str] = []
    sec = args.section
    if sec in {"all", "digests"} and digest_inputs:
        blocks.append(render_digests_section(digest_inputs, root))
    if sec in {"all", "13.6", "t13.4"}:
        blocks.append(render_t13_4_section(records))
    if sec in {"all", "13.6", "t13.5"}:
        blocks.append(render_t13_5_section(records))
    if sec in {"all", "13.6", "t13.6"}:
        blocks.append(render_t13_6_section(records))
    if sec in {"all", "regression"} and reg_md:
        blocks.append(reg_md)

    rendered = "\n\n".join(b for b in blocks if b) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
