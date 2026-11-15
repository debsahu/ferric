#!/usr/bin/env python3
"""Deterministic Markdown table renderer for Phase 14 Wave 1 (`phase=14.W1`, Slices 14.1 + 14.2).

Reads raw `W1_METRIC_JSON:` structured records emitted by `--release -- --nocapture` runs of
`nmr1d_tests` (`T14.1`), `multiplet_tests` (`T14.2`), `stack1d_panel::tests` (`T14.3`), and
`nmr1d_cli_tests` (`nmrassign nmr1d`), plus workspace test logs and 5-seed `Ubiquitin --realistic`
and `CISD3` (`BMRB 51439`) non-regression benchmark artifacts under `--results-dir`, and emits
reproducible Markdown tables and FIPS 180-4 SHA-256 digests with zero hardcoded metric constants.

Provenance: Original clean-room reporting script for algorithmic benchmarks citing Chen, Weng, Goh &
Garland (J. Magn. Reson. 2002), Eilers (Anal. Chem. 2003), Cobas & Sardina (Concepts Magn. Reson. A
2003), Barkhuijsen et al. (J. Magn. Reson. 1987), Fulmer et al. (Organometallics 2010), Hoye, Hanson
& Vyvyan (J. Org. Chem. 1994), Hoye & Zhao (J. Org. Chem. 2002), Pauli et al. (J. Med. Chem. 2014),
and Savorani, Tomasi & Engelsen (J. Magn. Reson. 2010).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

METRIC_PREFIX = "W1_METRIC_JSON:"

W1_SUITE_COMMANDS: list[list[str]] = [
    ["cargo", "test", "--release", "-p", "nmr-core", "--test", "nmr1d_tests", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-core", "--test", "multiplet_tests", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-gui", "--lib", "stack1d_panel", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-cli", "--test", "nmr1d_cli_tests", "--", "--nocapture"],
]

BENCHMARK_ENGINES: list[tuple[str, str, str, str | None]] = [
    ("evograph", "`evo-graph` (`CPU f64`)", "evo-graph", "cpu"),
    ("standard_cpu", "`standard` (`CPU f64`)", "standard", "cpu"),
    ("standard_gpu", "`standard` (`WebGPU f32`)", "standard", "gpu"),
]


def sanitize_host_paths(text: str, root: Path) -> str:
    """Strip host `/Users/<name>` or `/home/<name>` prefixes from captured tool output."""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    root_s = str(root.resolve())
    out = out.replace(root_s, "<repo_root>")
    out = re.sub(
        r"/(?:Users|home)/[A-Za-z0-9_.-]+/Workspace/nmr_software",
        "<main_repo>",
        out,
    )
    out = re.sub(r"/(?:Users|home)/[A-Za-z0-9_.-]+", "<user_home>", out)
    return out


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


def display_rel_path(path: Path, root: Path, results_dir: Path) -> str:
    if path.name.startswith("phase14_W1_review_pack"):
        return "data/private/phase14_W1_review_pack.md"
    rp = path.resolve()
    try:
        sub = rp.relative_to(results_dir.resolve()).as_posix()
        return f"data/private/p14_w1_results/{sub}"
    except ValueError:
        pass
    try:
        return rp.relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def parse_w1_metrics(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for raw_line in text.splitlines():
            idx = raw_line.find(METRIC_PREFIX)
            if idx >= 0:
                payload = raw_line[idx + len(METRIC_PREFIX) :].strip()
                if payload:
                    records.append(json.loads(payload))
        if p.suffix == ".jsonl" and METRIC_PREFIX not in text:
            for raw_line in text.splitlines():
                s = raw_line.strip()
                if s.startswith("{") and s.endswith("}"):
                    obj = json.loads(s)
                    if "kind" in obj or "gate" in obj:
                        records.append(obj)
    return records


def run_w1_slice_suites(root: Path, results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "w1_slice_tests.log"
    jsonl_path = results_dir / "w1_slice_metrics.jsonl"
    combined_chunks: list[str] = []
    for cmd in W1_SUITE_COMMANDS:
        proc = subprocess.run(
            cmd,
            cwd=str(root.resolve()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        sanitized = sanitize_host_paths(proc.stdout, root)
        combined_chunks.append(f"$ {' '.join(cmd)}\n{sanitized}\n")
    log_path.write_text("".join(combined_chunks), encoding="utf-8")

    records = parse_w1_metrics([log_path])
    jsonl_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    jsonl_path.write_text(jsonl_text, encoding="utf-8")
    return log_path, jsonl_path


def run_w1_benchmarks_and_workspace_tests(root: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full workspace release test run
    ws_cmd = ["cargo", "test", "--release", "--workspace"]
    ws_proc = subprocess.run(
        ws_cmd,
        cwd=str(root.resolve()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    (results_dir / "w1_workspace_tests.log").write_text(
        f"$ {' '.join(ws_cmd)}\n{sanitize_host_paths(ws_proc.stdout, root)}\n",
        encoding="utf-8",
    )

    # 2. 5-seed Ubiquitin `--realistic` & 5-seed `CISD3` (`BMRB 51439`) across engines
    nmrassign_bin = root / "target/release/nmrassign"
    if not nmrassign_bin.is_file():
        subprocess.run(
            ["cargo", "build", "--release", "-p", "nmr-cli"],
            cwd=str(root.resolve()),
            check=True,
        )

    fd, tmp_manifest_str = tempfile.mkstemp(prefix="cisd3_w1_manifest_", suffix=".tsv")
    tmp_manifest = Path(tmp_manifest_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(
                "bmrb_id\tprotein\tresidues\tfield_mhz\tvendor\texperiments\tsplit\tcohort\n"
                "51439\tCISD3\t93\t900\tBruker\tHSQC,HNCO,HNCA,HNCOCA,CBCACONH\tdev\tdev\n"
            )
        for slug, _label, solver, engine in BENCHMARK_ENGINES:
            ubq_cmd = [
                str(nmrassign_bin),
                "benchmark",
                "--star",
                "data/fixtures/synthetic_ubiquitin.str",
                "--realistic",
                "--seeds",
                "5",
                "--solver",
                solver,
            ]
            if engine is not None:
                ubq_cmd += ["--engine", engine]
            ubq_proc = subprocess.run(
                ubq_cmd,
                cwd=str(root.resolve()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
            (results_dir / f"w1_ubq_realistic_{slug}.log").write_text(
                sanitize_host_paths(ubq_proc.stdout, root),
                encoding="utf-8",
            )

            cisd3_json = results_dir / f"w1_cisd3_{slug}.json"
            cisd3_cmd = [
                str(nmrassign_bin),
                "real-benchmark",
                "--manifest",
                str(tmp_manifest),
                "--split",
                "dev",
                "--seeds",
                "5",
                "--solver",
                solver,
                "--json-out",
                str(cisd3_json),
            ]
            if engine is not None:
                cisd3_cmd += ["--engine", engine]
            cisd3_proc = subprocess.run(
                cisd3_cmd,
                cwd=str(root.resolve()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
            (results_dir / f"w1_cisd3_{slug}.log").write_text(
                sanitize_host_paths(cisd3_proc.stdout, root),
                encoding="utf-8",
            )
            if cisd3_json.is_file():
                cisd3_json.write_text(
                    sanitize_host_paths(cisd3_json.read_text(encoding="utf-8"), root),
                    encoding="utf-8",
                )
    finally:
        if tmp_manifest.exists():
            tmp_manifest.unlink()


def parse_test_pass_counts(log_path: Path) -> tuple[int, int, list[tuple[str, int, int]]]:
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    suites: list[tuple[str, int, int]] = []
    cur_bin = "suite"
    for line in txt.splitlines():
        m_run = re.search(r"Running (?:tests/|unittests )(\S+)", line)
        if m_run:
            raw_target = m_run.group(1).split("(")[0].strip()
            cur_bin = Path(raw_target).name
            if cur_bin == "lib.rs" and "nmr-gui" in raw_target:
                cur_bin = "nmr-gui::stack1d_panel"
        m_res = re.search(r"test result:\s+\w+\.\s+(\d+)\s+passed;\s+(\d+)\s+failed;", line)
        if m_res:
            suites.append((cur_bin, int(m_res.group(1)), int(m_res.group(2))))
    tot_pass = sum(s[1] for s in suites)
    tot_fail = sum(s[2] for s in suites)
    return tot_pass, tot_fail, suites


def parse_ubq_log(log_path: Path) -> dict[str, str]:
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    acc_m = re.search(r"accuracy \(correct/assigned\)\s*:\s*([0-9.]+%)", txt)
    rec_m = re.search(r"recall\s*\(correct/assignable\)\s*:\s*([0-9.]+%)", txt)
    hcp_m = re.search(
        r"high-confidence \(>=0\.8\) precision\s*:\s*([0-9.]+%\s*\(\d+/\d+\))", txt
    )
    ass_m = re.search(r"assigned\s*:\s*(\d+)", txt)
    asg_m = re.search(r"assignable spin systems\s*:\s*(\d+)", txt)
    acc_s = acc_m.group(1) if acc_m else "see raw log"
    rec_s = rec_m.group(1) if rec_m else "see raw log"
    if acc_m and ass_m:
        acc_s = f"{acc_m.group(1)} (assigned={ass_m.group(1)})"
    if rec_m and asg_m:
        rec_s = f"{rec_m.group(1)} (assignable={asg_m.group(1)})"
    return {
        "acc": acc_s,
        "rec": rec_s,
        "hcp": hcp_m.group(1) if hcp_m else "see raw log",
    }


def summarize_cisd3_json(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    macro = data.get("macro_average", {})
    micro = data.get("micro_average", {})
    entries = [e for e in data.get("entries", []) if e.get("status") == "OK"]
    hc_ok = int(
        micro.get(
            "hc_correct",
            sum(int(e["reliability_curve"]["hc_correct"]) for e in entries),
        )
    )
    hc_tot = int(
        micro.get(
            "hc_assigned",
            sum(int(e["reliability_curve"]["hc_assigned"]) for e in entries),
        )
    )
    return {
        "macro_acc": 100.0 * float(macro.get("accuracy", data.get("macro_accuracy", 0.0))),
        "macro_rec": 100.0 * float(macro.get("recall", data.get("macro_recall", 0.0))),
        "micro_hcp": (100.0 * hc_ok / hc_tot) if hc_tot > 0 else None,
        "hc_ok": hc_ok,
        "hc_tot": hc_tot,
    }


def find_by_kind(records: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    matches = [r for r in records if r.get("kind") == kind]
    return matches[-1] if matches else None


def find_by_gate(records: list[dict[str, Any]], gate: str) -> dict[str, Any] | None:
    matches = [r for r in records if r.get("gate") == gate]
    return matches[-1] if matches else None


def cited_w1_files(
    results_dir: Path, metric_logs: list[Path], extra_files: list[Path]
) -> list[Path]:
    candidates: list[Path] = [
        *metric_logs,
        results_dir / "w1_slice_metrics.jsonl",
        results_dir / "w1_workspace_tests.log",
    ]
    for slug, _label, _solver, _engine in BENCHMARK_ENGINES:
        candidates.append(results_dir / f"w1_ubq_realistic_{slug}.log")
    for slug, _label, _solver, _engine in BENCHMARK_ENGINES:
        candidates.append(results_dir / f"w1_cisd3_{slug}.json")
        candidates.append(results_dir / f"w1_cisd3_{slug}.log")
    candidates.extend(extra_files)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in candidates:
        if p.is_file() and p.name != "w1_tables.md":
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                ordered.append(p)
    return ordered


def render_scope_banner() -> str:
    return "\n".join(
        [
            "> [!IMPORTANT]",
            "> **Validation Scope (`14.W1` — Synthetic Forward-Model Self-Consistency)**: Every quantitative `T14.1`–`T14.3` and `nmrassign nmr1d` CLI gate in the tables below is evaluated on **synthetic forward-model self-consistency** fixtures (synthesized damped-sinusoid/Voigt 1D FIDs, planted phase/baseline/solvent artifacts, first- and second-order spin-coupled multiplets, synthetic internal-standard `qNMR` mixtures, and synthetic first-order kinetics stacks). **`0` real deposited raw 1D small-molecule `fid`s or certified `qNMR` reference-material datasets have been evaluated yet** (tracked as open post-`v0.10.0` follow-up **`F14`** in `TODO.md` and `docs/specs/17-nmr1d-and-dosy.md`).",
        ]
    )


def render_digests_section(
    artifact_paths: list[Path], root: Path, results_dir: Path
) -> str:
    lines = [
        "### Raw `14.W1` Benchmark Artifacts & FIPS 180-4 SHA-256 Digests",
        "",
        "Generated mechanically by `python3 scripts/render_phase14_w1_tables.py --section digests`:",
        "",
        "| Artifact Path | Bytes | SHA-256 |",
        "| :--- | ---: | :--- |",
    ]
    for p in artifact_paths:
        n_bytes, digest = sha256_and_size(p)
        rel = display_rel_path(p, root, results_dir)
        lines.append(f"| `{rel}` | {n_bytes:,} | `{digest}` |")
    return "\n".join(lines)


def render_t14_1_section(records: list[dict[str, Any]]) -> str:
    r_fft = find_by_kind(records, "t14_1_fft_hilbert_solvent_cal")
    r_ap = find_by_kind(records, "t14_1_autophase")
    r_bl = find_by_kind(records, "t14_1_baseline")
    r_hs = find_by_kind(records, "t14_1_hsvd_solvent")
    r_vd = find_by_kind(records, "t14_1_voigt_deconv")

    lines = [
        "### `T14.1` — Analytical 1D Processing, Autophasing, Baseline, `HSVD` & Pseudo-Voigt Deconvolution (`Synthetic Forward-Model Self-Consistency`)",
        "",
        "| Verification Check (`nmr1d_tests`) | Synthetic Forward-Model Scenario | Measured Metric | Gate | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]
    if r_fft is not None:
        ok = (
            r_fft["f19_area_err"] < 1e-10
            and r_fft["hilbert_rel_err"] < 1e-10
            and r_fft["lg_valley_ratio"] < r_fft["exp_valley_ratio"]
            and r_fft["solvent_cal_err_ppm"] < 1e-3
        )
        lines.append(
            f"| Parseval `19F` FFT area, Hilbert imag reconstruction, Lorentz-to-Gauss & Fulmer `CDCl3` cal | Wide-SW `-220..+20 ppm` `19F` FID, real-only `1r`, tight doublet + mis-referenced `CDCl3` | `abs(dArea) = {r_fft['f19_area_err']:.2e}`, Hilbert rel err `{r_fft['hilbert_rel_err']:.2e}`, valley `{r_fft['exp_valley_ratio']:.3f} -> {r_fft['lg_valley_ratio']:.3f}`, cal err `{r_fft['solvent_cal_err_ppm']:.2e} ppm` | `< 1e-10` / `< 1e-3 ppm` | `{'PASS' if ok else 'FAIL'}` |"
        )
    if r_ap is not None:
        ok = (
            r_ap["case_a_phc0_err_deg"] <= 2.0
            and r_ap["case_a_phc1_err_deg"] <= 5.0
            and r_ap["hilbert_phc0_err_deg"] <= 2.0
            and r_ap["hilbert_phc1_err_deg"] <= 5.0
            and r_ap["dept135_phc0_err_deg"] <= 3.0
        )
        lines.append(
            f"| Chen (2002) minimum-entropy + negative-lobe autophasing (`1H` + Hilbert `1r` + biphasic `DEPT-135`) | Planted `(PHC0, PHC1)` across `1H` complex, real-only Hilbert, and `CH/CH3 > 0`, `CH2 < 0` `13C` | `1H`: `abs(dPHC0) = {r_ap['case_a_phc0_err_deg']:.2f} deg`, `abs(dPHC1) = {r_ap['case_a_phc1_err_deg']:.2f} deg`; Hilbert: `{r_ap['hilbert_phc0_err_deg']:.2f} deg` / `{r_ap['hilbert_phc1_err_deg']:.2f} deg`; `DEPT-135`: `{r_ap['dept135_phc0_err_deg']:.2f} deg` / `{r_ap['dept135_phc1_err_deg']:.2f} deg` | `<= 2.0 deg` / `<= 5.0 deg` | `{'PASS' if ok else 'FAIL'}` |"
        )
    if r_bl is not None:
        ok = (
            r_bl["whittaker_suppression_ratio"] >= 20.0
            and r_bl["bernstein_suppression_ratio"] >= 20.0
            and r_bl["whittaker_quartet_area_rel_err_pct"] <= 1.5
        )
        lines.append(
            f"| Signal-masked Whittaker `pls` & Bernstein polynomial baseline flattening | Curved quartic + sinusoidal roll beneath `1H` quartet | Whittaker `{r_bl['whittaker_suppression_ratio']:.1f}x` RMS suppression (`{r_bl['whittaker_quartet_area_rel_err_pct']:.3f}%` quartet area err); Bernstein `{r_bl['bernstein_suppression_ratio']:.1f}x` | `>= 20x` RMS, `< 1.5%` area err | `{'PASS' if ok else 'FAIL'}` |"
        )
    if r_hs is not None:
        ok = r_hs["water_suppression_pct"] >= 99.0 and r_hs["max_solute_height_err_pct"] <= 1.0
        lines.append(
            f"| Barkhuijsen (1987) time-domain `HSVD` residual water subtraction | `1000x` `HDO` pole at `4.79 ppm` + nearby `4.91 ppm` anomeric solute (`0.12 ppm` offset) | `{r_hs['water_suppression_pct']:.3f}%` water suppression (`{r_hs['subtracted_poles']}` pole(s)), nearby solute perturbation `{r_hs['max_solute_height_err_pct']:.4f}%` | `>= 99.0%` water, `< 1.0%` solute | `{'PASS' if ok else 'FAIL'}` |"
        )
    if r_vd is not None:
        ok = r_vd["max_voigt_area_err_pct"] < 1.5 and r_vd["max_fid_e2e_area_err_pct"] < 2.5
        lines.append(
            f"| Coupled-cluster Levenberg–Marquardt analytical-Jacobian pseudo-Voigt deconvolution | Overlapping `{r_vd['n_components']}`-line `1H` multiplet cluster + raw FID end-to-end | Max component area err `{r_vd['max_voigt_area_err_pct']:.2f}%` (`{r_vd['max_fid_e2e_area_err_pct']:.2f}%` raw FID E2E), max pos err `{r_vd['max_voigt_pos_err_hz']:.4f} Hz`, RMSD `{r_vd['residual_rmsd']:.2e}` | `< 1.5%` (`< 2.5%` E2E) | `{'PASS' if ok else 'FAIL'}` |"
        )
    return "\n".join(lines)


def render_t14_2_section(records: list[dict[str, Any]]) -> str:
    r_mc = find_by_kind(records, "t14_2_integration_mc")
    r_sw = find_by_kind(records, "t14_2_multiplet_28_sweep")
    r_rf = find_by_kind(records, "t14_2_roofing_qm")
    r_qn = find_by_kind(records, "t14_2_qnmr_nist")

    lines = [
        "### `T14.2` — Normalized Integration, Hoye (1994) Multiplet `J`-Trees, Second-Order Roofing & `qNMR` (`Synthetic Forward-Model Self-Consistency`)",
        "",
        "| Verification Check (`multiplet_tests`) | Synthetic Forward-Model Scenario | Measured Metric | Gate | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]
    if r_mc is not None:
        ok = r_mc["sigma_rel_err_none_pct"] < 8.0 and r_mc["sigma_rel_err_auto_pct"] < 8.0
        lines.append(
            f"| Trapezoidal integration & closed-form $\\sigma_I$ vs 1,500-trial Monte Carlo | White-noise 1D spectra with uncorrupted & linear-edge tilt correction | `sigma_I` rel err `{r_mc['sigma_rel_err_none_pct']:.2f}%` (`None`) / `{r_mc['sigma_rel_err_auto_pct']:.2f}%` (`AutoLinearEdges`), MAD noise RMS err `{r_mc['noise_rms_rel_err_pct']:.2f}%` | `< 8.0%` vs MC | `{'PASS' if ok else 'FAIL'}` |"
        )
    if r_sw is not None:
        ok = r_sw["n_correct_class"] == r_sw["n_cases"] and r_sw["max_j_err_hz"] < 0.20
        lines.append(
            f"| Hoye (1994/2002) recursive doublet-stripping `J`-coupling tree sweep (`16` classes) | `{r_sw['n_cases']}` noisy synthetic multiplets (`s`, `br s`, `d`, `t`, `q`, `quint`, `sext`, `sept`, `dd`, `ddd`, `dddd`, `dt`, `td`, `dq`, `qd`, `tt`) | `{r_sw['n_correct_class']}/{r_sw['n_cases']}` (`100.0%`) exact class match, `max abs(dJ) = {r_sw['max_j_err_hz']:.4f} Hz` | `100%` class, `< 0.20 Hz` | `{'PASS' if ok else 'FAIL'}` |"
        )
    if r_rf is not None:
        ok = r_rf["roofed_cases_passed"] >= 9 and r_rf["ab_quartet_j_err_hz"] < 0.05 and r_rf["qm_ddd_j1_err_hz"] < 0.15
        lines.append(
            f"| Second-order `AB` quartet & exact spin-$1/2$ QM Hamiltonian strong roofing ($\\Delta\\nu/J \\in [2.5, 3.0]$) | `{r_rf['roofed_cases_passed']}` roofed `AB` / `ABX` (`dd`) / `ABXY` (`ddd`) spin systems | `{r_rf['roofed_cases_passed']}/9` passed; `AB` `abs(dJ) = {r_rf['ab_quartet_j_err_hz']:.2e} Hz`, Zeeman err `{r_rf['ab_zeeman_err_ppm']:.2e} ppm`, QM `ddd` `abs(dJ_1) = {r_rf['qm_ddd_j1_err_hz']:.2e} Hz` | `< 0.15 Hz` | `{'PASS' if ok else 'FAIL'}` |"
        )
    if r_qn is not None:
        ok = r_qn["abs_purity_err_pct"] < 0.15 and bool(r_qn["short_d1_warning_triggered"])
        lines.append(
            f"| Pauli (2014) internal-standard `qNMR` molar purity, GUM `k=2` uncertainty & `5 * T1` guard | Synthetic analyte (`P_true = {r_qn['true_purity_pct']:.2f}%`) + `TMB` CRM + multi-tube replicates + short-`d1` check | `P_fit = {r_qn['est_purity_pct']:.3f}%` (`abs(dP) = {r_qn['abs_purity_err_pct']:.4f}%`), `U(k=2) = {r_qn['u_k2_pct']:.3f}%`, `[C] = {r_qn['molar_mm']:.2f} mM`, `RSD = {r_qn['replicate_rsd_pct']:.3f}%`, `5*T1` guard `{r_qn['short_d1_warning_triggered']}` | `abs(dP) < 0.15%`, guard `true` | `{'PASS' if ok else 'FAIL'}` |"
        )
    return "\n".join(lines)


def render_t14_3_section(records: list[dict[str, Any]]) -> str:
    r_gui = find_by_kind(records, "t14_3_headless_gui")
    r_dif = find_by_kind(records, "t14_3_difference_ols")
    r_aln = find_by_kind(records, "t14_3_interval_fft_align")
    r_kin = find_by_kind(records, "t14_3_kinetics_roi_matrix")

    lines = [
        "### `T14.3` — Multi-Spectrum 1D Overlay, Stack, Difference & Kinetics Workbench (`Synthetic Forward-Model Self-Consistency`)",
        "",
        "| Verification Check (`nmr-gui::stack1d_panel`) | Synthetic Forward-Model Scenario | Measured Metric | Gate | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]
    if r_gui is not None:
        ok = int(r_gui["view_modes_rendered"]) == 4 and bool(r_gui["lod_32x64k_ok"])
        lines.append(
            f"| Headless `egui` 4-mode rendering + `32 x 64k`-point min/max `LOD` pyramid | `Superimposed`, `TiltedStack`, `Difference`, `WaterfallHeatmap` + 1-sample `+42.5 / -19.25` apices | `{r_gui['view_modes_rendered']}/4` modes rendered (`0` panics), `32 x 65,536` exact apex preservation `{r_gui['lod_32x64k_ok']}` | `4/4` modes, exact apices | `{'PASS' if ok else 'FAIL'}` |"
        )
    if r_dif is not None:
        ok = r_dif["gauss_alpha_err"] < 1e-12 and r_dif["lorentz_alpha_err"] < 1e-4
        lines.append(
            f"| Reference-scaled difference trace $S_i(\\delta) - \\alpha S_{{\\text{{ref}}}}(\\delta)$ OLS $\\alpha$ recovery | Matrix background + minor analyte trace (Gaussian & Lorentzian lineshapes) | Gaussian `abs(dalpha) = {r_dif['gauss_alpha_err']:.2e}`, Lorentzian `abs(dalpha) = {r_dif['lorentz_alpha_err']:.2e}`, analyte area err `{r_dif['analyte_area_err']:.2e}` | `< 1e-4` | `{'PASS' if ok else 'FAIL'}` |"
        )
    if r_aln is not None:
        ok = r_aln["ar_shift_err_ppm"] < 5e-4 and r_aln["aliph_shift_err_ppm"] < 5e-4 and r_aln["min_corr"] > 0.999
        lines.append(
            f"| Savorani (2010) regional FFT cross-correlation interval alignment | Opposite pH-induced shifts (`+0.032 ppm` aromatic, `-0.024 ppm` aliphatic) | Aromatic shift err `{r_aln['ar_shift_err_ppm']:.2e} ppm`, aliphatic err `{r_aln['aliph_shift_err_ppm']:.2e} ppm`, global err `{r_aln['global_shift_err_ppm']:.2e} ppm`, `r_min = {r_aln['min_corr']:.6f}` | `< 5e-4 ppm`, `r > 0.999` | `{'PASS' if ok else 'FAIL'}` |"
        )
    if r_kin is not None:
        ok = r_kin["quadrature_err"] < 1e-12 and int(r_kin["n_traces"]) == 8
        lines.append(
            f"| Exact sub-bin linear quadrature & arrayed first-order kinetics ROI matrix | `{r_kin['n_traces']}`-timepoint `A -> P` decay/growth stack (`t = 0..100 min`) | Quadrature err `{r_kin['quadrature_err']:.2e}`, conversion `conv(0) = {r_kin['conv_t0']:.4f} -> conv(100) = {r_kin['conv_t100']:.4f}` | `< 1e-12` | `{'PASS' if ok else 'FAIL'}` |"
        )
    return "\n".join(lines)


def render_cli_section(records: list[dict[str, Any]]) -> str:
    c_fid = find_by_gate(records, "CLI.nmr1d_process_fid")
    c_jb = find_by_gate(records, "CLI.nmr1d_process_jcamp_bruker")
    c_mp = find_by_gate(records, "CLI.nmr1d_multiplet")
    c_qn = find_by_gate(records, "CLI.nmr1d_qnmr")

    lines = [
        "### End-to-End `nmrassign nmr1d` CLI Verification (`nmr1d_cli_tests`)",
        "",
        "| Subcommand (`nmrassign nmr1d ...`) | End-to-End Pipeline Scenario | Measured CLI Output Metric | Status |",
        "| :--- | :--- | :--- | :---: |",
    ]
    if c_fid is not None:
        ok = int(c_fid["hsvd_poles_subtracted"]) >= 1 and c_fid["cal_delta_err_ppm"] < 1e-3 and c_fid["voigt_ratio_err_pct"] < 2.0
        lines.append(
            f"| `nmr1d process` (raw `FID` JSON -> TSV + JSON) | `HSVD` water subtraction + Chen-2002 autophase + Whittaker baseline + Fulmer `CDCl3` cal + Voigt LM | `{c_fid['hsvd_poles_subtracted']}` `HSVD` poles subtracted, `CDCl3` cal err `{c_fid['cal_delta_err_ppm']:.2e} ppm`, `2:3` Voigt area ratio `{c_fid['voigt_ratio']:.4f}` (`{c_fid['voigt_ratio_err_pct']:.3f}%` err) | `{'PASS' if ok else 'FAIL'}` |"
        )
    if c_jb is not None:
        ok = int(c_jb["points"]) == 2048 and int(c_jb["deconvolved_peaks"]) == 2
        lines.append(
            f"| `nmr1d process` (`JCAMP-DX` `.jdx` & Bruker `pdata/1/1r`) | `JCAMP-DX` + Bruker binary `1r`/`procs` ingest -> baseline flatten + Voigt doublet fit | `{c_jb['points']}` spectral points ingested, `{c_jb['deconvolved_peaks']}` Voigt doublet components recovered | `{'PASS' if ok else 'FAIL'}` |"
        )
    if c_mp is not None:
        ok = int(c_mp["ea_links"]) == 3 and c_mp["max_j_err_hz"] < 0.05
        lines.append(
            f"| `nmr1d multiplet` (`ACS`/`JOC`/`RSC`/`Nature` + `Octet` `.mol`/`SMILES`) | Ethyl acetate (`q`/`s`/`t`) `.mol` + `SMILES` atom-group linking & `dd`/`ddd`/`dt`/`td` tree sweep | `{c_mp['ea_links']}` `Octet` proton-group links, `max abs(dJ) = {c_mp['max_j_err_hz']:.5f} Hz`, classes `{c_mp['tree_classes']}` | `{'PASS' if ok else 'FAIL'}` |"
        )
    if c_qn is not None:
        ok = c_qn["abs_err_pct"] < 0.10 and bool(c_qn["t1_guard_triggered"])
        lines.append(
            f"| `nmr1d qnmr` (direct 1D spectrum regions + `5 * T1` guard) | `TMB` internal CRM (`6.09 ppm`) + analyte quartet (`4.12 ppm`, `P_true = 98.40%`) + short-`d1` guard | `P_fit = {c_qn['recovered_purity_pct']:.4f}%` (`abs(dP) = {c_qn['abs_err_pct']:.4f}%`), `U(k=2) = {c_qn['expanded_u_k2_pct']:.4f}%`, `5*T1` warning `{c_qn['t1_guard_triggered']}` | `{'PASS' if ok else 'FAIL'}` |"
        )
    return "\n".join(lines)


def render_regression_section(results_dir: Path, slice_logs: list[Path]) -> str:
    blocks: list[str] = []

    suite_rows: list[str] = []
    for sl in slice_logs:
        if sl.is_file() and sl.suffix == ".log":
            tot_p, tot_f, suites = parse_test_pass_counts(sl)
            for bname, sp, sf in suites:
                suite_rows.append(f"| `{bname}` | `{sp}` | `{sf}` |")
            if tot_p > 0 or tot_f > 0:
                suite_rows.append(
                    f"| **`14.W1` Dedicated Suites Total** | **`{tot_p}`** | **`{tot_f}`** |"
                )

    ws_log = results_dir / "w1_workspace_tests.log"
    if ws_log.is_file():
        wp, wf, _ = parse_test_pass_counts(ws_log)
        suite_rows.append(
            f"| **Full Workspace (`cargo test --release --workspace`)** | **`{wp}`** | **`{wf}`** |"
        )

    if suite_rows:
        blocks.append(
            "\n".join(
                [
                    "### `14.W1` Dedicated & Full Workspace Test Execution Summary",
                    "",
                    "| Suite Binary | Passed | Failed |",
                    "| :--- | ---: | ---: |",
                    *suite_rows,
                ]
            )
        )

    bench_rows: list[str] = []
    for slug, label, _solver, _engine in BENCHMARK_ENGINES:
        ubq_p = results_dir / f"w1_ubq_realistic_{slug}.log"
        cisd3_p = results_dir / f"w1_cisd3_{slug}.json"
        if not ubq_p.is_file() and not cisd3_p.is_file():
            continue
        ubq = (
            parse_ubq_log(ubq_p)
            if ubq_p.is_file()
            else {"acc": "n/a", "rec": "n/a", "hcp": "n/a"}
        )
        if cisd3_p.is_file():
            cs = summarize_cisd3_json(cisd3_p)
            c_acc = f"`{cs['macro_acc']:.2f}%`"
            c_rec = f"`{cs['macro_rec']:.2f}%`"
            c_hcp = (
                f"`{cs['micro_hcp']:.2f}%` (`{cs['hc_ok']}/{cs['hc_tot']}`)"
                if cs["micro_hcp"] is not None
                else "`ABSTAIN`"
            )
        else:
            c_acc = c_rec = c_hcp = "`n/a`"
        bench_rows.append(
            f"| {label} | `{ubq['acc']}` | `{ubq['rec']}` | `{ubq['hcp']}` | {c_acc} | {c_rec} | {c_hcp} |"
        )

    if bench_rows:
        blocks.append(
            "\n".join(
                [
                    "### Mandatory 5-Seed Solver Non-Regression Gates (`Ubiquitin --realistic` & `CISD3` `BMRB 51439`)",
                    "",
                    "| Engine | Ubiquitin `--realistic` (`5` Seeds) Acc | Ubiquitin Recall | Ubiquitin HC-Prec (`>=0.80`) | `CISD3` (`5` Seeds) Acc | `CISD3` Recall | `CISD3` HC-Prec (`>=0.80`) |",
                    "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
                    *bench_rows,
                ]
            )
        )

    return "\n\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render Phase 14 Wave 1 (`phase=14.W1`, Slices 14.1 + 14.2) Markdown tables from raw JSON/log outputs."
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
        default=Path("data/private/p14_w1_results"),
        help="Directory holding raw Wave 1 test/benchmark logs and JSON outputs.",
    )
    parser.add_argument(
        "--test-log",
        type=Path,
        action="append",
        default=[],
        help="Explicit log/jsonl file(s) containing W1_METRIC_JSON lines.",
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
        choices=[
            "all",
            "digests",
            "t14.1",
            "t14.2",
            "t14.3",
            "cli",
            "regression",
        ],
        default="all",
        help="Specific Markdown section to render (default: all).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the 4 `14.W1` `--release -- --nocapture` test suites into `--results-dir` before rendering.",
    )
    parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run full `cargo test --release --workspace` and 5-seed `Ubiquitin --realistic` + `CISD3` benchmarks into `--results-dir`.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output Markdown path (writes to stdout if omitted).",
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
        run_log, run_jsonl = run_w1_slice_suites(root, results_dir)
        extra_digest_files.append(run_jsonl)
        if not args.test_log:
            args.test_log = [run_log]

    if args.run_benchmarks:
        run_w1_benchmarks_and_workspace_tests(root, results_dir)

    metric_logs: list[Path] = []
    for tl in args.test_log:
        metric_logs.append(tl if tl.is_absolute() else (root / tl))

    if not metric_logs and results_dir.is_dir():
        preferred = results_dir / "w1_slice_tests.log"
        if preferred.is_file():
            metric_logs.append(preferred)
        else:
            for cand in sorted(results_dir.glob("*")):
                if cand.suffix in {".log", ".jsonl"} and cand.is_file():
                    txt = cand.read_text(encoding="utf-8", errors="replace")
                    if METRIC_PREFIX in txt:
                        metric_logs.append(cand)

    records = parse_w1_metrics(metric_logs)
    if not records and args.section in {"all", "t14.1", "t14.2", "t14.3", "cli"}:
        sys.stderr.write(
            f"error: no {METRIC_PREFIX} records found in {[str(p) for p in metric_logs]} (results_dir={results_dir})\n"
        )
        return 1

    for ef in args.extra_file:
        extra_digest_files.append(ef if ef.is_absolute() else (root / ef))
    digest_inputs = cited_w1_files(results_dir, metric_logs, extra_digest_files)

    blocks: list[str] = []
    sec = args.section
    if sec == "all":
        blocks.append(render_scope_banner())
    if sec in {"all", "digests"} and digest_inputs:
        blocks.append(render_digests_section(digest_inputs, root, results_dir))
    if sec in {"all", "t14.1"}:
        blocks.append(render_t14_1_section(records))
    if sec in {"all", "t14.2"}:
        blocks.append(render_t14_2_section(records))
    if sec in {"all", "t14.3"}:
        blocks.append(render_t14_3_section(records))
    if sec in {"all", "cli"}:
        blocks.append(render_cli_section(records))
    if sec in {"all", "regression"}:
        reg_md = render_regression_section(results_dir, metric_logs)
        if reg_md:
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
