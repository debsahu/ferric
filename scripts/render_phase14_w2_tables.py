#!/usr/bin/env python3
"""Deterministic Markdown table renderer for Phase 14 Wave 2 (Slices 14.3 + 14.4: T14.4 DOSY + T14.5 Metabolomics).

Reads raw `W2_METRIC_JSON:` structured records emitted by `--release -- --nocapture` runs of
`dosy_tests`, `dosy_gpu::tests`, `metabolomics_tests`, `dosy_panel::tests`,
`metabolomics_panel::tests`, and `dosy_metabolomics_cli_tests`, plus workspace test logs and
5-seed `Ubiquitin --realistic` / `CISD3` (`BMRB 51439`) benchmark artifacts under `--results-dir`,
and emits reproducible Markdown tables and FIPS 180-4 SHA-256 digests with zero hardcoded metrics.

Provenance: Original clean-room reporting script for algorithmic benchmarks citing Stejskal &
Tanner (J. Chem. Phys. 1965), Wu, Chen & Johnson (J. Magn. Reson. A 1995), Provencher
(Comput. Phys. Commun. 1982), Golub, Heath & Wahba (Technometrics 1979), Evans et al.
(Angew. Chem. Int. Ed. 2013), Savorani, Tomasi & Engelsen (J. Magn. Reson. 2010), De Meyer et al.
(Anal. Chem. 2008), Dieterle et al. (Anal. Chem. 2006), Trygg & Wold (J. Chemom. 2002),
Westerhuis et al. (Metabolomics 2008), Wiklund et al. (Anal. Chem. 2008), and Cloarec et al.
(Anal. Chem. 2005).
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

METRIC_PREFIX = "W2_METRIC_JSON:"

HOST_PATH_RE = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+")

SLICE_14_W2_COMMANDS = [
    ["cargo", "test", "--release", "-p", "nmr-core", "--test", "dosy_tests", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-gpu", "--lib", "dosy_gpu", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-gui", "--lib", "dosy_panel", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-core", "--test", "metabolomics_tests", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-gui", "--lib", "metabolomics_panel", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-cli", "--test", "dosy_metabolomics_cli_tests", "--", "--nocapture"],
]

BENCHMARK_ENGINES = [
    ("evograph", "`evo-graph` (`CPU f64`)", ["evo-graph", "cpu"]),
    ("standard_cpu", "`standard` (`CPU f64`)", ["standard", "cpu"]),
    ("standard_gpu", "`standard` (`WebGPU f32`)", ["standard", "gpu"]),
]


def sanitize_host_paths(text: str, root: Path) -> str:
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    root_s = str(root.resolve())
    out = out.replace(root_s, "<repo_root>")
    out = re.sub(
        r"/(?:Users|home)/[A-Za-z0-9_.-]+/Workspace/nmr_software",
        "<main_repo>",
        out,
    )
    return HOST_PATH_RE.sub("<user_home>", out)


def run_w2_slice_suite(root: Path, results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "w2_slice_14_3_14_4_tests.log"
    jsonl_path = results_dir / "w2_slice_14_3_14_4_metrics.jsonl"
    root_s = str(root.resolve())
    chunks: list[str] = []
    for cmd in SLICE_14_W2_COMMANDS:
        proc = subprocess.run(
            cmd,
            cwd=root_s,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        chunks.append(f"$ {' '.join(cmd)}\n{sanitize_host_paths(proc.stdout, root)}\n")
    log_path.write_text("".join(chunks), encoding="utf-8")

    records = parse_w2_metrics([log_path])
    jsonl_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    jsonl_path.write_text(jsonl_text, encoding="utf-8")
    return log_path, jsonl_path


def run_w2_benchmarks_and_workspace(
    root: Path, results_dir: Path, cisd3_data_root: Path | None
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    root_s = str(root.resolve())

    # 1. Full workspace test suite
    ws_cmd = ["cargo", "test", "--release", "--workspace"]
    ws_proc = subprocess.run(
        ws_cmd,
        cwd=root_s,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (results_dir / "w2_workspace_tests.log").write_text(
        f"$ {' '.join(ws_cmd)}\n{sanitize_host_paths(ws_proc.stdout, root)}\n",
        encoding="utf-8",
    )
    if ws_proc.returncode != 0:
        sys.stderr.write(ws_proc.stdout[-4000:])
        ws_proc.check_returncode()

    # 2. 5-seed Ubiquitin --realistic & 5-seed CISD3 (BMRB 51439)
    fd, tmp_manifest_str = tempfile.mkstemp(prefix="cisd3_w2_manifest_", suffix=".tsv")
    tmp_manifest = Path(tmp_manifest_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(
                "bmrb_id\tprotein\tresidues\tfield_mhz\tvendor\texperiments\tsplit\tcohort\n"
                "51439\tCISD3\t93\t900\tBruker\tHSQC,HNCO,HNCA,HNCOCA,CBCACONH\tdev\tdev\n"
            )
        for slug, _label, (solver, engine) in BENCHMARK_ENGINES:
            ubq_cmd = [
                "cargo",
                "run",
                "--release",
                "-p",
                "nmr-cli",
                "--",
                "benchmark",
                "--star",
                "data/fixtures/synthetic_ubiquitin.str",
                "--realistic",
                "--seeds",
                "5",
                "--solver",
                solver,
                "--engine",
                engine,
            ]
            ubq_proc = subprocess.run(
                ubq_cmd,
                cwd=root_s,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
            (results_dir / f"w2_ubq_realistic_{slug}.log").write_text(
                f"$ {' '.join(ubq_cmd)}\n{sanitize_host_paths(ubq_proc.stdout, root)}\n",
                encoding="utf-8",
            )

            if cisd3_data_root is not None and cisd3_data_root.is_dir():
                cisd3_json_p = results_dir / f"w2_cisd3_{slug}.json"
                cisd3_cmd = [
                    "cargo",
                    "run",
                    "--release",
                    "-p",
                    "nmr-cli",
                    "--",
                    "real-benchmark",
                    "--manifest",
                    str(tmp_manifest),
                    "--split",
                    "dev",
                    "--solver",
                    solver,
                    "--engine",
                    engine,
                    "--seeds",
                    "5",
                    "--data-root",
                    str(cisd3_data_root),
                    "--json-out",
                    str(cisd3_json_p),
                ]
                cisd3_proc = subprocess.run(
                    cisd3_cmd,
                    cwd=root_s,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True,
                )
                display_cmd = [
                    "<cisd3_manifest.tsv>" if a == str(tmp_manifest) else (
                        "data/private/bmrb" if a == str(cisd3_data_root) else (
                            f"data/private/p14_w2_results/{cisd3_json_p.name}"
                            if a == str(cisd3_json_p)
                            else a
                        )
                    )
                    for a in cisd3_cmd
                ]
                clean_stdout = sanitize_host_paths(cisd3_proc.stdout, root).replace(
                    str(tmp_manifest), "<cisd3_manifest.tsv>"
                )
                (results_dir / f"w2_cisd3_{slug}.log").write_text(
                    f"$ {' '.join(display_cmd)}\n{clean_stdout}\n",
                    encoding="utf-8",
                )
                if cisd3_json_p.is_file():
                    clean_json = sanitize_host_paths(
                        cisd3_json_p.read_text(encoding="utf-8"), root
                    ).replace(str(tmp_manifest), "<cisd3_manifest.tsv>")
                    cisd3_json_p.write_text(clean_json, encoding="utf-8")
    finally:
        if tmp_manifest.exists():
            tmp_manifest.unlink()


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
    if path.name.startswith("phase14_W2_review_pack"):
        return "data/private/phase14_W2_review_pack.md"
    rp = path.resolve()
    try:
        sub = rp.relative_to(results_dir.resolve()).as_posix()
        return f"data/private/p14_w2_results/{sub}"
    except ValueError:
        pass
    try:
        sub_priv = rp.relative_to(results_dir.resolve().parent).as_posix()
        return f"data/private/{sub_priv}"
    except ValueError:
        pass
    try:
        return rp.relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def parse_w2_metrics(paths: list[Path]) -> list[dict[str, Any]]:
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
            if payload:
                records.append(json.loads(payload))
        if p.suffix == ".jsonl" and METRIC_PREFIX not in text:
            for raw_line in text.splitlines():
                s = raw_line.strip()
                if s.startswith("{") and s.endswith("}"):
                    obj = json.loads(s)
                    if "kind" in obj:
                        records.append(obj)
    return records


def parse_test_pass_counts(log_path: Path) -> tuple[int, int, list[tuple[str, int, int]]]:
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    suites: list[tuple[str, int, int]] = []
    cur_bin = "suite"
    cur_cmd_label: str | None = None
    for line in txt.splitlines():
        if line.startswith("$ cargo test"):
            parts = line.split()
            pkg = parts[parts.index("-p") + 1] if "-p" in parts else ""
            if "--lib" in parts:
                lib_idx = parts.index("--lib")
                filt = parts[lib_idx + 1] if lib_idx + 1 < len(parts) and not parts[lib_idx + 1].startswith("-") else "lib"
                cur_cmd_label = f"{pkg}::{filt}" if pkg else filt
            elif "--test" in parts:
                t_idx = parts.index("--test")
                t_name = parts[t_idx + 1] if t_idx + 1 < len(parts) else "test"
                cur_cmd_label = f"{pkg}::{t_name}" if pkg else t_name
            else:
                cur_cmd_label = None
        m_run = re.search(r"Running (?:tests/|unittests )(\S+)", line)
        if m_run:
            raw_name = Path(m_run.group(1).split("(")[0].strip()).name
            cur_bin = cur_cmd_label if (cur_cmd_label and raw_name == "lib.rs") else (cur_cmd_label or raw_name)
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


def find_single(records: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    matches = [r for r in records if r.get("kind") == kind]
    return matches[-1] if matches else None


def find_all(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("kind") == kind]


def fmt_status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def cited_w2_files(
    root: Path, results_dir: Path, metric_logs: list[Path], extra_files: list[Path]
) -> list[Path]:
    candidates: list[Path] = [
        *metric_logs,
        results_dir / "w2_slice_14_3_14_4_metrics.jsonl",
        results_dir / "w2_workspace_tests.log",
    ]
    for slug, _label, _pair in BENCHMARK_ENGINES:
        candidates.append(results_dir / f"w2_ubq_realistic_{slug}.log")
        candidates.append(results_dir / f"w2_cisd3_{slug}.json")
        candidates.append(results_dir / f"w2_cisd3_{slug}.log")
    candidates.extend(extra_files)
    return [p for p in candidates if p.is_file()]


def render_digests_section(
    artifact_paths: list[Path], root: Path, results_dir: Path
) -> str:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in artifact_paths:
        if p.is_file():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                ordered.append(p)

    lines = [
        "### Cited Raw Results Artifacts & FIPS 180-4 SHA-256 Digests (`14.W2`)",
        "",
        "Generated mechanically by `python3 scripts/render_phase14_w2_tables.py --section digests`:",
        "",
        "| Artifact Path | Bytes | SHA-256 |",
        "| :--- | ---: | :--- |",
    ]
    for p in ordered:
        n_bytes, digest = sha256_and_size(p)
        rel = display_rel_path(p, root, results_dir)
        lines.append(f"| `{rel}` | {n_bytes:,} | `{digest}` |")
    return "\n".join(lines)


def render_t14_4_section(records: list[dict[str, Any]]) -> str:
    cpu_disc = find_single(records, "dosy_cpu_discrete")
    gpu_par = find_single(records, "dosy_webgpu_parity")
    ilt_list = find_all(records, "dosy_ilt_bimodal")
    hydro = find_single(records, "dosy_hydrodynamics")
    cli_fit = find_single(records, "dosy_fit_cli")

    lines = [
        "### Slice `14.3` (`T14.4`) — Pseudo-2D `DOSY` CPU/WebGPU Stejskal–Tanner, Opt-In Tikhonov `ILT` & Stokes–Einstein Hydrodynamics",
        "",
        "> **Validation Scope (`Synthetic Forward-Model Self-Consistency`)**: Every quantitative `T14.4` gate below evaluates synthetic forward-model Stejskal–Tanner / Wu–Johnson multi-exponential decays and analytical viscosity/Stokes–Einstein references (`0` real deposited raw Bruker `difflist`/`ser` or Varian `fid`/`procpar` experimental `DOSY` datasets evaluated yet — tracked openly as `F16`).",
        "",
        "#### Discrete 1–3 Component Stejskal–Tanner / Wu–Johnson CPU (`f64`) & WebGPU (`f32`) Fitting + `BIC` Selection",
        "",
        "| Verification Check | Engine | Measured Metric | Quantitative Gate | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]

    if cpu_disc is not None:
        n_tr = int(cpu_disc["n_trials"])
        k1_ok = int(cpu_disc["pure_k1_correct"])
        k2_ok = int(cpu_disc["mixed_k2_correct"])
        lines.append(
            f"| `BIC` model-order selection (`40` noisy seeds, `SNR=40` pure / `SNR=60` mixed) | CPU `f64` | Pure `K=1`: `{k1_ok}/{n_tr}` (`{100.0 * k1_ok / n_tr:.1f}%`), Mixed `K=2`: `{k2_ok}/{n_tr}` (`{100.0 * k2_ok / n_tr:.1f}%`) | `>= 95.0%` (`38/40`) | {fmt_status(bool(cpu_disc['passed']))} |"
        )
        lines.append(
            f"| Noisy diffusivity accuracy (`K=1`, `K=2`, and `32`-point `K=3` tri-exponential) | CPU `f64` | `K=1`: `{100.0 * cpu_disc['mean_pure_rel_err']:.2f}%`, `K=2`: `({100.0 * cpu_disc['mean_mix_d1_rel_err']:.2f}%, {100.0 * cpu_disc['mean_mix_d2_rel_err']:.2f}%)`, `K={cpu_disc['tri_selected_k']}` max err: `{100.0 * cpu_disc['max_tri_rel_err']:.4f}%` | `< 2.5%` / `< 8.0%` / `< 2.0%` | {fmt_status(bool(cpu_disc['passed']))} |"
        )

    if gpu_par is not None:
        lines.append(
            f"| Parallel column-wise WGSL `LM` (`{gpu_par['n_columns']}` cols, `K=1..2` + `BIC`) vs CPU `f64` | WebGPU `f32` (`{gpu_par['adapter']}`) | `100%` `BIC` order match; max `abs(dD)/D` mono `{100.0 * gpu_par['max_mono_rel_diff']:.4f}%`, bi `{100.0 * gpu_par['max_bi_rel_diff']:.4f}%` | `< 1.0%` GPU–CPU diff | {fmt_status(bool(gpu_par['passed']))} |"
        )

    if cli_fit is not None:
        lines.append(
            f"| `nmrassign dosy fit` CLI (`Bruker difflist/acqus` + `Varian procpar` + `--engine gpu`) | CPU `f64` + WebGPU `f32` | `K=1` err `{100.0 * cli_fit['k1_rel_err']:.3f}%`, `K=2` err `{100.0 * cli_fit['k2_rel_err_max']:.3f}%`, GPU–CPU `abs(d log10 D) = {cli_fit['max_gpu_dex_diff']:.2e}` dex | `< 1.5%` / `< 0.015` dex | {fmt_status(bool(cli_fit['passed']))} |"
        )

    lines.extend(
        [
            "",
            "#### Opt-In `CONTIN`-Style Second-Derivative Tikhonov `ILT` & Stokes–Einstein Hydrodynamics",
            "",
            "| Subsystem & Strategy | Selected `alpha` / Reference | Measured Recovery | Quantitative Gate | Status |",
            "| :--- | :--- | :--- | :--- | :---: |",
        ]
    )

    for rec in sorted(ilt_list, key=lambda r: str(r.get("strategy", ""))):
        lines.append(
            f"| Tikhonov `ILT` (`SecondDerivative` + `NNLS`, `{rec['strategy']}`) | `alpha = {rec['selected_lambda']:.3e}` (`{rec['n_modes']}` modes) | `abs(d log10 D) = ({rec['log10_err_slow']:.4f}, {rec['log10_err_fast']:.4f})` dex; areas `({100.0 * rec['area_slow']:.1f}%, {100.0 * rec['area_fast']:.1f}%)` | `< 0.08` dex, `50% +- 12%` | {fmt_status(bool(rec['passed']))} |"
        )

    if hydro is not None:
        lines.append(
            f"| Stokes–Einstein $R_h$ & empirical power-law `MW` (`D2O`, `298.15 K`) | Lysozyme (`14.3 kDa`) & Sucrose (`342.3 Da`) | Lysozyme: `R_h = {hydro['lysozyme_rh_a']:.2f} A`, `MW = {hydro['lysozyme_mw_da']:.0f} Da`; Sucrose: `R_h = {hydro['sucrose_rh_a']:.2f} A`, `MW = {hydro['sucrose_mw_da']:.1f} Da` | `[18.8, 20.2] A` (`< 8%` MW) / `[4.4, 4.8] A` (`< 10%` MW) | {fmt_status(bool(hydro['passed']))} |"
        )

    return "\n".join(lines)


def render_t14_5_section(records: list[dict[str, Any]]) -> str:
    pqn = find_single(records, "metab_pqn_vs_tsa")
    aib = find_single(records, "metab_aib_vs_uniform")
    opls_o = find_single(records, "metab_oplsda_ortho")
    opls_n = find_single(records, "metab_oplsda_null")
    stocsy = find_single(records, "metab_stocsy_lactate")
    cli_e2e = find_single(records, "metabolomics_e2e_cli")
    cli_null = find_single(records, "metabolomics_null_guard_cli")

    lines = [
        "### Slice `14.4` (`T14.5`) — Untargeted 1D NMR Metabolomics (`AIB`, `PQN`, `PCA`, `OPLS-DA` & `STOCSY`, Spec `20`)",
        "",
        "> **Validation Scope (`Synthetic Forward-Model Self-Consistency`)**: Every quantitative `T14.5` gate below evaluates synthetic biofluid cohorts with planted dilution factors, pH jitter, orthogonal within-group drift, and spiked metabolite fold changes against the built-in 28-entry CC0 `BMRB` canonical reference table (`0` deposited public MetaboLights experimental 1D cohorts such as `MTBLS1` evaluated yet — tracked openly as `F17`).",
        "",
        "#### Preprocessing (`PQN` Dilution Invariance & `De Meyer 2008` Adaptive Intelligent Binning)",
        "",
        "| Preprocessing Gate | Perturbation / Comparison | Measured Metric | Quantitative Gate | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]

    if pqn is not None:
        lines.append(
            f"| `PQN` (`Dieterle 2006`) vs `TSA` dilution & spike invariance | `2x` & `5x` dilution + `18x` D-Glucose spike | `PQN` max non-spiked error `{100.0 * pqn['max_pqn_rel_err']:.3f}%` vs `TSA` distortion `{100.0 * pqn['min_tsa_spike_rel_err']:.1f}%` | `PQN < 1.0%` (`TSA > 40%`) | {fmt_status(bool(pqn['passed']))} |"
        )

    if aib is not None:
        lines.append(
            f"| `Adaptive Intelligent Binning` (`AIB`) vs `0.04 ppm` uniform bins | L-Lactate `4.11 ppm` quartet & L-Valine `0.985 ppm` doublet | `AIB` single-bucket capture `({100.0 * aib['aib_lac_q_share']:.1f}%, {100.0 * aib['aib_val_1_share']:.1f}%)` vs Uniform split `({100.0 * aib['uni_lac_q_share']:.1f}%, {100.0 * aib['uni_val_1_share']:.1f}%)` | `AIB > 95.0%` | {fmt_status(bool(aib['passed']))} |"
        )

    lines.extend(
        [
            "",
            "#### Chemometrics (`PCA`, Leak-Free Stratified `OPLS-DA`, Null-Label Guard & `STOCSY` Biomarker Identification)",
            "",
            "| Chemometrics / CLI Gate | Cohort / Driver | Measured Metric | Quantitative Gate | Status |",
            "| :--- | :--- | :--- | :--- | :---: |",
        ]
    )

    if opls_o is not None:
        lines.append(
            f"| `OPLS-DA` (`1 + 1` NIPALS, `4.5x` orthogonal drift) | `N = 24`, `P = 80`, `7`-fold stratified CV + `100` perms | `R2X_o = {opls_o['r2_x_orthogonal']:.3f}` vs `R2X_p = {opls_o['r2_x_predictive']:.3f}`, `R2Y = {opls_o['r2_y']:.4f}`, `Q2 = {opls_o['q2']:.4f}`, `perm_p = {opls_o['perm_p_q2']:.4f}` | `Q2 > 0.85`, `perm_p <= 0.02` | {fmt_status(bool(opls_o['passed']))} |"
        )

    if opls_n is not None:
        lines.append(
            f"| Anti-overfitting null-label guard (`Westerhuis 2008`) | `P = 200 >> N = 24` shuffled labels (`120` perms) | In-sample `R2Y = {opls_n['in_sample_r2_y']:.3f}`, leak-free `Q2 = {opls_n['null_q2']:.4f}`, `perm_p = {opls_n['null_perm_p_q2']:.4f}`, `validated = {opls_n['is_validated']}` | `Q2 < 0.10`, `perm_p > 0.10`, `validated = false` | {fmt_status(bool(opls_n['passed']))} |"
        )

    if stocsy is not None:
        lines.append(
            f"| `STOCSY` (`Cloarec 2005`) + CC0 `BMRB` library match | Driver `{stocsy['driver_ppm']:.3f} ppm` (`L-Lactate` `CH3` doublet) | Recovers `{stocsy['quartet_ppm']:.3f} ppm` `CH` quartet (`r = {stocsy['quartet_r']:.4f}`, `r^2 = {stocsy['quartet_r2']:.4f}`, `cov 3H:1H = {stocsy['stoich_ratio_3_to_1']:.2f}:1`), #1 `{stocsy['top_bmrb_id']}` (`score = {stocsy['top_score']:.3f}`) | `r^2 > 0.90`, `#1 = bmse000036` | {fmt_status(bool(stocsy['passed']))} |"
        )

    if cli_e2e is not None and cli_null is not None:
        lines.append(
            f"| `nmrassign metabolomics bin / pca / oplsda / stocsy` CLI | `24`-sample demo + null-shuffled guard | `P = {cli_e2e['n_buckets']}`, `PC1 = {100.0 * cli_e2e['pca_pc1_var']:.1f}%`, `OPLS Q2 = {cli_e2e['opls_q2']:.4f}` (`p = {cli_e2e['opls_perm_p']:.4f}`), null `Q2 = {cli_null['null_q2']:.4f}` (`OVERFIT WARNING`), `STOCSY #1 = {cli_e2e['stocsy_top_bmrb_id']}` | End-to-end CLI parity | {fmt_status(bool(cli_e2e['passed'] and cli_null['passed']))} |"
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
                    f"| **Wave `14.W2` Dedicated Total** | **`{tot_p}`** | **`{tot_f}`** |"
                )

    ws_log = results_dir / "w2_workspace_tests.log"
    if ws_log.is_file():
        wp, wf, _ = parse_test_pass_counts(ws_log)
        suite_rows.append(
            f"| **Full Workspace (`cargo test --release --workspace`)** | **`{wp}`** | **`{wf}`** |"
        )

    if suite_rows:
        blocks.append(
            "\n".join(
                [
                    "### Wave `14.W2` & Workspace Test Execution Summary",
                    "",
                    "| Suite Binary | Passed | Failed |",
                    "| :--- | ---: | ---: |",
                    *suite_rows,
                ]
            )
        )

    bench_rows: list[str] = []
    for slug, label, _pair in BENCHMARK_ENGINES:
        ubq_p = results_dir / f"w2_ubq_realistic_{slug}.log"
        cisd3_p = results_dir / f"w2_cisd3_{slug}.json"
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
        description="Render Phase 14 Wave 2 (Slices 14.3 + 14.4: DOSY + Metabolomics) Markdown tables from raw JSON/log outputs."
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
        default=Path("data/private/p14_w2_results"),
        help="Directory holding raw Phase 14 Wave 2 test/benchmark logs and JSON outputs.",
    )
    parser.add_argument(
        "--test-log",
        type=Path,
        action="append",
        default=[],
        help="Explicit log/jsonl file(s) containing W2_METRIC_JSON lines.",
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
            "14.3",
            "14.4",
            "t14.4",
            "t14.5",
            "regression",
        ],
        default="all",
        help="Specific Markdown section to render (default: all).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the 6 Wave 14.W2 `--release -- --nocapture` test suites into `--results-dir` before rendering.",
    )
    parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Also run full workspace tests + 5-seed Ubiquitin `--realistic` & 5-seed `CISD3` non-regression gates.",
    )
    parser.add_argument(
        "--cisd3-data-root",
        type=Path,
        default=None,
        help="Directory containing `bmrb51439_cisd3` when running `--run-benchmarks`.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output Markdown path (also writes to stdout when omitted).",
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
        run_log, run_jsonl = run_w2_slice_suite(root, results_dir)
        extra_digest_files.append(run_jsonl)
        if not args.test_log:
            args.test_log = [run_log]

    if args.run_benchmarks:
        cisd3_root = args.cisd3_data_root
        if cisd3_root is None:
            cand = root / "data/private"
            if (cand / "bmrb51439_cisd3").is_dir():
                cisd3_root = cand
        run_w2_benchmarks_and_workspace(root, results_dir, cisd3_root)

    metric_logs: list[Path] = []
    for tl in args.test_log:
        metric_logs.append(tl if tl.is_absolute() else (root / tl))

    if not metric_logs and results_dir.is_dir():
        preferred = results_dir / "w2_slice_14_3_14_4_tests.log"
        if preferred.is_file():
            metric_logs.append(preferred)
        else:
            for cand in sorted(results_dir.glob("*")):
                if cand.suffix in {".log", ".jsonl"} and cand.is_file():
                    txt = cand.read_text(encoding="utf-8", errors="replace")
                    if METRIC_PREFIX in txt:
                        metric_logs.append(cand)

    records = parse_w2_metrics(metric_logs)
    if not records and args.section in {"all", "14.3", "14.4", "t14.4", "t14.5"}:
        sys.stderr.write(
            f"error: no {METRIC_PREFIX} records found in {[str(p) for p in metric_logs]} (results_dir={results_dir})\n"
        )
        return 1

    for ef in args.extra_file:
        extra_digest_files.append(ef if ef.is_absolute() else (root / ef))
    digest_inputs = cited_w2_files(root, results_dir, metric_logs, extra_digest_files)

    blocks: list[str] = []
    sec = args.section
    if sec in {"all", "digests"} and digest_inputs:
        blocks.append(render_digests_section(digest_inputs, root, results_dir))
    if sec in {"all", "14.3", "t14.4"}:
        blocks.append(render_t14_4_section(records))
    if sec in {"all", "14.4", "t14.5"}:
        blocks.append(render_t14_5_section(records))
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
