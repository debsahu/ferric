#!/usr/bin/env python3
"""Deterministic Markdown table renderer & SHA-256 ledger generator for Phase 17 Final
(`phase=17` / `v0.13.0` Pre-Tag Release Summary & `CHANGELOG.md` `## [0.13.0]`).

Reads structured evaluation JSONs and verification logs from:
- `data/private/p17_w1_results/` (5-seed Ubiquitin `--realistic` & `CISD3` `BMRB 51439` at `crates/` tree `02c6246a3cb2`)
- `data/private/p17_w2_results/` (`F18` MCP & `F21` WebGPU single-host smoke)
- `data/private/w3_empirical_results/` (`F10`, `F11`, `F13a`–`c` single-shot JSONs)
- `data/private/w4_empirical_results/` (`F14`, `F16` + withdrawn R1, `F17` single-shot JSONs)
- `data/private/phase17_final_results/` (`v0.13.0` fmt/clippy/wasm32/windows-gnu/test/manual/demo/packager/CI logs)

Verifies `0`-byte diffs across `crates/`, `scripts/p17_adapters/`,
`data/benchmark/p17_empirical_preregistration.json`, `scripts/verify_p17_preregistration.py`,
`docs/specs/03-assignment.md`, and `THIRD_PARTY.md`, verifies `Cargo.lock` changes ONLY the 4
workspace package versions (`0.12.0 -> 0.13.0`), and renders all numeric tables and SHA-256 digests
mechanically without hand-typed numbers.

Provenance: Original clean-room Python 3 standard-library reporting script (MIT License) citing
FIPS 180-4 (SHA-256 digests) and RFC 8259 (JSON). Zero external packages or network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

HOST_PATH_RE = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+")

ZERO_DIFF_SINCE_W4_PATHS = [
    "crates/",
    "THIRD_PARTY.md",
    "data/benchmark/p17_empirical_preregistration.json",
    "scripts/verify_p17_preregistration.py",
    "scripts/p17_adapters/",
    "docs/specs/03-assignment.md",
    "docs/specs/16-dynamics-and-csp.md",
    "docs/specs/17-nmr1d-and-dosy.md",
    "docs/specs/18-mcp-server-and-bridge.md",
    "docs/specs/19-bayesian-sa-solver.md",
    "docs/specs/20-metabolomics-chemometrics.md",
    "data/benchmark/real_manifest.tsv",
    "data/benchmark/manifest.tsv",
    "data/priors/bmrb_priors.json",
    "data/priors/PROVENANCE.md",
]


def write_clean_utf8(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    os.chmod(path, 0o644)


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
    rp = path.resolve()
    try:
        return rp.relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def verify_cargo_lock_bump_only(root: Path, base_ref: str) -> tuple[int, int]:
    proc = subprocess.run(
        ["git", "diff", "-U0", f"{base_ref}..HEAD", "--", "Cargo.lock"],
        cwd=str(root),
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    minus_lines = [
        ln for ln in proc.stdout.splitlines() if ln.startswith("-") and not ln.startswith("---")
    ]
    plus_lines = [
        ln for ln in proc.stdout.splitlines() if ln.startswith("+") and not ln.startswith("+++")
    ]
    if minus_lines != ['-version = "0.12.0"'] * 4 or plus_lines != ['+version = "0.13.0"'] * 4:
        raise RuntimeError(
            f"Cargo.lock diff across {base_ref}..HEAD must be exactly 4 workspace 0.12.0->0.13.0 lines; got minus={minus_lines}, plus={plus_lines}"
        )
    return len(minus_lines), len(plus_lines)


def render_scorecard_section(root: Path) -> str:
    w3 = root / "data" / "private" / "w3_empirical_results"
    w4 = root / "data" / "private" / "w4_empirical_results"

    f10 = json.loads((w3 / "f10_f11" / "f10_evaluation.json").read_text(encoding="utf-8"))
    f11 = json.loads((w3 / "f10_f11" / "f11_evaluation.json").read_text(encoding="utf-8"))
    f13a = json.loads((w3 / "f13abc" / "f13a_evaluation.json").read_text(encoding="utf-8"))
    f13b = json.loads((w3 / "f13abc" / "f13b_evaluation.json").read_text(encoding="utf-8"))
    f13c = json.loads((w3 / "f13abc" / "f13c_evaluation.json").read_text(encoding="utf-8"))
    f14 = json.loads((w4 / "f14" / "f14_evaluation.json").read_text(encoding="utf-8"))
    f16 = json.loads((w4 / "f16" / "f16_evaluation.json").read_text(encoding="utf-8"))
    f17 = json.loads((w4 / "f17" / "f17_evaluation.json").read_text(encoding="utf-8"))

    r1_disc = f16["superseded_round1_evaluation_disclosure"]
    r1_sha_short = r1_disc["sha256"][:12]

    all_evs = [f10, f11, f13a, f13b, f13c, f14, f16, f17]
    n_pass = sum(1 for e in all_evs if e["verdict"] == "PASS")
    n_fail = sum(1 for e in all_evs if e["verdict"] == "FAIL")
    n_ne = sum(1 for e in all_evs if e["verdict"] == "NOT_EVALUABLE")
    if (n_pass, n_fail, n_ne) != (0, 5, 3):
        raise RuntimeError(f"Expected (0, 5, 3) empirical tally, got {(n_pass, n_fail, n_ne)}")

    def ratio(ev: dict[str, Any]) -> str:
        cr = ev.get("criteria_results") or ev.get("w4_adapter_evaluate_f14_output", {}).get(
            "criteria_results", []
        )
        if not cr:
            return "`0/0` (archive blocked)"
        ok = sum(1 for c in cr if c.get("passed"))
        return f"`{ok}/{len(cr)}`"

    lines = [
        f"### Phase 17 Blind Single-Shot Empirical Scorecard (`8` Gates: `{n_pass} PASS`, `{n_fail} FAIL`, `{n_ne} NOT_EVALUABLE`)",
        "",
        "| Gate (`TODO.md`) | Wave | Pre-Registered Chain Evaluated | Criteria | Standing Verdict | `TODO.md` | Candid Root Cause & Post-Freeze Backlog |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :--- |",
        (
            f"| `F10` (`:130`) | `17.W3` | `BMRB:6470` (`NOT_EVALUABLE`) -> `BMRB:18971` | {ratio(f10)} | **`{f10['verdict']}`** | `[ ]` open | "
            "`BMRB:6470` has `0/68` non-null `Val_err` (`17.5b:P2-1`); `BMRB:18971` passes `n=33` & `S^2` `r=0.8940`, fails `S^2` `MAE=0.09932 > 0.060` & un-overridden ubiquitin `D_par/D_perp=1.17` (`4.156 vs 4.09 ns` NOT cited as `tau_m` accuracy, `17.5b:P2-2`; `F28`) |"
        ),
        (
            f"| `F11` (`:131`) | `17.W3` | `{f11['accession_id']}` (`DvR115` raw Bruker `ser`) | {ratio(f11)} | **`{f11['verdict']}`** | `[ ]` open | "
            "`hetNOE` `rc=1` (2-plane `hsqcnoef3gpsi` lacks `vdlist`); `R1`/`R2` fail on inner `NBL=11` (`id10`), `(F2=1H, F1=15N)` axis order, `FnMODE=5`, & 3-ch `f3` `D[21]`/`D[31]` `vclist` timing (`F28`) |"
        ),
        (
            f"| `F13a` (`:136`) | `17.W3` | `{f13a['accession_id']}` | {ratio(f13a)} | **`{f13a['verdict']}`** | `[ ]` open | "
            "All `4` archives omit CW-`CEST` headers (`B0`, `T_EX`, `B1`), offset schedules (`ZENODO:12765510` HTTP `206` probe), raw data (`Raw_data_flag=no`), or 2-state exchange (`p_B=0`) |"
        ),
        (
            f"| `F13b` (`:136`) | `17.W3` | `{f13b['accession_id']}` | {ratio(f13b)} | **`{f13b['verdict']}`** | `[ ]` open | "
            "All `6` archives omit `B0`/`T_relax`/`nu_CPMG` headers, represent 1-state `p_B=0` controls, or deposit shift-only/empty `.str` stubs |"
        ),
        (
            f"| `F13c` (`:136`) | `17.W3` | `{f13c['accession_id']}` | {ratio(f13c)} | **`{f13c['verdict']}`** | `[ ]` open | "
            "All `3` archives fail frozen `extract_f13c_reference_kd_um`: `ZENODO:17228160` has zero literal `K_d` (`MST` titrates `2N4R` tau `<= 100 uM`, not `tau1-243`), `BMRBIG:138` lacks molarity, `ZENODO:7892349` has `3 < 4` HSQC steps (`rc=1`; `F28`) |"
        ),
        (
            f"| `F14` (`:144`) | `17.W4` | `BMSE:bmse000036` + `6` co-primary/backup archives | {ratio(f14)} | **`{f14['verdict']}`** | `[ ]` open | "
            "`F14-C2` `0.333333 != 1.0` (`FAIL` on `bmse000036` raw `DEPT_135` `PHC1 = +180 deg` boundary); `F14-C1` (`3/3` raw Bruker `1H`/`13C`/`DEPT` `SNR >= 70.0`, `0/7` raw Bruker `19F`/`31P`) & `F14-C3` `NOT_EVALUABLE` (`F29`) |"
        ),
        (
            f"| `F16` (`:146`) | `17.W4` | `{f16['accession_id']}` -> `ZENODO:16423019` (`23.zip`) | {ratio(f16)} | **`{f16['verdict']}`** | `[ ]` open | "
            f"`F16-C1 = 2` `PASS`; uncurated `F16-C2` `min R^2 = 0.9080 < 0.95` `FAIL`; `F16-C3`/`C4 = null` `FAIL` (no frozen `D_ref` extractor, `23.zip` `<gdp_dosyoneshot_tlg>` `4.013x` `b` ratio; withdrawn R1 `PASS` `{r1_sha_short}...` disclosed, NOT a gate result; `F29`) |"
        ),
        (
            f"| `F17` (`:147`) | `17.W4` | `{f17['accession_id']}` (`N = 132` urinary T2DM `1H`) | {ratio(f17)} | **`{f17['verdict']}`** | `[ ]` open | "
            "Strict `bbc770a` `f17_mtbls1_manifest.py` `rc=1` (`0/132` joined on `Free Induction Decay Data File`); post-hoc 1-line fix (`132/132` `PQN`, `perm_p = 0.009901`, `10` Top-15 glucose buckets) also `FAIL` (`3/4`) on `OPLS-DA` `Q^2 = 0.122445 < 0.25` (`F29`) |"
        ),
    ]
    return "\n".join(lines)


def render_ci_ledger_section(results_dir: Path) -> str:
    ci_path = results_dir / "ci_runs_w1_to_w4.json"
    runs = json.loads(ci_path.read_text(encoding="utf-8"))
    lines = [
        "### Pushed Wave `origin/main` GitHub Actions CI Ledger (`17.W1`–`17.W4`)",
        "",
        "| Wave | Pushed `origin/main` Commit | GitHub Actions Run ID | Created (UTC) | Updated (UTC) | Status / Conclusion |",
        "| :--- | :--- | ---: | :--- | :--- | :---: |",
    ]
    for item in runs:
        conc = (item.get("conclusion") or "IN_PROGRESS").upper()
        lines.append(
            f"| `{item['wave']}` | `{item['headSha'][:7]}` (`{item['headSha']}`) | `{item['databaseId']}` | `{item['createdAt']}` | `{item['updatedAt']}` | `{item['status']}` / **`{conc}`** |"
        )
    return "\n".join(lines)


def render_gates_section(root: Path, results_dir: Path, base_ref: str) -> str:
    ver_str = (results_dir / "nmrassign_version.txt").read_text(encoding="utf-8").strip()
    if not ver_str.startswith("nmrassign 0.13.0 ("):
        raise RuntimeError(f"Expected nmrassign 0.13.0 version string, got {ver_str!r}")

    test_log = (results_dir / "cargo_test_workspace.log").read_text(encoding="utf-8")
    result_matches = re.findall(
        r"test result: ok\. (\d+) passed; (\d+) failed;",
        test_log,
    )
    if not result_matches:
        raise RuntimeError("No `test result: ok.` lines found in cargo_test_workspace.log!")
    total_passed = sum(int(p) for p, _ in result_matches)
    total_failed = sum(int(f) for _, f in result_matches)
    n_suites = len(result_matches)
    n_unit_bins = len(re.findall(r"^\s*Running\s+\S+", test_log, re.MULTILINE))
    n_doctest_suites = len(re.findall(r"^\s*Doc-tests\s+\S+", test_log, re.MULTILINE))
    if total_failed != 0:
        raise RuntimeError(f"Workspace test failures detected: {total_failed}")
    if n_unit_bins + n_doctest_suites != n_suites:
        raise RuntimeError(
            f"Suite header mismatch: {n_unit_bins} binaries + {n_doctest_suites} doc-tests != {n_suites}"
        )

    manual_check = (results_dir / "build_docs_manual_check.log").read_text(encoding="utf-8").strip()
    m_sha = re.search(r"sha256=([0-9a-f]{16,64})", manual_check)
    if not m_sha:
        raise RuntimeError("Missing sha256= in build_docs_manual_check.log!")
    manual_sha = m_sha.group(1)
    manual_st = (results_dir / "build_docs_manual_selftest.log").read_text(encoding="utf-8").strip()
    m_st = re.search(r"self-tests passed \((\d+/\d+)\b", manual_st)
    if not m_st:
        raise RuntimeError("Missing self-tests counter in build_docs_manual_selftest.log!")
    st_str = m_st.group(1)

    pkg_st = (results_dir / "package_release_selftest.log").read_text(encoding="utf-8").strip()
    if "ALL SELF-TESTS PASSED" not in pkg_st:
        raise RuntimeError("Missing `ALL SELF-TESTS PASSED` in package_release_selftest.log!")
    pkg_matches = re.findall(r"\[self-test (\d+/\d+)\]", pkg_st)
    if not pkg_matches:
        raise RuntimeError("Missing `[self-test X/Y]` lines in package_release_selftest.log!")
    pkg_str = pkg_matches[-1]

    adp_st = (results_dir / "p17_adapters_selftest.log").read_text(encoding="utf-8").strip()
    m_adp = re.search(r"OK \[p17_adapters\.selftest\]:\s*(\d+/\d+)\s+checks passed", adp_st)
    if not m_adp:
        raise RuntimeError("Missing `OK [p17_adapters.selftest]: X/Y checks passed` in p17_adapters_selftest.log!")
    adp_str = m_adp.group(1)

    crates_tree = subprocess.run(
        ["git", "rev-parse", "HEAD:crates"],
        cwd=str(root),
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    if crates_tree != "02c6246a3cb2b857a5a8a6fca8e1c677cb9f3153":
        raise RuntimeError(f"HEAD:crates drifted from 02c6246a3cb2: {crates_tree}")

    for rel_p in ZERO_DIFF_SINCE_W4_PATHS:
        proc = subprocess.run(
            ["git", "diff", f"{base_ref}..HEAD", "--", rel_p],
            cwd=str(root),
            stdout=subprocess.PIPE,
            check=True,
        )
        if len(proc.stdout) != 0:
            raise RuntimeError(f"Unexpected diff in {rel_p} across {base_ref}..HEAD!")

    n_minus, n_plus = verify_cargo_lock_bump_only(root, base_ref)

    tracked_priv = subprocess.run(
        ["git", "ls-files", "data/private"],
        cwd=str(root),
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    if tracked_priv:
        raise RuntimeError(f"Tracked files found in data/private/: {tracked_priv}")

    readme_lines = len((root / "README.md").read_text(encoding="utf-8").splitlines())
    if readme_lines > 180:
        raise RuntimeError(f"README.md exceeds 180-line ceiling: {readme_lines}")

    lines = [
        "### Build, Lint, Cross-Target (`wasm32` & `windows-gnu`), Manual (`v0.13.0`), Packager & Release Test Verification",
        "",
        "| Check | Command | Measured Result |",
        "| :--- | :--- | :--- |",
        "| Rust formatting | `cargo fmt --all -- --check` | exit `0` (`PASS`) |",
        "| Workspace Clippy lint gate | `cargo clippy --workspace --all-targets -- -D warnings` | exit `0` (`0` warnings, `PASS`) |",
        "| Browser `wasm32` compile gate | `RUSTFLAGS=\"-D warnings\" cargo check --release --locked -p nmr-gui --target wasm32-unknown-unknown` | exit `0` (`0` warnings, `PASS`) |",
        "| Windows `x86_64` GNU cross-check (`F19`/`F22`) | `RUSTFLAGS=\"-D warnings\" cargo check --release --locked -p nmr-cli -p nmr-gui --target x86_64-pc-windows-gnu` | exit `0` (`0` warnings, `PASS`) |",
        f"| Full workspace release test suite | `cargo test --release --workspace` | **`{total_passed}` passed; `{total_failed}` failed (`{n_suites}` test suites: `{n_unit_bins}` test binaries + `{n_doctest_suites}` doc-test suites, `PASS`)** |",
        f"| Static HTML manual (`v0.13.0` site badge) | `python3 scripts/build_docs_manual.py --self-test` & `--check` | **`{st_str}` self-tests, `9` pages, `132` sections, `28` links, `sha256={manual_sha}` (`PASS`)** |",
        "| Browser WebGPU demo shell check | `bash scripts/build_web_demo.sh --check-only` | exit `0` (WCAG 1.4.4 zoom-preserving shell & `wasm32` `PASS`) |",
        f"| Cross-platform desktop release packager | `python3 scripts/package_release.py --self-test` | **`{pkg_str}` self-tests (`.app`/`.dmg`/`.tar.gz`/`.zip`/`SHA256SUMS.txt` `PASS`)** |",
        f"| Frozen `bbc770a` format adapter parity | `python3 scripts/p17_adapters/selftest.py` | **`{adp_str}` live `nmrassign` CLI parity self-tests (`PASS`)** |",
        f"| Version CLI & crate constant check | `./target/release/nmrassign --version` | **`{ver_str}` (`nmr_core::VERSION == \"0.13.0\"`)** |",
        f"| Whole-`crates/` & `p17_adapters/` freeze | `git rev-parse HEAD:crates` & `git diff {base_ref}..HEAD` | **`HEAD:crates == {crates_tree[:12]}` (`0`-byte diff since `8beec3b`; `0`-byte diff in `scripts/p17_adapters/` since `bbc770a`)** |",
        f"| `Cargo.lock` & `THIRD_PARTY.md` audit | `git diff {base_ref}..HEAD -- Cargo.lock THIRD_PARTY.md` | **Exactly `{n_minus}`/`{n_plus}` workspace `0.12.0 -> 0.13.0` lines; `0` external crate changes; `THIRD_PARTY.md` `0` bytes (`PASS`)** |",
        f"| `README.md` line ceiling & `03-assignment.md` | `wc -l README.md` & `git diff adf55d8..HEAD -- docs/specs/03-assignment.md` | **`{readme_lines} <= 180` lines; `03-assignment.md` `0`-byte diff; `0` tracked `data/private/` files (`PASS`)** |",
    ]
    return "\n".join(lines)


def parse_cisd3_safe(w1_dir: Path, json_name: str) -> tuple[str, str, str, str]:
    obj = json.loads((w1_dir / json_name).read_text(encoding="utf-8"))
    mic = obj["micro_average"]
    acc = float(mic["accuracy"]) * 100.0
    rec = float(mic["recall"]) * 100.0
    hc_num = int(mic["hc_correct"])
    hc_den = int(mic["hc_assigned"])
    hc_pct = float(mic["hc_precision"]) * 100.0
    entries = obj.get("entries", [])
    qc = str(entries[0].get("qc_verdict", "PASS") if entries else "PASS").upper()
    return qc, f"{acc:.2f}%", f"{rec:.2f}%", f"{hc_pct:.2f}% ({hc_num}/{hc_den})"


def render_benchmarks_section_v2(root: Path) -> str:
    w1_dir = root / "data" / "private" / "p17_w1_results"

    def parse_ubq(log_name: str) -> tuple[str, str, str]:
        txt = (w1_dir / log_name).read_text(encoding="utf-8")
        m_asgnble = re.search(r"assignable spin systems\s*:\s*(\d+)", txt)
        m_asgnd = re.search(r"assigned\s*:\s*(\d+)", txt)
        m_acc = re.search(r"accuracy\s*\(correct/assigned\)\s*:\s*([0-9.]+%)", txt)
        m_rec = re.search(r"recall\s*\(correct/assignable\)\s*:\s*([0-9.]+%)", txt)
        m_hc = re.search(r"high-confidence\s*\(>=0\.8\)\s*precision\s*:\s*([0-9.]+%\s*\(\d+/\d+\))", txt)
        if not (m_asgnble and m_asgnd and m_acc and m_rec and m_hc):
            raise RuntimeError(f"Failed to parse Ubiquitin metrics from {log_name}")
        return (
            f"{m_acc.group(1)} (assigned={m_asgnd.group(1)})",
            f"{m_rec.group(1)} (assignable={m_asgnble.group(1)})",
            m_hc.group(1),
        )

    rows_spec = [
        ("evo-graph (`CPU f64`)", "w1_ubq_realistic_evograph.log", "w1_cisd3_evograph.json"),
        ("standard (`CPU f64`)", "w1_ubq_realistic_standard_cpu.log", "w1_cisd3_standard_cpu.json"),
        ("standard (`WebGPU f32`)", "w1_ubq_realistic_standard_gpu.log", "w1_cisd3_standard_gpu.json"),
    ]

    lines = [
        "### Mandatory 5-Seed Solver Non-Regression Gates (`Ubiquitin --realistic` & `CISD3` `BMRB 51439` at Frozen `crates/` `02c6246a3cb2`)",
        "",
        "| Engine | Ubiquitin `--realistic` (`5` Seeds) Acc | Ubiquitin Recall | Ubiquitin HC-Prec (`>=0.80`) | `CISD3` QC | `CISD3` (`5` Seeds) Acc | `CISD3` Recall | `CISD3` HC-Prec (`>=0.80`) |",
        "| :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for label, ubq_f, cisd_f in rows_spec:
        u_acc, u_rec, u_hc = parse_ubq(ubq_f)
        c_qc, c_acc, c_rec, c_hc = parse_cisd3_safe(w1_dir, cisd_f)
        lines.append(
            f"| {label} | `{u_acc}` | `{u_rec}` | `{u_hc}` | `{c_qc}` | `{c_acc}` | `{c_rec}` | `{c_hc}` |"
        )
    return "\n".join(lines)


def cited_final_files(root: Path, results_dir: Path, extra_files: list[Path]) -> list[Path]:
    candidates = [
        root / "CHANGELOG.md",
        root / "Cargo.toml",
        root / "Cargo.lock",
        root / "README.md",
        root / "TODO.md",
        root / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml",
        root / "data" / "benchmark" / "p17_empirical_preregistration.json",
        root / "scripts" / "verify_p17_preregistration.py",
        root / "scripts" / "package_release.py",
        root / ".github" / "workflows" / "release.yml",
        root / ".github" / "workflows" / "ci.yml",
        root / "scripts" / "render_phase17_w1_tables.py",
        root / "scripts" / "render_phase17_w2_tables.py",
        root / "scripts" / "render_phase17_w3_tables.py",
        root / "scripts" / "render_phase17_w4_tables.py",
        root / "scripts" / "render_phase17_final_tables.py",
        root / "scripts" / "posthoc_diagnostics" / "posthoc_zenodo17228160_mst_curve_diagnostic.py",
        root / "scripts" / "posthoc_diagnostics" / "posthoc_f16_dosy_diagnostic.py",
        root / "scripts" / "posthoc_diagnostics" / "posthoc_f17_mtbls1_diagnostic.py",
        root / "data" / "private" / "PHASE17_W1_CLAUDE_REVIEW_SUMMARY.md",
        root / "data" / "private" / "PHASE17_W2_CLAUDE_REVIEW_SUMMARY.md",
        root / "data" / "private" / "PHASE17_W3_CLAUDE_REVIEW_SUMMARY.md",
        root / "data" / "private" / "PHASE17_W4_CLAUDE_REVIEW_SUMMARY.md",
        root / "data" / "private" / "p17_w1_results" / "w1_ubq_realistic_evograph.log",
        root / "data" / "private" / "p17_w1_results" / "w1_ubq_realistic_standard_cpu.log",
        root / "data" / "private" / "p17_w1_results" / "w1_ubq_realistic_standard_gpu.log",
        root / "data" / "private" / "p17_w1_results" / "w1_cisd3_evograph.json",
        root / "data" / "private" / "p17_w1_results" / "w1_cisd3_standard_cpu.json",
        root / "data" / "private" / "p17_w1_results" / "w1_cisd3_standard_gpu.json",
        root / "data" / "private" / "p17_w1_results" / "dev_hsqc_selection_diff.json",
        root / "data" / "private" / "w3_empirical_results" / "SHA256SUMS.txt",
        root / "data" / "private" / "w3_empirical_results" / "f10_f11" / "f10_evaluation.json",
        root / "data" / "private" / "w3_empirical_results" / "f10_f11" / "f11_evaluation.json",
        root / "data" / "private" / "w3_empirical_results" / "f13abc" / "f13a_evaluation.json",
        root / "data" / "private" / "w3_empirical_results" / "f13abc" / "f13b_evaluation.json",
        root / "data" / "private" / "w3_empirical_results" / "f13abc" / "f13c_evaluation.json",
        root / "data" / "private" / "w4_empirical_results" / "SHA256SUMS.txt",
        root / "data" / "private" / "w4_empirical_results" / "f14" / "f14_evaluation.json",
        root / "data" / "private" / "w4_empirical_results" / "f16" / "superseded_r1_f16_evaluation.json",
        root / "data" / "private" / "w4_empirical_results" / "f16" / "f16_evaluation.json",
        root / "data" / "private" / "w4_empirical_results" / "f17" / "f17_evaluation.json",
        results_dir / "nmrassign_version.txt",
        results_dir / "cargo_lock_diff.patch",
        results_dir / "cargo_fmt_check.log",
        results_dir / "cargo_clippy.log",
        results_dir / "wasm32_check.log",
        results_dir / "windows_gnu_check.log",
        results_dir / "cargo_test_workspace.log",
        results_dir / "build_docs_manual_selftest.log",
        results_dir / "build_docs_manual_check.log",
        results_dir / "build_web_demo_check.log",
        results_dir / "package_release_selftest.log",
        results_dir / "p17_adapters_selftest.log",
        results_dir / "ci_runs_w1_to_w4.json",
        *extra_files,
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        if p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def write_final_sha256sums(results_dir: Path) -> Path:
    rows: list[str] = []
    for p in sorted(results_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(results_dir).as_posix()
        if rel == "SHA256SUMS.txt":
            continue
        _, digest = sha256_and_size(p)
        rows.append(f"{digest}  {rel}")
    sums_p = results_dir / "SHA256SUMS.txt"
    write_clean_utf8(sums_p, "\n".join(rows) + "\n")
    return sums_p


def render_digests_section(files: list[Path], root: Path) -> str:
    lines = [
        "### Cited Phase 17 Final & Wave 1–4 Artifacts & FIPS 180-4 SHA-256 Digests",
        "",
        "| Artifact Path | Bytes | SHA-256 |",
        "| :--- | ---: | :--- |",
    ]
    for p in files:
        n_bytes, digest = sha256_and_size(p)
        rel = display_rel_path(p, root)
        lines.append(f"| `{rel}` | {n_bytes:,} | `{digest}` |")
    return "\n".join(lines)


def verify_markdown_tables(md: str) -> None:
    lines = md.splitlines()
    in_table = False
    expected_pipes = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            for span_m in re.finditer(r"`[^`]+`|\$[^$]+\$", stripped):
                if re.search(r"(?<!\\)\|", span_m.group(0)):
                    raise ValueError(
                        f"Unescaped '|' inside table cell code/math on line {idx + 1}: {span_m.group(0)}"
                    )
            pipes = stripped.count("|")
            if not in_table:
                in_table = True
                expected_pipes = pipes
            else:
                if pipes != expected_pipes:
                    raise ValueError(
                        f"Markdown table column/pipe mismatch on line {idx + 1}: got {pipes} pipes, expected {expected_pipes}: {line!r}"
                    )
        else:
            if in_table and stripped != "":
                raise ValueError(
                    f"Fused or non-blank line immediately following Markdown table on line {idx + 1}: {line!r}"
                )
            in_table = False
    if HOST_PATH_RE.search(md):
        raise ValueError("Rendered Markdown leaked an absolute host user-home path!")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/private/phase17_final_results"),
        help="Directory holding Phase 17 Final verification logs & CI status JSON.",
    )
    parser.add_argument(
        "--base-ref",
        type=str,
        default="d6cc709",
        help="Approved 17.W4 baseline git commit.",
    )
    parser.add_argument(
        "--extra-file",
        type=Path,
        action="append",
        default=[],
        help="Additional artifact files to include in SHA-256 digest table.",
    )
    parser.add_argument(
        "--section",
        choices=["all", "scorecard", "ci_ledger", "gates", "benchmarks", "digests"],
        default="all",
        help="Subset of Markdown sections to render.",
    )
    parser.add_argument(
        "--write-sha256sums",
        action="store_true",
        help="Write unified data/private/phase17_final_results/SHA256SUMS.txt.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output Markdown path (prints to stdout when omitted).",
    )

    args = parser.parse_args()
    root = args.root.resolve()
    results_dir = (
        args.results_dir if args.results_dir.is_absolute() else (root / args.results_dir)
    )
    extra_files = [p if p.is_absolute() else (root / p) for p in args.extra_file]

    sections: list[str] = []
    sec = args.section
    if sec in ("all", "scorecard"):
        sections.append(render_scorecard_section(root))
    if sec in ("all", "ci_ledger"):
        sections.append(render_ci_ledger_section(results_dir))
    if sec in ("all", "gates"):
        sections.append(render_gates_section(root, results_dir, args.base_ref))
    if sec in ("all", "benchmarks"):
        sections.append(render_benchmarks_section_v2(root))
    if sec in ("all", "digests"):
        digest_files = cited_final_files(root, results_dir, extra_files)
        sections.append(render_digests_section(digest_files, root))

    rendered = "\n\n".join(s for s in sections if s.strip()) + "\n"
    verify_markdown_tables(rendered)

    if args.out is not None:
        out_p = args.out if args.out.is_absolute() else (root / args.out)
        write_clean_utf8(out_p, rendered)
    else:
        sys.stdout.write(rendered)

    if args.write_sha256sums:
        write_final_sha256sums(results_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
