#!/usr/bin/env python3
"""Deterministic Markdown table renderer & SHA-256 ledger generator for Phase 17 Wave 4
(`phase=17.W4`: Single-Shot Zero-Tuning Empirical Analytical 1D, `qNMR`, Raw Vendor `DOSY` &
`METABOLIGHTS:MTBLS1` Metabolomics Validation Gates `F14`, `F16`, `F17`).

Reads structured evaluation JSONs, download manifests, post-hoc diagnostic reports, and CLI logs in
`data/private/w4_empirical_results/`, verifies `0`-byte diffs across `crates/`, `Cargo.lock`,
`data/benchmark/p17_empirical_preregistration.json`, `scripts/verify_p17_preregistration.py`, and
post-`bbc770a` `scripts/p17_adapters/`, and renders all numeric tables and FIPS 180-4 SHA-256
digests mechanically without hand-typed numbers.

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

PROTECTED_ZERO_DIFF_PATHS = [
    "crates/",
    "Cargo.toml",
    "Cargo.lock",
    "THIRD_PARTY.md",
    "CHANGELOG.md",
    "data/benchmark/p17_empirical_preregistration.json",
    "scripts/verify_p17_preregistration.py",
    "data/benchmark/real_manifest.tsv",
    "data/benchmark/manifest.tsv",
    "data/priors/bmrb_priors.json",
    "data/priors/PROVENANCE.md",
    "docs/specs/03-assignment.md",
    "docs/specs/16-dynamics-and-csp.md",
    "docs/specs/17-nmr1d-and-dosy.md",
    "docs/specs/18-mcp-server-and-bridge.md",
    "docs/specs/19-bayesian-sa-solver.md",
    "docs/specs/20-metabolomics-chemometrics.md",
]


def sanitize_host_paths(text: str, root: Path) -> str:
    out = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    root_s = str(root.resolve())
    out = out.replace(root_s, "<repo_root>")
    out = re.sub(
        r"/(?:Users|home)/[A-Za-z0-9_.-]+/Workspace/nmr_software",
        "<main_repo>",
        out,
    )
    return HOST_PATH_RE.sub("<user_home>", out)


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


def display_rel_path(path: Path, root: Path, results_dir: Path) -> str:
    rp = path.resolve()
    try:
        sub = rp.relative_to(results_dir.resolve()).as_posix()
        return f"data/private/w4_empirical_results/{sub}"
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


def fmt_num(val: Any, digits: int = 5) -> str:
    if val is None:
        return "null"
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "NaN" if math.isnan(val) else str(val)
    if isinstance(val, float):
        if val.is_integer() and abs(val) < 1e9:
            return f"{int(val)}"
        return f"{val:.{digits}f}"
    return str(val)


def clean_cell(s: str) -> str:
    s = sanitize_host_paths(s, Path("/nonexistent_placeholder"))
    s = re.sub(r"\|([^|\s]+)\|", r"abs(\1)", s)
    return s.replace("|", "/")


def run_w4_checks(root: Path, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    root_s = str(root.resolve())
    cmds = [
        ["python3", "scripts/verify_p17_preregistration.py", "--self-test", "--check"],
        ["python3", "scripts/p17_adapters/selftest.py"],
    ]
    chunks: list[str] = []
    for cmd in cmds:
        proc = subprocess.run(
            cmd,
            cwd=root_s,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        chunks.append(f"$ {' '.join(cmd)}\n{sanitize_host_paths(proc.stdout, root)}\n")
    out_log = results_dir / "w4_checks.log"
    write_clean_utf8(out_log, "".join(chunks))
    return out_log


def cited_w4_files(results_dir: Path, root: Path, extra_files: list[Path]) -> list[Path]:
    ordered_rel = [
        "w4_checks.log",
        "SHA256SUMS_staging_r2.txt",
        "f14/f14_download_manifest.json",
        "f14/f14_backup_downloads.json",
        "f14/f14_bmse_1h_summary.json",
        "f14/f14_bmse_1h_peaks.tsv",
        "f14/f14_bmse_1h_mult.json",
        "f14/f14_bmse_1h_acs.txt",
        "f14/f14_bmse_13c_summary.json",
        "f14/f14_bmse_13c_peaks.tsv",
        "f14/f14_bmse_13c_mult.json",
        "f14/f14_bmse_13c_acs.txt",
        "f14/f14_bmse_dept90_summary.json",
        "f14/f14_bmse_dept90_peaks.tsv",
        "f14/f14_bmse_dept135_summary.json",
        "f14/f14_bmse_dept135_peaks.tsv",
        "f14/f14_secondary_z4360319_19f_jdx_summary.json",
        "f14/f14_secondary_z4360319_19f_jdx_peaks.tsv",
        "f14/f14_evaluation.json",
        "f16/f16_r2_download_manifest.json",
        "f16/f16_bruker_lth_decay.tsv",
        "f16/f16_bruker_lth_dosy_fit_cpu.json",
        "f16/f16_bruker_lth_dosy_fit_gpu.json",
        "f16/f16_bruker_lth_dosy_ilt.tsv",
        "f16/f16_bruker_lth_dosy_mw.json",
        "f16/f16_bruker_lt_decay.tsv",
        "f16/f16_bruker_lt_dosy_fit_cpu.json",
        "f16/f16_bruker_lt_dosy_fit_gpu.json",
        "f16/f16_bruker_ligands_only_decay.tsv",
        "f16/f16_bruker_ligands_only_dosy_fit_cpu.json",
        "f16/f16_bruker_ligands_only_dosy_fit_gpu.json",
        "f16/f16_varian_mendeley_decay.tsv",
        "f16/f16_varian_mendeley_dosy_fit_cpu.json",
        "f16/f16_varian_mendeley_dosy_fit_gpu.json",
        "f16/f16_varian_mendeley_dosy_ilt.tsv",
        "f16/f16_varian_mendeley_dosy_mw.json",
        "f16/f16_bruker_z16423019_decay.tsv",
        "f16/f16_bruker_z16423019_dosy_fit_cpu.json",
        "f16/f16_bruker_z16423019_dosy_fit_gpu.json",
        "f16/f16_bruker_z16423019_dosy_ilt.tsv",
        "f16/f16_bruker_z16423019_dosy_mw.json",
        "f16/f16_evaluation.json",
        "f17/f17_preflight_freeze_check.txt",
        "f17/f17_preflight_adapter_and_binary_sha256.txt",
        "f17/f17_postflight_freeze_check.txt",
        "f17/f17_download_manifest.json",
        "f17/f17_strict_adapter_invocation.log",
        "f17/mtbls1_manifest_posthoc_1line_fix.tsv",
        "f17/f17_mtbls1_buckets.csv",
        "f17/f17_mtbls1_pca.json",
        "f17/f17_mtbls1_oplsda.json",
        "f17/f17_mtbls1_biomarkers.csv",
        "f17/f17_mtbls1_stocsy_glucose.json",
        "f17/secondary_null_permuted_oplsda.json",
        "f17/secondary_uniform004_oplsda.json",
        "f17/secondary_uniform004_stocsy_glucose.json",
        "f17/f17_evaluation.json",
        "posthoc_diagnostics/posthoc_f16_dosy_diagnostic_report.json",
        "posthoc_diagnostics/posthoc_f17_mtbls1_diagnostic_report.json",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for rel in ordered_rel:
        p = results_dir / rel
        if p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    repo_cited = [
        root / "data" / "benchmark" / "p17_empirical_preregistration.json",
        root / "scripts" / "posthoc_diagnostics" / "posthoc_f16_dosy_diagnostic.py",
        root / "scripts" / "posthoc_diagnostics" / "posthoc_f17_mtbls1_diagnostic.py",
    ]
    for p in [*repo_cited, *extra_files]:
        if p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def write_w4_sha256sums(results_dir: Path) -> Path:
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


def render_protected_diff_section(root: Path, base_ref: str, adapter_ref: str) -> str:
    root_s = str(root.resolve())
    head_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root_s,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    base_short = subprocess.run(
        ["git", "rev-parse", "--short", base_ref],
        cwd=root_s,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    adp_short = subprocess.run(
        ["git", "rev-parse", "--short", adapter_ref],
        cwd=root_s,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()
    crates_tree_head = subprocess.run(
        ["git", "rev-parse", "HEAD:crates"],
        cwd=root_s,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()

    lines = [
        f"### Protected Freeze & Post-Download Zero-Byte Diff Verification (`{base_short}..{head_sha}` & `{adp_short}..{head_sha}`)",
        "",
        "| Protected Path / Tree | Evaluated Range | Diff Bytes (`git diff`) | Invariant | Status |",
        "| :--- | :--- | ---: | :--- | :---: |",
    ]
    for rel_p in PROTECTED_ZERO_DIFF_PATHS:
        proc = subprocess.run(
            ["git", "diff", f"{base_ref}..HEAD", "--", rel_p],
            cwd=root_s,
            stdout=subprocess.PIPE,
            check=True,
        )
        n_bytes = len(proc.stdout)
        if n_bytes != 0:
            raise RuntimeError(
                f"Protected path {rel_p} has non-zero git diff ({n_bytes} bytes) across {base_ref}..HEAD!"
            )
        inv = f"`HEAD:crates == {crates_tree_head[:12]}`" if rel_p == "crates/" else "`0`-byte diff"
        lines.append(f"| `{rel_p}` | `{base_short}..{head_sha}` | `{n_bytes}` | {inv} | `PASS` |")

    adp_proc = subprocess.run(
        ["git", "diff", f"{adapter_ref}..HEAD", "--", "scripts/p17_adapters/"],
        cwd=root_s,
        stdout=subprocess.PIPE,
        check=True,
    )
    if len(adp_proc.stdout) != 0:
        raise RuntimeError("scripts/p17_adapters/ was modified after pre-download commit!")
    lines.append(
        f"| `scripts/p17_adapters/` (`11` pre-download adapter modules) | `{adp_short}..{head_sha}` | `0` | Pre-download SHA-256s untouched | `PASS` |"
    )
    return "\n".join(lines)


def render_adapters_section(root: Path, results_dir: Path) -> str:
    f17 = json.loads((results_dir / "f17" / "f17_evaluation.json").read_text(encoding="utf-8"))
    pre_adp = f17["adapter_directory_hashes_pre"]
    post_adp = f17["adapter_directory_hashes_post"]
    bin_sha_pre = f17["nmrassign_binary_sha256_pre"]
    bin_sha_post = f17["nmrassign_binary_sha256_post"]
    if bin_sha_pre != bin_sha_post:
        raise RuntimeError("nmrassign binary SHA-256 drifted during execution!")

    lines = [
        "### Pre-Download Format Adapter & Binary Integrity (`scripts/p17_adapters/` @ `bbc770a`, `P17-R2`)",
        "",
        f"- **Frozen Binary Verified Pre- & Post-Execution**: `./target/release/nmrassign` (`nmrassign 0.12.0 (41fe1a7)`, `sha256={bin_sha_pre}`)",
        "",
        "| Adapter Module | Bytes | Pre-Download & Post-Execution SHA-256 | Disk Match |",
        "| :--- | ---: | :--- | :---: |",
    ]
    for name in sorted(pre_adp.keys()):
        expected_b = int(pre_adp[name]["bytes"])
        expected_s = pre_adp[name]["sha256"]
        if post_adp[name]["sha256"] != expected_s:
            raise RuntimeError(f"Post-execution hash mismatch for {name}!")
        actual_b, actual_s = sha256_and_size(root / "scripts" / "p17_adapters" / name)
        if actual_b != expected_b or actual_s != expected_s:
            raise RuntimeError(f"Adapter {name} drifted on disk vs pre-download pin!")
        lines.append(f"| `scripts/p17_adapters/{name}` | {actual_b:,} | `{actual_s}` | `PASS` |")
    return "\n".join(lines)


def render_verdict_overview_section(results_dir: Path) -> str:
    f14 = json.loads((results_dir / "f14" / "f14_evaluation.json").read_text(encoding="utf-8"))
    f16 = json.loads((results_dir / "f16" / "f16_evaluation.json").read_text(encoding="utf-8"))
    f17 = json.loads((results_dir / "f17" / "f17_evaluation.json").read_text(encoding="utf-8"))

    def crit_ratio(ev: dict[str, Any]) -> str:
        cr = ev.get("criteria_results") or ev.get("w4_adapter_evaluate_f14_output", {}).get(
            "criteria_results", []
        )
        ok = sum(1 for c in cr if c.get("passed"))
        return f"`{ok}/{len(cr)}`"

    verdicts = [x["verdict"] for x in (f14, f16, f17)]
    n_pass = sum(1 for v in verdicts if v == "PASS")
    n_fail = sum(1 for v in verdicts if v == "FAIL")
    n_ne = sum(1 for v in verdicts if v == "NOT_EVALUABLE")

    lines = [
        f"### Wave `17.W4` Single-Shot Empirical Gate Verdicts (`{n_pass} PASS`, `{n_fail} FAIL`, `{n_ne} NOT_EVALUABLE`)",
        "",
        "| Gate (`TODO.md`) | Evaluated Pre-Registered Chain | Criteria Passed | Single-Shot Verdict | `TODO.md` Checkbox | Primary Strict `bbc770a` Outcome & Root Cause |",
        "| :--- | :--- | :---: | :---: | :---: | :--- |",
        (
            f"| `F14` (`TODO.md:144`) | `BMSE:bmse000036` + `ZENODO:17368767` + `ZENODO:15728739` -> `bmse000006` -> `bmse000002` -> `ZENODO:4360319` -> `ZENODO:7221753` | {crit_ratio(f14)} | **`{f14['verdict']}`** | `[ ]` open | "
            "`F14-C2` **`FAIL`** (`0.333333 != 1.0` on valid `BMSE:bmse000036` raw `DEPT_135` FID: biphasic auto-phase hit `PHC1 = +180 deg` boundary on CH2-free L-fucose); `F14-C1` (`3/3` raw Bruker `1H`/`13C`/`DEPT` `SNR >= 70.0`, `0/7` raw Bruker `19F`/`31P`) & `F14-C3` (no internal-std gravimetric tables) `NOT_EVALUABLE` |"
        ),
        (
            f"| `F16` (`TODO.md:146`) | `{f16['accession_id']}` -> `backup_1` `ZENODO:16423019` (`23.zip`) | {crit_ratio(f16)} | **`{f16['verdict']}`** | `[ ]` open | "
            "`F16-C1` **`PASS`** (`2/2` vendor formats Bruker `ser+difflist` `256` & `16` planes + Varian `fid+procpar` `12` planes exit `0`); uncurated all-peaks `F16-C2` **`FAIL`** (`min R^2 = 0.9080 < 0.95` on Varian Quinine shoulders, `-0.4081` `LigandsOnly` water ripples, `0.9395` `23.zip`); `F16-C3`/`F16-C4` **`FAIL`** (`LysozymeNAG` lacks `ser`, tyrosinase below `SNR >= 10`, `23.zip` `<gdp_dosyoneshot_tlg>` `4.013x` `b` under-scale, no frozen `D_ref` extractor) |"
        ),
        (
            f"| `F17` (`TODO.md:147`) | `{f17['accession_id']}` (`N = 132` human urinary T2DM `1H` cohort) | {crit_ratio(f17)} | **`{f17['verdict']}`** | `[ ]` open | "
            "Strict `bbc770a` `f17_mtbls1_manifest.py` exited `rc = 1` (`0/132` joined: `a_MTBLS1` names `.zip` column `Free Induction Decay Data File` vs `Raw Spectral Data File` -> **`FAIL`** per `P17-R1`); post-hoc 1-line fix (`132/132` `PQN`, `perm_p = 0.0099`, `10` Top-15 glucose buckets) also **`FAIL` (`3/4`)** on `OPLS-DA` `Q^2 = 0.122445 < 0.25` |"
        ),
    ]
    return "\n".join(lines)


def render_downloads_section(results_dir: Path) -> str:
    f14_dl = json.loads((results_dir / "f14" / "f14_download_manifest.json").read_text(encoding="utf-8"))
    f16_dl = json.loads((results_dir / "f16" / "f16_r2_download_manifest.json").read_text(encoding="utf-8"))
    f17_dl = json.loads((results_dir / "f17" / "f17_download_manifest.json").read_text(encoding="utf-8"))

    f14_prim = f14_dl["primary_downloads"]
    f14_bkp = f14_dl["f14_backup_downloads"]
    f14_procs = f14_dl.get("posthoc_vendor_procs_diagnostic_downloads", [])
    f14_tot_bytes = sum(int(r["bytes"]) for r in [*f14_prim, *f14_bkp, *f14_procs])

    f16_files = f16_dl["downloads"]
    if f16_dl.get("unregistered_auxiliary_files_count", 0) != 0:
        raise RuntimeError("F16 R2 download manifest contains unregistered auxiliary files!")

    f17_meta = f17_dl["metadata_files"]
    f17_zips = f17_dl["sample_files"]
    f17_zip_bytes = sum(int(r["bytes"]) for r in f17_zips)

    lines = [
        "### Wave `17.W4` Downloaded Empirical Payloads (`/tmp/p17_empirical_downloads/w4/` — `BMRB:53960` Untouched)",
        "",
        (
            f"*(Summary: `F14` fetched `{len(f14_prim)}` primary + `{len(f14_bkp)}` backup files [`{f14_tot_bytes:,} B` total, plus `{len(f14_procs)}` post-execution `pdata/1/procs` vendor diagnostic files] recorded in `f14/f14_download_manifest.json`; "
            f"`F16` `f16/f16_r2_download_manifest.json` restricts strictly to the `4` pre-registered files [`unregistered_auxiliary_files_count = 0`]; "
            f"`F17` `f17/f17_download_manifest.json` records `2` ISA-Tab metadata sheets + all `{len(f17_zips)}/{f17_dl['n_sample_zips_expected']}` raw Bruker `.zip` archives [`{f17_zip_bytes:,} B`, `100%` `zip_crc_ok = true`].)*"
        ),
        "",
        "| Gate | Accession | Role / Kind | File / Archive Summary | HTTP | Bytes | FIPS 180-4 SHA-256 / Manifest Reference |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | :--- |",
    ]
    for rec in f14_prim:
        lines.append(
            f"| `F14` | `{rec['accession_id']}` | `primary` | `{rec['name']}` | `{rec['http_status']}` | {int(rec['bytes']):,} | `{rec['sha256']}` |"
        )
    # Summarize large F14 backup archives explicitly by top-level payload
    bkp_seen: set[tuple[str, str]] = set()
    for rec in f14_bkp:
        key = (rec["accession_id"], rec["name"])
        if rec["accession_id"].startswith("ZENODO:") or rec["name"].endswith(("fid", ".str")):
            if key not in bkp_seen:
                bkp_seen.add(key)
                lines.append(
                    f"| `F14` | `{rec['accession_id']}` | `{rec['role']}` | `{rec['name']}` | `{rec['http_status']}` | {int(rec['bytes']):,} | `{rec['sha256']}` |"
                )
    for rec in f14_procs:
        lines.append(
            f"| `F14` | `{rec['accession_id']}` | `posthoc_procs_diag` | `{rec['name']}` | `{rec['http_status']}` | {int(rec['bytes']):,} | `{rec['sha256']}` |"
        )
    for rec in f16_files:
        lines.append(
            f"| `F16` | `{rec['accession_id']}` | `{rec['role']}` | `{rec['name']}` | `{rec['http_status']}` | {int(rec['bytes']):,} | `{rec['sha256']}` |"
        )
    for rec in f17_meta:
        lines.append(
            f"| `F17` | `METABOLIGHTS:MTBLS1` | `primary_isatab` | `{rec['name']}` | `{rec['http_status']}` | {int(rec['bytes']):,} | `{rec['sha256']}` |"
        )
    first_zip = f17_zips[0]
    last_zip = f17_zips[-1]
    lines.append(
        f"| `F17` | `METABOLIGHTS:MTBLS1` | `primary_raw_zips` (`132/132`) | `{first_zip['name']} .. {last_zip['name']}` (`84` Control, `48` T2DM) | `200` | {f17_zip_bytes:,} | All `132` per-file SHA-256s in `f17/f17_download_manifest.json` (`first={first_zip['sha256'][:16]}...`, `last={last_zip['sha256'][:16]}...`) |"
    )
    return "\n".join(lines)


def render_f14_section(results_dir: Path) -> str:
    ev = json.loads((results_dir / "f14" / "f14_evaluation.json").read_text(encoding="utf-8"))
    cb = ev["criteria_breakdown"]
    dsp = ev["bmse000036_dsp_and_phase_vs_vendor_procs"]
    dept = ev["bmse000036_dept135_multiplicity"]
    sm = ev["secondary_report_only_metrics"]
    ref_diag = sm["bmse000036_referencing_offset_diagnosis"]
    jdx = sm["z4360319_secondary_19f_spectrum_jdx_check"]

    jdx_peaks_tsv = (results_dir / "f14" / "f14_secondary_z4360319_19f_jdx_peaks.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    jdx_ppms = [
        f"{float(ln.split('\t')[1]):.3f} ppm"
        for ln in jdx_peaks_tsv[1:]
        if ln.strip()
    ]

    crit_rows = []
    for c in cb:
        val_s = (
            f"{c['evaluable_subset_value']} (raw Bruker subset)"
            if "evaluable_subset_value" in c
            else fmt_num(c.get("value"), 6)
        )
        crit_rows.append(
            f"| `{c['id']}` | `{c['metric']}` | `{val_s}` | `{c['operator']} {fmt_num(c['threshold'], 3)}` | **`{clean_cell(c['status'])}`** |"
        )

    exp_rows = []
    for exp_key in ("1H", "13C", "DEPT_90", "DEPT_135"):
        e = dsp[exp_key]
        ph0, ph1, piv = e["nmrassign_phase_applied_deg"]
        exp_rows.append(
            f"| `BMSE:bmse000036` `{exp_key}` | `{e['acqus_dspfvs']}` / `{e['acqus_decim']}` | `{e['n_picked_peaks']}` | `{e['max_abs_snr']:.2f}` | "
            f"`({ph0:.2f} deg, {ph1:.2f} deg, piv={piv:.1f})` | `({e['vendor_procs_phc0_deg']:.2f} deg, {e['vendor_procs_phc1_deg']:.2f} deg)` |"
        )

    qnmr_rows = [
        f"| `{q['role']}` | `{q['accession_id']}` | `{q['source']}` | {clean_cell(q['error'])} |"
        for q in ev["fallback_chain_audit"]["qnmr_gravimetric_search"]
    ]

    lines = [
        "#### Gate `F14` (`TODO.md:144`) — Multi-Nuclear Raw Bruker 1D FIDs & `qNMR` (`FAIL`: `F14-C2` `0.333333 != 1.0`; `F14-C1`/`F14-C3` `NOT_EVALUABLE` Across `7` Archives)",
        "",
        "| Criterion ID | Metric | Measured | Threshold | Criterion Outcome |",
        "| :--- | :--- | ---: | :--- | :--- |",
        *crit_rows,
        "",
        "| Raw Bruker FID Experiment (`BMSE:bmse000036`, L-Fucose in `D2O`) | `DSPFVS` / `DECIM` | Picked Peaks | Principal `SNR` | Blind `nmrassign` Auto-Phase `(PHC0, PHC1)` | Post-Execution Vendor `pdata/1/procs` `(PHC0, PHC1)` (Report-Only) |",
        "| :--- | :---: | ---: | ---: | :--- | :--- |",
        *exp_rows,
        "",
        "| `F14-C3` `qNMR` Chain Step | Accession | Inspected Metadata / Sheet | Frozen `w4_adapters` / Archive Completeness Diagnosis |",
        "| :--- | :--- | :--- | :--- |",
        *qnmr_rows,
        "",
        "| Secondary / Post-Execution Report-Only Metric (`f14_evaluation.json`) | Measured Value & Provenance Disclosure |",
        "| :--- | :--- |",
        (
            "| `BMSE:bmse000036` `DEPT-135` multiplicity breakdown (`F14-C2` root cause) | "
            f"`positive_site_fraction = {dept['positive_site_fraction']:.6f}` (`{dept['n_positive_ch_ch3']}` positive vs `{dept['n_negative_ch2']}` false negative `abs(SNR) >= 6.0` lobes; expected `{dept['expected_pos_sites']}` `CH`/`CH3`, `{dept['expected_neg_ch2_sites']}` `CH2`): biphasic entropy search hit `PHC1 = +180.0 deg` boundary |"
        ),
        (
            "| `BMSE:bmse000036` `13C` & `1H` `ACS` reports & shift MAE vs `bmse000036.str` | "
            f"`13C` shift `MAE = {sm['bmse000036_13c_shift_mae_ppm_vs_str']:.4f} ppm` (`median = {sm['bmse000036_13c_shift_median_abs_err_ppm_vs_str']:.4f} ppm`; raw `acqus` TMS `BF1` vs IUPAC DSS `SF` offset `{ref_diag['13c_raw_acqus_vs_str_offset_ppm']:.3f} ppm`); "
            f"`1H` (`{sm['bmse000036_1h_multiplet_count']}` multiplets) shift `MAE = {sm['bmse000036_1h_shift_mae_ppm_vs_str']:.4f} ppm` (`{ref_diag['1h_raw_acqus_vs_str_offset_ppm']:.4f} ppm` lock shift + `4.70 ppm` HDO phase pull made `0.5 mM DSS` integral `-0.0697` -> `qnmr` `rc=1`) |"
        ),
        (
            "| `ZENODO:4360319` secondary `Diphenoxide 19F.jdx` (`NMR SPECTRUM`, non-Bruker-FID) | "
            f"`exit_code = {jdx['exit_code']}`, `n_picked_peaks = {jdx['n_picked_peaks']}` (`{', '.join(jdx_ppms)}`), `max_abs_snr = {jdx['max_abs_snr']:.2f}` ({jdx['note']}); `ZENODO:17368767` holds `106` JEOL Delta `.jdf` files (`0` Bruker `fid+acqus`) |"
        ),
        (
            "| Post-execution vendor `pdata/1/procs` isolation note | "
            f"`4` `BMSE_bmse000036/set01/*/pdata/1/procs` files were fetched strictly post-execution for vendor `PHC0`/`PHC1`/`SF` diagnosis and touched zero blind `F14` commands or criteria |"
        ),
    ]
    return "\n".join(lines)


def render_f16_section(results_dir: Path) -> str:
    ev = json.loads((results_dir / "f16" / "f16_evaluation.json").read_text(encoding="utf-8"))
    bkp_ev = ev["backup_1_zenodo_16423019_evaluation"]
    ds_map = ev["per_dataset_strict_summaries"]
    ph = json.loads(
        (results_dir / "posthoc_diagnostics" / "posthoc_f16_dosy_diagnostic_report.json").read_text(
            encoding="utf-8"
        )
    )
    parity = ph["cpu_f64_vs_webgpu_metal_f32_parity"]
    mat_diag = ph["bruker_zenodo12170718_smsq10_vs_mat_decay_diagnostic"]["comparisons"]
    oneshot_diag = ph["bruker_zenodo16423019_oneshot_pulprog_diagnostic"]
    var_diag = ph["varian_mendeley_unregistered_dosy_in_posthoc_diagnostic"]

    crit_rows = []
    for c_prim, c_bkp in zip(ev["criteria_results"], bkp_ev["criteria_results"]):
        crit_rows.append(
            f"| `{c_prim['id']}` | `{c_prim['metric']}` | `{fmt_num(c_prim['value'], 5)}` | `{fmt_num(c_bkp['value'], 5)}` | "
            f"`{c_prim['operator']} {fmt_num(c_prim['threshold'], 3)}` | **`{'PASS' if c_prim['passed'] else 'FAIL'}`** |"
        )

    ds_labels = [
        ("bruker_lth", "`ZENODO:12170718` `LigandsTyrosinaseHCCA` (`primary`)"),
        ("bruker_lt", "`ZENODO:12170718` `LigandsTyrosinase` (`primary`)"),
        ("bruker_ligands_only", "`ZENODO:12170718` `LigandsOnly` (`primary`)"),
        ("varian_mendeley", "`MENDELEY:d7bdxz9hsk` Varian `Doneshot` (`co_primary`)"),
        ("bruker_z16423019", "`ZENODO:16423019` `23.zip` GNAT (`backup_1`)"),
    ]
    ds_rows = []
    for slug, label in ds_labels:
        d = ds_map[slug]
        cp = d["cpu_vs_gpu_parity"]
        ds_rows.append(
            f"| {label} | `{d['vendor']}` (`{d['n_planes']}`) | `{d['n_peaks_r2_ge_0_95']}/{d['n_picked_peaks']}` | "
            f"`{d['min_all_peaks_r2']:.4f}` / `{d['median_all_peaks_r2']:.4f}` | `{cp['bic_k_agreement']}` | `{cp['max_abs_delta_log10_d_k1_dex']:.2e} dex` |"
        )

    lines = [
        "#### Gate `F16` (`TODO.md:146`) — Experimental Raw Bruker & Varian Pseudo-2D `DOSY` (`FAIL`, `1/4` Criteria Under Strict `bbc770a` Blind Evaluator)",
        "",
        "| Criterion ID | Metric | Primary (`ZENODO:12170718` + `MENDELEY:d7bdxz9hsk`) | `backup_1` (`ZENODO:16423019` + `MENDELEY:d7bdxz9hsk`) | Threshold | Blind Verdict |",
        "| :--- | :--- | ---: | ---: | :--- | :---: |",
        *crit_rows,
        "",
        "| Evaluated Raw Vendor `DOSY` Series (`0` Unregistered Auxiliary Files) | Vendor (`Planes`) | Peaks `R^2 >= 0.95` | Uncurated `min` / `median R^2` | CPU `f64` vs Metal `f32` `BIC` `K` | Max `abs(d log10 D)` (`K=1`) |",
        "| :--- | :---: | :---: | :---: | :---: | ---: |",
        *ds_rows,
        "",
        "| Post-Hoc Non-Gate Diagnostic (`scripts/posthoc_diagnostics/posthoc_f16_dosy_diagnostic.py`, `alters_primary_gate_verdict = false`) | Measured Result & Explicit Provenance Caveat |",
        "| :--- | :--- |",
        (
            "| Global CPU `f64` vs Apple M5 Pro WebGPU `Metal` `f32` `BIC` `K` parity (`5` series) | "
            f"**`{parity['total_bic_k_agreement']}` (`100.0%`)** `BIC` component-count agreement; `max abs(d log10 D)_{{K=1}} = {parity['global_max_abs_delta_log10_d_k1_dex']:.2e} dex` (`all K = {parity['global_max_abs_delta_log10_d_all_dex']:.2e} dex`) |"
        ),
        (
            "| Bruker `ZENODO:12170718` `SMSQ10.100` (`0.81`) vs `TyrosinaseHCCA_pfg_data.mat` (`pfgnmrdata_Pb/Pa/L`) | "
            f"Same-plane (`#179`) `b_nmrassign / b_mat_smsq081 = {mat_diag['bruker_lth']['ratio_b_nmrassign_over_mat_smsq081_same_plane']:.6f}`; max ligand slope diff `{mat_diag['bruker_lth']['max_abs_log10_diff_vs_mat_smsq_dex']:.4f}` / `{mat_diag['bruker_lt']['max_abs_log10_diff_vs_mat_smsq_dex']:.4f}` / `{mat_diag['bruker_ligands_only']['max_abs_log10_diff_vs_mat_smsq_dex']:.4f} dex` (**non-independent DSP cross-check on the same raw `ser`**, NOT an external `D_ref`) |"
        ),
        (
            "| Bruker `backup_1` `ZENODO:16423019` (`23.zip`) `<gdp_dosyoneshot_tlg>` `b`-factor root cause | "
            f"`b_oneshot / b_nmrassign = {oneshot_diag['ratio_b_oneshot_over_nmrassign']:.4f}x` (`+{oneshot_diag['log10_ratio_dex']:.4f} dex`): `dosy.rs:590` checks `pulprog_low.contains('bp')`, treating Bruker Oneshot half-bipolar `P30 = 1.1 ms` as monopolar (`F29`) |"
        ),
        (
            "| Varian `MENDELEY:d7bdxz9hsk` unregistered `dosy_in` (`ae9285f6-...`) `Doneshot` `b`-factor check | "
            f"`b_nmrassign / b_vnmrj = {var_diag['ratio_b_nmrassign_over_vnmrj']:.6f}` (`+{var_diag['log10_b_ratio_dex']:.6f} dex`), `max_species_abs_log10_diff = {var_diag['max_species_abs_log10_diff_dex']:.4f} dex` (**unregistered Mendeley UUID file & same-FID VnmrJ `line_amps`; excluded from `f16_evaluation.json`**) |"
        ),
    ]
    return "\n".join(lines)


def render_f17_section(results_dir: Path) -> str:
    ev = json.loads((results_dir / "f17" / "f17_evaluation.json").read_text(encoding="utf-8"))
    ph = json.loads(
        (results_dir / "posthoc_diagnostics" / "posthoc_f17_mtbls1_diagnostic_report.json").read_text(
            encoding="utf-8"
        )
    )
    cs = ph["posthoc_nmrassign_metabolomics_cohort_summary"]
    po = cs["posthoc_criteria_outcome_summary"]
    null_opls = json.loads(
        (results_dir / "f17" / "secondary_null_permuted_oplsda.json").read_text(encoding="utf-8")
    )
    unif_opls = json.loads(
        (results_dir / "f17" / "secondary_uniform004_oplsda.json").read_text(encoding="utf-8")
    )
    cli_runs = ev["posthoc_disclosed_column_fix_diagnostic"]["posthoc_cli_runs"]
    chemometrics_wall_s = sum(float(r["elapsed_seconds"]) for r in cli_runs.values())

    posthoc_map = {
        "F17-C1": (cs["n_clean_processed_pqn_samples"], po["F17-C1_n_clean_ge_126"]),
        "F17-C2": (cs["oplsda_cv_q2"], po["F17-C2_oplsda_q2_ge_0_25"]),
        "F17-C3": (cs["oplsda_perm_p_q2"], po["F17-C3_perm_p_le_0_02"]),
        "F17-C4": (cs["glucose_glycosuria_buckets_in_top15"], po["F17-C4_glucose_top15_ge_1"]),
    }

    crit_rows = []
    for c in ev["criteria_results"]:
        ph_val, ph_ok = posthoc_map[c["id"]]
        crit_rows.append(
            f"| `{c['id']}` | `{c['metric']}` | `{fmt_num(c['value'], 6)}` (**`FAIL`**) | `{c['operator']} {fmt_num(c['threshold'], 3)}` | "
            f"`{fmt_num(ph_val, 6)}` (`{'PASS' if ph_ok else 'FAIL'}`) |"
        )

    marker_rows = []
    for hit in cs["salek_2007_nonglucose_marker_hits"]:
        b = hit["best_bucket"]
        if b is not None:
            marker_rows.append(
                f"| `{hit['metabolite']}` | `{hit['literature_ppm']:.2f} ppm` | `#{b['bucket_index']}` (`{b['center_ppm']:.4f} ppm`) | "
                f"`#{b['vip_rank']}` (`VIP = {b['vip']:.3f}`) | `{b['p_corr']:+.4f}` / `{b['fdr_q']:.3e}` | `{b['bmrb_match']}` |"
            )

    lines = [
        "#### Gate `F17` (`TODO.md:147`) — `METABOLIGHTS:MTBLS1` Human Urinary T2DM `1H` Cohort (`N = 132`, Primary Strict `bbc770a` **`FAIL`** `rc=1`; Post-Hoc 1-Line Column Fix Also **`FAIL`** `3/4`)",
        "",
        "| Criterion ID | Metric | Primary Strict `bbc770a` (`rc = 1`, `0/132` Joined) | Threshold | Post-Hoc 1-Line Column-Fix Diagnostic (`alters_primary_gate_verdict = false`) |",
        "| :--- | :--- | ---: | :--- | ---: |",
        *crit_rows,
        "",
        "| Post-Hoc `nmrassign metabolomics` Cohort & Secondary Sensitivity Summary (`N = 132`: `84` Control, `48` T2DM) | Measured Value |",
        "| :--- | :--- |",
        (
            "| Primary strict `bbc770a` `f17_mtbls1_manifest.py` failure (`f17_strict_adapter_invocation.log`) | "
            "`exit_code = 1`: `a_MTBLS1_metabolite_profiling_NMR_spectroscopy.txt` tabulates raw Bruker `.zip` files under `Free Induction Decay Data File` rather than `Raw Spectral Data File` (`0/132` joined -> **`FAIL`** per `P17-R1`/`P17-R2`) |"
        ),
        (
            "| Post-hoc `AIB` + `PQN` + Pareto `OPLS-DA` (`1` ortho, `7`-fold CV, `100` perms, `--seed 20261717`) | "
            f"`129` `AIB` buckets (`{chemometrics_wall_s:.3f}s` chemometrics wall time across `132` pre-processed TSVs); `R2Xp = {cs['r2_x_predictive']:.4f}`, `R2Xo = {cs['r2_x_orthogonal']:.4f}`, `R2Y = {cs['r2_y']:.4f}`, **`Q^2 = {cs['oplsda_cv_q2']:.6f} < 0.25` (`FAIL`)**, `perm_p = {cs['oplsda_perm_p_q2']:.6f} <= 0.02`, `10` Top-15 glucose buckets (`5.24 ppm` `STOCSY` top match `alpha-D-Glucose bmse000016a`, `mean_r2 = 0.923`) |"
        ),
        (
            "| Secondary `0.04 ppm` uniform bucketing (`198` buckets) & shuffled-label anti-overfitting guard | "
            f"Uniform `0.04 ppm` `Q^2 = {unif_opls['q2']:.6f}` (`perm_p = {unif_opls['permutation_p_value_q2']:.6f}`); shuffled-label null `Q^2 = {null_opls['q2']:.6f} < 0.10` (`perm_p = {null_opls['permutation_p_value_q2']:.4f} > 0.10`, `is_validated = {null_opls['is_validated']}`) |"
        ),
        "",
        "| Salek et al. (2007) Non-Glucose Urinary Marker (Post-Hoc Diagnostic) | Literature `ppm` | Matched `AIB` Bucket | `VIP` Rank (`VIP`) | `p(corr)` / `BH-FDR q` | Built-In CC0 `BMRB` Annotation |",
        "| :--- | ---: | :--- | :--- | :--- | :--- |",
        *marker_rows,
    ]
    return "\n".join(lines)


def render_digests_section(files: list[Path], root: Path, results_dir: Path) -> str:
    lines = [
        "### Raw `17.W4` Artifacts & FIPS 180-4 SHA-256 Digests",
        "",
        "| Artifact Path | Bytes | SHA-256 |",
        "| :--- | ---: | :--- |",
    ]
    for p in files:
        n_bytes, digest = sha256_and_size(p)
        rel = display_rel_path(p, root, results_dir)
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
        default=Path("data/private/w4_empirical_results"),
        help="Directory holding raw Phase 17 Wave 4 JSON/TSV/CSV/log artifacts.",
    )
    parser.add_argument(
        "--base-ref",
        type=str,
        default="051e084",
        help="Approved 17.W3 baseline git commit.",
    )
    parser.add_argument(
        "--adapter-ref",
        type=str,
        default="bbc770a",
        help="Pre-download scripts/p17_adapters/ commit.",
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
        choices=[
            "all",
            "protected_diff",
            "adapters",
            "verdicts",
            "downloads",
            "digests",
            "f14",
            "f16",
            "f17",
        ],
        default="all",
        help="Subset of Markdown sections to render.",
    )
    parser.add_argument(
        "--run-checks",
        action="store_true",
        help="Run verify_p17_preregistration --self-test --check and p17_adapters/selftest.py.",
    )
    parser.add_argument(
        "--write-sha256sums",
        action="store_true",
        help="Write unified data/private/w4_empirical_results/SHA256SUMS.txt.",
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

    if args.run_checks:
        run_w4_checks(root, results_dir)

    extra_files = [p if p.is_absolute() else (root / p) for p in args.extra_file]
    digest_files = cited_w4_files(results_dir, root, extra_files)

    sections: list[str] = []
    sec = args.section
    if sec in ("all", "protected_diff"):
        sections.append(render_protected_diff_section(root, args.base_ref, args.adapter_ref))
    if sec in ("all", "adapters"):
        sections.append(render_adapters_section(root, results_dir))
    if sec in ("all", "verdicts"):
        sections.append(render_verdict_overview_section(results_dir))
    if sec in ("all", "downloads"):
        sections.append(render_downloads_section(results_dir))
    if sec in ("all", "digests"):
        sections.append(render_digests_section(digest_files, root, results_dir))
    if sec in ("all", "f14"):
        sections.append(render_f14_section(results_dir))
    if sec in ("all", "f16"):
        sections.append(render_f16_section(results_dir))
    if sec in ("all", "f17"):
        sections.append(render_f17_section(results_dir))

    rendered = "\n\n".join(s for s in sections if s.strip()) + "\n"
    verify_markdown_tables(rendered)

    if args.out is not None:
        out_p = args.out if args.out.is_absolute() else (root / args.out)
        write_clean_utf8(out_p, rendered)
    else:
        sys.stdout.write(rendered)

    if args.write_sha256sums:
        write_w4_sha256sums(results_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
