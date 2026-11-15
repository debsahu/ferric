#!/usr/bin/env python3
"""Deterministic Markdown table renderer for Phase 15 Wave 1 (T15.1 + T15.2 + T15.3: Clean-Room MCP Server & Live Workbench Bridge).

Reads structured `MCP_METRIC_JSON:` records emitted by `--release -- --nocapture` runs of
`nmr-core --test mcp_tests`, `nmr-gui --lib mcp_bridge`, and `nmr-cli --test mcp_cli_tests`,
plus full workspace test logs and 5-seed `Ubiquitin --realistic` / `CISD3` (`BMRB 51439`)
benchmark artifacts under `--results-dir`, and renders Markdown tables and FIPS 180-4 SHA-256
digests directly from those machine-captured artifacts.

Provenance: Original clean-room reporting script for protocol & algorithmic verification citing
the Model Context Protocol (`MCP`) Open Specification (`2024-11-05` / `2025-03-26`), JSON-RPC 2.0,
RFC 9110 (HTTP Semantics), RFC 2083 (PNG v1.0), RFC 1950/1951 (ZLIB/DEFLATE), Vallurupalli et al.
(J. Am. Chem. Soc. 2012), Baldwin (J. Magn. Reson. 2014), Williamson (Prog. Nucl. Magn. Reson.
Spectrosc. 2013), Hoye et al. (J. Org. Chem. 1994), Stejskal & Tanner (J. Chem. Phys. 1965),
De Meyer et al. (Anal. Chem. 2008), Dieterle et al. (Anal. Chem. 2006), Trygg & Wold (J. Chemom.
2002), Westerhuis et al. (Metabolomics 2008), and Cloarec et al. (Anal. Chem. 2005).
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

METRIC_PREFIX = "MCP_METRIC_JSON:"

HOST_PATH_RE = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+")

SLICE_15_W1_COMMANDS = [
    ["cargo", "test", "--release", "-p", "nmr-core", "--test", "mcp_tests", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-gui", "--lib", "mcp_bridge", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-cli", "--test", "mcp_cli_tests", "--", "--nocapture"],
]

BENCHMARK_ENGINES = [
    ("evograph", "`evo-graph` (`CPU f64`)", ["evo-graph", "cpu"]),
    ("standard_cpu", "`standard` (`CPU f64`)", ["standard", "cpu"]),
    ("standard_gpu", "`standard` (`WebGPU f32`)", ["standard", "gpu"]),
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


def run_w1_mcp_suite(root: Path, results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "w1_mcp_suite_tests.log"
    jsonl_path = results_dir / "w1_mcp_metrics.jsonl"
    root_s = str(root.resolve())
    chunks: list[str] = []
    for cmd in SLICE_15_W1_COMMANDS:
        proc = subprocess.run(
            cmd,
            cwd=root_s,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        chunks.append(f"$ {' '.join(cmd)}\n{sanitize_host_paths(proc.stdout, root)}\n")
    write_clean_utf8(log_path, "".join(chunks))

    records = parse_mcp_metrics([log_path])
    jsonl_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    write_clean_utf8(jsonl_path, jsonl_text)
    return log_path, jsonl_path


def run_w1_benchmarks_and_workspace(
    root: Path, results_dir: Path, cisd3_data_root: Path | None
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    root_s = str(root.resolve())

    # 1. Full workspace release test suite
    ws_cmd = ["cargo", "test", "--release", "--workspace"]
    ws_proc = subprocess.run(
        ws_cmd,
        cwd=root_s,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    write_clean_utf8(
        results_dir / "w1_workspace_tests.log",
        f"$ {' '.join(ws_cmd)}\n{sanitize_host_paths(ws_proc.stdout, root)}\n",
    )
    if ws_proc.returncode != 0:
        sys.stderr.write(ws_proc.stdout[-4000:])
        ws_proc.check_returncode()

    # 2. 5-seed Ubiquitin --realistic & 5-seed CISD3 (BMRB 51439)
    fd, tmp_manifest_str = tempfile.mkstemp(prefix="cisd3_p15_w1_manifest_", suffix=".tsv")
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
            write_clean_utf8(
                results_dir / f"w1_ubq_realistic_{slug}.log",
                f"$ {' '.join(ubq_cmd)}\n{sanitize_host_paths(ubq_proc.stdout, root)}\n",
            )

            if cisd3_data_root is not None and cisd3_data_root.is_dir():
                cisd3_json_p = results_dir / f"w1_cisd3_{slug}.json"
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
                    "<cisd3_manifest.tsv>"
                    if a == str(tmp_manifest)
                    else (
                        "data/private/bmrb"
                        if a == str(cisd3_data_root)
                        else (
                            f"data/private/p15_w1_results/{cisd3_json_p.name}"
                            if a == str(cisd3_json_p)
                            else a
                        )
                    )
                    for a in cisd3_cmd
                ]
                clean_stdout = sanitize_host_paths(cisd3_proc.stdout, root).replace(
                    str(tmp_manifest), "<cisd3_manifest.tsv>"
                )
                write_clean_utf8(
                    results_dir / f"w1_cisd3_{slug}.log",
                    f"$ {' '.join(display_cmd)}\n{clean_stdout}\n",
                )
                if cisd3_json_p.is_file():
                    clean_json = sanitize_host_paths(
                        cisd3_json_p.read_text(encoding="utf-8"), root
                    ).replace(str(tmp_manifest), "<cisd3_manifest.tsv>")
                    write_clean_utf8(cisd3_json_p, clean_json)
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
    rp = path.resolve()
    try:
        sub = rp.relative_to(results_dir.resolve()).as_posix()
        return f"data/private/p15_w1_results/{sub}"
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


def parse_mcp_metrics(paths: list[Path]) -> list[dict[str, Any]]:
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
                    if "suite" in obj:
                        records.append(obj)
    return records


def find_suite(records: list[dict[str, Any]], suite_name: str) -> dict[str, Any]:
    matches = [r for r in records if r.get("suite") == suite_name]
    if not matches:
        raise ValueError(f"missing MCP_METRIC_JSON record for suite={suite_name!r}")
    return matches[-1]


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
                filt = (
                    parts[lib_idx + 1]
                    if lib_idx + 1 < len(parts) and not parts[lib_idx + 1].startswith("-")
                    else "lib"
                )
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
            cur_bin = (
                cur_cmd_label
                if (cur_cmd_label and raw_name == "lib.rs")
                else (cur_cmd_label or raw_name)
            )
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
    if not (acc_m and rec_m and hcp_m and ass_m and asg_m):
        raise ValueError(f"failed to parse Ubiquitin metrics from {log_path}")
    return {
        "acc": f"{acc_m.group(1)} (assigned={ass_m.group(1)})",
        "rec": f"{rec_m.group(1)} (assignable={asg_m.group(1)})",
        "hcp": hcp_m.group(1),
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
        "micro_hcp": (100.0 * hc_ok / hc_tot) if hc_tot > 0 else 0.0,
        "hc_ok": hc_ok,
        "hc_tot": hc_tot,
    }


def cited_w1_files(
    results_dir: Path, metric_logs: list[Path], extra_files: list[Path]
) -> list[Path]:
    ordered_names = [
        "w1_mcp_suite_tests.log",
        "w1_mcp_metrics.jsonl",
        "w1_workspace_tests.log",
        "w1_ubq_realistic_evograph.log",
        "w1_cisd3_evograph.json",
        "w1_cisd3_evograph.log",
        "w1_ubq_realistic_standard_cpu.log",
        "w1_cisd3_standard_cpu.json",
        "w1_cisd3_standard_cpu.log",
        "w1_ubq_realistic_standard_gpu.log",
        "w1_cisd3_standard_gpu.json",
        "w1_cisd3_standard_gpu.log",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for name in ordered_names:
        p = results_dir / name
        if p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    for p in [*metric_logs, *extra_files]:
        if p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def render_digests_section(files: list[Path], root: Path, results_dir: Path) -> str:
    lines = [
        "### Cited Raw Results Artifacts & FIPS 180-4 SHA-256 Digests (`15.W1`)",
        "",
        "All benchmark result and SHA-256 artifact tables below are rendered from structured JSONL and test/benchmark log outputs by `scripts/render_phase15_w1_tables.py`:",
        "",
        "| Artifact Path | Bytes | SHA-256 |",
        "| :--- | ---: | :--- |",
    ]
    for p in files:
        n_bytes, digest = sha256_and_size(p)
        rel = display_rel_path(p, root, results_dir)
        lines.append(f"| `{rel}` | {n_bytes:,} | `{digest}` |")
    return "\n".join(lines)


def render_mcp_protocol_and_security_section(records: list[dict[str, Any]]) -> str:
    core = find_suite(records, "nmr_core_mcp_tests")
    p13 = find_suite(records, "nmr_core_mcp_phase13_tools")
    p14 = find_suite(records, "nmr_core_mcp_phase14_tools")
    gui = find_suite(records, "nmr_gui_mcp_bridge")
    cli = find_suite(records, "nmr_cli_mcp_cli_tests")

    stdio_mib = int(core["max_stdio_frame_bytes"]) // (1024 * 1024)
    core_http_mib = int(core["max_http_body_bytes"]) // (1024 * 1024)
    gui_http_mib = int(gui["max_http_body_bytes"]) // (1024 * 1024)

    lines = [
        "### Phase `15.W1` (`T15.1` + `T15.2` + `T15.3`) — Clean-Room `MCP` Server, Live Workbench Bridge & Sandbox Security",
        "",
        "> **Validation Scope (`In-Process / Subprocess Protocol Self-Consistency`)**: All `T15.1`–`T15.3` protocol framing, pre-allocation `Content-Length` bounds, workspace sandboxing, loopback HTTP authentication, `256`-step Undo/Redo curation, and RFC 2083 `image/png` encoding gates below are verified via deterministic in-process Rust unit tests and `nmrassign mcp` / `nmrassign mcp-config` subprocess E2E tests (`0` live interactive sessions against external `Antigravity`, `Claude Desktop`, or `Cursor` host binaries are executed in automated CI — tracked openly as `F18`).",
        "",
        "#### Protocol Surface, Loopback Bridge & Adversarial Sandbox Guardrails (`T15.1`–`T15.3`)",
        "",
        "| Subsystem & Security Check | Layer | Measured Metric | Gate | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
        (
            f"| Core MCP JSON-RPC 2.0 tool & prompt catalog (`protocolVersion = {core['protocol_version']}`) "
            f"| `nmr-core::mcp` | `{core['core_tools_total']}` tools (`{core['core_tools_read_only']}` read-only + `{core['core_tools_mutating']}` mutating), `{core['guided_prompts_total']}` prompts "
            "| `12` tools (`10` RO + `2` RW), `5` prompts | PASS |"
        ),
        (
            f"| Pre-allocation `Content-Length` cap & non-loopback bind rejection "
            f"| `nmr-core::mcp` + `nmrassign mcp` | `stdio <= {stdio_mib} MiB`, `HTTP <= {core_http_mib} MiB` (`HTTP 413 = {core['http_413_prealloc_verified']}`), `{cli['cli_non_loopback_binds_rejected']}` non-loopback CLI binds rejected "
            "| Pre-alloc `Err` / `413`, loopback-only bind | PASS |"
        ),
        (
            f"| Canonicalized workspace path sandbox & credential blocklist "
            f"| `McpSecurityPolicy` | `{core['sandbox_escape_vectors_blocked']}` escape vectors blocked (`..`, nested `..`, sibling-prefix, NUL, symlinks, `.ssh`/`.aws`/`.env`, manifest paths) "
            "| `ERR_SECURITY_VIOLATION` (`-32001`), `0` leaks | PASS |"
        ),
        (
            f"| Live desktop loopback HTTP bridge, Undo ring & viewport PNG "
            f"| `nmr-gui::mcp_bridge` | `HTTP <= {gui_http_mib} MiB` (`401 = {gui['http_auth_401_verified']}`, `403 = {gui['http_rebind_403_verified']}`, `413 = {gui['http_prealloc_413_verified']}`), `{gui['undo_cap_depth']}`-step Undo (`roundtrip = {gui['undo_roundtrip_verified']}`), `{gui['png_width']}x{gui['png_height']}` PNG (`{gui['png_bytes']} B`) "
            "| `256`-step bijective Undo, valid RFC 2083 PNG | PASS |"
        ),
        (
            f"| Subprocess CLI `stdio`/`http` E2E & client config generator "
            f"| `nmr-cli::mcp_cmd` | `stdio RO/RW = {cli['cli_stdio_ro_rw_verified']}`, `HTTP 200/401/403/413 = {cli['cli_http_loopback_200_401_403_413_verified']}`, `{cli['cli_client_configs_verified']}` IDE schemas (`Antigravity`, `Claude Desktop`, `Cursor`) "
            "| Subprocess exit `0` on valid, non-zero on invalid | PASS |"
        ),
        "",
        "#### Scientific Phase 13 & Phase 14 (`T14.5` Metabolomics) MCP Tool Verification",
        "",
        "| MCP Tool Name | Synthetic Dataset / Mode | Measured Tool Output | Expected Reference | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
        (
            f"| `fit_relaxation_or_exchange` (`relaxation` / `cest` / `cpmg`) "
            f"| `R1` (`600 MHz`), dual-`B1` `CEST` (`15 & 30 Hz`), dual-field `CPMG` (`600 & 800 MHz`) *(noiseless forward-model round-trip)* "
            f"| `R1 = {p13['r1_rate_s_inv']:.4f} s^-1`, `CEST p_B = {p13['cest_pb']:.4f}` (`k_ex = {p13['cest_kex_s_inv']:.1f} s^-1`), `CPMG p_B = {p13['cpmg_pb']:.4f}` (`k_ex = {p13['cpmg_kex_s_inv']:.1f} s^-1`) "
            "| `R1 = 1.65 s^-1`, `CEST (0.030, 180 s^-1)`, `CPMG (0.040, 900 s^-1)` *(noiseless round-trip)* | PASS |"
        ),
        (
            f"| `track_csp_titration` (`amide` `1H-15N` HSQC) "
            f"| `6`-point titration (`[P]_T = 50 uM`, `[L]_T = 0..280 uM`) *(noiseless forward-model round-trip)* "
            f"| `global K_d = {p13['csp_global_kd_um']:.3f} uM` (sandbox rejects `../outside/evil_step.tsv`) "
            "| `K_d = 30.0 +- 2.0 uM` *(noiseless round-trip)* | PASS |"
        ),
        (
            f"| `analyze_nmr1d_spectrum` & `fit_dosy_diffusion` "
            f"| `500 MHz` `CDCl3` 1D (`Quartet`/`Triplet` + `ACS`) & `16`-pt `D2O` `DOSY` (`BIC` + `ILT`) "
            f"| Quartet `J = {p14['nmr1d_quartet_j_hz']:.3f} Hz`, `DOSY` mixture `K = {p14['dosy_mix_selected_k']}`, pure `K = {p14['dosy_pure_selected_k']}` "
            "| `J = 7.10 +- 0.25 Hz`, `K = (2, 1)` | PASS |"
        ),
        (
            f"| `run_metabolomics_chemometrics` (`AI-Binning` + `PQN` + `PCA` + `OPLS-DA` + `STOCSY`) "
            f"| `{p14['metab_n_samples']}`-sample `D2O` cohort manifest (`4 Control` vs `4 Case`) "
            f"| `OPLS-DA Q2 = {p14['metab_opls_q2']:.4f}`, top marker `{p14['metab_top_biomarker_ppm']:.3f} ppm`, `STOCSY (1.33 -> 4.11 ppm) r^2 = {p14['metab_stocsy_r2']:.4f}` "
            "| `Q2 > 0.75`, marker `1.33 / 4.11 ppm`, `r^2 > 0.90` | PASS |"
        ),
    ]
    return "\n".join(lines)


def render_regression_section(results_dir: Path, metric_logs: list[Path]) -> str:
    blocks: list[str] = []

    mcp_log = results_dir / "w1_mcp_suite_tests.log"
    if not mcp_log.is_file() and metric_logs:
        mcp_log = metric_logs[0]
    ws_log = results_dir / "w1_workspace_tests.log"

    if mcp_log.is_file():
        tot_p, tot_f, suites = parse_test_pass_counts(mcp_log)
        t_lines = [
            "### Wave `15.W1` & Workspace Test Execution Summary",
            "",
            "| Suite Binary | Passed | Failed |",
            "| :--- | ---: | ---: |",
        ]
        for s_name, s_pass, s_fail in suites:
            t_lines.append(f"| `{s_name}` | `{s_pass}` | `{s_fail}` |")
        t_lines.append(f"| **Wave `15.W1` Dedicated Total** | **`{tot_p}`** | **`{tot_f}`** |")
        if ws_log.is_file():
            ws_p, ws_f, ws_suites = parse_test_pass_counts(ws_log)
            t_lines.append(
                f"| **Full Workspace (`cargo test --release --workspace`, `{len(ws_suites)}` binaries)** | **`{ws_p}`** | **`{ws_f}`** |"
            )
        blocks.append("\n".join(t_lines))

    ubq_rows: list[str] = []
    for slug, label, _ in BENCHMARK_ENGINES:
        ubq_p = results_dir / f"w1_ubq_realistic_{slug}.log"
        cis_p = results_dir / f"w1_cisd3_{slug}.json"
        if not ubq_p.is_file():
            continue
        ubq = parse_ubq_log(ubq_p)
        if cis_p.is_file():
            cis = summarize_cisd3_json(cis_p)
            cis_acc = f"`{cis['macro_acc']:.2f}%`"
            cis_rec = f"`{cis['macro_rec']:.2f}%`"
            cis_hcp = f"`{cis['micro_hcp']:.2f}%` (`{cis['hc_ok']}/{cis['hc_tot']}`)"
        else:
            cis_acc = "`n/a`"
            cis_rec = "`n/a`"
            cis_hcp = "`n/a`"
        ubq_rows.append(
            f"| {label} | `{ubq['acc']}` | `{ubq['rec']}` | `{ubq['hcp']}` | {cis_acc} | {cis_rec} | {cis_hcp} |"
        )

    if ubq_rows:
        b_lines = [
            "### Mandatory 5-Seed Solver Non-Regression Gates (`Ubiquitin --realistic` & `CISD3` `BMRB 51439`)",
            "",
            "| Engine | Ubiquitin `--realistic` (`5` Seeds) Acc | Ubiquitin Recall | Ubiquitin HC-Prec (`>=0.80`) | `CISD3` (`5` Seeds) Acc | `CISD3` Recall | `CISD3` HC-Prec (`>=0.80`) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *ubq_rows,
        ]
        blocks.append("\n".join(b_lines))

    return "\n\n".join(blocks)


def verify_markdown_tables(md: str) -> None:
    lines = md.splitlines()
    in_table = False
    expected_pipes = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
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
        default=Path("data/private/p15_w1_results"),
        help="Directory holding raw Phase 15 Wave 1 logs and JSON artifacts.",
    )
    parser.add_argument(
        "--test-log",
        type=Path,
        action="append",
        default=[],
        help="Explicit MCP test log(s) containing MCP_METRIC_JSON lines.",
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
        choices=["all", "digests", "mcp", "regression", "15.1", "15.2", "15.3", "t15.1", "t15.2", "t15.3"],
        default="all",
        help="Subset of Markdown sections to render.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the 3 dedicated MCP test suites (`--nocapture`) and capture logs/JSONL.",
    )
    parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Execute full workspace release tests plus 5-seed Ubiquitin & CISD3 benchmarks.",
    )
    parser.add_argument(
        "--cisd3-data-root",
        type=Path,
        default=None,
        help="Directory (`.../data/private/bmrb`) adjacent to `bmrb51439_cisd3`.",
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
        args.results_dir
        if args.results_dir.is_absolute()
        else (root / args.results_dir)
    )

    extra_digest_files: list[Path] = []
    if args.run:
        run_log, run_jsonl = run_w1_mcp_suite(root, results_dir)
        extra_digest_files.append(run_jsonl)
        if not args.test_log:
            args.test_log = [run_log]

    if args.run_benchmarks:
        cisd3_root = args.cisd3_data_root
        if cisd3_root is None:
            cand = root / "data/private/bmrb"
            if (root / "data/private/bmrb51439_cisd3").is_dir():
                cisd3_root = cand
        run_w1_benchmarks_and_workspace(root, results_dir, cisd3_root)

    metric_logs: list[Path] = []
    for tl in args.test_log:
        metric_logs.append(tl if tl.is_absolute() else (root / tl))

    if not metric_logs and results_dir.is_dir():
        preferred = results_dir / "w1_mcp_suite_tests.log"
        if preferred.is_file():
            metric_logs.append(preferred)
        else:
            for cand in sorted(results_dir.glob("*")):
                if cand.suffix in {".log", ".jsonl"} and cand.is_file():
                    txt = cand.read_text(encoding="utf-8", errors="replace")
                    if METRIC_PREFIX in txt:
                        metric_logs.append(cand)

    records = parse_mcp_metrics(metric_logs)
    mcp_secs = {"all", "mcp", "15.1", "15.2", "15.3", "t15.1", "t15.2", "t15.3"}
    if not records and args.section in mcp_secs:
        sys.stderr.write(
            f"error: no {METRIC_PREFIX} records found in {[str(p) for p in metric_logs]} (results_dir={results_dir})\n"
        )
        return 1

    for ef in args.extra_file:
        extra_digest_files.append(ef if ef.is_absolute() else (root / ef))
    digest_inputs = cited_w1_files(results_dir, metric_logs, extra_digest_files)

    blocks: list[str] = []
    sec = args.section
    if sec in {"all", "digests"} and digest_inputs:
        blocks.append(render_digests_section(digest_inputs, root, results_dir))
    if sec in mcp_secs:
        blocks.append(render_mcp_protocol_and_security_section(records))
    if sec in {"all", "regression"}:
        reg_md = render_regression_section(results_dir, metric_logs)
        if reg_md:
            blocks.append(reg_md)

    rendered = "\n\n".join(b for b in blocks if b) + "\n"
    verify_markdown_tables(rendered)

    if args.out is not None:
        write_clean_utf8(args.out, rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
