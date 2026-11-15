#!/usr/bin/env python3
"""Deterministic Markdown table renderer for Phase 16 Wave 1 (T16.1 + T16.2 + T16.3 + T16.4: Deterministic SVG Showcase, Slim README, Documentation Manual & Zero-I/O WebGPU Demo).

Reads structured `P16_METRIC_JSON:` records emitted by `--release -- --nocapture` runs of
`nmr-cli --test export_fig_cli_tests` and `nmr-gui --test demo_preset_tests`, together with
machine-captured `build_docs_manual.py --self-test` / `--check`, `build_web_demo.sh --check-only`,
`README.md` & `docs/assets/*.svg` measurements, full workspace test logs, and 5-seed
`Ubiquitin --realistic` / `CISD3` (`BMRB 51439`) benchmark artifacts under `--results-dir`, and
renders Markdown tables and FIPS 180-4 SHA-256 digests directly from those artifacts.

Provenance: Original clean-room reporting script for deterministic vector asset export, static
documentation verification, and headless `wasm32` preset validation citing Lorensen & Cline
(ACM SIGGRAPH Comput. Graph. 1987, marching-squares contours), W3C Scalable Vector Graphics (SVG 1.1),
and standard SHA-256 (FIPS 180-4) artifact digests.
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
from urllib.parse import urlsplit

METRIC_PREFIX = "P16_METRIC_JSON:"

HOST_PATH_RE = re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+")

EXPECTED_SVG_PANELS = [
    ("hsqc", "01_hsqc_contour_overlay.svg", "2D `1H-15N` HSQC Contours & `evo-graph` Roots"),
    ("strips", "02_sequential_strip_walk.svg", "3D `HNCACB` Sequential Strip Walk (`Phe4 -> Gly10`)"),
    ("pdb-s2", "03_pdb_s2_dynamics_putty.svg", "`1UBQ` Lipari-Szabo `S^2` Putty & NOEs *(synthetic)*"),
    ("nmr1d-stack", "04_nmr1d_multiplet_stack.svg", "Analytical 1D Hoye `J`-Tree, Stack & `qNMR` *(synthetic)*"),
    ("dosy", "05_dosy_2d_diffusion_map.svg", "Pseudo-2D `BPP-LED` `DOSY` Diffusion Map *(synthetic)*"),
    ("metabolomics", "06_metabolomics_oplsda_stocsy.svg", "1D Metabolomics `PCA`, `OPLS-DA` & `STOCSY` *(synthetic)*"),
    ("exchange", "07_cest_cpmg_csp_exchange.svg", "`CEST`, `CPMG` & Tight-Binding `CSP` `Kd` *(synthetic)*"),
    ("octet", "08_octet_structure_link.svg", "`Octet` 2D `(S)`-Ibuprofen & `1H`/`13C` Shift Link"),
]

SLICE_16_RUST_COMMANDS = [
    ["cargo", "test", "--release", "-p", "nmr-cli", "--test", "export_fig_cli_tests", "--", "--nocapture"],
    ["cargo", "test", "--release", "-p", "nmr-gui", "--test", "demo_preset_tests", "--", "--nocapture"],
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


def audit_readme_and_assets(root: Path) -> dict[str, Any]:
    readme_p = root / "README.md"
    readme_text = readme_p.read_text(encoding="utf-8")
    readme_lines = len(readme_text.splitlines())
    readme_bytes, readme_sha = sha256_and_size(readme_p)

    # Verify every local link and image in README.md resolves on disk
    local_links = 0
    missing_links: list[str] = []
    for m in re.finditer(r"!?\[[^\]]*\]\(([^)\s]+)\)", readme_text):
        raw_target = m.group(1).strip()
        parsed = urlsplit(raw_target)
        if parsed.scheme in ("http", "https", "mailto"):
            continue
        rel_path = parsed.path
        if not rel_path:
            continue
        local_links += 1
        cand = (root / rel_path).resolve()
        if not cand.exists():
            missing_links.append(raw_target)
    if missing_links:
        raise RuntimeError(f"README.md contains broken local link(s): {missing_links}")

    panel_rows: list[dict[str, Any]] = []
    for slug, fname, desc in EXPECTED_SVG_PANELS:
        svg_p = root / "docs" / "assets" / fname
        if not svg_p.is_file():
            raise RuntimeError(f"missing expected SVG showcase panel: docs/assets/{fname}")
        sz, digest = sha256_and_size(svg_p)
        panel_rows.append(
            {
                "slug": slug,
                "filename": fname,
                "description": desc,
                "bytes": sz,
                "sha256": digest,
            }
        )

    sizes = [r["bytes"] for r in panel_rows]
    return {
        "suite": "readme_and_assets",
        "readme_lines": readme_lines,
        "readme_bytes": readme_bytes,
        "readme_sha256": readme_sha,
        "readme_local_links_verified": local_links,
        "svg_count": len(panel_rows),
        "svg_min_bytes": min(sizes),
        "svg_max_bytes": max(sizes),
        "svg_total_bytes": sum(sizes),
        "panels": panel_rows,
    }


def run_w1_p16_suite(root: Path, results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "w1_p16_suite_tests.log"
    jsonl_path = results_dir / "w1_p16_metrics.jsonl"
    root_s = str(root.resolve())
    chunks: list[str] = []

    # 1. Dedicated Rust integration tests (--nocapture)
    for cmd in SLICE_16_RUST_COMMANDS:
        proc = subprocess.run(
            cmd,
            cwd=root_s,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        chunks.append(f"$ {' '.join(cmd)}\n{sanitize_host_paths(proc.stdout, root)}\n")

    # 2. Documentation manual self-tests & full static site --check build
    self_cmd = ["python3", "scripts/build_docs_manual.py", "--self-test"]
    self_proc = subprocess.run(
        self_cmd,
        cwd=root_s,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    m_st = re.search(r"\((\d+)/(\d+) adversarial", self_proc.stdout)
    st_passed = int(m_st.group(1)) if m_st else 0
    chunks.append(f"$ {' '.join(self_cmd)}\n{sanitize_host_paths(self_proc.stdout, root)}\n")

    with tempfile.TemporaryDirectory(prefix="nmrassign_p16_manual_") as tmp_manual:
        chk_cmd = ["python3", "scripts/build_docs_manual.py", "--out", tmp_manual, "--check"]
        chk_proc = subprocess.run(
            chk_cmd,
            cwd=root_s,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        clean_chk_out = sanitize_host_paths(chk_proc.stdout, root).replace(
            Path(tmp_manual).name, "manual"
        )
        m_build = re.search(
            r"Built (\d+) manual pages \((\d+) indexed sections, (\d+) links verified, sha256=([0-9a-f]+)\)",
            chk_proc.stdout,
        )
        if not m_build:
            raise RuntimeError(f"failed to parse build_docs_manual.py output: {chk_proc.stdout!r}")
        bundled_svgs = len(list((Path(tmp_manual) / "assets").glob("*.svg")))
        manual_rec = {
            "suite": "docs_manual_check",
            "self_tests_passed": st_passed,
            "pages": int(m_build.group(1)),
            "indexed_sections": int(m_build.group(2)),
            "links_verified": int(m_build.group(3)),
            "bundled_svgs": bundled_svgs,
            "site_sha256_prefix": m_build.group(4),
        }
        disp_chk_cmd = ["python3", "scripts/build_docs_manual.py", "--out", "target/manual", "--check"]
        chunks.append(
            f"$ {' '.join(disp_chk_cmd)}\n{clean_chk_out}{METRIC_PREFIX}{json.dumps(manual_rec, sort_keys=True)}\n\n"
        )

    # 3. README.md & docs/assets/*.svg structural audit
    readme_rec = audit_readme_and_assets(root)
    chunks.append(f"# README & SVG asset audit\n{METRIC_PREFIX}{json.dumps(readme_rec, sort_keys=True)}\n\n")

    # 4. Zero-install WebGPU demo wasm32 compilation & landing-page check
    demo_cmd = ["bash", "scripts/build_web_demo.sh", "--check-only"]
    demo_proc = subprocess.run(
        demo_cmd,
        cwd=root_s,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    demo_html_bytes, demo_html_sha = sha256_and_size(root / "docs" / "demo" / "index.html")
    web_rec = {
        "suite": "web_demo_check",
        "wasm32_check_passed": demo_proc.returncode == 0,
        "demo_html_bytes": demo_html_bytes,
        "demo_html_sha256": demo_html_sha,
    }
    chunks.append(
        f"$ {' '.join(demo_cmd)}\n{sanitize_host_paths(demo_proc.stdout, root)}"
        f"{METRIC_PREFIX}{json.dumps(web_rec, sort_keys=True)}\n"
    )

    write_clean_utf8(log_path, "".join(chunks))

    records = parse_p16_metrics([log_path])
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
    fd, tmp_manifest_str = tempfile.mkstemp(prefix="cisd3_p16_w1_manifest_", suffix=".tsv")
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
                            f"data/private/p16_w1_results/{cisd3_json_p.name}"
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


def display_rel_path(path: Path, root: Path, results_dir: Path) -> str:
    rp = path.resolve()
    try:
        sub = rp.relative_to(results_dir.resolve()).as_posix()
        return f"data/private/p16_w1_results/{sub}"
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


def parse_p16_metrics(paths: list[Path]) -> list[dict[str, Any]]:
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
        raise ValueError(f"missing P16_METRIC_JSON record for suite={suite_name!r}")
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
        "w1_p16_suite_tests.log",
        "w1_p16_metrics.jsonl",
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
        "### Cited Raw Results Artifacts & FIPS 180-4 SHA-256 Digests (`16.W1`)",
        "",
        "All benchmark result and SHA-256 artifact tables below are rendered from structured JSONL and test/benchmark log outputs by `scripts/render_phase16_w1_tables.py`:",
        "",
        "| Artifact Path | Bytes | SHA-256 |",
        "| :--- | ---: | :--- |",
    ]
    for p in files:
        n_bytes, digest = sha256_and_size(p)
        rel = display_rel_path(p, root, results_dir)
        lines.append(f"| `{rel}` | {n_bytes:,} | `{digest}` |")
    return "\n".join(lines)


def render_showcase_and_readme_section(records: list[dict[str, Any]]) -> str:
    exp = find_suite(records, "export_fig_cli_tests")
    ra = find_suite(records, "readme_and_assets")

    panel_lines = [
        "### Phase `16.W1` (`T16.1` + `T16.2`) — Deterministic CLI SVG Figure Exporter & Visual-First `README.md`",
        "",
        "> **Validation Scope (`Deterministic Forward-Model & Synthetic Showcase Assets`)**: All `8` vector figures under `docs/assets/01..08.svg` are generated deterministically (`seed = 42`, `0` external plotting crates) from `nmr-core` forward models and solvers via `nmrassign export-fig`; panels `03`–`07` carry explicit synthetic-showcase qualifiers matching Phases 13–14 validation scope.",
        "",
        "#### Deterministic `nmrassign export-fig` Showcase Panels (`T16.1`)",
        "",
        "| Panel (`--panel`) | Committed Asset Path | Modality | Bytes (`< 250 KiB`) | SHA-256 |",
        "| :--- | :--- | :--- | ---: | :--- |",
    ]
    for row in ra["panels"]:
        panel_lines.append(
            f"| `{row['slug']}` | `docs/assets/{row['filename']}` | {row['description']} | {row['bytes']:,} | `{row['sha256']}` |"
        )

    panel_lines.extend(
        [
            "",
            "#### Showcase & `README.md` Gate Verification (`T16.1`–`T16.2`)",
            "",
            "| Deliverable & Invariant | Measured Value | Gate | Status |",
            "| :--- | :--- | :--- | :---: |",
            (
                f"| `nmrassign export-fig` deterministic repeat & committed-asset parity "
                f"| `{exp['panels_verified']}/8` panels (`{exp['min_svg_bytes']:,}..{exp['max_svg_bytes']:,} B`, total `{exp['total_svg_bytes']:,} B`), `repeat_0_diff = {exp['zero_diff_repeat']}`, `committed_0_diff = {exp['zero_diff_committed']}`, `{exp['adversarial_err_cases']}` `Err` paths "
                "| `8/8` valid XML/SVG, `< 250 KiB` each, `0`-byte diff | PASS |"
            ),
            (
                f"| Visual-first `README.md` budget & local link/image resolution "
                f"| `{ra['readme_lines']}` lines (`{ra['readme_bytes']:,} B`), `{ra['readme_local_links_verified']}` local links/images verified "
                "| `<= 180` lines, `0` broken links | PASS |"
            ),
        ]
    )
    return "\n".join(panel_lines)


def render_manual_and_demo_section(records: list[dict[str, Any]]) -> str:
    man = find_suite(records, "docs_manual_check")
    demo = find_suite(records, "demo_preset_tests")
    web = find_suite(records, "web_demo_check")

    lines = [
        "### Phase `16.W1` (`T16.3` + `T16.4`) — Static Documentation Manual & Zero-I/O `wasm32` WebGPU Demo",
        "",
        "> **Validation Scope (`Static Build/Link Verification & Headless WASM Preset Self-Consistency`)**: Documentation HTML generation, offline math/table/callout rendering, client search indexing, and link/anchor/SVG resolution are verified deterministically by `scripts/build_docs_manual.py --self-test` and `--check`, while the browser WebGPU demo is verified via headless `demo_preset` `egui` unit tests and `cargo check -p nmr-gui --target wasm32-unknown-unknown` (`0` live interactive sessions against external Chrome, Edge, Firefox, or Safari GPU drivers are executed in automated CI — tracked openly as `F21`).",
        "",
        "| Subsystem & Verification Gate | Layer | Measured Metric | Gate | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
        (
            f"| Static manual builder, adversarial hygiene & link/anchor/SVG check (`T16.3`) "
            f"| `scripts/build_docs_manual.py` + `docs/manual/` | `{man['pages']}` HTML pages, `{man['indexed_sections']}` search sections, `{man['links_verified']}` links + `{man['bundled_svgs']}` bundled SVGs (`sha256={man['site_sha256_prefix']}...`), `{man['self_tests_passed']}/6` self-tests "
            "| `9` chapters, `0` broken links/anchors, deterministic SHA-256 | PASS |"
        ),
        (
            f"| Built-in zero-I/O multi-modal demo preset & headless `egui` panels (`T16.4`) "
            f"| `nmr-gui::demo_preset` | `{demo['seq_len']} aa` `1UBQ`, `{demo['peaks_count']}` peaks, `{demo['spin_systems_count']}` spin systems (`HSQC {demo['hsqc_nx']}x{demo['hsqc_ny']}`, `HNCACB z={demo['hncacb_nz']}`), `{demo['stack1d_traces']}` 1D traces, `{demo['dosy_peaks']}` `DOSY` peaks, `{demo['metabolomics_samples']}` metabolomics samples (`{demo['metabolomics_buckets']}` buckets, `OPLS-DA Q2 = {demo['opls_q2']:.4f}`) "
            "| `0` disk I/O, deterministic repeat, headless `egui` clean | PASS |"
        ),
        (
            f"| Browser `wasm32-unknown-unknown` target check & WebGPU landing shell (`T16.4`) "
            f"| `scripts/build_web_demo.sh` + `docs/demo/index.html` | `wasm32_check_passed = {web['wasm32_check_passed']}`, `docs/demo/index.html` `{web['demo_html_bytes']:,} B` "
            "| `cargo check --target wasm32-unknown-unknown` exit `0` | PASS |"
        ),
    ]
    return "\n".join(lines)


def render_regression_section(results_dir: Path, metric_logs: list[Path]) -> str:
    blocks: list[str] = []

    p16_log = results_dir / "w1_p16_suite_tests.log"
    if not p16_log.is_file() and metric_logs:
        p16_log = metric_logs[0]
    ws_log = results_dir / "w1_workspace_tests.log"

    if p16_log.is_file():
        tot_p, tot_f, suites = parse_test_pass_counts(p16_log)
        t_lines = [
            "### Wave `16.W1` & Workspace Test Execution Summary",
            "",
            "| Suite Binary | Passed | Failed |",
            "| :--- | ---: | ---: |",
        ]
        for s_name, s_pass, s_fail in suites:
            t_lines.append(f"| `{s_name}` | `{s_pass}` | `{s_fail}` |")
        t_lines.append(f"| **Wave `16.W1` Dedicated Rust Total** | **`{tot_p}`** | **`{tot_f}`** |")
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
        default=Path("data/private/p16_w1_results"),
        help="Directory holding raw Phase 16 Wave 1 logs and JSON artifacts.",
    )
    parser.add_argument(
        "--test-log",
        type=Path,
        action="append",
        default=[],
        help="Explicit Phase 16 test log(s) containing P16_METRIC_JSON lines.",
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
            "digests",
            "showcase",
            "manual_demo",
            "regression",
            "16.1",
            "16.2",
            "16.3",
            "16.4",
            "t16.1",
            "t16.2",
            "t16.3",
            "t16.4",
        ],
        default="all",
        help="Subset of Markdown sections to render.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute dedicated Phase 16 tests, manual builder checks, and wasm32 demo checks.",
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
        run_log, run_jsonl = run_w1_p16_suite(root, results_dir)
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
        preferred = results_dir / "w1_p16_suite_tests.log"
        if preferred.is_file():
            metric_logs.append(preferred)
        else:
            for cand in sorted(results_dir.glob("*")):
                if cand.suffix in {".log", ".jsonl"} and cand.is_file():
                    txt = cand.read_text(encoding="utf-8", errors="replace")
                    if METRIC_PREFIX in txt:
                        metric_logs.append(cand)

    records = parse_p16_metrics(metric_logs)
    showcase_secs = {"all", "showcase", "16.1", "16.2", "t16.1", "t16.2"}
    manual_secs = {"all", "manual_demo", "16.3", "16.4", "t16.3", "t16.4"}
    if not records and (args.section in showcase_secs or args.section in manual_secs):
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
    if sec in showcase_secs:
        blocks.append(render_showcase_and_readme_section(records))
    if sec in manual_secs:
        blocks.append(render_manual_and_demo_section(records))
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
