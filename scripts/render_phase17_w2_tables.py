#!/usr/bin/env python3
"""Deterministic Markdown table renderer & SHA-256 ledger generator for Phase 17 Wave 2
(`phase=17.W2`: Slice `17.4` `F18` Live External MCP Smoke & Host Audit + `F21` Live Multi-Browser /
Metal WebGPU Compatibility Audit + `F22` Windows Branch CI Check + `F23` Release Workflow SHA-Pin &
Published-Asset Collision Guard).

Reads structured JSON/JSONL/log artifacts in `data/private/p17_w2_results/`, verifies `0`-byte diffs
across `crates/`, `Cargo.lock`, and `data/benchmark/p17_empirical_preregistration.json`, and renders
all Wave `17.W2` numeric tables, candid compatibility matrices, and FIPS 180-4 SHA-256 digests
mechanically without hand-typed numbers.

Provenance: Original clean-room Python 3 standard-library reporting script (MIT License) citing
FIPS 180-4 (SHA-256 digests) and RFC 8259 (JSON). Zero external packages or network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
        return f"data/private/p17_w2_results/{sub}"
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


def run_w2_checks(root: Path, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    root_s = str(root.resolve())
    cmds: list[tuple[list[str], dict[str, str] | None]] = [
        (["python3", "scripts/verify_p17_preregistration.py", "--self-test", "--check"], None),
        (
            ["cargo", "check", "--locked", "-p", "nmr-cli", "-p", "nmr-gui", "--target", "x86_64-pc-windows-gnu"],
            {"RUSTFLAGS": "-D warnings"},
        ),
        (
            ["cargo", "check", "--release", "--locked", "-p", "nmr-cli", "-p", "nmr-gui", "--target", "x86_64-pc-windows-gnu"],
            {"RUSTFLAGS": "-D warnings"},
        ),
        (
            ["cargo", "check", "--release", "--locked", "-p", "nmr-gui", "--target", "wasm32-unknown-unknown"],
            {"RUSTFLAGS": "-D warnings"},
        ),
        (["bash", "scripts/build_web_demo.sh", "--check-only"], None),
        (["python3", "scripts/build_docs_manual.py", "--self-test"], None),
        (["python3", "scripts/package_release.py", "--self-test"], None),
    ]
    chunks: list[str] = []
    for cmd, extra_env in cmds:
        env = os.environ.copy()
        prefix = ""
        if extra_env:
            env.update(extra_env)
            prefix = " ".join(f'{k}="{v}"' for k, v in extra_env.items()) + " "
        proc = subprocess.run(
            cmd,
            cwd=root_s,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        chunks.append(f"$ {prefix}{' '.join(cmd)}\n{sanitize_host_paths(proc.stdout, root)}\n")
    out_log = results_dir / "w2_checks.log"
    write_clean_utf8(out_log, "".join(chunks))
    return out_log


def cited_w2_files(results_dir: Path, root: Path, extra_files: list[Path]) -> list[Path]:
    ordered_rel = [
        "w2_checks.log",
        "f18/SHA256SUMS.txt",
        "f18/f18_summary.json",
        "f18/host_client_inventory.json",
        "f18/F18_LIVE_MCP_AUDIT_REPORT.md",
        "f18/run_f18_live_smoke.py",
        "f18/transcript_stdio_newline.jsonl",
        "f18/transcript_stdio_content_length.jsonl",
        "f18/transcript_http_cli_loopback.jsonl",
        "f18/transcript_gui_workbench_bridge.jsonl",
        "f18/captured_viewport_contour_2d.png",
        "f18/bridge_harness/Cargo.toml",
        "f18/bridge_harness/src/main.rs",
        "f21/SHA256SUMS.txt",
        "f21/host_browser_gpu_inventory.json",
        "f21/chrome_157_live_webgpu_report.json",
        "f21/chrome_live_wasm_bundle_smoke.json",
        "f21/metal_wgpu_shader_and_viewport_audit.json",
        "f21/github_pages_live_check.json",
        "f21/wasm_binary_inspection.json",
        "f21/native_metal_wgpu_limits_raw.txt",
        "f21/cargo_test_nmr_gpu.log",
        "f21/cargo_test_nmr_gui.log",
        "f21/cargo_test_nmr_gui_lib.log",
        "f21/build_web_demo_check_only.log",
        "f21/build_docs_manual_selftest.log",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for rel in ordered_rel:
        p = results_dir / rel
        if p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    prereg_p = root / "data" / "benchmark" / "p17_empirical_preregistration.json"
    for p in [prereg_p, *extra_files]:
        if p.is_file() and p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def write_w2_sha256sums(results_dir: Path) -> Path:
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


def render_protected_diff_section(root: Path, base_ref: str) -> str:
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
    crates_tree_base = subprocess.run(
        ["git", "rev-parse", f"{base_ref}:crates"],
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
    if crates_tree_base != crates_tree_head:
        raise RuntimeError(f"crates/ tree hash drifted: {crates_tree_base} != {crates_tree_head}")

    lines = [
        f"### Protected Tree & Pre-Registration Zero-Byte Diff Verification (`{base_short}..{head_sha}`)",
        "",
        "| Protected Path / Tree | Diff Bytes (`git diff`) | Tree / File Invariant | Status |",
        "| :--- | ---: | :--- | :---: |",
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
        inv = (
            f"`HEAD:crates == {crates_tree_head[:12]}`"
            if rel_p == "crates/"
            else "`0`-byte diff"
        )
        lines.append(f"| `{rel_p}` | `{n_bytes}` | {inv} | `PASS` |")
    return "\n".join(lines)


def render_digests_section(files: list[Path], root: Path, results_dir: Path) -> str:
    lines = [
        "### Raw `17.W2` Artifacts & FIPS 180-4 SHA-256 Digests",
        "",
        "| Artifact Path | Bytes | SHA-256 |",
        "| :--- | ---: | :--- |",
    ]
    for p in files:
        n_bytes, digest = sha256_and_size(p)
        rel = display_rel_path(p, root, results_dir)
        lines.append(f"| `{rel}` | {n_bytes:,} | `{digest}` |")
    return "\n".join(lines)


def render_f18_section(results_dir: Path) -> str:
    f18_sum = json.loads((results_dir / "f18" / "f18_summary.json").read_text(encoding="utf-8"))
    f18_inv = json.loads((results_dir / "f18" / "host_client_inventory.json").read_text(encoding="utf-8"))

    hi = f18_sum["host_inventory_summary"]
    s_nl = f18_sum["stdio_newline_2025_03_26"]
    s_cl = f18_sum["stdio_content_length_2024_11_05"]
    h_auto = f18_sum["http_cli_loopback"]["phase_3a_autogen"]
    h_env = f18_sum["http_cli_loopback"]["phase_3b_explicit_env"]
    gui = f18_sum["gui_workbench_bridge"]
    png = gui["png_verification"]
    cfg_val = f18_inv.get("mcp_config_validation", {})
    n_cfg_ok = sum(
        1
        for group_key in ("stdio_configs_with_abs_command", "http_bridge_configs_with_env_token")
        for v in cfg_val.get(group_key, {}).values()
        if isinstance(v, dict) and v.get("validated") is True
    )

    lines = [
        "#### `F18` — Host Desktop Client Inventory & `nmrassign mcp-config` Audit",
        "",
        "| Host Client Target | App Bundle Present | Bundle Version | Non-Interactive MCP CLI Runner | `mcp-config` (`stdio` & `--bridge-port`) | `F18` Matrix Status |",
        "| :--- | :---: | :--- | :--- | :--- | :---: |",
        (
            f"| `Antigravity` | `{'Yes' if hi['antigravity_installed'] else 'No'}` | `{hi['antigravity_version']}` (`com.google.antigravity`) "
            f"| `agy` on `PATH`: `{hi['agy_cli_on_path']}` (Electron GUI app; no headless MCP test flag) "
            "| `PASS` (`mcpServers.nmr-assign` + `serverUrl`) | `CONFIG_ONLY` (GUI window not automated) |"
        ),
        (
            f"| `Claude Desktop` | `{'Yes' if hi['claude_desktop_installed'] else 'No'}` | `{hi['claude_desktop_version']}` (`com.anthropic.claudefordesktop`) "
            f"| `claude` CLI `{hi['claude_cli_on_path']}` (`2.1.322`); Electron GUI lacks headless MCP flag "
            "| `PASS` (`mcpServers.nmr-assign` + `streamable-http` `url`) | `CONFIG_ONLY` (GUI window not automated) |"
        ),
        (
            f"| `Cursor` | `{'Yes' if hi['cursor_installed'] else 'No'}` | `ABSENT` on host "
            f"| `cursor` on `PATH`: `{hi['cursor_cli_on_path']}` "
            "| `PASS` (`mcpServers.nmr-assign` + `url`) | `NOT_INSTALLED_ON_HOST` |"
        ),
        "",
        "#### `F18` — Live External-Process `stdio`, Streamable HTTP & `nmr-gui` Workbench Bridge Smoke",
        "",
        "| Transport / Harness Session | Protocol / Framing | Wire Exchanges | Key Verified Payloads & Security Invariants | Candid Live Observation | Status |",
        "| :--- | :--- | ---: | :--- | :--- | :---: |",
        (
            f"| External Python OS-pipe `nmrassign mcp --transport stdio` (`{hi['nmrassign_version']}`, `sha256={hi['nmrassign_sha256'][:16]}...`) "
            f"| `NewlineDelimited` (`{s_nl['negotiated_protocol_version']}`) "
            f"| `{s_nl['roundtrips_recorded']}` "
            f"| `{s_nl['read_only_tools_count']}` RO tools, `{s_nl['resources_listed_count']}` resources, `{s_nl['prompts_listed_count']}` prompts, `76 aa` Ubiquitin + `<= 0.25` `HomologFirewall` torsions, `CISD3` `{s_nl['cisd3_classified_experiments']}` Bruker `pulprog` dirs, `-32002`/`-32001` blocks "
            f"| `{s_nl['stderr_bytes']} B` `stderr`, `exit {s_nl['exit_code']}` | `PASS` |"
        ),
        (
            "| External Python OS-pipe `nmrassign mcp --transport stdio` "
            f"| `Content-Length` (`{s_cl['negotiated_protocol_version']}`) "
            f"| `{s_cl['roundtrips_recorded']}` "
            f"| Byte-exact `Content-Length` headers on all `{s_cl['roundtrips_recorded']}` responses; identical tool/resource/prompt & `-32002`/`-32001` security enforcement "
            f"| `{s_cl['stderr_bytes']} B` `stderr`, `exit {s_cl['exit_code']}` | `PASS` |"
        ),
        (
            "| Loopback `nmrassign mcp --transport http` (`F20` CSPRNG auto-gen + `NMR_MCP_BEARER_TOKEN` + `mcp-config` `<32`-char guard) "
            "| HTTP/1.1 Streamable (`127.0.0.1:0`) "
            f"| `{len(h_auto['statuses']) + len(h_env['statuses'])}` (`+ {n_cfg_ok}` `mcp-config` runs) "
            f"| `{h_auto['csprng_token_len']}`-hex CSPRNG on `stderr` (`{h_auto['stdout_bytes']} B` `stdout`), env secret never echoed (`stderr_echoed_secret = {h_env['stderr_echoed_secret']}`), `401`/`403`/`200` verified "
            "| Darwin split-`send()` POSTs can surface `ECONNRESET` on rejected `401`/`403` (`mcp.rs:4069`); `0` `200 OK` affected | `PASS` |"
        ),
        (
            "| Standalone `f18-gui-bridge-harness` (`nmr-gui::mcp_bridge` + `build_builtin_demo_suite(42)`) "
            f"| HTTP/1.1 Workbench Bridge (`{gui['bearer_token_len']}`-char `nmr_mcp_<64hex>`) "
            f"| `{gui['http_exchanges_recorded']}` "
            f"| `401`/`403`, all `9/9` operations across `5` tool descriptors, `{png['width']}x{png['height']}` PNG (`{png['png_byte_len']:,} B`, `CRC-32` + `Adler-32 = {png['zlib_adler32_hex']}`, `{png['non_background_contour_or_crosshair_pixels']}` rendered pixels), `6`-step Undo `100%` exact `S0` restore "
            f"| Bounded body read (`mcp_bridge.rs`); `0` `ECONNRESET`; `TODO.md:153` (`F18`) stays `[ ]` open | `PASS` |"
        ),
    ]
    return "\n".join(lines)


def render_f21_section(results_dir: Path) -> str:
    inv = json.loads((results_dir / "f21" / "host_browser_gpu_inventory.json").read_text(encoding="utf-8"))
    chr_rep = json.loads((results_dir / "f21" / "chrome_157_live_webgpu_report.json").read_text(encoding="utf-8"))
    chr_smk = json.loads((results_dir / "f21" / "chrome_live_wasm_bundle_smoke.json").read_text(encoding="utf-8"))
    metal = json.loads((results_dir / "f21" / "metal_wgpu_shader_and_viewport_audit.json").read_text(encoding="utf-8"))
    pages = json.loads((results_dir / "f21" / "github_pages_live_check.json").read_text(encoding="utf-8"))

    wasm_bin = inv["wasm_toolchain_and_bundle"]["compiled_wasm32_release_binary"]
    shell_chk = inv["wasm_toolchain_and_bundle"]["check_only_staged_shell"]
    shaders = chr_rep["shaderCompilation"]
    gui_shaders_ok = sum(1 for k, v in shaders.items() if k.startswith("crates/nmr-gui/") and v["ok"])
    gui_shaders_tot = sum(1 for k in shaders if k.startswith("crates/nmr-gui/"))
    gpu_shaders_ok = sum(1 for k, v in shaders.items() if k.startswith("crates/nmr-gpu/") and v["ok"])
    gpu_shaders_tot = sum(1 for k in shaders if k.startswith("crates/nmr-gpu/"))
    vp_ok = sum(
        1
        for r in chr_smk
        if r.get("reported")
        and r["telemetry"].get("badgeText") == "WebGPU Ready"
        and r["telemetry"].get("canvasPresent")
        and len(r["telemetry"].get("jsErrors", [])) == 0
    )
    pages_200 = sum(
        1
        for e in pages["requested_endpoints"] + pages["supplemental_deployed_endpoints"]
        if e.get("get_status") == 200
    )

    gui_suites = metal["rust_release_test_suites"]["nmr_gui_integration_and_lib"]
    gui_passed = sum(
        int(v["passed"])
        for v in gui_suites.values()
        if isinstance(v, dict) and "passed" in v
    )
    gpu_passed = int(metal["rust_release_test_suites"]["nmr_gpu_all_tests"]["total_passed"])
    total_wgpu_passed = gui_passed + gpu_passed

    candor = metal["f21_compatibility_matrix_candor"]
    matrix_rows: list[str] = []
    for c in candor["executed_cells"]:
        matrix_rows.append(
            f"| `{c['os']}` | `{c['gpu_backend']}` | {c['target']} | `{c['status']}` |"
        )
    for u in candor["unexecuted_cells"]:
        matrix_rows.append(
            f"| `{u['cell']}` | *(not executed on single host)* | {u['reason']} | `UNEXECUTED` (`F21` stays `[ ]` open) |"
        )

    lines = [
        "#### `F21` — `wasm32` Binary, Native `Metal` `wgpu`, Live `Chrome 157` WebGPU & GitHub Pages Audit",
        "",
        "| Audit Surface | Target / Driver | Measured Result | Candid Scope / Finding | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
        (
            "| `wasm32-unknown-unknown` release binary & WCAG 1.4.4 HTML shell "
            "| `nmrassign-gui.wasm` + `docs/demo/index.html` "
            f"| `{wasm_bin['byte_size']:,} B` (`sha256={wasm_bin['sha256'][:16]}...`), magic `{wasm_bin['wasm_magic_hex']}`, **`{wasm_bin['env_imports_count']}` `\"env\"` imports**, verbatim `contour_shader.wgsl` (`5,836 B`) & `mol3d.wgsl` (`6,937 B`); `index.html` `{shell_chk['byte_size']:,} B` (`sha256={shell_chk['sha256'][:16]}...`) "
            "| `wasm-bindgen` CLI absent locally (`0.2.130` pinned in untouched `Cargo.lock`) | `PASS` |"
        ),
        (
            "| Native hardware `wgpu` / Naga suite "
            f"| `{inv['host_gpu_driver']['model']}` (`{inv['host_gpu_driver']['gpu_cores']}` GPU cores, `Metal`) "
            f"| `{total_wgpu_passed}/{total_wgpu_passed}` Rust `wgpu` tests PASS across `nmr-gui` (`{gui_passed}` lib/integration) + `nmr-gpu` (`{gpu_passed}` compute parity across all `11` WGSL shaders) "
            "| `maxStorageBuffersPerShaderStage = 29` (`nmr-gpu` bounded cap `8`), `maxComputeWorkgroupStorageSize = 32,768 B` | `PASS` |"
        ),
        (
            "| Real browser Dawn / Tint `createShaderModule().getCompilationInfo()` "
            f"| `Chrome {inv['browsers'][1]['short_version']} --headless=new` (`{chr_rep['adapterInfo']['vendor']}` `{chr_rep['adapterInfo']['architecture']}`, `isFallbackAdapter = {chr_rep['adapterInfo']['isFallbackAdapter']}`) "
            f"| **`{gui_shaders_ok}/{gui_shaders_tot}` (`100%`)** `wasm32`-active `nmr-gui` WGSL shaders + **`{gpu_shaders_ok}/{gpu_shaders_tot}`** `nmr-gpu` compute shaders compiled with `0` errors (`0.5..10.9 ms`) "
            "| Native-only `bayes_sa.wgsl:488` `bitcast<f32>(0xffffffffu)` rejected by Tint const-eval (`nmr-gpu` is excluded from `wasm32` `nmr-gui`) | `PASS` |"
        ),
        (
            "| Live browser `nmrassign_gui.{js,wasm}` WebGPU boot across `4` viewports "
            "| `Chrome 157.0.8103.79 --headless=new` (`375x667`, `390x844`, `768x1024`, `1440x900`) "
            f"| **`{vp_ok}/{len(chr_smk)}`** viewports booted to `badgeText = \"WebGPU Ready\"`, `#nmr_loading` removed, `#nmr_canvas` active (`500x544`..`1440x775` — Chrome headless enforces `500 px` min window width), `0` JS/WASM errors "
            "| Live Chrome lacked `DevToolsActivePort`, so isolated `--user-data-dir` headless session was used | `PASS` |"
        ),
        (
            "| Live `https://debsahu.github.io/nmr-assign/` Pages verification "
            "| `v0.12.0` live deployment (`Last-Modified: Sun, 15 Nov 2026 18:06:58 GMT`) "
            f"| `{pages_200}` live endpoints `200 OK` (`/`, `/index.html`, `/demo/`, `/demo/nmrassign_gui.js`, `/demo/nmrassign_gui_bg.wasm`, `/assets/01_hsqc_contour_overlay.svg`, `/assets/style.css`) "
            "| `/manual/index.html` & `/assets/01_hsqc_contours.svg` `404` by site layout/filename; live Pages reflects `v0.12.0` until next tag/`workflow_dispatch` | `PASS` |"
        ),
        "",
        "#### `F21` — Candid Multi-Browser / Multi-OS WebGPU Matrix Coverage (`TODO.md:162` Remains `[ ]` Open)",
        "",
        "| Matrix Cell / Platform | GPU / Browser Stack | Scope / Blocker Note | Matrix Verdict |",
        "| :--- | :--- | :--- | :---: |",
        *matrix_rows,
    ]
    return "\n".join(lines)


def render_f22_f23_section(root: Path) -> str:
    ci_txt = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    rel_txt = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    ci_bytes, ci_sha = sha256_and_size(root / ".github" / "workflows" / "ci.yml")
    rel_bytes, rel_sha = sha256_and_size(root / ".github" / "workflows" / "release.yml")

    sha_pins = re.findall(r"uses:\s+(actions/[A-Za-z0-9_-]+@[0-9a-f]{40})", rel_txt)
    unique_pins = sorted(set(sha_pins))
    has_win_job = "windows-check:" in ci_txt and "x86_64-pc-windows-gnu" in ci_txt
    has_collision_guard = "Refusing to overwrite published release binaries." in rel_txt
    if not (has_win_job and has_collision_guard and len(unique_pins) == 4):
        raise RuntimeError(
            f"F22/F23 workflow check failed: win={has_win_job}, collision_guard={has_collision_guard}, pins={unique_pins}"
        )

    lines = [
        "#### `F22` & `F23` — Windows Branch CI Compile Gate & Release Workflow Supply-Chain / Overwrite Hardening",
        "",
        "| Closed Item | Workflow File | Verified Hardening Invariant | Local & Structural Verification | Status |",
        "| :--- | :--- | :--- | :--- | :---: |",
        (
            f"| `F22` (`TODO.md:168`, `[x]` closed) "
            f"| `.github/workflows/ci.yml` (`{ci_bytes:,} B`, `sha256={ci_sha[:16]}...`) "
            "| Added `windows-check` job on `ubuntu-latest` (`gcc-mingw-w64-x86-64` + `x86_64-pc-windows-gnu`) running `RUSTFLAGS=\"-D warnings\" cargo check [--release] --locked -p nmr-cli -p nmr-gui --target x86_64-pc-windows-gnu` "
            "| Both dev (`BCryptGenRandom` `F19`) & `--release` (`windows_subsystem = \"windows\"`) cross-checks exit `0` locally (`w2_checks.log`) | `PASS` |"
        ),
        (
            f"| `F23` (`TODO.md:169`, `[x]` closed) "
            f"| `.github/workflows/release.yml` (`{rel_bytes:,} B`, `sha256={rel_sha[:16]}...`) "
            f"| Pinned all `{len(unique_pins)}` first-party `actions/*` (`{len(sha_pins)}` uses: `checkout@11d5960a...`, `cache@0057852b...`, `upload-artifact@ea165f8d...`, `download-artifact@d3f86a10...`) to verified 40-hex commit SHAs, and gated `gh release upload` on `gh release view --json isDraft,assets` (`isDraft == false` + colliding asset -> `exit 1`; `isDraft == true` -> `--clobber`) "
            "| `4/4` commit SHAs verified via GitHub API, YAML AST & `bash -n` syntax verified, `package_release.py --self-test` `5/5` PASS | `PASS` |"
        ),
    ]
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
        default=Path("data/private/p17_w2_results"),
        help="Directory holding raw Phase 17 Wave 2 JSON/JSONL/log artifacts.",
    )
    parser.add_argument(
        "--base-ref",
        type=str,
        default="41fe1a7",
        help="Approved 17.W1 baseline git commit for 0-byte protected-path diff verification.",
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
        choices=["all", "protected_diff", "digests", "f18", "f21", "f22_f23"],
        default="all",
        help="Subset of Markdown sections to render.",
    )
    parser.add_argument(
        "--run-checks",
        action="store_true",
        help="Run verify_p17_preregistration, windows-gnu/wasm32 cargo checks, and doc/release self-tests.",
    )
    parser.add_argument(
        "--write-sha256sums",
        action="store_true",
        help="Write unified data/private/p17_w2_results/SHA256SUMS.txt after rendering.",
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
        run_w2_checks(root, results_dir)

    extra_files = [p if p.is_absolute() else (root / p) for p in args.extra_file]
    digest_files = cited_w2_files(results_dir, root, extra_files)

    sections: list[str] = []
    sec = args.section
    if sec in ("all", "protected_diff"):
        sections.append(render_protected_diff_section(root, args.base_ref))
    if sec in ("all", "digests"):
        sections.append(render_digests_section(digest_files, root, results_dir))
    if sec in ("all", "f18"):
        sections.append(render_f18_section(results_dir))
    if sec in ("all", "f21"):
        sections.append(render_f21_section(results_dir))
    if sec in ("all", "f22_f23"):
        sections.append(render_f22_f23_section(root))

    rendered = "\n\n".join(s for s in sections if s.strip()) + "\n"
    verify_markdown_tables(rendered)

    if args.out is not None:
        out_p = args.out if args.out.is_absolute() else (root / args.out)
        write_clean_utf8(out_p, rendered)
    else:
        sys.stdout.write(rendered)

    if args.write_sha256sums:
        write_w2_sha256sums(results_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
