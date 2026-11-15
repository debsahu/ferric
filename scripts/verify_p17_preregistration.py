#!/usr/bin/env python3
"""Zero-dependency validator for Phase 17 single-shot empirical pre-registration.

Validates ``data/benchmark/p17_empirical_preregistration.json`` against:
1. Schema version, baseline tag (``v0.12.0``), protocol rules (solver/WGSL freeze
   with ``F12(a)`` pre-download exception, deterministic format-conversion algebra,
   strict fallback chain without tuning, candid single-shot reporting), and full
   completeness across all 8 empirical validation gates:
   ``F10``, ``F11``, ``F13a``, ``F13b``, ``F13c``, ``F14``, ``F16``, ``F17``.
2. Strict open-data license allowlist: ``CC0``, ``CC-BY-4.0``, ``Public Domain``.
3. 100% accession disjointness against:
   - ``data/benchmark/real_manifest.tsv`` (40 backbone FID benchmark entries),
   - ``data/benchmark/manifest.tsv`` (12 Phase-1 benchmark entries),
   - ``data/priors/PROVENANCE.md`` & ``data/priors/bmrb_priors.json`` (21 prior entries),
   - ``excluded_contaminated_entries`` (``BMRB:53960`` blind-firewall exclusion).
4. Zero ``F10``-``F17`` download directories or target archive files present in
   ``data/private/`` (checked in both the active checkout and any linked git main
   worktree) at pre-registration time.

Provenance: Original clean-room Python 3 standard-library validator (MIT License).
Zero network calls are performed by this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlsplit

EXPECTED_GATES: Tuple[str, ...] = (
    "F10",
    "F11",
    "F13a",
    "F13b",
    "F13c",
    "F14",
    "F16",
    "F17",
)

STRICT_LICENSE_ALLOWLIST: Set[str] = {
    "CC0",
    "CC-BY-4.0",
    "Public Domain",
}

REQUIRED_PROTECTED_CORE_FILES: Tuple[str, ...] = (
    "crates/nmr-core/src/relax_io.rs",
    "crates/nmr-core/src/relax_fit.rs",
    "crates/nmr-core/src/modelfree.rs",
    "crates/nmr-core/src/cest.rs",
    "crates/nmr-core/src/cpmg.rs",
    "crates/nmr-core/src/csp.rs",
    "crates/nmr-core/src/nmr1d.rs",
    "crates/nmr-core/src/multiplet.rs",
    "crates/nmr-core/src/dosy.rs",
    "crates/nmr-core/src/metabolomics.rs",
)

ALLOWED_PRE_DOWNLOAD_EXCEPTION_FILES: Set[str] = {
    "crates/nmr-gpu/src/shaders/relax_mc.wgsl",
}

MANDATORY_EXCLUDED_BMRB_IDS: Set[str] = {"53960"}

ALLOWED_PRIMARY_ROLES: Set[str] = {"primary", "co_primary"}
ALLOWED_BACKUP_ROLES: Tuple[str, ...] = (
    "backup_1",
    "backup_2",
    "backup_3",
    "backup_4",
    "backup_5",
)

# Directory or file prefixes in data/private/ that indicate F10-F17 downloads.
# Note: `p17_prereg_candidates.md` (metadata-only HEAD catalog) and `p17_w1_results` / `p17_w2_results`
# (pre-download Wave 1 & Wave 2 benchmark/test log directories, whose children are recursively scanned
# below against `download_basenames`, `FORBIDDEN_PRIVATE_GATE_PATTERNS`, and `specific_token_re`) are allowed.
ALLOWED_P17_PRIVATE_RESULT_DIRS: Set[str] = {"p17_w1_results", "p17_w2_results"}

FORBIDDEN_PRIVATE_GATE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|[_\-])f1(?:0|1|3|3a|3b|3c|4|6|7)(?:$|[_\-])", re.IGNORECASE),
    re.compile(r"^p17[_\-](?!(?:prereg_candidates\.md|w[12]_results)$)", re.IGNORECASE),
    re.compile(r"(?:^|[_\-])mtbls\d*", re.IGNORECASE),
    re.compile(r"(?:^|[_\-])bmrbig\d+", re.IGNORECASE),
    re.compile(r"(?:^|[_\-])bmse\d+", re.IGNORECASE),
    re.compile(r"(?:^|[_\-])zenodo[_\-]?\d+", re.IGNORECASE),
    re.compile(r"(?:^|[_\-])mendeley", re.IGNORECASE),
)

# Generic URL basenames that are too broad on their own unless paired with a path,
# handled explicitly alongside exact archive basenames.
GENERIC_BASENAMES: Set[str] = {
    "fid",
    "ser",
    "acqus",
    "acqu2s",
    "acqu3s",
    "procpar",
    "pulseprogram",
    "vdlist",
    "vclist",
    "content",
    "file_downloaded",
    "readme.txt",
}


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _normalize_bmrb_token(raw: str) -> Optional[str]:
    """Return canonical numeric/named BMRB ID when ``raw`` refers to a BMRB polymer entry."""
    s = raw.strip()
    if not s:
        return None
    low = s.lower()
    if low == "ubiquitin_local":
        return "ubiquitin_local"
    # Exclude BMRbig (`bmrbig138` / `BMRBIG:138`) and BMSE (`bmse000036` / `BMSE:...`)
    if low.startswith(("bmrbig", "bmse", "zenodo:", "mendeley:", "metabolights:", "pdb:")):
        return None
    m = re.match(r"^(?:bmrb:|bmrb|bmr)?(\d+)$", low)
    if m:
        return m.group(1)
    return None


def _extract_real_manifest_bmrb_ids(path: Path, errors: List[str]) -> Set[str]:
    ids: Set[str] = set()
    if not path.is_file():
        errors.append(f"missing benchmark file: {path}")
        return ids
    header_seen = False
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cols = stripped.split("\t")
        if not header_seen:
            header_seen = True
            if cols[0] != "bmrb_id":
                errors.append(f"{path}:{line_no}: expected `bmrb_id` header, got {cols[0]!r}")
            continue
        norm = _normalize_bmrb_token(cols[0])
        if norm is None:
            errors.append(f"{path}:{line_no}: could not parse BMRB ID {cols[0]!r}")
        else:
            ids.add(norm)
    if len(ids) < 40:
        errors.append(f"{path}: expected >= 40 benchmark entries, parsed {len(ids)}")
    return ids


def _extract_phase1_manifest_bmrb_ids(path: Path, errors: List[str]) -> Set[str]:
    ids: Set[str] = set()
    if not path.is_file():
        errors.append(f"missing Phase-1 benchmark manifest: {path}")
        return ids
    header_seen = False
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cols = stripped.split("\t")
        if not header_seen:
            header_seen = True
            if cols[0] != "bmrb_id":
                errors.append(f"{path}:{line_no}: expected `bmrb_id` header, got {cols[0]!r}")
            continue
        norm = _normalize_bmrb_token(cols[0])
        if norm is None:
            errors.append(f"{path}:{line_no}: could not parse BMRB ID {cols[0]!r}")
        else:
            ids.add(norm)
    if len(ids) < 12:
        errors.append(f"{path}: expected >= 12 Phase-1 benchmark entries, parsed {len(ids)}")
    return ids


def _extract_priors_bmrb_ids(
    priors_json_path: Path, provenance_md_path: Path, errors: List[str]
) -> Set[str]:
    ids: Set[str] = set()
    if not priors_json_path.is_file():
        errors.append(f"missing priors JSON: {priors_json_path}")
    else:
        try:
            priors_obj = json.loads(priors_json_path.read_text(encoding="utf-8"))
            if not isinstance(priors_obj, dict) or len(priors_obj) < 20:
                errors.append(
                    f"{priors_json_path}: expected >= 20 amino-acid entries, got {type(priors_obj).__name__}"
                )
            for m in re.finditer(r"\bbmr(\d+)\b", json.dumps(priors_obj), flags=re.IGNORECASE):
                ids.add(m.group(1))
        except json.JSONDecodeError as exc:
            errors.append(f"{priors_json_path}: invalid JSON: {exc}")

    if not provenance_md_path.is_file():
        errors.append(f"missing priors provenance doc: {provenance_md_path}")
        return ids

    in_training_section = False
    for raw in provenance_md_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("## Training BMRB Entries"):
            in_training_section = True
            continue
        if in_training_section and stripped.startswith("## "):
            break
        if in_training_section and stripped.startswith("|"):
            m = re.search(r"`bmr(\d+)`", stripped, flags=re.IGNORECASE)
            if m:
                ids.add(m.group(1))

    if len(ids) < 21:
        errors.append(
            f"{provenance_md_path}: expected >= 21 prior training BMRB entries, parsed {len(ids)}"
        )
    return ids


def _discover_private_dirs(repo_root: Path) -> List[Path]:
    """Return existing ``data/private`` directories in ``repo_root`` and any linked main worktree."""
    found: List[Path] = []
    seen_resolved: Set[Path] = set()

    def _add_candidate(cand: Path) -> None:
        if cand.exists() and cand.is_dir():
            resolved = cand.resolve()
            if resolved not in seen_resolved:
                seen_resolved.add(resolved)
                found.append(cand)

    _add_candidate(repo_root / "data" / "private")

    dot_git = repo_root / ".git"
    if dot_git.is_file():
        # Linked git worktree: `.git` contains `gitdir: <main_repo>/.git/worktrees/<name>`
        try:
            first = dot_git.read_text(encoding="utf-8").splitlines()[0].strip()
            if first.startswith("gitdir:"):
                gitdir_raw = first.split(":", 1)[1].strip()
                gitdir_path = (repo_root / gitdir_raw).resolve()
                if gitdir_path.parent.name == "worktrees" and gitdir_path.parent.parent.name == ".git":
                    main_repo = gitdir_path.parent.parent.parent
                    _add_candidate(main_repo / "data" / "private")
        except OSError:
            pass

    return found


def _validate_dataset_entry(
    gate_id: str,
    ds: Any,
    idx: int,
    is_primary: bool,
    errors: List[str],
    candidate_accessions: List[Tuple[str, str]],
    download_tokens: Set[str],
    download_basenames: Set[str],
) -> Optional[str]:
    label = f"gates.{gate_id}.{'primary_datasets' if is_primary else 'backup_datasets'}[{idx}]"
    if not isinstance(ds, dict):
        errors.append(f"{label}: dataset entry must be a JSON object")
        return None

    role = ds.get("role")
    if not isinstance(role, str) or not role.strip():
        errors.append(f"{label}: missing non-empty string `role`")
        role_str = ""
    else:
        role_str = role.strip()
        if is_primary and role_str not in ALLOWED_PRIMARY_ROLES:
            errors.append(
                f"{label}: invalid primary role {role_str!r} (expected one of {sorted(ALLOWED_PRIMARY_ROLES)})"
            )
        if not is_primary and role_str not in ALLOWED_BACKUP_ROLES:
            errors.append(
                f"{label}: invalid backup role {role_str!r} (expected one of {list(ALLOWED_BACKUP_ROLES)})"
            )

    acc = ds.get("accession_id")
    if not isinstance(acc, str) or ":" not in acc or not acc.strip():
        errors.append(f"{label}: `accession_id` must be `<NAMESPACE>:<ID>`, got {acc!r}")
    else:
        acc_clean = acc.strip()
        ns, bare_id = acc_clean.split(":", 1)
        if not ns or not bare_id:
            errors.append(f"{label}: malformed `accession_id` {acc_clean!r}")
        else:
            candidate_accessions.append((gate_id, acc_clean))
            ns_low = ns.lower()
            id_low = bare_id.lower()
            if ns_low == "bmrb":
                download_tokens.add(f"bmr{id_low}")
                download_tokens.add(f"bmrb{id_low}")
                download_tokens.add(id_low)
            elif ns_low == "bmrbig":
                download_tokens.add(f"bmrbig{id_low}")
            elif ns_low == "bmse":
                download_tokens.add(id_low)
            elif ns_low in ("zenodo", "mendeley", "metabolights"):
                download_tokens.add(id_low)

    lic = ds.get("license")
    if not isinstance(lic, str) or lic not in STRICT_LICENSE_ALLOWLIST:
        errors.append(
            f"{label}: license {lic!r} violates strict allowlist {sorted(STRICT_LICENSE_ALLOWLIST)}"
        )

    doi = ds.get("citation_doi")
    if not isinstance(doi, str) or not re.match(r"^10\.\d{4,9}/\S+$", doi.strip()):
        errors.append(f"{label}: `citation_doi` must be a valid `10.xxxx/...` DOI, got {doi!r}")

    title = ds.get("title")
    if not isinstance(title, str) or len(title.strip()) < 8:
        errors.append(f"{label}: missing descriptive `title`")

    files = ds.get("files")
    if not isinstance(files, list) or not files:
        errors.append(f"{label}: `files` must be a non-empty list")
    else:
        for f_idx, f_obj in enumerate(files):
            flabel = f"{label}.files[{f_idx}]"
            if not isinstance(f_obj, dict):
                errors.append(f"{flabel}: file entry must be an object")
                continue
            fname = f_obj.get("name")
            furl = f_obj.get("url")
            fbytes = f_obj.get("head_verified_bytes")
            if not isinstance(fname, str) or not fname.strip():
                errors.append(f"{flabel}: missing non-empty `name`")
            else:
                base_name = Path(fname.strip().rstrip("/")).name
                if base_name and base_name.lower() not in GENERIC_BASENAMES:
                    # Exclude local PDB references like 1D3Z.pdb / 1UBQ.pdb from private-dir false alarms,
                    # while flagging any NMR archive/saveframe filename if downloaded into data/private/.
                    if not base_name.lower().endswith(".pdb"):
                        download_basenames.add(base_name.lower())
            if not isinstance(furl, str) or not furl.startswith("https://"):
                errors.append(f"{flabel}: `url` must start with `https://`, got {furl!r}")
            else:
                parsed = urlsplit(furl)
                if not parsed.netloc or not parsed.path:
                    errors.append(f"{flabel}: malformed URL {furl!r}")
                url_leaf = Path(unquote(parsed.path).rstrip("/")).name
                if (
                    url_leaf
                    and url_leaf.lower() not in GENERIC_BASENAMES
                    and not url_leaf.lower().endswith(".pdb")
                ):
                    download_basenames.add(url_leaf.lower())
            if fbytes is not None and (not _is_strict_int(fbytes) or fbytes <= 0):
                errors.append(
                    f"{flabel}: `head_verified_bytes` must be a positive integer or null, got {fbytes!r}"
                )

    return role_str


def validate_preregistration(
    repo_root: Path,
    manifest_path: Optional[Path] = None,
    extra_private_dirs: Optional[Sequence[Path]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Run all Phase 17 pre-registration checks; return ``(errors, summary_stats)``."""
    errors: List[str] = []
    stats: Dict[str, Any] = {}

    mpath = manifest_path or (repo_root / "data" / "benchmark" / "p17_empirical_preregistration.json")
    if not mpath.is_file():
        return [f"missing pre-registration manifest: {mpath}"], stats

    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{mpath}: invalid JSON: {exc}"], stats

    if not isinstance(manifest, dict):
        return [f"{mpath}: top-level JSON value must be an object"], stats

    # 1. Top-level metadata & protocol invariants
    if not _is_strict_int(manifest.get("schema_version")) or manifest["schema_version"] != 1:
        errors.append(f"schema_version must be integer 1, got {manifest.get('schema_version')!r}")
    if manifest.get("phase") != "17":
        errors.append(f"phase must be '17', got {manifest.get('phase')!r}")
    if manifest.get("frozen_solver_baseline_tag") != "v0.12.0":
        errors.append(
            f"frozen_solver_baseline_tag must be 'v0.12.0', got {manifest.get('frozen_solver_baseline_tag')!r}"
        )
    if manifest.get("zero_bytes_downloaded_at_registration") is not True:
        errors.append("zero_bytes_downloaded_at_registration must be strictly `true`")

    manifest_allowlist = manifest.get("license_allowlist")
    if not isinstance(manifest_allowlist, list) or not manifest_allowlist:
        errors.append("license_allowlist must be a non-empty list")
    elif not set(manifest_allowlist).issubset(STRICT_LICENSE_ALLOWLIST):
        errors.append(
            f"license_allowlist {manifest_allowlist!r} contains entries outside {sorted(STRICT_LICENSE_ALLOWLIST)}"
        )

    expected_disjoint_files = [
        "data/benchmark/real_manifest.tsv",
        "data/benchmark/manifest.tsv",
        "data/priors/bmrb_priors.json",
    ]
    if manifest.get("disjointness_verified_against") != expected_disjoint_files:
        errors.append(
            f"disjointness_verified_against must equal {expected_disjoint_files!r}, "
            f"got {manifest.get('disjointness_verified_against')!r}"
        )

    excluded_raw = manifest.get("excluded_contaminated_entries")
    excluded_bmrb_ids: Set[str] = set()
    if not isinstance(excluded_raw, list) or not excluded_raw:
        errors.append("excluded_contaminated_entries must be a non-empty list containing 'BMRB:53960'")
    else:
        for item in excluded_raw:
            if not isinstance(item, str):
                errors.append(f"excluded_contaminated_entries entry must be a string, got {item!r}")
                continue
            norm = _normalize_bmrb_token(item)
            if norm:
                excluded_bmrb_ids.add(norm)
        if "BMRB:53960" not in excluded_raw or not MANDATORY_EXCLUDED_BMRB_IDS.issubset(excluded_bmrb_ids):
            errors.append("excluded_contaminated_entries must explicitly include 'BMRB:53960'")

    # 2. Protocol rules (P17-R1 .. P17-R4) & frozen solver file SHA-256 verification
    rules = manifest.get("protocol_rules")
    if not isinstance(rules, dict):
        errors.append("protocol_rules must be a JSON object")
    else:
        r1 = rules.get("rule_1_solver_and_shader_freeze")
        if not isinstance(r1, dict) or r1.get("enforced") is not True:
            errors.append("protocol_rules.rule_1_solver_and_shader_freeze must be enforced=true")
        else:
            core_files = r1.get("protected_core_files")
            wgsl_files = r1.get("protected_wgsl_shaders")
            exceptions = r1.get("allowed_pre_download_w1_exceptions")
            sha_map = r1.get("v0120_baseline_sha256")
            w1_sha_map = r1.get("w1_enforced_sha256")

            if not isinstance(core_files, list) or set(core_files) != set(REQUIRED_PROTECTED_CORE_FILES):
                errors.append(
                    "rule_1_solver_and_shader_freeze.protected_core_files must match exact 10 solver modules"
                )

            shader_dir = repo_root / "crates" / "nmr-gpu" / "src" / "shaders"
            disk_wgsl = (
                sorted(p.relative_to(repo_root).as_posix() for p in shader_dir.glob("*.wgsl"))
                if shader_dir.is_dir()
                else []
            )
            if not isinstance(wgsl_files, list) or not wgsl_files:
                errors.append("rule_1_solver_and_shader_freeze.protected_wgsl_shaders must be non-empty")
            elif disk_wgsl and set(wgsl_files) != set(disk_wgsl):
                errors.append(
                    f"protected_wgsl_shaders mismatch vs disk shaders: manifest={sorted(wgsl_files)} disk={disk_wgsl}"
                )

            pinned_w1_exceptions: Dict[str, Tuple[int, str]] = {}
            if not isinstance(exceptions, list):
                errors.append("allowed_pre_download_w1_exceptions must be a list")
            else:
                for exc_obj in exceptions:
                    if not isinstance(exc_obj, dict) or not isinstance(exc_obj.get("file"), str):
                        errors.append("malformed entry in allowed_pre_download_w1_exceptions")
                        continue
                    f_rel = exc_obj["file"]
                    if f_rel not in ALLOWED_PRE_DOWNLOAD_EXCEPTION_FILES:
                        errors.append(
                            f"unauthorized solver freeze exception {f_rel!r} (only F12(a) relax_mc.wgsl allowed)"
                        )
                    pin_bytes = exc_obj.get("pinned_w1_bytes")
                    pin_sha = exc_obj.get("pinned_w1_sha256")
                    if not _is_strict_int(pin_bytes) or pin_bytes <= 0 or not isinstance(pin_sha, str) or len(pin_sha) != 64:
                        errors.append(
                            f"allowed_pre_download_w1_exceptions entry for {f_rel!r} missing pinned_w1_bytes / 64-hex pinned_w1_sha256"
                        )
                    else:
                        pinned_w1_exceptions[f_rel] = (pin_bytes, pin_sha)

            if not isinstance(sha_map, dict):
                errors.append("rule_1_solver_and_shader_freeze.v0120_baseline_sha256 must be an object")
            if not isinstance(w1_sha_map, dict):
                errors.append("rule_1_solver_and_shader_freeze.w1_enforced_sha256 must be an object")
            elif isinstance(sha_map, dict):
                all_protected = list(REQUIRED_PROTECTED_CORE_FILES) + (
                    wgsl_files if isinstance(wgsl_files, list) else []
                )
                for rel_path in all_protected:
                    v0120_rec = sha_map.get(rel_path)
                    if not isinstance(v0120_rec, dict) or not isinstance(v0120_rec.get("sha256"), str):
                        errors.append(f"missing v0120_baseline_sha256 digest for {rel_path}")
                        continue
                    rec = w1_sha_map.get(rel_path)
                    if not isinstance(rec, dict) or not isinstance(rec.get("sha256"), str) or not _is_strict_int(rec.get("bytes")):
                        errors.append(f"missing w1_enforced_sha256 digest/bytes for {rel_path}")
                        continue
                    if rel_path in pinned_w1_exceptions:
                        exp_bytes, exp_sha = pinned_w1_exceptions[rel_path]
                        if rec["sha256"] != exp_sha or rec["bytes"] != exp_bytes:
                            errors.append(
                                f"w1_enforced_sha256 mismatch vs pinned_w1_sha256 for {rel_path}"
                            )
                    else:
                        if rec["sha256"] != v0120_rec["sha256"] or rec.get("bytes") != v0120_rec.get("bytes"):
                            errors.append(
                                f"non-exception file {rel_path} has w1_enforced_sha256 != v0120_baseline_sha256"
                            )
                    abs_path = repo_root / rel_path
                    if not abs_path.is_file():
                        errors.append(f"protected solver/shader file missing on disk: {rel_path}")
                        continue
                    actual_bytes = abs_path.stat().st_size
                    actual_sha = _sha256_file(abs_path)
                    if actual_sha != rec["sha256"] or actual_bytes != rec["bytes"]:
                        errors.append(
                            f"protected solver/shader file modified vs 17.W1 pinned freeze: {rel_path} "
                            f"(expected {rec['sha256']} [{rec['bytes']} B], got {actual_sha} [{actual_bytes} B])"
                        )

        r2 = rules.get("rule_2_a_priori_deterministic_format_conversion")
        if (
            not isinstance(r2, dict)
            or r2.get("enforced") is not True
            or r2.get("zero_physics_or_parameter_tuning") is not True
            or not isinstance(r2.get("fixed_conversions"), list)
            or len(r2["fixed_conversions"]) < 3
        ):
            errors.append(
                "protocol_rules.rule_2_a_priori_deterministic_format_conversion is incomplete"
            )
        else:
            conv_ids = {
                c.get("id") for c in r2["fixed_conversions"] if isinstance(c, dict)
            }
            for req_id in (
                "relaxation_time_to_rate",
                "bmrb6470_dual_field_saveframe_disambiguation",
            ):
                if req_id not in conv_ids:
                    errors.append(f"rule_2 missing required fixed conversion {req_id!r}")

        r3 = rules.get("rule_3_strict_fallback_order_without_tuning")
        if (
            not isinstance(r3, dict)
            or r3.get("enforced") is not True
            or r3.get("zero_parameter_tuning_on_fallback") is not True
            or not isinstance(r3.get("fallback_chain_order"), list)
            or r3["fallback_chain_order"][:3] != ["primary", "co_primary", "backup_1"]
        ):
            errors.append("protocol_rules.rule_3_strict_fallback_order_without_tuning is invalid")

        r4 = rules.get("rule_4_candid_single_shot_reporting")
        if (
            not isinstance(r4, dict)
            or r4.get("enforced") is not True
            or r4.get("single_shot_max_runs_per_dataset") != 1
            or r4.get("report_both_agreements_and_discrepancies") is not True
        ):
            errors.append("protocol_rules.rule_4_candid_single_shot_reporting is invalid")

    # 3. Validate all 8 gates
    todo_text = ""
    todo_path = repo_root / "TODO.md"
    if todo_path.is_file():
        todo_lines = todo_path.read_text(encoding="utf-8").splitlines()
    else:
        todo_lines = []
        errors.append(f"missing TODO.md at {todo_path}")

    gates = manifest.get("gates")
    candidate_accessions: List[Tuple[str, str]] = []
    download_tokens: Set[str] = {"53960", "bmr53960", "bmrb53960"}
    download_basenames: Set[str] = set()
    head_verified_file_count = 0
    total_file_urls = 0

    if not isinstance(gates, dict):
        errors.append("`gates` must be a JSON object")
    else:
        actual_keys = tuple(gates.keys())
        if set(actual_keys) != set(EXPECTED_GATES):
            errors.append(
                f"`gates` must contain exact keys {list(EXPECTED_GATES)}, got {list(actual_keys)}"
            )
        for gate_id in EXPECTED_GATES:
            g = gates.get(gate_id)
            if not isinstance(g, dict):
                errors.append(f"gates.{gate_id}: missing or non-object gate entry")
                continue

            todo_item = g.get("todo_item")
            if not isinstance(todo_item, str) or not re.match(r"^TODO\.md:\d+$", todo_item):
                errors.append(f"gates.{gate_id}.todo_item must match `TODO.md:<line>`, got {todo_item!r}")
            elif todo_lines:
                line_num = int(todo_item.split(":", 1)[1])
                if line_num < 1 or line_num > len(todo_lines):
                    errors.append(
                        f"gates.{gate_id}.todo_item {todo_item!r} out of range (1..{len(todo_lines)})"
                    )
                else:
                    base_gate = gate_id[:3] if gate_id.startswith("F13") else gate_id
                    if f"`{base_gate}`" not in todo_lines[line_num - 1]:
                        errors.append(
                            f"gates.{gate_id}.todo_item {todo_item} does not reference `{base_gate}`"
                        )

            prim = g.get("primary_datasets")
            if not isinstance(prim, list) or not prim:
                errors.append(f"gates.{gate_id}.primary_datasets must be a non-empty list")
            else:
                roles = []
                for idx, ds in enumerate(prim):
                    r = _validate_dataset_entry(
                        gate_id,
                        ds,
                        idx,
                        True,
                        errors,
                        candidate_accessions,
                        download_tokens,
                        download_basenames,
                    )
                    if r:
                        roles.append(r)
                    if isinstance(ds, dict) and isinstance(ds.get("files"), list):
                        for f in ds["files"]:
                            if isinstance(f, dict):
                                total_file_urls += 1
                                if _is_strict_int(f.get("head_verified_bytes")):
                                    head_verified_file_count += 1
                if not roles or roles[0] != "primary":
                    errors.append(f"gates.{gate_id}.primary_datasets[0].role must be 'primary'")

            bkp = g.get("backup_datasets")
            if not isinstance(bkp, list) or not bkp:
                errors.append(f"gates.{gate_id}.backup_datasets must be a non-empty list")
            else:
                expected_bkp_roles = list(ALLOWED_BACKUP_ROLES[: len(bkp)])
                actual_bkp_roles: List[str] = []
                for idx, ds in enumerate(bkp):
                    r = _validate_dataset_entry(
                        gate_id,
                        ds,
                        idx,
                        False,
                        errors,
                        candidate_accessions,
                        download_tokens,
                        download_basenames,
                    )
                    if r:
                        actual_bkp_roles.append(r)
                    if isinstance(ds, dict) and isinstance(ds.get("files"), list):
                        for f in ds["files"]:
                            if isinstance(f, dict):
                                total_file_urls += 1
                                if _is_strict_int(f.get("head_verified_bytes")):
                                    head_verified_file_count += 1
                if actual_bkp_roles and actual_bkp_roles != expected_bkp_roles:
                    errors.append(
                        f"gates.{gate_id}.backup_datasets roles must be strictly ordered {expected_bkp_roles}, "
                        f"got {actual_bkp_roles}"
                    )

            cli = g.get("exact_cli_command_template")
            if not isinstance(cli, str) or not cli.strip().startswith("nmrassign "):
                errors.append(
                    f"gates.{gate_id}.exact_cli_command_template must start with `nmrassign `"
                )

            adapter = g.get("deterministic_format_adapter_spec")
            if (
                not isinstance(adapter, dict)
                or not isinstance(adapter.get("required"), bool)
                or not isinstance(adapter.get("adapter_kind"), str)
                or not adapter["adapter_kind"].strip()
                or not isinstance(adapter.get("rules"), list)
                or not adapter["rules"]
                or not all(isinstance(s, str) and s.strip() for s in adapter["rules"])
            ):
                errors.append(f"gates.{gate_id}.deterministic_format_adapter_spec is invalid")

            metrics = g.get("evaluation_metrics_reported")
            if (
                not isinstance(metrics, list)
                or not metrics
                or not all(isinstance(m, str) and len(m.strip()) >= 8 for m in metrics)
            ):
                errors.append(f"gates.{gate_id}.evaluation_metrics_reported must be a non-empty string list")

        # Gate-specific coverage invariants
        f14 = gates.get("F14")
        if isinstance(f14, dict):
            f14_cli = str(f14.get("exact_cli_command_template", "")).lower()
            for req_tok in ("1h", "13c", "dept90", "dept135", "19f", "31p", "qnmr"):
                if req_tok not in f14_cli:
                    errors.append(f"gates.F14.exact_cli_command_template missing {req_tok!r} coverage")

        f16 = gates.get("F16")
        if isinstance(f16, dict):
            f16_accs = {
                ds.get("accession_id")
                for ds in f16.get("primary_datasets", [])
                if isinstance(ds, dict)
            }
            if not {"ZENODO:12170718", "MENDELEY:d7bdxz9hsk"}.issubset(f16_accs):
                errors.append(
                    "gates.F16.primary_datasets must include both Bruker (ZENODO:12170718) and Varian (MENDELEY:d7bdxz9hsk) DOSY"
                )

        f17 = gates.get("F17")
        if isinstance(f17, dict):
            f17_accs = {
                ds.get("accession_id")
                for ds in f17.get("primary_datasets", [])
                if isinstance(ds, dict)
            }
            if "METABOLIGHTS:MTBLS1" not in f17_accs:
                errors.append("gates.F17.primary_datasets must include METABOLIGHTS:MTBLS1")
            f17_cli = str(f17.get("exact_cli_command_template", ""))
            for subcmd in ("metabolomics bin", "metabolomics pca", "metabolomics oplsda", "metabolomics stocsy"):
                if subcmd not in f17_cli:
                    errors.append(f"gates.F17.exact_cli_command_template missing `{subcmd}`")

    # 4. 100% Accession Disjointness Verification
    real_bmrb_ids = _extract_real_manifest_bmrb_ids(
        repo_root / "data" / "benchmark" / "real_manifest.tsv", errors
    )
    phase1_bmrb_ids = _extract_phase1_manifest_bmrb_ids(
        repo_root / "data" / "benchmark" / "manifest.tsv", errors
    )
    prior_bmrb_ids = _extract_priors_bmrb_ids(
        repo_root / "data" / "priors" / "bmrb_priors.json",
        repo_root / "data" / "priors" / "PROVENANCE.md",
        errors,
    )
    contaminated_ids = MANDATORY_EXCLUDED_BMRB_IDS | excluded_bmrb_ids

    for gate_id, acc_str in candidate_accessions:
        norm_bmrb = _normalize_bmrb_token(acc_str)
        if norm_bmrb is not None:
            if norm_bmrb in real_bmrb_ids:
                errors.append(
                    f"disjointness violation in {gate_id}: {acc_str} overlaps data/benchmark/real_manifest.tsv ({norm_bmrb})"
                )
            if norm_bmrb in phase1_bmrb_ids:
                errors.append(
                    f"disjointness violation in {gate_id}: {acc_str} overlaps data/benchmark/manifest.tsv ({norm_bmrb})"
                )
            if norm_bmrb in prior_bmrb_ids:
                errors.append(
                    f"disjointness violation in {gate_id}: {acc_str} overlaps data/priors/PROVENANCE.md ({norm_bmrb})"
                )
            if norm_bmrb in contaminated_ids:
                errors.append(
                    f"blind-firewall violation in {gate_id}: {acc_str} is in excluded_contaminated_entries ({norm_bmrb})"
                )

    # 5. Zero F10-F17 Download Directories / Files in data/private/
    private_dirs = _discover_private_dirs(repo_root)
    if extra_private_dirs:
        for p in extra_private_dirs:
            if p.is_dir() and p not in private_dirs:
                private_dirs.append(p)

    # Build strict token regexes for candidate BMRB/Zenodo/BMSE/BMRbig/Mendeley/MTBLS IDs
    # Only multi-digit (>= 4 digits) or prefixed tokens (`bmrbig138`, `bmse000036`, `d7bdxz9hsk`, `mtbls1`)
    # are matched across general `data/private/` names to avoid false hits on short numbers.
    specific_tokens = sorted(
        tok
        for tok in download_tokens
        if len(tok) >= 4 or not tok.isdigit()
    )
    specific_token_re = (
        re.compile(
            r"(?:^|[^a-z0-9])(?:" + "|".join(re.escape(t) for t in specific_tokens) + r")(?:$|[^a-z0-9])",
            re.IGNORECASE,
        )
        if specific_tokens
        else None
    )
    candidate_bmrb_numeric = {
        _normalize_bmrb_token(acc)
        for _, acc in candidate_accessions
        if _normalize_bmrb_token(acc) is not None
    } | contaminated_ids

    def _check_private_entry(entry_path: Path) -> None:
        ename = entry_path.name
        elow = ename.lower()
        if elow in download_basenames:
            errors.append(
                f"pre-registration zero-download violation: target file `{entry_path}` exists in data/private/"
            )
            return
        if any(pat.search(ename) for pat in FORBIDDEN_PRIVATE_GATE_PATTERNS):
            errors.append(
                f"pre-registration zero-download violation: F10-F17 entry `{entry_path}` exists in data/private/"
            )
            return
        if specific_token_re is not None and specific_token_re.search(ename):
            errors.append(
                f"pre-registration zero-download violation: candidate accession `{entry_path}` exists in data/private/"
            )
            return

    for priv_dir in private_dirs:
        for child in sorted(priv_dir.iterdir()):
            name = child.name
            if name in (".gitkeep", "p17_prereg_candidates.md"):
                continue
            if name in ALLOWED_P17_PRIVATE_RESULT_DIRS and child.is_dir():
                for nested in sorted(child.rglob("*")):
                    _check_private_entry(nested)
                continue
            _check_private_entry(child)

        bmrb_subdir = priv_dir / "bmrb"
        if bmrb_subdir.is_dir():
            for sub in sorted(bmrb_subdir.iterdir()):
                norm_sub = _normalize_bmrb_token(sub.name)
                if norm_sub is not None and norm_sub in candidate_bmrb_numeric:
                    errors.append(
                        f"pre-registration zero-download violation: BMRB directory `{sub}` exists in data/private/bmrb/"
                    )

    unique_accessions = sorted({acc for _, acc in candidate_accessions})
    stats = {
        "manifest_path": str(mpath),
        "manifest_sha256": _sha256_file(mpath),
        "gates_verified": len(EXPECTED_GATES),
        "unique_candidate_accessions": len(unique_accessions),
        "total_dataset_entries": len(candidate_accessions),
        "total_file_urls": total_file_urls,
        "head_verified_file_urls": head_verified_file_count,
        "disjoint_real_manifest_count": len(real_bmrb_ids),
        "disjoint_phase1_manifest_count": len(phase1_bmrb_ids),
        "disjoint_priors_count": len(prior_bmrb_ids),
        "private_dirs_checked": [str(p) for p in private_dirs],
    }
    return errors, stats


def _build_minimal_fixture_repo(tmp_root: Path, real_repo: Path) -> Tuple[Path, Dict[str, Any]]:
    """Create an isolated temporary repo layout mirroring the real manifest & benchmark tables."""
    for rel in (
        "TODO.md",
        "data/benchmark/p17_empirical_preregistration.json",
        "data/benchmark/real_manifest.tsv",
        "data/benchmark/manifest.tsv",
        "data/priors/bmrb_priors.json",
        "data/priors/PROVENANCE.md",
    ):
        src = real_repo / rel
        dst = tmp_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for rel in REQUIRED_PROTECTED_CORE_FILES:
        src = real_repo / rel
        dst = tmp_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for src in sorted((real_repo / "crates" / "nmr-gpu" / "src" / "shaders").glob("*.wgsl")):
        rel = src.relative_to(real_repo)
        dst = tmp_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    (tmp_root / "data" / "private").mkdir(parents=True, exist_ok=True)
    (tmp_root / "data" / "private" / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_root / "data" / "private" / "p17_prereg_candidates.md").write_text(
        "# Metadata only\n", encoding="utf-8"
    )

    manifest_obj = json.loads(
        (tmp_root / "data" / "benchmark" / "p17_empirical_preregistration.json").read_text(
            encoding="utf-8"
        )
    )
    return tmp_root, manifest_obj


def run_self_tests(real_repo: Path) -> int:
    """Execute live repository check plus 9 adversarial mutation unit tests."""
    live_errors, live_stats = validate_preregistration(real_repo)
    if live_errors:
        print("FAIL [self-test 0/9: live repository check]:", file=sys.stderr)
        for err in live_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    cases_passed = 0
    total_cases = 9

    def _write_manifest(root: Path, obj: Dict[str, Any]) -> Path:
        p = root / "data" / "benchmark" / "p17_empirical_preregistration.json"
        p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
        return p

    with tempfile.TemporaryDirectory(prefix="p17_prereg_selftest_") as tmp_str:
        base_tmp = Path(tmp_str)

        # Baseline clean fixture check
        clean_root, base_manifest = _build_minimal_fixture_repo(base_tmp / "clean", real_repo)
        errs, _ = validate_preregistration(clean_root)
        if errs:
            print(f"FAIL [self-test clean fixture]: unexpected errors {errs}", file=sys.stderr)
            return 1

        # Case 1: Missing gate (`F13b` deleted)
        m1 = copy.deepcopy(base_manifest)
        del m1["gates"]["F13b"]
        _write_manifest(clean_root, m1)
        errs, _ = validate_preregistration(clean_root)
        assert any("F13b" in e for e in errs), f"Case 1 did not flag missing F13b: {errs}"
        cases_passed += 1

        # Case 2: Forbidden non-commercial / copyleft license (`CC-BY-NC-4.0`)
        m2 = copy.deepcopy(base_manifest)
        m2["gates"]["F13a"]["primary_datasets"][0]["license"] = "CC-BY-NC-4.0"
        _write_manifest(clean_root, m2)
        errs, _ = validate_preregistration(clean_root)
        assert any("violates strict allowlist" in e for e in errs), f"Case 2 failed: {errs}"
        cases_passed += 1

        # Case 3: Disjointness violation against `real_manifest.tsv` (`BMRB:51439`)
        m3 = copy.deepcopy(base_manifest)
        m3["gates"]["F11"]["primary_datasets"][0]["accession_id"] = "BMRB:51439"
        _write_manifest(clean_root, m3)
        errs, _ = validate_preregistration(clean_root)
        assert any("real_manifest.tsv" in e for e in errs), f"Case 3 failed: {errs}"
        cases_passed += 1

        # Case 4: Disjointness violation against `manifest.tsv` (`BMRB:17769`)
        m4 = copy.deepcopy(base_manifest)
        m4["gates"]["F10"]["backup_datasets"][0]["accession_id"] = "BMRB:17769"
        _write_manifest(clean_root, m4)
        errs, _ = validate_preregistration(clean_root)
        assert any("manifest.tsv" in e for e in errs), f"Case 4 failed: {errs}"
        cases_passed += 1

        # Case 5: Disjointness violation against `data/priors/PROVENANCE.md` (`BMRB:4022`)
        m5 = copy.deepcopy(base_manifest)
        m5["gates"]["F10"]["backup_datasets"][1]["accession_id"] = "BMRB:4022"
        _write_manifest(clean_root, m5)
        errs, _ = validate_preregistration(clean_root)
        assert any("PROVENANCE.md" in e for e in errs), f"Case 5 failed: {errs}"
        cases_passed += 1

        # Case 6: Blind-firewall violation using contaminated `BMRB:53960`
        m6 = copy.deepcopy(base_manifest)
        m6["gates"]["F11"]["backup_datasets"][0]["accession_id"] = "BMRB:53960"
        _write_manifest(clean_root, m6)
        errs, _ = validate_preregistration(clean_root)
        assert any("53960" in e for e in errs), f"Case 6 failed: {errs}"
        cases_passed += 1

        # Restore valid manifest before filesystem & freeze mutation tests
        _write_manifest(clean_root, base_manifest)

        # Case 7: Pre-existing download directory in `data/private/` (`f10_bmrb6470` and `bmrb/18380`)
        bad_dir = clean_root / "data" / "private" / "f10_bmrb6470"
        bad_dir.mkdir()
        errs, _ = validate_preregistration(clean_root)
        assert any("zero-download violation" in e for e in errs), f"Case 7a failed: {errs}"
        bad_dir.rmdir()

        bad_bmrb_dir = clean_root / "data" / "private" / "bmrb" / "18380"
        bad_bmrb_dir.mkdir(parents=True)
        errs, _ = validate_preregistration(clean_root)
        assert any("data/private/bmrb/" in e for e in errs), f"Case 7b failed: {errs}"
        shutil.rmtree(clean_root / "data" / "private" / "bmrb")
        cases_passed += 1

        # Case 8: Pre-existing downloaded target archive (`data/private/CEST.zip` and nested inside `p17_w1_results/`)
        w1_ok_dir = clean_root / "data" / "private" / "p17_w1_results"
        w1_ok_dir.mkdir()
        (w1_ok_dir / "w1_tables.md").write_text("# Wave 1 tables\n", encoding="utf-8")
        errs_w1_ok, _ = validate_preregistration(clean_root)
        assert not errs_w1_ok, f"Case 8a unexpectedly flagged clean p17_w1_results: {errs_w1_ok}"

        bad_nested = w1_ok_dir / "CEST.zip"
        bad_nested.write_bytes(b"PK\x03\x04")
        errs_nested, _ = validate_preregistration(clean_root)
        assert any("CEST.zip" in e for e in errs_nested), f"Case 8b failed nested check: {errs_nested}"
        shutil.rmtree(w1_ok_dir)

        bad_file = clean_root / "data" / "private" / "CEST.zip"
        bad_file.write_bytes(b"PK\x03\x04")
        errs, _ = validate_preregistration(clean_root)
        assert any("CEST.zip" in e for e in errs), f"Case 8c failed: {errs}"
        bad_file.unlink()
        cases_passed += 1

        # Case 9: All 21 solver/shader files (incl. post-F12(a) pinned `relax_mc.wgsl` and `modelfree.rs`) reject tampering
        relax_wgsl = clean_root / "crates" / "nmr-gpu" / "src" / "shaders" / "relax_mc.wgsl"
        orig_relax = relax_wgsl.read_text(encoding="utf-8")
        relax_wgsl.write_text(orig_relax + "\n// unauthorized post-W1 shader edit\n", encoding="utf-8")
        errs_wgsl_tampered, _ = validate_preregistration(clean_root)
        assert any("relax_mc.wgsl" in e for e in errs_wgsl_tampered), (
            f"Case 9a failed to reject tampered relax_mc.wgsl: {errs_wgsl_tampered}"
        )
        relax_wgsl.write_text(orig_relax, encoding="utf-8")

        mf_rs = clean_root / "crates" / "nmr-core" / "src" / "modelfree.rs"
        mf_rs.write_text(mf_rs.read_text(encoding="utf-8") + "\n// post-freeze edit\n", encoding="utf-8")
        errs_tampered, _ = validate_preregistration(clean_root)
        assert any("modelfree.rs" in e for e in errs_tampered), f"Case 9b failed: {errs_tampered}"
        cases_passed += 1

    print(
        f"OK [self-test]: live manifest verified + {cases_passed}/{total_cases} adversarial rejection tests passed "
        f"(manifest SHA-256 = {live_stats['manifest_sha256'][:16]}...)"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Phase 17 empirical pre-registration manifest schema, licenses, disjointness, and zero-download firewall."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate data/benchmark/p17_empirical_preregistration.json against repository state.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run live validation plus 9 isolated adversarial rejection unit tests.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root directory (defaults to parent of scripts/).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit path to p17_empirical_preregistration.json.",
    )
    args = parser.parse_args(argv)

    if not args.check and not args.self_test:
        args.check = True

    repo_root = args.repo_root.resolve()

    if args.self_test:
        rc = run_self_tests(repo_root)
        if rc != 0:
            return rc

    if args.check:
        errors, stats = validate_preregistration(repo_root, args.manifest)
        if errors:
            print("FAIL [verify_p17_preregistration]:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        print(
            "OK [verify_p17_preregistration]: "
            f"gates={stats['gates_verified']}/8, "
            f"datasets={stats['total_dataset_entries']} ({stats['unique_candidate_accessions']} unique accessions), "
            f"urls={stats['total_file_urls']} ({stats['head_verified_file_urls']} HEAD-sized), "
            f"disjoint_vs=(real:{stats['disjoint_real_manifest_count']}, "
            f"phase1:{stats['disjoint_phase1_manifest_count']}, "
            f"priors:{stats['disjoint_priors_count']}, excluded:53960), "
            f"private_dirs_clean={len(stats['private_dirs_checked'])}, "
            f"sha256={stats['manifest_sha256']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
