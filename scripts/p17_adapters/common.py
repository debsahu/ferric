#!/usr/bin/env python3
"""Shared cryptographic hashing, statistics, manifest criteria evaluation, and NMR-STAR v3.2 I/O.

Provenance:
- Original clean-room Python 3 standard-library implementation for Phase 17 blind
  empirical benchmarking (Rule P17-R2 of ``data/benchmark/p17_empirical_preregistration.json``).
- Public STAR / NMR-STAR v3.2 syntax: Hall, S. R., Allen, F. H. & Brown, I. D.,
  *Acta Crystallogr. A* 1991, 47, 655-685; Ulrich, E. L. et al., *Nucleic Acids Res.*
  2008, 36, D402-D408 (BioMagResBank).
- Zero external dependencies, zero network calls, and zero numeric constants beyond
  those explicitly bound in ``data/benchmark/p17_empirical_preregistration.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

DEFAULT_MANIFEST_REL = Path("data/benchmark/p17_empirical_preregistration.json")

# Standard IUPAC 1-letter to 3-letter amino-acid table (syntactic sequence token mapping).
AA_ONE_TO_THREE: Dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}
AA_THREE_SET = frozenset(AA_ONE_TO_THREE.values())

STAR_MISSING_TOKENS = frozenset({"", ".", "?", "na", "n/a", "nan", "none", "null"})


# ============================================================================
# 1. FIPS 180-4 SHA-256 Hashing & Pre-Invocation Ledger Helpers
# ============================================================================


def sha256_bytes(data: bytes) -> str:
    """Return lowercase 64-hex FIPS 180-4 SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Union[Path, str]) -> str:
    """Return lowercase 64-hex FIPS 180-4 SHA-256 digest of file ``path``."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_adapter_directory(adapter_dir: Optional[Union[Path, str]] = None) -> Dict[str, Dict[str, Any]]:
    """Compute pre-invocation byte sizes and SHA-256 digests for all ``.py`` files in ``adapter_dir``."""
    root = Path(adapter_dir) if adapter_dir is not None else Path(__file__).resolve().parent
    out: Dict[str, Dict[str, Any]] = {}
    for py_file in sorted(root.glob("*.py")):
        if py_file.is_file():
            out[py_file.name] = {
                "bytes": py_file.stat().st_size,
                "sha256": sha256_file(py_file),
            }
    return out


# ============================================================================
# 2. Deterministic Exact-f64 Summary Statistics
# ============================================================================


def _finite_values(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for v in values:
        fv = float(v)
        if math.isfinite(fv):
            out.append(fv)
    return out


def _finite_pairs(xs: Sequence[float], ys: Sequence[float]) -> Tuple[List[float], List[float]]:
    if len(xs) != len(ys):
        raise ValueError(f"paired statistic length mismatch: {len(xs)} != {len(ys)}")
    fx: List[float] = []
    fy: List[float] = []
    for x, y in zip(xs, ys):
        vx = float(x)
        vy = float(y)
        if math.isfinite(vx) and math.isfinite(vy):
            fx.append(vx)
            fy.append(vy)
    return fx, fy


def median(values: Sequence[float]) -> float:
    """Deterministic sample median over finite elements (``nan`` if empty)."""
    clean = sorted(_finite_values(values))
    n = len(clean)
    if n == 0:
        return math.nan
    mid = n // 2
    if n % 2 == 1:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / float(2)


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Deterministic two-pass compensated Pearson correlation coefficient ``r``."""
    fx, fy = _finite_pairs(xs, ys)
    n = len(fx)
    if n < 2:
        return math.nan
    mx = math.fsum(fx) / float(n)
    my = math.fsum(fy) / float(n)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(fx, fy))
    sxx = math.fsum((x - mx) * (x - mx) for x in fx)
    syy = math.fsum((y - my) * (y - my) for y in fy)
    denom = math.sqrt(sxx * syy)
    if denom <= 0.0 or not math.isfinite(denom):
        return math.nan
    r = sxy / denom
    return max(-1.0, min(1.0, r))


def _average_ranks(vals: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(vals), key=lambda kv: kv[1])
    ranks = [0.0] * len(vals)
    i = 0
    n = len(indexed)
    while i < n:
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        # 1-based average rank for ties in [i, j)
        avg_rank = float((i + 1) + j) / float(2)
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Tie-aware Spearman rank correlation coefficient ``rho``."""
    fx, fy = _finite_pairs(xs, ys)
    if len(fx) < 2:
        return math.nan
    return pearson_r(_average_ranks(fx), _average_ranks(fy))


def mae(preds: Sequence[float], refs: Sequence[float]) -> float:
    """Mean absolute error ``mean(|pred - ref|)``."""
    fp, fr = _finite_pairs(preds, refs)
    if not fp:
        return math.nan
    return math.fsum(abs(p - r) for p, r in zip(fp, fr)) / float(len(fp))


def rmse(preds: Sequence[float], refs: Sequence[float]) -> float:
    """Root-mean-square error ``sqrt(mean((pred - ref)^2))``."""
    fp, fr = _finite_pairs(preds, refs)
    if not fp:
        return math.nan
    return math.sqrt(math.fsum((p - r) * (p - r) for p, r in zip(fp, fr)) / float(len(fp)))


def median_abs_err(preds: Sequence[float], refs: Sequence[float]) -> float:
    """Median absolute error ``median(|pred - ref|)``."""
    fp, fr = _finite_pairs(preds, refs)
    if not fp:
        return math.nan
    return median([abs(p - r) for p, r in zip(fp, fr)])


def median_rel_abs_err(preds: Sequence[float], refs: Sequence[float]) -> float:
    """Median relative absolute error ``median(|pred - ref| / |ref|)`` over nonzero ``ref``."""
    fp, fr = _finite_pairs(preds, refs)
    rels = [abs(p - r) / abs(r) for p, r in zip(fp, fr) if abs(r) > 0.0]
    if not rels:
        return math.nan
    return median(rels)


# ============================================================================
# 3. Pre-Registration Manifest Loader & Bound-Constant Extractor
# ============================================================================


def find_repo_root(start: Optional[Union[Path, str]] = None) -> Path:
    """Walk upward from ``start`` (or this module) to locate the repository root."""
    cur = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if cur.is_file():
        cur = cur.parent
    for cand in [cur, *cur.parents]:
        if (cand / DEFAULT_MANIFEST_REL).is_file():
            return cand
    raise FileNotFoundError(
        f"could not locate {DEFAULT_MANIFEST_REL} above {cur}"
    )


def load_preregistration_manifest(
    manifest_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Load ``data/benchmark/p17_empirical_preregistration.json`` directly from disk."""
    if manifest_path is not None:
        mpath = Path(manifest_path)
    else:
        mpath = find_repo_root() / DEFAULT_MANIFEST_REL
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or "gates" not in manifest:
        raise ValueError(f"invalid pre-registration manifest at {mpath}")
    return manifest


def _require_regex_groups(pattern: str, text: str, context: str) -> Tuple[str, ...]:
    m = re.search(pattern, text)
    if m is None:
        raise ValueError(f"failed extracting bound manifest constant ({context}) from: {text!r}")
    return m.groups()


def extract_manifest_bound_constants(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract every gate-bound numeric/domain literal directly from ``manifest``.

    Guarantees zero hardcoded physics thresholds or unregistered algorithmic constants
    anywhere in ``scripts/p17_adapters/``.
    """
    gates = manifest["gates"]
    r2_convs = {
        c["id"]: c
        for c in manifest["protocol_rules"]["rule_2_a_priori_deterministic_format_conversion"][
            "fixed_conversions"
        ]
    }

    # F10 bound constants
    f10 = gates["F10"]
    (f10_hi_mhz_s,) = _require_regex_groups(
        r"field_mhz\s*=\s*([0-9.]+)",
        f10["bound_parameters"]["ubq_600_r1.tsv"]["formula"],
        "F10 600 MHz field",
    )
    (f10_lo_mhz_s,) = _require_regex_groups(
        r"field_mhz\s*=\s*([0-9.]+)",
        f10["bound_parameters"]["ubq_360_r1.tsv"]["formula"],
        "F10 360 MHz field",
    )
    (f10_tau_ref_s,) = _require_regex_groups(
        r"abs\(tau_m_ns\s*-\s*([0-9.]+)\)",
        f10["pass_criteria"][0]["metric"],
        "F10 tau_m_ref",
    )
    (f10_aniso_ref_s,) = _require_regex_groups(
        r"abs\(d_par_over_d_perp\s*-\s*([0-9.]+)\)",
        f10["pass_criteria"][1]["metric"],
        "F10 d_par_over_d_perp_ref",
    )

    # F11 Bruker hsqct2etf3gpsi3d vclist loop-timing formula
    f11_vclist_formula = r2_convs["bruker_hsqct2etf3gpsi3d_vclist_loop_timing"]["formula"]
    (
        d_mult_s,
        d_idx_s,
        p_mult_s,
        p_idx_s,
        us_scale_s,
        p_fb_mult_s,
        p_fb_idx_s,
    ) = _require_regex_groups(
        r"([0-9.]+)\s*\*\s*D\[(\d+)\]\s*\+\s*([0-9.]+)\s*\*\s*\(P\[(\d+)\]\s*\*\s*([0-9.eE+-]+)\)"
        r".*?P\[\d+\]\s*=\s*([0-9.]+)\s*\*\s*P\[(\d+)\]",
        f11_vclist_formula,
        "F11 hsqct2etf3gpsi3d vclist timing",
    )

    # F13a low-pB floor formula
    f13a_c1_src = gates["F13a"]["pass_criteria"][0]["reference_value_or_source"]
    (f13a_abs_floor_s, f13a_pb_cut_s, f13a_rel_eq_s) = _require_regex_groups(
        r"<=\s*([0-9.]+)\s*when\s*p_B_ref\s*<\s*([0-9.]+).*?\*\s*([0-9.]+)\)",
        f13a_c1_src,
        "F13a low-p_B floor",
    )

    # F13c mM -> uM unit scale
    (f13c_mm_to_um_s,) = _require_regex_groups(
        r"\*\s*([0-9.]+)",
        gates["F13c"]["bound_parameters"]["P0_uM"]["formula"],
        "F13c mM to uM conversion factor",
    )

    # F14 SNR, DEPT-135 site counts, and qNMR centroid half-window
    f14 = gates["F14"]
    (f14_snr_s,) = _require_regex_groups(
        r"SNR\s*>=\s*([0-9.]+)",
        f14["pass_criteria"][0]["rationale"],
        "F14 principal SNR",
    )
    (f14_pos_sites_s, f14_neg_sites_s) = _require_regex_groups(
        r"=\s*(\d+)\s+positive carbon sites,\s*(\d+)\s+negative CH2",
        f14["pass_criteria"][1]["reference_value_or_source"],
        "F14 L-fucose DEPT-135 carbon site counts",
    )
    (f14_ch2_snr_s,) = _require_regex_groups(
        r"SNR\s*>=\s*([0-9.]+)\s+CH2",
        f14["pass_criteria"][1]["rationale"],
        "F14 DEPT-135 CH2 SNR",
    )
    (f14_qnmr_hw_s,) = _require_regex_groups(
        r"\+-\s*([0-9.]+)\s*ppm",
        f14["bound_parameters"]["STD_REGION_HI_LO"]["key_or_column"],
        "F14 qNMR centroid half-window",
    )

    # F16 DOSY locked-g0 F2 integration & header solvent/temperature formulas
    f16 = gates["F16"]
    f16_decay_formula = f16["bound_parameters"]["decay.tsv"]["formula"]
    (
        f16_lb_s,
        f16_zf_s,
        f16_snr_s,
        f16_hdo_lo_s,
        f16_hdo_hi_s,
        f16_box_mult_s,
    ) = _require_regex_groups(
        r"([0-9.]+)\s*Hz\s+exponential apodization,\s*(\d+)x\s+zero-fill.*?"
        r"SNR\s*>=\s*([0-9.]+).*?\[([0-9.]+),\s*([0-9.]+)\]\s*ppm.*?"
        r"\+-([0-9.]+)\*FWHM",
        f16_decay_formula,
        "F16 DOSY locked-g0 box integration",
    )
    f16_solv_formula = f16["bound_parameters"]["header_solvent"]["formula"]
    (f16_solv_set_s, f16_solv_fb_s) = _require_regex_groups(
        r"\{([^}]+)\}`?\s*\(fallback\s*`([^`]+)`",
        f16_solv_formula,
        "F16 DOSY header solvents",
    )
    f16_temp_formula = f16["bound_parameters"]["header_temp_k"]["formula"]
    (f16_te_cut_s, f16_c2k_s, f16_temp_fb_s) = _require_regex_groups(
        r">=\s*([0-9.]+)\s*else\s*temp_C\s*\+\s*([0-9.]+)\s*\(fallback\s*([0-9.]+)",
        f16_temp_formula,
        "F16 DOSY header temperature",
    )

    # F17 MTBLS1 top-15 VIP/STOCSY & urinary D-glucose windows
    f17 = gates["F17"]
    f17_c4_src = f17["pass_criteria"][3]["reference_value_or_source"]
    (
        f17_top_k_s,
        f17_driver_s,
        f17_g1_lo_s,
        f17_g1_hi_s,
        f17_g2_lo_s,
        f17_g2_hi_s,
    ) = _require_regex_groups(
        r"Top-(\d+)\s+ranked OPLS-DA VIP buckets or\s*([0-9.]+)\s*ppm STOCSY.*?"
        r"\[([0-9.]+),\s*([0-9.]+)\]\s*ppm.*?\[([0-9.]+),\s*([0-9.]+)\]\s*ppm",
        f17_c4_src,
        "F17 D-glucose glycosuria windows",
    )

    return {
        "excluded_contaminated_entries": tuple(manifest.get("excluded_contaminated_entries", [])),
        "F10": {
            "bmrb6470_high_field_mhz": float(f10_hi_mhz_s),
            "bmrb6470_low_field_mhz": float(f10_lo_mhz_s),
            "tau_m_ref_ns": float(f10_tau_ref_s),
            "d_par_over_d_perp_ref": float(f10_aniso_ref_s),
        },
        "F11": {
            "d_mult": float(d_mult_s),
            "d_index": int(d_idx_s),
            "p_mult": float(p_mult_s),
            "p_index": int(p_idx_s),
            "us_to_s": float(us_scale_s),
            "p_fallback_mult": float(p_fb_mult_s),
            "p_fallback_index": int(p_fb_idx_s),
        },
        "F13a": {
            "low_pb_cutoff": float(f13a_pb_cut_s),
            "low_pb_abs_floor": float(f13a_abs_floor_s),
            "low_pb_rel_equiv": float(f13a_rel_eq_s),
        },
        "F13c": {
            "mm_to_um": float(f13c_mm_to_um_s),
        },
        "F14": {
            "min_principal_snr": float(f14_snr_s),
            "fucose_expected_pos_sites": int(f14_pos_sites_s),
            "fucose_expected_neg_ch2": int(f14_neg_sites_s),
            "dept135_ch2_min_snr": float(f14_ch2_snr_s),
            "qnmr_centroid_half_window_ppm": float(f14_qnmr_hw_s),
        },
        "F16": {
            "lb_hz": float(f16_lb_s),
            "zf_factor": int(f16_zf_s),
            "min_snr": float(f16_snr_s),
            "hdo_mask_ppm": (float(f16_hdo_lo_s), float(f16_hdo_hi_s)),
            "box_fwhm_mult": float(f16_box_mult_s),
            "allowed_solvents": [s.strip() for s in f16_solv_set_s.split(",") if s.strip()],
            "fallback_solvent": f16_solv_fb_s.strip(),
            "temp_kelvin_cutoff": float(f16_te_cut_s),
            "celsius_to_kelvin_offset": float(f16_c2k_s),
            "fallback_temp_k": float(f16_temp_fb_s),
        },
        "F17": {
            "top_k": int(f17_top_k_s),
            "stocsy_driver_ppm": float(f17_driver_s),
            "glucose_windows_ppm": [
                (float(f17_g1_lo_s), float(f17_g1_hi_s)),
                (float(f17_g2_lo_s), float(f17_g2_hi_s)),
            ],
        },
    }


# ============================================================================
# 4. Strict Manifest-Driven Pass-Criteria Evaluator
# ============================================================================


def _compare_op(value: float, operator: str, threshold: float) -> bool:
    if not math.isfinite(value) or not math.isfinite(threshold):
        return False
    if operator == "<=":
        return value <= threshold
    if operator == ">=":
        return value >= threshold
    if operator == "==":
        return value == threshold
    if operator == "<":
        return value < threshold
    if operator == ">":
        return value > threshold
    raise ValueError(f"unsupported pass_criteria operator: {operator!r}")


def evaluate_gate_criteria(
    gate_id: str,
    metrics: Mapping[str, Union[int, float]],
    *,
    exit_code: int = 0,
    accession_id: Optional[str] = None,
    nucleus: Optional[str] = None,
    n_sites: Optional[int] = None,
    converged: bool = True,
    archive_not_evaluable_reason: Optional[str] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate ``gate_id`` strictly against ``data/benchmark/p17_empirical_preregistration.json``.

    Zero numeric thresholds live in Python code: every threshold, fallback override,
    nucleus-specific threshold, and disjunctive multi-site clause is read directly
    from the pre-registration manifest.
    """
    mf = manifest if manifest is not None else load_preregistration_manifest()
    gates = mf.get("gates", {})
    if gate_id not in gates:
        raise KeyError(f"unknown gate_id {gate_id!r} (expected one of {sorted(gates)})")
    gate_spec = gates[gate_id]

    if archive_not_evaluable_reason is not None:
        reason_clean = archive_not_evaluable_reason.strip()
        if not reason_clean:
            raise ValueError("archive_not_evaluable_reason must be a non-empty verbatim failure description")
        return {
            "gate_id": gate_id,
            "verdict": "NOT_EVALUABLE",
            "exit_code": exit_code,
            "accession_id": accession_id,
            "converged": converged,
            "all_criteria_passed": False,
            "not_evaluable_reason": reason_clean,
            "criteria_results": [],
        }

    conjunction = gate_spec.get("pass_criteria_conjunction", "ALL")
    if conjunction != "ALL":
        raise ValueError(f"unsupported pass_criteria_conjunction {conjunction!r} on {gate_id}")

    criteria_results: List[Dict[str, Any]] = []
    for crit in gate_spec["pass_criteria"]:
        cid = crit["id"]
        metric_name = crit["metric"]
        op = crit["operator"]
        effective_threshold = float(crit["threshold"])
        threshold_source = "primary"

        fb_overrides = crit.get("fallback_threshold_overrides")
        if isinstance(fb_overrides, Mapping) and accession_id and accession_id in fb_overrides:
            effective_threshold = float(fb_overrides[accession_id])
            threshold_source = f"fallback_threshold_overrides[{accession_id}]"

        nuc_overrides = crit.get("nucleus_specific_thresholds")
        if isinstance(nuc_overrides, Mapping) and nucleus:
            nuc_norm = nucleus.strip().upper()
            if nuc_norm in nuc_overrides:
                effective_threshold = float(nuc_overrides[nuc_norm])
                threshold_source = f"nucleus_specific_thresholds[{nuc_norm}]"

        raw_val = metrics.get(metric_name, math.nan)
        val_f = float(raw_val) if raw_val is not None else math.nan
        primary_passed = _compare_op(val_f, op, effective_threshold)

        disjunctive_used = False
        disjunctive_passed = False
        disj_clause = crit.get("disjunctive_clause_when_n_sites_ge_5")
        if isinstance(disj_clause, Mapping):
            min_sites = int(disj_clause["min_n_sites"])
            if n_sites is not None and int(n_sites) >= min_sites:
                disjunctive_used = True
                d_metric = disj_clause["metric"]
                d_op = disj_clause["operator"]
                d_thresh = float(disj_clause["threshold"])
                d_val_raw = metrics.get(d_metric, math.nan)
                d_val_f = float(d_val_raw) if d_val_raw is not None else math.nan
                disjunctive_passed = _compare_op(d_val_f, d_op, d_thresh)

        crit_passed = primary_passed or (disjunctive_used and disjunctive_passed)
        criteria_results.append(
            {
                "id": cid,
                "metric": metric_name,
                "value": val_f,
                "operator": op,
                "threshold": effective_threshold,
                "threshold_source": threshold_source,
                "primary_passed": primary_passed,
                "disjunctive_eligible": disjunctive_used,
                "disjunctive_passed": disjunctive_passed,
                "passed": crit_passed,
            }
        )

    all_passed = (
        int(exit_code) == 0
        and bool(converged)
        and bool(criteria_results)
        and all(c["passed"] for c in criteria_results)
    )
    verdict = "PASS" if all_passed else "FAIL"
    return {
        "gate_id": gate_id,
        "verdict": verdict,
        "exit_code": int(exit_code),
        "accession_id": accession_id,
        "nucleus": nucleus,
        "n_sites": n_sites,
        "converged": bool(converged),
        "all_criteria_passed": all_passed,
        "criteria_results": criteria_results,
    }


# ============================================================================
# 5. NMR-STAR v3.2 / v3.1 Saveframe & Loop Parser
# ============================================================================


@dataclass
class StarLoop:
    """Parsed STAR ``loop_`` ... ``stop_`` table inside a saveframe."""

    tags: List[str]
    rows: List[List[str]]

    @property
    def category(self) -> str:
        if not self.tags:
            return ""
        first = self.tags[0]
        if "." in first:
            return first.split(".", 1)[0]
        return first

    def find_col(self, candidates: Sequence[str]) -> Optional[int]:
        low_tags = [t.lower() for t in self.tags]
        for cand in candidates:
            target = cand.lower()
            suffix = "." + target.lstrip("_")
            for idx, tag in enumerate(low_tags):
                if tag == target or tag.endswith(suffix):
                    return idx
        return None


@dataclass
class StarSaveframe:
    """Parsed NMR-STAR saveframe preserving file order, scalar tags, and loops."""

    name: str
    index: int
    tags: Dict[str, str] = field(default_factory=dict)
    loops: List[StarLoop] = field(default_factory=list)

    def get_tag(self, candidates: Sequence[str]) -> Optional[str]:
        low_map = {k.lower(): v for k, v in self.tags.items()}
        for cand in candidates:
            target = cand.lower()
            suffix = "." + target.lstrip("_")
            if target in low_map:
                val = low_map[target]
                if val.lower() not in STAR_MISSING_TOKENS:
                    return val
            for k, v in low_map.items():
                if k.endswith(suffix) and v.lower() not in STAR_MISSING_TOKENS:
                    return v
        return None

    @property
    def sf_category(self) -> str:
        cat = self.get_tag(["Sf_category", "Saveframe_category"])
        return cat if cat is not None else ""


def _tokenize_star_line(line: str, out: List[Tuple[str, bool]]) -> None:
    n = len(line)
    i = 0
    while i < n:
        c = line[i]
        if c.isspace():
            i += 1
            continue
        if c == "#":
            break
        if c in ("'", '"'):
            quote = c
            i += 1
            buf: List[str] = []
            while i < n:
                if line[i] == quote and (i + 1 == n or line[i + 1].isspace()):
                    i += 1
                    break
                buf.append(line[i])
                i += 1
            out.append(("".join(buf), True))
        else:
            buf = []
            while i < n and not line[i].isspace():
                buf.append(line[i])
                i += 1
            out.append(("".join(buf), False))


def tokenize_star(text: str) -> List[Tuple[str, bool]]:
    """Tokenize STAR / NMR-STAR v3.2 text into ``(token, is_quoted)`` pairs."""
    tokens: List[Tuple[str, bool]] = []
    lines = text.splitlines()
    n_lines = len(lines)
    i = 0
    while i < n_lines:
        line = lines[i]
        if line.startswith(";"):
            block = [line[1:]]
            i += 1
            while i < n_lines and not lines[i].startswith(";"):
                block.append(lines[i])
                i += 1
            tokens.append(("\n".join(block).strip(), True))
            i += 1
            continue
        _tokenize_star_line(line, tokens)
        i += 1
    return tokens


def parse_nmrstar_saveframes(text: str) -> List[StarSaveframe]:
    """Parse NMR-STAR v2.1 / v3.1 / v3.2 text into ordered ``StarSaveframe`` records.

    Handles ``_Auto_relaxation``, ``_Heteronucl_T1_relaxation``, ``_Heteronucl_T2_relaxation``,
    ``_Heteronucl_NOE`` / ``_Heteronucl_NOEs``, ``_Order_parameter`` / ``_Order_param``,
    ``_Atom_chem_shift``, and top-level loop-only STAR blocks (wrapped in a synthetic
    ``save_global`` frame if no ``save_<name>`` wrapper is present).
    """
    tokens = tokenize_star(text)
    saveframes: List[StarSaveframe] = []
    current_sf: Optional[StarSaveframe] = None

    def _ensure_sf() -> StarSaveframe:
        nonlocal current_sf
        if current_sf is None:
            current_sf = StarSaveframe(name="global", index=len(saveframes))
            saveframes.append(current_sf)
        return current_sf

    n = len(tokens)
    i = 0
    while i < n:
        tok, is_quoted = tokens[i]
        if not is_quoted:
            low = tok.lower()
            if low.startswith("data_"):
                i += 1
                continue
            if low == "save_":
                current_sf = None
                i += 1
                continue
            if low.startswith("save_") and len(tok) > 5:
                current_sf = StarSaveframe(name=tok[5:], index=len(saveframes))
                saveframes.append(current_sf)
                i += 1
                continue
            if low == "loop_":
                sf = _ensure_sf()
                i += 1
                loop_tags: List[str] = []
                while i < n and not tokens[i][1] and tokens[i][0].startswith("_"):
                    loop_tags.append(tokens[i][0])
                    i += 1
                k_cols = max(1, len(loop_tags))
                vals: List[str] = []
                while i < n:
                    vt, vq = tokens[i]
                    vlow = vt.lower()
                    if not vq and (
                        vlow == "stop_"
                        or vlow == "loop_"
                        or vlow.startswith("save_")
                        or (len(vals) % k_cols == 0 and vt.startswith("_"))
                    ):
                        break
                    vals.append(vt)
                    i += 1
                if i < n and not tokens[i][1] and tokens[i][0].lower() == "stop_":
                    i += 1
                if loop_tags and len(vals) >= len(loop_tags):
                    n_rows = len(vals) // len(loop_tags)
                    rows = [
                        vals[r * len(loop_tags) : (r + 1) * len(loop_tags)]
                        for r in range(n_rows)
                    ]
                    sf.loops.append(StarLoop(tags=loop_tags, rows=rows))
                continue
            if tok.startswith("_"):
                sf = _ensure_sf()
                if i + 1 < n:
                    val_tok, val_quoted = tokens[i + 1]
                    if val_quoted or not (
                        val_tok.startswith("_")
                        or val_tok.lower() in ("loop_", "stop_", "save_")
                        or val_tok.lower().startswith("save_")
                    ):
                        sf.tags[tok] = val_tok
                        i += 2
                        continue
        i += 1

    return saveframes


def parse_star_float(raw: Optional[str]) -> Optional[float]:
    """Parse an NMR-STAR numeric cell, returning ``None`` for ``.`` / ``?`` / ``NA``."""
    if raw is None:
        return None
    s = raw.strip()
    if s.lower() in STAR_MISSING_TOKENS:
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def parse_star_int(raw: Optional[str]) -> Optional[int]:
    """Parse an NMR-STAR integer sequence index, returning ``None`` for missing cells."""
    if raw is None:
        return None
    s = raw.strip()
    if s.lower() in STAR_MISSING_TOKENS:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def parse_residue_token(token: str) -> Tuple[int, str]:
    """Parse common residue identifiers (``23``, ``23ASN``, ``ASN23``, ``N23``, ``23N-H``) into ``(seq_id, res_name)``."""
    s = token.strip()
    if not s:
        raise ValueError("empty residue token")
    # Remove trailing atom specifier such as `N-H`, `N-HN`, `_N`, `:N`
    core = re.split(r"[:/]", s)[0]
    core = re.sub(r"(?:[-_]?[NHC]{1,2}\d*(?:-[NHC]{1,2}\d*)?)$", "", core) or s

    if core.lstrip("-").isdigit():
        return int(core), "UNK"

    # Case 1: `23ASN` or `23N`
    m_num_first = re.match(r"^(-?\d+)\s*([A-Za-z]{1,3})$", core)
    if m_num_first:
        seq_id = int(m_num_first.group(1))
        aa_raw = m_num_first.group(2).upper()
        res_name = AA_ONE_TO_THREE.get(aa_raw, aa_raw if aa_raw in AA_THREE_SET else "UNK")
        return seq_id, res_name

    # Case 2: `ASN23` or `N23`
    m_aa_first = re.match(r"^([A-Za-z]{1,3})\s*(-?\d+)", core)
    if m_aa_first:
        aa_raw = m_aa_first.group(1).upper()
        seq_id = int(m_aa_first.group(2))
        res_name = AA_ONE_TO_THREE.get(aa_raw, aa_raw if aa_raw in AA_THREE_SET else "UNK")
        return seq_id, res_name

    # Fallback: first signed integer anywhere in token
    m_any = re.search(r"(-?\d+)", s)
    if m_any:
        return int(m_any.group(1)), "UNK"
    raise ValueError(f"could not parse integer residue seq_id from {token!r}")


def extract_nmrstar_backbone_amide_shifts(star_text: str) -> Dict[int, Dict[str, Any]]:
    """Extract assigned ``_Atom_chem_shift`` backbone ``(H, N)`` chemical shifts by ``seq_id``."""
    out: Dict[int, Dict[str, Any]] = {}
    for sf in parse_nmrstar_saveframes(star_text):
        for lp in sf.loops:
            val_c = lp.find_col(["_Atom_chem_shift.Val", "_Chem_shift_value"])
            if val_c is None:
                continue
            sid_c = lp.find_col(
                ["_Atom_chem_shift.Seq_ID", "_Atom_chem_shift.Comp_index_ID", "_Residue_seq_code"]
            )
            comp_c = lp.find_col(["_Atom_chem_shift.Comp_ID", "_Residue_label"])
            atom_c = lp.find_col(["_Atom_chem_shift.Atom_ID", "_Atom_name"])
            if sid_c is None or comp_c is None or atom_c is None:
                continue
            for row in lp.rows:
                sid = parse_star_int(row[sid_c])
                val = parse_star_float(row[val_c])
                if sid is None or val is None:
                    continue
                comp = row[comp_c].strip().upper()
                atom = row[atom_c].strip().upper()
                if atom == "HN":
                    atom = "H"
                rec = out.setdefault(sid, {"seq_id": sid, "res_name": comp, "shifts": {}})
                rec["shifts"][atom] = val
            if out:
                return out
    return out
