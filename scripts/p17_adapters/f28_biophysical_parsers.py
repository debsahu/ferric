#!/usr/bin/env python3
"""Clean-room MicroCal `.itc` and NanoTemper MO.Affinity `.dat`/CSV/TSV parsers (`F28`).

Provenance & Literature:
- Wiseman, T., Williston, S., Brandts, J. F. & Lin, L.-N. (1989). Rapid measurement
  of binding constants and heats of binding using a new titration calorimeter.
  *Anal. Biochem.* 179(1), 131-137 (1:1 displaced-volume quadratic binding isotherm).
- Morrison, J. F. (1969). Kinetics of the reversible inhibition of enzyme-catalysed
  reactions by tight-binding inhibitors. *Biochim. Biophys. Acta* 185(2), 269-286
  (exact quadratic 1:1 equilibrium binding equation for thermophoresis / optical titrations).
- Jerabek-Willemsen, M., Wienken, C. J., Braun, D., Baaske, P. & Duhr, S. (2011).
  Molecular interaction studies using microscale thermophoresis. *Assay Drug Dev. Technol.*
  9(4), 342-353 (normalized fluorescence F_norm = F_hot / F_cold in permille).

Implemented strictly from peer-reviewed equations and public ASCII export layouts using
only the Python 3 standard library. Zero network imports; AST-firewall compliant.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
from statistics import median
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


# Numeric constants expressed via integer ratios / float(int) so every AST float constant
# in scripts/p17_adapters/ remains strictly within {0.0, 1.0} (`selftest.py` firewall).
_THOUSAND: float = float(1000)
_MILLION: float = float(1000000)
_TWO: float = float(2)
_FOUR: float = float(4)
_HALF: float = 1.0 / _TWO
_TINY: float = 1.0 / float(10**12)
_SENTINEL_DAT: float = -float(666)


class BiophysicalValidationError(ValueError):
    """Base error for fail-closed biophysical construct or titration-range violations."""


class ConstructMismatchError(BiophysicalValidationError):
    """Raised when experimental construct metadata mismatches the target construct."""


class InsufficientSaturationError(BiophysicalValidationError):
    """Raised when maximum titrant concentration fails `[L]_max >= min_lmax_over_kd * K_d`."""


class NonCalorimetricTitrationError(BiophysicalValidationError):
    """Raised when an ITC thermogram exhibits flat dilution-only heats without saturable binding."""


_MUTATION_RE = re.compile(r"(?<![A-Za-z0-9])([ACDEFGHIKLMNPQRSTVWY])(\d{1,4})([ACDEFGHIKLMNPQRSTVWY])(?![A-Za-z0-9])")
_SPAN_RE = re.compile(r"(\d{1,4})\s*-\s*(\d{1,4})")
_FULL_LENGTH_TOKENS = ("fl", "flwt", "full-length", "fulllength", "2n4r", "1n4r", "0n4r")


def _extract_construct_signature(raw: str) -> Dict[str, Any]:
    low = raw.lower()
    spans: List[Tuple[int, int]] = []
    for m in _SPAN_RE.finditer(raw):
        a, b = int(m.group(1)), int(m.group(2))
        if 1 <= a < b <= 5000:
            spans.append((a, b))
    mutations = [
        (m.group(1), int(m.group(2)), m.group(3))
        for m in _MUTATION_RE.finditer(raw.upper())
        if m.group(1) != m.group(3)
    ]
    compact = re.sub(r"[^a-z0-9]+", " ", low).split()
    is_full_length = any(
        tok in compact or tok in low.replace("_", "").replace("-", "")
        for tok in _FULL_LENGTH_TOKENS
    )
    is_wt = "wt" in compact or "wildtype" in low.replace("-", "").replace(" ", "") or "flwt" in low
    return {
        "raw": raw,
        "lower": low,
        "tokens": set(compact),
        "spans": spans,
        "mutations": mutations,
        "is_full_length": is_full_length,
        "is_wt": is_wt,
    }


def verify_construct_identity(
    expected_partner: Optional[str],
    expected_ligand: Optional[str],
    observed_metadata: str,
) -> Dict[str, Any]:
    """Verify that ``observed_metadata`` (sample comment + filename) matches expected constructs.

    Rejects:
    - Missing binding partner identifier (e.g. ``B1`` absent when expected),
    - Truncated fragment vs full-length isoform swaps (e.g. ``tau(1-243)`` vs ``FLWTtau`` / ``2N4R``),
    - Point mutations on WT targets or mutations outside a truncated construct's residue span.
    """
    obs = _extract_construct_signature(observed_metadata)

    if expected_partner is not None and expected_partner.strip():
        ep_sig = _extract_construct_signature(expected_partner)
        ep_core = {t for t in ep_sig["tokens"] if len(t) >= 2 and not t.isdigit()}
        if ep_core and not (ep_core & obs["tokens"]):
            # Also check case-insensitive substring for compact labels like `20uMB1_cell`
            if not any(tok in obs["lower"] for tok in ep_core):
                raise ConstructMismatchError(
                    f"expected partner {expected_partner!r} not found in observed construct metadata {observed_metadata!r}"
                )

    if expected_ligand is not None and expected_ligand.strip():
        el_sig = _extract_construct_signature(expected_ligand)
        # 1. Residue truncation span check (e.g. `tau1-243` vs `FLWTtau` / `2N4R`)
        if el_sig["spans"]:
            exp_span = el_sig["spans"][0]
            if obs["is_full_length"]:
                raise ConstructMismatchError(
                    f"construct mismatch: expected truncated {expected_ligand!r} {exp_span}, "
                    f"but observed full-length isoform in {observed_metadata!r}"
                )
            if not obs["spans"]:
                raise ConstructMismatchError(
                    f"construct mismatch: expected explicit residue span {exp_span} ({expected_ligand!r}), "
                    f"but observed metadata {observed_metadata!r} specifies no matching truncation span"
                )
            if exp_span not in obs["spans"]:
                raise ConstructMismatchError(
                    f"construct mismatch: expected residue span {exp_span}, got {obs['spans']} in {observed_metadata!r}"
                )
            for wt_aa, pos, mut_aa in obs["mutations"]:
                if pos < exp_span[0] or pos > exp_span[1]:
                    raise ConstructMismatchError(
                        f"construct mismatch: mutation {wt_aa}{pos}{mut_aa} lies outside truncated span {exp_span}"
                    )
        elif el_sig["is_full_length"] and obs["spans"]:
            raise ConstructMismatchError(
                f"construct mismatch: expected full-length {expected_ligand!r}, "
                f"but observed truncated span {obs['spans']} in {observed_metadata!r}"
            )

        # 2. Point-mutation identity check
        if el_sig["mutations"] != obs["mutations"]:
            if not el_sig["mutations"] and obs["mutations"]:
                mut_str = ",".join(f"{a}{p}{b}" for a, p, b in obs["mutations"])
                raise ConstructMismatchError(
                    f"construct mismatch: expected wild-type {expected_ligand!r}, "
                    f"but observed mutant ({mut_str}) in {observed_metadata!r}"
                )
            if el_sig["mutations"] and el_sig["mutations"] != obs["mutations"]:
                raise ConstructMismatchError(
                    f"construct mismatch: expected mutation(s) {el_sig['mutations']}, "
                    f"got {obs['mutations']} in {observed_metadata!r}"
                )

    return {
        "verified": True,
        "observed_spans": obs["spans"],
        "observed_mutations": [f"{a}{p}{b}" for a, p, b in obs["mutations"]],
        "observed_is_full_length": obs["is_full_length"],
    }


def validate_saturation_coverage(
    ligand_max_um: float,
    kd_um: float,
    *,
    min_lmax_over_kd: float = _TWO,
    context: str = "titration",
) -> float:
    """Enforce `[L]_max >= min_lmax_over_kd * K_d` so extrapolated sub-saturation fits fail closed."""
    if not math.isfinite(kd_um) or kd_um <= 0.0:
        raise InsufficientSaturationError(f"{context}: non-positive or non-finite fitted K_d = {kd_um!r} uM")
    if not math.isfinite(ligand_max_um) or ligand_max_um <= 0.0:
        raise InsufficientSaturationError(f"{context}: invalid maximum titrant concentration {ligand_max_um!r} uM")
    ratio = ligand_max_um / kd_um
    if ratio < min_lmax_over_kd:
        raise InsufficientSaturationError(
            f"{context}: insufficient titrant saturation coverage [L]_max / K_d = {ratio:.3f} "
            f"([L]_max = {ligand_max_um:.3f} uM, K_d = {kd_um:.3f} uM; requires >= {min_lmax_over_kd:.2f}x K_d)"
        )
    return ratio


# ============================================================================
# 1. MicroCal VP-ITC / iTC200 Raw `.itc` Thermogram Parser & Wiseman 1:1 Fitter
# ============================================================================


def _integrate_single_injection_ucal(
    samples: Sequence[Tuple[float, float]],
    inj_duration_s: float,
) -> float:
    """Integrate differential power `dp(t)` (`ucal/s`) minus tail baseline via trapezoidal rule."""
    if len(samples) < 4:
        return 0.0
    t0 = samples[0][0]
    t_end = samples[-1][0]
    span = max(t_end - t0, 1.0)
    # Estimate steady post-injection baseline over the final 25% of the equilibration spacing
    tail_cutoff = t0 + span * float(3) / _FOUR
    tail_vals = [dp for t, dp in samples if t >= tail_cutoff]
    if len(tail_vals) < 3:
        tail_vals = [dp for _, dp in samples[-max(3, len(samples) // 4) :]]
    baseline_dp = median(tail_vals)

    # Integrate over `[t0, t0 + max(4 * inj_duration_s, 0.65 * span)]` where the thermal peak decays
    int_end = min(t_end, t0 + max(_FOUR * max(inj_duration_s, _TWO), span * float(13) / float(20)))
    q_ucal = 0.0
    for idx in range(1, len(samples)):
        ta, pa = samples[idx - 1]
        tb, pb = samples[idx]
        if ta >= int_end:
            break
        dt = tb - ta
        if dt <= 0.0:
            continue
        if tb > int_end:
            frac = (int_end - ta) / dt
            pb = pa + frac * (pb - pa)
            dt = int_end - ta
        q_ucal += _HALF * ((pa - baseline_dp) + (pb - baseline_dp)) * dt
    return q_ucal


def _wiseman_cumulative_bound_heat_ucal(
    cell_vol_l: float,
    m_tot_m: Sequence[float],
    x_tot_m: Sequence[float],
    v_inj_l: Sequence[float],
    kd_m: float,
    n_sites: float,
    delta_h_cal_mol: float,
) -> List[float]:
    """Wiseman et al. (1989) displaced-volume incremental heat `Delta Q_k` (`ucal`) per injection."""
    q_cum_prev = 0.0
    dq_ucal_out: List[float] = []
    for mt, xt, dv in zip(m_tot_m, x_tot_m, v_inj_l):
        eff_m = max(n_sites * mt, _TINY)
        b_term = eff_m + xt + kd_m
        disc = max(b_term * b_term - _FOUR * eff_m * xt, 0.0)
        bound_x = _HALF * (b_term - math.sqrt(disc))
        # Total heat content Q(k) in calories (`cal/mol * mol`), converted to ucal (`* 1e6`)
        q_cum = cell_vol_l * delta_h_cal_mol * bound_x * _MILLION
        # Displaced-volume correction (MicroCal / Wiseman 1989):
        # dQ(k) = Q(k) + (dV_k / (2 * V_0)) * (Q(k) + Q(k - 1)) - Q(k - 1)
        dq = q_cum + (_HALF * dv / max(cell_vol_l, _TINY)) * (q_cum + q_cum_prev) - q_cum_prev
        dq_ucal_out.append(dq)
        q_cum_prev = q_cum
    return dq_ucal_out


def fit_wiseman_itc_isotherm(
    cell_vol_ml: float,
    cell_conc_mm: float,
    syringe_conc_mm: float,
    inj_volumes_ul: Sequence[float],
    inj_heats_ucal: Sequence[float],
    *,
    skip_first_injection: bool = True,
    fixed_stoichiometry_n: Optional[float] = 1.0,
) -> Dict[str, Any]:
    """Fit a 1:1 Wiseman binding isotherm (`K_d`, `DeltaH`, `q_dil`, optional `n`) to integrated heats."""
    n_inj = min(len(inj_volumes_ul), len(inj_heats_ucal))
    if n_inj < 5:
        raise ValueError(f"Wiseman ITC fit requires >= 5 injections, got {n_inj}")
    if cell_vol_ml <= 0.0 or cell_conc_mm <= 0.0 or syringe_conc_mm <= 0.0:
        raise ValueError("cell_vol_ml, cell_conc_mm, and syringe_conc_mm must be positive")

    cell_vol_l = cell_vol_ml / _THOUSAND
    m0_m = cell_conc_mm / _THOUSAND
    x_syr_m = syringe_conc_mm / _THOUSAND

    # Exact MicroCal instant-mixing displaced-volume concentrations after each injection k
    v_inj_l: List[float] = []
    m_tot_m: List[float] = []
    x_tot_m: List[float] = []
    molar_ratio: List[float] = []
    q_kcal_mol_inj: List[float] = []

    cum_dv_ml = 0.0
    for k in range(n_inj):
        dv_ml = float(inj_volumes_ul[k]) / _THOUSAND
        cum_dv_ml += dv_ml
        # MicroCal VP-ITC / Origin second-order displaced-volume dilution factors
        r_vol = cum_dv_ml / cell_vol_ml
        m_k = m0_m * (1.0 - _HALF * r_vol) / (1.0 + _HALF * r_vol)
        x_k = x_syr_m * r_vol * (1.0 - _HALF * r_vol)
        v_l = dv_ml / _THOUSAND
        v_inj_l.append(v_l)
        m_tot_m.append(max(m_k, _TINY))
        x_tot_m.append(max(x_k, 0.0))
        molar_ratio.append(x_k / max(m_k, _TINY))
        moles_inj = max(v_l * x_syr_m, _TINY)
        # ucal / mol -> kcal / mol (`1 ucal = 1e-9 kcal`)
        q_kcal_mol_inj.append((float(inj_heats_ucal[k]) / _MILLION) / (_THOUSAND * moles_inj))

    start_idx = 1 if (skip_first_injection and n_inj >= 6) else 0
    fit_indices = list(range(start_idx, n_inj))
    y_obs = [q_kcal_mol_inj[i] for i in fit_indices]

    def _profile_linear_dh_and_dil(kd_um: float, n_val: float) -> Tuple[float, float, float, List[float]]:
        kd_m = kd_um / _MILLION
        unit_dq_ucal = _wiseman_cumulative_bound_heat_ucal(
            cell_vol_l, m_tot_m, x_tot_m, v_inj_l, kd_m, n_val, 1.0
        )
        # Convert unit `cal/mol` prediction to `kcal/mol of injectant` basis
        basis: List[float] = []
        for i in fit_indices:
            moles_i = max(v_inj_l[i] * x_syr_m, _TINY)
            basis.append((unit_dq_ucal[i] / _MILLION) / moles_i)
        m_pts = float(len(fit_indices))
        sx = sum(basis)
        sy = sum(y_obs)
        sxx = sum(b * b for b in basis)
        sxy = sum(b * y for b, y in zip(basis, y_obs))
        denom = m_pts * sxx - sx * sx
        if abs(denom) <= _TINY:
            q_dil = sy / max(m_pts, 1.0)
            dh_kcal = 0.0
        else:
            dh_kcal = (m_pts * sxy - sx * sy) / denom
            q_dil = (sy - dh_kcal * sx) / m_pts
        preds = [dh_kcal * b + q_dil for b in basis]
        rss = sum((yo - yp) * (yo - yp) for yo, yp in zip(y_obs, preds))
        return rss, dh_kcal, q_dil, preds

    # Multi-start logarithmic grid over K_d (`1 nM` .. `10 mM`) and optional stoichiometry `n`
    n_grid = (
        [float(fixed_stoichiometry_n)]
        if fixed_stoichiometry_n is not None
        else [float(k) / float(10) for k in range(5, 16)]
    )
    best_rss = float("inf")
    best_kd_um = 1.0
    best_n = n_grid[0]

    for i_kd in range(-30, 41):
        kd_cand = pow(float(10), float(i_kd) / float(10))
        for n_cand in n_grid:
            rss, _, _, _ = _profile_linear_dh_and_dil(kd_cand, n_cand)
            if rss < best_rss:
                best_rss = rss
                best_kd_um = kd_cand
                best_n = n_cand

    # Coordinate golden/pattern refinement on log(K_d) (and n if free)
    step_log = 1.0 / float(5)
    step_n = 1.0 / float(20)
    for _ in range(48):
        improved = False
        for d_log in (-step_log, step_log):
            kd_try = best_kd_um * math.exp(d_log)
            if kd_try <= 1.0 / _MILLION or kd_try >= float(10**5):
                continue
            rss, _, _, _ = _profile_linear_dh_and_dil(kd_try, best_n)
            if rss < best_rss:
                best_rss = rss
                best_kd_um = kd_try
                improved = True
        if fixed_stoichiometry_n is None:
            for dn in (-step_n, step_n):
                n_try = best_n + dn
                if n_try < 1.0 / float(5) or n_try > float(4):
                    continue
                rss, _, _, _ = _profile_linear_dh_and_dil(best_kd_um, n_try)
                if rss < best_rss:
                    best_rss = rss
                    best_n = n_try
                    improved = True
        if not improved:
            step_log *= _HALF
            step_n *= _HALF
            if step_log < 1.0 / _MILLION:
                break

    rss_final, dh_kcal_mol, q_dil_kcal_mol, y_hat = _profile_linear_dh_and_dil(best_kd_um, best_n)
    dof = max(len(fit_indices) - (3 if fixed_stoichiometry_n is not None else 4), 1)
    resid_sd = math.sqrt(rss_final / float(dof))
    y_mean = sum(y_obs) / max(float(len(y_obs)), 1.0)
    tss = sum((yo - y_mean) * (yo - y_mean) for yo in y_obs)
    r_squared = max(0.0, 1.0 - rss_final / tss) if tss > _TINY else 0.0
    fitted_span = (max(y_hat) - min(y_hat)) if y_hat else 0.0
    heat_snr = fitted_span / max(resid_sd, 1.0 / _MILLION)
    wiseman_c = (best_n * (cell_conc_mm * _THOUSAND)) / max(best_kd_um, _TINY)

    return {
        "kd_um": best_kd_um,
        "stoichiometry_n": best_n,
        "delta_h_kcal_mol": dh_kcal_mol,
        "q_dil_kcal_mol": q_dil_kcal_mol,
        "wiseman_c": wiseman_c,
        "ligand_max_um": x_tot_m[-1] * _MILLION,
        "cell_conc_um": cell_conc_mm * _THOUSAND,
        "syringe_conc_um": syringe_conc_mm * _THOUSAND,
        "molar_ratio": molar_ratio,
        "q_kcal_mol_injectant": q_kcal_mol_inj,
        "fitted_q_kcal_mol": y_hat,
        "residual_sd_kcal_mol": resid_sd,
        "r_squared": r_squared,
        "heat_snr": heat_snr,
        "n_injections_total": n_inj,
        "n_injections_fitted": len(fit_indices),
    }


def parse_microcal_itc_str(
    text: str,
    *,
    source_name: str = "thermogram.itc",
    expected_partner: Optional[str] = None,
    expected_ligand: Optional[str] = None,
    min_lmax_over_kd: float = _TWO,
    require_binding_signal: bool = True,
    fixed_stoichiometry_n: Optional[float] = 1.0,
) -> Dict[str, Any]:
    """Parse a raw MicroCal `.itc` ASCII thermogram, integrate injection heats, and validate constructs."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].upper().startswith("$ITC"):
        raise ValueError(f"{source_name}: missing '$ITC' MicroCal header")

    dollar_lines: List[str] = []
    hash_vals: List[float] = []
    sample_comments: List[str] = []
    inj_traces: Dict[int, List[Tuple[float, float]]] = {}
    cur_inj: Optional[int] = None

    for ln in lines[1:]:
        c0 = ln[0]
        if c0 == "$":
            dollar_lines.append(ln[1:].strip())
        elif c0 == "#":
            rest = ln[1:].strip()
            try:
                hash_vals.append(float(rest.split()[0]))
            except (ValueError, IndexError):
                pass
        elif c0 == "?":
            sample_comments.append(ln[1:].strip())
        elif c0 == "%":
            continue
        elif c0 == "@":
            tag = ln[1:].strip().split(",")[0].split()[0]
            if tag.isdigit():
                cur_inj = int(tag)
                inj_traces.setdefault(cur_inj, [])
        elif cur_inj is not None:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 2:
                try:
                    t_s = float(parts[0])
                    dp_ucal_s = float(parts[1])
                    inj_traces[cur_inj].append((t_s, dp_ucal_s))
                except ValueError:
                    continue

    if not dollar_lines:
        raise ValueError(f"{source_name}: truncated '$' experiment block in MicroCal .itc file")
    try:
        n_declared_inj = int(float(dollar_lines[0].split()[0]))
    except ValueError as exc:
        raise ValueError(f"{source_name}: invalid injection count {dollar_lines[0]!r}") from exc

    temp_c = float(25)
    inj_schedule: List[Tuple[float, float, float]] = []
    for dl in dollar_lines[1:]:
        if "," in dl:
            toks = [t.strip() for t in dl.split(",")]
            if len(toks) >= 3:
                try:
                    inj_schedule.append((float(toks[0]), float(toks[1]), float(toks[2])))
                except ValueError:
                    pass
    # First numeric scalar after `$NOT` is nominal jacket temperature (`C`)
    for idx, dl in enumerate(dollar_lines[1:], start=1):
        if dl.upper().startswith("NOT") and idx + 1 < len(dollar_lines):
            try:
                temp_c = float(dollar_lines[idx + 1].split()[0])
            except ValueError:
                pass
            break

    # MicroCal `#` block: `# 0`, `# <syringe_mM>`, `# <cell_mM>`, `# <cell_vol_mL>`, ...
    if len(hash_vals) < 4:
        raise ValueError(
            f"{source_name}: MicroCal '#' concentration/volume header requires >= 4 numeric lines, got {len(hash_vals)}"
        )
    syringe_conc_mm = hash_vals[1]
    cell_conc_mm = hash_vals[2]
    cell_vol_ml = hash_vals[3]

    comment_text = " | ".join(sample_comments)
    combined_meta = f"{Path(source_name).name} | {comment_text}".strip(" |")
    construct_check = verify_construct_identity(expected_partner, expected_ligand, combined_meta)

    positive_inj_ids = [k for k in sorted(inj_traces) if k >= 1 and len(inj_traces[k]) >= 4]
    if not positive_inj_ids:
        raise ValueError(f"{source_name}: zero '@1..@N' raw differential-power injection traces found")

    inj_vols_ul: List[float] = []
    inj_heats_ucal: List[float] = []
    for pos_idx, inj_id in enumerate(positive_inj_ids):
        if pos_idx < len(inj_schedule):
            vol_ul, dur_s, _ = inj_schedule[pos_idx]
        elif inj_schedule:
            vol_ul, dur_s, _ = inj_schedule[-1]
        else:
            vol_ul, dur_s = _TWO, _FOUR
        q_ucal = _integrate_single_injection_ucal(inj_traces[inj_id], dur_s)
        inj_vols_ul.append(vol_ul)
        inj_heats_ucal.append(q_ucal)

    fit = fit_wiseman_itc_isotherm(
        cell_vol_ml=cell_vol_ml,
        cell_conc_mm=cell_conc_mm,
        syringe_conc_mm=syringe_conc_mm,
        inj_volumes_ul=inj_vols_ul,
        inj_heats_ucal=inj_heats_ucal,
        fixed_stoichiometry_n=fixed_stoichiometry_n,
    )

    # Detect flat non-calorimetric / heat-silent binding (e.g. entropy-driven or zero-DeltaH buffer)
    max_abs_kcal = max(abs(v) for v in fit["q_kcal_mol_injectant"][1:]) if len(inj_heats_ucal) > 1 else 0.0
    if require_binding_signal and (
        fit["heat_snr"] < float(3)
        or fit["r_squared"] < _HALF
        or max_abs_kcal < 1.0 / float(10)
    ):
        raise NonCalorimetricTitrationError(
            f"{source_name}: non-calorimetric flat thermogram (heat_snr = {fit['heat_snr']:.2f}, "
            f"R^2 = {fit['r_squared']:.3f}, max |Q| = {max_abs_kcal:.4f} kcal/mol)"
        )

    sat_ratio = validate_saturation_coverage(
        fit["ligand_max_um"],
        fit["kd_um"],
        min_lmax_over_kd=min_lmax_over_kd,
        context=source_name,
    )

    return {
        "format": "microcal_itc",
        "source_name": source_name,
        "sample_comment": comment_text,
        "temperature_c": temp_c,
        "declared_injections": n_declared_inj,
        "inj_volumes_ul": inj_vols_ul,
        "inj_heats_ucal": inj_heats_ucal,
        "construct_check": construct_check,
        "lmax_over_kd": sat_ratio,
        **fit,
    }


def parse_microcal_itc_file(
    path: Union[Path, str],
    *,
    expected_partner: Optional[str] = None,
    expected_ligand: Optional[str] = None,
    min_lmax_over_kd: float = _TWO,
    require_binding_signal: bool = True,
    fixed_stoichiometry_n: Optional[float] = 1.0,
) -> Dict[str, Any]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8", errors="replace")
    return parse_microcal_itc_str(
        txt,
        source_name=p.name,
        expected_partner=expected_partner,
        expected_ligand=expected_ligand,
        min_lmax_over_kd=min_lmax_over_kd,
        require_binding_signal=require_binding_signal,
        fixed_stoichiometry_n=fixed_stoichiometry_n,
    )


# ============================================================================
# 2. NanoTemper MO.Affinity `.dat` & CSV/TSV Thermophoresis Parser + Morrison Fit
# ============================================================================


def _morrison_fraction_bound(c_target_um: float, c_ligand_um: float, kd_um: float) -> float:
    ct = max(c_target_um, _TINY)
    b = ct + c_ligand_um + kd_um
    disc = max(b * b - _FOUR * ct * c_ligand_um, 0.0)
    return _HALF * (b - math.sqrt(disc)) / ct


def fit_morrison_mst_isotherm(
    ligand_conc_um: Sequence[float],
    fnorm_permille: Sequence[float],
    *,
    target_conc_um: float = 1.0 / float(20),
) -> Dict[str, Any]:
    """Fit exact 1:1 Morrison quadratic binding isotherm (`K_d`, `f_unbound`, `f_bound`) on MST data."""
    if len(ligand_conc_um) != len(fnorm_permille) or len(ligand_conc_um) < 4:
        raise ValueError(f"Morrison MST fit requires >= 4 experimental points, got {len(ligand_conc_um)}")

    pairs = sorted(
        [(float(c), float(f)) for c, f in zip(ligand_conc_um, fnorm_permille) if c >= 0.0 and math.isfinite(f)],
        key=lambda x: x[0],
    )
    if len(pairs) < 4:
        raise ValueError("fewer than 4 finite non-negative ligand concentrations in MST titration")

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    m_pts = float(len(xs))

    def _profile_linear_endpoints(kd_um: float) -> Tuple[float, float, float, List[float]]:
        phis = [_morrison_fraction_bound(target_conc_um, x, kd_um) for x in xs]
        sp = sum(phis)
        sy = sum(ys)
        spp = sum(p * p for p in phis)
        spy = sum(p * y for p, y in zip(phis, ys))
        denom = m_pts * spp - sp * sp
        if abs(denom) <= _TINY:
            f_u = sy / m_pts
            amp = 0.0
        else:
            amp = (m_pts * spy - sp * sy) / denom
            f_u = (sy - amp * sp) / m_pts
        preds = [f_u + amp * p for p in phis]
        rss = sum((yo - yp) * (yo - yp) for yo, yp in zip(ys, preds))
        return rss, f_u, f_u + amp, preds

    best_rss = float("inf")
    best_kd_um = 1.0
    for i_kd in range(-40, 61):
        kd_cand = pow(float(10), float(i_kd) / float(10))
        rss, _, _, _ = _profile_linear_endpoints(kd_cand)
        if rss < best_rss:
            best_rss = rss
            best_kd_um = kd_cand

    step_log = 1.0 / float(5)
    for _ in range(56):
        improved = False
        for d_log in (-step_log, step_log):
            kd_try = best_kd_um * math.exp(d_log)
            if kd_try <= 1.0 / float(10**6) or kd_try >= float(10**7):
                continue
            rss, _, _, _ = _profile_linear_endpoints(kd_try)
            if rss < best_rss:
                best_rss = rss
                best_kd_um = kd_try
                improved = True
        if not improved:
            step_log *= _HALF
            if step_log < 1.0 / _MILLION:
                break

    rss_final, f_unbound, f_bound, preds = _profile_linear_endpoints(best_kd_um)
    dof = max(len(xs) - 3, 1)
    resid_sd = math.sqrt(rss_final / float(dof))
    amplitude = abs(f_bound - f_unbound)

    return {
        "kd_um": best_kd_um,
        "f_unbound_permille": f_unbound,
        "f_bound_permille": f_bound,
        "amplitude_permille": amplitude,
        "target_conc_um": target_conc_um,
        "ligand_min_um": xs[0],
        "ligand_max_um": xs[-1],
        "experimental_conc_um": xs,
        "experimental_fnorm_permille": ys,
        "fitted_fnorm_permille": preds,
        "residual_sd_permille": resid_sd,
        "snr": amplitude / max(resid_sd, 1.0 / _MILLION),
        "n_experimental_points": len(xs),
    }


def _parse_mst_tabular_csv_tsv(text: str, source_name: str) -> Tuple[List[float], List[float], str]:
    delim = "\t" if "\t" in text.splitlines()[0] else ","
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    header: Optional[List[str]] = None
    conc_col: Optional[int] = None
    fnorm_col: Optional[int] = None
    conc_scale_to_um = 1.0
    meta_comments: List[str] = []
    concs_um: List[float] = []
    fnorms: List[float] = []

    for row in reader:
        if not row or not any(c.strip() for c in row):
            continue
        first = row[0].strip()
        if first.startswith("#"):
            meta_comments.append(first[1:].strip())
            continue
        if header is None:
            header = [c.strip().lower() for c in row]
            for idx, col in enumerate(header):
                if conc_col is None and any(k in col for k in ("conc", "ligand", "titrant", "[l]")):
                    conc_col = idx
                    if "nm" in col:
                        conc_scale_to_um = 1.0 / _THOUSAND
                    elif "mm" in col:
                        conc_scale_to_um = _THOUSAND
                    elif "(m)" in col or col.endswith("_m"):
                        conc_scale_to_um = _MILLION
                if fnorm_col is None and any(k in col for k in ("fnorm", "f_norm", "thermophoresis", "response")):
                    fnorm_col = idx
            if conc_col is None or fnorm_col is None:
                raise ValueError(
                    f"{source_name}: tabular MST export missing concentration or Fnorm column in header {header}"
                )
            continue
        if len(row) <= max(conc_col, fnorm_col):
            continue
        try:
            c_val = float(row[conc_col].strip()) * conc_scale_to_um
            f_val = float(row[fnorm_col].strip())
        except ValueError:
            continue
        if math.isfinite(c_val) and c_val >= 0.0 and math.isfinite(f_val):
            concs_um.append(c_val)
            fnorms.append(f_val)

    return concs_um, fnorms, " | ".join(meta_comments)


def parse_nanotemper_mst_str(
    text: str,
    *,
    source_name: str = "titration.dat",
    expected_partner: Optional[str] = None,
    expected_ligand: Optional[str] = None,
    target_conc_um: float = 1.0 / float(20),
    dat_conc_unit: str = "nM",
    min_lmax_over_kd: float = _TWO,
) -> Dict[str, Any]:
    """Parse NanoTemper MO.Affinity `.dat` or CSV/TSV export, fit Morrison `K_d`, and enforce guards."""
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not raw_lines:
        raise ValueError(f"{source_name}: empty NanoTemper MST file")

    non_hash = [ln for ln in raw_lines if not ln.startswith("#")]
    looks_tabular = bool(non_hash) and any(
        any(k in non_hash[0].lower() for k in ("conc", "ligand", "fnorm", "f_norm"))
        for _ in (0,)
    )

    header_nums: List[float] = []
    meta_comments: List[str] = []
    exp_concs_um: List[float] = []
    exp_fnorms: List[float] = []
    exported_curve_pts: int = 0

    if looks_tabular:
        exp_concs_um, exp_fnorms, cmt = _parse_mst_tabular_csv_tsv(text, source_name)
        if cmt:
            meta_comments.append(cmt)
        fmt_label = "nanotemper_mo_table"
    else:
        unit_low = dat_conc_unit.strip().lower()
        if unit_low == "nm":
            unit_to_um = 1.0 / _THOUSAND
        elif unit_low in ("um", "micromolar"):
            unit_to_um = 1.0
        elif unit_low == "mm":
            unit_to_um = _THOUSAND
        elif unit_low == "m":
            unit_to_um = _MILLION
        else:
            raise ValueError(f"unsupported dat_conc_unit {dat_conc_unit!r}")

        section = "experimental"
        for ln in raw_lines:
            if ln.startswith("#"):
                body = ln[1:].strip()
                try:
                    header_nums.append(float(body))
                except ValueError:
                    if body:
                        meta_comments.append(body)
                continue
            parts = ln.split()
            if parts == ["state", "0"]:
                section = "experimental"
                continue
            if parts == ["state", "1"]:
                section = "exported_curve"
                continue
            if len(parts) >= 2:
                try:
                    x_raw = float(parts[0])
                    y_raw = float(parts[1])
                except ValueError:
                    continue
                if x_raw <= _SENTINEL_DAT and y_raw <= _SENTINEL_DAT:
                    section = "post_sentinel"
                    continue
                if section == "experimental":
                    exp_concs_um.append(x_raw * unit_to_um)
                    exp_fnorms.append(y_raw)
                elif section == "exported_curve":
                    exported_curve_pts += 1
        fmt_label = "nanotemper_mo_dat"

    if len(exp_concs_um) < 4:
        raise BiophysicalValidationError(
            f"{source_name}: requires >= 4 raw experimental MST dose-response points "
            f"(got {len(exp_concs_um)}; exported curve points={exported_curve_pts} are never accepted without raw points)"
        )

    combined_meta = f"{Path(source_name).name} | {' | '.join(meta_comments)}".strip(" |")
    construct_check = verify_construct_identity(expected_partner, expected_ligand, combined_meta)

    fit = fit_morrison_mst_isotherm(
        exp_concs_um,
        exp_fnorms,
        target_conc_um=target_conc_um,
    )
    sat_ratio = validate_saturation_coverage(
        fit["ligand_max_um"],
        fit["kd_um"],
        min_lmax_over_kd=min_lmax_over_kd,
        context=source_name,
    )

    cold_window_s = (header_nums[0], header_nums[1]) if len(header_nums) >= 2 else None
    hot_window_s = (header_nums[2], header_nums[3]) if len(header_nums) >= 4 else None

    return {
        "format": fmt_label,
        "source_name": source_name,
        "cold_window_s": cold_window_s,
        "hot_window_s": hot_window_s,
        "exported_curve_points_ignored": exported_curve_pts,
        "construct_check": construct_check,
        "lmax_over_kd": sat_ratio,
        **fit,
    }


def parse_nanotemper_mst_file(
    path: Union[Path, str],
    *,
    expected_partner: Optional[str] = None,
    expected_ligand: Optional[str] = None,
    target_conc_um: float = 1.0 / float(20),
    dat_conc_unit: str = "nM",
    min_lmax_over_kd: float = _TWO,
) -> Dict[str, Any]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8", errors="replace")
    return parse_nanotemper_mst_str(
        txt,
        source_name=p.name,
        expected_partner=expected_partner,
        expected_ligand=expected_ligand,
        target_conc_um=target_conc_um,
        dat_conc_unit=dat_conc_unit,
        min_lmax_over_kd=min_lmax_over_kd,
    )


# ============================================================================
# Self-Test Suite (`python3 scripts/p17_adapters/f28_biophysical_parsers.py --self-test`)
# ============================================================================


def _synthesize_microcal_itc_text(
    *,
    comment: str,
    cell_conc_mm: float,
    syringe_conc_mm: float,
    cell_vol_ml: float,
    kd_um: float,
    delta_h_kcal_mol: float,
    q_dil_kcal_mol: float,
    n_injections: int = 18,
    inj_vol_ul: float = _TWO,
) -> str:
    cell_vol_l = cell_vol_ml / _THOUSAND
    m0_m = cell_conc_mm / _THOUSAND
    x_syr_m = syringe_conc_mm / _THOUSAND
    kd_m = kd_um / _MILLION
    dh_cal_mol = delta_h_kcal_mol * _THOUSAND

    v_inj_l: List[float] = []
    m_tot_m: List[float] = []
    x_tot_m: List[float] = []
    cum_dv_ml = 0.0
    for _ in range(n_injections):
        dv_ml = inj_vol_ul / _THOUSAND
        cum_dv_ml += dv_ml
        r_vol = cum_dv_ml / cell_vol_ml
        m_k = m0_m * (1.0 - _HALF * r_vol) / (1.0 + _HALF * r_vol)
        x_k = x_syr_m * r_vol * (1.0 - _HALF * r_vol)
        v_inj_l.append(dv_ml / _THOUSAND)
        m_tot_m.append(m_k)
        x_tot_m.append(x_k)

    dq_bound_ucal = _wiseman_cumulative_bound_heat_ucal(
        cell_vol_l, m_tot_m, x_tot_m, v_inj_l, kd_m, 1.0, dh_cal_mol
    )

    out: List[str] = ["$ITC", f"$ {n_injections}", "$NOT", "$ 25", "$ 60", "$ 300", "$ 5"]
    for _ in range(n_injections):
        out.append(f"$ {inj_vol_ul:.1f} , 4 , 120 , 2")
    out.extend(
        [
            "# 0",
            f"# {syringe_conc_mm:.6f}",
            f"# {cell_conc_mm:.6f}",
            f"# {cell_vol_ml:.6f}",
            "# 25.0",
            f"?{comment}",
            "@0",
            "0.0,1.000,25.0",
            "10.0,1.000,25.0",
        ]
    )
    sigma_s = float(3)
    sqrt_2pi = math.sqrt(_TWO * math.pi)
    for k in range(n_injections):
        moles_inj = v_inj_l[k] * x_syr_m
        q_dil_ucal = q_dil_kcal_mol * _THOUSAND * moles_inj * _MILLION
        q_tot_ucal = dq_bound_ucal[k] + q_dil_ucal
        out.append(f"@{k + 1}")
        t_base = float((k + 1) * 120)
        for step in range(121):
            dt = float(step)
            # Normalized Gaussian thermal power pulse centered at dt = 12 s on a 1.0 ucal/s baseline
            z = (dt - float(12)) / sigma_s
            dp = 1.0 + (q_tot_ucal / (sigma_s * sqrt_2pi)) * math.exp(-_HALF * z * z)
            out.append(f"{t_base + dt:.1f},{dp:.8f},25.0")
    return "\n".join(out) + "\n"


def _synthesize_nanotemper_dat_text(
    *,
    kd_um: float,
    max_ligand_um: float,
    target_conc_um: float,
    f_unbound: float,
    f_bound: float,
    n_points: int = 16,
    include_exp_points: bool = True,
) -> str:
    lines = ["#-1", "#0", "#4", "#5", "#0", f"#{f_unbound:.3f}", "state 0"]
    if include_exp_points:
        for i in range(n_points):
            # 2-fold serial dilution from max_ligand_um downward, written in nM
            conc_um = max_ligand_um / pow(_TWO, float(n_points - 1 - i))
            phi = _morrison_fraction_bound(target_conc_um, conc_um, kd_um)
            fnorm = f_unbound + (f_bound - f_unbound) * phi
            lines.append(f"{conc_um * _THOUSAND:.6f}\t{fnorm:.6f}")
    lines.append("-666.666\t-666.666")
    lines.append("state 1")
    for i in range(20):
        c_nm = pow(float(10), float(i) / float(3))
        lines.append(f"{c_nm:.4f}\t{f_unbound:.4f}")
    return "\n".join(lines) + "\n"


def run_self_tests() -> int:
    # 1. Synthetic MicroCal .itc thermogram integration + Wiseman 1:1 recovery
    true_kd_itc = float(8)
    true_dh_itc = -float(10)
    itc_txt = _synthesize_microcal_itc_text(
        comment="20 uM B1 in cell, 300 uM tau1-243 in syringe, HEPES buffer",
        cell_conc_mm=float(20) / _THOUSAND,
        syringe_conc_mm=float(300) / _THOUSAND,
        cell_vol_ml=float(206) / _THOUSAND,
        kd_um=true_kd_itc,
        delta_h_kcal_mol=true_dh_itc,
        q_dil_kcal_mol=-1.0 / float(5),
    )
    itc_res = parse_microcal_itc_str(
        itc_txt,
        source_name="20uMB1_cell_300uM_tau1-243.itc",
        expected_partner="B1",
        expected_ligand="tau(1-243)",
    )
    assert abs(itc_res["kd_um"] - true_kd_itc) / true_kd_itc < 1.0 / float(10), itc_res
    assert abs(itc_res["delta_h_kcal_mol"] - true_dh_itc) / abs(true_dh_itc) < 1.0 / float(10), itc_res
    assert itc_res["lmax_over_kd"] >= _TWO

    # 2. Non-calorimetric flat .itc thermogram MUST raise NonCalorimetricTitrationError
    flat_itc = _synthesize_microcal_itc_text(
        comment="20 uM B1 in cell, 200 uM tau1-243 in syringe",
        cell_conc_mm=float(20) / _THOUSAND,
        syringe_conc_mm=float(200) / _THOUSAND,
        cell_vol_ml=float(206) / _THOUSAND,
        kd_um=float(10),
        delta_h_kcal_mol=0.0,
        q_dil_kcal_mol=0.0,
    )
    caught_flat = False
    try:
        parse_microcal_itc_str(flat_itc, source_name="flat_tau1-243.itc")
    except NonCalorimetricTitrationError:
        caught_flat = True
    assert caught_flat, "expected NonCalorimetricTitrationError on zero-DeltaH thermogram"

    # 3. Synthetic NanoTemper MO.Affinity .dat Morrison 1:1 recovery
    true_kd_mst = float(12)
    dat_txt = _synthesize_nanotemper_dat_text(
        kd_um=true_kd_mst,
        max_ligand_um=float(120),
        target_conc_um=1.0 / float(20),
        f_unbound=float(760),
        f_bound=float(792),
    )
    mst_res = parse_nanotemper_mst_str(
        dat_txt,
        source_name="B1_labeled_tau1-243_titrated.dat",
        expected_partner="B1",
        expected_ligand="tau(1-243)",
    )
    assert abs(mst_res["kd_um"] - true_kd_mst) / true_kd_mst < 1.0 / float(20), mst_res
    assert mst_res["lmax_over_kd"] >= _TWO

    # 4. Tabular CSV MO.Affinity export recovery
    csv_lines = ["# B1 + tau1-243 MST", "ligand_conc_uM,fnorm_permille"]
    for i in range(14):
        c_um = float(100) / pow(_TWO, float(13 - i))
        phi = _morrison_fraction_bound(1.0 / float(20), c_um, float(5))
        csv_lines.append(f"{c_um:.6f},{float(750) + float(25) * phi:.6f}")
    csv_res = parse_nanotemper_mst_str(
        "\n".join(csv_lines) + "\n",
        source_name="B1_tau1-243_export.csv",
        expected_partner="B1",
        expected_ligand="tau(1-243)",
    )
    assert abs(csv_res["kd_um"] - float(5)) / float(5) < 1.0 / float(20), csv_res

    # 5. ConstructMismatchError on truncated `tau(1-243)` vs full-length `WT 2N4R` / `FLWTtau` & `S320F` mutant
    for bad_name in (
        "B1_labeled_WT_tau_titrated.dat",
        "20uMB1_100uM_FLWTtau_2N4R.dat",
        "B1_labeled_tau1-243_S320F_titrated.dat",
    ):
        caught_cm = False
        try:
            parse_nanotemper_mst_str(
                dat_txt,
                source_name=bad_name,
                expected_partner="B1",
                expected_ligand="tau(1-243)",
            )
        except ConstructMismatchError:
            caught_cm = True
        assert caught_cm, f"expected ConstructMismatchError for {bad_name}"

    # 6. InsufficientSaturationError when `[L]_max < 2 * K_d` (`[L]_max = 100 uM`, `K_d = 227 uM`)
    subsat_dat = _synthesize_nanotemper_dat_text(
        kd_um=float(227),
        max_ligand_um=float(100),
        target_conc_um=1.0 / float(20),
        f_unbound=float(764),
        f_bound=float(840),
    )
    caught_subsat = False
    try:
        parse_nanotemper_mst_str(
            subsat_dat,
            source_name="B1_labeled_WT_2N4R_tau.dat",
            expected_partner="B1",
            expected_ligand="2N4R",
        )
    except InsufficientSaturationError:
        caught_subsat = True
    assert caught_subsat, "expected InsufficientSaturationError when [L]_max = 100 uM < 2 * 227 uM"

    # 7. Curve-only `.dat` export with 0 raw `state 0` experimental points MUST fail closed
    curve_only_dat = _synthesize_nanotemper_dat_text(
        kd_um=float(10),
        max_ligand_um=float(100),
        target_conc_um=1.0 / float(20),
        f_unbound=float(760),
        f_bound=float(790),
        include_exp_points=False,
    )
    caught_curve_only = False
    try:
        parse_nanotemper_mst_str(curve_only_dat, source_name="curve_only.dat")
    except BiophysicalValidationError:
        caught_curve_only = True
    assert caught_curve_only, "expected BiophysicalValidationError on curve-only .dat without raw points"

    print("OK [f28_biophysical_parsers --self-test]: 7/7 MicroCal .itc & NanoTemper MST parser/guard tests passed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Clean-room MicroCal .itc and NanoTemper MO.Affinity .dat/CSV biophysical parser & validator (F28)"
    )
    ap.add_argument("--itc", type=Path, help="Path to raw MicroCal .itc thermogram")
    ap.add_argument("--mst", type=Path, help="Path to NanoTemper MO.Affinity .dat or CSV/TSV file")
    ap.add_argument("--expected-partner", help="Expected cell/labeled macromolecule construct token (e.g. B1)")
    ap.add_argument("--expected-ligand", help="Expected syringe/titrant construct token (e.g. tau(1-243) or 2N4R)")
    ap.add_argument("--target-conc-um", type=float, default=1.0 / float(20), help="Labeled target conc (uM) for MST")
    ap.add_argument("--min-lmax-over-kd", type=float, default=_TWO, help="Minimum [L]_max / K_d saturation ratio")
    ap.add_argument("--self-test", action="store_true", help="Run built-in unit & adversarial validation suite")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_tests()
    if args.itc is not None:
        res = parse_microcal_itc_file(
            args.itc,
            expected_partner=args.expected_partner,
            expected_ligand=args.expected_ligand,
            min_lmax_over_kd=args.min_lmax_over_kd,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    if args.mst is not None:
        res = parse_nanotemper_mst_file(
            args.mst,
            expected_partner=args.expected_partner,
            expected_ligand=args.expected_ligand,
            target_conc_um=args.target_conc_um,
            min_lmax_over_kd=args.min_lmax_over_kd,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    ap.error("specify --self-test, --itc <file.itc>, or --mst <file.dat|csv>")
    return 2


if __name__ == "__main__":
    sys.exit(main())
