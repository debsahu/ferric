//! Ligand titration 2D `HSQC` fast-exchange peak trajectory tracking (`O(N^3)` Hungarian /
//! Kuhn–Munkres smooth assignment), combined amide chemical shift perturbation (`CSP`),
//! Williamson (2013) iterative `2-sigma` binding-site threshold detection, dilution-aware `[P]_t`
//! correction, and per-residue + global exact quadratic `1:1` tight-binding `K_d` isotherm
//! fitting with Monte Carlo confidence intervals and TSV/JSON export (`T13.6`).
//!
//! Provenance: Clean-room original implementation grounded in peer-reviewed spectroscopy and
//! combinatorial optimization literature:
//! - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2013, 73, 1–16 — measuring ligand
//!   binding affinities and mapping protein interfaces from chemical shift perturbations,
//!   `alpha_N = 0.14` / `0.20` for `Gly`, iterative `2-sigma` threshold trimming, and exact
//!   quadratic tight-binding `K_d` isotherm).
//! - Fielding, L. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2007, 51, 219–242 — NMR determination of
//!   protein–ligand dissociation constants and stoichiometry/dilution effects).
//! - Kuhn, H. W. (*Naval Res. Logist. Quart.* 1955, 2, 83–97) & Munkres, J. (*J. Soc. Indust.
//!   Appl. Math.* 1957, 5, 32–38 — polynomial minimum-cost bipartite matching).

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

/// Canonical `15N`-to-`1H` chemical shift scaling factor `alpha_N` per Williamson (*Prog. Nucl.
/// Magn. Reson. Spectrosc.* 2013): `0.20` for Glycine and `0.14` for all other residues.
#[inline]
pub fn amide_alpha_n(res_name: &str) -> f64 {
    let r = res_name.trim();
    if r.eq_ignore_ascii_case("GLY") || r.eq_ignore_ascii_case("G") {
        0.20
    } else {
        0.14
    }
}

/// Combined amide chemical shift perturbation `Delta_delta_comb = sqrt(dH^2 + (alpha_N * dN)^2)` (`ppm`).
#[inline]
pub fn combined_amide_csp_ppm(dh_ppm: f64, dn_ppm: f64, res_name: &str) -> f64 {
    let alpha = amide_alpha_n(res_name);
    dh_ppm.hypot(alpha * dn_ppm)
}

/// Exact dilution-corrected `[P]_t` (`uM`) and `[L]_t` (`uM`) when adding `v_added_ul` of ligand
/// stock (`ligand_stock_um`) into an initial apo sample volume `v0_ul` (`initial_protein_um`).
#[inline]
pub fn diluted_conc_from_stock_addition(
    initial_protein_um: f64,
    v0_ul: f64,
    ligand_stock_um: f64,
    v_added_ul: f64,
) -> Result<(f64, f64), String> {
    if !v0_ul.is_finite() || v0_ul <= 0.0 || !v_added_ul.is_finite() || v_added_ul < 0.0 {
        return Err("invalid titration sample/addition volume".to_string());
    }
    if !initial_protein_um.is_finite()
        || initial_protein_um < 0.0
        || !ligand_stock_um.is_finite()
        || ligand_stock_um < 0.0
    {
        return Err("invalid protein or ligand stock concentration".to_string());
    }
    let v_tot = v0_ul + v_added_ul;
    let p_t = initial_protein_um * (v0_ul / v_tot);
    let l_t = ligand_stock_um * (v_added_ul / v_tot);
    Ok((p_t, l_t))
}

/// Dilution-corrected `[P]_t` (`uM`) as a function of target final `[L]_t` (`uM`) added from a
/// concentrated stock of `ligand_stock_um`: `[P]_t = [P]_0 * (1 - [L]_t / [L]_stock)`.
#[inline]
pub fn diluted_protein_conc_for_target_ligand(
    initial_protein_um: f64,
    ligand_stock_um: f64,
    target_ligand_um: f64,
) -> Result<f64, String> {
    if !ligand_stock_um.is_finite() || ligand_stock_um <= 0.0 {
        return Err("ligand_stock_um must be > 0".to_string());
    }
    if target_ligand_um < 0.0 || target_ligand_um >= ligand_stock_um {
        return Err(format!(
            "target_ligand_um ({target_ligand_um}) must satisfy 0 <= [L]_t < ligand_stock_um ({ligand_stock_um})"
        ));
    }
    Ok(initial_protein_um * (1.0 - target_ligand_um / ligand_stock_um))
}

/// Assigned apo `2D 1H-15N HSQC` cross-peak (`[L]_t = 0`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssignedAmidePeak {
    /// `0`-based sequence index (`mol3d_overlay::QuantitativeTrackData` key).
    pub seq_index: usize,
    /// `1`-based or author sequence number (`seq_id`).
    pub seq_id: i32,
    pub res_name: String,
    pub h_ppm: f64,
    pub n_ppm: f64,
}

/// Unassigned `2D 1H-15N HSQC` peak coordinates at a titration point.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnassignedHsqcPeak {
    pub h_ppm: f64,
    pub n_ppm: f64,
}

/// One ligand titration spectrum step with unassigned `2D HSQC` cross-peaks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnassignedTitrationStep {
    pub label: String,
    /// Total protein concentration `[P]_t` (`uM`) at this step (after any stock dilution).
    pub protein_conc_um: f64,
    /// Total ligand concentration `[L]_t` (`uM`) at this step.
    pub ligand_conc_um: f64,
    pub peaks: Vec<UnassignedHsqcPeak>,
}

/// Configuration for Hungarian smooth trajectory tracking and `K_d` fitting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CspAnalysisConfig {
    /// Maximum allowed single-step combined shift jump (`ppm`) before treating a peak as broadened/missing.
    pub max_step_csp_ppm: f64,
    /// Weight penalizing perpendicular deviation from a linear fast-exchange shift vector.
    pub collinearity_penalty: f64,
    /// Uniform `1-sigma` combined `CSP` measurement uncertainty (`ppm`, e.g. `0.004`).
    pub csp_sigma_ppm: f64,
    /// Number of parametric Monte Carlo replicates for `95%` confidence intervals (`K_d` & `ddelta_max`).
    pub n_mc: usize,
    /// Deterministic PRNG seed for Monte Carlo confidence intervals.
    pub mc_seed: u64,
}

impl Default for CspAnalysisConfig {
    fn default() -> Self {
        Self {
            max_step_csp_ppm: 0.45,
            collinearity_penalty: 0.35,
            csp_sigma_ppm: 0.004,
            n_mc: 400,
            mc_seed: 20261306,
        }
    }
}

/// Tracked titration point for a single residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueTitrationPoint {
    pub protein_conc_um: f64,
    pub ligand_conc_um: f64,
    /// Observed `(h_ppm, n_ppm)` if peak was matched (`None` if exchange-broadened / missing).
    pub matched_peak: Option<(f64, f64)>,
    /// Combined shift perturbation `Delta_delta_comb` (`ppm`) relative to apo (`None` if unmatched).
    pub csp_ppm: Option<f64>,
}

/// Tracked titration trajectory across all `[L]_t` points for one residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueCspTrajectory {
    pub seq_index: usize,
    pub seq_id: i32,
    pub res_name: String,
    pub apo_h_ppm: f64,
    pub apo_n_ppm: f64,
    pub points: Vec<ResidueTitrationPoint>,
    /// Maximum observed `Delta_delta_comb` (`ppm`) across the titration.
    pub max_observed_csp_ppm: f64,
}

/// Williamson (2013) iterative `2-sigma`-trimmed binding-site detection summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrimmedCspThresholds {
    pub baseline_mean_ppm: f64,
    pub baseline_sd_ppm: f64,
    pub threshold_1sigma_ppm: f64,
    pub threshold_2sigma_ppm: f64,
    pub iterations: usize,
    pub significant_seq_ids: Vec<i32>,
}

/// Single-residue `1:1` isotherm fit result (exact quadratic tight-binding + naive MM comparison).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueKdFit {
    pub seq_index: usize,
    pub seq_id: i32,
    pub res_name: String,
    pub max_observed_csp_ppm: f64,
    pub significant_binding_site: bool,
    pub kd_um: f64,
    pub kd_sd_um: f64,
    pub kd_ci95_low_um: f64,
    pub kd_ci95_high_um: f64,
    pub ddelta_max_ppm: f64,
    pub ddelta_max_sd_ppm: f64,
    /// Naive Michaelis–Menten (`[L]_free ≈ [L]_t`) apparent `K_d` (`uM`) for comparison.
    pub naive_mm_kd_um: f64,
    pub chi2: f64,
    pub n_points: usize,
}

/// Complete global + per-residue ligand titration `CSP` & tight-binding `K_d` analysis report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CspGlobalFitResult {
    /// Global exact quadratic `1:1` tight-binding dissociation constant `K_d` (`uM`).
    pub global_kd_um: f64,
    pub global_kd_sd_um: f64,
    pub global_kd_ci95_low_um: f64,
    pub global_kd_ci95_high_um: f64,
    /// Global naive Michaelis–Menten (`[L]_free ≈ [L]_t`) `K_d` (`uM`), exposing depletion bias
    /// when `K_d <= [P]_t`.
    pub global_naive_mm_kd_um: f64,
    pub global_chi2: f64,
    pub global_reduced_chi2: f64,
    pub global_dof: usize,
    pub thresholds: TrimmedCspThresholds,
    pub trajectories: Vec<ResidueCspTrajectory>,
    /// Per-residue `ddelta_max` at the global `K_d` (`seq_id -> ddelta_max_ppm`) plus individual fits.
    pub residue_fits: Vec<ResidueKdFit>,
    pub global_ddelta_max_by_seq_id: Vec<(i32, f64, f64)>,
}

// ============================================================================
// 1. Exact O(N^3) Kuhn–Munkres / Hungarian Minimum-Cost Smooth Peak Tracker
// ============================================================================

/// Solve the minimum-cost rectangular assignment problem (`n_rows <= n_cols`) in exact
/// `O(n_rows^2 * n_cols)` time using potentials `(u, v)` and shortest augmenting paths
/// (Kuhn 1955; Munkres 1957; Jonker & Volgenant 1987).
///
/// Returns `row_to_col` of length `n_rows` (`0`-based column indices) and total minimum cost.
pub fn hungarian_min_cost_assignment(cost: &[Vec<f64>]) -> Result<(Vec<usize>, f64), String> {
    let n = cost.len();
    if n == 0 {
        return Ok((Vec::new(), 0.0));
    }
    let m = cost[0].len();
    if m < n {
        return Err(format!(
            "hungarian_min_cost_assignment requires n_cols ({m}) >= n_rows ({n})"
        ));
    }
    for (r, row) in cost.iter().enumerate() {
        if row.len() != m {
            return Err(format!("row {r} length {} != {m}", row.len()));
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(format!("non-finite cost in row {r}"));
        }
    }

    // 1-indexed arrays `u[0..=n]`, `v[0..=m]`, `p[0..=m]` (col -> matched row), `way[0..=m]`
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];

        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;

            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut row_to_col = vec![0usize; n];
    let mut total_cost = 0.0;
    for j in 1..=m {
        if p[j] > 0 {
            let r = p[j] - 1;
            row_to_col[r] = j - 1;
            total_cost += cost[r][j - 1];
        }
    }
    Ok((row_to_col, total_cost))
}

/// Track fast-exchange `2D HSQC` peaks across ligand titration steps sorted by `[L]_t`, combining
/// directional linear-trajectory extrapolation with exact Hungarian global minimum-cost matching.
pub fn track_titration_trajectories(
    apo_peaks: &[AssignedAmidePeak],
    steps: &[UnassignedTitrationStep],
    config: &CspAnalysisConfig,
) -> Result<Vec<ResidueCspTrajectory>, String> {
    if apo_peaks.is_empty() {
        return Err(
            "track_titration_trajectories requires at least one assigned apo peak".to_string(),
        );
    }

    let mut sorted_steps: Vec<&UnassignedTitrationStep> = steps.iter().collect();
    sorted_steps.sort_by(|a, b| a.ligand_conc_um.total_cmp(&b.ligand_conc_um));

    let n_res = apo_peaks.len();
    let mut trajectories: Vec<ResidueCspTrajectory> = apo_peaks
        .iter()
        .map(|ap| ResidueCspTrajectory {
            seq_index: ap.seq_index,
            seq_id: ap.seq_id,
            res_name: ap.res_name.clone(),
            apo_h_ppm: ap.h_ppm,
            apo_n_ppm: ap.n_ppm,
            points: vec![ResidueTitrationPoint {
                protein_conc_um: sorted_steps
                    .first()
                    .map(|s| s.protein_conc_um)
                    .unwrap_or(100.0),
                ligand_conc_um: 0.0,
                matched_peak: Some((ap.h_ppm, ap.n_ppm)),
                csp_ppm: Some(0.0),
            }],
            max_observed_csp_ppm: 0.0,
        })
        .collect();

    // Maintain last matched `(h, n, l_conc)` and previous `(h, n, l_conc)` for smooth linear prediction
    let mut last_pos: Vec<(f64, f64, f64)> = apo_peaks
        .iter()
        .map(|ap| (ap.h_ppm, ap.n_ppm, 0.0))
        .collect();
    let mut prev_pos: Vec<Option<(f64, f64, f64)>> = vec![None; n_res];

    let max_gate_cost = config.max_step_csp_ppm.max(0.05).powi(2);

    for step in sorted_steps {
        if !step.ligand_conc_um.is_finite() || step.ligand_conc_um < 0.0 {
            return Err(format!(
                "titration step '{}' has invalid negative or non-finite ligand_conc_um ({})",
                step.label, step.ligand_conc_um
            ));
        }
        // If caller explicitly passed an apo `[L]_t == 0` spectrum in `steps`, merge its
        // `[P]_t` into the initial apo point instead of double-counting `[L]_t = 0`.
        if step.ligand_conc_um <= 1e-12 {
            if step.protein_conc_um.is_finite() && step.protein_conc_um > 0.0 {
                for traj in &mut trajectories {
                    traj.points[0].protein_conc_um = step.protein_conc_um;
                }
            }
            continue;
        }

        let n_peaks = step.peaks.len();
        // Columns `0..n_peaks` = observed HSQC peaks; columns `n_peaks .. n_peaks + n_res` = slack unmatched slots
        let n_cols = n_peaks + n_res;
        let mut cost = vec![vec![max_gate_cost * 4.0; n_cols]; n_res];

        for r in 0..n_res {
            let alpha = amide_alpha_n(&apo_peaks[r].res_name);
            let (lh, ln, ll) = last_pos[r];

            // Extrapolate expected position along fast-exchange chord with damped saturation cap
            let (pred_h, pred_n) = if let Some((ph, pn, pl)) = prev_pos[r] {
                let dl_prev = (ll - pl).max(1e-6);
                let dl_cur = (step.ligand_conc_um - ll).max(0.0);
                // Bound saturation damping: chord slope decays as [L]_t grows, cap gain <= 0.85
                let gain = (dl_cur / dl_prev).clamp(0.0, 0.85);
                (lh + gain * (lh - ph), ln + gain * (ln - pn))
            } else {
                (lh, ln)
            };

            // Established chord unit vector from apo in `(dH, alpha * dN)` space
            let chord_x = lh - apo_peaks[r].h_ppm;
            let chord_y = alpha * (ln - apo_peaks[r].n_ppm);
            let chord_len = chord_x.hypot(chord_y);

            for (c, pk) in step.peaks.iter().enumerate() {
                let dx_pred = pk.h_ppm - pred_h;
                let dy_pred = alpha * (pk.n_ppm - pred_n);
                let mut c_val = dx_pred * dx_pred + dy_pred * dy_pred;

                // Also penalize backward motion or perpendicular kink when chord is already established
                if chord_len > 0.015 {
                    let ux = chord_x / chord_len;
                    let uy = chord_y / chord_len;
                    let tot_x = pk.h_ppm - apo_peaks[r].h_ppm;
                    let tot_y = alpha * (pk.n_ppm - apo_peaks[r].n_ppm);
                    let perp = tot_x * uy - tot_y * ux;
                    let step_along = (pk.h_ppm - lh) * ux + alpha * (pk.n_ppm - ln) * uy;
                    c_val += config.collinearity_penalty * perp * perp;
                    if step_along < -0.01 {
                        c_val += 2.0 * step_along * step_along;
                    }
                }
                cost[r][c] = c_val;
            }

            // Dedicated slack column for residue `r`
            cost[r][n_peaks + r] = max_gate_cost;
        }

        let (assignment, _) = hungarian_min_cost_assignment(&cost)?;
        for (r, &col) in assignment.iter().enumerate() {
            if col < n_peaks && cost[r][col] <= max_gate_cost {
                let pk = step.peaks[col];
                let csp = combined_amide_csp_ppm(
                    pk.h_ppm - apo_peaks[r].h_ppm,
                    pk.n_ppm - apo_peaks[r].n_ppm,
                    &apo_peaks[r].res_name,
                );
                prev_pos[r] = Some(last_pos[r]);
                last_pos[r] = (pk.h_ppm, pk.n_ppm, step.ligand_conc_um);
                trajectories[r].max_observed_csp_ppm =
                    trajectories[r].max_observed_csp_ppm.max(csp);
                trajectories[r].points.push(ResidueTitrationPoint {
                    protein_conc_um: step.protein_conc_um,
                    ligand_conc_um: step.ligand_conc_um,
                    matched_peak: Some((pk.h_ppm, pk.n_ppm)),
                    csp_ppm: Some(csp),
                });
            } else {
                trajectories[r].points.push(ResidueTitrationPoint {
                    protein_conc_um: step.protein_conc_um,
                    ligand_conc_um: step.ligand_conc_um,
                    matched_peak: None,
                    csp_ppm: None,
                });
            }
        }
    }

    Ok(trajectories)
}

// ============================================================================
// 2. Williamson (2013) Iterative 2-Sigma Trimmed Binding-Site Detection
// ============================================================================

/// Compute iterative `2-sigma`-trimmed baseline statistics and binding-site residue calls
/// (`Delta_delta_comb >= mu_trim + 2 * sigma_trim`, Williamson *Prog. NMR Spectrosc.* 2013, §3.1).
pub fn compute_williamson_2sigma_threshold(seq_ids_and_csp: &[(i32, f64)]) -> TrimmedCspThresholds {
    if seq_ids_and_csp.is_empty() {
        return TrimmedCspThresholds {
            baseline_mean_ppm: 0.0,
            baseline_sd_ppm: 0.0,
            threshold_1sigma_ppm: 0.0,
            threshold_2sigma_ppm: 0.0,
            iterations: 0,
            significant_seq_ids: Vec::new(),
        };
    }

    let min_keep = (seq_ids_and_csp.len() / 2)
        .max(3)
        .min(seq_ids_and_csp.len());
    let mut active: Vec<f64> = seq_ids_and_csp.iter().map(|&(_, v)| v.max(0.0)).collect();
    active.sort_by(|a, b| a.total_cmp(b));

    let mut mean = 0.0;
    let mut sd = 0.0;
    let mut iters = 0usize;

    for iter in 1..=25 {
        iters = iter;
        let n = active.len() as f64;
        mean = active.iter().sum::<f64>() / n;
        let var = if active.len() > 1 {
            active.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };
        sd = var.sqrt();
        let cutoff = mean + 2.0 * sd;

        if active.len() <= min_keep {
            break;
        }
        let next_len = active
            .iter()
            .filter(|&&v| v <= cutoff)
            .count()
            .max(min_keep);
        if next_len == active.len() {
            break;
        }
        active.truncate(next_len);
    }

    let t1 = mean + sd;
    let t2 = mean + 2.0 * sd;
    let sig_ids: Vec<i32> = seq_ids_and_csp
        .iter()
        .filter(|&&(_, v)| v >= t2 && v > 0.0)
        .map(|&(id, _)| id)
        .collect();

    TrimmedCspThresholds {
        baseline_mean_ppm: mean,
        baseline_sd_ppm: sd,
        threshold_1sigma_ppm: t1,
        threshold_2sigma_ppm: t2,
        iterations: iters,
        significant_seq_ids: sig_ids,
    }
}

// ============================================================================
// 3. Exact Quadratic 1:1 Tight-Binding & Naive Michaelis–Menten Isotherms
// ============================================================================

/// Exact `1:1` stoichiometric bound fraction `f_B([P]_t, [L]_t, K_d) in [0, 1]`:
/// `(([P]_t + [L]_t + K_d) - sqrt(([P]_t + [L]_t + K_d)^2 - 4 [P]_t [L]_t)) / (2 [P]_t)`.
#[inline]
pub fn quadratic_bound_fraction(protein_conc_um: f64, ligand_conc_um: f64, kd_um: f64) -> f64 {
    let l = ligand_conc_um.max(0.0);
    let kd = kd_um.max(1e-12);
    if l == 0.0 {
        return 0.0;
    }
    let p = protein_conc_um.max(0.0);
    if p <= 1e-12 {
        return l / (l + kd);
    }
    let sum = p + l + kd;
    // Numerically stable quadratic root avoiding cancellation when `4*p*l << sum^2`:
    // `(sum - sqrt(sum^2 - 4*p*l)) / (2*p) == (2 * l) / (sum + sqrt(sum^2 - 4*p*l))`!
    let disc = (sum * sum - 4.0 * p * l).max(0.0);
    ((2.0 * l) / (sum + disc.sqrt())).clamp(0.0, 1.0)
}

/// Exact quadratic `1:1` tight-binding chemical shift perturbation `Delta_delta_obs` (`ppm`).
#[inline]
pub fn quadratic_tight_binding_csp(
    ddelta_max_ppm: f64,
    protein_conc_um: f64,
    ligand_conc_um: f64,
    kd_um: f64,
) -> f64 {
    ddelta_max_ppm * quadratic_bound_fraction(protein_conc_um, ligand_conc_um, kd_um)
}

/// Naive Michaelis–Menten / Langmuir isotherm assuming `[L]_free ≈ [L]_t` (neglecting ligand
/// depletion by protein: `ddelta_max * [L]_t / ([L]_t + K_d)`).
#[inline]
pub fn naive_michaelis_menten_csp(ddelta_max_ppm: f64, ligand_conc_um: f64, kd_um: f64) -> f64 {
    let l = ligand_conc_um.max(0.0);
    let kd = kd_um.max(1e-12);
    if l == 0.0 {
        0.0
    } else {
        ddelta_max_ppm * (l / (l + kd))
    }
}

/// Fit naive Michaelis–Menten (`[L]_free ≈ [L]_t`) `K_d` across one or more residue curves via
/// exact 1D Variable Projection (`VarPro`) over `ln(K_d)`.
pub fn fit_naive_michaelis_menten_kd(
    curves: &[&[(f64, f64, f64)]],
    sigma_ppm: f64,
) -> (f64, Vec<f64>, f64) {
    let w = 1.0 / sigma_ppm.max(1e-5).powi(2);
    let eval_kd = |kd: f64| -> (f64, Vec<f64>) {
        let mut chi2 = 0.0;
        let mut dmax_vec = Vec::with_capacity(curves.len());
        for c in curves {
            let mut ff = 0.0;
            let mut fy = 0.0;
            for &(_p_t, l_t, y) in *c {
                let f = if l_t <= 0.0 { 0.0 } else { l_t / (l_t + kd) };
                ff += f * f;
                fy += f * y;
            }
            let dmax = if ff > 1e-18 { (fy / ff).max(0.0) } else { 0.0 };
            for &(_p_t, l_t, y) in *c {
                let f = if l_t <= 0.0 { 0.0 } else { l_t / (l_t + kd) };
                let r = y - dmax * f;
                chi2 += w * r * r;
            }
            dmax_vec.push(dmax);
        }
        (chi2, dmax_vec)
    };

    optimize_1d_varpro_kd(eval_kd)
}

/// Fit exact quadratic `1:1` tight-binding `K_d` across one or more residue curves `[(p_t_um, l_t_um, csp_ppm)]`
/// using globally convergent `VarPro` (`240`-point log grid across `1e-4 .. 1e6 uM` + 40-step
/// golden-section polish to `< 1e-10` relative `K_d` precision!).
pub fn fit_quadratic_tight_binding_varpro(
    curves: &[&[(f64, f64, f64)]],
    sigma_ppm: f64,
) -> (f64, Vec<f64>, f64) {
    let w = 1.0 / sigma_ppm.max(1e-5).powi(2);
    let eval_kd = |kd: f64| -> (f64, Vec<f64>) {
        let mut chi2 = 0.0;
        let mut dmax_vec = Vec::with_capacity(curves.len());
        for c in curves {
            let mut ff = 0.0;
            let mut fy = 0.0;
            for &(p_t, l_t, y) in *c {
                let f = quadratic_bound_fraction(p_t, l_t, kd);
                ff += f * f;
                fy += f * y;
            }
            let dmax = if ff > 1e-18 { (fy / ff).max(0.0) } else { 0.0 };
            for &(p_t, l_t, y) in *c {
                let f = quadratic_bound_fraction(p_t, l_t, kd);
                let r = y - dmax * f;
                chi2 += w * r * r;
            }
            dmax_vec.push(dmax);
        }
        (chi2, dmax_vec)
    };

    optimize_1d_varpro_kd(eval_kd)
}

fn optimize_1d_varpro_kd<F>(eval_kd: F) -> (f64, Vec<f64>, f64)
where
    F: Fn(f64) -> (f64, Vec<f64>),
{
    let n_grid = 240usize;
    let ln_min = (1e-4_f64).ln();
    let ln_max = (1e6_f64).ln();
    let step = (ln_max - ln_min) / ((n_grid - 1) as f64);

    let mut best_i = 0usize;
    let mut best_c2 = f64::INFINITY;
    for i in 0..n_grid {
        let ln_k = ln_min + (i as f64) * step;
        let (c2, _) = eval_kd(ln_k.exp());
        if c2 < best_c2 {
            best_c2 = c2;
            best_i = i;
        }
    }

    let mut lo = ln_min + ((best_i.saturating_sub(2)) as f64) * step;
    let mut hi = (ln_min + ((best_i + 2).min(n_grid - 1) as f64) * step).max(lo + 1e-6);
    let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
    let mut x1 = hi - phi * (hi - lo);
    let mut x2 = lo + phi * (hi - lo);
    let (mut f1, _) = eval_kd(x1.exp());
    let (mut f2, _) = eval_kd(x2.exp());

    for _ in 0..48 {
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - phi * (hi - lo);
            (f1, _) = eval_kd(x1.exp());
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + phi * (hi - lo);
            (f2, _) = eval_kd(x2.exp());
        }
    }

    let opt_kd = (0.5 * (lo + hi)).exp();
    let (opt_c2, opt_dmax) = eval_kd(opt_kd);
    (opt_kd, opt_dmax, opt_c2)
}

/// Perform full per-residue and global quadratic `1:1` tight-binding `K_d` analysis on tracked
/// [`ResidueCspTrajectory`] records, including Williamson `2-sigma` interface detection and
/// parametric Monte Carlo `95%` confidence intervals.
pub fn fit_csp_global_kd(
    trajectories: &[ResidueCspTrajectory],
    config: &CspAnalysisConfig,
) -> Result<CspGlobalFitResult, String> {
    if trajectories.is_empty() {
        return Err("fit_csp_global_kd requires at least one residue trajectory".to_string());
    }

    let sat_pairs: Vec<(i32, f64)> = trajectories
        .iter()
        .map(|t| (t.seq_id, t.max_observed_csp_ppm))
        .collect();
    let thresholds = compute_williamson_2sigma_threshold(&sat_pairs);

    // Extract `(p_t, l_t, csp)` tuples per residue (including apo `l_t = 0` and `l_t > 0` points)
    let obs_tuples: Vec<Vec<(f64, f64, f64)>> = trajectories
        .iter()
        .map(|t| {
            t.points
                .iter()
                .filter_map(|pt| {
                    pt.csp_ppm
                        .map(|y| (pt.protein_conc_um, pt.ligand_conc_um, y))
                })
                .collect()
        })
        .collect();

    // Determine residues used for the global `K_d` fit: significant `>= 2 sigma` binding-site hits
    // with `>= 4` points (falling back to top-perturbation residues if all shifts are uniform).
    let mut global_indices: Vec<usize> = trajectories
        .iter()
        .enumerate()
        .filter(|(i, t)| {
            thresholds.significant_seq_ids.contains(&t.seq_id) && obs_tuples[*i].len() >= 4
        })
        .map(|(i, _)| i)
        .collect();

    if global_indices.is_empty() {
        let mut ranked: Vec<usize> = (0..trajectories.len())
            .filter(|&i| obs_tuples[i].len() >= 4 && trajectories[i].max_observed_csp_ppm > 0.01)
            .collect();
        ranked.sort_by(|&a, &b| {
            trajectories[b]
                .max_observed_csp_ppm
                .total_cmp(&trajectories[a].max_observed_csp_ppm)
        });
        global_indices = ranked.into_iter().take(8.min(trajectories.len())).collect();
    }
    if global_indices.is_empty() {
        return Err(
            "insufficient matched titration points (>= 4 with CSP > 0.01 ppm) to fit Kd"
                .to_string(),
        );
    }

    let global_refs: Vec<&[(f64, f64, f64)]> = global_indices
        .iter()
        .map(|&i| obs_tuples[i].as_slice())
        .collect();
    let (global_kd_um, global_dmax_vec, global_chi2) =
        fit_quadratic_tight_binding_varpro(&global_refs, config.csp_sigma_ppm);
    let (global_naive_mm_kd_um, _, _) =
        fit_naive_michaelis_menten_kd(&global_refs, config.csp_sigma_ppm);

    // Parametric Monte Carlo bootstrap for global `K_d` & `ddelta_max` CIs
    let mut rng = Rng::new(config.mc_seed);
    let n_mc = config.n_mc;
    let mut mc_kds = Vec::with_capacity(n_mc);
    let mut mc_dmax = vec![Vec::with_capacity(n_mc); global_indices.len()];

    for _ in 0..n_mc {
        let mut syn_curves: Vec<Vec<(f64, f64, f64)>> = Vec::with_capacity(global_indices.len());
        for (gi, &ri) in global_indices.iter().enumerate() {
            let dmax = global_dmax_vec[gi];
            let mut sc = Vec::with_capacity(obs_tuples[ri].len());
            for &(p_t, l_t, _) in &obs_tuples[ri] {
                let y_true = quadratic_tight_binding_csp(dmax, p_t, l_t, global_kd_um);
                let y_noisy = if l_t == 0.0 {
                    0.0
                } else {
                    (y_true + rng.normal(0.0, config.csp_sigma_ppm)).max(0.0)
                };
                sc.push((p_t, l_t, y_noisy));
            }
            syn_curves.push(sc);
        }
        let refs: Vec<&[(f64, f64, f64)]> = syn_curves.iter().map(|v| v.as_slice()).collect();
        let (k_mc, dm_mc, _) = fit_quadratic_tight_binding_varpro(&refs, config.csp_sigma_ppm);
        mc_kds.push(k_mc);
        for (gi, &dm) in dm_mc.iter().enumerate() {
            mc_dmax[gi].push(dm);
        }
    }

    let (global_kd_sd_um, global_kd_ci95_low_um, global_kd_ci95_high_um) =
        summarize_mc_samples(&mut mc_kds, global_kd_um);

    let mut global_ddelta_max_by_seq_id = Vec::with_capacity(global_indices.len());
    for (gi, &ri) in global_indices.iter().enumerate() {
        let (sd_dm, _, _) = summarize_mc_samples(&mut mc_dmax[gi], global_dmax_vec[gi]);
        global_ddelta_max_by_seq_id.push((trajectories[ri].seq_id, global_dmax_vec[gi], sd_dm));
    }

    // Per-residue individual fits (fitted independently for all binding-site / informative residues)
    let mut residue_fits = Vec::with_capacity(trajectories.len());
    for (ri, traj) in trajectories.iter().enumerate() {
        let is_sig = thresholds.significant_seq_ids.contains(&traj.seq_id);
        let pts = &obs_tuples[ri];
        if pts.len() >= 3 && traj.max_observed_csp_ppm >= 0.015 {
            let single_ref = [pts.as_slice()];
            let (rkd, rdm, rc2) =
                fit_quadratic_tight_binding_varpro(&single_ref, config.csp_sigma_ppm);
            let (r_mm_kd, _, _) = fit_naive_michaelis_menten_kd(&single_ref, config.csp_sigma_ppm);

            let n_res_mc = (config.n_mc / 2).clamp(40, 200);
            let mut r_mc_kds = Vec::with_capacity(n_res_mc);
            let mut r_mc_dms = Vec::with_capacity(n_res_mc);
            for _ in 0..n_res_mc {
                let sc: Vec<(f64, f64, f64)> = pts
                    .iter()
                    .map(|&(p_t, l_t, _)| {
                        let yt = quadratic_tight_binding_csp(rdm[0], p_t, l_t, rkd);
                        let yn = if l_t == 0.0 {
                            0.0
                        } else {
                            (yt + rng.normal(0.0, config.csp_sigma_ppm)).max(0.0)
                        };
                        (p_t, l_t, yn)
                    })
                    .collect();
                let sref = [sc.as_slice()];
                let (km, dmm, _) = fit_quadratic_tight_binding_varpro(&sref, config.csp_sigma_ppm);
                r_mc_kds.push(km);
                r_mc_dms.push(dmm[0]);
            }
            let (ksd, klo, khi) = summarize_mc_samples(&mut r_mc_kds, rkd);
            let (dmsd, _, _) = summarize_mc_samples(&mut r_mc_dms, rdm[0]);

            residue_fits.push(ResidueKdFit {
                seq_index: traj.seq_index,
                seq_id: traj.seq_id,
                res_name: traj.res_name.clone(),
                max_observed_csp_ppm: traj.max_observed_csp_ppm,
                significant_binding_site: is_sig,
                kd_um: rkd,
                kd_sd_um: ksd,
                kd_ci95_low_um: klo,
                kd_ci95_high_um: khi,
                ddelta_max_ppm: rdm[0],
                ddelta_max_sd_ppm: dmsd,
                naive_mm_kd_um: r_mm_kd,
                chi2: rc2,
                n_points: pts.len(),
            });
        } else {
            residue_fits.push(ResidueKdFit {
                seq_index: traj.seq_index,
                seq_id: traj.seq_id,
                res_name: traj.res_name.clone(),
                max_observed_csp_ppm: traj.max_observed_csp_ppm,
                significant_binding_site: is_sig,
                kd_um: global_kd_um,
                kd_sd_um: 0.0,
                kd_ci95_low_um: global_kd_um,
                kd_ci95_high_um: global_kd_um,
                ddelta_max_ppm: traj.max_observed_csp_ppm,
                ddelta_max_sd_ppm: 0.0,
                naive_mm_kd_um: global_naive_mm_kd_um,
                chi2: 0.0,
                n_points: pts.len(),
            });
        }
    }

    let global_pts: usize = global_indices.iter().map(|&i| obs_tuples[i].len()).sum();
    let global_params = 1 + global_indices.len();
    let global_dof = global_pts.saturating_sub(global_params);

    Ok(CspGlobalFitResult {
        global_kd_um,
        global_kd_sd_um,
        global_kd_ci95_low_um,
        global_kd_ci95_high_um,
        global_naive_mm_kd_um,
        global_chi2,
        global_reduced_chi2: global_chi2 / (global_dof.max(1) as f64),
        global_dof,
        thresholds,
        trajectories: trajectories.to_vec(),
        residue_fits,
        global_ddelta_max_by_seq_id,
    })
}

fn summarize_mc_samples(samples: &mut [f64], point_est: f64) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, point_est, point_est);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / (n as f64);
    let sd = if n > 1 {
        (samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / ((n - 1) as f64)).sqrt()
    } else {
        0.0
    };
    let lo = samples[(n * 25 / 1000).min(n - 1)];
    let hi = samples[(n * 975 / 1000).min(n - 1)];
    (sd, lo, hi)
}

// ============================================================================
// 4. TSV & JSON Export Helpers
// ============================================================================

/// Format a [`CspGlobalFitResult`] summary table as TSV.
pub fn format_csp_tsv(res: &CspGlobalFitResult) -> String {
    let mut out = format!(
        "# global_kd_um={:.4}\tglobal_kd_sd_um={:.4}\tci95=[{:.4},{:.4}]\tnaive_mm_kd_um={:.4}\tthreshold_1sigma={:.5}\tthreshold_2sigma={:.5}\n",
        res.global_kd_um,
        res.global_kd_sd_um,
        res.global_kd_ci95_low_um,
        res.global_kd_ci95_high_um,
        res.global_naive_mm_kd_um,
        res.thresholds.threshold_1sigma_ppm,
        res.thresholds.threshold_2sigma_ppm,
    );
    out.push_str(
        "seq_index\tseq_id\tres_name\tmax_csp_ppm\tsignificant\tkd_um\tkd_sd_um\tddelta_max_ppm\tnaive_mm_kd_um\tn_points\n",
    );
    for r in &res.residue_fits {
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.5}\t{}\t{:.4}\t{:.4}\t{:.5}\t{:.4}\t{}\n",
            r.seq_index,
            r.seq_id,
            r.res_name,
            r.max_observed_csp_ppm,
            r.significant_binding_site,
            r.kd_um,
            r.kd_sd_um,
            r.ddelta_max_ppm,
            r.naive_mm_kd_um,
            r.n_points,
        ));
    }
    out
}

/// Export tracked peak trajectories across all `[L]_t` steps as TSV.
pub fn write_csp_trajectories_tsv(trajectories: &[ResidueCspTrajectory]) -> String {
    let mut out = String::from(
        "seq_index\tseq_id\tres_name\tprotein_conc_um\tligand_conc_um\th_ppm\tn_ppm\tcsp_ppm\n",
    );
    for t in trajectories {
        for pt in &t.points {
            let (h_str, n_str, c_str) = match (pt.matched_peak, pt.csp_ppm) {
                (Some((h, n)), Some(c)) => {
                    (format!("{h:.5}"), format!("{n:.5}"), format!("{c:.5}"))
                }
                _ => ("NA".to_string(), "NA".to_string(), "NA".to_string()),
            };
            out.push_str(&format!(
                "{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{}\n",
                t.seq_index,
                t.seq_id,
                t.res_name,
                pt.protein_conc_um,
                pt.ligand_conc_um,
                h_str,
                n_str,
                c_str,
            ));
        }
    }
    out
}

/// Format a [`CspGlobalFitResult`] as pretty JSON.
pub fn format_csp_json(res: &CspGlobalFitResult) -> Result<String, String> {
    serde_json::to_string_pretty(res).map_err(|e| format!("failed to serialize CSP JSON: {e}"))
}
