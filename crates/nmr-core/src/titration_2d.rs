//! Cross-series 2D NMR peak trajectory tracking, per-titration-step exchange-regime
//! verification, and coupled 2D Bloch–McConnell titration lineshape simulation and global
//! multi-residue kinetic/thermodynamic fitting (`T22.8`).
//!
//! Provenance: Clean-room original implementation strictly from peer-reviewed NMR spectroscopy
//! and numerical optimization literature:
//! - Waudby, C. A., Ramos, A., Cabrita, L. D. & Christodoulou, J. (*Sci. Rep.* 2016, 6,
//!   24826, DOI: 10.1038/srep24826 — Two-Dimensional NMR Lineshape Analysis of Protein–Ligand Interactions:
//!   coupled 2D Bloch–McConnell HSQC lineshape simulation and global fitting of `K_d`,
//!   `k_off`, `Delta_omega_H`, and `Delta_omega_N`).
//! - McConnell, H. M. (*J. Chem. Phys.* 1958, 28, 430–431 — coupled Bloch equations with
//!   two-site chemical exchange).
//! - States, D. J., Haberkorn, R. A. & Ruben, D. J. (*J. Magn. Reson.* 1982, 48, 286–292 —
//!   hypercomplex pure-absorption 2D phase-sensitive quadrature detection).
//! - Kuhn, H. W. (*Naval Res. Logist. Quart.* 1955, 2, 83–97) & Munkres, J. (*J. Soc. Indust.
//!   Appl. Math.* 1957, 5, 32–38 — exact polynomial-time minimum-cost bipartite matching).
//! - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2013, 73, 1–16 — fast-exchange
//!   exact quadratic `1:1` tight-binding chemical shift perturbation isotherm and
//!   slow/intermediate/fast exchange-regime criteria `|Delta_omega| / k_ex`).
//! - Marquardt, D. W. (*J. Soc. Indust. Appl. Math.* 1963, 11, 431–441 — damped Gauss–Newton /
//!   Levenberg–Marquardt nonlinear least-squares optimization).

#![allow(clippy::needless_range_loop)]

use std::f64::consts::{PI, TAU};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use serde::{Deserialize, Serialize};

use crate::csp::{
    amide_alpha_n, fit_quadratic_tight_binding_varpro, hungarian_min_cost_assignment,
    quadratic_bound_fraction,
};

// ============================================================================
// Constants & Error Types
// ============================================================================

/// Ratio of `15N` to `1H` gyromagnetic magnitudes `|gamma_15N / gamma_1H|` (IUPAC 2001).
pub const GAMMA_N15_OVER_H1: f64 = 0.101_329_118;

/// Upper dimensionless exchange ratio `alpha = |Delta_omega| / k_ex` bound for fast exchange.
pub const FAST_EXCHANGE_ALPHA_MAX: f64 = 0.3;

/// Lower dimensionless exchange ratio `alpha = |Delta_omega| / k_ex` bound for slow exchange.
pub const SLOW_EXCHANGE_ALPHA_MIN: f64 = 3.0;

/// Fail-closed error type for 2D titration trajectory tracking, exchange-regime classification,
/// and coupled 2D Bloch–McConnell lineshape simulation/fitting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Titration2dError {
    /// Required input slice was empty or had too few elements.
    EmptyInput(String),
    /// A numeric parameter was non-finite or violated its physical domain.
    InvalidParameter {
        name: String,
        value: f64,
        reason: String,
    },
    /// Series coordinates were not strictly monotonic (increasing or decreasing).
    InvalidSeriesOrder {
        step_index: usize,
        prev: f64,
        curr: f64,
    },
    /// Array/grid dimensions were inconsistent.
    DimensionMismatch {
        context: String,
        expected: usize,
        actual: usize,
    },
    /// Internal numerical solver failure (e.g. non-finite residual or singular matrix).
    NumericalFailure(String),
}

impl fmt::Display for Titration2dError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput(msg) => write!(f, "empty input: {msg}"),
            Self::InvalidParameter {
                name,
                value,
                reason,
            } => write!(f, "invalid parameter `{name}` ({value}): {reason}"),
            Self::InvalidSeriesOrder {
                step_index,
                prev,
                curr,
            } => write!(
                f,
                "series coordinates must be strictly monotonic (step {step_index}: {curr} after {prev})"
            ),
            Self::DimensionMismatch {
                context,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch in {context}: expected {expected}, got {actual}"
            ),
            Self::NumericalFailure(msg) => write!(f, "numerical failure: {msg}"),
        }
    }
}

impl std::error::Error for Titration2dError {}

#[inline]
fn validate_finite(name: &str, value: f64) -> Result<f64, Titration2dError> {
    if !value.is_finite() {
        return Err(Titration2dError::InvalidParameter {
            name: name.to_string(),
            value,
            reason: "must be finite".to_string(),
        });
    }
    Ok(value)
}

#[inline]
fn validate_positive(name: &str, value: f64) -> Result<f64, Titration2dError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Titration2dError::InvalidParameter {
            name: name.to_string(),
            value,
            reason: "must be finite and > 0".to_string(),
        });
    }
    Ok(value)
}

#[inline]
fn validate_non_negative(name: &str, value: f64) -> Result<f64, Titration2dError> {
    if !value.is_finite() || value < 0.0 {
        return Err(Titration2dError::InvalidParameter {
            name: name.to_string(),
            value,
            reason: "must be finite and >= 0".to_string(),
        });
    }
    Ok(value)
}

// ============================================================================
// 1. Cross-Series 2D Peak Trajectory Tracker (Velocity-Extrapolated Kuhn–Munkres)
// ============================================================================

/// A 2D `(1H, 15N)` cross-peak observed at a single titration or temperature series step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Peak2d {
    /// Caller-assigned peak identifier within the step.
    pub peak_id: usize,
    /// Direct `1H` chemical shift (`ppm`).
    pub h_ppm: f64,
    /// Indirect `15N` chemical shift (`ppm`).
    pub n_ppm: f64,
    /// Optional peak height/intensity (used for diagnostics only).
    pub intensity: Option<f64>,
    /// Optional assignment label (e.g. `"G23"`, typically populated on the reference step 0).
    pub label: Option<String>,
    /// Optional amino-acid residue name (e.g. `"GLY"` for `alpha_N = 0.20`).
    pub res_name: Option<String>,
}

/// One step in a titration (ligand concentration / ratio) or temperature 2D peak series.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeriesPeakList {
    /// Human-readable step label (e.g. `"50 uM"` or `"298 K"`).
    pub label: String,
    /// Monotonic independent series coordinate `c_t` (e.g. `[L]_t` in `uM`, molar ratio, or `T` in `K`).
    pub series_coord: f64,
    /// Optional total protein concentration `[P]_t` (`uM`).
    pub protein_conc_um: Option<f64>,
    /// Optional total ligand concentration `[L]_t` (`uM`).
    pub ligand_conc_um: Option<f64>,
    /// Observed 2D peaks at this series step.
    pub peaks: Vec<Peak2d>,
}

/// Configuration for velocity-extrapolated Kuhn–Munkres cross-series peak trajectory tracking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryTrackConfig {
    /// Default `15N` chemical shift scaling factor `alpha_N` (Williamson 2013: `0.14`).
    pub n_scale_factor: f64,
    /// If `true` and a track's `res_name` is `"GLY"` / `"G"`, use `alpha_N = 0.20`.
    pub use_glycine_alpha: bool,
    /// Hard maximum distance (`ppm`, normalized `(dH, alpha_N * dN)`) between predicted position
    /// and candidate peak before forcing dropout.
    pub max_step_ppm: f64,
    /// Explicit dummy/disappear cost in the Kuhn–Munkres assignment matrix for missing or
    /// exchange-broadened peaks.
    pub disappear_cost: f64,
    /// Explicit additive cost penalty when re-linking a previously dropped-out (coasting) track.
    pub reappear_cost: f64,
    /// Maximum consecutive missing steps before a coasting track terminates permanently.
    pub max_gap_steps: usize,
    /// Weight penalizing squared perpendicular deviation from the established velocity ray.
    pub collinearity_weight: f64,
    /// Weight penalizing squared backward motion opposing the established velocity direction.
    pub backward_weight: f64,
    /// Exponential smoothing factor in `(0, 1]` for velocity updates (`1.0` = pure finite difference).
    pub velocity_smoothing: f64,
    /// Maximum ratio `|Delta_c_cur / Delta_c_prev|` used during linear velocity extrapolation.
    pub max_extrapolation_ratio: f64,
    /// Minimum cost margin between optimal assignment and any 2-track swap or runner-up peak
    /// below which a [`TrajectoryAmbiguityFlag`] is emitted.
    pub ambiguity_cost_margin: f64,
    /// Normalized peak separation (`ppm`) below which two assigned tracks at the same step
    /// are flagged as a close crossing proximity.
    pub crossing_proximity_ppm: f64,
    /// If `true`, unmatched peaks at steps `t >= 1` initialize new tracks (e.g. slow-exchange
    /// bound peaks appearing mid-series).
    pub allow_new_tracks: bool,
}

impl Default for TrajectoryTrackConfig {
    fn default() -> Self {
        Self {
            n_scale_factor: 0.14,
            use_glycine_alpha: true,
            max_step_ppm: 0.45,
            disappear_cost: 0.45 * 0.45,
            reappear_cost: 0.0025,
            max_gap_steps: 3,
            collinearity_weight: 1.5,
            backward_weight: 2.5,
            velocity_smoothing: 0.85,
            max_extrapolation_ratio: 4.0,
            ambiguity_cost_margin: 0.0025,
            crossing_proximity_ppm: 0.03,
            allow_new_tracks: false,
        }
    }
}

/// Classification of why a trajectory matching step was flagged as ambiguous.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmbiguityReason {
    /// Swapping two tracks' peak assignments yields a total cost within `ambiguity_cost_margin`.
    NearTieCrossingSwap,
    /// Two matched tracks are spatially closer than `crossing_proximity_ppm` at this step.
    CloseCrossingProximity,
    /// A single track had a runner-up candidate peak within `ambiguity_cost_margin`.
    RunnerUpPeakNearTie,
    /// A track's matched peak cost was within `ambiguity_cost_margin` of the disappear cost.
    DropoutGateNearTie,
}

/// Diagnostic flag emitted whenever a step contains a near-tie crossing or ambiguous assignment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryAmbiguityFlag {
    pub step_index: usize,
    pub series_coord: f64,
    pub track_id: usize,
    pub competing_track_id: Option<usize>,
    pub peak_index: Option<usize>,
    pub competing_peak_index: Option<usize>,
    pub cost_margin: f64,
    pub peak_separation_ppm: Option<f64>,
    pub reason: AmbiguityReason,
    pub message: String,
}

/// Per-step status of a single tracked peak trajectory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TrackStepStatus {
    /// Track was matched to an observed peak at this step.
    Matched {
        peak_index: usize,
        peak_id: usize,
        h_ppm: f64,
        n_ppm: f64,
        predicted_h_ppm: f64,
        predicted_n_ppm: f64,
        prediction_error_ppm: f64,
        assignment_cost: f64,
        reappeared_after_gap: usize,
    },
    /// Track had no valid peak match (e.g. broadened out by intermediate exchange) and is coasting.
    DroppedOut {
        predicted_h_ppm: f64,
        predicted_n_ppm: f64,
        consecutive_misses: usize,
    },
    /// Track had not been spawned yet at this step (when `allow_new_tracks = true`).
    NotYetBorn,
    /// Track exceeded `max_gap_steps` consecutive dropouts and was terminated.
    Terminated,
}

/// Observation entry for one track at one series step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackedStepObservation {
    pub step_index: usize,
    pub series_coord: f64,
    pub status: TrackStepStatus,
}

/// Complete trajectory of one peak identity across the 2D series.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeakTrajectory {
    pub track_id: usize,
    pub label: Option<String>,
    pub res_name: Option<String>,
    pub birth_step: usize,
    pub observations: Vec<TrackedStepObservation>,
    pub n_matched: usize,
    pub n_dropouts: usize,
    pub max_csp_ppm: f64,
}

/// Output of cross-series peak trajectory tracking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryTrackingResult {
    pub trajectories: Vec<PeakTrajectory>,
    pub ambiguity_flags: Vec<TrajectoryAmbiguityFlag>,
    pub unmatched_peaks_per_step: Vec<Vec<usize>>,
    pub total_assignment_cost: f64,
}

#[derive(Clone, Debug)]
struct ActiveTrackState {
    track_id: usize,
    alpha_n: f64,
    origin_h: f64,
    origin_n: f64,
    last_h: f64,
    last_n: f64,
    last_coord: f64,
    last_step_delta_coord: Option<f64>,
    /// Velocity `(dh/dc, dn/dc)` in raw `(ppm / coord_unit)`.
    vel_h: Option<f64>,
    vel_n: Option<f64>,
    consecutive_misses: usize,
    terminated: bool,
}

impl ActiveTrackState {
    fn predict(&self, next_coord: f64, max_extrap_ratio: f64) -> (f64, f64) {
        if let (Some(vh), Some(vn)) = (self.vel_h, self.vel_n) {
            let dc = next_coord - self.last_coord;
            let eff_dc = if let Some(prev_dc) = self.last_step_delta_coord {
                let prev_abs = prev_dc.abs().max(1e-12);
                let max_abs = prev_abs * max_extrap_ratio;
                if dc.abs() > max_abs {
                    dc.signum() * max_abs
                } else {
                    dc
                }
            } else {
                dc
            };
            (self.last_h + vh * eff_dc, self.last_n + vn * eff_dc)
        } else {
            (self.last_h, self.last_n)
        }
    }

    fn update_match(&mut self, h_ppm: f64, n_ppm: f64, coord: f64, smoothing: f64) {
        let dc = coord - self.last_coord;
        if dc.abs() > 1e-12 {
            let inst_vh = (h_ppm - self.last_h) / dc;
            let inst_vn = (n_ppm - self.last_n) / dc;
            match (self.vel_h, self.vel_n) {
                (Some(old_vh), Some(old_vn)) => {
                    self.vel_h = Some((1.0 - smoothing) * old_vh + smoothing * inst_vh);
                    self.vel_n = Some((1.0 - smoothing) * old_vn + smoothing * inst_vn);
                }
                _ => {
                    self.vel_h = Some(inst_vh);
                    self.vel_n = Some(inst_vn);
                }
            }
            self.last_step_delta_coord = Some(dc);
        }
        self.last_h = h_ppm;
        self.last_n = n_ppm;
        self.last_coord = coord;
        self.consecutive_misses = 0;
    }
}

fn validate_track_config(config: &TrajectoryTrackConfig) -> Result<(), Titration2dError> {
    validate_positive("n_scale_factor", config.n_scale_factor)?;
    validate_positive("max_step_ppm", config.max_step_ppm)?;
    validate_positive("disappear_cost", config.disappear_cost)?;
    validate_non_negative("reappear_cost", config.reappear_cost)?;
    validate_non_negative("collinearity_weight", config.collinearity_weight)?;
    validate_non_negative("backward_weight", config.backward_weight)?;
    if !config.velocity_smoothing.is_finite()
        || config.velocity_smoothing <= 0.0
        || config.velocity_smoothing > 1.0
    {
        return Err(Titration2dError::InvalidParameter {
            name: "velocity_smoothing".to_string(),
            value: config.velocity_smoothing,
            reason: "must be in (0, 1]".to_string(),
        });
    }
    validate_positive("max_extrapolation_ratio", config.max_extrapolation_ratio)?;
    validate_non_negative("ambiguity_cost_margin", config.ambiguity_cost_margin)?;
    validate_non_negative("crossing_proximity_ppm", config.crossing_proximity_ppm)?;
    Ok(())
}

/// Track 2D `(1H, 15N)` cross-peak trajectories across a ligand titration or temperature series
/// (`T >= 2` steps) using linear velocity extrapolation, collinearity/direction consistency,
/// exact Kuhn–Munkres minimum-cost bipartite matching with explicit disappear/reappear costs,
/// and near-tie crossing ambiguity detection.
pub fn track_titration_trajectories(
    steps: &[SeriesPeakList],
    config: &TrajectoryTrackConfig,
) -> Result<TrajectoryTrackingResult, Titration2dError> {
    validate_track_config(config)?;
    if steps.len() < 2 {
        return Err(Titration2dError::EmptyInput(
            "track_titration_trajectories requires at least 2 series steps".to_string(),
        ));
    }

    // Validate finite coordinates, peak coordinates, and strict monotonicity of series_coord.
    let mut direction: Option<f64> = None;
    for (t, step) in steps.iter().enumerate() {
        validate_finite("series_coord", step.series_coord)?;
        if let Some(p) = step.protein_conc_um {
            validate_non_negative("protein_conc_um", p)?;
        }
        if let Some(l) = step.ligand_conc_um {
            validate_non_negative("ligand_conc_um", l)?;
        }
        for pk in &step.peaks {
            validate_finite("peak.h_ppm", pk.h_ppm)?;
            validate_finite("peak.n_ppm", pk.n_ppm)?;
            if let Some(inten) = pk.intensity {
                validate_finite("peak.intensity", inten)?;
            }
        }
        if t > 0 {
            let prev = steps[t - 1].series_coord;
            let diff = step.series_coord - prev;
            if diff.abs() <= 1e-12 {
                return Err(Titration2dError::InvalidSeriesOrder {
                    step_index: t,
                    prev,
                    curr: step.series_coord,
                });
            }
            match direction {
                None => direction = Some(diff.signum()),
                Some(dir) if diff * dir <= 0.0 => {
                    return Err(Titration2dError::InvalidSeriesOrder {
                        step_index: t,
                        prev,
                        curr: step.series_coord,
                    });
                }
                _ => {}
            }
        }
    }

    if steps[0].peaks.is_empty() && !config.allow_new_tracks {
        return Err(Titration2dError::EmptyInput(
            "step 0 has no seed peaks and allow_new_tracks is false".to_string(),
        ));
    }

    let n_steps = steps.len();
    let mut trajectories: Vec<PeakTrajectory> = Vec::new();
    let mut active_states: Vec<ActiveTrackState> = Vec::new();
    let mut unmatched_peaks_per_step: Vec<Vec<usize>> = vec![Vec::new(); n_steps];
    let mut ambiguity_flags: Vec<TrajectoryAmbiguityFlag> = Vec::new();
    let mut total_assignment_cost = 0.0;

    // Initialize tracks from step 0.
    let c0 = steps[0].series_coord;
    for (peak_idx, pk) in steps[0].peaks.iter().enumerate() {
        let track_id = trajectories.len();
        let alpha_n = if config.use_glycine_alpha {
            pk.res_name
                .as_deref()
                .map(amide_alpha_n)
                .unwrap_or(config.n_scale_factor)
        } else {
            config.n_scale_factor
        };
        trajectories.push(PeakTrajectory {
            track_id,
            label: pk.label.clone(),
            res_name: pk.res_name.clone(),
            birth_step: 0,
            observations: vec![TrackedStepObservation {
                step_index: 0,
                series_coord: c0,
                status: TrackStepStatus::Matched {
                    peak_index: peak_idx,
                    peak_id: pk.peak_id,
                    h_ppm: pk.h_ppm,
                    n_ppm: pk.n_ppm,
                    predicted_h_ppm: pk.h_ppm,
                    predicted_n_ppm: pk.n_ppm,
                    prediction_error_ppm: 0.0,
                    assignment_cost: 0.0,
                    reappeared_after_gap: 0,
                },
            }],
            n_matched: 1,
            n_dropouts: 0,
            max_csp_ppm: 0.0,
        });
        active_states.push(ActiveTrackState {
            track_id,
            alpha_n,
            origin_h: pk.h_ppm,
            origin_n: pk.n_ppm,
            last_h: pk.h_ppm,
            last_n: pk.n_ppm,
            last_coord: c0,
            last_step_delta_coord: None,
            vel_h: None,
            vel_n: None,
            consecutive_misses: 0,
            terminated: false,
        });
    }

    // Track step-by-step across t = 1..n_steps.
    for (t, step) in steps.iter().enumerate().skip(1) {
        let coord = step.series_coord;
        let live_indices: Vec<usize> = active_states
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.terminated)
            .map(|(idx, _)| idx)
            .collect();

        // Mark already-terminated tracks for this step.
        for (idx, state) in active_states.iter().enumerate() {
            if state.terminated {
                trajectories[idx].observations.push(TrackedStepObservation {
                    step_index: t,
                    series_coord: coord,
                    status: TrackStepStatus::Terminated,
                });
            }
        }

        let n_live = live_indices.len();
        let n_peaks = step.peaks.len();
        let mut matched_peak_flags = vec![false; n_peaks];

        if n_live > 0 {
            let n_cols = n_peaks + n_live;
            let forbidden_cost = (config.disappear_cost * 100.0).max(1e3);
            let mut cost = vec![vec![forbidden_cost; n_cols]; n_live];
            let mut predictions = vec![(0.0_f64, 0.0_f64); n_live];
            let mut pred_errors = vec![vec![f64::INFINITY; n_peaks]; n_live];

            for (row, &track_idx) in live_indices.iter().enumerate() {
                let st = &active_states[track_idx];
                let (pred_h, pred_n) = st.predict(coord, config.max_extrapolation_ratio);
                predictions[row] = (pred_h, pred_n);

                // Normalized velocity direction vector (when established).
                let dc = coord - st.last_coord;
                let vel_unit = match (st.vel_h, st.vel_n) {
                    (Some(vh), Some(vn)) => {
                        let vx = vh * dc;
                        let vy = st.alpha_n * vn * dc;
                        let vnorm = vx.hypot(vy);
                        if vnorm > 1e-4 {
                            Some((vx / vnorm, vy / vnorm))
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                // Gate scales mildly with gap length so a reappearing peak after dropout can link.
                let gap_scale = 1.0 + 0.5 * (st.consecutive_misses as f64);
                let gate_ppm = config.max_step_ppm * gap_scale;
                let reappear_pen = if st.consecutive_misses > 0 {
                    config.reappear_cost
                } else {
                    0.0
                };

                for (col, pk) in step.peaks.iter().enumerate() {
                    let dx_pred = pk.h_ppm - pred_h;
                    let dy_pred = st.alpha_n * (pk.n_ppm - pred_n);
                    let dist_pred = dx_pred.hypot(dy_pred);
                    pred_errors[row][col] = dist_pred;

                    if dist_pred > gate_ppm {
                        // Outside hard gate: strictly worse than disappear_cost.
                        cost[row][col] = config.disappear_cost + 1.0 + dist_pred * dist_pred;
                        continue;
                    }

                    let mut c_val = dist_pred * dist_pred + reappear_pen;
                    if let Some((ux, uy)) = vel_unit {
                        let sx = pk.h_ppm - st.last_h;
                        let sy = st.alpha_n * (pk.n_ppm - st.last_n);
                        let perp = sx * uy - sy * ux;
                        let along = sx * ux + sy * uy;
                        c_val += config.collinearity_weight * perp * perp;
                        if along < 0.0 {
                            c_val += config.backward_weight * along * along;
                        }
                    }
                    cost[row][col] = c_val;
                }

                // Dedicated dummy/dropout column for this track.
                cost[row][n_peaks + row] = config.disappear_cost;
            }

            let (assignment, step_cost) =
                hungarian_min_cost_assignment(&cost).map_err(Titration2dError::NumericalFailure)?;
            total_assignment_cost += step_cost;

            // Detect ambiguities:
            // (a) Pairwise 2-track crossing swap near-ties & spatial proximity.
            for r1 in 0..n_live {
                let c1 = assignment[r1];
                if c1 >= n_peaks {
                    continue;
                }
                for r2 in (r1 + 1)..n_live {
                    let c2 = assignment[r2];
                    if c2 >= n_peaks {
                        continue;
                    }
                    let cur_pair = cost[r1][c1] + cost[r2][c2];
                    let swap_pair = cost[r1][c2] + cost[r2][c1];
                    let swap_margin = (swap_pair - cur_pair).max(0.0);

                    let alpha_pair = 0.5
                        * (active_states[live_indices[r1]].alpha_n
                            + active_states[live_indices[r2]].alpha_n);
                    let sep_ppm = (step.peaks[c1].h_ppm - step.peaks[c2].h_ppm)
                        .hypot(alpha_pair * (step.peaks[c1].n_ppm - step.peaks[c2].n_ppm));

                    if swap_margin <= config.ambiguity_cost_margin {
                        ambiguity_flags.push(TrajectoryAmbiguityFlag {
                            step_index: t,
                            series_coord: coord,
                            track_id: active_states[live_indices[r1]].track_id,
                            competing_track_id: Some(active_states[live_indices[r2]].track_id),
                            peak_index: Some(c1),
                            competing_peak_index: Some(c2),
                            cost_margin: swap_margin,
                            peak_separation_ppm: Some(sep_ppm),
                            reason: AmbiguityReason::NearTieCrossingSwap,
                            message: format!(
                                "step {t} (coord={coord:.4}): tracks {} and {} have near-tie crossing swap margin {swap_margin:.3e} (sep={sep_ppm:.4} ppm)",
                                active_states[live_indices[r1]].track_id,
                                active_states[live_indices[r2]].track_id,
                            ),
                        });
                    } else if sep_ppm <= config.crossing_proximity_ppm {
                        ambiguity_flags.push(TrajectoryAmbiguityFlag {
                            step_index: t,
                            series_coord: coord,
                            track_id: active_states[live_indices[r1]].track_id,
                            competing_track_id: Some(active_states[live_indices[r2]].track_id),
                            peak_index: Some(c1),
                            competing_peak_index: Some(c2),
                            cost_margin: swap_margin,
                            peak_separation_ppm: Some(sep_ppm),
                            reason: AmbiguityReason::CloseCrossingProximity,
                            message: format!(
                                "step {t} (coord={coord:.4}): tracks {} and {} pass within {sep_ppm:.4} ppm (swap margin {swap_margin:.3e})",
                                active_states[live_indices[r1]].track_id,
                                active_states[live_indices[r2]].track_id,
                            ),
                        });
                    }
                }
            }

            // (b) Per-track runner-up peak or dropout-gate near-tie.
            for r in 0..n_live {
                let c_assigned = assignment[r];
                let assigned_cost = cost[r][c_assigned];
                if c_assigned < n_peaks {
                    let mut best_alt_col: Option<usize> = None;
                    let mut best_alt_cost = f64::INFINITY;
                    for c in 0..n_peaks {
                        if c != c_assigned && cost[r][c] < best_alt_cost {
                            best_alt_cost = cost[r][c];
                            best_alt_col = Some(c);
                        }
                    }
                    if let Some(alt_c) = best_alt_col {
                        let margin = (best_alt_cost - assigned_cost).max(0.0);
                        if margin <= config.ambiguity_cost_margin {
                            let sep = (step.peaks[c_assigned].h_ppm - step.peaks[alt_c].h_ppm)
                                .hypot(
                                    active_states[live_indices[r]].alpha_n
                                        * (step.peaks[c_assigned].n_ppm - step.peaks[alt_c].n_ppm),
                                );
                            ambiguity_flags.push(TrajectoryAmbiguityFlag {
                                step_index: t,
                                series_coord: coord,
                                track_id: active_states[live_indices[r]].track_id,
                                competing_track_id: None,
                                peak_index: Some(c_assigned),
                                competing_peak_index: Some(alt_c),
                                cost_margin: margin,
                                peak_separation_ppm: Some(sep),
                                reason: AmbiguityReason::RunnerUpPeakNearTie,
                                message: format!(
                                    "step {t} (coord={coord:.4}): track {} has runner-up peak {alt_c} within cost margin {margin:.3e}",
                                    active_states[live_indices[r]].track_id,
                                ),
                            });
                        }
                    }
                    let gate_margin = (config.disappear_cost - assigned_cost).max(0.0);
                    if gate_margin <= config.ambiguity_cost_margin {
                        ambiguity_flags.push(TrajectoryAmbiguityFlag {
                            step_index: t,
                            series_coord: coord,
                            track_id: active_states[live_indices[r]].track_id,
                            competing_track_id: None,
                            peak_index: Some(c_assigned),
                            competing_peak_index: None,
                            cost_margin: gate_margin,
                            peak_separation_ppm: None,
                            reason: AmbiguityReason::DropoutGateNearTie,
                            message: format!(
                                "step {t} (coord={coord:.4}): track {} matched peak {c_assigned} within {gate_margin:.3e} of dropout cost",
                                active_states[live_indices[r]].track_id,
                            ),
                        });
                    }
                }
            }

            // Apply assignments and update track states.
            for (row, &track_idx) in live_indices.iter().enumerate() {
                let col = assignment[row];
                let (pred_h, pred_n) = predictions[row];
                if col < n_peaks && cost[row][col] < config.disappear_cost {
                    matched_peak_flags[col] = true;
                    let pk = &step.peaks[col];
                    let gap_before = active_states[track_idx].consecutive_misses;
                    let origin_h = active_states[track_idx].origin_h;
                    let origin_n = active_states[track_idx].origin_n;
                    let alpha_n = active_states[track_idx].alpha_n;
                    let csp = (pk.h_ppm - origin_h).hypot(alpha_n * (pk.n_ppm - origin_n));

                    active_states[track_idx].update_match(
                        pk.h_ppm,
                        pk.n_ppm,
                        coord,
                        config.velocity_smoothing,
                    );

                    let traj = &mut trajectories[track_idx];
                    traj.n_matched += 1;
                    traj.max_csp_ppm = traj.max_csp_ppm.max(csp);
                    traj.observations.push(TrackedStepObservation {
                        step_index: t,
                        series_coord: coord,
                        status: TrackStepStatus::Matched {
                            peak_index: col,
                            peak_id: pk.peak_id,
                            h_ppm: pk.h_ppm,
                            n_ppm: pk.n_ppm,
                            predicted_h_ppm: pred_h,
                            predicted_n_ppm: pred_n,
                            prediction_error_ppm: pred_errors[row][col],
                            assignment_cost: cost[row][col],
                            reappeared_after_gap: gap_before,
                        },
                    });
                } else {
                    let st = &mut active_states[track_idx];
                    st.consecutive_misses += 1;
                    let misses = st.consecutive_misses;
                    if misses > config.max_gap_steps {
                        st.terminated = true;
                    }
                    let traj = &mut trajectories[track_idx];
                    traj.n_dropouts += 1;
                    traj.observations.push(TrackedStepObservation {
                        step_index: t,
                        series_coord: coord,
                        status: TrackStepStatus::DroppedOut {
                            predicted_h_ppm: pred_h,
                            predicted_n_ppm: pred_n,
                            consecutive_misses: misses,
                        },
                    });
                }
            }
        }

        // Collect unmatched peaks and optionally spawn new tracks.
        for (peak_idx, pk) in step.peaks.iter().enumerate() {
            if !matched_peak_flags[peak_idx] {
                if config.allow_new_tracks {
                    let track_id = trajectories.len();
                    let alpha_n = if config.use_glycine_alpha {
                        pk.res_name
                            .as_deref()
                            .map(amide_alpha_n)
                            .unwrap_or(config.n_scale_factor)
                    } else {
                        config.n_scale_factor
                    };
                    let mut obs = Vec::with_capacity(t + 1);
                    for (prev_t, prev_step) in steps.iter().enumerate().take(t) {
                        obs.push(TrackedStepObservation {
                            step_index: prev_t,
                            series_coord: prev_step.series_coord,
                            status: TrackStepStatus::NotYetBorn,
                        });
                    }
                    obs.push(TrackedStepObservation {
                        step_index: t,
                        series_coord: coord,
                        status: TrackStepStatus::Matched {
                            peak_index: peak_idx,
                            peak_id: pk.peak_id,
                            h_ppm: pk.h_ppm,
                            n_ppm: pk.n_ppm,
                            predicted_h_ppm: pk.h_ppm,
                            predicted_n_ppm: pk.n_ppm,
                            prediction_error_ppm: 0.0,
                            assignment_cost: 0.0,
                            reappeared_after_gap: 0,
                        },
                    });
                    trajectories.push(PeakTrajectory {
                        track_id,
                        label: pk.label.clone(),
                        res_name: pk.res_name.clone(),
                        birth_step: t,
                        observations: obs,
                        n_matched: 1,
                        n_dropouts: 0,
                        max_csp_ppm: 0.0,
                    });
                    active_states.push(ActiveTrackState {
                        track_id,
                        alpha_n,
                        origin_h: pk.h_ppm,
                        origin_n: pk.n_ppm,
                        last_h: pk.h_ppm,
                        last_n: pk.n_ppm,
                        last_coord: coord,
                        last_step_delta_coord: None,
                        vel_h: None,
                        vel_n: None,
                        consecutive_misses: 0,
                        terminated: false,
                    });
                } else {
                    unmatched_peaks_per_step[t].push(peak_idx);
                }
            }
        }
    }

    Ok(TrajectoryTrackingResult {
        trajectories,
        ambiguity_flags,
        unmatched_peaks_per_step,
        total_assignment_cost,
    })
}

// ============================================================================
// 2. Per-Titration-Step Exchange-Regime Verification
// ============================================================================

/// NMR chemical exchange regime classified by dimensionless ratio `alpha = |Delta_omega| / k_ex`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExchangeRegime {
    /// Fast exchange (`alpha < 0.3`): single Lorentzian peak at population-weighted shift.
    Fast,
    /// Intermediate exchange (`0.3 <= alpha <= 3.0`): severe broadening, skew, coalescence.
    Intermediate,
    /// Slow exchange (`alpha > 3.0`): resolved free and bound doublet peaks.
    Slow,
}

impl ExchangeRegime {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Intermediate => "Intermediate",
            Self::Slow => "Slow",
        }
    }
}

/// Classify dimensionless exchange ratio `alpha = |Delta_omega| / k_ex` into `Fast` (`< 0.3`),
/// `Intermediate` (`0.3 ..= 3.0`), or `Slow` (`> 3.0`).
#[inline]
pub fn classify_alpha(alpha: f64) -> ExchangeRegime {
    let a = alpha.abs();
    if a < FAST_EXCHANGE_ALPHA_MAX {
        ExchangeRegime::Fast
    } else if a <= SLOW_EXCHANGE_ALPHA_MIN {
        ExchangeRegime::Intermediate
    } else {
        ExchangeRegime::Slow
    }
}

/// Exact `1:1` equilibrium populations, free ligand concentration, and microscopic exchange rates
/// for a single titration point.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingEquilibriumState {
    pub p_a: f64,
    pub p_b: f64,
    pub l_free_um: f64,
    pub k_on_um_inv_s_inv: f64,
    pub k_ab_s_inv: f64,
    pub k_ba_s_inv: f64,
    pub k_ex_s_inv: f64,
}

/// Compute exact `1:1` binding equilibrium state `(p_A, p_B, [L]_free, k_on, k_AB, k_BA, k_ex)`
/// for total protein `[P]_t` (`uM`), total ligand `[L]_t` (`uM`), dissociation constant `K_d` (`uM`),
/// and dissociation rate `k_off` (`s^-1`).
pub fn binding_equilibrium_rates(
    protein_conc_um: f64,
    ligand_conc_um: f64,
    kd_um: f64,
    k_off_s_inv: f64,
) -> Result<BindingEquilibriumState, Titration2dError> {
    validate_positive("protein_conc_um", protein_conc_um)?;
    validate_non_negative("ligand_conc_um", ligand_conc_um)?;
    validate_positive("kd_um", kd_um)?;
    validate_positive("k_off_s_inv", k_off_s_inv)?;

    let p_b = quadratic_bound_fraction(protein_conc_um, ligand_conc_um, kd_um);
    let p_a = (1.0 - p_b).clamp(0.0, 1.0);
    let pl_um = protein_conc_um * p_b;
    let l_free_um = (ligand_conc_um - pl_um).max(0.0);
    let k_on_um_inv_s_inv = k_off_s_inv / kd_um;
    let k_ab_s_inv = k_on_um_inv_s_inv * l_free_um;
    let k_ba_s_inv = k_off_s_inv;
    let k_ex_s_inv = k_ab_s_inv + k_ba_s_inv;

    Ok(BindingEquilibriumState {
        p_a,
        p_b,
        l_free_um,
        k_on_um_inv_s_inv,
        k_ab_s_inv,
        k_ba_s_inv,
        k_ex_s_inv,
    })
}

/// Concentration coordinates for a single titration point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TitrationConcentrationPoint {
    pub step_index: usize,
    pub label: String,
    pub protein_conc_um: f64,
    pub ligand_conc_um: f64,
}

/// Input parameters for per-titration-step exchange-regime verification of a residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExchangeRegimeInput {
    pub label: String,
    pub kd_um: f64,
    pub k_off_s_inv: f64,
    /// `1H` chemical shift difference `delta_B - delta_A` (`ppm`).
    pub dw_h_ppm: f64,
    /// `15N` chemical shift difference `delta_B - delta_A` (`ppm`).
    pub dw_n_ppm: f64,
    /// Spectrometer `1H` Larmor frequency (`MHz`, e.g. `600.0`).
    pub h_larmor_mhz: f64,
    /// Optional explicit `15N` Larmor frequency (`MHz`); defaults to `h_larmor_mhz * GAMMA_N15_OVER_H1`.
    pub n_larmor_mhz: Option<f64>,
}

/// Detailed exchange-regime audit for one titration step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepExchangeRegime {
    pub step_index: usize,
    pub label: String,
    pub protein_conc_um: f64,
    pub ligand_conc_um: f64,
    pub p_a: f64,
    pub p_b: f64,
    pub l_free_um: f64,
    pub k_on_um_inv_s_inv: f64,
    pub k_ab_s_inv: f64,
    pub k_ba_s_inv: f64,
    pub k_ex_s_inv: f64,
    pub abs_dw_h_rad_s: f64,
    pub abs_dw_n_rad_s: f64,
    /// Step-dependent dimensionless exchange ratio `|Delta_omega_H| / k_ex(t)`.
    pub alpha_h: f64,
    /// Step-dependent dimensionless exchange ratio `|Delta_omega_N| / k_ex(t)`.
    pub alpha_n: f64,
    /// Reference zero-ligand exchange ratio `|Delta_omega_H| / k_off`.
    pub alpha_off_h: f64,
    /// Reference zero-ligand exchange ratio `|Delta_omega_N| / k_off`.
    pub alpha_off_n: f64,
    pub regime_h: ExchangeRegime,
    pub regime_n: ExchangeRegime,
    pub overall_regime: ExchangeRegime,
}

/// Explicit warning emitted whenever fast-exchange quadratic CSP fitting is applied to
/// titration points in the intermediate or slow exchange regime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExchangeRegimeWarning {
    pub residue_label: String,
    pub step_indices: Vec<usize>,
    pub worst_regime: ExchangeRegime,
    pub max_alpha_h: f64,
    pub max_alpha_n: f64,
    pub message: String,
}

/// Complete per-step exchange-regime report for one residue across a titration series.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueExchangeRegimeReport {
    pub label: String,
    pub steps: Vec<StepExchangeRegime>,
    pub worst_regime_h: ExchangeRegime,
    pub worst_regime_n: ExchangeRegime,
    pub overall_worst_regime: ExchangeRegime,
    /// `true` iff every titration step is strictly in the `Fast` exchange regime on both `1H` and `15N`.
    pub fast_exchange_csp_valid: bool,
    pub warnings: Vec<ExchangeRegimeWarning>,
}

/// Classify the chemical exchange regime (`Fast`, `Intermediate`, `Slow`) on both `1H` and `15N`
/// axes at every titration step, emitting explicit warnings whenever intermediate or slow
/// exchange invalidates fast-exchange quadratic CSP fitting.
pub fn classify_exchange_regime(
    input: &ExchangeRegimeInput,
    points: &[TitrationConcentrationPoint],
) -> Result<ResidueExchangeRegimeReport, Titration2dError> {
    if points.is_empty() {
        return Err(Titration2dError::EmptyInput(
            "classify_exchange_regime requires at least one titration concentration point"
                .to_string(),
        ));
    }
    validate_positive("kd_um", input.kd_um)?;
    validate_positive("k_off_s_inv", input.k_off_s_inv)?;
    validate_finite("dw_h_ppm", input.dw_h_ppm)?;
    validate_finite("dw_n_ppm", input.dw_n_ppm)?;
    let h_mhz = validate_positive("h_larmor_mhz", input.h_larmor_mhz)?;
    let n_mhz = match input.n_larmor_mhz {
        Some(v) => validate_positive("n_larmor_mhz", v)?,
        None => h_mhz * GAMMA_N15_OVER_H1,
    };

    let abs_dw_h_rad_s = TAU * input.dw_h_ppm.abs() * h_mhz;
    let abs_dw_n_rad_s = TAU * input.dw_n_ppm.abs() * n_mhz;
    let alpha_off_h = abs_dw_h_rad_s / input.k_off_s_inv;
    let alpha_off_n = abs_dw_n_rad_s / input.k_off_s_inv;

    let mut step_reports = Vec::with_capacity(points.len());
    let mut worst_h = ExchangeRegime::Fast;
    let mut worst_n = ExchangeRegime::Fast;
    let mut non_fast_steps = Vec::new();
    let mut max_alpha_h = 0.0_f64;
    let mut max_alpha_n = 0.0_f64;

    for pt in points {
        let eq = binding_equilibrium_rates(
            pt.protein_conc_um,
            pt.ligand_conc_um,
            input.kd_um,
            input.k_off_s_inv,
        )?;
        let alpha_h = abs_dw_h_rad_s / eq.k_ex_s_inv;
        let alpha_n = abs_dw_n_rad_s / eq.k_ex_s_inv;
        let regime_h = classify_alpha(alpha_h);
        let regime_n = classify_alpha(alpha_n);
        let overall = regime_h.max(regime_n);

        worst_h = worst_h.max(regime_h);
        worst_n = worst_n.max(regime_n);
        max_alpha_h = max_alpha_h.max(alpha_h);
        max_alpha_n = max_alpha_n.max(alpha_n);
        if overall != ExchangeRegime::Fast {
            non_fast_steps.push(pt.step_index);
        }

        step_reports.push(StepExchangeRegime {
            step_index: pt.step_index,
            label: pt.label.clone(),
            protein_conc_um: pt.protein_conc_um,
            ligand_conc_um: pt.ligand_conc_um,
            p_a: eq.p_a,
            p_b: eq.p_b,
            l_free_um: eq.l_free_um,
            k_on_um_inv_s_inv: eq.k_on_um_inv_s_inv,
            k_ab_s_inv: eq.k_ab_s_inv,
            k_ba_s_inv: eq.k_ba_s_inv,
            k_ex_s_inv: eq.k_ex_s_inv,
            abs_dw_h_rad_s,
            abs_dw_n_rad_s,
            alpha_h,
            alpha_n,
            alpha_off_h,
            alpha_off_n,
            regime_h,
            regime_n,
            overall_regime: overall,
        });
    }

    let overall_worst = worst_h.max(worst_n);
    let fast_exchange_csp_valid = non_fast_steps.is_empty();
    let mut warnings = Vec::new();
    if !fast_exchange_csp_valid {
        warnings.push(ExchangeRegimeWarning {
            residue_label: input.label.clone(),
            step_indices: non_fast_steps.clone(),
            worst_regime: overall_worst,
            max_alpha_h,
            max_alpha_n,
            message: format!(
                "residue '{}' enters {} exchange on {} step(s) {:?} (max alpha_H={max_alpha_h:.2}, max alpha_N={max_alpha_n:.2}): fast-exchange quadratic CSP fitting is invalid/biased; use coupled 2D Bloch-McConnell lineshape analysis",
                input.label,
                overall_worst.as_str(),
                non_fast_steps.len(),
                non_fast_steps,
            ),
        });
    }

    Ok(ResidueExchangeRegimeReport {
        label: input.label.clone(),
        steps: step_reports,
        worst_regime_h: worst_h,
        worst_regime_n: worst_n,
        overall_worst_regime: overall_worst,
        fast_exchange_csp_valid,
        warnings,
    })
}

// ============================================================================
// 3. Exact 2x2 Complex Arithmetic, Spectral Decomposition & Matrix Exponential
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };
    const ONE: Self = Self { re: 1.0, im: 0.0 };

    #[inline]
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    #[inline]
    fn exp(self) -> Self {
        let r = self.re.exp();
        let (s, c) = self.im.sin_cos();
        Self {
            re: r * c,
            im: r * s,
        }
    }

    #[inline]
    fn sqrt(self) -> Self {
        let r = self.re.hypot(self.im);
        if r == 0.0 {
            return Self::ZERO;
        }
        if self.re >= 0.0 {
            let re = (0.5 * (r + self.re)).sqrt();
            let im = 0.5 * self.im / re;
            Self { re, im }
        } else {
            let im_abs = (0.5 * (r - self.re)).sqrt();
            let im = if self.im >= 0.0 { im_abs } else { -im_abs };
            let re = (0.5 * self.im / im).abs();
            Self { re, im }
        }
    }
}

impl Add for C64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for C64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Neg for C64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl Mul for C64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Mul<f64> for C64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Div for C64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        if rhs.re.abs() >= rhs.im.abs() {
            let r = rhs.im / rhs.re;
            let denom = rhs.re + rhs.im * r;
            Self::new(
                (self.re + self.im * r) / denom,
                (self.im - self.re * r) / denom,
            )
        } else {
            let r = rhs.re / rhs.im;
            let denom = rhs.im + rhs.re * r;
            Self::new(
                (self.re * r + self.im) / denom,
                (self.im * r - self.re) / denom,
            )
        }
    }
}

/// 2x2 complex Bloch–McConnell Liouvillian for one spectral dimension:
/// `L = [[-R2A + i*wA - kAB, kBA], [kAB, -R2B + i*wB - kBA]]`.
#[derive(Clone, Copy, Debug)]
struct Liouvillian2x2 {
    a: C64,
    b: C64,
    c: C64,
    d: C64,
}

impl Liouvillian2x2 {
    #[inline]
    fn new(r2a: f64, wa: f64, r2b: f64, wb: f64, k_ab: f64, k_ba: f64) -> Self {
        Self {
            a: C64::new(-r2a - k_ab, wa),
            b: C64::new(k_ba, 0.0),
            c: C64::new(k_ab, 0.0),
            d: C64::new(-r2b - k_ba, wb),
        }
    }

    /// Exact closed-form 2x2 complex matrix exponential `exp(L * t)` for `t >= 0`,
    /// completely free of exceptional-point singularities (`delta -> 0`) and overflow.
    fn expm(self, t: f64) -> [[C64; 2]; 2] {
        let mu = (self.a + self.d) * 0.5;
        let half_diff = (self.a - self.d) * 0.5;
        let delta = (half_diff * half_diff + self.b * self.c).sqrt();
        let z = delta * t;

        let (c_id, c_b) = if z.norm() >= 1e-4 {
            let e1 = ((mu + delta) * t).exp();
            let e2 = ((mu - delta) * t).exp();
            ((e1 + e2) * 0.5, (e1 - e2) / (delta * 2.0))
        } else {
            // Exact Taylor expansion around exceptional point z = 0 (error < 1e-32 for |z| < 1e-4).
            let z2 = z * z;
            let z4 = z2 * z2;
            let z6 = z4 * z2;
            let cosh_z = C64::ONE + z2 * (1.0 / 2.0) + z4 * (1.0 / 24.0) + z6 * (1.0 / 720.0);
            let sinhc_z = C64::ONE + z2 * (1.0 / 6.0) + z4 * (1.0 / 120.0) + z6 * (1.0 / 5040.0);
            let emu = (mu * t).exp();
            (emu * cosh_z, emu * (sinhc_z * t))
        };

        [
            [c_id + c_b * half_diff, c_b * self.b],
            [c_b * self.c, c_id - c_b * half_diff],
        ]
    }

    /// Exact Cramer-rule resolvent column `Re[(i*omega*I - L)^-1 * p0]` (2-element real vector).
    /// Non-singular for all real `omega` even at the exact coalescence exceptional point.
    #[inline]
    fn resolvent_col_real(self, omega: f64, p_a: f64, p_b: f64) -> [f64; 2] {
        let m11 = C64::new(-self.a.re, omega - self.a.im);
        let m22 = C64::new(-self.d.re, omega - self.d.im);
        let m12 = -self.b;
        let m21 = -self.c;
        let det = m11 * m22 - m12 * m21;
        let v0 = (m22 * p_a - m12 * p_b) / det;
        let v1 = (-m21 * p_a + m11 * p_b) / det;
        [v0.re, v1.re]
    }

    /// Exact Cramer-rule resolvent row `Re[1^T * (i*omega*I - L)^-1]` (2-element real vector).
    #[inline]
    fn resolvent_row_real(self, omega: f64) -> [f64; 2] {
        let m11 = C64::new(-self.a.re, omega - self.a.im);
        let m22 = C64::new(-self.d.re, omega - self.d.im);
        let m12 = -self.b;
        let m21 = -self.c;
        let det = m11 * m22 - m12 * m21;
        let r0 = (m22 - m21) / det;
        let r1 = (-m12 + m11) / det;
        [r0.re, r1.re]
    }
}

/// Precomputed 2x2 spectral projector representation of a Liouvillian for explicit eigenmode
/// Lorentzian evaluation (falls back seamlessly to exact resolvent at exceptional points).
struct SpectralDecomp2x2 {
    liouvillian: Liouvillian2x2,
    use_eigenmodes: bool,
    lambda1: C64,
    lambda2: C64,
    /// `P1 * p0` and `P2 * p0` (for F1 column) or `1^T * P1` and `1^T * P2` (for F2 row).
    mode1_vec: [C64; 2],
    mode2_vec: [C64; 2],
}

impl SpectralDecomp2x2 {
    fn for_col(l: Liouvillian2x2, p_a: f64, p_b: f64) -> Self {
        let mu = (l.a + l.d) * 0.5;
        let half_diff = (l.a - l.d) * 0.5;
        let delta = (half_diff * half_diff + l.b * l.c).sqrt();
        let scale = 1.0 + mu.norm();
        if delta.norm() > 1e-6 * scale {
            let inv_2d = C64::ONE / (delta * 2.0);
            // P1 = 0.5*I + inv_2d * B, P2 = 0.5*I - inv_2d * B
            let b_p0 = [half_diff * p_a + l.b * p_b, l.c * p_a - half_diff * p_b];
            let half_p0 = [C64::new(0.5 * p_a, 0.0), C64::new(0.5 * p_b, 0.0)];
            let m1 = [half_p0[0] + inv_2d * b_p0[0], half_p0[1] + inv_2d * b_p0[1]];
            let m2 = [half_p0[0] - inv_2d * b_p0[0], half_p0[1] - inv_2d * b_p0[1]];
            Self {
                liouvillian: l,
                use_eigenmodes: true,
                lambda1: mu + delta,
                lambda2: mu - delta,
                mode1_vec: m1,
                mode2_vec: m2,
            }
        } else {
            Self {
                liouvillian: l,
                use_eigenmodes: false,
                lambda1: mu,
                lambda2: mu,
                mode1_vec: [C64::ZERO; 2],
                mode2_vec: [C64::ZERO; 2],
            }
        }
    }

    fn for_row(l: Liouvillian2x2) -> Self {
        let mu = (l.a + l.d) * 0.5;
        let half_diff = (l.a - l.d) * 0.5;
        let delta = (half_diff * half_diff + l.b * l.c).sqrt();
        let scale = 1.0 + mu.norm();
        if delta.norm() > 1e-6 * scale {
            let inv_2d = C64::ONE / (delta * 2.0);
            // 1^T * B = [half_diff + c, b - half_diff]
            let one_b = [half_diff + l.c, l.b - half_diff];
            let half_one = C64::new(0.5, 0.0);
            let m1 = [half_one + inv_2d * one_b[0], half_one + inv_2d * one_b[1]];
            let m2 = [half_one - inv_2d * one_b[0], half_one - inv_2d * one_b[1]];
            Self {
                liouvillian: l,
                use_eigenmodes: true,
                lambda1: mu + delta,
                lambda2: mu - delta,
                mode1_vec: m1,
                mode2_vec: m2,
            }
        } else {
            Self {
                liouvillian: l,
                use_eigenmodes: false,
                lambda1: mu,
                lambda2: mu,
                mode1_vec: [C64::ZERO; 2],
                mode2_vec: [C64::ZERO; 2],
            }
        }
    }

    #[inline]
    fn eval_col_real(&self, omega: f64, p_a: f64, p_b: f64) -> [f64; 2] {
        if self.use_eigenmodes {
            let d1 = C64::new(-self.lambda1.re, omega - self.lambda1.im);
            let d2 = C64::new(-self.lambda2.re, omega - self.lambda2.im);
            let v0 = self.mode1_vec[0] / d1 + self.mode2_vec[0] / d2;
            let v1 = self.mode1_vec[1] / d1 + self.mode2_vec[1] / d2;
            [v0.re, v1.re]
        } else {
            self.liouvillian.resolvent_col_real(omega, p_a, p_b)
        }
    }

    #[inline]
    fn eval_row_real(&self, omega: f64) -> [f64; 2] {
        if self.use_eigenmodes {
            let d1 = C64::new(-self.lambda1.re, omega - self.lambda1.im);
            let d2 = C64::new(-self.lambda2.re, omega - self.lambda2.im);
            let r0 = self.mode1_vec[0] / d1 + self.mode2_vec[0] / d2;
            let r1 = self.mode1_vec[1] / d1 + self.mode2_vec[1] / d2;
            [r0.re, r1.re]
        } else {
            self.liouvillian.resolvent_row_real(omega)
        }
    }
}

// ============================================================================
// 4. Coupled 2D Bloch–McConnell Lineshape Simulator (Analytical & Time-Domain FFT)
// ============================================================================

/// Per-residue 2-state 2D `(15N, 1H)` Bloch–McConnell lineshape parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bloch2dResidueParams {
    /// Free-state (apo) `1H` chemical shift (`ppm`).
    pub delta_a_h_ppm: f64,
    /// Bound-state (holo) `1H` chemical shift (`ppm`).
    pub delta_b_h_ppm: f64,
    /// Free-state (apo) `15N` chemical shift (`ppm`).
    pub delta_a_n_ppm: f64,
    /// Bound-state (holo) `15N` chemical shift (`ppm`).
    pub delta_b_n_ppm: f64,
    /// Free-state `1H` transverse relaxation rate `R2A_H` (`s^-1`).
    pub r2a_h_s_inv: f64,
    /// Bound-state `1H` transverse relaxation rate `R2B_H` (`s^-1`).
    pub r2b_h_s_inv: f64,
    /// Free-state `15N` transverse relaxation rate `R2A_N` (`s^-1`).
    pub r2a_n_s_inv: f64,
    /// Bound-state `15N` transverse relaxation rate `R2B_N` (`s^-1`).
    pub r2b_n_s_inv: f64,
    /// Overall integrated amplitude scale `I_0` (positive).
    pub amplitude: f64,
}

/// Two-state kinetic/population state for a single titration point.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bloch2dKineticState {
    pub p_a: f64,
    pub p_b: f64,
    pub k_ab_s_inv: f64,
    pub k_ba_s_inv: f64,
}

impl From<BindingEquilibriumState> for Bloch2dKineticState {
    fn from(eq: BindingEquilibriumState) -> Self {
        Self {
            p_a: eq.p_a,
            p_b: eq.p_b,
            k_ab_s_inv: eq.k_ab_s_inv,
            k_ba_s_inv: eq.k_ba_s_inv,
        }
    }
}

/// 2D `(F1 = 15N, F2 = 1H)` frequency grid for an ROI patch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Roi2dGrid {
    /// Indirect `15N` chemical shift axis (`ppm`, length `n1`).
    pub n_ppm_axis: Vec<f64>,
    /// Direct `1H` chemical shift axis (`ppm`, length `n2`).
    pub h_ppm_axis: Vec<f64>,
    /// Spectrometer `1H` Larmor frequency (`MHz`, e.g. `600.0`).
    pub h_larmor_mhz: f64,
    /// Spectrometer `15N` Larmor frequency (`MHz`, e.g. `60.8`).
    pub n_larmor_mhz: f64,
}

impl Roi2dGrid {
    /// Construct a uniform 2D `(15N, 1H)` ROI grid centered around `(center_n_ppm, center_h_ppm)`.
    pub fn uniform(
        n_min_ppm: f64,
        n_max_ppm: f64,
        n1: usize,
        h_min_ppm: f64,
        h_max_ppm: f64,
        n2: usize,
        h_larmor_mhz: f64,
    ) -> Result<Self, Titration2dError> {
        if n1 == 0 || n2 == 0 {
            return Err(Titration2dError::EmptyInput(
                "ROI grid dimensions n1 and n2 must be >= 1".to_string(),
            ));
        }
        validate_finite("n_min_ppm", n_min_ppm)?;
        validate_finite("n_max_ppm", n_max_ppm)?;
        validate_finite("h_min_ppm", h_min_ppm)?;
        validate_finite("h_max_ppm", h_max_ppm)?;
        validate_positive("h_larmor_mhz", h_larmor_mhz)?;

        let n_ppm_axis = if n1 == 1 {
            vec![0.5 * (n_min_ppm + n_max_ppm)]
        } else {
            (0..n1)
                .map(|i| n_min_ppm + (n_max_ppm - n_min_ppm) * (i as f64) / ((n1 - 1) as f64))
                .collect()
        };
        let h_ppm_axis = if n2 == 1 {
            vec![0.5 * (h_min_ppm + h_max_ppm)]
        } else {
            (0..n2)
                .map(|j| h_min_ppm + (h_max_ppm - h_min_ppm) * (j as f64) / ((n2 - 1) as f64))
                .collect()
        };
        Ok(Self {
            n_ppm_axis,
            h_ppm_axis,
            h_larmor_mhz,
            n_larmor_mhz: h_larmor_mhz * GAMMA_N15_OVER_H1,
        })
    }

    fn validate(&self) -> Result<(), Titration2dError> {
        if self.n_ppm_axis.is_empty() || self.h_ppm_axis.is_empty() {
            return Err(Titration2dError::EmptyInput(
                "Roi2dGrid axes must be non-empty".to_string(),
            ));
        }
        validate_positive("h_larmor_mhz", self.h_larmor_mhz)?;
        validate_positive("n_larmor_mhz", self.n_larmor_mhz)?;
        for &v in &self.n_ppm_axis {
            validate_finite("n_ppm_axis", v)?;
        }
        for &v in &self.h_ppm_axis {
            validate_finite("h_ppm_axis", v)?;
        }
        Ok(())
    }
}

fn validate_bloch_params(params: &Bloch2dResidueParams) -> Result<(), Titration2dError> {
    validate_finite("delta_a_h_ppm", params.delta_a_h_ppm)?;
    validate_finite("delta_b_h_ppm", params.delta_b_h_ppm)?;
    validate_finite("delta_a_n_ppm", params.delta_a_n_ppm)?;
    validate_finite("delta_b_n_ppm", params.delta_b_n_ppm)?;
    validate_positive("r2a_h_s_inv", params.r2a_h_s_inv)?;
    validate_positive("r2b_h_s_inv", params.r2b_h_s_inv)?;
    validate_positive("r2a_n_s_inv", params.r2a_n_s_inv)?;
    validate_positive("r2b_n_s_inv", params.r2b_n_s_inv)?;
    validate_positive("amplitude", params.amplitude)?;
    Ok(())
}

fn validate_kinetic_state(state: &Bloch2dKineticState) -> Result<(), Titration2dError> {
    validate_non_negative("p_a", state.p_a)?;
    validate_non_negative("p_b", state.p_b)?;
    validate_non_negative("k_ab_s_inv", state.k_ab_s_inv)?;
    validate_non_negative("k_ba_s_inv", state.k_ba_s_inv)?;
    if state.p_a + state.p_b <= 0.0 {
        return Err(Titration2dError::InvalidParameter {
            name: "p_a + p_b".to_string(),
            value: state.p_a + state.p_b,
            reason: "total population must be > 0".to_string(),
        });
    }
    Ok(())
}

/// Evaluate the exact analytical hypercomplex pure-absorption 2D Bloch–McConnell lineshape
/// `I(Omega_1, Omega_2) = I_0 * Re[1^T (i*Omega_2*I - L_2)^-1] * Re[(i*Omega_1*I - L_1)^-1 p_0]`
/// over a 2D `(F1 = 15N, F2 = 1H)` ROI grid (row-major `n1 * n2`), with zero FFT truncation
/// artifacts and unconditional stability through coalescence exceptional points.
pub fn simulate_2d_lineshape_analytical(
    grid: &Roi2dGrid,
    params: &Bloch2dResidueParams,
    state: &Bloch2dKineticState,
) -> Result<Vec<f64>, Titration2dError> {
    grid.validate()?;
    validate_bloch_params(params)?;
    validate_kinetic_state(state)?;

    let mut out = vec![0.0_f64; grid.n_ppm_axis.len() * grid.h_ppm_axis.len()];
    eval_2d_lineshape_into(grid, params, state, 1.0, &mut out);
    Ok(out)
}

#[inline]
fn eval_2d_lineshape_into(
    grid: &Roi2dGrid,
    params: &Bloch2dResidueParams,
    state: &Bloch2dKineticState,
    extra_scale: f64,
    out: &mut [f64],
) {
    let n1 = grid.n_ppm_axis.len();
    let n2 = grid.h_ppm_axis.len();
    // Shift origin to midpoint of free/bound shifts to avoid floating-point cancellation
    // from large absolute ppm offsets (e.g. 120 ppm * 60.8 MHz).
    let ref_n = 0.5 * (params.delta_a_n_ppm + params.delta_b_n_ppm);
    let ref_h = 0.5 * (params.delta_a_h_ppm + params.delta_b_h_ppm);
    let scale_n = TAU * grid.n_larmor_mhz;
    let scale_h = TAU * grid.h_larmor_mhz;

    let wa1 = scale_n * (params.delta_a_n_ppm - ref_n);
    let wb1 = scale_n * (params.delta_b_n_ppm - ref_n);
    let wa2 = scale_h * (params.delta_a_h_ppm - ref_h);
    let wb2 = scale_h * (params.delta_b_h_ppm - ref_h);

    let l1 = Liouvillian2x2::new(
        params.r2a_n_s_inv,
        wa1,
        params.r2b_n_s_inv,
        wb1,
        state.k_ab_s_inv,
        state.k_ba_s_inv,
    );
    let l2 = Liouvillian2x2::new(
        params.r2a_h_s_inv,
        wa2,
        params.r2b_h_s_inv,
        wb2,
        state.k_ab_s_inv,
        state.k_ba_s_inv,
    );

    let mut c1 = vec![[0.0_f64; 2]; n1];
    for (i, &n_ppm) in grid.n_ppm_axis.iter().enumerate() {
        let omega1 = scale_n * (n_ppm - ref_n);
        c1[i] = l1.resolvent_col_real(omega1, state.p_a, state.p_b);
    }

    let mut r2 = vec![[0.0_f64; 2]; n2];
    for (j, &h_ppm) in grid.h_ppm_axis.iter().enumerate() {
        let omega2 = scale_h * (h_ppm - ref_h);
        r2[j] = l2.resolvent_row_real(omega2);
    }

    let amp = params.amplitude * extra_scale;
    for i in 0..n1 {
        let c1a = amp * c1[i][0];
        let c1b = amp * c1[i][1];
        let row_offset = i * n2;
        for j in 0..n2 {
            out[row_offset + j] = c1a * r2[j][0] + c1b * r2[j][1];
        }
    }
}

/// Evaluate the exact 2D Bloch–McConnell lineshape explicitly via the 2x2 eigenmode spectral
/// projector Lorentzian double-sum (with seamless resolvent fallback at exceptional points).
pub fn simulate_2d_lineshape_eigenmodes(
    grid: &Roi2dGrid,
    params: &Bloch2dResidueParams,
    state: &Bloch2dKineticState,
) -> Result<Vec<f64>, Titration2dError> {
    grid.validate()?;
    validate_bloch_params(params)?;
    validate_kinetic_state(state)?;

    let n1 = grid.n_ppm_axis.len();
    let n2 = grid.h_ppm_axis.len();
    let ref_n = 0.5 * (params.delta_a_n_ppm + params.delta_b_n_ppm);
    let ref_h = 0.5 * (params.delta_a_h_ppm + params.delta_b_h_ppm);
    let scale_n = TAU * grid.n_larmor_mhz;
    let scale_h = TAU * grid.h_larmor_mhz;

    let l1 = Liouvillian2x2::new(
        params.r2a_n_s_inv,
        scale_n * (params.delta_a_n_ppm - ref_n),
        params.r2b_n_s_inv,
        scale_n * (params.delta_b_n_ppm - ref_n),
        state.k_ab_s_inv,
        state.k_ba_s_inv,
    );
    let l2 = Liouvillian2x2::new(
        params.r2a_h_s_inv,
        scale_h * (params.delta_a_h_ppm - ref_h),
        params.r2b_h_s_inv,
        scale_h * (params.delta_b_h_ppm - ref_h),
        state.k_ab_s_inv,
        state.k_ba_s_inv,
    );

    let spec1 = SpectralDecomp2x2::for_col(l1, state.p_a, state.p_b);
    let spec2 = SpectralDecomp2x2::for_row(l2);

    let mut c1 = vec![[0.0_f64; 2]; n1];
    for (i, &n_ppm) in grid.n_ppm_axis.iter().enumerate() {
        let omega1 = scale_n * (n_ppm - ref_n);
        c1[i] = spec1.eval_col_real(omega1, state.p_a, state.p_b);
    }

    let mut r2 = vec![[0.0_f64; 2]; n2];
    for (j, &h_ppm) in grid.h_ppm_axis.iter().enumerate() {
        let omega2 = scale_h * (h_ppm - ref_h);
        r2[j] = spec2.eval_row_real(omega2);
    }

    let mut out = vec![0.0_f64; n1 * n2];
    let amp = params.amplitude;
    for i in 0..n1 {
        let c1a = amp * c1[i][0];
        let c1b = amp * c1[i][1];
        let row_offset = i * n2;
        for j in 0..n2 {
            out[row_offset + j] = c1a * r2[j][0] + c1b * r2[j][1];
        }
    }
    Ok(out)
}

/// In-place iterative radix-2 Cooley–Tukey forward DFT (`X[k] = sum_n x[n] exp(-i*2*pi*k*n/N)`).
fn fft_radix2_inplace(buf: &mut [C64]) -> Result<(), Titration2dError> {
    let n = buf.len();
    if n == 0 || !n.is_power_of_two() {
        return Err(Titration2dError::InvalidParameter {
            name: "fft_len".to_string(),
            value: n as f64,
            reason: "must be a power of two >= 1".to_string(),
        });
    }
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let angle = -TAU / (len as f64);
        let wlen = C64::new(angle.cos(), angle.sin());
        let mut i = 0usize;
        while i < n {
            let mut w = C64::ONE;
            for k in 0..half {
                let u = buf[i + k];
                let v = buf[i + k + half] * w;
                buf[i + k] = u + v;
                buf[i + k + half] = u - v;
                w = w * wlen;
            }
            i += len;
        }
        len <<= 1;
    }
    Ok(())
}

/// Reference numerical time-domain 2D States-TPPI Bloch–McConnell simulator computing
/// `exp(L_1 t_1) p_0` followed by `1^T exp(L_2 t_2)` via exact 2x2 matrix exponentials and
/// hypercomplex 2D FFT (returned fftshifted in ascending chemical shift order).
#[allow(clippy::too_many_arguments)]
pub fn simulate_2d_lineshape_time_domain_fft(
    n1: usize,
    n2: usize,
    sw1_hz: f64,
    sw2_hz: f64,
    carrier_n_ppm: f64,
    carrier_h_ppm: f64,
    h_larmor_mhz: f64,
    n_larmor_mhz: f64,
    params: &Bloch2dResidueParams,
    state: &Bloch2dKineticState,
) -> Result<(Roi2dGrid, Vec<f64>), Titration2dError> {
    if n1 < 2 || !n1.is_power_of_two() || n2 < 2 || !n2.is_power_of_two() {
        return Err(Titration2dError::InvalidParameter {
            name: "n1/n2".to_string(),
            value: (n1.min(n2)) as f64,
            reason: "FFT sizes n1 and n2 must be powers of two >= 2".to_string(),
        });
    }
    validate_positive("sw1_hz", sw1_hz)?;
    validate_positive("sw2_hz", sw2_hz)?;
    validate_finite("carrier_n_ppm", carrier_n_ppm)?;
    validate_finite("carrier_h_ppm", carrier_h_ppm)?;
    validate_positive("h_larmor_mhz", h_larmor_mhz)?;
    validate_positive("n_larmor_mhz", n_larmor_mhz)?;
    validate_bloch_params(params)?;
    validate_kinetic_state(state)?;

    let dt1 = 1.0 / sw1_hz;
    let dt2 = 1.0 / sw2_hz;
    let wa1 = TAU * n_larmor_mhz * (params.delta_a_n_ppm - carrier_n_ppm);
    let wb1 = TAU * n_larmor_mhz * (params.delta_b_n_ppm - carrier_n_ppm);
    let wa2 = TAU * h_larmor_mhz * (params.delta_a_h_ppm - carrier_h_ppm);
    let wb2 = TAU * h_larmor_mhz * (params.delta_b_h_ppm - carrier_h_ppm);

    let l1 = Liouvillian2x2::new(
        params.r2a_n_s_inv,
        wa1,
        params.r2b_n_s_inv,
        wb1,
        state.k_ab_s_inv,
        state.k_ba_s_inv,
    );
    let l2 = Liouvillian2x2::new(
        params.r2a_h_s_inv,
        wa2,
        params.r2b_h_s_inv,
        wb2,
        state.k_ab_s_inv,
        state.k_ba_s_inv,
    );

    // Precompute v(t1) = exp(L1 t1) p0 (with first-point 0.5 Riemann weight for one-sided FT).
    let mut v1 = vec![[C64::ZERO; 2]; n1];
    for (k1, slot) in v1.iter_mut().enumerate() {
        let t1 = (k1 as f64) * dt1;
        let e1 = l1.expm(t1);
        let w = if k1 == 0 { 0.5 * dt1 } else { dt1 };
        slot[0] = (e1[0][0] * state.p_a + e1[0][1] * state.p_b) * w;
        slot[1] = (e1[1][0] * state.p_a + e1[1][1] * state.p_b) * w;
    }

    // Precompute w2(t2)^T = 1^T exp(L2 t2) (with first-point 0.5 Riemann weight).
    let mut w2 = vec![[C64::ZERO; 2]; n2];
    for (k2, slot) in w2.iter_mut().enumerate() {
        let t2 = (k2 as f64) * dt2;
        let e2 = l2.expm(t2);
        let w = if k2 == 0 { 0.5 * dt2 } else { dt2 };
        slot[0] = (e2[0][0] + e2[1][0]) * w;
        slot[1] = (e2[0][1] + e2[1][1]) * w;
    }

    // States-TPPI hypercomplex 2D FFT:
    // 1. FFT along t2 for cos (Re[v1]) and sin (Im[v1]), taking Re in F2.
    let mut interferogram = vec![C64::ZERO; n1 * n2];
    let mut buf_cos = vec![C64::ZERO; n2];
    let mut buf_sin = vec![C64::ZERO; n2];

    for k1 in 0..n1 {
        let vr0 = v1[k1][0].re;
        let vr1 = v1[k1][1].re;
        let vi0 = v1[k1][0].im;
        let vi1 = v1[k1][1].im;
        for k2 in 0..n2 {
            buf_cos[k2] = w2[k2][0] * vr0 + w2[k2][1] * vr1;
            buf_sin[k2] = w2[k2][0] * vi0 + w2[k2][1] * vi1;
        }
        fft_radix2_inplace(&mut buf_cos)?;
        fft_radix2_inplace(&mut buf_sin)?;

        let row_off = k1 * n2;
        for f2 in 0..n2 {
            interferogram[row_off + f2] = C64::new(buf_cos[f2].re, buf_sin[f2].re);
        }
    }

    // 2. FFT along t1 of hypercomplex interferogram, taking Re in F1 and fftshifting both axes.
    let mut col_buf = vec![C64::ZERO; n1];
    let mut spec = vec![0.0_f64; n1 * n2];
    let half1 = n1 / 2;
    let half2 = n2 / 2;
    let amp = params.amplitude;

    for f2 in 0..n2 {
        for k1 in 0..n1 {
            col_buf[k1] = interferogram[k1 * n2 + f2];
        }
        fft_radix2_inplace(&mut col_buf)?;
        let out_j = (f2 + half2) % n2;
        for f1 in 0..n1 {
            let out_i = (f1 + half1) % n1;
            spec[out_i * n2 + out_j] = amp * col_buf[f1].re;
        }
    }

    // Build ascending ppm axes matching the fftshifted bins [-N/2 .. N/2 - 1].
    let df1_ppm = sw1_hz / ((n1 as f64) * n_larmor_mhz);
    let df2_ppm = sw2_hz / ((n2 as f64) * h_larmor_mhz);
    let n_ppm_axis: Vec<f64> = (0..n1)
        .map(|i| carrier_n_ppm + ((i as isize) - (half1 as isize)) as f64 * df1_ppm)
        .collect();
    let h_ppm_axis: Vec<f64> = (0..n2)
        .map(|j| carrier_h_ppm + ((j as isize) - (half2 as isize)) as f64 * df2_ppm)
        .collect();

    Ok((
        Roi2dGrid {
            n_ppm_axis,
            h_ppm_axis,
            h_larmor_mhz,
            n_larmor_mhz,
        },
        spec,
    ))
}

// ============================================================================
// 5. Global Multi-Residue 2D Bloch–McConnell Levenberg–Marquardt Fitter
// ============================================================================

/// Observed 2D ROI patch for one residue at one ligand titration step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Roi2dPatch {
    pub step_index: usize,
    pub protein_conc_um: f64,
    pub ligand_conc_um: f64,
    pub grid: Roi2dGrid,
    /// Observed 2D intensities in row-major order (`n1 * n2`).
    pub intensities: Vec<f64>,
    /// Per-pixel Gaussian noise standard deviation (`> 0`).
    pub noise_sigma: f64,
    /// Optional known per-spectrum scaling factor (e.g. `1.0` or dilution factor `[P]_t / [P]_0`).
    pub spectrum_scale: f64,
}

/// Complete titration ROI series for a single residue entering global 2D lineshape fitting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidueTitration2dSeries {
    pub label: String,
    pub res_name: String,
    pub patches: Vec<Roi2dPatch>,
    /// Optional user-supplied initial parameter guess (auto-estimated from ROI patches when `None`).
    pub initial_guess: Option<Bloch2dResidueParams>,
}

/// Configuration for global multi-residue 2D Bloch–McConnell Levenberg–Marquardt fitting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Titration2dFitConfig {
    pub initial_kd_um: Option<f64>,
    pub initial_k_off_s_inv: Option<f64>,
    pub kd_bounds_um: (f64, f64),
    pub k_off_bounds_s_inv: (f64, f64),
    pub r2_bounds_s_inv: (f64, f64),
    pub max_lm_iterations: usize,
    pub rel_chi2_tol: f64,
    pub rel_param_tol: f64,
    /// Number of top coarse-grid starting basins polished with full Schur-complement LM.
    pub n_coarse_starts: usize,
    /// Combined CSP uncertainty (`ppm`) used for the fast-exchange quadratic CSP comparison.
    pub csp_sigma_ppm: f64,
}

impl Default for Titration2dFitConfig {
    fn default() -> Self {
        Self {
            initial_kd_um: None,
            initial_k_off_s_inv: None,
            kd_bounds_um: (1e-3, 1e5),
            k_off_bounds_s_inv: (0.1, 1e5),
            r2_bounds_s_inv: (0.5, 1000.0),
            max_lm_iterations: 80,
            rel_chi2_tol: 1e-9,
            rel_param_tol: 1e-8,
            n_coarse_starts: 3,
            csp_sigma_ppm: 0.004,
        }
    }
}

/// Fitted 2D Bloch–McConnell parameters and Cramér–Rao uncertainties for one residue.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FittedResidue2d {
    pub label: String,
    pub res_name: String,
    pub params: Bloch2dResidueParams,
    /// `1-sigma` Cramér–Rao standard errors derived from the inverse Fisher information matrix.
    pub uncertainties: Bloch2dResidueParams,
    pub dw_h_ppm: f64,
    pub dw_h_sd_ppm: f64,
    pub dw_n_ppm: f64,
    pub dw_n_sd_ppm: f64,
    pub dw_h_rad_s: f64,
    pub dw_n_rad_s: f64,
    pub chi2: f64,
    pub n_pixels: usize,
    pub exchange_regime: ResidueExchangeRegimeReport,
}

/// Fast-exchange quadratic CSP isotherm fit comparison against coupled 2D Bloch–McConnell analysis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastExchangeCspComparison {
    pub csp_kd_um: f64,
    pub csp_chi2: f64,
    pub kd_relative_discrepancy: f64,
    pub per_residue_ddelta_max_ppm: Vec<(String, f64)>,
    /// `true` iff every residue at every titration point is strictly in fast exchange.
    pub fast_exchange_assumption_valid: bool,
    pub warnings: Vec<ExchangeRegimeWarning>,
}

/// Global multi-residue coupled 2D Bloch–McConnell titration fit report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Titration2dFitResult {
    pub kd_um: f64,
    pub kd_sd_um: f64,
    pub k_off_s_inv: f64,
    pub k_off_sd_s_inv: f64,
    pub k_on_um_inv_s_inv: f64,
    pub k_on_sd_um_inv_s_inv: f64,
    pub chi2: f64,
    pub reduced_chi2: f64,
    pub dof: usize,
    pub n_pixels_total: usize,
    pub n_params: usize,
    pub lm_iterations: usize,
    pub converged: bool,
    pub residues: Vec<FittedResidue2d>,
    pub fast_exchange_comparison: FastExchangeCspComparison,
}

/// Internal per-residue unconstrained/box-constrained parameter vector (9 params):
/// `[delta_a_h, delta_b_h, delta_a_n, delta_b_n, ln_r2a_h, ln_r2b_h, ln_r2a_n, ln_r2b_n, ln_amp]`.
const N_LOCAL: usize = 9;

#[derive(Clone, Debug)]
struct LocalBounds {
    h_min: f64,
    h_max: f64,
    n_min: f64,
    n_max: f64,
    ln_r2_min: f64,
    ln_r2_max: f64,
}

fn params_to_local(p: &Bloch2dResidueParams) -> [f64; N_LOCAL] {
    [
        p.delta_a_h_ppm,
        p.delta_b_h_ppm,
        p.delta_a_n_ppm,
        p.delta_b_n_ppm,
        p.r2a_h_s_inv.ln(),
        p.r2b_h_s_inv.ln(),
        p.r2a_n_s_inv.ln(),
        p.r2b_n_s_inv.ln(),
        p.amplitude.ln(),
    ]
}

fn local_to_params(v: &[f64; N_LOCAL]) -> Bloch2dResidueParams {
    Bloch2dResidueParams {
        delta_a_h_ppm: v[0],
        delta_b_h_ppm: v[1],
        delta_a_n_ppm: v[2],
        delta_b_n_ppm: v[3],
        r2a_h_s_inv: v[4].exp(),
        r2b_h_s_inv: v[5].exp(),
        r2a_n_s_inv: v[6].exp(),
        r2b_n_s_inv: v[7].exp(),
        amplitude: v[8].exp(),
    }
}

fn clamp_local(v: &mut [f64; N_LOCAL], b: &LocalBounds) {
    v[0] = v[0].clamp(b.h_min, b.h_max);
    v[1] = v[1].clamp(b.h_min, b.h_max);
    v[2] = v[2].clamp(b.n_min, b.n_max);
    v[3] = v[3].clamp(b.n_min, b.n_max);
    v[4] = v[4].clamp(b.ln_r2_min, b.ln_r2_max);
    v[5] = v[5].clamp(b.ln_r2_min, b.ln_r2_max);
    v[6] = v[6].clamp(b.ln_r2_min, b.ln_r2_max);
    v[7] = v[7].clamp(b.ln_r2_min, b.ln_r2_max);
    v[8] = v[8].clamp(-30.0, 40.0);
}

fn compute_local_bounds(
    res: &ResidueTitration2dSeries,
    config: &Titration2dFitConfig,
) -> LocalBounds {
    let mut h_min = f64::INFINITY;
    let mut h_max = f64::NEG_INFINITY;
    let mut n_min = f64::INFINITY;
    let mut n_max = f64::NEG_INFINITY;
    for patch in &res.patches {
        for &h in &patch.grid.h_ppm_axis {
            h_min = h_min.min(h);
            h_max = h_max.max(h);
        }
        for &n in &patch.grid.n_ppm_axis {
            n_min = n_min.min(n);
            n_max = n_max.max(n);
        }
    }
    let h_pad = ((h_max - h_min).abs() * 0.5).max(0.25);
    let n_pad = ((n_max - n_min).abs() * 0.5).max(1.5);
    LocalBounds {
        h_min: h_min - h_pad,
        h_max: h_max + h_pad,
        n_min: n_min - n_pad,
        n_max: n_max + n_pad,
        ln_r2_min: config.r2_bounds_s_inv.0.ln(),
        ln_r2_max: config.r2_bounds_s_inv.1.ln(),
    }
}

/// Extract sub-pixel maximum `(h_ppm, n_ppm, max_val, i1_max, i2_max)` from a 2D ROI patch.
fn find_patch_peak_subpixel(patch: &Roi2dPatch) -> (f64, f64, f64, usize, usize) {
    let n1 = patch.grid.n_ppm_axis.len();
    let n2 = patch.grid.h_ppm_axis.len();
    let mut best_i = 0usize;
    let mut best_j = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for i in 0..n1 {
        let row_off = i * n2;
        for j in 0..n2 {
            let v = patch.intensities[row_off + j];
            if v > best_v {
                best_v = v;
                best_i = i;
                best_j = j;
            }
        }
    }

    let n_ppm = if best_i > 0 && best_i + 1 < n1 {
        let ym = patch.intensities[(best_i - 1) * n2 + best_j];
        let y0 = best_v;
        let yp = patch.intensities[(best_i + 1) * n2 + best_j];
        let denom = ym - 2.0 * y0 + yp;
        let frac = if denom.abs() > 1e-18 {
            (0.5 * (ym - yp) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let step_n = 0.5 * (patch.grid.n_ppm_axis[best_i + 1] - patch.grid.n_ppm_axis[best_i - 1]);
        patch.grid.n_ppm_axis[best_i] + frac * step_n
    } else {
        patch.grid.n_ppm_axis[best_i]
    };

    let h_ppm = if best_j > 0 && best_j + 1 < n2 {
        let ym = patch.intensities[best_i * n2 + (best_j - 1)];
        let y0 = best_v;
        let yp = patch.intensities[best_i * n2 + (best_j + 1)];
        let denom = ym - 2.0 * y0 + yp;
        let frac = if denom.abs() > 1e-18 {
            (0.5 * (ym - yp) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let step_h = 0.5 * (patch.grid.h_ppm_axis[best_j + 1] - patch.grid.h_ppm_axis[best_j - 1]);
        patch.grid.h_ppm_axis[best_j] + frac * step_h
    } else {
        patch.grid.h_ppm_axis[best_j]
    };

    (h_ppm, n_ppm, best_v, best_i, best_j)
}

/// Estimate HWHM linewidth `R2` (`s^-1`) from a 1D slice through a 2D Lorentzian peak.
fn estimate_slice_r2(
    axis_ppm: &[f64],
    slice_vals: &[f64],
    peak_idx: usize,
    larmor_mhz: f64,
) -> f64 {
    let peak_val = slice_vals[peak_idx];
    if peak_val <= 0.0 || axis_ppm.len() < 3 {
        return 25.0;
    }
    let half = 0.5 * peak_val;
    let peak_ppm = axis_ppm[peak_idx];
    let mut widths = Vec::new();

    // Scan left
    for i in (0..peak_idx).rev() {
        if slice_vals[i] <= half {
            let v1 = slice_vals[i];
            let v2 = slice_vals[i + 1];
            let frac = if (v2 - v1).abs() > 1e-18 {
                ((half - v1) / (v2 - v1)).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let cross_ppm = axis_ppm[i] + frac * (axis_ppm[i + 1] - axis_ppm[i]);
            widths.push((peak_ppm - cross_ppm).abs());
            break;
        }
    }
    // Scan right
    for i in (peak_idx + 1)..axis_ppm.len() {
        if slice_vals[i] <= half {
            let v1 = slice_vals[i - 1];
            let v2 = slice_vals[i];
            let frac = if (v1 - v2).abs() > 1e-18 {
                ((v1 - half) / (v1 - v2)).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let cross_ppm = axis_ppm[i - 1] + frac * (axis_ppm[i] - axis_ppm[i - 1]);
            widths.push((cross_ppm - peak_ppm).abs());
            break;
        }
    }

    if widths.is_empty() {
        25.0
    } else {
        let hwhm_ppm = widths.iter().sum::<f64>() / (widths.len() as f64);
        (TAU * larmor_mhz * hwhm_ppm).clamp(2.0, 250.0)
    }
}

/// Auto-estimate initial per-residue 2D Bloch–McConnell parameters from the lowest- and
/// highest-ligand ROI patches (handling fast, intermediate, and slow exchange).
fn auto_init_residue_params(
    res: &ResidueTitration2dSeries,
    kd_guess_um: f64,
    bounds: &LocalBounds,
) -> Bloch2dResidueParams {
    if let Some(ref g) = res.initial_guess {
        let mut v = params_to_local(g);
        clamp_local(&mut v, bounds);
        return local_to_params(&v);
    }

    let mut min_idx = 0usize;
    let mut max_idx = 0usize;
    for (idx, p) in res.patches.iter().enumerate() {
        if p.ligand_conc_um < res.patches[min_idx].ligand_conc_um {
            min_idx = idx;
        }
        if p.ligand_conc_um > res.patches[max_idx].ligand_conc_um {
            max_idx = idx;
        }
    }
    let apo_patch = &res.patches[min_idx];
    let holo_patch = &res.patches[max_idx];

    let (ha, na, apo_max, ia, ja) = find_patch_peak_subpixel(apo_patch);
    let n1 = apo_patch.grid.n_ppm_axis.len();
    let n2 = apo_patch.grid.h_ppm_axis.len();
    let h_slice_a: Vec<f64> = (0..n2)
        .map(|j| apo_patch.intensities[ia * n2 + j])
        .collect();
    let n_slice_a: Vec<f64> = (0..n1)
        .map(|i| apo_patch.intensities[i * n2 + ja])
        .collect();
    let r2a_h = estimate_slice_r2(
        &apo_patch.grid.h_ppm_axis,
        &h_slice_a,
        ja,
        apo_patch.grid.h_larmor_mhz,
    );
    let r2a_n = estimate_slice_r2(
        &apo_patch.grid.n_ppm_axis,
        &n_slice_a,
        ia,
        apo_patch.grid.n_larmor_mhz,
    );

    // Estimate bound shift: if grids match, also inspect difference spectrum to locate slow-exchange
    // bound peaks even when holo_patch is < 50% saturated.
    let (mut hb, mut nb, _, ib, jb) = find_patch_peak_subpixel(holo_patch);
    let mut r2b_h = r2a_h;
    let mut r2b_n = r2a_n;

    let alpha_n = amide_alpha_n(&res.res_name);
    let sep_raw = (hb - ha).hypot(alpha_n * (nb - na));
    if sep_raw < 0.015
        && holo_patch.grid.n_ppm_axis.len() == n1
        && holo_patch.grid.h_ppm_axis.len() == n2
        && apo_max > 0.0
    {
        let ratio = (holo_patch.intensities[ia * n2 + ja] / apo_max).clamp(0.0, 1.0);
        let mut best_diff = f64::NEG_INFINITY;
        let mut di = ib;
        let mut dj = jb;
        for i in 0..n1 {
            for j in 0..n2 {
                let d =
                    holo_patch.intensities[i * n2 + j] - ratio * apo_patch.intensities[i * n2 + j];
                if d > best_diff {
                    best_diff = d;
                    di = i;
                    dj = j;
                }
            }
        }
        if best_diff > 2.0 * holo_patch.noise_sigma {
            hb = holo_patch.grid.h_ppm_axis[dj];
            nb = holo_patch.grid.n_ppm_axis[di];
        }
    } else {
        // Extrapolate fast/intermediate shift for partial saturation at the highest ligand point.
        let pb_last = quadratic_bound_fraction(
            holo_patch.protein_conc_um,
            holo_patch.ligand_conc_um,
            kd_guess_um,
        );
        if pb_last > 0.25 && pb_last < 0.98 {
            hb = ha + (hb - ha) / pb_last;
            nb = na + (nb - na) / pb_last;
        }
        let hn1 = holo_patch.grid.n_ppm_axis.len();
        let hn2 = holo_patch.grid.h_ppm_axis.len();
        let h_slice_b: Vec<f64> = (0..hn2)
            .map(|j| holo_patch.intensities[ib * hn2 + j])
            .collect();
        let n_slice_b: Vec<f64> = (0..hn1)
            .map(|i| holo_patch.intensities[i * hn2 + jb])
            .collect();
        r2b_h = estimate_slice_r2(
            &holo_patch.grid.h_ppm_axis,
            &h_slice_b,
            jb,
            holo_patch.grid.h_larmor_mhz,
        );
        r2b_n = estimate_slice_r2(
            &holo_patch.grid.n_ppm_axis,
            &n_slice_b,
            ib,
            holo_patch.grid.n_larmor_mhz,
        );
    }

    // Peak height of pure 2D Lorentzian at (wa1, wa2) is I_0 / (R2A_N * R2A_H).
    let amp = (apo_max.max(1e-6) * r2a_n * r2a_h / apo_patch.spectrum_scale.max(1e-9)).max(1e-6);

    let mut local = params_to_local(&Bloch2dResidueParams {
        delta_a_h_ppm: ha,
        delta_b_h_ppm: hb,
        delta_a_n_ppm: na,
        delta_b_n_ppm: nb,
        r2a_h_s_inv: r2a_h,
        r2b_h_s_inv: r2b_h,
        r2a_n_s_inv: r2a_n,
        r2b_n_s_inv: r2b_n,
        amplitude: amp,
    });
    clamp_local(&mut local, bounds);
    local_to_params(&local)
}

/// Compute VarPro-optimal linear amplitude `A_r > 0` and weighted chi^2 for one residue.
fn varpro_amplitude_and_chi2(
    res: &ResidueTitration2dSeries,
    params_unit_amp: &Bloch2dResidueParams,
    kd_um: f64,
    k_off_s_inv: f64,
) -> Result<(f64, f64), Titration2dError> {
    let mut mm = 0.0_f64;
    let mut my = 0.0_f64;
    let mut yy = 0.0_f64;
    let mut buf = Vec::new();
    for patch in &res.patches {
        let eq = binding_equilibrium_rates(
            patch.protein_conc_um,
            patch.ligand_conc_um,
            kd_um,
            k_off_s_inv,
        )?;
        let st = Bloch2dKineticState::from(eq);
        let n_pix = patch.intensities.len();
        buf.resize(n_pix, 0.0);
        eval_2d_lineshape_into(
            &patch.grid,
            params_unit_amp,
            &st,
            patch.spectrum_scale,
            &mut buf,
        );
        let w = 1.0 / (patch.noise_sigma * patch.noise_sigma);
        for (&m, &y) in buf.iter().zip(patch.intensities.iter()) {
            mm += w * m * m;
            my += w * m * y;
            yy += w * y * y;
        }
    }
    let amp = if mm > 1e-30 { (my / mm).max(1e-9) } else { 1.0 };
    let chi2 = (yy - 2.0 * amp * my + amp * amp * mm).max(0.0);
    Ok((amp, chi2))
}

fn eval_total_chi2(
    residues: &[ResidueTitration2dSeries],
    ln_kd: f64,
    ln_koff: f64,
    locals: &[[f64; N_LOCAL]],
) -> Result<(f64, Vec<f64>), Titration2dError> {
    let kd = ln_kd.exp();
    let koff = ln_koff.exp();
    let mut total = 0.0_f64;
    let mut per_res = Vec::with_capacity(residues.len());
    let mut buf = Vec::new();
    for (r, res) in residues.iter().enumerate() {
        let params = local_to_params(&locals[r]);
        let mut c2 = 0.0_f64;
        for patch in &res.patches {
            let eq =
                binding_equilibrium_rates(patch.protein_conc_um, patch.ligand_conc_um, kd, koff)?;
            let st = Bloch2dKineticState::from(eq);
            buf.resize(patch.intensities.len(), 0.0);
            eval_2d_lineshape_into(&patch.grid, &params, &st, patch.spectrum_scale, &mut buf);
            let inv_s2 = 1.0 / (patch.noise_sigma * patch.noise_sigma);
            for (&m, &y) in buf.iter().zip(patch.intensities.iter()) {
                let d = y - m;
                c2 += d * d * inv_s2;
            }
        }
        per_res.push(c2);
        total += c2;
    }
    Ok((total, per_res))
}

/// Invert a small dense `N x N` matrix via Gauss–Jordan elimination with partial pivoting.
fn invert_small_matrix<const N: usize>(mat: &[[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let mut a = *mat;
    let mut inv = [[0.0_f64; N]; N];
    for i in 0..N {
        inv[i][i] = 1.0;
    }

    for col in 0..N {
        let mut piv_row = col;
        let mut piv_abs = a[col][col].abs();
        for r in (col + 1)..N {
            let v = a[r][col].abs();
            if v > piv_abs {
                piv_abs = v;
                piv_row = r;
            }
        }
        if !piv_abs.is_finite() || piv_abs < 1e-18 {
            return None;
        }
        if piv_row != col {
            a.swap(col, piv_row);
            inv.swap(col, piv_row);
        }
        let diag = a[col][col];
        for j in 0..N {
            a[col][j] /= diag;
            inv[col][j] /= diag;
        }
        for r in 0..N {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor != 0.0 {
                for j in 0..N {
                    a[r][j] -= factor * a[col][j];
                    inv[r][j] -= factor * inv[col][j];
                }
            }
        }
    }
    Some(inv)
}

#[derive(Clone, Debug)]
struct LmCandidate {
    ln_kd: f64,
    ln_koff: f64,
    locals: Vec<[f64; N_LOCAL]>,
    chi2: f64,
    per_res_chi2: Vec<f64>,
    iterations: usize,
    converged: bool,
}

/// Accumulate block-arrow normal equations `(H_GG, H_Gr, H_rr, g_G, g_r)` using central finite
/// differences for nonlinear parameters and exact analytic derivative for `ln(A_r)`.
#[allow(clippy::type_complexity)]
fn assemble_block_arrow_normal_equations(
    residues: &[ResidueTitration2dSeries],
    ln_kd: f64,
    ln_koff: f64,
    locals: &[[f64; N_LOCAL]],
) -> Result<
    (
        [[f64; 2]; 2],
        Vec<[[f64; N_LOCAL]; 2]>,
        Vec<[[f64; N_LOCAL]; N_LOCAL]>,
        [f64; 2],
        Vec<[f64; N_LOCAL]>,
        f64,
    ),
    Titration2dError,
> {
    let n_res = residues.len();
    let mut h_gg = [[0.0_f64; 2]; 2];
    let mut h_gr = vec![[[0.0_f64; N_LOCAL]; 2]; n_res];
    let mut h_rr = vec![[[0.0_f64; N_LOCAL]; N_LOCAL]; n_res];
    let mut g_g = [0.0_f64; 2];
    let mut g_r = vec![[0.0_f64; N_LOCAL]; n_res];
    let mut total_chi2 = 0.0_f64;

    let kd0 = ln_kd.exp();
    let koff0 = ln_koff.exp();

    let h_glob = [1e-5_f64, 1e-5_f64];
    let h_loc = [
        1e-5_f64, // delta_a_h
        1e-5_f64, // delta_b_h
        5e-5_f64, // delta_a_n
        5e-5_f64, // delta_b_n
        1e-5_f64, // ln_r2a_h
        1e-5_f64, // ln_r2b_h
        1e-5_f64, // ln_r2a_n
        1e-5_f64, // ln_r2b_n
    ];

    let mut base = Vec::new();
    let mut plus = Vec::new();
    let mut minus = Vec::new();
    let mut j_glob: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut j_loc: [Vec<f64>; N_LOCAL] = std::array::from_fn(|_| Vec::new());

    for (r, res) in residues.iter().enumerate() {
        let loc0 = locals[r];
        let params0 = local_to_params(&loc0);

        for patch in &res.patches {
            let n_pix = patch.intensities.len();
            base.resize(n_pix, 0.0);
            plus.resize(n_pix, 0.0);
            minus.resize(n_pix, 0.0);
            for v in &mut j_glob {
                v.resize(n_pix, 0.0);
            }
            for v in &mut j_loc {
                v.resize(n_pix, 0.0);
            }

            let eq0 =
                binding_equilibrium_rates(patch.protein_conc_um, patch.ligand_conc_um, kd0, koff0)?;
            let st0 = Bloch2dKineticState::from(eq0);
            eval_2d_lineshape_into(&patch.grid, &params0, &st0, patch.spectrum_scale, &mut base);

            // Global derivatives: 0 = ln(Kd), 1 = ln(k_off)
            for g_idx in 0..2 {
                let h = h_glob[g_idx];
                let (kd_p, koff_p) = if g_idx == 0 {
                    ((ln_kd + h).exp(), koff0)
                } else {
                    (kd0, (ln_koff + h).exp())
                };
                let (kd_m, koff_m) = if g_idx == 0 {
                    ((ln_kd - h).exp(), koff0)
                } else {
                    (kd0, (ln_koff - h).exp())
                };
                let st_p = Bloch2dKineticState::from(binding_equilibrium_rates(
                    patch.protein_conc_um,
                    patch.ligand_conc_um,
                    kd_p,
                    koff_p,
                )?);
                let st_m = Bloch2dKineticState::from(binding_equilibrium_rates(
                    patch.protein_conc_um,
                    patch.ligand_conc_um,
                    kd_m,
                    koff_m,
                )?);
                eval_2d_lineshape_into(
                    &patch.grid,
                    &params0,
                    &st_p,
                    patch.spectrum_scale,
                    &mut plus,
                );
                eval_2d_lineshape_into(
                    &patch.grid,
                    &params0,
                    &st_m,
                    patch.spectrum_scale,
                    &mut minus,
                );
                let inv_2h = 0.5 / h;
                for k in 0..n_pix {
                    j_glob[g_idx][k] = (plus[k] - minus[k]) * inv_2h;
                }
            }

            // Local nonlinear derivatives 0..8
            for l_idx in 0..8 {
                let h = h_loc[l_idx];
                let mut lp = loc0;
                let mut lm = loc0;
                lp[l_idx] += h;
                lm[l_idx] -= h;
                let pp = local_to_params(&lp);
                let pm = local_to_params(&lm);
                eval_2d_lineshape_into(&patch.grid, &pp, &st0, patch.spectrum_scale, &mut plus);
                eval_2d_lineshape_into(&patch.grid, &pm, &st0, patch.spectrum_scale, &mut minus);
                let inv_2h = 0.5 / h;
                for k in 0..n_pix {
                    j_loc[l_idx][k] = (plus[k] - minus[k]) * inv_2h;
                }
            }
            // Local derivative 8: dI / d(ln A_r) = I (exact!)
            j_loc[8].copy_from_slice(&base);

            // Accumulate weighted normal equations.
            let inv_s2 = 1.0 / (patch.noise_sigma * patch.noise_sigma);
            for k in 0..n_pix {
                let resid = patch.intensities[k] - base[k];
                total_chi2 += resid * resid * inv_s2;

                let jg0 = j_glob[0][k];
                let jg1 = j_glob[1][k];
                let w_jg0 = inv_s2 * jg0;
                let w_jg1 = inv_s2 * jg1;

                g_g[0] += w_jg0 * resid;
                g_g[1] += w_jg1 * resid;
                h_gg[0][0] += w_jg0 * jg0;
                h_gg[0][1] += w_jg0 * jg1;
                h_gg[1][1] += w_jg1 * jg1;

                let mut jl = [0.0_f64; N_LOCAL];
                for a in 0..N_LOCAL {
                    jl[a] = j_loc[a][k];
                }
                for a in 0..N_LOCAL {
                    let w_jla = inv_s2 * jl[a];
                    g_r[r][a] += w_jla * resid;
                    h_gr[r][0][a] += w_jg0 * jl[a];
                    h_gr[r][1][a] += w_jg1 * jl[a];
                    for b in a..N_LOCAL {
                        h_rr[r][a][b] += w_jla * jl[b];
                    }
                }
            }
        }
    }

    h_gg[1][0] = h_gg[0][1];
    for r in 0..n_res {
        for a in 0..N_LOCAL {
            for b in (a + 1)..N_LOCAL {
                h_rr[r][b][a] = h_rr[r][a][b];
            }
        }
    }

    Ok((h_gg, h_gr, h_rr, g_g, g_r, total_chi2))
}

/// Solve the damped block-arrow Schur-complement system for `(delta_global, delta_locals)`.
fn solve_schur_step(
    h_gg: &[[f64; 2]; 2],
    h_gr: &[[[f64; N_LOCAL]; 2]],
    h_rr: &[[[f64; N_LOCAL]; N_LOCAL]],
    g_g: &[f64; 2],
    g_r: &[[f64; N_LOCAL]],
    lambda: f64,
) -> Option<([f64; 2], Vec<[f64; N_LOCAL]>)> {
    let n_res = h_rr.len();
    let mut s_mat = *h_gg;
    s_mat[0][0] += lambda * h_gg[0][0].max(1.0);
    s_mat[1][1] += lambda * h_gg[1][1].max(1.0);
    let mut s_rhs = *g_g;

    let mut inv_h_rr = Vec::with_capacity(n_res);
    let mut inv_g_r = Vec::with_capacity(n_res);

    for r in 0..n_res {
        let mut blk = h_rr[r];
        for a in 0..N_LOCAL {
            blk[a][a] += lambda * h_rr[r][a][a].max(1.0);
        }
        let blk_inv = invert_small_matrix(&blk)?;

        // y_r = blk_inv * g_r[r]
        let mut y_r = [0.0_f64; N_LOCAL];
        for a in 0..N_LOCAL {
            let mut sum = 0.0;
            for b in 0..N_LOCAL {
                sum += blk_inv[a][b] * g_r[r][b];
            }
            y_r[a] = sum;
        }

        // Update Schur RHS: s_rhs -= H_Gr * y_r
        for gi in 0..2 {
            let mut dot = 0.0;
            for a in 0..N_LOCAL {
                dot += h_gr[r][gi][a] * y_r[a];
            }
            s_rhs[gi] -= dot;
        }

        // Compute tmp = blk_inv * H_rG (N_LOCAL x 2) and update S -= H_Gr * tmp
        let mut tmp = [[0.0_f64; 2]; N_LOCAL];
        for a in 0..N_LOCAL {
            for gi in 0..2 {
                let mut sum = 0.0;
                for b in 0..N_LOCAL {
                    sum += blk_inv[a][b] * h_gr[r][gi][b];
                }
                tmp[a][gi] = sum;
            }
        }
        for gi in 0..2 {
            for gj in 0..2 {
                let mut dot = 0.0;
                for a in 0..N_LOCAL {
                    dot += h_gr[r][gi][a] * tmp[a][gj];
                }
                s_mat[gi][gj] -= dot;
            }
        }

        inv_h_rr.push(blk_inv);
        inv_g_r.push(y_r);
    }

    let s_inv = invert_small_matrix(&s_mat)?;
    let dg = [
        s_inv[0][0] * s_rhs[0] + s_inv[0][1] * s_rhs[1],
        s_inv[1][0] * s_rhs[0] + s_inv[1][1] * s_rhs[1],
    ];
    if !dg[0].is_finite() || !dg[1].is_finite() {
        return None;
    }

    let mut d_locals = Vec::with_capacity(n_res);
    for r in 0..n_res {
        // dr = inv_h_rr[r] * (g_r[r] - H_rG * dg) = inv_g_r[r] - inv_h_rr[r] * (H_rG * dg)
        let mut hrg_dg = [0.0_f64; N_LOCAL];
        for a in 0..N_LOCAL {
            hrg_dg[a] = h_gr[r][0][a] * dg[0] + h_gr[r][1][a] * dg[1];
        }
        let mut dr = [0.0_f64; N_LOCAL];
        for a in 0..N_LOCAL {
            let mut sub = 0.0;
            for b in 0..N_LOCAL {
                sub += inv_h_rr[r][a][b] * hrg_dg[b];
            }
            let val = inv_g_r[r][a] - sub;
            if !val.is_finite() {
                return None;
            }
            dr[a] = val;
        }
        d_locals.push(dr);
    }

    Some((dg, d_locals))
}

/// Compute exact Cramér–Rao covariance blocks `(Sigma_GG, Sigma_rr_vec)` from the un-damped
/// block-arrow Fisher information matrix (with tiny safeguard ridge only if near-singular).
fn compute_cramer_rao_covariance(
    h_gg: &[[f64; 2]; 2],
    h_gr: &[[[f64; N_LOCAL]; 2]],
    h_rr: &[[[f64; N_LOCAL]; N_LOCAL]],
) -> ([[f64; 2]; 2], Vec<[[f64; N_LOCAL]; N_LOCAL]>) {
    let n_res = h_rr.len();
    for ridge in [0.0_f64, 1e-12, 1e-9, 1e-6, 1e-3] {
        let mut s_mat = *h_gg;
        if ridge > 0.0 {
            s_mat[0][0] += ridge * h_gg[0][0].max(1.0);
            s_mat[1][1] += ridge * h_gg[1][1].max(1.0);
        }
        let mut inv_rr_list = Vec::with_capacity(n_res);
        let mut tmp_list = Vec::with_capacity(n_res);
        let mut ok = true;

        for r in 0..n_res {
            let mut blk = h_rr[r];
            if ridge > 0.0 {
                for a in 0..N_LOCAL {
                    blk[a][a] += ridge * h_rr[r][a][a].max(1.0);
                }
            }
            let Some(blk_inv) = invert_small_matrix(&blk) else {
                ok = false;
                break;
            };
            let mut tmp = [[0.0_f64; 2]; N_LOCAL];
            for a in 0..N_LOCAL {
                for gi in 0..2 {
                    let mut sum = 0.0;
                    for b in 0..N_LOCAL {
                        sum += blk_inv[a][b] * h_gr[r][gi][b];
                    }
                    tmp[a][gi] = sum;
                }
            }
            for gi in 0..2 {
                for gj in 0..2 {
                    let mut dot = 0.0;
                    for a in 0..N_LOCAL {
                        dot += h_gr[r][gi][a] * tmp[a][gj];
                    }
                    s_mat[gi][gj] -= dot;
                }
            }
            inv_rr_list.push(blk_inv);
            tmp_list.push(tmp);
        }

        if !ok {
            continue;
        }
        if let Some(sigma_gg) = invert_small_matrix(&s_mat) {
            if sigma_gg[0][0] >= 0.0 && sigma_gg[1][1] >= 0.0 {
                let mut sigma_rr_list = Vec::with_capacity(n_res);
                for r in 0..n_res {
                    let mut cov = inv_rr_list[r];
                    // cov += tmp * sigma_gg * tmp^T
                    for a in 0..N_LOCAL {
                        let t0 =
                            tmp_list[r][a][0] * sigma_gg[0][0] + tmp_list[r][a][1] * sigma_gg[1][0];
                        let t1 =
                            tmp_list[r][a][0] * sigma_gg[0][1] + tmp_list[r][a][1] * sigma_gg[1][1];
                        for b in 0..N_LOCAL {
                            cov[a][b] += t0 * tmp_list[r][b][0] + t1 * tmp_list[r][b][1];
                        }
                    }
                    sigma_rr_list.push(cov);
                }
                return (sigma_gg, sigma_rr_list);
            }
        }
    }

    ([[0.0; 2]; 2], vec![[[0.0; N_LOCAL]; N_LOCAL]; n_res])
}

fn run_schur_lm(
    residues: &[ResidueTitration2dSeries],
    bounds: &[LocalBounds],
    config: &Titration2dFitConfig,
    init_ln_kd: f64,
    init_ln_koff: f64,
    init_locals: Vec<[f64; N_LOCAL]>,
) -> Result<LmCandidate, Titration2dError> {
    let ln_kd_lo = config.kd_bounds_um.0.ln();
    let ln_kd_hi = config.kd_bounds_um.1.ln();
    let ln_koff_lo = config.k_off_bounds_s_inv.0.ln();
    let ln_koff_hi = config.k_off_bounds_s_inv.1.ln();

    let mut ln_kd = init_ln_kd.clamp(ln_kd_lo, ln_kd_hi);
    let mut ln_koff = init_ln_koff.clamp(ln_koff_lo, ln_koff_hi);
    let mut locals = init_locals;
    for (r, loc) in locals.iter_mut().enumerate() {
        clamp_local(loc, &bounds[r]);
    }

    let (mut chi2, mut per_res_chi2) = eval_total_chi2(residues, ln_kd, ln_koff, &locals)?;
    let mut lambda = 1e-2_f64;
    let mut iterations = 0usize;
    let mut converged = false;

    for iter in 1..=config.max_lm_iterations {
        iterations = iter;
        let (h_gg, h_gr, h_rr, g_g, g_r, _) =
            assemble_block_arrow_normal_equations(residues, ln_kd, ln_koff, &locals)?;

        let mut accepted = false;
        let mut trial_lambda = lambda;
        for _ in 0..14 {
            if let Some((dg, d_locs)) =
                solve_schur_step(&h_gg, &h_gr, &h_rr, &g_g, &g_r, trial_lambda)
            {
                let cand_ln_kd = (ln_kd + dg[0]).clamp(ln_kd_lo, ln_kd_hi);
                let cand_ln_koff = (ln_koff + dg[1]).clamp(ln_koff_lo, ln_koff_hi);
                let mut cand_locals = locals.clone();
                let mut max_rel_step = ((cand_ln_kd - ln_kd).abs() / (1.0 + ln_kd.abs()))
                    .max((cand_ln_koff - ln_koff).abs() / (1.0 + ln_koff.abs()));

                for (r, cloc) in cand_locals.iter_mut().enumerate() {
                    for a in 0..N_LOCAL {
                        cloc[a] += d_locs[r][a];
                    }
                    clamp_local(cloc, &bounds[r]);
                    for a in 0..N_LOCAL {
                        let rel = (cloc[a] - locals[r][a]).abs() / (1.0 + locals[r][a].abs());
                        if rel > max_rel_step {
                            max_rel_step = rel;
                        }
                    }
                }

                let (cand_chi2, cand_per_res) =
                    eval_total_chi2(residues, cand_ln_kd, cand_ln_koff, &cand_locals)?;
                if cand_chi2.is_finite() && cand_chi2 <= chi2 {
                    let rel_chi2_imp = (chi2 - cand_chi2) / (chi2 + 1e-15);
                    ln_kd = cand_ln_kd;
                    ln_koff = cand_ln_koff;
                    locals = cand_locals;
                    chi2 = cand_chi2;
                    per_res_chi2 = cand_per_res;
                    lambda = (trial_lambda / 3.0).max(1e-10);
                    accepted = true;
                    if rel_chi2_imp < config.rel_chi2_tol && max_rel_step < config.rel_param_tol {
                        converged = true;
                    }
                    break;
                }
            }
            trial_lambda *= 5.0;
            if trial_lambda > 1e12 {
                break;
            }
        }

        if converged || !accepted {
            if !accepted {
                // No descent step found even at maximal damping -> stationary point reached.
                converged = true;
            }
            break;
        }
    }

    Ok(LmCandidate {
        ln_kd,
        ln_koff,
        locals,
        chi2,
        per_res_chi2,
        iterations,
        converged,
    })
}

fn validate_fit_inputs(
    residues: &[ResidueTitration2dSeries],
    config: &Titration2dFitConfig,
) -> Result<usize, Titration2dError> {
    if residues.is_empty() {
        return Err(Titration2dError::EmptyInput(
            "fit_titration_2d_lineshape requires at least one residue series".to_string(),
        ));
    }
    validate_positive("kd_bounds_um.0", config.kd_bounds_um.0)?;
    validate_positive("kd_bounds_um.1", config.kd_bounds_um.1)?;
    if config.kd_bounds_um.0 >= config.kd_bounds_um.1 {
        return Err(Titration2dError::InvalidParameter {
            name: "kd_bounds_um".to_string(),
            value: config.kd_bounds_um.0,
            reason: "lower bound must be < upper bound".to_string(),
        });
    }
    validate_positive("k_off_bounds_s_inv.0", config.k_off_bounds_s_inv.0)?;
    validate_positive("k_off_bounds_s_inv.1", config.k_off_bounds_s_inv.1)?;
    if config.k_off_bounds_s_inv.0 >= config.k_off_bounds_s_inv.1 {
        return Err(Titration2dError::InvalidParameter {
            name: "k_off_bounds_s_inv".to_string(),
            value: config.k_off_bounds_s_inv.0,
            reason: "lower bound must be < upper bound".to_string(),
        });
    }
    validate_positive("r2_bounds_s_inv.0", config.r2_bounds_s_inv.0)?;
    validate_positive("r2_bounds_s_inv.1", config.r2_bounds_s_inv.1)?;
    if config.r2_bounds_s_inv.0 >= config.r2_bounds_s_inv.1 {
        return Err(Titration2dError::InvalidParameter {
            name: "r2_bounds_s_inv".to_string(),
            value: config.r2_bounds_s_inv.0,
            reason: "lower bound must be < upper bound".to_string(),
        });
    }
    if let Some(kd0) = config.initial_kd_um {
        validate_positive("initial_kd_um", kd0)?;
    }
    if let Some(koff0) = config.initial_k_off_s_inv {
        validate_positive("initial_k_off_s_inv", koff0)?;
    }
    validate_positive("csp_sigma_ppm", config.csp_sigma_ppm)?;
    if config.max_lm_iterations == 0 {
        return Err(Titration2dError::InvalidParameter {
            name: "max_lm_iterations".to_string(),
            value: 0.0,
            reason: "must be >= 1".to_string(),
        });
    }

    let mut total_pixels = 0usize;
    for res in residues {
        if res.patches.len() < 3 {
            return Err(Titration2dError::EmptyInput(format!(
                "residue '{}' requires at least 3 titration ROI patches (got {})",
                res.label,
                res.patches.len()
            )));
        }
        if let Some(ref g) = res.initial_guess {
            validate_bloch_params(g)?;
        }
        let mut min_l = f64::INFINITY;
        let mut max_l = f64::NEG_INFINITY;
        for patch in &res.patches {
            patch.grid.validate()?;
            validate_positive("protein_conc_um", patch.protein_conc_um)?;
            validate_non_negative("ligand_conc_um", patch.ligand_conc_um)?;
            validate_positive("noise_sigma", patch.noise_sigma)?;
            validate_positive("spectrum_scale", patch.spectrum_scale)?;
            let expected = patch.grid.n_ppm_axis.len() * patch.grid.h_ppm_axis.len();
            if patch.intensities.len() != expected {
                return Err(Titration2dError::DimensionMismatch {
                    context: format!(
                        "residue '{}' step {} ROI intensities",
                        res.label, patch.step_index
                    ),
                    expected,
                    actual: patch.intensities.len(),
                });
            }
            for &y in &patch.intensities {
                validate_finite("patch.intensities", y)?;
            }
            min_l = min_l.min(patch.ligand_conc_um);
            max_l = max_l.max(patch.ligand_conc_um);
            total_pixels += expected;
        }
        if (max_l - min_l).abs() <= 1e-12 {
            return Err(Titration2dError::InvalidParameter {
                name: "ligand_conc_um".to_string(),
                value: max_l,
                reason: "titration series must span distinct ligand concentrations".to_string(),
            });
        }
    }

    let n_params = 2 + N_LOCAL * residues.len();
    if total_pixels <= n_params {
        return Err(Titration2dError::DimensionMismatch {
            context: "total 2D ROI pixels must exceed fitted parameter count".to_string(),
            expected: n_params + 1,
            actual: total_pixels,
        });
    }
    Ok(total_pixels)
}

/// Global multi-residue coupled 2D Bloch–McConnell titration lineshape fitter using a coarse
/// `(K_d, k_off)` Variable-Projection grid search followed by block-arrow Schur-complement
/// Levenberg–Marquardt optimization, Cramér–Rao covariance analysis, per-step exchange-regime
/// verification, and fast-exchange quadratic CSP isotherm comparison.
pub fn fit_titration_2d_lineshape(
    residues: &[ResidueTitration2dSeries],
    config: &Titration2dFitConfig,
) -> Result<Titration2dFitResult, Titration2dError> {
    let n_pixels_total = validate_fit_inputs(residues, config)?;
    let n_res = residues.len();
    let n_params = 2 + N_LOCAL * n_res;
    let bounds: Vec<LocalBounds> = residues
        .iter()
        .map(|r| compute_local_bounds(r, config))
        .collect();

    // Determine ligand concentration scale for the coarse K_d grid.
    let mut min_pos_l = f64::INFINITY;
    let mut max_l = 0.0_f64;
    for res in residues {
        for p in &res.patches {
            if p.ligand_conc_um > 0.0 {
                min_pos_l = min_pos_l.min(p.ligand_conc_um);
            }
            max_l = max_l.max(p.ligand_conc_um);
        }
    }
    let kd_grid_lo = (min_pos_l * 0.05)
        .max(config.kd_bounds_um.0)
        .min(config.kd_bounds_um.1 * 0.5);
    let kd_grid_hi = (max_l * 3.0)
        .min(config.kd_bounds_um.1)
        .max(kd_grid_lo * 2.0);
    let koff_grid_lo = 5.0_f64.max(config.k_off_bounds_s_inv.0);
    let koff_grid_hi = 5000.0_f64
        .min(config.k_off_bounds_s_inv.1)
        .max(koff_grid_lo * 2.0);

    let n_kd_grid = 7usize;
    let n_koff_grid = 7usize;
    let mut coarse_scores: Vec<(f64, f64, f64, Vec<[f64; N_LOCAL]>)> = Vec::new();

    for i_kd in 0..n_kd_grid {
        let frac_kd = (i_kd as f64) / ((n_kd_grid - 1) as f64);
        let kd = (kd_grid_lo.ln() + frac_kd * (kd_grid_hi.ln() - kd_grid_lo.ln())).exp();

        let base_inits: Vec<Bloch2dResidueParams> = residues
            .iter()
            .enumerate()
            .map(|(r, res)| {
                let mut p = auto_init_residue_params(res, kd, &bounds[r]);
                p.amplitude = 1.0;
                p
            })
            .collect();

        for i_koff in 0..n_koff_grid {
            let frac_k = (i_koff as f64) / ((n_koff_grid - 1) as f64);
            let koff = (koff_grid_lo.ln() + frac_k * (koff_grid_hi.ln() - koff_grid_lo.ln())).exp();

            let mut sum_c2 = 0.0_f64;
            let mut locals = Vec::with_capacity(n_res);
            for (r, res) in residues.iter().enumerate() {
                let (amp, c2) = varpro_amplitude_and_chi2(res, &base_inits[r], kd, koff)?;
                sum_c2 += c2;
                let mut p = base_inits[r].clone();
                p.amplitude = amp;
                let mut loc = params_to_local(&p);
                clamp_local(&mut loc, &bounds[r]);
                locals.push(loc);
            }
            coarse_scores.push((sum_c2, kd, koff, locals));
        }
    }

    // Include explicit user initial guess as an additional candidate start when provided.
    if config.initial_kd_um.is_some() || config.initial_k_off_s_inv.is_some() {
        let kd = config
            .initial_kd_um
            .unwrap_or_else(|| (0.5 * (kd_grid_lo.ln() + kd_grid_hi.ln())).exp())
            .clamp(config.kd_bounds_um.0, config.kd_bounds_um.1);
        let koff = config
            .initial_k_off_s_inv
            .unwrap_or(150.0)
            .clamp(config.k_off_bounds_s_inv.0, config.k_off_bounds_s_inv.1);
        let mut sum_c2 = 0.0;
        let mut locals = Vec::with_capacity(n_res);
        for (r, res) in residues.iter().enumerate() {
            let mut p = auto_init_residue_params(res, kd, &bounds[r]);
            p.amplitude = 1.0;
            let (amp, c2) = varpro_amplitude_and_chi2(res, &p, kd, koff)?;
            sum_c2 += c2;
            p.amplitude = amp;
            let mut loc = params_to_local(&p);
            clamp_local(&mut loc, &bounds[r]);
            locals.push(loc);
        }
        coarse_scores.push((sum_c2, kd, koff, locals));
    }

    coarse_scores.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Select top distinct starting basins for full Schur-complement LM polishing.
    let n_starts = config.n_coarse_starts.max(1);
    let mut selected_starts: Vec<(f64, f64, Vec<[f64; N_LOCAL]>)> = Vec::new();
    for &(_, kd, koff, ref locs) in &coarse_scores {
        let far_enough = selected_starts.iter().all(|&(skd, skoff, _)| {
            let d_kd = (kd.ln() - skd.ln()).abs();
            let d_koff = (koff.ln() - skoff.ln()).abs();
            d_kd.hypot(d_koff) > 0.65
        });
        if selected_starts.is_empty() || far_enough {
            selected_starts.push((kd, koff, locs.clone()));
            if selected_starts.len() >= n_starts {
                break;
            }
        }
    }
    if selected_starts.is_empty() {
        if let Some((_, kd, koff, locs)) = coarse_scores.into_iter().next() {
            selected_starts.push((kd, koff, locs));
        } else {
            return Err(Titration2dError::NumericalFailure(
                "coarse grid initialization produced no valid start".to_string(),
            ));
        }
    }

    let mut best: Option<LmCandidate> = None;
    for (kd0, koff0, locs0) in selected_starts {
        let cand = run_schur_lm(residues, &bounds, config, kd0.ln(), koff0.ln(), locs0)?;
        let take = match &best {
            None => true,
            Some(prev) => cand.chi2 < prev.chi2,
        };
        if take {
            best = Some(cand);
        }
    }

    let best = best.ok_or_else(|| {
        Titration2dError::NumericalFailure("LM optimizer failed to produce a candidate".to_string())
    })?;

    // Compute exact Cramér–Rao covariance at the optimum.
    let (h_gg, h_gr, h_rr, _, _, _) =
        assemble_block_arrow_normal_equations(residues, best.ln_kd, best.ln_koff, &best.locals)?;
    let (sigma_gg, sigma_rr_list) = compute_cramer_rao_covariance(&h_gg, &h_gr, &h_rr);

    let kd_um = best.ln_kd.exp();
    let k_off_s_inv = best.ln_koff.exp();
    let k_on_um_inv_s_inv = k_off_s_inv / kd_um;

    let dof = n_pixels_total.saturating_sub(n_params);
    let reduced_chi2 = if dof > 0 {
        best.chi2 / (dof as f64)
    } else {
        best.chi2
    };

    let kd_sd_um = kd_um * sigma_gg[0][0].max(0.0).sqrt();
    let k_off_sd_s_inv = k_off_s_inv * sigma_gg[1][1].max(0.0).sqrt();
    let var_ln_kon = (sigma_gg[1][1] + sigma_gg[0][0] - 2.0 * sigma_gg[0][1]).max(0.0);
    let k_on_sd_um_inv_s_inv = k_on_um_inv_s_inv * var_ln_kon.sqrt();

    // Build per-residue fitted reports, exchange-regime audits, and CSP comparison curves.
    let mut fitted_residues = Vec::with_capacity(n_res);
    let mut csp_curves: Vec<Vec<(f64, f64, f64)>> = Vec::with_capacity(n_res);
    let mut csp_labels: Vec<String> = Vec::with_capacity(n_res);
    let mut all_warnings: Vec<ExchangeRegimeWarning> = Vec::new();
    let mut all_fast_valid = true;

    for (r, res) in residues.iter().enumerate() {
        let params = local_to_params(&best.locals[r]);
        let cov = &sigma_rr_list[r];

        let uncertainties = Bloch2dResidueParams {
            delta_a_h_ppm: cov[0][0].max(0.0).sqrt(),
            delta_b_h_ppm: cov[1][1].max(0.0).sqrt(),
            delta_a_n_ppm: cov[2][2].max(0.0).sqrt(),
            delta_b_n_ppm: cov[3][3].max(0.0).sqrt(),
            r2a_h_s_inv: params.r2a_h_s_inv * cov[4][4].max(0.0).sqrt(),
            r2b_h_s_inv: params.r2b_h_s_inv * cov[5][5].max(0.0).sqrt(),
            r2a_n_s_inv: params.r2a_n_s_inv * cov[6][6].max(0.0).sqrt(),
            r2b_n_s_inv: params.r2b_n_s_inv * cov[7][7].max(0.0).sqrt(),
            amplitude: params.amplitude * cov[8][8].max(0.0).sqrt(),
        };

        let dw_h_ppm = params.delta_b_h_ppm - params.delta_a_h_ppm;
        let dw_n_ppm = params.delta_b_n_ppm - params.delta_a_n_ppm;
        let dw_h_sd_ppm = (cov[0][0] + cov[1][1] - 2.0 * cov[0][1]).max(0.0).sqrt();
        let dw_n_sd_ppm = (cov[2][2] + cov[3][3] - 2.0 * cov[2][3]).max(0.0).sqrt();

        let h_mhz = res.patches[0].grid.h_larmor_mhz;
        let n_mhz = res.patches[0].grid.n_larmor_mhz;
        let dw_h_rad_s = TAU * dw_h_ppm * h_mhz;
        let dw_n_rad_s = TAU * dw_n_ppm * n_mhz;

        let conc_pts: Vec<TitrationConcentrationPoint> = res
            .patches
            .iter()
            .map(|p| TitrationConcentrationPoint {
                step_index: p.step_index,
                label: format!("step_{}", p.step_index),
                protein_conc_um: p.protein_conc_um,
                ligand_conc_um: p.ligand_conc_um,
            })
            .collect();

        let regime_report = classify_exchange_regime(
            &ExchangeRegimeInput {
                label: res.label.clone(),
                kd_um,
                k_off_s_inv,
                dw_h_ppm,
                dw_n_ppm,
                h_larmor_mhz: h_mhz,
                n_larmor_mhz: Some(n_mhz),
            },
            &conc_pts,
        )?;
        if !regime_report.fast_exchange_csp_valid {
            all_fast_valid = false;
            all_warnings.extend(regime_report.warnings.clone());
        }

        // Extract apparent peak-maximum CSP curve for fast-exchange comparison.
        let mut apo_idx = 0usize;
        for (idx, p) in res.patches.iter().enumerate() {
            if p.ligand_conc_um < res.patches[apo_idx].ligand_conc_um {
                apo_idx = idx;
            }
        }
        let (h0, n0, _, _, _) = find_patch_peak_subpixel(&res.patches[apo_idx]);
        let alpha_n = amide_alpha_n(&res.res_name);
        let mut curve = Vec::new();
        for p in &res.patches {
            let (h_obs, n_obs, pk_max, _, _) = find_patch_peak_subpixel(p);
            if pk_max > 3.0 * p.noise_sigma {
                let csp = (h_obs - h0).hypot(alpha_n * (n_obs - n0));
                curve.push((p.protein_conc_um, p.ligand_conc_um, csp));
            }
        }
        if curve.len() >= 2 {
            csp_curves.push(curve);
            csp_labels.push(res.label.clone());
        }

        let n_pix: usize = res.patches.iter().map(|p| p.intensities.len()).sum();
        fitted_residues.push(FittedResidue2d {
            label: res.label.clone(),
            res_name: res.res_name.clone(),
            params,
            uncertainties,
            dw_h_ppm,
            dw_h_sd_ppm,
            dw_n_ppm,
            dw_n_sd_ppm,
            dw_h_rad_s,
            dw_n_rad_s,
            chi2: best.per_res_chi2[r],
            n_pixels: n_pix,
            exchange_regime: regime_report,
        });
    }

    let (csp_kd_um, csp_dmax_vec, csp_chi2) = if !csp_curves.is_empty() {
        let refs: Vec<&[(f64, f64, f64)]> = csp_curves.iter().map(Vec::as_slice).collect();
        fit_quadratic_tight_binding_varpro(&refs, config.csp_sigma_ppm)
    } else {
        (kd_um, Vec::new(), 0.0)
    };
    let kd_relative_discrepancy = (csp_kd_um - kd_um).abs() / kd_um.max(1e-12);
    let per_residue_ddelta_max_ppm: Vec<(String, f64)> =
        csp_labels.into_iter().zip(csp_dmax_vec).collect();

    Ok(Titration2dFitResult {
        kd_um,
        kd_sd_um,
        k_off_s_inv,
        k_off_sd_s_inv,
        k_on_um_inv_s_inv,
        k_on_sd_um_inv_s_inv,
        chi2: best.chi2,
        reduced_chi2,
        dof,
        n_pixels_total,
        n_params,
        lm_iterations: best.iterations,
        converged: best.converged,
        residues: fitted_residues,
        fast_exchange_comparison: FastExchangeCspComparison {
            csp_kd_um,
            csp_chi2,
            kd_relative_discrepancy,
            per_residue_ddelta_max_ppm,
            fast_exchange_assumption_valid: all_fast_valid,
            warnings: all_warnings,
        },
    })
}

/// Utility helper returning the exact 2D Lorentzian integral `pi^2 * I_0` for a normalized
/// two-state coupled 2D Bloch–McConnell spectrum (invariant across exchange regimes since
/// `int Re[1/(R + i*w)] dw = pi` and `p_A + p_B = 1`).
#[inline]
pub fn analytical_2d_integrated_volume(amplitude: f64) -> f64 {
    PI * PI * amplitude
}
