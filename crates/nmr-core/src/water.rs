//! Clean-room 1D, 2D, and 3D post-FFT frequency-domain solvent / water resonance suppression
//! and $t_1$-noise ridge removal.
//!
//! **Operating domain:** The functions in this module (`suppress_water_spectrum_1d`,
//! `suppress_water_spectrum_2d`, `suppress_water_spectrum_3d`) operate on **already-processed
//! frequency-domain spectra** (`Spectrum1D`, `Spectrum2D`, `Spectrum3D`) by reconstructing a
//! causal analytic pseudo-FID via inverse FFT / discrete Hilbert transform (Bartholdi & Ernst 1973),
//! subtracting the carrier-shifted Gaussian-convolved low-frequency solvent component (adapted from
//! Marion, Ikura & Bax 1989 with kernel radius strictly capped below a quarter of the pseudo-FID
//! length), and removing indirect $t_1$-noise ridges via Klevit (1985) sliding-median subtraction.
//! (For pre-apodization raw time-domain FID solvent filtering prior to FFT, see `process.rs`.)
//!
//! Provenance: Original clean-room implementation strictly from peer-reviewed NMR signal-processing
//! literature (zero external or commercial source code consulted):
//! - Marion, D., Ikura, M., & Bax, A. (*J. Magn. Reson.* **1989**, 84, 425–430) — carrier-shifted
//!   low-frequency Gaussian time-domain convolution subtraction of residual $\text{H}_2\text{O}$ /
//!   $\text{HDO}$ solvent signals with reflective $t = 0$ boundary conditioning.
//! - Klevit, R. E. (*Methods Enzymol.* **1985**, 118, 227–242) — indirect-dimension $t_1$-noise
//!   ridge suppression via rank-order median baseline subtraction along solvent-contaminated columns.
//! - Bartholdi, E., & Ernst, R. R. (*J. Magn. Reson.* **1973**, 11, 9–19) — discrete causal Hilbert
//!   transform reconstruction of the dispersion channel from real-only frequency spectra (`1r`,
//!   `2rr`, `3rrr`, `.ucsf`).
//! - Gottlieb, H. E., Kotlyar, V., & Nudelman, A. (*J. Org. Chem.* **1997**, 62, 7512–7515);
//!   Fulmer, G. R., et al. (*Organometallics* **2010**, 29, 2176–2179) — residual water chemical
//!   shifts in aqueous ($\sim 4.70\text{ ppm}$) and $\text{DMSO-}d_6$ ($\sim 3.33\text{ ppm}$) solutions.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::fid::Complex64;
use crate::nmr1d::{hilbert_reconstruct_imaginary_1d, Spectrum1D};
use crate::ops::fft_arbitrary_in_place;
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};

/// Sliding median window length (in frequency bins) for Klevit (1985) indirect $t_1$-ridge removal,
/// wide enough (`65` bins) to preserve narrow $\text{H}^\alpha$ / solute cross-peaks riding on the
/// water ridge while removing vertical $t_1$-noise streaks and diagonal solvent pedestals.
const KLEVIT_MEDIAN_WINDOW_PTS: usize = 65;

/// Configuration for 1D, 2D, and 3D post-FFT frequency-domain water suppression
/// (adapted from Marion, Ikura & Bax 1989) and Klevit (1985) $t_1$-ridge median subtraction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WaterSuppressionConfig {
    /// Explicit solvent chemical shift in ppm (`None` = auto-detect via [`detect_water_ppm`]).
    pub water_ppm: Option<f64>,
    /// Nominal outer half-width parameter for the solvent stopband in Hz (`35.0 Hz` default).
    /// The effective 95%-retention stopband half-width scales with the axis digital resolution
    /// ($\Delta\nu_{\text{pt}} = \text{SW}_{\text{Hz}} / N$, clamped to $\le 0.50 \times \text{half\_width\_hz}$
    /// when $\Delta\nu_{\text{pt}}$ permits) so solute peaks at $\pm 0.05\text{ ppm}$ retain $> 97\%$
    /// amplitude and peaks at $\ge 0.08\text{ ppm}$ retain $> 99\%$ amplitude on well-digitized 1D/2D spectra.
    pub half_width_hz: f64,
    /// Subtract Klevit (1985) sliding-median $t_1$-noise ridges in the indirect water slab.
    pub suppress_t1_ridge: bool,
    /// Strictly opt-in (`false` by default): when `axis_y` is also `${}^{1}\text{H}$ (homonuclear `NOESY`,
    /// `TOCSY`, `COSY`, `ROESY`), apply symmetric indirect `F1` Marion–Ikura–Bax + `F2` row ridge removal.
    /// Leaving this `false` by default prevents indirect `F1` attenuation of $\text{H}^N\text{--}\text{H}^\alpha$
    /// cross-peaks near $F_1 \approx 4.50\text{--}4.65\text{ ppm}$.
    #[serde(default, alias = "suppress_indirect_f1")]
    pub suppress_homonuclear_f1: bool,
}

impl WaterSuppressionConfig {
    /// Whether opt-in indirect homonuclear `F1` water suppression is enabled.
    #[inline]
    pub fn suppress_indirect_f1(&self) -> bool {
        self.suppress_homonuclear_f1
    }

    /// Builder helper to explicitly enable or disable indirect homonuclear `F1` water suppression.
    #[inline]
    pub fn with_suppress_indirect_f1(mut self, enabled: bool) -> Self {
        self.suppress_homonuclear_f1 = enabled;
        self
    }
}

impl Default for WaterSuppressionConfig {
    fn default() -> Self {
        Self {
            water_ppm: None,
            half_width_hz: 35.0,
            suppress_t1_ridge: true,
            suppress_homonuclear_f1: false,
        }
    }
}

fn median_slice_mut(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mid = vals.len() / 2;
    let (_, &mut m, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    m
}

fn robust_abs_mad_and_max(profile: &[f64]) -> (f64, f64) {
    if profile.is_empty() {
        return (0.0, 0.0);
    }
    let mut abs_vals: Vec<f64> = Vec::with_capacity(profile.len());
    let mut gmax = 0.0_f64;
    for &v in profile {
        let av = if v.is_finite() { v.abs() } else { 0.0 };
        if av > gmax {
            gmax = av;
        }
        abs_vals.push(av);
    }
    let med = median_slice_mut(&mut abs_vals);
    for v in &mut abs_vals {
        *v = (*v - med).abs();
    }
    let sigma = 1.4826 * median_slice_mut(&mut abs_vals);
    (sigma, gmax)
}

/// Find `(peak_abs, refined_ppm)` inside `[lo_ppm, hi_ppm]` on `axis` using a phase-invariant
/// local $|S(\omega)|^2$ energy centroid (`±0.025 ppm`, `|v| >= 0.35 * peak_abs`) so both pure
/// absorptive singlets and antisymmetric dispersive `1r` water derivatives localize to their true center.
fn find_band_extremum(
    axis: &AxisPpm,
    profile: &[f64],
    lo_ppm: f64,
    hi_ppm: f64,
) -> Option<(f64, f64)> {
    let n = axis.size.min(profile.len());
    if n == 0 {
        return None;
    }
    let (lo, hi) = (lo_ppm.min(hi_ppm), lo_ppm.max(hi_ppm));
    let mut best_idx: Option<usize> = None;
    let mut best_abs = 0.0_f64;
    for (i, &v) in profile.iter().take(n).enumerate() {
        if !v.is_finite() {
            continue;
        }
        let ppm = axis.index_to_ppm(i as f64);
        if (lo..=hi).contains(&ppm) {
            let av = v.abs();
            if av > best_abs || (best_idx.is_none() && av >= 0.0) {
                best_abs = av;
                best_idx = Some(i);
            }
        }
    }
    let peak_idx = best_idx?;
    let peak_ppm = axis.index_to_ppm(peak_idx as f64);
    if best_abs <= 0.0 {
        return Some((0.0, peak_ppm));
    }

    let step_abs = axis.step_ppm.abs().max(1e-9);
    let rad = ((0.025 / step_abs).ceil() as usize).clamp(2, 64);
    let i0 = peak_idx.saturating_sub(rad);
    let i1 = (peak_idx + rad).min(n - 1);
    let cut = 0.35 * best_abs;
    let mut w_sum = 0.0_f64;
    let mut wp_sum = 0.0_f64;
    for (i, &v) in profile.iter().enumerate().take(i1 + 1).skip(i0) {
        if !v.is_finite() {
            continue;
        }
        let ppm = axis.index_to_ppm(i as f64);
        if !(lo..=hi).contains(&ppm) {
            continue;
        }
        let av = v.abs();
        if av >= cut {
            let w = av * av;
            w_sum += w;
            wp_sum += w * ppm;
        }
    }
    let refined_ppm = if w_sum > 0.0 {
        (wp_sum / w_sum).clamp(lo, hi)
    } else {
        peak_ppm
    };
    Some((best_abs, refined_ppm))
}

/// Detect the dominant residual water / HDO chemical shift (in ppm) along `axis` from 1D `profile`:
/// - Searches `4.60..=5.00 ppm` for aqueous $\text{H}_2\text{O}/\text{D}_2\text{O}$ ($\sim 4.70\text{ ppm}$).
/// - Searches `3.22..=3.48 ppm` when `solvent_hint` indicates `DMSO` (`"DMSO"`, `"DMSO-d6"`, etc.)
///   or when no strong `4.7 ppm` water peak is present and a dominant solvent singlet sits near `3.33 ppm`
///   (Gottlieb et al. 1997; Fulmer et al. 2010).
/// - Falls back to `4.70 ppm` when `profile` is empty or flat noise.
pub fn detect_water_ppm(axis: &AxisPpm, profile: &[f64], solvent_hint: Option<&str>) -> f64 {
    let hint_low = solvent_hint.unwrap_or("").trim().to_ascii_lowercase();
    let is_dmso_hint = hint_low.contains("dmso");
    let is_aqueous_hint =
        hint_low.contains("h2o") || hint_low.contains("d2o") || hint_low.contains("water");

    let n = axis.size.min(profile.len());
    if n == 0 {
        if is_dmso_hint && axis.min_ppm() <= 3.33 && axis.max_ppm() >= 3.33 {
            return 3.33;
        }
        return 4.70;
    }

    let (sigma, global_max) = robust_abs_mad_and_max(&profile[..n]);
    if global_max <= 0.0 {
        if is_dmso_hint && axis.min_ppm() <= 3.33 && axis.max_ppm() >= 3.33 {
            return 3.33;
        }
        return 4.70;
    }

    let h2o_cand = find_band_extremum(axis, &profile[..n], 4.60, 5.00);
    let dmso_cand = find_band_extremum(axis, &profile[..n], 3.22, 3.48);
    let noise_gate = (2.5 * sigma).max(1e-30);

    if is_dmso_hint {
        if let Some((amp, ppm)) = dmso_cand {
            if amp > noise_gate {
                return ppm;
            }
        }
        if axis.min_ppm() <= 3.33 && axis.max_ppm() >= 3.33 {
            return 3.33;
        }
        return 4.70;
    }

    if is_aqueous_hint {
        if let Some((amp, ppm)) = h2o_cand {
            if amp > noise_gate {
                return ppm;
            }
        }
        return 4.70;
    }

    let h2o_amp = h2o_cand.map(|c| c.0).unwrap_or(0.0);
    let dmso_amp = dmso_cand.map(|c| c.0).unwrap_or(0.0);

    // Un-hinted DMSO-d6 detection (e.g. `.ucsf` files without solvent header):
    // When no strong ~4.70 ppm solvent peak exists (`h2o_amp < 0.15 * global_max`) and a dominant
    // water singlet in `3.22..3.48 ppm` dominates both the spectrum (`>= 0.50 * global_max`) and
    // the 4.7 ppm band by > 5x, select the ~3.33 ppm DMSO water peak.
    if let Some((_, dmso_ppm)) = dmso_cand {
        if dmso_amp > (6.0 * sigma).max(1e-30)
            && dmso_amp >= 0.50 * global_max
            && h2o_amp < 0.15 * global_max
            && dmso_amp > 5.0 * h2o_amp
        {
            return dmso_ppm;
        }
    }

    if let Some((amp, h2o_ppm)) = h2o_cand {
        if amp > noise_gate {
            return h2o_ppm;
        }
    }

    4.70
}

fn is_proton_axis(axis: &AxisPpm) -> bool {
    let up = axis.label.trim().to_ascii_uppercase();
    if up.contains('H') {
        return true;
    }
    if up.contains('C') || up.contains('N') || up.contains('P') || up.contains('F') {
        return false;
    }
    axis.obs_mhz >= 290.0 && axis.min_ppm() >= -5.0 && axis.max_ppm() <= 22.0
}

fn water_overlaps_axis(axis: &AxisPpm, water_ppm: f64, half_width_hz: f64) -> bool {
    if axis.size < 4 {
        return false;
    }
    let margin_ppm = 2.0 * (half_width_hz.abs().max(1.0) / axis.obs_mhz.max(1.0));
    water_ppm >= axis.min_ppm() - margin_ppm && water_ppm <= axis.max_ppm() + margin_ppm
}

/// Compute the digital-resolution-scaled effective 95%-retention stopband half-width (in Hz):
/// scales with $\Delta\nu_{\text{pt}} = \text{SW}_{\text{Hz}} / N$ and clamps to $\le 0.65 \times \text{half\_width\_hz}$
/// when digital resolution permits, so solute peaks at $\pm 0.05\text{ ppm}$ (`25..30 Hz` at `500..600 MHz`)
/// retain $> 98\%$ amplitude while dispersive `1r` water wings are suppressed $> 1000\times$.
#[inline]
fn effective_mib_half_width_hz(axis: &AxisPpm, half_width_hz: f64) -> f64 {
    let req = half_width_hz.abs().max(1.0);
    let hz_per_pt = (axis.step_ppm.abs() * axis.obs_mhz.max(1.0)).max(1e-3);
    (0.65 * req).max(1.75 * hz_per_pt).min(req)
}

/// Precomputed Gaussian time-domain convolution kernel for Marion, Ikura & Bax (1989):
/// - Scales the effective stopband half-width `hw_eff` by the axis digital resolution $\Delta\nu_{\text{pt}}$,
/// - Strictly caps the convolution kernel radius to $\le \text{valid\_len} / 4$ (`radius << valid_len`,
///   where `valid_len = n / 2 + 1` is the causal analytic pseudo-FID length) so the Gaussian window
///   never spans the entire pseudo-FID on short 2D/3D axes.
fn build_mib_gaussian_weights(axis: &AxisPpm, half_width_hz: f64) -> Vec<f64> {
    let n = axis.size.max(1);
    let valid_len = (n / 2 + 1).max(4);
    let max_radius = (valid_len / 4)
        .max(4)
        .min(valid_len.saturating_sub(1).max(1));
    let sw_hz = (axis.step_ppm.abs() * (n as f64) * axis.obs_mhz.max(1.0)).max(1.0);
    let hw_eff = effective_mib_half_width_hz(axis, half_width_hz);
    let sigma_target = (1.5_f64).sqrt() * sw_hz / (PI * hw_eff);
    let max_sigma = ((max_radius as f64) / 3.5).max(1.5);
    let sigma_pts = sigma_target.clamp(1.5, max_sigma);
    let radius = ((3.5 * sigma_pts).ceil() as usize).clamp(2, max_radius);
    let inv_two_sigma2 = 0.5 / (sigma_pts * sigma_pts);
    let mut weights = Vec::with_capacity(radius + 1);
    for d in 0..=radius {
        let df = d as f64;
        weights.push((-df * df * inv_two_sigma2).exp());
    }
    weights
}

/// Apply Marion–Ikura–Bax (1989) carrier-shifted Gaussian time-domain convolution subtraction
/// plus smooth raised-cosine residual notch polish inside the immediate water core on a single
/// descending-ppm 1D trace (`real`, with optional pre-existing `imag`).
#[allow(clippy::too_many_arguments)]
fn suppress_mib_trace_in_place(
    axis: &AxisPpm,
    real: &mut [f64],
    mut imag: Option<&mut [f64]>,
    hilbert_endpoints: bool,
    water_ppm: f64,
    half_width_hz: f64,
    gauss_weights: &[f64],
    work: &mut Vec<Complex64>,
    orig_td: &mut Vec<Complex64>,
) {
    let n = axis.size.min(real.len());
    if n < 8 || !water_overlaps_axis(axis, water_ppm, half_width_hz) {
        return;
    }
    let half = n / 2;
    let hw = half_width_hz.abs().max(1.0);
    let obs_mhz = axis.obs_mhz.max(1.0);
    let hz_per_pt = (axis.step_ppm.abs() * obs_mhz).max(1e-3);
    let hw_eff = effective_mib_half_width_hz(axis, half_width_hz);

    // Pre-check whether a local solvent peak or dispersive solvent glitch is actually centered at
    // `water_ppm` on this trace (as opposed to flat baseline or the monotonically decaying tail of
    // an adjacent solute peak at e.g. 4.60 / 4.65 ppm when no water peak is present).
    let mut pre_hi = Vec::new();
    let mut pre_lo = Vec::new();
    for (j, &rv) in real.iter().take(n).enumerate() {
        let signed_df_hz = (axis.index_to_ppm(j as f64) - water_ppm) * obs_mhz;
        let abs_df_hz = signed_df_hz.abs();
        if (2.50 * hw..=6.00 * hw).contains(&abs_df_hz) && rv.is_finite() {
            if signed_df_hz >= 0.0 {
                pre_hi.push(rv);
            } else {
                pre_lo.push(rv);
            }
        }
    }
    let pre_base = match (pre_hi.len() >= 2, pre_lo.len() >= 2) {
        (true, true) => 0.5 * (median_slice_mut(&mut pre_hi) + median_slice_mut(&mut pre_lo)),
        (true, false) => median_slice_mut(&mut pre_hi),
        (false, true) => median_slice_mut(&mut pre_lo),
        (false, false) => 0.0,
    };

    let core_check_hz = (0.35 * hw).max(2.0 * hz_per_pt);
    let shoulder_hi_hz = (2.2 * core_check_hz).max(core_check_hz + 2.0 * hz_per_pt);
    let mut core_dev = 0.0_f64;
    let mut down_shoulder_dev = 0.0_f64;
    let mut up_shoulder_dev = 0.0_f64;
    for (j, &r_val) in real.iter().enumerate().take(n) {
        let rv = if r_val.is_finite() {
            r_val - pre_base
        } else {
            0.0
        };
        let iv = imag
            .as_ref()
            .and_then(|im| im.get(j).copied())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0);
        let mag = rv.hypot(iv);
        let signed_df_hz = (axis.index_to_ppm(j as f64) - water_ppm) * obs_mhz;
        let abs_df_hz = signed_df_hz.abs();
        if abs_df_hz <= core_check_hz {
            core_dev = core_dev.max(mag);
        } else if abs_df_hz <= shoulder_hi_hz {
            if signed_df_hz > 0.0 {
                down_shoulder_dev = down_shoulder_dev.max(mag);
            } else {
                up_shoulder_dev = up_shoulder_dev.max(mag);
            }
        }
    }
    if core_dev <= 1e-12 || core_dev < down_shoulder_dev.max(up_shoulder_dev) {
        return;
    }

    work.clear();
    work.resize(n, [0.0, 0.0]);
    let reconstruct_hilbert = imag.is_none();
    let scale_hilbert_ep = reconstruct_hilbert || hilbert_endpoints;
    if let Some(ref im_slice) = imag {
        for (m, dst) in work.iter_mut().enumerate() {
            let j = (half + n - (m % n)) % n;
            *dst = [real[j], im_slice[j]];
        }
    } else {
        for (m, dst) in work.iter_mut().enumerate() {
            let j = (half + n - (m % n)) % n;
            *dst = [real[j], 0.0];
        }
    }

    // Inverse FFT to time domain (`work[0]` = t = 0).
    fft_arbitrary_in_place(work, true);

    // Reconstruct causal analytic FID via Bartholdi & Ernst (1973) if input was real-only.
    let valid_len = if reconstruct_hilbert {
        if n.is_multiple_of(2) {
            for c in work.iter_mut().take(half).skip(1) {
                c[0] *= 2.0;
                c[1] *= 2.0;
            }
            for c in work.iter_mut().skip(half + 1) {
                *c = [0.0, 0.0];
            }
        } else {
            for c in work.iter_mut().take(half + 1).skip(1) {
                c[0] *= 2.0;
                c[1] *= 2.0;
            }
            for c in work.iter_mut().skip(half + 1) {
                *c = [0.0, 0.0];
            }
        }
        (half + 1).min(n)
    } else {
        let mut pos_e = 0.0_f64;
        let mut tail_e = 0.0_f64;
        for c in work.iter().take((half + 1).min(n)) {
            pos_e += c[0] * c[0] + c[1] * c[1];
        }
        for c in work.iter().skip(half + 1) {
            tail_e += c[0] * c[0] + c[1] * c[1];
        }
        if tail_e < 1e-6 * pos_e {
            (half + 1).min(n)
        } else {
            n
        }
    };

    let max_radius = (valid_len / 4)
        .max(4)
        .min(valid_len.saturating_sub(1).max(1));
    let sw_hz = (hz_per_pt * (n as f64)).max(1.0);
    let sigma_target = (1.5_f64).sqrt() * sw_hz / (PI * hw_eff);
    let radius_capped = (gauss_weights.len().saturating_sub(1) as f64) < 3.30 * sigma_target;

    if valid_len >= 4 && !gauss_weights.is_empty() {
        // Continuous DFT bin offset `k_water = half - j_water`:
        // A resonance at `j_water` oscillates as `exp(+i * 2*pi * k_water * m / n)` in `work[m]`.
        let j_water = axis.ppm_to_index(water_ppm);
        let k_water = (half as f64) - j_water;
        let dphi = 2.0 * PI * k_water / (n as f64);

        // Shift `omega_water` to DC (`exp(-i * dphi * m)`).
        // When `reconstruct_hilbert` doubled `m = 1..half` (`2 * IDFT(R)[m]`), scale the endpoints
        // `m = 0` (and `m = half` for even `n`) by `2.0` before Gaussian convolution and restore
        // `0.5` afterward so a pure water sinusoid is flat across `m = 0..=half` with zero DC leak!
        orig_td.clear();
        orig_td.reserve(valid_len);
        for (m, &c) in work.iter().take(valid_len).enumerate() {
            let ep_in = if scale_hilbert_ep && (m == 0 || (n.is_multiple_of(2) && m == half)) {
                2.0
            } else {
                1.0
            };
            let (s, co) = ((m as f64) * dphi).sin_cos();
            orig_td.push([
                ep_in * (c[0] * co + c[1] * s),
                ep_in * (-c[0] * s + c[1] * co),
            ]);
        }

        // Convolve shifted FID with normalized Gaussian window using reflective boundary padding
        // at m = 0 and m = valid_len - 1 (Marion, Ikura & Bax 1989) with radius strictly capped
        // to `<= valid_len / 4` (`max_radius`), and subtract slow DC water, then shift back.
        let radius = (gauss_weights.len() - 1).min(max_radius).min(valid_len - 1);
        let last = (valid_len - 1) as isize;
        let w0 = gauss_weights[0];
        for (m, dst) in work.iter_mut().take(valid_len).enumerate() {
            let mut sum_re = w0 * orig_td[m][0];
            let mut sum_im = w0 * orig_td[m][1];
            let mut w_tot = w0;
            let mi = m as isize;
            for (d, &w) in gauss_weights.iter().enumerate().take(radius + 1).skip(1) {
                let di = d as isize;
                let idx_l = (mi - di).unsigned_abs().min(valid_len - 1);
                let raw_r = mi + di;
                let idx_r = if raw_r <= last {
                    raw_r as usize
                } else {
                    (2 * last - raw_r).max(0) as usize
                };
                sum_re += w * (orig_td[idx_l][0] + orig_td[idx_r][0]);
                sum_im += w * (orig_td[idx_l][1] + orig_td[idx_r][1]);
                w_tot += 2.0 * w;
            }
            let inv_w = 1.0 / w_tot;
            let ep_out = if scale_hilbert_ep && (m == 0 || (n.is_multiple_of(2) && m == half)) {
                0.5
            } else {
                1.0
            };
            let sub_re = ep_out * (orig_td[m][0] - sum_re * inv_w);
            let sub_im = ep_out * (orig_td[m][1] - sum_im * inv_w);
            let (s, co) = ((m as f64) * dphi).sin_cos();
            *dst = [sub_re * co - sub_im * s, sub_re * s + sub_im * co];
        }
    }

    // Forward FFT back to descending-ppm frequency domain.
    fft_arbitrary_in_place(work, false);

    // Smooth C^1 raised-cosine residual notch polish strictly inside the water core
    // (`inner_zero_hz .. outer_edge_hz`, bounded well below ±0.05 ppm) so clipped or non-causal
    // dispersive `1r` water glitches flatten cleanly to local baseline while preserving nearby
    // solute resonances (`>= 0.05 ppm`).
    let inner_zero_hz = (0.15 * hw).max(4.2 * hz_per_pt).min(0.35 * hw);
    let outer_edge_hz = (0.45 * hw)
        .max(inner_zero_hz + 1.5 * hz_per_pt)
        .min(0.48 * hw);
    let ramp_hz = (outer_edge_hz - inner_zero_hz).max(1e-6);

    // Estimate local baseline from symmetric downfield/upfield outer flanks (`2.5 * hw .. 6.0 * hw`)
    // so odd dispersive 1/df water wings (+D downfield, -D upfield) cancel identically to zero,
    // nearby H-alpha solute peaks (`>= 0.08 ppm` ~ `40..64 Hz`) do not bias the baseline, and any
    // constant/linear baseline offset B0 is preserved.
    let mut flank_hi = Vec::new();
    let mut flank_lo = Vec::new();
    for j in 0..n {
        let m = (half + n - (j % n)) % n;
        let signed_df_hz = (axis.index_to_ppm(j as f64) - water_ppm) * obs_mhz;
        let abs_df_hz = signed_df_hz.abs();
        if (2.50 * hw..=6.00 * hw).contains(&abs_df_hz) && work[m][0].is_finite() {
            if signed_df_hz >= 0.0 {
                flank_hi.push(work[m][0]);
            } else {
                flank_lo.push(work[m][0]);
            }
        }
    }
    let local_base = match (flank_hi.len() >= 2, flank_lo.len() >= 2) {
        (true, true) => 0.5 * (median_slice_mut(&mut flank_hi) + median_slice_mut(&mut flank_lo)),
        (true, false) => median_slice_mut(&mut flank_hi),
        (false, true) => median_slice_mut(&mut flank_lo),
        (false, false) => 0.0,
    };

    for j in 0..n {
        let m = (half + n - (j % n)) % n;
        let df_hz = (axis.index_to_ppm(j as f64) - water_ppm).abs() * obs_mhz;
        let taper = if df_hz >= outer_edge_hz {
            1.0
        } else if df_hz <= inner_zero_hz {
            0.0
        } else {
            let u = (df_hz - inner_zero_hz) / ramp_hz;
            0.5 * (1.0 - (PI * u).cos())
        };
        // When `radius` was capped by `max_radius = valid_len / 4` on a short 2D/3D axis,
        // localize the MIB subtraction envelope in frequency domain so the capped time-domain
        // kernel cannot broaden the stopband beyond `hw_eff`.
        let w_loc = if radius_capped {
            (-3.0 * (df_hz / hw_eff).powi(2)).exp()
        } else {
            1.0
        };
        let mib_re = real[j] - w_loc * (real[j] - work[m][0]);
        let re_val = local_base + (mib_re - local_base) * taper;
        real[j] = re_val;
        if let Some(ref mut im_slice) = imag {
            let mib_im = im_slice[j] - w_loc * (im_slice[j] - work[m][1]);
            im_slice[j] = mib_im * taper;
        }
    }
}

/// Subtract a `window_pts`-point (`65`-pt) sliding-median baseline along `trace` weighted by `blend`
/// (`Klevit, Methods Enzymol. 1985, 118, 227–242`), preserving narrow cross-peaks (`FWHM << window_pts`)
/// while removing broad indirect $t_1$-noise ridges and diagonal water pedestals.
/// Bounded by the whole-trace median magnitude so a ridge-free column with isolated cross-peaks
/// never self-subtracts cross-peak amplitude.
fn subtract_klevit_sliding_median_1d(trace: &mut [f64], window_pts: usize, blend: f64) {
    let n = trace.len();
    if n < 4 || blend <= 0.0 {
        return;
    }
    let mut abs_all: Vec<f64> = trace
        .iter()
        .map(|&v| if v.is_finite() { v.abs() } else { 0.0 })
        .collect();
    let max_ridge = 1.5 * median_slice_mut(&mut abs_all);
    if max_ridge <= 1e-12 {
        return;
    }
    let eff_win = if n < window_pts {
        ((n / 2) | 1).max(3).min(n)
    } else {
        window_pts | 1
    };
    let half_w = eff_win / 2;
    let orig = trace.to_vec();
    let mut stack_scratch = [0.0_f64; KLEVIT_MEDIAN_WINDOW_PTS];
    let mut heap_scratch: Vec<f64> = if eff_win > KLEVIT_MEDIAN_WINDOW_PTS {
        vec![0.0; eff_win]
    } else {
        Vec::new()
    };

    for (i, dst) in trace.iter_mut().enumerate() {
        let s = i.saturating_sub(half_w);
        let e = (i + half_w).min(n - 1);
        let len = e - s + 1;
        let med = if len <= KLEVIT_MEDIAN_WINDOW_PTS {
            stack_scratch[..len].copy_from_slice(&orig[s..=e]);
            median_slice_mut(&mut stack_scratch[..len])
        } else {
            heap_scratch[..len].copy_from_slice(&orig[s..=e]);
            median_slice_mut(&mut heap_scratch[..len])
        };
        let bounded_med = med.clamp(-max_ridge, max_ridge);
        *dst = orig[i] - blend * bounded_med;
    }
}

/// Smooth feathering weight in `[0.0, 1.0]` across the Klevit indirect water-ridge slab
/// $|F - \delta_{\text{water}}| \le 2.4 \times (\text{half\_width\_hz} / \text{obs\_mhz})$ (`1.0` inside
/// `1.8x`, raised-cosine taper across `1.8x..2.4x` so slab edges never leave a step seam).
#[inline]
fn klevit_slab_blend(ppm: f64, water_ppm: f64, half_width_hz: f64, obs_mhz: f64) -> f64 {
    let hw_ppm = half_width_hz.abs().max(1.0) / obs_mhz.max(1.0);
    let d = (ppm - water_ppm).abs();
    let inner = 1.8 * hw_ppm;
    let outer = 2.4 * hw_ppm;
    if d >= outer {
        0.0
    } else if d <= inner {
        1.0
    } else {
        let u = (d - inner) / (outer - inner).max(1e-12);
        0.5 * (1.0 + (PI * u).cos())
    }
}

fn parallel_rows_mut<T, F>(slice: &mut [T], row_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    if slice.is_empty() || row_len == 0 {
        return;
    }
    let total_rows = slice.len() / row_len;
    if total_rows == 0 {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);
        let rows_per_chunk = total_rows.div_ceil(threads * 2).max(1);
        let chunk_elems = rows_per_chunk * row_len;
        if slice.len() <= chunk_elems {
            f(0, slice);
        } else {
            std::thread::scope(|s| {
                for (ci, chunk) in slice.chunks_mut(chunk_elems).enumerate() {
                    let f_ref = &f;
                    let base_row = ci * rows_per_chunk;
                    s.spawn(move || f_ref(base_row, chunk));
                }
            });
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        f(0, slice);
    }
}

/// Suppress the solvent / water peak in a processed 1D spectrum (`Spectrum1D`) in place via causal
/// Hilbert reconstruction (when `spec.imag` is `None`), carrier-shifted Marion–Ikura–Bax (1989)
/// Gaussian time-domain convolution subtraction (`radius <= valid_len / 4`), and smooth raised-cosine
/// residual notch polish inside the immediate water core. Returns the detected or configured `water_ppm`.
pub fn suppress_water_spectrum_1d(spec: &mut Spectrum1D, cfg: &WaterSuppressionConfig) -> f64 {
    let water_ppm = cfg
        .water_ppm
        .unwrap_or_else(|| detect_water_ppm(&spec.axis, &spec.real, spec.solvent.as_deref()));
    if spec.len() < 8 {
        return water_ppm;
    }
    let was_real_only = spec.imag.is_none();
    if was_real_only {
        hilbert_reconstruct_imaginary_1d(spec);
    }
    let weights = build_mib_gaussian_weights(&spec.axis, cfg.half_width_hz);
    let mut work = Vec::with_capacity(spec.axis.size);
    let mut orig_td = Vec::with_capacity(spec.axis.size);
    suppress_mib_trace_in_place(
        &spec.axis,
        &mut spec.real,
        spec.imag.as_deref_mut(),
        was_real_only,
        water_ppm,
        cfg.half_width_hz,
        &weights,
        &mut work,
        &mut orig_td,
    );
    water_ppm
}

/// Suppress the solvent / water resonance and indirect $t_1$-noise ridges in a processed 2D spectrum
/// (`Spectrum2D`) in place, returning the detected or configured `water_ppm`:
/// 1. Along direct `F2` (`axis_x`, when `${}^{1}\text{H}$ or `cfg.water_ppm.is_some()`): real-to-causal
///    Hilbert -> IFFT -> Marion–Ikura–Bax (1989) Gaussian time-domain water subtraction (`radius <= valid_len / 4`)
///    + smooth core residual polish on rows containing a dominant solvent peak.
/// 2. Along `F1` (`axis_y`) columns within $|F_2 - \delta_{\text{water}}| \le 2.4 \times (\text{half\_width\_hz} / \text{obs\_mhz})$:
///    ridge-gated Klevit (1985) `65`-pt sliding-median $t_1$-ridge subtraction, preserving narrow $\text{H}^\alpha$
///    cross-peaks while removing vertical $t_1$-noise streaks.
/// 3. When opt-in `cfg.suppress_homonuclear_f1` is enabled and `axis_y` is also `${}^{1}\text{H}$ (`NOESY`, `TOCSY`,
///    `COSY`, `ROESY`): symmetric `F1` column Marion–Ikura–Bax + `F2` row `65`-pt sliding-median ridge subtraction
///    along $|F_1 - \delta_{\text{water}}|$.
pub fn suppress_water_spectrum_2d(spec: &mut Spectrum2D, cfg: &WaterSuppressionConfig) -> f64 {
    let nx = spec.axis_x.size;
    let ny = spec.axis_y.size;
    let x_is_h = is_proton_axis(&spec.axis_x);
    let y_is_h = is_proton_axis(&spec.axis_y);

    // Build row-persistent solvent profile: require at least 25% of indirect rows (`q75 > 0`)
    // to carry signal at column `ix` (or homonuclear diagonal `(ix, iy)` dominance) so an isolated
    // off-water cross-peak at e.g. `(F2 = 4.65, F1 = 2.00)` on a water-free 2D spectrum is never
    // mistaken for residual water, while a true `(4.70, 4.70)` volcano + ridge peaks sharply at 4.70.
    let water_ppm = if let Some(w) = cfg.water_ppm {
        w
    } else if nx > 0 && ny > 0 && spec.data.len() >= nx * ny && (x_is_h || !y_is_h) {
        let mut prof_x = vec![0.0_f64; nx];
        let mut col_tmp = vec![0.0_f64; ny];
        let q75_idx = ((ny * 3) / 4).min(ny - 1);
        for (ix, px) in prof_x.iter_mut().enumerate() {
            let mut col_sum = 0.0_f64;
            for (iy, ct) in col_tmp.iter_mut().enumerate() {
                let v = spec.data[iy * nx + ix];
                let av = if v.is_finite() { (v as f64).abs() } else { 0.0 };
                *ct = av;
                col_sum += av;
            }
            let (_, &mut q75, _) = col_tmp.select_nth_unstable_by(q75_idx, |a, b| a.total_cmp(b));
            let mean_abs = col_sum / (ny as f64);
            // Isolated cross-peaks occupy << 25% of rows (`q75` near zero relative to `mean_abs`).
            if q75 >= 0.05 * mean_abs {
                *px = col_sum;
            }
        }
        detect_water_ppm(&spec.axis_x, &prof_x, None)
    } else if nx > 0 && ny > 0 && spec.data.len() >= nx * ny && y_is_h {
        let mut prof_y = vec![0.0_f64; ny];
        let mut row_tmp = vec![0.0_f64; nx];
        let q75_idx = ((nx * 3) / 4).min(nx - 1);
        for (iy, py) in prof_y.iter_mut().enumerate() {
            let row = &spec.data[iy * nx..(iy + 1) * nx];
            let mut row_sum = 0.0_f64;
            for (rt, &v) in row_tmp.iter_mut().zip(row.iter()) {
                let av = if v.is_finite() { (v as f64).abs() } else { 0.0 };
                *rt = av;
                row_sum += av;
            }
            let (_, &mut q75, _) = row_tmp.select_nth_unstable_by(q75_idx, |a, b| a.total_cmp(b));
            let mean_abs = row_sum / (nx as f64);
            if q75 >= 0.05 * mean_abs {
                *py = row_sum;
            }
        }
        detect_water_ppm(&spec.axis_y, &prof_y, None)
    } else {
        4.70
    };

    if nx < 8 || ny == 0 || spec.data.len() < nx * ny {
        return water_ppm;
    }

    let process_x = x_is_h || cfg.water_ppm.is_some();
    if process_x && water_overlaps_axis(&spec.axis_x, water_ppm, cfg.half_width_hz) {
        let weights_x = build_mib_gaussian_weights(&spec.axis_x, cfg.half_width_hz);
        let ax_x = spec.axis_x.clone();
        let hw = cfg.half_width_hz;
        parallel_rows_mut(&mut spec.data[..ny * nx], nx, |_, chunk| {
            let mut row_f64 = vec![0.0_f64; nx];
            let mut work = Vec::with_capacity(nx);
            let mut orig_td = Vec::with_capacity(nx);
            for row in chunk.chunks_exact_mut(nx) {
                for (d, &s) in row_f64.iter_mut().zip(row.iter()) {
                    *d = if s.is_finite() { s as f64 } else { 0.0 };
                }
                suppress_mib_trace_in_place(
                    &ax_x,
                    &mut row_f64,
                    None,
                    true,
                    water_ppm,
                    hw,
                    &weights_x,
                    &mut work,
                    &mut orig_td,
                );
                for (d, &s) in row.iter_mut().zip(row_f64.iter()) {
                    *d = s as f32;
                }
            }
        });
    }

    // Homonuclear `1H-1H` (`NOESY`, `TOCSY`, `COSY`, `ROESY`) opt-in symmetric `F1` column Marion–Ikura–Bax:
    // Run prior to Klevit sliding-median passes so the diagonal `(water_ppm, water_ppm)` volcano and
    // broad `F1` water dispersive wings are removed when explicitly opted into (`cfg.suppress_homonuclear_f1`).
    let process_homo_f1 = cfg.suppress_homonuclear_f1 && y_is_h && ny >= 8;
    if process_homo_f1 && water_overlaps_axis(&spec.axis_y, water_ppm, cfg.half_width_hz) {
        let weights_y = build_mib_gaussian_weights(&spec.axis_y, cfg.half_width_hz);
        let ax_y = spec.axis_y.clone();
        let hw = cfg.half_width_hz;
        // Transpose into `[nx * ny]` so columns transform in parallel with unit-stride cache locality.
        let mut transposed = vec![0.0_f32; nx * ny];
        for iy in 0..ny {
            let src_row = &spec.data[iy * nx..(iy + 1) * nx];
            for (ix, &v) in src_row.iter().enumerate() {
                transposed[ix * ny + iy] = v;
            }
        }
        parallel_rows_mut(&mut transposed, ny, |_, chunk| {
            let mut col_f64 = vec![0.0_f64; ny];
            let mut work = Vec::with_capacity(ny);
            let mut orig_td = Vec::with_capacity(ny);
            for col in chunk.chunks_exact_mut(ny) {
                for (d, &s) in col_f64.iter_mut().zip(col.iter()) {
                    *d = if s.is_finite() { s as f64 } else { 0.0 };
                }
                suppress_mib_trace_in_place(
                    &ax_y,
                    &mut col_f64,
                    None,
                    true,
                    water_ppm,
                    hw,
                    &weights_y,
                    &mut work,
                    &mut orig_td,
                );
                for (d, &s) in col.iter_mut().zip(col_f64.iter()) {
                    *d = s as f32;
                }
            }
        });
        for ix in 0..nx {
            let src_col = &transposed[ix * ny..(ix + 1) * ny];
            for (iy, &v) in src_col.iter().enumerate() {
                spec.data[iy * nx + ix] = v;
            }
        }
    }

    if cfg.suppress_t1_ridge {
        // Klevit (1985) `65`-pt sliding-median `F1` column ridge subtraction inside `F2` water slab.
        if process_x && ny >= 4 {
            let mut col_buf = vec![0.0_f64; ny];
            for ix in 0..nx {
                let f2_ppm = spec.axis_x.index_to_ppm(ix as f64);
                let blend =
                    klevit_slab_blend(f2_ppm, water_ppm, cfg.half_width_hz, spec.axis_x.obs_mhz);
                if blend <= 0.0 {
                    continue;
                }
                for (iy, dst) in col_buf.iter_mut().enumerate() {
                    let v = spec.data[iy * nx + ix];
                    *dst = if v.is_finite() { v as f64 } else { 0.0 };
                }
                subtract_klevit_sliding_median_1d(&mut col_buf, KLEVIT_MEDIAN_WINDOW_PTS, blend);
                for (iy, &val) in col_buf.iter().enumerate() {
                    spec.data[iy * nx + ix] = val as f32;
                }
            }
        }

        // Symmetric `F2` row `65`-pt sliding-median ridge subtraction inside `F1` water slab on homonuclear 2D.
        if process_homo_f1 && nx >= 4 {
            let mut row_buf = vec![0.0_f64; nx];
            for iy in 0..ny {
                let f1_ppm = spec.axis_y.index_to_ppm(iy as f64);
                let blend =
                    klevit_slab_blend(f1_ppm, water_ppm, cfg.half_width_hz, spec.axis_y.obs_mhz);
                if blend <= 0.0 {
                    continue;
                }
                let row_slice = &mut spec.data[iy * nx..(iy + 1) * nx];
                for (d, &s) in row_buf.iter_mut().zip(row_slice.iter()) {
                    *d = if s.is_finite() { s as f64 } else { 0.0 };
                }
                subtract_klevit_sliding_median_1d(&mut row_buf, KLEVIT_MEDIAN_WINDOW_PTS, blend);
                for (d, &s) in row_slice.iter_mut().zip(row_buf.iter()) {
                    *d = s as f32;
                }
            }
        }
    }

    // Smooth radial raised-cosine polish on the homonuclear `(water_ppm, water_ppm)` diagonal core
    // (`hypot(df_x_hz, df_y_hz) <= 0.45 * half_width_hz`, i.e. within ~15 Hz of `(4.70, 4.70)` in BOTH F1 and F2)
    // so any residual pedestal around the diagonal water volcano flattens cleanly to zero without
    // touching off-diagonal cross-peaks at `(8.30, 4.50..4.65)`.
    if process_x
        && y_is_h
        && water_overlaps_axis(&spec.axis_x, water_ppm, cfg.half_width_hz)
        && water_overlaps_axis(&spec.axis_y, water_ppm, cfg.half_width_hz)
    {
        let hw = cfg.half_width_hz.abs().max(1.0);
        let inner_hz = 0.18 * hw;
        let outer_hz = 0.45 * hw;
        let ramp_hz = (outer_hz - inner_hz).max(1e-6);
        let obs_x = spec.axis_x.obs_mhz.max(1.0);
        let obs_y = spec.axis_y.obs_mhz.max(1.0);
        for iy in 0..ny {
            let dy_hz = (spec.axis_y.index_to_ppm(iy as f64) - water_ppm).abs() * obs_y;
            if dy_hz >= outer_hz {
                continue;
            }
            for ix in 0..nx {
                let dx_hz = (spec.axis_x.index_to_ppm(ix as f64) - water_ppm).abs() * obs_x;
                let r_hz = dx_hz.hypot(dy_hz);
                if r_hz < outer_hz {
                    let taper = if r_hz <= inner_hz {
                        0.0
                    } else {
                        let u = (r_hz - inner_hz) / ramp_hz;
                        0.5 * (1.0 - (PI * u).cos())
                    };
                    spec.data[iy * nx + ix] *= taper as f32;
                }
            }
        }
    }

    water_ppm
}

/// Suppress the solvent / water resonance and indirect $t_1$/$t_2$ water-slab ridges in a 3D
/// spectrum (`Spectrum3D`) in place, returning the detected or configured `water_ppm`:
/// 1. Applies `F3` (`axis_x`) Marion–Ikura–Bax (1989) row water subtraction + smooth residual polish
///    across all `(iy, iz)` traces in parallel.
/// 2. Applies Klevit (1985) sliding-median ridge subtraction across the indirect `(iy, iz)` planes
///    inside the `F3` water slab $|F_3 - \delta_{\text{water}}| \le 3.0 \times (\text{half\_width\_hz} / \text{obs\_mhz})$.
pub fn suppress_water_spectrum_3d(spec: &mut Spectrum3D, cfg: &WaterSuppressionConfig) -> f64 {
    let nx = spec.axis_x.size;
    let ny = spec.axis_y.size;
    let nz = spec.axis_z.size;
    let total_rows = ny.saturating_mul(nz);
    let total_elems = total_rows.saturating_mul(nx);
    let x_is_h = is_proton_axis(&spec.axis_x);

    let water_ppm = if let Some(w) = cfg.water_ppm {
        w
    } else if nx > 0 && total_rows > 0 && spec.data.len() >= total_elems {
        let mut prof_x = vec![0.0_f64; nx];
        if total_rows >= 4 {
            let mut col = Vec::with_capacity(total_rows);
            for (ix, px_slot) in prof_x.iter_mut().enumerate().take(nx) {
                col.clear();
                let mut sum_abs = 0.0_f64;
                for r in 0..total_rows {
                    let v = spec.data[r * nx + ix];
                    if v.is_finite() {
                        let a = (v as f64).abs();
                        sum_abs += a;
                        col.push(a);
                    }
                }
                if !col.is_empty() {
                    let q75_idx = (3 * col.len()) / 4;
                    let (_, &mut q75, _) =
                        col.select_nth_unstable_by(q75_idx, |a, b| a.total_cmp(b));
                    let mean_abs = sum_abs / (col.len() as f64);
                    if q75 >= 0.05 * mean_abs {
                        *px_slot = sum_abs;
                    }
                }
            }
        } else {
            for row in spec.data[..total_elems].chunks_exact(nx) {
                for (p, &v) in prof_x.iter_mut().zip(row.iter()) {
                    if v.is_finite() {
                        *p += (v as f64).abs();
                    }
                }
            }
        }
        detect_water_ppm(&spec.axis_x, &prof_x, None)
    } else {
        4.70
    };

    if nx < 8 || total_rows == 0 || spec.data.len() < total_elems {
        return water_ppm;
    }

    let process_x = x_is_h || cfg.water_ppm.is_some();
    if process_x && water_overlaps_axis(&spec.axis_x, water_ppm, cfg.half_width_hz) {
        let weights_x = build_mib_gaussian_weights(&spec.axis_x, cfg.half_width_hz);
        let ax_x = spec.axis_x.clone();
        let hw = cfg.half_width_hz;
        parallel_rows_mut(&mut spec.data[..total_elems], nx, |_, chunk| {
            let mut row_f64 = vec![0.0_f64; nx];
            let mut work = Vec::with_capacity(nx);
            let mut orig_td = Vec::with_capacity(nx);
            for row in chunk.chunks_exact_mut(nx) {
                for (d, &s) in row_f64.iter_mut().zip(row.iter()) {
                    *d = if s.is_finite() { s as f64 } else { 0.0 };
                }
                suppress_mib_trace_in_place(
                    &ax_x,
                    &mut row_f64,
                    None,
                    true,
                    water_ppm,
                    hw,
                    &weights_x,
                    &mut work,
                    &mut orig_td,
                );
                for (d, &s) in row.iter_mut().zip(row_f64.iter()) {
                    *d = s as f32;
                }
            }
        });
    }

    if cfg.suppress_t1_ridge && process_x {
        let plane = ny * nx;
        let mut y_buf = vec![0.0_f64; ny];
        let mut z_buf = vec![0.0_f64; nz];
        for ix in 0..nx {
            let f3_ppm = spec.axis_x.index_to_ppm(ix as f64);
            let blend =
                klevit_slab_blend(f3_ppm, water_ppm, cfg.half_width_hz, spec.axis_x.obs_mhz);
            if blend <= 0.0 {
                continue;
            }
            if ny >= 4 {
                for iz in 0..nz {
                    let pbase = iz * plane + ix;
                    for (iy, dst) in y_buf.iter_mut().enumerate() {
                        let v = spec.data[pbase + iy * nx];
                        *dst = if v.is_finite() { v as f64 } else { 0.0 };
                    }
                    subtract_klevit_sliding_median_1d(&mut y_buf, KLEVIT_MEDIAN_WINDOW_PTS, blend);
                    for (iy, &val) in y_buf.iter().enumerate() {
                        spec.data[pbase + iy * nx] = val as f32;
                    }
                }
            }
            if nz >= 4 {
                for iy in 0..ny {
                    let yoff = iy * nx + ix;
                    for (iz, dst) in z_buf.iter_mut().enumerate() {
                        let v = spec.data[iz * plane + yoff];
                        *dst = if v.is_finite() { v as f64 } else { 0.0 };
                    }
                    subtract_klevit_sliding_median_1d(&mut z_buf, KLEVIT_MEDIAN_WINDOW_PTS, blend);
                    for (iz, &val) in z_buf.iter().enumerate() {
                        spec.data[iz * plane + yoff] = val as f32;
                    }
                }
            }
        }
    }

    water_ppm
}

/// Alias for [`suppress_water_spectrum_3d`]: suppress residual water and indirect $t_1$/$t_2$
/// water-slab ridges across a 3D spectrum (`Spectrum3D`) in place, returning `water_ppm`.
#[inline]
pub fn suppress_water_3d(spec: &mut Spectrum3D, cfg: &WaterSuppressionConfig) -> f64 {
    suppress_water_spectrum_3d(spec, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nmr1d::{fft_1d, synthesize_fid_1d, Experiment1DKind, SyntheticLine1D};

    #[test]
    fn test_detect_water_ppm_aqueous_dmso_and_fallback() {
        let axis = AxisPpm::from_bounds("1H", 4096, 11.0, -1.0, 600.13);

        // 1. Empty / flat profile -> fallback 4.70 (or 3.33 with DMSO hint)
        assert!((detect_water_ppm(&axis, &[], None) - 4.70).abs() < 1e-9);
        assert!((detect_water_ppm(&axis, &[], Some("DMSO-d6")) - 3.33).abs() < 1e-9);

        // 2. Aqueous water at 4.73 ppm
        let mut prof_h2o = vec![1e-3; axis.size];
        for (i, dst) in prof_h2o.iter_mut().enumerate() {
            let d = (axis.index_to_ppm(i as f64) - 4.73) / 0.008;
            *dst += 500.0 / (1.0 + d * d);
        }
        let w_h2o = detect_water_ppm(&axis, &prof_h2o, None);
        assert!((w_h2o - 4.73).abs() < 0.005, "got {w_h2o}");

        // 3. DMSO-d6 water singlet at 3.34 ppm with weak solute H-alpha at 4.68 ppm (un-hinted & hinted)
        let mut prof_dmso = vec![1e-3; axis.size];
        for (i, dst) in prof_dmso.iter_mut().enumerate() {
            let p = axis.index_to_ppm(i as f64);
            let d_w = (p - 3.34) / 0.006;
            let d_ha = (p - 4.68) / 0.006;
            *dst += 1000.0 / (1.0 + d_w * d_w) + 5.0 / (1.0 + d_ha * d_ha);
        }
        let w_dmso_auto = detect_water_ppm(&axis, &prof_dmso, None);
        assert!((w_dmso_auto - 3.34).abs() < 0.005, "got {w_dmso_auto}");
        let w_dmso_hint = detect_water_ppm(&axis, &prof_dmso, Some("DMSO"));
        assert!((w_dmso_hint - 3.34).abs() < 0.005, "got {w_dmso_hint}");
    }

    #[test]
    fn test_suppress_water_1d_flattens_dispersive_water_and_preserves_0p08_ppm_neighbor() {
        // Synthesize 600 MHz 1D 1H FID with a huge dispersive water resonance at 4.700 ppm
        // (1000x amplitude, 45 deg phase distortion) and a nearby H-alpha singlet 0.080 ppm away (4.780 ppm).
        let lines_ref = [SyntheticLine1D::lorentzian(4.780, 1.0, 3.0)];
        let mut td_ref = synthesize_fid_1d(
            "ref",
            Experiment1DKind::Proton1D,
            4096,
            7200.0,
            600.0,
            4.70,
            &lines_ref,
        );
        td_ref.data.resize(8192, [0.0, 0.0]);
        let spec_ref = fft_1d(&td_ref);

        let lines_wet = [
            SyntheticLine1D {
                ppm: 4.700,
                amplitude: 200.0,
                fwhm_hz: 0.5,
                gauss_fwhm_hz: 6.0,
                phase_deg: 0.0,
            },
            SyntheticLine1D::lorentzian(4.780, 1.0, 3.0),
            SyntheticLine1D::lorentzian(8.250, 1.0, 4.0),
        ];
        let mut td_wet = synthesize_fid_1d(
            "wet",
            Experiment1DKind::Proton1D,
            4096,
            7200.0,
            600.0,
            4.70,
            &lines_wet,
        );
        td_wet.data.resize(8192, [0.0, 0.0]);
        let mut spec_1r = fft_1d(&td_wet);
        // Simulate real-only processed `1r` on disk (`imag = None`).
        spec_1r.imag = None;

        let idx_water = spec_1r.axis.ppm_to_index(4.700).round() as usize;
        let idx_ha = spec_1r.axis.ppm_to_index(4.780).round() as usize;
        let ref_ha_amp = spec_ref.real[idx_ha];
        let raw_water_max = spec_1r.real[idx_water - 8..=idx_water + 8]
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        assert!(raw_water_max > 100.0 * ref_ha_amp);

        let cfg = WaterSuppressionConfig::default();
        let used_w = suppress_water_spectrum_1d(&mut spec_1r, &cfg);
        assert!((used_w - 4.700).abs() < 0.01, "used_w={used_w}");

        // Measure far-away peak-free baseline (around 1.0 ppm) so we measure residual deviation
        // from true baseline as well as absolute reduction.
        let idx_base = spec_1r.axis.ppm_to_index(1.000).round() as usize;
        let far_base = spec_1r.real[idx_base];
        let post_water_dev = spec_1r.real[idx_water - 4..=idx_water + 4]
            .iter()
            .fold(0.0_f64, |m, &v| m.max((v - far_base).abs()));
        assert!(
            post_water_dev < 1e-3 * raw_water_max && spec_1r.real[idx_water].abs() < 2e-3 * raw_water_max,
            "water peak must flatten to baseline (>1000x vs baseline): before={raw_water_max}, dev={post_water_dev}, val={}",
            spec_1r.real[idx_water]
        );

        let ref_base = spec_ref.real[idx_base];
        let post_ha_amp = spec_1r.real[idx_ha] - far_base;
        let retention = post_ha_amp / (ref_ha_amp - ref_base);
        assert!(
            retention > 0.95 && retention < 1.05,
            "solute peak 0.08 ppm from water must retain >95% amplitude, got {:.2}%",
            retention * 100.0
        );

        // Also verify a strongly dispersive (45 deg, 1000x) real-only `1r` water spike flattens >1000x to baseline.
        let lines_disp = [SyntheticLine1D {
            ppm: 4.700,
            amplitude: 1000.0,
            fwhm_hz: 6.0,
            gauss_fwhm_hz: 0.0,
            phase_deg: 45.0,
        }];
        let mut td_disp = synthesize_fid_1d(
            "disp",
            Experiment1DKind::Proton1D,
            4096,
            7200.0,
            600.0,
            4.70,
            &lines_disp,
        );
        td_disp.data.resize(8192, [0.0, 0.0]);
        let mut spec_disp = fft_1d(&td_disp);
        spec_disp.imag = None;
        let disp_raw_max = spec_disp.real[idx_water - 8..=idx_water + 8]
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        let _ = suppress_water_spectrum_1d(&mut spec_disp, &cfg);
        let disp_base = spec_disp.real[idx_base];
        let disp_post_dev = spec_disp.real[idx_water - 4..=idx_water + 4]
            .iter()
            .fold(0.0_f64, |m, &v| m.max((v - disp_base).abs()));
        assert!(disp_post_dev < 1e-3 * disp_raw_max);
    }

    #[test]
    fn test_suppress_water_2d_homonuclear_volcano_ridge_and_halpha_preservation() {
        let nx = 512usize;
        let ny = 256usize;
        let axis_x = AxisPpm::from_bounds("1H", nx, 10.0, 0.0, 500.0);
        let axis_y = AxisPpm::from_bounds("1H", ny, 10.0, 0.0, 500.0);
        let mut data = vec![0.0_f32; ny * nx];

        // Inject:
        // (a) Giant diagonal (4.70, 4.70) water volcano (amp = 5000)
        // (b) Vertical F1 t1-noise ridge at F2 = 4.70..4.75 ppm (amp = +80)
        // (c) Horizontal F2 ridge at F1 = 4.70 ppm (amp = +60)
        // (d) Narrow real H-alpha cross-peak riding RIGHT ON the vertical ridge wing at (F2 = 4.74 ppm, F1 = 8.20 ppm, amp = +25)
        // (e) Clean HN-HB cross-peak at (F2 = 8.20 ppm, F1 = 2.10 ppm, amp = +20)
        for iy in 0..ny {
            let y_ppm = axis_y.index_to_ppm(iy as f64);
            for ix in 0..nx {
                let x_ppm = axis_x.index_to_ppm(ix as f64);
                let dx_w = (x_ppm - 4.70) / 0.025;
                let dy_w = (y_ppm - 4.70) / 0.035;
                let volcano = 5000.0 / ((1.0 + dx_w * dx_w) * (1.0 + dy_w * dy_w));
                let v_ridge = 80.0 * (-0.5 * ((x_ppm - 4.72) / 0.04).powi(2)).exp();
                let h_ridge = 60.0 * (-0.5 * ((y_ppm - 4.70) / 0.04).powi(2)).exp();
                let ha_peak = 25.0
                    * (-0.5 * ((x_ppm - 4.80) / 0.018).powi(2)
                        - 0.5 * ((y_ppm - 8.20) / 0.045).powi(2))
                    .exp();
                let hb_peak = 20.0
                    * (-0.5 * ((x_ppm - 8.20) / 0.018).powi(2)
                        - 0.5 * ((y_ppm - 2.10) / 0.045).powi(2))
                    .exp();
                data[iy * nx + ix] = (volcano + v_ridge + h_ridge + ha_peak + hb_peak) as f32;
            }
        }

        let mut spec = Spectrum2D {
            name: "synthetic_noesy".to_string(),
            axis_x,
            axis_y,
            data,
        };
        let ix_hb = spec.axis_x.ppm_to_index(8.20).round() as usize;
        let iy_hb = spec.axis_y.ppm_to_index(2.10).round() as usize;
        let pre_hb = spec.get(ix_hb, iy_hb);
        let used_w = suppress_water_spectrum_2d(&mut spec, &WaterSuppressionConfig::default());
        assert!((used_w - 4.70).abs() < 0.03, "used_w={used_w}");

        let ix_w = spec.axis_x.ppm_to_index(4.70).round() as usize;
        let iy_w = spec.axis_y.ppm_to_index(4.70).round() as usize;
        assert!(
            spec.get(ix_w, iy_w).abs() < 5.0,
            "diagonal water volcano (5000) must be wiped out, got {}",
            spec.get(ix_w, iy_w)
        );

        // Ridge background at (F2 = 4.72, F1 = 6.50) must be wiped out (< 3.0 from +80.0)
        let ix_r = spec.axis_x.ppm_to_index(4.72).round() as usize;
        let iy_r = spec.axis_y.ppm_to_index(6.50).round() as usize;
        assert!(
            spec.get(ix_r, iy_r).abs() < 3.0,
            "vertical t1-noise ridge must be suppressed, got {}",
            spec.get(ix_r, iy_r)
        );

        // Real narrow cross-peaks at (4.80, 8.20) and (8.20, 2.10) must be preserved!
        let ix_ha = spec.axis_x.ppm_to_index(4.80).round() as usize;
        let iy_ha = spec.axis_y.ppm_to_index(8.20).round() as usize;
        assert!(
            spec.get(ix_ha, iy_ha) > 22.0,
            "narrow H-alpha cross-peak riding near water ridge must be preserved, got {}",
            spec.get(ix_ha, iy_ha)
        );
        assert!(
            (spec.get(ix_hb, iy_hb) - pre_hb).abs() < 0.25,
            "off-water cross-peak must be untouched: pre={pre_hb}, post={}",
            spec.get(ix_hb, iy_hb)
        );
    }

    #[test]
    fn test_suppress_water_3d_removes_f3_water_slab_and_preserves_amide_peaks() {
        let nx = 256usize;
        let ny = 16usize;
        let nz = 16usize;
        let axis_x = AxisPpm::from_bounds("1H", nx, 10.5, 2.5, 600.0);
        let axis_y = AxisPpm::from_bounds("15N", ny, 130.0, 105.0, 60.8);
        let axis_z = AxisPpm::from_bounds("13C", nz, 68.0, 42.0, 150.9);
        let mut data = vec![0.0_f32; nz * ny * nx];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let x_ppm = axis_x.index_to_ppm(ix as f64);
                    let dw = (x_ppm - 4.70) / 0.03;
                    let mut v = 800.0 / (1.0 + dw * dw) + 30.0 * (-0.5 * dw * dw).exp();
                    if iy == 8 && iz == 8 {
                        let da = (x_ppm - 8.35) / 0.025;
                        v += 50.0 * (-0.5 * da * da).exp();
                    }
                    data[(iz * ny + iy) * nx + ix] = v as f32;
                }
            }
        }

        let mut cube = Spectrum3D {
            name: "synthetic_3d".to_string(),
            exp: "HNCA".to_string(),
            axis_x,
            axis_y,
            axis_z,
            data,
        };
        let ix_w = cube.axis_x.ppm_to_index(4.70).round() as usize;
        let ix_hn = cube.axis_x.ppm_to_index(8.35).round() as usize;
        let pre_hn_amp = cube.get(ix_hn, 8, 8) - cube.get(ix_hn, 4, 4);
        let used_w = suppress_water_spectrum_3d(&mut cube, &WaterSuppressionConfig::default());
        assert!((used_w - 4.70).abs() < 0.04, "used_w={used_w}");

        assert!(cube.get(ix_w, 4, 4).abs() < 2.0);
        assert!(
            cube.get(ix_hn, 8, 8) > 0.98 * pre_hn_amp,
            "3D amide peak must retain >98% amplitude: pre={pre_hn_amp}, post={}",
            cube.get(ix_hn, 8, 8)
        );
    }

    #[test]
    fn test_suppress_water_3d_alias_preserves_indirect_1h_strip_peaks_in_hccconh() {
        let nx = 128usize;
        let ny = 12usize;
        let nz = 32usize;
        let axis_x = AxisPpm::from_bounds("1H", nx, 10.7, 0.7, 800.13);
        let axis_y = AxisPpm::from_bounds("15N", ny, 130.0, 108.0, 81.08);
        let axis_z = AxisPpm::from_bounds("1H", nz, 9.5, -0.5, 800.13);
        let mut data = vec![0.0_f32; nz * ny * nx];

        for iz in 0..nz {
            let z_ppm = axis_z.index_to_ppm(iz as f64);
            for iy in 0..ny {
                for ix in 0..nx {
                    let x_ppm = axis_x.index_to_ppm(ix as f64);
                    let dw = (x_ppm - 4.70) / 0.03;
                    let mut v = 1200.0 / (1.0 + dw * dw);
                    if iy == 5 {
                        let dx = (x_ppm - 8.28) / 0.035;
                        let dz_ali = (z_ppm - 1.85) / 0.18;
                        let dz_ha = (z_ppm - 4.70) / 0.18;
                        v += 65.0 * (-0.5 * (dx * dx + dz_ali * dz_ali)).exp();
                        v += 48.0 * (-0.5 * (dx * dx + dz_ha * dz_ha)).exp();
                    }
                    data[(iz * ny + iy) * nx + ix] = v as f32;
                }
            }
        }

        let mut cube = Spectrum3D {
            name: "hccconh_test".to_string(),
            exp: "HCCCONH".to_string(),
            axis_x,
            axis_y,
            axis_z,
            data,
        };
        let ix_w = cube.axis_x.ppm_to_index(4.70).round() as usize;
        let ix_hn = cube.axis_x.ppm_to_index(8.28).round() as usize;
        let iz_ali = cube.axis_z.ppm_to_index(1.85).round() as usize;
        let iz_ha = cube.axis_z.ppm_to_index(4.70).round() as usize;
        let pre_ali = cube.get(ix_hn, 5, iz_ali);
        let pre_ha = cube.get(ix_hn, 5, iz_ha);

        let w_ppm = suppress_water_3d(&mut cube, &WaterSuppressionConfig::default());
        assert!((w_ppm - 4.70).abs() < 0.05);
        assert!(cube.get(ix_w, 5, iz_ali).abs() < 5.0);
        assert!(cube.get(ix_hn, 5, iz_ali) > 0.96 * pre_ali);
        assert!(cube.get(ix_hn, 5, iz_ha) > 0.96 * pre_ha);
    }

    #[test]
    fn test_v0150_p1_5_quantitative_solute_retention_2d_and_1d() {
        // 1. 2D Homonuclear 1H-1H spectrum with NO water peak present and cross-peaks at
        //    (8.30, 4.50), (8.30, 4.60), (8.30, 4.65), and (4.65, 2.00).
        let nx = 512usize;
        let ny = 256usize;
        let axis_x = AxisPpm::from_bounds("1H", nx, 10.0, 0.0, 500.0);
        let axis_y = AxisPpm::from_bounds("1H", ny, 10.0, 0.0, 500.0);

        let probes = [
            (8.30_f64, 4.50_f64, 0.01_f64, "(8.30, 4.50)"),
            (8.30_f64, 4.60_f64, 0.02_f64, "(8.30, 4.60)"),
            (8.30_f64, 4.65_f64, 0.02_f64, "(8.30, 4.65)"),
            (4.65_f64, 2.00_f64, 0.02_f64, "(4.65, 2.00)"),
        ];

        for &(px, py, max_loss, label) in &probes {
            let mut data = vec![0.0_f32; ny * nx];
            for iy in 0..ny {
                let y_ppm = axis_y.index_to_ppm(iy as f64);
                for ix in 0..nx {
                    let x_ppm = axis_x.index_to_ppm(ix as f64);
                    let val = 100.0
                        * (-0.5 * ((x_ppm - px) / 0.020).powi(2)
                            - 0.5 * ((y_ppm - py) / 0.045).powi(2))
                        .exp();
                    data[iy * nx + ix] = val as f32;
                }
            }
            let mut spec = Spectrum2D {
                name: format!("dry_probe_{label}"),
                axis_x: axis_x.clone(),
                axis_y: axis_y.clone(),
                data,
            };
            let ix0 = spec.axis_x.ppm_to_index(px).round() as usize;
            let iy0 = spec.axis_y.ppm_to_index(py).round() as usize;
            let pre_amp = spec.get(ix0, iy0) as f64;
            let _ = suppress_water_spectrum_2d(&mut spec, &WaterSuppressionConfig::default());
            let post_amp = spec.get(ix0, iy0) as f64;
            let loss = (pre_amp - post_amp) / pre_amp;
            assert!(
                loss.abs() <= max_loss,
                "2D cross-peak {label} lost {:.2}% amplitude (pre={pre_amp:.3}, post={post_amp:.3}, max allowed={:.1}%)",
                loss * 100.0,
                max_loss * 100.0
            );
        }

        // 2a. Pure 1D filter response on dry solute peaks at ±0.20, ±0.10, and ±0.05 ppm from 4.700 ppm
        for &(offset_ppm, max_loss) in &[
            (0.20_f64, 0.005_f64),
            (0.10_f64, 0.01_f64),
            (0.05_f64, 0.02_f64),
        ] {
            for sign in [-1.0_f64, 1.0_f64] {
                let solute_ppm = 4.700 + sign * offset_ppm;
                let lines_dry = [SyntheticLine1D::lorentzian(solute_ppm, 1.0, 2.5)];
                let mut td_dry = synthesize_fid_1d(
                    "dry_1d",
                    Experiment1DKind::Proton1D,
                    4096,
                    7200.0,
                    600.0,
                    4.70,
                    &lines_dry,
                );
                td_dry.data.resize(8192, [0.0, 0.0]);
                let spec_ref = fft_1d(&td_dry);
                let mut spec_dry = spec_ref.clone();
                spec_dry.imag = None;

                let idx_s = spec_dry.axis.ppm_to_index(solute_ppm).round() as usize;
                let idx_base = spec_dry.axis.ppm_to_index(1.000).round() as usize;
                let ref_amp = spec_ref.real[idx_s] - spec_ref.real[idx_base];
                let cfg_explicit = WaterSuppressionConfig {
                    water_ppm: Some(4.700),
                    ..WaterSuppressionConfig::default()
                };
                let _ = suppress_water_spectrum_1d(&mut spec_dry, &cfg_explicit);
                let post_amp = spec_dry.real[idx_s] - spec_dry.real[idx_base];
                let loss = (ref_amp - post_amp) / ref_amp;
                assert!(
                    loss.abs() <= max_loss,
                    "dry 1D solute peak at {solute_ppm:.3} ppm ({:+.2} ppm) lost {:.2}% (max {:.1}%)",
                    sign * offset_ppm,
                    loss * 100.0,
                    max_loss * 100.0
                );
            }
        }

        // 2b. 1D solute peaks at ±0.20, ±0.10, and ±0.05 ppm superposed on 200x residual water
        let water_line = SyntheticLine1D {
            ppm: 4.700,
            amplitude: 200.0,
            fwhm_hz: 0.5,
            gauss_fwhm_hz: 6.0,
            phase_deg: 0.0,
        };
        let mut td_water_only = synthesize_fid_1d(
            "water_only_1d",
            Experiment1DKind::Proton1D,
            4096,
            7200.0,
            600.0,
            4.70,
            std::slice::from_ref(&water_line),
        );
        td_water_only.data.resize(8192, [0.0, 0.0]);
        let mut spec_water_only = fft_1d(&td_water_only);
        spec_water_only.imag = None;
        let _ =
            suppress_water_spectrum_1d(&mut spec_water_only, &WaterSuppressionConfig::default());

        for &(offset_ppm, max_loss) in &[
            (0.20_f64, 0.005_f64),
            (0.10_f64, 0.01_f64),
            (0.05_f64, 0.02_f64),
        ] {
            for sign in [-1.0_f64, 1.0_f64] {
                let solute_ppm = 4.700 + sign * offset_ppm;
                let lines_ref = [SyntheticLine1D::lorentzian(solute_ppm, 1.0, 2.5)];
                let mut td_ref = synthesize_fid_1d(
                    "ref_1d",
                    Experiment1DKind::Proton1D,
                    4096,
                    7200.0,
                    600.0,
                    4.70,
                    &lines_ref,
                );
                td_ref.data.resize(8192, [0.0, 0.0]);
                let spec_ref = fft_1d(&td_ref);

                let lines_wet = [
                    water_line,
                    SyntheticLine1D::lorentzian(solute_ppm, 1.0, 2.5),
                ];
                let mut td_wet = synthesize_fid_1d(
                    "wet_1d",
                    Experiment1DKind::Proton1D,
                    4096,
                    7200.0,
                    600.0,
                    4.70,
                    &lines_wet,
                );
                td_wet.data.resize(8192, [0.0, 0.0]);
                let mut spec_wet = fft_1d(&td_wet);
                spec_wet.imag = None;

                let idx_s = spec_wet.axis.ppm_to_index(solute_ppm).round() as usize;
                let idx_base = spec_wet.axis.ppm_to_index(1.000).round() as usize;
                let _ =
                    suppress_water_spectrum_1d(&mut spec_wet, &WaterSuppressionConfig::default());
                let ref_amp = spec_ref.real[idx_s] - spec_ref.real[idx_base];
                let post_solute = (spec_wet.real[idx_s] - spec_water_only.real[idx_s])
                    - (spec_wet.real[idx_base] - spec_water_only.real[idx_base]);
                let loss = (ref_amp - post_solute) / ref_amp;
                assert!(
                    loss.abs() <= max_loss,
                    "1D solute peak at {solute_ppm:.3} ppm ({:+.2} ppm from water) lost {:.2}% (max allowed {:.1}%)",
                    sign * offset_ppm,
                    loss * 100.0,
                    max_loss * 100.0
                );
            }
        }

        // 3. Also verify 2D cross-peak retention at (8.30, 4.50) (< 1% loss) and (8.30, 4.60) (< 2% loss)
        //    when a 2000x water volcano + F2 ridge IS present at 4.70 ppm (both default and opt-in F1)!
        for opt_in_f1 in [false, true] {
            for &(py, max_loss) in &[(4.50_f64, 0.01_f64), (4.60_f64, 0.02_f64)] {
                let mut data_ref = vec![0.0_f32; ny * nx];
                let mut data_wet = vec![0.0_f32; ny * nx];
                for iy in 0..ny {
                    let y_ppm = axis_y.index_to_ppm(iy as f64);
                    for ix in 0..nx {
                        let x_ppm = axis_x.index_to_ppm(ix as f64);
                        let pk = 100.0
                            * (-0.5 * ((x_ppm - 8.30) / 0.020).powi(2)
                                - 0.5 * ((y_ppm - py) / 0.035).powi(2))
                            .exp();
                        let dx_w = (x_ppm - 4.70) / 0.020;
                        let dy_w = (y_ppm - 4.70) / 0.030;
                        let water = 2000.0 / ((1.0 + dx_w * dx_w) * (1.0 + dy_w * dy_w))
                            + 40.0 * (-0.5 * ((x_ppm - 4.70) / 0.03).powi(2)).exp();
                        data_ref[iy * nx + ix] = pk as f32;
                        data_wet[iy * nx + ix] = (pk + water) as f32;
                    }
                }
                let spec_2d_ref = Spectrum2D {
                    name: "ref_2d".into(),
                    axis_x: axis_x.clone(),
                    axis_y: axis_y.clone(),
                    data: data_ref,
                };
                let mut spec_2d_wet = Spectrum2D {
                    name: "wet_2d".into(),
                    axis_x: axis_x.clone(),
                    axis_y: axis_y.clone(),
                    data: data_wet,
                };
                let ix0 = spec_2d_ref.axis_x.ppm_to_index(8.30).round() as usize;
                let iy0 = spec_2d_ref.axis_y.ppm_to_index(py).round() as usize;
                let pre = spec_2d_ref.get(ix0, iy0) as f64;
                let cfg_2d = WaterSuppressionConfig::default().with_suppress_indirect_f1(opt_in_f1);
                let _ = suppress_water_spectrum_2d(&mut spec_2d_wet, &cfg_2d);
                let post = spec_2d_wet.get(ix0, iy0) as f64;
                let loss = (pre - post) / pre;
                assert!(
                    loss.abs() <= max_loss,
                    "wet 2D cross-peak (8.30, {py:.2}) with opt_in_f1={opt_in_f1} lost {:.2}% (max {:.1}%)",
                    loss * 100.0,
                    max_loss * 100.0
                );
            }
        }
    }
}
