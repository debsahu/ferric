//! Classical baseline 2D and 3D NMR peak picker with real-spectrum artifact hygiene
//! and two-pass experiment-corroborated 3D strip rescue.
//!
//! Provenance: Original clean-room implementation of `docs/specs/02-peak-picking.md` Stages A & A2:
//! - Pass 1 (Stage A): two-pass outlier-trimmed MAD noise estimation (`1.4826 * median(|x - median(x)|)`),
//!   amide `1H` and experiment-specific `13C` spectral window gating, local indirect-axis column/strip
//!   MAD `t1`-ridge thresholding and FT truncation sinc-sidelobe suppression, automatic `180°` phase
//!   polarity detection for single-sign 3D triple-resonance spectra, resolved `^3J_HNHa` `1H` doublet
//!   coalescing, local extrema detection, and 3-point log-Gaussian / parabolic sub-voxel refinement.
//! - Pass 2 (Stage A2, `StripBank3D` + `corroborated_re_pick_3d`): original clean-room two-pass
//!   experiment-guided corroborated re-picking in compact 3D `(H, N)` halo strip cubes, exploiting
//!   complementary triple-resonance coherence-transfer pathways (`HNCA`/`HN(CO)CA`,
//!   `HNCACB`/`CBCA(CO)NH`, `HNCO`/`HN(CA)CO`; Kay, Ikura, Tschudin & Bax, *J. Magn. Reson.* 1990,
//!   89, 496–514; Grzesiek & Bax, *J. Am. Chem. Soc.* 1992, 114, 6291–6293) strictly without
//!   sequence or chemical-shift prior inputs.

use std::collections::BTreeMap;

use crate::peaks::Peak;
use crate::spectrum::{AxisPpm, Spectrum2D, Spectrum3D};

#[derive(Clone, Debug)]
pub struct PickerConfig {
    /// Signal-to-noise threshold multiplier (`k * noise`, default `5.0`).
    pub snr_threshold: f32,
    /// Allow negative peaks (required for phase-sensitive `HNCACB`).
    pub allow_negative: bool,
    /// Strip half-widths `(H_ppm, N_ppm)` when restricting 3D picking under HSQC roots.
    pub strip_tol_h: f64,
    pub strip_tol_n: f64,
    /// Amide `1H` window `(min_ppm, max_ppm)` applied to `15N-HSQC` when the `1H` axis spans wide
    /// sweep widths (`min_ppm < 4.5`) or when customized (`None` disables gating, e.g. `13C-HSQC`).
    pub amide_h_window: Option<(f64, f64)>,
    /// Multiplier on local indirect-axis column/strip MAD for `t1` noise ridge suppression (default `5.5`).
    pub ridge_snr: f32,
    /// Minimum fraction of column/strip parent amplitude for FT truncation sinc suppression (default `0.025`).
    pub sinc_frac: f32,
    /// Coalesce resolved `^3J_HNHa` `1H` doublets (`|dH| * obs_mhz <= max_j_hnha_hz`, `|dN| <= 0.08 ppm`)
    /// in high-digital-resolution `15N-HSQC` spectra (`|step_ppm| * obs_mhz <= 6.5 Hz`, default `true`).
    pub coalesce_j_doublets: bool,
    /// Maximum `^3J_HNHa` scalar splitting in Hz for `15N-HSQC` doublet coalescing (default `9.5` Hz),
    /// converted to ppm via the `1H` axis spectrometer frequency `obs_mhz`.
    pub max_j_hnha_hz: f64,
}

impl Default for PickerConfig {
    fn default() -> Self {
        Self {
            snr_threshold: 5.0,
            allow_negative: false,
            strip_tol_h: 0.035,
            strip_tol_n: 0.35,
            amide_h_window: Some((5.5, 11.5)),
            ridge_snr: 5.5,
            sinc_frac: 0.025,
            coalesce_j_doublets: true,
            max_j_hnha_hz: 9.5,
        }
    }
}

fn median_f32(vals: &mut [f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mid = vals.len() / 2;
    let (_, m, _) = vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    *m
}

/// Robust MAD noise estimate (`1.4826 * median(|x - median(x)|)`) after trimming signal outliers.
pub fn estimate_noise_mad(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 1.0;
    }
    // Stride large 3D arrays for fast deterministic O(N/stride) noise estimation
    let stride = (data.len() / 65_536).max(1);
    let mut sample: Vec<f32> = data.iter().step_by(stride).copied().collect();
    let med0 = median_f32(&mut sample);

    let mut dev0: Vec<f32> = sample.iter().map(|&v| (v - med0).abs()).collect();
    let mad0 = (1.4826 * median_f32(&mut dev0)).max(1e-12);

    // Exclude peak regions (> 3.5 sigma) and recompute MAD on baseline noise
    let mut bg: Vec<f32> = sample
        .into_iter()
        .filter(|&v| (v - med0).abs() <= 3.5 * mad0)
        .collect();
    if bg.len() < 16 {
        return mad0;
    }
    let med1 = median_f32(&mut bg);
    let mut dev1: Vec<f32> = bg.into_iter().map(|v| (v - med1).abs()).collect();
    (1.4826 * median_f32(&mut dev1)).max(1e-12)
}

/// Compute two-pass outlier-trimmed baseline median, local MAD (floored at `fallback_mad`), and `max_abs`.
fn slice_trimmed_median_mad_and_max(
    vals: &[f32],
    scratch: &mut Vec<f32>,
    fallback_mad: f32,
) -> (f32, f32, f32) {
    let mut max_abs = 0.0f32;
    for &v in vals {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if vals.len() < 8 {
        return (0.0, fallback_mad, max_abs);
    }
    scratch.clear();
    scratch.extend_from_slice(vals);
    let med0 = median_f32(scratch);
    for x in scratch.iter_mut() {
        *x = (*x - med0).abs();
    }
    let mad0 = (1.4826 * median_f32(scratch)).max(1e-12);
    if vals.len() < 24 {
        return (med0, fallback_mad, max_abs);
    }

    scratch.clear();
    scratch.extend(
        vals.iter()
            .copied()
            .filter(|&v| (v - med0).abs() <= 3.5 * mad0),
    );
    if scratch.len() < 12 {
        return (med0, mad0.max(fallback_mad), max_abs);
    }
    let med1 = median_f32(scratch);
    for x in scratch.iter_mut() {
        *x = (*x - med1).abs();
    }
    let mad1 = (1.4826 * median_f32(scratch)).max(fallback_mad);
    (med1, mad1, max_abs)
}

/// Compute two-pass outlier-trimmed local MAD (floored at `fallback_mad`) and `max_abs` for a 1D slice.
fn slice_trimmed_mad_and_max(
    vals: &[f32],
    scratch: &mut Vec<f32>,
    fallback_mad: f32,
) -> (f32, f32) {
    let (_, mad, max_abs) = slice_trimmed_median_mad_and_max(vals, scratch, fallback_mad);
    (mad, max_abs)
}

/// Sub-voxel 3-point peak offset in `[-0.5, 0.5]` using log-Gaussian fit (with parabolic fallback).
#[inline]
fn subvoxel_offset(ym1: f32, y0: f32, yp1: f32) -> f64 {
    let a = (ym1.abs()) as f64;
    let b = (y0.abs()) as f64;
    let c = (yp1.abs()) as f64;
    if a > 0.0
        && b > a
        && b > c
        && c > 0.0
        && ym1.signum() == y0.signum()
        && yp1.signum() == y0.signum()
    {
        let la = a.ln();
        let lb = b.ln();
        let lc = c.ln();
        let denom = la - 2.0 * lb + lc;
        if denom.abs() > 1e-12 {
            return (0.5 * (la - lc) / denom).clamp(-0.5, 0.5);
        }
    }
    let denom = a - 2.0 * b + c;
    if denom.abs() > 1e-12 {
        (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

#[inline]
fn active_amide_window(axis_min_ppm: f64, cfg: &PickerConfig) -> Option<(f64, f64)> {
    let (w0, w1) = cfg.amide_h_window?;
    let (lo, hi) = (w0.min(w1), w0.max(w1));
    let custom = (lo - 5.5).abs() > 1e-6 || (hi - 11.5).abs() > 1e-6;
    if axis_min_ppm < 4.5 || custom {
        Some((lo, hi))
    } else {
        None
    }
}

#[inline]
fn carbon_bounds_for_exp(exp: &str) -> Option<(f64, f64)> {
    match exp {
        "HNCACB" | "CBCACONH" | "HNCA" | "HNCOCA" => Some((8.0, 76.0)),
        "HNCO" | "HNCACO" => Some((164.0, 187.0)),
        _ => None,
    }
}

/// Coalesce symmetric `^3J_HNHa` `1H` doublets (`|dN| <= 0.08 ppm`, `|dH| <= max_j_ppm`,
/// `min_ht >= 0.40 * max_ht`) in high-resolution `15N-HSQC` spectra into intensity-weighted centroids,
/// where `max_j_ppm = max_j_hnha_hz / obs_mhz`.
fn coalesce_hsqc_j_doublets(mut peaks: Vec<Peak>, max_j_ppm: f64) -> Vec<Peak> {
    if peaks.len() < 2 || max_j_ppm <= 0.0 {
        return peaks;
    }
    peaks.sort_by(|a, b| b.height.unwrap_or(0.0).total_cmp(&a.height.unwrap_or(0.0)));
    let n = peaks.len();
    let mut merged = vec![false; n];
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        if merged[i] {
            continue;
        }
        let pi = &peaks[i];
        let hi = pi.height.unwrap_or(0.0);
        let mut best_j: Option<usize> = None;
        let mut best_dh = f64::INFINITY;
        for j in (i + 1)..n {
            if merged[j] || peaks[j].sign != pi.sign {
                continue;
            }
            let pj = &peaks[j];
            let hj = pj.height.unwrap_or(0.0);
            let dn = (pi.n - pj.n).abs();
            let dh = (pi.h - pj.h).abs();
            if dn <= 0.08 && dh <= max_j_ppm && hj >= 0.40 * hi && dh < best_dh {
                best_dh = dh;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            merged[j] = true;
            let pj = &peaks[j];
            let hj = pj.height.unwrap_or(0.0);
            let w_sum = (hi + hj).max(1e-12);
            let h_cen = (hi * pi.h + hj * pj.h) / w_sum;
            let n_cen = (hi * pi.n + hj * pj.n) / w_sum;
            let vol = match (pi.volume, pj.volume) {
                (Some(va), Some(vb)) => Some(va + vb),
                (a, b) => a.or(b),
            };
            out.push(Peak {
                exp: pi.exp.clone(),
                h: h_cen,
                n: n_cen,
                c: pi.c,
                sign: pi.sign,
                height: Some(hi + hj),
                volume: vol,
                h2: pi.h2,
            });
        } else {
            out.push(pi.clone());
        }
    }
    out
}

/// Pick 2D `HSQC` root peaks from a processed 2D spectrum.
pub fn pick_hsqc_2d(spec: &Spectrum2D, cfg: &PickerConfig) -> Vec<Peak> {
    let nx = spec.nx();
    let ny = spec.ny();
    if nx < 3 || ny < 3 {
        return Vec::new();
    }
    let noise = estimate_noise_mad(&spec.data);
    let thresh = cfg.snr_threshold * noise;
    let h_gate = active_amide_window(spec.axis_x.min_ppm(), cfg);

    // Identify active 1H columns within the optional amide window (plus 1-point margin for subvoxel fit)
    let step_h = spec.axis_x.step_ppm.abs();
    let mut active_x = vec![false; nx];
    let mut active_ix_list = Vec::with_capacity(nx);
    let mut spec_max_abs = 0.0f32;
    for (ix, slot) in active_x.iter_mut().enumerate().take(nx - 1).skip(1) {
        if let Some((lo, hi)) = h_gate {
            let hp = spec.axis_x.index_to_ppm(ix as f64);
            if hp < lo - step_h || hp > hi + step_h {
                continue;
            }
        }
        *slot = true;
        active_ix_list.push(ix);
    }
    for iy in 0..ny {
        for &ix in &active_ix_list {
            let a = spec.get(ix, iy).abs();
            if a > spec_max_abs {
                spec_max_abs = a;
            }
        }
    }

    // Activate local ridge MAD and FT truncation sinc suppression in high-dynamic-range spectra
    let high_dr = spec_max_abs > 40.0 * noise;
    let mut col_thresh = vec![thresh; nx];
    let mut row_thresh = vec![thresh; ny];
    if high_dr {
        let mut slice_buf = Vec::with_capacity(nx.max(ny));
        let mut scratch = Vec::with_capacity(nx.max(ny));
        for &ix in &active_ix_list {
            slice_buf.clear();
            for iy in 0..ny {
                slice_buf.push(spec.get(ix, iy));
            }
            let (cmad, cmax) = slice_trimmed_mad_and_max(&slice_buf, &mut scratch, noise);
            col_thresh[ix] = thresh.max(cfg.ridge_snr * cmad).max(cfg.sinc_frac * cmax);
        }
        for (iy, rt) in row_thresh.iter_mut().enumerate().take(ny - 1).skip(1) {
            slice_buf.clear();
            for &ix in &active_ix_list {
                slice_buf.push(spec.get(ix, iy));
            }
            let (rmad, _) = slice_trimmed_mad_and_max(&slice_buf, &mut scratch, noise);
            *rt = thresh.max(cfg.ridge_snr * rmad);
        }
    }

    let mut peaks = Vec::new();

    for (iy, &r_thr) in row_thresh.iter().enumerate().take(ny - 1).skip(1) {
        for &ix in &active_ix_list {
            let v = spec.get(ix, iy);
            let loc_thr = col_thresh[ix].max(r_thr);
            if v.abs() < loc_thr || (!cfg.allow_negative && v < 0.0) {
                continue;
            }
            let sign = if v >= 0.0 { 1 } else { -1 };
            let mut is_extremum = true;
            'nbr: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nv = spec.get((ix as isize + dx) as usize, (iy as isize + dy) as usize);
                    if (sign > 0 && nv >= v) || (sign < 0 && nv <= v) {
                        is_extremum = false;
                        break 'nbr;
                    }
                }
            }
            if !is_extremum {
                continue;
            }
            let dx = subvoxel_offset(spec.get(ix - 1, iy), v, spec.get(ix + 1, iy));
            let dy = subvoxel_offset(spec.get(ix, iy - 1), v, spec.get(ix, iy + 1));
            let h = spec.axis_x.index_to_ppm(ix as f64 + dx);
            if let Some((lo, hi)) = h_gate {
                if h < lo || h > hi {
                    continue;
                }
            }
            let n = spec.axis_y.index_to_ppm(iy as f64 + dy);
            let mut vol = 0.0f64;
            for wy in -1isize..=1 {
                for wx in -1isize..=1 {
                    let sv = spec.get((ix as isize + wx) as usize, (iy as isize + wy) as usize);
                    if sv.signum() == v.signum() {
                        vol += (sv.abs()) as f64;
                    }
                }
            }
            peaks.push(Peak {
                exp: "HSQC".to_string(),
                h,
                n,
                c: None,
                sign,
                height: Some((v.abs()) as f64),
                volume: Some(vol),
                h2: None,
            });
        }
    }

    let obs_mhz = if spec.axis_x.obs_mhz > 10.0 {
        spec.axis_x.obs_mhz
    } else {
        600.0
    };
    let max_j_ppm = cfg.max_j_hnha_hz.max(0.0) / obs_mhz;
    let min_f2_sidelobe_dh = (12.0 / obs_mhz).max(if cfg.coalesce_j_doublets {
        max_j_ppm
    } else {
        0.0
    });

    // Suppress near-parent F1 (indirect) and F2 (direct) truncation sinc sidelobes
    if high_dr && peaks.len() > 1 {
        let min_parent = (40.0 * noise) as f64;
        let base_sinc = cfg.sinc_frac as f64;
        let mut keep = vec![true; peaks.len()];
        for i in 0..peaks.len() {
            let pi = &peaks[i];
            let hti = pi.height.unwrap_or(0.0);
            for (j, pj) in peaks.iter().enumerate() {
                if i == j || pj.sign != pi.sign {
                    continue;
                }
                let htj = pj.height.unwrap_or(0.0);
                if htj <= hti || htj < min_parent {
                    continue;
                }
                let dh = (pi.h - pj.h).abs();
                let dn = (pi.n - pj.n).abs();
                // F1 indirect sidelobe along the same 1H column
                if dh <= 0.025 && dn > 0.08 {
                    let env = base_sinc + 0.16 / (1.0 + (dn / 0.65).powi(2));
                    if hti < htj * env {
                        keep[i] = false;
                        break;
                    }
                }
                // F2 direct sidelobe along the same indirect row (above ^3J_HNHa / 10 Hz in ppm)
                if dn <= 0.25 && dh > min_f2_sidelobe_dh && dh <= 0.35 {
                    let env = 0.015 + 0.12 / (1.0 + (dh / 0.06).powi(2));
                    if hti < htj * env {
                        keep[i] = false;
                        break;
                    }
                }
            }
        }
        let mut filtered = Vec::with_capacity(peaks.len());
        for (p, k) in peaks.into_iter().zip(keep) {
            if k {
                filtered.push(p);
            }
        }
        peaks = filtered;
    }

    // Require digital resolution <= 6.5 Hz/pt (e.g. SI=2048 over 14 ppm up to 950 MHz)
    // to resolve ^3J_HNHa doublets without merging coarse-grid peaks
    if cfg.coalesce_j_doublets
        && cfg.max_j_hnha_hz > 0.0
        && cfg.amide_h_window.is_some()
        && step_h * obs_mhz <= 6.5
    {
        peaks = coalesce_hsqc_j_doublets(peaks, max_j_ppm);
    }

    peaks
}

/// Pick 2D `C13HSQC` peaks `(1H, 13C)` from an aliphatic or aromatic `13C`-HSQC spectrum.
pub fn pick_c13_hsqc_2d(spec: &Spectrum2D, cfg: &PickerConfig) -> Vec<Peak> {
    let mut c13_cfg = cfg.clone();
    c13_cfg.amide_h_window = None;
    c13_cfg.coalesce_j_doublets = false;
    let mut base = pick_hsqc_2d(spec, &c13_cfg);
    for p in &mut base {
        p.exp = "C13HSQC".to_string();
        p.c = Some(p.n);
        p.n = 0.0;
    }
    base
}

/// Pick 3D triple-resonance peaks restricted to `(H, N)` strips around `hsqc_roots`.
pub fn pick_triple_3d(spec: &Spectrum3D, hsqc_roots: &[Peak], cfg: &PickerConfig) -> Vec<Peak> {
    let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
    if nx < 3 || ny < 3 || nz < 3 {
        return Vec::new();
    }
    let noise = estimate_noise_mad(&spec.data);
    let thresh = cfg.snr_threshold * noise;
    let allow_neg = cfg.allow_negative || spec.exp == "HNCACB";
    let c_bounds = carbon_bounds_for_exp(&spec.exp);
    let h_gate = if hsqc_roots.is_empty() && c_bounds.is_some() {
        active_amide_window(spec.axis_x.min_ppm(), cfg)
    } else {
        None
    };

    let y_is_n = spec.axis_y.label.contains('N')
        || (spec.axis_y.max_ppm() > 95.0
            && spec.axis_y.min_ppm() < 145.0
            && !spec.axis_z.label.contains('N'));

    let (n_n, n_c) = if y_is_n { (ny, nz) } else { (nz, ny) };
    let axis_n = if y_is_n { &spec.axis_y } else { &spec.axis_z };
    let axis_c = if y_is_n { &spec.axis_z } else { &spec.axis_y };

    let step_c = axis_c.step_ppm.abs();
    let mut active_c = Vec::with_capacity(n_c);
    for ic in 1..n_c - 1 {
        let cp = axis_c.index_to_ppm(ic as f64);
        if let Some((clo, chi)) = c_bounds {
            if cp < clo - step_c || cp > chi + step_c {
                continue;
            }
        }
        active_c.push(ic);
    }
    if active_c.is_empty() {
        return Vec::new();
    }

    // Preselect active (ix, in_idx) strip columns matching hsqc_roots (or h_gate when rootless)
    let step_h = spec.axis_x.step_ppm.abs();
    let mut active_strips: Vec<(usize, usize)> = Vec::new();
    for in_idx in 1..n_n - 1 {
        let n_ppm_grid = axis_n.index_to_ppm(in_idx as f64);
        if !hsqc_roots.is_empty()
            && !hsqc_roots
                .iter()
                .any(|r| (r.n - n_ppm_grid).abs() <= cfg.strip_tol_n)
        {
            continue;
        }
        for ix in 1..nx - 1 {
            let h_ppm_grid = spec.axis_x.index_to_ppm(ix as f64);
            if let Some((lo, hi)) = h_gate {
                if h_ppm_grid < lo - step_h || h_ppm_grid > hi + step_h {
                    continue;
                }
            }
            if !hsqc_roots.is_empty()
                && !hsqc_roots.iter().any(|r| {
                    (r.h - h_ppm_grid).abs() <= cfg.strip_tol_h
                        && (r.n - n_ppm_grid).abs() <= cfg.strip_tol_n
                })
            {
                continue;
            }
            active_strips.push((ix, in_idx));
        }
    }
    if active_strips.is_empty() {
        return Vec::new();
    }

    let get_voxel = |ix: usize, in_idx: usize, ic: usize| -> f32 {
        if y_is_n {
            spec.get(ix, in_idx, ic)
        } else {
            spec.get(ix, ic, in_idx)
        }
    };

    // Scan active strips to measure peak dynamic range and detect 180-degree inverted single-sign phase
    let mut max_pos = 0.0f32;
    let mut max_neg = 0.0f32;
    for &(ix, in_idx) in &active_strips {
        for &ic in &active_c {
            let v = get_voxel(ix, in_idx, ic);
            if v > max_pos {
                max_pos = v;
            } else if -v > max_neg {
                max_neg = -v;
            }
        }
    }
    let single_sign_backbone = !allow_neg
        && matches!(
            spec.exp.as_str(),
            "HNCA" | "HNCOCA" | "CBCACONH" | "HNCO" | "HNCACO"
        );
    let invert_polarity = single_sign_backbone && max_neg > 2.5 * max_pos.max(thresh);
    let spec_max_abs = if invert_polarity {
        max_neg
    } else if allow_neg {
        max_pos.max(max_neg)
    } else {
        max_pos
    };
    let high_dr = spec_max_abs > 40.0 * noise;
    let far_sinc_frac = (cfg.sinc_frac * 0.4).min(0.012);

    let mut strip_buf = Vec::with_capacity(n_c);
    let mut scratch = Vec::with_capacity(n_c);
    let mut out = Vec::new();

    for &(ix, in_idx) in &active_strips {
        let loc_thr = if high_dr {
            strip_buf.clear();
            for ic in 0..n_c {
                strip_buf.push(get_voxel(ix, in_idx, ic));
            }
            let (smad, smax) = slice_trimmed_mad_and_max(&strip_buf, &mut scratch, noise);
            thresh.max(cfg.ridge_snr * smad).max(far_sinc_frac * smax)
        } else {
            thresh
        };

        for &ic in &active_c {
            let (iy, iz) = if y_is_n { (in_idx, ic) } else { (ic, in_idx) };
            let raw_v = spec.get(ix, iy, iz);
            let eff_v = if invert_polarity { -raw_v } else { raw_v };
            if eff_v.abs() < loc_thr || (!allow_neg && eff_v < 0.0) {
                continue;
            }

            let raw_sign = if raw_v >= 0.0 { 1 } else { -1 };
            let mut is_extremum = true;
            'nbr: for dz in -1isize..=1 {
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let nv = spec.get(
                            (ix as isize + dx) as usize,
                            (iy as isize + dy) as usize,
                            (iz as isize + dz) as usize,
                        );
                        if (raw_sign > 0 && nv >= raw_v) || (raw_sign < 0 && nv <= raw_v) {
                            is_extremum = false;
                            break 'nbr;
                        }
                    }
                }
            }
            if !is_extremum {
                continue;
            }

            let dx = subvoxel_offset(spec.get(ix - 1, iy, iz), raw_v, spec.get(ix + 1, iy, iz));
            let dy = subvoxel_offset(spec.get(ix, iy - 1, iz), raw_v, spec.get(ix, iy + 1, iz));
            let dz = subvoxel_offset(spec.get(ix, iy, iz - 1), raw_v, spec.get(ix, iy, iz + 1));

            let h = spec.axis_x.index_to_ppm(ix as f64 + dx);
            if let Some((lo, hi)) = h_gate {
                if h < lo || h > hi {
                    continue;
                }
            }
            let (n, c) = if y_is_n {
                (
                    spec.axis_y.index_to_ppm(iy as f64 + dy),
                    spec.axis_z.index_to_ppm(iz as f64 + dz),
                )
            } else {
                (
                    spec.axis_z.index_to_ppm(iz as f64 + dz),
                    spec.axis_y.index_to_ppm(iy as f64 + dy),
                )
            };
            if let Some((clo, chi)) = c_bounds {
                if c < clo || c > chi {
                    continue;
                }
            }

            let mut vol = 0.0f64;
            for wz in -1isize..=1 {
                for wy in -1isize..=1 {
                    for wx in -1isize..=1 {
                        let sv = spec.get(
                            (ix as isize + wx) as usize,
                            (iy as isize + wy) as usize,
                            (iz as isize + wz) as usize,
                        );
                        if sv.signum() == raw_v.signum() {
                            vol += (sv.abs()) as f64;
                        }
                    }
                }
            }

            let sign = if invert_polarity || raw_v >= 0.0 {
                1
            } else {
                -1
            };
            out.push(Peak {
                exp: spec.exp.clone(),
                h,
                n,
                c: Some(c),
                sign,
                height: Some((raw_v.abs()) as f64),
                volume: Some(vol),
                h2: None,
            });
        }
    }

    // Suppress near-parent 13C/15N truncation sinc sidelobes within the same (H, N) strip
    if high_dr && out.len() > 1 {
        let min_parent = (40.0 * noise) as f64;
        let half_sinc = 0.5 * (cfg.sinc_frac as f64);
        let mut keep = vec![true; out.len()];
        for i in 0..out.len() {
            let pi = &out[i];
            let Some(ci) = pi.c else { continue };
            let hti = pi.height.unwrap_or(0.0);
            for (j, pj) in out.iter().enumerate() {
                if i == j {
                    continue;
                }
                let Some(cj) = pj.c else { continue };
                let htj = pj.height.unwrap_or(0.0);
                if htj <= hti || htj < min_parent {
                    continue;
                }
                let dh = (pi.h - pj.h).abs();
                let dn = (pi.n - pj.n).abs();
                let dc = (ci - cj).abs();
                // 13C indirect sinc ripple along the same (H, N) strip
                if dh <= 0.025 && dn <= 0.30 && dc > 0.25 && dc <= 6.0 {
                    let env = half_sinc + 0.16 / (1.0 + (dc / 0.85).powi(2));
                    if hti < htj * env {
                        keep[i] = false;
                        break;
                    }
                }
                // 15N indirect sinc ripple at identical (H, C) plane
                if dh <= 0.020 && dc <= 0.25 && dn > 0.20 && dn <= 4.0 && pi.sign == pj.sign {
                    let env = half_sinc + 0.14 / (1.0 + (dn / 0.70).powi(2));
                    if hti < htj * env {
                        keep[i] = false;
                        break;
                    }
                }
            }
        }
        let mut filtered = Vec::with_capacity(out.len());
        for (p, k) in out.into_iter().zip(keep) {
            if k {
                filtered.push(p);
            }
        }
        out = filtered;
    }

    out
}

/// Configuration for Pass-2 experiment-corroborated 3D strip rescue (`corroborated_re_pick_3d`).
///
/// Guard against confirmation bias: Pass 2 accepts no sequence or prior shift inputs and inspects
/// raw 3D strips strictly inside narrow `+-ca_cb_window_ppm` (`0.15 ppm`) or `+-co_window_ppm`
/// (`0.12 ppm`) windows pinned by high-SNR (`>= min_anchor_snr`, default `5.5 sigma`) Pass-1
/// experimental peaks.
#[derive(Clone, Debug)]
pub struct CorroboratedRepickConfig {
    /// Minimum local-strip MAD SNR (`height / strip_mad`) required for a Pass-1 peak to serve
    /// as a corroborating anchor (default `5.5`).
    pub min_anchor_snr: f32,
    /// Conservative local-strip MAD floor (`k * strip_mad`, default `3.8`) for intra-strip
    /// cross-experiment rescue at a single pinned carbon shift within the same `(H, N)` strip.
    pub intra_strip_snr_floor: f32,
    /// Conservative local-strip MAD floor (`k * strip_mad`, default `4.2`) for cross-strip
    /// `0`-peak sequential pool rescue on multi-experiment validated backbone strips.
    pub cross_strip_snr_floor: f32,
    /// Half-width of the `13CA` / `13CB` corroboration window in ppm (default `0.15`).
    pub ca_cb_window_ppm: f64,
    /// Half-width of the `13C'` carbonyl corroboration window in ppm (default `0.12`).
    pub co_window_ppm: f64,
    /// Enable cross-strip sequential pool rescue on `0`-peak channels of validated backbone strips
    /// (default `true`).
    pub enable_cross_strip: bool,
}

impl Default for CorroboratedRepickConfig {
    fn default() -> Self {
        Self {
            min_anchor_snr: 5.5,
            intra_strip_snr_floor: 3.8,
            cross_strip_snr_floor: 4.2,
            ca_cb_window_ppm: 0.15,
            co_window_ppm: 0.12,
            enable_cross_strip: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ActiveStripCol {
    ix: usize,
    in_idx: usize,
    smed: f32,
    smad: f32,
    smax: f32,
}

/// Compact 1-voxel `(H, N)` halo strip representation of a 3D triple-resonance cube around
/// `hsqc_roots`, retaining bit-exact `3x3x3` local-extremum, sub-voxel refinement, volume, and
/// local-strip MAD statistics in ~1–5% of the memory of a full `Spectrum3D`.
#[derive(Clone, Debug)]
pub struct StripCube3D {
    exp: String,
    axis_h: AxisPpm,
    axis_n: AxisPpm,
    axis_c: AxisPpm,
    noise: f32,
    allow_neg: bool,
    invert_polarity: bool,
    high_dr: bool,
    far_sinc_frac: f32,
    c_bounds: Option<(f64, f64)>,
    ic_store_lo: usize,
    ic_store_len: usize,
    active_strips: Vec<ActiveStripCol>,
    halo_keys: Vec<u64>,
    col_data: Vec<f32>,
}

impl StripCube3D {
    /// Extract a compact `(H, N)` halo strip cube around `hsqc_roots` from a full `Spectrum3D`.
    pub fn from_spectrum_3d(spec: &Spectrum3D, hsqc_roots: &[Peak], cfg: &PickerConfig) -> Self {
        let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
        let noise = estimate_noise_mad(&spec.data);
        let thresh = cfg.snr_threshold * noise;
        let allow_neg = cfg.allow_negative || spec.exp == "HNCACB";
        let c_bounds = carbon_bounds_for_exp(&spec.exp);

        let y_is_n = spec.axis_y.label.contains('N')
            || (spec.axis_y.max_ppm() > 95.0
                && spec.axis_y.min_ppm() < 145.0
                && !spec.axis_z.label.contains('N'));
        let (n_n, n_c) = if y_is_n { (ny, nz) } else { (nz, ny) };
        let axis_n = if y_is_n {
            spec.axis_y.clone()
        } else {
            spec.axis_z.clone()
        };
        let axis_c = if y_is_n {
            spec.axis_z.clone()
        } else {
            spec.axis_y.clone()
        };

        if nx < 3 || ny < 3 || nz < 3 || hsqc_roots.is_empty() {
            return Self {
                exp: spec.exp.clone(),
                axis_h: spec.axis_x.clone(),
                axis_n,
                axis_c,
                noise,
                allow_neg,
                invert_polarity: false,
                high_dr: false,
                far_sinc_frac: (cfg.sinc_frac * 0.4).min(0.012),
                c_bounds,
                ic_store_lo: 0,
                ic_store_len: 0,
                active_strips: Vec::new(),
                halo_keys: Vec::new(),
                col_data: Vec::new(),
            };
        }

        let step_c = axis_c.step_ppm.abs();
        let mut active_c = Vec::with_capacity(n_c);
        for ic in 1..n_c - 1 {
            let cp = axis_c.index_to_ppm(ic as f64);
            if let Some((clo, chi)) = c_bounds {
                if cp < clo - step_c || cp > chi + step_c {
                    continue;
                }
            }
            active_c.push(ic);
        }

        let mut raw_active: Vec<(usize, usize)> = Vec::new();
        for in_idx in 1..n_n - 1 {
            let np = axis_n.index_to_ppm(in_idx as f64);
            if !hsqc_roots
                .iter()
                .any(|r| (r.n - np).abs() <= cfg.strip_tol_n)
            {
                continue;
            }
            for ix in 1..nx - 1 {
                let hp = spec.axis_x.index_to_ppm(ix as f64);
                if hsqc_roots.iter().any(|r| {
                    (r.h - hp).abs() <= cfg.strip_tol_h && (r.n - np).abs() <= cfg.strip_tol_n
                }) {
                    raw_active.push((ix, in_idx));
                }
            }
        }

        if active_c.is_empty() || raw_active.is_empty() {
            return Self {
                exp: spec.exp.clone(),
                axis_h: spec.axis_x.clone(),
                axis_n,
                axis_c,
                noise,
                allow_neg,
                invert_polarity: false,
                high_dr: false,
                far_sinc_frac: (cfg.sinc_frac * 0.4).min(0.012),
                c_bounds,
                ic_store_lo: 0,
                ic_store_len: 0,
                active_strips: Vec::new(),
                halo_keys: Vec::new(),
                col_data: Vec::new(),
            };
        }

        let get_voxel = |ix: usize, in_idx: usize, ic: usize| -> f32 {
            if y_is_n {
                spec.get(ix, in_idx, ic)
            } else {
                spec.get(ix, ic, in_idx)
            }
        };

        let mut max_pos = 0.0f32;
        let mut max_neg = 0.0f32;
        for &(ix, in_idx) in &raw_active {
            for &ic in &active_c {
                let v = get_voxel(ix, in_idx, ic);
                if v > max_pos {
                    max_pos = v;
                } else if -v > max_neg {
                    max_neg = -v;
                }
            }
        }
        let single_sign_backbone = !allow_neg
            && matches!(
                spec.exp.as_str(),
                "HNCA" | "HNCOCA" | "CBCACONH" | "HNCO" | "HNCACO"
            );
        let invert_polarity = single_sign_backbone && max_neg > 2.5 * max_pos.max(thresh);
        let spec_max_abs = if invert_polarity {
            max_neg
        } else if allow_neg {
            max_pos.max(max_neg)
        } else {
            max_pos
        };
        let high_dr = spec_max_abs > 40.0 * noise;
        let far_sinc_frac = (cfg.sinc_frac * 0.4).min(0.012);

        // Compute full-column outlier-trimmed median, MAD & max identically to Pass 1 before cropping 13C storage
        let mut strip_buf = Vec::with_capacity(n_c);
        let mut scratch = Vec::with_capacity(n_c);
        let mut active_strips = Vec::with_capacity(raw_active.len());
        for &(ix, in_idx) in &raw_active {
            strip_buf.clear();
            for ic in 0..n_c {
                strip_buf.push(get_voxel(ix, in_idx, ic));
            }
            let (smed, smad, smax) =
                slice_trimmed_median_mad_and_max(&strip_buf, &mut scratch, noise);
            active_strips.push(ActiveStripCol {
                ix,
                in_idx,
                smed,
                smad,
                smax,
            });
        }

        let ic_min = *active_c.first().unwrap_or(&1);
        let ic_max = *active_c.last().unwrap_or(&(n_c - 2));
        let ic_store_lo = ic_min.saturating_sub(1);
        let ic_store_hi = (ic_max + 1).min(n_c - 1);
        let ic_store_len = ic_store_hi - ic_store_lo + 1;

        let mut halo_keys = Vec::with_capacity(raw_active.len() * 4);
        for &(ix, in_idx) in &raw_active {
            for dn in -1isize..=1 {
                for dx in -1isize..=1 {
                    let jx = (ix as isize + dx) as u64;
                    let jn = (in_idx as isize + dn) as u64;
                    halo_keys.push((jx << 32) | jn);
                }
            }
        }
        halo_keys.sort_unstable();
        halo_keys.dedup();

        let mut col_data = Vec::with_capacity(halo_keys.len() * ic_store_len);
        for &key in &halo_keys {
            let ix = (key >> 32) as usize;
            let in_idx = (key & 0xFFFF_FFFF) as usize;
            for ic in ic_store_lo..=ic_store_hi {
                col_data.push(get_voxel(ix, in_idx, ic));
            }
        }

        Self {
            exp: spec.exp.clone(),
            axis_h: spec.axis_x.clone(),
            axis_n,
            axis_c,
            noise,
            allow_neg,
            invert_polarity,
            high_dr,
            far_sinc_frac,
            c_bounds,
            ic_store_lo,
            ic_store_len,
            active_strips,
            halo_keys,
            col_data,
        }
    }

    /// Experiment identifier (e.g. `"HNCACB"`, `"CBCACONH"`, `"HNCA"`, `"HNCOCA"`, `"HNCO"`, `"HNCACO"`).
    pub fn exp(&self) -> &str {
        &self.exp
    }

    /// Approximate heap footprint of the extracted strip cube in bytes.
    pub fn byte_size(&self) -> usize {
        self.col_data.len() * std::mem::size_of::<f32>()
            + self.halo_keys.len() * std::mem::size_of::<u64>()
            + self.active_strips.len() * std::mem::size_of::<ActiveStripCol>()
    }

    #[inline]
    fn get(&self, ix: usize, in_idx: usize, ic: usize) -> f32 {
        let key = ((ix as u64) << 32) | (in_idx as u64);
        let slot = self
            .halo_keys
            .binary_search(&key)
            .expect("1-voxel halo column present in StripCube3D");
        self.col_data[slot * self.ic_store_len + (ic - self.ic_store_lo)]
    }

    fn strip_mad_at(&self, h: f64, n: f64) -> f32 {
        if self.active_strips.is_empty() {
            return self.noise;
        }
        let max_ix = (self.axis_h.size.saturating_sub(1)) as f64;
        let max_in = (self.axis_n.size.saturating_sub(1)) as f64;
        let ix = self.axis_h.ppm_to_index(h).round().clamp(0.0, max_ix) as usize;
        let in_idx = self.axis_n.ppm_to_index(n).round().clamp(0.0, max_in) as usize;
        let mut best = self.noise;
        let mut best_d = usize::MAX;
        for s in &self.active_strips {
            let d = s.ix.abs_diff(ix) + s.in_idx.abs_diff(in_idx);
            if d < best_d {
                best_d = d;
                best = s.smad;
                if d == 0 {
                    break;
                }
            }
        }
        best
    }

    #[allow(clippy::too_many_arguments)]
    fn search_corroborated_window(
        &self,
        h_r: f64,
        n_r: f64,
        c_target: f64,
        win_ppm: f64,
        sign_req: Option<i32>,
        min_local_snr: f32,
        existing_exp_peaks: &[&Peak],
        cfg: &PickerConfig,
    ) -> Option<(Peak, f32)> {
        if self.ic_store_len < 3 || self.active_strips.is_empty() {
            return None;
        }
        let step_c = self.axis_c.step_ppm.abs();
        let f0 = self.axis_c.ppm_to_index(c_target - win_ppm - step_c);
        let f1 = self.axis_c.ppm_to_index(c_target + win_ppm + step_c);
        if !f0.is_finite() || !f1.is_finite() {
            return None;
        }
        let i0 = f0.floor() as isize;
        let i1 = f1.ceil() as isize;
        let ic_min_valid = (self.ic_store_lo + 1) as isize;
        let ic_max_valid = (self.ic_store_lo + self.ic_store_len - 2) as isize;
        let lo_isize = i0.min(i1).max(ic_min_valid);
        let hi_isize = i0.max(i1).min(ic_max_valid);
        if lo_isize < 0 || lo_isize > hi_isize {
            return None;
        }
        let ic_lo = lo_isize as usize;
        let ic_hi = hi_isize as usize;

        let snr_floor = min_local_snr.max(3.8);
        let mut best: Option<(Peak, f32, f64)> = None;

        for scol in &self.active_strips {
            let hp_grid = self.axis_h.index_to_ppm(scol.ix as f64);
            let np_grid = self.axis_n.index_to_ppm(scol.in_idx as f64);
            if (hp_grid - h_r).abs() > cfg.strip_tol_h || (np_grid - n_r).abs() > cfg.strip_tol_n {
                continue;
            }
            // Strictly enforce local-strip MAD floor (scol.smad >= self.noise) regardless of high_dr
            let base_thr = snr_floor * scol.smad;
            let loc_thr = if self.high_dr {
                base_thr.max(self.far_sinc_frac * scol.smax)
            } else {
                base_thr
            };

            for ic in ic_lo..=ic_hi {
                let raw_v = self.get(scol.ix, scol.in_idx, ic);
                let eff_v = if self.invert_polarity { -raw_v } else { raw_v };
                if eff_v.abs() < loc_thr || (!self.allow_neg && eff_v < 0.0) {
                    continue;
                }
                let sign = if self.invert_polarity || raw_v >= 0.0 {
                    1
                } else {
                    -1
                };
                if let Some(sr) = sign_req {
                    if sign != sr {
                        continue;
                    }
                }

                let raw_sign = if raw_v >= 0.0 { 1 } else { -1 };
                let sgn_f = raw_sign as f32;
                // Guard against t1 baseline/pedestal ridges that raise the DC floor along the 13C column:
                // deduct strip baseline median shifts beyond normal zero-mean thermal sampling jitter.
                let pedestal = (sgn_f * scol.smed - 0.60 * scol.smad).max(0.0);
                if eff_v.abs() - pedestal < base_thr {
                    continue;
                }

                let mut is_extremum = true;
                let mut halo_hn_sum = 0.0f32;
                'nbr: for dc in -1isize..=1 {
                    for dn in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dx == 0 && dn == 0 && dc == 0 {
                                continue;
                            }
                            let nv = self.get(
                                (scol.ix as isize + dx) as usize,
                                (scol.in_idx as isize + dn) as usize,
                                (ic as isize + dc) as usize,
                            );
                            if (raw_sign > 0 && nv >= raw_v) || (raw_sign < 0 && nv <= raw_v) {
                                is_extremum = false;
                                break 'nbr;
                            }
                            if dc == 0 {
                                halo_hn_sum += sgn_f * nv;
                            }
                        }
                    }
                }
                if !is_extremum {
                    continue;
                }

                // Q2(ii) t1 noise ridge guard: require genuine 2D (H, N) transverse prominence
                // across the 3x3 (dh, dn) halo at slice `ic` (using 8-voxel halo mean and 3-voxel
                // marginal row/column sums so sub-voxel split 3D Gaussian peaks are preserved while
                // flat (H, N) sheets and 1D H/N ridges are rejected).
                let c0 = raw_v.abs();
                let mut col_m1 = 0.0f32;
                let mut col_0 = 0.0f32;
                let mut col_p1 = 0.0f32;
                let mut row_m1 = 0.0f32;
                let mut row_0 = 0.0f32;
                let mut row_p1 = 0.0f32;
                for dn in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let pv = sgn_f
                            * self.get(
                                (scol.ix as isize + dx) as usize,
                                (scol.in_idx as isize + dn) as usize,
                                ic,
                            );
                        match dx {
                            -1 => col_m1 += pv,
                            0 => col_0 += pv,
                            _ => col_p1 += pv,
                        }
                        match dn {
                            -1 => row_m1 += pv,
                            0 => row_0 += pv,
                            _ => row_p1 += pv,
                        }
                    }
                }
                let halo_hn_mean = halo_hn_sum * 0.125;
                if halo_hn_mean > 0.84 * c0
                    || (c0 - halo_hn_mean) < 0.65 * scol.smad
                    || (col_0 > 0.0 && col_m1 > 0.92 * col_0 && col_p1 > 0.92 * col_0)
                    || (row_0 > 0.0 && row_m1 > 0.92 * row_0 && row_p1 > 0.92 * row_0)
                {
                    continue;
                }

                let dx = subvoxel_offset(
                    self.get(scol.ix - 1, scol.in_idx, ic),
                    raw_v,
                    self.get(scol.ix + 1, scol.in_idx, ic),
                );
                let dn = subvoxel_offset(
                    self.get(scol.ix, scol.in_idx - 1, ic),
                    raw_v,
                    self.get(scol.ix, scol.in_idx + 1, ic),
                );
                let dc = subvoxel_offset(
                    self.get(scol.ix, scol.in_idx, ic - 1),
                    raw_v,
                    self.get(scol.ix, scol.in_idx, ic + 1),
                );

                let h = self.axis_h.index_to_ppm(scol.ix as f64 + dx);
                let n = self.axis_n.index_to_ppm(scol.in_idx as f64 + dn);
                let c = self.axis_c.index_to_ppm(ic as f64 + dc);

                if (c - c_target).abs() > win_ppm {
                    continue;
                }
                if let Some((clo, chi)) = self.c_bounds {
                    if c < clo || c > chi {
                        continue;
                    }
                }
                // Require tight centering on the target HSQC root so neighbor bleed is never rescued
                let hd = ((h - h_r) / 0.028).hypot((n - n_r) / 0.28);
                if hd > 1.0 {
                    continue;
                }

                let hti = (raw_v.abs()) as f64;
                if self.high_dr {
                    let min_parent = (40.0 * self.noise) as f64;
                    let half_sinc = 0.5 * (cfg.sinc_frac as f64);
                    let mut is_sidelobe = false;
                    for pj in existing_exp_peaks {
                        let Some(cj) = pj.c else { continue };
                        let htj = pj.height.unwrap_or(0.0);
                        if htj <= hti || htj < min_parent {
                            continue;
                        }
                        let dhp = (h - pj.h).abs();
                        let dnp = (n - pj.n).abs();
                        let dcp = (c - cj).abs();
                        if dhp <= 0.025 && dnp <= 0.30 && dcp > 0.25 && dcp <= 6.0 {
                            let env = half_sinc + 0.16 / (1.0 + (dcp / 0.85).powi(2));
                            if hti < htj * env {
                                is_sidelobe = true;
                                break;
                            }
                        }
                        if dhp <= 0.020
                            && dcp <= 0.25
                            && dnp > 0.20
                            && dnp <= 4.0
                            && sign == pj.sign
                        {
                            let env = half_sinc + 0.14 / (1.0 + (dnp / 0.70).powi(2));
                            if hti < htj * env {
                                is_sidelobe = true;
                                break;
                            }
                        }
                        // Cross-sign F1 truncation sinc sidelobe guard in phase-sensitive HNCACB
                        if self.allow_neg
                            && sign != pj.sign
                            && dhp <= 0.035
                            && dnp <= 0.35
                            && dcp <= 0.65
                            && hti < 0.16 * htj
                        {
                            is_sidelobe = true;
                            break;
                        }
                    }
                    if is_sidelobe {
                        continue;
                    }
                }

                // Skip if Pass 1 already picked this extremum
                if existing_exp_peaks.iter().any(|pj| {
                    pj.sign == sign
                        && (pj.h - h).abs() <= 0.030
                        && (pj.n - n).abs() <= 0.30
                        && (pj.c.unwrap_or(0.0) - c).abs() <= 0.20
                }) {
                    continue;
                }

                let mut vol = 0.0f64;
                for wc in -1isize..=1 {
                    for wn in -1isize..=1 {
                        for wx in -1isize..=1 {
                            let sv = self.get(
                                (scol.ix as isize + wx) as usize,
                                (scol.in_idx as isize + wn) as usize,
                                (ic as isize + wc) as usize,
                            );
                            if sv.signum() == raw_v.signum() {
                                vol += (sv.abs()) as f64;
                            }
                        }
                    }
                }

                let lsnr = eff_v.abs() / scol.smad.max(1e-12);
                let rank = (lsnr as f64) * (-0.35 * hd * hd).exp();
                if best.as_ref().is_none_or(|b| rank > b.2) {
                    best = Some((
                        Peak {
                            exp: self.exp.clone(),
                            h,
                            n,
                            c: Some(c),
                            sign,
                            height: Some(hti),
                            volume: Some(vol),
                            h2: None,
                        },
                        lsnr,
                        rank,
                    ));
                }
            }
        }
        best.map(|(pk, lsnr, _)| (pk, lsnr))
    }
}

/// Multi-experiment bank of compact 3D strip cubes (`StripCube3D`) accumulated during streaming
/// Pass-1 3D picking so Pass 2 (`corroborated_re_pick_3d`) can inspect all experiments jointly
/// without keeping full 3D cubes (`~130–540 MB` each) in memory simultaneously.
#[derive(Clone, Debug, Default)]
pub struct StripBank3D {
    cubes: BTreeMap<String, StripCube3D>,
}

impl StripBank3D {
    /// Create an empty multi-experiment strip bank.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a pre-extracted `StripCube3D` keyed by its experiment name.
    pub fn insert_cube(&mut self, cube: StripCube3D) {
        self.cubes.insert(cube.exp.clone(), cube);
    }

    /// Run Pass-1 `pick_triple_3d` on `spec` and store its compact `StripCube3D` in this bank.
    pub fn pick_and_record_3d(
        &mut self,
        spec: &Spectrum3D,
        hsqc_roots: &[Peak],
        cfg: &PickerConfig,
    ) -> Vec<Peak> {
        let p1 = pick_triple_3d(spec, hsqc_roots, cfg);
        if !hsqc_roots.is_empty() {
            self.insert_cube(StripCube3D::from_spectrum_3d(spec, hsqc_roots, cfg));
        }
        p1
    }

    /// Look up the strip cube for an experiment name (e.g. `"HNCACB"`).
    pub fn get(&self, exp: &str) -> Option<&StripCube3D> {
        self.cubes.get(exp)
    }

    /// Number of 3D experiments stored in this bank.
    pub fn len(&self) -> usize {
        self.cubes.len()
    }

    /// Whether no 3D strip cubes have been recorded.
    pub fn is_empty(&self) -> bool {
        self.cubes.is_empty()
    }

    /// Total heap memory used across all stored strip cubes in bytes.
    pub fn byte_size(&self) -> usize {
        self.cubes.values().map(StripCube3D::byte_size).sum()
    }
}

/// Pass-2 experiment-corroborated 3D strip rescue (`docs/specs/02-peak-picking.md` Stage A2).
///
/// Returns ONLY the newly rescued Pass-2 3D peaks (caller may extend `pass1_peaks` with the result).
/// Never uses amino-acid sequence or chemical-shift priors; every search window is pinned by a
/// high-SNR (`>= repick_cfg.min_anchor_snr`) Pass-1 experimental peak.
pub fn corroborated_re_pick_3d(
    bank: &StripBank3D,
    hsqc_roots: &[Peak],
    pass1_peaks: &[Peak],
    cfg: &PickerConfig,
    repick_cfg: &CorroboratedRepickConfig,
) -> Vec<Peak> {
    if bank.is_empty() || hsqc_roots.is_empty() {
        return Vec::new();
    }

    let tol_h = 0.030;
    let tol_n = 0.30;
    let anchor_min_snr = repick_cfg.min_anchor_snr.max(4.5);
    let intra_snr = repick_cfg.intra_strip_snr_floor.max(3.8);
    let cross_snr = repick_cfg.cross_strip_snr_floor.max(3.8);
    let ca_cb_win = repick_cfg.ca_cb_window_ppm.clamp(0.05, 0.25);
    let co_win = repick_cfg.co_window_ppm.clamp(0.04, 0.20);
    let min_amide_h = cfg
        .amide_h_window
        .map(|(a, b)| a.min(b).max(6.0))
        .unwrap_or(0.0);

    // Q2(i): Determine majority sign of 2D HSQC roots and reject Pass-2 rescue on opposite-sign
    // roots (e.g. folded Arg Nε/Nη sidechain peaks folded with inverted sign in 15N).
    let n_neg_roots = hsqc_roots.iter().filter(|r| r.sign < 0).count();
    let majority_hsqc_sign = if n_neg_roots * 2 > hsqc_roots.len() {
        -1
    } else {
        1
    };
    let root_matches_majority_sign = |r: &Peak| -> bool {
        let sgn = if r.sign < 0 { -1 } else { 1 };
        sgn == majority_hsqc_sign
    };

    // Identify paired sidechain NH2 doublets (`|dN| <= 0.12 ppm`, `0.30 <= |dH| <= 1.20 ppm`)
    let mut is_nh2_pair = vec![false; hsqc_roots.len()];
    for i in 0..hsqc_roots.len() {
        for j in (i + 1)..hsqc_roots.len() {
            let dn = (hsqc_roots[i].n - hsqc_roots[j].n).abs();
            let dh = (hsqc_roots[i].h - hsqc_roots[j].h).abs();
            if dn <= 0.12 && (0.30..=1.20).contains(&dh) {
                is_nh2_pair[i] = true;
                is_nh2_pair[j] = true;
            }
        }
    }

    let mut by_exp: BTreeMap<String, Vec<&Peak>> = BTreeMap::new();
    for p in pass1_peaks {
        if p.exp != "HSQC" && p.c.is_some() {
            by_exp.entry(p.exp.clone()).or_default().push(p);
        }
    }

    #[derive(Clone)]
    struct RootPeak {
        exp: String,
        c: f64,
        sign: i32,
        lsnr: f32,
    }

    let mut root_peaks: Vec<Vec<RootPeak>> = vec![Vec::new(); hsqc_roots.len()];
    for p in pass1_peaks {
        let Some(c) = p.c else { continue };
        if p.exp == "HSQC" {
            continue;
        }
        let Some(cube) = bank.get(&p.exp) else {
            continue;
        };
        let smad = cube.strip_mad_at(p.h, p.n);
        let lsnr = (p.height.unwrap_or(0.0).abs() as f32) / smad.max(1e-12);
        for (ri, r) in hsqc_roots.iter().enumerate() {
            let d = ((r.h - p.h) / tol_h).hypot((r.n - p.n) / tol_n);
            if d <= 1.0 {
                root_peaks[ri].push(RootPeak {
                    exp: p.exp.clone(),
                    c,
                    sign: p.sign,
                    lsnr,
                });
            }
        }
    }

    let has_hncoca_cube = bank.get("HNCOCA").is_some();
    let mut rescued: Vec<Peak> = Vec::new();

    let has_near = |ri: usize, exp: &str, c0: f64, sgn: Option<i32>, dc: f64, extra: &[Peak]| {
        root_peaks[ri]
            .iter()
            .any(|rp| rp.exp == exp && sgn.is_none_or(|s| rp.sign == s) && (rp.c - c0).abs() <= dc)
            || extra.iter().any(|ep| {
                ep.exp == exp
                    && sgn.is_none_or(|s| ep.sign == s)
                    && ((ep.h - hsqc_roots[ri].h) / tol_h).hypot((ep.n - hsqc_roots[ri].n) / tol_n)
                        <= 1.0
                    && (ep.c.unwrap_or(0.0) - c0).abs() <= dc
            })
    };

    // Pass 2A: Intra-strip cross-experiment corroboration at the same HSQC root
    for (ri, r) in hsqc_roots.iter().enumerate() {
        if !root_matches_majority_sign(r)
            || r.h < min_amide_h
            || (is_nh2_pair[ri]
                && !root_peaks[ri].iter().any(|rp| {
                    (rp.exp == "HNCA" || rp.exp == "HNCACB") && rp.lsnr >= anchor_min_snr
                }))
        {
            continue;
        }

        let n_hnca = root_peaks[ri].iter().filter(|rp| rp.exp == "HNCA").count();

        // 1a. HNCOCA (>= anchor_min_snr) -> HNCACB(+1), and -> HNCA(+1) when strip has 0 HNCA peaks
        let hncoca_anchors: Vec<f64> = root_peaks[ri]
            .iter()
            .filter(|rp| {
                rp.exp == "HNCOCA" && rp.lsnr >= anchor_min_snr && (35.0..=76.0).contains(&rp.c)
            })
            .map(|rp| rp.c)
            .collect();
        for c0 in hncoca_anchors {
            if let Some(cube) = bank.get("HNCACB") {
                if !has_near(ri, "HNCACB", c0, Some(1), 0.30, &rescued) {
                    let existing = by_exp.get("HNCACB").map(Vec::as_slice).unwrap_or(&[]);
                    if let Some((pk, _)) = cube.search_corroborated_window(
                        r.h,
                        r.n,
                        c0,
                        ca_cb_win,
                        Some(1),
                        intra_snr,
                        existing,
                        cfg,
                    ) {
                        rescued.push(pk);
                    }
                }
            }
            if n_hnca == 0 {
                if let Some(cube) = bank.get("HNCA") {
                    if !has_near(ri, "HNCA", c0, Some(1), 0.30, &rescued) {
                        let existing = by_exp.get("HNCA").map(Vec::as_slice).unwrap_or(&[]);
                        if let Some((pk, _)) = cube.search_corroborated_window(
                            r.h,
                            r.n,
                            c0,
                            ca_cb_win,
                            Some(1),
                            intra_snr,
                            existing,
                            cfg,
                        ) {
                            rescued.push(pk);
                        }
                    }
                }
            }
        }

        // 1b. CBCACONH (>= anchor_min_snr) -> HNCACB (+CA / -CB typing), and -> HNCA when HNCOCA is absent or 0 HNCA
        let cbca_anchors: Vec<f64> = root_peaks[ri]
            .iter()
            .filter(|rp| rp.exp == "CBCACONH" && rp.lsnr >= anchor_min_snr)
            .map(|rp| rp.c)
            .collect();
        for c0 in cbca_anchors {
            if let Some(cube) = bank.get("HNCACB") {
                if !has_near(ri, "HNCACB", c0, None, 0.30, &rescued) {
                    let hncoca_match = root_peaks[ri]
                        .iter()
                        .any(|rp| rp.exp == "HNCOCA" && (rp.c - c0).abs() <= 0.25);
                    let sign_req = if hncoca_match || c0 > 73.0 {
                        Some(1)
                    } else if c0 < 38.0 {
                        Some(-1)
                    } else {
                        None
                    };
                    let existing = by_exp.get("HNCACB").map(Vec::as_slice).unwrap_or(&[]);
                    if let Some((pk, _)) = cube.search_corroborated_window(
                        r.h, r.n, c0, ca_cb_win, sign_req, intra_snr, existing, cfg,
                    ) {
                        rescued.push(pk);
                    }
                }
            }
            if (!has_hncoca_cube || n_hnca == 0) && (38.5..=76.0).contains(&c0) {
                if let Some(cube) = bank.get("HNCA") {
                    if !has_near(ri, "HNCA", c0, Some(1), 0.30, &rescued) {
                        let existing = by_exp.get("HNCA").map(Vec::as_slice).unwrap_or(&[]);
                        if let Some((pk, _)) = cube.search_corroborated_window(
                            r.h,
                            r.n,
                            c0,
                            ca_cb_win,
                            Some(1),
                            intra_snr,
                            existing,
                            cfg,
                        ) {
                            rescued.push(pk);
                        }
                    }
                }
            }
        }

        // 1c. HNCA / HNCACB(+1) (>= anchor_min_snr) -> HNCOCA when strip has 0 HNCOCA peaks
        if !root_peaks[ri].iter().any(|rp| rp.exp == "HNCOCA") {
            if let Some(cube) = bank.get("HNCOCA") {
                let existing = by_exp.get("HNCOCA").map(Vec::as_slice).unwrap_or(&[]);
                let mut best_cand: Option<(Peak, f32)> = None;
                let floor_ca_to_im1 = intra_snr.max(4.0);
                for rp in &root_peaks[ri] {
                    if (rp.exp == "HNCA" || (rp.exp == "HNCACB" && rp.sign > 0))
                        && rp.lsnr >= anchor_min_snr
                        && (38.0..=76.0).contains(&rp.c)
                    {
                        if let Some(cand) = cube.search_corroborated_window(
                            r.h,
                            r.n,
                            rp.c,
                            ca_cb_win,
                            Some(1),
                            floor_ca_to_im1,
                            existing,
                            cfg,
                        ) {
                            if best_cand.as_ref().is_none_or(|b| cand.1 > b.1) {
                                best_cand = Some(cand);
                            }
                        }
                    }
                }
                if let Some((pk, _)) = best_cand {
                    rescued.push(pk);
                }
            }
        }

        // 2a. HNCO (>= anchor_min_snr) -> HNCACO (corroborates C'(i-1) in HNCACO)
        if let Some(cube) = bank.get("HNCACO") {
            for rp in &root_peaks[ri] {
                if rp.exp == "HNCO"
                    && rp.lsnr >= anchor_min_snr
                    && !has_near(ri, "HNCACO", rp.c, Some(1), 0.18, &rescued)
                {
                    let existing = by_exp.get("HNCACO").map(Vec::as_slice).unwrap_or(&[]);
                    if let Some((pk, _)) = cube.search_corroborated_window(
                        r.h,
                        r.n,
                        rp.c,
                        co_win,
                        Some(1),
                        intra_snr,
                        existing,
                        cfg,
                    ) {
                        rescued.push(pk);
                    }
                }
            }
        }

        // 2b. HNCACO (>= anchor_min_snr) -> HNCO when strip has 0 HNCO peaks
        if !root_peaks[ri].iter().any(|rp| rp.exp == "HNCO") {
            if let Some(cube) = bank.get("HNCO") {
                let existing = by_exp.get("HNCO").map(Vec::as_slice).unwrap_or(&[]);
                let mut best_cand: Option<(Peak, f32)> = None;
                for rp in &root_peaks[ri] {
                    if rp.exp == "HNCACO" && rp.lsnr >= anchor_min_snr {
                        if let Some(cand) = cube.search_corroborated_window(
                            r.h,
                            r.n,
                            rp.c,
                            co_win,
                            Some(1),
                            intra_snr,
                            existing,
                            cfg,
                        ) {
                            if best_cand.as_ref().is_none_or(|b| cand.1 > b.1) {
                                best_cand = Some(cand);
                            }
                        }
                    }
                }
                if let Some((pk, _)) = best_cand {
                    rescued.push(pk);
                }
            }
        }
    }

    if !repick_cfg.enable_cross_strip {
        return rescued;
    }

    // Build global high-SNR experimental pools strictly from majority-sign Pass-1 non-NH2 backbone peaks
    let mut pool_co_im1: Vec<f64> = Vec::new();
    let mut pool_ca_im1: Vec<f64> = Vec::new();
    for (ri, rps) in root_peaks.iter().enumerate() {
        if !root_matches_majority_sign(&hsqc_roots[ri])
            || hsqc_roots[ri].h < min_amide_h
            || is_nh2_pair[ri]
        {
            continue;
        }
        for rp in rps {
            if rp.lsnr < anchor_min_snr {
                continue;
            }
            match rp.exp.as_str() {
                "HNCO" => pool_co_im1.push(rp.c),
                "HNCOCA" => pool_ca_im1.push(rp.c),
                "CBCACONH" if (38.5..=76.0).contains(&rp.c) => {
                    let is_cb = rps
                        .iter()
                        .any(|q| q.exp == "HNCACB" && q.sign < 0 && (q.c - rp.c).abs() <= 0.25);
                    if !is_cb {
                        pool_ca_im1.push(rp.c);
                    }
                }
                _ => {}
            }
        }
    }
    let dedup_shifts = |mut v: Vec<f64>, min_sep: f64| -> Vec<f64> {
        v.sort_by(|a, b| a.total_cmp(b));
        let mut out: Vec<f64> = Vec::new();
        for x in v {
            if out.last().is_none_or(|&last| (x - last) > min_sep) {
                out.push(x);
            }
        }
        out
    };
    let pool_co_im1 = dedup_shifts(pool_co_im1, 0.05);
    let pool_ca_im1 = dedup_shifts(pool_ca_im1, 0.06);

    // Pass 2B: Cross-strip 0-peak rescue on multi-experiment validated backbone strips
    for (ri, r) in hsqc_roots.iter().enumerate() {
        if !root_matches_majority_sign(r) || r.h < min_amide_h || is_nh2_pair[ri] {
            continue;
        }
        let mut distinct_exps: Vec<&str> = root_peaks[ri]
            .iter()
            .filter(|rp| rp.lsnr >= anchor_min_snr)
            .map(|rp| rp.exp.as_str())
            .collect();
        distinct_exps.sort_unstable();
        distinct_exps.dedup();
        let has_intra_backbone =
            distinct_exps.contains(&"HNCA") || distinct_exps.contains(&"HNCACB");
        if distinct_exps.len() < 2 || (!has_intra_backbone && !distinct_exps.contains(&"HNCO")) {
            continue;
        }

        // Cross-strip CA(i) rescue ONLY when HNCA or HNCACB(+1) has 0 positive peaks
        for target_exp in ["HNCA", "HNCACB"] {
            let Some(cube) = bank.get(target_exp) else {
                continue;
            };
            let n_pos = root_peaks[ri]
                .iter()
                .filter(|rp| rp.exp == target_exp && rp.sign > 0)
                .count()
                + rescued
                    .iter()
                    .filter(|ep| {
                        ep.exp == target_exp
                            && ep.sign > 0
                            && ((ep.h - r.h) / tol_h).hypot((ep.n - r.n) / tol_n) <= 1.0
                    })
                    .count();
            if n_pos == 0 {
                let existing = by_exp.get(target_exp).map(Vec::as_slice).unwrap_or(&[]);
                let mut best_cand: Option<(Peak, f32)> = None;
                for &c_cand in &pool_ca_im1 {
                    if let Some(cand) = cube.search_corroborated_window(
                        r.h,
                        r.n,
                        c_cand,
                        ca_cb_win,
                        Some(1),
                        cross_snr,
                        existing,
                        cfg,
                    ) {
                        if best_cand.as_ref().is_none_or(|b| cand.1 > b.1) {
                            best_cand = Some(cand);
                        }
                    }
                }
                if let Some((pk, _)) = best_cand {
                    rescued.push(pk);
                }
            }
        }

        // Cross-strip HNCO rescue ONLY when strip has 0 HNCO peaks and >= 2 aliphatic backbone anchors
        if has_intra_backbone
            && !root_peaks[ri].iter().any(|rp| rp.exp == "HNCO")
            && !rescued.iter().any(|ep| {
                ep.exp == "HNCO" && ((ep.h - r.h) / tol_h).hypot((ep.n - r.n) / tol_n) <= 1.0
            })
        {
            if let Some(cube) = bank.get("HNCO") {
                let existing = by_exp.get("HNCO").map(Vec::as_slice).unwrap_or(&[]);
                let mut best_cand: Option<(Peak, f32)> = None;
                for &c_cand in &pool_co_im1 {
                    if let Some(cand) = cube.search_corroborated_window(
                        r.h,
                        r.n,
                        c_cand,
                        co_win,
                        Some(1),
                        cross_snr,
                        existing,
                        cfg,
                    ) {
                        if best_cand.as_ref().is_none_or(|b| cand.1 > b.1) {
                            best_cand = Some(cand);
                        }
                    }
                }
                if let Some((pk, _)) = best_cand {
                    rescued.push(pk);
                }
            }
        }
    }

    rescued
}

/// Convenience two-pass 3D triple-resonance peak picker across a bundle of in-memory 3D cubes:
/// runs Pass 1 (`pick_triple_3d`) on every cube while recording a compact `StripBank3D`, then
/// appends Pass-2 corroborated rescues (`corroborated_re_pick_3d`).
pub fn pick_triple_3d_two_pass(
    spectra: &[&Spectrum3D],
    hsqc_roots: &[Peak],
    cfg: &PickerConfig,
    repick_cfg: &CorroboratedRepickConfig,
) -> Vec<Peak> {
    let mut bank = StripBank3D::new();
    let mut peaks = Vec::new();
    for spec in spectra {
        peaks.extend(bank.pick_and_record_3d(spec, hsqc_roots, cfg));
    }
    let rescued = corroborated_re_pick_3d(&bank, hsqc_roots, &peaks, cfg, repick_cfg);
    peaks.extend(rescued);
    peaks
}

// ============================================================================
// Interactive 3D Cube / Orthogonal Plane Peak Picker & Sub-Voxel Curation Helpers
// ============================================================================

/// General 3D `(F3, F2, F1)` peak picked directly on a volumetric [`Spectrum3D`] cube
/// (`f3_ppm` along `axis_x`, `f2_ppm` along `axis_y`, `f1_ppm` along `axis_z`).
#[derive(Clone, Debug, PartialEq)]
pub struct CubePeak3D {
    pub exp: String,
    pub f3_ppm: f64,
    pub f2_ppm: f64,
    pub f1_ppm: f64,
    pub sign: i32,
    pub height: f64,
    pub volume: f64,
    pub snr: f64,
    pub label: String,
}

impl CubePeak3D {
    /// Convert to a core [`Peak`] preserving physical nucleus channels (`1H`, `15N`, `13C`, `1H_indir`).
    pub fn to_core_peak(&self, nuc_x: &str, nuc_y: &str, nuc_z: &str) -> Peak {
        let abs_h = self.height.abs();
        let abs_v = self.volume.abs();
        if nuc_x == "1H" && nuc_y == "15N" && nuc_z == "1H" {
            Peak {
                exp: self.exp.clone(),
                h: self.f3_ppm,
                n: self.f2_ppm,
                c: Some(self.f1_ppm),
                sign: self.sign,
                height: Some(abs_h),
                volume: Some(abs_v),
                h2: Some(self.f1_ppm),
            }
        } else if nuc_x == "1H" && nuc_z == "15N" {
            let is_h_y = nuc_y == "1H";
            Peak {
                exp: self.exp.clone(),
                h: self.f3_ppm,
                n: self.f1_ppm,
                c: Some(self.f2_ppm),
                sign: self.sign,
                height: Some(abs_h),
                volume: Some(abs_v),
                h2: is_h_y.then_some(self.f2_ppm),
            }
        } else {
            Peak {
                exp: self.exp.clone(),
                h: self.f3_ppm,
                n: self.f2_ppm,
                c: Some(self.f1_ppm),
                sign: self.sign,
                height: Some(abs_h),
                volume: Some(abs_v),
                h2: (nuc_x == "1H" && nuc_z == "1H").then_some(self.f1_ppm),
            }
        }
    }
}

/// Configuration for [`pick_cube_3d_ridge_safe`] and [`pick_cube_3d_plane_ridge_safe`].
#[derive(Clone, Debug)]
pub struct Cube3dPickerConfig {
    /// Global outlier-trimmed MAD signal-to-noise multiplier (default `5.5`).
    pub snr_threshold: f32,
    /// Optional explicit contour-floor intensity cutoff (`0.0` = rely solely on `snr_threshold * noise`).
    pub min_abs_intensity: f32,
    /// Include negative local minima (e.g. `HNCACB` `-CB` or aliased resonances).
    pub allow_negative: bool,
    /// Exclude homonuclear diagonal self-peaks (`|ppm_a - ppm_b| <= diagonal_tol_ppm`) when two axes
    /// observe the same nucleus (e.g. `1H` on `F3` and `F1` in `H(CCO)NH` or `15N-NOESY`).
    pub exclude_diagonal: bool,
    /// Chemical-shift half-tolerance (ppm) for homonuclear diagonal exclusion (default `0.05`).
    pub diagonal_tol_ppm: f64,
    /// Exclude the residual direct-`1H` water slab (`|f3_ppm - water_ppm| <= water_half_width_ppm`)
    /// when `axis_x` is `${}^{1}\text{H}$ and spans across water.
    pub exclude_water_band: bool,
    /// Explicit water chemical shift in ppm (`None` = auto-detect on direct `1H` projection, fallback `4.70`).
    pub water_ppm: Option<f64>,
    /// Half-width (ppm) around `water_ppm` excluded on the direct `${}^{1}\text{H}$ axis (default `0.06`).
    pub water_half_width_ppm: f64,
    /// Multiplier on local 1D fiber MAD for indirect $t_1$/$t_2$ ridge rejection (default `5.5`).
    pub ridge_snr: f32,
    /// Minimum fraction of 1D fiber maximum for FT truncation sinc-ripple rejection (default `0.025`).
    pub sinc_frac: f32,
    /// Upper bound on returned peaks after sorting by `|height|` (default `4_000`).
    pub max_peaks: usize,
}

impl Default for Cube3dPickerConfig {
    fn default() -> Self {
        Self {
            snr_threshold: 5.5,
            min_abs_intensity: 0.0,
            allow_negative: true,
            exclude_diagonal: true,
            diagonal_tol_ppm: 0.05,
            exclude_water_band: true,
            water_ppm: None,
            water_half_width_ppm: 0.06,
            ridge_snr: 5.5,
            sinc_frac: 0.025,
            max_peaks: 4_000,
        }
    }
}

/// Larmor-frequency-ratio nucleus windows (`sf_mhz / proton_mhz`), identical to the windows used by
/// `nmr-gui::workbook::axis_physical_nucleus` so the GUI and the core picker always agree on the same cube.
fn nucleus_from_larmor_ratio(ratio: f64) -> Option<&'static str> {
    if (0.980..=1.020).contains(&ratio) {
        Some("1H")
    } else if (0.925..=0.958).contains(&ratio) {
        Some("19F")
    } else if (0.390..=0.420).contains(&ratio) {
        Some("31P")
    } else if (0.235..=0.270).contains(&ratio) {
        Some("13C")
    } else if (0.145..=0.165).contains(&ratio) {
        Some("2H")
    } else if (0.090..=0.115).contains(&ratio) {
        Some("15N")
    } else {
        None
    }
}

/// Infer the nucleus of a cube axis **from its label alone** (`20.W1:P1-1`).
///
/// Exact standard labels are matched first (`"HN"`, `"HA"`, `"H"`, `"1H"` -> `1H`; `"N"`, `"NH"`, `"15N"` -> `15N`;
/// `"C"`, `"CA"`, `"CB"`, `"CO"`, `"13C"` -> `13C`). In the substring fallback `H`/`1H` is tested
/// **before** a bare `N`, because the ubiquitous amide-proton label `"HN"` contains the letter `N` and used to
/// be misread as `15N` (which silently disabled the `4.70 ppm` water-band and diagonal exclusion).
fn nucleus_from_axis_label(axis: &AxisPpm) -> &'static str {
    let up: String = axis
        .label
        .trim()
        .trim_matches(|c: char| c == '<' || c == '>' || c == '"' || c == '\'')
        .trim()
        .to_ascii_uppercase();
    match up.as_str() {
        "1H" | "H1" | "H" | "HN" | "HA" | "HB" | "HC" | "PROTON" | "1H_IND" | "1H_ALIPH" => {
            return "1H";
        }
        "15N" | "N15" | "N" | "NH" => return "15N",
        "13C" | "C13" | "C" | "CA" | "CB" | "CO" | "CACB" => return "13C",
        "19F" | "F19" | "F" => return "19F",
        "31P" | "P31" | "P" => return "31P",
        _ => {}
    }
    if up.contains("1H") || up.contains("H1") || up.starts_with('H') {
        "1H"
    } else if up.contains("15N") || up.contains("N15") {
        "15N"
    } else if up.contains("13C") || up.contains("C13") {
        "13C"
    } else if up.contains('N') {
        "15N"
    } else if up.contains('C') {
        "13C"
    } else if up.contains('H') || (axis.max_ppm() <= 19.0 && axis.min_ppm() >= -6.0) {
        "1H"
    } else if axis.min_ppm() >= 88.0 && axis.max_ppm() <= 152.0 {
        "15N"
    } else {
        "13C"
    }
}

fn infer_cube_axis_iso(axis: &AxisPpm, ref_h_mhz: Option<f64>) -> &'static str {
    if let Some(hmhz) = ref_h_mhz.filter(|&m| m > 50.0) {
        if axis.obs_mhz.is_finite() && axis.obs_mhz > 5.0 {
            if let Some(n) = nucleus_from_larmor_ratio(axis.obs_mhz / hmhz) {
                // `2H` is not a supported cube nucleus tag in the picker; fall through to the label.
                if n != "2H" {
                    return n;
                }
            }
        }
    }
    nucleus_from_axis_label(axis)
}

/// Resolve the proton reference frequency of a cube: the largest `sf_mhz` among the axes that are
/// unambiguously `1H` (by label or by a proton-sized ppm window), falling back to the overall highest
/// spectrometer frequency when its ppm window is proton-like (the `1H` axis always has the highest `SF`
/// in the supported triple-resonance cubes).
fn cube_proton_reference_mhz(spec: &Spectrum3D) -> Option<f64> {
    let axes = [&spec.axis_x, &spec.axis_y, &spec.axis_z];
    let by_label = axes
        .iter()
        .filter(|a| a.obs_mhz.is_finite() && a.obs_mhz > 50.0 && nucleus_from_axis_label(a) == "1H")
        .map(|a| a.obs_mhz)
        .fold(0.0_f64, f64::max);
    if by_label > 50.0 {
        return Some(by_label);
    }
    let top = axes
        .iter()
        .filter(|a| a.obs_mhz.is_finite() && a.obs_mhz > 50.0)
        .max_by(|a, b| a.obs_mhz.total_cmp(&b.obs_mhz))?;
    (top.max_ppm() <= 25.0 && top.min_ppm() >= -10.0).then_some(top.obs_mhz)
}

fn cube_axis_nuclei(spec: &Spectrum3D) -> (&'static str, &'static str, &'static str) {
    let ref_h = cube_proton_reference_mhz(spec);
    let nx = infer_cube_axis_iso(&spec.axis_x, ref_h);
    let ny = infer_cube_axis_iso(&spec.axis_y, ref_h);
    let nz = infer_cube_axis_iso(&spec.axis_z, ref_h);
    (nx, ny, nz)
}

fn detect_direct_water_band(
    spec: &Spectrum3D,
    cfg: &Cube3dPickerConfig,
    nuc_x: &str,
) -> Option<(f64, f64)> {
    if !cfg.exclude_water_band || nuc_x != "1H" {
        return None;
    }
    let x_lo = spec.axis_x.min_ppm();
    let x_hi = spec.axis_x.max_ppm();
    let spans_h2o = x_lo < 4.55 && x_hi > 4.85;
    let spans_dmso = x_lo < 3.18 && x_hi > 3.50;
    if !spans_h2o && !spans_dmso && cfg.water_ppm.is_none() {
        return None;
    }
    let w_ppm = if let Some(explicit) = cfg.water_ppm {
        explicit
    } else {
        let nx = spec.nx();
        let total = spec.ny().saturating_mul(spec.nz());
        let stride = (total / 256).max(1);
        let mut prof = vec![0.0_f64; nx];
        for r_idx in (0..total).step_by(stride) {
            let base = r_idx * nx;
            if base + nx <= spec.data.len() {
                for (dst, &v) in prof.iter_mut().zip(&spec.data[base..base + nx]) {
                    if v.is_finite() {
                        *dst += (v as f64).abs();
                    }
                }
            }
        }
        crate::water::detect_water_ppm(&spec.axis_x, &prof, None)
    };
    let hw = cfg.water_half_width_ppm.abs().max(0.015);
    if w_ppm >= x_lo - hw && w_ppm <= x_hi + hw {
        Some((w_ppm - hw, w_ppm + hw))
    } else {
        None
    }
}

fn is_homonuclear_diagonal_hit(
    f3_ppm: f64,
    f2_ppm: f64,
    f1_ppm: f64,
    nucs: (&str, &str, &str),
    tol_ppm: f64,
) -> bool {
    let tol_x_y = if nucs.0 == "1H" {
        tol_ppm
    } else {
        tol_ppm.max(0.30)
    };
    let tol_x_z = if nucs.0 == "1H" {
        tol_ppm
    } else {
        tol_ppm.max(0.30)
    };
    let tol_y_z = if nucs.1 == "1H" {
        tol_ppm
    } else {
        tol_ppm.max(0.30)
    };
    (nucs.0 == nucs.1 && (f3_ppm - f2_ppm).abs() <= tol_x_y)
        || (nucs.0 == nucs.2 && (f3_ppm - f1_ppm).abs() <= tol_x_z)
        || (nucs.1 == nucs.2 && (f2_ppm - f1_ppm).abs() <= tol_y_z)
}

/// Trilinearly interpolate voxel amplitude in `spec` at `(f3_ppm, f2_ppm, f1_ppm)`.
pub fn sample_cube_3d_trilinear(spec: &Spectrum3D, f3_ppm: f64, f2_ppm: f64, f1_ppm: f64) -> f64 {
    let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
    if nx == 0 || ny == 0 || nz == 0 || spec.data.len() < nx.saturating_mul(ny).saturating_mul(nz) {
        return 0.0;
    }
    let xf = spec.axis_x.ppm_to_index(f3_ppm).clamp(0.0, (nx - 1) as f64);
    let yf = spec.axis_y.ppm_to_index(f2_ppm).clamp(0.0, (ny - 1) as f64);
    let zf = spec.axis_z.ppm_to_index(f1_ppm).clamp(0.0, (nz - 1) as f64);

    let x0 = (xf.floor() as usize).min(nx - 1);
    let x1 = (x0 + 1).min(nx - 1);
    let y0 = (yf.floor() as usize).min(ny - 1);
    let y1 = (y0 + 1).min(ny - 1);
    let z0 = (zf.floor() as usize).min(nz - 1);
    let z1 = (z0 + 1).min(nz - 1);

    let tx = (xf - x0 as f64) as f32;
    let ty = (yf - y0 as f64) as f32;
    let tz = (zf - z0 as f64) as f32;

    let lerp = |a: f32, b: f32, t: f32| a * (1.0 - t) + b * t;
    let c00 = lerp(spec.get(x0, y0, z0), spec.get(x1, y0, z0), tx);
    let c10 = lerp(spec.get(x0, y1, z0), spec.get(x1, y1, z0), tx);
    let c01 = lerp(spec.get(x0, y0, z1), spec.get(x1, y0, z1), tx);
    let c11 = lerp(spec.get(x0, y1, z1), spec.get(x1, y1, z1), tx);
    let c0 = lerp(c00, c10, ty);
    let c1 = lerp(c01, c11, ty);
    lerp(c0, c1, tz) as f64
}

/// Refine `(f3_ppm, f2_ppm, f1_ppm)` around `±search_vox` (`[rx, ry, rz]`) to the local 3D voxel
/// extremum + 3-point parabolic/log-Gaussian sub-voxel center, returning
/// `(fitted_f3, fitted_f2, fitted_f1, signed_height, volume, sign)`.
pub fn snap_cube_3d_extremum(
    spec: &Spectrum3D,
    f3_ppm: f64,
    f2_ppm: f64,
    f1_ppm: f64,
    search_vox: [usize; 3],
    allow_negative: bool,
) -> (f64, f64, f64, f64, f64, i32) {
    let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
    if nx < 3 || ny < 3 || nz < 3 || spec.data.len() < nx.saturating_mul(ny).saturating_mul(nz) {
        let amp = sample_cube_3d_trilinear(spec, f3_ppm, f2_ppm, f1_ppm);
        let sgn = if amp < 0.0 && allow_negative { -1 } else { 1 };
        return (f3_ppm, f2_ppm, f1_ppm, amp, amp.abs(), sgn);
    }

    let cx = spec
        .axis_x
        .ppm_to_index(f3_ppm)
        .round()
        .clamp(1.0, (nx - 2) as f64) as isize;
    let cy = spec
        .axis_y
        .ppm_to_index(f2_ppm)
        .round()
        .clamp(1.0, (ny - 2) as f64) as isize;
    let cz = spec
        .axis_z
        .ppm_to_index(f1_ppm)
        .round()
        .clamp(1.0, (nz - 2) as f64) as isize;

    let rx = (search_vox[0].min(8)) as isize;
    let ry = (search_vox[1].min(8)) as isize;
    let rz = (search_vox[2].min(8)) as isize;

    let mut best_ix = cx as usize;
    let mut best_iy = cy as usize;
    let mut best_iz = cz as usize;
    let mut best_score = f32::NEG_INFINITY;

    for dz in -rz..=rz {
        let iz = (cz + dz).clamp(1, (nz - 2) as isize) as usize;
        for dy in -ry..=ry {
            let iy = (cy + dy).clamp(1, (ny - 2) as isize) as usize;
            for dx in -rx..=rx {
                let ix = (cx + dx).clamp(1, (nx - 2) as isize) as usize;
                let v = spec.get(ix, iy, iz);
                let sc = if allow_negative { v.abs() } else { v };
                if sc > best_score {
                    best_score = sc;
                    best_ix = ix;
                    best_iy = iy;
                    best_iz = iz;
                }
            }
        }
    }

    let v0 = spec.get(best_ix, best_iy, best_iz);
    let sign = if v0 < 0.0 && allow_negative { -1 } else { 1 };
    let dx = if search_vox[0] > 0 {
        subvoxel_offset(
            spec.get(best_ix - 1, best_iy, best_iz),
            v0,
            spec.get(best_ix + 1, best_iy, best_iz),
        )
    } else {
        0.0
    };
    let dy = if search_vox[1] > 0 {
        subvoxel_offset(
            spec.get(best_ix, best_iy - 1, best_iz),
            v0,
            spec.get(best_ix, best_iy + 1, best_iz),
        )
    } else {
        0.0
    };
    let dz = if search_vox[2] > 0 {
        subvoxel_offset(
            spec.get(best_ix, best_iy, best_iz - 1),
            v0,
            spec.get(best_ix, best_iy, best_iz + 1),
        )
    } else {
        0.0
    };

    let out_f3 = if search_vox[0] > 0 {
        spec.axis_x.index_to_ppm(best_ix as f64 + dx)
    } else {
        f3_ppm
    };
    let out_f2 = if search_vox[1] > 0 {
        spec.axis_y.index_to_ppm(best_iy as f64 + dy)
    } else {
        f2_ppm
    };
    let out_f1 = if search_vox[2] > 0 {
        spec.axis_z.index_to_ppm(best_iz as f64 + dz)
    } else {
        f1_ppm
    };

    let mut vol = 0.0_f64;
    for wz in -1isize..=1 {
        for wy in -1isize..=1 {
            for wx in -1isize..=1 {
                let sv = spec.get(
                    (best_ix as isize + wx) as usize,
                    (best_iy as isize + wy) as usize,
                    (best_iz as isize + wz) as usize,
                );
                if sv.signum() == v0.signum() {
                    vol += (sv.abs()) as f64;
                }
            }
        }
    }

    (out_f3, out_f2, out_f1, v0 as f64, vol, sign)
}

/// Pick true 3D `(F3, F2, F1)` local extrema across a [`Spectrum3D`] cube with outlier-trimmed MAD
/// noise estimation, local 1D fiber ridge & FT sinc-ripple suppression, direct `${}^{1}\text{H}$
/// water-band exclusion, homonuclear diagonal exclusion, and 3-point sub-voxel interpolation,
/// sorted descending by `|height|`.
pub fn pick_cube_3d_ridge_safe(spec: &Spectrum3D, cfg: &Cube3dPickerConfig) -> Vec<CubePeak3D> {
    let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
    let total_len = nx.saturating_mul(ny).saturating_mul(nz);
    if nx < 3 || ny < 3 || nz < 3 || spec.data.len() < total_len {
        return Vec::new();
    }

    let noise = estimate_noise_mad(&spec.data[..total_len]).max(1e-12);
    let global_thr = (cfg.snr_threshold.max(1.5) * noise).max(cfg.min_abs_intensity.max(0.0));
    let nucs = cube_axis_nuclei(spec);
    let water_band = detect_direct_water_band(spec, cfg, nucs.0);
    let step_x = spec.axis_x.step_ppm.abs();

    let mut active_x = vec![false; nx];
    for (ix, slot) in active_x.iter_mut().enumerate().take(nx - 1).skip(1) {
        if let Some((w_lo, w_hi)) = water_band {
            let px = spec.axis_x.index_to_ppm(ix as f64);
            if px >= w_lo - 0.5 * step_x && px <= w_hi + 0.5 * step_x {
                continue;
            }
        }
        *slot = true;
    }

    let spec_max_abs = spec.data[..total_len]
        .iter()
        .step_by((total_len / 65_536).max(1))
        .fold(
            0.0_f32,
            |m, &v| if v.is_finite() { m.max(v.abs()) } else { m },
        );
    let high_dr = spec_max_abs > 32.0 * noise;

    let mut z_fiber_stats: BTreeMap<(usize, usize), (f32, f32)> = BTreeMap::new();
    let mut y_fiber_stats: BTreeMap<(usize, usize), (f32, f32)> = BTreeMap::new();
    let mut x_fiber_stats: BTreeMap<(usize, usize), (f32, f32)> = BTreeMap::new();
    let mut fiber_buf = Vec::with_capacity(nx.max(ny).max(nz));
    let mut scratch = Vec::with_capacity(nx.max(ny).max(nz));

    let plane_stride = ny * nx;
    let mut raw_peaks: Vec<CubePeak3D> = Vec::new();

    for iz in 1..nz - 1 {
        let z_base = iz * plane_stride;
        for iy in 1..ny - 1 {
            let row_base = z_base + iy * nx;
            let row = &spec.data[row_base..row_base + nx];
            for ix in 1..nx - 1 {
                if !active_x[ix] {
                    continue;
                }
                let raw_v = row[ix];
                if !raw_v.is_finite()
                    || raw_v.abs() < global_thr
                    || (!cfg.allow_negative && raw_v < 0.0)
                {
                    continue;
                }

                // 26-neighbor 3D local extremum test (deterministic index tie-break on flat plateaus)
                let pos_mode = raw_v >= 0.0;
                let mut is_ext = true;
                'nbr: for dz in -1isize..=1 {
                    let nz_i = (iz as isize + dz) as usize;
                    let nb_z = nz_i * plane_stride;
                    for dy in -1isize..=1 {
                        let ny_i = (iy as isize + dy) as usize;
                        let nb_row = nb_z + ny_i * nx;
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let nx_i = (ix as isize + dx) as usize;
                            let nv = spec.data[nb_row + nx_i];
                            let strictly_worse = if pos_mode { nv > raw_v } else { nv < raw_v };
                            let tie_earlier = nv == raw_v && (dz, dy, dx) < (0, 0, 0);
                            if strictly_worse || tie_earlier {
                                is_ext = false;
                                break 'nbr;
                            }
                        }
                    }
                }
                if !is_ext {
                    continue;
                }

                if high_dr {
                    let (z_med, z_thr) = *z_fiber_stats.entry((ix, iy)).or_insert_with(|| {
                        fiber_buf.clear();
                        for gz in 0..nz {
                            fiber_buf.push(spec.data[gz * plane_stride + iy * nx + ix]);
                        }
                        let (fmed, fmad, fmax) =
                            slice_trimmed_median_mad_and_max(&fiber_buf, &mut scratch, noise);
                        let dev_max = (fmax - fmed.abs()).max(0.0);
                        (
                            fmed,
                            global_thr
                                .max(cfg.ridge_snr * fmad)
                                .max((cfg.sinc_frac * 0.5) * dev_max),
                        )
                    });
                    if (raw_v - z_med).abs() < z_thr {
                        continue;
                    }

                    let (y_med, y_thr) = *y_fiber_stats.entry((ix, iz)).or_insert_with(|| {
                        fiber_buf.clear();
                        for gy in 0..ny {
                            fiber_buf.push(spec.data[z_base + gy * nx + ix]);
                        }
                        let (fmed, fmad, fmax) =
                            slice_trimmed_median_mad_and_max(&fiber_buf, &mut scratch, noise);
                        let dev_max = (fmax - fmed.abs()).max(0.0);
                        (
                            fmed,
                            global_thr
                                .max(cfg.ridge_snr * fmad)
                                .max((cfg.sinc_frac * 0.5) * dev_max),
                        )
                    });
                    if (raw_v - y_med).abs() < y_thr {
                        continue;
                    }

                    let (x_med, x_thr) = *x_fiber_stats.entry((iy, iz)).or_insert_with(|| {
                        fiber_buf.clear();
                        fiber_buf.extend_from_slice(row);
                        let (fmed, fmad, fmax) =
                            slice_trimmed_median_mad_and_max(&fiber_buf, &mut scratch, noise);
                        let dev_max = (fmax - fmed.abs()).max(0.0);
                        (
                            fmed,
                            global_thr
                                .max(cfg.ridge_snr * fmad)
                                .max((cfg.sinc_frac * 0.5) * dev_max),
                        )
                    });
                    if (raw_v - x_med).abs() < x_thr {
                        continue;
                    }
                }

                let dx = subvoxel_offset(row[ix - 1], raw_v, row[ix + 1]);
                let dy = subvoxel_offset(
                    spec.data[z_base + (iy - 1) * nx + ix],
                    raw_v,
                    spec.data[z_base + (iy + 1) * nx + ix],
                );
                let dz = subvoxel_offset(
                    spec.data[(iz - 1) * plane_stride + iy * nx + ix],
                    raw_v,
                    spec.data[(iz + 1) * plane_stride + iy * nx + ix],
                );

                let f3_ppm = spec.axis_x.index_to_ppm(ix as f64 + dx);
                let f2_ppm = spec.axis_y.index_to_ppm(iy as f64 + dy);
                let f1_ppm = spec.axis_z.index_to_ppm(iz as f64 + dz);

                if let Some((w_lo, w_hi)) = water_band {
                    if (w_lo..=w_hi).contains(&f3_ppm) {
                        continue;
                    }
                }
                if cfg.exclude_diagonal
                    && is_homonuclear_diagonal_hit(
                        f3_ppm,
                        f2_ppm,
                        f1_ppm,
                        nucs,
                        cfg.diagonal_tol_ppm,
                    )
                {
                    continue;
                }

                let mut vol = 0.0_f64;
                for wz in -1isize..=1 {
                    for wy in -1isize..=1 {
                        for wx in -1isize..=1 {
                            let sv = spec.get(
                                (ix as isize + wx) as usize,
                                (iy as isize + wy) as usize,
                                (iz as isize + wz) as usize,
                            );
                            if sv.signum() == raw_v.signum() {
                                vol += (sv.abs()) as f64;
                            }
                        }
                    }
                }

                raw_peaks.push(CubePeak3D {
                    exp: spec.exp.clone(),
                    f3_ppm,
                    f2_ppm,
                    f1_ppm,
                    sign: if raw_v < 0.0 { -1 } else { 1 },
                    height: raw_v as f64,
                    volume: vol,
                    snr: (raw_v.abs() / noise) as f64,
                    label: String::new(),
                });
            }
        }
    }

    raw_peaks.sort_by(|a, b| b.height.abs().total_cmp(&a.height.abs()));

    // Suppress close 1D indirect/direct FT sinc ripples around high-SNR parent peaks
    if high_dr && raw_peaks.len() > 1 {
        let min_parent = (35.0 * noise) as f64;
        let tol_x = (spec.axis_x.step_ppm.abs() * 1.8).max(0.02);
        let tol_y = (spec.axis_y.step_ppm.abs() * 1.8).max(0.12);
        let tol_z = (spec.axis_z.step_ppm.abs() * 1.8).max(0.06);
        let base_sinc = cfg.sinc_frac as f64;
        let mut keep = vec![true; raw_peaks.len()];
        for i in 1..raw_peaks.len() {
            let pi = &raw_peaks[i];
            let hi = pi.height.abs();
            for pj in raw_peaks.iter().take(i) {
                let hj = pj.height.abs();
                if hj < min_parent || hj <= hi {
                    break;
                }
                let dx = (pi.f3_ppm - pj.f3_ppm).abs();
                let dy = (pi.f2_ppm - pj.f2_ppm).abs();
                let dz = (pi.f1_ppm - pj.f1_ppm).abs();
                if dx <= tol_x && dy <= tol_y && dz > tol_z && dz <= tol_z * 14.0 {
                    let env = 0.5 * base_sinc + 0.14 / (1.0 + (dz / (tol_z * 3.0)).powi(2));
                    if hi < hj * env {
                        keep[i] = false;
                        break;
                    }
                }
                if dx <= tol_x && dz <= tol_z && dy > tol_y && dy <= tol_y * 14.0 {
                    let env = 0.5 * base_sinc + 0.14 / (1.0 + (dy / (tol_y * 3.0)).powi(2));
                    if hi < hj * env {
                        keep[i] = false;
                        break;
                    }
                }
            }
        }
        let mut filtered = Vec::with_capacity(raw_peaks.len());
        for (pk, k) in raw_peaks.into_iter().zip(keep) {
            if k {
                filtered.push(pk);
            }
        }
        raw_peaks = filtered;
    }

    if raw_peaks.len() > cfg.max_peaks.max(1) {
        raw_peaks.truncate(cfg.max_peaks.max(1));
    }
    raw_peaks
}

/// Pick ridge-safe 2D cross-peaks on a single orthogonal plane of `spec`
/// (`ortho_axis`: `0` = fixed `F3/axis_x`, `1` = fixed `F2/axis_y`, `2` = fixed `F1/axis_z`)
/// at `plane_idx`, with 3-point sub-voxel interpolation, sorted descending by `|height|`.
pub fn pick_cube_3d_plane_ridge_safe(
    spec: &Spectrum3D,
    ortho_axis: usize,
    plane_idx: usize,
    cfg: &Cube3dPickerConfig,
) -> Vec<CubePeak3D> {
    let (nx, ny, nz) = (spec.nx(), spec.ny(), spec.nz());
    let total_len = nx.saturating_mul(ny).saturating_mul(nz);
    if nx < 3 || ny < 3 || nz < 3 || spec.data.len() < total_len {
        return Vec::new();
    }

    let (nu, nv, n_ortho) = match ortho_axis {
        0 => (ny, nz, nx),
        1 => (nx, nz, ny),
        _ => (nx, ny, nz),
    };
    if nu < 3 || nv < 3 || n_ortho == 0 {
        return Vec::new();
    }
    let p_idx = plane_idx.min(n_ortho - 1);

    let voxel_at = |u: usize, v: usize, w: usize| -> f32 {
        match ortho_axis {
            0 => spec.get(w, u, v),
            1 => spec.get(u, w, v),
            _ => spec.get(u, v, w),
        }
    };

    let mut plane_vals = Vec::with_capacity(nu * nv);
    for iv in 0..nv {
        for iu in 0..nu {
            plane_vals.push(voxel_at(iu, iv, p_idx));
        }
    }

    let noise = estimate_noise_mad(&plane_vals)
        .max(estimate_noise_mad(&spec.data[..total_len]) * 0.5)
        .max(1e-12);
    let base_thr = (cfg.snr_threshold.max(1.5) * noise).max(cfg.min_abs_intensity.max(0.0));
    let nucs = cube_axis_nuclei(spec);
    let water_band = detect_direct_water_band(spec, cfg, nucs.0);

    let mut scratch = Vec::with_capacity(nu.max(nv));
    let mut slice_buf = Vec::with_capacity(nu.max(nv));
    let plane_max = plane_vals.iter().fold(
        0.0_f32,
        |m, &v| if v.is_finite() { m.max(v.abs()) } else { m },
    );
    let high_dr = plane_max > 30.0 * noise;

    let mut col_thr = vec![base_thr; nu];
    let mut row_thr = vec![base_thr; nv];
    if high_dr {
        for (iu, ct) in col_thr.iter_mut().enumerate() {
            slice_buf.clear();
            for iv in 0..nv {
                slice_buf.push(plane_vals[iv * nu + iu]);
            }
            let (cmad, cmax) = slice_trimmed_mad_and_max(&slice_buf, &mut scratch, noise);
            *ct = base_thr.max(cfg.ridge_snr * cmad).max(cfg.sinc_frac * cmax);
        }
        for (iv, rt) in row_thr.iter_mut().enumerate() {
            slice_buf.clear();
            slice_buf.extend_from_slice(&plane_vals[iv * nu..(iv + 1) * nu]);
            let (rmad, rmax) = slice_trimmed_mad_and_max(&slice_buf, &mut scratch, noise);
            *rt = base_thr.max(cfg.ridge_snr * rmad).max(cfg.sinc_frac * rmax);
        }
    }

    let mut out = Vec::new();
    for iv in 1..nv - 1 {
        for iu in 1..nu - 1 {
            let v0 = plane_vals[iv * nu + iu];
            let loc_thr = col_thr[iu].max(row_thr[iv]);
            if !v0.is_finite() || v0.abs() < loc_thr || (!cfg.allow_negative && v0 < 0.0) {
                continue;
            }
            let pos_mode = v0 >= 0.0;
            let mut is_ext = true;
            'nbr2: for dv in -1isize..=1 {
                for du in -1isize..=1 {
                    if du == 0 && dv == 0 {
                        continue;
                    }
                    let nv_val =
                        plane_vals[(iv as isize + dv) as usize * nu + (iu as isize + du) as usize];
                    let worse = if pos_mode { nv_val > v0 } else { nv_val < v0 };
                    let tie = nv_val == v0 && (dv, du) < (0, 0);
                    if worse || tie {
                        is_ext = false;
                        break 'nbr2;
                    }
                }
            }
            if !is_ext {
                continue;
            }

            let du = subvoxel_offset(
                plane_vals[iv * nu + (iu - 1)],
                v0,
                plane_vals[iv * nu + (iu + 1)],
            );
            let dv = subvoxel_offset(
                plane_vals[(iv - 1) * nu + iu],
                v0,
                plane_vals[(iv + 1) * nu + iu],
            );
            let dw = if p_idx >= 1 && p_idx + 1 < n_ortho {
                let wm = voxel_at(iu, iv, p_idx - 1);
                let wp = voxel_at(iu, iv, p_idx + 1);
                if (pos_mode && v0 >= wm && v0 >= wp) || (!pos_mode && v0 <= wm && v0 <= wp) {
                    subvoxel_offset(wm, v0, wp)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let (ix_f, iy_f, iz_f) = match ortho_axis {
                0 => (p_idx as f64 + dw, iu as f64 + du, iv as f64 + dv),
                1 => (iu as f64 + du, p_idx as f64 + dw, iv as f64 + dv),
                _ => (iu as f64 + du, iv as f64 + dv, p_idx as f64 + dw),
            };
            let f3_ppm = spec.axis_x.index_to_ppm(ix_f);
            let f2_ppm = spec.axis_y.index_to_ppm(iy_f);
            let f1_ppm = spec.axis_z.index_to_ppm(iz_f);

            if let Some((w_lo, w_hi)) = water_band {
                if (w_lo..=w_hi).contains(&f3_ppm) {
                    continue;
                }
            }
            if cfg.exclude_diagonal
                && is_homonuclear_diagonal_hit(f3_ppm, f2_ppm, f1_ppm, nucs, cfg.diagonal_tol_ppm)
            {
                continue;
            }

            let mut vol = 0.0_f64;
            for dv_i in -1isize..=1 {
                for du_i in -1isize..=1 {
                    let sv = plane_vals
                        [(iv as isize + dv_i) as usize * nu + (iu as isize + du_i) as usize];
                    if sv.signum() == v0.signum() {
                        vol += (sv.abs()) as f64;
                    }
                }
            }

            out.push(CubePeak3D {
                exp: spec.exp.clone(),
                f3_ppm,
                f2_ppm,
                f1_ppm,
                sign: if v0 < 0.0 { -1 } else { 1 },
                height: v0 as f64,
                volume: vol,
                snr: (v0.abs() / noise) as f64,
                label: String::new(),
            });
        }
    }

    out.sort_by(|a, b| b.height.abs().total_cmp(&a.height.abs()));
    if out.len() > cfg.max_peaks.max(1) {
        out.truncate(cfg.max_peaks.max(1));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::AxisPpm;

    fn render_two_peak_hsqc(obs_mhz_h: f64, dh_ppm: f64) -> Spectrum2D {
        // 1024 points across 2.0 ppm => 0.00195 ppm/pt (~0.98 Hz/pt at 500 MHz, ~1.56 Hz/pt at 800 MHz)
        let nx = 1024usize;
        let ny = 64usize;
        let axis_x = AxisPpm::from_bounds("1H", nx, 9.5, 7.5, obs_mhz_h);
        let axis_y = AxisPpm::from_bounds("15N", ny, 128.0, 112.0, obs_mhz_h * 0.101329);
        let mut data = vec![0.0f32; nx * ny];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (((i * 31 + 7) % 41) as f32 - 20.0) * 0.05;
        }
        let h_center = 8.40f64;
        let n_center = 120.0f64;
        for &h_pk in &[h_center - 0.5 * dh_ppm, h_center + 0.5 * dh_ppm] {
            let cx = axis_x.ppm_to_index(h_pk);
            let cy = axis_y.ppm_to_index(n_center);
            let ix0 = (cx.floor() as isize - 4).max(0) as usize;
            let ix1 = (cx.ceil() as isize + 4).min(nx as isize - 1) as usize;
            let iy0 = (cy.floor() as isize - 3).max(0) as usize;
            let iy1 = (cy.ceil() as isize + 3).min(ny as isize - 1) as usize;
            for iy in iy0..=iy1 {
                for ix in ix0..=ix1 {
                    let dx = (ix as f64 - cx) / 1.4;
                    let dy = (iy as f64 - cy) / 1.2;
                    data[iy * nx + ix] += 500.0 * (-0.5 * (dx * dx + dy * dy) as f32).exp();
                }
            }
        }
        Spectrum2D {
            name: "HSQC".to_string(),
            axis_x,
            axis_y,
            data,
        }
    }

    #[test]
    fn test_hsqc_j_doublet_threshold_scales_with_spectrometer_mhz() {
        let cfg = PickerConfig::default(); // max_j_hnha_hz = 9.5 Hz

        // 0.015 ppm separation = 7.5 Hz at 500 MHz (<= 9.5 Hz -> coalesced to 1 centroid)
        let spec_500 = render_two_peak_hsqc(500.0, 0.015);
        let pks_500 = pick_hsqc_2d(&spec_500, &cfg);
        assert_eq!(
            pks_500.len(),
            1,
            "0.015 ppm (7.5 Hz at 500 MHz) must coalesce into 1 doublet centroid, got {pks_500:?}"
        );
        assert!((pks_500[0].h - 8.40).abs() < 0.002);

        // Same 0.015 ppm separation = 12.0 Hz at 800 MHz (> 9.5 Hz -> two distinct resonances!)
        let spec_800 = render_two_peak_hsqc(800.0, 0.015);
        let pks_800 = pick_hsqc_2d(&spec_800, &cfg);
        assert_eq!(
            pks_800.len(),
            2,
            "0.015 ppm (12.0 Hz at 800 MHz) exceeds 9.5 Hz and must remain 2 distinct peaks, got {pks_800:?}"
        );
    }

    /// Synthetic `HCCCONH`-like cube with a `4.70 ppm` F3 water wall, a homonuclear `(8.20, 118.0, 8.20)`
    /// diagonal and 3 genuine sub-voxel cross-peaks. Axis `labels` / `sf_mhz` are caller-selected so the same
    /// physical cube can be presented with different (equally legitimate) label conventions.
    fn water_diag_cube(labels: (&str, &str, &str), sf_mhz: (f64, f64, f64)) -> Spectrum3D {
        let (nx, ny, nz) = (96usize, 24usize, 48usize);
        let axis_x = AxisPpm::from_bounds(labels.0, nx, 10.5, 1.0, sf_mhz.0);
        let axis_y = AxisPpm::from_bounds(labels.1, ny, 130.0, 107.0, sf_mhz.1);
        let axis_z = AxisPpm::from_bounds(labels.2, nz, 9.8, 0.4, sf_mhz.2);
        let mut data = vec![0.0_f32; nx * ny * nz];

        // Deterministic baseline noise
        let mut st = 0x1234_5678_9ABC_DEF0_u64;
        for cell in &mut data {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((st >> 32) as f32 / (u32::MAX as f32)) - 0.5;
            *cell = u * 4.0;
        }

        // Inject residual 4.70 ppm F3 water wall + homonuclear (8.20, 118.0, 8.20) diagonal peak
        let wx = axis_x.ppm_to_index(4.70);
        let dx_diag = axis_x.ppm_to_index(8.20);
        let dy_diag = axis_y.ppm_to_index(118.0);
        let dz_diag = axis_z.ppm_to_index(8.20);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dw = (ix as f64 - wx) / 1.1;
                    if dw.abs() <= 3.0 {
                        data[(iz * ny + iy) * nx + ix] += 9_000.0 * (-0.5 * dw * dw).exp() as f32;
                    }
                    let r2 = ((ix as f64 - dx_diag) / 1.3).powi(2)
                        + ((iy as f64 - dy_diag) / 1.1).powi(2)
                        + ((iz as f64 - dz_diag) / 1.3).powi(2);
                    if r2 <= 9.0 {
                        data[(iz * ny + iy) * nx + ix] += 6_000.0 * (-0.5 * r2).exp() as f32;
                    }
                }
            }
        }

        // Inject 3 genuine sub-voxel 3D cross-peaks (2 positive, 1 negative)
        let truth = [
            (8.42_f64, 121.3_f64, 4.18_f64, 1_450.0_f32),
            (7.91_f64, 115.6_f64, 1.42_f64, 980.0_f32),
            (8.42_f64, 121.3_f64, 2.86_f64, -760.0_f32),
        ];
        for &(f3, f2, f1, amp) in &truth {
            let cx = axis_x.ppm_to_index(f3);
            let cy = axis_y.ppm_to_index(f2);
            let cz = axis_z.ppm_to_index(f1);
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let r2 = ((ix as f64 - cx) / 1.25).powi(2)
                            + ((iy as f64 - cy) / 1.15).powi(2)
                            + ((iz as f64 - cz) / 1.25).powi(2);
                        if r2 <= 10.0 {
                            data[(iz * ny + iy) * nx + ix] += amp * (-0.5 * r2).exp() as f32;
                        }
                    }
                }
            }
        }

        Spectrum3D {
            name: "HCCCONH".to_string(),
            exp: "HCCCONH".to_string(),
            axis_x,
            axis_y,
            axis_z,
            data,
        }
    }

    /// `HNCA`-like cube (`F3 = 1H`, `F2 = 15N`, `F1 = 13C`) with a `4.70 ppm` F3 water wall and 3 sub-voxel
    /// peaks; axis labels are caller-selected (`20.W1:P1-1` label-convention regression).
    fn water_hnca_cube(labels: (&str, &str, &str)) -> Spectrum3D {
        let (nx, ny, nz) = (96usize, 24usize, 48usize);
        let axis_x = AxisPpm::from_bounds(labels.0, nx, 10.5, 1.0, 800.13);
        let axis_y = AxisPpm::from_bounds(labels.1, ny, 130.0, 107.0, 81.08);
        let axis_z = AxisPpm::from_bounds(labels.2, nz, 70.0, 40.0, 201.19);
        let mut data = vec![0.0_f32; nx * ny * nz];
        let mut st = 0x0BAD_C0FF_EE12_3456_u64;
        for cell in &mut data {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((st >> 32) as f32 / (u32::MAX as f32)) - 0.5;
            *cell = u * 4.0;
        }
        let wx = axis_x.ppm_to_index(4.70);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dw = (ix as f64 - wx) / 1.1;
                    if dw.abs() <= 3.0 {
                        data[(iz * ny + iy) * nx + ix] += 9_000.0 * (-0.5 * dw * dw).exp() as f32;
                    }
                }
            }
        }
        let truth = [
            (8.42_f64, 121.3_f64, 55.2_f64, 1_450.0_f32),
            (7.91_f64, 115.6_f64, 62.8_f64, 980.0_f32),
            (8.42_f64, 121.3_f64, 45.5_f64, -760.0_f32),
        ];
        for &(f3, f2, f1, amp) in &truth {
            let (cx, cy, cz) = (
                axis_x.ppm_to_index(f3),
                axis_y.ppm_to_index(f2),
                axis_z.ppm_to_index(f1),
            );
            for iz in 0..nz {
                for iy in 0..ny {
                    for ix in 0..nx {
                        let r2 = ((ix as f64 - cx) / 1.25).powi(2)
                            + ((iy as f64 - cy) / 1.15).powi(2)
                            + ((iz as f64 - cz) / 1.25).powi(2);
                        if r2 <= 10.0 {
                            data[(iz * ny + iy) * nx + ix] += amp * (-0.5 * r2).exp() as f32;
                        }
                    }
                }
            }
        }
        Spectrum3D {
            name: "HNCA".to_string(),
            exp: "HNCA".to_string(),
            axis_x,
            axis_y,
            axis_z,
            data,
        }
    }

    #[test]
    fn test_pick_cube_3d_ridge_safe_subvoxel_diagonal_water_and_plane_picker() {
        let cube = water_diag_cube(("1H", "15N", "1H"), (800.13, 81.08, 800.13));

        let picked = pick_cube_3d_ridge_safe(&cube, &Cube3dPickerConfig::default());
        assert_eq!(picked.len(), 3, "got {picked:?}");
        assert!(picked[0].height.abs() >= picked[1].height.abs());
        assert!(picked[1].height.abs() >= picked[2].height.abs());
        assert_eq!(picked[2].sign, -1);
        assert!((picked[0].f3_ppm - 8.42).abs() < 0.03);
        assert!((picked[0].f2_ppm - 121.3).abs() < 0.15);
        assert!((picked[0].f1_ppm - 4.18).abs() < 0.04);

        // Single-plane pick on the F2 (`15N = 121.3 ppm`) strip plane recovers the 2 peaks on that plane
        let iy_strip = cube.axis_y.ppm_to_index(121.3).round() as usize;
        let plane_pks =
            pick_cube_3d_plane_ridge_safe(&cube, 1, iy_strip, &Cube3dPickerConfig::default());
        assert_eq!(plane_pks.len(), 2, "got {plane_pks:?}");

        // 3D extremum snap from slightly perturbed coordinates recovers sub-voxel center
        let (sf3, sf2, sf1, sh, _, sgn) =
            snap_cube_3d_extremum(&cube, 8.36, 121.8, 4.05, [3, 2, 3], true);
        assert!((sf3 - 8.42).abs() < 0.03);
        assert!((sf2 - 121.3).abs() < 0.15);
        assert!((sf1 - 4.18).abs() < 0.04);
        assert!(sh > 1_200.0 && sgn == 1);
    }

    /// `20.W1:P1-1`: the same physical cube must be picked identically whether its axes carry the
    /// explicit isotope labels (`1H`/`15N`/`13CA`) or the ubiquitous amide-walk labels (`HN`/`N`/`CA`);
    /// `"HN"` is the amide PROTON, never `15N`, so the `4.70 ppm` water band and the homonuclear
    /// diagonal must be excluded in both runs (zero difference).
    #[test]
    fn test_cube_axis_labels_hn_n_ca_equal_1h_15n_13ca_with_water_artifact() {
        let cfg = Cube3dPickerConfig::default();
        // Z axis is the second proton dimension of the diagonal test cube; `CA`/`13CA` is only used for the
        // nucleus-inference regression below, so keep the picked cube physically identical (proton Z).
        let explicit = water_diag_cube(("1H", "15N", "1H"), (800.13, 81.08, 800.13));
        let walk = water_diag_cube(("HN", "N", "H"), (800.13, 81.08, 800.13));
        let ratio_only = water_diag_cube(("F3", "F2", "F1"), (800.13, 81.08, 800.13));

        assert_eq!(cube_axis_nuclei(&explicit), ("1H", "15N", "1H"));
        assert_eq!(cube_axis_nuclei(&walk), ("1H", "15N", "1H"));
        assert_eq!(cube_axis_nuclei(&ratio_only), ("1H", "15N", "1H"));

        let p_explicit = pick_cube_3d_ridge_safe(&explicit, &cfg);
        let p_walk = pick_cube_3d_ridge_safe(&walk, &cfg);
        let p_ratio = pick_cube_3d_ridge_safe(&ratio_only, &cfg);
        assert_eq!(p_explicit.len(), 3, "explicit labels: {p_explicit:?}");
        assert_eq!(
            p_walk.len(),
            p_explicit.len(),
            "`HN`/`N`/`H` labels must not add water-band false peaks: {p_walk:?}"
        );
        assert_eq!(p_ratio.len(), p_explicit.len(), "ratio-only: {p_ratio:?}");
        for ((a, b), c) in p_explicit.iter().zip(&p_walk).zip(&p_ratio) {
            for other in [b, c] {
                assert_eq!(a.sign, other.sign);
                assert!((a.f3_ppm - other.f3_ppm).abs() == 0.0);
                assert!((a.f2_ppm - other.f2_ppm).abs() == 0.0);
                assert!((a.f1_ppm - other.f1_ppm).abs() == 0.0);
                assert!((a.height - other.height).abs() == 0.0);
            }
        }
        // None of the picks may sit inside the 4.70 ppm water band (direct F3 axis).
        assert!(p_walk.iter().all(|p| (p.f3_ppm - 4.70).abs() > 0.06));

        // Literal label-set comparison on an `HNCA`-style cube: `1H/15N/13CA` vs `HN/N/CA` (and ratio-only).
        let h_a = water_hnca_cube(("1H", "15N", "13CA"));
        let h_b = water_hnca_cube(("HN", "N", "CA"));
        let h_c = water_hnca_cube(("F3", "F2", "F1"));
        assert_eq!(cube_axis_nuclei(&h_a), ("1H", "15N", "13C"));
        assert_eq!(cube_axis_nuclei(&h_b), ("1H", "15N", "13C"));
        assert_eq!(cube_axis_nuclei(&h_c), ("1H", "15N", "13C"));
        let (pa, pb, pc) = (
            pick_cube_3d_ridge_safe(&h_a, &cfg),
            pick_cube_3d_ridge_safe(&h_b, &cfg),
            pick_cube_3d_ridge_safe(&h_c, &cfg),
        );
        assert_eq!(pa.len(), 3, "explicit HNCA labels: {pa:?}");
        assert_eq!(pb.len(), pa.len(), "`HN`/`N`/`CA` HNCA cube: {pb:?}");
        assert_eq!(pc.len(), pa.len(), "ratio-only HNCA cube: {pc:?}");
        for ((a, b), c) in pa.iter().zip(&pb).zip(&pc) {
            for other in [b, c] {
                assert_eq!(a.sign, other.sign);
                assert!((a.f3_ppm - other.f3_ppm).abs() == 0.0);
                assert!((a.f2_ppm - other.f2_ppm).abs() == 0.0);
                assert!((a.f1_ppm - other.f1_ppm).abs() == 0.0);
                assert!((a.height - other.height).abs() == 0.0);
            }
        }
        assert!(pb.iter().all(|p| (p.f3_ppm - 4.70).abs() > 0.06));
    }

    /// `20.W1:P1-1`: label-set regression for the three-axis `HN`/`N`/`CA` convention, including the
    /// frequency-ratio-first rule (labels are only a fallback) shared with `workbook.rs`.
    #[test]
    fn test_infer_cube_axis_iso_label_and_ratio_rules() {
        let ax =
            |label: &str, sf: f64, hi: f64, lo: f64| AxisPpm::from_bounds(label, 64, hi, lo, sf);
        // Label-only fallback (no proton reference): `HN` is 1H, never 15N.
        assert_eq!(infer_cube_axis_iso(&ax("HN", 0.0, 10.5, 6.0), None), "1H");
        assert_eq!(
            infer_cube_axis_iso(&ax("N", 0.0, 130.0, 105.0), None),
            "15N"
        );
        assert_eq!(infer_cube_axis_iso(&ax("CA", 0.0, 70.0, 40.0), None), "13C");
        assert_eq!(infer_cube_axis_iso(&ax("<HN>", 0.0, 10.5, 6.0), None), "1H");
        assert_eq!(infer_cube_axis_iso(&ax("1H", 0.0, 10.5, 6.0), None), "1H");
        assert_eq!(
            infer_cube_axis_iso(&ax("15N", 0.0, 130.0, 105.0), None),
            "15N"
        );
        assert_eq!(
            infer_cube_axis_iso(&ax("13CA", 0.0, 70.0, 40.0), None),
            "13C"
        );
        // Frequency ratio wins over a misleading label (`HN` string on a 81 MHz axis is 15N).
        let h = 800.13;
        assert_eq!(
            infer_cube_axis_iso(&ax("HN", 81.08, 130.0, 105.0), Some(h)),
            "15N"
        );
        assert_eq!(
            infer_cube_axis_iso(&ax("N", 800.13, 10.5, 6.0), Some(h)),
            "1H"
        );
        assert_eq!(
            infer_cube_axis_iso(&ax("CA", 201.19, 70.0, 40.0), Some(h)),
            "13C"
        );
        assert_eq!(
            infer_cube_axis_iso(&ax("X", 752.9, -70.0, -140.0), Some(h)),
            "19F"
        );
    }
}
