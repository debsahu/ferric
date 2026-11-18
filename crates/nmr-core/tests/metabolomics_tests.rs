//! Comprehensive validation tests for Phase 14 Slice `T14.5`:
//! Clean-Room Untargeted 1D NMR Metabolomics, Adaptive Intelligent Binning,
//! PCA / OPLS-DA Chemometrics & STOCSY Biomarker Discovery (`nmr-core::metabolomics`).

use std::collections::{BTreeMap, BTreeSet};

use nmr_core::metabolomics::{
    align_cohort_interval_fft, bmrb_biofluid_metabolite_library, bucket_cohort_spectra,
    compute_stocsy_at_ppm, export_biomarker_ranking_csv, export_bucket_table_csv,
    export_study_summary_json, fit_opls_da, fit_pca, generate_demo_biofluid_cohort,
    lookup_bmrb_metabolite, normalize_bucket_matrix, preprocess_cohort_1d,
    run_metabolomics_pipeline, scale_bucket_matrix, synthesize_biofluid_spectrum_1d, AlignmentMode,
    AlignmentReference, BiofluidMaskPreset, BiofluidSynthOptions, BucketingMethod, CohortSample1D,
    ExclusionZonePpm, MetabolomicsPipelineConfig, MetabolomicsSummaryReportJson,
    NormalizationMethod, ScalingMethod,
};
use nmr_core::rng::Rng;

fn pearson_r(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let ma = a[..n].iter().sum::<f64>() / (n as f64);
    let mb = b[..n].iter().sum::<f64>() / (n as f64);
    let (mut num, mut da2, mut db2) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        num += da * db;
        da2 += da * da;
        db2 += db * db;
    }
    num / (da2 * db2).sqrt().max(1e-30)
}

/// [Test 1] PQN dilution-invariance test: 2x and 5x sample dilutions + massive 18x D-Glucose spike
/// must leave non-spiked metabolite concentrations accurate to `< 1%` under `PQN` (and `TSP`
/// internal standard), whereas `TSA` (Total Spectral Area) fails catastrophically (`> 40%` error).
#[test]
fn test_pqn_dilution_and_glucose_spike_invariance_vs_tsa_failure() {
    let mut rng = Rng::new(101);
    let bg_metabolites: &[(&str, f64)] = &[
        ("L-Lactate", 1.00),
        ("L-Alanine", 1.00),
        ("L-Valine", 1.00),
        ("Succinate", 1.00),
        ("Acetate", 1.00),
        ("Creatine", 1.00),
        ("Creatinine", 1.00),
        ("TMAO", 1.00),
        ("Glycine", 1.00),
        ("Formate", 1.00),
        ("Hippurate", 1.00),
    ];

    // 6 samples:
    // 0..3: normal dilution (1.0x, 1.0x, 0.5x [2x diluted], 0.2x [5x diluted]) with baseline 0.5 mM Glucose
    // 4: 0.5x (2x diluted) + 12x Glucose spike
    // 5: 0.2x (5x diluted) + 18x Glucose spike
    let specs_plan = [
        ("Ref_1x_A", 1.00, 0.50),
        ("Ref_1x_B", 1.00, 0.50),
        ("Dil_2x", 0.50, 0.50),
        ("Dil_5x", 0.20, 0.50),
        ("Dil_2x_Glc12x", 0.50, 6.00),
        ("Dil_5x_Glc18x", 0.20, 9.00),
    ];

    let mut samples = Vec::new();
    for &(label, dil, glc_c) in &specs_plan {
        let mut concs = bg_metabolites.to_vec();
        concs.push(("alpha-D-Glucose", 0.36 * glc_c));
        concs.push(("beta-D-Glucose", 0.64 * glc_c));
        let opts = BiofluidSynthOptions {
            high_ppm: 9.50,
            low_ppm: -0.30,
            n_points: 9_801,
            obs_mhz: 600.0,
            linewidth_hz: 1.4,
            dilution_factor: dil,
            // Internal standard added to NMR tube at fixed final concentration (0.5 mM) after urine collection:
            // wait: for internal-standard dilution correction when TSP is added *prior* to dilution (or scaled with dil),
            // let's scale TSP by `dil` so TSP tracks sample dilution too!
            tsp_concentration: 0.50 * dil,
            noise_sigma: 0.0005,
            ..Default::default()
        };
        let sp = synthesize_biofluid_spectrum_1d(label, &concs, &opts, &mut rng);
        samples.push(CohortSample1D::new(label, "Cohort", sp));
    }

    let cfg = MetabolomicsPipelineConfig {
        alignment: AlignmentMode::Disabled,
        mask_preset: BiofluidMaskPreset::Urine,
        bucketing: BucketingMethod::default(),
        ..Default::default()
    };
    let (grid, masked, aligned, is_areas, obs_mhz) = preprocess_cohort_1d(&samples, &cfg);
    let (buckets, raw_mat) =
        bucket_cohort_spectra(&aligned, &grid, &masked, obs_mhz, &cfg.bucketing);

    let (pqn_mat, _pqn_factors) = normalize_bucket_matrix(
        &raw_mat,
        &is_areas,
        NormalizationMethod::ProbabilisticQuotient,
    );
    let (tsa_mat, _) =
        normalize_bucket_matrix(&raw_mat, &is_areas, NormalizationMethod::TotalSpectralArea);
    let (is_mat, _) = normalize_bucket_matrix(
        &raw_mat,
        &is_areas,
        NormalizationMethod::InternalStandard {
            target_ppm: 0.0,
            half_window_ppm: 0.08,
        },
    );

    let mut max_pqn_err = 0.0_f64;
    let mut min_tsa_spike_err = f64::INFINITY;
    // Check non-spiked isolated buckets: Formate (8.448 ppm), Acetate (1.918 ppm), Alanine (1.478 ppm), Lactate (1.330 ppm)
    for target_ppm in [8.448, 1.918, 1.478, 1.330] {
        let b_idx = buckets
            .iter()
            .min_by(|a, b| {
                (a.center_ppm - target_ppm)
                    .abs()
                    .total_cmp(&(b.center_ppm - target_ppm).abs())
            })
            .expect("bucket exists")
            .index;
        assert!(
            (buckets[b_idx].center_ppm - target_ppm).abs() < 0.02,
            "Expected bucket near {target_ppm}"
        );

        let ref_pqn = pqn_mat[0][b_idx];
        let ref_tsa = tsa_mat[0][b_idx];
        let ref_is = is_mat[0][b_idx];

        for sample_i in 1..samples.len() {
            let pqn_rel_err = ((pqn_mat[sample_i][b_idx] - ref_pqn) / ref_pqn).abs();
            assert!(
                pqn_rel_err < 0.01,
                "PQN error on sample {} @ {:.3} ppm = {:.4} (>= 1%)",
                specs_plan[sample_i].0,
                target_ppm,
                pqn_rel_err
            );
            max_pqn_err = max_pqn_err.max(pqn_rel_err);

            let is_rel_err = ((is_mat[sample_i][b_idx] - ref_is) / ref_is).abs();
            assert!(
                is_rel_err < 0.01,
                "Internal-standard error on sample {} @ {:.3} ppm = {:.4} (>= 1%)",
                specs_plan[sample_i].0,
                target_ppm,
                is_rel_err
            );
        }

        // Verify TSA fails badly (> 40% distortion) on the 5x-diluted + 18x-Glucose-spiked sample!
        let tsa_spike_err = ((tsa_mat[5][b_idx] - ref_tsa) / ref_tsa).abs();
        assert!(
            tsa_spike_err > 0.40,
            "TSA should fail by >40% under 18x Glucose spike, got {:.4}",
            tsa_spike_err
        );
        min_tsa_spike_err = min_tsa_spike_err.min(tsa_spike_err);
    }
    println!(
        "W2_METRIC_JSON: {}",
        serde_json::json!({
            "kind": "metab_pqn_vs_tsa",
            "max_pqn_rel_err": max_pqn_err,
            "min_tsa_spike_rel_err": min_tsa_spike_err,
            "passed": true,
        })
    );
}

/// [Test 2] Interval FFT cross-correlation alignment rescues pH-shifted Citrate (`2.54 / 2.68 ppm`)
/// and L-Histidine (`7.09 / 7.87 ppm`) peaks without perturbing pH-stable aliphatic doublets.
#[test]
fn test_interval_fft_alignment_rescues_ph_shifted_citrate_and_histidine() {
    let mut rng = Rng::new(202);
    let ph_jitters = [
        -0.030, -0.024, -0.016, -0.008, 0.0, 0.0, 0.009, 0.017, 0.025, 0.032,
    ];

    let mut samples = Vec::new();
    for (i, &dj) in ph_jitters.iter().enumerate() {
        let mut offsets = BTreeMap::new();
        offsets.insert("Citrate".to_string(), dj);
        offsets.insert("L-Histidine".to_string(), 1.25 * dj);

        let concs = [
            ("Citrate", 1.50),
            ("L-Histidine", 1.20),
            ("L-Alanine", 1.00),
            ("L-Valine", 1.00),
        ];
        let opts = BiofluidSynthOptions {
            high_ppm: 8.60,
            low_ppm: 0.60,
            n_points: 8_001,
            obs_mhz: 600.0,
            linewidth_hz: 1.5,
            metabolite_shift_offsets: offsets,
            noise_sigma: 0.001,
            ..Default::default()
        };
        let sp = synthesize_biofluid_spectrum_1d(format!("S{i:02}"), &concs, &opts, &mut rng);
        samples.push(CohortSample1D::new(format!("S{i:02}"), "Urine", sp));
    }

    let cfg = MetabolomicsPipelineConfig {
        min_ppm: 0.70,
        max_ppm: 8.50,
        grid_step_ppm: 0.001,
        mask_preset: BiofluidMaskPreset::AqueousMinimal,
        alignment: AlignmentMode::AutomaticIntervals {
            interval_width_ppm: 0.28,
            max_shift_ppm: 0.06,
            reference: AlignmentReference::Median,
        },
        ..Default::default()
    };

    let (grid, masked, unaligned, _, _) = preprocess_cohort_1d(&samples, &cfg);
    let mut aligned = unaligned.clone();
    align_cohort_interval_fft(&mut aligned, &grid, &masked, &cfg.alignment);

    let region_mean_corr = |mat: &[Vec<f64>], lo_ppm: f64, hi_ppm: f64| -> f64 {
        let i0 = grid.partition_point(|&p| p > hi_ppm);
        let i1 = grid.partition_point(|&p| p >= lo_ppm).saturating_sub(1);
        let ref_row = &mat[4][i0..=i1]; // unshifted pH-neutral sample
        let mut sum_r = 0.0;
        let mut cnt = 0usize;
        for (idx, r) in mat.iter().enumerate() {
            if idx != 4 {
                sum_r += pearson_r(ref_row, &r[i0..=i1]);
                cnt += 1;
            }
        }
        sum_r / (cnt as f64)
    };

    let cit_before = region_mean_corr(&unaligned, 2.45, 2.76);
    let cit_after = region_mean_corr(&aligned, 2.45, 2.76);
    let his_before = region_mean_corr(&unaligned, 7.76, 7.98);
    let his_after = region_mean_corr(&aligned, 7.76, 7.98);
    let ala_after = region_mean_corr(&aligned, 1.42, 1.54);

    assert!(
        cit_before < 0.45 && cit_after > 0.96,
        "Citrate alignment failed: before={cit_before:.4}, after={cit_after:.4}"
    );
    assert!(
        his_before < 0.40 && his_after > 0.96,
        "Histidine alignment failed: before={his_before:.4}, after={his_after:.4}"
    );
    assert!(
        ala_after > 0.99,
        "pH-stable Alanine doublet should remain untouched (>0.99), got {ala_after:.4}"
    );
}

/// [Test 3] Adaptive Intelligent Binning (`De Meyer et al. 2008`) keeps J-coupled multiplets
/// (Lactate `4.11 ppm` quartet & `1.33 ppm` doublet, Valine `0.985` & `1.037 ppm` doublets)
/// inside single valley-bounded buckets, whereas uniform `0.04 ppm` binning splits multiplets.
#[test]
fn test_adaptive_intelligent_binning_preserves_j_multiplets_vs_uniform_splitting() {
    let mut rng = Rng::new(303);
    let concs = [("L-Lactate", 1.50), ("L-Valine", 1.50)];
    let opts = BiofluidSynthOptions {
        high_ppm: 9.50,
        low_ppm: 0.50,
        n_points: 9_001,
        obs_mhz: 600.0,
        linewidth_hz: 1.3,
        noise_sigma: 0.0005,
        ..Default::default()
    };
    let samples: Vec<CohortSample1D> = (0..4)
        .map(|i| {
            let sp = synthesize_biofluid_spectrum_1d(format!("M{i}"), &concs, &opts, &mut rng);
            CohortSample1D::new(format!("M{i}"), "G", sp)
        })
        .collect();

    let base_cfg = MetabolomicsPipelineConfig {
        min_ppm: 0.50,
        max_ppm: 9.50,
        grid_step_ppm: 0.001,
        mask_preset: BiofluidMaskPreset::Custom(vec![ExclusionZonePpm::new(4.71, 4.95)]),
        alignment: AlignmentMode::Disabled,
        ..Default::default()
    };
    let (grid, masked, specs, _, obs_mhz) = preprocess_cohort_1d(&samples, &base_cfg);

    let (uni_buckets, uni_mat) = bucket_cohort_spectra(
        &specs,
        &grid,
        &masked,
        obs_mhz,
        &BucketingMethod::Uniform {
            bin_width_ppm: 0.04,
            min_bucket_snr: 0.0,
        },
    );
    let (aib_buckets, aib_mat) =
        bucket_cohort_spectra(&specs, &grid, &masked, obs_mhz, &BucketingMethod::default());

    // Helper: fraction of total regional integral [lo, hi] captured by the single largest overlapping bucket
    let max_single_bucket_share = |buckets: &[nmr_core::metabolomics::SpectralBucket],
                                   mat: &[Vec<f64>],
                                   lo: f64,
                                   hi: f64|
     -> (f64, usize) {
        let step = 0.001;
        let i0 = grid.partition_point(|&p| p > hi);
        let i1 = grid.partition_point(|&p| p >= lo).saturating_sub(1);
        let mut true_area = 0.0;
        for k in i0..i1 {
            true_area += 0.5 * (specs[0][k] + specs[0][k + 1]) * step;
        }
        let mut best_area = 0.0;
        let mut best_idx = 0;
        for b in buckets {
            if b.high_ppm >= lo && b.low_ppm <= hi && mat[0][b.index] > best_area {
                best_area = mat[0][b.index];
                best_idx = b.index;
            }
        }
        (best_area / true_area.max(1e-15), best_idx)
    };

    // Uniform 0.04 ppm grid from 9.50 ppm places boundaries at 4.10 ppm (splitting Lactate 4.11 quartet!)
    // and at 0.98 ppm (splitting Valine 0.985 doublet!)
    let (uni_lac_q_share, _) = max_single_bucket_share(&uni_buckets, &uni_mat, 4.075, 4.145);
    let (uni_val_1_share, _) = max_single_bucket_share(&uni_buckets, &uni_mat, 0.962, 1.008);
    assert!(
        uni_lac_q_share < 0.78 && uni_val_1_share < 0.80,
        "Uniform 0.04 ppm binning should split Lactate quartet ({uni_lac_q_share:.3}) and Valine 0.985 doublet ({uni_val_1_share:.3})"
    );

    // Adaptive Intelligent Binning keeps each multiplet intact inside a single bucket (> 95% share)
    let (aib_lac_q_share, idx_lac_q) =
        max_single_bucket_share(&aib_buckets, &aib_mat, 4.075, 4.145);
    let (aib_lac_d_share, idx_lac_d) =
        max_single_bucket_share(&aib_buckets, &aib_mat, 1.305, 1.355);
    let (aib_val_1_share, idx_val_1) =
        max_single_bucket_share(&aib_buckets, &aib_mat, 0.962, 1.008);
    let (aib_val_2_share, idx_val_2) =
        max_single_bucket_share(&aib_buckets, &aib_mat, 1.014, 1.060);

    assert!(
        aib_lac_q_share > 0.95
            && aib_lac_d_share > 0.95
            && aib_val_1_share > 0.95
            && aib_val_2_share > 0.95,
        "AIB must keep J-multiplets whole: Lac_q={aib_lac_q_share:.3}, Lac_d={aib_lac_d_share:.3}, Val1={aib_val_1_share:.3}, Val2={aib_val_2_share:.3}"
    );
    assert_ne!(
        idx_val_1, idx_val_2,
        "Valine gamma1 (0.985 ppm) and gamma2 (1.037 ppm) doublets must occupy distinct valley-split buckets"
    );
    assert_ne!(idx_lac_q, idx_lac_d);
    println!(
        "W2_METRIC_JSON: {}",
        serde_json::json!({
            "kind": "metab_aib_vs_uniform",
            "aib_lac_q_share": aib_lac_q_share,
            "aib_val_1_share": aib_val_1_share,
            "uni_lac_q_share": uni_lac_q_share,
            "uni_val_1_share": uni_val_1_share,
            "passed": true,
        })
    );
}

/// [Test 4] Solvent, residual water (`4.80 ppm`), and urinary urea (`5.78 ppm`) region masking.
#[test]
fn test_water_and_urea_region_masking_excludes_solvent_artefacts() {
    let mut rng = Rng::new(404);
    let opts = BiofluidSynthOptions {
        residual_water_amp: 400.0,
        urea_amp: 180.0,
        tsp_concentration: 2.0,
        ..Default::default()
    };
    let sp = synthesize_biofluid_spectrum_1d(
        "U1",
        &[("L-Lactate", 1.0), ("Formate", 1.0)],
        &opts,
        &mut rng,
    );
    let samples = vec![
        CohortSample1D::new("U1", "Ctrl", sp.clone()),
        CohortSample1D::new("U2", "Ctrl", sp),
    ];

    let cfg = MetabolomicsPipelineConfig {
        mask_preset: BiofluidMaskPreset::Urine,
        alignment: AlignmentMode::Disabled,
        ..Default::default()
    };
    let res = run_metabolomics_pipeline(&samples, &cfg);
    assert!(res.buckets.len() >= 3);
    for b in &res.buckets {
        let overlaps_water = b.high_ppm > 4.65 && b.low_ppm < 5.05;
        let overlaps_urea = b.high_ppm > 5.55 && b.low_ppm < 6.05;
        assert!(
            !overlaps_water && !overlaps_urea,
            "Bucket [{:.3}, {:.3}] leaked into water/urea exclusion zone!",
            b.low_ppm,
            b.high_ppm
        );
    }
}

/// [Test 5] Deterministic PCA score/loading reconstruction + Hotelling $T^2$ leverage outlier and
/// Jackson–Mudholkar $Q$-residual (`SPE`) off-subspace contaminant detection.
#[test]
fn test_pca_svd_reconstruction_and_hotelling_t2_q_residual_outlier_detection() {
    let mut rng = Rng::new(505);
    let n = 22usize;
    let p = 40usize;

    // 2-factor ground-truth covariance manifold + Sample 20 (7x leverage outlier on Factor 1)
    // + Sample 21 (off-subspace contaminant spike on columns 35..40)
    let mut raw = vec![vec![0.0; p]; n];
    let mut groups = Vec::with_capacity(n);
    for (i, row) in raw.iter_mut().enumerate() {
        groups.push(if i % 2 == 0 { "G0" } else { "G1" }.to_string());
        let z1 = if i == 20 { 8.5 } else { rng.normal(0.0, 1.0) };
        let z2 = rng.normal(0.0, 0.8);
        for (j, slot) in row.iter_mut().enumerate() {
            let v1 = ((j as f64) * 0.21).sin();
            let v2 = ((j as f64) * 0.37).cos();
            *slot = 10.0 + 2.5 * z1 * v1 + 1.8 * z2 * v2 + rng.normal(0.0, 0.04);
        }
        if i == 21 {
            // Foreign contaminant resonances absent from normal cohort subspace
            for slot in &mut row[35..40] {
                *slot += 9.0;
            }
        }
    }

    let scaled = scale_bucket_matrix(&raw, ScalingMethod::MeanCenter);
    let pca = fit_pca(&scaled, &groups, 2);

    // Check loading orthonormality P^T P = I_2
    let dot00: f64 = pca.loadings[0].iter().map(|&v| v * v).sum();
    let dot11: f64 = pca.loadings[1].iter().map(|&v| v * v).sum();
    let dot01: f64 = pca.loadings[0]
        .iter()
        .zip(pca.loadings[1].iter())
        .map(|(&a, &b)| a * b)
        .sum();
    assert!((dot00 - 1.0).abs() < 1e-12);
    assert!((dot11 - 1.0).abs() < 1e-12);
    assert!(dot01.abs() < 1e-12);

    // Sample 20 must exceed the 95% Hotelling T^2 limit
    assert!(
        pca.hotelling_t2[20] > pca.hotelling_t2_limit_95 * 2.0,
        "Leverage outlier sample 20 not flagged by Hotelling T^2: {:.2} vs limit {:.2}",
        pca.hotelling_t2[20],
        pca.hotelling_t2_limit_95
    );

    // Sample 21 must exceed the 95% Q-residual (SPE) limit while normal samples stay well below
    assert!(
        pca.q_residuals[21] > pca.q_residual_limit_95 * 3.0,
        "Contaminant sample 21 not flagged by Q-residual: {:.2} vs limit {:.2}",
        pca.q_residuals[21],
        pca.q_residual_limit_95
    );
    let median_clean_q = pca.q_residuals[..20].iter().sum::<f64>() / 20.0;
    assert!(median_clean_q < pca.q_residual_limit_95);

    // Full-rank PCA reconstruction check (n - 1 = 21 components reconstructs centered 22-row matrix exactly)
    let pca_full = fit_pca(&scaled, &groups, 21);
    let max_recon_q = pca_full.q_residuals.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        max_recon_q < 1e-18,
        "Full-rank PCA reconstruction residual should vanish, got {max_recon_q:e}"
    );
}

/// [Test 6] OPLS-DA (`Trygg & Wold 2002`; `Bylesjö et al. 2006`) cleanly separates a predictive class
/// shift (`Q^2 > 0.85`, `p < 0.02`) from massive orthogonal within-group variation (`4x` larger
/// variance that hijacks unsupervised `PC1`).
#[test]
fn test_opls_da_separates_predictive_shift_from_large_orthogonal_variation() {
    let mut rng = Rng::new(606);
    let n = 24usize;
    let p = 80usize;
    let mut x_mat = vec![vec![0.0; p]; n];
    let mut groups = Vec::with_capacity(n);

    for (i, row) in x_mat.iter_mut().enumerate() {
        let is_case = i >= 12;
        groups.push(if is_case { "Case" } else { "Control" }.to_string());
        // Huge within-group orthogonal factor shared identically by Control and Case
        let ortho_drift = if i % 2 == 0 { 4.5 } else { -4.5 } + rng.normal(0.0, 0.5);
        let class_signal = if is_case { 1.0 } else { -1.0 };

        for (j, slot) in row.iter_mut().enumerate() {
            let ortho_loading = ((j as f64) * 0.17).cos();
            let pred_loading = if (10..16).contains(&j) { 1.35 } else { 0.0 };
            *slot = 20.0
                + ortho_drift * ortho_loading
                + class_signal * pred_loading
                + rng.normal(0.0, 0.18);
        }
    }

    // Unsupervised PCA PC1 captures the huge orthogonal drift rather than class separation
    let scaled = scale_bucket_matrix(&x_mat, ScalingMethod::Pareto);
    let pca = fit_pca(&scaled, &groups, 3);
    let y_bin: Vec<f64> = (0..n).map(|i| if i >= 12 { 1.0 } else { 0.0 }).collect();
    let pc1_vals: Vec<f64> = pca.scores.iter().map(|s| s[0]).collect();
    assert!(
        pearson_r(&pc1_vals, &y_bin).abs() < 0.25,
        "Unsupervised PC1 should be dominated by 4.5x orthogonal variation"
    );

    // Supervised OPLS-DA isolates orthogonal drift into t_o1 (corr(t_o1, y) == 0) and achieves Q^2 > 0.85!
    let opls = fit_opls_da(
        &x_mat,
        &groups,
        ScalingMethod::Pareto,
        1,
        7,
        100,
        20260606,
        Some("Control"),
        Some("Case"),
    )
    .expect("OPLS-DA fit");

    let corr_to_y = pearson_r(&opls.orthogonal_scores[0], &opls.y_binary).abs();
    assert!(
        corr_to_y < 1e-10,
        "Orthogonal score t_o1 must be strictly orthogonal to class vector y, got {corr_to_y:e}"
    );
    assert!(
        opls.r2_x_orthogonal > opls.r2_x_predictive * 2.5,
        "Expected R2X_ortho ({:.3}) >> R2X_pred ({:.3})",
        opls.r2_x_orthogonal,
        opls.r2_x_predictive
    );
    assert!(
        opls.q2 > 0.85,
        "Expected stratified CV Q^2 > 0.85, got {:.4}",
        opls.q2
    );
    assert!(
        opls.permutation.p_value_q2 <= 0.02 && opls.permutation.is_validated,
        "Expected permutation p <= 0.02 and validated=true, got p={:.4}",
        opls.permutation.p_value_q2
    );
    println!(
        "W2_METRIC_JSON: {}",
        serde_json::json!({
            "kind": "metab_oplsda_ortho",
            "r2_x_predictive": opls.r2_x_predictive,
            "r2_x_orthogonal": opls.r2_x_orthogonal,
            "r2_y": opls.r2_y,
            "q2": opls.q2,
            "perm_p_q2": opls.permutation.p_value_q2,
            "passed": true,
        })
    );
}

/// [Test 7] Anti-overfitting check (`Westerhuis et al. 2008`): on a high-dimensional (`P = 200 >> N = 24`)
/// cohort with purely random null class labels, leak-free in-fold stratified CV must yield
/// `Q^2 < 0.10` and permutation test must yield `p > 0.10` (`is_validated == false`).
#[test]
fn test_opls_da_null_labels_anti_overfitting_check() {
    let mut rng = Rng::new(707);
    let n = 24usize;
    let p = 200usize;
    let mut x_null = vec![vec![0.0; p]; n];
    for row in &mut x_null {
        let latent = rng.normal(0.0, 1.5);
        for (j, slot) in row.iter_mut().enumerate() {
            *slot = 5.0 + 0.3 * latent * ((j as f64) * 0.11).sin() + rng.normal(0.0, 1.0);
        }
    }
    // Purely arbitrary balanced labels completely independent of x_null
    let mut groups: Vec<String> = (0..n)
        .map(|i| if i < 12 { "Control" } else { "Case" }.to_string())
        .collect();
    rng.shuffle(&mut groups);

    let opls = fit_opls_da(
        &x_null,
        &groups,
        ScalingMethod::Pareto,
        1,
        6,
        120,
        20260707,
        Some("Control"),
        Some("Case"),
    )
    .expect("OPLS-DA null fit");

    // In P = 200 >> N = 24, in-sample R2Y is high (> 0.7), proving why R2Y alone is misleading,
    // whereas out-of-fold CV Q^2 is < 0.10 and permutation p > 0.10!
    assert!(
        opls.q2 < 0.10,
        "Leak-free CV Q^2 on null labels must be < 0.10, got {:.4}",
        opls.q2
    );
    assert!(
        opls.permutation.p_value_q2 > 0.10,
        "Permutation p-value on null labels must be > 0.10, got {:.4}",
        opls.permutation.p_value_q2
    );
    assert!(
        !opls.permutation.is_validated,
        "Null model must NEVER be flagged as validated!"
    );
    println!(
        "W2_METRIC_JSON: {}",
        serde_json::json!({
            "kind": "metab_oplsda_null",
            "in_sample_r2_y": opls.r2_y,
            "null_q2": opls.q2,
            "null_perm_p_q2": opls.permutation.p_value_q2,
            "is_validated": opls.permutation.is_validated,
            "passed": true,
        })
    );
}

/// [Test 8] Verify exact Krylov-subspace equivalence between `NIPALS PLS-DA(A = 2)` and
/// `OPLS-DA(1 predictive + 1 orthogonal)` on single-$y$ discriminant regression.
#[test]
fn test_pls_da_and_opls_da_prediction_subspace_equivalence() {
    let cohort = generate_demo_biofluid_cohort(808);
    let cfg = MetabolomicsPipelineConfig {
        pls_components: 2,
        opls_orthogonal_components: 1,
        n_permutations: 20,
        ..Default::default()
    };
    let study = run_metabolomics_pipeline(&cohort, &cfg);
    let pls = study.pls_da.as_ref().expect("PLS-DA");
    let opls = study.opls_da.as_ref().expect("OPLS-DA");

    assert!((pls.r2_y - opls.r2_y).abs() < 1e-10);
    for (yp, yo) in pls.y_fitted.iter().zip(opls.y_fitted.iter()) {
        assert!(
            (yp - yo).abs() < 1e-10,
            "PLS(2) and OPLS(1+1) fitted responses diverged: {yp} vs {yo}"
        );
    }
}

/// [Test 9] `S-plot` + `VIP` + `BH-FDR` Volcano recover exact spiked biomarkers (`L-Lactate +3.0x`,
/// `L-Valine +2.0x`, `TMAO -2.5x`) in a 24-spectrum synthetic biofluid cohort with `100%` top-rank precision.
#[test]
fn test_s_plot_vip_and_bh_fdr_volcano_recover_spiked_biomarkers_100_pct_precision() {
    let cohort = generate_demo_biofluid_cohort(909);
    let cfg = MetabolomicsPipelineConfig {
        n_permutations: 100,
        rng_seed: 9090,
        ..Default::default()
    };
    let study = run_metabolomics_pipeline(&cohort, &cfg);
    let opls = study.opls_da.as_ref().expect("OPLS-DA present");
    assert!(
        opls.q2 > 0.85 && opls.permutation.p_value_q2 <= 0.02,
        "24-sample demo biofluid OPLS-DA failed validation: Q2={:.3}, p={:.3}",
        opls.q2,
        opls.permutation.p_value_q2
    );

    // Ground-truth biomarker resonances:
    // L-Lactate: 1.330, 4.110 ppm
    // L-Valine:  0.985, 1.037, 2.268, 3.605 ppm
    // TMAO:      3.266 ppm
    let truth_shifts: &[(&str, f64, f64)] = &[
        ("L-Lactate", 1.330, 1.0),
        ("L-Lactate", 4.110, 1.0),
        ("L-Valine", 0.985, 1.0),
        ("L-Valine", 1.037, 1.0),
        ("L-Valine", 2.268, 1.0),
        ("L-Valine", 3.605, 1.0),
        ("TMAO", 3.266, -1.0),
    ];

    // 100% top-rank precision across top 6 ranked candidate buckets!
    assert!(study.biomarkers.len() >= 15);
    let mut recovered_names = BTreeSet::new();
    for cand in study.biomarkers.iter().take(6) {
        let hit = truth_shifts
            .iter()
            .find(|&&(_, ppm, _)| (cand.center_ppm - ppm).abs() <= 0.03);
        assert!(
            hit.is_some(),
            "False-positive bucket in top-6 ranks: rank {} @ {:.3} ppm (VIP={:.2}, q={:.2e})",
            cand.rank,
            cand.center_ppm,
            cand.vip,
            cand.fdr_q
        );
        let &(met_name, _, expected_sign) = hit.unwrap();
        recovered_names.insert(met_name);
        assert!(
            cand.vip > 1.2 && cand.s_plot_corr.abs() > 0.75 && cand.fdr_q < 1e-3,
            "Weak stats on true biomarker {met_name} @ {:.3} ppm: VIP={:.2}, p(corr)={:.3}, q={:.2e}",
            cand.center_ppm,
            cand.vip,
            cand.s_plot_corr,
            cand.fdr_q
        );
        assert!(
            cand.log2_fold_change * expected_sign > 0.5,
            "Wrong fold-change direction for {met_name}: log2_fc={:.3}",
            cand.log2_fold_change
        );
    }

    // Verify all 3 spiked metabolites (Lactate, Valine, TMAO) are recovered in the top 7 biomarkers
    for cand in study.biomarkers.iter().take(7) {
        for &(m_name, ppm, _) in truth_shifts {
            if (cand.center_ppm - ppm).abs() <= 0.03 {
                recovered_names.insert(m_name);
            }
        }
    }
    assert_eq!(
        recovered_names.len(),
        3,
        "Expected all 3 spiked biomarkers {{L-Lactate, L-Valine, TMAO}}, got {:?}",
        recovered_names
    );
}

/// [Test 10] `STOCSY` (`Cloarec et al. 2005`) driven from the L-Lactate `1.33 ppm` doublet automatically
/// recovers its intramolecular `4.11 ppm` quartet (`r^2 > 0.90`, `3 : 1` covariance stoichiometry)
/// and ranks `BMRB` `L-Lactate` (`bmse000036`) #1.
#[test]
fn test_stocsy_lactate_driver_recovers_411_quartet_and_matches_bmrb_lactate() {
    let cohort = generate_demo_biofluid_cohort(1010);
    let cfg = MetabolomicsPipelineConfig {
        n_permutations: 20,
        ..Default::default()
    };
    let study = run_metabolomics_pipeline(&cohort, &cfg);

    let stocsy = compute_stocsy_at_ppm(1.330, &study).expect("STOCSY at 1.33 ppm");
    assert!((stocsy.driver_ppm - 1.330).abs() < 0.02);

    let q_bucket = study
        .buckets
        .iter()
        .min_by(|a, b| {
            (a.center_ppm - 4.110)
                .abs()
                .total_cmp(&(b.center_ppm - 4.110).abs())
        })
        .expect("4.11 ppm quartet bucket");
    assert!((q_bucket.center_ppm - 4.110).abs() < 0.02);

    let r2_quartet = stocsy.bucket_r2[q_bucket.index];
    let r_quartet = stocsy.bucket_correlation[q_bucket.index];
    assert!(
        r_quartet > 0.95 && r2_quartet > 0.90,
        "STOCSY from 1.33 ppm failed to recover 4.11 ppm quartet with r^2 > 0.90: r={r_quartet:.4}, r^2={r2_quartet:.4}"
    );

    // Stoichiometric 3H (doublet) : 1H (quartet) covariance ratio check
    let cov_133 = stocsy.bucket_covariance[stocsy.driver_bucket_index];
    let cov_411 = stocsy.bucket_covariance[q_bucket.index];
    let stoich_ratio = cov_133 / cov_411.max(1e-18);
    assert!(
        (2.3..=3.7).contains(&stoich_ratio),
        "Expected ~3:1 STOCSY covariance stoichiometry between Lactate CH3 and CH, got {stoich_ratio:.2}"
    );

    // Unrelated background metabolites (Alanine 1.478 ppm, Formate 8.448 ppm) must have low r^2
    for bg_ppm in [1.478, 8.448] {
        let bg_b = study
            .buckets
            .iter()
            .min_by(|a, b| {
                (a.center_ppm - bg_ppm)
                    .abs()
                    .total_cmp(&(b.center_ppm - bg_ppm).abs())
            })
            .unwrap();
        assert!(
            stocsy.bucket_r2[bg_b.index] < 0.25,
            "Unrelated metabolite @ {bg_ppm} ppm had spurious STOCSY r^2 = {:.3}",
            stocsy.bucket_r2[bg_b.index]
        );
    }

    // Top BMRB match must be L-Lactate (bmse000036)
    let top_match = stocsy
        .metabolite_matches
        .first()
        .expect("At least one BMRB match");
    assert_eq!(top_match.name, "L-Lactate");
    assert_eq!(top_match.bmrb_id, "bmse000036");
    assert_eq!(top_match.matched_multiplets, 2);
    assert!(top_match.score > 0.85);
    println!(
        "W2_METRIC_JSON: {}",
        serde_json::json!({
            "kind": "metab_stocsy_lactate",
            "driver_ppm": stocsy.driver_ppm,
            "quartet_ppm": q_bucket.center_ppm,
            "quartet_r": r_quartet,
            "quartet_r2": r2_quartet,
            "stoich_ratio_3_to_1": stoich_ratio,
            "top_bmrb_id": top_match.bmrb_id,
            "top_score": top_match.score,
            "passed": true,
        })
    );
}

/// [Test 11] Built-in CC0/BMRB human biofluid library completeness (`>= 25` required canonical entries)
/// and `STOCSY` disambiguation of overlapping `~3.26 ppm` resonances (`TMAO` singlet vs `Taurine` triplets).
#[test]
fn test_bmrb_library_completeness_and_tmao_vs_taurine_stocsy_disambiguation() {
    let lib = bmrb_biofluid_metabolite_library();
    assert!(
        lib.len() >= 26,
        "Expected >= 26 BMRB biofluid entries, got {}",
        lib.len()
    );

    let required_names = [
        "L-Lactate",
        "L-Alanine",
        "L-Valine",
        "L-Leucine",
        "L-Isoleucine",
        "alpha-D-Glucose",
        "beta-D-Glucose",
        "Citrate",
        "Succinate",
        "Pyruvate",
        "Acetate",
        "Acetoacetate",
        "3-Hydroxybutyrate",
        "Creatine",
        "Creatinine",
        "TMAO",
        "Taurine",
        "Glycine",
        "L-Glutamine",
        "L-Glutamate",
        "L-Histidine",
        "L-Phenylalanine",
        "L-Tyrosine",
        "Formate",
        "Hippurate",
        "Myo-inositol",
    ];
    for name in required_names {
        assert!(
            lookup_bmrb_metabolite(name).is_some(),
            "Missing required canonical biofluid metabolite: {name}"
        );
    }

    // Cohort where Taurine (3.260 + 3.425 ppm) varies strongly while TMAO is absent/flat -> Taurine ranks #1
    let mut rng = Rng::new(1111);
    let mut tau_samples = Vec::new();
    for i in 0..14 {
        let tau_c = 0.5 + (i as f64) * 0.25;
        let concs = [
            ("Taurine", tau_c),
            ("L-Alanine", 1.0 + rng.normal(0.0, 0.08)),
            ("Creatine", 1.0 + rng.normal(0.0, 0.08)),
        ];
        let sp = synthesize_biofluid_spectrum_1d(
            format!("Tau_{i}"),
            &concs,
            &BiofluidSynthOptions {
                noise_sigma: 0.001,
                ..Default::default()
            },
            &mut rng,
        );
        tau_samples.push(CohortSample1D::new(
            format!("Tau_{i}"),
            if i < 7 { "Control" } else { "Case" },
            sp,
        ));
    }
    let tau_study = run_metabolomics_pipeline(
        &tau_samples,
        &MetabolomicsPipelineConfig {
            n_permutations: 10,
            ..Default::default()
        },
    );
    let st_tau = compute_stocsy_at_ppm(3.260, &tau_study).expect("STOCSY @ 3.26 ppm");
    assert_eq!(st_tau.metabolite_matches[0].name, "Taurine");
    assert_eq!(st_tau.metabolite_matches[0].matched_multiplets, 2);
}

/// [Test 12] Bit-for-bit pipeline determinism across identical seeds and RFC-4180 CSV / JSON summary export.
#[test]
fn test_pipeline_determinism_and_csv_json_export_roundtrip() {
    let cohort = generate_demo_biofluid_cohort(1212);
    let cfg = MetabolomicsPipelineConfig {
        n_permutations: 40,
        rng_seed: 424242,
        ..Default::default()
    };
    let run1 = run_metabolomics_pipeline(&cohort, &cfg);
    let run2 = run_metabolomics_pipeline(&cohort, &cfg);

    assert_eq!(run1.pca.scores, run2.pca.scores);
    assert_eq!(run1.pca.loadings, run2.pca.loadings);
    let op1 = run1.opls_da.as_ref().unwrap();
    let op2 = run2.opls_da.as_ref().unwrap();
    assert_eq!(op1.predictive_scores, op2.predictive_scores);
    assert_eq!(op1.q2, op2.q2);
    assert_eq!(op1.permutation.p_value_q2, op2.permutation.p_value_q2);

    let bucket_csv = export_bucket_table_csv(&run1);
    assert_eq!(bucket_csv.lines().count(), 25); // header + 24 samples

    let bm_csv = export_biomarker_ranking_csv(&run1);
    assert_eq!(bm_csv.lines().count(), run1.buckets.len() + 1);
    assert!(bm_csv.contains("L-Lactate"));

    let json_str = export_study_summary_json(&run1, 10).expect("JSON serialize");
    let parsed: MetabolomicsSummaryReportJson =
        serde_json::from_str(&json_str).expect("JSON deserialize");
    assert_eq!(parsed.n_samples, 24);
    assert_eq!(parsed.n_buckets, run1.buckets.len());
    assert_eq!(parsed.top_biomarkers.len(), 10);
    assert_eq!(parsed.opls_is_validated, Some(true));
}

/// [Test 13] Adversarial & degenerate inputs return `None`/`Err` cleanly without panicking.
#[test]
fn test_metabolomics_adversarial_and_degenerate_inputs() {
    use nmr_core::metabolomics::compute_stocsy;
    use nmr_core::nmr1d::{Experiment1DKind, Spectrum1D};
    use nmr_core::spectrum::AxisPpm;

    // 1. Empty cohort & single-sample cohort must never panic
    let empty_study = run_metabolomics_pipeline(&[], &MetabolomicsPipelineConfig::default());
    assert!(empty_study.buckets.is_empty());
    assert!(empty_study.opls_da.is_none());

    // 2. `fit_opls_da` rejects single-group cohorts, `< 2` samples per class, and group-count mismatch
    let x4 = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.2, 2.1, 2.9],
        vec![2.0, 1.0, 4.0],
        vec![2.1, 0.9, 4.2],
    ];
    let one_group = vec!["Control".to_string(); 4];
    assert!(fit_opls_da(
        &x4,
        &one_group,
        ScalingMethod::Pareto,
        1,
        5,
        10,
        42,
        None,
        None
    )
    .is_none());

    let singleton_case = vec![
        "Control".to_string(),
        "Control".to_string(),
        "Control".to_string(),
        "Case".to_string(),
    ];
    assert!(fit_opls_da(
        &x4,
        &singleton_case,
        ScalingMethod::Pareto,
        1,
        5,
        10,
        42,
        None,
        None
    )
    .is_none());

    assert!(fit_opls_da(
        &x4,
        &one_group[..2],
        ScalingMethod::Pareto,
        1,
        5,
        10,
        42,
        None,
        None
    )
    .is_none());

    // 3. Out-of-bounds STOCSY driver index returns `None`
    assert!(compute_stocsy(99, &[], &x4, None, &[]).is_none());

    // 4. Corrupted `Spectrum1D` (`NaN` intensity or `axis.size != real.len()`) rejected by `validate()`
    let ax = AxisPpm::from_bounds("1H", 16, 10.0, 0.0, 600.0);
    let mut nan_real = vec![1.0; 16];
    nan_real[7] = f64::NAN;
    let bad_nan = Spectrum1D::new("nan", Experiment1DKind::Proton1D, ax, nan_real, None);
    assert!(bad_nan.validate().is_err());

    let mut bad_len = Spectrum1D::new(
        "short",
        Experiment1DKind::Proton1D,
        AxisPpm::from_bounds("1H", 4, 10.0, 0.0, 600.0),
        vec![1.0; 4],
        None,
    );
    bad_len.axis.size = 16;
    assert!(bad_len.validate().is_err());
}

/// [Test 14 — `F29`] Leak-free `VarianceTransform::Log10Pseudocount`, `FeatureScaling::UnitVariance`,
/// and nested-CV-selected `1..=max_ortho` `OPLS-DA` rescue multi-orthogonal heteroscedastic urinary
/// cohorts (50x glycosuria dynamic range + 3 orthogonal confounders) while passing the Westerhuis
/// `P = 200 >> N = 24` shuffled-label `Q^2 < 0.10` anti-overfitting guard.
#[test]
fn test_f29_log10_uv_multi_ortho_oplsda_and_null_label_guard() {
    use nmr_core::metabolomics::{fit_opls_da_with_options, FeatureScaling, VarianceTransform};

    let mut rng = Rng::new(202629);
    let n = 40usize;
    let p = 90usize;
    let mut x_urine = vec![vec![0.0; p]; n];
    let mut groups = Vec::with_capacity(n);

    // Build realistic urinary cohort:
    // - 3 strong independent orthogonal factors (pH/hippurate/dietary log-normal spikes)
    // - Heteroscedastic 60x dynamic-range outlier spikes on non-predictive columns 0..6
    // - Multiplicative class biomarker fold-change (columns 20..28, +2.4x in Case)
    for (i, row) in x_urine.iter_mut().enumerate() {
        let is_case = i >= 20;
        groups.push(if is_case { "T2DM" } else { "Control" }.to_string());
        let z_o1 = if i % 2 == 0 { 1.3 } else { -1.3 } + rng.normal(0.0, 0.25);
        let z_o2 = if (i / 2) % 2 == 0 { 1.1 } else { -1.1 } + rng.normal(0.0, 0.25);
        let z_o3 = ((i as f64) * 0.7).sin() * 1.2;

        for (j, slot) in row.iter_mut().enumerate() {
            let o1 = ((j as f64) * 0.19).cos() * 0.55;
            let o2 = ((j as f64) * 0.31).sin() * 0.50;
            let o3 = ((j as f64) * 0.47).cos() * 0.45;
            let log_conc = z_o1 * o1 + z_o2 * o2 + z_o3 * o3 + rng.normal(0.0, 0.18);
            let mut val = log_conc.exp();
            if j < 6 {
                // Massive 60x heteroscedastic urinary dynamic-range columns unrelated to class
                val *= 60.0 * (1.0 + 0.8 * z_o1.exp());
            }
            if (20..28).contains(&j) && is_case {
                val *= 2.4;
            }
            *slot = val;
        }
    }

    // Baseline untransformed 1-ortho Pareto model is degraded by the 60x heteroscedastic columns + 3 ortho axes
    let base_opls = fit_opls_da(
        &x_urine,
        &groups,
        FeatureScaling::Pareto,
        1,
        7,
        60,
        202601,
        Some("Control"),
        Some("T2DM"),
    )
    .expect("baseline OPLS-DA");

    // Log10Pseudocount + UnitVariance + CV-selected `1..=4` orthogonal components cleanly isolates
    // the multi-orthogonal urinary variation and boosts out-of-fold Q^2!
    let f29_opls = fit_opls_da_with_options(
        &x_urine,
        &groups,
        VarianceTransform::Log10Pseudocount,
        FeatureScaling::UnitVariance,
        1,
        Some(4),
        7,
        80,
        202601,
        Some("Control"),
        Some("T2DM"),
    )
    .expect("F29 log10+UV+max_ortho OPLS-DA");

    assert!(
        f29_opls.n_orthogonal >= 2,
        "expected CV to select >= 2 orthogonal components on 3-factor cohort, got {}",
        f29_opls.n_orthogonal
    );
    assert!(
        f29_opls.q2 > base_opls.q2 + 0.15 && f29_opls.q2 > 0.65,
        "expected Log10+UV+max_ortho Q^2 ({:.3}) >> baseline Q^2 ({:.3})",
        f29_opls.q2,
        base_opls.q2
    );
    assert!(f29_opls.permutation.p_value_q2 <= 0.02 && f29_opls.permutation.is_validated);

    // Westerhuis et al. (2008) anti-overfitting check on P = 200 >> N = 24 pure null labels
    // WITH `Log10Pseudocount + UnitVariance + max_ortho = 4`: nested leak-free CV must STILL yield `Q^2 < 0.10`!
    let mut rng_null = Rng::new(707707);
    let n_null = 24usize;
    let p_null = 200usize;
    let mut x_null = vec![vec![0.0; p_null]; n_null];
    for row in &mut x_null {
        let lat1 = rng_null.normal(0.0, 0.8);
        let lat2 = rng_null.normal(0.0, 0.7);
        for (j, slot) in row.iter_mut().enumerate() {
            let l = lat1 * ((j as f64) * 0.13).sin()
                + lat2 * ((j as f64) * 0.29).cos()
                + rng_null.normal(0.0, 0.5);
            *slot = l.exp();
        }
    }
    let mut null_groups: Vec<String> = (0..n_null)
        .map(|i| if i < 12 { "Control" } else { "T2DM" }.to_string())
        .collect();
    rng_null.shuffle(&mut null_groups);

    let null_opls = fit_opls_da_with_options(
        &x_null,
        &null_groups,
        VarianceTransform::Log10Pseudocount,
        FeatureScaling::UnitVariance,
        1,
        Some(4),
        6,
        80,
        20260707,
        Some("Control"),
        Some("T2DM"),
    )
    .expect("null-label multi-ortho OPLS-DA");

    assert!(
        null_opls.q2 < 0.10,
        "Westerhuis null-label Q^2 under Log10+UV+max_ortho=4 must remain < 0.10, got {:.4}",
        null_opls.q2
    );
    assert!(!null_opls.permutation.is_validated);
}
