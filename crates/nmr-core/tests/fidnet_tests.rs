//! `[synthetic self-consistency]` Unit and integration tests for `nmr_core::fidnet`
//! (symmetric dilated-convolution `FID-Net` engine — untrained architecture; no bundled
//! pretrained weights).
//!
//! All tests in this suite validate analytical gradients, synthetic spin-Hamiltonian
//! convergence (25% NUS reconstruction vs `nmr_core::process` IST/FISTA baseline, virtual
//! 13C-13C J=35 Hz decoupling, denoising), cache-free inference parity & memory-budget caps,
//! strict `FIDNET-V1` SHA-256 weight-bundle integrity, hostile-input rejection,
//! calibrated abstention diagnostics (`DataInconsistency`, `OutOfDistributionNoise`,
//! `HighUncertainty`, `NonFiniteInput`, `OversizedFid`, `EnergyAnomaly`), and physical 2D/3D
//! States hypercomplex `±f1`/`±f2` indirect-column application.

use std::f64::consts::TAU;

use nmr_core::fid::{Complex64, Fid2D, Fid3D, FidAxisMeta, FidVendor, QuadMode, StoredDimProc};
use nmr_core::fidnet::{
    apply_fidnet_2d_columns, apply_fidnet_3d_columns, apply_fidnet_fid2d_indirect,
    apply_fidnet_fid3d_indirect, build_coupling_channel, compute_fid_mse, compute_fid_snr_db,
    extract_fid2d_states_interferogram, extract_fid3d_states_interferogram,
    fid_to_magnitude_spectrum, generate_poisson_gap_nus_mask, generate_synthetic_fid,
    train_fidnet_step, AbstainReason, Fid3DIndirectAxis, FidNetAdamConfig, FidNetAdamState,
    FidNetBundleMetadata, FidNetConfig, FidNetError, FidNetInferenceConfig, FidNetModel,
    FidNetTask, FidNetTrainExample, FidNetWeightBundle, FidNetWeights, SyntheticFidSpec,
    SyntheticSpin, FIDNET_BUNDLE_FORMAT_VERSION, FIDNET_WEIGHTS_STATUS, MAX_FIDNET_CHANNELS,
    MAX_FIDNET_INFERENCE_ACTIVATION_BYTES, MAX_FIDNET_LENGTH,
};
use nmr_core::process::{reconstruct_nus_1d_ist, NusIstConfig};
use nmr_core::rng::Rng;

fn make_axis(label: &str, complex_points: usize, sw_hz: f64) -> FidAxisMeta {
    FidAxisMeta {
        label: label.to_string(),
        td_raw: complex_points * 2,
        complex_points,
        sw_hz,
        obs_mhz: 600.0,
        carrier_mhz: 600.0,
        first_ppm: 10.0,
        quad_mode: QuadMode::States,
        states_recombined: true,
        group_delay: 0.0,
        conjugate_for_fft: false,
        stored_proc: StoredDimProc::default(),
    }
}

/// `[synthetic self-consistency]` Verify exact analytical backpropagation gradients against
/// central finite differences across every single parameter in a multi-block dilated `FidNetModel`.
#[test]
fn test_fidnet_analytical_vs_finite_difference_gradients() {
    for &input_residual_skip in &[true, false] {
        let cfg = FidNetConfig {
            task: FidNetTask::NusReconstruction,
            use_mask_channel: true,
            use_coupling_channel: true,
            residual_channels: 4,
            skip_channels: 3,
            kernel_size: 3,
            dilations: vec![1, 2, 4],
            input_residual_skip,
        };
        let model = FidNetModel::init_random(cfg.clone(), 42 + u64::from(input_residual_skip))
            .expect("init_random");

        let spec = SyntheticFidSpec {
            len: 20,
            sw_hz: 1000.0,
            spins: vec![
                SyntheticSpin {
                    freq_hz: 110.0,
                    r2_hz: 15.0,
                    amplitude: 1.0,
                    phase_rad: 0.2,
                    j_coupling_hz: 35.0,
                },
                SyntheticSpin {
                    freq_hz: -175.0,
                    r2_hz: 12.0,
                    amplitude: 0.7,
                    phase_rad: -0.4,
                    j_coupling_hz: 35.0,
                },
            ],
            noise_sigma: 0.05,
            nus_fraction: Some(0.45),
            conditioning_j_hz: Some(35.0),
            seed: 9001,
        };
        let sample = generate_synthetic_fid(&spec).expect("generate_synthetic_fid");
        let loss_weights: Vec<f64> = (0..spec.len).map(|t| 0.5 + 0.05 * (t as f64)).collect();

        let x_in = model
            .build_input_features(
                &sample.observed,
                sample.nus_mask.as_deref(),
                sample.coupling_channel.as_deref(),
            )
            .expect("build_input_features");
        let cache = model
            .forward_features_with_cache(&x_in, spec.len)
            .expect("forward_features_with_cache");
        let (_loss0, grads) = model
            .backward_mse(&cache, &sample.clean_singlet, Some(&loss_weights))
            .expect("backward_mse");
        let analytical = grads.to_flat_vec();
        let base_params = model.weights.to_flat_vec();
        assert_eq!(analytical.len(), base_params.len());

        let eps = 1e-6;
        let mut max_abs_err = 0.0f64;
        let mut max_rel_err = 0.0f64;
        for idx in 0..base_params.len() {
            let mut p_plus = base_params.clone();
            p_plus[idx] += eps;
            let m_plus = FidNetModel::new(
                cfg.clone(),
                FidNetWeights::from_flat_slice(&cfg, &p_plus).expect("plus weights"),
            )
            .expect("plus model");
            let c_plus = m_plus
                .forward_features_with_cache(&x_in, spec.len)
                .expect("plus fwd");
            let (l_plus, _) = m_plus
                .backward_mse(&c_plus, &sample.clean_singlet, Some(&loss_weights))
                .expect("plus loss");

            let mut p_minus = base_params.clone();
            p_minus[idx] -= eps;
            let m_minus = FidNetModel::new(
                cfg.clone(),
                FidNetWeights::from_flat_slice(&cfg, &p_minus).expect("minus weights"),
            )
            .expect("minus model");
            let c_minus = m_minus
                .forward_features_with_cache(&x_in, spec.len)
                .expect("minus fwd");
            let (l_minus, _) = m_minus
                .backward_mse(&c_minus, &sample.clean_singlet, Some(&loss_weights))
                .expect("minus loss");

            let fd = (l_plus - l_minus) / (2.0 * eps);
            let an = analytical[idx];
            let abs_err = (an - fd).abs();
            let rel_err = abs_err / an.abs().max(fd.abs()).max(1e-6);
            max_abs_err = max_abs_err.max(abs_err);
            max_rel_err = max_rel_err.max(rel_err);
        }

        assert!(
            max_abs_err < 1e-5,
            "input_residual_skip={input_residual_skip}: analytical vs FD max_abs_err={max_abs_err:.3e} >= 1e-5"
        );
        assert!(
            max_rel_err < 1e-4,
            "input_residual_skip={input_residual_skip}: analytical vs FD max_rel_err={max_rel_err:.3e} >= 1e-4"
        );
    }
}

/// `[synthetic self-consistency]` Train a compact toy `FID-Net` on synthetic 25% Poisson-gap NUS
/// multi-spin complex FIDs and compare honestly against BOTH zero-filling AND the existing
/// `nmr_core::process::reconstruct_nus_1d_ist` (FISTA/IST) baseline.
#[test]
fn test_fidnet_training_nus_25pct_reconstruction_convergence() {
    let cfg = FidNetConfig {
        task: FidNetTask::NusReconstruction,
        use_mask_channel: true,
        use_coupling_channel: false,
        residual_channels: 10,
        skip_channels: 10,
        kernel_size: 5,
        dilations: vec![1, 2, 4, 8],
        input_residual_skip: true,
    };
    let mut model = FidNetModel::init_random(cfg.clone(), 101).expect("init nus model");
    let mut adam = FidNetAdamState::new(
        &cfg,
        FidNetAdamConfig {
            lr: 1e-2,
            max_grad_norm: 5.0,
            ..FidNetAdamConfig::default()
        },
    )
    .expect("init adam");

    let len = 48usize;
    let sw_hz = 1200.0;
    let mut rng = Rng::new(202601);

    let make_nus_batch = |n: usize, rng: &mut Rng, seed_offset: u64| -> Vec<FidNetTrainExample> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let f1 = rng.uniform(-140.0, -40.0);
            let f2 = rng.uniform(40.0, 140.0);
            let spec = SyntheticFidSpec {
                len,
                sw_hz,
                spins: vec![
                    SyntheticSpin {
                        freq_hz: f1,
                        r2_hz: rng.uniform(10.0, 18.0),
                        amplitude: rng.uniform(0.7, 1.1),
                        phase_rad: rng.uniform(-0.3, 0.3),
                        j_coupling_hz: 0.0,
                    },
                    SyntheticSpin {
                        freq_hz: f2,
                        r2_hz: rng.uniform(10.0, 18.0),
                        amplitude: rng.uniform(0.6, 1.0),
                        phase_rad: rng.uniform(-0.3, 0.3),
                        j_coupling_hz: 0.0,
                    },
                ],
                noise_sigma: 0.02,
                nus_fraction: Some(0.25),
                conditioning_j_hz: None,
                seed: seed_offset + i as u64,
            };
            let s = generate_synthetic_fid(&spec).expect("gen nus fid");
            let sampled = s
                .nus_mask
                .as_ref()
                .expect("nus mask")
                .iter()
                .filter(|&&b| b)
                .count();
            assert_eq!(sampled, 12, "25% of 48 points = 12 sampled increments");
            out.push(FidNetTrainExample {
                input_fid: s.observed,
                nus_mask: s.nus_mask,
                coupling_channel: None,
                target_fid: s.clean_singlet,
                loss_weights: None,
            });
        }
        out
    };

    // Held-out validation set (unseen frequencies and Poisson-gap masks)
    let eval_set = make_nus_batch(8, &mut rng, 990_000);

    // Online synthetic spin-Hamiltonian mini-batch training (FID-Net protocol: Karunanithy & Hansen 2021)
    let mut early_loss_sum = 0.0;
    let mut late_loss_sum = 0.0;
    let n_steps = 180usize;
    for step in 0..n_steps {
        let batch = make_nus_batch(8, &mut rng, 10_000 + (step as u64) * 32);
        let loss = train_fidnet_step(&mut model, &mut adam, &batch).expect("train step");
        if step < 10 {
            early_loss_sum += loss;
        }
        if step >= n_steps - 10 {
            late_loss_sum += loss;
        }
    }
    let early_loss = early_loss_sum / 10.0;
    let late_loss = late_loss_sum / 10.0;

    assert!(
        late_loss < early_loss * 0.40,
        "NUS training loss did not converge strongly: early={early_loss:.5}, late={late_loss:.5}"
    );

    let mut baseline_mse_sum = 0.0;
    let mut ist_mse_sum = 0.0;
    let mut recon_mse_sum = 0.0;
    let mut baseline_snr_sum = 0.0;
    let mut ist_snr_sum = 0.0;
    let mut recon_snr_sum = 0.0;
    let ist_cfg = NusIstConfig::default();
    for ex in &eval_set {
        let mask = ex.nus_mask.as_deref().expect("nus mask");
        let pred = model
            .forward_fid_inference(&ex.input_fid, Some(mask), None)
            .expect("eval forward");
        let ist_pred = reconstruct_nus_1d_ist(&ex.input_fid, mask, &ist_cfg);
        baseline_mse_sum += compute_fid_mse(&ex.target_fid, &ex.input_fid);
        ist_mse_sum += compute_fid_mse(&ex.target_fid, &ist_pred);
        recon_mse_sum += compute_fid_mse(&ex.target_fid, &pred);
        baseline_snr_sum += compute_fid_snr_db(&ex.target_fid, &ex.input_fid);
        ist_snr_sum += compute_fid_snr_db(&ex.target_fid, &ist_pred);
        recon_snr_sum += compute_fid_snr_db(&ex.target_fid, &pred);
    }

    let n_eval = eval_set.len() as f64;
    let base_mse = baseline_mse_sum / n_eval;
    let ist_mse = ist_mse_sum / n_eval;
    let rec_mse = recon_mse_sum / n_eval;
    let base_snr = baseline_snr_sum / n_eval;
    let ist_snr = ist_snr_sum / n_eval;
    let rec_snr = recon_snr_sum / n_eval;
    println!(
        "[synthetic self-consistency] 25% NUS (untrained arch -> 180-step toy training): loss {early_loss:.4}->{late_loss:.4} | zero-fill MSE={base_mse:.4} (SNR={base_snr:.2} dB) | IST/FISTA MSE={ist_mse:.4} (SNR={ist_snr:.2} dB) | toy FID-Net MSE={rec_mse:.4} (SNR={rec_snr:.2} dB)"
    );

    assert!(
        ist_mse < base_mse * 0.25 && ist_snr > base_snr + 6.0,
        "IST/FISTA baseline sanity check failed: zf_mse={base_mse:.5}, ist_mse={ist_mse:.5}, zf_snr={base_snr:.2}, ist_snr={ist_snr:.2}"
    );
    assert!(
        rec_mse < base_mse * 0.45,
        "25% NUS toy FID-Net MSE did not improve substantially over zero-fill: baseline={base_mse:.5}, recon={rec_mse:.5}"
    );
    assert!(
        rec_snr > base_snr + 3.5,
        "25% NUS toy FID-Net SNR gain over zero-fill too small: baseline={base_snr:.2} dB, recon={rec_snr:.2} dB"
    );
}

/// `[synthetic self-consistency]` Train a compact `FID-Net` on synthetic virtual 13C-13C
/// (J = 35 Hz) homonuclear decoupling and verify both time-domain SNR/MSE gain and
/// frequency-domain doublet-to-singlet collapse.
#[test]
fn test_fidnet_training_virtual_decoupling_j35hz_doublet_to_singlet() {
    let cfg = FidNetConfig {
        task: FidNetTask::VirtualDecoupling,
        use_mask_channel: false,
        use_coupling_channel: true,
        residual_channels: 8,
        skip_channels: 8,
        kernel_size: 5,
        dilations: vec![1, 2, 4, 8],
        input_residual_skip: true,
    };
    let mut model = FidNetModel::init_random(cfg.clone(), 202).expect("init decoupling model");
    let mut adam = FidNetAdamState::new(
        &cfg,
        FidNetAdamConfig {
            lr: 8e-3,
            max_grad_norm: 5.0,
            ..FidNetAdamConfig::default()
        },
    )
    .expect("init adam");

    let len = 48usize;
    let sw_hz = 600.0;
    let j_hz = 35.0;
    let mut rng = Rng::new(202602);

    let make_j_batch = |n: usize, rng: &mut Rng, seed_offset: u64| -> Vec<FidNetTrainExample> {
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let spec = SyntheticFidSpec {
                len,
                sw_hz,
                spins: vec![SyntheticSpin {
                    freq_hz: rng.uniform(-120.0, 120.0),
                    r2_hz: rng.uniform(8.0, 14.0),
                    amplitude: rng.uniform(0.8, 1.2),
                    phase_rad: rng.uniform(-0.2, 0.2),
                    j_coupling_hz: j_hz,
                }],
                noise_sigma: 0.015,
                nus_fraction: None,
                conditioning_j_hz: Some(j_hz),
                seed: seed_offset + i as u64,
            };
            let s = generate_synthetic_fid(&spec).expect("gen coupled fid");
            out.push(FidNetTrainExample {
                input_fid: s.observed,
                nus_mask: None,
                coupling_channel: s.coupling_channel,
                target_fid: s.clean_singlet,
                loss_weights: None,
            });
        }
        out
    };

    // Online synthetic spin-Hamiltonian mini-batch training
    let mut early_loss_sum = 0.0;
    let mut late_loss_sum = 0.0;
    let n_steps = 120usize;
    for step in 0..n_steps {
        let batch = make_j_batch(8, &mut rng, 30_000 + (step as u64) * 32);
        let loss = train_fidnet_step(&mut model, &mut adam, &batch).expect("train step");
        if step < 10 {
            early_loss_sum += loss;
        }
        if step >= n_steps - 10 {
            late_loss_sum += loss;
        }
    }
    let early_loss = early_loss_sum / 10.0;
    let late_loss = late_loss_sum / 10.0;

    assert!(
        late_loss < early_loss * 0.25,
        "virtual decoupling training loss did not converge: early={early_loss:.5}, late={late_loss:.5}"
    );

    // Evaluate on a held-out centered 13C doublet at f0 = 0 Hz, J = 35 Hz
    let eval_spec = SyntheticFidSpec {
        len,
        sw_hz,
        spins: vec![SyntheticSpin {
            freq_hz: 0.0,
            r2_hz: 10.0,
            amplitude: 1.0,
            phase_rad: 0.0,
            j_coupling_hz: j_hz,
        }],
        noise_sigma: 0.01,
        nus_fraction: None,
        conditioning_j_hz: Some(j_hz),
        seed: 44_444,
    };
    let eval = generate_synthetic_fid(&eval_spec).expect("eval coupled fid");
    let decoupled = model
        .forward_fid(&eval.observed, None, eval.coupling_channel.as_deref())
        .expect("decoupled forward");

    let base_mse = compute_fid_mse(&eval.clean_singlet, &eval.observed);
    let dec_mse = compute_fid_mse(&eval.clean_singlet, &decoupled);
    let base_snr = compute_fid_snr_db(&eval.clean_singlet, &eval.observed);
    let dec_snr = compute_fid_snr_db(&eval.clean_singlet, &decoupled);
    println!(
        "[synthetic self-consistency] Virtual J=35Hz decoupling: loss {early_loss:.4}->{late_loss:.4}, MSE {base_mse:.4}->{dec_mse:.4}, SNR {base_snr:.2}->{dec_snr:.2} dB"
    );
    assert!(
        dec_mse < base_mse * 0.25,
        "virtual decoupling MSE did not drop sharply: coupled={base_mse:.5}, decoupled={dec_mse:.5}"
    );
    assert!(
        dec_snr > base_snr + 6.0,
        "virtual decoupling SNR gain too small: coupled={base_snr:.2} dB, decoupled={dec_snr:.2} dB"
    );

    // Verify spectral doublet-to-singlet collapse:
    // At 0 Hz (center bin = N/2), a J=35 Hz doublet has a trough while +/-17.5 Hz has two peaks;
    // after virtual decoupling, the center singlet intensity dominates over the +/-17.5 Hz doublet horns.
    let zf = 8usize;
    let spec_coupled = fid_to_magnitude_spectrum(&eval.observed, zf);
    let spec_decoupled = fid_to_magnitude_spectrum(&decoupled, zf);
    let n_fft = spec_coupled.len();
    let center_bin = n_fft / 2;
    let half_j_bins = ((0.5 * j_hz / sw_hz) * (n_fft as f64)).round() as usize;

    let coupled_center = spec_coupled[center_bin];
    let coupled_horns =
        0.5 * (spec_coupled[center_bin - half_j_bins] + spec_coupled[center_bin + half_j_bins]);
    let decoupled_center = spec_decoupled[center_bin];
    let decoupled_horns =
        0.5 * (spec_decoupled[center_bin - half_j_bins] + spec_decoupled[center_bin + half_j_bins]);

    assert!(
        coupled_horns > coupled_center,
        "sanity check: input coupled spectrum must be a doublet (horns={coupled_horns:.3} > center={coupled_center:.3})"
    );
    assert!(
        decoupled_center > decoupled_horns * 1.4,
        "virtual decoupling failed to collapse doublet into singlet (center={decoupled_center:.3}, horns={decoupled_horns:.3})"
    );
}

/// `[synthetic self-consistency]` Train a compact `FID-Net` on noisy complex FIDs and verify
/// measurable time-domain SNR gain and MSE reduction on held-out noisy FIDs.
#[test]
fn test_fidnet_training_denoising_convergence() {
    let cfg = FidNetConfig {
        task: FidNetTask::Denoising,
        use_mask_channel: false,
        use_coupling_channel: false,
        residual_channels: 8,
        skip_channels: 8,
        kernel_size: 5,
        dilations: vec![1, 2, 4, 8],
        input_residual_skip: true,
    };
    let mut model = FidNetModel::init_random(cfg.clone(), 303).expect("init denoise model");
    let mut adam = FidNetAdamState::new(
        &cfg,
        FidNetAdamConfig {
            lr: 7e-3,
            max_grad_norm: 5.0,
            ..FidNetAdamConfig::default()
        },
    )
    .expect("init adam");

    let len = 48usize;
    let sw_hz = 800.0;
    let mut rng = Rng::new(202603);

    let make_noisy_dataset =
        |n: usize, rng: &mut Rng, seed_offset: u64| -> Vec<FidNetTrainExample> {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let spec = SyntheticFidSpec {
                    len,
                    sw_hz,
                    spins: vec![
                        SyntheticSpin {
                            freq_hz: rng.uniform(-120.0, -30.0),
                            r2_hz: rng.uniform(10.0, 16.0),
                            amplitude: rng.uniform(0.8, 1.2),
                            phase_rad: rng.uniform(-0.3, 0.3),
                            j_coupling_hz: 0.0,
                        },
                        SyntheticSpin {
                            freq_hz: rng.uniform(30.0, 120.0),
                            r2_hz: rng.uniform(10.0, 16.0),
                            amplitude: rng.uniform(0.7, 1.1),
                            phase_rad: rng.uniform(-0.3, 0.3),
                            j_coupling_hz: 0.0,
                        },
                    ],
                    noise_sigma: 0.25,
                    nus_fraction: None,
                    conditioning_j_hz: None,
                    seed: seed_offset + i as u64,
                };
                let s = generate_synthetic_fid(&spec).expect("gen noisy fid");
                out.push(FidNetTrainExample {
                    input_fid: s.observed,
                    nus_mask: None,
                    coupling_channel: None,
                    target_fid: s.clean_singlet,
                    loss_weights: None,
                });
            }
            out
        };

    let train_set = make_noisy_dataset(16, &mut rng, 50_000);
    let eval_set = make_noisy_dataset(6, &mut rng, 60_000);

    let mut first_loss = 0.0;
    let mut last_loss = 0.0;
    for step in 0..75 {
        let loss = train_fidnet_step(&mut model, &mut adam, &train_set).expect("train step");
        if step == 0 {
            first_loss = loss;
        }
        last_loss = loss;
    }
    assert!(
        last_loss < first_loss * 0.50,
        "denoising training loss did not converge: first={first_loss:.5}, last={last_loss:.5}"
    );

    let mut base_mse = 0.0;
    let mut rec_mse = 0.0;
    let mut base_snr = 0.0;
    let mut rec_snr = 0.0;
    for ex in &eval_set {
        let pred = model
            .forward_fid(&ex.input_fid, None, None)
            .expect("eval denoise");
        base_mse += compute_fid_mse(&ex.target_fid, &ex.input_fid);
        rec_mse += compute_fid_mse(&ex.target_fid, &pred);
        base_snr += compute_fid_snr_db(&ex.target_fid, &ex.input_fid);
        rec_snr += compute_fid_snr_db(&ex.target_fid, &pred);
    }
    let n = eval_set.len() as f64;
    base_mse /= n;
    rec_mse /= n;
    base_snr /= n;
    rec_snr /= n;
    println!(
        "[synthetic self-consistency] Denoising: loss {first_loss:.4}->{last_loss:.4}, MSE {base_mse:.4}->{rec_mse:.4}, SNR {base_snr:.2}->{rec_snr:.2} dB"
    );

    assert!(
        rec_mse < base_mse * 0.60,
        "denoising MSE did not improve substantially: noisy={base_mse:.5}, denoised={rec_mse:.5}"
    );
    assert!(
        rec_snr > base_snr + 2.2,
        "denoising SNR gain too small: noisy={base_snr:.2} dB, denoised={rec_snr:.2} dB"
    );
}

/// `[synthetic self-consistency]` Verify strict `FIDNET-V1` JSON/binary round-trip serialization,
/// SHA-256 tamper rejection, dimension mismatch / NaN / oversized-FID rejection, and
/// explicit abstention triggers (`DataInconsistency`, `OutOfDistributionNoise`, `NonFiniteInput`,
/// `OversizedFid`, `EnergyAnomaly`).
#[test]
fn test_fidnet_bundle_integrity_hostile_inputs_and_abstention() {
    let cfg = FidNetConfig::nus_default(6, 4);
    let model = FidNetModel::init_random(cfg.clone(), 777).expect("init model");
    let bundle = model
        .to_bundle(FidNetBundleMetadata::default())
        .expect("to_bundle");
    assert_eq!(bundle.format_version, FIDNET_BUNDLE_FORMAT_VERSION);
    assert_eq!(bundle.sha256.len(), 64);

    // 1. JSON round-trip (0-ULP f32 canonical bundle + SHA-256 verification)
    let json = bundle.to_json().expect("to_json");
    let parsed_json = FidNetWeightBundle::from_json(&json).expect("from_json");
    assert_eq!(parsed_json, bundle);
    let restored = FidNetModel::from_bundle(&parsed_json).expect("from_bundle");
    assert_eq!(restored.config, model.config);

    // 2. Binary round-trip
    let bytes = bundle.to_bytes().expect("to_bytes");
    let parsed_bin = FidNetWeightBundle::from_bytes(&bytes).expect("from_bytes");
    assert_eq!(parsed_bin.config, bundle.config);
    assert_eq!(parsed_bin.weights, bundle.weights);
    assert_eq!(parsed_bin.sha256, bundle.sha256);

    // 3. Corrupted SHA-256 rejection (JSON & binary)
    let mut bad_sha = bundle.clone();
    bad_sha.sha256 = "0".repeat(64);
    assert!(matches!(
        bad_sha.verify(),
        Err(FidNetError::ChecksumMismatch { .. })
    ));
    let mut bad_bytes = bytes.clone();
    let last = bad_bytes.len() - 1;
    bad_bytes[last] = if bad_bytes[last] == b'0' { b'1' } else { b'0' };
    assert!(matches!(
        FidNetWeightBundle::from_bytes(&bad_bytes),
        Err(FidNetError::ChecksumMismatch { .. })
    ));

    // 4. Unsupported format version rejection
    let mut bad_ver = bundle.clone();
    bad_ver.format_version = "FIDNET-V99".to_string();
    assert!(matches!(
        bad_ver.verify(),
        Err(FidNetError::UnsupportedFormatVersion(_))
    ));

    // 5. Mismatched weight dimensions rejection
    let mut bad_dims = model.weights.clone();
    bad_dims.stem_b.pop();
    assert!(matches!(
        bad_dims.validate(&cfg),
        Err(FidNetError::InvalidWeights(_))
    ));

    // 6. NaN / Inf weight rejection
    let mut nan_weights = model.weights.clone();
    nan_weights.out_b[0] = f64::NAN;
    assert!(matches!(
        nan_weights.validate(&cfg),
        Err(FidNetError::InvalidWeights(_))
    ));
    let mut inf_weights = model.weights.clone();
    inf_weights.head1_w[0] = f64::INFINITY;
    assert!(matches!(
        inf_weights.validate(&cfg),
        Err(FidNetError::InvalidWeights(_))
    ));

    // 7. Hostile config cap rejection
    let oversized_cfg = FidNetConfig {
        residual_channels: MAX_FIDNET_CHANNELS + 1,
        ..cfg.clone()
    };
    assert!(matches!(
        oversized_cfg.validate(),
        Err(FidNetError::InvalidConfig(_))
    ));

    // 8. NaN & oversized FID rejection (hard error on forward_fid + abstention on infer_fid_with_diagnostics)
    let mut nan_fid: Vec<Complex64> = vec![[1.0, 0.0]; 16];
    nan_fid[5][1] = f64::NAN;
    assert!(matches!(
        model.forward_fid(&nan_fid, None, None),
        Err(FidNetError::NonFiniteInput(_))
    ));
    let nan_out =
        model.infer_fid_with_diagnostics(&nan_fid, None, None, &FidNetInferenceConfig::default());
    assert!(nan_out.diagnostic.abstained);
    assert!(nan_out
        .diagnostic
        .abstain_reasons
        .contains(&AbstainReason::NonFiniteInput));

    let huge_fid = vec![[0.0, 0.0]; MAX_FIDNET_LENGTH + 1];
    assert!(matches!(
        model.forward_fid(&huge_fid, None, None),
        Err(FidNetError::OversizedFid { .. })
    ));
    let huge_out =
        model.infer_fid_with_diagnostics(&huge_fid, None, None, &FidNetInferenceConfig::default());
    assert!(huge_out.diagnostic.abstained);
    assert!(huge_out
        .diagnostic
        .abstain_reasons
        .contains(&AbstainReason::OversizedFid));

    // 9. Data-consistency abstention trigger: corrupt output bias so reconstructed FID severely
    // violates measured NUS points.
    let mut inconsistent_model = model.clone();
    inconsistent_model.weights.out_b = vec![4.5, -4.5];
    let clean_spec = SyntheticFidSpec {
        len: 32,
        sw_hz: 1000.0,
        spins: vec![SyntheticSpin {
            freq_hz: 120.0,
            r2_hz: 20.0,
            amplitude: 1.0,
            phase_rad: 0.0,
            j_coupling_hz: 0.0,
        }],
        noise_sigma: 0.01,
        nus_fraction: Some(0.5),
        conditioning_j_hz: None,
        seed: 555,
    };
    let clean_sample = generate_synthetic_fid(&clean_spec).expect("gen sample");
    let diag_out = inconsistent_model.infer_fid_with_diagnostics(
        &clean_sample.observed,
        clean_sample.nus_mask.as_deref(),
        None,
        &FidNetInferenceConfig {
            max_data_inconsistency_rel: 0.25,
            ..FidNetInferenceConfig::default()
        },
    );
    assert!(diag_out.diagnostic.abstained);
    assert!(
        diag_out
            .diagnostic
            .abstain_reasons
            .contains(&AbstainReason::DataInconsistency),
        "expected DataInconsistency abstention, got {:?}",
        diag_out.diagnostic
    );
    assert!(diag_out.diagnostic.measured_data_residual_rel > 0.25);

    // 10. Out-of-distribution pure thermal noise abstention trigger
    let noise_spec = SyntheticFidSpec {
        len: 64,
        sw_hz: 1000.0,
        spins: vec![],
        noise_sigma: 1.0,
        nus_fraction: None,
        conditioning_j_hz: None,
        seed: 888,
    };
    let pure_noise = generate_synthetic_fid(&noise_spec).expect("pure noise");
    let ood_out = model.infer_fid_with_diagnostics(
        &pure_noise.observed,
        None,
        None,
        &FidNetInferenceConfig {
            min_input_snr: 1.5,
            ..FidNetInferenceConfig::default()
        },
    );
    assert!(ood_out.diagnostic.abstained);
    assert!(
        ood_out
            .diagnostic
            .abstain_reasons
            .contains(&AbstainReason::OutOfDistributionNoise),
        "expected OutOfDistributionNoise, got {:?}",
        ood_out.diagnostic
    );

    // 11. Honest untrained-architecture disclosure in default bundle metadata (22.2:P1-1)
    let default_meta = FidNetBundleMetadata::default();
    assert!(
        default_meta.description.contains(FIDNET_WEIGHTS_STATUS),
        "default metadata must disclose untrained architecture status"
    );
    assert!(
        !default_meta.provenance.contains("Reinstein")
            && !default_meta.provenance.contains("Phys. Chem. Lett."),
        "metadata must not contain fabricated JPCL 2023 citation"
    );

    // 12. Truncated binary bundle claiming large param count fails cleanly without pre-allocating (22.2:P2-1)
    let trunc_bytes = &bytes[..bytes.len() - 10];
    assert!(matches!(
        FidNetWeightBundle::from_bytes(trunc_bytes),
        Err(FidNetError::SerializationError(_))
    ));
}

/// Physical 2D States hypercomplex encoder:
/// `S_cos(t1, t0) = exp(-R2_1 t1) cos(2pi f1 t1 + phi1) * exp((-R2_0 + i 2pi f0) t0)`
/// `S_sin(t1, t0) = exp(-R2_1 t1) sin(2pi f1 t1 + phi1) * exp((-R2_0 + i 2pi f0) t0)`
#[allow(clippy::too_many_arguments)]
fn encode_physical_states_fid2d(
    n0: usize,
    sw0_hz: f64,
    f0_hz: f64,
    r2_0: f64,
    n1: usize,
    sw1_hz: f64,
    f1_hz: f64,
    r2_1: f64,
    nus_mask: Option<Vec<bool>>,
) -> Fid2D {
    let dt0 = 1.0 / sw0_hz;
    let dt1 = 1.0 / sw1_hz;
    let mut data = vec![[0.0; 2]; n1 * 2 * n0];
    for k1 in 0..n1 {
        let sampled = nus_mask.as_ref().is_none_or(|m| m[k1]);
        if !sampled {
            continue;
        }
        let t1 = (k1 as f64) * dt1;
        let env1 = (-r2_1 * t1).exp();
        let (sin1, cos1) = (TAU * f1_hz * t1).sin_cos();
        for ix in 0..n0 {
            let t0 = (ix as f64) * dt0;
            let env0 = (-r2_0 * t0).exp();
            let (sin0, cos0) = (TAU * f0_hz * t0).sin_cos();
            let dir_re = env0 * cos0;
            let dir_im = env0 * sin0;
            data[(k1 * 2) * n0 + ix] = [env1 * cos1 * dir_re, env1 * cos1 * dir_im];
            data[(k1 * 2 + 1) * n0 + ix] = [env1 * sin1 * dir_re, env1 * sin1 * dir_im];
        }
    }
    Fid2D {
        name: format!("states_2d_f1_{f1_hz:.0}hz"),
        vendor: FidVendor::Bruker,
        pulse_program: "hsqcetfpf3gpsi".to_string(),
        axis_x: make_axis("1H", n0, sw0_hz),
        axis_y: make_axis("15N", n1, sw1_hz),
        data,
        nus_mask,
        truncated_traces: None,
    }
}

/// Physical 3D States hypercomplex encoder:
/// `S_{c2,c1}(t2, t1, t0) = trig_{c2}(2pi f2 t2) e^{-R2_2 t2} * trig_{c1}(2pi f1 t1) e^{-R2_1 t1} * e^{(-R2_0 + i 2pi f0) t0}`
#[allow(clippy::too_many_arguments)]
fn encode_physical_states_fid3d(
    n0: usize,
    sw0_hz: f64,
    f0_hz: f64,
    n1: usize,
    sw1_hz: f64,
    f1_hz: f64,
    n2: usize,
    sw2_hz: f64,
    f2_hz: f64,
) -> Fid3D {
    let dt0 = 1.0 / sw0_hz;
    let dt1 = 1.0 / sw1_hz;
    let dt2 = 1.0 / sw2_hz;
    let mut data = vec![[0.0; 2]; n2 * 2 * n1 * 2 * n0];
    for k2 in 0..n2 {
        let t2 = (k2 as f64) * dt2;
        let env2 = (-8.0 * t2).exp();
        let (sin2, cos2) = (TAU * f2_hz * t2).sin_cos();
        let trig2 = [cos2, sin2];
        for k1 in 0..n1 {
            let t1 = (k1 as f64) * dt1;
            let env1 = (-8.0 * t1).exp();
            let (sin1, cos1) = (TAU * f1_hz * t1).sin_cos();
            let trig1 = [cos1, sin1];
            for (c2, &t2_trig) in trig2.iter().enumerate() {
                for (c1, &t1_trig) in trig1.iter().enumerate() {
                    let scale_ind = env2 * t2_trig * env1 * t1_trig;
                    for ix in 0..n0 {
                        let t0 = (ix as f64) * dt0;
                        let env0 = (-10.0 * t0).exp();
                        let (sin0, cos0) = (TAU * f0_hz * t0).sin_cos();
                        let idx = (((k2 * 2 + c2) * n1 + k1) * 2 + c1) * n0 + ix;
                        data[idx] = [scale_ind * env0 * cos0, scale_ind * env0 * sin0];
                    }
                }
            }
        }
    }
    Fid3D {
        name: format!("states_3d_f1_{f1_hz:.0}_f2_{f2_hz:.0}"),
        vendor: FidVendor::Bruker,
        pulse_program: "hncagp3d".to_string(),
        axis_x: make_axis("1H", n0, sw0_hz),
        axis_y: make_axis("15N", n1, sw1_hz),
        axis_z: make_axis("13C", n2, sw2_hz),
        data,
        nus_mask: None,
        truncated_traces: None,
    }
}

fn peak_freq_hz_from_spectrum(fid: &[Complex64], sw_hz: f64, zf: usize) -> f64 {
    let mag = fid_to_magnitude_spectrum(fid, zf);
    let n = mag.len();
    let (best_bin, _) =
        mag.iter().enumerate().fold(
            (0usize, -1.0_f64),
            |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            },
        );
    let half = (n / 2) as f64;
    ((best_bin as f64) - half) * (sw_hz / (n as f64))
}

/// `[synthetic self-consistency]` Regression test for `22.2:P0-1`: verify 2D & 3D States
/// hypercomplex interferogram extraction and indirect-axis application distinguish `+300 Hz`
/// vs `-300 Hz` as exact complex conjugates and recover the exact signed `±f1` / `±f2` frequency.
#[test]
fn test_fidnet_states_2d_and_3d_signed_frequency_preservation() {
    let n0 = 4usize;
    let sw0 = 4000.0;
    let f0 = 350.0;
    let n1 = 64usize;
    let sw1 = 2000.0;

    let fid_pos = encode_physical_states_fid2d(n0, sw0, f0, 10.0, n1, sw1, 300.0, 8.0, None);
    let fid_neg = encode_physical_states_fid2d(n0, sw0, f0, 10.0, n1, sw1, -300.0, 8.0, None);

    // 1. Direct verification: +300 Hz and -300 Hz States interferograms must be distinct exact
    // complex conjugates on BOTH direct channels (Re_direct and Im_direct).
    for dir_ch in 0..2 {
        // Choose ix = 1 so both cos(2pi f0 t0) and sin(2pi f0 t0) are non-zero.
        let col_pos =
            extract_fid2d_states_interferogram(&fid_pos, 1, dir_ch).expect("extract +300");
        let col_neg =
            extract_fid2d_states_interferogram(&fid_neg, 1, dir_ch).expect("extract -300");
        let mut max_conj_err = 0.0_f64;
        let mut max_diff = 0.0_f64;
        for (&[rp, ip], &[rn, ineg]) in col_pos.iter().zip(col_neg.iter()) {
            max_conj_err = max_conj_err.max((rp - rn).abs()).max((ip + ineg).abs());
            max_diff = max_diff.max((rp - rn).abs()).max((ip - ineg).abs());
        }
        assert!(
            max_conj_err < 1e-12,
            "dir_ch={dir_ch}: +300 Hz and -300 Hz must be exact complex conjugates (err={max_conj_err:.3e})"
        );
        assert!(
            max_diff > 0.25,
            "dir_ch={dir_ch}: +300 Hz and -300 Hz inputs must be strongly distinct (max_diff={max_diff:.3e})"
        );
    }

    // 2. Round-trip through apply_fidnet_fid2d_indirect with exact identity model (zero weights +
    // input_residual_skip=true) recovers exact signed +300 Hz and -300 Hz frequencies!
    let id_cfg = FidNetConfig::nus_default(4, 4);
    let id_weights = FidNetWeights::zeros(&id_cfg).expect("zeros");
    let id_model = FidNetModel::new(id_cfg, id_weights).expect("id model");
    let inf_cfg = FidNetInferenceConfig {
        uncertainty_passes: 2,
        ..FidNetInferenceConfig::default()
    };

    let (out_pos, diags_pos) =
        apply_fidnet_fid2d_indirect(&id_model, &fid_pos, None, &inf_cfg).expect("apply +300");
    let (out_neg, diags_neg) =
        apply_fidnet_fid2d_indirect(&id_model, &fid_neg, None, &inf_cfg).expect("apply -300");
    assert_eq!(diags_pos.len(), 2 * n0);
    assert_eq!(diags_neg.len(), 2 * n0);
    // Check that neither clean signal falsely abstained on non-zero channels
    assert!(
        !diags_pos[0].abstained,
        "clean +300 Hz channel should not abstain: {:?}",
        diags_pos[0]
    );
    assert!(
        !diags_neg[0].abstained,
        "clean -300 Hz channel should not abstain: {:?}",
        diags_neg[0]
    );

    let rec_pos = extract_fid2d_states_interferogram(&out_pos, 0, 0).expect("rec +300");
    let rec_neg = extract_fid2d_states_interferogram(&out_neg, 0, 0).expect("rec -300");
    let f_est_pos = peak_freq_hz_from_spectrum(&rec_pos, sw1, 8);
    let f_est_neg = peak_freq_hz_from_spectrum(&rec_neg, sw1, 8);
    assert!(
        (f_est_pos - 300.0).abs() < 6.0,
        "2D States +300 Hz recovery failed: got {f_est_pos:.2} Hz"
    );
    assert!(
        (f_est_neg - (-300.0)).abs() < 6.0,
        "2D States -300 Hz recovery failed: got {f_est_neg:.2} Hz"
    );

    // 3. Non-trivial random model MUST produce distinct outputs for +300 Hz vs -300 Hz!
    let rand_model = FidNetModel::init_random(FidNetConfig::nus_default(4, 4), 4242).expect("rand");
    let (rout_pos, _) =
        apply_fidnet_fid2d_indirect(&rand_model, &fid_pos, None, &inf_cfg).expect("rand +300");
    let (rout_neg, _) =
        apply_fidnet_fid2d_indirect(&rand_model, &fid_neg, None, &inf_cfg).expect("rand -300");
    let mut rand_max_diff = 0.0_f64;
    for (&[rp, ip], &[rn, ineg]) in rout_pos.data.iter().zip(rout_neg.data.iter()) {
        rand_max_diff = rand_max_diff.max((rp - rn).abs()).max((ip - ineg).abs());
    }
    assert!(
        rand_max_diff > 0.25,
        "2D FID-Net outputs for +300 Hz vs -300 Hz must be distinct (got max_diff={rand_max_diff:.3e})"
    );

    // 4. Physical 3D States verification along BOTH AxisY (±300 Hz) and AxisZ (±400 Hz)
    let n0_3d = 2usize;
    let n1_3d = 32usize;
    let n2_3d = 32usize;
    let sw1_3d = 2000.0;
    let sw2_3d = 2000.0;

    let fid3d_pp =
        encode_physical_states_fid3d(n0_3d, sw0, f0, n1_3d, sw1_3d, 300.0, n2_3d, sw2_3d, 400.0);
    let fid3d_mn =
        encode_physical_states_fid3d(n0_3d, sw0, f0, n1_3d, sw1_3d, -300.0, n2_3d, sw2_3d, -400.0);

    // Verify AxisY complex conjugation (+300 Hz vs -300 Hz at k2=0, c2=0, ix=0, dir_ch=0)
    let y_pos = extract_fid3d_states_interferogram(&fid3d_pp, Fid3DIndirectAxis::AxisY, 0, 0, 0, 0)
        .expect("3d y +300");
    let y_neg = extract_fid3d_states_interferogram(&fid3d_mn, Fid3DIndirectAxis::AxisY, 0, 0, 0, 0)
        .expect("3d y -300");
    let mut y_conj_err = 0.0_f64;
    let mut y_diff = 0.0_f64;
    for (&[rp, ip], &[rn, ineg]) in y_pos.iter().zip(y_neg.iter()) {
        y_conj_err = y_conj_err.max((rp - rn).abs()).max((ip + ineg).abs());
        y_diff = y_diff.max((ip - ineg).abs());
    }
    assert!(y_conj_err < 1e-12 && y_diff > 0.5);

    // Verify AxisZ complex conjugation (+400 Hz vs -400 Hz at k1=0, c1=0, ix=0, dir_ch=0)
    let z_pos = extract_fid3d_states_interferogram(&fid3d_pp, Fid3DIndirectAxis::AxisZ, 0, 0, 0, 0)
        .expect("3d z +400");
    let z_neg = extract_fid3d_states_interferogram(&fid3d_mn, Fid3DIndirectAxis::AxisZ, 0, 0, 0, 0)
        .expect("3d z -400");
    let mut z_conj_err = 0.0_f64;
    let mut z_diff = 0.0_f64;
    for (&[rp, ip], &[rn, ineg]) in z_pos.iter().zip(z_neg.iter()) {
        z_conj_err = z_conj_err.max((rp - rn).abs()).max((ip + ineg).abs());
        z_diff = z_diff.max((ip - ineg).abs());
    }
    assert!(z_conj_err < 1e-12 && z_diff > 0.5);

    // Apply along AxisY and AxisZ with identity model and recover exact signed frequencies
    let (out3_y_pos, diags3_y) = apply_fidnet_fid3d_indirect(
        &id_model,
        &fid3d_pp,
        Fid3DIndirectAxis::AxisY,
        None,
        &inf_cfg,
    )
    .expect("3d apply y +300");
    let (out3_y_neg, _) = apply_fidnet_fid3d_indirect(
        &id_model,
        &fid3d_mn,
        Fid3DIndirectAxis::AxisY,
        None,
        &inf_cfg,
    )
    .expect("3d apply y -300");
    assert_eq!(diags3_y.len(), n2_3d * 4 * n0_3d);

    let ry_pos =
        extract_fid3d_states_interferogram(&out3_y_pos, Fid3DIndirectAxis::AxisY, 0, 0, 0, 0)
            .expect("rec 3d y+");
    let ry_neg =
        extract_fid3d_states_interferogram(&out3_y_neg, Fid3DIndirectAxis::AxisY, 0, 0, 0, 0)
            .expect("rec 3d y-");
    assert!((peak_freq_hz_from_spectrum(&ry_pos, sw1_3d, 8) - 300.0).abs() < 10.0);
    assert!((peak_freq_hz_from_spectrum(&ry_neg, sw1_3d, 8) - (-300.0)).abs() < 10.0);

    let (out3_z_pos, diags3_z) = apply_fidnet_fid3d_indirect(
        &id_model,
        &fid3d_pp,
        Fid3DIndirectAxis::AxisZ,
        None,
        &inf_cfg,
    )
    .expect("3d apply z +400");
    let (out3_z_neg, _) = apply_fidnet_fid3d_indirect(
        &id_model,
        &fid3d_mn,
        Fid3DIndirectAxis::AxisZ,
        None,
        &inf_cfg,
    )
    .expect("3d apply z -400");
    assert_eq!(diags3_z.len(), n1_3d * 4 * n0_3d);

    let rz_pos =
        extract_fid3d_states_interferogram(&out3_z_pos, Fid3DIndirectAxis::AxisZ, 0, 0, 0, 0)
            .expect("rec 3d z+");
    let rz_neg =
        extract_fid3d_states_interferogram(&out3_z_neg, Fid3DIndirectAxis::AxisZ, 0, 0, 0, 0)
            .expect("rec 3d z-");
    assert!((peak_freq_hz_from_spectrum(&rz_pos, sw2_3d, 8) - 400.0).abs() < 10.0);
    assert!((peak_freq_hz_from_spectrum(&rz_neg, sw2_3d, 8) - (-400.0)).abs() < 10.0);

    // 5. Fail-closed on unrecombined or QF indirect axes
    let mut bad_fid2d = fid_pos.clone();
    bad_fid2d.axis_y.states_recombined = false;
    assert!(matches!(
        apply_fidnet_fid2d_indirect(&id_model, &bad_fid2d, None, &inf_cfg),
        Err(FidNetError::InvalidConfig(_))
    ));
    let mut qf_fid2d = fid_pos.clone();
    qf_fid2d.axis_y.quad_mode = QuadMode::Qf;
    assert!(matches!(
        apply_fidnet_fid2d_indirect(&id_model, &qf_fid2d, None, &inf_cfg),
        Err(FidNetError::InvalidConfig(_))
    ));
}

/// `[synthetic self-consistency]` Regression test for `22.2:P1-2`: verify cache-free
/// `forward_fid_inference` matches `forward_features_with_cache` bit-for-bit, and that
/// `FidNetConfig::validate()` rejects OOM-prone configs (`C = 256, K = 1, 64 blocks`) before
/// any allocation.
#[test]
fn test_fidnet_cache_free_inference_parity_and_memory_budget() {
    // 1. Exact parity between cache-free inference and cached forward pass
    for &input_residual_skip in &[true, false] {
        let cfg = FidNetConfig {
            task: FidNetTask::NusReconstruction,
            use_mask_channel: true,
            use_coupling_channel: true,
            residual_channels: 8,
            skip_channels: 6,
            kernel_size: 5,
            dilations: vec![1, 2, 4, 8],
            input_residual_skip,
        };
        let model = FidNetModel::init_random(cfg, 707 + u64::from(input_residual_skip))
            .expect("init model");
        let spec = SyntheticFidSpec {
            len: 64,
            sw_hz: 1000.0,
            spins: vec![SyntheticSpin {
                freq_hz: 135.0,
                r2_hz: 12.0,
                amplitude: 1.0,
                phase_rad: 0.25,
                j_coupling_hz: 35.0,
            }],
            noise_sigma: 0.03,
            nus_fraction: Some(0.5),
            conditioning_j_hz: Some(35.0),
            seed: 12345,
        };
        let s = generate_synthetic_fid(&spec).expect("gen fid");
        let x_in = model
            .build_input_features(
                &s.observed,
                s.nus_mask.as_deref(),
                s.coupling_channel.as_deref(),
            )
            .expect("build features");
        let cached = model
            .forward_features_with_cache(&x_in, spec.len)
            .expect("cached forward");
        let cache_free = model
            .forward_fid_inference(
                &s.observed,
                s.nus_mask.as_deref(),
                s.coupling_channel.as_deref(),
            )
            .expect("cache-free inference");
        assert_eq!(cached.output.len(), cache_free.len());
        let mut max_diff = 0.0_f64;
        for (&[cr, ci], &[ir, ii]) in cached.output.iter().zip(cache_free.iter()) {
            max_diff = max_diff.max((cr - ir).abs()).max((ci - ii).abs());
        }
        assert!(
            max_diff < 1e-14,
            "input_residual_skip={input_residual_skip}: cache-free vs cached max_diff={max_diff:.3e}"
        );
    }

    // 2. Claude probe config: C = 256, K = 1, 64 blocks MUST fail closed in validate()
    let oom_cfg = FidNetConfig {
        task: FidNetTask::NusReconstruction,
        use_mask_channel: true,
        use_coupling_channel: false,
        residual_channels: 256,
        skip_channels: 1,
        kernel_size: 1,
        dilations: vec![1; 64],
        input_residual_skip: true,
    };
    let est_inf_bytes = oom_cfg
        .inference_activation_bytes(MAX_FIDNET_LENGTH)
        .expect("activation bytes calc");
    assert!(est_inf_bytes > MAX_FIDNET_INFERENCE_ACTIVATION_BYTES);
    assert!(
        matches!(oom_cfg.validate(), Err(FidNetError::InvalidConfig(_))),
        "C=256, K=1, 64 blocks must fail validate() before allocating"
    );

    // 3. Valid deep config (C = 64, S = 64, K = 1, 64 blocks): passes validate() for cache-free
    // inference (164 MiB <= 256 MiB), while backprop cache at MAX_FIDNET_LENGTH (~13 GB) is
    // rejected cleanly before allocating.
    let deep_ok_cfg = FidNetConfig {
        task: FidNetTask::Denoising,
        use_mask_channel: false,
        use_coupling_channel: false,
        residual_channels: 64,
        skip_channels: 64,
        kernel_size: 1,
        dilations: vec![1; 64],
        input_residual_skip: true,
    };
    deep_ok_cfg
        .validate()
        .expect("C=64, S=64, 64 blocks should pass cache-free budget");
    let deep_model = FidNetModel::new(
        deep_ok_cfg.clone(),
        FidNetWeights::zeros(&deep_ok_cfg).expect("zeros"),
    )
    .expect("deep model");
    let big_x = vec![0.0_f64; deep_ok_cfg.in_channels() * MAX_FIDNET_LENGTH];
    assert!(
        matches!(
            deep_model.forward_features_with_cache(&big_x, MAX_FIDNET_LENGTH),
            Err(FidNetError::InvalidConfig(_))
        ),
        "backprop cache at MAX_FIDNET_LENGTH on 64-block C=64 model must fail closed before allocating"
    );
}

/// `[synthetic self-consistency]` Regression test for `22.2:P1-3`:
/// 1. `uncertainty_rms` is wired into `AbstainReason::HighUncertainty`.
/// 2. Clean truncated FID (`R2 = 3 Hz`, `64` points) does NOT falsely trigger `OutOfDistributionNoise`.
/// 3. Pure thermal white noise DOES trigger `OutOfDistributionNoise` across multiple seeds.
/// 4. `FidNetTask::Denoising` data-consistency check allows legitimate noise removal (~sigma_noise)
///    while flagging severe deviations far exceeding the noise floor.
#[test]
fn test_fidnet_abstention_uncertainty_truncated_fid_and_denoising_gate() {
    // 1. Clean truncated FID (exact Claude probe: R2 = 3 Hz, 64 points, zero and low noise)
    let id_cfg = FidNetConfig::denoising_default(4, 4);
    let id_model = FidNetModel::new(
        id_cfg.clone(),
        FidNetWeights::zeros(&id_cfg).expect("zeros"),
    )
    .expect("id model");

    for &noise_sigma in &[0.0_f64, 0.01_f64] {
        let trunc_spec = SyntheticFidSpec {
            len: 64,
            sw_hz: 1000.0,
            spins: vec![SyntheticSpin {
                freq_hz: 120.0,
                r2_hz: 3.0,
                amplitude: 1.0,
                phase_rad: 0.0,
                j_coupling_hz: 0.0,
            }],
            noise_sigma,
            nus_fraction: None,
            conditioning_j_hz: None,
            seed: 31415,
        };
        let trunc = generate_synthetic_fid(&trunc_spec).expect("gen truncated fid");
        let out = id_model.infer_fid_with_diagnostics(
            &trunc.observed,
            None,
            None,
            &FidNetInferenceConfig::default(),
        );
        assert!(
            !out.diagnostic
                .abstain_reasons
                .contains(&AbstainReason::OutOfDistributionNoise),
            "clean truncated FID (R2=3 Hz, 64 pts, sigma={noise_sigma}) falsely triggered OutOfDistributionNoise: {:?}",
            out.diagnostic
        );
        assert!(
            !out.diagnostic.abstained,
            "clean truncated FID (sigma={noise_sigma}) should not abstain on identity model: {:?}",
            out.diagnostic
        );
        assert!(
            out.diagnostic.estimated_input_snr > 10.0,
            "expected high spectral SNR for clean truncated FID, got {:.2}",
            out.diagnostic.estimated_input_snr
        );
    }

    // 2. Pure thermal white noise triggers OutOfDistributionNoise reliably across seeds
    let mut ood_hits = 0usize;
    let n_seeds = 12usize;
    for seed in 1000..(1000 + n_seeds as u64) {
        let noise_spec = SyntheticFidSpec {
            len: 64,
            sw_hz: 1000.0,
            spins: vec![],
            noise_sigma: 1.0,
            nus_fraction: None,
            conditioning_j_hz: None,
            seed,
        };
        let pure_noise = generate_synthetic_fid(&noise_spec).expect("pure noise");
        let out = id_model.infer_fid_with_diagnostics(
            &pure_noise.observed,
            None,
            None,
            &FidNetInferenceConfig::default(),
        );
        if out
            .diagnostic
            .abstain_reasons
            .contains(&AbstainReason::OutOfDistributionNoise)
        {
            ood_hits += 1;
        }
    }
    assert!(
        ood_hits >= 10,
        "pure white noise should reliably trigger OutOfDistributionNoise (hit {ood_hits}/{n_seeds})"
    );

    // 3. HighUncertainty abstention trigger: keep stem/blocks in linear tanh/sigmoid regime and
    // amplify post-activation skip/out projections so small input jitter creates large
    // perturbation RMS > max_uncertainty_rel * obs_rms.
    let mut unstable_model =
        FidNetModel::init_random(FidNetConfig::nus_default(6, 6), 909).expect("init unstable");
    for blk in &mut unstable_model.weights.blocks {
        for b in &mut blk.skip_b {
            *b = 0.5;
        }
        for w in &mut blk.skip_w {
            *w *= 8.0;
        }
    }
    for b in &mut unstable_model.weights.head1_b {
        *b = 0.5;
    }
    for w in &mut unstable_model.weights.out_w {
        *w *= 80.0;
    }
    let clean_spec = SyntheticFidSpec {
        len: 48,
        sw_hz: 1000.0,
        spins: vec![SyntheticSpin {
            freq_hz: 110.0,
            r2_hz: 12.0,
            amplitude: 1.0,
            phase_rad: 0.0,
            j_coupling_hz: 0.0,
        }],
        noise_sigma: 0.01,
        nus_fraction: None,
        conditioning_j_hz: None,
        seed: 5555,
    };
    let clean = generate_synthetic_fid(&clean_spec).expect("gen clean");
    let unc_out = unstable_model.infer_fid_with_diagnostics(
        &clean.observed,
        None,
        None,
        &FidNetInferenceConfig {
            uncertainty_passes: 4,
            uncertainty_jitter_rel: 0.02,
            max_uncertainty_rel: 0.20,
            ..FidNetInferenceConfig::default()
        },
    );
    assert!(
        unc_out.diagnostic.abstained
            && unc_out
                .diagnostic
                .abstain_reasons
                .contains(&AbstainReason::HighUncertainty),
        "expected HighUncertainty abstention on unstable high-gain model, got {:?}",
        unc_out.diagnostic
    );

    // 4. Denoising data-consistency gate: train a small denoiser (or verify on noisy FID with
    // sigma = 0.25 where true noise removal differs from input by ~0.35..0.45 relative L2).
    let den_cfg = FidNetConfig::denoising_default(6, 6);
    let mut den_model = FidNetModel::init_random(den_cfg.clone(), 404).expect("denoise model");
    let mut adam = FidNetAdamState::new(
        &den_cfg,
        FidNetAdamConfig {
            lr: 8e-3,
            max_grad_norm: 5.0,
            ..FidNetAdamConfig::default()
        },
    )
    .expect("adam");
    let noisy_spec = SyntheticFidSpec {
        len: 64,
        sw_hz: 800.0,
        spins: vec![
            SyntheticSpin {
                freq_hz: -90.0,
                r2_hz: 12.0,
                amplitude: 1.0,
                phase_rad: 0.0,
                j_coupling_hz: 0.0,
            },
            SyntheticSpin {
                freq_hz: 85.0,
                r2_hz: 12.0,
                amplitude: 0.9,
                phase_rad: 0.1,
                j_coupling_hz: 0.0,
            },
        ],
        noise_sigma: 0.25,
        nus_fraction: None,
        conditioning_j_hz: None,
        seed: 6060,
    };
    let noisy_sample = generate_synthetic_fid(&noisy_spec).expect("gen noisy");
    let train_ex = vec![FidNetTrainExample {
        input_fid: noisy_sample.observed.clone(),
        nus_mask: None,
        coupling_channel: None,
        target_fid: noisy_sample.clean_singlet.clone(),
        loss_weights: None,
    }];
    for _ in 0..80 {
        let _ = train_fidnet_step(&mut den_model, &mut adam, &train_ex).expect("step");
    }
    let den_out = den_model.infer_fid_with_diagnostics(
        &noisy_sample.observed,
        None,
        None,
        &FidNetInferenceConfig::default(),
    );
    assert!(
        !den_out
            .diagnostic
            .abstain_reasons
            .contains(&AbstainReason::DataInconsistency),
        "accurate denoising (removing ~sigma_noise) must NOT trigger DataInconsistency: {:?}",
        den_out.diagnostic
    );

    // Corrupting the denoiser with a huge DC bias far exceeding sigma_noise MUST still trigger DataInconsistency!
    let mut broken_den = den_model;
    broken_den.weights.out_b = vec![4.5, -4.5];
    let broken_out = broken_den.infer_fid_with_diagnostics(
        &noisy_sample.observed,
        None,
        None,
        &FidNetInferenceConfig::default(),
    );
    assert!(
        broken_out
            .diagnostic
            .abstain_reasons
            .contains(&AbstainReason::DataInconsistency),
        "broken denoiser far exceeding sigma_noise must trigger DataInconsistency: {:?}",
        broken_out.diagnostic
    );
}

/// `[synthetic self-consistency]` Verify 2D and 3D indirect-column batch application
/// (`apply_fidnet_2d_columns`, `apply_fidnet_fid2d_indirect`, `apply_fidnet_3d_columns`,
/// `apply_fidnet_fid3d_indirect`) using physical States hypercomplex fixtures.
#[test]
fn test_fidnet_2d_and_3d_indirect_column_application() {
    let cfg = FidNetConfig::nus_default(4, 4);
    let model = FidNetModel::init_random(cfg, 999).expect("init model");
    let inf_cfg = FidNetInferenceConfig {
        uncertainty_passes: 2,
        ..FidNetInferenceConfig::default()
    };

    let n0 = 4usize;
    let n1 = 16usize;
    let n2 = 8usize;
    let mask_1d = generate_poisson_gap_nus_mask(n1, 0.5, 42).expect("nus mask");

    // Generic 2D column batch
    let cols_2d: Vec<Vec<Complex64>> = (0..n0)
        .map(|ix| {
            let spec = SyntheticFidSpec {
                len: n1,
                sw_hz: 500.0,
                spins: vec![SyntheticSpin {
                    freq_hz: 40.0 + 15.0 * (ix as f64),
                    r2_hz: 15.0,
                    amplitude: 1.0,
                    phase_rad: 0.0,
                    j_coupling_hz: 0.0,
                }],
                noise_sigma: 0.01,
                nus_fraction: None,
                conditioning_j_hz: None,
                seed: 700 + ix as u64,
            };
            let s = generate_synthetic_fid(&spec).expect("gen col");
            s.observed
                .iter()
                .zip(mask_1d.iter())
                .map(|(&v, &m)| if m { v } else { [0.0, 0.0] })
                .collect()
        })
        .collect();

    let out_2d =
        apply_fidnet_2d_columns(&model, &cols_2d, Some(&mask_1d), None, &inf_cfg).expect("2d cols");
    assert_eq!(out_2d.len(), n0);
    for res in &out_2d {
        assert_eq!(res.reconstructed.len(), n1);
        assert!(res
            .reconstructed
            .iter()
            .all(|&[r, i]| r.is_finite() && i.is_finite()));
        assert!(res.diagnostic.uncertainty_rms.is_finite());
    }

    // Physical States Fid2D indirect axis application with 50% NUS mask
    let fid2d = encode_physical_states_fid2d(
        n0,
        8000.0,
        400.0,
        15.0,
        n1,
        2000.0,
        250.0,
        12.0,
        Some(mask_1d.clone()),
    );
    let (recon_2d, diags_2d) =
        apply_fidnet_fid2d_indirect(&model, &fid2d, None, &inf_cfg).expect("fid2d indirect");
    assert_eq!(diags_2d.len(), 2 * n0);
    assert!(recon_2d.nus_mask.is_none());
    assert_eq!(recon_2d.data.len(), fid2d.data.len());

    // Generic 3D column batch
    let planes_3d = vec![cols_2d.clone(), cols_2d.clone()];
    let out_3d = apply_fidnet_3d_columns(&model, &planes_3d, Some(&mask_1d), None, &inf_cfg)
        .expect("3d cols");
    assert_eq!(out_3d.len(), 2);
    assert_eq!(out_3d[0].len(), n0);

    // Physical States Fid3D indirect axis application (both AxisY and AxisZ)
    let mut fid3d =
        encode_physical_states_fid3d(n0, 8000.0, 400.0, n1, 2000.0, 220.0, n2, 4000.0, -310.0);
    fid3d.nus_mask = Some(vec![true; n2 * n1]);
    let (recon_3d_y, diags_y) =
        apply_fidnet_fid3d_indirect(&model, &fid3d, Fid3DIndirectAxis::AxisY, None, &inf_cfg)
            .expect("fid3d axis y");
    assert_eq!(diags_y.len(), n2 * 4 * n0);
    assert_eq!(recon_3d_y.data.len(), fid3d.data.len());

    let (recon_3d_z, diags_z) =
        apply_fidnet_fid3d_indirect(&model, &fid3d, Fid3DIndirectAxis::AxisZ, None, &inf_cfg)
            .expect("fid3d axis z");
    assert_eq!(diags_z.len(), n1 * 4 * n0);
    assert_eq!(recon_3d_z.data.len(), fid3d.data.len());

    // Coupling channel helper sanity check
    let cc = build_coupling_channel(16, 35.0, 1000.0).expect("coupling channel");
    assert_eq!(cc.len(), 16);
    assert!((cc[0] - 1.0).abs() < 1e-12);
}
