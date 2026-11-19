//! `[synthetic self-consistency]` Unit and integration tests for `nmr_core::fidnet`
//! (WaveNet / dilated-convolution `FID-Net` / `FID-Net-2` engine).
//!
//! All tests in this suite validate analytical gradients, synthetic spin-Hamiltonian
//! convergence (25% NUS reconstruction, virtual 13C-13C J=35 Hz decoupling, denoising),
//! strict `FIDNET-V1` SHA-256 weight-bundle integrity, hostile-input rejection,
//! data-consistency abstention diagnostics, and 2D/3D indirect-column batch application.

use nmr_core::fid::{Complex64, Fid2D, Fid3D, FidAxisMeta, FidVendor, QuadMode, StoredDimProc};
use nmr_core::fidnet::{
    apply_fidnet_2d_columns, apply_fidnet_3d_columns, apply_fidnet_fid2d_indirect,
    apply_fidnet_fid3d_indirect, build_coupling_channel, compute_fid_mse, compute_fid_snr_db,
    fid_to_magnitude_spectrum, generate_poisson_gap_nus_mask, generate_synthetic_fid,
    train_fidnet_step, AbstainReason, Fid3DIndirectAxis, FidNetAdamConfig, FidNetAdamState,
    FidNetBundleMetadata, FidNetConfig, FidNetError, FidNetInferenceConfig, FidNetModel,
    FidNetTask, FidNetTrainExample, FidNetWeightBundle, FidNetWeights, SyntheticFidSpec,
    SyntheticSpin, FIDNET_BUNDLE_FORMAT_VERSION, MAX_FIDNET_CHANNELS, MAX_FIDNET_LENGTH,
};
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
    for &output_residual in &[true, false] {
        let cfg = FidNetConfig {
            task: FidNetTask::NusReconstruction,
            use_mask_channel: true,
            use_coupling_channel: true,
            residual_channels: 4,
            skip_channels: 3,
            kernel_size: 3,
            dilations: vec![1, 2, 4],
            output_residual,
        };
        let model = FidNetModel::init_random(cfg.clone(), 42 + u64::from(output_residual))
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
            "output_residual={output_residual}: analytical vs FD max_abs_err={max_abs_err:.3e} >= 1e-5"
        );
        assert!(
            max_rel_err < 1e-4,
            "output_residual={output_residual}: analytical vs FD max_rel_err={max_rel_err:.3e} >= 1e-4"
        );
    }
}

/// `[synthetic self-consistency]` Train a compact `FID-Net` on synthetic 25% Poisson-gap NUS
/// multi-spin complex FIDs and verify real loss reduction + SNR/MSE gain over zero-filled NUS.
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
        output_residual: true,
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
    let mut recon_mse_sum = 0.0;
    let mut baseline_snr_sum = 0.0;
    let mut recon_snr_sum = 0.0;
    for ex in &eval_set {
        let pred = model
            .forward_fid(&ex.input_fid, ex.nus_mask.as_deref(), None)
            .expect("eval forward");
        baseline_mse_sum += compute_fid_mse(&ex.target_fid, &ex.input_fid);
        recon_mse_sum += compute_fid_mse(&ex.target_fid, &pred);
        baseline_snr_sum += compute_fid_snr_db(&ex.target_fid, &ex.input_fid);
        recon_snr_sum += compute_fid_snr_db(&ex.target_fid, &pred);
    }

    let n_eval = eval_set.len() as f64;
    let base_mse = baseline_mse_sum / n_eval;
    let rec_mse = recon_mse_sum / n_eval;
    let base_snr = baseline_snr_sum / n_eval;
    let rec_snr = recon_snr_sum / n_eval;
    println!(
        "[synthetic self-consistency] 25% NUS: loss {early_loss:.4}->{late_loss:.4}, MSE {base_mse:.4}->{rec_mse:.4}, SNR {base_snr:.2}->{rec_snr:.2} dB"
    );

    assert!(
        rec_mse < base_mse * 0.45,
        "25% NUS MSE did not improve substantially: baseline={base_mse:.5}, recon={rec_mse:.5}"
    );
    assert!(
        rec_snr > base_snr + 3.5,
        "25% NUS SNR gain too small: baseline={base_snr:.2} dB, recon={rec_snr:.2} dB"
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
        output_residual: true,
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
        output_residual: true,
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
}

/// `[synthetic self-consistency]` Verify 2D and 3D indirect-column batch application
/// (`apply_fidnet_2d_columns`, `apply_fidnet_fid2d_indirect`, `apply_fidnet_3d_columns`,
/// `apply_fidnet_fid3d_indirect`).
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

    // Raw Fid2D indirect axis application
    let mut fid2d_data = vec![[0.0; 2]; n1 * 2 * n0];
    for comp in 0..2 {
        for ix in 0..n0 {
            for k1 in 0..n1 {
                fid2d_data[(k1 * 2 + comp) * n0 + ix] = cols_2d[ix][k1];
            }
        }
    }
    let fid2d = Fid2D {
        name: "synthetic_hsqc".to_string(),
        vendor: FidVendor::Bruker,
        pulse_program: "hsqcetfpf3gpsi".to_string(),
        axis_x: make_axis("1H", n0, 8000.0),
        axis_y: make_axis("15N", n1, 2000.0),
        data: fid2d_data,
        nus_mask: Some(mask_1d.clone()),
        truncated_traces: None,
    };
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

    // Raw Fid3D indirect axis application (both AxisY and AxisZ)
    let mut fid3d_data = vec![[0.0; 2]; n2 * 2 * n1 * 2 * n0];
    for (idx, slot) in fid3d_data.iter_mut().enumerate() {
        let phase = (idx as f64) * 0.1;
        *slot = [phase.cos(), phase.sin()];
    }
    let fid3d = Fid3D {
        name: "synthetic_hnca".to_string(),
        vendor: FidVendor::Bruker,
        pulse_program: "hncagp3d".to_string(),
        axis_x: make_axis("1H", n0, 8000.0),
        axis_y: make_axis("15N", n1, 2000.0),
        axis_z: make_axis("13C", n2, 4000.0),
        data: fid3d_data,
        nus_mask: Some(vec![true; n2 * n1]),
        truncated_traces: None,
    };
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
