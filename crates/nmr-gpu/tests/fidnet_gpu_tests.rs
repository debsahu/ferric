//! `[synthetic self-consistency]` WebGPU (`wgpu` + WGSL) symmetric dilated-convolution `FID-Net`
//! numerical parity tests against `nmr_core::fidnet::FidNetModel` (`f64` CPU reference) and
//! hostile/limit guards.
//!
//! Note: tests exercise the untrained architecture with deterministic random weights (no bundled
//! pretrained weights).

use nmr_core::fid::Complex64;
use nmr_core::fidnet::{
    generate_synthetic_fid, FidNetConfig, FidNetModel, FidNetTask, SyntheticFidSpec, SyntheticSpin,
    MAX_FIDNET_LENGTH,
};
use nmr_gpu::fidnet_gpu::{validate_fidnet_gpu_dispatch_limits, GpuFidNet};
use nmr_gpu::GpuEngine;

/// `[synthetic self-consistency]` Verify CPU (`f64`) vs WebGPU (`f32`) forward parity
/// (`max |gpu - cpu| < 1e-4`) across single FIDs and 2D batches covering all task configurations
/// (NUS with mask channel, virtual decoupling with coupling channel, denoising, asymmetric
/// `residual_channels != skip_channels`, and `input_residual_skip = true/false`).
#[test]
fn test_fidnet_cpu_vs_webgpu_batched_parity() {
    let engine = match GpuEngine::new_headless() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Skipping WebGPU FID-Net test (no GPU adapter available): {err}");
            return;
        }
    };
    println!(
        "Testing WebGPU FID-Net on {} ({})",
        engine.adapter_name, engine.backend_name
    );

    let configs = [
        FidNetConfig {
            task: FidNetTask::NusReconstruction,
            use_mask_channel: true,
            use_coupling_channel: false,
            residual_channels: 12,
            skip_channels: 16,
            kernel_size: 5,
            dilations: vec![1, 2, 4, 8],
            input_residual_skip: true,
        },
        FidNetConfig {
            task: FidNetTask::VirtualDecoupling,
            use_mask_channel: false,
            use_coupling_channel: true,
            residual_channels: 10,
            skip_channels: 8,
            kernel_size: 7,
            dilations: vec![1, 2, 4, 8, 16],
            input_residual_skip: false,
        },
        FidNetConfig {
            task: FidNetTask::Denoising,
            use_mask_channel: true,
            use_coupling_channel: true,
            residual_channels: 8,
            skip_channels: 12,
            kernel_size: 3,
            dilations: vec![1, 2, 4],
            input_residual_skip: true,
        },
    ];

    let batch_size = 16usize;
    let t_len = 128usize;

    for (ci, cfg) in configs.iter().enumerate() {
        let model = FidNetModel::init_random(cfg.clone(), 1234 + ci as u64).expect("init model");
        let gpu_model = GpuFidNet::from_model(&engine, &model).expect("GpuFidNet::from_model");

        let mut fids = Vec::with_capacity(batch_size);
        let mut masks = Vec::with_capacity(batch_size);
        let mut couplings = Vec::with_capacity(batch_size);

        for bi in 0..batch_size {
            let spec = SyntheticFidSpec {
                len: t_len,
                sw_hz: 1200.0,
                spins: vec![
                    SyntheticSpin {
                        freq_hz: -220.0 + 25.0 * (bi as f64),
                        r2_hz: 14.0,
                        amplitude: 1.0,
                        phase_rad: 0.1 * (bi as f64),
                        j_coupling_hz: if cfg.use_coupling_channel { 35.0 } else { 0.0 },
                    },
                    SyntheticSpin {
                        freq_hz: 180.0 - 15.0 * (bi as f64),
                        r2_hz: 18.0,
                        amplitude: 0.65,
                        phase_rad: -0.2,
                        j_coupling_hz: if cfg.use_coupling_channel { 35.0 } else { 0.0 },
                    },
                ],
                noise_sigma: 0.08,
                nus_fraction: if cfg.use_mask_channel {
                    Some(0.35)
                } else {
                    None
                },
                conditioning_j_hz: if cfg.use_coupling_channel {
                    Some(35.0)
                } else {
                    None
                },
                seed: 50_000 + (ci as u64) * 100 + bi as u64,
            };
            let s = generate_synthetic_fid(&spec).expect("gen synthetic fid");
            fids.push(s.observed);
            masks.push(s.nus_mask.unwrap_or_else(|| vec![true; t_len]));
            couplings.push(s.coupling_channel.unwrap_or_else(|| vec![1.0; t_len]));
        }

        let mask_arg = if cfg.use_mask_channel {
            Some(masks.as_slice())
        } else {
            None
        };
        let cc_arg = if cfg.use_coupling_channel {
            Some(couplings.as_slice())
        } else {
            None
        };

        // 1. Single-FID parity check
        let cpu_single = model
            .forward_fid(
                &fids[0],
                mask_arg.map(|m| m[0].as_slice()),
                cc_arg.map(|c| c[0].as_slice()),
            )
            .expect("cpu single forward");
        let gpu_single = gpu_model
            .forward_fid_gpu(
                &engine,
                &fids[0],
                mask_arg.map(|m| m[0].as_slice()),
                cc_arg.map(|c| c[0].as_slice()),
            )
            .expect("gpu single forward");

        assert_eq!(cpu_single.len(), gpu_single.len());
        let mut max_single_diff = 0.0f64;
        for (c, g) in cpu_single.iter().zip(gpu_single.iter()) {
            max_single_diff = max_single_diff.max((c[0] - g[0]).abs());
            max_single_diff = max_single_diff.max((c[1] - g[1]).abs());
        }
        assert!(
            max_single_diff < 1e-4,
            "config {ci} single FID max |gpu - cpu| = {max_single_diff:.3e} >= 1e-4"
        );

        // 2. Full 2D batch parity check
        let gpu_batch = gpu_model
            .forward_batch_gpu(&engine, &fids, mask_arg, cc_arg)
            .expect("gpu batch forward");
        assert_eq!(gpu_batch.len(), batch_size);

        let mut max_batch_diff = 0.0f64;
        for bi in 0..batch_size {
            let cpu_out = model
                .forward_fid(
                    &fids[bi],
                    mask_arg.map(|m| m[bi].as_slice()),
                    cc_arg.map(|c| c[bi].as_slice()),
                )
                .expect("cpu batch item forward");
            assert_eq!(cpu_out.len(), gpu_batch[bi].len());
            for (c, g) in cpu_out.iter().zip(gpu_batch[bi].iter()) {
                assert!(g[0].is_finite() && g[1].is_finite());
                max_batch_diff = max_batch_diff.max((c[0] - g[0]).abs());
                max_batch_diff = max_batch_diff.max((c[1] - g[1]).abs());
            }
        }
        println!(
            "Config {ci} ({:?}, res={}, skip={}, k={}, skip_in={}): max |gpu - cpu| = {max_batch_diff:.3e}",
            cfg.task, cfg.residual_channels, cfg.skip_channels, cfg.kernel_size, cfg.input_residual_skip
        );
        assert!(
            max_batch_diff < 1e-4,
            "config {ci} batched FID max |gpu - cpu| = {max_batch_diff:.3e} >= 1e-4"
        );
    }
}

/// `[synthetic self-consistency]` Verify GPU buffer-limit validation and hostile/empty input guards
/// (no panics, no zero-sized WebGPU buffer creation).
#[test]
fn test_fidnet_gpu_limits_and_hostile_input_guards() {
    let cfg = FidNetConfig::nus_default(16, 16);

    // 1. Validate dispatch limits reject buffers exceeding baseline WebGPU 128 MiB limit
    let default_limits = wgpu::Limits::default();
    // batch=4096, t_len=1024, ch=16 => 4096 * 16 * 1024 * 4 B = 256 MiB > 128 MiB
    let limit_err = validate_fidnet_gpu_dispatch_limits(&default_limits, &cfg, 4096, 1024);
    assert!(
        limit_err.is_err(),
        "expected dispatch exceeding default 128 MiB storage limit to fail validation"
    );

    // Valid small dispatch passes
    assert!(validate_fidnet_gpu_dispatch_limits(&default_limits, &cfg, 16, 128).is_ok());

    // Oversized FID rejected by dispatch validator
    assert!(
        validate_fidnet_gpu_dispatch_limits(&default_limits, &cfg, 1, MAX_FIDNET_LENGTH + 1)
            .is_err()
    );

    // 2. Runtime hostile-input checks on actual headless WebGPU device (if available)
    let engine = match GpuEngine::new_headless() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Skipping runtime GPU hostile-input test (no adapter): {err}");
            return;
        }
    };
    let model = FidNetModel::init_random(cfg, 8080).expect("init model");
    let gpu = GpuFidNet::from_model(&engine, &model).expect("GpuFidNet::from_model");

    // Empty batch -> Ok(empty) without allocating 0-byte GPU buffers
    let empty_res = gpu
        .forward_batch_gpu(&engine, &[], None, None)
        .expect("empty batch should return Ok([])");
    assert!(empty_res.is_empty());

    // Empty individual FID -> Err
    assert!(gpu
        .forward_batch_gpu(&engine, &[Vec::new()], None, None)
        .is_err());

    // Mismatched FID lengths in batch -> Err
    assert!(gpu
        .forward_batch_gpu(
            &engine,
            &[vec![[1.0, 0.0]; 16], vec![[1.0, 0.0]; 32]],
            None,
            None,
        )
        .is_err());

    // Non-finite input (NaN / Inf) -> Err
    let mut bad_fid: Vec<Complex64> = vec![[0.5, -0.5]; 16];
    bad_fid[4][0] = f64::NAN;
    assert!(gpu.forward_fid_gpu(&engine, &bad_fid, None, None).is_err());

    bad_fid[4][0] = f64::INFINITY;
    assert!(gpu.forward_fid_gpu(&engine, &bad_fid, None, None).is_err());

    // Oversized FID -> Err
    let huge = vec![[0.0, 0.0]; MAX_FIDNET_LENGTH + 1];
    assert!(gpu.forward_fid_gpu(&engine, &huge, None, None).is_err());
}
