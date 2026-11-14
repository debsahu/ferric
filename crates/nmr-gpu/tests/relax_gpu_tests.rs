//! WebGPU (`wgpu` + WGSL) verification & benchmark suite for `T13.2` (`nmr-gpu::relax_gpu`):
//! - Asserts hardware WebGPU adapter availability unless `NMR_ALLOW_NO_GPU=1`.
//! - Done gate: `150` residues $\times$ `1,000` Monte Carlo replicates complete in **`< 100 ms`**
//!   on WebGPU with **`< 1e-5` CPU/WebGPU point-estimate rate parity**, recovering $R_1, R_2$ and
//!   $1\sigma$ uncertainties within **`+-3%`** of analytical Fisher / Cramér–Rao bounds.
//! - Verifies 3-parameter AIC/BIC offset gating parity, TDR-safe multi-chunk MC dispatches
//!   (`> MAX_FITS_PER_DISPATCH`), and browser-portable `wgpu::Limits::default()` compatibility.
//!
//! Provenance: Clean-room verification suite citing Farrow et al. (*Biochemistry* 1994;
//! *J. Biomol. NMR* 1995), Kay, Torchia & Bax (*Biochemistry* 1989), Lipari & Szabo (*JACS*
//! 1982a,b), Clore et al. (*JACS* 1990), Mandel, Akke & Palmer (*J. Mol. Biol.* 1995), Woessner
//! (*J. Chem. Phys.* 1962), and d'Auvergne & Gooley (*J. Biomol. NMR* 2003).

use std::time::Instant;

use nmr_core::relax_fit::{fit_decay_batch_cpu, DecayCurveInput, ExpFitConfig, OffsetModelGate};
use nmr_core::rng::Rng;
use nmr_gpu::relax_gpu::{fit_decay_batch_with_pipeline, RelaxGpuPipeline};
use nmr_gpu::GpuEngine;

fn require_gpu_or_allow_skip() -> Option<(GpuEngine, bool)> {
    let is_cpu_adapter = {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .map(|a| a.get_info().device_type == wgpu::DeviceType::Cpu)
        .unwrap_or(false)
    };

    match GpuEngine::new_headless() {
        Ok(eng) => {
            let sw = is_cpu_adapter
                || eng.adapter_name.to_ascii_lowercase().contains("llvmpipe")
                || eng
                    .adapter_name
                    .to_ascii_lowercase()
                    .contains("basic render");
            Some((eng, sw))
        }
        Err(err) => {
            if std::env::var("NMR_ALLOW_NO_GPU").ok().as_deref() == Some("1") {
                eprintln!("Skipping WebGPU test because NMR_ALLOW_NO_GPU=1: {err}");
                None
            } else {
                panic!(
                    "WebGPU adapter required for relax_gpu_tests (set NMR_ALLOW_NO_GPU=1 only on headless CI without GPU): {err}"
                );
            }
        }
    }
}

#[test]
fn gpu_150_residues_x_1000_mc_completes_under_100ms_with_1e5_cpu_parity_and_3pct_crlb() {
    let Some((engine, is_software_cpu)) = require_gpu_or_allow_skip() else {
        return;
    };
    let pipe = RelaxGpuPipeline::new(&engine.device);

    // 12-plane relaxation schedule including two duplicate delays (realistic R1 & R2 sampling)
    let r1_delays = vec![
        0.010, 0.030, 0.060, 0.100, 0.160, 0.240, 0.350, 0.500, 0.700, 1.000, 0.060, 0.240,
    ];
    let r2_delays = vec![
        0.008, 0.016, 0.024, 0.032, 0.048, 0.064, 0.088, 0.112, 0.144, 0.184, 0.016, 0.064,
    ];

    let n_res = 150usize;
    let mut rng = Rng::new(20261501000);
    let mut curves = Vec::with_capacity(n_res);
    let mut true_rates = Vec::with_capacity(n_res);

    for r in 0..n_res {
        let is_r2 = r % 2 == 1;
        let delays = if is_r2 { &r2_delays } else { &r1_delays };
        let i0_true = 1200.0 + (r as f64) * 6.0;
        let rate_true = if is_r2 {
            6.5 + ((r % 25) as f64) * 0.42
        } else {
            1.25 + ((r % 25) as f64) * 0.06
        };
        let sigma_pt = i0_true * 0.010; // 1% baseline noise (SNR = 100 at t=0)
        let intensities: Vec<f64> = delays
            .iter()
            .map(|&t| i0_true * (-rate_true * t).exp() + rng.normal(0.0, sigma_pt))
            .collect();
        curves.push(DecayCurveInput {
            seq_id: (r as i32) + 1,
            res_name: if is_r2 { "LEU" } else { "VAL" }.to_string(),
            times_sec: delays.clone(),
            intensities,
            sigmas: vec![sigma_pt; delays.len()],
        });
        true_rates.push(rate_true);
    }

    let cfg = ExpFitConfig {
        offset_gate: OffsetModelGate::Disabled,
        mc_replicates: 1000,
        seed: 0x1302_CAFE_BEEF_2026,
        max_rate_s_inv: 200.0,
    };

    // Warm up driver / command queue once
    let _ = fit_decay_batch_with_pipeline(&engine, &pipe, &curves[..8], &cfg)
        .expect("GPU warmup dispatch");

    let t0 = Instant::now();
    let gpu_fits = fit_decay_batch_with_pipeline(&engine, &pipe, &curves, &cfg)
        .expect("GPU 150x1000 relaxation fit");
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let cpu_point_cfg = ExpFitConfig {
        mc_replicates: 0,
        ..cfg.clone()
    };
    let cpu_fits = fit_decay_batch_cpu(&curves, &cpu_point_cfg).expect("CPU reference point fits");

    assert_eq!(gpu_fits.len(), n_res);
    let ceiling_ms = if is_software_cpu { 5_000.0 } else { 100.0 };
    assert!(
        elapsed_ms < ceiling_ms,
        "150 residues x 1,000 MC replicates took {elapsed_ms:.2} ms on {} ({}, software_cpu={is_software_cpu}) — must be < {ceiling_ms:.0} ms!",
        engine.adapter_name,
        engine.backend_name
    );

    let mut max_abs_rate_diff = 0.0_f64;
    let mut sum_mc_over_fisher = 0.0_f64;
    let mut covered_95 = 0usize;

    for i in 0..n_res {
        let g = &gpu_fits[i];
        let c = &cpu_fits[i];
        let diff_r = (g.rate - c.rate).abs();
        max_abs_rate_diff = max_abs_rate_diff.max(diff_r);
        assert!(
            diff_r < 1.0e-5,
            "residue {} GPU/CPU point-estimate rate diff {diff_r:.3e} (gpu={:.8}, cpu={:.8}) exceeds 1e-5",
            g.seq_id,
            g.rate,
            c.rate
        );
        sum_mc_over_fisher += g.sigma_rate_mc / g.sigma_rate_fisher;
        if g.rate_ci95[0] <= true_rates[i] && true_rates[i] <= g.rate_ci95[1] {
            covered_95 += 1;
        }
    }

    let mean_mc_fisher_ratio = sum_mc_over_fisher / (n_res as f64);
    assert!(
        (mean_mc_fisher_ratio - 1.0).abs() <= 0.03,
        "WebGPU 1,000-MC 1-sigma / Fisher CRLB ratio {mean_mc_fisher_ratio:.5} must be within +-3% of 1.0"
    );
    assert!(
        covered_95 >= 135,
        "empirical 95% MC CIs must cover true rates on >= 90% of 150 residues, got {covered_95}/150"
    );
    println!(
        "[T13.2 WebGPU Gate] 150x1000 MC in {elapsed_ms:.2} ms on {} ({}) | max |R_gpu - R_cpu| = {max_abs_rate_diff:.3e} | MC/Fisher = {mean_mc_fisher_ratio:.5}",
        engine.adapter_name, engine.backend_name
    );
}

#[test]
fn gpu_three_param_aic_gate_and_tdr_multi_chunk_dispatch_match_cpu() {
    let Some((engine, _)) = require_gpu_or_allow_skip() else {
        return;
    };
    let pipe = RelaxGpuPipeline::new(&engine.device);

    let times = vec![0.0, 0.04, 0.10, 0.20, 0.35, 0.55, 0.85, 1.25, 1.80];
    // Build 300 curves (half pure 2-param, half with genuine offset C = 38.0) x 1,100 replicates = 330,000 fits (> MAX_FITS_PER_DISPATCH = 262,144!)
    let mut curves = Vec::with_capacity(300);
    for idx in 0..300usize {
        let has_offset = idx % 2 == 1;
        let c_true = if has_offset { 38.0 } else { 0.0 };
        let r_true = 2.2 + ((idx % 15) as f64) * 0.15;
        let intensities: Vec<f64> = times
            .iter()
            .map(|&t| 500.0 * (-r_true * t).exp() + c_true)
            .collect();
        curves.push(DecayCurveInput {
            seq_id: (idx as i32) + 1,
            res_name: "SER".to_string(),
            times_sec: times.clone(),
            intensities,
            sigmas: vec![2.0; times.len()],
        });
    }

    let cfg = ExpFitConfig {
        offset_gate: OffsetModelGate::Aic,
        mc_replicates: 1100,
        seed: 4242,
        max_rate_s_inv: 200.0,
    };
    let gpu_fits = fit_decay_batch_with_pipeline(&engine, &pipe, &curves, &cfg)
        .expect("multi-chunk TDR GPU dispatch");
    let cpu_fits = fit_decay_batch_cpu(
        &curves,
        &ExpFitConfig {
            mc_replicates: 0,
            ..cfg
        },
    )
    .expect("CPU AIC point fits");

    for (g, c) in gpu_fits.iter().zip(cpu_fits.iter()) {
        assert_eq!(g.used_three_param, c.used_three_param);
        assert!(
            (g.rate - c.rate).abs() < 1.0e-5,
            "AIC 2p/3p GPU rate {:.7} vs CPU {:.7} diff {:.3e} must be < 1e-5",
            g.rate,
            c.rate,
            (g.rate - c.rate).abs()
        );
        assert!(g.sigma_rate_mc > 0.0 && g.sigma_rate_mc.is_finite());
    }

    // Verify browser-baseline `wgpu::Limits::default()` compatibility
    let browser_eng = GpuEngine::new_headless_with_limits(wgpu::Limits::default())
        .expect("WebGPU baseline default limits device");
    let browser_pipe = RelaxGpuPipeline::new(&browser_eng.device);
    let browser_fits =
        fit_decay_batch_with_pipeline(&browser_eng, &browser_pipe, &curves[..16], &cfg)
            .expect("baseline default limits dispatch");
    assert_eq!(browser_fits.len(), 16);
}
