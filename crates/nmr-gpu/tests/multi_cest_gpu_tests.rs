//! WebGPU parity & hostile-input verification suite for `nmr_gpu::multi_cest_gpu` (`T22.6`).
//!
//! All synthetic `DANTE-CEST` and `cos-CEST` profile benchmarks here are `[synthetic self-consistency]`
//! tests verifying CPU (`f64`) vs WebGPU (`f32`) parity `< 2e-4`, Dawn-clean WGSL shader execution,
//! baseline `wgpu::Limits::default()` compatibility, and hostile-input rejection.
//!
//! Provenance: Cites Carneiro et al. (*J. Biomol. NMR* 2015, 62, 237–246), Yuwen, Kay &
//! Bouvignies (*J. Phys. Chem. Lett.* 2018, 9, 2501–2507), Vallurupalli, Bouvignies & Kay
//! (*JACS* 2012, 134, 8148–8161), and Higham (*SIAM J. Matrix Anal. Appl.* 2005, 26, 1179–1193).

use std::time::Instant;

use nmr_core::cest::CestNormalization;
use nmr_core::multi_cest::{
    simulate_cos_cest_profile, simulate_dante_cest_profile, simulate_multi_cest_profile,
    CosCestParams, DanteCestParams, MultiCestScheme, MultiCestSimParams, MAX_COS_CEST_MOD_FREQ_HZ,
    MAX_COS_CEST_STEPS_PER_CYCLE, MAX_DANTE_PULSES,
};
use nmr_gpu::multi_cest_gpu::{
    simulate_cos_cest_batch_gpu, simulate_dante_cest_batch_gpu, simulate_multi_cest_batch_gpu,
    validate_multi_cest_storage_bytes,
};
use nmr_gpu::GpuEngine;

fn require_gpu_engine() -> Option<GpuEngine> {
    match GpuEngine::new_headless() {
        Ok(eng) => Some(eng),
        Err(e) => {
            if std::env::var("NMR_ALLOW_NO_GPU").as_deref() == Ok("1") {
                eprintln!("Skipping multi_cest_gpu_tests because NMR_ALLOW_NO_GPU=1 ({e})");
                None
            } else {
                panic!(
                    "WebGPU adapter is required for multi_cest_gpu_tests unless NMR_ALLOW_NO_GPU=1: {e}"
                );
            }
        }
    }
}

/// `[synthetic self-consistency]` Verify CPU `f64` vs WebGPU `f32` parity (`max |gpu - cpu| < 2e-4`)
/// across batched multi-`B1` `DANTE-CEST` profiles (including Gauss–Hermite `B1` inhomogeneity and
/// all 3 reference normalizations).
#[test]
fn test_webgpu_dante_cest_cpu_parity_under_2e4() {
    let Some(engine) = require_gpu_engine() else {
        return;
    };

    let larmor_mhz = 60.8106;
    let major_ppm = 118.0;
    let offsets: Vec<f64> = (-30..=30).map(|k| major_ppm + 0.25 * (k as f64)).collect();

    let tau_p = 15.0e-6;
    let tau_d = 1.985e-3; // sw_DANTE = 500 Hz (~8.22 ppm)
    let duty = tau_p / (tau_p + tau_d);

    let p_low_b1 = DanteCestParams {
        p_b: 0.025,
        k_ex_s_inv: 180.0,
        major_ppm,
        dw_ab_ppm: 4.2,
        r1a_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r1b_s_inv: 1.3,
        r2b_s_inv: 22.0,
        larmor_mhz,
        tau_pulse_s: tau_p,
        tau_delay_s: tau_d,
        n_pulses: 200, // T_EX = 0.40 s
        b1_pulse_hz: 15.0 / duty,
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::ZeroTimePhaseCycled,
    };
    let p_high_b1_inhom = DanteCestParams {
        b1_pulse_hz: 30.0 / duty,
        b1_inhomogeneity_frac: 0.08,
        n_b1_quad: 5,
        normalization: CestNormalization::SameDelayZeroB1,
        ..p_low_b1.clone()
    };
    let p_thermal = DanteCestParams {
        b1_pulse_hz: 20.0 / duty,
        b1_inhomogeneity_frac: 0.05,
        n_b1_quad: 5,
        normalization: CestNormalization::ThermalEquilibrium,
        ..p_low_b1.clone()
    };

    let cpu_0 = simulate_dante_cest_profile(&p_low_b1, &offsets).unwrap();
    let cpu_1 = simulate_dante_cest_profile(&p_high_b1_inhom, &offsets).unwrap();
    let cpu_2 = simulate_dante_cest_profile(&p_thermal, &offsets).unwrap();

    let gpu = simulate_dante_cest_batch_gpu(
        &engine,
        &[
            (p_low_b1, offsets.as_slice()),
            (p_high_b1_inhom, offsets.as_slice()),
            (p_thermal, offsets.as_slice()),
        ],
        5,
    )
    .expect("WebGPU DANTE-CEST batch");

    let mut max_diff = 0.0_f64;
    for (curve_i, (g_curve, c_curve)) in gpu.iter().zip([&cpu_0, &cpu_1, &cpu_2]).enumerate() {
        assert_eq!(g_curve.len(), c_curve.len());
        for (k, (&g, &c)) in g_curve.iter().zip(c_curve.iter()).enumerate() {
            assert!(
                g.is_finite(),
                "non-finite GPU DANTE output at curve {curve_i} offset {k}"
            );
            max_diff = max_diff.max((g - c).abs());
        }
    }
    println!(
        "[synthetic self-consistency] WebGPU DANTE-CEST parity (adapter='{}'): max |gpu - cpu| = {max_diff:.3e}",
        engine.adapter_name
    );
    assert!(
        max_diff < 2e-4,
        "WebGPU DANTE-CEST max |gpu - cpu| = {max_diff:.3e} >= 2e-4"
    );
}

/// `[synthetic self-consistency]` Verify CPU `f64` vs WebGPU `f32` parity (`max |gpu - cpu| < 2e-4`)
/// across batched multi-`B1` `cos-CEST` profiles (integer cycles, fractional remainder cycles,
/// odd & even Trotter steps, and Gauss–Hermite `B1` inhomogeneity).
#[test]
fn test_webgpu_cos_cest_cpu_parity_under_2e4() {
    let Some(engine) = require_gpu_engine() else {
        return;
    };

    let larmor_mhz = 60.8106;
    let major_ppm = 120.0;
    let offsets: Vec<f64> = (-28..=36).map(|k| major_ppm + 0.30 * (k as f64)).collect();

    let p_even_int = CosCestParams {
        p_b: 0.025,
        k_ex_s_inv: 180.0,
        major_ppm,
        dw_ab_ppm: 4.2,
        r1a_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r1b_s_inv: 1.3,
        r2b_s_inv: 22.0,
        larmor_mhz,
        t_ex_s: 0.40,
        b1_eff_hz: 15.0,
        mod_freq_hz: 300.0, // exact 120 cycles
        n_steps_per_cycle: 32,
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::ZeroTimePhaseCycled,
    };
    let p_frac_inhom = CosCestParams {
        t_ex_s: 0.3715, // fractional cycle remainder!
        b1_eff_hz: 28.0,
        mod_freq_hz: 250.0,
        n_steps_per_cycle: 24,
        b1_inhomogeneity_frac: 0.06,
        n_b1_quad: 5,
        normalization: CestNormalization::SameDelayZeroB1,
        ..p_even_int.clone()
    };
    let p_odd_steps = CosCestParams {
        t_ex_s: 0.30,
        b1_eff_hz: 22.0,
        mod_freq_hz: 200.0,
        n_steps_per_cycle: 25, // odd steps branch
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::ThermalEquilibrium,
        ..p_even_int.clone()
    };

    let cpu_0 = simulate_cos_cest_profile(&p_even_int, &offsets).unwrap();
    let cpu_1 = simulate_cos_cest_profile(&p_frac_inhom, &offsets).unwrap();
    let cpu_2 = simulate_cos_cest_profile(&p_odd_steps, &offsets).unwrap();

    let gpu = simulate_cos_cest_batch_gpu(
        &engine,
        &[
            (p_even_int, offsets.as_slice()),
            (p_frac_inhom, offsets.as_slice()),
            (p_odd_steps, offsets.as_slice()),
        ],
        5,
    )
    .expect("WebGPU cos-CEST batch");

    let mut max_diff = 0.0_f64;
    for (curve_i, (g_curve, c_curve)) in gpu.iter().zip([&cpu_0, &cpu_1, &cpu_2]).enumerate() {
        assert_eq!(g_curve.len(), c_curve.len());
        for (k, (&g, &c)) in g_curve.iter().zip(c_curve.iter()).enumerate() {
            assert!(
                g.is_finite(),
                "non-finite GPU cos-CEST output at curve {curve_i} offset {k}"
            );
            max_diff = max_diff.max((g - c).abs());
        }
    }
    println!(
        "[synthetic self-consistency] WebGPU cos-CEST parity (adapter='{}'): max |gpu - cpu| = {max_diff:.3e}",
        engine.adapter_name
    );
    assert!(
        max_diff < 2e-4,
        "WebGPU cos-CEST max |gpu - cpu| = {max_diff:.3e} >= 2e-4"
    );
}

/// `[synthetic self-consistency]` Verify mixed-scheme batching (`CW` + `DANTE` + `cos-CEST`),
/// baseline `wgpu::Limits::default()` compatibility, storage-limit guards, and hostile-input rejection.
#[test]
fn test_webgpu_multi_cest_default_limits_and_hostile_input_guards() {
    let engine = match GpuEngine::new_headless_with_limits(wgpu::Limits::default()) {
        Ok(eng) => eng,
        Err(e) => {
            if std::env::var("NMR_ALLOW_NO_GPU").as_deref() == Ok("1") {
                return;
            }
            panic!("WebGPU default-limits adapter required unless NMR_ALLOW_NO_GPU=1: {e}");
        }
    };

    let offsets: Vec<f64> = (-12..=12).map(|k| 118.0 + 0.5 * (k as f64)).collect();
    let base = MultiCestSimParams {
        p_b: 0.025,
        k_ex_s_inv: 180.0,
        major_ppm: 118.0,
        dw_ab_ppm: 4.2,
        r1a_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r1b_s_inv: 1.3,
        r2b_s_inv: 20.0,
        larmor_mhz: 60.8106,
        b1_inhomogeneity_frac: 0.0,
        n_b1_quad: 1,
        normalization: CestNormalization::ZeroTimePhaseCycled,
        scheme: MultiCestScheme::Cw {
            t_ex_s: 0.40,
            b1_hz: 20.0,
        },
    };
    let mut dante = base.clone();
    dante.scheme = MultiCestScheme::Dante {
        tau_pulse_s: 15.0e-6,
        tau_delay_s: 1.985e-3,
        n_pulses: 200,
        b1_pulse_hz: 20.0 / (15.0e-6 / 2.0e-3),
    };
    let mut cos_m = base.clone();
    cos_m.scheme = MultiCestScheme::CosModulated {
        t_ex_s: 0.40,
        b1_eff_hz: 20.0,
        mod_freq_hz: 250.0,
        n_steps_per_cycle: 32,
    };

    let cpu_cw = simulate_multi_cest_profile(&base, &offsets).unwrap();
    let cpu_dante = simulate_multi_cest_profile(&dante, &offsets).unwrap();
    let cpu_cos = simulate_multi_cest_profile(&cos_m, &offsets).unwrap();

    let gpu = simulate_multi_cest_batch_gpu(
        &engine,
        &[
            (base.clone(), offsets.as_slice()),
            (dante, offsets.as_slice()),
            (cos_m, offsets.as_slice()),
        ],
        1,
    )
    .expect("mixed-scheme default-limits WebGPU batch");

    let mut max_diff = 0.0_f64;
    for (g_curve, c_curve) in gpu.iter().zip([&cpu_cw, &cpu_dante, &cpu_cos]) {
        for (&g, &c) in g_curve.iter().zip(c_curve.iter()) {
            max_diff = max_diff.max((g - c).abs());
        }
    }
    assert!(
        max_diff < 2e-4,
        "mixed-scheme default-limits parity failed: {max_diff:.3e}"
    );

    // Hostile input rejection on GPU wrapper
    let mut bad_p = base.clone();
    bad_p.p_b = f64::NAN;
    assert!(simulate_multi_cest_batch_gpu(&engine, &[(bad_p, offsets.as_slice())], 1).is_err());

    assert!(
        simulate_multi_cest_batch_gpu(&engine, &[(base.clone(), &[f64::INFINITY])], 1).is_err()
    );

    // 22.6:P1-2 hostile GPU precheck regressions: must fail closed BEFORE GPU dispatch (< 5 ms)
    let mut hostile_steps = base.clone();
    hostile_steps.scheme = MultiCestScheme::CosModulated {
        t_ex_s: 0.40,
        b1_eff_hz: 20.0,
        mod_freq_hz: 250.0,
        n_steps_per_cycle: u32::MAX,
    };
    let t0 = Instant::now();
    assert!(
        simulate_multi_cest_batch_gpu(&engine, &[(hostile_steps, offsets.as_slice())], 1).is_err()
    );
    assert!(
        t0.elapsed().as_millis() < 5,
        "GPU hostile n_steps_per_cycle=u32::MAX took {:?}, expected < 5 ms",
        t0.elapsed()
    );

    let mut hostile_freq = base.clone();
    hostile_freq.scheme = MultiCestScheme::CosModulated {
        t_ex_s: 0.40,
        b1_eff_hz: 20.0,
        mod_freq_hz: 1e10,
        n_steps_per_cycle: 32,
    };
    let t1 = Instant::now();
    assert!(
        simulate_multi_cest_batch_gpu(&engine, &[(hostile_freq, offsets.as_slice())], 1).is_err()
    );
    assert!(
        t1.elapsed().as_millis() < 5,
        "GPU hostile mod_freq_hz=1e10 took {:?}, expected < 5 ms",
        t1.elapsed()
    );

    let mut over_dante = base.clone();
    over_dante.scheme = MultiCestScheme::Dante {
        tau_pulse_s: 1e-6,
        tau_delay_s: 1e-6,
        n_pulses: MAX_DANTE_PULSES + 1,
        b1_pulse_hz: 500.0,
    };
    assert!(
        simulate_multi_cest_batch_gpu(&engine, &[(over_dante, offsets.as_slice())], 1).is_err()
    );

    let mut over_mod = base.clone();
    over_mod.scheme = MultiCestScheme::CosModulated {
        t_ex_s: 0.40,
        b1_eff_hz: 20.0,
        mod_freq_hz: MAX_COS_CEST_MOD_FREQ_HZ + 1.0,
        n_steps_per_cycle: 32,
    };
    assert!(simulate_multi_cest_batch_gpu(&engine, &[(over_mod, offsets.as_slice())], 1).is_err());

    // Total batch work budget precheck (> MAX_TOTAL_PROPAGATOR_STEPS)
    let mut heavy_cos = base;
    heavy_cos.b1_inhomogeneity_frac = 0.05;
    heavy_cos.n_b1_quad = 9;
    heavy_cos.scheme = MultiCestScheme::CosModulated {
        t_ex_s: 0.401,
        b1_eff_hz: 20.0,
        mod_freq_hz: 250.0,
        n_steps_per_cycle: MAX_COS_CEST_STEPS_PER_CYCLE,
    };
    let big_offsets = vec![118.0; 25_000];
    let t2 = Instant::now();
    assert!(
        simulate_multi_cest_batch_gpu(&engine, &[(heavy_cos, big_offsets.as_slice())], 9).is_err()
    );
    assert!(
        t2.elapsed().as_millis() < 5,
        "GPU batch work-budget rejection took {:?}, expected < 5 ms",
        t2.elapsed()
    );

    // Storage limit guard rejects absurd buffer sizes cleanly
    let huge = engine.device.limits().max_buffer_size.saturating_add(4096);
    assert!(validate_multi_cest_storage_bytes(&engine.device, "hostile_test", huge).is_err());
}
