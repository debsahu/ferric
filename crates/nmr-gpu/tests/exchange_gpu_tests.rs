//! Quantitative WebGPU verification suite for `nmr-gpu::cest_gpu` (`T13.4`) and
//! `nmr-gpu::cpmg_gpu` (`T13.5`):
//! - WebGPU `7x7` and `10x10` Higham `[6/6]` Padé matrix exponential agreement `< 1e-5` against
//!   `f64` complex eigendecomposition and 24th-order Taylor references (including `DEST`
//!   `R2B = 18,000 s^-1`).
//! - Dual-`B1` (`15 Hz` & `30 Hz`) synthetic `15N` `CEST` global WebGPU fit recovering
//!   `p_B = 1.50% ± 0.05%`, `k_ex = 150 ± 5 s^-1`, and `Delta_varpi_AB` within `< 0.05 ppm` in
//!   `< 1.5 s`.
//! - Multi-field (`600 + 800 MHz`) global `CPMG` WebGPU fit (`Baldwin2014`) recovering
//!   `(p_B, k_ex, |Delta_omega|)` within `< 2%` error in `< 0.5 s`.
//! - Baseline `wgpu::Limits::default()` portability and `max_buffer_size` /
//!   `max_storage_buffer_binding_size` error-propagation guards.
//!
//! Provenance: Clean-room GPU verification citing McConnell (*J. Chem. Phys.* 1958),
//! Vallurupalli, Bouvignies & Kay (*JACS* 2012), Fawzi et al. (*Nature* 2011), Higham
//! (*SIAM J. Matrix Anal. Appl.* 2005), Carver & Richards (*JMR* 1972), Luz & Meiboom
//! (*J. Chem. Phys.* 1963), Baldwin (*JMR* 2014), and Trott & Palmer (*JMR* 2002).

use std::f64::consts::TAU;
use std::time::Instant;

use nmr_core::cest::{
    build_liouvillian_2state, build_liouvillian_3state, expm_eigen_reference,
    expm_high_order_reference, simulate_cest_2state_profile, simulate_cest_3state_profile,
    Cest2StateParams, Cest3StateParams, CestFitConfig, CestNormalization, CestProfileDataset,
};
use nmr_core::cpmg::{
    r1rho_trott_palmer_2002, r2eff_baldwin2014, r2eff_cr72, r2eff_flat, r2eff_lm63, CpmgFitConfig,
    CpmgModel, CpmgSeriesDataset,
};
use nmr_core::rng::Rng;
use nmr_gpu::cest_gpu::{
    expm_pade66_10_gpu, expm_pade66_7_gpu, fit_cest_global_gpu, simulate_cest_2state_batch_gpu,
    simulate_cest_3state_batch_gpu, validate_cest_storage_bytes,
};
use nmr_gpu::cpmg_gpu::{
    evaluate_cpmg_batch_gpu, fit_cpmg_global_gpu, profile_cpmg_grid_with_pipeline_async,
    validate_cpmg_storage_bytes, CpmgGpuEvalRequest, CpmgGpuPipeline,
};
use nmr_gpu::GpuEngine;

fn is_software_cpu_adapter_probe() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    if let Ok(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
    {
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return true;
        }
        let lower = info.name.to_ascii_lowercase();
        lower.contains("llvmpipe")
            || lower.contains("lavapipe")
            || lower.contains("basic render")
            || lower.contains("swiftshader")
    } else {
        false
    }
}

fn require_gpu_engine() -> Option<(GpuEngine, bool)> {
    match GpuEngine::new_headless() {
        Ok(eng) => {
            let lower = eng.adapter_name.to_ascii_lowercase();
            let is_sw_cpu = is_software_cpu_adapter_probe()
                || lower.contains("llvmpipe")
                || lower.contains("lavapipe")
                || lower.contains("basic render")
                || lower.contains("swiftshader");
            Some((eng, is_sw_cpu))
        }
        Err(e) => {
            if std::env::var("NMR_ALLOW_NO_GPU").as_deref() == Ok("1") {
                eprintln!("Skipping GPU exchange test because NMR_ALLOW_NO_GPU=1 ({e})");
                None
            } else {
                panic!(
                    "WebGPU adapter is required for exchange_gpu_tests unless NMR_ALLOW_NO_GPU=1: {e}"
                );
            }
        }
    }
}

#[test]
fn test_webgpu_pade66_expm_and_2state_3state_cest_dest_parity_under_1e5() {
    let Some((engine, _is_sw_cpu)) = require_gpu_engine() else {
        return;
    };

    // 1. 7x7 Liouvillians spanning weak/strong B1 CEST and extreme R2B = 18,000 s^-1 DEST
    let mut mats7 = Vec::new();
    for &b1_hz in &[5.0, 15.0, 30.0, 250.0] {
        for &r2b in &[18.0, 180.0, 18_000.0] {
            for &off_hz in &[-800.0, -90.0, 0.0, 240.0] {
                let mut lt = build_liouvillian_2state(
                    0.02,
                    180.0,
                    TAU * off_hz,
                    TAU * (off_hz + 260.0),
                    TAU * b1_hz,
                    0.0,
                    1.3,
                    14.0,
                    1.3,
                    r2b,
                );
                for row in &mut lt {
                    for v in row {
                        *v *= 0.40;
                    }
                }
                mats7.push(lt);
            }
        }
    }

    let gpu7 = expm_pade66_7_gpu(&engine, &mats7).expect("WebGPU 7x7 Pade [6/6]");
    let mut max_err7 = 0.0f64;
    for (lt, g_mat) in mats7.iter().zip(gpu7.iter()) {
        let ref_mat = expm_eigen_reference(lt).unwrap_or_else(|_| expm_high_order_reference(lt));
        for i in 0..7 {
            for j in 0..7 {
                max_err7 = max_err7.max((g_mat[i][j] - ref_mat[i][j]).abs());
            }
        }
    }
    assert!(
        max_err7 < 1e-5,
        "WebGPU f32 7x7 Pade [6/6] max diff vs f64 eigendecomp {max_err7:.3e} >= 1e-5"
    );

    // 2. 10x10 3-state linear & triangular Liouvillians
    let mut mats10 = Vec::new();
    for &(k_ab, k_bc, k_ac) in &[(160.0, 110.0, 0.0), (190.0, 140.0, 85.0)] {
        for &b1_hz in &[12.0, 25.0] {
            for &off_hz in &[-320.0, 0.0, 195.0, 410.0] {
                let mut lt = build_liouvillian_3state(
                    0.018,
                    0.012,
                    k_ab,
                    k_bc,
                    k_ac,
                    TAU * off_hz,
                    TAU * (off_hz + 220.0),
                    TAU * (off_hz - 190.0),
                    TAU * b1_hz,
                    0.0,
                    1.2,
                    13.5,
                    1.2,
                    21.0,
                    1.2,
                    26.0,
                );
                for row in &mut lt {
                    for v in row {
                        *v *= 0.35;
                    }
                }
                mats10.push(lt);
            }
        }
    }

    let gpu10 = expm_pade66_10_gpu(&engine, &mats10).expect("WebGPU 10x10 Pade [6/6]");
    let mut max_err10 = 0.0f64;
    for (lt, g_mat) in mats10.iter().zip(gpu10.iter()) {
        let ref_mat = expm_eigen_reference(lt).unwrap_or_else(|_| expm_high_order_reference(lt));
        for i in 0..10 {
            for j in 0..10 {
                max_err10 = max_err10.max((g_mat[i][j] - ref_mat[i][j]).abs());
            }
        }
    }
    println!(
        "[T13.4 WebGPU Pade66] adapter='{}', max_err_7x7={max_err7:.3e}, max_err_10x10={max_err10:.3e}",
        engine.adapter_name
    );
    assert!(
        max_err10 < 1e-5,
        "WebGPU f32 10x10 Pade [6/6] max diff vs f64 eigendecomp {max_err10:.3e} >= 1e-5"
    );

    // 3. Full 2-state (CEST + DEST + Gauss-Hermite B1 quadrature) & 3-state profile parity `< 1e-5`
    let offsets: Vec<f64> = (-30..=30).map(|k| 118.0 + 0.25 * (k as f64)).collect();
    let p2_cest = Cest2StateParams {
        p_b: 0.018,
        k_ex_s_inv: 160.0,
        major_ppm: 118.0,
        dw_ab_ppm: 4.2,
        r1a_s_inv: 1.3,
        r2a_s_inv: 14.0,
        r1b_s_inv: 1.3,
        r2b_s_inv: 22.0,
        larmor_mhz: 60.8106,
        t_ex_s: 0.40,
        b1_hz: 20.0,
        b1_inhomogeneity_frac: 0.05,
        n_b1_quad: 5,
        normalization: CestNormalization::SameDelayZeroB1,
    };
    let p2_dest = Cest2StateParams {
        r2b_s_inv: 15_000.0,
        b1_hz: 350.0,
        t_ex_s: 0.60,
        dw_ab_ppm: 0.0,
        b1_inhomogeneity_frac: 0.0,
        ..p2_cest.clone()
    };
    let cpu_cest = simulate_cest_2state_profile(&p2_cest, &offsets).unwrap();
    let cpu_dest = simulate_cest_2state_profile(&p2_dest, &offsets).unwrap();
    let gpu_2s = simulate_cest_2state_batch_gpu(
        &engine,
        &[(p2_cest, offsets.as_slice()), (p2_dest, offsets.as_slice())],
        5,
    )
    .expect("WebGPU 2-state CEST+DEST batch");
    for (g, c) in gpu_2s[0].iter().zip(cpu_cest.iter()) {
        assert!(
            (g - c).abs() < 1e-5,
            "2-state CEST GPU vs CPU diff {}",
            (g - c).abs()
        );
    }
    for (g, c) in gpu_2s[1].iter().zip(cpu_dest.iter()) {
        assert!(
            (g - c).abs() < 1e-5,
            "2-state DEST GPU vs CPU diff {}",
            (g - c).abs()
        );
    }

    let p3 = Cest3StateParams {
        p_b: 0.02,
        p_c: 0.012,
        k_ex_ab_s_inv: 150.0,
        k_ex_bc_s_inv: 110.0,
        k_ex_ac_s_inv: 70.0,
        major_ppm: 118.0,
        dw_ab_ppm: 3.8,
        dw_ac_ppm: -3.2,
        r1a_s_inv: 1.25,
        r2a_s_inv: 13.5,
        r1b_s_inv: 1.25,
        r2b_s_inv: 19.0,
        r1c_s_inv: 1.25,
        r2c_s_inv: 24.0,
        larmor_mhz: 60.8106,
        t_ex_s: 0.40,
        b1_hz: 18.0,
        b1_inhomogeneity_frac: 0.05,
        n_b1_quad: 5,
        normalization: CestNormalization::ThermalEquilibrium,
    };
    let cpu_3s = simulate_cest_3state_profile(&p3, &offsets).unwrap();
    let gpu_3s = simulate_cest_3state_batch_gpu(&engine, &[(p3, offsets.as_slice())], 5).unwrap();
    for (g, c) in gpu_3s[0].iter().zip(cpu_3s.iter()) {
        assert!(
            (g - c).abs() < 1e-5,
            "3-state CEST GPU vs CPU diff {}",
            (g - c).abs()
        );
    }
}

#[test]
fn test_webgpu_dual_b1_15hz_30hz_global_cest_fit_done_gate_under_1p5s() {
    let Some((engine, is_sw_cpu)) = require_gpu_engine() else {
        return;
    };

    let true_pb = 0.0150; // 1.50%
    let true_kex = 150.0; // 150 s^-1
    let larmor_15n_mhz = 60.8106;
    let t_ex_s = 0.40;
    let b1_fields = [15.0, 30.0];
    let noise_sd = 0.0020;

    let residues = [
        (12, "VAL", 119.5, 4.40, 1.35, 13.5, 19.0),
        (24, "ILE", 121.2, -3.75, 1.25, 14.5, 21.0),
        (38, "ASP", 116.8, 5.10, 1.45, 12.8, 17.5),
        (51, "LEU", 123.0, -4.60, 1.30, 15.0, 22.0),
        (64, "LYS", 118.1, 3.25, 1.40, 13.0, 18.5),
    ];

    let mut rng = Rng::new(0xCE57_600D_2026);
    let mut profiles = Vec::new();
    for &(seq_id, res_name, major_ppm, dw_ppm, r1a, r2a, r2b) in &residues {
        let offsets: Vec<f64> = (-32..=32).map(|k| major_ppm + 0.25 * (k as f64)).collect();
        for &b1_hz in &b1_fields {
            let p = Cest2StateParams {
                p_b: true_pb,
                k_ex_s_inv: true_kex,
                major_ppm,
                dw_ab_ppm: dw_ppm,
                r1a_s_inv: r1a,
                r2a_s_inv: r2a,
                r1b_s_inv: r1a,
                r2b_s_inv: r2b,
                larmor_mhz: larmor_15n_mhz,
                t_ex_s,
                b1_hz,
                b1_inhomogeneity_frac: 0.0,
                n_b1_quad: 1,
                normalization: CestNormalization::ThermalEquilibrium,
            };
            let clean = simulate_cest_2state_profile(&p, &offsets).unwrap();
            let noisy: Vec<f64> = clean
                .iter()
                .map(|&y| y + rng.normal(0.0, noise_sd))
                .collect();
            profiles.push(CestProfileDataset {
                seq_id,
                res_name: res_name.to_string(),
                larmor_mhz: larmor_15n_mhz,
                t_ex_s,
                b1_hz,
                b1_inhomogeneity_frac: 0.0,
                normalization: CestNormalization::ThermalEquilibrium,
                major_ppm_hint: Some(major_ppm),
                offsets_ppm: offsets.clone(),
                intensities: noisy,
                uncertainties: vec![noise_sd; offsets.len()],
            });
        }
    }

    let cfg = CestFitConfig {
        n_b1_quad: 1,
        compute_chi2_surface: false,
        ..CestFitConfig::default()
    };

    let max_cest_secs = if is_sw_cpu { 30.0 } else { 1.5 };

    let t0 = Instant::now();
    let fit = fit_cest_global_gpu(&engine, &profiles, &cfg).expect("WebGPU global CEST fit");
    let elapsed_s = t0.elapsed().as_secs_f64();

    assert!(
        elapsed_s < max_cest_secs,
        "WebGPU dual-B1 global CEST fit took {elapsed_s:.3} s (must be < {max_cest_secs} s on {})",
        engine.adapter_name
    );
    assert!(
        (fit.p_b - true_pb).abs() <= 0.0005,
        "p_B = {:.5} outside 1.50% ± 0.05%",
        fit.p_b
    );
    assert!(
        (fit.k_ex_s_inv - true_kex).abs() <= 5.0,
        "k_ex = {:.2} outside 150 ± 5 s^-1",
        fit.k_ex_s_inv
    );
    let mut max_dw_err = 0.0f64;
    for (rf, &(_, _, _, true_dw, _, _, _)) in fit.residues.iter().zip(residues.iter()) {
        let err = (rf.dw_ab_ppm - true_dw).abs();
        max_dw_err = max_dw_err.max(err);
        assert!(
            err < 0.05,
            "residue {} dw_AB error {err:.4} ppm >= 0.05 ppm (fit={:.4}, true={true_dw})",
            rf.seq_id,
            rf.dw_ab_ppm
        );
    }
    println!(
        "[T13.4 WebGPU CEST Fit] p_B={:.4}% (true=1.50%), k_ex={:.2} s^-1 (true=150), max|ddw|={max_dw_err:.4} ppm, time={elapsed_s:.3} s",
        fit.p_b * 100.0,
        fit.k_ex_s_inv
    );
}

#[test]
fn test_webgpu_multifield_600_800_cpmg_global_fit_under_2pct_and_0p5s() {
    let Some((engine, is_sw_cpu)) = require_gpu_engine() else {
        return;
    };

    // 1. Forward batch parity across Flat, LM63, CR72, Baldwin2014, and TrottPalmerR1Rho
    let reqs = vec![
        CpmgGpuEvalRequest::Flat { r20_s_inv: 13.75 },
        CpmgGpuEvalRequest::Lm63 {
            r20_s_inv: 12.5,
            p_b: 0.025,
            k_ex_s_inv: 1800.0,
            dw_rad_s: 950.0,
            nu_cpmg_hz: 200.0,
        },
        CpmgGpuEvalRequest::Cr72 {
            r2a_s_inv: 13.0,
            r2b_s_inv: 19.0,
            p_b: 0.03,
            k_ex_s_inv: 850.0,
            dw_rad_s: 1100.0,
            nu_cpmg_hz: 150.0,
        },
        CpmgGpuEvalRequest::Baldwin2014 {
            r2a_s_inv: 14.0,
            r2b_s_inv: 32.0,
            p_b: 0.04,
            k_ex_s_inv: 650.0,
            dw_rad_s: 1350.0,
            t_relax_s: 0.040,
            nu_cpmg_hz: 100.0,
        },
        CpmgGpuEvalRequest::TrottPalmerR1Rho {
            r1_s_inv: 1.4,
            r20_s_inv: 13.5,
            p_b: 0.025,
            k_ex_s_inv: 1200.0,
            omega_a_offset_rad_s: -900.0,
            dw_ab_rad_s: -1600.0,
            w1_rad_s: TAU * 350.0,
        },
    ];
    let gpu_vals = evaluate_cpmg_batch_gpu(&engine, &reqs).expect("WebGPU CPMG batch eval");
    let cpu_refs = [
        r2eff_flat(13.75),
        r2eff_lm63(12.5, 0.975 * 0.025 * 950.0 * 950.0, 1800.0, 200.0),
        r2eff_cr72(13.0, 19.0, 0.03, 850.0, 1100.0, 150.0),
        r2eff_baldwin2014(14.0, 32.0, 0.04, 650.0, 1350.0, 0.040, 100.0),
        r1rho_trott_palmer_2002(1.4, 13.5, 0.025, 1200.0, -900.0, -1600.0, TAU * 350.0).2,
    ];
    for (i, (&g, &c)) in gpu_vals.iter().zip(cpu_refs.iter()).enumerate() {
        assert!(
            (g - c).abs() < 2e-3,
            "CPMG model {i} GPU vs CPU diff {:.3e} (gpu={g}, cpu={c})",
            (g - c).abs()
        );
    }

    // 2. Multi-field (`600 + 800 MHz` 1H -> `60.8106 + 81.0808 MHz` 15N) global CPMG fit
    let true_pb = 0.025; // 2.5%
    let true_kex = 900.0; // 900 s^-1
    let fields_15n = [60.8106, 81.0808];
    let t_relax_s = 0.040;
    let nu_grid = vec![
        25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 550.0, 700.0, 850.0, 1000.0,
    ];
    let exchanging = [
        (15, "ALA", 2.80, [11.5, 13.8]),
        (28, "VAL", 3.60, [12.8, 15.4]),
        (41, "SER", 2.15, [10.4, 12.5]),
        (56, "THR", 4.30, [13.9, 16.7]),
        (72, "GLU", 3.15, [12.0, 14.4]),
        (85, "PHE", 2.55, [13.1, 15.7]),
    ];

    let mut rng = Rng::new(0xC946_600D_2026);
    let err_s_inv = 0.08;
    let mut series = Vec::new();
    for &(seq_id, res_name, dw_ppm, r20_pair) in &exchanging {
        for (fi, &larmor) in fields_15n.iter().enumerate() {
            let dw_rad = TAU * larmor * dw_ppm;
            let r2_vals: Vec<f64> = nu_grid
                .iter()
                .map(|&nu| {
                    r2eff_baldwin2014(
                        r20_pair[fi],
                        r20_pair[fi],
                        true_pb,
                        true_kex,
                        dw_rad,
                        t_relax_s,
                        nu,
                    ) + rng.normal(0.0, err_s_inv)
                })
                .collect();
            series.push(CpmgSeriesDataset {
                seq_id,
                res_name: res_name.to_string(),
                larmor_mhz: larmor,
                t_relax_s,
                nu_cpmg_hz: nu_grid.clone(),
                r2_eff_s_inv: r2_vals,
                r2_eff_err: vec![err_s_inv; nu_grid.len()],
            });
        }
    }
    // Non-exchanging decoy residue 99 (`dw = 0`)
    for &larmor in &fields_15n {
        series.push(CpmgSeriesDataset {
            seq_id: 99,
            res_name: "GLY".to_string(),
            larmor_mhz: larmor,
            t_relax_s,
            nu_cpmg_hz: nu_grid.clone(),
            r2_eff_s_inv: nu_grid
                .iter()
                .map(|_| 11.2 + rng.normal(0.0, err_s_inv))
                .collect(),
            r2_eff_err: vec![err_s_inv; nu_grid.len()],
        });
    }

    let max_cpmg_secs = if is_sw_cpu { 20.0 } else { 0.5 };

    let t0 = Instant::now();
    let fit = fit_cpmg_global_gpu(&engine, &series, &CpmgFitConfig::default())
        .expect("WebGPU multi-field global CPMG fit");
    let elapsed_s = t0.elapsed().as_secs_f64();

    assert!(
        elapsed_s < max_cpmg_secs,
        "WebGPU multi-field global CPMG fit took {elapsed_s:.3} s (must be < {max_cpmg_secs} s on {})",
        engine.adapter_name
    );
    assert_eq!(
        fit.residues.len(),
        exchanging.len(),
        "decoy residue 99 must be gated out"
    );

    let rel_pb = (fit.p_b - true_pb).abs() / true_pb;
    let rel_kex = (fit.k_ex_s_inv - true_kex).abs() / true_kex;
    assert!(
        rel_pb < 0.02,
        "WebGPU p_B rel error {rel_pb:.4} >= 2% (fit={}, true={true_pb})",
        fit.p_b
    );
    assert!(
        rel_kex < 0.02,
        "WebGPU k_ex rel error {rel_kex:.4} >= 2% (fit={}, true={true_kex})",
        fit.k_ex_s_inv
    );
    let mut max_rel_dw = 0.0f64;
    for (rf, &(_, _, true_dw, _)) in fit.residues.iter().zip(exchanging.iter()) {
        let rel_dw = (rf.dw_ppm - true_dw).abs() / true_dw;
        max_rel_dw = max_rel_dw.max(rel_dw);
        assert!(
            rel_dw < 0.02,
            "WebGPU res {} |dw| rel error {rel_dw:.4} >= 2% (fit={}, true={true_dw})",
            rf.seq_id,
            rf.dw_ppm
        );
    }
    println!(
        "[T13.5 WebGPU CPMG Fit] p_B={:.4}% (err={:.2}%), k_ex={:.2} s^-1 (err={:.2}%), max_dw_err={:.2}%, time={elapsed_s:.3} s",
        fit.p_b * 100.0,
        rel_pb * 100.0,
        fit.k_ex_s_inv,
        rel_kex * 100.0,
        max_rel_dw * 100.0
    );
}

#[test]
fn test_baseline_webgpu_limits_and_oversized_buffer_guards() {
    if std::env::var("NMR_ALLOW_NO_GPU").as_deref() == Ok("1") && GpuEngine::new_headless().is_err()
    {
        return;
    }
    let engine = GpuEngine::new_headless_with_limits(wgpu::Limits::default())
        .expect("baseline wgpu::Limits::default() engine");

    let lt = build_liouvillian_2state(0.02, 200.0, 120.0, 900.0, 95.0, 0.0, 1.3, 14.0, 1.3, 20.0);
    let res = expm_pade66_7_gpu(&engine, &[lt]).expect("expm7 on baseline WebGPU limits");
    assert!((res[0][0][0] - 1.0).abs() < 1e-6);

    let over = engine
        .device
        .limits()
        .max_storage_buffer_binding_size
        .max(engine.device.limits().max_buffer_size)
        + 1024;
    assert!(validate_cest_storage_bytes(&engine.device, "test_over", over).is_err());
    assert!(validate_cpmg_storage_bytes(&engine.device, "test_over", over).is_err());

    // Verify flat `job_offset` TDR chunking when a single `p_B` row has `n_kex * n_res = 40,000`
    // (`> MAX_CPMG_JOBS_PER_DISPATCH = 32,768`).
    let single_ds: Vec<CpmgSeriesDataset> = (0..200_i32)
        .map(|ri| CpmgSeriesDataset {
            seq_id: ri + 1,
            res_name: "ALA".to_string(),
            larmor_mhz: 60.8106,
            t_relax_s: 0.040,
            nu_cpmg_hz: vec![50.0, 200.0, 800.0],
            r2_eff_s_inv: vec![16.5, 13.8, 12.6],
            r2_eff_err: vec![0.10, 0.10, 0.10],
        })
        .collect();
    let res_groups: Vec<Vec<&CpmgSeriesDataset>> = single_ds.iter().map(|s| vec![s]).collect();
    let pb_row = vec![0.025];
    let kex_row: Vec<f64> = (0..200).map(|i| 100.0 + 15.0 * (i as f64)).collect();
    let pipe = CpmgGpuPipeline::new(&engine.device);
    let cells = pollster::block_on(profile_cpmg_grid_with_pipeline_async(
        &engine,
        &pipe,
        &res_groups,
        &pb_row,
        &kex_row,
        CpmgModel::Baldwin2014,
    ))
    .expect("single-pB-row >32,768 flat job chunking must succeed without overflow");
    assert_eq!(cells.len(), 40_000);
    assert!(cells
        .iter()
        .all(|c| c.best_chi2.is_finite() && c.best_dw_ppm > 0.0));
}
