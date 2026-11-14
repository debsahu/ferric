//! Integration and verification suite for `nmr-core::relax_io` (`T13.1`):
//! - Bruker (`vdlist`, `vclist`, `fqlist`) and Varian (`procpar` `relaxT`, `d2`, `ncyc`, `satfrq`) schedule parsers
//! - Plane-queued (`[plane][t1][quad][t2]`) and $t_1$-interleaved (`[t1][plane][quad][t2]` & `[t1][quad][plane][t2]`)
//!   raw Bruker `ser` and Varian binary `fid` pseudo-3D decoders
//! - Uniform locked-phase `ProcessingRecipe` 2D plane processing
//! - Quantitative Done Gate (`T13.1`): 20-plane synthetic Bruker & Varian pseudo-3D decay series down to `SNR = 4`
//!   achieving `< 1.5%` systematic locked-footprint amplitude bias while demonstrating `> 8%` positive bias
//!   from naive local-maximum picking at `SNR = 4`
//! - Robust 2D MAD noise + duplicate-delay replicate variance pooling + overlapping-peak linear deconvolution
//! - Lossless ASCII TSV and NEF v1.1 relaxation-series round-trips
//!
//! Provenance: Clean-room verification citing Farrow et al. (*Biochemistry* 1994 & *J. Biomol. NMR* 1995),
//! Kay, Torchia & Bax (*Biochemistry* 1989), and Korzhnev et al. (*Prog. Nucl. Magn. Reson. Spectrosc.* 2001).

use std::f64::consts::PI;
use std::fs;
use std::path::Path;

use nmr_core::process::{ApodizationWindow, BaselineConfig, PhaseSource};
use nmr_core::relax_io::{
    cpmg_frequencies_from_vclist, estimate_plane_noise_mad_2d, extract_naive_local_max_series,
    footprint_eval_2d, integrate_relax_planes, parse_bruker_fqlist, parse_bruker_vclist,
    parse_bruker_vdlist, parse_relax_series_nef_str, parse_relax_series_tsv_str,
    parse_varian_relax_schedule, process_pseudo3d_fids, read_bruker_pseudo3d, read_varian_pseudo3d,
    write_relax_series_nef_str, write_relax_series_tsv_str, FootprintShapeModel,
    Pseudo3DInterleaving, RelaxExperimentKind, RelaxIntegrationConfig, RelaxSchedule,
    TargetPeakSeed,
};
use nmr_core::rng::Rng;
use nmr_core::spectrum::{AxisPpm, Spectrum2D};
use serde_json::json;

fn emit_w3_metrics(record_id: &str, payload: serde_json::Value) {
    let line = serde_json::to_string(&payload).expect("serialize W3 metrics JSON");
    println!("[W3_METRICS_JSON] {line}");
    if let Ok(dir_str) = std::env::var("NMR_W3_METRICS_DIR") {
        if !dir_str.trim().is_empty() {
            let dir = Path::new(&dir_str);
            fs::create_dir_all(dir).expect("create NMR_W3_METRICS_DIR");
            let tmp_path = dir.join(format!("{record_id}.{}.tmp", std::process::id()));
            let out_path = dir.join(format!("{record_id}.json"));
            let pretty =
                serde_json::to_string_pretty(&payload).expect("pretty-serialize W3 metrics JSON");
            fs::write(&tmp_path, format!("{pretty}\n")).expect("write temp W3 metrics JSON");
            fs::rename(&tmp_path, &out_path).expect("commit W3 metrics JSON");
        }
    }
}

fn temp_test_dir(tag: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("nmr_relax_io_test_{}_{}", tag, std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create temp test dir");
    dir
}

#[test]
fn schedule_parsers_decode_bruker_vd_vc_fq_and_varian_procpar() {
    let vd_txt = "\
# Bruker T1 delay list with mixed suffixes
10m
50ms ; duplicate check later
500u
250us
0.100
0.5s
1.25s
50ms
";
    let vd = parse_bruker_vdlist(vd_txt).expect("parse vdlist");
    assert_eq!(vd.len(), 8);
    assert!((vd[0] - 0.010).abs() < 1.0e-12);
    assert!((vd[1] - 0.050).abs() < 1.0e-12);
    assert!((vd[2] - 5.0e-4).abs() < 1.0e-12);
    assert!((vd[3] - 2.5e-4).abs() < 1.0e-12);
    assert!((vd[4] - 0.100).abs() < 1.0e-12);
    assert!((vd[5] - 0.500).abs() < 1.0e-12);
    assert!((vd[6] - 1.250).abs() < 1.0e-12);
    assert!((vd[7] - 0.050).abs() < 1.0e-12);

    let vc_txt = "0\n2\n4\n8\n16\n4\n";
    let (counts, secs) = parse_bruker_vclist(vc_txt, 0.016).expect("parse vclist");
    assert_eq!(counts, vec![0, 2, 4, 8, 16, 4]);
    assert!((secs[2] - 0.064).abs() < 1.0e-12);
    let nu_cpmg = cpmg_frequencies_from_vclist(&counts, 0.040, 2).expect("cpmg frequencies");
    assert!((nu_cpmg[0] - 0.0).abs() < 1.0e-12);
    assert!((nu_cpmg[1] - 50.0).abs() < 1.0e-12);
    assert!((nu_cpmg[4] - 400.0).abs() < 1.0e-12);

    let fq_txt = "\
sfo hz
-600.0
0.0
+300Hz
+0.6kHz
+0.0015MHz
bf ppm
105.0
120.0ppm
";
    let fq = parse_bruker_fqlist(fq_txt, 60.81, 118.0).expect("parse fqlist");
    assert_eq!(fq.len(), 7);
    assert!((fq[0].offset_hz - (-600.0)).abs() < 1.0e-9);
    assert!((fq[0].offset_ppm - (118.0 - 600.0 / 60.81)).abs() < 1.0e-9);
    assert!((fq[3].offset_hz - 600.0).abs() < 1.0e-9);
    assert!((fq[4].offset_hz - 1500.0).abs() < 1.0e-9);
    assert!((fq[5].offset_ppm - 105.0).abs() < 1.0e-9);
    assert!((fq[5].offset_hz - (105.0 - 118.0) * 60.81).abs() < 1.0e-9);
    assert!((fq[6].offset_ppm - 120.0).abs() < 1.0e-9);

    let procpar_interleaved = "\
array 2 2 8192 0 0 2 1 0 1 64
1 \"relaxT,phase\"
0
pslabel 2 2 8192 0 0 2 1 0 1 64
1 \"gNhsqc_T2\"
0
phase 1 1 8 0 0 2 1 0 1 64
2 1 2
0
relaxT 1 1 8 0 0 2 1 0 1 64
4 0.016 0.032 0.064 0.128
0
";
    let v_sched =
        parse_varian_relax_schedule(procpar_interleaved, None).expect("parse varian procpar");
    assert_eq!(v_sched.param_name, "relaxT");
    assert_eq!(v_sched.experiment_kind, RelaxExperimentKind::R2);
    assert_eq!(v_sched.interleaving, Pseudo3DInterleaving::T1PlaneQuad);
    assert_eq!(
        v_sched.schedule.primary_values(),
        vec![0.016, 0.032, 0.064, 0.128]
    );

    let procpar_quad_outer = procpar_interleaved.replace("relaxT,phase", "phase,relaxT");
    let v_sched_2 =
        parse_varian_relax_schedule(&procpar_quad_outer, None).expect("parse varian phase,relaxT");
    assert_eq!(v_sched_2.interleaving, Pseudo3DInterleaving::T1QuadPlane);

    let vd_expected = [0.010, 0.050, 5.0e-4, 2.5e-4, 0.100, 0.500, 1.250, 0.050];
    let vd_max_err = vd
        .iter()
        .zip(vd_expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let fq_expected_hz = [
        -600.0,
        0.0,
        300.0,
        600.0,
        1500.0,
        (105.0 - 118.0) * 60.81,
        (120.0 - 118.0) * 60.81,
    ];
    let fq_max_err_hz = fq
        .iter()
        .zip(fq_expected_hz.iter())
        .map(|(f, e)| (f.offset_hz - e).abs())
        .fold(0.0_f64, f64::max);
    emit_w3_metrics(
        "t13_1_schedules",
        json!({
            "record_id": "t13_1_schedules",
            "task": "T13.1",
            "vdlist_count": vd.len(),
            "vdlist_max_abs_err_sec": vd_max_err,
            "vclist_count": counts.len(),
            "nu_cpmg_max_hz": nu_cpmg[4],
            "nu_cpmg_max_abs_err_hz": (nu_cpmg[4] - 400.0).abs(),
            "fqlist_count": fq.len(),
            "fqlist_max_abs_err_hz": fq_max_err_hz,
            "varian_param_name": v_sched.param_name,
            "varian_relaxt_count": v_sched.schedule.primary_values().len(),
            "varian_interleaving_modes_verified": 2
        }),
    );
}

#[allow(clippy::too_many_arguments)]
fn write_synthetic_bruker_pseudo3d_dir(
    dir: &Path,
    interleaving: Pseudo3DInterleaving,
    delays_sec: &[f64],
    peaks: &[(f64, f64, f64, f64)], // (delta_h, delta_n, i0, r_rate)
    noise_td_sd: f64,
    seed: u64,
) {
    let n0 = 128usize; // complex points direct 1H
    let n1 = 48usize; // complex points indirect 15N
    let td_x = 2 * n0;
    let td_y = 2 * n1;
    let n_planes = delays_sec.len();

    let bf1_x = 600.0_f64;
    let sfo1_x = 600.0 * (1.0 + 8.30e-6); // carrier at 8.30 ppm
    let sw_hz_x = 3600.0_f64; // 6.0 ppm sweep (5.3..11.3 ppm)

    let bf1_y = 60.80_f64;
    let sfo1_y = 60.80 * (1.0 + 118.0e-6); // carrier at 118.0 ppm
    let sw_ppm_y = 28.0_f64; // 104..132 ppm
    let sw_hz_y = sw_ppm_y * sfo1_y;

    let acqus = format!(
        "##TITLE= Synthetic Bruker Pseudo3D R2\n\
         ##$PULPROG= <hsqct2etf3gpsi3d>\n\
         ##$PARMODE= 2\n\
         ##$AQSEQ= 0\n\
         ##$BYTORDA= 0\n\
         ##$DTYPA= 2\n\
         ##$AQ_mod= 3\n\
         ##$TD= {td_x}\n\
         ##$SFO1= {sfo1_x:.8}\n\
         ##$BF1= {bf1_x:.8}\n\
         ##$SW_h= {sw_hz_x:.6}\n\
         ##$NUC1= <1H>\n\
         ##$GRPDLY= 0\n\
         ##END=\n"
    );
    fs::write(dir.join("acqus"), acqus).unwrap();

    // Put chemical-shift 15N in acqu2s and QF pseudo-axis in acqu3s
    let acqu2s = format!(
        "##TITLE= Indirect 15N\n\
         ##$FnMODE= 4\n\
         ##$TD= {td_y}\n\
         ##$SFO1= {sfo1_y:.8}\n\
         ##$BF1= {bf1_y:.8}\n\
         ##$SW= {sw_ppm_y:.6}\n\
         ##$SW_h= {sw_hz_y:.6}\n\
         ##$NUC1= <15N>\n\
         ##END=\n"
    );
    fs::write(dir.join("acqu2s"), acqu2s).unwrap();

    let acqu3s = format!(
        "##TITLE= Pseudo Delay Axis\n\
         ##$FnMODE= 1\n\
         ##$TD= {n_planes}\n\
         ##$SFO1= {sfo1_y:.8}\n\
         ##$BF1= {bf1_y:.8}\n\
         ##$SW= 1.0\n\
         ##$SW_h= 60.8\n\
         ##$NUC1= <off>\n\
         ##END=\n"
    );
    fs::write(dir.join("acqu3s"), acqu3s).unwrap();

    let il_tag = match interleaving {
        Pseudo3DInterleaving::PlaneQueued => "plane_queued",
        Pseudo3DInterleaving::T1PlaneQuad => "t1_plane_quad",
        Pseudo3DInterleaving::T1QuadPlane => "t1_quad_plane",
    };
    fs::write(
        dir.join("pulseprogram"),
        format!("; synthetic pseudo3d: {il_tag}\n; f2 reverse: false\n"),
    )
    .unwrap();

    let pdata1 = dir.join("pdata").join("1");
    fs::create_dir_all(&pdata1).unwrap();
    fs::write(
        pdata1.join("procs"),
        "##$PHC0= 0\n##$PHC1= 0\n##$WDW= 1\n##$LB= 4.0\n##$SI= 256\n##$BC_mod= 0\n##END=\n",
    )
    .unwrap();
    fs::write(
        pdata1.join("proc2s"),
        "##$PHC0= 0\n##$PHC1= 0\n##$WDW= 1\n##$LB= 3.0\n##$SI= 128\n##$REVERSE= no\n##$BC_mod= 0\n##END=\n",
    )
    .unwrap();

    let vd_lines: Vec<String> = delays_sec.iter().map(|d| format!("{:.6}s", d)).collect();
    fs::write(dir.join("vdlist"), vd_lines.join("\n") + "\n").unwrap();

    // Generate complex DQD + States time-domain traces with Bruker sign convention:
    // Downfield offset `f_hz = (delta_ppm * bf1 - sfo1 * 1e6)` (wait: `delta_ppm - carrier_ppm` * obs_mhz):
    // With `conjugate_for_fft == false` inside `process.rs` (since `axis.conjugate_for_fft == true` in `fid.rs`),
    // downfield frequencies oscillate as `exp(+i 2pi f t)`.
    let carrier_ppm_x = (sfo1_x - bf1_x) * 1.0e6 / bf1_x;
    let carrier_ppm_y = (sfo1_y - bf1_y) * 1.0e6 / bf1_y;
    let dt_x = 1.0 / sw_hz_x;
    let dt_y = 1.0 / sw_hz_y;
    let r2_td_x = 140.0_f64;
    let r2_td_y = 95.0_f64;

    let total_traces = n_planes * n1 * 2;
    let mut traces = vec![vec![[0.0_f64; 2]; n0]; total_traces];
    let mut rng = Rng::new(seed);

    for (p_idx, &t_delay) in delays_sec.iter().enumerate() {
        for k1 in 0..n1 {
            let t1 = (k1 as f64) * dt_y;
            let env1 = (-r2_td_y * t1).exp();
            for q in 0..2 {
                let tidx = match interleaving {
                    Pseudo3DInterleaving::PlaneQueued => (p_idx * n1 + k1) * 2 + q,
                    Pseudo3DInterleaving::T1PlaneQuad => (k1 * n_planes + p_idx) * 2 + q,
                    Pseudo3DInterleaving::T1QuadPlane => (k1 * 2 + q) * n_planes + p_idx,
                };
                let tr = &mut traces[tidx];
                for &(dh, dn, i0, r_rate) in peaks {
                    let amp = i0 * (-r_rate * t_delay).exp();
                    let fx = (dh - carrier_ppm_x) * bf1_x;
                    let fy = (dn - carrier_ppm_y) * bf1_y;
                    let mod_y = if q == 0 {
                        (2.0 * PI * fy * t1).cos()
                    } else {
                        (2.0 * PI * fy * t1).sin()
                    } * env1;
                    for (k0, sample) in tr.iter_mut().enumerate() {
                        let t0 = (k0 as f64) * dt_x;
                        let env0 = (-r2_td_x * t0).exp();
                        let (s0, c0) = (2.0 * PI * fx * t0).sin_cos();
                        let w = amp * mod_y * env0;
                        sample[0] += w * c0;
                        sample[1] += w * s0;
                    }
                }
                if noise_td_sd > 0.0 {
                    for sample in tr.iter_mut() {
                        sample[0] += rng.normal(0.0, noise_td_sd);
                        sample[1] += rng.normal(0.0, noise_td_sd);
                    }
                }
            }
        }
    }

    let mut ser_bytes = Vec::with_capacity(total_traces * n0 * 16);
    for tr in &traces {
        for &[re, im] in tr {
            ser_bytes.extend_from_slice(&re.to_le_bytes());
            ser_bytes.extend_from_slice(&im.to_le_bytes());
        }
    }
    fs::write(dir.join("ser"), ser_bytes).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn write_synthetic_varian_pseudo3d_dir(
    dir: &Path,
    interleaving: Pseudo3DInterleaving,
    delays_sec: &[f64],
    peaks: &[(f64, f64, f64, f64)],
    noise_td_sd: f64,
    seed: u64,
) {
    let n0 = 128usize;
    let ni = 48usize;
    let np = 2 * n0;
    let n_planes = delays_sec.len();

    let temp_c = 25.0_f64;
    let h2o_ppm = 4.766 - 0.0119 * (temp_c - 25.0);
    let bf_1h = 600.0_f64;
    let sfrq = bf_1h * (1.0 + h2o_ppm * 1.0e-6); // carrier at water ~4.766 ppm; let's use wide sw so 6..10.5 ppm is clean
    let sw_x = 8400.0_f64; // 14 ppm (-2.2..11.8 ppm)
    let obs_n = bf_1h * 0.101_329_118;
    let dfrq2 = obs_n * (1.0 + 118.0e-6);
    let sw_y = 28.0 * obs_n;

    let (array_str, seq_mode_line) = match interleaving {
        Pseudo3DInterleaving::PlaneQueued => (
            "relaxT,phase",
            "seq_mode 2 2 8192 0 0 2 1 0 1 64\n1 \"plane_queued\"\n0\n",
        ),
        Pseudo3DInterleaving::T1PlaneQuad => ("relaxT,phase", ""),
        Pseudo3DInterleaving::T1QuadPlane => ("phase,relaxT", ""),
    };
    let delays_joined: String = delays_sec
        .iter()
        .map(|d| format!("{d:.6}"))
        .collect::<Vec<_>>()
        .join(" ");

    let procpar = format!(
        "pslabel 2 2 8192 0 0 2 1 0 1 64\n1 \"gNhsqc_T1\"\n0\n\
         array 2 2 8192 0 0 2 1 0 1 64\n1 \"{array_str}\"\n0\n\
         {seq_mode_line}\
         sfrq 1 1 8 0 0 2 1 0 1 64\n1 {sfrq:.8}\n0\n\
         dfrq2 1 1 8 0 0 2 1 0 1 64\n1 {dfrq2:.8}\n0\n\
         dn2 2 2 8192 0 0 2 1 0 1 64\n1 \"N15\"\n0\n\
         temp 1 1 8 0 0 2 1 0 1 64\n1 25.0\n0\n\
         sw 1 1 8 0 0 2 1 0 1 64\n1 {sw_x:.6}\n0\n\
         sw1 1 1 8 0 0 2 1 0 1 64\n1 {sw_y:.6}\n0\n\
         ni 1 1 8 0 0 2 1 0 1 64\n1 {ni}\n0\n\
         rp 1 1 8 0 0 2 1 0 1 64\n1 0.0\n0\n\
         lp 1 1 8 0 0 2 1 0 1 64\n1 0.0\n0\n\
         rp1 1 1 8 0 0 2 1 0 1 64\n1 0.0\n0\n\
         lp1 1 1 8 0 0 2 1 0 1 64\n1 0.0\n0\n\
         fn 1 1 8 0 0 2 1 0 1 64\n1 512\n0\n\
         fn1 1 1 8 0 0 2 1 0 1 64\n1 256\n0\n\
         phase 1 1 8 0 0 2 1 0 1 64\n2 1 2\n0\n\
         relaxT 1 1 8 0 0 2 1 0 1 64\n{} {delays_joined}\n0\n",
        delays_sec.len()
    );
    fs::write(dir.join("procpar"), procpar).unwrap();

    let carrier_ppm_x = (sfrq - bf_1h) * 1.0e6 / bf_1h;
    let carrier_ppm_y = (dfrq2 - obs_n) * 1.0e6 / obs_n;
    let dt_x = 1.0 / sw_x;
    let dt_y = 1.0 / sw_y;
    let r2_td_x = 140.0_f64;
    let r2_td_y = 95.0_f64;

    let total_traces = n_planes * ni * 2;
    let mut traces = vec![vec![[0.0_f64; 2]; n0]; total_traces];
    let mut rng = Rng::new(seed);

    // Varian convention: downfield oscillates as `exp(-i 2pi f t)` (`axis.conjugate_for_fft == false` in `fid.rs`,
    // conjugated inside `process.rs`).
    for (p_idx, &t_delay) in delays_sec.iter().enumerate() {
        for k1 in 0..ni {
            let t1 = (k1 as f64) * dt_y;
            let env1 = (-r2_td_y * t1).exp();
            for q in 0..2 {
                let tidx = match interleaving {
                    Pseudo3DInterleaving::PlaneQueued => (p_idx * ni + k1) * 2 + q,
                    Pseudo3DInterleaving::T1PlaneQuad => (k1 * n_planes + p_idx) * 2 + q,
                    Pseudo3DInterleaving::T1QuadPlane => (k1 * 2 + q) * n_planes + p_idx,
                };
                let tr = &mut traces[tidx];
                for &(dh, dn, i0, r_rate) in peaks {
                    let amp = i0 * (-r_rate * t_delay).exp();
                    let fx = (dh - carrier_ppm_x) * bf_1h;
                    let fy = (dn - carrier_ppm_y) * obs_n;
                    let mod_y = if q == 0 {
                        (2.0 * PI * fy * t1).cos()
                    } else {
                        -(2.0 * PI * fy * t1).sin()
                    } * env1;
                    for (k0, sample) in tr.iter_mut().enumerate() {
                        let t0 = (k0 as f64) * dt_x;
                        let env0 = (-r2_td_x * t0).exp();
                        let (s0, c0) = (-2.0 * PI * fx * t0).sin_cos();
                        let w = amp * mod_y * env0;
                        sample[0] += w * c0;
                        sample[1] += w * s0;
                    }
                }
                if noise_td_sd > 0.0 {
                    for sample in tr.iter_mut() {
                        sample[0] += rng.normal(0.0, noise_td_sd);
                        sample[1] += rng.normal(0.0, noise_td_sd);
                    }
                }
            }
        }
    }

    // Write Varian binary `fid`: 1 block containing all `total_traces` in IEEE754 big-endian f32 (`status = 0x000D`)
    let ebytes = 4i32;
    let tbytes = (np as i32) * ebytes;
    let bbytes = 28 + (total_traces as i32) * tbytes;
    let status: u16 = 0x0001 | 0x0004 | 0x0008; // S_DATA | S_32 | S_FLOAT
    let mut fid_bytes = Vec::with_capacity(32 + (bbytes as usize));
    fid_bytes.extend_from_slice(&1i32.to_be_bytes()); // nblocks
    fid_bytes.extend_from_slice(&(total_traces as i32).to_be_bytes()); // ntraces
    fid_bytes.extend_from_slice(&(np as i32).to_be_bytes()); // np
    fid_bytes.extend_from_slice(&ebytes.to_be_bytes()); // ebytes
    fid_bytes.extend_from_slice(&tbytes.to_be_bytes()); // tbytes
    fid_bytes.extend_from_slice(&bbytes.to_be_bytes()); // bbytes
    fid_bytes.extend_from_slice(&0i16.to_be_bytes()); // vers_id
    fid_bytes.extend_from_slice(&status.to_be_bytes()); // status
    fid_bytes.extend_from_slice(&1i32.to_be_bytes()); // nbheaders

    // 28-byte datablockhead
    fid_bytes.extend_from_slice(&1i16.to_be_bytes()); // scale
    fid_bytes.extend_from_slice(&status.to_be_bytes()); // status
    fid_bytes.extend_from_slice(&1i16.to_be_bytes()); // index
    fid_bytes.extend_from_slice(&0i16.to_be_bytes()); // mode
    fid_bytes.extend_from_slice(&1i32.to_be_bytes()); // ctcount
    fid_bytes.extend_from_slice(&0.0f32.to_be_bytes()); // lpval
    fid_bytes.extend_from_slice(&0.0f32.to_be_bytes()); // rpval
    fid_bytes.extend_from_slice(&0.0f32.to_be_bytes()); // lvl
    fid_bytes.extend_from_slice(&0.0f32.to_be_bytes()); // tlt

    for tr in &traces {
        for &[re, im] in tr {
            fid_bytes.extend_from_slice(&(re as f32).to_be_bytes());
            fid_bytes.extend_from_slice(&(im as f32).to_be_bytes());
        }
    }
    fs::write(dir.join("fid"), fid_bytes).unwrap();
}

#[test]
fn bruker_and_varian_raw_pseudo3d_interleaving_modes_match_identically() {
    let delays = vec![0.01, 0.05, 0.12, 0.25, 0.05, 0.50];
    let peaks = vec![
        (8.42, 120.5, 100.0, 2.2),
        (7.65, 114.8, 95.0, 1.6),
        (9.10, 125.2, 110.0, 3.1),
    ];
    let seeds: Vec<TargetPeakSeed> = peaks
        .iter()
        .enumerate()
        .map(|(i, &(dh, dn, _, _))| TargetPeakSeed {
            seq_id: (i as i32) + 2,
            res_name: "ALA".to_string(),
            atom_h: "H".to_string(),
            atom_x: "N".to_string(),
            delta_h_ppm: dh,
            delta_x_ppm: dn,
        })
        .collect();

    // Test all 3 interleaving modes for Bruker and Varian (noiseless so every interleaving order must decode identically!)
    let mut bruker_results = Vec::new();
    for (idx, mode) in [
        Pseudo3DInterleaving::PlaneQueued,
        Pseudo3DInterleaving::T1PlaneQuad,
        Pseudo3DInterleaving::T1QuadPlane,
    ]
    .into_iter()
    .enumerate()
    {
        let bdir = temp_test_dir(&format!("bruker_il_{idx}"));
        write_synthetic_bruker_pseudo3d_dir(&bdir, mode, &delays, &peaks, 0.0, 101);
        let raw = read_bruker_pseudo3d(&bdir, None, None).expect("read bruker pseudo3d");
        assert_eq!(raw.interleaving, mode);
        assert_eq!(raw.planes.len(), delays.len());

        let (specs, recipe) = process_pseudo3d_fids(&raw.planes, None).expect("process bruker");
        assert!(recipe
            .dims
            .iter()
            .all(|d| d.phase_source == PhaseSource::Explicit));
        let ds = integrate_relax_planes(
            "bruker_test",
            raw.experiment_kind,
            raw.schedule,
            &specs,
            &seeds,
            &RelaxIntegrationConfig::default(),
        )
        .expect("integrate bruker");
        bruker_results.push(ds);
        let _ = fs::remove_dir_all(&bdir);
    }

    let mut varian_results = Vec::new();
    for (idx, mode) in [
        Pseudo3DInterleaving::PlaneQueued,
        Pseudo3DInterleaving::T1PlaneQuad,
        Pseudo3DInterleaving::T1QuadPlane,
    ]
    .into_iter()
    .enumerate()
    {
        let vdir = temp_test_dir(&format!("varian_il_{idx}"));
        write_synthetic_varian_pseudo3d_dir(&vdir, mode, &delays, &peaks, 0.0, 202);
        let raw = read_varian_pseudo3d(&vdir, None, None).expect("read varian pseudo3d");
        assert_eq!(raw.interleaving, mode);
        assert_eq!(raw.planes.len(), delays.len());

        let mut rec = nmr_core::process::ProcessingRecipe::from_fid_2d(&raw.planes[0]);
        rec.dims[0].solvent_filter = None;
        rec.dims[0].baseline = Some(BaselineConfig {
            enabled: false,
            poly_order: 1,
            mad_sigma: 3.0,
        });
        rec.dims[0].window = ApodizationWindow::Exponential { lb_hz: 4.0 };
        rec.dims[1].window = ApodizationWindow::Exponential { lb_hz: 3.0 };

        let (specs, _) = process_pseudo3d_fids(&raw.planes, Some(&rec)).expect("process varian");
        let ds = integrate_relax_planes(
            "varian_test",
            raw.experiment_kind,
            raw.schedule,
            &specs,
            &seeds,
            &RelaxIntegrationConfig::default(),
        )
        .expect("integrate varian");
        varian_results.push(ds);
        let _ = fs::remove_dir_all(&vdir);
    }

    let mut bruker_max_il_diff = 0.0_f64;
    let mut varian_max_il_diff = 0.0_f64;
    for mode_idx in 1..3 {
        for r in 0..peaks.len() {
            for p in 0..delays.len() {
                let b0 = bruker_results[0].traces[r].intensities[p];
                let bm = bruker_results[mode_idx].traces[r].intensities[p];
                let b_rel = ((bm - b0) / b0.abs()).abs();
                bruker_max_il_diff = bruker_max_il_diff.max(b_rel);
                assert!(
                    b_rel < 1.0e-11,
                    "Bruker interleaving mode {mode_idx} mismatch at res {r}, plane {p}: {bm} vs {b0}"
                );

                let v0 = varian_results[0].traces[r].intensities[p];
                let vm = varian_results[mode_idx].traces[r].intensities[p];
                let v_rel = ((vm - v0) / v0.abs()).abs();
                varian_max_il_diff = varian_max_il_diff.max(v_rel);
                assert!(
                    v_rel < 1.0e-11,
                    "Varian interleaving mode {mode_idx} mismatch at res {r}, plane {p}: {vm} vs {v0}"
                );
            }
        }
    }

    // Verify normalized decay ratios I(t)/I(t_0) match exp(-R*(t - t_0)) to < 0.2%
    let mut bruker_max_ratio_err = 0.0_f64;
    let mut varian_max_ratio_err = 0.0_f64;
    for (r, &(_, _, _, r_true)) in peaks.iter().enumerate() {
        let b_i0 = bruker_results[0].traces[r].intensities[0];
        let v_i0 = varian_results[0].traces[r].intensities[0];
        for (p, &t) in delays.iter().enumerate() {
            let expected_ratio = (-r_true * (t - delays[0])).exp();
            let b_ratio = bruker_results[0].traces[r].intensities[p] / b_i0;
            let v_ratio = varian_results[0].traces[r].intensities[p] / v_i0;
            let b_err = (b_ratio - expected_ratio).abs();
            let v_err = (v_ratio - expected_ratio).abs();
            bruker_max_ratio_err = bruker_max_ratio_err.max(b_err);
            varian_max_ratio_err = varian_max_ratio_err.max(v_err);
            assert!(
                b_err < 2.0e-3,
                "Bruker decay ratio {b_ratio:.5} vs {expected_ratio:.5}"
            );
            assert!(
                v_err < 2.0e-3,
                "Varian decay ratio {v_ratio:.5} vs {expected_ratio:.5}"
            );
        }
    }

    emit_w3_metrics(
        "t13_1_interleaving",
        json!({
            "record_id": "t13_1_interleaving",
            "task": "T13.1",
            "n_planes": delays.len(),
            "n_peaks": peaks.len(),
            "interleaving_modes": 3,
            "bruker_max_rel_interleaving_diff": bruker_max_il_diff,
            "varian_max_rel_interleaving_diff": varian_max_il_diff,
            "bruker_max_decay_ratio_err": bruker_max_ratio_err,
            "varian_max_decay_ratio_err": varian_max_ratio_err
        }),
    );
}

#[test]
fn twenty_plane_bruker_and_varian_snr4_done_gate_eliminates_positive_picking_bias() {
    // T13.1 Quantitative Done Gate:
    // Across a 20-plane synthetic Bruker & Varian pseudo-3D decay series down to SNR = 4,
    // locked-footprint integration recovers true amplitudes with `< 1.5%` systematic bias,
    // while naive local-maximum picking suffers `> 8%` positive bias at `SNR = 4`!

    // Part A: 20-plane analytical Voigt/Gaussian Spectrum2D stack (30 well-resolved residues)
    // where peak height on plane 0 is SNR = 80 and decays across 20 planes down to exact SNR = 4.0!
    let nx = 256usize;
    let ny = 128usize;
    let ax_x = AxisPpm::from_bounds("1H", nx, 10.5, 6.5, 600.0);
    let ax_y = AxisPpm::from_bounds("15N", ny, 130.0, 105.0, 60.81);
    let sigma_noise = 1.0_f64;
    let i0_true = 80.0_f64; // SNR = 80 on t = 0
    let r_true = 3.0_f64; // s^-1
                          // Final delay t_19 chosen so I(t_19) = 80 * exp(-3 * t_19) = 4.0 (exact SNR = 4.0!)
    let t_max = (80.0_f64 / 4.0_f64).ln() / r_true;
    let mut delays = Vec::with_capacity(20);
    for p in 0..18 {
        delays.push((p as f64) * t_max / 17.0);
    }
    // Include 2 duplicate delays (planes 18 & 19 duplicate planes 2 and 17)
    delays.push(delays[2]);
    delays.push(t_max); // duplicate of SNR = 4 plane!

    let n_res = 30usize; // 6 x 5 well-separated grid of backbone crosspeaks
    let mut seeds = Vec::with_capacity(n_res);
    let sigma_h = 0.024_f64;
    let sigma_n = 0.26_f64;
    for gy in 0..5 {
        for gx in 0..6 {
            let idx = gy * 6 + gx;
            let dh = 6.95 + (gx as f64) * 0.58;
            let dn = 108.0 + (gy as f64) * 4.6;
            seeds.push(TargetPeakSeed {
                seq_id: idx + 1,
                res_name: "VAL".to_string(),
                atom_h: "H".to_string(),
                atom_x: "N".to_string(),
                delta_h_ppm: dh,
                delta_x_ppm: dn,
            });
        }
    }

    // Evaluate ensemble systematic bias across 12 independent realizations (360 peak decays x 20 planes = 7,200 points)
    let n_trials = 12usize;
    let mut sum_locked_all = 0.0_f64;
    let mut sum_true_all = 0.0_f64;
    let mut sum_locked_snr4 = 0.0_f64;
    let mut sum_naive_snr4 = 0.0_f64;
    let mut sum_true_snr4 = 0.0_f64;
    let mut trial0_mad_max_err = 0.0_f64;
    let mut trial0_dup_dof = 0usize;
    let mut trial0_dup_var_scale = 1.0_f64;

    let cfg = RelaxIntegrationConfig {
        shape_model: FootprintShapeModel::Gaussian,
        patch_half_width_pts: [4, 4],
        max_centroid_shift_ppm: [0.03, 0.30],
        ref_plane_indices: Some(vec![0]),
        fit_local_baseline: false,
        duplicate_tol: 1.0e-7,
        account_spatial_noise_corr: false,
    };

    for trial in 0..n_trials {
        let mut rng = Rng::new(7000 + (trial as u64) * 97);
        let mut planes = Vec::with_capacity(20);
        for (p_idx, &t) in delays.iter().enumerate() {
            let amp_p = i0_true * (-r_true * t).exp();
            let mut data = vec![0.0f32; ny * nx];
            for iy in 0..ny {
                let py = ax_y.index_to_ppm(iy as f64);
                for ix in 0..nx {
                    let px = ax_x.index_to_ppm(ix as f64);
                    let mut val = rng.normal(0.0, sigma_noise);
                    for s in &seeds {
                        if (px - s.delta_h_ppm).abs() <= 0.15 && (py - s.delta_x_ppm).abs() <= 1.6 {
                            val += amp_p
                                * footprint_eval_2d(
                                    px,
                                    py,
                                    s.delta_h_ppm,
                                    s.delta_x_ppm,
                                    sigma_h,
                                    sigma_n,
                                    0.0,
                                    0.0,
                                );
                        }
                    }
                    data[iy * nx + ix] = val as f32;
                }
            }
            planes.push(Spectrum2D {
                name: format!("plane_{p_idx:02}"),
                axis_x: ax_x.clone(),
                axis_y: ax_y.clone(),
                data,
            });
        }

        let ds = integrate_relax_planes(
            "snr4_stack",
            RelaxExperimentKind::R2,
            RelaxSchedule::VariableDelaysSec {
                delays_sec: delays.clone(),
            },
            &planes,
            &seeds,
            &cfg,
        )
        .expect("integrate synthetic 20-plane stack");
        let naive = extract_naive_local_max_series(&planes, &seeds, [2, 2]);

        // Check MAD noise recovery on trial 0
        if trial == 0 {
            for &mad in &ds.plane_noise_mad {
                assert!(
                    (mad - sigma_noise).abs() < 0.08,
                    "2D MAD {mad:.4} should match sigma_noise={sigma_noise}"
                );
            }
            assert_eq!(ds.duplicate_dof, 2 * n_res);
            assert!(
                (ds.duplicate_variance_scale - 1.0).abs() < 0.25,
                "duplicate_variance_scale={:.4} should be ~1.0 under stationary Gaussian noise",
                ds.duplicate_variance_scale
            );
            trial0_mad_max_err = ds
                .plane_noise_mad
                .iter()
                .map(|&m| (m - sigma_noise).abs())
                .fold(0.0_f64, f64::max);
            trial0_dup_dof = ds.duplicate_dof;
            trial0_dup_var_scale = ds.duplicate_variance_scale;
        }

        for (tr, nv) in ds.traces.iter().zip(naive.iter()).take(n_res) {
            for (p_idx, &t) in delays.iter().enumerate() {
                if p_idx == 0 {
                    continue; // exclude reference plane 0 so test evaluates pure out-of-sample decay planes
                }
                let true_amp = i0_true * (-r_true * t).exp();
                sum_locked_all += tr.intensities[p_idx];
                sum_true_all += true_amp;
                if (true_amp - 4.0).abs() < 1.0e-6 {
                    sum_locked_snr4 += tr.intensities[p_idx];
                    sum_naive_snr4 += nv[p_idx];
                    sum_true_snr4 += true_amp;
                }
            }
        }
    }

    let rel_bias_locked_all = ((sum_locked_all - sum_true_all) / sum_true_all).abs() * 100.0;
    let rel_bias_locked_snr4 = ((sum_locked_snr4 - sum_true_snr4) / sum_true_snr4).abs() * 100.0;
    let rel_bias_naive_snr4 = ((sum_naive_snr4 - sum_true_snr4) / sum_true_snr4) * 100.0;

    assert!(
        rel_bias_locked_all < 1.5,
        "20-plane series locked-footprint systematic bias {rel_bias_locked_all:.3}% must be < 1.5%"
    );
    assert!(
        rel_bias_locked_snr4 < 1.5,
        "SNR=4 plane locked-footprint systematic bias {rel_bias_locked_snr4:.3}% must be < 1.5%"
    );
    assert!(
        rel_bias_naive_snr4 > 8.0,
        "SNR=4 naive local-max bias +{rel_bias_naive_snr4:.3}% must demonstrate > 8% positive noise bias"
    );

    // Part B: End-to-end 20-plane raw Bruker (`ser` + `vdlist`, interleaved) & raw Varian (`fid` + `procpar`, interleaved)
    // down to SNR = 4 in the processed 2D FT spectra!
    let raw_peaks: Vec<(f64, f64, f64, f64)> = (0..12)
        .map(|i| {
            let gx = i % 4;
            let gy = i / 4;
            (
                7.20 + (gx as f64) * 0.65,
                110.5 + (gy as f64) * 5.5,
                80.0,
                r_true,
            )
        })
        .collect();
    let raw_seeds: Vec<TargetPeakSeed> = raw_peaks
        .iter()
        .enumerate()
        .map(|(i, &(dh, dn, _, _))| TargetPeakSeed {
            seq_id: (i as i32) + 10,
            res_name: "LEU".to_string(),
            atom_h: "H".to_string(),
            atom_x: "N".to_string(),
            delta_h_ppm: dh,
            delta_x_ppm: dn,
        })
        .collect();

    // Calibrate noiseless processed peak height first so we set exact time-domain Gaussian SD for SNR = 4 at t_max!
    let b_cal_dir = temp_test_dir("bruker_20p_cal");
    write_synthetic_bruker_pseudo3d_dir(
        &b_cal_dir,
        Pseudo3DInterleaving::T1QuadPlane,
        &delays,
        &raw_peaks,
        0.0,
        1,
    );
    let b_cal_raw = read_bruker_pseudo3d(&b_cal_dir, None, None).unwrap();
    let mut b_rec = nmr_core::process::ProcessingRecipe::from_fid_2d(&b_cal_raw.planes[0]);
    b_rec.dims[0].solvent_filter = None;
    b_rec.dims[0].baseline = Some(BaselineConfig {
        enabled: false,
        poly_order: 1,
        mad_sigma: 3.0,
    });
    let (b_cal_specs, b_locked_rec) =
        process_pseudo3d_fids(&b_cal_raw.planes, Some(&b_rec)).unwrap();
    let b_cal_ds = integrate_relax_planes(
        "b_cal",
        RelaxExperimentKind::R2,
        b_cal_raw.schedule.clone(),
        &b_cal_specs,
        &raw_seeds,
        &RelaxIntegrationConfig::default(),
    )
    .unwrap();
    let b_cal_naive = extract_naive_local_max_series(&b_cal_specs, &raw_seeds, [4, 4]);
    let b_voxel_t0: f64 = b_cal_naive.iter().map(|v| v[0]).sum::<f64>() / (raw_peaks.len() as f64);
    let _ = fs::remove_dir_all(&b_cal_dir);

    // Measure processed MAD per unit time-domain noise
    let b_noise_dir = temp_test_dir("bruker_20p_ncal");
    write_synthetic_bruker_pseudo3d_dir(
        &b_noise_dir,
        Pseudo3DInterleaving::T1QuadPlane,
        &delays[..2],
        &[],
        1.0,
        2,
    );
    let b_noise_raw = read_bruker_pseudo3d(&b_noise_dir, None, None).unwrap();
    let (b_noise_specs, _) =
        process_pseudo3d_fids(&b_noise_raw.planes, Some(&b_locked_rec)).unwrap();
    let mad_per_unit_td = estimate_plane_noise_mad_2d(&b_noise_specs[0]);
    let _ = fs::remove_dir_all(&b_noise_dir);

    // At t_max, true discrete peak height is b_voxel_t0 / 20.0; set noise_td_sd so voxel SNR == 4.0!
    let target_mad = (b_voxel_t0 / 20.0) / 4.0;
    let td_noise_sd = target_mad / mad_per_unit_td;

    let mut b_locked_sum = 0.0_f64;
    let mut b_locked_true_sum = 0.0_f64;
    let mut b_naive_sum = 0.0_f64;
    let mut b_naive_true_sum = 0.0_f64;
    let mut v_locked_sum = 0.0_f64;
    let mut v_locked_true_sum = 0.0_f64;
    let mut v_naive_sum = 0.0_f64;
    let mut v_naive_true_sum = 0.0_f64;

    // Calibrate Varian noiseless reference amplitudes too
    let v_cal_dir = temp_test_dir("varian_20p_cal");
    write_synthetic_varian_pseudo3d_dir(
        &v_cal_dir,
        Pseudo3DInterleaving::T1PlaneQuad,
        &delays,
        &raw_peaks,
        0.0,
        3,
    );
    let v_cal_raw = read_varian_pseudo3d(&v_cal_dir, None, None).unwrap();
    let mut v_rec = nmr_core::process::ProcessingRecipe::from_fid_2d(&v_cal_raw.planes[0]);
    v_rec.dims[0].solvent_filter = None;
    v_rec.dims[0].baseline = Some(BaselineConfig {
        enabled: false,
        poly_order: 1,
        mad_sigma: 3.0,
    });
    // For Varian wide SW (8400 Hz), set r2_td_x = 240 Hz so FWHM spans ~2.3 independent Nyquist bins
    v_rec.dims[0].window = ApodizationWindow::Exponential { lb_hz: 45.0 };
    v_rec.dims[1].window = ApodizationWindow::Exponential { lb_hz: 15.0 };
    let (v_cal_specs, v_locked_rec) =
        process_pseudo3d_fids(&v_cal_raw.planes, Some(&v_rec)).unwrap();
    let v_cal_ds = integrate_relax_planes(
        "v_cal",
        RelaxExperimentKind::R1,
        v_cal_raw.schedule.clone(),
        &v_cal_specs,
        &raw_seeds,
        &RelaxIntegrationConfig::default(),
    )
    .unwrap();
    let v_cal_naive = extract_naive_local_max_series(&v_cal_specs, &raw_seeds, [4, 4]);
    let v_voxel_t0: f64 = v_cal_naive.iter().map(|v| v[0]).sum::<f64>() / (raw_peaks.len() as f64);
    let _ = fs::remove_dir_all(&v_cal_dir);

    let v_noise_dir = temp_test_dir("varian_20p_ncal");
    write_synthetic_varian_pseudo3d_dir(
        &v_noise_dir,
        Pseudo3DInterleaving::T1PlaneQuad,
        &delays[..2],
        &[],
        1.0,
        4,
    );
    let v_noise_raw = read_varian_pseudo3d(&v_noise_dir, None, None).unwrap();
    let (v_noise_specs, _) =
        process_pseudo3d_fids(&v_noise_raw.planes, Some(&v_locked_rec)).unwrap();
    let v_mad_per_unit = estimate_plane_noise_mad_2d(&v_noise_specs[0]);
    let _ = fs::remove_dir_all(&v_noise_dir);
    let v_td_noise_sd = ((v_voxel_t0 / 20.0) / 4.0) / v_mad_per_unit;

    for rep in 0..8 {
        let b_dir = temp_test_dir(&format!("bruker_snr4_{rep}"));
        write_synthetic_bruker_pseudo3d_dir(
            &b_dir,
            Pseudo3DInterleaving::T1QuadPlane,
            &delays,
            &raw_peaks,
            td_noise_sd,
            9000 + rep,
        );
        let b_raw = read_bruker_pseudo3d(&b_dir, None, None).unwrap();
        let (b_specs, _) = process_pseudo3d_fids(&b_raw.planes, Some(&b_locked_rec)).unwrap();
        let b_ds = integrate_relax_planes(
            "b_snr4",
            RelaxExperimentKind::R2,
            b_raw.schedule,
            &b_specs,
            &raw_seeds,
            &RelaxIntegrationConfig {
                ref_plane_indices: Some(vec![0]),
                ..RelaxIntegrationConfig::default()
            },
        )
        .unwrap();
        let b_naive = extract_naive_local_max_series(&b_specs, &raw_seeds, [4, 4]);
        for r in 0..raw_peaks.len() {
            for &p_snr4 in &[17usize, 19usize] {
                b_locked_sum += b_ds.traces[r].intensities[p_snr4];
                b_locked_true_sum += b_cal_ds.traces[r].intensities[p_snr4];
                b_naive_sum += b_naive[r][p_snr4];
                b_naive_true_sum += b_cal_naive[r][p_snr4];
            }
        }
        let _ = fs::remove_dir_all(&b_dir);

        let v_dir = temp_test_dir(&format!("varian_snr4_{rep}"));
        write_synthetic_varian_pseudo3d_dir(
            &v_dir,
            Pseudo3DInterleaving::T1PlaneQuad,
            &delays,
            &raw_peaks,
            v_td_noise_sd,
            9500 + rep,
        );
        let v_raw = read_varian_pseudo3d(&v_dir, None, None).unwrap();
        let (v_specs, _) = process_pseudo3d_fids(&v_raw.planes, Some(&v_locked_rec)).unwrap();
        let v_ds = integrate_relax_planes(
            "v_snr4",
            RelaxExperimentKind::R1,
            v_raw.schedule,
            &v_specs,
            &raw_seeds,
            &RelaxIntegrationConfig {
                ref_plane_indices: Some(vec![0]),
                ..RelaxIntegrationConfig::default()
            },
        )
        .unwrap();
        let v_naive = extract_naive_local_max_series(&v_specs, &raw_seeds, [4, 4]);
        for r in 0..raw_peaks.len() {
            for &p_snr4 in &[17usize, 19usize] {
                v_locked_sum += v_ds.traces[r].intensities[p_snr4];
                v_locked_true_sum += v_cal_ds.traces[r].intensities[p_snr4];
                v_naive_sum += v_naive[r][p_snr4];
                v_naive_true_sum += v_cal_naive[r][p_snr4];
            }
        }
        let _ = fs::remove_dir_all(&v_dir);
    }

    let b_locked_bias_pct = ((b_locked_sum - b_locked_true_sum) / b_locked_true_sum).abs() * 100.0;
    let b_naive_bias_pct = ((b_naive_sum - b_naive_true_sum) / b_naive_true_sum) * 100.0;
    let v_locked_bias_pct = ((v_locked_sum - v_locked_true_sum) / v_locked_true_sum).abs() * 100.0;
    let v_naive_bias_pct = ((v_naive_sum - v_naive_true_sum) / v_naive_true_sum) * 100.0;
    println!(
        "[T13.1 Raw 20-Plane SNR=4 Gate] Bruker: locked={b_locked_bias_pct:.3}%, naive=+{b_naive_bias_pct:.2}% | Varian: locked={v_locked_bias_pct:.3}%, naive=+{v_naive_bias_pct:.2}%"
    );

    assert!(
        b_locked_bias_pct < 1.5,
        "Bruker raw 20-plane SNR=4 locked-footprint bias {b_locked_bias_pct:.3}% must be < 1.5%"
    );
    assert!(
        b_naive_bias_pct > 8.0,
        "Bruker raw 20-plane SNR=4 naive local-max bias +{b_naive_bias_pct:.3}% must be > 8.0%"
    );
    assert!(
        v_locked_bias_pct < 1.5,
        "Varian raw 20-plane SNR=4 locked-footprint bias {v_locked_bias_pct:.3}% must be < 1.5%"
    );
    assert!(
        v_naive_bias_pct > 8.0,
        "Varian raw 20-plane SNR=4 naive local-max bias +{v_naive_bias_pct:.3}% must be > 8.0%"
    );

    emit_w3_metrics(
        "t13_1_snr4_gate",
        json!({
            "record_id": "t13_1_snr4_gate",
            "task": "T13.1",
            "n_planes": delays.len(),
            "snr_initial": i0_true / sigma_noise,
            "snr_final": 4.0,
            "stack_n_residues": n_res,
            "stack_n_trials": n_trials,
            "stack_total_points": n_res * n_trials * delays.len(),
            "stack_locked_all_bias_pct": rel_bias_locked_all,
            "stack_locked_snr4_bias_pct": rel_bias_locked_snr4,
            "stack_naive_snr4_bias_pct": rel_bias_naive_snr4,
            "stack_mad_max_abs_err": trial0_mad_max_err,
            "stack_duplicate_dof": trial0_dup_dof,
            "stack_duplicate_var_scale": trial0_dup_var_scale,
            "bruker_raw_n_peaks": raw_peaks.len(),
            "bruker_raw_n_reps": 8,
            "bruker_raw_locked_snr4_bias_pct": b_locked_bias_pct,
            "bruker_raw_naive_snr4_bias_pct": b_naive_bias_pct,
            "varian_raw_n_peaks": raw_peaks.len(),
            "varian_raw_n_reps": 8,
            "varian_raw_locked_snr4_bias_pct": v_locked_bias_pct,
            "varian_raw_naive_snr4_bias_pct": v_naive_bias_pct
        }),
    );
}

#[test]
fn overlapping_footprints_deconvolve_and_tsv_nef_roundtrip_cleanly() {
    let nx = 128usize;
    let ny = 64usize;
    let ax_x = AxisPpm::from_bounds("1H", nx, 9.5, 7.5, 600.0);
    let ax_y = AxisPpm::from_bounds("15N", ny, 124.0, 114.0, 60.81);

    // Two closely spaced overlapping crosspeaks (separated by ~1.2 sigma) with distinct decay rates!
    let seeds = vec![
        TargetPeakSeed {
            seq_id: 24,
            res_name: "GLU".to_string(),
            atom_h: "H".to_string(),
            atom_x: "N".to_string(),
            delta_h_ppm: 8.400,
            delta_x_ppm: 119.00,
        },
        TargetPeakSeed {
            seq_id: 25,
            res_name: "ASN".to_string(),
            atom_h: "H".to_string(),
            atom_x: "N".to_string(),
            delta_h_ppm: 8.432,
            delta_x_ppm: 119.22,
        },
    ];
    let delays = vec![0.0, 0.1, 0.3, 0.6, 0.1];
    let rates = [1.5_f64, 4.5_f64];
    let amps0 = [120.0_f64, 90.0_f64];
    let sh = 0.025_f64;
    let sn = 0.22_f64;

    let mut planes = Vec::new();
    for (p_idx, &t) in delays.iter().enumerate() {
        let mut data = vec![0.0f32; ny * nx];
        for iy in 0..ny {
            let py = ax_y.index_to_ppm(iy as f64);
            for ix in 0..nx {
                let px = ax_x.index_to_ppm(ix as f64);
                let v = amps0[0]
                    * (-rates[0] * t).exp()
                    * footprint_eval_2d(
                        px,
                        py,
                        seeds[0].delta_h_ppm,
                        seeds[0].delta_x_ppm,
                        sh,
                        sn,
                        0.0,
                        0.0,
                    )
                    + amps0[1]
                        * (-rates[1] * t).exp()
                        * footprint_eval_2d(
                            px,
                            py,
                            seeds[1].delta_h_ppm,
                            seeds[1].delta_x_ppm,
                            sh,
                            sn,
                            0.0,
                            0.0,
                        );
                data[iy * nx + ix] = v as f32;
            }
        }
        planes.push(Spectrum2D {
            name: format!("ov_p{p_idx}"),
            axis_x: ax_x.clone(),
            axis_y: ax_y.clone(),
            data,
        });
    }

    let cfg = RelaxIntegrationConfig {
        shape_model: FootprintShapeModel::Gaussian,
        patch_half_width_pts: [4, 4],
        max_centroid_shift_ppm: [0.005, 0.05],
        ref_plane_indices: Some(vec![0]),
        fit_local_baseline: false,
        duplicate_tol: 1.0e-7,
        account_spatial_noise_corr: false,
    };
    let ds = integrate_relax_planes(
        "overlap_r1",
        RelaxExperimentKind::R1,
        RelaxSchedule::VariableDelaysSec {
            delays_sec: delays.clone(),
        },
        &planes,
        &seeds,
        &cfg,
    )
    .expect("integrate overlapping pair");

    // Verify both peaks belong to the same overlap cluster and recover distinct decay ratios
    assert_eq!(
        ds.traces[0].footprint.overlap_cluster_id,
        ds.traces[1].footprint.overlap_cluster_id
    );
    let mut overlap_max_ratio_err = 0.0_f64;
    for (r, &rate) in rates.iter().enumerate() {
        let a0 = ds.traces[r].intensities[0];
        let a3 = ds.traces[r].intensities[3];
        let obs_ratio = a3 / a0;
        let exp_ratio = (-rate * delays[3]).exp();
        let err = (obs_ratio - exp_ratio).abs();
        overlap_max_ratio_err = overlap_max_ratio_err.max(err);
        assert!(
            err < 0.015,
            "overlapping peak {r} ratio {obs_ratio:.4} vs expected {exp_ratio:.4}"
        );
    }

    // TSV & NEF round-trip lossless check
    let tsv_txt = write_relax_series_tsv_str(&ds);
    let from_tsv = parse_relax_series_tsv_str(&tsv_txt).expect("parse relax TSV");
    assert_eq!(from_tsv.name, ds.name);
    assert_eq!(from_tsv.experiment_kind, ds.experiment_kind);
    assert_eq!(from_tsv.traces.len(), ds.traces.len());
    assert_eq!(
        from_tsv.plane_is_duplicate,
        vec![false, true, false, false, true]
    );

    let nef_txt = write_relax_series_nef_str(&ds);
    let from_nef = parse_relax_series_nef_str(&nef_txt).expect("parse relax NEF");
    assert_eq!(from_nef.experiment_kind, ds.experiment_kind);
    assert_eq!(from_nef.traces.len(), ds.traces.len());
    let mut max_tsv_err = 0.0_f64;
    let mut max_nef_err = 0.0_f64;
    for p in 0..delays.len() {
        let diff_tsv = (from_tsv.traces[0].intensities[p] - ds.traces[0].intensities[p]).abs();
        let diff_nef = (from_nef.traces[0].intensities[p] - ds.traces[0].intensities[p]).abs();
        max_tsv_err = max_tsv_err.max(diff_tsv);
        max_nef_err = max_nef_err.max(diff_nef);
        assert!(diff_tsv < 1.0e-3 && diff_nef < 1.0e-3);
    }

    // Also verify PseudoVoigt footprint (`eta_h`, `eta_x`, and `overlap_cluster_id > 0`) lossless TSV & NEF round-trip!
    let mut pv_seeds = seeds.clone();
    pv_seeds.push(TargetPeakSeed {
        seq_id: 40,
        res_name: "ALA".to_string(),
        atom_h: "H".to_string(),
        atom_x: "N".to_string(),
        delta_h_ppm: 7.900,
        delta_x_ppm: 116.50,
    });
    let mut pv_planes = Vec::new();
    for (p_idx, &t) in delays.iter().enumerate() {
        let mut data = vec![0.0f32; ny * nx];
        for iy in 0..ny {
            let py = ax_y.index_to_ppm(iy as f64);
            for ix in 0..nx {
                let px = ax_x.index_to_ppm(ix as f64);
                let v = 110.0
                    * (-1.8 * t).exp()
                    * footprint_eval_2d(px, py, 8.400, 119.00, sh, sn, 0.35, 0.45)
                    + 85.0
                        * (-3.6 * t).exp()
                        * footprint_eval_2d(px, py, 8.432, 119.22, sh, sn, 0.35, 0.45)
                    + 95.0
                        * (-2.2 * t).exp()
                        * footprint_eval_2d(px, py, 7.900, 116.50, sh, sn, 0.55, 0.25);
                data[iy * nx + ix] = v as f32;
            }
        }
        pv_planes.push(Spectrum2D {
            name: format!("pv_p{p_idx}"),
            axis_x: ax_x.clone(),
            axis_y: ax_y.clone(),
            data,
        });
    }
    let pv_ds = integrate_relax_planes(
        "pv_r1",
        RelaxExperimentKind::R1,
        RelaxSchedule::VariableDelaysSec {
            delays_sec: delays.clone(),
        },
        &pv_planes,
        &pv_seeds,
        &RelaxIntegrationConfig {
            shape_model: FootprintShapeModel::PseudoVoigt,
            ref_plane_indices: Some(vec![0]),
            ..RelaxIntegrationConfig::default()
        },
    )
    .expect("integrate PseudoVoigt stack");
    assert!(pv_ds.traces[2].footprint.eta_h > 0.10);
    assert!(pv_ds.traces[2].footprint.eta_x > 0.05);
    assert_ne!(
        pv_ds.traces[0].footprint.overlap_cluster_id,
        pv_ds.traces[2].footprint.overlap_cluster_id
    );

    let pv_from_tsv =
        parse_relax_series_tsv_str(&write_relax_series_tsv_str(&pv_ds)).expect("parse pv TSV");
    let pv_from_nef =
        parse_relax_series_nef_str(&write_relax_series_nef_str(&pv_ds)).expect("parse pv NEF");
    let mut pv_eta_max_err = 0.0_f64;
    let mut pv_vol_max_err = 0.0_f64;
    for idx in 0..3 {
        let orig = &pv_ds.traces[idx].footprint;
        for restored in [
            &pv_from_tsv.traces[idx].footprint,
            &pv_from_nef.traces[idx].footprint,
        ] {
            let eh_err = (restored.eta_h - orig.eta_h).abs();
            let ex_err = (restored.eta_x - orig.eta_x).abs();
            let vol_err =
                (restored.unit_volume_factor_ppm2() - orig.unit_volume_factor_ppm2()).abs();
            pv_eta_max_err = pv_eta_max_err.max(eh_err).max(ex_err);
            pv_vol_max_err = pv_vol_max_err.max(vol_err);
            assert!(eh_err < 2.0e-5);
            assert!(ex_err < 2.0e-5);
            assert_eq!(restored.overlap_cluster_id, orig.overlap_cluster_id);
            assert!(vol_err < 1.0e-5);
        }
    }

    emit_w3_metrics(
        "t13_1_overlap_roundtrip",
        json!({
            "record_id": "t13_1_overlap_roundtrip",
            "task": "T13.1",
            "overlap_cluster_shared": ds.traces[0].footprint.overlap_cluster_id == ds.traces[1].footprint.overlap_cluster_id,
            "overlap_max_decay_ratio_err": overlap_max_ratio_err,
            "gaussian_tsv_max_abs_err": max_tsv_err,
            "gaussian_nef_max_abs_err": max_nef_err,
            "pseudovoigt_eta_max_abs_err": pv_eta_max_err,
            "pseudovoigt_vol_factor_max_abs_err": pv_vol_max_err,
            "pseudovoigt_clusters_preserved": true
        }),
    );
}
