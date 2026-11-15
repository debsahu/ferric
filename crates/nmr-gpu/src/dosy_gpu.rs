//! Massively parallel WebGPU (`wgpu` + WGSL) compute engine for pseudo-2D `DOSY`
//! mono- and bi-exponential Stejskal–Tanner Levenberg–Marquardt fitting (`T14.4`).
//!
//! Provenance: Clean-room GPU compute implementation derived strictly from peer-reviewed
//! pulsed-field-gradient NMR and nonlinear least-squares literature:
//! - Stejskal, E. O. & Tanner, J. E. (*J. Chem. Phys.* 1965, 42, 288–292 — pulsed-gradient
//!   spin-echo diffusion attenuation).
//! - Wu, D. H., Chen, A. D. & Johnson, C. S. Jr. (*J. Magn. Reson. A* 1995, 115, 260–264 —
//!   bipolar gradient pulse pair LED `BPP-LED` attenuation).
//! - Golub, G. H. & Pereyra, V. (*SIAM J. Numer. Anal.* 1973, 10, 413–432 — Variable Projection
//!   separable nonlinear least-squares initialization).
//! - Marquardt, D. W. (*J. Soc. Ind. Appl. Math.* 1963, 11, 431–441 — damped Gauss–Newton
//!   nonlinear least squares).

use bytemuck::{Pod, Zeroable};
use nmr_core::dosy::{DosyFitConfig, ModelSelectionCriterion, DOSY_VARPRO_GRID_K12};
use wgpu::util::DeviceExt;

use crate::GpuEngine;

/// Maximum number of frequency columns / peaks fitted per single WebGPU compute dispatch
/// so ultra-wide 1D spectral grids (`64k`–`256k` points) never trip OS GPU TDR watchdogs.
pub const MAX_DOSY_COLUMNS_PER_DISPATCH: usize = 16_384;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DosyGpuParams {
    n_cols: u32,
    m_len: u32,
    max_components: u32,
    criterion_is_aic: u32,
    max_lm_iters: u32,
    use_known_sigma: u32,
    b_scale: f32,
    u_min: f32,
    u_max: f32,
    min_du_seed: f32,
    min_component_fraction: f32,
    min_d_ratio: f32,
    known_noise_sigma: f32,
    min_signal_floor: f32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DosyColumnGpuRaw {
    selected_k: u32,
    is_bi_resolved: u32,
    mono_iters: u32,
    bi_iters: u32,
    mono_i0: f32,
    mono_d: f32,
    mono_rss: f32,
    mono_bic: f32,
    bi_i0_1: f32,
    bi_d_1: f32,
    bi_i0_2: f32,
    bi_d_2: f32,
    bi_rss: f32,
    bi_bic: f32,
    mono_aic: f32,
    bi_aic: f32,
}

/// Per-column / per-peak result returned by the WebGPU `DOSY` compute engine.
#[derive(Clone, Debug, PartialEq)]
pub struct DosyGpuColumnFit {
    /// Selected exponential component order (`0` = below `min_signal_floor`, `1` = mono, `2` = bi).
    pub selected_k: usize,
    /// Whether the bi-exponential fit passed physical amplitude-share and diffusivity-ratio guards.
    pub is_bi_resolved: bool,
    /// Mono-exponential ($K = 1$) pre-exponential amplitude $I_0$.
    pub mono_i0: f64,
    /// Mono-exponential ($K = 1$) diffusion coefficient $D$ (`m^2 / s`).
    pub mono_d_m2_s: f64,
    /// Mono-exponential residual sum of squares.
    pub mono_rss: f64,
    /// Mono-exponential `AIC` and `BIC`.
    pub mono_aic: f64,
    pub mono_bic: f64,
    /// Bi-exponential ($K = 2$) sorted slower component `(I0_1, D_1)`.
    pub bi_i0_1: f64,
    pub bi_d_1_m2_s: f64,
    /// Bi-exponential ($K = 2$) sorted faster component `(I0_2, D_2)` with `D_1 <= D_2`.
    pub bi_i0_2: f64,
    pub bi_d_2_m2_s: f64,
    /// Bi-exponential residual sum of squares.
    pub bi_rss: f64,
    /// Bi-exponential `AIC` and `BIC`.
    pub bi_aic: f64,
    pub bi_bic: f64,
    /// LM iteration counts executed on the winning mono- and bi-exponential seeds.
    pub mono_iters: usize,
    pub bi_iters: usize,
}

#[derive(Clone, Copy, Debug)]
struct DosyDispatchByteSizes {
    out_bytes_aligned: u64,
}

/// Validate all 7 WebGPU `DOSY` buffer sizes and workgroup counts against `wgpu::Limits` before
/// allocating any GPU memory.
pub fn validate_dosy_gpu_dispatch_limits(
    limits: &wgpu::Limits,
    n_cols_dispatch: usize,
    m_len: usize,
) -> Result<(), String> {
    let _ = compute_and_validate_dosy_sizes(limits, n_cols_dispatch, m_len)?;
    Ok(())
}

fn compute_and_validate_dosy_sizes(
    limits: &wgpu::Limits,
    n_cols_dispatch: usize,
    m_len: usize,
) -> Result<DosyDispatchByteSizes, String> {
    if n_cols_dispatch > u32::MAX as usize || m_len > u32::MAX as usize {
        return Err("GPU DOSY dimensions exceed 32-bit WGSL u32 index limit".to_string());
    }
    let n_wg = n_cols_dispatch.div_ceil(64);
    if n_wg > limits.max_compute_workgroups_per_dimension as usize {
        return Err(format!(
            "GPU DOSY dispatch workgroups ({n_wg}) exceed device max_compute_workgroups_per_dimension ({})",
            limits.max_compute_workgroups_per_dimension
        ));
    }

    let check_storage = |name: &str, bytes: u64| -> Result<(), String> {
        if bytes > limits.max_buffer_size {
            return Err(format!(
                "GPU DOSY {name} ({bytes} B) exceeds device max_buffer_size ({} B)",
                limits.max_buffer_size
            ));
        }
        if bytes > limits.max_storage_buffer_binding_size {
            return Err(format!(
                "GPU DOSY {name} ({bytes} B) exceeds device max_storage_buffer_binding_size ({} B)",
                limits.max_storage_buffer_binding_size
            ));
        }
        Ok(())
    };

    let params_bytes = std::mem::size_of::<DosyGpuParams>() as u64;
    if params_bytes > limits.max_buffer_size
        || params_bytes > limits.max_uniform_buffer_binding_size
    {
        return Err("GPU DOSY params uniform buffer exceeds device uniform limits".to_string());
    }

    let b_bytes = (m_len as u64)
        .checked_mul(4)
        .ok_or("b_tilde byte overflow")?
        .max(4);
    check_storage("b_tilde", b_bytes)?;

    let q_len = DOSY_VARPRO_GRID_K12;
    let u_bytes = (q_len as u64) * 4;
    check_storage("u_grid", u_bytes)?;

    let e_elems = q_len.checked_mul(m_len).ok_or("e_basis element overflow")?;
    let e_bytes = (e_elems as u64)
        .checked_mul(4)
        .ok_or("e_basis byte overflow")?
        .max(4);
    check_storage("e_basis", e_bytes)?;

    let gram_bytes = ((q_len * q_len) as u64) * 4;
    check_storage("gram_mat", gram_bytes)?;

    let y_elems = n_cols_dispatch
        .checked_mul(m_len)
        .ok_or("y_matrix element count overflow")?;
    if y_elems > u32::MAX as usize {
        return Err("GPU DOSY y_matrix index exceeds u32::MAX".to_string());
    }
    let y_bytes = (y_elems as u64)
        .checked_mul(4)
        .ok_or("y_matrix byte size overflow")?
        .max(4);
    check_storage("y_matrix", y_bytes)?;

    let out_raw = (n_cols_dispatch as u64)
        .checked_mul(std::mem::size_of::<DosyColumnGpuRaw>() as u64)
        .ok_or("out_fits byte size overflow")?;
    let out_bytes_aligned = out_raw
        .div_ceil(wgpu::COPY_BUFFER_ALIGNMENT)
        .max(1)
        .checked_mul(wgpu::COPY_BUFFER_ALIGNMENT)
        .ok_or("out_fits aligned byte size overflow")?;
    check_storage("out_fits", out_bytes_aligned)?;

    Ok(DosyDispatchByteSizes { out_bytes_aligned })
}

struct CompiledDosyPipeline {
    bgl: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl CompiledDosyPipeline {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-dosy-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dosy.wgsl").into()),
        });

        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nmr-dosy-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                storage_ro(5),
                storage_rw(6),
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-dosy-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-dosy-fit-pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("fit_dosy_columns"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { bgl, pipeline }
    }
}

/// Asynchronously fit mono- and bi-exponential Stejskal–Tanner decay curves across `columns`
/// (`columns[col]` has length `b_s_m2.len()`) on WebGPU (`wasm32` + native compatible).
pub async fn fit_dosy_columns_gpu_async(
    engine: &GpuEngine,
    b_s_m2: &[f64],
    columns: &[Vec<f64>],
    cfg: &DosyFitConfig,
    min_signal_floor: f64,
) -> Result<Vec<DosyGpuColumnFit>, String> {
    let m_len = b_s_m2.len();
    if m_len < 3 {
        return Err(format!(
            "GPU DOSY fit requires >= 3 gradient points, got {m_len}"
        ));
    }
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    for (ci, col) in columns.iter().enumerate() {
        if col.len() != m_len {
            return Err(format!(
                "column {ci} intensity length ({}) != b_s_m2 length ({m_len})",
                col.len()
            ));
        }
        if col.iter().any(|v| !v.is_finite()) {
            return Err(format!("column {ci} contains non-finite intensity value"));
        }
    }
    if !cfg.d_min_m2_s.is_finite()
        || !cfg.d_max_m2_s.is_finite()
        || cfg.d_min_m2_s <= 0.0
        || cfg.d_max_m2_s <= cfg.d_min_m2_s
    {
        return Err(format!(
            "invalid DosyFitConfig diffusivity bounds [{}, {}]",
            cfg.d_min_m2_s, cfg.d_max_m2_s
        ));
    }

    let mut b_scale = 0.0_f64;
    for (bi, &bv) in b_s_m2.iter().enumerate() {
        if !bv.is_finite() || bv < 0.0 {
            return Err(format!("invalid b-value at index {bi}: {bv}"));
        }
        b_scale = b_scale.max(bv);
    }
    if b_scale <= 0.0 {
        return Err("all b-values are zero; cannot fit DOSY on GPU".to_string());
    }

    let lim = engine.device.limits();
    let chunk_cap = MAX_DOSY_COLUMNS_PER_DISPATCH.max(1);
    let _ = compute_and_validate_dosy_sizes(&lim, columns.len().min(chunk_cap), m_len)?;

    let b_tilde_f64: Vec<f64> = b_s_m2.iter().map(|&b| b / b_scale).collect();
    let b_tilde_f32: Vec<f32> = b_tilde_f64.iter().map(|&b| b as f32).collect();

    let u_min = (cfg.d_min_m2_s * b_scale).ln();
    let u_max = (cfg.d_max_m2_s * b_scale).ln();
    let q_len = DOSY_VARPRO_GRID_K12;
    let mut u_grid_f64 = vec![0.0_f64; q_len];
    let mut u_grid_f32 = vec![0.0_f32; q_len];
    let mut e_basis_f64 = vec![0.0_f64; q_len * m_len];
    let mut e_basis_f32 = vec![0.0_f32; q_len * m_len];

    for q in 0..q_len {
        let uq = u_min + (q as f64) * (u_max - u_min) / ((q_len - 1) as f64);
        u_grid_f64[q] = uq;
        u_grid_f32[q] = uq as f32;
        let dq = uq.exp();
        for i in 0..m_len {
            let ev = (-b_tilde_f64[i] * dq).exp();
            e_basis_f64[q * m_len + i] = ev;
            e_basis_f32[q * m_len + i] = ev as f32;
        }
    }

    let mut gram_f32 = vec![0.0_f32; q_len * q_len];
    for p in 0..q_len {
        for q in p..q_len {
            let mut s = 0.0_f64;
            for i in 0..m_len {
                s += e_basis_f64[p * m_len + i] * e_basis_f64[q * m_len + i];
            }
            let sf = s as f32;
            gram_f32[p * q_len + q] = sf;
            gram_f32[q * q_len + p] = sf;
        }
    }

    let dosy_pipe = CompiledDosyPipeline::new(&engine.device);

    let b_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-dosy-b-tilde"),
            contents: bytemuck::cast_slice(&b_tilde_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let u_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-dosy-u-grid"),
            contents: bytemuck::cast_slice(&u_grid_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let e_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-dosy-e-basis"),
            contents: bytemuck::cast_slice(&e_basis_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let gram_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-dosy-gram"),
            contents: bytemuck::cast_slice(&gram_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let max_comp_gpu = cfg.max_components.clamp(1, 2) as u32;
    let criterion_is_aic = u32::from(cfg.criterion == ModelSelectionCriterion::Aic);
    let (use_known_sigma, known_sigma_f32) = match cfg.known_noise_sigma {
        Some(s) if s.is_finite() && s > 0.0 => (1u32, s as f32),
        _ => (0u32, 0.0f32),
    };

    let mut outputs = Vec::with_capacity(columns.len());

    for chunk in columns.chunks(chunk_cap) {
        let c_len = chunk.len();
        let sizes = compute_and_validate_dosy_sizes(&lim, c_len, m_len)?;

        let mut y_flat = Vec::with_capacity(c_len * m_len);
        for col in chunk {
            for &v in col {
                y_flat.push(v as f32);
            }
        }

        let params = DosyGpuParams {
            n_cols: c_len as u32,
            m_len: m_len as u32,
            max_components: max_comp_gpu,
            criterion_is_aic,
            max_lm_iters: cfg.max_lm_iters.max(10) as u32,
            use_known_sigma,
            b_scale: b_scale as f32,
            u_min: u_min as f32,
            u_max: u_max as f32,
            min_du_seed: (1.25_f64.ln()) as f32,
            min_component_fraction: cfg.min_component_fraction as f32,
            min_d_ratio: cfg.min_d_ratio as f32,
            known_noise_sigma: known_sigma_f32,
            min_signal_floor: min_signal_floor.max(0.0) as f32,
            _pad0: 0,
            _pad1: 0,
        };

        let params_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nmr-dosy-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let y_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nmr-dosy-y-matrix"),
                contents: bytemuck::cast_slice(&y_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-dosy-out-fits"),
            size: sizes.out_bytes_aligned,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nmr-dosy-bg"),
            layout: &dosy_pipe.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: e_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gram_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nmr-dosy-enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nmr-dosy-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&dosy_pipe.pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            let n_wg = (c_len as u32).div_ceil(64);
            cpass.dispatch_workgroups(n_wg, 1, 1);
        }
        engine.queue.submit(Some(encoder.finish()));

        let raw_vec: Vec<DosyColumnGpuRaw> = engine.read_buffer_vec_async(&out_buf, c_len).await?;
        for r in raw_vec {
            outputs.push(DosyGpuColumnFit {
                selected_k: r.selected_k as usize,
                is_bi_resolved: r.is_bi_resolved != 0,
                mono_i0: r.mono_i0 as f64,
                mono_d_m2_s: r.mono_d as f64,
                mono_rss: r.mono_rss as f64,
                mono_aic: r.mono_aic as f64,
                mono_bic: r.mono_bic as f64,
                bi_i0_1: r.bi_i0_1 as f64,
                bi_d_1_m2_s: r.bi_d_1 as f64,
                bi_i0_2: r.bi_i0_2 as f64,
                bi_d_2_m2_s: r.bi_d_2 as f64,
                bi_rss: r.bi_rss as f64,
                bi_aic: r.bi_aic as f64,
                bi_bic: r.bi_bic as f64,
                mono_iters: r.mono_iters as usize,
                bi_iters: r.bi_iters as usize,
            });
        }
    }

    Ok(outputs)
}

/// Synchronous native wrapper around [`fit_dosy_columns_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn fit_dosy_columns_gpu(
    engine: &GpuEngine,
    b_s_m2: &[f64],
    columns: &[Vec<f64>],
    cfg: &DosyFitConfig,
    min_signal_floor: f64,
) -> Result<Vec<DosyGpuColumnFit>, String> {
    pollster::block_on(fit_dosy_columns_gpu_async(
        engine,
        b_s_m2,
        columns,
        cfg,
        min_signal_floor,
    ))
}

/// Convenience wrapper fitting a row-major pseudo-2D spectrum (`n_b` rows $\times$ `n_ppm` columns)
/// on WebGPU by transposing active columns into contiguous per-thread decay vectors.
#[cfg(not(target_arch = "wasm32"))]
pub fn fit_pseudo2d_matrix_gpu(
    engine: &GpuEngine,
    b_s_m2: &[f64],
    pseudo2d_row_major: &[f64],
    n_ppm: usize,
    cfg: &DosyFitConfig,
    min_signal_floor: f64,
) -> Result<Vec<DosyGpuColumnFit>, String> {
    let m_len = b_s_m2.len();
    if pseudo2d_row_major.len() != m_len * n_ppm {
        return Err(format!(
            "pseudo2d_row_major length ({}) != b_len ({m_len}) * n_ppm ({n_ppm})",
            pseudo2d_row_major.len()
        ));
    }
    let mut cols = vec![vec![0.0; m_len]; n_ppm];
    for bi in 0..m_len {
        let row = &pseudo2d_row_major[bi * n_ppm..(bi + 1) * n_ppm];
        for ci in 0..n_ppm {
            cols[ci][bi] = row[ci];
        }
    }
    fit_dosy_columns_gpu(engine, b_s_m2, &cols, cfg, min_signal_floor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nmr_core::dosy::{
        compute_b_values_s_m2, fit_stejskal_tanner_multi, DosyNucleus, DosySequenceType,
        GradientShape,
    };
    use nmr_core::rng::Rng;

    #[test]
    fn gpu_cpu_dosy_mono_and_bi_parity_under_1_percent() {
        let engine = GpuEngine::new_headless().expect("headless WebGPU adapter required");

        // 20-point BPP-LED gradient ramp (2.5 to 55 G/cm), Delta = 100 ms, delta = 2.4 ms, tau = 0.2 ms
        let grads: Vec<f64> = (0..20).map(|i| 2.5 + (i as f64) * (52.5 / 19.0)).collect();
        let b_vals = compute_b_values_s_m2(
            &grads,
            DosyNucleus::H1,
            DosySequenceType::BppLed,
            GradientShape::SineBell,
            0.100,
            0.0024,
            0.0002,
        )
        .expect("b-values");

        let d_slow = 2.0e-10;
        let d_fast = 8.5e-10;
        let mut rng = Rng::new(20261114);

        let mut cols = Vec::new();
        // 32 synthetic columns: pure slow, pure fast, mixed 50:50, mixed 35:65, and baseline noise
        for idx in 0..32usize {
            let mut y = Vec::with_capacity(b_vals.len());
            for &b in &b_vals {
                let clean = match idx % 5 {
                    0 => 100.0 * (-b * d_slow).exp(),
                    1 => 120.0 * (-b * d_fast).exp(),
                    2 => 60.0 * (-b * d_slow).exp() + 60.0 * (-b * d_fast).exp(),
                    3 => 42.0 * (-b * d_slow).exp() + 78.0 * (-b * d_fast).exp(),
                    _ => 0.0,
                };
                let noisy = clean + rng.normal(0.0, 0.25);
                y.push(noisy);
            }
            cols.push(y);
        }

        let cfg = DosyFitConfig {
            max_components: 2,
            bootstrap_samples: 0,
            ..DosyFitConfig::default()
        };

        let gpu_fits =
            fit_dosy_columns_gpu(&engine, &b_vals, &cols, &cfg, 5.0).expect("GPU DOSY fit");
        assert_eq!(gpu_fits.len(), cols.len());

        let mut max_mono_rel_diff = 0.0_f64;
        let mut max_bi_rel_diff = 0.0_f64;
        for (idx, g) in gpu_fits.iter().enumerate() {
            if idx % 5 == 4 {
                assert_eq!(g.selected_k, 0, "baseline column {idx} should be skipped");
                continue;
            }
            let cpu = fit_stejskal_tanner_multi(&b_vals, &cols[idx], &cfg).expect("CPU DOSY fit");
            assert_eq!(
                g.selected_k, cpu.selected_k,
                "BIC model order mismatch on column {idx}"
            );
            if idx % 5 <= 1 {
                assert_eq!(
                    g.selected_k, 1,
                    "pure single-species column {idx} must pick K=1"
                );
                let rel_d = (g.mono_d_m2_s - cpu.best_fit.components[0].d_m2_s).abs()
                    / cpu.best_fit.components[0].d_m2_s;
                assert!(
                    rel_d < 0.01,
                    "mono D GPU/CPU relative diff {rel_d:.4e} >= 1% on col {idx}"
                );
                max_mono_rel_diff = max_mono_rel_diff.max(rel_d);
            } else {
                assert_eq!(g.selected_k, 2, "mixed column {idx} must pick K=2");
                let rel_d1 = (g.bi_d_1_m2_s - cpu.best_fit.components[0].d_m2_s).abs()
                    / cpu.best_fit.components[0].d_m2_s;
                let rel_d2 = (g.bi_d_2_m2_s - cpu.best_fit.components[1].d_m2_s).abs()
                    / cpu.best_fit.components[1].d_m2_s;
                assert!(
                    rel_d1 < 0.01 && rel_d2 < 0.01,
                    "bi D GPU/CPU relative diff ({rel_d1:.4e}, {rel_d2:.4e}) >= 1% on col {idx}"
                );
                max_bi_rel_diff = max_bi_rel_diff.max(rel_d1.max(rel_d2));
            }
        }
        println!(
            "W2_METRIC_JSON: {{\"kind\":\"dosy_webgpu_parity\",\"adapter\":\"{}\",\"backend\":\"{}\",\"n_columns\":{},\"max_mono_rel_diff\":{:.6e},\"max_bi_rel_diff\":{:.6e},\"passed\":true}}",
            engine.adapter_name.replace('"', ""),
            engine.backend_name.replace('"', ""),
            cols.len(),
            max_mono_rel_diff,
            max_bi_rel_diff,
        );
    }

    #[test]
    fn gpu_dosy_works_under_strict_default_wgpu_limits() {
        let engine = GpuEngine::new_headless_with_limits(wgpu::Limits::default())
            .expect("default limits WebGPU adapter");
        let grads: Vec<f64> = (1..=16).map(|i| (i as f64) * 3.0).collect();
        let b_vals = compute_b_values_s_m2(
            &grads,
            DosyNucleus::H1,
            DosySequenceType::Ste,
            GradientShape::Rectangular,
            0.080,
            0.0020,
            0.0,
        )
        .expect("b-values");
        let col: Vec<f64> = b_vals
            .iter()
            .map(|&b| 50.0 * (-b * 5.0e-10).exp())
            .collect();
        let cfg = DosyFitConfig {
            max_components: 2,
            bootstrap_samples: 0,
            ..DosyFitConfig::default()
        };
        let res =
            fit_dosy_columns_gpu(&engine, &b_vals, &[col], &cfg, 1.0).expect("default limits fit");
        assert_eq!(res[0].selected_k, 1);
        assert!((res[0].mono_d_m2_s - 5.0e-10).abs() / 5.0e-10 < 1.0e-4);
    }

    #[test]
    fn gpu_pseudo2d_matrix_array_parity_and_limit_guards() {
        let engine = GpuEngine::new_headless().expect("headless WebGPU adapter");
        let grads: Vec<f64> = (0..18).map(|i| 2.0 + (i as f64) * (54.0 / 17.0)).collect();
        let b_vals = compute_b_values_s_m2(
            &grads,
            DosyNucleus::H1,
            DosySequenceType::BppLed,
            GradientShape::Rectangular,
            0.100,
            0.0020,
            0.0002,
        )
        .expect("b-values");

        let n_ppm = 128usize;
        let m_len = b_vals.len();
        let mut pseudo2d = vec![0.0_f64; m_len * n_ppm];
        let d1 = 2.0e-10_f64;
        let d2 = 8.5e-10_f64;

        for (bi, &b) in b_vals.iter().enumerate() {
            for ci in 0..n_ppm {
                let dx1 = ((ci as f64) - 30.0) / 3.0;
                let dx2 = ((ci as f64) - 64.0) / 3.0;
                let dx3 = ((ci as f64) - 96.0) / 3.0;
                let s1 = 90.0 * (-dx1 * dx1).exp() * (-b * d1).exp();
                let s2 = 50.0 * (-dx2 * dx2).exp() * ((-b * d1).exp() + (-b * d2).exp());
                let s3 = 85.0 * (-dx3 * dx3).exp() * (-b * d2).exp();
                pseudo2d[bi * n_ppm + ci] = s1 + s2 + s3;
            }
        }

        let cfg = DosyFitConfig {
            max_components: 2,
            bootstrap_samples: 0,
            ..DosyFitConfig::default()
        };
        let gpu = fit_pseudo2d_matrix_gpu(&engine, &b_vals, &pseudo2d, n_ppm, &cfg, 5.0)
            .expect("fit_pseudo2d_matrix_gpu");
        assert_eq!(gpu.len(), n_ppm);

        // Peak 1 at col 30 (pure slow D1)
        assert_eq!(gpu[30].selected_k, 1);
        assert!((gpu[30].mono_d_m2_s - d1).abs() / d1 < 0.005);

        // Peak 2 at col 64 (50:50 mixed D1 + D2)
        assert_eq!(gpu[64].selected_k, 2);
        assert!((gpu[64].bi_d_1_m2_s - d1).abs() / d1 < 0.005);
        assert!((gpu[64].bi_d_2_m2_s - d2).abs() / d2 < 0.005);

        // Peak 3 at col 96 (pure fast D2)
        assert_eq!(gpu[96].selected_k, 1);
        assert!((gpu[96].mono_d_m2_s - d2).abs() / d2 < 0.005);

        // Oversize dispatch validation must fail cleanly before allocating buffers
        let tiny_lim = wgpu::Limits::default();
        assert!(validate_dosy_gpu_dispatch_limits(&tiny_lim, 10_000_000, 64).is_err());
    }
}
