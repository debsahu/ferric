//! Batched WebGPU (`wgpu` + WGSL) Relaxation Dispersion (`CPMG` & Off-Resonance `R1rho`) forward
//! evaluator and multi-field global `(p_B, k_ex)` profiler (`T13.5`).
//!
//! Provenance: Clean-room GPU compute implementation from peer-reviewed biomolecular NMR literature:
//! - Luz, Z. & Meiboom, S. (*J. Chem. Phys.* 1963, 39, 366–370 — `LM63` fast-exchange dispersion).
//! - Carver, J. P. & Richards, R. E. (*J. Magn. Reson.* 1972, 6, 89–105 — `CR72` 2-site dispersion).
//! - Baldwin, A. J. (*J. Magn. Reson.* 2014, 244, 114–124 — `Baldwin2014` exact closed-form 2-site
//!   `R2,eff` across all timescales, arbitrary `p_B`, and `Delta_R2 = R2B - R2A != 0`).
//! - Trott, O. & Palmer, A. G. III (*J. Magn. Reson.* 2002, 154, 157–160 — asymmetric off-resonance
//!   `R1rho(omega_eff, theta)`).

use bytemuck::{Pod, Zeroable};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::cpmg::{fit_cpmg_global_with_grid, CpmgFitConfig, CpmgGlobalFitResult};
use nmr_core::cpmg::{profile_residue_dw_cpu, CpmgGridCellResult, CpmgModel, CpmgSeriesDataset};
use wgpu::util::DeviceExt;

use crate::GpuEngine;

/// Maximum forward or grid-cell threads per GPU dispatch to guarantee TDR safety across all
/// platforms (`512` workgroups of `64` threads = `32,768` jobs).
const MAX_CPMG_JOBS_PER_DISPATCH: usize = 32_768;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CpmgGpuUniforms {
    n_jobs: u32,
    n_res: u32,
    n_pb: u32,
    n_kex: u32,
    model_code: u32,
    job_offset: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CpmgPointEvalGpu {
    model_code: u32,
    _pad: u32,
    r2a: f32,
    r2b: f32,
    p_b: f32,
    k_ex: f32,
    dw_rad: f32,
    t_relax_or_r1: f32,
    nu_or_w1: f32,
    om_a_offset: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ResidueSpanGpu {
    series_start: u32,
    series_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SeriesMetaGpu {
    larmor_mhz: f32,
    t_relax_s: f32,
    pt_start: u32,
    pt_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ObsPointGpu {
    nu_cpmg_hz: f32,
    r2_obs: f32,
    inv_var: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GridCellOutGpu {
    best_chi2: f32,
    best_dw_ppm: f32,
}

/// Model selector for batched WebGPU relaxation dispersion point evaluations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CpmgGpuEvalRequest {
    Flat {
        r20_s_inv: f64,
    },
    Lm63 {
        r20_s_inv: f64,
        p_b: f64,
        k_ex_s_inv: f64,
        dw_rad_s: f64,
        nu_cpmg_hz: f64,
    },
    Cr72 {
        r2a_s_inv: f64,
        r2b_s_inv: f64,
        p_b: f64,
        k_ex_s_inv: f64,
        dw_rad_s: f64,
        nu_cpmg_hz: f64,
    },
    Baldwin2014 {
        r2a_s_inv: f64,
        r2b_s_inv: f64,
        p_b: f64,
        k_ex_s_inv: f64,
        dw_rad_s: f64,
        t_relax_s: f64,
        nu_cpmg_hz: f64,
    },
    TrottPalmerR1Rho {
        r1_s_inv: f64,
        r20_s_inv: f64,
        p_b: f64,
        k_ex_s_inv: f64,
        omega_a_offset_rad_s: f64,
        dw_ab_rad_s: f64,
        w1_rad_s: f64,
    },
}

impl CpmgGpuEvalRequest {
    fn to_gpu(self) -> CpmgPointEvalGpu {
        match self {
            Self::Flat { r20_s_inv } => CpmgPointEvalGpu {
                model_code: 0,
                _pad: 0,
                r2a: r20_s_inv as f32,
                r2b: r20_s_inv as f32,
                p_b: 0.0,
                k_ex: 0.0,
                dw_rad: 0.0,
                t_relax_or_r1: 0.04,
                nu_or_w1: 50.0,
                om_a_offset: 0.0,
            },
            Self::Lm63 {
                r20_s_inv,
                p_b,
                k_ex_s_inv,
                dw_rad_s,
                nu_cpmg_hz,
            } => CpmgPointEvalGpu {
                model_code: 1,
                _pad: 0,
                r2a: r20_s_inv as f32,
                r2b: r20_s_inv as f32,
                p_b: p_b as f32,
                k_ex: k_ex_s_inv as f32,
                dw_rad: dw_rad_s as f32,
                t_relax_or_r1: 0.04,
                nu_or_w1: nu_cpmg_hz as f32,
                om_a_offset: 0.0,
            },
            Self::Cr72 {
                r2a_s_inv,
                r2b_s_inv,
                p_b,
                k_ex_s_inv,
                dw_rad_s,
                nu_cpmg_hz,
            } => CpmgPointEvalGpu {
                model_code: 2,
                _pad: 0,
                r2a: r2a_s_inv as f32,
                r2b: r2b_s_inv as f32,
                p_b: p_b as f32,
                k_ex: k_ex_s_inv as f32,
                dw_rad: dw_rad_s as f32,
                t_relax_or_r1: 0.04,
                nu_or_w1: nu_cpmg_hz as f32,
                om_a_offset: 0.0,
            },
            Self::Baldwin2014 {
                r2a_s_inv,
                r2b_s_inv,
                p_b,
                k_ex_s_inv,
                dw_rad_s,
                t_relax_s,
                nu_cpmg_hz,
            } => CpmgPointEvalGpu {
                model_code: 3,
                _pad: 0,
                r2a: r2a_s_inv as f32,
                r2b: r2b_s_inv as f32,
                p_b: p_b as f32,
                k_ex: k_ex_s_inv as f32,
                dw_rad: dw_rad_s as f32,
                t_relax_or_r1: t_relax_s as f32,
                nu_or_w1: nu_cpmg_hz as f32,
                om_a_offset: 0.0,
            },
            Self::TrottPalmerR1Rho {
                r1_s_inv,
                r20_s_inv,
                p_b,
                k_ex_s_inv,
                omega_a_offset_rad_s,
                dw_ab_rad_s,
                w1_rad_s,
            } => CpmgPointEvalGpu {
                model_code: 4,
                _pad: 0,
                r2a: r20_s_inv as f32,
                r2b: r20_s_inv as f32,
                p_b: p_b as f32,
                k_ex: k_ex_s_inv as f32,
                dw_rad: dw_ab_rad_s as f32,
                t_relax_or_r1: r1_s_inv as f32,
                nu_or_w1: w1_rad_s as f32,
                om_a_offset: omega_a_offset_rad_s as f32,
            },
        }
    }
}

/// Pre-compiled WebGPU compute pipelines for `CPMG` / `R1rho` forward evaluation and `(p_B, k_ex)`
/// global multi-field grid profiling.
pub struct CpmgGpuPipeline {
    eval_pipeline: wgpu::ComputePipeline,
    grid_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl CpmgGpuPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-cpmg-fit-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/cpmg_fit.wgsl").into()),
        });

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

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nmr-cpmg-fit-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_rw(2),
                storage_ro(3),
                storage_ro(4),
                storage_ro(5),
                storage_ro(6),
                storage_ro(7),
                storage_rw(8),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-cpmg-fit-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let eval_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-cpmg-eval-pipe"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("eval_cpmg_batch"),
            compilation_options: Default::default(),
            cache: None,
        });
        let grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-cpmg-grid-pipe"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("grid_search_cpmg"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            eval_pipeline,
            grid_pipeline,
            bgl,
        }
    }
}

/// Validate requested GPU storage buffer capacity against `device.limits()`.
pub fn validate_cpmg_storage_bytes(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
) -> Result<(), String> {
    let lim = device.limits();
    if bytes > lim.max_buffer_size {
        return Err(format!(
            "GPU {label} buffer ({bytes} B) exceeds device max_buffer_size ({} B)",
            lim.max_buffer_size
        ));
    }
    if bytes > lim.max_storage_buffer_binding_size {
        return Err(format!(
            "GPU {label} buffer ({bytes} B) exceeds device max_storage_buffer_binding_size ({} B)",
            lim.max_storage_buffer_binding_size
        ));
    }
    Ok(())
}

#[inline]
fn aligned_copy_bytes(raw_bytes: u64) -> u64 {
    raw_bytes
        .max(wgpu::COPY_BUFFER_ALIGNMENT)
        .div_ceil(wgpu::COPY_BUFFER_ALIGNMENT)
        * wgpu::COPY_BUFFER_ALIGNMENT
}

struct DummyCpmgBuffers {
    pts: wgpu::Buffer,
    eval_out: wgpu::Buffer,
    pb: wgpu::Buffer,
    kex: wgpu::Buffer,
    spans: wgpu::Buffer,
    meta: wgpu::Buffer,
    obs: wgpu::Buffer,
    grid_out: wgpu::Buffer,
}

impl DummyCpmgBuffers {
    fn new(device: &wgpu::Device) -> Self {
        let mk_ro = |label: &str, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let mk_rw = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        Self {
            pts: mk_ro(
                "cpmg-dummy-pts",
                bytemuck::bytes_of(&CpmgPointEvalGpu::zeroed()),
            ),
            eval_out: mk_rw("cpmg-dummy-eval-out", 16),
            pb: mk_ro("cpmg-dummy-pb", bytemuck::cast_slice(&[0.0f32; 4])),
            kex: mk_ro("cpmg-dummy-kex", bytemuck::cast_slice(&[0.0f32; 4])),
            spans: mk_ro(
                "cpmg-dummy-spans",
                bytemuck::bytes_of(&ResidueSpanGpu::zeroed()),
            ),
            meta: mk_ro(
                "cpmg-dummy-meta",
                bytemuck::bytes_of(&SeriesMetaGpu::zeroed()),
            ),
            obs: mk_ro("cpmg-dummy-obs", bytemuck::bytes_of(&ObsPointGpu::zeroed())),
            grid_out: mk_rw("cpmg-dummy-grid-out", 16),
        }
    }
}

/// Evaluate a batch of `CPMG` (`Flat`, `LM63`, `CR72`, `Baldwin2014`) and `TrottPalmerR1Rho`
/// points on WebGPU (`async`).
pub async fn evaluate_cpmg_batch_gpu_async(
    engine: &GpuEngine,
    requests: &[CpmgGpuEvalRequest],
) -> Result<Vec<f64>, String> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }

    let total_in_bytes = (requests.len() as u64)
        .checked_mul(std::mem::size_of::<CpmgPointEvalGpu>() as u64)
        .ok_or_else(|| "GPU CPMG eval buffer byte size overflow".to_string())?;
    validate_cpmg_storage_bytes(&engine.device, "cpmg_eval_points", total_in_bytes)?;

    let pipe = CpmgGpuPipeline::new(&engine.device);
    let dummies = DummyCpmgBuffers::new(&engine.device);
    let max_wg = (engine.device.limits().max_compute_workgroups_per_dimension as usize) * 64;
    let chunk_cap = MAX_CPMG_JOBS_PER_DISPATCH.min(max_wg.max(64));

    let mut out_all = Vec::with_capacity(requests.len());
    for chunk in requests.chunks(chunk_cap) {
        let n_jobs = chunk.len();
        let gpu_pts: Vec<CpmgPointEvalGpu> = chunk.iter().copied().map(|r| r.to_gpu()).collect();
        let out_bytes = aligned_copy_bytes((n_jobs * 4) as u64);
        validate_cpmg_storage_bytes(&engine.device, "cpmg_eval_out", out_bytes)?;

        let uniforms = CpmgGpuUniforms {
            n_jobs: n_jobs as u32,
            n_res: 0,
            n_pb: 0,
            n_kex: 0,
            model_code: 0,
            job_offset: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let u_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cpmg-eval-uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let pts_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cpmg-eval-pts"),
                contents: bytemuck::cast_slice(&gpu_pts),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cpmg-eval-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cpmg-eval-bg"),
            layout: &pipe.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dummies.pb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dummies.kex.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dummies.spans.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dummies.meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: dummies.obs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: dummies.grid_out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cpmg-eval-enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpmg-eval-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipe.eval_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups((n_jobs as u32).div_ceil(64), 1, 1);
        }
        engine.queue.submit(Some(encoder.finish()));

        let vals: Vec<f32> = engine.read_buffer_vec_async(&out_buf, n_jobs).await?;
        out_all.extend(vals.into_iter().map(|v| v as f64));
    }
    Ok(out_all)
}

/// Synchronous native wrapper around [`evaluate_cpmg_batch_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn evaluate_cpmg_batch_gpu(
    engine: &GpuEngine,
    requests: &[CpmgGpuEvalRequest],
) -> Result<Vec<f64>, String> {
    pollster::block_on(evaluate_cpmg_batch_gpu_async(engine, requests))
}

/// Profile all `(p_B, k_ex, residue)` cells in parallel on WebGPU (`async`) using variable
/// projection over per-field `R20` and golden-section optimization over `|dw_ppm|`.
pub async fn profile_cpmg_grid_with_pipeline_async(
    engine: &GpuEngine,
    pipe: &CpmgGpuPipeline,
    res_groups: &[Vec<&CpmgSeriesDataset>],
    pb_grid: &[f64],
    kex_grid: &[f64],
    model: CpmgModel,
) -> Result<Vec<CpmgGridCellResult>, String> {
    let n_res = res_groups.len();
    let n_pb = pb_grid.len();
    let n_kex = kex_grid.len();
    if n_res == 0 || n_pb == 0 || n_kex == 0 {
        return Ok(Vec::new());
    }

    if model == CpmgModel::Numerical2x2 || model == CpmgModel::Flat {
        let mut out = Vec::with_capacity(n_pb * n_kex * n_res);
        for &pb in pb_grid {
            for &kex in kex_grid {
                for r_series in res_groups {
                    let (c2, dw, _) = profile_residue_dw_cpu(r_series, model, pb, kex);
                    out.push(CpmgGridCellResult {
                        best_chi2: c2,
                        best_dw_ppm: dw,
                    });
                }
            }
        }
        return Ok(out);
    }

    let model_code: u32 = match model {
        CpmgModel::Lm63 => 1,
        CpmgModel::Cr72 => 2,
        CpmgModel::Baldwin2014 => 3,
        _ => 3,
    };

    let n_res_u32 =
        u32::try_from(n_res).map_err(|_| format!("CPMG residue count {n_res} exceeds u32::MAX"))?;
    let n_pb_u32 =
        u32::try_from(n_pb).map_err(|_| format!("CPMG p_B grid count {n_pb} exceeds u32::MAX"))?;
    let n_kex_u32 = u32::try_from(n_kex)
        .map_err(|_| format!("CPMG k_ex grid count {n_kex} exceeds u32::MAX"))?;
    let total_jobs = n_pb
        .checked_mul(n_kex)
        .and_then(|v| v.checked_mul(n_res))
        .ok_or_else(|| "CPMG grid total job count overflowed usize".to_string())?;
    let _ = u32::try_from(total_jobs)
        .map_err(|_| format!("CPMG grid total job count {total_jobs} exceeds u32::MAX"))?;

    let mut spans = Vec::with_capacity(n_res);
    let mut series_meta = Vec::new();
    let mut obs_points = Vec::new();

    for r_series in res_groups {
        let s_start = u32::try_from(series_meta.len())
            .map_err(|_| "CPMG series_meta index exceeds u32::MAX".to_string())?;
        let s_len = u32::try_from(r_series.len())
            .map_err(|_| "CPMG residue series count exceeds u32::MAX".to_string())?;
        for s in r_series {
            let p_start = u32::try_from(obs_points.len())
                .map_err(|_| "CPMG obs_points index exceeds u32::MAX".to_string())?;
            let n_pts = s
                .nu_cpmg_hz
                .len()
                .min(s.r2_eff_s_inv.len())
                .min(s.r2_eff_err.len());
            let pt_len = u32::try_from(n_pts)
                .map_err(|_| "CPMG series point count exceeds u32::MAX".to_string())?;
            for i in 0..n_pts {
                let sig = s.r2_eff_err[i].max(1e-3);
                obs_points.push(ObsPointGpu {
                    nu_cpmg_hz: s.nu_cpmg_hz[i] as f32,
                    r2_obs: s.r2_eff_s_inv[i] as f32,
                    inv_var: (1.0 / (sig * sig)) as f32,
                    _pad: 0.0,
                });
            }
            series_meta.push(SeriesMetaGpu {
                larmor_mhz: s.larmor_mhz as f32,
                t_relax_s: s.t_relax_s as f32,
                pt_start: p_start,
                pt_len,
            });
        }
        spans.push(ResidueSpanGpu {
            series_start: s_start,
            series_len: s_len,
        });
    }

    let pb_f32: Vec<f32> = pb_grid.iter().map(|&v| v as f32).collect();
    let kex_f32: Vec<f32> = kex_grid.iter().map(|&v| v as f32).collect();

    validate_cpmg_storage_bytes(
        &engine.device,
        "cpmg_spans",
        std::mem::size_of_val(spans.as_slice()) as u64,
    )?;
    validate_cpmg_storage_bytes(
        &engine.device,
        "cpmg_series_meta",
        std::mem::size_of_val(series_meta.as_slice()) as u64,
    )?;
    validate_cpmg_storage_bytes(
        &engine.device,
        "cpmg_obs_points",
        std::mem::size_of_val(obs_points.as_slice()) as u64,
    )?;
    validate_cpmg_storage_bytes(
        &engine.device,
        "cpmg_pb_grid",
        std::mem::size_of_val(pb_f32.as_slice()) as u64,
    )?;
    validate_cpmg_storage_bytes(
        &engine.device,
        "cpmg_kex_grid",
        std::mem::size_of_val(kex_f32.as_slice()) as u64,
    )?;

    let dummies = DummyCpmgBuffers::new(&engine.device);
    let spans_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cpmg-grid-spans"),
            contents: bytemuck::cast_slice(&spans),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let meta_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cpmg-grid-meta"),
            contents: bytemuck::cast_slice(&series_meta),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let obs_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cpmg-grid-obs"),
            contents: bytemuck::cast_slice(&obs_points),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let pb_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cpmg-grid-pb"),
            contents: bytemuck::cast_slice(&pb_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let kex_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cpmg-grid-kex"),
            contents: bytemuck::cast_slice(&kex_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let max_wg =
        (engine.device.limits().max_compute_workgroups_per_dimension as usize).saturating_mul(64);
    let max_jobs = MAX_CPMG_JOBS_PER_DISPATCH.min(max_wg.max(64));

    let mut all_cells = Vec::with_capacity(total_jobs);
    let mut job_start = 0usize;
    while job_start < total_jobs {
        let n_jobs = (total_jobs - job_start).min(max_jobs);
        let out_bytes = aligned_copy_bytes((n_jobs * std::mem::size_of::<GridCellOutGpu>()) as u64);
        validate_cpmg_storage_bytes(&engine.device, "cpmg_grid_out", out_bytes)?;

        let uniforms = CpmgGpuUniforms {
            n_jobs: n_jobs as u32,
            n_res: n_res_u32,
            n_pb: n_pb_u32,
            n_kex: n_kex_u32,
            model_code,
            job_offset: job_start as u32,
            _pad1: 0,
            _pad2: 0,
        };
        let u_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cpmg-grid-uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let out_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cpmg-grid-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cpmg-grid-bg"),
            layout: &pipe.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummies.pts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummies.eval_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: kex_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: spans_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: obs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cpmg-grid-enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cpmg-grid-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipe.grid_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups((n_jobs as u32).div_ceil(64), 1, 1);
        }
        engine.queue.submit(Some(encoder.finish()));

        let cells_gpu: Vec<GridCellOutGpu> = engine.read_buffer_vec_async(&out_buf, n_jobs).await?;
        for c in cells_gpu {
            all_cells.push(CpmgGridCellResult {
                best_chi2: c.best_chi2 as f64,
                best_dw_ppm: c.best_dw_ppm as f64,
            });
        }
        job_start += n_jobs;
    }
    Ok(all_cells)
}

/// Multi-field (`e.g. 600 + 800 MHz`) global `(p_B, k_ex)` + per-residue `|dw_ppm|` `CPMG`
/// relaxation dispersion solver accelerated on WebGPU (`T13.5`).
#[cfg(not(target_arch = "wasm32"))]
pub fn fit_cpmg_global_gpu(
    engine: &GpuEngine,
    series: &[CpmgSeriesDataset],
    config: &CpmgFitConfig,
) -> Result<CpmgGlobalFitResult, String> {
    let pipe = CpmgGpuPipeline::new(&engine.device);
    fit_cpmg_global_with_grid(series, config, |res_groups, pb_grid, kex_grid, model| {
        pollster::block_on(profile_cpmg_grid_with_pipeline_async(
            engine, &pipe, res_groups, pb_grid, kex_grid, model,
        ))
    })
}
