//! Batched WebGPU (`wgpu` + WGSL) exponential relaxation curve fitter ($R_1, R_2, R_{1\rho}$) and
//! parametric Monte Carlo bootstrap engine (`T13.2`, `docs/specs/16-dynamics-and-csp.md` §1.4–§1.5).
//!
//! Provenance: Clean-room WebGPU compute implementation strictly from peer-reviewed NMR and
//! numerical optimization literature (zero external GPL/academic tool code consulted; zero external
//! product names in identifiers):
//! - Kay, L. E., Torchia, D. A. & Bax, A. (*Biochemistry* **1989**, 28, 8972–8979).
//! - Farrow, N. A. et al. (*Biochemistry* **1994**, 33, 5984–6003 & *J. Biomol. NMR* **1995**, 6,
//!   153–162).
//! - Lipari, G. & Szabo, A. (*J. Am. Chem. Soc.* **1982**, 104, 4546–4559 & 4559–4570).
//! - Clore, G. M. et al. (*J. Am. Chem. Soc.* **1990**, 112, 4989–4991).
//! - Mandel, A. M., Akke, M. & Palmer, A. G. III (*J. Mol. Biol.* **1995**, 246, 144–163 —
//!   parametric Monte Carlo confidence intervals for relaxation rates).
//! - Woessner, D. E. (*J. Chem. Phys.* **1962**, 37, 647–654).
//! - d'Auvergne, E. J. & Gooley, P. R. (*J. Biomol. NMR* **2003**, 25, 25–39).
//! - Levenberg, K. (*Quart. Appl. Math.* **1944**, 2, 164–168); Marquardt, D. W. (*J. Soc. Indust.
//!   Appl. Math.* **1963**, 11, 431–441); Golub, G. H. & Pereyra, V. (*SIAM J. Numer. Anal.*
//!   **1973**, 10, 413–432).

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use bytemuck::{Pod, Zeroable};
use nmr_core::relax_fit::{
    fisher_crlb_two_param, DecayCurveFit, DecayCurveInput, ExpFitConfig, OffsetModelGate,
};
use wgpu::util::DeviceExt;

use crate::GpuEngine;

/// Maximum number of relaxation delay planes per curve supported in WGSL private registers.
pub const MAX_GPU_CURVE_POINTS: usize = 64;

/// TDR-safe maximum number of Monte Carlo curve fits executed in a single GPU compute dispatch.
pub const MAX_FITS_PER_DISPATCH: usize = 262_144;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RelaxMcParamsGpu {
    n_curves: u32,
    mc_reps_total: u32,
    rep_offset: u32,
    rep_chunk: u32,
    offset_gate: u32,
    seed_lo: u32,
    seed_hi: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CurveMetaGpu {
    pt_offset: u32,
    n_pts: u32,
    seq_id_u32: u32,
    r_tilde_max: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CurvePointGpu {
    u: f32,
    y: f32,
    s_norm: f32,
    w: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PointFitGpu {
    i0_t: f32,
    r_t: f32,
    c_t: f32,
    chi2: f32,
    use_3p: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct StrictMapState {
    result: Option<Result<(), wgpu::BufferAsyncError>>,
    waker: Option<Waker>,
}

struct StrictMapFuture<'a> {
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    device: &'a wgpu::Device,
    state: Arc<Mutex<StrictMapState>>,
    #[cfg(not(target_arch = "wasm32"))]
    submission_index: Option<wgpu::SubmissionIndex>,
    #[cfg(not(target_arch = "wasm32"))]
    timeout_polls: u16,
}

impl Future for StrictMapFuture<'_> {
    type Output = Result<(), String>;

    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        {
            let mut guard = self
                .state
                .lock()
                .map_err(|_| "GPU map state mutex poisoned".to_string())?;
            if let Some(res) = guard.result.take() {
                return Poll::Ready(res.map_err(|e| format!("GPU buffer map failed: {e}")));
            }
            guard.waker = Some(cx.waker().clone());
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let sub_idx = self.submission_index.clone();
            match self.device.poll(wgpu::PollType::Wait {
                submission_index: sub_idx,
                timeout: Some(std::time::Duration::from_millis(200)),
            }) {
                Ok(_) => {}
                Err(wgpu::PollError::Timeout) => {
                    self.timeout_polls = self.timeout_polls.saturating_add(1);
                    if self.timeout_polls >= 300 {
                        return Poll::Ready(
                            Err("GPU device poll timed out after 60 s".to_string()),
                        );
                    }
                }
                Err(e) => {
                    return Poll::Ready(Err(format!("GPU device poll failed: {e}")));
                }
            }
            let mut guard = self
                .state
                .lock()
                .map_err(|_| "GPU map state mutex poisoned".to_string())?;
            if let Some(res) = guard.result.take() {
                guard.waker = None;
                drop(guard);
                self.submission_index = None;
                return Poll::Ready(res.map_err(|e| format!("GPU buffer map failed: {e}")));
            }
            let waker = cx.waker().clone();
            drop(guard);
            waker.wake();
        }

        Poll::Pending
    }
}

async fn read_buffer_strict_async<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<T>, String> {
    let elem_size = std::mem::size_of::<T>();
    if elem_size == 0 {
        return Ok(vec![T::zeroed(); count]);
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let byte_len_usize = count
        .checked_mul(elem_size)
        .ok_or_else(|| "GPU readback byte length overflow".to_string())?;
    let byte_len = u64::try_from(byte_len_usize)
        .map_err(|_| "GPU readback byte length overflow".to_string())?;
    let aligned_bytes =
        byte_len.div_ceil(wgpu::COPY_BUFFER_ALIGNMENT) * wgpu::COPY_BUFFER_ALIGNMENT;

    let max_buf = device.limits().max_buffer_size;
    if aligned_bytes > max_buf {
        return Err(format!(
            "GPU staging buffer ({aligned_bytes} B) exceeds device max_buffer_size ({max_buf} B)"
        ));
    }
    if aligned_bytes > src.size() {
        return Err(format!(
            "GPU aligned readback ({aligned_bytes} B) exceeds source buffer size ({} B)",
            src.size()
        ));
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nmr-relax-staging-read"),
        size: aligned_bytes.max(wgpu::COPY_BUFFER_ALIGNMENT),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("nmr-relax-readback-enc"),
    });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, aligned_bytes);
    let sub_idx = queue.submit(Some(enc.finish()));
    #[cfg(target_arch = "wasm32")]
    let _ = sub_idx;

    let slice = staging.slice(0..aligned_bytes);
    let state = Arc::new(Mutex::new(StrictMapState {
        result: None,
        waker: None,
    }));
    let state_cb = Arc::clone(&state);
    slice.map_async(wgpu::MapMode::Read, move |res| {
        if let Ok(mut guard) = state_cb.lock() {
            guard.result = Some(res);
            if let Some(waker) = guard.waker.take() {
                drop(guard);
                waker.wake();
            }
        }
    });

    StrictMapFuture {
        device,
        state,
        #[cfg(not(target_arch = "wasm32"))]
        submission_index: Some(sub_idx),
        #[cfg(not(target_arch = "wasm32"))]
        timeout_polls: 0,
    }
    .await?;

    let mapped = slice
        .get_mapped_range()
        .map_err(|e| format!("GPU mapped range failed: {e}"))?;
    let out: Vec<T> = bytemuck::cast_slice(&mapped[..byte_len_usize]).to_vec();
    drop(mapped);
    staging.unmap();
    Ok(out)
}

/// Pre-compiled WebGPU compute pipelines (`fit_point_estimates` + `fit_mc_replicates`) for batched
/// exponential relaxation rate fitting and parametric Monte Carlo bootstrap.
pub struct RelaxGpuPipeline {
    point_pipeline: wgpu::ComputePipeline,
    mc_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl RelaxGpuPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-relax-mc-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/relax_mc.wgsl").into()),
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
            label: Some("nmr-relax-mc-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_rw(3),
                storage_rw(4),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-relax-mc-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let point_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-relax-point-pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("fit_point_estimates"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mc_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-relax-mc-pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("fit_mc_replicates"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            point_pipeline,
            mc_pipeline,
            bgl,
        }
    }
}

fn fisher_crlb_three_param_local(
    times_sec: &[f64],
    sigmas: &[f64],
    i0: f64,
    rate: f64,
) -> (f64, f64) {
    let mut f = [[0.0_f64; 3]; 3];
    for (&t, &sig) in times_sec.iter().zip(sigmas.iter()) {
        let w = 1.0 / (sig * sig).max(1.0e-30);
        let e = (-rate * t).exp();
        let j = [e, -i0 * t * e, 1.0];
        for r in 0..3 {
            for c in 0..3 {
                f[r][c] += w * j[r] * j[c];
            }
        }
    }
    let c00 = f[1][1] * f[2][2] - f[1][2] * f[1][2];
    let c01 = f[1][2] * f[0][2] - f[0][1] * f[2][2];
    let c02 = f[0][1] * f[1][2] - f[1][1] * f[0][2];
    let c11 = f[0][0] * f[2][2] - f[0][2] * f[0][2];
    let det = f[0][0] * c00 + f[0][1] * c01 + f[0][2] * c02;
    if det.abs() <= 1.0e-30 {
        return (0.0, 0.0);
    }
    ((c00 / det).max(0.0).sqrt(), (c11 / det).max(0.0).sqrt())
}

fn quantile_sorted_local(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = (p.clamp(0.0, 1.0)) * ((sorted.len() - 1) as f64);
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = idx - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn encode_offset_gate(gate: OffsetModelGate) -> u32 {
    match gate {
        OffsetModelGate::Disabled => 0,
        OffsetModelGate::Aic => 1,
        OffsetModelGate::Aicc => 2,
        OffsetModelGate::Bic => 3,
        OffsetModelGate::AlwaysThreeParam => 4,
    }
}

fn align_copy_bytes(raw_bytes: u64) -> u64 {
    raw_bytes
        .max(wgpu::COPY_BUFFER_ALIGNMENT)
        .div_ceil(wgpu::COPY_BUFFER_ALIGNMENT)
        * wgpu::COPY_BUFFER_ALIGNMENT
}

/// Fit a batch of exponential relaxation decay curves and run parametric Monte Carlo bootstrap
/// asynchronously on WebGPU using a reusable `RelaxGpuPipeline`.
pub async fn fit_decay_batch_with_pipeline_async(
    engine: &GpuEngine,
    relax_pipe: &RelaxGpuPipeline,
    curves: &[DecayCurveInput],
    cfg: &ExpFitConfig,
) -> Result<Vec<DecayCurveFit>, String> {
    let n_curves = curves.len();
    if n_curves == 0 {
        return Ok(Vec::new());
    }

    let mut metas = Vec::with_capacity(n_curves);
    let mut points = Vec::new();
    let mut scales = Vec::with_capacity(n_curves); // (t_scale, s_scale)

    for c in curves {
        let n_pts = c.times_sec.len();
        if n_pts < 2 || c.intensities.len() != n_pts || c.sigmas.len() != n_pts {
            return Err(format!(
                "residue {} decay curve requires >= 2 matching (t, I, sigma) points",
                c.seq_id
            ));
        }
        if n_pts > MAX_GPU_CURVE_POINTS {
            return Err(format!(
                "residue {} has {n_pts} points, exceeding MAX_GPU_CURVE_POINTS ({MAX_GPU_CURVE_POINTS})",
                c.seq_id
            ));
        }
        let t_scale = c
            .times_sec
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .max(1.0e-6);
        let s_scale = c
            .intensities
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1.0e-12);
        let r_tilde_max = (cfg.max_rate_s_inv * t_scale).max(1.0);

        let pt_offset = u32::try_from(points.len())
            .map_err(|_| "GPU relaxation point table offset overflow".to_string())?;
        for i in 0..n_pts {
            let u = (c.times_sec[i] / t_scale) as f32;
            let y = (c.intensities[i] / s_scale) as f32;
            let sn = ((c.sigmas[i] / s_scale).max(1.0e-9)) as f32;
            let w = 1.0_f32 / (sn * sn);
            points.push(CurvePointGpu {
                u,
                y,
                s_norm: sn,
                w,
            });
        }
        metas.push(CurveMetaGpu {
            pt_offset,
            n_pts: n_pts as u32,
            seq_id_u32: c.seq_id as u32,
            r_tilde_max: r_tilde_max as f32,
        });
        scales.push((t_scale, s_scale));
    }

    let lim = engine.device.limits();
    let max_storage = lim.max_storage_buffer_binding_size.min(lim.max_buffer_size);
    let max_uniform = lim.max_uniform_buffer_binding_size.min(lim.max_buffer_size);
    let max_threads_per_dispatch = (lim.max_compute_workgroups_per_dimension as usize)
        .saturating_mul(64)
        .max(64);

    let uniform_bytes = std::mem::size_of::<RelaxMcParamsGpu>() as u64;
    if uniform_bytes > max_uniform {
        return Err(format!(
            "RelaxMcParamsGpu ({uniform_bytes} B) exceeds device max_uniform_buffer_binding_size ({max_uniform} B)"
        ));
    }

    let metas_bytes = (metas.len() * std::mem::size_of::<CurveMetaGpu>()) as u64;
    let points_bytes = (points.len() * std::mem::size_of::<CurvePointGpu>()) as u64;
    let point_fits_bytes = align_copy_bytes((n_curves * std::mem::size_of::<PointFitGpu>()) as u64);

    for (label, sz) in [
        ("metas", metas_bytes),
        ("points", points_bytes),
        ("point_fits", point_fits_bytes),
    ] {
        if sz > max_storage {
            return Err(format!(
                "GPU relaxation buffer '{label}' ({sz} B) exceeds device max storage binding ({max_storage} B)"
            ));
        }
    }
    if n_curves > max_threads_per_dispatch {
        return Err(format!(
            "n_curves ({n_curves}) exceeds single-dispatch GPU workgroup thread limit ({max_threads_per_dispatch})"
        ));
    }

    // Determine TDR-safe & binding-limit-safe Monte Carlo replicate chunk size per dispatch
    let max_fits_by_storage = (max_storage / 8) as usize; // `vec2<f32>` = 8 bytes per replicate
    let max_chunk_fits = MAX_FITS_PER_DISPATCH
        .min(max_threads_per_dispatch)
        .min(max_fits_by_storage)
        .max(n_curves);
    let reps_per_chunk = if cfg.mc_replicates == 0 {
        1usize
    } else {
        (max_chunk_fits / n_curves).max(1).min(cfg.mc_replicates)
    };
    let mc_buf_elems = n_curves
        .checked_mul(reps_per_chunk)
        .ok_or_else(|| "GPU MC buffer element count overflow".to_string())?;
    let mc_buf_bytes = align_copy_bytes((mc_buf_elems * 8) as u64);
    if mc_buf_bytes > max_storage {
        return Err(format!(
            "GPU relaxation MC sample buffer ({mc_buf_bytes} B) exceeds device max storage binding ({max_storage} B)"
        ));
    }

    let metas_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-relax-metas"),
            contents: bytemuck::cast_slice(&metas),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let points_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-relax-points"),
            contents: bytemuck::cast_slice(&points),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let point_fits_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nmr-relax-point-fits"),
        size: point_fits_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mc_samples_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nmr-relax-mc-samples"),
        size: mc_buf_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let seed_lo = (cfg.seed & 0xFFFF_FFFF) as u32;
    let seed_hi = (cfg.seed >> 32) as u32;
    let gate_code = encode_offset_gate(cfg.offset_gate);

    // Fast path: when `mc_replicates == 0` or `mc_replicates <= reps_per_chunk`, fuse `fit_point_estimates`
    // and `fit_mc_replicates` into ONE command encoder submission!
    if cfg.mc_replicates <= reps_per_chunk {
        let params = RelaxMcParamsGpu {
            n_curves: n_curves as u32,
            mc_reps_total: cfg.mc_replicates as u32,
            rep_offset: 0,
            rep_chunk: cfg.mc_replicates.max(1) as u32,
            offset_gate: gate_code,
            seed_lo,
            seed_hi,
            _pad: 0,
        };
        let params_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nmr-relax-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nmr-relax-bg"),
            layout: &relax_pipe.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: metas_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: points_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: point_fits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mc_samples_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nmr-relax-fused-enc"),
            });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nmr-relax-point-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&relax_pipe.point_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups((n_curves as u32).div_ceil(64), 1, 1);
        }
        if cfg.mc_replicates > 0 {
            let total_mc_threads = (n_curves * cfg.mc_replicates) as u32;
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nmr-relax-mc-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&relax_pipe.mc_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(total_mc_threads.div_ceil(64), 1, 1);
        }
        engine.queue.submit(Some(enc.finish()));

        let point_fits: Vec<PointFitGpu> =
            read_buffer_strict_async(&engine.device, &engine.queue, &point_fits_buf, n_curves)
                .await?;
        let all_mc: Vec<[f32; 2]> = if cfg.mc_replicates > 0 {
            read_buffer_strict_async(
                &engine.device,
                &engine.queue,
                &mc_samples_buf,
                n_curves * cfg.mc_replicates,
            )
            .await?
        } else {
            Vec::new()
        };

        return Ok(assemble_gpu_fits(
            curves,
            &scales,
            &point_fits,
            &all_mc,
            cfg.mc_replicates,
            cfg.mc_replicates,
        ));
    }

    // Multi-chunk TDR-safe path when `n_curves * mc_replicates > MAX_FITS_PER_DISPATCH`
    let mut per_curve_mc: Vec<Vec<[f32; 2]>> = (0..n_curves)
        .map(|_| Vec::with_capacity(cfg.mc_replicates))
        .collect();
    let mut rep_offset = 0usize;
    let mut first_chunk = true;

    while rep_offset < cfg.mc_replicates {
        let cur_chunk = (cfg.mc_replicates - rep_offset).min(reps_per_chunk);
        let params = RelaxMcParamsGpu {
            n_curves: n_curves as u32,
            mc_reps_total: cfg.mc_replicates as u32,
            rep_offset: rep_offset as u32,
            rep_chunk: cur_chunk as u32,
            offset_gate: gate_code,
            seed_lo,
            seed_hi,
            _pad: 0,
        };
        let params_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nmr-relax-params-chunk"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nmr-relax-bg-chunk"),
            layout: &relax_pipe.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: metas_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: points_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: point_fits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mc_samples_buf.as_entire_binding(),
                },
            ],
        });

        let mut enc = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nmr-relax-chunk-enc"),
            });
        if first_chunk {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nmr-relax-point-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&relax_pipe.point_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups((n_curves as u32).div_ceil(64), 1, 1);
            first_chunk = false;
        }
        {
            let total_threads = (n_curves * cur_chunk) as u32;
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nmr-relax-mc-chunk-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&relax_pipe.mc_pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups(total_threads.div_ceil(64), 1, 1);
        }
        engine.queue.submit(Some(enc.finish()));

        let chunk_samples: Vec<[f32; 2]> = read_buffer_strict_async(
            &engine.device,
            &engine.queue,
            &mc_samples_buf,
            n_curves * cur_chunk,
        )
        .await?;
        for ci in 0..n_curves {
            let slice = &chunk_samples[ci * cur_chunk..(ci + 1) * cur_chunk];
            per_curve_mc[ci].extend_from_slice(slice);
        }
        rep_offset += cur_chunk;
    }

    let point_fits: Vec<PointFitGpu> =
        read_buffer_strict_async(&engine.device, &engine.queue, &point_fits_buf, n_curves).await?;
    let flat_mc: Vec<[f32; 2]> = per_curve_mc.into_iter().flatten().collect();

    Ok(assemble_gpu_fits(
        curves,
        &scales,
        &point_fits,
        &flat_mc,
        cfg.mc_replicates,
        cfg.mc_replicates,
    ))
}

fn assemble_gpu_fits(
    curves: &[DecayCurveInput],
    scales: &[(f64, f64)],
    point_fits: &[PointFitGpu],
    mc_flat: &[[f32; 2]],
    mc_replicates: usize,
    stride_reps: usize,
) -> Vec<DecayCurveFit> {
    let mut out = Vec::with_capacity(curves.len());
    for (ci, curve) in curves.iter().enumerate() {
        let (t_scale, s_scale) = scales[ci];
        let pf = point_fits[ci];
        let use_3p = pf.use_3p != 0;
        let i0 = (pf.i0_t as f64) * s_scale;
        let rate = (pf.r_t as f64) / t_scale;
        let offset_c = if use_3p {
            (pf.c_t as f64) * s_scale
        } else {
            0.0
        };

        // Re-evaluate exact f64 chi2 at GPU-converged parameters for exact AIC/BIC reporting
        let mut chi2 = 0.0_f64;
        for k in 0..curve.times_sec.len() {
            let pred = i0 * (-rate * curve.times_sec[k]).exp() + offset_c;
            let z = (curve.intensities[k] - pred) / curve.sigmas[k].max(1.0e-15);
            chi2 += z * z;
        }
        let n = curve.times_sec.len();
        let k_params = if use_3p { 3 } else { 2 };
        let dof = n.saturating_sub(k_params);
        let aic = chi2 + 2.0 * (k_params as f64);
        let bic = chi2 + (k_params as f64) * (n as f64).ln();

        let (sigma_i0_fisher, sigma_rate_fisher) = if use_3p {
            fisher_crlb_three_param_local(&curve.times_sec, &curve.sigmas, i0, rate)
        } else {
            fisher_crlb_two_param(&curve.times_sec, &curve.sigmas, i0, rate)
        };

        if mc_replicates == 0 {
            out.push(DecayCurveFit {
                seq_id: curve.seq_id,
                res_name: curve.res_name.clone(),
                used_three_param: use_3p,
                i0,
                rate,
                offset_c,
                sigma_i0_fisher,
                sigma_rate_fisher,
                sigma_i0_mc: sigma_i0_fisher,
                sigma_rate_mc: sigma_rate_fisher,
                rate_ci68: [rate - sigma_rate_fisher, rate + sigma_rate_fisher],
                rate_ci95: [
                    rate - 1.95996 * sigma_rate_fisher,
                    rate + 1.95996 * sigma_rate_fisher,
                ],
                chi2,
                dof,
                aic,
                bic,
            });
            continue;
        }

        let base = ci * stride_reps;
        let mut mc_r = Vec::with_capacity(mc_replicates);
        let mut sum_i0 = 0.0_f64;
        let mut sum_r = 0.0_f64;
        for rep in 0..mc_replicates {
            let s = mc_flat[base + rep];
            let mi0 = (s[0] as f64) * s_scale;
            let mr = (s[1] as f64) / t_scale;
            sum_i0 += mi0;
            sum_r += mr;
            mc_r.push(mr);
        }
        let m_f = mc_replicates as f64;
        let mean_i0 = sum_i0 / m_f;
        let mean_r = sum_r / m_f;
        let denom = (mc_replicates.saturating_sub(1)).max(1) as f64;

        let mut var_i0 = 0.0_f64;
        let mut var_r = 0.0_f64;
        for rep in 0..mc_replicates {
            let s = mc_flat[base + rep];
            let di = (s[0] as f64) * s_scale - mean_i0;
            let dr = (s[1] as f64) / t_scale - mean_r;
            var_i0 += di * di;
            var_r += dr * dr;
        }
        var_i0 /= denom;
        var_r /= denom;

        mc_r.sort_by(|a, b| a.total_cmp(b));
        let rate_ci68 = [
            quantile_sorted_local(&mc_r, 0.158_655),
            quantile_sorted_local(&mc_r, 0.841_345),
        ];
        let rate_ci95 = [
            quantile_sorted_local(&mc_r, 0.025),
            quantile_sorted_local(&mc_r, 0.975),
        ];

        out.push(DecayCurveFit {
            seq_id: curve.seq_id,
            res_name: curve.res_name.clone(),
            used_three_param: use_3p,
            i0,
            rate,
            offset_c,
            sigma_i0_fisher,
            sigma_rate_fisher,
            sigma_i0_mc: var_i0.sqrt(),
            sigma_rate_mc: var_r.sqrt(),
            rate_ci68,
            rate_ci95,
            chi2,
            dof,
            aic,
            bic,
        });
    }
    out
}

/// Fit a batch of exponential relaxation decay curves and run parametric Monte Carlo bootstrap
/// asynchronously on WebGPU (`wasm32` + native compatible).
pub async fn fit_decay_batch_gpu_async(
    engine: &GpuEngine,
    curves: &[DecayCurveInput],
    cfg: &ExpFitConfig,
) -> Result<Vec<DecayCurveFit>, String> {
    let pipe = RelaxGpuPipeline::new(&engine.device);
    fit_decay_batch_with_pipeline_async(engine, &pipe, curves, cfg).await
}

/// Synchronous native convenience wrapper for `fit_decay_batch_gpu_async`.
#[cfg(not(target_arch = "wasm32"))]
pub fn fit_decay_batch_gpu(
    engine: &GpuEngine,
    curves: &[DecayCurveInput],
    cfg: &ExpFitConfig,
) -> Result<Vec<DecayCurveFit>, String> {
    pollster::block_on(fit_decay_batch_gpu_async(engine, curves, cfg))
}

/// Synchronous native convenience wrapper using a pre-compiled `RelaxGpuPipeline` (recommended
/// for repeated interactive / benchmark dispatches so shader compilation is amortized once).
#[cfg(not(target_arch = "wasm32"))]
pub fn fit_decay_batch_with_pipeline(
    engine: &GpuEngine,
    relax_pipe: &RelaxGpuPipeline,
    curves: &[DecayCurveInput],
    cfg: &ExpFitConfig,
) -> Result<Vec<DecayCurveFit>, String> {
    pollster::block_on(fit_decay_batch_with_pipeline_async(
        engine, relax_pipe, curves, cfg,
    ))
}
