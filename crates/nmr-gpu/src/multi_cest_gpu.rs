//! Batched WebGPU (`wgpu` + WGSL) `DANTE-CEST` & cosine-modulated `cos-CEST` dual-sideband
//! homogeneous `7x7` Bloch–McConnell propagator compute engine (`T22.6`).
//!
//! Provenance: Clean-room GPU compute implementation strictly from peer-reviewed literature:
//! - Carneiro, M. G., Reddy, J. G., Griesinger, C. & Lee, D. (*J. Biomol. NMR* 2015, 62,
//!   237–246 — `DANTE-CEST` pulse-train Bloch–McConnell propagator).
//! - Yuwen, T., Kay, L. E. & Bouvignies, G. (*J. Phys. Chem. Lett.* 2018, 9, 2501–2507 —
//!   cosine-modulated `cos-CEST` dual-sideband saturation propagator).
//! - Vallurupalli, P., Bouvignies, G. & Kay, L. E. (*J. Am. Chem. Soc.* 2012, 134, 8148–8161 —
//!   2-state `7x7` homogeneous Bloch–McConnell Liouvillian & `B1` inhomogeneity quadrature).
//! - Higham, N. J. (*SIAM J. Matrix Anal. Appl.* 2005, 26, 1179–1193 — scaling-and-squaring
//!   `[6/6]` Padé matrix exponential).

use bytemuck::{Pod, Zeroable};
use nmr_core::cest::gauss_hermite_normalized;
use nmr_core::multi_cest::{
    compute_multi_cest_e0_inv_i0, CosCestParams, DanteCestParams, MultiCestScheme,
    MultiCestSimParams,
};
use wgpu::util::DeviceExt;

use crate::GpuEngine;

/// Maximum `(curve, offset)` jobs dispatched per WebGPU submission for TDR safety across
/// Metal, Vulkan, DX12, and browser WebGPU (`256` workgroups of `64` threads).
const MAX_MULTI_CEST_JOBS_PER_DISPATCH: usize = 16_384;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MultiCestUniforms {
    n_jobs: u32,
    n_quad: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct QuadNodeGpu {
    x_node: f32,
    weight: f32,
    _pad0: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MultiCestCurveGpu {
    scheme: u32,
    n_pulses_or_steps: u32,
    offset_start: u32,
    offset_len: u32,

    p_b: f32,
    k_ex: f32,
    major_ppm: f32,
    dw_ab_ppm: f32,

    r1a: f32,
    r2a: f32,
    r1b: f32,
    r2b: f32,

    larmor_mhz: f32,
    t_ex_or_tau_p: f32,
    b1_hz: f32,
    tau_d_or_mod_freq: f32,

    b1_inhom_frac: f32,
    e0: f32,
    inv_i0: f32,
    _pad0: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct JobMapGpu {
    curve_idx: u32,
    local_offset_idx: u32,
}

/// Pre-compiled WebGPU compute pipeline for batched `CW-CEST`, `DANTE-CEST`, and `cos-CEST`
/// profile evaluation across `(offsets x B1)`.
pub struct MultiCestGpuPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl MultiCestGpuPipeline {
    /// Compile the `multi_cest.wgsl` compute shader and bind group layout on `device`.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-multi-cest-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/multi_cest.wgsl").into()),
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
            label: Some("nmr-multi-cest-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                storage_rw(5),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-multi-cest-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-multi-cest-pipe"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("eval_multi_cest_batch"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bgl }
    }
}

/// Validate requested GPU storage buffer capacity against `device.limits().max_buffer_size` and
/// `device.limits().max_storage_buffer_binding_size`.
pub fn validate_multi_cest_storage_bytes(
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

/// Evaluate a batch of multi-frequency `CEST` (`CW`, `DANTE-CEST`, `cos-CEST`) profiles on
/// WebGPU (`async`) using a pre-compiled [`MultiCestGpuPipeline`].
pub async fn simulate_multi_cest_batch_with_pipeline_async(
    engine: &GpuEngine,
    pipe: &MultiCestGpuPipeline,
    requests: &[(MultiCestSimParams, &[f64])],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }

    let quad_pairs = gauss_hermite_normalized(n_b1_quad.max(1));
    let quad_gpu: Vec<QuadNodeGpu> = quad_pairs
        .iter()
        .map(|&(x, w)| QuadNodeGpu {
            x_node: x as f32,
            weight: w as f32,
            _pad0: 0.0,
            _pad1: 0.0,
        })
        .collect();

    let mut curves_gpu = Vec::with_capacity(requests.len());
    let mut offsets_flat: Vec<f32> = Vec::new();
    let mut jobs_all: Vec<JobMapGpu> = Vec::new();

    for (c_idx, (p, offs)) in requests.iter().enumerate() {
        p.validate()?;
        for (i, &rf) in offs.iter().enumerate() {
            if !rf.is_finite() {
                return Err(format!(
                    "non-finite offset at curve {c_idx} index {i}: {rf}"
                ));
            }
        }
        let (e0, inv_i0) = compute_multi_cest_e0_inv_i0(p)?;
        let start = u32::try_from(offsets_flat.len())
            .map_err(|_| "multi-CEST offset index exceeds u32::MAX".to_string())?;
        let len = u32::try_from(offs.len())
            .map_err(|_| "multi-CEST profile offset count exceeds u32::MAX".to_string())?;
        let c_idx_u32 = u32::try_from(c_idx)
            .map_err(|_| "multi-CEST curve index exceeds u32::MAX".to_string())?;

        for (local_i, &rf) in offs.iter().enumerate() {
            offsets_flat.push(rf as f32);
            jobs_all.push(JobMapGpu {
                curve_idx: c_idx_u32,
                local_offset_idx: local_i as u32,
            });
        }

        let (scheme_code, n_pulses_or_steps, t_ex_or_tau_p, b1_hz, tau_d_or_mod_freq) =
            match p.scheme {
                MultiCestScheme::Cw { t_ex_s, b1_hz } => (0u32, 1u32, t_ex_s, b1_hz, 0.0),
                MultiCestScheme::Dante {
                    tau_pulse_s,
                    tau_delay_s,
                    n_pulses,
                    b1_pulse_hz,
                } => (1u32, n_pulses, tau_pulse_s, b1_pulse_hz, tau_delay_s),
                MultiCestScheme::CosModulated {
                    t_ex_s,
                    b1_eff_hz,
                    mod_freq_hz,
                    n_steps_per_cycle,
                } => (2u32, n_steps_per_cycle, t_ex_s, b1_eff_hz, mod_freq_hz),
            };

        curves_gpu.push(MultiCestCurveGpu {
            scheme: scheme_code,
            n_pulses_or_steps,
            offset_start: start,
            offset_len: len,
            p_b: p.p_b as f32,
            k_ex: p.k_ex_s_inv as f32,
            major_ppm: p.major_ppm as f32,
            dw_ab_ppm: p.dw_ab_ppm as f32,
            r1a: p.r1a_s_inv as f32,
            r2a: p.r2a_s_inv as f32,
            r1b: p.r1b_s_inv as f32,
            r2b: p.r2b_s_inv as f32,
            larmor_mhz: p.larmor_mhz as f32,
            t_ex_or_tau_p: t_ex_or_tau_p as f32,
            b1_hz: b1_hz as f32,
            tau_d_or_mod_freq: tau_d_or_mod_freq as f32,
            b1_inhom_frac: p.b1_inhomogeneity_frac as f32,
            e0: e0 as f32,
            inv_i0: inv_i0 as f32,
            _pad0: 0.0,
        });
    }

    if jobs_all.is_empty() {
        return Ok(vec![Vec::new(); requests.len()]);
    }

    validate_multi_cest_storage_bytes(
        &engine.device,
        "multi_cest_quad",
        std::mem::size_of_val(quad_gpu.as_slice()) as u64,
    )?;
    validate_multi_cest_storage_bytes(
        &engine.device,
        "multi_cest_curves",
        std::mem::size_of_val(curves_gpu.as_slice()) as u64,
    )?;
    validate_multi_cest_storage_bytes(
        &engine.device,
        "multi_cest_offsets",
        std::mem::size_of_val(offsets_flat.as_slice()) as u64,
    )?;

    let quad_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi-cest-quad"),
            contents: bytemuck::cast_slice(&quad_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let curves_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi-cest-curves"),
            contents: bytemuck::cast_slice(&curves_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let offs_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("multi-cest-offsets"),
            contents: bytemuck::cast_slice(&offsets_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let max_wg = (engine.device.limits().max_compute_workgroups_per_dimension as usize) * 64;
    let chunk_cap = MAX_MULTI_CEST_JOBS_PER_DISPATCH.min(max_wg.max(64));
    let mut flat_intensities = Vec::with_capacity(jobs_all.len());

    for job_chunk in jobs_all.chunks(chunk_cap) {
        let n_jobs = job_chunk.len();
        let job_bytes = std::mem::size_of_val(job_chunk) as u64;
        let out_bytes = aligned_copy_bytes((n_jobs * 4) as u64);
        validate_multi_cest_storage_bytes(&engine.device, "multi_cest_jobs", job_bytes)?;
        validate_multi_cest_storage_bytes(&engine.device, "multi_cest_out", out_bytes)?;

        let uniforms = MultiCestUniforms {
            n_jobs: n_jobs as u32,
            n_quad: quad_gpu.len() as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let u_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("multi-cest-uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let job_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("multi-cest-jobmap"),
                contents: bytemuck::cast_slice(job_chunk),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("multi-cest-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("multi-cest-bg"),
            layout: &pipe.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: quad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: curves_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: offs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: job_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("multi-cest-enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("multi-cest-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipe.pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups((n_jobs as u32).div_ceil(64), 1, 1);
        }
        engine.queue.submit(Some(encoder.finish()));

        let chunk_out: Vec<f32> = engine.read_buffer_vec_async(&out_buf, n_jobs).await?;
        flat_intensities.extend(chunk_out.into_iter().map(|v| v as f64));
    }

    let mut curves_out = Vec::with_capacity(requests.len());
    let mut cursor = 0usize;
    for (_, offs) in requests {
        let end = cursor + offs.len();
        curves_out.push(flat_intensities[cursor..end].to_vec());
        cursor = end;
    }
    Ok(curves_out)
}

/// Evaluate a batch of multi-frequency `CEST` profiles on WebGPU (`async`).
pub async fn simulate_multi_cest_batch_gpu_async(
    engine: &GpuEngine,
    requests: &[(MultiCestSimParams, &[f64])],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let pipe = MultiCestGpuPipeline::new(&engine.device);
    simulate_multi_cest_batch_with_pipeline_async(engine, &pipe, requests, n_b1_quad).await
}

/// Synchronous native wrapper around [`simulate_multi_cest_batch_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn simulate_multi_cest_batch_gpu(
    engine: &GpuEngine,
    requests: &[(MultiCestSimParams, &[f64])],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    pollster::block_on(simulate_multi_cest_batch_gpu_async(
        engine, requests, n_b1_quad,
    ))
}

/// Evaluate a batch of `DANTE-CEST` profiles across `(offsets x B1)` on WebGPU (`async`).
pub async fn simulate_dante_cest_batch_gpu_async(
    engine: &GpuEngine,
    requests: &[(DanteCestParams, &[f64])],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let multi: Vec<(MultiCestSimParams, &[f64])> = requests
        .iter()
        .map(|(p, offs)| (p.to_multi_params(), *offs))
        .collect();
    simulate_multi_cest_batch_gpu_async(engine, &multi, n_b1_quad).await
}

/// Synchronous native wrapper around [`simulate_dante_cest_batch_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn simulate_dante_cest_batch_gpu(
    engine: &GpuEngine,
    requests: &[(DanteCestParams, &[f64])],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    pollster::block_on(simulate_dante_cest_batch_gpu_async(
        engine, requests, n_b1_quad,
    ))
}

/// Evaluate a batch of cosine-modulated `cos-CEST` profiles across `(offsets x B1)` on WebGPU (`async`).
pub async fn simulate_cos_cest_batch_gpu_async(
    engine: &GpuEngine,
    requests: &[(CosCestParams, &[f64])],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let multi: Vec<(MultiCestSimParams, &[f64])> = requests
        .iter()
        .map(|(p, offs)| (p.to_multi_params(), *offs))
        .collect();
    simulate_multi_cest_batch_gpu_async(engine, &multi, n_b1_quad).await
}

/// Synchronous native wrapper around [`simulate_cos_cest_batch_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn simulate_cos_cest_batch_gpu(
    engine: &GpuEngine,
    requests: &[(CosCestParams, &[f64])],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    pollster::block_on(simulate_cos_cest_batch_gpu_async(
        engine, requests, n_b1_quad,
    ))
}
