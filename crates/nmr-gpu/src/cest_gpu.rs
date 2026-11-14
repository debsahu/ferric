//! Batched WebGPU (`wgpu` + WGSL) 2-state (`7x7`) and 3-state (`10x10`) homogeneous
//! Bloch–McConnell `CEST` and `DEST` Higham `[6/6]` Padé matrix-exponential engine (`T13.4`).
//!
//! Provenance: Clean-room GPU compute implementation from peer-reviewed biomolecular NMR and
//! numerical matrix analysis literature:
//! - McConnell, H. M. (*J. Chem. Phys.* 1958, 28, 430–431 — reaction-modified Bloch equations).
//! - Helgstrand, M., Hard, T. & Allard, P. (*J. Biomol. NMR* 2000, 18, 47–61 — homogeneous
//!   Cartesian Liouvillian formulation).
//! - Vallurupalli, P., Bouvignies, G. & Kay, L. E. (*J. Am. Chem. Soc.* 2012, 134, 8148–8161 —
//!   weak-`B1` `15N` CEST minor-state chemical shifts and Gauss–Hermite `B1` inhomogeneity).
//! - Fawzi, N. L., Ying, J., Ghirlando, R., Torchia, D. A. & Clore, G. M. (*Nature* 2011, 480,
//!   268–272 — Dark-State Exchange Saturation Transfer `DEST`).
//! - Higham, N. J. (*SIAM J. Matrix Anal. Appl.* 2005, 26, 1179–1193 — scaling-and-squaring
//!   `[6/6]` Padé matrix exponential).

use bytemuck::{Pod, Zeroable};
use nmr_core::cest::{
    build_liouvillian_2state, build_liouvillian_3state, expm_pade66, gauss_hermite_normalized,
    mat_vec_mul, Cest2StateParams, Cest3StateParams, CestNormalization,
};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::cest::{
    fit_cest_global_with_evaluator, CestCurveEvalSpec, CestFitConfig, CestGlobalFitResult,
    CestProfileDataset,
};
use wgpu::util::DeviceExt;

use crate::GpuEngine;

/// Maximum offset/matrix jobs dispatched per GPU submission to guarantee TDR safety across
/// browser WebGPU, Metal, Vulkan, and DX12 drivers (`512` workgroups of `64` threads).
const MAX_CEST_JOBS_PER_DISPATCH: usize = 32_768;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CestBatchUniforms {
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
struct Curve2StateGpu {
    p_b: f32,
    k_ex: f32,
    major_ppm: f32,
    dw_ab_ppm: f32,
    r1a: f32,
    r2a: f32,
    r1b: f32,
    r2b: f32,
    larmor_mhz: f32,
    t_ex_s: f32,
    b1_hz: f32,
    b1_inhom_frac: f32,
    e0: f32,
    inv_i0: f32,
    offset_start: u32,
    offset_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Curve3StateGpu {
    p_b: f32,
    p_c: f32,
    k_ex_ab: f32,
    k_ex_bc: f32,
    k_ex_ac: f32,
    major_ppm: f32,
    dw_ab_ppm: f32,
    dw_ac_ppm: f32,
    r1a: f32,
    r2a: f32,
    r1b: f32,
    r2b: f32,
    r1c: f32,
    r2c: f32,
    larmor_mhz: f32,
    t_ex_s: f32,
    b1_hz: f32,
    b1_inhom_frac: f32,
    e0: f32,
    inv_i0: f32,
    offset_start: u32,
    offset_len: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct JobMapGpu {
    curve_idx: u32,
    local_offset_idx: u32,
}

/// Pre-compiled WebGPU compute pipelines for `7x7` / `10x10` Padé `[6/6]` matrix exponentials
/// and batched 2-/3-state `CEST` / `DEST` saturation profile simulation.
pub struct CestGpuPipeline {
    expm7_pipeline: wgpu::ComputePipeline,
    expm10_pipeline: wgpu::ComputePipeline,
    cest2_pipeline: wgpu::ComputePipeline,
    cest3_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl CestGpuPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-bloch-cest-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bloch_cest.wgsl").into()),
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
            label: Some("nmr-bloch-cest-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                storage_ro(5),
                storage_rw(6),
                storage_ro(7),
                storage_rw(8),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-bloch-cest-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let mk_pipe = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            expm7_pipeline: mk_pipe("nmr-expm7-pipe", "eval_expm7_batch"),
            expm10_pipeline: mk_pipe("nmr-expm10-pipe", "eval_expm10_batch"),
            cest2_pipeline: mk_pipe("nmr-cest2-pipe", "eval_cest_2state"),
            cest3_pipeline: mk_pipe("nmr-cest3-pipe", "eval_cest_3state"),
            bgl,
        }
    }
}

/// Validate requested GPU storage buffer capacity against `device.limits().max_buffer_size` and
/// `device.limits().max_storage_buffer_binding_size`.
pub fn validate_cest_storage_bytes(
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

fn compute_2state_e0_inv_i0(p: &Cest2StateParams) -> Result<(f32, f32), String> {
    let pb = p.p_b.clamp(0.0, 0.999_999);
    let pa = 1.0 - pb;
    let e0 = match p.normalization {
        CestNormalization::ZeroTimePhaseCycled | CestNormalization::SameDelayZeroB1 => 0.0,
        CestNormalization::ThermalEquilibrium => 1.0,
    };
    let i0 = match p.normalization {
        CestNormalization::ZeroTimePhaseCycled | CestNormalization::ThermalEquilibrium => pa,
        CestNormalization::SameDelayZeroB1 => {
            let mut l0 = build_liouvillian_2state(
                pb,
                p.k_ex_s_inv,
                0.0,
                0.0,
                0.0,
                0.0,
                p.r1a_s_inv,
                p.r2a_s_inv,
                p.r1b_s_inv,
                p.r2b_s_inv,
            );
            for row in &mut l0 {
                for val in row {
                    *val *= p.t_ex_s;
                }
            }
            let prop = expm_pade66(&l0)?;
            let v0 = [e0, 0.0, 0.0, pa, 0.0, 0.0, pb];
            mat_vec_mul(&prop, &v0)[3].max(1e-12)
        }
    };
    Ok((e0 as f32, (1.0 / i0.max(1e-12)) as f32))
}

fn compute_3state_e0_inv_i0(p: &Cest3StateParams) -> Result<(f32, f32), String> {
    let pb = p.p_b.max(0.0);
    let pc = p.p_c.max(0.0);
    if pb + pc >= 0.999_999 {
        return Err("3-state CEST requires p_B + p_C < 1".to_string());
    }
    let pa = 1.0 - pb - pc;
    let e0 = match p.normalization {
        CestNormalization::ZeroTimePhaseCycled | CestNormalization::SameDelayZeroB1 => 0.0,
        CestNormalization::ThermalEquilibrium => 1.0,
    };
    let i0 = match p.normalization {
        CestNormalization::ZeroTimePhaseCycled | CestNormalization::ThermalEquilibrium => pa,
        CestNormalization::SameDelayZeroB1 => {
            let mut l0 = build_liouvillian_3state(
                pb,
                pc,
                p.k_ex_ab_s_inv,
                p.k_ex_bc_s_inv,
                p.k_ex_ac_s_inv,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                p.r1a_s_inv,
                p.r2a_s_inv,
                p.r1b_s_inv,
                p.r2b_s_inv,
                p.r1c_s_inv,
                p.r2c_s_inv,
            );
            for row in &mut l0 {
                for val in row {
                    *val *= p.t_ex_s;
                }
            }
            let prop = expm_pade66(&l0)?;
            let v0 = [e0, 0.0, 0.0, pa, 0.0, 0.0, pb, 0.0, 0.0, pc];
            mat_vec_mul(&prop, &v0)[3].max(1e-12)
        }
    };
    Ok((e0 as f32, (1.0 / i0.max(1e-12)) as f32))
}

struct DummyCestBuffers {
    quad: wgpu::Buffer,
    c2: wgpu::Buffer,
    c3: wgpu::Buffer,
    offs: wgpu::Buffer,
    jobs: wgpu::Buffer,
    out_int: wgpu::Buffer,
    in_mat: wgpu::Buffer,
    out_mat: wgpu::Buffer,
}

impl DummyCestBuffers {
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
            quad: mk_ro(
                "cest-dummy-quad",
                bytemuck::bytes_of(&QuadNodeGpu::zeroed()),
            ),
            c2: mk_ro(
                "cest-dummy-c2",
                bytemuck::bytes_of(&Curve2StateGpu::zeroed()),
            ),
            c3: mk_ro(
                "cest-dummy-c3",
                bytemuck::bytes_of(&Curve3StateGpu::zeroed()),
            ),
            offs: mk_ro("cest-dummy-offs", bytemuck::cast_slice(&[0.0f32; 4])),
            jobs: mk_ro("cest-dummy-jobs", bytemuck::bytes_of(&JobMapGpu::zeroed())),
            out_int: mk_rw("cest-dummy-out-int", 16),
            in_mat: mk_ro("cest-dummy-in-mat", bytemuck::cast_slice(&[0.0f32; 4])),
            out_mat: mk_rw("cest-dummy-out-mat", 16),
        }
    }
}

async fn expm_pade66_batch_gpu_inner<const N: usize>(
    engine: &GpuEngine,
    pipe: &CestGpuPipeline,
    matrices: &[[[f64; N]; N]],
) -> Result<Vec<[[f64; N]; N]>, String> {
    if N != 7 && N != 10 {
        return Err(format!(
            "expm_pade66_batch_gpu supports 7x7 or 10x10, got {N}x{N}"
        ));
    }
    if matrices.is_empty() {
        return Ok(Vec::new());
    }

    let elems_per_mat = N * N;
    let total_bytes = (matrices.len() as u64)
        .checked_mul((elems_per_mat * 4) as u64)
        .ok_or_else(|| "GPU matrix batch byte count overflow".to_string())?;
    validate_cest_storage_bytes(&engine.device, "expm_matrices", total_bytes)?;

    let dummies = DummyCestBuffers::new(&engine.device);
    let max_wg = (engine.device.limits().max_compute_workgroups_per_dimension as usize) * 64;
    let chunk_cap = MAX_CEST_JOBS_PER_DISPATCH.min(max_wg.max(64));

    let mut all_out = Vec::with_capacity(matrices.len());
    for chunk in matrices.chunks(chunk_cap) {
        let n_jobs = chunk.len();
        let n_elems = n_jobs * elems_per_mat;
        let mut flat_in = Vec::with_capacity(n_elems);
        for mat in chunk {
            for row in mat {
                for &v in row {
                    if !v.is_finite() {
                        return Err("non-finite matrix entry passed to GPU expm_pade66".to_string());
                    }
                    flat_in.push(v as f32);
                }
            }
        }

        let out_bytes = aligned_copy_bytes((n_elems * 4) as u64);
        validate_cest_storage_bytes(&engine.device, "expm_chunk_out", out_bytes)?;

        let uniforms = CestBatchUniforms {
            n_jobs: n_jobs as u32,
            n_quad: 1,
            _pad0: 0,
            _pad1: 0,
        };
        let u_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cest-expm-uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let in_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cest-expm-in"),
                contents: bytemuck::cast_slice(&flat_in),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cest-expm-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cest-expm-bg"),
            layout: &pipe.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dummies.quad.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dummies.c2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dummies.c3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dummies.offs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dummies.jobs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: dummies.out_int.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: in_buf.as_entire_binding(),
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
                label: Some("cest-expm-enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cest-expm-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(if N == 7 {
                &pipe.expm7_pipeline
            } else {
                &pipe.expm10_pipeline
            });
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups((n_jobs as u32).div_ceil(64), 1, 1);
        }
        engine.queue.submit(Some(encoder.finish()));

        let flat_out: Vec<f32> = engine.read_buffer_vec_async(&out_buf, n_elems).await?;
        for m_idx in 0..n_jobs {
            let mut mat = [[0.0f64; N]; N];
            let base = m_idx * elems_per_mat;
            for i in 0..N {
                for j in 0..N {
                    mat[i][j] = flat_out[base + i * N + j] as f64;
                }
            }
            all_out.push(mat);
        }
    }
    Ok(all_out)
}

/// Evaluate a batch of `7x7` Higham `[6/6]` Padé matrix exponentials on WebGPU (`async`).
pub async fn expm_pade66_7_gpu_async(
    engine: &GpuEngine,
    matrices: &[[[f64; 7]; 7]],
) -> Result<Vec<[[f64; 7]; 7]>, String> {
    let pipe = CestGpuPipeline::new(&engine.device);
    expm_pade66_batch_gpu_inner(engine, &pipe, matrices).await
}

/// Synchronous native wrapper around [`expm_pade66_7_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn expm_pade66_7_gpu(
    engine: &GpuEngine,
    matrices: &[[[f64; 7]; 7]],
) -> Result<Vec<[[f64; 7]; 7]>, String> {
    pollster::block_on(expm_pade66_7_gpu_async(engine, matrices))
}

/// Evaluate a batch of `10x10` Higham `[6/6]` Padé matrix exponentials on WebGPU (`async`).
pub async fn expm_pade66_10_gpu_async(
    engine: &GpuEngine,
    matrices: &[[[f64; 10]; 10]],
) -> Result<Vec<[[f64; 10]; 10]>, String> {
    let pipe = CestGpuPipeline::new(&engine.device);
    expm_pade66_batch_gpu_inner(engine, &pipe, matrices).await
}

/// Synchronous native wrapper around [`expm_pade66_10_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn expm_pade66_10_gpu(
    engine: &GpuEngine,
    matrices: &[[[f64; 10]; 10]],
) -> Result<Vec<[[f64; 10]; 10]>, String> {
    pollster::block_on(expm_pade66_10_gpu_async(engine, matrices))
}

/// Evaluate a batch of 2-state `CEST` / `DEST` saturation profiles on WebGPU (`async`) using
/// a pre-compiled [`CestGpuPipeline`].
pub async fn simulate_cest_2state_batch_with_pipeline_async(
    engine: &GpuEngine,
    pipe: &CestGpuPipeline,
    requests: &[(Cest2StateParams, &[f64])],
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
    let mut offsets_flat = Vec::new();
    let mut jobs_all = Vec::new();

    for (c_idx, (p, offs)) in requests.iter().enumerate() {
        let (e0, inv_i0) = compute_2state_e0_inv_i0(p)?;
        let start = offsets_flat.len() as u32;
        let len = offs.len() as u32;
        for (local_i, &rf) in offs.iter().enumerate() {
            offsets_flat.push(rf as f32);
            jobs_all.push(JobMapGpu {
                curve_idx: c_idx as u32,
                local_offset_idx: local_i as u32,
            });
        }
        curves_gpu.push(Curve2StateGpu {
            p_b: p.p_b as f32,
            k_ex: p.k_ex_s_inv as f32,
            major_ppm: p.major_ppm as f32,
            dw_ab_ppm: p.dw_ab_ppm as f32,
            r1a: p.r1a_s_inv as f32,
            r2a: p.r2a_s_inv as f32,
            r1b: p.r1b_s_inv as f32,
            r2b: p.r2b_s_inv as f32,
            larmor_mhz: p.larmor_mhz as f32,
            t_ex_s: p.t_ex_s as f32,
            b1_hz: p.b1_hz as f32,
            b1_inhom_frac: p.b1_inhomogeneity_frac as f32,
            e0,
            inv_i0,
            offset_start: start,
            offset_len: len,
        });
    }

    if jobs_all.is_empty() {
        return Ok(vec![Vec::new(); requests.len()]);
    }

    validate_cest_storage_bytes(
        &engine.device,
        "curves_2s",
        std::mem::size_of_val(curves_gpu.as_slice()) as u64,
    )?;
    validate_cest_storage_bytes(
        &engine.device,
        "offsets_ppm",
        std::mem::size_of_val(offsets_flat.as_slice()) as u64,
    )?;

    let dummies = DummyCestBuffers::new(&engine.device);
    let quad_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cest2-quad"),
            contents: bytemuck::cast_slice(&quad_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let curves_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cest2-curves"),
            contents: bytemuck::cast_slice(&curves_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let offs_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cest2-offsets"),
            contents: bytemuck::cast_slice(&offsets_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let max_wg = (engine.device.limits().max_compute_workgroups_per_dimension as usize) * 64;
    let chunk_cap = MAX_CEST_JOBS_PER_DISPATCH.min(max_wg.max(64));
    let mut flat_intensities = Vec::with_capacity(jobs_all.len());

    for job_chunk in jobs_all.chunks(chunk_cap) {
        let n_jobs = job_chunk.len();
        let job_bytes = std::mem::size_of_val(job_chunk) as u64;
        let out_bytes = aligned_copy_bytes((n_jobs * 4) as u64);
        validate_cest_storage_bytes(&engine.device, "cest2_jobs", job_bytes)?;
        validate_cest_storage_bytes(&engine.device, "cest2_out", out_bytes)?;

        let uniforms = CestBatchUniforms {
            n_jobs: n_jobs as u32,
            n_quad: quad_gpu.len() as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let u_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cest2-uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let job_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cest2-jobmap"),
                contents: bytemuck::cast_slice(job_chunk),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cest2-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cest2-bg"),
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
                    resource: dummies.c3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: offs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: job_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: dummies.in_mat.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: dummies.out_mat.as_entire_binding(),
                },
            ],
        });

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cest2-enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cest2-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipe.cest2_pipeline);
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

/// Evaluate a batch of 2-state `CEST` / `DEST` saturation profiles on WebGPU (`async`).
pub async fn simulate_cest_2state_batch_gpu_async(
    engine: &GpuEngine,
    requests: &[(Cest2StateParams, &[f64])],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let pipe = CestGpuPipeline::new(&engine.device);
    simulate_cest_2state_batch_with_pipeline_async(engine, &pipe, requests, n_b1_quad).await
}

/// Synchronous native wrapper around [`simulate_cest_2state_batch_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn simulate_cest_2state_batch_gpu(
    engine: &GpuEngine,
    requests: &[(Cest2StateParams, &[f64])],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    pollster::block_on(simulate_cest_2state_batch_gpu_async(
        engine, requests, n_b1_quad,
    ))
}

/// Evaluate a batch of 3-state (`10x10` linear or triangular `A <-> B <-> C`) `CEST` saturation
/// profiles on WebGPU (`async`).
pub async fn simulate_cest_3state_batch_gpu_async(
    engine: &GpuEngine,
    requests: &[(Cest3StateParams, &[f64])],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    let pipe = CestGpuPipeline::new(&engine.device);

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
    let mut offsets_flat = Vec::new();
    let mut jobs_all = Vec::new();

    for (c_idx, (p, offs)) in requests.iter().enumerate() {
        let (e0, inv_i0) = compute_3state_e0_inv_i0(p)?;
        let start = offsets_flat.len() as u32;
        let len = offs.len() as u32;
        for (local_i, &rf) in offs.iter().enumerate() {
            offsets_flat.push(rf as f32);
            jobs_all.push(JobMapGpu {
                curve_idx: c_idx as u32,
                local_offset_idx: local_i as u32,
            });
        }
        curves_gpu.push(Curve3StateGpu {
            p_b: p.p_b as f32,
            p_c: p.p_c as f32,
            k_ex_ab: p.k_ex_ab_s_inv as f32,
            k_ex_bc: p.k_ex_bc_s_inv as f32,
            k_ex_ac: p.k_ex_ac_s_inv as f32,
            major_ppm: p.major_ppm as f32,
            dw_ab_ppm: p.dw_ab_ppm as f32,
            dw_ac_ppm: p.dw_ac_ppm as f32,
            r1a: p.r1a_s_inv as f32,
            r2a: p.r2a_s_inv as f32,
            r1b: p.r1b_s_inv as f32,
            r2b: p.r2b_s_inv as f32,
            r1c: p.r1c_s_inv as f32,
            r2c: p.r2c_s_inv as f32,
            larmor_mhz: p.larmor_mhz as f32,
            t_ex_s: p.t_ex_s as f32,
            b1_hz: p.b1_hz as f32,
            b1_inhom_frac: p.b1_inhomogeneity_frac as f32,
            e0,
            inv_i0,
            offset_start: start,
            offset_len: len,
            _pad0: 0,
            _pad1: 0,
        });
    }

    if jobs_all.is_empty() {
        return Ok(vec![Vec::new(); requests.len()]);
    }

    validate_cest_storage_bytes(
        &engine.device,
        "curves_3s",
        std::mem::size_of_val(curves_gpu.as_slice()) as u64,
    )?;
    validate_cest_storage_bytes(
        &engine.device,
        "offsets_3s",
        std::mem::size_of_val(offsets_flat.as_slice()) as u64,
    )?;

    let dummies = DummyCestBuffers::new(&engine.device);
    let quad_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cest3-quad"),
            contents: bytemuck::cast_slice(&quad_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let curves_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cest3-curves"),
            contents: bytemuck::cast_slice(&curves_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let offs_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cest3-offsets"),
            contents: bytemuck::cast_slice(&offsets_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let max_wg = (engine.device.limits().max_compute_workgroups_per_dimension as usize) * 64;
    let chunk_cap = MAX_CEST_JOBS_PER_DISPATCH.min(max_wg.max(64));
    let mut flat_intensities = Vec::with_capacity(jobs_all.len());

    for job_chunk in jobs_all.chunks(chunk_cap) {
        let n_jobs = job_chunk.len();
        let job_bytes = std::mem::size_of_val(job_chunk) as u64;
        let out_bytes = aligned_copy_bytes((n_jobs * 4) as u64);
        validate_cest_storage_bytes(&engine.device, "cest3_jobs", job_bytes)?;
        validate_cest_storage_bytes(&engine.device, "cest3_out", out_bytes)?;

        let uniforms = CestBatchUniforms {
            n_jobs: n_jobs as u32,
            n_quad: quad_gpu.len() as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let u_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cest3-uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let job_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cest3-jobmap"),
                contents: bytemuck::cast_slice(job_chunk),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cest3-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cest3-bg"),
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
                    resource: dummies.c2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: curves_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: offs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: job_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: dummies.in_mat.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: dummies.out_mat.as_entire_binding(),
                },
            ],
        });

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cest3-enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cest3-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipe.cest3_pipeline);
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

/// Synchronous native wrapper around [`simulate_cest_3state_batch_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn simulate_cest_3state_batch_gpu(
    engine: &GpuEngine,
    requests: &[(Cest3StateParams, &[f64])],
    n_b1_quad: usize,
) -> Result<Vec<Vec<f64>>, String> {
    pollster::block_on(simulate_cest_3state_batch_gpu_async(
        engine, requests, n_b1_quad,
    ))
}

/// Fit global `(p_B, k_ex)` + per-residue `(major_ppm, dw_ab_ppm, R1A, R2A, R2B)` across multi-`B1`
/// `CEST` / `DEST` profiles using WebGPU batched Padé `[6/6]` Jacobian & grid evaluations paired
/// with Schur-complement Levenberg–Marquardt optimization (`T13.4`).
#[cfg(not(target_arch = "wasm32"))]
pub fn fit_cest_global_gpu(
    engine: &GpuEngine,
    profiles: &[CestProfileDataset],
    config: &CestFitConfig,
) -> Result<CestGlobalFitResult, String> {
    let pipe = CestGpuPipeline::new(&engine.device);
    fit_cest_global_with_evaluator(profiles, config, |specs: &[CestCurveEvalSpec], n_quad| {
        let mut requests = Vec::with_capacity(specs.len());
        for sp in specs {
            let prof = &profiles[sp.profile_idx];
            let p = Cest2StateParams {
                p_b: sp.p_b,
                k_ex_s_inv: sp.k_ex_s_inv,
                major_ppm: sp.major_ppm,
                dw_ab_ppm: sp.dw_ab_ppm,
                r1a_s_inv: sp.r1a_s_inv,
                r2a_s_inv: sp.r2a_s_inv,
                r1b_s_inv: sp.r1b_s_inv,
                r2b_s_inv: sp.r2b_s_inv,
                larmor_mhz: prof.larmor_mhz,
                t_ex_s: prof.t_ex_s,
                b1_hz: prof.b1_hz,
                b1_inhomogeneity_frac: prof.b1_inhomogeneity_frac,
                n_b1_quad: n_quad,
                normalization: prof.normalization,
            };
            requests.push((p, prof.offsets_ppm.as_slice()));
        }
        pollster::block_on(simulate_cest_2state_batch_with_pipeline_async(
            engine, &pipe, &requests, n_quad,
        ))
    })
}
