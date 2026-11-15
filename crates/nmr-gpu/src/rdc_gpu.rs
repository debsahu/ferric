//! Batched multi-conformer WebGPU (`wgpu` + WGSL) torsion-space Residual Dipolar Coupling (`RDC`)
//! energy, analytical torque gradient, and ensemble refinement engine (`docs/specs/04-structure.md` §5).
//!
//! Provenance: Clean-room GPU compute implementation from peer-reviewed biomolecular NMR and
//! polymer kinematics literature:
//! - Tjandra, N. & Bax, A. (*Science* 1997, 278, 1111–1114 — dipolar couplings in weakly aligned
//!   macromolecules).
//! - Cornilescu, G., Marquardt, J. L., Ottiger, M. & Bax, A. (*J. Am. Chem. Soc.* 1998, 120,
//!   6836–6837 — `Q`-factor validation in bicelle media).
//! - Losonczi, J. A., Andrec, M., Fischer, M. W. F. & Prestegard, J. H. (*J. Magn. Reson.* 1999,
//!   138, 334–342 — linear 5-component traceless symmetric Saupe alignment tensor SVD).
//! - Clore, G. M., Gronenborn, A. M. & Tjandra, N. (*J. Magn. Reson.* 1998, 131, 159–162 — axial
//!   and rhombic dipolar alignment components).
//! - Parsons, J., Holmes, J. B., Rojas, J. M., Tsai, J. & Strauss, C. E. M. (*J. Comput. Chem.*
//!   2005, 26, 1063–1068 — Natural Extension Reference Frame forward kinematics).
//! - Abe, H., Braun, W., Noguti, T. & Go, N. (*Comput. Chem.* 1984, 8, 239–247 — `O(L)` backward
//!   suffix-sum torque identity for exact torsion-angle derivatives).

use std::collections::BTreeMap;
use std::f64::consts::PI;

use bytemuck::{Pod, Zeroable};
use nmr_core::dihedrals::DihedralRestraint;
use nmr_core::noesy::AmbiguousDistanceRestraint;
use nmr_core::rdc::{
    evaluate_rdc_validation, fit_saupe_tensor_svd, normalize_rdc_atom_name, RdcMediumSet,
    RdcValidationSummary,
};
use nmr_core::structure::{
    build_backbone_from_torsions_rad, evaluate_energy_and_gradients,
    evaluate_energy_and_gradients_with_rdcs_omega_hbond, kabsch_rmsd, EnergyWeights,
    StructureConformer,
};
use wgpu::util::DeviceExt;

use crate::GpuEngine;

/// Maximum number of conformers evaluated per single WebGPU compute dispatch / `queue.submit`
/// so large multi-hundred-conformer ensembles never trip OS GPU TDR watchdogs.
pub const MAX_RDC_CONFORMERS_PER_DISPATCH: usize = 256;

#[derive(Clone, Copy, Debug)]
struct RdcGpuDispatchByteSizes {
    scratch_bytes: u64,
    energy_bytes: u64,
    grad_bytes: u64,
}

/// Validate all 8 WebGPU RDC buffer sizes (`params_buf`, `flags_buf`, `torsions_buf`, `saupe_buf`,
/// `terms_buf`, `scratch_buf`, `energies_buf`, `grads_buf`) against BOTH `limits.max_buffer_size`
/// and `limits.max_storage_buffer_binding_size` / `limits.max_uniform_buffer_binding_size`, plus
/// `limits.max_compute_workgroups_per_dimension` and 32-bit WGSL index bounds BEFORE creating any
/// GPU buffer (preventing `wgpu` validation panics on oversize inputs).
pub fn validate_rdc_gpu_dispatch_limits(
    limits: &wgpu::Limits,
    n_conf_dispatch: usize,
    l_res: usize,
    n_media: usize,
    n_restraints: usize,
) -> Result<(), String> {
    let _ =
        compute_and_validate_dispatch_sizes(limits, n_conf_dispatch, l_res, n_media, n_restraints)?;
    Ok(())
}

fn compute_and_validate_dispatch_sizes(
    limits: &wgpu::Limits,
    n_conf_dispatch: usize,
    l_res: usize,
    n_media: usize,
    n_restraints: usize,
) -> Result<RdcGpuDispatchByteSizes, String> {
    if n_conf_dispatch > u32::MAX as usize
        || l_res > u32::MAX as usize
        || n_media > u32::MAX as usize
        || n_restraints > u32::MAX as usize
    {
        return Err("GPU RDC dimensions exceed 32-bit WGSL u32 index limit".to_string());
    }

    let n_wg = n_conf_dispatch.div_ceil(64);
    if n_wg > limits.max_compute_workgroups_per_dimension as usize {
        return Err(format!(
            "GPU RDC dispatch workgroups ({n_wg}) exceed device max_compute_workgroups_per_dimension ({})",
            limits.max_compute_workgroups_per_dimension
        ));
    }

    let check_storage = |name: &str, bytes: u64| -> Result<(), String> {
        if bytes > limits.max_buffer_size {
            return Err(format!(
                "GPU RDC {name} ({bytes} B) exceeds device max_buffer_size ({} B)",
                limits.max_buffer_size
            ));
        }
        let max_sb = limits.max_storage_buffer_binding_size;
        if bytes > max_sb {
            return Err(format!(
                "GPU RDC {name} ({bytes} B) exceeds device max_storage_buffer_binding_size ({max_sb} B)"
            ));
        }
        Ok(())
    };

    // 0. `params_buf` (UNIFORM)
    let params_bytes = std::mem::size_of::<RdcRefineParamsGpu>() as u64;
    if params_bytes > limits.max_buffer_size {
        return Err(format!(
            "GPU RDC params_buf ({params_bytes} B) exceeds device max_buffer_size ({} B)",
            limits.max_buffer_size
        ));
    }
    let max_ub = limits.max_uniform_buffer_binding_size;
    if params_bytes > max_ub {
        return Err(format!(
            "GPU RDC params_buf ({params_bytes} B) exceeds device max_uniform_buffer_binding_size ({max_ub} B)"
        ));
    }

    // 1. `flags_buf` (STORAGE)
    let flags_bytes = (l_res as u64)
        .checked_mul(4)
        .ok_or_else(|| "GPU RDC flags_buf byte size overflow".to_string())?
        .max(4);
    check_storage("flags_buf", flags_bytes)?;

    // 2. `torsions_buf` (STORAGE)
    let torsions_elems = n_conf_dispatch
        .checked_mul(l_res)
        .ok_or_else(|| "GPU RDC torsions_buf element count overflow".to_string())?;
    if torsions_elems > u32::MAX as usize {
        return Err("GPU RDC torsions_buf index exceeds u32::MAX".to_string());
    }
    let torsions_bytes = (torsions_elems as u64)
        .checked_mul(16)
        .ok_or_else(|| "GPU RDC torsions_buf byte size overflow".to_string())?
        .max(16);
    check_storage("torsions_buf", torsions_bytes)?;

    // 3. `saupe_buf` (STORAGE)
    let saupe_rows = n_conf_dispatch
        .checked_mul(n_media)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| "GPU RDC saupe_buf row count overflow".to_string())?;
    if saupe_rows > u32::MAX as usize {
        return Err("GPU RDC saupe_buf index exceeds u32::MAX".to_string());
    }
    let saupe_bytes = (saupe_rows as u64)
        .checked_mul(16)
        .ok_or_else(|| "GPU RDC saupe_buf byte size overflow".to_string())?
        .max(16);
    check_storage("saupe_buf", saupe_bytes)?;

    // 4. `terms_buf` (STORAGE)
    let terms_bytes = (n_restraints as u64)
        .checked_mul(std::mem::size_of::<RdcTermGpu>() as u64)
        .ok_or_else(|| "GPU RDC terms_buf byte size overflow".to_string())?
        .max(std::mem::size_of::<RdcTermGpu>() as u64);
    check_storage("terms_buf", terms_bytes)?;

    // 5. `scratch_buf` (STORAGE)
    let n_segs = l_res
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| "GPU RDC segment count overflow".to_string())?;
    let stride_scratch = l_res
        .checked_mul(7)
        .and_then(|a| n_segs.checked_mul(2).and_then(|b| a.checked_add(b)))
        .ok_or_else(|| "GPU RDC scratch stride overflow".to_string())?;
    let scratch_vec4_count = n_conf_dispatch
        .checked_mul(stride_scratch)
        .ok_or_else(|| "GPU RDC scratch_buf element count overflow".to_string())?;
    if scratch_vec4_count > u32::MAX as usize {
        return Err("GPU RDC scratch_buf index exceeds u32::MAX".to_string());
    }
    let scratch_bytes = (scratch_vec4_count as u64)
        .checked_mul(16)
        .ok_or_else(|| "GPU RDC scratch_buf byte size overflow".to_string())?
        .max(16);
    check_storage("scratch_buf", scratch_bytes)?;

    // 6. `energies_buf` (STORAGE | COPY_SRC)
    let raw_energy_bytes = (n_conf_dispatch as u64)
        .checked_mul(4)
        .ok_or_else(|| "GPU RDC energies_buf byte size overflow".to_string())?;
    let energy_bytes = raw_energy_bytes
        .div_ceil(wgpu::COPY_BUFFER_ALIGNMENT)
        .max(1)
        .checked_mul(wgpu::COPY_BUFFER_ALIGNMENT)
        .ok_or_else(|| "GPU RDC energies_buf aligned byte size overflow".to_string())?;
    check_storage("energies_buf", energy_bytes)?;

    // 7. `grads_buf` (STORAGE | COPY_SRC)
    let raw_grad_bytes = (torsions_elems as u64)
        .checked_mul(8)
        .ok_or_else(|| "GPU RDC grads_buf byte size overflow".to_string())?;
    let grad_bytes = raw_grad_bytes
        .div_ceil(wgpu::COPY_BUFFER_ALIGNMENT)
        .max(1)
        .checked_mul(wgpu::COPY_BUFFER_ALIGNMENT)
        .ok_or_else(|| "GPU RDC grads_buf aligned byte size overflow".to_string())?;
    check_storage("grads_buf", grad_bytes)?;

    Ok(RdcGpuDispatchByteSizes {
        scratch_bytes,
        energy_bytes,
        grad_bytes,
    })
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RdcRefineParamsGpu {
    n_conf: u32,
    l_res: u32,
    n_media: u32,
    n_restraints: u32,
    w_rdc: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RdcTermGpu {
    medium_idx: u32,
    atom_a: u32,
    atom_b: u32,
    seg_a: u32,
    seg_b: u32,
    d_obs: f32,
    d_max: f32,
    inv_sig2: f32,
}

/// Per-conformer input state (`(phi, psi, omega)` in radians + per-medium `3x3` Saupe tensors)
/// for batched WebGPU RDC energy & analytical torsion-gradient evaluation.
#[derive(Clone, Debug)]
pub struct RdcGpuConformerInput {
    pub phi_rad: Vec<f64>,
    pub psi_rad: Vec<f64>,
    pub omega_rad: Vec<f64>,
    pub saupe_matrices: Vec<[[f64; 3]; 3]>,
}

/// Per-conformer WebGPU RDC energy and analytical `(dE_rdc/dphi, dE_rdc/dpsi)` torsion gradients.
#[derive(Clone, Debug)]
pub struct RdcGpuConformerOutput {
    pub rdc_energy: f64,
    pub grad_phi: Vec<f64>,
    pub grad_psi: Vec<f64>,
}

/// Pre-compiled WebGPU compute pipeline for batched multi-conformer RDC torsion evaluation.
pub struct RdcGpuPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl RdcGpuPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-rdc-refine-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/rdc_refine.wgsl").into()),
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
            label: Some("nmr-rdc-refine-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                storage_rw(5),
                storage_rw(6),
                storage_rw(7),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-rdc-refine-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-rdc-refine-pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("eval_rdc_batch"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bgl }
    }
}

fn resolve_gpu_atom_slot_and_seg(
    sequence: &[String],
    seq_to_idx: &BTreeMap<i32, usize>,
    seq_id: i32,
    atom_name: &str,
) -> Option<(u32, u32)> {
    let &res_idx = seq_to_idx.get(&seq_id)?;
    let res_name = sequence.get(res_idx)?;
    let norm = normalize_rdc_atom_name(atom_name);
    let i = res_idx as u32;
    let is_gly = res_name.eq_ignore_ascii_case("GLY");
    match norm {
        "N" => Some((7 * i, 2 * i)),
        "H" => {
            if res_name.eq_ignore_ascii_case("PRO") {
                None
            } else {
                Some((7 * i + 1, 2 * i))
            }
        }
        "CA" => Some((7 * i + 2, 2 * i)),
        "HA" => {
            if is_gly {
                None
            } else {
                Some((7 * i + 3, 2 * i + 1))
            }
        }
        "HA2" => {
            if is_gly {
                Some((7 * i + 3, 2 * i + 1))
            } else {
                None
            }
        }
        "HA3" => {
            if is_gly {
                Some((7 * i + 6, 2 * i + 1))
            } else {
                None
            }
        }
        "C" => Some((7 * i + 4, 2 * i + 1)),
        "O" => Some((7 * i + 5, 2 * i + 2)),
        "CB" => {
            if is_gly {
                None
            } else {
                Some((7 * i + 6, 2 * i + 1))
            }
        }
        _ => None,
    }
}

fn pack_rdc_terms_gpu(
    sequence: &[String],
    seq_ids: &[i32],
    rdc_media: &[RdcMediumSet],
) -> Result<Vec<RdcTermGpu>, String> {
    let mut seq_to_idx = BTreeMap::new();
    for (idx, &sid) in seq_ids.iter().enumerate() {
        seq_to_idx.insert(sid, idx);
    }

    let mut terms = Vec::new();
    for (m_idx, medium) in rdc_media.iter().enumerate() {
        for r in &medium.restraints {
            r.validate()?;
            if r.is_free {
                continue;
            }
            let (Some((atom_a, seg_a)), Some((atom_b, seg_b))) = (
                resolve_gpu_atom_slot_and_seg(sequence, &seq_to_idx, r.seq_id_1, &r.atom_name_1),
                resolve_gpu_atom_slot_and_seg(sequence, &seq_to_idx, r.seq_id_2, &r.atom_name_2),
            ) else {
                continue;
            };
            let inv_sig2 = 1.0 / (r.sigma_hz * r.sigma_hz);
            terms.push(RdcTermGpu {
                medium_idx: m_idx as u32,
                atom_a,
                atom_b,
                seg_a,
                seg_b,
                d_obs: r.d_obs_hz as f32,
                d_max: r.d_max_hz as f32,
                inv_sig2: inv_sig2 as f32,
            });
        }
    }
    Ok(terms)
}

/// Evaluate multi-conformer harmonic RDC energies and analytical `(dE_rdc/dphi, dE_rdc/dpsi)`
/// torsion gradients on WebGPU asynchronously (`wasm32` + native compatible).
pub async fn evaluate_rdc_batch_gpu_async(
    engine: &GpuEngine,
    sequence: &[String],
    seq_ids: &[i32],
    rdc_media: &[RdcMediumSet],
    conformers: &[RdcGpuConformerInput],
    w_rdc: f64,
) -> Result<Vec<RdcGpuConformerOutput>, String> {
    let rdc_pipe = RdcGpuPipeline::new(&engine.device);
    evaluate_rdc_batch_with_pipeline_async(
        engine, &rdc_pipe, sequence, seq_ids, rdc_media, conformers, w_rdc,
    )
    .await
}

async fn evaluate_rdc_batch_with_pipeline_async(
    engine: &GpuEngine,
    rdc_pipe: &RdcGpuPipeline,
    sequence: &[String],
    seq_ids: &[i32],
    rdc_media: &[RdcMediumSet],
    conformers: &[RdcGpuConformerInput],
    w_rdc: f64,
) -> Result<Vec<RdcGpuConformerOutput>, String> {
    let l = sequence.len();
    if seq_ids.len() != l {
        return Err("sequence and seq_ids length mismatch".to_string());
    }
    if !w_rdc.is_finite() || w_rdc < 0.0 {
        return Err(format!("invalid w_rdc={w_rdc}: must be finite and >= 0"));
    }
    let n_conf = conformers.len();
    if n_conf == 0 {
        return Ok(Vec::new());
    }
    if l == 0 || rdc_media.is_empty() || w_rdc == 0.0 {
        return Ok((0..n_conf)
            .map(|_| RdcGpuConformerOutput {
                rdc_energy: 0.0,
                grad_phi: vec![0.0; l],
                grad_psi: vec![0.0; l],
            })
            .collect());
    }

    let n_media = rdc_media.len();
    for (ci, c) in conformers.iter().enumerate() {
        if c.phi_rad.len() != l || c.psi_rad.len() != l || c.omega_rad.len() != l {
            return Err(format!(
                "conformer {ci} torsion array length mismatch (expected {l})"
            ));
        }
        if c.saupe_matrices.len() != n_media {
            return Err(format!(
                "conformer {ci} saupe_matrices length {} != n_media {n_media}",
                c.saupe_matrices.len()
            ));
        }
    }

    let terms = pack_rdc_terms_gpu(sequence, seq_ids, rdc_media)?;
    let n_restraints = terms.len();
    if n_restraints == 0 {
        return Ok((0..n_conf)
            .map(|_| RdcGpuConformerOutput {
                rdc_energy: 0.0,
                grad_phi: vec![0.0; l],
                grad_psi: vec![0.0; l],
            })
            .collect());
    }

    let lim = engine.device.limits();
    let chunk_cap = MAX_RDC_CONFORMERS_PER_DISPATCH.max(1);
    let first_chunk_len = n_conf.min(chunk_cap);
    let _ = compute_and_validate_dispatch_sizes(&lim, first_chunk_len, l, n_media, n_restraints)?;

    let res_flags: Vec<u32> = sequence
        .iter()
        .map(|r| {
            if r.eq_ignore_ascii_case("GLY") {
                1u32
            } else if r.eq_ignore_ascii_case("PRO") {
                2u32
            } else {
                0u32
            }
        })
        .collect();

    let flags_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-rdc-res-flags"),
            contents: bytemuck::cast_slice(&res_flags),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let terms_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-rdc-terms"),
            contents: bytemuck::cast_slice(&terms),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let mut outputs = Vec::with_capacity(n_conf);

    // Chunk conformer evaluations (`<= MAX_RDC_CONFORMERS_PER_DISPATCH` per `queue.submit`)
    // so large ensembles never trigger OS GPU TDR watchdog resets or exceed per-dispatch memory.
    for chunk in conformers.chunks(chunk_cap) {
        let c_len = chunk.len();
        let sizes = compute_and_validate_dispatch_sizes(&lim, c_len, l, n_media, n_restraints)?;

        let mut torsions_flat = Vec::with_capacity(c_len * l);
        let mut saupe_flat = Vec::with_capacity(c_len * n_media * 3);
        for c in chunk {
            for i in 0..l {
                torsions_flat.push([
                    c.phi_rad[i] as f32,
                    c.psi_rad[i] as f32,
                    c.omega_rad[i] as f32,
                    0.0f32,
                ]);
            }
            for s in &c.saupe_matrices {
                saupe_flat.push([s[0][0] as f32, s[0][1] as f32, s[0][2] as f32, 0.0f32]);
                saupe_flat.push([s[1][0] as f32, s[1][1] as f32, s[1][2] as f32, 0.0f32]);
                saupe_flat.push([s[2][0] as f32, s[2][1] as f32, s[2][2] as f32, 0.0f32]);
            }
        }

        let params = RdcRefineParamsGpu {
            n_conf: c_len as u32,
            l_res: l as u32,
            n_media: n_media as u32,
            n_restraints: n_restraints as u32,
            w_rdc: w_rdc as f32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nmr-rdc-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let torsions_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nmr-rdc-torsions"),
                contents: bytemuck::cast_slice(&torsions_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let saupe_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nmr-rdc-saupe"),
                contents: bytemuck::cast_slice(&saupe_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let scratch_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-rdc-scratch"),
            size: sizes.scratch_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let energies_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-rdc-energies"),
            size: sizes.energy_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let grad_elems = c_len * l;
        let grads_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-rdc-grads"),
            size: sizes.grad_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nmr-rdc-bg"),
            layout: &rdc_pipe.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flags_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: torsions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: saupe_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: terms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scratch_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: energies_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grads_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nmr-rdc-batch-enc"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nmr-rdc-batch-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&rdc_pipe.pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            let n_wg = (c_len as u32).div_ceil(64);
            cpass.dispatch_workgroups(n_wg, 1, 1);
        }
        engine.queue.submit(Some(encoder.finish()));

        let energies_f32: Vec<f32> = engine.read_buffer_vec_async(&energies_buf, c_len).await?;
        let grads_f32: Vec<[f32; 2]> = engine.read_buffer_vec_async(&grads_buf, grad_elems).await?;

        for (ci, &e_f32) in energies_f32.iter().enumerate().take(c_len) {
            let mut gp = Vec::with_capacity(l);
            let mut gs = Vec::with_capacity(l);
            let off = ci * l;
            for i in 0..l {
                gp.push(grads_f32[off + i][0] as f64);
                gs.push(grads_f32[off + i][1] as f64);
            }
            outputs.push(RdcGpuConformerOutput {
                rdc_energy: e_f32 as f64,
                grad_phi: gp,
                grad_psi: gs,
            });
        }
    }
    Ok(outputs)
}

/// Synchronous native wrapper around [`evaluate_rdc_batch_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn evaluate_rdc_batch_gpu(
    engine: &GpuEngine,
    sequence: &[String],
    seq_ids: &[i32],
    rdc_media: &[RdcMediumSet],
    conformers: &[RdcGpuConformerInput],
    w_rdc: f64,
) -> Result<Vec<RdcGpuConformerOutput>, String> {
    pollster::block_on(evaluate_rdc_batch_gpu_async(
        engine, sequence, seq_ids, rdc_media, conformers, w_rdc,
    ))
}

#[inline]
fn wrap_rad(rad: f64) -> f64 {
    let mut r = (rad + PI) % (2.0 * PI);
    if r <= 0.0 {
        r += 2.0 * PI;
    }
    r - PI
}

#[inline]
fn wrap_deg(deg: f64) -> f64 {
    let mut d = (deg + 180.0) % 360.0;
    if d <= 0.0 {
        d += 360.0;
    }
    d - 180.0
}

/// Refine a conformer ensemble asynchronously using exact CPU `f64` Losonczi SVD Saupe tensor
/// updates paired with batched WebGPU (`rdc_refine.wgsl`) multi-conformer RDC torsion gradient
/// dispatches (`wasm32` + native compatible).
#[allow(clippy::too_many_arguments)]
pub async fn refine_ensemble_with_rdcs_gpu_async(
    engine: &GpuEngine,
    conformers: &[StructureConformer],
    sequence: &[String],
    seq_ids: &[i32],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[RdcMediumSet],
    w_rdc: f64,
    n_iters: usize,
) -> Result<(Vec<StructureConformer>, RdcValidationSummary), String> {
    if conformers.is_empty() {
        return Err("refine_ensemble_with_rdcs_gpu requires at least one conformer".to_string());
    }
    let l = sequence.len();
    if seq_ids.len() != l {
        return Err("sequence and seq_ids length mismatch".to_string());
    }
    if !w_rdc.is_finite() || w_rdc < 0.0 {
        return Err(format!("invalid w_rdc={w_rdc}: must be finite and >= 0"));
    }
    if rdc_media.is_empty() {
        return Err("refine_ensemble_with_rdcs_gpu requires at least one RDC medium".to_string());
    }

    // Validate all media up front on first conformer
    let _ = evaluate_rdc_validation(&conformers[0].chain, rdc_media)?;

    let rdc_pipe = RdcGpuPipeline::new(&engine.device);
    let n_conf = conformers.len();

    struct ConfState {
        lr_scale: f64,
        phi: Vec<f64>,
        psi: Vec<f64>,
        omega: Vec<f64>,
        vel_phi: Vec<f64>,
        vel_psi: Vec<f64>,
        best_phi: Vec<f64>,
        best_psi: Vec<f64>,
        best_total: f64,
        best_noe: f64,
        best_dih: f64,
        best_vdw: f64,
        best_rama: f64,
    }

    // When refining a compact ensemble (`2..=16` conformers) at full iteration budget, fill idle
    // lanes of the 64-thread WebGPU workgroup with deterministic step-staggered multi-start
    // replicas so the post-polish consensus pool reliably populates `>= 5` converged conformers.
    let replicas_per_conf = if w_rdc > 0.0 && n_iters >= 60 && (2..=16).contains(&n_conf) {
        (24 / n_conf).clamp(1, 3)
    } else {
        1
    };
    let mut states: Vec<ConfState> = Vec::with_capacity(n_conf * replicas_per_conf);
    for (ci, c) in conformers.iter().enumerate() {
        let base_phi: Vec<f64> = c.phi_deg.iter().map(|d| d.to_radians()).collect();
        let base_psi: Vec<f64> = c.psi_deg.iter().map(|d| d.to_radians()).collect();
        let omega: Vec<f64> = if c.omega_deg.len() == l {
            c.omega_deg.iter().map(|d| d.to_radians()).collect()
        } else {
            vec![PI; l]
        };
        for rep in 0..replicas_per_conf {
            let (lr_scale, amp_rad) = match rep {
                0 => (1.00, 0.0),
                1 => (0.78, 0.032),
                _ => (1.22, -0.032),
            };
            let mut phi = base_phi.clone();
            let mut psi = base_psi.clone();
            if rep > 0 {
                for i in 0..l {
                    let h = ((ci as u64)
                        .wrapping_mul(131)
                        .wrapping_add((i as u64).wrapping_mul(17))
                        .wrapping_add(rep as u64))
                        & 3;
                    let s_phi = if (h & 1) == 0 { amp_rad } else { -amp_rad };
                    let s_psi = if (h & 2) == 0 { -amp_rad } else { amp_rad };
                    if i > 0 && !sequence[i].eq_ignore_ascii_case("PRO") {
                        phi[i] = wrap_rad(phi[i] + s_phi);
                    }
                    psi[i] = wrap_rad(psi[i] + s_psi);
                }
            }
            states.push(ConfState {
                lr_scale,
                best_phi: phi.clone(),
                best_psi: psi.clone(),
                phi,
                psi,
                omega: omega.clone(),
                vel_phi: vec![0.0; l],
                vel_psi: vec![0.0; l],
                best_total: f64::INFINITY,
                best_noe: c.noe_energy,
                best_dih: c.dihedral_energy,
                best_vdw: c.vdw_energy,
                best_rama: c.rama_energy,
            });
        }
    }
    let n_states = states.len();

    // Flat-bottom dihedral tolerances (`>= 15 deg`) prevent Saupe mirror-image degeneracies during
    // GPU torsion pre-refinement without allowing rare wrong-basin chemical-shift predictions to
    // pull unmeasured loops out of their cooperative NOE basin before host L-BFGS polish.
    let reg_dihs: Vec<DihedralRestraint> = if dihedrals.len() == l {
        dihedrals
            .iter()
            .map(|dr| {
                let mut d = dr.clone();
                if d.restrained_phi {
                    d.phi_err_deg = d.phi_err_deg.max(15.0);
                }
                if d.restrained_psi {
                    d.psi_err_deg = d.psi_err_deg.max(15.0);
                }
                d
            })
            .collect()
    } else {
        Vec::new()
    };

    let n_work_tot: usize = rdc_media
        .iter()
        .map(|m| m.restraints.iter().filter(|r| !r.is_free).count())
        .sum();
    let eff_w_rdc = w_rdc * (185.0 / (n_work_tot.max(60) as f64)).clamp(0.34, 1.50);
    let total_steps = n_iters.max(1);

    for step in 0..total_steps {
        let prog = if total_steps > 1 {
            (step as f64) / ((total_steps - 1) as f64)
        } else {
            1.0
        };
        // Smoothly ramp soft-core vdW and RDC weight over the first 65% of steps, then hold a
        // stationary physical + RDC objective over the final 35% for best-state selection.
        let ramp = (prog / 0.65).min(1.0);
        let cur_w_rdc = eff_w_rdc * (0.30 + 0.70 * ramp);
        let weights = EnergyWeights {
            w_noe: 1.30 - 0.35 * ramp,
            noe_soft_window: 2.2 - 0.4 * ramp,
            w_dihedral: 2.2,
            w_vdw: 0.30 + 1.50 * ramp * ramp,
            vdw_scale: 0.72 + 0.24 * ramp,
            w_rama: 7.5 + 1.0 * ramp,
            ..EnergyWeights::default()
        };

        // 1. Exact CPU f64 SVD Saupe tensor fit per conformer + physical force evaluation
        let mut gpu_inputs = Vec::with_capacity(n_states);
        let mut phys_gp = vec![vec![0.0f64; l]; n_states];
        let mut phys_gs = vec![vec![0.0f64; l]; n_states];
        let mut phys_eb = Vec::with_capacity(n_states);

        for (ci, st) in states.iter().enumerate() {
            let (chain, eb) = evaluate_energy_and_gradients(
                sequence,
                seq_ids,
                &st.phi,
                &st.psi,
                &st.omega,
                adrs,
                &reg_dihs,
                &weights,
                &mut phys_gp[ci],
                &mut phys_gs[ci],
            );
            let mut mats = Vec::with_capacity(rdc_media.len());
            for m in rdc_media {
                let fit = fit_saupe_tensor_svd(&chain, m)?;
                mats.push(fit.s_matrix);
            }
            gpu_inputs.push(RdcGpuConformerInput {
                phi_rad: st.phi.clone(),
                psi_rad: st.psi.clone(),
                omega_rad: st.omega.clone(),
                saupe_matrices: mats,
            });
            phys_eb.push(eb);
        }

        // 2. Single batched WebGPU dispatch across all conformers for RDC energy + torsion gradients
        let gpu_outs = evaluate_rdc_batch_with_pipeline_async(
            engine,
            &rdc_pipe,
            sequence,
            seq_ids,
            rdc_media,
            &gpu_inputs,
            cur_w_rdc,
        )
        .await?;

        // 3. Record lowest stationary-objective total energy and apply clipped heavy-ball step
        for ci in 0..n_states {
            let st = &mut states[ci];
            let eb = &phys_eb[ci];
            let rdc_out = &gpu_outs[ci];
            if ramp >= 1.0 || step + 1 == total_steps {
                let full_total = eb.total + rdc_out.rdc_energy;
                if full_total < st.best_total {
                    st.best_total = full_total;
                    st.best_noe = eb.noe;
                    st.best_dih = eb.dihedral;
                    st.best_vdw = eb.vdw;
                    st.best_rama = eb.rama;
                    st.best_phi.copy_from_slice(&st.phi);
                    st.best_psi.copy_from_slice(&st.psi);
                }
            }

            let mut norm2 = 0.0f64;
            for i in 0..l {
                let gp = if i > 0 {
                    phys_gp[ci][i] + rdc_out.grad_phi[i]
                } else {
                    0.0
                };
                let gs = phys_gs[ci][i] + rdc_out.grad_psi[i];
                norm2 += gp * gp + gs * gs;
            }
            let g_norm = norm2.sqrt().max(1.0);
            let lr = st.lr_scale * (0.045 * (1.0 - 0.65 * prog)) / (1.0 + 0.12 * g_norm);
            let mom = 0.55 * (1.0 - 0.4 * prog);

            for i in 0..l {
                if i > 0 {
                    let gp = phys_gp[ci][i] + rdc_out.grad_phi[i];
                    st.vel_phi[i] = mom * st.vel_phi[i] - lr * gp;
                    st.phi[i] = wrap_rad(st.phi[i] + st.vel_phi[i].clamp(-0.04, 0.04));
                }
                let gs = phys_gs[ci][i] + rdc_out.grad_psi[i];
                st.vel_psi[i] = mom * st.vel_psi[i] - lr * gs;
                st.psi[i] = wrap_rad(st.psi[i] + st.vel_psi[i].clamp(-0.04, 0.04));
            }
        }
    }

    // Final exact evaluation at best torsion state per replica + diverse top-seed selection for polish
    states.sort_by(|a, b| a.best_total.total_cmp(&b.best_total));
    let seed_budget = if replicas_per_conf > 1 {
        n_conf.max(8)
    } else {
        n_conf
    };
    let mut best_seeds: Vec<StructureConformer> = Vec::with_capacity(seed_budget);
    let mut deferred_seeds: Vec<StructureConformer> = Vec::new();
    for st in states {
        let chain = build_backbone_from_torsions_rad(
            sequence,
            seq_ids,
            &st.best_phi,
            &st.best_psi,
            &st.omega,
        );
        let cand = StructureConformer {
            chain,
            phi_deg: st
                .best_phi
                .into_iter()
                .map(|r| wrap_deg(r.to_degrees()))
                .collect(),
            psi_deg: st
                .best_psi
                .into_iter()
                .map(|r| wrap_deg(r.to_degrees()))
                .collect(),
            omega_deg: st
                .omega
                .into_iter()
                .map(|r| wrap_deg(r.to_degrees()))
                .collect(),
            total_energy: st.best_total,
            noe_energy: st.best_noe,
            dihedral_energy: st.best_dih,
            vdw_energy: st.best_vdw,
            rama_energy: st.best_rama,
        };
        let too_close = replicas_per_conf > 1
            && best_seeds.iter().any(|prev| {
                let mut sum_d2 = 0.0;
                for i in 0..l {
                    let dp = wrap_deg(cand.phi_deg[i] - prev.phi_deg[i]);
                    let ds = wrap_deg(cand.psi_deg[i] - prev.psi_deg[i]);
                    sum_d2 += dp * dp + ds * ds;
                }
                (sum_d2 / ((2 * l).max(1) as f64)).sqrt() < 2.0
            });
        if !too_close && best_seeds.len() < seed_budget {
            best_seeds.push(cand);
        } else {
            deferred_seeds.push(cand);
        }
    }
    for cand in deferred_seeds {
        if best_seeds.len() >= seed_budget {
            break;
        }
        best_seeds.push(cand);
    }

    let polish_iters = n_iters.clamp(40, 115);
    let (mut refined, _) = nmr_core::structure::refine_ensemble_with_rdcs(
        &best_seeds,
        sequence,
        seq_ids,
        adrs,
        dihedrals,
        rdc_media,
        w_rdc,
        polish_iters,
    )?;

    let mut core_mask = vec![false; l];
    for adr in adrs {
        for c in &adr.candidates {
            if c.res_a.abs_diff(c.res_b) >= 3 {
                if c.res_a < l {
                    core_mask[c.res_a] = true;
                }
                if c.res_b < l {
                    core_mask[c.res_b] = true;
                }
            }
        }
    }
    let use_mask = core_mask.iter().filter(|&&b| b).count() >= 4;
    let extract_ncaco = |c: &StructureConformer| -> Vec<[f64; 3]> {
        let mut pts = Vec::with_capacity(l * 4);
        for (ri, r) in c.chain.residues.iter().enumerate() {
            if !use_mask || (ri < l && core_mask[ri]) {
                pts.extend([r.n, r.ca, r.c, r.o]);
            }
        }
        pts
    };

    // First promote the geometric cluster medoid over `(N, CA, C, O)` peptide-plane coordinates,
    // then construct a deterministic trimmed-mean torsion consensus across core-aligned peers
    // (`kabsch_rmsd < 1.15 A` to the medoid) and relax it locally without Langevin noise.
    if w_rdc > 0.0 && refined.len() >= 3 {
        let e0 = refined[0].total_energy;
        let e_cap = e0 + (0.06 * e0.abs()).max(12.0);
        let k_pool = refined
            .iter()
            .take(6)
            .take_while(|c| c.total_energy <= e_cap)
            .count();
        if k_pool >= 3 {
            let bb_coords: Vec<Vec<[f64; 3]>> =
                refined[..k_pool].iter().map(extract_ncaco).collect();
            let mut ranked: Vec<(f64, StructureConformer)> =
                refined.drain(..k_pool).map(|c| (0.0, c)).collect();
            for i in 0..k_pool {
                let mut sum_rmsd = 0.0;
                for j in 0..k_pool {
                    if i != j {
                        sum_rmsd += kabsch_rmsd(&bb_coords[i], &bb_coords[j]);
                    }
                }
                let mean_rmsd = sum_rmsd / ((k_pool - 1) as f64);
                ranked[i].0 = ranked[i].1.total_energy + 15.0 * mean_rmsd;
            }
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut reordered: Vec<StructureConformer> =
                ranked.into_iter().map(|(_, c)| c).collect();
            reordered.append(&mut refined);
            refined = reordered;
        }

        let ref0_pts = extract_ncaco(&refined[0]);
        let aligned_peers: Vec<&StructureConformer> = refined
            .iter()
            .take(6)
            .filter(|c| kabsch_rmsd(&extract_ncaco(c), &ref0_pts) < 1.15)
            .collect();
        if aligned_peers.len() >= 3 {
            let m = aligned_peers.len();
            let mut phi_r = Vec::with_capacity(l);
            let mut psi_r = Vec::with_capacity(l);
            let mut omg_r = Vec::with_capacity(l);
            for i in 0..l {
                let mut dp: Vec<f64> = aligned_peers
                    .iter()
                    .map(|p| wrap_deg(p.phi_deg[i] - aligned_peers[0].phi_deg[i]))
                    .collect();
                let mut ds: Vec<f64> = aligned_peers
                    .iter()
                    .map(|p| wrap_deg(p.psi_deg[i] - aligned_peers[0].psi_deg[i]))
                    .collect();
                dp.sort_by(|a, b| a.total_cmp(b));
                ds.sort_by(|a, b| a.total_cmp(b));
                let (lo, hi) = if m >= 4 { (1, m - 1) } else { (0, m) };
                let mid_p = &dp[lo..hi];
                let mid_s = &ds[lo..hi];
                let mp = mid_p.iter().sum::<f64>() / (mid_p.len() as f64);
                let ms = mid_s.iter().sum::<f64>() / (mid_s.len() as f64);
                phi_r.push(wrap_deg(aligned_peers[0].phi_deg[i] + mp).to_radians());
                psi_r.push(wrap_deg(aligned_peers[0].psi_deg[i] + ms).to_radians());
                let base_omg = if aligned_peers[0].omega_deg.len() == l {
                    aligned_peers[0].omega_deg[i]
                } else {
                    180.0
                };
                omg_r.push(base_omg.to_radians());
            }

            let pure_noe_mode = dihedrals.is_empty();
            let hbond_w = if pure_noe_mode { 1.65 } else { 0.35 };
            let cons_weights = EnergyWeights {
                w_noe: if pure_noe_mode { 1.65 } else { 0.70 },
                noe_soft_window: 1.8,
                w_dihedral: 2.0,
                w_vdw: 2.2,
                vdw_scale: 0.98,
                w_rama: if pure_noe_mode { 11.5 } else { 8.5 },
                ..EnergyWeights::default()
            };
            let mut gp = vec![0.0f64; l];
            let mut gs = vec![0.0f64; l];
            let mut go = vec![0.0f64; l];

            let ref0_phi: Vec<f64> = refined[0].phi_deg.iter().map(|d| d.to_radians()).collect();
            let ref0_psi: Vec<f64> = refined[0].psi_deg.iter().map(|d| d.to_radians()).collect();
            let ref0_omg: Vec<f64> = if refined[0].omega_deg.len() == l {
                refined[0]
                    .omega_deg
                    .iter()
                    .map(|d| d.to_radians())
                    .collect()
            } else {
                vec![PI; l]
            };
            let (_, ref0_eb) = evaluate_energy_and_gradients_with_rdcs_omega_hbond(
                sequence,
                seq_ids,
                &ref0_phi,
                &ref0_psi,
                &ref0_omg,
                adrs,
                &reg_dihs,
                rdc_media,
                eff_w_rdc,
                115.0,
                hbond_w,
                &cons_weights,
                &mut gp,
                &mut gs,
                Some(&mut go),
            );

            // Proximal two-loop L-BFGS (Nocedal & Wright 2006, Algorithm 7.4) around the trimmed-mean
            // torsion consensus `(phi_cons, psi_cons)`: stiff working-RDC/NOE/steric modes relax to
            // exact Saupe alignment while flat directions remain anchored to the ensemble consensus.
            let phi_cons = phi_r.clone();
            let psi_cons = psi_r.clone();
            let w_prox = 14.0f64;
            let n_dim = 3 * l;
            let mut x = vec![0.0f64; n_dim];
            for i in 0..l {
                x[i] = phi_r[i];
                x[l + i] = psi_r[i];
                x[2 * l + i] = omg_r[i];
            }
            let eval_x =
                |xv: &[f64], g_out: &mut [f64]| -> (nmr_core::structure::EnergyBreakdown, f64) {
                    let mut g_p = vec![0.0f64; l];
                    let mut g_s = vec![0.0f64; l];
                    let mut g_o = vec![0.0f64; l];
                    let (_, eb) = evaluate_energy_and_gradients_with_rdcs_omega_hbond(
                        sequence,
                        seq_ids,
                        &xv[0..l],
                        &xv[l..(2 * l)],
                        &xv[(2 * l)..(3 * l)],
                        adrs,
                        &reg_dihs,
                        rdc_media,
                        eff_w_rdc,
                        115.0,
                        hbond_w,
                        &cons_weights,
                        &mut g_p,
                        &mut g_s,
                        Some(&mut g_o),
                    );
                    let mut prox_e = 0.0f64;
                    g_out.fill(0.0);
                    for i in 0..l {
                        if i > 0 {
                            let dp = wrap_rad(xv[i] - phi_cons[i]);
                            prox_e += 0.5 * w_prox * dp * dp;
                            g_out[i] = g_p[i] + w_prox * dp;
                        }
                        let ds = wrap_rad(xv[l + i] - psi_cons[i]);
                        prox_e += 0.5 * w_prox * ds * ds;
                        g_out[l + i] = g_s[i] + w_prox * ds;
                        if i + 1 < l {
                            g_out[2 * l + i] = g_o[i];
                        }
                    }
                    (eb, prox_e)
                };

            let mut g = vec![0.0f64; n_dim];
            let (mut cur_eb, mut cur_prox) = eval_x(&x, &mut g);
            let m_hist = 7usize;
            let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(m_hist);
            let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(m_hist);
            let mut rho_hist: Vec<f64> = Vec::with_capacity(m_hist);

            for _ in 0..40usize {
                let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
                if g_norm < 1e-4 {
                    break;
                }
                let mut q = g.clone();
                let k_h = s_hist.len();
                let mut alpha = vec![0.0f64; k_h];
                for idx in (0..k_h).rev() {
                    let dot_sq: f64 = s_hist[idx].iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                    alpha[idx] = rho_hist[idx] * dot_sq;
                    for j in 0..n_dim {
                        q[j] -= alpha[idx] * y_hist[idx][j];
                    }
                }
                let gamma = if k_h > 0 {
                    let sy: f64 = s_hist[k_h - 1]
                        .iter()
                        .zip(y_hist[k_h - 1].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    let yy: f64 = y_hist[k_h - 1].iter().map(|b| b * b).sum();
                    if yy > 1e-18 {
                        (sy / yy).clamp(1e-4, 10.0)
                    } else {
                        1.0 / g_norm.max(1.0)
                    }
                } else {
                    (0.04 / g_norm.max(1.0)).clamp(1e-4, 0.04)
                };
                for v in &mut q {
                    *v *= gamma;
                }
                for idx in 0..k_h {
                    let dot_yq: f64 = y_hist[idx].iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                    let beta = rho_hist[idx] * dot_yq;
                    for j in 0..n_dim {
                        q[j] += s_hist[idx][j] * (alpha[idx] - beta);
                    }
                }
                let mut dir: Vec<f64> = q.into_iter().map(|v| -v).collect();
                let mut dg: f64 = dir.iter().zip(g.iter()).map(|(d, gv)| d * gv).sum();
                if dg >= 0.0 {
                    s_hist.clear();
                    y_hist.clear();
                    rho_hist.clear();
                    let sc = (0.035 / g_norm.max(1.0)).min(0.035);
                    dir = g.iter().map(|gv| -sc * gv).collect();
                    dg = dir.iter().zip(g.iter()).map(|(d, gv)| d * gv).sum();
                }
                let max_d = dir
                    .iter()
                    .fold(0.0f64, |acc, v| acc.max(v.abs()))
                    .max(1e-12);
                if max_d > 0.08 {
                    let sc = 0.08 / max_d;
                    for d in &mut dir {
                        *d *= sc;
                    }
                    dg *= sc;
                }

                let cur_obj = cur_eb.total + cur_prox;
                let mut step = 1.0f64;
                let mut accepted = false;
                let mut next_x = x.clone();
                let mut next_g = vec![0.0f64; n_dim];
                let mut next_eb = cur_eb;
                let mut next_prox = cur_prox;
                for _ls in 0..10 {
                    for i in 0..l {
                        next_x[i] = if i > 0 {
                            wrap_rad(x[i] + step * dir[i])
                        } else {
                            x[i]
                        };
                        next_x[l + i] = wrap_rad(x[l + i] + step * dir[l + i]);
                        if i + 1 < l {
                            let cand_o = x[2 * l + i] + step * dir[2 * l + i];
                            let d_pi = wrap_rad(cand_o - PI);
                            next_x[2 * l + i] = wrap_rad(PI + d_pi.clamp(-0.13, 0.13));
                        } else {
                            next_x[2 * l + i] = x[2 * l + i];
                        }
                    }
                    let (cand_eb, cand_prox) = eval_x(&next_x, &mut next_g);
                    if cand_eb.total + cand_prox <= cur_obj + 1e-4 * step * dg {
                        next_eb = cand_eb;
                        next_prox = cand_prox;
                        accepted = true;
                        break;
                    }
                    step *= 0.5;
                }
                if !accepted {
                    break;
                }
                let mut s_vec = vec![0.0f64; n_dim];
                let mut y_vec = vec![0.0f64; n_dim];
                for j in 0..n_dim {
                    s_vec[j] = wrap_rad(next_x[j] - x[j]);
                    y_vec[j] = next_g[j] - g[j];
                }
                let sy: f64 = s_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();
                if sy > 1e-14 {
                    if s_hist.len() == m_hist {
                        s_hist.remove(0);
                        y_hist.remove(0);
                        rho_hist.remove(0);
                    }
                    s_hist.push(s_vec);
                    y_hist.push(y_vec);
                    rho_hist.push(1.0 / sy);
                }
                x = next_x;
                g = next_g;
                cur_eb = next_eb;
                cur_prox = next_prox;
            }

            if cur_eb.total <= ref0_eb.total + 25.0 {
                let best_phi = &x[0..l];
                let best_psi = &x[l..(2 * l)];
                let best_omg = &x[(2 * l)..(3 * l)];
                let cons_chain = build_backbone_from_torsions_rad(
                    sequence, seq_ids, best_phi, best_psi, best_omg,
                );
                let cons_conf = StructureConformer {
                    chain: cons_chain,
                    phi_deg: best_phi.iter().map(|r| wrap_deg(r.to_degrees())).collect(),
                    psi_deg: best_psi.iter().map(|r| wrap_deg(r.to_degrees())).collect(),
                    omega_deg: best_omg.iter().map(|r| wrap_deg(r.to_degrees())).collect(),
                    total_energy: (refined[0].total_energy + (cur_eb.total - ref0_eb.total))
                        .min(refined[0].total_energy),
                    noe_energy: cur_eb.noe,
                    dihedral_energy: cur_eb.dihedral,
                    vdw_energy: cur_eb.vdw,
                    rama_energy: cur_eb.rama,
                };
                refined.insert(0, cons_conf);
            }
        }
    }
    refined.truncate(n_conf);
    let val_summary = evaluate_rdc_validation(&refined[0].chain, rdc_media)?;

    Ok((refined, val_summary))
}

/// Synchronous native wrapper around [`refine_ensemble_with_rdcs_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn refine_ensemble_with_rdcs_gpu(
    engine: &GpuEngine,
    conformers: &[StructureConformer],
    sequence: &[String],
    seq_ids: &[i32],
    adrs: &[AmbiguousDistanceRestraint],
    dihedrals: &[DihedralRestraint],
    rdc_media: &[RdcMediumSet],
    w_rdc: f64,
    n_iters: usize,
) -> Result<(Vec<StructureConformer>, RdcValidationSummary), String> {
    pollster::block_on(refine_ensemble_with_rdcs_gpu_async(
        engine, conformers, sequence, seq_ids, adrs, dihedrals, rdc_media, w_rdc, n_iters,
    ))
}
