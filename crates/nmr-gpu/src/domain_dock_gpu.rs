//! Batched multi-replica WebGPU (`wgpu` + WGSL) multi-domain `RDC` / `PCS` / `PRE` / `NOE`
//! covalent linker-torsion and $\text{SE}(3)$ docking engine (`T21.5`,
//! `docs/specs/21-alphafold-integrative-nmr.md` §5.3).
//!
//! Provenance: Clean-room GPU compute implementation derived strictly from peer-reviewed literature
//! (`0` production `.unwrap()` / `.expect()`, `0` Dawn/Tint const-eval `bitcast`s):
//! - Ng, A. Y., Jordan, M. I. & Weiss, Y. (*Adv. Neural Inf. Process. Syst.* **2001**, 14, 849-856).
//! - Tolman, J. R., Al-Hashimi, H. M., Kay, L. E. & Prestegard, J. H. (*J. Am. Chem. Soc.* **2001**,
//!   123, 1416-1424).
//! - Braddock, D. T. et al. (*J. Am. Chem. Soc.* **2001**, 123, 8634-8635); Clore, G. M. &
//!   Schwieters, C. D. (*J. Am. Chem. Soc.* **2004**, 126, 2923-2938).
//! - Bertini, I. et al. (*J. Am. Chem. Soc.* **2007**, 129, 12786-12794); Schmitz, C. et al.
//!   (*J. Biomol. NMR* **2008**, 41, 179-189).
//! - Nilges, M. (*J. Mol. Biol.* **1995**, 245, 645-660 — $r^{-6}$-summed ambiguous NOE restraints).
//! - Parsons, J. et al. (*J. Comput. Chem.* **2005**, 26, 1063-1068 — `NeRF`); Abe, H., Braun, W.,
//!   Noguti, T. & Go, N. (*Comput. Chem.* **1984**, 8, 239-247 — tree suffix-sum torque gradient);
//!   Nocedal, J. (*Math. Comp.* **1980**, 35, 151, 773-782 — two-loop `L-BFGS`).

use std::f64::consts::PI;

use bytemuck::{Pod, Zeroable};
use nmr_core::af_concordance::{AfStructureModel, PaeMatrix};
use nmr_core::domain_dock::{
    cluster_pae_domains, cluster_pae_domains_for_model, evaluate_domain_dock_state_cpu,
    finalize_domain_dock_pipeline, generate_domain_dock_seed_states,
    prepare_domain_dock_inputs_full, quat_apply_world_rot_step, select_and_polish_top_states,
    DockResClass, DomainDockConfig, DomainDockEnergyBreakdown, DomainDockEnergyWeights,
    DomainDockGradient, DomainDockResult, DomainDockState, DomainLinkerKinematics, DomainPcsTagSet,
    DomainPreSiteSet, DomainRdcMediumSet, PaeDomainPartition, PreparedDomainDockRestraints,
};
use nmr_core::noesy::{v_dot, AmbiguousDistanceRestraint};
use nmr_core::rdc::RdcMediumSet;
use wgpu::util::DeviceExt;

use crate::GpuEngine;

/// Maximum number of multi-domain replicas evaluated per single WebGPU compute dispatch /
/// `queue.submit` so TDR watchdogs are never tripped (`R <= 256`).
pub const MAX_DOMAIN_DOCK_REPLICAS_PER_DISPATCH: usize = 256;

/// Explicit residue-count ceiling for WebGPU `domain_dock.wgsl` (`L <= 4096`, `21.W2:P2-13`),
/// ensuring `f32` exact integer representation inside the `O(L)` 3D spatial voxel grid hash.
pub const MAX_DOMAIN_DOCK_GPU_RESIDUES: usize = 4096;

const NO_BLOCK_SENTINEL: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
struct DomainDockDispatchSizes {
    scratch_bytes: u64,
    energies_bytes: u64,
    grads_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct DomainDockParamsGpu {
    n_replicas: u32,
    l_res: u32,
    n_rigid_blocks: u32,
    n_chains: u32,
    n_media: u32,
    n_rdc_terms: u32,
    n_pcs_tags: u32,
    n_pcs_terms: u32,
    n_pre_sites: u32,
    n_pre_terms: u32,
    n_noe_terms: u32,
    w_rdc: f32,
    w_pcs: f32,
    w_pre: f32,
    w_noe: f32,
    w_vdw: f32,
    vdw_scale: f32,
    w_rama: f32,
    noe_soft_window: f32,
    n_noe_cands: u32,
}

#[inline]
fn vdw_grid_cap_for_residues(l_res: usize) -> Result<usize, String> {
    if l_res == 0 {
        return Ok(64);
    }
    l_res
        .checked_mul(16)
        .and_then(usize::checked_next_power_of_two)
        .map(|c| c.max(64))
        .ok_or_else(|| "domain_dock_gpu vdw grid_cap overflow".to_string())
}

/// Pre-check all 7 WebGPU bindings (`1` uniform + `6` storage buffers), `L <= 4096`, and `R <= 256`
/// TDR ceiling against `wgpu::Limits` prior to creating any GPU buffer.
pub fn validate_domain_dock_gpu_limits(
    limits: &wgpu::Limits,
    n_replicas_dispatch: usize,
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
) -> Result<(), String> {
    let _ = compute_and_validate_domain_dock_sizes_full(
        limits,
        n_replicas_dispatch,
        kin.residues.len(),
        kin.rigid_blocks.len(),
        kin.chain_ranges.len(),
        prep.n_media,
        prep.rdc_terms.len(),
        prep.pcs_tags.len(),
        prep.pcs_terms.len(),
        prep.pre_sites.len(),
        prep.pre_terms.len(),
        prep.noe_terms.len(),
        prep.noe_candidates.len(),
    )?;
    Ok(())
}

/// Raw dimension limit validator for `domain_dock_gpu` (callable directly in adversarial limit tests).
#[allow(clippy::too_many_arguments)]
pub fn validate_domain_dock_gpu_limits_raw(
    limits: &wgpu::Limits,
    n_replicas_dispatch: usize,
    l_res: usize,
    n_rigid_blocks: usize,
    n_chains: usize,
    n_media: usize,
    n_rdc_terms: usize,
    n_pcs_tags: usize,
    n_pcs_terms: usize,
    n_pre_sites: usize,
    n_pre_terms: usize,
    n_noe_terms: usize,
) -> Result<(), String> {
    let _ = compute_and_validate_domain_dock_sizes_full(
        limits,
        n_replicas_dispatch,
        l_res,
        n_rigid_blocks,
        n_chains,
        n_media,
        n_rdc_terms,
        n_pcs_tags,
        n_pcs_terms,
        n_pre_sites,
        n_pre_terms,
        n_noe_terms,
        n_noe_terms,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compute_and_validate_domain_dock_sizes_full(
    limits: &wgpu::Limits,
    n_replicas_dispatch: usize,
    l_res: usize,
    n_rigid_blocks: usize,
    n_chains: usize,
    n_media: usize,
    n_rdc_terms: usize,
    n_pcs_tags: usize,
    n_pcs_terms: usize,
    n_pre_sites: usize,
    n_pre_terms: usize,
    n_noe_terms: usize,
    n_noe_cands: usize,
) -> Result<DomainDockDispatchSizes, String> {
    if n_replicas_dispatch == 0 {
        return Err("domain_dock_gpu dispatch requires n_replicas_dispatch >= 1".to_string());
    }
    if n_replicas_dispatch > MAX_DOMAIN_DOCK_REPLICAS_PER_DISPATCH {
        return Err(format!(
            "domain_dock_gpu n_replicas_dispatch ({n_replicas_dispatch}) exceeds TDR cap MAX_DOMAIN_DOCK_REPLICAS_PER_DISPATCH ({MAX_DOMAIN_DOCK_REPLICAS_PER_DISPATCH})"
        ));
    }
    if l_res > MAX_DOMAIN_DOCK_GPU_RESIDUES {
        return Err(format!(
            "domain_dock_gpu l_res ({l_res}) exceeds MAX_DOMAIN_DOCK_GPU_RESIDUES ({MAX_DOMAIN_DOCK_GPU_RESIDUES})"
        ));
    }
    if limits.max_storage_buffers_per_shader_stage < 6 {
        return Err(format!(
            "device max_storage_buffers_per_shader_stage ({}) < 6 required by domain_dock.wgsl",
            limits.max_storage_buffers_per_shader_stage
        ));
    }

    for &(nm, val) in &[
        ("l_res", l_res),
        ("n_rigid_blocks", n_rigid_blocks),
        ("n_chains", n_chains),
        ("n_media", n_media),
        ("n_rdc_terms", n_rdc_terms),
        ("n_pcs_tags", n_pcs_tags),
        ("n_pcs_terms", n_pcs_terms),
        ("n_pre_sites", n_pre_sites),
        ("n_pre_terms", n_pre_terms),
        ("n_noe_terms", n_noe_terms),
        ("n_noe_cands", n_noe_cands),
    ] {
        if val > u32::MAX as usize {
            return Err(format!(
                "domain_dock_gpu {nm} ({val}) exceeds 32-bit WGSL u32 index bound"
            ));
        }
    }

    let n_wg = n_replicas_dispatch.div_ceil(64);
    if n_wg > limits.max_compute_workgroups_per_dimension as usize {
        return Err(format!(
            "domain_dock_gpu workgroups ({n_wg}) exceed max_compute_workgroups_per_dimension ({})",
            limits.max_compute_workgroups_per_dimension
        ));
    }

    let check_storage = |name: &str, bytes: u64| -> Result<(), String> {
        if bytes > limits.max_buffer_size {
            return Err(format!(
                "domain_dock_gpu {name} ({bytes} B) exceeds device max_buffer_size ({} B)",
                limits.max_buffer_size
            ));
        }
        let max_sb = limits.max_storage_buffer_binding_size;
        if bytes > max_sb {
            return Err(format!(
                "domain_dock_gpu {name} ({bytes} B) exceeds device max_storage_buffer_binding_size ({max_sb} B)"
            ));
        }
        Ok(())
    };

    let params_bytes = std::mem::size_of::<DomainDockParamsGpu>() as u64;
    if params_bytes > limits.max_buffer_size
        || params_bytes > limits.max_uniform_buffer_binding_size
    {
        return Err(format!(
            "domain_dock_gpu params_buf ({params_bytes} B) exceeds uniform/buffer limits"
        ));
    }

    let add_u = |a: usize, b: usize, msg: &str| -> Result<usize, String> {
        a.checked_add(b)
            .ok_or_else(|| format!("domain_dock_gpu {msg} overflow"))
    };
    let mul_u = |a: usize, b: usize, msg: &str| -> Result<usize, String> {
        a.checked_mul(b)
            .ok_or_else(|| format!("domain_dock_gpu {msg} overflow"))
    };

    // 1. `static_floats` (`vec4<f32>` = 16 B)
    let sf_res = mul_u(13, l_res, "static_floats res")?;
    let sf_tags = mul_u(4, n_pcs_tags, "static_floats pcs_tags")?;
    let sf_pre = mul_u(2, n_pre_terms, "static_floats pre_terms")?;
    let mut sf_vec4s = add_u(sf_res, n_chains, "sf + chains")?;
    sf_vec4s = add_u(sf_vec4s, sf_tags, "sf + tags")?;
    sf_vec4s = add_u(sf_vec4s, n_pre_sites, "sf + pre_sites")?;
    sf_vec4s = add_u(sf_vec4s, n_rdc_terms, "sf + rdc")?;
    sf_vec4s = add_u(sf_vec4s, n_pcs_terms, "sf + pcs")?;
    sf_vec4s = add_u(sf_vec4s, sf_pre, "sf + pre")?;
    sf_vec4s = add_u(sf_vec4s, n_noe_terms, "sf + noe")?;
    if sf_vec4s > u32::MAX as usize {
        return Err("domain_dock_gpu static_floats u32 index overflow".to_string());
    }
    let static_floats_bytes = (sf_vec4s as u64)
        .checked_mul(16)
        .ok_or_else(|| "domain_dock_gpu static_floats byte overflow".to_string())?
        .max(16);
    check_storage("static_floats_buf", static_floats_bytes)?;

    // 2. `static_ints` (`u32` = 4 B)
    let mut si_u32s = mul_u(4, l_res, "static_ints res")?;
    si_u32s = add_u(si_u32s, mul_u(2, n_chains, "si chains")?, "si + chains")?;
    si_u32s = add_u(
        si_u32s,
        mul_u(4, n_rigid_blocks, "si blocks")?,
        "si + blocks",
    )?;
    si_u32s = add_u(si_u32s, mul_u(2, n_pcs_tags, "si tags")?, "si + tags")?;
    si_u32s = add_u(si_u32s, mul_u(2, n_pre_sites, "si sites")?, "si + sites")?;
    si_u32s = add_u(si_u32s, mul_u(5, n_rdc_terms, "si rdc")?, "si + rdc")?;
    si_u32s = add_u(si_u32s, mul_u(3, n_pcs_terms, "si pcs")?, "si + pcs")?;
    si_u32s = add_u(si_u32s, mul_u(4, n_pre_terms, "si pre")?, "si + pre")?;
    si_u32s = add_u(
        si_u32s,
        mul_u(2, n_noe_terms, "si noe_terms")?,
        "si + noe_terms",
    )?;
    si_u32s = add_u(
        si_u32s,
        mul_u(4, n_noe_cands, "si noe_cands")?,
        "si + noe_cands",
    )?;
    if si_u32s > u32::MAX as usize {
        return Err("domain_dock_gpu static_ints u32 index overflow".to_string());
    }
    let static_ints_bytes = (si_u32s as u64)
        .checked_mul(4)
        .ok_or_else(|| "domain_dock_gpu static_ints byte overflow".to_string())?
        .max(16);
    check_storage("static_ints_buf", static_ints_bytes)?;

    // 3. `replica_in` (`vec4<f32>` = 16 B)
    let in_stride = add_u(
        add_u(l_res, mul_u(2, n_chains, "in chains")?, "in l+c")?,
        mul_u(3, n_media, "in media")?,
        "in stride",
    )?;
    let in_vec4s = mul_u(n_replicas_dispatch, in_stride, "replica_in count")?;
    if in_vec4s > u32::MAX as usize {
        return Err("domain_dock_gpu replica_in u32 index overflow".to_string());
    }
    let replica_in_bytes = (in_vec4s as u64)
        .checked_mul(16)
        .ok_or_else(|| "domain_dock_gpu replica_in byte overflow".to_string())?
        .max(16);
    check_storage("replica_in_buf", replica_in_bytes)?;

    // 4. `scratch` (`vec4<f32>` = 16 B: `7*l + 4*blocks + 2*segs + 10*l + grid_cap`)
    let n_segs = add_u(mul_u(2, l_res, "2*l")?, 1, "n_segs")?;
    let grid_cap = vdw_grid_cap_for_residues(l_res)?;
    let mut sc_stride = add_u(
        mul_u(7, l_res, "7*l")?,
        mul_u(4, n_rigid_blocks, "4*blocks")?,
        "sc atoms+blocks",
    )?;
    sc_stride = add_u(sc_stride, mul_u(2, n_segs, "2*segs")?, "sc + segs")?;
    sc_stride = add_u(sc_stride, mul_u(10, l_res, "10*l vdw")?, "sc + vdw")?;
    sc_stride = add_u(sc_stride, grid_cap, "sc + grid_cap")?;

    let sc_vec4s = mul_u(n_replicas_dispatch, sc_stride, "scratch count")?;
    if sc_vec4s > u32::MAX as usize {
        return Err("domain_dock_gpu scratch u32 index overflow".to_string());
    }
    let scratch_bytes = (sc_vec4s as u64)
        .checked_mul(16)
        .ok_or_else(|| "domain_dock_gpu scratch byte overflow".to_string())?
        .max(16);
    check_storage("scratch_buf", scratch_bytes)?;

    // 5. `energies_out` (`2 * n_replicas` vec4s = `32 * n_replicas` B, 4-byte copy aligned)
    let en_vec4s = mul_u(2, n_replicas_dispatch, "energies vec4s")?;
    let energies_bytes = ((en_vec4s as u64) * 16)
        .div_ceil(wgpu::COPY_BUFFER_ALIGNMENT)
        .max(1)
        * wgpu::COPY_BUFFER_ALIGNMENT;
    check_storage("energies_buf", energies_bytes)?;

    // 6. `grads_out` (`(l_res + 2 * n_chains) * n_replicas` vec4s)
    let gr_stride = add_u(l_res, mul_u(2, n_chains, "gr chains")?, "gr stride")?;
    let gr_vec4s = mul_u(n_replicas_dispatch, gr_stride, "grads vec4s")?;
    if gr_vec4s > u32::MAX as usize {
        return Err("domain_dock_gpu grads u32 index overflow".to_string());
    }
    let grads_bytes = ((gr_vec4s as u64)
        .checked_mul(16)
        .ok_or_else(|| "domain_dock_gpu grads byte overflow".to_string())?)
    .div_ceil(wgpu::COPY_BUFFER_ALIGNMENT)
    .max(1)
        * wgpu::COPY_BUFFER_ALIGNMENT;
    check_storage("grads_buf", grads_bytes)?;

    Ok(DomainDockDispatchSizes {
        scratch_bytes,
        energies_bytes,
        grads_bytes,
    })
}

fn pick_chunk_size_for_device(
    limits: &wgpu::Limits,
    total_replicas: usize,
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
) -> Result<usize, String> {
    let mut chunk = total_replicas.clamp(1, MAX_DOMAIN_DOCK_REPLICAS_PER_DISPATCH);
    while chunk > 1 {
        if validate_domain_dock_gpu_limits(limits, chunk, kin, prep).is_ok() {
            return Ok(chunk);
        }
        chunk /= 2;
    }
    validate_domain_dock_gpu_limits(limits, 1, kin, prep)?;
    Ok(1)
}

/// Pre-compiled WebGPU compute pipeline for `shaders/domain_dock.wgsl`.
pub struct DomainDockGpuPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl DomainDockGpuPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nmr-domain-dock-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/domain_dock.wgsl").into()),
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
            label: Some("nmr-domain-dock-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_rw(4),
                storage_rw(5),
                storage_rw(6),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nmr-domain-dock-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nmr-domain-dock-pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("eval_domain_dock_batch"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self { pipeline, bgl }
    }
}

fn pack_static_buffers(
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
) -> (Vec<[f32; 4]>, Vec<u32>) {
    let l = kin.residues.len();
    let mut sf: Vec<[f32; 4]> = Vec::new();
    let mut si: Vec<u32> = Vec::new();

    let origin = if let Some(b0) = kin.rigid_blocks.first() {
        kin.residues[b0.end_res.saturating_sub(1)].ref_slots[4].0
    } else {
        kin.chain_ref_centroids.first().copied().unwrap_or([0.0; 3])
    };
    let sub_org_f32 = |p: [f64; 3], w: f32| -> [f32; 4] {
        [
            (p[0] - origin[0]) as f32,
            (p[1] - origin[1]) as f32,
            (p[2] - origin[2]) as f32,
            w,
        ]
    };

    // `off_bond_geom = 0` (`2 * l`)
    for rg in &kin.residues {
        sf.push([
            rg.r_c_prev_n as f32,
            rg.ang_ca_prev_c_prev_n as f32,
            rg.r_n_ca as f32,
            rg.ang_c_prev_n_ca as f32,
        ]);
        sf.push([
            rg.r_ca_c as f32,
            rg.ang_n_ca_c as f32,
            rg.r_c_o as f32,
            rg.ang_ca_c_o as f32,
        ]);
    }
    // `off_ref_tors = 2 * l` (`l`)
    for i in 0..l {
        sf.push([
            kin.ref_phi_rad[i] as f32,
            kin.ref_psi_rad[i] as f32,
            kin.ref_omega_rad[i] as f32,
            0.0,
        ]);
    }
    // `off_local_slots = 3 * l` (`7 * l`)
    for rg in &kin.residues {
        let is_blk = rg.rigid_block_id.is_some();
        for s_idx in 0..7 {
            let (p, pres) = rg.local_slots[s_idx];
            let w = if pres { 1.0 } else { 0.0 };
            if is_blk || !pres {
                sf.push([p[0] as f32, p[1] as f32, p[2] as f32, w]);
            } else {
                sf.push(sub_org_f32(p, w));
            }
        }
    }
    // `off_c_prev_virt = 10 * l` (`3 * l`)
    for rg in &kin.residues {
        if rg.rigid_block_id.is_some() {
            sf.push(sub_org_f32(rg.ref_slots[0].0, 0.0));
            sf.push(sub_org_f32(rg.ref_slots[2].0, 0.0));
            sf.push(sub_org_f32(rg.ref_slots[4].0, 0.0));
        } else {
            sf.push(sub_org_f32(rg.c_prev_virtual, 0.0));
            sf.push(sub_org_f32(rg.ref_slots[2].0, 0.0));
            sf.push(sub_org_f32(rg.ref_slots[4].0, 0.0));
        }
    }
    // `off_chain_centroids = 13 * l` (`n_chains`)
    for &c_af in &kin.chain_ref_centroids {
        sf.push(sub_org_f32(c_af, 0.0));
    }
    // `off_pcs_tags_f` (`4 * n_pcs_tags`)
    for tag in &prep.pcs_tags {
        sf.push([
            tag.metal_local_pos[0] as f32,
            tag.metal_local_pos[1] as f32,
            tag.metal_local_pos[2] as f32,
            0.0,
        ]);
        for row in &tag.dchi_local {
            sf.push([row[0] as f32, row[1] as f32, row[2] as f32, 0.0]);
        }
    }
    // `off_pre_sites_f` (`n_pre_sites`)
    for ps in &prep.pre_sites {
        sf.push([
            ps.site_local_pos[0] as f32,
            ps.site_local_pos[1] as f32,
            ps.site_local_pos[2] as f32,
            0.0,
        ]);
    }
    // `off_rdc_f` (`n_rdc_terms`)
    for rt in &prep.rdc_terms {
        sf.push([
            rt.d_obs_hz as f32,
            rt.d_max_eff_hz as f32,
            rt.inv_sigma2 as f32,
            0.0,
        ]);
    }
    // `off_pcs_f` (`n_pcs_terms`)
    for pt in &prep.pcs_terms {
        sf.push([pt.pcs_obs_eff_ppm as f32, pt.inv_sigma2 as f32, 0.0, 0.0]);
    }
    // `off_pre_f` (`2 * n_pre_terms`)
    for pr in &prep.pre_terms {
        sf.push([
            pr.gamma2_obs_s1 as f32,
            pr.inv_sigma2 as f32,
            pr.k_pre_a6_s1 as f32,
            0.0,
        ]);
        sf.push([pr.lower_a as f32, pr.upper_a as f32, 0.0, 0.0]);
    }
    // `off_noe_f` (`n_noe_terms`)
    for nt in &prep.noe_terms {
        sf.push([nt.lower_a as f32, nt.upper_a as f32, 0.0, 0.0]);
    }

    if sf.is_empty() {
        sf.push([0.0; 4]);
    }

    // ---- `static_ints` ----
    // `ioff_res = 0` (`4 * l`)
    for rg in &kin.residues {
        let rc_u = match rg.res_class {
            DockResClass::General => 0u32,
            DockResClass::Gly => 1u32,
            DockResClass::Pro => 2u32,
        };
        let bid_u = rg.rigid_block_id.map_or(NO_BLOCK_SENTINEL, |b| b as u32);
        let mut flg = 0u32;
        if rg.phi_active {
            flg |= 1u32;
        }
        if rg.psi_active {
            flg |= 2u32;
        }
        if rg.rama_active {
            flg |= 4u32;
        }
        if rg.is_chain_start {
            flg |= 8u32;
        }
        if rg.is_reentry_start {
            flg |= 16u32;
        }
        if rg.is_chain_end {
            flg |= 32u32;
        }
        si.extend([rc_u, bid_u, rg.chain_idx as u32, flg]);
    }
    // `ioff_chains` (`2 * n_chains`)
    for &(cs, ce) in &kin.chain_ranges {
        si.extend([cs as u32, ce as u32]);
    }
    // `ioff_blocks` (`4 * n_rigid_blocks`: `[start_res, end_res, master_block_id, chain_idx]`)
    for blk in &kin.rigid_blocks {
        si.extend([
            blk.start_res as u32,
            blk.end_res as u32,
            blk.master_block_id as u32,
            blk.chain_idx as u32,
        ]);
    }
    // `ioff_pcs_tags` (`2 * n_pcs_tags`)
    for tag in &prep.pcs_tags {
        si.extend([tag.host_block_id as u32, tag.host_seg_id as u32]);
    }
    // `ioff_pre_sites` (`2 * n_pre_sites`)
    for ps in &prep.pre_sites {
        si.extend([ps.host_block_id as u32, ps.host_seg_id as u32]);
    }
    // `ioff_rdc` (`5 * n_rdc_terms`)
    for rt in &prep.rdc_terms {
        si.extend([
            rt.medium_idx as u32,
            rt.atom_a as u32,
            rt.atom_b as u32,
            rt.seg_a as u32,
            rt.seg_b as u32,
        ]);
    }
    // `ioff_pcs` (`3 * n_pcs_terms`)
    for pt in &prep.pcs_terms {
        si.extend([pt.tag_idx as u32, pt.atom_slot as u32, pt.seg_idx as u32]);
    }
    // `ioff_pre` (`4 * n_pre_terms`)
    for pr in &prep.pre_terms {
        si.extend([
            pr.site_idx as u32,
            pr.atom_slot as u32,
            pr.seg_idx as u32,
            if pr.is_bounds_mode { 1u32 } else { 0u32 },
        ]);
    }
    // `ioff_noe_terms` (`2 * n_noe_terms`)
    for nt in &prep.noe_terms {
        si.extend([nt.cand_start as u32, nt.cand_len as u32]);
    }
    // `ioff_noe_cands` (`4 * n_noe_cands`)
    for nc in &prep.noe_candidates {
        si.extend([
            nc.atom_a as u32,
            nc.atom_b as u32,
            nc.seg_a as u32,
            nc.seg_b as u32,
        ]);
    }

    if si.is_empty() {
        si.extend([0u32; 4]);
    }

    (sf, si)
}

/// Replica input bundle `(state, per_medium_saupe_matrices)` for batched WebGPU evaluation.
#[derive(Clone, Debug)]
pub struct DomainDockGpuReplicaInput {
    pub state: DomainDockState,
    pub saupe_matrices: Vec<[[f64; 3]; 3]>,
}

/// Evaluate a batch of multi-domain docking states and analytical gradients on WebGPU asynchronously
/// (automatically chunking replicas into TDR- and storage-binding-safe dispatches).
pub async fn evaluate_domain_dock_batch_gpu_async(
    engine: &GpuEngine,
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    replicas: &[DomainDockGpuReplicaInput],
    weights: &DomainDockEnergyWeights,
) -> Result<Vec<(DomainDockEnergyBreakdown, DomainDockGradient)>, String> {
    if replicas.is_empty() {
        return Ok(Vec::new());
    }
    let pipe = DomainDockGpuPipeline::new(&engine.device);
    let lim = engine.device.limits();
    let chunk_cap = pick_chunk_size_for_device(&lim, replicas.len(), kin, prep)?;

    let (sf_data, si_data) = pack_static_buffers(kin, prep);
    let sf_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-domain-dock-static-floats"),
            contents: bytemuck::cast_slice(&sf_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let si_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-domain-dock-static-ints"),
            contents: bytemuck::cast_slice(&si_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let mut all_out = Vec::with_capacity(replicas.len());
    for chunk in replicas.chunks(chunk_cap) {
        let sub = dispatch_domain_dock_chunk_async(
            engine, &pipe, kin, prep, &sf_buf, &si_buf, chunk, weights,
        )
        .await?;
        all_out.extend(sub);
    }
    Ok(all_out)
}

/// Synchronous native wrapper around [`evaluate_domain_dock_batch_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn evaluate_domain_dock_batch_gpu(
    engine: &GpuEngine,
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    replicas: &[DomainDockGpuReplicaInput],
    weights: &DomainDockEnergyWeights,
) -> Result<Vec<(DomainDockEnergyBreakdown, DomainDockGradient)>, String> {
    pollster::block_on(evaluate_domain_dock_batch_gpu_async(
        engine, kin, prep, replicas, weights,
    ))
}

#[allow(clippy::too_many_arguments)]
async fn dispatch_domain_dock_chunk_async(
    engine: &GpuEngine,
    pipe: &DomainDockGpuPipeline,
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    sf_buf: &wgpu::Buffer,
    si_buf: &wgpu::Buffer,
    chunk: &[DomainDockGpuReplicaInput],
    weights: &DomainDockEnergyWeights,
) -> Result<Vec<(DomainDockEnergyBreakdown, DomainDockGradient)>, String> {
    let c_len = chunk.len();
    let l = kin.residues.len();
    let nc = kin.chain_ranges.len();
    let nm = prep.n_media;

    let lim = engine.device.limits();
    let sizes = compute_and_validate_domain_dock_sizes_full(
        &lim,
        c_len,
        l,
        kin.rigid_blocks.len(),
        nc,
        nm,
        prep.rdc_terms.len(),
        prep.pcs_tags.len(),
        prep.pcs_terms.len(),
        prep.pre_sites.len(),
        prep.pre_terms.len(),
        prep.noe_terms.len(),
        prep.noe_candidates.len(),
    )?;

    let in_stride = l + 2 * nc + 3 * nm;
    let mut rep_in_flat: Vec<[f32; 4]> = Vec::with_capacity(c_len * in_stride);
    for (ri, r_in) in chunk.iter().enumerate() {
        let st = &r_in.state;
        if st.phi_rad.len() != l || st.psi_rad.len() != l || st.omega_rad.len() != l {
            return Err(format!("replica {ri} torsion slice length != {l}"));
        }
        if st.chain_quats.len() != nc || st.chain_trans.len() != nc {
            return Err(format!("replica {ri} chain pose slice length != {nc}"));
        }
        if r_in.saupe_matrices.len() != nm {
            return Err(format!(
                "replica {ri} saupe_matrices length {} != n_media {nm}",
                r_in.saupe_matrices.len()
            ));
        }
        for i in 0..l {
            rep_in_flat.push([
                st.phi_rad[i] as f32,
                st.psi_rad[i] as f32,
                st.omega_rad[i] as f32,
                0.0,
            ]);
        }
        for c in 0..nc {
            let q = st.chain_quats[c];
            let t = st.chain_trans[c];
            rep_in_flat.push([q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32]);
            rep_in_flat.push([t[0] as f32, t[1] as f32, t[2] as f32, 0.0]);
        }
        for s in &r_in.saupe_matrices {
            rep_in_flat.push([s[0][0] as f32, s[0][1] as f32, s[0][2] as f32, 0.0]);
            rep_in_flat.push([s[1][0] as f32, s[1][1] as f32, s[1][2] as f32, 0.0]);
            rep_in_flat.push([s[2][0] as f32, s[2][1] as f32, s[2][2] as f32, 0.0]);
        }
    }
    if rep_in_flat.is_empty() {
        rep_in_flat.push([0.0; 4]);
    }

    let params = DomainDockParamsGpu {
        n_replicas: c_len as u32,
        l_res: l as u32,
        n_rigid_blocks: kin.rigid_blocks.len() as u32,
        n_chains: nc as u32,
        n_media: nm as u32,
        n_rdc_terms: prep.rdc_terms.len() as u32,
        n_pcs_tags: prep.pcs_tags.len() as u32,
        n_pcs_terms: prep.pcs_terms.len() as u32,
        n_pre_sites: prep.pre_sites.len() as u32,
        n_pre_terms: prep.pre_terms.len() as u32,
        n_noe_terms: prep.noe_terms.len() as u32,
        w_rdc: weights.w_rdc as f32,
        w_pcs: weights.w_pcs as f32,
        w_pre: weights.w_pre as f32,
        w_noe: weights.w_noe as f32,
        w_vdw: weights.w_vdw as f32,
        vdw_scale: weights.vdw_scale as f32,
        w_rama: weights.w_rama as f32,
        noe_soft_window: weights.noe_soft_window as f32,
        n_noe_cands: prep.noe_candidates.len() as u32,
    };

    let params_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-domain-dock-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let rep_in_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-domain-dock-replica-in"),
            contents: bytemuck::cast_slice(&rep_in_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let scratch_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nmr-domain-dock-scratch"),
        size: sizes.scratch_bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let energies_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nmr-domain-dock-energies"),
        size: sizes.energies_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let grads_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("nmr-domain-dock-grads"),
        size: sizes.grads_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nmr-domain-dock-bg"),
        layout: &pipe.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sf_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: si_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: rep_in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: scratch_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: energies_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: grads_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = engine
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nmr-domain-dock-enc"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nmr-domain-dock-pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipe.pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups((c_len as u32).div_ceil(64), 1, 1);
    }
    engine.queue.submit(Some(encoder.finish()));

    let grad_stride = l + 2 * nc;
    let en_raw: Vec<[f32; 4]> = engine
        .read_buffer_vec_async(&energies_buf, 2 * c_len)
        .await?;
    let gr_raw: Vec<[f32; 4]> = engine
        .read_buffer_vec_async(&grads_buf, c_len * grad_stride)
        .await?;

    let mut out = Vec::with_capacity(c_len);
    for ci in 0..c_len {
        let e0 = en_raw[2 * ci];
        let e1 = en_raw[2 * ci + 1];
        let eb = DomainDockEnergyBreakdown {
            total: e0[0] as f64,
            rdc: e0[1] as f64,
            pcs: e0[2] as f64,
            pre: e0[3] as f64,
            noe: e1[0] as f64,
            vdw: e1[1] as f64,
            rama: e1[2] as f64,
            reentry_closure: e1[3] as f64,
        };
        let g_off = ci * grad_stride;
        let mut gp = Vec::with_capacity(l);
        let mut gs = Vec::with_capacity(l);
        for i in 0..l {
            gp.push(gr_raw[g_off + i][0] as f64);
            gs.push(gr_raw[g_off + i][1] as f64);
        }
        let mut g_rot = Vec::with_capacity(nc);
        let mut g_tr = Vec::with_capacity(nc);
        for c in 0..nc {
            let r4 = gr_raw[g_off + l + 2 * c];
            let t4 = gr_raw[g_off + l + 2 * c + 1];
            g_rot.push([r4[0] as f64, r4[1] as f64, r4[2] as f64]);
            g_tr.push([t4[0] as f64, t4[1] as f64, t4[2] as f64]);
        }
        out.push((
            eb,
            DomainDockGradient {
                grad_phi: gp,
                grad_psi: gs,
                grad_chain_rot: g_rot,
                grad_chain_trans: g_tr,
            },
        ));
    }

    Ok(out)
}

#[inline]
fn wrap_rad_host(rad: f64) -> f64 {
    let mut r = (rad + PI) % (2.0 * PI);
    if r <= 0.0 {
        r += 2.0 * PI;
    }
    r - PI
}

/// Run the full WebGPU + host `f64` `L-BFGS` multi-domain docking workflow asynchronously (`wasm32` + native).
#[allow(clippy::too_many_arguments)]
pub async fn dock_domains_gpu_async(
    engine: &GpuEngine,
    model: &AfStructureModel,
    pae: &PaeMatrix,
    rdc_media_in: &[RdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_adrs: &[AmbiguousDistanceRestraint],
    cfg: &DomainDockConfig,
) -> Result<DomainDockResult, String> {
    let partition = cluster_pae_domains_for_model(model, pae, &cfg.pae_cfg)?;
    let kin = DomainLinkerKinematics::from_af_model(model, &partition)?;
    dock_domains_with_kinematics_gpu_async(
        engine,
        kin,
        partition,
        rdc_media_in,
        pcs_tags,
        pre_sites,
        noe_adrs,
        cfg,
    )
    .await
}

/// Run multi-chain `(chain_id, seq_id)` WebGPU + host `f64` `L-BFGS` docking across multiple polymer chains (`21.W2:P1-8`).
#[allow(clippy::too_many_arguments)]
pub async fn dock_domains_multi_chain_gpu_async(
    engine: &GpuEngine,
    models: &[AfStructureModel],
    pae: &PaeMatrix,
    domain_rdc_media: &[DomainRdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_adrs: &[AmbiguousDistanceRestraint],
    cfg: &DomainDockConfig,
) -> Result<DomainDockResult, String> {
    if models.is_empty() {
        return Err(
            "dock_domains_multi_chain_gpu_async requires >= 1 AfStructureModel".to_string(),
        );
    }
    let mut seq_ids = Vec::new();
    let mut chain_ids = Vec::new();
    let mut plddts = Vec::new();
    let mut any_plddt = false;
    for (m_idx, m) in models.iter().enumerate() {
        for r in &m.residues {
            seq_ids.push(r.seq_id);
            let cid = if r.chain_id.trim().is_empty() {
                format!("{}", (b'A' + (m_idx as u8).min(25)) as char)
            } else {
                r.chain_id.clone()
            };
            chain_ids.push(cid);
            if r.plddt.is_some() {
                any_plddt = true;
            }
            plddts.push(r.plddt.unwrap_or(100.0));
        }
    }
    let partition = cluster_pae_domains(
        pae,
        any_plddt.then_some(plddts.as_slice()),
        Some(&seq_ids),
        Some(&chain_ids),
        &cfg.pae_cfg,
    )?;
    let kin = DomainLinkerKinematics::from_multi_chain_models(models, &partition)?;
    let mut cfg_mc = cfg.clone();
    cfg_mc.chain_rdc_media = domain_rdc_media.to_vec();
    dock_domains_with_kinematics_gpu_async(
        engine,
        kin,
        partition,
        &[],
        pcs_tags,
        pre_sites,
        noe_adrs,
        &cfg_mc,
    )
    .await
}

/// Synchronous native wrapper around [`dock_domains_multi_chain_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn dock_domains_multi_chain_gpu(
    engine: &GpuEngine,
    models: &[AfStructureModel],
    pae: &PaeMatrix,
    domain_rdc_media: &[DomainRdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_adrs: &[AmbiguousDistanceRestraint],
    cfg: &DomainDockConfig,
) -> Result<DomainDockResult, String> {
    pollster::block_on(dock_domains_multi_chain_gpu_async(
        engine,
        models,
        pae,
        domain_rdc_media,
        pcs_tags,
        pre_sites,
        noe_adrs,
        cfg,
    ))
}

/// Replica bookkeeping returned by [`anneal_domain_dock_replicas_gpu_async`].
///
/// All energies are the GPU SEARCH objective: `nmr_core::domain_dock::evaluate_domain_dock_fixed_saupe_cpu`-equivalent
/// composite energy with per-state VarPro-fitted Saupe matrices and the kernel's
/// `k_loop = 25 * w_rama` re-entry closure (0 at `w_rama = 0`). It excludes the host's always-on
/// closure (`k >= 500`, `21F:B5`), which only the host `L-BFGS` polish and the final
/// [`select_and_polish_top_states`] ranking apply (`FIX3 21F:N2`).
#[derive(Clone, Debug)]
pub struct DomainDockGpuAnnealOutcome {
    /// Multi-start seed states (`generate_domain_dock_seed_states`).
    pub seed_states: Vec<DomainDockState>,
    /// GPU search-objective total energy of each seed state at full-ramp weights.
    pub seed_energies: Vec<f64>,
    /// Lowest-energy state visited per replica (seed included) at full-ramp weights.
    pub best_states: Vec<DomainDockState>,
    /// GPU search-objective total energy of `best_states[r]` (`<= seed_energies[r]`).
    pub best_energies: Vec<f64>,
}

/// Per-step annealing weights (`ramp` in `[0, 1]`; `ramp = 1` gives the full search weights).
fn annealing_step_weights(base: &DomainDockEnergyWeights, ramp: f64) -> DomainDockEnergyWeights {
    let mut w = *base;
    w.w_vdw = base.w_vdw * (0.20 + 0.80 * ramp * ramp);
    w.vdw_scale = base.vdw_scale * (0.78 + 0.22 * ramp);
    w
}

/// Evaluate the GPU search objective (per-state VarPro Saupe fit on host, then one kernel pass).
#[allow(clippy::too_many_arguments)]
async fn evaluate_states_gpu_search_objective(
    engine: &GpuEngine,
    gpu_pipe: &DomainDockGpuPipeline,
    kin: &DomainLinkerKinematics,
    prep: &PreparedDomainDockRestraints,
    sf_buf: &wgpu::Buffer,
    si_buf: &wgpu::Buffer,
    states: &[DomainDockState],
    weights: &DomainDockEnergyWeights,
    chunk_cap: usize,
) -> Result<Vec<(DomainDockEnergyBreakdown, DomainDockGradient)>, String> {
    let mut gpu_inputs: Vec<DomainDockGpuReplicaInput> = Vec::with_capacity(states.len());
    for st in states {
        let saupe = if weights.w_rdc > 0.0 && prep.n_media > 0 {
            let (slots, _) = kin.forward_slots_and_block_frames(st);
            prep.fit_global_saupe_matrices_on_slots(&slots)
        } else {
            vec![[[0.0; 3]; 3]; prep.n_media]
        };
        gpu_inputs.push(DomainDockGpuReplicaInput {
            state: st.clone(),
            saupe_matrices: saupe,
        });
    }
    let mut gpu_outs = Vec::with_capacity(states.len());
    for chunk in gpu_inputs.chunks(chunk_cap.max(1)) {
        let sub = dispatch_domain_dock_chunk_async(
            engine, gpu_pipe, kin, prep, sf_buf, si_buf, chunk, weights,
        )
        .await?;
        gpu_outs.extend(sub);
    }
    Ok(gpu_outs)
}

/// WebGPU multi-replica heavy-ball annealing over `n_rep` seed states, tracking each replica's
/// lowest GPU-search-objective state at full-ramp weights (see [`DomainDockGpuAnnealOutcome`]).
pub async fn anneal_domain_dock_replicas_gpu_async(
    engine: &GpuEngine,
    kin: &DomainLinkerKinematics,
    partition: &PaeDomainPartition,
    prep: &PreparedDomainDockRestraints,
    cfg: &DomainDockConfig,
    n_rep: usize,
) -> Result<DomainDockGpuAnnealOutcome, String> {
    let mut states =
        generate_domain_dock_seed_states(kin, partition, prep, &cfg.weights, n_rep, cfg.seed);

    let lim = engine.device.limits();
    let chunk_cap = pick_chunk_size_for_device(&lim, states.len(), kin, prep)?;

    let gpu_pipe = DomainDockGpuPipeline::new(&engine.device);
    let (sf_data, si_data) = pack_static_buffers(kin, prep);
    let sf_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-domain-dock-sf"),
            contents: bytemuck::cast_slice(&sf_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let si_buf = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nmr-domain-dock-si"),
            contents: bytemuck::cast_slice(&si_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let l = kin.residues.len();
    let nc = kin.chain_ranges.len();
    let mut vel_phi = vec![vec![0.0_f64; l]; states.len()];
    let mut vel_psi = vec![vec![0.0_f64; l]; states.len()];
    let mut vel_rot = vec![vec![[0.0_f64; 3]; nc]; states.len()];
    let mut vel_trans = vec![vec![[0.0_f64; 3]; nc]; states.len()];
    // Seed the per-replica best-energy bookkeeping with the SAME objective the kernel minimises
    // and that `best_energies` is later compared against (`FIX3 21F:N2`): one GPU pass over the
    // seed states at the full-ramp weights. The host `evaluate_domain_dock_state_cpu` objective
    // adds the always-on re-entry closure (`k >= 500`) that the kernel omits, so seeding with it
    // made every seed look worse than any annealed state and the kept replica depend on which
    // objective scored it. Closure is enforced afterwards by the host polish + final ranking.
    let full_w = annealing_step_weights(&cfg.weights, 1.0);
    let seed_gpu = evaluate_states_gpu_search_objective(
        engine, &gpu_pipe, kin, prep, &sf_buf, &si_buf, &states, &full_w, chunk_cap,
    )
    .await?;
    let seed_energies: Vec<f64> = seed_gpu.iter().map(|(eb, _)| eb.total).collect();
    let seed_states = states.clone();
    let mut best_states = states.clone();
    let mut best_energies = seed_energies.clone();

    let total_steps = cfg.anneal_steps.max(1);
    for step in 0..total_steps {
        let prog = if total_steps > 1 {
            (step as f64) / ((total_steps - 1) as f64)
        } else {
            1.0
        };
        let ramp = (prog / 0.65).min(1.0);
        let cur_w = annealing_step_weights(&cfg.weights, ramp);

        let gpu_outs = evaluate_states_gpu_search_objective(
            engine, &gpu_pipe, kin, prep, &sf_buf, &si_buf, &states, &cur_w, chunk_cap,
        )
        .await?;

        for (ri, ((eb, gr), st)) in gpu_outs.into_iter().zip(states.iter_mut()).enumerate() {
            if ramp >= 1.0 && eb.total < best_energies[ri] {
                best_energies[ri] = eb.total;
                best_states[ri] = st.clone();
            }

            let mut g2 = 0.0_f64;
            for &i in &kin.active_linker_residues {
                g2 += gr.grad_phi[i] * gr.grad_phi[i] + gr.grad_psi[i] * gr.grad_psi[i];
            }
            for c_idx in 1..nc {
                g2 += v_dot(gr.grad_chain_rot[c_idx], gr.grad_chain_rot[c_idx])
                    + v_dot(gr.grad_chain_trans[c_idx], gr.grad_chain_trans[c_idx]);
            }
            let g_norm = g2.sqrt().max(1.0);
            let lr = (0.025 * (1.0 - 0.65 * prog)) / (1.0 + 0.25 * g_norm);
            let mom = 0.45 * (1.0 - 0.40 * prog);

            for &i in &kin.active_linker_residues {
                if kin.residues[i].phi_active {
                    vel_phi[ri][i] = mom * vel_phi[ri][i] - lr * gr.grad_phi[i];
                    st.phi_rad[i] =
                        wrap_rad_host(st.phi_rad[i] + vel_phi[ri][i].clamp(-0.04, 0.04));
                }
                if kin.residues[i].psi_active {
                    vel_psi[ri][i] = mom * vel_psi[ri][i] - lr * gr.grad_psi[i];
                    st.psi_rad[i] =
                        wrap_rad_host(st.psi_rad[i] + vel_psi[ri][i].clamp(-0.04, 0.04));
                }
            }
            for c_idx in 1..nc {
                for ax in 0..3 {
                    vel_rot[ri][c_idx][ax] =
                        mom * vel_rot[ri][c_idx][ax] - lr * gr.grad_chain_rot[c_idx][ax];
                    vel_trans[ri][c_idx][ax] = mom * vel_trans[ri][c_idx][ax]
                        - (4.0 * lr) * gr.grad_chain_trans[c_idx][ax];
                    st.chain_trans[c_idx][ax] += vel_trans[ri][c_idx][ax].clamp(-0.6, 0.6);
                }
                let domg = [
                    vel_rot[ri][c_idx][0].clamp(-0.08, 0.08),
                    vel_rot[ri][c_idx][1].clamp(-0.08, 0.08),
                    vel_rot[ri][c_idx][2].clamp(-0.08, 0.08),
                ];
                st.chain_quats[c_idx] = quat_apply_world_rot_step(st.chain_quats[c_idx], domg);
            }
        }
    }
    Ok(DomainDockGpuAnnealOutcome {
        seed_states,
        seed_energies,
        best_states,
        best_energies,
    })
}

/// Synchronous native wrapper around [`anneal_domain_dock_replicas_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
pub fn anneal_domain_dock_replicas_gpu(
    engine: &GpuEngine,
    kin: &DomainLinkerKinematics,
    partition: &PaeDomainPartition,
    prep: &PreparedDomainDockRestraints,
    cfg: &DomainDockConfig,
    n_rep: usize,
) -> Result<DomainDockGpuAnnealOutcome, String> {
    pollster::block_on(anneal_domain_dock_replicas_gpu_async(
        engine, kin, partition, prep, cfg, n_rep,
    ))
}

/// Run WebGPU multi-replica annealing + host `f64` `L-BFGS` polish on a pre-built `DomainLinkerKinematics`.
#[allow(clippy::too_many_arguments)]
pub async fn dock_domains_with_kinematics_gpu_async(
    engine: &GpuEngine,
    kin: DomainLinkerKinematics,
    partition: PaeDomainPartition,
    rdc_media_in: &[RdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_adrs: &[AmbiguousDistanceRestraint],
    cfg: &DomainDockConfig,
) -> Result<DomainDockResult, String> {
    #[cfg(not(target_arch = "wasm32"))]
    let t0 = std::time::Instant::now();

    let mut rdc_media = rdc_media_in.to_vec();
    let pipe_in = prepare_domain_dock_inputs_full(
        &kin,
        &partition,
        &mut rdc_media,
        pcs_tags,
        pre_sites,
        noe_adrs,
        cfg,
    )?;
    let prep = &pipe_in.prep;

    let has_any_restraints = !prep.rdc_terms.is_empty()
        || !prep.pcs_terms.is_empty()
        || !prep.pre_terms.is_empty()
        || !prep.noe_terms.is_empty();

    let n_ens = cfg.ensemble_size.max(1);
    let engine_label = format!("WebGPU ({}, {})", engine.adapter_name, engine.backend_name);

    let polished = if !has_any_restraints || partition.domains.len() <= 1 {
        let st0 = kin.initial_state();
        let (eb0, _) = evaluate_domain_dock_state_cpu(&kin, prep, &st0, &cfg.weights);
        vec![(st0, eb0)]
    } else {
        let n_rep = cfg.n_replicas.max(n_ens);
        let anneal =
            anneal_domain_dock_replicas_gpu_async(engine, &kin, &partition, prep, cfg, n_rep)
                .await?;
        let best_states = anneal.best_states;

        select_and_polish_top_states(
            &kin,
            prep,
            best_states,
            &cfg.weights,
            n_ens,
            cfg.lbfgs_iters,
        )
    };

    #[cfg(not(target_arch = "wasm32"))]
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    #[cfg(target_arch = "wasm32")]
    let elapsed_ms = 0.0;

    finalize_domain_dock_pipeline(
        kin,
        partition,
        pipe_in,
        pre_sites,
        polished,
        engine_label,
        elapsed_ms,
    )
}

/// Synchronous native wrapper around [`dock_domains_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn dock_domains_gpu(
    engine: &GpuEngine,
    model: &AfStructureModel,
    pae: &PaeMatrix,
    rdc_media_in: &[RdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_adrs: &[AmbiguousDistanceRestraint],
    cfg: &DomainDockConfig,
) -> Result<DomainDockResult, String> {
    pollster::block_on(dock_domains_gpu_async(
        engine,
        model,
        pae,
        rdc_media_in,
        pcs_tags,
        pre_sites,
        noe_adrs,
        cfg,
    ))
}

/// Synchronous native wrapper around [`dock_domains_with_kinematics_gpu_async`].
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn dock_domains_with_kinematics_gpu(
    engine: &GpuEngine,
    kin: DomainLinkerKinematics,
    partition: PaeDomainPartition,
    rdc_media_in: &[RdcMediumSet],
    pcs_tags: &[DomainPcsTagSet],
    pre_sites: &[DomainPreSiteSet],
    noe_adrs: &[AmbiguousDistanceRestraint],
    cfg: &DomainDockConfig,
) -> Result<DomainDockResult, String> {
    pollster::block_on(dock_domains_with_kinematics_gpu_async(
        engine,
        kin,
        partition,
        rdc_media_in,
        pcs_tags,
        pre_sites,
        noe_adrs,
        cfg,
    ))
}
