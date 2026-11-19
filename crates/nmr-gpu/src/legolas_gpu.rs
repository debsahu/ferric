//! WebGPU (`wgpu` + WGSL) compute engine for `LEGOLAS` ANI Atomic Environment Vector (`AEV`)
//! evaluation and per-nucleus ensemble MLP chemical-shift inference (`T22.12`).
//!
//! # Provenance & License Attribution
//! Clean-room WebGPU compute implementation matching `nmr_core::legolas_shifts` to `< 1e-4`:
//! - Darrows, M. A., Kodituwakku, M., Xue, K., Pickering, I., Terrel, J. & Roitberg, A. E.
//!   (*J. Chem. Theory Comput.* **2025**, 21(8), 4266–4275;
//!   DOI: `10.1021/acs.jctc.5c00026`; <https://github.com/roitberg-group/legolas>,
//!   **MIT License**).
//! - Smith, J. S., Isayev, O. & Roitberg, A. E. (*Chem. Sci.* **2017**, 8, 3192–3203;
//!   DOI: `10.1039/C6SC05720A`) — ANI-1 radial and modified angular `AEV` descriptors.
//! - Behler, J. & Parrinello, M. (*Phys. Rev. Lett.* **2007**, 98, 146401).

use std::collections::{BTreeMap, HashMap};

use bytemuck::{Pod, Zeroable};
use nmr_core::legolas_shifts::{
    validate_atoms, AevConfig, AtomShiftPrediction, LegolasAtom, LegolasError, LegolasMlp,
    LegolasModel, LegolasPredictOptions, NucleusEnsemble, NucleusShiftEstimate, TargetNucleus,
    MAX_LEGOLAS_ATOMS_PER_CELL, MAX_LEGOLAS_NEIGHBORS_PER_ATOM, MIN_PAIR_DIST_ANGSTROM,
};
use wgpu::util::DeviceExt;

use crate::GpuEngine;

/// Maximum spatial hash table buckets (`2^18 = 262,144`) allocated on GPU for `O(N)` cell lookup.
const MAX_HASH_BUCKETS: usize = 1 << 18;

/// Target upper bound on threads per single MLP layer dispatch so dispatches stay well within
/// `max_compute_workgroups_per_dimension` and OS GPU TDR watchdogs.
const MAX_MLP_JOBS_PER_DISPATCH: usize = 2_097_152;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct AevParamsGpu {
    n_atoms: u32,
    n_queries: u32,
    bucket_mask: u32,
    aev_dim: u32,
    n_eta_r: u32,
    n_rs_r: u32,
    n_eta_a: u32,
    n_rs_a: u32,
    n_zeta: u32,
    n_theta: u32,
    rad_dim: u32,
    angular_formula: u32,
    r_cut_radial: f32,
    r_cut_angular: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuAtom {
    x: f32,
    y: f32,
    z: f32,
    species: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuCell {
    cx: i32,
    cy: i32,
    cz: i32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MlpLayerParamsGpu {
    n_queries: u32,
    n_members: u32,
    in_dim: u32,
    out_dim: u32,
    activation_kind: u32,
    broadcast_input: u32,
    apply_output_scale: u32,
    _pad0: u32,
    celu_alpha: f32,
    output_mean: f32,
    output_scale: f32,
    _pad1: f32,
}

#[inline]
fn spatial_hash_u32(cx: i32, cy: i32, cz: i32) -> u32 {
    let h = (cx as u32).wrapping_mul(73_856_093)
        ^ (cy as u32).wrapping_mul(19_349_663)
        ^ (cz as u32).wrapping_mul(83_492_791);
    h ^ (h >> 16)
}

fn check_storage_limit(limits: &wgpu::Limits, name: &str, bytes: u64) -> Result<(), String> {
    if bytes > limits.max_buffer_size {
        return Err(format!(
            "GPU LEGOLAS {name} ({bytes} B) exceeds device max_buffer_size ({} B)",
            limits.max_buffer_size
        ));
    }
    if bytes > limits.max_storage_buffer_binding_size {
        return Err(format!(
            "GPU LEGOLAS {name} ({bytes} B) exceeds device max_storage_buffer_binding_size ({} B)",
            limits.max_storage_buffer_binding_size
        ));
    }
    Ok(())
}

fn check_uniform_limit(limits: &wgpu::Limits, name: &str, bytes: u64) -> Result<(), String> {
    if bytes > limits.max_buffer_size {
        return Err(format!(
            "GPU LEGOLAS {name} ({bytes} B) exceeds device max_buffer_size ({} B)",
            limits.max_buffer_size
        ));
    }
    if bytes > limits.max_uniform_buffer_binding_size {
        return Err(format!(
            "GPU LEGOLAS {name} ({bytes} B) exceeds device max_uniform_buffer_binding_size ({} B)",
            limits.max_uniform_buffer_binding_size
        ));
    }
    Ok(())
}

fn check_workgroups_1d(limits: &wgpu::Limits, name: &str, n_jobs: usize) -> Result<u32, String> {
    let wg = n_jobs.div_ceil(64);
    let max_wg = limits.max_compute_workgroups_per_dimension as usize;
    if wg > max_wg {
        return Err(format!(
            "GPU LEGOLAS {name} workgroup count ({wg}) exceeds device max_compute_workgroups_per_dimension ({max_wg})"
        ));
    }
    u32::try_from(wg).map_err(|_| format!("GPU LEGOLAS {name} workgroup count overflow"))
}

struct PreparedAtomScene {
    gpu_atoms: Vec<GpuAtom>,
    gpu_cells: Vec<GpuCell>,
    bucket_starts: Vec<u32>,
    sorted_atoms: Vec<u32>,
    bucket_mask: u32,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl PreparedAtomScene {
    /// Enforce `MAX_LEGOLAS_NEIGHBORS_PER_ATOM` for every queried atom prior to GPU dispatch
    /// (`22.12:P2-2`), preventing quadratic angular TDR hangs on dense hostile coordinate inputs.
    fn validate_query_neighbor_caps(
        &self,
        atoms: &[LegolasAtom],
        query_indices: &[usize],
        cfg: &AevConfig,
    ) -> Result<(), String> {
        let r_cut_rad_sq = cfg.r_cut_radial * cfg.r_cut_radial;
        let r_cut_ang_sq = cfg.r_cut_angular * cfg.r_cut_angular;
        let min_dist_sq = MIN_PAIR_DIST_ANGSTROM * MIN_PAIR_DIST_ANGSTROM;

        for &q in query_indices {
            if q >= atoms.len() {
                return Err(format!(
                    "query atom index {q} out of range (n_atoms={})",
                    atoms.len()
                ));
            }
            let c = self.gpu_cells[q];
            let mut coarse_total = 0usize;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(bucket) = self.cells.get(&(c.cx + dx, c.cy + dy, c.cz + dz)) {
                            coarse_total = coarse_total.saturating_add(bucket.len());
                        }
                    }
                }
            }
            if coarse_total <= MAX_LEGOLAS_NEIGHBORS_PER_ATOM {
                continue;
            }
            let pos_i = atoms[q].position;
            let mut rad_count = 0usize;
            let mut ang_count = 0usize;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let Some(bucket) = self.cells.get(&(c.cx + dx, c.cy + dy, c.cz + dz))
                        else {
                            continue;
                        };
                        for &j in bucket {
                            if j == q {
                                continue;
                            }
                            let pos_j = atoms[j].position;
                            let vx = pos_j[0] - pos_i[0];
                            let vy = pos_j[1] - pos_i[1];
                            let vz = pos_j[2] - pos_i[2];
                            let r_sq = vx * vx + vy * vy + vz * vz;
                            if r_sq <= min_dist_sq {
                                continue;
                            }
                            if r_sq <= r_cut_rad_sq {
                                rad_count += 1;
                                if rad_count > MAX_LEGOLAS_NEIGHBORS_PER_ATOM {
                                    return Err(format!(
                                        "atom {q} radial neighbor count exceeds MAX_LEGOLAS_NEIGHBORS_PER_ATOM ({MAX_LEGOLAS_NEIGHBORS_PER_ATOM})"
                                    ));
                                }
                            }
                            if r_sq <= r_cut_ang_sq {
                                ang_count += 1;
                                if ang_count > MAX_LEGOLAS_NEIGHBORS_PER_ATOM {
                                    return Err(format!(
                                        "atom {q} angular neighbor count exceeds MAX_LEGOLAS_NEIGHBORS_PER_ATOM ({MAX_LEGOLAS_NEIGHBORS_PER_ATOM})"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn prepare_atom_scene(atoms: &[LegolasAtom], cfg: &AevConfig) -> Result<PreparedAtomScene, String> {
    validate_atoms(atoms)?;
    cfg.validate()?;
    let n = atoms.len();
    if n == 0 {
        return Err("cannot prepare GPU atom scene for empty atom list".to_string());
    }

    // Subtract f64 centroid before f32 packing to preserve translation invariance & mantissa precision
    let mut sum = [0.0_f64; 3];
    for a in atoms {
        sum[0] += a.position[0];
        sum[1] += a.position[1];
        sum[2] += a.position[2];
    }
    let inv_n = 1.0 / (n as f64);
    let centroid = [sum[0] * inv_n, sum[1] * inv_n, sum[2] * inv_n];

    let cell_size = cfg.r_cut_radial.max(cfg.r_cut_angular) + 1e-3;
    let inv_cell_size = 1.0 / cell_size;

    let mut num_buckets = 64usize;
    while num_buckets < n.saturating_mul(2) && num_buckets < MAX_HASH_BUCKETS {
        num_buckets <<= 1;
    }
    let bucket_mask = (num_buckets - 1) as u32;

    let mut gpu_atoms = Vec::with_capacity(n);
    let mut gpu_cells = Vec::with_capacity(n);
    let mut atom_buckets = Vec::with_capacity(n);
    let mut counts = vec![0u32; num_buckets];
    let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::with_capacity(n);

    for (idx, a) in atoms.iter().enumerate() {
        let rx = a.position[0] - centroid[0];
        let ry = a.position[1] - centroid[1];
        let rz = a.position[2] - centroid[2];
        let cx_f = (rx * inv_cell_size).floor();
        let cy_f = (ry * inv_cell_size).floor();
        let cz_f = (rz * inv_cell_size).floor();
        if !cx_f.is_finite()
            || !cy_f.is_finite()
            || !cz_f.is_finite()
            || cx_f.abs() > 1e8
            || cy_f.abs() > 1e8
            || cz_f.abs() > 1e8
        {
            return Err(format!(
                "atom {idx} relative cell index out of safe 32-bit range"
            ));
        }
        let cx = cx_f as i32;
        let cy = cy_f as i32;
        let cz = cz_f as i32;

        let cell_entry = cells.entry((cx, cy, cz)).or_default();
        cell_entry.push(idx);
        if cell_entry.len() > MAX_LEGOLAS_ATOMS_PER_CELL {
            return Err(format!(
                "GPU spatial cell ({cx}, {cy}, {cz}) occupancy ({}) exceeds MAX_LEGOLAS_ATOMS_PER_CELL ({MAX_LEGOLAS_ATOMS_PER_CELL}; MAX_LEGOLAS_NEIGHBORS_PER_ATOM = {MAX_LEGOLAS_NEIGHBORS_PER_ATOM})",
                cell_entry.len()
            ));
        }

        gpu_atoms.push(GpuAtom {
            x: rx as f32,
            y: ry as f32,
            z: rz as f32,
            species: a.species.as_u32(),
        });
        gpu_cells.push(GpuCell {
            cx,
            cy,
            cz,
            _pad: 0,
        });
        let b = (spatial_hash_u32(cx, cy, cz) & bucket_mask) as usize;
        atom_buckets.push(b);
        counts[b] = counts[b].saturating_add(1);
    }

    let mut bucket_starts = vec![0u32; num_buckets + 1];
    let mut acc = 0u32;
    for b in 0..num_buckets {
        bucket_starts[b] = acc;
        acc = acc.saturating_add(counts[b]);
    }
    bucket_starts[num_buckets] = acc;

    let mut cursors = bucket_starts[..num_buckets].to_vec();
    let mut sorted_atoms = vec![0u32; n];
    for (idx, &b) in atom_buckets.iter().enumerate() {
        let dst = cursors[b] as usize;
        if dst < sorted_atoms.len() {
            sorted_atoms[dst] = idx as u32;
        }
        cursors[b] = cursors[b].saturating_add(1);
    }

    Ok(PreparedAtomScene {
        gpu_atoms,
        gpu_cells,
        bucket_starts,
        sorted_atoms,
        bucket_mask,
        cells,
    })
}

fn pack_aev_tables(cfg: &AevConfig) -> Vec<f32> {
    let mut tables = Vec::with_capacity(
        cfg.eta_radial.len()
            + cfg.rs_radial.len()
            + cfg.eta_angular.len()
            + cfg.rs_angular.len()
            + cfg.zeta_angular.len()
            + cfg.theta_s_angular.len(),
    );
    tables.extend(cfg.eta_radial.iter().map(|&v| v as f32));
    tables.extend(cfg.rs_radial.iter().map(|&v| v as f32));
    tables.extend(cfg.eta_angular.iter().map(|&v| v as f32));
    tables.extend(cfg.rs_angular.iter().map(|&v| v as f32));
    tables.extend(cfg.zeta_angular.iter().map(|&v| v as f32));
    tables.extend(cfg.theta_s_angular.iter().map(|&v| v as f32));
    tables
}

/// Reusable WebGPU compute context for `LEGOLAS` ANI-AEV evaluation and ensemble MLP inference.
pub struct LegolasGpuEngine<'a> {
    engine: &'a GpuEngine,
    aev_pipeline: wgpu::ComputePipeline,
    mlp_pipeline: wgpu::ComputePipeline,
    aev_bgl: wgpu::BindGroupLayout,
    mlp_bgl: wgpu::BindGroupLayout,
    empty_bgl: wgpu::BindGroupLayout,
}

impl<'a> LegolasGpuEngine<'a> {
    /// Compile the `legolas_aev.wgsl` shader module and build its compute pipelines on `engine`.
    pub fn new(engine: &'a GpuEngine) -> Self {
        let device = &engine.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("legolas-aev-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/legolas_aev.wgsl").into()),
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

        let aev_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("legolas-aev-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                storage_ro(5),
                storage_ro(6),
                storage_rw(7),
            ],
        });
        let aev_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("legolas-aev-pl"),
            bind_group_layouts: &[Some(&aev_bgl)],
            immediate_size: 0,
        });
        let aev_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("legolas-aev-pipeline"),
            layout: Some(&aev_pl),
            module: &shader,
            entry_point: Some("compute_aev"),
            compilation_options: Default::default(),
            cache: None,
        });

        let empty_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("legolas-empty-bgl"),
            entries: &[],
        });
        let mlp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("legolas-mlp-bgl"),
            entries: &[
                uniform_entry(0),
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_rw(4),
            ],
        });
        let mlp_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("legolas-mlp-pl"),
            bind_group_layouts: &[Some(&empty_bgl), Some(&mlp_bgl)],
            immediate_size: 0,
        });
        let mlp_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("legolas-mlp-pipeline"),
            layout: Some(&mlp_pl),
            module: &shader,
            entry_point: Some("mlp_layer_forward"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            engine,
            aev_pipeline,
            mlp_pipeline,
            aev_bgl,
            mlp_bgl,
            empty_bgl,
        }
    }

    fn dispatch_aev_into_buffer(
        &self,
        scene: &PreparedAtomScene,
        query_indices: &[usize],
        cfg: &AevConfig,
    ) -> Result<wgpu::Buffer, String> {
        let device = &self.engine.device;
        let limits = device.limits();
        let n_atoms = scene.gpu_atoms.len();
        let n_queries = query_indices.len();
        let aev_dim = cfg.total_dim();

        let query_u32: Vec<u32> = query_indices
            .iter()
            .map(|&q| {
                if q >= n_atoms {
                    Err(format!(
                        "query atom index {q} out of range (n_atoms={n_atoms})"
                    ))
                } else {
                    u32::try_from(q).map_err(|_| "query index exceeds u32".to_string())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let tables = pack_aev_tables(cfg);
        let out_elems = n_queries
            .checked_mul(aev_dim)
            .ok_or_else(|| "GPU AEV output element count overflow".to_string())?;
        let out_bytes = u64::try_from(
            out_elems
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "GPU AEV output byte length overflow".to_string())?,
        )
        .map_err(|_| "GPU AEV output byte length overflow".to_string())?;

        check_uniform_limit(
            &limits,
            "aev_params",
            std::mem::size_of::<AevParamsGpu>() as u64,
        )?;
        check_storage_limit(
            &limits,
            "atoms",
            (scene.gpu_atoms.len() * std::mem::size_of::<GpuAtom>()) as u64,
        )?;
        check_storage_limit(
            &limits,
            "atom_cells",
            (scene.gpu_cells.len() * std::mem::size_of::<GpuCell>()) as u64,
        )?;
        check_storage_limit(
            &limits,
            "bucket_starts",
            (scene.bucket_starts.len() * std::mem::size_of::<u32>()) as u64,
        )?;
        check_storage_limit(
            &limits,
            "sorted_atoms",
            (scene.sorted_atoms.len() * std::mem::size_of::<u32>()) as u64,
        )?;
        check_storage_limit(
            &limits,
            "query_indices",
            (query_u32.len() * std::mem::size_of::<u32>()) as u64,
        )?;
        check_storage_limit(
            &limits,
            "aev_tables",
            (tables.len() * std::mem::size_of::<f32>()) as u64,
        )?;
        check_storage_limit(&limits, "aev_out", out_bytes)?;
        let wg_x = check_workgroups_1d(&limits, "compute_aev", n_queries)?;

        let params = AevParamsGpu {
            n_atoms: u32::try_from(n_atoms).map_err(|_| "n_atoms exceeds u32".to_string())?,
            n_queries: u32::try_from(n_queries).map_err(|_| "n_queries exceeds u32".to_string())?,
            bucket_mask: scene.bucket_mask,
            aev_dim: u32::try_from(aev_dim).map_err(|_| "aev_dim exceeds u32".to_string())?,
            n_eta_r: u32::try_from(cfg.eta_radial.len())
                .map_err(|_| "n_eta_r exceeds u32".to_string())?,
            n_rs_r: u32::try_from(cfg.rs_radial.len())
                .map_err(|_| "n_rs_r exceeds u32".to_string())?,
            n_eta_a: u32::try_from(cfg.eta_angular.len())
                .map_err(|_| "n_eta_a exceeds u32".to_string())?,
            n_rs_a: u32::try_from(cfg.rs_angular.len())
                .map_err(|_| "n_rs_a exceeds u32".to_string())?,
            n_zeta: u32::try_from(cfg.zeta_angular.len())
                .map_err(|_| "n_zeta exceeds u32".to_string())?,
            n_theta: u32::try_from(cfg.theta_s_angular.len())
                .map_err(|_| "n_theta exceeds u32".to_string())?,
            rad_dim: u32::try_from(cfg.radial_dim())
                .map_err(|_| "rad_dim exceeds u32".to_string())?,
            angular_formula: cfg.angular_formula.gpu_code(),
            r_cut_radial: cfg.r_cut_radial as f32,
            r_cut_angular: cfg.r_cut_angular as f32,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("legolas-aev-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let atoms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("legolas-atoms"),
            contents: bytemuck::cast_slice(&scene.gpu_atoms),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let cells_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("legolas-cells"),
            contents: bytemuck::cast_slice(&scene.gpu_cells),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let starts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("legolas-bucket-starts"),
            contents: bytemuck::cast_slice(&scene.bucket_starts),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let sorted_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("legolas-sorted-atoms"),
            contents: bytemuck::cast_slice(&scene.sorted_atoms),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let queries_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("legolas-query-indices"),
            contents: bytemuck::cast_slice(&query_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tables_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("legolas-aev-tables"),
            contents: bytemuck::cast_slice(&tables),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let aligned_out_bytes =
            out_bytes.div_ceil(wgpu::COPY_BUFFER_ALIGNMENT) * wgpu::COPY_BUFFER_ALIGNMENT;
        let aev_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("legolas-aev-out"),
            size: aligned_out_bytes.max(wgpu::COPY_BUFFER_ALIGNMENT),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("legolas-aev-bg"),
            layout: &self.aev_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: atoms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cells_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: starts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sorted_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: queries_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: tables_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: aev_out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("legolas-aev-enc"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("legolas-aev-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.aev_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg_x, 1, 1);
        }
        self.engine.queue.submit(Some(encoder.finish()));

        Ok(aev_out_buf)
    }

    async fn run_homogeneous_members_on_aev_buffer(
        &self,
        members: &[&LegolasMlp],
        activation: nmr_core::legolas_shifts::ActivationKind,
        output_mean: f64,
        output_scale: f64,
        aev_buf: &wgpu::Buffer,
        n_queries: usize,
    ) -> Result<Vec<f32>, String> {
        if members.is_empty() || n_queries == 0 {
            return Ok(Vec::new());
        }
        let device = &self.engine.device;
        let limits = device.limits();
        let n_members = members.len();
        let n_layers = members[0].layers.len();
        let last_layer = n_layers - 1;
        let empty_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("legolas-empty-bg"),
            layout: &self.empty_bgl,
            entries: &[],
        });

        let mut current_owned: Option<wgpu::Buffer> = None;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("legolas-mlp-enc"),
        });

        // Keep per-layer GPU resources alive until encoder submission finishes
        let mut live_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(n_layers * 4);
        let mut live_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(n_layers);
        let mut dispatch_wgs: Vec<u32> = Vec::with_capacity(n_layers);

        for li in 0..n_layers {
            let in_dim = members[0].layers[li].in_dim;
            let out_dim = members[0].layers[li].out_dim;
            let total_jobs = n_queries
                .checked_mul(n_members)
                .and_then(|v| v.checked_mul(out_dim))
                .ok_or_else(|| "GPU MLP job count overflow".to_string())?;
            let wg_x = check_workgroups_1d(&limits, "mlp_layer_forward", total_jobs)?;

            let mut packed_w = Vec::with_capacity(n_members * out_dim * in_dim);
            let mut packed_b = Vec::with_capacity(n_members * out_dim);
            for m in members {
                packed_w.extend(m.layers[li].weights.iter().map(|&v| v as f32));
                packed_b.extend(m.layers[li].biases.iter().map(|&v| v as f32));
            }

            let out_bytes = u64::try_from(
                total_jobs
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| "GPU MLP output byte size overflow".to_string())?,
            )
            .map_err(|_| "GPU MLP output byte size overflow".to_string())?;
            let aligned_out_bytes =
                out_bytes.div_ceil(wgpu::COPY_BUFFER_ALIGNMENT) * wgpu::COPY_BUFFER_ALIGNMENT;

            check_uniform_limit(
                &limits,
                "mlp_params",
                std::mem::size_of::<MlpLayerParamsGpu>() as u64,
            )?;
            check_storage_limit(
                &limits,
                "mlp_weights",
                (packed_w.len() * std::mem::size_of::<f32>()) as u64,
            )?;
            check_storage_limit(
                &limits,
                "mlp_biases",
                (packed_b.len() * std::mem::size_of::<f32>()) as u64,
            )?;
            check_storage_limit(&limits, "mlp_out", aligned_out_bytes)?;

            let is_hidden = li < last_layer;
            let params = MlpLayerParamsGpu {
                n_queries: u32::try_from(n_queries)
                    .map_err(|_| "n_queries exceeds u32".to_string())?,
                n_members: u32::try_from(n_members)
                    .map_err(|_| "n_members exceeds u32".to_string())?,
                in_dim: u32::try_from(in_dim).map_err(|_| "in_dim exceeds u32".to_string())?,
                out_dim: u32::try_from(out_dim).map_err(|_| "out_dim exceeds u32".to_string())?,
                activation_kind: if is_hidden { activation.gpu_code() } else { 0 },
                broadcast_input: u32::from(li == 0),
                apply_output_scale: u32::from(li == last_layer),
                _pad0: 0,
                celu_alpha: activation.celu_alpha_or_default() as f32,
                output_mean: output_mean as f32,
                output_scale: output_scale as f32,
                _pad1: 0.0,
            };

            let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("legolas-mlp-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let w_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("legolas-mlp-weights"),
                contents: bytemuck::cast_slice(&packed_w),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("legolas-mlp-biases"),
                contents: bytemuck::cast_slice(&packed_b),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let out_usage = if li == last_layer {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
            } else {
                wgpu::BufferUsages::STORAGE
            };
            let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("legolas-mlp-out"),
                size: aligned_out_bytes.max(wgpu::COPY_BUFFER_ALIGNMENT),
                usage: out_usage,
                mapped_at_creation: false,
            });

            let in_ref: &wgpu::Buffer = match current_owned.as_ref() {
                Some(prev) => prev,
                None => aev_buf,
            };
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("legolas-mlp-bg"),
                layout: &self.mlp_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: in_ref.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: w_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            });

            live_buffers.push(params_buf);
            live_buffers.push(w_buf);
            live_buffers.push(b_buf);
            if let Some(prev) = current_owned.take() {
                live_buffers.push(prev);
            }
            current_owned = Some(out_buf);
            live_bgs.push(bg);
            dispatch_wgs.push(wg_x);
        }

        for (bg, &wg_x) in live_bgs.iter().zip(dispatch_wgs.iter()) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("legolas-mlp-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mlp_pipeline);
            pass.set_bind_group(0, &empty_bg, &[]);
            pass.set_bind_group(1, bg, &[]);
            pass.dispatch_workgroups(wg_x, 1, 1);
        }
        self.engine.queue.submit(Some(encoder.finish()));

        let final_buf = current_owned
            .as_ref()
            .ok_or_else(|| "GPU MLP pipeline produced no output buffer".to_string())?;
        let out_count = n_queries * n_members;
        let out: Vec<f32> = self
            .engine
            .read_buffer_vec_async(final_buf, out_count)
            .await?;
        drop(live_bgs);
        drop(live_buffers);
        Ok(out)
    }

    async fn evaluate_ensemble_on_aev_buffer(
        &self,
        ensemble: &NucleusEnsemble,
        aev_buf: &wgpu::Buffer,
        n_queries: usize,
    ) -> Result<Vec<NucleusShiftEstimate>, String> {
        let n_members = ensemble.members.len();
        let flat_preds: Vec<f32> = if ensemble.is_homogeneous_architecture() {
            let refs: Vec<&LegolasMlp> = ensemble.members.iter().collect();
            self.run_homogeneous_members_on_aev_buffer(
                &refs,
                ensemble.activation,
                ensemble.output_mean,
                ensemble.output_scale,
                aev_buf,
                n_queries,
            )
            .await?
        } else {
            let mut combined = vec![0.0_f32; n_queries * n_members];
            for (m_idx, mlp) in ensemble.members.iter().enumerate() {
                let single = self
                    .run_homogeneous_members_on_aev_buffer(
                        &[mlp],
                        ensemble.activation,
                        ensemble.output_mean,
                        ensemble.output_scale,
                        aev_buf,
                        n_queries,
                    )
                    .await?;
                for q in 0..n_queries {
                    combined[q * n_members + m_idx] = single[q];
                }
            }
            combined
        };

        let mut estimates = Vec::with_capacity(n_queries);
        for q in 0..n_queries {
            let slice = &flat_preds[q * n_members..(q + 1) * n_members];
            let member_predictions_ppm: Vec<f64> = slice.iter().map(|&v| v as f64).collect();
            if member_predictions_ppm.iter().any(|v| !v.is_finite()) {
                return Err(format!(
                    "GPU LEGOLAS ensemble {:?} produced non-finite prediction at query {q}",
                    ensemble.nucleus
                ));
            }
            let mean_ppm =
                member_predictions_ppm.iter().sum::<f64>() / (member_predictions_ppm.len() as f64);
            let sigma_ens_ppm = if member_predictions_ppm.len() > 1 {
                let var = member_predictions_ppm
                    .iter()
                    .map(|&p| {
                        let d = p - mean_ppm;
                        d * d
                    })
                    .sum::<f64>()
                    / ((member_predictions_ppm.len() - 1) as f64);
                var.max(0.0).sqrt()
            } else {
                0.0
            };
            estimates.push(NucleusShiftEstimate {
                nucleus: ensemble.nucleus,
                mean_ppm,
                sigma_ens_ppm,
                member_predictions_ppm,
            });
        }
        Ok(estimates)
    }

    /// Async (`wasm32`-compatible) WebGPU computation of ANI-AEV vectors for `query_indices`.
    pub async fn compute_aevs_gpu_async(
        &self,
        atoms: &[LegolasAtom],
        query_indices: &[usize],
        cfg: &AevConfig,
    ) -> Result<Vec<Vec<f32>>, String> {
        cfg.validate()?;
        validate_atoms(atoms)?;
        if query_indices.is_empty() {
            return Ok(Vec::new());
        }
        let scene = prepare_atom_scene(atoms, cfg)?;
        scene.validate_query_neighbor_caps(atoms, query_indices, cfg)?;
        let aev_dim = cfg.total_dim();
        let aev_buf = self.dispatch_aev_into_buffer(&scene, query_indices, cfg)?;
        let flat: Vec<f32> = self
            .engine
            .read_buffer_vec_async(&aev_buf, query_indices.len() * aev_dim)
            .await?;
        let mut rows = Vec::with_capacity(query_indices.len());
        for q in 0..query_indices.len() {
            rows.push(flat[q * aev_dim..(q + 1) * aev_dim].to_vec());
        }
        Ok(rows)
    }

    /// Synchronous native WebGPU computation of ANI-AEV vectors for `query_indices`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn compute_aevs_gpu(
        &self,
        atoms: &[LegolasAtom],
        query_indices: &[usize],
        cfg: &AevConfig,
    ) -> Result<Vec<Vec<f32>>, String> {
        pollster::block_on(self.compute_aevs_gpu_async(atoms, query_indices, cfg))
    }

    /// Async (`wasm32`-compatible) WebGPU ensemble MLP inference over precomputed flat `f32` AEVs.
    pub async fn predict_ensemble_aevs_gpu_async(
        &self,
        ensemble: &NucleusEnsemble,
        aevs_flat: &[f32],
        n_queries: usize,
    ) -> Result<Vec<NucleusShiftEstimate>, String> {
        if n_queries == 0 {
            return Ok(Vec::new());
        }
        let aev_dim = ensemble.members.first().map_or(0, LegolasMlp::input_dim);
        ensemble.validate(aev_dim)?;
        if aevs_flat.len() != n_queries * aev_dim {
            return Err(format!(
                "aevs_flat length {} != n_queries ({n_queries}) * aev_dim ({aev_dim})",
                aevs_flat.len()
            ));
        }

        let max_out_dim = ensemble
            .members
            .iter()
            .flat_map(|m| m.layers.iter().map(|l| l.out_dim))
            .max()
            .unwrap_or(1)
            .max(1);
        let jobs_per_query = ensemble.members.len().saturating_mul(max_out_dim).max(1);
        let chunk_q = (MAX_MLP_JOBS_PER_DISPATCH / jobs_per_query).max(1);

        let limits = self.engine.device.limits();
        let mut all_estimates = Vec::with_capacity(n_queries);
        let mut start_q = 0usize;
        while start_q < n_queries {
            let end_q = (start_q + chunk_q).min(n_queries);
            let q_len = end_q - start_q;
            let slice = &aevs_flat[start_q * aev_dim..end_q * aev_dim];
            let bytes = std::mem::size_of_val(slice) as u64;
            check_storage_limit(&limits, "aev_in", bytes)?;
            let aev_buf =
                self.engine
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("legolas-aev-in"),
                        contents: bytemuck::cast_slice(slice),
                        usage: wgpu::BufferUsages::STORAGE,
                    });
            let ests = self
                .evaluate_ensemble_on_aev_buffer(ensemble, &aev_buf, q_len)
                .await?;
            all_estimates.extend(ests);
            start_q = end_q;
        }
        Ok(all_estimates)
    }

    /// Synchronous native WebGPU ensemble MLP inference over precomputed flat `f32` AEVs.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn predict_ensemble_aevs_gpu(
        &self,
        ensemble: &NucleusEnsemble,
        aevs_flat: &[f32],
        n_queries: usize,
    ) -> Result<Vec<NucleusShiftEstimate>, String> {
        pollster::block_on(self.predict_ensemble_aevs_gpu_async(ensemble, aevs_flat, n_queries))
    }

    /// Async (`wasm32`-compatible) end-to-end WebGPU `AEV + MLP` backbone chemical-shift prediction.
    ///
    /// **Fails closed** with `Err(LegolasError::UntrainedWeights { .. })` whenever
    /// `!model.metadata.weights_trained` (`22.12:P1-1`).
    pub async fn predict_shifts_gpu_async(
        &self,
        model: &LegolasModel,
        atoms: &[LegolasAtom],
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        self.predict_shifts_gpu_with_options_async(model, atoms, &LegolasPredictOptions::default())
            .await
    }

    /// Async (`wasm32`-compatible) end-to-end WebGPU `AEV + MLP` backbone chemical-shift prediction
    /// with explicit inference options.
    ///
    /// Keeps computed AEV tensors resident on GPU between `compute_aev` and `mlp_layer_forward`
    /// and returns predictions in exact atom-index order matching `LegolasModel::predict_shifts`.
    pub async fn predict_shifts_gpu_with_options_async(
        &self,
        model: &LegolasModel,
        atoms: &[LegolasAtom],
        options: &LegolasPredictOptions,
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        model.validate().map_err(LegolasError::Validation)?;
        if !model.metadata.weights_trained && !options.allow_untrained_for_testing {
            return Err(LegolasError::UntrainedWeights {
                provenance: model.metadata.weights_provenance.clone(),
            });
        }
        validate_atoms(atoms).map_err(LegolasError::Validation)?;

        let mut by_nucleus: BTreeMap<TargetNucleus, Vec<usize>> = BTreeMap::new();
        for (idx, atom) in atoms.iter().enumerate() {
            if let Some(nuc) = atom.target_nucleus {
                if model.ensembles.contains_key(&nuc) {
                    by_nucleus.entry(nuc).or_default().push(idx);
                }
            }
        }
        if by_nucleus.is_empty() {
            return Ok(Vec::new());
        }

        let scene =
            prepare_atom_scene(atoms, &model.aev_config).map_err(LegolasError::Validation)?;
        for indices in by_nucleus.values() {
            scene
                .validate_query_neighbor_caps(atoms, indices, &model.aev_config)
                .map_err(LegolasError::Validation)?;
        }
        let mut results: Vec<AtomShiftPrediction> = Vec::new();

        for (nuc, indices) in by_nucleus {
            let Some(ens) = model.ensembles.get(&nuc) else {
                continue;
            };
            let max_out_dim = ens
                .members
                .iter()
                .flat_map(|m| m.layers.iter().map(|l| l.out_dim))
                .max()
                .unwrap_or(1)
                .max(1);
            let jobs_per_query = ens.members.len().saturating_mul(max_out_dim).max(1);
            let chunk_q = (MAX_MLP_JOBS_PER_DISPATCH / jobs_per_query).max(1);

            for chunk in indices.chunks(chunk_q) {
                let aev_buf = self
                    .dispatch_aev_into_buffer(&scene, chunk, &model.aev_config)
                    .map_err(LegolasError::Gpu)?;
                let estimates = self
                    .evaluate_ensemble_on_aev_buffer(ens, &aev_buf, chunk.len())
                    .await
                    .map_err(LegolasError::Gpu)?;
                for (&atom_idx, estimate) in chunk.iter().zip(estimates) {
                    let atom = &atoms[atom_idx];
                    results.push(AtomShiftPrediction {
                        atom_index: atom_idx,
                        residue_seq_id: atom.residue_seq_id,
                        atom_name: atom.atom_name.clone(),
                        estimate,
                    });
                }
            }
        }

        results.sort_by_key(|p| p.atom_index);
        Ok(results)
    }

    /// Synchronous native end-to-end WebGPU `AEV + MLP` backbone chemical-shift prediction
    /// (fails closed on untrained weights).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn predict_shifts_gpu(
        &self,
        model: &LegolasModel,
        atoms: &[LegolasAtom],
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        pollster::block_on(self.predict_shifts_gpu_async(model, atoms))
    }

    /// Synchronous native end-to-end WebGPU `AEV + MLP` prediction with explicit options.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn predict_shifts_gpu_with_options(
        &self,
        model: &LegolasModel,
        atoms: &[LegolasAtom],
        options: &LegolasPredictOptions,
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        pollster::block_on(self.predict_shifts_gpu_with_options_async(model, atoms, options))
    }

    /// Async (`wasm32`-compatible) alias for `predict_shifts_gpu_async` (fails closed on untrained
    /// weights).
    pub async fn predict_atoms_gpu_async(
        &self,
        model: &LegolasModel,
        atoms: &[LegolasAtom],
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        self.predict_shifts_gpu_async(model, atoms).await
    }

    /// Async (`wasm32`-compatible) alias for `predict_shifts_gpu_with_options_async`.
    pub async fn predict_atoms_gpu_with_options_async(
        &self,
        model: &LegolasModel,
        atoms: &[LegolasAtom],
        options: &LegolasPredictOptions,
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        self.predict_shifts_gpu_with_options_async(model, atoms, options)
            .await
    }

    /// Synchronous native alias for `predict_shifts_gpu` (fails closed on untrained weights).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn predict_atoms_gpu(
        &self,
        model: &LegolasModel,
        atoms: &[LegolasAtom],
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        self.predict_shifts_gpu(model, atoms)
    }

    /// Synchronous native alias for `predict_shifts_gpu_with_options`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn predict_atoms_gpu_with_options(
        &self,
        model: &LegolasModel,
        atoms: &[LegolasAtom],
        options: &LegolasPredictOptions,
    ) -> Result<Vec<AtomShiftPrediction>, LegolasError> {
        self.predict_shifts_gpu_with_options(model, atoms, options)
    }
}
