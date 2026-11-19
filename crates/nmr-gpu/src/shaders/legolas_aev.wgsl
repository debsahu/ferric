// Clean-room WebGPU WGSL compute shaders for LEGOLAS ANI Atomic Environment Vector (AEV)
// radial + modified angular descriptor evaluation and per-nucleus ensemble MLP inference.
//
// Provenance:
// - Darrows, M. A., Kodituwakku, M., Xue, K., Pickering, I., Terrel, J. & Roitberg, A. E.,
//   J. Chem. Theory Comput. 2025, 21(8), 4266-4275 (DOI: 10.1021/acs.jctc.5c00026;
//   https://github.com/roitberg-group/legolas, MIT License).
// - Smith, J. S., Isayev, O. & Roitberg, A. E., Chem. Sci. 2017, 8, 3192-3203 (ANI-1 AEV).
// - Behler, J. & Parrinello, M., Phys. Rev. Lett. 2007, 98, 146401.

const PI_F32: f32 = 3.14159265358979323846;

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
};

struct GpuAtom {
    x: f32,
    y: f32,
    z: f32,
    species: u32,
};

struct GpuCell {
    cx: i32,
    cy: i32,
    cz: i32,
    _pad: u32,
};

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
};

// Group 0: ANI-AEV radial + modified angular descriptor evaluation
@group(0) @binding(0) var<uniform> aev_params: AevParamsGpu;
@group(0) @binding(1) var<storage, read> atoms: array<GpuAtom>;
@group(0) @binding(2) var<storage, read> atom_cells: array<GpuCell>;
@group(0) @binding(3) var<storage, read> bucket_starts: array<u32>;
@group(0) @binding(4) var<storage, read> sorted_atoms: array<u32>;
@group(0) @binding(5) var<storage, read> query_indices: array<u32>;
@group(0) @binding(6) var<storage, read> aev_tables: array<f32>;
@group(0) @binding(7) var<storage, read_write> aev_out: array<f32>;

// Group 1: Batched ensemble MLP dense layer forward pass
@group(1) @binding(0) var<uniform> mlp_params: MlpLayerParamsGpu;
@group(1) @binding(1) var<storage, read> mlp_in: array<f32>;
@group(1) @binding(2) var<storage, read> mlp_weights: array<f32>;
@group(1) @binding(3) var<storage, read> mlp_biases: array<f32>;
@group(1) @binding(4) var<storage, read_write> mlp_out: array<f32>;

fn spatial_hash(cx: i32, cy: i32, cz: i32) -> u32 {
    let h = (bitcast<u32>(cx) * 73856093u)
        ^ (bitcast<u32>(cy) * 19349663u)
        ^ (bitcast<u32>(cz) * 83492791u);
    return h ^ (h >> 16u);
}

fn cosine_cutoff(r: f32, r_cut: f32) -> f32 {
    if (r <= 0.0) {
        return 1.0;
    }
    if (r >= r_cut) {
        return 0.0;
    }
    return 0.5 * cos(PI_F32 * r / r_cut) + 0.5;
}

fn species_pair_index(a: u32, b: u32) -> u32 {
    let lo = min(a, b);
    let hi = max(a, b);
    return (lo * (11u - lo)) / 2u + (hi - lo);
}

fn safe_tanh(x: f32) -> f32 {
    let xc = clamp(x, -9.5, 9.5);
    let e2 = exp(2.0 * xc);
    return (e2 - 1.0) / (e2 + 1.0);
}

fn eval_activation(x: f32, kind: u32, celu_alpha: f32) -> f32 {
    if (kind == 1u) {
        if (x >= 0.0) {
            return x;
        }
        let z = clamp(x / celu_alpha, -40.0, 0.0);
        return celu_alpha * (exp(z) - 1.0);
    }
    if (kind == 2u) {
        let inner = clamp(0.79788456 * (x + 0.044715 * x * x * x), -9.5, 9.5);
        return 0.5 * x * (1.0 + safe_tanh(inner));
    }
    if (kind == 3u) {
        let xc = clamp(x, -20.0, 20.0);
        let sig = 1.0 / (1.0 + exp(-xc));
        return x * sig;
    }
    return x;
}

@compute @workgroup_size(64, 1, 1)
fn compute_aev(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= aev_params.n_queries) {
        return;
    }

    let atom_i = query_indices[q];
    let out_base = q * aev_params.aev_dim;
    for (var d: u32 = 0u; d < aev_params.aev_dim; d = d + 1u) {
        aev_out[out_base + d] = 0.0;
    }

    let pos_i = vec3<f32>(atoms[atom_i].x, atoms[atom_i].y, atoms[atom_i].z);
    let cell_i = atom_cells[atom_i];

    let off_eta_r = 0u;
    let off_rs_r = aev_params.n_eta_r;
    let off_eta_a = off_rs_r + aev_params.n_rs_r;
    let off_rs_a = off_eta_a + aev_params.n_eta_a;
    let off_zeta = off_rs_a + aev_params.n_rs_a;
    let off_theta = off_zeta + aev_params.n_zeta;

    let rad_per_sp = aev_params.n_eta_r * aev_params.n_rs_r;
    let zt_len = aev_params.n_zeta * aev_params.n_theta;
    let ang_per_pair = aev_params.n_eta_a * aev_params.n_rs_a * zt_len;
    let r_cut_rad = aev_params.r_cut_radial;
    let r_cut_ang = aev_params.r_cut_angular;
    let r_cut_rad_sq = r_cut_rad * r_cut_rad;
    let r_cut_ang_sq = r_cut_ang * r_cut_ang;

    for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let tc = vec3<i32>(cell_i.cx + dx, cell_i.cy + dy, cell_i.cz + dz);
                let bucket = spatial_hash(tc.x, tc.y, tc.z) & aev_params.bucket_mask;
                let b_start = bucket_starts[bucket];
                let b_end = bucket_starts[bucket + 1u];

                for (var p: u32 = b_start; p < b_end; p = p + 1u) {
                    let atom_j = sorted_atoms[p];
                    if (atom_j == atom_i) {
                        continue;
                    }
                    let cj = atom_cells[atom_j];
                    if (cj.cx != tc.x || cj.cy != tc.y || cj.cz != tc.z) {
                        continue;
                    }

                    let aj = atoms[atom_j];
                    let d_ij = vec3<f32>(aj.x - pos_i.x, aj.y - pos_i.y, aj.z - pos_i.z);
                    let r_ij_sq = dot(d_ij, d_ij);
                    if (r_ij_sq <= 1e-12) {
                        continue;
                    }
                    if (r_ij_sq > r_cut_rad_sq && r_ij_sq > r_cut_ang_sq) {
                        continue;
                    }

                    let r_ij = sqrt(r_ij_sq);

                    // Radial symmetry functions
                    if (r_ij <= r_cut_rad) {
                        let fc_r = cosine_cutoff(r_ij, r_cut_rad);
                        if (fc_r > 0.0) {
                            let sp_base = out_base + aj.species * rad_per_sp;
                            for (var ei: u32 = 0u; ei < aev_params.n_eta_r; ei = ei + 1u) {
                                let eta_r = aev_tables[off_eta_r + ei];
                                let row_base = sp_base + ei * aev_params.n_rs_r;
                                for (var ri: u32 = 0u; ri < aev_params.n_rs_r; ri = ri + 1u) {
                                    let rs_r = aev_tables[off_rs_r + ri];
                                    let dr = r_ij - rs_r;
                                    let exponent = -eta_r * dr * dr;
                                    if (exponent > -80.0) {
                                        aev_out[row_base + ri] =
                                            aev_out[row_base + ri] + exp(exponent) * fc_r;
                                    }
                                }
                            }
                        }
                    }

                    // Modified ANI angular symmetry functions over unordered pairs (j < k)
                    if (r_ij <= r_cut_ang) {
                        let fc_a_j = cosine_cutoff(r_ij, r_cut_ang);
                        if (fc_a_j <= 0.0) {
                            continue;
                        }
                        for (var dz2: i32 = -1; dz2 <= 1; dz2 = dz2 + 1) {
                            for (var dy2: i32 = -1; dy2 <= 1; dy2 = dy2 + 1) {
                                for (var dx2: i32 = -1; dx2 <= 1; dx2 = dx2 + 1) {
                                    let tc2 = vec3<i32>(
                                        cell_i.cx + dx2,
                                        cell_i.cy + dy2,
                                        cell_i.cz + dz2
                                    );
                                    let b2 = spatial_hash(tc2.x, tc2.y, tc2.z)
                                        & aev_params.bucket_mask;
                                    let b2_start = bucket_starts[b2];
                                    let b2_end = bucket_starts[b2 + 1u];

                                    for (var p2: u32 = b2_start; p2 < b2_end; p2 = p2 + 1u) {
                                        let atom_k = sorted_atoms[p2];
                                        if (atom_k <= atom_j || atom_k == atom_i) {
                                            continue;
                                        }
                                        let ck = atom_cells[atom_k];
                                        if (ck.cx != tc2.x || ck.cy != tc2.y || ck.cz != tc2.z) {
                                            continue;
                                        }

                                        let ak = atoms[atom_k];
                                        let d_ik = vec3<f32>(
                                            ak.x - pos_i.x,
                                            ak.y - pos_i.y,
                                            ak.z - pos_i.z
                                        );
                                        let r_ik_sq = dot(d_ik, d_ik);
                                        if (r_ik_sq <= 1e-12 || r_ik_sq > r_cut_ang_sq) {
                                            continue;
                                        }
                                        let r_ik = sqrt(r_ik_sq);
                                        let fc_a_k = cosine_cutoff(r_ik, r_cut_ang);
                                        if (fc_a_k <= 0.0) {
                                            continue;
                                        }

                                        var theta: f32;
                                        if (aev_params.angular_formula == 1u) {
                                            let cos_t = clamp(
                                                dot(d_ij, d_ik) / max(r_ij * r_ik, 1e-12),
                                                -1.0,
                                                1.0
                                            );
                                            let scaled_cos = clamp(0.95 * cos_t, -0.95, 0.95);
                                            // Use atan2(sin, cos) instead of acos(x): Mesa Lavapipe's SPIR-V acos
                                            // polynomial has ~1e-4 error, amplified by zeta=8 to ~8e-4.
                                            theta = atan2(
                                                sqrt(max(0.0, 1.0 - scaled_cos * scaled_cos)),
                                                scaled_cos
                                            );
                                        } else {
                                            theta = atan2(length(cross(d_ij, d_ik)), dot(d_ij, d_ik));
                                        }
                                        let r_mid = 0.5 * (r_ij + r_ik);
                                        let fc_prod = fc_a_j * fc_a_k;
                                        let pair_idx = species_pair_index(aj.species, ak.species);
                                        let pair_base =
                                            out_base + aev_params.rad_dim + pair_idx * ang_per_pair;

                                        for (var ei: u32 = 0u; ei < aev_params.n_eta_a; ei = ei + 1u) {
                                            let eta_a = aev_tables[off_eta_a + ei];
                                            for (var ri: u32 = 0u; ri < aev_params.n_rs_a; ri = ri + 1u) {
                                                let rs_a = aev_tables[off_rs_a + ri];
                                                let dm = r_mid - rs_a;
                                                let exponent = -eta_a * dm * dm;
                                                if (exponent <= -80.0) {
                                                    continue;
                                                }
                                                let rad_scale = exp(exponent) * fc_prod;
                                                let slot_base =
                                                    pair_base + (ei * aev_params.n_rs_a + ri) * zt_len;

                                                for (var zi: u32 = 0u; zi < aev_params.n_zeta; zi = zi + 1u) {
                                                    let zeta = aev_tables[off_zeta + zi];
                                                    let z_base = slot_base + zi * aev_params.n_theta;
                                                    for (var ti: u32 = 0u; ti < aev_params.n_theta; ti = ti + 1u) {
                                                        let theta_s = aev_tables[off_theta + ti];
                                                        let half_u = 0.5 * (1.0 + cos(theta - theta_s));
                                                        if (half_u > 1e-12) {
                                                            let ang_part = 2.0 * pow(half_u, zeta);
                                                            aev_out[z_base + ti] =
                                                                aev_out[z_base + ti]
                                                                + ang_part * rad_scale;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@compute @workgroup_size(64, 1, 1)
fn mlp_layer_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_jobs = mlp_params.n_queries * mlp_params.n_members * mlp_params.out_dim;
    let job = gid.x;
    if (job >= total_jobs) {
        return;
    }

    let o = job % mlp_params.out_dim;
    let qe = job / mlp_params.out_dim;
    let e = qe % mlp_params.n_members;
    let q = qe / mlp_params.n_members;

    let in_base = select(
        (q * mlp_params.n_members + e) * mlp_params.in_dim,
        q * mlp_params.in_dim,
        mlp_params.broadcast_input != 0u
    );
    let w_base = (e * mlp_params.out_dim + o) * mlp_params.in_dim;

    var acc: f32 = mlp_biases[e * mlp_params.out_dim + o];
    for (var i: u32 = 0u; i < mlp_params.in_dim; i = i + 1u) {
        acc = acc + mlp_weights[w_base + i] * mlp_in[in_base + i];
    }

    acc = eval_activation(acc, mlp_params.activation_kind, mlp_params.celu_alpha);
    if (mlp_params.apply_output_scale != 0u) {
        acc = mlp_params.output_mean + mlp_params.output_scale * acc;
    }

    mlp_out[job] = acc;
}
