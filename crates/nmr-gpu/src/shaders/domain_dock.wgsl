// Batched multi-replica WebGPU rigid-domain + covalent NeRF flexible-linker & multi-chain SE(3)
// hybrid RDC / PCS / PRE / inter-domain NOE / soft-core steric / Ramachandran energy & analytical
// torque-gradient kernel (`T21.5`, `docs/specs/21-alphafold-integrative-nmr.md` §5.3).
//
// Provenance: Clean-room WGSL compute implementation derived from peer-reviewed literature
// (`0` const-eval NaN/Inf bitcasts; 100% Chromium Dawn / Tint & wgpu/naga clean):
// - Tjandra, N. & Bax, A. (1997), Science 278, 1111-1114; Losonczi, J. A. et al. (1999),
//   J. Magn. Reson. 138, 334-342 (multi-medium Saupe alignment tensor RDC back-calculation).
// - Bertini, I. et al. (2007), J. Am. Chem. Soc. 129, 12786-12794; Schmitz, C. et al. (2008),
//   J. Biomol. NMR 41, 179-189 (traceless DeltaChi pseudocontact shift energy, translational force
//   & host-domain rotational torque).
// - Solomon, I. (1955), Phys. Rev. 99, 559-565; Clore, G. M. & Schwieters, C. D. (2004),
//   J. Am. Chem. Soc. 126, 2923-2938 (paramagnetic relaxation enhancement r^-6 restraints).
// - Parsons, J. et al. (2005), J. Comput. Chem. 26, 1063-1068 (NeRF forward kinematics).
// - Abe, H., Braun, W., Noguti, T. & Go, N. (1984), Comput. Chem. 8, 239-247 (O(L) local-pivot
//   backward suffix-sum torque propagation).
// - Lovell, S. C. et al. (2003), Proteins 50, 437-450 (Ramachandran torsion basin prior).

struct DomainDockParams {
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
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: DomainDockParams;
@group(0) @binding(1) var<storage, read> static_floats: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> static_ints: array<u32>;
@group(0) @binding(3) var<storage, read> replica_in: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> scratch: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> energies_out: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> grads_out: array<vec4<f32>>;

const PI_F32: f32 = 3.14159265;
const TWO_PI_F32: f32 = 6.28318531;
const DEG_TO_RAD_F32: f32 = 0.0174532925;
const BOND_N_H_F32: f32 = 1.010;
const ANG_C_N_H_F32: f32 = 2.08043247;
const PCS_CART_PREFACTOR_F32: f32 = 795.77471546;
const NO_BLOCK_SENTINEL: u32 = 4294967295u;

fn exact_rsqrt(x: f32) -> f32 {
    let y0 = inverseSqrt(x);
    return y0 * (1.5 - 0.5 * x * y0 * y0);
}

fn safe_unit3(v: vec3<f32>) -> vec3<f32> {
    let n2 = dot(v, v);
    if (n2 > 1e-20) {
        return v * exact_rsqrt(n2);
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

fn wrap_rad_f32(rad: f32) -> f32 {
    var r = (rad + PI_F32) - TWO_PI_F32 * floor((rad + PI_F32) / TWO_PI_F32);
    if (r <= 0.0) {
        r = r + TWO_PI_F32;
    }
    return r - PI_F32;
}

// Exact bond-displacement NeRF kernel: given `ab = b - a` and `bc = c - b` (`~1.3..1.5 A`),
// returns `cd = d - c` without ever subtracting large world position vectors!
fn nerf_delta3(
    ab: vec3<f32>,
    bc: vec3<f32>,
    r_cd: f32,
    ang_bcd: f32,
    sin_ch: f32,
    cos_ch: f32,
) -> vec3<f32> {
    let ex = safe_unit3(bc);
    let ab_rev = -ab;
    let perp = ab_rev - ex * dot(ab_rev, ex);
    let ey = safe_unit3(perp);
    let ez = cross(ex, ey);

    let sin_th = sin(ang_bcd);
    let cos_th = cos(ang_bcd);

    let dx = -r_cd * cos_th;
    let dy = r_cd * sin_th * cos_ch;
    let dz = r_cd * sin_th * sin_ch;
    return ex * dx + ey * dy + ez * dz;
}

fn nerf_place3(
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    r_cd: f32,
    ang_bcd: f32,
    sin_ch: f32,
    cos_ch: f32,
) -> vec3<f32> {
    return c + nerf_delta3(b - a, c - b, r_cd, ang_bcd, sin_ch, cos_ch);
}

fn quat_rotate_vec(q_raw: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let n2 = dot(q_raw, q_raw);
    var q = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    if (n2 > 1e-20) {
        q = q_raw * exact_rsqrt(n2);
    }
    let w = q.x;
    let u = q.yzw;
    let uv = cross(u, v);
    let uuv = cross(u, uv);
    return v + 2.0 * (uv * w + uuv);
}

fn mat3_rows_mul_vec(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(r0, v), dot(r1, v), dot(r2, v));
}

fn seg_pivot_pos_dock(atom_base: u32, seg: u32) -> vec3<f32> {
    if (seg == 0u) {
        return scratch[atom_base + 0u].xyz;
    }
    if ((seg & 1u) == 1u) {
        let i = (seg - 1u) / 2u;
        return scratch[atom_base + 7u * i + 2u].xyz;
    }
    let i = (seg - 2u) / 2u;
    return scratch[atom_base + 7u * i + 4u].xyz;
}

fn add_seg_grad_dock(
    atom_base: u32,
    force_base: u32,
    torque_base: u32,
    seg: u32,
    pos: vec3<f32>,
    g: vec3<f32>,
) {
    let piv = seg_pivot_pos_dock(atom_base, seg);
    let f_new = scratch[force_base + seg].xyz + g;
    let t_new = scratch[torque_base + seg].xyz + cross(pos - piv, g);
    scratch[force_base + seg] = vec4<f32>(f_new, 0.0);
    scratch[torque_base + seg] = vec4<f32>(t_new, 0.0);
}

fn vdw_min_r2_base_f32(ka: u32, kb: u32) -> f32 {
    let lo = min(ka, kb);
    let hi = max(ka, kb);
    if (lo == 0u && hi == 2u) {
        return 2.65 * 2.65;
    }
    if (lo == 2u && hi == 2u) {
        return 2.80 * 2.80;
    }
    if (lo == 3u && hi == 3u) {
        return 3.25 * 3.25;
    }
    if (hi == 3u) {
        return 3.00 * 3.00;
    }
    return 2.85 * 2.85;
}

// Lovell et al. (2003) smooth log-sum-exp Gaussian Ramachandran potential -> vec3(V, dV/dphi, dV/dpsi)
fn rama_energy_and_grad_f32(rc: u32, phi: f32, psi: f32) -> vec3<f32> {
    var n_b: u32 = 6u;
    var p0 = array<f32, 7>(-63.0, -122.0, -138.0, -75.0, -85.0, 56.0, 0.0);
    var s0 = array<f32, 7>(-42.0, 135.0, 168.0, 145.0, -12.0, 38.0, 0.0);
    var sp = array<f32, 7>(22.0, 28.0, 26.0, 22.0, 20.0, 16.0, 20.0);
    var ss = array<f32, 7>(22.0, 28.0, 24.0, 24.0, 22.0, 18.0, 20.0);
    var amp = array<f32, 7>(1.00, 1.00, 0.92, 0.85, 0.40, 0.25, 1.00);

    if (rc == 2u) {
        n_b = 2u;
        p0[0] = -63.0; s0[0] = 145.0; sp[0] = 13.0; ss[0] = 24.0; amp[0] = 1.00;
        p0[1] = -61.0; s0[1] = -38.0; sp[1] = 13.0; ss[1] = 22.0; amp[1] = 0.95;
    } else if (rc == 1u) {
        n_b = 7u;
        p0[0] = -65.0; s0[0] = -41.0; sp[0] = 30.0; ss[0] = 30.0; amp[0] = 1.00;
        p0[1] = 65.0;  s0[1] = 41.0;  sp[1] = 30.0; ss[1] = 30.0; amp[1] = 1.00;
        p0[2] = -80.0; s0[2] = 155.0; sp[2] = 34.0; ss[2] = 34.0; amp[2] = 0.95;
        p0[3] = 80.0;  s0[3] = -155.0;sp[3] = 34.0; ss[3] = 34.0; amp[3] = 0.95;
        p0[4] = 82.0;  s0[4] = 12.0;  sp[4] = 30.0; ss[4] = 30.0; amp[4] = 0.90;
        p0[5] = -82.0; s0[5] = -12.0; sp[5] = 30.0; ss[5] = 30.0; amp[5] = 0.90;
        p0[6] = 180.0; s0[6] = 180.0; sp[6] = 36.0; ss[6] = 36.0; amp[6] = 0.85;
    }

    let floor_log: f32 = -9.0;
    var max_log: f32 = floor_log;
    var logs = array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var dlog_p = array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var dlog_s = array<f32, 7>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    for (var bi: u32 = 0u; bi < n_b; bi = bi + 1u) {
        let dp = wrap_rad_f32(phi - p0[bi] * DEG_TO_RAD_F32);
        let ds = wrap_rad_f32(psi - s0[bi] * DEG_TO_RAD_F32);
        let sp_r = sp[bi] * DEG_TO_RAD_F32;
        let ss_r = ss[bi] * DEG_TO_RAD_F32;
        let inv_vp = 1.0 / (sp_r * sp_r);
        let inv_vs = 1.0 / (ss_r * ss_r);
        let lg = log(amp[bi]) - 0.5 * (dp * dp * inv_vp + ds * ds * inv_vs);
        logs[bi] = lg;
        dlog_p[bi] = -dp * inv_vp;
        dlog_s[bi] = -ds * inv_vs;
        if (lg > max_log) {
            max_log = lg;
        }
    }

    var sum_exp = exp(floor_log - max_log);
    var num_p: f32 = 0.0;
    var num_s: f32 = 0.0;
    for (var bi: u32 = 0u; bi < n_b; bi = bi + 1u) {
        let w = exp(logs[bi] - max_log);
        sum_exp = sum_exp + w;
        num_p = num_p + w * dlog_p[bi];
        num_s = num_s + w * dlog_s[bi];
    }

    let log_p = max_log + log(sum_exp);
    let raw_v = -log_p - 0.35;
    if (raw_v <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return vec3<f32>(raw_v, -num_p / sum_exp, -num_s / sum_exp);
}

@compute @workgroup_size(64)
fn eval_domain_dock_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let conf_idx = gid.x;
    if (conf_idx >= params.n_replicas) {
        return;
    }

    let l = params.l_res;
    let n_blocks = params.n_rigid_blocks;
    let n_chains = params.n_chains;
    let n_media = params.n_media;

    if (l == 0u) {
        energies_out[2u * conf_idx + 0u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        energies_out[2u * conf_idx + 1u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Static float & int table offsets
    let off_bond_geom = 0u;
    let off_ref_tors = 2u * l;
    let off_local_slots = 3u * l;
    let off_c_prev_virt = 10u * l;
    let off_chain_centroids = 13u * l;
    let off_pcs_tags_f = off_chain_centroids + n_chains;
    let off_pre_sites_f = off_pcs_tags_f + 4u * params.n_pcs_tags;
    let off_rdc_f = off_pre_sites_f + params.n_pre_sites;
    let off_pcs_f = off_rdc_f + params.n_rdc_terms;
    let off_pre_f = off_pcs_f + params.n_pcs_terms;
    let off_noe_f = off_pre_f + 2u * params.n_pre_terms;

    let ioff_res = 0u;
    let ioff_chains = 4u * l;
    let ioff_blocks = ioff_chains + 2u * n_chains;
    let ioff_pcs_tags = ioff_blocks + 2u * n_blocks;
    let ioff_pre_sites = ioff_pcs_tags + 2u * params.n_pcs_tags;
    let ioff_rdc = ioff_pre_sites + 2u * params.n_pre_sites;
    let ioff_pcs = ioff_rdc + 5u * params.n_rdc_terms;
    let ioff_pre = ioff_pcs + 3u * params.n_pcs_terms;
    let ioff_noe = ioff_pre + 4u * params.n_pre_terms;

    // Per-replica input, scratch, and gradient strides
    let in_stride = l + 2u * n_chains + 3u * n_media;
    let in_base = conf_idx * in_stride;
    let in_chains_base = in_base + l;
    let in_saupe_base = in_chains_base + 2u * n_chains;

    let n_segs = 2u * l + 1u;
    let scratch_stride = 7u * l + 4u * n_blocks + 2u * n_segs;
    let atom_base = conf_idx * scratch_stride;
    let blk_base = atom_base + 7u * l;
    let force_base = blk_base + 4u * n_blocks;
    let torque_base = force_base + n_segs;

    let grad_stride = l + 2u * n_chains;
    let grad_base = conf_idx * grad_stride;

    for (var s: u32 = 0u; s < n_segs; s = s + 1u) {
        scratch[force_base + s] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        scratch[torque_base + s] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // 1. Forward kinematics across all polymer chains (exact rigid-domain blocks + covalent NeRF linkers)
    for (var c_idx: u32 = 0u; c_idx < n_chains; c_idx = c_idx + 1u) {
        let cs = static_ints[ioff_chains + 2u * c_idx + 0u];
        let ce = static_ints[ioff_chains + 2u * c_idx + 1u];
        let q_c = replica_in[in_chains_base + 2u * c_idx + 0u];
        var tr_c = vec3<f32>(0.0, 0.0, 0.0);
        if (c_idx > 0u) {
            tr_c = replica_in[in_chains_base + 2u * c_idx + 1u].xyz;
        }
        let c_af = static_floats[off_chain_centroids + c_idx].xyz;

        var i: u32 = cs;
        while (i < ce) {
            let r_meta_off = ioff_res + 4u * i;
            let res_class = static_ints[r_meta_off + 0u];
            let bid = static_ints[r_meta_off + 1u];
            let r_flags = static_ints[r_meta_off + 3u];
            let bg0 = static_floats[off_bond_geom + 2u * i + 0u];
            let bg1 = static_floats[off_bond_geom + 2u * i + 1u];

            if (bid != NO_BLOCK_SENTINEL) {
                let b_start = static_ints[ioff_blocks + 2u * bid + 0u];
                let b_end = static_ints[ioff_blocks + 2u * bid + 1u];
                var n_s = vec3<f32>(0.0, 0.0, 0.0);
                var d_nca_s = vec3<f32>(1.0, 0.0, 0.0);
                var d_cac_s = vec3<f32>(0.0, 1.0, 0.0);

                if (i == cs) {
                    let ref_n = static_floats[off_c_prev_virt + 3u * i + 0u].xyz;
                    let ref_ca = static_floats[off_c_prev_virt + 3u * i + 1u].xyz;
                    let ref_c = static_floats[off_c_prev_virt + 3u * i + 2u].xyz;
                    n_s = c_af + tr_c + quat_rotate_vec(q_c, ref_n - c_af);
                    d_nca_s = quat_rotate_vec(q_c, ref_ca - ref_n);
                    d_cac_s = quat_rotate_vec(q_c, ref_c - ref_ca);
                } else {
                    let prev_a = atom_base + 7u * (i - 1u);
                    let n_prev = scratch[prev_a + 0u].xyz;
                    let ca_prev = scratch[prev_a + 2u].xyz;
                    let c_prev = scratch[prev_a + 4u].xyz;
                    let prev_flags = static_ints[ioff_res + 4u * (i - 1u) + 3u];
                    let prev_in = replica_in[in_base + (i - 1u)];
                    let prev_ref = static_floats[off_ref_tors + (i - 1u)];
                    let psi_prev = select(prev_ref.y, prev_in.y, (prev_flags & 2u) != 0u);
                    let omg_prev = prev_in.z;
                    let ref_phi_i = static_floats[off_ref_tors + i].x;

                    let d_prev_nca = ca_prev - n_prev;
                    let d_prev_cac = c_prev - ca_prev;
                    let d_cn = nerf_delta3(d_prev_nca, d_prev_cac, bg0.x, bg0.y, sin(psi_prev), cos(psi_prev));
                    n_s = c_prev + d_cn;
                    d_nca_s = nerf_delta3(d_prev_cac, d_cn, bg0.z, bg0.w, sin(omg_prev), cos(omg_prev));
                    d_cac_s = nerf_delta3(d_cn, d_nca_s, bg1.x, bg1.y, sin(ref_phi_i), cos(ref_phi_i));
                }

                let ex = safe_unit3(d_nca_s);
                let ey = safe_unit3(d_cac_s - ex * dot(d_cac_s, ex));
                let ez = cross(ex, ey);
                // Rows of R_b = [ex, ey, ez] (columns are ex, ey, ez)
                let rb_r0 = vec3<f32>(ex.x, ey.x, ez.x);
                let rb_r1 = vec3<f32>(ex.y, ey.y, ez.y);
                let rb_r2 = vec3<f32>(ex.z, ey.z, ez.z);

                let b_off = blk_base + 4u * bid;
                scratch[b_off + 0u] = vec4<f32>(n_s, 1.0);
                scratch[b_off + 1u] = vec4<f32>(rb_r0, 0.0);
                scratch[b_off + 2u] = vec4<f32>(rb_r1, 0.0);
                scratch[b_off + 3u] = vec4<f32>(rb_r2, 0.0);

                for (var j: u32 = b_start; j < b_end; j = j + 1u) {
                    let j_atom = atom_base + 7u * j;
                    let j_loc = off_local_slots + 7u * j;
                    for (var sl: u32 = 0u; sl < 7u; sl = sl + 1u) {
                        let lv = static_floats[j_loc + sl];
                        if (lv.w > 0.5) {
                            let wp = n_s + mat3_rows_mul_vec(rb_r0, rb_r1, rb_r2, lv.xyz);
                            scratch[j_atom + sl] = vec4<f32>(wp, 1.0);
                        } else {
                            scratch[j_atom + sl] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                        }
                    }
                }

                i = b_end;
                continue;
            }

            // Flexible linker / tail residue `i`
            let a_off = atom_base + 7u * i;
            let loc_off = off_local_slots + 7u * i;
            let cur_in = replica_in[in_base + i];
            let cur_ref = static_floats[off_ref_tors + i];
            let phi_i = select(cur_ref.x, cur_in.x, (r_flags & 1u) != 0u);
            let psi_i = select(cur_ref.y, cur_in.y, (r_flags & 2u) != 0u);

            var n_cur = vec3<f32>(0.0, 0.0, 0.0);
            var ca_cur = vec3<f32>(0.0, 0.0, 0.0);
            var c_cur = vec3<f32>(0.0, 0.0, 0.0);

            if (i == cs) {
                let c_prev_v = c_af + tr_c + quat_rotate_vec(q_c, static_floats[off_c_prev_virt + 3u * i + 0u].xyz - c_af);
                n_cur = c_af + tr_c + quat_rotate_vec(q_c, static_floats[loc_off + 0u].xyz - c_af);
                ca_cur = c_af + tr_c + quat_rotate_vec(q_c, static_floats[loc_off + 2u].xyz - c_af);
                let h_loc = static_floats[loc_off + 1u];
                if (h_loc.w > 0.5) {
                    let h0 = c_af + tr_c + quat_rotate_vec(q_c, h_loc.xyz - c_af);
                    scratch[a_off + 1u] = vec4<f32>(h0, 1.0);
                } else {
                    scratch[a_off + 1u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                }
                c_cur = nerf_place3(c_prev_v, n_cur, ca_cur, bg1.x, bg1.y, sin(phi_i), cos(phi_i));
            } else {
                let prev_a = atom_base + 7u * (i - 1u);
                let n_prev = scratch[prev_a + 0u].xyz;
                let ca_prev = scratch[prev_a + 2u].xyz;
                let c_prev = scratch[prev_a + 4u].xyz;
                let prev_flags = static_ints[ioff_res + 4u * (i - 1u) + 3u];
                let prev_in = replica_in[in_base + (i - 1u)];
                let prev_ref = static_floats[off_ref_tors + (i - 1u)];
                let psi_prev = select(prev_ref.y, prev_in.y, (prev_flags & 2u) != 0u);
                let omg_prev = prev_in.z;

                n_cur = nerf_place3(n_prev, ca_prev, c_prev, bg0.x, bg0.y, sin(psi_prev), cos(psi_prev));
                ca_cur = nerf_place3(ca_prev, c_prev, n_cur, bg0.z, bg0.w, sin(omg_prev), cos(omg_prev));
                if (res_class != 2u) {
                    let h_i = nerf_place3(ca_prev, c_prev, n_cur, BOND_N_H_F32, ANG_C_N_H_F32, -sin(omg_prev), -cos(omg_prev));
                    scratch[a_off + 1u] = vec4<f32>(h_i, 1.0);
                } else {
                    scratch[a_off + 1u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                }
                c_cur = nerf_place3(c_prev, n_cur, ca_cur, bg1.x, bg1.y, sin(phi_i), cos(phi_i));
            }

            scratch[a_off + 0u] = vec4<f32>(n_cur, 1.0);
            scratch[a_off + 2u] = vec4<f32>(ca_cur, 1.0);
            scratch[a_off + 4u] = vec4<f32>(c_cur, 1.0);

            let o_cur = nerf_place3(n_cur, ca_cur, c_cur, bg1.z, bg1.w, -sin(psi_i), -cos(psi_i));
            scratch[a_off + 5u] = vec4<f32>(o_cur, 1.0);

            let b1 = safe_unit3(n_cur - ca_cur);
            let b2 = safe_unit3(c_cur - ca_cur);
            let u_bis = safe_unit3(b1 + b2);
            let nrm = safe_unit3(cross(b1, b2));
            let dir_cb = safe_unit3(u_bis * (-0.5810) + nrm * 0.8139);
            let dir_ha = safe_unit3(u_bis * (-0.5810) + nrm * (-0.8139));
            if (res_class == 1u) {
                scratch[a_off + 3u] = vec4<f32>(ca_cur + dir_ha * 1.090, 1.0);
                scratch[a_off + 6u] = vec4<f32>(ca_cur + dir_cb * 1.090, 1.0);
            } else {
                scratch[a_off + 3u] = vec4<f32>(ca_cur + dir_ha * 1.090, 1.0);
                scratch[a_off + 6u] = vec4<f32>(ca_cur + dir_cb * 1.526, 1.0);
            }

            i = i + 1u;
        }
    }

    var e_rdc: f32 = 0.0;
    var c_rdc: f32 = 0.0;
    var e_pcs: f32 = 0.0;
    var c_pcs: f32 = 0.0;
    var e_pre: f32 = 0.0;
    var e_noe: f32 = 0.0;
    var e_vdw: f32 = 0.0;
    var e_rama: f32 = 0.0;

    // 2. Working RDC potential & forces
    if (params.w_rdc > 0.0 && params.n_rdc_terms > 0u) {
        let w_rdc = params.w_rdc;
        var rdc_blk_tq_s = array<vec3<f32>, 16>();
        var rdc_blk_tq_c = array<vec3<f32>, 16>();
        for (var k: u32 = 0u; k < params.n_rdc_terms; k = k + 1u) {
            let ib = ioff_rdc + 5u * k;
            let med_idx = static_ints[ib + 0u];
            let atom_a = static_ints[ib + 1u];
            let atom_b = static_ints[ib + 2u];
            let seg_a = static_ints[ib + 3u];
            let seg_b = static_ints[ib + 4u];

            let a4 = scratch[atom_base + atom_a];
            let b4 = scratch[atom_base + atom_b];
            if (a4.w < 0.5 || b4.w < 0.5) {
                continue;
            }
            let dv = b4.xyz - a4.xyz;
            let d2 = dot(dv, dv);
            if (d2 < 1e-12) {
                continue;
            }
            let inv_r = exact_rsqrt(d2);
            let u = dv * inv_r;

            let rf = static_floats[off_rdc_f + k];
            let d_obs = rf.x;
            let d_max = rf.y;
            let inv_s2 = rf.z;

            let sb = in_saupe_base + 3u * med_idx;
            let s0 = replica_in[sb + 0u].xyz;
            let s1 = replica_in[sb + 1u].xyz;
            let s2 = replica_in[sb + 2u].xyz;
            let su = mat3_rows_mul_vec(s0, s1, s2, u);
            let ut_s_u = dot(u, su);
            let d_calc = d_max * ut_s_u;
            let resid = d_calc - d_obs;
            let y_rdc = (0.5 * w_rdc * resid * resid * inv_s2) - c_rdc;
            let t_rdc = e_rdc + y_rdc;
            c_rdc = (t_rdc - e_rdc) - y_rdc;
            e_rdc = t_rdc;

            let pref = w_rdc * resid * inv_s2 * (2.0 * d_max * inv_r);
            let grad_pb = (su - u * ut_s_u) * pref;
            let res_a = atom_a / 7u;
            let res_b = atom_b / 7u;
            let blk_a = static_ints[ioff_res + 4u * res_a + 1u];
            let blk_b = static_ints[ioff_res + 4u * res_b + 1u];
            if (blk_a != NO_BLOCK_SENTINEL && blk_a == blk_b) {
                let b_start = static_ints[ioff_blocks + 2u * blk_a + 0u];
                let seg_blk = 2u * b_start;
                let pure_tq = cross(dv, grad_pb);
                if (blk_a < 16u) {
                    let y_tq = pure_tq - rdc_blk_tq_c[blk_a];
                    let t_tq = rdc_blk_tq_s[blk_a] + y_tq;
                    rdc_blk_tq_c[blk_a] = (t_tq - rdc_blk_tq_s[blk_a]) - y_tq;
                    rdc_blk_tq_s[blk_a] = t_tq;
                } else {
                    scratch[torque_base + seg_blk] = vec4<f32>(scratch[torque_base + seg_blk].xyz + pure_tq, 0.0);
                }
            } else if (seg_a == seg_b) {
                let pure_tq = cross(dv, grad_pb);
                scratch[torque_base + seg_a] = vec4<f32>(scratch[torque_base + seg_a].xyz + pure_tq, 0.0);
            } else {
                add_seg_grad_dock(atom_base, force_base, torque_base, seg_b, b4.xyz, grad_pb);
                add_seg_grad_dock(atom_base, force_base, torque_base, seg_a, a4.xyz, -grad_pb);
            }
        }
        let flush_n = min(n_blocks, 16u);
        for (var bi: u32 = 0u; bi < flush_n; bi = bi + 1u) {
            let b_start = static_ints[ioff_blocks + 2u * bi + 0u];
            let seg_blk = 2u * b_start;
            scratch[torque_base + seg_blk] = vec4<f32>(scratch[torque_base + seg_blk].xyz + rdc_blk_tq_s[bi], 0.0);
        }
    }

    // 3. Cross-domain PCS potential, forces & host-domain rotational torque
    if (params.w_pcs > 0.0 && params.n_pcs_terms > 0u) {
        let w_pcs = params.w_pcs;
        for (var k: u32 = 0u; k < params.n_pcs_terms; k = k + 1u) {
            let ib = ioff_pcs + 3u * k;
            let tag_idx = static_ints[ib + 0u];
            let atom_slot = static_ints[ib + 1u];
            let seg_idx = static_ints[ib + 2u];

            let a4 = scratch[atom_base + atom_slot];
            if (a4.w < 0.5) {
                continue;
            }

            let host_blk = static_ints[ioff_pcs_tags + 2u * tag_idx + 0u];
            let host_seg = static_ints[ioff_pcs_tags + 2u * tag_idx + 1u];
            let b_off = blk_base + 4u * host_blk;
            let orig_n = scratch[b_off + 0u].xyz;
            let rb0 = scratch[b_off + 1u].xyz;
            let rb1 = scratch[b_off + 2u].xyz;
            let rb2 = scratch[b_off + 3u].xyz;

            let tf = off_pcs_tags_f + 4u * tag_idx;
            let m_loc = static_floats[tf + 0u].xyz;
            let dc0 = static_floats[tf + 1u].xyz;
            let dc1 = static_floats[tf + 2u].xyz;
            let dc2 = static_floats[tf + 3u].xyz;

            let m_pos = orig_n + mat3_rows_mul_vec(rb0, rb1, rb2, m_loc);
            let v = a4.xyz - m_pos;
            let r2 = dot(v, v) + 0.25;
            let inv_r = exact_rsqrt(r2);
            let inv_r2 = inv_r * inv_r;
            let inv_r5 = inv_r2 * inv_r2 * inv_r;
            let inv_r7 = inv_r5 * inv_r2;

            // mv = (R_b * dchi_loc * R_b^T) * v
            let rbt_v = vec3<f32>(
                rb0.x * v.x + rb1.x * v.y + rb2.x * v.z,
                rb0.y * v.x + rb1.y * v.y + rb2.y * v.z,
                rb0.z * v.x + rb1.z * v.y + rb2.z * v.z,
            );
            let dc_v = mat3_rows_mul_vec(dc0, dc1, dc2, rbt_v);
            let mv = mat3_rows_mul_vec(rb0, rb1, rb2, dc_v);
            let v_mv = dot(v, mv);

            let pf = static_floats[off_pcs_f + k];
            let pcs_obs = pf.x;
            let inv_s2 = pf.y;
            let pcs_calc = PCS_CART_PREFACTOR_F32 * v_mv * inv_r5;
            let resid = pcs_calc - pcs_obs;
            let y_pcs = (0.5 * w_pcs * resid * resid * inv_s2) - c_pcs;
            let t_pcs = e_pcs + y_pcs;
            c_pcs = (t_pcs - e_pcs) - y_pcs;
            e_pcs = t_pcs;

            let d_e_dpcs = w_pcs * resid * inv_s2;
            let g_v = (mv * (2.0 * inv_r5) - v * (5.0 * v_mv * inv_r7)) * (d_e_dpcs * PCS_CART_PREFACTOR_F32);
            add_seg_grad_dock(atom_base, force_base, torque_base, seg_idx, a4.xyz, g_v);
            add_seg_grad_dock(atom_base, force_base, torque_base, host_seg, m_pos, -g_v);

            let rot_tq = cross(mv, v) * (d_e_dpcs * (2.0 * PCS_CART_PREFACTOR_F32 * inv_r5));
            scratch[torque_base + host_seg] = vec4<f32>(scratch[torque_base + host_seg].xyz + rot_tq, 0.0);
        }
    }

    // 4. Cross-domain Solomon r^-6 PRE potential & forces
    if (params.w_pre > 0.0 && params.n_pre_terms > 0u) {
        let w_pre = params.w_pre;
        for (var k: u32 = 0u; k < params.n_pre_terms; k = k + 1u) {
            let ib = ioff_pre + 4u * k;
            let site_idx = static_ints[ib + 0u];
            let atom_slot = static_ints[ib + 1u];
            let seg_idx = static_ints[ib + 2u];
            let is_bounds = static_ints[ib + 3u];

            let a4 = scratch[atom_base + atom_slot];
            if (a4.w < 0.5) {
                continue;
            }

            let host_blk = static_ints[ioff_pre_sites + 2u * site_idx + 0u];
            let host_seg = static_ints[ioff_pre_sites + 2u * site_idx + 1u];
            let b_off = blk_base + 4u * host_blk;
            let orig_n = scratch[b_off + 0u].xyz;
            let rb0 = scratch[b_off + 1u].xyz;
            let rb1 = scratch[b_off + 2u].xyz;
            let rb2 = scratch[b_off + 3u].xyz;
            let s_loc = static_floats[off_pre_sites_f + site_idx].xyz;
            let s_pos = orig_n + mat3_rows_mul_vec(rb0, rb1, rb2, s_loc);

            let pf0 = static_floats[off_pre_f + 2u * k + 0u];
            let pf1 = static_floats[off_pre_f + 2u * k + 1u];
            let dv = a4.xyz - s_pos;
            let r2_raw = dot(dv, dv);

            if (is_bounds != 0u) {
                let d = sqrt(r2_raw + 1e-4);
                let inv_s2 = pf0.y;
                let lo_a = pf1.x;
                let hi_a = pf1.y;
                if (d < lo_a) {
                    let u = lo_a - d;
                    e_pre = e_pre + 0.5 * w_pre * u * u * inv_s2;
                    let g = dv * (-w_pre * u * inv_s2 / d);
                    add_seg_grad_dock(atom_base, force_base, torque_base, seg_idx, a4.xyz, g);
                    add_seg_grad_dock(atom_base, force_base, torque_base, host_seg, s_pos, -g);
                } else if (d > hi_a) {
                    let u = d - hi_a;
                    e_pre = e_pre + 0.5 * w_pre * u * u * inv_s2;
                    let g = dv * (w_pre * u * inv_s2 / d);
                    add_seg_grad_dock(atom_base, force_base, torque_base, seg_idx, a4.xyz, g);
                    add_seg_grad_dock(atom_base, force_base, torque_base, host_seg, s_pos, -g);
                }
            } else {
                let g2_obs = pf0.x;
                let inv_s2 = pf0.y;
                let k_pre = pf0.z;
                let lo_a = pf1.x;
                let hi_a = pf1.y;

                let r2_eff = r2_raw + 4.0;
                let inv_r2 = 1.0 / r2_eff;
                let inv_r6 = inv_r2 * inv_r2 * inv_r2;
                let g2_calc = k_pre * inv_r6;
                let inv_sig = sqrt(inv_s2);
                let z = (g2_calc - g2_obs) * inv_sig;
                let kappa: f32 = 8.0;
                var huber_e: f32 = 0.0;
                var d_huber_dz: f32 = 0.0;
                if (abs(z) <= kappa) {
                    huber_e = 0.5 * z * z;
                    d_huber_dz = z;
                } else {
                    huber_e = kappa * (abs(z) - 0.5 * kappa);
                    d_huber_dz = select(-kappa, kappa, z >= 0.0);
                }
                e_pre = e_pre + w_pre * huber_e;
                let d_e_dg2 = w_pre * d_huber_dz * inv_sig;
                var g_pa = dv * (d_e_dg2 * (-6.0 * k_pre * inv_r6 * inv_r2));

                let d = sqrt(r2_raw + 1e-4);
                if (d > hi_a) {
                    let u = d - hi_a;
                    e_pre = e_pre + 0.5 * w_pre * u * u;
                    g_pa = g_pa + dv * (w_pre * u / d);
                } else if (d < lo_a) {
                    let u = lo_a - d;
                    e_pre = e_pre + 0.5 * w_pre * u * u;
                    g_pa = g_pa + dv * (-w_pre * u / d);
                }
                add_seg_grad_dock(atom_base, force_base, torque_base, seg_idx, a4.xyz, g_pa);
                add_seg_grad_dock(atom_base, force_base, torque_base, host_seg, s_pos, -g_pa);
            }
        }
    }

    // 5. Inter-domain / linker NOE distance restraints
    if (params.w_noe > 0.0 && params.n_noe_terms > 0u) {
        let w_noe = params.w_noe;
        let soft = max(params.noe_soft_window, 0.5);
        for (var k: u32 = 0u; k < params.n_noe_terms; k = k + 1u) {
            let ib = ioff_noe + 4u * k;
            let atom_a = static_ints[ib + 0u];
            let atom_b = static_ints[ib + 1u];
            let seg_a = static_ints[ib + 2u];
            let seg_b = static_ints[ib + 3u];
            let nf = static_floats[off_noe_f + k];
            let lo_a = nf.x;
            let hi_a = nf.y;

            let a4 = scratch[atom_base + atom_a];
            let b4 = scratch[atom_base + atom_b];
            if (a4.w < 0.5 || b4.w < 0.5) {
                continue;
            }
            let diff = a4.xyz - b4.xyz;
            let d = sqrt(dot(diff, diff) + 1e-6);
            var v_e: f32 = 0.0;
            var dv_dd: f32 = 0.0;
            if (d < lo_a) {
                let u = lo_a - d;
                v_e = w_noe * u * u;
                dv_dd = -2.0 * w_noe * u;
            } else if (d > hi_a) {
                let u = d - hi_a;
                if (u <= soft) {
                    v_e = w_noe * u * u;
                    dv_dd = 2.0 * w_noe * u;
                } else {
                    v_e = w_noe * (soft * soft + 2.0 * soft * (u - soft));
                    dv_dd = 2.0 * w_noe * soft;
                }
            }
            e_noe = e_noe + v_e;
            if (dv_dd != 0.0) {
                let g = diff * (dv_dd / d);
                add_seg_grad_dock(atom_base, force_base, torque_base, seg_a, a4.xyz, g);
                add_seg_grad_dock(atom_base, force_base, torque_base, seg_b, b4.xyz, -g);
            }
        }
    }

    // 6. Quartic soft-core excluded volume (exact match to CPU `accumulate_grid_hashed_vdw_cpu`)
    if (params.w_vdw > 0.0 && params.vdw_scale > 0.0) {
        let w_vdw = params.w_vdw;
        let s_vdw = params.vdw_scale;
        let s2 = s_vdw * s_vdw;
        let ca_cull_r = 8.6 * max(s_vdw, 0.75);
        let ca_cull_r2 = ca_cull_r * ca_cull_r;

        for (var i: u32 = 0u; i < l; i = i + 1u) {
            let im = ioff_res + 4u * i;
            let rc_i = static_ints[im + 0u];
            let blk_i = static_ints[im + 1u];
            let ch_i = static_ints[im + 2u];
            let ai = atom_base + 7u * i;
            let ni_p = scratch[ai + 0u].xyz;
            let cai_p = scratch[ai + 2u].xyz;
            let ci_p = scratch[ai + 4u].xyz;
            let oi_p = scratch[ai + 5u].xyz;
            let cbi_4 = scratch[ai + 6u];
            let has_cbi = (rc_i != 1u) && (cbi_4.w > 0.5);

            // Local `(i, i+1)` Ramachandran steric guard on non-co-block intra-chain pairs
            if (i + 1u < l) {
                let jm1 = ioff_res + 4u * (i + 1u);
                let rc_j1 = static_ints[jm1 + 0u];
                let blk_j1 = static_ints[jm1 + 1u];
                let ch_j1 = static_ints[jm1 + 2u];
                let same_blk_1 = (blk_i != NO_BLOCK_SENTINEL) && (blk_i == blk_j1);
                if (ch_i == ch_j1 && !same_blk_1) {
                    let aj1 = atom_base + 7u * (i + 1u);
                    let cj1_p = scratch[aj1 + 4u].xyz;
                    let oj1_p = scratch[aj1 + 5u].xyz;

                    let r2_oc = s2 * vdw_min_r2_base_f32(2u, 1u);
                    let df_oc = oi_p - cj1_p;
                    let d2_oc = dot(df_oc, df_oc);
                    if (d2_oc < r2_oc) {
                        let dl = r2_oc - d2_oc;
                        e_vdw = e_vdw + w_vdw * dl * dl;
                        let g = df_oc * (-4.0 * w_vdw * dl);
                        add_seg_grad_dock(atom_base, force_base, torque_base, 2u * i + 2u, oi_p, g);
                        add_seg_grad_dock(atom_base, force_base, torque_base, 2u * (i + 1u) + 1u, cj1_p, -g);
                    }

                    let r2_oo = s2 * vdw_min_r2_base_f32(2u, 2u);
                    let df_oo = oi_p - oj1_p;
                    let d2_oo = dot(df_oo, df_oo);
                    if (d2_oo < r2_oo) {
                        let dl = r2_oo - d2_oo;
                        e_vdw = e_vdw + w_vdw * dl * dl;
                        let g = df_oo * (-4.0 * w_vdw * dl);
                        add_seg_grad_dock(atom_base, force_base, torque_base, 2u * i + 2u, oi_p, g);
                        add_seg_grad_dock(atom_base, force_base, torque_base, 2u * (i + 1u) + 2u, oj1_p, -g);
                    }

                    let cbj1_4 = scratch[aj1 + 6u];
                    if (rc_j1 != 1u && cbj1_4.w > 0.5) {
                        let r2_ocb = s2 * (2.90 * 2.90);
                        let df_ocb = oi_p - cbj1_4.xyz;
                        let d2_ocb = dot(df_ocb, df_ocb);
                        if (d2_ocb < r2_ocb) {
                            let dl = r2_ocb - d2_ocb;
                            e_vdw = e_vdw + w_vdw * dl * dl;
                            let g = df_ocb * (-4.0 * w_vdw * dl);
                            add_seg_grad_dock(atom_base, force_base, torque_base, 2u * i + 2u, oi_p, g);
                            add_seg_grad_dock(atom_base, force_base, torque_base, 2u * (i + 1u) + 1u, cbj1_4.xyz, -g);
                        }
                    }
                }
            }

            // Non-local residue pairs `j > i`
            var p_i = array<vec3<f32>, 5>(ni_p, cai_p, ci_p, oi_p, cbi_4.xyz);
            var k_i = array<u32, 5>(0u, 1u, 1u, 2u, 3u);
            var s_i = array<u32, 5>(2u * i, 2u * i, 2u * i + 1u, 2u * i + 2u, 2u * i + 1u);
            let cnt_i = select(4u, 5u, has_cbi);

            for (var j: u32 = i + 1u; j < l; j = j + 1u) {
                let jm = ioff_res + 4u * j;
                let rc_j = static_ints[jm + 0u];
                let blk_j = static_ints[jm + 1u];
                let ch_j = static_ints[jm + 2u];
                if (blk_i != NO_BLOCK_SENTINEL && blk_i == blk_j) {
                    continue;
                }
                let same_chain = (ch_i == ch_j);
                if (same_chain && j < i + 2u) {
                    continue;
                }
                let aj = atom_base + 7u * j;
                let caj_p = scratch[aj + 2u].xyz;
                let d_caca = caj_p - cai_p;
                if (dot(d_caca, d_caca) > ca_cull_r2) {
                    continue;
                }

                let nj_p = scratch[aj + 0u].xyz;
                let cj_p = scratch[aj + 4u].xyz;
                let oj_p = scratch[aj + 5u].xyz;
                let cbj_4 = scratch[aj + 6u];
                let has_cbj = (rc_j != 1u) && (cbj_4.w > 0.5);

                var p_j = array<vec3<f32>, 5>(nj_p, caj_p, cj_p, oj_p, cbj_4.xyz);
                var k_j = array<u32, 5>(0u, 1u, 1u, 2u, 3u);
                var s_j = array<u32, 5>(2u * j, 2u * j, 2u * j + 1u, 2u * j + 2u, 2u * j + 1u);
                let cnt_j = select(4u, 5u, has_cbj);

                for (var ia: u32 = 0u; ia < cnt_i; ia = ia + 1u) {
                    for (var ja: u32 = 0u; ja < cnt_j; ja = ja + 1u) {
                        if (same_chain && j == i + 2u && ia == 2u && ja == 0u) {
                            continue;
                        }
                        let r2_min = s2 * vdw_min_r2_base_f32(k_i[ia], k_j[ja]);
                        let diff = p_i[ia] - p_j[ja];
                        let d2 = dot(diff, diff);
                        if (d2 < r2_min) {
                            let dl = r2_min - d2;
                            e_vdw = e_vdw + w_vdw * dl * dl;
                            let g = diff * (-4.0 * w_vdw * dl);
                            add_seg_grad_dock(atom_base, force_base, torque_base, s_i[ia], p_i[ia], g);
                            add_seg_grad_dock(atom_base, force_base, torque_base, s_j[ja], p_j[ja], -g);
                        }
                    }
                }
            }
        }
    }

    // 7. Backward suffix-sum torque propagation per chain + Lovell Ramachandran prior
    let w_rama = params.w_rama;
    var c_rev = n_chains;
    while (c_rev > 0u) {
        c_rev = c_rev - 1u;
        let cs = static_ints[ioff_chains + 2u * c_rev + 0u];
        let ce = static_ints[ioff_chains + 2u * c_rev + 1u];
        if (cs >= ce) {
            grads_out[grad_base + l + 2u * c_rev + 0u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            grads_out[grad_base + l + 2u * c_rev + 1u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            continue;
        }

        var suf_f = vec3<f32>(0.0, 0.0, 0.0);
        var suf_t = vec3<f32>(0.0, 0.0, 0.0);
        var cur_q = scratch[atom_base + 7u * (ce - 1u) + 4u].xyz;

        var i_rev = ce;
        while (i_rev > cs) {
            i_rev = i_rev - 1u;
            let r_off = atom_base + 7u * i_rev;
            let ni = scratch[r_off + 0u].xyz;
            let cai = scratch[r_off + 2u].xyz;
            let ci = scratch[r_off + 4u].xyz;

            let rm = ioff_res + 4u * i_rev;
            let rc = static_ints[rm + 0u];
            let r_flags = static_ints[rm + 3u];
            let phi_act = (r_flags & 1u) != 0u;
            let psi_act = (r_flags & 2u) != 0u;
            let rama_act = (r_flags & 4u) != 0u;

            suf_t = suf_t + cross(cur_q - ci, suf_f);
            cur_q = ci;
            let seg_psi = 2u * i_rev + 2u;
            suf_f = suf_f + scratch[force_base + seg_psi].xyz;
            suf_t = suf_t + scratch[torque_base + seg_psi].xyz;
            var g_psi: f32 = 0.0;
            if (psi_act) {
                g_psi = dot(safe_unit3(ci - cai), suf_t);
            }

            suf_t = suf_t + cross(cur_q - cai, suf_f);
            cur_q = cai;
            let seg_phi = 2u * i_rev + 1u;
            suf_f = suf_f + scratch[force_base + seg_phi].xyz;
            suf_t = suf_t + scratch[torque_base + seg_phi].xyz;
            var g_phi: f32 = 0.0;
            if (phi_act) {
                g_phi = dot(safe_unit3(cai - ni), suf_t);
            }

            if (w_rama > 0.0 && rama_act) {
                let cur_in = replica_in[in_base + i_rev];
                let cur_ref = static_floats[off_ref_tors + i_rev];
                let phi_v = select(cur_ref.x, cur_in.x, phi_act);
                let psi_v = select(cur_ref.y, cur_in.y, psi_act);
                let rv = rama_energy_and_grad_f32(rc, phi_v, psi_v);
                e_rama = e_rama + w_rama * rv.x;
                if (phi_act) {
                    g_phi = g_phi + w_rama * rv.y;
                }
                if (psi_act) {
                    g_psi = g_psi + w_rama * rv.z;
                }
            }

            grads_out[grad_base + i_rev] = vec4<f32>(g_phi, g_psi, 0.0, 0.0);

            if (i_rev == cs) {
                let seg_root = 2u * cs;
                let piv_root = seg_pivot_pos_dock(atom_base, seg_root);
                suf_t = suf_t + cross(cur_q - piv_root, suf_f);
                cur_q = piv_root;
                suf_f = suf_f + scratch[force_base + seg_root].xyz;
                suf_t = suf_t + scratch[torque_base + seg_root].xyz;
            }
        }

        if (c_rev >= 1u) {
            let c_af = static_floats[off_chain_centroids + c_rev].xyz;
            let tr_c = replica_in[in_chains_base + 2u * c_rev + 1u].xyz;
            let c_world = c_af + tr_c;
            let tq_c = suf_t + cross(cur_q - c_world, suf_f);
            grads_out[grad_base + l + 2u * c_rev + 0u] = vec4<f32>(tq_c, 0.0);
            grads_out[grad_base + l + 2u * c_rev + 1u] = vec4<f32>(suf_f, 0.0);
        } else {
            grads_out[grad_base + l + 0u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            grads_out[grad_base + l + 1u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        }
    }

    let total_e = e_rdc + e_pcs + e_pre + e_noe + e_vdw + e_rama;
    energies_out[2u * conf_idx + 0u] = vec4<f32>(total_e, e_rdc, e_pcs, e_pre);
    energies_out[2u * conf_idx + 1u] = vec4<f32>(e_noe, e_vdw, e_rama, 0.0);
}
