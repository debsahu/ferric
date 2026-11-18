// Clean-room WGSL compute shaders for backbone NMR unary typing U(s, k)
// and pairwise sequential connectivity P(s, t) matrices.
// Provenance: Implements docs/specs/03-assignment.md section 3.

struct ScoreParams {
    s_len: u32,
    k_len: u32,
    max_cands: u32,
    use_hn: u32,
    sigma_ca: f32,
    sigma_cb: f32,
    sigma_c: f32,
    range_ppm: f32,
    pair_floor: f32,
    support_bonus: f32,
    llr_floor: f32,
    gly_cb: f32,
    assign_cost: f32,
    forbid_zero_carbons: u32,
    has_excess_roots: u32,
    multi_slot_dataset: u32,
};

struct PriorEntry {
    mu: f32,
    sd: f32,
    present: u32,
    _pad: u32,
};

struct CandEntry {
    val: f32,
    support: u32,
};

struct SeqEntry {
    res_idx: u32,
    d_n: f32,
    d_ca: f32,
    d_cb: f32,
    d_c: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: ScoreParams;
// 20 residues x 5 atoms (H=0, N=1, CA=2, CB=3, C=4)
@group(0) @binding(1) var<storage, read> priors: array<PriorEntry>;
// Sequence residue entry per k (res_idx 0..19, 14 = PRO, 255 = UNK, plus covalent offsets)
@group(0) @binding(2) var<storage, read> seq_res: array<SeqEntry>;
// Root (H, N) per spin system: vec2<f32>(H, N)
@group(0) @binding(3) var<storage, read> roots_hn: array<vec2<f32>>;
// Candidates indexed [((s * 3 + atom_c) * 2 + off_idx) * max_cands + c]
// atom_c: 0=CA, 1=CB, 2=C; off_idx: 0=(i), 1=(i-1)
@group(0) @binding(4) var<storage, read> cands: array<CandEntry>;
// Counts indexed [(s * 3 + atom_c) * 2 + off_idx]
@group(0) @binding(5) var<storage, read> cand_counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_u: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_p: array<f32>;

const TAU: f32 = 6.283185307179586;
const FORBID: f32 = -1e9;
const GLY_IDX: u32 = 7u;
const PRO_IDX: u32 = 14u;
const NATURAL_PRO_FRACTION: f32 = 0.03628389; // 91.0 / 2508.0
const COVALENT_COIL_WEIGHT: f32 = 0.85;
const LN_COVALENT_COIL_WEIGHT: f32 = -0.16251893; // ln(0.85)
const LN_COVALENT_STRUCT_WEIGHT: f32 = -1.8971200; // ln(0.15)

fn logpdf(v: f32, mu: f32, sd: f32) -> f32 {
    let z = (v - mu) / sd;
    return -0.5 * z * z - log(sd * sqrt(TAU));
}

fn llr(res_idx: u32, atom_idx: u32, v: f32, d_mu: f32) -> f32 {
    if (res_idx >= 20u) {
        return 0.0;
    }
    let pe = priors[res_idx * 5u + atom_idx];
    if (pe.present == 0u) {
        return params.gly_cb;
    }
    var lp_res: f32;
    if (abs(d_mu) < 1e-6) {
        lp_res = logpdf(v, pe.mu, pe.sd);
    } else if (abs(d_mu) < 0.09) {
        lp_res = logpdf(v, pe.mu + d_mu, pe.sd);
    } else {
        let raw = d_mu / (COVALENT_COIL_WEIGHT * (1.0 - NATURAL_PRO_FRACTION));
        let d_non = -COVALENT_COIL_WEIGHT * NATURAL_PRO_FRACTION * raw;
        let a = LN_COVALENT_COIL_WEIGHT + logpdf(v, pe.mu + d_non + raw, pe.sd);
        let b = LN_COVALENT_STRUCT_WEIGHT + logpdf(v, pe.mu + d_non, pe.sd);
        let m = max(a, b);
        lp_res = m + log(exp(a - m) + exp(b - m));
    }

    var max_lp: f32 = -1e30;
    var count: u32 = 0u;
    for (var r: u32 = 0u; r < 20u; r = r + 1u) {
        let p_r = priors[r * 5u + atom_idx];
        if (p_r.present != 0u) {
            let lp = logpdf(v, p_r.mu, p_r.sd);
            if (lp > max_lp) {
                max_lp = lp;
            }
            count = count + 1u;
        }
    }
    var sum_exp: f32 = 0.0;
    for (var r: u32 = 0u; r < 20u; r = r + 1u) {
        let p_r = priors[r * 5u + atom_idx];
        if (p_r.present != 0u) {
            sum_exp = sum_exp + exp(logpdf(v, p_r.mu, p_r.sd) - max_lp);
        }
    }
    let bg = max_lp + log(sum_exp / f32(count));
    return max(lp_res - bg, params.llr_floor);
}

@compute @workgroup_size(16, 16, 1)
fn compute_unary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    let k = gid.y;
    if (s >= params.s_len || k >= params.k_len) {
        return;
    }
    let out_idx = s * params.k_len + k;
    if (k == 0u) {
        out_u[out_idx] = FORBID;
        return;
    }
    let cur = seq_res[k];
    let res = cur.res_idx;
    if (res == PRO_IDX) {
        out_u[out_idx] = FORBID;
        return;
    }
    let prev_e = seq_res[k - 1u];
    let prev = prev_e.res_idx;

    var sc: f32 = 0.0;
    var total_pos_cands: u32 = 0u;
    var has_obs_ca: u32 = 0u;
    var obs_slots: u32 = 0u;
    var max_sup: u32 = 0u;
    for (var atom_c: u32 = 0u; atom_c < 3u; atom_c = atom_c + 1u) {
        let atom_idx = atom_c + 2u; // 2=CA, 3=CB, 4=C
        for (var off_idx: u32 = 0u; off_idx < 2u; off_idx = off_idx + 1u) {
            let slot = (s * 3u + atom_c) * 2u + off_idx;
            let n_raw = cand_counts[slot];
            if (n_raw == 0u) {
                continue;
            }
            let base = slot * params.max_cands;
            let target_idx = select(k - 1u, k, off_idx == 0u);
            let te = seq_res[target_idx];
            let d_mu = select(select(te.d_c, te.d_cb, atom_c == 1u), te.d_ca, atom_c == 0u);
            var n_pos: u32 = 0u;
            var best: f32 = -1e30;
            for (var c: u32 = 0u; c < n_raw; c = c + 1u) {
                let ce = cands[base + c];
                if (ce.support > 0u) {
                    n_pos = n_pos + 1u;
                    if (ce.support > max_sup) {
                        max_sup = ce.support;
                    }
                    let term = llr(te.res_idx, atom_idx, ce.val, d_mu)
                        + params.support_bonus * (f32(ce.support) - 1.0);
                    if (term > best) {
                        best = term;
                    }
                }
            }
            if (n_pos > 0u) {
                total_pos_cands = total_pos_cands + n_pos;
                obs_slots = obs_slots + 1u;
                if (atom_c == 0u) {
                    has_obs_ca = 1u;
                }
                sc = sc + best - log(f32(n_pos));
            }
        }
    }

    if (params.forbid_zero_carbons != 0u && total_pos_cands == 0u) {
        out_u[out_idx] = FORBID;
        return;
    }
    if (params.has_excess_roots != 0u && has_obs_ca == 0u && res != GLY_IDX && prev != PRO_IDX) {
        out_u[out_idx] = FORBID;
        return;
    }

    if (params.use_hn != 0u) {
        let hn = roots_hn[s];
        sc = sc + llr(res, 1u, hn.y, cur.d_n) + llr(res, 0u, hn.x, 0.0);
    }
    if (params.has_excess_roots != 0u
        && params.multi_slot_dataset != 0u
        && (has_obs_ca == 0u || (obs_slots <= 1u && max_sup <= 1u)))
    {
        sc = sc - 4.5;
    }
    sc = sc - params.assign_cost;
    out_u[out_idx] = sc;
}

@compute @workgroup_size(16, 16, 1)
fn compute_pair(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    let t = gid.y;
    if (s >= params.s_len || t >= params.s_len) {
        return;
    }
    let out_idx = s * params.s_len + t;
    if (s == t) {
        out_p[out_idx] = FORBID;
        return;
    }

    let sigmas = array<f32, 3>(params.sigma_ca, params.sigma_cb, params.sigma_c);
    let log_range = log(params.range_ppm);
    var total_p: f32 = 0.0;

    for (var atom_c: u32 = 0u; atom_c < 3u; atom_c = atom_c + 1u) {
        let slot_a = (s * 3u + atom_c) * 2u + 0u;
        let slot_b = (t * 3u + atom_c) * 2u + 1u;
        let na = cand_counts[slot_a];
        let nb = cand_counts[slot_b];
        if (na == 0u || nb == 0u) {
            continue;
        }
        let sig = sigmas[atom_c];
        let const_term = -log(sig * sqrt(TAU)) + log_range;
        let base_a = slot_a * params.max_cands;
        let base_b = slot_b * params.max_cands;

        var best: f32 = -1e30;
        for (var ia: u32 = 0u; ia < na; ia = ia + 1u) {
            let va = cands[base_a + ia].val;
            for (var ib: u32 = 0u; ib < nb; ib = ib + 1u) {
                let vb = cands[base_b + ib].val;
                let d = (va - vb) / sig;
                let term = max(-0.5 * d * d + const_term, params.pair_floor);
                if (term > best) {
                    best = term;
                }
            }
        }
        let pen = log(f32(na * nb));
        total_p = total_p + (best - pen);
    }
    out_p[out_idx] = total_p;
}

// ============================================================================
// Opt-in AF Per-Nucleus Replacement Scoring Kernel (`T21.4`, @group(1) isolated)
// ============================================================================

struct AfScoreParams {
    s_len: u32,
    k_len: u32,
    max_cands: u32,
    use_hn: u32,
    support_bonus: f32,
    llr_floor: f32,
    gly_cb: f32,
    assign_cost: f32,
    forbid_zero_carbons: u32,
    has_excess_roots: u32,
    multi_slot_dataset: u32,
    af_clamp_total: f32,
};

struct AfSeqPriorEntry {
    mu_hi: f32,
    mu_lo: f32,
    inv_sd_hi: f32,
    inv_sd_lo: f32,
    ln_sd_hi: f32,
    ln_sd_lo: f32,
    present: u32,
    _pad0: u32,
};

struct AfPriorEntry {
    mu_af_hi: f32,
    mu_af_lo: f32,
    inv_sigma_af_hi: f32,
    inv_sigma_af_lo: f32,
    c_af_hi: f32,
    c_af_lo: f32,
    omega: f32,
    present: u32,
};

struct AfCandEntry {
    val_hi: f32,
    val_lo: f32,
    support: u32,
    _pad: u32,
};

struct AfSeqEntry {
    res_idx: u32,
    _pad0: u32,
    d_n_hi: f32,
    d_n_lo: f32,
    d_ca_hi: f32,
    d_ca_lo: f32,
    d_cb_hi: f32,
    d_cb_lo: f32,
    d_c_hi: f32,
    d_c_lo: f32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(0) var<uniform> af_params: AfScoreParams;
@group(1) @binding(1) var<storage, read> af_seq_priors: array<AfSeqPriorEntry>;
@group(1) @binding(2) var<storage, read> af_seq_res: array<AfSeqEntry>;
@group(1) @binding(3) var<storage, read> af_roots_hn: array<vec4<f32>>;
@group(1) @binding(4) var<storage, read> af_cands: array<AfCandEntry>;
@group(1) @binding(5) var<storage, read> af_cand_counts: array<u32>;
@group(1) @binding(6) var<storage, read_write> af_out_u: array<f32>;
@group(1) @binding(7) var<storage, read> af_priors: array<AfPriorEntry>;

const LN2_HI: f32 = 0.693145751953125;
const LN2_LO: f32 = 1.428606765330187e-6;
const INV_LN2: f32 = 1.4426950408889634;

fn ln_accurate(x: f32) -> f32 {
    let fr = frexp(x);
    var m = fr.fract;
    var e = f32(fr.exp);
    if (m < 0.70710678) {
        m = m * 2.0;
        e = e - 1.0;
    }
    let u = (m - 1.0) / (m + 1.0);
    let u2 = u * u;
    let tail = u2 * fma(u2, fma(u2, 0.1428571492, 0.2000000030), 0.3333333433);
    let ln_m = fma(2.0 * u, tail, 2.0 * u);
    return fma(e, LN2_HI, fma(e, LN2_LO, ln_m));
}

fn ln1p_accurate(x: f32) -> f32 {
    if (abs(x) < 0.25) {
        let u = x / (2.0 + x);
        let u2 = u * u;
        let tail = u2 * fma(u2, fma(u2, 0.1428571492, 0.2000000030), 0.3333333433);
        return fma(2.0 * u, tail, 2.0 * u);
    }
    return ln_accurate(1.0 + x);
}

fn exp_accurate(x: f32) -> f32 {
    if (x < -88.0) {
        return 0.0;
    }
    let xc = min(x, 88.0);
    let kf = round(xc * INV_LN2);
    let r = fma(-kf, LN2_HI, xc) - kf * LN2_LO;
    let p = fma(
        r,
        fma(
            r,
            fma(
                r,
                fma(r, fma(r, fma(r, 0.0013888889, 0.0083333338), 0.0416666679), 0.1666666716),
                0.5,
            ),
            1.0,
        ),
        1.0,
    );
    return ldexp(p, i32(kf));
}

fn logpdf_dyadic(v: vec2<f32>, pe: AfSeqPriorEntry, extra_hi: f32, extra_lo: f32) -> f32 {
    let diff = (v.x - (pe.mu_hi + extra_hi)) + ((v.y - pe.mu_lo) - extra_lo);
    let z = fma(diff, pe.inv_sd_hi, diff * pe.inv_sd_lo);
    let q_hi = -0.5 * z * z;
    let q_lo = fma(-0.5 * z, z, -q_hi);
    return (q_hi - pe.ln_sd_hi) + (q_lo - pe.ln_sd_lo);
}

// Returns vec2(l_seq, l_af) evaluated against the exact same pooled Gaussian background `bg_no_tau`.
fn af_llr_pair(
    res_idx: u32,
    atom_idx: u32,
    v: vec2<f32>,
    d_mu: vec2<f32>,
    ae: AfPriorEntry,
) -> vec2<f32> {
    if (res_idx >= 20u) {
        return vec2<f32>(0.0, 0.0);
    }
    let pe = af_seq_priors[res_idx * 5u + atom_idx];
    if (pe.present == 0u) {
        return vec2<f32>(af_params.gly_cb, af_params.gly_cb);
    }
    let d_mu_val = d_mu.x + d_mu.y;
    var lp_res: f32;
    if (abs(d_mu_val) < 1e-6) {
        lp_res = logpdf_dyadic(v, pe, 0.0, 0.0);
    } else if (abs(d_mu_val) < 0.09) {
        lp_res = logpdf_dyadic(v, pe, d_mu.x, d_mu.y);
    } else {
        let raw = d_mu_val / (COVALENT_COIL_WEIGHT * (1.0 - NATURAL_PRO_FRACTION));
        let d_non = -COVALENT_COIL_WEIGHT * NATURAL_PRO_FRACTION * raw;
        let tot_coil = d_non + raw;
        let coil_hi = round(tot_coil * 16.0) * 0.0625;
        let coil_lo = tot_coil - coil_hi;
        let str_hi = round(d_non * 16.0) * 0.0625;
        let str_lo = d_non - str_hi;
        let a = LN_COVALENT_COIL_WEIGHT + logpdf_dyadic(v, pe, coil_hi, coil_lo);
        let b = LN_COVALENT_STRUCT_WEIGHT + logpdf_dyadic(v, pe, str_hi, str_lo);
        let m = max(a, b);
        lp_res = m + ln1p_accurate(exp_accurate(min(a, b) - m));
    }

    var lps = array<f32, 20>();
    var max_lp: f32 = -1e30;
    var count: u32 = 0u;
    for (var r: u32 = 0u; r < 20u; r = r + 1u) {
        let p_r = af_seq_priors[r * 5u + atom_idx];
        if (p_r.present != 0u) {
            let lp = logpdf_dyadic(v, p_r, 0.0, 0.0);
            lps[r] = lp;
            if (lp > max_lp) {
                max_lp = lp;
            }
            count = count + 1u;
        }
    }
    var sum_exp: f32 = 0.0;
    for (var r: u32 = 0u; r < 20u; r = r + 1u) {
        let p_r = af_seq_priors[r * 5u + atom_idx];
        if (p_r.present != 0u) {
            sum_exp = sum_exp + exp_accurate(lps[r] - max_lp);
        }
    }
    let bg_no_tau = max_lp + ln_accurate(sum_exp / f32(count));
    let l_seq = max(lp_res - bg_no_tau, af_params.llr_floor);

    var l_af: f32 = l_seq;
    if (ae.present != 0u && ae.omega > 0.0) {
        let diff_af = (v.x - ae.mu_af_hi) + (v.y - ae.mu_af_lo);
        let z_af = fma(diff_af, ae.inv_sigma_af_hi, diff_af * ae.inv_sigma_af_lo);
        let q = 0.25 * z_af * z_af;
        let ln_t4_kernel = -2.5 * ln1p_accurate(q);
        let lp_af_no_tau = (ae.c_af_hi + ln_t4_kernel) + ae.c_af_lo;
        l_af = max(lp_af_no_tau - bg_no_tau, af_params.llr_floor);
    }
    return vec2<f32>(l_seq, l_af);
}

@compute @workgroup_size(16, 16, 1)
fn compute_unary_af(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    let k = gid.y;
    if (s >= af_params.s_len || k >= af_params.k_len) {
        return;
    }
    let out_idx = s * af_params.k_len + k;
    if (k == 0u) {
        af_out_u[out_idx] = FORBID;
        return;
    }
    let cur = af_seq_res[k];
    let res = cur.res_idx;
    if (res == PRO_IDX) {
        af_out_u[out_idx] = FORBID;
        return;
    }
    let prev_e = af_seq_res[k - 1u];
    let prev = prev_e.res_idx;

    var slot_seq = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var slot_delta = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var total_pos_cands: u32 = 0u;
    var has_obs_ca: u32 = 0u;
    var obs_slots: u32 = 0u;
    var max_sup: u32 = 0u;

    for (var atom_c: u32 = 0u; atom_c < 3u; atom_c = atom_c + 1u) {
        let atom_idx = atom_c + 2u; // 2=CA, 3=CB, 4=C
        for (var off_idx: u32 = 0u; off_idx < 2u; off_idx = off_idx + 1u) {
            let local_slot = atom_c * 2u + off_idx;
            let slot = s * 6u + local_slot;
            let n_raw = af_cand_counts[slot];
            if (n_raw == 0u) {
                continue;
            }
            let base = slot * af_params.max_cands;
            let target_idx = select(k - 1u, k, off_idx == 0u);
            let te = af_seq_res[target_idx];
            let d_mu = select(
                select(vec2<f32>(te.d_c_hi, te.d_c_lo), vec2<f32>(te.d_cb_hi, te.d_cb_lo), atom_c == 1u),
                vec2<f32>(te.d_ca_hi, te.d_ca_lo),
                atom_c == 0u,
            );
            let ae = af_priors[target_idx * 5u + atom_idx];
            let af_active = ae.present != 0u && ae.omega > 0.0;

            var n_pos: u32 = 0u;
            var best_seq: f32 = -1e30;
            var best_pool: f32 = -1e30;
            for (var c: u32 = 0u; c < n_raw; c = c + 1u) {
                let ce = af_cands[base + c];
                if (ce.support > 0u) {
                    n_pos = n_pos + 1u;
                    if (ce.support > max_sup) {
                        max_sup = ce.support;
                    }
                    let pair = af_llr_pair(
                        te.res_idx,
                        atom_idx,
                        vec2<f32>(ce.val_hi, ce.val_lo),
                        d_mu,
                        ae,
                    );
                    let bonus = af_params.support_bonus * (f32(ce.support) - 1.0);
                    let seq_c = pair.x + bonus;
                    if (seq_c > best_seq) {
                        best_seq = seq_c;
                    }
                    if (af_active) {
                        let l_pool = fma(1.0 - ae.omega, pair.x, ae.omega * pair.y);
                        let pool_c = l_pool + bonus;
                        if (pool_c > best_pool) {
                            best_pool = pool_c;
                        }
                    }
                }
            }
            if (n_pos > 0u) {
                total_pos_cands = total_pos_cands + n_pos;
                obs_slots = obs_slots + 1u;
                if (atom_c == 0u) {
                    has_obs_ca = 1u;
                }
                slot_seq[local_slot] = best_seq - ln_accurate(f32(n_pos));
                if (af_active) {
                    slot_delta[local_slot] = best_pool - best_seq;
                }
            }
        }
    }

    if (af_params.forbid_zero_carbons != 0u && total_pos_cands == 0u) {
        af_out_u[out_idx] = FORBID;
        return;
    }
    if (af_params.has_excess_roots != 0u && has_obs_ca == 0u && res != GLY_IDX && prev != PRO_IDX) {
        af_out_u[out_idx] = FORBID;
        return;
    }

    var hn_seq: f32 = 0.0;
    var hn_delta: f32 = 0.0;
    if (af_params.use_hn != 0u) {
        let hn = af_roots_hn[s];
        let ae_n = af_priors[k * 5u + 1u];
        let ae_h = af_priors[k * 5u];
        let np = af_llr_pair(
            res,
            1u,
            vec2<f32>(hn.z, hn.w),
            vec2<f32>(cur.d_n_hi, cur.d_n_lo),
            ae_n,
        );
        let hp = af_llr_pair(
            res,
            0u,
            vec2<f32>(hn.x, hn.y),
            vec2<f32>(0.0, 0.0),
            ae_h,
        );
        hn_seq = np.x + hp.x;
        if (ae_n.present != 0u && ae_n.omega > 0.0) {
            hn_delta = hn_delta + ae_n.omega * (np.y - np.x);
        }
        if (ae_h.present != 0u && ae_h.omega > 0.0) {
            hn_delta = hn_delta + ae_h.omega * (hp.y - hp.x);
        }
    }

    var tail_term: f32 = -af_params.assign_cost;
    if (af_params.has_excess_roots != 0u
        && af_params.multi_slot_dataset != 0u
        && (has_obs_ca == 0u || (obs_slots <= 1u && max_sup <= 1u)))
    {
        tail_term = tail_term - 4.5;
    }

    let sum_seq_a = (slot_seq[0] + slot_seq[1]) + (slot_seq[2] + slot_seq[3]);
    let sum_seq_b = (slot_seq[4] + slot_seq[5]) + (hn_seq + tail_term);
    let raw_delta_a = (slot_delta[0] + slot_delta[1]) + (slot_delta[2] + slot_delta[3]);
    let raw_delta_b = (slot_delta[4] + slot_delta[5]) + hn_delta;
    let clamped_delta = clamp(
        raw_delta_a + raw_delta_b,
        -af_params.af_clamp_total,
        af_params.af_clamp_total,
    );
    af_out_u[out_idx] = (sum_seq_a + sum_seq_b) + clamped_delta;
}
