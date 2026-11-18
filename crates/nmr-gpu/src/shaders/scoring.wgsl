// Clean-room WGSL compute shaders for backbone NMR unary typing U(s, k)
// (with optional heavy-tailed Student-t(nu = 4) 3D AlphaFold shift priors per
// docs/specs/21-alphafold-integrative-nmr.md section 4.2 & 4.4)
// and pairwise sequential connectivity P(s, t) matrices.
// Provenance: Implements docs/specs/03-assignment.md section 3 and
// Lange, Little & Taylor (J. Am. Stat. Assoc. 1989, 84, 881-896).

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
    enable_af_priors: u32,
    af_weight: f32,
    af_clamp_nats: f32,
    _af_pad: u32,
};

struct PriorEntry {
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
    mu_bg_hi: f32,
    mu_bg_lo: f32,
    inv_sigma_bg_hi: f32,
    inv_sigma_bg_lo: f32,
    log_sigma_ratio_hi: f32,
    log_sigma_ratio_lo: f32,
    w_af: f32,
    present: u32,
};

struct CandEntry {
    val_hi: f32,
    val_lo: f32,
    support: u32,
    _pad: u32,
};

struct SeqEntry {
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

@group(0) @binding(0) var<uniform> params: ScoreParams;
// 20 residues x 5 atoms (H=0, N=1, CA=2, CB=3, C=4)
@group(0) @binding(1) var<storage, read> priors: array<PriorEntry>;
// Sequence residue entry per k (res_idx 0..19, 14 = PRO, 255 = UNK, plus dyadic covalent offsets)
@group(0) @binding(2) var<storage, read> seq_res: array<SeqEntry>;
// Root (H_hi, H_lo, N_hi, N_lo) per spin system
@group(0) @binding(3) var<storage, read> roots_hn: array<vec4<f32>>;
// Candidates indexed [((s * 3 + atom_c) * 2 + off_idx) * max_cands + c]
// atom_c: 0=CA, 1=CB, 2=C; off_idx: 0=(i), 1=(i-1)
@group(0) @binding(4) var<storage, read> cands: array<CandEntry>;
// Counts indexed [(s * 3 + atom_c) * 2 + off_idx]
@group(0) @binding(5) var<storage, read> cand_counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_u: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_p: array<f32>;
// Opt-in k_len x 5 atoms Student-t(nu = 4) 3D shift priors (H=0, N=1, CA=2, CB=3, C=4)
@group(0) @binding(8) var<storage, read> af_priors: array<AfPriorEntry>;

const TAU: f32 = 6.283185307179586;
const FORBID: f32 = -1e9;
const GLY_IDX: u32 = 7u;
const PRO_IDX: u32 = 14u;
const NATURAL_PRO_FRACTION: f32 = 0.0362838916; // 91.0 / 2508.0
const COVALENT_COIL_WEIGHT: f32 = 0.85;
const LN_COVALENT_COIL_WEIGHT: f32 = -0.1625189295; // ln(0.85)
const LN_COVALENT_STRUCT_WEIGHT: f32 = -1.8971199849; // ln(0.15)
const LN2_HI: f32 = 0.693145751953125;
const LN2_LO: f32 = 1.428606765330187e-6;
const INV_LN2: f32 = 1.4426950408889634;

// Sub-0.5-ULP natural logarithm via WGSL built-in `frexp` + odd atanh series on [sqrt(0.5), sqrt(2)],
// avoiding multi-ULP GPU SFU `fast::log` drift across 8-slot sums.
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

// Sub-0.5-ULP exponential via Cody-Waite two-part ln(2) reduction + degree-6 Horner + `ldexp`.
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

// Exact dyadic-16th difference `(v_hi - mu_hi)` (0.0 ULP in f32) + bounded remainder `(v_lo - mu_lo)`,
// with FMA-compensated `inv_sd` scaling. `-0.5 * ln(2 * pi)` cancels identically in `llr()`.
fn logpdf_dyadic(v: vec2<f32>, pe: PriorEntry, extra_hi: f32, extra_lo: f32) -> f32 {
    let diff = (v.x - (pe.mu_hi + extra_hi)) + ((v.y - pe.mu_lo) - extra_lo);
    let z = fma(diff, pe.inv_sd_hi, diff * pe.inv_sd_lo);
    let q_hi = -0.5 * z * z;
    let q_lo = fma(-0.5 * z, z, -q_hi);
    return (q_hi - pe.ln_sd_hi) + (q_lo - pe.ln_sd_lo);
}

fn llr(res_idx: u32, atom_idx: u32, v: vec2<f32>, d_mu: vec2<f32>) -> f32 {
    if (res_idx >= 20u) {
        return 0.0;
    }
    let pe = priors[res_idx * 5u + atom_idx];
    if (pe.present == 0u) {
        return params.gly_cb;
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
        let p_r = priors[r * 5u + atom_idx];
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
        let p_r = priors[r * 5u + atom_idx];
        if (p_r.present != 0u) {
            sum_exp = sum_exp + exp_accurate(lps[r] - max_lp);
        }
    }
    let bg = max_lp + ln_accurate(sum_exp / f32(count));
    return max(lp_res - bg, params.llr_floor);
}

fn af_atom_llr(ae: AfPriorEntry, v: vec2<f32>) -> f32 {
    if (ae.present == 0u) {
        return 0.0;
    }
    let diff_af = (v.x - ae.mu_af_hi) + (v.y - ae.mu_af_lo);
    let diff_bg = (v.x - ae.mu_bg_hi) + (v.y - ae.mu_bg_lo);
    let z_af = fma(diff_af, ae.inv_sigma_af_hi, diff_af * ae.inv_sigma_af_lo);
    let z_bg = fma(diff_bg, ae.inv_sigma_bg_hi, diff_bg * ae.inv_sigma_bg_lo);
    let denom = fma(0.25 * z_bg, z_bg, 1.0);
    let dz = (z_af - z_bg) * (z_af + z_bg);
    let rel = (0.25 * dz) / denom;
    return fma(-2.5, ln1p_accurate(rel), -ae.log_sigma_ratio_hi) - ae.log_sigma_ratio_lo;
}

fn compute_af_delta_u(s: u32, k: u32) -> f32 {
    let w_k = af_priors[k * 5u].w_af;
    if (w_k <= 0.0 || params.af_weight <= 0.0) {
        return 0.0;
    }
    var af_terms = array<f32, 3>(0.0, 0.0, 0.0);

    for (var atom_c: u32 = 0u; atom_c < 3u; atom_c = atom_c + 1u) {
        let atom_idx = atom_c + 2u;
        let ae = af_priors[k * 5u + atom_idx];
        if (ae.present == 0u) {
            continue;
        }
        let slot = s * 6u + atom_c * 2u; // off_idx == 0 (intra-residue i)
        let n_raw = cand_counts[slot];
        if (n_raw == 0u) {
            continue;
        }
        let base = slot * params.max_cands;
        var n_pos: u32 = 0u;
        var best_af: f32 = -1e30;
        for (var c: u32 = 0u; c < n_raw; c = c + 1u) {
            let ce = cands[base + c];
            if (ce.support > 0u) {
                n_pos = n_pos + 1u;
                let term = af_atom_llr(ae, vec2<f32>(ce.val_hi, ce.val_lo));
                if (term > best_af) {
                    best_af = term;
                }
            }
        }
        if (n_pos > 0u) {
            af_terms[atom_c] = best_af - ln_accurate(f32(n_pos));
        }
    }

    var hn_af: f32 = 0.0;
    if (params.use_hn != 0u) {
        let hn = roots_hn[s];
        let ae_n = af_priors[k * 5u + 1u];
        let ae_h = af_priors[k * 5u];
        hn_af = af_atom_llr(ae_n, vec2<f32>(hn.z, hn.w)) + af_atom_llr(ae_h, vec2<f32>(hn.x, hn.y));
    }

    let raw_sum = (af_terms[0] + af_terms[1]) + (af_terms[2] + hn_af);
    let clamped = clamp(raw_sum, -params.af_clamp_nats, params.af_clamp_nats);
    return params.af_weight * w_k * clamped;
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

    var slot_terms = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var total_pos_cands: u32 = 0u;
    var has_obs_ca: u32 = 0u;
    var obs_slots: u32 = 0u;
    var max_sup: u32 = 0u;
    for (var atom_c: u32 = 0u; atom_c < 3u; atom_c = atom_c + 1u) {
        let atom_idx = atom_c + 2u; // 2=CA, 3=CB, 4=C
        for (var off_idx: u32 = 0u; off_idx < 2u; off_idx = off_idx + 1u) {
            let local_slot = atom_c * 2u + off_idx;
            let slot = s * 6u + local_slot;
            let n_raw = cand_counts[slot];
            if (n_raw == 0u) {
                continue;
            }
            let base = slot * params.max_cands;
            let target_idx = select(k - 1u, k, off_idx == 0u);
            let te = seq_res[target_idx];
            let d_mu = select(
                select(vec2<f32>(te.d_c_hi, te.d_c_lo), vec2<f32>(te.d_cb_hi, te.d_cb_lo), atom_c == 1u),
                vec2<f32>(te.d_ca_hi, te.d_ca_lo),
                atom_c == 0u,
            );
            var n_pos: u32 = 0u;
            var best: f32 = -1e30;
            for (var c: u32 = 0u; c < n_raw; c = c + 1u) {
                let ce = cands[base + c];
                if (ce.support > 0u) {
                    n_pos = n_pos + 1u;
                    if (ce.support > max_sup) {
                        max_sup = ce.support;
                    }
                    let term = llr(te.res_idx, atom_idx, vec2<f32>(ce.val_hi, ce.val_lo), d_mu)
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
                slot_terms[local_slot] = best - ln_accurate(f32(n_pos));
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

    var hn_term: f32 = 0.0;
    if (params.use_hn != 0u) {
        let hn = roots_hn[s];
        hn_term = llr(res, 1u, vec2<f32>(hn.z, hn.w), vec2<f32>(cur.d_n_hi, cur.d_n_lo))
            + llr(res, 0u, vec2<f32>(hn.x, hn.y), vec2<f32>(0.0, 0.0));
    }
    var tail_term: f32 = -params.assign_cost;
    if (params.has_excess_roots != 0u
        && params.multi_slot_dataset != 0u
        && (has_obs_ca == 0u || (obs_slots <= 1u && max_sup <= 1u)))
    {
        tail_term = tail_term - 4.5;
    }
    if (params.enable_af_priors != 0u) {
        tail_term = tail_term + compute_af_delta_u(s, k);
    }
    let sum_a = (slot_terms[0] + slot_terms[1]) + (slot_terms[2] + slot_terms[3]);
    let sum_b = (slot_terms[4] + slot_terms[5]) + (hn_term + tail_term);
    out_u[out_idx] = sum_a + sum_b;
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
    let log_range = ln_accurate(params.range_ppm);
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
        let const_term = -ln_accurate(sig * sqrt(TAU)) + log_range;
        let base_a = slot_a * params.max_cands;
        let base_b = slot_b * params.max_cands;

        var best: f32 = -1e30;
        for (var ia: u32 = 0u; ia < na; ia = ia + 1u) {
            let ca = cands[base_a + ia];
            for (var ib: u32 = 0u; ib < nb; ib = ib + 1u) {
                let cb = cands[base_b + ib];
                let dv = (ca.val_hi - cb.val_hi) + (ca.val_lo - cb.val_lo);
                let d = dv / sig;
                let term = max(fma(-0.5 * d, d, const_term), params.pair_floor);
                if (term > best) {
                    best = term;
                }
            }
        }
        let pen = ln_accurate(f32(na * nb));
        total_p = total_p + (best - pen);
    }
    out_p[out_idx] = total_p;
}
