// Relaxation Dispersion (`CPMG`: `Flat`, `LM63`, `CR72`, `Baldwin2014`, and Trott-Palmer `R1rho`)
// WebGPU (`WGSL`) forward evaluator and multi-field `(p_B, k_ex)` global grid profiler (`T13.5`).
//
// Provenance: Clean-room GPU compute implementation citing:
// - Luz, Z. & Meiboom, S. (J. Chem. Phys. 1963, 39, 366-370 — LM63).
// - Carver, J. P. & Richards, R. E. (J. Magn. Reson. 1972, 6, 89-105 — CR72).
// - Baldwin, A. J. (J. Magn. Reson. 2014, 244, 114-124 — exact closed-form 2-site R2,eff).
// - Trott, O. & Palmer, A. G. III (J. Magn. Reson. 2002, 154, 157-160 — off-resonance R1rho).

struct CpmgGpuUniforms {
    n_jobs: u32,
    n_res: u32,
    n_pb: u32,
    n_kex: u32,
    model_code: u32, // 0=Flat, 1=LM63, 2=CR72, 3=Baldwin2014, 4=TrottPalmerR1rho
    job_offset: u32,
    _pad1: u32,
    _pad2: u32,
}

struct CpmgPointEvalGpu {
    model_code: u32,
    _pad: u32,
    r2a: f32,
    r2b: f32,
    p_b: f32,
    k_ex: f32,
    dw_rad: f32,
    t_relax_or_r1: f32,
    nu_or_w1: f32,
    om_a_offset: f32,
}

struct ResidueSpanGpu {
    series_start: u32,
    series_len: u32,
}

struct SeriesMetaGpu {
    larmor_mhz: f32,
    t_relax_s: f32,
    pt_start: u32,
    pt_len: u32,
}

struct ObsPointGpu {
    nu_cpmg_hz: f32,
    r2_obs: f32,
    inv_var: f32,
    _pad: f32,
}

struct GridCellOutGpu {
    best_chi2: f32,
    best_dw_ppm: f32,
}

@group(0) @binding(0) var<uniform> params: CpmgGpuUniforms;
@group(0) @binding(1) var<storage, read> eval_points: array<CpmgPointEvalGpu>;
@group(0) @binding(2) var<storage, read_write> eval_outputs: array<f32>;
@group(0) @binding(3) var<storage, read> pb_grid: array<f32>;
@group(0) @binding(4) var<storage, read> kex_grid: array<f32>;
@group(0) @binding(5) var<storage, read> res_spans: array<ResidueSpanGpu>;
@group(0) @binding(6) var<storage, read> series_meta: array<SeriesMetaGpu>;
@group(0) @binding(7) var<storage, read> obs_points: array<ObsPointGpu>;
@group(0) @binding(8) var<storage, read_write> out_grid_cells: array<GridCellOutGpu>;

const TAU_F32: f32 = 6.283185307179586;
const SQRT2_F32: f32 = 1.4142135623730951;
const GOLDEN_PHI: f32 = 0.6180339887498949;

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn safe_acosh(x_in: f32) -> f32 {
    let x = max(x_in, 1.0);
    let xm1 = x - 1.0;
    if (xm1 < 1e-4) {
        return sqrt(2.0 * xm1) * (1.0 - xm1 / 12.0);
    }
    return log(x + sqrt(max(x * x - 1.0, 0.0)));
}

fn c_cosh_sinc(z: vec2<f32>, t: f32) -> array<vec2<f32>, 2> {
    let u = z * t;
    let u2_norm = dot(u, u);
    var res: array<vec2<f32>, 2>;
    if (u2_norm < 1e-5) {
        let u2 = c_mul(u, u);
        let u4 = c_mul(u2, u2);
        res[0] = vec2<f32>(1.0 + 0.5 * u2.x + u4.x / 24.0, 0.5 * u2.y + u4.y / 24.0);
        res[1] = vec2<f32>(t * (1.0 + u2.x / 6.0 + u4.x / 120.0), t * (u2.y / 6.0 + u4.y / 120.0));
    } else {
        let ch_x = cosh(u.x);
        let sh_x = sinh(u.x);
        let cs_y = cos(u.y);
        let sn_y = sin(u.y);
        res[0] = vec2<f32>(ch_x * cs_y, sh_x * sn_y);
        let sh = vec2<f32>(sh_x * cs_y, ch_x * sn_y);
        let z_abs2 = max(dot(z, z), 1e-30);
        res[1] = vec2<f32>((sh.x * z.x + sh.y * z.y) / z_abs2, (sh.y * z.x - sh.x * z.y) / z_abs2);
    }
    return res;
}

fn eval_lm63_f32(r20: f32, pb: f32, kex: f32, dw_rad: f32, nu: f32) -> f32 {
    if (pb <= 0.0 || kex <= 1e-6 || nu <= 0.0) {
        return r20;
    }
    let pa = 1.0 - pb;
    let phi_ex = pa * pb * dw_rad * dw_rad;
    let x = kex / (4.0 * nu);
    var fac: f32;
    if (abs(x) < 1e-2) {
        let x2 = x * x;
        fac = x2 / 3.0 - 2.0 * x2 * x2 / 15.0;
    } else {
        fac = 1.0 - tanh(x) / x;
    }
    return r20 + (phi_ex / kex) * fac;
}

fn eval_cr72_f32(r2a: f32, r2b: f32, pb_in: f32, kex: f32, dw_rad: f32, nu: f32) -> f32 {
    let pb = clamp(pb_in, 0.0, 0.999999);
    if (pb == 0.0 || kex <= 0.0 || nu <= 0.0) {
        return r2a;
    }
    let pa = 1.0 - pb;
    let kab = pb * kex;
    let kba = pa * kex;
    let alpha_m = (r2a - r2b) + kab - kba;
    let dw2 = dw_rad * dw_rad;
    let psi = alpha_m * alpha_m - dw2 + 4.0 * kab * kba;
    let zeta = 2.0 * dw_rad * alpha_m;
    let hyp = sqrt(psi * psi + zeta * zeta);
    if (hyp < 1e-24) {
        return 0.5 * (r2a + r2b + kex);
    }
    let inv_s = 1.0 / (2.0 * SQRT2_F32 * nu);
    let eta_p = inv_s * sqrt(max(hyp + psi, 0.0));
    let eta_m = inv_s * sqrt(max(hyp - psi, 0.0));
    let ratio = (psi + 2.0 * dw2) / hyp;
    let d_p = 0.5 * (1.0 + ratio);
    let d_m = 0.5 * (-1.0 + ratio);
    let arg = max(d_p * cosh(eta_p) - d_m * cos(eta_m), 1.0);
    return 0.5 * (r2a + r2b + kex) - nu * safe_acosh(arg);
}

fn eval_baldwin2014_f32(
    r2a: f32,
    r2b: f32,
    pb_in: f32,
    kex: f32,
    dw_rad: f32,
    t_relax: f32,
    nu: f32,
) -> f32 {
    let pb = clamp(pb_in, 0.0, 0.999999);
    if (pb == 0.0 || kex <= 0.0 || nu <= 0.0 || t_relax <= 0.0) {
        return r2a;
    }
    let pa = 1.0 - pb;
    let kab = pb * kex;
    let kba = pa * kex;
    let r2_bar = 0.5 * (r2a + r2b);
    let dr_ab = r2a - r2b;
    let alpha_m = dr_ab + kab - kba;
    let dw2 = dw_rad * dw_rad;

    let psi = alpha_m * alpha_m - dw2 + 4.0 * kab * kba;
    let zeta = 2.0 * dw_rad * alpha_m;
    let hyp = sqrt(psi * psi + zeta * zeta);

    let z_re = 0.5 * sqrt(0.5 * max(hyp + psi, 0.0));
    let z_im = 0.5 * sqrt(0.5 * max(hyp - psi, 0.0)) * select(-1.0, 1.0, zeta >= 0.0);
    let z = vec2<f32>(z_re, z_im);

    let tau = 1.0 / (4.0 * nu);
    let cs1 = c_cosh_sinc(z, tau);
    let cs2 = c_cosh_sinc(z, 2.0 * tau);
    let s1 = cs1[1];
    let c2 = cs2[0];
    let s2 = cs2[1];

    let t_m = 0.25 * (psi + 2.0 * dw2);
    let u_trace = dot(c2, c2) + dot(s2, s2) * t_m;

    let c2_conj = vec2<f32>(c2.x, -c2.y);
    let s2_conj = vec2<f32>(s2.x, -s2.y);
    let s1_sq = c_mul(s1, s1);
    let s1_sq_s2c = c_mul(s1_sq, s2_conj);
    let c2c_s2 = c_mul(c2_conj, s2);
    let beta1 = c2c_s2 + 2.0 * t_m * s1_sq_s2c;
    let beta2 = s2_conj;

    let vm = vec2<f32>(0.5 * (kex - dr_ab), -0.5 * dw_rad);
    let vm_conj = vec2<f32>(vm.x, -vm.y);
    let w_a_re = c_mul(beta1, vm).x + c_mul(beta2, vm_conj).x;

    let n_cyc = t_relax * nu;
    let base_rate = r2_bar + 0.5 * kex;

    if (u_trace > 1.0 + 1e-5) {
        let gamma = safe_acosh(u_trace);
        let sinh_g = sqrt(max(u_trace * u_trace - 1.0, 1e-20));
        let exp_m2ng = exp(-2.0 * n_cyc * gamma);
        let proj = max(0.5 * (1.0 + exp_m2ng) + w_a_re * 0.5 * (1.0 - exp_m2ng) / sinh_g, 1e-30);
        return base_rate - nu * gamma - log(proj) / t_relax;
    } else if (u_trace >= 1.0 - 1e-5) {
        let proj = max(1.0 + n_cyc * w_a_re, 1e-30);
        return base_rate - log(proj) / t_relax;
    } else {
        let theta = acos(clamp(u_trace, -1.0, 1.0));
        let sin_t = max(sin(theta), 1e-6);
        let proj = max(cos(n_cyc * theta) + w_a_re * sin(n_cyc * theta) / sin_t, 1e-30);
        return base_rate - log(proj) / t_relax;
    }
}

fn eval_trott_palmer_r1rho_f32(
    r1: f32,
    r20: f32,
    pb_in: f32,
    kex: f32,
    om_a: f32,
    dw_ab: f32,
    w1: f32,
) -> f32 {
    let pb = clamp(pb_in, 0.0, 0.999999);
    let pa = 1.0 - pb;
    let om_b = om_a + dw_ab;
    let om_bar = pa * om_a + pb * om_b;
    let w1_sq = w1 * w1;
    let w_ae_sq = om_a * om_a + w1_sq;
    let w_be_sq = om_b * om_b + w1_sq;
    let w_e_sq = max(om_bar * om_bar + w1_sq, 1e-20);
    let sin2 = w1_sq / w_e_sq;
    let cos2 = (om_bar * om_bar) / w_e_sq;
    let denom = (w_ae_sq * w_be_sq) / w_e_sq + kex * kex;
    let rex = select(0.0, (pa * pb * dw_ab * dw_ab * kex) / denom, denom > 1e-20);
    return r1 * cos2 + r20 * sin2 + sin2 * rex;
}

fn eval_rex_only_model(model_code: u32, pb: f32, kex: f32, dw_rad: f32, t_relax: f32, nu: f32) -> f32 {
    if (model_code == 1u) {
        return eval_lm63_f32(0.0, pb, kex, dw_rad, nu);
    } else if (model_code == 2u) {
        return eval_cr72_f32(0.0, 0.0, pb, kex, dw_rad, nu);
    } else {
        return eval_baldwin2014_f32(0.0, 0.0, pb, kex, dw_rad, t_relax, nu);
    }
}

fn residue_chi2_for_dw(res_idx: u32, model_code: u32, pb: f32, kex: f32, dw_ppm: f32) -> f32 {
    let span = res_spans[res_idx];
    var total_c2: f32 = 0.0;

    for (var si = 0u; si < span.series_len; si = si + 1u) {
        let sm = series_meta[span.series_start + si];
        let dw_rad = TAU_F32 * sm.larmor_mhz * dw_ppm;
        var w_sum: f32 = 0.0;
        var w_diff: f32 = 0.0;

        for (var pi = 0u; pi < sm.pt_len; pi = pi + 1u) {
            let pt = obs_points[sm.pt_start + pi];
            let rex = eval_rex_only_model(model_code, pb, kex, dw_rad, sm.t_relax_s, pt.nu_cpmg_hz);
            w_sum = w_sum + pt.inv_var;
            w_diff = w_diff + pt.inv_var * (pt.r2_obs - rex);
        }
        let r20 = max(w_diff / max(w_sum, 1e-12), 0.0);

        for (var pi = 0u; pi < sm.pt_len; pi = pi + 1u) {
            let pt = obs_points[sm.pt_start + pi];
            let rex = eval_rex_only_model(model_code, pb, kex, dw_rad, sm.t_relax_s, pt.nu_cpmg_hz);
            let diff = pt.r2_obs - (r20 + rex);
            total_c2 = total_c2 + pt.inv_var * diff * diff;
        }
    }
    return total_c2;
}

@compute @workgroup_size(64)
fn eval_cpmg_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_jobs) {
        return;
    }
    let p = eval_points[idx];
    var out: f32 = p.r2a;
    if (p.model_code == 0u) {
        out = p.r2a;
    } else if (p.model_code == 1u) {
        out = eval_lm63_f32(p.r2a, p.p_b, p.k_ex, p.dw_rad, p.nu_or_w1);
    } else if (p.model_code == 2u) {
        out = eval_cr72_f32(p.r2a, p.r2b, p.p_b, p.k_ex, p.dw_rad, p.nu_or_w1);
    } else if (p.model_code == 3u) {
        out = eval_baldwin2014_f32(p.r2a, p.r2b, p.p_b, p.k_ex, p.dw_rad, p.t_relax_or_r1, p.nu_or_w1);
    } else if (p.model_code == 4u) {
        out = eval_trott_palmer_r1rho_f32(
            p.t_relax_or_r1,
            p.r2a,
            p.p_b,
            p.k_ex,
            p.om_a_offset,
            p.dw_rad,
            p.nu_or_w1,
        );
    }
    eval_outputs[idx] = out;
}

@compute @workgroup_size(64)
fn grid_search_cpmg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_idx = gid.x;
    if (local_idx >= params.n_jobs) {
        return;
    }
    let global_idx = params.job_offset + local_idx;
    let res_idx = global_idx % params.n_res;
    let g_idx = global_idx / params.n_res;
    let ipb = g_idx / params.n_kex;
    let ikex = g_idx % params.n_kex;

    let pb = pb_grid[ipb];
    let kex = kex_grid[ikex];

    var best_dw: f32 = 1.0;
    var best_c2: f32 = 1e30;
    for (var k = 1u; k <= 36u; k = k + 1u) {
        let dw = 0.25 * f32(k);
        let c2 = residue_chi2_for_dw(res_idx, params.model_code, pb, kex, dw);
        if (c2 < best_c2) {
            best_c2 = c2;
            best_dw = dw;
        }
    }

    var lo = max(best_dw - 0.35, 0.02);
    var hi = min(best_dw + 0.35, 20.0);
    var x1 = hi - GOLDEN_PHI * (hi - lo);
    var x2 = lo + GOLDEN_PHI * (hi - lo);
    var f1 = residue_chi2_for_dw(res_idx, params.model_code, pb, kex, x1);
    var f2 = residue_chi2_for_dw(res_idx, params.model_code, pb, kex, x2);

    for (var it = 0u; it < 14u; it = it + 1u) {
        if (f1 < f2) {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - GOLDEN_PHI * (hi - lo);
            f1 = residue_chi2_for_dw(res_idx, params.model_code, pb, kex, x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + GOLDEN_PHI * (hi - lo);
            f2 = residue_chi2_for_dw(res_idx, params.model_code, pb, kex, x2);
        }
    }

    let opt_dw = 0.5 * (lo + hi);
    let opt_c2 = residue_chi2_for_dw(res_idx, params.model_code, pb, kex, opt_dw);
    out_grid_cells[local_idx] = GridCellOutGpu(opt_c2, opt_dw);
}
