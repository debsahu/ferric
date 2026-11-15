// Batched WebGPU (`wgpu` + WGSL) 2-parameter & AIC/BIC-gated 3-parameter exponential relaxation
// curve fitter (`I(t) = I0 * exp(-R * t) [+ C]`) and stateless counter-based parametric Monte
// Carlo bootstrap engine (`T13.2`, `docs/specs/16-dynamics-and-csp.md` §1.4–§1.5).
//
// Provenance: Clean-room WGSL compute implementation strictly from peer-reviewed NMR and numerical
// optimization literature (zero external GPL/academic tool code consulted; zero external product
// names in identifiers):
// - Kay, L. E., Torchia, D. A. & Bax, A. (Biochemistry 1989, 28, 8972–8979).
// - Farrow, N. A. et al. (Biochemistry 1994, 33, 5984–6003 & J. Biomol. NMR 1995, 6, 153–162).
// - Lipari, G. & Szabo, A. (J. Am. Chem. Soc. 1982, 104, 4546–4559 & 4559–4570).
// - Clore, G. M. et al. (J. Am. Chem. Soc. 1990, 112, 4989–4991).
// - Mandel, A. M., Akke, M. & Palmer, A. G. III (J. Mol. Biol. 1995, 246, 144–163 — parametric
//   Monte Carlo confidence intervals for relaxation rates).
// - Woessner, D. E. (J. Chem. Phys. 1962, 37, 647–654).
// - d'Auvergne, E. J. & Gooley, P. R. (J. Biomol. NMR 2003, 25, 25–39).
// - Levenberg, K. (Quart. Appl. Math. 1944, 2, 164–168); Marquardt, D. W. (J. Soc. Indust. Appl.
//   Math. 1963, 11, 431–441); Golub, G. H. & Pereyra, V. (SIAM J. Numer. Anal. 1973, 10, 413–432).

const MAX_CURVE_PTS: u32 = 64u;
const TWO_PI: f32 = 6.283185307179586;

struct RelaxMcParams {
    n_curves: u32,
    mc_reps_total: u32,
    rep_offset: u32,
    rep_chunk: u32,
    offset_gate: u32,
    seed_lo: u32,
    seed_hi: u32,
    _pad: u32,
}

struct CurveMetaGpu {
    pt_offset: u32,
    n_pts: u32,
    seq_id_u32: u32,
    r_tilde_max: f32,
}

struct CurvePointGpu {
    u: f32,
    y: f32,
    s_norm: f32,
    w: f32,
}

struct PointFitGpu {
    i0_t: f32,
    r_t: f32,
    c_t: f32,
    chi2: f32,
    use_3p: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: RelaxMcParams;
@group(0) @binding(1) var<storage, read> metas: array<CurveMetaGpu>;
@group(0) @binding(2) var<storage, read> points: array<CurvePointGpu>;
@group(0) @binding(3) var<storage, read_write> point_fits: array<PointFitGpu>;
@group(0) @binding(4) var<storage, read_write> mc_samples: array<vec2<f32>>;

// Stateless 4-word bijective Philox-style / PCG32 avalanche mixer returning two independent
// uniform random floats in (0, 1] for Box–Muller Gaussian synthesis.
fn hash4_u32(a: u32, b: u32, c: u32, d: u32) -> vec2<u32> {
    var x = a ^ 0x9E3779B9u;
    var y = b ^ 0x85EBCA6Bu;
    var z = c ^ 0xC2B2AE35u;
    var w = d ^ 0x27D4EB2Fu;

    for (var round: u32 = 0u; round < 4u; round = round + 1u) {
        x = (x + y) * 0x7FEB352Du;
        y = ((y << 13u) | (y >> 19u)) ^ x;
        z = (z + w) * 0x846CA68Bu;
        w = ((w << 17u) | (w >> 15u)) ^ z;
        let t = y;
        y = z;
        z = t + 0x9E3779B9u;
    }
    x = x ^ (x >> 16u);
    z = z ^ (z >> 16u);
    return vec2<u32>(x, z);
}

fn box_muller_pair(seed_lo: u32, seed_hi: u32, stream_a: u32, stream_b: u32) -> vec2<f32> {
    let h = hash4_u32(seed_lo, seed_hi, stream_a, stream_b);
    // Map u32 strictly into (0, 1) avoiding 0.0 for log()
    let u1 = (f32(h.x >> 8u) + 0.5) * (1.0 / 16777216.0);
    let u2 = (f32(h.y >> 8u) + 0.5) * (1.0 / 16777216.0);
    let rad = sqrt(max(-2.0 * log(u1), 0.0));
    let ang = TWO_PI * u2;
    return vec2<f32>(rad * cos(ang), rad * sin(ang));
}

fn varpro_two_param(
    pt_off: u32,
    n_pts: u32,
    y_arr: ptr<function, array<f32, 64>>,
    r_t: f32,
) -> vec2<f32> {
    var s1: f32 = 0.0;
    var s2: f32 = 0.0;
    for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
        let pt = points[pt_off + i];
        let e = exp(-r_t * pt.u);
        let we = pt.w * e;
        s1 = s1 + we * (*y_arr)[i];
        s2 = s2 + we * e;
    }
    var i0_t: f32 = 0.0;
    if (s2 > 1.0e-30) {
        i0_t = s1 / s2;
    }
    var chi2: f32 = 0.0;
    for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
        let pt = points[pt_off + i];
        let d = (*y_arr)[i] - i0_t * exp(-r_t * pt.u);
        chi2 = chi2 + pt.w * d * d;
    }
    return vec2<f32>(i0_t, chi2);
}

// Returns vec3<f32>(i0_t, r_t, chi2) matching `fit_two_param_normalized` on CPU.
fn fit_two_param(
    pt_off: u32,
    n_pts: u32,
    y_arr: ptr<function, array<f32, 64>>,
    r_tilde_max: f32,
) -> vec3<f32> {
    let grid_hi = max(min(r_tilde_max, 25.0), 0.5);
    var best_r: f32 = 0.0;
    let vp0 = varpro_two_param(pt_off, n_pts, y_arr, 0.0);
    var best_i0: f32 = vp0.x;
    var best_chi2: f32 = vp0.y;

    for (var g: u32 = 1u; g <= 32u; g = g + 1u) {
        let frac = f32(g) * (1.0 / 32.0);
        let r_cand = grid_hi * frac * frac;
        let vpc = varpro_two_param(pt_off, n_pts, y_arr, r_cand);
        if (vpc.y < best_chi2) {
            best_chi2 = vpc.y;
            best_i0 = vpc.x;
            best_r = r_cand;
        }
    }

    // Weighted log-linear starter on positive points
    var sw: f32 = 0.0;
    var su: f32 = 0.0;
    var sl: f32 = 0.0;
    var suu: f32 = 0.0;
    var sul: f32 = 0.0;
    var cnt: u32 = 0u;
    for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
        let yi = (*y_arr)[i];
        if (yi > 0.05) {
            let pt = points[pt_off + i];
            let wi = pt.w * yi * yi;
            let ly = log(yi);
            sw = sw + wi;
            su = su + wi * pt.u;
            sl = sl + wi * ly;
            suu = suu + wi * pt.u * pt.u;
            sul = sul + wi * pt.u * ly;
            cnt = cnt + 1u;
        }
    }
    let denom = sw * suu - su * su;
    if (cnt >= 2u && denom > 1.0e-20) {
        let r_lin = clamp(-(sw * sul - su * sl) / denom, 0.0, r_tilde_max);
        let vpl = varpro_two_param(pt_off, n_pts, y_arr, r_lin);
        if (vpl.y < best_chi2) {
            best_chi2 = vpl.y;
            best_i0 = vpl.x;
            best_r = r_lin;
        }
    }

    var i0_cur = best_i0;
    var r_cur = best_r;
    var chi2_cur = best_chi2;
    var lambda: f32 = 1.0e-4;

    for (var iter: u32 = 0u; iter < 24u; iter = iter + 1u) {
        var a00: f32 = 0.0;
        var a01: f32 = 0.0;
        var a11: f32 = 0.0;
        var g0: f32 = 0.0;
        var g1: f32 = 0.0;

        for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
            let pt = points[pt_off + i];
            let e = exp(-r_cur * pt.u);
            let res = (*y_arr)[i] - i0_cur * e;
            let j0 = e;
            let j1 = -i0_cur * pt.u * e;
            a00 = a00 + pt.w * j0 * j0;
            a01 = a01 + pt.w * j0 * j1;
            a11 = a11 + pt.w * j1 * j1;
            g0 = g0 + pt.w * j0 * res;
            g1 = g1 + pt.w * j1 * res;
        }

        var accepted = false;
        for (var sub: u32 = 0u; sub < 6u; sub = sub + 1u) {
            let d00 = a00 + lambda * max(a00, 1.0e-6);
            let d11 = a11 + lambda * max(a11, 1.0e-6);
            let det = d00 * d11 - a01 * a01;
            if (det <= 1.0e-28) {
                lambda = lambda * 10.0;
                continue;
            }
            let step_i0 = (d11 * g0 - a01 * g1) / det;
            let step_r = (d00 * g1 - a01 * g0) / det;

            let r_cand = clamp(r_cur + step_r, 0.0, r_tilde_max);
            let i0_lm = i0_cur + step_i0;
            var c2_lm: f32 = 0.0;
            for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
                let pt = points[pt_off + i];
                let d = (*y_arr)[i] - i0_lm * exp(-r_cand * pt.u);
                c2_lm = c2_lm + pt.w * d * d;
            }
            let vp_cand = varpro_two_param(pt_off, n_pts, y_arr, r_cand);
            var i0_next = i0_lm;
            var c2_next = c2_lm;
            if (vp_cand.y <= c2_lm) {
                i0_next = vp_cand.x;
                c2_next = vp_cand.y;
            }

            if (c2_next <= chi2_cur) {
                let dr = abs(r_cand - r_cur);
                let di = abs(i0_next - i0_cur);
                i0_cur = i0_next;
                r_cur = r_cand;
                chi2_cur = c2_next;
                lambda = max(lambda * 0.25, 1.0e-7);
                accepted = true;
                if (dr < 1.0e-7 && di < 1.0e-7) {
                    break;
                }
                break;
            }
            lambda = min(lambda * 8.0, 1.0e6);
        }
        if (!accepted) {
            break;
        }
    }

    // Exact 1D Variable-Projection Newton score polish (`phi(r) = sum_i w_i u_i e_i (y_i - i0(r) e_i) = 0`)
    // evaluated in residual form with Kahan summation to reach full `~1e-7` f32 root precision!
    for (var p_it: u32 = 0u; p_it < 8u; p_it = p_it + 1u) {
        if (r_cur <= 0.0 || r_cur >= r_tilde_max) {
            break;
        }
        var s1: f32 = 0.0;
        var s1_c: f32 = 0.0;
        var s2: f32 = 0.0;
        var s2_c: f32 = 0.0;
        var t2: f32 = 0.0;
        var q2: f32 = 0.0;
        for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
            let pt = points[pt_off + i];
            let e = exp(-r_cur * pt.u);
            let we = pt.w * e;
            let we2 = we * e;

            let v1 = (we * (*y_arr)[i]) - s1_c;
            let n1 = s1 + v1;
            s1_c = (n1 - s1) - v1;
            s1 = n1;

            let v2 = we2 - s2_c;
            let n2 = s2 + v2;
            s2_c = (n2 - s2) - v2;
            s2 = n2;

            t2 = t2 + we2 * pt.u;
            q2 = q2 + we2 * pt.u * pt.u;
        }
        if (s2 <= 1.0e-28) {
            break;
        }
        let i0_hat = s1 / s2;
        var phi: f32 = 0.0;
        var phi_c: f32 = 0.0;
        var q_res: f32 = 0.0;
        for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
            let pt = points[pt_off + i];
            let e = exp(-r_cur * pt.u);
            let res_i = (*y_arr)[i] - i0_hat * e;
            let term = (pt.w * pt.u * e * res_i) - phi_c;
            let nxt = phi + term;
            phi_c = (nxt - phi) - term;
            phi = nxt;
            q_res = q_res + pt.w * pt.u * pt.u * e * res_i;
        }
        let dphi = i0_hat * (q2 - (t2 * t2) / s2) - q_res + (phi * t2) / s2;
        // At a minimum of chi2_vp(r), d^2 chi2_vp / dr^2 = 2 * i0_hat * dphi > 0!
        if (abs(dphi) <= 1.0e-20 || i0_hat * dphi <= 0.0) {
            break;
        }
        let dr = clamp(-phi / dphi, -0.05, 0.05);
        r_cur = clamp(r_cur + dr, 0.0, r_tilde_max);
        if (abs(dr) < 5.0e-8) {
            break;
        }
    }
    let vp_fin = varpro_two_param(pt_off, n_pts, y_arr, r_cur);
    return vec3<f32>(vp_fin.x, r_cur, vp_fin.y);
}

fn kahan_add(acc: vec2<f32>, val: f32) -> vec2<f32> {
    let v = val - acc.y;
    let nxt = acc.x + v;
    let c = (nxt - acc.x) - v;
    return vec2<f32>(nxt, c);
}

// Returns vec3<f32>(i0_t, c_t, chi2) for 3-parameter VarPro at fixed `r_t`
// using weighted mean centering (`de_i = e_i - e_bar`, `dy_i = y_i - y_bar`) and Kahan summation
// to eliminate catastrophic cancellation in `2x2` normal equations on non-FMA `f32` backends.
fn varpro_three_param(
    pt_off: u32,
    n_pts: u32,
    y_arr: ptr<function, array<f32, 64>>,
    r_t: f32,
) -> vec3<f32> {
    var sw_acc = vec2<f32>(0.0, 0.0);
    var swe_acc = vec2<f32>(0.0, 0.0);
    var swy_acc = vec2<f32>(0.0, 0.0);

    for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
        let pt = points[pt_off + i];
        let e = exp(-r_t * pt.u);
        let yi = (*y_arr)[i];
        sw_acc = kahan_add(sw_acc, pt.w);
        swe_acc = kahan_add(swe_acc, pt.w * e);
        swy_acc = kahan_add(swy_acc, pt.w * yi);
    }
    let s_w = sw_acc.x;
    if (s_w <= 1.0e-28) {
        return vec3<f32>(0.0, 0.0, 1.0e38);
    }
    let e_bar = swe_acc.x / s_w;
    let y_bar = swy_acc.x / s_w;

    var sde2_acc = vec2<f32>(0.0, 0.0);
    var sdey_acc = vec2<f32>(0.0, 0.0);
    for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
        let pt = points[pt_off + i];
        let de = exp(-r_t * pt.u) - e_bar;
        let dy = (*y_arr)[i] - y_bar;
        sde2_acc = kahan_add(sde2_acc, pt.w * de * de);
        sdey_acc = kahan_add(sdey_acc, pt.w * de * dy);
    }

    var i0_t: f32 = 0.0;
    var c_t: f32 = y_bar;
    if (sde2_acc.x > 1.0e-24) {
        i0_t = sdey_acc.x / sde2_acc.x;
        c_t = y_bar - i0_t * e_bar;
    }

    var chi2_acc = vec2<f32>(0.0, 0.0);
    for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
        let pt = points[pt_off + i];
        let de = exp(-r_t * pt.u) - e_bar;
        let dy = (*y_arr)[i] - y_bar;
        let res = dy - i0_t * de;
        chi2_acc = kahan_add(chi2_acc, pt.w * res * res);
    }
    return vec3<f32>(i0_t, c_t, chi2_acc.x);
}

// Returns vec4<f32>(i0_t, r_t, c_t, chi2) matching `fit_three_param_normalized` on CPU.
fn fit_three_param(
    pt_off: u32,
    n_pts: u32,
    y_arr: ptr<function, array<f32, 64>>,
    r_tilde_max: f32,
) -> vec4<f32> {
    let grid_hi = max(min(r_tilde_max, 25.0), 0.5);
    var best_r: f32 = 0.5;
    let vp0 = varpro_three_param(pt_off, n_pts, y_arr, best_r);
    var best_i0: f32 = vp0.x;
    var best_c: f32 = vp0.y;
    var best_chi2: f32 = vp0.z;

    for (var g: u32 = 1u; g <= 40u; g = g + 1u) {
        let frac = f32(g) * (1.0 / 40.0);
        let r_cand = 0.02 + (grid_hi - 0.02) * frac * frac;
        let vpc = varpro_three_param(pt_off, n_pts, y_arr, r_cand);
        if (vpc.z < best_chi2) {
            best_chi2 = vpc.z;
            best_i0 = vpc.x;
            best_c = vpc.y;
            best_r = r_cand;
        }
    }

    var i0_cur = best_i0;
    var r_cur = best_r;
    var c_cur = best_c;
    var chi2_cur = best_chi2;
    var lambda: f32 = 1.0e-4;

    for (var iter: u32 = 0u; iter < 30u; iter = iter + 1u) {
        var a00: f32 = 0.0;
        var a01: f32 = 0.0;
        var a02: f32 = 0.0;
        var a11: f32 = 0.0;
        var a12: f32 = 0.0;
        var a22: f32 = 0.0;
        var g0: f32 = 0.0;
        var g1: f32 = 0.0;
        var g2: f32 = 0.0;

        for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
            let pt = points[pt_off + i];
            let e = exp(-r_cur * pt.u);
            let res = (*y_arr)[i] - (i0_cur * e + c_cur);
            let j0 = e;
            let j1 = -i0_cur * pt.u * e;
            let wi = pt.w;
            a00 = a00 + wi * j0 * j0;
            a01 = a01 + wi * j0 * j1;
            a02 = a02 + wi * j0;
            a11 = a11 + wi * j1 * j1;
            a12 = a12 + wi * j1;
            a22 = a22 + wi;
            g0 = g0 + wi * j0 * res;
            g1 = g1 + wi * j1 * res;
            g2 = g2 + wi * res;
        }

        var accepted = false;
        for (var sub: u32 = 0u; sub < 6u; sub = sub + 1u) {
            let d00 = a00 + lambda * max(a00, 1.0e-6);
            let d11 = a11 + lambda * max(a11, 1.0e-6);
            let d22 = a22 + lambda * max(a22, 1.0e-6);

            let c00 = d11 * d22 - a12 * a12;
            let c01 = a12 * a02 - a01 * d22;
            let c02 = a01 * a12 - d11 * a02;
            let c11 = d00 * d22 - a02 * a02;
            let c12 = a01 * a02 - d00 * a12;
            let c22 = d00 * d11 - a01 * a01;
            let det = d00 * c00 + a01 * c01 + a02 * c02;

            if (abs(det) <= 1.0e-24) {
                lambda = lambda * 10.0;
                continue;
            }
            let inv_det = 1.0 / det;
            let s0 = (c00 * g0 + c01 * g1 + c02 * g2) * inv_det;
            let s1 = (c01 * g0 + c11 * g1 + c12 * g2) * inv_det;
            let s2 = (c02 * g0 + c12 * g1 + c22 * g2) * inv_det;

            let r_cand = clamp(r_cur + s1, 0.0, r_tilde_max);
            let i0_lm = i0_cur + s0;
            let c_lm = c_cur + s2;
            var c2_lm: f32 = 0.0;
            for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
                let pt = points[pt_off + i];
                let d = (*y_arr)[i] - (i0_lm * exp(-r_cand * pt.u) + c_lm);
                c2_lm = c2_lm + pt.w * d * d;
            }
            let vpc = varpro_three_param(pt_off, n_pts, y_arr, r_cand);
            var i0_n = i0_lm;
            var c_n = c_lm;
            var c2_n = c2_lm;
            if (vpc.z <= c2_lm) {
                i0_n = vpc.x;
                c_n = vpc.y;
                c2_n = vpc.z;
            }

            if (c2_n <= chi2_cur) {
                let dr = abs(r_cand - r_cur);
                i0_cur = i0_n;
                r_cur = r_cand;
                c_cur = c_n;
                chi2_cur = c2_n;
                lambda = max(lambda * 0.3, 1.0e-7);
                accepted = true;
                if (dr < 1.0e-7) {
                    break;
                }
                break;
            }
            lambda = min(lambda * 8.0, 1.0e6);
        }
        if (!accepted) {
            break;
        }
    }

    // Exact analytical 1D Variable-Projection Newton score polish on centered variables:
    // Eliminating `c(r) = y_bar - i0(r) * e_bar` reduces 3-param VarPro to `min_r sum w_i (dy_i - i0(r) de_i)^2`
    // where `de_i = e_i - e_bar`, `g_i = -dde_i/dr = u_i e_i - ue_bar`, `h_i = d^2(de_i)/dr^2 = u_i^2 e_i - u2e_bar`,
    // and score `phi(r) = sum_i w_i g_i (dy_i - i0(r) de_i) = 0` with analytical derivative `dphi/dr`.
    var sw_acc = vec2<f32>(0.0, 0.0);
    var swy_acc = vec2<f32>(0.0, 0.0);
    for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
        let pt = points[pt_off + i];
        sw_acc = kahan_add(sw_acc, pt.w);
        swy_acc = kahan_add(swy_acc, pt.w * (*y_arr)[i]);
    }
    if (sw_acc.x > 1.0e-28) {
        let s_w = sw_acc.x;
        let y_bar = swy_acc.x / s_w;
        for (var p_it: u32 = 0u; p_it < 8u; p_it = p_it + 1u) {
            if (r_cur <= 0.0 || r_cur >= r_tilde_max) {
                break;
            }
            var swe_acc = vec2<f32>(0.0, 0.0);
            var sue_acc = vec2<f32>(0.0, 0.0);
            var su2e_acc = vec2<f32>(0.0, 0.0);
            for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
                let pt = points[pt_off + i];
                let e = exp(-r_cur * pt.u);
                let we = pt.w * e;
                let wue = we * pt.u;
                swe_acc = kahan_add(swe_acc, we);
                sue_acc = kahan_add(sue_acc, wue);
                su2e_acc = kahan_add(su2e_acc, wue * pt.u);
            }
            let e_bar = swe_acc.x / s_w;
            let ue_bar = sue_acc.x / s_w;
            let u2e_bar = su2e_acc.x / s_w;

            var s1_acc = vec2<f32>(0.0, 0.0);
            var s2_acc = vec2<f32>(0.0, 0.0);
            var t2_acc = vec2<f32>(0.0, 0.0);
            var q2_acc = vec2<f32>(0.0, 0.0);
            for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
                let pt = points[pt_off + i];
                let e = exp(-r_cur * pt.u);
                let de = e - e_bar;
                let dy = (*y_arr)[i] - y_bar;
                let g = pt.u * e - ue_bar;
                s1_acc = kahan_add(s1_acc, pt.w * de * dy);
                s2_acc = kahan_add(s2_acc, pt.w * de * de);
                t2_acc = kahan_add(t2_acc, pt.w * de * g);
                q2_acc = kahan_add(q2_acc, pt.w * g * g);
            }
            let s2 = s2_acc.x;
            if (s2 <= 1.0e-24) {
                break;
            }
            let i0_hat = s1_acc.x / s2;
            let t2 = t2_acc.x;
            let q2 = q2_acc.x;

            var phi_acc = vec2<f32>(0.0, 0.0);
            var qres_acc = vec2<f32>(0.0, 0.0);
            for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
                let pt = points[pt_off + i];
                let e = exp(-r_cur * pt.u);
                let de = e - e_bar;
                let dy = (*y_arr)[i] - y_bar;
                let res_i = dy - i0_hat * de;
                let g = pt.u * e - ue_bar;
                let h = pt.u * pt.u * e - u2e_bar;
                phi_acc = kahan_add(phi_acc, pt.w * g * res_i);
                qres_acc = kahan_add(qres_acc, pt.w * h * res_i);
            }
            let phi = phi_acc.x;
            let q_res = qres_acc.x;
            let dphi = i0_hat * (q2 - (t2 * t2) / s2) - q_res + (phi * t2) / s2;
            if (abs(dphi) <= 1.0e-20 || i0_hat * dphi <= 0.0) {
                break;
            }
            let dr = clamp(-phi / dphi, -0.05, 0.05);
            r_cur = clamp(r_cur + dr, 0.0, r_tilde_max);
            if (abs(dr) < 5.0e-8) {
                break;
            }
        }
    }
    let vp3_fin = varpro_three_param(pt_off, n_pts, y_arr, r_cur);
    return vec4<f32>(vp3_fin.x, r_cur, vp3_fin.y, vp3_fin.z);
}

fn info_criterion_gpu(chi2: f32, k: u32, n: u32, gate: u32) -> f32 {
    let kf = f32(k);
    let nf = f32(max(n, 1u));
    if (gate == 2u) {
        if (n <= k + 1u) {
            return 1.0e38;
        }
        let denom_i = i32(n) - i32(k) - 1;
        return chi2 + 2.0 * kf + (2.0 * kf * (kf + 1.0)) / f32(denom_i);
    }
    if (gate == 3u) {
        return chi2 + kf * log(nf);
    }
    return chi2 + 2.0 * kf;
}

@compute @workgroup_size(64)
fn fit_point_estimates(@builtin(global_invocation_id) gid: vec3<u32>) {
    let curve_idx = gid.x;
    if (curve_idx >= params.n_curves) {
        return;
    }
    let cmeta = metas[curve_idx];
    let n_pts = min(cmeta.n_pts, MAX_CURVE_PTS);
    var y_obs: array<f32, 64>;
    for (var i: u32 = 0u; i < n_pts; i = i + 1u) {
        y_obs[i] = points[cmeta.pt_offset + i].y;
    }

    let f2 = fit_two_param(cmeta.pt_offset, n_pts, &y_obs, cmeta.r_tilde_max);
    var out_i0 = f2.x;
    var out_r = f2.y;
    var out_c: f32 = 0.0;
    var out_chi2 = f2.z;
    var use_3p: u32 = 0u;

    if (params.offset_gate == 4u && n_pts >= 3u) {
        let f3 = fit_three_param(cmeta.pt_offset, n_pts, &y_obs, cmeta.r_tilde_max);
        out_i0 = f3.x;
        out_r = f3.y;
        out_c = f3.z;
        out_chi2 = f3.w;
        use_3p = 1u;
    } else if (params.offset_gate >= 1u && params.offset_gate <= 3u && n_pts >= 4u && !(params.offset_gate == 2u && n_pts <= 4u)) {
        let f3 = fit_three_param(cmeta.pt_offset, n_pts, &y_obs, cmeta.r_tilde_max);
        let ic2 = info_criterion_gpu(f2.z, 2u, n_pts, params.offset_gate);
        let ic3 = info_criterion_gpu(f3.w, 3u, n_pts, params.offset_gate);
        if (ic3 < ic2) {
            out_i0 = f3.x;
            out_r = f3.y;
            out_c = f3.z;
            out_chi2 = f3.w;
            use_3p = 1u;
        }
    }

    point_fits[curve_idx] = PointFitGpu(out_i0, out_r, out_c, out_chi2, use_3p, 0u, 0u, 0u);
}

@compute @workgroup_size(64)
fn fit_mc_replicates(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_chunk_fits = params.n_curves * params.rep_chunk;
    let tid = gid.x;
    if (tid >= total_chunk_fits) {
        return;
    }
    let curve_idx = tid / params.rep_chunk;
    let local_rep = tid % params.rep_chunk;
    let global_rep = params.rep_offset + local_rep;
    if (global_rep >= params.mc_reps_total) {
        return;
    }

    let cmeta = metas[curve_idx];
    let n_pts = min(cmeta.n_pts, MAX_CURVE_PTS);
    let pf = point_fits[curve_idx];

    var y_syn: array<f32, 64>;
    let stream_a = (cmeta.seq_id_u32 * 0x9E3779B9u) ^ (curve_idx * 0x85EBCA6Bu);
    var pair_idx: u32 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= n_pts) {
            break;
        }
        let stream_b = (global_rep * 131u) + pair_idx;
        let z = box_muller_pair(params.seed_lo, params.seed_hi, stream_a, stream_b);

        let pt0 = points[cmeta.pt_offset + i];
        let yhat0 = pf.i0_t * exp(-pf.r_t * pt0.u) + pf.c_t;
        y_syn[i] = yhat0 + pt0.s_norm * z.x;

        if (i + 1u < n_pts) {
            let pt1 = points[cmeta.pt_offset + i + 1u];
            let yhat1 = pf.i0_t * exp(-pf.r_t * pt1.u) + pf.c_t;
            y_syn[i + 1u] = yhat1 + pt1.s_norm * z.y;
        }

        i = i + 2u;
        pair_idx = pair_idx + 1u;
    }

    var res_i0: f32 = 0.0;
    var res_r: f32 = 0.0;
    if (pf.use_3p == 1u) {
        let f3 = fit_three_param(cmeta.pt_offset, n_pts, &y_syn, cmeta.r_tilde_max);
        res_i0 = f3.x;
        res_r = f3.y;
    } else {
        let f2 = fit_two_param(cmeta.pt_offset, n_pts, &y_syn, cmeta.r_tilde_max);
        res_i0 = f2.x;
        res_r = f2.y;
    }

    mc_samples[tid] = vec2<f32>(res_i0, res_r);
}
