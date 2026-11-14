// WebGPU (`wgpu` + WGSL) batched per-column / per-peak mono- and bi-exponential
// Stejskal-Tanner Levenberg-Marquardt diffusion fitter (`T14.4`).
//
// Provenance: Clean-room GPU compute shader derived from:
// - Stejskal, E. O. & Tanner, J. E. (J. Chem. Phys. 1965, 42, 288-292).
// - Wu, D. H., Chen, A. D. & Johnson, C. S. Jr. (J. Magn. Reson. A 1995, 115, 260-264).
// - Golub, G. H. & Pereyra, V. (SIAM J. Numer. Anal. 1973, 10, 413-432 — Variable Projection
//   separable nonlinear least squares initialization) + Marquardt (J. Soc. Ind. Appl. Math.
//   1963, 11, 431-441 — damped Gauss-Newton refinement).

const Q_GRID: u32 = 24u;

struct DosyGpuParams {
    n_cols: u32,
    m_len: u32,
    max_components: u32,
    criterion_is_aic: u32,
    max_lm_iters: u32,
    use_known_sigma: u32,
    b_scale: f32,
    u_min: f32,
    u_max: f32,
    min_du_seed: f32,
    min_component_fraction: f32,
    min_d_ratio: f32,
    known_noise_sigma: f32,
    min_signal_floor: f32,
    _pad0: u32,
    _pad1: u32,
}

struct DosyColumnOut {
    selected_k: u32,
    is_bi_resolved: u32,
    mono_iters: u32,
    bi_iters: u32,
    mono_i0: f32,
    mono_d: f32,
    mono_rss: f32,
    mono_bic: f32,
    bi_i0_1: f32,
    bi_d_1: f32,
    bi_i0_2: f32,
    bi_d_2: f32,
    bi_rss: f32,
    bi_bic: f32,
    mono_aic: f32,
    bi_aic: f32,
}

@group(0) @binding(0) var<uniform> params: DosyGpuParams;
@group(0) @binding(1) var<storage, read> b_tilde: array<f32>;
@group(0) @binding(2) var<storage, read> u_grid: array<f32>;
@group(0) @binding(3) var<storage, read> e_basis: array<f32>;
@group(0) @binding(4) var<storage, read> gram_mat: array<f32>;
@group(0) @binding(5) var<storage, read> y_matrix: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_fits: array<DosyColumnOut>;

struct MonoState {
    a: f32,
    u: f32,
    rss: f32,
    iters: u32,
}

struct BiState {
    a1: f32,
    u1: f32,
    a2: f32,
    u2: f32,
    rss: f32,
    iters: u32,
}

fn eval_rss_mono(col_base: u32, m_len: u32, inv_y_scale: f32, a: f32, u: f32) -> f32 {
    let d = exp(u);
    var rss: f32 = 0.0;
    for (var i: u32 = 0u; i < m_len; i = i + 1u) {
        let yt = y_matrix[col_base + i] * inv_y_scale;
        let pred = a * exp(-b_tilde[i] * d);
        let r = pred - yt;
        rss = rss + r * r;
    }
    return rss;
}

fn lm_polish_mono(
    col_base: u32,
    m_len: u32,
    inv_y_scale: f32,
    init_a: f32,
    init_u: f32,
) -> MonoState {
    var a: f32 = max(init_a, 1.0e-6);
    var u: f32 = clamp(init_u, params.u_min, params.u_max);
    var cur_rss: f32 = eval_rss_mono(col_base, m_len, inv_y_scale, a, u);
    var lam: f32 = 1.0e-3;
    var iters_used: u32 = 0u;

    for (var it: u32 = 0u; it < params.max_lm_iters; it = it + 1u) {
        iters_used = it + 1u;
        let d_cur = exp(u);
        var h00: f32 = 0.0;
        var h01: f32 = 0.0;
        var h11: f32 = 0.0;
        var g0: f32 = 0.0;
        var g1: f32 = 0.0;

        for (var i: u32 = 0u; i < m_len; i = i + 1u) {
            let bt = b_tilde[i];
            let yt = y_matrix[col_base + i] * inv_y_scale;
            let bd = bt * d_cur;
            let ek = exp(-bd);
            let j0 = ek;
            let j1 = -a * bd * ek;
            let r = a * ek - yt;
            g0 = g0 + j0 * r;
            g1 = g1 + j1 * r;
            h00 = h00 + j0 * j0;
            h01 = h01 + j0 * j1;
            h11 = h11 + j1 * j1;
        }

        if (max(abs(g0), abs(g1)) < 1.0e-7) {
            break;
        }

        var accepted: bool = false;
        for (var sub: u32 = 0u; sub < 7u; sub = sub + 1u) {
            let d00 = h00 + lam * max(h00, 1.0e-3) + 1.0e-7;
            let d11 = h11 + lam * max(h11, 1.0e-3) + 1.0e-7;
            let det = d00 * d11 - h01 * h01;
            if (det <= 1.0e-18) {
                lam = min(lam * 8.0, 1.0e7);
                continue;
            }
            let step_a = clamp((-g0 * d11 + g1 * h01) / det, -2.0, 2.0);
            let step_u = clamp((-d00 * g1 + h01 * g0) / det, -1.5, 1.5);
            let trial_a = max(a + step_a, 1.0e-7);
            let trial_u = clamp(u + step_u, params.u_min, params.u_max);
            let trial_rss = eval_rss_mono(col_base, m_len, inv_y_scale, trial_a, trial_u);
            if (trial_rss <= cur_rss) {
                let rel_impr = (cur_rss - trial_rss) / (1.0e-12 + cur_rss);
                let max_step = max(abs(trial_a - a), abs(trial_u - u));
                a = trial_a;
                u = trial_u;
                cur_rss = trial_rss;
                lam = max(lam * 0.3, 1.0e-7);
                accepted = true;
                if (rel_impr < 1.0e-7 && max_step < 1.0e-5) {
                    return MonoState(a, u, cur_rss, iters_used);
                }
                break;
            } else {
                lam = min(lam * 5.0, 1.0e7);
            }
        }
        if (!accepted && lam >= 1.0e7) {
            break;
        }
    }
    return MonoState(a, u, cur_rss, iters_used);
}

fn eval_rss_bi(
    col_base: u32,
    m_len: u32,
    inv_y_scale: f32,
    a1: f32,
    u1: f32,
    a2: f32,
    u2: f32,
) -> f32 {
    let d1 = exp(u1);
    let d2 = exp(u2);
    var rss: f32 = 0.0;
    for (var i: u32 = 0u; i < m_len; i = i + 1u) {
        let bt = b_tilde[i];
        let yt = y_matrix[col_base + i] * inv_y_scale;
        let pred = a1 * exp(-bt * d1) + a2 * exp(-bt * d2);
        let r = pred - yt;
        rss = rss + r * r;
    }
    return rss;
}

// Solve 4x4 linear system M * x = rhs via Gaussian elimination with partial pivoting.
fn solve_4x4(m_in: array<f32, 16>, rhs_in: array<f32, 4>, out_x: ptr<function, array<f32, 4>>) -> bool {
    var aug: array<f32, 20>;
    for (var r: u32 = 0u; r < 4u; r = r + 1u) {
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            aug[r * 5u + c] = m_in[r * 4u + c];
        }
        aug[r * 5u + 4u] = rhs_in[r];
    }

    for (var col: u32 = 0u; col < 4u; col = col + 1u) {
        var piv: u32 = col;
        var max_v: f32 = abs(aug[col * 5u + col]);
        for (var r: u32 = col + 1u; r < 4u; r = r + 1u) {
            let av = abs(aug[r * 5u + col]);
            if (av > max_v) {
                max_v = av;
                piv = r;
            }
        }
        if (max_v < 1.0e-15) {
            return false;
        }
        if (piv != col) {
            for (var c: u32 = col; c <= 4u; c = c + 1u) {
                let tmp = aug[col * 5u + c];
                aug[col * 5u + c] = aug[piv * 5u + c];
                aug[piv * 5u + c] = tmp;
            }
        }
        let diag = aug[col * 5u + col];
        for (var r: u32 = col + 1u; r < 4u; r = r + 1u) {
            let fac = aug[r * 5u + col] / diag;
            aug[r * 5u + col] = 0.0;
            for (var c: u32 = col + 1u; c <= 4u; c = c + 1u) {
                aug[r * 5u + c] = aug[r * 5u + c] - fac * aug[col * 5u + c];
            }
        }
    }

    for (var ri: i32 = 3; ri >= 0; ri = ri - 1) {
        let r = u32(ri);
        var s: f32 = aug[r * 5u + 4u];
        for (var c: u32 = r + 1u; c < 4u; c = c + 1u) {
            s = s - aug[r * 5u + c] * (*out_x)[c];
        }
        let diag = aug[r * 5u + r];
        if (abs(diag) < 1.0e-15) {
            return false;
        }
        (*out_x)[r] = s / diag;
    }
    return true;
}

fn varpro_polish_bi(
    col_base: u32,
    m_len: u32,
    inv_y_scale: f32,
    init_a1: f32,
    init_u1: f32,
    init_a2: f32,
    init_u2: f32,
    vp_iters: u32,
) -> vec4<f32> {
    var u1: f32 = clamp(init_u1, params.u_min, params.u_max);
    var u2: f32 = clamp(init_u2, params.u_min, params.u_max);
    var best_a1: f32 = max(init_a1, 1.0e-6);
    var best_a2: f32 = max(init_a2, 1.0e-6);

    // Evaluate initial optimal amplitudes
    var cur_rss: f32 = eval_rss_bi(col_base, m_len, inv_y_scale, best_a1, u1, best_a2, u2);
    {
        let d1 = exp(u1);
        let d2 = exp(u2);
        var g11: f32 = 0.0; var g12: f32 = 0.0; var g22: f32 = 0.0;
        var c1: f32 = 0.0; var c2: f32 = 0.0;
        for (var i: u32 = 0u; i < m_len; i = i + 1u) {
            let bt = b_tilde[i];
            let yt = y_matrix[col_base + i] * inv_y_scale;
            let e1 = exp(-bt * d1);
            let e2 = exp(-bt * d2);
            g11 = g11 + e1 * e1;
            g12 = g12 + e1 * e2;
            g22 = g22 + e2 * e2;
            c1 = c1 + e1 * yt;
            c2 = c2 + e2 * yt;
        }
        let det = g11 * g22 - g12 * g12;
        if (det > 1.0e-12 * g11 * g22) {
            let a1_0 = (g22 * c1 - g12 * c2) / det;
            let a2_0 = (g11 * c2 - g12 * c1) / det;
            if (a1_0 > 1.0e-5 && a2_0 > 1.0e-5) {
                best_a1 = a1_0;
                best_a2 = a2_0;
                cur_rss = eval_rss_bi(col_base, m_len, inv_y_scale, best_a1, u1, best_a2, u2);
            } else {
                return vec4<f32>(best_a1, u1, best_a2, u2);
            }
        }
    }

    var lam: f32 = 1.0e-3;
    for (var it: u32 = 0u; it < vp_iters; it = it + 1u) {
        let d1 = exp(u1);
        let d2 = exp(u2);
        var g11: f32 = 0.0; var g12: f32 = 0.0; var g22: f32 = 0.0;
        var p11: f32 = 0.0; var p21: f32 = 0.0;
        var p12: f32 = 0.0; var p22: f32 = 0.0;
        for (var i: u32 = 0u; i < m_len; i = i + 1u) {
            let bt = b_tilde[i];
            let bd1 = bt * d1;
            let bd2 = bt * d2;
            let e1 = exp(-bd1);
            let e2 = exp(-bd2);
            let phi1 = -best_a1 * bd1 * e1;
            let phi2 = -best_a2 * bd2 * e2;
            g11 = g11 + e1 * e1;
            g12 = g12 + e1 * e2;
            g22 = g22 + e2 * e2;
            p11 = p11 + e1 * phi1;
            p21 = p21 + e2 * phi1;
            p12 = p12 + e1 * phi2;
            p22 = p22 + e2 * phi2;
        }
        let det_g = g11 * g22 - g12 * g12;
        if (det_g <= 1.0e-12 * g11 * g22) {
            break;
        }
        let al1_1 = (g22 * p11 - g12 * p21) / det_g;
        let al1_2 = (g11 * p21 - g12 * p11) / det_g;
        let al2_1 = (g22 * p12 - g12 * p22) / det_g;
        let al2_2 = (g11 * p22 - g12 * p12) / det_g;

        var h00: f32 = 0.0; var h01: f32 = 0.0; var h11: f32 = 0.0;
        var gr0: f32 = 0.0; var gr1: f32 = 0.0;
        for (var i: u32 = 0u; i < m_len; i = i + 1u) {
            let bt = b_tilde[i];
            let yt = y_matrix[col_base + i] * inv_y_scale;
            let bd1 = bt * d1;
            let bd2 = bt * d2;
            let e1 = exp(-bd1);
            let e2 = exp(-bd2);
            let phi1 = -best_a1 * bd1 * e1;
            let phi2 = -best_a2 * bd2 * e2;
            let j0 = phi1 - al1_1 * e1 - al1_2 * e2;
            let j1 = phi2 - al2_1 * e1 - al2_2 * e2;
            let r = (best_a1 * e1 + best_a2 * e2) - yt;
            gr0 = gr0 + j0 * r;
            gr1 = gr1 + j1 * r;
            h00 = h00 + j0 * j0;
            h01 = h01 + j0 * j1;
            h11 = h11 + j1 * j1;
        }
        if (max(abs(gr0), abs(gr1)) < 1.0e-7) {
            break;
        }

        var accepted: bool = false;
        for (var sub: u32 = 0u; sub < 7u; sub = sub + 1u) {
            let hd00 = h00 + lam * max(h00, 1.0e-4) + 1.0e-8;
            let hd11 = h11 + lam * max(h11, 1.0e-4) + 1.0e-8;
            let det_h = hd00 * hd11 - h01 * h01;
            if (det_h <= 1.0e-18) {
                lam = min(lam * 8.0, 1.0e7);
                continue;
            }
            let du1 = clamp((-gr0 * hd11 + gr1 * h01) / det_h, -1.0, 1.0);
            let du2 = clamp((-hd00 * gr1 + h01 * gr0) / det_h, -1.0, 1.0);
            let tu1 = clamp(u1 + du1, params.u_min, params.u_max);
            let tu2 = clamp(u2 + du2, params.u_min, params.u_max);
            let td1 = exp(tu1);
            let td2 = exp(tu2);

            var tg11: f32 = 0.0; var tg12: f32 = 0.0; var tg22: f32 = 0.0;
            var tc1: f32 = 0.0; var tc2: f32 = 0.0;
            for (var i: u32 = 0u; i < m_len; i = i + 1u) {
                let bt = b_tilde[i];
                let yt = y_matrix[col_base + i] * inv_y_scale;
                let te1 = exp(-bt * td1);
                let te2 = exp(-bt * td2);
                tg11 = tg11 + te1 * te1;
                tg12 = tg12 + te1 * te2;
                tg22 = tg22 + te2 * te2;
                tc1 = tc1 + te1 * yt;
                tc2 = tc2 + te2 * yt;
            }
            let tdet = tg11 * tg22 - tg12 * tg12;
            if (tdet > 1.0e-12 * tg11 * tg22) {
                let ta1 = (tg22 * tc1 - tg12 * tc2) / tdet;
                let ta2 = (tg11 * tc2 - tg12 * tc1) / tdet;
                if (ta1 > 1.0e-5 && ta2 > 1.0e-5) {
                    let trss = eval_rss_bi(col_base, m_len, inv_y_scale, ta1, tu1, ta2, tu2);
                    if (trss <= cur_rss) {
                        let rel = (cur_rss - trss) / (1.0e-12 + cur_rss);
                        let mdu = max(abs(tu1 - u1), abs(tu2 - u2));
                        u1 = tu1;
                        u2 = tu2;
                        best_a1 = ta1;
                        best_a2 = ta2;
                        cur_rss = trss;
                        lam = max(lam * 0.3, 1.0e-7);
                        accepted = true;
                        if (rel < 1.0e-7 && mdu < 1.0e-5) {
                            return vec4<f32>(best_a1, u1, best_a2, u2);
                        }
                        break;
                    }
                }
            }
            lam = min(lam * 5.0, 1.0e7);
        }
        if (!accepted && lam >= 1.0e7) {
            break;
        }
    }
    return vec4<f32>(best_a1, u1, best_a2, u2);
}

fn lm_polish_bi(
    col_base: u32,
    m_len: u32,
    inv_y_scale: f32,
    init_a1: f32,
    init_u1: f32,
    init_a2: f32,
    init_u2: f32,
) -> BiState {
    let vp_iters = clamp(params.max_lm_iters / 2u, 6u, 25u);
    let vp = varpro_polish_bi(col_base, m_len, inv_y_scale, init_a1, init_u1, init_a2, init_u2, vp_iters);
    var a1: f32 = max(vp.x, 1.0e-6);
    var u1: f32 = clamp(vp.y, params.u_min, params.u_max);
    var a2: f32 = max(vp.z, 1.0e-6);
    var u2: f32 = clamp(vp.w, params.u_min, params.u_max);
    var cur_rss: f32 = eval_rss_bi(col_base, m_len, inv_y_scale, a1, u1, a2, u2);
    var lam: f32 = 1.0e-3;
    var iters_used: u32 = 0u;

    for (var it: u32 = 0u; it < params.max_lm_iters; it = it + 1u) {
        iters_used = it + 1u;
        let d1 = exp(u1);
        let d2 = exp(u2);

        var h: array<f32, 16>;
        var g: array<f32, 4>;
        for (var z: u32 = 0u; z < 16u; z = z + 1u) {
            h[z] = 0.0;
        }
        for (var z: u32 = 0u; z < 4u; z = z + 1u) {
            g[z] = 0.0;
        }

        for (var i: u32 = 0u; i < m_len; i = i + 1u) {
            let bt = b_tilde[i];
            let yt = y_matrix[col_base + i] * inv_y_scale;
            let bd1 = bt * d1;
            let bd2 = bt * d2;
            let e1 = exp(-bd1);
            let e2 = exp(-bd2);
            var j: array<f32, 4>;
            j[0] = e1;
            j[1] = -a1 * bd1 * e1;
            j[2] = e2;
            j[3] = -a2 * bd2 * e2;
            let r = (a1 * e1 + a2 * e2) - yt;
            for (var p: u32 = 0u; p < 4u; p = p + 1u) {
                g[p] = g[p] + j[p] * r;
                for (var q: u32 = 0u; q < 4u; q = q + 1u) {
                    h[p * 4u + q] = h[p * 4u + q] + j[p] * j[q];
                }
            }
        }

        let g_max = max(max(abs(g[0]), abs(g[1])), max(abs(g[2]), abs(g[3])));
        if (g_max < 1.0e-7) {
            break;
        }

        var accepted: bool = false;
        for (var sub: u32 = 0u; sub < 7u; sub = sub + 1u) {
            var hd = h;
            var rhs: array<f32, 4>;
            for (var p: u32 = 0u; p < 4u; p = p + 1u) {
                let diag = max(h[p * 4u + p], 1.0e-3);
                hd[p * 4u + p] = h[p * 4u + p] + lam * diag + 1.0e-7;
                rhs[p] = -g[p];
            }
            var step: array<f32, 4>;
            step[0] = 0.0;
            step[1] = 0.0;
            step[2] = 0.0;
            step[3] = 0.0;
            if (!solve_4x4(hd, rhs, &step)) {
                lam = min(lam * 8.0, 1.0e7);
                continue;
            }
            let t_a1 = max(a1 + clamp(step[0], -2.0, 2.0), 1.0e-7);
            let t_u1 = clamp(u1 + clamp(step[1], -1.5, 1.5), params.u_min, params.u_max);
            let t_a2 = max(a2 + clamp(step[2], -2.0, 2.0), 1.0e-7);
            let t_u2 = clamp(u2 + clamp(step[3], -1.5, 1.5), params.u_min, params.u_max);

            let t_rss = eval_rss_bi(col_base, m_len, inv_y_scale, t_a1, t_u1, t_a2, t_u2);
            if (t_rss <= cur_rss) {
                let rel_impr = (cur_rss - t_rss) / (1.0e-12 + cur_rss);
                let ms = max(
                    max(abs(t_a1 - a1), abs(t_u1 - u1)),
                    max(abs(t_a2 - a2), abs(t_u2 - u2)),
                );
                a1 = t_a1;
                u1 = t_u1;
                a2 = t_a2;
                u2 = t_u2;
                cur_rss = t_rss;
                lam = max(lam * 0.3, 1.0e-7);
                accepted = true;
                if (rel_impr < 1.0e-7 && ms < 1.0e-5) {
                    return BiState(a1, u1, a2, u2, cur_rss, iters_used);
                }
                break;
            } else {
                lam = min(lam * 5.0, 1.0e7);
            }
        }
        if (!accepted && lam >= 1.0e7) {
            break;
        }
    }

    return BiState(a1, u1, a2, u2, cur_rss, iters_used);
}

fn compute_ic(rss: f32, m_len: u32, k_comp: u32, y_scale: f32) -> vec2<f32> {
    let p = f32(2u * k_comp);
    let mf = f32(m_len);
    let extra_k = f32(max(k_comp, 1u) - 1u);
    let look_elsewhere = 2.0 * extra_k * log(6.0);
    if (params.use_known_sigma != 0u && params.known_noise_sigma > 0.0) {
        let chi2 = rss / (params.known_noise_sigma * params.known_noise_sigma);
        return vec2<f32>(chi2 + 2.0 * p + look_elsewhere, chi2 + p * log(mf) + look_elsewhere);
    }
    let sig_num_sq = (1.0e-6 * y_scale) * (1.0e-6 * y_scale);
    let s2 = max(rss / mf, sig_num_sq);
    let ll = mf * log(s2);
    return vec2<f32>(ll + 2.0 * p + look_elsewhere, ll + p * log(mf) + look_elsewhere);
}

@compute @workgroup_size(64)
fn fit_dosy_columns(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= params.n_cols) {
        return;
    }

    let m_len = params.m_len;
    let col_base = col * m_len;

    var y_scale: f32 = 0.0;
    for (var i: u32 = 0u; i < m_len; i = i + 1u) {
        y_scale = max(y_scale, abs(y_matrix[col_base + i]));
    }

    if (y_scale <= params.min_signal_floor || m_len < 3u) {
        out_fits[col] = DosyColumnOut(
            0u, 0u, 0u, 0u,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        );
        return;
    }

    let inv_y_scale = 1.0 / y_scale;
    var y_norm2: f32 = 0.0;
    for (var i: u32 = 0u; i < m_len; i = i + 1u) {
        let yt = y_matrix[col_base + i] * inv_y_scale;
        y_norm2 = y_norm2 + yt * yt;
    }

    // VarPro cross-correlation c_vec[q] = sum_m E_{q,m} * y_tilde_m
    var c_vec: array<f32, 24>;
    for (var q: u32 = 0u; q < Q_GRID; q = q + 1u) {
        var cq: f32 = 0.0;
        let eb = q * m_len;
        for (var i: u32 = 0u; i < m_len; i = i + 1u) {
            cq = cq + e_basis[eb + i] * (y_matrix[col_base + i] * inv_y_scale);
        }
        c_vec[q] = cq;
    }

    // ---- 1. Mono-exponential (K = 1) ----
    var best_q1_rss: f32 = 1.0e30;
    var seed_b_a: f32 = 1.0;
    var seed_b_u: f32 = 0.5 * (params.u_min + params.u_max);
    for (var q: u32 = 0u; q < Q_GRID; q = q + 1u) {
        let g_qq = gram_mat[q * Q_GRID + q];
        if (g_qq > 1.0e-15) {
            let aq = max(c_vec[q] / g_qq, 1.0e-5);
            let rss_q = max(y_norm2 - 2.0 * aq * c_vec[q] + aq * aq * g_qq, 0.0);
            if (rss_q < best_q1_rss) {
                best_q1_rss = rss_q;
                seed_b_a = aq;
                seed_b_u = u_grid[q];
            }
        }
    }

    var mono_best = lm_polish_mono(col_base, m_len, inv_y_scale, seed_b_a, seed_b_u);

    // Weighted log-linear regression seed A
    var sw: f32 = 0.0;
    var swx: f32 = 0.0;
    var swy: f32 = 0.0;
    var swxx: f32 = 0.0;
    var swxy: f32 = 0.0;
    var cnt_pos: u32 = 0u;
    for (var i: u32 = 0u; i < m_len; i = i + 1u) {
        let yt = y_matrix[col_base + i] * inv_y_scale;
        if (yt > 0.05) {
            let w = yt * yt;
            let x = b_tilde[i];
            let ly = log(yt);
            sw = sw + w;
            swx = swx + w * x;
            swy = swy + w * ly;
            swxx = swxx + w * x * x;
            swxy = swxy + w * x * ly;
            cnt_pos = cnt_pos + 1u;
        }
    }
    if (cnt_pos >= 2u) {
        let denom = sw * swxx - swx * swx;
        if (abs(denom) > 1.0e-14) {
            let slope = (sw * swxy - swx * swy) / denom;
            let intercept = (swy - slope * swx) / sw;
            let d_est = max(-slope, 1.0e-6);
            let a_est = clamp(exp(intercept), 0.05, 2.5);
            let u_est = clamp(log(d_est), params.u_min, params.u_max);
            let cand_a = lm_polish_mono(col_base, m_len, inv_y_scale, a_est, u_est);
            if (cand_a.rss < mono_best.rss) {
                mono_best = cand_a;
            }
        }
    }

    let mono_i0 = mono_best.a * y_scale;
    let mono_d = exp(mono_best.u) / params.b_scale;
    let mono_rss = mono_best.rss * y_scale * y_scale;
    let ic1 = compute_ic(mono_rss, m_len, 1u, y_scale);
    let mono_aic = ic1.x;
    let mono_bic = ic1.y;

    // ---- 2. Bi-exponential (K = 2) ----
    var selected_k: u32 = 1u;
    var is_bi_resolved: u32 = 0u;
    var bi_i0_1: f32 = mono_i0;
    var bi_d_1: f32 = mono_d;
    var bi_i0_2: f32 = 0.0;
    var bi_d_2: f32 = mono_d;
    var bi_rss: f32 = mono_rss;
    var bi_aic: f32 = mono_aic + 4.0;
    var bi_bic: f32 = mono_bic + 4.0 * log(f32(m_len));
    var bi_iters: u32 = 0u;

    if (params.max_components >= 2u && m_len >= 5u) {
        // Track top 3 positive 2x2 VarPro pairs
        var top_rss = array<f32, 3>(1.0e30, 1.0e30, 1.0e30);
        var top_a1 = array<f32, 3>(0.5, 0.5, 0.5);
        var top_a2 = array<f32, 3>(0.5, 0.5, 0.5);
        var top_u1 = array<f32, 3>(params.u_min, params.u_min, params.u_min);
        var top_u2 = array<f32, 3>(params.u_max, params.u_max, params.u_max);
        var n_found: u32 = 0u;

        for (var p: u32 = 0u; p < Q_GRID; p = p + 1u) {
            for (var q: u32 = p + 1u; q < Q_GRID; q = q + 1u) {
                if (u_grid[q] - u_grid[p] < params.min_du_seed) {
                    continue;
                }
                let g_pp = gram_mat[p * Q_GRID + p];
                let g_pq = gram_mat[p * Q_GRID + q];
                let g_qq = gram_mat[q * Q_GRID + q];
                let det = g_pp * g_qq - g_pq * g_pq;
                if (det <= 1.0e-10 * g_pp * g_qq) {
                    continue;
                }
                let cp = c_vec[p];
                let cq = c_vec[q];
                let a1 = (g_qq * cp - g_pq * cq) / det;
                let a2 = (g_pp * cq - g_pq * cp) / det;
                if (a1 > 1.0e-3 && a2 > 1.0e-3) {
                    let rss_pq = max(
                        y_norm2
                            - 2.0 * (a1 * cp + a2 * cq)
                            + a1 * a1 * g_pp
                            + 2.0 * a1 * a2 * g_pq
                            + a2 * a2 * g_qq,
                        0.0,
                    );
                    if (rss_pq < top_rss[2]) {
                        top_rss[2] = rss_pq;
                        top_a1[2] = a1;
                        top_a2[2] = a2;
                        top_u1[2] = u_grid[p];
                        top_u2[2] = u_grid[q];
                        if (top_rss[2] < top_rss[1]) {
                            let tr = top_rss[1]; top_rss[1] = top_rss[2]; top_rss[2] = tr;
                            let ta1 = top_a1[1]; top_a1[1] = top_a1[2]; top_a1[2] = ta1;
                            let ta2 = top_a2[1]; top_a2[1] = top_a2[2]; top_a2[2] = ta2;
                            let tu1 = top_u1[1]; top_u1[1] = top_u1[2]; top_u1[2] = tu1;
                            let tu2 = top_u2[1]; top_u2[1] = top_u2[2]; top_u2[2] = tu2;
                        }
                        if (top_rss[1] < top_rss[0]) {
                            let tr = top_rss[0]; top_rss[0] = top_rss[1]; top_rss[1] = tr;
                            let ta1 = top_a1[0]; top_a1[0] = top_a1[1]; top_a1[1] = ta1;
                            let ta2 = top_a2[0]; top_a2[0] = top_a2[1]; top_a2[1] = ta2;
                            let tu1 = top_u1[0]; top_u1[0] = top_u1[1]; top_u1[1] = tu1;
                            let tu2 = top_u2[0]; top_u2[0] = top_u2[1]; top_u2[1] = tu2;
                        }
                        n_found = min(n_found + 1u, 3u);
                    }
                }
            }
        }

        if (n_found == 0u) {
            let mid = 0.5 * (params.u_min + params.u_max);
            top_a1[0] = 0.5;
            top_a2[0] = 0.5;
            top_u1[0] = max(mid - 0.6, params.u_min);
            top_u2[0] = min(mid + 0.6, params.u_max);
            n_found = 1u;
        }

        var bi_best = lm_polish_bi(
            col_base,
            m_len,
            inv_y_scale,
            top_a1[0],
            top_u1[0],
            top_a2[0],
            top_u2[0],
        );
        for (var s_idx: u32 = 1u; s_idx < n_found; s_idx = s_idx + 1u) {
            let cand = lm_polish_bi(
                col_base,
                m_len,
                inv_y_scale,
                top_a1[s_idx],
                top_u1[s_idx],
                top_a2[s_idx],
                top_u2[s_idx],
            );
            if (cand.rss < bi_best.rss) {
                bi_best = cand;
            }
        }

        // Sort components by ascending u1 <= u2
        var s_a1 = bi_best.a1;
        var s_u1 = bi_best.u1;
        var s_a2 = bi_best.a2;
        var s_u2 = bi_best.u2;
        if (s_u2 < s_u1) {
            let ta = s_a1; s_a1 = s_a2; s_a2 = ta;
            let tu = s_u1; s_u1 = s_u2; s_u2 = tu;
        }

        bi_i0_1 = s_a1 * y_scale;
        bi_d_1 = exp(s_u1) / params.b_scale;
        bi_i0_2 = s_a2 * y_scale;
        bi_d_2 = exp(s_u2) / params.b_scale;
        bi_rss = bi_best.rss * y_scale * y_scale;
        bi_iters = bi_best.iters;

        let ic2 = compute_ic(bi_rss, m_len, 2u, y_scale);
        bi_aic = ic2.x;
        bi_bic = ic2.y;

        let amp_sum = max(bi_i0_1 + bi_i0_2, 1.0e-30);
        let min_frac = min(bi_i0_1, bi_i0_2) / amp_sum;
        let d_ratio = bi_d_2 / max(bi_d_1, 1.0e-30);
        let dof2 = f32(max(m_len - 4u, 1u));
        let sigma_hat = select(
            sqrt(bi_rss / dof2),
            params.known_noise_sigma,
            params.use_known_sigma != 0u && params.known_noise_sigma > 0.0,
        );
        var e1_norm2: f32 = 0.0;
        var e2_norm2: f32 = 0.0;
        let td1 = bi_d_1 * params.b_scale;
        let td2 = bi_d_2 * params.b_scale;
        for (var i: u32 = 0u; i < m_len; i = i + 1u) {
            let v1 = bi_i0_1 * exp(-b_tilde[i] * td1);
            let v2 = bi_i0_2 * exp(-b_tilde[i] * td2);
            e1_norm2 = e1_norm2 + v1 * v1;
            e2_norm2 = e2_norm2 + v2 * v2;
        }
        let snr_ok = sqrt(e1_norm2) >= 4.0 * sigma_hat && sqrt(e2_norm2) >= 4.0 * sigma_hat;
        if (min_frac >= params.min_component_fraction && d_ratio >= params.min_d_ratio && snr_ok) {
            is_bi_resolved = 1u;
            let sc1 = select(mono_bic, mono_aic, params.criterion_is_aic != 0u);
            let sc2 = select(bi_bic, bi_aic, params.criterion_is_aic != 0u);
            if (sc2 < sc1) {
                selected_k = 2u;
            }
        }
    }

    out_fits[col] = DosyColumnOut(
        selected_k,
        is_bi_resolved,
        mono_best.iters,
        bi_iters,
        mono_i0,
        mono_d,
        mono_rss,
        mono_bic,
        bi_i0_1,
        bi_d_1,
        bi_i0_2,
        bi_d_2,
        bi_rss,
        bi_bic,
        mono_aic,
        bi_aic,
    );
}
