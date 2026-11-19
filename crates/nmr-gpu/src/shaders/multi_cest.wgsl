// Multi-frequency `DANTE-CEST` & cosine-modulated `cos-CEST` dual-sideband homogeneous
// `7x7` Bloch-McConnell WebGPU (`WGSL`) propagator engine (`T22.6`).
//
// Provenance: Clean-room GPU compute implementation strictly from peer-reviewed literature:
// - McConnell, H. M. (J. Chem. Phys. 1958, 28, 430-431).
// - Helgstrand, M., Hard, T. & Allard, P. (J. Biomol. NMR 2000, 18, 47-61).
// - Vallurupalli, P., Bouvignies, G. & Kay, L. E. (J. Am. Chem. Soc. 2012, 134, 8148-8161).
// - Carneiro, M. G., Reddy, J. G., Griesinger, C. & Lee, D. (J. Biomol. NMR 2015, 62, 237-246).
// - Yuwen, T., Kay, L. E. & Bouvignies, G. (J. Phys. Chem. Lett. 2018, 9, 2501-2507).
// - Higham, N. J. (SIAM J. Matrix Anal. Appl. 2005, 26, 1179-1193).

struct MultiCestUniforms {
    n_jobs: u32,
    n_quad: u32,
    _pad0: u32,
    _pad1: u32,
}

struct QuadNodeGpu {
    x_node: f32,
    weight: f32,
    _pad0: f32,
    _pad1: f32,
}

struct MultiCestCurveGpu {
    // 0u = CW-CEST, 1u = DANTE-CEST, 2u = cos-CEST
    scheme: u32,
    n_pulses_or_steps: u32,
    offset_start: u32,
    offset_len: u32,

    p_b: f32,
    k_ex: f32,
    major_ppm: f32,
    dw_ab_ppm: f32,

    r1a: f32,
    r2a: f32,
    r1b: f32,
    r2b: f32,

    larmor_mhz: f32,
    t_ex_or_tau_p: f32,
    b1_hz: f32,
    tau_d_or_mod_freq: f32,

    b1_inhom_frac: f32,
    e0: f32,
    inv_i0: f32,
    _pad0: f32,
}

struct JobMapGpu {
    curve_idx: u32,
    local_offset_idx: u32,
}

@group(0) @binding(0) var<uniform> params: MultiCestUniforms;
@group(0) @binding(1) var<storage, read> quad_nodes: array<QuadNodeGpu>;
@group(0) @binding(2) var<storage, read> curves: array<MultiCestCurveGpu>;
@group(0) @binding(3) var<storage, read> offsets_ppm: array<f32>;
@group(0) @binding(4) var<storage, read> job_map: array<JobMapGpu>;
@group(0) @binding(5) var<storage, read_write> out_intensities: array<f32>;

const TAU_F32: f32 = 6.283185307179586;
const SQRT2_F32: f32 = 1.4142135623730951;
const PADE_B0: f32 = 1.0;
const PADE_B1: f32 = 0.5;
const PADE_B2: f32 = 0.11363636363636363;      // 5 / 44
const PADE_B3: f32 = 0.015151515151515152;     // 1 / 66
const PADE_B4: f32 = 0.0012626262626262627;    // 1 / 792
const PADE_B5: f32 = 0.00006313131313131313;   // 1 / 15840
const PADE_B6: f32 = 0.0000015031265031265032; // 1 / 665280
const MAX_DANTE_PULSES_GPU: u32 = 10000u;
const MAX_COS_STEPS_GPU: u32 = 256u;
const MAX_COS_CYCLES_GPU: f32 = 200000.0;

fn mat7_zero() -> array<f32, 49> {
    var z: array<f32, 49>;
    for (var idx = 0u; idx < 49u; idx = idx + 1u) {
        z[idx] = 0.0;
    }
    return z;
}

fn mat7_mul(
    a: ptr<function, array<f32, 49>>,
    b: ptr<function, array<f32, 49>>,
) -> array<f32, 49> {
    var c = mat7_zero();
    for (var i = 0u; i < 7u; i = i + 1u) {
        for (var k = 0u; k < 7u; k = k + 1u) {
            let aik = (*a)[i * 7u + k];
            if (aik != 0.0) {
                for (var j = 0u; j < 7u; j = j + 1u) {
                    c[i * 7u + j] = c[i * 7u + j] + aik * (*b)[k * 7u + j];
                }
            }
        }
    }
    return c;
}

fn mat7_square(a: ptr<function, array<f32, 49>>) -> array<f32, 49> {
    var c = mat7_zero();
    for (var i = 0u; i < 7u; i = i + 1u) {
        for (var k = 0u; k < 7u; k = k + 1u) {
            let aik = (*a)[i * 7u + k];
            if (aik != 0.0) {
                for (var j = 0u; j < 7u; j = j + 1u) {
                    c[i * 7u + j] = c[i * 7u + j] + aik * (*a)[k * 7u + j];
                }
            }
        }
    }
    return c;
}

// Given F_a = P_a - I and F_b = P_b - I, return P_a * P_b - I = F_a + F_b + F_a * F_b.
fn mat7_compose_f(
    fa: ptr<function, array<f32, 49>>,
    fb: ptr<function, array<f32, 49>>,
) -> array<f32, 49> {
    let ab = mat7_mul(fa, fb);
    var out: array<f32, 49>;
    for (var idx = 0u; idx < 49u; idx = idx + 1u) {
        out[idx] = (*fa)[idx] + (*fb)[idx] + ab[idx];
    }
    return out;
}

// Compute (I + F)^n - I in O(log n) matrix multiplications without cancellation near I.
fn mat7_pow_m1(f_in: ptr<function, array<f32, 49>>, n_in: u32) -> array<f32, 49> {
    var res = mat7_zero();
    if (n_in == 0u) {
        return res;
    }
    var base = *f_in;
    var n = n_in;
    loop {
        if (n == 0u) {
            break;
        }
        if ((n & 1u) == 1u) {
            res = mat7_compose_f(&res, &base);
        }
        n = n >> 1u;
        if (n > 0u) {
            let b2 = mat7_square(&base);
            for (var idx = 0u; idx < 49u; idx = idx + 1u) {
                base[idx] = 2.0 * base[idx] + b2[idx];
            }
        }
    }
    return res;
}

// Compute exp(A) - I for a 7x7 matrix via Higham (2005) scaling-and-squaring [6/6] Pade.
fn expm_m1_pade66_7(a_in: ptr<function, array<f32, 49>>) -> array<f32, 49> {
    var norm1: f32 = 0.0;
    for (var j = 0u; j < 7u; j = j + 1u) {
        var csum: f32 = 0.0;
        for (var i = 0u; i < 7u; i = i + 1u) {
            csum = csum + abs((*a_in)[i * 7u + j]);
        }
        norm1 = max(norm1, csum);
    }

    if (norm1 == 0.0) {
        return mat7_zero();
    }

    var s: u32 = 0u;
    if (norm1 > 0.5) {
        s = min(u32(max(0.0, ceil(log2(norm1 * 2.0)))), 30u);
    }
    let inv_scale = exp2(-f32(s));

    var a_s: array<f32, 49>;
    for (var idx = 0u; idx < 49u; idx = idx + 1u) {
        a_s[idx] = (*a_in)[idx] * inv_scale;
    }

    var a2 = mat7_square(&a_s);
    var a4 = mat7_square(&a2);
    let a6 = mat7_mul(&a2, &a4);

    var odd_in: array<f32, 49>;
    var v_mat: array<f32, 49>;
    for (var idx = 0u; idx < 49u; idx = idx + 1u) {
        odd_in[idx] = PADE_B5 * a4[idx] + PADE_B3 * a2[idx];
        v_mat[idx] = PADE_B6 * a6[idx] + PADE_B4 * a4[idx] + PADE_B2 * a2[idx];
    }
    for (var d = 0u; d < 7u; d = d + 1u) {
        odd_in[d * 7u + d] = odd_in[d * 7u + d] + PADE_B1;
        v_mat[d * 7u + d] = v_mat[d * 7u + d] + PADE_B0;
    }
    let u_mat = mat7_mul(&a_s, &odd_in);

    var lu: array<f32, 49>;
    var rhs: array<f32, 49>;
    for (var idx = 0u; idx < 49u; idx = idx + 1u) {
        lu[idx] = v_mat[idx] - u_mat[idx];
        rhs[idx] = 2.0 * u_mat[idx];
    }

    // Partial-pivoted Gaussian elimination on (V - U) * F = 2 * U
    for (var col = 0u; col < 7u; col = col + 1u) {
        var piv = col;
        var max_v = abs(lu[col * 7u + col]);
        for (var r = col + 1u; r < 7u; r = r + 1u) {
            let v = abs(lu[r * 7u + col]);
            if (v > max_v) {
                max_v = v;
                piv = r;
            }
        }
        if (piv != col) {
            for (var c = 0u; c < 7u; c = c + 1u) {
                let tl = lu[col * 7u + c];
                lu[col * 7u + c] = lu[piv * 7u + c];
                lu[piv * 7u + c] = tl;
                let tr = rhs[col * 7u + c];
                rhs[col * 7u + c] = rhs[piv * 7u + c];
                rhs[piv * 7u + c] = tr;
            }
        }
        let diag = lu[col * 7u + col];
        let safe_diag = select(diag, 1e-20, abs(diag) < 1e-20);
        let inv_diag = 1.0 / safe_diag;
        for (var r = col + 1u; r < 7u; r = r + 1u) {
            let fac = lu[r * 7u + col] * inv_diag;
            if (fac != 0.0) {
                for (var c = col + 1u; c < 7u; c = c + 1u) {
                    lu[r * 7u + c] = lu[r * 7u + c] - fac * lu[col * 7u + c];
                }
                for (var c = 0u; c < 7u; c = c + 1u) {
                    rhs[r * 7u + c] = rhs[r * 7u + c] - fac * rhs[col * 7u + c];
                }
            }
        }
    }

    var f_mat = mat7_zero();
    for (var ii = 0u; ii < 7u; ii = ii + 1u) {
        let i = 6u - ii;
        let diag = lu[i * 7u + i];
        let safe_diag = select(diag, 1e-20, abs(diag) < 1e-20);
        let inv_diag = 1.0 / safe_diag;
        for (var col = 0u; col < 7u; col = col + 1u) {
            var acc = rhs[i * 7u + col];
            for (var k = i + 1u; k < 7u; k = k + 1u) {
                acc = acc - lu[i * 7u + k] * f_mat[k * 7u + col];
            }
            f_mat[i * 7u + col] = acc * inv_diag;
        }
    }

    for (var step = 0u; step < s; step = step + 1u) {
        let f2 = mat7_square(&f_mat);
        for (var idx = 0u; idx < 49u; idx = idx + 1u) {
            f_mat[idx] = 2.0 * f_mat[idx] + f2[idx];
        }
    }
    return f_mat;
}

fn build_liouvillian_scaled_7(
    pa: f32,
    pb: f32,
    kab: f32,
    kba: f32,
    om_a: f32,
    om_b: f32,
    w1x: f32,
    r1a: f32,
    r2a: f32,
    r1b: f32,
    r2b: f32,
    dt: f32,
) -> array<f32, 49> {
    var lt = mat7_zero();
    lt[1u * 7u + 1u] = (-r2a - kab) * dt;
    lt[1u * 7u + 2u] = -om_a * dt;
    lt[1u * 7u + 4u] = kba * dt;

    lt[2u * 7u + 1u] = om_a * dt;
    lt[2u * 7u + 2u] = (-r2a - kab) * dt;
    lt[2u * 7u + 3u] = -w1x * dt;
    lt[2u * 7u + 5u] = kba * dt;

    lt[3u * 7u + 0u] = (r1a * pa) * dt;
    lt[3u * 7u + 2u] = w1x * dt;
    lt[3u * 7u + 3u] = (-r1a - kab) * dt;
    lt[3u * 7u + 6u] = kba * dt;

    lt[4u * 7u + 1u] = kab * dt;
    lt[4u * 7u + 4u] = (-r2b - kba) * dt;
    lt[4u * 7u + 5u] = -om_b * dt;

    lt[5u * 7u + 2u] = kab * dt;
    lt[5u * 7u + 4u] = om_b * dt;
    lt[5u * 7u + 5u] = (-r2b - kba) * dt;
    lt[5u * 7u + 6u] = -w1x * dt;

    lt[6u * 7u + 0u] = (r1b * pb) * dt;
    lt[6u * 7u + 3u] = kab * dt;
    lt[6u * 7u + 5u] = w1x * dt;
    lt[6u * 7u + 6u] = (-r1b - kba) * dt;
    return lt;
}

@compute @workgroup_size(64, 1, 1)
fn eval_multi_cest_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let job_idx = gid.x;
    if (job_idx >= params.n_jobs) {
        return;
    }

    let job = job_map[job_idx];
    let c = curves[job.curve_idx];
    let rf_ppm = offsets_ppm[c.offset_start + job.local_offset_idx];

    let pb = clamp(c.p_b, 0.0, 0.999999);
    let pa = 1.0 - pb;
    let kab = pb * c.k_ex;
    let kba = pa * c.k_ex;

    let scale_rad = TAU_F32 * c.larmor_mhz;
    let om_a = scale_rad * (c.major_ppm - rf_ppm);
    let om_b = scale_rad * ((c.major_ppm + c.dw_ab_ppm) - rf_ppm);
    let w1_nom = TAU_F32 * c.b1_hz;
    let sqrt2_sig = SQRT2_F32 * max(c.b1_inhom_frac, 0.0);

    let n_q = select(1u, params.n_quad, c.b1_inhom_frac > 0.0 && params.n_quad > 1u);
    var iz_a_acc: f32 = 0.0;

    if (c.scheme == 0u) {
        // CW-CEST
        let t_ex = c.t_ex_or_tau_p;
        for (var q = 0u; q < n_q; q = q + 1u) {
            let qn = quad_nodes[q];
            let xq = select(0.0, qn.x_node, n_q > 1u);
            let wq = select(1.0, qn.weight, n_q > 1u);
            let w1 = max(0.0, w1_nom * (1.0 + sqrt2_sig * xq));
            var lt = build_liouvillian_scaled_7(
                pa, pb, kab, kba, om_a, om_b, w1, c.r1a, c.r2a, c.r1b, c.r2b, t_ex
            );
            let f_tot = expm_m1_pade66_7(&lt);
            let iz_a = f_tot[3u * 7u + 0u] * c.e0
                + (1.0 + f_tot[3u * 7u + 3u]) * pa
                + f_tot[3u * 7u + 6u] * pb;
            iz_a_acc = iz_a_acc + wq * iz_a;
        }
    } else if (c.scheme == 1u) {
        // DANTE-CEST: U = (exp(L_delay * tau_d) * exp(L_pulse * tau_p))^n_pulses
        let tau_p = c.t_ex_or_tau_p;
        let tau_d = c.tau_d_or_mod_freq;
        let n_pulses = min(c.n_pulses_or_steps, MAX_DANTE_PULSES_GPU);

        var f_delay = mat7_zero();
        if (tau_d > 0.0) {
            var ld = build_liouvillian_scaled_7(
                pa, pb, kab, kba, om_a, om_b, 0.0, c.r1a, c.r2a, c.r1b, c.r2b, tau_d
            );
            f_delay = expm_m1_pade66_7(&ld);
        }

        for (var q = 0u; q < n_q; q = q + 1u) {
            let qn = quad_nodes[q];
            let xq = select(0.0, qn.x_node, n_q > 1u);
            let wq = select(1.0, qn.weight, n_q > 1u);
            let w1 = max(0.0, w1_nom * (1.0 + sqrt2_sig * xq));
            var lp = build_liouvillian_scaled_7(
                pa, pb, kab, kba, om_a, om_b, w1, c.r1a, c.r2a, c.r1b, c.r2b, tau_p
            );
            var f_pulse = expm_m1_pade66_7(&lp);
            var f_cycle = mat7_compose_f(&f_delay, &f_pulse);
            let f_tot = mat7_pow_m1(&f_cycle, n_pulses);
            let iz_a = f_tot[3u * 7u + 0u] * c.e0
                + (1.0 + f_tot[3u * 7u + 3u]) * pa
                + f_tot[3u * 7u + 6u] * pb;
            iz_a_acc = iz_a_acc + wq * iz_a;
        }
    } else {
        // cos-CEST: omega_1(t) = 2 * omega_{1,eff} * cos(2 * pi * nu_mod * t)
        let t_ex = c.t_ex_or_tau_p;
        let mod_freq = max(c.tau_d_or_mod_freq, 1e-12);
        let t_mod = 1.0 / mod_freq;
        let m_steps = clamp(c.n_pulses_or_steps, 4u, MAX_COS_STEPS_GPU);
        let dt = t_mod / f32(m_steps);
        let cycles_f = min(t_ex / t_mod, MAX_COS_CYCLES_GPU);
        let n_full = u32(max(0.0, floor(cycles_f + 1e-6)));
        let t_rem = max(0.0, t_ex - f32(n_full) * t_mod);
        let has_rem = t_rem > 1e-6 * t_mod;
        let is_even = (m_steps & 1u) == 0u;

        for (var q = 0u; q < n_q; q = q + 1u) {
            let qn = quad_nodes[q];
            let xq = select(0.0, qn.x_node, n_q > 1u);
            let wq = select(1.0, qn.weight, n_q > 1u);
            let w1_eff = max(0.0, w1_nom * (1.0 + sqrt2_sig * xq));

            var f_tot = mat7_zero();
            if (n_full > 0u) {
                var f_cycle = mat7_zero();
                if (is_even) {
                    let half = m_steps >> 1u;
                    var f_first = mat7_zero();
                    var f_second = mat7_zero();
                    for (var k = 0u; k < half; k = k + 1u) {
                        let phase = TAU_F32 * (f32(k) + 0.5) / f32(m_steps);
                        let w1_k = 2.0 * w1_eff * cos(phase);
                        var lk = build_liouvillian_scaled_7(
                            pa, pb, kab, kba, om_a, om_b, w1_k, c.r1a, c.r2a, c.r1b, c.r2b, dt
                        );
                        var f_k = expm_m1_pade66_7(&lk);
                        f_first = mat7_compose_f(&f_k, &f_first);
                        f_second = mat7_compose_f(&f_second, &f_k);
                    }
                    f_cycle = mat7_compose_f(&f_second, &f_first);
                } else {
                    for (var k = 0u; k < m_steps; k = k + 1u) {
                        let phase = TAU_F32 * (f32(k) + 0.5) / f32(m_steps);
                        let w1_k = 2.0 * w1_eff * cos(phase);
                        var lk = build_liouvillian_scaled_7(
                            pa, pb, kab, kba, om_a, om_b, w1_k, c.r1a, c.r2a, c.r1b, c.r2b, dt
                        );
                        var f_k = expm_m1_pade66_7(&lk);
                        f_cycle = mat7_compose_f(&f_k, &f_cycle);
                    }
                }
                f_tot = mat7_pow_m1(&f_cycle, n_full);
            }

            if (has_rem) {
                let m_rem = clamp(u32(ceil(f32(m_steps) * (t_rem / t_mod))), 1u, m_steps);
                let dt_rem = t_rem / f32(m_rem);
                for (var k = 0u; k < m_rem; k = k + 1u) {
                    let t_k = (f32(k) + 0.5) * dt_rem;
                    let w1_k = 2.0 * w1_eff * cos(TAU_F32 * mod_freq * t_k);
                    var lk = build_liouvillian_scaled_7(
                        pa, pb, kab, kba, om_a, om_b, w1_k, c.r1a, c.r2a, c.r1b, c.r2b, dt_rem
                    );
                    var f_k = expm_m1_pade66_7(&lk);
                    f_tot = mat7_compose_f(&f_k, &f_tot);
                }
            }

            let iz_a = f_tot[3u * 7u + 0u] * c.e0
                + (1.0 + f_tot[3u * 7u + 3u]) * pa
                + f_tot[3u * 7u + 6u] * pb;
            iz_a_acc = iz_a_acc + wq * iz_a;
        }
    }

    out_intensities[job_idx] = iz_a_acc * c.inv_i0;
}
