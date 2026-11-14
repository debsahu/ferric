// Two-state (`7x7`) and three-state (`10x10`) homogeneous Bloch-McConnell `CEST` & `DEST`
// WebGPU (`WGSL`) matrix-exponential compute engine (`T13.4`).
//
// Provenance: Clean-room GPU compute implementation citing:
// - McConnell, H. M. (J. Chem. Phys. 1958, 28, 430-431).
// - Helgstrand, M., Hard, T. & Allard, P. (J. Biomol. NMR 2000, 18, 47-61).
// - Vallurupalli, P., Bouvignies, G. & Kay, L. E. (J. Am. Chem. Soc. 2012, 134, 8148-8161).
// - Fawzi, N. L., Ying, J., Ghirlando, R., Torchia, D. A. & Clore, G. M. (Nature 2011, 480, 268-272).
// - Higham, N. J. (SIAM J. Matrix Anal. Appl. 2005, 26, 1179-1193 — scaling-and-squaring [6/6] Pade).

struct CestBatchUniforms {
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

struct Curve2StateGpu {
    p_b: f32,
    k_ex: f32,
    major_ppm: f32,
    dw_ab_ppm: f32,
    r1a: f32,
    r2a: f32,
    r1b: f32,
    r2b: f32,
    larmor_mhz: f32,
    t_ex_s: f32,
    b1_hz: f32,
    b1_inhom_frac: f32,
    e0: f32,
    inv_i0: f32,
    offset_start: u32,
    offset_len: u32,
}

struct Curve3StateGpu {
    p_b: f32,
    p_c: f32,
    k_ex_ab: f32,
    k_ex_bc: f32,
    k_ex_ac: f32,
    major_ppm: f32,
    dw_ab_ppm: f32,
    dw_ac_ppm: f32,
    r1a: f32,
    r2a: f32,
    r1b: f32,
    r2b: f32,
    r1c: f32,
    r2c: f32,
    larmor_mhz: f32,
    t_ex_s: f32,
    b1_hz: f32,
    b1_inhom_frac: f32,
    e0: f32,
    inv_i0: f32,
    offset_start: u32,
    offset_len: u32,
    _pad0: u32,
    _pad1: u32,
}

struct JobMapGpu {
    curve_idx: u32,
    local_offset_idx: u32,
}

@group(0) @binding(0) var<uniform> params: CestBatchUniforms;
@group(0) @binding(1) var<storage, read> quad_nodes: array<QuadNodeGpu>;
@group(0) @binding(2) var<storage, read> curves_2s: array<Curve2StateGpu>;
@group(0) @binding(3) var<storage, read> curves_3s: array<Curve3StateGpu>;
@group(0) @binding(4) var<storage, read> offsets_ppm: array<f32>;
@group(0) @binding(5) var<storage, read> job_map: array<JobMapGpu>;
@group(0) @binding(6) var<storage, read_write> out_intensities: array<f32>;
@group(0) @binding(7) var<storage, read> in_matrices: array<f32>;
@group(0) @binding(8) var<storage, read_write> out_matrices: array<f32>;

const TAU_F32: f32 = 6.283185307179586;
const SQRT2_F32: f32 = 1.4142135623730951;
const PADE_B0: f32 = 1.0;
const PADE_B1: f32 = 0.5;
const PADE_B2: f32 = 0.11363636363636363;      // 5 / 44
const PADE_B3: f32 = 0.015151515151515152;     // 1 / 66
const PADE_B4: f32 = 0.0012626262626262627;    // 1 / 792
const PADE_B5: f32 = 0.00006313131313131313;   // 1 / 15840
const PADE_B6: f32 = 0.0000015031265031265032; // 1 / 665280

fn mat7_mul(a: ptr<function, array<f32, 49>>, b: ptr<function, array<f32, 49>>) -> array<f32, 49> {
    var c: array<f32, 49>;
    for (var idx = 0u; idx < 49u; idx = idx + 1u) {
        c[idx] = 0.0;
    }
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

fn expm_pade66_7(a_in: ptr<function, array<f32, 49>>) -> array<f32, 49> {
    var norm1: f32 = 0.0;
    for (var j = 0u; j < 7u; j = j + 1u) {
        var csum: f32 = 0.0;
        for (var i = 0u; i < 7u; i = i + 1u) {
            csum = csum + abs((*a_in)[i * 7u + j]);
        }
        norm1 = max(norm1, csum);
    }

    var id: array<f32, 49>;
    for (var idx = 0u; idx < 49u; idx = idx + 1u) {
        id[idx] = 0.0;
    }
    for (var d = 0u; d < 7u; d = d + 1u) {
        id[d * 7u + d] = 1.0;
    }
    if (norm1 == 0.0) {
        return id;
    }

    var s: u32 = 0u;
    if (norm1 > 0.5) {
        s = u32(max(0.0, ceil(log2(norm1 / 0.5))));
    }
    let inv_scale = exp2(-f32(s));

    var a_s: array<f32, 49>;
    for (var idx = 0u; idx < 49u; idx = idx + 1u) {
        a_s[idx] = (*a_in)[idx] * inv_scale;
    }

    var a2 = mat7_mul(&a_s, &a_s);
    var a4 = mat7_mul(&a2, &a2);
    var a6 = mat7_mul(&a2, &a4);

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
    var u_mat = mat7_mul(&a_s, &odd_in);

    var lu: array<f32, 49>;
    var rhs: array<f32, 49>;
    for (var idx = 0u; idx < 49u; idx = idx + 1u) {
        lu[idx] = v_mat[idx] - u_mat[idx];
        rhs[idx] = 2.0 * u_mat[idx];
    }

    // Partial-pivoted Gaussian elimination on `(V - U) * F = 2 * U`
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
        let inv_diag = 1.0 / select(lu[col * 7u + col], 1e-20, abs(lu[col * 7u + col]) < 1e-20);
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

    var f_mat: array<f32, 49>;
    for (var ii = 0u; ii < 7u; ii = ii + 1u) {
        let i = 6u - ii;
        let inv_diag = 1.0 / select(lu[i * 7u + i], 1e-20, abs(lu[i * 7u + i]) < 1e-20);
        for (var col = 0u; col < 7u; col = col + 1u) {
            var acc = rhs[i * 7u + col];
            for (var k = i + 1u; k < 7u; k = k + 1u) {
                acc = acc - lu[i * 7u + k] * f_mat[k * 7u + col];
            }
            f_mat[i * 7u + col] = acc * inv_diag;
        }
    }

    for (var step = 0u; step < s; step = step + 1u) {
        let f2 = mat7_mul(&f_mat, &f_mat);
        for (var idx = 0u; idx < 49u; idx = idx + 1u) {
            f_mat[idx] = 2.0 * f_mat[idx] + f2[idx];
        }
    }

    for (var d = 0u; d < 7u; d = d + 1u) {
        f_mat[d * 7u + d] = f_mat[d * 7u + d] + 1.0;
    }
    return f_mat;
}

fn mat10_mul(a: ptr<function, array<f32, 100>>, b: ptr<function, array<f32, 100>>) -> array<f32, 100> {
    var c: array<f32, 100>;
    for (var idx = 0u; idx < 100u; idx = idx + 1u) {
        c[idx] = 0.0;
    }
    for (var i = 0u; i < 10u; i = i + 1u) {
        for (var k = 0u; k < 10u; k = k + 1u) {
            let aik = (*a)[i * 10u + k];
            if (aik != 0.0) {
                for (var j = 0u; j < 10u; j = j + 1u) {
                    c[i * 10u + j] = c[i * 10u + j] + aik * (*b)[k * 10u + j];
                }
            }
        }
    }
    return c;
}

fn expm_pade66_10(a_in: ptr<function, array<f32, 100>>) -> array<f32, 100> {
    var norm1: f32 = 0.0;
    for (var j = 0u; j < 10u; j = j + 1u) {
        var csum: f32 = 0.0;
        for (var i = 0u; i < 10u; i = i + 1u) {
            csum = csum + abs((*a_in)[i * 10u + j]);
        }
        norm1 = max(norm1, csum);
    }

    var id: array<f32, 100>;
    for (var idx = 0u; idx < 100u; idx = idx + 1u) {
        id[idx] = 0.0;
    }
    for (var d = 0u; d < 10u; d = d + 1u) {
        id[d * 10u + d] = 1.0;
    }
    if (norm1 == 0.0) {
        return id;
    }

    var s: u32 = 0u;
    if (norm1 > 0.5) {
        s = u32(max(0.0, ceil(log2(norm1 / 0.5))));
    }
    let inv_scale = exp2(-f32(s));

    var a_s: array<f32, 100>;
    for (var idx = 0u; idx < 100u; idx = idx + 1u) {
        a_s[idx] = (*a_in)[idx] * inv_scale;
    }

    var a2 = mat10_mul(&a_s, &a_s);
    var a4 = mat10_mul(&a2, &a2);
    var a6 = mat10_mul(&a2, &a4);

    var odd_in: array<f32, 100>;
    var v_mat: array<f32, 100>;
    for (var idx = 0u; idx < 100u; idx = idx + 1u) {
        odd_in[idx] = PADE_B5 * a4[idx] + PADE_B3 * a2[idx];
        v_mat[idx] = PADE_B6 * a6[idx] + PADE_B4 * a4[idx] + PADE_B2 * a2[idx];
    }
    for (var d = 0u; d < 10u; d = d + 1u) {
        odd_in[d * 10u + d] = odd_in[d * 10u + d] + PADE_B1;
        v_mat[d * 10u + d] = v_mat[d * 10u + d] + PADE_B0;
    }
    var u_mat = mat10_mul(&a_s, &odd_in);

    var lu: array<f32, 100>;
    var rhs: array<f32, 100>;
    for (var idx = 0u; idx < 100u; idx = idx + 1u) {
        lu[idx] = v_mat[idx] - u_mat[idx];
        rhs[idx] = 2.0 * u_mat[idx];
    }

    for (var col = 0u; col < 10u; col = col + 1u) {
        var piv = col;
        var max_v = abs(lu[col * 10u + col]);
        for (var r = col + 1u; r < 10u; r = r + 1u) {
            let v = abs(lu[r * 10u + col]);
            if (v > max_v) {
                max_v = v;
                piv = r;
            }
        }
        if (piv != col) {
            for (var c = 0u; c < 10u; c = c + 1u) {
                let tl = lu[col * 10u + c];
                lu[col * 10u + c] = lu[piv * 10u + c];
                lu[piv * 10u + c] = tl;
                let tr = rhs[col * 10u + c];
                rhs[col * 10u + c] = rhs[piv * 10u + c];
                rhs[piv * 10u + c] = tr;
            }
        }
        let inv_diag = 1.0 / select(lu[col * 10u + col], 1e-20, abs(lu[col * 10u + col]) < 1e-20);
        for (var r = col + 1u; r < 10u; r = r + 1u) {
            let fac = lu[r * 10u + col] * inv_diag;
            if (fac != 0.0) {
                for (var c = col + 1u; c < 10u; c = c + 1u) {
                    lu[r * 10u + c] = lu[r * 10u + c] - fac * lu[col * 10u + c];
                }
                for (var c = 0u; c < 10u; c = c + 1u) {
                    rhs[r * 10u + c] = rhs[r * 10u + c] - fac * rhs[col * 10u + c];
                }
            }
        }
    }

    var f_mat: array<f32, 100>;
    for (var ii = 0u; ii < 10u; ii = ii + 1u) {
        let i = 9u - ii;
        let inv_diag = 1.0 / select(lu[i * 10u + i], 1e-20, abs(lu[i * 10u + i]) < 1e-20);
        for (var col = 0u; col < 10u; col = col + 1u) {
            var acc = rhs[i * 10u + col];
            for (var k = i + 1u; k < 10u; k = k + 1u) {
                acc = acc - lu[i * 10u + k] * f_mat[k * 10u + col];
            }
            f_mat[i * 10u + col] = acc * inv_diag;
        }
    }

    for (var step = 0u; step < s; step = step + 1u) {
        let f2 = mat10_mul(&f_mat, &f_mat);
        for (var idx = 0u; idx < 100u; idx = idx + 1u) {
            f_mat[idx] = 2.0 * f_mat[idx] + f2[idx];
        }
    }

    for (var d = 0u; d < 10u; d = d + 1u) {
        f_mat[d * 10u + d] = f_mat[d * 10u + d] + 1.0;
    }
    return f_mat;
}

@compute @workgroup_size(64)
fn eval_expm7_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_jobs) {
        return;
    }
    var a: array<f32, 49>;
    let base = idx * 49u;
    for (var k = 0u; k < 49u; k = k + 1u) {
        a[k] = in_matrices[base + k];
    }
    let res = expm_pade66_7(&a);
    for (var k = 0u; k < 49u; k = k + 1u) {
        out_matrices[base + k] = res[k];
    }
}

@compute @workgroup_size(64)
fn eval_expm10_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_jobs) {
        return;
    }
    var a: array<f32, 100>;
    let base = idx * 100u;
    for (var k = 0u; k < 100u; k = k + 1u) {
        a[k] = in_matrices[base + k];
    }
    let res = expm_pade66_10(&a);
    for (var k = 0u; k < 100u; k = k + 1u) {
        out_matrices[base + k] = res[k];
    }
}

@compute @workgroup_size(64)
fn eval_cest_2state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let job_idx = gid.x;
    if (job_idx >= params.n_jobs) {
        return;
    }
    let job = job_map[job_idx];
    let c = curves_2s[job.curve_idx];
    let rf_ppm = offsets_ppm[c.offset_start + job.local_offset_idx];

    let pb = clamp(c.p_b, 0.0, 0.999999);
    let pa = 1.0 - pb;
    let kab = pb * c.k_ex;
    let kba = pa * c.k_ex;
    let t_ex = c.t_ex_s;

    let scale_rad = TAU_F32 * c.larmor_mhz;
    let om_a = scale_rad * (c.major_ppm - rf_ppm);
    let om_b = scale_rad * ((c.major_ppm + c.dw_ab_ppm) - rf_ppm);
    let w1_nom = TAU_F32 * c.b1_hz;
    let sqrt2_sig = SQRT2_F32 * max(c.b1_inhom_frac, 0.0);

    let n_q = select(1u, params.n_quad, c.b1_inhom_frac > 0.0 && params.n_quad > 1u);
    var iz_a_acc: f32 = 0.0;

    for (var q = 0u; q < n_q; q = q + 1u) {
        let qn = quad_nodes[q];
        let xq = select(0.0, qn.x_node, n_q > 1u);
        let wq = select(1.0, qn.weight, n_q > 1u);
        let w1 = max(0.0, w1_nom * (1.0 + sqrt2_sig * xq));

        var lt: array<f32, 49>;
        for (var k = 0u; k < 49u; k = k + 1u) {
            lt[k] = 0.0;
        }
        lt[1u * 7u + 1u] = (-c.r2a - kab) * t_ex;
        lt[1u * 7u + 2u] = -om_a * t_ex;
        lt[1u * 7u + 4u] = kba * t_ex;

        lt[2u * 7u + 1u] = om_a * t_ex;
        lt[2u * 7u + 2u] = (-c.r2a - kab) * t_ex;
        lt[2u * 7u + 3u] = -w1 * t_ex;
        lt[2u * 7u + 5u] = kba * t_ex;

        lt[3u * 7u + 0u] = (c.r1a * pa) * t_ex;
        lt[3u * 7u + 2u] = w1 * t_ex;
        lt[3u * 7u + 3u] = (-c.r1a - kab) * t_ex;
        lt[3u * 7u + 6u] = kba * t_ex;

        lt[4u * 7u + 1u] = kab * t_ex;
        lt[4u * 7u + 4u] = (-c.r2b - kba) * t_ex;
        lt[4u * 7u + 5u] = -om_b * t_ex;

        lt[5u * 7u + 2u] = kab * t_ex;
        lt[5u * 7u + 4u] = om_b * t_ex;
        lt[5u * 7u + 5u] = (-c.r2b - kba) * t_ex;
        lt[5u * 7u + 6u] = -w1 * t_ex;

        lt[6u * 7u + 0u] = (c.r1b * pb) * t_ex;
        lt[6u * 7u + 3u] = kab * t_ex;
        lt[6u * 7u + 5u] = w1 * t_ex;
        lt[6u * 7u + 6u] = (-c.r1b - kba) * t_ex;

        let prop = expm_pade66_7(&lt);
        // v0 = [e0, 0, 0, pa, 0, 0, pb] -> row 3 (`IzA`) dot v0:
        let iz_a = prop[3u * 7u + 0u] * c.e0 + prop[3u * 7u + 3u] * pa + prop[3u * 7u + 6u] * pb;
        iz_a_acc = iz_a_acc + wq * iz_a;
    }

    out_intensities[job_idx] = iz_a_acc * c.inv_i0;
}

@compute @workgroup_size(64)
fn eval_cest_3state(@builtin(global_invocation_id) gid: vec3<u32>) {
    let job_idx = gid.x;
    if (job_idx >= params.n_jobs) {
        return;
    }
    let job = job_map[job_idx];
    let c = curves_3s[job.curve_idx];
    let rf_ppm = offsets_ppm[c.offset_start + job.local_offset_idx];

    let pb = max(c.p_b, 0.0);
    let pc = max(c.p_c, 0.0);
    let pa = max(1.0 - pb - pc, 0.0);
    let kab = select(0.0, c.k_ex_ab * pb / (pa + pb), pa + pb > 0.0);
    let kba = select(0.0, c.k_ex_ab * pa / (pa + pb), pa + pb > 0.0);
    let kbc = select(0.0, c.k_ex_bc * pc / (pb + pc), pb + pc > 0.0);
    let kcb = select(0.0, c.k_ex_bc * pb / (pb + pc), pb + pc > 0.0);
    let kac = select(0.0, c.k_ex_ac * pc / (pa + pc), pa + pc > 0.0);
    let kca = select(0.0, c.k_ex_ac * pa / (pa + pc), pa + pc > 0.0);

    let ka_out = kab + kac;
    let kb_out = kba + kbc;
    let kc_out = kca + kcb;
    let t_ex = c.t_ex_s;

    let scale_rad = TAU_F32 * c.larmor_mhz;
    let om_a = scale_rad * (c.major_ppm - rf_ppm);
    let om_b = scale_rad * ((c.major_ppm + c.dw_ab_ppm) - rf_ppm);
    let om_c = scale_rad * ((c.major_ppm + c.dw_ac_ppm) - rf_ppm);
    let w1_nom = TAU_F32 * c.b1_hz;
    let sqrt2_sig = SQRT2_F32 * max(c.b1_inhom_frac, 0.0);

    let n_q = select(1u, params.n_quad, c.b1_inhom_frac > 0.0 && params.n_quad > 1u);
    var iz_a_acc: f32 = 0.0;

    for (var q = 0u; q < n_q; q = q + 1u) {
        let qn = quad_nodes[q];
        let xq = select(0.0, qn.x_node, n_q > 1u);
        let wq = select(1.0, qn.weight, n_q > 1u);
        let w1 = max(0.0, w1_nom * (1.0 + sqrt2_sig * xq));

        var lt: array<f32, 100>;
        for (var k = 0u; k < 100u; k = k + 1u) {
            lt[k] = 0.0;
        }
        lt[1u * 10u + 1u] = (-c.r2a - ka_out) * t_ex;
        lt[1u * 10u + 2u] = -om_a * t_ex;
        lt[1u * 10u + 4u] = kba * t_ex;
        lt[1u * 10u + 7u] = kca * t_ex;

        lt[2u * 10u + 1u] = om_a * t_ex;
        lt[2u * 10u + 2u] = (-c.r2a - ka_out) * t_ex;
        lt[2u * 10u + 3u] = -w1 * t_ex;
        lt[2u * 10u + 5u] = kba * t_ex;
        lt[2u * 10u + 8u] = kca * t_ex;

        lt[3u * 10u + 0u] = (c.r1a * pa) * t_ex;
        lt[3u * 10u + 2u] = w1 * t_ex;
        lt[3u * 10u + 3u] = (-c.r1a - ka_out) * t_ex;
        lt[3u * 10u + 6u] = kba * t_ex;
        lt[3u * 10u + 9u] = kca * t_ex;

        lt[4u * 10u + 1u] = kab * t_ex;
        lt[4u * 10u + 4u] = (-c.r2b - kb_out) * t_ex;
        lt[4u * 10u + 5u] = -om_b * t_ex;
        lt[4u * 10u + 7u] = kcb * t_ex;

        lt[5u * 10u + 2u] = kab * t_ex;
        lt[5u * 10u + 4u] = om_b * t_ex;
        lt[5u * 10u + 5u] = (-c.r2b - kb_out) * t_ex;
        lt[5u * 10u + 6u] = -w1 * t_ex;
        lt[5u * 10u + 8u] = kcb * t_ex;

        lt[6u * 10u + 0u] = (c.r1b * pb) * t_ex;
        lt[6u * 10u + 3u] = kab * t_ex;
        lt[6u * 10u + 5u] = w1 * t_ex;
        lt[6u * 10u + 6u] = (-c.r1b - kb_out) * t_ex;
        lt[6u * 10u + 9u] = kcb * t_ex;

        lt[7u * 10u + 1u] = kac * t_ex;
        lt[7u * 10u + 4u] = kbc * t_ex;
        lt[7u * 10u + 7u] = (-c.r2c - kc_out) * t_ex;
        lt[7u * 10u + 8u] = -om_c * t_ex;

        lt[8u * 10u + 2u] = kac * t_ex;
        lt[8u * 10u + 5u] = kbc * t_ex;
        lt[8u * 10u + 7u] = om_c * t_ex;
        lt[8u * 10u + 8u] = (-c.r2c - kc_out) * t_ex;
        lt[8u * 10u + 9u] = -w1 * t_ex;

        lt[9u * 10u + 0u] = (c.r1c * pc) * t_ex;
        lt[9u * 10u + 3u] = kac * t_ex;
        lt[9u * 10u + 6u] = kbc * t_ex;
        lt[9u * 10u + 8u] = w1 * t_ex;
        lt[9u * 10u + 9u] = (-c.r1c - kc_out) * t_ex;

        let prop = expm_pade66_10(&lt);
        let iz_a = prop[3u * 10u + 0u] * c.e0
            + prop[3u * 10u + 3u] * pa
            + prop[3u * 10u + 6u] * pb
            + prop[3u * 10u + 9u] * pc;
        iz_a_acc = iz_a_acc + wq * iz_a;
    }

    out_intensities[job_idx] = iz_a_acc * c.inv_i0;
}
