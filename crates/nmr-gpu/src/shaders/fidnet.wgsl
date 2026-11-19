// Clean-room WGSL compute shader for batched symmetric dilated-convolution `FID-Net`
// time-domain NMR inference (NUS reconstruction, virtual J-decoupling, denoising).
//
// Weight Status:
// Untrained architecture (no bundled pretrained weights; requires user-supplied or externally
// trained FIDNET-V1 weights).
//
// Provenance & License:
// Clean-room GPU compute implementation matching `nmr_core::fidnet::FidNetModel` (f64 CPU
// reference) to < 1e-4, based on the MIT-licensed FID-Net symmetric dilated-convolution
// architecture (with optional `input_residual_skip` shortcut; not the full FID-Net-2 architecture):
// - Karunanithy, G. & Hansen, D. F. (2021), J. Biomol. NMR 75, 179-191.
// - Karunanithy, G., Mackenzie, H. W. & Hansen, D. F. (2021), J. Am. Chem. Soc. 143, 16935-16942.
// - Karunanithy, G. et al. (2024), Nat. Commun.
// - Karunanithy, G. et al. (2024), Sci. Adv.
// - Reference repository: https://github.com/gogulan-k/FID-Net (MIT License).

struct FidNetParams {
    batch_size: u32,
    t_len: u32,
    in_ch: u32,
    res_ch: u32,

    skip_ch: u32,
    kernel_size: u32,
    dilation: u32,
    input_residual_skip: u32,

    off_stem_w: u32,
    off_stem_b: u32,
    off_filter_w: u32,
    off_filter_b: u32,

    off_gate_w: u32,
    off_gate_b: u32,
    off_res_w: u32,
    off_res_b: u32,

    off_skip_w: u32,
    off_skip_b: u32,
    off_head1_w: u32,
    off_head1_b: u32,

    off_out_w: u32,
    off_out_b: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> params: FidNetParams;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> input_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> h_src: array<f32>;
@group(0) @binding(4) var<storage, read_write> h_dst: array<f32>;
@group(0) @binding(5) var<storage, read_write> z_buf: array<f32>;
@group(0) @binding(6) var<storage, read_write> skip_buf: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_buf: array<f32>;

// Overflow-safe tanh matching `nmr_core::fidnet::safe_tanh` (exp only evaluated on <= 0.0).
fn safe_tanh(x: f32) -> f32 {
    let xc = clamp(x, -10.0, 10.0);
    if (xc >= 0.0) {
        let e = exp(-2.0 * xc);
        return (1.0 - e) / (1.0 + e);
    }
    let e = exp(2.0 * xc);
    return (e - 1.0) / (e + 1.0);
}

// Overflow-safe logistic sigmoid matching `nmr_core::fidnet::safe_sigmoid`.
fn safe_sigmoid(x: f32) -> f32 {
    let xc = clamp(x, -20.0, 20.0);
    if (xc >= 0.0) {
        return 1.0 / (1.0 + exp(-xc));
    }
    let e = exp(xc);
    return e / (1.0 + e);
}

// Stage 1: Input 1x1 stem projection [B, in_ch, T] -> h_src [B, res_ch, T].
@compute @workgroup_size(64, 1, 1)
fn fidnet_stem(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let oc = gid.y;
    let b = gid.z;
    if (t >= params.t_len || oc >= params.res_ch || b >= params.batch_size) {
        return;
    }

    var acc: f32 = weights[params.off_stem_b + oc];
    let w_row = params.off_stem_w + oc * params.in_ch;
    let in_b_base = b * params.in_ch * params.t_len;
    for (var ic: u32 = 0u; ic < params.in_ch; ic = ic + 1u) {
        acc = acc + weights[w_row + ic] * input_buf[in_b_base + ic * params.t_len + t];
    }
    h_src[(b * params.res_ch + oc) * params.t_len + t] = acc;
}

// Stage 2a: Dilated 1D gated-activation convolution z = tanh(W_f * h + b_f) * sigmoid(W_g * h + b_g).
@compute @workgroup_size(64, 1, 1)
fn fidnet_dilated_block(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let oc = gid.y;
    let b = gid.z;
    if (t >= params.t_len || oc >= params.res_ch || b >= params.batch_size) {
        return;
    }

    let radius: i32 = i32(params.kernel_size / 2u);
    let dil: i32 = i32(params.dilation);
    let t_i32: i32 = i32(t);
    let t_len_i32: i32 = i32(params.t_len);

    var f_acc: f32 = weights[params.off_filter_b + oc];
    var g_acc: f32 = weights[params.off_gate_b + oc];

    let h_b_base = b * params.res_ch * params.t_len;
    for (var ic: u32 = 0u; ic < params.res_ch; ic = ic + 1u) {
        let h_row = h_b_base + ic * params.t_len;
        let w_base = (oc * params.res_ch + ic) * params.kernel_size;
        for (var m: u32 = 0u; m < params.kernel_size; m = m + 1u) {
            let src_t = t_i32 + (i32(m) - radius) * dil;
            if (src_t >= 0 && src_t < t_len_i32) {
                let v = h_src[h_row + u32(src_t)];
                f_acc = f_acc + weights[params.off_filter_w + w_base + m] * v;
                g_acc = g_acc + weights[params.off_gate_w + w_base + m] * v;
            }
        }
    }

    z_buf[(b * params.res_ch + oc) * params.t_len + t] = safe_tanh(f_acc) * safe_sigmoid(g_acc);
}

// Stage 2b: Residual 1x1 projection + residual bypass h_dst = h_src + W_res * z + b_res.
@compute @workgroup_size(64, 1, 1)
fn fidnet_block_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let oc = gid.y;
    let b = gid.z;
    if (t >= params.t_len || oc >= params.res_ch || b >= params.batch_size) {
        return;
    }

    let idx = (b * params.res_ch + oc) * params.t_len + t;
    var acc: f32 = h_src[idx] + weights[params.off_res_b + oc];
    let w_row = params.off_res_w + oc * params.res_ch;
    let z_b_base = b * params.res_ch * params.t_len;
    for (var ic: u32 = 0u; ic < params.res_ch; ic = ic + 1u) {
        acc = acc + weights[w_row + ic] * z_buf[z_b_base + ic * params.t_len + t];
    }
    h_dst[idx] = acc;
}

// Stage 2c: Skip 1x1 projection accumulated into skip_buf += W_skip * z + b_skip.
@compute @workgroup_size(64, 1, 1)
fn fidnet_block_skip(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let os = gid.y;
    let b = gid.z;
    if (t >= params.t_len || os >= params.skip_ch || b >= params.batch_size) {
        return;
    }

    let idx = (b * params.skip_ch + os) * params.t_len + t;
    var acc: f32 = skip_buf[idx] + weights[params.off_skip_b + os];
    let w_row = params.off_skip_w + os * params.res_ch;
    let z_b_base = b * params.res_ch * params.t_len;
    for (var ic: u32 = 0u; ic < params.res_ch; ic = ic + 1u) {
        acc = acc + weights[w_row + ic] * z_buf[z_b_base + ic * params.t_len + t];
    }
    skip_buf[idx] = acc;
}

// Stage 3a: Post-skip head hidden layer z_buf = ReLU(W_h1 * ReLU(skip_buf) + b_h1).
@compute @workgroup_size(64, 1, 1)
fn fidnet_head_hidden(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let os = gid.y;
    let b = gid.z;
    if (t >= params.t_len || os >= params.skip_ch || b >= params.batch_size) {
        return;
    }

    var acc: f32 = weights[params.off_head1_b + os];
    let w_row = params.off_head1_w + os * params.skip_ch;
    let s_b_base = b * params.skip_ch * params.t_len;
    for (var is_ch: u32 = 0u; is_ch < params.skip_ch; is_ch = is_ch + 1u) {
        let sa = max(skip_buf[s_b_base + is_ch * params.t_len + t], 0.0);
        acc = acc + weights[w_row + is_ch] * sa;
    }
    z_buf[(b * params.skip_ch + os) * params.t_len + t] = max(acc, 0.0);
}

// Stage 3b: Final 1x1 linear projection to complex output [Re, Im] (+ optional input residual).
@compute @workgroup_size(64, 1, 1)
fn fidnet_head(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let o = gid.y;
    let b = gid.z;
    if (t >= params.t_len || o >= 2u || b >= params.batch_size) {
        return;
    }

    var acc: f32 = weights[params.off_out_b + o];
    let w_row = params.off_out_w + o * params.skip_ch;
    let z_b_base = b * params.skip_ch * params.t_len;
    for (var is_ch: u32 = 0u; is_ch < params.skip_ch; is_ch = is_ch + 1u) {
        acc = acc + weights[w_row + is_ch] * z_buf[z_b_base + is_ch * params.t_len + t];
    }
    if (params.input_residual_skip != 0u) {
        acc = acc + input_buf[(b * params.in_ch + o) * params.t_len + t];
    }
    out_buf[(b * 2u + o) * params.t_len + t] = acc;
}
