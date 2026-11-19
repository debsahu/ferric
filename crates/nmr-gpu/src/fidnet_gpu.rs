//! WebGPU (`wgpu` + WGSL) batched symmetric dilated-convolution `FID-Net`
//! compute engine (`GpuFidNet`).
//!
//! ## Weight Status
//! **Untrained architecture (no bundled pretrained weights; requires user-supplied or externally
//! trained `FIDNET-V1` weights).**
//!
//! ## Provenance & License Attribution
//! Clean-room GPU compute implementation matching `nmr_core::fidnet::FidNetModel` (`f64` CPU
//! reference) to `< 1e-4`, based on the MIT-licensed `FID-Net` symmetric dilated-convolution
//! architecture (with optional `input_residual_skip` shortcut; not the full `FID-Net-2`
//! architecture):
//! - Karunanithy, G. & Hansen, D. F. (2021), *J. Biomol. NMR* 75, 179–191.
//! - Karunanithy, G., Mackenzie, H. W. & Hansen, D. F. (2021), *J. Am. Chem. Soc.* 143(41),
//!   16935–16942.
//! - Karunanithy, G. et al. (2024), *Nat. Commun.*
//! - Karunanithy, G. et al. (2024), *Sci. Adv.*
//! - Reference architecture repository: <https://github.com/gogulan-k/FID-Net> (MIT License).

use bytemuck::{Pod, Zeroable};
use nmr_core::fid::Complex64;
use nmr_core::fidnet::{
    FidNetBlockOffsets, FidNetConfig, FidNetModel, FidNetPackedWeightsF32, MAX_FIDNET_BATCH,
    MAX_FIDNET_LENGTH,
};
use wgpu::util::DeviceExt;

use crate::GpuEngine;

const WORKGROUP_SIZE_T: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FidNetParamsGpu {
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
}

#[derive(Clone, Copy, Debug)]
struct FidNetBufferSizes {
    input_bytes: u64,
    h_bytes: u64,
    z_bytes: u64,
    skip_bytes: u64,
    out_bytes: u64,
}

fn align4(n: u64) -> u64 {
    let aligned = n.div_ceil(wgpu::COPY_BUFFER_ALIGNMENT) * wgpu::COPY_BUFFER_ALIGNMENT;
    aligned.max(wgpu::COPY_BUFFER_ALIGNMENT)
}

fn compute_buffer_sizes(
    config: &FidNetConfig,
    batch_size: usize,
    t_len: usize,
) -> Result<FidNetBufferSizes, String> {
    let in_ch = config.in_channels();
    let res_ch = config.residual_channels;
    let skip_ch = config.skip_channels;
    let z_ch = res_ch.max(skip_ch);
    let f32_bytes = std::mem::size_of::<f32>() as u64;

    let calc_bytes = |ch: usize| -> Result<u64, String> {
        let elems = (batch_size as u64)
            .checked_mul(ch as u64)
            .and_then(|v| v.checked_mul(t_len as u64))
            .ok_or_else(|| "FID-Net GPU buffer element count overflow".to_string())?;
        let raw = elems
            .checked_mul(f32_bytes)
            .ok_or_else(|| "FID-Net GPU buffer byte count overflow".to_string())?;
        Ok(align4(raw))
    };

    Ok(FidNetBufferSizes {
        input_bytes: calc_bytes(in_ch)?,
        h_bytes: calc_bytes(res_ch)?,
        z_bytes: calc_bytes(z_ch)?,
        skip_bytes: calc_bytes(skip_ch)?,
        out_bytes: calc_bytes(2)?,
    })
}

/// Validate all WebGPU storage buffer sizes and compute workgroup counts against `wgpu::Limits`
/// prior to allocating GPU memory.
pub fn validate_fidnet_gpu_dispatch_limits(
    limits: &wgpu::Limits,
    config: &FidNetConfig,
    batch_size: usize,
    t_len: usize,
) -> Result<(), String> {
    config.validate().map_err(|e| e.to_string())?;
    if batch_size == 0 || t_len == 0 {
        return Ok(());
    }
    if t_len > MAX_FIDNET_LENGTH {
        return Err(format!(
            "FID length {t_len} exceeds MAX_FIDNET_LENGTH ({MAX_FIDNET_LENGTH})"
        ));
    }
    if batch_size > MAX_FIDNET_BATCH {
        return Err(format!(
            "FID batch size {batch_size} exceeds MAX_FIDNET_BATCH ({MAX_FIDNET_BATCH})"
        ));
    }

    let sizes = compute_buffer_sizes(config, batch_size, t_len)?;
    let max_bind = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let max_req = sizes
        .input_bytes
        .max(sizes.h_bytes)
        .max(sizes.z_bytes)
        .max(sizes.skip_bytes)
        .max(sizes.out_bytes);
    if max_req > max_bind {
        return Err(format!(
            "FID-Net GPU dispatch requires {max_req} B buffer, exceeding device limit ({max_bind} B) for batch={batch_size}, t_len={t_len}"
        ));
    }

    let max_wg = u64::from(limits.max_compute_workgroups_per_dimension);
    let wg_x = (t_len as u64).div_ceil(u64::from(WORKGROUP_SIZE_T));
    let wg_y = (config.residual_channels.max(config.skip_channels).max(2)) as u64;
    let wg_z = batch_size as u64;
    if wg_x > max_wg || wg_y > max_wg || wg_z > max_wg {
        return Err(format!(
            "FID-Net GPU workgroup grid ({wg_x}x{wg_y}x{wg_z}) exceeds device limit ({max_wg})"
        ));
    }

    Ok(())
}

/// WebGPU compute engine for batched symmetric dilated-convolution `FID-Net` time-domain inference
/// (untrained architecture; requires user-supplied or externally trained `FIDNET-V1` weights).
pub struct GpuFidNet {
    config: FidNetConfig,
    off_stem_w: u32,
    off_stem_b: u32,
    block_offsets: Vec<FidNetBlockOffsets>,
    off_head1_w: u32,
    off_head1_b: u32,
    off_out_w: u32,
    off_out_b: u32,
    weights_buf: wgpu::Buffer,
    bgl: wgpu::BindGroupLayout,
    stem_pipeline: wgpu::ComputePipeline,
    dilated_block_pipeline: wgpu::ComputePipeline,
    block_residual_pipeline: wgpu::ComputePipeline,
    block_skip_pipeline: wgpu::ComputePipeline,
    head_hidden_pipeline: wgpu::ComputePipeline,
    head_pipeline: wgpu::ComputePipeline,
}

impl GpuFidNet {
    /// Compile the `fidnet.wgsl` compute pipelines and upload validated `f32` model weights to GPU.
    pub fn from_model(engine: &GpuEngine, model: &FidNetModel) -> Result<Self, String> {
        let packed: FidNetPackedWeightsF32 = model
            .weights
            .pack_f32(&model.config)
            .map_err(|e| e.to_string())?;
        let weight_bytes = (packed.data.len() as u64)
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| "FID-Net weight byte size overflow".to_string())?;
        let max_bind = engine
            .device
            .limits()
            .max_storage_buffer_binding_size
            .min(engine.device.limits().max_buffer_size);
        if weight_bytes > max_bind {
            return Err(format!(
                "FID-Net weight buffer ({weight_bytes} B) exceeds device storage binding limit ({max_bind} B)"
            ));
        }

        let shader = engine
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("nmr-fidnet-wgsl"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fidnet.wgsl").into()),
            });

        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bgl = engine
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("nmr-fidnet-bgl"),
                entries: &[
                    uniform_entry(0),
                    storage_ro(1),
                    storage_ro(2),
                    storage_rw(3),
                    storage_rw(4),
                    storage_rw(5),
                    storage_rw(6),
                    storage_rw(7),
                ],
            });
        let pl = engine
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("nmr-fidnet-pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

        let make_pipeline = |label: &str, entry: &str| {
            engine
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        let stem_pipeline = make_pipeline("nmr-fidnet-stem", "fidnet_stem");
        let dilated_block_pipeline =
            make_pipeline("nmr-fidnet-dilated-block", "fidnet_dilated_block");
        let block_residual_pipeline =
            make_pipeline("nmr-fidnet-block-residual", "fidnet_block_residual");
        let block_skip_pipeline = make_pipeline("nmr-fidnet-block-skip", "fidnet_block_skip");
        let head_hidden_pipeline = make_pipeline("nmr-fidnet-head-hidden", "fidnet_head_hidden");
        let head_pipeline = make_pipeline("nmr-fidnet-head", "fidnet_head");

        let weights_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nmr-fidnet-weights"),
                contents: bytemuck::cast_slice(&packed.data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        Ok(Self {
            config: model.config.clone(),
            off_stem_w: packed.off_stem_w,
            off_stem_b: packed.off_stem_b,
            block_offsets: packed.block_offsets,
            off_head1_w: packed.off_head1_w,
            off_head1_b: packed.off_head1_b,
            off_out_w: packed.off_out_w,
            off_out_b: packed.off_out_b,
            weights_buf,
            bgl,
            stem_pipeline,
            dilated_block_pipeline,
            block_residual_pipeline,
            block_skip_pipeline,
            head_hidden_pipeline,
            head_pipeline,
        })
    }

    /// Reference to the model architecture configuration uploaded to this GPU instance.
    pub fn config(&self) -> &FidNetConfig {
        &self.config
    }

    /// Execute batched `FID-Net` forward inference asynchronously on WebGPU (`wasm32`-compatible).
    ///
    /// Each entry in `fids` must share the same complex length `T` (`1 <= T <= MAX_FIDNET_LENGTH`).
    /// Large batches are automatically chunked to stay strictly within device buffer and
    /// workgroup limits.
    pub async fn forward_batch_gpu_async(
        &self,
        engine: &GpuEngine,
        fids: &[Vec<Complex64>],
        nus_masks: Option<&[Vec<bool>]>,
        coupling_channels: Option<&[Vec<f64>]>,
    ) -> Result<Vec<Vec<Complex64>>, String> {
        if fids.is_empty() {
            return Ok(Vec::new());
        }
        if fids.len() > MAX_FIDNET_BATCH {
            return Err(format!(
                "FID batch size {} exceeds MAX_FIDNET_BATCH ({MAX_FIDNET_BATCH})",
                fids.len()
            ));
        }
        let t_len = fids[0].len();
        if t_len == 0 {
            return Err("input FID in batch is empty".to_string());
        }
        if t_len > MAX_FIDNET_LENGTH {
            return Err(format!(
                "FID length {t_len} exceeds MAX_FIDNET_LENGTH ({MAX_FIDNET_LENGTH})"
            ));
        }
        if let Some(ms) = nus_masks {
            if ms.len() != fids.len() {
                return Err(format!(
                    "nus_masks batch length {} != fids length {}",
                    ms.len(),
                    fids.len()
                ));
            }
        }
        if let Some(ccs) = coupling_channels {
            if ccs.len() != fids.len() {
                return Err(format!(
                    "coupling_channels batch length {} != fids length {}",
                    ccs.len(),
                    fids.len()
                ));
            }
        }

        // Validate individual FIDs and channel shapes before touching GPU memory.
        for (bi, fid) in fids.iter().enumerate() {
            if fid.len() != t_len {
                return Err(format!(
                    "fids[{bi}] length {} != expected batch length {t_len}",
                    fid.len()
                ));
            }
            for (ti, &[re, im]) in fid.iter().enumerate() {
                if !re.is_finite() || !im.is_finite() {
                    return Err(format!(
                        "non-finite input in fids[{bi}][{ti}] = ({re}, {im})"
                    ));
                }
            }
            if self.config.use_mask_channel {
                if let Some(ms) = nus_masks {
                    if ms[bi].len() != t_len {
                        return Err(format!(
                            "nus_masks[{bi}] length {} != expected {t_len}",
                            ms[bi].len()
                        ));
                    }
                }
            }
            if self.config.use_coupling_channel {
                if let Some(ccs) = coupling_channels {
                    if ccs[bi].len() != t_len {
                        return Err(format!(
                            "coupling_channels[{bi}] length {} != expected {t_len}",
                            ccs[bi].len()
                        ));
                    }
                    if let Some((ti, &v)) = ccs[bi].iter().enumerate().find(|(_, x)| !x.is_finite())
                    {
                        return Err(format!("non-finite coupling_channels[{bi}][{ti}] = {v}"));
                    }
                }
            }
        }

        let limits = engine.device.limits();
        // Verify at least 1 FID fits on this device.
        validate_fidnet_gpu_dispatch_limits(&limits, &self.config, 1, t_len)?;

        // Determine maximum safe batch chunk size under device limits.
        let max_ch = self
            .config
            .in_channels()
            .max(self.config.residual_channels)
            .max(self.config.skip_channels)
            .max(2);
        let bytes_per_fid = (max_ch as u64)
            .saturating_mul(t_len as u64)
            .saturating_mul(std::mem::size_of::<f32>() as u64)
            .max(1);
        let max_bind = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let max_by_mem = (max_bind / bytes_per_fid).max(1) as usize;
        let max_by_wg = limits.max_compute_workgroups_per_dimension.max(1) as usize;
        let chunk_cap = max_by_mem.min(max_by_wg).min(fids.len()).max(1);

        let mut results = Vec::with_capacity(fids.len());
        let mut start = 0usize;
        while start < fids.len() {
            let end = (start + chunk_cap).min(fids.len());
            let mut chunk_out = self
                .dispatch_chunk_async(
                    engine,
                    &fids[start..end],
                    nus_masks.map(|m| &m[start..end]),
                    coupling_channels.map(|c| &c[start..end]),
                    t_len,
                )
                .await?;
            results.append(&mut chunk_out);
            start = end;
        }
        Ok(results)
    }

    async fn dispatch_chunk_async(
        &self,
        engine: &GpuEngine,
        fids: &[Vec<Complex64>],
        nus_masks: Option<&[Vec<bool>]>,
        coupling_channels: Option<&[Vec<f64>]>,
        t_len: usize,
    ) -> Result<Vec<Vec<Complex64>>, String> {
        let batch_size = fids.len();
        validate_fidnet_gpu_dispatch_limits(
            &engine.device.limits(),
            &self.config,
            batch_size,
            t_len,
        )?;
        let sizes = compute_buffer_sizes(&self.config, batch_size, t_len)?;

        let in_ch = self.config.in_channels();
        let mut input_f32 = vec![0.0f32; batch_size * in_ch * t_len];
        for (bi, fid) in fids.iter().enumerate() {
            let b_base = bi * in_ch * t_len;
            for (t, &[re, im]) in fid.iter().enumerate() {
                let re_f = re as f32;
                let im_f = im as f32;
                if !re_f.is_finite() || !im_f.is_finite() {
                    return Err(format!(
                        "FID value overflowed f32 at batch={bi}, t={t}: ({re}, {im})"
                    ));
                }
                input_f32[b_base + t] = re_f;
                input_f32[b_base + t_len + t] = im_f;
            }
            let mut ch = 2usize;
            if self.config.use_mask_channel {
                let c_base = b_base + ch * t_len;
                match nus_masks {
                    Some(ms) => {
                        for (t, &sampled) in ms[bi].iter().enumerate() {
                            input_f32[c_base + t] = if sampled { 1.0 } else { 0.0 };
                        }
                    }
                    None => {
                        for t in 0..t_len {
                            input_f32[c_base + t] = 1.0;
                        }
                    }
                }
                ch += 1;
            }
            if self.config.use_coupling_channel {
                let c_base = b_base + ch * t_len;
                match coupling_channels {
                    Some(ccs) => {
                        for (t, &v) in ccs[bi].iter().enumerate() {
                            let vf = v as f32;
                            if !vf.is_finite() {
                                return Err(format!(
                                    "coupling channel value overflowed f32 at batch={bi}, t={t}: {v}"
                                ));
                            }
                            input_f32[c_base + t] = vf;
                        }
                    }
                    None => {
                        for t in 0..t_len {
                            input_f32[c_base + t] = 1.0;
                        }
                    }
                }
            }
        }

        let input_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nmr-fidnet-input"),
                contents: bytemuck::cast_slice(&input_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let h_a = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-fidnet-h-a"),
            size: sizes.h_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let h_b = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-fidnet-h-b"),
            size: sizes.h_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let z_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-fidnet-z"),
            size: sizes.z_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let skip_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-fidnet-skip"),
            size: sizes.skip_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nmr-fidnet-out"),
            size: sizes.out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let batch_u32 =
            u32::try_from(batch_size).map_err(|_| "batch_size exceeds u32::MAX".to_string())?;
        let t_u32 = u32::try_from(t_len).map_err(|_| "t_len exceeds u32::MAX".to_string())?;
        let wg_x = t_u32.div_ceil(WORKGROUP_SIZE_T);
        let res_ch_u32 = self.config.residual_channels as u32;
        let skip_ch_u32 = self.config.skip_channels as u32;

        // Build per-block uniform buffers and ping-ponged bind groups so the entire dilated stack
        // executes inside a single CommandEncoder submission with strict inter-pass barriers.
        let mut param_bufs = Vec::with_capacity(self.block_offsets.len());
        let mut bind_groups = Vec::with_capacity(self.block_offsets.len());

        for (bi, (&dilation, offs)) in self
            .config
            .dilations
            .iter()
            .zip(self.block_offsets.iter())
            .enumerate()
        {
            let params = FidNetParamsGpu {
                batch_size: batch_u32,
                t_len: t_u32,
                in_ch: in_ch as u32,
                res_ch: res_ch_u32,
                skip_ch: skip_ch_u32,
                kernel_size: self.config.kernel_size as u32,
                dilation: dilation as u32,
                input_residual_skip: u32::from(self.config.input_residual_skip),
                off_stem_w: self.off_stem_w,
                off_stem_b: self.off_stem_b,
                off_filter_w: offs.filter_w,
                off_filter_b: offs.filter_b,
                off_gate_w: offs.gate_w,
                off_gate_b: offs.gate_b,
                off_res_w: offs.res_w,
                off_res_b: offs.res_b,
                off_skip_w: offs.skip_w,
                off_skip_b: offs.skip_b,
                off_head1_w: self.off_head1_w,
                off_head1_b: self.off_head1_b,
                off_out_w: self.off_out_w,
                off_out_b: self.off_out_b,
                pad0: 0,
                pad1: 0,
            };
            let p_buf = engine
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("nmr-fidnet-params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let (src_h, dst_h) = if bi % 2 == 0 {
                (&h_a, &h_b)
            } else {
                (&h_b, &h_a)
            };
            let bg = engine.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nmr-fidnet-bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: p_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.weights_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: src_h.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: dst_h.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: z_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: skip_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            });
            param_bufs.push(p_buf);
            bind_groups.push(bg);
        }

        let first_bg = bind_groups
            .first()
            .ok_or_else(|| "FID-Net model has zero blocks".to_string())?;

        let mut encoder = engine
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nmr-fidnet-encoder"),
            });
        encoder.clear_buffer(&skip_buf, 0, None);

        // 1. Stem 1x1 projection -> h_a
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nmr-fidnet-pass-stem"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.stem_pipeline);
            pass.set_bind_group(0, first_bg, &[]);
            pass.dispatch_workgroups(wg_x, res_ch_u32, batch_u32);
        }

        // 2. Symmetric dilated convolution blocks
        for bg in &bind_groups {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("nmr-fidnet-pass-dilated"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.dilated_block_pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg_x, res_ch_u32, batch_u32);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("nmr-fidnet-pass-res"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.block_residual_pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg_x, res_ch_u32, batch_u32);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("nmr-fidnet-pass-skip"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.block_skip_pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg_x, skip_ch_u32, batch_u32);
            }
        }

        // 3. Output head (ReLU -> 1x1 -> ReLU -> 1x1 linear)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nmr-fidnet-pass-head-hidden"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.head_hidden_pipeline);
            pass.set_bind_group(0, first_bg, &[]);
            pass.dispatch_workgroups(wg_x, skip_ch_u32, batch_u32);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nmr-fidnet-pass-head"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.head_pipeline);
            pass.set_bind_group(0, first_bg, &[]);
            pass.dispatch_workgroups(wg_x, 2, batch_u32);
        }

        engine.queue.submit(Some(encoder.finish()));
        drop(param_bufs);

        let out_count = batch_size * 2 * t_len;
        let out_f32: Vec<f32> = engine.read_buffer_vec_async(&out_buf, out_count).await?;
        let mut out_batch = Vec::with_capacity(batch_size);
        for bi in 0..batch_size {
            let base = bi * 2 * t_len;
            let mut fid_out = Vec::with_capacity(t_len);
            for t in 0..t_len {
                let re = f64::from(out_f32[base + t]);
                let im = f64::from(out_f32[base + t_len + t]);
                fid_out.push([re, im]);
            }
            out_batch.push(fid_out);
        }
        Ok(out_batch)
    }

    /// Synchronous native wrapper around [`Self::forward_batch_gpu_async`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn forward_batch_gpu(
        &self,
        engine: &GpuEngine,
        fids: &[Vec<Complex64>],
        nus_masks: Option<&[Vec<bool>]>,
        coupling_channels: Option<&[Vec<f64>]>,
    ) -> Result<Vec<Vec<Complex64>>, String> {
        pollster::block_on(self.forward_batch_gpu_async(engine, fids, nus_masks, coupling_channels))
    }

    /// Synchronous native convenience wrapper for a single 1D complex FID.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn forward_fid_gpu(
        &self,
        engine: &GpuEngine,
        fid: &[Complex64],
        nus_mask: Option<&[bool]>,
        coupling_channel: Option<&[f64]>,
    ) -> Result<Vec<Complex64>, String> {
        let fids = vec![fid.to_vec()];
        let masks_owned = nus_mask.map(|m| vec![m.to_vec()]);
        let cc_owned = coupling_channel.map(|c| vec![c.to_vec()]);
        let mut batch_out =
            self.forward_batch_gpu(engine, &fids, masks_owned.as_deref(), cc_owned.as_deref())?;
        batch_out
            .pop()
            .ok_or_else(|| "GPU FID-Net returned empty batch for single FID".to_string())
    }
}
