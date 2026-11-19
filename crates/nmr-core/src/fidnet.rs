//! Pure-Rust WaveNet / Dilated-Convolution Time-Domain `FID-Net` / `FID-Net-2` Engine
//! for Non-Uniform Sampling (NUS) reconstruction, virtual homonuclear $J$-decoupling, and
//! complex time-domain FID denoising.
//!
//! ## Provenance & License Attribution
//! Clean-room pure-Rust implementation of the MIT-licensed `FID-Net` / `FID-Net-2` dilated
//! WaveNet time-domain NMR architecture:
//! - Karunanithy, G. & Hansen, D. F. (2021), *J. Biomol. NMR* 75, 179–191
//!   ("FID-Net: A versatile deep neural network architecture for NMR spectral reconstruction
//!   and virtual decoupling").
//! - Karunanithy, G., Mackenzie, H. W. & Hansen, D. F. (2021), *J. Am. Chem. Soc.* 143(41),
//!   16935–16942 ("Virtual Homonuclear Decoupling in Direct Detection Nuclear Magnetic
//!   Resonance Experiments Using Deep Neural Networks").
//! - Karunanithy, G., Reinstein, J. & Hansen, D. F. (2023), *J. Phys. Chem. Lett.* 14(21),
//!   4951–4959 ("FID-Net-2: Improved Architecture for NMR Spectral Reconstruction and
//!   Virtual Decoupling").
//! - Reference architecture repository: <https://github.com/gogulan-k/FID-Net> (MIT License).
//! - WaveNet gated dilated convolution architecture: van den Oord, A. et al. (2016),
//!   *arXiv:1609.03499* ("WaveNet: A Generative Model for Raw Audio").
//! - Sine-weighted Poisson-gap NUS schedule generation: Hyberts, S. G., Takeuchi, K. &
//!   Wagner, G. (2010), *J. Am. Chem. Soc.* 132(7), 2145–2147.
//! - Adaptive moment estimation (`Adam`) optimizer: Kingma, D. P. & Ba, J. (2015), *ICLR*.

use std::f64::consts::{PI, TAU};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::fid::{Complex64, Fid2D, Fid3D};
use crate::ops::fft_arbitrary_in_place;
use crate::rng::Rng;
use crate::torsion_db::sha256_hex;

// ============================================================================
// Constants & Hostile-Input Caps
// ============================================================================

/// Canonical version identifier for serialized `FID-Net` weight bundles.
pub const FIDNET_BUNDLE_FORMAT_VERSION: &str = "FIDNET-V1";

/// Binary magic prefix for compact `FIDNET-V1` binary weight bundles.
pub const FIDNET_BINARY_MAGIC: &[u8; 8] = b"FIDNET1\0";

/// Maximum supported complex time-domain length per 1D FID.
pub const MAX_FIDNET_LENGTH: usize = 65_536;

/// Maximum residual or skip channel width permitted in a `FidNetConfig`.
pub const MAX_FIDNET_CHANNELS: usize = 256;

/// Maximum number of dilated WaveNet blocks permitted in a `FidNetConfig`.
pub const MAX_FIDNET_BLOCKS: usize = 64;

/// Maximum 1D dilated kernel width permitted in a `FidNetConfig`.
pub const MAX_FIDNET_KERNEL_SIZE: usize = 31;

/// Maximum per-block dilation factor permitted in a `FidNetConfig`.
pub const MAX_FIDNET_DILATION: usize = 4_096;

/// Maximum total scalar parameter count permitted in a `FidNetWeights` bundle
/// (guards against hostile allocation bombs).
pub const MAX_FIDNET_PARAMS: usize = 16_777_216;

/// Maximum batch size for multi-column inference.
pub const MAX_FIDNET_BATCH: usize = 65_536;

// ============================================================================
// Errors
// ============================================================================

/// Errors returned by `FID-Net` configuration, serialization, training, and inference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FidNetError {
    InvalidConfig(String),
    InvalidWeights(String),
    UnsupportedFormatVersion(String),
    ChecksumMismatch { expected: String, actual: String },
    SerializationError(String),
    NonFiniteInput(String),
    OversizedFid { len: usize, max_len: usize },
    EmptyInput(String),
    ShapeMismatch(String),
}

impl fmt::Display for FidNetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid FID-Net config: {msg}"),
            Self::InvalidWeights(msg) => write!(f, "invalid FID-Net weights: {msg}"),
            Self::UnsupportedFormatVersion(v) => {
                write!(f, "unsupported FID-Net bundle format version: {v}")
            }
            Self::ChecksumMismatch { expected, actual } => write!(
                f,
                "FID-Net weight bundle SHA-256 mismatch (expected {expected}, got {actual})"
            ),
            Self::SerializationError(msg) => write!(f, "FID-Net serialization error: {msg}"),
            Self::NonFiniteInput(msg) => write!(f, "non-finite FID-Net input: {msg}"),
            Self::OversizedFid { len, max_len } => {
                write!(f, "FID length {len} exceeds MAX_FIDNET_LENGTH ({max_len})")
            }
            Self::EmptyInput(msg) => write!(f, "empty FID-Net input: {msg}"),
            Self::ShapeMismatch(msg) => write!(f, "FID-Net shape mismatch: {msg}"),
        }
    }
}

impl std::error::Error for FidNetError {}

// ============================================================================
// Task, Configuration, Weights & Bundle Serialization
// ============================================================================

/// Target time-domain NMR task performed by a `FID-Net` / `FID-Net-2` model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FidNetTask {
    /// Non-uniformly sampled (NUS) time-domain reconstruction.
    NusReconstruction,
    /// Virtual homonuclear scalar-coupling ($J$) decoupling (e.g. $^{13}\text{C}\text{–}^{13}\text{C}$ or $^1\text{H}\text{–}^1\text{H}$).
    VirtualDecoupling,
    /// Time-domain thermal noise suppression.
    Denoising,
}

/// Architectural configuration for a WaveNet / dilated-convolution `FID-Net` model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FidNetConfig {
    /// Target NMR task.
    pub task: FidNetTask,
    /// Whether an explicit binary NUS sampling mask channel `m(t) ∈ {0, 1}` is appended.
    pub use_mask_channel: bool,
    /// Whether an explicit scalar-coupling conditioning channel `c(t)` is appended.
    pub use_coupling_channel: bool,
    /// Hidden channel count `C` inside the dilated residual stack.
    pub residual_channels: usize,
    /// Channel count `S` of the skip-connection accumulator and post-processing head.
    pub skip_channels: usize,
    /// Odd 1D convolution kernel size `K` (symmetric padding radius `(K - 1) / 2`).
    pub kernel_size: usize,
    /// Per-block dilation schedule (e.g. `[1, 2, 4, 8, 16]`).
    pub dilations: Vec<usize>,
    /// If `true`, adds the input complex FID `[Re(t), Im(t)]` as a global residual shortcut
    /// to the linear output head (FID-Net-2 residual formulation).
    pub output_residual: bool,
}

impl FidNetConfig {
    /// Compact default configuration for NUS reconstruction (`[Re, Im, mask]`).
    pub fn nus_default(residual_channels: usize, skip_channels: usize) -> Self {
        Self {
            task: FidNetTask::NusReconstruction,
            use_mask_channel: true,
            use_coupling_channel: false,
            residual_channels,
            skip_channels,
            kernel_size: 5,
            dilations: vec![1, 2, 4, 8],
            output_residual: true,
        }
    }

    /// Compact default configuration for virtual homonuclear decoupling (`[Re, Im, j_cond]`).
    pub fn virtual_decoupling_default(residual_channels: usize, skip_channels: usize) -> Self {
        Self {
            task: FidNetTask::VirtualDecoupling,
            use_mask_channel: false,
            use_coupling_channel: true,
            residual_channels,
            skip_channels,
            kernel_size: 5,
            dilations: vec![1, 2, 4, 8],
            output_residual: true,
        }
    }

    /// Compact default configuration for time-domain denoising (`[Re, Im]`).
    pub fn denoising_default(residual_channels: usize, skip_channels: usize) -> Self {
        Self {
            task: FidNetTask::Denoising,
            use_mask_channel: false,
            use_coupling_channel: false,
            residual_channels,
            skip_channels,
            kernel_size: 5,
            dilations: vec![1, 2, 4, 8],
            output_residual: true,
        }
    }

    /// Number of input channels (`2` for `[Re, Im]` + optional mask + optional coupling).
    #[inline]
    pub fn in_channels(&self) -> usize {
        2 + usize::from(self.use_mask_channel) + usize::from(self.use_coupling_channel)
    }

    /// Number of dilated WaveNet residual blocks.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.dilations.len()
    }

    /// Symmetric receptive field half-width in time points.
    #[inline]
    pub fn receptive_field_radius(&self) -> usize {
        let half_k = self.kernel_size / 2;
        self.dilations.iter().fold(0usize, |acc, &d| {
            acc.saturating_add(half_k.saturating_mul(d))
        })
    }

    /// Total scalar parameter count implied by this architecture (checked against overflow).
    pub fn parameter_count(&self) -> Result<usize, FidNetError> {
        let in_ch = self.in_channels();
        let c = self.residual_channels;
        let s = self.skip_channels;
        let k = self.kernel_size;
        let n_blocks = self.dilations.len();

        let stem = c
            .checked_mul(in_ch)
            .and_then(|v| v.checked_add(c))
            .ok_or_else(|| FidNetError::InvalidConfig("stem parameter count overflow".into()))?;
        let conv_w = c
            .checked_mul(c)
            .and_then(|v| v.checked_mul(k))
            .ok_or_else(|| FidNetError::InvalidConfig("block conv weight count overflow".into()))?;
        let res_w = c
            .checked_mul(c)
            .ok_or_else(|| FidNetError::InvalidConfig("block res weight count overflow".into()))?;
        let skip_w = s
            .checked_mul(c)
            .ok_or_else(|| FidNetError::InvalidConfig("block skip weight count overflow".into()))?;
        // Per block: filter_w + filter_b + gate_w + gate_b + res_w + res_b + skip_w + skip_b
        let per_block = conv_w
            .checked_mul(2)
            .and_then(|v| v.checked_add(c.checked_mul(3)?))
            .and_then(|v| v.checked_add(res_w))
            .and_then(|v| v.checked_add(skip_w))
            .and_then(|v| v.checked_add(s))
            .ok_or_else(|| FidNetError::InvalidConfig("block parameter count overflow".into()))?;
        let all_blocks = per_block
            .checked_mul(n_blocks)
            .ok_or_else(|| FidNetError::InvalidConfig("total blocks parameter overflow".into()))?;
        // Head: head1_w (s*s) + head1_b (s) + out_w (2*s) + out_b (2)
        let head = s
            .checked_mul(s)
            .and_then(|v| v.checked_add(s))
            .and_then(|v| v.checked_add(s.checked_mul(2)?))
            .and_then(|v| v.checked_add(2))
            .ok_or_else(|| FidNetError::InvalidConfig("head parameter count overflow".into()))?;

        stem.checked_add(all_blocks)
            .and_then(|v| v.checked_add(head))
            .ok_or_else(|| FidNetError::InvalidConfig("total parameter count overflow".into()))
    }

    /// Validate all structural dimensions and hostile-input caps.
    pub fn validate(&self) -> Result<(), FidNetError> {
        if self.residual_channels == 0 || self.residual_channels > MAX_FIDNET_CHANNELS {
            return Err(FidNetError::InvalidConfig(format!(
                "residual_channels={} must be in 1..={MAX_FIDNET_CHANNELS}",
                self.residual_channels
            )));
        }
        if self.skip_channels == 0 || self.skip_channels > MAX_FIDNET_CHANNELS {
            return Err(FidNetError::InvalidConfig(format!(
                "skip_channels={} must be in 1..={MAX_FIDNET_CHANNELS}",
                self.skip_channels
            )));
        }
        if self.kernel_size == 0
            || self.kernel_size > MAX_FIDNET_KERNEL_SIZE
            || self.kernel_size.is_multiple_of(2)
        {
            return Err(FidNetError::InvalidConfig(format!(
                "kernel_size={} must be an odd integer in 1..={MAX_FIDNET_KERNEL_SIZE}",
                self.kernel_size
            )));
        }
        if self.dilations.is_empty() || self.dilations.len() > MAX_FIDNET_BLOCKS {
            return Err(FidNetError::InvalidConfig(format!(
                "num_blocks={} must be in 1..={MAX_FIDNET_BLOCKS}",
                self.dilations.len()
            )));
        }
        for (idx, &d) in self.dilations.iter().enumerate() {
            if d == 0 || d > MAX_FIDNET_DILATION {
                return Err(FidNetError::InvalidConfig(format!(
                    "dilations[{idx}]={d} must be in 1..={MAX_FIDNET_DILATION}"
                )));
            }
        }
        let total = self.parameter_count()?;
        if total > MAX_FIDNET_PARAMS {
            return Err(FidNetError::InvalidConfig(format!(
                "parameter count {total} exceeds MAX_FIDNET_PARAMS ({MAX_FIDNET_PARAMS})"
            )));
        }
        Ok(())
    }
}

/// Weights for a single WaveNet dilated gated-activation block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FidNetBlockWeights {
    /// Filter dilated 1D convolution weights `[C, C, K]`, flattened `(out_c * C + in_c) * K + m`.
    pub filter_w: Vec<f64>,
    /// Filter convolution bias `[C]`.
    pub filter_b: Vec<f64>,
    /// Gate dilated 1D convolution weights `[C, C, K]`, flattened `(out_c * C + in_c) * K + m`.
    pub gate_w: Vec<f64>,
    /// Gate convolution bias `[C]`.
    pub gate_b: Vec<f64>,
    /// `1×1` residual projection weights `[C, C]`, flattened `out_c * C + in_c`.
    pub res_w: Vec<f64>,
    /// `1×1` residual projection bias `[C]`.
    pub res_b: Vec<f64>,
    /// `1×1` skip projection weights `[S, C]`, flattened `out_s * C + in_c`.
    pub skip_w: Vec<f64>,
    /// `1×1` skip projection bias `[S]`.
    pub skip_b: Vec<f64>,
}

/// Complete trainable parameter set for a `FID-Net` / `FID-Net-2` model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FidNetWeights {
    /// Input `1×1` stem weights `[C, in_ch]`, flattened `c * in_ch + ic`.
    pub stem_w: Vec<f64>,
    /// Input `1×1` stem bias `[C]`.
    pub stem_b: Vec<f64>,
    /// Stack of dilated gated-activation WaveNet blocks.
    pub blocks: Vec<FidNetBlockWeights>,
    /// First `1×1` post-skip head weights `[S, S]`, flattened `s_out * S + s_in`.
    pub head1_w: Vec<f64>,
    /// First `1×1` post-skip head bias `[S]`.
    pub head1_b: Vec<f64>,
    /// Final `1×1` linear complex output weights `[2, S]`, flattened `o * S + s`.
    pub out_w: Vec<f64>,
    /// Final `1×1` linear complex output bias `[2]`.
    pub out_b: Vec<f64>,
}

/// Flat `f32` weight packing and byte-aligned offsets shared with the WebGPU `fidnet.wgsl` shader.
#[derive(Clone, Debug, PartialEq)]
pub struct FidNetPackedWeightsF32 {
    pub data: Vec<f32>,
    pub off_stem_w: u32,
    pub off_stem_b: u32,
    pub block_offsets: Vec<FidNetBlockOffsets>,
    pub off_head1_w: u32,
    pub off_head1_b: u32,
    pub off_out_w: u32,
    pub off_out_b: u32,
}

/// Scalar element offsets (in `f32` units) into `FidNetPackedWeightsF32::data` for a single block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FidNetBlockOffsets {
    pub filter_w: u32,
    pub filter_b: u32,
    pub gate_w: u32,
    pub gate_b: u32,
    pub res_w: u32,
    pub res_b: u32,
    pub skip_w: u32,
    pub skip_b: u32,
}

impl FidNetWeights {
    /// Allocate zero-initialized weights matching `config`.
    pub fn zeros(config: &FidNetConfig) -> Result<Self, FidNetError> {
        config.validate()?;
        let in_ch = config.in_channels();
        let c = config.residual_channels;
        let s = config.skip_channels;
        let k = config.kernel_size;
        let mut blocks = Vec::with_capacity(config.num_blocks());
        for _ in 0..config.num_blocks() {
            blocks.push(FidNetBlockWeights {
                filter_w: vec![0.0; c * c * k],
                filter_b: vec![0.0; c],
                gate_w: vec![0.0; c * c * k],
                gate_b: vec![0.0; c],
                res_w: vec![0.0; c * c],
                res_b: vec![0.0; c],
                skip_w: vec![0.0; s * c],
                skip_b: vec![0.0; s],
            });
        }
        Ok(Self {
            stem_w: vec![0.0; c * in_ch],
            stem_b: vec![0.0; c],
            blocks,
            head1_w: vec![0.0; s * s],
            head1_b: vec![0.0; s],
            out_w: vec![0.0; 2 * s],
            out_b: vec![0.0; 2],
        })
    }

    /// Deterministic Xavier/Glorot initialization suitable for stable WaveNet optimization.
    pub fn init_xavier(config: &FidNetConfig, seed: u64) -> Result<Self, FidNetError> {
        config.validate()?;
        let mut rng = Rng::new(seed);
        let in_ch = config.in_channels();
        let c = config.residual_channels;
        let s = config.skip_channels;
        let k = config.kernel_size;
        let n_blocks = config.num_blocks();

        let mut sample_vec =
            |len: usize, sd: f64| -> Vec<f64> { (0..len).map(|_| rng.normal(0.0, sd)).collect() };

        let stem_sd = (2.0 / (in_ch + c) as f64).sqrt();
        let conv_sd = (1.0 / (c * k) as f64).sqrt();
        let res_sd = (1.0 / (2 * c * n_blocks.max(1)) as f64).sqrt();
        let skip_sd = (1.0 / ((c + s) * n_blocks.max(1)) as f64).sqrt();
        let head1_sd = (2.0 / (s + s) as f64).sqrt();
        let out_sd = if config.output_residual {
            0.1 * (1.0 / (s + 2) as f64).sqrt()
        } else {
            (2.0 / (s + 2) as f64).sqrt()
        };

        let mut blocks = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            blocks.push(FidNetBlockWeights {
                filter_w: sample_vec(c * c * k, conv_sd),
                filter_b: vec![0.0; c],
                gate_w: sample_vec(c * c * k, conv_sd),
                gate_b: vec![0.0; c],
                res_w: sample_vec(c * c, res_sd),
                res_b: vec![0.0; c],
                skip_w: sample_vec(s * c, skip_sd),
                skip_b: vec![0.0; s],
            });
        }

        Ok(Self {
            stem_w: sample_vec(c * in_ch, stem_sd),
            stem_b: vec![0.0; c],
            blocks,
            head1_w: sample_vec(s * s, head1_sd),
            head1_b: vec![0.0; s],
            out_w: sample_vec(2 * s, out_sd),
            out_b: vec![0.0; 2],
        })
    }

    /// Validate that every weight tensor has exact shape matching `config` and contains only finite values.
    pub fn validate(&self, config: &FidNetConfig) -> Result<(), FidNetError> {
        config.validate()?;
        let in_ch = config.in_channels();
        let c = config.residual_channels;
        let s = config.skip_channels;
        let k = config.kernel_size;

        let check_slice = |name: &str, v: &[f64], expected: usize| -> Result<(), FidNetError> {
            if v.len() != expected {
                return Err(FidNetError::InvalidWeights(format!(
                    "{name} length {} does not match expected {expected}",
                    v.len()
                )));
            }
            if let Some((idx, &val)) = v.iter().enumerate().find(|(_, x)| !x.is_finite()) {
                return Err(FidNetError::InvalidWeights(format!(
                    "{name}[{idx}] is non-finite ({val})"
                )));
            }
            Ok(())
        };

        check_slice("stem_w", &self.stem_w, c * in_ch)?;
        check_slice("stem_b", &self.stem_b, c)?;
        if self.blocks.len() != config.num_blocks() {
            return Err(FidNetError::InvalidWeights(format!(
                "blocks length {} does not match dilations length {}",
                self.blocks.len(),
                config.num_blocks()
            )));
        }
        for (bi, blk) in self.blocks.iter().enumerate() {
            check_slice(&format!("blocks[{bi}].filter_w"), &blk.filter_w, c * c * k)?;
            check_slice(&format!("blocks[{bi}].filter_b"), &blk.filter_b, c)?;
            check_slice(&format!("blocks[{bi}].gate_w"), &blk.gate_w, c * c * k)?;
            check_slice(&format!("blocks[{bi}].gate_b"), &blk.gate_b, c)?;
            check_slice(&format!("blocks[{bi}].res_w"), &blk.res_w, c * c)?;
            check_slice(&format!("blocks[{bi}].res_b"), &blk.res_b, c)?;
            check_slice(&format!("blocks[{bi}].skip_w"), &blk.skip_w, s * c)?;
            check_slice(&format!("blocks[{bi}].skip_b"), &blk.skip_b, s)?;
        }
        check_slice("head1_w", &self.head1_w, s * s)?;
        check_slice("head1_b", &self.head1_b, s)?;
        check_slice("out_w", &self.out_w, 2 * s)?;
        check_slice("out_b", &self.out_b, 2)?;
        Ok(())
    }

    /// Pack all weights into a contiguous `f32` buffer with exact offsets for WebGPU execution.
    pub fn pack_f32(&self, config: &FidNetConfig) -> Result<FidNetPackedWeightsF32, FidNetError> {
        self.validate(config)?;
        let total = config.parameter_count()?;
        let mut data = Vec::with_capacity(total);

        let push_slice = |dst: &mut Vec<f32>, src: &[f64]| -> Result<u32, FidNetError> {
            let off = u32::try_from(dst.len()).map_err(|_| {
                FidNetError::InvalidWeights("packed f32 weight offset exceeds u32::MAX".into())
            })?;
            for &v in src {
                let vf = v as f32;
                if !vf.is_finite() {
                    return Err(FidNetError::InvalidWeights(format!(
                        "weight {v} overflows f32 during GPU packing"
                    )));
                }
                dst.push(vf);
            }
            Ok(off)
        };

        let off_stem_w = push_slice(&mut data, &self.stem_w)?;
        let off_stem_b = push_slice(&mut data, &self.stem_b)?;
        let mut block_offsets = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            block_offsets.push(FidNetBlockOffsets {
                filter_w: push_slice(&mut data, &blk.filter_w)?,
                filter_b: push_slice(&mut data, &blk.filter_b)?,
                gate_w: push_slice(&mut data, &blk.gate_w)?,
                gate_b: push_slice(&mut data, &blk.gate_b)?,
                res_w: push_slice(&mut data, &blk.res_w)?,
                res_b: push_slice(&mut data, &blk.res_b)?,
                skip_w: push_slice(&mut data, &blk.skip_w)?,
                skip_b: push_slice(&mut data, &blk.skip_b)?,
            });
        }
        let off_head1_w = push_slice(&mut data, &self.head1_w)?;
        let off_head1_b = push_slice(&mut data, &self.head1_b)?;
        let off_out_w = push_slice(&mut data, &self.out_w)?;
        let off_out_b = push_slice(&mut data, &self.out_b)?;

        Ok(FidNetPackedWeightsF32 {
            data,
            off_stem_w,
            off_stem_b,
            block_offsets,
            off_head1_w,
            off_head1_b,
            off_out_w,
            off_out_b,
        })
    }

    /// Flatten all scalar parameters in canonical order into `Vec<f64>`.
    pub fn to_flat_vec(&self) -> Vec<f64> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.stem_w);
        out.extend_from_slice(&self.stem_b);
        for blk in &self.blocks {
            out.extend_from_slice(&blk.filter_w);
            out.extend_from_slice(&blk.filter_b);
            out.extend_from_slice(&blk.gate_w);
            out.extend_from_slice(&blk.gate_b);
            out.extend_from_slice(&blk.res_w);
            out.extend_from_slice(&blk.res_b);
            out.extend_from_slice(&blk.skip_w);
            out.extend_from_slice(&blk.skip_b);
        }
        out.extend_from_slice(&self.head1_w);
        out.extend_from_slice(&self.head1_b);
        out.extend_from_slice(&self.out_w);
        out.extend_from_slice(&self.out_b);
        out
    }

    /// Reconstruct `FidNetWeights` from a canonical flat `f64` vector.
    pub fn from_flat_slice(config: &FidNetConfig, flat: &[f64]) -> Result<Self, FidNetError> {
        config.validate()?;
        let expected = config.parameter_count()?;
        if flat.len() != expected {
            return Err(FidNetError::InvalidWeights(format!(
                "flat weight slice length {} does not match expected {expected}",
                flat.len()
            )));
        }
        let in_ch = config.in_channels();
        let c = config.residual_channels;
        let s = config.skip_channels;
        let k = config.kernel_size;
        let mut cursor = 0usize;
        let mut take = |n: usize| -> Result<Vec<f64>, FidNetError> {
            let end = cursor
                .checked_add(n)
                .ok_or_else(|| FidNetError::InvalidWeights("weight cursor overflow".into()))?;
            let slice = flat.get(cursor..end).ok_or_else(|| {
                FidNetError::InvalidWeights("unexpected end of flat weights".into())
            })?;
            cursor = end;
            Ok(slice.to_vec())
        };

        let stem_w = take(c * in_ch)?;
        let stem_b = take(c)?;
        let mut blocks = Vec::with_capacity(config.num_blocks());
        for _ in 0..config.num_blocks() {
            blocks.push(FidNetBlockWeights {
                filter_w: take(c * c * k)?,
                filter_b: take(c)?,
                gate_w: take(c * c * k)?,
                gate_b: take(c)?,
                res_w: take(c * c)?,
                res_b: take(c)?,
                skip_w: take(s * c)?,
                skip_b: take(s)?,
            });
        }
        let head1_w = take(s * s)?;
        let head1_b = take(s)?;
        let out_w = take(2 * s)?;
        let out_b = take(2)?;

        let weights = Self {
            stem_w,
            stem_b,
            blocks,
            head1_w,
            head1_b,
            out_w,
            out_b,
        };
        weights.validate(config)?;
        Ok(weights)
    }
}

/// Optional provenance metadata embedded in a `FIDNET-V1` weight bundle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FidNetBundleMetadata {
    pub description: String,
    pub license: String,
    pub provenance: String,
}

impl Default for FidNetBundleMetadata {
    fn default() -> Self {
        Self {
            description: "FERRIC pure-Rust FID-Net / FID-Net-2 WaveNet model".to_string(),
            license: "MIT".to_string(),
            provenance: "Clean-room implementation of FID-Net / FID-Net-2 (Karunanithy & Hansen 2021, 2023; https://github.com/gogulan-k/FID-Net, MIT)".to_string(),
        }
    }
}

/// Weights for a single WaveNet dilated gated-activation block in canonical IEEE-754 `f32`
/// bundle precision (0-ULP JSON and binary round-trip, identical to WebGPU execution precision).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FidNetBlockWeightsF32 {
    pub filter_w: Vec<f32>,
    pub filter_b: Vec<f32>,
    pub gate_w: Vec<f32>,
    pub gate_b: Vec<f32>,
    pub res_w: Vec<f32>,
    pub res_b: Vec<f32>,
    pub skip_w: Vec<f32>,
    pub skip_b: Vec<f32>,
}

/// Complete serialized parameter set in canonical IEEE-754 `f32` bundle precision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FidNetWeightsF32 {
    pub stem_w: Vec<f32>,
    pub stem_b: Vec<f32>,
    pub blocks: Vec<FidNetBlockWeightsF32>,
    pub head1_w: Vec<f32>,
    pub head1_b: Vec<f32>,
    pub out_w: Vec<f32>,
    pub out_b: Vec<f32>,
}

impl FidNetWeightsF32 {
    /// Narrow validated `f64` training weights into canonical `f32` bundle weights.
    pub fn from_f64(config: &FidNetConfig, src: &FidNetWeights) -> Result<Self, FidNetError> {
        src.validate(config)?;
        let narrow = |v: &[f64], name: &str| -> Result<Vec<f32>, FidNetError> {
            let mut out = Vec::with_capacity(v.len());
            for (idx, &x) in v.iter().enumerate() {
                let xf = x as f32;
                if !xf.is_finite() {
                    return Err(FidNetError::InvalidWeights(format!(
                        "{name}[{idx}] ({x}) overflows f32"
                    )));
                }
                out.push(if xf == 0.0 { 0.0 } else { xf });
            }
            Ok(out)
        };
        let mut blocks = Vec::with_capacity(src.blocks.len());
        for (bi, blk) in src.blocks.iter().enumerate() {
            blocks.push(FidNetBlockWeightsF32 {
                filter_w: narrow(&blk.filter_w, &format!("blocks[{bi}].filter_w"))?,
                filter_b: narrow(&blk.filter_b, &format!("blocks[{bi}].filter_b"))?,
                gate_w: narrow(&blk.gate_w, &format!("blocks[{bi}].gate_w"))?,
                gate_b: narrow(&blk.gate_b, &format!("blocks[{bi}].gate_b"))?,
                res_w: narrow(&blk.res_w, &format!("blocks[{bi}].res_w"))?,
                res_b: narrow(&blk.res_b, &format!("blocks[{bi}].res_b"))?,
                skip_w: narrow(&blk.skip_w, &format!("blocks[{bi}].skip_w"))?,
                skip_b: narrow(&blk.skip_b, &format!("blocks[{bi}].skip_b"))?,
            });
        }
        Ok(Self {
            stem_w: narrow(&src.stem_w, "stem_w")?,
            stem_b: narrow(&src.stem_b, "stem_b")?,
            blocks,
            head1_w: narrow(&src.head1_w, "head1_w")?,
            head1_b: narrow(&src.head1_b, "head1_b")?,
            out_w: narrow(&src.out_w, "out_w")?,
            out_b: narrow(&src.out_b, "out_b")?,
        })
    }

    /// Widen validated `f32` bundle weights into `f64` reference weights.
    pub fn to_f64(&self, config: &FidNetConfig) -> Result<FidNetWeights, FidNetError> {
        self.validate(config)?;
        let widen = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| f64::from(x)).collect() };
        let blocks = self
            .blocks
            .iter()
            .map(|blk| FidNetBlockWeights {
                filter_w: widen(&blk.filter_w),
                filter_b: widen(&blk.filter_b),
                gate_w: widen(&blk.gate_w),
                gate_b: widen(&blk.gate_b),
                res_w: widen(&blk.res_w),
                res_b: widen(&blk.res_b),
                skip_w: widen(&blk.skip_w),
                skip_b: widen(&blk.skip_b),
            })
            .collect();
        Ok(FidNetWeights {
            stem_w: widen(&self.stem_w),
            stem_b: widen(&self.stem_b),
            blocks,
            head1_w: widen(&self.head1_w),
            head1_b: widen(&self.head1_b),
            out_w: widen(&self.out_w),
            out_b: widen(&self.out_b),
        })
    }

    /// Validate that every `f32` weight tensor matches `config` dimensions and is finite.
    pub fn validate(&self, config: &FidNetConfig) -> Result<(), FidNetError> {
        config.validate()?;
        let in_ch = config.in_channels();
        let c = config.residual_channels;
        let s = config.skip_channels;
        let k = config.kernel_size;

        let check_slice = |name: &str, v: &[f32], expected: usize| -> Result<(), FidNetError> {
            if v.len() != expected {
                return Err(FidNetError::InvalidWeights(format!(
                    "{name} length {} does not match expected {expected}",
                    v.len()
                )));
            }
            if let Some((idx, &val)) = v.iter().enumerate().find(|(_, x)| !x.is_finite()) {
                return Err(FidNetError::InvalidWeights(format!(
                    "{name}[{idx}] is non-finite ({val})"
                )));
            }
            Ok(())
        };

        check_slice("stem_w", &self.stem_w, c * in_ch)?;
        check_slice("stem_b", &self.stem_b, c)?;
        if self.blocks.len() != config.num_blocks() {
            return Err(FidNetError::InvalidWeights(format!(
                "blocks length {} does not match dilations length {}",
                self.blocks.len(),
                config.num_blocks()
            )));
        }
        for (bi, blk) in self.blocks.iter().enumerate() {
            check_slice(&format!("blocks[{bi}].filter_w"), &blk.filter_w, c * c * k)?;
            check_slice(&format!("blocks[{bi}].filter_b"), &blk.filter_b, c)?;
            check_slice(&format!("blocks[{bi}].gate_w"), &blk.gate_w, c * c * k)?;
            check_slice(&format!("blocks[{bi}].gate_b"), &blk.gate_b, c)?;
            check_slice(&format!("blocks[{bi}].res_w"), &blk.res_w, c * c)?;
            check_slice(&format!("blocks[{bi}].res_b"), &blk.res_b, c)?;
            check_slice(&format!("blocks[{bi}].skip_w"), &blk.skip_w, s * c)?;
            check_slice(&format!("blocks[{bi}].skip_b"), &blk.skip_b, s)?;
        }
        check_slice("head1_w", &self.head1_w, s * s)?;
        check_slice("head1_b", &self.head1_b, s)?;
        check_slice("out_w", &self.out_w, 2 * s)?;
        check_slice("out_b", &self.out_b, 2)?;
        Ok(())
    }

    /// Flatten all `f32` bundle parameters in canonical order.
    pub fn to_flat_vec(&self) -> Vec<f32> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.stem_w);
        out.extend_from_slice(&self.stem_b);
        for blk in &self.blocks {
            out.extend_from_slice(&blk.filter_w);
            out.extend_from_slice(&blk.filter_b);
            out.extend_from_slice(&blk.gate_w);
            out.extend_from_slice(&blk.gate_b);
            out.extend_from_slice(&blk.res_w);
            out.extend_from_slice(&blk.res_b);
            out.extend_from_slice(&blk.skip_w);
            out.extend_from_slice(&blk.skip_b);
        }
        out.extend_from_slice(&self.head1_w);
        out.extend_from_slice(&self.head1_b);
        out.extend_from_slice(&self.out_w);
        out.extend_from_slice(&self.out_b);
        out
    }

    /// Reconstruct `FidNetWeightsF32` from a canonical flat `f32` slice.
    pub fn from_flat_slice(config: &FidNetConfig, flat: &[f32]) -> Result<Self, FidNetError> {
        config.validate()?;
        let expected = config.parameter_count()?;
        if flat.len() != expected {
            return Err(FidNetError::InvalidWeights(format!(
                "flat f32 weight slice length {} does not match expected {expected}",
                flat.len()
            )));
        }
        let in_ch = config.in_channels();
        let c = config.residual_channels;
        let s = config.skip_channels;
        let k = config.kernel_size;
        let mut cursor = 0usize;
        let mut take = |n: usize| -> Result<Vec<f32>, FidNetError> {
            let end = cursor
                .checked_add(n)
                .ok_or_else(|| FidNetError::InvalidWeights("weight cursor overflow".into()))?;
            let slice = flat.get(cursor..end).ok_or_else(|| {
                FidNetError::InvalidWeights("unexpected end of flat weights".into())
            })?;
            cursor = end;
            Ok(slice.to_vec())
        };

        let stem_w = take(c * in_ch)?;
        let stem_b = take(c)?;
        let mut blocks = Vec::with_capacity(config.num_blocks());
        for _ in 0..config.num_blocks() {
            blocks.push(FidNetBlockWeightsF32 {
                filter_w: take(c * c * k)?,
                filter_b: take(c)?,
                gate_w: take(c * c * k)?,
                gate_b: take(c)?,
                res_w: take(c * c)?,
                res_b: take(c)?,
                skip_w: take(s * c)?,
                skip_b: take(s)?,
            });
        }
        let head1_w = take(s * s)?;
        let head1_b = take(s)?;
        let out_w = take(2 * s)?;
        let out_b = take(2)?;

        let weights = Self {
            stem_w,
            stem_b,
            blocks,
            head1_w,
            head1_b,
            out_w,
            out_b,
        };
        weights.validate(config)?;
        Ok(weights)
    }
}

/// Strictly versioned, SHA-256-authenticated `FIDNET-V1` weight bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FidNetWeightBundle {
    pub format_version: String,
    pub config: FidNetConfig,
    #[serde(default)]
    pub metadata: FidNetBundleMetadata,
    pub weights: FidNetWeightsF32,
    /// Lowercase 64-character hex FIPS 180-4 SHA-256 digest over canonical header + config + IEEE-754 `f32` weights.
    pub sha256: String,
}

fn task_tag_u8(task: FidNetTask) -> u8 {
    match task {
        FidNetTask::NusReconstruction => 1,
        FidNetTask::VirtualDecoupling => 2,
        FidNetTask::Denoising => 3,
    }
}

fn task_from_tag_u8(tag: u8) -> Result<FidNetTask, FidNetError> {
    match tag {
        1 => Ok(FidNetTask::NusReconstruction),
        2 => Ok(FidNetTask::VirtualDecoupling),
        3 => Ok(FidNetTask::Denoising),
        other => Err(FidNetError::InvalidConfig(format!(
            "unknown FidNetTask tag {other}"
        ))),
    }
}

/// Compute canonical FIPS 180-4 SHA-256 digest over format version, architecture config, and IEEE-754 `f32` weights.
pub fn compute_fidnet_bundle_sha256(
    format_version: &str,
    config: &FidNetConfig,
    weights: &FidNetWeightsF32,
) -> Result<String, FidNetError> {
    weights.validate(config)?;
    let mut payload = Vec::new();
    payload.extend_from_slice(format_version.as_bytes());
    payload.push(0);
    payload.push(task_tag_u8(config.task));
    payload.push(u8::from(config.use_mask_channel));
    payload.push(u8::from(config.use_coupling_channel));
    payload.push(u8::from(config.output_residual));
    payload.extend_from_slice(&(config.residual_channels as u32).to_le_bytes());
    payload.extend_from_slice(&(config.skip_channels as u32).to_le_bytes());
    payload.extend_from_slice(&(config.kernel_size as u32).to_le_bytes());
    payload.extend_from_slice(&(config.dilations.len() as u32).to_le_bytes());
    for &d in &config.dilations {
        payload.extend_from_slice(&(d as u32).to_le_bytes());
    }
    for val in weights.to_flat_vec() {
        // Canonicalize -0.0 to +0.0 for deterministic cross-platform hashing.
        let canon = if val == 0.0 { 0.0 } else { val };
        payload.extend_from_slice(&canon.to_bits().to_le_bytes());
    }
    Ok(sha256_hex(&payload))
}

impl FidNetWeightBundle {
    /// Construct a verified `FIDNET-V1` bundle from `config` and `f64` training weights.
    pub fn new(
        config: FidNetConfig,
        weights: &FidNetWeights,
        metadata: FidNetBundleMetadata,
    ) -> Result<Self, FidNetError> {
        let weights_f32 = FidNetWeightsF32::from_f64(&config, weights)?;
        let sha256 =
            compute_fidnet_bundle_sha256(FIDNET_BUNDLE_FORMAT_VERSION, &config, &weights_f32)?;
        Ok(Self {
            format_version: FIDNET_BUNDLE_FORMAT_VERSION.to_string(),
            config,
            metadata,
            weights: weights_f32,
            sha256,
        })
    }

    /// Verify version, architecture caps, weight shapes/finiteness, and SHA-256 digest.
    pub fn verify(&self) -> Result<(), FidNetError> {
        if self.format_version != FIDNET_BUNDLE_FORMAT_VERSION {
            return Err(FidNetError::UnsupportedFormatVersion(
                self.format_version.clone(),
            ));
        }
        let actual =
            compute_fidnet_bundle_sha256(&self.format_version, &self.config, &self.weights)?;
        if actual != self.sha256 {
            return Err(FidNetError::ChecksumMismatch {
                expected: self.sha256.clone(),
                actual,
            });
        }
        Ok(())
    }

    /// Serialize bundle to JSON (`FIDNET-V1`).
    pub fn to_json(&self) -> Result<String, FidNetError> {
        self.verify()?;
        serde_json::to_string_pretty(self)
            .map_err(|e| FidNetError::SerializationError(e.to_string()))
    }

    /// Deserialize and strictly verify a `FIDNET-V1` JSON bundle.
    pub fn from_json(json: &str) -> Result<Self, FidNetError> {
        let bundle: Self = serde_json::from_str(json)
            .map_err(|e| FidNetError::SerializationError(e.to_string()))?;
        bundle.verify()?;
        Ok(bundle)
    }

    /// Serialize bundle to compact deterministic `FIDNET-V1` binary bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FidNetError> {
        self.verify()?;
        let flat = self.weights.to_flat_vec();
        let mut out =
            Vec::with_capacity(64 + self.config.dilations.len() * 4 + flat.len() * 4 + 64);
        out.extend_from_slice(FIDNET_BINARY_MAGIC);
        out.push(task_tag_u8(self.config.task));
        out.push(u8::from(self.config.use_mask_channel));
        out.push(u8::from(self.config.use_coupling_channel));
        out.push(u8::from(self.config.output_residual));
        out.extend_from_slice(&(self.config.residual_channels as u32).to_le_bytes());
        out.extend_from_slice(&(self.config.skip_channels as u32).to_le_bytes());
        out.extend_from_slice(&(self.config.kernel_size as u32).to_le_bytes());
        out.extend_from_slice(&(self.config.dilations.len() as u32).to_le_bytes());
        for &d in &self.config.dilations {
            out.extend_from_slice(&(d as u32).to_le_bytes());
        }
        out.extend_from_slice(&(flat.len() as u64).to_le_bytes());
        for &v in &flat {
            let canon = if v == 0.0 { 0.0 } else { v };
            out.extend_from_slice(&canon.to_bits().to_le_bytes());
        }
        out.extend_from_slice(self.sha256.as_bytes());
        Ok(out)
    }

    /// Deserialize and strictly verify a compact `FIDNET-V1` binary bundle.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FidNetError> {
        let mut pos = 0usize;
        let read_exact = |p: &mut usize, n: usize| -> Result<&[u8], FidNetError> {
            let end = p
                .checked_add(n)
                .ok_or_else(|| FidNetError::SerializationError("binary cursor overflow".into()))?;
            let slice = bytes
                .get(*p..end)
                .ok_or_else(|| FidNetError::SerializationError("truncated binary bundle".into()))?;
            *p = end;
            Ok(slice)
        };

        let magic = read_exact(&mut pos, 8)?;
        if magic != FIDNET_BINARY_MAGIC {
            return Err(FidNetError::UnsupportedFormatVersion(format!(
                "invalid binary magic {:?}",
                magic
            )));
        }
        let flags = read_exact(&mut pos, 4)?;
        let task = task_from_tag_u8(flags[0])?;
        let use_mask_channel = match flags[1] {
            0 => false,
            1 => true,
            v => return Err(FidNetError::InvalidConfig(format!("invalid mask flag {v}"))),
        };
        let use_coupling_channel = match flags[2] {
            0 => false,
            1 => true,
            v => {
                return Err(FidNetError::InvalidConfig(format!(
                    "invalid coupling flag {v}"
                )))
            }
        };
        let output_residual = match flags[3] {
            0 => false,
            1 => true,
            v => {
                return Err(FidNetError::InvalidConfig(format!(
                    "invalid output_residual flag {v}"
                )))
            }
        };

        let read_u32 = |p: &mut usize| -> Result<u32, FidNetError> {
            let s = read_exact(p, 4)?;
            Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        };
        let read_u64 = |p: &mut usize| -> Result<u64, FidNetError> {
            let s = read_exact(p, 8)?;
            Ok(u64::from_le_bytes([
                s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
            ]))
        };

        let residual_channels = read_u32(&mut pos)? as usize;
        let skip_channels = read_u32(&mut pos)? as usize;
        let kernel_size = read_u32(&mut pos)? as usize;
        let num_blocks = read_u32(&mut pos)? as usize;
        if num_blocks == 0 || num_blocks > MAX_FIDNET_BLOCKS {
            return Err(FidNetError::InvalidConfig(format!(
                "num_blocks={num_blocks} out of range 1..={MAX_FIDNET_BLOCKS}"
            )));
        }
        let mut dilations = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            dilations.push(read_u32(&mut pos)? as usize);
        }

        let config = FidNetConfig {
            task,
            use_mask_channel,
            use_coupling_channel,
            residual_channels,
            skip_channels,
            kernel_size,
            dilations,
            output_residual,
        };
        config.validate()?;

        let n_params_u64 = read_u64(&mut pos)?;
        let expected_params = config.parameter_count()?;
        if n_params_u64 != expected_params as u64 {
            return Err(FidNetError::InvalidWeights(format!(
                "binary bundle declares {n_params_u64} params, expected {expected_params}"
            )));
        }

        let mut flat = Vec::with_capacity(expected_params);
        for _ in 0..expected_params {
            let bits = read_u32(&mut pos)?;
            let val = f32::from_bits(bits);
            if !val.is_finite() {
                return Err(FidNetError::InvalidWeights(
                    "non-finite f32 weight in binary bundle".into(),
                ));
            }
            flat.push(val);
        }

        let sha_bytes = read_exact(&mut pos, 64)?;
        if pos != bytes.len() {
            return Err(FidNetError::SerializationError(format!(
                "trailing {} bytes after FID-Net binary bundle",
                bytes.len() - pos
            )));
        }
        let sha256 = std::str::from_utf8(sha_bytes)
            .map_err(|e| FidNetError::SerializationError(format!("invalid UTF-8 sha256: {e}")))?
            .to_string();

        let weights = FidNetWeightsF32::from_flat_slice(&config, &flat)?;
        let bundle = Self {
            format_version: FIDNET_BUNDLE_FORMAT_VERSION.to_string(),
            config,
            metadata: FidNetBundleMetadata::default(),
            weights,
            sha256,
        };
        bundle.verify()?;
        Ok(bundle)
    }
}

// ============================================================================
// Overflow-Safe Activations (Matched Identically by WebGPU fidnet.wgsl)
// ============================================================================

/// Overflow-safe clamped `tanh` matching WebGPU `fidnet.wgsl` (`exp` only called on `<= 0`).
#[inline]
pub fn safe_tanh(x: f64) -> f64 {
    let xc = x.clamp(-10.0, 10.0);
    if xc >= 0.0 {
        let e = (-2.0 * xc).exp();
        (1.0 - e) / (1.0 + e)
    } else {
        let e = (2.0 * xc).exp();
        (e - 1.0) / (e + 1.0)
    }
}

/// Exact derivative of `safe_tanh(x)` given `t = safe_tanh(x)`.
#[inline]
fn safe_tanh_deriv(x: f64, t: f64) -> f64 {
    if x.abs() >= 10.0 {
        0.0
    } else {
        1.0 - t * t
    }
}

/// Overflow-safe clamped logistic sigmoid matching WebGPU `fidnet.wgsl` (`exp` only called on `<= 0`).
#[inline]
pub fn safe_sigmoid(x: f64) -> f64 {
    let xc = x.clamp(-20.0, 20.0);
    if xc >= 0.0 {
        1.0 / (1.0 + (-xc).exp())
    } else {
        let e = xc.exp();
        e / (1.0 + e)
    }
}

/// Exact derivative of `safe_sigmoid(x)` given `s = safe_sigmoid(x)`.
#[inline]
fn safe_sigmoid_deriv(x: f64, s: f64) -> f64 {
    if x.abs() >= 20.0 {
        0.0
    } else {
        s * (1.0 - s)
    }
}

/// Build the normalized scalar-coupling phase-modulation conditioning channel `cos(pi * J * t)`
/// for virtual homonuclear decoupling (`t = k / sw_hz`).
pub fn build_coupling_channel(len: usize, j_hz: f64, sw_hz: f64) -> Result<Vec<f64>, FidNetError> {
    if len == 0 {
        return Err(FidNetError::EmptyInput(
            "coupling channel length must be > 0".into(),
        ));
    }
    if len > MAX_FIDNET_LENGTH {
        return Err(FidNetError::OversizedFid {
            len,
            max_len: MAX_FIDNET_LENGTH,
        });
    }
    if !j_hz.is_finite() || !sw_hz.is_finite() || sw_hz <= 0.0 {
        return Err(FidNetError::NonFiniteInput(format!(
            "invalid coupling parameters j_hz={j_hz}, sw_hz={sw_hz}"
        )));
    }
    let dt = 1.0 / sw_hz;
    Ok((0..len)
        .map(|k| (PI * j_hz * (k as f64) * dt).cos())
        .collect())
}

// ============================================================================
// Core FID-Net Model, Forward Pass & Exact Analytical Backpropagation
// ============================================================================

/// Complete WaveNet / dilated-convolution `FID-Net` / `FID-Net-2` model.
#[derive(Clone, Debug, PartialEq)]
pub struct FidNetModel {
    pub config: FidNetConfig,
    pub weights: FidNetWeights,
}

/// Intermediate activation cache for a single block (used by exact backpropagation).
#[derive(Clone, Debug)]
pub struct FidNetBlockCache {
    pub f_pre: Vec<f64>,
    pub g_pre: Vec<f64>,
    pub tf: Vec<f64>,
    pub sg: Vec<f64>,
    pub z: Vec<f64>,
}

/// Full forward-pass activation cache for exact analytical backpropagation.
#[derive(Clone, Debug)]
pub struct FidNetCache {
    pub len: usize,
    pub x_in: Vec<f64>,
    /// Residual stream states `h[0..=num_blocks]`, each `[C * T]`.
    pub h_states: Vec<Vec<f64>>,
    pub blocks: Vec<FidNetBlockCache>,
    pub skip_sum: Vec<f64>,
    pub skip_act: Vec<f64>,
    pub h1_pre: Vec<f64>,
    pub h1_act: Vec<f64>,
    pub output: Vec<Complex64>,
}

/// Analytical parameter gradients matching the exact structure of `FidNetWeights`.
pub type FidNetGradients = FidNetWeights;

impl FidNetModel {
    /// Construct and validate a `FidNetModel` from `config` and `weights`.
    pub fn new(config: FidNetConfig, weights: FidNetWeights) -> Result<Self, FidNetError> {
        weights.validate(&config)?;
        Ok(Self { config, weights })
    }

    /// Initialize a new `FidNetModel` with deterministic Xavier/Glorot weights.
    pub fn init_random(config: FidNetConfig, seed: u64) -> Result<Self, FidNetError> {
        let weights = FidNetWeights::init_xavier(&config, seed)?;
        Ok(Self { config, weights })
    }

    /// Construct a `FidNetModel` from a verified `FIDNET-V1` weight bundle.
    pub fn from_bundle(bundle: &FidNetWeightBundle) -> Result<Self, FidNetError> {
        bundle.verify()?;
        let weights = bundle.weights.to_f64(&bundle.config)?;
        Ok(Self {
            config: bundle.config.clone(),
            weights,
        })
    }

    /// Export this model as a verified `FIDNET-V1` weight bundle with SHA-256 integrity digest.
    pub fn to_bundle(
        &self,
        metadata: FidNetBundleMetadata,
    ) -> Result<FidNetWeightBundle, FidNetError> {
        FidNetWeightBundle::new(self.config.clone(), &self.weights, metadata)
    }

    /// Assemble channel-major `[in_ch * T]` input features from a complex FID and optional
    /// NUS mask / coupling conditioning channels, validating length and finiteness.
    pub fn build_input_features(
        &self,
        fid: &[Complex64],
        nus_mask: Option<&[bool]>,
        coupling_channel: Option<&[f64]>,
    ) -> Result<Vec<f64>, FidNetError> {
        let t_len = fid.len();
        if t_len == 0 {
            return Err(FidNetError::EmptyInput("input FID is empty".into()));
        }
        if t_len > MAX_FIDNET_LENGTH {
            return Err(FidNetError::OversizedFid {
                len: t_len,
                max_len: MAX_FIDNET_LENGTH,
            });
        }
        for (i, &[re, im]) in fid.iter().enumerate() {
            if !re.is_finite() || !im.is_finite() {
                return Err(FidNetError::NonFiniteInput(format!(
                    "FID sample [{i}] = ({re}, {im}) is non-finite"
                )));
            }
        }
        if self.config.use_mask_channel {
            if let Some(mask) = nus_mask {
                if mask.len() != t_len {
                    return Err(FidNetError::ShapeMismatch(format!(
                        "nus_mask length {} != FID length {t_len}",
                        mask.len()
                    )));
                }
            }
        }
        if self.config.use_coupling_channel {
            if let Some(cc) = coupling_channel {
                if cc.len() != t_len {
                    return Err(FidNetError::ShapeMismatch(format!(
                        "coupling_channel length {} != FID length {t_len}",
                        cc.len()
                    )));
                }
                if let Some((i, &v)) = cc.iter().enumerate().find(|(_, x)| !x.is_finite()) {
                    return Err(FidNetError::NonFiniteInput(format!(
                        "coupling_channel[{i}] = {v} is non-finite"
                    )));
                }
            }
        }

        let in_ch = self.config.in_channels();
        let mut x_in = vec![0.0; in_ch * t_len];
        for t in 0..t_len {
            x_in[t] = fid[t][0];
            x_in[t_len + t] = fid[t][1];
        }
        let mut ch = 2usize;
        if self.config.use_mask_channel {
            let base = ch * t_len;
            match nus_mask {
                Some(mask) => {
                    for (t, &sampled) in mask.iter().enumerate() {
                        x_in[base + t] = if sampled { 1.0 } else { 0.0 };
                    }
                }
                None => {
                    for t in 0..t_len {
                        x_in[base + t] = 1.0;
                    }
                }
            }
            ch += 1;
        }
        if self.config.use_coupling_channel {
            let base = ch * t_len;
            match coupling_channel {
                Some(cc) => {
                    x_in[base..base + t_len].copy_from_slice(cc);
                }
                None => {
                    for t in 0..t_len {
                        x_in[base + t] = 1.0;
                    }
                }
            }
        }
        Ok(x_in)
    }

    /// Execute the exact `f64` forward pass on pre-packed channel-major features `[in_ch * T]`,
    /// returning the reconstructed complex FID and full activation cache for backpropagation.
    pub fn forward_features_with_cache(
        &self,
        x_in: &[f64],
        t_len: usize,
    ) -> Result<FidNetCache, FidNetError> {
        let in_ch = self.config.in_channels();
        if t_len == 0 {
            return Err(FidNetError::EmptyInput("t_len must be > 0".into()));
        }
        if t_len > MAX_FIDNET_LENGTH {
            return Err(FidNetError::OversizedFid {
                len: t_len,
                max_len: MAX_FIDNET_LENGTH,
            });
        }
        if x_in.len() != in_ch * t_len {
            return Err(FidNetError::ShapeMismatch(format!(
                "feature buffer length {} != in_ch ({in_ch}) * t_len ({t_len})",
                x_in.len()
            )));
        }
        if let Some((i, &v)) = x_in.iter().enumerate().find(|(_, x)| !x.is_finite()) {
            return Err(FidNetError::NonFiniteInput(format!(
                "x_in[{i}] = {v} is non-finite"
            )));
        }

        let c = self.config.residual_channels;
        let s = self.config.skip_channels;
        let k = self.config.kernel_size;
        let radius = (k / 2) as isize;
        let n_blocks = self.config.num_blocks();

        // 1. Stem 1x1 projection: [in_ch, T] -> [C, T]
        let mut h0 = vec![0.0; c * t_len];
        for oc in 0..c {
            let bias = self.weights.stem_b[oc];
            let dst = &mut h0[oc * t_len..(oc + 1) * t_len];
            dst.fill(bias);
            for ic in 0..in_ch {
                let w = self.weights.stem_w[oc * in_ch + ic];
                let src = &x_in[ic * t_len..(ic + 1) * t_len];
                for t in 0..t_len {
                    dst[t] += w * src[t];
                }
            }
        }

        let mut h_states = Vec::with_capacity(n_blocks + 1);
        h_states.push(h0);
        let mut block_caches = Vec::with_capacity(n_blocks);
        let mut skip_sum = vec![0.0; s * t_len];

        // 2. Stack of dilated gated-activation WaveNet blocks
        for (bi, (&dilation, blk)) in self
            .config
            .dilations
            .iter()
            .zip(self.weights.blocks.iter())
            .enumerate()
        {
            let h_in = &h_states[bi];
            let d_isize = dilation as isize;
            let mut f_pre = vec![0.0; c * t_len];
            let mut g_pre = vec![0.0; c * t_len];

            for oc in 0..c {
                let fb = blk.filter_b[oc];
                let gb = blk.gate_b[oc];
                let f_row = &mut f_pre[oc * t_len..(oc + 1) * t_len];
                let g_row = &mut g_pre[oc * t_len..(oc + 1) * t_len];
                f_row.fill(fb);
                g_row.fill(gb);

                for ic in 0..c {
                    let src_row = &h_in[ic * t_len..(ic + 1) * t_len];
                    let w_base = (oc * c + ic) * k;
                    for m in 0..k {
                        let wf = blk.filter_w[w_base + m];
                        let wg = blk.gate_w[w_base + m];
                        let offset = (m as isize - radius) * d_isize;
                        let t_start = if offset < 0 { (-offset) as usize } else { 0 };
                        let t_end = if offset > 0 {
                            t_len.saturating_sub(offset as usize)
                        } else {
                            t_len
                        };
                        if t_start < t_end {
                            for t in t_start..t_end {
                                let src_idx = (t as isize + offset) as usize;
                                let v = src_row[src_idx];
                                f_row[t] += wf * v;
                                g_row[t] += wg * v;
                            }
                        }
                    }
                }
            }

            let mut tf = vec![0.0; c * t_len];
            let mut sg = vec![0.0; c * t_len];
            let mut z = vec![0.0; c * t_len];
            for idx in 0..(c * t_len) {
                let t_val = safe_tanh(f_pre[idx]);
                let s_val = safe_sigmoid(g_pre[idx]);
                tf[idx] = t_val;
                sg[idx] = s_val;
                z[idx] = t_val * s_val;
            }

            // Residual 1x1 projection + residual bypass
            let mut h_next = h_in.clone();
            for oc in 0..c {
                let rb = blk.res_b[oc];
                let dst = &mut h_next[oc * t_len..(oc + 1) * t_len];
                for v in dst.iter_mut() {
                    *v += rb;
                }
                for ic in 0..c {
                    let wr = blk.res_w[oc * c + ic];
                    let z_row = &z[ic * t_len..(ic + 1) * t_len];
                    for t in 0..t_len {
                        dst[t] += wr * z_row[t];
                    }
                }
            }

            // Skip 1x1 projection accumulated into skip_sum
            for os in 0..s {
                let sb = blk.skip_b[os];
                let dst = &mut skip_sum[os * t_len..(os + 1) * t_len];
                for v in dst.iter_mut() {
                    *v += sb;
                }
                for ic in 0..c {
                    let ws = blk.skip_w[os * c + ic];
                    let z_row = &z[ic * t_len..(ic + 1) * t_len];
                    for t in 0..t_len {
                        dst[t] += ws * z_row[t];
                    }
                }
            }

            h_states.push(h_next);
            block_caches.push(FidNetBlockCache {
                f_pre,
                g_pre,
                tf,
                sg,
                z,
            });
        }

        // 3. Output head: ReLU(skip_sum) -> 1x1 -> ReLU -> 1x1 linear -> [Re, Im]
        let skip_act: Vec<f64> = skip_sum.iter().map(|&v| v.max(0.0)).collect();
        let mut h1_pre = vec![0.0; s * t_len];
        for os in 0..s {
            let hb = self.weights.head1_b[os];
            let dst = &mut h1_pre[os * t_len..(os + 1) * t_len];
            dst.fill(hb);
            for is in 0..s {
                let wh = self.weights.head1_w[os * s + is];
                let src = &skip_act[is * t_len..(is + 1) * t_len];
                for t in 0..t_len {
                    dst[t] += wh * src[t];
                }
            }
        }
        let h1_act: Vec<f64> = h1_pre.iter().map(|&v| v.max(0.0)).collect();

        let mut output = vec![[0.0; 2]; t_len];
        for (o, out_b) in self.weights.out_b.iter().enumerate() {
            for t in 0..t_len {
                let mut acc = *out_b;
                for is in 0..s {
                    acc += self.weights.out_w[o * s + is] * h1_act[is * t_len + t];
                }
                if self.config.output_residual {
                    acc += x_in[o * t_len + t];
                }
                output[t][o] = acc;
            }
        }

        Ok(FidNetCache {
            len: t_len,
            x_in: x_in.to_vec(),
            h_states,
            blocks: block_caches,
            skip_sum,
            skip_act,
            h1_pre,
            h1_act,
            output,
        })
    }

    /// Execute raw forward pass on a complex FID (no input scaling or abstention post-processing).
    pub fn forward_fid(
        &self,
        fid: &[Complex64],
        nus_mask: Option<&[bool]>,
        coupling_channel: Option<&[f64]>,
    ) -> Result<Vec<Complex64>, FidNetError> {
        let x_in = self.build_input_features(fid, nus_mask, coupling_channel)?;
        let cache = self.forward_features_with_cache(&x_in, fid.len())?;
        Ok(cache.output)
    }

    /// Exact analytical backpropagation for weighted complex MSE loss:
    /// `L = (1 / (2 * T)) * sum_t w_t * ((re_pred - re_tgt)^2 + (im_pred - im_tgt)^2)`.
    pub fn backward_mse(
        &self,
        cache: &FidNetCache,
        target: &[Complex64],
        loss_weights: Option<&[f64]>,
    ) -> Result<(f64, FidNetGradients), FidNetError> {
        let t_len = cache.len;
        if target.len() != t_len {
            return Err(FidNetError::ShapeMismatch(format!(
                "target length {} != cached FID length {t_len}",
                target.len()
            )));
        }
        if let Some(lw) = loss_weights {
            if lw.len() != t_len {
                return Err(FidNetError::ShapeMismatch(format!(
                    "loss_weights length {} != cached FID length {t_len}",
                    lw.len()
                )));
            }
        }

        let in_ch = self.config.in_channels();
        let c = self.config.residual_channels;
        let s = self.config.skip_channels;
        let k = self.config.kernel_size;
        let radius = (k / 2) as isize;
        let inv_t = 1.0 / (t_len as f64);

        let mut grads = FidNetWeights::zeros(&self.config)?;

        // 1. Output MSE loss & dL/dy
        let mut loss = 0.0;
        let mut d_y = vec![0.0; 2 * t_len];
        for t in 0..t_len {
            let wt = loss_weights.map_or(1.0, |lw| lw[t]);
            if !wt.is_finite() || !target[t][0].is_finite() || !target[t][1].is_finite() {
                return Err(FidNetError::NonFiniteInput(format!(
                    "non-finite target or weight at t={t}"
                )));
            }
            let diff_re = cache.output[t][0] - target[t][0];
            let diff_im = cache.output[t][1] - target[t][1];
            loss += 0.5 * inv_t * wt * (diff_re * diff_re + diff_im * diff_im);
            d_y[t] = inv_t * wt * diff_re;
            d_y[t_len + t] = inv_t * wt * diff_im;
        }

        // 2. Output 1x1 linear layer backward
        let mut d_h1_act = vec![0.0; s * t_len];
        for o in 0..2 {
            let dy_row = &d_y[o * t_len..(o + 1) * t_len];
            let mut b_acc = 0.0;
            for &v in dy_row {
                b_acc += v;
            }
            grads.out_b[o] += b_acc;
            for is in 0..s {
                let h1_row = &cache.h1_act[is * t_len..(is + 1) * t_len];
                let w_val = self.weights.out_w[o * s + is];
                let dh1_row = &mut d_h1_act[is * t_len..(is + 1) * t_len];
                let mut w_acc = 0.0;
                for t in 0..t_len {
                    w_acc += dy_row[t] * h1_row[t];
                    dh1_row[t] += w_val * dy_row[t];
                }
                grads.out_w[o * s + is] += w_acc;
            }
        }

        // 3. Head1 ReLU + 1x1 backward
        let mut d_h1_pre = vec![0.0; s * t_len];
        for idx in 0..(s * t_len) {
            if cache.h1_pre[idx] > 0.0 {
                d_h1_pre[idx] = d_h1_act[idx];
            }
        }
        let mut d_skip_act = vec![0.0; s * t_len];
        for os in 0..s {
            let dh_row = &d_h1_pre[os * t_len..(os + 1) * t_len];
            let mut b_acc = 0.0;
            for &v in dh_row {
                b_acc += v;
            }
            grads.head1_b[os] += b_acc;
            for is in 0..s {
                let sa_row = &cache.skip_act[is * t_len..(is + 1) * t_len];
                let w_val = self.weights.head1_w[os * s + is];
                let dsa_row = &mut d_skip_act[is * t_len..(is + 1) * t_len];
                let mut w_acc = 0.0;
                for t in 0..t_len {
                    w_acc += dh_row[t] * sa_row[t];
                    dsa_row[t] += w_val * dh_row[t];
                }
                grads.head1_w[os * s + is] += w_acc;
            }
        }

        // 4. Skip ReLU backward
        let mut d_skip_sum = vec![0.0; s * t_len];
        for idx in 0..(s * t_len) {
            if cache.skip_sum[idx] > 0.0 {
                d_skip_sum[idx] = d_skip_act[idx];
            }
        }

        // 5. WaveNet blocks backward (reverse order)
        let mut d_h = vec![0.0; c * t_len];
        for bi in (0..self.config.num_blocks()).rev() {
            let blk_w = &self.weights.blocks[bi];
            let blk_g = &mut grads.blocks[bi];
            let blk_c = &cache.blocks[bi];
            let h_in = &cache.h_states[bi];
            let d_isize = self.config.dilations[bi] as isize;

            let mut d_z = vec![0.0; c * t_len];

            // Skip 1x1 backward
            for os in 0..s {
                let ds_row = &d_skip_sum[os * t_len..(os + 1) * t_len];
                let mut b_acc = 0.0;
                for &v in ds_row {
                    b_acc += v;
                }
                blk_g.skip_b[os] += b_acc;
                for ic in 0..c {
                    let z_row = &blk_c.z[ic * t_len..(ic + 1) * t_len];
                    let w_val = blk_w.skip_w[os * c + ic];
                    let dz_row = &mut d_z[ic * t_len..(ic + 1) * t_len];
                    let mut w_acc = 0.0;
                    for t in 0..t_len {
                        w_acc += ds_row[t] * z_row[t];
                        dz_row[t] += w_val * ds_row[t];
                    }
                    blk_g.skip_w[os * c + ic] += w_acc;
                }
            }

            // Residual 1x1 backward
            for oc in 0..c {
                let dh_row = &d_h[oc * t_len..(oc + 1) * t_len];
                let mut b_acc = 0.0;
                for &v in dh_row {
                    b_acc += v;
                }
                blk_g.res_b[oc] += b_acc;
                for ic in 0..c {
                    let z_row = &blk_c.z[ic * t_len..(ic + 1) * t_len];
                    let w_val = blk_w.res_w[oc * c + ic];
                    let dz_row = &mut d_z[ic * t_len..(ic + 1) * t_len];
                    let mut w_acc = 0.0;
                    for t in 0..t_len {
                        w_acc += dh_row[t] * z_row[t];
                        dz_row[t] += w_val * dh_row[t];
                    }
                    blk_g.res_w[oc * c + ic] += w_acc;
                }
            }

            // Gated activation backward: z = tanh(f) * sigmoid(g)
            let mut d_f = vec![0.0; c * t_len];
            let mut d_g = vec![0.0; c * t_len];
            for idx in 0..(c * t_len) {
                let dz = d_z[idx];
                d_f[idx] = dz * blk_c.sg[idx] * safe_tanh_deriv(blk_c.f_pre[idx], blk_c.tf[idx]);
                d_g[idx] = dz * blk_c.tf[idx] * safe_sigmoid_deriv(blk_c.g_pre[idx], blk_c.sg[idx]);
            }

            // Residual bypass gradient + dilated filter & gate conv backward
            let mut d_h_prev = d_h;
            for oc in 0..c {
                let df_row = &d_f[oc * t_len..(oc + 1) * t_len];
                let dg_row = &d_g[oc * t_len..(oc + 1) * t_len];
                let mut fb_acc = 0.0;
                let mut gb_acc = 0.0;
                for t in 0..t_len {
                    fb_acc += df_row[t];
                    gb_acc += dg_row[t];
                }
                blk_g.filter_b[oc] += fb_acc;
                blk_g.gate_b[oc] += gb_acc;

                for ic in 0..c {
                    let h_row = &h_in[ic * t_len..(ic + 1) * t_len];
                    let dh_prev_row = &mut d_h_prev[ic * t_len..(ic + 1) * t_len];
                    let w_base = (oc * c + ic) * k;
                    for m in 0..k {
                        let wf = blk_w.filter_w[w_base + m];
                        let wg = blk_w.gate_w[w_base + m];
                        let offset = (m as isize - radius) * d_isize;
                        let t_start = if offset < 0 { (-offset) as usize } else { 0 };
                        let t_end = if offset > 0 {
                            t_len.saturating_sub(offset as usize)
                        } else {
                            t_len
                        };
                        if t_start < t_end {
                            let mut fw_acc = 0.0;
                            let mut gw_acc = 0.0;
                            for t in t_start..t_end {
                                let src_idx = (t as isize + offset) as usize;
                                let hv = h_row[src_idx];
                                let dfv = df_row[t];
                                let dgv = dg_row[t];
                                fw_acc += dfv * hv;
                                gw_acc += dgv * hv;
                                dh_prev_row[src_idx] += wf * dfv + wg * dgv;
                            }
                            blk_g.filter_w[w_base + m] += fw_acc;
                            blk_g.gate_w[w_base + m] += gw_acc;
                        }
                    }
                }
            }
            d_h = d_h_prev;
        }

        // 6. Stem 1x1 backward
        for oc in 0..c {
            let dh_row = &d_h[oc * t_len..(oc + 1) * t_len];
            let mut b_acc = 0.0;
            for &v in dh_row {
                b_acc += v;
            }
            grads.stem_b[oc] += b_acc;
            for ic in 0..in_ch {
                let x_row = &cache.x_in[ic * t_len..(ic + 1) * t_len];
                let mut w_acc = 0.0;
                for t in 0..t_len {
                    w_acc += dh_row[t] * x_row[t];
                }
                grads.stem_w[oc * in_ch + ic] += w_acc;
            }
        }

        Ok((loss, grads))
    }
}

// ============================================================================
// Adam Optimizer & Training Step
// ============================================================================

/// Configuration for the `Adam` stochastic gradient optimizer (Kingma & Ba 2015).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FidNetAdamConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
}

impl Default for FidNetAdamConfig {
    fn default() -> Self {
        Self {
            lr: 5e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            max_grad_norm: 5.0,
        }
    }
}

/// Persistent first/second moment state for `Adam` optimization of a `FidNetModel`.
#[derive(Clone, Debug)]
pub struct FidNetAdamState {
    pub cfg: FidNetAdamConfig,
    pub step: u64,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

impl FidNetAdamState {
    pub fn new(config: &FidNetConfig, cfg: FidNetAdamConfig) -> Result<Self, FidNetError> {
        let n_params = config.parameter_count()?;
        Ok(Self {
            cfg,
            step: 0,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
        })
    }
}

/// Single training example for `train_fidnet_step`.
#[derive(Clone, Debug)]
pub struct FidNetTrainExample {
    pub input_fid: Vec<Complex64>,
    pub nus_mask: Option<Vec<bool>>,
    pub coupling_channel: Option<Vec<f64>>,
    pub target_fid: Vec<Complex64>,
    pub loss_weights: Option<Vec<f64>>,
}

/// Execute one exact analytical-backpropagation Adam optimization step over a mini-batch
/// of time-domain FIDs, returning the mean batch MSE loss.
pub fn train_fidnet_step(
    model: &mut FidNetModel,
    adam: &mut FidNetAdamState,
    batch: &[FidNetTrainExample],
) -> Result<f64, FidNetError> {
    if batch.is_empty() {
        return Err(FidNetError::EmptyInput(
            "training mini-batch must not be empty".into(),
        ));
    }
    let n_params = model.config.parameter_count()?;
    if adam.m.len() != n_params || adam.v.len() != n_params {
        return Err(FidNetError::ShapeMismatch(format!(
            "Adam state length ({}, {}) != model parameter count {n_params}",
            adam.m.len(),
            adam.v.len()
        )));
    }

    let mut acc_grad = vec![0.0; n_params];
    let mut total_loss = 0.0;
    let inv_batch = 1.0 / (batch.len() as f64);

    for ex in batch {
        let x_in = model.build_input_features(
            &ex.input_fid,
            ex.nus_mask.as_deref(),
            ex.coupling_channel.as_deref(),
        )?;
        let cache = model.forward_features_with_cache(&x_in, ex.input_fid.len())?;
        let (loss, g) = model.backward_mse(&cache, &ex.target_fid, ex.loss_weights.as_deref())?;
        total_loss += loss * inv_batch;
        let flat_g = g.to_flat_vec();
        for (dst, &gv) in acc_grad.iter_mut().zip(flat_g.iter()) {
            *dst += gv * inv_batch;
        }
    }

    // Global L2 gradient clipping for stability
    let grad_norm = acc_grad.iter().map(|g| g * g).sum::<f64>().sqrt();
    if adam.cfg.max_grad_norm > 0.0 && grad_norm > adam.cfg.max_grad_norm {
        let scale = adam.cfg.max_grad_norm / grad_norm;
        for g in &mut acc_grad {
            *g *= scale;
        }
    }

    adam.step = adam.step.saturating_add(1);
    let b1 = adam.cfg.beta1;
    let b2 = adam.cfg.beta2;
    let bc1 = 1.0 - b1.powi(adam.step.min(i32::MAX as u64) as i32);
    let bc2 = 1.0 - b2.powi(adam.step.min(i32::MAX as u64) as i32);

    let mut params = model.weights.to_flat_vec();
    for i in 0..n_params {
        let g = acc_grad[i] + adam.cfg.weight_decay * params[i];
        adam.m[i] = b1 * adam.m[i] + (1.0 - b1) * g;
        adam.v[i] = b2 * adam.v[i] + (1.0 - b2) * g * g;
        let m_hat = adam.m[i] / bc1;
        let v_hat = adam.v[i] / bc2;
        params[i] -= adam.cfg.lr * m_hat / (v_hat.sqrt() + adam.cfg.eps);
    }

    model.weights = FidNetWeights::from_flat_slice(&model.config, &params)?;
    Ok(total_loss)
}

// ============================================================================
// Synthetic Spin-Hamiltonian FID Generator & Poisson-Gap NUS Schedule
// ============================================================================

/// Parameters of a single synthetic NMR resonance in a time-domain spin-Hamiltonian FID.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticSpin {
    /// Resonance offset frequency in Hz (typically in `[-sw_hz / 2, sw_hz / 2]`).
    pub freq_hz: f64,
    /// Transverse relaxation rate $R_2$ in $\text{s}^{-1}$ (`>= 0`).
    pub r2_hz: f64,
    /// Signal amplitude magnitude (`>= 0`).
    pub amplitude: f64,
    /// Zero-order phase in radians.
    pub phase_rad: f64,
    /// Homonuclear scalar coupling constant $J$ in Hz (`0.0` for uncoupled singlet,
    /// e.g. `35.0` Hz for $^1J_{\text{C}\alpha\text{C}\beta}$ or `7.0` Hz for $^3J_{\text{HH}}$).
    pub j_coupling_hz: f64,
}

/// Specification for generating a synthetic complex NMR FID with optional noise, $J$-coupling,
/// and sine-weighted Poisson-gap NUS mask.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticFidSpec {
    pub len: usize,
    pub sw_hz: f64,
    pub spins: Vec<SyntheticSpin>,
    pub noise_sigma: f64,
    /// Optional target NUS sampling density in `(0.0, 1.0]`; `None` generates uniform sampling.
    pub nus_fraction: Option<f64>,
    /// Reference coupling constant in Hz for generating the optional conditioning channel.
    pub conditioning_j_hz: Option<f64>,
    pub seed: u64,
}

/// Generated synthetic time-domain NMR dataset with ground-truth clean singlet, clean coupled FID,
/// and noisy / NUS-masked observed FID.
#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticFidSample {
    /// Ground-truth noiseless, uncoupled (singlet), uniformly sampled complex FID.
    pub clean_singlet: Vec<Complex64>,
    /// Ground-truth noiseless, uniformly sampled FID including $\cos(\pi J t)$ modulation.
    pub clean_coupled: Vec<Complex64>,
    /// Observed complex FID (coupled + complex Gaussian noise, zeroed at unsampled NUS points).
    pub observed: Vec<Complex64>,
    /// Optional binary NUS sampling mask (`true` = measured, `false` = unsampled).
    pub nus_mask: Option<Vec<bool>>,
    /// Optional coupling-constant conditioning channel `cos(pi * J_ref * t)`.
    pub coupling_channel: Option<Vec<f64>>,
}

/// Generate a sine-weighted Poisson-gap NUS binary mask (Hyberts, Takeuchi & Wagner, *JACS* 2010),
/// always sampling `t = 0` and tuning the gap-rate parameter via deterministic bisection to hit
/// `target_fraction` accurately.
pub fn generate_poisson_gap_nus_mask(
    len: usize,
    target_fraction: f64,
    seed: u64,
) -> Result<Vec<bool>, FidNetError> {
    if len == 0 {
        return Err(FidNetError::EmptyInput(
            "NUS mask length must be > 0".into(),
        ));
    }
    if len > MAX_FIDNET_LENGTH {
        return Err(FidNetError::OversizedFid {
            len,
            max_len: MAX_FIDNET_LENGTH,
        });
    }
    if !target_fraction.is_finite() || target_fraction <= 0.0 || target_fraction > 1.0 {
        return Err(FidNetError::InvalidConfig(format!(
            "target_fraction={target_fraction} must be in (0.0, 1.0]"
        )));
    }
    let target_count = ((len as f64) * target_fraction)
        .round()
        .clamp(1.0, len as f64) as usize;
    if target_count == len {
        return Ok(vec![true; len]);
    }
    if target_count == 1 {
        let mut mask = vec![false; len];
        mask[0] = true;
        return Ok(mask);
    }

    let sample_with_scale = |scale: f64, trial_seed: u64| -> Vec<usize> {
        let mut rng = Rng::new(trial_seed);
        let mut idxs = Vec::with_capacity(target_count + 8);
        let mut pos = 0usize;
        while pos < len {
            idxs.push(pos);
            let frac = (pos as f64 + 0.5) / (len as f64);
            // Sine-weighted gap rate: denser at early t where SNR is highest.
            let sin_w = (0.5 * PI * frac).sin();
            let lambda = (scale * sin_w).max(0.0);
            let gap = rng.poisson(lambda);
            pos = pos.saturating_add(1).saturating_add(gap);
        }
        idxs
    };

    // Binary search on gap scale parameter to match target_count closely.
    let mut lo = 0.0f64;
    let mut hi = (len as f64) / (target_count as f64).max(1.0) * 6.0;
    let mut best_idxs = sample_with_scale(hi * 0.5, seed);
    let mut best_err = best_idxs.len().abs_diff(target_count);

    for iter in 0..24u64 {
        let mid = 0.5 * (lo + hi);
        let idxs = sample_with_scale(
            mid,
            seed.wrapping_add(iter.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        let err = idxs.len().abs_diff(target_count);
        if err < best_err {
            best_err = err;
            best_idxs = idxs.clone();
        }
        if idxs.len() > target_count {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let mut mask = vec![false; len];
    for idx in best_idxs {
        if idx < len {
            mask[idx] = true;
        }
    }
    mask[0] = true;

    // Fine deterministic adjustment to hit exact target_count while preserving mask[0] = true.
    let mut current = mask.iter().filter(|&&b| b).count();
    let mut rng = Rng::new(seed ^ 0xA5A5_5A5A_1234_5678);
    if current < target_count {
        let mut unsampled: Vec<usize> = (1..len).filter(|&i| !mask[i]).collect();
        while current < target_count && !unsampled.is_empty() {
            let pick = rng.usize(unsampled.len());
            let idx = unsampled.swap_remove(pick);
            mask[idx] = true;
            current += 1;
        }
    } else if current > target_count {
        let mut sampled: Vec<usize> = (1..len).filter(|&i| mask[i]).collect();
        while current > target_count && !sampled.is_empty() {
            let pick = rng.usize(sampled.len());
            let idx = sampled.swap_remove(pick);
            mask[idx] = false;
            current -= 1;
        }
    }

    Ok(mask)
}

/// Generate a synthetic multi-spin complex NMR FID from physical spin-Hamiltonian parameters
/// (resonance frequencies, $R_2$ decay rates, homonuclear $J$-splittings, complex Gaussian noise,
/// and optional sine-weighted Poisson-gap NUS mask).
pub fn generate_synthetic_fid(spec: &SyntheticFidSpec) -> Result<SyntheticFidSample, FidNetError> {
    if spec.len == 0 {
        return Err(FidNetError::EmptyInput(
            "synthetic FID length must be > 0".into(),
        ));
    }
    if spec.len > MAX_FIDNET_LENGTH {
        return Err(FidNetError::OversizedFid {
            len: spec.len,
            max_len: MAX_FIDNET_LENGTH,
        });
    }
    if !spec.sw_hz.is_finite() || spec.sw_hz <= 0.0 {
        return Err(FidNetError::InvalidConfig(format!(
            "sw_hz={} must be finite and > 0",
            spec.sw_hz
        )));
    }
    if !spec.noise_sigma.is_finite() || spec.noise_sigma < 0.0 {
        return Err(FidNetError::InvalidConfig(format!(
            "noise_sigma={} must be finite and >= 0",
            spec.noise_sigma
        )));
    }
    for (idx, sp) in spec.spins.iter().enumerate() {
        if !sp.freq_hz.is_finite()
            || !sp.r2_hz.is_finite()
            || sp.r2_hz < 0.0
            || !sp.amplitude.is_finite()
            || !sp.phase_rad.is_finite()
            || !sp.j_coupling_hz.is_finite()
        {
            return Err(FidNetError::InvalidConfig(format!(
                "invalid spin[{idx}] parameters: {sp:?}"
            )));
        }
    }

    let dt = 1.0 / spec.sw_hz;
    let mut clean_singlet = vec![[0.0; 2]; spec.len];
    let mut clean_coupled = vec![[0.0; 2]; spec.len];

    for (k, (cs, cc)) in clean_singlet
        .iter_mut()
        .zip(clean_coupled.iter_mut())
        .enumerate()
    {
        let t = (k as f64) * dt;
        let mut s_re = 0.0;
        let mut s_im = 0.0;
        let mut c_re = 0.0;
        let mut c_im = 0.0;
        for sp in &spec.spins {
            let decay = sp.amplitude * (-sp.r2_hz * t).exp();
            let angle = TAU * sp.freq_hz * t + sp.phase_rad;
            let (sin_a, cos_a) = angle.sin_cos();
            let base_re = decay * cos_a;
            let base_im = decay * sin_a;
            s_re += base_re;
            s_im += base_im;

            let j_mod = if sp.j_coupling_hz != 0.0 {
                (PI * sp.j_coupling_hz * t).cos()
            } else {
                1.0
            };
            c_re += base_re * j_mod;
            c_im += base_im * j_mod;
        }
        *cs = [s_re, s_im];
        *cc = [c_re, c_im];
    }

    let nus_mask = match spec.nus_fraction {
        Some(frac) => Some(generate_poisson_gap_nus_mask(
            spec.len,
            frac,
            spec.seed.wrapping_add(0x517C_C1B7_2722_0A95),
        )?),
        None => None,
    };

    let mut rng = Rng::new(spec.seed);
    let mut observed = vec![[0.0; 2]; spec.len];
    for t in 0..spec.len {
        let sampled = nus_mask.as_ref().is_none_or(|m| m[t]);
        if sampled {
            let n_re = if spec.noise_sigma > 0.0 {
                rng.normal(0.0, spec.noise_sigma)
            } else {
                0.0
            };
            let n_im = if spec.noise_sigma > 0.0 {
                rng.normal(0.0, spec.noise_sigma)
            } else {
                0.0
            };
            observed[t] = [clean_coupled[t][0] + n_re, clean_coupled[t][1] + n_im];
        }
    }

    let coupling_channel = match spec.conditioning_j_hz {
        Some(j_hz) => Some(build_coupling_channel(spec.len, j_hz, spec.sw_hz)?),
        None => None,
    };

    Ok(SyntheticFidSample {
        clean_singlet,
        clean_coupled,
        observed,
        nus_mask,
        coupling_channel,
    })
}

// ============================================================================
// Uncertainty, Energy Conservation & Artifact-Abstention Diagnostics
// ============================================================================

/// Explicit reason(s) why `FID-Net` abstained from certifying a neural reconstruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AbstainReason {
    /// Neural reconstruction violates measured time-domain data consistency beyond threshold.
    DataInconsistency,
    /// Input FID exhibits out-of-distribution thermal noise / insufficient SNR.
    OutOfDistributionNoise,
    /// Input FID or conditioning channel contains `NaN` or `±Inf`.
    NonFiniteInput,
    /// Input FID is empty or exceeds `MAX_FIDNET_LENGTH`.
    OversizedFid,
    /// Reconstructed Parseval energy anomalously diverges from expected input energy.
    EnergyAnomaly,
}

/// Configuration for certified `FID-Net` inference with data-consistency and OOD abstention checks.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FidNetInferenceConfig {
    /// Automatically scale input FID by RMS of observed samples and rescale output back.
    pub normalize_input: bool,
    /// Maximum tolerated relative L2 residual on measured time points
    /// (`||pred_obs - meas_obs||_2 / ||meas_obs||_2`) before emitting `DataInconsistency`.
    pub max_data_inconsistency_rel: f64,
    /// Minimum estimated input time-domain SNR (`head_rms / tail_rms`) before emitting
    /// `OutOfDistributionNoise`.
    pub min_input_snr: f64,
    /// Maximum tolerated ratio of reconstructed energy to (sampling-density-corrected) input energy.
    pub max_energy_ratio: f64,
    /// Minimum tolerated ratio of reconstructed energy to input energy (for non-trivial inputs).
    pub min_energy_ratio: f64,
    /// Number of perturbed diagnostic passes (`0` disables multi-pass epistemic variance check).
    pub uncertainty_passes: usize,
    /// Relative Gaussian perturbation amplitude used during multi-pass uncertainty estimation.
    pub uncertainty_jitter_rel: f64,
    /// Deterministic RNG seed for multi-pass diagnostic perturbations.
    pub uncertainty_seed: u64,
}

impl Default for FidNetInferenceConfig {
    fn default() -> Self {
        Self {
            normalize_input: true,
            max_data_inconsistency_rel: 0.35,
            min_input_snr: 1.25,
            max_energy_ratio: 10.0,
            min_energy_ratio: 0.05,
            uncertainty_passes: 4,
            uncertainty_jitter_rel: 0.01,
            uncertainty_seed: 0xF1D4_E700_0000_0001,
        }
    }
}

/// Quantitative diagnostic and abstention certificate emitted alongside every `FID-Net` inference.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FidNetDiagnostic {
    /// Relative L2 discrepancy on measured (observed) time-domain points.
    pub measured_data_residual_rel: f64,
    /// Estimated time-domain SNR (`early_rms / tail_rms` on observed samples).
    pub estimated_input_snr: f64,
    /// Ratio of reconstructed Parseval spectral energy to sampling-density-corrected input energy.
    pub energy_conservation_ratio: f64,
    /// Multi-pass epistemic/perturbation RMS uncertainty (`0.0` when `uncertainty_passes == 0`).
    pub uncertainty_rms: f64,
    /// `true` iff one or more `AbstainReason` guards triggered.
    pub abstained: bool,
    /// Ordered list of triggered abstention reasons (empty iff `!abstained`).
    pub abstain_reasons: Vec<AbstainReason>,
}

/// Output of certified `FID-Net` inference on a single complex 1D FID.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FidNetInferenceOutput {
    pub reconstructed: Vec<Complex64>,
    pub diagnostic: FidNetDiagnostic,
}

fn estimate_fid_snr_and_energy(fid: &[Complex64], nus_mask: Option<&[bool]>) -> (f64, f64, usize) {
    let mut obs_mags_sq: Vec<f64> = Vec::with_capacity(fid.len());
    for (t, &[re, im]) in fid.iter().enumerate() {
        let sampled = nus_mask.is_none_or(|m| m.get(t).copied().unwrap_or(false));
        if sampled {
            obs_mags_sq.push(re * re + im * im);
        }
    }
    let n_obs = obs_mags_sq.len();
    if n_obs == 0 {
        return (0.0, 0.0, 0);
    }
    let obs_energy: f64 = obs_mags_sq.iter().sum();
    if n_obs < 8 {
        return (f64::INFINITY, obs_energy, n_obs);
    }
    let head_n = (n_obs / 4).max(2);
    let tail_n = (n_obs / 4).max(2);
    let head_rms = (obs_mags_sq[..head_n].iter().sum::<f64>() / (head_n as f64)).sqrt();
    let tail_rms = (obs_mags_sq[n_obs - tail_n..].iter().sum::<f64>() / (tail_n as f64)).sqrt();
    let snr = if tail_rms <= 1e-15 {
        if head_rms > 1e-15 {
            1e6
        } else {
            0.0
        }
    } else {
        head_rms / tail_rms
    };
    (snr, obs_energy, n_obs)
}

impl FidNetModel {
    /// Run certified `FID-Net` inference with input normalization, measured-point data-consistency
    /// verification, Parseval energy conservation check, optional multi-pass perturbation
    /// uncertainty estimation, and explicit artifact abstention (`FidNetDiagnostic`).
    ///
    /// Hostile inputs (`NaN`/`Inf`, empty, or `len > MAX_FIDNET_LENGTH`) never panic; they return
    /// an abstained `FidNetInferenceOutput` tagged with `NonFiniteInput` / `OversizedFid`.
    pub fn infer_fid_with_diagnostics(
        &self,
        fid: &[Complex64],
        nus_mask: Option<&[bool]>,
        coupling_channel: Option<&[f64]>,
        cfg: &FidNetInferenceConfig,
    ) -> FidNetInferenceOutput {
        // 1. Guard empty / oversized FID
        if fid.is_empty() || fid.len() > MAX_FIDNET_LENGTH {
            return FidNetInferenceOutput {
                reconstructed: Vec::new(),
                diagnostic: FidNetDiagnostic {
                    measured_data_residual_rel: f64::INFINITY,
                    estimated_input_snr: 0.0,
                    energy_conservation_ratio: 0.0,
                    uncertainty_rms: 0.0,
                    abstained: true,
                    abstain_reasons: vec![AbstainReason::OversizedFid],
                },
            };
        }

        // 2. Guard non-finite samples or conditioning channel
        let fid_non_finite = fid
            .iter()
            .any(|&[re, im]| !re.is_finite() || !im.is_finite());
        let cc_non_finite = coupling_channel
            .is_some_and(|cc| cc.len() != fid.len() || cc.iter().any(|v| !v.is_finite()));
        let mask_bad_len = nus_mask.is_some_and(|m| m.len() != fid.len());
        if fid_non_finite || cc_non_finite || mask_bad_len {
            return FidNetInferenceOutput {
                reconstructed: vec![[0.0; 2]; fid.len()],
                diagnostic: FidNetDiagnostic {
                    measured_data_residual_rel: f64::INFINITY,
                    estimated_input_snr: 0.0,
                    energy_conservation_ratio: 0.0,
                    uncertainty_rms: 0.0,
                    abstained: true,
                    abstain_reasons: vec![AbstainReason::NonFiniteInput],
                },
            };
        }

        // 3. Estimate input SNR and RMS normalization scale on measured points
        let (est_snr, obs_energy, n_obs) = estimate_fid_snr_and_energy(fid, nus_mask);
        let obs_rms = if n_obs > 0 {
            (obs_energy / (n_obs as f64)).sqrt()
        } else {
            0.0
        };
        let scale = if cfg.normalize_input && obs_rms > 1e-12 {
            obs_rms
        } else {
            1.0
        };
        let inv_scale = 1.0 / scale;

        let norm_fid: Vec<Complex64> = fid
            .iter()
            .map(|&[re, im]| [re * inv_scale, im * inv_scale])
            .collect();

        let recon_norm = match self.forward_fid(&norm_fid, nus_mask, coupling_channel) {
            Ok(v) => v,
            Err(_) => {
                return FidNetInferenceOutput {
                    reconstructed: vec![[0.0; 2]; fid.len()],
                    diagnostic: FidNetDiagnostic {
                        measured_data_residual_rel: f64::INFINITY,
                        estimated_input_snr: est_snr,
                        energy_conservation_ratio: 0.0,
                        uncertainty_rms: 0.0,
                        abstained: true,
                        abstain_reasons: vec![AbstainReason::NonFiniteInput],
                    },
                };
            }
        };

        let reconstructed: Vec<Complex64> = recon_norm
            .iter()
            .map(|&[re, im]| [re * scale, im * scale])
            .collect();

        // 4. Multi-pass perturbation epistemic uncertainty check
        let mut uncertainty_rms = 0.0;
        if cfg.uncertainty_passes > 0 && cfg.uncertainty_jitter_rel > 0.0 && obs_rms > 1e-12 {
            let mut rng = Rng::new(cfg.uncertainty_seed);
            let mut sum_sq_dev = 0.0;
            let mut valid_passes = 0usize;
            for _ in 0..cfg.uncertainty_passes {
                let mut pert = norm_fid.clone();
                for (t, sample) in pert.iter_mut().enumerate() {
                    let sampled = nus_mask.is_none_or(|m| m[t]);
                    if sampled {
                        sample[0] += rng.normal(0.0, cfg.uncertainty_jitter_rel);
                        sample[1] += rng.normal(0.0, cfg.uncertainty_jitter_rel);
                    }
                }
                if let Ok(p_out) = self.forward_fid(&pert, nus_mask, coupling_channel) {
                    let mut pass_dev = 0.0;
                    for t in 0..fid.len() {
                        let dr = (p_out[t][0] - recon_norm[t][0]) * scale;
                        let di = (p_out[t][1] - recon_norm[t][1]) * scale;
                        pass_dev += dr * dr + di * di;
                    }
                    sum_sq_dev += pass_dev / (fid.len() as f64);
                    valid_passes += 1;
                }
            }
            if valid_passes > 0 {
                uncertainty_rms = (sum_sq_dev / (valid_passes as f64)).sqrt();
            }
        }

        // 5. Measured-point data-consistency residual
        let mut obs_err_sq = 0.0;
        for t in 0..fid.len() {
            let sampled = nus_mask.is_none_or(|m| m[t]);
            if sampled {
                let dr = reconstructed[t][0] - fid[t][0];
                let di = reconstructed[t][1] - fid[t][1];
                obs_err_sq += dr * dr + di * di;
            }
        }
        let measured_data_residual_rel = if obs_energy > 1e-24 {
            (obs_err_sq / obs_energy).sqrt()
        } else if obs_err_sq > 1e-24 {
            f64::INFINITY
        } else {
            0.0
        };

        // 6. Parseval energy conservation ratio (accounting for NUS sampling fraction)
        let recon_energy: f64 = reconstructed
            .iter()
            .map(|&[re, im]| re * re + im * im)
            .sum();
        let sampling_frac = if !fid.is_empty() {
            (n_obs as f64) / (fid.len() as f64)
        } else {
            1.0
        };
        let expected_full_energy = if sampling_frac > 0.0 {
            obs_energy / sampling_frac
        } else {
            0.0
        };
        let energy_conservation_ratio = if expected_full_energy > 1e-24 {
            recon_energy / expected_full_energy
        } else if recon_energy > 1e-24 {
            f64::INFINITY
        } else {
            1.0
        };

        // 7. Assemble abstention reasons
        let mut abstain_reasons = Vec::new();
        if !reconstructed
            .iter()
            .all(|&[re, im]| re.is_finite() && im.is_finite())
        {
            abstain_reasons.push(AbstainReason::NonFiniteInput);
        }
        // Data-consistency check applies directly to NUS reconstruction and denoising;
        // for virtual decoupling, the network intentionally removes cos(pi J t) modulation
        // so measured-point residual is checked only when `task != VirtualDecoupling` or
        // when the caller explicitly tightens `max_data_inconsistency_rel`.
        if self.config.task != FidNetTask::VirtualDecoupling
            && measured_data_residual_rel > cfg.max_data_inconsistency_rel
        {
            abstain_reasons.push(AbstainReason::DataInconsistency);
        }
        if est_snr < cfg.min_input_snr && obs_energy > 1e-24 {
            abstain_reasons.push(AbstainReason::OutOfDistributionNoise);
        }
        if obs_energy > 1e-24
            && (energy_conservation_ratio > cfg.max_energy_ratio
                || energy_conservation_ratio < cfg.min_energy_ratio)
        {
            abstain_reasons.push(AbstainReason::EnergyAnomaly);
        }

        let abstained = !abstain_reasons.is_empty();
        FidNetInferenceOutput {
            reconstructed,
            diagnostic: FidNetDiagnostic {
                measured_data_residual_rel,
                estimated_input_snr: est_snr,
                energy_conservation_ratio,
                uncertainty_rms,
                abstained,
                abstain_reasons,
            },
        }
    }
}

// ============================================================================
// 2D & 3D Indirect-Column Batch Application
// ============================================================================

/// Selector for which indirect axis of a 3D hypercomplex `Fid3D` to process with `FID-Net`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fid3DIndirectAxis {
    /// Indirect axis `F2` (`axis_y`, length `axis_y.complex_points`).
    AxisY,
    /// Indirect axis `F1` (`axis_z`, length `axis_z.complex_points`).
    AxisZ,
}

/// Apply a `FidNetModel` across a 2D batch of indirect columns (`[n_cols][t1_len]`).
pub fn apply_fidnet_2d_columns(
    model: &FidNetModel,
    columns: &[Vec<Complex64>],
    nus_mask: Option<&[bool]>,
    coupling_channel: Option<&[f64]>,
    cfg: &FidNetInferenceConfig,
) -> Result<Vec<FidNetInferenceOutput>, FidNetError> {
    if columns.len() > MAX_FIDNET_BATCH {
        return Err(FidNetError::InvalidConfig(format!(
            "2D batch size {} exceeds MAX_FIDNET_BATCH ({MAX_FIDNET_BATCH})",
            columns.len()
        )));
    }
    let mut outputs = Vec::with_capacity(columns.len());
    for col in columns {
        outputs.push(model.infer_fid_with_diagnostics(col, nus_mask, coupling_channel, cfg));
    }
    Ok(outputs)
}

/// Apply a `FidNetModel` directly along the indirect `axis_y` (`F1`) columns of a raw 2D
/// hypercomplex `Fid2D` (operating across all `(sin_comp, ix)` columns), returning the
/// reconstructed `Fid2D` and per-column diagnostics.
pub fn apply_fidnet_fid2d_indirect(
    model: &FidNetModel,
    fid: &Fid2D,
    coupling_channel: Option<&[f64]>,
    cfg: &FidNetInferenceConfig,
) -> Result<(Fid2D, Vec<FidNetDiagnostic>), FidNetError> {
    let n0 = fid.axis_x.complex_points;
    let n1 = fid.axis_y.complex_points;
    let expected_len = n1
        .checked_mul(2)
        .and_then(|v| v.checked_mul(n0))
        .ok_or_else(|| FidNetError::ShapeMismatch("Fid2D dimensions overflow".into()))?;
    if fid.data.len() != expected_len {
        return Err(FidNetError::ShapeMismatch(format!(
            "Fid2D data length {} != n1({n1}) * 2 * n0({n0})",
            fid.data.len()
        )));
    }
    let n_cols = n0.saturating_mul(2);
    if n_cols > MAX_FIDNET_BATCH {
        return Err(FidNetError::InvalidConfig(format!(
            "Fid2D indirect column count {n_cols} exceeds MAX_FIDNET_BATCH ({MAX_FIDNET_BATCH})"
        )));
    }

    let mut out_fid = fid.clone();
    let mut diags = Vec::with_capacity(n_cols);
    let mut col_buf = vec![[0.0; 2]; n1];

    for comp in 0..2 {
        for ix in 0..n0 {
            for (k1, slot) in col_buf.iter_mut().enumerate() {
                *slot = fid.data[(k1 * 2 + comp) * n0 + ix];
            }
            let res = model.infer_fid_with_diagnostics(
                &col_buf,
                fid.nus_mask.as_deref(),
                coupling_channel,
                cfg,
            );
            if res.reconstructed.len() == n1 {
                for (k1, &val) in res.reconstructed.iter().enumerate() {
                    out_fid.data[(k1 * 2 + comp) * n0 + ix] = val;
                }
            }
            diags.push(res.diagnostic);
        }
    }
    if model.config.task == FidNetTask::NusReconstruction {
        out_fid.nus_mask = None;
    }
    Ok((out_fid, diags))
}

/// Apply a `FidNetModel` across a 3D collection of indirect columns (`[n_planes][n_cols][t_len]`).
pub fn apply_fidnet_3d_columns(
    model: &FidNetModel,
    planes: &[Vec<Vec<Complex64>>],
    nus_mask: Option<&[bool]>,
    coupling_channel: Option<&[f64]>,
    cfg: &FidNetInferenceConfig,
) -> Result<Vec<Vec<FidNetInferenceOutput>>, FidNetError> {
    let total_cols: usize = planes.iter().map(Vec::len).sum();
    if total_cols > MAX_FIDNET_BATCH {
        return Err(FidNetError::InvalidConfig(format!(
            "3D batch column count {total_cols} exceeds MAX_FIDNET_BATCH ({MAX_FIDNET_BATCH})"
        )));
    }
    let mut out_planes = Vec::with_capacity(planes.len());
    for plane in planes {
        let mut out_cols = Vec::with_capacity(plane.len());
        for col in plane {
            out_cols.push(model.infer_fid_with_diagnostics(col, nus_mask, coupling_channel, cfg));
        }
        out_planes.push(out_cols);
    }
    Ok(out_planes)
}

/// Apply a `FidNetModel` along either indirect axis (`AxisY` = `F2`, `AxisZ` = `F1`) of a
/// raw 3D hypercomplex `Fid3D`, returning the reconstructed `Fid3D` and per-column diagnostics.
pub fn apply_fidnet_fid3d_indirect(
    model: &FidNetModel,
    fid: &Fid3D,
    axis: Fid3DIndirectAxis,
    coupling_channel: Option<&[f64]>,
    cfg: &FidNetInferenceConfig,
) -> Result<(Fid3D, Vec<FidNetDiagnostic>), FidNetError> {
    let n0 = fid.axis_x.complex_points;
    let n1 = fid.axis_y.complex_points;
    let n2 = fid.axis_z.complex_points;
    let expected_len = n2
        .checked_mul(2)
        .and_then(|v| v.checked_mul(n1))
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_mul(n0))
        .ok_or_else(|| FidNetError::ShapeMismatch("Fid3D dimensions overflow".into()))?;
    if fid.data.len() != expected_len {
        return Err(FidNetError::ShapeMismatch(format!(
            "Fid3D data length {} != n2({n2})*2*n1({n1})*2*n0({n0})",
            fid.data.len()
        )));
    }

    let mut out_fid = fid.clone();
    let mut diags = Vec::new();

    match axis {
        Fid3DIndirectAxis::AxisY => {
            let n_cols = n2 * 4 * n0;
            if n_cols > MAX_FIDNET_BATCH {
                return Err(FidNetError::InvalidConfig(format!(
                    "Fid3D AxisY column count {n_cols} exceeds MAX_FIDNET_BATCH ({MAX_FIDNET_BATCH})"
                )));
            }
            let mut col_buf = vec![[0.0; 2]; n1];
            let mut mask_buf = vec![true; n1];
            for k2 in 0..n2 {
                let row_mask = fid.nus_mask.as_ref().and_then(|m| {
                    if m.len() == n2 * n1 {
                        mask_buf.copy_from_slice(&m[k2 * n1..(k2 + 1) * n1]);
                        Some(mask_buf.as_slice())
                    } else {
                        None
                    }
                });
                for c2 in 0..2 {
                    for c1 in 0..2 {
                        for ix in 0..n0 {
                            for (k1, slot) in col_buf.iter_mut().enumerate() {
                                let idx = (((k2 * 2 + c2) * n1 + k1) * 2 + c1) * n0 + ix;
                                *slot = fid.data[idx];
                            }
                            let res = model.infer_fid_with_diagnostics(
                                &col_buf,
                                row_mask,
                                coupling_channel,
                                cfg,
                            );
                            if res.reconstructed.len() == n1 {
                                for (k1, &val) in res.reconstructed.iter().enumerate() {
                                    let idx = (((k2 * 2 + c2) * n1 + k1) * 2 + c1) * n0 + ix;
                                    out_fid.data[idx] = val;
                                }
                            }
                            diags.push(res.diagnostic);
                        }
                    }
                }
            }
        }
        Fid3DIndirectAxis::AxisZ => {
            let n_cols = n1 * 4 * n0;
            if n_cols > MAX_FIDNET_BATCH {
                return Err(FidNetError::InvalidConfig(format!(
                    "Fid3D AxisZ column count {n_cols} exceeds MAX_FIDNET_BATCH ({MAX_FIDNET_BATCH})"
                )));
            }
            let mut col_buf = vec![[0.0; 2]; n2];
            let mut mask_buf = vec![true; n2];
            for k1 in 0..n1 {
                let col_mask = fid.nus_mask.as_ref().and_then(|m| {
                    if m.len() == n2 * n1 {
                        for k2 in 0..n2 {
                            mask_buf[k2] = m[k2 * n1 + k1];
                        }
                        Some(mask_buf.as_slice())
                    } else {
                        None
                    }
                });
                for c2 in 0..2 {
                    for c1 in 0..2 {
                        for ix in 0..n0 {
                            for (k2, slot) in col_buf.iter_mut().enumerate() {
                                let idx = (((k2 * 2 + c2) * n1 + k1) * 2 + c1) * n0 + ix;
                                *slot = fid.data[idx];
                            }
                            let res = model.infer_fid_with_diagnostics(
                                &col_buf,
                                col_mask,
                                coupling_channel,
                                cfg,
                            );
                            if res.reconstructed.len() == n2 {
                                for (k2, &val) in res.reconstructed.iter().enumerate() {
                                    let idx = (((k2 * 2 + c2) * n1 + k1) * 2 + c1) * n0 + ix;
                                    out_fid.data[idx] = val;
                                }
                            }
                            diags.push(res.diagnostic);
                        }
                    }
                }
            }
        }
    }

    Ok((out_fid, diags))
}

// ============================================================================
// Quantitative Signal / Spectral Quality Metrics
// ============================================================================

/// Compute complex time-domain mean squared error `(1 / (2*T)) * sum_t |a[t] - b[t]|^2`.
pub fn compute_fid_mse(reference: &[Complex64], estimate: &[Complex64]) -> f64 {
    if reference.is_empty() || reference.len() != estimate.len() {
        return f64::INFINITY;
    }
    let sum_sq: f64 = reference
        .iter()
        .zip(estimate.iter())
        .map(|(&[ar, ai], &[br, bi])| {
            let dr = ar - br;
            let di = ai - bi;
            dr * dr + di * di
        })
        .sum();
    sum_sq / (2.0 * reference.len() as f64)
}

/// Compute signal-to-reconstruction-error ratio in dB: `10 * log10(||ref||^2 / ||est - ref||^2)`.
pub fn compute_fid_snr_db(reference: &[Complex64], estimate: &[Complex64]) -> f64 {
    if reference.is_empty() || reference.len() != estimate.len() {
        return f64::NEG_INFINITY;
    }
    let sig_energy: f64 = reference.iter().map(|&[r, i]| r * r + i * i).sum();
    let err_energy: f64 = reference
        .iter()
        .zip(estimate.iter())
        .map(|(&[ar, ai], &[br, bi])| {
            let dr = ar - br;
            let di = ai - bi;
            dr * dr + di * di
        })
        .sum();
    if err_energy <= 1e-30 {
        return 300.0;
    }
    if sig_energy <= 1e-30 {
        return f64::NEG_INFINITY;
    }
    10.0 * (sig_energy / err_energy).log10()
}

/// Compute centered 1D magnitude spectrum of a complex FID (with optional zero-fill factor `>= 1`).
pub fn fid_to_magnitude_spectrum(fid: &[Complex64], zero_fill_factor: usize) -> Vec<f64> {
    if fid.is_empty() {
        return Vec::new();
    }
    let zf = zero_fill_factor.max(1);
    let n = fid.len() * zf;
    let mut buf = vec![[0.0; 2]; n];
    buf[..fid.len()].copy_from_slice(fid);
    // Apply half-weight on t = 0 (standard NMR Fourier baseline convention)
    buf[0][0] *= 0.5;
    buf[0][1] *= 0.5;
    fft_arbitrary_in_place(&mut buf, false);
    // FFT shift so DC (0 Hz) is centered at n/2
    let half = n / 2;
    let mut mag = vec![0.0; n];
    for (i, &[re, im]) in buf.iter().enumerate() {
        let shifted = (i + half) % n;
        mag[shifted] = re.hypot(im);
    }
    mag
}
