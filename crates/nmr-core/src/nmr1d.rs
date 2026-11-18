//! Analytical 1D small-molecule and biomolecular NMR digital signal processing (`1H`, `13C`,
//! `DEPT-90`, `DEPT-135`, `19F`, `31P`), minimum-entropy automatic phase correction,
//! Whittaker / Bernstein automatic baseline flattening, time-domain `HSVD` / Hankel–Lanczos
//! solvent resonance subtraction, deuterated solvent & trace impurity reference lookup tables,
//! and coupled-cluster Levenberg–Marquardt pseudo-Voigt lineshape deconvolution.
//!
//! Provenance: Clean-room original implementation of `docs/specs/17-nmr1d-and-dosy.md` §1
//! derived strictly from peer-reviewed literature (zero external source code, macros, or
//! parameter libraries consulted):
//! - Causality, zero-filling & discrete Hilbert transform:
//!   Bartholdi, E. & Ernst, R. R., *J. Magn. Reson.* **1973**, 11, 9–19;
//!   Otting, G., Widmer, H., Wagner, G. & Wüthrich, K., *J. Magn. Reson.* **1986**, 66, 187–193;
//!   Hoch, J. C. & Stern, A. S., *NMR Data Processing*, Wiley-Liss, New York, **1996**.
//! - Lorentz-to-Gauss resolution enhancement & `DEPT-90`/`DEPT-135` polarization editing:
//!   Ferrige, A. G. & Lindon, J. C., *J. Magn. Reson.* **1978**, 31, 337–340;
//!   Doddrell, D. M., Pegg, D. T. & Bendall, M. R., *J. Magn. Reson.* **1982**, 48, 323–327.
//! - Automatic `PHC0`/`PHC1` phase correction via first-derivative Shannon entropy minimization
//!   with dispersion & negative-absorption penalties:
//!   Chen, L., Weng, Z., Goh, L. & Garland, M., *J. Magn. Reson.* **2002**, 158, 164–168;
//!   de Brouwer, H., *J. Magn. Reson.* **2009**, 201, 230–238.
//! - Continuous-wavelet signal-free baseline classification, Whittaker penalized least squares,
//!   and Bernstein polynomial baseline fitting:
//!   Eilers, P. H. C., *Anal. Chem.* **2003**, 75, 3631–3636;
//!   Cobas, J. C., Bernstein, M. A., Martín-Pastor, M. & Tahoces, P. G.,
//!   *J. Magn. Reson.* **2006**, 183, 145–151.
//! - Time-domain Hankel Singular Value Decomposition (`HSVD` / Hankel–Lanczos) solvent
//!   resonance parameterization and subtraction:
//!   Barkhuijsen, H., de Beer, R. & van Ormondt, D., *J. Magn. Reson.* **1987**, 73, 553–557;
//!   Pijnappel, W. W. F., van den Boogaart, A., de Beer, R. & van Ormondt, D.,
//!   *J. Magn. Reson.* **1992**, 97, 122–134.
//! - Deuterated solvent residual signals and trace laboratory impurity chemical shifts:
//!   Gottlieb, H. E., Kotlyar, V. & Nudelman, A., *J. Org. Chem.* **1997**, 62, 7512–7515;
//!   Fulmer, G. R., Miller, A. J. M., Sherden, N. H., Gottlieb, H. E., Nudelman, A.,
//!   Stoltz, B. M., Bercaw, J. E. & Goldberg, K. I., *Organometallics* **2010**, 29, 2176–2179.
//! - Pseudo-Voigt lineshape parameterization & Levenberg–Marquardt multiplet deconvolution:
//!   Levenberg, K., *Quart. Appl. Math.* **1944**, 2, 164–168;
//!   Marquardt, D. W., *J. Soc. Ind. Appl. Math.* **1963**, 11, 431–441;
//!   Wertheim, G. K., Butler, M. A., West, K. W. & Buchanan, D. N. E.,
//!   *Rev. Sci. Instrum.* **1974**, 45, 1369–1371;
//!   Thompson, P., Cox, D. E. & Hastings, J. B., *J. Appl. Crystallogr.* **1987**, 20, 79–83.
//! - IUPAC unified indirect chemical-shift $\Xi$ scale & aqueous `DSS` vs organic `TMS` `${}^{13}\text{C}$ referencing:
//!   Wishart, D. S., Bigam, C. G., Yao, J., Abildgaard, F., Dyson, H. J., Oldfield, E.,
//!   Markley, J. L. & Sykes, B. D., *J. Biomol. NMR* **1995**, 6, 135–140;
//!   Markley, J. L. et al., *Pure Appl. Chem.* **1998**, 70, 117–142;
//!   Harris, R. K., Becker, E. D., Cabral de Menezes, S. M., Goodfellow, R. & Granger, P.,
//!   *Pure Appl. Chem.* **2001**, 73, 1795–1818;
//!   Harris, R. K., Becker, E. D., Cabral de Menezes, S. M., Granger, P., Hoffman, R. E. &
//!   Zilm, K. W., *Pure Appl. Chem.* **2008**, 80, 59–84.
//! - Clean-room JEOL Delta `.jdf` 1D binary header, parameter table, and multi-stage FIR decimation
//!   group-delay format: implemented strictly from the public JEOL Delta NMR File Format
//!   documentation (`"JEOL.NMR"` 1296-byte header + 64-byte parameter table records).

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

use crate::fid::{Complex64, Fid1D};
use crate::ops::fft_arbitrary_in_place;
use crate::spectrum::AxisPpm;

// ============================================================================
// 1. Multi-Nuclear 1D Representation, Apodization, Zero-Fill, FFT & Referencing
// ============================================================================

/// Nuclear isotope for 1D NMR spectroscopy with IUPAC indirect $\Xi$ frequency ratios.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Nucleus1D {
    H1,
    C13,
    F19,
    P31,
    N15,
    Other(String),
}

impl Nucleus1D {
    /// Parse common nucleus strings (`"1H"`, `"H1"`, `"13C"`, `"19F"`, `"31P"`, `"15N"`).
    pub fn parse(label: &str) -> Self {
        let norm = label
            .trim()
            .to_ascii_uppercase()
            .replace(['{', '}', ' '], "");
        match norm.as_str() {
            "1H" | "H1" | "H" | "PROTON" => Self::H1,
            "13C" | "C13" | "C" | "13C1H" => Self::C13,
            "19F" | "F19" | "F" => Self::F19,
            "31P" | "P31" | "P" => Self::P31,
            "15N" | "N15" | "N" => Self::N15,
            _ => Self::Other(label.trim().to_string()),
        }
    }

    /// Canonical display string (`"1H"`, `"13C"`, `"19F"`, `"31P"`, `"15N"`).
    pub fn as_str(&self) -> &str {
        match self {
            Self::H1 => "1H",
            Self::C13 => "13C",
            Self::F19 => "19F",
            Self::P31 => "31P",
            Self::N15 => "15N",
            Self::Other(s) => s.as_str(),
        }
    }

    /// IUPAC indirect chemical-shift referencing ratio $\Xi = \nu_0(X) / \nu_0({}^{1}\text{H}_{\text{TMS/DSS}})$
    /// (Harris et al., *Pure Appl. Chem.* **2001**, 73, 1795–1818; Wishart et al., *J. Biomol. NMR* **1995**, 6, 135–140).
    pub fn xi_ratio(&self) -> f64 {
        match self {
            Self::H1 => 1.000_000_00,
            Self::C13 => 0.251_450_20,
            Self::F19 => 0.940_940_11,
            Self::P31 => 0.404_807_42,
            Self::N15 => 0.101_367_67,
            Self::Other(_) => 1.0,
        }
    }

    /// Gyromagnetic ratio $|\gamma|$ in $\text{rad s}^{-1}\text{ T}^{-1}$.
    pub fn gyromagnetic_rad_s_t(&self) -> f64 {
        match self {
            Self::H1 => 2.675_221_874_4e8,
            Self::C13 => 6.728_284e7,
            Self::F19 => 2.518_148e8,
            Self::P31 => 1.083_94e8,
            Self::N15 => 2.712_618e7,
            Self::Other(_) => 2.675_221_874_4e8,
        }
    }

    /// Typical standard spectral window `(low_ppm, high_ppm)` for routine 1D acquisitions.
    pub fn default_ppm_bounds(&self) -> (f64, f64) {
        match self {
            Self::H1 => (-1.0, 14.0),
            Self::C13 => (-10.0, 230.0),
            Self::F19 => (-250.0, 50.0),
            Self::P31 => (-50.0, 250.0),
            Self::N15 => (60.0, 160.0),
            Self::Other(_) => (-10.0, 20.0),
        }
    }
}

/// Expected resonance sign polarity for 1D phase correction and peak picking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Polarity1D {
    /// All true absorption lines have positive amplitude (`1H`, `13C{1H}`, `DEPT-90`, `19F`, `31P`).
    AllPositive,
    /// Absorption lines carry positive and negative editing signs (`DEPT-135`: `CH`/`CH3 > 0`, `CH2 < 0`).
    Biphasic,
}

/// Supported 1D NMR experiment modalities.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Experiment1DKind {
    Proton1D,
    Carbon13Decoupled,
    Dept90,
    Dept135,
    Fluorine19,
    Phosphorus31,
    Custom(Nucleus1D),
}

impl Experiment1DKind {
    /// Infer 1D experiment kind from spectrometer pulse-program name and observed nucleus.
    pub fn from_pulse_program(pulprog: &str, nucleus_label: &str) -> Self {
        let p = pulprog.trim().to_ascii_lowercase();
        if p.contains("dept135") || p.contains("deptq135") {
            return Self::Dept135;
        }
        if p.contains("dept90") {
            return Self::Dept90;
        }
        match Nucleus1D::parse(nucleus_label) {
            Nucleus1D::H1 => Self::Proton1D,
            Nucleus1D::C13 => Self::Carbon13Decoupled,
            Nucleus1D::F19 => Self::Fluorine19,
            Nucleus1D::P31 => Self::Phosphorus31,
            other => Self::Custom(other),
        }
    }

    /// Observed nucleus for this 1D experiment.
    pub fn nucleus(&self) -> Nucleus1D {
        match self {
            Self::Proton1D => Nucleus1D::H1,
            Self::Carbon13Decoupled | Self::Dept90 | Self::Dept135 => Nucleus1D::C13,
            Self::Fluorine19 => Nucleus1D::F19,
            Self::Phosphorus31 => Nucleus1D::P31,
            Self::Custom(n) => n.clone(),
        }
    }

    /// Expected absorption polarity (`Biphasic` for `DEPT-135`, `AllPositive` otherwise).
    pub fn default_polarity(&self) -> Polarity1D {
        match self {
            Self::Dept135 => Polarity1D::Biphasic,
            _ => Polarity1D::AllPositive,
        }
    }

    /// Recommended default exponential line broadening in Hz.
    pub fn default_line_broadening_hz(&self) -> f64 {
        match self {
            Self::Proton1D => 0.3,
            Self::Carbon13Decoupled | Self::Dept90 | Self::Dept135 => 1.0,
            Self::Fluorine19 => 1.0,
            Self::Phosphorus31 => 1.0,
            Self::Custom(_) => 0.5,
        }
    }
}

/// Carbon $\text{CH}_n$ multiplicity deduced from `${}^{13}\text{C}\{{}^{1}\text{H}\}$, `DEPT-90`, and `DEPT-135`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CarbonDeptMultiplicity {
    /// Quaternary carbon ($\text{C}$): present in broadband `${}^{13}\text{C}$, absent in `DEPT-90` & `DEPT-135`.
    QuaternaryC,
    /// Methine ($\text{CH}$): positive in both `DEPT-90` and `DEPT-135`.
    Ch,
    /// Methylene ($\text{CH}_2$): absent in `DEPT-90`, negative in `DEPT-135`.
    Ch2,
    /// Methyl ($\text{CH}_3$): absent in `DEPT-90`, positive in `DEPT-135`.
    Ch3,
}

/// Theoretical `DEPT` polarization-transfer editing amplitude for $\text{CH}_n$ ($n \in \{0, 1, 2, 3\}$)
/// as a function of the final ${}^{1}\text{H}$ read flip angle $\theta$ in degrees
/// (Doddrell, Pegg & Bendall, *J. Magn. Reson.* **1982**, 48, 323–327).
pub fn dept_transfer_amplitude(attached_protons: u8, theta_deg: f64) -> f64 {
    let th = theta_deg.to_radians();
    match attached_protons {
        0 => 0.0,
        1 => th.sin(),
        2 => (2.0 * th).sin(),
        3 => 0.75 * (th.sin() + (3.0 * th).sin()),
        _ => 0.0,
    }
}

/// Classify a `${}^{13}\text{C}$ resonance's $\text{CH}_n$ multiplicity from normalized `DEPT-90` and
/// `DEPT-135` intensities at its chemical shift (`noise_thresh` > 0).
pub fn classify_dept_multiplicity(
    dept90_amp: f64,
    dept135_amp: f64,
    noise_thresh: f64,
) -> CarbonDeptMultiplicity {
    let thr = noise_thresh.abs().max(1e-12);
    if dept135_amp < -thr {
        CarbonDeptMultiplicity::Ch2
    } else if dept90_amp > thr && dept135_amp > 0.25 * dept90_amp {
        CarbonDeptMultiplicity::Ch
    } else if dept135_amp > thr {
        CarbonDeptMultiplicity::Ch3
    } else {
        CarbonDeptMultiplicity::QuaternaryC
    }
}

/// 1D time-domain apodization window functions (Hoch & Stern 1996; Ferrige & Lindon 1978).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Apodization1D {
    /// Unmodified rectangular window ($w[n] = 1$).
    None,
    /// Matched / sensitivity-enhancing exponential decay: $w[n] = \exp(-\pi\,\text{lb\_hz}\,t_n)$.
    Exponential { lb_hz: f64 },
    /// Pure Gaussian broadening of target FWHM `gb_hz`:
    /// $w[n] = \exp(-(\pi\,\text{gb\_hz}\,t_n / (2\sqrt{\ln 2}))^2)$.
    Gaussian { gb_hz: f64 },
    /// Lorentz-to-Gauss resolution enhancement (Ferrige & Lindon 1978):
    /// narrows Lorentzian decay by `narrow_lb_hz > 0` and imposes Gaussian FWHM `gb_hz > 0`,
    /// with optional echo maximum position `center_frac` in `[0.0, 1.0)`.
    LorentzToGauss {
        narrow_lb_hz: f64,
        gb_hz: f64,
        center_frac: f64,
    },
    /// Cosine-bell ($p = 1$) or squared cosine-bell ($p = 2$):
    /// $w[n] = \cos^p(\pi n / (2 N_{\text{td}}))$.
    CosineBell { power: f64 },
    /// Shifted sine-bell: $w[n] = \sin^p(\pi (\alpha + (1 - \alpha) n / N_{\text{td}}))$, $\alpha \in [0, 1)$.
    ShiftedSineBell { offset_pi_frac: f64, power: f64 },
}

/// Complex 1D time-domain free induction decay (`FID`) ready for apodization, `HSVD`, zero-filling, and FFT.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimeDomain1D {
    pub name: String,
    pub experiment: Experiment1DKind,
    /// Spectral width in Hz (`1 / dt`).
    pub sw_hz: f64,
    /// Spectrometer observation / reference frequency (`0 ppm` denominator) in MHz.
    pub obs_mhz: f64,
    /// Chemical shift (ppm) at the center of the spectral window (transmitter carrier).
    pub carrier_ppm: f64,
    /// Number of active acquired complex time-domain samples prior to zero-filling.
    pub active_td: usize,
    /// Digital oversampling group delay in complex points (`0.0` if already compensated or analog).
    pub group_delay_pts: f64,
    /// Whether time-domain samples must be complex-conjugated prior to `-i` forward DFT.
    pub conjugate_for_fft: bool,
    /// Optional deuterated or aqueous solvent identifier propagated from acquisition metadata.
    #[serde(default)]
    pub solvent: Option<String>,
    /// Complex `(re, im)` samples (length `>= active_td` after zero-filling).
    pub data: Vec<Complex64>,
}

impl TimeDomain1D {
    /// Construct a new 1D FID from complex samples and acquisition parameters.
    pub fn new(
        name: impl Into<String>,
        experiment: Experiment1DKind,
        sw_hz: f64,
        obs_mhz: f64,
        carrier_ppm: f64,
        data: Vec<Complex64>,
    ) -> Self {
        let active_td = data.len();
        Self {
            name: name.into(),
            experiment,
            sw_hz: sw_hz.max(1e-6),
            obs_mhz: obs_mhz.max(1e-6),
            carrier_ppm,
            active_td,
            group_delay_pts: 0.0,
            conjugate_for_fft: false,
            solvent: None,
            data,
        }
    }

    /// Convert a raw vendor `Fid1D` (`crates/nmr-core/src/fid.rs`) into `TimeDomain1D`,
    /// immediately shifting out any integer digital-filter group delay (`Moskau 2002`) so `data[0]`
    /// sits at $t \approx 0$ prior to `HSVD` or apodization while preserving sub-sample `group_delay_pts`.
    pub fn from_fid1d(fid: &Fid1D) -> Self {
        let obs_mhz = fid.axis_x.obs_mhz.max(1e-6);
        let sw_hz = fid.axis_x.sw_hz.max(1e-6);
        let sw_ppm = sw_hz / obs_mhz;
        let carrier_ppm = fid.axis_x.first_ppm - 0.5 * sw_ppm;
        let exp = Experiment1DKind::from_pulse_program(&fid.pulse_program, &fid.axis_x.label);
        let mut td = Self {
            name: fid.name.clone(),
            experiment: exp,
            sw_hz,
            obs_mhz,
            carrier_ppm,
            active_td: fid.data.len(),
            group_delay_pts: fid.axis_x.group_delay,
            conjugate_for_fft: !fid.axis_x.conjugate_for_fft,
            solvent: None,
            data: fid.data.clone(),
        };
        compensate_group_delay_1d(&mut td);
        td
    }

    /// Dwell time $\Delta t = 1 / \text{SW}_{\text{Hz}}$ in seconds.
    #[inline]
    pub fn dwell_s(&self) -> f64 {
        1.0 / self.sw_hz.max(1e-9)
    }
}

mod axis_ppm_serde {
    use crate::spectrum::AxisPpm;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct AxisPpmRepr {
        label: String,
        size: usize,
        first_ppm: f64,
        step_ppm: f64,
        obs_mhz: f64,
    }

    pub fn serialize<S: Serializer>(axis: &AxisPpm, serializer: S) -> Result<S::Ok, S::Error> {
        AxisPpmRepr {
            label: axis.label.clone(),
            size: axis.size,
            first_ppm: axis.first_ppm,
            step_ppm: axis.step_ppm,
            obs_mhz: axis.obs_mhz,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<AxisPpm, D::Error> {
        let r = AxisPpmRepr::deserialize(deserializer)?;
        Ok(AxisPpm {
            label: r.label,
            size: r.size,
            first_ppm: r.first_ppm,
            step_ppm: r.step_ppm,
            obs_mhz: r.obs_mhz,
        })
    }
}

/// Processed 1D NMR frequency-domain spectrum on a descending-ppm `AxisPpm` grid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Spectrum1D {
    pub name: String,
    pub experiment: Experiment1DKind,
    #[serde(with = "axis_ppm_serde")]
    pub axis: AxisPpm,
    /// Absorption-channel intensities across `0..axis.size`.
    pub real: Vec<f64>,
    /// Optional causal dispersion-channel intensities across `0..axis.size`
    /// (automatically reconstructed via Hilbert transform if needed during re-phasing).
    pub imag: Option<Vec<f64>>,
    /// Optional deuterated solvent identifier (e.g. `"CDCl3"`, `"DMSO-d6"`).
    pub solvent: Option<String>,
    /// Optional sample temperature in Kelvin.
    pub temperature_k: Option<f64>,
    /// Cumulative `(phc0_deg, phc1_deg, pivot_frac)` applied since FFT.
    pub phase_applied: (f64, f64, f64),
}

impl PartialEq for Spectrum1D {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.experiment == other.experiment
            && self.axis.label == other.axis.label
            && self.axis.size == other.axis.size
            && (self.axis.first_ppm - other.axis.first_ppm).abs() <= 1e-12
            && (self.axis.step_ppm - other.axis.step_ppm).abs() <= 1e-12
            && (self.axis.obs_mhz - other.axis.obs_mhz).abs() <= 1e-12
            && self.real == other.real
            && self.imag == other.imag
            && self.solvent == other.solvent
            && self.temperature_k == other.temperature_k
            && self.phase_applied == other.phase_applied
    }
}

impl Spectrum1D {
    /// Construct a frequency-domain `Spectrum1D` from an `AxisPpm` and real/optional-imaginary vectors.
    pub fn new(
        name: impl Into<String>,
        experiment: Experiment1DKind,
        axis: AxisPpm,
        real: Vec<f64>,
        imag: Option<Vec<f64>>,
    ) -> Self {
        assert_eq!(axis.size, real.len(), "AxisPpm size must match real.len()");
        if let Some(ref im) = imag {
            assert_eq!(real.len(), im.len(), "imag.len() must match real.len()");
        }
        Self {
            name: name.into(),
            experiment,
            axis,
            real,
            imag,
            solvent: None,
            temperature_k: None,
            phase_applied: (0.0, 0.0, 0.5),
        }
    }

    /// Number of frequency bins.
    #[inline]
    pub fn len(&self) -> usize {
        self.real.len()
    }

    /// Returns `true` if the spectrum has zero points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.real.is_empty()
    }

    /// Chemical shift in ppm at bin index `idx`.
    #[inline]
    pub fn ppm_at(&self, idx: usize) -> f64 {
        self.axis.index_to_ppm(idx as f64)
    }

    /// Frequency step magnitude in ppm (`> 0`).
    #[inline]
    pub fn bin_width_ppm(&self) -> f64 {
        self.axis.step_ppm.abs()
    }

    /// Frequency step magnitude in Hz (`> 0`).
    #[inline]
    pub fn bin_width_hz(&self) -> f64 {
        self.axis.step_ppm.abs() * self.axis.obs_mhz
    }

    /// Shift the entire chemical-shift axis by `delta_ppm` (`new_ppm = old_ppm + delta_ppm`).
    pub fn apply_reference_offset(&mut self, delta_ppm: f64) {
        self.axis.first_ppm += delta_ppm;
    }

    /// Calibrate the chemical-shift axis so a peak currently at `observed_ppm` moves to `target_ppm`.
    /// Returns the applied offset `target_ppm - observed_ppm`.
    pub fn calibrate_reference_peak(&mut self, observed_ppm: f64, target_ppm: f64) -> f64 {
        let delta = target_ppm - observed_ppm;
        self.apply_reference_offset(delta);
        delta
    }
}

/// Parametric synthetic resonance line for exact physics test generation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticLine1D {
    /// Resonance chemical shift in ppm.
    pub ppm: f64,
    /// Integrated area in Hz-intensity units (`amplitude` of time-domain cisoid at `t = 0`).
    pub amplitude: f64,
    /// Natural Lorentzian full width at half maximum (`FWHM`) in Hz.
    pub fwhm_hz: f64,
    /// Optional Gaussian inhomogeneous broadening `FWHM` in Hz (`0.0` for pure Lorentzian).
    pub gauss_fwhm_hz: f64,
    /// Additional per-line phase offset in degrees (`0.0` for pure absorption, `180.0` for inverted `CH2`).
    pub phase_deg: f64,
}

impl SyntheticLine1D {
    /// Pure Lorentzian resonance line with `gauss_fwhm_hz = 0` and `phase_deg = 0`.
    pub fn lorentzian(ppm: f64, amplitude: f64, fwhm_hz: f64) -> Self {
        Self {
            ppm,
            amplitude,
            fwhm_hz,
            gauss_fwhm_hz: 0.0,
            phase_deg: 0.0,
        }
    }
}

/// Synthesize an analytical multi-component complex 1D FID:
/// $$z[n] = \sum_k A_k e^{i \phi_k} \exp\bigl(-\pi W_{L,k} t_n - (\pi W_{G,k} t_n / (2\sqrt{\ln 2}))^2 + i 2\pi \Delta\nu_k t_n\bigr).$$
pub fn synthesize_fid_1d(
    name: impl Into<String>,
    experiment: Experiment1DKind,
    n_td: usize,
    sw_hz: f64,
    obs_mhz: f64,
    carrier_ppm: f64,
    lines: &[SyntheticLine1D],
) -> TimeDomain1D {
    let dt = 1.0 / sw_hz.max(1e-9);
    let inv_two_sqrt_ln2 = 0.5 / (2.0_f64.ln()).sqrt();
    let mut data = vec![[0.0, 0.0]; n_td];
    for line in lines {
        let df_hz = (line.ppm - carrier_ppm) * obs_mhz;
        let phi0 = line.phase_deg.to_radians();
        let r_lor = PI * line.fwhm_hz.max(0.0);
        let r_gau = PI * line.gauss_fwhm_hz.max(0.0) * inv_two_sqrt_ln2;
        for (n, sample) in data.iter_mut().enumerate() {
            let t = (n as f64) * dt;
            let env = line.amplitude * (-r_lor * t - (r_gau * t).powi(2)).exp();
            let ang = phi0 + 2.0 * PI * df_hz * t;
            let (s, c) = ang.sin_cos();
            sample[0] += env * c;
            sample[1] += env * s;
        }
    }
    TimeDomain1D::new(name, experiment, sw_hz, obs_mhz, carrier_ppm, data)
}

/// Evaluate the 1D apodization weight vector $w[0..n_{\text{apod}}]$.
pub fn build_apodization_weights_1d(
    window: Apodization1D,
    n_apod: usize,
    sw_hz: f64,
    first_point_scale: f64,
) -> Vec<f64> {
    if n_apod == 0 {
        return Vec::new();
    }
    let dt = 1.0 / sw_hz.max(1e-9);
    let t_aq = (n_apod as f64) * dt;
    let nf = n_apod as f64;
    let inv_two_sqrt_ln2 = 0.5 / (2.0_f64.ln()).sqrt();
    let mut w = Vec::with_capacity(n_apod);

    match window {
        Apodization1D::None => w.resize(n_apod, 1.0),
        Apodization1D::Exponential { lb_hz } => {
            for n in 0..n_apod {
                let t = (n as f64) * dt;
                w.push((-PI * lb_hz * t).exp());
            }
        }
        Apodization1D::Gaussian { gb_hz } => {
            let g_rate = PI * gb_hz.abs() * inv_two_sqrt_ln2;
            for n in 0..n_apod {
                let t = (n as f64) * dt;
                w.push((-(g_rate * t).powi(2)).exp());
            }
        }
        Apodization1D::LorentzToGauss {
            narrow_lb_hz,
            gb_hz,
            center_frac,
        } => {
            let a = PI * narrow_lb_hz.abs();
            let g_explicit = (PI * gb_hz.abs() * inv_two_sqrt_ln2).powi(2);
            let b = if center_frac > 1e-6 && t_aq > 0.0 {
                let t_max = center_frac.clamp(1e-4, 0.99) * t_aq;
                (a / (2.0 * t_max)).max(g_explicit).max(1e-12)
            } else {
                g_explicit.max(1e-12)
            };
            let mut max_w = 0.0_f64;
            for n in 0..n_apod {
                let t = (n as f64) * dt;
                let val = (a * t - b * t * t).exp();
                max_w = max_w.max(val);
                w.push(val);
            }
            if max_w > 1.0 {
                let inv = 1.0 / max_w;
                for v in &mut w {
                    *v *= inv;
                }
            }
        }
        Apodization1D::CosineBell { power } => {
            let p = power.max(0.0);
            for n in 0..n_apod {
                let c = (0.5 * PI * (n as f64) / nf).cos().max(0.0);
                w.push(c.powf(p));
            }
        }
        Apodization1D::ShiftedSineBell {
            offset_pi_frac,
            power,
        } => {
            let alpha = offset_pi_frac.clamp(0.0, 0.99);
            let p = power.max(0.0);
            for n in 0..n_apod {
                let s = (PI * (alpha + (1.0 - alpha) * (n as f64) / nf))
                    .sin()
                    .max(0.0);
                w.push(s.powf(p));
            }
        }
    }

    w[0] *= first_point_scale;
    w
}

/// Shift out the integer pre-echo samples of a digital oversampling FIR filter (`group_delay_pts >= 0.5`,
/// Moskau, *Concepts Magn. Reson.* **2002**, 15, 164–176) in place so `fid.data[0]` corresponds to
/// $t \approx 0$, retaining only the sub-sample residual `gd_res` in `[-0.5, 0.5]` for `fft_1d`.
pub fn compensate_group_delay_1d(fid: &mut TimeDomain1D) {
    if fid.group_delay_pts >= 0.5 && !fid.data.is_empty() {
        let gd_int = (fid.group_delay_pts.round() as usize).min(fid.data.len().saturating_sub(1));
        if gd_int > 0 {
            fid.data.drain(0..gd_int);
            fid.active_td = fid.active_td.saturating_sub(gd_int).min(fid.data.len());
            fid.group_delay_pts -= gd_int as f64;
        }
    }
}

/// Apply an apodization window and Otting first-point multiplier (`first_point_scale = 0.5` standard)
/// over the active acquired samples `0..fid.active_td`, zeroing any samples beyond `active_td`.
pub fn apply_apodization_1d(fid: &mut TimeDomain1D, window: Apodization1D, first_point_scale: f64) {
    compensate_group_delay_1d(fid);
    let n_apod = fid.active_td.min(fid.data.len());
    let weights = build_apodization_weights_1d(window, n_apod, fid.sw_hz, first_point_scale);
    for (sample, &wt) in fid.data.iter_mut().take(n_apod).zip(weights.iter()) {
        sample[0] *= wt;
        sample[1] *= wt;
    }
    for sample in fid.data.iter_mut().skip(n_apod) {
        *sample = [0.0, 0.0];
    }
}

/// Zero-fill `fid` to at least `target_points` complex samples.
pub fn zero_fill_1d(fid: &mut TimeDomain1D, target_points: usize) {
    if target_points > fid.data.len() {
        fid.data.resize(target_points, [0.0, 0.0]);
    }
}

/// Zero-fill `fid` by `factor` (e.g. `2` for $2\times$ zero-filling) rounded up to the next power of two.
pub fn zero_fill_pow2_1d(fid: &mut TimeDomain1D, factor: usize) {
    let base = fid.data.len().saturating_mul(factor.max(1)).max(1);
    let target = base.next_power_of_two();
    zero_fill_1d(fid, target);
}

/// Transform `TimeDomain1D` to a calibrated descending-ppm `Spectrum1D` via exact `f64` FFT,
/// compensating any causal `group_delay_pts` shift + sub-sample linear phase.
pub fn fft_1d(fid: &TimeDomain1D) -> Spectrum1D {
    let n = fid.data.len().max(1);
    let mut work = vec![[0.0, 0.0]; n];

    // Step 1: Integer shift + sub-sample residual for digital oversampling group delay (Moskau 2002).
    let (gd_int, gd_res) = if fid.group_delay_pts >= 0.5 {
        let gi = fid.group_delay_pts.round() as usize;
        (gi, fid.group_delay_pts - (gi as f64))
    } else if fid.group_delay_pts.abs() <= 0.5 {
        (0usize, fid.group_delay_pts)
    } else {
        (0usize, 0.0)
    };

    for (dst_idx, dst) in work.iter_mut().enumerate() {
        let src_idx = dst_idx + gd_int;
        if src_idx < fid.data.len() {
            let c = fid.data[src_idx];
            *dst = if fid.conjugate_for_fft {
                [c[0], -c[1]]
            } else {
                c
            };
        }
    }

    // Step 2: Forward complex DFT (`-i` exponent).
    fft_arbitrary_in_place(&mut work, false);

    // Step 3: Map to descending-ppm NMR bin order `m_j = (half + n - j) % n` and scale by `2 * dt`.
    let half = n / 2;
    let scale = 2.0 * fid.dwell_s();
    let nf = n as f64;
    let half_f = half as f64;

    let mut real = Vec::with_capacity(n);
    let mut imag = Vec::with_capacity(n);
    for j in 0..n {
        let m = (half + n - (j % n)) % n;
        let mut re = work[m][0] * scale;
        let mut im = work[m][1] * scale;
        if gd_res.abs() > 1e-12 {
            let f_j = (half_f - (j as f64)) / nf;
            let phi = 2.0 * PI * gd_res * f_j;
            let (sp, cp) = phi.sin_cos();
            let (r0, i0) = (re, im);
            re = r0 * cp - i0 * sp;
            im = r0 * sp + i0 * cp;
        }
        real.push(re);
        imag.push(im);
    }

    let sw_ppm = fid.sw_hz / fid.obs_mhz;
    let axis = AxisPpm {
        label: fid.experiment.nucleus().as_str().to_string(),
        size: n,
        first_ppm: fid.carrier_ppm + 0.5 * sw_ppm,
        step_ppm: -sw_ppm / nf,
        obs_mhz: fid.obs_mhz,
    };

    let mut spec = Spectrum1D::new(
        fid.name.clone(),
        fid.experiment.clone(),
        axis,
        real,
        Some(imag),
    );
    spec.solvent = fid.solvent.clone();
    spec
}

/// Reconstruct the causal imaginary dispersion channel of a descending-ppm `Spectrum1D` from its
/// real absorption channel via discrete Hilbert transform (`IFFT -> [1, 2..2, 1, 0..0] -> FFT`,
/// Bartholdi & Ernst 1973; Hoch & Stern 1996).
pub fn hilbert_reconstruct_imaginary_1d(spec: &mut Spectrum1D) {
    let n = spec.len();
    if n <= 1 {
        spec.imag = Some(vec![0.0; n]);
        return;
    }
    let half = n / 2;
    // Map descending-ppm `j` back to raw DFT bin `m`: since `j` maps to `m = (half + n - j) % n`,
    // the inverse index map is symmetric: `j = (half + n - m) % n`.
    let mut work = vec![[0.0, 0.0]; n];
    for (m, dst) in work.iter_mut().enumerate() {
        let j = (half + n - (m % n)) % n;
        *dst = [spec.real[j], 0.0];
    }

    fft_arbitrary_in_place(&mut work, true);
    if n.is_multiple_of(2) {
        for c in work.iter_mut().take(half).skip(1) {
            c[0] *= 2.0;
            c[1] *= 2.0;
        }
        for c in work.iter_mut().skip(half + 1) {
            *c = [0.0, 0.0];
        }
    } else {
        for c in work.iter_mut().take(half + 1).skip(1) {
            c[0] *= 2.0;
            c[1] *= 2.0;
        }
        for c in work.iter_mut().skip(half + 1) {
            *c = [0.0, 0.0];
        }
    }
    fft_arbitrary_in_place(&mut work, false);

    let mut imag = Vec::with_capacity(n);
    for j in 0..n {
        let m = (half + n - (j % n)) % n;
        imag.push(work[m][1]);
    }
    spec.imag = Some(imag);
}

// ============================================================================
// 2. Interactive Pivot & Minimum-Entropy Automatic Phase Correction (Chen 2002)
// ============================================================================

/// Configurable pivot location for first-order phase `PHC1` rotation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Pivot1D {
    /// Fractional coordinate $u \in [0, 1]$ across `0..=(N - 1)` (`0.0` = downfield edge, `0.5` = center).
    Fraction(f64),
    /// Explicit bin index `0..N`.
    Index(usize),
    /// Chemical shift $\delta_{\text{piv}}$ in ppm on `spec.axis`.
    Ppm(f64),
    /// Automatically place the pivot at the bin of maximum complex envelope $|S[k]|$.
    MaxPeak,
}

impl Pivot1D {
    /// Resolve this pivot specification to a normalized fraction $u_{\text{piv}} \in [0, 1]$ on `spec`.
    pub fn resolve_fraction(&self, spec: &Spectrum1D) -> f64 {
        let n = spec.len();
        if n <= 1 {
            return 0.0;
        }
        let denom = (n - 1) as f64;
        match *self {
            Self::Fraction(u) => u.clamp(0.0, 1.0),
            Self::Index(idx) => ((idx.min(n - 1)) as f64 / denom).clamp(0.0, 1.0),
            Self::Ppm(ppm) => (spec.axis.ppm_to_index(ppm) / denom).clamp(0.0, 1.0),
            Self::MaxPeak => {
                let mut best_i = n / 2;
                let mut best_e = -1.0_f64;
                if let Some(ref im) = spec.imag {
                    for (i, (&r, &m)) in spec.real.iter().zip(im.iter()).enumerate() {
                        let e = r * r + m * m;
                        if e > best_e {
                            best_e = e;
                            best_i = i;
                        }
                    }
                } else {
                    for (i, &r) in spec.real.iter().enumerate() {
                        let e = r.abs();
                        if e > best_e {
                            best_e = e;
                            best_i = i;
                        }
                    }
                }
                (best_i as f64 / denom).clamp(0.0, 1.0)
            }
        }
    }
}

/// Normalize an angle in degrees into `[-180.0, 180.0)`.
#[inline]
pub fn wrap_deg_180(deg: f64) -> f64 {
    let mut w = (deg + 180.0).rem_euclid(360.0) - 180.0;
    if (w - 180.0).abs() < 1e-12 {
        w = -180.0;
    }
    w
}

/// Shortest signed angular difference `a_deg - b_deg` wrapped into `[-180.0, 180.0)`.
#[inline]
pub fn phase_diff_deg(a_deg: f64, b_deg: f64) -> f64 {
    wrap_deg_180(a_deg - b_deg)
}

/// Re-express an identical linear phase profile `(phc0_deg, phc1_deg)` from `old_pivot` to `new_pivot`.
pub fn convert_phase_pivot_1d(
    spec: &Spectrum1D,
    phc0_deg: f64,
    phc1_deg: f64,
    old_pivot: Pivot1D,
    new_pivot: Pivot1D,
) -> (f64, f64) {
    let u_old = old_pivot.resolve_fraction(spec);
    let u_new = new_pivot.resolve_fraction(spec);
    let new_phc0 = wrap_deg_180(phc0_deg + phc1_deg * (u_new - u_old));
    (new_phc0, phc1_deg)
}

/// Apply interactive/manual `(phc0_deg, phc1_deg)` phase rotation around `pivot` in place:
/// $$S'[j] = S[j] \exp\bigl(+i\,\text{rad}(\text{PHC0} + \text{PHC1}\cdot(u_j - u_{\text{piv}}))\bigr).$$
/// Automatically reconstructs causal `imag` via Hilbert transform if `spec.imag` is `None`.
pub fn apply_phase_1d(spec: &mut Spectrum1D, phc0_deg: f64, phc1_deg: f64, pivot: Pivot1D) {
    let n = spec.len();
    if n == 0 {
        return;
    }
    if spec.imag.is_none() {
        hilbert_reconstruct_imaginary_1d(spec);
    }
    let u_piv = pivot.resolve_fraction(spec);
    let denom = (n.saturating_sub(1)).max(1) as f64;
    let Some(im_vec) = spec.imag.as_mut() else {
        return;
    };

    for (j, (re, im)) in spec.real.iter_mut().zip(im_vec.iter_mut()).enumerate() {
        let u_j = if n > 1 { (j as f64) / denom } else { 0.0 };
        let phi = (phc0_deg + phc1_deg * (u_j - u_piv)).to_radians();
        let (sp, cp) = phi.sin_cos();
        let (r0, i0) = (*re, *im);
        *re = r0 * cp - i0 * sp;
        *im = r0 * sp + i0 * cp;
    }

    // Update cumulative phase state around the newly applied pivot.
    let (prev_p0, prev_p1, prev_piv) = spec.phase_applied;
    let prev_p0_at_new = prev_p0 + prev_p1 * (u_piv - prev_piv);
    spec.phase_applied = (
        wrap_deg_180(prev_p0_at_new + phc0_deg),
        prev_p1 + phc1_deg,
        u_piv,
    );
}

fn default_auto_mask_aqueous_hdo() -> bool {
    true
}

fn default_biphasic_phc1_prior_weight() -> f64 {
    0.06
}

/// Returns `true` when `solvent` names an aqueous $\text{D}_2\text{O}$ / $\text{H}_2\text{O}$ sample
/// whose residual `~4.70–4.79 ppm` $\text{HDO}$ singlet can dominate `${}^{1}\text{H}$ dynamic range.
pub fn is_aqueous_solvent_label(solvent: &str) -> bool {
    let clean = solvent
        .trim_matches(['<', '>', '"', '\'', ' '])
        .trim()
        .to_ascii_lowercase();
    matches!(
        clean.as_str(),
        "d2o"
            | "h2o"
            | "h2o+d2o"
            | "d2o+h2o"
            | "90% h2o+10% d2o"
            | "90%h2o+10%d2o"
            | "deuterium oxide"
            | "water"
            | "water-d2"
            | "pbs"
    ) || clean.starts_with("h2o+")
        || clean.starts_with("d2o+")
        || clean.contains("90% h2o")
}

/// Configuration for automatic 1D minimum-entropy phase correction
/// (Chen et al., *J. Magn. Reson.* **2002**, 158, 164–168; de Brouwer, *J. Magn. Reson.* **2009**, 201, 230–238).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AutoPhase1DConfig {
    pub pivot: Pivot1D,
    /// Resonance polarity (`None` = use `spec.experiment.default_polarity()`).
    pub polarity: Option<Polarity1D>,
    /// Maximum magnitude of first-order phase `|PHC1|` searched in degrees (default `180.0`).
    pub max_abs_phc1_deg: f64,
    /// Coarse grid step in degrees for `PHC0` over `[-180, 180)` (default `10.0`).
    pub coarse_phc0_step_deg: f64,
    /// Coarse grid step in degrees for `PHC1` over `[-max_abs_phc1_deg, +max_abs_phc1_deg]` (default `15.0`).
    pub coarse_phc1_step_deg: f64,
    /// Weight $\gamma_{\text{neg}}$ of the scale-invariant negative-lobe penalty in `AllPositive` mode.
    pub negative_penalty_weight: f64,
    /// Weight $\gamma_{\text{disp}}$ of the dispersive peak-wing antisymmetry penalty.
    pub dispersion_penalty_weight: f64,
    /// Optional chemical-shift intervals `(ppm_min, ppm_max)` excluded from entropy evaluation
    /// (e.g. residual un-suppressed solvent spike).
    pub excluded_ppm_regions: Vec<(f64, f64)>,
    /// Optional positive reference chemical-shift window `(ppm_min, ppm_max)` used to disambiguate
    /// global $0^\circ$ vs $180^\circ$ sign in `Polarity1D::Biphasic` (`DEPT-135`).
    pub biphasic_positive_ppm: Option<(f64, f64)>,
    /// Automatically mask dominant residual `HDO` / water (`4.60..4.90 ppm`) during `${}^{1}\text{H}$
    /// minimum-entropy autophasing when aqueous `D2O`/`H2O` solvent is detected or an overwhelming
    /// dynamic-range singlet sits in `4.60..4.90 ppm` alongside valid solute peaks (`F29`, default `true`).
    #[serde(default = "default_auto_mask_aqueous_hdo")]
    pub auto_mask_aqueous_hdo: bool,
    /// Gentle quadratic `PHC1` regularization weight $\gamma_{\text{phc1}} (\text{PHC1} / 180^\circ)^2$
    /// in `Polarity1D::Biphasic` (`DEPT-135`) preventing CH2-free unimodal `DEPT-135` spectra from
    /// trapping at `PHC1 = ±180°` boundaries (`F29`, default `0.06`).
    #[serde(default = "default_biphasic_phc1_prior_weight")]
    pub biphasic_phc1_prior_weight: f64,
}

impl Default for AutoPhase1DConfig {
    fn default() -> Self {
        Self {
            pivot: Pivot1D::Fraction(0.5),
            polarity: None,
            max_abs_phc1_deg: 180.0,
            coarse_phc0_step_deg: 10.0,
            coarse_phc1_step_deg: 15.0,
            negative_penalty_weight: 8.0,
            dispersion_penalty_weight: 1.5,
            excluded_ppm_regions: Vec::new(),
            biphasic_positive_ppm: None,
            auto_mask_aqueous_hdo: true,
            biphasic_phc1_prior_weight: 0.06,
        }
    }
}

/// Solution returned by automatic 1D minimum-entropy phase correction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhaseResult1D {
    /// Optimal zero-order phase in degrees at `pivot_frac`.
    pub phc0_deg: f64,
    /// Optimal first-order phase in degrees across the full spectrum (`u = 0..1`).
    pub phc1_deg: f64,
    /// Resolved fractional pivot coordinate $u_{\text{piv}} \in [0, 1]$.
    pub pivot_frac: f64,
    /// Final composite minimum-entropy + dispersion objective value.
    pub objective: f64,
}

fn median_in_place(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len().is_multiple_of(2) {
        0.5 * (vals[mid - 1] + vals[mid])
    } else {
        vals[mid]
    }
}

fn robust_mad(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut buf = vals.to_vec();
    let med = median_in_place(&mut buf);
    for v in &mut buf {
        *v = (*v - med).abs();
    }
    1.4826 * median_in_place(&mut buf)
}

/// Solve optimal `(PHC0, PHC1)` around `config.pivot` via Chen et al. (2002) first-derivative
/// Shannon entropy minimization with scale-invariant negative-lobe and dispersive-wing penalties.
pub fn auto_phase_1d(spec: &Spectrum1D, config: &AutoPhase1DConfig) -> PhaseResult1D {
    let n = spec.len();
    let u_piv = config.pivot.resolve_fraction(spec);
    if n < 8 {
        return PhaseResult1D {
            phc0_deg: 0.0,
            phc1_deg: 0.0,
            pivot_frac: u_piv,
            objective: 0.0,
        };
    }

    // Obtain complex bins (reconstructing imaginary via Hilbert if real-only).
    let imag_owned: Vec<f64>;
    let imag_orig: &[f64] = if let Some(ref im) = spec.imag {
        im.as_slice()
    } else {
        let mut tmp = spec.clone();
        hilbert_reconstruct_imaginary_1d(&mut tmp);
        imag_owned = tmp.imag.unwrap_or_else(|| vec![0.0; n]);
        imag_owned.as_slice()
    };

    let polarity = config
        .polarity
        .unwrap_or_else(|| spec.experiment.default_polarity());
    let denom = (n - 1) as f64;

    // Build inclusion mask excluding user-masked ppm regions.
    let mut active = vec![true; n];
    if !config.excluded_ppm_regions.is_empty() {
        for (j, flag) in active.iter_mut().enumerate() {
            let ppm = spec.ppm_at(j);
            for &(a, b) in &config.excluded_ppm_regions {
                let (lo, hi) = (a.min(b), a.max(b));
                if ppm >= lo && ppm <= hi {
                    *flag = false;
                    break;
                }
            }
        }
    }
    if active.iter().filter(|&&f| f).count() < 8 {
        active.fill(true);
    }

    // Estimate thermal noise floor from complex first-difference MAD (preferring non-HDO bins on 1H).
    let mut diffs = Vec::with_capacity(n - 1);
    let is_proton = spec.experiment == Experiment1DKind::Proton1D;
    for j in 0..(n - 1) {
        if active[j] && active[j + 1] {
            if is_proton {
                let p = spec.ppm_at(j);
                if (4.55..=4.95).contains(&p) {
                    continue;
                }
            }
            let dr = spec.real[j + 1] - spec.real[j];
            let di = imag_orig[j + 1] - imag_orig[j];
            diffs.push((0.5 * (dr * dr + di * di)).sqrt());
        }
    }
    if diffs.len() < 8 {
        diffs.clear();
        for j in 0..(n - 1) {
            if active[j] && active[j + 1] {
                let dr = spec.real[j + 1] - spec.real[j];
                let di = imag_orig[j + 1] - imag_orig[j];
                diffs.push((0.5 * (dr * dr + di * di)).sqrt());
            }
        }
    }
    let sigma_noise = 1.4826 * median_in_place(&mut diffs);
    let tau_neg = 2.5 * sigma_noise;

    // Working references for autophasing (never mutating `spec`; allocated only when a dominant
    // radiation-damped 1H HDO Cauchy pole wing is stripped internally).
    let real_hdo_stripped: Vec<f64>;
    let imag_hdo_stripped: Vec<f64>;
    let mut real_ref: &[f64] = spec.real.as_slice();
    let mut imag_ref: &[f64] = imag_orig;

    // Automatic aqueous HDO / residual water mask (`4.60..4.90 ppm`) + 1-pole Cauchy dispersive-wing
    // stripping on internal phase-evaluation buffers for 1H autophasing (`F29`):
    // triggered when sample solvent is aqueous (`D2O` / `H2O`) OR a high dynamic-range singlet
    // (`>= 8x` max solute envelope and `SNR >= 50`) sits inside `4.60..4.90 ppm`, provided
    // genuine solute signals exist outside the water band so radiation-damped HDO `1/Δω` wings
    // cannot hijack minimum-entropy or negative-excursion phase optimization.
    if config.auto_mask_aqueous_hdo && is_proton {
        let solv_is_aq = spec
            .solvent
            .as_deref()
            .is_some_and(is_aqueous_solvent_label);
        let mut max_env_hdo = 0.0_f64;
        let mut j_hdo = 0usize;
        let mut hdo_band_bins = 0usize;
        let mut max_env_solute = 0.0_f64;
        let mut non_hdo_active = 0usize;
        for (j, &is_act) in active.iter().enumerate() {
            if !is_act {
                continue;
            }
            let ppm = spec.ppm_at(j);
            let ej = (spec.real[j] * spec.real[j] + imag_orig[j] * imag_orig[j]).sqrt();
            if (4.60..=4.90).contains(&ppm) {
                hdo_band_bins += 1;
                if ej > max_env_hdo {
                    max_env_hdo = ej;
                    j_hdo = j;
                }
            } else {
                non_hdo_active += 1;
                if !(4.45..=5.05).contains(&ppm) {
                    max_env_solute = max_env_solute.max(ej);
                }
            }
        }
        let high_dr_hdo = max_env_hdo >= 8.0 * max_env_solute.max(1e-30)
            && max_env_hdo >= 50.0 * sigma_noise.max(1e-30);
        let has_solute = non_hdo_active >= 16 && max_env_solute >= (6.0 * sigma_noise).max(1e-24);
        if (solv_is_aq || high_dr_hdo) && has_solute {
            for (j, flag) in active.iter_mut().enumerate() {
                let ppm = spec.ppm_at(j);
                if (4.60..=4.90).contains(&ppm) {
                    *flag = false;
                }
            }
            // When a prominent water peak (`>= 3x` max solute and `SNR >= 20`) sits in `4.60..4.90 ppm`,
            // fit and subtract its single complex Cauchy/Lorentzian pole `S_hdo(x) = C / (x - dz)` from
            // internal `(real_ref, imag_ref)` buffers so long-range `1/Δω` dispersive skirts outside
            // `[4.60, 4.90] ppm` do not bias solute `PHC0`/`PHC1` (preserving nearby `4.55 ppm` solutes).
            if max_env_hdo >= 3.0 * max_env_solute.max(1e-30)
                && max_env_hdo >= 20.0 * sigma_noise.max(1e-30)
                && hdo_band_bins >= 3
            {
                let mut fit_bins: Vec<usize> = Vec::new();
                let core_floor = 0.15 * max_env_hdo;
                let lo_b = j_hdo.saturating_sub(64);
                let hi_b = (j_hdo + 64).min(n - 1);
                for (j, (&sr, &si)) in spec.real[lo_b..=hi_b]
                    .iter()
                    .zip(&imag_orig[lo_b..=hi_b])
                    .enumerate()
                {
                    let idx = lo_b + j;
                    let ppm = spec.ppm_at(idx);
                    if (4.60..=4.90).contains(&ppm) && (sr * sr + si * si).sqrt() >= core_floor {
                        fit_bins.push(idx);
                    }
                }
                if fit_bins.len() < 3 {
                    fit_bins.clear();
                    for j in j_hdo.saturating_sub(2)..=(j_hdo + 2).min(n - 1) {
                        if (4.60..=4.90).contains(&spec.ppm_at(j)) {
                            fit_bins.push(j);
                        }
                    }
                }
                if fit_bins.len() >= 3 {
                    let mut g00 = 0.0_f64;
                    let mut g01_r = 0.0_f64;
                    let mut g01_i = 0.0_f64;
                    let mut r0 = 0.0_f64;
                    let mut r1_r = 0.0_f64;
                    let mut r1_i = 0.0_f64;
                    for &jk in &fit_bins {
                        let x = (jk as f64) - (j_hdo as f64);
                        let sr = spec.real[jk];
                        let si = imag_orig[jk];
                        let e2 = sr * sr + si * si;
                        g00 += e2;
                        g01_r += sr;
                        g01_i -= si;
                        r0 += x * e2;
                        r1_r += x * sr;
                        r1_i += x * si;
                    }
                    let g11 = fit_bins.len() as f64;
                    let det = g00 * g11 - (g01_r * g01_r + g01_i * g01_i);
                    if det > 1e-24 * g00 * g11 {
                        let dz_r = (g11 * r0 - (g01_r * r1_r - g01_i * r1_i)) / det;
                        let dz_i = -(g01_r * r1_i + g01_i * r1_r) / det;
                        let c_r = (-g01_r * r0 + g00 * r1_r) / det;
                        let c_i = (g01_i * r0 + g00 * r1_i) / det;
                        let max_hw = (hdo_band_bins as f64).max(4.0);
                        if dz_r.is_finite()
                            && dz_i.is_finite()
                            && c_r.is_finite()
                            && c_i.is_finite()
                            && dz_r.abs() <= 4.0
                            && dz_i.abs() >= 0.10
                            && dz_i.abs() <= max_hw
                        {
                            let mut resid_e2 = 0.0_f64;
                            for &jk in &fit_bins {
                                let x = (jk as f64) - (j_hdo as f64);
                                let dr = x - dz_r;
                                let di = -dz_i;
                                let den2 = dr * dr + di * di;
                                let hat_r = (c_r * dr + c_i * di) / den2;
                                let hat_i = (c_i * dr - c_r * di) / den2;
                                let er = spec.real[jk] - hat_r;
                                let ei = imag_orig[jk] - hat_i;
                                resid_e2 += er * er + ei * ei;
                            }
                            if resid_e2 <= 0.25 * g00 {
                                let mut rw = spec.real.clone();
                                let mut iw = imag_orig.to_vec();
                                // Evaluate the exact N-periodic discrete-Fourier cotangent pole kernel
                                // `K_N(z) = (pi / N) * cot((pi / N) * z)` so Nyquist wrap-around ramp
                                // (`cot(u) - 1/u = -u/3`) cannot mimic first-order phase `PHC1`, then
                                // remove the residual `i * beta * C` DC offset via baseline median.
                                let pi_over_n = PI / (n as f64);
                                let mut base_re: Vec<f64> = Vec::with_capacity(n);
                                let mut base_im: Vec<f64> = Vec::with_capacity(n);
                                for (j, ((r_slot, i_slot), &is_act)) in rw
                                    .iter_mut()
                                    .zip(iw.iter_mut())
                                    .zip(active.iter())
                                    .enumerate()
                                {
                                    let x = (j as f64) - (j_hdo as f64);
                                    let u = pi_over_n * (x - dz_r);
                                    let v = -pi_over_n * dz_i;
                                    let den_cot = ((2.0 * v).cosh() - (2.0 * u).cos()).max(1e-30);
                                    let k_r = pi_over_n * (2.0 * u).sin() / den_cot;
                                    let k_i = -pi_over_n * (2.0 * v).sinh() / den_cot;
                                    *r_slot -= c_r * k_r - c_i * k_i;
                                    *i_slot -= c_r * k_i + c_i * k_r;
                                    let ppm = spec.ppm_at(j);
                                    if is_act && !(4.45..=5.05).contains(&ppm) {
                                        base_re.push(*r_slot);
                                        base_im.push(*i_slot);
                                    }
                                }
                                if base_re.len() >= 16 {
                                    let dc_r = median_in_place(&mut base_re);
                                    let dc_i = median_in_place(&mut base_im);
                                    for (r_slot, i_slot) in rw.iter_mut().zip(iw.iter_mut()) {
                                        *r_slot -= dc_r;
                                        *i_slot -= dc_i;
                                    }
                                }
                                real_hdo_stripped = rw;
                                imag_hdo_stripped = iw;
                                real_ref = real_hdo_stripped.as_slice();
                                imag_ref = imag_hdo_stripped.as_slice();
                            }
                        }
                    }
                }
            }
        }
    }

    // Spectral energy centroid `u_c` and phase-invariant complex envelope `|S[j]|`.
    let mut e_sum = 0.0_f64;
    let mut ue_sum = 0.0_f64;
    let mut env = vec![0.0_f64; n];
    let mut env_max = 0.0_f64;
    for j in 0..n {
        let e = real_ref[j] * real_ref[j] + imag_ref[j] * imag_ref[j];
        let ej = e.sqrt();
        env[j] = ej;
        if active[j] {
            env_max = env_max.max(ej);
            let u = (j as f64) / denom;
            e_sum += e;
            ue_sum += u * e;
        }
    }
    let u_c = if e_sum > 1e-30 { ue_sum / e_sum } else { 0.5 };

    // Build a phase-invariant signal+wing mask (`|S[j]|` dilated by `±wing_dilate`) so pure empty
    // baseline thermal noise does not dilute Chen's first-derivative Shannon entropy `H_1(R)`.
    let env_sig_cut = (3.5 * sigma_noise).max(0.005 * env_max);
    let wing_dilate = (n / 256).clamp(12, 48);
    let mut sig_active = vec![false; n];
    let mut dist = wing_dilate + 1;
    for j in 0..n {
        if active[j] && env[j] >= env_sig_cut {
            dist = 0;
        } else {
            dist = dist.saturating_add(1);
        }
        if dist <= wing_dilate && active[j] {
            sig_active[j] = true;
        }
    }
    dist = wing_dilate + 1;
    for j in (0..n).rev() {
        if active[j] && env[j] >= env_sig_cut {
            dist = 0;
        } else {
            dist = dist.saturating_add(1);
        }
        if dist <= wing_dilate && active[j] {
            sig_active[j] = true;
        }
    }
    let sig_indices: Vec<usize> = (0..n).filter(|&j| sig_active[j]).collect();
    let use_sig_mask = sig_indices.len() >= 16;

    // Lightly smooth `env` (`[0.25, 0.5, 0.25]`) to locate all envelope maxima and select isolated
    // singlets (no neighbor peak within `3 * wing_p` bins) for per-peak dispersive symmetry locking.
    let mut env_s = env.clone();
    for j in 1..(n - 1) {
        env_s[j] = 0.25 * env[j - 1] + 0.5 * env[j] + 0.25 * env[j + 1];
    }
    let env_cand_floor = (4.5 * sigma_noise).max(0.015 * env_max);
    let mut all_env_maxima: Vec<usize> = Vec::new();
    for j in 2..(n - 2) {
        if active[j]
            && env_s[j] >= env_cand_floor
            && env_s[j] >= env_s[j - 1]
            && env_s[j] > env_s[j + 1]
        {
            all_env_maxima.push(j);
        }
    }

    let env_strong_thresh = (8.0 * sigma_noise).max(0.04 * env_max);
    let refine_subbin_peak = |j: usize, wing_p: usize| -> (f64, f64, f64, f64) {
        let bg_re = 0.5 * (real_ref[j - wing_p] + real_ref[j + wing_p]);
        let bg_im = 0.5 * (imag_ref[j - wing_p] + imag_ref[j + wing_p]);
        let loc_e2 = |idx: usize| -> f64 {
            let dr = real_ref[idx] - bg_re;
            let di = imag_ref[idx] - bg_im;
            (dr * dr + di * di).max(1e-60)
        };
        let mut jc = j;
        if loc_e2(j - 1) > loc_e2(jc) {
            jc = j - 1;
        }
        if loc_e2(j + 1) > loc_e2(jc) {
            jc = j + 1;
        }
        let (e2_l, e2_c, e2_r) = (loc_e2(jc - 1), loc_e2(jc), loc_e2(jc + 1));
        let (el, ec, er) = (e2_l.sqrt(), e2_c.sqrt(), e2_r.sqrt());
        let sub = if el > 0.15 * ec && er > 0.15 * ec {
            let (ql, qc, qr) = (1.0 / e2_l, 1.0 / e2_c, 1.0 / e2_r);
            let den_q = ql - 2.0 * qc + qr;
            if den_q > 1e-30 {
                (0.5 * (ql - qr) / den_q).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        } else {
            let den = el - 2.0 * ec + er;
            if den.abs() > 1e-18 && ec >= el && ec >= er {
                (0.5 * (el - er) / den).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        };
        ((jc as f64) + sub, ec, bg_re, bg_im)
    };

    // `(sub_bin_center, wing_bins)`
    let mut raw_sym_peaks: Vec<(f64, usize, f64)> = Vec::new();
    for (idx_m, &j) in all_env_maxima.iter().enumerate() {
        if env[j] < env_strong_thresh {
            continue;
        }
        // Estimate envelope half-width in bins.
        let half_e = 0.5 * env_s[j];
        let mut hw = 2usize;
        for d in 1..=24 {
            if j >= d && j + d < n && (env_s[j - d] <= half_e || env_s[j + d] <= half_e) {
                hw = d;
                break;
            }
        }
        let wing_p = (2 * hw + 2).clamp(4, 16);
        if j <= wing_p + 2 || j + wing_p + 3 >= n {
            continue;
        }
        // Envelope must decay below 40% at `±wing_p` and have no neighboring envelope maximum within `3 * wing_p`.
        if env_s[j - wing_p] > 0.40 * env_s[j] || env_s[j + wing_p] > 0.40 * env_s[j] {
            continue;
        }
        let iso_dist = 3 * wing_p;
        let has_neighbor = all_env_maxima
            .iter()
            .enumerate()
            .any(|(k_m, &other_j)| k_m != idx_m && j.abs_diff(other_j) <= iso_dist);
        if has_neighbor {
            continue;
        }
        let (c_pos, ec, _, _) = refine_subbin_peak(j, wing_p);
        raw_sym_peaks.push((c_pos, wing_p, ec));
    }
    // In `Polarity1D::Biphasic` (`DEPT-135`), crowded methine/methylene regions (e.g. carbohydrate
    // ring `CH` carbons in `66..74 ppm`) may leave `< 4` strictly isolated singlets; include
    // moderately separated narrow `13C` lines with tighter wings so intermediate chemical shifts
    // also lock dispersive symmetry.
    if polarity == Polarity1D::Biphasic && raw_sym_peaks.len() < 6 {
        for (idx_m, &j) in all_env_maxima.iter().enumerate() {
            if env[j] < env_strong_thresh {
                continue;
            }
            if raw_sym_peaks
                .iter()
                .any(|&(c, _, _)| ((j as f64) - c).abs() < 4.0)
            {
                continue;
            }
            let half_e = 0.5 * env_s[j];
            let mut hw = 1usize;
            for d in 1..=16 {
                if j >= d && j + d < n && (env_s[j - d] <= half_e || env_s[j + d] <= half_e) {
                    hw = d;
                    break;
                }
            }
            let wing_p = (hw + 2).clamp(3, 10);
            if j <= wing_p + 2 || j + wing_p + 3 >= n {
                continue;
            }
            if env_s[j - wing_p] > 0.50 * env_s[j] || env_s[j + wing_p] > 0.50 * env_s[j] {
                continue;
            }
            let iso_dist = (wing_p + 2).max(5);
            let has_close_neighbor = all_env_maxima
                .iter()
                .enumerate()
                .any(|(k_m, &other_j)| k_m != idx_m && j.abs_diff(other_j) <= iso_dist);
            if has_close_neighbor {
                continue;
            }
            let (c_pos, ec, _, _) = refine_subbin_peak(j, wing_p);
            raw_sym_peaks.push((c_pos, wing_p, ec));
        }
    }
    raw_sym_peaks.sort_by(|a, b| b.2.total_cmp(&a.2));
    let sym_peaks: Vec<(f64, usize)> = raw_sym_peaks
        .into_iter()
        .take(16)
        .map(|(c, w, _)| (c, w))
        .collect();

    // Lanczos-3 (`a = 3`) windowed-sinc complex interpolator so narrow (`~1 bin` FWHM) `13C`/`DEPT`
    // peaks suffer zero linear-interpolation chord asymmetry.
    let sample_complex_at = |pos: f64| -> (f64, f64) {
        let base = pos.floor() as isize;
        let frac = pos - (base as f64);
        if frac.abs() <= 1e-12 {
            let idx = (base.clamp(0, (n - 1) as isize)) as usize;
            (real_ref[idx], imag_ref[idx])
        } else {
            let mut sum_re = 0.0_f64;
            let mut sum_im = 0.0_f64;
            let mut sum_w = 0.0_f64;
            for k in -2isize..=3isize {
                let idx = (base + k).clamp(0, (n - 1) as isize) as usize;
                let x = frac - (k as f64);
                let w = if x.abs() < 1e-12 {
                    1.0
                } else if x.abs() >= 3.0 {
                    0.0
                } else {
                    let px = PI * x;
                    let px3 = px / 3.0;
                    (px.sin() / px) * (px3.sin() / px3)
                };
                sum_re += w * real_ref[idx];
                sum_im += w * imag_ref[idx];
                sum_w += w;
            }
            let inv = 1.0 / sum_w.max(1e-30);
            (sum_re * inv, sum_im * inv)
        }
    };

    let sample_phase_at = |pos: f64, phi0_rad: f64, step_rad: f64| -> f64 {
        let (re, im) = sample_complex_at(pos);
        let phi = phi0_rad + pos * step_rad;
        let (sp, cp) = phi.sin_cos();
        re * cp - im * sp
    };

    // For `Polarity1D::Biphasic` (`DEPT-135`), precompute collar-subtracted complex apex vectors
    // `(c_pos, re_loc, im_loc)` at up to 24 strong envelope maxima. Because a pure absorption
    // Lorentzian of EITHER sign (`+1` `CH`/`CH3` or `-1` `CH2`) has zero imaginary dispersion
    // (`Im(S_loc * exp(i*phi)) == 0`) at its sub-bin envelope center while a dispersive peak has
    // `|Im| == |S_loc|`, this sign-invariant penalty prevents `PHC1 = ±180°` boundary wraps on
    // CH2-free `DEPT-135` spectra (`F29`) without penalizing genuine inverted `CH2` lines.
    let biphasic_apex_peaks: Vec<(f64, f64, f64)> = if polarity == Polarity1D::Biphasic {
        let mut cands: Vec<(f64, f64, f64, f64)> = Vec::new();
        for &j in &all_env_maxima {
            if env[j] < env_strong_thresh {
                continue;
            }
            let half_e = 0.5 * env_s[j];
            let mut hw = 1usize;
            for d in 1..=16 {
                if j >= d && j + d < n && (env_s[j - d] <= half_e || env_s[j + d] <= half_e) {
                    hw = d;
                    break;
                }
            }
            let w_col = (hw + 2).clamp(3, 8);
            if j <= w_col + 2 || j + w_col + 3 >= n {
                continue;
            }
            let (c_pos, ec, bg_re, bg_im) = refine_subbin_peak(j, w_col);
            let (re_c, im_c) = sample_complex_at(c_pos);
            cands.push((c_pos, re_c - bg_re, im_c - bg_im, ec));
        }
        cands.sort_by(|a, b| b.3.total_cmp(&a.3));
        cands
            .into_iter()
            .take(24)
            .map(|(p, r, m, _)| (p, r, m))
            .collect()
    } else {
        Vec::new()
    };

    // Objective evaluator parameterized by `(phi_c_deg, phc1_deg)`.
    let mut r_buf = vec![0.0_f64; n];
    let eval_objective = |phi_c_deg: f64, phc1_deg: f64, r_out: &mut [f64]| -> f64 {
        let p1_rad = phc1_deg.to_radians();
        if p1_rad.abs() > 1.001 * config.max_abs_phc1_deg.to_radians() + 0.5 {
            return 1e12;
        }
        let phi0_rad = phi_c_deg.to_radians() - p1_rad * u_c;
        let step_rad = p1_rad / denom;

        let mut r2_sum = 0.0_f64;
        let mut neg_sum = 0.0_f64;
        let mut d_sum = 0.0_f64;

        if use_sig_mask {
            for &j in &sig_indices {
                let phi = phi0_rad + (j as f64) * step_rad;
                let (sp, cp) = phi.sin_cos();
                let rj = real_ref[j] * cp - imag_ref[j] * sp;
                r_out[j] = rj;
                r2_sum += rj * rj;
                if polarity == Polarity1D::AllPositive && rj < -tau_neg {
                    let ex = -rj - tau_neg;
                    neg_sum += ex * ex;
                }
            }
            for w in sig_indices.windows(2) {
                if w[1] == w[0] + 1 {
                    d_sum += (r_out[w[1]] - r_out[w[0]]).abs();
                }
            }
        } else {
            for j in 0..n {
                let phi = phi0_rad + (j as f64) * step_rad;
                let (sp, cp) = phi.sin_cos();
                let rj = real_ref[j] * cp - imag_ref[j] * sp;
                r_out[j] = rj;
                if active[j] {
                    r2_sum += rj * rj;
                    if polarity == Polarity1D::AllPositive && rj < -tau_neg {
                        let ex = -rj - tau_neg;
                        neg_sum += ex * ex;
                    }
                }
            }
            for j in 0..(n - 1) {
                if active[j] && active[j + 1] {
                    d_sum += (r_out[j + 1] - r_out[j]).abs();
                }
            }
        }

        // First-derivative Shannon entropy H_1(R) (Chen et al. 2002, Eqs. 3-6), normalized to [0, 1]
        // by `ln(N_eff)` so penalty weights remain resolution-invariant across FFT sizes.
        let inv_d = 1.0 / (d_sum + 1e-30);
        let mut h1 = 0.0_f64;
        let mut n_diff = 0usize;
        if use_sig_mask {
            for w in sig_indices.windows(2) {
                if w[1] == w[0] + 1 {
                    n_diff += 1;
                    let hk = (r_out[w[1]] - r_out[w[0]]).abs() * inv_d;
                    if hk > 1e-30 {
                        h1 -= hk * hk.ln();
                    }
                }
            }
        } else {
            for j in 0..(n - 1) {
                if active[j] && active[j + 1] {
                    n_diff += 1;
                    let hk = (r_out[j + 1] - r_out[j]).abs() * inv_d;
                    if hk > 1e-30 {
                        h1 -= hk * hk.ln();
                    }
                }
            }
        }
        let norm_ln = ((n_diff.max(4)) as f64).ln();
        let h1_norm = h1 / norm_ln;

        let p_neg = if polarity == Polarity1D::AllPositive {
            config.negative_penalty_weight * (neg_sum / (r2_sum + 1e-30))
        } else {
            0.0
        };

        // Per-peak normalized dispersive wing-antisymmetry penalty across isolated singlets
        // (equal leverage across downfield and upfield peaks for crisp `PHC1` locking).
        let p_disp = if !sym_peaks.is_empty() && config.dispersion_penalty_weight > 0.0 {
            let mut asym_acc = 0.0_f64;
            for &(c_pos, wing_p) in &sym_peaks {
                let rc = sample_phase_at(c_pos, phi0_rad, step_rad);
                let mut pk_num = 0.0_f64;
                let mut pk_den = rc * rc;
                let d_start = if polarity == Polarity1D::Biphasic || wing_p > 4 {
                    1
                } else {
                    2
                };
                for d in d_start..=wing_p {
                    let df = d as f64;
                    let rl = sample_phase_at(c_pos - df, phi0_rad, step_rad);
                    let rr = sample_phase_at(c_pos + df, phi0_rad, step_rad);
                    let diff = rr - rl;
                    pk_num += diff * diff;
                    pk_den += rl * rl + rr * rr;
                }
                asym_acc += pk_num / (pk_den + 1e-30);
            }
            config.dispersion_penalty_weight * (asym_acc / (sym_peaks.len() as f64))
        } else {
            0.0
        };

        let p_biphasic_reg = if polarity == Polarity1D::Biphasic {
            let apex_disp = if !biphasic_apex_peaks.is_empty() {
                let mut acc = 0.0_f64;
                for &(c_pos, re_loc, im_loc) in &biphasic_apex_peaks {
                    let phi = phi0_rad + c_pos * step_rad;
                    let (sp, cp) = phi.sin_cos();
                    let im_rot = re_loc * sp + im_loc * cp;
                    let e2 = re_loc * re_loc + im_loc * im_loc + 1e-30;
                    acc += (im_rot * im_rot) / e2;
                }
                0.75 * config.dispersion_penalty_weight * (acc / (biphasic_apex_peaks.len() as f64))
            } else {
                0.0
            };
            let phc1_scale = config.max_abs_phc1_deg.max(180.0);
            let prior =
                config.biphasic_phc1_prior_weight.max(0.0) * (phc1_deg / phc1_scale).powi(2);
            apex_disp + prior
        } else {
            0.0
        };

        h1_norm + p_neg + p_disp + p_biphasic_reg
    };

    // Stage 1: Coarse 2D grid scan over `(phi_c, phc1)` keeping top 6 distinct basins.
    let p0_step = config.coarse_phc0_step_deg.clamp(2.0, 30.0);
    let p1_max = config.max_abs_phc1_deg.max(0.0);
    let p1_step = config.coarse_phc1_step_deg.clamp(2.0, 45.0);

    let n_p0 = ((360.0 / p0_step).round() as usize).max(12);
    let n_p1 = if p1_max <= 1e-6 {
        1usize
    } else {
        ((2.0 * p1_max / p1_step).round() as usize + 1).max(3)
    };

    let mut candidates: Vec<(f64, f64, f64)> = Vec::with_capacity(n_p0 * n_p1);
    for i0 in 0..n_p0 {
        let pc = -180.0 + 360.0 * (i0 as f64) / (n_p0 as f64);
        for i1 in 0..n_p1 {
            let p1 = if n_p1 == 1 {
                0.0
            } else {
                -p1_max + 2.0 * p1_max * (i1 as f64) / ((n_p1 - 1) as f64)
            };
            let score = eval_objective(pc, p1, &mut r_buf);
            candidates.push((pc, p1, score));
        }
    }
    candidates.sort_by(|a, b| a.2.total_cmp(&b.2));

    let mut seeds: Vec<(f64, f64)> = Vec::with_capacity(6);
    for &(pc, p1, _) in &candidates {
        let distinct = seeds
            .iter()
            .all(|&(sc, s1)| phase_diff_deg(pc, sc).abs() > 15.0 || (p1 - s1).abs() > 25.0);
        if distinct {
            seeds.push((pc, p1));
            if seeds.len() >= 6 {
                break;
            }
        }
    }
    if seeds.is_empty() {
        seeds.push((0.0, 0.0));
    }

    // Stage 2: Full-resolution 8-direction Hooke–Jeeves pattern refinement down to `0.005 deg`.
    let mut best_pc = seeds[0].0;
    let mut best_p1 = seeds[0].1;
    let mut best_obj = eval_objective(best_pc, best_p1, &mut r_buf);

    let dirs: [(f64, f64); 8] = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    ];

    for (seed_pc, seed_p1) in seeds {
        let mut cur_pc = seed_pc;
        let mut cur_p1 = seed_p1;
        let mut cur_val = eval_objective(cur_pc, cur_p1, &mut r_buf);
        let mut step0 = p0_step * 0.6;
        let mut step1 = if p1_max <= 1e-6 { 0.0 } else { p1_step * 0.6 };

        for _ in 0..80 {
            let mut improved = false;
            for &(d0, d1) in &dirs {
                if step1 == 0.0 && d1 != 0.0 {
                    continue;
                }
                let cand_pc = wrap_deg_180(cur_pc + d0 * step0);
                let cand_p1 = (cur_p1 + d1 * step1).clamp(-p1_max, p1_max);
                let val = eval_objective(cand_pc, cand_p1, &mut r_buf);
                if val + 1e-12 < cur_val {
                    cur_pc = cand_pc;
                    cur_p1 = cand_p1;
                    cur_val = val;
                    improved = true;
                }
            }
            if !improved {
                step0 *= 0.5;
                step1 *= 0.5;
                if step0 < 0.005 && step1 < 0.01 {
                    break;
                }
            }
        }

        if cur_val < best_obj {
            best_obj = cur_val;
            best_pc = cur_pc;
            best_p1 = cur_p1;
        }
    }

    // Stage 3: Global 0 vs 180 deg sign lock on full active array.
    let p1_final_rad = best_p1.to_radians();
    let phi0_final_rad = best_pc.to_radians() - p1_final_rad * u_c;
    let step_final_rad = p1_final_rad / denom;
    for (j, dst) in r_buf.iter_mut().enumerate() {
        let (sp, cp) = (phi0_final_rad + (j as f64) * step_final_rad).sin_cos();
        *dst = real_ref[j] * cp - imag_ref[j] * sp;
    }
    match polarity {
        Polarity1D::AllPositive => {
            let mut max_pos = 0.0_f64;
            let mut max_neg = 0.0_f64;
            for (j, &rj) in r_buf.iter().enumerate() {
                if active[j] {
                    max_pos = max_pos.max(rj);
                    max_neg = max_neg.max(-rj);
                }
            }
            if max_neg > max_pos {
                best_pc = wrap_deg_180(best_pc + 180.0);
                best_obj = eval_objective(best_pc, best_p1, &mut r_buf);
            }
        }
        Polarity1D::Biphasic => {
            if let Some((a, b)) = config.biphasic_positive_ppm {
                let (lo, hi) = (a.min(b), a.max(b));
                let mut band_sum = 0.0_f64;
                for (j, &rj) in r_buf.iter().enumerate() {
                    let ppm = spec.ppm_at(j);
                    if active[j] && ppm >= lo && ppm <= hi && rj.abs() > tau_neg {
                        band_sum += rj * rj.abs();
                    }
                }
                if band_sum < 0.0 {
                    best_pc = wrap_deg_180(best_pc + 180.0);
                    best_obj = eval_objective(best_pc, best_p1, &mut r_buf);
                }
            } else {
                // Automatic `DEPT-135` `0° vs 180°` sign disambiguation (`F29`, Doddrell et al. 1982):
                // methines (`CH`, `68..220 ppm` outside terminal `=CH2` `108..122 ppm`) and primary
                // methyls (`CH3`, `-5..22.5 ppm`) carry positive polarization, while aliphatic
                // methylenes (`CH2`, `24..62 ppm`) carry negative polarization.
                let sig_cut = (4.0 * sigma_noise).max(tau_neg);
                let mut ch_ch3_signed = 0.0_f64;
                let mut ch2_signed = 0.0_f64;
                let mut pos_e2 = 0.0_f64;
                let mut neg_e2 = 0.0_f64;
                for (j, &rj) in r_buf.iter().enumerate() {
                    if !active[j] || rj.abs() <= sig_cut {
                        continue;
                    }
                    let e2 = rj * rj;
                    let s_e2 = rj * rj.abs();
                    if rj >= 0.0 {
                        pos_e2 += e2;
                    } else {
                        neg_e2 += e2;
                    }
                    let ppm = spec.ppm_at(j);
                    let in_ch_band =
                        (68.0..=220.0).contains(&ppm) && !(108.0..=122.0).contains(&ppm);
                    let in_ch3_band = (-5.0..=22.5).contains(&ppm);
                    if in_ch_band || in_ch3_band {
                        ch_ch3_signed += s_e2;
                    } else if (24.0..=62.0).contains(&ppm) {
                        ch2_signed += s_e2;
                    }
                }
                let tot_e2 = pos_e2 + neg_e2;
                let dept_score = ch_ch3_signed - 0.5 * ch2_signed;
                let should_flip = if tot_e2 > 0.0 && dept_score.abs() > 0.05 * tot_e2 {
                    dept_score < 0.0
                } else {
                    neg_e2 > pos_e2
                };
                if should_flip {
                    best_pc = wrap_deg_180(best_pc + 180.0);
                    best_obj = eval_objective(best_pc, best_p1, &mut r_buf);
                }
            }
        }
    }

    let phc0_at_piv = wrap_deg_180(best_pc + best_p1 * (u_piv - u_c));
    PhaseResult1D {
        phc0_deg: phc0_at_piv,
        phc1_deg: best_p1,
        pivot_frac: u_piv,
        objective: best_obj,
    }
}

/// Automatically solve `(PHC0, PHC1)` via `auto_phase_1d` and apply it to `spec` in place.
pub fn auto_phase_and_apply_1d(spec: &mut Spectrum1D, config: &AutoPhase1DConfig) -> PhaseResult1D {
    let sol = auto_phase_1d(spec, config);
    apply_phase_1d(
        spec,
        sol.phc0_deg,
        sol.phc1_deg,
        Pivot1D::Fraction(sol.pivot_frac),
    );
    sol
}

// ============================================================================
// 3. Automatic Baseline Correction (Whittaker Smoother & Bernstein Polynomial)
// ============================================================================

/// Baseline fitting model for 1D NMR spectra
/// (Eilers, *Anal. Chem.* **2003**, 75, 3631–3636; Cobas et al., *J. Magn. Reson.* **2006**, 183, 145–151).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BaselineMethod1D {
    /// Whittaker penalized least squares $(W + \lambda D_2^\top D_2) b = W y$ weighted by the
    /// Cobas continuous-wavelet signal-free mask and iteratively refined residuals.
    Whittaker {
        /// Smoothness penalty $\lambda > 0$ on second differences $\Delta^2 b$ (e.g. `1e5`–`1e7`).
        lambda: f64,
    },
    /// Asymmetric Whittaker smoother ($w_i = p$ when $y_i > b_i$, $1 - p$ when $y_i \le b_i$)
    /// suitable for strictly positive absorption spectra.
    AsymmetricWhittaker { lambda: f64, asymmetry_p: f64 },
    /// Bernstein polynomial basis $B_{k,m}(u) = \binom{m}{k} u^k (1 - u)^{m - k}$ ($u \in [0, 1]$)
    /// fitted over the Cobas signal-free baseline mask (`degree` typically `3..=12`).
    Bernstein { degree: usize },
}

/// Configuration for automatic 1D baseline detection and subtraction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaselineConfig1D {
    pub method: BaselineMethod1D,
    /// Multi-scale Ricker wavelet detection threshold $\kappa$ in baseline MAD $\sigma$ units (default `3.2`).
    pub wavelet_sigma_thresh: f64,
    /// Characteristic Ricker wavelet scale $\sigma_0$ in bins (default `8`).
    pub wavelet_scale_pts: usize,
    /// Morphological half-window dilation in bins around detected peaks to mask Lorentzian wings (default `20`).
    pub peak_dilation_pts: usize,
    /// Number of iteratively reweighted residual-masking passes (default `3`).
    pub refine_iters: usize,
}

impl Default for BaselineConfig1D {
    fn default() -> Self {
        Self {
            method: BaselineMethod1D::Whittaker { lambda: 1.0e6 },
            wavelet_sigma_thresh: 3.2,
            wavelet_scale_pts: 8,
            peak_dilation_pts: 20,
            refine_iters: 3,
        }
    }
}

/// Estimated baseline vector and signal-free classification mask.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaselineResult1D {
    /// Smooth baseline vector $b[0..N]$.
    pub baseline: Vec<f64>,
    /// Signal-free classification mask (`true` = baseline bin, `false` = resonance signal / wing).
    pub is_baseline: Vec<bool>,
    /// Estimated thermal baseline RMS noise $\hat{\sigma}_{\text{base}}$.
    pub baseline_noise_sigma: f64,
}

/// Convolve `y` with a zero-mean Ricker (Mexican-hat) wavelet of scale `sigma_pts`:
/// $\psi_\sigma(k) = (1 - (k / \sigma)^2) \exp(-k^2 / (2 \sigma^2))$, orthogonal to constant and linear trends.
fn ricker_wavelet_response(y: &[f64], sigma_pts: usize) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    let s = (sigma_pts.max(2)) as f64;
    let half = ((4.0 * s).ceil() as usize).min(n.saturating_sub(1));
    let mut kernel = Vec::with_capacity(2 * half + 1);
    let mut k_sum = 0.0_f64;
    for d in -(half as isize)..=(half as isize) {
        let u = (d as f64) / s;
        let val = (1.0 - u * u) * (-0.5 * u * u).exp();
        k_sum += val;
        kernel.push(val);
    }
    // Enforce exact discrete zero mean so constant offsets have 0 response to machine precision.
    let k_mean = k_sum / (kernel.len() as f64);
    let mut l2 = 0.0_f64;
    for v in &mut kernel {
        *v -= k_mean;
        l2 += *v * *v;
    }
    let inv_norm = 1.0 / l2.sqrt().max(1e-30);
    for v in &mut kernel {
        *v *= inv_norm;
    }

    let hi = half as isize;
    let n_isize = n as isize;
    let mut out = vec![0.0_f64; n];
    for (i, dst) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f64;
        for (ki, &kv) in kernel.iter().enumerate() {
            let d = (ki as isize) - hi;
            // Linear edge extrapolation `2 * y[edge] - y[mirror]` preserves linear ramps at boundaries!
            let idx = (i as isize) + d;
            let sample = if idx < 0 {
                let mir = (-idx).min(n_isize - 1) as usize;
                2.0 * y[0] - y[mir]
            } else if idx >= n_isize {
                let mir = (2 * (n_isize - 1) - idx).max(0) as usize;
                2.0 * y[n - 1] - y[mir]
            } else {
                y[idx as usize]
            };
            acc += kv * sample;
        }
        *dst = acc;
    }
    out
}

/// Detect signal-free baseline regions via multi-scale continuous-wavelet thresholding and
/// morphological wing dilation (Cobas et al., *J. Magn. Reson.* **2006**, 183, 145–151).
pub fn detect_signal_free_mask_cobas(
    y: &[f64],
    scale_pts: usize,
    sigma_thresh: f64,
    dilation_pts: usize,
) -> Vec<bool> {
    let n = y.len();
    if n < 8 {
        return vec![true; n];
    }
    let s1 = scale_pts.clamp(2, (n / 8).max(2));
    let s2 = (2 * s1).min((n / 4).max(2));
    let w1 = ricker_wavelet_response(y, s1);
    let w2 = ricker_wavelet_response(y, s2);

    let mut detail: Vec<f64> = w1
        .iter()
        .zip(w2.iter())
        .map(|(&a, &b)| a.abs().max(b.abs()))
        .collect();

    // Iteratively clip signal peaks from `detail` to converge onto the pure baseline wavelet noise floor.
    let mut sorted = detail.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    // Initial estimate from lowest 60% of detail magnitudes.
    let quiet_n = ((0.60 * (n as f64)) as usize).clamp(4, n);
    let mut sigma_w = 1.4826 * median_in_place(&mut sorted[..quiet_n].to_vec()).max(1e-30);
    for _ in 0..3 {
        let cap = 3.5 * sigma_w;
        let mut inliers: Vec<f64> = detail.iter().copied().filter(|&v| v <= cap).collect();
        if inliers.len() >= n / 8 {
            sigma_w = (1.4826 * median_in_place(&mut inliers)).max(1e-30);
        }
    }

    let thresh = sigma_thresh.max(1.5) * sigma_w;
    let mut raw_signal = vec![false; n];
    for (i, &d) in detail.iter().enumerate() {
        if d > thresh {
            raw_signal[i] = true;
        }
    }

    // Dilate detected signal peaks by `dilation_pts` using an O(N) distance counter.
    let mut is_baseline = vec![true; n];
    let mut dist = dilation_pts + 1;
    for i in 0..n {
        if raw_signal[i] {
            dist = 0;
        } else {
            dist = dist.saturating_add(1);
        }
        if dist <= dilation_pts {
            is_baseline[i] = false;
        }
    }
    dist = dilation_pts + 1;
    for i in (0..n).rev() {
        if raw_signal[i] {
            dist = 0;
        } else {
            dist = dist.saturating_add(1);
        }
        if dist <= dilation_pts {
            is_baseline[i] = false;
        }
    }

    // Ensure at least 10% of points remain weighted so baseline remains well-posed on ultra-crowded spectra.
    let base_cnt = is_baseline.iter().filter(|&&b| b).count();
    if base_cnt < (n / 10).max(4) {
        let mut idx_det: Vec<(usize, f64)> = detail.drain(..).enumerate().collect();
        idx_det.sort_by(|a, b| a.1.total_cmp(&b.1));
        is_baseline.fill(false);
        for &(i, _) in idx_det.iter().take((n / 5).max(4).min(n)) {
            is_baseline[i] = true;
        }
    }

    is_baseline
}

/// Solve the symmetric positive-definite pentadiagonal Whittaker penalized least-squares system
/// $(W + \lambda D_2^\top D_2)\, b = W y$ in exact $O(N)$ time via banded $LDL^\top$ factorization
/// (Eilers, *Anal. Chem.* **2003**, 75, 3631–3636).
pub fn fit_whittaker_baseline(y: &[f64], weights: &[f64], lambda: f64) -> Vec<f64> {
    let n = y.len();
    assert_eq!(n, weights.len(), "weights.len() must equal y.len()");
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![y[0]];
    }
    if n == 2 {
        return y.to_vec();
    }

    let lam = lambda.max(0.0);
    let mut d0 = vec![1e-12_f64; n];
    let mut d1 = vec![0.0_f64; n - 1];
    let mut d2 = vec![0.0_f64; n - 2];
    let mut rhs = vec![0.0_f64; n];

    for i in 0..n {
        let wi = weights[i].max(0.0);
        d0[i] += wi;
        rhs[i] = wi * y[i];
    }

    for r in 0..(n - 2) {
        d0[r] += lam;
        d0[r + 1] += 4.0 * lam;
        d0[r + 2] += lam;
        d1[r] -= 2.0 * lam;
        d1[r + 1] -= 2.0 * lam;
        d2[r] += lam;
    }

    // Banded unit-lower-triangular LDL^T factorization: A = L * diag(dd) * L^T.
    let mut dd = vec![0.0_f64; n];
    let mut e1 = vec![0.0_f64; n];
    let mut e2 = vec![0.0_f64; n];

    dd[0] = d0[0].max(1e-30);
    if n >= 2 {
        e1[1] = d1[0] / dd[0];
        dd[1] = (d0[1] - e1[1] * e1[1] * dd[0]).max(1e-30);
    }
    for i in 2..n {
        e2[i] = d2[i - 2] / dd[i - 2];
        e1[i] = (d1[i - 1] - e2[i] * dd[i - 2] * e1[i - 1]) / dd[i - 1];
        dd[i] = (d0[i] - e1[i] * e1[i] * dd[i - 1] - e2[i] * e2[i] * dd[i - 2]).max(1e-30);
    }

    // Forward substitution: L z = rhs
    let mut z = vec![0.0_f64; n];
    z[0] = rhs[0];
    if n >= 2 {
        z[1] = rhs[1] - e1[1] * z[0];
    }
    for i in 2..n {
        z[i] = rhs[i] - e1[i] * z[i - 1] - e2[i] * z[i - 2];
    }

    // Backward substitution: L^T b = D^{-1} z
    let mut b = vec![0.0_f64; n];
    b[n - 1] = z[n - 1] / dd[n - 1];
    if n >= 2 {
        b[n - 2] = z[n - 2] / dd[n - 2] - e1[n - 1] * b[n - 1];
    }
    for i in (0..(n - 2)).rev() {
        b[i] = z[i] / dd[i] - e1[i + 1] * b[i + 1] - e2[i + 2] * b[i + 2];
    }
    b
}

/// Evaluate all degree-`m` Bernstein basis polynomials $B_{0,m}(u), \dots, B_{m,m}(u)$ at $u \in [0, 1]$
/// stably in $O(m^2)$ operations without factorial overflow.
fn eval_bernstein_basis(u: f64, degree: usize, out: &mut [f64]) {
    let u = u.clamp(0.0, 1.0);
    let om = 1.0 - u;
    out[0] = 1.0;
    for j in 1..=degree {
        let mut saved = 0.0_f64;
        for val in out.iter_mut().take(j) {
            let tmp = *val;
            *val = saved + om * tmp;
            saved = u * tmp;
        }
        out[j] = saved;
    }
}

fn solve_real_linear_system(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let m = b.len();
    for col in 0..m {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for (r, row) in a.iter().enumerate().take(m).skip(col + 1) {
            if row[col].abs() > best {
                best = row[col].abs();
                piv = r;
            }
        }
        if best <= 1e-24 {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let diag = a[col][col];
        let (top, bot) = a.split_at_mut(col + 1);
        let prow = &top[col];
        for (off, rrow) in bot.iter_mut().enumerate() {
            let r = col + 1 + off;
            let f = rrow[col] / diag;
            b[r] -= f * b[col];
            for (dst, &pv) in rrow[col..m].iter_mut().zip(&prow[col..m]) {
                *dst -= f * pv;
            }
        }
    }
    let mut x = vec![0.0_f64; m];
    for r in (0..m).rev() {
        let mut s = b[r];
        for k in (r + 1)..m {
            s -= a[r][k] * x[k];
        }
        x[r] = s / a[r][r];
    }
    Some(x)
}

/// Fit a weighted Bernstein polynomial baseline of degree `degree` (`0..=20`) over `[0, 1]`.
pub fn fit_bernstein_baseline(y: &[f64], weights: &[f64], degree: usize) -> Vec<f64> {
    let n = y.len();
    assert_eq!(n, weights.len(), "weights.len() must equal y.len()");
    if n == 0 {
        return Vec::new();
    }
    let m = degree.min(20).min(n.saturating_sub(1));
    let p = m + 1;
    let denom = (n.saturating_sub(1)).max(1) as f64;

    let mut ata = vec![vec![0.0_f64; p]; p];
    let mut aty = vec![0.0_f64; p];
    let mut basis = vec![0.0_f64; p];
    let mut y_sum = 0.0_f64;
    let mut w_sum = 0.0_f64;

    for i in 0..n {
        let wi = weights[i].max(0.0);
        if wi <= 0.0 {
            continue;
        }
        let u = if n > 1 { (i as f64) / denom } else { 0.0 };
        eval_bernstein_basis(u, m, &mut basis);
        y_sum += wi * y[i];
        w_sum += wi;
        for r in 0..p {
            let wbr = wi * basis[r];
            aty[r] += wbr * y[i];
            for c in r..p {
                ata[r][c] += wbr * basis[c];
            }
        }
    }

    let ridge = 1e-10 * (w_sum / (p as f64)).max(1.0);
    for (r, row) in ata.iter_mut().enumerate().take(p) {
        row[r] += ridge;
    }
    for r in 1..p {
        let (up, lo) = ata.split_at_mut(r);
        for (c, urow) in up.iter().enumerate() {
            lo[0][c] = urow[r];
        }
    }

    let fallback = if w_sum > 0.0 { y_sum / w_sum } else { 0.0 };
    let coeffs = solve_real_linear_system(&mut ata, &mut aty).unwrap_or_else(|| vec![fallback; p]);

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let u = if n > 1 { (i as f64) / denom } else { 0.0 };
        eval_bernstein_basis(u, m, &mut basis);
        let val: f64 = basis.iter().zip(coeffs.iter()).map(|(b, c)| b * c).sum();
        out.push(val);
    }
    out
}

/// Estimate the baseline of `y` using `BaselineConfig1D` (Cobas signal-free mask + Whittaker or Bernstein).
pub fn estimate_baseline_1d(y: &[f64], config: &BaselineConfig1D) -> BaselineResult1D {
    let n = y.len();
    if n == 0 {
        return BaselineResult1D {
            baseline: Vec::new(),
            is_baseline: Vec::new(),
            baseline_noise_sigma: 0.0,
        };
    }

    let mut is_baseline = detect_signal_free_mask_cobas(
        y,
        config.wavelet_scale_pts,
        config.wavelet_sigma_thresh,
        config.peak_dilation_pts,
    );

    let mut weights: Vec<f64> = is_baseline
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();

    let mut baseline = match config.method {
        BaselineMethod1D::Whittaker { lambda } => fit_whittaker_baseline(y, &weights, lambda),
        BaselineMethod1D::AsymmetricWhittaker {
            lambda,
            asymmetry_p,
        } => {
            let p = asymmetry_p.clamp(1e-5, 0.2);
            let mut w_asym = vec![1.0_f64; n];
            let mut b = fit_whittaker_baseline(y, &weights, lambda);
            for _ in 0..config.refine_iters.max(6) {
                for i in 0..n {
                    let base_w = if is_baseline[i] { 1.0 } else { 0.05 };
                    w_asym[i] = base_w * if y[i] > b[i] { p } else { 1.0 - p };
                }
                b = fit_whittaker_baseline(y, &w_asym, lambda);
            }
            b
        }
        BaselineMethod1D::Bernstein { degree } => fit_bernstein_baseline(y, &weights, degree),
    };

    // Iteratively refine residual-based outlier masking on masked baseline bins (Whittaker & Bernstein).
    if !matches!(config.method, BaselineMethod1D::AsymmetricWhittaker { .. }) {
        for _ in 0..config.refine_iters {
            let base_res: Vec<f64> = (0..n)
                .filter(|&i| is_baseline[i])
                .map(|i| y[i] - baseline[i])
                .collect();
            let sigma = robust_mad(&base_res);
            if sigma <= 1e-15 {
                break;
            }
            let cut = 3.0 * sigma;
            let mut kept = 0usize;
            for i in 0..n {
                let ok = is_baseline[i] && (y[i] - baseline[i]).abs() <= cut;
                weights[i] = if ok { 1.0 } else { 0.0 };
                if ok {
                    kept += 1;
                }
            }
            if kept < (n / 12).max(4) {
                break;
            }
            for i in 0..n {
                is_baseline[i] = weights[i] > 0.5;
            }
            baseline = match config.method {
                BaselineMethod1D::Whittaker { lambda } => {
                    fit_whittaker_baseline(y, &weights, lambda)
                }
                BaselineMethod1D::Bernstein { degree } => {
                    fit_bernstein_baseline(y, &weights, degree)
                }
                BaselineMethod1D::AsymmetricWhittaker { .. } => unreachable!(),
            };
        }
    }

    let final_res: Vec<f64> = (0..n)
        .filter(|&i| is_baseline[i])
        .map(|i| y[i] - baseline[i])
        .collect();
    let baseline_noise_sigma = robust_mad(&final_res);

    BaselineResult1D {
        baseline,
        is_baseline,
        baseline_noise_sigma,
    }
}

/// Estimate and subtract the 1D baseline from `spec.real` in place, returning `BaselineResult1D`.
pub fn correct_baseline_1d(spec: &mut Spectrum1D, config: &BaselineConfig1D) -> BaselineResult1D {
    let res = estimate_baseline_1d(&spec.real, config);
    for (r, &b) in spec.real.iter_mut().zip(res.baseline.iter()) {
        *r -= b;
    }
    res
}

// ============================================================================
// 4. Time-Domain HSVD / Hankel–Lanczos Solvent Subtraction & Frequency Notch
// ============================================================================

/// Complex arithmetic helpers for HSVD subspace algebra.
#[inline]
fn c_add(a: Complex64, b: Complex64) -> Complex64 {
    [a[0] + b[0], a[1] + b[1]]
}

#[inline]
fn c_sub(a: Complex64, b: Complex64) -> Complex64 {
    [a[0] - b[0], a[1] - b[1]]
}

#[inline]
fn c_mul(a: Complex64, b: Complex64) -> Complex64 {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

#[inline]
fn c_conj(a: Complex64) -> Complex64 {
    [a[0], -a[1]]
}

#[inline]
fn c_abs2(a: Complex64) -> f64 {
    a[0] * a[0] + a[1] * a[1]
}

#[inline]
fn c_abs(a: Complex64) -> f64 {
    a[0].hypot(a[1])
}

#[inline]
fn c_div(a: Complex64, b: Complex64) -> Complex64 {
    let d = c_abs2(b);
    if d <= 1e-60 {
        [0.0, 0.0]
    } else {
        [
            (a[0] * b[0] + a[1] * b[1]) / d,
            (a[1] * b[0] - a[0] * b[1]) / d,
        ]
    }
}

#[inline]
fn c_sqrt(z: Complex64) -> Complex64 {
    let r = c_abs(z);
    if r == 0.0 {
        return [0.0, 0.0];
    }
    let u = (0.5 * (r + z[0]).max(0.0)).sqrt();
    let v = (0.5 * (r - z[0]).max(0.0)).sqrt();
    [u, if z[1] < 0.0 { -v } else { v }]
}

#[inline]
fn c_dot_hermitian(x: &[Complex64], y: &[Complex64]) -> Complex64 {
    let mut s = [0.0, 0.0];
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        // conj(xi) * yi
        s[0] += xi[0] * yi[0] + xi[1] * yi[1];
        s[1] += xi[0] * yi[1] - xi[1] * yi[0];
    }
    s
}

#[inline]
fn c_vec_norm(x: &[Complex64]) -> f64 {
    x.iter().map(|&c| c_abs2(c)).sum::<f64>().sqrt()
}

/// Solve complex linear system $A X = B$ ($A \in \mathbb{C}^{m \times m}$, $B \in \mathbb{C}^{m \times r}$)
/// via column-pivoted Gaussian elimination.
fn solve_complex_matrix_system(
    a: &mut [Vec<Complex64>],
    b: &mut [Vec<Complex64>],
) -> Option<Vec<Vec<Complex64>>> {
    let m = a.len();
    if m == 0 {
        return Some(Vec::new());
    }
    let nrhs = b[0].len();
    for col in 0..m {
        let mut piv = col;
        let mut best = c_abs(a[col][col]);
        for (r, row) in a.iter().enumerate().take(m).skip(col + 1) {
            let v = c_abs(row[col]);
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best <= 1e-24 {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        let diag = a[col][col];
        let (top_a, bot_a) = a.split_at_mut(col + 1);
        let (top_b, bot_b) = b.split_at_mut(col + 1);
        let prow_a = &top_a[col];
        let prow_b = &top_b[col];
        for (rrow_a, rrow_b) in bot_a.iter_mut().zip(bot_b.iter_mut()) {
            let f = c_div(rrow_a[col], diag);
            if c_abs2(f) <= 1e-60 {
                continue;
            }
            for (dst, &pv) in rrow_a[col..m].iter_mut().zip(&prow_a[col..m]) {
                *dst = c_sub(*dst, c_mul(f, pv));
            }
            for (dst, &pv) in rrow_b[0..nrhs].iter_mut().zip(&prow_b[0..nrhs]) {
                *dst = c_sub(*dst, c_mul(f, pv));
            }
        }
    }
    let mut x = vec![vec![[0.0, 0.0]; nrhs]; m];
    for r in (0..m).rev() {
        let diag = a[r][r];
        for rhs_j in 0..nrhs {
            let mut s = b[r][rhs_j];
            for k in (r + 1)..m {
                s = c_sub(s, c_mul(a[r][k], x[k][rhs_j]));
            }
            x[r][rhs_j] = c_div(s, diag);
        }
    }
    Some(x)
}

/// Classical real symmetric Jacobi eigensolver returning eigenvalues (descending) and orthonormal
/// columns $W \in \mathbb{R}^{m \times m}$ ($A W = W \operatorname{diag}(\lambda)$).
fn jacobi_symmetric_eigen_desc(a_in: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let m = a_in.len();
    let mut a = a_in.to_vec();
    let mut v = vec![vec![0.0_f64; m]; m];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    if m <= 1 {
        let ev = if m == 1 { vec![a[0][0]] } else { Vec::new() };
        return (ev, v);
    }

    let mut col_p = vec![0.0_f64; m];
    let mut col_q = vec![0.0_f64; m];
    for _sweep in 0..60 {
        let mut max_off = 0.0_f64;
        for (p, row_p) in a.iter().enumerate().take(m) {
            for &val in row_p.iter().take(m).skip(p + 1) {
                max_off = max_off.max(val.abs());
            }
        }
        if max_off <= 1e-14 {
            break;
        }
        for p in 0..m {
            for q in (p + 1)..m {
                let apq = a[p][q];
                if apq.abs() <= 1e-15 * (a[p][p].abs() + a[q][q].abs()).max(1e-30) {
                    a[p][q] = 0.0;
                    a[q][p] = 0.0;
                    continue;
                }
                let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                let app = a[p][p];
                let aqq = a[q][q];
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                for (r, ((row_a, cp), cq)) in a
                    .iter_mut()
                    .zip(col_p.iter_mut())
                    .zip(col_q.iter_mut())
                    .enumerate()
                    .take(m)
                {
                    if r != p && r != q {
                        let arp = row_a[p];
                        let arq = row_a[q];
                        let rp = c * arp - s * arq;
                        let rq = s * arp + c * arq;
                        row_a[p] = rp;
                        row_a[q] = rq;
                        *cp = rp;
                        *cq = rq;
                    }
                }
                for (r, (&cp, &cq)) in col_p.iter().zip(col_q.iter()).enumerate().take(m) {
                    if r != p && r != q {
                        a[p][r] = cp;
                        a[q][r] = cq;
                    }
                }
                for row_v in v.iter_mut().take(m) {
                    let vrp = row_v[p];
                    let vrq = row_v[q];
                    row_v[p] = c * vrp - s * vrq;
                    row_v[q] = s * vrp + c * vrq;
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let eigvals = order.iter().map(|&i| a[i][i]).collect();
    let mut w_sorted = vec![vec![0.0_f64; m]; m];
    for r in 0..m {
        for (new_c, &old_c) in order.iter().enumerate() {
            w_sorted[r][new_c] = v[r][old_c];
        }
    }
    (eigvals, w_sorted)
}

/// Compute all $K$ complex eigenvalues of a general complex matrix $Z \in \mathbb{C}^{K \times K}$
/// via unitary Householder Hessenberg reduction and Wilkinson-shifted complex Givens QR iteration.
fn complex_matrix_eigenvalues(z_in: &[Vec<Complex64>]) -> Vec<Complex64> {
    let n = z_in.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![z_in[0][0]];
    }
    let mut h = z_in.to_vec();

    // Step 1: Unitary Householder similarity reduction `H <- P H P` (`P = I - 2 u u^H`) to upper Hessenberg.
    for k in 0..n.saturating_sub(2) {
        let m = n - 1 - k;
        let mut x_norm2 = 0.0_f64;
        for row in h.iter().take(n).skip(k + 1) {
            x_norm2 += c_abs2(row[k]);
        }
        let x_norm = x_norm2.sqrt();
        if x_norm <= 1e-30 {
            continue;
        }
        let x0 = h[k + 1][k];
        let x0_abs = c_abs(x0);
        let phase = if x0_abs > 1e-30 {
            [x0[0] / x0_abs, x0[1] / x0_abs]
        } else {
            [1.0, 0.0]
        };
        let mut u = Vec::with_capacity(m);
        u.push(c_add(x0, [phase[0] * x_norm, phase[1] * x_norm]));
        for row in h.iter().take(n).skip(k + 2) {
            u.push(row[k]);
        }
        let u_norm = c_vec_norm(&u);
        if u_norm <= 1e-30 {
            continue;
        }
        let inv_un = 1.0 / u_norm;
        for ui in &mut u {
            ui[0] *= inv_un;
            ui[1] *= inv_un;
        }

        // Left multiply: rows `k+1..n`, cols `k..n` by `(I - 2 u u^H)`.
        let mut gammas = vec![[0.0, 0.0]; n - k];
        for (ui, row) in u.iter().zip(h.iter().skip(k + 1).take(m)) {
            let conj_ui = c_conj(*ui);
            for (g, &val) in gammas.iter_mut().zip(row.iter().skip(k).take(n - k)) {
                *g = c_add(*g, c_mul(conj_ui, val));
            }
        }
        for (ui, row) in u.iter().zip(h.iter_mut().skip(k + 1).take(m)) {
            for (&g, val) in gammas.iter().zip(row.iter_mut().skip(k).take(n - k)) {
                let two_g = [2.0 * g[0], 2.0 * g[1]];
                *val = c_sub(*val, c_mul(*ui, two_g));
            }
        }

        // Right multiply: rows `0..n`, cols `k+1..n` by `(I - 2 u u^H)`.
        for row in h.iter_mut().take(n) {
            let mut beta = [0.0, 0.0];
            for i in 0..m {
                beta = c_add(beta, c_mul(row[k + 1 + i], u[i]));
            }
            let two_beta = [2.0 * beta[0], 2.0 * beta[1]];
            for i in 0..m {
                row[k + 1 + i] = c_sub(row[k + 1 + i], c_mul(two_beta, c_conj(u[i])));
            }
        }

        for row in h.iter_mut().take(n).skip(k + 2) {
            row[k] = [0.0, 0.0];
        }
    }

    // Step 2: Wilkinson-shifted unitary complex Givens QR iteration.
    let mut eigvals = Vec::with_capacity(n);
    let mut n_act = n;
    let mut iter_on_eig = 0usize;
    let mut total_iters = 0usize;
    let max_total_iters = 120 * n;
    let mut givens_c = vec![1.0_f64; n];
    let mut givens_s = vec![[0.0, 0.0]; n];

    while n_act > 1 && total_iters < max_total_iters {
        // Check trailing subdiagonal deflation.
        let sub_last = c_abs(h[n_act - 1][n_act - 2]);
        let diag_scale =
            (c_abs(h[n_act - 2][n_act - 2]) + c_abs(h[n_act - 1][n_act - 1])).max(1e-30);
        if sub_last <= 1e-13 * diag_scale {
            eigvals.push(h[n_act - 1][n_act - 1]);
            n_act -= 1;
            iter_on_eig = 0;
            continue;
        }

        // Find active unreduced Hessenberg block `[low..n_act]`.
        let mut low = n_act - 1;
        while low > 0 {
            let sc = (c_abs(h[low - 1][low - 1]) + c_abs(h[low][low])).max(1e-30);
            if c_abs(h[low][low - 1]) <= 1e-13 * sc {
                h[low][low - 1] = [0.0, 0.0];
                break;
            }
            low -= 1;
        }
        if low == n_act - 1 {
            eigvals.push(h[n_act - 1][n_act - 1]);
            n_act -= 1;
            iter_on_eig = 0;
            continue;
        }

        total_iters += 1;
        iter_on_eig += 1;

        // Compute Wilkinson shift `mu` from trailing 2x2 block (or exceptional shift every 15 steps).
        let mu = if iter_on_eig.is_multiple_of(15) {
            let mag = c_abs(h[n_act - 1][n_act - 2]);
            let ang = 0.73 * (iter_on_eig as f64);
            c_add(h[n_act - 1][n_act - 1], [mag * ang.cos(), mag * ang.sin()])
        } else {
            let a = h[n_act - 2][n_act - 2];
            let b = h[n_act - 2][n_act - 1];
            let c = h[n_act - 1][n_act - 2];
            let d = h[n_act - 1][n_act - 1];
            let delta = [0.5 * (a[0] - d[0]), 0.5 * (a[1] - d[1])];
            let disc = c_sqrt(c_add(c_mul(delta, delta), c_mul(b, c)));
            let l1 = c_add(d, c_add(delta, disc));
            let l2 = c_add(d, c_sub(delta, disc));
            if c_abs2(c_sub(l1, d)) <= c_abs2(c_sub(l2, d)) {
                l1
            } else {
                l2
            }
        };

        for (j, row) in h.iter_mut().enumerate().take(n_act).skip(low) {
            row[j] = c_sub(row[j], mu);
        }

        // Left Givens sweep: H - mu I -> R = Q^H (H - mu I).
        for i in low..(n_act - 1) {
            let alpha = h[i][i];
            let beta = h[i + 1][i];
            let a_abs = c_abs(alpha);
            let b_abs = c_abs(beta);
            let r = a_abs.hypot(b_abs);
            let (ci, si) = if r <= 1e-30 {
                (1.0, [0.0, 0.0])
            } else if a_abs <= 1e-30 {
                (0.0, [1.0, 0.0])
            } else {
                let ci = a_abs / r;
                let conj_p = [alpha[0] / a_abs, -alpha[1] / a_abs];
                let si = c_mul(conj_p, [beta[0] / r, beta[1] / r]);
                (ci, si)
            };
            givens_c[i] = ci;
            givens_s[i] = si;
            let conj_si = c_conj(si);

            let (head, tail) = h.split_at_mut(i + 1);
            let row_top = &mut head[i][i..n_act];
            let row_bot = &mut tail[0][i..n_act];
            for (top_ref, bot_ref) in row_top.iter_mut().zip(row_bot.iter_mut()) {
                let top = *top_ref;
                let bot = *bot_ref;
                *top_ref = c_add([ci * top[0], ci * top[1]], c_mul(conj_si, bot));
                *bot_ref = c_add(
                    [
                        -si[0] * top[0] + si[1] * top[1],
                        -si[0] * top[1] - si[1] * top[0],
                    ],
                    [ci * bot[0], ci * bot[1]],
                );
            }
            h[i + 1][i] = [0.0, 0.0];
        }

        // Right Givens sweep: H_next = R Q + mu I.
        for i in low..(n_act - 1) {
            let ci = givens_c[i];
            let si = givens_s[i];
            let neg_conj_si = [-si[0], si[1]];
            let r_lim = (i + 2).min(n_act);
            for row in h.iter_mut().take(r_lim).skip(low) {
                let left = row[i];
                let right = row[i + 1];
                row[i] = c_add([ci * left[0], ci * left[1]], c_mul(si, right));
                row[i + 1] = c_add(c_mul(neg_conj_si, left), [ci * right[0], ci * right[1]]);
            }
        }

        for (j, row) in h.iter_mut().enumerate().take(n_act).skip(low) {
            row[j] = c_add(row[j], mu);
        }
    }

    for (j, row) in h.iter().enumerate().take(n_act) {
        eigvals.push(row[j]);
    }
    eigvals
}

/// Fast $O(N_H \log N_H)$ Hankel matrix-vector multiplication $u = H v$ and $w = H^H u$
/// for $H_{i,j} = z[i + j]$ ($0 \le i < L$, $0 \le j < M$, $N_H = L + M - 1$).
struct FastHankelOperator {
    l_rows: usize,
    m_cols: usize,
    fft_len_fwd: usize,
    fft_len_adj: usize,
    z_fwd_hat: Vec<Complex64>,
    z_conj_adj_hat: Vec<Complex64>,
}

impl FastHankelOperator {
    fn new(z: &[Complex64], l_rows: usize, m_cols: usize) -> Self {
        let nh = l_rows + m_cols - 1;
        assert!(z.len() >= nh);
        let fft_len_fwd = (l_rows + 2 * m_cols - 1).next_power_of_two();
        let fft_len_adj = (2 * l_rows + m_cols - 1).next_power_of_two();

        let mut z_fwd_hat = vec![[0.0, 0.0]; fft_len_fwd];
        z_fwd_hat[..nh].copy_from_slice(&z[..nh]);
        fft_arbitrary_in_place(&mut z_fwd_hat, false);

        let mut z_conj_adj_hat = vec![[0.0, 0.0]; fft_len_adj];
        for (dst, &src) in z_conj_adj_hat.iter_mut().take(nh).zip(z.iter().take(nh)) {
            *dst = c_conj(src);
        }
        fft_arbitrary_in_place(&mut z_conj_adj_hat, false);

        Self {
            l_rows,
            m_cols,
            fft_len_fwd,
            fft_len_adj,
            z_fwd_hat,
            z_conj_adj_hat,
        }
    }

    /// Compute $u = H v$ ($v \in \mathbb{C}^M \to u \in \mathbb{C}^L$).
    fn mul_vec(&self, v: &[Complex64], out_u: &mut [Complex64]) {
        let mut b = vec![[0.0, 0.0]; self.fft_len_fwd];
        for j in 0..self.m_cols {
            b[self.m_cols - 1 - j] = v[j];
        }
        fft_arbitrary_in_place(&mut b, false);
        for (bi, &zi) in b.iter_mut().zip(self.z_fwd_hat.iter()) {
            *bi = c_mul(zi, *bi);
        }
        fft_arbitrary_in_place(&mut b, true);
        out_u[..self.l_rows]
            .copy_from_slice(&b[(self.m_cols - 1)..(self.m_cols - 1 + self.l_rows)]);
    }

    /// Compute $w = H^H u$ ($u \in \mathbb{C}^L \to w \in \mathbb{C}^M$).
    fn mul_adjoint(&self, u: &[Complex64], out_w: &mut [Complex64]) {
        let mut b = vec![[0.0, 0.0]; self.fft_len_adj];
        for i in 0..self.l_rows {
            b[self.l_rows - 1 - i] = u[i];
        }
        fft_arbitrary_in_place(&mut b, false);
        for (bi, &zi) in b.iter_mut().zip(self.z_conj_adj_hat.iter()) {
            *bi = c_mul(zi, *bi);
        }
        fft_arbitrary_in_place(&mut b, true);
        out_w[..self.m_cols]
            .copy_from_slice(&b[(self.l_rows - 1)..(self.l_rows - 1 + self.m_cols)]);
    }
}

/// Golub–Kahan Lanczos bidiagonalization with two-pass DGKS full reorthogonalization,
/// returning the top-$K$ orthonormal left singular vectors $U_K \in \mathbb{C}^{L \times K}$ of $H$.
fn hankel_lanczos_top_left_singular_vectors(
    op: &FastHankelOperator,
    rank_k: usize,
) -> Vec<Vec<Complex64>> {
    let l = op.l_rows;
    let m = op.m_cols;
    let k_eff = rank_k.clamp(1, l.min(m));
    let m_lan = (k_eff + 10).max(2 * k_eff).min(l.min(m));

    let mut p_vecs: Vec<Vec<Complex64>> = Vec::with_capacity(m_lan);
    let mut q_vecs: Vec<Vec<Complex64>> = Vec::with_capacity(m_lan);
    let mut alphas: Vec<f64> = Vec::with_capacity(m_lan);
    let mut betas: Vec<f64> = Vec::with_capacity(m_lan);

    // Deterministic broadband seed vector `q_0` exciting all frequency modes.
    let mut q0 = vec![[0.0, 0.0]; m];
    q0[0] = [1.0, 0.0];
    for (j, qj) in q0.iter_mut().enumerate().skip(1) {
        let ang = 0.618_033_988_75 * ((j * j) as f64);
        *qj = [
            0.25 * ang.cos() / (j as f64).sqrt(),
            0.25 * ang.sin() / (j as f64).sqrt(),
        ];
    }
    let q0_n = c_vec_norm(&q0).max(1e-30);
    for c in &mut q0 {
        c[0] /= q0_n;
        c[1] /= q0_n;
    }
    q_vecs.push(q0);

    let mut u_buf = vec![[0.0, 0.0]; l];
    let mut v_buf = vec![[0.0, 0.0]; m];

    for j in 0..m_lan {
        op.mul_vec(&q_vecs[j], &mut u_buf);
        if j > 0 {
            let b_prev = betas[j - 1];
            for (u_i, &p_prev) in u_buf.iter_mut().zip(p_vecs[j - 1].iter()) {
                u_i[0] -= b_prev * p_prev[0];
                u_i[1] -= b_prev * p_prev[1];
            }
        }
        // Two-pass DGKS Gram-Schmidt against `p_vecs[0..j]`.
        for _pass in 0..2 {
            for p_prev in &p_vecs {
                let proj = c_dot_hermitian(p_prev, &u_buf);
                for (u_i, &pk) in u_buf.iter_mut().zip(p_prev.iter()) {
                    *u_i = c_sub(*u_i, c_mul(proj, pk));
                }
            }
        }
        let alpha = c_vec_norm(&u_buf);
        if alpha <= 1e-28 {
            break;
        }
        let inv_a = 1.0 / alpha;
        let pj: Vec<Complex64> = u_buf
            .iter()
            .map(|&c| [c[0] * inv_a, c[1] * inv_a])
            .collect();
        p_vecs.push(pj);
        alphas.push(alpha);

        if j + 1 < m_lan {
            op.mul_adjoint(&p_vecs[j], &mut v_buf);
            for (v_i, &qj) in v_buf.iter_mut().zip(q_vecs[j].iter()) {
                v_i[0] -= alpha * qj[0];
                v_i[1] -= alpha * qj[1];
            }
            for _pass in 0..2 {
                for q_prev in &q_vecs {
                    let proj = c_dot_hermitian(q_prev, &v_buf);
                    for (v_i, &qk) in v_buf.iter_mut().zip(q_prev.iter()) {
                        *v_i = c_sub(*v_i, c_mul(proj, qk));
                    }
                }
            }
            let beta = c_vec_norm(&v_buf);
            if beta <= 1e-28 {
                break;
            }
            let inv_b = 1.0 / beta;
            let q_next: Vec<Complex64> = v_buf
                .iter()
                .map(|&c| [c[0] * inv_b, c[1] * inv_b])
                .collect();
            q_vecs.push(q_next);
            betas.push(beta);
        }
    }

    let s_steps = p_vecs.len();
    if s_steps == 0 {
        return Vec::new();
    }
    let k_out = k_eff.min(s_steps);

    // Form `T = B B^T` (`s_steps x s_steps` real symmetric tridiagonal):
    // B has `alphas[0..s_steps]` on diagonal and `betas[0..s_steps-1]` on superdiagonal.
    let mut bbt = vec![vec![0.0_f64; s_steps]; s_steps];
    for i in 0..s_steps {
        let mut d = alphas[i] * alphas[i];
        if i < betas.len() {
            d += betas[i] * betas[i];
        }
        bbt[i][i] = d;
        if i + 1 < s_steps && i < betas.len() {
            let off = betas[i] * alphas[i + 1];
            bbt[i][i + 1] = off;
            bbt[i + 1][i] = off;
        }
    }
    let (_eig, w_mat) = jacobi_symmetric_eigen_desc(&bbt);

    // Project `U_K[r][c] = sum_{j=0..s_steps-1} p_vecs[j][r] * w_mat[j][c]` for `c in 0..k_out`.
    let mut u_k = vec![vec![[0.0, 0.0]; k_out]; l];
    for j in 0..s_steps {
        let pj = &p_vecs[j];
        let wj = &w_mat[j];
        for (r, u_row) in u_k.iter_mut().enumerate() {
            let p_val = pj[r];
            for (c, dst) in u_row.iter_mut().enumerate().take(k_out) {
                let wt = wj[c];
                dst[0] += p_val[0] * wt;
                dst[1] += p_val[1] * wt;
            }
        }
    }
    u_k
}

/// Parameterized damped complex exponential pole extracted by `HSVD`
/// ($z_k[n] = c_k \exp((-\pi W_k + i 2\pi \Delta\nu_k)\, n \Delta t)$).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HsvdPole {
    /// Chemical shift $\delta_k$ in ppm.
    pub ppm: f64,
    /// Frequency offset $\Delta\nu_k$ from carrier in Hz.
    pub freq_hz: f64,
    /// Exponential damping rate $d_k$ in $\text{s}^{-1}$ ($d_k = \pi\,\text{FWHM}_{\text{Hz}}$).
    pub damping_s_inv: f64,
    /// Lorentzian full width at half maximum $W_k = d_k / \pi$ in Hz.
    pub fwhm_hz: f64,
    /// Complex initial amplitude $c_k = |c_k| e^{i \phi_k}$.
    pub complex_amplitude: Complex64,
    /// Modulus $|c_k|$ of initial amplitude.
    pub magnitude: f64,
    /// Phase $\phi_k$ in degrees.
    pub phase_deg: f64,
}

/// Configuration for time-domain `HSVD` / Hankel–Lanczos solvent resonance subtraction
/// (Barkhuijsen, de Beer & van Ormondt, *J. Magn. Reson.* **1987**, 73, 553–557;
/// Pijnappel et al., *J. Magn. Reson.* **1992**, 97, 122–134).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HsvdConfig {
    ///Subspace model order $K$ (number of complex damped exponentials fitted jointly, default `24`).
    pub subspace_rank: usize,
    /// Maximum initial FID complex samples used to form the Hankel matrix (default `1024`).
    pub max_hankel_points: usize,
    /// Chemical-shift intervals `(ppm_min, ppm_max)` identifying solvent resonances to subtract.
    pub solvent_ppm_windows: Vec<(f64, f64)>,
}

impl Default for HsvdConfig {
    fn default() -> Self {
        Self {
            subspace_rank: 24,
            max_hankel_points: 1024,
            solvent_ppm_windows: vec![(4.55, 5.05)],
        }
    }
}

/// Summary of `HSVD` decomposition and solvent pole subtraction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HsvdResult {
    /// All physical signal poles identified in the FID subspace.
    pub all_poles: Vec<HsvdPole>,
    /// Subset of poles falling within `solvent_ppm_windows` that were subtracted from the FID.
    pub subtracted_poles: Vec<HsvdPole>,
}

/// Decompose `fid` into `subspace_rank` damped complex exponentials via Hankel–Lanczos `HSVD`.
pub fn decompose_hsvd_1d(fid: &TimeDomain1D, config: &HsvdConfig) -> Vec<HsvdPole> {
    let n_avail = fid.active_td.min(fid.data.len());
    let nh = n_avail.min(config.max_hankel_points.max(32));
    if nh < 8 {
        return Vec::new();
    }
    let l_rows = nh / 2;
    let m_cols = nh - l_rows + 1;
    let k = config.subspace_rank.clamp(1, (l_rows - 1).min(m_cols));

    let op = FastHankelOperator::new(&fid.data[..nh], l_rows, m_cols);
    let u_k = hankel_lanczos_top_left_singular_vectors(&op, k);
    if u_k.is_empty() || u_k[0].is_empty() {
        return Vec::new();
    }
    let k_act = u_k[0].len();

    // Form `G = U_up^H U_up` and `R = U_up^H U_down` (`k_act x k_act`), where
    // `U_up = u_k[0..l_rows-1]` and `U_down = u_k[1..l_rows]`.
    let mut g_mat = vec![vec![[0.0, 0.0]; k_act]; k_act];
    let mut r_mat = vec![vec![[0.0, 0.0]; k_act]; k_act];
    for row in 0..(l_rows - 1) {
        let u_up = &u_k[row];
        let u_dn = &u_k[row + 1];
        for i in 0..k_act {
            let ui_conj = c_conj(u_up[i]);
            for j in 0..k_act {
                g_mat[i][j] = c_add(g_mat[i][j], c_mul(ui_conj, u_up[j]));
                r_mat[i][j] = c_add(r_mat[i][j], c_mul(ui_conj, u_dn[j]));
            }
        }
    }
    for (i, row_g) in g_mat.iter_mut().enumerate() {
        row_g[i][0] += 1e-12;
    }

    let Some(z_mat) = solve_complex_matrix_system(&mut g_mat, &mut r_mat) else {
        return Vec::new();
    };
    let raw_eigs = complex_matrix_eigenvalues(&z_mat);

    // Filter physical / near-unit-disk poles (`0.05 < |lambda| <= 1.002`; clamp `<= 0.999999` for stability).
    let mut poles_lambda: Vec<Complex64> = Vec::with_capacity(raw_eigs.len());
    for lam in raw_eigs {
        let mag = c_abs(lam);
        if mag.is_finite() && mag > 0.05 && mag <= 1.005 {
            if mag > 0.999_999 {
                let sc = 0.999_999 / mag;
                poles_lambda.push([lam[0] * sc, lam[1] * sc]);
            } else {
                poles_lambda.push(lam);
            }
        }
    }
    if poles_lambda.is_empty() {
        return Vec::new();
    }

    // Solve joint complex Vandermonde least squares `min_c ||V c - z[0..n_fit]||_2^2` via `V^H V c = V^H z`.
    let p_cnt = poles_lambda.len();
    let n_fit = n_avail.min(nh.max(1024));
    let mut vhv = vec![vec![[0.0, 0.0]; p_cnt]; p_cnt];
    let mut vhz = vec![vec![[0.0, 0.0]; 1]; p_cnt];
    let mut powers = vec![[1.0, 0.0]; p_cnt];

    for sample in fid.data.iter().take(n_fit) {
        let zn = *sample;
        for i in 0..p_cnt {
            let vi_conj = c_conj(powers[i]);
            vhz[i][0] = c_add(vhz[i][0], c_mul(vi_conj, zn));
            for j in i..p_cnt {
                vhv[i][j] = c_add(vhv[i][j], c_mul(vi_conj, powers[j]));
            }
        }
        for i in 0..p_cnt {
            powers[i] = c_mul(powers[i], poles_lambda[i]);
        }
    }
    let mut diag_trace = 0.0_f64;
    for (i, row_v) in vhv.iter().enumerate() {
        diag_trace += row_v[i][0].abs();
    }
    let ridge = 1e-11 * (diag_trace / (p_cnt as f64)).max(1.0);
    for i in 0..p_cnt {
        let (head, tail) = vhv.split_at_mut(i + 1);
        let row_i = &mut head[i];
        row_i[i][0] += ridge;
        for (offset, row_j) in tail.iter_mut().take(p_cnt - (i + 1)).enumerate() {
            row_j[i] = c_conj(row_i[i + 1 + offset]);
        }
    }

    let Some(amps_mat) = solve_complex_matrix_system(&mut vhv, &mut vhz) else {
        return Vec::new();
    };

    let dt = fid.dwell_s();
    let sign_conv = if fid.conjugate_for_fft { -1.0 } else { 1.0 };
    let mut out_poles = Vec::with_capacity(p_cnt);

    for (lam, amp_row) in poles_lambda.into_iter().zip(amps_mat) {
        let c = amp_row[0];
        let mag_lam = c_abs(lam).max(1e-30);
        let theta = lam[1].atan2(lam[0]);
        let damping = (-mag_lam.ln() / dt).max(0.0);
        let freq_hz = sign_conv * theta / (2.0 * PI * dt);
        let ppm = fid.carrier_ppm + freq_hz / fid.obs_mhz;
        let amp_mag = c_abs(c);
        let phase_deg = sign_conv * c[1].atan2(c[0]).to_degrees();
        out_poles.push(HsvdPole {
            ppm,
            freq_hz,
            damping_s_inv: damping,
            fwhm_hz: damping / PI,
            complex_amplitude: c,
            magnitude: amp_mag,
            phase_deg,
        });
    }

    out_poles.sort_by(|a, b| b.magnitude.total_cmp(&a.magnitude));
    out_poles
}

/// Subtract time-domain solvent resonances inside `config.solvent_ppm_windows` from `fid` in place
/// via Hankel–Lanczos `HSVD`, preserving overlapping or nearby solute resonances.
pub fn subtract_solvent_hsvd_1d(fid: &mut TimeDomain1D, config: &HsvdConfig) -> HsvdResult {
    compensate_group_delay_1d(fid);
    let all_poles = decompose_hsvd_1d(fid, config);
    let mut subtracted_poles = Vec::new();

    for pole in &all_poles {
        let in_window = config.solvent_ppm_windows.iter().any(|&(a, b)| {
            let (lo, hi) = (a.min(b), a.max(b));
            pole.ppm >= lo && pole.ppm <= hi
        });
        if in_window {
            subtracted_poles.push(*pole);
        }
    }

    if !subtracted_poles.is_empty() {
        let dt = fid.dwell_s();
        let sign_conv = if fid.conjugate_for_fft { -1.0 } else { 1.0 };
        let n_sub = fid.active_td.min(fid.data.len());
        for pole in &subtracted_poles {
            let omega = sign_conv * 2.0 * PI * pole.freq_hz * dt;
            let decay = (-pole.damping_s_inv * dt).exp();
            let lam = [decay * omega.cos(), decay * omega.sin()];
            let mut cur = pole.complex_amplitude;
            for sample in fid.data.iter_mut().take(n_sub) {
                sample[0] -= cur[0];
                sample[1] -= cur[1];
                cur = c_mul(cur, lam);
            }
        }
    }

    HsvdResult {
        all_poles,
        subtracted_poles,
    }
}

/// Configuration for frequency-domain notch / polynomial baseline bridge solvent suppression.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NotchConfig1D {
    pub ppm_min: f64,
    pub ppm_max: f64,
    /// Width of each flanking baseline collar in ppm used to fit the interpolating polynomial (default `0.15`).
    pub collar_ppm: f64,
    /// Polynomial bridge degree (`1..=3`, default `3`).
    pub poly_degree: usize,
}

impl NotchConfig1D {
    pub fn new(ppm_min: f64, ppm_max: f64) -> Self {
        Self {
            ppm_min: ppm_min.min(ppm_max),
            ppm_max: ppm_min.max(ppm_max),
            collar_ppm: 0.15,
            poly_degree: 3,
        }
    }
}

/// Suppress a residual frequency-domain solvent artifact inside `[ppm_min, ppm_max]` by fitting
/// a local polynomial bridge across the flanking collars `[ppm_min - collar, ppm_min]` and
/// `[ppm_max, ppm_max + collar]`.
pub fn suppress_solvent_notch_1d(spec: &mut Spectrum1D, config: &NotchConfig1D) {
    let n = spec.len();
    if n < 4 {
        return;
    }
    let lo = config.ppm_min.min(config.ppm_max);
    let hi = config.ppm_min.max(config.ppm_max);
    let col = config.collar_ppm.abs().max(2.0 * spec.bin_width_ppm());
    let deg = config.poly_degree.clamp(1, 3);
    let m = deg + 1;

    let mid = 0.5 * (lo + hi);
    let scale = (0.5 * (hi - lo) + col).max(1e-9);

    let mut ata = vec![vec![0.0_f64; m]; m];
    let mut aty = vec![0.0_f64; m];
    let mut collar_pts = 0usize;

    for (j, &yj) in spec.real.iter().enumerate() {
        let ppm = spec.ppm_at(j);
        let in_left_collar = ppm >= lo - col && ppm < lo;
        let in_right_collar = ppm > hi && ppm <= hi + col;
        if in_left_collar || in_right_collar {
            collar_pts += 1;
            let x = (ppm - mid) / scale;
            let mut xp = [1.0_f64; 4];
            for d in 1..m {
                xp[d] = xp[d - 1] * x;
            }
            for r in 0..m {
                aty[r] += xp[r] * yj;
                for c in r..m {
                    ata[r][c] += xp[r] * xp[c];
                }
            }
        }
    }

    if collar_pts < m {
        for j in 0..n {
            let ppm = spec.ppm_at(j);
            if ppm >= lo && ppm <= hi {
                spec.real[j] = 0.0;
            }
        }
        return;
    }

    for (r, row) in ata.iter_mut().enumerate().take(m) {
        row[r] += 1e-10;
    }
    for r in 1..m {
        let (up, dn) = ata.split_at_mut(r);
        for (c, urow) in up.iter().enumerate() {
            dn[0][c] = urow[r];
        }
    }
    let coeffs = solve_real_linear_system(&mut ata, &mut aty).unwrap_or_else(|| vec![0.0; m]);

    for j in 0..n {
        let ppm = spec.ppm_at(j);
        if ppm >= lo && ppm <= hi {
            let x = (ppm - mid) / scale;
            let mut val = 0.0_f64;
            let mut xk = 1.0_f64;
            for &ck in coeffs.iter().take(m) {
                val += ck * xk;
                xk *= x;
            }
            spec.real[j] = val;
            if let Some(ref mut im) = spec.imag {
                im[j] = 0.0;
            }
        }
    }
}

// ============================================================================
// 5. Deuterated Solvent Residual & Trace Impurity Reference Table (Fulmer 2010)
// ============================================================================

/// Published residual solvent NMR signal entry (Fulmer et al., *Organometallics* **2010**, 29, 2176–2179).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolventSignalRef {
    pub ppm: f64,
    pub multiplicity: &'static str,
    pub num_lines: u8,
    /// Deuterium heteronuclear coupling ${}^2J_{\text{HD}}$ or ${}^1J_{\text{CD}}$ in Hz (`0.0` for singlets).
    pub j_hd_or_cd_hz: f64,
    pub site_label: &'static str,
}

/// Complete `${}^{1}\text{H}$ and `${}^{13}\text{C}$ reference record for a standard deuterated NMR solvent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeuteratedSolventEntry {
    pub canonical_name: &'static str,
    pub formula: &'static str,
    pub aliases: &'static [&'static str],
    pub h1_residuals: &'static [SolventSignalRef],
    /// Dissolved $\text{H}_2\text{O}$ / $\text{HDO}$ chemical shift in `${}^{1}\text{H}$ NMR (ppm at $298\text{ K}$).
    pub h2o_h1_ppm: f64,
    pub c13_residuals: &'static [SolventSignalRef],
}

/// Published trace laboratory impurity shift in a given deuterated solvent (Fulmer et al. 2010).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceImpurityEntry {
    pub compound: &'static str,
    pub site: &'static str,
    pub solvent: &'static str,
    pub nucleus: &'static str,
    pub ppm: f64,
    pub multiplicity: &'static str,
}

/// Matched trace impurity candidate returned by `match_trace_impurities`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpurityMatch {
    pub entry: TraceImpurityEntry,
    pub delta_ppm: f64,
}

const DEUTERATED_SOLVENTS: &[DeuteratedSolventEntry] = &[
    DeuteratedSolventEntry {
        canonical_name: "CDCl3",
        formula: "CDCl3",
        aliases: &[
            "cdcl3",
            "chloroform-d",
            "chloroform-d1",
            "trichloromethane-d",
        ],
        h1_residuals: &[SolventSignalRef {
            ppm: 7.26,
            multiplicity: "s",
            num_lines: 1,
            j_hd_or_cd_hz: 0.0,
            site_label: "CHCl3",
        }],
        h2o_h1_ppm: 1.56,
        c13_residuals: &[SolventSignalRef {
            ppm: 77.16,
            multiplicity: "t",
            num_lines: 3,
            j_hd_or_cd_hz: 31.6,
            site_label: "CDCl3",
        }],
    },
    DeuteratedSolventEntry {
        canonical_name: "DMSO-d6",
        formula: "(CD3)2SO",
        aliases: &["dmso-d6", "dmso", "(cd3)2so", "dimethylsulfoxide-d6"],
        h1_residuals: &[SolventSignalRef {
            ppm: 2.50,
            multiplicity: "quint",
            num_lines: 5,
            j_hd_or_cd_hz: 1.9,
            site_label: "CHD2SOCD3",
        }],
        h2o_h1_ppm: 3.33,
        c13_residuals: &[SolventSignalRef {
            ppm: 39.52,
            multiplicity: "sept",
            num_lines: 7,
            j_hd_or_cd_hz: 21.0,
            site_label: "(CD3)2SO",
        }],
    },
    DeuteratedSolventEntry {
        canonical_name: "D2O",
        formula: "D2O",
        aliases: &["d2o", "deuterium oxide", "water-d2"],
        h1_residuals: &[SolventSignalRef {
            ppm: 4.79,
            multiplicity: "s",
            num_lines: 1,
            j_hd_or_cd_hz: 0.0,
            site_label: "HDO",
        }],
        h2o_h1_ppm: 4.79,
        c13_residuals: &[],
    },
    DeuteratedSolventEntry {
        canonical_name: "CD3OD",
        formula: "CD3OD",
        aliases: &["cd3od", "meod", "methanol-d4", "cd3od-d4"],
        h1_residuals: &[SolventSignalRef {
            ppm: 3.31,
            multiplicity: "quint",
            num_lines: 5,
            j_hd_or_cd_hz: 1.7,
            site_label: "CHD2OD",
        }],
        h2o_h1_ppm: 4.87,
        c13_residuals: &[SolventSignalRef {
            ppm: 49.00,
            multiplicity: "sept",
            num_lines: 7,
            j_hd_or_cd_hz: 21.4,
            site_label: "CD3OD",
        }],
    },
    DeuteratedSolventEntry {
        canonical_name: "CD3CN",
        formula: "CD3CN",
        aliases: &["cd3cn", "mecn-d3", "acetonitrile-d3"],
        h1_residuals: &[SolventSignalRef {
            ppm: 1.94,
            multiplicity: "quint",
            num_lines: 5,
            j_hd_or_cd_hz: 2.5,
            site_label: "CHD2CN",
        }],
        h2o_h1_ppm: 2.13,
        c13_residuals: &[
            SolventSignalRef {
                ppm: 1.32,
                multiplicity: "sept",
                num_lines: 7,
                j_hd_or_cd_hz: 20.6,
                site_label: "CD3",
            },
            SolventSignalRef {
                ppm: 118.26,
                multiplicity: "s",
                num_lines: 1,
                j_hd_or_cd_hz: 0.0,
                site_label: "CN",
            },
        ],
    },
    DeuteratedSolventEntry {
        canonical_name: "acetone-d6",
        formula: "(CD3)2CO",
        aliases: &["acetone-d6", "(cd3)2co", "acetone"],
        h1_residuals: &[SolventSignalRef {
            ppm: 2.05,
            multiplicity: "quint",
            num_lines: 5,
            j_hd_or_cd_hz: 2.2,
            site_label: "CHD2COCD3",
        }],
        h2o_h1_ppm: 2.84,
        c13_residuals: &[
            SolventSignalRef {
                ppm: 29.84,
                multiplicity: "sept",
                num_lines: 7,
                j_hd_or_cd_hz: 19.4,
                site_label: "CD3",
            },
            SolventSignalRef {
                ppm: 206.26,
                multiplicity: "s",
                num_lines: 1,
                j_hd_or_cd_hz: 0.0,
                site_label: "CO",
            },
        ],
    },
    DeuteratedSolventEntry {
        canonical_name: "THF-d8",
        formula: "C4D8O",
        aliases: &["thf-d8", "tetrahydrofuran-d8", "thf"],
        h1_residuals: &[
            SolventSignalRef {
                ppm: 3.58,
                multiplicity: "m",
                num_lines: 5,
                j_hd_or_cd_hz: 1.7,
                site_label: "OCHD",
            },
            SolventSignalRef {
                ppm: 1.72,
                multiplicity: "m",
                num_lines: 5,
                j_hd_or_cd_hz: 1.7,
                site_label: "CH2CHD",
            },
        ],
        h2o_h1_ppm: 2.46,
        c13_residuals: &[
            SolventSignalRef {
                ppm: 67.21,
                multiplicity: "quint",
                num_lines: 5,
                j_hd_or_cd_hz: 22.3,
                site_label: "OCD2",
            },
            SolventSignalRef {
                ppm: 25.31,
                multiplicity: "quint",
                num_lines: 5,
                j_hd_or_cd_hz: 20.4,
                site_label: "CD2",
            },
        ],
    },
    DeuteratedSolventEntry {
        canonical_name: "C6D6",
        formula: "C6D6",
        aliases: &["c6d6", "benzene-d6"],
        h1_residuals: &[SolventSignalRef {
            ppm: 7.16,
            multiplicity: "s",
            num_lines: 1,
            j_hd_or_cd_hz: 0.0,
            site_label: "C6D5H",
        }],
        h2o_h1_ppm: 0.40,
        c13_residuals: &[SolventSignalRef {
            ppm: 128.06,
            multiplicity: "t",
            num_lines: 3,
            j_hd_or_cd_hz: 24.3,
            site_label: "C6D6",
        }],
    },
    DeuteratedSolventEntry {
        canonical_name: "CD2Cl2",
        formula: "CD2Cl2",
        aliases: &[
            "cd2cl2",
            "dcm-d2",
            "dichloromethane-d2",
            "methylene chloride-d2",
        ],
        h1_residuals: &[SolventSignalRef {
            ppm: 5.32,
            multiplicity: "t",
            num_lines: 3,
            j_hd_or_cd_hz: 1.1,
            site_label: "CHDCl2",
        }],
        h2o_h1_ppm: 1.52,
        c13_residuals: &[SolventSignalRef {
            ppm: 53.84,
            multiplicity: "quint",
            num_lines: 5,
            j_hd_or_cd_hz: 27.2,
            site_label: "CD2Cl2",
        }],
    },
    DeuteratedSolventEntry {
        canonical_name: "toluene-d8",
        formula: "C7D8",
        aliases: &["toluene-d8", "tol-d8", "c7d8", "methylbenzene-d8"],
        h1_residuals: &[
            SolventSignalRef {
                ppm: 2.08,
                multiplicity: "quint",
                num_lines: 5,
                j_hd_or_cd_hz: 2.3,
                site_label: "CHD2",
            },
            SolventSignalRef {
                ppm: 6.97,
                multiplicity: "m",
                num_lines: 5,
                j_hd_or_cd_hz: 0.0,
                site_label: "ArH-para",
            },
            SolventSignalRef {
                ppm: 7.01,
                multiplicity: "m",
                num_lines: 5,
                j_hd_or_cd_hz: 0.0,
                site_label: "ArH-ortho",
            },
            SolventSignalRef {
                ppm: 7.09,
                multiplicity: "m",
                num_lines: 5,
                j_hd_or_cd_hz: 0.0,
                site_label: "ArH-meta",
            },
        ],
        h2o_h1_ppm: 0.43,
        c13_residuals: &[
            SolventSignalRef {
                ppm: 20.43,
                multiplicity: "sept",
                num_lines: 7,
                j_hd_or_cd_hz: 19.3,
                site_label: "CD3",
            },
            SolventSignalRef {
                ppm: 125.13,
                multiplicity: "t",
                num_lines: 3,
                j_hd_or_cd_hz: 24.0,
                site_label: "C4",
            },
            SolventSignalRef {
                ppm: 127.96,
                multiplicity: "t",
                num_lines: 3,
                j_hd_or_cd_hz: 24.0,
                site_label: "C3/C5",
            },
            SolventSignalRef {
                ppm: 128.87,
                multiplicity: "t",
                num_lines: 3,
                j_hd_or_cd_hz: 24.0,
                site_label: "C2/C6",
            },
            SolventSignalRef {
                ppm: 137.48,
                multiplicity: "s",
                num_lines: 1,
                j_hd_or_cd_hz: 0.0,
                site_label: "C1-ipso",
            },
        ],
    },
];

const TRACE_IMPURITIES: &[TraceImpurityEntry] = &[
    // Water / TMS / grease / common solvents in CDCl3 (Fulmer et al. 2010)
    TraceImpurityEntry {
        compound: "H2O",
        site: "OH",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 1.56,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "TMS",
        site: "Si(CH3)4",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 0.00,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "TMS",
        site: "Si(CH3)4",
        solvent: "CDCl3",
        nucleus: "13C",
        ppm: 0.00,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "silicone grease",
        site: "Si(CH3)2O",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 0.07,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "silicone grease",
        site: "Si(CH3)2O",
        solvent: "CDCl3",
        nucleus: "13C",
        ppm: 1.04,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "H-grease",
        site: "CH2",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 1.26,
        multiplicity: "br s",
    },
    TraceImpurityEntry {
        compound: "H-grease",
        site: "CH2",
        solvent: "CDCl3",
        nucleus: "13C",
        ppm: 29.76,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "ethyl acetate",
        site: "OCH2",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 4.12,
        multiplicity: "q",
    },
    TraceImpurityEntry {
        compound: "ethyl acetate",
        site: "COCH3",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 2.05,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "ethyl acetate",
        site: "CH2CH3",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 1.26,
        multiplicity: "t",
    },
    TraceImpurityEntry {
        compound: "ethyl acetate",
        site: "CO",
        solvent: "CDCl3",
        nucleus: "13C",
        ppm: 171.36,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "diethyl ether",
        site: "OCH2",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 3.48,
        multiplicity: "q",
    },
    TraceImpurityEntry {
        compound: "diethyl ether",
        site: "CH3",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 1.21,
        multiplicity: "t",
    },
    TraceImpurityEntry {
        compound: "dichloromethane",
        site: "CH2Cl2",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 5.30,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "dichloromethane",
        site: "CH2Cl2",
        solvent: "CDCl3",
        nucleus: "13C",
        ppm: 53.52,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "acetone",
        site: "CH3",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 2.17,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "acetone",
        site: "CO",
        solvent: "CDCl3",
        nucleus: "13C",
        ppm: 207.07,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "methanol",
        site: "CH3",
        solvent: "CDCl3",
        nucleus: "1H",
        ppm: 3.49,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "methanol",
        site: "CH3",
        solvent: "CDCl3",
        nucleus: "13C",
        ppm: 50.41,
        multiplicity: "s",
    },
    // DMSO-d6 common impurities (Fulmer et al. 2010)
    TraceImpurityEntry {
        compound: "H2O",
        site: "OH",
        solvent: "DMSO-d6",
        nucleus: "1H",
        ppm: 3.33,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "TMS",
        site: "Si(CH3)4",
        solvent: "DMSO-d6",
        nucleus: "1H",
        ppm: 0.00,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "ethyl acetate",
        site: "OCH2",
        solvent: "DMSO-d6",
        nucleus: "1H",
        ppm: 4.03,
        multiplicity: "q",
    },
    TraceImpurityEntry {
        compound: "ethyl acetate",
        site: "COCH3",
        solvent: "DMSO-d6",
        nucleus: "1H",
        ppm: 1.99,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "ethyl acetate",
        site: "CH2CH3",
        solvent: "DMSO-d6",
        nucleus: "1H",
        ppm: 1.17,
        multiplicity: "t",
    },
    TraceImpurityEntry {
        compound: "dichloromethane",
        site: "CH2Cl2",
        solvent: "DMSO-d6",
        nucleus: "1H",
        ppm: 5.76,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "dichloromethane",
        site: "CH2Cl2",
        solvent: "DMSO-d6",
        nucleus: "13C",
        ppm: 54.95,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "DMF",
        site: "CHO",
        solvent: "DMSO-d6",
        nucleus: "1H",
        ppm: 7.95,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "DMF",
        site: "NCH3-a",
        solvent: "DMSO-d6",
        nucleus: "1H",
        ppm: 2.89,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "DMF",
        site: "NCH3-b",
        solvent: "DMSO-d6",
        nucleus: "1H",
        ppm: 2.73,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "acetone",
        site: "CH3",
        solvent: "DMSO-d6",
        nucleus: "1H",
        ppm: 2.09,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "methanol",
        site: "CH3",
        solvent: "DMSO-d6",
        nucleus: "1H",
        ppm: 3.16,
        multiplicity: "d",
    },
    // D2O, CD3OD, CD3CN, acetone-d6, THF-d8, C6D6, CD2Cl2, toluene-d8 water & standards
    TraceImpurityEntry {
        compound: "DSS",
        site: "Si(CH3)3",
        solvent: "D2O",
        nucleus: "1H",
        ppm: 0.00,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "1,4-dioxane",
        site: "CH2",
        solvent: "D2O",
        nucleus: "1H",
        ppm: 3.75,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "1,4-dioxane",
        site: "CH2",
        solvent: "D2O",
        nucleus: "13C",
        ppm: 67.19,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "methanol",
        site: "CH3",
        solvent: "D2O",
        nucleus: "1H",
        ppm: 3.34,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "methanol",
        site: "CH3",
        solvent: "D2O",
        nucleus: "13C",
        ppm: 49.50,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "H2O",
        site: "OH",
        solvent: "CD3OD",
        nucleus: "1H",
        ppm: 4.87,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "H2O",
        site: "OH",
        solvent: "CD3CN",
        nucleus: "1H",
        ppm: 2.13,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "H2O",
        site: "OH",
        solvent: "acetone-d6",
        nucleus: "1H",
        ppm: 2.84,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "H2O",
        site: "OH",
        solvent: "THF-d8",
        nucleus: "1H",
        ppm: 2.46,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "H2O",
        site: "OH",
        solvent: "C6D6",
        nucleus: "1H",
        ppm: 0.40,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "H2O",
        site: "OH",
        solvent: "CD2Cl2",
        nucleus: "1H",
        ppm: 1.52,
        multiplicity: "s",
    },
    TraceImpurityEntry {
        compound: "H2O",
        site: "OH",
        solvent: "toluene-d8",
        nucleus: "1H",
        ppm: 0.43,
        multiplicity: "s",
    },
];

/// Return the complete built-in table of 10 standard deuterated NMR solvents (Fulmer et al. 2010).
pub fn all_deuterated_solvents() -> &'static [DeuteratedSolventEntry] {
    DEUTERATED_SOLVENTS
}

/// Look up a deuterated NMR solvent entry by canonical name or case-insensitive alias.
pub fn lookup_deuterated_solvent(name: &str) -> Option<&'static DeuteratedSolventEntry> {
    let key = name.trim().to_ascii_lowercase();
    DEUTERATED_SOLVENTS.iter().find(|entry| {
        entry.canonical_name.eq_ignore_ascii_case(&key)
            || entry.aliases.iter().any(|a| a.eq_ignore_ascii_case(&key))
    })
}

/// Return published residual solvent signals for `(solvent_name, nucleus)`.
pub fn solvent_residual_shifts(
    solvent_name: &str,
    nucleus: &Nucleus1D,
) -> Option<&'static [SolventSignalRef]> {
    let entry = lookup_deuterated_solvent(solvent_name)?;
    match nucleus {
        Nucleus1D::H1 => Some(entry.h1_residuals),
        Nucleus1D::C13 => Some(entry.c13_residuals),
        _ => None,
    }
}

/// Return all built-in trace laboratory impurity reference entries.
pub fn all_trace_impurities() -> &'static [TraceImpurityEntry] {
    TRACE_IMPURITIES
}

/// Find trace laboratory impurities in `solvent_name` for `nucleus` within `±tol_ppm` of `observed_ppm`,
/// sorted by ascending `|delta_ppm|`.
pub fn match_trace_impurities(
    solvent_name: &str,
    nucleus: &Nucleus1D,
    observed_ppm: f64,
    tol_ppm: f64,
) -> Vec<ImpurityMatch> {
    let Some(solv) = lookup_deuterated_solvent(solvent_name) else {
        return Vec::new();
    };
    let nuc_str = nucleus.as_str();
    let tol = tol_ppm.abs();
    let mut matches: Vec<ImpurityMatch> = TRACE_IMPURITIES
        .iter()
        .filter(|e| e.solvent == solv.canonical_name && e.nucleus == nuc_str)
        .filter_map(|&e| {
            let d = observed_ppm - e.ppm;
            (d.abs() <= tol).then_some(ImpurityMatch {
                entry: e,
                delta_ppm: d,
            })
        })
        .collect();
    matches.sort_by(|a, b| a.delta_ppm.abs().total_cmp(&b.delta_ppm.abs()));
    matches
}

/// Automatically calibrate `spec`'s chemical shift axis to the primary Fulmer (2010) residual
/// solvent peak within `±search_window_ppm`, returning the applied shift offset `delta_ppm`.
pub fn auto_calibrate_solvent_1d(
    spec: &mut Spectrum1D,
    solvent_name: &str,
    search_window_ppm: f64,
) -> Result<f64, String> {
    let nuc = spec.experiment.nucleus();
    let refs = solvent_residual_shifts(solvent_name, &nuc).ok_or_else(|| {
        format!(
            "no reference shifts registered for solvent '{}' on nucleus '{}'",
            solvent_name,
            nuc.as_str()
        )
    })?;
    let primary = refs.first().ok_or_else(|| {
        format!(
            "solvent '{}' has zero residual signals on nucleus '{}'",
            solvent_name,
            nuc.as_str()
        )
    })?;

    let win = search_window_ppm.abs().max(0.02);
    let lo = primary.ppm - win;
    let hi = primary.ppm + win;

    let n = spec.len();
    let mut best_j: Option<usize> = None;
    let mut best_y = f64::NEG_INFINITY;
    for j in 1..n.saturating_sub(1) {
        let ppm = spec.ppm_at(j);
        if ppm >= lo && ppm <= hi && spec.real[j] > best_y {
            best_y = spec.real[j];
            best_j = Some(j);
        }
    }

    let j = best_j.ok_or_else(|| {
        format!(
            "no spectral points inside [{:.3}, {:.3}] ppm for solvent '{}'",
            lo, hi, solvent_name
        )
    })?;

    // Parabolic 3-point sub-bin peak centroid.
    let (yl, yc, yr) = (spec.real[j - 1], spec.real[j], spec.real[j + 1]);
    let denom = yl - 2.0 * yc + yr;
    let sub = if denom.abs() > 1e-18 && yc >= yl && yc >= yr {
        (0.5 * (yl - yr) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    let obs_ppm = spec.axis.index_to_ppm((j as f64) + sub);
    spec.solvent = lookup_deuterated_solvent(solvent_name).map(|s| s.canonical_name.to_string());
    Ok(spec.calibrate_reference_peak(obs_ppm, primary.ppm))
}

/// IUPAC 2001/2008 primary organic `${}^{13}\text{C}$ unified frequency ratio
/// $\Xi({}^{13}\text{C}, \text{Me}_4\text{Si } 1\%\text{ in CDCl}_3) = 0.251\,450\,20$
/// (Harris et al., *Pure Appl. Chem.* **2001**, 73, 1795–1818; **2008**, 80, 59–84).
pub const XI_C13_TMS: f64 = 0.251_450_20;

/// IUPAC / biomolecular aqueous `${}^{13}\text{C}$ unified frequency ratio
/// $\Xi({}^{13}\text{C}, \text{DSS in D}_2\text{O}) = 0.251\,449\,53$
/// (Wishart et al., *J. Biomol. NMR* **1995**, 6, 135–140; Markley et al.,
/// *Pure Appl. Chem.* **1998**, 70, 117–142; Harris et al. **2001**).
pub const XI_C13_DSS: f64 = 0.251_449_53;

/// Chemical shift of aqueous $\text{DSS}$ `${}^{13}\text{C}$ methyls ($\delta_{\text{DSS}} \equiv 0\text{ ppm}$)
/// when measured on a spectrometer `BF1`/`SF` locked to the organic $\text{TMS}$ `${}^{13}\text{C}$ scale
/// in $\text{D}_2\text{O}$ at $298\text{ K}$ ($\delta_{\text{TMS}}(\text{DSS}) = -2.371\text{ ppm}$,
/// $\nu_{\text{ref,DSS}} / \nu_{\text{ref,TMS}} = 1 - 2.371 \times 10^{-6}$;
/// Harris et al. 2001/2008; Wishart et al. 1995).
pub const C13_DSS_VS_TMS_OFFSET_PPM: f64 = -2.371;

/// Chemical-shift zero-point standard (`TMS` organic vs `DSS` aqueous biomolecular) for `${}^{13}\text{C}$
/// scale conversion (`F29`, Harris et al. 2001/2008; Wishart et al. 1995).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReferenceStandard {
    /// Tetramethylsilane (`1% TMS in CDCl3`, organic chemistry primary standard).
    Tms,
    /// Sodium 3-(trimethylsilyl)-1-propanesulfonate (`DSS in D2O`, aqueous/BMRB standard).
    Dss,
}

impl ReferenceStandard {
    /// Parse `"tms"` or `"dss"` (case-insensitive).
    pub fn parse(label: &str) -> Result<Self, String> {
        match label.trim().to_ascii_lowercase().as_str() {
            "tms" | "me4si" => Ok(Self::Tms),
            "dss" | "tsp-dss" => Ok(Self::Dss),
            other => Err(format!(
                "unknown 13C reference standard {other:?} (expected `dss` or `tms`)"
            )),
        }
    }

    /// Canonical lowercase label (`"tms"` or `"dss"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tms => "tms",
            Self::Dss => "dss",
        }
    }

    /// IUPAC unified `${}^{13}\text{C}$ $\Xi$ ratio for this standard.
    pub fn xi_c13(&self) -> f64 {
        match self {
            Self::Tms => XI_C13_TMS,
            Self::Dss => XI_C13_DSS,
        }
    }
}

/// Convert a `${}^{13}\text{C}$ chemical shift `ppm` between organic `TMS` (`BF1`/`SF` lock) and
/// aqueous `DSS` (`BMRB`) zero-point scales using the exact spectrometer reference-frequency ratio
/// $\nu_{\text{ref}}(\text{DSS}) / \nu_{\text{ref}}(\text{TMS}) = 1 + \text{C13\_DSS\_VS\_TMS\_OFFSET\_PPM} \times 10^{-6}$
/// (`-2.371 ppm` when converting `DSS -> TMS`, `+2.371 ppm` when converting `TMS -> DSS`).
#[inline]
pub fn convert_c13_ppm_reference(ppm: f64, from: ReferenceStandard, to: ReferenceStandard) -> f64 {
    if from == to {
        return ppm;
    }
    let eps_ppm = match (from, to) {
        (ReferenceStandard::Dss, ReferenceStandard::Tms) => C13_DSS_VS_TMS_OFFSET_PPM,
        (ReferenceStandard::Tms, ReferenceStandard::Dss) => {
            -C13_DSS_VS_TMS_OFFSET_PPM / (1.0 + C13_DSS_VS_TMS_OFFSET_PPM * 1.0e-6)
        }
        _ => 0.0,
    };
    ppm + (1.0 + ppm * 1.0e-6) * eps_ppm
}

/// Convert a `${}^{13}\text{C}$ chemical shift `ppm` using the pure lock-free IUPAC unified
/// $\Xi({}^{13}\text{C})$ ratio (`0.25145020` vs `0.25144953`, Harris et al. 2001/2008).
#[inline]
pub fn convert_c13_ppm_xi_ratio(ppm: f64, from: ReferenceStandard, to: ReferenceStandard) -> f64 {
    if from == to {
        return ppm;
    }
    let eps_ppm = ((from.xi_c13() - to.xi_c13()) / to.xi_c13()) * 1.0e6;
    ppm + (1.0 + ppm * 1.0e-6) * eps_ppm
}

/// Convert a `${}^{13}\text{C}$ `Spectrum1D` (`Carbon13Decoupled`, `Dept90`, or `Dept135`) in place
/// from `from` reference standard to `to` reference standard (`F29`), updating `first_ppm`, `step_ppm`,
/// and `obs_mhz` consistently and returning the chemical-shift offset $\Delta\delta$ applied at the
/// spectrum midpoint.
pub fn convert_c13_spectrum_reference_1d(
    spec: &mut Spectrum1D,
    from: ReferenceStandard,
    to: ReferenceStandard,
) -> Result<f64, String> {
    if spec.experiment.nucleus() != Nucleus1D::C13 {
        return Err(format!(
            "13C DSS/TMS reference conversion requires a 13C spectrum, got nucleus '{}'",
            spec.experiment.nucleus().as_str()
        ));
    }
    if from == to {
        return Ok(0.0);
    }
    let sf_dss_over_tms = 1.0 + C13_DSS_VS_TMS_OFFSET_PPM * 1.0e-6;
    let rho = match (from, to) {
        (ReferenceStandard::Dss, ReferenceStandard::Tms) => sf_dss_over_tms,
        (ReferenceStandard::Tms, ReferenceStandard::Dss) => 1.0 / sf_dss_over_tms,
        _ => 1.0,
    };
    let mid_before = 0.5 * (spec.axis.first_ppm + spec.axis.last_ppm());
    spec.axis.first_ppm = convert_c13_ppm_reference(spec.axis.first_ppm, from, to);
    spec.axis.step_ppm *= rho;
    spec.axis.obs_mhz /= rho;
    let mid_after = 0.5 * (spec.axis.first_ppm + spec.axis.last_ppm());
    Ok(mid_after - mid_before)
}

// ============================================================================
// 6. Global 1D Peak Picking & Levenberg–Marquardt Pseudo-Voigt Deconvolution
// ============================================================================

/// Configuration for global 1D peak picking (with optional curvature shoulder seeding).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PeakPickConfig1D {
    /// Minimum signal-to-noise ratio (`|height| / sigma_noise`, default `6.0`).
    pub min_snr: f64,
    /// Minimum relative height relative to global `max_abs` (`0.002` = `0.2%`, suppresses `-100 dB`
    /// FFT truncation ripples on ultra-high-SNR / synthetic noise-free spectra).
    pub min_rel_height: f64,
    /// Minimum peak separation in Hz (default `0.4`).
    pub min_separation_hz: f64,
    /// Polarity override (`None` = `spec.experiment.default_polarity()`).
    pub polarity: Option<Polarity1D>,
    /// Seed unresolved multiplet shoulders from Gaussian-smoothed negative second-derivative maxima (`-y''`).
    pub detect_shoulders: bool,
    /// Minimum relative height of a curvature shoulder relative to the local cluster maximum (default `0.18`).
    pub min_shoulder_rel_height: f64,
}

impl Default for PeakPickConfig1D {
    fn default() -> Self {
        Self {
            min_snr: 6.0,
            min_rel_height: 0.005,
            min_separation_hz: 0.4,
            polarity: None,
            detect_shoulders: false,
            min_shoulder_rel_height: 0.18,
        }
    }
}

/// 1D picked peak candidate prior to (or seeding) global lineshape deconvolution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Peak1DPicked {
    pub ppm: f64,
    pub height: f64,
    pub snr: f64,
    pub est_fwhm_ppm: f64,
    pub est_fwhm_hz: f64,
    pub is_shoulder: bool,
}

/// Estimate robust 1D thermal baseline noise $\hat{\sigma}$ from signal-free bins.
pub fn estimate_noise_sigma_1d(spec: &Spectrum1D) -> f64 {
    let n = spec.len();
    if n < 8 {
        return 1e-9;
    }
    let mask = detect_signal_free_mask_cobas(&spec.real, 8, 3.2, 16);
    let base_vals: Vec<f64> = spec
        .real
        .iter()
        .zip(mask.iter())
        .filter_map(|(&v, &b)| b.then_some(v))
        .collect();
    let s_base = robust_mad(&base_vals);
    if s_base > 1e-18 {
        return s_base;
    }
    let diffs: Vec<f64> = spec
        .real
        .windows(2)
        .map(|w| (w[1] - w[0]) / std::f64::consts::SQRT_2)
        .collect();
    robust_mad(&diffs).max(1e-12)
}

/// Compute 1D Morse topological prominence of an extremum at `j` with sign `sign` (`+1.0` for positive
/// maximum, `-1.0` for negative minimum), rejecting 1-bin thermal noise bumps on broad Lorentzian wings.
fn topological_prominence_1d(y: &[f64], j: usize, sign: f64) -> f64 {
    let n = y.len();
    let v0 = sign * y[j];
    let mut min_left = v0;
    let mut hit_higher_left = false;
    for k in (0..j).rev() {
        let vk = sign * y[k];
        if vk > v0 {
            hit_higher_left = true;
            break;
        }
        if vk < min_left {
            min_left = vk;
        }
    }
    let mut min_right = v0;
    let mut hit_higher_right = false;
    for &yk in y.iter().take(n).skip(j + 1) {
        let vk = sign * yk;
        if vk >= v0 {
            hit_higher_right = true;
            break;
        }
        if vk < min_right {
            min_right = vk;
        }
    }
    let key_col = match (hit_higher_left, hit_higher_right) {
        (true, true) => min_left.max(min_right),
        (true, false) => min_left,
        (false, true) => min_right,
        (false, false) => min_left.min(min_right),
    };
    (v0 - key_col).max(0.0)
}

/// Pick global 1D peaks across `spec`, supporting both `AllPositive` and `Biphasic` (`DEPT-135`)
/// spectra as well as optional second-derivative shoulder detection for overlapping multiplets.
pub fn pick_peaks_1d(spec: &Spectrum1D, config: &PeakPickConfig1D) -> Vec<Peak1DPicked> {
    let n = spec.len();
    if n < 5 {
        return Vec::new();
    }
    let sigma = estimate_noise_sigma_1d(spec).max(1e-12);
    let max_abs = spec.real.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let thresh = (config.min_snr.max(1.0) * sigma).max(config.min_rel_height.max(0.0) * max_abs);
    let prom_thresh =
        (0.85 * config.min_snr.max(1.0) * sigma).max(config.min_rel_height.max(0.0) * max_abs);
    let polarity = config
        .polarity
        .unwrap_or_else(|| spec.experiment.default_polarity());
    let dx_ppm = spec.bin_width_ppm().max(1e-9);
    let obs_mhz = spec.axis.obs_mhz.max(1e-6);
    let min_sep_ppm = config.min_separation_hz.max(0.0) / obs_mhz;

    let mut raw: Vec<Peak1DPicked> = Vec::new();

    for j in 1..(n - 1) {
        let y0 = spec.real[j];
        let is_pos_max = y0 >= thresh && y0 >= spec.real[j - 1] && y0 > spec.real[j + 1];
        let is_neg_min = polarity == Polarity1D::Biphasic
            && y0 <= -thresh
            && y0 <= spec.real[j - 1]
            && y0 < spec.real[j + 1];

        if !is_pos_max && !is_neg_min {
            continue;
        }
        let sign = y0.signum();
        if topological_prominence_1d(&spec.real, j, sign) < prom_thresh {
            continue;
        }

        let (yl, yr) = (spec.real[j - 1], spec.real[j + 1]);
        let denom = yl - 2.0 * y0 + yr;
        let sub = if denom.abs() > 1e-18 {
            (0.5 * (yl - yr) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let ppm = spec.axis.index_to_ppm((j as f64) + sub);
        let height = y0 - 0.25 * (yl - yr) * sub;

        // Estimate local FWHM by walking left and right to half height (bounded before valleys).
        let half_h = 0.5 * y0;
        let sign = y0.signum();
        let mut left_bins = 1.0_f64;
        for k in (0..j).rev() {
            if sign * spec.real[k] <= sign * half_h {
                let y_prev = spec.real[k + 1];
                let frac =
                    ((y_prev - half_h) / (y_prev - spec.real[k]).abs().max(1e-18)).clamp(0.0, 1.0);
                left_bins = ((j - k - 1) as f64) + frac;
                break;
            }
            if k + 1 < j && sign * spec.real[k] > sign * spec.real[k + 1] {
                left_bins = (j - k) as f64;
                break;
            }
            left_bins = (j - k) as f64;
            if left_bins > 200.0 {
                break;
            }
        }
        let mut right_bins = 1.0_f64;
        for k in (j + 1)..n {
            if sign * spec.real[k] <= sign * half_h {
                let y_prev = spec.real[k - 1];
                let frac =
                    ((y_prev - half_h) / (y_prev - spec.real[k]).abs().max(1e-18)).clamp(0.0, 1.0);
                right_bins = ((k - j - 1) as f64) + frac;
                break;
            }
            if k > j + 1 && sign * spec.real[k] > sign * spec.real[k - 1] {
                right_bins = (k - j) as f64;
                break;
            }
            right_bins = (k - j) as f64;
            if right_bins > 200.0 {
                break;
            }
        }

        let fwhm_bins = (2.0 * left_bins.min(right_bins)).clamp(1.5, 400.0);
        let fwhm_ppm = fwhm_bins * dx_ppm;
        raw.push(Peak1DPicked {
            ppm,
            height,
            snr: height.abs() / sigma,
            est_fwhm_ppm: fwhm_ppm,
            est_fwhm_hz: fwhm_ppm * obs_mhz,
            is_shoulder: false,
        });
    }

    // Optional linewidth-smoothed Ricker curvature (`-y''`) shoulder seeding for coalesced multiplets.
    if config.detect_shoulders && n >= 17 {
        let neg_d2 = ricker_wavelet_response(&spec.real, 3);
        let d2_sigma = robust_mad(&neg_d2).max(1e-15);
        for j in 4..(n - 4) {
            let y0 = spec.real[j];
            if y0 < thresh {
                continue;
            }
            // Must lie on a monotonic flank (not a flat baseline tail).
            let dy_l = spec.real[j] - spec.real[j - 1];
            let dy_r = spec.real[j + 1] - spec.real[j];
            if dy_l * dy_r <= 0.0 {
                continue;
            }
            let c0 = neg_d2[j];
            if c0 > 8.0 * d2_sigma && c0 >= neg_d2[j - 1] && c0 > neg_d2[j + 1] {
                let ppm = spec.ppm_at(j);
                let near_existing = raw
                    .iter()
                    .any(|p| (p.ppm - ppm).abs() <= min_sep_ppm.max(3.0 * dx_ppm));
                if near_existing {
                    continue;
                }
                let local_max = raw
                    .iter()
                    .filter(|p| (p.ppm - ppm).abs() <= 0.15)
                    .map(|p| p.height.abs())
                    .fold(max_abs, f64::max);
                if y0.abs() >= config.min_shoulder_rel_height * local_max {
                    let med_fwhm = raw
                        .iter()
                        .filter(|p| (p.ppm - ppm).abs() <= 0.15)
                        .map(|p| p.est_fwhm_ppm)
                        .next()
                        .unwrap_or(4.0 * dx_ppm);
                    raw.push(Peak1DPicked {
                        ppm,
                        height: y0 * 0.75,
                        snr: y0.abs() / sigma,
                        est_fwhm_ppm: med_fwhm,
                        est_fwhm_hz: med_fwhm * obs_mhz,
                        is_shoulder: true,
                    });
                }
            }
        }
    }

    // Deduplicate peaks closer than `min_sep_ppm` keeping stronger amplitude first.
    raw.sort_by(|a, b| b.height.abs().total_cmp(&a.height.abs()));
    let mut kept: Vec<Peak1DPicked> = Vec::with_capacity(raw.len());
    for cand in raw {
        if kept.iter().all(|p| (p.ppm - cand.ppm).abs() >= min_sep_ppm) {
            kept.push(cand);
        }
    }
    // Return ordered high-to-low ppm (standard NMR tabular order).
    kept.sort_by(|a, b| b.ppm.total_cmp(&a.ppm));
    kept
}

/// Fitted pseudo-Voigt resonance component returned by Levenberg–Marquardt deconvolution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoigtPeak1D {
    /// Fitted resonance center $\mu_k$ in ppm.
    pub center_ppm: f64,
    /// Fitted full width at half maximum $\gamma_k$ in ppm.
    pub fwhm_ppm: f64,
    /// Fitted full width at half maximum in Hz ($\gamma_k \cdot \nu_{\text{obs}}$).
    pub fwhm_hz: f64,
    /// Lorentzian mixing fraction $\eta_k \in [0, 1]$ (`1.0` = pure Lorentzian, `0.0` = pure Gaussian).
    pub lorentz_fraction: f64,
    /// Gaussian mixing fraction $1 - \eta_k \in [0, 1]$.
    pub gauss_fraction: f64,
    /// Exact analytical integrated area $A_k$ in $\text{intensity} \times \text{ppm}$.
    pub area_ppm: f64,
    /// Exact analytical integrated area in $\text{intensity} \times \text{Hz}$ ($A_k \cdot \nu_{\text{obs}}$).
    pub area_hz: f64,
    /// Evaluated component apex height $V_k(\mu_k)$.
    pub height: f64,
}

/// Configuration for coupled-cluster Levenberg–Marquardt pseudo-Voigt deconvolution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeconvolveConfig1D {
    pub pick_config: PeakPickConfig1D,
    /// Maximum LM iterations per coupled multiplet cluster (default `120`).
    pub max_lm_iters: usize,
    /// Initial Lorentzian fraction $\eta^{(0)} \in [0, 1]$ (default `0.85`).
    pub init_lorentz_fraction: f64,
    /// Merge clusters whose peak centers are separated by less than `cluster_merge_hz` (default `80.0`).
    pub cluster_merge_hz: f64,
    /// Padding around each cluster's outermost peaks in multiples of initial FWHM (default `12.0`).
    pub window_pad_fwhm_mult: f64,
    /// Maximum number of residual-peak insertion passes inside a cluster when residual exceeds
    /// `residual_insert_snr * sigma_noise` (default `2`).
    pub max_residual_insertions: usize,
    pub residual_insert_snr: f64,
}

impl Default for DeconvolveConfig1D {
    fn default() -> Self {
        Self {
            pick_config: PeakPickConfig1D::default(),
            max_lm_iters: 120,
            init_lorentz_fraction: 0.85,
            cluster_merge_hz: 80.0,
            window_pad_fwhm_mult: 12.0,
            max_residual_insertions: 2,
            residual_insert_snr: 5.5,
        }
    }
}

/// Per-cluster deconvolution fit diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeconvolutionClusterResult {
    pub ppm_min: f64,
    pub ppm_max: f64,
    pub peaks: Vec<VoigtPeak1D>,
    pub residual_rmsd: f64,
    pub iterations: usize,
}

/// Global 1D deconvolution result across a region or full spectrum.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeconvolutionResult1D {
    /// Fitted pseudo-Voigt components sorted high-to-low ppm.
    pub peaks: Vec<VoigtPeak1D>,
    /// Reconstructed synthetic sum trace across `0..spec.len()`.
    pub fitted_spectrum: Vec<f64>,
    /// Residual trace `spec.real - fitted_spectrum`.
    pub residual_spectrum: Vec<f64>,
    /// Root-mean-square residual across all fitted cluster windows.
    pub residual_rmsd: f64,
    /// Individual coupled-cluster fits.
    pub clusters: Vec<DeconvolutionClusterResult>,
}

const C_LORENTZ: f64 = 2.0 / PI;
const ALPHA_GAUSS: f64 = 4.0 * std::f64::consts::LN_2;

#[inline]
fn c_gauss_norm() -> f64 {
    2.0 * (std::f64::consts::LN_2 / PI).sqrt()
}

/// Evaluate a single area-normalized pseudo-Voigt component $V(x; \mu, \gamma, \eta, A)$ at `x_ppm`.
#[inline]
pub fn eval_pseudo_voigt_1d(
    x_ppm: f64,
    center_ppm: f64,
    fwhm_ppm: f64,
    lorentz_frac: f64,
    area_ppm: f64,
) -> f64 {
    let g = fwhm_ppm.max(1e-12);
    let eta = lorentz_frac.clamp(0.0, 1.0);
    let u = (x_ppm - center_ppm) / g;
    let lor = C_LORENTZ / (g * (1.0 + 4.0 * u * u));
    let gau = (c_gauss_norm() / g) * (-ALPHA_GAUSS * u * u).exp();
    area_ppm * (eta * lor + (1.0 - eta) * gau)
}

/// Evaluate the sum of `peaks` at `x_ppm`.
pub fn eval_pseudo_voigt_sum_1d(x_ppm: f64, peaks: &[VoigtPeak1D]) -> f64 {
    peaks
        .iter()
        .map(|p| {
            eval_pseudo_voigt_1d(
                x_ppm,
                p.center_ppm,
                p.fwhm_ppm,
                p.lorentz_fraction,
                p.area_ppm,
            )
        })
        .sum()
}

/// Evaluate $V(x; \mu, \gamma, \eta, A)$ and its exact analytical partial derivatives
/// `[dV/d_mu, dV/d_gamma, dV/d_eta, dV/d_A]`.
#[inline]
fn eval_pseudo_voigt_with_grad(
    x_ppm: f64,
    mu: f64,
    gamma: f64,
    eta: f64,
    area: f64,
) -> (f64, [f64; 4]) {
    let g = gamma.max(1e-12);
    let inv_g = 1.0 / g;
    let u = (x_ppm - mu) * inv_g;
    let u2 = u * u;
    let inv_lor_den = 1.0 / (1.0 + 4.0 * u2);
    let lor = C_LORENTZ * inv_g * inv_lor_den;
    let gau = c_gauss_norm() * inv_g * (-ALPHA_GAUSS * u2).exp();

    let unit_prof = eta * lor + (1.0 - eta) * gau;
    let val = area * unit_prof;

    let dlor_dmu = (8.0 * u * inv_g * inv_lor_den) * lor;
    let dgau_dmu = (2.0 * ALPHA_GAUSS * u * inv_g) * gau;
    let dv_dmu = area * (eta * dlor_dmu + (1.0 - eta) * dgau_dmu);

    let dlor_dg = ((4.0 * u2 - 1.0) * inv_g * inv_lor_den) * lor;
    let dgau_dg = ((2.0 * ALPHA_GAUSS * u2 - 1.0) * inv_g) * gau;
    let dv_dg = area * (eta * dlor_dg + (1.0 - eta) * dgau_dg);

    let dv_deta = area * (lor - gau);
    let dv_da = unit_prof;

    (val, [dv_dmu, dv_dg, dv_deta, dv_da])
}

/// Fit `K` overlapping pseudo-Voigt components simultaneously on `(xs, ys)` via linear least-squares
/// amplitude initialization + projected analytical-Jacobian Levenberg–Marquardt.
#[allow(clippy::too_many_arguments)]
fn fit_voigt_cluster_lm(
    xs: &[f64],
    ys: &[f64],
    obs_mhz: f64,
    dx_ppm: f64,
    polarity: Polarity1D,
    seeds: &[(f64, f64, f64)], // (center_ppm, fwhm_ppm, approx_height)
    init_eta: f64,
    max_iters: usize,
) -> (Vec<VoigtPeak1D>, f64, usize) {
    let m_pts = xs.len();
    let k = seeds.len();
    if m_pts == 0 || k == 0 {
        return (Vec::new(), 0.0, 0);
    }

    let x_lo = xs.iter().copied().fold(f64::INFINITY, f64::min) - 2.0 * dx_ppm;
    let x_hi = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 2.0 * dx_ppm;
    let g_min = (0.4 * dx_ppm).max(0.05 / obs_mhz.max(1.0));
    let g_max = ((x_hi - x_lo) * 0.8).max(10.0 * dx_ppm);
    let eta0 = init_eta.clamp(0.0, 1.0);

    // Step 1: Solve linear least-squares initial areas `A_1..A_K` given `(mu_0, gamma_0, eta_0)`
    // so strongly overlapping shoulders start with unbiased decoupled areas!
    let mut b_cols = vec![vec![0.0_f64; m_pts]; k];
    for (c_idx, &(mu, g, _)) in seeds.iter().enumerate() {
        let gc = g.clamp(g_min, g_max);
        for (i, &x) in xs.iter().enumerate() {
            b_cols[c_idx][i] = eval_pseudo_voigt_1d(x, mu, gc, eta0, 1.0);
        }
    }
    let mut btb = vec![vec![0.0_f64; k]; k];
    let mut bty = vec![0.0_f64; k];
    for r in 0..k {
        for i in 0..m_pts {
            bty[r] += b_cols[r][i] * ys[i];
        }
        for c in r..k {
            let s: f64 = b_cols[r]
                .iter()
                .zip(b_cols[c].iter())
                .take(m_pts)
                .map(|(&br, &bc)| br * bc)
                .sum();
            btb[r][c] = s;
            btb[c][r] = s;
        }
        btb[r][r] += 1e-10;
    }
    let init_areas = solve_real_linear_system(&mut btb, &mut bty).unwrap_or_else(|| {
        seeds
            .iter()
            .map(|&(_, g, h)| {
                let unit0 = eval_pseudo_voigt_1d(0.0, 0.0, g.clamp(g_min, g_max), eta0, 1.0);
                h / unit0.max(1e-12)
            })
            .collect()
    });

    // Parameter vector `theta` of length `4 * K`: `[mu_c, gamma_c, eta_c, area_c]`.
    let mut theta = vec![0.0_f64; 4 * k];
    for c_idx in 0..k {
        let (mu, g, h) = seeds[c_idx];
        theta[4 * c_idx] = mu.clamp(x_lo, x_hi);
        theta[4 * c_idx + 1] = g.clamp(g_min, g_max);
        theta[4 * c_idx + 2] = eta0;
        let a_raw = init_areas[c_idx];
        theta[4 * c_idx + 3] = if polarity == Polarity1D::AllPositive {
            if a_raw > 0.0 {
                a_raw
            } else {
                let unit0 = eval_pseudo_voigt_1d(0.0, 0.0, g.clamp(g_min, g_max), eta0, 1.0);
                (0.5 * h.max(0.0)) / unit0.max(1e-12)
            }
        } else {
            a_raw
        };
    }

    let project_theta = |th: &mut [f64]| {
        for c_idx in 0..k {
            th[4 * c_idx] = th[4 * c_idx].clamp(x_lo, x_hi);
            th[4 * c_idx + 1] = th[4 * c_idx + 1].clamp(g_min, g_max);
            th[4 * c_idx + 2] = th[4 * c_idx + 2].clamp(0.0, 1.0);
            if polarity == Polarity1D::AllPositive && th[4 * c_idx + 3] < 0.0 {
                th[4 * c_idx + 3] = 0.0;
            }
        }
    };
    project_theta(&mut theta);

    // Compute fixed scale-invariant intra-cluster lineshape coherence weights `(w_gamma, w_eta)`
    // from initial diagonal sensitivities (`k >= 2`) so overlapping Voigt/pseudo-Voigt wings do
    // not trade Lorentzian tail fraction for neighbor area while heterogeneous peaks remain free.
    let (w_gamma, w_eta) = if k >= 2 {
        let mut sum_gg = 0.0_f64;
        let mut sum_ee = 0.0_f64;
        for c_idx in 0..k {
            for &x in xs {
                let (_, g4) = eval_pseudo_voigt_with_grad(
                    x,
                    theta[4 * c_idx],
                    theta[4 * c_idx + 1],
                    theta[4 * c_idx + 2],
                    theta[4 * c_idx + 3],
                );
                sum_gg += g4[1] * g4[1];
                sum_ee += g4[2] * g4[2];
            }
        }
        let kf = k as f64;
        (0.008 * (sum_gg / kf), 0.022 * (sum_ee / kf))
    } else {
        (0.0, 0.0)
    };

    let eval_data_rss = |th: &[f64], res_out: &mut [f64]| -> f64 {
        let mut rss = 0.0_f64;
        for (i, (&x, &y)) in xs.iter().zip(ys.iter()).enumerate() {
            let mut yhat = 0.0_f64;
            for c_idx in 0..k {
                yhat += eval_pseudo_voigt_1d(
                    x,
                    th[4 * c_idx],
                    th[4 * c_idx + 1],
                    th[4 * c_idx + 2],
                    th[4 * c_idx + 3],
                );
            }
            let r = y - yhat;
            res_out[i] = r;
            rss += r * r;
        }
        rss
    };

    let eval_penalized_obj = |th: &[f64], res_out: &mut [f64]| -> f64 {
        let mut obj = eval_data_rss(th, res_out);
        if k >= 2 {
            let kf = k as f64;
            let mean_g: f64 = (0..k).map(|c| th[4 * c + 1]).sum::<f64>() / kf;
            let mean_e: f64 = (0..k).map(|c| th[4 * c + 2]).sum::<f64>() / kf;
            for c in 0..k {
                let dg = th[4 * c + 1] - mean_g;
                let de = th[4 * c + 2] - mean_e;
                obj += w_gamma * dg * dg + w_eta * de * de;
            }
        }
        obj
    };

    let n_par = 4 * k;
    let mut res = vec![0.0_f64; m_pts];
    let mut trial_res = vec![0.0_f64; m_pts];
    let mut cur_rss = eval_penalized_obj(&theta, &mut res);
    let mut lam = 1e-3_f64;
    let mut iters_used = 0usize;

    let mut jtj = vec![vec![0.0_f64; n_par]; n_par];
    let mut jtr = vec![0.0_f64; n_par];
    let mut row_j = vec![0.0_f64; n_par];

    for iter in 0..max_iters {
        iters_used = iter + 1;
        for r in 0..n_par {
            jtr[r] = 0.0;
            jtj[r].fill(0.0);
        }

        for (i, &x) in xs.iter().enumerate() {
            let ri = res[i];
            for c_idx in 0..k {
                let (_, g4) = eval_pseudo_voigt_with_grad(
                    x,
                    theta[4 * c_idx],
                    theta[4 * c_idx + 1],
                    theta[4 * c_idx + 2],
                    theta[4 * c_idx + 3],
                );
                row_j[4 * c_idx] = g4[0];
                row_j[4 * c_idx + 1] = g4[1];
                row_j[4 * c_idx + 2] = g4[2];
                row_j[4 * c_idx + 3] = g4[3];
            }
            for a in 0..n_par {
                let ja = row_j[a];
                jtr[a] += ja * ri;
                for b in a..n_par {
                    jtj[a][b] += ja * row_j[b];
                }
            }
        }
        for a in 1..n_par {
            let (up, dn) = jtj.split_at_mut(a);
            for (b, urow) in up.iter().enumerate() {
                dn[0][b] = urow[a];
            }
        }
        if k >= 2 {
            let kf = k as f64;
            let mean_g: f64 = (0..k).map(|c| theta[4 * c + 1]).sum::<f64>() / kf;
            let mean_e: f64 = (0..k).map(|c| theta[4 * c + 2]).sum::<f64>() / kf;
            for a_c in 0..k {
                let ig_a = 4 * a_c + 1;
                let ie_a = 4 * a_c + 2;
                jtr[ig_a] -= w_gamma * (theta[ig_a] - mean_g);
                jtr[ie_a] -= w_eta * (theta[ie_a] - mean_e);
                for b_c in 0..k {
                    let proj = if a_c == b_c {
                        1.0 - 1.0 / kf
                    } else {
                        -1.0 / kf
                    };
                    jtj[ig_a][4 * b_c + 1] += w_gamma * proj;
                    jtj[ie_a][4 * b_c + 2] += w_eta * proj;
                }
            }
        }

        // Active-set projection on active box boundaries where gradient points outward.
        let mut active_locked = vec![false; n_par];
        for c_idx in 0..k {
            let i_mu = 4 * c_idx;
            let i_g = 4 * c_idx + 1;
            let i_eta = 4 * c_idx + 2;
            let i_a = 4 * c_idx + 3;
            if (theta[i_mu] <= x_lo + 1e-15 && jtr[i_mu] < 0.0)
                || (theta[i_mu] >= x_hi - 1e-15 && jtr[i_mu] > 0.0)
            {
                active_locked[i_mu] = true;
            }
            if (theta[i_g] <= g_min + 1e-15 && jtr[i_g] < 0.0)
                || (theta[i_g] >= g_max - 1e-15 && jtr[i_g] > 0.0)
            {
                active_locked[i_g] = true;
            }
            if (theta[i_eta] <= 1e-15 && jtr[i_eta] < 0.0)
                || (theta[i_eta] >= 1.0 - 1e-15 && jtr[i_eta] > 0.0)
            {
                active_locked[i_eta] = true;
            }
            if polarity == Polarity1D::AllPositive && theta[i_a] <= 1e-15 && jtr[i_a] < 0.0 {
                active_locked[i_a] = true;
            }
        }

        let mut accepted = false;
        for _sub in 0..8 {
            let mut h_mat = jtj.clone();
            let mut g_vec = jtr.clone();
            for (p, &locked) in active_locked.iter().enumerate().take(n_par) {
                if locked {
                    h_mat[p][..n_par].fill(0.0);
                    for row_h in h_mat.iter_mut().take(n_par) {
                        row_h[p] = 0.0;
                    }
                    h_mat[p][p] = 1.0;
                    g_vec[p] = 0.0;
                } else {
                    let d_scale = h_mat[p][p].abs().max(1e-8);
                    h_mat[p][p] += lam * d_scale + 1e-12;
                }
            }

            let Some(step) = solve_real_linear_system(&mut h_mat, &mut g_vec) else {
                lam *= 10.0;
                continue;
            };

            let mut cand = theta.clone();
            for p in 0..n_par {
                cand[p] += step[p];
            }
            project_theta(&mut cand);

            let trial_rss = eval_penalized_obj(&cand, &mut trial_res);
            if trial_rss < cur_rss {
                let rel_impr = (cur_rss - trial_rss) / cur_rss.max(1e-30);
                theta = cand;
                cur_rss = trial_rss;
                std::mem::swap(&mut res, &mut trial_res);
                lam = (lam * 0.33).max(1e-11);
                accepted = true;
                if rel_impr < 1e-10 {
                    iters_used = iter + 1;
                    break;
                }
                break;
            } else {
                lam *= 5.0;
                if lam > 1e12 {
                    break;
                }
            }
        }

        if !accepted && lam > 1e12 {
            break;
        }
        if cur_rss <= 1e-30 {
            break;
        }
    }

    let final_data_rss = eval_data_rss(&theta, &mut res);
    let rmsd = (final_data_rss / (m_pts as f64)).sqrt();
    let mut peaks = Vec::with_capacity(k);
    for c_idx in 0..k {
        let mu = theta[4 * c_idx];
        let g = theta[4 * c_idx + 1];
        let eta = theta[4 * c_idx + 2];
        let area = theta[4 * c_idx + 3];
        let height = eval_pseudo_voigt_1d(mu, mu, g, eta, area);
        peaks.push(VoigtPeak1D {
            center_ppm: mu,
            fwhm_ppm: g,
            fwhm_hz: g * obs_mhz,
            lorentz_fraction: eta,
            gauss_fraction: 1.0 - eta,
            area_ppm: area,
            area_hz: area * obs_mhz,
            height,
        });
    }
    peaks.sort_by(|a, b| b.center_ppm.total_cmp(&a.center_ppm));
    (peaks, rmsd, iters_used)
}

/// Deconvolve all resonances inside `[ppm_min, ppm_max]` jointly as a single coupled multiplet
/// cluster via projected Levenberg–Marquardt pseudo-Voigt optimization.
/// If `explicit_seeds` is `Some(...)`, uses those peaks directly; otherwise picks peaks (+ shoulders)
/// inside `[ppm_min, ppm_max]` and performs optional residual-peak insertion.
pub fn deconvolve_region_1d(
    spec: &Spectrum1D,
    ppm_min: f64,
    ppm_max: f64,
    explicit_seeds: Option<&[Peak1DPicked]>,
    config: &DeconvolveConfig1D,
) -> DeconvolutionClusterResult {
    let (lo, hi) = (ppm_min.min(ppm_max), ppm_min.max(ppm_max));
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (j, &y) in spec.real.iter().enumerate() {
        let x = spec.ppm_at(j);
        if x >= lo && x <= hi {
            xs.push(x);
            ys.push(y);
        }
    }
    if xs.is_empty() {
        return DeconvolutionClusterResult {
            ppm_min: lo,
            ppm_max: hi,
            peaks: Vec::new(),
            residual_rmsd: 0.0,
            iterations: 0,
        };
    }

    let polarity = config
        .pick_config
        .polarity
        .unwrap_or_else(|| spec.experiment.default_polarity());
    let obs_mhz = spec.axis.obs_mhz.max(1e-6);
    let dx_ppm = spec.bin_width_ppm().max(1e-9);
    let sigma = estimate_noise_sigma_1d(spec).max(1e-12);

    let initial_picks: Vec<Peak1DPicked> = if let Some(seeds) = explicit_seeds {
        seeds.to_vec()
    } else {
        pick_peaks_1d(spec, &config.pick_config)
            .into_iter()
            .filter(|p| p.ppm >= lo && p.ppm <= hi)
            .collect()
    };

    if initial_picks.is_empty() {
        let rss: f64 = ys.iter().map(|v| v * v).sum();
        return DeconvolutionClusterResult {
            ppm_min: lo,
            ppm_max: hi,
            peaks: Vec::new(),
            residual_rmsd: (rss / (ys.len() as f64)).sqrt(),
            iterations: 0,
        };
    }

    let mut seeds: Vec<(f64, f64, f64)> = initial_picks
        .iter()
        .map(|p| (p.ppm, p.est_fwhm_ppm, p.height))
        .collect();

    let (mut peaks, mut rmsd, mut total_iters) = fit_voigt_cluster_lm(
        &xs,
        &ys,
        obs_mhz,
        dx_ppm,
        polarity,
        &seeds,
        config.init_lorentz_fraction,
        config.max_lm_iters,
    );

    // Optional residual-peak insertion for completely unresolved hidden multiplet components.
    if config.max_residual_insertions > 0 {
        let cluster_max = ys.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let res_thresh = (config.residual_insert_snr * sigma).max(0.03 * cluster_max);
        let min_sep_ppm =
            (config.pick_config.min_separation_hz.max(0.2) / obs_mhz).max(2.0 * dx_ppm);
        for _ins in 0..config.max_residual_insertions {
            let mut best_idx: Option<usize> = None;
            let mut best_res = 0.0_f64;
            for i in 1..xs.len().saturating_sub(1) {
                if peaks
                    .iter()
                    .any(|p| (p.center_ppm - xs[i]).abs() < min_sep_ppm)
                {
                    continue;
                }
                let r_i = ys[i] - eval_pseudo_voigt_sum_1d(xs[i], &peaks);
                let r_l = ys[i - 1] - eval_pseudo_voigt_sum_1d(xs[i - 1], &peaks);
                let r_r = ys[i + 1] - eval_pseudo_voigt_sum_1d(xs[i + 1], &peaks);
                let is_ext = (r_i >= res_thresh && r_i >= r_l && r_i > r_r)
                    || (polarity == Polarity1D::Biphasic
                        && r_i <= -res_thresh
                        && r_i <= r_l
                        && r_i < r_r);
                if is_ext && r_i.abs() > best_res.abs() {
                    best_res = r_i;
                    best_idx = Some(i);
                }
            }
            let Some(bi) = best_idx else {
                break;
            };
            let med_g = peaks.first().map(|p| p.fwhm_ppm).unwrap_or(3.0 * dx_ppm);
            seeds = peaks
                .iter()
                .map(|p| (p.center_ppm, p.fwhm_ppm, p.height))
                .collect();
            seeds.push((xs[bi], med_g, best_res));
            let (next_peaks, next_rmsd, it2) = fit_voigt_cluster_lm(
                &xs,
                &ys,
                obs_mhz,
                dx_ppm,
                polarity,
                &seeds,
                config.init_lorentz_fraction,
                config.max_lm_iters,
            );
            total_iters += it2;
            if next_rmsd < 0.92 * rmsd {
                peaks = next_peaks;
                rmsd = next_rmsd;
            } else {
                break;
            }
        }
    }

    DeconvolutionClusterResult {
        ppm_min: lo,
        ppm_max: hi,
        peaks,
        residual_rmsd: rmsd,
        iterations: total_iters,
    }
}

/// Perform global 1D peak picking and coupled-cluster Levenberg–Marquardt pseudo-Voigt
/// deconvolution across the entire `Spectrum1D`.
pub fn deconvolve_spectrum_1d(
    spec: &Spectrum1D,
    config: &DeconvolveConfig1D,
) -> DeconvolutionResult1D {
    let n = spec.len();
    if n == 0 {
        return DeconvolutionResult1D {
            peaks: Vec::new(),
            fitted_spectrum: Vec::new(),
            residual_spectrum: Vec::new(),
            residual_rmsd: 0.0,
            clusters: Vec::new(),
        };
    }

    let picks = pick_peaks_1d(spec, &config.pick_config);
    if picks.is_empty() {
        let rss: f64 = spec.real.iter().map(|v| v * v).sum();
        return DeconvolutionResult1D {
            peaks: Vec::new(),
            fitted_spectrum: vec![0.0; n],
            residual_spectrum: spec.real.clone(),
            residual_rmsd: (rss / (n as f64)).sqrt(),
            clusters: Vec::new(),
        };
    }

    let obs_mhz = spec.axis.obs_mhz.max(1e-6);
    let merge_gap_ppm = config.cluster_merge_hz.max(0.0) / obs_mhz;
    let pad_mult = config.window_pad_fwhm_mult.max(4.0);
    let ax_min = spec.axis.min_ppm();
    let ax_max = spec.axis.max_ppm();

    // `picks` are ordered descending by `ppm`. Group adjacent picks into coupled clusters.
    let mut groups: Vec<(f64, f64, Vec<Peak1DPicked>)> = Vec::new();
    for p in picks {
        let pad = (pad_mult * p.est_fwhm_ppm).max(0.03);
        let p_lo = (p.ppm - pad).max(ax_min);
        let p_hi = (p.ppm + pad).min(ax_max);
        if let Some(last) = groups.last_mut() {
            let last_min_center = last.2.last().map(|q| q.ppm).unwrap_or(last.0);
            if p_hi >= last.0 || (last_min_center - p.ppm) <= merge_gap_ppm {
                last.0 = last.0.min(p_lo);
                last.1 = last.1.max(p_hi);
                last.2.push(p);
                continue;
            }
        }
        groups.push((p_lo, p_hi, vec![p]));
    }

    let mut all_peaks = Vec::new();
    let mut clusters = Vec::with_capacity(groups.len());
    let mut rss_win = 0.0_f64;
    let mut win_pts = 0usize;

    for (lo, hi, group_seeds) in groups {
        let cres = deconvolve_region_1d(spec, lo, hi, Some(&group_seeds), config);
        let n_in = spec
            .real
            .iter()
            .enumerate()
            .filter(|&(j, _)| {
                let x = spec.ppm_at(j);
                x >= cres.ppm_min && x <= cres.ppm_max
            })
            .count();
        rss_win += cres.residual_rmsd * cres.residual_rmsd * (n_in as f64);
        win_pts += n_in;
        all_peaks.extend_from_slice(&cres.peaks);
        clusters.push(cres);
    }

    all_peaks.sort_by(|a, b| b.center_ppm.total_cmp(&a.center_ppm));

    // Evaluate full reconstructed model spectrum (evaluating each component within 40 FWHMs, or all if narrow).
    let mut fitted_spectrum = vec![0.0_f64; n];
    for pk in &all_peaks {
        let reach = (60.0 * pk.fwhm_ppm).max(0.5);
        let i0 = spec
            .axis
            .ppm_to_index(pk.center_ppm + reach)
            .floor()
            .max(0.0) as usize;
        let i1 = (spec.axis.ppm_to_index(pk.center_ppm - reach).ceil() as usize).min(n - 1);
        let (s, e) = (i0.min(i1), i0.max(i1));
        for (j, dst) in fitted_spectrum.iter_mut().enumerate().take(e + 1).skip(s) {
            *dst += eval_pseudo_voigt_1d(
                spec.ppm_at(j),
                pk.center_ppm,
                pk.fwhm_ppm,
                pk.lorentz_fraction,
                pk.area_ppm,
            );
        }
    }

    let residual_spectrum: Vec<f64> = spec
        .real
        .iter()
        .zip(fitted_spectrum.iter())
        .map(|(&y, &yh)| y - yh)
        .collect();
    let residual_rmsd = if win_pts > 0 {
        (rss_win / (win_pts as f64)).sqrt()
    } else {
        0.0
    };

    DeconvolutionResult1D {
        peaks: all_peaks,
        fitted_spectrum,
        residual_spectrum,
        residual_rmsd,
        clusters,
    }
}

// ============================================================================
// 7. Validated 1D Ingest (`Bruker pdata 1r/procs`, `Bruker fid`, `JCAMP-DX`, `JSON`, `TSV`; `F15`)
// ============================================================================

/// Maximum permitted 1D trace length (`4,194,304` points = `32 MiB` `f64`) enforced **prior** to allocation.
pub const MAX_1D_INGEST_POINTS: usize = 4_194_304;

/// Maximum permitted 1D input file size (`64 MiB`) checked via `fs::metadata` prior to buffer allocation.
pub const MAX_1D_INGEST_BYTES: u64 = 64 * 1024 * 1024;

/// Parsed 1D input payload (either complex time-domain FID or processed frequency-domain spectrum).
#[derive(Clone, Debug)]
pub enum Loaded1DData {
    TimeDomain(TimeDomain1D),
    FrequencyDomain(Spectrum1D),
}

impl TimeDomain1D {
    /// Validate finite acquisition parameters, sample bounds (`2..=MAX_1D_INGEST_POINTS`), and finite complex FID values.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.data.len();
        if n < 2 {
            return Err(format!(
                "TimeDomain1D {:?} contains {n} complex samples (minimum 2 required)",
                self.name
            ));
        }
        if n > MAX_1D_INGEST_POINTS {
            return Err(format!(
                "TimeDomain1D {:?} length {n} exceeds MAX_1D_INGEST_POINTS ({MAX_1D_INGEST_POINTS})",
                self.name
            ));
        }
        if self.active_td < 2 || self.active_td > n {
            return Err(format!(
                "TimeDomain1D {:?} active_td={} out of valid range 2..={n}",
                self.name, self.active_td
            ));
        }
        if !self.sw_hz.is_finite() || self.sw_hz <= 0.0 {
            return Err(format!(
                "TimeDomain1D {:?} has invalid sw_hz={}",
                self.name, self.sw_hz
            ));
        }
        if !self.obs_mhz.is_finite() || self.obs_mhz <= 0.0 {
            return Err(format!(
                "TimeDomain1D {:?} has invalid obs_mhz={}",
                self.name, self.obs_mhz
            ));
        }
        if !self.carrier_ppm.is_finite() || !self.group_delay_pts.is_finite() {
            return Err(format!(
                "TimeDomain1D {:?} has non-finite carrier_ppm or group_delay_pts",
                self.name
            ));
        }
        for (idx, &[re, im]) in self.data.iter().enumerate() {
            if !re.is_finite() || !im.is_finite() {
                return Err(format!(
                    "TimeDomain1D {:?} contains non-finite sample at index {idx}: ({re}, {im})",
                    self.name
                ));
            }
        }
        Ok(())
    }
}

impl Spectrum1D {
    /// Validate finite axis metadata, `axis.size == real.len()`, optional `imag` length, and finite spectral intensities.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.real.len();
        if n < 2 {
            return Err(format!(
                "Spectrum1D {:?} contains {n} points (minimum 2 required)",
                self.name
            ));
        }
        if n > MAX_1D_INGEST_POINTS {
            return Err(format!(
                "Spectrum1D {:?} length {n} exceeds MAX_1D_INGEST_POINTS ({MAX_1D_INGEST_POINTS})",
                self.name
            ));
        }
        if self.axis.size != n {
            return Err(format!(
                "Spectrum1D {:?} axis.size ({}) disagrees with real.len() ({n})",
                self.name, self.axis.size
            ));
        }
        if !self.axis.obs_mhz.is_finite()
            || self.axis.obs_mhz <= 0.0
            || !self.axis.first_ppm.is_finite()
            || !self.axis.step_ppm.is_finite()
            || self.axis.step_ppm.abs() == 0.0
        {
            return Err(format!(
                "Spectrum1D {:?} has invalid axis metadata (obs_mhz={}, first_ppm={}, step_ppm={})",
                self.name, self.axis.obs_mhz, self.axis.first_ppm, self.axis.step_ppm
            ));
        }
        if !self.phase_applied.0.is_finite()
            || !self.phase_applied.1.is_finite()
            || !self.phase_applied.2.is_finite()
        {
            return Err(format!(
                "Spectrum1D {:?} has non-finite phase_applied metadata",
                self.name
            ));
        }
        for (idx, &v) in self.real.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!(
                    "Spectrum1D {:?} contains non-finite real intensity at index {idx}: {v}",
                    self.name
                ));
            }
        }
        if let Some(ref im) = self.imag {
            if im.len() != n {
                return Err(format!(
                    "Spectrum1D {:?} imag.len() ({}) disagrees with real.len() ({n})",
                    self.name,
                    im.len()
                ));
            }
            for (idx, &v) in im.iter().enumerate() {
                if !v.is_finite() {
                    return Err(format!(
                        "Spectrum1D {:?} contains non-finite imag intensity at index {idx}: {v}",
                        self.name
                    ));
                }
            }
        }
        Ok(())
    }
}

fn find_jcamp_param(txt: &str, key_norm: &str) -> Option<String> {
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("##") {
            let rest = rest.strip_prefix('$').unwrap_or(rest);
            if let Some((k, v)) = rest.split_once('=') {
                let norm: String = k
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric())
                    .map(|c| c.to_ascii_uppercase())
                    .collect();
                if norm == key_norm {
                    return Some(v.trim().to_string());
                }
            }
        }
    }
    None
}

fn decode_bruker_1d_channel_bytes(
    label: &str,
    raw_bytes: &[u8],
    si: usize,
    word_size: usize,
    dtype: u8,
    big_endian: bool,
    scale: f64,
) -> Result<Vec<f64>, String> {
    let expected_bytes = (si as u64)
        .checked_mul(word_size as u64)
        .ok_or_else(|| format!("Bruker SI={si} byte count overflow"))?;
    let file_len = raw_bytes.len() as u64;
    if file_len < expected_bytes {
        return Err(format!(
            "truncated Bruker {label}: has {file_len} bytes, expected {expected_bytes} bytes for SI={si}"
        ));
    }
    if file_len != expected_bytes {
        return Err(format!(
            "Bruker procs SI={si} ({expected_bytes} bytes) disagrees with {label} length ({file_len} bytes)"
        ));
    }
    let mut out = Vec::with_capacity(si);
    for i in 0..si {
        let val = if dtype == 2 {
            let slice = &raw_bytes[i * 8..(i + 1) * 8];
            let chunk: [u8; 8] = slice
                .try_into()
                .map_err(|_| format!("short 8-byte word at index {i}"))?;
            let v = if big_endian {
                f64::from_be_bytes(chunk)
            } else {
                f64::from_le_bytes(chunk)
            };
            v * scale
        } else {
            let slice = &raw_bytes[i * 4..(i + 1) * 4];
            let chunk: [u8; 4] = slice
                .try_into()
                .map_err(|_| format!("short 4-byte word at index {i}"))?;
            let raw = if big_endian {
                i32::from_be_bytes(chunk)
            } else {
                i32::from_le_bytes(chunk)
            };
            (raw as f64) * scale
        };
        if !val.is_finite() {
            return Err(format!(
                "non-finite intensity in Bruker {label} at point {i}: {val}"
            ));
        }
        out.push(val);
    }
    Ok(out)
}

/// Decode a processed Bruker 1D `pdata` spectrum directly from in-memory `procs`, `1r`, optional `1i`, and optional `acqus` (`wasm32`-compatible).
pub fn decode_bruker_pdata_1d_bytes(
    name: &str,
    procs: &str,
    bytes_1r: &[u8],
    bytes_1i: Option<&[u8]>,
    acqus_opt: Option<&str>,
    mhz_override: Option<f64>,
) -> Result<Spectrum1D, String> {
    if procs.len() > 4 * 1024 * 1024 {
        return Err(format!(
            "Bruker procs for '{name}' exceeds 4 MiB ({} bytes)",
            procs.len()
        ));
    }
    let req_finite = |key: &str| -> Result<f64, String> {
        let s = find_jcamp_param(procs, key)
            .ok_or_else(|| format!("Bruker procs is missing parameter {key}"))?;
        let v = s
            .parse::<f64>()
            .map_err(|_| format!("Bruker procs parameter {key}={s:?} is not a number"))?;
        if !v.is_finite() {
            return Err(format!("Bruker procs parameter {key}={s:?} is non-finite"));
        }
        Ok(v)
    };
    let si_f = req_finite("SI")?;
    if si_f < 2.0 || (si_f - si_f.round()).abs() > 1e-9 {
        return Err(format!("Bruker procs SI={si_f} must be an integer >= 2"));
    }
    if si_f > (MAX_1D_INGEST_POINTS as f64) {
        return Err(format!(
            "Bruker procs declared SI={si_f:.0} exceeds MAX_1D_INGEST_POINTS ({MAX_1D_INGEST_POINTS})"
        ));
    }
    let si = si_f.round() as usize;
    let offset = req_finite("OFFSET")?;
    let sw_p = req_finite("SWP")?;
    let sf = req_finite("SF")?;
    if sf <= 0.0 || sw_p <= 0.0 {
        return Err(format!(
            "Bruker procs SF ({sf}) and SW_p ({sw_p}) must be positive"
        ));
    }
    let nc_proc = match find_jcamp_param(procs, "NCPROC") {
        Some(s) => {
            let v = s
                .parse::<i32>()
                .map_err(|_| format!("Bruker procs NCPROC={s:?} is not an integer"))?;
            if !(-128..=128).contains(&v) {
                return Err(format!("Bruker procs NCPROC={v} out of range [-128, 128]"));
            }
            v
        }
        None => 0,
    };
    let big_endian = match find_jcamp_param(procs, "BYTORDP").as_deref() {
        Some("1") => true,
        Some("0") | None => false,
        Some(other) => {
            return Err(format!(
                "Bruker procs BYTORDP={other:?} must be 0 (little-endian) or 1 (big-endian)"
            ))
        }
    };
    let dtype = match find_jcamp_param(procs, "DTYPP") {
        Some(s) => {
            let d = s
                .parse::<u8>()
                .map_err(|_| format!("Bruker procs DTYPP={s:?} is not an integer"))?;
            if d != 0 && d != 2 {
                return Err(format!(
                    "unsupported Bruker procs DTYPP={d} (expected 0=int32 or 2=float64)"
                ));
            }
            d
        }
        None => 0,
    };
    let word_size = if dtype == 2 { 8usize } else { 4usize };
    let scale = 2.0_f64.powi(nc_proc);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(format!("non-finite Bruker NCPROC={nc_proc} scale factor"));
    }

    let real = decode_bruker_1d_channel_bytes(
        &format!("{name}/1r"),
        bytes_1r,
        si,
        word_size,
        dtype,
        big_endian,
        scale,
    )?;
    let imag = match bytes_1i {
        Some(b1i) => Some(decode_bruker_1d_channel_bytes(
            &format!("{name}/1i"),
            b1i,
            si,
            word_size,
            dtype,
            big_endian,
            scale,
        )?),
        None => None,
    };

    let acqus = acqus_opt.unwrap_or("");
    let nuc_raw = find_jcamp_param(acqus, "NUC1")
        .or_else(|| find_jcamp_param(procs, "AXNUC"))
        .unwrap_or_else(|| "1H".to_string());
    let nuc_clean = nuc_raw.trim_matches(['<', '>', '"', ' ']);
    let pulprog = find_jcamp_param(acqus, "PULPROG").unwrap_or_default();
    let exp =
        Experiment1DKind::from_pulse_program(pulprog.trim_matches(['<', '>', '"', ' ']), nuc_clean);
    let mhz = mhz_override.unwrap_or(sf);
    if !mhz.is_finite() || mhz <= 0.0 {
        return Err(format!("invalid observation frequency mhz={mhz}"));
    }
    let last_x = offset - (((si - 1) as f64) * sw_p) / sf / (si as f64);
    let axis = AxisPpm::from_bounds(exp.nucleus().as_str(), si, offset, last_x, mhz);
    let mut spec = Spectrum1D::new(name.to_string(), exp, axis, real, imag);
    if let Some(solv) = find_jcamp_param(acqus, "SOLVENT") {
        let sc = solv.trim_matches(['<', '>', '"', ' ']).trim();
        if !sc.is_empty() && !sc.eq_ignore_ascii_case("none") {
            spec.solvent = Some(sc.to_string());
        }
    }
    spec.validate()?;
    Ok(spec)
}

fn decode_bruker_1d_channel(
    path: &std::path::Path,
    si: usize,
    word_size: usize,
    dtype: u8,
    big_endian: bool,
    scale: f64,
) -> Result<Vec<f64>, String> {
    let expected_bytes = (si as u64)
        .checked_mul(word_size as u64)
        .ok_or_else(|| format!("Bruker SI={si} byte count overflow"))?;
    let meta = std::fs::metadata(path)
        .map_err(|e| format!("failed to stat Bruker binary {}: {e}", path.display()))?;
    let file_len = meta.len();
    if file_len < expected_bytes {
        return Err(format!(
            "truncated Bruker {}: file has {file_len} bytes, expected {expected_bytes} bytes for SI={si}",
            path.display()
        ));
    }
    if file_len != expected_bytes {
        return Err(format!(
            "Bruker procs SI={si} ({expected_bytes} bytes) disagrees with {} file length ({file_len} bytes)",
            path.display()
        ));
    }
    let raw_bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read Bruker {}: {e}", path.display()))?;
    if raw_bytes.len() as u64 != expected_bytes {
        return Err(format!(
            "Bruker {} byte length changed during read ({} vs {expected_bytes})",
            path.display(),
            raw_bytes.len()
        ));
    }

    let mut out = Vec::with_capacity(si);
    for i in 0..si {
        let val = if dtype == 2 {
            let slice = &raw_bytes[i * 8..(i + 1) * 8];
            let chunk: [u8; 8] = slice
                .try_into()
                .map_err(|_| format!("short 8-byte word at index {i}"))?;
            let v = if big_endian {
                f64::from_be_bytes(chunk)
            } else {
                f64::from_le_bytes(chunk)
            };
            v * scale
        } else {
            let slice = &raw_bytes[i * 4..(i + 1) * 4];
            let chunk: [u8; 4] = slice
                .try_into()
                .map_err(|_| format!("short 4-byte word at index {i}"))?;
            let raw = if big_endian {
                i32::from_be_bytes(chunk)
            } else {
                i32::from_le_bytes(chunk)
            };
            (raw as f64) * scale
        };
        if !val.is_finite() {
            return Err(format!(
                "non-finite intensity in Bruker {} at point {i}: {val}",
                path.display()
            ));
        }
        out.push(val);
    }
    Ok(out)
}

/// Load a processed Bruker 1D `pdata` directory (`dir/procs` + `dir/1r` or `dir/pdata/1/procs` + `dir/pdata/1/1r`),
/// validating `SI <= MAX_1D_INGEST_POINTS` and verifying exact `1r` (and optional `1i`) byte length via
/// `fs::metadata` **before** allocating any spectrum buffer (`F15`).
pub fn load_bruker_pdata_1d_dir(
    dir: &std::path::Path,
    mhz_override: Option<f64>,
) -> Result<Option<Spectrum1D>, String> {
    let candidates = [dir.to_path_buf(), dir.join("pdata").join("1")];
    for pdir in &candidates {
        let procs_path = pdir.join("procs");
        let data_1r_path = pdir.join("1r");
        if !procs_path.is_file() || !data_1r_path.is_file() {
            continue;
        }
        let procs_len = std::fs::metadata(&procs_path)
            .map_err(|e| format!("failed to stat {}: {e}", procs_path.display()))?
            .len();
        if procs_len > 4 * 1024 * 1024 {
            return Err(format!(
                "Bruker procs {} exceeds 4 MiB ({procs_len} bytes)",
                procs_path.display()
            ));
        }
        let procs = std::fs::read_to_string(&procs_path)
            .map_err(|e| format!("failed to read {}: {e}", procs_path.display()))?;

        let req_finite = |key: &str| -> Result<f64, String> {
            let s = find_jcamp_param(&procs, key)
                .ok_or_else(|| format!("Bruker procs is missing parameter {key}"))?;
            let v = s
                .parse::<f64>()
                .map_err(|_| format!("Bruker procs parameter {key}={s:?} is not a number"))?;
            if !v.is_finite() {
                return Err(format!("Bruker procs parameter {key}={s:?} is non-finite"));
            }
            Ok(v)
        };

        let si_f = req_finite("SI")?;
        if si_f < 2.0 || (si_f - si_f.round()).abs() > 1e-9 {
            return Err(format!("Bruker procs SI={si_f} must be an integer >= 2"));
        }
        if si_f > (MAX_1D_INGEST_POINTS as f64) {
            return Err(format!(
                "Bruker procs declared SI={si_f:.0} exceeds MAX_1D_INGEST_POINTS ({MAX_1D_INGEST_POINTS})"
            ));
        }
        let si = si_f.round() as usize;
        let offset = req_finite("OFFSET")?;
        let sw_p = req_finite("SWP")?;
        let sf = req_finite("SF")?;
        if sf <= 0.0 || sw_p <= 0.0 {
            return Err(format!(
                "Bruker procs SF ({sf}) and SW_p ({sw_p}) must be positive"
            ));
        }

        let nc_proc = match find_jcamp_param(&procs, "NCPROC") {
            Some(s) => {
                let v = s
                    .parse::<i32>()
                    .map_err(|_| format!("Bruker procs NCPROC={s:?} is not an integer"))?;
                if !(-128..=128).contains(&v) {
                    return Err(format!("Bruker procs NCPROC={v} out of range [-128, 128]"));
                }
                v
            }
            None => 0,
        };
        let big_endian = match find_jcamp_param(&procs, "BYTORDP").as_deref() {
            Some("1") => true,
            Some("0") | None => false,
            Some(other) => {
                return Err(format!(
                    "Bruker procs BYTORDP={other:?} must be 0 (little-endian) or 1 (big-endian)"
                ))
            }
        };
        let dtype = match find_jcamp_param(&procs, "DTYPP") {
            Some(s) => {
                let d = s
                    .parse::<u8>()
                    .map_err(|_| format!("Bruker procs DTYPP={s:?} is not an integer"))?;
                if d != 0 && d != 2 {
                    return Err(format!(
                        "unsupported Bruker procs DTYPP={d} (expected 0=int32 or 2=float64)"
                    ));
                }
                d
            }
            None => 0,
        };
        let word_size = if dtype == 2 { 8usize } else { 4usize };
        let scale = 2.0_f64.powi(nc_proc);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("non-finite Bruker NCPROC={nc_proc} scale factor"));
        }

        // Pre-allocation file-size check + decode on `1r` (and `1i` if present)
        let real =
            decode_bruker_1d_channel(&data_1r_path, si, word_size, dtype, big_endian, scale)?;
        let data_1i_path = pdir.join("1i");
        let imag = if data_1i_path.is_file() {
            Some(decode_bruker_1d_channel(
                &data_1i_path,
                si,
                word_size,
                dtype,
                big_endian,
                scale,
            )?)
        } else {
            None
        };

        let acqus = pdir
            .parent()
            .and_then(|p| p.parent())
            .map(|exp_dir| exp_dir.join("acqus"))
            .filter(|p| p.is_file())
            .or_else(|| {
                let direct_acqus = dir.join("acqus");
                direct_acqus.is_file().then_some(direct_acqus)
            })
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_default();

        let nuc_raw = find_jcamp_param(&acqus, "NUC1")
            .or_else(|| find_jcamp_param(&procs, "AXNUC"))
            .unwrap_or_else(|| "1H".to_string());
        let nuc_clean = nuc_raw.trim_matches(['<', '>', '"', ' ']);
        let pulprog = find_jcamp_param(&acqus, "PULPROG").unwrap_or_default();
        let exp = Experiment1DKind::from_pulse_program(
            pulprog.trim_matches(['<', '>', '"', ' ']),
            nuc_clean,
        );
        let mhz = mhz_override.unwrap_or(sf);
        if !mhz.is_finite() || mhz <= 0.0 {
            return Err(format!("invalid observation frequency mhz={mhz}"));
        }
        let last_x = offset - (((si - 1) as f64) * sw_p) / sf / (si as f64);
        let axis = AxisPpm::from_bounds(exp.nucleus().as_str(), si, offset, last_x, mhz);
        let name = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("bruker_1d")
            .to_string();
        let mut spec = Spectrum1D::new(name, exp, axis, real, imag);
        if let Some(solv) = find_jcamp_param(&acqus, "SOLVENT") {
            let sc = solv.trim_matches(['<', '>', '"', ' ']).trim();
            if !sc.is_empty() && !sc.eq_ignore_ascii_case("none") {
                spec.solvent = Some(sc.to_string());
            }
        }
        spec.validate()?;
        return Ok(Some(spec));
    }
    Ok(None)
}

fn parse_experiment_kind_label_1d(label: &str) -> Result<Experiment1DKind, String> {
    let norm = label
        .trim()
        .to_ascii_lowercase()
        .replace(['-', '_', '{', '}', ' '], "");
    if norm.is_empty() {
        return Err("empty experiment/nucleus label in 1D metadata".to_string());
    }
    match norm.as_str() {
        "1h" | "h1" | "proton" | "proton1d" => Ok(Experiment1DKind::Proton1D),
        "13c" | "c13" | "13c1h" | "carbon" | "carbon13" | "carbon13decoupled" => {
            Ok(Experiment1DKind::Carbon13Decoupled)
        }
        "dept90" => Ok(Experiment1DKind::Dept90),
        "dept135" | "deptq135" => Ok(Experiment1DKind::Dept135),
        "19f" | "f19" | "fluorine19" => Ok(Experiment1DKind::Fluorine19),
        "31p" | "p31" | "phosphorus31" => Ok(Experiment1DKind::Phosphorus31),
        _ => Ok(Experiment1DKind::Custom(Nucleus1D::parse(label))),
    }
}

fn parse_finite_meta_f64(
    meta: &std::collections::BTreeMap<String, String>,
    key: &str,
) -> Result<Option<f64>, String> {
    let Some(raw) = meta.get(key) else {
        return Ok(None);
    };
    let v = raw
        .parse::<f64>()
        .map_err(|_| format!("invalid numeric metadata {key}={raw:?} in 1D TSV"))?;
    if !v.is_finite() {
        return Err(format!(
            "non-finite numeric metadata {key}={raw:?} in 1D TSV"
        ));
    }
    Ok(Some(v))
}

/// Parse a 2- or 3-column 1D TSV/CSV string (either frequency-domain `ppm\treal[\timag]` or
/// time-domain `# domain=fid` / `t_s\tre\tim`), rejecting malformed rows, column-width mismatches,
/// `< 2` rows, degenerate axes, and `NaN`/`Inf` values (`F15`).
pub fn parse_tsv_1d_str(
    text: &str,
    name: &str,
    mhz_override: Option<f64>,
) -> Result<Loaded1DData, String> {
    let mut meta: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    let mut header_cols: Vec<String> = Vec::new();
    let mut header_seen = false;
    let mut expected_cols: Option<usize> = None;
    let mut rows: Vec<Vec<f64>> = Vec::new();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            for part in comment.split([';', ',']) {
                if let Some((k, v)) = part.split_once('=') {
                    meta.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
                }
            }
            continue;
        }
        let fields: Vec<&str> = line
            .split(['\t', ','])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if fields.len() < 2 {
            return Err(format!(
                "malformed 1D TSV row at line {} (expected >= 2 tab/comma columns): {line:?}",
                line_idx + 1
            ));
        }
        let first_f64 = fields[0].parse::<f64>();
        if rows.is_empty() && !header_seen && first_f64.is_err() {
            header_cols = fields.iter().map(|s| s.to_ascii_lowercase()).collect();
            header_seen = true;
            continue;
        }
        if let Some(nc) = expected_cols {
            if fields.len() != nc {
                return Err(format!(
                    "inconsistent column count at 1D TSV line {}: got {}, expected {nc}",
                    line_idx + 1,
                    fields.len()
                ));
            }
        } else {
            expected_cols = Some(fields.len());
        }
        if rows.len() >= MAX_1D_INGEST_POINTS {
            return Err(format!(
                "1D TSV exceeds MAX_1D_INGEST_POINTS ({MAX_1D_INGEST_POINTS})"
            ));
        }
        let mut vals = Vec::with_capacity(fields.len());
        for tok in &fields {
            let v = tok.parse::<f64>().map_err(|e| {
                format!(
                    "invalid numeric value {tok:?} at 1D TSV line {}: {e}",
                    line_idx + 1
                )
            })?;
            if !v.is_finite() {
                return Err(format!(
                    "non-finite value {tok:?} at 1D TSV line {}",
                    line_idx + 1
                ));
            }
            vals.push(v);
        }
        rows.push(vals);
    }

    if rows.len() < 2 {
        return Err("1D TSV input contains fewer than 2 numeric rows".to_string());
    }

    let exp = meta
        .get("experiment")
        .or_else(|| meta.get("nucleus"))
        .map(|s| parse_experiment_kind_label_1d(s))
        .transpose()?
        .unwrap_or(Experiment1DKind::Proton1D);

    let meta_mhz = match parse_finite_meta_f64(&meta, "obs_mhz")? {
        Some(v) => Some(v),
        None => parse_finite_meta_f64(&meta, "mhz")?,
    };
    let mhz = mhz_override.or(meta_mhz).unwrap_or(500.0);
    if !mhz.is_finite() || mhz <= 0.0 {
        return Err(format!("invalid 1D observation frequency obs_mhz={mhz}"));
    }

    let first_col_name = header_cols.first().map(String::as_str).unwrap_or("");
    let is_fid = meta
        .get("domain")
        .is_some_and(|d| d.eq_ignore_ascii_case("time") || d.eq_ignore_ascii_case("fid"))
        || matches!(first_col_name, "t" | "t_s" | "time" | "time_s" | "sample");

    if is_fid {
        let meta_sw = parse_finite_meta_f64(&meta, "sw_hz")?;
        if let Some(sw) = meta_sw {
            if sw <= 0.0 {
                return Err(format!("1D FID TSV sw_hz={sw} must be > 0"));
            }
        }
        let has_time_col = rows[0].len() >= 3;
        let dt_from_col = if has_time_col {
            let dt = rows[1][0] - rows[0][0];
            if !dt.is_finite() || dt <= 0.0 {
                return Err(format!(
                    "1D FID TSV time column must be strictly increasing (dt={dt})"
                ));
            }
            Some(dt)
        } else {
            None
        };
        let sw_hz = meta_sw
            .or_else(|| dt_from_col.map(|dt| 1.0 / dt))
            .unwrap_or(6000.0);
        if !sw_hz.is_finite() || sw_hz <= 0.0 {
            return Err(format!("invalid 1D FID sw_hz={sw_hz}"));
        }
        let carrier_ppm = parse_finite_meta_f64(&meta, "carrier_ppm")?.unwrap_or(4.70);
        let gd = parse_finite_meta_f64(&meta, "group_delay_pts")?.unwrap_or(0.0);
        let mut data = Vec::with_capacity(rows.len());
        for r in &rows {
            if r.len() >= 3 {
                data.push([r[1], r[2]]);
            } else {
                data.push([r[0], r[1]]);
            }
        }
        let mut td = TimeDomain1D::new(name.to_string(), exp, sw_hz, mhz, carrier_ppm, data);
        td.group_delay_pts = gd;
        td.validate()?;
        return Ok(Loaded1DData::TimeDomain(td));
    }

    let mut pts: Vec<(f64, f64, Option<f64>)> = rows
        .into_iter()
        .map(|r| (r[0], r[1], r.get(2).copied()))
        .collect();
    if pts.first().map(|p| p.0).unwrap_or(0.0) < pts.last().map(|p| p.0).unwrap_or(0.0) {
        pts.reverse();
    }
    let high_ppm = pts[0].0;
    let low_ppm = pts.last().map(|p| p.0).unwrap_or(high_ppm);
    if (high_ppm - low_ppm).abs() <= 1e-12 {
        return Err(format!(
            "1D TSV spectrum {name:?} has degenerate chemical-shift bounds ({high_ppm} == {low_ppm})"
        ));
    }
    let real: Vec<f64> = pts.iter().map(|p| p.1).collect();
    let has_imag = pts.iter().all(|p| p.2.is_some());
    let imag: Option<Vec<f64>> = has_imag.then(|| pts.iter().filter_map(|p| p.2).collect());

    let axis = AxisPpm::from_bounds(exp.nucleus().as_str(), real.len(), high_ppm, low_ppm, mhz);
    let mut spec = Spectrum1D::new(name.to_string(), exp, axis, real, imag);
    if let Some(solv) = meta.get("solvent") {
        if !solv.is_empty() && solv != "none" {
            spec.solvent = Some(solv.clone());
        }
    }
    spec.validate()?;
    Ok(Loaded1DData::FrequencyDomain(spec))
}

/// Parse and strictly validate a JSON-serialized `Spectrum1D` or `TimeDomain1D` (`F15`).
pub fn parse_json_1d_str(text: &str, mhz_override: Option<f64>) -> Result<Loaded1DData, String> {
    let val: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("malformed 1D JSON: {e}"))?;
    let obj = val
        .as_object()
        .ok_or_else(|| "1D JSON root must be a JSON object".to_string())?;

    // Pre-check declared sizes before typed deserialization
    if let Some(ax) = obj.get("axis").and_then(|a| a.as_object()) {
        if let Some(sz) = ax.get("size").and_then(|s| s.as_u64()) {
            if sz > MAX_1D_INGEST_POINTS as u64 {
                return Err(format!(
                    "1D JSON declared axis.size={sz} exceeds MAX_1D_INGEST_POINTS ({MAX_1D_INGEST_POINTS})"
                ));
            }
        }
    }
    if let Some(atd) = obj.get("active_td").and_then(|s| s.as_u64()) {
        if atd > MAX_1D_INGEST_POINTS as u64 {
            return Err(format!(
                "1D JSON declared active_td={atd} exceeds MAX_1D_INGEST_POINTS ({MAX_1D_INGEST_POINTS})"
            ));
        }
    }

    if obj.contains_key("real") && obj.contains_key("axis") {
        let mut spec: Spectrum1D = serde_json::from_value(val)
            .map_err(|e| format!("invalid `Spectrum1D` JSON schema: {e}"))?;
        if let Some(m) = mhz_override {
            spec.axis.obs_mhz = m;
        }
        spec.validate()?;
        return Ok(Loaded1DData::FrequencyDomain(spec));
    }
    if obj.contains_key("data") && obj.contains_key("sw_hz") {
        let mut td: TimeDomain1D = serde_json::from_value(val)
            .map_err(|e| format!("invalid `TimeDomain1D` JSON schema: {e}"))?;
        if let Some(m) = mhz_override {
            td.obs_mhz = m;
        }
        td.validate()?;
        return Ok(Loaded1DData::TimeDomain(td));
    }
    Err("JSON object matched neither `Spectrum1D` nor `TimeDomain1D` schema".to_string())
}

/// Returns `true` if `bytes` begins with the 8-byte JEOL Delta NMR file magic `"JEOL.NMR"`.
#[inline]
pub fn is_jeol_jdf_magic(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[0..8] == b"JEOL.NMR"
}

/// Compute the multi-stage FIR digital decimation group delay (in final decimated complex samples)
/// from JEOL Delta `orders` (`"S o_1 ... o_S"` or `"o_1 ... o_S"`) and `factors` (`"f_1 ... f_S"`):
/// $$\tau_{\text{gd}} = \sum_{s=0}^{S-1} \frac{o_s - 1}{2 \prod_{k=s}^{S-1} f_k}.$$
pub fn compute_jeol_fir_group_delay(orders_str: &str, factors_str: &str) -> Option<f64> {
    let factors: Vec<u64> = factors_str
        .split_whitespace()
        .map(|t| t.parse::<u64>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if factors.is_empty() || factors.iter().any(|&f| f == 0 || f > 65_536) {
        return None;
    }
    let raw_orders: Vec<u64> = orders_str
        .split_whitespace()
        .map(|t| t.parse::<u64>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    let orders: &[u64] =
        if raw_orders.len() == factors.len() + 1 && (raw_orders[0] as usize) == factors.len() {
            &raw_orders[1..]
        } else if raw_orders.len() == factors.len() {
            raw_orders.as_slice()
        } else {
            return None;
        };
    if orders.iter().any(|&o| o == 0 || o > 1_000_000) {
        return None;
    }

    let s_cnt = factors.len();
    let mut gd = 0.0_f64;
    for s in 0..s_cnt {
        let mut prod = 1.0_f64;
        for &fk in &factors[s..s_cnt] {
            prod *= fk as f64;
        }
        gd += ((orders[s] - 1) as f64) / (2.0 * prod);
    }
    gd.is_finite().then_some(gd)
}

/// Clean-room JEOL Delta `.jdf` 1D FID / processed-spectrum parser (`F29` & `F15`).
///
/// Parses the 1296-byte big-endian `"JEOL.NMR"` header and 64-byte parameter table records
/// (`Endian`, `Major_Version`, `Data_Dimension_Number == 1`, `Data_Format`, `Data_Axis_Type`,
/// `Data_Axis_Units`, `Data_Points`, `Data_Offset_Start`/`Stop`, `Data_Axis_Start`/`Stop`,
/// `Base_Freq`, `X_DOMAIN`, `X_FREQ`, `X_SWEEP`, `X_OFFSET`, `DIGITAL_FILTER`, `orders`,
/// `factors`, `Trapezoid`, `solvent`), enforcing pre-allocation `MAX_1D_INGEST_BYTES`,
/// `MAX_1D_INGEST_POINTS`, truncation, and `NaN`/`Inf` validation.
pub fn parse_jeol_jdf_1d_bytes(
    name: &str,
    bytes: &[u8],
    mhz_override: Option<f64>,
) -> Result<Loaded1DData, String> {
    const JEOL_HEADER_BYTES: usize = 1296;
    if let Some(m) = mhz_override {
        if !m.is_finite() || m <= 0.0 {
            return Err(format!("invalid mhz_override={m} (must be finite and > 0)"));
        }
    }
    if (bytes.len() as u64) > MAX_1D_INGEST_BYTES {
        return Err(format!(
            "JEOL .jdf {name:?} ({} bytes) exceeds MAX_1D_INGEST_BYTES ({MAX_1D_INGEST_BYTES})",
            bytes.len()
        ));
    }
    if bytes.len() < JEOL_HEADER_BYTES {
        return Err(format!(
            "truncated JEOL .jdf {name:?}: {} bytes < {JEOL_HEADER_BYTES}-byte header",
            bytes.len()
        ));
    }
    if !is_jeol_jdf_magic(bytes) {
        return Err(format!(
            "invalid JEOL .jdf {name:?}: missing \"JEOL.NMR\" 8-byte header magic"
        ));
    }

    let read_be_u32 = |off: usize| -> Result<u32, String> {
        let s = bytes
            .get(off..off + 4)
            .ok_or_else(|| format!("JEOL .jdf header u32 out of bounds at {off}"))?;
        let arr: [u8; 4] = s
            .try_into()
            .map_err(|_| format!("JEOL .jdf header u32 slice error at {off}"))?;
        Ok(u32::from_be_bytes(arr))
    };
    let read_be_f64 = |off: usize| -> Result<f64, String> {
        let s = bytes
            .get(off..off + 8)
            .ok_or_else(|| format!("JEOL .jdf header f64 out of bounds at {off}"))?;
        let arr: [u8; 8] = s
            .try_into()
            .map_err(|_| format!("JEOL .jdf header f64 slice error at {off}"))?;
        Ok(f64::from_be_bytes(arr))
    };

    let endian_flag = bytes[8];
    let little_endian = match endian_flag {
        0 => false,
        1 => true,
        other => {
            return Err(format!(
                "invalid JEOL .jdf {name:?}: Endian byte={other} (expected 0=BE or 1=LE)"
            ))
        }
    };
    let major_ver = bytes[9];
    if major_ver == 0 {
        return Err(format!(
            "invalid JEOL .jdf {name:?}: Major_Version=0 is unsupported"
        ));
    }
    let dim_num = bytes[12];
    if dim_num != 1 {
        return Err(format!(
            "unsupported JEOL .jdf {name:?}: Data_Dimension_Number={dim_num} (expected 1D)"
        ));
    }

    let data_format = bytes[14] & 0x3F;
    let word_bytes: usize = match data_format {
        1 => 8,     // 64Bit_Float
        2 => 4,     // 32Bit_Float
        3 | 4 => 4, // 32Bit_Integer
        other => {
            return Err(format!(
            "unsupported JEOL .jdf {name:?}: Data_Format={other} (expected 1=f64, 2=f32, or 3=i32)"
        ))
        }
    };

    let axis_type = bytes[24];
    let (channels, is_interleaved) = match axis_type {
        1 => (1usize, false), // Real only
        3 => (2usize, false), // Split Complex (Real[0..N], then Imag[0..N])
        4 => (2usize, true),  // Interleaved Real_Complex ([re0, im0, re1, im1, ...])
        other => {
            return Err(format!(
                "unsupported JEOL .jdf {name:?}: Data_Axis_Type[0]={other} (expected 1=Real, 3=Complex, or 4=Real_Complex)"
            ))
        }
    };

    let axis_unit_prefix_nib = ((bytes[32] as i8) >> 4) as i32;
    let axis_unit_base = bytes[33];

    let n_pts_u32 = read_be_u32(176)?;
    let n_pts = n_pts_u32 as usize;
    if n_pts < 2 {
        return Err(format!(
            "JEOL .jdf {name:?} has Data_Points[0]={n_pts} (minimum 2 required)"
        ));
    }
    if n_pts > MAX_1D_INGEST_POINTS {
        return Err(format!(
            "JEOL .jdf {name:?} declared Data_Points[0]={n_pts} exceeds MAX_1D_INGEST_POINTS ({MAX_1D_INGEST_POINTS})"
        ));
    }

    let off_start_u32 = read_be_u32(208)?;
    let off_stop_u32 = read_be_u32(240)?;
    let (off_start, off_stop) = if off_start_u32 == 0 && off_stop_u32 == 0 {
        (0usize, n_pts - 1)
    } else {
        let s = off_start_u32 as usize;
        let e = off_stop_u32 as usize;
        if s > e || e >= n_pts {
            return Err(format!(
                "JEOL .jdf {name:?} invalid Data_Offset range [{s}..={e}] for Data_Points={n_pts}"
            ));
        }
        (s, e)
    };
    let valid_len = off_stop - off_start + 1;
    if valid_len < 2 {
        return Err(format!(
            "JEOL .jdf {name:?} active slice [{off_start}..={off_stop}] has fewer than 2 points"
        ));
    }

    let ax_start_raw = read_be_f64(272)?;
    let ax_stop_raw = read_be_f64(336)?;
    let base_freq_mhz = read_be_f64(1064)?;
    if !ax_start_raw.is_finite() || !ax_stop_raw.is_finite() || !base_freq_mhz.is_finite() {
        return Err(format!(
            "JEOL .jdf {name:?} contains non-finite header axis bounds or Base_Freq"
        ));
    }

    let param_start = read_be_u32(1212)? as usize;
    let param_length = read_be_u32(1216)? as usize;
    let data_start = read_be_u32(1284)? as usize;
    let data_length = read_be_u32(1292)? as usize;

    let expected_data_bytes = n_pts
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(word_bytes))
        .ok_or_else(|| format!("JEOL .jdf {name:?} data byte count overflow"))?;
    if data_start < JEOL_HEADER_BYTES {
        return Err(format!(
            "JEOL .jdf {name:?} invalid Data_Start={data_start} (< {JEOL_HEADER_BYTES})"
        ));
    }
    if data_length > 0 && data_length < expected_data_bytes {
        return Err(format!(
            "JEOL .jdf {name:?} Data_Length={data_length} smaller than expected {expected_data_bytes} bytes"
        ));
    }
    let data_end = data_start
        .checked_add(expected_data_bytes)
        .ok_or_else(|| format!("JEOL .jdf {name:?} Data_Start + size overflow"))?;
    if data_end > bytes.len() {
        return Err(format!(
            "truncated JEOL .jdf {name:?}: file has {} bytes, data section requires {data_end} bytes",
            bytes.len()
        ));
    }

    // Parse optional 64-byte parameter table records.
    let read_endian_u32 = |buf: &[u8]| -> Result<u32, String> {
        let a: [u8; 4] = buf
            .get(0..4)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| "short JEOL u32".to_string())?;
        Ok(if little_endian {
            u32::from_le_bytes(a)
        } else {
            u32::from_be_bytes(a)
        })
    };
    let read_endian_i32 = |buf: &[u8]| -> Result<i32, String> {
        let a: [u8; 4] = buf
            .get(0..4)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| "short JEOL i32".to_string())?;
        Ok(if little_endian {
            i32::from_le_bytes(a)
        } else {
            i32::from_be_bytes(a)
        })
    };
    let read_endian_i16 = |buf: &[u8]| -> Result<i16, String> {
        let a: [u8; 2] = buf
            .get(0..2)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| "short JEOL i16".to_string())?;
        Ok(if little_endian {
            i16::from_le_bytes(a)
        } else {
            i16::from_be_bytes(a)
        })
    };
    let read_endian_f64 = |buf: &[u8]| -> Result<f64, String> {
        let a: [u8; 8] = buf
            .get(0..8)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| "short JEOL f64".to_string())?;
        Ok(if little_endian {
            f64::from_le_bytes(a)
        } else {
            f64::from_be_bytes(a)
        })
    };

    let mut x_domain: Option<String> = None;
    let mut x_freq_hz: Option<f64> = None;
    let mut x_sweep_hz: Option<f64> = None;
    let mut x_sweep_clipped_hz: Option<f64> = None;
    let mut x_offset_val: Option<(f64, u8, i32)> = None;
    let mut x_points_param: Option<usize> = None;
    let mut digital_filter_enabled = false;
    let mut digital_filter_compensated = false;
    let mut orders_str: Option<String> = None;
    let mut factors_str: Option<String> = None;
    let mut trapezoid_gd: Option<f64> = None;
    let mut solvent: Option<String> = None;
    let mut pulprog_hint = String::new();

    if param_start >= JEOL_HEADER_BYTES && param_length >= 16 && param_start + 16 <= bytes.len() {
        let param_end = param_start.saturating_add(param_length).min(bytes.len());
        if param_end >= param_start + 16 {
            let hdr_p = &bytes[param_start..param_start + 16];
            let rec_size = (read_endian_u32(&hdr_p[0..4])? as usize).max(64);
            let low_idx = read_endian_u32(&hdr_p[4..8])? as usize;
            let high_idx = read_endian_u32(&hdr_p[8..12])? as usize;
            let avail_recs = (param_end - (param_start + 16)) / rec_size;
            let n_recs = if high_idx >= low_idx {
                (high_idx - low_idx + 1).min(avail_recs).min(16_384)
            } else {
                avail_recs.min(16_384)
            };
            for r_i in 0..n_recs {
                let off = param_start + 16 + r_i * rec_size;
                if off + 64 > param_end {
                    break;
                }
                let rec = &bytes[off..off + 64];
                let _scaler = read_endian_i16(&rec[4..6])?;
                let unit_nib = ((rec[6] as i8) >> 4) as i32;
                let unit_base = rec[7];
                let vtype = read_endian_u32(&rec[32..36])?;
                let raw_name = &rec[36..64];
                let name_end = raw_name
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(raw_name.len());
                let pname = String::from_utf8_lossy(&raw_name[..name_end])
                    .trim()
                    .to_string();
                if pname.is_empty() {
                    continue;
                }
                let pkey = pname.to_ascii_lowercase();

                let si_scale = if (-8..=8).contains(&unit_nib) {
                    10.0_f64.powi(-3 * unit_nib)
                } else {
                    1.0
                };

                let as_str_val = || -> Option<String> {
                    if vtype != 0 {
                        return None;
                    }
                    let vb = &rec[16..32];
                    let ve = vb.iter().position(|&b| b == 0).unwrap_or(vb.len());
                    let s = String::from_utf8_lossy(&vb[..ve]).trim().to_string();
                    (!s.is_empty()).then_some(s)
                };
                let as_f64_val = || -> Option<f64> {
                    let v = match vtype {
                        1 => read_endian_i32(&rec[16..20]).ok()? as f64,
                        2 => read_endian_f64(&rec[16..24]).ok()?,
                        _ => return None,
                    };
                    v.is_finite().then_some(v)
                };

                match pkey.as_str() {
                    "x_domain" => {
                        if let Some(s) = as_str_val() {
                            x_domain = Some(s);
                        }
                    }
                    "x_freq" => {
                        if let Some(v) = as_f64_val() {
                            let hz = if unit_base == 13 { v * si_scale } else { v };
                            if hz.is_finite() && hz > 0.0 {
                                x_freq_hz = Some(hz);
                            }
                        }
                    }
                    "x_sweep" => {
                        if let Some(v) = as_f64_val() {
                            let hz = if unit_base == 13 { v * si_scale } else { v };
                            if hz.is_finite() && hz > 0.0 {
                                x_sweep_hz = Some(hz);
                            }
                        }
                    }
                    "x_sweep_clipped" => {
                        if let Some(v) = as_f64_val() {
                            let hz = if unit_base == 13 { v * si_scale } else { v };
                            if hz.is_finite() && hz > 0.0 {
                                x_sweep_clipped_hz = Some(hz);
                            }
                        }
                    }
                    "x_offset" => {
                        if let Some(v) = as_f64_val() {
                            x_offset_val = Some((v, unit_base, unit_nib));
                        }
                    }
                    "x_points" => {
                        if let Some(v) = as_f64_val() {
                            if v >= 2.0 && v <= (MAX_1D_INGEST_POINTS as f64) {
                                x_points_param = Some(v.round() as usize);
                            }
                        }
                    }
                    "digital_filter" => {
                        if let Some(s) = as_str_val() {
                            digital_filter_enabled =
                                s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("on");
                        } else if let Some(v) = as_f64_val() {
                            digital_filter_enabled = v != 0.0;
                        }
                    }
                    "digital_filter_status" => {
                        if let Some(s) = as_str_val() {
                            let sl = s.to_ascii_lowercase();
                            if sl.contains("comp") || sl == "done" || sl == "cp" {
                                digital_filter_compensated = true;
                            }
                        }
                    }
                    "orders" => {
                        orders_str = as_str_val();
                    }
                    "factors" => {
                        factors_str = as_str_val();
                    }
                    "trapezoid" => {
                        if let Some(v) = as_f64_val() {
                            if v >= 0.0 {
                                trapezoid_gd = Some(v);
                            }
                        } else if let Some(s) = as_str_val() {
                            if let Some(tok) = s.split_whitespace().next() {
                                if let Ok(v) = tok.parse::<f64>() {
                                    if v.is_finite() && v >= 0.0 {
                                        trapezoid_gd = Some(v);
                                    }
                                }
                            }
                        }
                    }
                    "solvent" => {
                        if let Some(s) = as_str_val() {
                            if !s.eq_ignore_ascii_case("none") {
                                solvent = Some(s);
                            }
                        }
                    }
                    "experiment" | "x_title" | "x_pulse" => {
                        if let Some(s) = as_str_val() {
                            if !pulprog_hint.is_empty() {
                                pulprog_hint.push(' ');
                            }
                            pulprog_hint.push_str(&s);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Resolve nucleus & experiment kind (`1H`, `13C`, `19F`, `31P`, `15N`, `DEPT-90`, `DEPT-135`).
    let x_domain_low = x_domain.as_deref().map(|s| s.to_ascii_lowercase());
    let nuc_str = match x_domain_low.as_deref() {
        Some(d) if d.contains("proton") || d == "1h" || d == "h1" => "1H",
        Some(d) if d.contains("carbon") || d == "13c" || d == "c13" => "13C",
        Some(d) if d.contains("fluor") || d == "19f" || d == "f19" => "19F",
        Some(d) if d.contains("phosph") || d == "31p" || d == "p31" => "31P",
        Some(d) if d.contains("nitro") || d == "15n" || d == "n15" => "15N",
        Some(other) => other,
        None => "1H",
    };
    let exp = Experiment1DKind::from_pulse_program(&pulprog_hint, nuc_str);

    let obs_mhz = mhz_override
        .or_else(|| (base_freq_mhz > 0.0).then_some(base_freq_mhz))
        .or_else(|| x_freq_hz.map(|hz| hz * 1.0e-6))
        .ok_or_else(|| {
            format!(
                "JEOL .jdf {name:?} has missing or non-positive observation frequency (Base_Freq={base_freq_mhz})"
            )
        })?;
    if !obs_mhz.is_finite() || obs_mhz <= 0.0 {
        return Err(format!(
            "JEOL .jdf {name:?} resolved invalid observation frequency obs_mhz={obs_mhz}"
        ));
    }

    let carrier_ppm = match x_offset_val {
        Some((v, 26, _)) => v, // PPM
        Some((v, 13, nib)) => {
            let mult = if (-8..=8).contains(&nib) {
                10.0_f64.powi(-3 * nib)
            } else {
                1.0
            };
            (v * mult) / obs_mhz
        }
        Some((v, _, _)) => v,
        None => {
            let (lo_d, hi_d) = exp.nucleus().default_ppm_bounds();
            0.5 * (lo_d + hi_d)
        }
    };
    if !carrier_ppm.is_finite() {
        return Err(format!(
            "JEOL .jdf {name:?} has non-finite X_OFFSET carrier_ppm={carrier_ppm}"
        ));
    }

    // Decode data section scalars with strict finite checks.
    let decode_scalar = |byte_off: usize| -> Result<f64, String> {
        let v = match data_format {
            1 => {
                let a: [u8; 8] = bytes
                    .get(byte_off..byte_off + 8)
                    .and_then(|s| s.try_into().ok())
                    .ok_or_else(|| format!("JEOL .jdf f64 read out of bounds at {byte_off}"))?;
                if little_endian {
                    f64::from_le_bytes(a)
                } else {
                    f64::from_be_bytes(a)
                }
            }
            2 => {
                let a: [u8; 4] = bytes
                    .get(byte_off..byte_off + 4)
                    .and_then(|s| s.try_into().ok())
                    .ok_or_else(|| format!("JEOL .jdf f32 read out of bounds at {byte_off}"))?;
                let vf = if little_endian {
                    f32::from_le_bytes(a)
                } else {
                    f32::from_be_bytes(a)
                };
                vf as f64
            }
            _ => {
                let a: [u8; 4] = bytes
                    .get(byte_off..byte_off + 4)
                    .and_then(|s| s.try_into().ok())
                    .ok_or_else(|| format!("JEOL .jdf i32 read out of bounds at {byte_off}"))?;
                let vi = if little_endian {
                    i32::from_le_bytes(a)
                } else {
                    i32::from_be_bytes(a)
                };
                vi as f64
            }
        };
        if !v.is_finite() {
            return Err(format!(
                "JEOL .jdf {name:?} contains non-finite data value ({v}) at byte offset {byte_off}"
            ));
        }
        Ok(v)
    };

    let mut re_vec = Vec::with_capacity(valid_len);
    let mut im_vec = if channels == 2 {
        Some(Vec::with_capacity(valid_len))
    } else {
        None
    };

    if channels == 1 {
        for idx in off_start..=off_stop {
            re_vec.push(decode_scalar(data_start + idx * word_bytes)?);
        }
    } else if is_interleaved {
        for idx in off_start..=off_stop {
            let base = data_start + idx * 2 * word_bytes;
            re_vec.push(decode_scalar(base)?);
            if let Some(ref mut im) = im_vec {
                im.push(decode_scalar(base + word_bytes)?);
            }
        }
    } else {
        let imag_base = data_start + n_pts * word_bytes;
        for idx in off_start..=off_stop {
            re_vec.push(decode_scalar(data_start + idx * word_bytes)?);
            if let Some(ref mut im) = im_vec {
                im.push(decode_scalar(imag_base + idx * word_bytes)?);
            }
        }
    }

    let axis_si = if (-8..=8).contains(&axis_unit_prefix_nib) {
        10.0_f64.powi(-3 * axis_unit_prefix_nib)
    } else {
        1.0
    };
    let ax_start = ax_start_raw * axis_si;
    let ax_stop = ax_stop_raw * axis_si;

    let is_time_domain = match axis_unit_base {
        28 => true,       // Sec
        26 | 13 => false, // PPM or Hz
        _ => ax_start >= 0.0 && ax_stop > ax_start && x_sweep_hz.is_some(),
    };

    if is_time_domain {
        let sw_hz = x_sweep_hz
            .or_else(|| {
                let dur = ax_stop - ax_start;
                (dur > 1.0e-12).then(|| ((valid_len - 1) as f64) / dur)
            })
            .ok_or_else(|| {
                format!("JEOL .jdf FID {name:?} missing positive X_SWEEP and valid time duration")
            })?;
        if !sw_hz.is_finite() || sw_hz <= 0.0 {
            return Err(format!(
                "JEOL .jdf FID {name:?} has invalid spectral width sw_hz={sw_hz}"
            ));
        }
        let mut cplx = Vec::with_capacity(valid_len);
        if let Some(ref im) = im_vec {
            for (&r, &m) in re_vec.iter().zip(im.iter()) {
                cplx.push([r, m]);
            }
        } else {
            for &r in &re_vec {
                cplx.push([r, 0.0]);
            }
        }

        let mut td = TimeDomain1D::new(name.to_string(), exp, sw_hz, obs_mhz, carrier_ppm, cplx);
        if let Some(xp) = x_points_param {
            let eff_xp = xp.saturating_sub(off_start).clamp(2, td.data.len());
            td.active_td = eff_xp;
        }
        td.conjugate_for_fft = true;
        td.solvent = solvent;

        let gd = if digital_filter_enabled && !digital_filter_compensated {
            match (orders_str.as_deref(), factors_str.as_deref()) {
                (Some(os), Some(fs)) => compute_jeol_fir_group_delay(os, fs).or(trapezoid_gd),
                _ => trapezoid_gd,
            }
            .unwrap_or(0.0)
        } else {
            0.0
        };
        td.group_delay_pts = gd;
        compensate_group_delay_1d(&mut td);
        td.validate()?;
        Ok(Loaded1DData::TimeDomain(td))
    } else {
        // Frequency-domain JEOL `.jdf` (`PPM` = 26 or `Hz` = 13).
        // Negate JEOL's dispersion `imag` channel so causal Hilbert sign matches `fft_1d`.
        if let Some(ref mut im) = im_vec {
            for v in im.iter_mut() {
                *v = -*v;
            }
        }
        let (mut p_first, mut p_last) = if axis_unit_base == 13 {
            (
                carrier_ppm + ax_start / obs_mhz,
                carrier_ppm + ax_stop / obs_mhz,
            )
        } else if (ax_start_raw - ax_stop_raw).abs() > 1.0e-12 {
            (ax_start_raw, ax_stop_raw)
        } else if let Some(sw) = x_sweep_clipped_hz.or(x_sweep_hz) {
            let sw_ppm = sw / obs_mhz;
            (carrier_ppm + 0.5 * sw_ppm, carrier_ppm - 0.5 * sw_ppm)
        } else {
            return Err(format!(
                "JEOL .jdf spectrum {name:?} has degenerate frequency axis bounds [{ax_start_raw}, {ax_stop_raw}]"
            ));
        };

        if !p_first.is_finite() || !p_last.is_finite() || (p_first - p_last).abs() <= 1.0e-12 {
            return Err(format!(
                "JEOL .jdf spectrum {name:?} has invalid chemical-shift bounds [{p_first}, {p_last}]"
            ));
        }
        if p_first < p_last {
            std::mem::swap(&mut p_first, &mut p_last);
            re_vec.reverse();
            if let Some(ref mut im) = im_vec {
                im.reverse();
            }
        }
        let axis = AxisPpm::from_bounds(
            exp.nucleus().as_str(),
            re_vec.len(),
            p_first,
            p_last,
            obs_mhz,
        );
        let mut spec = Spectrum1D::new(name.to_string(), exp, axis, re_vec, im_vec);
        spec.solvent = solvent;
        spec.validate()?;
        Ok(Loaded1DData::FrequencyDomain(spec))
    }
}

/// Load a standalone 1D NMR dataset from an in-memory byte slice (`JEOL .jdf`, `JCAMP-DX`,
/// `JSON`, or 2-/3-column `TSV`/`CSV`, hardened per `F15` & `F29`).
pub fn load_1d_from_bytes(
    name: &str,
    bytes: &[u8],
    mhz_override: Option<f64>,
    _prefer_fid: bool,
) -> Result<Loaded1DData, String> {
    if let Some(m) = mhz_override {
        if !m.is_finite() || m <= 0.0 {
            return Err(format!("invalid mhz_override={m} (must be finite and > 0)"));
        }
    }
    if (bytes.len() as u64) > MAX_1D_INGEST_BYTES {
        return Err(format!(
            "1D input {name:?} ({} bytes) exceeds MAX_1D_INGEST_BYTES ({MAX_1D_INGEST_BYTES})",
            bytes.len()
        ));
    }

    let ext = name
        .rsplit_once('.')
        .map(|(_, e)| e.to_ascii_lowercase())
        .unwrap_or_default();
    let stem = name.rsplit_once('/').map(|(_, f)| f).unwrap_or(name);
    let stem = stem.rsplit_once('.').map(|(s, _)| s).unwrap_or(stem);

    if ext == "jdf" || is_jeol_jdf_magic(bytes) {
        return parse_jeol_jdf_1d_bytes(stem, bytes, mhz_override);
    }

    let text = std::str::from_utf8(bytes)
        .map_err(|e| format!("failed to decode 1D text input {name:?} as UTF-8: {e}"))?;

    if ext == "json" {
        return parse_json_1d_str(text, mhz_override).map_err(|e| format!("{e} ({name})"));
    }

    if ext == "jdx" || ext == "dx" || ext == "jcamp" || text.trim_start().starts_with("##TITLE=") {
        let blocks = crate::octet_analyt::jcamp_spec1d::read_jcamp(text)?;
        let first = blocks
            .first()
            .ok_or_else(|| format!("JCAMP-DX file {name:?} contained zero spectrum blocks"))?;
        if first.y.len() < 2 {
            return Err(format!(
                "JCAMP-DX file {name:?} contains fewer than 2 points"
            ));
        }
        let nuc_str = first.nucleus.as_deref().unwrap_or("1H");
        let exp = Experiment1DKind::from_pulse_program("", nuc_str);
        let mhz = mhz_override.or(first.mhz).unwrap_or(500.0);
        if !mhz.is_finite() || mhz <= 0.0 {
            return Err(format!("invalid JCAMP-DX observation frequency mhz={mhz}"));
        }
        let mut xs = first.x_axis();
        let mut ys = first.y.clone();
        if xs.first().copied().unwrap_or(0.0) < xs.last().copied().unwrap_or(0.0) {
            xs.reverse();
            ys.reverse();
        }
        let high_ppm = xs[0];
        let low_ppm = *xs.last().unwrap_or(&high_ppm);
        if !high_ppm.is_finite() || !low_ppm.is_finite() || (high_ppm - low_ppm).abs() <= 1e-12 {
            return Err(format!(
                "JCAMP-DX file {name:?} has invalid x-axis bounds [{low_ppm}, {high_ppm}]"
            ));
        }
        let axis = AxisPpm::from_bounds(exp.nucleus().as_str(), ys.len(), high_ppm, low_ppm, mhz);
        let title = if first.title.trim().is_empty() {
            stem.to_string()
        } else {
            first.title.clone()
        };
        let mut spec = Spectrum1D::new(title, exp, axis, ys, None);
        spec.solvent = first.solvent.clone();
        spec.validate()?;
        return Ok(Loaded1DData::FrequencyDomain(spec));
    }

    parse_tsv_1d_str(text, stem, mhz_override).map_err(|e| format!("{e} ({name})"))
}

/// Load a 1D time-domain FID or processed spectrum from a Bruker `fid`/`pdata` directory,
/// JEOL Delta `.jdf` file, `JCAMP-DX` (`.jdx`/`.dx`), JSON (`Spectrum1D`/`TimeDomain1D`),
/// or 2-/3-column TSV file (`F15` & `F29`).
pub fn load_1d_input(
    path: &std::path::Path,
    mhz_override: Option<f64>,
    prefer_fid: bool,
) -> Result<Loaded1DData, String> {
    if let Some(m) = mhz_override {
        if !m.is_finite() || m <= 0.0 {
            return Err(format!("invalid mhz_override={m} (must be finite and > 0)"));
        }
    }
    let name = path
        .file_stem()
        .or_else(|| path.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("spectrum_1d");

    if path.is_dir() {
        let read_bruker_solvent_opt = |d: &std::path::Path| -> Option<String> {
            let acqus_txt = std::fs::read_to_string(d.join("acqus"))
                .or_else(|_| std::fs::read_to_string(d.join("acqu")))
                .ok()?;
            let solv = find_jcamp_param(&acqus_txt, "SOLVENT")?;
            let sc = solv.trim_matches(['<', '>', '"', ' ']).trim();
            (!sc.is_empty() && !sc.eq_ignore_ascii_case("none")).then(|| sc.to_string())
        };
        let fid_file = path.join("fid");
        if prefer_fid && fid_file.is_file() {
            let fid1d = crate::fid::read_bruker_fid_1d(path)?;
            let mut td = TimeDomain1D::from_fid1d(&fid1d);
            if let Some(m) = mhz_override {
                td.obs_mhz = m;
            }
            td.solvent = read_bruker_solvent_opt(path);
            td.validate()?;
            return Ok(Loaded1DData::TimeDomain(td));
        }
        if let Some(spec) = load_bruker_pdata_1d_dir(path, mhz_override)? {
            return Ok(Loaded1DData::FrequencyDomain(spec));
        }
        if fid_file.is_file() {
            let fid1d = crate::fid::read_bruker_fid_1d(path)?;
            let mut td = TimeDomain1D::from_fid1d(&fid1d);
            if let Some(m) = mhz_override {
                td.obs_mhz = m;
            }
            td.solvent = read_bruker_solvent_opt(path);
            td.validate()?;
            return Ok(Loaded1DData::TimeDomain(td));
        }
        return Err(format!(
            "directory {} contains neither Bruker 1D `fid`+`acqus` nor `procs`+`1r`",
            path.display()
        ));
    }

    let meta = std::fs::metadata(path)
        .map_err(|e| format!("failed to stat 1D input {}: {e}", path.display()))?;
    if meta.len() > MAX_1D_INGEST_BYTES {
        return Err(format!(
            "1D input {} ({} bytes) exceeds MAX_1D_INGEST_BYTES ({MAX_1D_INGEST_BYTES})",
            path.display(),
            meta.len()
        ));
    }

    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read 1D input {}: {e}", path.display()))?;
    let disp_name = path.file_name().and_then(|s| s.to_str()).unwrap_or(name);
    load_1d_from_bytes(disp_name, &bytes, mhz_override, prefer_fid)
        .map_err(|e| format!("{e} ({})", path.display()))
}

/// Load a processed 1D spectrum from Bruker `pdata` (`procs` + `1r`), `JCAMP-DX`, JSON, TSV, or
/// raw time-domain FID (automatically apodizing, zero-filling, FFT-ing, and minimum-entropy
/// autophasing when the input is time-domain; hardened per `F15`).
/// Prepare a [`Loaded1DData`] item for interactive studio inspection (`wasm32` + native):
/// preserves `FrequencyDomain` spectra as-is; for raw `TimeDomain` FIDs, applies matched exponential
/// apodization (`first_point_scale = 0.5`), power-of-two zero-filling (capped at `131_072` points),
/// digital-filter-compensated `fft_1d`, minimum-entropy `auto_phase_and_apply_1d`, Whittaker
/// `correct_baseline_1d`, and optional solvent reference calibration when `solvent` metadata is present.
pub fn prepare_loaded_1d_for_studio(loaded: Loaded1DData) -> Result<Spectrum1D, String> {
    match loaded {
        Loaded1DData::FrequencyDomain(spec) => {
            spec.validate()?;
            Ok(spec)
        }
        Loaded1DData::TimeDomain(mut td) => {
            let lb = td.experiment.default_line_broadening_hz();
            apply_apodization_1d(&mut td, Apodization1D::Exponential { lb_hz: lb }, 0.5);
            let target_zf = (td.data.len().saturating_mul(2))
                .next_power_of_two()
                .min(131_072);
            if target_zf > td.data.len() {
                zero_fill_1d(&mut td, target_zf);
            }
            let mut spec = fft_1d(&td);
            let _ = auto_phase_and_apply_1d(&mut spec, &AutoPhase1DConfig::default());
            let _ = correct_baseline_1d(&mut spec, &BaselineConfig1D::default());
            if let Some(solv) = spec.solvent.clone() {
                let _ = auto_calibrate_solvent_1d(&mut spec, &solv, 0.25);
            }
            spec.validate()?;
            Ok(spec)
        }
    }
}

/// Load one or more 1D NMR datasets directly from in-memory `(relative_path, bytes)` entries
/// (`wasm32` browser folder/file drop & picker + native multi-file/archive drop).
///
/// Supports:
/// - Multi-experiment Bruker study folders (`study/1/...`, `study/10/...`), single expno folders
///   (`10/fid`+`acqus` and/or `10/pdata/1/1r`+`procs`), `pdata/1` folders, or flat sibling file
///   selections (`1r` + `procs` [+ `1i`, `acqus`], or `fid` + `acqus` [+ `acqu`, `procs`]).
/// - Embedded `.zip`, `.tar`, `.tar.gz`, and `.tgz` archives (expanded automatically).
/// - Standalone `.jdx` / `.dx` / `.jcamp` JCAMP-DX files, 1D `.json`, and 1D `ppm\treal` `.tsv`/`.csv`.
/// - Clear, actionable guidance when a bare Bruker binary (`1r`, `1i`, `fid`, `2rr`, `ser`) is dropped
///   without its sibling parameter file (`procs` / `acqus`).
pub fn load_1d_from_virtual_entries(
    entries: &[(&str, &[u8])],
    mhz_override: Option<f64>,
    prefer_fid: bool,
) -> Result<Vec<Loaded1DData>, String> {
    use std::collections::{BTreeMap, BTreeSet};

    if entries.is_empty() {
        return Err("No files provided for 1D NMR loading.".to_string());
    }

    // 1. Normalize paths, discard OS/VCS junk, and expand any `.zip`/`.tar`/`.tar.gz` archives.
    let mut owned_archive_items: Vec<(String, Vec<u8>)> = Vec::new();
    let mut flat_refs: Vec<(String, usize, bool)> = Vec::new();
    for (idx, &(raw_path, bytes)) in entries.iter().enumerate() {
        let norm = raw_path
            .replace('\\', "/")
            .trim_start_matches("./")
            .trim_start_matches('/')
            .trim_end_matches('/')
            .to_string();
        if norm.is_empty() {
            continue;
        }
        if norm.split('/').any(|seg| {
            seg.is_empty()
                || seg == "."
                || seg == ".."
                || seg == ".git"
                || seg == "__MACOSX"
                || seg.eq_ignore_ascii_case(".ds_store")
                || seg.eq_ignore_ascii_case("thumbs.db")
                || seg.starts_with("._")
        }) {
            continue;
        }
        let low = norm.to_ascii_lowercase();
        let is_arch = low.ends_with(".zip")
            || low.ends_with(".tar")
            || low.ends_with(".tar.gz")
            || low.ends_with(".tgz")
            || crate::classify::is_zip_magic(bytes)
            || (bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b);
        if is_arch {
            let unpacked = crate::classify::extract_archive_full_entries_bounded(
                &norm,
                bytes,
                512 * 1024 * 1024,
                256 * 1024 * 1024,
            )?;
            for (sub_rel, sub_bytes) in unpacked {
                let slot = owned_archive_items.len();
                owned_archive_items.push((sub_rel.clone(), sub_bytes));
                flat_refs.push((sub_rel, slot, true));
            }
        } else {
            flat_refs.push((norm, idx, false));
        }
    }

    let get_bytes = |slot: usize, from_arch: bool| -> &[u8] {
        if from_arch {
            &owned_archive_items[slot].1
        } else {
            entries[slot].1
        }
    };

    // Map `dir -> filename_lower -> (orig_rel_path, slot, from_arch)`
    let mut dir_map: BTreeMap<String, BTreeMap<String, (String, usize, bool)>> = BTreeMap::new();
    for (rel, slot, from_arch) in &flat_refs {
        let (dir, fname) = match rel.rsplit_once('/') {
            Some((d, f)) => (d.to_string(), f.to_string()),
            None => (String::new(), rel.clone()),
        };
        dir_map
            .entry(dir)
            .or_default()
            .insert(fname.to_ascii_lowercase(), (rel.clone(), *slot, *from_arch));
    }

    let parent_dir = |d: &str| -> Option<String> {
        if d.is_empty() {
            None
        } else {
            match d.rsplit_once('/') {
                Some((p, _)) => Some(p.to_string()),
                None => Some(String::new()),
            }
        }
    };

    let exp_dir_for_pdata = |pdir: &str| -> String {
        if let Some(p1) = parent_dir(pdir) {
            let p1_name = p1.rsplit_once('/').map(|(_, n)| n).unwrap_or(&p1);
            if p1_name.eq_ignore_ascii_case("pdata") {
                if let Some(exp) = parent_dir(&p1) {
                    return exp;
                }
            }
        }
        pdir.to_string()
    };

    let label_for_dir = |exp_dir: &str, fallback: &str| -> String {
        if exp_dir.is_empty() {
            fallback.to_string()
        } else {
            exp_dir.to_string()
        }
    };

    // Identify 2D/3D directories so we skip them cleanly for 1D extraction.
    let mut non_1d_exp_dirs: BTreeSet<String> = BTreeSet::new();
    let mut bare_1r_seen = false;
    let mut bare_fid_seen = false;
    let mut bare_2rr_seen = false;
    let mut bare_ser_seen = false;

    for (dir, files) in &dir_map {
        if files.contains_key("2rr") || files.contains_key("3rrr") {
            non_1d_exp_dirs.insert(exp_dir_for_pdata(dir));
            if files.contains_key("2rr")
                && (!files.contains_key("procs") || !files.contains_key("proc2s"))
            {
                bare_2rr_seen = true;
            }
        }
        if files.contains_key("ser")
            || files.contains_key("acqu2s")
            || files.contains_key("acqu3s")
            || files.contains_key("acqu2")
            || files.contains_key("acqu3")
        {
            non_1d_exp_dirs.insert(dir.clone());
            if files.contains_key("ser")
                && (!files.contains_key("acqus") || !files.contains_key("acqu2s"))
            {
                bare_ser_seen = true;
            }
        }
        if (files.contains_key("1r") || files.contains_key("1i")) && !files.contains_key("procs") {
            bare_1r_seen = true;
        }
        if files.contains_key("fid") && !files.contains_key("acqus") && !files.contains_key("acqu")
        {
            bare_fid_seen = true;
        }
    }

    // Collect candidate Bruker 1D `pdata` dirs grouped by `exp_dir` (choosing lowest procno per exp_dir)
    let mut pdata_by_exp: BTreeMap<String, (u32, String)> = BTreeMap::new();
    for (dir, files) in &dir_map {
        if files.contains_key("2rr") || files.contains_key("3rrr") {
            continue;
        }
        if files.contains_key("1r") && files.contains_key("procs") {
            let exp_d = exp_dir_for_pdata(dir);
            if non_1d_exp_dirs.contains(&exp_d) {
                continue;
            }
            let procno: u32 = dir
                .rsplit_once('/')
                .map(|(_, last)| last)
                .unwrap_or(dir)
                .parse::<u32>()
                .unwrap_or(u32::MAX - 1);
            let replace = match pdata_by_exp.get(&exp_d) {
                None => true,
                Some((best_p, _)) => procno < *best_p,
            };
            if replace {
                pdata_by_exp.insert(exp_d, (procno, dir.clone()));
            }
        }
    }

    // Collect candidate Bruker 1D `fid` dirs
    let mut fid_by_exp: BTreeSet<String> = BTreeSet::new();
    for (dir, files) in &dir_map {
        if non_1d_exp_dirs.contains(dir) {
            continue;
        }
        if files.contains_key("fid") && (files.contains_key("acqus") || files.contains_key("acqu"))
        {
            fid_by_exp.insert(dir.clone());
        }
    }

    let mut all_bruker_exps: Vec<String> = pdata_by_exp
        .keys()
        .chain(fid_by_exp.iter())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    // Sort numeric expnos naturally (`1`, `2`, `10` instead of `1`, `10`, `2`)
    all_bruker_exps.sort_by(|a, b| {
        let last_a = a.rsplit_once('/').map(|(_, n)| n).unwrap_or(a);
        let last_b = b.rsplit_once('/').map(|(_, n)| n).unwrap_or(b);
        match (last_a.parse::<u64>(), last_b.parse::<u64>()) {
            (Ok(na), Ok(nb)) => a
                .rsplit_once('/')
                .map(|(p, _)| p)
                .cmp(&b.rsplit_once('/').map(|(p, _)| p))
                .then(na.cmp(&nb)),
            _ => a.cmp(b),
        }
    });

    let mut results: Vec<Loaded1DData> = Vec::new();
    let mut consumed_files: BTreeSet<String> = BTreeSet::new();

    for exp_d in &all_bruker_exps {
        let has_pdata = pdata_by_exp.contains_key(exp_d);
        let has_fid = fid_by_exp.contains(exp_d);

        let load_pdata = |pdir: &str| -> Result<Loaded1DData, String> {
            let pfiles = &dir_map[pdir];
            let (r_rel, r_slot, r_arch) = &pfiles["1r"];
            let (p_rel, p_slot, p_arch) = &pfiles["procs"];
            let procs_txt = String::from_utf8_lossy(get_bytes(*p_slot, *p_arch));
            let bytes_1r = get_bytes(*r_slot, *r_arch);
            let bytes_1i = pfiles.get("1i").map(|(_, s, a)| get_bytes(*s, *a));
            let acqus_txt = dir_map
                .get(exp_d)
                .and_then(|ef| ef.get("acqus").or_else(|| ef.get("acqu")))
                .or_else(|| pfiles.get("acqus").or_else(|| pfiles.get("acqu")))
                .map(|(_, s, a)| String::from_utf8_lossy(get_bytes(*s, *a)).into_owned());
            let disp = label_for_dir(exp_d, if pdir.is_empty() { "bruker_1d" } else { pdir });
            let spec = decode_bruker_pdata_1d_bytes(
                &disp,
                &procs_txt,
                bytes_1r,
                bytes_1i,
                acqus_txt.as_deref(),
                mhz_override,
            )?;
            let _ = (r_rel, p_rel);
            Ok(Loaded1DData::FrequencyDomain(spec))
        };

        let load_fid = |edir: &str| -> Result<Loaded1DData, String> {
            let efiles = &dir_map[edir];
            let (_, f_slot, f_arch) = &efiles["fid"];
            let fid_bytes = get_bytes(*f_slot, *f_arch);
            let acqus_txt = efiles
                .get("acqus")
                .or_else(|| efiles.get("acqu"))
                .map(|(_, s, a)| String::from_utf8_lossy(get_bytes(*s, *a)).into_owned())
                .unwrap_or_default();
            let acqu_txt = efiles
                .get("acqu")
                .map(|(_, s, a)| String::from_utf8_lossy(get_bytes(*s, *a)).into_owned());
            let procs_txt = pdata_by_exp
                .get(edir)
                .and_then(|(_, pd)| dir_map.get(pd)?.get("procs"))
                .or_else(|| efiles.get("procs"))
                .map(|(_, s, a)| String::from_utf8_lossy(get_bytes(*s, *a)).into_owned());
            let disp = label_for_dir(edir, "bruker_fid_1d");
            let fid1d = crate::fid::read_bruker_fid_1d_from_bytes(
                &disp,
                &acqus_txt,
                acqu_txt.as_deref(),
                procs_txt.as_deref(),
                fid_bytes,
            )?;
            let mut td = TimeDomain1D::from_fid1d(&fid1d);
            if let Some(m) = mhz_override {
                td.obs_mhz = m;
            }
            if let Some(solv) = find_jcamp_param(&acqus_txt, "SOLVENT") {
                let sc = solv.trim_matches(['<', '>', '"', ' ']).trim();
                if !sc.is_empty() && !sc.eq_ignore_ascii_case("none") {
                    td.solvent = Some(sc.to_string());
                }
            }
            td.validate()?;
            Ok(Loaded1DData::TimeDomain(td))
        };

        let item = if prefer_fid && has_fid {
            load_fid(exp_d)?
        } else if has_pdata {
            let (_, ref pdir) = pdata_by_exp[exp_d];
            load_pdata(pdir)?
        } else {
            load_fid(exp_d)?
        };
        if let Some((_, pdir)) = pdata_by_exp.get(exp_d) {
            for (rel, _, _) in dir_map[pdir].values() {
                consumed_files.insert(rel.clone());
            }
        }
        if let Some(efiles) = dir_map.get(exp_d) {
            for (rel, _, _) in efiles.values() {
                consumed_files.insert(rel.clone());
            }
        }
        results.push(item);
    }

    // 3. Standalone 1D files (JEOL `.jdf`, `.jdx`/`.dx`/`.jcamp`, 1D `.json`, 1D `.tsv`/`.csv`)
    for (rel, slot, from_arch) in &flat_refs {
        if consumed_files.contains(rel) {
            continue;
        }
        let fname = rel.rsplit_once('/').map(|(_, f)| f).unwrap_or(rel);
        let low = fname.to_ascii_lowercase();
        let bytes = get_bytes(*slot, *from_arch);
        if bytes.len() as u64 > MAX_1D_INGEST_BYTES {
            continue;
        }
        let ext = fname
            .rsplit_once('.')
            .map(|(_, e)| e.to_ascii_lowercase())
            .unwrap_or_default();

        if ext == "jdf" || is_jeol_jdf_magic(bytes) {
            let stem = fname.rsplit_once('.').map(|(s, _)| s).unwrap_or(fname);
            let loaded = parse_jeol_jdf_1d_bytes(stem, bytes, mhz_override)?;
            results.push(loaded);
            continue;
        }

        let is_candidate_text = matches!(
            ext.as_str(),
            "jdx" | "dx" | "jcamp" | "json" | "tsv" | "csv" | "txt"
        );
        if !is_candidate_text {
            continue;
        }
        let Ok(text) = std::str::from_utf8(bytes) else {
            continue;
        };
        if ext == "json" {
            if let Ok(loaded) = parse_json_1d_str(text, mhz_override) {
                results.push(loaded);
            }
            continue;
        }
        if ext == "jdx"
            || ext == "dx"
            || ext == "jcamp"
            || (low != "acqus"
                && low != "acqu"
                && low != "procs"
                && low != "proc"
                && text.trim_start().starts_with("##TITLE=")
                && text.contains("##XYDATA="))
        {
            let blocks = crate::octet_analyt::jcamp_spec1d::read_jcamp(text)?;
            let first = blocks
                .first()
                .ok_or_else(|| format!("JCAMP-DX file '{rel}' contained zero spectrum blocks"))?;
            if first.y.len() < 2 {
                return Err(format!(
                    "JCAMP-DX file '{rel}' contains fewer than 2 points"
                ));
            }
            let nuc_str = first.nucleus.as_deref().unwrap_or("1H");
            let exp = Experiment1DKind::from_pulse_program("", nuc_str);
            let mhz = mhz_override.or(first.mhz).unwrap_or(500.0);
            if !mhz.is_finite() || mhz <= 0.0 {
                return Err(format!("invalid JCAMP-DX observation frequency mhz={mhz}"));
            }
            let mut xs = first.x_axis();
            let mut ys = first.y.clone();
            if xs.first().copied().unwrap_or(0.0) < xs.last().copied().unwrap_or(0.0) {
                xs.reverse();
                ys.reverse();
            }
            let high_ppm = xs[0];
            let low_ppm = *xs.last().unwrap_or(&high_ppm);
            if !high_ppm.is_finite() || !low_ppm.is_finite() || (high_ppm - low_ppm).abs() <= 1e-12
            {
                return Err(format!(
                    "JCAMP-DX file '{rel}' has invalid x-axis bounds [{low_ppm}, {high_ppm}]"
                ));
            }
            let axis =
                AxisPpm::from_bounds(exp.nucleus().as_str(), ys.len(), high_ppm, low_ppm, mhz);
            let title = if first.title.trim().is_empty() {
                fname.to_string()
            } else {
                first.title.clone()
            };
            let mut spec = Spectrum1D::new(title, exp, axis, ys, None);
            spec.solvent = first.solvent.clone();
            spec.validate()?;
            results.push(Loaded1DData::FrequencyDomain(spec));
            continue;
        }
        if matches!(ext.as_str(), "tsv" | "csv" | "txt") {
            let stem = fname.rsplit_once('.').map(|(s, _)| s).unwrap_or(fname);
            if let Ok(loaded) = parse_tsv_1d_str(text, stem, mhz_override) {
                results.push(loaded);
            }
        }
    }

    if !results.is_empty() {
        return Ok(results);
    }

    // 4. Actionable bare-file guidance when no complete 1D dataset could be assembled.
    if bare_1r_seen {
        return Err(
            "Bruker '1r' requires its sibling 'procs' file — drop the whole Bruker experiment or 'pdata/1' folder (or select '1r' + 'procs' together)."
                .to_string(),
        );
    }
    if bare_fid_seen {
        return Err(
            "Bruker 'fid' requires its sibling 'acqus' file — drop the whole Bruker experiment folder (or select 'fid' + 'acqus' together)."
                .to_string(),
        );
    }
    if bare_2rr_seen {
        return Err(
            "Bruker '2rr' requires its sibling 'procs' and 'proc2s' files — drop the whole Bruker experiment or 'pdata/1' folder."
                .to_string(),
        );
    }
    if bare_ser_seen {
        return Err(
            "Bruker 'ser' requires its sibling 'acqus' and 'acqu2s' files — drop the whole Bruker experiment folder."
                .to_string(),
        );
    }

    Err("No 1D NMR datasets found in dropped items.".to_string())
}

pub fn load_processed_spectrum_1d(
    path: &std::path::Path,
    mhz_override: Option<f64>,
) -> Result<Spectrum1D, String> {
    match load_1d_input(path, mhz_override, false)? {
        Loaded1DData::FrequencyDomain(spec) => Ok(spec),
        Loaded1DData::TimeDomain(mut td) => {
            let lb = td.experiment.default_line_broadening_hz();
            apply_apodization_1d(&mut td, Apodization1D::Exponential { lb_hz: lb }, 0.5);
            zero_fill_pow2_1d(&mut td, 2);
            let mut spec = fft_1d(&td);
            let _ = auto_phase_and_apply_1d(&mut spec, &AutoPhase1DConfig::default());
            spec.validate()?;
            Ok(spec)
        }
    }
}

#[cfg(test)]
mod virtual_ingest_1d_tests {
    use super::*;

    fn crc32_ieee(bytes: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFF_FFFF;
        for &b in bytes {
            let mut x = (crc ^ (b as u32)) & 0xFF;
            for _ in 0..8 {
                x = if x & 1 != 0 {
                    (x >> 1) ^ 0xEDB8_8320
                } else {
                    x >> 1
                };
            }
            crc = (crc >> 8) ^ x;
        }
        !crc
    }

    /// Build a PKZIP archive with optional bit-3 streaming data descriptor (`local_sizes_zero = true`)
    /// and either method `0` (Store) or method `8` (RFC 1951 stored-block DEFLATE).
    fn build_test_zip(files: &[(&str, &[u8], u16)], local_sizes_zero: bool) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cd = Vec::new();

        for &(name, raw, method) in files {
            let crc = crc32_ieee(raw);
            let comp: Vec<u8> = if method == 8 {
                // RFC 1951 BFINAL=1, BTYPE=00 (stored block): 0x01 + u16 LEN + u16 !LEN + raw
                let len = raw.len() as u16;
                let nlen = !len;
                let mut d = Vec::with_capacity(5 + raw.len());
                d.push(0x01);
                d.extend_from_slice(&len.to_le_bytes());
                d.extend_from_slice(&nlen.to_le_bytes());
                d.extend_from_slice(raw);
                d
            } else {
                raw.to_vec()
            };

            let local_off = out.len() as u32;
            let flags: u16 = if local_sizes_zero { 0x0008 } else { 0x0000 };
            out.extend_from_slice(b"PK\x03\x04");
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&flags.to_le_bytes());
            out.extend_from_slice(&method.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            if local_sizes_zero {
                out.extend_from_slice(&0u32.to_le_bytes());
                out.extend_from_slice(&0u32.to_le_bytes());
                out.extend_from_slice(&0u32.to_le_bytes());
            } else {
                out.extend_from_slice(&crc.to_le_bytes());
                out.extend_from_slice(&(comp.len() as u32).to_le_bytes());
                out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
            }
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&comp);
            if local_sizes_zero {
                out.extend_from_slice(b"PK\x07\x08");
                out.extend_from_slice(&crc.to_le_bytes());
                out.extend_from_slice(&(comp.len() as u32).to_le_bytes());
                out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
            }

            cd.extend_from_slice(b"PK\x01\x02");
            cd.extend_from_slice(&20u16.to_le_bytes());
            cd.extend_from_slice(&20u16.to_le_bytes());
            cd.extend_from_slice(&flags.to_le_bytes());
            cd.extend_from_slice(&method.to_le_bytes());
            cd.extend_from_slice(&0u32.to_le_bytes());
            cd.extend_from_slice(&crc.to_le_bytes());
            cd.extend_from_slice(&(comp.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(raw.len() as u32).to_le_bytes());
            cd.extend_from_slice(&(name.len() as u16).to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u16.to_le_bytes());
            cd.extend_from_slice(&0u32.to_le_bytes());
            cd.extend_from_slice(&local_off.to_le_bytes());
            cd.extend_from_slice(name.as_bytes());
        }

        let cd_off = out.len() as u32;
        let cd_len = cd.len() as u32;
        out.extend_from_slice(&cd);
        out.extend_from_slice(b"PK\x05\x06");
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&cd_len.to_le_bytes());
        out.extend_from_slice(&cd_off.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn test_virtual_bruker_pdata_1d_and_zip_streaming_descriptor() {
        let si = 64usize;
        let procs = format!(
            "##TITLE= test\n##$SI= {si}\n##$OFFSET= 10.0\n##$SW_p= 5000.0\n##$SF= 500.0\n##$NC_proc= 0\n##$BYTORDP= 0\n##$DTYPP= 0\n##$AXNUC= <1H>\n##END=\n"
        );
        let acqus =
            "##TITLE= acqus\n##$PULPROG= <zg30>\n##$NUC1= <1H>\n##$SOLVENT= <CDCl3>\n##END=\n";
        let mut bytes_1r = Vec::with_capacity(si * 4);
        for i in 0..si {
            let v: i32 = if i == 18 { 120_000 } else { (i as i32) * 10 };
            bytes_1r.extend_from_slice(&v.to_le_bytes());
        }

        // 1. Direct virtual entries (multi-expno folder `study/10` and `study/2` sorted `2` then `10`)
        let entries = [
            ("study/10/acqus", acqus.as_bytes()),
            ("study/10/pdata/1/procs", procs.as_bytes()),
            ("study/10/pdata/1/1r", bytes_1r.as_slice()),
            ("study/2/pdata/1/procs", procs.as_bytes()),
            ("study/2/pdata/1/1r", bytes_1r.as_slice()),
            ("study/__MACOSX/._1r", b"junk"),
        ];
        let loaded = load_1d_from_virtual_entries(&entries, None, false).expect("pdata 1D load");
        assert_eq!(loaded.len(), 2);
        let s0 = prepare_loaded_1d_for_studio(loaded.into_iter().next().unwrap()).unwrap();
        assert_eq!(s0.name, "study/2");
        assert_eq!(s0.len(), si);

        // 2. Same dataset inside a macOS-style streaming bit-3 `.zip` (method 8 Deflate + method 0 Store)
        let zip_bytes = build_test_zip(
            &[
                ("exp1/acqus", acqus.as_bytes(), 0),
                ("exp1/pdata/1/procs", procs.as_bytes(), 8),
                ("exp1/pdata/1/1r", bytes_1r.as_slice(), 8),
            ],
            true,
        );
        let z_loaded =
            load_1d_from_virtual_entries(&[("sample_1d.zip", zip_bytes.as_slice())], None, false)
                .expect("ZIP 1D load");
        assert_eq!(z_loaded.len(), 1);
        let zs = prepare_loaded_1d_for_studio(z_loaded.into_iter().next().unwrap()).unwrap();
        assert_eq!(zs.len(), si);
        assert_eq!(zs.solvent.as_deref(), Some("CDCl3"));
    }

    #[test]
    fn test_virtual_bruker_fid_1d_auto_fft_and_bare_file_errors() {
        // Bare `1r` without `procs` -> clear actionable error
        let err_1r = load_1d_from_virtual_entries(&[("pdata/1/1r", &[0u8; 128])], None, false)
            .expect_err("bare 1r must error");
        assert!(err_1r.contains("requires its sibling 'procs'"), "{err_1r}");

        // Bare `fid` without `acqus` -> clear actionable error
        let err_fid = load_1d_from_virtual_entries(&[("1/fid", &[0u8; 256])], None, false)
            .expect_err("bare fid must error");
        assert!(
            err_fid.contains("requires its sibling 'acqus'"),
            "{err_fid}"
        );

        // Complete `fid` + `acqus` -> TimeDomain -> prepare_loaded_1d_for_studio -> phased Spectrum1D
        let n_cplx = 64usize;
        let acqus = format!(
            "##TITLE= acqus\n##$TD= {}\n##$SW_h= 5000.0\n##$SFO1= 500.0\n##$BF1= 500.0\n##$BYTORDA= 0\n##$DTYPA= 0\n##$AQ_mod= 3\n##$GRPDLY= 0\n##$NUC1= <1H>\n##$PULPROG= <zg30>\n##END=\n",
            n_cplx * 2
        );
        let mut fid_bytes = Vec::with_capacity(n_cplx * 2 * 4);
        for k in 0..n_cplx {
            let t = (k as f64) / 5000.0;
            let env = (-t * 15.0).exp();
            let re = (env * (2.0 * std::f64::consts::PI * 420.0 * t).cos() * 50_000.0) as i32;
            let im = (env * (2.0 * std::f64::consts::PI * 420.0 * t).sin() * 50_000.0) as i32;
            fid_bytes.extend_from_slice(&re.to_le_bytes());
            fid_bytes.extend_from_slice(&im.to_le_bytes());
        }
        let loaded = load_1d_from_virtual_entries(
            &[
                ("ibuprofen/1/acqus", acqus.as_bytes()),
                ("ibuprofen/1/fid", fid_bytes.as_slice()),
            ],
            None,
            true,
        )
        .expect("fid 1D load");
        assert_eq!(loaded.len(), 1);
        assert!(matches!(loaded[0], Loaded1DData::TimeDomain(_)));
        let spec = prepare_loaded_1d_for_studio(loaded.into_iter().next().unwrap())
            .expect("studio FFT/phase");
        assert!(spec.len() >= 128);
        let max_real = spec.real.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_real > 0.0);
    }
}
