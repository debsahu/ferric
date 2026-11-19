//! Phase 22 Processing & Structural NMR Tools Panel (`v0.17.1`, native & `wasm32`).
//!
//! Wires four core Phase 22 quantitative engines into a unified 4-tab connected-folder workbench
//! window (`P22ToolsPanel`):
//! 1. **Deconv 2D/3D (VarPro)** (`nmr_core::deconv_nd`, `T22.1`): Coupled multi-peak separable
//!    Variable Projection (`VarPro`) Voigt/Lorentzian/Gaussian deconvolution on an active 2D
//!    spectrum ROI window or 1-click overlapped doublet synthetic demo, reporting fitted centers,
//!    linewidths, amplitudes, volumes, `reduced_chi2`, `crlb_valid`, and fail-closed diagnostics
//!    (`ActiveBoundHit`, `ModelMisspecified`, `IllConditionedCluster`, `DegeneratePeakPair`).
//! 2. **MaxEnt NUS** (`nmr_core::maxent_nus`, `T22.13`): Convex accelerated 2D hypercomplex
//!    Maximum-Entropy non-uniform sampling reconstruction with configurable `lambda` (fixed or
//!    Morozov discrepancy principle), `def`, `max_iter`, `States` / `States-TPPI` quadrature
//!    encoding (surfacing the fail-closed error on `EchoAntiecho`), KKT stationarity residual,
//!    duality gap, and explicit amplitude-nonlinearity disclosure vs linear/IST NUS.
//! 3. **ARTSY RDC** (`nmr_core::artsy_rdc`, `T22.9`): Intensity-based TROSY `^1J_NH + ^1D_NH` and
//!    `^1D_NH +/- sigma` extraction from editable TSV (`seq_id, res_name, i_ref, i_att, sigma`)
//!    or 1-click synthetic demo, with sortable table, quality flags (`Good`, `NearSaturation`,
//!    `LowSnr`, `OutOfDomainRatio`), and 1-click Copy TSV.
//! 4. **2D PCS & Curie CCR** (`nmr_core::pcs_transfer`, `T22.10`): Unassigned 2D diamagnetic-to-
//!    paramagnetic `PCS` cone assignment transfer (Kuhn-Munkres minimum-cost matching +
//!    Sinkhorn-Knopp entropic optimal-transport posteriors + iterative EM Huber-IRLS refinement)
//!    paired with Curie-spin / dipolar cross-correlated relaxation (`Gamma_CCR`) `DeltaChi` and
//!    `chi_iso` susceptibility-tensor fitting.
//!
//! # Provenance & Clean-Room Statement
//! Original clean-room GUI integration strictly grounded in peer-reviewed literature:
//! - Golub, G. H. & Pereyra, V. (*SIAM J. Numer. Anal.* **1973**, 10, 413-432); Kaufman, L.
//!   (*BIT* **1975**, 15, 49-57); Thompson, P., Cox, D. E. & Hastings, J. B. (*J. Appl.
//!   Crystallogr.* **1987**, 20, 79-83); Cramér (1946) & Rao (1945) — separable VarPro & CRLB.
//! - Skilling, J. & Bryan, R. K. (*Mon. Not. R. Astron. Soc.* **1984**, 211, 111-124); Hoch, J. C.
//!   & Stern, A. S. (*NMR Data Processing*, **1996**); Beck, A. & Teboulle, M. (*SIAM J. Imaging
//!   Sci.* **2009**, 2, 183-202); O'Donoghue, B. & Candès, E. (*Found. Comput. Math.* **2015**,
//!   15, 715-732); Worley, B. (*J. Magn. Reson.* **2016**, 265, 90-98); Morozov, V. A. (*Soviet
//!   Math. Dokl.* **1966**, 7, 414-417); Hyberts, S. G., Takeuchi, K. & Wagner, G. (*J. Am. Chem.
//!   Soc.* **2010**, 132, 2145-2147) — convex MaxEnt NUS and Poisson-gap sampling.
//! - Fitzkee, N. C. & Bax, A. (*J. Biomol. NMR* **2010**, 48, 65-70); Ying, J. et al. (*J. Biomol.
//!   NMR* **2011**, 51, 89-103) — ARTSY quantitative J-correlation RDC extraction.
//! - Bertini, I., Luchinat, C. & Parigi, G. (*Prog. Nucl. Magn. Reson. Spectrosc.* **2002**, 40,
//!   249-273); Pintacuda, G. et al. (*J. Biomol. NMR* **2003**, 27, 115-132); Orton, H. W. et al.
//!   (*J. Biomol. NMR* **2016**, 64, 243-254); Schmitz, C. et al. (*J. Biomol. NMR* **2008**, 41,
//!   179-189); Kuhn (1955), Munkres (1957), Cuturi (*NeurIPS* **2013**), Huber (1964) — 2D PCS
//!   assignment transfer and joint Curie-spin CCR susceptibility-tensor fitting.

use std::cmp::Ordering;
use std::f64::consts::PI;
use std::fmt::Write as _;

use eframe::egui;
use nmr_core::artsy_rdc::{
    artsy_abs_coupling_from_rdc, artsy_forward_intensities, extract_artsy_rdc, ArtsyConfig,
    ArtsyIntensityPair, ArtsyQualityFlag, ArtsyResidueRdc, DEFAULT_ARTSY_DEPHASING_S,
    DEFAULT_J_NH_ISO_HZ,
};
use nmr_core::deconv_nd::{
    deconvolve_roi_2d, synthesize_roi_2d, DeconvConfig, DeconvDiagnostic, DeconvResult2D,
    LineshapeKind, LinewidthSharing, PeakModel2D, PeakSeed2D, Roi2D,
};
use nmr_core::fid::QuadMode;
use nmr_core::maxent_nus::{
    poisson_gap_mask_2d, reconstruct_maxent_2d, MaxEntConfig, MaxEntLambdaMode, MaxEntResult,
    MAXENT_LINEARITY_DISCLOSURE, MAX_MAXENT_ITERATIONS,
};
use nmr_core::noesy::Vec3;
use nmr_core::paramagnetic::{predict_pcs_from_tensor, DeltaChiTensor, PcsObservation};
use nmr_core::pcs_transfer::{
    chi_iso_from_mu_eff_bohr, delta_chi_tensor_from_spherical, fit_curie_ccr_chi_iso,
    fit_joint_pcs_ccr, iterative_pcs_assignment_transfer, match_pcs_assignment_transfer,
    predict_curie_dd_ccr_s_inv, CcrFitResult, CcrObservation, CcrPhysicsParams, CcrSpinPair,
    CcrTensorModel, DiamagneticPeak2D, IterativePcsTransferConfig, IterativePcsTransferResult,
    JointPcsCcrFitOptions, JointPcsCcrFitResult, ParamagneticPeak2D, PcsMatchStatus,
    PcsTransferConfig, PcsTransferResult,
};
use nmr_core::rdc::RdcSignConvention;
use nmr_core::rng::Rng;
use nmr_core::spectrum::Spectrum2D;

use crate::vnmrj_rail::draw_folder_tab;
use crate::workbench_mode::{TYPE_BODY_PX, TYPE_CAPTION_PX, TYPE_HEADER_PX};

// ============================================================================
// 1. Public Tab Enum & Theme Palette
// ============================================================================

/// The four Phase 22 Processing & Structural NMR tool tabs in [`P22ToolsPanel`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum P22ToolTab {
    /// Coupled 2D/3D multi-peak Voigt/Lorentzian VarPro deconvolution (`T22.1`).
    #[default]
    Deconv2d3d,
    /// Convex accelerated 2D hypercomplex MaxEnt NUS reconstruction (`T22.13`).
    MaxEntNus,
    /// Intensity-based ARTSY `^1J_NH + ^1D_NH` & `^1D_NH` RDC extraction (`T22.9`).
    ArtsyRdc,
    /// Unassigned 2D PCS assignment transfer & Curie-spin `Gamma_CCR` tensor fit (`T22.10`).
    PcsCurieCcr,
}

impl P22ToolTab {
    /// Canonical left-to-right folder-tab ordering.
    pub const ALL: [Self; 4] = [
        Self::Deconv2d3d,
        Self::MaxEntNus,
        Self::ArtsyRdc,
        Self::PcsCurieCcr,
    ];

    /// Short folder-tab title rendered by [`draw_folder_tab`].
    pub const fn label(self) -> &'static str {
        match self {
            Self::Deconv2d3d => "Deconv 2D/3D (VarPro)",
            Self::MaxEntNus => "MaxEnt NUS",
            Self::ArtsyRdc => "ARTSY RDC",
            Self::PcsCurieCcr => "2D PCS & Curie CCR",
        }
    }

    /// Crisp 1-line `Input -> Output` spectroscopist guidance header shown at the top of each tab.
    pub const fn guidance_header(self) -> &'static str {
        match self {
            Self::Deconv2d3d => {
                "Input: Active 2D spectrum ROI window or synthetic overlapped doublet + editable peak seeds -> Output: Coupled VarPro centers, FWHMs, amplitudes, CRLB SEs, reduced chi2 & fail-closed diagnostics."
            }
            Self::MaxEntNus => {
                "Input: 2D NUS schedule (Poisson-gap demo or pasted t2,t1 list), quadrature mode, lambda & max_iter -> Output: Convex 2D MaxEnt spectrum, KKT residual, duality gap & nonlinearity disclosure."
            }
            Self::ArtsyRdc => {
                "Input: ARTSY peak-intensity TSV (seq_id, res_name, i_ref, i_att, sigma), dephasing T & J_iso -> Output: Sortable |1J_NH + 1D_NH| and 1D_NH +/- sigma table, quality flags & Copy TSV."
            }
            Self::PcsCurieCcr => {
                "Input: Assigned diamagnetic + unassigned paramagnetic 2D peaks (with optional 3D coords & Gamma_CCR) -> Output: Kuhn-Munkres/Sinkhorn PCS transfer + Curie-spin DeltaChi & chi_iso tensor fit."
            }
        }
    }
}

/// Theme-aware colors for [`P22ToolsPanel`] in both Paper White and Dark Oscilloscope modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct P22ToolsTheme {
    pane_bg: egui::Color32,
    card_bg: egui::Color32,
    plot_bg: egui::Color32,
    border: egui::Color32,
    text_primary: egui::Color32,
    text_muted: egui::Color32,
    accent: egui::Color32,
    ok_bg: egui::Color32,
    ok_fg: egui::Color32,
    warn_bg: egui::Color32,
    warn_fg: egui::Color32,
    err_bg: egui::Color32,
    err_fg: egui::Color32,
    guide_bg: egui::Color32,
}

impl P22ToolsTheme {
    const fn for_mode(paper_white: bool) -> Self {
        if paper_white {
            Self {
                pane_bg: egui::Color32::from_rgb(255, 255, 255),
                card_bg: egui::Color32::from_rgb(248, 250, 252),
                plot_bg: egui::Color32::from_rgb(241, 245, 249),
                border: egui::Color32::from_rgb(186, 198, 214),
                text_primary: egui::Color32::from_rgb(15, 23, 42),
                text_muted: egui::Color32::from_rgb(71, 85, 105),
                accent: egui::Color32::from_rgb(2, 132, 199),
                ok_bg: egui::Color32::from_rgb(220, 252, 231),
                ok_fg: egui::Color32::from_rgb(21, 128, 61),
                warn_bg: egui::Color32::from_rgb(254, 243, 199),
                warn_fg: egui::Color32::from_rgb(161, 98, 7),
                err_bg: egui::Color32::from_rgb(254, 226, 226),
                err_fg: egui::Color32::from_rgb(185, 28, 28),
                guide_bg: egui::Color32::from_rgb(239, 246, 255),
            }
        } else {
            Self {
                pane_bg: egui::Color32::from_rgb(15, 21, 32),
                card_bg: egui::Color32::from_rgb(21, 30, 46),
                plot_bg: egui::Color32::from_rgb(11, 17, 28),
                border: egui::Color32::from_rgb(68, 86, 114),
                text_primary: egui::Color32::from_rgb(241, 245, 249),
                text_muted: egui::Color32::from_rgb(148, 163, 184),
                accent: egui::Color32::from_rgb(56, 189, 248),
                ok_bg: egui::Color32::from_rgb(12, 62, 42),
                ok_fg: egui::Color32::from_rgb(110, 231, 183),
                warn_bg: egui::Color32::from_rgb(68, 44, 14),
                warn_fg: egui::Color32::from_rgb(252, 211, 77),
                err_bg: egui::Color32::from_rgb(76, 18, 26),
                err_fg: egui::Color32::from_rgb(252, 165, 165),
                guide_bg: egui::Color32::from_rgb(16, 34, 54),
            }
        }
    }
}

// ============================================================================
// 2. Tab 1 State: Coupled 2D/3D VarPro Deconvolution (`T22.1`)
// ============================================================================

/// Data source for 2D VarPro lineshape deconvolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeconvDataSource {
    /// Synthesized overlapped 2D doublet ROI (`synthesize_roi_2d`).
    #[default]
    SyntheticDemo,
    /// Selected ppm window cropped from the currently active 2D spectrum.
    ActiveSpectrum2D,
}

/// Non-destructively editable state for the `Deconv 2D/3D (VarPro)` tab.
#[derive(Clone, Debug)]
pub struct DeconvTabState {
    pub source: DeconvDataSource,
    /// Active 2D spectrum ROI window bounds in ppm (`F1` = indirect/Y, `F2` = direct/X).
    pub f1_min_ppm: f64,
    pub f1_max_ppm: f64,
    pub f2_min_ppm: f64,
    pub f2_max_ppm: f64,
    /// Synthetic overlapped-doublet generator parameters (non-destructively editable).
    pub demo_grid_n: usize,
    pub demo_separation_fwhm: f64,
    pub demo_fwhm_f1: f64,
    pub demo_fwhm_f2: f64,
    pub demo_amp_1: f64,
    pub demo_amp_2: f64,
    pub demo_noise_sigma: f64,
    pub demo_seed: u64,
    /// Editable VarPro solver configuration.
    pub lineshape: LineshapeKind,
    pub linewidth_sharing: LinewidthSharing,
    pub fit_mixing: bool,
    pub fit_baseline: bool,
    pub max_iterations: usize,
    pub use_known_sigma: bool,
    pub known_noise_sigma: f64,
    pub use_center_shift_bound: bool,
    pub max_center_shift_fwhm: f64,
    /// Editable peak seeds (never wiped on run or error).
    pub seeds: Vec<PeakSeed2D>,
    /// Last ROI passed to `deconvolve_roi_2d` (for 2D/slice preview).
    pub last_roi: Option<Roi2D>,
    /// Last successful `nmr_core::deconv_nd::deconvolve_roi_2d` output.
    pub result: Option<DeconvResult2D>,
    /// Last status or error message.
    pub status_message: String,
    pub error_message: Option<String>,
}

impl Default for DeconvTabState {
    fn default() -> Self {
        Self {
            source: DeconvDataSource::SyntheticDemo,
            f1_min_ppm: 118.5,
            f1_max_ppm: 121.5,
            f2_min_ppm: 8.10,
            f2_max_ppm: 8.50,
            demo_grid_n: 23,
            demo_separation_fwhm: 0.85,
            demo_fwhm_f1: 0.55,
            demo_fwhm_f2: 0.055,
            demo_amp_1: 1.0,
            demo_amp_2: 0.72,
            demo_noise_sigma: 0.005,
            demo_seed: 0x2201_0001,
            lineshape: LineshapeKind::PseudoVoigt,
            linewidth_sharing: LinewidthSharing::SharedAcrossPlanes,
            fit_mixing: true,
            fit_baseline: false,
            max_iterations: 120,
            use_known_sigma: true,
            known_noise_sigma: 0.005,
            use_center_shift_bound: false,
            max_center_shift_fwhm: 1.5,
            seeds: vec![
                PeakSeed2D {
                    center_f1: 119.75,
                    center_f2: 8.275,
                    fwhm_f1: 0.50,
                    fwhm_f2: 0.050,
                    eta_f1: 0.55,
                    eta_f2: 0.55,
                },
                PeakSeed2D {
                    center_f1: 120.25,
                    center_f2: 8.325,
                    fwhm_f1: 0.50,
                    fwhm_f2: 0.050,
                    eta_f1: 0.55,
                    eta_f2: 0.55,
                },
            ],
            last_roi: None,
            result: None,
            status_message: "Click Synthetic Demo or Run 2D VarPro Deconvolution.".to_string(),
            error_message: None,
        }
    }
}

impl DeconvTabState {
    /// Build a [`DeconvConfig`] from the current editable UI parameters.
    pub fn build_config(&self) -> DeconvConfig {
        DeconvConfig {
            lineshape: self.lineshape,
            linewidth_sharing: self.linewidth_sharing,
            fit_mixing: self.fit_mixing,
            fit_baseline: self.fit_baseline,
            max_iterations: self.max_iterations.clamp(5, 500),
            noise_sigma: if self.use_known_sigma && self.known_noise_sigma > 0.0 {
                Some(self.known_noise_sigma)
            } else {
                None
            },
            max_center_shift_fwhm: if self.use_center_shift_bound
                && self.max_center_shift_fwhm > 0.0
            {
                Some(self.max_center_shift_fwhm)
            } else {
                None
            },
            ..DeconvConfig::default()
        }
    }

    /// Synthesize the overlapped 2D doublet ROI from the current editable demo parameters.
    pub fn build_synthetic_doublet_roi(&self) -> Result<(Roi2D, Vec<PeakSeed2D>), String> {
        let n = self.demo_grid_n.clamp(11, 41);
        let f1_span = (self.demo_fwhm_f1 * 5.0).max(1.2);
        let f2_span = (self.demo_fwhm_f2 * 5.0).max(0.14);
        let f1_mid = 120.0;
        let f2_mid = 8.30;

        let coords_f1: Vec<f64> = (0..n)
            .map(|i| {
                let frac = (i as f64) / ((n.saturating_sub(1)).max(1) as f64);
                (f1_mid - 0.5 * f1_span) + frac * f1_span
            })
            .collect();
        let coords_f2: Vec<f64> = (0..n)
            .map(|i| {
                let frac = (i as f64) / ((n.saturating_sub(1)).max(1) as f64);
                (f2_mid - 0.5 * f2_span) + frac * f2_span
            })
            .collect();

        let half_sep_f1 = 0.5 * self.demo_separation_fwhm * self.demo_fwhm_f1;
        let half_sep_f2 = 0.5 * self.demo_separation_fwhm * self.demo_fwhm_f2;
        let true_peaks = [
            PeakModel2D {
                amplitude: self.demo_amp_1,
                center_f1: f1_mid - half_sep_f1,
                center_f2: f2_mid - half_sep_f2,
                fwhm_f1: self.demo_fwhm_f1,
                fwhm_f2: self.demo_fwhm_f2,
                eta_f1: 0.65,
                eta_f2: 0.60,
            },
            PeakModel2D {
                amplitude: self.demo_amp_2,
                center_f1: f1_mid + half_sep_f1,
                center_f2: f2_mid + half_sep_f2,
                fwhm_f1: self.demo_fwhm_f1 * 1.08,
                fwhm_f2: self.demo_fwhm_f2 * 0.94,
                eta_f1: 0.65,
                eta_f2: 0.60,
            },
        ];

        let mut roi = synthesize_roi_2d(&coords_f1, &coords_f2, self.lineshape, &true_peaks, 0.0)
            .map_err(|e| e.to_string())?;
        if self.demo_noise_sigma > 0.0 {
            let mut rng = Rng::new(self.demo_seed);
            for v in &mut roi.values {
                *v += rng.normal(0.0, self.demo_noise_sigma);
            }
        }

        let seeds = vec![
            PeakSeed2D {
                center_f1: true_peaks[0].center_f1 - 0.08 * self.demo_fwhm_f1,
                center_f2: true_peaks[0].center_f2 + 0.06 * self.demo_fwhm_f2,
                fwhm_f1: self.demo_fwhm_f1 * 0.92,
                fwhm_f2: self.demo_fwhm_f2 * 1.08,
                eta_f1: 0.5,
                eta_f2: 0.5,
            },
            PeakSeed2D {
                center_f1: true_peaks[1].center_f1 + 0.07 * self.demo_fwhm_f1,
                center_f2: true_peaks[1].center_f2 - 0.06 * self.demo_fwhm_f2,
                fwhm_f1: self.demo_fwhm_f1 * 1.05,
                fwhm_f2: self.demo_fwhm_f2 * 0.90,
                eta_f1: 0.5,
                eta_f2: 0.5,
            },
        ];
        Ok((roi, seeds))
    }

    /// Load the 1-click overlapped doublet synthetic demo and immediately execute VarPro.
    pub fn load_and_run_synthetic_demo(&mut self) -> Result<(), String> {
        self.source = DeconvDataSource::SyntheticDemo;
        self.lineshape = LineshapeKind::PseudoVoigt;
        self.use_known_sigma = true;
        self.known_noise_sigma = self.demo_noise_sigma.max(1e-4);
        self.use_center_shift_bound = false;
        let (roi, seeds) = self.build_synthetic_doublet_roi()?;
        self.seeds = seeds;
        self.last_roi = Some(roi);
        self.run(None)
    }

    /// Sync ROI bounds and initial local-maximum seed(s) from an active [`Spectrum2D`].
    pub fn sync_from_active_2d(&mut self, spec: &Spectrum2D) {
        let ny = spec.ny();
        let nx = spec.nx();
        if ny == 0 || nx == 0 {
            return;
        }
        // Locate peak maximum in the spectrum and frame a compact ROI around it.
        let mut best_ix = nx / 2;
        let mut best_iy = ny / 2;
        let mut best_val = f32::NEG_INFINITY;
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = iy * nx + ix;
                if let Some(&v) = spec.data.get(idx) {
                    if v.is_finite() && v > best_val {
                        best_val = v;
                        best_ix = ix;
                        best_iy = iy;
                    }
                }
            }
        }
        let half_y = (ny / 10).clamp(4, 12);
        let half_x = (nx / 10).clamp(4, 12);
        let y0 = best_iy.saturating_sub(half_y);
        let y1 = (best_iy + half_y).min(ny.saturating_sub(1));
        let x0 = best_ix.saturating_sub(half_x);
        let x1 = (best_ix + half_x).min(nx.saturating_sub(1));

        let p_y0 = spec.axis_y.index_to_ppm(y0 as f64);
        let p_y1 = spec.axis_y.index_to_ppm(y1 as f64);
        let p_x0 = spec.axis_x.index_to_ppm(x0 as f64);
        let p_x1 = spec.axis_x.index_to_ppm(x1 as f64);

        self.f1_min_ppm = p_y0.min(p_y1);
        self.f1_max_ppm = p_y0.max(p_y1);
        self.f2_min_ppm = p_x0.min(p_x1);
        self.f2_max_ppm = p_x0.max(p_x1);

        let peak_f1 = spec.axis_y.index_to_ppm(best_iy as f64);
        let peak_f2 = spec.axis_x.index_to_ppm(best_ix as f64);
        let est_fwhm_f1 = (spec.axis_y.step_ppm.abs() * 3.0).max(0.05);
        let est_fwhm_f2 = (spec.axis_x.step_ppm.abs() * 3.0).max(0.01);
        self.seeds = vec![PeakSeed2D {
            center_f1: peak_f1,
            center_f2: peak_f2,
            fwhm_f1: est_fwhm_f1,
            fwhm_f2: est_fwhm_f2,
            eta_f1: 0.5,
            eta_f2: 0.5,
        }];
        self.source = DeconvDataSource::ActiveSpectrum2D;
    }

    /// Run `nmr_core::deconv_nd::deconvolve_roi_2d` using the current editable ROI and seeds.
    pub fn run(&mut self, active_2d: Option<&Spectrum2D>) -> Result<(), String> {
        let roi = match self.source {
            DeconvDataSource::SyntheticDemo => {
                if let Some(existing) = self.last_roi.clone() {
                    existing
                } else {
                    let (r, _) = self.build_synthetic_doublet_roi()?;
                    r
                }
            }
            DeconvDataSource::ActiveSpectrum2D => {
                let spec = active_2d.ok_or_else(|| {
                    "No active 2D spectrum loaded; switch to Synthetic Demo or load a 2D spectrum."
                        .to_string()
                })?;
                Roi2D::from_spectrum_window(
                    spec,
                    (self.f1_min_ppm, self.f1_max_ppm),
                    (self.f2_min_ppm, self.f2_max_ppm),
                )
                .map_err(|e| e.to_string())?
            }
        };

        let cfg = self.build_config();
        self.last_roi = Some(roi.clone());
        match deconvolve_roi_2d(&roi, &self.seeds, &cfg) {
            Ok(res) => {
                let diag_str = format_deconv_diagnostic(res.diagnostic);
                self.status_message = format!(
                    "VarPro complete: {} peak(s), {} iters, status={}, chi2_nu={:.3}, CRLB={}",
                    res.peaks.len(),
                    res.iterations,
                    diag_str,
                    res.reduced_chi2,
                    if res.crlb_valid {
                        "VALID"
                    } else {
                        "SUPPRESSED"
                    }
                );
                self.error_message = None;
                self.result = Some(res);
                Ok(())
            }
            Err(err) => {
                let msg = err.to_string();
                self.error_message = Some(msg.clone());
                self.status_message = "VarPro failed closed on input validation.".to_string();
                Err(msg)
            }
        }
    }
}

fn format_deconv_diagnostic(diag: DeconvDiagnostic) -> String {
    match diag {
        DeconvDiagnostic::Converged => "Converged".to_string(),
        DeconvDiagnostic::ActiveBoundHit => "ActiveBoundHit".to_string(),
        DeconvDiagnostic::ModelMisspecified { reduced_chi2 } => {
            format!("ModelMisspecified (chi2_nu={reduced_chi2:.2})")
        }
        DeconvDiagnostic::IllConditionedCluster => "IllConditionedCluster".to_string(),
        DeconvDiagnostic::DegeneratePeakPair => "DegeneratePeakPair".to_string(),
        DeconvDiagnostic::NonConverged => "NonConverged".to_string(),
    }
}

// ============================================================================
// 3. Tab 2 State: Convex Accelerated MaxEnt NUS (`T22.13`)
// ============================================================================

/// Regularization policy selector in the MaxEnt NUS tab.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MaxEntLambdaUiMode {
    #[default]
    Fixed,
    DiscrepancyPrinciple,
}

/// Non-destructively editable state for the `MaxEnt NUS` tab.
#[derive(Clone, Debug)]
pub struct MaxEntTabState {
    pub n2: usize,
    pub n1: usize,
    pub zero_fill_factor: usize,
    pub sampling_fraction: f64,
    pub poisson_seed: u64,
    pub demo_noise_sigma: f64,
    pub quad_mode_axis2: QuadMode,
    pub quad_mode_axis1: QuadMode,
    pub lambda_ui_mode: MaxEntLambdaUiMode,
    pub lambda: f64,
    pub def: f64,
    pub discrepancy_sigma: f64,
    pub target_chi2_nu: f64,
    pub max_iter: usize,
    pub kkt_tol: f64,
    pub accelerate: bool,
    pub adaptive_restart: bool,
    /// Editable 0-based `(t2, t1)` NUS schedule text buffer (one coordinate pair per line).
    pub schedule_text: String,
    /// Last sampled mask used for the preview.
    pub last_mask: Vec<bool>,
    pub last_sampled_count: usize,
    /// Last successful `reconstruct_maxent_2d` result.
    pub result: Option<MaxEntResult<[f64; 4]>>,
    pub status_message: String,
    pub error_message: Option<String>,
}

impl Default for MaxEntTabState {
    fn default() -> Self {
        let mut st = Self {
            n2: 16,
            n1: 16,
            zero_fill_factor: 2,
            sampling_fraction: 0.35,
            poisson_seed: 0x2213_0042,
            demo_noise_sigma: 0.008,
            quad_mode_axis2: QuadMode::States,
            quad_mode_axis1: QuadMode::States,
            lambda_ui_mode: MaxEntLambdaUiMode::Fixed,
            lambda: 12.0,
            def: 0.25,
            discrepancy_sigma: 0.008,
            target_chi2_nu: 1.0,
            max_iter: 250,
            kkt_tol: 1.0e-5,
            accelerate: true,
            adaptive_restart: true,
            schedule_text: String::new(),
            last_mask: Vec::new(),
            last_sampled_count: 0,
            result: None,
            status_message:
                "Generate a Poisson-gap schedule or paste (t2, t1) coordinates and run MaxEnt."
                    .to_string(),
            error_message: None,
        };
        let _ = st.generate_poisson_gap_schedule();
        st
    }
}

impl MaxEntTabState {
    /// Generate a deterministic 2D Poisson-gap NUS schedule and populate `self.schedule_text`.
    pub fn generate_poisson_gap_schedule(&mut self) -> Result<usize, String> {
        let n2 = self.n2.clamp(8, 64);
        let n1 = self.n1.clamp(8, 64);
        self.n2 = n2;
        self.n1 = n1;
        let total = n2 * n1;
        let target = ((total as f64) * self.sampling_fraction.clamp(0.05, 1.0))
            .round()
            .clamp(1.0, total as f64) as usize;
        let mask = poisson_gap_mask_2d(n2, n1, target, 2.0, self.poisson_seed)?;
        let mut buf = format!("# 2D Poisson-gap NUS schedule ({n2}x{n1}, target={target})\n");
        let mut count = 0usize;
        for k2 in 0..n2 {
            for k1 in 0..n1 {
                if mask[k2 * n1 + k1] {
                    let _ = writeln!(buf, "{k2}\t{k1}");
                    count += 1;
                }
            }
        }
        self.schedule_text = buf;
        self.last_mask = mask;
        self.last_sampled_count = count;
        Ok(count)
    }

    /// Parse `self.schedule_text` into a validated `Vec<bool>` mask of size `n2 * n1`.
    pub fn parse_schedule_mask(&self) -> Result<(Vec<bool>, usize), String> {
        let n2 = self.n2;
        let n1 = self.n1;
        if n2 == 0 || n1 == 0 {
            return Err("MaxEnt grid dimensions n2 and n1 must both be >= 1".to_string());
        }
        let mut mask = vec![false; n2 * n1];
        let mut count = 0usize;
        for (line_idx, raw_line) in self.schedule_text.lines().enumerate() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = trimmed
                .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
                .filter(|s| !s.is_empty())
                .collect();
            if parts.len() < 2 {
                return Err(format!(
                    "schedule line {} expected 't2 t1' integer coordinates, got '{trimmed}'",
                    line_idx + 1
                ));
            }
            let k2: usize = parts[0].parse().map_err(|_| {
                format!(
                    "schedule line {} invalid t2 coordinate '{}'",
                    line_idx + 1,
                    parts[0]
                )
            })?;
            let k1: usize = parts[1].parse().map_err(|_| {
                format!(
                    "schedule line {} invalid t1 coordinate '{}'",
                    line_idx + 1,
                    parts[1]
                )
            })?;
            if k2 >= n2 || k1 >= n1 {
                return Err(format!(
                    "schedule line {} coordinate ({k2}, {k1}) out of bounds for grid {n2}x{n1}",
                    line_idx + 1
                ));
            }
            if !mask[k2 * n1 + k1] {
                mask[k2 * n1 + k1] = true;
                count += 1;
            }
        }
        if count == 0 {
            return Err("NUS schedule contains zero sampled (t2, t1) points".to_string());
        }
        Ok((mask, count))
    }

    /// Synthesize a realistic 3-peak 2D hypercomplex FID in the selected `quad_modes` encoding.
    fn synthesize_demo_fid_2d(&self, mask: &[bool]) -> Vec<[f64; 4]> {
        let n2 = self.n2;
        let n1 = self.n1;
        let peaks: [(f64, f64, f64, f64, f64); 3] = [
            (1.0, 0.22, 0.28, 0.08, 0.08),
            (0.65, 0.61, 0.44, 0.09, 0.07),
            (0.35, 0.36, 0.74, 0.10, 0.09),
        ];
        let mut rng = Rng::new(self.poisson_seed ^ 0x51A7_E500);
        let mut obs = vec![[0.0_f64; 4]; n2 * n1];
        for k2 in 0..n2 {
            let t2 = k2 as f64;
            let sgn2 = if matches!(self.quad_mode_axis2, QuadMode::StatesTppi) && k2 % 2 == 1 {
                -1.0
            } else {
                1.0
            };
            for k1 in 0..n1 {
                let idx = k2 * n1 + k1;
                if !mask[idx] {
                    continue;
                }
                let t1 = k1 as f64;
                let sgn1 = if matches!(self.quad_mode_axis1, QuadMode::StatesTppi) && k1 % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                let sign = sgn2 * sgn1;
                let mut q = [0.0_f64; 4];
                for &(amp, f2, f1, r2, r1) in &peaks {
                    let env2 = (-r2 * t2).exp();
                    let env1 = (-r1 * t1).exp();
                    let ph2 = 2.0 * PI * f2 * t2;
                    let ph1 = 2.0 * PI * f1 * t1;
                    let c2 = env2 * ph2.cos();
                    let s2 = env2 * ph2.sin();
                    let c1 = env1 * ph1.cos();
                    let s1 = env1 * ph1.sin();
                    q[0] += amp * c2 * c1;
                    q[1] += amp * c2 * s1;
                    q[2] += amp * s2 * c1;
                    q[3] += amp * s2 * s1;
                }
                if self.demo_noise_sigma > 0.0 {
                    for ch in &mut q {
                        *ch += rng.normal(0.0, self.demo_noise_sigma);
                    }
                }
                obs[idx] = [q[0] * sign, q[1] * sign, q[2] * sign, q[3] * sign];
            }
        }
        obs
    }

    /// Build [`MaxEntConfig`] from the current editable UI controls.
    pub fn build_config(&self) -> MaxEntConfig {
        let lambda_mode = match self.lambda_ui_mode {
            MaxEntLambdaUiMode::Fixed => MaxEntLambdaMode::Fixed {
                lambda: self.lambda,
            },
            MaxEntLambdaUiMode::DiscrepancyPrinciple => MaxEntLambdaMode::DiscrepancyPrinciple {
                noise_sigma: self.discrepancy_sigma,
                target_chi2_nu: self.target_chi2_nu,
                chi2_tol: 0.02,
                max_outer_iterations: 24,
            },
        };
        MaxEntConfig {
            def: self.def,
            lambda_mode,
            noise_sigma: if self.demo_noise_sigma > 0.0 {
                Some(self.demo_noise_sigma)
            } else {
                None
            },
            max_iterations: self.max_iter.clamp(1, MAX_MAXENT_ITERATIONS),
            kkt_tol: self.kkt_tol,
            accelerate: self.accelerate,
            adaptive_restart: self.adaptive_restart,
            adaptive_step_size: true,
            zero_fill_factor: self.zero_fill_factor.clamp(1, 2),
            record_objective_trace: false,
        }
    }

    /// Load the 1-click Poisson-gap 2D MaxEnt synthetic demo and execute reconstruction.
    pub fn load_and_run_synthetic_demo(&mut self) -> Result<(), String> {
        self.quad_mode_axis2 = QuadMode::States;
        self.quad_mode_axis1 = QuadMode::States;
        self.generate_poisson_gap_schedule()?;
        self.run()
    }

    /// Execute `nmr_core::maxent_nus::reconstruct_maxent_2d` using the current editable schedule
    /// and quadrature mode (surfacing fail-closed errors such as `EchoAntiecho`).
    pub fn run(&mut self) -> Result<(), String> {
        let (mask, count) = match self.parse_schedule_mask() {
            Ok(pair) => pair,
            Err(err) => {
                self.error_message = Some(err.clone());
                self.status_message = "MaxEnt schedule parse error (fail-closed).".to_string();
                return Err(err);
            }
        };
        let obs_quad = self.synthesize_demo_fid_2d(&mask);
        let cfg = self.build_config();
        let quad_modes = [self.quad_mode_axis2, self.quad_mode_axis1];
        self.last_mask = mask.clone();
        self.last_sampled_count = count;

        match reconstruct_maxent_2d(&obs_quad, self.n2, self.n1, &mask, quad_modes, &cfg) {
            Ok(res) => {
                self.status_message = format!(
                    "MaxEnt 2D finished: {} iters, KKT={:.2e}, gap={:.2e}, lambda={:.3}, converged={}",
                    res.iterations,
                    res.kkt_residual_inf,
                    res.duality_gap,
                    res.final_lambda,
                    res.converged
                );
                self.error_message = None;
                self.result = Some(res);
                Ok(())
            }
            Err(err) => {
                self.error_message = Some(err.clone());
                self.status_message = "MaxEnt 2D failed closed (see diagnostic below).".to_string();
                Err(err)
            }
        }
    }
}

// ============================================================================
// 4. Tab 3 State: ARTSY RDC Extraction (`T22.9`)
// ============================================================================

/// Sortable columns for the ARTSY RDC results table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArtsySortColumn {
    #[default]
    SeqId,
    ResName,
    RawRatio,
    AbsJPlusD,
    DNh,
    SigmaDNh,
    SnrRef,
    Quality,
}

/// Non-destructively editable state for the `ARTSY RDC` tab.
#[derive(Clone, Debug)]
pub struct ArtsyTabState {
    pub tsv_input: String,
    pub dephasing_delay_ms: f64,
    pub j_iso_default_hz: f64,
    pub j_iso_default_sigma_hz: f64,
    pub scan_ratio_att_over_ref: f64,
    pub delta_r2_s_inv: f64,
    pub sign_convention: RdcSignConvention,
    pub min_ref_snr: f64,
    pub near_saturation_threshold: f64,
    pub mc_trials: usize,
    pub sort_col: ArtsySortColumn,
    pub sort_ascending: bool,
    pub results: Vec<ArtsyResidueRdc>,
    pub status_message: String,
    pub error_message: Option<String>,
    pub last_copied_notice: Option<String>,
}

impl Default for ArtsyTabState {
    fn default() -> Self {
        let mut st = Self {
            tsv_input: String::new(),
            dephasing_delay_ms: DEFAULT_ARTSY_DEPHASING_S * 1000.0,
            j_iso_default_hz: DEFAULT_J_NH_ISO_HZ,
            j_iso_default_sigma_hz: 0.0,
            scan_ratio_att_over_ref: 1.0,
            delta_r2_s_inv: 0.0,
            sign_convention: RdcSignConvention::SignNormalized,
            min_ref_snr: 10.0,
            near_saturation_threshold: 0.90,
            mc_trials: 0,
            sort_col: ArtsySortColumn::SeqId,
            sort_ascending: true,
            results: Vec::new(),
            status_message: "Paste ARTSY TSV or click Synthetic Demo to extract 1D_NH couplings."
                .to_string(),
            error_message: None,
            last_copied_notice: None,
        };
        st.populate_synthetic_demo_tsv();
        st
    }
}

impl ArtsyTabState {
    /// Build [`ArtsyConfig`] from the current editable UI parameters.
    pub fn build_config(&self) -> ArtsyConfig {
        ArtsyConfig {
            dephasing_delay_s: self.dephasing_delay_ms * 1.0e-3,
            j_iso_default_hz: self.j_iso_default_hz,
            j_iso_default_sigma_hz: self.j_iso_default_sigma_hz,
            sign_convention: self.sign_convention,
            delta_r2_s_inv: self.delta_r2_s_inv,
            scan_ratio_att_over_ref: self.scan_ratio_att_over_ref,
            near_saturation_threshold: self.near_saturation_threshold,
            min_ref_snr: self.min_ref_snr,
            mc_trials: self.mc_trials.min(2000),
            ..ArtsyConfig::default()
        }
    }

    /// Populate `self.tsv_input` with a physically exact synthetic ARTSY amide dataset.
    pub fn populate_synthetic_demo_tsv(&mut self) {
        let cfg = ArtsyConfig::default();
        let specs: [(i32, &str, f64, f64, f64); 12] = [
            (2, "GLN", -14.2, 1200.0, 14.0),
            (3, "ILE", 11.8, 1350.0, 15.0),
            (4, "PHE", -8.5, 1180.0, 14.0),
            (5, "VAL", 16.4, 1420.0, 15.0),
            (6, "LYS", -19.0, 1080.0, 14.0),
            (7, "THR", 4.2, 1280.0, 15.0),
            (12, "THR", 21.5, 1150.0, 16.0),
            (15, "LEU", -5.6, 1310.0, 14.0),
            (23, "ILE", 9.4, 1260.0, 15.0),
            (30, "ILE", -12.8, 1210.0, 14.0),
            // Low-SNR residue (SNR ~ 6.5 < 10.0)
            (44, "ILE", 7.5, 98.0, 15.0),
            // Near-saturation residue (very large positive coupling magnitude -> |q| >= 0.90)
            (68, "HIS", -68.0, 1100.0, 14.0),
        ];

        let mut buf = String::from("seq_id\tres_name\ti_ref\ti_att\tsigma\n");
        for (seq_id, res, d_nh, i0, sigma) in specs {
            if let Ok((i_ref, i_att)) =
                artsy_abs_coupling_from_rdc(d_nh, cfg.j_iso_default_hz.abs(), cfg.sign_convention)
                    .and_then(|abs_jd| artsy_forward_intensities(abs_jd, i0, &cfg))
            {
                let _ = writeln!(buf, "{seq_id}\t{res}\t{i_ref:.3}\t{i_att:.3}\t{sigma:.3}");
            }
        }
        self.tsv_input = buf;
    }

    /// Load the 1-click synthetic demo TSV and run `nmr_core::artsy_rdc::extract_artsy_rdc`.
    pub fn load_and_run_synthetic_demo(&mut self) -> Result<(), String> {
        self.dephasing_delay_ms = DEFAULT_ARTSY_DEPHASING_S * 1000.0;
        self.j_iso_default_hz = DEFAULT_J_NH_ISO_HZ;
        self.scan_ratio_att_over_ref = 1.0;
        self.delta_r2_s_inv = 0.0;
        self.populate_synthetic_demo_tsv();
        self.run()
    }

    /// Parse `self.tsv_input` and compute `^1J_NH + ^1D_NH` & `^1D_NH +/- sigma` for every row.
    pub fn run(&mut self) -> Result<(), String> {
        let cfg = self.build_config();
        if let Err(err) = cfg.validate() {
            self.error_message = Some(err.clone());
            self.status_message = "ARTSY config validation error.".to_string();
            return Err(err);
        }

        let mut parsed = Vec::new();
        for (line_idx, raw_line) in self.tsv_input.lines().enumerate() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = trimmed
                .split(|c: char| c == '\t' || c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .collect();
            if cols.len() < 5 {
                let err = format!(
                    "line {} expected 5 columns (seq_id res_name i_ref i_att sigma), got {}",
                    line_idx + 1,
                    cols.len()
                );
                self.error_message = Some(err.clone());
                return Err(err);
            }
            // Skip header line gracefully
            if cols[0].eq_ignore_ascii_case("seq_id") || cols[0].eq_ignore_ascii_case("seq") {
                continue;
            }
            let seq_id: i32 = cols[0].parse().map_err(|_| {
                let err = format!("line {} invalid seq_id '{}'", line_idx + 1, cols[0]);
                self.error_message = Some(err.clone());
                err
            })?;
            let res_name = cols[1].trim().to_ascii_uppercase();
            let i_ref: f64 = cols[2].parse().map_err(|_| {
                let err = format!("line {} invalid i_ref '{}'", line_idx + 1, cols[2]);
                self.error_message = Some(err.clone());
                err
            })?;
            let i_att: f64 = cols[3].parse().map_err(|_| {
                let err = format!("line {} invalid i_att '{}'", line_idx + 1, cols[3]);
                self.error_message = Some(err.clone());
                err
            })?;
            let sigma_ref: f64 = cols[4].parse().map_err(|_| {
                let err = format!("line {} invalid sigma '{}'", line_idx + 1, cols[4]);
                self.error_message = Some(err.clone());
                err
            })?;
            let sigma_att: f64 = if cols.len() >= 6 {
                cols[5].parse().map_err(|_| {
                    let err = format!("line {} invalid sigma_att '{}'", line_idx + 1, cols[5]);
                    self.error_message = Some(err.clone());
                    err
                })?
            } else {
                sigma_ref
            };

            let pair = ArtsyIntensityPair {
                i_ref,
                i_att,
                sigma_ref,
                sigma_att,
            };
            let res =
                extract_artsy_rdc(seq_id, &res_name, &pair, None, None, &cfg).map_err(|e| {
                    let err = format!("line {} (residue {seq_id} {res_name}): {e}", line_idx + 1);
                    self.error_message = Some(err.clone());
                    err
                })?;
            parsed.push(res);
        }

        if parsed.is_empty() {
            let err = "ARTSY TSV input contains zero data rows.".to_string();
            self.error_message = Some(err.clone());
            return Err(err);
        }

        let n_good = parsed
            .iter()
            .filter(|r| r.quality == ArtsyQualityFlag::Good)
            .count();
        let n_ood = parsed
            .iter()
            .filter(|r| r.quality == ArtsyQualityFlag::OutOfDomainRatio)
            .count();
        self.status_message = format!(
            "Extracted {} ARTSY residues ({} Good, {} flagged/out-of-domain).",
            parsed.len(),
            n_good,
            parsed.len() - n_good
        );
        if n_ood > 0 {
            let _ = write!(
                self.status_message,
                " {n_ood} OutOfDomainRatio row(s) failed closed."
            );
        }
        self.error_message = None;
        self.results = parsed;
        self.apply_sort();
        Ok(())
    }

    /// Sort `self.results` in place according to `(self.sort_col, self.sort_ascending)`.
    pub fn apply_sort(&mut self) {
        let col = self.sort_col;
        let asc = self.sort_ascending;
        self.results.sort_by(|a, b| {
            let ord = match col {
                ArtsySortColumn::SeqId => a.seq_id.cmp(&b.seq_id),
                ArtsySortColumn::ResName => a.res_name.cmp(&b.res_name),
                ArtsySortColumn::RawRatio => a.aligned.raw_ratio.total_cmp(&b.aligned.raw_ratio),
                ArtsySortColumn::AbsJPlusD => {
                    opt_f64_cmp(a.aligned.abs_j_plus_d_hz, b.aligned.abs_j_plus_d_hz)
                }
                ArtsySortColumn::DNh => opt_f64_cmp(a.d_nh_hz, b.d_nh_hz),
                ArtsySortColumn::SigmaDNh => opt_f64_cmp(a.sigma_d_nh_hz, b.sigma_d_nh_hz),
                ArtsySortColumn::SnrRef => a.aligned.snr_ref.total_cmp(&b.aligned.snr_ref),
                ArtsySortColumn::Quality => {
                    artsy_quality_rank(a.quality).cmp(&artsy_quality_rank(b.quality))
                }
            };
            if asc {
                ord
            } else {
                ord.reverse()
            }
        });
    }

    /// Format the current sorted ARTSY RDC table as a clean tab-separated string for clipboard export.
    pub fn export_tsv(&self) -> String {
        let mut out = String::from(
            "seq_id\tres_name\ti_att_over_i_ref\tabs_1J_plus_1D_hz\t1D_NH_hz\tsigma_1D_NH_hz\tsnr_ref\tquality\n",
        );
        for r in &self.results {
            let abs_jd = r
                .aligned
                .abs_j_plus_d_hz
                .map_or_else(|| "NA".to_string(), |v| format!("{v:.3}"));
            let d_nh = r
                .d_nh_hz
                .map_or_else(|| "NA".to_string(), |v| format!("{v:.3}"));
            let sig = r
                .sigma_d_nh_hz
                .map_or_else(|| "NA".to_string(), |v| format!("{v:.3}"));
            let _ = writeln!(
                out,
                "{}\t{}\t{:.5}\t{}\t{}\t{}\t{:.1}\t{}",
                r.seq_id,
                r.res_name,
                r.aligned.raw_ratio,
                abs_jd,
                d_nh,
                sig,
                r.aligned.snr_ref,
                artsy_quality_label(r.quality)
            );
        }
        out
    }
}

fn opt_f64_cmp(a: Option<f64>, b: Option<f64>) -> Ordering {
    match (a, b) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

const fn artsy_quality_rank(q: ArtsyQualityFlag) -> u8 {
    match q {
        ArtsyQualityFlag::Good => 0,
        ArtsyQualityFlag::NearSaturation => 1,
        ArtsyQualityFlag::LowSnr => 2,
        ArtsyQualityFlag::OutOfDomainRatio => 3,
    }
}

const fn artsy_quality_label(q: ArtsyQualityFlag) -> &'static str {
    match q {
        ArtsyQualityFlag::Good => "Good",
        ArtsyQualityFlag::NearSaturation => "NearSaturation",
        ArtsyQualityFlag::LowSnr => "LowSnr",
        ArtsyQualityFlag::OutOfDomainRatio => "OutOfDomainRatio",
    }
}

// ============================================================================
// 5. Tab 4 State: 2D PCS Assignment Transfer & Curie-Spin CCR (`T22.10`)
// ============================================================================

/// Execution strategy for the `2D PCS & Curie CCR` tab.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PcsCcrMode {
    /// Iterative EM Kuhn-Munkres + Sinkhorn-Knopp + Huber-IRLS `DeltaChi` refinement and joint
    /// Curie-spin `Gamma_CCR` fit (when 3D coordinates are provided).
    #[default]
    IterativeEmAndJointCcr,
    /// Single-pass Kuhn-Munkres + Sinkhorn-Knopp 2D cone/diagonal matching (with optional tensor).
    SinglePassTransfer,
}

/// Parsed diamagnetic + optional CCR input row.
#[derive(Clone, Debug)]
struct ParsedPcsCcrRow {
    dia: DiamagneticPeak2D,
    ccr: Option<CcrObservation>,
}

/// Non-destructively editable state for the `2D PCS & Curie CCR` tab.
#[derive(Clone, Debug)]
pub struct PcsCcrTabState {
    pub mode: PcsCcrMode,
    /// Editable diamagnetic peak TSV:
    /// `seq_id label h_ppm x_ppm [hx hy hz xx xy xz [ccr_s_inv ccr_sigma]]`
    pub dia_tsv: String,
    /// Editable paramagnetic peak TSV: `peak_id h_ppm x_ppm [intensity]`
    pub para_tsv: String,
    /// Metal ion position in Angstroms.
    pub metal_pos: Vec3,
    /// Single-pass & EM matching parameters.
    pub sigma_h_ppm: f64,
    pub sigma_x_ppm: f64,
    pub model_sigma_ppm: f64,
    pub r_bleach_angstrom: f64,
    pub mahalanobis_gate_sigma: f64,
    pub sinkhorn_temperature: f64,
    /// Optional rough initial `DeltaChi` tensor guess for EM / cone transfer.
    pub use_initial_tensor: bool,
    pub init_dchi_ax: f64,
    pub init_dchi_rh: f64,
    pub init_euler_deg: [f64; 3],
    /// Mandatory Curie-spin CCR physics parameters (explicitly editable, never hidden).
    pub tau_m_ns: f64,
    pub proton_mhz: f64,
    pub temperature_k: f64,
    pub spin_pair: CcrSpinPair,
    pub tensor_model: CcrTensorModel,
    pub fit_pcs_offset: bool,
    /// Last parsed peaks (for 2D shift-vector plot).
    pub last_dia_peaks: Vec<DiamagneticPeak2D>,
    pub last_para_peaks: Vec<ParamagneticPeak2D>,
    /// Results.
    pub transfer_result: Option<PcsTransferResult>,
    pub em_result: Option<IterativePcsTransferResult>,
    pub joint_fit: Option<JointPcsCcrFitResult>,
    pub ccr_only_fit: Option<CcrFitResult>,
    pub status_message: String,
    pub error_message: Option<String>,
    pub last_copied_notice: Option<String>,
}

impl Default for PcsCcrTabState {
    fn default() -> Self {
        let mut st = Self {
            mode: PcsCcrMode::IterativeEmAndJointCcr,
            dia_tsv: String::new(),
            para_tsv: String::new(),
            metal_pos: [0.0, 0.0, 0.0],
            sigma_h_ppm: 0.02,
            sigma_x_ppm: 0.10,
            model_sigma_ppm: 0.04,
            r_bleach_angstrom: 9.0,
            mahalanobis_gate_sigma: 5.5,
            sinkhorn_temperature: 1.0,
            use_initial_tensor: true,
            init_dchi_ax: 10.5,
            init_dchi_rh: 2.5,
            init_euler_deg: [18.0, 32.0, -12.0],
            tau_m_ns: 8.5,
            proton_mhz: 800.0,
            temperature_k: 298.0,
            spin_pair: CcrSpinPair::N15H1,
            tensor_model: CcrTensorModel::FullChiTensor,
            fit_pcs_offset: false,
            last_dia_peaks: Vec::new(),
            last_para_peaks: Vec::new(),
            transfer_result: None,
            em_result: None,
            joint_fit: None,
            ccr_only_fit: None,
            status_message: "Click Synthetic Demo or paste diamagnetic/paramagnetic peak TSVs."
                .to_string(),
            error_message: None,
            last_copied_notice: None,
        };
        st.populate_synthetic_demo_tsvs();
        st
    }
}

impl PcsCcrTabState {
    /// Populate editable diamagnetic & paramagnetic peak TSVs from an exact lanthanide
    /// (`Tb3+`-like `DeltaChi` + `chi_iso`) synthetic 3D amide shell.
    pub fn populate_synthetic_demo_tsvs(&mut self) {
        self.metal_pos = [0.0, 0.0, 0.0];
        self.tau_m_ns = 8.5;
        self.proton_mhz = 800.0;
        self.temperature_k = 298.0;
        self.spin_pair = CcrSpinPair::N15H1;
        self.tensor_model = CcrTensorModel::FullChiTensor;
        self.r_bleach_angstrom = 9.0;

        // Ground-truth lanthanide tensor (10^-32 m^3) and perturbed initial guess for EM
        let true_euler = [
            22.0_f64.to_radians(),
            35.0_f64.to_radians(),
            -15.0_f64.to_radians(),
        ];
        let true_tensor =
            match delta_chi_tensor_from_spherical(self.metal_pos, 12.0, 3.2, true_euler) {
                Ok(t) => t,
                Err(_) => return,
            };
        let true_chi_iso = chi_iso_from_mu_eff_bohr(9.5, self.temperature_k).unwrap_or(35.0);
        let params = match CcrPhysicsParams::new(
            self.tau_m_ns,
            self.proton_mhz,
            self.temperature_k,
            self.spin_pair,
        ) {
            Ok(p) => p,
            Err(_) => return,
        };

        // Rough initial tensor (~15% offset) to showcase real EM convergence
        self.use_initial_tensor = true;
        self.init_dchi_ax = 10.2;
        self.init_dchi_rh = 2.4;
        self.init_euler_deg = [16.0, 29.0, -9.0];

        // 14 residues: residue 10 is inside the PRE bleaching sphere (r ~ 7.2 A < 9.0 A),
        // residues 11..23 span well-conditioned 3D directions at r = 13..22 A.
        let shells: [(i32, &str, f64, f64, Vec3, Vec3); 14] = [
            (10, "ALA10", 8.12, 121.4, [4.2, 4.1, 4.1], [4.7, 4.7, 4.7]),
            (11, "GLY11", 8.45, 108.8, [13.5, 2.0, 1.5], [14.3, 2.4, 2.0]),
            (
                12,
                "ILE12",
                7.82,
                118.6,
                [-2.5, 14.2, 3.1],
                [-2.1, 15.1, 3.5],
            ),
            (
                13,
                "VAL13",
                8.95,
                124.2,
                [1.8, -3.2, 15.0],
                [2.4, -3.5, 15.8],
            ),
            (
                14,
                "LEU14",
                7.42,
                122.0,
                [-14.0, -3.8, 2.4],
                [-14.8, -4.2, 2.8],
            ),
            (15, "PHE15", 8.68, 116.5, [9.2, 10.4, 4.5], [9.9, 11.0, 4.9]),
            (
                16,
                "THR16",
                8.21,
                113.4,
                [-8.8, 9.6, -7.2],
                [-9.4, 10.1, -7.8],
            ),
            (
                17,
                "SER17",
                7.64,
                115.8,
                [6.5, -11.8, 7.8],
                [7.0, -12.6, 8.2],
            ),
            (
                18,
                "ASP18",
                8.34,
                120.2,
                [11.0, -6.2, -9.4],
                [11.7, -6.7, -9.9],
            ),
            (
                19,
                "GLU19",
                7.96,
                123.5,
                [-10.2, -9.5, -8.4],
                [-10.8, -10.1, -8.9],
            ),
            (
                20,
                "LYS20",
                8.76,
                125.1,
                [3.4, 12.6, -10.5],
                [3.9, 13.3, -11.0],
            ),
            (
                21,
                "ARG21",
                7.28,
                119.3,
                [-12.4, 5.2, 11.8],
                [-13.1, 5.6, 12.4],
            ),
            (
                22,
                "ASN22",
                8.54,
                117.6,
                [15.6, 7.4, -5.8],
                [16.3, 8.0, -6.2],
            ),
            (
                23,
                "GLN23",
                8.05,
                114.2,
                [-5.6, -14.8, 9.5],
                [-6.1, -15.5, 10.1],
            ),
        ];

        let mut dia_buf = String::from(
            "seq_id\tlabel\th_ppm\tx_ppm\thx\thy\thz\txx\txy\txz\tccr_s_inv\tccr_sigma\n",
        );
        let mut para_rows: Vec<(usize, f64, f64, f64)> = Vec::new();
        let mut rng = Rng::new(0x2210_0099);

        for (idx, (seq_id, label, h_dia, x_dia, ch, cx)) in shells.iter().enumerate() {
            let ccr_obs = predict_curie_dd_ccr_s_inv(
                *cx,
                *ch,
                self.metal_pos,
                true_chi_iso,
                Some(&true_tensor),
                &params,
                1.0,
            )
            .ok();

            if let Some(ccr_val) = ccr_obs {
                let ccr_noisy = ccr_val + rng.normal(0.0, 0.08);
                let _ = writeln!(
                    dia_buf,
                    "{seq_id}\t{label}\t{h_dia:.3}\t{x_dia:.3}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{ccr_noisy:.4}\t0.10",
                    ch[0], ch[1], ch[2], cx[0], cx[1], cx[2]
                );
            } else {
                let _ = writeln!(
                    dia_buf,
                    "{seq_id}\t{label}\t{h_dia:.3}\t{x_dia:.3}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
                    ch[0], ch[1], ch[2], cx[0], cx[1], cx[2]
                );
            }

            // Skip bleached residue 10 in paramagnetic spectrum to demonstrate PRE bleaching!
            if *seq_id == 10 {
                continue;
            }
            let pcs_h = predict_pcs_from_tensor(*ch, &true_tensor);
            let pcs_x = predict_pcs_from_tensor(*cx, &true_tensor);
            let h_para = h_dia + pcs_h + rng.normal(0.0, 0.004);
            let x_para = x_dia + pcs_x + rng.normal(0.0, 0.018);
            para_rows.push((101 + idx, h_para, x_para, 1.0));
        }

        // Permute paramagnetic peak order deterministically so matching is genuinely unassigned
        para_rows.reverse();
        let mut para_buf = String::from("peak_id\th_ppm\tx_ppm\tintensity\n");
        for (pid, hp, xp, inten) in para_rows {
            let _ = writeln!(para_buf, "{pid}\t{hp:.4}\t{xp:.4}\t{inten:.2}");
        }

        self.dia_tsv = dia_buf;
        self.para_tsv = para_buf;
    }

    fn parse_dia_tsv(&self) -> Result<Vec<ParsedPcsCcrRow>, String> {
        let mut rows = Vec::new();
        for (line_idx, raw_line) in self.dia_tsv.lines().enumerate() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = trimmed
                .split(|c: char| c == '\t' || c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .collect();
            if cols.len() < 4 {
                return Err(format!(
                    "diamagnetic line {} expected >= 4 columns (seq_id label h_ppm x_ppm), got {}",
                    line_idx + 1,
                    cols.len()
                ));
            }
            if cols[0].eq_ignore_ascii_case("seq_id") || cols[0].eq_ignore_ascii_case("seq") {
                continue;
            }
            let seq_id: i32 = cols[0].parse().map_err(|_| {
                format!(
                    "diamagnetic line {} invalid seq_id '{}'",
                    line_idx + 1,
                    cols[0]
                )
            })?;
            let label = cols[1].to_string();
            let h_ppm: f64 = cols[2].parse().map_err(|_| {
                format!(
                    "diamagnetic line {} invalid h_ppm '{}'",
                    line_idx + 1,
                    cols[2]
                )
            })?;
            let x_ppm: f64 = cols[3].parse().map_err(|_| {
                format!(
                    "diamagnetic line {} invalid x_ppm '{}'",
                    line_idx + 1,
                    cols[3]
                )
            })?;

            let (coord_h, coord_x) = if cols.len() >= 10 {
                let hx = parse_f64_col(cols[4], "hx", line_idx)?;
                let hy = parse_f64_col(cols[5], "hy", line_idx)?;
                let hz = parse_f64_col(cols[6], "hz", line_idx)?;
                let xx = parse_f64_col(cols[7], "xx", line_idx)?;
                let xy = parse_f64_col(cols[8], "xy", line_idx)?;
                let xz = parse_f64_col(cols[9], "xz", line_idx)?;
                (Some([hx, hy, hz]), Some([xx, xy, xz]))
            } else if cols.len() >= 7 {
                let hx = parse_f64_col(cols[4], "hx", line_idx)?;
                let hy = parse_f64_col(cols[5], "hy", line_idx)?;
                let hz = parse_f64_col(cols[6], "hz", line_idx)?;
                (Some([hx, hy, hz]), None)
            } else {
                (None, None)
            };

            let ccr = if cols.len() >= 12 {
                if let (Some(ch), Some(cx)) = (coord_h, coord_x) {
                    let rate = parse_f64_col(cols[10], "ccr_s_inv", line_idx)?;
                    let sig = parse_f64_col(cols[11], "ccr_sigma", line_idx)?;
                    Some(CcrObservation::new(seq_id, cx, ch, rate, sig))
                } else {
                    None
                }
            } else {
                None
            };

            rows.push(ParsedPcsCcrRow {
                dia: DiamagneticPeak2D {
                    seq_id,
                    label,
                    h_ppm,
                    x_ppm,
                    coord_h_angstrom: coord_h,
                    coord_x_angstrom: coord_x,
                },
                ccr,
            });
        }
        if rows.is_empty() {
            return Err("Diamagnetic peak TSV contains zero data rows.".to_string());
        }
        Ok(rows)
    }

    fn parse_para_tsv(&self) -> Result<Vec<ParamagneticPeak2D>, String> {
        let mut peaks = Vec::new();
        for (line_idx, raw_line) in self.para_tsv.lines().enumerate() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = trimmed
                .split(|c: char| c == '\t' || c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty())
                .collect();
            if cols.len() < 3 {
                return Err(format!(
                    "paramagnetic line {} expected >= 3 columns (peak_id h_ppm x_ppm), got {}",
                    line_idx + 1,
                    cols.len()
                ));
            }
            if cols[0].eq_ignore_ascii_case("peak_id") || cols[0].eq_ignore_ascii_case("id") {
                continue;
            }
            let peak_id: usize = cols[0].parse().map_err(|_| {
                format!(
                    "paramagnetic line {} invalid peak_id '{}'",
                    line_idx + 1,
                    cols[0]
                )
            })?;
            let h_ppm = parse_f64_col(cols[1], "h_ppm", line_idx)?;
            let x_ppm = parse_f64_col(cols[2], "x_ppm", line_idx)?;
            let intensity = if cols.len() >= 4 {
                Some(parse_f64_col(cols[3], "intensity", line_idx)?)
            } else {
                None
            };
            peaks.push(ParamagneticPeak2D {
                peak_id,
                h_ppm,
                x_ppm,
                intensity,
            });
        }
        Ok(peaks)
    }

    /// Build [`PcsTransferConfig`] from editable UI fields.
    pub fn build_transfer_config(&self) -> PcsTransferConfig {
        PcsTransferConfig {
            sigma_h_ppm: self.sigma_h_ppm,
            sigma_x_ppm: self.sigma_x_ppm,
            model_sigma_ppm: self.model_sigma_ppm,
            r_bleach_angstrom: self.r_bleach_angstrom,
            mahalanobis_gate_sigma: self.mahalanobis_gate_sigma,
            sinkhorn_temperature: self.sinkhorn_temperature,
            ..PcsTransferConfig::default()
        }
    }

    /// Load the 1-click synthetic demo and run PCS transfer + joint Curie CCR tensor fitting.
    pub fn load_and_run_synthetic_demo(&mut self) -> Result<(), String> {
        self.mode = PcsCcrMode::IterativeEmAndJointCcr;
        self.populate_synthetic_demo_tsvs();
        self.run()
    }

    /// Execute Kuhn-Munkres / Sinkhorn PCS transfer and (when 3D coordinates + CCR rates are
    /// present) iterative EM `DeltaChi` and joint `PCS + Gamma_CCR` susceptibility-tensor fitting.
    pub fn run(&mut self) -> Result<(), String> {
        let parsed_rows = match self.parse_dia_tsv() {
            Ok(r) => r,
            Err(err) => {
                self.error_message = Some(err.clone());
                return Err(err);
            }
        };
        let para_peaks = match self.parse_para_tsv() {
            Ok(p) => p,
            Err(err) => {
                self.error_message = Some(err.clone());
                return Err(err);
            }
        };

        let dia_peaks: Vec<DiamagneticPeak2D> = parsed_rows.iter().map(|r| r.dia.clone()).collect();
        let ccr_obs: Vec<CcrObservation> = parsed_rows.iter().filter_map(|r| r.ccr).collect();
        self.last_dia_peaks = dia_peaks.clone();
        self.last_para_peaks = para_peaks.clone();

        let transfer_cfg = self.build_transfer_config();
        let init_tensor = if self.use_initial_tensor {
            let euler_rad = [
                self.init_euler_deg[0].to_radians(),
                self.init_euler_deg[1].to_radians(),
                self.init_euler_deg[2].to_radians(),
            ];
            Some(
                delta_chi_tensor_from_spherical(
                    self.metal_pos,
                    self.init_dchi_ax,
                    self.init_dchi_rh,
                    euler_rad,
                )
                .inspect_err(|e| {
                    self.error_message = Some(e.clone());
                })?,
            )
        } else {
            None
        };

        let n_with_coords = dia_peaks
            .iter()
            .filter(|d| d.coord_h_angstrom.is_some() || d.coord_x_angstrom.is_some())
            .count();

        let (active_transfer, fitted_em) = match self.mode {
            PcsCcrMode::IterativeEmAndJointCcr if n_with_coords >= 5 && init_tensor.is_some() => {
                let em_cfg = IterativePcsTransferConfig {
                    transfer_cfg,
                    fit_offset: self.fit_pcs_offset,
                    ..IterativePcsTransferConfig::default()
                };
                match iterative_pcs_assignment_transfer(
                    &dia_peaks,
                    &para_peaks,
                    self.metal_pos,
                    init_tensor.as_ref(),
                    &[],
                    &em_cfg,
                ) {
                    Ok(em_res) => (em_res.transfer.clone(), Some(em_res)),
                    Err(err) => {
                        self.error_message = Some(err.clone());
                        return Err(err);
                    }
                }
            }
            _ => {
                let res = match_pcs_assignment_transfer(
                    &dia_peaks,
                    &para_peaks,
                    init_tensor.as_ref(),
                    Some(self.metal_pos),
                    &transfer_cfg,
                )
                .inspect_err(|e| {
                    self.error_message = Some(e.clone());
                })?;
                (res, None)
            }
        };

        // Collect matched 3D PCS observations for joint PCS + Curie-spin Gamma_CCR fitting
        let mut pcs_obs: Vec<PcsObservation> = Vec::new();
        for tr in &active_transfer.transfers {
            if let PcsMatchStatus::Matched { .. } = tr.status {
                if let Some(dia) = dia_peaks.get(tr.dia_index) {
                    if let (Some(ch), Some(dh)) = (dia.coord_h_angstrom, tr.observed_pcs_h_ppm) {
                        pcs_obs.push(
                            PcsObservation::new(dia.seq_id, ch, dh)
                                .with_uncertainty(self.sigma_h_ppm.max(1e-4)),
                        );
                    }
                    if let (Some(cx), Some(dx)) = (dia.coord_x_angstrom, tr.observed_pcs_x_ppm) {
                        pcs_obs.push(
                            PcsObservation::new(dia.seq_id, cx, dx)
                                .with_uncertainty(self.sigma_x_ppm.max(1e-4)),
                        );
                    }
                }
            }
        }

        let mut joint_fit = None;
        let mut ccr_only_fit = None;
        if !ccr_obs.is_empty() {
            let params = CcrPhysicsParams::new(
                self.tau_m_ns,
                self.proton_mhz,
                self.temperature_k,
                self.spin_pair,
            )
            .inspect_err(|e| {
                self.error_message = Some(e.clone());
            })?;
            let min_pcs = if self.fit_pcs_offset { 7 } else { 6 };
            if pcs_obs.len() >= min_pcs {
                let opts = JointPcsCcrFitOptions {
                    tensor_model: self.tensor_model,
                    fit_pcs_offset: self.fit_pcs_offset,
                    huber_k: Some(2.0),
                    max_irls_iters: 25,
                    cv_folds: 5,
                };
                match fit_joint_pcs_ccr(&pcs_obs, &ccr_obs, self.metal_pos, &params, &opts) {
                    Ok(jf) => joint_fit = Some(jf),
                    Err(err) => {
                        self.error_message = Some(err.clone());
                        return Err(err);
                    }
                }
            } else {
                let fixed_dc = fitted_em
                    .as_ref()
                    .map(|e| &e.fitted_tensor)
                    .or(init_tensor.as_ref());
                match fit_curie_ccr_chi_iso(&ccr_obs, self.metal_pos, fixed_dc, &params, 5) {
                    Ok(cf) => ccr_only_fit = Some(cf),
                    Err(err) => {
                        self.error_message = Some(err.clone());
                        return Err(err);
                    }
                }
            }
        }

        let mut msg = format!(
            "Matched {}/{} residues ({} bleached, {} unmatched).",
            active_transfer.n_matched,
            active_transfer.transfers.len(),
            active_transfer.n_unmatched_bleached,
            active_transfer.n_unmatched_no_peak
        );
        if let Some(jf) = &joint_fit {
            let _ = write!(
                msg,
                " Joint PCS+CCR: dchi_ax={:.2}, dchi_rh={:.2}, chi_iso={:.2} (Q_pcs={:.3}, Q_ccr={:.3}).",
                jf.delta_chi.dchi_ax,
                jf.delta_chi.dchi_rh,
                jf.chi_iso_1e32_m3,
                jf.pcs_q_factor,
                jf.ccr_q_factor
            );
        } else if let Some(em) = &fitted_em {
            let _ = write!(
                msg,
                " EM DeltaChi ({} iters): dchi_ax={:.2}, dchi_rh={:.2}, Q={:.3}.",
                em.iterations,
                em.fitted_tensor.dchi_ax,
                em.fitted_tensor.dchi_rh,
                em.fitted_tensor.q_factor
            );
        }

        self.status_message = msg;
        self.error_message = None;
        self.transfer_result = Some(active_transfer);
        self.em_result = fitted_em;
        self.joint_fit = joint_fit;
        self.ccr_only_fit = ccr_only_fit;
        Ok(())
    }

    /// Export matched PCS assignment-transfer rows as TSV.
    pub fn export_matched_tsv(&self) -> String {
        let mut out = String::from(
            "seq_id\tlabel\tstatus\tpara_peak_id\tr_metal_A\tobs_pcs_h_ppm\tobs_pcs_x_ppm\tdiag_res_ppm\tposterior\tmargin\n",
        );
        if let Some(res) = &self.transfer_result {
            for tr in &res.transfers {
                let (st_str, pid_str) = match tr.status {
                    PcsMatchStatus::Matched { para_peak_id, .. } => {
                        ("Matched", para_peak_id.to_string())
                    }
                    PcsMatchStatus::UnmatchedBleached => ("UnmatchedBleached", "NA".to_string()),
                    PcsMatchStatus::UnmatchedNoPeak => ("UnmatchedNoPeak", "NA".to_string()),
                };
                let rm = tr
                    .metal_distance_angstrom
                    .map_or_else(|| "NA".to_string(), |v| format!("{v:.2}"));
                let dh = tr
                    .observed_pcs_h_ppm
                    .map_or_else(|| "NA".to_string(), |v| format!("{v:.4}"));
                let dx = tr
                    .observed_pcs_x_ppm
                    .map_or_else(|| "NA".to_string(), |v| format!("{v:.4}"));
                let dr = tr
                    .diagonal_residual_ppm
                    .map_or_else(|| "NA".to_string(), |v| format!("{v:.4}"));
                let _ = writeln!(
                    out,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}",
                    tr.seq_id,
                    tr.label,
                    st_str,
                    pid_str,
                    rm,
                    dh,
                    dx,
                    dr,
                    tr.posterior_confidence,
                    tr.posterior_margin
                );
            }
        }
        out
    }

    /// Active fitted [`DeltaChiTensor`] from either the joint PCS+CCR fit or EM PCS transfer.
    pub fn active_tensor(&self) -> Option<&DeltaChiTensor> {
        self.joint_fit
            .as_ref()
            .map(|j| &j.delta_chi)
            .or_else(|| self.em_result.as_ref().map(|e| &e.fitted_tensor))
    }
}

fn parse_f64_col(token: &str, name: &str, line_idx: usize) -> Result<f64, String> {
    let val: f64 = token
        .parse()
        .map_err(|_| format!("line {} invalid {name} '{token}'", line_idx + 1))?;
    if !val.is_finite() {
        return Err(format!("line {} non-finite {name} '{token}'", line_idx + 1));
    }
    Ok(val)
}

// ============================================================================
// 6. Main Panel Struct & Public Entry Points
// ============================================================================

/// Tabbed GUI panel wiring `T22.1` (2D/3D VarPro Deconvolution), `T22.13` (Convex MaxEnt NUS),
/// `T22.9` (ARTSY RDC), and `T22.10` (2D PCS Transfer & Curie CCR).
#[derive(Clone, Debug)]
pub struct P22ToolsPanel {
    /// Whether the floating window is currently open.
    pub open: bool,
    /// Currently selected folder tab.
    pub active_tab: P22ToolTab,
    /// Tab 1: Coupled 2D/3D VarPro deconvolution state.
    pub deconv: DeconvTabState,
    /// Tab 2: Convex accelerated 2D MaxEnt NUS state.
    pub maxent: MaxEntTabState,
    /// Tab 3: Intensity-based ARTSY RDC extraction state.
    pub artsy: ArtsyTabState,
    /// Tab 4: Unassigned 2D PCS assignment transfer & Curie CCR state.
    pub pcs_ccr: PcsCcrTabState,
}

impl Default for P22ToolsPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl P22ToolsPanel {
    /// Construct a new [`P22ToolsPanel`] with sensible defaults across all 4 tabs.
    pub fn new() -> Self {
        Self {
            open: false,
            active_tab: P22ToolTab::Deconv2d3d,
            deconv: DeconvTabState::default(),
            maxent: MaxEntTabState::default(),
            artsy: ArtsyTabState::default(),
            pcs_ccr: PcsCcrTabState::default(),
        }
    }

    /// Open the panel window and switch directly to `tab`.
    pub fn open_tab(&mut self, tab: P22ToolTab) {
        self.open = true;
        self.active_tab = tab;
    }

    /// Render the panel inside a resizable floating `egui::Window` when `self.open` is `true`.
    pub fn show_window(
        &mut self,
        ctx: &egui::Context,
        active_2d: Option<&Spectrum2D>,
        paper_white: bool,
    ) {
        if !self.open {
            return;
        }
        let mut keep_open = self.open;
        egui::Window::new("Processing & Structural NMR Tools")
            .open(&mut keep_open)
            .collapsible(true)
            .resizable(true)
            .default_width(820.0)
            .default_height(640.0)
            .min_width(540.0)
            .min_height(420.0)
            .show(ctx, |ui| {
                self.show_contents(ui, active_2d, paper_white);
            });
        self.open = keep_open;
    }

    /// Render the 4 connected folder-style tabs and the active tool workspace inside any host `Ui`.
    pub fn show_contents(
        &mut self,
        ui: &mut egui::Ui,
        active_2d: Option<&Spectrum2D>,
        paper_white: bool,
    ) {
        let theme = P22ToolsTheme::for_mode(paper_white);

        ui.label(
            egui::RichText::new("Processing & Structural NMR Tools")
                .size(TYPE_HEADER_PX)
                .strong()
                .color(theme.text_primary),
        );
        ui.add_space(4.0);

        // Connected folder-style tab strip (`vnmrj_rail::draw_folder_tab`)
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            for tab in P22ToolTab::ALL {
                if draw_folder_tab(ui, tab.label(), self.active_tab == tab).clicked() {
                    self.active_tab = tab;
                }
            }
        });

        // Joined content frame directly beneath the active folder tab
        egui::Frame::NONE
            .fill(theme.pane_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .inner_margin(egui::Margin::symmetric(10, 8))
            .show(ui, |ui| {
                // Crisp 1-line "Input -> Output" user-guidance header at the top of every tab
                egui::Frame::NONE
                    .fill(theme.guide_bg)
                    .stroke(egui::Stroke::new(1.0, theme.border))
                    .corner_radius(4.0)
                    .inner_margin(egui::Margin::symmetric(8, 5))
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(self.active_tab.guidance_header())
                                .size(TYPE_CAPTION_PX)
                                .color(theme.text_primary),
                        );
                    });

                ui.add_space(6.0);

                egui::ScrollArea::vertical()
                    .id_salt(("p22_tools_scroll", self.active_tab))
                    .auto_shrink([false, false])
                    .show(ui, |ui| match self.active_tab {
                        P22ToolTab::Deconv2d3d => {
                            show_deconv_tab(ui, &mut self.deconv, active_2d, &theme);
                        }
                        P22ToolTab::MaxEntNus => {
                            show_maxent_tab(ui, &mut self.maxent, &theme);
                        }
                        P22ToolTab::ArtsyRdc => {
                            show_artsy_tab(ui, &mut self.artsy, &theme);
                        }
                        P22ToolTab::PcsCurieCcr => {
                            show_pcs_ccr_tab(ui, &mut self.pcs_ccr, &theme);
                        }
                    });
            });
    }
}

// ============================================================================
// 7. Tab 1 UI: Coupled 2D/3D VarPro Deconvolution
// ============================================================================

fn show_deconv_tab(
    ui: &mut egui::Ui,
    st: &mut DeconvTabState,
    active_2d: Option<&Spectrum2D>,
    theme: &P22ToolsTheme,
) {
    // Action bar + data source switcher
    ui.horizontal_wrapped(|ui| {
        if ui
            .button(
                egui::RichText::new("Synthetic Demo")
                    .size(TYPE_BODY_PX)
                    .strong(),
            )
            .on_hover_text("Load overlapped 2D doublet ROI + peak seeds and run VarPro")
            .clicked()
        {
            let _ = st.load_and_run_synthetic_demo();
        }
        if ui
            .button(egui::RichText::new("Run 2D VarPro Deconvolution").size(TYPE_BODY_PX))
            .clicked()
        {
            let _ = st.run(active_2d);
        }
        ui.separator();
        ui.radio_value(
            &mut st.source,
            DeconvDataSource::SyntheticDemo,
            egui::RichText::new("Synthetic Doublet").size(TYPE_BODY_PX),
        );
        let has_2d = active_2d.is_some();
        ui.add_enabled_ui(has_2d, |ui| {
            ui.radio_value(
                &mut st.source,
                DeconvDataSource::ActiveSpectrum2D,
                egui::RichText::new("Active 2D Spectrum ROI").size(TYPE_BODY_PX),
            );
            if ui
                .button(egui::RichText::new("Sync ROI from Active 2D").size(TYPE_CAPTION_PX))
                .clicked()
            {
                if let Some(spec) = active_2d {
                    st.sync_from_active_2d(spec);
                }
            }
        });
    });

    ui.add_space(4.0);

    // Editable ROI / Demo & Solver Parameters card (always non-destructively editable)
    egui::Frame::NONE
        .fill(theme.card_bg)
        .stroke(egui::Stroke::new(1.0, theme.border))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("ROI & VarPro Configuration (editable at any time)")
                    .size(TYPE_BODY_PX)
                    .strong()
                    .color(theme.text_primary),
            );
            ui.add_space(3.0);

            match st.source {
                DeconvDataSource::ActiveSpectrum2D => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new("F1 ppm:").size(TYPE_CAPTION_PX));
                        ui.add(
                            egui::DragValue::new(&mut st.f1_min_ppm)
                                .speed(0.05)
                                .prefix("min "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut st.f1_max_ppm)
                                .speed(0.05)
                                .prefix("max "),
                        );
                        ui.separator();
                        ui.label(egui::RichText::new("F2 ppm:").size(TYPE_CAPTION_PX));
                        ui.add(
                            egui::DragValue::new(&mut st.f2_min_ppm)
                                .speed(0.01)
                                .prefix("min "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut st.f2_max_ppm)
                                .speed(0.01)
                                .prefix("max "),
                        );
                    });
                }
                DeconvDataSource::SyntheticDemo => {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new("Sep (FWHM):").size(TYPE_CAPTION_PX));
                        let mut regenerate = ui
                            .add(
                                egui::DragValue::new(&mut st.demo_separation_fwhm)
                                    .range(0.15..=3.0)
                                    .speed(0.02),
                            )
                            .changed();
                        ui.label(egui::RichText::new("Noise sigma:").size(TYPE_CAPTION_PX));
                        regenerate |= ui
                            .add(
                                egui::DragValue::new(&mut st.demo_noise_sigma)
                                    .range(0.0..=0.25)
                                    .speed(0.001)
                                    .max_decimals(4),
                            )
                            .changed();
                        ui.label(egui::RichText::new("Grid:").size(TYPE_CAPTION_PX));
                        regenerate |= ui
                            .add(egui::DragValue::new(&mut st.demo_grid_n).range(11..=41))
                            .changed();
                        if ui
                            .button(egui::RichText::new("Rebuild Demo Patch").size(TYPE_CAPTION_PX))
                            .clicked()
                            || regenerate
                        {
                            if let Ok((roi, _)) = st.build_synthetic_doublet_roi() {
                                st.last_roi = Some(roi);
                            }
                        }
                    });
                }
            }

            ui.add_space(3.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Lineshape:").size(TYPE_CAPTION_PX));
                egui::ComboBox::from_id_salt("p22_deconv_lineshape")
                    .selected_text(format!("{:?}", st.lineshape))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut st.lineshape,
                            LineshapeKind::PseudoVoigt,
                            "PseudoVoigt",
                        );
                        ui.selectable_value(
                            &mut st.lineshape,
                            LineshapeKind::Lorentzian,
                            "Lorentzian",
                        );
                        ui.selectable_value(&mut st.lineshape, LineshapeKind::Gaussian, "Gaussian");
                    });

                ui.label(egui::RichText::new("Sharing:").size(TYPE_CAPTION_PX));
                egui::ComboBox::from_id_salt("p22_deconv_sharing")
                    .selected_text(format!("{:?}", st.linewidth_sharing))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut st.linewidth_sharing,
                            LinewidthSharing::SharedAcrossPlanes,
                            "SharedAcrossPlanes",
                        );
                        ui.selectable_value(
                            &mut st.linewidth_sharing,
                            LinewidthSharing::SharedPerCluster,
                            "SharedPerCluster",
                        );
                        ui.selectable_value(
                            &mut st.linewidth_sharing,
                            LinewidthSharing::Independent,
                            "Independent",
                        );
                    });

                ui.checkbox(
                    &mut st.fit_mixing,
                    egui::RichText::new("Fit eta").size(TYPE_CAPTION_PX),
                );
                ui.checkbox(
                    &mut st.fit_baseline,
                    egui::RichText::new("Fit baseline").size(TYPE_CAPTION_PX),
                );
                ui.label(egui::RichText::new("Max iter:").size(TYPE_CAPTION_PX));
                ui.add(egui::DragValue::new(&mut st.max_iterations).range(10..=500));
            });

            ui.horizontal_wrapped(|ui| {
                ui.checkbox(
                    &mut st.use_known_sigma,
                    egui::RichText::new("Known sigma (chi2 gate):").size(TYPE_CAPTION_PX),
                );
                ui.add_enabled(
                    st.use_known_sigma,
                    egui::DragValue::new(&mut st.known_noise_sigma)
                        .range(1e-5..=10.0)
                        .speed(0.0005)
                        .max_decimals(5),
                );
                ui.separator();
                ui.checkbox(
                    &mut st.use_center_shift_bound,
                    egui::RichText::new("Center shift cap (xFWHM):").size(TYPE_CAPTION_PX),
                );
                ui.add_enabled(
                    st.use_center_shift_bound,
                    egui::DragValue::new(&mut st.max_center_shift_fwhm)
                        .range(0.05..=5.0)
                        .speed(0.05),
                );
            });
        });

    ui.add_space(4.0);

    // Editable Peak Seeds table
    egui::Frame::NONE
        .fill(theme.card_bg)
        .stroke(egui::Stroke::new(1.0, theme.border))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("Editable Peak Seeds ({})", st.seeds.len()))
                        .size(TYPE_BODY_PX)
                        .strong()
                        .color(theme.text_primary),
                );
                if ui
                    .button(egui::RichText::new("+ Add Seed").size(TYPE_CAPTION_PX))
                    .clicked()
                {
                    let next = st.seeds.last().copied().unwrap_or(PeakSeed2D {
                        center_f1: 120.0,
                        center_f2: 8.30,
                        fwhm_f1: 0.50,
                        fwhm_f2: 0.05,
                        eta_f1: 0.5,
                        eta_f2: 0.5,
                    });
                    st.seeds.push(PeakSeed2D {
                        center_f1: next.center_f1 + 0.25,
                        center_f2: next.center_f2 + 0.025,
                        ..next
                    });
                }
            });

            let mut remove_idx: Option<usize> = None;
            egui::Grid::new("p22_deconv_seeds_grid")
                .striped(true)
                .spacing([8.0, 3.0])
                .show(ui, |ui| {
                    for hdr in [
                        "#",
                        "F1 center",
                        "F2 center",
                        "F1 FWHM",
                        "F2 FWHM",
                        "Eta F1",
                        "Action",
                    ] {
                        ui.label(egui::RichText::new(hdr).size(TYPE_CAPTION_PX).strong());
                    }
                    ui.end_row();
                    let can_remove = st.seeds.len() > 1;
                    for (idx, seed) in st.seeds.iter_mut().enumerate() {
                        ui.label(
                            egui::RichText::new(format!("P{}", idx + 1)).size(TYPE_CAPTION_PX),
                        );
                        ui.add(egui::DragValue::new(&mut seed.center_f1).speed(0.02));
                        ui.add(egui::DragValue::new(&mut seed.center_f2).speed(0.005));
                        ui.add(
                            egui::DragValue::new(&mut seed.fwhm_f1)
                                .range(0.001..=20.0)
                                .speed(0.01),
                        );
                        ui.add(
                            egui::DragValue::new(&mut seed.fwhm_f2)
                                .range(0.0005..=5.0)
                                .speed(0.002),
                        );
                        ui.add(
                            egui::DragValue::new(&mut seed.eta_f1)
                                .range(0.0..=1.0)
                                .speed(0.02),
                        );
                        seed.eta_f2 = seed.eta_f1;
                        if ui
                            .add_enabled(
                                can_remove,
                                egui::Button::new(
                                    egui::RichText::new("Remove").size(TYPE_CAPTION_PX),
                                ),
                            )
                            .clicked()
                        {
                            remove_idx = Some(idx);
                        }
                        ui.end_row();
                    }
                });
            if let Some(i) = remove_idx {
                if i < st.seeds.len() {
                    st.seeds.remove(i);
                }
            }
        });

    ui.add_space(4.0);

    // Error banner if fail-closed at validation
    if let Some(err) = &st.error_message {
        show_status_banner(
            ui,
            &format!("FAIL-CLOSED ERROR: {err}"),
            theme.err_bg,
            theme.err_fg,
        );
        ui.add_space(4.0);
    }

    // Results section
    if let Some(res) = &st.result {
        let (badge_bg, badge_fg, diag_text) = match res.diagnostic {
            DeconvDiagnostic::Converged => (theme.ok_bg, theme.ok_fg, "Converged".to_string()),
            DeconvDiagnostic::ActiveBoundHit => (
                theme.warn_bg,
                theme.warn_fg,
                format!(
                    "FAIL-CLOSED: ActiveBoundHit ({} active bound(s))",
                    res.active_bounds.len()
                ),
            ),
            DeconvDiagnostic::ModelMisspecified { reduced_chi2 } => (
                theme.err_bg,
                theme.err_fg,
                format!("FAIL-CLOSED: ModelMisspecified (reduced_chi2 = {reduced_chi2:.3})"),
            ),
            DeconvDiagnostic::IllConditionedCluster => (
                theme.err_bg,
                theme.err_fg,
                format!(
                    "FAIL-CLOSED: IllConditionedCluster (kappa = {:.2e})",
                    res.basis_condition_number
                ),
            ),
            DeconvDiagnostic::DegeneratePeakPair => (
                theme.err_bg,
                theme.err_fg,
                format!(
                    "FAIL-CLOSED: DegeneratePeakPair (min sep = {:.2} FWHM)",
                    res.min_pair_separation_fwhm
                ),
            ),
            DeconvDiagnostic::NonConverged => (
                theme.warn_bg,
                theme.warn_fg,
                format!(
                    "FAIL-CLOSED: NonConverged after {} iterations",
                    res.iterations
                ),
            ),
        };

        show_status_banner(ui, &diag_text, badge_bg, badge_fg);
        ui.add_space(4.0);

        // Summary metrics bar
        egui::Frame::NONE
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    metric_chip(
                        ui,
                        "reduced_chi2",
                        &format!("{:.4}", res.reduced_chi2),
                        theme,
                    );
                    metric_chip(
                        ui,
                        "crlb_valid",
                        if res.crlb_valid {
                            "true"
                        } else {
                            "false (suppressed)"
                        },
                        theme,
                    );
                    metric_chip(ui, "iters", &res.iterations.to_string(), theme);
                    metric_chip(ui, "DoF", &res.dof.to_string(), theme);
                    metric_chip(
                        ui,
                        "resid_sigma",
                        &format!("{:.4e}", res.residual_sigma),
                        theme,
                    );
                    metric_chip(
                        ui,
                        "kappa(Phi)",
                        &format!("{:.2}", res.basis_condition_number),
                        theme,
                    );
                });

                if !res.active_bounds.is_empty() {
                    ui.add_space(4.0);
                    let bounds_str: Vec<String> = res
                        .active_bounds
                        .iter()
                        .map(|b| {
                            format!(
                                "{} ({:?}: {:.4} vs {:.4})",
                                b.param_name, b.side, b.value, b.limit
                            )
                        })
                        .collect();
                    ui.label(
                        egui::RichText::new(format!(
                            "Active bounds suppressing CRLB: {}",
                            bounds_str.join("; ")
                        ))
                        .size(TYPE_CAPTION_PX)
                        .color(theme.warn_fg),
                    );
                }

                ui.add_space(4.0);
                egui::Grid::new("p22_deconv_results_table")
                    .striped(true)
                    .spacing([10.0, 3.0])
                    .show(ui, |ui| {
                        for hdr in [
                            "Peak",
                            "F1 Center (+/- SE)",
                            "F2 Center (+/- SE)",
                            "F1 FWHM (+/- SE)",
                            "F2 FWHM (+/- SE)",
                            "Amplitude (+/- SE)",
                            "Volume",
                        ] {
                            ui.label(egui::RichText::new(hdr).size(TYPE_CAPTION_PX).strong());
                        }
                        ui.end_row();
                        for (idx, pk) in res.peaks.iter().enumerate() {
                            ui.label(
                                egui::RichText::new(format!("P{}", idx + 1)).size(TYPE_CAPTION_PX),
                            );
                            ui.label(
                                egui::RichText::new(fmt_val_se(pk.center_f1, pk.center_f1_se, 4))
                                    .size(TYPE_CAPTION_PX),
                            );
                            ui.label(
                                egui::RichText::new(fmt_val_se(pk.center_f2, pk.center_f2_se, 4))
                                    .size(TYPE_CAPTION_PX),
                            );
                            ui.label(
                                egui::RichText::new(fmt_val_se(pk.fwhm_f1, pk.fwhm_f1_se, 4))
                                    .size(TYPE_CAPTION_PX),
                            );
                            ui.label(
                                egui::RichText::new(fmt_val_se(pk.fwhm_f2, pk.fwhm_f2_se, 4))
                                    .size(TYPE_CAPTION_PX),
                            );
                            ui.label(
                                egui::RichText::new(fmt_val_se(pk.amplitude, pk.amplitude_se, 4))
                                    .size(TYPE_CAPTION_PX),
                            );
                            ui.label(
                                egui::RichText::new(format!("{:.4e}", pk.volume))
                                    .size(TYPE_CAPTION_PX),
                            );
                            ui.end_row();
                        }
                    });
            });

        // Compact 1D cross-section preview (Observed vs VarPro Model vs Residual)
        if let Some(roi) = &st.last_roi {
            ui.add_space(4.0);
            paint_deconv_slice_preview(ui, roi, res, theme);
        }
    } else {
        ui.label(
            egui::RichText::new(&st.status_message)
                .size(TYPE_CAPTION_PX)
                .color(theme.text_muted),
        );
    }
}

fn fmt_val_se(val: f64, se: Option<f64>, prec: usize) -> String {
    match se {
        Some(s) if s.is_finite() => format!("{val:.prec$} +/- {s:.prec$}"),
        _ => format!("{val:.prec$} (--)"),
    }
}

fn paint_deconv_slice_preview(
    ui: &mut egui::Ui,
    roi: &Roi2D,
    res: &DeconvResult2D,
    theme: &P22ToolsTheme,
) {
    let n1 = roi.n1();
    let n2 = roi.n2();
    if n1 == 0 || n2 == 0 || res.fitted_values.len() != roi.values.len() {
        return;
    }
    let row = n1 / 2;
    let obs_slice = &roi.values[row * n2..(row + 1) * n2];
    let fit_slice = &res.fitted_values[row * n2..(row + 1) * n2];
    let resid_slice = &res.residuals[row * n2..(row + 1) * n2];

    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &v in obs_slice
        .iter()
        .chain(fit_slice.iter())
        .chain(resid_slice.iter())
    {
        if v.is_finite() {
            y_min = y_min.min(v);
            y_max = y_max.max(v);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() || (y_max - y_min).abs() < 1e-12 {
        return;
    }

    let desired = egui::vec2(ui.available_width().max(260.0), 110.0);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter();
    painter.rect(
        rect,
        4.0,
        theme.plot_bg,
        egui::Stroke::new(1.0, theme.border),
        egui::StrokeKind::Inside,
    );

    let plot = rect.shrink2(egui::vec2(10.0, 20.0));
    let to_pt = |i: usize, y: f64| -> egui::Pos2 {
        let fx = if n2 > 1 {
            (i as f32) / ((n2 - 1) as f32)
        } else {
            0.5
        };
        let fy = ((y - y_min) / (y_max - y_min)) as f32;
        egui::pos2(
            plot.left() + fx * plot.width(),
            plot.bottom() - fy.clamp(0.0, 1.0) * plot.height(),
        )
    };

    let obs_pts: Vec<egui::Pos2> = obs_slice
        .iter()
        .enumerate()
        .map(|(i, &v)| to_pt(i, v))
        .collect();
    let fit_pts: Vec<egui::Pos2> = fit_slice
        .iter()
        .enumerate()
        .map(|(i, &v)| to_pt(i, v))
        .collect();
    let res_pts: Vec<egui::Pos2> = resid_slice
        .iter()
        .enumerate()
        .map(|(i, &v)| to_pt(i, v))
        .collect();

    painter.add(egui::Shape::line(
        obs_pts,
        egui::Stroke::new(1.5, theme.text_muted),
    ));
    painter.add(egui::Shape::line(
        fit_pts,
        egui::Stroke::new(1.8, theme.accent),
    ));
    painter.add(egui::Shape::line(
        res_pts,
        egui::Stroke::new(1.2, theme.warn_fg),
    ));

    painter.text(
        egui::pos2(rect.left() + 8.0, rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        "Mid-F1 Slice: Observed (muted) | VarPro Model (accent) | Residual (amber)",
        egui::FontId::proportional(TYPE_CAPTION_PX),
        theme.text_primary,
    );
}

// ============================================================================
// 8. Tab 2 UI: Convex Accelerated MaxEnt NUS (`T22.13`)
// ============================================================================

fn show_maxent_tab(ui: &mut egui::Ui, st: &mut MaxEntTabState, theme: &P22ToolsTheme) {
    ui.horizontal_wrapped(|ui| {
        if ui
            .button(
                egui::RichText::new("Synthetic Demo")
                    .size(TYPE_BODY_PX)
                    .strong(),
            )
            .on_hover_text("Generate 2D Poisson-gap schedule and run convex MaxEnt reconstruction")
            .clicked()
        {
            let _ = st.load_and_run_synthetic_demo();
        }
        if ui
            .button(egui::RichText::new("Run 2D MaxEnt").size(TYPE_BODY_PX))
            .clicked()
        {
            let _ = st.run();
        }
        if ui
            .button(egui::RichText::new("Regenerate Poisson-Gap Schedule").size(TYPE_CAPTION_PX))
            .clicked()
        {
            let _ = st.generate_poisson_gap_schedule();
        }
    });

    ui.add_space(4.0);

    // Editable MaxEnt configuration card
    egui::Frame::NONE
        .fill(theme.card_bg)
        .stroke(egui::Stroke::new(1.0, theme.border))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("MaxEnt Convex Solver & Quadrature Configuration")
                    .size(TYPE_BODY_PX)
                    .strong()
                    .color(theme.text_primary),
            );
            ui.add_space(3.0);

            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Grid n2 x n1:").size(TYPE_CAPTION_PX));
                ui.add(egui::DragValue::new(&mut st.n2).range(8..=64));
                ui.label(egui::RichText::new("x").size(TYPE_CAPTION_PX));
                ui.add(egui::DragValue::new(&mut st.n1).range(8..=64));
                ui.label(egui::RichText::new("ZF:").size(TYPE_CAPTION_PX));
                ui.add(egui::DragValue::new(&mut st.zero_fill_factor).range(1..=2));
                ui.label(egui::RichText::new("NUS frac:").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.sampling_fraction)
                        .range(0.10..=1.0)
                        .speed(0.02),
                );
                ui.label(egui::RichText::new("max_iter:").size(TYPE_CAPTION_PX));
                ui.add(egui::DragValue::new(&mut st.max_iter).range(10..=2000));
            });

            ui.add_space(3.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Lambda mode:").size(TYPE_CAPTION_PX));
                ui.radio_value(
                    &mut st.lambda_ui_mode,
                    MaxEntLambdaUiMode::Fixed,
                    egui::RichText::new("Fixed").size(TYPE_CAPTION_PX),
                );
                ui.radio_value(
                    &mut st.lambda_ui_mode,
                    MaxEntLambdaUiMode::DiscrepancyPrinciple,
                    egui::RichText::new("Morozov Discrepancy").size(TYPE_CAPTION_PX),
                );
                match st.lambda_ui_mode {
                    MaxEntLambdaUiMode::Fixed => {
                        ui.label(egui::RichText::new("lambda:").size(TYPE_CAPTION_PX));
                        ui.add(
                            egui::DragValue::new(&mut st.lambda)
                                .range(0.01..=1.0e5)
                                .speed(0.2),
                        );
                    }
                    MaxEntLambdaUiMode::DiscrepancyPrinciple => {
                        ui.label(egui::RichText::new("sigma:").size(TYPE_CAPTION_PX));
                        ui.add(
                            egui::DragValue::new(&mut st.discrepancy_sigma)
                                .range(1e-5..=1.0)
                                .speed(0.001)
                                .max_decimals(5),
                        );
                        ui.label(egui::RichText::new("target chi2_nu:").size(TYPE_CAPTION_PX));
                        ui.add(
                            egui::DragValue::new(&mut st.target_chi2_nu)
                                .range(0.2..=3.0)
                                .speed(0.05),
                        );
                    }
                }
                ui.label(egui::RichText::new("def:").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.def)
                        .range(1e-4..=100.0)
                        .speed(0.02),
                );
            });

            ui.add_space(3.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Quad F2:").size(TYPE_CAPTION_PX));
                quad_mode_combo(ui, "p22_maxent_quad_f2", &mut st.quad_mode_axis2);
                ui.label(egui::RichText::new("Quad F1:").size(TYPE_CAPTION_PX));
                quad_mode_combo(ui, "p22_maxent_quad_f1", &mut st.quad_mode_axis1);
                ui.checkbox(
                    &mut st.accelerate,
                    egui::RichText::new("FISTA").size(TYPE_CAPTION_PX),
                );
                ui.checkbox(
                    &mut st.adaptive_restart,
                    egui::RichText::new("Adaptive Restart").size(TYPE_CAPTION_PX),
                );
            });
        });

    ui.add_space(4.0);

    // Editable NUS schedule buffer
    egui::Frame::NONE
        .fill(theme.card_bg)
        .stroke(egui::Stroke::new(1.0, theme.border))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(
                    "Editable 2D NUS Schedule (0-based 't2 t1' coordinates per line; comments '#')",
                )
                .size(TYPE_CAPTION_PX)
                .strong(),
            );
            egui::ScrollArea::vertical()
                .id_salt("p22_maxent_sched_scroll")
                .max_height(88.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut st.schedule_text)
                            .font(egui::FontId::monospace(TYPE_CAPTION_PX))
                            .desired_width(f32::INFINITY)
                            .desired_rows(4),
                    );
                });
        });

    ui.add_space(4.0);

    // Fail-closed error banner (surfaces EchoAntiecho rejection cleanly)
    if let Some(err) = &st.error_message {
        show_status_banner(
            ui,
            &format!("FAIL-CLOSED MAXENT ERROR: {err}"),
            theme.err_bg,
            theme.err_fg,
        );
        ui.add_space(4.0);
    }

    // Results & Honest Nonlinearity Disclosure
    if let Some(res) = &st.result {
        let (bg, fg) = if res.converged {
            (theme.ok_bg, theme.ok_fg)
        } else {
            (theme.warn_bg, theme.warn_fg)
        };
        show_status_banner(ui, &st.status_message, bg, fg);
        ui.add_space(4.0);

        egui::Frame::NONE
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                let total_grid = (st.n2 * st.n1).max(1);
                let cov_pct = 100.0 * (st.last_sampled_count as f64) / (total_grid as f64);
                ui.horizontal_wrapped(|ui| {
                    metric_chip(
                        ui,
                        "KKT ||grad||_inf",
                        &format!("{:.3e}", res.kkt_residual_inf),
                        theme,
                    );
                    metric_chip(
                        ui,
                        "Duality Gap",
                        &format!("{:.3e}", res.duality_gap),
                        theme,
                    );
                    metric_chip(
                        ui,
                        "Final lambda",
                        &format!("{:.4}", res.final_lambda),
                        theme,
                    );
                    metric_chip(ui, "Entropy S", &format!("{:.3}", res.entropy), theme);
                    metric_chip(
                        ui,
                        "Iters (total)",
                        &format!("{} ({})", res.iterations, res.total_iterations),
                        theme,
                    );
                    metric_chip(
                        ui,
                        "NUS Coverage",
                        &format!("{}/{} ({:.1}%)", st.last_sampled_count, total_grid, cov_pct),
                        theme,
                    );
                    if let Some(c2) = res.chi2_reduced {
                        metric_chip(ui, "chi2_nu", &format!("{c2:.3}"), theme);
                    }
                });
            });

        ui.add_space(4.0);
        paint_maxent_preview(ui, st, res, theme);
        ui.add_space(4.0);
    }

    // Always display honest MaxEnt nonlinearity disclosure vs linear/IST NUS
    let disclosure = st
        .result
        .as_ref()
        .map_or(MAXENT_LINEARITY_DISCLOSURE, |r| r.linearity_note.as_str());
    egui::Frame::NONE
        .fill(theme.warn_bg)
        .stroke(egui::Stroke::new(1.0, theme.warn_fg))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("Scientific Candor: MaxEnt Nonlinearity vs IST")
                    .size(TYPE_CAPTION_PX)
                    .strong()
                    .color(theme.warn_fg),
            );
            ui.label(
                egui::RichText::new(disclosure)
                    .size(TYPE_CAPTION_PX)
                    .color(theme.text_primary),
            );
        });
}

fn quad_mode_combo(ui: &mut egui::Ui, id: &str, mode: &mut QuadMode) {
    let text = match *mode {
        QuadMode::States => "States",
        QuadMode::StatesTppi => "States-TPPI",
        QuadMode::EchoAntiecho => "Echo-AntiEcho (fail-closed)",
        other => {
            // Fallback label if another variant is ever set programmatically
            match other {
                QuadMode::Dqd => "Dqd",
                QuadMode::Qf => "Qf",
                _ => "States",
            }
        }
    };
    egui::ComboBox::from_id_salt(id)
        .selected_text(egui::RichText::new(text).size(TYPE_CAPTION_PX))
        .show_ui(ui, |ui| {
            ui.selectable_value(mode, QuadMode::States, "States");
            ui.selectable_value(mode, QuadMode::StatesTppi, "States-TPPI");
            ui.selectable_value(mode, QuadMode::EchoAntiecho, "Echo-AntiEcho (fail-closed)");
        });
}

fn paint_maxent_preview(
    ui: &mut egui::Ui,
    st: &MaxEntTabState,
    res: &MaxEntResult<[f64; 4]>,
    theme: &P22ToolsTheme,
) {
    if res.spectrum_shape.len() != 2 {
        return;
    }
    let f2 = res.spectrum_shape[0];
    let f1 = res.spectrum_shape[1];
    if f2 == 0 || f1 == 0 || res.spectrum.len() != f2 * f1 {
        return;
    }
    let max_rr = res
        .spectrum
        .iter()
        .map(|q| q[0].abs())
        .fold(0.0_f64, f64::max)
        .max(1e-12);

    let desired = egui::vec2(ui.available_width().max(260.0), 120.0);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter();
    painter.rect(
        rect,
        4.0,
        theme.plot_bg,
        egui::Stroke::new(1.0, theme.border),
        egui::StrokeKind::Inside,
    );

    let half_w = (rect.width() - 24.0) * 0.5;
    let map_h = (rect.height() - 28.0).max(40.0);
    let left_rect = egui::Rect::from_min_size(
        egui::pos2(rect.left() + 8.0, rect.top() + 20.0),
        egui::vec2(half_w.max(60.0), map_h),
    );
    let right_rect = egui::Rect::from_min_size(
        egui::pos2(left_rect.right() + 8.0, rect.top() + 20.0),
        egui::vec2(half_w.max(60.0), map_h),
    );

    // Left: NUS Sampling Mask
    painter.text(
        egui::pos2(left_rect.left(), rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        "2D NUS Sampling Mask (t2, t1)",
        egui::FontId::proportional(TYPE_CAPTION_PX),
        theme.text_primary,
    );
    if st.last_mask.len() == st.n2 * st.n1 && st.n2 > 0 && st.n1 > 0 {
        let cw = left_rect.width() / (st.n1 as f32);
        let ch = left_rect.height() / (st.n2 as f32);
        for k2 in 0..st.n2 {
            for k1 in 0..st.n1 {
                if st.last_mask[k2 * st.n1 + k1] {
                    let cell = egui::Rect::from_min_size(
                        egui::pos2(
                            left_rect.left() + (k1 as f32) * cw,
                            left_rect.top() + (k2 as f32) * ch,
                        ),
                        egui::vec2(cw.max(1.5), ch.max(1.5)),
                    );
                    painter.rect_filled(cell, 0.0, theme.accent);
                }
            }
        }
    }

    // Right: Reconstructed 2D MaxEnt RR Spectrum intensity map
    painter.text(
        egui::pos2(right_rect.left(), rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        "Reconstructed 2D MaxEnt Spectrum (RR)",
        egui::FontId::proportional(TYPE_CAPTION_PX),
        theme.text_primary,
    );
    let step2 = (f2 / 32).max(1);
    let step1 = (f1 / 32).max(1);
    let bins2 = f2.div_ceil(step2);
    let bins1 = f1.div_ceil(step1);
    let cw = right_rect.width() / (bins1 as f32);
    let ch = right_rect.height() / (bins2 as f32);
    for b2 in 0..bins2 {
        let m2 = (b2 * step2).min(f2 - 1);
        for b1 in 0..bins1 {
            let m1 = (b1 * step1).min(f1 - 1);
            let frac = (res.spectrum[m2 * f1 + m1][0].abs() / max_rr).clamp(0.0, 1.0) as f32;
            if frac > 0.08 {
                let alpha = (frac * 255.0).round() as u8;
                let col = egui::Color32::from_rgba_unmultiplied(
                    theme.accent.r(),
                    theme.accent.g(),
                    theme.accent.b(),
                    alpha,
                );
                let cell = egui::Rect::from_min_size(
                    egui::pos2(
                        right_rect.left() + (b1 as f32) * cw,
                        right_rect.top() + (b2 as f32) * ch,
                    ),
                    egui::vec2(cw.max(1.5), ch.max(1.5)),
                );
                painter.rect_filled(cell, 0.0, col);
            }
        }
    }
}

// ============================================================================
// 9. Tab 3 UI: ARTSY RDC Extraction (`T22.9`)
// ============================================================================

fn show_artsy_tab(ui: &mut egui::Ui, st: &mut ArtsyTabState, theme: &P22ToolsTheme) {
    ui.horizontal_wrapped(|ui| {
        if ui
            .button(
                egui::RichText::new("Synthetic Demo")
                    .size(TYPE_BODY_PX)
                    .strong(),
            )
            .on_hover_text("Load 12-residue synthetic ARTSY TSV and extract 1D_NH couplings")
            .clicked()
        {
            let _ = st.load_and_run_synthetic_demo();
        }
        if ui
            .button(egui::RichText::new("Compute ARTSY RDCs").size(TYPE_BODY_PX))
            .clicked()
        {
            let _ = st.run();
        }
        let has_res = !st.results.is_empty();
        if ui
            .add_enabled(
                has_res,
                egui::Button::new(egui::RichText::new("Copy TSV").size(TYPE_BODY_PX)),
            )
            .clicked()
        {
            let tsv = st.export_tsv();
            ui.ctx().copy_text(tsv);
            st.last_copied_notice = Some(format!(
                "Copied {} ARTSY RDC rows to clipboard.",
                st.results.len()
            ));
        }
        if let Some(notice) = &st.last_copied_notice {
            ui.label(
                egui::RichText::new(notice)
                    .size(TYPE_CAPTION_PX)
                    .color(theme.ok_fg),
            );
        }
    });

    ui.add_space(4.0);

    // Editable ARTSY Experimental Parameters
    egui::Frame::NONE
        .fill(theme.card_bg)
        .stroke(egui::Stroke::new(1.0, theme.border))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("ARTSY Experimental & Quality Parameters")
                    .size(TYPE_BODY_PX)
                    .strong()
                    .color(theme.text_primary),
            );
            ui.add_space(3.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Dephasing T (ms):").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.dephasing_delay_ms)
                        .range(1.0..=40.0)
                        .speed(0.05),
                );
                ui.label(egui::RichText::new("1J_NH iso (Hz):").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.j_iso_default_hz)
                        .range(-120.0..=-60.0)
                        .speed(0.1),
                );
                ui.label(egui::RichText::new("Scan ratio N_att/N_ref:").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.scan_ratio_att_over_ref)
                        .range(0.1..=10.0)
                        .speed(0.05),
                );
                ui.label(egui::RichText::new("Delta R2 (s^-1):").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.delta_r2_s_inv)
                        .range(-50.0..=50.0)
                        .speed(0.2),
                );
            });
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Sign convention:").size(TYPE_CAPTION_PX));
                ui.radio_value(
                    &mut st.sign_convention,
                    RdcSignConvention::SignNormalized,
                    egui::RichText::new("SignNormalized (Bax)").size(TYPE_CAPTION_PX),
                );
                ui.radio_value(
                    &mut st.sign_convention,
                    RdcSignConvention::PhysicalGammaSign,
                    egui::RichText::new("PhysicalGammaSign").size(TYPE_CAPTION_PX),
                );
                ui.label(egui::RichText::new("Min SNR_ref:").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.min_ref_snr)
                        .range(0.0..=100.0)
                        .speed(0.5),
                );
                ui.label(egui::RichText::new("MC trials:").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.mc_trials)
                        .range(0..=2000)
                        .speed(50.0),
                );
            });
        });

    ui.add_space(4.0);

    // Editable ARTSY TSV input box
    egui::Frame::NONE
        .fill(theme.card_bg)
        .stroke(egui::Stroke::new(1.0, theme.border))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(
                    "Editable Input TSV (seq_id, res_name, i_ref, i_att, sigma[, sigma_att])",
                )
                .size(TYPE_CAPTION_PX)
                .strong(),
            );
            egui::ScrollArea::vertical()
                .id_salt("p22_artsy_tsv_scroll")
                .max_height(110.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut st.tsv_input)
                            .font(egui::FontId::monospace(TYPE_CAPTION_PX))
                            .desired_width(f32::INFINITY)
                            .desired_rows(5),
                    );
                });
        });

    ui.add_space(4.0);

    if let Some(err) = &st.error_message {
        show_status_banner(
            ui,
            &format!("ARTSY ERROR: {err}"),
            theme.err_bg,
            theme.err_fg,
        );
        ui.add_space(4.0);
    }

    if !st.results.is_empty() {
        show_status_banner(ui, &st.status_message, theme.ok_bg, theme.ok_fg);
        ui.add_space(4.0);

        let mut clicked_col: Option<ArtsySortColumn> = None;
        egui::Frame::NONE
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                egui::Grid::new("p22_artsy_results_table")
                    .striped(true)
                    .spacing([10.0, 3.0])
                    .show(ui, |ui| {
                        let headers: [(&str, ArtsySortColumn); 8] = [
                            ("Seq", ArtsySortColumn::SeqId),
                            ("Res", ArtsySortColumn::ResName),
                            ("I_att/I_ref", ArtsySortColumn::RawRatio),
                            ("|1J+1D| (Hz)", ArtsySortColumn::AbsJPlusD),
                            ("1D_NH (Hz)", ArtsySortColumn::DNh),
                            ("+/- sigma (Hz)", ArtsySortColumn::SigmaDNh),
                            ("SNR_ref", ArtsySortColumn::SnrRef),
                            ("Quality", ArtsySortColumn::Quality),
                        ];
                        for (label, col) in headers {
                            let arrow = if st.sort_col == col {
                                if st.sort_ascending {
                                    " ^"
                                } else {
                                    " v"
                                }
                            } else {
                                ""
                            };
                            if ui
                                .button(
                                    egui::RichText::new(format!("{label}{arrow}"))
                                        .size(TYPE_CAPTION_PX)
                                        .strong(),
                                )
                                .clicked()
                            {
                                clicked_col = Some(col);
                            }
                        }
                        ui.end_row();

                        for r in &st.results {
                            ui.label(
                                egui::RichText::new(r.seq_id.to_string()).size(TYPE_CAPTION_PX),
                            );
                            ui.label(egui::RichText::new(&r.res_name).size(TYPE_CAPTION_PX));
                            ui.label(
                                egui::RichText::new(format!("{:.4}", r.aligned.raw_ratio))
                                    .size(TYPE_CAPTION_PX),
                            );
                            let jd_str = r
                                .aligned
                                .abs_j_plus_d_hz
                                .map_or_else(|| "--".to_string(), |v| format!("{v:.2}"));
                            ui.label(egui::RichText::new(jd_str).size(TYPE_CAPTION_PX));

                            let d_str = r
                                .d_nh_hz
                                .map_or_else(|| "--".to_string(), |v| format!("{v:+.2}"));
                            ui.label(egui::RichText::new(d_str).size(TYPE_CAPTION_PX).strong());

                            let sig_str = r
                                .sigma_d_nh_hz
                                .map_or_else(|| "--".to_string(), |v| format!("{v:.2}"));
                            ui.label(egui::RichText::new(sig_str).size(TYPE_CAPTION_PX));

                            ui.label(
                                egui::RichText::new(format!("{:.1}", r.aligned.snr_ref))
                                    .size(TYPE_CAPTION_PX),
                            );

                            let q_col = match r.quality {
                                ArtsyQualityFlag::Good => theme.ok_fg,
                                ArtsyQualityFlag::NearSaturation | ArtsyQualityFlag::LowSnr => {
                                    theme.warn_fg
                                }
                                ArtsyQualityFlag::OutOfDomainRatio => theme.err_fg,
                            };
                            ui.label(
                                egui::RichText::new(artsy_quality_label(r.quality))
                                    .size(TYPE_CAPTION_PX)
                                    .strong()
                                    .color(q_col),
                            );
                            ui.end_row();
                        }
                    });
            });

        if let Some(col) = clicked_col {
            if st.sort_col == col {
                st.sort_ascending = !st.sort_ascending;
            } else {
                st.sort_col = col;
                st.sort_ascending = true;
            }
            st.apply_sort();
        }
    } else {
        ui.label(
            egui::RichText::new(&st.status_message)
                .size(TYPE_CAPTION_PX)
                .color(theme.text_muted),
        );
    }
}

// ============================================================================
// 10. Tab 4 UI: 2D PCS Assignment Transfer & Curie-Spin CCR (`T22.10`)
// ============================================================================

fn show_pcs_ccr_tab(ui: &mut egui::Ui, st: &mut PcsCcrTabState, theme: &P22ToolsTheme) {
    ui.horizontal_wrapped(|ui| {
        if ui
            .button(egui::RichText::new("Synthetic Demo").size(TYPE_BODY_PX).strong())
            .on_hover_text(
                "Load synthetic lanthanide diamagnetic + paramagnetic 2D peaks and run PCS + Curie CCR",
            )
            .clicked()
        {
            let _ = st.load_and_run_synthetic_demo();
        }
        if ui
            .button(
                egui::RichText::new("Run PCS Transfer & Curie CCR Fit").size(TYPE_BODY_PX),
            )
            .clicked()
        {
            let _ = st.run();
        }
        if ui
            .add_enabled(
                st.transfer_result.is_some(),
                egui::Button::new(egui::RichText::new("Copy Matched TSV").size(TYPE_BODY_PX)),
            )
            .clicked()
        {
            let tsv = st.export_matched_tsv();
            ui.ctx().copy_text(tsv);
            st.last_copied_notice = Some("Copied PCS transfer table to clipboard.".to_string());
        }
        if let Some(notice) = &st.last_copied_notice {
            ui.label(
                egui::RichText::new(notice)
                    .size(TYPE_CAPTION_PX)
                    .color(theme.ok_fg),
            );
        }
    });

    ui.add_space(4.0);

    // Editable Matching, Metal Position & Mandatory Curie-CCR Physics Parameters
    egui::Frame::NONE
        .fill(theme.card_bg)
        .stroke(egui::Stroke::new(1.0, theme.border))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(8, 6))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new("PCS Matching & Curie-Spin CCR Parameters (editable)")
                    .size(TYPE_BODY_PX)
                    .strong()
                    .color(theme.text_primary),
            );
            ui.add_space(3.0);

            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Solver:").size(TYPE_CAPTION_PX));
                ui.radio_value(
                    &mut st.mode,
                    PcsCcrMode::IterativeEmAndJointCcr,
                    egui::RichText::new("Iterative EM + Joint PCS/CCR").size(TYPE_CAPTION_PX),
                );
                ui.radio_value(
                    &mut st.mode,
                    PcsCcrMode::SinglePassTransfer,
                    egui::RichText::new("Single-Pass Cone/Diagonal").size(TYPE_CAPTION_PX),
                );
                ui.separator();
                ui.label(egui::RichText::new("Metal (A):").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.metal_pos[0])
                        .speed(0.1)
                        .prefix("x "),
                );
                ui.add(
                    egui::DragValue::new(&mut st.metal_pos[1])
                        .speed(0.1)
                        .prefix("y "),
                );
                ui.add(
                    egui::DragValue::new(&mut st.metal_pos[2])
                        .speed(0.1)
                        .prefix("z "),
                );
            });

            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("sigma_H (ppm):").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.sigma_h_ppm)
                        .range(0.001..=1.0)
                        .speed(0.002),
                );
                ui.label(egui::RichText::new("sigma_X (ppm):").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.sigma_x_ppm)
                        .range(0.005..=5.0)
                        .speed(0.01),
                );
                ui.label(egui::RichText::new("r_bleach (A):").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.r_bleach_angstrom)
                        .range(0.0..=25.0)
                        .speed(0.2),
                );
                ui.label(egui::RichText::new("Sinkhorn eps:").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.sinkhorn_temperature)
                        .range(0.05..=10.0)
                        .speed(0.05),
                );
            });

            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("CCR tau_m (ns):").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.tau_m_ns)
                        .range(0.5..=200.0)
                        .speed(0.2),
                );
                ui.label(egui::RichText::new("1H MHz:").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.proton_mhz)
                        .range(100.0..=1500.0)
                        .speed(10.0),
                );
                ui.label(egui::RichText::new("Temp (K):").size(TYPE_CAPTION_PX));
                ui.add(
                    egui::DragValue::new(&mut st.temperature_k)
                        .range(250.0..=360.0)
                        .speed(1.0),
                );
                egui::ComboBox::from_id_salt("p22_ccr_spin_pair")
                    .selected_text(
                        egui::RichText::new(format!("{:?}", st.spin_pair)).size(TYPE_CAPTION_PX),
                    )
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut st.spin_pair, CcrSpinPair::N15H1, "N15H1");
                        ui.selectable_value(&mut st.spin_pair, CcrSpinPair::H1N15, "H1N15");
                        ui.selectable_value(&mut st.spin_pair, CcrSpinPair::C13H1, "C13H1");
                        ui.selectable_value(&mut st.spin_pair, CcrSpinPair::H1C13, "H1C13");
                    });
                egui::ComboBox::from_id_salt("p22_ccr_tensor_model")
                    .selected_text(
                        egui::RichText::new(format!("{:?}", st.tensor_model)).size(TYPE_CAPTION_PX),
                    )
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut st.tensor_model,
                            CcrTensorModel::FullChiTensor,
                            "FullChiTensor",
                        );
                        ui.selectable_value(
                            &mut st.tensor_model,
                            CcrTensorModel::IsotropicChiOnly,
                            "IsotropicChiOnly",
                        );
                    });
            });

            ui.horizontal_wrapped(|ui| {
                ui.checkbox(
                    &mut st.use_initial_tensor,
                    egui::RichText::new("Initial DeltaChi guess:").size(TYPE_CAPTION_PX),
                );
                ui.add_enabled_ui(st.use_initial_tensor, |ui| {
                    ui.add(
                        egui::DragValue::new(&mut st.init_dchi_ax)
                            .speed(0.2)
                            .prefix("ax "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut st.init_dchi_rh)
                            .speed(0.1)
                            .prefix("rh "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut st.init_euler_deg[0])
                            .speed(1.0)
                            .prefix("a "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut st.init_euler_deg[1])
                            .speed(1.0)
                            .prefix("b "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut st.init_euler_deg[2])
                            .speed(1.0)
                            .prefix("g "),
                    );
                });
                ui.checkbox(
                    &mut st.fit_pcs_offset,
                    egui::RichText::new("Fit PCS offset").size(TYPE_CAPTION_PX),
                );
            });
        });

    ui.add_space(4.0);

    // Side-by-side editable diamagnetic and paramagnetic TSV inputs
    ui.columns(2, |cols| {
        egui::Frame::NONE
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(6, 5))
            .show(&mut cols[0], |ui| {
                ui.label(
                    egui::RichText::new(
                        "Diamagnetic Peaks TSV (seq label H X [hx hy hz xx xy xz [ccr sig]])",
                    )
                    .size(TYPE_CAPTION_PX)
                    .strong(),
                );
                egui::ScrollArea::vertical()
                    .id_salt("p22_pcs_dia_scroll")
                    .max_height(96.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut st.dia_tsv)
                                .font(egui::FontId::monospace(TYPE_CAPTION_PX))
                                .desired_width(f32::INFINITY)
                                .desired_rows(4),
                        );
                    });
            });

        egui::Frame::NONE
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(6, 5))
            .show(&mut cols[1], |ui| {
                ui.label(
                    egui::RichText::new("Unassigned Paramagnetic Peaks TSV (peak_id H X [inten])")
                        .size(TYPE_CAPTION_PX)
                        .strong(),
                );
                egui::ScrollArea::vertical()
                    .id_salt("p22_pcs_para_scroll")
                    .max_height(96.0)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut st.para_tsv)
                                .font(egui::FontId::monospace(TYPE_CAPTION_PX))
                                .desired_width(f32::INFINITY)
                                .desired_rows(4),
                        );
                    });
            });
    });

    ui.add_space(4.0);

    if let Some(err) = &st.error_message {
        show_status_banner(
            ui,
            &format!("PCS / CURIE CCR ERROR: {err}"),
            theme.err_bg,
            theme.err_fg,
        );
        ui.add_space(4.0);
    }

    // Fitted Susceptibility Tensor Parameters Summary Card
    if let Some(tensor) = st.active_tensor() {
        show_status_banner(ui, &st.status_message, theme.ok_bg, theme.ok_fg);
        ui.add_space(4.0);

        egui::Frame::NONE
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Fitted Susceptibility Tensor Parameters (10^-32 m^3)")
                        .size(TYPE_BODY_PX)
                        .strong()
                        .color(theme.text_primary),
                );
                ui.add_space(3.0);
                ui.horizontal_wrapped(|ui| {
                    metric_chip(ui, "DeltaChi_ax", &format!("{:.3}", tensor.dchi_ax), theme);
                    metric_chip(ui, "DeltaChi_rh", &format!("{:.3}", tensor.dchi_rh), theme);
                    metric_chip(
                        ui,
                        "Rhombicity",
                        &format!("{:.3}", tensor.rhombicity),
                        theme,
                    );
                    metric_chip(
                        ui,
                        "Euler ZYZ (deg)",
                        &format!(
                            "({:.1}, {:.1}, {:.1})",
                            tensor.euler_zyz_rad[0].to_degrees(),
                            tensor.euler_zyz_rad[1].to_degrees(),
                            tensor.euler_zyz_rad[2].to_degrees()
                        ),
                        theme,
                    );
                    metric_chip(
                        ui,
                        "PCS RMSD (ppm)",
                        &format!("{:.4}", tensor.rmsd_ppm),
                        theme,
                    );
                    metric_chip(ui, "PCS Q", &format!("{:.4}", tensor.q_factor), theme);
                });

                if let Some(jf) = &st.joint_fit {
                    ui.add_space(3.0);
                    ui.horizontal_wrapped(|ui| {
                        metric_chip(
                            ui,
                            "chi_iso (10^-32 m^3)",
                            &format!(
                                "{:.2} +/- {:.2}",
                                jf.chi_iso_1e32_m3, jf.chi_iso_uncertainty_1e32_m3
                            ),
                            theme,
                        );
                        metric_chip(
                            ui,
                            "mu_eff (mu_B)",
                            &format!("{:.2}", jf.mu_eff_bohr),
                            theme,
                        );
                        metric_chip(
                            ui,
                            "CCR RMSD (s^-1)",
                            &format!("{:.3}", jf.ccr_rmsd_s_inv),
                            theme,
                        );
                        metric_chip(ui, "CCR Q", &format!("{:.4}", jf.ccr_q_factor), theme);
                        if let Some(qf) = jf.joint_q_free {
                            metric_chip(ui, "Joint Q_free", &format!("{qf:.4}"), theme);
                        }
                        metric_chip(ui, "cond(A)", &format!("{:.1}", jf.condition_number), theme);
                    });
                }
            });
        ui.add_space(4.0);
    } else if let Some(cf) = &st.ccr_only_fit {
        egui::Frame::NONE
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    metric_chip(
                        ui,
                        "chi_iso (10^-32 m^3)",
                        &format!(
                            "{:.2} +/- {:.2}",
                            cf.chi_iso_1e32_m3, cf.chi_iso_uncertainty_1e32_m3
                        ),
                        theme,
                    );
                    metric_chip(
                        ui,
                        "mu_eff (mu_B)",
                        &format!("{:.2}", cf.mu_eff_bohr),
                        theme,
                    );
                    metric_chip(ui, "CCR Q", &format!("{:.4}", cf.q_factor), theme);
                });
            });
        ui.add_space(4.0);
    } else if st.transfer_result.is_some() {
        show_status_banner(ui, &st.status_message, theme.ok_bg, theme.ok_fg);
        ui.add_space(4.0);
    }

    // Matched Assignments Table + 2D Shift Vector Preview
    if let Some(tr_res) = &st.transfer_result {
        paint_pcs_vector_preview(ui, &st.last_dia_peaks, &st.last_para_peaks, tr_res, theme);
        ui.add_space(4.0);

        egui::Frame::NONE
            .fill(theme.card_bg)
            .stroke(egui::Stroke::new(1.0, theme.border))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(8, 6))
            .show(ui, |ui| {
                egui::Grid::new("p22_pcs_results_table")
                    .striped(true)
                    .spacing([9.0, 3.0])
                    .show(ui, |ui| {
                        for hdr in [
                            "Seq", "Label", "Status", "r_M (A)", "dH_PCS", "dX_PCS", "Diag Res",
                            "Post. P", "Margin",
                        ] {
                            ui.label(egui::RichText::new(hdr).size(TYPE_CAPTION_PX).strong());
                        }
                        ui.end_row();
                        for row in &tr_res.transfers {
                            ui.label(
                                egui::RichText::new(row.seq_id.to_string()).size(TYPE_CAPTION_PX),
                            );
                            ui.label(egui::RichText::new(&row.label).size(TYPE_CAPTION_PX));
                            let (st_label, st_col) = match row.status {
                                PcsMatchStatus::Matched { para_peak_id, .. } => {
                                    (format!("Matched #{para_peak_id}"), theme.ok_fg)
                                }
                                PcsMatchStatus::UnmatchedBleached => {
                                    ("Bleached (PRE)".to_string(), theme.warn_fg)
                                }
                                PcsMatchStatus::UnmatchedNoPeak => {
                                    ("Unmatched".to_string(), theme.err_fg)
                                }
                            };
                            ui.label(
                                egui::RichText::new(st_label)
                                    .size(TYPE_CAPTION_PX)
                                    .color(st_col),
                            );
                            ui.label(
                                egui::RichText::new(
                                    row.metal_distance_angstrom
                                        .map_or_else(|| "--".to_string(), |v| format!("{v:.1}")),
                                )
                                .size(TYPE_CAPTION_PX),
                            );
                            ui.label(
                                egui::RichText::new(
                                    row.observed_pcs_h_ppm
                                        .map_or_else(|| "--".to_string(), |v| format!("{v:+.3}")),
                                )
                                .size(TYPE_CAPTION_PX),
                            );
                            ui.label(
                                egui::RichText::new(
                                    row.observed_pcs_x_ppm
                                        .map_or_else(|| "--".to_string(), |v| format!("{v:+.3}")),
                                )
                                .size(TYPE_CAPTION_PX),
                            );
                            ui.label(
                                egui::RichText::new(
                                    row.diagonal_residual_ppm
                                        .map_or_else(|| "--".to_string(), |v| format!("{v:+.3}")),
                                )
                                .size(TYPE_CAPTION_PX),
                            );
                            ui.label(
                                egui::RichText::new(format!("{:.3}", row.posterior_confidence))
                                    .size(TYPE_CAPTION_PX),
                            );
                            ui.label(
                                egui::RichText::new(format!("{:+.3}", row.posterior_margin))
                                    .size(TYPE_CAPTION_PX),
                            );
                            ui.end_row();
                        }
                    });
            });
    } else {
        ui.label(
            egui::RichText::new(&st.status_message)
                .size(TYPE_CAPTION_PX)
                .color(theme.text_muted),
        );
    }
}

fn paint_pcs_vector_preview(
    ui: &mut egui::Ui,
    dia: &[DiamagneticPeak2D],
    para: &[ParamagneticPeak2D],
    res: &PcsTransferResult,
    theme: &P22ToolsTheme,
) {
    if dia.is_empty() {
        return;
    }
    let mut h_min = f64::INFINITY;
    let mut h_max = f64::NEG_INFINITY;
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    for d in dia {
        h_min = h_min.min(d.h_ppm);
        h_max = h_max.max(d.h_ppm);
        x_min = x_min.min(d.x_ppm);
        x_max = x_max.max(d.x_ppm);
    }
    for p in para {
        h_min = h_min.min(p.h_ppm);
        h_max = h_max.max(p.h_ppm);
        x_min = x_min.min(p.x_ppm);
        x_max = x_max.max(p.x_ppm);
    }
    let h_span = (h_max - h_min).max(0.2);
    let x_span = (x_max - x_min).max(1.0);

    let desired = egui::vec2(ui.available_width().max(260.0), 110.0);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter();
    painter.rect(
        rect,
        4.0,
        theme.plot_bg,
        egui::Stroke::new(1.0, theme.border),
        egui::StrokeKind::Inside,
    );
    let plot = rect.shrink2(egui::vec2(12.0, 20.0));
    let to_pt = |h: f64, x: f64| -> egui::Pos2 {
        // NMR convention: high ppm on left (1H) and top/bottom (15N)
        let fx = ((h_max - h) / h_span) as f32;
        let fy = ((x - x_min) / x_span) as f32;
        egui::pos2(
            plot.left() + fx.clamp(0.0, 1.0) * plot.width(),
            plot.top() + fy.clamp(0.0, 1.0) * plot.height(),
        )
    };

    for tr in &res.transfers {
        if let PcsMatchStatus::Matched { para_index, .. } = tr.status {
            if let (Some(d), Some(p)) = (dia.get(tr.dia_index), para.get(para_index)) {
                painter.line_segment(
                    [to_pt(d.h_ppm, d.x_ppm), to_pt(p.h_ppm, p.x_ppm)],
                    egui::Stroke::new(1.2, theme.accent),
                );
            }
        }
    }
    for d in dia {
        painter.circle_stroke(
            to_pt(d.h_ppm, d.x_ppm),
            3.5,
            egui::Stroke::new(1.4, theme.text_muted),
        );
    }
    for p in para {
        let c = to_pt(p.h_ppm, p.x_ppm);
        painter.rect_filled(
            egui::Rect::from_center_size(c, egui::vec2(5.5, 5.5)),
            1.0,
            theme.ok_fg,
        );
    }

    painter.text(
        egui::pos2(rect.left() + 8.0, rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        "2D PCS Shift Vectors: Diamagnetic (circle) -> Matched Paramagnetic (square)",
        egui::FontId::proportional(TYPE_CAPTION_PX),
        theme.text_primary,
    );
}

// ============================================================================
// 11. Shared UI Helpers (all >= TYPE_CAPTION_PX = 11.0 px)
// ============================================================================

fn show_status_banner(ui: &mut egui::Ui, text: &str, bg: egui::Color32, fg: egui::Color32) {
    egui::Frame::NONE
        .fill(bg)
        .stroke(egui::Stroke::new(1.0, fg))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(8, 5))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(text)
                    .size(TYPE_BODY_PX)
                    .strong()
                    .color(fg),
            );
        });
}

fn metric_chip(ui: &mut egui::Ui, label: &str, value: &str, theme: &P22ToolsTheme) {
    egui::Frame::NONE
        .fill(theme.pane_bg)
        .stroke(egui::Stroke::new(1.0, theme.border))
        .corner_radius(4.0)
        .inner_margin(egui::Margin::symmetric(6, 3))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.label(
                    egui::RichText::new(format!("{label}:"))
                        .size(TYPE_CAPTION_PX)
                        .color(theme.text_muted),
                );
                ui.label(
                    egui::RichText::new(value)
                        .size(TYPE_BODY_PX)
                        .strong()
                        .color(theme.text_primary),
                );
            });
        });
}
