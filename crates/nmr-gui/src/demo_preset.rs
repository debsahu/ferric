//! Built-in zero-I/O multi-modal demo preset (`T16.4`) for desktop and `wasm32-unknown-unknown`
//! WebGPU browser deployments.
//!
//! # Clean-Room Provenance
//! Original clean-room preset builder assembling embedded public-domain `CC0 1.0` human Ubiquitin
//! (`1UBQ`, Vijay-Kumar, Bugg & Cook, *J. Mol. Biol.* **1987**, 194, 531–544) coordinates and
//! synthetic forward-model reference fixtures (`synthetic_ubiquitin.str`) purely in memory with
//! zero filesystem I/O:
//! - 2D $^{1}\text{H}\text{–}^{15}\text{N}$ `HSQC` & 3D `HNCACB` / `HNCO` volumetric strips (`nmr_core::simulate`),
//! - 3D `1UBQ` macromolecular backbone & sidechain coordinates (`crate::mol3d_viewer`),
//! - 4-mode 1D multiplet reaction/kinetics stack & ROI integrals (`crate::stack1d_panel`),
//! - Pseudo-2D `BPP-LED` `DOSY` multi-component diffusion decay series (`crate::dosy_panel`;
//!   Stejskal & Tanner *J. Chem. Phys.* **1965**; Wu, Chen & Johnson *J. Magn. Reson. A* **1995**),
//! - Two-group (`Control` vs `Case`) 1D biofluid metabolomics cohort with `AI-Binning`, `PQN`,
//!   `PCA`, `NIPALS` `OPLS-DA`, and `STOCSY` (`crate::metabolomics_panel`; De Meyer et al. **2008**;
//!   Dieterle et al. **2006**; Trygg & Wold **2002**; Cloarec et al. **2005**).

use nmr_core::nmrstar::{read_star_str, Reference};
use nmr_core::peaks::Peak;
use nmr_core::simulate::{
    render_hsqc_2d, render_triple_3d, simulate, SimConfig, SpectrumRenderConfig,
};
use nmr_core::spectrum::{Spectrum2D, Spectrum3D};
use nmr_core::spinsystems::{build_spin_systems, SpinSystem, Tolerances};

use crate::dosy_panel::DosyPanelState;
use crate::metabolomics_panel::MetabolomicsStudioPanel;
use crate::mol3d_viewer::Mol3dViewerPanel;
use crate::stack1d_panel::{build_kinetics_demo_traces, Stack1dState};

/// Embedded synthetic Ubiquitin NMR-STAR v3.2 reference shifts (`76 aa`, `CC0` fixture).
pub const EMBEDDED_SYNTHETIC_UBIQUITIN_STR: &str =
    include_str!("../../../data/fixtures/synthetic_ubiquitin.str");

/// Embedded human Ubiquitin `1UBQ` X-ray structure (`76 aa`, wwPDB `CC0 1.0`).
pub const EMBEDDED_UBIQUITIN_1UBQ_PDB: &str =
    include_str!("../../../data/fixtures/ubiquitin_1ubq_xray.pdb");

/// Canonical deterministic seed for the built-in multi-modal demo suite.
pub const BUILTIN_DEMO_SUITE_SEED: u64 = 42;

/// Complete in-memory multi-modal demo dataset ready to populate native desktop or `wasm32`
/// browser workbench sessions without touching the filesystem.
pub struct BuiltinDemoSuite {
    /// Parsed `76 aa` Ubiquitin chemical shift reference.
    pub ubiquitin_ref: Reference,
    /// Simulated 2D `HSQC` and 3D triple-resonance peak table.
    pub peaks: Vec<Peak>,
    /// Ground-truth spin-system -> sequence index mapping for synthetic validation.
    pub truth: Vec<i32>,
    /// Assembled `SpinSystem` roots with `CA`/`CB`/`C'` candidates.
    pub systems: Vec<SpinSystem>,
    /// Rendered 2D $^{1}\text{H}\text{–}^{15}\text{N}$ `HSQC` spectrum (`512 x 256`).
    pub hsqc_2d: Spectrum2D,
    /// Rendered 3D `HNCACB` strip cube (`CA > 0`, `CB < 0`).
    pub hncacb_3d: Spectrum3D,
    /// Rendered 3D `HNCO` carbonyl strip cube.
    pub hnco_3d: Spectrum3D,
    /// Embedded `1UBQ` PDB text (`76 aa`).
    pub pdb_text: &'static str,
    /// Populated 3D biomolecular cartoon/backbone structure viewer state.
    pub mol3d_panel: Mol3dViewerPanel,
    /// Populated 1D multiplet stack & ROI kinetics comparison state.
    pub stack1d_panel: Stack1dState,
    /// Populated pseudo-2D `DOSY` diffusion decay series & discrete `BIC` fits.
    pub dosy_panel: DosyPanelState,
    /// Populated 2-group (`Control` vs `Case`) 1D metabolomics & `OPLS-DA`/`STOCSY` studio.
    pub metabolomics_panel: MetabolomicsStudioPanel,
}

impl BuiltinDemoSuite {
    /// Human-readable one-line status banner summarizing all populated modalities.
    pub fn status_summary(&self) -> String {
        let n_res = self.ubiquitin_ref.sequence.len();
        let n_sys = self.systems.len();
        let n_traces = self.stack1d_panel.traces.len();
        let n_dosy = self.dosy_panel.peaks.len();
        let n_cohort = self.metabolomics_panel.samples.len();
        format!(
            "Loaded Built-in Demo Suite: Ubiquitin ({n_res} aa, {n_sys} spin systems, 2D HSQC + 3D HNCACB + 1UBQ 3D), \
             {n_traces} 1D multiplet traces, {n_dosy} DOSY decays & {n_cohort}-sample 2-group Metabolomics cohort"
        )
    }
}

/// Construct the complete built-in multi-modal demo suite strictly in memory (`0` filesystem I/O).
pub fn build_builtin_demo_suite(seed: u64) -> Result<BuiltinDemoSuite, String> {
    let ubiquitin_ref = read_star_str(EMBEDDED_SYNTHETIC_UBIQUITIN_STR, None)
        .map_err(|e| format!("failed to parse embedded synthetic_ubiquitin.str: {e}"))?;

    let (peaks, truth) = simulate(
        &ubiquitin_ref,
        &SimConfig {
            p_missing_peak: 0.04,
            spurious_rate: 0.03,
            seed,
            ..SimConfig::default()
        },
    );
    let systems = build_spin_systems(&peaks, &Tolerances::default(), 3, false)
        .map_err(|e| format!("failed to build demo spin systems: {e}"))?;

    let render_cfg = SpectrumRenderConfig {
        nx_h: 512,
        ny_n: 256,
        nz_c: 96,
        seed: seed.wrapping_add(7),
        ..SpectrumRenderConfig::default()
    };
    let hsqc_2d = render_hsqc_2d(&peaks, &render_cfg);
    let hncacb_3d = render_triple_3d("HNCACB", &peaks, &render_cfg);
    let hnco_3d = render_triple_3d("HNCO", &peaks, &render_cfg);

    let mut mol3d_panel = Mol3dViewerPanel::new();
    mol3d_panel
        .load_structure_text(
            EMBEDDED_UBIQUITIN_1UBQ_PDB,
            &ubiquitin_ref.sequence,
            &ubiquitin_ref.seq_ids,
        )
        .map_err(|e| format!("failed to parse embedded 1UBQ PDB coordinates: {e}"))?;

    let (traces, rois) = build_kinetics_demo_traces();
    let mut stack1d_panel = Stack1dState::with_traces(traces);
    stack1d_panel.roi_regions = rois;

    let dosy_panel = DosyPanelState::with_synthetic_demo();

    let mut metabolomics_panel = MetabolomicsStudioPanel::new();
    metabolomics_panel.demo_seed = 20261115_u64.wrapping_add(seed);
    metabolomics_panel.load_demo_cohort();

    Ok(BuiltinDemoSuite {
        ubiquitin_ref,
        peaks,
        truth,
        systems,
        hsqc_2d,
        hncacb_3d,
        hnco_3d,
        pdb_text: EMBEDDED_UBIQUITIN_1UBQ_PDB,
        mol3d_panel,
        stack1d_panel,
        dosy_panel,
        metabolomics_panel,
    })
}
