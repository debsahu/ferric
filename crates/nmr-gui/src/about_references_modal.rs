//! In-App `About FERRIC & Scientific References` Modal (`v0.17.1`, desktop & `wasm32`).
//!
//! Provenance: Original clean-room attribution, licensing, open-data acknowledgment, and
//! searchable peer-reviewed NMR bibliography modal for the FERRIC workbench. Every citation listed
//! in [`VERIFIED_CITATIONS`] corresponds directly to a classic peer-reviewed methodology paper
//! implemented from scratch in `nmr-core` / `nmr-gpu` / `nmr-gui`, with exact verified authors,
//! titles, journals, years, volumes, and pages (zero fabricated DOIs, zero emojis).

use eframe::egui;

use crate::vnmrj_rail::draw_folder_tab;
use crate::workbench_mode::{TYPE_BODY_PX, TYPE_CAPTION_PX, TYPE_HEADER_PX};

/// Scientific domain grouping for verified NMR citations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CitationDomain {
    /// Fourier processing, phasing, baseline correction, and solvent suppression.
    Processing1dNd,
    /// Protein triple-resonance backbone assignment and chemical-shift referencing.
    BackboneAssignment,
    /// Spin relaxation, Lipari-Szabo model-free dynamics, and ligand CSP exchange.
    DynamicsExchange,
    /// Karplus couplings, RDC validation, secondary-structure propensities, and RCI flexibility.
    StructureRdc,
    /// Small-molecule 1D multiplet analysis, solvent shift tables, and DOSY diffusion.
    SmallMoleculeDosy,
    /// 1D spectral alignment, normalization, OPLS-DA, and STOCSY metabolomics.
    MetabolomicsChemometrics,
}

impl CitationDomain {
    /// Ordered catalog of all scientific domains.
    pub const ALL: [Self; 6] = [
        Self::Processing1dNd,
        Self::BackboneAssignment,
        Self::DynamicsExchange,
        Self::StructureRdc,
        Self::SmallMoleculeDosy,
        Self::MetabolomicsChemometrics,
    ];

    /// Compact tab label used in the domain filter bar.
    pub const fn short_label(self) -> &'static str {
        match self {
            Self::Processing1dNd => "Processing",
            Self::BackboneAssignment => "Backbone",
            Self::DynamicsExchange => "Dynamics",
            Self::StructureRdc => "Structure",
            Self::SmallMoleculeDosy => "1D & DOSY",
            Self::MetabolomicsChemometrics => "Metabolomics",
        }
    }

    /// Full section heading for grouped display.
    pub const fn title(self) -> &'static str {
        match self {
            Self::Processing1dNd => "1D / nD Fourier Processing, Phasing & Baseline",
            Self::BackboneAssignment => "Protein Backbone Assignment & Referencing",
            Self::DynamicsExchange => "Spin Relaxation, Dynamics & Chemical Exchange",
            Self::StructureRdc => "Structure Validation, Torsion Angles & RDCs",
            Self::SmallMoleculeDosy => "Small-Molecule 1D Multiplets, Solvents & DOSY",
            Self::MetabolomicsChemometrics => "Metabolomics, Alignment & Chemometrics",
        }
    }
}

/// Verified peer-reviewed scientific reference implemented in FERRIC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScientificCitation {
    pub domain: CitationDomain,
    pub authors: &'static str,
    pub year: u16,
    pub title: &'static str,
    pub journal: &'static str,
    pub volume_pages: &'static str,
    pub used_for: &'static str,
}

impl ScientificCitation {
    /// Format as a standard plain-text academic citation string suitable for manuscripts.
    pub fn format_reference(&self) -> String {
        format!(
            "{} ({}). {}. {}, {}.",
            self.authors, self.year, self.title, self.journal, self.volume_pages
        )
    }

    /// Case-insensitive substring match against authors, title, journal, year, domain, and usage.
    pub fn matches_query(&self, query_lower: &str) -> bool {
        if query_lower.is_empty() {
            return true;
        }
        let year_str = self.year.to_string();
        self.authors.to_ascii_lowercase().contains(query_lower)
            || self.title.to_ascii_lowercase().contains(query_lower)
            || self.journal.to_ascii_lowercase().contains(query_lower)
            || self.volume_pages.to_ascii_lowercase().contains(query_lower)
            || self.used_for.to_ascii_lowercase().contains(query_lower)
            || self
                .domain
                .title()
                .to_ascii_lowercase()
                .contains(query_lower)
            || year_str.contains(query_lower)
    }
}

/// Open scientific data resource acknowledged in the modal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenDataAcknowledgment {
    pub name: &'static str,
    pub licence: &'static str,
    pub description: &'static str,
    pub citation: &'static str,
}

/// Open public-domain and Creative Commons scientific databases acknowledged by FERRIC.
pub const OPEN_DATA_ACKNOWLEDGMENTS: &[OpenDataAcknowledgment] = &[
    OpenDataAcknowledgment {
        name: "BMRB (Biological Magnetic Resonance Data Bank)",
        licence: "CC0 1.0 Public Domain",
        description: "Reference biomolecular chemical-shift statistics, benchmark NMR-STAR v3 entries, and amino-acid spin-system priors.",
        citation: "Ulrich, E. L. et al. (2008). Nucleic Acids Res., 36, D402-D408.",
    },
    OpenDataAcknowledgment {
        name: "PDB / wwPDB (Protein Data Bank & Chemical Component Dictionary)",
        licence: "CC0 1.0 Public Domain",
        description: "Experimental macromolecular 3D coordinates and standard IUPAC residue covalent topology tables.",
        citation: "Berman, H. M. et al. (2000). Nucleic Acids Res., 28, 235-242.",
    },
    OpenDataAcknowledgment {
        name: "AlphaFold DB (EMBL-EBI / Google DeepMind)",
        licence: "CC-BY-4.0",
        description: "Predicted 3D protein models, pLDDT confidence profiles, and PAE domain matrices for integrative NMR concordance.",
        citation: "Varadi, M. et al. (2022). Nucleic Acids Res., 50, D439-D444; Jumper, J. et al. (2021). Nature, 596, 583-589.",
    },
    OpenDataAcknowledgment {
        name: "NP-MRD (Natural Products Magnetic Resonance Database)",
        licence: "CC-BY-4.0",
        description: "Open small-molecule and natural-product 1H/13C reference spectra and chemical-shift benchmarks.",
        citation: "Wishart, D. S. et al. (2022). Nucleic Acids Res., 50, D665-D677.",
    },
];

/// Curated catalog of verified classic NMR literature citations (zero fabricated DOIs).
pub const VERIFIED_CITATIONS: &[ScientificCitation] = &[
    // 1. Processing1dNd
    ScientificCitation {
        domain: CitationDomain::Processing1dNd,
        authors: "Cooley, J. W.; Tukey, J. W.",
        year: 1965,
        title: "An algorithm for the machine calculation of complex Fourier series",
        journal: "Math. Comput.",
        volume_pages: "19, 297-301",
        used_for: "Radix-2 FFT / IFFT time-to-frequency spectral transforms and cross-correlation.",
    },
    ScientificCitation {
        domain: CitationDomain::Processing1dNd,
        authors: "Bartholdi, E.; Ernst, R. R.",
        year: 1973,
        title: "Fourier spectroscopy and the causality principle",
        journal: "J. Magn. Reson.",
        volume_pages: "11, 9-19",
        used_for: "Discrete Hilbert reconstruction of imaginary dispersion channels from real-only 1D spectra.",
    },
    ScientificCitation {
        domain: CitationDomain::Processing1dNd,
        authors: "Barkhuijsen, H.; de Beer, R.; van Ormondt, D.",
        year: 1987,
        title: "Improved algorithm for noniterative time-domain model fitting to exponentially damped magnetic resonance signals",
        journal: "J. Magn. Reson.",
        volume_pages: "73, 553-557",
        used_for: "HSVD / LPSVD time-domain solvent and water-resonance suppression.",
    },
    ScientificCitation {
        domain: CitationDomain::Processing1dNd,
        authors: "Delaglio, F.; Grzesiek, S.; Vuister, G. W.; Zhu, G.; Pfeifer, J.; Bax, A.",
        year: 1995,
        title: "NMRPipe: A multidimensional spectral processing system based on UNIX pipes",
        journal: "J. Biomol. NMR",
        volume_pages: "6, 277-293",
        used_for: "NMRPipe 512-float spectral header conventions and multidimensional processing compatibility.",
    },
    ScientificCitation {
        domain: CitationDomain::Processing1dNd,
        authors: "Chen, L.; Weng, Z.; Goh, L.; Garland, M.",
        year: 2002,
        title: "An efficient algorithm for automatic phase correction of NMR spectra based on entropy minimization",
        journal: "J. Magn. Reson.",
        volume_pages: "158, 164-168",
        used_for: "Automatic zero- and first-order phase correction (PHC0 / PHC1) via spectral entropy minimization.",
    },
    ScientificCitation {
        domain: CitationDomain::Processing1dNd,
        authors: "Eilers, P. H. C.",
        year: 2003,
        title: "A perfect smoother",
        journal: "Anal. Chem.",
        volume_pages: "75, 3631-3636",
        used_for: "Whittaker and asymmetric least-squares baseline correction for 1D/nD NMR spectra.",
    },
    // 2. BackboneAssignment
    ScientificCitation {
        domain: CitationDomain::BackboneAssignment,
        authors: "Kay, L. E.; Ikura, M.; Tschudin, R.; Bax, A.",
        year: 1990,
        title: "Three-dimensional triple-resonance NMR spectroscopy of isotopically enriched proteins",
        journal: "J. Magn. Reson.",
        volume_pages: "89, 496-514",
        used_for: "3D HNCA, HNCO, and triple-resonance backbone sequential connectivity rules.",
    },
    ScientificCitation {
        domain: CitationDomain::BackboneAssignment,
        authors: "Spera, S.; Bax, A.",
        year: 1991,
        title: "Empirical correlation between protein backbone conformation and Calpha and Cbeta 13C nuclear magnetic resonance chemical shifts",
        journal: "J. Am. Chem. Soc.",
        volume_pages: "113, 5490-5492",
        used_for: "Secondary 13CA/13CB chemical-shift correlations for spin-system scoring and secondary structure.",
    },
    ScientificCitation {
        domain: CitationDomain::BackboneAssignment,
        authors: "Grzesiek, S.; Bax, A.",
        year: 1992,
        title: "Correlating backbone amide and side chain resonances in larger proteins by multiple relayed triple resonance NMR",
        journal: "J. Am. Chem. Soc.",
        volume_pages: "114, 6291-6293",
        used_for: "3D HNCACB and CBCA(CO)NH intra- and inter-residue aliphatic carbon matching.",
    },
    ScientificCitation {
        domain: CitationDomain::BackboneAssignment,
        authors: "Wishart, D. S.; Sykes, B. D.; Richards, F. M.",
        year: 1992,
        title: "The chemical shift index: A fast and simple method for the assignment of protein secondary structure through NMR spectroscopy",
        journal: "Biochemistry",
        volume_pages: "31, 1647-1651",
        used_for: "Chemical Shift Index (CSI) helix/strand classification and concordance reports.",
    },
    ScientificCitation {
        domain: CitationDomain::BackboneAssignment,
        authors: "Wishart, D. S.; Bigam, C. G.; Yao, J.; Abildgaard, F.; Dyson, H. J.; Oldfield, E.; Markley, J. L.; Sykes, B. D.",
        year: 1995,
        title: "1H, 13C and 15N chemical shift referencing in biomolecular NMR",
        journal: "J. Biomol. NMR",
        volume_pages: "6, 135-140",
        used_for: "Indirect DSS/TMS heteronuclear chemical-shift referencing and Xi frequency ratios.",
    },
    ScientificCitation {
        domain: CitationDomain::BackboneAssignment,
        authors: "Markley, J. L.; Bax, A.; Arata, Y.; Hilbers, C. W.; Kaptein, R.; Sykes, B. D.; Wright, P. E.; Wuthrich, K.",
        year: 1998,
        title: "Recommendations for the presentation of NMR structures of proteins and nucleic acids",
        journal: "J. Mol. Biol.",
        volume_pages: "280, 933-952",
        used_for: "IUPAC atom nomenclature, stereospecific labeling, and NMR-STAR reporting standards.",
    },
    ScientificCitation {
        domain: CitationDomain::BackboneAssignment,
        authors: "Harris, R. K.; Becker, E. D.; Cabral de Menezes, S. M.; Goodfellow, R.; Granger, P.",
        year: 2001,
        title: "NMR nomenclature: Nuclear spin properties and conventions for chemical shifts (IUPAC Recommendations 2001)",
        journal: "Pure Appl. Chem.",
        volume_pages: "73, 1795-1818",
        used_for: "Unified IUPAC chemical-shift scale and gyromagnetic reference standards.",
    },
    // 3. DynamicsExchange
    ScientificCitation {
        domain: CitationDomain::DynamicsExchange,
        authors: "Lipari, G.; Szabo, A.",
        year: 1982,
        title: "Model-free approach to the interpretation of nuclear magnetic resonance relaxation in macromolecules. 1. Theory and range of validity",
        journal: "J. Am. Chem. Soc.",
        volume_pages: "104, 4546-4559",
        used_for: "Lipari-Szabo order parameter (S2) and internal correlation-time relaxation analysis.",
    },
    ScientificCitation {
        domain: CitationDomain::DynamicsExchange,
        authors: "Kay, L. E.; Torchia, D. A.; Bax, A.",
        year: 1989,
        title: "Backbone dynamics of proteins as studied by 15N inverse detected heteronuclear NMR spectroscopy: Application to staphylococcal nuclease",
        journal: "Biochemistry",
        volume_pages: "28, 8972-8979",
        used_for: "15N R1, R2, and heteronuclear NOE spectral-density parameterization.",
    },
    ScientificCitation {
        domain: CitationDomain::DynamicsExchange,
        authors: "Farrow, N. A.; Muhandiram, R.; Singer, A. U.; Pascal, S. M.; Kay, C. M.; Gish, G.; Shoelson, S. E.; Pawson, T.; Forman-Kay, J. D.; Kay, L. E.",
        year: 1994,
        title: "Backbone dynamics of a free and phosphopeptide-complexed Src homology 2 domain studied by 15N NMR relaxation",
        journal: "Biochemistry",
        volume_pages: "33, 5984-6003",
        used_for: "Multi-field 15N relaxation fitting and reduced spectral density mapping.",
    },
    ScientificCitation {
        domain: CitationDomain::DynamicsExchange,
        authors: "Mandel, A. M.; Akke, M.; Palmer, A. G. III",
        year: 1995,
        title: "Backbone dynamics of Escherichia coli ribonuclease HI: Correlations with structure and function in an active enzyme",
        journal: "J. Mol. Biol.",
        volume_pages: "246, 144-163",
        used_for: "Model-free M1-M5 selection criteria including Rex chemical-exchange terms.",
    },
    ScientificCitation {
        domain: CitationDomain::DynamicsExchange,
        authors: "Williamson, M. P.",
        year: 2013,
        title: "Using chemical shift perturbation to characterise ligand binding",
        journal: "Prog. Nucl. Magn. Reson. Spectrosc.",
        volume_pages: "73, 1-16",
        used_for: "Weighted 1H/15N CSP titration tracking, binding-site thresholding, and Kd fitting.",
    },
    // 4. StructureRdc
    ScientificCitation {
        domain: CitationDomain::StructureRdc,
        authors: "Karplus, M.",
        year: 1959,
        title: "Contact electron-spin coupling of nuclear magnetic moments",
        journal: "J. Chem. Phys.",
        volume_pages: "30, 11-15",
        used_for: "Vicinal 3J scalar-coupling dihedral restraint equations.",
    },
    ScientificCitation {
        domain: CitationDomain::StructureRdc,
        authors: "Tjandra, N.; Bax, A.",
        year: 1997,
        title: "Direct measurement of distances and angles in biomolecules by NMR in a dilute liquid crystalline medium",
        journal: "Science",
        volume_pages: "278, 1111-1114",
        used_for: "Residual dipolar coupling (RDC) orientational alignment tensor physics.",
    },
    ScientificCitation {
        domain: CitationDomain::StructureRdc,
        authors: "Cornilescu, G.; Marquardt, J. L.; Ottiger, M.; Bax, A.",
        year: 1998,
        title: "Validation of protein structure from anisotropic carbonyl chemical shifts in a dilute liquid crystalline phase",
        journal: "J. Am. Chem. Soc.",
        volume_pages: "120, 6836-6837",
        used_for: "RDC Q-factor quality metric for 3D structure validation.",
    },
    ScientificCitation {
        domain: CitationDomain::StructureRdc,
        authors: "Berjanskii, M. V.; Wishart, D. S.",
        year: 2005,
        title: "A simple method to predict protein flexibility using secondary chemical shifts",
        journal: "J. Am. Chem. Soc.",
        volume_pages: "127, 14970-14971",
        used_for: "Random Coil Index (RCI) S2 backbone rigidity prediction and AlphaFold concordance.",
    },
    ScientificCitation {
        domain: CitationDomain::StructureRdc,
        authors: "Marsh, J. A.; Singh, V. K.; Jia, Z.; Forman-Kay, J. D.",
        year: 2006,
        title: "Sensitivity of secondary structure propensities to sequence differences between alpha- and gamma-synuclein",
        journal: "Protein Sci.",
        volume_pages: "15, 2795-2804",
        used_for: "Secondary Structure Propensity (SSP) scoring for folded and disordered proteins.",
    },
    // 5. SmallMoleculeDosy
    ScientificCitation {
        domain: CitationDomain::SmallMoleculeDosy,
        authors: "Stejskal, E. O.; Tanner, J. E.",
        year: 1965,
        title: "Spin diffusion measurements: Spin echoes in the presence of a time-dependent field gradient",
        journal: "J. Chem. Phys.",
        volume_pages: "42, 288-292",
        used_for: "Pulsed-field-gradient attenuation equation for DOSY diffusion coefficient fitting.",
    },
    ScientificCitation {
        domain: CitationDomain::SmallMoleculeDosy,
        authors: "Hoye, T. R.; Hanson, P. R.; Vyvyan, J. R.",
        year: 1994,
        title: "A practical guide to first-order multiplet analysis in 1H NMR spectroscopy",
        journal: "J. Org. Chem.",
        volume_pages: "59, 4096-4103",
        used_for: "First-order 1H multiplet J-splitting tree analysis and ACS reporting.",
    },
    ScientificCitation {
        domain: CitationDomain::SmallMoleculeDosy,
        authors: "Wu, D. H.; Chen, A. D.; Johnson, C. S. Jr.",
        year: 1995,
        title: "An improved diffusion-ordered spectroscopy experiment incorporating bipolar-gradient pulses",
        journal: "J. Magn. Reson. A",
        volume_pages: "115, 260-264",
        used_for: "BPP-LED bipolar-pulse-pair gradient factor modeling in pseudo-2D DOSY.",
    },
    ScientificCitation {
        domain: CitationDomain::SmallMoleculeDosy,
        authors: "Gottlieb, H. E.; Kotlyar, V.; Nudelman, A.",
        year: 1997,
        title: "NMR chemical shifts of common laboratory solvents as trace impurities",
        journal: "J. Org. Chem.",
        volume_pages: "62, 7512-7515",
        used_for: "Automated residual deuterated solvent and trace laboratory impurity annotation in 1D spectra.",
    },
    ScientificCitation {
        domain: CitationDomain::SmallMoleculeDosy,
        authors: "Johnson, C. S. Jr.",
        year: 1999,
        title: "Diffusion ordered nuclear magnetic resonance spectroscopy: Principles and applications",
        journal: "Prog. Nucl. Magn. Reson. Spectrosc.",
        volume_pages: "34, 203-256",
        used_for: "Pseudo-2D DOSY map reconstruction and multi-component mixture resolution.",
    },
    ScientificCitation {
        domain: CitationDomain::SmallMoleculeDosy,
        authors: "Hoye, T. R.; Zhao, H.",
        year: 2002,
        title: "A method for easily determining coupling constant values: An addendum to 'A practical guide to first-order multiplet analysis in 1H NMR spectroscopy'",
        journal: "J. Org. Chem.",
        volume_pages: "67, 4014-4016",
        used_for: "Recursive intensity peeling for coupled 1H multiplet J-constant extraction.",
    },
    // 6. MetabolomicsChemometrics
    ScientificCitation {
        domain: CitationDomain::MetabolomicsChemometrics,
        authors: "Trygg, J.; Wold, S.",
        year: 2002,
        title: "Orthogonal projections to latent structures (O-PLS)",
        journal: "J. Chemom.",
        volume_pages: "16, 119-128",
        used_for: "Supervised OPLS-DA predictive/orthogonal latent decomposition in metabolomics cohorts.",
    },
    ScientificCitation {
        domain: CitationDomain::MetabolomicsChemometrics,
        authors: "Cloarec, O.; Dumas, M. E.; Craig, A.; Barton, R. H.; Trygg, J.; Hudson, J.; Blancher, C.; Gauguier, D.; Lindon, J. C.; Holmes, E.; Nicholson, J.",
        year: 2005,
        title: "Statistical total correlation spectroscopy: An exploratory approach for latent biomarker identification from metabolic 1H NMR data sets",
        journal: "Anal. Chem.",
        volume_pages: "77, 1282-1289",
        used_for: "STOCSY structural covariance / correlation driver traces across 1D cohorts.",
    },
    ScientificCitation {
        domain: CitationDomain::MetabolomicsChemometrics,
        authors: "Dieterle, F.; Ross, A.; Schlotterbeck, G.; Senn, H.",
        year: 2006,
        title: "Probabilistic quotient normalization as robust method to account for dilution of complex biological mixtures",
        journal: "Anal. Chem.",
        volume_pages: "78, 4281-4290",
        used_for: "Probabilistic Quotient Normalization (PQN) across biological NMR sample cohorts.",
    },
    ScientificCitation {
        domain: CitationDomain::MetabolomicsChemometrics,
        authors: "Wiklund, S.; Johansson, E.; Sjostrom, L.; Mellerowicz, E. J.; Edlund, U.; Shockcor, J. P.; Gottfries, J.; Moritz, T.; Trygg, J.",
        year: 2008,
        title: "Visualization of GC/TOF-MS-based metabolomics data for identification of biochemically interesting compounds using OPLS class models",
        journal: "Anal. Chem.",
        volume_pages: "80, 115-122",
        used_for: "OPLS-DA S-plot covariance vs correlation biomarker ranking.",
    },
    ScientificCitation {
        domain: CitationDomain::MetabolomicsChemometrics,
        authors: "Savorani, F.; Tomasi, G.; Engelsen, S. B.",
        year: 2010,
        title: "icoshift: A versatile tool for the rapid alignment of 1D NMR spectra",
        journal: "J. Magn. Reson.",
        volume_pages: "202, 190-202",
        used_for: "Interval FFT cross-correlation alignment for 1D NMR arrays and kinetics stacks.",
    },
];

/// State for the in-app `About FERRIC & Scientific References` modal window.
#[derive(Clone, Debug, Default)]
pub struct AboutReferencesModal {
    /// Whether the modal window is currently open.
    pub open: bool,
    /// Case-insensitive live search query across citations and open-data acknowledgments.
    pub search_query: String,
    /// Optional domain filter (`None` = show all domains).
    pub selected_domain: Option<CitationDomain>,
    /// Most recently copied citation text (used for status feedback and headless verification).
    pub last_copied_citation: Option<String>,
}

impl AboutReferencesModal {
    /// Create a new closed modal instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Open the modal window.
    pub fn open_modal(&mut self) {
        self.open = true;
    }

    /// Close the modal window.
    pub fn close_modal(&mut self) {
        self.open = false;
    }

    /// Return all verified citations matching `selected_domain` and `search_query`.
    pub fn filtered_citations(&self) -> Vec<&'static ScientificCitation> {
        let q = self.search_query.trim().to_ascii_lowercase();
        VERIFIED_CITATIONS
            .iter()
            .filter(|c| self.selected_domain.is_none_or(|dom| c.domain == dom))
            .filter(|c| c.matches_query(&q))
            .collect()
    }

    /// Render the searchable `egui::Window::new("About FERRIC & Scientific References")`.
    ///
    /// Returns `true` while open. Safe on both native desktop and `wasm32-unknown-unknown`.
    pub fn show(&mut self, ctx: &egui::Context) -> bool {
        if !self.open {
            return false;
        }
        let mut is_open = self.open;
        let version_banner = format!("FERRIC v{}", nmr_core::VERSION);

        egui::Window::new("About FERRIC & Scientific References")
            .id(egui::Id::unique("ferric_about_references_modal"))
            .open(&mut is_open)
            .collapsible(false)
            .resizable(true)
            .default_width(720.0)
            .default_height(560.0)
            .min_width(480.0)
            .min_height(360.0)
            .show(ctx, |ui| {
                let dark_mode = ui.visuals().dark_mode;
                let card_fill = if dark_mode {
                    egui::Color32::from_rgb(18, 26, 40)
                } else {
                    egui::Color32::from_rgb(248, 250, 252)
                };
                let card_border = if dark_mode {
                    egui::Color32::from_rgb(48, 64, 88)
                } else {
                    egui::Color32::from_rgb(203, 213, 225)
                };
                let badge_fill = if dark_mode {
                    egui::Color32::from_rgb(14, 52, 74)
                } else {
                    egui::Color32::from_rgb(224, 242, 254)
                };
                let badge_fg = if dark_mode {
                    egui::Color32::from_rgb(125, 211, 252)
                } else {
                    egui::Color32::from_rgb(3, 105, 161)
                };
                // 1. Version, Clean-Room Provenance & MIT Licence Banner
                egui::Frame::group(ui.style())
                    .fill(card_fill)
                    .stroke(egui::Stroke::new(1.0, card_border))
                    .corner_radius(6.0)
                    .inner_margin(egui::Margin::symmetric(10, 8))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::new(&version_banner)
                                    .size(TYPE_HEADER_PX)
                                    .strong(),
                            );
                            ui.add_space(6.0);
                            egui::Frame::NONE
                                .fill(badge_fill)
                                .corner_radius(4.0)
                                .inner_margin(egui::Margin::symmetric(6, 2))
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new("MIT Licence")
                                            .size(TYPE_CAPTION_PX)
                                            .color(badge_fg)
                                            .strong(),
                                    );
                                });
                            egui::Frame::NONE
                                .fill(badge_fill)
                                .corner_radius(4.0)
                                .inner_margin(egui::Margin::symmetric(6, 2))
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new("Clean-Room Rust + WebGPU")
                                            .size(TYPE_CAPTION_PX)
                                            .color(badge_fg),
                                    );
                                });
                        });
                        ui.add_space(3.0);
                        ui.label(
                            egui::RichText::new(
                                "FID-to-structure Engine for Relaxation, Resonance assignment, Integration & Chemometrics. Implemented strictly from peer-reviewed literature and open data standards (NMR-STAR v3, NEF v1.1, JCAMP-DX).",
                            )
                            .size(TYPE_BODY_PX),
                        );
                    });

                ui.add_space(6.0);

                // 2. Open Data Acknowledgments (BMRB CC0, PDB, AlphaFold DB CC-BY-4.0, NP-MRD CC-BY-4.0)
                egui::CollapsingHeader::new(
                    egui::RichText::new(
                        "Open Data Acknowledgments (BMRB CC0, PDB, AlphaFold DB CC-BY-4.0, NP-MRD CC-BY-4.0)",
                    )
                    .size(TYPE_BODY_PX)
                    .strong(),
                )
                .id_salt("about_open_data_acks")
                .default_open(true)
                .show(ui, |ui| {
                    for ack in OPEN_DATA_ACKNOWLEDGMENTS {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(egui::RichText::new(ack.name).size(TYPE_BODY_PX).strong());
                            ui.label(
                                egui::RichText::new(format!("[{}]", ack.licence))
                                    .size(TYPE_CAPTION_PX)
                                    .monospace()
                                    .color(badge_fg),
                            );
                        });
                        ui.label(
                            egui::RichText::new(format!("{} ({})", ack.description, ack.citation))
                                .size(TYPE_CAPTION_PX)
                                .weak(),
                        );
                        ui.add_space(3.0);
                    }
                });

                ui.add_space(6.0);

                // 3. Live Search Bar + Domain Folder Tabs
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Filter References:").size(TYPE_BODY_PX));
                    ui.add(
                        egui::TextEdit::singleline(&mut self.search_query)
                            .id_salt("about_refs_search_input")
                            .hint_text("Search author, year, journal, or method (e.g. Lipari, Bax, icoshift, DOSY)...")
                            .desired_width(320.0),
                    );
                    if !self.search_query.is_empty()
                        && ui
                            .button(egui::RichText::new("Clear").size(TYPE_BODY_PX))
                            .clicked()
                    {
                        self.search_query.clear();
                    }
                });

                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    if draw_folder_tab(ui, "All", self.selected_domain.is_none()).clicked() {
                        self.selected_domain = None;
                    }
                    for dom in CitationDomain::ALL {
                        if draw_folder_tab(
                            ui,
                            dom.short_label(),
                            self.selected_domain == Some(dom),
                        )
                        .clicked()
                        {
                            self.selected_domain = Some(dom);
                        }
                    }
                });

                let filtered = self.filtered_citations();

                // 4. Domain-Grouped Verified Citations List
                egui::Frame::group(ui.style())
                    .fill(card_fill)
                    .stroke(egui::Stroke::new(1.0, card_border))
                    .inner_margin(egui::Margin::symmetric(8, 6))
                    .show(ui, |ui| {
                        let scroll_h = (ui.available_height() - 34.0).max(160.0);
                        egui::ScrollArea::vertical()
                            .id_salt("about_refs_scroll")
                            .max_height(scroll_h)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                if filtered.is_empty() {
                                    ui.label(
                                        egui::RichText::new(
                                            "No verified citations matched your search filter.",
                                        )
                                        .size(TYPE_BODY_PX)
                                        .weak(),
                                    );
                                    return;
                                }
                                for dom in CitationDomain::ALL {
                                    let in_dom: Vec<&ScientificCitation> = filtered
                                        .iter()
                                        .copied()
                                        .filter(|c| c.domain == dom)
                                        .collect();
                                    if in_dom.is_empty() {
                                        continue;
                                    }
                                    ui.add_space(3.0);
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{} ({})",
                                            dom.title(),
                                            in_dom.len()
                                        ))
                                        .size(TYPE_BODY_PX)
                                        .strong()
                                        .color(badge_fg),
                                    );
                                    ui.separator();
                                    for (idx, cit) in in_dom.iter().enumerate() {
                                        let formatted = cit.format_reference();
                                        ui.push_id((dom, idx), |ui| {
                                            ui.horizontal(|ui| {
                                                ui.vertical(|ui| {
                                                    ui.set_max_width(
                                                        (ui.available_width() - 108.0).max(200.0),
                                                    );
                                                    ui.label(
                                                        egui::RichText::new(&formatted)
                                                            .size(TYPE_BODY_PX),
                                                    );
                                                    ui.label(
                                                        egui::RichText::new(cit.used_for)
                                                            .size(TYPE_CAPTION_PX)
                                                            .weak(),
                                                    );
                                                });
                                                ui.with_layout(
                                                    egui::Layout::right_to_left(
                                                        egui::Align::Center,
                                                    ),
                                                    |ui| {
                                                        if ui
                                                            .button(
                                                                egui::RichText::new("Copy Citation")
                                                                    .size(TYPE_CAPTION_PX),
                                                            )
                                                            .on_hover_text(
                                                                "Copy formatted reference to clipboard",
                                                            )
                                                            .clicked()
                                                        {
                                                            ctx.copy_text(formatted.clone());
                                                            self.last_copied_citation =
                                                                Some(formatted);
                                                        }
                                                    },
                                                );
                                            });
                                        });
                                        ui.add_space(4.0);
                                    }
                                }
                            });
                    });

                ui.add_space(4.0);

                // 5. Footer: Count + Copy All Filtered + Status Feedback
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "Showing {} of {} verified NMR references",
                            filtered.len(),
                            VERIFIED_CITATIONS.len()
                        ))
                        .size(TYPE_CAPTION_PX)
                        .weak(),
                    );
                    if !filtered.is_empty()
                        && ui
                            .button(
                                egui::RichText::new("Copy All Filtered").size(TYPE_CAPTION_PX),
                            )
                            .clicked()
                    {
                        let joined = filtered
                            .iter()
                            .map(|c| c.format_reference())
                            .collect::<Vec<_>>()
                            .join("\n");
                        ctx.copy_text(joined.clone());
                        self.last_copied_citation = Some(joined);
                    }
                    if self.last_copied_citation.is_some() {
                        ui.label(
                            egui::RichText::new("Copied citation to clipboard.")
                                .size(TYPE_CAPTION_PX)
                                .color(badge_fg),
                        );
                    }
                });
            });

        self.open = is_open;
        self.open
    }
}
