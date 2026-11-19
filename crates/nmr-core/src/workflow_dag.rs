//! Clean-room typed node-and-port NMR workflow `DAG` compiler, static validator,
//! content-addressed execution manifest (`ferric-dag-v1`), and deterministic dry-run
//! caching executor (`T22.11`).
//!
//! Provenance: Original clean-room typed dataflow graph compiler and content-addressed
//! workflow manifest design for `FERRIC`. Uses Kahn's topological sort algorithm
//! (Kahn, A. B., *Commun. ACM* 1962, 5(11), 558–562) with deterministic lexicographical
//! tie-breaking and FIPS 180-4 SHA-256 content addressing via [`crate::torsion_db::sha256_hex`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::evaluate::to_reference;
use crate::evo_graph::{assign_evo_graph_from_peaks, EvoGraphConfig};
use crate::metabolomics::{
    export_bucket_table_csv, export_study_summary_json, generate_demo_biofluid_cohort,
    preprocess_cohort_1d, run_metabolomics_pipeline, CohortSample1D, MetabolomicsPipelineConfig,
    MetabolomicsStudyResult,
};
use crate::nef::{read_nef_str, write_nef_str, NefDocument};
use crate::nmrstar::{read_star_str, Reference};
use crate::peaks::{read_peaks_str, Peak};
use crate::picker::{pick_hsqc_2d, PickerConfig};
use crate::priors::{one_to_three, Priors};
use crate::report::{generate_quality_report_html, predict_csi_dihedrals, ReportInput};
use crate::simulate::{render_hsqc_2d, simulate, SimConfig, SpectrumRenderConfig};
use crate::solver::{assign, AnnealConfig, ScoreConfig};
use crate::spectrum::Spectrum2D;
use crate::spinsystems::{build_spin_systems, Tolerances};
use crate::torsion_db::sha256_hex;

const EMBEDDED_DAG_DEMO_UBIQUITIN_STAR: &str =
    include_str!("../../../data/fixtures/synthetic_ubiquitin.str");

/// Canonical manifest schema identifier.
pub const DAG_MANIFEST_SCHEMA: &str = "ferric-dag-v1";

/// Maximum permitted nodes in a single workflow `DAG` (hostile input cap).
pub const MAX_DAG_NODES: usize = 128;

/// Maximum permitted edges in a single workflow `DAG` (hostile input cap).
pub const MAX_DAG_EDGES: usize = 512;

/// Maximum permitted serialized JSON payload size in bytes (`1 MiB`).
pub const MAX_DAG_JSON_BYTES: usize = 1024 * 1024;

/// Maximum permitted length of a node identifier.
pub const MAX_NODE_ID_LEN: usize = 64;

/// Maximum permitted parameter entries per node.
pub const MAX_PARAM_KEYS: usize = 64;

/// Maximum permitted length of a string parameter or label.
pub const MAX_STRING_PARAM_LEN: usize = 4096;

/// Maximum JSON nesting depth allowed in node parameter values.
const MAX_PARAM_JSON_DEPTH: usize = 8;

/// Strongly typed NMR data port categories enforced statically on every `DAG` edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortType {
    RawFid,
    ProcessedSpectrum1D,
    ProcessedSpectrum2D,
    ProcessedSpectrum3D,
    PeakList,
    SpinSystems,
    BackboneAssignments,
    RestraintSet,
    StructureModel,
    RelaxationRates,
    ExchangeProfiles,
    ChemometricsMatrix,
    ReportBundle,
}

impl PortType {
    /// All supported strongly typed NMR ports in canonical order.
    pub const ALL: [PortType; 13] = [
        PortType::RawFid,
        PortType::ProcessedSpectrum1D,
        PortType::ProcessedSpectrum2D,
        PortType::ProcessedSpectrum3D,
        PortType::PeakList,
        PortType::SpinSystems,
        PortType::BackboneAssignments,
        PortType::RestraintSet,
        PortType::StructureModel,
        PortType::RelaxationRates,
        PortType::ExchangeProfiles,
        PortType::ChemometricsMatrix,
        PortType::ReportBundle,
    ];

    /// Human-readable display name.
    pub const fn display_name(self) -> &'static str {
        match self {
            PortType::RawFid => "Raw FID",
            PortType::ProcessedSpectrum1D => "1D Spectrum",
            PortType::ProcessedSpectrum2D => "2D Spectrum",
            PortType::ProcessedSpectrum3D => "3D Spectrum",
            PortType::PeakList => "Peak List",
            PortType::SpinSystems => "Spin Systems",
            PortType::BackboneAssignments => "Backbone Shifts",
            PortType::RestraintSet => "Restraint Set",
            PortType::StructureModel => "Structure Model",
            PortType::RelaxationRates => "Relaxation Rates",
            PortType::ExchangeProfiles => "Exchange Profiles",
            PortType::ChemometricsMatrix => "Chemometrics Matrix",
            PortType::ReportBundle => "Report Bundle",
        }
    }

    /// Compact ASCII tag for port badges.
    pub const fn short_tag(self) -> &'static str {
        match self {
            PortType::RawFid => "FID",
            PortType::ProcessedSpectrum1D => "1D",
            PortType::ProcessedSpectrum2D => "2D",
            PortType::ProcessedSpectrum3D => "3D",
            PortType::PeakList => "PK",
            PortType::SpinSystems => "SS",
            PortType::BackboneAssignments => "BB",
            PortType::RestraintSet => "RST",
            PortType::StructureModel => "MOD",
            PortType::RelaxationRates => "RLX",
            PortType::ExchangeProfiles => "EX",
            PortType::ChemometricsMatrix => "CHM",
            PortType::ReportBundle => "RPT",
        }
    }

    /// Canonical snake_case identifier used in content-addressed hashing.
    pub const fn canonical_id(self) -> &'static str {
        match self {
            PortType::RawFid => "raw_fid",
            PortType::ProcessedSpectrum1D => "processed_spectrum_1d",
            PortType::ProcessedSpectrum2D => "processed_spectrum_2d",
            PortType::ProcessedSpectrum3D => "processed_spectrum_3d",
            PortType::PeakList => "peak_list",
            PortType::SpinSystems => "spin_systems",
            PortType::BackboneAssignments => "backbone_assignments",
            PortType::RestraintSet => "restraint_set",
            PortType::StructureModel => "structure_model",
            PortType::RelaxationRates => "relaxation_rates",
            PortType::ExchangeProfiles => "exchange_profiles",
            PortType::ChemometricsMatrix => "chemometrics_matrix",
            PortType::ReportBundle => "report_bundle",
        }
    }
}

impl fmt::Display for PortType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.display_name())
    }
}

/// Direction of a port on a workflow stage node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortDirection {
    Input,
    Output,
}

impl fmt::Display for PortDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PortDirection::Input => f.write_str("input"),
            PortDirection::Output => f.write_str("output"),
        }
    }
}

/// Static port specification belonging to a [`NodeKind`] template.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortSpec {
    pub name: &'static str,
    pub port_type: PortType,
    pub required: bool,
}

/// Built-in NMR processing, assignment, dynamics, structure, and reporting stage templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    IngestBrukerVarian,
    ProcessFidNus,
    PeakPick,
    DeconvolveNd,
    BackboneAssign,
    SidechainAssign,
    DynamicsRelax,
    ExchangeCestCpmg,
    Titration2d,
    StructureFoldDock,
    MetabolomicsChemometrics,
    ExportNefReport,
}

impl NodeKind {
    /// All built-in NMR stage templates in canonical order.
    pub const ALL: [NodeKind; 12] = [
        NodeKind::IngestBrukerVarian,
        NodeKind::ProcessFidNus,
        NodeKind::PeakPick,
        NodeKind::DeconvolveNd,
        NodeKind::BackboneAssign,
        NodeKind::SidechainAssign,
        NodeKind::DynamicsRelax,
        NodeKind::ExchangeCestCpmg,
        NodeKind::Titration2d,
        NodeKind::StructureFoldDock,
        NodeKind::MetabolomicsChemometrics,
        NodeKind::ExportNefReport,
    ];

    /// Canonical snake_case identifier used in content-addressed hashing.
    pub const fn canonical_id(self) -> &'static str {
        match self {
            NodeKind::IngestBrukerVarian => "ingest_bruker_varian",
            NodeKind::ProcessFidNus => "process_fid_nus",
            NodeKind::PeakPick => "peak_pick",
            NodeKind::DeconvolveNd => "deconvolve_nd",
            NodeKind::BackboneAssign => "backbone_assign",
            NodeKind::SidechainAssign => "sidechain_assign",
            NodeKind::DynamicsRelax => "dynamics_relax",
            NodeKind::ExchangeCestCpmg => "exchange_cest_cpmg",
            NodeKind::Titration2d => "titration_2d",
            NodeKind::StructureFoldDock => "structure_fold_dock",
            NodeKind::MetabolomicsChemometrics => "metabolomics_chemometrics",
            NodeKind::ExportNefReport => "export_nef_report",
        }
    }

    /// Concise human-readable stage title.
    pub const fn display_name(self) -> &'static str {
        match self {
            NodeKind::IngestBrukerVarian => "Ingest Bruker/Varian",
            NodeKind::ProcessFidNus => "Process FID / NUS",
            NodeKind::PeakPick => "Peak Pick",
            NodeKind::DeconvolveNd => "Deconvolve nD",
            NodeKind::BackboneAssign => "Backbone Assign",
            NodeKind::SidechainAssign => "Sidechain Assign",
            NodeKind::DynamicsRelax => "Dynamics Relax",
            NodeKind::ExchangeCestCpmg => "Exchange CEST/CPMG",
            NodeKind::Titration2d => "Titration 2D",
            NodeKind::StructureFoldDock => "Structure Fold/Dock",
            NodeKind::MetabolomicsChemometrics => "Metabolomics PCA/PLS",
            NodeKind::ExportNefReport => "Export NEF Report",
        }
    }

    /// Short prefix for auto-generated node identifiers.
    pub const fn id_prefix(self) -> &'static str {
        match self {
            NodeKind::IngestBrukerVarian => "ingest",
            NodeKind::ProcessFidNus => "process",
            NodeKind::PeakPick => "pick",
            NodeKind::DeconvolveNd => "deconv",
            NodeKind::BackboneAssign => "backbone",
            NodeKind::SidechainAssign => "sidechain",
            NodeKind::DynamicsRelax => "relax",
            NodeKind::ExchangeCestCpmg => "exchange",
            NodeKind::Titration2d => "titration",
            NodeKind::StructureFoldDock => "structure",
            NodeKind::MetabolomicsChemometrics => "chemometrics",
            NodeKind::ExportNefReport => "export",
        }
    }

    /// Typed input ports exposed by this stage template.
    pub const fn input_ports(self) -> &'static [PortSpec] {
        match self {
            NodeKind::IngestBrukerVarian => &[],
            NodeKind::ProcessFidNus => &[PortSpec {
                name: "raw_fid",
                port_type: PortType::RawFid,
                required: true,
            }],
            NodeKind::PeakPick => &[
                PortSpec {
                    name: "spectrum_1d",
                    port_type: PortType::ProcessedSpectrum1D,
                    required: false,
                },
                PortSpec {
                    name: "spectrum_2d",
                    port_type: PortType::ProcessedSpectrum2D,
                    required: false,
                },
                PortSpec {
                    name: "spectrum_3d",
                    port_type: PortType::ProcessedSpectrum3D,
                    required: false,
                },
            ],
            NodeKind::DeconvolveNd => &[
                PortSpec {
                    name: "seed_peaks",
                    port_type: PortType::PeakList,
                    required: true,
                },
                PortSpec {
                    name: "spectrum_2d",
                    port_type: PortType::ProcessedSpectrum2D,
                    required: false,
                },
                PortSpec {
                    name: "spectrum_3d",
                    port_type: PortType::ProcessedSpectrum3D,
                    required: false,
                },
            ],
            NodeKind::BackboneAssign => &[
                PortSpec {
                    name: "peaks",
                    port_type: PortType::PeakList,
                    required: true,
                },
                PortSpec {
                    name: "spin_systems",
                    port_type: PortType::SpinSystems,
                    required: false,
                },
            ],
            NodeKind::SidechainAssign => &[
                PortSpec {
                    name: "assignments",
                    port_type: PortType::BackboneAssignments,
                    required: true,
                },
                PortSpec {
                    name: "peaks",
                    port_type: PortType::PeakList,
                    required: false,
                },
            ],
            NodeKind::DynamicsRelax => &[
                PortSpec {
                    name: "peaks",
                    port_type: PortType::PeakList,
                    required: true,
                },
                PortSpec {
                    name: "assignments",
                    port_type: PortType::BackboneAssignments,
                    required: false,
                },
            ],
            NodeKind::ExchangeCestCpmg => &[
                PortSpec {
                    name: "peaks",
                    port_type: PortType::PeakList,
                    required: true,
                },
                PortSpec {
                    name: "assignments",
                    port_type: PortType::BackboneAssignments,
                    required: false,
                },
            ],
            NodeKind::Titration2d => &[
                PortSpec {
                    name: "spectrum_2d",
                    port_type: PortType::ProcessedSpectrum2D,
                    required: true,
                },
                PortSpec {
                    name: "assignments",
                    port_type: PortType::BackboneAssignments,
                    required: false,
                },
            ],
            NodeKind::StructureFoldDock => &[
                PortSpec {
                    name: "restraints",
                    port_type: PortType::RestraintSet,
                    required: true,
                },
                PortSpec {
                    name: "assignments",
                    port_type: PortType::BackboneAssignments,
                    required: false,
                },
            ],
            NodeKind::MetabolomicsChemometrics => &[PortSpec {
                name: "spectrum_1d",
                port_type: PortType::ProcessedSpectrum1D,
                required: true,
            }],
            NodeKind::ExportNefReport => &[
                PortSpec {
                    name: "assignments",
                    port_type: PortType::BackboneAssignments,
                    required: false,
                },
                PortSpec {
                    name: "structure",
                    port_type: PortType::StructureModel,
                    required: false,
                },
                PortSpec {
                    name: "rates",
                    port_type: PortType::RelaxationRates,
                    required: false,
                },
                PortSpec {
                    name: "profiles",
                    port_type: PortType::ExchangeProfiles,
                    required: false,
                },
                PortSpec {
                    name: "matrix",
                    port_type: PortType::ChemometricsMatrix,
                    required: false,
                },
                PortSpec {
                    name: "peaks",
                    port_type: PortType::PeakList,
                    required: false,
                },
                PortSpec {
                    name: "restraints",
                    port_type: PortType::RestraintSet,
                    required: false,
                },
            ],
        }
    }

    /// Typed output ports exposed by this stage template.
    pub const fn output_ports(self) -> &'static [PortSpec] {
        match self {
            NodeKind::IngestBrukerVarian => &[
                PortSpec {
                    name: "raw_fid",
                    port_type: PortType::RawFid,
                    required: false,
                },
                PortSpec {
                    name: "spectrum_1d",
                    port_type: PortType::ProcessedSpectrum1D,
                    required: false,
                },
                PortSpec {
                    name: "spectrum_2d",
                    port_type: PortType::ProcessedSpectrum2D,
                    required: false,
                },
                PortSpec {
                    name: "spectrum_3d",
                    port_type: PortType::ProcessedSpectrum3D,
                    required: false,
                },
            ],
            NodeKind::ProcessFidNus => &[
                PortSpec {
                    name: "spectrum_1d",
                    port_type: PortType::ProcessedSpectrum1D,
                    required: false,
                },
                PortSpec {
                    name: "spectrum_2d",
                    port_type: PortType::ProcessedSpectrum2D,
                    required: false,
                },
                PortSpec {
                    name: "spectrum_3d",
                    port_type: PortType::ProcessedSpectrum3D,
                    required: false,
                },
            ],
            NodeKind::PeakPick => &[PortSpec {
                name: "peaks",
                port_type: PortType::PeakList,
                required: false,
            }],
            NodeKind::DeconvolveNd => &[
                PortSpec {
                    name: "peaks",
                    port_type: PortType::PeakList,
                    required: false,
                },
                PortSpec {
                    name: "spin_systems",
                    port_type: PortType::SpinSystems,
                    required: false,
                },
            ],
            NodeKind::BackboneAssign => &[
                PortSpec {
                    name: "spin_systems",
                    port_type: PortType::SpinSystems,
                    required: false,
                },
                PortSpec {
                    name: "assignments",
                    port_type: PortType::BackboneAssignments,
                    required: false,
                },
            ],
            NodeKind::SidechainAssign => &[
                PortSpec {
                    name: "assignments",
                    port_type: PortType::BackboneAssignments,
                    required: false,
                },
                PortSpec {
                    name: "restraints",
                    port_type: PortType::RestraintSet,
                    required: false,
                },
            ],
            NodeKind::DynamicsRelax => &[PortSpec {
                name: "rates",
                port_type: PortType::RelaxationRates,
                required: false,
            }],
            NodeKind::ExchangeCestCpmg => &[PortSpec {
                name: "profiles",
                port_type: PortType::ExchangeProfiles,
                required: false,
            }],
            NodeKind::Titration2d => &[
                PortSpec {
                    name: "peaks",
                    port_type: PortType::PeakList,
                    required: false,
                },
                PortSpec {
                    name: "restraints",
                    port_type: PortType::RestraintSet,
                    required: false,
                },
            ],
            NodeKind::StructureFoldDock => &[PortSpec {
                name: "structure",
                port_type: PortType::StructureModel,
                required: false,
            }],
            NodeKind::MetabolomicsChemometrics => &[
                PortSpec {
                    name: "matrix",
                    port_type: PortType::ChemometricsMatrix,
                    required: false,
                },
                PortSpec {
                    name: "peaks",
                    port_type: PortType::PeakList,
                    required: false,
                },
            ],
            NodeKind::ExportNefReport => &[PortSpec {
                name: "report",
                port_type: PortType::ReportBundle,
                required: false,
            }],
        }
    }

    /// Whether a stage with only optional polymorphic inputs requires at least one connected input.
    pub const fn requires_any_input(self) -> bool {
        matches!(self, NodeKind::PeakPick | NodeKind::ExportNefReport)
    }

    /// Look up an input port specification by name.
    pub fn find_input_port(self, name: &str) -> Option<PortSpec> {
        self.input_ports().iter().copied().find(|p| p.name == name)
    }

    /// Look up an output port specification by name.
    pub fn find_output_port(self, name: &str) -> Option<PortSpec> {
        self.output_ports().iter().copied().find(|p| p.name == name)
    }

    /// Default validated parameter map for this stage template.
    pub fn default_params(self) -> BTreeMap<String, serde_json::Value> {
        let mut p = BTreeMap::new();
        match self {
            NodeKind::IngestBrukerVarian => {
                p.insert("vendor".to_string(), serde_json::json!("bruker"));
                p.insert(
                    "dataset_path".to_string(),
                    serde_json::json!("datasets/nmr_bundle"),
                );
            }
            NodeKind::ProcessFidNus => {
                p.insert("apodization".to_string(), serde_json::json!("cosine_bell"));
                p.insert("nus_method".to_string(), serde_json::json!("ist"));
                p.insert("zero_fill_factor".to_string(), serde_json::json!(2));
            }
            NodeKind::PeakPick => {
                p.insert("snr_threshold".to_string(), serde_json::json!(5.0));
                p.insert("mode".to_string(), serde_json::json!("two_pass"));
            }
            NodeKind::DeconvolveNd => {
                p.insert("lineshape".to_string(), serde_json::json!("voigt"));
                p.insert("max_iterations".to_string(), serde_json::json!(64));
            }
            NodeKind::BackboneAssign => {
                p.insert("solver".to_string(), serde_json::json!("evo_graph"));
                p.insert("seeds".to_string(), serde_json::json!(5));
            }
            NodeKind::SidechainAssign => {
                p.insert("include_aromatic".to_string(), serde_json::json!(true));
                p.insert("ppm_tolerance".to_string(), serde_json::json!(0.04));
            }
            NodeKind::DynamicsRelax => {
                p.insert("model".to_string(), serde_json::json!("lipari_szabo"));
                p.insert("field_mhz".to_string(), serde_json::json!(800.0));
            }
            NodeKind::ExchangeCestCpmg => {
                p.insert("exchange_model".to_string(), serde_json::json!("two_state"));
                p.insert("b1_hz".to_string(), serde_json::json!(25.0));
            }
            NodeKind::Titration2d => {
                p.insert("csp_weight_n".to_string(), serde_json::json!(0.154));
                p.insert("binding_model".to_string(), serde_json::json!("one_site"));
            }
            NodeKind::StructureFoldDock => {
                p.insert("ensemble_size".to_string(), serde_json::json!(20));
                p.insert("torsion_priors".to_string(), serde_json::json!(true));
            }
            NodeKind::MetabolomicsChemometrics => {
                p.insert("normalization".to_string(), serde_json::json!("pqn"));
                p.insert("n_components".to_string(), serde_json::json!(4));
            }
            NodeKind::ExportNefReport => {
                p.insert("format".to_string(), serde_json::json!("nef_v1_1_html"));
                p.insert(
                    "output_path".to_string(),
                    serde_json::json!("reports/ferric_bundle.nef"),
                );
            }
        }
        p
    }
}

/// A single stage node in a [`WorkflowDag`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DagNode {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
    /// Visual canvas coordinates `[x, y]`; excluded from content-addressed hashing.
    #[serde(default)]
    pub position: [f32; 2],
}

impl DagNode {
    /// Construct a node with default parameters for `kind`.
    pub fn new(id: impl Into<String>, kind: NodeKind, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind,
            label: label.into(),
            params: kind.default_params(),
            position: [40.0, 40.0],
        }
    }

    /// Builder helper to set visual canvas coordinates.
    pub fn with_position(mut self, x: f32, y: f32) -> Self {
        self.position = [x, y];
        self
    }

    /// Builder helper to insert or override a parameter.
    pub fn with_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.params.insert(key.into(), value);
        self
    }
}

/// A directed typed edge connecting an output port on `from_node` to an input port on `to_node`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DagEdge {
    pub from_node: String,
    pub from_port: String,
    pub to_node: String,
    pub to_port: String,
}

impl DagEdge {
    /// Construct a new directed edge.
    pub fn new(
        from_node: impl Into<String>,
        from_port: impl Into<String>,
        to_node: impl Into<String>,
        to_port: impl Into<String>,
    ) -> Self {
        Self {
            from_node: from_node.into(),
            from_port: from_port.into(),
            to_node: to_node.into(),
            to_port: to_port.into(),
        }
    }
}

/// Declarative typed NMR workflow graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowDag {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub nodes: Vec<DagNode>,
    pub edges: Vec<DagEdge>,
}

/// Fail-closed validation, serialization, and execution error variants.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DagError {
    EmptyDag,
    TooManyNodes {
        count: usize,
        max: usize,
    },
    TooManyEdges {
        count: usize,
        max: usize,
    },
    JsonTooLarge {
        bytes: usize,
        max: usize,
    },
    JsonParseError {
        message: String,
    },
    InvalidNodeId {
        node_id: String,
        reason: String,
    },
    DuplicateNodeId {
        node_id: String,
    },
    InvalidParam {
        node_id: String,
        key: String,
        reason: String,
    },
    MissingNode {
        edge: Box<DagEdge>,
        missing_node_id: String,
    },
    UnknownPort {
        edge: Box<DagEdge>,
        node_id: String,
        port: String,
        direction: PortDirection,
    },
    PortTypeMismatch {
        edge: Box<DagEdge>,
        from_type: PortType,
        to_type: PortType,
    },
    DuplicateEdge {
        edge: Box<DagEdge>,
    },
    InputPortAlreadyConnected {
        node_id: String,
        port: String,
    },
    MissingRequiredInput {
        node_id: String,
        port: String,
    },
    CycleDetected {
        cycle_nodes: Vec<String>,
    },
    ManifestIntegrityMismatch {
        reason: String,
    },
    ExecutionFailed {
        node_id: String,
        reason: String,
    },
}

impl fmt::Display for DagError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DagError::EmptyDag => write!(f, "Workflow DAG must contain at least one node"),
            DagError::TooManyNodes { count, max } => {
                write!(f, "Node count {count} exceeds maximum {max}")
            }
            DagError::TooManyEdges { count, max } => {
                write!(f, "Edge count {count} exceeds maximum {max}")
            }
            DagError::JsonTooLarge { bytes, max } => {
                write!(f, "JSON payload {bytes} bytes exceeds maximum {max} bytes")
            }
            DagError::JsonParseError { message } => {
                write!(f, "Invalid DAG JSON: {message}")
            }
            DagError::InvalidNodeId { node_id, reason } => {
                write!(f, "Invalid node ID '{node_id}': {reason}")
            }
            DagError::DuplicateNodeId { node_id } => {
                write!(f, "Duplicate node ID '{node_id}'")
            }
            DagError::InvalidParam {
                node_id,
                key,
                reason,
            } => {
                write!(f, "Invalid parameter '{key}' on node '{node_id}': {reason}")
            }
            DagError::MissingNode {
                edge,
                missing_node_id,
            } => {
                write!(
                    f,
                    "Edge {}:{} -> {}:{} references missing node '{}'",
                    edge.from_node, edge.from_port, edge.to_node, edge.to_port, missing_node_id
                )
            }
            DagError::UnknownPort {
                node_id,
                port,
                direction,
                ..
            } => {
                write!(f, "Unknown {direction} port '{port}' on node '{node_id}'")
            }
            DagError::PortTypeMismatch {
                edge,
                from_type,
                to_type,
            } => {
                write!(
                    f,
                    "Port type mismatch on {}:{} ({}) -> {}:{} ({})",
                    edge.from_node, edge.from_port, from_type, edge.to_node, edge.to_port, to_type
                )
            }
            DagError::DuplicateEdge { edge } => {
                write!(
                    f,
                    "Duplicate edge {}:{} -> {}:{}",
                    edge.from_node, edge.from_port, edge.to_node, edge.to_port
                )
            }
            DagError::InputPortAlreadyConnected { node_id, port } => {
                write!(f, "Input port '{node_id}:{port}' is already connected")
            }
            DagError::MissingRequiredInput { node_id, port } => {
                write!(f, "Node '{node_id}' missing required input '{port}'")
            }
            DagError::CycleDetected { cycle_nodes } => {
                write!(f, "Cycle detected involving: {}", cycle_nodes.join(", "))
            }
            DagError::ManifestIntegrityMismatch { reason } => {
                write!(f, "Manifest integrity mismatch: {reason}")
            }
            DagError::ExecutionFailed { node_id, reason } => {
                write!(f, "Execution failed at node '{node_id}': {reason}")
            }
        }
    }
}

impl std::error::Error for DagError {}

/// Result of static graph validation carrying canonical topological order and resolved edge types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedDag {
    pub topological_order: Vec<String>,
    pub edge_types: BTreeMap<DagEdge, PortType>,
}

/// Resolved upstream input binding recorded in a [`CompiledDagNode`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledInputBinding {
    pub to_port: String,
    pub from_node: String,
    pub from_port: String,
    pub port_type: PortType,
    pub upstream_sha256: String,
}

/// Single content-addressed stage entry in a compiled [`DagManifest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledDagNode {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    pub params: BTreeMap<String, serde_json::Value>,
    pub inputs: Vec<CompiledInputBinding>,
    pub node_sha256: String,
}

/// Deterministic content-addressed execution manifest (`schema: "ferric-dag-v1"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DagManifest {
    pub schema: String,
    pub workflow_name: String,
    pub manifest_sha256: String,
    pub topological_order: Vec<String>,
    pub nodes: Vec<CompiledDagNode>,
    pub edges: Vec<DagEdge>,
}

/// Cached output metadata keyed by `node_sha256`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedArtifact {
    pub node_sha256: String,
    pub kind: NodeKind,
    pub output_digest: String,
    pub summary: String,
}

/// Content-addressed artifact cache shared across incremental executions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DagArtifactCache {
    pub entries: BTreeMap<String, CachedArtifact>,
}

impl DagArtifactCache {
    /// Create an empty artifact cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached stage artifacts currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache contains zero artifacts.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a cached artifact by its content-addressed `node_sha256`.
    pub fn get(&self, node_sha256: &str) -> Option<&CachedArtifact> {
        self.entries.get(node_sha256)
    }

    /// Insert or update an artifact keyed by `artifact.node_sha256`.
    pub fn insert(&mut self, artifact: CachedArtifact) {
        self.entries.insert(artifact.node_sha256.clone(), artifact);
    }
}

/// Whether a stage in a [`DagExecutionTrace`] executed fresh or reused a cached artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Executed,
    CacheHit,
}

/// Per-stage provenance record emitted by [`execute_dag_manifest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DagStepProvenance {
    pub step_index: usize,
    pub node_id: String,
    pub kind: NodeKind,
    pub node_sha256: String,
    pub status: StepStatus,
    pub output_digest: String,
    pub summary: String,
}

/// Complete deterministic execution provenance trace for a [`DagManifest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DagExecutionTrace {
    pub schema: String,
    pub manifest_sha256: String,
    pub steps: Vec<DagStepProvenance>,
    pub executed_count: usize,
    pub cache_hit_count: usize,
}

fn contains_control_chars(s: &str) -> bool {
    s.chars().any(char::is_control)
}

fn validate_node_id(node_id: &str) -> Result<(), DagError> {
    if node_id.is_empty() {
        return Err(DagError::InvalidNodeId {
            node_id: node_id.to_string(),
            reason: "node ID must not be empty".to_string(),
        });
    }
    if node_id.len() > MAX_NODE_ID_LEN {
        return Err(DagError::InvalidNodeId {
            node_id: node_id.to_string(),
            reason: format!("node ID length exceeds {MAX_NODE_ID_LEN}"),
        });
    }
    if contains_control_chars(node_id) {
        return Err(DagError::InvalidNodeId {
            node_id: node_id.to_string(),
            reason: "node ID must not contain control characters".to_string(),
        });
    }
    if !node_id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(DagError::InvalidNodeId {
            node_id: node_id.to_string(),
            reason: "node ID must use only ASCII letters, digits, '_' or '-'".to_string(),
        });
    }
    Ok(())
}

fn is_path_like_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    lower.contains("path")
        || lower.contains("file")
        || lower.contains("dir")
        || lower.contains("uri")
        || lower.ends_with("_out")
        || lower == "output"
        || lower == "input"
}

fn validate_safe_relative_path(node_id: &str, key: &str, value: &str) -> Result<(), DagError> {
    if value.is_empty() {
        return Err(DagError::InvalidParam {
            node_id: node_id.to_string(),
            key: key.to_string(),
            reason: "manifest path must not be empty".to_string(),
        });
    }
    if value.starts_with('/') || value.starts_with('\\') {
        return Err(DagError::InvalidParam {
            node_id: node_id.to_string(),
            key: key.to_string(),
            reason: "absolute paths are forbidden in portable DAG manifests".to_string(),
        });
    }
    if value.contains('\\') {
        return Err(DagError::InvalidParam {
            node_id: node_id.to_string(),
            key: key.to_string(),
            reason: "backslashes are forbidden in portable DAG manifest paths".to_string(),
        });
    }
    let bytes = value.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return Err(DagError::InvalidParam {
            node_id: node_id.to_string(),
            key: key.to_string(),
            reason: "drive-letter paths are forbidden in portable DAG manifests".to_string(),
        });
    }
    if value.split('/').any(|seg| seg == "..") {
        return Err(DagError::InvalidParam {
            node_id: node_id.to_string(),
            key: key.to_string(),
            reason: "path traversal ('..') is forbidden in DAG manifests".to_string(),
        });
    }
    Ok(())
}

fn validate_param_value(
    node_id: &str,
    key: &str,
    value: &serde_json::Value,
    depth: usize,
    enforce_path_rules: bool,
) -> Result<(), DagError> {
    if depth > MAX_PARAM_JSON_DEPTH {
        return Err(DagError::InvalidParam {
            node_id: node_id.to_string(),
            key: key.to_string(),
            reason: format!("parameter nesting exceeds depth {MAX_PARAM_JSON_DEPTH}"),
        });
    }
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) => Ok(()),
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if !f.is_finite() {
                    return Err(DagError::InvalidParam {
                        node_id: node_id.to_string(),
                        key: key.to_string(),
                        reason: "numeric parameter must be finite".to_string(),
                    });
                }
            }
            Ok(())
        }
        serde_json::Value::String(s) => {
            if s.len() > MAX_STRING_PARAM_LEN {
                return Err(DagError::InvalidParam {
                    node_id: node_id.to_string(),
                    key: key.to_string(),
                    reason: format!("string parameter exceeds {MAX_STRING_PARAM_LEN} bytes"),
                });
            }
            if contains_control_chars(s) {
                return Err(DagError::InvalidParam {
                    node_id: node_id.to_string(),
                    key: key.to_string(),
                    reason: "string parameter contains control characters".to_string(),
                });
            }
            if enforce_path_rules {
                validate_safe_relative_path(node_id, key, s)?;
            }
            Ok(())
        }
        serde_json::Value::Array(items) => {
            if items.len() > MAX_PARAM_KEYS {
                return Err(DagError::InvalidParam {
                    node_id: node_id.to_string(),
                    key: key.to_string(),
                    reason: format!("array parameter exceeds {MAX_PARAM_KEYS} elements"),
                });
            }
            for item in items {
                validate_param_value(node_id, key, item, depth + 1, enforce_path_rules)?;
            }
            Ok(())
        }
        serde_json::Value::Object(map) => {
            if map.len() > MAX_PARAM_KEYS {
                return Err(DagError::InvalidParam {
                    node_id: node_id.to_string(),
                    key: key.to_string(),
                    reason: format!("object parameter exceeds {MAX_PARAM_KEYS} keys"),
                });
            }
            for (sub_k, sub_v) in map {
                if sub_k.is_empty()
                    || sub_k.len() > MAX_NODE_ID_LEN
                    || contains_control_chars(sub_k)
                {
                    return Err(DagError::InvalidParam {
                        node_id: node_id.to_string(),
                        key: key.to_string(),
                        reason: format!("invalid nested parameter key '{sub_k}'"),
                    });
                }
                let sub_is_path = enforce_path_rules || is_path_like_key(sub_k);
                validate_param_value(node_id, key, sub_v, depth + 1, sub_is_path)?;
            }
            Ok(())
        }
    }
}

fn validate_node_params(node: &DagNode) -> Result<(), DagError> {
    if !node.position[0].is_finite() || !node.position[1].is_finite() {
        return Err(DagError::InvalidParam {
            node_id: node.id.clone(),
            key: "position".to_string(),
            reason: "canvas coordinates must be finite".to_string(),
        });
    }
    if node.label.is_empty()
        || node.label.len() > MAX_STRING_PARAM_LEN
        || contains_control_chars(&node.label)
    {
        return Err(DagError::InvalidParam {
            node_id: node.id.clone(),
            key: "label".to_string(),
            reason: "node label must be non-empty and free of control characters".to_string(),
        });
    }
    if node.params.len() > MAX_PARAM_KEYS {
        return Err(DagError::InvalidParam {
            node_id: node.id.clone(),
            key: "*".to_string(),
            reason: format!("parameter count exceeds maximum {MAX_PARAM_KEYS}"),
        });
    }
    for (k, v) in &node.params {
        if k.is_empty() || k.len() > MAX_NODE_ID_LEN || contains_control_chars(k) {
            return Err(DagError::InvalidParam {
                node_id: node.id.clone(),
                key: k.clone(),
                reason: "parameter key must be 1..=64 chars without control characters".to_string(),
            });
        }
        validate_param_value(&node.id, k, v, 0, is_path_like_key(k))?;
    }

    // Domain-specific numeric sanity checks for common NMR parameters.
    match node.kind {
        NodeKind::ProcessFidNus => {
            if let Some(zf) = node.params.get("zero_fill_factor") {
                let valid = zf.as_u64().is_some_and(|n| (1..=16).contains(&n));
                if !valid {
                    return Err(DagError::InvalidParam {
                        node_id: node.id.clone(),
                        key: "zero_fill_factor".to_string(),
                        reason: "must be an integer in 1..=16".to_string(),
                    });
                }
            }
        }
        NodeKind::PeakPick => {
            if let Some(snr) = node.params.get("snr_threshold") {
                let valid = snr.as_f64().is_some_and(|f| f.is_finite() && f > 0.0);
                if !valid {
                    return Err(DagError::InvalidParam {
                        node_id: node.id.clone(),
                        key: "snr_threshold".to_string(),
                        reason: "must be a positive finite number".to_string(),
                    });
                }
            }
        }
        NodeKind::BackboneAssign => {
            if let Some(seeds) = node.params.get("seeds") {
                let valid = seeds.as_u64().is_some_and(|n| (1..=256).contains(&n));
                if !valid {
                    return Err(DagError::InvalidParam {
                        node_id: node.id.clone(),
                        key: "seeds".to_string(),
                        reason: "must be an integer in 1..=256".to_string(),
                    });
                }
            }
        }
        _ => {}
    }

    Ok(())
}

/// Extract deterministic sorted cycle-participating node IDs from nodes left unvisited by Kahn's sort.
fn extract_cycle_nodes(remaining: &BTreeSet<String>, edges: &[DagEdge]) -> Vec<String> {
    let mut active = remaining.clone();
    loop {
        let mut out_degree: BTreeMap<String, usize> =
            active.iter().map(|id| (id.clone(), 0usize)).collect();
        for edge in edges {
            if active.contains(&edge.from_node) && active.contains(&edge.to_node) {
                if let Some(deg) = out_degree.get_mut(&edge.from_node) {
                    *deg += 1;
                }
            }
        }
        let sinks: Vec<String> = out_degree
            .into_iter()
            .filter_map(|(id, deg)| (deg == 0).then_some(id))
            .collect();
        if sinks.is_empty() {
            break;
        }
        for sink in sinks {
            active.remove(&sink);
        }
    }
    if active.is_empty() {
        remaining.iter().cloned().collect()
    } else {
        active.into_iter().collect()
    }
}

/// Run Kahn's deterministic topological sort over a validated set of node IDs and edges.
fn kahn_topological_sort(
    node_ids: &BTreeSet<String>,
    edges: &[DagEdge],
) -> Result<Vec<String>, DagError> {
    let mut in_degree: BTreeMap<String, usize> =
        node_ids.iter().map(|id| (id.clone(), 0usize)).collect();
    let mut outgoing: BTreeMap<String, Vec<String>> =
        node_ids.iter().map(|id| (id.clone(), Vec::new())).collect();

    for edge in edges {
        if let Some(list) = outgoing.get_mut(&edge.from_node) {
            list.push(edge.to_node.clone());
        }
        if let Some(deg) = in_degree.get_mut(&edge.to_node) {
            *deg += 1;
        }
    }

    let mut ready: BTreeSet<String> = in_degree
        .iter()
        .filter_map(|(id, &deg)| (deg == 0).then_some(id.clone()))
        .collect();

    let mut order = Vec::with_capacity(node_ids.len());
    while let Some(curr) = ready.pop_first() {
        order.push(curr.clone());
        if let Some(neighbors) = outgoing.get(&curr) {
            for next in neighbors {
                if let Some(deg) = in_degree.get_mut(next) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        ready.insert(next.clone());
                    }
                }
            }
        }
    }

    if order.len() < node_ids.len() {
        let visited: BTreeSet<String> = order.into_iter().collect();
        let remaining: BTreeSet<String> = node_ids.difference(&visited).cloned().collect();
        let cycle_nodes = extract_cycle_nodes(&remaining, edges);
        return Err(DagError::CycleDetected { cycle_nodes });
    }

    Ok(order)
}

/// Validate a single candidate edge against an existing `WorkflowDag` without requiring
/// unrelated nodes to already have all required inputs wired (used during interactive editing).
pub fn validate_candidate_edge(
    dag: &WorkflowDag,
    candidate: &DagEdge,
) -> Result<PortType, DagError> {
    if dag.edges.len() >= MAX_DAG_EDGES {
        return Err(DagError::TooManyEdges {
            count: dag.edges.len() + 1,
            max: MAX_DAG_EDGES,
        });
    }

    let mut node_map: BTreeMap<String, NodeKind> = BTreeMap::new();
    for node in &dag.nodes {
        node_map.insert(node.id.clone(), node.kind);
    }

    let from_kind =
        node_map
            .get(&candidate.from_node)
            .copied()
            .ok_or_else(|| DagError::MissingNode {
                edge: Box::new(candidate.clone()),
                missing_node_id: candidate.from_node.clone(),
            })?;
    let to_kind =
        node_map
            .get(&candidate.to_node)
            .copied()
            .ok_or_else(|| DagError::MissingNode {
                edge: Box::new(candidate.clone()),
                missing_node_id: candidate.to_node.clone(),
            })?;

    let from_spec = from_kind
        .find_output_port(&candidate.from_port)
        .ok_or_else(|| DagError::UnknownPort {
            edge: Box::new(candidate.clone()),
            node_id: candidate.from_node.clone(),
            port: candidate.from_port.clone(),
            direction: PortDirection::Output,
        })?;
    let to_spec =
        to_kind
            .find_input_port(&candidate.to_port)
            .ok_or_else(|| DagError::UnknownPort {
                edge: Box::new(candidate.clone()),
                node_id: candidate.to_node.clone(),
                port: candidate.to_port.clone(),
                direction: PortDirection::Input,
            })?;

    if from_spec.port_type != to_spec.port_type {
        return Err(DagError::PortTypeMismatch {
            edge: Box::new(candidate.clone()),
            from_type: from_spec.port_type,
            to_type: to_spec.port_type,
        });
    }

    if dag.edges.iter().any(|e| e == candidate) {
        return Err(DagError::DuplicateEdge {
            edge: Box::new(candidate.clone()),
        });
    }

    if dag
        .edges
        .iter()
        .any(|e| e.to_node == candidate.to_node && e.to_port == candidate.to_port)
    {
        return Err(DagError::InputPortAlreadyConnected {
            node_id: candidate.to_node.clone(),
            port: candidate.to_port.clone(),
        });
    }

    let node_ids: BTreeSet<String> = node_map.keys().cloned().collect();
    let mut trial_edges = Vec::with_capacity(dag.edges.len() + 1);
    trial_edges.extend(dag.edges.iter().cloned());
    trial_edges.push(candidate.clone());
    let _ = kahn_topological_sort(&node_ids, &trial_edges)?;

    Ok(from_spec.port_type)
}

/// Strictly validate a [`WorkflowDag`]: bounds, IDs, parameters, port type-checking,
/// cycle detection (Kahn's algorithm), and dangling required-input detection.
pub fn validate_dag(dag: &WorkflowDag) -> Result<ValidatedDag, DagError> {
    if dag.nodes.is_empty() {
        return Err(DagError::EmptyDag);
    }
    if dag.nodes.len() > MAX_DAG_NODES {
        return Err(DagError::TooManyNodes {
            count: dag.nodes.len(),
            max: MAX_DAG_NODES,
        });
    }
    if dag.edges.len() > MAX_DAG_EDGES {
        return Err(DagError::TooManyEdges {
            count: dag.edges.len(),
            max: MAX_DAG_EDGES,
        });
    }
    if dag.name.is_empty()
        || dag.name.len() > MAX_STRING_PARAM_LEN
        || contains_control_chars(&dag.name)
    {
        return Err(DagError::InvalidParam {
            node_id: "<workflow>".to_string(),
            key: "name".to_string(),
            reason: "workflow name must be non-empty and free of control characters".to_string(),
        });
    }
    if dag.description.len() > MAX_STRING_PARAM_LEN || contains_control_chars(&dag.description) {
        return Err(DagError::InvalidParam {
            node_id: "<workflow>".to_string(),
            key: "description".to_string(),
            reason: "workflow description must be free of control characters".to_string(),
        });
    }

    let mut nodes_by_id: BTreeMap<String, &DagNode> = BTreeMap::new();
    for node in &dag.nodes {
        validate_node_id(&node.id)?;
        validate_node_params(node)?;
        if nodes_by_id.insert(node.id.clone(), node).is_some() {
            return Err(DagError::DuplicateNodeId {
                node_id: node.id.clone(),
            });
        }
    }

    let mut seen_edges: BTreeSet<DagEdge> = BTreeSet::new();
    let mut connected_inputs: BTreeSet<(String, String)> = BTreeSet::new();
    let mut edge_types: BTreeMap<DagEdge, PortType> = BTreeMap::new();

    for edge in &dag.edges {
        let from_node =
            nodes_by_id
                .get(&edge.from_node)
                .copied()
                .ok_or_else(|| DagError::MissingNode {
                    edge: Box::new(edge.clone()),
                    missing_node_id: edge.from_node.clone(),
                })?;
        let to_node =
            nodes_by_id
                .get(&edge.to_node)
                .copied()
                .ok_or_else(|| DagError::MissingNode {
                    edge: Box::new(edge.clone()),
                    missing_node_id: edge.to_node.clone(),
                })?;

        let from_spec = from_node
            .kind
            .find_output_port(&edge.from_port)
            .ok_or_else(|| DagError::UnknownPort {
                edge: Box::new(edge.clone()),
                node_id: edge.from_node.clone(),
                port: edge.from_port.clone(),
                direction: PortDirection::Output,
            })?;
        let to_spec =
            to_node
                .kind
                .find_input_port(&edge.to_port)
                .ok_or_else(|| DagError::UnknownPort {
                    edge: Box::new(edge.clone()),
                    node_id: edge.to_node.clone(),
                    port: edge.to_port.clone(),
                    direction: PortDirection::Input,
                })?;

        if from_spec.port_type != to_spec.port_type {
            return Err(DagError::PortTypeMismatch {
                edge: Box::new(edge.clone()),
                from_type: from_spec.port_type,
                to_type: to_spec.port_type,
            });
        }

        if !seen_edges.insert(edge.clone()) {
            return Err(DagError::DuplicateEdge {
                edge: Box::new(edge.clone()),
            });
        }

        if !connected_inputs.insert((edge.to_node.clone(), edge.to_port.clone())) {
            return Err(DagError::InputPortAlreadyConnected {
                node_id: edge.to_node.clone(),
                port: edge.to_port.clone(),
            });
        }

        edge_types.insert(edge.clone(), from_spec.port_type);
    }

    let node_ids: BTreeSet<String> = nodes_by_id.keys().cloned().collect();
    let topological_order = kahn_topological_sort(&node_ids, &dag.edges)?;

    // Check dangling required inputs in deterministic node-ID order.
    for (node_id, node) in &nodes_by_id {
        for spec in node.kind.input_ports() {
            if spec.required
                && !connected_inputs.contains(&(node_id.clone(), spec.name.to_string()))
            {
                return Err(DagError::MissingRequiredInput {
                    node_id: node_id.clone(),
                    port: spec.name.to_string(),
                });
            }
        }
        if node.kind.requires_any_input() {
            let has_any =
                node.kind.input_ports().iter().any(|spec| {
                    connected_inputs.contains(&(node_id.clone(), spec.name.to_string()))
                });
            if !has_any {
                return Err(DagError::MissingRequiredInput {
                    node_id: node_id.clone(),
                    port: "<any_input>".to_string(),
                });
            }
        }
    }

    Ok(ValidatedDag {
        topological_order,
        edge_types,
    })
}

/// Produce canonical, whitespace-free, sorted-key JSON for deterministic hashing.
fn canonical_json_value(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else if let Some(f) = n.as_f64() {
                // Normalize -0.0 to 0.0 for canonical representation.
                let clean = if f == 0.0 { 0.0 } else { f };
                out.push_str(&format!("{clean:?}"));
            } else {
                out.push_str(&n.to_string());
            }
        }
        serde_json::Value::String(s) => {
            if let Ok(encoded) = serde_json::to_string(s) {
                out.push_str(&encoded);
            }
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (idx, item) in items.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                canonical_json_value(item, out);
            }
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            let mut sorted_keys: Vec<&String> = map.keys().collect();
            sorted_keys.sort();
            out.push('{');
            for (idx, key) in sorted_keys.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                if let Ok(encoded_key) = serde_json::to_string(key) {
                    out.push_str(&encoded_key);
                }
                out.push(':');
                if let Some(val) = map.get(*key) {
                    canonical_json_value(val, out);
                }
            }
            out.push('}');
        }
    }
}

fn canonical_params_string(params: &BTreeMap<String, serde_json::Value>) -> String {
    let mut out = String::new();
    out.push('{');
    for (idx, (k, v)) in params.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        if let Ok(encoded_k) = serde_json::to_string(k) {
            out.push_str(&encoded_k);
        }
        out.push(':');
        canonical_json_value(v, &mut out);
    }
    out.push('}');
    out
}

/// Compile a [`WorkflowDag`] into a canonical, content-addressed [`DagManifest`] (`ferric-dag-v1`).
///
/// The resulting `node_sha256` and `manifest_sha256` digests are invariant to node/edge
/// insertion order, JSON parameter key order, and visual canvas coordinates.
pub fn compile_dag_manifest(dag: &WorkflowDag) -> Result<DagManifest, DagError> {
    let validated = validate_dag(dag)?;

    let mut nodes_by_id: BTreeMap<String, &DagNode> = BTreeMap::new();
    for node in &dag.nodes {
        nodes_by_id.insert(node.id.clone(), node);
    }

    let mut canonical_edges = dag.edges.clone();
    canonical_edges.sort();

    let mut incoming_by_node: BTreeMap<String, Vec<&DagEdge>> = BTreeMap::new();
    for edge in &canonical_edges {
        incoming_by_node
            .entry(edge.to_node.clone())
            .or_default()
            .push(edge);
    }

    let mut node_hashes: BTreeMap<String, String> = BTreeMap::new();
    let mut compiled_nodes = Vec::with_capacity(validated.topological_order.len());

    for node_id in &validated.topological_order {
        let Some(node) = nodes_by_id.get(node_id).copied() else {
            return Err(DagError::InvalidNodeId {
                node_id: node_id.clone(),
                reason: "internal node lookup failure".to_string(),
            });
        };

        let mut bindings = Vec::new();
        if let Some(in_edges) = incoming_by_node.get(node_id) {
            for edge in in_edges {
                let upstream_sha256 = node_hashes
                    .get(&edge.from_node)
                    .cloned()
                    .unwrap_or_default();
                let port_type = validated
                    .edge_types
                    .get(*edge)
                    .copied()
                    .unwrap_or(PortType::RawFid);
                bindings.push(CompiledInputBinding {
                    to_port: edge.to_port.clone(),
                    from_node: edge.from_node.clone(),
                    from_port: edge.from_port.clone(),
                    port_type,
                    upstream_sha256,
                });
            }
        }
        bindings.sort_by(|a, b| {
            a.to_port
                .cmp(&b.to_port)
                .then_with(|| a.from_node.cmp(&b.from_node))
                .then_with(|| a.from_port.cmp(&b.from_port))
        });

        let canonical_params = canonical_params_string(&node.params);
        let mut node_hash_input = String::new();
        node_hash_input.push_str("ferric-dag-node-v1\nkind=");
        node_hash_input.push_str(node.kind.canonical_id());
        node_hash_input.push_str("\nparams=");
        node_hash_input.push_str(&canonical_params);
        node_hash_input.push_str("\ninputs=[");
        for (idx, b) in bindings.iter().enumerate() {
            if idx > 0 {
                node_hash_input.push(',');
            }
            node_hash_input.push_str(&b.to_port);
            node_hash_input.push('<');
            node_hash_input.push_str(b.port_type.canonical_id());
            node_hash_input.push(':');
            node_hash_input.push_str(&b.from_port);
            node_hash_input.push('@');
            node_hash_input.push_str(&b.upstream_sha256);
        }
        node_hash_input.push_str("]\n");

        let node_sha256 = sha256_hex(node_hash_input.as_bytes());
        node_hashes.insert(node.id.clone(), node_sha256.clone());

        compiled_nodes.push(CompiledDagNode {
            id: node.id.clone(),
            kind: node.kind,
            label: node.label.clone(),
            params: node.params.clone(),
            inputs: bindings,
            node_sha256,
        });
    }

    let mut manifest_hash_input = String::new();
    manifest_hash_input.push_str("schema=");
    manifest_hash_input.push_str(DAG_MANIFEST_SCHEMA);
    manifest_hash_input.push_str("\nnodes=[");
    for (idx, cn) in compiled_nodes.iter().enumerate() {
        if idx > 0 {
            manifest_hash_input.push(',');
        }
        manifest_hash_input.push_str(&cn.id);
        manifest_hash_input.push('@');
        manifest_hash_input.push_str(&cn.node_sha256);
    }
    manifest_hash_input.push_str("]\nedges=[");
    for (idx, edge) in canonical_edges.iter().enumerate() {
        if idx > 0 {
            manifest_hash_input.push(',');
        }
        manifest_hash_input.push_str(&edge.from_node);
        manifest_hash_input.push(':');
        manifest_hash_input.push_str(&edge.from_port);
        manifest_hash_input.push_str("->");
        manifest_hash_input.push_str(&edge.to_node);
        manifest_hash_input.push(':');
        manifest_hash_input.push_str(&edge.to_port);
    }
    manifest_hash_input.push_str("]\n");

    let manifest_sha256 = sha256_hex(manifest_hash_input.as_bytes());

    Ok(DagManifest {
        schema: DAG_MANIFEST_SCHEMA.to_string(),
        workflow_name: dag.name.clone(),
        manifest_sha256,
        topological_order: validated.topological_order,
        nodes: compiled_nodes,
        edges: canonical_edges,
    })
}

/// Fail-closed integrity verification of a compiled [`DagManifest`]:
/// 1. Verifies schema identifier, hostile bounds, and valid graph structure/parameters.
/// 2. Recomputes canonical Kahn topological ordering, canonical sorted edges, every input
///    binding (`upstream_sha256` & `port_type`), every `node_sha256`, and `manifest_sha256`.
/// 3. Rejects any forged hash, parameter tampering, altered edge/input binding, or
///    non-topological / reversed node ordering with [`DagError::ManifestIntegrityMismatch`].
pub fn verify_dag_manifest(manifest: &DagManifest) -> Result<(), DagError> {
    if manifest.schema != DAG_MANIFEST_SCHEMA {
        return Err(DagError::ManifestIntegrityMismatch {
            reason: format!(
                "unsupported manifest schema '{}'; expected '{DAG_MANIFEST_SCHEMA}'",
                manifest.schema
            ),
        });
    }
    if manifest.nodes.is_empty() {
        return Err(DagError::ManifestIntegrityMismatch {
            reason: "manifest contains zero nodes".to_string(),
        });
    }
    if manifest.nodes.len() > MAX_DAG_NODES {
        return Err(DagError::ManifestIntegrityMismatch {
            reason: format!(
                "manifest node count {} exceeds maximum {MAX_DAG_NODES}",
                manifest.nodes.len()
            ),
        });
    }
    if manifest.edges.len() > MAX_DAG_EDGES {
        return Err(DagError::ManifestIntegrityMismatch {
            reason: format!(
                "manifest edge count {} exceeds maximum {MAX_DAG_EDGES}",
                manifest.edges.len()
            ),
        });
    }
    if manifest.topological_order.len() != manifest.nodes.len() {
        return Err(DagError::ManifestIntegrityMismatch {
            reason: format!(
                "topological_order length {} does not match node count {}",
                manifest.topological_order.len(),
                manifest.nodes.len()
            ),
        });
    }

    let reconstructed = WorkflowDag {
        name: manifest.workflow_name.clone(),
        description: String::new(),
        nodes: manifest
            .nodes
            .iter()
            .map(|n| DagNode {
                id: n.id.clone(),
                kind: n.kind,
                label: n.label.clone(),
                params: n.params.clone(),
                position: [0.0, 0.0],
            })
            .collect(),
        edges: manifest.edges.clone(),
    };

    let expected =
        compile_dag_manifest(&reconstructed).map_err(|e| DagError::ManifestIntegrityMismatch {
            reason: format!("manifest graph validation failed: {e}"),
        })?;

    if manifest.topological_order != expected.topological_order {
        return Err(DagError::ManifestIntegrityMismatch {
            reason: "topological_order does not match canonical Kahn topological order".to_string(),
        });
    }
    if manifest.edges != expected.edges {
        return Err(DagError::ManifestIntegrityMismatch {
            reason: "manifest edges do not match canonical sorted edge list".to_string(),
        });
    }
    if manifest.nodes.len() != expected.nodes.len() {
        return Err(DagError::ManifestIntegrityMismatch {
            reason: "manifest node count mismatch".to_string(),
        });
    }

    for (idx, (actual, exp)) in manifest.nodes.iter().zip(expected.nodes.iter()).enumerate() {
        if actual.id != exp.id {
            return Err(DagError::ManifestIntegrityMismatch {
                reason: format!(
                    "node ordering mismatch at step {idx}: expected '{}', found '{}'",
                    exp.id, actual.id
                ),
            });
        }
        if actual.kind != exp.kind {
            return Err(DagError::ManifestIntegrityMismatch {
                reason: format!(
                    "node kind mismatch for '{}': expected '{}', found '{}'",
                    exp.id,
                    exp.kind.canonical_id(),
                    actual.kind.canonical_id()
                ),
            });
        }
        if actual.inputs != exp.inputs {
            return Err(DagError::ManifestIntegrityMismatch {
                reason: format!(
                    "compiled input bindings mismatch for node '{}' (possible tampered upstream_sha256 or port binding)",
                    exp.id
                ),
            });
        }
        if actual.node_sha256 != exp.node_sha256 {
            return Err(DagError::ManifestIntegrityMismatch {
                reason: format!(
                    "node_sha256 mismatch for node '{}': expected '{}', found '{}'",
                    exp.id, exp.node_sha256, actual.node_sha256
                ),
            });
        }
    }

    if manifest.manifest_sha256 != expected.manifest_sha256 {
        return Err(DagError::ManifestIntegrityMismatch {
            reason: format!(
                "manifest_sha256 mismatch: expected '{}', found '{}'",
                expected.manifest_sha256, manifest.manifest_sha256
            ),
        });
    }

    Ok(())
}

/// Execute a compiled [`DagManifest`] in deterministic topological order using a
/// caller-supplied pure/mockable stage `runner`, reusing cached upstream artifacts
/// keyed by `node_sha256`.
///
/// Fail-closed: calls [`verify_dag_manifest`] BEFORE consulting `cache` or executing
/// any stage, rejecting forged hashes or non-topological / reversed node ordering.
pub fn execute_dag_manifest<F>(
    manifest: &DagManifest,
    cache: &mut DagArtifactCache,
    mut runner: F,
) -> Result<DagExecutionTrace, DagError>
where
    F: FnMut(&CompiledDagNode, &[&CachedArtifact]) -> Result<String, DagError>,
{
    verify_dag_manifest(manifest)?;

    let mut steps = Vec::with_capacity(manifest.nodes.len());
    let mut executed_count = 0usize;
    let mut cache_hit_count = 0usize;

    for (step_index, node) in manifest.nodes.iter().enumerate() {
        if let Some(cached) = cache
            .get(&node.node_sha256)
            .filter(|art| art.kind == node.kind)
        {
            cache_hit_count += 1;
            steps.push(DagStepProvenance {
                step_index,
                node_id: node.id.clone(),
                kind: node.kind,
                node_sha256: node.node_sha256.clone(),
                status: StepStatus::CacheHit,
                output_digest: cached.output_digest.clone(),
                summary: cached.summary.clone(),
            });
            continue;
        }

        let mut upstream_artifacts = Vec::with_capacity(node.inputs.len());
        for input in &node.inputs {
            if let Some(art) = cache.get(&input.upstream_sha256) {
                upstream_artifacts.push(art);
            }
        }

        let summary = runner(node, &upstream_artifacts)?;
        let digest_payload = format!(
            "ferric-dag-output-v1\nnode_sha256={}\nsummary={}\n",
            node.node_sha256, summary
        );
        let output_digest = sha256_hex(digest_payload.as_bytes());

        cache.insert(CachedArtifact {
            node_sha256: node.node_sha256.clone(),
            kind: node.kind,
            output_digest: output_digest.clone(),
            summary: summary.clone(),
        });

        executed_count += 1;
        steps.push(DagStepProvenance {
            step_index,
            node_id: node.id.clone(),
            kind: node.kind,
            node_sha256: node.node_sha256.clone(),
            status: StepStatus::Executed,
            output_digest,
            summary,
        });
    }

    Ok(DagExecutionTrace {
        schema: manifest.schema.clone(),
        manifest_sha256: manifest.manifest_sha256.clone(),
        steps,
        executed_count,
        cache_hit_count,
    })
}

/// Pure deterministic dry-run executor for a compiled [`DagManifest`] with content-addressed caching.
pub fn execute_dag_manifest_dry_run(
    manifest: &DagManifest,
    cache: &mut DagArtifactCache,
) -> Result<DagExecutionTrace, DagError> {
    execute_dag_manifest(manifest, cache, |node, upstream| {
        let short_hash: String = node.node_sha256.chars().take(12).collect();
        Ok(format!(
            "dry-run {} ({}) inputs={} sha256={}",
            node.kind.canonical_id(),
            node.id,
            upstream.len(),
            short_hash
        ))
    })
}

/// Inputs and execution options for real `nmr-core` stage execution of a compiled [`DagManifest`].
#[derive(Debug, Clone, Default)]
pub struct DagRunOptions {
    /// Optional NMR-STAR text supplying reference sequence and shifts.
    pub star_text: Option<String>,
    /// Optional TSV peak list text (`experiment\tH\tN\tC\tsign...`).
    pub peaks_tsv: Option<String>,
    /// Optional NEF v1.1 text supplying molecular system and/or spectrum peaks.
    pub nef_text: Option<String>,
    /// Optional raw 1-letter or FASTA amino-acid sequence.
    pub sequence: Option<String>,
    /// Optional sandboxed root directory for resolving relative `dataset_path` node parameters.
    pub data_root: Option<PathBuf>,
    /// Optional output directory where `ExportNefReport` writes real artifacts.
    pub output_dir: Option<PathBuf>,
    /// When `true` and no explicit inline or disk inputs are found, use deterministic built-in
    /// synthetic fixtures (tagged explicitly in provenance summaries).
    pub use_demo_fixture: bool,
    /// Deterministic PRNG seed for spectral simulation and solver stages.
    pub seed: u64,
    /// Optional cap on solver replicas (defaults to bounded deterministic count `8`).
    pub max_runs: Option<usize>,
    /// Optional cap on solver steps per spin system (defaults to `400`).
    pub max_steps_per_system: Option<usize>,
}

/// Real artifacts, metrics, and provenance trace produced by [`execute_dag_manifest_real`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DagRunOutput {
    pub trace: DagExecutionTrace,
    pub artifacts: BTreeMap<String, String>,
    pub written_files: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nef_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_html: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics_json: Option<serde_json::Value>,
}

struct DagRealRuntime {
    reference: Option<Reference>,
    sequence: Vec<String>,
    seq_ids: Vec<i32>,
    peaks: Vec<Peak>,
    spectrum_2d: Option<Spectrum2D>,
    pred_ref: Option<Reference>,
    confidence: BTreeMap<usize, f64>,
    n_spin_systems: usize,
    n_assigned: usize,
    n_high_confidence: usize,
    solver_score: Option<f64>,
    metabolomics_cohort: Option<Vec<CohortSample1D>>,
    metabolomics_study: Option<MetabolomicsStudyResult>,
    source_tag: String,
    artifacts: BTreeMap<String, String>,
    written_files: Vec<String>,
    nef_text: Option<String>,
    report_html: Option<String>,
    metrics_json: Option<serde_json::Value>,
}

impl DagRealRuntime {
    fn new() -> Self {
        Self {
            reference: None,
            sequence: Vec::new(),
            seq_ids: Vec::new(),
            peaks: Vec::new(),
            spectrum_2d: None,
            pred_ref: None,
            confidence: BTreeMap::new(),
            n_spin_systems: 0,
            n_assigned: 0,
            n_high_confidence: 0,
            solver_score: None,
            metabolomics_cohort: None,
            metabolomics_study: None,
            source_tag: String::new(),
            artifacts: BTreeMap::new(),
            written_files: Vec::new(),
            nef_text: None,
            report_html: None,
            metrics_json: None,
        }
    }

    fn parse_sequence_text(raw: &str) -> Result<(Vec<String>, Vec<i32>), String> {
        let letters: String = raw
            .lines()
            .filter(|l| {
                let t = l.trim_start();
                !t.starts_with('>') && !t.starts_with(';') && !t.starts_with('#')
            })
            .flat_map(str::chars)
            .filter(|c| c.is_ascii_alphabetic())
            .collect();
        if letters.is_empty() {
            return Err("sequence contains no amino-acid letters".to_string());
        }
        let mut seq = Vec::with_capacity(letters.len());
        for ch in letters.chars() {
            let three = one_to_three(ch);
            if three == "UNK" {
                return Err(format!("unrecognized amino-acid letter '{ch}'"));
            }
            seq.push(three.to_string());
        }
        let seq_ids: Vec<i32> = (1..=(seq.len() as i32)).collect();
        Ok((seq, seq_ids))
    }

    fn resolve_sandboxed_input_path(
        node_id: &str,
        data_root: &Path,
        rel_path: &str,
    ) -> Result<Option<PathBuf>, DagError> {
        validate_safe_relative_path(node_id, "dataset_path", rel_path)?;
        let Ok(canon_root) = std::fs::canonicalize(data_root) else {
            return Ok(None);
        };
        let joined = canon_root.join(rel_path);
        if !joined.exists() {
            return Ok(None);
        }
        let canon_target =
            std::fs::canonicalize(&joined).map_err(|e| DagError::ExecutionFailed {
                node_id: node_id.to_string(),
                reason: format!("failed to canonicalize dataset path '{rel_path}': {e}"),
            })?;
        if !canon_target.starts_with(&canon_root) {
            return Err(DagError::ExecutionFailed {
                node_id: node_id.to_string(),
                reason: format!("dataset path '{rel_path}' escapes sandboxed data_root"),
            });
        }
        Ok(Some(canon_target))
    }

    fn load_inputs_if_needed(
        &mut self,
        node_id: &str,
        dataset_path: Option<&str>,
        is_metabolomics: bool,
        opts: &DagRunOptions,
    ) -> Result<String, DagError> {
        if !self.source_tag.is_empty() {
            return Ok(self.source_tag.clone());
        }

        let mut loaded_parts = Vec::new();

        // 1. Optional sandboxed disk dataset_path under `opts.data_root`
        if let (Some(root), Some(rel)) = (opts.data_root.as_deref(), dataset_path) {
            if let Some(path) = Self::resolve_sandboxed_input_path(node_id, root, rel)? {
                if path.is_file() {
                    let ext = path
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(str::to_ascii_lowercase)
                        .unwrap_or_default();
                    let text =
                        std::fs::read_to_string(&path).map_err(|e| DagError::ExecutionFailed {
                            node_id: node_id.to_string(),
                            reason: format!("failed to read '{}': {e}", path.display()),
                        })?;
                    if ext == "str" || text.contains("_Atom_chem_shift.") {
                        let r =
                            read_star_str(&text, None).map_err(|e| DagError::ExecutionFailed {
                                node_id: node_id.to_string(),
                                reason: format!("invalid NMR-STAR file '{rel}': {e}"),
                            })?;
                        self.sequence = r.sequence.clone();
                        self.seq_ids = r.seq_ids.clone();
                        loaded_parts.push(format!(
                            "file_star({}) residues={} shifts={}",
                            rel,
                            r.sequence.len(),
                            r.shifts.len()
                        ));
                        self.reference = Some(r);
                    } else if ext == "nef" || text.contains("data_nef_") {
                        let doc = read_nef_str(&text).map_err(|e| DagError::ExecutionFailed {
                            node_id: node_id.to_string(),
                            reason: format!("invalid NEF file '{rel}': {e}"),
                        })?;
                        let r = doc.to_reference().map_err(|e| DagError::ExecutionFailed {
                            node_id: node_id.to_string(),
                            reason: format!("failed to extract reference from NEF '{rel}': {e}"),
                        })?;
                        self.sequence = r.sequence.clone();
                        self.seq_ids = r.seq_ids.clone();
                        self.peaks = doc.to_peaks();
                        loaded_parts.push(format!(
                            "file_nef({}) residues={} peaks={}",
                            rel,
                            r.sequence.len(),
                            self.peaks.len()
                        ));
                        self.reference = Some(r);
                    } else if ext == "tsv" {
                        let pks = read_peaks_str(&text).map_err(|e| DagError::ExecutionFailed {
                            node_id: node_id.to_string(),
                            reason: format!("invalid peaks TSV '{rel}': {e}"),
                        })?;
                        loaded_parts.push(format!("file_peaks({}) count={}", rel, pks.len()));
                        self.peaks = pks;
                    }
                } else if path.is_dir() {
                    // Check for known NMR-STAR / NEF / TSV files inside the dataset directory
                    if let Ok(entries) = std::fs::read_dir(&path) {
                        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
                        paths.sort();
                        for fpath in paths {
                            if !fpath.is_file() {
                                continue;
                            }
                            let ext = fpath
                                .extension()
                                .and_then(|s| s.to_str())
                                .map(str::to_ascii_lowercase)
                                .unwrap_or_default();
                            if ext == "str" && self.reference.is_none() {
                                if let Ok(txt) = std::fs::read_to_string(&fpath) {
                                    if let Ok(r) = read_star_str(&txt, None) {
                                        self.sequence = r.sequence.clone();
                                        self.seq_ids = r.seq_ids.clone();
                                        loaded_parts.push(format!(
                                            "dir_star residues={} shifts={}",
                                            r.sequence.len(),
                                            r.shifts.len()
                                        ));
                                        self.reference = Some(r);
                                    }
                                }
                            } else if ext == "tsv" && self.peaks.is_empty() {
                                if let Ok(txt) = std::fs::read_to_string(&fpath) {
                                    if let Ok(pks) = read_peaks_str(&txt) {
                                        loaded_parts.push(format!("dir_peaks count={}", pks.len()));
                                        self.peaks = pks;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Explicit inline payloads (`star_text`, `nef_text`, `peaks_tsv`, `sequence`)
        if let Some(star_txt) = opts.star_text.as_deref().filter(|s| !s.trim().is_empty()) {
            let r = read_star_str(star_txt, None).map_err(|e| DagError::ExecutionFailed {
                node_id: node_id.to_string(),
                reason: format!("invalid inline NMR-STAR input: {e}"),
            })?;
            self.sequence = r.sequence.clone();
            self.seq_ids = r.seq_ids.clone();
            loaded_parts.push(format!(
                "inline_star residues={} shifts={}",
                r.sequence.len(),
                r.shifts.len()
            ));
            self.reference = Some(r);
        }
        if let Some(nef_txt) = opts.nef_text.as_deref().filter(|s| !s.trim().is_empty()) {
            let doc = read_nef_str(nef_txt).map_err(|e| DagError::ExecutionFailed {
                node_id: node_id.to_string(),
                reason: format!("invalid inline NEF input: {e}"),
            })?;
            let r = doc.to_reference().map_err(|e| DagError::ExecutionFailed {
                node_id: node_id.to_string(),
                reason: format!("failed to extract reference from inline NEF: {e}"),
            })?;
            if self.sequence.is_empty() {
                self.sequence = r.sequence.clone();
                self.seq_ids = r.seq_ids.clone();
            }
            if self.reference.is_none() {
                self.reference = Some(r);
            }
            let nef_peaks = doc.to_peaks();
            if !nef_peaks.is_empty() && self.peaks.is_empty() {
                self.peaks = nef_peaks;
            }
            loaded_parts.push(format!("inline_nef peaks={}", self.peaks.len()));
        }
        if let Some(tsv) = opts.peaks_tsv.as_deref().filter(|s| !s.trim().is_empty()) {
            let pks = read_peaks_str(tsv).map_err(|e| DagError::ExecutionFailed {
                node_id: node_id.to_string(),
                reason: format!("invalid inline peaks TSV: {e}"),
            })?;
            loaded_parts.push(format!("inline_peaks count={}", pks.len()));
            self.peaks = pks;
        }
        if let Some(seq_txt) = opts.sequence.as_deref().filter(|s| !s.trim().is_empty()) {
            let (seq, sids) =
                Self::parse_sequence_text(seq_txt).map_err(|e| DagError::ExecutionFailed {
                    node_id: node_id.to_string(),
                    reason: format!("invalid inline sequence: {e}"),
                })?;
            loaded_parts.push(format!("inline_seq residues={}", seq.len()));
            self.sequence = seq;
            self.seq_ids = sids;
        }

        // 3. Explicit synthetic demo fixture fallback ONLY when `use_demo_fixture` is enabled
        if loaded_parts.is_empty() && opts.use_demo_fixture {
            if is_metabolomics {
                let cohort = generate_demo_biofluid_cohort(opts.seed);
                loaded_parts.push(format!(
                    "demo_fixture_metabolomics samples={}",
                    cohort.len()
                ));
                self.metabolomics_cohort = Some(cohort);
            } else {
                let r = read_star_str(EMBEDDED_DAG_DEMO_UBIQUITIN_STAR, None).map_err(|e| {
                    DagError::ExecutionFailed {
                        node_id: node_id.to_string(),
                        reason: format!("failed to load embedded demo ubiquitin fixture: {e}"),
                    }
                })?;
                self.sequence = r.sequence.clone();
                self.seq_ids = r.seq_ids.clone();
                loaded_parts.push(format!(
                    "demo_fixture_ubiquitin residues={} shifts={}",
                    r.sequence.len(),
                    r.shifts.len()
                ));
                self.reference = Some(r);
            }
        }

        if loaded_parts.is_empty() {
            let ds = dataset_path.unwrap_or("<unspecified>");
            return Err(DagError::ExecutionFailed {
                node_id: node_id.to_string(),
                reason: format!(
                    "no input dataset available for '{ds}'; supply star_text/peaks_tsv/nef_text, --data-dir containing the dataset, --demo (use_demo_fixture=true), or --dry-run"
                ),
            });
        }

        self.source_tag = loaded_parts.join("; ");
        Ok(self.source_tag.clone())
    }

    fn ensure_processed(&mut self, node_id: &str, opts: &DagRunOptions) -> Result<(), DagError> {
        if self.spectrum_2d.is_some()
            || !self.peaks.is_empty()
            || self.metabolomics_cohort.is_some()
        {
            return Ok(());
        }
        if let Some(ref ref_data) = self.reference {
            let (sim_peaks, _) = simulate(
                ref_data,
                &SimConfig {
                    seed: opts.seed,
                    ..SimConfig::default()
                },
            );
            let rcfg = SpectrumRenderConfig {
                nx_h: 128,
                ny_n: 64,
                ..SpectrumRenderConfig::default()
            };
            let hsqc = render_hsqc_2d(&sim_peaks, &rcfg);
            self.peaks = sim_peaks;
            self.spectrum_2d = Some(hsqc);
            return Ok(());
        }
        Err(DagError::ExecutionFailed {
            node_id: node_id.to_string(),
            reason: "no spectrum, reference shifts, or peak list available for processing"
                .to_string(),
        })
    }

    fn write_safe_output_file(
        node_id: &str,
        out_dir: &Path,
        rel_path: &str,
        content: &str,
    ) -> Result<String, DagError> {
        validate_safe_relative_path(node_id, "output_path", rel_path)?;
        std::fs::create_dir_all(out_dir).map_err(|e| DagError::ExecutionFailed {
            node_id: node_id.to_string(),
            reason: format!("failed to create output directory: {e}"),
        })?;
        let canon_out = std::fs::canonicalize(out_dir).map_err(|e| DagError::ExecutionFailed {
            node_id: node_id.to_string(),
            reason: format!("failed to canonicalize output directory: {e}"),
        })?;
        let target = canon_out.join(rel_path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|e| DagError::ExecutionFailed {
                node_id: node_id.to_string(),
                reason: format!("failed to create output parent directory: {e}"),
            })?;
            let canon_parent =
                std::fs::canonicalize(parent).map_err(|e| DagError::ExecutionFailed {
                    node_id: node_id.to_string(),
                    reason: format!("failed to canonicalize output parent directory: {e}"),
                })?;
            if !canon_parent.starts_with(&canon_out) {
                return Err(DagError::ExecutionFailed {
                    node_id: node_id.to_string(),
                    reason: format!("output path '{rel_path}' escapes output directory"),
                });
            }
        }
        std::fs::write(&target, content).map_err(|e| DagError::ExecutionFailed {
            node_id: node_id.to_string(),
            reason: format!("failed to write artifact '{rel_path}': {e}"),
        })?;
        Ok(rel_path.to_string())
    }
}

/// Execute a compiled [`DagManifest`] deterministically against real `nmr-core` processing,
/// peak-picking, backbone-assignment, metabolomics-chemometrics, and NEF/HTML export engines.
///
/// Always performs fail-closed [`verify_dag_manifest`] integrity checking BEFORE consulting
/// `cache` or running any stage.
pub fn execute_dag_manifest_real(
    manifest: &DagManifest,
    cache: &mut DagArtifactCache,
    opts: &DagRunOptions,
) -> Result<DagRunOutput, DagError> {
    let is_metabolomics_pipeline = manifest
        .nodes
        .iter()
        .any(|n| n.kind == NodeKind::MetabolomicsChemometrics);
    let ingest_dataset_path: Option<String> = manifest
        .nodes
        .iter()
        .find(|n| n.kind == NodeKind::IngestBrukerVarian)
        .and_then(|n| n.params.get("dataset_path"))
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned);

    let mut rt = DagRealRuntime::new();

    let trace = execute_dag_manifest(manifest, cache, |node, _upstream| match node.kind {
        NodeKind::IngestBrukerVarian => {
            let ds = node.params.get("dataset_path").and_then(|v| v.as_str());
            let info = rt.load_inputs_if_needed(&node.id, ds, is_metabolomics_pipeline, opts)?;
            let vendor = node
                .params
                .get("vendor")
                .and_then(|v| v.as_str())
                .unwrap_or("bruker");
            Ok(format!("ingest vendor={vendor} {info}"))
        }
        NodeKind::ProcessFidNus => {
            rt.load_inputs_if_needed(
                &node.id,
                ingest_dataset_path.as_deref(),
                is_metabolomics_pipeline,
                opts,
            )?;
            let nus = node
                .params
                .get("nus_method")
                .and_then(|v| v.as_str())
                .unwrap_or("ist");
            let zf = node
                .params
                .get("zero_fill_factor")
                .and_then(|v| v.as_u64())
                .unwrap_or(2);
            if let Some(ref cohort) = rt.metabolomics_cohort {
                let cfg = MetabolomicsPipelineConfig::default();
                let (grid, masked, _, _, _) = preprocess_cohort_1d(cohort, &cfg);
                let active_pts = masked.iter().filter(|&&m| !m).count();
                Ok(format!(
                    "process_1d_cohort samples={} grid_pts={} active_pts={} zf={zf}",
                    cohort.len(),
                    grid.len(),
                    active_pts
                ))
            } else {
                rt.ensure_processed(&node.id, opts)?;
                if let Some(ref spec) = rt.spectrum_2d {
                    Ok(format!(
                        "process_2d_hsqc grid={}x{} peaks={} nus={nus} zf={zf}",
                        spec.axis_x.size,
                        spec.axis_y.size,
                        rt.peaks.len()
                    ))
                } else {
                    Ok(format!(
                        "process_passthrough peaks={} nus={nus} zf={zf}",
                        rt.peaks.len()
                    ))
                }
            }
        }
        NodeKind::PeakPick => {
            rt.load_inputs_if_needed(
                &node.id,
                ingest_dataset_path.as_deref(),
                is_metabolomics_pipeline,
                opts,
            )?;
            rt.ensure_processed(&node.id, opts)?;
            let snr = node
                .params
                .get("snr_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(5.0);
            let hsqc_picked = if let Some(ref spec) = rt.spectrum_2d {
                let cfg = PickerConfig {
                    snr_threshold: snr as f32,
                    ..PickerConfig::default()
                };
                pick_hsqc_2d(spec, &cfg).len()
            } else {
                rt.peaks.iter().filter(|p| p.exp == "HSQC").count()
            };
            Ok(format!(
                "peak_pick hsqc_picked={hsqc_picked} total_peaks={} snr={snr:.2}",
                rt.peaks.len()
            ))
        }
        NodeKind::DeconvolveNd => {
            rt.load_inputs_if_needed(
                &node.id,
                ingest_dataset_path.as_deref(),
                is_metabolomics_pipeline,
                opts,
            )?;
            rt.ensure_processed(&node.id, opts)?;
            let lineshape = node
                .params
                .get("lineshape")
                .and_then(|v| v.as_str())
                .unwrap_or("voigt");
            Ok(format!(
                "deconvolve_nd lineshape={lineshape} peaks={}",
                rt.peaks.len()
            ))
        }
        NodeKind::BackboneAssign => {
            rt.load_inputs_if_needed(
                &node.id,
                ingest_dataset_path.as_deref(),
                is_metabolomics_pipeline,
                opts,
            )?;
            rt.ensure_processed(&node.id, opts)?;
            if rt.sequence.is_empty() {
                return Err(DagError::ExecutionFailed {
                    node_id: node.id.clone(),
                    reason: "BackboneAssign requires a non-empty protein sequence".to_string(),
                });
            }
            if rt.peaks.is_empty() {
                return Err(DagError::ExecutionFailed {
                    node_id: node.id.clone(),
                    reason: "BackboneAssign requires non-empty NMR peak lists".to_string(),
                });
            }
            let solver_name = node
                .params
                .get("solver")
                .and_then(|v| v.as_str())
                .unwrap_or("evo_graph");
            let priors = Priors::default();
            let runs = opts.max_runs.unwrap_or(8).clamp(1, 64);
            let steps = opts.max_steps_per_system.unwrap_or(400).clamp(100, 4000);

            let (systems, assign_res) = if solver_name == "standard" {
                let systems = build_spin_systems(&rt.peaks, &Tolerances::default(), 3, false)
                    .map_err(|e| DagError::ExecutionFailed {
                        node_id: node.id.clone(),
                        reason: format!("spin-system assembly failed: {e}"),
                    })?;
                let anneal_cfg = AnnealConfig {
                    runs,
                    steps_per_system: steps,
                    seed: opts.seed,
                    ..AnnealConfig::default()
                };
                let res = assign(
                    &systems,
                    &rt.sequence,
                    &priors,
                    &ScoreConfig::default(),
                    &anneal_cfg,
                );
                (systems, res)
            } else {
                let evo_cfg = EvoGraphConfig {
                    population_size: runs.clamp(4, 32),
                    num_islands: runs.clamp(2, 8),
                    generations: 3,
                    steps_per_system_per_gen: (steps / 2).clamp(100, 800),
                    seed: opts.seed,
                    ..EvoGraphConfig::default()
                };
                let (systems, res, _) =
                    assign_evo_graph_from_peaks(&rt.peaks, &rt.sequence, &priors, &evo_cfg);
                (systems, res)
            };

            let (pred_ref, conf) = to_reference(
                &assign_res,
                &systems,
                &rt.sequence,
                &rt.seq_ids,
                0.0,
                Some(&priors),
            );
            let mut n_assigned = 0usize;
            let mut n_hc = 0usize;
            for (s, &pos) in assign_res.position.iter().enumerate() {
                if pos >= 0 {
                    n_assigned += 1;
                    if assign_res.confidence.get(s).copied().unwrap_or(0.0) >= 0.80 {
                        n_hc += 1;
                    }
                }
            }
            rt.n_spin_systems = systems.len();
            rt.n_assigned = n_assigned;
            rt.n_high_confidence = n_hc;
            rt.solver_score = Some(assign_res.score);
            rt.pred_ref = Some(pred_ref);
            rt.confidence = conf;
            rt.metrics_json = Some(serde_json::json!({
                "solver": solver_name,
                "n_residues": rt.sequence.len(),
                "n_peaks": rt.peaks.len(),
                "n_spin_systems": rt.n_spin_systems,
                "n_assigned": rt.n_assigned,
                "n_high_confidence": rt.n_high_confidence,
                "solver_score": assign_res.score,
            }));

            Ok(format!(
                "backbone_assign solver={solver_name} residues={} spin_systems={} assigned={} high_conf={} score={:.2}",
                rt.sequence.len(),
                rt.n_spin_systems,
                rt.n_assigned,
                rt.n_high_confidence,
                assign_res.score
            ))
        }
        NodeKind::MetabolomicsChemometrics => {
            rt.load_inputs_if_needed(&node.id, ingest_dataset_path.as_deref(), true, opts)?;
            let cohort =
                rt.metabolomics_cohort
                    .as_ref()
                    .ok_or_else(|| DagError::ExecutionFailed {
                        node_id: node.id.clone(),
                        reason: "MetabolomicsChemometrics requires a 1D NMR cohort".to_string(),
                    })?;
            let mut cfg = MetabolomicsPipelineConfig::default();
            if let Some(n_comp) = node.params.get("n_components").and_then(|v| v.as_u64()) {
                cfg.pca_components = (n_comp as usize).clamp(1, 8);
            }
            let study = run_metabolomics_pipeline(cohort, &cfg);
            let pc1 = study
                .pca
                .explained_variance_ratio
                .first()
                .copied()
                .unwrap_or(0.0);
            let n_samples = study.sample_ids.len();
            let n_buckets = study.buckets.len();
            let n_biomarkers = study.biomarkers.len();
            rt.metrics_json = Some(serde_json::json!({
                "n_samples": n_samples,
                "n_buckets": n_buckets,
                "pca_pc1_explained_variance": pc1,
                "n_biomarkers": n_biomarkers,
            }));
            rt.metabolomics_study = Some(study);
            Ok(format!(
                "metabolomics_chemometrics samples={n_samples} buckets={n_buckets} pc1_var={pc1:.4} biomarkers={n_biomarkers}"
            ))
        }
        NodeKind::DynamicsRelax => {
            rt.load_inputs_if_needed(
                &node.id,
                ingest_dataset_path.as_deref(),
                is_metabolomics_pipeline,
                opts,
            )?;
            rt.ensure_processed(&node.id, opts)?;
            let field_mhz = node
                .params
                .get("field_mhz")
                .and_then(|v| v.as_f64())
                .unwrap_or(800.0);
            let hsqc_count = rt.peaks.iter().filter(|p| p.exp == "HSQC").count();
            Ok(format!(
                "dynamics_relax field_mhz={field_mhz:.1} hsqc_roots={hsqc_count} total_peaks={}",
                rt.peaks.len()
            ))
        }
        NodeKind::ExchangeCestCpmg => {
            rt.load_inputs_if_needed(
                &node.id,
                ingest_dataset_path.as_deref(),
                is_metabolomics_pipeline,
                opts,
            )?;
            rt.ensure_processed(&node.id, opts)?;
            let b1_hz = node
                .params
                .get("b1_hz")
                .and_then(|v| v.as_f64())
                .unwrap_or(25.0);
            let hsqc_count = rt.peaks.iter().filter(|p| p.exp == "HSQC").count();
            Ok(format!(
                "exchange_cest_cpmg b1_hz={b1_hz:.1} hsqc_roots={hsqc_count} total_peaks={}",
                rt.peaks.len()
            ))
        }
        NodeKind::SidechainAssign | NodeKind::Titration2d | NodeKind::StructureFoldDock => {
            rt.load_inputs_if_needed(
                &node.id,
                ingest_dataset_path.as_deref(),
                is_metabolomics_pipeline,
                opts,
            )?;
            let n_res = rt.sequence.len();
            let n_pks = rt.peaks.len();
            Ok(format!(
                "{} residues={n_res} peaks={n_pks}",
                node.kind.canonical_id()
            ))
        }
        NodeKind::ExportNefReport => {
            let out_rel = node
                .params
                .get("output_path")
                .and_then(|v| v.as_str())
                .unwrap_or("reports/ferric_bundle.nef");

            if let Some(ref study) = rt.metabolomics_study {
                let summary_json = export_study_summary_json(study, 10).map_err(|e| {
                    DagError::ExecutionFailed {
                        node_id: node.id.clone(),
                        reason: format!("failed to serialize metabolomics study JSON: {e}"),
                    }
                })?;
                let csv = export_bucket_table_csv(study);
                let json_sha = sha256_hex(summary_json.as_bytes());
                let csv_sha = sha256_hex(csv.as_bytes());
                rt.artifacts
                    .insert("metabolomics_summary.json".to_string(), json_sha.clone());
                rt.artifacts
                    .insert("metabolomics_buckets.csv".to_string(), csv_sha);
                if let Some(out_dir) = opts.output_dir.as_deref() {
                    let written = DagRealRuntime::write_safe_output_file(
                        &node.id,
                        out_dir,
                        out_rel,
                        &summary_json,
                    )?;
                    rt.written_files.push(written);
                    let csv_rel = format!("{out_rel}.buckets.csv");
                    let written_csv =
                        DagRealRuntime::write_safe_output_file(&node.id, out_dir, &csv_rel, &csv)?;
                    rt.written_files.push(written_csv);
                }
                let short_sha: String = json_sha.chars().take(12).collect();
                Ok(format!(
                    "export_metabolomics json_bytes={} csv_bytes={} sha256={short_sha}",
                    summary_json.len(),
                    csv.len()
                ))
            } else {
                let export_ref =
                    rt.pred_ref
                        .as_ref()
                        .or(rt.reference.as_ref())
                        .ok_or_else(|| DagError::ExecutionFailed {
                            node_id: node.id.clone(),
                            reason: "ExportNefReport has no assigned or reference shifts to export"
                                .to_string(),
                        })?;
                let conf_opt = if rt.confidence.is_empty() {
                    None
                } else {
                    Some(&rt.confidence)
                };
                let nef_doc = NefDocument::from_reference_and_peaks(
                    &manifest.workflow_name,
                    export_ref,
                    conf_opt,
                    &rt.peaks,
                );
                let nef_str = write_nef_str(&nef_doc);
                let nef_sha = sha256_hex(nef_str.as_bytes());

                let mut report_in = ReportInput::new(
                    &manifest.workflow_name,
                    export_ref.clone(),
                    rt.confidence.clone(),
                );
                report_in.n_peaks = Some(rt.peaks.len());
                if rt.n_spin_systems > 0 {
                    report_in.n_spin_systems = Some(rt.n_spin_systems);
                }
                report_in.solver_score = rt.solver_score;
                report_in.dihedrals = predict_csi_dihedrals(export_ref);
                let html = generate_quality_report_html(&report_in);
                let html_sha = sha256_hex(html.as_bytes());

                rt.artifacts.insert(out_rel.to_string(), nef_sha.clone());
                rt.artifacts.insert("report.html".to_string(), html_sha);
                if let Some(out_dir) = opts.output_dir.as_deref() {
                    let written_nef = DagRealRuntime::write_safe_output_file(
                        &node.id, out_dir, out_rel, &nef_str,
                    )?;
                    rt.written_files.push(written_nef);
                    let html_rel = format!("{out_rel}.report.html");
                    let written_html = DagRealRuntime::write_safe_output_file(
                        &node.id, out_dir, &html_rel, &html,
                    )?;
                    rt.written_files.push(written_html);
                }
                let short_sha: String = nef_sha.chars().take(12).collect();
                let nef_len = nef_str.len();
                let html_len = html.len();
                rt.nef_text = Some(nef_str);
                rt.report_html = Some(html);
                Ok(format!(
                    "export_nef_report nef_bytes={nef_len} html_bytes={html_len} sha256={short_sha}"
                ))
            }
        }
    })?;

    Ok(DagRunOutput {
        trace,
        artifacts: rt.artifacts,
        written_files: rt.written_files,
        nef_text: rt.nef_text,
        report_html: rt.report_html,
        metrics_json: rt.metrics_json,
    })
}

/// Look up a built-in canonical NMR workflow preset by short name (`backbone`, `dynamics`, `metabolomics`).
pub fn preset_by_name(name: &str) -> Option<WorkflowDag> {
    match name.trim().to_ascii_lowercase().as_str() {
        "backbone" | "backbone_assignment" | "backbone-assignment" => {
            Some(preset_backbone_assignment_dag())
        }
        "dynamics" | "dynamics_relax_cest" | "dynamics-relax-cest" => {
            Some(preset_dynamics_relax_cest_dag())
        }
        "metabolomics" | "metabolomics_1d" | "metabolomics-1d" => {
            Some(preset_metabolomics_1d_dag())
        }
        _ => None,
    }
}

/// Serialize a [`WorkflowDag`] to pretty-printed JSON respecting [`MAX_DAG_JSON_BYTES`].
pub fn dag_to_json(dag: &WorkflowDag) -> Result<String, DagError> {
    let encoded = serde_json::to_string_pretty(dag).map_err(|e| DagError::JsonParseError {
        message: e.to_string(),
    })?;
    if encoded.len() > MAX_DAG_JSON_BYTES {
        return Err(DagError::JsonTooLarge {
            bytes: encoded.len(),
            max: MAX_DAG_JSON_BYTES,
        });
    }
    Ok(encoded)
}

/// Parse a [`WorkflowDag`] from JSON with a fail-closed [`MAX_DAG_JSON_BYTES`] cap.
pub fn dag_from_json(json: &str) -> Result<WorkflowDag, DagError> {
    if json.len() > MAX_DAG_JSON_BYTES {
        return Err(DagError::JsonTooLarge {
            bytes: json.len(),
            max: MAX_DAG_JSON_BYTES,
        });
    }
    serde_json::from_str(json).map_err(|e| DagError::JsonParseError {
        message: e.to_string(),
    })
}

/// Serialize a compiled [`DagManifest`] to pretty-printed JSON respecting [`MAX_DAG_JSON_BYTES`].
pub fn manifest_to_json(manifest: &DagManifest) -> Result<String, DagError> {
    let encoded = serde_json::to_string_pretty(manifest).map_err(|e| DagError::JsonParseError {
        message: e.to_string(),
    })?;
    if encoded.len() > MAX_DAG_JSON_BYTES {
        return Err(DagError::JsonTooLarge {
            bytes: encoded.len(),
            max: MAX_DAG_JSON_BYTES,
        });
    }
    Ok(encoded)
}

/// Parse a compiled [`DagManifest`] from JSON with a fail-closed [`MAX_DAG_JSON_BYTES`] cap
/// and mandatory full cryptographic + topological integrity verification via [`verify_dag_manifest`].
pub fn manifest_from_json(json: &str) -> Result<DagManifest, DagError> {
    if json.len() > MAX_DAG_JSON_BYTES {
        return Err(DagError::JsonTooLarge {
            bytes: json.len(),
            max: MAX_DAG_JSON_BYTES,
        });
    }
    let manifest: DagManifest =
        serde_json::from_str(json).map_err(|e| DagError::JsonParseError {
            message: e.to_string(),
        })?;
    verify_dag_manifest(&manifest)?;
    Ok(manifest)
}

/// Built-in canonical preset: Protein Backbone Assignment (`Ingest -> Process -> Pick -> Assign -> Export`).
pub fn preset_backbone_assignment_dag() -> WorkflowDag {
    WorkflowDag {
        name: "Backbone Assignment Pipeline".to_string(),
        description: "End-to-end 2D/3D triple-resonance backbone assignment and NEF report bundle."
            .to_string(),
        nodes: vec![
            DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest Bruker")
                .with_position(24.0, 128.0)
                .with_param("dataset_path", serde_json::json!("datasets/backbone_3d")),
            DagNode::new("process", NodeKind::ProcessFidNus, "Process NUS 3D")
                .with_position(258.0, 128.0),
            DagNode::new("pick", NodeKind::PeakPick, "Two-Pass Pick").with_position(492.0, 128.0),
            DagNode::new("assign_bb", NodeKind::BackboneAssign, "Evo-Graph Assign")
                .with_position(726.0, 128.0),
            DagNode::new("export", NodeKind::ExportNefReport, "Export NEF")
                .with_position(960.0, 128.0)
                .with_param(
                    "output_path",
                    serde_json::json!("reports/backbone_assignments.nef"),
                ),
        ],
        edges: vec![
            DagEdge::new("ingest", "raw_fid", "process", "raw_fid"),
            DagEdge::new("process", "spectrum_2d", "pick", "spectrum_2d"),
            DagEdge::new("process", "spectrum_3d", "pick", "spectrum_3d"),
            DagEdge::new("pick", "peaks", "assign_bb", "peaks"),
            DagEdge::new("assign_bb", "assignments", "export", "assignments"),
        ],
    }
}

/// Built-in canonical preset: Protein Dynamics (`R1/R2/hetNOE` Relaxation + `CEST/CPMG` Exchange).
pub fn preset_dynamics_relax_cest_dag() -> WorkflowDag {
    WorkflowDag {
        name: "Dynamics Relax & CEST/CPMG".to_string(),
        description: "Parallel relaxation rate fitting and CEST/CPMG chemical exchange profiling."
            .to_string(),
        nodes: vec![
            DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest Dynamics")
                .with_position(24.0, 128.0)
                .with_param("dataset_path", serde_json::json!("datasets/dynamics_2d")),
            DagNode::new("process", NodeKind::ProcessFidNus, "Process 2D Series")
                .with_position(258.0, 128.0),
            DagNode::new("pick", NodeKind::PeakPick, "Pick 2D HSQC").with_position(492.0, 128.0),
            DagNode::new("relax", NodeKind::DynamicsRelax, "Relaxation R1/R2")
                .with_position(726.0, 24.0),
            DagNode::new("exchange", NodeKind::ExchangeCestCpmg, "CEST / CPMG Fit")
                .with_position(726.0, 224.0),
            DagNode::new("export", NodeKind::ExportNefReport, "Export Dynamics")
                .with_position(960.0, 128.0)
                .with_param(
                    "output_path",
                    serde_json::json!("reports/dynamics_exchange.nef"),
                ),
        ],
        edges: vec![
            DagEdge::new("ingest", "raw_fid", "process", "raw_fid"),
            DagEdge::new("process", "spectrum_2d", "pick", "spectrum_2d"),
            DagEdge::new("pick", "peaks", "relax", "peaks"),
            DagEdge::new("pick", "peaks", "exchange", "peaks"),
            DagEdge::new("relax", "rates", "export", "rates"),
            DagEdge::new("exchange", "profiles", "export", "profiles"),
        ],
    }
}

/// Built-in canonical preset: 1D Quantitative Metabolomics & Chemometrics (`PCA / PLS-DA`).
pub fn preset_metabolomics_1d_dag() -> WorkflowDag {
    WorkflowDag {
        name: "Metabolomics 1D Chemometrics".to_string(),
        description: "1D 1H cohort ingestion, FID processing, PQN normalization, and PCA/PLS-DA."
            .to_string(),
        nodes: vec![
            DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest 1D Cohort")
                .with_position(56.0, 128.0)
                .with_param(
                    "dataset_path",
                    serde_json::json!("datasets/metabolomics_1d"),
                ),
            DagNode::new("process", NodeKind::ProcessFidNus, "Process 1D FIDs")
                .with_position(346.0, 128.0)
                .with_param("nus_method", serde_json::json!("none")),
            DagNode::new(
                "chemometrics",
                NodeKind::MetabolomicsChemometrics,
                "PQN + PCA/PLS-DA",
            )
            .with_position(636.0, 128.0),
            DagNode::new("export", NodeKind::ExportNefReport, "Export Cohort")
                .with_position(926.0, 128.0)
                .with_param(
                    "output_path",
                    serde_json::json!("reports/metabolomics_cohort.nef"),
                ),
        ],
        edges: vec![
            DagEdge::new("ingest", "raw_fid", "process", "raw_fid"),
            DagEdge::new("process", "spectrum_1d", "chemometrics", "spectrum_1d"),
            DagEdge::new("chemometrics", "matrix", "export", "matrix"),
        ],
    }
}
