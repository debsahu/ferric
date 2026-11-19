//! Clean-room interactive `egui` typed node-and-port NMR workflow `DAG` composer panel (`T22.11`).
//!
//! Provenance: Original clean-room visual node-graph editor and manifest inspector for
//! `FERRIC` backed by [`nmr_core::workflow_dag`]. Supports native desktop and
//! `wasm32-unknown-unknown` targets with zero emojis and a strict `>= 11.0 px` typography floor.

use std::collections::BTreeMap;

use eframe::egui;
use nmr_core::workflow_dag::{
    compile_dag_manifest, execute_dag_manifest_dry_run, manifest_to_json,
    preset_backbone_assignment_dag, preset_dynamics_relax_cest_dag, preset_metabolomics_1d_dag,
    validate_candidate_edge, DagArtifactCache, DagEdge, DagError, DagExecutionTrace, DagManifest,
    DagNode, NodeKind, PortType, WorkflowDag, MAX_DAG_NODES,
};

use crate::workbench_mode::{font_px, MIN_FONT_PX};

/// Logical width of each workflow stage card on the canvas.
pub const DAG_NODE_WIDTH: f32 = 204.0;
/// Logical height of the title + kind header strip inside a node card.
pub const DAG_NODE_HEADER_H: f32 = 42.0;
/// Logical vertical pitch of each port row inside a node card.
pub const DAG_PORT_ROW_H: f32 = 21.0;
/// Logical bottom padding inside a node card.
pub const DAG_NODE_BOTTOM_PAD: f32 = 8.0;
/// Radius of each typed port circle.
pub const DAG_PORT_RADIUS: f32 = 5.0;

const MAX_VISIBLE_EDGE_ROWS: usize = 4;
const MAX_VISIBLE_JSON_LINES: usize = 8;
const MAX_JSON_LINE_CHARS: usize = 70;
const MAX_BANNER_ERR_CHARS: usize = 92;

/// Built-in workflow presets selectable in the composer toolbar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DagPresetKind {
    Backbone,
    Dynamics,
    Metabolomics,
}

impl DagPresetKind {
    pub const ALL: [DagPresetKind; 3] = [
        DagPresetKind::Backbone,
        DagPresetKind::Dynamics,
        DagPresetKind::Metabolomics,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            DagPresetKind::Backbone => "Backbone",
            DagPresetKind::Dynamics => "Dynamics",
            DagPresetKind::Metabolomics => "Metabolomics",
        }
    }
}

/// Optional host-facing events emitted during a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowDagAction {
    PresetLoaded(DagPresetKind),
    NodeAdded(String),
    NodeDeleted(String),
    EdgeConnected(DagEdge),
    EdgeRejected(String),
    EdgeDisconnected(DagEdge),
    DryRunCompleted {
        manifest_sha256: String,
        executed: usize,
        cache_hits: usize,
    },
}

/// Visual palette for `WorkflowDagPanel` (`Paper White` vs `Dark Oscilloscope`).
#[derive(Debug, Clone, Copy)]
pub struct DagTheme {
    pub canvas_bg: egui::Color32,
    pub grid_line: egui::Color32,
    pub card_bg: egui::Color32,
    pub card_header_bg: egui::Color32,
    pub card_border: egui::Color32,
    pub card_selected_border: egui::Color32,
    pub text_primary: egui::Color32,
    pub text_secondary: egui::Color32,
    pub valid_bg: egui::Color32,
    pub valid_fg: egui::Color32,
    pub invalid_bg: egui::Color32,
    pub invalid_fg: egui::Color32,
    pub code_bg: egui::Color32,
    pub code_border: egui::Color32,
}

impl DagTheme {
    pub const fn for_mode(paper_white: bool) -> Self {
        if paper_white {
            Self {
                canvas_bg: egui::Color32::from_rgb(248, 249, 251),
                grid_line: egui::Color32::from_rgb(228, 232, 238),
                card_bg: egui::Color32::from_rgb(255, 255, 255),
                card_header_bg: egui::Color32::from_rgb(236, 241, 248),
                card_border: egui::Color32::from_rgb(164, 174, 190),
                card_selected_border: egui::Color32::from_rgb(20, 96, 210),
                text_primary: egui::Color32::from_rgb(24, 30, 40),
                text_secondary: egui::Color32::from_rgb(76, 86, 102),
                valid_bg: egui::Color32::from_rgb(230, 246, 236),
                valid_fg: egui::Color32::from_rgb(18, 114, 58),
                invalid_bg: egui::Color32::from_rgb(252, 234, 234),
                invalid_fg: egui::Color32::from_rgb(176, 34, 34),
                code_bg: egui::Color32::from_rgb(244, 246, 250),
                code_border: egui::Color32::from_rgb(198, 206, 218),
            }
        } else {
            Self {
                canvas_bg: egui::Color32::from_rgb(18, 22, 28),
                grid_line: egui::Color32::from_rgb(30, 36, 46),
                card_bg: egui::Color32::from_rgb(28, 34, 44),
                card_header_bg: egui::Color32::from_rgb(38, 46, 60),
                card_border: egui::Color32::from_rgb(76, 88, 108),
                card_selected_border: egui::Color32::from_rgb(88, 166, 255),
                text_primary: egui::Color32::from_rgb(232, 238, 246),
                text_secondary: egui::Color32::from_rgb(162, 174, 192),
                valid_bg: egui::Color32::from_rgb(22, 54, 38),
                valid_fg: egui::Color32::from_rgb(112, 226, 152),
                invalid_bg: egui::Color32::from_rgb(64, 24, 28),
                invalid_fg: egui::Color32::from_rgb(255, 132, 132),
                code_bg: egui::Color32::from_rgb(22, 27, 34),
                code_border: egui::Color32::from_rgb(62, 72, 88),
            }
        }
    }
}

/// Deterministic color coding for all 13 strongly typed NMR data ports.
pub const fn port_type_color(port_type: PortType, paper_white: bool) -> egui::Color32 {
    if paper_white {
        match port_type {
            PortType::RawFid => egui::Color32::from_rgb(184, 92, 24),
            PortType::ProcessedSpectrum1D => egui::Color32::from_rgb(36, 118, 196),
            PortType::ProcessedSpectrum2D => egui::Color32::from_rgb(24, 142, 154),
            PortType::ProcessedSpectrum3D => egui::Color32::from_rgb(44, 92, 186),
            PortType::PeakList => egui::Color32::from_rgb(164, 116, 14),
            PortType::SpinSystems => egui::Color32::from_rgb(126, 68, 186),
            PortType::BackboneAssignments => egui::Color32::from_rgb(28, 144, 76),
            PortType::RestraintSet => egui::Color32::from_rgb(186, 56, 118),
            PortType::StructureModel => egui::Color32::from_rgb(20, 128, 112),
            PortType::RelaxationRates => egui::Color32::from_rgb(196, 72, 48),
            PortType::ExchangeProfiles => egui::Color32::from_rgb(154, 52, 154),
            PortType::ChemometricsMatrix => egui::Color32::from_rgb(76, 134, 34),
            PortType::ReportBundle => egui::Color32::from_rgb(72, 86, 108),
        }
    } else {
        match port_type {
            PortType::RawFid => egui::Color32::from_rgb(242, 148, 72),
            PortType::ProcessedSpectrum1D => egui::Color32::from_rgb(92, 178, 255),
            PortType::ProcessedSpectrum2D => egui::Color32::from_rgb(72, 210, 218),
            PortType::ProcessedSpectrum3D => egui::Color32::from_rgb(112, 156, 255),
            PortType::PeakList => egui::Color32::from_rgb(244, 198, 76),
            PortType::SpinSystems => egui::Color32::from_rgb(190, 138, 255),
            PortType::BackboneAssignments => egui::Color32::from_rgb(86, 218, 134),
            PortType::RestraintSet => egui::Color32::from_rgb(244, 118, 178),
            PortType::StructureModel => egui::Color32::from_rgb(82, 224, 198),
            PortType::RelaxationRates => egui::Color32::from_rgb(252, 126, 98),
            PortType::ExchangeProfiles => egui::Color32::from_rgb(224, 122, 232),
            PortType::ChemometricsMatrix => egui::Color32::from_rgb(152, 218, 92),
            PortType::ReportBundle => egui::Color32::from_rgb(184, 196, 214),
        }
    }
}

/// Geometry and validation telemetry recorded during the most recent rendered frame.
#[derive(Debug, Clone, Default)]
pub struct DagPanelTelemetry {
    pub nodes_drawn: usize,
    pub edges_drawn: usize,
    pub ports_drawn: usize,
    pub is_valid: bool,
    pub last_manifest_sha256: Option<String>,
    pub last_canvas_rect: Option<egui::Rect>,
    pub node_rects: BTreeMap<String, egui::Rect>,
    pub input_port_centers: BTreeMap<(String, String), egui::Pos2>,
    pub output_port_centers: BTreeMap<(String, String), egui::Pos2>,
}

/// Interactive typed node-and-port NMR workflow `DAG` composer widget.
pub struct WorkflowDagPanel {
    pub dag: WorkflowDag,
    pub active_preset: Option<DagPresetKind>,
    pub selected_node_id: Option<String>,
    pub new_node_kind: NodeKind,
    pub connect_from_node: String,
    pub connect_from_port: String,
    pub connect_to_node: String,
    pub connect_to_port: String,
    pub show_manifest_json: bool,
    pub manifest_page: usize,
    pub edge_page: usize,
    pub cache: DagArtifactCache,
    pub last_trace: Option<DagExecutionTrace>,
    pub last_interaction_error: Option<String>,
    pub compiled_result: Result<DagManifest, DagError>,
    pub cached_manifest_json: String,
    pub last_telemetry: DagPanelTelemetry,
    dragging_node: Option<(String, egui::Vec2)>,
    pending_wire: Option<(String, String)>,
    next_node_counter: usize,
}

impl Default for WorkflowDagPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkflowDagPanel {
    /// Construct a new panel pre-loaded with the canonical Backbone Assignment workflow preset.
    pub fn new() -> Self {
        let mut panel = Self {
            dag: preset_backbone_assignment_dag(),
            active_preset: Some(DagPresetKind::Backbone),
            selected_node_id: None,
            new_node_kind: NodeKind::DeconvolveNd,
            connect_from_node: String::new(),
            connect_from_port: String::new(),
            connect_to_node: String::new(),
            connect_to_port: String::new(),
            show_manifest_json: true,
            manifest_page: 0,
            edge_page: 0,
            cache: DagArtifactCache::new(),
            last_trace: None,
            last_interaction_error: None,
            compiled_result: Err(DagError::EmptyDag),
            cached_manifest_json: String::new(),
            last_telemetry: DagPanelTelemetry::default(),
            dragging_node: None,
            pending_wire: None,
            next_node_counter: 1,
        };
        panel.sync_connect_selectors();
        panel.refresh_validation();
        panel
    }

    /// Load one of the three built-in canonical NMR workflow presets.
    pub fn load_preset(&mut self, preset: DagPresetKind) {
        self.dag = match preset {
            DagPresetKind::Backbone => preset_backbone_assignment_dag(),
            DagPresetKind::Dynamics => preset_dynamics_relax_cest_dag(),
            DagPresetKind::Metabolomics => preset_metabolomics_1d_dag(),
        };
        self.active_preset = Some(preset);
        self.selected_node_id = self.dag.nodes.first().map(|n| n.id.clone());
        self.dragging_node = None;
        self.pending_wire = None;
        self.last_interaction_error = None;
        self.last_trace = None;
        self.manifest_page = 0;
        self.edge_page = 0;
        self.sync_connect_selectors();
        self.refresh_validation();
    }

    /// Re-validate and compile the current `WorkflowDag` into a content-addressed manifest.
    pub fn refresh_validation(&mut self) {
        self.compiled_result = compile_dag_manifest(&self.dag);
        self.cached_manifest_json = match &self.compiled_result {
            Ok(manifest) => manifest_to_json(manifest)
                .unwrap_or_else(|e| format!("{{\n  \"error\": \"{e}\"\n}}")),
            Err(err) => format!("{{\n  \"valid\": false,\n  \"diagnostic\": \"{err}\"\n}}"),
        };
    }

    /// Compute the rendered logical height of a stage node card.
    pub fn node_card_height(kind: NodeKind) -> f32 {
        let rows = kind
            .input_ports()
            .len()
            .max(kind.output_ports().len())
            .max(1);
        DAG_NODE_HEADER_H + (rows as f32) * DAG_PORT_ROW_H + DAG_NODE_BOTTOM_PAD
    }

    /// Add a new typed stage node with a unique ID and non-overlapping canvas coordinates.
    pub fn add_node(&mut self, kind: NodeKind) -> Result<String, DagError> {
        if self.dag.nodes.len() >= MAX_DAG_NODES {
            let err = DagError::TooManyNodes {
                count: self.dag.nodes.len() + 1,
                max: MAX_DAG_NODES,
            };
            self.last_interaction_error = Some(err.to_string());
            return Err(err);
        }
        let prefix = kind.id_prefix();
        let mut candidate_id = format!("{prefix}_{}", self.next_node_counter);
        while self.dag.nodes.iter().any(|n| n.id == candidate_id) {
            self.next_node_counter = self.next_node_counter.saturating_add(1);
            candidate_id = format!("{prefix}_{}", self.next_node_counter);
        }
        self.next_node_counter = self.next_node_counter.saturating_add(1);

        let idx = self.dag.nodes.len();
        let col = (idx % 5) as f32;
        let row = ((idx / 5) % 2) as f32;
        let x = 24.0 + col * 234.0;
        let y = 24.0 + row * 212.0;

        let label = format!(
            "{} {}",
            kind.display_name(),
            self.next_node_counter.saturating_sub(1)
        );
        self.dag
            .nodes
            .push(DagNode::new(candidate_id.clone(), kind, label).with_position(x, y));
        self.selected_node_id = Some(candidate_id.clone());
        self.last_interaction_error = None;
        self.sync_connect_selectors();
        self.refresh_validation();
        Ok(candidate_id)
    }

    /// Delete a node by ID along with all incident edges.
    pub fn delete_node(&mut self, node_id: &str) -> bool {
        let before = self.dag.nodes.len();
        self.dag.nodes.retain(|n| n.id != node_id);
        if self.dag.nodes.len() == before {
            return false;
        }
        self.dag
            .edges
            .retain(|e| e.from_node != node_id && e.to_node != node_id);
        if self.selected_node_id.as_deref() == Some(node_id) {
            self.selected_node_id = self.dag.nodes.first().map(|n| n.id.clone());
        }
        if self
            .pending_wire
            .as_ref()
            .is_some_and(|(nid, _)| nid == node_id)
        {
            self.pending_wire = None;
        }
        self.last_interaction_error = None;
        self.sync_connect_selectors();
        self.refresh_validation();
        true
    }

    /// Attempt to add a typed edge; fails closed on port type mismatch, duplicate, or cycle.
    pub fn connect_edge(&mut self, edge: DagEdge) -> Result<(), DagError> {
        match validate_candidate_edge(&self.dag, &edge) {
            Ok(_) => {
                self.dag.edges.push(edge);
                self.last_interaction_error = None;
                self.refresh_validation();
                Ok(())
            }
            Err(err) => {
                self.last_interaction_error = Some(err.to_string());
                Err(err)
            }
        }
    }

    /// Remove an edge by index.
    pub fn disconnect_edge(&mut self, index: usize) -> Option<DagEdge> {
        if index >= self.dag.edges.len() {
            return None;
        }
        let removed = self.dag.edges.remove(index);
        self.last_interaction_error = None;
        self.refresh_validation();
        Some(removed)
    }

    /// Execute a deterministic cached dry-run of the compiled manifest.
    pub fn run_dry_run(&mut self) -> Result<DagExecutionTrace, DagError> {
        self.refresh_validation();
        let manifest = match &self.compiled_result {
            Ok(m) => m.clone(),
            Err(err) => {
                self.last_interaction_error = Some(err.to_string());
                return Err(err.clone());
            }
        };
        let trace = execute_dag_manifest_dry_run(&manifest, &mut self.cache)?;
        self.last_trace = Some(trace.clone());
        self.last_interaction_error = None;
        Ok(trace)
    }

    /// Return the current pretty-printed JSON manifest preview string.
    pub fn manifest_json_preview(&self) -> &str {
        &self.cached_manifest_json
    }

    fn sync_connect_selectors(&mut self) {
        if !self
            .dag
            .nodes
            .iter()
            .any(|n| n.id == self.connect_from_node)
        {
            self.connect_from_node = self
                .dag
                .nodes
                .first()
                .map(|n| n.id.clone())
                .unwrap_or_default();
        }
        if !self.dag.nodes.iter().any(|n| n.id == self.connect_to_node) {
            self.connect_to_node = self
                .dag
                .nodes
                .get(1)
                .or_else(|| self.dag.nodes.first())
                .map(|n| n.id.clone())
                .unwrap_or_default();
        }
        if let Some(from_node) = self
            .dag
            .nodes
            .iter()
            .find(|n| n.id == self.connect_from_node)
        {
            let valid = from_node
                .kind
                .find_output_port(&self.connect_from_port)
                .is_some();
            if !valid {
                self.connect_from_port = from_node
                    .kind
                    .output_ports()
                    .first()
                    .map(|p| p.name.to_string())
                    .unwrap_or_default();
            }
        } else {
            self.connect_from_port.clear();
        }
        if let Some(to_node) = self.dag.nodes.iter().find(|n| n.id == self.connect_to_node) {
            let valid = to_node
                .kind
                .find_input_port(&self.connect_to_port)
                .is_some();
            if !valid {
                self.connect_to_port = to_node
                    .kind
                    .input_ports()
                    .first()
                    .map(|p| p.name.to_string())
                    .unwrap_or_default();
            }
        } else {
            self.connect_to_port.clear();
        }
    }

    /// Render the interactive `WorkflowDagPanel` inside `ui`.
    pub fn show(&mut self, ui: &mut egui::Ui, paper_white: bool) -> Vec<WorkflowDagAction> {
        let mut actions = Vec::new();
        let theme = DagTheme::for_mode(paper_white);

        // Keep validation synced if caller mutated `self.dag` externally.
        self.refresh_validation();
        self.sync_connect_selectors();

        let body_sz = font_px(12.0);
        let small_sz = font_px(MIN_FONT_PX);

        // 1. Top Toolbar
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Presets:").size(body_sz).strong());
            for preset in DagPresetKind::ALL {
                let selected = self.active_preset == Some(preset);
                if ui
                    .selectable_label(selected, egui::RichText::new(preset.label()).size(body_sz))
                    .clicked()
                {
                    self.load_preset(preset);
                    actions.push(WorkflowDagAction::PresetLoaded(preset));
                }
            }

            ui.separator();

            ui.label(egui::RichText::new("Stage:").size(body_sz));
            egui::ComboBox::from_id_salt("workflow_dag_add_kind")
                .width(164.0)
                .selected_text(
                    egui::RichText::new(self.new_node_kind.display_name()).size(small_sz),
                )
                .show_ui(ui, |ui| {
                    for kind in NodeKind::ALL {
                        ui.selectable_value(
                            &mut self.new_node_kind,
                            kind,
                            egui::RichText::new(kind.display_name()).size(small_sz),
                        );
                    }
                });
            if ui
                .button(egui::RichText::new("Add Node").size(body_sz))
                .clicked()
            {
                if let Ok(new_id) = self.add_node(self.new_node_kind) {
                    actions.push(WorkflowDagAction::NodeAdded(new_id));
                }
            }

            ui.separator();

            if ui
                .button(egui::RichText::new("Dry Run").size(body_sz))
                .clicked()
            {
                if let Ok(trace) = self.run_dry_run() {
                    actions.push(WorkflowDagAction::DryRunCompleted {
                        manifest_sha256: trace.manifest_sha256,
                        executed: trace.executed_count,
                        cache_hits: trace.cache_hit_count,
                    });
                }
            }

            let json_btn = if self.show_manifest_json {
                "Hide JSON"
            } else {
                "Show JSON"
            };
            if ui
                .button(egui::RichText::new(json_btn).size(body_sz))
                .clicked()
            {
                self.show_manifest_json = !self.show_manifest_json;
            }
        });

        ui.add_space(4.0);

        // 2. Live Validation Status Banner (single fixed-height bar so canvas never jumps on error)
        let mut clear_interaction_error = false;
        let (banner_bg, banner_fg, badge_text, detail_text, show_dismiss) =
            if let Some(err_msg) = &self.last_interaction_error {
                (
                    theme.invalid_bg,
                    theme.invalid_fg,
                    "Rejected:",
                    truncate_ascii(err_msg, MAX_BANNER_ERR_CHARS),
                    true,
                )
            } else {
                match &self.compiled_result {
                    Ok(manifest) => {
                        let short_sha: String = manifest.manifest_sha256.chars().take(12).collect();
                        let trace_suffix = match &self.last_trace {
                            Some(t) => format!(
                                " | Run: {} exec, {} cached",
                                t.executed_count, t.cache_hit_count
                            ),
                            None => String::new(),
                        };
                        (
                            theme.valid_bg,
                            theme.valid_fg,
                            "Valid DAG",
                            format!(
                                "{} nodes, {} edges | {} | sha256: {}{}",
                                manifest.nodes.len(),
                                manifest.edges.len(),
                                manifest.schema,
                                short_sha,
                                trace_suffix
                            ),
                            false,
                        )
                    }
                    Err(err) => (
                        theme.invalid_bg,
                        theme.invalid_fg,
                        "Invalid DAG",
                        truncate_ascii(&err.to_string(), MAX_BANNER_ERR_CHARS),
                        false,
                    ),
                }
            };

        egui::Frame::new()
            .fill(banner_bg)
            .corner_radius(4.0)
            .inner_margin(egui::Margin::symmetric(8, 5))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(badge_text)
                            .size(body_sz)
                            .strong()
                            .color(banner_fg),
                    );
                    ui.label(
                        egui::RichText::new(detail_text)
                            .size(small_sz)
                            .monospace()
                            .color(theme.text_primary),
                    );
                    if show_dismiss
                        && ui
                            .button(egui::RichText::new("Dismiss").size(small_sz))
                            .clicked()
                    {
                        clear_interaction_error = true;
                    }
                });
            });
        if clear_interaction_error {
            self.last_interaction_error = None;
        }

        ui.add_space(6.0);

        // 3. Interactive Node-and-Port DAG Canvas
        let inspector_reserve = 224.0_f32;
        let canvas_h = (ui.available_height() - inspector_reserve).max(340.0);
        let canvas_w = ui.available_width().max(400.0);
        let (canvas_rect, canvas_resp) = ui.allocate_exact_size(
            egui::vec2(canvas_w, canvas_h),
            egui::Sense::click_and_drag(),
        );

        self.render_canvas(
            ui,
            canvas_rect,
            &canvas_resp,
            paper_white,
            &theme,
            &mut actions,
        );

        ui.add_space(6.0);

        // 4. Bottom Inspector & JSON Manifest Preview Strip
        self.render_bottom_inspector(ui, &theme, &mut actions);

        actions
    }

    fn render_canvas(
        &mut self,
        ui: &mut egui::Ui,
        canvas_rect: egui::Rect,
        canvas_resp: &egui::Response,
        paper_white: bool,
        theme: &DagTheme,
        actions: &mut Vec<WorkflowDagAction>,
    ) {
        let painter = ui.painter_at(canvas_rect);

        // Canvas backdrop & subtle orthogonal grid
        painter.rect_filled(canvas_rect, 4.0, theme.canvas_bg);
        let grid_step = 32.0_f32;
        let mut gx = canvas_rect.min.x + grid_step;
        while gx < canvas_rect.max.x {
            painter.line_segment(
                [
                    egui::pos2(gx, canvas_rect.min.y),
                    egui::pos2(gx, canvas_rect.max.y),
                ],
                egui::Stroke::new(1.0, theme.grid_line),
            );
            gx += grid_step;
        }
        let mut gy = canvas_rect.min.y + grid_step;
        while gy < canvas_rect.max.y {
            painter.line_segment(
                [
                    egui::pos2(canvas_rect.min.x, gy),
                    egui::pos2(canvas_rect.max.x, gy),
                ],
                egui::Stroke::new(1.0, theme.grid_line),
            );
            gy += grid_step;
        }
        painter.rect_stroke(
            canvas_rect,
            4.0,
            egui::Stroke::new(1.0, theme.card_border),
            egui::StrokeKind::Inside,
        );

        // Precompute node rects and typed port centers
        let mut node_rects: BTreeMap<String, egui::Rect> = BTreeMap::new();
        let mut input_centers: BTreeMap<(String, String), egui::Pos2> = BTreeMap::new();
        let mut output_centers: BTreeMap<(String, String), egui::Pos2> = BTreeMap::new();
        let mut ports_drawn = 0usize;

        for node in &self.dag.nodes {
            let h = Self::node_card_height(node.kind);
            let min = egui::pos2(
                canvas_rect.min.x + node.position[0],
                canvas_rect.min.y + node.position[1],
            );
            let rect = egui::Rect::from_min_size(min, egui::vec2(DAG_NODE_WIDTH, h));
            node_rects.insert(node.id.clone(), rect);

            for (idx, port) in node.kind.input_ports().iter().enumerate() {
                let cy = rect.min.y + DAG_NODE_HEADER_H + (idx as f32 + 0.5) * DAG_PORT_ROW_H;
                input_centers.insert(
                    (node.id.clone(), port.name.to_string()),
                    egui::pos2(rect.min.x + 8.0, cy),
                );
                ports_drawn += 1;
            }
            for (idx, port) in node.kind.output_ports().iter().enumerate() {
                let cy = rect.min.y + DAG_NODE_HEADER_H + (idx as f32 + 0.5) * DAG_PORT_ROW_H;
                output_centers.insert(
                    (node.id.clone(), port.name.to_string()),
                    egui::pos2(rect.max.x - 8.0, cy),
                );
                ports_drawn += 1;
            }
        }

        // Pointer interactions: port wiring, node selection, node dragging
        let pointer_pos = ui.input(|i| i.pointer.interact_pos());
        let primary_pressed = ui.input(|i| i.pointer.primary_pressed());
        let primary_down = ui.input(|i| i.pointer.primary_down());
        let primary_released = ui.input(|i| i.pointer.primary_released());

        if primary_pressed && canvas_resp.hovered() {
            if let Some(pos) = pointer_pos {
                let mut handled = false;

                // 1. Check click on output port -> start pending wire
                for ((nid, pname), center) in &output_centers {
                    if pos.distance(*center) <= 11.0 {
                        self.pending_wire = Some((nid.clone(), pname.clone()));
                        self.selected_node_id = Some(nid.clone());
                        handled = true;
                        break;
                    }
                }

                // 2. Check click on input port while pending wire is active -> complete edge
                if !handled {
                    if let Some((from_node, from_port)) = self.pending_wire.clone() {
                        for ((to_node, to_port), center) in &input_centers {
                            if pos.distance(*center) <= 13.0 {
                                let edge = DagEdge::new(
                                    from_node.clone(),
                                    from_port.clone(),
                                    to_node.clone(),
                                    to_port.clone(),
                                );
                                match self.connect_edge(edge.clone()) {
                                    Ok(()) => actions.push(WorkflowDagAction::EdgeConnected(edge)),
                                    Err(err) => actions
                                        .push(WorkflowDagAction::EdgeRejected(err.to_string())),
                                }
                                self.pending_wire = None;
                                handled = true;
                                break;
                            }
                        }
                    }
                }

                // 3. Check click inside a node card -> select & begin drag
                if !handled {
                    for node in self.dag.nodes.iter().rev() {
                        if let Some(rect) = node_rects.get(&node.id) {
                            if rect.contains(pos) {
                                self.selected_node_id = Some(node.id.clone());
                                self.dragging_node = Some((node.id.clone(), pos - rect.min));
                                self.pending_wire = None;
                                handled = true;
                                break;
                            }
                        }
                    }
                }

                if !handled {
                    self.pending_wire = None;
                }
            }
        }

        if primary_down {
            if let (Some((drag_id, grab_offset)), Some(pos)) = (&self.dragging_node, pointer_pos) {
                if let Some(node) = self.dag.nodes.iter_mut().find(|n| &n.id == drag_id) {
                    let h = Self::node_card_height(node.kind);
                    let max_x = (canvas_rect.width() - DAG_NODE_WIDTH - 8.0).max(8.0);
                    let max_y = (canvas_rect.height() - h - 8.0).max(8.0);
                    let new_x = (pos.x - grab_offset.x - canvas_rect.min.x).clamp(8.0, max_x);
                    let new_y = (pos.y - grab_offset.y - canvas_rect.min.y).clamp(8.0, max_y);
                    node.position = [new_x, new_y];
                }
            }
        }
        if primary_released {
            self.dragging_node = None;
        }

        // Draw directed cubic Bezier edges
        let mut edges_drawn = 0usize;
        for edge in &self.dag.edges {
            let from_pos = output_centers.get(&(edge.from_node.clone(), edge.from_port.clone()));
            let to_pos = input_centers.get(&(edge.to_node.clone(), edge.to_port.clone()));
            if let (Some(&p0), Some(&p1)) = (from_pos, to_pos) {
                let port_color = self
                    .dag
                    .nodes
                    .iter()
                    .find(|n| n.id == edge.from_node)
                    .and_then(|n| n.kind.find_output_port(&edge.from_port))
                    .map(|spec| port_type_color(spec.port_type, paper_white))
                    .unwrap_or(theme.card_selected_border);

                draw_bezier_edge(&painter, p0, p1, port_color, 2.2);
                edges_drawn += 1;
            }
        }

        // Draw interactive pending wire preview if active
        if let (Some((from_node, from_port)), Some(cursor)) = (&self.pending_wire, pointer_pos) {
            if let Some(&p0) = output_centers.get(&(from_node.clone(), from_port.clone())) {
                draw_bezier_edge(&painter, p0, cursor, theme.card_selected_border, 1.8);
            }
        }

        // Draw stage node cards and typed ports
        let title_font = egui::FontId::proportional(font_px(12.5));
        let sub_font = egui::FontId::proportional(font_px(MIN_FONT_PX));
        let port_font = egui::FontId::proportional(font_px(MIN_FONT_PX));

        for node in &self.dag.nodes {
            let Some(&rect) = node_rects.get(&node.id) else {
                continue;
            };
            let is_selected = self.selected_node_id.as_deref() == Some(node.id.as_str());

            // Card body & header
            painter.rect_filled(rect, 6.0, theme.card_bg);
            let header_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), DAG_NODE_HEADER_H));
            painter.rect_filled(header_rect, 6.0, theme.card_header_bg);

            let border_color = if is_selected {
                theme.card_selected_border
            } else {
                theme.card_border
            };
            let border_width = if is_selected { 2.0 } else { 1.0 };
            painter.rect_stroke(
                rect,
                6.0,
                egui::Stroke::new(border_width, border_color),
                egui::StrokeKind::Inside,
            );

            // Title and ID subtitle (strictly non-overlapping vertical layout)
            let title_str = truncate_ascii(&node.label, 22);
            painter.text(
                egui::pos2(rect.min.x + 9.0, rect.min.y + 5.0),
                egui::Align2::LEFT_TOP,
                title_str,
                title_font.clone(),
                theme.text_primary,
            );
            let sub_str = truncate_ascii(&format!("id: {}", node.id), 25);
            painter.text(
                egui::pos2(rect.min.x + 9.0, rect.min.y + 23.0),
                egui::Align2::LEFT_TOP,
                sub_str,
                sub_font.clone(),
                theme.text_secondary,
            );

            // Input ports (left side)
            for port in node.kind.input_ports() {
                if let Some(&center) = input_centers.get(&(node.id.clone(), port.name.to_string()))
                {
                    let col = port_type_color(port.port_type, paper_white);
                    painter.circle_filled(center, DAG_PORT_RADIUS, col);
                    painter.circle_stroke(
                        center,
                        DAG_PORT_RADIUS,
                        egui::Stroke::new(1.0, theme.card_bg),
                    );
                    let lbl = truncate_ascii(port.name, 12);
                    painter.text(
                        egui::pos2(center.x + 8.0, center.y),
                        egui::Align2::LEFT_CENTER,
                        lbl,
                        port_font.clone(),
                        theme.text_primary,
                    );
                }
            }

            // Output ports (right side)
            for port in node.kind.output_ports() {
                if let Some(&center) = output_centers.get(&(node.id.clone(), port.name.to_string()))
                {
                    let col = port_type_color(port.port_type, paper_white);
                    painter.circle_filled(center, DAG_PORT_RADIUS, col);
                    painter.circle_stroke(
                        center,
                        DAG_PORT_RADIUS,
                        egui::Stroke::new(1.0, theme.card_bg),
                    );
                    let lbl = truncate_ascii(port.name, 12);
                    painter.text(
                        egui::pos2(center.x - 8.0, center.y),
                        egui::Align2::RIGHT_CENTER,
                        lbl,
                        port_font.clone(),
                        theme.text_primary,
                    );
                }
            }
        }

        self.last_telemetry = DagPanelTelemetry {
            nodes_drawn: self.dag.nodes.len(),
            edges_drawn,
            ports_drawn,
            is_valid: self.compiled_result.is_ok(),
            last_manifest_sha256: self
                .compiled_result
                .as_ref()
                .ok()
                .map(|m| m.manifest_sha256.clone()),
            last_canvas_rect: Some(canvas_rect),
            node_rects,
            input_port_centers: input_centers,
            output_port_centers: output_centers,
        };
    }

    fn render_bottom_inspector(
        &mut self,
        ui: &mut egui::Ui,
        theme: &DagTheme,
        actions: &mut Vec<WorkflowDagAction>,
    ) {
        let body_sz = font_px(12.0);
        let small_sz = font_px(MIN_FONT_PX);

        if self.show_manifest_json {
            ui.columns(2, |cols| {
                self.render_node_edge_controls(&mut cols[0], theme, body_sz, small_sz, actions);
                self.render_manifest_preview(&mut cols[1], theme, body_sz, small_sz);
            });
        } else {
            self.render_node_edge_controls(ui, theme, body_sz, small_sz, actions);
        }
    }

    fn render_node_edge_controls(
        &mut self,
        ui: &mut egui::Ui,
        theme: &DagTheme,
        body_sz: f32,
        small_sz: f32,
        actions: &mut Vec<WorkflowDagAction>,
    ) {
        egui::Frame::new()
            .fill(theme.code_bg)
            .stroke(egui::Stroke::new(1.0, theme.code_border))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());

                // Selected node bar
                ui.horizontal(|ui| {
                    let sel_text = match &self.selected_node_id {
                        Some(id) => {
                            let kind_str = self
                                .dag
                                .nodes
                                .iter()
                                .find(|n| &n.id == id)
                                .map(|n| n.kind.display_name())
                                .unwrap_or("Unknown");
                            format!("Selected: {} ({})", truncate_ascii(id, 18), kind_str)
                        }
                        None => "Selected: none".to_string(),
                    };
                    ui.label(egui::RichText::new(sel_text).size(body_sz).strong());
                    let can_del = self.selected_node_id.is_some();
                    if ui
                        .add_enabled(
                            can_del,
                            egui::Button::new(egui::RichText::new("Delete Node").size(small_sz)),
                        )
                        .clicked()
                    {
                        if let Some(id) = self.selected_node_id.clone() {
                            if self.delete_node(&id) {
                                actions.push(WorkflowDagAction::NodeDeleted(id));
                            }
                        }
                    }
                });

                ui.add_space(4.0);

                // Connect edge controls
                let node_ids: Vec<String> = self.dag.nodes.iter().map(|n| n.id.clone()).collect();
                let from_ports: Vec<&'static str> = self
                    .dag
                    .nodes
                    .iter()
                    .find(|n| n.id == self.connect_from_node)
                    .map(|n| n.kind.output_ports().iter().map(|p| p.name).collect())
                    .unwrap_or_default();
                let to_ports: Vec<&'static str> = self
                    .dag
                    .nodes
                    .iter()
                    .find(|n| n.id == self.connect_to_node)
                    .map(|n| n.kind.input_ports().iter().map(|p| p.name).collect())
                    .unwrap_or_default();

                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("dag_conn_from_node")
                        .width(88.0)
                        .selected_text(
                            egui::RichText::new(truncate_ascii(&self.connect_from_node, 11))
                                .size(small_sz),
                        )
                        .show_ui(ui, |ui| {
                            for id in &node_ids {
                                ui.selectable_value(
                                    &mut self.connect_from_node,
                                    id.clone(),
                                    egui::RichText::new(id).size(small_sz),
                                );
                            }
                        });
                    egui::ComboBox::from_id_salt("dag_conn_from_port")
                        .width(88.0)
                        .selected_text(
                            egui::RichText::new(truncate_ascii(&self.connect_from_port, 11))
                                .size(small_sz),
                        )
                        .show_ui(ui, |ui| {
                            for p in &from_ports {
                                ui.selectable_value(
                                    &mut self.connect_from_port,
                                    (*p).to_string(),
                                    egui::RichText::new(*p).size(small_sz),
                                );
                            }
                        });
                    ui.label(egui::RichText::new("->").size(small_sz));
                    egui::ComboBox::from_id_salt("dag_conn_to_node")
                        .width(88.0)
                        .selected_text(
                            egui::RichText::new(truncate_ascii(&self.connect_to_node, 11))
                                .size(small_sz),
                        )
                        .show_ui(ui, |ui| {
                            for id in &node_ids {
                                ui.selectable_value(
                                    &mut self.connect_to_node,
                                    id.clone(),
                                    egui::RichText::new(id).size(small_sz),
                                );
                            }
                        });
                    egui::ComboBox::from_id_salt("dag_conn_to_port")
                        .width(88.0)
                        .selected_text(
                            egui::RichText::new(truncate_ascii(&self.connect_to_port, 11))
                                .size(small_sz),
                        )
                        .show_ui(ui, |ui| {
                            for p in &to_ports {
                                ui.selectable_value(
                                    &mut self.connect_to_port,
                                    (*p).to_string(),
                                    egui::RichText::new(*p).size(small_sz),
                                );
                            }
                        });
                    if ui
                        .button(egui::RichText::new("Connect").size(small_sz))
                        .clicked()
                    {
                        let edge = DagEdge::new(
                            self.connect_from_node.clone(),
                            self.connect_from_port.clone(),
                            self.connect_to_node.clone(),
                            self.connect_to_port.clone(),
                        );
                        match self.connect_edge(edge.clone()) {
                            Ok(()) => actions.push(WorkflowDagAction::EdgeConnected(edge)),
                            Err(err) => {
                                actions.push(WorkflowDagAction::EdgeRejected(err.to_string()))
                            }
                        }
                    }
                });

                ui.add_space(4.0);

                // Paginated edge list (zero partial-clip scroll lines)
                let total_edges = self.dag.edges.len();
                let max_page = total_edges.saturating_sub(1) / MAX_VISIBLE_EDGE_ROWS;
                if self.edge_page > max_page {
                    self.edge_page = max_page;
                }
                let start = self.edge_page * MAX_VISIBLE_EDGE_ROWS;
                let end = (start + MAX_VISIBLE_EDGE_ROWS).min(total_edges);

                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("Edges ({total_edges}):"))
                            .size(small_sz)
                            .strong(),
                    );
                    if total_edges > MAX_VISIBLE_EDGE_ROWS {
                        if ui
                            .add_enabled(
                                self.edge_page > 0,
                                egui::Button::new(egui::RichText::new("Prev").size(small_sz)),
                            )
                            .clicked()
                        {
                            self.edge_page = self.edge_page.saturating_sub(1);
                        }
                        if ui
                            .add_enabled(
                                self.edge_page < max_page,
                                egui::Button::new(egui::RichText::new("Next").size(small_sz)),
                            )
                            .clicked()
                        {
                            self.edge_page += 1;
                        }
                    }
                });

                let mut disconnect_idx: Option<usize> = None;
                for idx in start..end {
                    if let Some(edge) = self.dag.edges.get(idx) {
                        ui.horizontal(|ui| {
                            let desc = format!(
                                "{}:{} -> {}:{}",
                                edge.from_node, edge.from_port, edge.to_node, edge.to_port
                            );
                            ui.label(
                                egui::RichText::new(truncate_ascii(&desc, 52))
                                    .size(small_sz)
                                    .monospace(),
                            );
                            if ui
                                .button(egui::RichText::new("Disconnect").size(small_sz))
                                .clicked()
                            {
                                disconnect_idx = Some(idx);
                            }
                        });
                    }
                }
                if let Some(idx) = disconnect_idx {
                    if let Some(removed) = self.disconnect_edge(idx) {
                        actions.push(WorkflowDagAction::EdgeDisconnected(removed));
                    }
                }
            });
    }

    fn render_manifest_preview(
        &mut self,
        ui: &mut egui::Ui,
        theme: &DagTheme,
        body_sz: f32,
        small_sz: f32,
    ) {
        egui::Frame::new()
            .fill(theme.code_bg)
            .stroke(egui::Stroke::new(1.0, theme.code_border))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());

                let all_lines: Vec<&str> = self.cached_manifest_json.lines().collect();
                let total_pages = all_lines.len().div_ceil(MAX_VISIBLE_JSON_LINES).max(1);
                if self.manifest_page >= total_pages {
                    self.manifest_page = total_pages.saturating_sub(1);
                }

                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Manifest Export Preview")
                            .size(body_sz)
                            .strong(),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "Page {}/{}",
                            self.manifest_page + 1,
                            total_pages
                        ))
                        .size(small_sz)
                        .color(theme.text_secondary),
                    );
                    if ui
                        .add_enabled(
                            self.manifest_page > 0,
                            egui::Button::new(egui::RichText::new("Prev").size(small_sz)),
                        )
                        .clicked()
                    {
                        self.manifest_page = self.manifest_page.saturating_sub(1);
                    }
                    if ui
                        .add_enabled(
                            self.manifest_page + 1 < total_pages,
                            egui::Button::new(egui::RichText::new("Next").size(small_sz)),
                        )
                        .clicked()
                    {
                        self.manifest_page += 1;
                    }
                });

                ui.add_space(4.0);

                let start = self.manifest_page * MAX_VISIBLE_JSON_LINES;
                let end = (start + MAX_VISIBLE_JSON_LINES).min(all_lines.len());
                for line in &all_lines[start..end] {
                    ui.label(
                        egui::RichText::new(truncate_ascii(line, MAX_JSON_LINE_CHARS))
                            .size(small_sz)
                            .monospace()
                            .color(theme.text_primary),
                    );
                }
            });
    }
}

fn draw_bezier_edge(
    painter: &egui::Painter,
    p0: egui::Pos2,
    p1: egui::Pos2,
    color: egui::Color32,
    width: f32,
) {
    let dx = (p1.x - p0.x).abs().max(48.0) * 0.45;
    let c1 = egui::pos2(p0.x + dx, p0.y);
    let c2 = egui::pos2(p1.x - dx, p1.y);
    let bezier = egui::epaint::CubicBezierShape::from_points_stroke(
        [p0, c1, c2, p1],
        false,
        egui::Color32::TRANSPARENT,
        egui::Stroke::new(width, color),
    );
    painter.add(egui::Shape::CubicBezier(bezier));

    // Arrowhead triangle pointing into destination input port
    let tip = egui::pos2(p1.x - DAG_PORT_RADIUS, p1.y);
    let arrow = vec![
        tip,
        egui::pos2(tip.x - 8.0, tip.y - 4.0),
        egui::pos2(tip.x - 8.0, tip.y + 4.0),
    ];
    painter.add(egui::Shape::convex_polygon(
        arrow,
        color,
        egui::Stroke::NONE,
    ));
}

fn truncate_ascii(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    if max_chars <= 3 {
        return text.chars().take(max_chars).collect();
    }
    let mut out: String = text.chars().take(max_chars - 3).collect();
    out.push_str("...");
    out
}
