//! Intuitive 3-Step `Assignment Setup` Drawer for protein backbone resonance assignment.
//!
//! Provenance: Original clean-room human-computer interaction and progressive-disclosure workflow
//! design for `FERRIC` (`docs/specs/06-gui-workbench.md`, `docs/specs/12-workbench-ux.md`),
//! unifying sequence entry, 2D/3D triple-resonance spectrum readiness & unified peak picking,
//! solver/regime configuration, and post-assignment confidence triage into one structured drawer.

use eframe::egui;

use crate::workbench_mode::{TYPE_BODY_PX, TYPE_CAPTION_PX, TYPE_HEADER_PX};

/// Canonical 7 backbone triple-resonance experiment labels displayed in Step 2.
pub const CANONICAL_BACKBONE_EXPERIMENTS: [&str; 7] = [
    "HSQC",
    "HNCACB",
    "CBCA(CO)NH",
    "HNCA",
    "HN(CO)CA",
    "HNCO",
    "HN(CA)CO",
];

/// Assignment solver engine selectable in Step 3.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AssignSolverEngine {
    #[default]
    EvoGraph,
    StandardSa,
    BayesSa,
}

impl AssignSolverEngine {
    pub const ALL: [Self; 3] = [Self::EvoGraph, Self::StandardSa, Self::BayesSa];

    pub const fn label(self) -> &'static str {
        match self {
            Self::EvoGraph => "Evo-Graph",
            Self::StandardSa => "Standard SA",
            Self::BayesSa => "Bayes-SA",
        }
    }
}

/// Protein / spectroscopy regime selectable in Step 3.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AssignProteinRegime {
    #[default]
    Globular,
    Idp13C,
    Paramagnetic,
}

impl AssignProteinRegime {
    pub const ALL: [Self; 3] = [Self::Globular, Self::Idp13C, Self::Paramagnetic];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Globular => "Globular",
            Self::Idp13C => "13C-IDP",
            Self::Paramagnetic => "Paramagnetic",
        }
    }
}

/// One row of the Step 2 7-slot spectra & picked peaks data-input mapper.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentChecklistRow {
    /// Canonical backbone slot label (`HSQC`, `HNCACB`, `CBCA(CO)NH`, `HNCA`, `HN(CO)CA`, `HNCO`,
    /// `HN(CA)CO`).
    pub exp_name: String,
    /// True when a spectrum or peak list is loaded for this slot.
    pub loaded: bool,
    /// Real picked peak count for this slot.
    pub peak_count: usize,
    /// Optional loaded source filename / folder / expno label (shown when loaded).
    pub source_label: Option<String>,
    /// Optional Bruker `PULPROG` or pulse-sequence tag (shown alongside or in place of source).
    pub pulprog: Option<String>,
    /// Optional 3D-to-HSQC amide registration drift `(delta_h_ppm, delta_n_ppm)` in ppm.
    pub registration_drift_ppm: Option<(f64, f64)>,
}

impl ExperimentChecklistRow {
    pub fn new(exp_name: impl Into<String>, loaded: bool, peak_count: usize) -> Self {
        Self {
            exp_name: exp_name.into(),
            loaded,
            peak_count,
            source_label: None,
            pulprog: None,
            registration_drift_ppm: None,
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        let s = source.into();
        self.source_label = if s.trim().is_empty() { None } else { Some(s) };
        self
    }

    pub fn with_pulprog(mut self, pulprog: impl Into<String>) -> Self {
        let s = pulprog.into();
        self.pulprog = if s.trim().is_empty() { None } else { Some(s) };
        self
    }

    pub fn with_registration_drift(mut self, delta_h_ppm: f64, delta_n_ppm: f64) -> Self {
        if delta_h_ppm.is_finite() && delta_n_ppm.is_finite() {
            self.registration_drift_ppm = Some((delta_h_ppm, delta_n_ppm));
        }
        self
    }

    /// Human-readable loaded source / `pulprog` summary, or `"Not loaded"` when empty.
    pub fn display_source(&self) -> String {
        if !self.loaded {
            return "Not loaded".to_string();
        }
        match (
            self.source_label
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty()),
            self.pulprog
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty()),
        ) {
            (Some(src), Some(pp)) if !src.eq_ignore_ascii_case(pp) => format!("{src} ({pp})"),
            (Some(src), _) => src.to_string(),
            (None, Some(pp)) => pp.to_string(),
            (None, None) => "Loaded".to_string(),
        }
    }
}

/// Post-assignment confidence triage category in Step 3.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TriageBucket {
    #[default]
    HighConfidence,
    Ambiguous,
    Unassigned,
    Locked,
}

impl TriageBucket {
    pub const ALL: [Self; 4] = [
        Self::HighConfidence,
        Self::Ambiguous,
        Self::Unassigned,
        Self::Locked,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::HighConfidence => "High >= 0.80",
            Self::Ambiguous => "Ambiguous",
            Self::Unassigned => "Unassigned",
            Self::Locked => "Locked",
        }
    }

    pub const fn accent_color(self) -> egui::Color32 {
        match self {
            Self::HighConfidence => egui::Color32::from_rgb(76, 214, 148),
            Self::Ambiguous => egui::Color32::from_rgb(248, 178, 68),
            Self::Unassigned => egui::Color32::from_rgb(242, 108, 114),
            Self::Locked => egui::Color32::from_rgb(90, 198, 252),
        }
    }
}

/// Post-run residue triage summary broken into the four clickable confidence buckets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AssignTriageSummary {
    /// 0-based residue indices assigned with posterior confidence `>= 0.80`.
    pub high_conf_residues: Vec<usize>,
    /// 0-based residue indices assigned with posterior confidence `< 0.80`.
    pub ambiguous_residues: Vec<usize>,
    /// 0-based residue indices currently unassigned (excluding prolines if caller filtered).
    pub unassigned_residues: Vec<usize>,
    /// 0-based residue indices hard-locked by the user.
    pub locked_residues: Vec<usize>,
    /// Optional human-readable residue labels indexed by 0-based residue slot (e.g. `"M1"`, `"Q2"`).
    pub residue_labels: Vec<String>,
}

impl AssignTriageSummary {
    /// Return the slice of 0-based residue indices belonging to `bucket`.
    pub fn residues_for(&self, bucket: TriageBucket) -> &[usize] {
        match bucket {
            TriageBucket::HighConfidence => &self.high_conf_residues,
            TriageBucket::Ambiguous => &self.ambiguous_residues,
            TriageBucket::Unassigned => &self.unassigned_residues,
            TriageBucket::Locked => &self.locked_residues,
        }
    }

    /// True if at least one triage bucket is populated.
    pub fn has_any(&self) -> bool {
        !self.high_conf_residues.is_empty()
            || !self.ambiguous_residues.is_empty()
            || !self.unassigned_residues.is_empty()
            || !self.locked_residues.is_empty()
    }

    /// Format a compact label for residue index `res_idx`.
    pub fn label_for(&self, res_idx: usize) -> String {
        self.residue_labels
            .get(res_idx)
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("R{}", res_idx + 1))
    }
}

/// Read-only workbench state passed into [`AssignSetupDrawerState`] on every frame.
#[derive(Clone, Debug, Default)]
pub struct AssignSetupInput {
    /// Number of residues in the currently loaded protein sequence (`0` = no sequence loaded).
    pub sequence_len: usize,
    /// Optional short preview of the loaded sequence (e.g. `"MQIFVKTLTG..."`).
    pub sequence_preview: String,
    /// Status checklist rows for backbone experiments (`HSQC`, `HNCACB`, `CBCA(CO)NH`, `HNCA`,
    /// `HN(CO)CA`, `HNCO`, `HN(CA)CO`). When empty, all 7 canonical experiments are shown as missing.
    pub experiments: Vec<ExperimentChecklistRow>,
    /// Whether an AlphaFold / PDB 3D structure is currently loaded for structure-informed priors.
    pub has_af_structure: bool,
    /// True while a background solver job is actively running.
    pub solver_running: bool,
    /// Post-run residue triage summary powering Step 3 clickable confidence pills.
    pub triage: AssignTriageSummary,
    /// Optional latest status message from picking or solving.
    pub last_status: String,
}

impl AssignSetupInput {
    /// Build a canonical 7-experiment checklist from `(loaded, peak_count)` pairs ordered as
    /// `[HSQC, HNCACB, CBCA(CO)NH, HNCA, HN(CO)CA, HNCO, HN(CA)CO]`.
    pub fn canonical_checklist(stats: [(bool, usize); 7]) -> Vec<ExperimentChecklistRow> {
        CANONICAL_BACKBONE_EXPERIMENTS
            .iter()
            .zip(stats)
            .map(|(&name, (loaded, peak_count))| {
                ExperimentChecklistRow::new(name, loaded, peak_count)
            })
            .collect()
    }

    fn effective_experiments(&self) -> Vec<ExperimentChecklistRow> {
        if self.experiments.is_empty() {
            Self::canonical_checklist([(false, 0); 7])
        } else {
            self.experiments.clone()
        }
    }
}

/// Actions emitted by [`AssignSetupDrawerState`] to the host workbench.
#[derive(Clone, Debug, PartialEq)]
pub enum AssignSetupAction {
    /// User clicked `Load FASTA/STAR` to open a sequence or NMR-STAR file.
    LoadSequenceFile,
    /// User clicked `Apply Sequence` with a validated clean 1-letter sequence string.
    ApplySequenceText(String),
    /// User clicked `Pick All (2D + 3D)` with the selected SNR threshold.
    PickAllSpectra { snr_threshold: f64 },
    /// User clicked `Auto` next to the `SNR:` slider to estimate threshold from spectrum stats.
    AutoEstimateSnr,
    /// User clicked `Run Assignment` with selected solver engine, regime, and AF-prior toggle.
    RunAssignment {
        engine: AssignSolverEngine,
        regime: AssignProteinRegime,
        use_af_priors: bool,
    },
    /// User clicked a post-run confidence pill or residue chip to jump to `0`-based residue index.
    JumpToResidue(usize),
}

/// Interactive 7-slot 3D data-input mapper actions emitted by Step 2 of [`AssignSetupDrawerState`],
/// drained by the host workbench via [`AssignSetupDrawerState::take_slot_actions`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssignSlotAction {
    /// User clicked `Load Study Folder...` to batch-load a multi-experiment study directory.
    LoadStudyFolder,
    /// User clicked `Load...` on a canonical backbone slot (e.g. `"HNCACB"`).
    LoadSlotSpectrum(String),
    /// User clicked `Clear` on a loaded canonical backbone slot (e.g. `"HNCO"`).
    ClearSlotSpectrum(String),
    /// User overrode the experiment-type assignment of a loaded slot via its ComboBox.
    ReassignSlotExperiment { from_slot: String, to_exp: String },
}

/// Parse an inline 1-letter amino acid sequence or FASTA block into a validated uppercase 1-letter
/// sequence string (`ACDEFGHIKLMNPQRSTVWY`). Rejects empty input or invalid letters fail-closed.
pub fn parse_inline_sequence(raw: &str) -> Result<String, String> {
    let mut seq = String::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('>') || trimmed.starts_with(';') {
            continue;
        }
        for ch in trimmed.chars() {
            if ch.is_ascii_whitespace() || ch.is_ascii_digit() || ch == '-' || ch == '*' {
                continue;
            }
            let up = ch.to_ascii_uppercase();
            if matches!(
                up,
                'A' | 'C'
                    | 'D'
                    | 'E'
                    | 'F'
                    | 'G'
                    | 'H'
                    | 'I'
                    | 'K'
                    | 'L'
                    | 'M'
                    | 'N'
                    | 'P'
                    | 'Q'
                    | 'R'
                    | 'S'
                    | 'T'
                    | 'V'
                    | 'W'
                    | 'Y'
            ) {
                seq.push(up);
            } else {
                return Err(format!(
                    "Unsupported residue code '{ch}' (expected standard 1-letter IUPAC amino acids)"
                ));
            }
        }
    }
    if seq.is_empty() {
        return Err("Enter a 1-letter sequence or FASTA block before applying.".to_string());
    }
    Ok(seq)
}

/// Stateful 3-step `Assignment Setup` collapsible drawer / floating window (`wasm32`-safe).
#[derive(Clone, Debug)]
pub struct AssignSetupDrawerState {
    /// Whether the collapsible drawer / window is currently expanded/open.
    pub open: bool,
    /// Inline 1-letter / FASTA sequence paste buffer (Step 1).
    pub sequence_paste_buf: String,
    /// Unified 2D+3D peak picking SNR threshold (Step 2).
    pub snr_threshold: f64,
    /// Selected solver engine (Step 3).
    pub engine: AssignSolverEngine,
    /// Selected protein/spectroscopy regime (Step 3).
    pub regime: AssignProteinRegime,
    /// Enable AlphaFold / structural chemical-shift priors during assignment (Step 3).
    pub use_af_priors: bool,
    /// Active post-run triage bucket selected by clicking a confidence pill.
    pub active_triage_bucket: Option<TriageBucket>,
    /// Cycling cursor inside `active_triage_bucket` when repeatedly clicking the same pill.
    pub triage_cursor: usize,
    /// Latest inline validation error (e.g. invalid FASTA character).
    pub validation_error: Option<String>,
    /// Bounding rect from the most recent rendered frame.
    pub last_drawer_rect: Option<egui::Rect>,
    /// Slot-level data-input mapper actions emitted during the most recent rendered frame.
    pending_slot_actions: Vec<AssignSlotAction>,
}

impl Default for AssignSetupDrawerState {
    fn default() -> Self {
        Self::new()
    }
}

impl AssignSetupDrawerState {
    pub fn new() -> Self {
        Self {
            open: true,
            sequence_paste_buf: String::new(),
            snr_threshold: 5.0,
            engine: AssignSolverEngine::EvoGraph,
            regime: AssignProteinRegime::Globular,
            use_af_priors: true,
            active_triage_bucket: None,
            triage_cursor: 0,
            validation_error: None,
            last_drawer_rect: None,
            pending_slot_actions: Vec::new(),
        }
    }

    /// Drain all Step 2 7-slot data-input mapper actions emitted during the last frame.
    pub fn take_slot_actions(&mut self) -> Vec<AssignSlotAction> {
        std::mem::take(&mut self.pending_slot_actions)
    }

    /// Render the drawer inside a collapsible floating `egui::Window` when `self.open` is true.
    pub fn show_window(
        &mut self,
        ctx: &egui::Context,
        input: &AssignSetupInput,
    ) -> Vec<AssignSetupAction> {
        let mut actions = Vec::new();
        if !self.open {
            self.last_drawer_rect = None;
            self.pending_slot_actions.clear();
            return actions;
        }
        let mut keep_open = self.open;
        let resp = egui::Window::new("Assignment Setup")
            .open(&mut keep_open)
            .collapsible(true)
            .resizable(true)
            .default_width(420.0)
            .min_width(340.0)
            .show(ctx, |ui| {
                actions = self.show_contents(ui, input);
            });
        self.open = keep_open;
        if let Some(ir) = resp {
            self.last_drawer_rect = Some(ir.response.rect);
        }
        actions
    }

    /// Render the 3-step `Assignment Setup` contents inside any host `egui::Ui` (side drawer,
    /// collapsible dock panel, or floating window).
    pub fn show_contents(
        &mut self,
        ui: &mut egui::Ui,
        input: &AssignSetupInput,
    ) -> Vec<AssignSetupAction> {
        let mut actions = Vec::new();
        self.pending_slot_actions.clear();

        let inner = ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Assignment Setup")
                        .size(TYPE_HEADER_PX)
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let toggle_lbl = if self.open { "Collapse" } else { "Expand" };
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new(toggle_lbl).size(TYPE_CAPTION_PX),
                        ))
                        .clicked()
                    {
                        self.open = !self.open;
                    }
                });
            });

            if !self.open {
                return;
            }

            ui.separator();

            egui::ScrollArea::vertical()
                .id_salt("assign_setup_drawer_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // ================================================================
                    // Step 1 — Sequence
                    // ================================================================
                    self.render_step1_sequence(ui, input, &mut actions);
                    ui.add_space(6.0);

                    // ================================================================
                    // Step 2 — Spectra & Peaks 7-Slot Data Input Mapper
                    // ================================================================
                    self.render_step2_spectra(ui, input, &mut actions);
                    ui.add_space(6.0);

                    // ================================================================
                    // Step 3 — Solve & Triage
                    // ================================================================
                    self.render_step3_solve_and_triage(ui, input, &mut actions);

                    if !input.last_status.is_empty() {
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(&input.last_status)
                                .size(TYPE_CAPTION_PX)
                                .color(egui::Color32::from_rgb(155, 195, 225)),
                        );
                    }
                });
        });

        self.last_drawer_rect = Some(inner.response.rect);
        actions
    }

    fn render_step1_sequence(
        &mut self,
        ui: &mut egui::Ui,
        input: &AssignSetupInput,
        actions: &mut Vec<AssignSetupAction>,
    ) {
        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_rgb(18, 23, 33))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(44, 56, 76)))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("Step 1 - Sequence")
                            .size(TYPE_BODY_PX)
                            .strong()
                            .color(egui::Color32::from_rgb(225, 235, 248)),
                    );
                    if input.sequence_len > 0 {
                        draw_status_pill(
                            ui,
                            &format!("{} aa loaded", input.sequence_len),
                            egui::Color32::from_rgb(22, 62, 46),
                            egui::Color32::from_rgb(92, 228, 158),
                        );
                    } else {
                        draw_status_pill(
                            ui,
                            "Warning: No sequence",
                            egui::Color32::from_rgb(76, 46, 20),
                            egui::Color32::from_rgb(252, 184, 82),
                        );
                    }
                });

                if !input.sequence_preview.is_empty() {
                    ui.label(
                        egui::RichText::new(&input.sequence_preview)
                            .monospace()
                            .size(TYPE_CAPTION_PX)
                            .color(egui::Color32::from_rgb(152, 174, 200)),
                    );
                }

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new("Load FASTA/STAR").size(TYPE_BODY_PX),
                        ))
                        .on_hover_text("Load protein sequence from FASTA or NMR-STAR v3 file")
                        .clicked()
                    {
                        actions.push(AssignSetupAction::LoadSequenceFile);
                    }
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new("Apply Sequence").size(TYPE_BODY_PX),
                        ))
                        .on_hover_text("Validate and apply the pasted 1-letter or FASTA sequence")
                        .clicked()
                    {
                        match parse_inline_sequence(&self.sequence_paste_buf) {
                            Ok(clean) => {
                                self.validation_error = None;
                                actions.push(AssignSetupAction::ApplySequenceText(clean));
                            }
                            Err(err) => {
                                self.validation_error = Some(err);
                            }
                        }
                    }
                });

                ui.add_space(4.0);
                ui.add(
                    egui::TextEdit::multiline(&mut self.sequence_paste_buf)
                        .id_salt("assign_setup_seq_paste")
                        .desired_rows(2)
                        .desired_width(f32::INFINITY)
                        .font(egui::FontId::monospace(TYPE_CAPTION_PX))
                        .hint_text("Paste 1-letter sequence or FASTA (>name\\nMQIFVK...)"),
                );

                if let Some(err) = &self.validation_error {
                    ui.label(
                        egui::RichText::new(err)
                            .size(TYPE_CAPTION_PX)
                            .color(egui::Color32::from_rgb(248, 122, 122)),
                    );
                }
            });
    }

    fn render_step2_spectra(
        &mut self,
        ui: &mut egui::Ui,
        input: &AssignSetupInput,
        actions: &mut Vec<AssignSetupAction>,
    ) {
        let rows = input.effective_experiments();
        let loaded_count = rows.iter().filter(|r| r.loaded).count();
        let total_peaks: usize = rows.iter().map(|r| r.peak_count).sum();

        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_rgb(18, 23, 33))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(44, 56, 76)))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new("Step 2 - Spectra & Peaks")
                            .size(TYPE_BODY_PX)
                            .strong()
                            .color(egui::Color32::from_rgb(225, 235, 248)),
                    );
                    draw_status_pill(
                        ui,
                        &format!("{loaded_count}/{} loaded | {total_peaks} peaks", rows.len()),
                        egui::Color32::from_rgb(24, 38, 56),
                        egui::Color32::from_rgb(132, 206, 250),
                    );
                });
                ui.add_space(4.0);

                // Top batch actions: Load Study Folder + Pick All (2D + 3D)
                ui.horizontal_wrapped(|ui| {
                    let study_btn = egui::Button::new(
                        egui::RichText::new("Load Study Folder...")
                            .size(TYPE_CAPTION_PX)
                            .strong(),
                    )
                    .fill(egui::Color32::from_rgb(34, 52, 78));
                    if ui
                        .add(study_btn)
                        .on_hover_text(
                            "Select a study directory to auto-discover and map Bruker, NMRPipe, or Sparky 2D/3D backbone spectra",
                        )
                        .clicked()
                    {
                        self.pending_slot_actions
                            .push(AssignSlotAction::LoadStudyFolder);
                    }

                    let pick_btn = egui::Button::new(
                        egui::RichText::new("Pick All (2D + 3D)")
                            .size(TYPE_CAPTION_PX)
                            .strong(),
                    )
                    .fill(egui::Color32::from_rgb(28, 72, 92));
                    if ui
                        .add(pick_btn)
                        .on_hover_text(
                            "Pick 2D HSQC roots and all loaded 3D triple-resonance cubes in one pass",
                        )
                        .clicked()
                    {
                        actions.push(AssignSetupAction::PickAllSpectra {
                            snr_threshold: self.snr_threshold,
                        });
                    }
                });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("SNR:").size(TYPE_CAPTION_PX));
                    ui.add(
                        egui::Slider::new(&mut self.snr_threshold, 2.0..=200.0)
                            .logarithmic(true)
                            .fixed_decimals(1)
                            .clamping(egui::SliderClamping::Always),
                    );
                    if ui
                        .add(egui::Button::new(
                            egui::RichText::new("Auto").size(TYPE_CAPTION_PX),
                        ))
                        .on_hover_text(
                            "Auto-estimate SNR threshold from spectrum MAD noise, dynamic range & contour floor",
                        )
                        .clicked()
                    {
                        actions.push(AssignSetupAction::AutoEstimateSnr);
                    }
                });
                ui.add_space(4.0);

                // Interactive 7-slot backbone spectrum data-input mapper
                for (row_idx, row) in rows.iter().enumerate() {
                    let row_bg = if row_idx % 2 == 0 {
                        egui::Color32::from_rgb(22, 28, 40)
                    } else {
                        egui::Color32::from_rgb(17, 22, 32)
                    };
                    egui::Frame::new()
                        .fill(row_bg)
                        .stroke(egui::Stroke::new(
                            0.8,
                            if row.loaded {
                                egui::Color32::from_rgb(44, 78, 72)
                            } else {
                                egui::Color32::from_rgb(34, 42, 56)
                            },
                        ))
                        .corner_radius(3.0)
                        .inner_margin(egui::Margin::symmetric(6, 3))
                        .show(ui, |ui| {
                            // Line 1: Canonical slot name, real picked peak count, per-slot Load/Clear
                            ui.horizontal(|ui| {
                                let slot_col = if row.loaded {
                                    egui::Color32::from_rgb(86, 224, 152)
                                } else {
                                    egui::Color32::from_rgb(156, 170, 190)
                                };
                                ui.label(
                                    egui::RichText::new(&row.exp_name)
                                        .monospace()
                                        .strong()
                                        .size(TYPE_CAPTION_PX)
                                        .color(slot_col),
                                );
                                let pk_col = if row.peak_count > 0 {
                                    egui::Color32::from_rgb(218, 232, 248)
                                } else {
                                    egui::Color32::from_rgb(126, 140, 160)
                                };
                                ui.label(
                                    egui::RichText::new(format!("{} pk", row.peak_count))
                                        .monospace()
                                        .size(TYPE_CAPTION_PX)
                                        .color(pk_col),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui
                                            .add_enabled(
                                                row.loaded,
                                                egui::Button::new(
                                                    egui::RichText::new("Clear")
                                                        .size(TYPE_CAPTION_PX),
                                                ),
                                            )
                                            .on_hover_text(format!(
                                                "Remove loaded spectrum from {} slot",
                                                row.exp_name
                                            ))
                                            .clicked()
                                        {
                                            self.pending_slot_actions.push(
                                                AssignSlotAction::ClearSlotSpectrum(
                                                    row.exp_name.clone(),
                                                ),
                                            );
                                        }
                                        if ui
                                            .add(egui::Button::new(
                                                egui::RichText::new("Load...")
                                                    .size(TYPE_CAPTION_PX),
                                            ))
                                            .on_hover_text(format!(
                                                "Load a spectrum or peak list into {} slot",
                                                row.exp_name
                                            ))
                                            .clicked()
                                        {
                                            self.pending_slot_actions.push(
                                                AssignSlotAction::LoadSlotSpectrum(
                                                    row.exp_name.clone(),
                                                ),
                                            );
                                        }
                                    },
                                );
                            });

                            // Line 2: Loaded source / pulprog + experiment-type override ComboBox
                            ui.horizontal(|ui| {
                                let full_src = row.display_source();
                                let short_src = truncate_slot_label(&full_src, 24);
                                let src_col = if row.loaded {
                                    egui::Color32::from_rgb(176, 206, 232)
                                } else {
                                    egui::Color32::from_rgb(122, 136, 156)
                                };
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(short_src)
                                            .monospace()
                                            .size(TYPE_CAPTION_PX)
                                            .color(src_col),
                                    )
                                    .wrap_mode(egui::TextWrapMode::Extend),
                                )
                                .on_hover_text(full_src);

                                if row.loaded {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            egui::ComboBox::from_id_salt(format!(
                                                "assign_slot_exp_override_{}",
                                                row.exp_name
                                            ))
                                            .width(98.0)
                                            .selected_text(
                                                egui::RichText::new(&row.exp_name)
                                                    .monospace()
                                                    .size(TYPE_CAPTION_PX),
                                            )
                                            .show_ui(ui, |ui| {
                                                for &cand in &CANONICAL_BACKBONE_EXPERIMENTS {
                                                    let is_cur =
                                                        cand.eq_ignore_ascii_case(&row.exp_name);
                                                    if ui
                                                        .selectable_label(
                                                            is_cur,
                                                            egui::RichText::new(cand)
                                                                .monospace()
                                                                .size(TYPE_CAPTION_PX),
                                                        )
                                                        .clicked()
                                                        && !is_cur
                                                    {
                                                        self.pending_slot_actions.push(
                                                            AssignSlotAction::ReassignSlotExperiment {
                                                                from_slot: row.exp_name.clone(),
                                                                to_exp: cand.to_string(),
                                                            },
                                                        );
                                                    }
                                                }
                                            });
                                        },
                                    );
                                }
                            });

                            // Optional Line 3: 3D-to-HSQC registration drift readout
                            if let Some((dh, dn)) = row.registration_drift_ppm {
                                if dh.is_finite() && dn.is_finite() {
                                    let warn_drift = dh.abs() > 0.03 || dn.abs() > 0.25;
                                    let drift_col = if warn_drift {
                                        egui::Color32::from_rgb(248, 182, 74)
                                    } else {
                                        egui::Color32::from_rgb(126, 214, 186)
                                    };
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(format!(
                                                    "Drift: ΔH {dh:+.3} ΔN {dn:+.2} ppm"
                                                ))
                                                .monospace()
                                                .size(TYPE_CAPTION_PX)
                                                .color(drift_col),
                                            )
                                            .wrap_mode(egui::TextWrapMode::Extend),
                                        )
                                        .on_hover_text(
                                            "3D-to-HSQC amide registration drift (ppm)",
                                        );
                                    });
                                }
                            }
                        });
                    ui.add_space(2.0);
                }
            });
    }

    fn render_step3_solve_and_triage(
        &mut self,
        ui: &mut egui::Ui,
        input: &AssignSetupInput,
        actions: &mut Vec<AssignSetupAction>,
    ) {
        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_rgb(18, 23, 33))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(44, 56, 76)))
            .corner_radius(4.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("Step 3 - Solve & Triage")
                        .size(TYPE_BODY_PX)
                        .strong()
                        .color(egui::Color32::from_rgb(225, 235, 248)),
                );
                ui.add_space(4.0);

                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Engine:").size(TYPE_CAPTION_PX));
                    for eng in AssignSolverEngine::ALL {
                        let sel = self.engine == eng;
                        if ui
                            .add(egui::Button::selectable(
                                sel,
                                egui::RichText::new(eng.label()).size(TYPE_CAPTION_PX),
                            ))
                            .clicked()
                        {
                            self.engine = eng;
                        }
                    }
                });

                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Regime:").size(TYPE_CAPTION_PX));
                    for reg in AssignProteinRegime::ALL {
                        let sel = self.regime == reg;
                        if ui
                            .add(egui::Button::selectable(
                                sel,
                                egui::RichText::new(reg.label()).size(TYPE_CAPTION_PX),
                            ))
                            .clicked()
                        {
                            self.regime = reg;
                        }
                    }
                });

                ui.horizontal_wrapped(|ui| {
                    if input.has_af_structure {
                        ui.checkbox(
                            &mut self.use_af_priors,
                            egui::RichText::new("Use AF Structure Priors (PDB loaded)")
                                .size(TYPE_CAPTION_PX),
                        )
                        .on_hover_text(
                            "Apply structure-informed secondary-structure priors from the loaded AlphaFold/PDB model",
                        );
                    } else {
                        let mut disabled_af = false;
                        ui.add_enabled(
                            false,
                            egui::Checkbox::new(
                                &mut disabled_af,
                                egui::RichText::new("Use AF Structure Priors (no PDB/AF loaded)")
                                    .size(TYPE_CAPTION_PX),
                            ),
                        )
                        .on_disabled_hover_text(
                            "Load an AlphaFold or PDB structure first to activate structure-informed priors.",
                        );
                    }
                });

                ui.add_space(6.0);
                let run_text = if input.solver_running {
                    "Running Assignment..."
                } else {
                    "Run Assignment"
                };
                let run_btn = egui::Button::new(
                    egui::RichText::new(run_text)
                        .size(TYPE_BODY_PX)
                        .strong()
                        .color(egui::Color32::from_rgb(240, 248, 255)),
                )
                .fill(egui::Color32::from_rgb(24, 112, 86))
                .min_size(egui::vec2(ui.available_width().max(140.0), 32.0));
                if ui
                    .add_enabled(!input.solver_running, run_btn)
                    .on_hover_text("Run backbone resonance assignment with current settings")
                    .clicked()
                {
                    actions.push(AssignSetupAction::RunAssignment {
                        engine: self.engine,
                        regime: self.regime,
                        use_af_priors: self.use_af_priors && input.has_af_structure,
                    });
                }

                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("Confidence Triage (click pill to jump):")
                        .size(TYPE_CAPTION_PX)
                        .color(egui::Color32::from_rgb(165, 185, 210)),
                );
                ui.horizontal_wrapped(|ui| {
                    for bucket in TriageBucket::ALL {
                        let list = input.triage.residues_for(bucket);
                        let count = list.len();
                        let selected = self.active_triage_bucket == Some(bucket);
                        let accent = bucket.accent_color();
                        let fill = if selected {
                            accent.gamma_multiply(0.32)
                        } else {
                            egui::Color32::from_rgb(26, 33, 46)
                        };
                        let stroke = egui::Stroke::new(
                            if selected { 1.4 } else { 1.0 },
                            if count > 0 {
                                accent
                            } else {
                                egui::Color32::from_rgb(56, 68, 86)
                            },
                        );
                        let text_col = if count > 0 {
                            accent
                        } else {
                            egui::Color32::from_rgb(122, 134, 152)
                        };
                        let pill = egui::Button::new(
                            egui::RichText::new(format!("{} ({count})", bucket.label()))
                                .size(TYPE_CAPTION_PX)
                                .color(text_col),
                        )
                        .fill(fill)
                        .stroke(stroke)
                        .corner_radius(10.0);
                        if ui.add(pill).clicked() && !list.is_empty() {
                            if self.active_triage_bucket == Some(bucket) {
                                self.triage_cursor = (self.triage_cursor + 1) % list.len();
                            } else {
                                self.active_triage_bucket = Some(bucket);
                                self.triage_cursor = 0;
                            }
                            if let Some(&res_idx) = list.get(self.triage_cursor) {
                                actions.push(AssignSetupAction::JumpToResidue(res_idx));
                            }
                        }
                    }
                });

                if let Some(bucket) = self.active_triage_bucket {
                    let list = input.triage.residues_for(bucket);
                    if !list.is_empty() {
                        ui.add_space(4.0);
                        ui.horizontal_wrapped(|ui| {
                            let max_chips = 16usize;
                            for (pos, &res_idx) in list.iter().take(max_chips).enumerate() {
                                let chip_lbl = input.triage.label_for(res_idx);
                                let is_cur = pos == self.triage_cursor;
                                if ui
                                    .add(egui::Button::selectable(
                                        is_cur,
                                        egui::RichText::new(chip_lbl)
                                            .monospace()
                                            .size(TYPE_CAPTION_PX),
                                    ))
                                    .clicked()
                                {
                                    self.triage_cursor = pos;
                                    actions.push(AssignSetupAction::JumpToResidue(res_idx));
                                }
                            }
                            if list.len() > max_chips {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "+{} more",
                                        list.len() - max_chips
                                    ))
                                    .size(TYPE_CAPTION_PX)
                                    .color(egui::Color32::from_rgb(138, 152, 174)),
                                );
                            }
                        });
                    }
                }
            });
    }
}

fn truncate_slot_label(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars || max_chars <= 3 {
        s.to_string()
    } else {
        let head: String = s.chars().take(max_chars - 3).collect();
        format!("{head}...")
    }
}

fn draw_status_pill(ui: &mut egui::Ui, text: &str, fill: egui::Color32, text_color: egui::Color32) {
    egui::Frame::new()
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, text_color.gamma_multiply(0.55)))
        .corner_radius(10.0)
        .inner_margin(egui::Margin::symmetric(7, 2))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(text)
                    .size(TYPE_CAPTION_PX)
                    .color(text_color),
            );
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workbench_mode::{apply_type_scale, audit_text_layout, MIN_FONT_PX};

    fn click_at(pos: egui::Pos2) -> Vec<egui::Event> {
        vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ]
    }

    #[test]
    fn test_assign_setup_drawer_auto_snr_button_and_wide_slider_range() {
        let vp = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 960.0));
        let ctx = egui::Context::default();
        let mut drawer = AssignSetupDrawerState::new();
        let input = AssignSetupInput {
            sequence_len: 76,
            sequence_preview: "MQIFVKTLTG...".to_string(),
            experiments: AssignSetupInput::canonical_checklist([(true, 72); 7]),
            ..Default::default()
        };

        let render = |drawer: &mut AssignSetupDrawerState,
                      events: Vec<egui::Event>|
         -> (egui::FullOutput, Vec<AssignSetupAction>) {
            let mut emitted = Vec::new();
            let mut out = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(vp),
                    events,
                    ..Default::default()
                },
                |ui| {
                    apply_type_scale(ui.ctx());
                    egui::Panel::left("assign_setup_auto_snr_test_host")
                        .default_size(380.0)
                        .show(ui, |ui| {
                            emitted = drawer.show_contents(ui, &input);
                        });
                    egui::CentralPanel::default().show(ui, |ui| {
                        ui.label("Main");
                    });
                },
            );
            out.textures_delta.clear();
            (out, emitted)
        };

        // Two warmup passes for layout stabilization + text audit
        let _ = render(&mut drawer, Vec::new());
        let (out, _) = render(&mut drawer, Vec::new());
        let audit = audit_text_layout(&out.shapes, vp);
        assert!(audit.min_font_px >= MIN_FONT_PX);
        assert!(
            audit.clipped.is_empty(),
            "clipped text with Auto SNR button: {:?}",
            audit.clipped
        );
        assert!(
            audit.overlapping.is_empty(),
            "overlapping text with Auto SNR button: {:?}",
            audit.overlapping
        );

        // Clicking `Auto` button emits AssignSetupAction::AutoEstimateSnr
        let auto_rect = audit
            .texts
            .iter()
            .find(|t| t.text == "Auto")
            .expect("missing Auto SNR button")
            .rect;
        let (_, acts_auto) = render(&mut drawer, click_at(auto_rect.center()));
        assert_eq!(acts_auto, vec![AssignSetupAction::AutoEstimateSnr]);

        // Cryo-probe SNR values above 20.0 up to 200.0 are preserved and emitted
        drawer.snr_threshold = 185.0;
        let (out_hi, _) = render(&mut drawer, Vec::new());
        assert!((drawer.snr_threshold - 185.0).abs() < 1e-6);
        let pick_rect = audit_text_layout(&out_hi.shapes, vp)
            .texts
            .iter()
            .find(|t| t.text.contains("Pick All (2D + 3D)"))
            .expect("missing Pick All button")
            .rect;
        let (_, acts_pick) = render(&mut drawer, click_at(pick_rect.center()));
        assert_eq!(
            acts_pick,
            vec![AssignSetupAction::PickAllSpectra {
                snr_threshold: 185.0
            }]
        );

        // Interactive slider drag across the rail reaches cryo-probe range > 20.0 up to 200.0
        drawer.snr_threshold = 5.0;
        let (out_sl, _) = render(&mut drawer, Vec::new());
        let audit_sl = audit_text_layout(&out_sl.shapes, vp);
        let snr_rect = audit_sl
            .texts
            .iter()
            .find(|t| t.text == "SNR:")
            .expect("missing SNR: label")
            .rect;
        let val_rect = audit_sl
            .texts
            .iter()
            .find(|t| t.text == "5.0")
            .expect("missing slider value readout")
            .rect;
        let start = egui::pos2(snr_rect.right() + 24.0, snr_rect.center().y);
        let end = egui::pos2(
            (val_rect.left() - 4.0).max(start.x + 40.0),
            snr_rect.center().y,
        );
        let _ = render(
            &mut drawer,
            vec![
                egui::Event::PointerMoved(start),
                egui::Event::PointerButton {
                    pos: start,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        let _ = render(
            &mut drawer,
            vec![
                egui::Event::PointerMoved(end),
                egui::Event::PointerButton {
                    pos: end,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        assert!(
            drawer.snr_threshold > 50.0 && drawer.snr_threshold <= 200.0,
            "dragging SNR slider to right end must reach cryo-probe range (50..=200), got {}",
            drawer.snr_threshold
        );
    }
}
