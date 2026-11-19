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

/// One row of the Step 2 spectra & picked peaks checklist.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExperimentChecklistRow {
    pub exp_name: String,
    pub loaded: bool,
    pub peak_count: usize,
}

impl ExperimentChecklistRow {
    pub fn new(exp_name: impl Into<String>, loaded: bool, peak_count: usize) -> Self {
        Self {
            exp_name: exp_name.into(),
            loaded,
            peak_count,
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
    /// User clicked `Run Assignment` with selected solver engine, regime, and AF-prior toggle.
    RunAssignment {
        engine: AssignSolverEngine,
        regime: AssignProteinRegime,
        use_af_priors: bool,
    },
    /// User clicked a post-run confidence pill or residue chip to jump to `0`-based residue index.
    JumpToResidue(usize),
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
        }
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
            return actions;
        }
        let mut keep_open = self.open;
        let resp = egui::Window::new("Assignment Setup")
            .open(&mut keep_open)
            .collapsible(true)
            .resizable(true)
            .default_width(360.0)
            .min_width(300.0)
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

            // ================================================================
            // Step 1 — Sequence
            // ================================================================
            self.render_step1_sequence(ui, input, &mut actions);
            ui.add_space(6.0);

            // ================================================================
            // Step 2 — Spectra & Peaks Checklist
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

                egui::Grid::new("assign_setup_spectra_checklist")
                    .num_columns(3)
                    .spacing([10.0, 3.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("Experiment")
                                .size(TYPE_CAPTION_PX)
                                .strong(),
                        );
                        ui.label(egui::RichText::new("Status").size(TYPE_CAPTION_PX).strong());
                        ui.label(egui::RichText::new("Peaks").size(TYPE_CAPTION_PX).strong());
                        ui.end_row();

                        for row in &rows {
                            ui.label(
                                egui::RichText::new(&row.exp_name)
                                    .monospace()
                                    .size(TYPE_CAPTION_PX),
                            );
                            if row.loaded {
                                ui.label(
                                    egui::RichText::new("Loaded")
                                        .size(TYPE_CAPTION_PX)
                                        .color(egui::Color32::from_rgb(86, 224, 152)),
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new("Not loaded")
                                        .size(TYPE_CAPTION_PX)
                                        .color(egui::Color32::from_rgb(126, 140, 160)),
                                );
                            }
                            let pk_col = if row.peak_count > 0 {
                                egui::Color32::from_rgb(218, 232, 248)
                            } else {
                                egui::Color32::from_rgb(126, 140, 160)
                            };
                            ui.label(
                                egui::RichText::new(format!("{}", row.peak_count))
                                    .monospace()
                                    .size(TYPE_CAPTION_PX)
                                    .color(pk_col),
                            );
                            ui.end_row();
                        }
                    });

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("SNR:").size(TYPE_CAPTION_PX));
                    ui.add(
                        egui::Slider::new(&mut self.snr_threshold, 2.0..=20.0)
                            .fixed_decimals(1)
                            .clamping(egui::SliderClamping::Always),
                    );
                });
                ui.add_space(4.0);
                let pick_btn = egui::Button::new(
                    egui::RichText::new("Pick All (2D + 3D)")
                        .size(TYPE_BODY_PX)
                        .strong(),
                )
                .fill(egui::Color32::from_rgb(28, 72, 92))
                .min_size(egui::vec2(ui.available_width().max(140.0), 28.0));
                if ui
                    .add(pick_btn)
                    .on_hover_text(
                        "Pick 2D HSQC roots and all loaded 3D triple-resonance strips in one pass",
                    )
                    .clicked()
                {
                    actions.push(AssignSetupAction::PickAllSpectra {
                        snr_threshold: self.snr_threshold,
                    });
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
                    let af_label = if input.has_af_structure {
                        "Use AF Structure Priors (PDB loaded)"
                    } else {
                        "Use AF Structure Priors"
                    };
                    ui.checkbox(
                        &mut self.use_af_priors,
                        egui::RichText::new(af_label).size(TYPE_CAPTION_PX),
                    );
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
                        use_af_priors: self.use_af_priors,
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
