//! Clean-room modular UX components for the `FERRIC` interactive WebGPU & `egui` workbench.
//!
//! Provenance: Original clean-room human-computer interaction and state-machine architecture
//! (`docs/specs/06-gui-workbench.md`, `docs/specs/12-workbench-ux.md`, `docs/specs/15-structure-viewer.md`),
//! IUPAC covalent sidechain graph visualization (Markley et al., *J. Mol. Biol.* 1998, 280,
//! 933–952; wwPDB Chemical Component Dictionary `CCD`, public domain; Ulrich et al., *Nucleic
//! Acids Res.* 2008, 36, D402–D408), 3D Carson–Bugg / Kabsch–Sander / Kabsch biomolecular
//! structure viewport, workflow navigation, fuzzy command-palette search, and keyboard
//! accelerators for native desktop and `wasm32-unknown-unknown` targets.

pub mod assign_setup_drawer;
pub mod brand_icon;
pub mod command_palette;
pub mod contour_painter;
pub mod demo_preset;
pub mod dosy_panel;
pub mod dynamics_panel;
pub mod ingest_panel;
pub mod integrative_af_panel;
pub mod interaction_grammar;
pub mod keymap;
pub mod label_layout;
pub mod mcp_bridge;
pub mod metabolomics_panel;
pub mod mol3d_overlay;
pub mod mol3d_viewer;
pub mod octet_canvas;
pub mod relax_panel;
pub mod session_io;
pub mod sidechain_widget;
pub mod ss_ribbon_widget;
pub mod stack1d_panel;
pub mod strip_tiler;
pub mod undo_journal;
pub mod workbench_mode;
pub mod workbook;
pub mod workflow_bar;
pub mod workflow_dag_panel;
