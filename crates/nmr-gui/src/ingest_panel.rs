//! Zero-friction spectrometer folder & archive drag-and-drop loader and editable
//! Experiment Selection Matrix (`T09.2`, desktop + `wasm32` WebGPU).
//!
//! Provenance: Clean-room original implementation of `docs/specs/12-workbench-ux.md` §1,
//! grounded in published multidimensional NMR pulse-sequence physics and open interchange specs:
//! - Kay, L. E., Ikura, M., Tschudin, R., & Bax, A. (*J. Magn. Reson.* 1990, 89, 496–514) —
//!   triple-resonance `HNCO`, `HNCA`, and `HN(CO)CA` transfer pathways.
//! - Grzesiek, S., & Bax, A. (*J. Am. Chem. Soc.* 1992, 114, 6291–6293) — `CBCA(CO)NH` and
//!   `HNCACB` sequential `Cα`/`Cβ` correlation windows.
//! - Bermel, W., Bertini, I., Felli, I. C., Lee, Y.-M., Luchinat, C., & Pierattelli, R.
//!   (*J. Am. Chem. Soc.* 2009, 131, 15339–15345) — protonless `13C`-detected `CON`, `CACO`,
//!   `CANCO`, and `CBCACO` experiments for intrinsically disordered and paramagnetic proteins.
//! - Tugarinov, V., Hwang, P. M., Ollerenshaw, J. E., & Kay, L. E. (*J. Am. Chem. Soc.* 2003,
//!   125, 10120–10121) — `1H-13C` methyl-TROSY `HMQC` spectroscopy.
//! - Gutmanas, A., et al. (*Nat. Struct. Mol. Biol.* 2015, 22, 433–434) — NMR Exchange Format
//!   (`NEF v1.1`) session and shift/peak interchange.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
#[cfg(target_arch = "wasm32")]
use std::pin::Pin;
use std::sync::Arc;

use eframe::egui;
use nmr_core::classify::{
    classify_nmr_entries, extract_archive_full_entries_bounded, extract_archive_virtual_entries,
    is_nmrpipe_magic, is_ucsf_magic, is_zip_magic, is_zip_path_lower, ClassifiedExperiment,
    DataState, ExperimentDimensionality, ProblemRegime, PulseSequenceClass, VirtualEntry,
    WorkspaceInventory,
};
#[cfg(not(target_arch = "wasm32"))]
use nmr_core::classify::{
    classify_nmr_workspace, is_ucsf_extension_lower, recompute_workspace_regime,
};
use nmr_core::nef::read_nef_str;
use nmr_core::nmr1d::{
    load_1d_from_virtual_entries, prepare_loaded_1d_for_studio, Experiment1DKind, Spectrum1D,
};
use nmr_core::nmrstar::read_star_str;
use nmr_core::priors::{one_to_three, three_to_one};
use nmr_core::session::read_session_nef_str;
use nmr_core::spectrum::AxisPpm;

#[cfg(target_arch = "wasm32")]
mod wasm_folder_bridge {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(inline_js = r#"
const STATE = {
  installed: false,
  batches: [],
  errors: [],
};
const MAX_FILE_BYTES = 64 * 1024 * 1024;
const MAX_TOTAL_BYTES = 256 * 1024 * 1024;

function wakeEguiCanvas() {
  if (typeof window === "undefined") return;
  const kick = () => {
    try {
      window.dispatchEvent(new Event("resize"));
      if (typeof document !== "undefined") {
        const c = document.getElementById("nmr_canvas");
        if (c && typeof PointerEvent !== "undefined") {
          c.dispatchEvent(new PointerEvent("pointermove", { bubbles: true, clientX: 8, clientY: 8 }));
        }
      }
    } catch (_) {}
  };
  kick();
  setTimeout(kick, 35);
  setTimeout(kick, 160);
}

function readFileEntry(fileEntry, relPath, acc) {
  return new Promise((resolve) => {
    fileEntry.file(
      async (file) => {
        try {
          if (file.size > MAX_FILE_BYTES) {
            STATE.errors.push(`Skipped '${relPath}' (${(file.size / 1048576).toFixed(1)} MiB exceeds 64 MiB per-file browser cap).`);
            resolve();
            return;
          }
          if (acc.totalBytes + file.size > MAX_TOTAL_BYTES) {
            acc.capped = true;
            resolve();
            return;
          }
          const buf = await file.arrayBuffer();
          acc.totalBytes += buf.byteLength;
          acc.files.push([relPath, new Uint8Array(buf)]);
        } catch (err) {
          STATE.errors.push(`Failed reading '${relPath}': ${err}`);
        }
        resolve();
      },
      (err) => {
        STATE.errors.push(`Browser file entry error on '${relPath}': ${err}`);
        resolve();
      }
    );
  });
}

function readAllDirEntries(reader) {
  return new Promise((resolve, reject) => {
    const all = [];
    function step() {
      reader.readEntries(
        (batch) => {
          if (!batch || batch.length === 0) {
            resolve(all);
          } else {
            for (let i = 0; i < batch.length; i++) all.push(batch[i]);
            step();
          }
        },
        (err) => reject(err)
      );
    }
    step();
  });
}

async function walkEntry(entry, prefix, acc) {
  if (!entry || acc.capped) return;
  const name = entry.name || "";
  if (name === ".git" || name === "__MACOSX" || name === ".DS_Store" || name.startsWith("._")) {
    return;
  }
  const rel = prefix ? `${prefix}/${name}` : name;
  if (entry.isFile) {
    await readFileEntry(entry, rel, acc);
  } else if (entry.isDirectory) {
    try {
      const reader = entry.createReader();
      const children = await readAllDirEntries(reader);
      for (let i = 0; i < children.length; i++) {
        await walkEntry(children[i], rel, acc);
      }
    } catch (err) {
      STATE.errors.push(`Failed reading directory '${rel}': ${err}`);
    }
  }
}

async function ingestFileList(fileList) {
  const acc = { files: [], totalBytes: 0, capped: false };
  for (let i = 0; i < fileList.length; i++) {
    const f = fileList[i];
    const rel = (f.webkitRelativePath && f.webkitRelativePath.length > 0) ? f.webkitRelativePath : f.name;
    if (!rel || rel.includes("__MACOSX") || rel.endsWith(".DS_Store")) continue;
    if (f.size > MAX_FILE_BYTES) {
      STATE.errors.push(`Skipped '${rel}' (> 64 MiB browser per-file limit).`);
      continue;
    }
    if (acc.totalBytes + f.size > MAX_TOTAL_BYTES) {
      acc.capped = true;
      break;
    }
    try {
      const buf = await f.arrayBuffer();
      acc.totalBytes += buf.byteLength;
      acc.files.push([rel, new Uint8Array(buf)]);
    } catch (err) {
      STATE.errors.push(`Failed reading '${rel}': ${err}`);
    }
  }
  if (acc.capped) {
    STATE.errors.push("Reached 256 MiB browser drop memory cap; some trailing files were omitted.");
  }
  if (acc.files.length > 0) {
    STATE.batches.push(acc.files);
  }
  wakeEguiCanvas();
}

export function js_install_folder_drop_listener() {
  if (STATE.installed || typeof window === "undefined") return;
  STATE.installed = true;

  window.addEventListener(
    "dragover",
    (ev) => {
      if (ev.dataTransfer) {
        ev.preventDefault();
      }
    },
    true
  );

  window.addEventListener(
    "drop",
    (ev) => {
      if (!ev.dataTransfer || !ev.dataTransfer.items) return;
      const items = ev.dataTransfer.items;
      const entries = [];
      let hasDir = false;
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        if (it.kind === "file" && typeof it.webkitGetAsEntry === "function") {
          const entry = it.webkitGetAsEntry();
          if (entry) {
            if (entry.isDirectory) hasDir = true;
            entries.push(entry);
          }
        }
      }
      if (!hasDir) return;

      // Intercept directory drop BEFORE winit's FileReader throws NotFoundError on a directory!
      ev.preventDefault();
      ev.stopPropagation();
      if (typeof ev.stopImmediatePropagation === "function") {
        ev.stopImmediatePropagation();
      }

      (async () => {
        const acc = { files: [], totalBytes: 0, capped: false };
        for (let i = 0; i < entries.length; i++) {
          await walkEntry(entries[i], "", acc);
        }
        if (acc.capped) {
          STATE.errors.push("Reached 256 MiB browser folder memory cap; some trailing files were omitted.");
        }
        if (acc.files.length > 0) {
          STATE.batches.push(acc.files);
        } else if (STATE.errors.length === 0) {
          STATE.errors.push("Dropped folder contained no readable NMR files.");
        }
        wakeEguiCanvas();
      })();
    },
    true
  );
}

export function js_open_folder_picker() {
  if (typeof document === "undefined") return;
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.setAttribute("webkitdirectory", "");
  input.setAttribute("directory", "");
  input.style.display = "none";
  input.addEventListener("change", async () => {
    if (input.files && input.files.length > 0) {
      await ingestFileList(input.files);
    }
    input.remove();
  });
  document.body.appendChild(input);
  input.click();
}

export function js_open_files_picker() {
  if (typeof document === "undefined") return;
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.style.display = "none";
  input.addEventListener("change", async () => {
    if (input.files && input.files.length > 0) {
      await ingestFileList(input.files);
    }
    input.remove();
  });
  document.body.appendChild(input);
  input.click();
}

export function js_has_pending_batch() {
  return STATE.batches.length > 0;
}

export function js_pop_batch_len() {
  if (STATE.batches.length === 0) return 0;
  STATE.active = STATE.batches.shift();
  return STATE.active.length;
}

export function js_active_item_path(idx) {
  return STATE.active && STATE.active[idx] ? STATE.active[idx][0] : "";
}

export function js_active_item_bytes(idx) {
  return STATE.active && STATE.active[idx] ? STATE.active[idx][1] : new Uint8Array(0);
}

export function js_clear_active_batch() {
  STATE.active = null;
}

export function js_pop_error() {
  return STATE.errors.length > 0 ? STATE.errors.shift() : "";
}
"#)]
    extern "C" {
        pub fn js_install_folder_drop_listener();
        pub fn js_open_folder_picker();
        pub fn js_open_files_picker();
        pub fn js_has_pending_batch() -> bool;
        pub fn js_pop_batch_len() -> usize;
        pub fn js_active_item_path(idx: usize) -> String;
        pub fn js_active_item_bytes(idx: usize) -> Vec<u8>;
        pub fn js_clear_active_batch();
        pub fn js_pop_error() -> String;
    }
}

/// Install the capture-phase browser directory drop listener (`wasm32` only; no-op on native).
pub fn install_browser_folder_drop_hook() {
    #[cfg(target_arch = "wasm32")]
    wasm_folder_bridge::js_install_folder_drop_listener();
}

/// Open a browser `<input webkitdirectory>` Bruker/spectrometer folder picker (`wasm32` only).
pub fn trigger_browser_folder_picker() {
    #[cfg(target_arch = "wasm32")]
    wasm_folder_bridge::js_open_folder_picker();
}

/// Open a browser `<input type="file" multiple>` multi-file picker (`wasm32` only).
pub fn trigger_browser_files_picker() {
    #[cfg(target_arch = "wasm32")]
    wasm_folder_bridge::js_open_files_picker();
}

/// Poll any completed browser folder/picker batch of [`DroppedPayload`] items (`wasm32` only).
pub fn poll_browser_dropped_payloads() -> Option<Vec<DroppedPayload>> {
    #[cfg(target_arch = "wasm32")]
    {
        if !wasm_folder_bridge::js_has_pending_batch() {
            return None;
        }
        let n = wasm_folder_bridge::js_pop_batch_len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let path = wasm_folder_bridge::js_active_item_path(i);
            let bytes = wasm_folder_bridge::js_active_item_bytes(i);
            out.push(DroppedPayload::new(path, bytes));
        }
        wasm_folder_bridge::js_clear_active_batch();
        Some(out)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        None
    }
}

/// Poll any asynchronous browser directory/file reader warning or error (`wasm32` only).
pub fn poll_browser_drop_error() -> Option<String> {
    #[cfg(target_arch = "wasm32")]
    {
        let s = wasm_folder_bridge::js_pop_error();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        None
    }
}

/// Sanitize a user-pasted filesystem path (macOS Finder/Terminal drag-paste quotes, `\ ` shell
/// escapes like `My\ Drive`, and leading `~/`) while preserving Windows `C:\...` path separators.
pub fn sanitize_user_path_input(raw: &str) -> String {
    let mut s = raw.trim();
    while s.len() >= 2
        && ((s.starts_with('\'') && s.ends_with('\'')) || (s.starts_with('"') && s.ends_with('"')))
    {
        s = s[1..s.len() - 1].trim();
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(&next) = chars.peek() {
                if matches!(
                    next,
                    ' ' | '(' | ')' | '[' | ']' | '\'' | '"' | '&' | '!' | '$' | '@' | ','
                ) {
                    out.push(next);
                    let _ = chars.next();
                    continue;
                }
            }
        }
        out.push(ch);
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(rest) = out.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            let mut full = PathBuf::from(home);
            full.push(rest);
            return full.display().to_string();
        }
    }
    out
}

/// Return true when an ingest error indicates that a bare Bruker file (`2rr`, `1r`, `fid`, `ser`, ...)
/// was picked without its sibling parameter files so the UI can surface a 1-click `[Open Folder]` CTA.
pub fn is_bare_bruker_file_error(msg: &str) -> bool {
    msg.contains("Bruker '")
        || msg.contains("requires its sibling")
        || msg.contains("Cannot read folder")
}

/// Ordered catalog of all [`PulseSequenceClass`] options exposed in the per-row override dropdown.
pub const ALL_PULSE_CLASSES: &[PulseSequenceClass] = &[
    PulseSequenceClass::N15Hsqc,
    PulseSequenceClass::Hncacb,
    PulseSequenceClass::Cbcaconh,
    PulseSequenceClass::Hnca,
    PulseSequenceClass::Hncoca,
    PulseSequenceClass::Hnco,
    PulseSequenceClass::Hncaco,
    PulseSequenceClass::C13Hsqc,
    PulseSequenceClass::C13Hmqc,
    PulseSequenceClass::Cosy2D,
    PulseSequenceClass::Tocsy2D,
    PulseSequenceClass::Noesy2D,
    PulseSequenceClass::Roesy2D,
    PulseSequenceClass::JRes2D,
    PulseSequenceClass::C13Hmbc,
    PulseSequenceClass::N15Hmbc,
    PulseSequenceClass::HsqcTocsy2D,
    PulseSequenceClass::Inadequate2D,
    PulseSequenceClass::Dosy2D,
    PulseSequenceClass::Hetero2D,
    PulseSequenceClass::Ccconh,
    PulseSequenceClass::Hccconh,
    PulseSequenceClass::HcchTocsy,
    PulseSequenceClass::N15Noesy,
    PulseSequenceClass::C13Noesy,
    PulseSequenceClass::Con2D,
    PulseSequenceClass::Caco2D,
    PulseSequenceClass::Canco3D,
    PulseSequenceClass::Cbcaco3D,
    PulseSequenceClass::RelaxR1,
    PulseSequenceClass::RelaxR2,
    PulseSequenceClass::HetNoe,
    PulseSequenceClass::Cpmg,
    PulseSequenceClass::Cest,
    PulseSequenceClass::TitrationSeries,
    PulseSequenceClass::Proton1D,
    PulseSequenceClass::Carbon13_1D,
    PulseSequenceClass::Dept1D,
    PulseSequenceClass::Fluorine19_1D,
    PulseSequenceClass::Phosphorus31_1D,
    PulseSequenceClass::StdWaterLogsy1D,
    PulseSequenceClass::Hetero1D,
    PulseSequenceClass::Unknown,
];

/// Human-readable spectroscopist label for a [`PulseSequenceClass`].
pub fn pulse_class_display_label(cls: PulseSequenceClass) -> &'static str {
    match cls {
        PulseSequenceClass::N15Hsqc => "2D 15N-HSQC / TROSY",
        PulseSequenceClass::C13Hsqc => "2D 13C-HSQC",
        PulseSequenceClass::C13Hmqc => "2D 13C-Methyl HMQC",
        PulseSequenceClass::Cosy2D => "2D 1H-1H COSY / DQF-COSY",
        PulseSequenceClass::Tocsy2D => "2D 1H-1H TOCSY / DIPSI",
        PulseSequenceClass::Noesy2D => "2D 1H-1H NOESY",
        PulseSequenceClass::Roesy2D => "2D 1H-1H ROESY / T-ROESY",
        PulseSequenceClass::JRes2D => "2D 1H J-Resolved",
        PulseSequenceClass::C13Hmbc => "2D 1H-13C HMBC / H2BC",
        PulseSequenceClass::N15Hmbc => "2D 1H-15N HMBC",
        PulseSequenceClass::HsqcTocsy2D => "2D 1H-13C HSQC-TOCSY",
        PulseSequenceClass::Inadequate2D => "2D 13C-13C INADEQUATE / ADEQUATE",
        PulseSequenceClass::Dosy2D => "Pseudo-2D DOSY Diffusion",
        PulseSequenceClass::Hetero2D => "2D Heteronuclear (19F/31P/X)",
        PulseSequenceClass::Hnco => "3D HNCO",
        PulseSequenceClass::Hncaco => "3D HN(CA)CO",
        PulseSequenceClass::Hnca => "3D HNCA",
        PulseSequenceClass::Hncoca => "3D HN(CO)CA",
        PulseSequenceClass::Hncacb => "3D HNCACB",
        PulseSequenceClass::Cbcaconh => "3D CBCA(CO)NH",
        PulseSequenceClass::Ccconh => "3D CC(CO)NH",
        PulseSequenceClass::Hccconh => "3D H(CCO)NH",
        PulseSequenceClass::HcchTocsy => "3D HCCH-TOCSY",
        PulseSequenceClass::N15Noesy => "3D 15N-NOESY",
        PulseSequenceClass::C13Noesy => "3D 13C-NOESY",
        PulseSequenceClass::Con2D => "2D CON (13C-det)",
        PulseSequenceClass::Caco2D => "2D CACO (13C-det)",
        PulseSequenceClass::Canco3D => "3D CANCO (13C-det)",
        PulseSequenceClass::Cbcaco3D => "3D CBCACO (13C-det)",
        PulseSequenceClass::RelaxR1 => "Pseudo-3D R1 (T1)",
        PulseSequenceClass::RelaxR2 => "Pseudo-3D R2 (T2)",
        PulseSequenceClass::HetNoe => "{1H}-15N hetNOE",
        PulseSequenceClass::Cpmg => "CPMG Dispersion",
        PulseSequenceClass::Cest => "CEST Exchange",
        PulseSequenceClass::TitrationSeries => "Titration CSP Series",
        PulseSequenceClass::Proton1D => "1D 1H",
        PulseSequenceClass::Carbon13_1D => "1D 13C{1H}",
        PulseSequenceClass::Dept1D => "1D DEPT-90/135/DEPTQ",
        PulseSequenceClass::Fluorine19_1D => "1D 19F",
        PulseSequenceClass::Phosphorus31_1D => "1D 31P",
        PulseSequenceClass::StdWaterLogsy1D => "1D STD / WaterLOGSY",
        PulseSequenceClass::Hetero1D => "1D Heteronuclear",
        PulseSequenceClass::Unknown => "Unknown / 1D Decoy",
    }
}

/// Human-readable label for a [`ProblemRegime`].
pub fn problem_regime_display_label(regime: ProblemRegime) -> &'static str {
    match regime {
        ProblemRegime::GlobularBackbone => "Globular 1H/15N/13C Backbone",
        ProblemRegime::SidechainNoesy => "Backbone + Sidechain TOCSY / 3D NOESY",
        ProblemRegime::Idp13cDetected => "IDP Protonless 13C-Detected Suite",
        ProblemRegime::MethylTrosy => "High-MW 13C-Methyl TROSY",
        ProblemRegime::Paramagnetic => "Paramagnetic PCS / PRE Suite",
        ProblemRegime::RelaxationDynamics => "Spin Relaxation & Exchange Dynamics",
        ProblemRegime::TitrationCsp => "Ligand Titration CSP Series",
        ProblemRegime::SmallMolecule2D => "2D Homonuclear & Small-Molecule Correlation Suite",
    }
}

/// Default selection rule: include 2D/3D/Pseudo3D/4D classified experiments while excluding
/// `1D` checks (`pulsecal`, `zgpr`) and `Unknown` decoys by default.
pub fn should_select_experiment_by_default(exp: &ClassifiedExperiment) -> bool {
    if exp.dimensionality == ExperimentDimensionality::Dim1D
        || exp.pulse_class == PulseSequenceClass::Unknown
    {
        return false;
    }
    exp.selected_by_default
}

/// Parse a sequence string (FASTA, raw 1-letter, or whitespace-separated 3-letter codes)
/// into canonical uppercase 3-letter amino-acid codes (`MET`, `GLN`, ...).
pub fn parse_sequence_three_letter(input: &str) -> Vec<String> {
    let mut body_lines = Vec::new();
    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('>') || trimmed.starts_with('#') {
            continue;
        }
        body_lines.push(trimmed);
    }
    if body_lines.is_empty() {
        return Vec::new();
    }
    let joined = body_lines.join(" ");
    let tokens: Vec<&str> = joined.split_whitespace().collect();
    let looks_three_letter = tokens.len() >= 2
        && tokens
            .iter()
            .all(|t| t.len() == 3 && t.chars().all(|c| c.is_ascii_alphabetic()));

    if looks_three_letter {
        tokens.into_iter().map(|t| t.to_ascii_uppercase()).collect()
    } else {
        let mut out = Vec::new();
        for ch in joined.chars() {
            if ch.is_ascii_alphabetic() {
                let tri = one_to_three(ch.to_ascii_uppercase());
                if tri != "UNK" || ch.eq_ignore_ascii_case(&'X') {
                    out.push(tri.to_string());
                }
            }
        }
        out
    }
}

/// Extract a 1-letter sequence string from a FASTA, NMR-STAR (`.str`), or NEF (`.nef`) text buffer.
pub fn extract_sequence_from_auxiliary_text(filename: &str, text: &str) -> Option<String> {
    let lower = filename.to_ascii_lowercase();
    if lower.ends_with(".str") || lower.ends_with(".star") || lower.ends_with(".bmrb") {
        if let Ok(r) = read_star_str(text, None) {
            if !r.sequence.is_empty() {
                let one: String = r.sequence.iter().map(|s| three_to_one(s)).collect();
                return Some(one);
            }
        }
    }
    if lower.ends_with(".nef") || text.contains("data_nef_") {
        if let Ok(sess) = read_session_nef_str(text) {
            if !sess.sequence.is_empty() {
                let one: String = sess.sequence.iter().map(|s| three_to_one(s)).collect();
                return Some(one);
            }
        }
        if let Ok(doc) = read_nef_str(text) {
            if let Ok(r) = doc.to_reference() {
                if !r.sequence.is_empty() {
                    let one: String = r.sequence.iter().map(|s| three_to_one(s)).collect();
                    return Some(one);
                }
            }
        }
    }
    if lower.ends_with(".fasta")
        || lower.ends_with(".fa")
        || lower.ends_with(".faa")
        || lower.ends_with(".seq")
    {
        let parsed = parse_sequence_three_letter(text);
        if !parsed.is_empty() {
            let one: String = parsed.iter().map(|s| three_to_one(s)).collect();
            return Some(one);
        }
    }
    None
}

/// Cross-platform in-memory implementation of [`egui::DroppedFile`] usable in unit/integration tests
/// and browser/clipboard byte-buffer pipelines on both native desktop and `wasm32`.
#[derive(Clone, Debug)]
pub struct InMemoryDroppedFile {
    path: PathBuf,
    bytes: Vec<u8>,
}

impl InMemoryDroppedFile {
    /// Construct an in-memory [`egui::DroppedFile`] with `name` and `bytes`.
    pub fn new(name: impl Into<PathBuf>, bytes: Vec<u8>) -> Self {
        Self {
            path: name.into(),
            bytes,
        }
    }

    /// Wrap as a shared [`egui::DroppedFileHandle`].
    pub fn into_handle(name: impl Into<PathBuf>, bytes: Vec<u8>) -> egui::DroppedFileHandle {
        Arc::new(Self::new(name, bytes))
    }
}

impl egui::DroppedFile for InMemoryDroppedFile {
    fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn bytes(&self) -> Result<Vec<u8>, String> {
        Ok(self.bytes.clone())
    }

    #[cfg(target_arch = "wasm32")]
    fn bytes_async(
        &self,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>, String>> + '_>> {
        let out = self.bytes.clone();
        Box::pin(async move { Ok(out) })
    }
}

/// Cross-platform `(name, bytes)` payload produced from browser or in-memory [`egui::DroppedFile`] handles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DroppedPayload {
    pub name: String,
    pub bytes: Vec<u8>,
}

impl DroppedPayload {
    pub fn new(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            bytes,
        }
    }
}

/// Resolved experiment selection row emitted when the user triggers **"Load Selected Spectra"**
/// or **"Process, Pick & Run"**.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedExperimentSpec {
    pub row_index: usize,
    pub experiment: ClassifiedExperiment,
    pub effective_class: PulseSequenceClass,
    pub solver_exp_name: Option<&'static str>,
    pub resolved_path: Option<PathBuf>,
}

/// Actionable execution plan built from the current [`IngestPanelState`].
#[derive(Clone, Debug, PartialEq)]
pub struct IngestExecutionPlan {
    pub root_name: String,
    pub root_path: Option<PathBuf>,
    pub problem_regime: ProblemRegime,
    pub sequence_three_letter: Vec<String>,
    pub selected_experiments: Vec<SelectedExperimentSpec>,
    /// Retained normalized `(rel_path, bytes)` map for in-memory / browser `wasm32` 1D/2D/3D loading.
    pub in_memory_files: Arc<BTreeMap<String, Vec<u8>>>,
}

/// User action emitted by [`IngestPanelState::show`].
#[derive(Clone, Debug, PartialEq)]
pub enum IngestAction {
    /// Re-scan a folder or `.zip` / `.tar` / `.tar.gz` path typed into the path bar (native desktop).
    ScanNativePath(String),
    /// Load selected spectra + sequence into the workbench without running automated solving.
    LoadSelectedSpectra(IngestExecutionPlan),
    /// One-click pipeline: load/process selected experiments, corroborated 2D/3D pick, QC, and solve.
    ProcessPickAndRun(IngestExecutionPlan),
    /// Open checked 1D rows directly inside `1D Array Studio` (`stack1d_panel`).
    OpenSelected1dInStudio(IngestExecutionPlan),
}

/// Result of inspecting one or more dropped files/folders/archives.
#[derive(Clone, Debug)]
pub enum IngestDropOutcome {
    /// A `.nef` workbench session file was dropped and should be restored directly.
    NefSession {
        source_name: String,
        nef_bytes: Vec<u8>,
    },
    /// A single standalone `.str` or `.tsv` text table was dropped.
    TextTable { source_name: String, text: String },
    /// One or more 1D NMR spectra (`Bruker fid/1r`, `.jdx`, `.json`, `.tsv`, or 1D folder/archive) ready for `1D Array Studio`.
    Spectra1D {
        source_name: String,
        spectra: Vec<Spectrum1D>,
    },
    /// A spectrometer folder, `.zip`/`.tar`/`.tar.gz` archive, or 2D/3D spectrum set was classified for the Ingest Matrix.
    Workspace(Box<IngestPanelState>),
}

/// Stateful controller for the **Editable Experiment Selection Matrix** (`T09.2`).
#[derive(Clone, Debug)]
pub struct IngestPanelState {
    pub open: bool,
    pub workspace_path_input: String,
    pub root_path: Option<PathBuf>,
    pub inventory: WorkspaceInventory,
    /// Per-row include/exclude checkbox (`1D` and `Unknown` decoys default to `false`).
    pub selected: Vec<bool>,
    /// Per-row effective [`PulseSequenceClass`] (initialized from auto-classifier, user-overridable).
    pub effective_classes: Vec<PulseSequenceClass>,
    /// True for rows where the user explicitly overrode [`PulseSequenceClass`].
    pub user_overridden: Vec<bool>,
    /// Optional native filesystem path per row when classified from disk.
    pub row_resolved_paths: Vec<Option<PathBuf>>,
    /// Retained normalized `(rel_path, bytes)` map for browser `wasm32` / archive 1D/2D/3D loading.
    pub in_memory_files: Arc<BTreeMap<String, Vec<u8>>>,
    /// Editable polymer sequence box (1-letter FASTA or 3-letter residue codes).
    pub sequence_text: String,
    /// Provenance note when `sequence_text` was auto-populated from `.fasta`/`.str`/`.nef`.
    pub sequence_source_label: Option<String>,
    /// Optional inline error message displayed inside the modal.
    pub status_error: Option<String>,
}

impl Default for IngestPanelState {
    fn default() -> Self {
        Self::from_inventory(WorkspaceInventory {
            root_name: "workspace".to_string(),
            experiments: Vec::new(),
            auxiliary_files: Vec::new(),
            problem_regime: ProblemRegime::GlobularBackbone,
            secondary_regimes: Vec::new(),
            missing_input_hints: Vec::new(),
            elapsed_ms: 0.0,
        })
    }
}

impl IngestPanelState {
    /// Initialize an editable Experiment Selection Matrix from a [`WorkspaceInventory`].
    pub fn from_inventory(inventory: WorkspaceInventory) -> Self {
        let n = inventory.experiments.len();
        let has_2d_plus = inventory
            .experiments
            .iter()
            .any(|e| e.dimensionality != ExperimentDimensionality::Dim1D);
        let selected: Vec<bool> = inventory
            .experiments
            .iter()
            .map(|e| {
                if has_2d_plus {
                    should_select_experiment_by_default(e)
                } else {
                    e.pulse_class != PulseSequenceClass::Unknown && e.selected_by_default
                }
            })
            .collect();
        let effective_classes: Vec<PulseSequenceClass> = inventory
            .experiments
            .iter()
            .map(|e| e.pulse_class)
            .collect();

        Self {
            open: true,
            workspace_path_input: String::new(),
            root_path: None,
            inventory,
            selected,
            effective_classes,
            user_overridden: vec![false; n],
            row_resolved_paths: vec![None; n],
            in_memory_files: Arc::new(BTreeMap::new()),
            sequence_text: String::new(),
            sequence_source_label: None,
            status_error: None,
        }
    }

    /// Classify a native folder, single spectrum, or `.tar`/`.tar.gz` archive on disk and populate
    /// per-row paths plus any discovered `.fasta` / `.str` / `.nef` polymer sequence.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_native_path(path: &Path) -> Result<Self, String> {
        let inv = classify_nmr_workspace(path)?;
        let mut state = Self::from_inventory(inv);
        state.workspace_path_input = path.display().to_string();
        state.root_path = Some(path.to_path_buf());

        let is_dir = path.is_dir();
        for (idx, exp) in state.inventory.experiments.iter().enumerate() {
            let candidate = if is_dir {
                if exp.relative_path == "." || exp.relative_path.is_empty() {
                    path.to_path_buf()
                } else {
                    path.join(&exp.relative_path)
                }
            } else if path.is_file()
                && !is_archive_name_lower(&path.to_string_lossy().to_ascii_lowercase())
            {
                path.to_path_buf()
            } else {
                PathBuf::new()
            };
            if !candidate.as_os_str().is_empty() && candidate.exists() {
                state.row_resolved_paths[idx] = Some(candidate);
            }
        }

        // Auto-populate sequence from discovered auxiliary files on disk (or inside `.tar`/`.tar.gz`)
        if is_dir {
            for aux in &state.inventory.auxiliary_files {
                let aux_path = path.join(&aux.relative_path);
                if let Ok(txt) = std::fs::read_to_string(&aux_path) {
                    if let Some(seq1) =
                        extract_sequence_from_auxiliary_text(&aux.relative_path, &txt)
                    {
                        state.sequence_source_label = Some(format!(
                            "{} ({} aa)",
                            aux.relative_path,
                            seq1.chars().count()
                        ));
                        state.sequence_text = seq1;
                        break;
                    }
                }
            }
        } else if path.is_file()
            && is_archive_name_lower(&path.to_string_lossy().to_ascii_lowercase())
        {
            if let Ok(bytes) = std::fs::read(path) {
                if let Ok(entries) =
                    extract_archive_virtual_entries(&state.inventory.root_name, &bytes)
                {
                    state.populate_sequence_from_virtual_entries(&entries);
                }
            }
        }

        Ok(state)
    }

    /// Populate `sequence_text` from header/table bytes in a `VirtualEntry` slice when empty.
    pub fn populate_sequence_from_virtual_entries(&mut self, entries: &[VirtualEntry]) {
        if !self.sequence_text.trim().is_empty() {
            return;
        }
        for entry in entries {
            if entry.header_bytes.is_empty() {
                continue;
            }
            if let Ok(txt) = std::str::from_utf8(&entry.header_bytes) {
                if let Some(seq1) = extract_sequence_from_auxiliary_text(&entry.relative_path, txt)
                {
                    self.sequence_source_label = Some(format!(
                        "{} ({} aa)",
                        entry.relative_path,
                        seq1.chars().count()
                    ));
                    self.sequence_text = seq1;
                    break;
                }
            }
        }
    }

    /// Toggle row selection in the matrix.
    pub fn set_row_selected(&mut self, row: usize, include: bool) {
        if let Some(slot) = self.selected.get_mut(row) {
            *slot = include;
        }
    }

    /// User override of `PulseSequenceClass` for `row`, persisted across selection changes and plan builds.
    pub fn override_pulse_class(&mut self, row: usize, new_class: PulseSequenceClass) {
        if row >= self.effective_classes.len() {
            return;
        }
        let prev_class = self.effective_classes[row];
        self.effective_classes[row] = new_class;
        self.user_overridden[row] = new_class != self.inventory.experiments[row].pulse_class;

        if new_class == PulseSequenceClass::Unknown {
            self.selected[row] = false;
        } else if prev_class == PulseSequenceClass::Unknown
            && self.inventory.experiments[row].dimensionality != ExperimentDimensionality::Dim1D
        {
            self.selected[row] = true;
        }
    }

    /// Select all 2D/3D backbone & sidechain experiments (excluding `1D` and `Unknown`).
    pub fn select_all_2d_3d(&mut self) {
        for (i, exp) in self.inventory.experiments.iter().enumerate() {
            let cls = self.effective_classes[i];
            self.selected[i] = exp.dimensionality != ExperimentDimensionality::Dim1D
                && cls != PulseSequenceClass::Unknown;
        }
    }

    /// Uncheck every experiment row.
    pub fn deselect_all(&mut self) {
        self.selected.fill(false);
    }

    /// Count of currently checked rows.
    pub fn selected_count(&self) -> usize {
        self.selected.iter().filter(|&&b| b).count()
    }

    /// Parse the inline sequence box into uppercase 3-letter residue codes.
    pub fn parsed_sequence_three_letter(&self) -> Vec<String> {
        parse_sequence_three_letter(&self.sequence_text)
    }

    /// True when the active workspace or selected experiments represent small-molecule / homonuclear
    /// 2D correlations, DOSY diffusion, or 1D spectra that do not require a polymer sequence.
    pub fn is_sequence_free_regime(&self) -> bool {
        if self.inventory.problem_regime == ProblemRegime::SmallMolecule2D {
            return true;
        }
        let any_protein_backbone = self.effective_classes.iter().any(|c| {
            c.is_1h_backbone_triple() || c.is_13c_detected() || *c == PulseSequenceClass::N15Hsqc
        });
        if any_protein_backbone {
            return false;
        }
        let sel_classes: Vec<PulseSequenceClass> = self
            .effective_classes
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.selected.get(i).copied().unwrap_or(false))
            .map(|(_, &c)| c)
            .collect();
        !sel_classes.is_empty()
            && sel_classes
                .iter()
                .all(|c| c.is_small_molecule_2d() || c.is_1d_experiment())
    }

    /// Compute live missing-input hints, clearing the sequence hint once `sequence_text` is populated
    /// or when in `SmallMolecule2D` / `Dosy2D` regime, and never penalizing multi-magnet acquisitions.
    pub fn dynamic_missing_hints(&self) -> Vec<String> {
        let has_seq = !self.parsed_sequence_three_letter().is_empty();
        let seq_free = self.is_sequence_free_regime();
        let mut hints: Vec<String> = self
            .inventory
            .missing_input_hints
            .iter()
            .filter(|h| {
                let low = h.to_ascii_lowercase();
                if (has_seq || seq_free) && (low.contains("sequence") || low.contains("fasta")) {
                    return false;
                }
                true
            })
            .cloned()
            .collect();

        if !has_seq
            && !seq_free
            && !hints
                .iter()
                .any(|h| h.to_ascii_lowercase().contains("sequence"))
        {
            hints.push(
                "No polymer sequence loaded yet — paste FASTA/1-letter sequence below before solving."
                    .to_string(),
            );
        }
        hints
    }

    /// Collect all checked rows with their effective (possibly user-overridden) [`PulseSequenceClass`].
    pub fn selected_experiments(&self) -> Vec<SelectedExperimentSpec> {
        self.inventory
            .experiments
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.selected.get(i).copied().unwrap_or(false))
            .map(|(i, exp)| {
                let cls = self.effective_classes[i];
                SelectedExperimentSpec {
                    row_index: i,
                    experiment: exp.clone(),
                    effective_class: cls,
                    solver_exp_name: cls.canonical_solver_exp(),
                    resolved_path: self.row_resolved_paths.get(i).and_then(|p| p.clone()),
                }
            })
            .collect()
    }

    /// Build a concrete [`IngestExecutionPlan`] from the current matrix state.
    pub fn build_execution_plan(&self) -> IngestExecutionPlan {
        IngestExecutionPlan {
            root_name: self.inventory.root_name.clone(),
            root_path: self.root_path.clone(),
            problem_regime: self.inventory.problem_regime,
            sequence_three_letter: self.parsed_sequence_three_letter(),
            selected_experiments: self.selected_experiments(),
            in_memory_files: Arc::clone(&self.in_memory_files),
        }
    }
}

fn is_archive_name_lower(lower: &str) -> bool {
    lower.ends_with(".tar")
        || lower.ends_with(".tar.gz")
        || lower.ends_with(".tgz")
        || is_zip_path_lower(lower)
}

/// Check whether a filename or leading magic bytes represent a `.zip` / `.tar` / `.tar.gz` / `.tgz` archive.
pub fn is_archive_filename_or_magic(name: &str, bytes: &[u8]) -> bool {
    let lower = name.to_ascii_lowercase();
    if is_archive_name_lower(&lower) || is_zip_magic(bytes) {
        return true;
    }
    if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        return true;
    }
    bytes.len() > 262 && &bytes[257..262] == b"ustar"
}

fn looks_like_nef_bytes(name: &str, bytes: &[u8]) -> bool {
    if name.trim().to_ascii_lowercase().ends_with(".nef") {
        return true;
    }
    let head = &bytes[..bytes.len().min(4096)];
    if let Ok(txt) = std::str::from_utf8(head) {
        let trimmed = txt.trim_start();
        trimmed.starts_with("data_nef_") || txt.contains("_nmrassign_workbench")
    } else {
        false
    }
}

/// Decode a 1D UCSF Sparky (`ndim == 1`, `.ucsf`/`.usf`/`.sp2`) or 1D NMRPipe (`FDDIMCOUNT == 1`, `.ft1`)
/// processed byte buffer directly into a [`Spectrum1D`] for `1D Array Studio`.
pub fn try_load_1d_ucsf_or_pipe_bytes(name: &str, bytes: &[u8]) -> Option<Spectrum1D> {
    let short_name = Path::new(name)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(name)
        .to_string();

    // 1. 1D UCSF Sparky v2 (`180`-byte global header + `128`-byte axis header + big-endian `f32` tiles)
    if is_ucsf_magic(bytes) && bytes.len() >= 308 && bytes[10] == 1 {
        let ax_hdr = &bytes[180..308];
        let nuc_end = ax_hdr[..6].iter().position(|&b| b == 0).unwrap_or(6);
        let raw_lbl = String::from_utf8_lossy(&ax_hdr[..nuc_end])
            .trim()
            .to_string();
        let label = if raw_lbl.is_empty() {
            "1H".to_string()
        } else {
            raw_lbl
        };
        let npts_i32 = i32::from_be_bytes([ax_hdr[8], ax_hdr[9], ax_hdr[10], ax_hdr[11]]);
        let tile_i32 = i32::from_be_bytes([ax_hdr[16], ax_hdr[17], ax_hdr[18], ax_hdr[19]]);
        if npts_i32 <= 0 || tile_i32 <= 0 {
            return None;
        }
        let npoints = npts_i32 as usize;
        let tile_size = tile_i32 as usize;
        let obs_mhz = f32::from_be_bytes([ax_hdr[20], ax_hdr[21], ax_hdr[22], ax_hdr[23]]) as f64;
        let sw_hz = f32::from_be_bytes([ax_hdr[24], ax_hdr[25], ax_hdr[26], ax_hdr[27]]) as f64;
        let xmtr_ppm = f32::from_be_bytes([ax_hdr[28], ax_hdr[29], ax_hdr[30], ax_hdr[31]]) as f64;
        if !obs_mhz.is_finite()
            || obs_mhz <= 0.0
            || !sw_hz.is_finite()
            || sw_hz <= 0.0
            || !xmtr_ppm.is_finite()
        {
            return None;
        }
        let nt0 = npoints.div_ceil(tile_size);
        let needed_bytes = 308usize.checked_add(nt0.checked_mul(tile_size)?.checked_mul(4)?)?;
        if bytes.len() < needed_bytes {
            return None;
        }
        let mut real = Vec::with_capacity(npoints);
        let body = &bytes[308..];
        for bt in 0..nt0 {
            let base = bt * tile_size;
            let valid = (npoints - base).min(tile_size);
            let off = bt * tile_size * 4;
            for chunk in body[off..off + valid * 4].chunks_exact(4) {
                let v = f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
                real.push(if v.is_finite() { v } else { 0.0 });
            }
        }
        let sw_ppm = sw_hz / obs_mhz;
        let axis_x = AxisPpm {
            label: label.clone(),
            size: npoints,
            first_ppm: xmtr_ppm + 0.5 * sw_ppm,
            step_ppm: -sw_ppm / (npoints as f64),
            obs_mhz,
        };
        let exp = Experiment1DKind::from_pulse_program(&short_name, &label);
        return Some(Spectrum1D::new(short_name, exp, axis_x, real, None));
    }

    // 2. 1D NMRPipe (`.ft1`, `2048`-byte header + `f32` body)
    if is_nmrpipe_magic(bytes) && bytes.len() >= 2048 {
        let le_magic = f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let big_endian = (le_magic - 2.345).abs() >= 1e-2;
        let read_hdr_f32 = |idx: usize| -> f32 {
            let o = idx * 4;
            let arr = [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]];
            if big_endian {
                f32::from_be_bytes(arr)
            } else {
                f32::from_le_bytes(arr)
            }
        };
        let ndim = read_hdr_f32(9).round() as isize;
        let ny = read_hdr_f32(219).round() as isize;
        if ndim != 1 && !(ndim <= 2 && ny <= 1) {
            return None;
        }
        let nx = (read_hdr_f32(99).round() as isize).max(1) as usize;
        let code0 = read_hdr_f32(24).round() as isize;
        let slot = match code0 {
            2 => 0usize,
            1 => 1usize,
            3 => 2usize,
            4 => 3usize,
            _ => 0usize,
        };
        const SW_IDX: [usize; 4] = [100, 229, 11, 29];
        const ORIG_IDX: [usize; 4] = [101, 249, 12, 30];
        const OBS_IDX: [usize; 4] = [119, 218, 10, 28];
        const LBL_IDX: [usize; 4] = [16, 18, 20, 22];

        let sw_hz = (read_hdr_f32(SW_IDX[slot]) as f64).max(1e-3);
        let orig_hz = read_hdr_f32(ORIG_IDX[slot]) as f64;
        let obs_mhz = (read_hdr_f32(OBS_IDX[slot]) as f64).max(1e-3);
        let lbl_off = LBL_IDX[slot] * 4;
        let lbl_slice = &bytes[lbl_off..lbl_off + 8];
        let lbl_end = lbl_slice.iter().position(|&b| b == 0).unwrap_or(8);
        let raw_lbl = String::from_utf8_lossy(&lbl_slice[..lbl_end])
            .trim()
            .to_string();
        let label = if raw_lbl.is_empty() {
            "1H".to_string()
        } else {
            raw_lbl
        };

        let is_complex = (read_hdr_f32(56).round() as i32) == 0
            && bytes.len() >= 2048usize.saturating_add(nx.saturating_mul(8));
        let needed_floats = if is_complex { nx * 2 } else { nx };
        if bytes.len() < 2048usize.saturating_add(needed_floats.saturating_mul(4)) {
            return None;
        }
        let decode_vec = |raw: &[u8], count: usize| -> Vec<f64> {
            let mut out = Vec::with_capacity(count);
            for chunk in raw[..count * 4].chunks_exact(4) {
                let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                let v = if big_endian {
                    f32::from_be_bytes(arr) as f64
                } else {
                    f32::from_le_bytes(arr) as f64
                };
                out.push(if v.is_finite() { v } else { 0.0 });
            }
            out
        };
        let body = &bytes[2048..];
        let real = decode_vec(body, nx);
        let imag = is_complex.then(|| decode_vec(&body[nx * 4..], nx));

        let n_f = nx as f64;
        let first_hz = orig_hz + sw_hz * ((n_f - 1.0) / n_f);
        let axis_x = AxisPpm {
            label: label.clone(),
            size: nx,
            first_ppm: first_hz / obs_mhz,
            step_ppm: -sw_hz / (obs_mhz * n_f),
            obs_mhz,
        };
        let exp = Experiment1DKind::from_pulse_program(&short_name, &label);
        return Some(Spectrum1D::new(short_name, exp, axis_x, real, imag));
    }

    None
}
/// - Single `.nef` drop -> [`IngestDropOutcome::NefSession`]
/// - Single `.str` / `.tsv` text table -> [`IngestDropOutcome::TextTable`]
/// - `.tar` / `.tar.gz` / `.tgz` archive or multi-file spectrometer bundle ->
///   `extract_archive_virtual_entries` / `VirtualEntry` -> `classify_nmr_entries` -> [`IngestPanelState`].
fn normalize_mem_rel(raw: &str) -> String {
    let norm = raw
        .replace('\\', "/")
        .trim_start_matches("./")
        .trim_start_matches('/')
        .trim_end_matches('/')
        .to_string();
    // Strip common single top-level wrapper prefix only when looking up relative paths later; keep canonical slash form here.
    norm
}

pub fn ingest_dropped_payloads(payloads: &[DroppedPayload]) -> Result<IngestDropOutcome, String> {
    if payloads.is_empty() {
        return Err("No dropped files provided".to_string());
    }

    if payloads.len() == 1 {
        let p = &payloads[0];
        let lower = p.name.trim().to_ascii_lowercase();
        if looks_like_nef_bytes(&p.name, &p.bytes) {
            return Ok(IngestDropOutcome::NefSession {
                source_name: p.name.clone(),
                nef_bytes: p.bytes.clone(),
            });
        }
        if !is_archive_filename_or_magic(&p.name, &p.bytes) {
            if lower.ends_with(".jdx") || lower.ends_with(".dx") || lower.ends_with(".jcamp") {
                let loaded =
                    load_1d_from_virtual_entries(&[(&p.name, p.bytes.as_slice())], None, false)?;
                let mut spectra = Vec::with_capacity(loaded.len());
                for item in loaded {
                    spectra.push(prepare_loaded_1d_for_studio(item)?);
                }
                return Ok(IngestDropOutcome::Spectra1D {
                    source_name: p.name.clone(),
                    spectra,
                });
            }
            if lower.ends_with(".json") {
                if let Ok(loaded) =
                    load_1d_from_virtual_entries(&[(&p.name, p.bytes.as_slice())], None, false)
                {
                    let mut spectra = Vec::with_capacity(loaded.len());
                    for item in loaded {
                        spectra.push(prepare_loaded_1d_for_studio(item)?);
                    }
                    return Ok(IngestDropOutcome::Spectra1D {
                        source_name: p.name.clone(),
                        spectra,
                    });
                }
            }
            if lower.ends_with(".str") || lower.ends_with(".star") || lower.ends_with(".tsv") {
                if let Ok(text) = std::str::from_utf8(&p.bytes) {
                    return Ok(IngestDropOutcome::TextTable {
                        source_name: p.name.clone(),
                        text: text.to_string(),
                    });
                }
            }
        }
    }

    const MAX_BROWSER_RETAINED_BYTES: usize = 256 * 1024 * 1024;
    let mut virtual_entries: Vec<VirtualEntry> = Vec::new();
    let mut mem_map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut mem_total = 0usize;

    for p in payloads {
        if is_archive_filename_or_magic(&p.name, &p.bytes) {
            let extracted = extract_archive_virtual_entries(&p.name, &p.bytes)?;
            virtual_entries.extend(extracted);
            if let Ok(full_items) = extract_archive_full_entries_bounded(
                &p.name,
                &p.bytes,
                512 * 1024 * 1024,
                MAX_BROWSER_RETAINED_BYTES.saturating_sub(mem_total),
            ) {
                for (rel, b) in full_items {
                    if mem_total + b.len() <= MAX_BROWSER_RETAINED_BYTES {
                        mem_total += b.len();
                        mem_map.insert(normalize_mem_rel(&rel), b);
                    }
                }
            }
        } else {
            let rel = normalize_mem_rel(&p.name);
            let cap = p.bytes.len().min(256 * 1024);
            virtual_entries.push(VirtualEntry::header_only(
                rel.clone(),
                p.bytes[..cap].to_vec(),
                p.bytes.len() as u64,
            ));
            if mem_total + p.bytes.len() <= MAX_BROWSER_RETAINED_BYTES {
                mem_total += p.bytes.len();
                mem_map.insert(rel, p.bytes.clone());
            }
        }
    }

    let mut inv = classify_nmr_entries(&virtual_entries);
    if payloads.len() == 1 {
        let fname = Path::new(&payloads[0].name)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(&payloads[0].name);
        if !fname.is_empty() {
            inv.root_name = fname.to_string();
        }
    } else if let Some(first) = mem_map.keys().next() {
        if let Some((top, _)) = first.split_once('/') {
            if !top.is_empty() && mem_map.keys().all(|k| k.starts_with(&format!("{top}/"))) {
                inv.root_name = top.to_string();
            }
        }
    }

    let has_2d_plus = inv
        .experiments
        .iter()
        .any(|e| !matches!(e.dimensionality, ExperimentDimensionality::Dim1D));

    // When the drop contains NO 2D/3D/4D protein experiments, attempt direct 1D loading
    // (or surface actionable bare-file `1r`/`fid`/`2rr`/`ser` errors instead of an empty matrix!).
    if !has_2d_plus {
        let refs: Vec<(&str, &[u8])> = mem_map
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        match load_1d_from_virtual_entries(&refs, None, false) {
            Ok(loaded) if !loaded.is_empty() => {
                let mut spectra = Vec::with_capacity(loaded.len());
                for item in loaded {
                    spectra.push(prepare_loaded_1d_for_studio(item)?);
                }
                return Ok(IngestDropOutcome::Spectra1D {
                    source_name: inv.root_name.clone(),
                    spectra,
                });
            }
            Err(err) if inv.experiments.is_empty() && err.starts_with("Bruker '") => {
                return Err(err);
            }
            _ => {}
        }
        // Also support 1D `.ucsf`/`.usf`/`.sp2` (`ndim == 1`) and 1D `.ft1` byte payloads directly!
        let mut ucsf_pipe_1d = Vec::new();
        for (rel, bytes) in &mem_map {
            if let Some(sp) = try_load_1d_ucsf_or_pipe_bytes(rel, bytes) {
                ucsf_pipe_1d.push(sp);
            }
        }
        if !ucsf_pipe_1d.is_empty() {
            return Ok(IngestDropOutcome::Spectra1D {
                source_name: inv.root_name.clone(),
                spectra: ucsf_pipe_1d,
            });
        }
        if inv.experiments.is_empty() {
            for p in payloads {
                let bname = Path::new(&p.name)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(&p.name)
                    .to_ascii_lowercase();
                if matches!(
                    bname.as_str(),
                    "2rr" | "2ii" | "2ir" | "2ri" | "proc2s" | "proc2" | "clevels"
                ) {
                    return Err(format!(
                        "Bruker '{bname}' requires its sibling 'procs' and 'proc2s' parameter files in the same folder. Click 'Open Folder' and select the 'pdata/1' (or experiment '10003') directory, or select '2rr', 'procs', and 'proc2s' together."
                    ));
                }
                if matches!(bname.as_str(), "3rrr" | "4rrrr" | "proc3s" | "proc3") {
                    return Err(format!(
                        "Bruker '{bname}' requires its sibling 'procs', 'proc2s', and 'proc3s' parameter files. Click 'Open Folder' and select the 'pdata/1' directory."
                    ));
                }
                if matches!(bname.as_str(), "ser" | "acqu2s" | "acqu2" | "acqu3s") {
                    return Err(format!(
                        "Bruker '{bname}' requires its sibling 'acqus' and 'acqu2s' parameter files. Click 'Open Folder' and select the experiment directory (e.g. '3' or '10003')."
                    ));
                }
                if matches!(
                    bname.as_str(),
                    "1r" | "1i" | "procs" | "fid" | "acqus" | "acqu"
                ) {
                    return Err(format!(
                        "Bruker '{bname}' requires its sibling parameter/binary files in the same folder. Click 'Open Folder' and select the Bruker experiment or 'pdata/1' directory."
                    ));
                }
            }
        }
    }

    let mut panel = IngestPanelState::from_inventory(inv);
    panel.in_memory_files = Arc::new(mem_map);
    panel.populate_sequence_from_virtual_entries(&virtual_entries);
    Ok(IngestDropOutcome::Workspace(Box::new(panel)))
}

/// Native helper: inspect a disk path (directory, archive, single file, or bare Bruker `1r`/`fid`/`2rr`/`ser`)
/// and route 1D-only datasets directly to [`IngestDropOutcome::Spectra1D`] while preserving
/// [`IngestDropOutcome::Workspace`] for 2D/3D/4D multi-experiment workspaces.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_native_path_1d_or_workspace(path: &Path) -> Result<IngestDropOutcome, String> {
    let p_str = path.display().to_string();
    let lower = p_str.to_ascii_lowercase();
    let fname_low = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    // Bare Bruker sibling resolution on native desktop: dropping `1r`, `1i`, `fid`, `procs`, `acqus`, `2rr`, `3rrr`, or `ser`
    // automatically inspects `path.parent()` on disk so users never hit a missing-sibling error!
    if path.is_file() {
        if matches!(fname_low.as_str(), "1r" | "1i" | "procs") {
            if let Some(parent) = path.parent() {
                if !parent.join("2rr").exists() && !parent.join("3rrr").exists() {
                    let loaded = nmr_core::nmr1d::load_1d_input(parent, None, false)?;
                    let spec = prepare_loaded_1d_for_studio(loaded)?;
                    return Ok(IngestDropOutcome::Spectra1D {
                        source_name: parent.display().to_string(),
                        spectra: vec![spec],
                    });
                } else {
                    let exp_root = parent
                        .parent()
                        .filter(|p| {
                            p.file_name()
                                .and_then(|s| s.to_str())
                                .is_some_and(|n| n.eq_ignore_ascii_case("pdata"))
                        })
                        .and_then(|p| p.parent())
                        .unwrap_or(parent);
                    let panel = IngestPanelState::from_native_path(exp_root)?;
                    return Ok(IngestDropOutcome::Workspace(Box::new(panel)));
                }
            }
        }
        if matches!(
            fname_low.as_str(),
            "2rr" | "3rrr" | "4rrrr" | "proc2s" | "proc3s"
        ) {
            if let Some(parent) = path.parent() {
                let exp_root = parent
                    .parent()
                    .filter(|p| {
                        p.file_name()
                            .and_then(|s| s.to_str())
                            .is_some_and(|n| n.eq_ignore_ascii_case("pdata"))
                    })
                    .and_then(|p| p.parent())
                    .unwrap_or(parent);
                let panel = IngestPanelState::from_native_path(exp_root)?;
                return Ok(IngestDropOutcome::Workspace(Box::new(panel)));
            }
        }
        if matches!(fname_low.as_str(), "fid" | "acqus" | "acqu") {
            if let Some(parent) = path.parent() {
                if !parent.join("ser").exists()
                    && !parent.join("acqu2s").exists()
                    && !parent.join("acqu3s").exists()
                {
                    let loaded = nmr_core::nmr1d::load_1d_input(parent, None, false)?;
                    let spec = prepare_loaded_1d_for_studio(loaded)?;
                    return Ok(IngestDropOutcome::Spectra1D {
                        source_name: parent.display().to_string(),
                        spectra: vec![spec],
                    });
                } else {
                    let panel = IngestPanelState::from_native_path(parent)?;
                    return Ok(IngestDropOutcome::Workspace(Box::new(panel)));
                }
            }
        }
        if matches!(fname_low.as_str(), "ser" | "acqu2s" | "acqu3s") {
            if let Some(parent) = path.parent() {
                let panel = IngestPanelState::from_native_path(parent)?;
                return Ok(IngestDropOutcome::Workspace(Box::new(panel)));
            }
        }
        if is_ucsf_extension_lower(&lower) {
            if let Ok(bytes) = std::fs::read(path) {
                if let Some(spec) = try_load_1d_ucsf_or_pipe_bytes(&p_str, &bytes) {
                    return Ok(IngestDropOutcome::Spectra1D {
                        source_name: p_str,
                        spectra: vec![spec],
                    });
                }
            }
        }
        if lower.ends_with(".jdx")
            || lower.ends_with(".dx")
            || lower.ends_with(".jcamp")
            || lower.ends_with(".ft1")
        {
            let spec = if lower.ends_with(".ft1") {
                let ft1 = nmr_core::ops::read_ft1_1d(path)?;
                let exp =
                    nmr_core::nmr1d::Experiment1DKind::from_pulse_program("", &ft1.axis_x.label);
                let real: Vec<f64> = ft1.data.iter().map(|&v| v as f64).collect();
                let imag = ft1
                    .imag
                    .map(|im| im.into_iter().map(|v| v as f64).collect());
                Spectrum1D::new(ft1.name, exp, ft1.axis_x, real, imag)
            } else {
                nmr_core::nmr1d::load_processed_spectrum_1d(path, None)?
            };
            return Ok(IngestDropOutcome::Spectra1D {
                source_name: p_str,
                spectra: vec![spec],
            });
        }
        if lower.ends_with(".json") {
            if let Ok(loaded) = nmr_core::nmr1d::load_1d_input(path, None, false) {
                let spec = prepare_loaded_1d_for_studio(loaded)?;
                return Ok(IngestDropOutcome::Spectra1D {
                    source_name: p_str,
                    spectra: vec![spec],
                });
            }
        }
        if is_archive_name_lower(&lower) {
            let bytes = std::fs::read(path)
                .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
            return ingest_dropped_payloads(&[DroppedPayload::new(p_str, bytes)]);
        }
    }

    let panel = IngestPanelState::from_native_path(path)?;
    let has_2d_plus = panel
        .inventory
        .experiments
        .iter()
        .any(|e| !matches!(e.dimensionality, ExperimentDimensionality::Dim1D));

    if !has_2d_plus {
        let mut spectra = Vec::new();
        // 1. Direct directory 1D check (e.g. `pdata/1` or single `expno` folder)
        if path.is_dir() {
            if let Ok(loaded) = nmr_core::nmr1d::load_1d_input(path, None, false) {
                if let Ok(spec) = prepare_loaded_1d_for_studio(loaded) {
                    spectra.push(spec);
                }
            }
            if spectra.is_empty() {
                for (idx, exp) in panel.inventory.experiments.iter().enumerate() {
                    if matches!(exp.dimensionality, ExperimentDimensionality::Dim1D) {
                        if let Some(Some(ref rp)) = panel.row_resolved_paths.get(idx) {
                            if let Ok(loaded) = nmr_core::nmr1d::load_1d_input(rp, None, false) {
                                if let Ok(mut spec) = prepare_loaded_1d_for_studio(loaded) {
                                    if !exp.relative_path.is_empty() && exp.relative_path != "." {
                                        spec.name = exp.relative_path.clone();
                                    }
                                    spectra.push(spec);
                                }
                            }
                        }
                    }
                }
            }
            // Also pick up any standalone `.jdx`/`.dx`/`.jcamp` files directly inside `path`
            if spectra.is_empty() {
                if let Ok(rd) = std::fs::read_dir(path) {
                    let mut children: Vec<_> = rd.flatten().map(|e| e.path()).collect();
                    children.sort();
                    for ch in children {
                        let cl = ch.to_string_lossy().to_ascii_lowercase();
                        if cl.ends_with(".jdx") || cl.ends_with(".dx") || cl.ends_with(".jcamp") {
                            if let Ok(s) = nmr_core::nmr1d::load_processed_spectrum_1d(&ch, None) {
                                spectra.push(s);
                            }
                        } else if ch.is_dir() {
                            if let Ok(loaded) = nmr_core::nmr1d::load_1d_input(&ch, None, false) {
                                if let Ok(s) = prepare_loaded_1d_for_studio(loaded) {
                                    spectra.push(s);
                                }
                            }
                        }
                    }
                }
            }
        }
        if !spectra.is_empty() {
            return Ok(IngestDropOutcome::Spectra1D {
                source_name: panel.inventory.root_name.clone(),
                spectra,
            });
        }
    }

    Ok(IngestDropOutcome::Workspace(Box::new(panel)))
}

/// Inspect `egui::DroppedFileHandle` items on native desktop (`DroppedFile::path` + fallback `DroppedFile::bytes`).
#[cfg(not(target_arch = "wasm32"))]
pub fn ingest_dropped_files_native(
    files: &[egui::DroppedFileHandle],
) -> Result<IngestDropOutcome, String> {
    if files.is_empty() {
        return Err("No dropped files provided".to_string());
    }

    if files.len() == 1 {
        let f = &files[0];
        let p = f.path();
        let p_str = p.display().to_string();
        let lower = p_str.to_ascii_lowercase();

        if lower.ends_with(".nef") {
            let nef_bytes = if p.exists() {
                std::fs::read(p)
                    .map_err(|e| format!("failed to read dropped NEF file {}: {e}", p.display()))?
            } else {
                f.bytes()?
            };
            return Ok(IngestDropOutcome::NefSession {
                source_name: p_str,
                nef_bytes,
            });
        }

        if p.exists() {
            if p.is_file()
                && !is_archive_name_lower(&lower)
                && (lower.ends_with(".str") || lower.ends_with(".star") || lower.ends_with(".tsv"))
            {
                let text = std::fs::read_to_string(p)
                    .map_err(|e| format!("failed to read {}: {e}", p.display()))?;
                return Ok(IngestDropOutcome::TextTable {
                    source_name: p_str,
                    text,
                });
            }
            return load_native_path_1d_or_workspace(p);
        }
    }

    // Multi-file native drop on disk: if all dropped items are files inside the same 1D folder (e.g. `1r` + `procs`),
    // try 1D payload ingest before falling back to 2D/3D workspace classification!
    if files.iter().all(|f| f.path().is_file()) {
        let mut payloads = Vec::new();
        let mut ok_all = true;
        for f in files {
            match std::fs::read(f.path()) {
                Ok(b) if b.len() <= 64 * 1024 * 1024 => {
                    payloads.push(DroppedPayload::new(f.path().display().to_string(), b));
                }
                _ => {
                    ok_all = false;
                    break;
                }
            }
        }
        if ok_all {
            if let Ok(out @ IngestDropOutcome::Spectra1D { .. }) =
                ingest_dropped_payloads(&payloads)
            {
                return Ok(out);
            }
        }
    }

    // Multi-file native drop on disk (e.g. multiple `.ucsf` + `.fasta`) or in-memory `DroppedFile::bytes`
    if files.iter().all(|f| f.path().exists()) {
        let mut combined_exps = Vec::new();
        let mut combined_aux = Vec::new();
        let mut row_paths = Vec::new();
        let mut seq_found: Option<(String, String)> = None;

        for f in files {
            let p = f.path();
            let inv = classify_nmr_workspace(p)?;
            for exp in inv.experiments {
                let rp = if p.is_dir() && exp.relative_path != "." && !exp.relative_path.is_empty()
                {
                    p.join(&exp.relative_path)
                } else {
                    p.to_path_buf()
                };
                row_paths.push(Some(rp));
                combined_exps.push(exp);
            }
            for aux in inv.auxiliary_files {
                if seq_found.is_none() {
                    let ap = if p.is_dir() {
                        p.join(&aux.relative_path)
                    } else {
                        p.to_path_buf()
                    };
                    if let Ok(txt) = std::fs::read_to_string(&ap) {
                        if let Some(seq1) =
                            extract_sequence_from_auxiliary_text(&aux.relative_path, &txt)
                        {
                            seq_found = Some((
                                format!("{} ({} aa)", aux.relative_path, seq1.chars().count()),
                                seq1,
                            ));
                        }
                    }
                }
                combined_aux.push(aux);
            }
        }

        let mut inv = WorkspaceInventory {
            root_name: format!("{} dropped items", files.len()),
            experiments: combined_exps,
            auxiliary_files: combined_aux,
            problem_regime: ProblemRegime::GlobularBackbone,
            secondary_regimes: Vec::new(),
            missing_input_hints: Vec::new(),
            elapsed_ms: 0.0,
        };
        recompute_workspace_regime(&mut inv);
        let mut panel = IngestPanelState::from_inventory(inv);
        panel.row_resolved_paths = row_paths;
        if let Some((lbl, seq1)) = seq_found {
            panel.sequence_source_label = Some(lbl);
            panel.sequence_text = seq1;
        }
        return Ok(IngestDropOutcome::Workspace(Box::new(panel)));
    }

    let mut payloads = Vec::with_capacity(files.len());
    for f in files {
        let name = f.path().display().to_string();
        let bytes = if f.path().exists() {
            std::fs::read(f.path())
                .map_err(|e| format!("failed to read {}: {e}", f.path().display()))?
        } else {
            f.bytes()?
        };
        payloads.push(DroppedPayload::new(name, bytes));
    }
    ingest_dropped_payloads(&payloads)
}

/// Asynchronously inspect `egui::DroppedFileHandle` items on `wasm32-unknown-unknown` via `DroppedFile::bytes_async`.
#[cfg(target_arch = "wasm32")]
pub async fn ingest_dropped_files_wasm(
    files: Vec<egui::DroppedFileHandle>,
) -> Result<IngestDropOutcome, String> {
    let mut payloads = Vec::with_capacity(files.len());
    for f in files {
        let name = f.path().display().to_string();
        let bytes = f.bytes_async().await.map_err(|e| {
            if e.contains("NotFoundError") || e.contains("The object can not be found") {
                format!(
                    "Cannot read folder '{name}' as a single file — click 'Open Folder' in the toolbar or drop a .zip/.tar.gz archive (or drop the folder directly onto the canvas)."
                )
            } else {
                e
            }
        })?;
        payloads.push(DroppedPayload::new(name, bytes));
    }
    ingest_dropped_payloads(&payloads)
}

fn regime_badge_colors(regime: ProblemRegime) -> (egui::Color32, egui::Color32) {
    match regime {
        ProblemRegime::GlobularBackbone => (
            egui::Color32::from_rgb(20, 62, 54),
            egui::Color32::from_rgb(105, 240, 185),
        ),
        ProblemRegime::SidechainNoesy => (
            egui::Color32::from_rgb(24, 54, 78),
            egui::Color32::from_rgb(125, 215, 255),
        ),
        ProblemRegime::Idp13cDetected => (
            egui::Color32::from_rgb(56, 38, 82),
            egui::Color32::from_rgb(210, 175, 255),
        ),
        ProblemRegime::MethylTrosy => (
            egui::Color32::from_rgb(66, 52, 22),
            egui::Color32::from_rgb(255, 215, 115),
        ),
        ProblemRegime::Paramagnetic => (
            egui::Color32::from_rgb(72, 34, 38),
            egui::Color32::from_rgb(255, 155, 155),
        ),
        ProblemRegime::RelaxationDynamics | ProblemRegime::TitrationCsp => (
            egui::Color32::from_rgb(28, 56, 70),
            egui::Color32::from_rgb(135, 230, 240),
        ),
        ProblemRegime::SmallMolecule2D => (
            egui::Color32::from_rgb(18, 60, 56),
            egui::Color32::from_rgb(115, 245, 205),
        ),
    }
}

fn format_axes_summary(exp: &ClassifiedExperiment) -> String {
    if exp.axes.is_empty() {
        return exp.quadrature_badge.clone();
    }
    let nucs: Vec<&str> = exp.axes.iter().map(|a| a.nucleus.as_str()).collect();
    format!("{} · {}", nucs.join("/"), exp.quadrature_badge)
}

impl IngestPanelState {
    /// Render the modal **Editable Experiment Selection Matrix** (`T09.2`) and return any triggered action.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<IngestAction> {
        if !self.open {
            return None;
        }

        let mut window_open = self.open;
        let mut emitted: Option<IngestAction> = None;

        egui::Window::new("Spectrometer Folder & Archive Ingest Matrix")
            .id(egui::Id::unique("nmr_ingest_selection_matrix_modal"))
            .open(&mut window_open)
            .collapsible(false)
            .resizable(true)
            .default_width(980.0)
            .default_height(610.0)
            .show(ctx, |ui| {
                // 1. Native Path Bar + Scan button
                #[cfg(not(target_arch = "wasm32"))]
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Folder / Archive:").strong());
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.workspace_path_input)
                            .hint_text("Drop a Bruker/Varian/UCSF/NMRPipe folder or .tar/.tar.gz archive, or enter path…")
                            .desired_width(560.0),
                    );
                    let enter_scan =
                        resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui.button("Scan Folder / Archive").clicked() || enter_scan {
                        let cleaned = sanitize_user_path_input(&self.workspace_path_input);
                        if !cleaned.is_empty() {
                            self.workspace_path_input = cleaned.clone();
                            emitted = Some(IngestAction::ScanNativePath(cleaned));
                        }
                    }
                    if self.inventory.elapsed_ms > 0.0 {
                        ui.label(
                            egui::RichText::new(format!("({:.1} ms)", self.inventory.elapsed_ms))
                                .small()
                                .monospace()
                                .weak(),
                        );
                    }
                });

                #[cfg(target_arch = "wasm32")]
                ui.horizontal(|ui| {
                    if ui
                        .button("Open Bruker / Spectrometer Folder…")
                        .on_hover_text("Select an entire Bruker study/expno/pdata folder from disk")
                        .clicked()
                    {
                        trigger_browser_folder_picker();
                    }
                    if ui
                        .button("Open Files / .zip / .tar.gz…")
                        .on_hover_text("Select .zip/.tar.gz archives, .ucsf/.ft2/.jdx spectra, or sibling Bruker files")
                        .clicked()
                    {
                        trigger_browser_files_picker();
                    }
                });

                if let Some(err) = &self.status_error {
                    ui.colored_label(egui::Color32::from_rgb(255, 130, 130), err);
                }

                // 2. Detected Primary / Secondary ProblemRegime Banner
                let (bg_col, fg_col) = regime_badge_colors(self.inventory.problem_regime);
                egui::Frame::group(ui.style())
                    .fill(bg_col)
                    .inner_margin(egui::Margin::symmetric(8, 5))
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "Detected Regime: {}",
                                    problem_regime_display_label(self.inventory.problem_regime)
                                ))
                                .strong()
                                .color(fg_col),
                            );
                            for &sec in &self.inventory.secondary_regimes {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "+ {}",
                                        problem_regime_display_label(sec)
                                    ))
                                    .small()
                                    .color(egui::Color32::from_rgb(210, 225, 245)),
                                );
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Root: {} ({} experiments)",
                                        self.inventory.root_name,
                                        self.inventory.experiments.len()
                                    ))
                                    .small()
                                    .monospace(),
                                );
                            });
                        });
                    });

                // 3. Missing-Input Hints Alert Strip
                let hints = self.dynamic_missing_hints();
                if !hints.is_empty() {
                    egui::Frame::group(ui.style())
                        .fill(egui::Color32::from_rgb(56, 43, 18))
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(215, 165, 65)))
                        .inner_margin(egui::Margin::symmetric(8, 4))
                        .show(ui, |ui| {
                            for h in &hints {
                                ui.label(
                                    egui::RichText::new(format!("Note: {h}"))
                                        .small()
                                        .color(egui::Color32::from_rgb(255, 218, 130)),
                                );
                            }
                        });
                }

                // 4. Inline Polymer Sequence Box (hidden / collapsed when in sequence-free SmallMolecule2D / DOSY regime)
                let seq_free = self.is_sequence_free_regime();
                let parsed_seq = self.parsed_sequence_three_letter();
                if seq_free && parsed_seq.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "2D Correlation / Diffusion suite ready — no polymer sequence required. Click 'Load Selected Spectra' below to inspect & overlay.",
                        )
                        .small()
                        .color(egui::Color32::from_rgb(125, 235, 185)),
                    );
                } else {
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "Polymer Sequence ({} aa):",
                                parsed_seq.len()
                            ))
                            .strong(),
                        );
                        if let Some(src) = &self.sequence_source_label {
                            ui.label(
                                egui::RichText::new(format!("Auto-discovered from {src}"))
                                    .small()
                                    .color(egui::Color32::from_rgb(115, 235, 175)),
                            );
                        }
                        if ui.small_button("Clear Sequence").clicked() {
                            self.sequence_text.clear();
                            self.sequence_source_label = None;
                        }
                    });
                    ui.add(
                        egui::TextEdit::multiline(&mut self.sequence_text)
                            .font(egui::TextStyle::Monospace)
                            .desired_rows(2)
                            .desired_width(f32::INFINITY)
                            .hint_text(
                                "Paste FASTA, 1-letter (MQIFVKTL…), or 3-letter sequence here…",
                            ),
                    );
                }

                // 5. Matrix Selection Toolbar & Scrollable Editable Experiment Grid
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "Selected: {} / {} experiments",
                            self.selected_count(),
                            self.inventory.experiments.len()
                        ))
                        .strong(),
                    );
                    if ui.small_button("Select All 2D/3D").clicked() {
                        self.select_all_2d_3d();
                    }
                    if ui.small_button("Clear Selection").clicked() {
                        self.deselect_all();
                    }
                    ui.label(
                        egui::RichText::new("1D checks & Unknown decoys are unchecked by default.")
                            .small()
                            .weak(),
                    );
                });

                egui::ScrollArea::both()
                    .id_salt("ingest_matrix_scroll")
                    .max_height(275.0)
                    .show(ui, |ui| {
                        egui::Grid::new("ingest_experiment_matrix_grid")
                            .striped(true)
                            .min_col_width(42.0)
                            .show(ui, |ui| {
                                ui.strong("Use");
                                ui.strong("Relative Path");
                                ui.strong("Pulse Sequence Class");
                                ui.strong("Vendor");
                                ui.strong("Dim");
                                ui.strong("State");
                                ui.strong("MHz");
                                ui.strong("Axes & Quadrature");
                                ui.strong("NUS");
                                ui.end_row();

                                let n_rows = self.inventory.experiments.len();
                                for i in 0..n_rows {
                                    let mut checked = self.selected[i];
                                    if ui.checkbox(&mut checked, "").changed() {
                                        self.set_row_selected(i, checked);
                                    }

                                    let (rel_path, pp_tip, vendor, dim, dstate, mhz, axes_txt, nus_is, nus_badge) = {
                                        let exp = &self.inventory.experiments[i];
                                        let tip = format!(
                                            "Display: {}\nPULPROG/seqfil: {}\nTitle: {}",
                                            exp.display_name,
                                            exp.pulse_program.as_deref().unwrap_or("-"),
                                            exp.title.as_deref().unwrap_or("-")
                                        );
                                        (
                                            exp.relative_path.clone(),
                                            tip,
                                            exp.vendor,
                                            exp.dimensionality,
                                            exp.data_state,
                                            exp.spectrometer_mhz,
                                            format_axes_summary(exp),
                                            exp.nus.is_nus,
                                            exp.nus.badge.clone(),
                                        )
                                    };

                                    ui.monospace(&rel_path).on_hover_text(pp_tip);

                                    let cur_cls = self.effective_classes[i];
                                    let overridden = self.user_overridden[i];
                                    let combo_lbl = if overridden {
                                        format!("* {}", pulse_class_display_label(cur_cls))
                                    } else {
                                        pulse_class_display_label(cur_cls).to_string()
                                    };
                                    egui::ComboBox::from_id_salt(("ingest_cls_override", i))
                                        .selected_text(combo_lbl)
                                        .width(170.0)
                                        .show_ui(ui, |ui| {
                                            for &cand in ALL_PULSE_CLASSES {
                                                if ui
                                                    .selectable_label(
                                                        cand == cur_cls,
                                                        pulse_class_display_label(cand),
                                                    )
                                                    .clicked()
                                                {
                                                    self.override_pulse_class(i, cand);
                                                }
                                            }
                                        });

                                    ui.monospace(vendor.as_str());
                                    ui.monospace(dim.as_str());

                                    let state_col = match dstate {
                                        DataState::Processed | DataState::Both => {
                                            egui::Color32::from_rgb(110, 230, 175)
                                        }
                                        DataState::RawFid => egui::Color32::from_rgb(255, 205, 110),
                                    };
                                    ui.colored_label(state_col, dstate.as_str());

                                    if let Some(f_mhz) = mhz {
                                        ui.monospace(format!("{f_mhz:.0}"));
                                    } else {
                                        ui.monospace("-");
                                    }

                                    ui.label(egui::RichText::new(axes_txt).small().monospace());

                                    let nus_col = if nus_is {
                                        egui::Color32::from_rgb(125, 215, 255)
                                    } else {
                                        egui::Color32::from_rgb(145, 160, 180)
                                    };
                                    ui.colored_label(nus_col, nus_badge);
                                    ui.end_row();
                                }
                            });
                    });

                ui.separator();
                ui.horizontal(|ui| {
                    let has_sel = self.selected_count() > 0;
                    let primary_green = egui::Color32::from_rgb(82, 224, 152);
                    let primary_fg = egui::Color32::from_rgb(15, 25, 20);
                    if self.is_sequence_free_regime() {
                        let load_cta = egui::Button::new(
                            egui::RichText::new("Load Selected Spectra")
                                .strong()
                                .color(primary_fg),
                        )
                        .fill(primary_green);
                        if ui.add_enabled(has_sel, load_cta).clicked() {
                            emitted = Some(IngestAction::LoadSelectedSpectra(
                                self.build_execution_plan(),
                            ));
                        }
                        if ui
                            .add_enabled(has_sel, egui::Button::new("Process, Pick & Run"))
                            .clicked()
                        {
                            emitted = Some(IngestAction::ProcessPickAndRun(
                                self.build_execution_plan(),
                            ));
                        }
                    } else {
                        let run_btn = egui::Button::new(
                            egui::RichText::new("Process, Pick & Run")
                                .strong()
                                .color(primary_fg),
                        )
                        .fill(primary_green);
                        if ui.add_enabled(has_sel, run_btn).clicked() {
                            emitted = Some(IngestAction::ProcessPickAndRun(
                                self.build_execution_plan(),
                            ));
                        }
                        if ui
                            .add_enabled(has_sel, egui::Button::new("Load Selected Spectra"))
                            .clicked()
                        {
                            emitted = Some(IngestAction::LoadSelectedSpectra(
                                self.build_execution_plan(),
                            ));
                        }
                    }

                    let has_sel_1d = self
                        .inventory
                        .experiments
                        .iter()
                        .enumerate()
                        .any(|(idx, e)| {
                            self.selected.get(idx).copied().unwrap_or(false)
                                && matches!(e.dimensionality, ExperimentDimensionality::Dim1D)
                        });
                    if has_sel_1d
                        && ui
                            .button("Open Checked 1D in 1D Studio")
                            .on_hover_text("Send checked 1D experiments directly into 1D Array & Multiplet Studio")
                            .clicked()
                    {
                        emitted = Some(IngestAction::OpenSelected1dInStudio(
                            self.build_execution_plan(),
                        ));
                    }

                    if ui.button("Cancel").clicked() {
                        self.open = false;
                    }
                });
            });

        if !window_open {
            self.open = false;
        }
        if matches!(
            emitted,
            Some(
                IngestAction::LoadSelectedSpectra(_)
                    | IngestAction::ProcessPickAndRun(_)
                    | IngestAction::OpenSelected1dInStudio(_)
            )
        ) {
            self.open = false;
        }
        emitted
    }
}
