# nmr-assign Documentation Manual

**`nmr-assign` (`v0.12.0`)** is an open-source, 100% clean-room biomolecular and analytical NMR platform written in pure **Rust** and **WebGPU** (`wgpu` / `WGSL`). Starting from raw Bruker (`fid`/`ser` + executed `pulseprogram`) or Varian/Agilent (`fid` + `procpar`) time-domain acquisitions, processed 2D/3D/4D spectra, or picked peak tables, `nmr-assign` delivers automated resonance assignments with honest empirical confidence calibration, 3D macromolecular ensembles, quantitative spin relaxation and chemical exchange kinetics, small-molecule 1D multiplet / `qNMR` / `DOSY` analytics, untargeted 1D metabolomics (`AI-Binning`, `PQN`, `PCA`, `OPLS-DA`, `STOCSY`), an integrated 2D chemical structure studio (`Octet-Draw`), and a sandboxed Model Context Protocol (`MCP`) server + live desktop bridge.

> [!TIP] Quick Navigation & Zero-Install Live Browser Demo
> - Press **`/`** or **`Cmd/Ctrl+K`** anywhere in this manual to search across all chapters, equations, CLI flags, MCP tools, and keyboard shortcuts. Toggle **Dark / Light** mode in the top-right header.
> - Launch the **[Zero-Install Live WebGPU Browser Demo](demo/index.html)** directly in Chromium/Firefox/Safari with WebGPU enabled—preloaded in memory (`0` filesystem I/O) with synthetic Ubiquitin `2D HSQC` + `3D HNCACB` strips + `1UBQ` 3D coordinates + 1D multiplet stack + `DOSY` decay series + 2-group `Metabolomics` cohort!

---

## Architecture & Workspace Crates

Every algorithm in `nmr-assign` is implemented first as an audited `f64` reference engine in `nmr-core`, paired with portable `wgpu` compute kernels in `nmr-gpu` that execute identically across Apple Metal, Vulkan, DirectX 12, and browser WebGPU (`wasm32-unknown-unknown`):

| Workspace Crate | Primary Responsibility | Targets |
| :--- | :--- | :--- |
| `crates/nmr-core` | Deterministic `f64` reference DSP, `pulseprogram` physics interpreter, hypercomplex `IST`/`FISTA` NUS, classical & 3D CNN peak pickers, multi-regime backbone/sidechain/methyl solvers, `DatasetQcReport` isotonic calibration, 3D NOESY/torsion/`RDC` folding, $R_1$/$R_2$/Model-Free/`CEST`/`CPMG`/`CSP` kinetics, 1D Voigt/Hoye/`qNMR`/`DOSY`, untargeted 1D `metabolomics` (`AI-Binning`/`PQN`/`PCA`/`OPLS-DA`/`STOCSY`), `Octet` 2D cheminformatics, and JSON-RPC 2.0 `MCP` engine | Native (`x86_64`, `aarch64`) & `wasm32` |
| `crates/nmr-gpu` | Portable `@group(0)` `WGSL` compute shaders (`16 KiB` workgroup-SRAM multi-replica simulated annealing, `Q27.36` fixed-point energy parity, 3D separable CNN inference, Saupe `RDC` gradient batches, Monte Carlo relaxation, Bloch–McConnell `CEST`, `CPMG`, and `DOSY` VarPro+LM) with upfront `wgpu::Limits` & TDR chunking guards | Native GPU & Browser WebGPU |
| `crates/nmr-gui` | Immediate-mode `eframe` / `egui` + `wgpu` scientific workbench: 60+ fps GPU 2D contour overlays, synchronized 3D strip walks, sequence ribbon & 2D covalent sidechain graph, 1300+ fps 3D cartoon ribbon / putty / restraint-cylinder viewer, Dynamics Studio, 4-mode 1D Stack, 2D `DOSY` & 4-quadrant `Metabolomics` studios, `Octet` Chemical Studio, built-in zero-I/O Demo Suite, and loopback `MCP` bridge | Desktop App & Zero-Install Browser WASM |
| `crates/nmr-cli` | Unified `nmrassign` command-line binary exposing headless processing, assignment, structure folding, dynamics, 1D/`DOSY`/`metabolomics`, self-hosted REST job server (`serve`), `MCP` server (`mcp` / `mcp-config`), deterministic publication SVG figure export (`export-fig`), and reproducible BMRB benchmark suites | Native CLI (`macOS`, `Linux`, `Windows`) |

---

## Manual Chapter Guide

| Chapter | Scope & Core Workflows |
| :--- | :--- |
| **[01 · Quickstart & Multi-Regime Workflows](01-quickstart-and-workflows.md)** | 60-second CLI & drag-and-drop GUI quickstart, Experiment Selection Matrix, *"What Next?"* stepper, and end-to-end tutorials across **folded globular proteins**, **perdeuterated `13C`-Methyl TROSY (`ILVMAT`)**, **proline-rich `13C`-detected IDPs**, **paramagnetic metalloproteins (`PCS`/`PRE`)**, and **small-molecule 1D & `DOSY` mixtures**. |
| **[02 · Custom Pulse-Program Physics & DSP](02-custom-pulprog-and-dsp.md)** | Sandboxed executed Bruker `pulseprogram` (`#ifdef` / `ZGOPTNS`) & Varian `procpar` interpreter, `States-TPPI` vs Rance–Kay `Echo-Antiecho`, swapped `AQSEQ` (`321` vs `312`), empirical `{sign, fold}` `2D HSQC` strip-matching rescue, hypercomplex `IST`/`FISTA` NUS, and 1D Chen autophasing, Whittaker/Bernstein baseline & `HSVD` solvent suppression. |
| **[03 · Backbone Solvers, QC & Calibration](03-solvers-and-calibration.md)** | Mathematical guide to `standard` (CPU/WebGPU SA), `evo-graph` (stoichiometric evolutionary + factor-graph BP), and opt-in `bayes-sa` (Bishop et al. 2023 specific-heat $C_v(T)$ SA), plus `DatasetQcReport` gating, 10-bin PAV isotonic calibration, and **honest unseen-protein empirical precision caveats**. |
| **[04 · Dynamics, Exchange & Ligand CSP](04-dynamics-and-exchange.md)** | Locked-shape pseudo-3D plane integration (`SNR = 4`), GPU $R_1$/$R_2$/hetNOE & spectral density mapping, Lipari–Szabo / Mandel–Akke–Palmer `M1`–`M5` ($S^2, \tau_e, R_{\text{ex}}, S_f^2, \tau_s$) with anisotropic diffusion, 2-/3-state Bloch–McConnell `CEST`/`DEST`, Carver–Richards & Baldwin `CPMG` / $R_{1\rho}$, and tight-binding quadratic ligand `CSP` $K_d$ isotherms. |
| **[05 · Analytical 1D Multiplets, qNMR & DOSY](05-nmr1d-multiplets-and-dosy.md)** | Multi-nuclear (`1H`, `13C{1H}`, `DEPT-90/135`, `19F`, `31P`) Voigt deconvolution, recursive Hoye `J`-coupling trees & AB roofing, `ACS`/`JOC`/`RSC`/`Nature` journal round-trips, NIST-traceable GUM `qNMR` molar purity, 4-mode multi-spectrum overlay/stack/difference/heatmap, and `DOSY` discrete `BIC` Stejskal–Tanner vs opt-in Tikhonov `ILT` caveats. |
| **[06 · MCP Integration, CLI & Keyboard Reference](06-mcp-and-cli-reference.md)** | Complete `nmrassign` subcommand & flag dictionary (including `nmr1d`, `dosy`, `metabolomics`, `export-fig`, `mcp`, `mcp-config`), `MCP` `stdio` & authenticated loopback desktop bridge setup for **Antigravity**, **Claude Desktop**, and **Cursor**, `ReadOnly`/`ReadWrite` & path-sandbox guardrails, Undo-backed AI curation, and the two-key NMR spectrum chord cheat-sheet. |
| **[07 · Clean-Room Provenance & Bibliography](07-clean-room-provenance.md)** | Non-negotiable clean-room engineering rules, sequence-only & `HomologFirewall` scientific governance, `CC0 1.0` BMRB/wwPDB open-data chain of custody, `MIT`/`Apache-2.0` dependency audit, and the master peer-reviewed bibliography across every module. |
| **[08 · Untargeted 1D Metabolomics & Chemometrics](08-metabolomics-and-chemometrics.md)** | Interval FFT cohort alignment, recursive valley-seeking `AI-Binning`, `PQN` dilution normalization, `Pareto`/`UV` scaling, `PCA` Hotelling $T^2$ / $Q$-residual outlier diagnostics, leak-free `k`-fold `NIPALS` `OPLS-DA` ($Q^2$, permutation $p$-value, Westerhuis $P \gg N$ overfitting guards, `VIP`, Wiklund `S-plot`, `BH-FDR` Volcano), `STOCSY` driver-peak spin-system connectivity, and `MetabolomicsStudioPanel`. |

---

## Installation & Building from Source

### Prerequisites

- **Rust stable toolchain** (`1.81+`, via `rustup`) with `cargo`.
- **GPU backend (automatic)**: Metal on macOS, Vulkan on Linux/Windows, or DirectX 12 on Windows. When no hardware GPU adapter is present, all commands transparently support `--engine cpu` (`f64` multi-threaded CPU execution).
- **Optional WASM target** (for building the browser WebGPU bundle): `rustup target add wasm32-unknown-unknown`.

### Build Commands

```bash
# Build the release CLI binary (target/release/nmrassign)
cargo build --release -p nmr-cli

# Verify version, embedded short Git SHA, and hardware WebGPU adapter
./target/release/nmrassign --version

# Launch the native desktop WebGPU NMR & Chemical Studio workbench
cargo run --release -p nmr-gui

# Run the release workspace test suite and 5-seed realistic physics gate
cargo test --release --workspace
cargo run --release -p nmr-cli -- benchmark \
  --star data/fixtures/synthetic_ubiquitin.str --realistic --seeds 5
```

### Rebuilding This Documentation Manual Locally

The documentation site has zero external Python package or network dependencies:

```bash
python3 scripts/build_docs_manual.py --out target/manual --check

# Optional: compile and stage the live wasm32 WebGPU browser demo into target/manual/demo/
bash scripts/build_web_demo.sh --out target/manual/demo
```

Passing `--check` validates every cross-chapter link, `#anchor` fragment, static asset reference, Markdown table separator, clean-room token hygiene, and deterministic SHA-256 build reproducibility before exiting `0`.

---

## Open Interchange Formats

`nmr-assign` reads and writes open, documented scientific formats throughout every stage of the pipeline—never locking user projects into opaque binary blobs:

| Domain | Supported Input Formats | Supported Output / Session Formats |
| :--- | :--- | :--- |
| **Raw Time-Domain FIDs** | Bruker `fid`/`ser` + `acqus`/`acqu2s`/`acqu3s` + executed `pulseprogram` + `nuslist`/`vdlist`/`vclist`/`fqlist`/`difflist`; Varian/Agilent `fid`/`origfid` + `procpar` + `sampling.sch` | Processed NMRPipe (`.ft1`, `.ft2`, `.ft3`, `test%03d.ft3`), Bruker `pdata` (`2rr`, `3rrr`), Sparky `.ucsf`, plus deterministic `processing_recipe.json` |
| **Processed Spectra** | Bruker `pdata/1/{1r,2rr,3rrr}`, NMRPipe `.ft1`/`.ft2`/`.ft3`, Sparky `.ucsf`, `JCAMP-DX` (`.jdx`) | NMRPipe `.ft2`/`.ft3`, Bruker `pdata`, `.ucsf`, `JCAMP-DX` |
| **Peaks & Chemical Shifts** | 5- or 7-column Peak `TSV`, `NMR-STAR v3.2` (`.str`), NMR Exchange Format (`NEF v1.1`, `.nef`) | `NMR-STAR v3.2` (`_Atom_chem_shift` with IUPAC `Ambiguity_code`), lossless `NEF v1.1` session files, 7-column Peak `TSV` |
| **3D Structure & Restraints** | `.pdb`, `.cif` / `.mmcif`, multi-medium `RDC` `TSV`/`NEF`, `3J` scalar coupling & NOESY `TSV` | Multi-model `PDB` ensembles, homolog-firewalled torsion `TSV`/`JSON`/`NEF` (with `exclusion_list_sha256`), self-contained `HTML+SVG` quality reports |
| **2D Chemistry & 1D NMR** | `SMILES`, Reaction `SMILES`, `InChI`, `MOL` V2000/V3000, `SDF`, `CDXML`, binary `CDX`, `.oct`, IUPAC systematic names | `SVG`, `PDF 1.4`, antialiased `PNG`, `SDF` (`NMR_ASSIGNMENT`), `CDXML`/`CDX`, `.oct`, and `ACS`/`JOC`/`RSC`/`Nature` multiplet & `qNMR` `JSON`/`TSV` |
| **1D Metabolomics Cohorts** | `cohort_manifest.tsv` (`sample_id`, `group`, `path`) referencing `.jdx`, Bruker `1r`, NMRPipe `.ft1`, or `.tsv` 1D traces | `AI-Binning` / `PQN` feature matrix `TSV`, `PCA` Hotelling $T^2$/$Q$ `JSON`, `OPLS-DA` `VIP`/`S-plot`/`Volcano` `TSV`+`JSON`, and `STOCSY` connectivity `TSV` |
