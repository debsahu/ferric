# FERRIC Documentation Manual

**FERRIC** (**F**ID-to-structure **E**ngine for **R**elaxation, **R**esonance assignment, **I**ntegration & **C**hemometrics) is an open-source, 100% clean-room biomolecular and analytical NMR platform written in pure **Rust** and **WebGPU** (`wgpu` / `WGSL`). Starting from raw Bruker (`fid`/`ser` + executed `pulseprogram`) or Varian/Agilent (`fid` + `procpar`) time-domain acquisitions, processed 2D/3D/4D spectra, or picked peak tables, FERRIC delivers automated resonance assignments with honest empirical confidence calibration, 3D macromolecular ensembles, quantitative spin relaxation and chemical exchange kinetics, small-molecule 1D multiplet / `qNMR` / `DOSY` analytics, untargeted 1D metabolomics (`AI-Binning`, `PQN`, `PCA`, `OPLS-DA`, `STOCSY`), an integrated 2D chemical structure studio (`Octet-Draw`), and a sandboxed Model Context Protocol (`MCP`) server + live desktop bridge.

> [!TIP] Quick Navigation & Zero-Install Live Browser Demo
> - Press **`/`** or **`Cmd/Ctrl+K`** anywhere in this manual to search across all chapters, equations, CLI flags, MCP tools, and keyboard shortcuts. Toggle **Dark / Light** mode in the top-right header.
> - Launch the **[Zero-Install Live WebGPU Browser Demo](demo/index.html)** directly in Chromium/Firefox/Safari with WebGPU enabled—preloaded in memory (`0` filesystem I/O) with synthetic Ubiquitin `2D HSQC` + `3D HNCACB` strips + `1UBQ` 3D coordinates + 1D multiplet stack + `DOSY` decay series + 2-group `Metabolomics` cohort!

---

## Architecture & Workspace Crates

Every algorithm in FERRIC is implemented first as an audited `f64` reference engine in `nmr-core`, paired with portable `wgpu` compute kernels in `nmr-gpu` that execute identically across Apple Metal, Vulkan, DirectX 12, and browser WebGPU (`wasm32-unknown-unknown`):

| Workspace Crate | Primary Responsibility | Targets |
| :--- | :--- | :--- |
| `crates/nmr-core` | Deterministic `f64` reference DSP, `pulseprogram` physics interpreter, hypercomplex `IST`/`FISTA` NUS, classical & 3D CNN peak pickers, multi-regime backbone/sidechain/methyl solvers, `DatasetQcReport` isotonic calibration, 3D NOESY/torsion/`RDC` folding, $R_1$/$R_2$/Model-Free/`CEST`/`CPMG`/`CSP` kinetics, 1D Voigt/Hoye/`qNMR`/`DOSY`, untargeted 1D `metabolomics` (`AI-Binning`/`PQN`/`PCA`/`OPLS-DA`/`STOCSY`), `Octet` 2D cheminformatics, and JSON-RPC 2.0 `MCP` engine | Native (`x86_64`, `aarch64`) & `wasm32` |
| `crates/nmr-gpu` | Portable `@group(0)` `WGSL` compute shaders (`16 KiB` workgroup-SRAM multi-replica simulated annealing, `Q27.36` fixed-point energy parity, 3D separable CNN inference, Saupe `RDC` gradient batches, Monte Carlo relaxation, Bloch–McConnell `CEST`, `CPMG`, and `DOSY` VarPro+LM) with upfront `wgpu::Limits` & TDR chunking guards | Native GPU & Browser WebGPU |
| `crates/nmr-gui` | Immediate-mode `eframe` / `egui` + `wgpu` scientific workbench: 60+ fps GPU 2D contour overlays, synchronized 3D strip walks, sequence ribbon & 2D covalent sidechain graph, 1300+ fps 3D cartoon ribbon / putty / restraint-cylinder viewer, Dynamics Studio, 4-mode 1D Stack, 2D `DOSY` & 4-quadrant `Metabolomics` studios, `Octet` Chemical Studio, built-in zero-I/O Demo Suite, and loopback `MCP` bridge | Desktop App & Zero-Install Browser WASM |
| `crates/nmr-cli` | Unified `ferric` command-line binary (the `nmrassign` alias was removed in `v0.15.0`) exposing headless processing, assignment, structure folding, dynamics, 1D/`DOSY`/`metabolomics`, self-hosted REST job server (`serve`), `MCP` server (`mcp` / `mcp-config`), deterministic publication SVG figure export (`export-fig`), interactive tour export (`export-tour`), and reproducible BMRB benchmark suites | Native CLI (`macOS`, `Linux`, `Windows`) |

---

## Manual Chapter Guide

| Chapter | Scope & Core Workflows |
| :--- | :--- |
| **[01 · Quickstart & Multi-Regime Workflows](01-quickstart-and-workflows.md)** | 60-second CLI & drag-and-drop GUI quickstart, Experiment Selection Matrix, *"What Next?"* stepper, and end-to-end tutorials across **folded globular proteins**, **perdeuterated `13C`-Methyl TROSY (`ILVMAT`)**, **proline-rich `13C`-detected IDPs**, **paramagnetic metalloproteins (`PCS`/`PRE`)**, and **small-molecule 1D & `DOSY` mixtures**. |
| **[02 · Custom Pulse-Program Physics & DSP](02-custom-pulprog-and-dsp.md)** | Sandboxed executed Bruker `pulseprogram` (`#ifdef` / `ZGOPTNS`) & Varian `procpar` interpreter, `States-TPPI` vs Rance–Kay `Echo-Antiecho`, swapped `AQSEQ` (`321` vs `312`), empirical `{sign, fold}` `2D HSQC` strip-matching rescue, hypercomplex `IST`/`FISTA` NUS, and 1D Chen autophasing, Whittaker/Bernstein baseline & `HSVD` solvent suppression. |
| **[03 · Backbone Solvers, QC & Calibration](03-solvers-and-calibration.md)** | Mathematical guide to `standard` (CPU/WebGPU SA), `evo-graph` (stoichiometric evolutionary + factor-graph BP), and opt-in `bayes-sa` (Bishop et al. 2023 specific-heat $C_v(T)$ SA), plus `DatasetQcReport` gating, 10-bin PAV isotonic calibration, and **honest unseen-protein empirical precision caveats**. |
| **[04 · Dynamics, Exchange & Ligand CSP](04-dynamics-and-exchange.md)** | Locked-shape pseudo-3D plane integration (`SNR = 4`), GPU $R_1$/$R_2$/hetNOE & spectral density mapping, Lipari–Szabo / Mandel–Akke–Palmer `M1`–`M5` ($S^2, \tau_e, R_{\text{ex}}, S_f^2, \tau_s$) with anisotropic diffusion, 2-/3-state Bloch–McConnell `CEST`/`DEST`, Carver–Richards & Baldwin `CPMG` / $R_{1\rho}$, and tight-binding quadratic ligand `CSP` $K_d$ isotherms. |
| **[05 · Analytical 1D Multiplets, qNMR & DOSY](05-nmr1d-multiplets-and-dosy.md)** | Multi-nuclear (`1H`, `13C{1H}`, `DEPT-90/135`, `19F`, `31P`) Voigt deconvolution, recursive Hoye `J`-coupling trees & AB roofing, `ACS`/`JOC`/`RSC`/`Nature` journal round-trips, NIST-traceable GUM `qNMR` molar purity, 4-mode multi-spectrum overlay/stack/difference/heatmap, and `DOSY` discrete `BIC` Stejskal–Tanner vs opt-in Tikhonov `ILT` caveats. |
| **[06 · MCP Integration, CLI & Keyboard Reference](06-mcp-and-cli-reference.md)** | Complete `ferric` subcommand & flag dictionary (including `nmr1d`, `dosy`, `metabolomics`, `export-fig`, `mcp`, `mcp-config`), `MCP` `stdio` & authenticated loopback desktop bridge setup for **Antigravity**, **Claude Desktop**, and **Cursor**, `ReadOnly`/`ReadWrite` & path-sandbox guardrails, Undo-backed AI curation, and the two-key NMR spectrum chord cheat-sheet. |
| **[07 · Clean-Room Provenance & Bibliography](07-clean-room-provenance.md)** | Non-negotiable clean-room engineering rules, sequence-only & `HomologFirewall` scientific governance, `CC0 1.0` BMRB/wwPDB open-data chain of custody, `MIT`/`Apache-2.0` dependency audit, and the master peer-reviewed bibliography across every module. |
| **[08 · Untargeted 1D Metabolomics & Chemometrics](08-metabolomics-and-chemometrics.md)** | Interval FFT cohort alignment, recursive valley-seeking `AI-Binning`, `PQN` dilution normalization, `Pareto`/`UV` scaling, `PCA` Hotelling $T^2$ / $Q$-residual outlier diagnostics, leak-free `k`-fold `NIPALS` `OPLS-DA` ($Q^2$, permutation $p$-value, Westerhuis $P \gg N$ overfitting guards, `VIP`, Wiklund `S-plot`, `BH-FDR` Volcano), `STOCSY` driver-peak spin-system connectivity, and `MetabolomicsStudioPanel`. |
| **[09 · AlphaFold & Integrative Structural NMR](09-alphafold-and-integrative-nmr.md)** | Guarded `mmCIF`/`PAE` ingestion, Kabsch–Sander `8 -> 4`-state secondary structure & `RCI-S²` concordance (`af-validate`), firewalled 3D-to-shift forward physics (`predict-shifts`), opt-in Student-$t_4$ 3D-prior & `PAE`-gated `15N-NOESY` assignment (`assign --af-structure`), `PAE` spectral multi-domain `RDC`/`PCS`/`PRE` docking (`structure --pae-domains`), `AlphaFold 3` JSON / `Boltz-2` YAML co-folding bridge (`cofold`), and Bayesian `CEST`/`CPMG` excited-state hypothesis testing (`exchange-state`). |
| **[10 · References & Acknowledgments](10-references-and-acknowledgments.md)** | Complete Crossref-verified peer-reviewed bibliography across all 7 scientific domains (exact authors, titles, journals, volumes, pages, and DOIs), plus open-data licensing and community acknowledgments (`BMRB`/`BMRB-BMSE`, `RCSB PDB`, `AlphaFold DB`, `NP-MRD`, `UniProt`, `CATH`, `Octet-Draw`). |

---

## Feature & Access Map (GUI Path + CLI Command + Input Formats)

Every FERRIC workflow follows a single progressive path from **raw ingest → processing → analysis → open-format export**, reachable identically from `ferric-gui` (using the 6-mode segmented top switcher `Process` · `Assign` · `Structure` · `Dynamics` · `Small molecule` · `Metabolomics`, top menus, or `Cmd/Ctrl+K` Command Palette) and from the headless `ferric` CLI:

| Stage & Capability | GUI Access Path (`ferric-gui`) | CLI Command (`ferric`) | Accepted Input Formats | Primary Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **1. Raw Dataset Discovery & Ingest** | Drag & drop folder/archive onto canvas, or **File → Open Ingest Matrix...** (`Cmd/Ctrl+O`) | `ferric inspect <path> --json` | Bruker (`fid`/`ser`, `acqus*`, `pulseprogram`, `pdata/1`), Varian (`fid`, `procpar`), JEOL `.jdf`, NMRPipe `.ft*`, Sparky `.ucsf`, `.jdx`, `.tar`/`.tgz`, `.str`, `.nef` | Experiment Selection Matrix & JSON dataset inventory |
| **2. Raw FID DSP, NUS & Water Suppression** | **Process** mode → **Process → Process All Raw FIDs** / **Reconstruct NUS** / **Suppress Water Peak** (`ws`) | `ferric process --in <fid_dir> --out <out_path> --auto-phase --lp --nus-method fista` | Bruker `fid`/`ser` + `acqus*` + `nuslist`, Varian `fid` + `procpar` + `sampling.sch`, processed `.ft*`/`.ucsf` | Processed `.ft1`/`.ft2`/`.ft3`, Bruker `pdata`, `.ucsf`, `processing_recipe.json` |
| **3. 2D/3D Peak Picking & Weak-Peak Rescue** | **Assign** mode → **Peaks → Pick 2D HSQC Roots** / **Pick All 3D Spectra** / **Corroborated Rescue** | `ferric pick --fid-hsqc <dir> --fid-triple HNCACB=<dir> --out <peaks.tsv> --method classical` | Raw Bruker/Varian FID directories or processed `.ft2`/`.ft3`, Bruker `2rr`/`3rrr`, `.ucsf` | 7-column Peak `TSV` (`H`, `N`, `C`, `sign`, `intensity`, `snr`) |
| **4. Backbone Assignment (Globular, IDP `13C`, Paramagnetic)** | **Assign** mode → **Assignment → Run Backbone Assignment** (`evo-graph` / `standard` / `bayes-sa`, `3D` strip walk) | `ferric assign --seq <seq.fasta> --peaks <peaks.tsv> --out <prefix> --solver evo-graph --engine auto` | `FASTA`/sequence + Peak `TSV` (or `CON`/`CACO`/`CANCO`/`CBCACON` for `--regime idp-13c`; `--pdb` + `--pcs` for `--regime paramagnetic`) | `NMR-STAR v3.2` (`.str`), `NEF v1.1` (`.nef`), calibrated `DatasetQcReport` `JSON` & `HTML` report |
| **5. End-to-End One-Shot Pipeline** | **File → Open Ingest Matrix...** → follow 6-stage *"What Next?"* stepper | `ferric pipeline --seq <seq.fasta> --fid-hsqc <dir> --fid-triple HNCACB=<dir> --out-dir <dir> --engine auto` | Raw FIDs, processed spectra, Peak `TSV`, `.str`, or `.nef` | Full processed spectra, peaks, assignments, `.nef`, and self-contained `HTML+SVG` QC report |
| **6. Sidechain, Aromatic, Stereospecific & Methyl-TROSY** | **Assign** mode → **Structure → Run Joint Sidechain Assignment** / **Inspect Sidechain Covalent Graph** | `ferric sidechain --star <ref.str> --peaks <peaks.tsv> --aromatic --stereospec --out <all_atom.str>` · `ferric methyl --pdb <model.pdb> --hmqc <hmqc> --noesy <noesy> --labeling ILVMAT -o <out.str>` | Assigned `.str` + sidechain/aromatic/methyl `TSV` or 2D/3D/4D spectra, `3J` couplings `TSV`, `PDB` coordinates | All-atom & stereospecific `NMR-STAR v3.2` (`Ambiguity_code`) and `NEF v1.1` shifts |
| **7. Secondary Structure, Torsions & 3D NOE/RDC Folding** | **Structure** mode → **Structure → Predict Backbone Dihedrals** / **Assign NOESY** / **Fold Ensemble** | `ferric propensity --shifts <assigned.str>` · `ferric torsions --shifts <assigned.str> -o <dihedrals.nef> --include-chi1` · `ferric structure --star <assigned.str> --noesy-peaks <noesy.tsv> --rdc <rdcs.tsv> --torsion-db --out-pdb <ensemble.pdb>` | Assigned `.str`/`.nef`, `3D NOESY` peak `TSV`, multi-medium `RDC` `TSV`/`NEF` | `SSP`/`δ2D` tracks, firewalled `(\phi, \psi, \chi_1)` restraints (`SHA-256` digest), multi-model `PDB` ensemble |
| **8. AlphaFold Concordance, 3D Shift Forward Physics & Integrative Modeling** | **Structure** mode → **PAE & Co-Fold** sub-view (concordance ribbon, `PAE` heatmap, co-fold ranker, excited-state explorer) | `ferric af-validate --structure <model.cif> --shifts <assigned.str> -o <qc.json>` · `ferric predict-shifts --structure <model.cif> -o <pred.str>` · `ferric structure --pae-domains --af-structure <model.cif> --af-pae <pae.json> --rdc <rdcs.tsv> -o <docked.cif>` · `ferric cofold export` / `ferric exchange-state` | `.cif`/`.mmcif` (`_ma_qa_metric_local`), `.pdb`, `PAE` `JSON` (`AFDB`/`ColabFold`/`AF3`), `.str`/`.nef`, `RDC`/`PCS`/`PRE`/`CEST`/`CPMG` `TSV` | Rigidity concordance `JSON`/`HTML`, firewalled 3D shift `.str`, docked multi-domain `.cif`, `AF3` `JSON`/`Boltz-2` `YAML`, excited-state `BIC` `JSON` |
| **9. Spin Relaxation (`R1`/`R2`/`hetNOE`) & Model-Free `S^2`** | **Dynamics** mode → **Relaxation** sub-view → **Dynamics → Fit R1/R2/hetNOE** / **Run Model-Free** | `ferric relax --series <dir_or_tsv> --kind r1 --peaks <peaks.tsv> --out <r1.tsv>` · `ferric modelfree --r1 <r1.tsv> --r2 <r2.tsv> --hetnoe <noe.tsv> --pdb <model.pdb> --out <mf.tsv>` | Bruker pseudo-3D `ser` + `vdlist`/`vclist`, Varian arrayed `fid` + `procpar`, `2D` stacks, or intensity `TSV`/`NEF` | Per-residue `R1`/`R2`/`hetNOE`, reduced spectral densities, `M1`–`M5` `S^2`/`R_ex` `TSV` & 3D putty |
| **10. Chemical Exchange (`CEST`/`CPMG`/`R1rho`) & Ligand `CSP` `Kd`** | **Dynamics** mode → **Exchange** sub-view (`CEST`/`DEST`, `CPMG`/`R1rho`, `CSP` `Kd` tabs) | `ferric cest --in <profiles.tsv> --out <cest.tsv>` · `ferric cpmg --in <cpmg.tsv> --model baldwin2014 --out <cpmg.tsv>` · `ferric csp --apo-peaks <apo.tsv> --step 50=<t1.tsv> --out <csp.tsv>` | Multi-$B_1$ `CEST`/`DEST` `TSV`, multi-field `CPMG`/`R1rho` `TSV`, apo + ligand-titration peak `TSV`/`.str` | Global `(p_B, k_ex, \Delta\varpi)` fits, `\chi^2` surfaces, trajectories, `2\sigma` binding sites & `K_d` `TSV`/`JSON` |
| **11. Analytical 1D (`1H`/`13C`/`DEPT`/`19F`/`31P`), Multiplets, `qNMR` & `Octet` Chemistry** | **Process** mode (1D Stack & phase/baseline/multiplet tools) + **Small molecule** mode → **Sketcher** sub-view | `ferric nmr1d process --in <1d_fid> --auto-phase --baseline whittaker --out <spec.json>` · `ferric nmr1d multiplet --in <spec.json> --style acs --out <mult.tsv>` · `ferric nmr1d qnmr` · `ferric chem` | Bruker `1D fid`, Varian `fid`, JEOL `.jdf`, `.jdx`, `SMILES`/`InChI`/`MOL`/`SDF`/`CDXML`/`CDX`/`.oct` | Deconvolved multiplets, `ACS`/`JOC`/`RSC`/`Nature` reports, GUM `qNMR` purity, `SVG`/`PDF`/`PNG` depictions |
| **12. Pseudo-2D `DOSY` Diffusion & Hydrodynamic Sizing** | **Small molecule** mode → **DOSY** sub-view (2D `(\delta, \log_{10} D)` map + decay inspector) | `ferric dosy fit --in <dosy_dir_or_tsv> --criterion bic -o <dosy.json>` · `ferric dosy mw` | Bruker pseudo-2D `ser` + `difflist`, Varian `fid` + `procpar`, or gradient-attenuation `TSV` | Discrete `BIC` Stejskal–Tanner / opt-in `ILT` `D` fits, Stokes–Einstein `R_h` & `MW` `JSON`/`TSV` |
| **13. Untargeted 1D Metabolomics & Chemometrics** | **Metabolomics** mode (4-quadrant Cohort Overlay, `PCA`/`OPLS-DA` Scores, `S-plot`/Volcano, `STOCSY` Studio) | `ferric metabolomics oplsda --manifest <cohort_manifest.tsv> --out <study.json>` · `ferric metabolomics stocsy` | `cohort_manifest.tsv` referencing `.jdx`, Bruker `1r`, NMRPipe `.ft1`, or 1D `TSV`/`CSV` | `AI-Binning`/`PQN` tables, `PCA` `T^2`/`Q`, leak-free `OPLS-DA` (`Q^2`, permutation `p`), `VIP`/`S-plot`/`BH-FDR` & `STOCSY` |
| **14. Reproducible Workflow `DAG`, `MCP` AI Copilot & Publication Export** | **Tools → Workflow DAG Composer** (or `Cmd/Ctrl+K` → `Workflow DAG Composer`); loopback `MCP` bridge in top status panel | `ferric dag preset backbone -o <dag.json>` · `ferric dag run --in <dag.json> --out-dir <dir>` · `ferric mcp --transport stdio --root <dir> --mode read-only` · `ferric export-fig --out-dir <dir> --panel all` | `ferric-dag-v1` `JSON` graphs/manifests, workspace directories, `.nef`/`.str` projects | Content-addressed `SHA-256` `DAG` cache & traces, `JSON-RPC 2.0` `MCP` tools, `HTML` reports, deterministic `SVG` figures |

---

## Interactive Module Video Tours (`/tours/`)

Launch the **[Interactive Showcase Cinema Gallery](tours/index.html)** (`60 fps` HTML5 `<canvas>` live parameter explorer + pre-rendered `1280×720` `.mp4` / `.webm` HD videos generated deterministically from `ferric export-tour`):

| Module | Interactive Tour & HD Video | Validation Scope |
| :--- | :--- | :--- |
| **01** | **[Raw-FID DSP & Pulprog Engine](tours/01_fid_pulprog_nus.html)** | Raw Bruker/Varian FID & `pulprog` AST ingest; `25%` Poisson-gap NUS IST/FISTA demo is **synthetic** |
| **02** | **[HSQC Strips & Backbone Solvers](tours/02_hsqc_strips_solvers.html)** | **Synthetic** Ubiquitin (`--realistic` `98.4%`, `310/315`), **dev** `CISD3` (`BMRB 51439`: `64.38%`, `59/59` HC); default `evo-graph` unseen cold micro HC-prec `88.25%–95.30%` (latest `88.25%` **FAILS** `>= 94%` gate; cold HCP claim formally **withdrawn** under `13.1:P1-2`) |
| **03** | **[3D Structure, NOE & RDC](tours/03_structure_noe_rdc.html)** | **Cold Holdout Disclosed (`F8`)**: unseen `1KHM -> 1ZZK` (`1.86–4.14 Å`, `0/5 < 1.80 Å`) |
| **04** | **[Relaxation & Lipari-Szabo](tours/04_relax_modelfree.html)** | `SYNTHETIC FORWARD-MODEL SHOWCASE` (`F10`/`F11` open) |
| **05** | **[CEST, CPMG & Tight-Binding CSP](tours/05_exchange_cest_cpmg_csp.html)** | `SYNTHETIC FORWARD-MODEL SHOWCASE` (`F13` open) |
| **06** | **[1D Multiplets, qNMR & Octet](tours/06_nmr1d_multiplet_qnmr_octet.html)** | `SYNTHETIC FORWARD-MODEL SHOWCASE` (`F14` open) |
| **07** | **[DOSY Diffusion & Rh Sizing](tours/07_dosy_diffusion.html)** | `SYNTHETIC FORWARD-MODEL SHOWCASE` (`F16` open) |
| **08** | **[Metabolomics, STOCSY & MCP](tours/08_metabolomics_mcp.html)** | `SYNTHETIC & PROTOCOL SHOWCASE` (`F17`/`F18` open) |

---

## Installation & Building from Source

### Prebuilt Packages (NMR Workstations, macOS & Windows)

#### RHEL 9 / AlmaLinux 9 / Rocky 9 / Fedora (Bruker & Varian Workstations with NVIDIA GPU)

FERRIC's `linux-x86_64` release binaries are compiled inside AlmaLinux 9 (`glibc <= 2.34`) specifically for spectrometer acquisition and processing workstations, and distributed through the official GPG-signed DNF/YUM RPM repository ([debsahu.github.io/ferric-rpm](https://debsahu.github.io/ferric-rpm/), plus 32-bit `i386`/`i686` static CLI RPMs):

```bash
# Signed DNF/YUM RPM repository install & future upgrades
sudo curl -fsSL https://debsahu.github.io/ferric-rpm/ferric.repo -o /etc/yum.repos.d/ferric.repo
sudo dnf install -y ferric
sudo dnf upgrade ferric    # subsequent releases

# Verify CLI and hardware Vulkan GPU compute
ferric --version
ferric-gui
```

- **GPG Verification (`gpgcheck=1` + `repo_gpgcheck=1`)**: Packages and `repomd.xml` metadata are signed with [`RPM-GPG-KEY-ferric`](https://debsahu.github.io/ferric-rpm/RPM-GPG-KEY-ferric) (RSA-4096 / SHA-256 fingerprint `A484 B4E4 D0D7 6A41 91DD 6228 8E1B 8E3E EAF5 655B`).
- **Air-gapped / direct `.rpm` install** (for spectrometer hosts without outbound HTTPS):
  ```bash
  VER=0.17.2
  TMP="$(mktemp -d)" && (
    cd "$TMP"
    curl -fsSLO "https://github.com/debsahu/ferric/releases/download/v${VER}/ferric-v${VER}-linux-x86_64.rpm"
    curl -fsSLO "https://github.com/debsahu/ferric/releases/download/v${VER}/SHA256SUMS.txt"
    grep "ferric-v${VER}-linux-x86_64.rpm" SHA256SUMS.txt | sha256sum -c -
    sudo dnf install -y "./ferric-v${VER}-linux-x86_64.rpm"
  )
  rm -rf "$TMP"
  ```

- **NVIDIA Vulkan GPU**: Proprietary NVIDIA Vulkan drivers are selected automatically by `wgpu` for both `ferric-gui` and `ferric ... --engine gpu` (verified on RHEL 9.8 + NVIDIA Quadro P400). You can confirm hardware GPU execution from a source checkout or with any `.str` file via:
  ```bash
  ferric benchmark --star data/fixtures/synthetic_ubiquitin.str --realistic --seeds 1 --engine gpu
  ```
- **Minimal RHEL 9 installs**: If a minimal or server install is missing X11/Wayland/Vulkan desktop runtime libraries for `ferric-gui`, install them with:
  ```bash
  sudo dnf install -y libxkbcommon libX11 vulkan-loader mesa-vulkan-drivers
  ```
- **No-`sudo` user-local install into `~/.local/bin` (with desktop launcher & icons)**:
  ```bash
  VER=0.17.2
  TMP="$(mktemp -d)" && (
    cd "$TMP"
    curl -fsSLO "https://github.com/debsahu/ferric/releases/download/v${VER}/ferric-v${VER}-linux-x86_64.tar.gz"
    curl -fsSLO "https://github.com/debsahu/ferric/releases/download/v${VER}/SHA256SUMS.txt"
    grep "ferric-v${VER}-linux-x86_64.tar.gz" SHA256SUMS.txt | sha256sum -c -
    tar -xzf "ferric-v${VER}-linux-x86_64.tar.gz"
    "./ferric-v${VER}-linux-x86_64/install-desktop.sh" --prefix "$HOME/.local"
  )
  rm -rf "$TMP"
  "$HOME/.local/bin/ferric" --version
  ```

#### Other Prebuilt Platforms & Direct Cargo Install

- **Debian / Ubuntu (`x86_64`)**: Download `ferric-v0.17.2-linux-x86_64.deb` from GitHub Releases and run `sudo apt install ./ferric-v0.17.2-linux-x86_64.deb`.
- **Legacy 32-bit Linux (`i686` CLI-only)**: Static musl `ferric` CLI packages (`ferric-v0.17.2-linux-i686-cli.rpm`, `.deb`, and `.tar.gz`).
- **macOS (Apple Silicon & Intel)**:
  ```bash
  brew tap debsahu/tap
  brew install ferric                        # CLI (macOS arm64/x86_64 + Linux x86_64)
  brew install --cask --no-quarantine ferric # FERRIC.app GUI + CLI (macOS arm64/x86_64)
  ```
  Or download the `.dmg` / `.tar.gz` archives directly from [**GitHub Releases**](https://github.com/debsahu/ferric/releases). Because `FERRIC.app` is ad-hoc signed (not Apple-notarized), after dragging it from a `.dmg` into `/Applications`, clear the download quarantine flag once with `xattr -dr com.apple.quarantine /Applications/FERRIC.app` (or click **Open Anyway** in **System Settings → Privacy & Security**; `brew install --cask --no-quarantine ferric` skips this automatically).
- **Windows (`x86_64`)**: Download `ferric-v0.17.2-windows-x86_64.zip` (`ferric-gui.exe` + `ferric.exe`).
- **Direct Cargo Install (any platform with Rust)**:
  ```bash
  cargo install --git https://github.com/debsahu/ferric.git --tag v0.17.2 nmr-cli
  cargo install --git https://github.com/debsahu/ferric.git --tag v0.17.2 nmr-gui
  ```

### Building from Source — Prerequisites

- **Rust stable toolchain** (`1.81+`, via `rustup`) with `cargo`.
- **GPU backend (automatic)**: Metal on macOS, Vulkan on Linux/Windows, or DirectX 12 on Windows. When no hardware GPU adapter is present, all commands transparently support `--engine cpu` (`f64` multi-threaded CPU execution).
- **Optional WASM target** (for building the browser WebGPU bundle): `rustup target add wasm32-unknown-unknown`.

### Build Commands

```bash
# Build the release CLI binary (target/release/ferric)
cargo build --release -p nmr-cli

# Verify version, embedded short Git SHA, and hardware WebGPU adapter
./target/release/ferric --version

# Launch the native desktop WebGPU NMR & Chemical Studio workbench (ferric-gui)
cargo run --release -p nmr-gui --bin ferric-gui

# Run the release workspace test suite and 5-seed realistic physics gate
cargo test --release --workspace
cargo run --release -p nmr-cli --bin ferric -- benchmark \
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

FERRIC reads and writes open, documented scientific formats throughout every stage of the pipeline—never locking user projects into opaque binary blobs:

| Domain | Supported Input Formats | Supported Output / Session Formats |
| :--- | :--- | :--- |
| **Raw Time-Domain FIDs** | Bruker `fid`/`ser` + `acqus`/`acqu2s`/`acqu3s` + executed `pulseprogram` + `nuslist`/`vdlist`/`vclist`/`fqlist`/`difflist`; Varian/Agilent `fid`/`origfid` + `procpar` + `sampling.sch` | Processed NMRPipe (`.ft1`, `.ft2`, `.ft3`, `test%03d.ft3`), Bruker `pdata` (`2rr`, `3rrr`), Sparky `.ucsf`, plus deterministic `processing_recipe.json` |
| **Processed Spectra** | Bruker `pdata/1/{1r,2rr,3rrr}`, NMRPipe `.ft1`/`.ft2`/`.ft3`, Sparky `.ucsf`, `JCAMP-DX` (`.jdx`) | NMRPipe `.ft2`/`.ft3`, Bruker `pdata`, `.ucsf`, `JCAMP-DX` |
| **Peaks & Chemical Shifts** | 5- or 7-column Peak `TSV`, `NMR-STAR v3.2` (`.str`), NMR Exchange Format (`NEF v1.1`, `.nef`) | `NMR-STAR v3.2` (`_Atom_chem_shift` with IUPAC `Ambiguity_code`), lossless `NEF v1.1` session files, 7-column Peak `TSV` |
| **3D Structure, AlphaFold & Restraints** | `.pdb`, `.cif` / `.mmcif` (`_ma_qa_metric_local`), `PAE` `JSON` (`AFDB`, `ColabFold`, `AF3`), multi-medium `RDC` `TSV`/`NEF`, `PCS`/`PRE`/`NOESY` `TSV`, `CEST`/`CPMG` exchange `TSV` | Multi-model `PDB`/`mmCIF` ensembles, homolog-firewalled torsion & shift `STR`/`NEF`/`TSV`/`JSON` (`exclusion_list_sha256`), `AlphaFold 3` `JSON` (`version: 2`), `Boltz-2` `YAML` (`version: 1`), excited-state `BIC` `JSON`/`TSV`, self-contained `HTML+SVG` reports |
| **2D Chemistry & 1D NMR** | `SMILES`, Reaction `SMILES`, `InChI`, `MOL` V2000/V3000, `SDF`, `CDXML`, binary `CDX`, `.oct`, IUPAC systematic names | `SVG`, `PDF 1.4`, antialiased `PNG`, `SDF` (`NMR_ASSIGNMENT`), `CDXML`/`CDX`, `.oct`, and `ACS`/`JOC`/`RSC`/`Nature` multiplet & `qNMR` `JSON`/`TSV` |
| **1D Metabolomics Cohorts** | `cohort_manifest.tsv` (`sample_id`, `group`, `spectrum_path`) referencing `.jdx`, Bruker `1r`, NMRPipe `.ft1`, or `.tsv`/`.csv` 1D traces | `AI-Binning` / `PQN` bucket table `CSV` + `study.json`, `PCA` Hotelling $T^2$/$Q$ `JSON`, `OPLS-DA` `VIP`/`S-plot`/`Volcano` `CSV`+`JSON`, and `STOCSY` connectivity `JSON`/`CSV` |
