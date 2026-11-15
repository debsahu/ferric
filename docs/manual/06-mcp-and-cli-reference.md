# Chapter 6: MCP Integration, CLI & Keyboard Reference

This chapter serves as the complete reference for the `nmrassign` command-line interface, the clean-room **Model Context Protocol (`MCP`)** server and authenticated desktop workbench bridge (`nmr-core::mcp`, `nmr-gui::mcp_bridge`), and interactive keyboard accelerators (`nmr-gui::keymap`).

---

## `nmrassign` CLI Subcommand Reference

Every `nmrassign` subcommand runs deterministically on macOS, Linux, and Windows (`--version` / `-V` prints the semantic version and embedded short Git SHA). Engine flags accept `--engine gpu` (WebGPU: Metal/Vulkan/DX12), `--engine cpu` (`f64` multi-threaded CPU), or `--engine auto` (hardware WebGPU with graceful CPU fallback).

### Summary of Subcommands

| Category | Subcommand | Primary Function |
| :--- | :--- | :--- |
| **Discovery & DSP** | `inspect` | Scan spectrometer folders or `.tar`/`.tgz`/`.zip` archives (`< 50 ms`) and classify experiments & regimes |
| **Discovery & DSP** | `process` | Deterministic 2D/3D Bruker/Varian FID processing, `IST`/`FISTA` NUS reconstruction & `processing_recipe.json` export |
| **Peak Picking** | `pick` | Classical MAD/parabolic or 2D/3D separable CNN (`--method neural`) peak picking + Pass-2 `StripBank3D` weak-peak rescue |
| **Assignment** | `assign` | Sequence-only backbone assignment (`globular`, `idp-13c`, `paramagnetic`) across `evo-graph`, `standard`, and opt-in experimental `bayes-sa` |
| **Assignment** | `sidechain` | Joint aliphatic, aromatic (`--aromatic --ph`), and Bayesian stereospecific prochiral (`--stereospec`) assignment |
| **Assignment** | `methyl` | Supramolecular `13C`-Methyl TROSY (`ILVMAT`, `ILV-proS`/`proR`) structure-based GPU contact-graph assignment |
| **Ensemble & Structure** | `propensity` | IDP Secondary Structure Propensity (`SSP`) and 4-state (`alpha`, `beta`, `PPII`, `coil`) HMM populations |
| **Ensemble & Structure** | `torsions` | Strictly post-assignment 3/9-mer `(phi, psi[, chi1])` prediction under the 4-layer `HomologFirewall` |
| **Ensemble & Structure** | `structure` | 7-cycle iterative ambiguous NOESY ADR + torsion-angle NeRF + multi-medium Saupe `RDC` CPU/WebGPU ensemble folding |
| **Interchange & Reports** | `pipeline` | End-to-end single-command workflow (`FIDs/spectra/peaks + seq -> shifts.str + result.nef + report.html`) |
| **Interchange & Reports** | `report` | Standalone `HTML+SVG` quality report (`CSI`, strip ladders, Ramachandran, `RDC` $Q$/$Q_{\text{free}}$, IDP propensities) |
| **Interchange & Reports** | `export-fig` | Deterministic publication-ready SVG figure showcase generator across all 8 workbench modalities |
| **Interchange & Reports** | `nef-import` / `nef-export` | Lossless conversion between `NEF v1.1` (`.nef`), `NMR-STAR v3.2` (`.str`), and 7-column Peak `TSV` |
| **Dynamics & Exchange** | `relax` / `modelfree` / `cest` / `cpmg` / `csp` | Batched GPU $R_1$/$R_2$/hetNOE, Lipari–Szabo `M1`–`M5` $S^2$, Bloch–McConnell `CEST`, Baldwin `CPMG`, and tight-binding `CSP` $K_d$ |
| **1D, DOSY & Chemometrics** | `nmr1d` / `dosy` / `metabolomics` / `chem` | Analytical 1D (`process`/`multiplet`/`qnmr`), pseudo-2D `DOSY` (`fit`/`ilt`/`mw`), untargeted 1D `metabolomics` (`bin`/`pca`/`oplsda`/`stocsy`), and `Octet` 2D chemical studio |
| **Services & AI** | `serve` / `mcp` / `mcp-config` | Self-hosted loopback REST job dashboard (`serve`) and sandboxed JSON-RPC 2.0 `MCP` server & client config generator |
| **Benchmarks & Data** | `fetch-bmrb` / `priors` / `simulate` / `score` / `benchmark` / `benchmark-suite` / `real-benchmark` / `e2e-benchmark` | Rate-limited CC0 BMRB archive downloader, prior builder, volumetric physics simulator, permutation-invariant scorer, and CI benchmark suites |

### Core Subcommand Syntax & Examples

#### 1. `inspect`, `process` & `pick`

```bash
# Classify all nested Bruker / Varian experiments as human-readable table or JSON
nmrassign inspect ./data/study_dir [--json]

# Process uniform or NUS 2D/3D raw FIDs into .ft2 / .ft3 or Bruker pdata
nmrassign process \
  --in ./data/study_dir/3 \
  --out ./out/hncacb.ft3 \
  --exp HNCACB \
  [--recipe custom_recipe.json] [--crop-h 5.5:11.5] [--no-crop] \
  [--lp] [--auto-phase] [--phase-x 0:0] [--hilbert] [--nus-method fista|ist]

# Pick peaks from processed spectra OR directly from raw FID directories
nmrassign pick \
  --fid-hsqc ./data/study_dir/2 \
  --fid-triple HNCACB=./data/study_dir/3 \
  --fid-triple CBCACONH=./data/study_dir/4 \
  --method classical \
  --snr 5.0 \
  --out ./out/peaks.tsv
```

#### 2. `assign`, `sidechain`, `methyl` & `propensity`

```bash
# Multi-regime backbone assignment (globular | idp-13c | paramagnetic)
nmrassign assign \
  --peaks ./out/peaks.tsv \
  --seq ./data/protein.fasta \
  --out ./out/assigned \
  [--regime globular|idp-13c|paramagnetic] \
  [--solver evo-graph|standard|bayes-sa] \
  [--engine auto|gpu|cpu] [--runs 16] [--steps 1500] \
  [--locks "12:ALA15,27:SER42"] [--no-auto-ref] \
  [--nef-out ./out/session.nef] [--report-out ./out/report.html] [--json-out ./out/qc.json]

# Aliphatic, aromatic, and stereospecific prochiral sidechain assignment
nmrassign sidechain \
  --star ./out/assigned.str \
  --peaks ./out/sidechain_peaks.tsv \
  [--aromatic] [--ph 6.8] [--his-hmqc ./data/his.ft2] \
  [--hbcbgcdhd ./data/hd.ft2] [--hbcbgcdce ./data/he.ft2] \
  [--stereospec] [--j-couplings ./data/3j.tsv] [--intra-noes ./data/intra_noe.tsv] \
  --out ./out/all_atom.str

# Supramolecular 13C-Methyl TROSY contact-graph assignment
nmrassign methyl \
  --pdb ./data/template.pdb \
  --hmqc ./data/methyl_hmqc.ft2 \
  --noesy ./data/methyl_noesy_3d.ft3 \
  --labeling ILV-proS \
  --engine gpu \
  -o ./out/methyl_assigned.str

# IDP Secondary Structure Propensity (SSP + 4-state delta2D populations)
nmrassign propensity \
  --shifts ./out/idp_assigned.str \
  --temp-k 298.15 --ph 7.0 --format tsv \
  -o ./out/propensity.tsv
```

#### 3. `torsions`, `structure`, `report`, `nef-import` / `nef-export` & `pipeline`

```bash
# Post-assignment 3/9-mer torsion prediction (chi1 omitted unless --include-chi1)
nmrassign torsions \
  --shifts ./out/all_atom.str \
  -o ./out/torsions.tsv \
  [--include-chi1] [--exclude-pdb 1UBQ,1D3Z] [--exclude-bmrb 17769]

# 3D NOESY + torsion + multi-medium Saupe RDC folding (default 20% RdcFree holdout)
nmrassign structure \
  --star ./out/all_atom.str \
  --noesy-peaks ./data/noesy_3d.tsv \
  --rdc ./data/rdcs.tsv \
  --torsion-db \
  --rdc-weight 0.75 --rdc-free-frac 0.20 \
  --conformers 20 --engine auto \
  --out-pdb ./out/ensemble.pdb

# Generate standalone HTML+SVG quality report
nmrassign report \
  --shifts ./out/all_atom.str \
  --peaks ./out/peaks.tsv \
  --pdb ./out/ensemble.pdb \
  --rdc ./data/rdcs.tsv \
  --out ./out/quality_report.html

# Single-command end-to-end workflow (raw FIDs/spectra/peaks + sequence -> STAR + NEF + HTML report)
nmrassign pipeline \
  --seq ./data/protein.fasta \
  --fid-hsqc ./data/study_dir/2 \
  --fid-triple HNCACB=./data/study_dir/3 \
  --fid-triple CBCACONH=./data/study_dir/4 \
  --out-dir ./out/pipeline_run \
  [--solver evo-graph] [--engine auto] [--fold]

# Lossless NEF v1.1 round-trip interchange
nmrassign nef-export --star ./out/all_atom.str --peaks ./out/peaks.tsv --out ./out/project.nef
nmrassign nef-import --in ./out/project.nef --out-star ./out/reloaded.str --out-peaks ./out/reloaded.tsv
```

#### 4. Dynamics, Chemical Exchange, 1D `qNMR`, `DOSY` & Metabolomics

See [Chapter 4: Relaxation Dynamics, CEST, CPMG & CSP](04-dynamics-and-exchange.md), [Chapter 5: 1D Multiplets, qNMR & DOSY](05-nmr1d-multiplets-and-dosy.md), and [Chapter 8: Untargeted 1D Metabolomics & Chemometrics](08-metabolomics-and-chemometrics.md) for physical models and validation-scope notes:

```bash
# Batched GPU R1, R2, R1rho or steady-state {1H}-15N hetNOE curve fitting + reduced J(omega)
nmrassign relax \
  --series ./data/ubq_r1_600 \
  --kind r1 --field-mhz 600.0 --mc-trials 500 --engine gpu \
  --out ./out/r1_600.tsv

# Lipari-Szabo / Mandel-Akke-Palmer M1-M5 model-free & anisotropic diffusion tensor fit
nmrassign modelfree \
  --r1 ./out/r1_600.tsv --r2 ./out/r2_600.tsv --hetnoe ./out/noe_600.tsv \
  --r1 ./out/r1_800.tsv --r2 ./out/r2_800.tsv --hetnoe ./out/noe_800.tsv \
  --pdb ./data/template.pdb --diffusion auto --criterion aicc \
  --out ./out/modelfree.tsv

# Homogeneous Padé [6/6] Bloch-McConnell 15N/13C CEST & DEST excited-state fitting
nmrassign cest \
  --in ./data/cest_15hz.tsv --in ./data/cest_30hz.tsv \
  --states 2 --b1-quad 5 --engine gpu \
  --out ./out/cest_fit.tsv

# Exact Baldwin (2014) / Carver-Richards multi-field CPMG relaxation dispersion
nmrassign cpmg \
  --in ./data/cpmg_600.tsv --in ./data/cpmg_800.tsv \
  --model baldwin2014 --engine gpu \
  --out ./out/cpmg_fit.tsv

# Fast-exchange HSQC/HMQC ligand titration trajectory tracking & exact quadratic tight-binding Kd
nmrassign csp \
  --apo-peaks ./data/apo_hsqc.tsv \
  --step 20=./data/hsqc_20um.tsv --step 50=./data/hsqc_50um.tsv \
  --step 100=./data/hsqc_100um.tsv --step 250=./data/hsqc_250um.tsv \
  --protein-um 100.0 --experiment hsqc \
  --out ./out/csp_kd.tsv

# Analytical 1D DSP, Voigt/Hoye J-trees, journal export & NIST-traceable GUM qNMR purity
nmrassign nmr1d process --in ./data/compound_1h/1 --solvent CDCl3 --hsvd --deconvolve --out ./out/spec1d.jdx
nmrassign nmr1d multiplet --in ./out/spec1d.jdx --solvent CDCl3 --style acs --out ./out/multiplets.txt
nmrassign nmr1d qnmr --in ./out/spec1d.jdx \
  --std-id DMSO2 --std-mass-mg 5.12 --std-purity 0.9994 \
  --analyte-mass-mg 12.40 --analyte-mw 180.16 --analyte-protons 3 \
  --analyte-ppm 2.28:2.42 --out ./out/qnmr_report.json

# Pseudo-2D Stejskal-Tanner / Wu-Johnson BPP-LED DOSY (default BIC fit; opt-in Tikhonov ILT & MW)
nmrassign dosy fit --in ./data/mixture_dosy/10 --criterion bic --solvent CDCl3 --temp-k 298.15 --engine auto --out ./out/dosy_bic.tsv
nmrassign dosy ilt --in ./data/mixture_dosy/10 --strategy gcv --out ./out/dosy_ilt.tsv
nmrassign dosy mw --d 7.8e-10 --solvent CDCl3 --temp-k 298.15

# Untargeted 1D Metabolomics: AI-Binning + PQN, PCA Hotelling T^2/Q, k-fold OPLS-DA & STOCSY
nmrassign metabolomics bin --manifest cohort_manifest.tsv --out cohort_matrix.tsv --method aib --norm pqn --align-ppm 0.025
nmrassign metabolomics pca --matrix cohort_matrix.tsv --out pca_report.json --components 3 --scale pareto
nmrassign metabolomics oplsda --matrix cohort_matrix.tsv --out oplsda_biomarkers.tsv --ortho 1 --scale pareto --cv-folds 7 --permutations 200
nmrassign metabolomics stocsy --matrix cohort_matrix.tsv --driver-ppm 1.33 --out stocsy_1p33ppm.tsv
```

#### 5. `chem`, `export-fig`, `serve`, `mcp`, `mcp-config` & Reproducible Benchmark Suites

```bash
# Print version & embedded Git SHA
nmrassign --version

# 2D chemical rendering, IUPAC <-> Structure, and 1D JCAMP-DX synthesis
nmrassign chem --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --solvent CDCl3 --mhz 600 --out caffeine.svg

# Deterministic publication-ready SVG figure showcase across all 8 workbench modalities
nmrassign export-fig --out-dir docs/assets [--panel all|hsqc|strips|pdb-s2|nmr1d-stack|dosy|metabolomics|exchange|octet]

# Self-hosted loopback HTTP REST job service & browser dashboard
nmrassign serve --bind 127.0.0.1:8080 --storage ./jobs

# Sandboxed JSON-RPC 2.0 MCP server (default --mode read-only) & client config generator
nmrassign mcp --transport stdio --root ./project_dir --mode read-only
nmrassign mcp --transport http --bind 127.0.0.1:54912 --token <session-token> --root ./project_dir --mode read-only
nmrassign mcp-config --client antigravity --root ./project_dir --mode read-only [--bridge-port 54912]

# Rate-limited (<= 1 req/s) CC0 BMRB downloader writing SHA-256 PROVENANCE.md
nmrassign fetch-bmrb --id 17769 --dest ./data/private/bmrb/17769

# Recompute amino-acid chemical shift priors JSON & run realistic Bruker 3D physics simulation
nmrassign priors --stars ./data/reference_stars --out ./out/custom_priors.json
nmrassign simulate --star data/fixtures/synthetic_ubiquitin.str --realistic --spectra-dir ./out/sim_cubes --out ./out/sim_peaks.tsv --seed 0

# Permutation-invariant shift scoring, 5-seed realistic gate, multi-protein suite, E2E CNN gate & real-FID evaluation
nmrassign score --pred ./out/assigned.str --truth ./data/ref.str --hc 0.80
nmrassign benchmark --star data/fixtures/synthetic_ubiquitin.str --realistic --seeds 5
nmrassign benchmark-suite --manifest data/benchmark/manifest.tsv --solver evo-graph --seed 0
nmrassign e2e-benchmark --star data/fixtures/synthetic_ubiquitin.str --noise 1.0 --seed 42
nmrassign real-benchmark --manifest data/benchmark/real_manifest.tsv --split dev --solver evo-graph --seeds 5
```

---

## Model Context Protocol (`MCP`) Server & Live Desktop Bridge

`nmr-assign` implements the open **Model Context Protocol (`MCP`, JSON-RPC 2.0)** specification in pure safe Rust (`serde`, `serde_json`, `std::io`, `std::net`—zero external SDK or image-encoding crates) across two complementary transports:

> [!IMPORTANT] MCP Validation Scope (In-Process / Subprocess Protocol Self-Consistency)
> JSON-RPC 2.0 framing (`stdio` newline & `Content-Length` pre-allocation caps), canonicalized path sandboxing, loopback HTTP bearer-token / DNS-rebinding guards, `256`-step Undo curation reversibility, and RFC 2083 viewport `PNG` encoding are verified via deterministic Rust unit tests and `nmrassign mcp` CLI subprocess E2E tests (`0` live interactive GUI sessions against external Antigravity, Claude Desktop, or Cursor binaries are executed in automated CI; end-to-end multi-client host smoke testing is tracked openly under follow-up **`F18`**).

| Transport Mode | Implementation | Ideal Use Case |
| :--- | :--- | :--- |
| **Headless `stdio` Server** (`StdioCli`) | `nmr-core::mcp` (`run_mcp_stdio_loop`) | Autonomous workspace inspection, QC triage, peak/NMR-STAR analysis, and homolog-firewalled torsion prediction inside CLI/IDE coding agents. Auto-detects both newline-delimited JSON-RPC and `Content-Length` framing (`<= 16 MiB` frame cap). |
| **Live Desktop HTTP Bridge** (`HttpBridge`) | `nmr-gui::mcp_bridge` (`127.0.0.1:<port>`) | Interactive human-in-the-loop spectroscopist pairing with the running `nmr-gui` desktop workbench: inspect active strip crosshairs, navigate to ambiguous residues, propose **Undo-safe (`Cmd/Ctrl+Z`)** curation edits, and stream live `image/png` viewport snapshots. |

---

### Setting Up Antigravity, Claude Desktop & Cursor

Client JSON configurations can be generated programmatically via `generate_mcp_client_config` (`nmrassign mcp-config`) or copied in one click from the **`🔌 MCP Bridge`** panel inside `nmr-gui`.

#### 1. Headless `stdio` Mode (`ReadOnly` Default)

Add the following block to your client's `mcpServers` configuration file (for **Antigravity** `mcp_config.json`, **Claude Desktop** `claude_desktop_config.json`, or **Cursor** `.cursor/mcp.json`), substituting your binary and project directory:

```json
{
  "mcpServers": {
    "nmr-assign": {
      "command": "/opt/nmr-assign/bin/nmrassign",
      "args": [
        "mcp",
        "--transport",
        "stdio",
        "--root",
        "/path/to/nmr_workspace",
        "--mode",
        "read-only"
      ]
    }
  }
}
```

*(To allow curated peak-list or NMR-STAR file export strictly inside `/path/to/nmr_workspace`, replace `"read-only"` with `"read-write"`.)*

#### 2. Live `nmr-gui` Desktop Loopback Bridge Mode (`HttpBridge`)

1. Open `nmr-gui` on your desktop, navigate to the **`🔌 MCP Bridge`** panel, choose **`ReadOnly`** (default) or **`ReadWrite`**, and click **Start Local Bridge**.
2. `nmr-gui` binds strictly to `127.0.0.1:<port>` (ephemeral loopback port by default) and generates a fresh random session bearer token.
3. Click **Copy MCP Config** for **Antigravity**, **Claude Desktop**, or **Cursor**, which produces:

```json
{
  "mcpServers": {
    "nmrassign-workbench": {
      "url": "http://127.0.0.1:54912/mcp",
      "headers": {
        "Authorization": "Bearer <session-bearer-token>"
      }
    }
  }
}
```

---

### `ReadOnly` / `ReadWrite` Permission & Sandboxing Guardrails

Every `MCP` request passes through defense-in-depth security checks (`McpSecurityPolicy`):

1. **Default-Safe `ReadOnly` Mode**: Both the `stdio` server and the `nmr-gui` bridge default to **`McpAccessMode::ReadOnly`**. State-modifying tools (`write_curated_peak_tsv`, `export_nmrstar_shifts`, `navigate_workbench`, `propose_curation_edit`, `undo_last_ai_edit`) carry MCP safety annotations (`readOnlyHint: false`, `destructiveHint: true`) and are unconditionally rejected (`-32002 ERR_READ_ONLY_DENIED` / `isError: true`) unless the user explicitly enables `ReadWrite`.
2. **Canonicalized Path Sandbox (`resolve_sandboxed_path`)**:
   - Rejects empty paths, embedded null bytes (`\0`), and any lexical `..` (`ParentDir`) traversal before touching the filesystem.
   - Blocks access to hidden credential and secret locations (`.ssh`, `.aws`, `.gnupg`, `.env`, `.env.*`, `.netrc`, `.kube`, `id_rsa`, `id_ed25519`).
   - Canonicalizes every configured `allowed_root` and target path via `std::fs::canonicalize`, blocking symlinks or dangling links escaping `--root`, and formats responses as workspace-relative paths so host user home directories are never leaked.
3. **Loopback-Only Binding, Constant-Time Auth & DNS-Rebinding Defense**:
   - The desktop HTTP bridge refuses non-loopback bind addresses (`0.0.0.0`, LAN IPs, `[::]`).
   - Every HTTP request must carry `Authorization: Bearer <token>` verified via non-branching constant-time byte comparison (`401 Unauthorized`), a loopback `Host` header (`127.0.0.1` or `localhost`), and **zero** cross-origin browser `Origin` headers (`403 Forbidden`), defeating local browser CSRF and DNS-rebinding attacks.
4. **100% Reversible AI Curation (`256`-Step `McpUndoLog`)**:
   - In `nmr-gui`, AI curation commands (`AssignSpinSystemToResidue`, `UnassignResidue`, `TogglePeakExclusion`, `JumpToResidue`, `SetCrosshairPpm`, `HighlightResidues3d`) execute exclusively through `McpUndoLog` (`MAX_MCP_UNDO_DEPTH = 256`), preserving bijective `(cluster_id <-> seq_id)` displacements so pressing **`Cmd/Ctrl+Z`** (or calling `undo_last_ai_edit`) restores the exact prior state in one step.

### Built-In `MCP` Tools, Resources, Prompts & Error Codes

#### Workspace `stdio` Tools (`nmr-core::mcp`) & Desktop Bridge Tools (`nmr-gui::mcp_bridge`)

| Tool Name | Surface | Mode | Description |
| :--- | :---: | :---: | :--- |
| `inspect_nmr_directory` | `stdio` | `ReadOnly` | Classify nested Bruker/Varian folders and decode `pulprog` physics |
| `read_nmrstar_summary` | `stdio` | `ReadOnly` | Parse sequence, backbone/sidechain chemical shift counts & ambiguity codes |
| `read_peak_list_summary` | `stdio` | `ReadOnly` | Summarize picked peak counts per experiment and assembled `SpinSystem` roots |
| `evaluate_dataset_qc` | `stdio` | `ReadOnly` | Run `DatasetQcReport` completeness, aliphatic corroboration & referencing checks |
| `predict_homolog_free_torsions` | `stdio` | `ReadOnly` | Predict 3/9-mer $(\phi, \psi)$ restraints under the `<= 25%` 4-layer `HomologFirewall` |
| `fit_relaxation_or_exchange` | `stdio` | `ReadOnly` | Fit $R_1$/$R_2$/hetNOE, Lipari–Szabo `M1`–`M5` $S^2$, Bloch–McConnell `CEST`, or Baldwin `CPMG` |
| `track_csp_titration` | `stdio` | `ReadOnly` | Track 2D `HSQC`/`HMQC` ligand titration trajectories and fit quadratic tight-binding $K_d$ |
| `analyze_nmr1d_spectrum` | `stdio` | `ReadOnly` | Run 1D DSP, Fulmer solvent referencing, Voigt/Hoye `J`-trees, journal formatting & GUM `qNMR` |
| `fit_dosy_diffusion` | `stdio` | `ReadOnly` | Fit discrete `BIC` Stejskal–Tanner / `BPP-LED` `DOSY` decays, opt-in `ILT`, and Stokes–Einstein `MW` |
| `run_metabolomics_chemometrics` | `stdio` | `ReadOnly` | Run sandboxed `AI-Binning`, `PQN`, `PCA` Hotelling $T^2$/$Q$, leak-free `OPLS-DA`, and `STOCSY` |
| `write_curated_peak_tsv` | `stdio` | `ReadWrite` | Write filtered/curated 7-column peak `TSV` strictly inside `allowed_roots` |
| `export_nmrstar_shifts` | `stdio` | `ReadWrite` | Write `NMR-STAR v3.2` chemical shifts strictly inside `allowed_roots` |
| `get_workbench_state` | `gui` | `ReadOnly` | Snapshot active residue, `(1H, 15N, 13C)` crosshair, ranked strip candidates, undo depth & QC |
| `navigate_workbench` | `gui` | `ReadWrite` | Jump strips/crosshairs to a residue or highlight 3D PDB pocket residues (Undo-logged) |
| `capture_viewport_png` | `gui` | `ReadOnly` | Encode active 2D contour / 3D strip / 3D ribbon viewport as RFC 2083 `image/png` Base64 |
| `propose_curation_edit` | `gui` | `ReadWrite` | Apply an Undo-logged spin-system assignment, unassignment, or peak exclusion |
| `undo_last_ai_edit` | `gui` | `ReadWrite` | Revert the most recent AI curation action from `McpUndoLog` |

#### Built-In `nmr://` Resources & Guided Spectroscopist Prompts

- **Resources (`resources/list`, `resources/read`)**: `nmr://workspace/policy`, `nmr://workspace/inventory`, `nmr://catalog/experiments`, and `nmr://file/<root_index>/<relative_path>`.
- **Prompts (`prompts/list`, `prompts/get`)**:
  - `triage-qc`: Step-by-step audit of unlabelled referencing offsets, `DatasetQcReport` gates, and unseen-protein confidence caveats.
  - `resolve-ambiguous-strips`: Multi-pathway `(i, i-1)` strip ladder disambiguation and pre-Proline shift checks.
  - `deconvolve-1d-mixture-dosy`: Fulmer referencing, Voigt/Hoye multiplet trees, `qNMR` stoichiometry, and `BIC` `DOSY` vs opt-in `ILT`.
  - `fit-cest-cpmg-exchange`: Locked-shape plane integration, flat-null gate, multi-field scaling, and Bloch–McConnell `CEST` / Baldwin `CPMG`.
  - `analyze-metabolomics-cohort`: Step-by-step `AI-Binning` + `PQN`, `PCA` Hotelling $T^2$/$Q$ outlier check, leak-free `k`-fold `OPLS-DA` ($Q^2$ & permutation $p$-value overfitting guards), and `STOCSY` spin-system annotation.

#### JSON-RPC 2.0 & MCP Error Codes

| Code | Constant | Meaning |
| :---: | :--- | :--- |
| `-32700` | `ERR_PARSE_ERROR` | Malformed UTF-8 or invalid JSON syntax |
| `-32600` | `ERR_INVALID_REQUEST` | Missing `"jsonrpc": "2.0"`, invalid `"method"`/`"id"`, or empty batch |
| `-32601` | `ERR_METHOD_NOT_FOUND` | Unknown JSON-RPC method |
| `-32602` | `ERR_INVALID_PARAMS` | Unknown tool/prompt name or invalid parameter schema |
| `-32603` | `ERR_INTERNAL_ERROR` | Internal serialization or I/O error |
| `-32001` | `ERR_SECURITY_VIOLATION` | Sandbox escape (`..`, escaping symlink, secret file, outside `allowed_roots`) |
| `-32002` | `ERR_READ_ONLY_DENIED` | Mutating tool called while `McpAccessMode::ReadOnly` is active |
| `-32003` | `ERR_RESOURCE_NOT_FOUND` | Unknown or missing `nmr://` resource URI |

---

## Interactive Workbench Keyboard Shortcuts

`nmr-gui` features a deterministic finite-state two-key NMR chord dispatcher (`450 ms` inter-key timeout; press `Esc` to cancel a pending prefix). Single- and two-key chords automatically stand down whenever a text input field or the `Octet` 2D Chemical Studio canvas holds focus.

### Two-Key & Single-Key NMR Spectrum Accelerators

| Keys | Category | Action | Description |
| :---: | :--- | :--- | :--- |
| **`vt`** | Spectrum & Contours | View / Contour Threshold Popup | Open interactive base contour floor, geometric step, and level count popover |
| **`ct`** | Spectrum & Contours | Raise Contour Floor (`×1.25`) | Multiply positive/negative contour floor by `1.25×` to suppress baseline noise |
| **`cb`** | Spectrum & Contours | Lower Contour Floor (`÷1.25`) | Divide positive/negative contour floor by `1.25×` to reveal weak cross-peaks |
| **`oz`** | Zoom & Navigation | Zoom Out to Full Sweep | Reset 2D viewport bounds to the full spectral sweep width |
| **`zi`** | Zoom & Navigation | Zoom In Viewport | Zoom into the active 2D or strip spectral region by `1.25×` |
| **`zo`** | Zoom & Navigation | Zoom Out Viewport | Zoom out of the active 2D or strip spectral region by `1.25×` |
| **`Space`** | Zoom & Navigation | Toggle Crosshair Lock | Pin or unpin synchronized `(1H, 15N, 13C)` frequency crosshairs across views |
| **`pa`** | Peak Editing | Add Peak at Crosshair | Insert a new cross-peak at the active cursor or locked crosshair |
| **`pd`** | Peak Editing | Delete Selected Peak(s) | Delete selected cross-peak(s) (reversible via `Cmd/Ctrl+Z`) |
| **`pc`** | Peak Editing | Parabolic Peak Center Snap | Snap selected peak(s) to the local 3-point log-parabolic extremum |
| **`pl`** | Peak Editing | Toggle Peak Labels | Show or hide spin-system IDs and residue labels on the contour canvas |
| **`st`** | Strips & Assignment | Tile `i-1 / i / i+1` Strips | Tile ranked predecessor (`i-1`), active (`i`), and successor (`i+1`) 3D strips |
| **`at`** | Strips & Assignment | Residue Assignment Popover | Open `U(s, k)` typing scores and hard-lock assignment popover for active root |
| **`?`** | Help & Reference | Keyboard Cheat-Sheet Modal | Open searchable reference modal of all chords and global shortcuts |

### Global Modifier Shortcuts

| Shortcut | Category | Action |
| :---: | :--- | :--- |
| **`Cmd/Ctrl+K`** | Navigation | Open fuzzy command palette across all `50+` workbench actions |
| **`Cmd/Ctrl+Z`** | Session & Undo | Undo most recent peak, assignment, lock, contour, or AI bridge edit (`256` steps) |
| **`Cmd/Ctrl+Shift+Z`** | Session & Undo | Redo most recently undone workbench command |
| **`Cmd/Ctrl+O`** | File & Session | Open spectrometer folder / archive Experiment Selection Matrix |
| **`Cmd/Ctrl+S`** | File & Session | Save lossless `NEF v1.1` workbench session |
| **`Cmd/Ctrl+Shift+S`** | File & Session | Save lossless `NEF v1.1` session to a new path |
| **`Cmd/Ctrl+E`** | File & Session | Export assigned chemical shifts (`NMR-STAR v3.2` `.str`) |
| **`Cmd/Ctrl+Enter`** | Assignment | Run automated backbone resonance assignment |
