# 21. AlphaFold, Secondary Structure & Integrative Structural NMR (`T21.1`–`T21.6b`)

> **Validation Scope & Scientific Governance (`AGENTS.md`)**:
> 1. **Sequence-Only Backbone Assignment Preserved (`docs/specs/03-assignment.md` Untouched)**: Default `ferric assign` (`--solver standard|evo-graph|bayes-sa`) remains strictly sequence-only with zero 3D coordinate inputs. AlphaFold 3D shift and `PAE`-gated `15N-NOESY` priors (`T21.4`) are **strictly opt-in** via `--af-structure <model.cif>` and leave `docs/specs/03-assignment.md` completely unmodified.
> 2. **Strict Non-Homologous `HomologFirewall` (`<= 25%` Sequence Identity, `TM-score < 0.50`)**: All empirical $(\phi_i, \psi_i, \phi_{i-1}, \psi_{i-1}, \chi_1)$ Fourier/von Mises shift hypersurfaces in `T21.2` (`nmr-core::struct_shifts`) are fitted exclusively from FERRIC's clean-room CC0 `data/torsion_db/torsion_corpus.tsv` (`129` chains, `13,647` residues; `SHA-256 = e34d3001...`) with mandatory 4-layer `HomologFirewall` exclusion (`L1` PDB/BMRB/UniProt, `L2` CATH/ECOD topology, `L3` NW+SW+7-mer `<= 25%` sequence identity, `L4` `TM-score < 0.50`) and cryptographic `exclusion_list_sha256` audit tracking during every benchmark run.
> 3. **Single- vs Multi-Medium Dipolar Candor (`F8` Carried Forward)**: In `T21.5` (`domain_dock`), a single alignment medium's `RDC` tensor carries intrinsic 4-fold principal-axis $\pi$-rotation cone degeneracy and zero translational sensitivity; unambiguous 3D multi-domain rigid-body placement requires either $\ge 2$ non-coaxial `RDC` alignment media paired with covalent linker / steric constraints, or translational-plus-orientational paramagnetic `PCS` ($\Delta\boldsymbol{\chi}$, $r_{\text{M}}^{-3}$) / `PRE` ($\Gamma_2$, $r_{\text{M}}^{-6}$) / inter-domain `NOE` restraints, always cross-validated under `ByPeptidePlane` `20%` `RdcFree` holdout.

---

## Provenance & Clean-Room Statement

Original clean-room technical specification and mathematical architecture for Phase 21 (`T21.1`–`T21.6b`). Zero source code, binaries, decompiled output, neural network weights, residue libraries, macro scripts, or parameter tables from any copyleft (`GPL`/`AGPL`), academic-licensed, or proprietary secondary-structure classifier, chemical-shift predictor, structure validation tool, macromolecular docking suite, or NMR refinement package were consulted. No external product names appear in any Rust module, struct, enum, trait, function, or CLI identifier (`kabsch_sander_ss`, `af_concordance`, `struct_shifts`, `af_assign_priors`, `domain_dock`, `cofold_bridge`, `excited_state`). Every physical forward model, graph clustering algorithm, statistical criterion, and file schema is derived strictly from peer-reviewed literature, public open-format specifications, and `CC0` / `CC-BY-4.0` archive standards:

1. **AlphaFold Confidence Metrics (`pLDDT`, `PAE`), ModelArchive mmCIF & Public Database Schemas (`T21.1`)**:
   - Jumper, J., Evans, R., Pritzel, A. et al. (*Nature* **2021**, 596, 583–589) — AlphaFold 2 architecture, per-residue predicted Local Distance Difference Test (`pLDDT` $\in [0, 100]$), and $N \times N$ Predicted Aligned Error (`PAE`, Å).
   - Mariani, V., Biasini, M., Barbato, A. & Schwede, T. (*Bioinformatics* **2013**, 29, 2722–2728) — superposition-free Local Distance Difference Test (`lDDT`).
   - Varadi, M., Anyango, S., Deshpande, M. et al. (*Nucleic Acids Res.* **2022**, 50, D439–D444) & Varadi, M., Bertoni, D., Magana, P. et al. (*Nucleic Acids Res.* **2024**, 52, D368–D375) — AlphaFold Protein Structure Database (`AFDB` v4, `CC-BY-4.0`), ModelArchive `_ma_qa_metric_local` mmCIF quality-assessment records, and `PAE` JSON schema.
   - Tunyasuvunakool, K. et al. (*Nature* **2021**, 596, 590–596) & Alderson, T. R., Pritišanac, I., Kolarić, Đ., Moses, A. M. & Forman-Kay, J. D. (*Proc. Natl. Acad. Sci. U.S.A.* **2023**, 120, e2304302120) — physical interpretation of `pLDDT < 50` disorder vs high-`pLDDT` conditionally folding segments in intrinsically disordered regions.
2. **Electrostatic Hydrogen-Bond Secondary Structure & Solution Rigidity Concordance (`T21.1`, `T21.3`)**:
   - Kabsch, W. & Sander, C. (*Biopolymers* **1983**, 22, 2577–2637) — partial-charge electrostatic $\text{C}'=\text{O}_i \cdots \text{H–N}_j$ hydrogen-bond criterion ($E < -0.5\text{ kcal mol}^{-1}$), $n$-turn helices ($n \in \{3, 4, 5\}$), and $\beta$-bridge / $\beta$-ladder classification.
   - Adzhubei, A. A., Sternberg, M. J. E. & Makarov, A. A. (*J. Mol. Biol.* **2013**, 425, 2100–2132) — polyproline II (`PPII`) left-handed extended helix $(\phi, \psi)$ dihedral definition in non-hydrogen-bonded polypeptide segments.
   - Berjanskii, M. V. & Wishart, D. S. (*J. Am. Chem. Soc.* **2005**, 127, 14970–14971) — Random Coil Index (`RCI`) and chemical-shift order parameter $S^2_{\text{RCI}}$.
   - Fowler, N. J., Sljoka, A. & Williamson, M. P. (*Nat. Commun.* **2020**, 11, 5982) & Fowler, N. J. & Williamson, M. P. (*Structure* **2022**, 30, 925–933) — validation of protein structures and AlphaFold models in solution by comparing local structural rigidity against `RCI` via Spearman rank correlation $\rho_s$ and RMSD-based rigidity offset scoring.
   - Robertson, A. J., Courtney, J. M., Shen, Y., Ying, J. & Bax, A. (*J. Am. Chem. Soc.* **2021**, 143, 19306–19310) & Zweckstetter, M. (*Protein Sci.* **2021**, 30, 2333–2337) — concordance of AlphaFold models with experimental solution NMR dipolar couplings and order parameters.
3. **3D Structure-to-Chemical-Shift Forward Physics & CC0-Fitted Torsion Surfaces (`T21.2`)**:
   - Shen, Y. & Bax, A. (*J. Biomol. NMR* **2010**, 48, 13–22) — additive decomposition of backbone and $^{13}\text{C}^\beta$ chemical shifts into local $(\phi, \psi, \chi_1)$ conformation, hydrogen-bond geometry, aromatic ring-current fields, and peptide-group anisotropy (re-derived from published physical equations and fitted exclusively to FERRIC's `CC0` `torsion_corpus.tsv` with zero external parameter tables).
   - Haigh, C. W. & Mallion, R. B. (*Mol. Phys.* **1972**, 24, 955–970; *Prog. Nucl. Magn. Reson. Spectrosc.* **1979**, 13, 303–344) & Case, D. A. (*J. Biomol. NMR* **1995**, 6, 341–346) — signed triangular-projection aromatic ring-current shielding tensors across `Phe`, `Tyr`, `Trp`, and `His` $\pi$-systems.
   - Wagner, G., Pardi, A. & Bax, A. (*J. Am. Chem. Soc.* **1983**, 105, 5948–5949) & Pardi, A., Wagner, G. & Wüthrich, K. (*Eur. J. Biochem.* **1983**, 137, 445–454) — inverse-cube $r_{\text{HO}}^{-3}$ amide proton and carbonyl $^{13}\text{C}'$ hydrogen-bond chemical shift perturbations.
   - McConnell, H. M. (*J. Chem. Phys.* **1957**, 27, 226–229); Buckingham, A. D. (*Can. J. Chem.* **1960**, 38, 300–307); Sitkoff, D. & Case, D. A. (*Prog. Nucl. Magn. Reson. Spectrosc.* **1998**, 32, 165–190) — peptide carbonyl magnetic susceptibility anisotropy and bond-parallel electric-field shielding $\Delta\delta_{\text{EF}} = -A_\parallel E_\parallel$.
   - Sahakyan, A. B., Vranken, W. F., Cavalli, A. & Vendruscolo, M. (*Angew. Chem. Int. Ed.* **2011**, 50, 9620–9623) — sidechain methyl ($^1\text{H}^{\text{m}}, {}^{13}\text{C}^{\text{m}}$ for `Ile`, `Leu`, `Val`, `Met`, `Ala`, `Thr`) $\gamma$-gauche rotamer and ring-current shift physics.
4. **Heavy-Tailed 3D Assignment Priors & `PAE`-Gated Amide NOE Connectivity (`T21.4`)**:
   - Lange, K. L., Little, R. J. A. & Taylor, J. M. G. (*J. Am. Stat. Assoc.* **1989**, 84, 881–896) — robust statistical modeling using heavy-tailed Student-$t$ likelihoods.
   - Wüthrich, K. (*NMR of Proteins and Nucleic Acids*, Wiley, **1986**) & Grzesiek, S., Wingfield, P., Stahl, S., Kaufman, J. D. & Bax, A. (*J. Biomol. NMR* **1995**, 5, 225–233) — `3D 15N-edited NOESY` $d_{\text{NN}}(i, j) \le 5.5\text{ \AA}$ amide proton connectivity in perdeuterated and sparse-carbon backbone assignment.
5. **Spectral `PAE` Domain Clustering, Generalized Degree of Order & Multi-Domain `RDC`/`PCS`/`PRE` Docking (`T21.5`)**:
   - Ng, A. Y., Jordan, M. I. & Weiss, Y. (*Adv. Neural Inf. Process. Syst.* **2001**, 14, 849–856) & Von Luxburg, U. (*Stat. Comput.* **2007**, 17, 395–416) — normalized symmetric graph Laplacian spectral clustering.
   - Tolman, J. R., Al-Hashimi, H. M., Kay, L. E. & Prestegard, J. H. (*J. Am. Chem. Soc.* **2001**, 123, 1416–1424) — Generalized Degree of Order ($\vartheta = \sqrt{\frac{2}{3}\sum_{\alpha,\beta} S_{\alpha\beta}^2}$) and inter-domain orientational dynamics detection from domain-specific Saupe order tensors.
   - Braddock, D. T., Cai, M., Baber, J. L., Huang, Y. & Clore, G. M. (*J. Am. Chem. Soc.* **2001**, 123, 8634–8635) & Clore, G. M. & Schwieters, C. D. (*J. Am. Chem. Soc.* **2004**, 126, 2923–2938) — rigid-body multi-domain orientation and flexible linker torsion refinement driven by dipolar couplings and long-range paramagnetic relaxation enhancements.
   - Bertini, I., Gupta, Y. K., Luchinat, C., Parigi, G., Peana, M., Sgheri, L. & Yuan, J. (*J. Am. Chem. Soc.* **2007**, 129, 12786–12794) — simultaneous translational and orientational determination of multi-domain protein architectures from lanthanide/metalloprotein $\Delta\boldsymbol{\chi}$ pseudocontact shifts.
6. **`CSP`/`PRE`-Guided Co-Folding Bridge (`AlphaFold 3` / `Boltz-2`) & Multi-Pose Complex Re-Scoring (`T21.6a`)**:
   - Abramson, J., Adler, J., Dunger, J. et al. (*Nature* **2024**, 630, 493–500) — AlphaFold 3 biomolecular complex prediction and public JSON input format specification (`dialect: "alphafold3"`).
   - Wohlwend, J., Corso, G., Passaro, S. et al. (*bioRxiv* **2024**, DOI: `10.1101/2024.11.19.624167`) & Passaro, S., Corso, G., Wohlwend, J. et al. (*bioRxiv* **2025**, DOI: `10.1101/2025.06.14.659707`) — MIT-licensed `Boltz-1` / `Boltz-2` biomolecular interaction & affinity architecture and declarative YAML `pocket` / `contact` restraint specification.
   - Shrake, A. & Rupley, J. A. (*J. Mol. Biol.* **1973**, 79, 351–371) — Fibonacci-sphere solvent-accessible surface area (`SASA`) quadrature separating surface binding-pocket residues from buried core allosteric relays.
   - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* **2013**, 73, 1–16) & McCoy, M. A. & Wyss, D. F. (*J. Am. Chem. Soc.* **2002**, 124, 11758–11763) — quantitative ligand ring-current and hydrogen-bond chemical-shift perturbation pose scoring.
7. **Bayesian Excited-State (`CEST` / `CPMG` / $R_{1\rho}$) Structural Hypothesis Engine (`T21.6b`)**:
   - Vallurupalli, P., Hansen, D. F. & Kay, L. E. (*Proc. Natl. Acad. Sci. U.S.A.* **2008**, 105, 11766–11771); Vallurupalli, P., Sekhar, A., Yuwen, T. & Kay, L. E. (*J. Biomol. NMR* **2017**, 67, 243–271); Bouvignies, G., Vallurupalli, P., Hansen, D. F. et al. (*Nature* **2011**, 477, 111–114) — excited-state chemical shifts $\varpi_B = \varpi_A + \Delta\varpi_{AB}$ from `CPMG` and `CEST` experiments and structural characterization of sparsely populated protein conformers.
   - Skrynnikov, N. R., Dahlquist, F. W. & Kay, L. E. (*J. Am. Chem. Soc.* **2002**, 124, 12352–12360) — signed vs unsigned $|\Delta\varpi_{AB}|$ dispersion likelihoods.
   - Schubert, M., Labudde, D., Oschkinat, H. & Schmieder, P. (*J. Biomol. NMR* **2002**, 24, 149–154) & Shen, Y. & Bax, A. (*J. Biomol. NMR* **2010**, 46, 199–204) — $^{13}\text{C}^\beta - {}^{13}\text{C}^\gamma$ ($\Delta_{\beta\gamma}$) *cis*- vs *trans*-Proline chemical-shift invariants.
   - Pelton, J. G., Torchia, D. A., Meadow, N. D. & Roseman, S. (*Protein Sci.* **1993**, 2, 543–558) — histidine imidazole $^{15}\text{N}^{\delta 1} / {}^{15}\text{N}^{\varepsilon 2}$ tautomer and protonation chemical shifts.
   - Wayment-Steele, H. K., Ojoawo, A., Otten, R. et al. (*Nature* **2024**, 625, 832–839) — subsampled/clustered MSA co-evolutionary conformer ensembles.
   - Schwarz, G. (*Ann. Stat.* **1978**, 6, 461–464) & Kass, R. E. & Raftery, A. E. (*J. Am. Stat. Assoc.* **1995**, 90, 773–795) — Bayesian Information Criterion (`BIC`) and Bayes factor ($2\ln\mathrm{BF}$) hypothesis testing.

---

## Zero-Collision Three-Wave Execution Architecture (`21.W1`–`21.W3`)

Phase 21 splits into **three sequential waves** where every parallel worker within a wave owns **strictly disjoint source and test files** (`workspace:"share"`). Single-line `pub mod` exports in `crates/nmr-core/src/lib.rs`, `crates/nmr-gpu/src/lib.rs`, and top-level CLI subcommand arms in `crates/nmr-cli/src/main.rs` are isolated into per-task `*_cmd.rs` modules and reconciled cleanly by `@merge-shepherd` at wave barrier merge:

| Wave | Task ID | Crate & Module Scope | Strictly Disjoint Owned Files | Forbidden Touch Surface (`0`-Byte Diff) |
| :--- | :--- | :--- | :--- | :--- |
| **`21.W1`** | `T21.1` | `nmr-core::{kabsch_sander_ss, af_concordance}`, `nmr-cli::af_validate_cmd` | `crates/nmr-core/src/kabsch_sander_ss.rs`, `crates/nmr-core/src/af_concordance.rs`, `crates/nmr-core/src/report.rs` (AF concordance SVG section), `crates/nmr-core/tests/af_concordance_tests.rs`, `crates/nmr-cli/src/af_validate_cmd.rs`, `crates/nmr-cli/tests/af_validate_cli_tests.rs` | `struct_shifts.rs`, `nmr-gui/*`, `nmr-gpu/*`, `03-assignment.md` |
| **`21.W1`** | `T21.2` | `nmr-core::struct_shifts`, `nmr-cli::predict_shifts_cmd` | `crates/nmr-core/src/struct_shifts.rs`, `crates/nmr-core/tests/struct_shifts_tests.rs`, `crates/nmr-cli/src/predict_shifts_cmd.rs`, `crates/nmr-cli/tests/predict_shifts_cli_tests.rs` | `kabsch_sander_ss.rs`, `af_concordance.rs`, `report.rs`, `nmr-gui/*`, `03-assignment.md` |
| **`21.W1`** | `T21.3` | `nmr-gui::{ss_ribbon_widget, mol3d_viewer, mol3d_overlay}` | `crates/nmr-gui/src/ss_ribbon_widget.rs`, `crates/nmr-gui/src/mol3d_viewer.rs`, `crates/nmr-gui/src/mol3d_overlay.rs`, `crates/nmr-gui/src/lib.rs`, `crates/nmr-gui/src/main.rs`, `crates/nmr-gui/tests/ss_ribbon_widget_tests.rs` | `crates/nmr-core/*`, `crates/nmr-gpu/*`, `crates/nmr-cli/*` |
| **`21.W2`** | `T21.4` | `nmr-core::af_assign_priors`, `nmr-gpu` `U`-matrix parity, `nmr-cli::af_assign_cmd` | `crates/nmr-core/src/af_assign_priors.rs`, `crates/nmr-core/src/solver.rs` (opt-in hook only), `crates/nmr-core/src/evo_graph.rs` (opt-in hook only), `crates/nmr-gpu/src/lib.rs` + `crates/nmr-gpu/src/shaders/scoring.wgsl` (opt-in AF buffer), `crates/nmr-core/tests/af_assign_priors_tests.rs`, `crates/nmr-gpu/tests/af_assign_gpu_tests.rs`, `crates/nmr-cli/src/af_assign_cmd.rs` | `docs/specs/03-assignment.md`, `domain_dock*`, `rdc*`, `structure.rs`, `nmr-gui/*` |
| **`21.W2`** | `T21.5` | `nmr-core::domain_dock`, `nmr-gpu::domain_dock_gpu`, `nmr-cli::domain_dock_cmd` | `crates/nmr-core/src/domain_dock.rs`, `crates/nmr-gpu/src/domain_dock_gpu.rs`, `crates/nmr-gpu/src/shaders/domain_dock.wgsl`, `crates/nmr-core/tests/domain_dock_tests.rs`, `crates/nmr-gpu/tests/domain_dock_gpu_tests.rs`, `crates/nmr-cli/src/domain_dock_cmd.rs` | `af_assign_priors.rs`, `solver.rs`, `evo_graph.rs`, `scoring.wgsl`, `03-assignment.md`, `nmr-gui/*` |
| **`21.W3`** | `T21.6a` | `nmr-core::cofold_bridge`, `nmr-cli::cofold_cmd` | `crates/nmr-core/src/cofold_bridge.rs`, `crates/nmr-core/tests/cofold_bridge_tests.rs`, `crates/nmr-cli/src/cofold_cmd.rs`, `crates/nmr-cli/tests/cofold_cli_tests.rs` | `excited_state.rs`, `exchange_cmd.rs`, `mcp.rs`, `nmr-gui/*`, `docs/manual/*` |
| **`21.W3`** | `T21.6b` | `nmr-core::{excited_state, mcp}`, `nmr-cli::exchange_cmd`, `nmr-gui::integrative_af_panel`, `docs/manual/` | `crates/nmr-core/src/excited_state.rs`, `crates/nmr-core/tests/excited_state_tests.rs`, `crates/nmr-cli/src/exchange_cmd.rs`, `crates/nmr-gui/src/integrative_af_panel.rs`, `crates/nmr-gui/src/lib.rs`, `crates/nmr-gui/src/main.rs`, `crates/nmr-core/src/mcp.rs`, `crates/nmr-core/tests/mcp_tests.rs`, `docs/specs/18-mcp-server-and-bridge.md`, `docs/manual/09-alphafold-and-integrative-nmr.md`, `docs/manual/{index,06-mcp-and-cli-reference,07-clean-room-provenance}.md`, `scripts/build_docs_manual.py` | `cofold_bridge.rs`, `cofold_cmd.rs`, `03-assignment.md` |

---

## 1. Wave `21.W1` — AlphaFold Ingestion, Kabsch–Sander `8 -> 4`-State Secondary Structure & NMR Concordance (`T21.1`)

### 1.1 AlphaFold mmCIF, PDB & `PAE` JSON Ingestion (`nmr-core::af_concordance`)

1. **Per-Residue `pLDDT` Extraction (`parse_af_structure_model`)**:
   - **Primary ModelArchive mmCIF Table**: Scans `loop_` `_ma_qa_metric_local` (`label_asym_id`, `label_seq_id`, `label_comp_id`, `metric_id`, `metric_value`) cross-referenced against `_ma_qa_metric` (`id`, `name`, `mode`, `type`) to extract exact `pLDDT` values $\in [0.0, 100.0]$ per polymer residue without relying on isotropic $B$-factor overloading.
   - **Controlled `B_iso_or_equiv` / PDB `tempFactor` Fallback**: When `_ma_qa_metric_local` is absent:
     - Checks `_exptl.method` / PDB `EXPDTA`: if the coordinate file explicitly declares crystallographic or solution NMR experimental provenance (`X-RAY DIFFRACTION`, `SOLUTION NMR`, `ELECTRON MICROSCOPY`) and `--force-bfactor-plddt` is false, sets `plddt_per_residue = None` (`CoordinateProvenance::Experimental`) so thermal $B$-factors ($\text{\AA}^2$) are never misinterpreted as `pLDDT`!
     - For computational models (`_ma_protocol_step`, `ALPHAFOLD`, `BOLTZ`, `COLABFOLD`, or `EXPDTA` absent with all `CA` `B_iso_or_equiv` $\in [0.0, 100.0]$; if $\in [0.0, 1.0]$ with median $> 0.15$ and max $\le 1.0$, automatically rescales `[0, 1] -> [0, 100]` with an audit note in `AfIngestDiagnostic`), extracts residue `CA` `B_iso_or_equiv` (or mean backbone heavy-atom `pLDDT` on AF3 atom-level outputs).
   - **Hostile-Input Guards**: Pre-allocation coordinate size cap (`<= 64 MiB`), strict `.is_finite()` checks on all $(x, y, z)$ and `pLDDT` values, and explicit `Err` on truncated mmCIF loops or `NaN`/`Inf` entries (`0` production `.unwrap()`/`.expect()`).

2. **`PAE` JSON $N \times N$ Parser (`parse_pae_json_str` -> `PaeMatrix`)**:
   - Supports all three public `PAE` JSON dialects without external schema crates:
     - **AFDB v3/v4**: `[{"predicted_aligned_error": [[f32; N]; N], "max_predicted_aligned_error": f32}]` (plus legacy sparse v1/v2 `residue1`/`residue2`/`distance` array fallback),
     - **AF2 / ColabFold**: `{"pae": [[f32; N]; N], "max_pae": f32, "plddt": [f32; N]}`,
     - **AlphaFold 3 `confidences.json`**: `{"pae": [[f32; N_tok]; N_tok], "atom_plddts": [...], "token_chain_ids": [...], "token_res_ids": [...]}` aggregating multi-token modified residues or selecting target polymer chain tokens.
   - Pre-allocation dimension guard `1 <= N <= MAX_PAE_RESIDUES` (`4,096`, `<= 64 MiB` `Vec<f32>` storage), verifying exact square matrix lengths (`row.len() == N`), finite values $\in [0.0, 64.0]\text{ \AA}$, and sequence length match $N == L_{\text{chain}}$ before evaluation.

3. **Optional Rate-Limited AFDB Fetcher (`nmr-cli::af_validate_cmd`, Opt-In CLI Only)**:
   - `ferric af-validate --fetch-uniprot <UNIPROT_ID> --cache-dir data/private/afdb/` queries `https://alphafold.ebi.ac.uk/api/prediction/<UNIPROT_ID>` strictly from `nmr-cli` (`nmr-core` performs zero network I/O, preserving deterministic `wasm32` builds).
   - Enforces strict `[A-Z0-9]{6,10}` UniProt accession validation, `<= 1 req/s` token-bucket rate limiting, `User-Agent: ferric-af-fetch`, HTTPS host allowlist (`alphafold.ebi.ac.uk` only), path-traversal rejection on output filenames, and writes `PROVENANCE.md` recording upstream URLs, `AFDB` version, `CC-BY-4.0` license attribution, UTC timestamp, byte sizes, and `FIPS 180-4` `SHA-256` digests.

### 1.2 Clean-Room Kabsch–Sander (1983) `8 -> 4`-State Secondary-Structure Classifier (`nmr-core::kabsch_sander_ss`)

1. **Amide Proton Reconstruction & Partial-Charge Electrostatic H-Bond Energy**:
   - For every non-Proline residue $j \ge 1$ with intact peptide bond $d(\text{C}'_{j-1}, \text{N}_j) \le 2.5\text{ \AA}$, reconstructs ideal trans-peptide amide proton position $\mathbf{r}_{\text{H}_j} = \mathbf{r}_{\text{N}_j} + 1.00\text{ \AA}\cdot\widehat{(\mathbf{r}_{\text{C}'_{j-1}} - \mathbf{r}_{\text{O}_{j-1}})}$ whenever explicit `H`/`HN` coordinates are absent (Kabsch & Sander *Biopolymers* **1983**, 22, p. 2580).
   - Evaluates the Kabsch–Sander electrostatic interaction energy between carbonyl donor $\text{C}'_i=\text{O}_i$ and amide acceptor $\text{N}_j\text{–H}_j$ for all $|i - j| \ge 2$ (`r_min >= 0.5 Å` clash floor):
     $$E_{\text{HB}}(i \to j) = q_1 q_2 \left(\frac{1}{r_{\text{O}_i\text{N}_j}} + \frac{1}{r_{\text{C}'_i\text{H}_j}} - \frac{1}{r_{\text{O}_i\text{H}_j}} - \frac{1}{r_{\text{C}'_i\text{N}_j}}\right)\cdot 332\text{ kcal mol}^{-1}, \qquad q_1 = 0.42,\; q_2 = 0.20$$
     retaining up to the two strongest H-bonds per donor and per acceptor satisfying $E_{\text{HB}}(i \to j) < -0.5\text{ kcal mol}^{-1}$.

2. **8-State Taxonomy (`KabschSanderState8`) & Deterministic `8 -> 4`-State NMR Reduction (`SecondaryStructure4`)**:
   - Assigns elementary patterns exactly per Kabsch & Sander (1983):
     - $n$-turns ($n \in \{3, 4, 5\}$) where $E_{\text{HB}}(i \to i + n) < -0.5\text{ kcal mol}^{-1}$; consecutive overlapping $4$-turns form $\alpha$-helices (`H`), $3$-turns form $3_{10}$-helices (`G`), and $5$-turns form $\pi$-helices (`I`).
     - Parallel ($\text{HB}(i-1 \to j) \land \text{HB}(j \to i+1)$ or symmetric) and antiparallel ($\text{HB}(i \to j) \land \text{HB}(j \to i)$ or $\text{HB}(i-1 \to j+1) \land \text{HB}(j-1 \to i+1)$) $\beta$-bridges: isolated single bridges -> `B`; $\ge 2$ contiguous bridges (including Kabsch–Sander $1$- vs $\le 4$-residue $\beta$-bulges) -> extended strand `E`.
     - Unassigned $n$-turn interior/flanking H-bonded turns -> `T`; virtual $\text{C}^\alpha$ bend angle $\kappa_i = \arccos\!\bigl(\widehat{\mathbf{r}_{\text{C}\alpha_i} - \mathbf{r}_{\text{C}\alpha_{i-2}}} \cdot \widehat{\mathbf{r}_{\text{C}\alpha_{i+2}} - \mathbf{r}_{\text{C}\alpha_i}}\bigr) > 70^\circ$ -> `S`; remaining residues -> `C`, resolved under precedence `H > B > E > G > I > T > S > C`.
   - Collapses `KabschSanderState8` + $(\phi_i, \psi_i)$ into FERRIC's 4-state NMR secondary-structure alphabet (`SecondaryStructure4`, matching `T11.3` `ConformationalState`):
     - **`Helix` (`H`)**: `{H, G, I}`,
     - **`Sheet` (`E`)**: `{E, B}`,
     - **`Ppii` (`P`)**: `{T, S, C}` lacking backbone H-bonds whose $(\phi_i, \psi_i)$ falls inside the left-handed polyproline II basin ($\phi_i \in [-105^\circ, -45^\circ]$, $\psi_i \in [+115^\circ, +175^\circ]$, Adzhubei et al. **2013**) across $\ge 2$ consecutive residues or adjacent to `Pro`,
     - **`Coil` (`C`)**: all remaining `{T, S, C}` residues.

### 1.3 `pLDDT` vs Experimental `RCI-S²` / Lipari–Szabo $S^2$ Concordance & Regime Classifier

1. **Experimental Rigidity Track $S^2_{\text{exp},i} \in [0, 1]$**:
   - Uses experimental Lipari–Szabo generalized order parameters $S^2_i$ (`T13.3` `modelfree` TSV) when supplied via `--s2 <modelfree.tsv>`; otherwise computes 6-nucleus unlabelled-offset-corrected Berjanskii–Wishart `RCI` order parameters $S^2_{\text{RCI},i} \in [0, 1]$ (Berjanskii & Wishart *JACS* **2005**) directly from the assigned shift table (`NMR-STAR` / `NEF`).
2. **Dual Rigidity Concordance Metrics (Fowler, Sljoka & Williamson 2020, 2022)**:
   - Maps per-residue $\text{pLDDT}_i \in [0, 100]$ onto effective solution rigidity $\hat{S}^2_{\text{AF},i} = \operatorname{clamp}\!\bigl((\text{pLDDT}_i - 35.0) / 55.0,\; 0.0,\; 1.0\bigr)$ and evaluates across all $N_{\text{eval}}$ assigned non-terminal residues ($N_{\text{eval}} \ge 10$ required, else `ConcordanceSummary::InsufficientCoverage`):
     $$\rho_s = \operatorname{Spearman}\bigl(\{\text{pLDDT}_i\}_{i \in \mathcal{E}},\, \{S^2_{\text{exp},i}\}_{i \in \mathcal{E}}\bigr), \qquad \text{RMSD}_{\text{rig}} = \sqrt{\frac{1}{N_{\text{eval}}}\sum_{i \in \mathcal{E}} \bigl(\hat{S}^2_{\text{AF},i} - S^2_{\text{exp},i}\bigr)^2}$$
     reporting tie-adjusted Spearman $\rho_s \in [-1, +1]$, correlation score $C_{\text{corr}} = 50(1 + \rho_s) \in [0, 100]$, and RMSD concordance score $C_{\text{rmsd}} = 100\exp\bigl(-(\text{RMSD}_{\text{rig}} / 0.22)^2\bigr) \in [0, 100]$.
3. **4-Class Per-Residue Structural Discordance Taxonomy (`AfNmrConcordanceClass`)**:
   Combining $\text{pLDDT}_i$, $S^2_{\text{exp},i}$, 4-state `SecondaryStructure4` vs `T11.3` `SSP`/`δ2D` populations $(p_\alpha, p_\beta, p_{\text{PPII}}, p_{\text{coil}})$, and homolog-firewalled `T12.3` $(\hat{\phi}_i^{\text{NMR}}, \hat{\psi}_i^{\text{NMR}})$ circular dihedral divergence $\Delta\theta_i = \sqrt{\Delta_{\text{circ}}(\phi_i^{\text{AF}}, \hat{\phi}_i^{\text{NMR}})^2 + \Delta_{\text{circ}}(\psi_i^{\text{AF}}, \hat{\psi}_i^{\text{NMR}})^2}$:
   - **`ConcordantOrdered`**: $\text{pLDDT}_i \ge 70.0$, $S^2_{\text{exp},i} \ge 0.65$, no `Helix <-> Sheet` secondary-structure inversion, and $\Delta\theta_i < 60^\circ$ whenever `TorsionDb` classifies residue $i$ as `Good`.
   - **`ConcordantDisordered`**: $\text{pLDDT}_i < 70.0$ and $S^2_{\text{exp},i} < 0.65$ (flexible loop, tail, linker, or IDP region in both AF and solution).
   - **`ConditionalFoldingOrCrystalArtifact`**: $\text{pLDDT}_i \ge 70.0$ with structured `Helix` or `Sheet` in the 3D model, whereas solution NMR reports disorder ($S^2_{\text{exp},i} < 0.60$ and $p_{\text{coil},i} + p_{\text{PPII},i} \ge 0.65$), identifying partner-induced folding elements ($\alpha$-MoRFs, Alderson et al. **2023**) or crystal-packing helices.
   - **`DynamicHingeOrAltState`**: Either (a) $\text{pLDDT}_i \ge 70.0$ and $S^2_{\text{exp},i} \ge 0.65$ with large local conformational divergence ($\Delta\theta_i \ge 60^\circ$ or `Helix <-> Sheet` state flip), or (b) $\text{pLDDT}_i < 70.0$ while solution NMR is rigidly ordered ($S^2_{\text{exp},i} \ge 0.72$), flagging an alternate solution state, oligomerization interface, or hinge switch.
   - Unassigned NMR residues ($< 2$ backbone shifts and no $S^2_i$) emit `UnassignedNmr`.
4. **CLI Contract**:
   `ferric af-validate --structure <model.cif|model.pdb> [--pae <pae.json>] --shifts <assigned.str|assigned.nef> [--s2 <modelfree.tsv>] [--fetch-uniprot <id> --cache-dir <dir>] [--exclude-pdb <ids>] [--exclude-bmrb <ids>] [-o <out.json|out.tsv|report.html>]`

---

## 2. Wave `21.W1` — Clean-Room 3D Structure-to-Chemical-Shift Forward Predictor (`T21.2`, `nmr-core::struct_shifts`)

### 2.1 Additive Physical & CC0-Firewalled Shift Decomposition

For each residue $i \in \{0, \dots, L-1\}$ and backbone nucleus $a \in \{\text{H}^{\text{N}}, \text{N}, \text{C}^\alpha, \text{C}^\beta, \text{C}', \text{H}^\alpha\}$ plus `ILVMAT` methyl pairs $m \in \{\text{Ile }\delta_1, \text{Leu }\delta_1/\delta_2, \text{Val }\gamma_1/\gamma_2, \text{Met }\varepsilon, \text{Ala }\beta, \text{Thr }\gamma_2\}$:

$$\delta^{\text{struct}}_{i,a}(\mathbf{X}) = \delta^{\text{RC}}_{i,a}(T, \text{pH}) + \Delta\delta^{\text{tor}}_{i,a}(\phi_i, \psi_i, \phi_{i-1}, \psi_{i-1}, \chi_{1,i}) + \Delta\delta^{\text{HB}}_{i,a}(\mathbf{X}) + \Delta\delta^{\text{RC-arom}}_{i,a}(\mathbf{X}) + \Delta\delta^{\text{pep/EF}}_{i,a}(\mathbf{X})$$

1. **Sequence-Corrected Random-Coil Baseline $\delta^{\text{RC}}_{i,a}(T, \text{pH})$**:
   Evaluates `ncIDP` base coil shifts + $(i \pm 1)$ sequence-neighbor corrections (including pre-Proline $\Delta\delta_{\text{CA}} = -1.95\text{ ppm}, \Delta\delta_{\text{C}'} = -1.95\text{ ppm}$) + Platzer Henderson–Hasselbalch pH titration + Kjaergaard linear temperature coefficients (`T11.2`).

2. **CC0-Fitted Trigonometric Fourier / von Mises Torsion Hypersurfaces $\Delta\delta^{\text{tor}}_{i,a}$ (`<= 25%` `HomologFirewall`)**:
   - Subtracting physical non-local terms ($\Delta\delta^{\text{RC-arom}}$, $\Delta\delta^{\text{HB}}$) from the experimental secondary shifts in FERRIC's CC0 `data/torsion_db/torsion_corpus.tsv` (`129` chains, `13,647` residues), `fit_torsion_fourier_surfaces` solves ridge-regularized ($L_2$ penalty $\lambda_{\text{ridge}} = 2.5$) tensor-product Fourier series of harmonic order $K = 3$ on $(\phi_i, \psi_i)$ (`49` basis functions $\{1, \cos k\phi_i, \sin k\phi_i\}_{k=1}^3 \otimes \{1, \cos m\psi_i, \sin m\psi_i\}_{m=1}^3$), augmented by:
     - Preceding-residue $K' = 2$ $(\phi_{i-1}, \psi_{i-1})$ Fourier basis on $^{15}\text{N}_i$ and $^1\text{H}^{\text{N}}_i$ (capturing peptide-bond $\psi_{i-1}$ sensitivity; Shen & Bax **2010**),
     - Sidechain $\chi_{1,i}$ harmonics $\{1, \cos\chi_1, \sin\chi_1, \cos 3\chi_1\}$ on $\text{C}^\alpha_i, \text{C}^\beta_i, \text{H}^\alpha_i$ for non-`Gly`/`Ala` residues,
     - Stratified across structural residue classes (`General`, `Gly`, `Pro`, `PrePro`, `BetaBranched` `Ile/Val/Thr`).
   - **Mandatory `HomologFirewall` Invariant**: In `predict_structure_shifts_with_firewall(structure, firewall_query, cfg)`, `torsion_corpus.tsv` is filtered **first** through all 4 layers of `nmr_core::torsion_db::HomologFirewall` (`L1` PDB/BMRB/UniProt, `L2` CATH/ECOD topology, `L3` NW+SW+7-mer `max_seq_identity <= 0.25`, and `L4` `TM-score < 0.50` computed directly against `structure`'s $\text{C}^\alpha$ trace!), solving the `~65 x 65` Cholesky ridge normal equations in `< 15 ms` strictly on non-homologous templates and recording `exclusion_list_sha256` and `max_retained_seq_identity` in `StructShiftPredictionReport`.

3. **Haigh–Mallion (1972) Conjugated Aromatic Ring-Current Tensor $\Delta\delta^{\text{RC-arom}}_{i,a}(\mathbf{X})$**:
   - Enumerates planar conjugated rings $R$ (`Phe` `CG–CD1–CE1–CZ–CE2–CD2` $i_R = 1.00$; `Tyr` 6-ring $i_R = 0.84$; `His` `CG–ND1–CE1–NE2–CD2` $i_R = 0.53$; `Trp` pyrrole 5-ring $i_R = 0.56$ and fused benzene 6-ring $i_R = 1.04$; Case **1995**) within $12.0\text{ \AA}$ of target spin $\mathbf{r}_p$ (excluding intra-ring covalent self-terms):
     $$\Delta\delta^{\text{RC-arom}}_{i,a}(\mathbf{r}_p) = \sum_{R} i_R B_{\text{HM},a} \sum_{(u,v) \in \partial R} \frac{1}{2}\Bigl[\hat{\mathbf{n}}_R \cdot \bigl((\mathbf{r}_u - \mathbf{r}_p) \times (\mathbf{r}_v - \mathbf{r}_p)\bigr)\Bigr]\left(\frac{1}{r_{up}^3} + \frac{1}{r_{vp}^3}\right)$$
     with proton Haigh–Mallion constant $B_{\text{HM},\text{H}} = 5.455\text{ ppm \AA}$ (`H^N`, `H^alpha`, `H^methyl`), heavy-atom susceptibility scaling on `15N`/`13C`, and smooth short-range clash damping $r_{up} \leftarrow \max(r_{up}, 1.80\text{ \AA})$.

4. **Wagner–Pardi–Bax Hydrogen-Bond Deshielding $\Delta\delta^{\text{HB}}_{i,a}$ & Peptide-Plane Anisotropy / Electric-Field Shifts $\Delta\delta^{\text{pep/EF}}_{i,a}$**:
   - **Amide $^1\text{H}^{\text{N}}$ & $^{15}\text{N}$ H-bond shifts** (Wagner, Pardi & Bax *JACS* **1983**): for backbone or sidechain carbonyl/carboxylate oxygen acceptors $\text{O}_k$ within $r_{\text{H}_i\text{O}_k} \in [1.55, 3.20]\text{ \AA}$ and $\theta_{\text{N}_i\text{H}_i\text{O}_k} \ge 110^\circ$,
     $$\Delta\delta^{\text{HB}}_{\text{H}^{\text{N}},i} = c_{\text{HB,H}}\left(\frac{1}{r_{\text{H}_i\text{O}_k}^3} - \frac{1}{r_0^3}\right)\cos^2\theta_{\text{N}_i\text{H}_i\text{O}_k}, \qquad r_0 = 2.60\text{ \AA}$$
     paired with donor carbonyl $^{13}\text{C}'_i$ H-bond deshielding when $\text{O}_i$ accepts an H-bond.
   - **Peptide-group magnetic anisotropy & bond-parallel electric field** (McConnell **1957**; Buckingham **1960**; Sitkoff & Case **1998**): sums point-dipole carbonyl $\Delta\boldsymbol{\chi}_{\text{C=O}}$ shielding from non-adjacent peptide groups within $8.0\text{ \AA}$ plus partial-charge electric-field projection $\Delta\delta^{\text{EF}} = -A_{\parallel,a}(\hat{\mathbf{u}}_{\text{bond}} \cdot \mathbf{E})$ along $\text{C}^\alpha\text{–H}^\alpha$ and $\text{N–H}^{\text{N}}$ bonds.

5. **`ILVMAT` Sidechain Methyl (`^1H^m, ^{13}C^m`) Forward Model (`--methyl-ilvmat`)**:
   - Predicts `(Ile CD1/HD1*, Leu CD1/HD1* & CD2/HD2*, Val CG1/HG1* & CG2/HG2*, Met CE/HE*, Ala CB/HB*, Thr CG2/HG2*)` shifts by combining $\chi_1 / \chi_2$ $\gamma$-gauche $^{13}\text{C}$ shielding ($\approx -4.8\text{ ppm}$ per $\gamma$-gauche heavy-atom substituent; Sahakyan et al. **2011**) with 3-proton rotamer-averaged Haigh–Mallion $^1\text{H}^{\text{m}}$ ring-current and peptide anisotropy shifts.

6. **Smooth `pLDDT`-Sigmoid Mean & Predictive Variance Blending**:
   - When predicting shifts from an AlphaFold `.cif` carrying per-residue $\text{pLDDT}_i \in [0, 100]$, defines structural confidence weight
     $$w_{\text{AF}}(\text{pLDDT}_i) = \frac{1}{1 + \exp\!\bigl(-(\text{pLDDT}_i - 70.0) / 8.0\bigr)} \in (0, 1)$$
     and blends both mean shift and predictive variance smoothly into sequence-corrected coil priors ($\delta^{\text{RC}}_{i,a}, \sigma^2_{\text{coil},a}$) across low-`pLDDT` disordered segments:
     $$\hat{\delta}_{i,a} = w_{\text{AF},i}\,\delta^{\text{struct}}_{i,a} + (1 - w_{\text{AF},i})\,\delta^{\text{RC}}_{i,a}, \qquad \hat{\sigma}^2_{i,a} = w_{\text{AF},i}^2\,\sigma^2_{\text{struct},a} + \bigl(1 - w_{\text{AF},i}^2\bigr)\,\sigma^2_{\text{coil},a}$$
     (when `pLDDT` is absent on experimental `.pdb`/`.cif` coordinates, $w_{\text{AF},i} \equiv 1.0$).

7. **CLI Contract**:
   `ferric predict-shifts --structure <model.cif|model.pdb> [--ph 6.5] [--temp-k 298.15] [--methyl-ilvmat] [--exclude-pdb <ids>] [--exclude-bmrb <ids>] [--exclude-topology <id>] [-o <out.str|out.nef|out.tsv|out.json>]`

---

## 3. Wave `21.W1` — Zero-Emoji Stacked Vector Secondary-Structure Ribbon & 3D `pLDDT`/Discordance Modes (`T21.3`)

### 3.1 Three-Track Vector Secondary-Structure & Concordance Widget (`nmr-gui::ss_ribbon_widget`)

Rendered via pure `egui::Painter` geometric primitives (`0` emoji characters anywhere in source, UI strings, tooltips, or tests; responsive horizontal zoom/scroll synchronized with sequence residue index `0..L`):

1. **Track 1 — 3D Coordinate Kabsch–Sander Secondary Structure (`3D SS`)**:
   - `Helix` (`H`/`G`/`I`): smooth multi-lobe sinusoidal / shaded cylinder wave in warm coral (`#E05A47`).
   - `Sheet` (`E`/`B`): directional horizontal bar terminating in a crisp filled polygon arrowhead at the C-terminal strand residue in royal gold/teal (`#2A9D8F`).
   - `Ppii` (`P`): compact geometric diamond ticks (`#7B61FF`).
   - `Coil` (`C`): thin centered horizontal baseline stroke.
2. **Track 2 — Live Solution NMR `CSI` / 4-State `SSP` Propensity (`NMR SS`)**:
   - Updates incrementally in `< 1.0 ms` during partial or completed backbone assignment using `nmr-core::idp_propensity::compute_idp_propensity`, drawing proportional upward $\alpha$-helix (`SSP > 0`) and downward $\beta$-strand (`SSP < 0`) vector bars (with unassigned residues rendered as subtle hollow dashes), letting users watch experimental NMR secondary structure emerge alongside the 3D coordinate reference.
3. **Track 3 — `RCI-S²` / Lipari–Szabo $S^2$ vs `pLDDT` Concordance Strip (`AF vs NMR`)**:
   - Renders color-vision-safe vector blocks for `AfNmrConcordanceClass`:
     - `ConcordantOrdered`: `#16A34A` (emerald),
     - `ConcordantDisordered`: `#64748B` (slate),
     - `ConditionalFoldingOrCrystalArtifact`: `#D97706` (amber, diagonal hatch stroke),
     - `DynamicHingeOrAltState`: `#DC2626` (crimson, double-border highlight),
     - `UnassignedNmr`: muted translucent outline.
   - Clicking any residue column fires `SyncCrosshair` + `Mol3dViewerPanel::highlight_sequence_residue(Some(k), true)`, snapping the 2D `HSQC`, 3D sequential walk strips, 2D covalent sidechain inspector, and 3D `wgpu` camera simultaneously.

### 3.2 AlphaFold `pLDDT` & Concordance Color/Putty Modes in `mol3d_viewer` & `mol3d_overlay`

Extends `QuantitativeTrack` in `crates/nmr-gui/src/mol3d_overlay.rs` and `mol3d_viewer.rs` (`wasm32`-clean, `< 2.0 ms` in-place vertex update):
- **`AlphaFoldPlddt`**: Applies the canonical 4-band AlphaFold `pLDDT` palette (Varadi et al. **2022**):
  - `pLDDT >= 90.0`: `#0053D6` (Very High),
  - `70.0 <= pLDDT < 90.0`: `#65CBF3` (Confident),
  - `50.0 <= pLDDT < 70.0`: `#FFDB13` (Low),
  - `pLDDT < 50.0`: `#FF7D45` (Very Low),
  with optional disorder putty radius scaling $r_k = r_{\min} + \bigl(1 - \text{pLDDT}_k / 100\bigr)(r_{\max} - r_{\min})$.
- **`AfNmrConcordance`**: Maps the 4-class `AfNmrConcordanceClass` colors directly onto the 3D cartoon ribbon, swelling putty radius on `ConditionalFoldingOrCrystalArtifact` and `DynamicHingeOrAltState` discrepancies so discordant loops, hinges, and MoRFs stand out immediately in 3D.

---

## 4. Wave `21.W2` — Opt-In 3D-Prior-Assisted Backbone & `PAE`-Gated `15N-NOESY` Assignment (`T21.4`)

### 4.1 Opt-In Boundary & Unlabelled Reference Registration (`nmr-core::af_assign_priors`)

1. **Zero Default Mutation**: `ferric assign` without `--af-structure` passes `af_priors: None`, executing the bit-identical `specs/03` / `specs/10` / `specs/19` sequence-only pipeline (`docs/specs/03-assignment.md` remains `0`-byte diff).
2. **Unlabelled Global Shift Offset Registration (`estimate_af_shift_offsets`)**:
   Before evaluating `U_{s,k}`, estimates global referencing offsets $(\hat{\delta}^{\text{off}}_{\text{H}}, \hat{\delta}^{\text{off}}_{\text{N}}, \hat{\delta}^{\text{off}}_{\text{CA}}, \hat{\delta}^{\text{off}}_{\text{CB}}, \hat{\delta}^{\text{off}}_{\text{C}'})$ strictly between the unordered multiset of observed spin-system shifts and unordered `T21.2` predicted shifts $\{\hat{\delta}_{k,a}\}$ via grid profile-likelihood + Huber IRLS (`0` ground-truth labels, exact permutation invariance `< 1e-12`).

### 4.2 Heavy-Tailed Student-$t(\nu = 4)$ Clamped Unary Injection into $U_{s,k}$

To prevent locally mispredicted AlphaFold loops, domain interfaces, or ligand-free apo hinges from vetoing true experimental sequential triple-resonance walks (`HNCACB`, `CBCACONH`, `HNCA`, `HNCO`), `af_assign_priors` models 3D shift residuals with a heavy-tailed Student-$t(\nu = 4)$ density (Lange, Little & Taylor **1989**) damped by $w_{\text{AF}}(\text{pLDDT}_k)$ and symmetrically clamped to $[-U_{\max}^{\text{clamp}}, +U_{\max}^{\text{clamp}}]$:

$$z_{s,k,a}^{\text{AF}} = \frac{(\delta^{\text{obs}}_{s,a} - \hat{\delta}^{\text{off}}_a) - \hat{\delta}_{k,a}}{\hat{\sigma}_{k,a}}, \qquad \ell_{t_4}(z) = -\frac{5}{2}\ln\!\left(1 + \frac{z^2}{4}\right)$$

$$\Delta U^{\text{AF}}_{s,k} = \lambda_{\text{AF}}\,w_{\text{AF}}(\text{pLDDT}_k)\,\operatorname{clamp}\!\left(\sum_{a \in \mathcal{O}_s} \Bigl[\ell_{t_4}\bigl(z_{s,k,a}^{\text{AF}}\bigr) - \ell_{\text{bg},k,a}\bigl(\delta^{\text{obs}}_{s,a}\bigr)\Bigr],\; -U_{\max}^{\text{clamp}},\; +U_{\max}^{\text{clamp}}\right)$$

with default scale $\lambda_{\text{AF}} = 0.65$ and hard per-residue ceiling $U_{\max}^{\text{clamp}} = 4.0\text{ nats}$ (`af_llr_clamp_nats = 4.0`):
- Where $\text{pLDDT}_k < 50$, $w_{\text{AF}}(\text{pLDDT}_k) < 0.08$, smoothly turning off 3D bias in disordered tails.
- Even on a confident (`pLDDT = 90`) residue whose true solution conformation differs sharply from the AF model (`|z| = 8`), $\Delta U^{\text{AF}}_{s,k} \ge -\lambda_{\text{AF}} U_{\max}^{\text{clamp}} = -2.60\text{ nats}$, which is easily overridden by two consistent sequential carbon/carbonyl bridges (`+8` to `+18 nats`).

### 4.3 `PAE`-Gated `3D 15N-NOESY` Amide Contact Bonus (`Sparse HNCA-Only / Perdeuterated Regime`)

In perdeuterated high-MW or sparse `HNCA`-only datasets lacking $^{13}\text{C}^\beta$ (`HNCACB`/`CBCACONH` absent), `af_assign_priors` optionally consumes a `3D 15N-edited NOESY-HSQC` peak list (`--noesy-15n <peaks.tsv|peaks.nef>`):
1. **Confident Contact Mask $\mathcal{C}_{\text{AF}}$**: Pair $(j, k)$ ($j \ne k$) enters $\mathcal{C}_{\text{AF}}$ if and only if:
   - $d(\text{H}^{\text{N}}_j, \text{H}^{\text{N}}_k) \le 5.5\text{ \AA}$,
   - $\text{pLDDT}_j \ge 70.0$ and $\text{pLDDT}_k \ge 70.0$, and
   - Symmetrized $\overline{\text{PAE}}(j, k) = \frac{1}{2}\bigl(\text{PAE}(j, k) + \text{PAE}(k, j)\bigr) \le 6.0\text{ \AA}$ (when `PAE` JSON is supplied; if `PAE` is omitted, only local sequence separation $|j - k| \le 5$ contacts are admitted so unverified inter-domain contacts never fire!).
2. **Sequential & Medium-Range Pairwise Log-Odds**:
   - Sequential $(k-1, k)$ helical/turn $d_{\text{NN}}(k-1, k) \le 4.5\text{ \AA}$ crosspeaks inject a bounded corroboration bonus $\Delta P^{\text{NOE}}_{s_1, s_2}(k-1, k) \in [0, +2.50]\text{ nats}$ scaled by $(3.0\text{ \AA} / d_{\text{NN}})^6$ and bidirectional crosspeak reciprocity.
   - Medium/long-range $\beta$-sheet $d_{\text{NN}}(j, k) \le 5.0\text{ \AA}$ contacts (`|j - k| >= 2`) inject a soft pairwise factor into `evo-graph` belief propagation and post-SA consensus scoring.

### 4.4 WebGPU `U`-Matrix Parity & CLI Contract

- Uploads packed `f32` `(mu_af, inv_sigma_af, w_af)` tables to `crates/nmr-gpu/src/shaders/scoring.wgsl`, matching CPU `f64` $U + \Delta U^{\text{AF}}$ to `max |U_cpu - U_gpu| < 1.0e-5` across all `f32` workgroups (and `0.0` diff when `enable_af_priors == 0u`).
- CLI: `ferric assign --af-structure <model.cif|pdb> [--af-pae <pae.json>] [--noesy-15n <noesy.tsv|nef>] [--af-weight 0.65] [--af-clamp-nats 4.0] ...`

---

## 5. Wave `21.W2` — `PAE` Spectral Domain Clustering & Multi-Domain `RDC`/`PCS`/`PRE` Docking (`T21.5`)

### 5.1 Symmetrized `PAE` Affinity Graph & Spectral Domain Partitioning (`nmr-core::domain_dock`)

1. **Ordered Core & Symmetrized Affinity Matrix**:
   - Restricts initial graph partitioning to ordered residues $\mathcal{O} = \{i \in \{0,\dots,L-1\} : \text{pLDDT}_i \ge 70.0\}$ and forms the symmetrized `PAE` matrix and Gaussian affinity kernel:
     $$\overline{\text{PAE}}_{ij} = \frac{1}{2}\bigl(\text{PAE}_{ij} + \text{PAE}_{ji}\bigr), \qquad A_{ij} = \exp\!\left(-\frac{\overline{\text{PAE}}_{ij}^2}{2\sigma_{\text{PAE}}^2}\right)\,\mathbb{I}\bigl[\overline{\text{PAE}}_{ij} \le 12.0\text{ \AA}\bigr], \quad \sigma_{\text{PAE}} = 5.0\text{ \AA}$$
2. **Normalized Laplacian Eigengap Clustering (Ng, Jordan & Weiss 2001)**:
   - Evaluates $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ via deterministic cyclic Jacobi eigensolver, selects domain count $D \in \{1, \dots, D_{\max}\}$ ($D_{\max} = 8$) maximizing the Laplacian eigengap $\lambda_{D+1} - \lambda_D$ subject to intra-domain median $\overline{\text{PAE}}_{\text{intra}}(d) \le 5.0\text{ \AA}$ and minimum contiguous/globular domain size $|\mathcal{D}_d| \ge 20$ residues, and assigns row-normalized spectral embeddings via deterministic $k$-means++ (`DetRng`).
   - Intervening residues between rigid domains $\mathcal{D}_d$ and $\mathcal{D}_{d+1}$ ($\text{pLDDT}_i < 70.0$ or high cross-domain $\overline{\text{PAE}}$) form flexible torsion-angle linkers $\mathcal{L}_{\text{link}}$ (while disordered N-/C-termini remain unrestrained tails).

### 5.2 Tolman et al. (2001) Per-Domain Generalized Degree of Order ($\vartheta_d / \vartheta_1$) Motion Test

Before imposing a single static global alignment tensor across all $D$ domains:
1. Fits working (`!is_free`) RDCs independently within each rigid domain $d \in \{1, \dots, D\}$ and alignment medium $m$ via `T12.4` 5-parameter Losonczi Saupe SVD (`fit_saupe_tensor_svd`, requiring $n_{\text{work}}(d, m) \ge 8$ and condition number $\kappa(\mathbf{A}) \le 10^8$).
2. Computes each domain's Generalized Degree of Order (Tolman et al. *JACS* **2001**):
   $$\vartheta_{d,m} = \sqrt{\frac{2}{3}\sum_{\alpha,\beta \in \{x,y,z\}} \bigl(S^{(d,m)}_{\alpha\beta}\bigr)^2}$$
   with parametric Monte Carlo uncertainty $\sigma(\vartheta_{d,m})$ propagated from per-coupling $\sigma_k$.
3. Evaluates the inter-domain order ratio $\rho_{\text{GDO}}(d, m) = \vartheta_{d,m} / \max_{d'} \vartheta_{d',m}$:
   - **`RigidlyCoupledDomains`** ($\min_d \rho_{\text{GDO}}(d, m) \ge 0.80$ or within $2\sigma_{\text{MC}}$ of $1.0$): Domains tumble with a shared molecular alignment tensor $\mathbf{S}^{(m)}$; full static/small-ensemble $\text{SO}(3)^D$ + linker docking proceeds under shared global Saupe SVD.
   - **`DecoupledInterDomainMotion`** ($\min_d \rho_{\text{GDO}}(d, m) < 0.80$ at $> 2\sigma_{\text{MC}}$): Flags large-amplitude inter-domain orientational averaging in `DomainDockReport`, emits an explicit `stderr` & report warning against single-conformer shared-tensor over-interpretation, and scales domain $d$'s dipolar amplitude by $\rho_{\text{GDO}}(d, m)$ (or optimizes a $K$-state ensemble when `--ensemble-size > 1`).

### 5.3 Joint $\text{SO}(3)^D$ + Flexible Linker Kinematics & Hybrid `RDC`/`PCS`/`PRE` Refinement (`nmr-gpu::domain_dock_gpu`)

1. **Covalent Rigid-Domain + Linker Kinematics (`DomainLinkerKinematics`)**:
   - For single-chain multi-domain proteins, internal $(\phi_i, \psi_i, \omega_i)$ angles inside every rigid domain $\mathcal{D}_d$ are frozen to their input AlphaFold coordinates (`0.000 Å` intra-domain RMSD drift), while linker torsions $\{(\phi_l, \psi_l)\}_{l \in \mathcal{L}_{\text{link}}}$ vary freely under `NeRF` forward kinematics (`specs/04` §1.1), rotating and translating downstream domains $\mathcal{D}_{d' > d}$ on their exact covalent polypeptide manifold without chain breaks!
   - For multi-chain non-covalent domain assemblies (`chain_A` + `chain_B`), domain $1$ defines the reference frame while each partner domain $d \ge 2$ carries explicit rigid-body coordinates $(\mathbf{q}_d \in \mathbb{S}^3, \mathbf{t}_d \in \mathbb{R}^3)$.
2. **Composite Hybrid Restraint Objective**:
   $$E_{\text{dock}}(\boldsymbol{\theta}_{\text{link}}, \{\mathbf{q}_d, \mathbf{t}_d\}) = E_{\text{RDC}}^{\text{work}} + E_{\text{PCS}}^{\text{work}} + E_{\text{PRE}} + E_{\text{NOE}}^{\text{inter}} + E_{\text{vdW}}^{\text{inter+link}} + E_{\text{Rama}}^{\text{link}}$$
   - **Multi-Medium `RDC` Term ($E_{\text{RDC}}^{\text{work}}$)**: Variable-projected global 5-parameter Saupe SVD (`T12.4`) over working (`!is_free`) couplings across all rigid domains.
   - **Paramagnetic `PCS` ($\Delta\boldsymbol{\chi}$, $r_{\text{M}}^{-3}$) & `PRE` ($\Gamma_2$, $r_{\text{M}}^{-6}$) Terms (`T11.4`, Bertini et al. 2007; Clore & Schwieters 2004)**: Fits the 5-parameter traceless $\Delta\boldsymbol{\chi}$ tensor from intra-domain `PCS`s in the metal-bearing domain $d_0$ (locking metal position $\mathbf{r}_{\text{M}}$ in $\mathcal{D}_{d_0}$'s frame) and evaluates long-range cross-domain `PCS` residuals ($\Delta\delta^{\text{PCS}}_k = \frac{1}{12\pi r_k^3}\operatorname{Tr}[\Delta\boldsymbol{\chi}(3\hat{\mathbf{r}}_k\hat{\mathbf{r}}_k^\top - \mathbf{I})]$) plus Solomon `PRE` $r_k^{-6}$ distance restraints across partner domains $\mathcal{D}_{d \ne d_0}$, simultaneously pinning both 3D translation and orientation!
   - **Steric Excluded Volume & Linker Ramachandran Prior**: Grid-hashed heavy-atom soft-core quartic clash potential between distinct domains/linkers + Lovell Ramachandran torsion potential on $l \in \mathcal{L}_{\text{link}}$.
3. **Batched WebGPU Multi-Replica Annealing + Host `f64` L-BFGS Polish**:
   - `nmr-gpu::domain_dock_gpu` (`shaders/domain_dock.wgsl`) evaluates batched multi-conformer linker `NeRF` + rigid-domain transforms, global Saupe `RDC` / `PCS` / `PRE` / steric energies, and analytical linker-torsion + $\mathfrak{se}(3)$ gradients across `R <= 256` replicas per TDR-safe dispatch (`validate_domain_dock_gpu_limits` pre-checking `max_buffer_size` and workgroup limits; `< 1.0e-5` CPU/WebGPU energy & gradient parity), followed by host `f64` two-loop `L-BFGS` polish.
4. **Mandatory `ByPeptidePlane` `20%` `RdcFree` Cross-Validation**:
   - Holds out `20%` of peptide planes stratified across every rigid domain (`RdcFreeStratification::ByPeptidePlane`, `0` `is_free` leakage into Saupe SVD or gradients) and reports per-domain and global `Q_work`, `ByPeptidePlane Q_free`, `PCS Q_free`, and inter-domain $\vartheta_d / \vartheta_1$.
5. **CLI Contract**:
   `ferric structure --pae-domains --af-structure <model.cif> --af-pae <pae.json> [--rdc <rdcs.tsv|nef>] [--pcs <pcs.tsv>] [--pre <pre.tsv>] [--noesy <inter_noe.tsv>] [--rdc-free-frac 0.20] [--ensemble-size 10] [--engine auto|gpu|cpu] -o <docked.pdb|docked.cif>`

---

## 6. Wave `21.W3` — `CSP`/`PRE`-Guided Co-Folding Bridge (`AF3` / `Boltz-2`) & Multi-Pose Complex Re-Scoring (`T21.6a`)

### 6.1 Clean-Room `AlphaFold 3` JSON & `Boltz-2` YAML Restraint Exporter (`nmr-core::cofold_bridge`)

1. **Experimental Interface Restraint Extraction (`extract_nmr_interface_restraints`)**:
   - **Surface-Filtered `2\sigma` `CSP` Active Pocket (`T13.6` + Shrake–Rupley `SASA`)**: Identifies active titration residues satisfying $\Delta\delta_{\text{CSP},i} > \overline{\Delta\delta}_{\text{trimmed}} + 2\sigma_{\text{trimmed}}$ (`nmr-core::csp`). When an apo receptor 3D model (`.cif`/`.pdb`) is supplied, computes 92-point Fibonacci-sphere Shrake–Rupley (`1973`, probe radius $1.4\text{ \AA}$) relative solvent-accessible surface area $\text{rSASA}_i = \text{SASA}_i / \text{SASA}_{\max}(\text{AA}_i)$ and retains surface-exposed residues ($\text{rSASA}_i \ge 0.20$) as direct `pocket` contacts while classifying buried core shifts ($\text{rSASA}_i < 0.20$) separately as `AllostericCorePerturbation`!
   - **Pairwise Distance Contacts (`PRE` & Intermolecular `NOE`s)**: Extracts explicit upper distance bounds $d_{\max} \in [4.5, 14.0]\text{ \AA}$ from intermolecular `NOE` restraints and `T11.4` paramagnetic `PRE` bleaching / $\Gamma_2$ tables.
   - **Small-Molecule Ligand & Polymer Sequences**: Accepts multi-chain FASTA (`protein`, `rna`, `dna`), `CCD` 3-letter codes, or `SMILES` / `MOL` / `.oct` small-molecule ligands directly from `Octet-Draw` (`nmr-core::octet_mol`).

2. **Target Co-Folding Schemas (`export_af3_json`, `export_boltz2_yaml`)**:
   - **AlphaFold 3 Input JSON (`dialect: "alphafold3"`, `version: 2`, Abramson et al. 2024)**:
     Emits deterministic, schema-conformant JSON (`{"name": ..., "modelSeeds": [...], "sequences": [{"protein": {"id": "A", "sequence": ...}}, {"ligand": {"id": "L", "smiles": ...}}], "bondedAtomPairs": [...], "dialect": "alphafold3", "version": 2}`), paired with `<prefix>.nmr_restraints.json` recording the extracted `CSP` pocket & `PRE`/`NOE` contact list for downstream `ferric cofold score` pose selection (explicitly documenting that public AF3 JSON `version: 2` carries sequence/ligand/covalent topology while pocket/contact conditioning is natively enforced in `Boltz-2` YAML and `ferric cofold score`).
   - **Boltz-2 Input YAML (`version: 1`, Wohlwend et al. 2024; Passaro et al. 2025)**:
     Emits clean, deterministic zero-dependency YAML containing:
     - `version: 1`,
     - `sequences`: `protein` / `rna` / `dna` / `ligand` (`smiles` or `ccd`),
     - `constraints`:
       - `pocket`: `{binder: <partner_or_ligand_chain_id>, contacts: [[<receptor_chain_id>, <1_based_seq_idx>], ...], max_distance: 6.0, force: false}` populated from surface-exposed `2\sigma` `CSP` hits,
       - `contact`: `{token1: [<chain_1>, <1_based_res_or_atom_name>], token2: [<chain_2>, <1_based_res_or_atom_name>], max_distance: <upper_A>, force: false}` populated from intermolecular `NOE` and `PRE` restraints,
     - Optional `properties: [{affinity: {binder: <ligand_chain_id>}}]` when small-molecule ligand $K_d$ comparison is requested.

### 6.2 Multi-Pose Co-Folded Complex NMR Re-Scorer (`score_cofold_poses`)

Given $M \ge 1$ candidate complex poses (`.cif`/`.pdb` models from `AF3`, `Boltz-2`, or rigid/flexible docking, plus optional co-folding confidence JSONs `iptm` / interface `PAE`), `score_cofold_poses` evaluates each pose $m$ without altering internal coordinates and computes a calibrated composite NMR log-score:

$$S_{\text{total}}(m) = w_{\text{CSP}}\,S_{\text{CSP}}(m) - w_{\text{NOE}}\,E_{\text{viol}}^{\text{NOE/PRE}}(m) - w_{\text{RDC}}\,Q_{\text{free}}^{\text{RDC}}(m) - w_{\text{PCS}}\,Q^{\text{PCS}}(m)$$

1. **Forward-Physics Interface `CSP` Score ($S_{\text{CSP}}(m)$)**:
   - Evaluates complex-vs-apo differential shifts $|\Delta\delta^{\text{calc}}_i(m)| = \sqrt{(\Delta\delta^{\text{calc}}_{\text{H},i})^2 + (\alpha_N \Delta\delta^{\text{calc}}_{\text{N},i})^2}$ by running `T21.2` (`struct_shifts`) on the isolated receptor vs complex pose $m$ — including intermolecular Haigh–Mallion ring-current shifts from partner `Phe`/`Tyr`/`Trp`/`His` rings **and** Hückel-aromatic small-molecule ligand rings (`nmr-core::octet_mol`), intermolecular $\text{N–H}\cdots\text{O}$ H-bonds, and interface $\Delta\text{rSASA}_i$ contact burial — combining Spearman $\rho_s\bigl(\Delta\delta^{\text{obs}}_{\text{CSP}}, |\Delta\delta^{\text{calc}}(m)|\bigr)$ with heavy-atom interface contact ($d_{\min} \le 5.0\text{ \AA}$) ROC-AUC on `2\sigma` `CSP` pocket hits.
2. **Intermolecular `NOE` & `PRE` Violation Penalty ($E_{\text{viol}}^{\text{NOE/PRE}}(m)$)**:
   - Sum-averaged $r_{\text{eff}} = (\sum_{a,b} r_{ab}^{-6})^{-1/6}$ upper/lower-bound quadratic violation energy across intermolecular `NOE`s and `PRE` bounds.
3. **Complex Dipolar & Paramagnetic Quality Factors ($Q_{\text{free}}^{\text{RDC}}(m)$, $Q^{\text{PCS}}(m)$)**:
   - Joint multi-subunit Saupe SVD (`ByPeptidePlane` `20%` `Q_free`) and 5-parameter $\Delta\boldsymbol{\chi}$ `PCS` $Q$-factor when supplied.
4. **Decoy-Cluster Abstention Guard (`PoseVerdict`)**:
   - Emits `PoseVerdict::InconclusiveOrAllViolated` whenever every candidate pose exhibits severe intermolecular distance violations ($\max\Delta_{\text{viol}} > 1.5\text{ \AA}$), fails `CSP` pocket contact enrichment ($\text{AUC}_{\text{CSP}} < 0.58$ when $\ge 5$ surface `CSP` hits exist), or exceeds `ByPeptidePlane Q_free > 0.65`.
5. **CLI & Shared Wave `21.W3` API Contract**:
   - Public Rust signature in `nmr_core::cofold_bridge` (frozen for `T21.6b` MCP consumption):
     `pub fn score_cofold_poses(poses: &[StructurePoseInput], restraints: &CofoldRestraintBundle, cfg: &PoseScoreConfig) -> Result<PoseScoreReport, NmrError>`
   - CLI: `ferric cofold export ...` and `ferric cofold score ...`

---

## 7. Wave `21.W3` — Bayesian Excited-State (`CEST`/`CPMG`) Structural Hypothesis Engine, GUI Studio, `MCP` & Manual (`T21.6b`)

### 7.1 Bayesian Excited-State ($\varpi_B = \varpi_A + \Delta\varpi_{AB}$) Hypothesis Engine (`nmr-core::excited_state`)

1. **Signed (`CEST` / $R_{1\rho}$) vs Unsigned (`CPMG` $|\Delta\varpi|$) Differential Shift Likelihoods**:
   - Let ground-state shifts be $\varpi_{A,i,a}$ on structure $\mathbf{X}_A$, and let hypothesis $H(\boldsymbol{\theta}_H)$ predict differential chemical shift $\mu_{\Delta,i,a}^{(H)}(\boldsymbol{\theta}_H) = \hat{\delta}_{B,i,a}^{(H)}(\boldsymbol{\theta}_H) - \hat{\delta}_{A,i,a}(\mathbf{X}_A)$ with combined variance $\sigma_{\text{tot},i,a}^2 = \sigma_{\text{exp},i,a}^2 + \sigma_{\text{fwd},H,a}^2$ (evaluating *differential* shifts cancels static per-residue force-field offsets on ground state $A$!).
   - **Signed `CEST` / off-resonance $R_{1\rho}$ rows (`ExchangeSignMode::Signed`, $\Delta\varpi_{AB,i,a} \in \mathbb{R}$)**:
     $$\ell_{\text{signed}}\bigl(\Delta\varpi_{AB,i,a} \mid \mu_{\Delta,i,a}^{(H)}, \sigma_{\text{tot},i,a}\bigr) = \ln f_{t_5}\!\left(\frac{\Delta\varpi_{AB,i,a} - \mu_{\Delta,i,a}^{(H)}}{\sigma_{\text{tot},i,a}}\right) - \ln\sigma_{\text{tot},i,a}$$
   - **Unsigned `CPMG` rows (`ExchangeSignMode::UnsignedMagnitude`, $y_{i,a} = |\Delta\varpi_{AB,i,a}| \ge 0$, Skrynnikov et al. 2002)**:
     Exact marginalization over unobserved sign $s_{i,a} \in \{-1, +1\}$ via the folded Student-$t_5$ density:
     $$\ell_{\text{unsigned}}\bigl(y_{i,a} \mid \mu_{\Delta,i,a}^{(H)}, \sigma_{\text{tot},i,a}\bigr) = \ln\!\left[f_{t_5}\!\left(\frac{y_{i,a} - \mu_{\Delta,i,a}^{(H)}}{\sigma_{\text{tot},i,a}}\right) + f_{t_5}\!\left(\frac{-y_{i,a} - \mu_{\Delta,i,a}^{(H)}}{\sigma_{\text{tot},i,a}}\right)\right] - \ln\sigma_{\text{tot},i,a}$$
     supporting mixed multi-experiment tables (e.g., signed $^{15}\text{N}$ `CEST` + unsigned $^{13}\text{C}^\alpha$ `CPMG`) seamlessly in a single joint log-likelihood $\ell(H) = \sum_{(i,a) \in \mathcal{D}} \ell_{i,a}(H)$.

2. **Four Candidate Biophysical Hypotheses ($H_1$–$H_4$) vs Null / Unstructured Perturbation ($H_0$)**:
   - **`LocalUnfolding` ($H_1$, Vallurupalli, Hansen & Kay *PNAS* 2008; $k = 2$)**:
     Scans contiguous cooperative unfolding windows $[u, v]$ covering the exchanging locus, setting $\hat{\delta}_{B,i,a}^{(H_1)} = \delta^{\text{RC}}_{i,a}(T, \text{pH})$ (`ncIDP` random coil from `T11.2`) for $i \in [u, v]$ (so $\mu_{\Delta,i,a}^{(H_1)} = \delta^{\text{RC}}_{i,a} - \varpi_{A,i,a}$) and $\mu_{\Delta,i,a}^{(H_1)} = 0$ outside $[u, v]$ (with `1`-residue boundary damping `0.5`).
   - **`XxxProCisTrans` ($H_2$, Schubert et al. 2002; Shen & Bax 2010; $k = 1$)**:
     Scans candidate Proline residues $p$ within spatial cutoff $d(\text{C}^\alpha_i, \text{C}^\alpha_p) \le 8.0\text{ \AA}$ or sequence window $|i - p| \le 3$, flipping $\omega_{p-1}$ between *trans* ($180^\circ$) and *cis* ($0^\circ$) with canonical Schubert/BMRB differential signatures:
     - Proline $p$: $\Delta\delta_{\text{CB}}(\text{trans}\to\text{cis}) = +2.40\text{ ppm}$, $\Delta\delta_{\text{CG}}(\text{trans}\to\text{cis}) = -2.60\text{ ppm}$ ($\Delta_{\beta\gamma} = \delta_{\text{CB}} - \delta_{\text{CG}}: 4.5 \pm 1.2\text{ ppm} \leftrightarrow 9.6 \pm 1.3\text{ ppm}$), $\Delta\delta_{\text{CA}} = -0.55\text{ ppm}$, $\Delta\delta_{\text{C}'} = -0.85\text{ ppm}$;
     - Preceding residue $p-1$: $\Delta\delta_{\text{CA}} = -1.15\text{ ppm}$, $\Delta\delta_{\text{C}'} = -1.35\text{ ppm}$; following amide $p+1$: $\Delta\delta_{\text{N}} / \Delta\delta_{\text{H}^{\text{N}}}$ peptide-anisotropy & H-bond switch (`T21.2`).
   - **`RotamerOrHisTautomerFlip` ($H_3$, Haigh & Mallion 1972; Pelton et al. 1993; $k = 2$)**:
     Scans (a) $\chi_1 \in \{g^+\, (+60^\circ), t\, (180^\circ), g^-\, (-60^\circ)\}$ and $\chi_2 \in \{\pm 90^\circ, 0^\circ\}$ sidechain rotamer flips of steric-clash-compatible `Phe`/`Tyr`/`Trp`/`His` rings within $10.0\text{ \AA}$ of exchanging nuclei, evaluating exact differential Haigh–Mallion ring-current shifts $\Delta\delta^{\text{RC-arom}}(\mathbf{X}_{B,\text{rot}}) - \Delta\delta^{\text{RC-arom}}(\mathbf{X}_A)$ across all surrounding backbone and methyl spins, and (b) `His` imidazole $\mathrm{N}^{\delta 1}\text{–H} \leftrightarrow \mathrm{N}^{\varepsilon 2}\text{–H}$ tautomer exchange ($\Delta\delta_{\text{ND1}} \approx \pm 82.0\text{ ppm}$, $\Delta\delta_{\text{NE2}} \approx \mp 82.0\text{ ppm}$, $\Delta\delta_{\text{CD2}} \approx \mp 7.2\text{ ppm}$, $\Delta\delta_{\text{CG}} \approx \pm 4.8\text{ ppm}$; Pelton **1993** Table 1) and $\mathrm{His}^0 \leftrightarrow \mathrm{HisH}^+$ protonation exchange.
   - **`Alternate3DConformer` ($H_4$, Bouvignies et al. 2011; Wayment-Steele et al. 2024; $k = 1$)**:
     Evaluates each candidate alternative 3D coordinate model $\mathbf{X}_{B,m}$ supplied via `--alt-structures <alt1.cif,alt2.pdb,...>` (e.g., alternative crystal states, homolog conformations, or subsampled-MSA AlphaFold ensembles) through `T21.2` (`struct_shifts`, enforcing `HomologFirewall` when benchmarking), setting $\mu_{\Delta,i,a}^{(H_4,m)} = \delta^{\text{struct}}(\mathbf{X}_{B,m})_{i,a} - \delta^{\text{struct}}(\mathbf{X}_A)_{i,a}$.

3. **`BIC`, Kass–Raftery Bayes Factors & Calibration Abstention**:
   $$\mathrm{BIC}(H) = -2\ell_{\max}(H) + k_H \ln N_{\text{obs}}, \qquad 2\ln\mathrm{BF}_{1,2} = \mathrm{BIC}\bigl(H_{(2)}\bigr) - \mathrm{BIC}\bigl(H_{(1)}\bigr)$$
   Posterior model weights $w_H \propto \exp(-\frac{1}{2}\mathrm{BIC}(H))$ are reported alongside reduced $\chi^2_\nu(H_{(1)})$ and Pearson/Spearman correlation on $\Delta\varpi_{AB}$. `classify_excited_state` emits `ExcitedStateVerdict::Committed(H_{(1)})` only when $N_{\text{obs}} \ge 4$, $2\ln\mathrm{BF}_{1,2} \ge 6.0$ (Kass & Raftery **1995** positive/strong threshold), $2\ln\mathrm{BF}_{1,H_0} \ge 10.0$, and $\chi^2_\nu(H_{(1)}) \le 2.5$; otherwise it abstains honestly as `ExcitedStateVerdict::AmbiguousOrUnmodeledState`.
4. **CLI Contract**:
   `ferric exchange-state --shifts <ground.str|nef> --structure <ground.cif|pdb> --exchange <cest_or_cpmg.tsv> [--alt-structures <m1.cif,m2.pdb>] [--ph 6.5] [--temp-k 298.15] [-o <excited_state.json|tsv>]`

### 7.2 Interactive `nmr-gui` Integrative Structural NMR Studio (`nmr-gui::integrative_af_panel`)

Adds a zero-emoji dockable/tabbed studio panel in `crates/nmr-gui/src/integrative_af_panel.rs` (desktop + `wasm32-unknown-unknown`):
- **1-Click `.cif` + `PAE` JSON Loader & Concordance Summary**: Displays global Spearman $\rho_s$, `RMSD_rig`, 4-class residue counts, and syncs the 3-track `ss_ribbon_widget` (`T21.3`) + `mol3d_viewer` `AlphaFoldPlddt` / `AfNmrConcordance` color modes.
- **Interactive $N \times N$ `PAE` Domain Heatmap**: Renders the green-to-white (`0..31.75 Å`) `PAE` matrix with overlaid `T21.5` spectral rigid-domain bounding boxes and Tolman $\vartheta_d / \vartheta_1$ GDO badges; clicking any `(i, j)` cell or domain block highlights the domain pair in `mol3d_viewer`.
- **Co-Folding Restraint Preview & Pose Ranker (`T21.6a`)**: Interactive preview/export of `AlphaFold 3` JSON and `Boltz-2` YAML restraints from loaded `CSP`/`PRE`/`NOE` tables, plus sortable multi-pose complex re-scoring table.
- **Excited-State Hypothesis Explorer (`T21.6b`)**: `BIC` / $2\ln\mathrm{BF}$ bar chart and interactive $\Delta\varpi_{\text{obs}}$ vs $\Delta\varpi_{\text{calc}}$ parity plot linked to `Dynamics & Titration Studio` (`T13.4`–`T13.5`) and 3D ribbon highlighting.

### 7.3 Read-Only `MCP` Tools (`nmr-core::mcp`) & Documentation Manual Chapter `09` (`docs/manual/`)

1. **Four New Read-Only `MCP` Tools (`readOnlyHint = true`, `destructiveHint = false`, `McpSecurityPolicy` Canonical Path Sandboxed)**:
   - `af_concordance`: Runs `T21.1` concordance & 4-class discordance analysis on workspace `.cif`/`.pdb` + `.str`/`.nef` (+ optional `pae.json` & `modelfree.tsv`).
   - `predict_structure_shifts`: Runs `T21.2` 3D-to-shift forward prediction enforcing `DEFAULT_FIREWALL_MAX_SEQ_IDENTITY = 0.25` (`HomologFirewall` `<= 25%` ceiling + `exclusion_list_sha256` in JSON response).
   - `score_cofold_poses`: Runs `T21.6a` multi-pose complex NMR re-scoring across workspace `.cif`/`.pdb` poses and `CSP`/`NOE`/`PRE`/`RDC`/`PCS` tables.
   - `classify_excited_state`: Runs `T21.6b` Bayesian `BIC` / Bayes-factor structural hypothesis classification on `CEST`/`CPMG`/`R1ρ` shift differences.
   - Adds guided prompt `validate-and-dock-alphafold-model` to `nmr-core::mcp`.
2. **User Manual & Provenance Updates**:
   - Authors `docs/manual/09-alphafold-and-integrative-nmr.md`, updates `docs/manual/index.md`, `06-mcp-and-cli-reference.md`, `07-clean-room-provenance.md`, `docs/specs/18-mcp-server-and-bridge.md`, and registers `"09-alphafold-and-integrative-nmr.md"` in `EXPECTED_MANUAL_CHAPTERS` inside `scripts/build_docs_manual.py`, passing `python3 scripts/build_docs_manual.py --self-test` and `python3 scripts/build_docs_manual.py --check`.

---

## 8. Quantitative Verification Gates (`T21.1`–`T21.6b`)

| Task | Quantitative Verification Gate |
| :--- | :--- |
| **`T21.1`** | `0` production `.unwrap()`/`.expect()` across hostile mmCIF/`PAE` inputs (truncated `_ma_qa_metric_local`, `NaN`/`Inf` coords/PAE, $N \ne L$, `PAE > 4,096` pre-alloc `Err`, crystallographic `B_iso` guard, `--fetch-uniprot` URL allowlist & `<= 1 req/s` limiter); Kabsch–Sander `8 -> 4`-state classifier achieves `>= 96.0%` agreement on `1D3Z`/`1UBQ` Ubiquitin without reading `HELIX`/`SHEET` headers; concordance classifier achieves `>= 85%` `ConcordantOrdered` ($\rho_s \ge 0.70$) on folded Ubiquitin, `>= 85%` `ConcordantDisordered` on $\alpha$-Synuclein (`bmr6968`), `100%` `ConditionalFoldingOrCrystalArtifact` detection on planted high-`pLDDT` IDP $\alpha$-MoRFs (`bmr15176`/`bmr51984`), and `>= 90%` `DynamicHingeOrAltState` recall on planted $\alpha \leftrightarrow \beta$ conformational switches. |
| **`T21.2`** | Under strict 4-layer `HomologFirewall` (`max_seq_identity <= 0.25`, `TM-score < 0.50`, deterministic `exclusion_list_sha256` verified), out-of-sample 3D-to-shift evaluation across the 9 folded BMRB/PDB benchmark proteins achieves non-homologous RMSE `<= 1.35 ppm` (`CA`), `<= 1.45 ppm` (`CB`), `<= 1.30 ppm` (`C'`), `<= 0.55 ppm` (`H^N`), `<= 2.85 ppm` (`15N`), and `<= 0.35 ppm` (`H^alpha`), with Haigh–Mallion ring-current ablation confirming `>= 20%` proton RMSE reduction on aromatic-proximal ($d \le 5.5\text{ \AA}$) `H^N`/`H^alpha`/`ILVMAT` spins, and `< 1e-6` convergence to `ncIDP` coil shifts as `pLDDT -> 0`. |
| **`T21.3`** | `0` emoji characters across all touched files; headless `egui` unit tests verify all 3 stacked vector ribbon tracks (`3D Kabsch–Sander`, live partial-assignment `CSI`/`SSP`, `Concordance`), bidirectional 1-click residue-to-strip `SyncCrosshair` coupling, `< 2.0 ms` `AlphaFoldPlddt` 4-band & `AfNmrConcordance` 3D putty updates, and `0`-warning native + `wasm32-unknown-unknown` compilation. |
| **`T21.4`** | `0`-byte diff on `docs/specs/03-assignment.md`; `< 1e-15` identity when `--af-structure` is omitted; `AGENTS.md` 5-seed `--realistic` Ubiquitin & 5-seed `CISD3` non-regression gates PASS; CPU vs WebGPU `U`-matrix parity `max |U_cpu - U_gpu| < 1.0e-5` with Student-$t(\nu = 4)$ AF priors enabled; corrupting a 12-residue AF loop by `120°` $(\phi, \psi)$ causes `0.0%` accuracy loss on full triple-resonance Ubiquitin (`U_clamp = 4.0 nats` verified); on sparse `CB`-stripped `HNCA`-only + `3D 15N-NOESY` benchmarks, `--af-structure` + `PAE <= 6 Å` gating improves assignment accuracy by `>= +8.0` percentage points with `0` false inter-domain `PAE > 6 Å` pairwise bonuses. |
| **`T21.5`** | Symmetrized `PAE` normalized spectral clustering recovers exact 2- and 3-domain boundaries (`Adjusted Rand Index >= 0.98`); Tolman $\vartheta_d / \vartheta_1$ GDO diagnostic separates rigid co-tumbling (`|\Delta\vartheta|/\vartheta_1 < 0.05`) from `2x`-attenuated decoupled domain motion (`DecoupledInterDomainMotion` flagged `100%`); CPU/WebGPU batch energy & gradient parity `< 1.0e-5`; from a `60°`-rotated, `8 Å`-translated two-domain AF starting pose, joint 2-medium `RDC` + `PCS`/`PRE` linker-kinematic docking under `ByPeptidePlane` `20%` `RdcFree` holdout achieves `< 1.20 Å` all-domain backbone RMSD and `ByPeptidePlane Q_free < 0.25` in `< 2.0 s` on WebGPU (`0.000 Å` intra-domain rigid distortion). |
| **`T21.6a`** | Deterministic schema-valid `AlphaFold 3` JSON (`dialect == "alphafold3"`, `version == 2`) and `Boltz-2` YAML (`pocket` & `contact` `1`-based indexing, Shrake–Rupley `rSASA >= 0.20` surface pocket filtering) unit tests PASS; across a `10`-pose complex decoy benchmark (`1` near-native `< 1.5 Å` iRMSD + `9` `4–18 Å` decoys), `ferric cofold score` ranks the native pose `#1` (`Top-1 = 100%`, Spearman $\rho_s \ge 0.80$) and returns `InconclusiveOrAllViolated` when only decoy poses are supplied. |
| **`T21.6b`** | Across all `4` excited-state classes (`LocalUnfolding`, `XxxProCisTrans` $\Delta_{\beta\gamma}: 4.5 \to 9.6\text{ ppm}$, `RotamerOrHisTautomerFlip` `Phe` $\chi_1$ ring-current & Pelton `His` $\pm 82\text{ ppm}$ `ND1`/`NE2`, `Alternate3DConformer`), `classify_excited_state` achieves `100%` (`4/4`) top-1 `BIC` recovery ($2\ln\mathrm{BF} > 10$) on both **signed** (`CEST`/`R1ρ`) and **unsigned** (`CPMG` $|\Delta\varpi|$) exchange inputs while abstaining cleanly on pure noise; all `4` new read-only `MCP` tools pass `ReadOnly` & `McpSecurityPolicy` sandbox escape tests; `integrative_af_panel` compiles `wasm32`-clean; and `python3 scripts/build_docs_manual.py --self-test && python3 scripts/build_docs_manual.py --check` exits `0`. |
