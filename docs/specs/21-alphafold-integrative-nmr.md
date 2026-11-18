# 21. AlphaFold, Secondary Structure & Integrative Structural NMR (`T21.1`–`T21.6b`)

> **Validation Scope & Scientific Governance (`AGENTS.md` & `Rule 5`)**:
> 1. **Sequence-Only Backbone Assignment Preserved (`docs/specs/03-assignment.md` Untouched)**: Default `ferric assign` (`--solver standard|evo-graph|bayes-sa`) remains strictly sequence-only (`af_priors: None`) with zero 3D coordinate inputs and bit-identical CPU/WebGPU `U`/`P` scoring (`5c4e87d` golden snapshots). AlphaFold 3D shift and `PAE`-gated `15N-NOESY` priors (`T21.4`) are **strictly opt-in** via `--af-structure <model.cif>`, emit `confidence_calibrated = false` (`uncalibrated_af_prior = true`), and leave `docs/specs/03-assignment.md` completely unmodified (`0`-byte diff).
> 2. **Strict Non-Homologous `HomologFirewall` (`<= 25%` Sequence Identity, `TM-score < 0.50`, `CC0` + `CC-BY-4.0` Attribution)**: Empirical $(\phi_i, \psi_i, \phi_{i-1}, \psi_{i-1}, \phi_{i+1}, \psi_{i+1}, \chi_1, \omega)$ Fourier shift hypersurfaces in `T21.2` (`nmr-core::struct_shifts`) are fitted on the fly from `data/torsion_db/torsion_corpus.tsv` (`129` chains, `13,647` residues; shifts/coords from `CC0 1.0` BMRB/wwPDB, UniProt & CATH metadata under `CC-BY-4.0`; `4` entries overlap `bmrb_priors.json` and are caught by `L1`–`L4` firewall when benchmarking) with mandatory fail-closed 4-layer `HomologFirewall` exclusion (`L1` PDB/BMRB/UniProt, `L2` CATH/ECOD topology, `L3` NW+SW+7-mer `<= 25%` sequence identity, `L4` `TM-score < 0.50`) and cryptographic `exclusion_list_sha256` audit tracking.
> 3. **Single- vs Multi-Medium Dipolar Candor (`F8` Carried Forward)**: In `T21.5` (`domain_dock`), a single alignment medium's `RDC` tensor carries intrinsic 4-fold principal-axis $\pi$-rotation cone degeneracy and zero translational sensitivity; unambiguous 3D multi-domain rigid-body placement requires either $\ge 2$ non-coaxial `RDC` alignment media paired with covalent linker / steric constraints, or translational-plus-orientational paramagnetic `PCS` ($\Delta\boldsymbol{\chi}$, $r_{\text{M}}^{-3}$) / SI Solomon `PRE` ($\Gamma_2$, $r_{\text{M}}^{-6}$ with mandatory user-supplied $\tau_c$ and $\omega_H$) / inter-domain `NOE` restraints, always cross-validated under `ByPeptidePlane` `20%` `RdcFree` and domain-stratified `20%` `PCS` $Q_{\text{free}}$ holdouts.
> 4. **Candid Provenance Labelling**: Current Phase 21 regression suites evaluate `[dev-set]` folded PDB/BMRB structures, `[pre-cutoff AFDB]` `AF-P0CG48-F1` v6 models, and `[synthetic self-consistency]` forward-simulated benchmarks; pre-registered cold post-cutoff empirical benchmarks (`AF2`/`AF3`, sparse `HNCA`+`NOESY`, 2-domain `CaM`/`MPD-ubq`, experimental `CSP`/`PRE` complexes, and published `CEST`/`CPMG` excited states) are tracked on the maintainer Science track (`TODO.md`).

---

## Provenance & Clean-Room Statement

Original clean-room technical specification and mathematical architecture for Phase 21 (`T21.1`–`T21.6b`). Zero source code, binaries, decompiled output, neural network weights, residue libraries, macro scripts, or parameter tables from any copyleft (`GPL`/`AGPL`), academic-licensed, or proprietary secondary-structure classifier, chemical-shift predictor, structure validation tool, macromolecular docking suite, or NMR refinement package were consulted. No external product names appear in any Rust module, struct, enum, trait, function, or CLI identifier (`kabsch_sander_ss`, `af_concordance`, `struct_shifts`, `af_assign_priors`, `domain_dock`, `cofold_bridge`, `excited_state`). Every physical forward model, graph clustering algorithm, statistical criterion, and file schema is derived strictly from peer-reviewed literature, public open-format specifications, and `CC0` / `CC-BY-4.0` archive standards:

1. **AlphaFold Confidence Metrics (`pLDDT`, `PAE`), ModelArchive mmCIF & Public Database Schemas (`T21.1`)**:
   - Jumper, J., Evans, R., Pritzel, A. et al. (*Nature* **2021**, 596, 583–589) — AlphaFold 2 architecture, per-residue predicted Local Distance Difference Test (`pLDDT` $\in [0, 100]$), and $N \times N$ Predicted Aligned Error (`PAE`, Å).
   - Mariani, V., Biasini, M., Barbato, A. & Schwede, T. (*Bioinformatics* **2013**, 29, 2722–2728) — superposition-free Local Distance Difference Test (`lDDT`).
   - Varadi, M., Anyango, S., Deshpande, M. et al. (*Nucleic Acids Res.* **2022**, 50, D439–D444) & Varadi, M., Bertoni, D., Magana, P. et al. (*Nucleic Acids Res.* **2024**, 52, D368–D375) — AlphaFold Protein Structure Database (`AFDB` v4/v6, `CC-BY-4.0`), ModelArchive `_ma_qa_metric_local` mmCIF quality-assessment records, and `PAE` JSON schema.
   - Tunyasuvunakool, K. et al. (*Nature* **2021**, 596, 590–596) & Alderson, T. R., Pritišanac, I., Kolarić, Đ., Moses, A. M. & Forman-Kay, J. D. (*Proc. Natl. Acad. Sci. U.S.A.* **2023**, 120, e2304302120) — physical interpretation of `pLDDT < 50` disorder vs high-`pLDDT` conditionally folding segments in intrinsically disordered regions.
2. **Electrostatic Hydrogen-Bond Secondary Structure & Solution Rigidity Concordance (`T21.1`, `T21.3`)**:
   - Kabsch, W. & Sander, C. (*Biopolymers* **1983**, 22, 2577–2637) — partial-charge electrostatic $\text{C}'=\text{O}_i \cdots \text{H–N}_j$ hydrogen-bond criterion ($E < -0.5\text{ kcal mol}^{-1}$), $n$-turns ($n \in \{3, 4, 5\}$), $\beta$-bridge / $\beta$-ladder classification, and precedence `H > E > B > G > I > T > S > C`.
   - Adzhubei, A. A., Sternberg, M. J. E. & Makarov, A. A. (*J. Mol. Biol.* **2013**, 425, 2100–2132) — polyproline II (`PPII`) left-handed extended helix $(\phi, \psi)$ dihedral definition in non-hydrogen-bonded polypeptide segments.
   - Berjanskii, M. V. & Wishart, D. S. (*J. Am. Chem. Soc.* **2005**, 127, 14970–14971) — Random Coil Index (`RCI`, scaling constant `17.7` and ceiling `0.175`) and chemical-shift order parameter proxy `RCI-S²`.
   - Fowler, N. J., Sljoka, A. & Williamson, M. P. (*Nat. Commun.* **2020**, 11, 5982) & Fowler, N. J. & Williamson, M. P. (*Structure* **2022**, 30, 925–933) — inspiration for comparing structural model confidence/rigidity against NMR `RCI-S²` via Spearman rank correlation $\rho_s$ and RMSD offset scoring *(FERRIC implements an original `pLDDT`-to-rigidity heuristic rather than Fowler et al.'s `FIRST` constraint-network pebble game)*.
   - Robertson, A. J., Courtney, J. M., Shen, Y., Ying, J. & Bax, A. (*J. Am. Chem. Soc.* **2021**, 143, 19306–19310) & Zweckstetter, M. (*Protein Sci.* **2021**, 30, 2333–2337) — concordance of AlphaFold models with experimental solution NMR dipolar couplings and order parameters.
3. **3D Structure-to-Chemical-Shift Forward Physics & Firewalled Torsion Surfaces (`T21.2`)**:
   - Shen, Y. & Bax, A. (*J. Biomol. NMR* **2010**, 48, 13–22) — additive decomposition of backbone and $^{13}\text{C}^\beta$ chemical shifts into local $(\phi, \psi, \chi_1)$ conformation, hydrogen-bond geometry, aromatic ring-current fields, and peptide-group anisotropy (fitted on the fly to FERRIC's `torsion_corpus.tsv` with zero external parameter tables).
   - Haigh, C. W. & Mallion, R. B. (*Mol. Phys.* **1972**, 24, 955–970; *Prog. Nucl. Magn. Reson. Spectrosc.* **1979**, 13, 303–344) & Case, D. A. (*J. Biomol. NMR* **1995**, 6, 341–346) — signed triangular-projection aromatic ring-current shielding tensors across `Phe`, `Tyr`, `Trp`, and `His` $\pi$-systems.
   - Wagner, G., Pardi, A. & Bax, A. (*J. Am. Chem. Soc.* **1983**, 105, 5948–5949) & Pardi, A., Wagner, G. & Wüthrich, K. (*Eur. J. Biochem.* **1983**, 137, 445–454) — empirical inverse-cube $r_{\text{HO}}^{-3}$ amide proton and carbonyl $^{13}\text{C}'$ hydrogen-bond chemical-shift correlation.
   - McConnell, H. M. (*J. Chem. Phys.* **1957**, 27, 226–229); Buckingham, A. D. (*Can. J. Chem.* **1960**, 38, 300–307); Sitkoff, D. & Case, D. A. (*Prog. Nucl. Magn. Reson. Spectrosc.* **1998**, 32, 165–190); MacKerell, A. D. Jr. et al. (*J. Phys. Chem. B* **1998**, 102, 3586–3616) — peptide carbonyl magnetic susceptibility anisotropy, H-bond angular dependence, and CHARMM22 partial-charge bond-parallel electric-field shielding $\Delta\delta_{\text{EF}} = -A_\parallel E_\parallel$.
   - Golub, G. H., Heath, M. & Wahba, G. (*Technometrics* **1979**, 21, 215–223) — Generalized Cross-Validation (`GCV`) out-of-sample ridge predictive variance estimation.
   - Sahakyan, A. B., Vranken, W. F., Cavalli, A. & Vendruscolo, M. (*Angew. Chem. Int. Ed.* **2011**, 50, 9620–9623) — sidechain methyl ($^1\text{H}^{\text{m}}, {}^{13}\text{C}^{\text{m}}$ for `Ile`, `Leu`, `Val`, `Met`, `Ala`, `Thr`) $\gamma$-gauche rotamer and ring-current shift physics.
4. **Heavy-Tailed Convex Replacement Assignment Priors & `PAE`-Gated Amide NOE Connectivity (`T21.4`)**:
   - Lange, K. L., Little, R. J. A. & Taylor, J. M. G. (*J. Am. Stat. Assoc.* **1989**, 84, 881–896) — robust statistical modeling using heavy-tailed Student-$t$ likelihoods.
   - Wüthrich, K. (*NMR of Proteins and Nucleic Acids*, Wiley, **1986**) & Grzesiek, S., Wingfield, P., Stahl, S., Kaufman, J. D. & Bax, A. (*J. Biomol. NMR* **1995**, 5, 225–233) — `3D 15N-edited NOESY` $d_{\text{NN}}(i, j) \le 5.5\text{ \AA}$ amide proton connectivity in perdeuterated and sparse-carbon backbone assignment.
5. **Spectral `PAE` Domain Clustering, Generalized Degree of Order & Multi-Domain `RDC`/`PCS`/`PRE` Docking (`T21.5`)**:
   - Ng, A. Y., Jordan, M. I. & Weiss, Y. (*Adv. Neural Inf. Process. Syst.* **2001**, 14, 849–856) & Von Luxburg, U. (*Stat. Comput.* **2007**, 17, 395–416) — normalized symmetric graph Laplacian spectral clustering.
   - Tolman, J. R., Al-Hashimi, H. M., Kay, L. E. & Prestegard, J. H. (*J. Am. Chem. Soc.* **2001**, 123, 1416–1424) — Generalized Degree of Order ($\vartheta = \sqrt{\frac{2}{3}\sum_{\alpha,\beta} S_{\alpha\beta}^2}$) and inter-domain orientational dynamics detection from domain-specific Saupe order tensors.
   - Solomon, I. (*Phys. Rev.* **1955**, 99, 559–565); Battiste, J. L. & Wagner, G. (*Biochemistry* **2000**, 39, 5355–5365); Braddock, D. T. et al. (*J. Am. Chem. Soc.* **2001**, 123, 8634–8635); Clore, G. M. & Schwieters, C. D. (*J. Am. Chem. Soc.* **2004**, 126, 2923–2938); Iwahara, J., Schwieters, C. D. & Clore, G. M. (*J. Am. Chem. Soc.* **2004**, 126, 5879–5896) — exact SI Solomon–Bloembergen dipolar $\Gamma_2$ (`K_Solomon = 1.2339e16 Å^6 s^-2`), nitroxide tether allowance, and rigid-body multi-domain orientation + linker torsion refinement.
   - Bertini, I., Gupta, Y. K., Luchinat, C., Parigi, G., Peana, M., Sgheri, L. & Yuan, J. (*J. Am. Chem. Soc.* **2007**, 129, 12786–12794) — simultaneous translational and orientational determination of multi-domain protein architectures from lanthanide/metalloprotein $\Delta\boldsymbol{\chi}$ pseudocontact shifts.
   - Nilges, M. (*J. Mol. Biol.* **1995**, 245, 645–660) — sum-averaged $r_{\text{eff}} = (\sum r_{ab}^{-6})^{-1/6}$ ambiguous inter-domain NOE restraints.
6. **`CSP`/`PRE`-Guided Co-Folding Bridge (`AlphaFold 3` / `Boltz-2`) & Multi-Pose Complex Re-Scoring (`T21.6a`)**:
   - Abramson, J., Adler, J., Dunger, J. et al. (*Nature* **2024**, 630, 493–500) — AlphaFold 3 biomolecular complex prediction and public JSON input format specification (`dialect: "alphafold3"`, `version: 2`, sequence/ligand/covalent-bond schema without pocket conditioning).
   - Wohlwend, J., Corso, G., Passaro, S. et al. (*bioRxiv* **2024**, DOI: `10.1101/2024.11.19.624167`) & Passaro, S., Corso, G., Wohlwend, J. et al. (*bioRxiv* **2025**, DOI: `10.1101/2025.06.14.659707`) — MIT-licensed `Boltz-1` / `Boltz-2` biomolecular interaction & affinity architecture and declarative YAML `pocket` / `contact` restraint specification.
   - Shrake, A. & Rupley, J. A. (*J. Mol. Biol.* **1973**, 79, 351–371) — Fibonacci-sphere solvent-accessible surface area (`SASA`) quadrature separating surface binding-pocket residues from buried core allosteric relays.
   - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* **2013**, 73, 1–16) — chemical-shift perturbation binding site mapping and burial/contact enrichment.
7. **Bayesian Excited-State (`CEST` / `CPMG` / $R_{1\rho}$) Structural Hypothesis Engine (`T21.6b`)**:
   - Vallurupalli, P., Hansen, D. F. & Kay, L. E. (*Proc. Natl. Acad. Sci. U.S.A.* **2008**, 105, 11766–11771); Vallurupalli, P., Sekhar, A., Yuwen, T. & Kay, L. E. (*J. Biomol. NMR* **2017**, 67, 243–271); Bouvignies, G., Vallurupalli, P., Hansen, D. F. et al. (*Nature* **2011**, 477, 111–114) — excited-state chemical shifts $\varpi_B = \varpi_A + \Delta\varpi_{AB}$ from `CPMG` and `CEST` experiments.
   - Skrynnikov, N. R., Dahlquist, F. W. & Kay, L. E. (*J. Am. Chem. Soc.* **2002**, 124, 12352–12360) — signed vs unsigned $|\Delta\varpi_{AB}|$ dispersion likelihoods.
   - Schubert, M., Labudde, D., Oschkinat, H. & Schmieder, P. (*J. Biomol. NMR* **2002**, 24, 149–154) & Shen, Y. & Bax, A. (*J. Biomol. NMR* **2010**, 46, 199–204) — $^{13}\text{C}^\beta - {}^{13}\text{C}^\gamma$ ($\Delta_{\beta\gamma}$) *cis*- vs *trans*-Proline chemical-shift invariants.
   - Pelton, J. G., Torchia, D. A., Meadow, N. D. & Roseman, S. (*Protein Sci.* **1993**, 2, 543–558) — histidine imidazole $^{15}\text{N}^{\delta 1} / {}^{15}\text{N}^{\varepsilon 2}$ and $^{13}\text{C}^\gamma / {}^{13}\text{C}^{\delta 2}$ tautomer and protonation chemical shifts.
   - Wayment-Steele, H. K., Ojoawo, A., Otten, R. et al. (*Nature* **2024**, 625, 832–839) — subsampled/clustered MSA co-evolutionary conformer ensembles.
   - Schwarz, G. (*Ann. Stat.* **1978**, 6, 461–464) & Kass, R. E. & Raftery, A. E. (*J. Am. Stat. Assoc.* **1995**, 90, 773–795) — discrete-family marginalized Bayesian Information Criterion (`BIC`) and Bayes factor ($2\ln\mathrm{BF}$) hypothesis testing.

---

## 1. AlphaFold Ingestion, Kabsch–Sander `8 -> 4`-State Secondary Structure & NMR Concordance (`T21.1`)

### 1.1 Fail-Closed Coordinate Provenance & `PAE` JSON Ingestion (`nmr-core::af_concordance`)

1. **Fail-Closed `pLDDT` & Predicted-Model Provenance (`parse_af_structure_model`, `require_predicted_model`)**:
   - **Verified ModelArchive mmCIF `pLDDT` (`VerifiedMaQaMetricLocal`)**: Scans `loop_` `_ma_qa_metric_local` cross-referenced strictly against `_ma_qa_metric` where `name`/`type` explicitly denotes `pLDDT` (rejecting non-`pLDDT` local scores such as `QMEANDisCo`).
   - **Verified `AF-*` ModelArchive B-Factor Fallback**: Accepted without opt-in **only** when the coordinate header carries verified `AF-*` / `_ma_protocol_step` computational provenance (`0` experimental `EXPDTA` / `_exptl.method` / `CRYST1` crystal markers) **and** `B_iso_or_equiv` values lie in `[0, 100]` (or `[0, 1]` rescaled to `[0, 100]`) with non-zero variance (rejecting constant-`B` placeholders such as `0.00` or `50.00`).
   - **Fail-Closed Default (`require_predicted_model(&model, treat_as_model)`)**: Experimental crystal/NMR structures (`1UBQ`, including crystal files containing `REMARK 300 ... ALPHAFOLD` free-text comments), headerless `ATOM`-only PDBs, constant-$B$ PDBs, and non-`pLDDT` `_ma_qa_metric` mmCIFs classify as `Experimental` or `Unknown` (`plddt_per_residue = None`) and are **rejected (`Err`)** by `ferric af-validate`, `ferric assign --af-structure`, and `nmr-gui`'s `mol3d_viewer` unless the user explicitly passes `--treat-as-model` (`Treat B-factors as pLDDT`).
   - **Hostile-Input Guards**: Pre-allocation coordinate size cap (`<= 64 MiB`), strict `.is_finite()` checks on all $(x, y, z)$ and `pLDDT` values, and explicit `Err` on truncated mmCIF loops or `NaN`/`Inf` entries (`0` production `.unwrap()`/`.expect()`).

2. **Multi-Dialect `PAE` JSON $N \times N$ Parser (`parse_pae_json_str` -> `PaeMatrix`)**:
   - Supports `AFDB` v3/v4/v6 (`predicted_aligned_error`), legacy sparse `AFDB` v1/v2 (`residue1`/`residue2`/`distance`), `AF2`/`ColabFold` (`pae` + `max_pae` + `plddt`), and `AlphaFold 3` `confidences.json` (`pae` + `token_chain_ids` + `token_res_ids`).
   - **Multi-Chain AF3 Disambiguation**: When an AF3 `confidences.json` contains $> 1$ distinct `token_chain_ids` and no explicit `--chain <asym_id>` is supplied, `parse_pae_json_str_for_chain` fails closed with `Err` listing the available chain IDs rather than silently slicing the first chain.
   - **ColabFold `plddt` Cross-Check**: When `pae.json` carries a `plddt` vector alongside structure-file `pLDDT`s, `validate_pae_plddt_consistency` verifies residue count and max per-residue deviation $\le 1.0$ `pLDDT` unit (`Err` on mismatched model/PAE pairs).
   - Pre-allocation dimension guard `1 <= N <= MAX_PAE_RESIDUES` (`4,096`, `<= 64 MiB`), verifying square matrix lengths, finite values $\in [0.0, 64.0]\text{ \AA}$, and $N == L_{\text{chain}}$.

3. **Opt-In Rate-Limited AFDB CLI Fetcher (`nmr-cli::af_validate_cmd`)**:
   - `ferric af-validate --fetch-uniprot <UNIPROT_ID> --cache-dir data/private/afdb/` runs strictly in `nmr-cli` (`0` network I/O in `nmr-core`), enforcing `[A-Z0-9]{6,10}` validation, `https://alphafold.ebi.ac.uk` HTTPS allowlist, `<= 1 req/s` rate limiting, path-traversal rejection, and writing `PROVENANCE.md` with `AFDB` version, `CC-BY-4.0` attribution, and `FIPS 180-4` `SHA-256` digests.

### 1.2 Clean-Room Kabsch–Sander (1983) `8 -> 4`-State Secondary-Structure Classifier (`nmr-core::kabsch_sander_ss`)

1. **Amide Proton Reconstruction & Partial-Charge Electrostatic H-Bond Energy**:
   - Reconstructs missing *trans*-peptide amide protons $\mathbf{r}_{\text{H}_j} = \mathbf{r}_{\text{N}_j} + 1.00\text{ \AA}\cdot\widehat{(\mathbf{r}_{\text{C}'_{j-1}} - \mathbf{r}_{\text{O}_{j-1}})}$ when $d(\text{C}'_{j-1}, \text{N}_j) \le 2.5\text{ \AA}$ and evaluates Kabsch–Sander electrostatic interaction energy ($q_1 = 0.42, q_2 = 0.20$, `r_min >= 0.5 Å` clash floor) for $|i - j| \ge 2$:
     $$E_{\text{HB}}(i \to j) = 332\,q_1 q_2 \left(\frac{1}{r_{\text{O}_i\text{N}_j}} + \frac{1}{r_{\text{C}'_i\text{H}_j}} - \frac{1}{r_{\text{O}_i\text{H}_j}} - \frac{1}{r_{\text{C}'_i\text{N}_j}}\right)\text{ kcal mol}^{-1}$$
     retaining the two strongest H-bonds per donor and acceptor satisfying $E_{\text{HB}} < -0.5\text{ kcal mol}^{-1}$.
2. **Canonical 8-State Precedence & Deterministic `8 -> 4`-State NMR Reduction**:
   - Enforces the exact Kabsch & Sander (*Biopolymers* **1983**, 22, 2577) summary precedence **`H > E > B > G > I > T > S > C`** so extended $\beta$-ladders (`E`) strictly outrank isolated $\beta$-bridges (`B`), and demotes any orphan `< 3`-residue `G` ($3_{10}$) or `< 5`-residue `I` ($\pi$) fragment remaining after higher-priority overwrites to H-bonded turn `T`.
   - Maps `KabschSanderState8` + $(\phi_i, \psi_i)$ into `SecondaryStructure4`: **`Helix` (`H`)** `{H, G, I}`, **`Sheet` (`E`)** `{E, B}`, **`Ppii` (`P`)** non-H-bonded `{T, S, C}` in $\phi \in [-105^\circ, -45^\circ], \psi \in [+115^\circ, +175^\circ]$ (Adzhubei et al. **2013**), and **`Coil` (`C`)** remaining `{T, S, C}`.

### 1.3 Original `pLDDT` vs `RCI-S²` / Lipari–Szabo $S^2$ Concordance & 6-Class Gap-Free Taxonomy

1. **Rigidity Source Distinction & Berjanskii–Wishart (`2005`) `RCI-S²` Scaling**:
   - Explicitly records `RigiditySource::ExperimentalRelaxationS2` when Lipari–Szabo $S^2_i$ (`--s2 <modelfree.tsv>`) is supplied vs `RigiditySource::ShiftDerivedRci` (**`RCI-S²`**, never labelled `"experimental S²"`) when derived from chemical shifts.
   - Uses the canonical Berjanskii & Wishart (*JACS* **2005**, 127, 14970) `RCI` scaling constant **`17.7`** (`RCI_i <= 0.175` ceiling) and applies `[0.25, 0.50, 0.25]` 3-point smoothing **strictly across contiguous assigned neighbors** (`seq_id` step `1`), never averaging across unassigned sequence gaps.
2. **Original Rigidity Mapping, Ordered-Core Correlation & Block-Bootstrap 95% CIs**:
   - Maps $\text{pLDDT}_i \in [0, 100]$ via FERRIC's original heuristic $\hat{S}^2_{\text{AF},i} = \operatorname{clamp}\!\bigl((\text{pLDDT}_i - 35.0) / 55.0,\; 0.0,\; 1.0\bigr)$ and reports:
     - `spearman_rho: Option<f64>` (`None` rather than `0.0` whenever either vector has zero variance, e.g. constant `pLDDT`),
     - `spearman_rho_ordered: Option<f64>` restricted to predicted-ordered core residues ($\text{pLDDT}_i \ge 70.0$) so terminal disorder contrast does not inflate global $\rho_s$,
     - `spearman_ci95` and `rmsd_rig_ci95` via deterministic 500-replicate moving-block bootstrap (block length `5` residues).
3. **6-Class Gap-Free Per-Residue Concordance Taxonomy (`AfNmrConcordanceClass`)**:
   With explicit single thresholds `PLDDT_ORDERED_CUTOFF = 70.0` and `S2_ORDERED_CUTOFF = 0.65` (eliminating any dead zone on $S^2 \in [0.60, 0.72)$):
   - **`ConcordantOrdered`**: $\text{pLDDT}_i \ge 70.0$, $S^2_i \ge 0.65$, no `Helix <-> Sheet` flip, and circular dihedral divergence $\Delta\theta_i < 60^\circ$ (when `TorsionDb` confidence is `Good`).
   - **`ConcordantDisordered`**: $\text{pLDDT}_i < 70.0$ and $S^2_i < 0.65$.
   - **`ModelOrderedSolutionDisordered`** *(serde alias `ConditionalFoldingOrCrystalArtifact`)*: $\text{pLDDT}_i \ge 70.0$ and $S^2_i < 0.65$ (high-confidence model helix/sheet or loop that is dynamically disordered in solution, e.g. conditionally folding $\alpha$-MoRFs).
   - **`DynamicHingeOrAltState`**: Either ($\text{pLDDT}_i < 70.0$ and $S^2_i \ge 0.65$) or ($\text{pLDDT}_i \ge 70.0$, $S^2_i \ge 0.65$, and `Helix <-> Sheet` flip or $\Delta\theta_i \ge 60^\circ$).
   - **`Indeterminate`**: Assigned NMR shifts present, but model lacks verified `pLDDT` (`plddt_i = None`, e.g. X-ray `1UBQ` without `--treat-as-model`; never circularly defaulted to `85.0`).
   - **`UnassignedNmr`**: Fewer than `2` assigned backbone shifts and no experimental $S^2_i$.
4. **CLI Contract**:
   `ferric af-validate --structure <model.cif|model.pdb> [--treat-as-model] [--pae <pae.json>] [--chain <id>] --shifts <assigned.str|assigned.nef> [--s2 <modelfree.tsv>] [--fetch-uniprot <id> --cache-dir <dir>] [--exclude-pdb <ids>] [--exclude-bmrb <ids>] [-o <out.json|out.tsv|report.html>]`

---

## 2. Clean-Room 3D Structure-to-Chemical-Shift Forward Predictor (`T21.2`, `nmr-core::struct_shifts`)

### 2.1 Additive Physical & Firewalled 93-Basis Fourier Shift Decomposition

For each residue $i \in \{0, \dots, L-1\}$ and backbone nucleus $a \in \{\text{H}^{\text{N}}, \text{N}, \text{C}^\alpha, \text{C}^\beta, \text{C}', \text{H}^\alpha\}$ plus optional `ILVMAT` methyls:

$$\delta^{\text{struct}}_{i,a}(\mathbf{X}) = \delta^{\text{RC}}_{i,a}(T, \text{pH}) + \Delta\delta^{\text{tor}}_{i,a}(\phi_i, \psi_i, \phi_{i-1}, \psi_{i-1}, \phi_{i+1}, \psi_{i+1}, \chi_{1,i}, \omega_i, \omega_{i-1}) + \Delta\delta^{\text{HB}}_{i,a}(\mathbf{X}) + \Delta\delta^{\text{RC-arom}}_{i,a}(\mathbf{X}) + \Delta\delta^{\text{pep/EF}}_{i,a}(\mathbf{X})$$

1. **Harmonized Canonical `(CA, phi, psi, chi1)` H-Bond Geometry (`21.W1:P2-3`)**:
   - Because `torsion_corpus.tsv` reconstructs peptide planes from $(\text{C}^\alpha_i, \phi_i, \psi_i, \chi_{1,i})$, `canonicalize_model_for_hbond` evaluates H-bond shifts on the harmonized canonical geometry during both corpus training subtraction and query inference (`1UBQ` query vs `1UBQ_A` corpus `15N` H-bond RMS mismatch reduced from `0.154 ppm` to `0.00439 ppm`).
2. **93-Dimensional Trigonometric Fourier Ridge Basis Across All 6 Backbone Nuclei**:
   - Evaluates the identical **`93`-feature** design vector across all 6 backbone nuclei (`TOTAL_BASIS_DIM = 93` = `49` `7×7` order-3 $(\phi_i, \psi_i)$ tensor-product Fourier harmonics + `10` order-2 $(\phi_{i-1}, \psi_{i-1})$ harmonics + `4` order-1 $(\phi_{i+1}, \psi_{i+1})$ harmonics + `6` $\chi_{1,i}$ harmonics + `4` Kabsch–Sander H-bond & secondary-structure indicators + `20` amino-acid indicators), solved via Cholesky ridge regression ($\lambda = 2.5$).
3. **Fail-Closed 4-Layer `HomologFirewall` (`<= 25%` Seq-ID, `TM-score < 0.50`)**:
   - Filters `torsion_corpus.tsv` through `L1`–`L4` before fitting. In `L4`, query chains with `< 8` resolved $\text{C}^\alpha$ coordinates fail closed (`Err`) rather than skipping structural similarity screening, while short corpus chains (`< 8` $\text{C}^\alpha$) are skipped individually and tallied in `StructShiftPredictionReport::l4_short_templates_skipped`.
4. **Haigh–Mallion (`1972`) Ring Currents, Empirical $r_{\text{HO}}^{-3}\cos^2\theta_{\text{NHO}}$ H-Bonds & CHARMM22 Electric Fields**:
   - Evaluates signed Haigh–Mallion (`1972`) / Case (`1995`) conjugated ring currents within $12.0\text{ \AA}$, empirical FERRIC physics-inspired inverse-cube $r_{\text{HO}}^{-3}\cos^2\theta_{\text{NHO}}$ hydrogen-bond shifts (Wagner, Pardi & Bax **1983**; Sitkoff & Case **1998**), McConnell (`1957`) carbonyl susceptibility anisotropy, and Buckingham (`1960`) / Sitkoff–Case (`1998`) bond-parallel electric-field shifts parameterized with **CHARMM22** backbone partial charges ($q_{\text{N}} = -0.47, q_{\text{H}} = +0.31, q_{\text{C}'} = +0.51, q_{\text{O}} = -0.51, q_{\text{C}\alpha} = +0.07, q_{\text{H}\alpha} = +0.09$; MacKerell et al. **1998**).
5. **Golub–Heath–Wahba (`1979`) GCV Out-of-Sample $\hat{\sigma}_{\text{struct},a}$ & Monotonic Low-`pLDDT` Widening**:
   - Computes out-of-sample predictive error $\hat{\sigma}_{\text{struct},a}$ via Golub–Heath–Wahba (`1979`) Generalized Cross-Validation (`GCV` LOO PRESS from exact Cholesky inverse diagonal) and sets $\sigma_{\text{coil},a} = \max(\text{coil\_sigma\_ppm}_a,\; 1.15\,\hat{\sigma}_{\text{struct},a})$ (`coil_sigma_ppm`: `H 0.62, N 3.10, CA 1.55, CB 1.65, C' 1.50, HA 0.36 ppm` $\ge$ `struct_sigma_ppm`) in
     $$\hat{\delta}_{i,a} = w_{\text{AF},i}\,\delta^{\text{struct}}_{i,a} + (1 - w_{\text{AF},i})\,\delta^{\text{RC}}_{i,a}, \qquad \hat{\sigma}^2_{i,a} = w_{\text{AF},i}^2\,\hat{\sigma}^2_{\text{struct},a} + \bigl(1 - w_{\text{AF},i}^2\bigr)\,\sigma^2_{\text{coil},a}$$
     guaranteeing predictive uncertainty $\hat{\sigma}_{i,a}(\text{pLDDT})$ widens monotonically as `pLDDT` decreases (`pLDDT 95 -> 25`).

---

## 3. Consolidated `Structure` Mode, Vector Secondary-Structure Ribbon & 3D `pLDDT`/Concordance Putty (`T21.3` & `S9`)

Inside `ferric-gui`, all 3D coordinate, Kabsch–Sander secondary-structure, `PAE` domain, co-folding, and excited-state tools live inside the single consolidated top-level **`Structure`** mode (`0` duplicate AlphaFold entry points, `0` emoji characters):

1. **Three-Track Vector Secondary-Structure & Concordance Ribbon (`nmr-gui::ss_ribbon_widget`)**:
   - **Track 1 (`3D SS`)**: Kabsch–Sander coral helices (`#E05A47`), teal $\beta$-strand arrows (`#2A9D8F`), `PPII` diamond ticks (`#7B61FF`), and coil baseline.
   - **Track 2 (`NMR SS`)**: Live solution NMR `SSP` propensity bars (`nmr-core::idp_propensity`).
   - **Track 3 (`AF vs NMR`)**: Renders all 6 `AfNmrConcordanceClass` states (`ConcordantOrdered` `#16A34A`, `ConcordantDisordered` `#64748B`, `ModelOrderedSolutionDisordered` hatched `#D97706`, `DynamicHingeOrAltState` double-bordered `#DC2626`, `Indeterminate` neutral slate-outline when `pLDDT` is absent, and `UnassignedNmr` hollow outline).
2. **`mol3d_viewer` & Cached `PAE` Heatmap (`nmr-gui::integrative_af_panel`)**:
   - Enforces `require_predicted_model` before enabling `AlphaFoldPlddt` 4-band coloring (`#0053D6` / `#65CBF3` / `#FFDB13` / `#FF7D45`), and caches the $N \times N$ `PAE` matrix as a single `egui::TextureHandle` (`1` GPU upload across static frames).

---

## 4. Opt-In 3D-Prior-Assisted Backbone & `PAE`-Gated `15N-NOESY` Assignment (`T21.4`)

### 4.1 Bit-Exact Sequence-Only Isolation & Fail-Closed Provenance (`nmr-core::af_assign_priors`)

1. **Bit-Exact `5c4e87d` Sequence-Only Shader Preserved**: `crates/nmr-gpu/src/shaders/scoring.wgsl` keeps the sequence-only `@group(0)` `compute_unary` kernel byte-for-operation identical to `5c4e87d` (`181/189` Ubiquitin `--realistic`, `57/59` `CISD3` WebGPU + 12 golden `U`/`P` snapshots). Opt-in AF unary scoring executes in a separate `@group(1)` `compute_unary_af` entry point dispatched only when `af_priors.is_some()`.
2. **Fail-Closed `--treat-as-model` Gate**: `ferric assign --af-structure <path>` calls `nmr_core::af_concordance::require_predicted_model(&model, treat_as_model)` (`!treat_as_model`), rejecting crystallographic or headerless PDBs unless `--treat-as-model` is explicitly passed.
3. **Uncalibrated Posterior Disclosure (`uncalibrated_af_prior = true`)**: Because the 10-bin PAV isotonic lookup table in `qc.rs` was calibrated exclusively on sequence-only scores, `af_priors.is_some()` bypasses PAV remapping and sets `DatasetQcReport::confidence_calibrated = false` and `uncalibrated_af_prior = true`, emitting `"UNCALIBRATED (AF 3D prior enabled — isotonic table trained on sequence-only)"` in CLI, JSON, NEF, and HTML outputs.

### 4.2 Convex Per-Nucleus Log-Linear Replacement Algebra (Zero Double-Counting)

Let `seq_offsets` be the single unlabelled reference offset vector estimated by `estimate_reference_offsets` (`0` separate `af_shift_offsets` gauge drift), and let `AF_SIGMA_OOS_PPM` (`H: 0.52, N: 2.80, CA: 1.05, CB: 1.20, C: 1.05 ppm`) floor the `T21.2` predictive uncertainty $\sigma^{\text{eff}}_{k,a} = \max(\hat{\sigma}_{k,a}, \sigma^{\text{OOS}}_a)$. For each observed nucleus $a \in \mathcal{O}_s$, `af_assign_priors` evaluates both the sequence-prior log-odds $\ell_{\text{seq}}(s, k, a)$ and the clamped Student-$t(\nu = 4)$ AF log-odds $\ell_{\text{AF}}(s, k, a)$ against the **identical** `Priors::background` null density $\ell_{\text{bg},a}$, and forms a **convex replacement** weighted by $\omega_r = \operatorname{clamp}\!\bigl(\lambda_{\text{AF}}\,w_{\text{AF}}(\text{pLDDT}_k),\; 0.0,\; 1.0\bigr)$:

$$U^{\text{AF}}_{s,k} = \left[\sum_{a \in \mathcal{O}_s} \Bigl((1 - \omega_r)\,\ell_{\text{seq}}(s, k, a) + \omega_r\,\operatorname{clamp}\bigl(\ell_{t_4}(z^{\text{AF}}_{s,k,a}) - \ell_{\text{bg},a},\; -U_{\max}^{\text{clamp}}, +U_{\max}^{\text{clamp}}\bigr)\Bigr)\right] - \ln(n_{\text{pos}})$$

- When $\omega_r = 0$ (`--af-weight 0` or `pLDDT -> 0`), $U^{\text{AF}}_{s,k} \equiv U^{\text{seq}}_{s,k}$ identically.
- When $\omega_r = 1$, the AF likelihood cleanly **replaces** the sequence BMRB Gaussian for that nucleus rather than double-counting `AA_k` identity, applying `-ln(n_pos)` exactly once per spin system.
- **Scope-Limit Note on Sparse `HNCA`-Only Loop Errors**: On full 7-spectrum triple-resonance data, corrupting a 12-residue loop (`res 40..52`) by $120^\circ$ at overconfident `pLDDT = 95` causes `0` accuracy loss (`349/360 -> 352/360` `standard`, `349/360 -> 350/360` `evo-graph` `[synthetic self-consistency]`), whereas on sparse `CB`-stripped `HNCA`-only data overall accuracy rises (`6.82% -> 11.93%`) while local inside-loop (`40..52`) accuracy drops (`7/60 -> 1/60` `[synthetic self-consistency]`); users on sparse `HNCA`-only datasets should inspect suspect loops via `ferric af-validate`.

### 4.3 `PAE <= 6.0 Å` Gated `3D 15N-NOESY` Amide Contacts & Symmetric Site Cap

- Admits predicted $d(\text{H}^{\text{N}}_j, \text{H}^{\text{N}}_k) \le 5.5\text{ \AA}$ contacts only when $\text{pLDDT}_j, \text{pLDDT}_k \ge 70$ and symmetrized $\overline{\text{PAE}}(j, k) \le 6.0\text{ \AA}$ (`|j - k| <= 5` fallback when `PAE` JSON is omitted).
- `--noe-weight` (`cfg.noe_weight`) scales both sequential $\Delta P^{\text{NOE}}$ and medium/long-range factor-graph contacts (`--noe-weight 0.0` yields `0.0` NOE contribution), and total per-site pairwise NOE contribution is symmetrically bounded to `[-NOE_MAX_SITE_BONUS_NATS, +NOE_MAX_SITE_BONUS_NATS]` (`4.0 nats`) in `evo-graph` belief propagation and consensus SA.
- CLI: `ferric assign --af-structure <model.cif|pdb> [--treat-as-model] [--af-pae <pae.json>] [--noesy-15n <noesy.tsv|nef>] [--af-weight 0.65] [--noe-weight 1.0] [--af-clamp-nats 4.0] ...`

---

## 5. `PAE` Spectral Domain Clustering & Multi-Domain `RDC`/`PCS`/`PRE` Docking (`T21.5`)

### 5.1 Discontinuous Rigid-Domain Master-Frame Coordinates & Strict Linker Exclusion (`nmr-core::domain_dock`)

1. **Symmetrized `PAE` Normalized Laplacian Clustering**: Partitions ordered residues ($\text{pLDDT}_i \ge 70.0$) via cyclic Jacobi eigendecomposition of $\mathbf{L}_{\text{sym}} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ ($\overline{\text{PAE}}_{\text{intra}} \le 5.0\text{ \AA}$, $|\mathcal{D}_d| \ge 20$).
2. **Master-Frame Discontinuous Domain Geometry (`extract_rigid_domain_coords`)**: Slices rigid-domain atom coordinates directly from the input 3D master frame rather than re-integrating through `NeRF`, guaranteeing **`0.000 Å` intra-domain RMSD drift** even for discontinuous multi-segment domains (e.g. residues `1..40` + `90..130` clasping an inserted domain `45..85`).
3. **Chain-Qualified `(chain_id, seq_id)` Keys & Strict Linker/Tail Exclusion**: All `RDC`, `PCS`, `PRE`, and `NOE` restraints bind via `(chain_id, seq_id)` tuples (never colliding homodimer residue numbers across chains `A` and `B`). Flexible linker, disordered terminal tail, and low-confidence ($\text{pLDDT} < 70$) residues are **strictly excluded** from both working Saupe SVD tensor fits and `ByPeptidePlane` `RdcFree` holdouts so rigid-body tensors are never corrupted by fast-averaging linkers.
4. **Pre-Dock Intra-Domain `Q_free <= 0.50` Gate & Genuine `20%` `PCS` $Q_{\text{free}}$**: Evaluates intra-domain `ByPeptidePlane Q_free` before docking (`WarnProceed` / `Abstain` when $> 0.50$), and stratifies `PCS` restraints `20%` (`1`-in-`5` per domain) into a genuine held-out `pcs_q_free: Option<f64>` (`None` when `< 5` `PCS` rows exist; never aliasing `pcs_q_work`).

### 5.2 Exact CODATA SI Solomon `PRE`, `7.0 Å` `PCS` Metal VarPro, Nilges Ambiguous `NOE` & `O(L)` GPU `vdW`

1. **Exact SI Solomon–Bloembergen Dipolar `PRE` (`SOLOMON_K_PROTON_NITROXIDE_A6_S2 = 1.2339e16`)**:
   - Evaluates the exact CODATA 2018 SI dipolar constant for $^1\text{H}$ + $S = 1/2$ nitroxide ($g_e = 2.002319$):
     $$K_{\text{Solomon}} = \frac{1}{15}\left(\frac{\mu_0}{4\pi}\right)^2 \gamma_H^2 g_e^2 \mu_B^2 S(S+1) = 1.2339 \times 10^{16}\text{ \AA}^6\text{ s}^{-2}, \qquad \Gamma_{2,\text{calc}}(r) = \frac{K_{\text{Solomon}}}{r^6}\left(4\tau_c + \frac{3\tau_c}{1 + \omega_H^2\tau_c^2}\right)$$
   - Evaluates unsoftened $r^{-6}$ for $r \ge r_{\min} = 3.5\text{ \AA}$ (smooth clash cap only below $3.5\text{ \AA}$), **requires** explicit `tau_c_ns` (`--pre-tau-c-ns`) and `proton_mhz` (`--pre-proton-mhz`) whenever `--pre` is supplied (`Err` if omitted), and supports `--mtsl-tether-allowance-a` (`default = 4.0 Å` flat-bottom nitroxide rotamer allowance; Iwahara, Schwieters & Clore **2004**).
2. **`PCS` Metal Anchor (`<= 7.0 Å`) & 8-Parameter VarPro Refinement**: Requires an explicit `# metal_anchor X Y Z` header or `--pcs-metal-xyz X,Y,Z` within `7.0 Å` of a coordinating residue (`Err` otherwise; never places a lanthanide at a domain centroid) and refines $(\mathbf{r}_{\text{M}}, \Delta\boldsymbol{\chi})$ via bounded `1.5 Å` 8-parameter `VarPro` SVD (`T11.4`).
3. **Nilges (`1995`) Ambiguous `NOE`s & `O(L)` Voxel-Grid WebGPU `vdW`**: Aggregates duplicate `restraint_id` rows via $r_{\text{eff}} = (\sum_{a,b} r_{ab}^{-6})^{-1/6}$, and evaluates inter-domain excluded volume in `crates/nmr-gpu/src/shaders/domain_dock.wgsl` via an `O(L)` `32×32×32` spatial hash grid (`4.5 Å` cells, `27`-cell stencil; `< 1e-5` CPU/WebGPU parity).
4. **CLI Contract**:
   `ferric structure --pae-domains --af-structure <model.cif> --af-pae <pae.json> [--rdc <rdcs.tsv|nef>] [--pcs <pcs.tsv> --pcs-metal-xyz X,Y,Z] [--pre <pre.tsv> --pre-tau-c-ns <ns> --pre-proton-mhz <mhz> [--mtsl-tether-allowance-a 4.0]] [--noesy <inter_noe.tsv>] [--rdc-free-frac 0.20] [--ensemble-size 10] [--engine auto|gpu|cpu] -o <docked.pdb|docked.cif>`

---

## 6. `CSP`/`PRE`-Guided Co-Folding Bridge (`AF3` / `Boltz-2`) & Multi-Pose Complex Re-Scoring (`T21.6a`)

### 6.1 Fail-Closed Pocket Extraction, Solomon `PRE` Conversion & Schema-Valid Export (`nmr-core::cofold_bridge`)

1. **Fail-Closed Apo SASA & `10.0 Å` Spatial Cluster Filter (`extract_nmr_interface_restraints`)**:
   - Refuses to export `CSP` pocket restraints when `apo_structure` is `None` unless `--allow-unfiltered-no-apo` is explicitly set (preventing buried hydrophobic core shifts from becoming surface `pocket` constraints).
   - Filters `2\sigma` surface `CSP` hits ($\text{rSASA}_i \ge 0.20$) through a `10.0 Å` single-linkage $\text{C}^\alpha$ graph clusterer, retaining the dominant contiguous binding patch while reclassifying isolated distal surface shifts ($> 10.0\text{ \AA}$) as `DistalSurfaceOutlier`.
   - Converts `PRE` $\Gamma_2$ rows via exact SI Solomon–Bloembergen physics (`K_Solomon = 1.2339e16 Å^6 s^-2`, requiring `--tau-c-ns`; `Err` if omitted) plus nitroxide tether allowance, and **drops** bleached/weak bounds exceeding `max_contact_distance` (`20.0 Å`) with a diagnostic rather than clamping them to false `20.0 Å` attractions.
2. **Needleman–Wunsch FASTA/PDB Re-Indexing & Schema Guards (`export_af3_json`, `export_boltz2_yaml`)**:
   - Aligns each receptor PDB/mmCIF chain against the full-length export FASTA via gap-preserving Needleman–Wunsch alignment (`>= 80%` identity over observed PDB residues) so N-terminal tags or crystallographic loop gaps never shift 1-based `Boltz-2` token indices (`re_index_pose_to_fasta` applies the inverse mapping during `score_cofold_poses`).
   - Rejects atom-named `contact` tokens on `SMILES`-defined ligands (`Err` directing users to residue-level `pocket` constraints or `CCD` ligands), validates single `ccd` codes (`Err` on comma-separated strings), enforces `Boltz-2` `max_distance in [4.0, 20.0] Å`, and states candidly that `AlphaFold 3` JSON (`version: 2`) carries sequence/ligand/covalent inputs only (no pocket conditioning; NMR restraints live in `<prefix>.nmr_restraints.json` for post-prediction `ferric cofold score`).

### 6.2 Multi-Pose Complex NMR Re-Scorer (`score_cofold_poses`, `BurialAndContactOnly` Default)

1. **`CspScoringMode::BurialAndContactOnly` Default (`A7`)**:
   - By default, `S_CSP(m)` evaluates only validated geometric observables: heavy-atom interface contact ($d_{\min} \le 5.0\text{ \AA}$) ROC-AUC on `2\sigma` surface `CSP` pocket hits combined with Spearman $\rho_s\bigl(\Delta\delta^{\text{obs}}_{\text{CSP}}, \Delta\text{rSASA}_i(m)\bigr)$. Uncalibrated intermolecular ligand ring-current + H-bond shift summation is gated behind explicit `--experimental-csp-ring-hbond` (`CspScoringMode::ExperimentalRingAndHBond`).
2. **Candid Decoy-Rejection Scope Note**:
   - On non-circular multi-pose benchmarks, **`CSP`-only** scoring ranks near-native poses `#1` when a native pose is present in the pool, whereas **decoy-only** rejection (`PoseVerdict::InconclusiveOrAllViolated`) requires distance or orientational restraints (`NOE`/`PRE`/`RDC`/`PCS`): under `CSP`-only, decoy poses sharing the same surface patch remain admissible (`6/9` decoy-only runs admissible vs `0/9` with `CSP + intermolecular NOE` `[synthetic self-consistency]`).

---

## 7. Bayesian Excited-State (`CEST`/`CPMG`) Structural Hypothesis Engine, Studio & `MCP` (`T21.6b`)

### 7.1 Discrete-Scan Marginalized `BIC`, Experimental-Noise `H0` & Scope Limits (`nmr-core::excited_state`)

1. **Input Integrity & Distinct-Residue Floor (`MIN_DISTINCT_EXCHANGING_RESIDUES = 4`)**:
   - Rejects duplicate `(chain_id, seq_id, atom_name)` rows across single or multiple `--exchange` files (`Err`, preventing $2\ln\mathrm{BF}$ inflation by repeating a file) and requires $\ge 4$ **distinct exchanging residues** (`n_distinct_residues >= 4`) before evaluating `Committed` verdicts (4 nuclei on 1 residue abstain as `AmbiguousOrUnmodeledState`).
   - Requires pre-filtered statistically significant `CPMG` dispersions by default (`cpmg_prefiltered_significant_only = true`; raw unfiltered rows require `--cpmg-unfiltered` with $|\Delta\varpi|/\sigma_{\text{exp}} \ge 3.0$).
2. **Discrete-Scan Marginalized Likelihood & `BIC` (`n = n_distinct_residues`)**:
   - Because $H_1$ (`LocalUnfolding` windows $[u, v]$), $H_2$ (`XxxProCisTrans` Proline sites), $H_3$ (`RotamerOrHisTautomerFlip` rotamer/His states), and $H_4$ (`Alternate3DConformer` `--alt-structures`) scan discrete candidate sets of size $M_H \ge 1$ with **$k_H = 0$ fitted continuous parameters**, `excited_state` evaluates the uniform-prior **marginalized log-likelihood** via log-sum-exp rather than taking $\ell_{\max}(H)$ with ad-hoc $k_H \in \{1, 2\}$:
     $$\ln\bar{L}_H = \ln\!\left(\frac{1}{M_H}\sum_{m=1}^{M_H} \exp\bigl(\ell(H, m)\bigr)\right), \qquad \mathrm{BIC}(H) = -2\ln\bar{L}_H + k_H\ln n_{\text{distinct\_residues}}\quad (k_H = 0)$$
     naturally penalizing large discrete hypothesis spaces ($-\ln M_H$ Occam factor, invariant to appending duplicate `--alt-structures` conformers!).
3. **Experimental-Noise-Only Null `H0` & `UnmodeledSignificantExchange`**:
   - Evaluates `H0` ($\Delta\varpi_{AB} = 0$) strictly against experimental measurement uncertainty $\sigma_{\text{exp},i,a}$ (`0` forward-model $\sigma_{\text{fwd}}$ inflation on `H0`). When `H0` is strongly rejected ($2\ln\mathrm{BF}_{\text{best},H_0} \ge 10$ or $\chi^2_{\nu,\text{exp}}(H_0) > 4.0$) yet every physical candidate $H_1..H_4$ fits poorly ($\chi^2_\nu(H_{\text{best}}) > 2.5$), `classify_excited_state` emits **`ExcitedStateVerdict::UnmodeledSignificantExchange`** rather than letting `H0` win or force-committing a wrong mechanism.
   - Requires head-to-head $2\ln\mathrm{BF}_{1,\text{next\_non\_null}} \ge 6.0$ against the runner-up physical model even when only one non-null family beats `H0`.
4. **Corrected Pelton (`1993`) `His` `CG` Sign, Sidechain-`15N` Eligibility & Unsigned-`CPMG` Direction Guard**:
   - Corrects `His` $\mathrm{N}^{\delta 1}\text{–H} \to \mathrm{N}^{\varepsilon 2}\text{–H}$ differential shifts to Pelton et al. (*Protein Sci.* **1993**, 2, 543, Table 1): $\Delta\delta_{\text{CG}} = \mathbf{+8.3\text{ ppm}}$ (`127.8 -> 136.1 ppm`), $\Delta\delta_{\text{CD2}} = -7.2\text{ ppm}$ (`124.2 -> 117.0 ppm`), $\Delta\delta_{\text{ND1}} \approx -82\text{ ppm}$, $\Delta\delta_{\text{NE2}} \approx +82\text{ ppm}$; enables `His` tautomer/protonation sub-hypotheses whenever sidechain `ND1`/`NE2` rows are present even without `CD2`/`CG`; and returns `HisDirectionality::AmbiguousDirectionOnUnsignedCpmg` when all `His`-proximal rows are unsigned `CPMG` magnitudes.
5. **Explicit Biophysical Scope Limit (Endpoint-Only $\Delta\varpi_{AB}$)**:
   - `nmr-core::excited_state` scores pure 2-state chemical-shift endpoint differences $\Delta\varpi_{AB} = \varpi_B - \varpi_A$ (`H1` full local coil `[u, v]`, `H2` full *cis/trans* switch, `H3` discrete rotamer/tautomer state, `H4` supplied 3D conformer). It fits **zero** fractional partial-unfolding amplitude $\alpha \in (0, 1)$ and **zero** excited-state population $p_B$ or exchange rate $k_{\text{ex}}$ (which are fitted upstream by `nmr-core::{cest, cpmg}`); partial helix fraying or unmodeled multi-state intermediates surface honestly as `AmbiguousOrUnmodeledState` or `UnmodeledSignificantExchange`.

### 7.2 Hardened Read-Only `MCP` Tools (`nmr-core::mcp`, `nmr-gui::mcp_bridge`)

- **Pre-Read `symlink_metadata` & Resource Caps**: All Phase 21 MCP file readers (`af_concordance`, `predict_structure_shifts`, `score_cofold_poses`, `classify_excited_state`) inspect `std::fs::symlink_metadata` *before* opening paths, rejecting FIFOs/sockets/character-or-block devices (`ft.is_file()` required), enforcing `<= 64 MiB` per file (`MAX_MCP_TOOL_FILE_BYTES`), capping `pose_paths` at `MAX_MCP_COFOLD_POSES = 32`, rejecting `file://` / `http://` / `nmr://` URI schemes in path parameters (`ERR_SECURITY_VIOLATION`), and sanitizing parse/I/O errors so raw input lines or absolute host paths never echo into JSON-RPC responses.
- **HTTP Bridge Per-Request `ReadOnly` Downgrade & Unified Undo**: `nmr-gui::mcp_bridge` honors `X-Nmr-Access-Mode: read-only` per request even when the desktop bridge is started in `ReadWrite` mode (never allowing header escalation from `ReadOnly` to `ReadWrite`), and records all AI curations on the unified monotonic User+MCP `UndoJournal`.
- **Prompt `validate-and-dock-alphafold-model`**: Defaults `restraints_path` to `".nmr_restraints.json"` (`CofoldRestraintBundle` JSON) and guides the 4 implemented MCP tools (`af_concordance` -> `predict_structure_shifts` -> `score_cofold_poses` -> `classify_excited_state`) while directing multi-domain `PAE` docking explicitly to CLI `ferric structure --pae-domains`.

---

## 8. Quantitative Verification Status & Pre-Registered Cold Science Follow-Up (`T21.1`–`T21.6b`)

| Task | Verified Engineering & Statistical Gate (Current Provenance Tag) | Pre-Registered Cold Empirical Follow-Up (Maintainer Science Track) |
| :--- | :--- | :--- |
| **`T21.1`** | Fail-closed `require_predicted_model`, multi-chain AF3 `--chain` guard, ColabFold `plddt` check, Kabsch–Sander `H > E > B > G > I > T > S > C` (`97.37%` [`74/76`] on `1D3Z` vs `1UBQ` `[synthetic self-consistency]`), `17.7` `RCI` constant, gap-aware smoothing, `spearman_rho: Option<f64>`, ordered-core $\rho_s$, 500-rep block-bootstrap CIs, and 6-class gap-free `AfNmrConcordanceClass` (`Indeterminate` on `1UBQ` without `pLDDT`; real AFDB v6 `AF-P0CG48-F1` Ubiquitin whole $\rho_s = 0.507$, ordered-core $\rho_s = 0.442$, `89.5%` `ConcordantOrdered` `[pre-cutoff AFDB]`; `[synthetic self-consistency]` on planted MoRF/hinge fixtures). | Cold post-2021-09-30 proteins with experimental Lipari–Szabo $S^2$ + `AF2`/`AF3` models (`SWARM_FIX_BRIEF` Science #1). |
| **`T21.2`** | Harmonized training/query H-bond geometry (`21.W1:P2-3`), 93-basis Fourier ridge across all 6 nuclei, CHARMM22 EF charges, Golub–Heath–Wahba GCV $\hat{\sigma}$, fail-closed L4 `< 8` CA firewall, and deduplicated **8-protein `[dev-set]`** firewalled RMSEs (`CA 0.986` vs seq-only `2.028`, `CB 1.143` vs `1.671`, `C' 1.012` vs `1.586`, `H^N 0.512` vs `0.656`, `15N 2.756` vs `3.731`, `1HA 0.270` vs `0.470 ppm`) + `21.6%` (`0.655 -> 0.513 ppm`, `n = 232`) aromatic-proximal proton RMSE reduction `[dev-set]`. | Frozen hyper-parameters evaluated once on a disjoint `>= 20`-protein cold BMRB/PDB hold-out (`Science #2`). |
| **`T21.3` / `S9`** | Single consolidated `Structure` mode (`5..12 / 12` controls/mode across `13/13` cases, `66.2%–79.6%` true plot/canvas share at `1440×900`, `0` clipped/overlapping texts at `1280×800`, `>= 11.0 px` fonts, `0` emojis), single `nmr_core::af_concordance` Track 3 (`Indeterminate` without `pLDDT`), cached `PAE` texture (`1` upload), and `169/169` debug & release `nmr-gui` tests passing (`0` `epaint` panics). | Live Dawn `14/14` WGSL + `1440×900` & `1280×800` browser screenshot audit. |
| **`T21.4`** | `0`-byte diff on `03-assignment.md`; `5c4e87d` WebGPU `181/189` & `57/59` restored + 12 golden `U`/`P` snapshots; convex per-nucleus replacement (`0` double-counting), `AF_SIGMA_OOS_PPM` (`0.52, 2.80, 1.05, 1.20, 1.05 ppm`), `uncalibrated_af_prior = true`, `--noe-weight` + `4.0 nats` symmetric site cap, and 4-way `[synthetic self-consistency]` `[pre-cutoff AFDB]` 3-seed ablation (`AF-P0CG48-F1` v6 + `1D3Z` NOESY: `evo-graph` `9.39%` plain -> `7.18%` seq+NOESY -> `19.89%` AF shift -> `25.41%` AF+NOESY; `standard` `6.63% -> 7.18% -> 16.57% -> 16.02%`; sparse `120°` loop `40..52` drop `7/60 -> 1/60` candidly disclosed). | `>= 3` real post-cutoff proteins with experimental sparse `HNCA` + `15N-NOESY` peak lists and `AF3`/`Boltz` models (`Science #3`). |
| **`T21.5`** | `0.000 Å` discontinuous master-frame domain drift, strict linker/tail/`pLDDT < 70` exclusion, genuine `20%` `pcs_q_free`, pre-dock intra-domain `Q_free <= 0.50` gate, `(chain_id, seq_id)` keys, exact SI Solomon `K = 1.2339e16` + mandatory `tau_c`, `7.0 Å` `# metal_anchor` + VarPro, Nilges ambiguous NOEs, `O(L)` voxel-grid GPU `vdW`, and `[synthetic self-consistency]` noisy-domain (`0.257 Å` intra-domain -> `0.494 Å` global RMSD, `Q_free = 0.222`) + experimental `1D3Z` RDC on `1UBQ` (`Q_free = 0.219` `[in-sample]`). | Real experimental 2-domain benchmarks (`CaM` 2-medium RDC + `MPD-ubq` lanthanide `PCS`; `Science #4`). |
| **`T21.6a`** | SI Solomon `K = 1.2339e16` + mandatory `tau_c`, weak `PRE` drop, fail-closed no-apo guard + `10 Å` `DistalSurfaceOutlier` filter, NW FASTA/PDB re-indexing, SMILES contact refusal, `BurialAndContactOnly` default, and non-circular `[synthetic self-consistency]` ablations (`Top-1 = 100%` across `CSP-only`, `PRE-only`, `CSP+NOE`; `6/9` CSP-only vs `0/9` CSP+NOE decoy-only admissibility disclosed). | Real `AF3`/`Boltz-2` pose pools on `>= 5` experimental complexes with published `CSP`s/`PRE`s (`Science #5`). |
| **`T21.6b`** | Duplicate `(chain, seq_id, atom)` rejection, `>= 4` distinct residues gate, discrete-scan marginalized `BIC` ($k_H = 0$, $n = n_{\text{distinct\_residues}}$), experimental-noise-only `H0` + `UnmodeledSignificantExchange`, `His` `CG +8.3 ppm` + `AmbiguousDirectionOnUnsignedCpmg`, endpoint-only scope disclosure, `4/4` `[synthetic self-consistency]` recovery, and hardened `ReadOnly` `MCP` tools. | Published experimental `CEST`/`CPMG` datasets (`T4L L99A`, `SH3` unfolding, `CypA`/`BPTI` *cis/trans*, `His` tautomer; `Science #6`). |
