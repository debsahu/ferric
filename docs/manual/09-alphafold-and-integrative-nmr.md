# Chapter 9: AlphaFold, Secondary Structure & Integrative Structural NMR

FERRIC integrates computational 3D structural models (`AlphaFold 2`, `AlphaFold 3`, `ColabFold`, `Boltz-2`, and experimental `PDB`/`mmCIF` ensembles) with solution NMR observables across the macromolecular workflow: fail-closed coordinate and Predicted Aligned Error (`PAE`) ingestion (`nmr-core::af_concordance`), Kabsch–Sander electrostatic hydrogen-bond secondary-structure classification (`nmr-core::kabsch_sander_ss`), homolog-firewalled 93-basis 3D-to-chemical-shift forward physics (`nmr-core::struct_shifts`), opt-in convex-replacement 3D-prior backbone and `PAE`-gated `15N-NOESY` assignment (`nmr-core::af_assign_priors`), `PAE` spectral rigid-domain multi-medium `RDC`/`PCS`/`PRE` docking (`nmr-core::domain_dock`), `CSP`/`PRE`-guided co-folding restraint export and pose re-scoring (`nmr-core::cofold_bridge`), and Bayesian excited-state (`CEST`/`CPMG`/$R_{1\rho}$) structural hypothesis testing (`nmr-core::excited_state`).

> [!IMPORTANT] Scientific Governance, Provenance Tags & Scope Boundaries
> 1. **Sequence-Only Default Assignment Preserved (`0`-Byte Diff on `specs/03`)**: Default `ferric assign` (`standard`, `evo-graph`, `bayes-sa`) remains strictly sequence-only (`af_priors: None`) with bit-exact CPU/WebGPU scoring (`5c4e87d` golden `U`/`P` snapshots). AlphaFold 3D shift and `PAE`-gated `15N-NOESY` priors are **strictly opt-in** via `--af-structure <model.cif>` and emit `confidence_calibrated = false` (`uncalibrated_af_prior = true`) because FERRIC's 10-bin isotonic calibrator was trained exclusively on sequence-only scores.
> 2. **Mandatory Fail-Closed 4-Layer `HomologFirewall` (`<= 25%` Seq-ID, `TM-score < 0.50`)**: All empirical $(\phi_i, \psi_i, \phi_{i-1}, \psi_{i-1}, \phi_{i+1}, \psi_{i+1}, \chi_1, \omega)$ trigonometric Fourier shift surfaces (`93` basis functions per nucleus) in `nmr-core::struct_shifts` are fitted on the fly from `data/torsion_db/torsion_corpus.tsv` (`129` chains, `13,647` residues; shifts/coords from `CC0 1.0` BMRB/wwPDB, UniProt & CATH annotations under `CC-BY-4.0`) after excluding homologous templates across `L1` (PDB/BMRB/UniProt IDs), `L2` (CATH/ECOD fold topology), `L3` (Needleman–Wunsch, Smith–Waterman, and 7-mer `<= 25%` sequence identity), and `L4` (Zhang–Skolnick `TM-score < 0.50`; query chains with `< 8` $\text{C}^\alpha$ atoms fail closed with `Err`), recording `exclusion_list_sha256` and `max_retained_seq_identity` in every output report.
> 3. **Single- vs Multi-Medium Dipolar Candor (`F8` Carried Forward)**: A single alignment medium's `RDC` Saupe tensor carries intrinsic 4-fold principal-axis $\pi$-rotation cone degeneracy and zero translational sensitivity. Unambiguous 3D multi-domain rigid-body placement requires either $\ge 2$ non-coaxial `RDC` alignment media paired with covalent linker and steric constraints, or translational-plus-orientational paramagnetic `PCS` ($\Delta\boldsymbol{\chi}$, $r_{\text{M}}^{-3}$), exact SI Solomon `PRE` ($\Gamma_2$, $r_{\text{M}}^{-6}$ with mandatory `--pre-tau-c-ns` and `--pre-proton-mhz`), or inter-domain `NOE` restraints, cross-validated under `ByPeptidePlane` `20%` `RdcFree` and domain-stratified `20%` `PCS` $Q_{\text{free}}$ holdouts.
> 4. **Validation Provenance & Scope Honesty**: Current Phase 21 automated benchmarks measure **`[dev-set]`** 8-protein folded BMRB/PDB shift accuracy, **`[pre-cutoff AFDB]`** `AF-P0CG48-F1` v6 Ubiquitin ablations, and **`[synthetic self-consistency]`** forward-model recovery; cold post-cutoff empirical benchmarks are tracked openly in `TODO.md`. Public `AlphaFold 3` JSON (`dialect: "alphafold3"`, `version: 2`, Abramson et al. 2024) carries sequence, ligand, and covalent bond inputs only (no pocket conditioning; NMR restraints are written to `<prefix>.nmr_restraints.json` for `ferric cofold score`), whereas declarative `pocket` and `contact` conditioning is native to `Boltz-2` YAML (`version: 1`).

---

## 1. Fail-Closed AlphaFold Ingestion, Kabsch–Sander `8 -> 4`-State Secondary Structure & NMR Concordance

### Guarded `mmCIF`, `PDB` & `PAE` JSON Ingestion (`nmr-core::af_concordance`)

`parse_af_structure_model`, `require_predicted_model`, and `parse_pae_json_str` ingest predicted and experimental 3D models with strict hostile-input and provenance bounds (`<= 64 MiB` coordinate cap, `<= 4,096` residue square `PAE` cap, `.is_finite()` checks, `0` network I/O in `nmr-core`):

1. **Fail-Closed `pLDDT` Provenance Guard (`require_predicted_model`)**:
   - Extracts per-residue `pLDDT` $\in [0, 100]$ from ModelArchive `_ma_qa_metric_local` cross-referenced strictly against `_ma_qa_metric` where `name`/`type` denotes `pLDDT` (rejecting non-`pLDDT` local scores such as `QMEANDisCo`), or from verified `AF-*` / `_ma_protocol_step` headers with non-constant `[0, 100]` (or rescaled `[0, 1]`) B-factors and zero crystallographic markers.
   - Experimental X-ray/NMR structures (`1UBQ`, even if `REMARK` text mentions `ALPHAFOLD`), headerless `ATOM`-only PDBs, and constant-$B$ PDBs classify as `Experimental` or `Unknown` (`plddt_per_residue = None`) and are **rejected by default** in `ferric af-validate`, `ferric assign --af-structure`, and `ferric-gui` unless `--treat-as-model` (`Treat B-factors as pLDDT`) is explicitly supplied.
2. **Multi-Dialect `PAE` JSON Parser & Cross-Checks**:
   - Supports `AFDB` v3/v4/v6 (`predicted_aligned_error`), legacy sparse `AFDB` v1/v2, `ColabFold` (`pae` + `max_pae` + `plddt`, cross-checking `plddt` against coordinate `pLDDT`s to $\le 1.0$ unit), and `AlphaFold 3` `confidences.json` (`pae` + `token_chain_ids` + `token_res_ids`, requiring explicit `--chain <id>` whenever multiple polymer chains are present).
3. **Opt-In Rate-Limited `AFDB` CLI Fetcher**:
   - `ferric af-validate --fetch-uniprot <UNIPROT_ID> --cache-dir <dir>` validates `[A-Z0-9]{6,10}` accessions, enforces `https://alphafold.ebi.ac.uk` HTTPS allowlisting and `<= 1 req/s` rate limiting, blocks filename traversal, and writes `PROVENANCE.md` with `CC-BY-4.0` attribution and `FIPS 180-4` `SHA-256` digests.

### Kabsch–Sander (`1983`) Electrostatic `8 -> 4`-State Classifier (`nmr-core::kabsch_sander_ss`)

Without reading `HELIX` or `SHEET` coordinate headers, FERRIC reconstructs ideal *trans*-peptide amide protons when absent and evaluates the Kabsch–Sander partial-charge electrostatic carbonyl–amide hydrogen-bond energy ($q_1 = 0.42, q_2 = 0.20$) across all $\lvert i - j \rvert \ge 2$:

$$E_{\text{HB}}(i \to j) = 332\,q_1 q_2 \left(\frac{1}{r_{\text{O}_i\text{N}_j}} + \frac{1}{r_{\text{C}'_i\text{H}_j}} - \frac{1}{r_{\text{O}_i\text{H}_j}} - \frac{1}{r_{\text{C}'_i\text{N}_j}}\right)\text{ kcal mol}^{-1}$$

Retaining up to the two strongest H-bonds per donor and acceptor below $-0.5\text{ kcal mol}^{-1}$, FERRIC enforces the canonical Kabsch–Sander precedence **`H > E > B > G > I > T > S > C`** (extended $\beta$-ladders `E` outranking isolated $\beta$-bridges `B`, with orphan `< 3`-residue `G` or `< 5`-residue `I` fragments demoted to `T`) and maps onto `SecondaryStructure4`:
- **`Helix` (`H`)**: $\alpha$-, $3_{10}$-, and $\pi$-helices (`{H, G, I}`),
- **`Sheet` (`E`)**: extended $\beta$-ladders and isolated $\beta$-bridges (`{E, B}`),
- **`Ppii` (`P`)**: non-H-bonded segments in the left-handed polyproline II basin ($\phi \in [-105^\circ, -45^\circ], \psi \in [+115^\circ, +175^\circ]$; Adzhubei et al. 2013),
- **`Coil` (`C`)**: remaining turns, bends, and irregular loops (`97.37%` [`74/76`] agreement between NMR `1D3Z` and X-ray `1UBQ` Ubiquitin `[in-sample]`).

### Original `pLDDT` vs `RCI-S²` / Lipari–Szabo $S^2$ Concordance (`ferric af-validate`)

Inspired by Fowler, Sljoka & Williamson (2020, 2022), FERRIC maps $\text{pLDDT}_i$ via an original heuristic $\hat{S}^2_{\text{AF},i} = \operatorname{clamp}((\text{pLDDT}_i - 35)/55,\, 0,\, 1)$ (rather than `FIRST` constraint-network rigidity) and compares against either experimental Lipari–Szabo $S^2_i$ (`--s2 <modelfree.tsv>`, `RigiditySource::ExperimentalRelaxationS2`) or Berjanskii–Wishart (`2005`) chemical-shift `RCI-S²` (`RigiditySource::ShiftDerivedRci`, canonical `17.7` scaling constant, `RCI <= 0.175`, smoothed strictly across contiguous assigned neighbors without bridging unassigned gaps). Reports include `spearman_rho` (`null` on zero-variance vectors), ordered-core `spearman_rho_ordered` ($\text{pLDDT}_i \ge 70$), 500-replicate moving-block bootstrap 95% CIs (`spearman_ci95`, `rmsd_rig_ci95`), and a gap-free 6-class per-residue taxonomy (`PLDDT_ORDERED_CUTOFF = 70.0`, `S2_ORDERED_CUTOFF = 0.65`):

| Concordance Class | Physical Criteria | Structural Interpretation |
| :--- | :--- | :--- |
| **`ConcordantOrdered`** | $\text{pLDDT}_i \ge 70$, $S^2_i \ge 0.65$, $\Delta\theta_i < 60^\circ$, no `H <-> E` flip | Rigid globular core confirmed in solution |
| **`ConcordantDisordered`** | $\text{pLDDT}_i < 70$ and $S^2_i < 0.65$ | Flexible tail, inter-domain linker, or IDP segment |
| **`ModelOrderedSolutionDisordered`** | $\text{pLDDT}_i \ge 70$ and $S^2_i < 0.65$ *(alias `ConditionalFoldingOrCrystalArtifact`)* | Partner-induced $\alpha$-MoRF (Alderson et al. 2023) or model-ordered segment dynamically disordered in solution |
| **`DynamicHingeOrAltState`** | ($\text{pLDDT}_i < 70$, $S^2_i \ge 0.65$) or ($\text{pLDDT}_i \ge 70$, $S^2_i \ge 0.65$ with $\Delta\theta_i \ge 60^\circ$ or `H <-> E` flip) | Conformational switch, domain hinge, or oligomerization interface |
| **`Indeterminate`** | Assigned NMR shifts present, but `pLDDT` is absent (`None`) | Experimental X-ray/NMR coordinate input without verified `pLDDT` (never circularly defaulted to `85.0`) |
| **`UnassignedNmr`** | Fewer than `2` assigned backbone shifts and no experimental $S^2_i$ | Unassigned or Proline/exchange-broadened NMR residue |

```bash
ferric af-validate \
  --structure AF-P0CG48-F1-model_v6.cif \
  --pae AF-P0CG48-F1-predicted_aligned_error_v6.json \
  --shifts assigned.str \
  [--s2 modelfree.tsv] [--chain A] [--treat-as-model] \
  --exclude-pdb 1UBQ,1D3Z --exclude-bmrb 17769 \
  -o af_concordance.json
```

---

## 2. Clean-Room 93-Basis 3D Structure-to-Chemical-Shift Forward Predictor (`T21.2`)

`nmr-core::struct_shifts` (`ferric predict-shifts`) predicts backbone ($\text{H}^{\text{N}}, {}^{15}\text{N}, {}^{13}\text{C}^\alpha, {}^{13}\text{C}^\beta, {}^{13}\text{C}', {}^{1}\text{H}^\alpha$) and optional `ILVMAT` sidechain methyl ($^1\text{H}^{\text{m}}, {}^{13}\text{C}^{\text{m}}$) chemical shifts via an additive physical decomposition (Shen & Bax 2010):

$$\delta^{\text{struct}}_{i,a}(\mathbf{X}) = \delta^{\text{RC}}_{i,a}(T, \text{pH}) + \Delta\delta^{\text{tor}}_{i,a}(\phi_i, \psi_i, \phi_{i-1}, \psi_{i-1}, \phi_{i+1}, \psi_{i+1}, \chi_{1,i}, \omega_i, \omega_{i-1}) + \Delta\delta^{\text{HB}}_{i,a}(\mathbf{X}) + \Delta\delta^{\text{RC-arom}}_{i,a}(\mathbf{X}) + \Delta\delta^{\text{pep/EF}}_{i,a}(\mathbf{X})$$

1. **Sequence/pH/Temperature Random-Coil Baseline (`ncIDP`)**: Tamiola et al. (2010) neighbor-corrected coil shifts (including pre-Proline `-1.95 ppm` on `CA`/`C'`), Platzer et al. (2014) Henderson–Hasselbalch pH titration, and Kjaergaard et al. (2011) linear temperature coefficients.
2. **Firewalled 93-Basis Trigonometric Fourier Ridge Regression**: Harmonizes canonical $(\text{C}^\alpha, \phi, \psi, \chi_1)$ H-bond geometry between corpus subtraction and query inference (`21.W1:P2-3`), fitting all **`93` basis functions** (`49` `7×7` order-3 $(\phi_i, \psi_i)$ + `10` order-2 $(\phi_{i-1}, \psi_{i-1})$ + `4` order-1 $(\phi_{i+1}, \psi_{i+1})$ + `6` $\chi_{1,i}$ + `4` Kabsch–Sander H-bond & SS indicators + `20` amino-acid indicators; Cholesky $\lambda = 2.5$) across all 6 backbone nuclei strictly on `HomologFirewall`-retained chains (`max_seq_identity <= 0.25`, `TM-score < 0.50`).
3. **Haigh–Mallion (`1972`) / Case (`1995`) Ring Currents, Empirical $r_{\text{HO}}^{-3}\cos^2\theta_{\text{NHO}}$ H-Bonds & CHARMM22 Electric Fields**: Evaluates signed triangular-projection ring currents across `Phe`, `Tyr`, `His`, and `Trp` rings within $12.0\text{ \AA}$ (`21.6%` [`0.655 -> 0.513 ppm`, `n = 232`] proton RMSE reduction on ring-proximal spins $\le 5.5\text{ \AA}$ `[dev-set]`), empirical inverse-cube H-bond shifts (Wagner, Pardi & Bax 1983; Sitkoff & Case 1998), McConnell carbonyl anisotropy, and CHARMM22 partial-charge bond-parallel electric fields (MacKerell et al. 1998).
4. **Golub–Heath–Wahba (`1979`) GCV Predictive Uncertainty & Smooth `pLDDT` Blending**: Computes out-of-sample GCV standard errors $\hat{\sigma}_{\text{struct},a}$ and blends smoothly into `ncIDP` coil priors as `pLDDT` drops, widening $\hat{\sigma}_{i,a}$ monotonically at low `pLDDT`. Across the deduplicated **8-protein `[dev-set]`** non-homologous folded BMRB/PDB benchmark, out-of-sample RMSEs beat the `ncIDP` sequence-only baseline on all 6 nuclei: **`0.986` vs `2.028 ppm` (`CA`)**, **`1.143` vs `1.671 ppm` (`CB`)**, **`1.012` vs `1.586 ppm` (`C'`)**, **`0.512` vs `0.656 ppm` (`H^N`)**, **`2.756` vs `3.731 ppm` (`15N`)**, and **`0.270` vs `0.470 ppm` (`HA`, 5 proteins)** `[dev-set]`.

```bash
ferric predict-shifts \
  --structure model.cif \
  --ph 6.5 --temp-k 298.15 --methyl-ilvmat \
  --exclude-pdb 1UBQ,1D3Z --exclude-bmrb 17769 \
  -o predicted_shifts.str
```

---

## 3. Consolidated `Structure` Mode, Vector Secondary-Structure Ribbon & 3D Putty (`T21.3` & `S9`)

Inside `ferric-gui`, switching to the top-level **`Structure`** mode (`Process | Assign | Structure | Dynamics | Small molecule | Metabolomics`) unifies the 3D `wgpu` cartoon viewer, the 3-track vector secondary-structure ribbon (`nmr-gui::ss_ribbon_widget`), and the `PAE & Co-Fold` inspector (`nmr-gui::integrative_af_panel`) with zero duplicate toolbar clutter (`0` emojis, `<= 12` primary controls/mode, `>= 11.0 px` typography; see the canonical unified mouse/keyboard interaction table in [Chapter 6](06-mcp-and-cli-reference.md#interactive-workbench-keyboard-shortcuts)):

- **Track 1 (`3D SS`)**: Kabsch–Sander sinusoidal coral helices (`#E05A47`), directional teal $\beta$-strand arrows (`#2A9D8F`), `PPII` diamond ticks (`#7B61FF`), and coil baseline.
- **Track 2 (`NMR SS`)**: Live `< 1.0 ms` solution NMR `SSP` propensity bars (`SSP > 0` $\alpha$-helix, `SSP < 0` $\beta$-strand).
- **Track 3 (`AF vs NMR`)**: Color-vision-safe 6-state `AfNmrConcordanceClass` blocks (emerald `ConcordantOrdered`, slate `ConcordantDisordered`, hatched amber `ModelOrderedSolutionDisordered`, double-bordered crimson `DynamicHingeOrAltState`, neutral slate-outline `Indeterminate` when `pLDDT` is absent, and hollow `UnassignedNmr`). Clicking any residue column synchronizes 2D `HSQC`, 3D strips, and the 3D camera simultaneously.

---

## 4. Opt-In Convex-Replacement 3D Priors & `PAE`-Gated `15N-NOESY` Assignment (`T21.4`)

For sparse `HNCA`-only or perdeuterated proteins where `13CB` transfer fails, `ferric assign --af-structure <model.cif>` (`nmr-core::af_assign_priors`) injects tempered 3D priors in a dedicated `@group(1)` GPU kernel (`compute_unary_af`) without double-counting amino-acid identity:

1. **Unified Offset Frame & Out-of-Sample Error Floor**: Uses the single unlabelled `estimate_reference_offsets` shift vector (`0` dual-offset gauge drift) and floors `T21.2` $\hat{\sigma}_{k,a}$ with `AF_SIGMA_OOS_PPM` (`H 0.52, N 2.80, CA 1.05, CB 1.20, C' 1.05 ppm`).
2. **Per-Nucleus Convex Log-Linear Replacement (`0` Double-Counting)**: For each observed nucleus $a$, evaluates sequence log-odds $\ell_{\text{seq}}(s, k, a)$ and clamped Student-$t(\nu = 4)$ AF log-odds $\ell_{\text{AF}}(s, k, a)$ against the **same** `Priors::background` density and forms $(1 - \omega_r)\ell_{\text{seq}} + \omega_r\ell_{\text{AF}}$ with $\omega_r = \operatorname{clamp}(\lambda_{\text{AF}}\,w_{\text{AF}}(\text{pLDDT}_k), 0, 1)$ and a single $-\ln(n_{\text{pos}})$ term (`max |U_cpu - U_gpu| < 1.0e-5`).
3. **`PAE <= 6.0 Å` Gated `3D 15N-NOESY` Contacts & Symmetric Site Cap**: Admits predicted $d(\text{H}^{\text{N}}_j, \text{H}^{\text{N}}_k) \le 5.5\text{ \AA}$ contacts only when $\text{pLDDT}_j, \text{pLDDT}_k \ge 70$ and symmetrized $\overline{\text{PAE}}(j, k) \le 6.0\text{ \AA}$, scales by `--noe-weight`, and bounds per-site pairwise NOE bonuses symmetrically to `[-4.0, +4.0] nats`.
4. **Candid Scope & Pre-Cutoff Caveats**:
   - Output reports set `confidence_calibrated = false` and `uncalibrated_af_prior = true` (`PAV` isotonic table bypassed).
   - On 3-seed `CB`-stripped `HSQC + HNCA` (`0` `CB`, `0` `C'`) Ubiquitin (`AF-P0CG48-F1` v6 `[pre-cutoff AFDB]` + `1D3Z` NOESY `[synthetic self-consistency]`), `evo-graph` progresses from `9.39%` (seq-only) and `7.18%` (seq + `15N-NOESY`) to `19.89%` (AF shift alone) and **`25.41%`** (AF shift + `PAE`-gated `15N-NOESY`; `standard`: `6.63% -> 7.18% -> 16.57% -> 16.02%`).
   - On full 7-spectrum triple-resonance data, corrupting a 12-residue loop (`res 40..52`) by $120^\circ$ at `pLDDT = 95` causes `0` accuracy loss (`349/360 -> 352/360` `standard`, `349/360 -> 350/360` `evo-graph`), whereas on sparse `HNCA`-only data overall accuracy rises (`6.82% -> 11.93%`) while local inside-loop (`40..52`) accuracy drops (`7/60 -> 1/60` `[synthetic self-consistency]`); inspect suspect loops with `ferric af-validate`.

```bash
ferric assign \
  --peaks sparse_hnca.tsv \
  --seq protein.fasta \
  --af-structure model.cif \
  --af-pae pae.json \
  --noesy-15n noesy15n.tsv \
  --af-weight 0.65 --noe-weight 1.0 --af-clamp-nats 4.0 \
  --solver evo-graph --engine auto \
  --out assigned_af
```

---

## 5. `PAE` Spectral Domain Clustering & Multi-Domain `RDC`/`PCS`/`PRE` Docking (`T21.5`)

`ferric structure --pae-domains` (`nmr-core::domain_dock`, `nmr-gpu::domain_dock_gpu`) resolves multi-domain solution architectures while freezing intra-domain geometry to `0.000 Å` drift (including discontinuous multi-segment domains):

1. **Normalized Laplacian Eigengap Domain Partitioning & Linker Exclusion**: Partitions ordered residues ($\text{pLDDT}_i \ge 70$) via the symmetric Laplacian eigengap ($\overline{\text{PAE}}_{\text{intra}} \le 5\text{ \AA}$, $\lvert\mathcal{D}_d\rvert \ge 20$), keys all restraints by `(chain_id, seq_id)`, and strictly excludes flexible linkers, disordered tails, and $\text{pLDDT} < 70$ sites from Saupe SVD and `ByPeptidePlane` `RdcFree` holdouts.
2. **Pre-Dock Intra-Domain `Q_free <= 0.50` Gate & Tolman (`2001`) GDO Motion Test**: Verifies intra-domain `Q_free <= 0.50` before docking and computes per-domain Generalized Degree of Order $\vartheta_{d,m}$; ratios $\rho_{\text{GDO}} < 0.80$ at $> 2\sigma_{\text{MC}}$ flag **`DecoupledInterDomainMotion`**.
3. **Exact SI Solomon `PRE`, `7.0 Å` `PCS` Metal VarPro, Nilges `NOE` & `O(L)` WebGPU `vdW`**: Evaluates exact CODATA SI Solomon $\Gamma_2$ (`K_Solomon = 1.2339e16 Å^6 s^-2`, unsoftened $r^{-6}$ for $r \ge 3.5\text{ \AA}$, requiring explicit `--pre-tau-c-ns` and `--pre-proton-mhz`, plus `--mtsl-tether-allowance-a 4.0`), requires an explicit `# metal_anchor X Y Z` / `--pcs-metal-xyz` within `7.0 Å` of coordinating atoms with 8-parameter `VarPro` refinement and genuine `20%` domain-stratified `pcs_q_free`, aggregates ambiguous inter-domain NOEs via $r_{\text{eff}} = (\sum r_{ab}^{-6})^{-1/6}$, and accelerates steric clash evaluation on WebGPU with an `O(L)` `32×32×32` spatial hash grid.

```bash
ferric structure --pae-domains \
  --af-structure multidomain.cif \
  --af-pae multidomain_pae.json \
  --rdc medium1_pf1.tsv --rdc medium2_peg.tsv \
  --pcs lanthanide_pcs.tsv --pcs-metal-xyz 12.4,-4.2,8.1 \
  --pre mtsl_pre.tsv --pre-tau-c-ns 9.5 --pre-proton-mhz 600.13 \
  --rdc-free-frac 0.20 --ensemble-size 10 --engine auto \
  -o docked_ensemble.pdb
```

---

## 6. `CSP`/`PRE`-Guided Co-Folding Bridge (`AlphaFold 3` / `Boltz-2`) & Pose Re-Scoring (`T21.6a`)

`nmr-core::cofold_bridge` (`ferric cofold export` and `ferric cofold score`) connects experimental titration `CSP`, `PRE`, `NOE`, `RDC`, and `PCS` restraints to biomolecular complex co-folding:

- **Fail-Closed Shrake–Rupley (`rSASA >= 0.20`) + `10.0 Å` Cluster Pocket Filter**: Requires an apo receptor 3D model (`--apo-structure`; refusing unfiltered pocket export unless `--allow-unfiltered-no-apo` is passed), separates `2\sigma` surface `CSP` hits ($\text{rSASA}_i \ge 0.20$) from buried **`AllostericCorePerturbation`** relays, and retains the dominant `10.0 Å` spatial $\text{C}^\alpha$ patch while reclassifying isolated distal hits as **`DistalSurfaceOutlier`**.
- **Exact Solomon `PRE` Conversion & NW FASTA/PDB Re-Indexing**: Converts `PRE` $\Gamma_2$ rows using `K_Solomon = 1.2339e16 Å^6 s^-2` (requiring `--tau-c-ns`) and drops weak bounds $> 20.0\text{ \AA}$ rather than clamping them; aligns PDB residue numbers to full-length FASTA indices via gap-preserving Needleman–Wunsch alignment (`>= 80%` identity); refuses atom-named `contact` constraints on `SMILES` ligands (`Err`); and exports schema-valid `Boltz-2` YAML (`version: 1`) and `AlphaFold 3` JSON (`version: 2`, paired with `<prefix>.nmr_restraints.json`).
- **Multi-Pose Re-Scorer (`BurialAndContactOnly` Default & Decoy Candor)**: Defaults to `CspScoringMode::BurialAndContactOnly` (`5.0 Å` contact ROC-AUC + $\Delta\text{rSASA}$ Spearman $\rho_s$; uncalibrated ring-current/H-bond `CSP` summation requires `--experimental-csp-ring-hbond`). While `CSP`-only ranks near-native poses `#1` when present, rejecting **decoy-only** pools (`PoseVerdict::InconclusiveOrAllViolated`) requires distance or orientational restraints (`NOE`/`PRE`/`RDC`/`PCS`: `6/9` decoy-only runs remain admissible under `CSP`-only vs `0/9` under `CSP + intermolecular NOE` `[synthetic self-consistency]`).

```bash
# Export Boltz-2 YAML (native pocket & contact conditioning) or AlphaFold 3 JSON + .nmr_restraints.json
ferric cofold export \
  --format boltz2 \
  --fasta complex.fasta --ligand-smiles "c1ccc(cc1)C(=O)N" \
  --apo-structure apo_receptor.cif \
  --csp csp_fit.tsv --pre intermolecular_pre.tsv --tau-c-ns 8.5 \
  -o complex_boltz2.yaml

# Re-score co-folded candidate complex poses against solution NMR restraints
ferric cofold score \
  --poses pose_0.cif,pose_1.cif,pose_2.cif \
  --apo-structure apo_receptor.cif \
  --csp csp_fit.tsv --pre intermolecular_pre.tsv --tau-c-ns 8.5 --rdc complex_rdc.tsv \
  -o pose_ranking.json
```

---

## 7. Bayesian Excited-State (`CEST` / `CPMG` / $R_{1\rho}$) Hypothesis Engine (`T21.6b`)

Relaxation dispersion (`CPMG`, $R_{1\rho}$) and saturation transfer (`CEST`, `DEST`, [Chapter 4](04-dynamics-and-exchange.md)) yield chemical-shift differences $\Delta\varpi_{AB,i,a} = \varpi_{B,i,a} - \varpi_{A,i,a}$ for sparsely populated (`0.5%–10%`) invisible protein conformers. `nmr-core::excited_state` (`ferric exchange-state`) evaluates candidate endpoint mechanisms against ground-state structure $\mathbf{X}_A$ and shifts $\varpi_A$:

### Signed & Folded-Unsigned Student-$t_5$ Likelihoods + Discrete-Scan Marginalized `BIC`

Evaluating differential shifts $\mu_{\Delta,i,a}^{(H,m)} = \hat{\delta}_{B,i,a}^{(H,m)} - \hat{\delta}_{A,i,a}(\mathbf{X}_A)$ cancels static per-residue offsets on ground state $A$. With $\sigma_{\text{tot},i,a} = \sqrt{\sigma_{\text{exp},i,a}^2 + \sigma_{\text{fwd},a}^2}$ and normalized Student-$t_5$ density $f_{t_5}(z) = \frac{8}{3\pi\sqrt{5}}(1 + z^2/5)^{-3}$, signed `CEST` rows use $\ln f_{t_5}((\Delta\varpi_{AB} - \mu_\Delta)/\sigma_{\text{tot}}) - \ln\sigma_{\text{tot}}$ while unsigned `CPMG` rows ($y = \lvert\Delta\varpi_{AB}\rvert \ge 0$, Skrynnikov et al. 2002) marginalize over $\pm\mu_\Delta$. Because $H_1$–$H_4$ scan discrete candidate sets of size $M_H$ with $k_H = 0$ continuous fitted parameters, FERRIC computes the uniform-prior **marginalized log-likelihood** $\ln\bar{L}_H = \ln\bigl(\frac{1}{M_H}\sum_{m=1}^{M_H} e^{\ell(H,m)}\bigr)$ and $\mathrm{BIC}(H) = -2\ln\bar{L}_H$ ($n = n_{\text{distinct\_residues}}$):

| Code | Hypothesis Family | Discrete Scan $M_H$ ($k_H = 0$) | Physical Forward Endpoint Model ($\mu_\Delta$) |
| :---: | :--- | :---: | :--- |
| **`H0`** | `NullOrUnstructuredPerturbation` | `1` | $\mu_{\Delta,i,a} = 0$ evaluated strictly against experimental noise $\sigma_{\text{exp},i,a}$ (`0` $\sigma_{\text{fwd}}$ inflation) |
| **`H1`** | `LocalUnfolding` | Windows $[u, v]$ | Full cooperative unfolding to pH/T-corrected `ncIDP` coil ($\mu_\Delta = \delta^{\text{RC}} - \varpi_A$; Vallurupalli, Hansen & Kay 2008) |
| **`H2`** | `XxxProCisTrans` | Proline sites $p$ | Spatial ($\le 8.0\text{ \AA}$) or local ($\lvert i - p \rvert \le 3$) `Xxx–Pro` $\omega_{p-1}$ *trans* $\leftrightarrow$ *cis* switch ($\Delta\delta_{\text{CB}} = \pm 2.40$, $\Delta\delta_{\text{CG}} = \mp 2.60\text{ ppm}$; Schubert 2002) |
| **`H3`** | `RotamerOrHisTautomerFlip` | Ring rotamers & `His` states | Clash-checked `Phe`/`Tyr`/`Trp`/`His` $(\chi_1, \chi_2)$ ring-current flips ($\le 10.0\text{ \AA}$) + Pelton (1993) `His` $\mathrm{N}^{\delta 1}\text{–H} \to \mathrm{N}^{\varepsilon 2}\text{–H}$ ($\Delta\delta_{\text{ND1}} \approx -82$, $\Delta\delta_{\text{NE2}} \approx +82$, $\Delta\delta_{\text{CD2}} = -7.2$, $\Delta\delta_{\text{CG}} = +8.3\text{ ppm}$; unsigned `CPMG` flags `AmbiguousDirectionOnUnsignedCpmg`) |
| **`H4`** | `Alternate3DConformer` | `--alt-structures` ($M_{H_4}$) | Candidate 3D conformers scored via firewalled $\delta^{\text{struct}}(\mathbf{X}_{B,m}) - \delta^{\text{struct}}(\mathbf{X}_A)$ (duplicate conformers leave $\ln\bar{L}_{H_4}$ invariant) |

> [!NOTE] Input Gates, `UnmodeledSignificantExchange` & Endpoint-Only Scope Limit
> - **Input integrity**: Duplicate `(chain_id, seq_id, atom_name)` rows across `--exchange` files are rejected (`Err`), `CPMG` inputs default to pre-filtered significant dispersions (`--cpmg-unfiltered` gates raw rows at $\lvert\Delta\varpi\rvert/\sigma_{\text{exp}} \ge 3$), and `Committed` verdicts require **`n_distinct_residues >= 4`**, $2\ln\mathrm{BF}_{1,2} \ge 6.0$ against the runner-up non-null family, $2\ln\mathrm{BF}_{1,H_0} \ge 10.0$, and $\chi^2_\nu \le 2.5$.
> - **Unmodeled exchange**: When `H0` is strongly rejected yet all $H_1$–$H_4$ yield $\chi^2_\nu > 2.5$, `classify_excited_state` emits **`ExcitedStateVerdict::UnmodeledSignificantExchange`**.
> - **Endpoint-only scope**: `excited_state` models pure 2-state shift endpoints $\Delta\varpi_{AB}$ with **no** fractional partial-unfolding parameter $\alpha \in (0, 1)$ and **no** population $p_B$ term (fit kinetics first via `ferric cest` / `ferric cpmg`).

```bash
ferric exchange-state \
  --shifts ground_assigned.str \
  --structure ground_model.cif \
  --exchange cest_15n_fit.tsv \
  --exchange cpmg_13c_fit.tsv \
  [--cpmg-unfiltered] [--alt-structures alt_closed.cif,alt_hinge.pdb] \
  --ph 6.5 --temp-k 298.15 \
  -o excited_state_report.json
```

---

## 8. Consolidated `Structure` Studio (`nmr-gui`) & Hardened `ReadOnly` `MCP` Tools

- **Consolidated `Structure` Mode (`nmr-gui`)**: Selecting **`Structure`** in the single top-level workbench switcher (`Process | Assign | Structure | Dynamics | Small molecule | Metabolomics`) provides `3D Model` and `PAE & Co-Fold` views with a GPU-cached `egui::TextureHandle` `PAE` domain heatmap, neutral `[synthetic demo — no experimental restraints]` provenance badges on built-in demo states, and shared monotonic User+MCP Undo/Redo (`Cmd/Ctrl+Z`).
- **Hardened Workspace-Sandboxed `ReadOnly` `MCP` Tools ([Chapter 6](06-mcp-and-cli-reference.md))**: `af_concordance`, `predict_structure_shifts` (`<= 25%` seq-id + `TM < 0.50` `HomologFirewall`), `score_cofold_poses` (`<= 32` poses), `classify_excited_state`, and prompt `validate-and-dock-alphafold-model` (`restraints_path = ".nmr_restraints.json"`) enforce pre-open `symlink_metadata` regular-file & `64 MiB` size checks, URI scheme rejection, sanitized error messages, and per-request `X-Nmr-Access-Mode: read-only` downgrade on the desktop HTTP bridge.
