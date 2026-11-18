# Chapter 9: AlphaFold, Secondary Structure & Integrative Structural NMR

FERRIC integrates computational 3D structural models (`AlphaFold 2`, `AlphaFold 3`, `ColabFold`, `Boltz-2`, and experimental `PDB`/`mmCIF` ensembles) with solution NMR observables across the complete macromolecular workflow: hostile-input-safe coordinate and Predicted Aligned Error (`PAE`) ingestion (`nmr-core::af_concordance`), Kabsch–Sander electrostatic hydrogen-bond secondary-structure classification (`nmr-core::kabsch_sander_ss`), homolog-firewalled 3D-to-chemical-shift forward physics (`nmr-core::struct_shifts`), opt-in 3D-prior-assisted backbone and `PAE`-gated `15N-NOESY` assignment (`nmr-core::af_assign_priors`), `PAE` spectral rigid-domain multi-medium `RDC`/`PCS`/`PRE` docking (`nmr-core::domain_dock`), `CSP`/`PRE`-guided co-folding restraint export and pose re-scoring (`nmr-core::cofold_bridge`), and Bayesian excited-state (`CEST`/`CPMG`/$R_{1\rho}$) structural hypothesis testing (`nmr-core::excited_state`).

> [!IMPORTANT] Scientific Governance, Homolog Firewall & Dipolar Candor
> 1. **Sequence-Only Default Assignment Preserved**: Default `ferric assign` (`standard`, `evo-graph`, `bayes-sa`) remains strictly sequence-only with zero 3D coordinate inputs. AlphaFold 3D shift and `PAE`-gated `15N-NOESY` priors are **strictly opt-in** via `--af-structure <model.cif>` and leave sequence-only assignment scoring completely unmodified when omitted.
> 2. **Mandatory 4-Layer `HomologFirewall` (`<= 25%` Sequence Identity, `TM-score < 0.50`)**: All empirical $(\phi_i, \psi_i, \phi_{i-1}, \psi_{i-1}, \chi_1)$ trigonometric Fourier shift surfaces in `nmr-core::struct_shifts` are fitted on the fly (`< 15 ms`) exclusively from FERRIC's public-domain (`CC0 1.0`) `data/torsion_db/torsion_corpus.tsv` (`129` chains, `13,647` residues) after excluding homologous templates across `L1` (PDB/BMRB/UniProt/family IDs), `L2` (CATH/ECOD fold topology), `L3` (Needleman–Wunsch, Smith–Waterman, and 7-mer `<= 25%` sequence identity), and `L4` (Zhang–Skolnick `TM-score < 0.50` computed directly against the query $\text{C}^\alpha$ trace), recording `exclusion_list_sha256` and `max_retained_seq_identity` in every output report.
> 3. **Single- vs Multi-Medium Dipolar Candor (`F8` Carried Forward)**: A single alignment medium's `RDC` Saupe tensor carries intrinsic 4-fold principal-axis $\pi$-rotation cone degeneracy and zero translational sensitivity. Unambiguous 3D multi-domain rigid-body placement requires either $\ge 2$ non-coaxial `RDC` alignment media paired with covalent linker and steric constraints, or translational-plus-orientational paramagnetic `PCS` ($\Delta\boldsymbol{\chi}$, $r_{\text{M}}^{-3}$), Solomon `PRE` ($\Gamma_2$, $r_{\text{M}}^{-6}$), or inter-domain `NOE` restraints, always cross-validated under `ByPeptidePlane` `20%` `RdcFree` holdout. (On unseen *de novo* folding without domain templates, cold holdout `1KHM -> 1ZZK` spans `1.86–4.14 Å`, tracked under `F8`.)
> 4. **`AlphaFold 3` JSON vs `Boltz-2` YAML Restraint Schema Honesty**: Public `AlphaFold 3` input JSON (`dialect: "alphafold3"`, `version: 2`, Abramson et al. 2024) encodes polymer sequences, `SMILES`/`CCD` ligands, and `bondedAtomPairs` covalent topology, paired by FERRIC with `<prefix>.nmr_restraints.json` for post-prediction `ferric cofold score` pose selection; declarative surface `pocket` and pairwise distance `contact` conditioning during neural inference is natively supported in `Boltz-2` YAML (`version: 1`, Wohlwend et al. 2024; Passaro et al. 2025) and re-scored quantitatively across both tools by `ferric cofold score`.

---

## 1. AlphaFold Ingestion, Kabsch–Sander `8 -> 4`-State Secondary Structure & NMR Concordance

### Guarded `mmCIF`, `PDB` & `PAE` JSON Ingestion (`nmr-core::af_concordance`)

`parse_af_structure_model` and `parse_pae_json_str` ingest predicted and experimental 3D models with strict hostile-input bounds (`<= 64 MiB` coordinate cap, `<= 4,096` residue square `PAE` cap, `.is_finite()` checks, `0` network I/O in `nmr-core`):

1. **Per-Residue `pLDDT` Provenance Guard**:
   - Reads exact per-residue `pLDDT` $\in [0, 100]$ from ModelArchive `_ma_qa_metric_local` cross-referenced against `_ma_qa_metric`.
   - When `_ma_qa_metric_local` is absent, inspects `_exptl.method` / `PDB` `EXPDTA` / crystallographic `CRYST1` unit cells: experimental `X-RAY DIFFRACTION`, `SOLUTION NMR`, and `ELECTRON MICROSCOPY` structures set `plddt_per_residue = None` (`Experimental` provenance) unless `--force-bfactor-plddt` is passed, preventing thermal crystallographic $B$-factors ($\text{\AA}^2$) from ever being misinterpreted as `pLDDT`. Computational `[0, 1]` confidence columns are automatically rescaled to `[0, 100]` with an explicit diagnostic note.
2. **Multi-Dialect `PAE` JSON Parser**:
   - Supports `AFDB` v3/v4 (`predicted_aligned_error`), legacy sparse `AFDB` v1/v2 (`residue1`/`residue2`/`distance`), `ColabFold` (`pae` + `max_pae`), and `AlphaFold 3` `confidences.json` (`pae` + `token_chain_ids` + `token_res_ids`).
3. **Opt-In Rate-Limited `AFDB` CLI Fetcher**:
   - `ferric af-validate --fetch-uniprot <UNIPROT_ID> --cache-dir <dir>` validates `[A-Z0-9]{6,10}` accessions, enforces a strict `https://alphafold.ebi.ac.uk` HTTPS allowlist and `<= 1 req/s` token-bucket rate limit (`User-Agent: ferric-af-fetch`), blocks filename traversal, and writes `PROVENANCE.md` with `CC-BY-4.0` attribution and `FIPS 180-4` `SHA-256` digests.

### Kabsch–Sander (`1983`) Electrostatic `8 -> 4`-State Classifier (`nmr-core::kabsch_sander_ss`)

Without reading `HELIX` or `SHEET` coordinate headers, FERRIC reconstructs ideal *trans*-peptide amide protons when absent and evaluates the Kabsch–Sander partial-charge electrostatic carbonyl–amide hydrogen-bond energy ($q_1 = 0.42, q_2 = 0.20$) across all $|i - j| \ge 2$:

$$E_{\text{HB}}(i \to j) = 332\,q_1 q_2 \left(\frac{1}{r_{\text{O}_i\text{N}_j}} + \frac{1}{r_{\text{C}'_i\text{H}_j}} - \frac{1}{r_{\text{O}_i\text{H}_j}} - \frac{1}{r_{\text{C}'_i\text{N}_j}}\right)\text{ kcal mol}^{-1}$$

Retaining up to the two strongest H-bonds per donor and acceptor below $-0.5\text{ kcal mol}^{-1}$, FERRIC assigns the 8-state taxonomy (`H`, `G`, `I`, `E`, `B`, `T`, `S`, `C`) and maps it onto FERRIC's 4-state NMR secondary-structure alphabet (`SecondaryStructure4`):
- **`Helix` (`H`)**: $\alpha$-, $3_{10}$-, and $\pi$-helices (`{H, G, I}`),
- **`Sheet` (`E`)**: extended $\beta$-ladders and isolated $\beta$-bridges (`{E, B}`),
- **`Ppii` (`P`)**: non-H-bonded segments in the left-handed polyproline II basin ($\phi \in [-105^\circ, -45^\circ], \psi \in [+115^\circ, +175^\circ]$; Adzhubei et al. 2013),
- **`Coil` (`C`)**: remaining turns, bends, and irregular loops (`97.37%` [`74/76`] agreement between NMR `1D3Z` and X-ray `1UBQ` Ubiquitin).

### `pLDDT` vs `RCI-S²` / Lipari–Szabo $S^2$ Concordance (`ferric af-validate`)

Comparing effective AlphaFold rigidity $\hat{S}^2_{\text{AF},i} = \operatorname{clamp}((\text{pLDDT}_i - 35)/55,\, 0,\, 1)$ against experimental Lipari–Szabo $S^2_i$ (`--s2 <modelfree.tsv>`) or 6-nucleus Berjanskii–Wishart `RCI` order parameters $S^2_{\text{RCI},i}$ (Fowler, Sljoka & Williamson 2020, 2022), FERRIC reports tie-adjusted Spearman $\rho_s$, `RMSD_rig`, and classifies every assigned residue into four regimes:

| Concordance Class | Physical Criteria | Structural Interpretation |
| :--- | :--- | :--- |
| **`ConcordantOrdered`** | $\text{pLDDT}_i \ge 70$, $S^2_{\text{exp},i} \ge 0.65$, $\Delta\theta_i < 60^\circ$, no `H <-> E` flip | Rigid globular core confirmed in solution |
| **`ConcordantDisordered`** | $\text{pLDDT}_i < 70$ and $S^2_{\text{exp},i} < 0.65$ | Flexible tail, inter-domain linker, or IDP segment |
| **`ConditionalFoldingOrCrystalArtifact`** | $\text{pLDDT}_i \ge 70$ (`Helix`/`Sheet` in 3D) but $S^2_{\text{exp},i} < 0.60$ & coil/PPII $\ge 0.65$ | Partner-induced $\alpha$-MoRF (Alderson et al. 2023) or crystal-packing helix disordered in apo solution |
| **`DynamicHingeOrAltState`** | Ordered discordance ($\Delta\theta_i \ge 60^\circ$ or `H <-> E` flip) or low `pLDDT` with rigid NMR ($S^2_{\text{exp},i} \ge 0.72$) | Conformational switch, domain hinge, or homomeric oligomerization interface |

```bash
ferric af-validate \
  --structure model.cif \
  --pae pae.json \
  --shifts assigned.str \
  [--s2 modelfree.tsv] \
  --exclude-pdb 1UBQ,1D3Z --exclude-bmrb 17769 \
  -o af_concordance.json
```

---

## 2. Clean-Room 3D Structure-to-Chemical-Shift Forward Predictor (`T21.2`)

`nmr-core::struct_shifts` (`ferric predict-shifts`) predicts backbone ($\text{H}^{\text{N}}, {}^{15}\text{N}, {}^{13}\text{C}^\alpha, {}^{13}\text{C}^\beta, {}^{13}\text{C}', {}^{1}\text{H}^\alpha$) and optional `ILVMAT` sidechain methyl ($^1\text{H}^{\text{m}}, {}^{13}\text{C}^{\text{m}}$) chemical shifts via an additive physical decomposition (Shen & Bax 2010):

$$\delta^{\text{struct}}_{i,a}(\mathbf{X}) = \delta^{\text{RC}}_{i,a}(T, \text{pH}) + \Delta\delta^{\text{tor}}_{i,a}(\phi_i, \psi_i, \phi_{i-1}, \psi_{i-1}, \phi_{i+1}, \psi_{i+1}, \chi_{1,i}) + \Delta\delta^{\text{HB}}_{i,a}(\mathbf{X}) + \Delta\delta^{\text{RC-arom}}_{i,a}(\mathbf{X}) + \Delta\delta^{\text{pep/EF}}_{i,a}(\mathbf{X})$$

1. **Sequence/pH/Temperature Random-Coil Baseline (`ncIDP`)**: Tamiola et al. (2010) neighbor-corrected coil shifts (including pre-Proline `-1.95 ppm` on `CA`/`C'`), Platzer et al. (2014) Henderson–Hasselbalch pH titration, and Kjaergaard et al. (2011) linear temperature coefficients.
2. **Firewalled Trigonometric Fourier Torsion Hypersurfaces**: Hierarchical Cholesky ridge regression ($\lambda = 2.5$) across `5` structural residue classes (`General`, `Gly`, `Pro`, `PrePro`, `BetaBranched`) fitted strictly to `HomologFirewall`-retained chains (`max_seq_identity <= 0.25`, `TM-score < 0.50`).
3. **Haigh–Mallion (`1972`) / Case (`1995`) Conjugated Ring Currents**: Signed triangular-projection shielding across `Phe`, `Tyr`, `His`, and `Trp` 5-/6-membered $\pi$-rings within $12.0\text{ \AA}$ (`22.3%` proton RMSE reduction on ring-proximal spins $\le 5.5\text{ \AA}$).
4. **Wagner–Pardi–Bax (`1983`) $r_{\text{HO}}^{-3}\cos^2\theta_{\text{NHO}}$ Hydrogen Bonds & McConnell/Buckingham Peptide Anisotropy/Electric Fields**: Captures secondary-structure H-bond deshielding and through-space carbonyl dipole / $\text{C–H}, \text{N–H}$ bond-parallel polarization.
5. **Sahakyan (`2011`) `ILVMAT` Methyl $\gamma$-Gauche & Smooth `pLDDT`-Sigmoid Blending**: Blends structural and coil predictions smoothly via $w_{\text{AF}}(\text{pLDDT}_i) = [1 + \exp(-(\text{pLDDT}_i - 70)/8)]^{-1}$, converging to `ncIDP` coil shifts (`< 1e-6 ppm`) in disordered regions (`pLDDT -> 0`). Across the 9-protein non-homologous folded BMRB/PDB benchmark, out-of-sample RMSEs reach **`0.985 ppm` (`CA`)**, **`1.140 ppm` (`CB`)**, **`1.009 ppm` (`C'`)**, **`0.505 ppm` (`H^N`)**, **`2.804 ppm` (`15N`)**, and **`0.270 ppm` (`HA`)**.

```bash
ferric predict-shifts \
  --structure model.cif \
  --ph 6.5 --temp-k 298.15 --methyl-ilvmat \
  --exclude-pdb 1UBQ,1D3Z --exclude-bmrb 17769 \
  -o predicted_shifts.str
```

---

## 3. Stacked Vector Secondary-Structure Ribbon & 3D Putty Modes (`T21.3`)

Inside `ferric-gui`, `nmr-gui::ss_ribbon_widget` renders a three-track vector secondary-structure and concordance ribbon synchronized with the 2D `HSQC`, 3D strip walks, and 3D `wgpu` cartoon viewer:

- **Track 1 (`3D SS`)**: Kabsch–Sander sinusoidal coral helices (`#E05A47`), directional teal $\beta$-strand arrows (`#2A9D8F`), `PPII` diamond ticks (`#7B61FF`), and coil baseline.
- **Track 2 (`NMR SS`)**: Incremental `< 1.0 ms` solution NMR `SSP` propensity bars (`SSP > 0` upward $\alpha$-helix, `SSP < 0` downward $\beta$-strand, hollow dashes on unassigned residues).
- **Track 3 (`AF vs NMR`)**: Color-vision-safe `AfNmrConcordanceClass` blocks (emerald `ConcordantOrdered`, slate `ConcordantDisordered`, hatched amber `ConditionalFoldingOrCrystalArtifact`, double-bordered crimson `DynamicHingeOrAltState`). Clicking any residue column fires `SyncCrosshair` to center the 2D/3D strips and 3D camera simultaneously. In `mol3d_viewer`, selecting **`AlphaFold pLDDT`** applies the canonical 4-band Varadi (2022) palette (`#0053D6` / `#65CBF3` / `#FFDB13` / `#FF7D45`), while **`AF vs NMR Concordance`** highlights discordant loops and hinges with putty radius scaling.

---

## 4. Opt-In 3D-Prior-Assisted Backbone & `PAE`-Gated `15N-NOESY` Assignment (`T21.4`)

For sparse `HNCA`-only or high-molecular-weight perdeuterated proteins where `13CB` transfer fails, `ferric assign --af-structure <model.cif>` (`nmr-core::af_assign_priors`) injects tempered 3D priors without risking locally wrong AlphaFold loops overriding true sequential triple-resonance walks:

1. **Unlabelled Reference Offset Registration**: Estimates per-nucleus offsets $(\hat{\delta}^{\text{off}}_{\text{H}}, \hat{\delta}^{\text{off}}_{\text{N}}, \hat{\delta}^{\text{off}}_{\text{CA}}, \hat{\delta}^{\text{off}}_{\text{CB}}, \hat{\delta}^{\text{off}}_{\text{C}'})$ strictly between unordered observed spin systems and unordered `T21.2` predictions (`< 1e-12` permutation-invariant).
2. **Heavy-Tailed Student-$t_4$ Clamped Unary Log-Odds**: Sums Student-$t(\nu = 4)$ log-likelihood ratios across observed intra-residue nuclei, weights by $w_{\text{AF}}(\text{pLDDT}_k)$, and clamps strictly to $[-U_{\max}^{\text{clamp}}, +U_{\max}^{\text{clamp}}]$ (`--af-clamp-nats 4.0`, `--af-weight 0.65`, `max |\Delta U_{\text{AF}}| \le 2.60 nats`), matching CPU `f64` and WebGPU `scoring.wgsl` to `< 1.0e-5`. Corrupting a 12-residue AlphaFold loop by $120^\circ$ in $(\phi, \psi)$ causes **`0.0%` accuracy loss** on full triple-resonance Ubiquitin.
3. **`PAE <= 6.0 Å` Gated `3D 15N-NOESY` Amide Contacts**: Admits predicted $d(\text{H}^{\text{N}}_j, \text{H}^{\text{N}}_k) \le 5.5\text{ \AA}$ contacts only when $\text{pLDDT}_j, \text{pLDDT}_k \ge 70$ and symmetrized $\overline{\text{PAE}}(j, k) \le 6.0\text{ \AA}$ (or $|j - k| \le 5$ when `PAE` JSON is omitted), preventing uncertain inter-domain contacts from injecting false cross-domain links (`+17.13` percentage-point accuracy gain on `CB`-stripped `HSQC + HNCA + 15N-NOESY` with `evo-graph`).

```bash
ferric assign \
  --peaks sparse_hnca.tsv \
  --seq protein.fasta \
  --af-structure model.cif \
  --af-pae pae.json \
  --noesy-15n noesy15n.tsv \
  --af-weight 0.65 --af-clamp-nats 4.0 \
  --solver evo-graph --engine auto \
  --out assigned_af
```

---

## 5. `PAE` Spectral Domain Clustering & Multi-Domain `RDC`/`PCS`/`PRE` Docking (`T21.5`)

AlphaFold frequently predicts individual globular domains to `< 1.0 Å` backbone RMSD while leaving inter-domain orientations across flexible linkers uncertain (`PAE > 15 Å`). `ferric structure --pae-domains` (`nmr-core::domain_dock`, `nmr-gpu::domain_dock_gpu`) resolves multi-domain solution architectures:

1. **Normalized Laplacian Eigengap Domain Partitioning (Ng, Jordan & Weiss 2001)**: Builds the Gaussian affinity graph $A_{ij} = \exp(-\overline{\text{PAE}}_{ij}^2 / 50)\,\mathbb{I}[\overline{\text{PAE}}_{ij} \le 12\text{ \AA}]$ over ordered residues ($\text{pLDDT}_i \ge 70$), selects rigid domain count $D$ via the symmetric Laplacian eigengap ($\overline{\text{PAE}}_{\text{intra}} \le 5\text{ \AA}$, $|\mathcal{D}_d| \ge 20$), and assigns intervening low-`pLDDT` / high-`PAE` segments as covalent `NeRF` torsion linkers.
2. **Tolman (`2001`) Per-Domain Generalized Degree of Order ($\vartheta_d / \vartheta_1$) Motion Test**: Fits independent 5-parameter Losonczi Saupe tensors per domain and medium to compute $\vartheta_{d,m} = \sqrt{\frac{2}{3}\sum_{\alpha,\beta} (S^{(d,m)}_{\alpha\beta})^2}$. Ratios $\rho_{\text{GDO}} < 0.80$ at $> 2\sigma_{\text{MC}}$ flag **`DecoupledInterDomainMotion`** so large-amplitude inter-domain averaging is never over-interpreted as a single static rigid-body pose.
3. **WebGPU + `f64` `L-BFGS` Linker-Kinematic Hybrid Refinement**: Freezes internal coordinates inside each rigid domain (`0.000 Å` intra-domain distortion) while optimizing linker $(\phi_l, \psi_l)$ torsions (or multi-chain $(\mathbf{q}_d, \mathbf{t}_d)$ rigid bodies) against working multi-medium `RDC`s, lanthanide/metalloprotein `PCS` ($\Delta\boldsymbol{\chi}$, $r_{\text{M}}^{-3}$), Solomon `PRE` ($r_{\text{M}}^{-6}$), and inter-domain `NOE`s under mandatory `ByPeptidePlane` `20%` `RdcFree` holdout.

```bash
ferric structure --pae-domains \
  --af-structure multidomain.cif \
  --af-pae multidomain_pae.json \
  --rdc medium1_pf1.tsv --rdc medium2_peg.tsv \
  --pcs lanthanide_pcs.tsv --pre mtsl_pre.tsv \
  --rdc-free-frac 0.20 --ensemble-size 10 --engine auto \
  -o docked_ensemble.pdb
```

---

## 6. `CSP`/`PRE`-Guided Co-Folding Bridge (`AlphaFold 3` / `Boltz-2`) & Pose Re-Scoring (`T21.6a`)

`nmr-core::cofold_bridge` (`ferric cofold export` and `ferric cofold score`) connects experimental titration `CSP`, `PRE`, `NOE`, `RDC`, and `PCS` restraints to biomolecular complex co-folding:

- **Shrake–Rupley (`rSASA >= 0.20`) Surface Pocket Filtering**: Using 92-point Fibonacci-sphere solvent accessibility on the apo receptor structure, `2\sigma` `CSP` hits (`nmr-core::csp`) are separated into direct surface binding-pocket contacts ($\text{rSASA}_i \ge 0.20$) vs buried **`AllostericCorePerturbation`** relays ($\text{rSASA}_i < 0.20$), preventing buried core shifts from pulling ligands into the protein interior.
- **Dual Schema Export**: Exports deterministic `Boltz-2` YAML (`version: 1`, `constraints: [{pocket: ...}, {contact: ...}]` with `1`-based token indexing and optional `affinity` property) and `AlphaFold 3` JSON (`dialect: "alphafold3"`, `version: 2`, plus companion `<prefix>.nmr_restraints.json`), accepting protein/RNA/DNA FASTA chains alongside `Octet` `SMILES`, `MOL`, or `CCD` ligands.
- **Multi-Pose Complex NMR Re-Scorer (`score_cofold_poses`)**: Ranks candidate `.cif`/`.pdb` complex poses without coordinate distortion by combining intermolecular Haigh–Mallion ring-current (partner aromatic + Hückel-aromatic small-molecule ligand rings) + H-bond + $\Delta\text{rSASA}$ interface `CSP` concordance, $r^{-6}$-averaged intermolecular `NOE`/`PRE` violation penalties, `ByPeptidePlane` `RDC` $Q_{\text{free}}$, and `PCS` $Q$-factors—emitting **`PoseVerdict::InconclusiveOrAllViolated`** whenever all candidate poses violate experimental distance or pocket gates.

```bash
# Export Boltz-2 YAML (with native pocket & contact constraints) or AlphaFold 3 JSON
ferric cofold export \
  --format boltz2 \
  --fasta complex.fasta --ligand-smiles "c1ccc(cc1)C(=O)N" \
  --apo-structure apo_receptor.cif \
  --csp csp_fit.tsv --pre intermolecular_pre.tsv \
  -o complex_boltz2.yaml

# Re-score co-folded candidate complex poses against solution NMR restraints
ferric cofold score \
  --poses pose_0.cif,pose_1.cif,pose_2.cif \
  --apo-structure apo_receptor.cif \
  --csp csp_fit.tsv --pre intermolecular_pre.tsv --rdc complex_rdc.tsv \
  -o pose_ranking.json
```

---

## 7. Bayesian Excited-State (`CEST` / `CPMG` / $R_{1\rho}$) Hypothesis Engine (`T21.6b`)

Relaxation dispersion (`CPMG`, $R_{1\rho}$) and saturation transfer (`CEST`, `DEST`, [Chapter 4](04-dynamics-and-exchange.md)) yield chemical-shift differences $\Delta\varpi_{AB,i,a} = \varpi_{B,i,a} - \varpi_{A,i,a}$ for sparsely populated (`0.5%–10%`) invisible protein conformers. `nmr-core::excited_state` (`ferric exchange-state`) evaluates candidate structural mechanisms against ground-state structure $\mathbf{X}_A$ and shifts $\varpi_A$:

### Exact Normalized Signed & Folded-Unsigned Student-$t_5$ Likelihoods

Evaluating differential shifts $\mu_{\Delta,i,a}^{(H)} = \hat{\delta}_{B,i,a}^{(H)} - \hat{\delta}_{A,i,a}(\mathbf{X}_A)$ cancels static per-residue force-field offsets on ground state $A$. With combined uncertainty $\sigma_{\text{tot},i,a} = \sqrt{\sigma_{\text{exp},i,a}^2 + \sigma_{\text{fwd},a}^2}$ and normalized unit Student-$t_5$ density $f_{t_5}(z) = \frac{8}{3\pi\sqrt{5}}(1 + z^2/5)^{-3}$:

- **Signed `CEST` / Off-Resonance $R_{1\rho}$ (`ExchangeSignMode::Signed`, $\Delta\varpi_{AB} \in \mathbb{R}$)**:

$$\ell_{\text{signed}}\bigl(\Delta\varpi_{AB} \mid \mu_\Delta, \sigma_{\text{tot}}\bigr) = \ln f_{t_5}\!\left(\frac{\Delta\varpi_{AB} - \mu_\Delta}{\sigma_{\text{tot}}}\right) - \ln\sigma_{\text{tot}}$$

- **Unsigned `CPMG` Magnitude (`ExchangeSignMode::UnsignedMagnitude`, $y = \lvert\Delta\varpi_{AB}\rvert \ge 0$, Skrynnikov et al. 2002)**:
  Marginalizes exactly over the unobserved sign on $[0, \infty)$ without dropping normalization constants:

$$\ell_{\text{unsigned}}\bigl(y \mid \mu_\Delta, \sigma_{\text{tot}}\bigr) = \ln\!\left[f_{t_5}\!\left(\frac{y - \mu_\Delta}{\sigma_{\text{tot}}}\right) + f_{t_5}\!\left(\frac{-y - \mu_\Delta}{\sigma_{\text{tot}}}\right)\right] - \ln\sigma_{\text{tot}}$$

Mixed tables combining signed `15N` `CEST` rows with unsigned `13CA`/`13CB` `CPMG` rows are scored seamlessly in a single joint log-likelihood $\ell(H) = \sum_{(i,a)} \ell_{i,a}(H)$.

### Candidate Biophysical Hypotheses ($H_0$–$H_4$) & Kass–Raftery `BIC` Gates

| Code | Hypothesis Family | $k_H$ | Physical Forward Differential Model ($\mu_\Delta$) |
| :---: | :--- | :---: | :--- |
| **`H0`** | `NullOrUnstructuredPerturbation` | `0` | Zero structured conformational shift ($\mu_{\Delta,i,a} = 0$ across all spins) |
| **`H1`** | `LocalUnfolding` | `2` | Cooperative unfolding across contiguous window $[u, v]$ to pH/temperature-corrected `ncIDP` coil ($\mu_\Delta = \delta^{\text{RC}} - \varpi_A$, `0.5` damping at $u-1, v+1$; Vallurupalli, Hansen & Kay 2008) |
| **`H2`** | `XxxProCisTrans` | `1` | Spatial ($\le 8.0\text{ \AA}$) or local ($\lvert i - p \rvert \le 3$) `Xxx–Pro` $\omega_{p-1}$ *trans* $\leftrightarrow$ *cis* switch: Schubert (2002) $\Delta\delta_{\text{CB}} = \pm 2.40$, $\Delta\delta_{\text{CG}} = \mp 2.60\text{ ppm}$ ($\Delta_{\beta\gamma}: 4.5 \leftrightarrow 9.6\text{ ppm}$), `p-1` `CA`/`C'`, and `p+1` amide H-bond/anisotropy switch |
| **`H3`** | `RotamerOrHisTautomerFlip` | `2` | Steric-clash-checked `Phe`/`Tyr`/`Trp`/`His` $(\chi_1, \chi_2)$ sidechain flips ($\le 10.0\text{ \AA}$) via differential Haigh–Mallion ring currents, plus Pelton (1993) `His` $\mathrm{N}^{\delta 1}\text{–H} \leftrightarrow \mathrm{N}^{\varepsilon 2}\text{–H}$ ($\pm 82.0\text{ ppm}$ `ND1`/`NE2`, $\mp 7.2\text{ ppm}$ `CD2`, $\pm 4.8\text{ ppm}$ `CG`) & $\mathrm{His}^0 \leftrightarrow \mathrm{HisH}^+$ transitions |
| **`H4`** | `Alternate3DConformer` | `1` | Candidate 3D conformers (`--alt-structures`, e.g. crystal states or subsampled-MSA AlphaFold ensembles; Wayment-Steele et al. 2024) scored via firewalled $\delta^{\text{struct}}(\mathbf{X}_{B,m}) - \delta^{\text{struct}}(\mathbf{X}_A)$ |

Hypotheses are ranked by Schwarz (`1978`) $\mathrm{BIC}(H) = -2\ell_{\max}(H) + k_H \ln N_{\text{obs}}$ and Kass–Raftery (`1995`) Bayes factors $2\ln\mathrm{BF}_{1,2} = \mathrm{BIC}(H_{(2)}) - \mathrm{BIC}(H_{(1)})$. `classify_excited_state` emits **`ExcitedStateVerdict::Committed(H_{(1)})`** if and only if $N_{\text{obs}} \ge 4$, $2\ln\mathrm{BF}_{1,2} \ge 6.0$, $2\ln\mathrm{BF}_{1,H_0} \ge 10.0$, and reduced $\chi^2_\nu(H_{(1)}) \le 2.5$; otherwise it abstains honestly as **`ExcitedStateVerdict::AmbiguousOrUnmodeledState`**.

```bash
ferric exchange-state \
  --shifts ground_assigned.str \
  --structure ground_model.cif \
  --exchange cest_15n_fit.tsv \
  --exchange cpmg_13c_fit.tsv \
  [--alt-structures alt_closed.cif,alt_hinge.pdb] \
  --ph 6.5 --temp-k 298.15 \
  -o excited_state_report.json
```

---

## 8. Integrative Structural Studio (`nmr-gui`) & Sandboxed `MCP` Tools

- **Desktop & Browser WebGPU Studio (`nmr-gui::integrative_af_panel`)**: Accessible via the **Window** menu, **`Cmd/Ctrl+K`** command palette, or the **3D Structure** dock tab (adding zero permanent chrome bars to the primary spectrum canvas), the studio provides 1-click `.cif` + `PAE` JSON loading, an interactive $N \times N$ green-to-white `PAE` domain heatmap with overlaid spectral domain bounding boxes and Tolman $\vartheta_d / \vartheta_1$ GDO badges, `AlphaFold 3` / `Boltz-2` restraint preview & multi-pose ranking, and a `BIC` / $\Delta\varpi_{\text{obs}}$ vs $\Delta\varpi_{\text{calc}}$ parity explorer for excited states.
- **Workspace-Sandboxed Read-Only `MCP` Tools ([Chapter 6](06-mcp-and-cli-reference.md))**: AI coding and scientific agents can invoke `af_concordance`, `predict_structure_shifts` (enforcing `DEFAULT_FIREWALL_MAX_SEQ_IDENTITY = 0.25` + `exclusion_list_sha256`), `score_cofold_poses`, `classify_excited_state`, and the guided prompt workflow `validate-and-dock-alphafold-model` strictly under `McpAccessMode::ReadOnly` and `McpSecurityPolicy` canonical path containment.
