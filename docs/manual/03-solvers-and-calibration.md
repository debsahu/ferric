# Chapter 3: Backbone Solvers, Dataset QC & Honest Calibration

FERRIC provides three complementary, 100% clean-room backbone resonance assignment engines—**`standard`**, **`evo-graph`**, and the opt-in experimental Bayesian statistical-mechanics solver **`bayes-sa`**—backed by unified dataset quality control (`nmr-core::qc`) and isotonic confidence calibration. Every solver outputs shifts strictly in the experimental spectrum's ppm frame and obeys the **sequence-only invariant** (zero 3D coordinates or homolog templates used during sequence-based backbone assignment).

> [!TIP] Interactive Tour & HD Video (Synthetic / Dev · Cold HCP Gate Failed & Withdrawn)
> Launch **[Module 02: HSQC Strips & Backbone Solvers](tours/02_hsqc_strips_solvers.html)** — **synthetic** Ubiquitin (`--realistic` `98.4%`, `310/315`), **dev** `CISD3` (`BMRB 51439`: `64.38%`, `59/59` HC); default `evo-graph` unseen cold micro HC-prec `88.25%–95.30%` (latest single-shot cold cohort scored `88.25%`, which **FAILS** the `>= 94%` gate; cold HCP claim formally **withdrawn** under `13.1:P1-2`). Or browse **[All 8 Module Tours](tours/index.html)**.

---

## Shared Physical Scoring Foundations

Before comparing optimization strategies, all three engines share four essential physical safeguards:

1. **Unlabelled Global Reference-Offset Alignment (`auto_reference = true`)**: Experimental carrier calibrations (TMS vs DSS, indirect $\Xi$ water offsets, uncorrected temperature drift) routinely shift ${}^{13}\text{C}^\alpha/{}^{13}\text{C}^\beta$, ${}^{13}\text{C}'$, or ${}^{15}\text{N}$ by `0.5–3.0 ppm`. `estimate_reference_offsets_detailed` maximizes the outlier-robust marginal mixture log-likelihood of observed spin-system shifts against sequence-weighted CC0 BMRB priors using an aliphatic tether ($\Delta\delta_{\text{CA}} \approx \Delta\delta_{\text{CB}}$, which cancels opposite-sign $\alpha/\beta$ secondary shifts while capturing true referencing offsets) and a log-likelihood-ratio deadband so already-calibrated DSS datasets remain at exact `0.0 ppm`.
2. **Primary-Sequence Covalent Pre-Proline Corrections (`delta_covalent`)**: Residues immediately preceding Proline (`seq[k+1] == PRO`) experience steric and tertiary-imide electronic upfield shifts of `(-1.95, -1.80) ppm` on `(CA, C')` (`(-1.00, -1.50) ppm` for `Gly`, `(-1.75, -1.70) ppm` for `Pro`; Wishart et al. 1995; Schwarzinger et al. 2001; MacArthur & Thornton 1991). Evaluated as a centered `0.85 / 0.15` coil/PPII-vs-helical-kink mixture, pre-Proline slots gain strong upfield selectivity without over-penalizing kinked $\alpha$-helical pre-Proline residues.
3. **Sub-Stoichiometric Minor-Conformer Penalty**: Slow *cis/trans* peptidyl-prolyl isomerization or N-terminal fraying (e.g., Ubiquitin `Gln2` 9% minor conformer) produces genuine but weak `< 15%`-median shadow roots. Intensity-aware root priors prevent minor-state shadows from displacing major-state assignments.
4. **Hard-Locked User Constraints (`--locks`)**: Interactive or CLI-specified `(spin_system -> residue)` locks are enforced strictly across all replicas, island populations, and coordinate-descent polish passes (`confidence = 1.0`).

---

## Solver 1: `standard` (`CPU` `f64` & `WebGPU` Multi-Replica SA)

```bash
# Run standard multi-replica simulated annealing on WebGPU (or --engine cpu)
ferric assign --peaks peaks.tsv --seq protein.fasta --solver standard --engine gpu --out out/std
```

Implemented in `crates/nmr-core/src/solver.rs` and `crates/nmr-gpu/` (`docs/specs/03-assignment.md`):
- **Objective**: Maximizes $\sum_k U(x_k, k) + \sum_k P(x_{k-1}, x_k)$ over injective maps from spin systems to non-Proline, non-N-terminal sequence positions (`Pro` and `k = 0` are strictly forbidden).
- **Unary Typing `U(s, k)`**: Log-likelihood ratio of `(i, i-1)` `CA`, `CB`, `C'`, `N`, `H` shifts against sequence-slot covalent priors normalized by residue-averaged background density $B_X(v)$, with soft `Gly` `CB` mispick floor ($\ln 0.02$) and `CA`-less excess-root gating.
- **Pairwise Sequential Link `P(s, t)`**: Multi-pathway Gaussian shift agreement between intra-residue candidates on strip $s$ and sequential `i-1` candidates on strip $t$, rewarding multi-experiment corroboration.
- **WebGPU `16 KiB` SRAM Execution**: Evaluates `U`/`P` matrices (`< 1e-13` CPU parity) and runs massively parallel independent Metropolis–Hastings simulated annealing replicas directly inside on-chip `16 KiB` workgroup shared memory.

---

## Solver 2: `evo-graph` (Stoichiometric Evolutionary + Factor-Graph BP)

```bash
# Default high-robustness CPU solver for raw picked peak tables
ferric assign --peaks peaks.tsv --seq protein.fasta --solver evo-graph --out out/evograph
```

Implemented in `crates/nmr-core/src/evo_graph.rs` (`docs/specs/10-evo-graph-solver.md`; Schmidt & Güntert 2012; Klukowski, Riek & Güntert 2022):
1. **Sequence-Stoichiometry Peak Budgeting (`filter_peaks_stoichiometric`)**: Computes exact expected peak counts $N_{\text{exp}}(E, \text{seq})$ per experiment, removes paired `Asn`/`Gln` sidechain $\text{NH}_2$ doublets (same ${}^{15}\text{N}$, $\Delta\delta_H \in [0.30, 1.25]\text{ ppm}$, matching `HNCO` carbonyls, zero `CA`), suppresses `t1` sinc sidelobes and unresolved ${}^3J_{\text{HNH}\alpha}$ splittings, and aligns rigid per-cube $(\Delta H, \Delta N)$ drift against `2D HSQC`.
2. **Island-Model Fragment Crossover + SA**: Evolves `64` individuals across `16` islands over `5` generations, recombining contiguous high-scoring sequential peptide fragments via order-preserving crossover interleaved with local Metropolis–Hastings annealing.
3. **Pass-2 Factor-Graph Belief Propagation**: Refines chemical-shift offsets on high-confidence Pass-1 anchors, propagates damped (`T = 1.0 nat`) loopy sum-product marginal beliefs over the sequence factor graph, and prunes inconsistent edge outliers.

> [!NOTE] CPU-Only Precision Architecture
> `evo-graph` executes strictly in `f64` on multi-threaded CPU (`--engine gpu` returns an explicit error suggesting `--engine cpu` or `--solver standard|bayes-sa`).

---

## Solver 3: Opt-In Experimental `bayes-sa` (Bayesian Specific-Heat $C_v(T)$ SA)

```bash
# Opt-in Bayesian pseudo-energy + adaptive specific-heat SA on WebGPU or CPU
ferric assign --peaks peaks.tsv --seq protein.fasta --solver bayes-sa --engine gpu --out out/bayes
```

Implemented 100% clean-room in `crates/nmr-core/src/bayes_sa.rs` and `crates/nmr-gpu/src/bayes_sa_gpu.rs` (`docs/specs/19-bayesian-sa-solver.md`), citing **Bishop, Torres-Montalvo, Kotaru, Mimun & Wand (*Nature Communications* 2023, 14, 1556; DOI: `10.1038/s41467-023-37219-z`)** and **Kirkpatrick, Gelatt & Vecchi (*Science* 1983, 220, 671–680)**, using exclusively our CC0 BMRB prior corpus (`data/priors/bmrb_priors.json`).

### Mathematical Formulation

1. **Conformer-Stratified Heavy-Tail Shift Posteriors**: Decomposes BMRB amino-acid shift distributions across secondary-structure classes $c \in \{\alpha\text{-helix}, \beta\text{-strand}, \text{coil}\}$ with cooperative first-order Markov persistence $P(c_k \mid c_{k-1})$ and evaluates each candidate under a heavy-tail **Student-$t(\nu = 5)$ + uniform spectral-window outlier mixture**:

$$p(x \mid \mu, \sigma_t, R_a) = (1 - \epsilon_{\text{out}}) \frac{\Gamma(3)}{\sqrt{5\pi}\,\Gamma(5/2)\,\sigma_t}\left(1 + \frac{(x - \mu)^2}{5\sigma_t^2}\right)^{-3} + \frac{\epsilon_{\text{out}}}{R_a}$$

   - Extreme `15 ppm` artifacts or ring-current outliers remain bounded below by $\epsilon_{\text{out}} / R_a$, never vetoing a multi-channel match.
   - Unobserved channels (`cnt == 0`) contribute exact `0.0` log-Bayes factor identically in numerator and denominator.

2. **Rare-Shift Sequential Log-Bayes Factors & Null Chemical Potential $\mu_{\text{null}}$**: Sequential links normalize Gaussian overlap against the sequence-shrunk background coincidence density $p_{\text{rand}, a}(\bar{\delta})$:

$$\operatorname{BF}_a(s, t) = (1 - p_{\text{leak}}) \sum_{u, v} w_u^{(0)} w_v^{(-1)} \frac{\mathcal{N}(\delta_u^{(0)} - \delta_v^{(-1)} \mid 0, 2\sigma_a^2)}{p_{\text{rand}, a}(\bar{\delta}_{u,v})} + p_{\text{leak}}, \qquad E_{\text{link}}(s, t) = -\sum_a \ln \operatorname{BF}_a(s, t)$$

   Matches in rare chemical-shift wings (`Gly CA ~ 45 ppm`, `Ser/Thr CB ~ 63–70 ppm`) earn stronger negative link energy than crowded `56 ppm` `CA` matches at identical $\Delta\delta = 0$. Unassigned/null slots ($x_k = -1$) carry exact $E = 0$ relative to calibrated chemical potential $\mu_{\text{null}} > 0$, isolating Proline breaks and exchange-broadened gaps cleanly on `480+ aa` proteins.

3. **Thermodynamic Specific-Heat $C_v(T)$ Adaptive Cooling**: Across $R$ parallel replicas (`32` CPU, `64/128` WebGPU), online Welford intra-replica thermal energy variance $\sigma_E^2(T_m)$ defines the dimensionless specific heat and adaptive cooling schedule:

$$C_v(T_m) = \frac{\langle E^2 \rangle_{T_m} - \langle E \rangle_{T_m}^2}{T_m^2}, \qquad T_{m+1} = \max\!\left(T_{\min},\; T_m \exp\!\left(-\frac{\Delta\beta_0}{\sqrt{1 + \eta\, C_v(T_m)/N_{\text{active}}}}\right)\right)$$

   Cooling automatically decelerates across the cooperative chain-nucleation phase transition at $T_{\text{crit}}$ where $C_v(T)$ peaks, and accelerates through high-$T$ disordered and low-$T$ frozen regimes (`480 aa` WebGPU benchmark completes in `3.55 s` with `< 4.6e-5` `Q27.36` fixed-point CPU/WebGPU energy parity).

> [!WARNING] Why `bayes-sa` Remains Opt-In Experimental
> While `bayes-sa` achieves outstanding speed and accuracy on large proteins (`480 aa`: `94.60%` acc / `97.40%` HCP in `3.55 s`), synthetic realistic Ubiquitin (`97.1%` CPU / `96.7%` GPU acc, `> 99%` HCP), and 26-fold non-cold LOPO (`pooled_ECE_10 = 0.0323` CPU / `0.0369` GPU), on the single-shot unseen `cold_p13_w2` cohort (subsequently relabeled `seen_p13_w2` in `real_manifest.tsv`) it exhibited higher high-confidence overconfidence (`86.18%` CPU / `88.03%` GPU micro-HCP) than `evo-graph` and `standard`. Consequently, `evo-graph` / `standard` remain the defaults, and `--solver bayes-sa` is provided strictly as an **opt-in experimental** solver.

---

## Choosing the Right Solver

| Scenario | Recommended Flag | Rationale |
| :--- | :--- | :--- |
| Raw picked 3D spectra (`5–250 aa`) with `NH2` / baseline noise | `--solver evo-graph` | Stoichiometric peak budgeting + 2-pass factor-graph belief propagation handles noisy real peak lists cleanly |
| Interactive fast GPU re-annealing with user locks (`--locks`) | `--solver standard --engine gpu` | Sub-second WebGPU `16 KiB` SRAM turnaround with strong empirical calibration |
| Large single-chain globular proteins (`250–800 aa`) or sparse gaps | `--solver bayes-sa --engine gpu` | Null chemical potential $\mu_{\text{null}}$ + rare-shift Bayes factors + 3-tier `anneal_bayes_256`/`1024`/`global` GPU kernels |
| Proline-rich IDPs with `2D CON` + `3D CANCO`/`CBCACON` | `--regime idp-13c --engine gpu` | Unbroken `CON(i)` dipeptide walk through Prolines with T/pH `ncIDP` coil priors ([Chapter 1](01-quickstart-and-workflows.md#protein-workflow-3-proline-rich-idps-13c-detected-protonless-nmr)) |
| `[2Fe-2S]`, high-spin $3d$, or lanthanide metalloproteins | `--regime paramagnetic --solver evo-graph` | `u_para` blind-zone prior prevents forcing noise roots into `PRE`-bleached coordination spheres |

---

## Isotonic Confidence Calibration & Dataset QC Gates

Raw replica consensus fractions or factor-graph marginals are **not** calibrated probabilities on real experimental spectra with missing experiments, incomplete spin systems, or referencing drift. `nmr-core::qc` applies a two-tier statistical calibration and quality-control pipeline:

### 1. Dataset-Level QC Verdicts (`Pass`, `Warn`, `Abstain`)

`evaluate_dataset_qc` inspects observable completeness, corroborated sequential connectivity (`P >= 3.5` nats), bidirectional aliphatic fraction (`min_bidir_aliphatic_frac = 0.58` in `CANONICAL_PROTEIN_HC_GATE`), unlabelled offset log-likelihood gain per observation, and truncated-FID status:
- **`DatasetQcVerdict::Pass`**: Full per-residue calibrated confidences in `[0, 1]` emitted; calls with `confidence >= 0.80` qualify as **High-Confidence (`HC`)**.
- **`DatasetQcVerdict::Warn`**: Emits calibrated confidences while setting `warn_sub_hc_uncalibrated = true`, explicitly flagging sub-HC (`< 0.80`) assignments as uncalibrated tentative hypotheses.
- **Protein-Level HC Cap / `Abstain`**: On severely incomplete or uncorroborated datasets (such as `CA`-missing / single-pathway failures like `bmr5691`, `bmr18561`, `bmr7175`, or `bmr18662`), caps maximum confidence below `0.80` (`0` HC calls emitted) rather than outputting false high-confidence assignments.

### 2. Engine-Specific 10-Bin PAV Isotonic Calibration

Each of the 5 solver/engine pairs (`EVO_GRAPH`, `STANDARD_CPU`, `STANDARD_GPU`, `BAYES_SA_CPU`, `BAYES_SA_GPU`) maps raw scores through dedicated Pool-Adjacent-Violators (`PAV`) monotone piecewise-linear knots (`UNIFORM_CALIBRATION_GRID_10`) regularized toward conservative `Beta` priors (`std_gpu_w = 0.00` pure WebGPU `f32` `Beta(0.35, 0.65)` knots; `bayes_gpu_w = 0.20`).

On the **26-protein non-cold development/seen pool** (`13 dev + 7 seen_p12 + 6 seen_p13`) evaluated via 26-fold Leave-One-Protein-Out (`LOPO`) cross-validation, all 5 engines pass the active non-cold calibration gates (`pooled_ECE_10 <= 0.050` and residue-weighted `micro_HCP >= 94.0%`):

| Engine | 26-Fold OOF `pooled_ECE_10` | 26-Fold OOF `micro_HCP` (`conf >= 0.80`) |
| :--- | :---: | :---: |
| `evo-graph` (`CPU f64`) | `0.0222` | `95.62%` (`4455 / 4659`) |
| `standard` (`CPU f64`) | `0.0237` | `94.07%` (`2284 / 2428`) |
| `standard` (`WebGPU f32`) | `0.0340` | `94.61%` (`2248 / 2376`) |
| `bayes-sa` (`CPU f64`, opt-in) | `0.0323` | `94.21%` (`3891 / 4130`) |
| `bayes-sa` (`WebGPU f32`, opt-in) | `0.0369` | `94.40%` (`3020 / 3199`) |

*(Methodological notes: (a) Earlier 13-fold `dev` gates `macro_HCP >= 95.0%` and per-protein `mean_debiased_ECE_10 <= 0.065` were formally retired on the heterogeneous 26-protein cohort because equal `1/N` protein weighting over-weights near-abstention entries with a handful of HC calls and small-`N` 10-bin occupancy variance inflates single-protein ECE; (b) `min_bidir_aliphatic_frac` was designed after inspecting `seen_p12` negatives `5691` and `18561`, so the 26-fold LOPO refit is out-of-fold with respect to thresholds/knots but **in-sample with respect to feature design**.)*

---

## Honest Unseen-Protein (Cold-Cohort) Empirical Precision Caveats

> [!IMPORTANT] Unseen-Protein High-Confidence Precision (`conf >= 0.80`) Caveat
> **FERRIC does NOT claim that `confidence >= 0.80` guarantees `>= 94%` precision on arbitrary unseen experimental proteins.**
>
> Real public archival depositions exhibit failure modes that out-of-fold cross-validation on previously inspected cohorts cannot fully anticipate—such as mixed-batch/multi-year referencing mismatches between `2D HSQC` and 3D cubes (e.g., `bmr16238`), unannotated post-acquisition preprocessing, or extreme missing-pathway sparsity (`bmr15819`). Across three strictly pre-registered, single-shot unseen BMRB cohorts (`cold_p12`, `cold_p13`, and `cold_p13_w2` [subsequently relabeled `seen_p13_w2` after single-shot evaluation]), the empirically observed unseen-protein micro high-confidence (`>= 0.80`) precision spans **`86.18%–96.31%`** depending on engine and dataset quality:

| Solver & Engine | `cold_p12` Micro-HCP | `cold_p13` Micro-HCP | `cold_p13_w2` (`seen_p13_w2`) Micro-HCP | Observed Unseen Cold Range |
| :--- | :---: | :---: | :---: | :---: |
| **`evo-graph` (`CPU`)** | `94.54%` (`1005/1063`) | `95.30%` (`1116/1171`) | `88.25%` (`849/962`) | **`88.25% – 95.30%`** |
| **`standard` (`CPU`)** | `91.21%` (`560/614`) | `94.14%` (`498/529`) | `89.14%` (`640/718`) | **`89.14% – 94.14%`** |
| **`standard` (`WebGPU`)** | `96.31%` (`496/515`) | `94.53%` (`484/512`) | `88.58%` (`613/692`) | **`88.58% – 96.31%`** |
| **`bayes-sa` (`CPU`, opt-in)** | — *(pre-`T13.0b`)* | — *(pre-`T13.0b`)* | `86.18%` (`1216/1411`) | **`86.18%`** |
| **`bayes-sa` (`WebGPU`, opt-in)** | — *(pre-`T13.0b`)* | — *(pre-`T13.0b`)* | `88.03%` (`1125/1278`) | **`88.03%`** |

### Best-Practice Curation Checklist for Experimentalists

1. **Always inspect `DatasetQcReport` first**: Check inter-experiment `(1H, 15N, 13C)` referencing offsets (`detailed_offsets`)—if `HNCA`, `HNCACB`, and `CBCACONH` come from different spectrometer sessions or years, align carrier offsets before trusting automated `HC` calls.
2. **Treat `0.80+` as ~88–95% empirical confidence, not 100% ground truth**: Verify sequential `CA`/`CB`/`C'` strip walk junctions in `ferric-gui` (press `st` on boundary residues flanking Prolines, Glycines, or `Warn`-flagged stretches) and lock verified anchor residues (`at` popover or `--locks`) before re-annealing ambiguous regions.
3. **Never treat `< 0.80` calls under `Warn` as calibrated probabilities**: Sub-HC assignments are candidate hypotheses intended to guide interactive strip inspection.
