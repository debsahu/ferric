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

## Real-Data IDP Workflows: Polarity, Root Gating, `CAS-NMR`, `idp-13c` (`v0.17.3`) & Unified Joint `HN + 13C` (`idp-joint`, `v0.18.0`)

Real experimental IDP datasets combine narrow chemical-shift dispersion, high Proline content, fast amide-proton solvent exchange, and pulse-program-specific sign or referencing quirks. `ferric assign` and `ferric pipeline` provide four dedicated controls for real-data IDP processing (`v0.17.3`) together with the unified joint `HN + 2D CON` Proline-bridging regime **`--regime idp-joint`** (`v0.18.0`, specified in `docs/specs/14-regime-solvers.md` §5).

### 1. Automatic `HNCACB` / `HN(CO)CACB` Sign Polarity (`--hncacb-polarity`)

Different Bruker/Varian pulse programs and processing phases produce opposite sign conventions between ${}^{13}\text{C}^\alpha$ and ${}^{13}\text{C}^\beta$ (`CA > 0, CB < 0` vs inverted `CB > 0, CA < 0`), while `CBCA(CO)NH` is often acquired single-sign.

- **`--hncacb-polarity auto` (default)**: Inspects each spectrum independently using unambiguous `Ala CB` (`14.5–22.5 ppm`) and `Thr CB` (`65.5–72.5 ppm`) diagnostic bands (explicitly excluding the `60–65 ppm` `Val`/`Ile`/`Thr CA` overlap region) together with minority-sign and minimum-count safeguards. If an inverted `cb-positive` spectrum is detected, FERRIC normalizes peak signs internally to `CA >= 0, CB < 0` and records the decision in `qc_report.hncacb_polarity`.
- **Manual overrides**: Pass `--hncacb-polarity cb-negative` (standard `CA > 0, CB < 0`) or `--hncacb-polarity cb-positive` (`CB > 0, CA < 0`) when diagnosing sparse or heavily truncated spectra where `auto` conservatively leaves signs unchanged.

### 2. `HNCO`-Corroborated Root Gating & `NH2` Merging (`--roots hnco`)

Raw `2D HSQC` peak tables on disordered proteins often include baseline spikes, water-ridge artifacts, and paired `Asn`/`Gln` sidechain $\text{NH}_2$ doublets (shared ${}^{15}\text{N}$ in `108–115 ppm`, both $^1\text{H}$ in `6.4–7.9 ppm`, $|\Delta\delta_H| \approx 0.7\text{ ppm}$).

- **`--roots all` (default)**: Preserves 1-to-1 `HSQC` root indexing.
- **`--roots hnco`**: Retains only `HSQC` roots corroborated by `3D HNCO` (falling back to any non-`HSQC` 3D correlation if `HNCO` is absent) and merges `Asn`/`Gln` sidechain $\text{NH}_2$ doublet pairs prior to spin-system assembly, preventing sidechain peaks from stealing backbone slots.

### 3. Amino-Acid-Selective `CAS-NMR` Subspectra (`--cas`)

Carbon-detected amino-acid-selective (`CAS`) `2D CON`-readout subspectra (Sahu, Bastidas & Showalter 2014) filter magnetization by amino-acid spin topology and project onto the peptide-bond $(C'_{i-1}, N_i)$ plane without needing multiple selectively labelled samples. Because every `CON` peak bridges two residues $(i-1, i)$, directionality is critical:

| `CAS` Mode | Typed Sequence Slot | Physical Meaning | CLI Example |
| :--- | :---: | :--- | :--- |
| **`cacon`** | **`i - 1`** | Selects the residue **preceding** the `(C'_{i-1}, N_i)` peptide bond | `--cas cacon:ser=cacon_ser.tsv` |
| **`canco`** | **`i`** | Selects the residue **following** the `(C'_{i-1}, N_i)` peptide bond | `--cas canco:ala=canco_ala.tsv` |

- **Repeatable syntax**: Pass `--cas` once per subspectrum (`cacon:<res>=<path>`, `canco:<res>=<path>`, `cacon_<res>=<path>`, or `<res>:canco=<path>`). Residue selectors accept single- or three-letter codes, comma/`+`-separated sets, and standard shorthands (`ser`, `ala`, `gly`, `thr`, `tavi` = `Thr/Ala/Val/Ile`, `asx` = `Asp/Asn`, `glx` = `Glu/Gln`, `ilv` = `Ile/Leu/Val`). Always state `cacon:` or `canco:` explicitly; ambiguous inputs fail closed instead of silently defaulting.
- **Overlap & breakthrough calibration**: `CAS` peaks are scored via overlap-fanout-damped, breakthrough-calibrated log-likelihood ratios (`c13_detected::cas_llr_adjustment`) with bounded asymmetric mismatch penalties (`~ -0.74 nats` clean, softening under excess hits/crowding) so crowded `CON` clusters never override corroborated `CA`/`CB` sequential walks.
- **Regime usage**: In `--regime globular`, `--cas` routes through CPU `f64` `--solver standard`. In `--regime idp-13c` and `--regime idp-joint`, `CAS` subspectra combine directly with `2D CON`, `H(CC)CON`, `(H)N(CA)CON`, and `HN` triple-resonance channels.

### 4. `13C`-Detected `idp-13c` Workflow (`v0.17.3`)

`--regime idp-13c` builds continuous through-Proline `CON(i)` dipeptide walks from `2D CON` roots plus `3D H(CC)CON` / `CBCACON` / `CANCO` / `NCACO`, repeatable `--cas` subspectra, directional `3D (H)N(CA)CON` (`--hncacon`) `N–N` edges, and optional reciprocal `3D BT_HNCACONH` (`--hncaconh`) `N–N` links, guarded by capped unlabelled auto-referencing (`~3.5 ppm` $^{13}\text{C}$, `~1.5 ppm` $^{15}\text{N}$, with TMS-vs-DSS and carbon-spread polarity hints):

```bash
# 13C-detected CON-rooted IDP workflow with CAS + (H)N(CA)CON
ferric assign --seq idp.fasta --regime idp-13c \
  --con con_2d.tsv \
  --cccon hcccon_3d.tsv \
  --cas cacon:ser=cacon_ser.tsv \
  --cas cacon:gly=cacon_gly.tsv \
  --cas canco:ala=canco_ala.tsv \
  --hncacon hncacon_3d.tsv \
  --hncacb-polarity auto \
  --temp-k 298.15 --ph 6.5 \
  --engine auto --out out/idp_13c
```

> [!IMPORTANT] `--engine gpu` Fail-Closed Policy for `idp-13c` Extensions & `idp-joint`
> Pure `13C`-detected runs (`--con` + `--caco`/`--canco`/`--cbcacon`/`--cccon`/`--ncaco`) execute on WebGPU via `--engine gpu` or `--engine auto`. When CPU-only `f64` extensions are active (`--cas`, `--hncacon`, `--hncaconh`, or `HNCO`$\to$`HNCACB` attachment) — or whenever `--regime idp-joint` is selected — **`--engine gpu` fails closed with an explicit hard error (zero silent CPU fallback)**; pass `--engine cpu` or `--engine auto` instead.

### 5. Unified Joint `HN + 2D CON` Proline-Bridging Solver (`--regime idp-joint`, `v0.18.0`) & `HN` $\leftrightarrow$ `CON` Cross-Validation

When an IDP project includes **both** `1H`-detected triple-resonance spectra (`BEST-HSQC`, `HNCO`, `HN(CA)CO`, `HNCACB`, `HN(CO)CACB` / `CBCA(CO)NH` / `HNCOCACB`, `BT_HNCACONH`) and `13C`-detected spectra (`2D CON`, `H(CC)CON`, `(H)N(CA)CON`, `CAS`), **`--regime idp-joint`** (`crates/nmr-core/src/joint_idp_solver.rs`, `docs/specs/14-regime-solvers.md` §5) unifies both routes into a single peptide-bond graph. If `--regime` is omitted on `ferric assign` or `ferric pipeline` and both `1H`-amide (`--peaks` / `--hsqc`) and `--con` inputs are supplied, FERRIC **auto-selects `--regime idp-joint`**:

```bash
# Unified joint HN + 2D CON + Proline-bridge + CAS + N-N IDP assignment (CPU f64)
ferric assign --seq idp.fasta --regime idp-joint \
  --peaks hn_triple_peaks.tsv \
  --con con_2d.tsv \
  --cccon hcccon_3d.tsv \
  --cas cacon:ser=cacon_ser.tsv \
  --cas canco:ala=canco_ala.tsv \
  --hncacon hncacon_3d.tsv \
  --hncaconh bt_hncaconh_3d.tsv \
  --roots hnco --hncacb-polarity auto \
  --temp-k 298.15 --ph 6.5 \
  --engine cpu --out out/idp_joint
```

Key algorithmic safeguards in `--regime idp-joint`:
1. **Exact Hungarian `(C'_{i-1}, N_i)` Fusion & Proline Bridging**: Solves noise-scaled 1-to-1 Hungarian matching between `HNCO` amide roots and `2D CON` peaks to create **`AmideFused`** nodes, preserves Proline **`ProlineConBridge`** nodes (`15N` **`131.5–144.5 ppm` DSS**, no $^1\text{H}^{\text{N}}$) bridging directly across `-Xaa-Pro-` and `-Pro-Pro-` junctions, and retains unmatched non-Proline **`ConOnlyAmide`** nodes across exchange-broadened amide gaps.
2. **Crowded-HSQC-Robust TROSY `46 Hz (1/2 1J_NH)` Calibration & IDP Strip Hygiene**: Auto-calibrates field-scaled TROSY offsets — `46 Hz (1/2 1J_NH)`, field-scaled to ppm (`-0.058 ppm 1H` / `+0.567 ppm 15N` at `800 MHz`; Pervushin et al. 1997) — across all four sign quadrants using 1-to-1 coincidence matching minus an empirical crowded-HSQC decoy baseline (`>= 5` matched roots), prunes multi-peak symmetric `F1(C')` NUS sidelobes while keeping genuine 2-peak `(C'_i, C'_{i-1})` doublets intact, guards `Gly CB` bleed-through (`CA` in `41.0–46.5 ppm` DSS), applies all four pre-Proline shift offsets (`C' -1.90`, `CA -1.80`, `N +1.65`, `CB -0.35 ppm`) plus the Platzer/`ncIDP` C-terminal `COO-` carbonyl prior (`~179.0 ppm` DSS; Platzer et al. 2014), and runs conservative 2-pass anchor-lock propagation without any Pass-2 confidence floor.
3. **Predictor-Mismatch QC, Non-Circular `JOINT_IDP_CALIBRATION_KNOTS` & Explicit Provisional `ABSTAIN` Outputs**: Evaluates 2-swap, cycle, and `2..=4`-residue motif-swap competitors with bilateral $(i-1 \to i \to i+1)$ / energy-gap rank separation and a Bayesian model-misspecification ceiling (`eps_model`), audits per-residue-type shift bias against `ncIDP` (`evaluate_predictor_mismatch_qc` $\to$ `QC = WARN`), maps posteriors through non-double-discounted PAV knots fitted on `7,811` non-`ABSTAIN` predictor-error samples across `4` disjoint synthetic sequences (`top knot = 0.932`, candidly reflecting the `~91–94%` moderate/realistic IDP ceiling vs the `95%` folded globular target), and enforces low-complexity repeat (`QC = ABSTAIN`, `<= 0.15`), decoy/root-inflation (`>= 1.35x`), fusion-scatter, and ambiguity gates:
   - **Fair 3-seed synthetic benchmarks (`0` ground-truth root hints, equal `0.14/0.50` 3D-strip anchoring)**: `idp-joint` achieves `93.0%` acc / `93.7%` Pro / `96.5%` HC prec (`125.0` HC/seed) vs `65.5%` acc / `82.5%` Pro / `96.8%` HC prec (`74.0` HC/seed) for fair standalone `idp-13c` and `54.4%` acc / `0.0%` Pro / `92.8%` HC prec (`65.0` HC/seed; `73.4%` assignable ceiling due to Prolines + exchange-broadened amides) for `evo-graph`; on the OOF predictor-bias probe (`N ±1.0`, `C'/CA/CB ±0.4 ppm`, `2.2x` coil), OOF `ECE = 0.030`, `0/8` `QC = PASS`, `84.1%` HC precision, and unbiased `2.2x` ranking separates `[0.8, 0.9)` (`91.9%`) from `[0.9, 1.0]` (`100.0%`, `ECE = 0.048`).
   - **Provisional `ABSTAIN` handling & unlabelled Clec16a replay**: Under `QC = ABSTAIN`, CLI outputs are explicitly marked provisional in STAR (`_Assigned_chem_shift_list.Details`), TSV (`# PROVISIONAL_OUTPUT: QC=ABSTAIN` + `qc_verdict` column), JSON (`"provisional": true`), NEF comments, and stderr advisories — and the GUI withholds `session.state.assignments` behind a **`PROVISIONAL - QC=ABSTAIN (Assignments Withheld)`** banner. On unlabelled Clec16a replay (`355` nodes, `159` residues with shifts, `24/24 Pro`, `0` HC, `dC = -2.837` / `dCA = -2.734` / `dCB = -2.651` / `dN = +0.399 ppm`), `idp-joint` honestly returns `QC = ABSTAIN` under `2.25x` root inflation (**synthetic accuracy benchmarks only — zero real-data ground-truth accuracy claim**; when `idp-joint` abstains on inflated real data, prefer `--regime idp-13c` for initial anchors).
4. **`HN` $\leftrightarrow$ `CON` Cross-Validation in `ferric-gui` (`idp_joint_panel.rs`)**:
   The desktop and WebAssembly workbench wires `IdpJointPanelState` directly into `NmrWorkbenchApp`, `ingest_panel.rs` (strictly requiring `HN` + `2D CON` to auto-select `idp-joint`, matching CLI `[auto-regime]` on `assign` and `pipeline`, and populating validated `CAS` slots up to `MAX_CAS_SLOTS = 16`), the sequence ribbon, and the 3D strip tiler — automatically clearing stale route badges (`clear_joint_route_state()`) across every sequence/ingest/reset/non-joint solve transition and emitting a runnable `cli_preview_command` matching `ferric assign`:
   - **Start with `XVal` (`CrossValidatedHnAndCon`) stretches and `[Pro]` (`ConProBridge`) junctions**: Residues corroborated by both `HN` and `CON` routes (and Proline bridges linking two cross-validated flanks) form the primary verification scaffold.
   - **Audit `HN` (`HnRoute`) and `CON` (`ConRouteOnly`) transitions**: Click any residue in the concordance table to jump straight to its 3D strip walk, inspect `CAS` (`cacon` `i-1` vs `canco` `i`) typing and C-terminal `COO-` hints, lock confirmed anchors, and re-run `assign_joint_idp_cpu` live.

---

## Choosing the Right Solver

| Scenario | Recommended Flag | Rationale |
| :--- | :--- | :--- |
| Raw picked 3D spectra (`5–250 aa`) with `NH2` / baseline noise | `--solver evo-graph` | Stoichiometric peak budgeting + 2-pass factor-graph belief propagation handles noisy real peak lists cleanly; add `--roots hnco` and `--hncacb-polarity auto` on noisy/IDP `HSQC`s |
| Interactive fast GPU re-annealing with user locks (`--locks`) | `--solver standard --engine gpu` | Sub-second WebGPU `16 KiB` SRAM turnaround with strong empirical calibration |
| Large single-chain globular proteins (`250–800 aa`) or sparse gaps | `--solver bayes-sa --engine gpu` | Null chemical potential $\mu_{\text{null}}$ + rare-shift Bayes factors + 3-tier `anneal_bayes_256`/`1024`/`global` GPU kernels |
| Proline-rich IDPs with `2D CON` + `3D CANCO`/`CBCACON` (pure `13C`) | `--regime idp-13c --engine gpu` | Unbroken `CON(i)` dipeptide walk through Prolines with T/pH `ncIDP` coil priors ([Chapter 1](01-quickstart-and-workflows.md#protein-workflow-3-proline-rich-idps-13c-detected-protonless-nmr)) |
| Proline-rich IDPs with `2D CON` + `CAS` + `(H)N(CA)CON` (`13C`-only) | `--regime idp-13c --engine auto` | Combines `CON`, `H(CC)CON`, `--cas`, and `--hncacon` on CPU `f64` |
| Proline-rich IDPs with **both** `1H` triple-resonance and `2D CON`/`13C` (`v0.18.0`) | `--regime idp-joint --engine cpu` | Exact Hungarian `HN ↔ CON` fusion, Proline bridging (`131.5–144.5 ppm`), TROSY `46 Hz (1/2 1J_NH)` alignment, calibrated `CAS`, and `XVal`/`[Pro]` cross-route reporting (auto-selected when both `HN` and `--con` are provided) |
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
