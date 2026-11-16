# 20. Untargeted 1D NMR Metabolomics, Adaptive Intelligent Bucketing, PCA / OPLS-DA Chemometrics & STOCSY Biomarker Discovery (`T14.5`)

> **Validation Scope (`T14.5` Synthetic Forward-Model Self-Consistency)**: Every quantitative Phase 14 verification gate in this specification is a **synthetic forward-model self-consistency test** over synthesized biofluid cohorts with planted dilution factors, pH shifts, and metabolite concentration perturbations rather than an empirical benchmark on deposited experimental biofluid cohorts (`0` public MetaboLights `MTBLS1` or clinical 1D cohorts evaluated yet — see open follow-up `F17` in `TODO.md` and [Validation Scope (Synthetic Forward-Model Self-Consistency)](#validation-scope-synthetic-forward-model-self-consistency) below).

## Provenance & Clean-Room Statement
Original clean-room specification and mathematical architecture for Phase 14 Slice `T14.5` (`crates/nmr-core/src/metabolomics.rs`, `crates/nmr-cli/src/metabolomics_cmd.rs`, `crates/nmr-gui/src/metabolomics_panel.rs`). Zero source code, binaries, macro scripts, parameter files, or spectral libraries from any proprietary, academic-licensed, or copyleft (GPL/AGPL) metabolomics or chemometrics package were consulted. Every signal-processing operator, multivariate decomposition, cross-validation safeguard, statistical test, and metabolite reference table is derived strictly from peer-reviewed publications and public-domain CC0 entries in the Biological Magnetic Resonance Data Bank (`BMRB`):

1. **Interval Fast Fourier Cross-Correlation Local Peak Alignment**:
   - Savorani, F., Tomasi, G. & Engelsen, S. B. (*J. Magn. Reson.* **2010**, 202, 190–202) — segmented FFT cross-correlation lag estimation with baseline edge continuation to align pH-, ionic-strength-, and temperature-sensitive resonances without distorting multiplet integrals.
2. **Uniform & Adaptive Intelligent Spectral Bucketing**:
   - Holmes, E., Foxall, P. J. D., Nicholson, J. K., Maddock, J., Donarski, J. A., Wilson, I. D. & Lindon, J. C. (*Anal. Biochem.* **1994**, 220, 284–296) — uniform fixed-width spectral integration (`0.04 ppm` / `0.01 ppm`).
   - De Meyer, T., Sinnaeve, D., Van Gasse, B., Tsiporkova, E., Rietzschel, E. R., De Buyzere, M. L., Gillebert, T. C., Bekaert, S., Martins, J. C. & Van Criekinge, W. (*Anal. Chem.* **2008**, 80, 3783–3790) — Adaptive Intelligent Binning (`AIB`): recursive valley-seeking spectral bucketing maximizing the resolution-weighted binness objective $V_b$.
3. **Biofluid Dilution Normalization & Variable Scaling**:
   - Dieterle, F., Ross, A., Schlotterbeck, G. & Senn, H. (*Anal. Chem.* **2006**, 78, 4281–4290) — Probabilistic Quotient Normalization (`PQN`) against a reference spectrum to achieve invariance to both overall sample dilution (e.g., renal water output in urine) and large single-metabolite concentration spikes, alongside Total Spectral Area (`TSA`) and internal-standard (`TSP` / `DSS` at `0.00 ppm`) normalization.
   - van den Berg, R. A., Hoefsloot, H. C. J., Westerhuis, J. A., Smilde, A. K. & van der Werf, M. J. (*BMC Genomics* **2006**, 7, 142) — Mean-Centering, Pareto scaling ($1/\sqrt{s_j}$), and Unit-Variance (`UV`, $1/s_j$) scaling.
4. **PCA, NIPALS `PLS-DA`, `OPLS-DA`, Cross-Validation & Permutation Safeguards**:
   - Hotelling, H. (*Ann. Math. Stat.* **1931**, 2, 360–378); Jackson, J. E. & Mudholkar, G. S. (*Technometrics* **1979**, 21, 341–349) — multivariate Hotelling $T^2$ leverage ellipses and residual Squared Prediction Error ($Q$-residual / `SPE`) critical thresholds.
   - Trygg, J. & Wold, S. (*J. Chemom.* **2002**, 16, 119–128) — Orthogonal Projections to Latent Structures (`O-PLS`).
   - Bylesjö, M., Rantalainen, M., Cloarec, O., Nicholson, J. K., Holmes, E. & Trygg, J. (*J. Chemom.* **2006**, 20, 341–351) — OPLS discriminant analysis (`OPLS-DA`) separating class-predictive variation ($t_p p_p^\top$) from structured within-class orthogonal variation ($T_o P_o^\top$).
   - Westerhuis, J. A., Hoefsloot, H. C. J., Smit, S., Vis, D. J., Smilde, A. K., van Velzen, E. J. J., van Duijnhoven, J. P. M. & van Dorsten, F. A. (*Metabolomics* **2008**, 4, 81–89) — leak-free cross-validated $Q^2$ and $N$-permutation $p$-value assessment for discriminant models when $P \gg N$.
5. **Biomarker Discovery (`S-Plot`, `VIP` & FDR-Adjusted Volcano Ranking)**:
   - Wiklund, S., Johansson, E., Sjöström, L., Mellerowicz, E. J., Edlund, U., Shockcor, J. P., Gottfries, J., Moritz, T. & Trygg, J. (*Anal. Chem.* **2008**, 80, 115–122) — OPLS-DA `S-plot` combining variable magnitude $\operatorname{Cov}(t_p, \tilde{X}_j)$ with effect reliability $\operatorname{Corr}(t_p, \tilde{X}_j)$, plus Variable Importance in Projection (`VIP`).
   - Welch, B. L. (*Biometrika* **1947**, 34, 28–35); Benjamini, Y. & Hochberg, Y. (*J. R. Stat. Soc. B* **1995**, 57, 289–300) — unequal-variance two-sample Welch $t$-test, $\log_2$ fold-change, and step-up False Discovery Rate (`BH-FDR`) Volcano ranking.
6. **Statistical Total Correlation Spectroscopy (`STOCSY`) & Public Biofluid Metabolite Reference Library**:
   - Cloarec, O., Dumas, M.-E., Craig, A., Barton, R. H., Trygg, J., Hudson, J., Blancher, C., Gauguier, D., Lindon, J. C., Holmes, E. & Nicholson, J. K. (*Anal. Chem.* **2005**, 77, 1282–1289) — driver-peak covariance line-shape $\operatorname{Cov}(d, X_j)$ paired with squared Pearson correlation $r^2(d, X_j)$ across a spectral cohort to recover intramolecular spin-system connectivity and co-regulated pathway resonances.
   - Ulrich, E. L. et al. (*Nucleic Acids Res.* **2008**, 36, D402–D408) — BMRB public-domain CC0 small-molecule metabolomics archive (`bmse*`, aqueous pH `7.0–7.4` referenced to `DSS`/`TSP = 0.00 ppm`).

## Validation Scope (Synthetic Forward-Model Self-Consistency)

1. **Synthetic Forward-Model Self-Consistency Only (`T14.5`)**: Every quantitative `T14.5` (`Metabolomics`) verification gate in Phase 14 — including `PQN` dilution and 18$\times$ D-Glucose spike invariance (`< 1%` error on non-spiked background metabolites vs `> 40%` distortion under Total Spectral Area `TSA`), interval FFT cross-correlation local peak alignment rescuing pH-shifted Citrate (`2.54 / 2.68 ppm`) and L-Histidine (`7.09 / 7.87 ppm`) to mean correlation `r > 0.96` while keeping neutral aliphatic doublets untouched (`r > 0.99`), Adaptive Intelligent Binning (`AIB`, De Meyer et al. 2008) preserving `> 95%` single-bucket share on $J$-coupled Lactate and Valine multiplets vs uniform `0.04 ppm` multiplet splitting (`< 0.80`), residual water (`4.65–5.05 ppm`) and urinary urea (`5.55–6.05 ppm`) exclusion masking, deterministic cyclic Jacobi SVD `PCA` + 95% Hotelling $T^2$ leverage and Jackson–Mudholkar $Q$-residual (`SPE`) contaminant detection, NIPALS `OPLS-DA` orthogonal variation filtering ($r(t_{o,1}, y) = 0$, leak-free stratified $k$-fold $Q^2 > 0.85$, $p_{\text{perm}} \le 0.02$) paired with Westerhuis et al. (2008) high-dimensional ($P = 200 \gg N = 24$) null-label anti-overfitting rejection ($Q^2 < 0.10$, $p_{\text{perm}} > 0.10$, `is_validated == false`), Wiklund `S-plot` + `VIP` + `BH-FDR` Welch Volcano `100%` top-6 precision on planted metabolite markers, and Cloarec `STOCSY` driver covariance/correlation (`r^2 > 0.90`, ~`3 : 1` Lactate $\text{CH}_3 : \text{CH}$ stoichiometry) + library matching — evaluates synthetic Lorentzian first-order multiplet biofluid cohorts (`synthesize_biofluid_spectrum_1d`, `generate_demo_biofluid_cohort`) generated from and matched against the **same built-in 28-entry CC0 `BMRB`/literature canonical shift reference table** with planted dilution factors, pH offsets, and metabolite concentration perturbations.
2. **No Real Experimental MetaboLights or Clinical Biofluid Cohort Evaluated Yet (`F17`)**: **`0` real public MetaboLights (e.g., `MTBLS1` human Type-2 diabetes urinary 1D ${}^{1}\text{H}$ NMR study) or deposited experimental biofluid 1D cohorts have been benchmarked yet**. Consequently, Phase 14 `T14.5` metrics validate algorithmic/numerical correctness, leak-free cross-validation and permutation calibration, and deterministic CLI/GUI reproducibility rather than empirical biomarker discovery accuracy under real biological matrix effects, macromolecular protein/lipid broad envelopes, non-ideal water-suppression phase/baseline roll, ionic-strength adduct shifts, or uncatalogued xenobiotics. Pre-registered single-shot evaluation on a public experimental 1D metabolomics cohort (`MTBLS1` or equivalent) is tracked as open post-`v0.10.0` follow-up `- [ ] (A) F17` in `TODO.md`, and the `v0.10.0` `CHANGELOG.md` and Phase 14 review summaries carry this identical synthetic self-consistency qualifier.

---

## 1. Cohort 1D Preprocessing, Referencing & Region Masking

Given a cohort of $N$ processed 1D ${}^{1}\text{H}$ spectra $\{S_i\}_{i=0}^{N-1}$ (`nmr_core::nmr1d::Spectrum1D`) annotated with sample identifiers and group labels $g_i$:

1. **Internal Standard Calibration & Integral Extraction (`TSP` / `DSS` at `0.00 ppm`)**:
   - Prior to solvent/reference masking, if `calibrate_internal_standard` or `NormalizationMethod::InternalStandard` is enabled, locate the parabolic sub-bin apex $\hat{\delta}_{\text{IS},i}$ inside $[\delta_{\text{IS}} - \Delta_{\text{tol}}, \delta_{\text{IS}} + \Delta_{\text{tol}}]$ (default `0.00 ± 0.12 ppm`), optionally shift `axis.first_ppm` by $-\hat{\delta}_{\text{IS},i}$, and integrate the positive singlet area $A_{\text{IS},i} = \int_{-0.08}^{+0.08} \max(S_i(\delta), 0)\,d\delta$.
2. **Optional Per-Spectrum Baseline Flattening**:
   - Delegates to `nmr1d::correct_baseline_1d` (Cobas wavelet mask + Whittaker penalized least squares or Bernstein basis) when `baseline_correction` is requested.
3. **Common Chemical Shift Grid Resampling**:
   - All $N$ spectra are interpolated onto a shared descending uniform grid $\delta_m = \delta_{\max} - m\,\Delta\delta$ ($m = 0, \dots, M-1$, default `[0.50, 9.50]` ppm at $\Delta\delta = 0.001\text{ ppm}$). If a spectrum already matches the grid within $10^{-9}\text{ ppm}$, intensities are copied verbatim.
4. **Solvent, Water & Urea Region Masking (`BiofluidMaskPreset`)**:
   - Points inside any exclusion interval $[\delta_{\text{lo}}, \delta_{\text{hi}}]$ are flagged in a boolean mask `masked[m] = true` and zeroed across all $N$ traces:
     - `SerumPlasma`: internal standard collar `[-0.50, 0.50]` ppm (or `[-0.50, 0.20]` ppm) and suppressed $\text{H}_2\text{O}/\text{HDO}$ `[4.68, 5.00]` ppm.
     - `Urine`: internal standard `[-0.50, 0.20]` ppm, $\text{H}_2\text{O}/\text{HDO}$ `[4.65, 5.05]` ppm, and broad exchangeable urea $\text{CO(NH}_2)_2$ `[5.55, 6.05]` ppm.
     - `AqueousMinimal`: $\text{H}_2\text{O}/\text{HDO}$ `[4.70, 4.95]` ppm only.
     - `Custom(Vec<(f64, f64)>)`: arbitrary user-specified intervals.

---

## 2. Interval FFT Cross-Correlation Local Peak Alignment

Small pH or ionic-strength differences across biofluid samples induce localized peak shifts (notably polyprotic organic acids such as Citrate at `2.54 / 2.68 ppm` and imidazole rings such as Histidine at `7.09 / 7.87 ppm`) while leaving neutral aliphatic methyls unshifted. Following Savorani, Tomasi & Engelsen (*J. Magn. Reson.* **2010**, 202, 190–202):

1. **Cohort Reference Trace $r \in \mathbb{R}^M$**:
   - Default `AlignmentReference::Median`: $r[m] = \operatorname{median}_{i} S_i[m]$, immune to outlier samples and extreme concentration spikes. (`Mean` and `SampleIndex(idx)` are also supported.)
2. **Valley-Snapped Spectral Intervals**:
   - Each unmasked spectral block is partitioned into intervals $I_q = [a_q, b_q]$ of target width $\Delta\delta_{\text{int}}$ (default `0.22 ppm`), snapping interior cut points $b_q$ within $\pm 0.04\text{ ppm}$ to the lowest local minimum of $r$ so interval boundaries never bisect an active multiplet. Explicit user-supplied intervals are also supported.
3. **Zero-Padded FFT Cross-Correlation**:
   - For interval $I = [a, b]$ of length $L = b - a + 1$ and maximum permitted shift $k_{\max} = \lfloor \Delta\delta_{\max} / \Delta\delta \rfloor$ (default $\Delta\delta_{\max} = 0.06\text{ ppm}$):
     - Subtract local baseline floor $\tilde{r}_I = \max(r_I - q_{0.1}(r_I), 0)$ and $\tilde{s}_{i,I} = \max(s_{i,I} - q_{0.1}(s_{i,I}), 0)$. If either $\|\tilde{r}_I\|_2$ or $\|\tilde{s}_{i,I}\|_2$ falls below noise floor $\tau_{\text{align}}$, set shift $\Delta_i = 0$ so baseline noise is never aligned to noise.
     - Zero-pad both segments to $P = \operatorname{next\_pow2}(2 L + 2 k_{\max} + 1)$, evaluate forward DFTs $\hat{R} = \mathcal{F}(\tilde{r}_I)$ and $\hat{S} = \mathcal{F}(\tilde{s}_{i,I})$ via `ops::fft_arbitrary_in_place`, and form the cross-correlation sequence:
       $$c[\tau] = \operatorname{Re}\,\mathcal{F}^{-1}\!\bigl(\hat{R}^* \odot \hat{S}\bigr)[(\tau + P) \bmod P], \qquad \tau \in [-k_{\max}, +k_{\max}].$$
     - Select $\tau^* = \arg\max_{|\tau| \le k_{\max}} c[\tau]$ with parabolic sub-bin refinement $\delta\tau = \frac{c[\tau^*-1] - c[\tau^*+1]}{2(c[\tau^*-1] - 2 c[\tau^*] + c[\tau^*+1])} \in [-0.5, 0.5]$ whenever $|\tau^*| < k_{\max}$.
     - Resample $s_{i,I}[n] \leftarrow s_{i,I}(n + \tau^* + \delta\tau)$ by linear interpolation, filling vacated boundary samples at the quiet edge of $I$ with the flanking 5-point median edge baseline (preventing wrap-around artefacts).

---

## 3. Uniform & Adaptive Intelligent Binning (`AIB`)

### 3.1 Uniform Fixed-Width Binning (Holmes et al. 1994)
Each unmasked contiguous spectral region is subdivided into fixed steps of width $\Delta\delta_{\text{bin}}$ (default `0.04 ppm`). While simple, rigid grid edges frequently cleave $J$-coupled multiplets across adjacent bins.

### 3.2 Adaptive Intelligent Binning (De Meyer et al. 2008)
To place bucket boundaries strictly in inter-multiplet valleys while preserving $J$-coupled multiplets inside single buckets:

1. **Hz-Calibrated Multiplet Envelope $R_{\text{env}}$**:
   - Construct the composite positive reference profile $R_{\text{raw}}[m] = \frac{1}{2}\operatorname{mean}_i \max(S_i[m], 0) + \frac{1}{2}\max_i \max(S_i[m], 0)$ and convolve within each unmasked region with a normalized Gaussian kernel of standard deviation $\sigma_{\text{Hz}}$ (`multiplet_coalesce_hz`, default `6.5 Hz`, i.e. $\sigma_{\text{pts}} = \sigma_{\text{Hz}} / (\nu_{\text{obs}} \Delta\delta)$).
   - Because two equal Gaussian components separated by $d$ coalesce into a single unimodal peak iff $\sigma \ge d / 2$, setting $\sigma_{\text{Hz}} = 6.5\text{ Hz}$ merges intra-multiplet splittings up to $J \approx 13\text{ Hz}$ (e.g., $7.0\text{ Hz}$ Lactate doublets/quartets and Valine doublets) into smooth unimodal mounds while preserving true valleys between distinct resonances separated by $\ge 18\text{ Hz}$ (e.g., Valine $\gamma_1$ at `0.985 ppm` vs $\gamma_2$ at `1.037 ppm`, $\Delta\nu = 31.2\text{ Hz}$ at `600 MHz`).
2. **Recursive `V_b` Valley Optimization**:
   - Given resolution parameter $r \in (0, 1)$ (default $r = 0.5$), minimum bucket width $\Delta\delta_{\min}$ (default `0.020 ppm`), and maximum bucket width $\Delta\delta_{\max}$ (default `0.100 ppm`), define the De Meyer binness objective on $[a, b]$:
     $$V_b(a, b) = \bigl((b - a)\,\Delta\delta\bigr)^r \max\!\Bigl(0,\, \max_{m \in [a,b]} R_{\text{env}}[m] - \max(R_{\text{env}}[a], R_{\text{env}}[b])\Bigr).$$
   - First split at deep baseline stretches where $R_{\text{env}}[m] \le \tau_{\text{base}}$ so isolated multiplet clusters start tightly bracketed by their baseline feet.
   - Recursively inspect candidate interior local minima $v \in [a + n_{\min}, b - n_{\min}]$ of $R_{\text{env}}$ and select $v^* = \arg\max_v \bigl[V_b(a, v) + V_b(v, b)\bigr]$:
     - Subdivide at $v^*$ whenever $V_b(a, v^*) + V_b(v^*, b) > V_b(a, b)$, or when $(b - a)\,\Delta\delta > \Delta\delta_{\max}$ (falling back to the lowest interior point if $(b - a)\,\Delta\delta > \Delta\delta_{\max}$ and no interior local minimum exists).
   - Optionally prune pure baseline noise buckets whose maximum cohort SNR falls below `min_bucket_snr` (default `3.0` for `AdaptiveIntelligent`, `0.0` for `Uniform`).
3. **Trapezoidal Integration**:
   - Each retained bucket $j \in \{0, \dots, P-1\}$ is integrated across samples via the composite trapezoidal rule:
     $$X_{i,j}^{\text{raw}} = \sum_{m = a_j}^{b_j - 1} \frac{S_i[m] + S_i[m+1]}{2}\,\Delta\delta.$$

---

## 4. Normalization & Variable Scaling

### 4.1 Sample Dilution Normalization (`X_raw -> X_norm`)
1. **Total Spectral Area (`TSA`)**:
   $$X_{i,j}^{\text{TSA}} = X_{i,j}^{\text{raw}} \cdot \frac{A_{\text{ref}}}{\sum_{k=1}^P \max(X_{i,k}^{\text{raw}}, 0)}, \qquad A_{\text{ref}} = \operatorname{median}_i \sum_{k=1}^P \max(X_{i,k}^{\text{raw}}, 0).$$
   *Limitation*: A large concentration spike in a single abundant metabolite (e.g., urinary D-Glucose in glycosuria) inflates the denominator $\sum_k X_{i,k}^{\text{raw}}$ and spuriously depresses all unchanged metabolites.
2. **Internal Standard (`TSP` / `DSS`)**:
   $$X_{i,j}^{\text{IS}} = X_{i,j}^{\text{raw}} \cdot \frac{A_{\text{IS,ref}}}{A_{\text{IS},i}}.$$
3. **Probabilistic Quotient Normalization (`PQN`, Dieterle et al. 2006)**:
   - Pass 1 obtains rough scale-equalized profiles $X_i^{(0)} = X_i^{\text{raw}} / \operatorname{median}_{k \in \Omega_+}(X_{i,k}^{\text{raw}})$ so heterogeneous sample dilutions cannot bias the cohort median reference spectrum $x^{\text{ref}}_j = \operatorname{median}_i X_{i,j}^{(0)}$ (rescaled to median cohort amplitude).
   - Over signal-bearing buckets $\Omega_{\text{sig}} = \{j : x_j^{\text{ref}} \ge \tau_{\text{sig}}\}$ (excluding baseline noise buckets per Dieterle et al. 2006), compute per-bucket quotients and the most probable dilution factor $d_i$:
     $$q_{i,j} = \frac{X_{i,j}^{\text{raw}}}{x_j^{\text{ref}}}, \qquad d_i = \operatorname{median}_{j \in \Omega_{\text{sig}}} q_{i,j}, \qquad X_{i,j}^{\text{PQN}} = \frac{X_{i,j}^{\text{raw}}}{d_i}.$$
   - Because the median has a 50% breakdown point, even massive concentration spikes spanning $< 50\%$ of signal-bearing buckets leave $d_i$ and all non-spiked metabolite concentrations invariant.

### 4.2 Variable Scaling (`X_norm -> X_tilde`, van den Berg et al. 2006)
With column mean $\bar{X}_j = \frac{1}{N}\sum_{i=1}^N X_{i,j}^{\text{norm}}$ and sample standard deviation $s_j = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (X_{i,j}^{\text{norm}} - \bar{X}_j)^2}$:
- `MeanCenter`: $\tilde{X}_{i,j} = X_{i,j}^{\text{norm}} - \bar{X}_j$.
- `Pareto` (default): $\tilde{X}_{i,j} = (X_{i,j}^{\text{norm}} - \bar{X}_j) / \sqrt{s_j}$ (zeroed if $s_j \le 10^{-15}$).
- `UnitVariance`: $\tilde{X}_{i,j} = (X_{i,j}^{\text{norm}} - \bar{X}_j) / s_j$.

**Leak-Free Cross-Validation Rule**: Inside every cross-validation fold and permutation replicate, $(\bar{X}_j, s_j)$ are estimated **exclusively** on the training split and applied out-of-sample to the held-out split.

---

## 5. Deterministic PCA, Hotelling $T^2$ & $Q$-Residuals (`SPE`)

Given scaled matrix $\tilde{X} \in \mathbb{R}^{N \times P}$ and $A \le \min(N - 1, P)$ components:
1. **Deterministic Cyclic Jacobi SVD**:
   - Diagonalize $\tilde{X}\tilde{X}^\top \in \mathbb{R}^{N \times N}$ (when $N \le P$) or $\tilde{X}^\top \tilde{X} \in \mathbb{R}^{P \times P}$ (when $P < N$) via deterministic cyclic Jacobi rotations to obtain singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r$, orthonormal loadings $P_A = [p_1, \dots, p_A] \in \mathbb{R}^{P \times A}$ ($P_A^\top P_A = I_A$), and orthogonal scores $T_A = \tilde{X} P_A \in \mathbb{R}^{N \times A}$.
   - Canonical sign convention: for each component $a$, find $j^* = \arg\max_j |p_{j,a}|$ and enforce $p_{j^*,a} > 0$ so $(t_a, p_a)$ are bit-reproducible across platforms.
2. **Hotelling $T^2$ & 95% Ellipses**:
   $$s_{t_a}^2 = \frac{\sigma_a^2}{N - 1}, \qquad T_i^2 = \sum_{a=1}^A \frac{t_{i,a}^2}{s_{t_a}^2}, \qquad T_{\text{crit}}^2(\alpha; A, N) = \frac{A (N - 1)}{N - A}\, F_{1-\alpha}(A, N - A).$$
   For any 2D score plane $(a, b)$, the global 95% Hotelling $T^2$ ellipse has semi-axes $r_a = s_{t_a}\sqrt{\frac{2(N-1)}{N-2} F_{0.95}(2, N-2)}$ and $r_b = s_{t_b}\sqrt{\frac{2(N-1)}{N-2} F_{0.95}(2, N-2)}$, supplemented by per-group 95% covariance ellipses around each group mean.
3. **$Q$-Residual (`SPE`, Jackson & Mudholkar 1979)**:
   $$Q_i = \|\tilde{x}_i - P_A t_i\|_2^2, \qquad Q_{\text{crit}}(\alpha) = \theta_1 \left[\frac{z_{1-\alpha}\sqrt{2\theta_2 h_0^2}}{\theta_1} + 1 + \frac{\theta_2 h_0 (h_0 - 1)}{\theta_1^2}\right]^{1/h_0},$$
   where $\theta_r = \sum_{k > A} \lambda_k^r$ ($\lambda_k = \sigma_k^2 / (N - 1)$) and $h_0 = 1 - \frac{2\theta_1\theta_3}{3\theta_2^2}$.

---

## 6. NIPALS `PLS-DA`, `OPLS-DA` & Anti-Overfitting Validation

### 6.1 Single-$y$ NIPALS `OPLS-DA` (`1` Predictive + $A_o$ Orthogonal Components; Trygg & Wold 2002; Bylesjö et al. 2006)
For binary class contrast $y_i \in \{0, 1\}$ with centered $\tilde{y}_i = y_i - \bar{y}$:
1. Predictive weight direction: $w_p = \tilde{X}^\top \tilde{y} / \|\tilde{X}^\top \tilde{y}\|_2$.
2. Setting $X^{(0)} = \tilde{X}$, iterate $k = 1, \dots, A_o$:
   $$t = X^{(k-1)} w_p, \qquad p = \frac{(X^{(k-1)})^\top t}{t^\top t}, \qquad w_{o,k} = \frac{p - (w_p^\top p) w_p}{\|p - (w_p^\top p) w_p\|_2},$$
   $$t_{o,k} = X^{(k-1)} w_{o,k}, \qquad p_{o,k} = \frac{(X^{(k-1)})^\top t_{o,k}}{t_{o,k}^\top t_{o,k}}, \qquad X^{(k)} = X^{(k-1)} - t_{o,k} p_{o,k}^\top.$$
   By construction, $w_{o,k}^\top w_p = 0 \implies t_{o,k}^\top \tilde{y} = 0$ identically for all $k$: orthogonal scores capture pure within-class variation uncorrelated with class membership.
3. Final predictive component on $X_{\text{filt}} = X^{(A_o)}$:
   $$t_p = X_{\text{filt}} w_p, \qquad p_p = \frac{X_{\text{filt}}^\top t_p}{t_p^\top t_p}, \qquad c_p = \frac{\tilde{y}^\top t_p}{t_p^\top t_p}, \qquad \hat{y} = \bar{y} + c_p t_p.$$
4. Variance decomposition:
   $$R^2 X_p = \frac{\|t_p p_p^\top\|_F^2}{\|\tilde{X}\|_F^2}, \qquad R^2 X_o = \frac{\|T_o P_o^\top\|_F^2}{\|\tilde{X}\|_F^2}, \qquad R^2 Y = 1 - \frac{\|\tilde{y} - c_p t_p\|_2^2}{\|\tilde{y}\|_2^2}.$$

### 6.2 Multi-Component NIPALS `PLS-DA`
Standard $A$-component NIPALS deflation ($w_a \propto X^{(a-1)\top} y^{(a-1)}$, $t_a = X^{(a-1)} w_a$, $p_a = X^{(a-1)\top} t_a / (t_a^\top t_a)$, $c_a = y^{(a-1)\top} t_a / (t_a^\top t_a)$). For single-$y$, `PLS-DA(1 + A_o)` and `OPLS-DA(1, A_o)` span the identical Krylov subspace $\mathcal{K}_{1+A_o}(\tilde{X}^\top\tilde{X}, \tilde{X}^\top\tilde{y})$ and yield identical $\hat{y}$ predictions.

### 6.3 Overfitting Safeguards (`Westerhuis et al. 2008`)
When $P \gg N$, in-sample $R^2Y$ approaches $1.0$ even on random noise. Every supervised model therefore enforces two mandatory safeguards:
1. **Leak-Free Stratified $k$-Fold Cross-Validation $Q^2$**:
   - Class-stratified deterministic folds ($k = \min(k_{\text{req}}, n_0, n_1)$) refit variable centering/scaling and model weights strictly inside each training split, predicting out-of-fold $\hat{y}_i^{\text{CV}}$:
     $$Q^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i^{\text{CV}})^2}{\sum_{i=1}^N (y_i - \bar{y})^2} \in (-\infty, 1].$$
2. **Deterministic $N$-Permutation Test (`nmr_core::rng::Rng`)**:
   - Permutes class vector $y \mapsto y^{(m)}$ $N_{\text{perm}}$ times, re-running full leak-free stratified $k$-fold CV on each permuted label vector:
     $$p_{\text{perm}}(Q^2) = \frac{1 + \#\{m : Q^{2\,(m)} \ge Q^{2\,\text{obs}}\}}{1 + N_{\text{perm}}}.$$
   - A model is marked `is_validated = true` iff $Q^2 \ge 0.40$ AND $p_{\text{perm}}(Q^2) \le 0.05$; otherwise `OverfitWarning` is raised in both data output and GUI banner.

---

## 7. Biomarker Ranking (`S-Plot`, `VIP`, & `BH-FDR` Volcano)

1. **OPLS-DA `S-Plot` (Wiklund et al. 2008)**:
   $$p(\text{cov})_j = \frac{t_p^\top \tilde{X}_{:,j}}{N - 1}, \qquad p(\text{corr})_j = \frac{p(\text{cov})_j}{s_{t_p}\, s_{\tilde{X}_j}} \in [-1, +1].$$
2. **Variable Importance in Projection (`VIP`)**:
   - PLS-DA: $\operatorname{VIP}_j = \sqrt{P \cdot \bigl(\sum_{a=1}^A \text{SSY}_a w_{j,a}^2\bigr) / \bigl(\sum_{a=1}^A \text{SSY}_a\bigr)}$.
   - OPLS-DA predictive VIP: $\operatorname{VIP}_{\text{pred},j} = \sqrt{P}\, |w_{p,j}|$ ($\frac{1}{P}\sum_j \operatorname{VIP}_{\text{pred},j}^2 = 1$, canonical importance threshold $\operatorname{VIP} > 1.0$).
3. **Univariate Volcano (`Welch t + BH-FDR + log2 FC`)**:
   - Evaluated on normalized concentration-proportional $X^{\text{norm}}$: stabilized $\log_2(\mu_{1,j} / \mu_{0,j})$, two-sided Welch–Satterthwaite Student-$t$ $p_j$ via regularized incomplete beta function $I_{\nu/(\nu+t^2)}(\nu/2, 1/2)$, and step-up Benjamini–Hochberg adjusted $q_j$.
4. **Composite Biomarker Rank Score**:
   - Ranks candidate buckets by $\operatorname{VIP}_j \cdot |p(\text{corr})_j| \cdot \bigl(1 + \min(-\log_{10} q_j, 6) / 6\bigr)$, surfacing biomarkers that combine large spectral flux (`VIP`, `p(cov)`), consistent multivariate class correlation (`|p(corr)|`), and FDR-controlled univariate significance (`q_BH < 0.05`).

---

## 8. `STOCSY` & CC0/BMRB Biofluid Metabolite Library Matching

### 8.1 `STOCSY` Driver Analysis (Cloarec et al. 2005)
Given driver vector $d = X_{:,k^*}^{\text{norm}} \in \mathbb{R}^N$ at driver chemical shift $\delta_{k^*}$:
$$c_j = \frac{1}{N - 1}\sum_{i=1}^N (d_i - \bar{d})(X_{i,j}^{\text{norm}} - \bar{X}_j^{\text{norm}}), \qquad r_j = \frac{c_j}{s_d\, s_{X_j^{\text{norm}}}}, \qquad r_j^2 \in [0, 1].$$
Plotting $c_j$ coloured by $r_j^2$ reconstructs the pure multiplet sub-spectrum and stoichiometric proton ratios of the molecule bearing $\delta_{k^*}$.

### 8.2 Built-In CC0/BMRB Biofluid Metabolite Library (`>= 25` Canonical Metabolites)
Includes 28 curated aqueous (${}^{1}\text{H}$, pH `7.0–7.4`, `DSS = 0.00 ppm`) BMRB reference spin systems (`L-Lactate`, `L-Alanine`, `L-Valine`, `L-Leucine`, `L-Isoleucine`, `alpha-D-Glucose`, `beta-D-Glucose`, `Citrate`, `Succinate`, `Pyruvate`, `Acetate`, `Acetoacetate`, `3-Hydroxybutyrate`, `Creatine`, `Creatinine`, `TMAO`, `Taurine`, `Glycine`, `L-Glutamine`, `L-Glutamate`, `L-Histidine`, `L-Phenylalanine`, `L-Tyrosine`, `Formate`, `Hippurate`, `Myo-inositol`, `Acetone`, `Dimethylamine`).

For a `STOCSY` driver at $\delta_{k^*}$, candidate metabolites containing a resonance within tolerance of $\delta_{k^*}$ are scored by combining:
1. Proton-weighted mean positive `STOCSY` $r^2$ across all non-masked multiplets of the metabolite,
2. Fraction of observable multiplets recovered with $r > 0$ and $r^2 \ge 0.50$,
3. Chemical-shift match Gaussian proximity, and
4. Multi-peak stoichiometric covariance cosine similarity between observed positive $(c_{j_1}, \dots, c_{j_L})$ and theoretical proton counts $(h_1, \dots, h_L)$.

---

## 9. Data Schemas & 4-Quadrant GUI Architecture

### 9.1 Core Rust Types (`crates/nmr-core/src/metabolomics.rs`)
- `CohortSample1D { id, group, spectrum, metadata }`
- `MetabolomicsPipelineConfig` (grid bounds/step, `BiofluidMaskPreset`, `AlignmentConfig`, `BucketingConfig`, `NormalizationMethod`, `ScalingMethod`, `pca_components`, `opls_orthogonal_components`, `cv_folds`, `n_permutations`, `rng_seed`, `case_group`, `control_group`)
- `MetabolomicsStudyResult` (`buckets`, `aligned_spectra`, `raw_bucket_matrix`, `normalized_bucket_matrix`, `scaled_bucket_matrix`, `dilution_factors`, `pca`, `pls_da`, `opls_da`, `biomarkers`, `default_stocsy`)
- Exporters: `export_bucket_table_csv`, `export_biomarker_ranking_csv`, `export_study_summary_json`.

### 9.2 Interactive `wasm32`-Safe Studio (`crates/nmr-gui/src/metabolomics_panel.rs`)
4-quadrant layout inside `WorkspaceMode::Metabolomics`:
- **Quadrant A (Top-Left)**: Cohort 1D spectral overlay / `STOCSY` covariance trace with $r^2$ heat-coloring, exclusion bands, and clickable bucket boundaries.
- **Quadrant B (Top-Right)**: PCA (`PC1` vs `PC2`) / OPLS-DA (`t_p` vs `t_o1`) score plot with 95% Hotelling $T^2$ ellipses and outlier rings.
- **Quadrant C (Bottom-Left)**: Clickable OPLS-DA `S-plot` / `BH-FDR` Volcano plot synced to Quadrant A and live `STOCSY`.
- **Quadrant D (Bottom-Right)**: Ranked candidate biomarker & BMRB metabolite identification table.

### 9.3 Command-Line Interface (`crates/nmr-cli/src/metabolomics_cmd.rs`)
- `ferric metabolomics bin`: cohort preprocessing, water/urea/TSP region exclusion, optional interval FFT peak alignment, Adaptive Intelligent Binning (`AIB`, default) or fixed-width Uniform binning, `PQN` / `TSA` / `TSP` dilution normalization, and `Pareto` / `UV` / `MeanCenter` scaling to CSV/TSV & reusable study JSON.
- `ferric metabolomics pca`: deterministic cyclic Jacobi SVD `PCA`, explained variance ratios, 95% Hotelling $T^2$ leverage limits, and Jackson–Mudholkar $Q$-residual (`SPE`) outlier diagnostics.
- `ferric metabolomics oplsda`: NIPALS `OPLS-DA` (`1` predictive + $A_o$ orthogonal components), leak-free stratified $k$-fold cross-validated $Q^2$, deterministic $N$-permutation $p$-value (`OverfitWarning` / `is_validated`), `VIP`, Wiklund `S-plot`, and `BH-FDR` Welch Volcano biomarker ranking.
- `ferric metabolomics stocsy`: 1D/2D `STOCSY` covariance & correlation ($r, r^2$) trace from a user-selected driver chemical shift (`--driver-ppm`) plus CC0 `BMRB` biofluid library marker matching.
