# Chapter 5: Analytical 1D Multiplets, Hoye J-Trees, qNMR & Pseudo-2D DOSY

Beyond macromolecular multidimensional assignment, `nmr-assign` includes a complete clean-room analytical 1D NMR and diffusion studio (`nmr-core::nmr1d`, `multiplet`, `dosy`, `nmr-gpu::dosy_gpu`, `nmr-gui::stack1d_panel`, `dosy_panel`) tightly integrated with the **`Octet` 2D Chemical Structure Studio**.

> [!IMPORTANT] Validation Scope (Synthetic Forward-Model Self-Consistency)
> Quantitative accuracy numbers in this chapter (Voigt coalesced-shoulder deconvolution, roofed `AB`/`ABX` Hoye `J`-tree recovery, ISO GUM `qNMR` molar purity, and discrete `BIC` / Tikhonov `ILT` `DOSY` diffusivity recovery across `T14.1`–`T14.4`) are **synthetic forward-model self-consistency evaluations** (plus exact $2^N$ quantum spin-$1/2$ Hamiltonian cross-checks). **`0` real deposited raw small-molecule 1D `fid`s, wet-lab certified `qNMR` reference mixtures (`NIST`/`USP` CRMs), or experimental pseudo-2D `DOSY` `ser` datasets have been evaluated end-to-end yet**; external experimental repository validation is tracked openly under follow-ups **`F14`** and **`F16`**.

---

## Multi-Nuclear 1D Referencing & Global Voigt Deconvolution

### Built-In Solvent Residual & Trace Impurity Lookup (`Fulmer 2010`)

`nmr-core::nmr1d` embeds the complete peer-reviewed ${}^{1}\text{H}$ and ${}^{13}\text{C}$ residual solvent multiplet ($2J_{\text{HD}}$ / $1J_{\text{CD}}$) and trace laboratory impurity table (Gottlieb et al. 1997; Fulmer et al. *Organometallics* 2010) across **10 deuterated solvents**:
- `CDCl3` (`7.26 ppm` ${}^{1}\text{H}$, `77.16 ppm` ${}^{13}\text{C}$ triplet), `DMSO-d6` (`2.50` / `39.52 ppm` septet), `D2O` (`4.79 ppm`), `CD3OD` (`3.31` / `49.00 ppm`), `CD3CN` (`1.94` / `1.32, 118.26 ppm`), `acetone-d6` (`2.05` / `29.84, 206.26 ppm`), `THF-d8`, `C6D6`, `CD2Cl2`, and `toluene-d8`.
- Automatically flags common residual trace solvents and laboratory contaminants (`H2O`/`HDO`, `TMS`, `DSS`, hydrocarbon grease, silicone grease, `EtOAc`, `Et2O`, `CH2Cl2`, acetone, `MeOH`, `EtOH`, `THF`, toluene, 1,4-dioxane, `DMF`) via `match_solvent_or_impurity_peaks_1d` so impurity singlets are never mistaken for analyte protons.

### Global Coupled-Cluster Pseudo-Voigt Deconvolution (`deconvolve_1d`)

Simple local-extremum peak picking fails when scalar-coupled multiplets overlap within `< 1.0 FWHM` or exhibit Lorentzian-Gaussian (`Voigt`) shimming tails. After 1D autophasing and Whittaker baseline flattening ([Chapter 2](02-custom-pulprog-and-dsp.md#analytical-1d-autophasing-whittaker-baseline-and-hsvd)), `deconvolve_1d` combines Morse topological prominence (`pick_peaks_1d`) and Ricker second-derivative shoulder seeding with analytical-Jacobian projected Levenberg–Marquardt optimization over coupled pseudo-Voigt lineshapes (Thompson, Cox & Hastings 1987):

$$V(\nu; \nu_0, A, \gamma, \eta) = A \left[\eta \frac{\gamma / \pi}{(\nu - \nu_0)^2 + \gamma^2} + (1 - \eta) \sqrt{\frac{\ln 2}{\pi}}\frac{1}{\gamma}\exp\!\left(-\ln 2 \frac{(\nu - \nu_0)^2}{\gamma^2}\right)\right], \qquad \eta \in [0, 1]$$

- Supports **`1H`**, **`13C{1H}`**, **`DEPT-90`**, signed **`DEPT-135`** ($\text{CH}/\text{CH}_3 > 0$, $\text{CH}_2 < 0$), **`19F`**, and **`31P`** spectra.
- Separates coalesced shoulders down to `0.65–0.72 × FWHM` with `< 2%` component area error and `< 0.15 Hz` centroid error.

---

## Region Integration & Recursive Hoye Multiplet `J`-Trees

### Quantitative Region Integration (`integrate_spectrum1d_regions`)

Each integral region supports three baseline modes (`None`, `Explicit { bias, slope }`, and automatic edge ordinary-least-squares `AutoLinearEdges`), optional deconvolved Voigt component area summation, interactive renormalization (`ReferenceRegion` to `1.00 H` / `2.00 H` / `3.00 H`, `TotalProtons`, or `ExplicitScale`), and exact quadrature + edge-OLS baseline noise variance propagation ($\sigma_I$) from spectral noise RMS.

### Recursive Multiplet Decomposition & Second-Order Roofing (`Hoye 1994, 2002`)

`nmr-core::multiplet` decomposes picked or Voigt-deconvolved multiplet clusters into scalar coupling constants $J_1 \ge J_2 \ge \dots$ (`Hz`, computed field-independently via $|\Delta\delta| \times \nu_0$) using recursive sub-multiplet stripping (Hoye, Hanson & Vyvyan *J. Org. Chem.* 1994; Hoye & Zhao *J. Org. Chem.* 2002):

1. **16 Canonical Multiplet Classes**: Classifies `s`, `br s`, `d`, `t`, `q`, `quint`, `sext`, `sept`, `dd`, `ddd`, `dt`, `td`, `dq`, `qd`, `tt`, and `m` (complex/higher-order fallback).
2. **Second-Order Roofing ("Leaning") Correction**: When coupled nuclei share nearby chemical shifts ($\Delta\nu / J \sim 2.5\text{–}8.0$), inner lines intensify by $1 + \sin 2\theta$ while outer lines weaken by $1 - \sin 2\theta$ ($\tan 2\theta = J / \Delta\nu$; Pople, Schneider & Bernstein 1959). `analyze_multiplet_peaks` optimizes roofing angle $\theta$, Rayleigh linewidth coalescence, and multi-Lorentzian cross-talk simultaneously—recovering $|J|$ within **`< 0.20 Hz`** even on strongly roofed `AB`, `ABX`, `A2X`, and `A3X` spin systems ($\Delta\nu / J = 2.5..3.2$).
3. **Closed-Form `AB` Quartet Solver & Exact $2^N$ Quantum Hamiltonian Validator**: Includes `solve_ab_quartet` ($\Delta\nu_{AB} = \sqrt{(\nu_1 - \nu_4)(\nu_2 - \nu_3)}$, $|J_{AB}| = |\nu_1 - \nu_2|$) and an exact $2^N \times 2^N$ secular spin-$1/2$ density-matrix / Hamiltonian eigensolver (`simulate_exact_spin_half_multiplet` for `1..=8` spins) to verify second-order line positions and intensities.

---

## Publication Journal Export & `Octet` 2D Structure Linking

### Lossless `ACS`, `JOC`, `RSC` & `Nature` Formatter & Parser

`format_journal_nmr_report` and `parse_journal_nmr_report` generate—and losslessly round-trip parse—ready-to-publish experimental paragraphs across **4 journal house styles** (`ACS`, `JOC`, `RSC`, `Nature`) and **3 text encodings** (`Unicode`, `Ascii`, `Latex`), including decoupled `{1H}` headers and negative `19F`/`31P` shifts:

```text
¹H NMR (600 MHz, CDCl₃) δ 7.98 (dd, J = 7.9, 1.7 Hz, 1H), 7.56 (ddd, J = 8.1, 7.4, 1.7 Hz, 1H), 7.31 (td, J = 7.6, 1.2 Hz, 1H), 7.12 (dd, J = 8.1, 1.2 Hz, 1H), 2.35 (s, 3H).
```

### Bidirectional `Octet` 2D Structure ↔ Multiplet Assignment

Multiplet regions link directly to 0-based `Octet` atom indices, 1-based `MOL`/`SDF` V2000 atom numbers, and `SMILES` `[atom:map]` tags (`MultipletAssignmentTable`, `auto_assign_1h_multiplets`):
- Hovering a multiplet in `nmr-gui` illuminates matching symmetry-equivalent atoms with color-coded `RenderScene` halos on the 2D `Octet` chemical drawing (and vice versa).
- Round-trips cleanly via `SDF` `<NMR_ASSIGNMENT>` property tags and structured `TSV`.

---

## NIST-Traceable Internal-Standard `qNMR` Purity

Quantitative ${}^{1}\text{H}$ NMR (`qNMR`) determines absolute analyte mass-fraction purity $P_x$ without analyte-specific calibration curves because integrated ${}^{1}\text{H}$ peak area is strictly proportional to molar spin count under full longitudinal relaxation (Malz & Jancke 2005; Pauli et al. *J. Med. Chem.* 2014):

$$P_x = \frac{I_x}{I_{\text{std}}} \cdot \frac{N_{\text{std}}}{N_x} \cdot \frac{\text{MW}_x}{\text{MW}_{\text{std}}} \cdot \frac{m_{\text{std}}}{m_x} \cdot P_{\text{std}}$$

`nmr-core::multiplet` (`calculate_qnmr_purity`, `calculate_qnmr_replicates`) provides:
- **Built-in NIST/USP Certified Reference Standards (`qnmr_certified_standards`)**: Dimethyl sulfone (`DMSO2`), Maleic acid (`MA`), `1,3,5`-Trimethoxybenzene (`TMB`), Dimethyl terephthalate (`DMT`), Benzoic acid (`BA`), and `3,5`-Dinitrobenzoic acid (`DNBA`).
- **ISO GUM Uncertainty Propagation**: Propagates integral noise ($\sigma_{I_x}, \sigma_{I_{\text{std}}}$), microbalance weighing uncertainties ($\sigma_{m_x}, \sigma_{m_{\text{std}}}$), molecular-weight uncertainties, and certified reference purity uncertainty ($\sigma_{P_{\text{std}}}$) into combined standard uncertainty $u_c(P_x)$, expanded uncertainty $U_{k=2} = 2 u_c(P_x)$ (`95%` coverage), molar concentration (`mol/L` & `mM`), and multi-tube replicate `%RSD`.
- **$5 T_1$ Longitudinal Relaxation Guard**: Audits acquisition recycle delay $d_1 + t_{\text{aq}}$ against $5 \max(T_{1,x}, T_{1,\text{std}})$ and computes the analytical Ernst steady-state saturation bias (Ernst & Anderson 1966), emitting an explicit quantitative warning whenever incomplete relaxation could bias $I_x / I_{\text{std}}$.

---

## 4-Mode Multi-Spectrum 1D Comparison & Kinetics Studio

The interactive **`📚 1D Stack & Kinetics`** panel (`nmr-gui::stack1d_panel`, native desktop & `wasm32`) compares $N$ 1D spectra (`1H`/`13C`/`DEPT-90`/`DEPT-135` editing suites, reaction-kinetics time courses, pH/temperature arrays, ligand `STD`/`WaterLOGSY` screens, batch QC, or experimental vs `Octet`-predicted 1D) at `>= 60 fps` (`32 × 64k` points via min/max level-of-detail decimation):

1. **Superimposed Overlay**: Direct pixel-aligned comparison with per-trace color, vertical scale, and opacity controls.
2. **Tilted / Vertical Stack**: Publication staggered stacks with continuous vertical offset and horizontal oblique shear angle.
3. **Scaled Difference / Residual Trace**: Live subtraction $D_i(\delta) = S_i(\delta) - \alpha S_{\text{ref}}(\delta)$ highlighting subtle impurity appearance, `STD` saturation differences, or reaction intermediates.
4. **Pseudo-2D Waterfall / Heatmap**: Color-mapped $(\delta \times \text{trace index / time})$ intensity matrix for large kinetic or titration arrays.

Includes **interval FFT cross-correlation peak alignment** (`align_traces_interval_fft`; Savorani, Tomasi & Engelsen 2010) to remove small pH/temperature peak drift prior to difference or arrayed integral kinetics export (`TSV`, `SVG`, `PNG`).

---

## Pseudo-2D `DOSY`: Stejskal–Tanner vs Tikhonov `ILT` Caveats

Diffusion-Ordered Spectroscopy (`DOSY`) separates mixture components non-destructively by translational self-diffusion coefficient $D$ ($\text{m}^2\,\text{s}^{-1}$). `nmr-core::dosy`, `nmr-gpu::dosy_gpu`, and `nmr-gui::dosy_panel` support raw Bruker (`difflist` + `acqus` `D20`, `P30`, `D16`, `GPNAM`) and Varian (`procpar` `gzlvl1`, `gt1`, `del`, `tau`, `gcal`) diffusion datasets.

### Exact Gradient-Shape & Sequence Attenuation Physics (`b`-Factor)

Signal intensity decays across gradient strengths $g$ ($\text{T m}^{-1}$) as $I(b) = \sum_{k=1}^K I_{0,k} \exp(-b D_k)$ with exact SI diffusion weighting $b$ ($\text{s m}^{-2}$; Stejskal & Tanner 1965; Wu, Chen & Johnson 1995; Pelta et al. 2002):

$$b(g) = \bigl(\gamma\, \sigma_{\text{shape}}\, g\, \delta\bigr)^2 \, \Delta_{\text{eff}}$$

- **Gradient envelope factors ($\sigma_{\text{shape}}$)**: `Rectangular` (`1.0`), half-sine `SineBell` (`SINE.100`, $\sigma_{\text{sine}} = 2/\pi$), and `SmoothedSquare`.
- **Sequence effective diffusion delays ($\Delta_{\text{eff}}$)**:
  - Monopolar stimulated echo (`STE`): $\Delta_{\text{eff}} = \Delta - \frac{\delta}{3}$.
  - Bipolar gradient pulse pair LED (`BPP-LED`, e.g., Bruker `dstebpgp3s` / `ledbpgp2s`, where total bipolar encoding width $\delta = 2 P_{30}$): $\Delta_{\text{eff}} = \Delta - \frac{\delta}{3} - \frac{\tau}{2}$.

### Default Quantitative Fit: Discrete `BIC` VarPro + LM (`K = 1, 2, 3`)

By default (`nmrassign dosy fit --criterion bic`), `nmr-assign` fits discrete $K \in \{1, 2, 3\}$ multi-exponential decays via **Variable Projection + Kaufman projected Gauss–Newton + Levenberg–Marquardt** on CPU (`f64`) and massively parallel WebGPU compute shaders (`dosy.wgsl`, `< 0.5%` relative $D$ difference and exact `BIC` model-order parity vs CPU, chunked at `16,384` columns):

$$\operatorname{BIC}(K) = M \ln\!\left(\frac{\text{RSS}_K}{M}\right) + 2K \ln M$$

augmented by a Davies rate-grid look-elsewhere penalty, a $4\sigma$ component $\ell_2$-SNR resolution guard, and residual/parametric bootstrap `95%` confidence intervals $[D_{k,2.5\%}, D_{k,97.5\%}]$ (resolving overlapped `K = 2` components down to $D_1 / D_2 = 1.30$ at per-component `SNR 56.3–68.8` mixed [`SNR = 40` pure], and recovering noiseless 32-point `K = 3` tri-exponential components to `9.73e-10%`).

### Opt-In Continuous Tikhonov `ILT` (`CONTIN`-Style) & Conditioning Caveats

For polydisperse polymers or continuous size distributions, `solve_tikhonov_ilt` (`nmrassign dosy ilt --strategy gcv|lcurve|discrepancy|fixed`, Provencher 1982; Lawson & Hanson 1974 `NNLS`) solves the second-derivative Tikhonov-regularized non-negative inverse Laplace problem over a log-spaced diffusivity grid $\{D_q\}_{q=1}^Q$:

$$\min_{\mathbf{f} \ge 0} \; \|\mathbf{K}\mathbf{f} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{L}_2 \mathbf{f}\|_2^2$$

selecting $\lambda > 0$ over a 24-point logarithmic $\lambda$ sweep via Generalized Cross-Validation (`gcv`, default), Menger-curvature `lcurve` corner detection (which frequently selects the same optimal $\lambda$ grid index as `gcv` on smooth bimodal decays), baseline-$\sigma$ Morozov `discrepancy`, or an explicit `--fixed-lambda` value.

> [!CAUTION] Inverse Laplace Transform (`ILT`) Conditioning Caveat
> Real-valued continuous Laplace inversion is **exponentially ill-conditioned** (singular values of $K_{m,q} = e^{-b_m D_q}$ decay super-exponentially):
> 1. **Compromise peaks under overlap**: Two unresolved overlapping multiplets with diffusivities $D_1$ and $D_2$ ($D_1 / D_2 \lesssim 2.0$) at moderate SNR can coalesce into a single false `ILT` peak at the geometric mean $\sqrt{D_1 D_2}$.
> 2. **Regularization-dependent peak splitting**: Under-regularized $\lambda$ (or identity-matrix $\|\mathbf{f}\|_2^2$ penalties instead of second-difference $\|\mathbf{L}_2\mathbf{f}\|_2^2$) splits broad distributions into artificial narrow spikes, whereas over-regularized $\lambda$ merges distinct species.
>
> **Always use the default `BIC`-selected discrete Stejskal–Tanner fit (`K = 1..3`) with bootstrap 95% CIs for quantitative small-molecule mixture assignment**, and treat opt-in Tikhonov `ILT` contours (`ILT_ARTIFACT_CAVEAT` banner in `dosy_panel`) as qualitative polydispersity visualizations verified across a $\lambda$ sensitivity sweep.

### Stokes–Einstein Hydrodynamic Radius $R_h$ & Molecular Weight

From fitted $D$, `nmr-core::dosy` computes temperature-compensated Stokes–Einstein hydrodynamic radii and empirical power-law molecular weights (Einstein 1905; Evans et al. *Angew. Chem. Int. Ed.* 2013) using built-in dynamic viscosities $\eta(T)$ across `D2O`, `H2O`, `CDCl3`, `DMSO-d6`, `CD3OD`, and `CD3CN`:

$$R_h = \frac{k_B T}{6\pi \eta(T) D}, \qquad \widehat{\text{MW}} = \left(\frac{\alpha_{\text{solv}}}{D}\right)^{1/\beta_{\text{solv}}}$$

### Headless CLI Quick Reference (`nmr1d` & `dosy`)

```bash
# 1. Process raw 1D FID (or .jdx / Bruker 1r), autophase, Whittaker baseline, HSVD & Voigt pick
nmrassign nmr1d process --in ./data/compound_1h/1 --solvent CDCl3 --hsvd --deconvolve --out ./out/spec1d.jdx

# 2. Extract Hoye J-coupling multiplet trees & ACS/JOC/RSC/Nature experimental paragraph
nmrassign nmr1d multiplet --in ./out/spec1d.jdx --solvent CDCl3 --style acs --out ./out/multiplets.txt

# 3. Compute ISO GUM internal-standard qNMR purity (% w/w, U_k=2, and 5*T1 saturation audit)
nmrassign nmr1d qnmr --in ./out/spec1d.jdx \
  --std-id DMSO2 --std-mass-mg 5.12 --std-purity 0.9994 \
  --analyte-mass-mg 12.40 --analyte-mw 180.16 --analyte-protons 3 \
  --analyte-ppm 2.28:2.42 --out ./out/qnmr_report.json

# 4. Discrete BIC Stejskal-Tanner / BPP-LED DOSY fit, opt-in Tikhonov ILT & Stokes-Einstein MW
nmrassign dosy fit --in ./data/mixture_dosy/10 --criterion bic --solvent CDCl3 --temp-k 298.15 --engine auto --out ./out/dosy_bic.tsv
nmrassign dosy ilt --in ./data/mixture_dosy/10 --strategy gcv --out ./out/dosy_ilt.tsv
nmrassign dosy mw --d 7.8e-10 --solvent CDCl3 --temp-k 298.15
```
