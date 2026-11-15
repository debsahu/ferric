# 17. Analytical 1D NMR Processing, Multiplet / qNMR Analysis, Multi-Spectrum Comparison, and Pseudo-2D DOSY Studio

> [!IMPORTANT]
> **Validation Scope (`Phase 14` Synthetic Forward-Model Self-Consistency)**: Every quantitative `T14.1`–`T14.3` verification gate in §§1–3 (and `T14.4` in §4) is a **synthetic forward-model self-consistency test** rather than an empirical benchmark on deposited raw spectrometer acquisitions or physical certified reference materials. **`0` real deposited raw 1D small-molecule `fid`s or certified `qNMR` reference-material datasets (`F14`)** and **`0` real deposited raw Bruker `difflist`/`ser` `DOSY` datasets (`F16`)** have been evaluated yet; `v0.10.0` `CHANGELOG.md` and all Phase 14 review summaries carry this explicit qualifier. See [Validation Scope (Synthetic Forward-Model Self-Consistency)](#validation-scope-synthetic-forward-model-self-consistency) below.

## Provenance & Clean-Room Statement
Original clean-room specification and mathematical design for Phase 14 (`v0.10.0`: `T14.1`–`T14.4`). Zero source code, macros, parameter libraries, binaries, or output templates from any proprietary, academic-licensed, or copyleft (GPL/AGPL) NMR processing, chemometrics, or assignment package were consulted. Every mathematical model, chemical-shift reference value, and numerical solver is derived strictly from peer-reviewed publications and public spectrometer format documentation:

1. **1D Digital Signal Processing, Apodization, Phase & Baseline Correction (`T14.1`)**:
   - Bartholdi, E. & Ernst, R. R. (*J. Magn. Reson.* **1973**, 11, 9–19) — causality, zero-filling information preservation, and discrete Hilbert reconstruction of imaginary dispersion spectra.
   - Ferrige, A. G. & Lindon, J. C. (*J. Magn. Reson.* **1978**, 31, 337–340) — Lorentz-to-Gauss resolution-enhancement multiplication.
   - Doddrell, D. M., Pegg, D. T. & Bendall, M. R. (*J. Magn. Reson.* **1982**, 48, 323–327) — Distortionless Enhancement by Polarization Transfer (`DEPT-90` / `DEPT-135`) $\text{CH}_n$ editing physics.
   - Otting, G., Widmer, H., Wagner, G. & Wüthrich, K. (*J. Magn. Reson.* **1986**, 66, 187–193) — first-point $0.5$ scaling to eliminate discrete Fourier transform baseline DC offset.
   - Hoch, J. C. & Stern, A. S. (*NMR Data Processing*, Wiley-Liss, **1996**) — apodization windows, Nyquist frequency ordering, and Kramers–Kronig dispersion relations.
   - Chen, L., Weng, Z., Goh, L. & Garland, M. (*J. Magn. Reson.* **2002**, 158, 164–168); de Brouwer, H. (*J. Magn. Reson.* **2009**, 201, 230–238) — automatic `PHC0`/`PHC1` phase correction via first-derivative Shannon entropy minimization with dispersion / negative-lobe penalties.
   - Eilers, P. H. C. (*Anal. Chem.* **2003**, 75, 3631–3636); Cobas, J. C., Bernstein, M. A., Martín-Pastor, M. & Tahoces, P. G. (*J. Magn. Reson.* **2006**, 183, 145–151) — signal-free continuous-wavelet baseline classification, Whittaker penalized least-squares smoother, and Bernstein polynomial baseline fitting.
2. **Solvent Suppression, Reference Tables & Pseudo-Voigt Deconvolution (`T14.1`)**:
   - Barkhuijsen, H., de Beer, R. & van Ormondt, D. (*J. Magn. Reson.* **1987**, 73, 553–557); Pijnappel, W. W. F., van den Boogaart, A., de Beer, R. & van Ormondt, D. (*J. Magn. Reson.* **1992**, 97, 122–134) — Hankel Singular Value Decomposition (`HSVD` / Hankel–Lanczos) state-space parameterization and time-domain solvent resonance subtraction.
   - Gottlieb, H. E., Kotlyar, V. & Nudelman, A. (*J. Org. Chem.* **1997**, 62, 7512–7515); Fulmer, G. R., Miller, A. J. M., Sherden, N. H., Gottlieb, H. E., Nudelman, A., Stoltz, B. M., Bercaw, J. E. & Goldberg, K. I. (*Organometallics* **2010**, 29, 2176–2179) — deuterated solvent residual peaks and trace laboratory impurity chemical shifts for ${}^{1}\text{H}$ and ${}^{13}\text{C}$ NMR.
   - Levenberg, K. (*Quart. Appl. Math.* **1944**, 2, 164–168); Marquardt, D. W. (*J. Soc. Ind. Appl. Math.* **1963**, 11, 431–441); Wertheim, G. K. et al. (*Rev. Sci. Instrum.* **1974**, 45, 1369–1371); Thompson, P., Cox, D. E. & Hastings, J. B. (*J. Appl. Crystallogr.* **1987**, 20, 79–83) — damped nonlinear least-squares global pseudo-Voigt lineshape deconvolution.
3. **Recursive Multiplet Coupling Trees, `Octet` Linking & Quantitative NMR (`T14.2`)**:
   - Hoye, T. R., Hanson, P. R. & Vyvyan, J. R. (*J. Org. Chem.* **1994**, 59, 4096–4103); Hoye, T. R. & Zhao, H. (*J. Org. Chem.* **2002**, 67, 4014–4016); Pople, J. A., Schneider, W. G. & Bernstein, H. J. (*High-Resolution Nuclear Magnetic Resonance*, McGraw-Hill, **1959**, Ch. 6) — recursive sub-doublet peeling $J$-coupling tree extraction, coalesced multiplet topologies, and second-order $\Delta\nu / J$ roofing ("leaning") intensity & Zeeman-shift analysis.
   - Pauli, G. F., Chen, S.-N., Simmler, C., Lankin, D. C., Gödecke, T., Jaki, B. U., Friesen, J. B., McAlpine, J. B. & Napolitano, J. G. (*J. Med. Chem.* **2014**, 57, 9220–9231); Malz, F. & Jancke, H. (*J. Pharm. Biomed. Anal.* **2005**, 38, 813–823) — internal-standard quantitative ${}^{1}\text{H}$ NMR (`qNMR`) mass-fraction purity & molarity determination, longitudinal relaxation ($5 T_1$) constraints, replicate `RSD`, and GUM uncertainty propagation.
4. **Multi-Spectrum 1D Alignment & Comparison (`T14.3`)**:
   - Savorani, F., Tomasi, G. & Engelsen, S. B. (*J. Magn. Reson.* **2010**, 202, 190–202) — interval correlation-shifting fast Fourier cross-correlation peak alignment across arrayed 1D NMR series.
5. **Diffusion-Ordered Spectroscopy (`DOSY`) Physics & Inversion (`T14.4`)**:
   - Stejskal, E. O. & Tanner, J. E. (*J. Chem. Phys.* **1965**, 42, 288–292); Wu, D. H., Chen, A. D. & Johnson, C. S. (*J. Magn. Reson. A* **1995**, 115, 260–264); Pelta, M. D., Morris, G. A., Stchedroff, M. J. & Hammond, S. J. (*Magn. Reson. Chem.* **2002**, 40, 147–152) — pulsed field gradient spin-echo (`STE`) and bipolar gradient pulse pair longitudinal eddy-current delay (`BPP-LED`) diffusion attenuation equations and gradient shape factors.
   - Lawson, C. L. & Hanson, R. J. (*Solving Least Squares Problems*, Prentice-Hall, **1974**); Provencher, S. W. (*Comput. Phys. Commun.* **1982**, 27, 213–227) — non-negative Tikhonov-regularized Inverse Laplace Transform (`ILT`) over log-spaced diffusion grids.
   - Evans, R., Deng, Z., Rogerson, A. K., McLachlan, A. S., Richards, J. J., Nilsson, M. & Morris, G. A. (*Angew. Chem. Int. Ed.* **2013**, 52, 3199–3202) — Stokes–Einstein hydrodynamic radius $R_h$ and empirical power-law molecular-weight estimation from translational diffusion coefficients.

---

## Validation Scope (Synthetic Forward-Model Self-Consistency)

1. **Synthetic Forward-Model Self-Consistency Only (`T14.1`–`T14.4`)**: Every quantitative verification gate across `T14.1` (`nmr-core::nmr1d`), `T14.2` (`nmr-core::multiplet`), `T14.3` (`nmr-gui::stack1d_panel`), `T14.4` (`nmr-core::dosy`, `nmr-gpu::dosy_gpu`, `nmr-gui::dosy_panel`), `nmr1d_cli_tests`, and `dosy_metabolomics_cli_tests` evaluates synthetic data generated from analytical forward models:
   - **1D DSP & Deconvolution (`T14.1`)**: synthesized multi-component complex damped-sinusoid / Voigt 1D FIDs (`synthesize_fid_1d`), planted zero-/first-order phase rotations (`PHC0`, `PHC1`), synthetic polynomial/sinusoidal baseline roll, planted damped-exponential `H2O`/`HDO` solvent poles, planted systematic reference offsets against Fulmer (2010) residual solvent shifts, and synthetic overlapping pseudo-Voigt multiplets.
   - **Integration, Multiplet Trees & `qNMR` (`T14.2`)**: synthetic Lorentzian/pseudo-Voigt traces with stationary Gaussian noise and planted linear baseline ramps (`1,500`-trial Monte Carlo variance check), 28 noisy synthetic first-order & coalesced multiplet patterns (`s` through `ddd`/`dt`/`td`/`tt`/`m`), synthetic roofed `AB` / `ABX` / `AMX` spin systems (`simulate_exact_spin_half_multiplet` diagonalizing exact spin-$1/2$ Zeeman + scalar-coupling Hamiltonians), synthetic gravimetric `qNMR` mixtures with analytical stoichiometry and planted `T1` relaxation parameters, and synthetic `Octet` `SMILES`/`.mol` spectra.
   - **Multi-Spectrum 1D Stack, Alignment & Kinetics (`T14.3`)**: synthetic 8-step first-order $A \to B$ reaction kinetics stacks, synthetic on-/off-resonance STD difference pairs with planted matrix scaling $\alpha$, and synthetic multi-interval pH/temperature-drifted 1D traces.
   - **Pseudo-2D `DOSY` Inversion & Hydrodynamics (`T14.4`)**: synthetic Stejskal–Tanner / Wu–Johnson multi-exponential attenuation curves ($K \in \{1, 2, 3\}$) with stationary Gaussian noise, planted 2- and 3-component diffusion mixtures (`Lysozyme` + `Sucrose` + `HDO`), synthesized Bruker `difflist`/`acqus` (`dstebpgp3s`, `ledbpgp2s`, `SINE.100`) and Varian `procpar` (`gzlvl1`, `gcal_`, `gt1`, `del`) headers, synthetic bimodal log-$D$ `CONTIN`-style Tikhonov `ILT` test profiles, and analytical Stokes–Einstein hydrodynamic radii / Evans (2013) empirical power-law molecular-weight curves.
   These gates rigorously verify mathematical self-consistency, CPU/WebGPU numerical inversion parity, exact uncertainty propagation, lossless parser/formatter round-trips, and headless GUI rendering, **not** empirical performance on uncurated spectrometer hardware artifacts (radiation damping, shim asymmetry, `13C` satellites, thermal convection, eddy-current phase roll, pulsed-field-gradient spatial non-uniformity, or weighing/volumetric laboratory variance).
2. **Zero Real Deposited Small-Molecule Raw `fid`s or Certified `qNMR` Reference Materials Evaluated Yet (`F14`)**: **`0` real deposited raw 1D small-molecule `fid`s** (across `${}^{1}\text{H}$, `${}^{13}\text{C}\{{}^{1}\text{H}\}$, `DEPT-90`/`DEPT-135`, `${}^{19}\text{F}$, `${}^{31}\text{P}$) and **`0` physical certified `qNMR` reference-material datasets** (e.g., NIST/USP primary calibrant experimental acquisitions) have been evaluated in Phase 14. Pre-registered single-shot evaluation without parameter tuning on real deposited small-molecule raw 1D `fid`s and at least one experimental `qNMR` reference dataset is tracked as open post-`v0.10.0` follow-up `- [ ] (A) F14` in `TODO.md` and is required before claiming empirical small-molecule 1D processing or `qNMR` assay accuracy.
3. **Zero Real Deposited Experimental Bruker `difflist`/`ser` or Varian `DOSY` Datasets Evaluated Yet (`F16`)**: **`0` real deposited raw Bruker `difflist`/`ser` (`dstebpgp3s` / `ledbpgp2s`) or Varian `fid`/`procpar` experimental pseudo-2D `DOSY` acquisitions** have been evaluated in Phase 14. Pre-registered single-shot evaluation without parameter tuning on deposited multi-component experimental `DOSY` acquisitions (auditing thermal convection, gradient non-linearity, baseline offset, and unresolved multiplet overlap) is tracked as open post-`v0.10.0` follow-up `- [ ] (A) F16` in `TODO.md` and is required before claiming empirical experimental `DOSY` accuracy.
4. **Mandatory Release Candor (`v0.10.0` `CHANGELOG.md`)**: Both `TODO.md` (`T14.1`–`T14.5`) and the `v0.10.0` `CHANGELOG.md` release notes carry this explicit **`synthetic forward-model self-consistency`** qualifier alongside open follow-ups `F14`, `F16`, and `F17` (`F15` adversarial 1D ingest hardening completed in `14.W2`).

---

## 1. 1D Small-Molecule & Biomolecular DSP Core (`crates/nmr-core/src/nmr1d.rs`, `T14.1`)

### 1.1 Multi-Nuclear Representation, Apodization, Zero-Filling, FFT & Referencing
The 1D DSP module supports small-molecule, natural-product, organometallic, fluorinated, organophosphorus, and biomolecular 1D experiments through explicit nuclear and spin-editing metadata:
- **Nuclei (`Nucleus1D`)**: ${}^{1}\text{H}$ ($\Xi = 1.00000000$), ${}^{13}\text{C}$ ($\Xi = 0.25145020$), ${}^{19}\text{F}$ ($\Xi = 0.94094011$), ${}^{31}\text{P}$ ($\Xi = 0.40480742$), ${}^{15}\text{N}$ ($\Xi = 0.10136767$), and arbitrary user-specified nuclei.
- **Experiment kinds (`Experiment1DKind`)**:
  - `Proton1D` (${}^{1}\text{H}$), `Carbon13Decoupled` (${}^{13}\text{C}\{{}^{1}\text{H}\}$), `Fluorine19` (${}^{19}\text{F}$), `Phosphorus31` (${}^{31}\text{P}$): expected absorption polarity `Polarity1D::AllPositive`.
  - `Dept90`: ${}^{13}\text{C}$ polarization transfer at read flip angle $\theta = 90^\circ$ ($\text{CH} > 0$, $\text{CH}_2 \approx 0$, $\text{CH}_3 \approx 0$, $\text{C}_{\text{quat}} = 0$; Doddrell et al. 1982), `Polarity1D::AllPositive`.
  - `Dept135`: ${}^{13}\text{C}$ polarization transfer at $\theta = 135^\circ$ ($\text{CH}, \text{CH}_3 > 0$, $\text{CH}_2 < 0$, $\text{C}_{\text{quat}} = 0$), `Polarity1D::Biphasic`.
  - Theoretical DEPT transfer factor for $\text{CH}_n$ ($n \in \{0, 1, 2, 3\}$) at read pulse angle $\theta$:
    $$I_{\text{DEPT}}(n, \theta) = \begin{cases} 0, & n = 0, \\ \sin\theta, & n = 1, \\ \sin(2\theta), & n = 2, \\ \frac{3}{4}(\sin\theta + \sin 3\theta), & n = 3. \end{cases}$$

Given a complex time-domain FID $z[n] \in \mathbb{C}$ ($n = 0, \dots, N_{\text{td}} - 1$) with dwell time $\Delta t = 1 / \text{SW}_{\text{Hz}}$, observation frequency $\nu_{\text{obs}}\text{ (MHz)}$, and carrier chemical shift $\delta_{\text{carrier}}\text{ (ppm)}$ (with digital oversampling group delay `group_delay_pts` compensated prior to `HSVD` or apodization):

1. **Apodization (`Apodization1D`)** at $t_n = n\,\Delta t$ ($T_{\text{aq}} = N_{\text{td}}\,\Delta t$):
   - *Exponential line broadening* ($\text{LB}$ in Hz):
     $$w_{\text{exp}}[n] = \exp(-\pi\,\text{LB}\,t_n).$$
   - *Gaussian broadening* ($\text{GB}$ FWHM in Hz):
     $$w_{\text{gauss}}[n] = \exp\!\left(-\left(\frac{\pi\,\text{GB}\,t_n}{2\sqrt{\ln 2}}\right)^2\right).$$
   - *Lorentz-to-Gauss resolution enhancement* (Ferrige & Lindon 1978; Lorentzian narrowing $w_L = |\text{LB}_{\text{hz}}|$ Hz, Gaussian target width $w_G = \text{GB}_{\text{hz}}$ Hz, optional echo maximum fraction $c \in [0, 1)$):
     - When $c = 0$: $w_{\text{LG}}[n] = \exp\!\left(+\pi w_L t_n - \left(\frac{\pi w_G t_n}{2\sqrt{\ln 2}}\right)^2\right)$, normalized by $\max_m w_{\text{LG}}[m]$ if $> 1$ to prevent overflow on over-sharpened parameters.
     - When $c > 0$ ($t_{\max} = c\,T_{\text{aq}}$): $w_{\text{LG}}[n] = \exp\!\left(a t_n - b t_n^2\right) / \exp(a^2 / (4b))$ with $a = \pi w_L$ and $b = \max\!\left(\frac{a}{2 t_{\max}}, \left(\frac{\pi w_G}{2\sqrt{\ln 2}}\right)^2\right)$.
   - *Cosine-bell / shifted sine-bell* ($\alpha \in [0, 1)$, power $p > 0$; pure cosine-bell has $\alpha = 0.5$):
     $$w_{\text{cos}}[n] = \cos^p\!\left(\frac{\pi n}{2 N_{\text{td}}}\right), \qquad w_{\text{sin}}[n] = \sin^p\!\left(\pi\left(\alpha + (1 - \alpha)\frac{n}{N_{\text{td}}}\right)\right).$$
   - First-point baseline scaling (Otting et al. 1986): $z_{\text{win}}[0] \leftarrow f_0 w[0] z[0]$ ($f_0 = 0.5$ by default when linear phase at $t = 0$ is zero).

2. **Zero-Filling & Calibrated Forward FFT**:
   - Appending zeros to length $N \ge 2 N_{\text{td}}$ (typically power-of-two $N = 2^q$) preserves full complex causality so the real and imaginary parts obey discrete Kramers–Kronig / Hilbert relations (Bartholdi & Ernst 1973).
   - With standard forward DFT $Z[m] = \sum_{n=0}^{N-1} \tilde{z}[n] e^{-i 2\pi m n / N}$ (conjugating $\tilde{z}[n] = z_{\text{win}}^*[n]$ when `conjugate_for_fft == true`), bins are mapped onto the standard descending-ppm NMR axis $j \in \{0, \dots, N-1\}$ via
     $$m_j = (N / 2 + N - j) \bmod N, \qquad S[j] = (2 \Delta t)\, Z[m_j],$$
     $$\delta_j = \delta_{\text{first}} + j \cdot \Delta\delta, \qquad \delta_{\text{first}} = \delta_{\text{carrier}} + \frac{\text{SW}_{\text{Hz}}}{2 \nu_{\text{obs}}}, \qquad \Delta\delta = -\frac{\text{SW}_{\text{Hz}}}{N \nu_{\text{obs}}}.$$
   - Under the $(2 \Delta t)$ normalization with $f_0 = 0.5$, a unit-amplitude FID resonance $z[n] = A e^{(-\pi W + i 2\pi \Delta\nu) n \Delta t}$ transforms to a continuous-area-preserving Lorentzian whose integral in Hz ($\sum_j \text{Re}\,S[j]\,|\Delta\nu_{\text{bin}}|$) equals $A$ and whose integral in ppm ($\sum_j \text{Re}\,S[j]\,|\Delta\delta|$) equals $A / \nu_{\text{obs}}$.

3. **Chemical Shift Referencing**:
   - Direct offset shift: $\delta_{\text{first}} \leftarrow \delta_{\text{first}} + (\delta_{\text{target}} - \delta_{\text{obs}})$.
   - Automatic solvent referencing (`auto_calibrate_solvent_1d`) locates the parabolic sub-bin centroid of the strongest peak inside $[\delta_{\text{Fulmer}} - \Delta\delta_{\text{win}}, \delta_{\text{Fulmer}} + \Delta\delta_{\text{win}}]$ and translates `axis.first_ppm` so the residual solvent resonance aligns with its published reference shift.

---

### 1.2 Interactive Pivot Phase & Minimum-Entropy Autophasing
For normalized grid coordinate $u_j = \frac{j}{N - 1} \in [0, 1]$ and configurable pivot $u_{\text{piv}} \in [0, 1]$ (`Pivot1D::Fraction`, `Index`, `Ppm`, or `MaxPeak`), the zero-order (`PHC0`, degrees) and first-order (`PHC1`, degrees across the full sweep width) phase rotation is:
$$\phi_j = \frac{\pi}{180}\Bigl[\text{PHC0} + \text{PHC1} \cdot (u_j - u_{\text{piv}})\Bigr], \qquad S'[j] = S[j]\, e^{+i \phi_j}.$$
If `Spectrum1D` holds only real intensities ($I_0[j]$ absent, e.g. imported from `1r`-only Bruker `pdata` or real-only `JCAMP-DX`), the causal imaginary dispersion channel is reconstructed automatically prior to rotation via exact discrete Hilbert transform (`hilbert_reconstruct_imaginary_1d`).

Automatic phase determination (`auto_phase_1d` / `auto_phase_and_apply_1d`) minimizes the composite Chen et al. (2002) + dispersion-symmetry objective over $R_j(\text{PHC0}, \text{PHC1}) = \text{Re}\bigl(S[j] e^{+i\phi_j}\bigr)$:
$$E(\text{PHC0}, \text{PHC1}) = H_1(R) + P_{\text{neg}}(R) + P_{\text{disp}}(R),$$
where:
1. **Normalized first-derivative Shannon entropy** (Chen et al. 2002) evaluated over the phase-invariant complex-envelope signal+wing mask $\Omega_{\text{sig}}$ (excluding any caller-masked solvent windows):
   $$d_j = |R_{j+1} - R_j|, \qquad h_j = \frac{d_j}{\sum_{l \in \Omega_{\text{sig}}} d_l + \epsilon}, \qquad H_1(R) = -\frac{1}{\ln |\Omega_{\text{sig}}|}\sum_{j \in \Omega_{\text{sig}}} h_j \ln(h_j + \epsilon).$$
2. **Scale-invariant negative-absorption penalty** (active in `Polarity1D::AllPositive`, noise threshold $\tau = c_\sigma \hat{\sigma}$ from complex first-difference MAD):
   $$P_{\text{neg}}(R) = \gamma_{\text{neg}} \frac{\sum_{R_j < -\tau} (|R_j| - \tau)^2}{\sum_{j \in \Omega_{\text{sig}}} R_j^2 + \epsilon}.$$
3. **Per-peak dispersive wing-antisymmetry penalty** (active in both `AllPositive` and `Biphasic`, de Brouwer 2009): across isolated singlet maxima $\{p_m\}$ of $|S[j]|$ with parabolic sub-bin centers $c_m$ and local half-window $w_m$:
   $$P_{\text{disp}}(R) = \frac{\gamma_{\text{disp}}}{M} \sum_{m=1}^M \frac{\sum_{\delta=1}^{w_m} \bigl(R(c_m + \delta) - R(c_m - \delta)\bigr)^2}{R(c_m)^2 + \sum_{\delta=1}^{w_m} \bigl(R(c_m + \delta)^2 + R(c_m - \delta)^2\bigr) + \epsilon}.$$

Optimization is parameterized internally around the spectral energy centroid $u_c = \bigl(\sum_j |S[j]|^2 u_j\bigr) / \bigl(\sum_j |S[j]|^2\bigr)$ via coarse 2D global grid search followed by multi-basin 8-direction Hooke–Jeeves pattern refinement and exact pivot conversion $\text{PHC0}^* = \text{wrap}_{180}\bigl(\phi_c^* + \text{PHC1}^*(u_{\text{piv}} - u_c)\bigr)$.

---

### 1.3 Automatic Baseline Correction (Cobas Wavelet Mask + Whittaker Smoother & Bernstein Basis)
1. **Signal-Free Baseline Weight Mask** (Cobas et al. 2006):
   - Convolves $y \in \mathbb{R}^N$ with normalized Ricker (Mexican-hat) wavelets $\psi_\sigma(k) = \left(1 - \frac{k^2}{\sigma^2}\right) e^{-k^2 / (2\sigma^2)}$ across dyadic scales $\sigma \in \{\sigma_0, 2\sigma_0\}$.
   - Estimates baseline wavelet noise $\hat{\sigma}_W$ via robust MAD on the quietest half of the spectrum, sets $w_i = 0$ wherever $|W_\sigma y[i]| > \kappa \hat{\sigma}_W$, and dilates signal regions by `peak_dilation_pts` to mask Lorentzian peak wings ($w_i = 1$ on pure baseline).
2. **Whittaker Penalized Least Squares** (Eilers 2003; Cobas et al. 2006):
   - Solves the second-difference penalized objective
     $$\min_{b \in \mathbb{R}^N} \sum_{i=0}^{N-1} w_i (y_i - b_i)^2 + \lambda \sum_{i=0}^{N-3} (b_{i+2} - 2 b_{i+1} + b_i)^2 \iff (W + \lambda D_2^\top D_2)\, b = W y$$
     in exact $O(N)$ time and memory using symmetric pentadiagonal $LDL^\top$ factorization, with iteratively reweighted residual masking (`BaselineMethod1D::Whittaker` and `AsymmetricWhittaker`).
3. **Bernstein Polynomial Baseline Fit**:
   - Evaluates the degree-$m$ Bernstein basis $B_{k,m}(u) = \binom{m}{k} u^k (1 - u)^{m - k}$ on $u_i = \frac{i}{N-1} \in [0, 1]$ and solves the masked weighted normal equations $(B^\top W B + \mu I)\, c = B^\top W y$ for variation-diminishing polynomial baselines without Runge edge oscillation.

---

### 1.4 Solvent & Water Resonance Suppression (`HSVD` + Frequency Notch)
1. **Time-Domain Hankel–Lanczos / `HSVD` Subtraction** (Barkhuijsen et al. 1987; Pijnappel et al. 1992):
   - Models the initial $N_H$ complex FID samples ($N_H = L + M - 1$, $L = \lfloor N_H / 2 \rfloor$) as a sum of $K$ exponentially damped complex sinusoids $z[n] = \sum_{k=1}^K c_k \lambda_k^n + \varepsilon_n$, where $\lambda_k = \exp\bigl((-d_k + i 2\pi f_k)\Delta t\bigr)$.
   - Computes the leading $K$-dimensional left singular subspace $U_K \in \mathbb{C}^{L \times K}$ via Golub–Kahan Lanczos bidiagonalization with full reorthogonalization and $O(N_H \log N_H)$ FFT Hankel matrix-vector products, solves $U_{\uparrow} Z \approx U_{\downarrow}$, extracts complex eigenvalues $\{\lambda_1, \dots, \lambda_K\}$ via unitary Hessenberg reduction + Wilkinson-shifted complex QR iteration, solves joint complex Vandermonde least squares $\min_{c \in \mathbb{C}^K} \|V(\lambda)\, c - z\|_2^2$ across all $K$ poles simultaneously, and subtracts strictly the poles inside `solvent_ppm_windows`:
     $$z_{\text{clean}}[n] = z[n] - \sum_{k : \delta_k \in [\delta_{\min}, \delta_{\max}]} c_k \lambda_k^n.$$
2. **Frequency-Domain Notch / Polynomial Suppression** (`suppress_solvent_notch_1d`):
   - Fits a local polynomial of degree $p \in \{1, 2, 3\}$ across flanking baseline collars $[\delta_{\min} - \Delta_c, \delta_{\min}]$ and $[\delta_{\max}, \delta_{\max} + \Delta_c]$ to bridge residual frequency-domain solvent spikes smoothly.

---

### 1.5 Deuterated Solvent Residual & Trace Impurity Reference Table
Built-in reference database compiled from Fulmer et al. (*Organometallics* **2010**, 29, 2176–2179) and Gottlieb et al. (*J. Org. Chem.* **1997**, 62, 7512–7515) covering ${}^{1}\text{H}$ and ${}^{13}\text{C}$ residual solvent signals, ${}^2J_{\text{HD}}$ / ${}^1J_{\text{CD}}$ multiplicities, dissolved $\text{H}_2\text{O}$ / $\text{HDO}$ positions, and common laboratory trace impurities (`TMS`, `H2O`, ethyl acetate, diethyl ether, dichloromethane, acetone, methanol, ethanol, tetrahydrofuran, toluene, benzene, `DMF`, 1,4-dioxane, hydrocarbon grease, silicone grease) across all 10 standard deuterated solvents:
`CDCl3`, `DMSO-d6`, `D2O`, `CD3OD`, `CD3CN`, `acetone-d6`, `THF-d8`, `C6D6`, `CD2Cl2`, and `toluene-d8`.

---

### 1.6 Global 1D Peak Picking & Pseudo-Voigt Lineshape Deconvolution
1. **Peak, Topological Prominence & Shoulder Seeding (`pick_peaks_1d`)**:
   - Identifies local extrema exceeding both $\text{SNR}_{\min} \cdot \hat{\sigma}_{\text{noise}}$ and 1D Morse topological prominence above saddle contours with 3-point parabolic sub-bin refinement, plus optional Ricker-smoothed negative second-derivative ($-y''$) curvature maxima to seed unresolved multiplet shoulders.
2. **Area-Normalized Pseudo-Voigt Model** (Wertheim et al. 1974; Thompson et al. 1987):
   $$V_k(x; \mu_k, \gamma_k, \eta_k, A_k) = A_k \bigl[\eta_k L(x; \mu_k, \gamma_k) + (1 - \eta_k) G(x; \mu_k, \gamma_k)\bigr], \qquad \int_{-\infty}^{+\infty} V_k(x)\,dx = A_k.$$
3. **Coupled-Cluster Projected Levenberg–Marquardt Optimization (`deconvolve_spectrum_1d`, `deconvolve_region_1d`)**:
   - Groups overlapping multiplet clusters within `cluster_merge_hz`, seeds initial areas by linear least squares, optimizes $(\mu_k, \gamma_k, \eta_k, A_k)$ via analytical-Jacobian projected Levenberg–Marquardt with optional residual-peak insertion, and outputs fitted `VoigtPeak1D` components, reconstructed sum trace, and residual `RMSD`.

### 1.7 Hardened 1D Ingest (`nmr-core::nmr1d`, `F15`) & CLI Driver (`nmrassign nmr1d process`, `crates/nmr-cli/src/nmr1d_cmd.rs`)
`nmr_core::nmr1d::load_1d_input` and `load_processed_spectrum_1d` (`nmrassign nmr1d process`) ingest raw Bruker 1D `fid`+`acqus` directories, processed Bruker `pdata` (`procs`+`1r`, optional `1i`), `JCAMP-DX` (`.jdx`/`.dx`), JSON (`TimeDomain1D`/`Spectrum1D`), or 2-/3-column TSV inputs with strict pre-allocation bounds (`MAX_1D_INGEST_POINTS = 4_194_304`, `MAX_1D_INGEST_BYTES = 64 MiB`, exact `1r`/`1i` byte-length vs `SI` verification, and `NaN`/`Inf` rejection; `F15`), execute the requested `HSVD` $\to$ apodization $\to$ zero-fill $\to$ FFT $\to$ pivot/`Chen-2002` autophase $\to$ Whittaker/Bernstein baseline $\to$ Fulmer-2010 solvent calibration $\to$ impurity matching $\to$ global Voigt deconvolution pipeline, and write processed TSV/JSON spectra alongside `--peaks-out` and `--json-out` summaries.

---

## 2. Integration, Recursive Multiplet $J$-Coupling Trees, `Octet` Linking & `qNMR` Purity (`crates/nmr-core/src/multiplet.rs`, `T14.2`)

### 2.1 Region Integration, Edge-OLS Baseline & Exact Variance Propagation
Over each integration window $[\delta_{\text{low}}, \delta_{\text{high}}]$ (`IntegralRegionSpec`), `integrate_regions` supports `BaselineMode::None`, `Explicit { bias, slope }`, and default `AutoLinearEdges { edge_fraction, min_edge_points }` (ordinary least-squares linear baseline $B(\delta) = \text{bias} + \text{slope}\cdot(\delta - \delta_{\text{low}})$ fitted over boundary wing samples). Per-point baseline noise $\sigma_y$ (`estimate_spectral_noise_rms`) is propagated analytically through both composite trapezoidal quadrature weights and the edge-OLS baseline subtraction operator so `abs_sigma` and `norm_sigma` (`NormalizationMode::{None, ReferenceRegion, TotalProtons, ExplicitScale}`) agree with Monte Carlo ensemble standard deviations within statistical sampling error, with optional summation of deconvolved `VoigtPeakComponent` areas.

### 2.2 Recursive Multiplet Tree Extraction, Second-Order `AB` Solver & Roofing (`Hoye 1994 / 2002`)
Given sub-peaks picked in $[\delta_{\text{low}}, \delta_{\text{high}}]$ (with parabolic interpolation and optional Lorentzian tail cross-talk subtraction):
1. **Recursive Hoye Sub-Doublet Peeling & Coalesced Generative Verification** (Hoye, Hanson & Vyvyan 1994; Hoye & Zhao 2002):
   - Peels elementary spin-$1/2$ splittings from the outermost transitions inward while testing coalesced generative trees ($J_1 \ge J_2 \ge \dots$) within frequency tolerance `freq_tol_hz` and coalescence threshold `coalesce_hz`.
   - Accounts for second-order AB roofing ("leaning") via $I_{\text{inner}} / I_{\text{outer}} = \frac{1 + \sin 2\theta}{1 - \sin 2\theta}$ ($\sin 2\theta = J / \sqrt{\Delta\nu^2 + J^2}$, bounded by `min_delta_nu_over_j`), recording per-coupling `leaning_sin2theta`, `estimated_delta_nu_over_j`, `LeaningDirection`, geometric center `center_ppm`, and roofing-corrected `ab_zeeman_ppm`.
   - Provides closed-form 4-line second-order AB spin-system inversion (`solve_ab_quartet`: $J_{AB} = \frac{1}{2}[(\nu_1 - \nu_2) + (\nu_3 - \nu_4)]$, $\Delta\nu_{AB} = \sqrt{(\nu_1 - \nu_4)(\nu_2 - \nu_3)}$) and an exact $2^N$-state spin-$1/2$ Zeeman + scalar-coupling quantum Hamiltonian simulator (`simulate_exact_spin_half_multiplet`) for strongly coupled validation.
2. **Canonical Classification & Lossless Journal Characterization (`ACS`, `JOC`, `RSC`, `Nature`)**:
   - Classifies all 16 canonical patterns: `s`, `br s`, `d`, `t`, `q`, `quint`, `sext`, `sept`, `dd`, `ddd`, `dt` ($J_d > J_t$), `td` ($J_t > J_d$), `dq`, `qd`, `tt`, and fallback `m`.
   - `format_journal_nmr_report` and `parse_journal_nmr_report` provide lossless formatting and round-trip parsing across `PublicationStyle::{Acs, Joc, Rsc, Nature}`, `TextEncoding::{Unicode, Ascii, Latex}`, and `ReportNucleus::{H1, C13, F19, P31, N15}` (including negative ${}^{19}\text{F}$/${}^{31}\text{P}$ shifts and `${}^{13}\text{C}\{{}^{1}\text{H}\}$` decoupling).

### 2.3 Bidirectional `Octet` 2D Structure Atom-Assignment Linking
`StructureAtomMap` and `MultipletAssignmentTable` synchronize 0-based `octet_mol::Molecule` / `RenderScene` atom indices (`octet_atom_idx`), 1-based MDL `MOL`/`SDF` atom numbers (`sdf_atom_number = octet_atom_idx + 1`), and OpenSMILES `[atom:map]` labels (`smiles_atom_map`), with topological Morgan/HOSE symmetry equivalence classes and automatic 1-to-1 `${}^{1}\text{H}$ multiplet-to-structure matching (`auto_assign_1h_multiplets`) exported via `nmrassign nmr1d multiplet --mol|--smiles --links-out`.

### 2.4 Internal-Standard Quantitative NMR (`qNMR`, Pauli et al. 2014; Malz & Jancke 2005)
`calculate_qnmr_purity` and `calculate_qnmr_replicates` (`nmrassign nmr1d qnmr`) compute analyte mass-fraction purity $P_x$ and molar concentration $C_x$ (`mol/L` and `mM` when solution volume $V$ is supplied) against certified internal calibrants (including built-in `qnmr_certified_standards()` lookup for `DMSO2`, `MA`, `TMB`, `DMT`, `BA`, and `DNBA`):
$$P_x = \frac{I_x}{I_{\text{std}}} \cdot \frac{N_{\text{std}}}{N_x} \cdot \frac{\text{MW}_x}{\text{MW}_{\text{std}}} \cdot \frac{m_{\text{std}}}{m_x} \cdot P_{\text{std}},$$
propagating full GUM standard ($k = 1$) and expanded 95% ($k = 2$) uncertainties across $u(I_x)$, $u(I_{\text{std}})$, $u(\text{MW}_x)$, $u(\text{MW}_{\text{std}})$, $u(m_x)$, $u(m_{\text{std}})$, and $u(P_{\text{std}})$, multi-signal / multi-tube replicate `RSD`, and auditing the Pauli (2014) $T_{\text{rep}} = D_1 + \text{AQ} \ge 5\,T_{1,\max}$ longitudinal relaxation requirement (with analytical steady-state Ernst-angle differential saturation bias estimation when $T_{1,x}$ and $T_{1,\text{std}}$ are supplied).

---

## 3. Multi-Spectrum 1D Comparison, Stack, Difference & Kinetics Studio (`crates/nmr-gui/src/stack1d_panel.rs`, `T14.3`)

### 3.1 Four Comparison Geometries (`Stack1dViewMode`) & Multi-Resolution `LodPyramid1d`
Given an ordered collection of $M$ 1D traces $\{S_m(\delta)\}_{m=0}^{M-1}$ (`Trace1d`, each pre-indexed with a binary min/max `LodPyramid1d` preserving sub-pixel sharp peak apices across $32 \times 64\text{k}$ points at 60 fps) wired into the native desktop & `wasm32` `nmr-gui` workbench (`📚 1D Stack / Kinetics`):
1. **`SuperimposedOverlay`**: Overlays $y_m(\delta) = \alpha_m S_m(\delta) + y_{0,m}$ with per-trace visibility, lock, color, gain $\alpha_m$, and baseline offset.
2. **`StackedOffset`**: Oblique pseudo-3D / vertical stack $x_{\text{screen}}(m, \delta) = x(\delta) + m\,\Delta x_{\text{tilt}}$, $y_{\text{screen}}(m, \delta) = y(\alpha_m S_m(\delta)) + m\,\Delta y_{\text{step}}$ for clean titration, pH, VT, and reaction-time series inspection.
3. **`DifferenceSpectrum`**: Computes $\Delta S_m(\delta) = S_m(\delta) - \alpha_m^* S_{\text{ref}}(\delta)$ for STD-NMR, batch impurity profiling, and reaction difference spectroscopy, with either manual $\alpha_m$ or ordinary-least-squares reference-region matching (`estimate_difference_alpha_ols`):
   $$\alpha_m^* = \frac{\sum_{\delta_j \in \Omega_{\text{ref}}} S_m(\delta_j)\, S_{\text{ref}}(\delta_j)}{\sum_{\delta_j \in \Omega_{\text{ref}}} S_{\text{ref}}(\delta_j)^2 + \epsilon}.$$
4. **`Pseudo2dHeatmap`**: Renders intensity $S_m(\delta)$ as a cached `egui::ColorImage` 2D $(\delta \times \text{covariate}_m)$ raster / waterfall map across time, ligand equivalents, temperature, or pH.

### 3.2 Normalization, Interval Correlation-Shifting FFT Alignment & Kinetics Export
- **Normalization modes (`normalize_traces`)**: raw intensity, total absolute area ($\sum_j |S_m(\delta_j)|\,\Delta\delta = 1$), reference-region peak apex, or reference-region integral normalization.
- **Interval correlation-shifting FFT alignment (`align_traces_interval_fft`)** (Savorani et al., *J. Magn. Reson.* **2010**, 202, 190–202): globally or across user-defined chemical-shift intervals $I = [\delta_{\min}, \delta_{\max}]$, computes zero-padded radix-2 FFT circular cross-correlation with parabolic sub-sample peak interpolation and cubic Catmull–Rom resampling (`cubic_catmull_rom_sample`) + cosine-tapered boundary blending:
  $$C_m[\Delta] = \mathcal{F}^{-1}\!\Bigl(\mathcal{F}(\tilde{S}_{\text{ref}, I}) \odot \overline{\mathcal{F}(\tilde{S}_{m, I})}\Bigr)[\Delta], \qquad \Delta_m^* = \arg\max_{|\Delta| \le \Delta_{\max}} C_m[\Delta].$$
- **Kinetics / Titration ROI Matrix & Vector SVG Export (`compute_roi_matrix`, `export_stack1d_svg`)**: integrates $R$ user-selected regions $\{\Omega_r\}_{r=1}^R$ across all $M$ traces to emit reaction conversion / decay ratios (`RoiRatioFormula`), clean TSV matrices, and publication SVG figures, with direct drag-and-drop / paste import of `JCAMP-DX` (`.jdx`/`.dx`) and `ppm\treal` TSV 1D traces in `nmr-gui`.

---

## 4. Pseudo-2D Diffusion-Ordered Spectroscopy (`DOSY`) Engine, WebGPU Solver & Panel (`T14.4`)

### 4.1 Stejskal–Tanner Attenuation Factor $b$ & Spectrometer Ingest (`crates/nmr-core/src/dosy.rs`)
For pulsed field gradient diffusion experiments, signal amplitude as a function of gradient strength $g$ ($\text{T m}^{-1}$, where $1\text{ G cm}^{-1} = 10^{-2}\text{ T m}^{-1}$) decays according to $I(b) = \sum_{k=1}^K I_{0,k} \exp(-b D_k)$ with $D_k$ in $\text{m}^2\text{ s}^{-1}$ and diffusion weighting factor $b$ ($\text{s m}^{-2}$) (Stejskal & Tanner 1965; Wu, Chen & Johnson 1995; Pelta et al. 2002):
$$b(g) = \bigl(\gamma\, g_{\text{eff}}(g)\, \delta\bigr)^2 \, \Delta_{\text{eff}},$$
where:
- $\gamma$ is the nuclear gyromagnetic ratio ($\gamma_{1\text{H}} = 2.6752218744 \times 10^8\text{ rad s}^{-1}\text{ T}^{-1}$),
- $\delta$ is the total diffusion-encoding gradient duration per defocusing/refocusing pair (`2 * P30` for bipolar Bruker `dstebpgp3s` / `ledbpgp2s` where `P30` is a single half-bipolar lobe, or `P30` for monopolar `STE`),
- $g_{\text{eff}}(g) = \sigma_{\text{shape}}\, g$ accounts for gradient pulse envelope ($\sigma_{\text{rect}} = 1$, $\sigma_{\text{sine}} = 2 / \pi$ for half-sine `SINE.100` lobes),
- $\Delta_{\text{eff}}$ is the effective diffusion delay:
  - Stimulated echo (`STE`): $\Delta_{\text{eff}} = \Delta - \frac{\delta}{3}$,
  - Bipolar gradient pulse pair LED (`BPP-LED`): $\Delta_{\text{eff}} = \Delta - \frac{\delta}{3} - \frac{\tau}{2}$, where $\tau$ is the inter-bipolar gradient/RF recovery delay.

Spectrometer ingest parses both Bruker (`difflist` in $\text{G cm}^{-1}$ or `%` scaled against `acqus` `D[20]` = $\Delta$, `P[30]`, `GPNAM`) and Varian/Agilent (`procpar` `gzlvl1` DAC units $\times$ `gcal_` calibration, `gt1` = $\delta$, `del` = $\Delta$).

### 4.2 Discrete Multi-Exponential Levenberg–Marquardt (`CPU` & `WebGPU`) + Model Selection
1. **Default Reproducible Discrete Fit ($K \in \{1, 2, 3\}$)**:
   - Fits $I(b_m) = \sum_{k=1}^K I_{0,k} \exp(-b_m D_k)$ ($I_{0,k} > 0$, $0 < D_1 < \dots < D_K$) via Levenberg–Marquardt on CPU (`f64`, $K = 1, 2, 3$) and massively parallel per-column / per-peak WGSL compute shaders (`crates/nmr-gpu/src/dosy_gpu.rs` + `shaders/dosy.wgsl`, `f32` for $K = 1, 2$, verified to $< 1\%$ relative $D$ parity against `nmr-core::dosy`).
   - Selects component count $K^* \in \{1, \dots, K_{\max}\}$ minimizing the Bayesian Information Criterion (`BIC`, or optional `AIC` with Gaussian residual variance $\hat{\sigma}^2 = \text{RSS}_K / M$, $p_K = 2K$ parameters):
     $$\text{BIC}(K) = M \ln\!\left(\frac{\text{RSS}_K}{M}\right) + 2K \ln M, \qquad \text{AIC}(K) = M \ln\!\left(\frac{\text{RSS}_K}{M}\right) + 4K,$$
     preventing spurious multi-exponential splitting on pure single-species peaks, and estimates 95% confidence intervals $[D_{k, 2.5\%}, D_{k, 97.5\%}]$ via residual bootstrap resampling.

2. **Opt-In Regularized Inverse Laplace Transform (`ILT`, Provencher 1982; Lawson & Hanson 1974)**:
   - Over a log-uniform grid $\{D_q\}_{q=1}^Q \subset [D_{\min}, D_{\max}]$ with Laplace kernel $K_{m,q} = \exp(-b_m D_q)$, solves the second-difference Tikhonov-regularized non-negative least-squares problem:
     $$\min_{f \ge 0} \|K f - y\|_2^2 + \lambda \|L_2 f\|_2^2$$
     via projected active-set / NNLS on the augmented system $\begin{bmatrix} K \\ \sqrt{\lambda}\, L_2 \end{bmatrix} f \approx \begin{bmatrix} y \\ 0 \end{bmatrix}$, selecting $\lambda > 0$ via baseline-$\sigma$ discrepancy, Generalized Cross-Validation (`GCV`), or L-curve corner curvature.
   - **Scientific Caveat**: Continuous real-valued Laplace inversion is exponentially ill-conditioned; unresolved chemical-shift overlap or low SNR can shift apparent `ILT` peaks toward intermediate "compromise" diffusion coefficients or split broad polydisperse modes into narrow pseudo-peaks. Discrete `BIC`-selected Stejskal–Tanner fitting is therefore the default quantitative mode, with `ILT` exposed as an opt-in continuous distribution view accompanied by explicit conditioning caveats.

### 4.3 Pseudo-2D $(\delta, \log_{10} D)$ Synthesis & Stokes–Einstein Hydrodynamics
- **Pseudo-2D `DOSY` Projection**: synthesizes $Z(\delta_i, \log_{10} D_q) = \sum_{k=1}^{K^*_i} I_{i,k} \exp\!\left(-\frac{(\log_{10} D_q - \log_{10} D_{i,k})^2}{2 \sigma_{\log_{10} D, i,k}^2}\right)$ aligned beneath the 1D top trace (`crates/nmr-gui/src/dosy_panel.rs` wired into `WorkspaceMode::Dosy` in `crates/nmr-gui/src/main.rs`, `wasm32`-compatible).
- **Stokes–Einstein Hydrodynamic Radius & Power-Law Molecular Weight** (Evans et al. 2013):
  $$R_h = \frac{k_B T}{6 \pi \eta(T) D}, \qquad \widehat{\text{MW}} = \left(\frac{\alpha_{\text{solv}}}{D}\right)^{1 / \beta_{\text{solv}}},$$
  with built-in temperature-compensated dynamic viscosities $\eta(T)$ ($\text{Pa s}$) for $\text{D}_2\text{O}$, $\text{H}_2\text{O}$, $\text{CDCl}_3$, $\text{DMSO-}d_6$, $\text{CD}_3\text{OD}$, and $\text{CD}_3\text{CN}$, plus optional internal diffusivity-calibrant scaling (`estimate_mw_with_internal_reference`).

### 4.4 Command-Line Interface (`crates/nmr-cli/src/dosy_cmd.rs`)
- `nmrassign dosy fit`: ingests `DOSY` decay TSV/JSON tables or Bruker (`difflist` + `acqus`) / Varian (`procpar`) gradient directories (or `--demo`), runs reproducible discrete `1..=3` component Stejskal–Tanner / Wu–Johnson fitting on CPU (`--engine cpu`) or WebGPU (`--engine gpu`, `fit_dosy_columns_gpu`), selects model order via `BIC` (default) or `AIC`, computes residual bootstrap 95% CIs (`--bootstrap`), and writes TSV/JSON summaries.
- `nmrassign dosy ilt`: opt-in `CONTIN`-style second-derivative (or zeroth-/first-order) Tikhonov non-negative Inverse Laplace Transform over a log-spaced diffusion grid (`--strategy gcv|lcurve|discrepancy|fixed`, plus `--alpha-sweep-out` auditing mode count and residual norm across $\alpha$).
- `nmrassign dosy mw`: Stokes–Einstein $R_h$ ($\text{\AA}$, $\text{nm}$) and Evans (2013) empirical power-law molecular-weight estimation across built-in deuterated solvents or explicit `--viscosity-mpas`, with optional `--standard D_ref:MW_ref` internal calibration.
