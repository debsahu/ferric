# 17. Analytical 1D NMR Processing, Multiplet / qNMR Analysis, Multi-Spectrum Comparison, and Pseudo-2D DOSY Studio

## Provenance & Clean-Room Statement
Original clean-room specification and mathematical design for Phase 14 (`v0.10.0`: `T14.1`–`T14.4`). Zero source code, macros, parameter libraries, binaries, or output templates from any proprietary, academic-licensed, or copyleft (GPL/AGPL) NMR processing or assignment package were consulted. Every mathematical model, chemical-shift reference value, and numerical solver is derived strictly from peer-reviewed publications and public spectrometer format documentation:

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
3. **Recursive Multiplet Coupling Trees & Quantitative NMR (`T14.2`)**:
   - Hoye, T. R., Hanson, P. R. & Vyvyan, J. R. (*J. Org. Chem.* **1994**, 59, 4096–4103); Hoye, T. R. & Zhao, H. (*J. Org. Chem.* **2002**, 67, 4014–4016) — practical recursive $J$-coupling tree extraction and second-order $\Delta\nu / J$ roofing ("leaning") intensity analysis.
   - Pauli, G. F., Chen, S.-N., Simmler, C., Lankin, D. C., Gödecke, T., Jaki, B. U., Friesen, J. B., McAlpine, J. B. & Napolitano, J. G. (*J. Med. Chem.* **2014**, 57, 9220–9231) — internal-standard quantitative ${}^{1}\text{H}$ NMR (`qNMR`) molar purity determination, longitudinal relaxation ($5 T_1$) constraints, and uncertainty propagation.
4. **Multi-Spectrum 1D Alignment & Comparison (`T14.3`)**:
   - Savorani, F., Tomasi, G. & Engelsen, S. B. (*J. Magn. Reson.* **2010**, 202, 190–202) — interval correlation-shifting (`icoshift`) fast Fourier cross-correlation peak alignment across arrayed 1D NMR series.
5. **Diffusion-Ordered Spectroscopy (`DOSY`) Physics & Inversion (`T14.4`)**:
   - Stejskal, E. O. & Tanner, J. E. (*J. Chem. Phys.* **1965**, 42, 288–292); Wu, D. H., Chen, A. D. & Johnson, C. S. (*J. Magn. Reson. A* **1995**, 115, 260–264); Pelta, M. D., Morris, G. A., Stchedroff, M. J. & Hammond, S. J. (*Magn. Reson. Chem.* **2002**, 40, 147–152) — pulsed field gradient spin-echo (`STE`) and bipolar gradient pulse pair longitudinal eddy-current delay (`BPP-LED`) diffusion attenuation equations and gradient shape factors.
   - Lawson, C. L. & Hanson, R. J. (*Solving Least Squares Problems*, Prentice-Hall, **1974**); Provencher, S. W. (*Comput. Phys. Commun.* **1982**, 27, 213–227) — non-negative Tikhonov-regularized Inverse Laplace Transform (`ILT`) over log-spaced diffusion grids.
   - Evans, R., Deng, Z., Rogerson, A. K., McLachlan, A. S., Richards, J. J., Nilsson, M. & Morris, G. A. (*Angew. Chem. Int. Ed.* **2013**, 52, 3199–3202) — Stokes–Einstein hydrodynamic radius $R_h$ and empirical power-law molecular-weight estimation from translational diffusion coefficients.

---

## 1. 1D Small-Molecule & Biomolecular DSP Core (`crates/nmr-core/src/nmr1d.rs`, `T14.1`)

### 1.1 Multi-Nuclear Representation, Apodization, Zero-Filling, FFT & Referencing
The 1D DSP module supports small-molecule, natural-product, organometallic, fluorinated, organophosphorus, and biomolecular 1D experiments through explicit nuclear and spin-editing metadata:
- **Nuclei (`Nucleus1D`)**: `${}^{1}\text{H}$ ($\Xi = 1.00000000$), `${}^{13}\text{C}$ ($\Xi = 0.25145020$), `${}^{19}\text{F}$ ($\Xi = 0.94094011$), `${}^{31}\text{P}$ ($\Xi = 0.40480742$), `${}^{15}\text{N}$ ($\Xi = 0.10136767$), and arbitrary user-specified nuclei.
- **Experiment kinds (`Experiment1DKind`)**:
  - `Proton1D` (${}^{1}\text{H}$), `Carbon13Decoupled` (${}^{13}\text{C}\{{}^{1}\text{H}\}$), `Fluorine19` (${}^{19}\text{F}$), `Phosphorus31` (${}^{31}\text{P}$): expected absorption polarity `Polarity1D::AllPositive`.
  - `Dept90`: `${}^{13}\text{C}$ polarization transfer at read flip angle $\theta = 90^\circ$ ($\text{CH} > 0$, $\text{CH}_2 \approx 0$, $\text{CH}_3 \approx 0$, $\text{C}_{\text{quat}} = 0$; Doddrell et al. 1982), `Polarity1D::AllPositive`.
  - `Dept135`: `${}^{13}\text{C}$ polarization transfer at $\theta = 135^\circ$ ($\text{CH}, \text{CH}_3 > 0$, $\text{CH}_2 < 0$, $\text{C}_{\text{quat}} = 0$), `Polarity1D::Biphasic`.
  - Theoretical DEPT transfer factor for $\text{CH}_n$ ($n \in \{0, 1, 2, 3\}$) at read pulse angle $\theta$:
    $$I_{\text{DEPT}}(n, \theta) = \begin{cases} 0, & n = 0, \\ \sin\theta, & n = 1, \\ \sin(2\theta), & n = 2, \\ \frac{3}{4}(\sin\theta + \sin 3\theta), & n = 3. \end{cases}$$

Given a complex time-domain FID $z[n] \in \mathbb{C}$ ($n = 0, \dots, N_{\text{td}} - 1$) with dwell time $\Delta t = 1 / \text{SW}_{\text{Hz}}$, observation frequency $\nu_{\text{obs}}\text{ (MHz)}$, and carrier chemical shift $\delta_{\text{carrier}}\text{ (ppm)}$:

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
   - Automatic solvent referencing locates the parabolic sub-bin centroid of the strongest peak inside $[\delta_{\text{Fulmer}} - \Delta\delta_{\text{win}}, \delta_{\text{Fulmer}} + \Delta\delta_{\text{win}}]$ and translates `axis.first_ppm` so the residual solvent resonance aligns with its published reference shift.

---

### 1.2 Interactive Pivot Phase & Minimum-Entropy Autophasing
For normalized grid coordinate $u_j = \frac{j}{N - 1} \in [0, 1]$ and configurable pivot $u_{\text{piv}} \in [0, 1]$ (expressible as fractional index, bin index, chemical shift $\delta_{\text{piv}}$, or automatic maximum-envelope peak), the zero-order (`PHC0`, degrees) and first-order (`PHC1`, degrees across the full sweep width) phase rotation is:
$$\phi_j = \frac{\pi}{180}\Bigl[\text{PHC0} + \text{PHC1} \cdot (u_j - u_{\text{piv}})\Bigr], \qquad S'[j] = S[j]\, e^{+i \phi_j}.$$
If `Spectrum1D` holds only real intensities ($I_0[j]$ absent), the causal imaginary dispersion channel is reconstructed prior to rotation via exact discrete Hilbert transform on the un-reversed DFT ordering (`IFFT -> [1, 2, ..., 2, 1, 0, ..., 0] -> FFT`).

Automatic phase determination minimizes the composite Chen et al. (2002) + dispersion-symmetry objective over $R_j(\text{PHC0}, \text{PHC1}) = \text{Re}\bigl(S[j] e^{+i\phi_j}\bigr)$:
$$E(\text{PHC0}, \text{PHC1}) = H_1(R) + P_{\text{neg}}(R) + P_{\text{disp}}(R),$$
where:
1. **Normalized first-derivative Shannon entropy** (Chen et al. 2002) evaluated over the phase-invariant complex-envelope signal+wing mask $\Omega_{\text{sig}}$:
   $$d_j = |R_{j+1} - R_j|, \qquad h_j = \frac{d_j}{\sum_{l \in \Omega_{\text{sig}}} d_l + \epsilon}, \qquad H_1(R) = -\frac{1}{\ln |\Omega_{\text{sig}}|}\sum_{j \in \Omega_{\text{sig}}} h_j \ln(h_j + \epsilon).$$
2. **Scale-invariant negative-absorption penalty** (active in `Polarity1D::AllPositive`, noise threshold $\tau = c_\sigma \hat{\sigma}$ from complex first-difference MAD):
   $$P_{\text{neg}}(R) = \gamma_{\text{neg}} \frac{\sum_{R_j < -\tau} (|R_j| - \tau)^2}{\sum_{j \in \Omega_{\text{sig}}} R_j^2 + \epsilon}.$$
3. **Per-peak dispersive wing-antisymmetry penalty** (active in both `AllPositive` and `Biphasic`, de Brouwer 2009): across isolated singlet maxima $\{p_m\}$ of $|S[j]|$ with parabolic sub-bin centers $c_m$ and local half-window $w_m$:
   $$P_{\text{disp}}(R) = \frac{\gamma_{\text{disp}}}{M} \sum_{m=1}^M \frac{\sum_{\delta=1}^{w_m} \bigl(R(c_m + \delta) - R(c_m - \delta)\bigr)^2}{R(c_m)^2 + \sum_{\delta=1}^{w_m} \bigl(R(c_m + \delta)^2 + R(c_m - \delta)^2\bigr) + \epsilon}.$$

To eliminate the narrow diagonal valley coupling `PHC0` and `PHC1`, optimization is parameterized internally around the spectral energy centroid $u_c = \bigl(\sum_j |S[j]|^2 u_j\bigr) / \bigl(\sum_j |S[j]|^2\bigr)$ via coarse 2D global grid search over $(\phi_c, \text{PHC1}) \in [-180^\circ, 180^\circ) \times [-\text{PHC1}_{\max}, +\text{PHC1}_{\max}]$ followed by multi-basin 8-direction Hooke–Jeeves pattern refinement to $< 10^{-2}\text{ deg}$ and exact pivot conversion $\text{PHC0}^* = \text{wrap}_{180}\bigl(\phi_c^* + \text{PHC1}^*(u_{\text{piv}} - u_c)\bigr)$.

---

### 1.3 Automatic Baseline Correction (Cobas Wavelet Mask + Whittaker Smoother & Bernstein Basis)
1. **Signal-Free Baseline Weight Mask** (Cobas et al. 2006):
   - Convolves $y \in \mathbb{R}^N$ with normalized Ricker (Mexican-hat) wavelets $\psi_\sigma(k) = \left(1 - \frac{k^2}{\sigma^2}\right) e^{-k^2 / (2\sigma^2)}$ (zero-mean and orthogonal to linear ramps) across dyadic scales $\sigma \in \{\sigma_0, 2\sigma_0\}$.
   - Estimates baseline wavelet noise $\hat{\sigma}_W$ iteratively via robust MAD on the quietest half of the spectrum, sets $w_i = 0$ wherever $|W_\sigma y[i]| > \kappa \hat{\sigma}_W$, and dilates signal regions by `dilation_pts` to mask Lorentzian peak wings ($w_i = 1$ on pure baseline).
2. **Whittaker Penalized Least Squares** (Eilers 2003; Cobas et al. 2006):
   - Solves the second-difference penalized objective
     $$\min_{b \in \mathbb{R}^N} \sum_{i=0}^{N-1} w_i (y_i - b_i)^2 + \lambda \sum_{i=0}^{N-3} (b_{i+2} - 2 b_{i+1} + b_i)^2 \iff (W + \lambda D_2^\top D_2)\, b = W y$$
     in exact $O(N)$ time and memory using symmetric pentadiagonal $LDL^\top$ factorization, with optional iteratively reweighted residual masking (and asymmetric least-squares weights $w_i = p\,\mathbb{I}[y_i > b_i] + (1-p)\,\mathbb{I}[y_i \le b_i]$).
3. **Bernstein Polynomial Baseline Fit**:
   - Evaluates the degree-$m$ Bernstein basis on $u_i = \frac{i}{N-1} \in [0, 1]$:
     $$B_{k,m}(u) = \binom{m}{k} u^k (1 - u)^{m - k}, \qquad k = 0, \dots, m,$$
     and solves the masked weighted normal equations $(B^\top W B + \mu I)\, c = B^\top W y$ for smooth, variation-diminishing global polynomial baselines without Runge edge oscillation.

---

### 1.4 Solvent & Water Resonance Suppression (`HSVD` + Frequency Notch)
1. **Time-Domain Hankel–Lanczos / `HSVD` Subtraction** (Barkhuijsen et al. 1987; Pijnappel et al. 1992):
   - Models the initial $N_H$ complex FID samples ($N_H = L + M - 1$, $L = \lfloor N_H / 2 \rfloor + 1$) as a sum of $K$ exponentially damped complex sinusoids $z[n] = \sum_{k=1}^K c_k \lambda_k^n + \varepsilon_n$, where $\lambda_k = \exp\bigl((-d_k + i 2\pi f_k)\Delta t\bigr)$.
   - Forms the $L \times M$ Hankel operator $H_{i,j} = z[i + j]$ and computes its leading $K$-dimensional left singular subspace $U_K \in \mathbb{C}^{L \times K}$ via Golub–Kahan Lanczos bidiagonalization with full reorthogonalization and $O(N_H \log N_H)$ FFT Hankel matrix-vector products.
   - Solves the shift-invariance relation $U_{\uparrow} Z \approx U_{\downarrow}$ ($Z = (U_{\uparrow}^H U_{\uparrow})^{-1} U_{\uparrow}^H U_{\downarrow} \in \mathbb{C}^{K \times K}$), extracts complex eigenvalues $\{\lambda_1, \dots, \lambda_K\}$ via unitary Hessenberg reduction and Wilkinson-shifted complex QR iteration, and converts each pole $\lambda_k$ to chemical shift $\delta_k$ and damping $d_k$.
   - Solves joint complex linear least squares $\min_{c \in \mathbb{C}^K} \|V(\lambda)\, c - z\|_2^2$ across **all** $K$ signal poles simultaneously so non-orthogonal overlap between a dominant solvent peak and an adjacent solute peak does not bias the solvent amplitude, then subtracts strictly the poles falling inside the target solvent window $[\delta_{\min}, \delta_{\max}]$ across the full FID $n = 0, \dots, N_{\text{td}} - 1$:
     $$z_{\text{clean}}[n] = z[n] - \sum_{k : \delta_k \in [\delta_{\min}, \delta_{\max}]} c_k \lambda_k^n.$$
2. **Frequency-Domain Notch / Polynomial Suppression**:
   - Fits a weighted local polynomial of degree $p \in \{1, 2, 3\}$ to flanking baseline collars $[\delta_{\min} - \Delta_c, \delta_{\min}]$ and $[\delta_{\max}, \delta_{\max} + \Delta_c]$ and replaces the interior solvent notch $[\delta_{\min}, \delta_{\max}]$ smoothly.

---

### 1.5 Deuterated Solvent Residual & Trace Impurity Reference Table
Built-in reference database compiled from Fulmer et al. (*Organometallics* **2010**, 29, 2176–2179) and Gottlieb et al. (*J. Org. Chem.* **1997**, 62, 7512–7515) covering `${}^{1}\text{H}$ and `${}^{13}\text{C}$ residual solvent signals, ${}^2J_{\text{HD}}$ / ${}^1J_{\text{CD}}$ multiplicities, dissolved $\text{H}_2\text{O}$ / $\text{HDO}$ positions, and common laboratory trace impurities (`TMS`, `H2O`, ethyl acetate, diethyl ether, dichloromethane, acetone, methanol, ethanol, tetrahydrofuran, toluene, benzene, `DMF`, 1,4-dioxane, hydrocarbon grease, silicone grease) across all 10 standard deuterated solvents:
`CDCl3`, `DMSO-d6`, `D2O`, `CD3OD`, `CD3CN`, `acetone-d6`, `THF-d8`, `C6D6`, `CD2Cl2`, and `toluene-d8`.

---

### 1.6 Global 1D Peak Picking & Pseudo-Voigt Lineshape Deconvolution
1. **Peak, Topological Prominence & Shoulder Seeding**:
   - Identifies local extrema exceeding both $\text{SNR}_{\min} \cdot \hat{\sigma}_{\text{noise}}$ and 1D Morse topological prominence above saddle contours (eliminating 1-bin thermal noise fluctuations on broad Lorentzian wings) with 3-point parabolic sub-bin refinement, plus optional Ricker-smoothed negative second-derivative ($-y''$) curvature maxima to seed unresolved multiplet shoulders.
2. **Area-Normalized Pseudo-Voigt Model** (Wertheim et al. 1974; Thompson et al. 1987):
   Each resonance component $k \in \{1, \dots, K\}$ is parameterized directly by center $\mu_k$ (ppm), full width at half maximum $\gamma_k > 0$ (ppm), Lorentzian fraction $\eta_k \in [0, 1]$, and exact analytical integrated area $A_k$ ($\text{intensity} \times \text{ppm}$):
   $$V_k(x; \mu_k, \gamma_k, \eta_k, A_k) = A_k \bigl[\eta_k L(x; \mu_k, \gamma_k) + (1 - \eta_k) G(x; \mu_k, \gamma_k)\bigr],$$
   $$L(x; \mu_k, \gamma_k) = \frac{2}{\pi \gamma_k} \frac{1}{1 + 4\left(\frac{x - \mu_k}{\gamma_k}\right)^2}, \qquad G(x; \mu_k, \gamma_k) = \frac{2\sqrt{\ln 2}}{\sqrt{\pi}\,\gamma_k} \exp\!\left(-4\ln 2 \left(\frac{x - \mu_k}{\gamma_k}\right)^2\right),$$
   satisfying $\int_{-\infty}^{+\infty} L(x)\,dx = \int_{-\infty}^{+\infty} G(x)\,dx = 1$ identically.
3. **Coupled-Cluster Projected Levenberg–Marquardt Optimization**:
   - Peaks whose Lorentzian wings interact within `merge_gap_ppm` are grouped into coupled spectral windows and initialized via non-negative/linear least squares for $A_1, \dots, A_K$ followed by analytical-Jacobian projected Levenberg–Marquardt $(J^\top J + \lambda_{\text{LM}} \operatorname{diag}(J^\top J))\,\Delta\theta = J^\top r$ under box constraints $\gamma_k \in [\gamma_{\min}, \gamma_{\max}]$, $\eta_k \in [0, 1]$, returning fitted positions, FWHMs (ppm & Hz), Lorentz/Gauss fractions, analytical areas, peak heights, and window residual `RMSD`.

---

## 2. Integration, Recursive Multiplet $J$-Coupling Trees & `qNMR` Purity (`crates/nmr-core/src/multiplet.rs`, `T14.2`)

### 2.1 Region Integration & Uncertainty Propagation
Over an integration interval $[\delta_a, \delta_b]$ ($M$ spectral bins, bin width $|\Delta\delta|$), optional local linear bias/slope correction estimates $b_{\text{lin}}(\delta)$ from edge collars so $I_{\text{corr}} = \sum_{j \in [\delta_a, \delta_b]} (y_j - b_{\text{lin},j})\,|\Delta\delta|$. Propagated thermal noise standard uncertainty for uncorrelated bin noise $\sigma_y$ (with baseline edge-averaging factor) is $\sigma_I = \sigma_y |\Delta\delta| \sqrt{M (1 + M / M_{\text{collar}})}$. Relative proton counts normalize against a user-selected reference region of stoichiometry $N_{\text{ref}}$ (`1.00 H`, `2.00 H`, `3.00 H`, etc.).

### 2.2 Recursive Multiplet Tree Extraction with Second-Order Roofing (`Hoye 1994 / 2002`)
Given ordered component frequencies $\nu_1 > \nu_2 > \dots > \nu_K$ (Hz, $\nu_k = \delta_k \nu_{\text{obs}}$) and areas/intensities $a_1, \dots, a_K$:
1. **Symmetric & Roofing-Tolerant Pair Decomposition** (Hoye, Hanson & Vyvyan 1994; Hoye & Zhao 2002):
   - Candidate outermost splittings $J_1 = |\nu_1 - \nu_r|$ are tested recursively by stripping doublets (or higher binomial $1:\binom{m}{1}:\dots:1$ sub-multiplets when degenerate within $\Delta J_{\text{tol}}$) from the outer edges inward.
   - Second-order AB "leaning" (roofing) modifies outer/inner line intensity ratios by $I_{\text{inner}} / I_{\text{outer}} = \frac{1 + \sin 2\theta}{1 - \sin 2\theta}$ where $\sin 2\theta = J / \sqrt{\Delta\nu^2 + J^2}$. Candidate branches accept asymmetric pair intensities consistent with $|\Delta\nu / J| \ge (\Delta\nu / J)_{\min}$ while preserving total pairwise center-of-gravity and sum of intensities.
2. **Canonical Classification & Journal Formatting**:
   - Classifies `s`, `br s`, `d`, `t`, `q`, `quint`, `sext`, `sept`, `dd`, `ddd`, `dt`, `td`, `dq`, `qd`, `tt`, and fallback `m` (reporting chemical-shift range $\delta_{\min}\text{–}\delta_{\max}$ for `m`, weighted centroid $\bar{\delta}$ and sorted descending $J_1 \ge J_2 \ge \dots$ in Hz otherwise).
   - Formats and parses `ACS`/`JOC`, `RSC`, and `Nature` experimental strings (Unicode, plain ASCII, or LaTeX) and links multiplets bidirectionally to `Octet` 2D molecular graph atom indices.

### 2.3 Internal-Standard Quantitative NMR (`qNMR`, Pauli et al. 2014)
Analyte mass-fraction purity $P_x$ against a certified internal calibrant (`std`) in the same NMR tube is:
$$P_x = \frac{I_x}{I_{\text{std}}} \cdot \frac{N_{\text{std}}}{N_x} \cdot \frac{\text{MW}_x}{\text{MW}_{\text{std}}} \cdot \frac{m_{\text{std}}}{m_x} \cdot P_{\text{std}},$$
where $I_x, I_{\text{std}}$ are absolute integrals, $N_x, N_{\text{std}}$ are the number of magnetically equivalent nuclei contributing to each signal, $\text{MW}_x, \text{MW}_{\text{std}}$ are molar masses ($\text{g mol}^{-1}$), $m_x, m_{\text{std}}$ are weighed masses, and $P_{\text{std}}$ is calibrant purity. First-order relative combined uncertainty satisfies:
$$\left(\frac{u(P_x)}{P_x}\right)^2 = \left(\frac{u(I_x)}{I_x}\right)^2 + \left(\frac{u(I_{\text{std}})}{I_{\text{std}}}\right)^2 + \left(\frac{u(m_x)}{m_x}\right)^2 + \left(\frac{u(m_{\text{std}})}{m_{\text{std}}}\right)^2 + \left(\frac{u(P_{\text{std}})}{P_{\text{std}}}\right)^2,$$
combined with replicate relative standard deviation (`RSD`) across multi-tube / multi-region measurements and a quantitative recycle-delay guard flagging whenever $t_{\text{recycle}} = d_1 + t_{\text{aq}} < 5\,T_{1,\max}$ (since incomplete longitudinal recovery introduces systematic error $\le e^{-5} \approx 0.67\%$ only at $\ge 5 T_1$).

---

## 3. Multi-Spectrum 1D Comparison, Stack, Difference & Kinetics Studio (`crates/nmr-gui/src/stack1d_panel.rs`, `T14.3`)

### 3.1 Four Comparison Geometries (`Stack1dViewMode`)
Given an ordered collection of $M$ 1D traces $\{S_m(\delta)\}_{m=0}^{M-1}$ resampled or evaluated over a common descending chemical-shift viewport $[\delta_{\text{low}}, \delta_{\text{high}}]$:
1. **`SuperimposedOverlay`**: Overlays $y_m(\delta) = \alpha_m S_m(\delta) + y_{0,m}$ with per-trace visibility, lock, color, gain $\alpha_m$, and baseline offset.
2. **`StackedOffset`**: Oblique pseudo-3D / vertical stack $x_{\text{screen}}(m, \delta) = x(\delta) + m\,\Delta x_{\text{tilt}}$, $y_{\text{screen}}(m, \delta) = y(\alpha_m S_m(\delta)) + m\,\Delta y_{\text{step}}$ for clean titration, pH, VT, and reaction-time series inspection.
3. **`DifferenceSpectrum`**: Computes $\Delta S_m(\delta) = S_m(\delta) - \alpha_m^* S_{\text{ref}}(\delta)$ for STD-NMR, batch impurity profiling, and reaction difference spectroscopy, with either manual $\alpha_m$ or least-squares reference-region matching:
   $$\alpha_m^* = \frac{\sum_{\delta_j \in \Omega_{\text{ref}}} S_m(\delta_j)\, S_{\text{ref}}(\delta_j)}{\sum_{\delta_j \in \Omega_{\text{ref}}} S_{\text{ref}}(\delta_j)^2 + \epsilon}.$$
4. **`Pseudo2dHeatmap`**: Renders intensity $S_m(\delta)$ as a 2D $(\delta \times \text{covariate}_m)$ raster / waterfall map across time, ligand equivalents, temperature, or pH.

### 3.2 Normalization & `icoshift` FFT Cross-Correlation Alignment
- **Normalization modes**: raw intensity, total-area ($\sum_j |S_m(\delta_j)|\,\Delta\delta = 1$), or reference-ROI peak height / integral normalization.
- **Regional FFT shift alignment** (Savorani et al. 2010): within a user-selected alignment interval $I = [j_0, j_1]$ of length $L$, zero-padded circular cross-correlation via radix-2 FFT
  $$C_m[\Delta] = \mathcal{F}^{-1}\!\Bigl(\mathcal{F}(\tilde{S}_{\text{ref}, I}) \odot \overline{\mathcal{F}(\tilde{S}_{m, I})}\Bigr)[\Delta], \qquad \Delta_m^* = \arg\max_{|\Delta| \le \Delta_{\max}} C_m[\Delta],$$
  co-aligns pH/temperature/ionic-strength peak drift in $O(L \log L)$ time while preserving peak areas.
- **Kinetics / Titration ROI Matrix Export**: integrates $R$ regions $\{\Omega_r\}_{r=1}^R$ across all $M$ traces to emit clean TSV tables (`trace_id`, `covariate`, `area_1..R`, `norm_ratio_1..R`).

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
     via projected active-set / NNLS on the augmented system $\begin{bmatrix} K \\ \sqrt{\lambda}\, L_2 \end{bmatrix} f \approx \begin{bmatrix} y \\ 0 \end{bmatrix}$, selecting $\lambda > 0$ via Generalized Cross-Validation (`GCV`) or L-curve corner curvature.
   - **Scientific Caveat**: Continuous real-valued Laplace inversion is exponentially ill-conditioned; unresolved chemical-shift overlap or low SNR can shift apparent `ILT` peaks toward intermediate "compromise" diffusion coefficients or split broad polydisperse modes into narrow pseudo-peaks. Discrete `BIC`-selected Stejskal–Tanner fitting is therefore the default quantitative mode, with `ILT` exposed as an opt-in continuous distribution view accompanied by explicit conditioning caveats.

### 4.3 Pseudo-2D $(\delta, \log_{10} D)$ Synthesis & Stokes–Einstein Hydrodynamics
- **Pseudo-2D `DOSY` Projection**: synthesizes $Z(\delta_i, \log_{10} D_q) = \sum_{k=1}^{K^*_i} I_{i,k} \exp\!\left(-\frac{(\log_{10} D_q - \log_{10} D_{i,k})^2}{2 \sigma_{\log_{10} D, i,k}^2}\right)$ aligned beneath the 1D top trace (`crates/nmr-gui/src/dosy_panel.rs`, `wasm32`-compatible).
- **Stokes–Einstein Hydrodynamic Radius & Power-Law Molecular Weight** (Evans et al. 2013):
  $$R_h = \frac{k_B T}{6 \pi \eta(T) D}, \qquad \widehat{\text{MW}} = \left(\frac{\alpha_{\text{solv}}}{D}\right)^{1 / \beta_{\text{solv}}},$$
  with built-in temperature-compensated dynamic viscosities $\eta(T)$ ($\text{Pa s}$) for $\text{D}_2\text{O}$, $\text{H}_2\text{O}$, $\text{CDCl}_3$, $\text{DMSO-}d_6$, $\text{CD}_3\text{OD}$, and $\text{CD}_3\text{CN}$.
