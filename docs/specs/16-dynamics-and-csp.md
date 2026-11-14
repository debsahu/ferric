# 16. Spin Relaxation, Model-Free Dynamics, Chemical Exchange & Ligand CSP

## Provenance & Clean-Room Statement
Original clean-room specification and mathematical design. Zero source code, binaries, decompiled output, residue libraries, macros, parameter files, or output files from any proprietary, academic-licensed, or copyleft (`GPL`/`AGPL`) dynamics, relaxation, or assignment software were consulted. All physical equations, numerical optimization methods, statistical criteria, and vendor file layouts are derived strictly from peer-reviewed literature and public format specifications:

1. **Backbone $^{15}\text{N}$ Spin Relaxation & Reduced Spectral Density Mapping**:
   - Kay, L. E., Torchia, D. A. & Bax, A. (*Biochemistry* **1989**, 28, 8972–8979) — $^{15}\text{N}$ $T_1, T_2$, and steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$ NOE equations and outlier-trimmed $R_2 / R_1$ correlation-time estimation.
   - Farrow, N. A., Muhandiram, R., Singer, A. U., Pascal, S. M., Kay, C. M., Gish, G., Shoelson, S. E., Pawson, T., Forman-Kay, J. D. & Kay, L. E. (*Biochemistry* **1994**, 33, 5984–6003) — quantitative backbone $^{15}\text{N}$ relaxation analysis and replicate noise estimation.
   - Farrow, N. A., Zhang, O., Szabo, A., Torchia, D. A. & Kay, L. E. (*J. Biomol. NMR* **1995**, 6, 153–162); Lefèvre, J.-F., Dayie, K. T., Peng, J. W. & Wagner, G. (*Biochemistry* **1996**, 35, 2674–2686) — reduced spectral density mapping $J(0), J(\omega_N), J(0.87\omega_H)$ under $\omega^{-2}$ high-frequency proton scaling.
2. **Lipari–Szabo / Clore / Palmer Model-Free Formalism & Rotational Diffusion**:
   - Lipari, G. & Szabo, A. (*J. Am. Chem. Soc.* **1982a**, 104, 4546–4559; **1982b**, 104, 4559–4570) — model-free generalized order parameter $S^2$ and effective internal correlation time $\tau_e$ (`M1`, `M2`).
   - Clore, G. M., Szabo, A., Bax, A., Kay, L. E., Driscoll, P. C. & Gronenborn, A. M. (*J. Am. Chem. Soc.* **1990**, 112, 4989–4991) — extended two-timescale model-free formalism (`M5`: $S^2 = S_f^2 S_s^2$, slow internal correlation time $\tau_s$).
   - Mandel, A. M., Akke, M. & Palmer, A. G. III (*J. Mol. Biol.* **1995**, 246, 144–163) — conformational exchange $R_{\text{ex}}$ models (`M3`, `M4`), parameter identifiability, parametric Monte Carlo bootstrap, and hierarchical $F$-test model selection.
   - Woessner, D. E. (*J. Chem. Phys.* **1962**, 37, 647–654); Tjandra, N., Feller, S. E., Pastor, R. W. & Bax, A. (*J. Am. Chem. Soc.* **1995**, 117, 12562–12566); Lee, L. K., Rance, M., Chazin, W. J. & Palmer, A. G. III (*J. Biomol. NMR* **1997**, 9, 287–298) — exact 3-component axially symmetric (prolate & oblate) and 5-component fully anisotropic rotational diffusion correlation functions.
   - d'Auvergne, E. J. & Gooley, P. R. (*J. Biomol. NMR* **2003**, 25, 25–39 & **2006**, 35, 117–135) — AIC/BIC model selection, boundary/identifiability model elimination, and global diffusion + local internal motion optimization.
3. **Slow-to-Fast $\mu\text{s–ms}$ Chemical Exchange & Ligand Titration Overview (§§3–4)**:
   - Vallurupalli, P., Bouvignies, G. & Kay, L. E. (*J. Am. Chem. Soc.* **2012**, 134, 8148–8161); Fawzi, N. L. et al. (*Nature* **2011**, 480, 268–272) — $^{15}\text{N}$ / $^{13}\text{C}$ Chemical Exchange Saturation Transfer (`CEST`) and Dark-State Exchange Saturation Transfer (`DEST`) homogeneous 2-state/3-state Bloch–McConnell propagation.
   - Carver, J. P. & Richards, R. E. (*J. Magn. Reson.* **1972**, 6, 89–105); Luz, Z. & Meiboom, S. (*J. Chem. Phys.* **1963**, 39, 366–370); Baldwin, A. J. (*J. Magn. Reson.* **2014**, 244, 114–124); Trott, O. & Palmer, A. G. III (*J. Magn. Reson.* **2002**, 154, 157–160) — Carr–Purcell–Meiboom–Gill (`CPMG`) relaxation dispersion and off-resonance $R_{1\rho}$.
   - Williamson, M. P. (*Prog. Nucl. Magn. Reson. Spectrosc.* **2013**, 73, 1–16) — chemical shift perturbation (`CSP`) weighting and quadratic tight-binding $K_d$ isotherms.
4. **Separable Nonlinear Optimization & Statistical Estimation**:
   - Levenberg, K. (*Quart. Appl. Math.* **1944**, 2, 164–168); Marquardt, D. W. (*J. Soc. Indust. Appl. Math.* **1963**, 11, 431–441); Golub, G. H. & Pereyra, V. (*SIAM J. Numer. Anal.* **1973**, 10, 413–432 — Variable Projection).

---

## 1. Pseudo-3D Relaxation Series Integration & Exponential Curve Fitting (`T13.1`, `T13.2`)

### 1.1 Acquisition Schedule Parsers & Pseudo-3D Demultiplexing (`crates/nmr-core/src/relax_io.rs`)
Pseudo-3D spin-relaxation ($R_1, R_2, R_{1\rho}$, steady-state $\{^1\text{H}\}\text{–}^{15}\text{N}$ hetNOE) and chemical-exchange (`CPMG`, `CEST`, `DEST`) experiments acquire a series of $P$ 2D planes indexed by an arrayed delay, spin-echo loop count, or saturation frequency offset (`RelaxSchedule`):

- **Bruker schedule files**:
  - `parse_bruker_vdlist`: parses variable relaxation delays (`VariableDelaysSec { delays_sec }`) with explicit unit suffixes (`s`, `ms`, `us`/`u`, `ns`; bare numbers default to seconds), stripping `#`, `;`, and `//` comments.
  - `parse_bruker_vclist(text, unit_cycle_sec)`: parses non-negative integer CPMG/spin-lock loop counts $n_{\text{cyc}} \in \mathbb{Z}_{\ge 0}$ into `LoopCounts { counts, unit_cycle_sec, values_sec }` where $T_p = n_{\text{cyc},p} \cdot \tau_{\text{unit}}$.
  - `parse_bruker_fqlist(text, obs_mhz, carrier_ppm)`: parses saturation offsets (`Hz`, `kHz`, `MHz`, or `ppm`, respecting JCAMP header directives `##$FQ1LIST` / `ppm` mode) into `FrequencyOffsets { offsets: Vec<FqOffset> }` storing paired `(offset_hz, offset_ppm)`.
- **Varian/Agilent `procpar` arrayed schedules (`parse_varian_relax_schedule`)**:
  - Extracts arrayed numeric vectors (`relaxT`, `d2`, `time_T2`, `tau`, `ncyc`, `satfrq`, `tof`, `dof2`) and deduces loop nesting (`Pseudo3DInterleaving`) directly from the VnmrJ `array` string:
    - `PlaneQueued` (`[plane][t1][quad][t2]`): sequential 2D planes.
    - `T1PlaneQuad` (`[t1][plane][quad][t2]`): `array = "relaxT,phase"`.
    - `T1QuadPlane` (`[t1][quad][plane][t2]`): `array = "phase,relaxT"` (and Bruker interleaved `ivd`/`ivc`/`ifq` inner loop).
- **Raw Pseudo-3D Decoders (`read_bruker_pseudo3d`, `read_varian_pseudo3d`) & Uniform Processing (`process_pseudo3d_fids`)**:
  - Demultiplexes raw Bruker `ser` (1024-byte block-aligned `int32`/`float32`/`float64`, `Dqd`/`States`/`StatesTppi`/`EchoAntiecho`, FIR `GRPDLY` circular shift) and Varian `fid` (`datafilehead` + `datablockhead` + `f1coef`) into `RawPseudo3D { planes: Vec<Fid2D>, schedule, ... }`.
  - `process_pseudo3d_fids` solves automatic phase parameters once on the highest-energy reference plane, locks explicit `(p0_deg, p1_deg)` with `PhaseSource::Explicit` in `ProcessingRecipe`, and applies identical apodization, zero-filling, phase rotation, and ROI bounds to all $P$ planes.

### 1.2 Shared/Locked 2D Footprint Linear Least-Squares Integrator (`integrate_relax_planes`)
Independent per-plane local-maximum peak picking (`extract_naive_local_max_series`) maximizes over noisy grid points in a local window, introducing severe positive order-statistic bias ($\mathbb{E}[\max_j (S_j + \varepsilon_j)] > \max_j S_j$, exceeding $+8\%$ at $\mathrm{SNR} = 4$) that flattens exponential decays and systematically underestimates $R_1$ and $R_2$.

`integrate_relax_planes` eliminates positive picking bias completely via a two-stage **locked-footprint linear least-squares** estimator:
1. **High-SNR Reference Composite & Connected-Component Footprint Fitting**:
   - Per-plane background noise $\hat{\sigma}_{\text{mad}, p}$ is estimated on each 2D plane via robust Median Absolute Deviation:
     $$\hat{\sigma}_{\text{mad}, p} = 1.4826 \cdot \operatorname{median}_{x, y} \bigl|S_p(x, y) - \operatorname{median}(S_p)\bigr|$$
   - On the high-SNR reference spectrum (`ref_plane_indices` inverse-variance composite or highest-SNR plane), target crosspeak seeds (`TargetPeakSeed`) are partitioned into connected overlap clusters via patch bounding-box union-find (`patch_half_width_pts`).
   - Each crosspeak $r$ is modeled by a unit-apex separable 2D Gaussian or equal-FWHM pseudo-Voigt profile (`FootprintShapeModel::{Gaussian, PseudoVoigt}`, $\Psi_r(\delta_H, \delta_X) = 1$):
     $$\Psi_r(x, y) = \psi_1(x - \delta_{H,r}; \sigma_{H,r}, \eta_{H,r}) \, \psi_1(y - \delta_{X,r}; \sigma_{X,r}, \eta_{X,r})$$
     $$\psi_1(\Delta; \sigma, \eta) = (1 - \eta) \exp\!\left(-\frac{\Delta^2}{2\sigma^2}\right) + \frac{\eta}{1 + \Delta^2 / (2 \ln 2 \, \sigma^2)}$$
   - Isolated singletons fit $(\delta_H, \delta_X, \sigma_H, \sigma_X [, \eta_H, \eta_X])$ via Variable-Projected Levenberg–Marquardt (`fit_footprint_on_reference`); overlapping multiplets (`members.len() >= 2`) jointly project out all cluster member amplitudes simultaneously (`refine_cluster_footprints_on_reference`) so neighboring peaks never inflate single-peak linewidths.
2. **Locked Linear Projection Across All Planes**:
   - With footprint shapes $\{\Psi_r\}$ locked across all $p \in \{0, \dots, P-1\}$, the design matrix $\mathbf{\Psi}_C \in \mathbb{R}^{N_{\text{pix}} \times K}$ over cluster $C$ (with optional constant local baseline column when `fit_local_baseline = true`) is inverted once via regularized Cholesky:
     $$\mathbf{W}_C = (\mathbf{\Psi}_C^\top \mathbf{\Psi}_C)^{-1} \mathbf{\Psi}_C^\top, \qquad \hat{A}_{r, p} = \sum_{u=1}^{N_{\text{pix}}} W_{r, u} S_p(x_u, y_u)$$
   - Because $\hat{A}_{r, p}$ is a fixed linear functional of $S_p$ independent of the noise realization in plane $p \notin \mathcal{R}_{\text{ref}}$, $\mathbb{E}[\hat{A}_{r, p}] = A_{r, p}^{\text{true}}$ is **unbiased down to arbitrarily low SNR** (`< 1.5%` empirical ensemble bias at `SNR = 4`).
3. **Spatial Noise Correlation & Duplicate-Delay Empirical-Bayes Variance Pooling**:
   - Baseline amplitude variance propagates white pixel variance $\hat{\sigma}_{\text{mad}, p}^2 \sum_u W_{r,u}^2$ (and optional stationary 2D FT noise autocorrelation $\hat{\rho}(\Delta x, \Delta y)$ when `account_spatial_noise_corr = true`).
   - When duplicate delays/offsets $|x_p - x_q| \le \text{duplicate\_tol}$ exist, replicate residuals are pooled across all $M$ peaks and duplicate groups $g$:
     $$\chi^2_{\text{dup}} = \sum_{g} \sum_{r=1}^{M} \sum_{p \in g} \frac{(\hat{A}_{r,p} - \bar{A}_{r,g})^2}{\sigma_{0,r,p}^2}, \qquad \nu_{\text{dup}} = M \sum_g (|g| - 1)$$
     shrinking toward unity with prior weight $\nu_0 = 8$:
     $$s_{\text{pool}}^2 = \frac{\nu_0 + \chi^2_{\text{dup}}}{\nu_0 + \nu_{\text{dup}}}, \qquad \sigma_{r, p} = s_{\text{pool}} \, \sigma_{0, r, p}$$

### 1.3 Relaxation-Series TSV & NEF Interchange
- `write_relax_series_tsv_str` / `parse_relax_series_tsv_str`: lossless round-trip tabular format carrying header metadata (`experiment_kind`, `schedule` encoding, `duplicate_dof`, `duplicate_variance_scale`), per-plane MAD noise, and per-(residue, plane) `(intensity, sigma, volume)` rows.
- `write_relax_series_nef_str` / `parse_relax_series_nef_str`: NEF v1.1 `save_nmrassign_relaxation_series_<name>` saveframe with `_nef_relaxation_list` header tags and `loop_ _nef_relaxation` records.

### 1.4 Batched CPU & WebGPU Exponential Curve Fitter (`relax_fit.rs`, `relax_gpu.rs`, `shaders/relax_mc.wgsl`)
Each extracted decay curve (`DecayCurveInput { seq_id, res_name, times_sec, intensities, sigmas }`) is normalized to dimensionless coordinates $u_k = t_k / t_{\max} \in [0, 1]$, $y_k = I_k / I_{\text{scale}} \in [-1, 1]$, $s_k = \sigma_k / I_{\text{scale}}$, and $w_k = s_k^{-2}$:

1. **2-Parameter & AIC/BIC-Gated 3-Parameter Models (`OffsetModelGate`)**:
   - Canonical 2-parameter decay: $I(t) = I_0 e^{-R t}$ (`OffsetModelGate::Disabled`).
   - Optional 3-parameter offset decay: $I(t) = I_0 e^{-R t} + C$, gated against the 2-parameter fit by `Aic` ($\chi^2 + 2k$), `Aicc`, `Bic` ($\chi^2 + k \ln n$), or `AlwaysThreeParam`.
   - Solved via coarse quadratic Variable-Projection grid scan + weighted log-linear starter + damped $2\times 2$ / $3\times 3$ Levenberg–Marquardt with exact linear-amplitude VarPro projection at every iteration, followed on `f32` WebGPU (`relax_mc.wgsl`) by Kahan-summed residual-form 1D VarPro Newton score polishing ($\phi(\tilde{R}) = \sum_k w_k u_k e^{-\tilde{R} u_k}(y_k - \hat{I}_0(\tilde{R}) e^{-\tilde{R} u_k}) = 0$) to eliminate $\sqrt{\epsilon_{\text{f32}}}$ zero-order $\chi^2$ rounding and achieve **`< 1e-5` CPU/WebGPU rate parity**.
2. **Analytical Cramér–Rao / Fisher Bounds & Parametric Monte Carlo Bootstrap**:
   - Exact $2 \times 2$ Fisher Information Matrix for $I(t) = I_0 e^{-R t}$ (`fisher_crlb_two_param`):
     $$\mathbf{F} = \begin{pmatrix} \sum_k \sigma_k^{-2} e^{-2 R t_k} & -I_0 \sum_k \sigma_k^{-2} t_k e^{-2 R t_k} \\ -I_0 \sum_k \sigma_k^{-2} t_k e^{-2 R t_k} & I_0^2 \sum_k \sigma_k^{-2} t_k^2 e^{-2 R t_k} \end{pmatrix}, \quad \sigma_{\text{Fisher}}(R) = \sqrt{(\mathbf{F}^{-1})_{11}}$$
   - Batched parametric Monte Carlo (`ExpFitConfig::mc_replicates`, e.g. `1,000+` replicates $\times$ `150+` residues) synthesizes $I_k^{(b)} = \hat{I}(t_k) + \varepsilon_k^{(b)}$ ($\varepsilon_k^{(b)} \sim \mathcal{N}(0, \sigma_k^2)$) using stateless counter-based 4-round bijective hashing + Box–Muller transforms in `relax_mc.wgsl`, reporting `sigma_i0_mc`, `sigma_rate_mc`, `rate_ci68`, and `rate_ci95`.
   - **WebGPU Safety (`relax_gpu.rs`)**: Pre-validates `max_buffer_size`, `max_storage_buffer_binding_size`, `max_uniform_buffer_binding_size`, and `max_compute_workgroups_per_dimension`; chunks dispatches exceeding `MAX_FITS_PER_DISPATCH = 262,144` for TDR safety; and propagates `device.poll` errors cleanly without silent suppression.

### 1.5 Steady-State $\{^1\text{H}\}\text{–}^{15}\text{N}$ hetNOE, Outlier-Trimmed $\tau_m$, & Reduced Spectral Density Mapping
- **Steady-State hetNOE (`compute_hetnoe`, `compute_hetnoe_batch`)**:
  $$\mathrm{NOE} = \frac{I_{\text{sat}}}{I_{\text{unsat}}}, \qquad \sigma_{\mathrm{NOE}} = \frac{\sqrt{\sigma_{\text{sat}}^2 + \mathrm{NOE}^2 \, \sigma_{\text{unsat}}^2}}{|I_{\text{unsat}}|}$$
  plus optional parametric Monte Carlo asymmetric quantiles (`noe_ci68`, `noe_ci95`) across both positive (`+0.8`) and disordered negative (`< 0`) steady-state enhancements.
- **Outlier-Trimmed $R_2 / R_1$ Global $\tau_m$ & Axial Diffusion Estimator (`estimate_trimmed_tau_m`, `estimate_initial_axial_diffusion_from_r2_r1`)**:
  - Following Kay, Torchia & Bax (*Biochemistry* 1989), residues with $\mathrm{hetNOE} < \text{min\_hetnoe}$ (`0.65`) are excluded first, followed by iterative `1.5 SD` rejection around mean $R_2 / R_1$ to remove $R_{\text{ex}}$-inflated outliers before exact monotonic bisection inversion of $\langle R_2 / R_1 \rangle \mapsto \tau_m$ (and multi-start + $4\times 4$ LM fitting of Woessner axial $(D_\parallel, D_\perp, \theta, \phi)$ when PDB $\text{N–H}$ vectors are supplied).
- **Reduced Spectral Density Mapping (`map_reduced_spectral_density`)**:
  - With signed $\gamma_N < 0$ ($r_{\gamma} = \gamma_N / \gamma_H \approx -0.101398$), $f_+ = 1 + r_\gamma \approx 0.89860$, $f_- = 1 - r_\gamma \approx 1.10140$, and $\omega^{-2}$ high-frequency scaling $6 J(\omega_H + \omega_N) - J(\omega_H - \omega_N) = (6 f_+^{-2} - f_-^{-2}) J(\omega_H) = 5 J(0.870\omega_H)$ (Farrow et al. *J. Biomol. NMR* 1995):
    $$\sigma_{\text{NH}} = (\mathrm{NOE} - 1) R_1 \frac{\gamma_N}{\gamma_H}, \qquad J(0.87\omega_H) = \frac{4 \sigma_{\text{NH}}}{5 d^2}$$
    $$J(\omega_N) = \frac{R_1 - \alpha_{R_1} J(0.87\omega_H)}{\tfrac{3}{4} d^2 + c^2}, \qquad J(0) = \frac{R_2 - (\tfrac{3}{8} d^2 + \tfrac{1}{2} c^2) J(\omega_N) - \alpha_{R_2} J(0.87\omega_H)}{\tfrac{1}{2} d^2 + \tfrac{2}{3} c^2}$$
    where $\alpha_{R_1} = \frac{d^2}{4} \frac{5 (6 f_+^{-2} + f_-^{-2})}{6 f_+^{-2} - f_-^{-2}}$ and $\alpha_{R_2} = \frac{d^2}{8} \frac{5 (6 + 6 f_+^{-2} + f_-^{-2})}{6 f_+^{-2} - f_-^{-2}}$, with exact analytical linear-Jacobian $1\sigma$ error propagation (`sigma_j_0`, `sigma_j_omega_n`, `sigma_j_087_omega_h`).

---

## 2. Lipari–Szabo / Palmer Model-Free & Anisotropic Diffusion Engine (`T13.3`, `crates/nmr-core/src/modelfree.rs`)

### 2.1 Dipolar + CSA Spin-Relaxation Equations & Physical Sign Conventions
Using CODATA 2018 constants (`RelaxPhysicsConfig`: $\gamma_H = +2.6752218744 \times 10^8\text{ rad s}^{-1}\text{ T}^{-1}$, signed $\gamma_N = -2.71261804 \times 10^7\text{ rad s}^{-1}\text{ T}^{-1} < 0$, default $r_{\text{NH}} = 1.02\text{ \AA}$, configurable $^{15}\text{N}$ CSA $\Delta\sigma = -172.0\text{ ppm}$):
$$d = -\frac{\mu_0 \hbar \gamma_H \gamma_N}{4\pi r_{\text{NH}}^3}, \qquad c(B_0) = \frac{\omega_N(B_0) \Delta\sigma}{\sqrt{3}}, \qquad \omega_N(B_0) = \frac{\gamma_N}{\gamma_H}\omega_H(B_0) < 0$$
At any static magnetic field $B_0$ (`field_mhz`), `predict_r1_r2_noe` evaluates:
$$R_1(B_0) = \frac{d^2}{4}\bigl[J(\omega_H - \omega_N) + 3 J(\omega_N) + 6 J(\omega_H + \omega_N)\bigr] + c(B_0)^2 J(\omega_N)$$
$$R_2(B_0) = \frac{d^2}{8}\bigl[4 J(0) + J(\omega_H - \omega_N) + 3 J(\omega_N) + 6 J(\omega_H) + 6 J(\omega_H + \omega_N)\bigr] + \frac{c(B_0)^2}{6}\bigl[4 J(0) + 3 J(\omega_N)\bigr] + R_{\text{ex}}(B_0)$$
$$\sigma_{\text{NH}}(B_0) = \frac{d^2}{4}\bigl[6 J(\omega_H + \omega_N) - J(\omega_H - \omega_N)\bigr], \qquad \mathrm{NOE}(B_0) = 1 + \frac{\gamma_H}{\gamma_N}\frac{\sigma_{\text{NH}}(B_0)}{R_1(B_0)}$$
where fast-limit conformational exchange scales quadratically with static field relative to `ref_field_mhz`:
$$R_{\text{ex}}(B_0) = R_{\text{ex,ref}} \left(\frac{\omega_N(B_0)}{\omega_N(B_{0,\text{ref}})}\right)^2 = R_{\text{ex,ref}} \left(\frac{B_0}{B_{0,\text{ref}}}\right)^2$$

### 2.2 Canonical Internal Motion Models (`ModelFreeId::{M1..M5}`)
Given global rotational diffusion components $\{(c_a, \tau_a)\}_{a=1}^A$ ($\sum_a c_a = 1$), `spectral_density_modelfree` evaluates:
$$J(\omega) = \frac{2}{5} \sum_{a=1}^{A} c_a \left[ \frac{S^2 \tau_a}{1 + (\omega \tau_a)^2} + \frac{(S_f^2 - S^2) \tau'_{a,e}}{1 + (\omega \tau'_{a,e})^2} \right], \qquad \frac{1}{\tau'_{a,e}} = \frac{1}{\tau_a} + \frac{1}{\tau_e}$$
spanning the five canonical Lipari–Szabo / Clore / Mandel–Akke–Palmer models (`LocalMfParams`):
- **`M1`** ($k = 1$): $(S^2)$, with $S_f^2 = 1$, $\tau_e = 0$, $R_{\text{ex,ref}} = 0$.
- **`M2`** ($k = 2$): $(S^2, \tau_e)$, with $S_f^2 = 1$, $R_{\text{ex,ref}} = 0$.
- **`M3`** ($k = 2$): $(S^2, R_{\text{ex,ref}})$, with $S_f^2 = 1$, $\tau_e = 0$.
- **`M4`** ($k = 3$): $(S^2, \tau_e, R_{\text{ex,ref}})$, with $S_f^2 = 1$.
- **`M5`** ($k = 3$, Clore et al. 1990): $(S_f^2, S_s^2, \tau_s)$ with generalized order parameter $S^2 = S_f^2 S_s^2$, fast libration $\tau_f \to 0$, slow internal amplitude $S_f^2 - S^2 = S_f^2(1 - S_s^2)$, and $R_{\text{ex,ref}} = 0$.

### 2.3 Rotational Diffusion Symmetries (`DiffusionTensor`)
Backbone amide $\text{N–H}$ unit bond vectors $\mathbf{u}_i \in \mathbb{S}^2$ are extracted from PDB coordinates (`extract_nh_unit_vectors_from_pdb` / `extract_nh_unit_vectors_from_chain`, excluding proline). Three global rotational diffusion models (`DiffusionTensorKind`, plus `Auto`) are supported:
1. **`Isotropic { tau_m_sec }`** ($A = 1$, 1 global parameter): $c_1 = 1$, $\tau_1 = \tau_m$.
2. **`AxialSymmetric { d_perp_s_inv, d_par_s_inv, theta_rad, phi_rad }`** ($A = 3$, 4 global parameters — both prolate $D_\parallel > D_\perp$ and oblate $D_\parallel < D_\perp$, Woessner 1962): with symmetry axis $\hat{\mathbf{d}} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$ and $\mu = \mathbf{u} \cdot \hat{\mathbf{d}}$:
   $$c_1 = \tfrac{1}{4}(3\mu^2 - 1)^2, \quad c_2 = 3\mu^2(1 - \mu^2), \quad c_3 = \tfrac{3}{4}(1 - \mu^2)^2$$
   $$\tau_1^{-1} = 6 D_\perp, \qquad \tau_2^{-1} = 5 D_\perp + D_\parallel, \qquad \tau_3^{-1} = 2 D_\perp + 4 D_\parallel$$
3. **`FullyAnisotropic { dx_s_inv, dy_s_inv, dz_s_inv, alpha_rad, beta_rad, gamma_rad }`** ($A = 5$, 6 global parameters — Woessner 1962; Lee et al. 1997): transforming $\mathbf{e} = R_{\text{ZYZ}}(\alpha, \beta, \gamma)\mathbf{u}$, with $D_s = \tfrac{1}{3}(D_x + D_y + D_z)$, $\Delta = \sqrt{D_x^2 + D_y^2 + D_z^2 - D_x D_y - D_x D_z - D_y D_z}$, and $\delta_k = (D_k - D_s)/\Delta$:
   $$\tau_1^{-1} = 4 D_x + D_y + D_z \; (c_1 = 3 e_y^2 e_z^2), \quad \tau_2^{-1} = D_x + 4 D_y + D_z \; (c_2 = 3 e_x^2 e_z^2), \quad \tau_3^{-1} = D_x + D_y + 4 D_z \; (c_3 = 3 e_x^2 e_y^2)$$
   $$\tau_4^{-1} = 6 D_s + 2\Delta, \quad \tau_5^{-1} = 6 D_s - 2\Delta, \quad c_4 = \tfrac{1}{2}(P - Q), \quad c_5 = \tfrac{1}{2}(P + Q)$$
   where $P = \tfrac{1}{2}[3(e_x^4 + e_y^4 + e_z^4) - 1]$ and $Q = \tfrac{1}{2}\sum_{k \in \{x,y,z\}} \delta_k (3 e_k^4 + 6 e_m^2 e_n^2 - 1)$, reducing continuously to both prolate and oblate axial Woessner expressions with $\sum_{a=1}^5 c_a \equiv 1$.

### 2.4 Model Elimination, Selection Criteria, & Global Alternating Optimization
- **Per-Residue Multi-Start Bounded LM (`fit_single_model_on_residue`)**: Optimizes $(S^2, \ln\tau_e, R_{\text{ex,ref}})$ or $(S_f^2, S_s^2, \ln\tau_s)$ from deterministic multi-start grids, followed by exact physical-coordinate Fisher covariance inversion (`local_fisher_uncertainties`).
- **d'Auvergne & Gooley (2003, 2006) + Mandel–Akke–Palmer (1995) Model Elimination (`enable_model_elimination`)**: Eliminates over-parameterized fits ($k > m_{\text{obs}}$), boundary-pinned timescales ($\tau_e < 2\text{ ps}$ or $> 0.48\tau_m$; $\tau_s < 150\text{ ps}$ or $> 0.52\tau_m$), negligible/non-physical amplitudes ($S^2 > 0.985$ on `M2`/`M4`; $R_{\text{ex,ref}} < 0.20\text{ s}^{-1}$ on `M3`/`M4`; $S_f^2 > 0.975$ or $S_s^2 > 0.975$ on `M5`), and unidentifiable extra parameters ($\sigma_{\text{Fisher}}(\theta) / \theta$ exceeding identifiability bounds) prior to statistical model ranking.
- **Model Selection (`ModelSelectionCriterion`)**: Supports `Aic` ($\chi^2 + 2k$), `Aicc`, `Bic` ($\chi^2 + k \ln m_{\text{obs}}$), and `PalmerFTest` (nested $\alpha = 0.05$ $\Delta\chi^2 > 3.841 / 5.991$ hierarchy).
- **Profiled Global Alternating Optimization (`fit_modelfree_dataset`)**: Initializes global diffusion from Kay–Torchia–Bax outlier-trimmed $R_2 / R_1$, then alternates multi-threaded per-residue `M1`–`M5` selection with global diffusion optimization where fast local parameters are warm-profiled inside `global_profiled_chi2` to decouple global $\tau_m / \mathbf{D}$ from local $S_i^2$. Exports tabular summaries via `write_modelfree_tsv_str`.

---

## 3. Slow-to-Intermediate Chemical Exchange Overview: `CEST`/`DEST` & `CPMG`/`R1rho` (`T13.4`, `T13.5`)

### 3.1 $^{15}\text{N}$ / $^{13}\text{C}$ `CEST` & `DEST` Bloch–McConnell Solver (`T13.4`)
For slow exchange ($k_{\text{ex}} = k_{AB} + k_{BA} \sim 10\text{–}500\text{ s}^{-1}$) with sparsely populated excited states ($p_B \sim 0.5\text{–}10\%$), saturation transfer intensity profiles $I(\omega_{\text{rf}}) / I_0$ acquired from pseudo-3D `fqlist` / `satfrq` stacks (§1.1) are modeled by homogeneous 2-state ($7 \times 7$: $[E/2, I_{xA}, I_{yA}, I_{zA}, I_{xB}, I_{yB}, I_{zB}]^\top$) and 3-state ($10 \times 10$) Bloch–McConnell evolution over spin-lock duration $T_{\text{EX}}$ under weak CW RF irradiation $\omega_1 = 2\pi \nu_1$ across multiple $B_1$ fields:
$$\frac{d\mathbf{M}}{dt} = (\mathbf{L}_0 + \mathbf{L}_{\text{rf}}(\omega_{\text{rf}}, \omega_1) + \mathbf{K})\,\mathbf{M}, \qquad \mathbf{M}(T_{\text{EX}}) = \exp\bigl((\mathbf{L}_0 + \mathbf{L}_{\text{rf}} + \mathbf{K})\,T_{\text{EX}}\bigr)\,\mathbf{M}(0)$$
evaluated via scaling-and-squaring `[6/6]` Padé matrix exponentials on CPU and batched WebGPU compute shaders to fit $(p_B, k_{\text{ex}}, \Delta\varpi_{AB}, R_{1A}, R_{2A}, R_{2B})$.

### 3.2 `CPMG` Relaxation Dispersion & Off-Resonance $R_{1\rho}$ (`T13.5`)
For intermediate-to-fast $\mu\text{s–ms}$ conformational exchange ($k_{\text{ex}} \sim 200\text{–}5{,}000\text{ s}^{-1}$) extracted from `vclist` / `ncyc` constant-time CPMG series ($R_{2,\text{eff}}(\nu_{\text{CPMG}}) = -\frac{1}{T_{\text{relax}}} \ln \frac{I(\nu_{\text{CPMG}})}{I_0}$ at $\nu_{\text{CPMG}} = \frac{1}{4\tau_{\text{cp}}}$):
- **Fast-exchange Luz–Meiboom (1963)**:
  $$R_{2,\text{eff}}(\nu_{\text{CPMG}}) = R_2^0 + \frac{\Phi_{\text{ex}}}{k_{\text{ex}}}\left[1 - \frac{4\nu_{\text{CPMG}}}{k_{\text{ex}}}\tanh\!\left(\frac{k_{\text{ex}}}{4\nu_{\text{CPMG}}}\right)\right], \qquad \Phi_{\text{ex}} = p_A p_B \Delta\omega^2$$
- **All-timescale Carver–Richards (1972) & Exact Baldwin (2014) Analytical Dispersion**: Exact analytical eigenvalue expression handling unequal intrinsic transverse relaxation rates ($R_{2A}^0 \ne R_{2B}^0$) and arbitrary $|\Delta\omega| / k_{\text{ex}}$ across dual static fields ($B_{0,1}, B_{0,2}$), gated against flat no-exchange $R_{2,\text{eff}} = R_2^0$ by AIC/BIC and $\Delta R_{2,\text{eff}} = R_{2,\text{eff}}(\nu_{\min}) - R_{2,\text{eff}}(\nu_{\max})$ significance.
- **Off-Resonance $R_{1\rho}$ (Trott & Palmer 2002)**:
  $$R_{1\rho}(\theta, \omega_e) = R_1 \cos^2\theta + \left(R_2^0 + \frac{p_A p_B \Delta\omega^2 k_{\text{ex}}}{\omega_{A,\text{eff}}^2 \omega_{B,\text{eff}}^2 / \omega_e^2 + k_{\text{ex}}^2}\right)\sin^2\theta$$

---

## 4. Ligand Titration Chemical Shift Perturbation (`CSP`) & Tight-Binding $K_d$ (`T13.6`)

Across a fast-exchange 2D $^{1}\text{H}\text{–}^{15}\text{N}$ (or $^{1}\text{H}\text{–}^{13}\text{C}$) HSQC ligand titration series at known total receptor $[P]_t$ and total ligand $[L]_t$ concentrations:
1. **Peak Trajectory Tracking & Williamson (2013) Nucleus-Weighted Euclidean CSP**:
   $$\Delta\delta_{\text{CSP}} = \sqrt{(\Delta\delta_H)^2 + (\alpha_X \Delta\delta_X)^2}$$
   with default gyromagnetic/chemical-shift range scaling $\alpha_N = 0.14$ (`0.20` for Glycine backbone amides; $\alpha_C = 0.30$ for methyl $^{13}\text{C}$).
2. **Exact Quadratic Stoichiometric Tight-Binding Isotherm (`1:1` Morrison / Williamson Equation)**:
   $$\Delta\delta([P]_t, [L]_t) = \Delta\delta_{\max} \frac{([P]_t + [L]_t + K_d) - \sqrt{([P]_t + [L]_t + K_d)^2 - 4 [P]_t [L]_t}}{2 [P]_t}$$
   supporting both per-residue and global shared-$K_d$ Variable-Projected Levenberg–Marquardt fitting across active binding-pocket residues ($\Delta\delta_{\text{CSP}} > \bar{\Delta\delta} + 2\sigma_{\text{trimmed}}$), Monte Carlo confidence intervals on $K_d$, and PDB B-factor / JSON surface mapping for pocket visualization.
