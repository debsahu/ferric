# Chapter 2: Custom Pulse-Program Physics & Multidimensional DSP

Real experimental NMR repositories rarely conform to textbook defaults: laboratories rename pulse programs (`custom_tr_seq_v2`, `my_t1_trosy_interleaved`), synthesize manual `States-TPPI` phase loops inside `FnMODE = QF`, transpose `F1`/`F2` acquisition loops (`AQSEQ = 312`), interleave relaxation delays inside $t_1$, write 0-based or 1-based `nuslist` schedules, or leave unprocessed indirect `REVERSE` flags unset. `nmr-assign` resolves these ambiguities directly from acquisition physics (`nmr-core::pulprog`, `fid`, `process`, `ops`, and `nmr1d`).

---

## Sandboxed Executed `pulseprogram` & `procpar` Interpreter

Rather than guessing experiment behavior from filename strings or external spectrometer library directories, `nmr-core::pulprog` parses the **executed `pulseprogram`** (or `pulseprogram.precomp`) archived directly inside each Bruker experiment directory alongside `acqus`/`acqu2s`/`acqu3s`/`acqu4s`, and the full ASCII **`procpar`** dictionary inside Varian/Agilent `.fid` directories.

### Conditional Preprocessor & Symlink Containment

1. **Exact `ZGOPTNS` macro evaluation**: Evaluates `#define`, `#undef`, `#ifdef`, `#ifndef`, `#if`, `#elif`, `#else`, and `#endif` branches against the exact `-D...` and `-U...` compiler flags recorded in `acqus` `##$ZGOPTNS` (for example `-DLABEL_CN -DTROSY -DDEUTERATED`), stripping C-style `/* ... */` and Bruker `;` comments while preserving quoted `"d0=..."` runtime delay equations.
2. **Fail-closed `#include` sandboxing**: Experiment-local `#include "..."` files are canonicalized via `validate_local_include_path` strictly inside the experiment root—rejecting `..` parent traversal, absolute paths, symlink escapes outside `<exp_dir>`, cyclic includes (`MAX_INCLUDE_DEPTH = 8`), and diamond DAG expansion bombs (`MAX_TOTAL_INCLUDES = 64`, `MAX_PREPROCESSED_LINES = 65,536`).
3. **Active Magnetization-Transfer Graph (`MagnetizationTransferGraph`)**: Tracks active RF channels (`1H`, `15N`, `13CA`, `13CB`, `13CO`), carrier jumps (`fq=cnst*`), shaped pulses (`SPNAM`), composite decoupling (`CPDPRG`), INEPT / constant-time / TOCSY / NOESY / Stejskal–Tanner gradient delays, and TROSY selection—classifying renamed laboratory pulse sequences accurately in `< 1 ms`.

---

## Quadrature Detection: States-TPPI vs Echo-Antiecho

Every indirect dimension is demultiplexed on load into vendor-neutral hypercomplex `(cos, sin)` FID pairs on the full Nyquist grid (`Fid2D`, `Fid3D`) before any NUS reconstruction or Fourier transformation runs.

### Bruker Indirect Quadrature (`FnMODE` & Manual Phase Loops)

| Mode | Bruker `FnMODE` | Physical Acquisition & Clean-Room Recombination |
| :--- | :---: | :--- |
| **`QF`** | `1` | Single-channel / magnitude mode (e.g., `DOSY` gradient arrays); `sin` channel is zero (`states_recombined = false`; rejected if passed to hypercomplex `IST`/`FISTA`). |
| **`States`** | `4` | Paired cosine/sine increments: $S_{\cos}(t_1, t_2) = S_{2k}(t_2)$, $S_{\sin}(t_1, t_2) = S_{2k+1}(t_2)$ (States, Haberkorn & Ruben 1982). |
| **`States-TPPI`** | `5` | $180^\circ$ receiver/pulse alternation on odd increments shifting axial peaks to the Nyquist edge: $S_{\cos}(t_1, t_2) = (-1)^k S_{2k}(t_2)$, $S_{\sin}(t_1, t_2) = (-1)^k S_{2k+1}(t_2)$ at true 0-based grid coordinate $k$ (Marion et al. 1989). |
| **`Echo-Antiecho`** | `6` | Sensitivity-enhanced P/N-type gradient coherence selection (`Rance–Kay`; Palmer et al. 1991; Kay, Keifer & Saarinen 1992): recombined into **two separate hypercomplex direct-FID channels** $(S_{\cos}, S_{\sin})$ prior to direct FFT: |

$$S_{\cos}(t_1, t_2) = \frac{S_+(t_1, t_2) + S_-(t_1, t_2)}{2}, \qquad S_{\sin}(t_1, t_2) = \frac{S_+(t_1, t_2) - S_-(t_1, t_2)}{2i}$$

> [!WARNING] Why Scalar `S_cos + i S_sin` Addition Fails on Echo-Antiecho
> Naively forming the scalar sum $S_{\cos} + i S_{\sin}$ collapses algebraically back to $S_+(t_1, t_2)$, discarding $S_-$ and destroying indirect frequency sign discrimination! `nmr-core::fid` maintains $(S_{\cos}, S_{\sin})$ as distinct complex direct traces (`out_cos = [-(im0 + im1), re0 + re1]`, `out_sin = [re1 - re0, im1 - im0]` under Bruker's conjugate direct-FT convention) and verifies Rance–Kay encoding/decoding gradient polarity directly from `GPZ` ratios and bipolar `*EA ... 180° ... *-1*EA` pairs in `pulseprogram`.

**Custom & Legacy Fallbacks**:
- **Manual quadrature under `FnMODE = 0` or `1 (QF)`**: When a custom `pulseprogram` increments phase registers (`ip*`, `dp*`) or gradient polarity (`igrad`) inside a `lo to <lbl> times 2` loop while leaving `acqus` `FnMODE = QF`, `nmr-core::pulprog` detects the synthesized `States`, `States-TPPI`, or `Echo-Antiecho` loop automatically.
- **Legacy `MC2` headers**: Older acquisitions lacking `acqu2s` `FnMODE` fall back safely to `proc2s` `##$MC2` and executed-`pulseprogram` `mc2` directives.

### Varian / Agilent `procpar` Quadrature & `origfid` Guard

1. **8-Float Hypercomplex Combination (`fcoef`, `f1coef`, `f2coef`)**: Applies the exact 8-coefficient linear transformation $(c_0 \dots c_7)$ to paired `phase = 1, 2` traces:

$$z_{\cos} = (c_0 - i c_1) z_1 + (c_2 - i c_3) z_2, \qquad z_{\sin} = (c_4 - i c_5) z_1 + (c_6 - i c_7) z_2$$

In 3D datasets (`ni2 > 1`), `f1coef` takes strict precedence over stale 2D `fcoef` template strings, and `conjugate_for_fft` adapts automatically between standard `1 0 0 0 0 0 -1 0`, constant-time `1 0 0 0 0 0 1 0`, and Rance–Kay cross-term `1 0 -1 0 0 -1 0 -1` matrices.
2. **Untouched `origfid` recovery**: Some post-acquisition scripts overwrite Varian `fid` with pre-recombined traces without updating `procpar` `f1coef` or `array`. Whenever an untouched raw `origfid` of matching byte length accompanies `fid`, `nmr-core::fid` reads `origfid` to prevent double Rance–Kay recombination.

### Initial Evolution Delay $t_1(0)$, Half-Dwell `PHC1` & Folding

By comparing the effective initial indirect delay $t_1(0) = d_0 + \frac{4 p_{90}}{\pi}$ (or Varian `f1180 = 'y'`) against half a dwell $\frac{\Delta t}{2} = \frac{1}{2\,\text{SW}_{\text{Hz}}}$ (Marion et al. 1989; Otting et al. 1986), the interpreter predicts:
- **Zero-delay sampling ($t_1(0) \approx 0$)**: `(PHC0, PHC1) = (0°, 0°)` with first-point scaling `FCOR = 0.5` to eliminate baseline DC offset.
- **Half-dwell sampling ($t_1(0) = \Delta t / 2$)**: `(PHC0, PHC1) = (-90°, +180°)` (or `(+90°, -180°)`) with `FCOR = 1.0`.
- **Aliasing physics**: Distinguishes cyclical `States` wrap-around (`SameSignWrap`) from Nyquist mirror-inverted `States-TPPI` (`MirrorInvertedWrap`) and constant-time mirror folding (`ConstantTimeMirror`).

---

## Loop Ordering: Swapped `AQSEQ`, Pseudo-3D & NUS Schedules

### Swapped 3D `AQSEQ` (`321` vs `312`) & 1024-Byte Block Padding

In Bruker 2D/3D `ser` files, every acquired 1D trace of `TD` scalar words is padded to a **1024-byte (`256`-word) boundary**:

$$\text{stride\_bytes} = \left\lceil \frac{\text{TD} \cdot \text{word\_bytes}}{1024} \right\rceil \times 1024$$

In 3D experiments, loop nesting between `F2` (`acqu2s`) and `F1` (`acqu3s`) is governed by `AQSEQ`:
- **`AQSEQ = 0` (`3-2-1`)**: `F2` (`15N` or inner axis) increments inside the outer `F1` loop.
- **`AQSEQ = 1` (`3-1-2`)**: `F1` increments inside the outer `F2` loop.

`nmr-core::pulprog` resolves `AQSEQ` not only from `##$AQSEQ`, but also from explicit `aqseq 312` pulse-program directives and manual `lo to` `id0`-inside-`id10` delay increment loops when `acqus` retains a default `0`, transposing both uniform `ser` planes and 2-column `nuslist` coordinates consistently.

### Plane-Queued vs $t_1$-Interleaved Pseudo-3D Series

For spin-relaxation ($R_1, R_2$, hetNOE), `CEST`, `CPMG`, and `DOSY` datasets, `pulprog` inspects loop topology (`vdlist`, `vclist`, `fqlist`, `difflist`, or Varian `array` strings such as `"relaxT,d2,phase"` vs `"d2,relaxT,phase"`) to separate:
- **`PlaneQueued`**: Full 2D `(t1, t2)` plane acquired for delay `m = 0`, followed by `m = 1`, etc.
- **`T1Interleaved`**: Delay list (`ivd`/`ivc`/`ifq`) stepped inside each $t_1$ increment (either inside or outside the `times 2` quadrature loop, tracked via `pseudo_3d_list_inner_to_quad`), eliminating temperature or spectrometer drift biases across long relaxation runs.

### 0-Based vs 1-Based NUS Schedules

Both Bruker (`FnTYPE = 2`, `NusTD`, `nuslist`) and Varian (`sampling = 'sparse'`, `sampling.sch`) NUS schedules are decoded automatically:
- Uniform TopSpin 4 files that write `FnTYPE = 0` and `NUSLIST = <automatic>` with `NusTD == TD` are guarded against false NUS triggering.
- `decode_nuslist_str` normalizes both **0-based** (`0 .. n_cplx - 1`) and **1-based** (`1 .. n_cplx`) integer grid schedules, averages repeated weighted-NUS coordinates, and rejects out-of-bounds or trace-mismatched schedules with loud errors.

---

## Empirical `{sign, fold}` Verification Against `2D HSQC`

Even after header and `pulseprogram` decoding, archived public depositions occasionally contain mis-recorded `REVERSE` flags or post-acquisition script edits that flip indirect axis orientation or shift a 3D cube by half a spectral width. At Step 0 of 3D registration (`register_3d_hn_to_hsqc_2d_with_audit`), `nmr-assign` evaluates all $4 \times 4 = 16$ Klein four-group ($\mathbb{Z}_2 \times \mathbb{Z}_2$) indirect-axis hypotheses $(\text{op}_N, \text{op}_C) \in \{\text{Canonical}, \text{SignFlip}, \text{HalfSwShift}, \text{MirrorFold}\}^2$:

1. **`15N` Root-Strip Likelihood $\ln L_N(\text{op}_N)$**: Measures max-abs `(1H, 15N)` sky-projection SNR concentration at picked `2D 15N-HSQC` roots (allowing small cross-experiment carrier drift $(\delta_H, \delta_N) \in [-0.12, +0.12] \times [-0.60, +0.60]\text{ ppm}$).
2. **`13C` Band & Polarity Prior $\ln L_C(\text{op}_C \mid \text{op}_N)$**: Scores 1D `13C` strip extrema against BMRB carbonyl (`175.8 ± 2.35 ppm`), `CA`-only (`41.5..68.0 ppm`), or biphasic `HNCACB` (`CA > 0`, `CB < 0`) envelopes.
3. **Dual Bayesian + Physical-Violation Gate**: Because dense amide projections can produce large raw log-Bayes factors on already-valid spectra, overriding `Canonical` requires **both** $\Delta \ln P \ge 12.0\text{ nats}$ **and** strict physical-envelope violation gates (`>= 5` matched roots, `matched_hsqc_fraction >= 0.40` with `>= +0.20` improvement for `op_N != Canonical`, and genuine out-of-envelope / Nyquist-pileup evidence for `op_C != Canonical`). Across a post-hoc audit of `79` real 3D BMRB cubes (`24` cold + `55` dev), `0/79` false rescues fired while `10/10` corrupted cubes and a mis-flagged experimental `CISD3` `HNCO` (`10.26% -> 94.87%`) were rescued cleanly.
4. **Full Audit Trail**: Any applied rescue emits `warning: [axis-rescue]` on `stderr`, a banner in `nmr-gui`, and an `AxisRescueAuditRecord` in `processing_recipe.json`, `.nef`, and `report.html` (disable with `--no-axis-rescue`).

---

## Hypercomplex `IST` / `FISTA` NUS & Multidimensional DSP

### The 9-Step Deterministic Processing Pipeline (`nmr-core::process`)

```bash
# Reconstruct a 3D NUS Bruker directory into NMRPipe .ft3 + processing_recipe.json
nmrassign process \
  --in ./data/cisd3/cbcaconh_nus \
  --out ./out/cbcaconh.ft3 \
  --exp CBCACONH \
  --nus-method fista \
  --crop-h 5.5:11.5
```

Executed strictly in order per dataset:
1. **Digital Filter Group-Delay Removal**: Shifts direct complex samples by $m = \operatorname{round}(G)$ (`GRPDLY` or Westler/Moskau `(DSPFVS, DECIM)` table) and applies exact sub-sample linear phase $\Delta p_1 = -360^\circ (G - m)$.
2. **Optional Time-Domain Solvent Convolution**: Gaussian low-pass subtraction (Marion, Ikura & Bax 1989) suppressing residual `H2O` ridges.
3. **Hypercomplex `IST` / `FISTA` NUS Reconstruction**:
   - Enforces `states_recombined == true` (`Err` on `QF`) so every cross-peak is a single sparse complex (`Fid2D`) or bicomplex quaternion (`Fid3D`) exponential.
   - Solves the $L_1$-regularized inverse problem across amide `1H` columns in parallel (`std::thread::scope`):

$$\min_{X} \; \frac{1}{2}\bigl\| \mathcal{P}_\Omega \mathcal{F}^{-1}(X) - Y_{\text{obs}} \bigr\|_2^2 + \lambda \|X\|_1$$

   - **Dynamic Residual $\sigma_{\text{MAD}}$ Thresholding**: Estimates $\hat{\sigma}_{\text{res}}^{(t)} = 1.4826\,\operatorname{MAD}(\|G^{(t)}\|_2)$ from the **current frequency residual** $G^{(t)}$ (rather than the initial PSF-sidelobe-contaminated dirty spectrum), soft-thresholding at $\tau_t = \max(\tau_{\text{sched}}^{(t)}, 2.5\hat{\sigma}_{\text{res}}^{(t)})$ followed by active-set debiasing and exact data consistency $Z[k] = Y_{\text{obs}}[k]$ on $k \in \Omega$ (Stern, Donoho & Hoch 2007; Beck & Teboulle 2009; Hyberts et al. 2012).
4. **Apodization & Opt-In Mirror-Image Linear Prediction (`--lp`)**: Sine-bell, squared sine-bell, exponential, Lorentz-to-Gauss, trapezoid, or `J`-modulation windows (`nmr-core::ops`). Indirect Burg/SVD mirror-image LP (Zhu & Bax 1990) is **`false` by default** to avoid autoregressive bandwidth inflation unless `--lp` is explicitly requested.
5. **Zero-Filling & Radix-2 / Bluestein FFT**: Exact Cooley–Tukey radix-2 or Bluestein arbitrary-$N$ `f64` FFT with Nyquist shift and `REVERSE` handling.
6. **Stored or Blind 1D/2D/3D Autophasing**: Uses calibrated `procs`/`procpar` phases when present, otherwise runs Chen (2002) derivative-entropy + Kramers–Kronig dispersion minimization with capped-SNR global polarity locking (`CA(+)` in `48..60 ppm` vs `CB(-)` in `14..38 ppm` on `HNCACB`, safely excluding `Ser`/`Thr` `CB` at `60–66 ppm`).
7. **Imaginary Discard, Polynomial/Median Baseline & ROI Crop**: Halves memory per transformed axis, flattens Legendre or Friedrichs median baselines, and crops direct `1H` (`5.5..11.5 ppm` default; `--no-crop` for full sweep).

> [!NOTE] Truncated Acquisition Safety (`--allow-truncated-fid`)
> Aborted acquisitions whose `ser`/`fid` file is shorter than the header grid fail closed by default. Passing `--allow-truncated-fid` zero-fills missing trailing blocks, emits a prominent `stderr` warning, and forces `DatasetQcVerdict::Warn` (`truncated_fid_zero_filled = true`).

---

## Analytical 1D Autophasing, Whittaker Baseline & `HSVD`

For high-resolution small-molecule and biomolecular 1D NMR (`1H`, `13C{1H}`, `DEPT-90`, `DEPT-135`, `19F`, `31P`), `nmr-core::nmr1d` (`nmrassign nmr1d process`) provides dedicated analytical DSP routines:

> [!NOTE] 1D DSP Validation Scope (Synthetic Forward-Model Self-Consistency)
> Quantitative accuracy numbers reported below (`< 2°` `PHC0`, `< 5°` `PHC1`, `< 2.5%` `HSVD` solute integral bias at `35 Hz` offset) come from synthetic forward-model 1D benchmarks (`T14.1`); evaluation on real deposited raw small-molecule 1D Bruker/Varian `fid` acquisitions is tracked openly as follow-up **`F14`**.

### 1. Pivot & Minimum-Entropy 1D Autophasing (`auto_phase_1d`)

Phases around any configurable `Pivot1D` (`MaxPeak`, `Center`, `Ppm(f64)`, `Index(usize)`) by minimizing first-derivative Shannon entropy $H = -\sum_i h_i \ln h_i$ ($h_i = |R'_i| / \sum_j |R'_j|$) plus a Kramers–Kronig dispersion-antisymmetry penalty and negative-lobe penalty (Chen et al. 2002; de Brouwer 2009):
- **`AllPositive` mode**: Standard absorptive `1H`, `13C{1H}`, `DEPT-90`, `19F`, `31P` spectra (`< 2°` `PHC0`, `< 5°` `PHC1` error).
- **`Biphasic` mode**: Preserves positive $\text{CH}/\text{CH}_3$ and inverted negative $\text{CH}_2$ multiplets on `DEPT-135` without penalizing genuine negative resonances.
- **Real-only spectrum recovery**: If imaginary data were discarded upstream, `reconstruct_imaginary_hilbert` rebuilds the causal dispersion channel via discrete Hilbert transform (Bartholdi & Ernst 1973) before phasing.

### 2. Whittaker Penalized-Least-Squares & Bernstein Baselines

1. **Continuous-Wavelet Signal Mask (`detect_baseline_mask_1d`)**: Convolves the spectrum with a Ricker Mexican-hat wavelet to classify signal-free baseline bins $w_i \in \{0, 1\}$.
2. **Exact $O(N)$ Whittaker Smoother (`BaselineMethod1D::Whittaker` & `AsymmetricWhittaker`)**: Solves the banded pentadiagonal Cholesky system (Eilers 2003; Cobas et al. 2006):

$$\min_{\mathbf{b}} \; \sum_{i=0}^{N-1} w_i (y_i - b_i)^2 + \lambda \sum_{i=1}^{N-2} (b_{i+1} - 2b_i + b_{i-1})^2$$

with optional iteratively reweighted asymmetric weights penalizing baseline excursion above noise troughs, or low-degree **Bernstein** polynomial basis fitting (`BaselineMethod1D::Bernstein`) for broad rolling probe backgrounds.

### 3. Time-Domain `HSVD` Solvent Pole Subtraction

When a `500×` residual `H2O`/`HDO` peak obscures nearby $\text{H}^\alpha$ or small-molecule solute multiplets, `subtract_solvent_hsvd_1d` models the FID as a sum of damped complex exponentials via **FFT-accelerated Hankel–Lanczos bidiagonalization** and QR pole extraction (Barkhuijsen et al. 1987; Pijnappel et al. 1992):

$$z[n] \approx \sum_{k=1}^{K} c_k \exp\bigl((-\alpha_k + i 2\pi f_k) n \Delta t + i \phi_k\bigr)$$

Subtracting strictly the poles $(f_k, \alpha_k)$ falling inside the solvent frequency window preserves solute peaks just `35 Hz` (`< 0.06 ppm`) away to within `< 2.5%` integral area—avoiding the broad baseline distortions of frequency-domain zeroing (with $C^1$-continuous cubic-Hermite `suppress_solvent_notch_1d` available when only real frequency data exist).
