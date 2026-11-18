# Chapter 7: Clean-Room Provenance, Open Data Policy & Bibliography

FERRIC is engineered from the ground up as a **100% clean-room, permissively licensed (`MIT`) scientific platform**. Every mathematical model, binary spectrometer reader, statistical calibrator, GPU shader, and graphical studio traces directly to peer-reviewed literature, public open specifications, and public-domain (`CC0 1.0`) archival data.

> [!TIP] Interactive Tour & HD Video Showcase
> Every panel in the **[Interactive Showcase Cinema Gallery](tours/index.html)** is generated deterministically from clean-room `nmr-core` outputs (`ferric export-tour`, zero external CDN dependencies) with explicit per-module validation scope badges.

---

## Clean-Room Engineering Rules

### Permitted Sources

1. **Peer-Reviewed Scientific Literature**: Published journal articles, textbooks, and their supplementary mathematical derivations.
2. **Public Format & Protocol Specifications**: Published Bruker JCAMP-DX v5.0 and Varian/Agilent binary header manuals, IUPAC-IUBMB-IUPAB NMR nomenclature recommendations, `NMR-STAR v3.2`, NMR Exchange Format (`NEF v1.1`), wwPDB `.pdb`/`.mmcif`, `MOL`/`SDF` V2000/V3000, `JCAMP-DX`, RFC 2083/8259/9110, and the open Model Context Protocol (`MCP`) specification.
3. **Public-Domain Archival Data (`CC0 1.0`)**: Biological Magnetic Resonance Data Bank (`BMRB`), Worldwide Protein Data Bank (`wwPDB`), and the PDB Chemical Component Dictionary (`CCD`).
4. **Allowlisted Permissive Open-Source Code**: Software distributed strictly under `MIT`, `BSD-2-Clause`, `BSD-3-Clause`, `Apache-2.0`, `ISC`, or `Zlib` licenses, documented in `THIRD_PARTY.md`.

### Strictly Prohibited Sources

- **Zero closed, academic-restricted, or commercial artifacts**: No source code, compiled binaries, decompiled output, residue topology libraries, pulse-program macros, parameter tables, keytables, or output templates from any commercial or academic-licensed NMR processing, resonance assignment, structure calculation, torsion prediction, or relaxation dispersion software package were ever inspected or incorporated.
- **Zero copyleft (`GPL` / `AGPL`) code inspection**: Algorithms originally published alongside `GPL` or `AGPL` reference tools are implemented strictly from their peer-reviewed papers and mathematical equations without reading copyleft source code.
- **Provenance module headers**: Every algorithmic module in `crates/nmr-core/` and `crates/nmr-gpu/` begins with an explicit `Provenance:` rustdoc header citing the exact paper(s) and equations implemented or recording original clean-room methods.

---

## Scientific Governance & Benchmark Integrity

To prevent data leakage, circular self-validation, or inflated accuracy claims, FERRIC enforces four structural invariants in code and CI:

1. **Sequence-Only Default Backbone Assignment**: Default `ferric assign` uses only the 1D amino-acid sequence, experimental peak tables/spectra, and fixed BMRB chemical-shift priors (`data/priors/bmrb_priors.json`, derived from `36` public BMRB entries disjoint from all benchmark manifests); 3D structural shift and `PAE`-gated `15N-NOESY` priors (`--af-structure`) are strictly opt-in ([Chapter 9](09-alphafold-and-integrative-nmr.md)).
2. **Unlabelled Reference-Offset Estimation**: Global per-nucleus chemical-shift offsets (`estimate_reference_offsets_detailed`, `estimate_af_shift_offsets`, and `ferric score`) are estimated strictly without ground-truth assignment labels (`< 1e-12` permutation-invariant).
3. **4-Layer `HomologFirewall` (`nmr-core::torsion_db`, `nmr-core::struct_shifts`)**: Both post-assignment 3/9-mer $(\phi, \psi, \chi_1)$ prediction and 3D-to-chemical-shift Fourier hypersurface fitting (`torsion_corpus.tsv`) exclude homologous templates across `L1` (PDB/BMRB/UniProt/family IDs), `L2` (CATH/ECOD fold topology), `L3` (global Needleman–Wunsch & local Smith–Waterman `<= 25%` sequence identity at `>= 50%` coverage under both `BLOSUM62` and exact-identity scoring + `6/7` k-mer filter), and `L4` (Zhang–Skolnick `TM-score < 0.50`), emitting a deterministic `exclusion_list_sha256` audit digest.
4. **Pre-Registered Single-Shot Cold Cohort Rotation**: Real-spectrum BMRB entries (`data/benchmark/real_manifest.tsv`) are partitioned into development/seen cohorts and pre-registered unseen (`cold_*`) cohorts. Each cold cohort is evaluated **once** after code freeze, its exact empirical accuracy and high-confidence precision are recorded verbatim ([Chapter 3](03-solvers-and-calibration.md#honest-unseen-protein-cold-cohort-empirical-precision-caveats)), and spent cold cohorts are immediately relabeled `seen_*` so they are never reused as unseen evidence.

---

## Open `CC0 1.0` Data Policy & Private Data Isolation

- **BMRB (`https://bmrb.io/`), UniProt & CATH**: Empirical amino-acid backbone and sidechain priors (`data/priors/PROVENANCE.md`), torsion corpus shifts/coordinates (`data/torsion_db/PROVENANCE.md`, `SHA-256` verified via `scripts/build_torsion_corpus.py --verify`; UniProt accession mappings and CATH fold annotations attributed under `CC-BY-4.0`, with `4` `bmrb_priors` overlap entries caught by `HomologFirewall`), and public benchmark manifests (`data/benchmark/PROVENANCE.md`) derive from `CC0 1.0 Universal` BMRB/wwPDB depositions and `CC-BY-4.0` metadata, with every accession ID and SHA-256 digest recorded.
- **wwPDB, ModelArchive & AlphaFold Database (`https://www.wwpdb.org/`, `https://alphafold.ebi.ac.uk/`)**: Reference macromolecular coordinates (`data/pdb_refs/`) and standard 20-amino-acid covalent topologies derive from `CC0 1.0` wwPDB entries (`.licenses/pdb_database_LICENSE.txt`); opt-in CLI downloads via `ferric af-validate --fetch-uniprot` record `CC-BY-4.0` attribution and `FIPS 180-4` `SHA-256` digests in `PROVENANCE.md`.
- **Unpublished Experimental Data (`data/private/`)**: Unpublished laboratory datasets remain strictly inside git-ignored `data/private/` and never enter public commits or fixtures.

---

## Permissive Licensing (`MIT` & `Apache-2.0`)

FERRIC is distributed under the **MIT License** (`LICENSE`). All `327` transitive packages in `Cargo.lock` are audited via `cargo metadata` against the workspace allowlist (`MIT`, `BSD-2-Clause`, `BSD-3-Clause`, `Apache-2.0`, `ISC`, `Zlib`; zero `GPL` or `AGPL` crates):

| Direct Dependency | SPDX License | Role in Workspace |
| :--- | :--- | :--- |
| `serde`, `serde_json` | `MIT OR Apache-2.0` | JSON serialization for priors, `ProcessingRecipe`, `.oct`, and `MCP` JSON-RPC 2.0 |
| `bytemuck` | `Zlib OR Apache-2.0 OR MIT` | Safe zero-copy POD struct casting for GPU uniform & storage buffers |
| `pollster` | `Apache-2.0 OR MIT` | Minimal single-threaded executor for headless `wgpu` initialization |
| `wgpu` | `MIT OR Apache-2.0` | Cross-platform WebGPU / Metal / Vulkan / DX12 compute & 3D rendering |
| `eframe` (`egui`) | `MIT OR Apache-2.0` | Immediate-mode native (`winit`) & browser (`WebRunner`) GUI workbench |
| `wasm-bindgen`, `wasm-bindgen-futures`, `web-sys` | `MIT OR Apache-2.0` | Browser WebAssembly & WebGPU DOM bindings (`wasm32-unknown-unknown`) |

**Adapted Open-Source Attribution**:
- **Octet-Draw** (`https://github.com/SamChemMan/Octet-Draw`, Copyright (c) 2026 Sam Ellis, **`Apache-2.0`**): 2D chemical graph/stereochemistry layout (`octet_mol`), HOSE ¹H/¹³C NMR & exact-mass/IUPAC analytical chemistry (`octet_analyt`), publication-grade `ACS`/`RSC`/`JOC`/`Angewandte` vector/raster rendering (`octet_render`), and interactive `egui` drawing workflows are adapted in pure Rust under Apache License 2.0 (`THIRD_PARTY.md`).

---

## Master Peer-Reviewed Bibliography

### 1. Quadrature Detection, Digital Filtering, Multidimensional DSP & NUS

- **Cooley, J. W. & Tukey, J. W.** (1965). An algorithm for the machine calculation of complex Fourier series. *Math. Comput.* **19**, 297–301.
- **Bluestein, L.** (1970). A linear filtering approach to the computation of discrete Fourier transform. *IEEE Trans. Audio Electroacoust.* **18**, 451–455.
- **Bartholdi, E. & Ernst, R. R.** (1973). Fourier spectroscopy and the causality principle. *J. Magn. Reson.* **11**, 9–19.
- **Ferrige, A. G. & Lindon, J. C.** (1978). Resolution enhancement in FT NMR through the use of a double exponential function. *J. Magn. Reson.* **31**, 337–340.
- **States, D. J., Haberkorn, R. A. & Ruben, D. J.** (1982). A two-dimensional nuclear Overhauser experiment with pure absorption phase in four quadrants. *J. Magn. Reson.* **48**, 286–292.
- **Otting, G., Widmer, H., Wagner, G. & Wüthrich, K.** (1986). Origin of $t_1$ and $t_2$ ridges in 2D NMR spectra and procedures for suppression. *J. Magn. Reson.* **66**, 187–193.
- **Ernst, R. R., Bodenhausen, G. & Wokaun, A.** (1987). *Principles of Nuclear Magnetic Resonance in One and Two Dimensions*. Clarendon Press, Oxford.
- **Marion, D., Ikura, M. & Bax, A.** (1989). Improved solvent suppression in one- and two-dimensional NMR spectra by convolution of time-domain data. *J. Magn. Reson.* **84**, 425–430.
- **Marion, D., Ikura, M., Tschudin, R. & Bax, A.** (1989). Rapid recording of 2D NMR spectra without phase cycling: application to the study of hydrogen exchange in proteins. *J. Magn. Reson.* **85**, 393–399.
- **Zhu, G. & Bax, A.** (1990). Improved linear prediction for truncated signals of known phase. *J. Magn. Reson.* **90**, 405–410.
- **Palmer, A. G. III, Cavanagh, J., Wright, P. E. & Rance, M.** (1991). Sensitivity improvement in proton-detected two-dimensional heteronuclear correlation NMR spectroscopy. *J. Magn. Reson.* **93**, 151–170.
- **Kay, L. E., Keifer, P. & Saarinen, T.** (1992). Pure absorption gradient enhanced heteronuclear single quantum correlation spectroscopy with improved sensitivity. *J. Am. Chem. Soc.* **114**, 10663–10665.
- **Delaglio, F., Grzesiek, S., Vuister, G. W., Zhu, G., Pfeifer, J. & Bax, A.** (1995). Multidimensional spectral processing using vector-stream operators. *J. Biomol. NMR* **6**, 277–293.
- **Friedrichs, M. S.** (1995). A model-free algorithm for the removal of baseline artifacts. *J. Biomol. NMR* **5**, 147–153.
- **Hoch, J. C. & Stern, A. S.** (1996). *NMR Data Processing*. Wiley-Liss, New York.
- **Moskau, D.** (2002). Application of real time digital filters in NMR spectroscopy. *Concepts Magn. Reson.* **15**, 164–176.
- **Chen, L., Weng, Z., Goh, L. & Garland, M.** (2002). An efficient algorithm for automatic phase correction of NMR spectra based on entropy minimization. *J. Magn. Reson.* **158**, 164–168.
- **Cavanagh, J., Fairbrother, W. J., Palmer, A. G. III, Rance, M. & Skelton, N. J.** (2007). *Protein NMR Spectroscopy: Principles and Practice*, 2nd ed. Academic Press.
- **Stern, A. S., Donoho, D. L. & Hoch, J. C.** (2007). NMR data processing using iterative thresholding and minimum $l_1$-norm reconstruction. *J. Magn. Reson.* **188**, 295–300.
- **Beck, A. & Teboulle, M.** (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM J. Imaging Sci.* **2**, 183–202.
- **Holland, D. J., Bostock, M. J., Gladden, L. F. & Nietlispach, D.** (2011). Fast multidimensional NMR spectroscopy using compressed sensing. *Angew. Chem. Int. Ed.* **50**, 6548–6551.
- **Hyberts, S. G., Milbradt, A. G., Wagner, A. B., Arthanari, H. & Wagner, G.** (2012). Application of iterative soft thresholding for fast reconstruction of NMR data non-uniformly sampled with multidimensional Poisson Gap scheduling. *J. Biomol. NMR* **52**, 315–327.

### 2. Triple-Resonance Physics, Peak Picking, Backbone Solvers & Calibration

- **Kirkpatrick, S., Gelatt, C. D. & Vecchi, M. P.** (1983). Optimization by simulated annealing. *Science* **220**, 671–680.
- **Chow, C. K.** (1970). On optimum recognition error and reject tradeoff. *IEEE Trans. Inf. Theory* **16**, 41–46.
- **Kay, L. E., Ikura, M., Tschudin, R. & Bax, A.** (1990). Three-dimensional triple-resonance NMR spectroscopy of isotopically enriched proteins. *J. Magn. Reson.* **89**, 496–514.
- **Spera, S. & Bax, A.** (1991). Empirical correlation between protein backbone conformation and $\text{C}^\alpha$ and $\text{C}^\beta$ $^{13}\text{C}$ nuclear magnetic resonance chemical shifts. *J. Am. Chem. Soc.* **113**, 5490–5492.
- **MacArthur, M. W. & Thornton, J. M.** (1991). Influence of proline residues on protein conformation. *J. Mol. Biol.* **218**, 397–412.
- **Grzesiek, S. & Bax, A.** (1992). Correlating backbone amide and side chain resonances in larger proteins by multiple relayed triple resonance NMR. *J. Am. Chem. Soc.* **114**, 6291–6293.
- **Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D.** (1995). $^{1}\text{H}$, $^{13}\text{C}$ and $^{15}\text{N}$ random coil NMR chemical shifts of the common amino acids. I. Investigations of nearest-neighbor effects. *J. Biomol. NMR* **5**, 67–81.
- **Wishart, D. S., Bigam, C. G., Yao, J., Abildgaard, F., Dyson, H. J., Oldfield, E., Markley, J. L. & Sykes, B. D.** (1995). $^{1}\text{H}$, $^{13}\text{C}$ and $^{15}\text{N}$ chemical shift referencing in biomolecular NMR. *J. Biomol. NMR* **6**, 135–140.
- **Schwarzinger, S., Kroon, G. J. A., Foss, T. R., Chung, J., Wright, P. E. & Dyson, H. J.** (2001). Sequence-dependent correction of random coil NMR chemical shifts. *J. Am. Chem. Soc.* **123**, 2970–2978.
- **Zhang, H., Neal, S. & Wishart, D. S.** (2003). RefDB: a database of uniformly referenced protein chemical shifts. *J. Biomol. NMR* **25**, 173–195.
- **Wang, L. & Wishart, D. S.** (2005). A simple method to check chemically mis-referenced protein NMR chemical shifts. *J. Biomol. NMR* **31**, 143–148.
- **Ferro, C. A. T. & Fricker, T. E.** (2012). A bias-corrected decomposition of the Brier score. *Q. J. R. Meteorol. Soc.* **138**, 1954–1960.
- **Schmidt, E. & Güntert, P.** (2012). A new algorithm for reliable and general NMR resonance assignment. *J. Am. Chem. Soc.* **134**, 12817–12829.
- **Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q.** (2017). On calibration of modern neural networks. *Proc. 34th ICML* (**PMLR 70**), 1321–1330.
- **Klukowski, P., Augoff, M., Zieba, M., Drabikowski, M., Gonczarek, A. & Walczak, M. J.** (2018). NMRNet: a deep learning approach to automated peak picking of protein NMR spectra. *Bioinformatics* **34**, 2590–2597.
- **Li, D.-W., Hansen, A. L., Bruschweiler-Li, L., Yuan, C. & Brüschweiler, R.** (2021). Deep neural network peak picking of 2D and 3D multidimensional NMR spectra. *Nat. Commun.* **12**, 5229.
- **Klukowski, P., Riek, R. & Güntert, P.** (2022). Rapid protein assignments and structures from raw NMR spectra with the deep learning technique. *Nat. Commun.* **13**, 6151.
- **Bishop, K. P., Torres-Montalvo, G., Kotaru, S., Mimun, L. C. & Wand, A. J.** (2023). Robust automated backbone triple resonance NMR assignments of proteins using Bayesian-based simulated annealing. *Nat. Commun.* **14**, 1556 (DOI: `10.1038/s41467-023-37219-z`).

### 3. Sidechains, Aromatics, Stereospecificity & Supramolecular Methyl-TROSY

- **Kuhn, H. W.** (1955). The Hungarian method for the assignment problem. *Naval Res. Logist. Quart.* **2**, 83–97; **Munkres, J.** (1957). *J. Soc. Ind. Appl. Math.* **5**, 32–38.
- **Bax, A., Clore, G. M. & Gronenborn, A. M.** (1990). $^{1}\text{H}$–$^{1}\text{H}$ correlation via isotropic mixing of $^{13}\text{C}$ magnetization, a new three-dimensional approach for assigning $^{1}\text{H}$ and $^{13}\text{C}$ spectra of $^{13}\text{C}$-enriched proteins. *J. Magn. Reson.* **88**, 425–431.
- **Güntert, P., Braun, W., Billeter, M. & Wüthrich, K.** (1991). Automated stereospecific $^{1}\text{H}$ NMR assignments and their impact on the precision of protein structure determinations in solution. *J. Am. Chem. Soc.* **111**, 3997–4004.
- **Pelton, J. G., Torchia, D. A., Meadow, N. D. & Roseman, S.** (1993). Tautomeric states of the active-site histidines of phosphorylated and unphosphorylated $\text{III}^{\text{Glc}}$ via two-dimensional heteronuclear NMR. *Protein Sci.* **2**, 543–558.
- **Yamazaki, T., Forman-Kay, J. D. & Kay, L. E.** (1993). Two-dimensional NMR experiments for correlating carbon-13 $\beta$ and proton $\delta/\epsilon$ chemical shifts of aromatic residues in $^{13}\text{C}$-labeled proteins via scalar couplings. *J. Am. Chem. Soc.* **115**, 11054–11055.
- **Nilges, M.** (1995). Calculation of protein structures with ambiguous distance restraints: automated assignment of ambiguous NOE crosspeaks and disulphide bridges. *J. Mol. Biol.* **245**, 645–660.
- **Markley, J. L. et al.** (1998). Recommendations for the presentation of NMR structures of proteins and nucleic acids (IUPAC-IUBMB-IUPAB). *J. Mol. Biol.* **280**, 933–952.
- **Sharma, D. & Rajarathnam, K.** (2000). $^{13}\text{C}$ NMR chemical shifts can predict disulfide bond formation. *J. Biomol. NMR* **18**, 165–171.
- **Lovell, S. C., Word, J. M., Richardson, J. S. & Richardson, D. C.** (2000). The penultimate rotamer library. *Proteins* **40**, 389–408.
- **Tugarinov, V., Hwang, P. M., Ollerenshaw, J. E. & Kay, L. E.** (2003). Cross-correlated relaxation enhanced $^{1}\text{H}$–$^{13}\text{C}$ NMR spectroscopy of methyl groups in very high molecular weight proteins and protein complexes. *J. Am. Chem. Soc.* **125**, 10420–10428.
- **Torizawa, T., Ono, M., Terauchi, T. & Kainosho, M.** (2005). Stereo-array isotope labeling (SAIL) aromatic ring NMR. *J. Am. Chem. Soc.* **127**, 12620–12626.
- **Xu, Y., Matthews, S. et al.** (2009). Automated assignment of methyl NMR resonances from NOE networks. *J. Am. Chem. Soc.* **131**, 9480–9481.
- **Gans, P. et al.** (2010). Stereospecific isotopic labeling of methyl groups for NMR spectroscopic studies of high-molecular-weight proteins. *Angew. Chem. Int. Ed.* **49**, 1958–1962.
- **Chao, F.-A., Shi, L., Masterson, L. R. & Veglia, G.** (2012). Stereospecific and automated methyl resonance assignment in large proteins. *J. Magn. Reson.* **214**, 103–110.
- **Pritišanac, I., Degiacomi, M. T., Alderson, T. R., Carneiro, M. G., Ab, E., Baldwin, A. J. & Kay, L. E.** (2017). Automatic assignment of methyl-NMR spectra of supramolecular machines using graph theory. *J. Am. Chem. Soc.* **139**, 9523–9533.

### 4. IDPs (`13C`-Detected), Propensity & Paramagnetic Metalloproteins

- **Solomon, I.** (1955). Relaxation processes in a system of two spins. *Phys. Rev.* **99**, 559–565.
- **Bertini, I., Luchinat, C. & Parigi, G.** (2002). Magnetic susceptibility in paramagnetic NMR. *Prog. Nucl. Magn. Reson. Spectrosc.* **40**, 249–273.
- **Bermel, W., Bertini, I., Felli, I. C., Lee, Y.-M., Luchinat, C. & Pierattelli, R.** (2006). Protonless NMR experiments for sequence-specific assignment of backbone nuclei in unfolded proteins. *J. Am. Chem. Soc.* **128**, 3918–3919.
- **Marsh, J. A., Singh, V. K., Jia, Z. & Forman-Kay, J. D.** (2006). Sensitivity of secondary structure propensities to sequence differences between $\alpha$- and $\gamma$-synuclein: implications for fibrillation. *Protein Sci.* **15**, 2795–2804.
- **Schmitz, C., Stanton-Cook, M. J., Su, X.-C., Otting, G. & Huber, T.** (2008). $\Delta\chi$-tensor determination and pseudocontact shift modeling in paramagnetic proteins. *J. Biomol. NMR* **41**, 179–189.
- **Bermel, W., Bertini, I., Csizmok, V., Felli, I. C., Pierattelli, R. & Tompa, P.** (2009). H-start vs C-start $^{13}\text{C}$-detected experiments (`3D CANCO`, `CBCACON`, `CCC(CO)N`) for intrinsically disordered proteins. *J. Magn. Reson.* **198**, 275–281.
- **Tamiola, K., Acar, B. & Mulder, F. A. A.** (2010). Sequence-specific random coil chemical shifts of intrinsically disordered proteins (`ncIDP`). *J. Am. Chem. Soc.* **132**, 18000–18003.
- **Kjaergaard, M. & Poulsen, F. M.** (2011). Sequence correction of random coil chemical shifts: correlation between neighbor correction factors and changes in the Ramachandran distribution. *J. Biomol. NMR* **50**, 157–165.
- **Camilloni, C., De Simone, A., Vranken, W. F. & Vendruscolo, M.** (2012). Determination of secondary structure populations in disordered states of proteins using nuclear magnetic resonance chemical shifts ($\delta\text{2D}$). *Biochemistry* **51**, 2224–2231.
- **Platzer, G., Okon, M. & McIntosh, L. P.** (2014). pH-dependent random coil $^{1}\text{H}$, $^{13}\text{C}$, and $^{15}\text{N}$ chemical shifts of the ionizable amino acids. *J. Biomol. NMR* **60**, 109–129.

### 5. Torsions, 3D NOESY Folding, Residual Dipolar Couplings (`RDC`) & 3D Ribbons

- **Kabsch, W.** (1976). A solution for the best rotation to relate two sets of vectors. *Acta Crystallogr. A* **32**, 922–923.
- **Carson, M. & Bugg, C. E.** (1986). Algorithm for ribbon models of proteins. *J. Mol. Graph.* **4**, 121–122.
- **Henikoff, S. & Henikoff, J. G.** (1992). Amino acid substitution matrices from protein blocks (`BLOSUM62`). *Proc. Natl. Acad. Sci. U.S.A.* **89**, 10915–10919.
- **Tjandra, N. & Bax, A.** (1997). Direct measurement of distances and angles in biomolecules by NMR in a dilute liquid crystalline medium. *Science* **278**, 1111–1114.
- **Cornilescu, G., Marquardt, J. L., Ottiger, M. & Bax, A.** (1998). Validation of protein structure from anisotropic carbonyl chemical shifts in a dilute liquid crystalline phase. *J. Am. Chem. Soc.* **120**, 6836–6837.
- **Losonczi, J. A., Andrec, M., Fischer, M. W. F. & Prestegard, J. H.** (1999). Order matrix analysis of residual dipolar couplings using singular value decomposition. *J. Magn. Reson.* **138**, 334–342.
- **Lovell, S. C. et al.** (2003). Structure validation by $\text{C}^\alpha$ geometry: $\phi, \psi$ and $\text{C}^\beta$ deviation. *Proteins* **50**, 437–450.
- **Zhang, Y. & Skolnick, J.** (2004). Scoring function for automated assessment of protein structure template quality (`TM-score`). *Proteins* **57**, 402–410.
- **Parsons, J., Holmes, J. B., Rojas, J. M., Tsai, J. & Strauss, C. E. M.** (2005). Practical conversion from torsion space to Cartesian space for *in silico* protein synthesis (`NeRF`). *J. Comput. Chem.* **26**, 1063–1068.
- **Berjanskii, M. V. & Wishart, D. S.** (2005). A simple method to predict protein flexibility using secondary chemical shifts (`RCI`). *J. Am. Chem. Soc.* **127**, 14970–14971.
- **Shen, Y., Vernon, R., Baker, D. & Bax, A.** (2008). Consistent blind protein structure generation from NMR chemical shift data. *Proc. Natl. Acad. Sci. U.S.A.* **105**, 4685–4690.
- **Shen, Y. & Bax, A.** (2013). Protein backbone and sidechain torsion angles predicted from NMR chemical shifts using artificial neural networks. *J. Biomol. NMR* **56**, 227–241.

### 6. Spin Relaxation, Model-Free Dynamics ($S^2$), `CEST`, `CPMG` & Ligand `CSP`

- **McConnell, H. M.** (1958). Reaction rates by nuclear magnetic resonance. *J. Chem. Phys.* **28**, 430–431.
- **Woessner, D. E.** (1962). Nuclear spin relaxation in ellipsoids undergoing rotational Brownian motion. *J. Chem. Phys.* **37**, 647–654.
- **Luz, Z. & Meiboom, S.** (1963). Nuclear magnetic resonance study of the protolysis of trimethylammonium ion in aqueous solution—order of the reaction with respect to solvent. *J. Chem. Phys.* **39**, 366–370.
- **Carver, J. P. & Richards, R. E.** (1972). A general two-site solution for the chemical exchange produced dependence of $T_2$ upon the Carr–Purcell pulse separation. *J. Magn. Reson.* **6**, 89–105.
- **Lipari, G. & Szabo, A.** (1982). Model-free approach to the interpretation of nuclear magnetic resonance relaxation in macromolecules. 1. Theory and range of validity; 2. Analysis of experimental results. *J. Am. Chem. Soc.* **104**, 4546–4559 & 4559–4570.
- **Kay, L. E., Torchia, D. A. & Bax, A.** (1989). Backbone dynamics of proteins as studied by $^{15}\text{N}$ inverse detected heteronuclear NMR spectroscopy: application to staphylococcal nuclease. *Biochemistry* **28**, 8972–8979.
- **Clore, G. M., Szabo, A., Bax, A., Kay, L. E., Driscoll, P. C. & Gronenborn, A. M.** (1990). Deviations from the simple two-parameter model-free approach to the interpretation of nitrogen-15 nuclear magnetic relaxation of proteins. *J. Am. Chem. Soc.* **112**, 4989–4991.
- **Farrow, N. A. et al.** (1994). Backbone dynamics of a free and a phosphopeptide-complexed Src homology 2 domain studied by $^{15}\text{N}$ NMR relaxation. *Biochemistry* **33**, 5984–6003.
- **Mandel, A. M., Akke, M. & Palmer, A. G. III** (1995). Backbone dynamics of *Escherichia coli* ribonuclease HI: correlations with structure and function in an active enzyme. *J. Mol. Biol.* **246**, 144–163.
- **Farrow, N. A., Zhang, O., Szabo, A., Torchia, D. A. & Kay, L. E.** (1995). Spectral density function mapping using $^{15}\text{N}$ relaxation data exclusively. *J. Biomol. NMR* **6**, 153–162.
- **Lefèvre, J.-F., Dayie, K. T., Peng, J. W. & Wagner, G.** (1996). Internal mobility in the partially folded DNA binding and dimerization domains of GAL4: NMR analysis of the N–H spectral density functions. *Biochemistry* **35**, 2674–2686.
- **Lee, L. K., Rance, M., Chazin, W. J. & Palmer, A. G. III** (1997). Rotational diffusion anisotropy of proteins from simultaneous analysis of $^{15}\text{N}$ and $^{13}\text{C}^\alpha$ nuclear spin relaxation. *J. Biomol. NMR* **9**, 287–298.
- **Trott, O. & Palmer, A. G. III** (2002). $R_{1\rho}$ relaxation outside of the fast-exchange limit. *J. Magn. Reson.* **154**, 157–160.
- **d'Auvergne, E. J. & Gooley, P. R.** (2003). The use of model selection in the model-free analysis of protein dynamics. *J. Biomol. NMR* **25**, 25–39; and (2006) *J. Biomol. NMR* **35**, 117–135.
- **Higham, N. J.** (2005). The scaling and squaring method for the matrix exponential revisited. *SIAM J. Matrix Anal. Appl.* **26**, 1179–1193.
- **Fawzi, N. L., Ying, J., Ghirlando, R., Torchia, D. A. & Clore, G. M.** (2011). Atomic-resolution dynamics on the surface of amyloid-$\beta$ protofibrils probed by solution NMR (`DEST`). *Nature* **480**, 268–272.
- **Vallurupalli, P., Bouvignies, G. & Kay, L. E.** (2012). Studying "invisible" excited protein states in slow exchange with a major state conformation (`CEST`). *J. Am. Chem. Soc.* **134**, 8148–8161.
- **Williamson, M. P.** (2013). Using chemical shift perturbation to measure ligand binding. *Prog. Nucl. Magn. Reson. Spectrosc.* **73**, 1–16.
- **Baldwin, A. J.** (2014). An exact solution for $R_{2,\text{eff}}$ in CPMG experiments in the case of two site chemical exchange. *J. Magn. Reson.* **244**, 114–124.

### 7. Analytical 1D NMR, Hoye Multiplet Trees, `qNMR`, Chemometrics & `DOSY`

- **Einstein, A.** (1905). Über die von der molekularkinetischen Theorie der Wärme geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten Teilchen. *Ann. Phys.* **322**, 549–560.
- **Hotelling, H.** (1931). The generalization of Student's ratio ($T^2$). *Ann. Math. Stat.* **2**, 360–378.
- **Levenberg, K.** (1944). *Quart. Appl. Math.* **2**, 164–168; **Marquardt, D. W.** (1963). *J. Soc. Ind. Appl. Math.* **11**, 431–441.
- **Pople, J. A., Schneider, W. G. & Bernstein, H. J.** (1959). *High-Resolution Nuclear Magnetic Resonance*. McGraw-Hill.
- **Stejskal, E. O. & Tanner, J. E.** (1965). Spin diffusion measurements: spin echoes in the presence of a time-dependent field gradient. *J. Chem. Phys.* **42**, 288–292.
- **Ernst, R. R. & Anderson, W. A.** (1966). Application of Fourier transform spectroscopy to magnetic resonance. *Rev. Sci. Instrum.* **37**, 93–102.
- **Golub, G. H. & Pereyra, V.** (1973). The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate (`VarPro`). *SIAM J. Numer. Anal.* **10**, 413–432.
- **Lawson, C. L. & Hanson, R. J.** (1974). *Solving Least Squares Problems* (`NNLS`). Prentice-Hall.
- **Jackson, J. E. & Mudholkar, G. S.** (1979). Control procedures for residuals associated with principal component analysis ($Q$-statistic). *Technometrics* **21**, 341–349.
- **Doddrell, D. M., Pegg, D. T. & Bendall, M. R.** (1982). Distortionless enhancement of NMR signals by polarization transfer (`DEPT`). *J. Magn. Reson.* **48**, 323–327.
- **Provencher, S. W.** (1982). A constrained regularization method for inverting data represented by linear algebraic or integral equations. *Comput. Phys. Commun.* **27**, 213–227 & 229–242.
- **Barkhuijsen, H., de Beer, R. & van Ormondt, D.** (1987). Improved algorithm for noniterative time-domain model fitting to exponentially damped magnetic resonance signals (`HSVD`). *J. Magn. Reson.* **73**, 553–557.
- **Thompson, P., Cox, D. E. & Hastings, J. B.** (1987). Rietveld refinement of Debye–Scherrer synchrotron X-ray data from $\text{Al}_2\text{O}_3$ (pseudo-Voigt approximation). *J. Appl. Crystallogr.* **20**, 79–83.
- **Pijnappel, W. W. F., van den Boogaart, A., de Beer, R. & van Ormondt, D.** (1992). SVD-based quantification of magnetic resonance signals. *J. Magn. Reson.* **97**, 122–134.
- **Hoye, T. R., Hanson, P. R. & Vyvyan, J. R.** (1994). A practical guide to first-order multiplet analysis in $^{1}\text{H}$ NMR spectroscopy. *J. Org. Chem.* **59**, 4096–4103.
- **Benjamini, Y. & Hochberg, Y.** (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing (`BH-FDR`). *J. R. Stat. Soc. B* **57**, 289–300.
- **Wu, D. H., Chen, A. D. & Johnson, C. S. Jr.** (1995). An improved diffusion-ordered spectroscopy experiment incorporating bipolar-gradient pulses (`BPP-LED`). *J. Magn. Reson. A* **115**, 260–264.
- **Gottlieb, H. E., Kotlyar, V. & Nudelman, A.** (1997). NMR chemical shifts of common laboratory solvents as trace impurities. *J. Org. Chem.* **62**, 7512–7515.
- **Johnson, C. S. Jr.** (1999). Diffusion ordered nuclear magnetic resonance spectroscopy: principles and applications. *Prog. Nucl. Magn. Reson. Spectrosc.* **34**, 203–256.
- **Hoye, T. R. & Zhao, H.** (2002). A method for easily determining coupling constant values: an addendum to "A practical guide to first-order multiplet analysis in $^{1}\text{H}$ NMR spectroscopy." *J. Org. Chem.* **67**, 4014–4016.
- **Pelta, M. D., Morris, G. A., Stchedroff, M. J. & Hammond, S. J.** (2002). A one-shot high-resolution `DOSY` experiment. *Magn. Reson. Chem.* **40**, 147–152.
- **Trygg, J. & Wold, S.** (2002). Orthogonal projections to latent structures (`O-PLS`). *J. Chemom.* **16**, 119–128.
- **Eilers, P. H. C.** (2003). A perfect smoother (Whittaker penalized least squares). *Anal. Chem.* **75**, 3631–3636.
- **Chong, I.-G. & Jun, C.-H.** (2005). Performance of some variable selection methods when multicollinearity is present (`VIP`). *Chemom. Intell. Lab. Syst.* **78**, 103–112.
- **Cloarec, O., Dumas, M.-E., Craig, A., Barton, R. H., Trygg, J., Hudson, J., Blancher, C., Gauguier, D., Lindon, J. C., Holmes, E. & Nicholson, J.** (2005). Statistical total correlation spectroscopy (`STOCSY`): an exploratory approach for latent biomarker identification from metabolic $^{1}\text{H}$ NMR data sets. *Anal. Chem.* **77**, 1282–1289.
- **Malz, F. & Jancke, H.** (2005). Validation of quantitative NMR (`qNMR`). *J. Pharm. Biomed. Anal.* **38**, 813–823.
- **Cobas, J. C., Bernstein, M. A., Martín-Pastor, M. & Tahoces, P. G.** (2006). A new general-purpose fully automatic baseline-correction procedure for 1D and 2D NMR data. *J. Magn. Reson.* **183**, 145–151.
- **Dieterle, F., Ross, A., Schlotterbeck, G. & Senn, H.** (2006). Probabilistic quotient normalization (`PQN`) as robust method to account for dilution of complex biological mixtures. *Anal. Chem.* **78**, 4281–4290.
- **van den Berg, R. A., Hoefsloot, H. C. J., Westerhuis, J. A., Smilde, A. K. & van der Werf, M. J.** (2006). Centering, scaling, and transformations: improving the biological information content of metabolomics data. *BMC Genomics* **7**, 142.
- **De Meyer, T., Sinnaeve, D., Van Gasse, B., Tsiporkova, E., Rietzschel, E. R., De Buyzere, M. L., Gillebert, T. C., Bekaert, S., Martins, J. C. & Van Criekinge, W.** (2008). NMR-based characterization of metabolic alterations in hypertension using an adaptive, intelligent binning algorithm (`AI-Binning`). *Anal. Chem.* **80**, 3783–3790.
- **Westerhuis, J. A., Hoefsloot, H. C. J., Smit, S., Vis, D. J., Smilde, A. K., van Velzen, E. J. J., van Duijnhoven, J. P. M. & van Dorsten, F. A.** (2008). Assessment of PLSDA cross validation. *Metabolomics* **4**, 81–89.
- **Wiklund, S., Johansson, E., Sjöström, L., Mellerowicz, E. J., Edlund, U., Shockcor, J. P., Gottfries, J., Moritz, T. & Trygg, J.** (2008). Visualization of GC/TOF-MS-based metabolomics data for identification of biochemically interesting compounds using `OPLS` class models (`S-plot`). *Anal. Chem.* **80**, 115–122.
- **de Brouwer, H.** (2009). Evaluation of algorithms for automated phase correction of NMR spectra. *J. Magn. Reson.* **201**, 230–238.
- **Fulmer, G. R., Miller, A. J. M., Sherden, N. H., Gottlieb, H. E., Nudelman, A., Stoltz, B. M., Bercaw, J. E. & Goldberg, K. I.** (2010). NMR chemical shifts of trace impurities: common laboratory solvents, organics, and gases in deuterated solvents relevant to the organometallic chemist. *Organometallics* **29**, 2176–2179.
- **Phipson, B. & Smyth, G. K.** (2010). Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. *Stat. Appl. Genet. Mol. Biol.* **9**, Article 39.
- **Savorani, F., Tomasi, G. & Engelsen, S. B.** (2010). Interval correlation-shifting FFT peak alignment of 1D NMR spectra. *J. Magn. Reson.* **202**, 190–202.
- **Evans, R., Deng, Z., Rogerson, A. K., McLachlan, A. S., Richards, J. J., Nilsson, M. & Morris, G. A.** (2013). Quantitative interpretation of diffusion-ordered NMR spectra: can we rationalize small molecule diffusion coefficients? *Angew. Chem. Int. Ed.* **52**, 3199–3202.
- **Golub, G. H. & Van Loan, C. F.** (2013). *Matrix Computations*, 4th ed. Johns Hopkins University Press.
- **Pauli, G. F., Chen, S.-N., Simmler, C., Lankin, D. C., Gödecke, T., Jaki, B. U., Friesen, J. B., McAlpine, J. B. & Napolitano, J. G.** (2014). Importance of purity evaluation and the potential of quantitative $^{1}\text{H}$ NMR as a purity assay. *J. Med. Chem.* **57**, 9220–9231.

### 8. Open Archival Repositories, Interchange Formats & Protocol Standards

- **Ulrich, E. L. et al.** (2008). BioMagResBank (`BMRB` / `NMR-STAR v3.2`). *Nucleic Acids Res.* **36**, D402–D408.
- **Gutmanas, A. et al.** (2015). NMR Exchange Format (`NEF v1.1`): a unified standard for representation of NMR restraint and chemical shift data. *Nat. Struct. Mol. Biol.* **22**, 433–434.
- **Westbrook, J. D. et al.** (2015). The chemical component dictionary (`wwPDB CCD`): complete descriptions of constituent molecules in experimentally determined 3D macromolecules. *Bioinformatics* **31**, 1274–1278.
- **Model Context Protocol (`MCP`) Specification** (`2024-11-05` & `2025-03-26`) & **JSON-RPC 2.0 Specification**; **RFC 2083** (PNG), **RFC 1950/1951** (ZLIB/DEFLATE), **RFC 4648** (Base64), **RFC 8259** (JSON), and **RFC 9110** (HTTP Semantics).

### 9. AlphaFold Concordance, 3D Shift Forward Physics, Multi-Domain Docking, Co-Folding & Excited States

- **McConnell, H. M.** (1957). Theory of nuclear magnetic shielding by electron currents. *J. Chem. Phys.* **27**, 226–229.
- **Buckingham, A. D.** (1960). Chemical shifts in the nuclear magnetic resonance spectra of molecules containing polar groups. *Can. J. Chem.* **38**, 300–307.
- **Haigh, C. W. & Mallion, R. B.** (1972). New tables of "ring current" shielding in proton magnetic resonance. *Mol. Phys.* **24**, 955–970; and (1979) *Prog. Nucl. Magn. Reson. Spectrosc.* **13**, 303–344.
- **Shrake, A. & Rupley, J. A.** (1973). Environment and exposure to solvent of protein atoms. *J. Mol. Biol.* **79**, 351–371.
- **Schwarz, G.** (1978). Estimating the dimension of a model (`BIC`). *Ann. Stat.* **6**, 461–464.
- **Golub, G. H., Heath, M. & Wahba, G.** (1979). Generalized cross-validation as a method for choosing a good ridge parameter (`GCV`). *Technometrics* **21**, 215–223.
- **Kabsch, W. & Sander, C.** (1983). Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features. *Biopolymers* **22**, 2577–2637.
- **Wagner, G., Pardi, A. & Bax, A.** (1983). Hydrogen bond length and $^{1}\text{H}$ NMR chemical shifts in proteins. *J. Am. Chem. Soc.* **105**, 5948–5949.
- **Lange, K. L., Little, R. J. A. & Taylor, J. M. G.** (1989). Robust statistical modeling using the $t$ distribution. *J. Am. Stat. Assoc.* **84**, 881–896.
- **Case, D. A.** (1995). Calibration of ring-current effects in proteins and nucleic acids. *J. Biomol. NMR* **6**, 341–346.
- **Kass, R. E. & Raftery, A. E.** (1995). Bayes factors. *J. Am. Stat. Assoc.* **90**, 773–795.
- **MacKerell, A. D. Jr. et al.** (1998). All-atom empirical potential for molecular modeling and dynamics studies of proteins (`CHARMM22`). *J. Phys. Chem. B* **102**, 3586–3616.
- **Sitkoff, D. & Case, D. A.** (1998). Theories of chemical shift anisotropies in proteins and nucleic acids. *Prog. Nucl. Magn. Reson. Spectrosc.* **32**, 165–190.
- **Braddock, D. T., Cai, M., Baber, J. L., Huang, Y. & Clore, G. M.** (2001). Rapid identification of medium- to large-scale interdomain motion in modular proteins using dipolar couplings. *J. Am. Chem. Soc.* **123**, 8634–8635.
- **Ng, A. Y., Jordan, M. I. & Weiss, Y.** (2001). On spectral clustering: analysis and an algorithm. *Adv. Neural Inf. Process. Syst.* **14**, 849–856.
- **Tolman, J. R., Al-Hashimi, H. M., Kay, L. E. & Prestegard, J. H.** (2001). Structural and dynamic analysis of residual dipolar coupling data for proteins (`GDO`). *J. Am. Chem. Soc.* **123**, 1416–1424.
- **Schubert, M., Labudde, D., Oschkinat, H. & Schmieder, P.** (2002). A software tool for the prediction of Xaa-Pro peptide bond conformations in proteins based on $^{13}\text{C}$ chemical shift statistics. *J. Biomol. NMR* **24**, 149–154.
- **Skrynnikov, N. R., Dahlquist, F. W. & Kay, L. E.** (2002). Reconstructing NMR spectra of "invisible" excited protein states using HSQC and HMQC experiments. *J. Am. Chem. Soc.* **124**, 12352–12360.
- **Clore, G. M. & Schwieters, C. D.** (2004). Amplitudes of protein backbone dynamics and correlated motions in a small $\alpha/\beta$ protein: correspondence of dipolar coupling and heteronuclear relaxation measurements. *J. Am. Chem. Soc.* **126**, 2923–2938.
- **Bertini, I., Gupta, Y. K., Luchinat, C., Parigi, G., Peana, M., Sgheri, L. & Yuan, J.** (2007). Paramagnetism-based NMR restraints provide maximum allowed probabilities for the different conformations of partially independent protein domains. *J. Am. Chem. Soc.* **129**, 12786–12794.
- **Vallurupalli, P., Hansen, D. F. & Kay, L. E.** (2008). Structures of invisible, excited protein states by relaxation dispersion NMR spectroscopy. *Proc. Natl. Acad. Sci. U.S.A.* **105**, 11766–11771.
- **Shen, Y. & Bax, A.** (2010). Empirical NMR chemical shift prediction from protein structure and additive physical shift decomposition. *J. Biomol. NMR* **48**, 13–22; and (2010) *J. Biomol. NMR* **46**, 199–204.
- **Bouvignies, G., Vallurupalli, P., Hansen, D. F., Correia, B. E., Lange, O., Bah, A., Vernon, R. M., Dahlquist, F. W., Baker, D. & Kay, L. E.** (2011). Solution structure of a minor and transiently formed state of a T4 lysozyme mutant. *Nature* **477**, 111–114.
- **Sahakyan, A. B., Vranken, W. F., Cavalli, A. & Vendruscolo, M.** (2011). Structure-based prediction of methyl chemical shifts in proteins. *Angew. Chem. Int. Ed.* **50**, 9620–9623.
- **Adzhubei, A. A., Sternberg, M. J. E. & Makarov, A. A.** (2013). Polyproline-II helix in proteins: structure and function. *J. Mol. Biol.* **425**, 2100–2132.
- **Fowler, N. J., Sljoka, A. & Williamson, M. P.** (2020). A method for validating the accuracy of NMR protein structures. *Nat. Commun.* **11**, 5982; and **Fowler, N. J. & Williamson, M. P.** (2022). *Structure* **30**, 925–933.
- **Jumper, J., Evans, R., Pritzel, A. et al.** (2021). Highly accurate protein structure prediction with AlphaFold. *Nature* **596**, 583–589.
- **Varadi, M., Anyango, S., Deshpande, M. et al.** (2022). AlphaFold Protein Structure Database (`AFDB`). *Nucleic Acids Res.* **50**, D439–D444; and **Varadi, M. et al.** (2024). *Nucleic Acids Res.* **52**, D368–D375.
- **Alderson, T. R., Pritišanac, I., Kolarić, Đ., Moses, A. M. & Forman-Kay, J. D.** (2023). Systematic identification of conditionally folded intrinsically disordered regions by AlphaFold2. *Proc. Natl. Acad. Sci. U.S.A.* **120**, e2304302120.
- **Abramson, J., Adler, J., Dunger, J. et al.** (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* **630**, 493–500.
- **Wayment-Steele, H. K., Ojoawo, A., Otten, R. et al.** (2024). Predicting multiple conformations via sequence clustering and AlphaFold2. *Nature* **625**, 832–839.
- **Wohlwend, J., Corso, G., Passaro, S. et al.** (2024). Boltz-1: democratizing biomolecular interaction modeling. *bioRxiv* DOI: `10.1101/2024.11.19.624167`; and **Passaro, S., Corso, G., Wohlwend, J. et al.** (2025). Boltz-2: towards accurate and efficient binding affinity prediction. *bioRxiv* DOI: `10.1101/2025.06.14.659707`.
