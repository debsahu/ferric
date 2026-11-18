# Public JEOL Delta `.jdf` Fixture Provenance (`20.W1:P2-2`)

## File

- **Fixture path**: `data/fixtures/jeol_public/rutin_q1h_400mhz_dmsod6.jdf`
- **Upstream filename**: `Rutin_3080ug200uL_DMSOd6_qHNMR_400MHz_Jeol.jdf` (archived inside `Rutin_NMRdata_400MHz_DMSOd6_Jeol.zip`)
- **Size**: `555,392` bytes (`32,768` split-complex `f64` FID samples + `JEOL.NMR` v1.2 header and `64`-byte parameter table)
- **SHA-256**: `bb76e9d4a8bb9dd66b8ddbaeffcee10ce3635f615861caa75630a46453e0cf71`

## Upstream Source & Licence

- **Primary archive**: Harvard Dataverse, `doi:10.7910/DVN/ZAZDNM` — Pauli, G. F. et al., *"Raw NMR Data Files for Rutin (`400 MHz`, `DMSO-d6`, JEOL ECZ400S)"* (`https://dataverse.harvard.edu/api/access/datafile/3597321`, `Rutin_NMRdata_400MHz_DMSOd6_Jeol/Rutin_3080ug200uL_DMSOd6_qHNMR_400MHz_Jeol.jdf`).
- **Byte-identical open mirror**: `https://raw.githubusercontent.com/cheminfo/jeol-data-test/HEAD/data/Rutin_3080ug200uL_DMSOd6_qHNMR_400MHz_Jeol.jdf` (`MIT` repository hosting the unmodified `CC0-1.0` Dataverse `.jdf`).
- **Licence**: `CC0-1.0` (Creative Commons Zero v1.0 Universal Public Domain Dedication on Harvard Dataverse `doi:10.7910/DVN/ZAZDNM`).
- **Retrieved**: `2026-11-18` (byte-for-byte identity between the Harvard Dataverse archive entry and mirror verified via `SHA-256`).

## Acquisition & Format Metadata Verified by `nmr_core::nmr1d::parse_jeol_jdf_1d_bytes`

- **Header magic / endianness**: `"JEOL.NMR"`, `Endian = 1` (little-endian payload, big-endian fixed `1296`-byte file header), `Data_Dimension_Number = 1`, `Data_Format = 1` (`64Bit_Float`), `Data_Axis_Type[0] = 3` (split-complex `re[0..N]` followed by `im[0..N]`).
- **Spectrometer & nucleus**: `1H` quantitative 1D (`qHNMR`), `X_FREQ = 399.78219838 MHz`, `SOLVENT = DMSO-D6`.
- **Digital filter**: `DIGITAL_FILTER = TRUE`, `ORDERS = "2 54 73"`, `FACTORS = "8 2"`, multistage FIR group delay compensated cleanly in `TimeDomain1D`.

## Format Documentation Reference

- JEOL Resonance Inc., *Delta NMR Format Specification (`JEOL.NMR` v1.2 `1296`-byte header & `64`-byte parameter records)*, public vendor format documentation; clean-room implementation in `crates/nmr-core/src/nmr1d.rs` (`parse_jeol_jdf_1d_bytes`, `compute_jeol_fir_group_delay`).
