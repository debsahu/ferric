# Provenance: AlphaFold DB Human Ubiquitin Test Fixtures (`AF-P0CG48-F1`, `v6`)

- **API Endpoint**: `https://alphafold.ebi.ac.uk/api/prediction/P0CG48`
- **Model File**: `AF-P0CG48-F1-model_v6.cif`
  - **URL**: `https://alphafold.ebi.ac.uk/files/AF-P0CG48-F1-model_v6.cif`
  - **Size**: `636703` bytes
  - **SHA-256**: `02b8ea3d522fabee58abed221fad1a2101e9ad633ddecbe30bc96d158fc2c408`
  - **Note**: AFDB superseded `v4` (`AF-P0CG48-F1-model_v4.cif` now returns HTTP 404); `v6` (`latestVersion: 6`, `modelCreatedDate: 2025-08-01T00:00:00Z`, `AlphaFold Monomer v2.0 pipeline`) is the live AFDB release and contains 685 residues (`P0CG48`, Polyubiquitin-C), whose repeat 1 (`1..=76`) matches 76-aa human Ubiquitin with real `_ma_qa_metric` / `_ma_qa_metric_local` `pLDDT` records.
- **PAE File**: `AF-P0CG48-F1-predicted_aligned_error_v6.json`
  - **URL**: `https://alphafold.ebi.ac.uk/files/AF-P0CG48-F1-predicted_aligned_error_v6.json`
  - **Size**: `1353822` bytes
  - **SHA-256**: `edb163b4dcd75ecd253ef482c15555b524fda3fb91f75ef965ad5426762b1c64`
- **Retrieval Date (UTC)**: `2026-11-18T14:56:45Z` (identical blobs shared with Slice `S2` `data/afdb/` commit `9f45a228f75410d2a883da7cc5e04bfe5e067a49`).
- **Licence**: `CC-BY-4.0` (Varadi et al., *Nucleic Acids Res.* **2024**, 52, D368-D375; Jumper et al., *Nature* **2021**, 596, 583-589).
- **Training-Cutoff Caveat (`A1`)**: AlphaFold 2 trained on PDB structures released up to `2018-04-30`, and Human Ubiquitin (`1UBQ` 1987, `1D3Z` 1999) predates that cutoff. Thus `AF-P0CG48-F1` is an in-training-era protein (partially memorized) and is NOT evidence of cold post-cutoff AlphaFold generalization; it is used strictly to verify authentic AFDB ingestion, `PAE` gating, and non-circular ablation behaviour against distinct-conformer `1UBQ` simulated NOESYs.
