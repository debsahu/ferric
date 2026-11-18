# AlphaFold Protein Structure Database (AFDB) Download Provenance (`data/afdb/`)

- **Archive**: AlphaFold Protein Structure Database (EMBL-EBI / Google DeepMind)
- **License**: Creative Commons Attribution 4.0 International (`CC-BY-4.0`; Jumper, J. et al., *Nature* **2021**, 596, 583–589; Varadi, M. et al., *Nucleic Acids Res.* **2022**, 50, D439–D444; Varadi, M. et al., *Nucleic Acids Res.* **2024**, 52, D368–D375)
- **User-Agent**: `ferric-af-fetch` (`<= 1 req/s`)

## `P0CG48` (`AF-P0CG48-F1`, AFDB `v6` — retrieved `2026-11-18T14:58:00Z`)

- **UniProt Accession / ID**: `P0CG48` (`UBC_HUMAN`, *Homo sapiens* Polyubiquitin-C, 685 aa: 9 tandem 76-residue ubiquitin repeats `1..76`, `77..152`, ..., `609..684` + C-terminal `Val685`; repeat 1 `residues 1..=76` is exact 76-residue monomeric Human Ubiquitin `MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG`)
- **Metadata API URL**: `https://alphafold.ebi.ac.uk/api/prediction/P0CG48` (`HTTP 200`, `latestVersion: 6`, `modelCreatedDate: 2025-08-01T00:00:00Z`, `toolUsed: "AlphaFold Monomer v2.0 pipeline"`; note that legacy `AF-P62988-F1` and `_v4` file URLs return `HTTP 404` on the live AFDB server)
- **Coordinate mmCIF**: `data/afdb/AF-P0CG48-F1-model_v6.cif`
  - Upstream URL: `https://alphafold.ebi.ac.uk/files/AF-P0CG48-F1-model_v6.cif`
  - Retrieval Date (UTC): `2026-11-18T14:58:00Z`
  - Size: `636703` bytes
  - FIPS 180-4 SHA-256: `02b8ea3d522fabee58abed221fad1a2101e9ad633ddecbe30bc96d158fc2c408`
  - Embedded Quality-Assessment Schema: `_ma_qa_metric` (`id=1 global pLDDT`, `id=2 local pLDDT`) + `_ma_qa_metric_local` (`685` residue rows; global mean `pLDDT = 88.65`, repeat 1 `1..=76` mean `pLDDT = 90.02`, core `1..=71` `pLDDT = 78.31..96.81`, flexible C-terminal `RLRGG` tail `72..=76` `pLDDT = [73.94, 60.28, 54.91, 59.34, 70.00]`, repeat 9 C-terminal tail `680..=685` `pLDDT = [78.25, 62.25, 54.28, 46.66, 38.44, 33.41]`)
- **Predicted Aligned Error JSON**: `data/afdb/AF-P0CG48-F1-predicted_aligned_error_v6.json`
  - Upstream URL: `https://alphafold.ebi.ac.uk/files/AF-P0CG48-F1-predicted_aligned_error_v6.json`
  - Retrieval Date (UTC): `2026-11-18T14:58:00Z`
  - Size: `1353822` bytes
  - FIPS 180-4 SHA-256: `edb163b4dcd75ecd253ef482c15555b524fda3fb91f75ef965ad5426762b1c64`
  - Schema: AFDB dense `[{"predicted_aligned_error": [[...]], "max_predicted_aligned_error": 31.75}]` (`685 x 685`)

## Scientific Candor & AlphaFold2 Training-Cutoff Note (`A1`, `21.W1:P0-1`)

- **AlphaFold2 Training Cutoff (`2018-04-30`)**: AlphaFold Monomer v2.0 was trained on PDB coordinate entries released on or before **`2018-04-30`**. Human Ubiquitin structures — including X-ray `1UBQ` (released `1987-10-15`), solution NMR `1D3Z` (released `1999-02-02`), and polyubiquitin `5B83` (released `2016-06-22`) — all **predate** the `2018-04-30` training cutoff.
- **Explicit Template Record in `AF-P0CG48-F1-model_v6.cif`**: Lines `1639–1645` (`_ma_template_ref_db_details`) of `AF-P0CG48-F1-model_v6.cif` explicitly record `5B83` (`PDB`) as an input template structure used by the AlphaFold Monomer v2.0 pipeline.
- **Evidential Classification**: Any concordance (`T21.1`), forward shift prediction (`T21.2`), or assignment prior (`T21.4`) benchmark evaluated on `AF-P0CG48-F1` (or any pre-`2018-04-30` AF2 / pre-`2021-09-30` AF3 target) measures compatibility on a **pre-cutoff, template-assisted target seen during AlphaFold training** (`dev-set` / `pre-cutoff AFDB fixture`), and **must not** be cited as cold post-cutoff generalization evidence.
