# Golden `5c4e87d` Scoring Matrix Snapshots

Generated directly from pre-`1be01af` commit `5c4e87d` (`feat(nmr-gui,todo): wire T21.1<->T21.3 SS/concordance bridges and mark Wave 21.W1 complete`) on `Apple M5 Pro` (`Metal`) via `unary_matrix`, `pair_matrix`, and `GpuEngine::score_matrices_gpu`.

Used by `crates/nmr-gpu/tests/af_assign_gpu_tests.rs` to verify non-tautological `0.0` bit-exact regression parity of the sequence-only CPU and WebGPU scoring paths (`21.W2:P0-1`).

See `MANIFEST.tsv` for exact dimensions (`S x K`, `S x S`), dtypes (`f64le`, `f32le`), and filenames.
