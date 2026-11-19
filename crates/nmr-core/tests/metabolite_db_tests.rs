//! Integration tests for `nmr_core::metabolite_db`:
//! - Built-in CC0 `BMRB-BMSE` compact aqueous metabolite index (`>= 40` metabolites, valid SMILES)
//! - Fast `< 5 ms` offline identification for L-Lactate, L-Alanine, D-Glucose (`1H` & `13C`)
//! - Multiplicity / scalar `J`-coupling discrimination & `STOCSY` correlated cluster (`r >= 0.85`)
//! - Strict HTTPS host allowlist SSRF rejection (`api.bmrb.io`, `np-mrd.org` only)
//! - Deterministic mock desktop online query (`BMRB v2` + `NP-MRD`) and fail-closed offline `Err`
//! - 1D reference spectrum synthesis (`synthesize_reference_spectrum_1d`) for overlay plotting.

use std::collections::BTreeSet;
use std::time::Instant;

use nmr_core::metabolite_db::{
    build_bmrb_v2_shift_search_url, build_npmrd_shift_search_url,
    extract_stocsy_cluster_query_peaks, lookup_offline_metabolite, offline_metabolite_count,
    offline_metabolite_database, query_offline_metabolites, synthesize_reference_spectrum_1d,
    validate_metabolite_online_url, MetaboliteNucleus, MetaboliteSource, OnlineHost,
    PeakMultiplicity, QueryPeak, DEFAULT_STOCSY_CLUSTER_MIN_R,
};
use nmr_core::metabolomics::{
    compute_stocsy_at_ppm, generate_demo_biofluid_cohort, run_metabolomics_pipeline,
    MetabolomicsPipelineConfig,
};
use nmr_core::octet_mol::parse_smiles;

#[test]
fn test_offline_database_completeness_uniqueness_and_octet_smiles_validity() {
    let db = offline_metabolite_database();
    assert!(
        db.len() >= 40,
        "expected >= 40 canonical aqueous metabolites, got {}",
        db.len()
    );
    assert_eq!(db.len(), offline_metabolite_count());

    let mut seen_ids = BTreeSet::new();
    for entry in &db {
        assert!(
            seen_ids.insert(entry.bmse_id.clone()),
            "duplicate bmse_id: {}",
            entry.bmse_id
        );
        assert!(
            entry.bmse_id.starts_with("bmse"),
            "expected bmse* accession, got {}",
            entry.bmse_id
        );
        assert!(!entry.name.trim().is_empty());
        assert!(!entry.formula.trim().is_empty());
        assert!(!entry.smiles.trim().is_empty());
        assert_eq!(entry.solvent, "D2O");
        assert!(
            (6.8..=7.6).contains(&entry.ph),
            "unexpected aqueous pH {} for {}",
            entry.ph,
            entry.name
        );
        assert_eq!(entry.source, MetaboliteSource::BmrbBmseOffline);
        assert!(
            !entry.peaks_1h.is_empty(),
            "missing 1H peaks for {}",
            entry.name
        );
        assert!(
            !entry.shifts_13c.is_empty(),
            "missing 13C shifts for {}",
            entry.name
        );
        for p in &entry.peaks_1h {
            assert!(p.ppm.is_finite() && p.ppm > 0.0 && p.ppm < 12.0);
            assert!(p.n_protons >= 1);
        }
        for c in &entry.shifts_13c {
            assert!(c.ppm.is_finite() && c.ppm > 0.0 && c.ppm < 230.0);
            assert!(c.n_carbons >= 1);
        }
        // Every SMILES must parse cleanly in Octet-Draw for 1-click "Send to Octet".
        let mol = parse_smiles(&entry.smiles).unwrap_or_else(|e| {
            panic!(
                "SMILES {:?} for {} ({}) failed octet_mol::parse_smiles: {e}",
                entry.smiles, entry.name, entry.bmse_id
            )
        });
        assert!(
            !mol.atoms.is_empty(),
            "parsed Octet molecule for {} was empty",
            entry.name
        );
    }

    // Spot-check canonical biofluid entries requested by spec.
    for required in [
        "L-Lactate",
        "L-Alanine",
        "alpha-D-Glucose",
        "beta-D-Glucose",
        "Acetate",
        "Citrate",
        "Succinate",
        "Creatine",
        "Creatinine",
        "Glycine",
        "Taurine",
        "Choline",
        "TMAO",
        "L-Glutamine",
        "L-Glutamate",
        "L-Valine",
        "L-Leucine",
        "L-Isoleucine",
        "L-Tyrosine",
        "L-Phenylalanine",
        "L-Histidine",
        "Formate",
        "Pyruvate",
        "Fumarate",
        "Ethanol",
        "Methanol",
        "Acetone",
        "Betaine",
        "Myo-inositol",
        "L-Serine",
        "L-Threonine",
        "L-Proline",
        "L-Lysine",
        "L-Arginine",
        "L-Aspartate",
        "L-Asparagine",
        "L-Methionine",
        "L-Tryptophan",
        "Hippurate",
        "3-Hydroxybutyrate",
        "Acetoacetate",
    ] {
        assert!(
            lookup_offline_metabolite(required).is_some(),
            "required metabolite {required:?} missing from offline index"
        );
    }
}

#[test]
fn test_offline_identification_lactate_alanine_glucose_1h_and_13c_fast() {
    // 1. L-Lactate 1H (with small experimental jitter + latency check < 5 ms)
    let lac_q = vec![
        QueryPeak::with_multiplet(1.332, PeakMultiplicity::Doublet, vec![6.9]),
        QueryPeak::with_multiplet(4.108, PeakMultiplicity::Quartet, vec![6.9]),
    ];
    let t0 = Instant::now();
    let lac_hits = query_offline_metabolites(&lac_q, MetaboliteNucleus::H1, 0.03, 5);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(
        elapsed_ms < 5.0,
        "offline query took {elapsed_ms:.3} ms (must be < 5 ms)"
    );
    assert!(!lac_hits.is_empty());
    assert_eq!(lac_hits[0].entry.name, "L-Lactate");
    assert_eq!(lac_hits[0].entry.bmse_id, "bmse000036");
    assert_eq!(lac_hits[0].matched_ref_peaks, 2);
    assert_eq!(lac_hits[0].total_ref_peaks, 2);
    assert!(lac_hits[0].score > 0.95, "score={}", lac_hits[0].score);

    // 2. L-Alanine 1H
    let ala_q = vec![
        QueryPeak::with_multiplet(1.476, PeakMultiplicity::Doublet, vec![7.2]),
        QueryPeak::with_multiplet(3.780, PeakMultiplicity::Quartet, vec![7.2]),
    ];
    let ala_hits = query_offline_metabolites(&ala_q, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(ala_hits[0].entry.name, "L-Alanine");
    assert_eq!(ala_hits[0].entry.bmse_id, "bmse000028");
    assert!(ala_hits[0].score > 0.95);

    // 3. alpha-D-Glucose 1H
    let aglc_q = vec![
        QueryPeak::from_ppm(5.233),
        QueryPeak::from_ppm(3.833),
        QueryPeak::from_ppm(3.710),
        QueryPeak::from_ppm(3.533),
        QueryPeak::from_ppm(3.408),
    ];
    let aglc_hits = query_offline_metabolites(&aglc_q, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(aglc_hits[0].entry.name, "alpha-D-Glucose");
    assert_eq!(aglc_hits[0].matched_ref_peaks, 5);
    assert_eq!(aglc_hits[0].total_ref_peaks, 5);
    assert!(aglc_hits[0].score > 0.98);

    // 4. beta-D-Glucose 1H
    let bglc_q = vec![
        QueryPeak::from_ppm(4.642),
        QueryPeak::from_ppm(3.895),
        QueryPeak::from_ppm(3.720),
        QueryPeak::from_ppm(3.475),
        QueryPeak::from_ppm(3.400),
        QueryPeak::from_ppm(3.242),
    ];
    let bglc_hits = query_offline_metabolites(&bglc_q, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(bglc_hits[0].entry.name, "beta-D-Glucose");
    assert_eq!(bglc_hits[0].matched_ref_peaks, 6);
    assert!(bglc_hits[0].score > 0.98);

    // 5. 13C identification for L-Lactate and L-Alanine
    let lac_c13 = vec![
        QueryPeak::from_ppm(22.9),
        QueryPeak::from_ppm(71.1),
        QueryPeak::from_ppm(185.0),
    ];
    let lac_c13_hits = query_offline_metabolites(&lac_c13, MetaboliteNucleus::C13, 0.50, 5);
    assert_eq!(lac_c13_hits[0].entry.name, "L-Lactate");
    assert_eq!(lac_c13_hits[0].matched_ref_peaks, 3);
    assert!(lac_c13_hits[0].score > 0.95);

    let ala_c13 = vec![
        QueryPeak::from_ppm(18.9),
        QueryPeak::from_ppm(53.3),
        QueryPeak::from_ppm(178.5),
    ];
    let ala_c13_hits = query_offline_metabolites(&ala_c13, MetaboliteNucleus::C13, 0.50, 5);
    assert_eq!(ala_c13_hits[0].entry.name, "L-Alanine");
    assert!(ala_c13_hits[0].score > 0.98);
}

#[test]
fn test_multiplicity_and_j_consistency_disambiguates_overlapping_shifts() {
    // Taurine triplet @ 3.260 (J=6.6) + 3.425 (J=6.6) vs TMAO singlet @ 3.266
    let taurine_q = vec![
        QueryPeak::with_multiplet(3.261, PeakMultiplicity::Triplet, vec![6.6]),
        QueryPeak::with_multiplet(3.426, PeakMultiplicity::Triplet, vec![6.6]),
    ];
    let hits = query_offline_metabolites(&taurine_q, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(hits[0].entry.name, "Taurine");
    assert!(hits[0].mult_j_score > 0.98);

    // Single singlet peak @ 3.266 -> TMAO outranks Taurine
    let tmao_q = vec![QueryPeak::with_multiplet(
        3.266,
        PeakMultiplicity::Singlet,
        vec![],
    )];
    let hits_tmao = query_offline_metabolites(&tmao_q, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(hits_tmao[0].entry.name, "TMAO");
}

#[test]
fn test_stocsy_correlated_cluster_identification_synthetic_and_cohort() {
    // 1. Direct STOCSY cluster filtering: L-Lactate peaks r >= 0.85, distractors r < 0.85
    let stocsy_cluster = vec![
        QueryPeak::from_stocsy_cluster(1.331, 1.00),
        QueryPeak::from_stocsy_cluster(4.109, 0.96),
        QueryPeak::from_stocsy_cluster(1.918, 0.28), // Acetate noise (r < 0.85)
        QueryPeak::from_stocsy_cluster(2.405, 0.14), // Succinate noise (r < 0.85)
        QueryPeak::from_stocsy_cluster(8.448, -0.35), // Formate anticorrelated
    ];
    let hits = query_offline_metabolites(&stocsy_cluster, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(hits[0].entry.name, "L-Lactate");
    assert_eq!(hits[0].matched_ref_peaks, 2);
    assert!(
        (hits[0].precision - 1.0).abs() < 1e-6,
        "low-r distractors should be filtered out, got precision={}",
        hits[0].precision
    );

    // 2. End-to-end from 24-sample synthetic biofluid cohort STOCSY at L-Lactate 1.330 ppm
    let cohort = generate_demo_biofluid_cohort(20261115);
    let cfg = MetabolomicsPipelineConfig {
        n_permutations: 10,
        ..Default::default()
    };
    let study = run_metabolomics_pipeline(&cohort, &cfg);
    let stocsy = compute_stocsy_at_ppm(1.330, &study).expect("STOCSY at 1.330 ppm");
    let cluster_peaks =
        extract_stocsy_cluster_query_peaks(&study.buckets, &stocsy, DEFAULT_STOCSY_CLUSTER_MIN_R);
    assert!(
        cluster_peaks.len() >= 2,
        "expected >= 2 correlated Lactate buckets (1.33 + 4.11 ppm), got {}",
        cluster_peaks.len()
    );
    let cohort_hits = query_offline_metabolites(&cluster_peaks, MetaboliteNucleus::H1, 0.04, 5);
    assert_eq!(cohort_hits[0].entry.name, "L-Lactate");
    assert_eq!(cohort_hits[0].entry.bmse_id, "bmse000036");
}

#[test]
fn test_strict_https_host_allowlist_ssrf_rejection() {
    // Strictly permitted hosts
    assert_eq!(
        validate_metabolite_online_url(
            "https://api.bmrb.io/v2/search/chemical_shifts?shift_val=1.3300&threshold=0.0300&atom_type=H&database=metabolomics"
        ),
        Ok(OnlineHost::BmrbApiV2)
    );
    assert_eq!(
        validate_metabolite_online_url(
            "https://np-mrd.org/api/v1/spectra/search?nucleus=1H&shifts=1.3300,4.1100&tolerance=0.0300"
        ),
        Ok(OnlineHost::NpMrd)
    );
    assert!(build_bmrb_v2_shift_search_url(1.33, MetaboliteNucleus::H1, 0.03).is_ok());
    assert!(build_npmrd_shift_search_url(
        &[QueryPeak::from_ppm(1.33)],
        MetaboliteNucleus::H1,
        0.03
    )
    .is_ok());

    // SSRF / scheme / lookalike / port / userinfo / traversal attacks MUST be rejected
    let forbidden = [
        "http://api.bmrb.io/v2/search/chemical_shifts",
        "http://np-mrd.org/api/v1/spectra/search",
        "file:///etc/passwd",
        "ftp://api.bmrb.io/v2/x",
        "https://api.bmrb.io.evil.example/v2/search",
        "https://evil-api.bmrb.io/v2/search",
        "https://np-mrd.org.attacker.net/api/v1/spectra/search",
        "https://www.np-mrd.org/api/v1/spectra/search",
        "https://bmrb.io/v2/search",
        "https://hmdb.ca/spectra/nmr_one_d/1",
        "https://user@api.bmrb.io/v2/search",
        "https://api.bmrb.io:8443/v2/search",
        "https://np-mrd.org:443/api/v1/spectra/search",
        "https://127.0.0.1/v2/search",
        "https://[::1]/v2/search",
        "https://169.254.169.254/latest/meta-data/",
        "https://api.bmrb.io/v2/../admin",
        "https://api.bmrb.io/",
        "https://api.bmrb.io",
        "https://api.bmrb.io/v2/search\r\nX-Evil: 1",
        "https://api.bmrb.io/v2/search\\foo",
    ];
    for url in forbidden {
        assert!(
            validate_metabolite_online_url(url).is_err(),
            "security regression: URL {url:?} should have been rejected by host allowlist"
        );
    }
}

#[test]
fn test_synthesize_reference_spectrum_1d_1h_and_13c() {
    let lac = lookup_offline_metabolite("L-Lactate").expect("L-Lactate in DB");

    // 1H reference spectrum
    let spec_1h = synthesize_reference_spectrum_1d(&lac, MetaboliteNucleus::H1, 600.0, 2048);
    assert_eq!(spec_1h.bmse_id, "bmse000036");
    assert_eq!(spec_1h.ppm_grid.len(), 2048);
    assert_eq!(spec_1h.intensities.len(), 2048);
    // Descending ppm axis
    assert!(spec_1h.ppm_grid[0] > spec_1h.ppm_grid[2047]);
    // 2 lines (doublet) + 4 lines (quartet) = 6 multiplet sticks
    assert_eq!(spec_1h.sticks.len(), 6);
    let max_i = spec_1h.intensities.iter().copied().fold(0.0_f64, f64::max);
    assert!((max_i - 1.0).abs() < 1e-6, "expected max=1.0, got {max_i}");
    // Strong signal near 1.330 ppm and 4.110 ppm, near-zero at 7.0 ppm
    let val_at = |target: f64| -> f64 {
        spec_1h
            .ppm_grid
            .iter()
            .enumerate()
            .filter(|(_, &p)| (p - target).abs() < 0.025)
            .map(|(i, _)| spec_1h.intensities[i])
            .fold(0.0_f64, f64::max)
    };
    assert!(
        val_at(1.330) > 0.70,
        "expected strong doublet near 1.330 ppm"
    );
    assert!(val_at(4.110) > 0.05, "expected quartet near 4.110 ppm");
    assert!(val_at(7.000) < 0.01, "expected baseline near 7.000 ppm");

    // 13C reference spectrum
    let spec_13c = synthesize_reference_spectrum_1d(&lac, MetaboliteNucleus::C13, 150.9, 1024);
    assert_eq!(spec_13c.ppm_grid.len(), 1024);
    assert_eq!(spec_13c.sticks.len(), 3);
    assert!(spec_13c.ppm_grid[0] >= 200.0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_online_query_mock_transport_and_fail_closed_offline() {
    use nmr_core::metabolite_db::{
        query_online_bmrb_and_npmrd_with_transport, OnlineMetaboliteTransport, OnlineQuerySources,
    };

    struct MockOkTransport {
        seen_urls: Vec<String>,
    }
    impl OnlineMetaboliteTransport for MockOkTransport {
        fn get_https_json(&mut self, url: &str) -> Result<String, String> {
            let host = validate_metabolite_online_url(url)?;
            self.seen_urls.push(url.to_string());
            match host {
                OnlineHost::BmrbApiV2 => Ok(r#"{
                    "columns": ["Entry_ID", "Comp_ID", "Atom_ID", "Val"],
                    "data": [
                        ["bmse000036", "L-Lactate", "H3", 1.330],
                        ["bmse000036", "L-Lactate", "H2", 4.110]
                    ]
                }"#
                .to_string()),
                OnlineHost::NpMrd => Ok(r#"{
                    "results": [
                        {
                            "np_mrd_id": "NP0000123",
                            "name": "L-Lactic acid",
                            "formula": "C3H6O3",
                            "smiles": "C[C@H](O)C(=O)O",
                            "peaks": [1.331, 4.109]
                        }
                    ]
                }"#
                .to_string()),
            }
        }
    }

    let query = vec![QueryPeak::from_ppm(1.330), QueryPeak::from_ppm(4.110)];
    let mut ok_t = MockOkTransport {
        seen_urls: Vec::new(),
    };
    let hits = query_online_bmrb_and_npmrd_with_transport(
        &query,
        MetaboliteNucleus::H1,
        0.03,
        OnlineQuerySources::all(),
        &mut ok_t,
    )
    .expect("mock online query succeeds");
    assert!(ok_t
        .seen_urls
        .iter()
        .any(|u| u.starts_with("https://api.bmrb.io/v2/")));
    assert!(ok_t
        .seen_urls
        .iter()
        .any(|u| u.starts_with("https://np-mrd.org/")));
    assert!(
        hits.iter()
            .any(|h| h.entry.source == MetaboliteSource::BmrbOnline
                && h.entry.bmse_id == "bmse000036")
    );
    assert!(
        hits.iter()
            .any(|h| h.entry.source == MetaboliteSource::NpMrdOnline
                && h.entry.bmse_id == "NP0000123")
    );

    // Fail-closed offline test
    struct OfflineTransport;
    impl OnlineMetaboliteTransport for OfflineTransport {
        fn get_https_json(&mut self, url: &str) -> Result<String, String> {
            let _ = validate_metabolite_online_url(url)?;
            Err("network unreachable (offline)".to_string())
        }
    }
    let mut off_t = OfflineTransport;
    let err = query_online_bmrb_and_npmrd_with_transport(
        &query,
        MetaboliteNucleus::H1,
        0.03,
        OnlineQuerySources::all(),
        &mut off_t,
    )
    .expect_err("offline query must fail closed with Err");
    assert!(err.contains("offline") || err.contains("failed"));
}
