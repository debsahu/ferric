//! Integration tests for `nmr_core::metabolite_db`:
//! - Built-in live-verified `BMRB-BMSE` compact aqueous metabolite index (`>= 40` metabolites, valid SMILES)
//! - Fast `< 5 ms` offline identification for L-(+) Lactic acid (`bmse000208`), L-Alanine (`bmse000028`),
//!   D-(+)-Glucose (`bmse000015`) (`1H` & `13C`)
//! - Multi-peak mixture specificity ranking (`L-Isoleucine 5/6` strictly above 1-peak singlets)
//! - Strict 1-to-1 peak matching and tolerance cap (`tol <= 2.0 ppm`)
//! - Multiplicity / scalar `J`-coupling discrimination & `STOCSY` correlated cluster (`r >= 0.85`)
//! - Strict HTTPS host allowlist SSRF rejection (`api.bmrb.io`, `np-mrd.org` only)
//! - Honest `BMRB v2` REST API query (`shift=`, real compound names via `/v2/entry/<id>`, never
//!   overwriting online hits from the offline table, no fake partial `recall = 1.0`) and honest
//!   `NP-MRD` web search URL (`https://np-mrd.org/`), plus fail-closed offline `Err`
//! - 1D reference spectrum synthesis (`synthesize_reference_spectrum_1d`) for overlay plotting.

use std::collections::BTreeSet;
use std::time::Instant;

use nmr_core::metabolite_db::{
    build_bmrb_v2_entry_metadata_url, build_bmrb_v2_shift_search_url,
    extract_stocsy_cluster_query_peaks, lookup_offline_metabolite, npmrd_web_search_url,
    offline_metabolite_count, offline_metabolite_database, parse_bmrb_v2_entry_metadata_and_score,
    parse_bmrb_v2_response, query_offline_metabolites, synthesize_reference_spectrum_1d,
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
        "expected >= 40 live-verified BMRB-BMSE aqueous metabolites, got {}",
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
            entry.bmse_id.starts_with("bmse")
                && entry.bmse_id.len() == 10
                && entry.bmse_id[4..].bytes().all(|b| b.is_ascii_digit()),
            "expected canonical bmse000XXX accession, got {}",
            entry.bmse_id
        );
        assert!(!entry.name.trim().is_empty());
        assert!(!entry.formula.trim().is_empty());
        assert!(!entry.smiles.trim().is_empty());
        assert_eq!(entry.solvent, "D2O");
        assert!(
            (1.0..=12.0).contains(&entry.ph),
            "unexpected aqueous pH {} for {}",
            entry.ph,
            entry.name
        );
        assert_eq!(entry.source, MetaboliteSource::BmrbBmseOffline);
        assert!(
            !entry.source.label().contains("CC0"),
            "BMRB source label must not claim CC0: {}",
            entry.source.label()
        );
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

    // Spot-check live-verified BMRB-BMSE accessions and exact compound names.
    for (alias, expected_id) in [
        ("L-Lactate", "bmse000208"),
        ("L-Alanine", "bmse000028"),
        ("D-Glucose", "bmse000015"),
        ("Acetate", "bmse000191"),
        ("Citrate", "bmse000076"),
        ("Succinate", "bmse000183"),
        ("Creatine", "bmse000078"),
        ("Creatinine", "bmse000155"),
        ("Glycine", "bmse000089"),
        ("Taurine", "bmse000120"),
        ("Choline", "bmse000285"),
        ("TMAO", "bmse000426"),
        ("L-Glutamine", "bmse000038"),
        ("L-Glutamate", "bmse000037"),
        ("L-Valine", "bmse000052"),
        ("L-Leucine", "bmse000042"),
        ("L-Isoleucine", "bmse000041"),
        ("L-Tyrosine", "bmse000051"),
        ("L-Phenylalanine", "bmse000045"),
        ("L-Histidine", "bmse000039"),
        ("Formate", "bmse000203"),
        ("Pyruvate", "bmse000112"),
        ("Fumarate", "bmse000083"),
        ("Ethanol", "bmse000297"),
        ("Methanol", "bmse000294"),
        ("Betaine", "bmse000069"),
        ("Myo-inositol", "bmse000103"),
        ("L-Serine", "bmse000048"),
        ("L-Threonine", "bmse000049"),
        ("L-Proline", "bmse000047"),
        ("L-Lysine", "bmse000043"),
        ("L-Arginine", "bmse000029"),
        ("L-Aspartate", "bmse000031"),
        ("L-Asparagine", "bmse000030"),
        ("L-Methionine", "bmse000044"),
        ("L-Tryptophan", "bmse000050"),
        ("Hippurate", "bmse000408"),
        ("3-Hydroxybutyrate", "bmse000161"),
        ("Glycerol", "bmse000184"),
        ("L-Cysteine", "bmse000034"),
        ("Uracil", "bmse000187"),
    ] {
        let entry = lookup_offline_metabolite(alias)
            .unwrap_or_else(|| panic!("required metabolite {alias:?} missing from offline index"));
        assert_eq!(
            entry.bmse_id, expected_id,
            "unexpected bmse_id for {alias:?}"
        );
    }

    // Verify Hippuric acid (bmse000408) CH2 in D2O is a Singlet (amide NH exchanges with D2O).
    let hipp = lookup_offline_metabolite("bmse000408").expect("Hippuric acid");
    let ch2 = hipp
        .peaks_1h
        .iter()
        .find(|p| (p.ppm - 3.944).abs() < 0.01)
        .expect("Hippuric acid CH2 peak at 3.944 ppm");
    assert_eq!(ch2.multiplicity, PeakMultiplicity::Singlet);
}

#[test]
fn test_offline_identification_lactate_alanine_glucose_1h_and_13c_fast() {
    // 1. L-(+) Lactic acid (bmse000208) 1H (with small experimental jitter + latency check < 5 ms)
    let lac_q = vec![
        QueryPeak::with_multiplet(1.320, PeakMultiplicity::Doublet, vec![7.2]),
        QueryPeak::with_multiplet(4.100, PeakMultiplicity::Quartet, vec![7.0]),
    ];
    let t0 = Instant::now();
    let lac_hits = query_offline_metabolites(&lac_q, MetaboliteNucleus::H1, 0.03, 5);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(
        elapsed_ms < 5.0,
        "offline query took {elapsed_ms:.3} ms (must be < 5 ms)"
    );
    assert!(!lac_hits.is_empty());
    assert_eq!(lac_hits[0].entry.bmse_id, "bmse000208");
    assert_eq!(lac_hits[0].entry.name, "L-(+) Lactic acid");
    assert_eq!(lac_hits[0].matched_ref_peaks, 2);
    assert_eq!(lac_hits[0].total_ref_peaks, 2);
    assert!(lac_hits[0].score > 0.95, "score={}", lac_hits[0].score);

    // 2. L-Alanine (bmse000028) 1H
    let ala_q = vec![
        QueryPeak::with_multiplet(1.471, PeakMultiplicity::Doublet, vec![7.2]),
        QueryPeak::with_multiplet(3.771, PeakMultiplicity::Quartet, vec![7.2]),
    ];
    let ala_hits = query_offline_metabolites(&ala_q, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(ala_hits[0].entry.bmse_id, "bmse000028");
    assert_eq!(ala_hits[0].entry.name, "L-Alanine");
    assert!(ala_hits[0].score > 0.95);

    // 3. D-(+)-Glucose (bmse000015) 1H
    let glc_q = vec![
        QueryPeak::from_ppm(3.232),
        QueryPeak::from_ppm(3.401),
        QueryPeak::from_ppm(3.459),
        QueryPeak::from_ppm(3.517),
        QueryPeak::from_ppm(3.726),
        QueryPeak::from_ppm(3.828),
        QueryPeak::from_ppm(3.887),
        QueryPeak::from_ppm(4.636),
        QueryPeak::from_ppm(5.222),
    ];
    let glc_hits = query_offline_metabolites(&glc_q, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(glc_hits[0].entry.bmse_id, "bmse000015");
    assert_eq!(glc_hits[0].entry.name, "D-(+)-Glucose");
    assert_eq!(glc_hits[0].matched_ref_peaks, 9);
    assert_eq!(glc_hits[0].total_ref_peaks, 9);
    assert!(glc_hits[0].score > 0.98);

    // 4. 13C identification for L-(+) Lactic acid and L-Alanine
    let lac_c13 = vec![
        QueryPeak::from_ppm(22.81),
        QueryPeak::from_ppm(71.23),
        QueryPeak::from_ppm(185.32),
    ];
    let lac_c13_hits = query_offline_metabolites(&lac_c13, MetaboliteNucleus::C13, 0.50, 5);
    assert_eq!(lac_c13_hits[0].entry.bmse_id, "bmse000208");
    assert_eq!(lac_c13_hits[0].matched_ref_peaks, 3);
    assert!(lac_c13_hits[0].score > 0.95);

    let ala_c13 = vec![
        QueryPeak::from_ppm(18.83),
        QueryPeak::from_ppm(53.20),
        QueryPeak::from_ppm(178.56),
    ];
    let ala_c13_hits = query_offline_metabolites(&ala_c13, MetaboliteNucleus::C13, 0.50, 5);
    assert_eq!(ala_c13_hits[0].entry.bmse_id, "bmse000028");
    assert!(ala_c13_hits[0].score > 0.98);

    // 5. Absurd tolerance (> 2.0 ppm) must be rejected cleanly
    assert!(query_offline_metabolites(&lac_q, MetaboliteNucleus::H1, 1e300, 5).is_empty());
    assert!(query_offline_metabolites(&lac_q, MetaboliteNucleus::H1, 2.5, 5).is_empty());
}

#[test]
fn test_mixture_scoring_ranks_multipeak_isoleucine_above_singlet_coincidences() {
    // Regression test for 171.metab:P1-4:
    // Dense 20-peak mixture containing 5 of 6 L-Isoleucine peaks (bmse000041) plus
    // exact 1-peak singlets for Acetic acid (1.905), Formate (8.447), Methanol (3.346),
    // Glycine (3.545), and 11 additional mixture peaks.
    let mixture = vec![
        // 5/6 L-Isoleucine peaks (omitting 1.458)
        QueryPeak::from_ppm(0.927),
        QueryPeak::from_ppm(0.998),
        QueryPeak::from_ppm(1.249),
        QueryPeak::from_ppm(1.969),
        QueryPeak::from_ppm(3.657),
        // Exact 1-peak singlets
        QueryPeak::from_ppm(1.905), // Acetic acid
        QueryPeak::from_ppm(8.447), // Formate
        QueryPeak::from_ppm(3.346), // Methanol
        QueryPeak::from_ppm(3.545), // Glycine
        // Additional background mixture peaks
        QueryPeak::from_ppm(0.810),
        QueryPeak::from_ppm(1.110),
        QueryPeak::from_ppm(1.620),
        QueryPeak::from_ppm(2.220),
        QueryPeak::from_ppm(2.810),
        QueryPeak::from_ppm(4.350),
        QueryPeak::from_ppm(5.100),
        QueryPeak::from_ppm(6.150),
        QueryPeak::from_ppm(6.720),
        QueryPeak::from_ppm(7.020),
        QueryPeak::from_ppm(8.950),
    ];

    let hits = query_offline_metabolites(&mixture, MetaboliteNucleus::H1, 0.03, 25);
    let ile_pos = hits
        .iter()
        .position(|h| h.entry.bmse_id == "bmse000041")
        .expect("L-Isoleucine must appear in mixture hits");
    let ile = &hits[ile_pos];
    assert_eq!(ile.matched_ref_peaks, 5);
    assert_eq!(ile.total_ref_peaks, 6);

    for singlet_id in ["bmse000191", "bmse000203", "bmse000294", "bmse000089"] {
        let s_pos = hits
            .iter()
            .position(|h| h.entry.bmse_id == singlet_id)
            .unwrap_or_else(|| panic!("singlet {singlet_id} should appear in hits"));
        assert!(
            ile.score > hits[s_pos].score + 0.15 && ile_pos < s_pos,
            "L-Isoleucine (5/6, rank={}, score={:.3}) must rank strictly above 1-peak singlet {} (rank={}, score={:.3})",
            ile_pos + 1,
            ile.score,
            singlet_id,
            s_pos + 1,
            hits[s_pos].score
        );
    }

    // Strict 1-to-1 matching: a single query peak must never claim > 1 reference peak.
    let single_q = vec![QueryPeak::from_ppm(1.007)];
    let single_hits = query_offline_metabolites(&single_q, MetaboliteNucleus::H1, 0.03, 10);
    for h in &single_hits {
        assert_eq!(
            h.matched_ref_peaks, 1,
            "single query peak must match at most 1 reference peak for {} (got {})",
            h.entry.name, h.matched_ref_peaks
        );
    }
}

#[test]
fn test_multiplicity_and_j_consistency_disambiguates_overlapping_shifts() {
    // Taurine (bmse000120) triplet @ 3.246 (J=6.8) + 3.410 (J=6.8) vs TMAO (bmse000426) singlet @ 3.249
    let taurine_q = vec![
        QueryPeak::with_multiplet(3.246, PeakMultiplicity::Triplet, vec![6.8]),
        QueryPeak::with_multiplet(3.410, PeakMultiplicity::Triplet, vec![6.8]),
    ];
    let hits = query_offline_metabolites(&taurine_q, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(hits[0].entry.bmse_id, "bmse000120");
    assert_eq!(hits[0].entry.name, "Taurine");
    assert!(hits[0].mult_j_score > 0.98);

    // Single singlet peak @ 3.249 -> Trimethylamine N-oxide (bmse000426) outranks Taurine
    let tmao_q = vec![QueryPeak::with_multiplet(
        3.249,
        PeakMultiplicity::Singlet,
        vec![],
    )];
    let hits_tmao = query_offline_metabolites(&tmao_q, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(hits_tmao[0].entry.bmse_id, "bmse000426");
    assert_eq!(hits_tmao[0].entry.name, "Trimethylamine N-oxide");
}

#[test]
fn test_stocsy_correlated_cluster_identification_synthetic_and_cohort() {
    // 1. Direct STOCSY cluster filtering: L-Lactate peaks r >= 0.85, distractors r < 0.85
    let stocsy_cluster = vec![
        QueryPeak::from_stocsy_cluster(1.319, 1.00),
        QueryPeak::from_stocsy_cluster(4.101, 0.96),
        QueryPeak::from_stocsy_cluster(1.905, 0.28), // Acetic acid noise (r < 0.85)
        QueryPeak::from_stocsy_cluster(2.391, 0.14), // Succinic acid noise (r < 0.85)
        QueryPeak::from_stocsy_cluster(8.447, -0.35), // Formate anticorrelated
    ];
    let hits = query_offline_metabolites(&stocsy_cluster, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(hits[0].entry.bmse_id, "bmse000208");
    assert_eq!(hits[0].entry.name, "L-(+) Lactic acid");
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
    assert_eq!(
        cohort_hits[0].entry.bmse_id,
        "bmse000208",
        "cluster_peaks={cluster_peaks:?} hits={:?}",
        cohort_hits
            .iter()
            .map(|h| (
                &h.entry.bmse_id,
                &h.entry.name,
                h.score,
                h.matched_ref_peaks,
                h.total_ref_peaks,
                h.mean_abs_error_ppm
            ))
            .collect::<Vec<_>>()
    );
    assert_eq!(cohort_hits[0].entry.name, "L-(+) Lactic acid");
}

#[test]
fn test_strict_https_host_allowlist_ssrf_rejection_and_honest_urls() {
    // Strictly permitted hosts & live BMRB v2 `shift=` query parameter
    let bmrb_url =
        build_bmrb_v2_shift_search_url(1.318, MetaboliteNucleus::H1, 0.03).expect("BMRB URL");
    assert!(
        bmrb_url.contains("?shift=1.3180&") && !bmrb_url.contains("shift_val="),
        "BMRB v2 URL must use `shift=` (not `shift_val=`): {bmrb_url}"
    );
    assert_eq!(
        validate_metabolite_online_url(&bmrb_url),
        Ok(OnlineHost::BmrbApiV2)
    );

    let meta_url = build_bmrb_v2_entry_metadata_url("bmse000208").expect("entry metadata URL");
    assert_eq!(
        validate_metabolite_online_url(&meta_url),
        Ok(OnlineHost::BmrbApiV2)
    );
    assert!(build_bmrb_v2_entry_metadata_url("../evil").is_err());
    assert!(build_bmrb_v2_shift_search_url(1.318, MetaboliteNucleus::H1, 2.5).is_err());

    // Honest NP-MRD web portal URL (no invented /api/v1/spectra/search 404 endpoint)
    assert_eq!(npmrd_web_search_url(), "https://np-mrd.org/");
    assert_eq!(
        validate_metabolite_online_url("https://np-mrd.org/searches"),
        Ok(OnlineHost::NpMrd)
    );

    // SSRF / scheme / lookalike / port / userinfo / traversal attacks MUST be rejected
    let forbidden = [
        "http://api.bmrb.io/v2/search/chemical_shifts",
        "http://np-mrd.org/searches",
        "file:///etc/passwd",
        "ftp://api.bmrb.io/v2/x",
        "https://api.bmrb.io.evil.example/v2/search",
        "https://evil-api.bmrb.io/v2/search",
        "https://np-mrd.org.attacker.net/searches",
        "https://www.np-mrd.org/searches",
        "https://bmrb.io/v2/search",
        "https://hmdb.ca/spectra/nmr_one_d/1",
        "https://user@api.bmrb.io/v2/search",
        "https://api.bmrb.io:8443/v2/search",
        "https://np-mrd.org:443/searches",
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
fn test_parse_bmrb_v2_never_overwrites_from_offline_or_inflates_partial_recall() {
    // Regression test for 171.metab:P0-2 and 171.metab:P1-3:
    // Realistic BMRB v2 /search/chemical_shifts response uses Comp_ID = "1" and returns
    // only the matching methyl atoms (H7/H8/H9 at 1.318 ppm).
    let raw_search_json = r#"{
        "columns": [
            "Atom_chem_shift.Entry_ID",
            "Atom_chem_shift.Comp_ID",
            "Atom_chem_shift.Atom_ID",
            "Atom_chem_shift.Val"
        ],
        "data": [
            ["bmse000208", "1", "H7", 1.318],
            ["bmse000208", "1", "H8", 1.318],
            ["bmse000208", "1", "H9", 1.318],
            ["bmse000049", "1", "H10", 1.318]
        ]
    }"#;
    let q = vec![QueryPeak::from_ppm(1.318), QueryPeak::from_ppm(4.101)];
    let hits = parse_bmrb_v2_response(raw_search_json, &q, MetaboliteNucleus::H1, 0.03)
        .expect("parse raw BMRB v2 search response");
    assert_eq!(hits.len(), 2);
    for hit in &hits {
        // Must NEVER treat numeric Comp_ID "1" as a compound name, and must NEVER
        // overwrite name/SMILES from the offline database.
        assert_ne!(hit.entry.name, "1");
        assert_eq!(hit.entry.name, hit.entry.bmse_id);
        assert!(
            hit.entry.smiles.is_empty(),
            "raw search hit must not copy SMILES from offline table"
        );
        // Equivalent methyl atoms (H7/H8/H9) must deduplicate to 1 peak, not 3.
        assert_eq!(hit.matched_ref_peaks, 1);
        // Partial search hit must NEVER report spurious recall == 1.0!
        assert!(
            hit.recall < 0.50,
            "partial search hit {} must not get recall=1.0 (got {})",
            hit.entry.bmse_id,
            hit.recall
        );
    }

    // Entry metadata parser uses ONLY what the online API returns — even if the API returns
    // a distinct name/SMILES for an ID that exists in the offline table.
    let custom_meta_json = r#"{
        "bmse000049": {
            "Chem_comp.Name": ["Custom API Threonine Name"],
            "Chem_comp.Formula": ["C4 H9 N O3"],
            "Chem_comp_SMILES.String": ["CC(O)C(N)C(=O)O"],
            "Atom_chem_shift.Atom_ID": ["H7", "H8", "H9", "H10", "H11"],
            "Atom_chem_shift.Atom_type": ["H", "H", "H", "H", "H"],
            "Atom_chem_shift.Val": ["1.318", "1.318", "1.318", "3.573", "4.241"]
        }
    }"#;
    let thr_hit = parse_bmrb_v2_entry_metadata_and_score(
        "bmse000049",
        custom_meta_json,
        &q,
        MetaboliteNucleus::H1,
        0.03,
    )
    .expect("parse metadata")
    .expect("matched bmse000049");
    assert_eq!(thr_hit.entry.name, "Custom API Threonine Name");
    assert_eq!(thr_hit.entry.smiles, "CC(O)C(N)C(=O)O");
    // Only 1 of 3 distinct proton shifts matched (1.318 matched; 3.573 & 4.241 did not).
    assert_eq!(thr_hit.matched_ref_peaks, 1);
    assert_eq!(thr_hit.total_ref_peaks, 3);
    assert!((thr_hit.recall - (1.0 / 3.0)).abs() < 1e-6);
}

#[test]
fn test_synthesize_reference_spectrum_1d_1h_and_13c() {
    let lac = lookup_offline_metabolite("L-Lactate").expect("L-Lactate in DB");
    assert_eq!(lac.bmse_id, "bmse000208");

    // 1H reference spectrum
    let spec_1h = synthesize_reference_spectrum_1d(&lac, MetaboliteNucleus::H1, 600.0, 2048);
    assert_eq!(spec_1h.bmse_id, "bmse000208");
    assert_eq!(spec_1h.ppm_grid.len(), 2048);
    assert_eq!(spec_1h.intensities.len(), 2048);
    // Descending ppm axis
    assert!(spec_1h.ppm_grid[0] > spec_1h.ppm_grid[2047]);
    // 2 lines (doublet) + 4 lines (quartet) = 6 multiplet sticks
    assert_eq!(spec_1h.sticks.len(), 6);
    let max_i = spec_1h.intensities.iter().copied().fold(0.0_f64, f64::max);
    assert!((max_i - 1.0).abs() < 1e-6, "expected max=1.0, got {max_i}");
    // Strong signal near 1.318 ppm and 4.101 ppm, near-zero at 7.0 ppm
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
        val_at(1.318) > 0.70,
        "expected strong doublet near 1.318 ppm"
    );
    assert!(val_at(4.101) > 0.05, "expected quartet near 4.101 ppm");
    assert!(val_at(7.000) < 0.01, "expected baseline near 7.000 ppm");

    // 13C reference spectrum
    let spec_13c = synthesize_reference_spectrum_1d(&lac, MetaboliteNucleus::C13, 150.9, 1024);
    assert_eq!(spec_13c.ppm_grid.len(), 1024);
    assert_eq!(spec_13c.sticks.len(), 3);
    assert!(spec_13c.ppm_grid[0] >= 200.0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_online_query_mock_transport_bmrb_v2_enrichment_and_fail_closed_offline() {
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
            assert_eq!(
                host,
                OnlineHost::BmrbApiV2,
                "online transport must only call live BMRB v2 API, not invented NP-MRD 404 endpoint"
            );
            if url.contains("/v2/search/chemical_shifts?shift=1.3180") {
                Ok(r#"{
                    "columns": ["Atom_chem_shift.Entry_ID", "Atom_chem_shift.Comp_ID", "Atom_chem_shift.Atom_ID", "Atom_chem_shift.Val"],
                    "data": [
                        ["bmse000208", "1", "H7", 1.318],
                        ["bmse000208", "1", "H8", 1.318],
                        ["bmse000208", "1", "H9", 1.318],
                        ["bmse000049", "1", "H10", 1.318]
                    ]
                }"#
                .to_string())
            } else if url.contains("/v2/search/chemical_shifts?shift=4.1010") {
                Ok(r#"{
                    "columns": ["Atom_chem_shift.Entry_ID", "Atom_chem_shift.Comp_ID", "Atom_chem_shift.Atom_ID", "Atom_chem_shift.Val"],
                    "data": [
                        ["bmse000208", "1", "H10", 4.101]
                    ]
                }"#
                .to_string())
            } else if url.contains("/v2/entry/bmse000208?") {
                Ok(r#"{
                    "bmse000208": {
                        "Chem_comp.Name": ["L-(+) Lactic acid"],
                        "Chem_comp.Formula": ["C3 H6 O3"],
                        "Chem_comp_SMILES.String": ["C[C@@H](C(=O)O)O"],
                        "Atom_chem_shift.Atom_ID": ["C3", "C2", "C1", "H10", "H7", "H8", "H9"],
                        "Atom_chem_shift.Atom_type": ["C", "C", "C", "H", "H", "H", "H"],
                        "Atom_chem_shift.Val": ["185.321", "71.227", "22.812", "4.101", "1.318", "1.318", "1.318"]
                    }
                }"#
                .to_string())
            } else if url.contains("/v2/entry/bmse000049?") {
                Ok(r#"{
                    "bmse000049": {
                        "Chem_comp.Name": ["L-Threonine"],
                        "Chem_comp.Formula": ["C4 H9 N O3"],
                        "Chem_comp_SMILES.String": ["C[C@H]([C@@H](C(=O)O)N)O"],
                        "Atom_chem_shift.Atom_ID": ["H7", "H8", "H9", "H10", "H11"],
                        "Atom_chem_shift.Atom_type": ["H", "H", "H", "H", "H"],
                        "Atom_chem_shift.Val": ["1.318", "1.318", "1.318", "3.573", "4.241"]
                    }
                }"#
                .to_string())
            } else {
                Err(format!("unexpected mock URL: {url}"))
            }
        }
    }

    let query = vec![QueryPeak::from_ppm(1.318), QueryPeak::from_ppm(4.101)];
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
        .any(|u| u.contains("/v2/search/chemical_shifts?shift=")));
    assert!(ok_t
        .seen_urls
        .iter()
        .any(|u| u.contains("/v2/entry/bmse000208?")));
    assert!(!ok_t
        .seen_urls
        .iter()
        .any(|u| u.contains("/api/v1/spectra/search")));

    assert_eq!(hits.len(), 2);
    // Full 2/2 L-(+) Lactic acid must outrank partial 1/3 L-Threonine
    assert_eq!(hits[0].entry.source, MetaboliteSource::BmrbOnline);
    assert_eq!(hits[0].entry.bmse_id, "bmse000208");
    assert_eq!(hits[0].entry.name, "L-(+) Lactic acid");
    assert_eq!(hits[0].matched_ref_peaks, 2);
    assert_eq!(hits[0].total_ref_peaks, 2);
    assert!((hits[0].recall - 1.0).abs() < 1e-6);

    assert_eq!(hits[1].entry.bmse_id, "bmse000049");
    assert_eq!(hits[1].entry.name, "L-Threonine");
    assert_eq!(hits[1].matched_ref_peaks, 1);
    assert_eq!(hits[1].total_ref_peaks, 3);
    assert!((hits[1].recall - (1.0 / 3.0)).abs() < 1e-6);
    assert!(hits[0].score > hits[1].score + 0.30);

    // Requesting npmrd_only must return an honest guidance Err (no 404 call)
    let mut no_net = MockOkTransport {
        seen_urls: Vec::new(),
    };
    let npmrd_err = query_online_bmrb_and_npmrd_with_transport(
        &query,
        MetaboliteNucleus::H1,
        0.03,
        OnlineQuerySources::npmrd_only(),
        &mut no_net,
    )
    .expect_err("npmrd_only should guide user to web URL");
    assert!(npmrd_err.contains("https://np-mrd.org/"));
    assert!(no_net.seen_urls.is_empty());

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

#[test]
fn test_citrate_ab_quartet_peaks_match_live_bmrb_peak_char() {
    // Regression test for 171.metab:P1-B:
    // Live https://api.bmrb.io/v2/entry/bmse000076 saveframe `spectral_peak_HSQC` loop
    // `_Peak_char` (`Spectral_dim_ID = 1`) reports Citrate's two observed 1H AB doublets at
    // 2.531 and 2.661 ppm (|2J_AB| = 15.2 Hz from `spectral_peak_1Hp5` / `spectral_peak_1H_2`),
    // rather than the empty spin-system midpoint valley at 2.579 ppm.
    let cit = lookup_offline_metabolite("Citrate").expect("Citrate (bmse000076) in offline DB");
    assert_eq!(cit.bmse_id, "bmse000076");
    assert_eq!(cit.peaks_1h.len(), 2);
    assert!((cit.peaks_1h[0].ppm - 2.531).abs() < 1e-6);
    assert!((cit.peaks_1h[1].ppm - 2.661).abs() < 1e-6);
    assert_eq!(cit.peaks_1h[0].multiplicity, PeakMultiplicity::Doublet);
    assert_eq!(cit.peaks_1h[1].multiplicity, PeakMultiplicity::Doublet);

    // Querying observed biofluid AB doublet centres (~2.53 / 2.66 ppm) matches Citrate #1 (2/2)
    // both with and without explicit multiplicity/J annotations.
    let q_ppm = vec![QueryPeak::from_ppm(2.530), QueryPeak::from_ppm(2.660)];
    let hits_ppm = query_offline_metabolites(&q_ppm, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(hits_ppm[0].entry.bmse_id, "bmse000076");
    assert_eq!(hits_ppm[0].matched_ref_peaks, 2);
    assert_eq!(hits_ppm[0].total_ref_peaks, 2);
    assert!(hits_ppm[0].score > 0.98, "score={}", hits_ppm[0].score);

    let q_ab = vec![
        QueryPeak::with_multiplet(2.535, PeakMultiplicity::Doublet, vec![15.3]),
        QueryPeak::with_multiplet(2.675, PeakMultiplicity::Doublet, vec![15.3]),
    ];
    let hits_ab = query_offline_metabolites(&q_ab, MetaboliteNucleus::H1, 0.03, 5);
    assert_eq!(hits_ab[0].entry.bmse_id, "bmse000076");
    assert_eq!(hits_ab[0].matched_ref_peaks, 2);
    assert!(hits_ab[0].score > 0.95, "score={}", hits_ab[0].score);

    // Synthesized 1D reference spectrum produces 4 AB quartet sticks with a valley at 2.579 ppm.
    let spec = synthesize_reference_spectrum_1d(&cit, MetaboliteNucleus::H1, 600.0, 4096);
    assert_eq!(spec.sticks.len(), 4);
    let max_near = |target: f64| -> f64 {
        spec.ppm_grid
            .iter()
            .enumerate()
            .filter(|(_, &p)| (p - target).abs() < 0.02)
            .map(|(i, _)| spec.intensities[i])
            .fold(0.0_f64, f64::max)
    };
    assert!(max_near(2.531) > 0.70);
    assert!(max_near(2.661) > 0.70);
    assert!(
        max_near(2.579) < 0.15,
        "2.579 ppm midpoint between Citrate AB doublets should be a valley"
    );
}

#[test]
fn test_demo_mixture_14_metabolites_singlets_outrank_partial_false_positives() {
    // Regression test for 171.metab:P1-A:
    // 1. In the 14-metabolite biofluid demo mixture (13 distinct BMRB-BMSE entries:
    //    D-(+)-Glucose [alpha + beta], L-Valine, Hippurate, Citrate, L-Lactate, L-Alanine,
    //    Creatine, Creatinine, plus the 5 true singlets Formate, Glycine, Acetate, Succinate,
    //    TMAO), local chance-match scoring places ALL 13 true metabolites in the top 13
    //    (including all 5 true singlets inside top-14 and strictly above every partial-match
    //    false positive).
    let demo_multiplet_q = vec![
        QueryPeak::with_multiplet(8.448, PeakMultiplicity::Singlet, vec![]), // Formate
        QueryPeak::with_multiplet(7.832, PeakMultiplicity::Doublet, vec![7.6]), // Hippurate
        QueryPeak::with_multiplet(7.635, PeakMultiplicity::Triplet, vec![7.4]),
        QueryPeak::with_multiplet(7.548, PeakMultiplicity::Triplet, vec![7.6]),
        QueryPeak::with_multiplet(5.233, PeakMultiplicity::Doublet, vec![3.7]), // alpha/beta-Glucose
        QueryPeak::with_multiplet(4.642, PeakMultiplicity::Doublet, vec![7.9]),
        QueryPeak::with_multiplet(4.110, PeakMultiplicity::Quartet, vec![6.9]), // L-Lactate
        QueryPeak::with_multiplet(4.052, PeakMultiplicity::Singlet, vec![]),    // Creatinine
        QueryPeak::with_multiplet(3.968, PeakMultiplicity::Singlet, vec![]),    // Hippurate CH2
        QueryPeak::with_multiplet(3.928, PeakMultiplicity::Singlet, vec![]),    // Creatine CH2
        QueryPeak::with_multiplet(3.895, PeakMultiplicity::DoubletOfDoublets, vec![12.2, 2.1]),
        QueryPeak::with_multiplet(3.833, PeakMultiplicity::Multiplet, vec![]),
        QueryPeak::with_multiplet(3.778, PeakMultiplicity::Quartet, vec![7.2]), // L-Alanine
        QueryPeak::with_multiplet(3.718, PeakMultiplicity::Multiplet, vec![]),
        QueryPeak::with_multiplet(3.605, PeakMultiplicity::Doublet, vec![4.4]), // L-Valine
        QueryPeak::with_multiplet(3.555, PeakMultiplicity::Singlet, vec![]),    // Glycine
        QueryPeak::with_multiplet(3.533, PeakMultiplicity::DoubletOfDoublets, vec![9.8, 3.7]),
        QueryPeak::with_multiplet(3.475, PeakMultiplicity::Triplet, vec![9.2]),
        QueryPeak::with_multiplet(3.401, PeakMultiplicity::Multiplet, vec![]),
        QueryPeak::with_multiplet(3.266, PeakMultiplicity::Singlet, vec![]), // TMAO
        QueryPeak::with_multiplet(3.242, PeakMultiplicity::DoubletOfDoublets, vec![9.2, 7.9]),
        QueryPeak::with_multiplet(3.043, PeakMultiplicity::Singlet, vec![]), // Creatinine CH3
        QueryPeak::with_multiplet(3.032, PeakMultiplicity::Singlet, vec![]), // Creatine CH3
        QueryPeak::with_multiplet(2.675, PeakMultiplicity::Doublet, vec![15.3]), // Citrate AB
        QueryPeak::with_multiplet(2.535, PeakMultiplicity::Doublet, vec![15.3]),
        QueryPeak::with_multiplet(2.405, PeakMultiplicity::Singlet, vec![]), // Succinate
        QueryPeak::with_multiplet(2.268, PeakMultiplicity::Multiplet, vec![]), // L-Valine
        QueryPeak::with_multiplet(1.918, PeakMultiplicity::Singlet, vec![]), // Acetate
        QueryPeak::with_multiplet(1.478, PeakMultiplicity::Doublet, vec![7.2]), // L-Alanine
        QueryPeak::with_multiplet(1.330, PeakMultiplicity::Doublet, vec![6.9]), // L-Lactate
        QueryPeak::with_multiplet(1.037, PeakMultiplicity::Doublet, vec![7.0]), // L-Valine
        QueryPeak::with_multiplet(0.985, PeakMultiplicity::Doublet, vec![7.0]),
    ];

    let hits = query_offline_metabolites(&demo_multiplet_q, MetaboliteNucleus::H1, 0.03, 30);
    let true_singlets = [
        ("bmse000203", "Formate"),
        ("bmse000191", "Acetic acid"),
        ("bmse000183", "Succinic acid"),
        ("bmse000089", "Glycine"),
        ("bmse000426", "Trimethylamine N-oxide"),
    ];
    let best_partial_fp_score = hits
        .iter()
        .filter(|h| h.matched_ref_peaks < h.total_ref_peaks)
        .map(|h| h.score)
        .fold(0.0_f64, f64::max);

    for (bmse_id, label) in true_singlets {
        let pos = hits
            .iter()
            .position(|h| h.entry.bmse_id == bmse_id)
            .unwrap_or_else(|| panic!("true singlet {label} ({bmse_id}) missing from hits"));
        assert!(
            pos < 14,
            "true singlet {label} ({bmse_id}) must rank inside top-14 on 14-metabolite demo mixture (got rank #{}, score={:.4})",
            pos + 1,
            hits[pos].score
        );
        assert!(
            hits[pos].score > best_partial_fp_score,
            "true singlet {label} ({bmse_id}, score={:.4}) must outrank every partial-match false positive (best partial FP score={best_partial_fp_score:.4})",
            hits[pos].score
        );
    }

    // Citrate must also be matched 2/2 inside top-10 on the demo mixture.
    let cit_pos = hits
        .iter()
        .position(|h| h.entry.bmse_id == "bmse000076")
        .expect("Citrate must match in demo mixture");
    assert!(cit_pos < 10 && hits[cit_pos].matched_ref_peaks == 2);

    // 2. Even on a blind PPM-only query (zero multiplicity annotations, +-0.002 ppm jitter),
    //    a 1/1 singlet with delta = 0.002 ppm must strictly outrank 2/9, 2/6, 3/5, and 3/4
    //    partial-match false positives.
    let ppm_only_q: Vec<QueryPeak> = [
        1.320, 4.099, 1.473, 3.769, 0.983, 1.032, 2.268, 3.597, 3.234, 3.399, 3.461, 3.515, 3.728,
        3.826, 3.889, 4.634, 5.224, 2.533, 2.659, 2.393, 1.907, 3.026, 3.918, 3.036, 4.040, 3.251,
        3.547, 8.449, 3.946, 7.528, 7.620, 7.806,
    ]
    .into_iter()
    .map(QueryPeak::from_ppm)
    .collect();
    let ppm_hits = query_offline_metabolites(&ppm_only_q, MetaboliteNucleus::H1, 0.03, 40);
    let best_ppm_partial_fp = ppm_hits
        .iter()
        .filter(|h| h.matched_ref_peaks < h.total_ref_peaks)
        .map(|h| h.score)
        .fold(0.0_f64, f64::max);
    for (bmse_id, label) in true_singlets {
        let hit = ppm_hits
            .iter()
            .find(|h| h.entry.bmse_id == bmse_id)
            .unwrap_or_else(|| panic!("{label} missing from PPM-only hits"));
        assert!(
            hit.score > best_ppm_partial_fp,
            "PPM-only 1/1 singlet {label} (score={:.4}) must outrank partial-match FPs (best={best_ppm_partial_fp:.4})",
            hit.score
        );
    }
}
