//! CLI integration tests for `ferric af-validate` (`T21.1`, `docs/specs/21-alphafold-integrative-nmr.md` §1, §8).

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::{self, Command};
use std::time::{Duration, Instant};

#[path = "../src/af_validate_cmd.rs"]
#[allow(dead_code)]
mod af_validate_cmd;

use af_validate_cmd::{
    execute_afdb_fetch, sanitize_afdb_filename, validate_afdb_https_url, validate_uniprot_id,
    AfdbFetchConfig, AfdbHttpTransport, AfdbRateLimiter, AFDB_ALLOWED_HOST, AFDB_USER_AGENT,
    MIN_AFDB_REQUEST_INTERVAL_MS,
};
use nmr_core::af_concordance::{parse_af_structure_model, AfIngestOptions};
use nmr_core::torsion_db::sha256_hex;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn ferric_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ferric"))
}

struct MockAfdbTransport {
    responses: BTreeMap<String, Vec<u8>>,
    calls: Vec<String>,
}

impl MockAfdbTransport {
    fn new() -> Self {
        Self {
            responses: BTreeMap::new(),
            calls: Vec::new(),
        }
    }

    fn insert(&mut self, url: &str, body: impl Into<Vec<u8>>) {
        self.responses.insert(url.to_string(), body.into());
    }
}

impl AfdbHttpTransport for MockAfdbTransport {
    fn get_https_bytes(&mut self, url: &str, max_bytes: usize) -> Result<Vec<u8>, String> {
        validate_afdb_https_url(url)?;
        self.calls.push(url.to_string());
        let data = self
            .responses
            .get(url)
            .ok_or_else(|| format!("mock 404 for '{url}'"))?
            .clone();
        if data.len() > max_bytes {
            return Err("exceeded max_bytes".to_string());
        }
        Ok(data)
    }
}

fn build_ubiquitin_af_cif_and_pae() -> (String, String) {
    let pdb_path = repo_root().join("data/fixtures/ubiquitin_1ubq_xray.pdb");
    let base = parse_af_structure_model(&pdb_path, &AfIngestOptions::default()).unwrap();
    let chain = base.to_pdb_chain();
    assert_eq!(chain.residues.len(), 76);

    let mut cif = String::from(
        "data_AF-P62988-F1\n_entry.id AF-P62988-F1\n_ma_protocol_step.step_id 1\n\
         loop_\n_ma_qa_metric.id\n_ma_qa_metric.name\n_ma_qa_metric.mode\n1 pLDDT local\n#\n\
         loop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n\
         _atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n_atom_site.auth_seq_id\n\
         _atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.B_iso_or_equiv\n",
    );
    let mut serial = 1usize;
    let plddts: Vec<f64> = vec![
        89.4, 95.1, 97.8, 98.5, 98.4, 96.8, 91.2, 81.4, 76.2, 75.8, // 1..10
        78.9, 91.5, 96.4, 97.9, 98.3, 96.7, 94.8, 92.6, 91.8, 90.9, // 11..20
        92.1, 94.6, 97.4, 98.1, 98.5, 98.7, 98.6, 98.4, 98.5, 98.3, // 21..30
        97.6, 95.8, 93.4, 91.2, 88.6, 89.4, 90.2, 90.8, 88.5, 86.4, // 31..40
        91.8, 95.6, 97.9, 98.4, 97.2, 88.9, 85.4, 83.2, 80.6, 79.8, // 41..50
        80.4, 82.6, 86.5, 91.2, 94.8, 96.2, 95.9, 94.1, 91.6, 87.8, // 51..60
        86.9, 87.5, 88.4, 89.6, 91.4, 95.2, 97.6, 98.3, 97.8, 95.4, // 61..70
        84.6, 68.4, 55.2, 44.6, 38.4, 35.1, // 71..76
    ];

    for (idx, r) in chain.residues.iter().enumerate() {
        let s = idx + 1;
        for (sym, aname, xyz) in [
            ("N", "N", r.n),
            ("C", "CA", r.ca),
            ("C", "C", r.c),
            ("O", "O", r.o),
        ] {
            cif.push_str(&format!(
                "ATOM {serial} {sym} {aname} {} A {s} {} {:.3} {:.3} {:.3} {:.2}\n",
                r.res_name, r.seq_id, xyz[0], xyz[1], xyz[2], plddts[idx]
            ));
            serial += 1;
        }
    }
    cif.push_str(
        "#\nloop_\n_ma_qa_metric_local.label_asym_id\n_ma_qa_metric_local.label_seq_id\n\
         _ma_qa_metric_local.label_comp_id\n_ma_qa_metric_local.metric_id\n_ma_qa_metric_local.metric_value\n",
    );
    for (idx, r) in chain.residues.iter().enumerate() {
        cif.push_str(&format!(
            "A {} {} 1 {:.2}\n",
            idx + 1,
            r.res_name,
            plddts[idx]
        ));
    }
    cif.push_str("#\n");

    let mut rows = Vec::with_capacity(76);
    for i in 0..76 {
        let mut row = Vec::with_capacity(76);
        for j in 0..76 {
            let v = if i == j {
                0.2
            } else if i >= 72 || j >= 72 {
                18.5
            } else {
                1.8
            };
            row.push(format!("{v:.1}"));
        }
        rows.push(format!("[{}]", row.join(",")));
    }
    let pae = format!(
        "[{{\"predicted_aligned_error\": [{}], \"max_predicted_aligned_error\": 31.75}}]",
        rows.join(",")
    );
    (cif, pae)
}

#[test]
fn test_afdb_fetcher_allowlist_uniprot_validation_rate_limiter_and_sha256_provenance() {
    assert_eq!(AFDB_USER_AGENT, "ferric-af-fetch");
    assert_eq!(AFDB_ALLOWED_HOST, "alphafold.ebi.ac.uk");
    const { assert!(MIN_AFDB_REQUEST_INTERVAL_MS >= 1_000) };

    // 1. Strict `[A-Z0-9]{6,10}` UniProt accession checks
    assert_eq!(validate_uniprot_id("P62988").unwrap(), "P62988");
    assert_eq!(validate_uniprot_id("A0A024RBG1").unwrap(), "A0A024RBG1");
    assert!(validate_uniprot_id("p62988").is_err());
    assert!(validate_uniprot_id("P1234").is_err());
    assert!(validate_uniprot_id("A0A024RBG19").is_err());
    assert!(validate_uniprot_id("../P62988").is_err());
    assert!(validate_uniprot_id("P62988/x").is_err());

    // 2. Strict HTTPS host allowlist (`https://alphafold.ebi.ac.uk/...` only)
    assert!(validate_afdb_https_url("https://alphafold.ebi.ac.uk/api/prediction/P62988").is_ok());
    assert!(validate_afdb_https_url("http://alphafold.ebi.ac.uk/api/prediction/P62988").is_err());
    assert!(
        validate_afdb_https_url("https://alphafold.ebi.ac.uk.evil.org/files/AF-P62988-F1.cif")
            .is_err()
    );
    assert!(
        validate_afdb_https_url("https://user@alphafold.ebi.ac.uk/files/AF-P62988-F1.cif").is_err()
    );
    assert!(
        validate_afdb_https_url("https://alphafold.ebi.ac.uk:8443/files/AF-P62988-F1.cif").is_err()
    );
    assert!(validate_afdb_https_url("https://alphafold.ebi.ac.uk/files/../secret.cif").is_err());
    assert!(validate_afdb_https_url("https://example.com/AF-P62988-F1.cif").is_err());

    // 3. Filename path-traversal guard
    assert_eq!(
        sanitize_afdb_filename("AF-P62988-F1-model_v4.cif").unwrap(),
        "AF-P62988-F1-model_v4.cif"
    );
    assert!(sanitize_afdb_filename("../evil.cif").is_err());
    assert!(sanitize_afdb_filename(".hidden.cif").is_err());
    assert!(sanitize_afdb_filename("sub/evil.cif").is_err());
    assert!(sanitize_afdb_filename("sub\\evil.cif").is_err());

    // 4. Token-bucket rate limiter (`<= 1 req/s` check + real wall-clock sleep verification)
    let mut prod_limiter =
        AfdbRateLimiter::new(Duration::from_millis(MIN_AFDB_REQUEST_INTERVAL_MS));
    assert_eq!(prod_limiter.acquire(), Duration::ZERO);
    let wait_rem = prod_limiter.wait_needed_at(Instant::now());
    assert!(
        wait_rem >= Duration::from_millis(950),
        "expected >= 950 ms remaining in <= 1 req/s rate limiter, got {wait_rem:?}"
    );

    let mut fast_limiter = AfdbRateLimiter::new(Duration::from_millis(90));
    let t0 = Instant::now();
    fast_limiter.acquire();
    fast_limiter.acquire();
    assert!(t0.elapsed() >= Duration::from_millis(80));
    assert_eq!(fast_limiter.acquired_count(), 2);

    // 5. Poisoned redirect/cifUrl host in AFDB API JSON must be rejected before download
    let dir = env::temp_dir().join(format!("ferric_af_fetch_test_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let mut poisoned = MockAfdbTransport::new();
    poisoned.insert(
        "https://alphafold.ebi.ac.uk/api/prediction/P62988",
        r#"[{"entryId":"AF-P62988-F1","latestVersion":4,"cifUrl":"https://attacker.example/AF-P62988-F1-model_v4.cif"}]"#,
    );
    let poison_err = execute_afdb_fetch(
        &AfdbFetchConfig::new("P62988", dir.join("poison")),
        &mut poisoned,
    )
    .unwrap_err();
    assert!(poison_err.contains("allowlisted"), "got: {poison_err}");

    // 6. Valid offline AFDB bundle writes `.cif`, `pae.json`, and `PROVENANCE.md` with FIPS 180-4 SHA-256
    let (cif_txt, pae_txt) = build_ubiquitin_af_cif_and_pae();
    let expected_cif_sha = sha256_hex(cif_txt.as_bytes());
    let expected_pae_sha = sha256_hex(pae_txt.as_bytes());

    let api_url = "https://alphafold.ebi.ac.uk/api/prediction/P62988";
    let cif_url = "https://alphafold.ebi.ac.uk/files/AF-P62988-F1-model_v4.cif";
    let pae_url = "https://alphafold.ebi.ac.uk/files/AF-P62988-F1-predicted_aligned_error_v4.json";

    let mut mock = MockAfdbTransport::new();
    mock.insert(
        api_url,
        format!(
            r#"[{{"entryId":"AF-P62988-F1","latestVersion":4,"cifUrl":"{cif_url}","paeDocUrl":"{pae_url}"}}]"#
        ),
    );
    mock.insert(cif_url, cif_txt.as_bytes());
    mock.insert(pae_url, pae_txt.as_bytes());

    let cache_ok = dir.join("afdb_cache");
    let res = execute_afdb_fetch(&AfdbFetchConfig::new("P62988", &cache_ok), &mut mock).unwrap();
    assert_eq!(res.cif_sha256, expected_cif_sha);
    assert_eq!(res.pae_sha256.as_deref(), Some(expected_pae_sha.as_str()));
    assert!(res.cif_path.is_file());
    assert!(res.pae_path.as_ref().unwrap().is_file());
    let prov = fs::read_to_string(&res.provenance_path).unwrap();
    assert!(prov.contains("CC-BY-4.0"));
    assert!(prov.contains("ferric-af-fetch"));
    assert!(prov.contains(&expected_cif_sha));
    assert!(prov.contains(&expected_pae_sha));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_ferric_af_validate_cli_json_tsv_html_nef_and_hostile_err_exit_codes() {
    let dir = env::temp_dir().join(format!("ferric_af_validate_cli_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let (cif_txt, pae_txt) = build_ubiquitin_af_cif_and_pae();
    let cif_path = dir.join("AF-P62988-F1.cif");
    let pae_path = dir.join("AF-P62988-F1-pae.json");
    fs::write(&cif_path, &cif_txt).unwrap();
    fs::write(&pae_path, &pae_txt).unwrap();

    let star_path = repo_root().join("data/fixtures/bmr17769_ubiquitin_allatom.str");
    let out_json = dir.join("ubq_concordance.json");
    let out_tsv = dir.join("ubq_concordance.tsv");
    let out_html = dir.join("ubq_report.html");

    // 1. JSON output with `--pae`, `--exclude-pdb`, `--exclude-bmrb`
    let status_json = Command::new(ferric_bin())
        .args([
            "af-validate",
            "--structure",
            cif_path.to_str().unwrap(),
            "--pae",
            pae_path.to_str().unwrap(),
            "--shifts",
            star_path.to_str().unwrap(),
            "--exclude-pdb",
            "1D3Z,1UBQ",
            "--exclude-bmrb",
            "17769",
            "-o",
            out_json.to_str().unwrap(),
        ])
        .output()
        .expect("run ferric af-validate json");
    assert!(
        status_json.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&status_json.stderr)
    );
    let js_val: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&out_json).unwrap()).unwrap();
    assert_eq!(js_val["summary"]["status"], "evaluated");
    assert!(js_val["summary"]["spearman_rho_s"].as_f64().unwrap() >= 0.70);
    assert!(js_val["max_retained_seq_identity"].as_f64().unwrap() <= 0.25);
    assert_eq!(js_val["pae_n_residues"].as_u64(), Some(76));

    // 2. TSV output
    let out_t = Command::new(ferric_bin())
        .args([
            "af-validate",
            "--structure",
            cif_path.to_str().unwrap(),
            "--shifts",
            star_path.to_str().unwrap(),
            "-o",
            out_tsv.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out_t.status.success());
    let tsv_body = fs::read_to_string(&out_tsv).unwrap();
    assert!(tsv_body.contains("concordance_class"));
    assert_eq!(
        tsv_body
            .lines()
            .filter(|l| !l.starts_with('#') && !l.starts_with("seq_id"))
            .count(),
        76
    );

    // 3. HTML report output (zero emojis, inline SVG `#svg-af-concordance`)
    let out_h = Command::new(ferric_bin())
        .args([
            "af-validate",
            "--structure",
            cif_path.to_str().unwrap(),
            "--pae",
            pae_path.to_str().unwrap(),
            "--shifts",
            star_path.to_str().unwrap(),
            "-o",
            out_html.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out_h.status.success());
    let html_body = fs::read_to_string(&out_html).unwrap();
    assert!(html_body.contains("id=\"section-af-concordance\""));
    assert!(html_body.contains("id=\"svg-af-concordance\""));
    let sec_pos = html_body
        .find("id=\"section-af-concordance\"")
        .expect("missing #section-af-concordance");
    let sec_end = html_body[sec_pos..]
        .find("</section>")
        .map(|off| sec_pos + off)
        .unwrap_or(html_body.len());
    assert!(
        html_body[sec_pos..sec_end].is_ascii(),
        "AF concordance HTML section must be pure ASCII with zero emojis"
    );
    assert!(
        !html_body.chars().any(|ch| matches!(
            ch as u32,
            0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0xFE0F
        )),
        "HTML report must contain zero emoji characters"
    );

    // 4. NEF shifts input via `ferric nef-export` -> `ferric af-validate --shifts <nef>`
    let nef_path = dir.join("ubq.nef");
    let nef_exp = Command::new(ferric_bin())
        .args([
            "nef-export",
            "--star",
            star_path.to_str().unwrap(),
            "--out",
            nef_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(nef_exp.status.success());

    let nef_tsv = dir.join("ubq_from_nef.tsv");
    let out_nef = Command::new(ferric_bin())
        .args([
            "af-validate",
            "--structure",
            cif_path.to_str().unwrap(),
            "--shifts",
            nef_path.to_str().unwrap(),
            "-o",
            nef_tsv.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out_nef.status.success(),
        "NEF shifts af-validate failed: {}",
        String::from_utf8_lossy(&out_nef.stderr)
    );

    // 5. Hostile CLI errors exit non-zero cleanly without panicking
    let bad_pae = dir.join("mismatch_pae.json");
    fs::write(&bad_pae, r#"{"pae": [[0.2, 1.0], [1.0, 0.2]]}"#).unwrap();
    let bad_out = Command::new(ferric_bin())
        .args([
            "af-validate",
            "--structure",
            cif_path.to_str().unwrap(),
            "--pae",
            bad_pae.to_str().unwrap(),
            "--shifts",
            star_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!bad_out.status.success());
    let stderr_bad = String::from_utf8_lossy(&bad_out.stderr);
    assert!(!stderr_bad.contains("panicked"));
    assert!(stderr_bad.contains("PAE matrix dimension N = 2"));

    let bad_unp = Command::new(ferric_bin())
        .args([
            "af-validate",
            "--fetch-uniprot",
            "bad_id!",
            "--cache-dir",
            dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!bad_unp.status.success());

    let _ = fs::remove_dir_all(&dir);
}
