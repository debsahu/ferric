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

fn load_real_afdb_p0cg48_cif_and_pae() -> (PathBuf, PathBuf) {
    let cif_path = repo_root().join("data/afdb/AF-P0CG48-F1-model_v6.cif");
    let pae_path = repo_root().join("data/afdb/AF-P0CG48-F1-predicted_aligned_error_v6.json");
    assert!(cif_path.is_file(), "missing {}", cif_path.display());
    assert!(pae_path.is_file(), "missing {}", pae_path.display());
    (cif_path, pae_path)
}

#[test]
fn test_afdb_fetcher_allowlist_uniprot_validation_rate_limiter_and_sha256_provenance() {
    assert_eq!(AFDB_USER_AGENT, "ferric-af-fetch");
    assert_eq!(AFDB_ALLOWED_HOST, "alphafold.ebi.ac.uk");
    const { assert!(MIN_AFDB_REQUEST_INTERVAL_MS >= 1_000) };

    // 1. Strict `[A-Z0-9]{6,10}` UniProt accession checks
    assert_eq!(validate_uniprot_id("P0CG48").unwrap(), "P0CG48");
    assert_eq!(validate_uniprot_id("A0A024RBG1").unwrap(), "A0A024RBG1");
    assert!(validate_uniprot_id("p0cg48").is_err());
    assert!(validate_uniprot_id("P1234").is_err());
    assert!(validate_uniprot_id("A0A024RBG19").is_err());
    assert!(validate_uniprot_id("../P0CG48").is_err());
    assert!(validate_uniprot_id("P0CG48/x").is_err());

    // 2. Strict HTTPS host allowlist (`https://alphafold.ebi.ac.uk/...` only)
    assert!(validate_afdb_https_url("https://alphafold.ebi.ac.uk/api/prediction/P0CG48").is_ok());
    assert!(validate_afdb_https_url("http://alphafold.ebi.ac.uk/api/prediction/P0CG48").is_err());
    assert!(
        validate_afdb_https_url("https://alphafold.ebi.ac.uk.evil.org/files/AF-P0CG48-F1.cif")
            .is_err()
    );
    assert!(
        validate_afdb_https_url("https://user@alphafold.ebi.ac.uk/files/AF-P0CG48-F1.cif").is_err()
    );
    assert!(
        validate_afdb_https_url("https://alphafold.ebi.ac.uk:8443/files/AF-P0CG48-F1.cif").is_err()
    );
    assert!(validate_afdb_https_url("https://alphafold.ebi.ac.uk/files/../secret.cif").is_err());
    assert!(validate_afdb_https_url("https://example.com/AF-P0CG48-F1.cif").is_err());

    // 3. Filename path-traversal guard
    assert_eq!(
        sanitize_afdb_filename("AF-P0CG48-F1-model_v6.cif").unwrap(),
        "AF-P0CG48-F1-model_v6.cif"
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
        "https://alphafold.ebi.ac.uk/api/prediction/P0CG48",
        r#"[{"entryId":"AF-P0CG48-F1","latestVersion":6,"cifUrl":"https://attacker.example/AF-P0CG48-F1-model_v6.cif"}]"#,
    );
    let poison_err = execute_afdb_fetch(
        &AfdbFetchConfig::new("P0CG48", dir.join("poison")),
        &mut poisoned,
    )
    .unwrap_err();
    assert!(poison_err.contains("allowlisted"), "got: {poison_err}");

    // 6. Valid offline AFDB bundle using real committed `AF-P0CG48-F1` v6 bytes writes `.cif`, `pae.json`, and `PROVENANCE.md`
    let (real_cif_path, real_pae_path) = load_real_afdb_p0cg48_cif_and_pae();
    let cif_bytes = fs::read(&real_cif_path).unwrap();
    let pae_bytes = fs::read(&real_pae_path).unwrap();
    let expected_cif_sha = sha256_hex(&cif_bytes);
    let expected_pae_sha = sha256_hex(&pae_bytes);

    let api_url = "https://alphafold.ebi.ac.uk/api/prediction/P0CG48";
    let cif_url = "https://alphafold.ebi.ac.uk/files/AF-P0CG48-F1-model_v6.cif";
    let pae_url = "https://alphafold.ebi.ac.uk/files/AF-P0CG48-F1-predicted_aligned_error_v6.json";

    let mut mock = MockAfdbTransport::new();
    mock.insert(
        api_url,
        format!(
            r#"[{{"entryId":"AF-P0CG48-F1","latestVersion":6,"cifUrl":"{cif_url}","paeDocUrl":"{pae_url}"}}]"#
        ),
    );
    mock.insert(cif_url, cif_bytes);
    mock.insert(pae_url, pae_bytes);

    let cache_ok = dir.join("afdb_cache");
    let res = execute_afdb_fetch(&AfdbFetchConfig::new("P0CG48", &cache_ok), &mut mock).unwrap();
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
fn test_ferric_af_validate_cli_json_tsv_html_nef_and_fail_closed_provenance_exit_codes() {
    let dir = env::temp_dir().join(format!("ferric_af_validate_cli_{}", process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let (cif_path, pae_path) = load_real_afdb_p0cg48_cif_and_pae();
    let _ = parse_af_structure_model(&cif_path, &AfIngestOptions::default()).unwrap();

    let star_path = repo_root().join("data/fixtures/bmr17769_ubiquitin_allatom.str");
    let out_json = dir.join("ubq_concordance.json");
    let out_tsv = dir.join("ubq_concordance.tsv");
    let out_html = dir.join("ubq_report.html");

    // 1. JSON output with real AFDB `AF-P0CG48-F1-model_v6.cif` + `--pae`, `--exclude-pdb`, `--exclude-bmrb`
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
            "17769,5387",
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
    assert!(js_val["spearman_rho"].as_f64().is_some());
    assert!(js_val["spearman_rho_ci95"].is_array());
    assert!(js_val["ordered_core_spearman_rho"].as_f64().is_some());
    assert!(js_val["max_retained_seq_identity"].as_f64().unwrap() <= 0.25);
    assert_eq!(js_val["pae_n_residues"].as_u64(), Some(685));

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
    assert!(tsv_body.contains("ordered_core_rho_s="));
    assert_eq!(
        tsv_body
            .lines()
            .filter(|l| !l.starts_with('#') && !l.starts_with("seq_id"))
            .count(),
        685
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

    // 5. `21.W1:P0-2`: Experimental `1ubq.pdb` AND headerless `1ubq.pdb` MUST fail closed (`!status.success()`)
    //    when `--treat-as-model` is not supplied!
    let xray_pdb = repo_root().join("data/fixtures/ubiquitin_1ubq_xray.pdb");
    let out_xray_fail = Command::new(ferric_bin())
        .args([
            "af-validate",
            "--structure",
            xray_pdb.to_str().unwrap(),
            "--shifts",
            star_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        !out_xray_fail.status.success(),
        "experimental 1UBQ PDB must fail closed without --treat-as-model"
    );
    assert!(String::from_utf8_lossy(&out_xray_fail.stderr)
        .contains("predicted model provenance check failed"));

    let stripped_pdb_path = dir.join("stripped_1ubq.pdb");
    let stripped_body: String = fs::read_to_string(&xray_pdb)
        .unwrap()
        .lines()
        .filter(|l| l.starts_with("ATOM  ") || l.starts_with("TER") || l.starts_with("END"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&stripped_pdb_path, stripped_body).unwrap();
    let out_stripped_fail = Command::new(ferric_bin())
        .args([
            "af-validate",
            "--structure",
            stripped_pdb_path.to_str().unwrap(),
            "--shifts",
            star_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        !out_stripped_fail.status.success(),
        "headerless 1UBQ PDB with B <= 100 must fail closed without --treat-as-model"
    );

    // 6. Hostile CLI errors exit non-zero cleanly without panicking
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
