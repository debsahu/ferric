//! CLI subcommand `ferric af-validate` (`T21.1`, `docs/specs/21-alphafold-integrative-nmr.md` §1):
//! AlphaFold / ModelArchive mmCIF, PDB & `PAE` JSON validation against solution NMR chemical
//! shifts (`RCI-S²`) or Lipari–Szabo $S^2$ order parameters, plus optional CLI-only rate-limited
//! (`<= 1 req/s`) `alphafold.ebi.ac.uk` archive fetcher writing `SHA-256` `PROVENANCE.md`.
//!
//! Provenance: Clean-room original implementation citing Jumper et al. (*Nature* **2021**, 596,
//! 583–589), Varadi et al. (*Nucleic Acids Res.* **2022**, 50, D439–D444; **2024**, 52, D368–D375),
//! Kabsch & Sander (*Biopolymers* **1983**, 22, 2577–2637), Adzhubei et al. (*J. Mol. Biol.*
//! **2013**, 425, 2100–2132), Berjanskii & Wishart (*J. Am. Chem. Soc.* **2005**, 127, 14970–14971),
//! and Fowler, Sljoka & Williamson (*Nat. Commun.* **2020**, 11, 5982; *Structure* **2022**, 30,
//! 925–933).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use nmr_core::af_concordance::{
    evaluate_af_nmr_concordance, format_af_concordance_json, format_af_concordance_tsv,
    parse_af_structure_model, parse_modelfree_s2_tsv_file, parse_pae_json_file,
    require_predicted_model, AfConcordanceConfig, AfConcordanceReport, AfIngestOptions,
    ConcordanceSummary, MAX_AF_STRUCTURE_BYTES,
};
use nmr_core::nef::read_nef_file;
use nmr_core::nmrstar::{read_star, Reference};
use nmr_core::report::{write_quality_report_html, ReportInput};
use nmr_core::torsion_db::sha256_hex;

/// Mandatory `User-Agent` header sent on all `alphafold.ebi.ac.uk` HTTPS requests.
pub const AFDB_USER_AGENT: &str = "ferric-af-fetch";

/// Strictly allowlisted HTTPS host for opt-in AlphaFold Database downloads.
pub const AFDB_ALLOWED_HOST: &str = "alphafold.ebi.ac.uk";

/// Canonical base URL for AlphaFold Database API queries.
pub const DEFAULT_AFDB_BASE_URL: &str = "https://alphafold.ebi.ac.uk";

/// Minimum request spacing (`1,050 ms` -> strictly `<= 1 req/s`).
pub const MIN_AFDB_REQUEST_INTERVAL_MS: u64 = 1_050;

/// Validate a UniProt primary accession against strict `[A-Z0-9]{6,10}` syntax.
pub fn validate_uniprot_id(raw: &str) -> Result<String, String> {
    let s = raw.trim();
    if !(6..=10).contains(&s.len()) {
        return Err(format!(
            "invalid UniProt accession '{raw}': expected 6..=10 uppercase alphanumeric characters ([A-Z0-9]{{6,10}})"
        ));
    }
    if !s
        .bytes()
        .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
    {
        return Err(format!(
            "invalid UniProt accession '{raw}': must match strict [A-Z0-9]{{6,10}} (uppercase ASCII A-Z and 0-9 only)"
        ));
    }
    Ok(s.to_string())
}

/// Enforce the strict HTTPS host allowlist (`https://alphafold.ebi.ac.uk/...` only),
/// rejecting non-HTTPS schemes, userinfo (`@`), custom ports, path traversal, or lookalike hosts.
pub fn validate_afdb_https_url(url: &str) -> Result<(), String> {
    let u = url.trim();
    let Some(after_scheme) = u.strip_prefix("https://") else {
        return Err(format!(
            "blocked URL '{url}': only https://{AFDB_ALLOWED_HOST}/... is permitted"
        ));
    };
    let (authority, path_part) = match after_scheme.split_once('/') {
        Some((auth, rest)) => (auth, rest),
        None => (after_scheme, ""),
    };
    if authority != AFDB_ALLOWED_HOST {
        return Err(format!(
            "blocked host '{authority}' in URL '{url}': strictly allowlisted to '{AFDB_ALLOWED_HOST}'"
        ));
    }
    if path_part.is_empty()
        || path_part.contains("..")
        || path_part.contains('\\')
        || path_part.as_bytes().contains(&0u8)
    {
        return Err(format!("blocked unsafe or empty path in AFDB URL '{url}'"));
    }
    Ok(())
}

/// Validate a single output filename component, rejecting any path separators or `..` traversal.
pub fn sanitize_afdb_filename(raw_name: &str) -> Result<String, String> {
    let s = raw_name.trim();
    if s.is_empty()
        || s.starts_with('.')
        || s.contains("..")
        || s.contains('/')
        || s.contains('\\')
        || s.contains(':')
        || s.as_bytes().contains(&0u8)
    {
        return Err(format!(
            "rejected unsafe output filename '{raw_name}' (path traversal guard)"
        ));
    }
    if !s
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        return Err(format!(
            "rejected filename '{raw_name}' containing non-[A-Za-z0-9._-] characters"
        ));
    }
    Ok(s.to_string())
}

/// Token-bucket rate limiter enforcing `<= 1 req/s` (`min_interval >= 1000 ms` in production).
#[derive(Clone, Debug)]
pub struct AfdbRateLimiter {
    min_interval: Duration,
    last_acquire: Option<Instant>,
    acquired_count: usize,
}

impl AfdbRateLimiter {
    pub fn new(min_interval: Duration) -> Self {
        Self {
            min_interval,
            last_acquire: None,
            acquired_count: 0,
        }
    }

    /// Compute remaining wait duration required at `now` before the next token becomes available.
    pub fn wait_needed_at(&self, now: Instant) -> Duration {
        match self.last_acquire {
            Some(prev) => {
                let elapsed = now.saturating_duration_since(prev);
                self.min_interval.saturating_sub(elapsed)
            }
            None => Duration::ZERO,
        }
    }

    /// Block until a rate-limit token is available and consume it.
    pub fn acquire(&mut self) -> Duration {
        let now = Instant::now();
        let wait = self.wait_needed_at(now);
        if !wait.is_zero() {
            thread::sleep(wait);
        }
        self.last_acquire = Some(Instant::now());
        self.acquired_count += 1;
        wait
    }

    pub fn acquired_count(&self) -> usize {
        self.acquired_count
    }
}

/// Pluggable HTTP GET transport so unit/CLI tests can verify rate-limiting, allowlists, and
/// `PROVENANCE.md` generation deterministically offline while production uses `curl`.
pub trait AfdbHttpTransport {
    fn get_https_bytes(&mut self, url: &str, max_bytes: usize) -> Result<Vec<u8>, String>;
}

/// Production rate-limited HTTPS client invoking `curl` with `User-Agent: ferric-af-fetch`.
pub struct CurlAfdbTransport {
    pub limiter: AfdbRateLimiter,
}

impl CurlAfdbTransport {
    pub fn new(min_interval: Duration) -> Self {
        Self {
            limiter: AfdbRateLimiter::new(min_interval),
        }
    }
}

impl AfdbHttpTransport for CurlAfdbTransport {
    fn get_https_bytes(&mut self, url: &str, max_bytes: usize) -> Result<Vec<u8>, String> {
        validate_afdb_https_url(url)?;
        self.limiter.acquire();
        let max_str = max_bytes.to_string();
        let out = Command::new("curl")
            .args([
                "-fsSL",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--max-filesize",
                &max_str,
                "-A",
                AFDB_USER_AGENT,
                url,
            ])
            .output()
            .map_err(|e| format!("failed to invoke curl for '{url}': {e}"))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(format!(
                "AFDB HTTPS GET failed for '{url}' (status {}): {}",
                out.status,
                stderr.trim()
            ));
        }
        if out.stdout.len() > max_bytes {
            return Err(format!(
                "AFDB response from '{url}' ({} bytes) exceeded cap ({max_bytes} bytes)",
                out.stdout.len()
            ));
        }
        Ok(out.stdout)
    }
}

/// Configuration for `execute_afdb_fetch`.
#[derive(Clone, Debug)]
pub struct AfdbFetchConfig {
    pub uniprot_id: String,
    pub cache_dir: PathBuf,
    pub api_base_url: String,
    pub max_bytes: usize,
}

impl AfdbFetchConfig {
    pub fn new(uniprot_id: impl Into<String>, cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            uniprot_id: uniprot_id.into(),
            cache_dir: cache_dir.into(),
            api_base_url: DEFAULT_AFDB_BASE_URL.to_string(),
            max_bytes: MAX_AF_STRUCTURE_BYTES,
        }
    }
}

/// Downloaded and SHA-256-verified AFDB model & PAE bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AfdbFetchResult {
    pub uniprot_id: String,
    pub entry_id: String,
    pub afdb_version: u32,
    pub cif_url: String,
    pub cif_path: PathBuf,
    pub cif_bytes: usize,
    pub cif_sha256: String,
    pub pae_url: Option<String>,
    pub pae_path: Option<PathBuf>,
    pub pae_bytes: Option<usize>,
    pub pae_sha256: Option<String>,
    pub provenance_path: PathBuf,
}

#[derive(Deserialize)]
struct AfdbPredictionEntry {
    #[serde(rename = "entryId")]
    entry_id: String,
    #[serde(rename = "latestVersion", default = "default_afdb_ver")]
    latest_version: u32,
    #[serde(rename = "cifUrl")]
    cif_url: String,
    #[serde(rename = "paeDocUrl", default)]
    pae_doc_url: Option<String>,
}

fn default_afdb_ver() -> u32 {
    4
}

fn format_utc_iso8601(unix_secs: u64) -> String {
    let days = (unix_secs / 86_400) as i64;
    let rem = unix_secs % 86_400;
    let hour = rem / 3_600;
    let minute = (rem % 3_600) / 60;
    let second = rem % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!("{year:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn now_utc_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_utc_iso8601(secs)
}

fn filename_from_afdb_url(url: &str) -> Result<String, String> {
    validate_afdb_https_url(url)?;
    let last = url
        .rsplit('/')
        .next()
        .ok_or_else(|| format!("missing filename in AFDB URL '{url}'"))?;
    let clean = last.split('?').next().unwrap_or(last);
    sanitize_afdb_filename(clean)
}

/// Execute rate-limited (`<= 1 req/s`), host-allowlisted (`alphafold.ebi.ac.uk`) AFDB fetch and
/// write `PROVENANCE.md` with `CC-BY-4.0` attribution and `FIPS 180-4` `SHA-256` digests.
pub fn execute_afdb_fetch(
    cfg: &AfdbFetchConfig,
    transport: &mut dyn AfdbHttpTransport,
) -> Result<AfdbFetchResult, String> {
    let uniprot = validate_uniprot_id(&cfg.uniprot_id)?;
    let base = cfg.api_base_url.trim_end_matches('/');
    let api_url = format!("{base}/api/prediction/{uniprot}");
    validate_afdb_https_url(&api_url)?;

    fs::create_dir_all(&cfg.cache_dir).map_err(|e| {
        format!(
            "failed to create AFDB cache directory '{}': {e}",
            cfg.cache_dir.display()
        )
    })?;

    let api_bytes = transport.get_https_bytes(&api_url, 4 * 1024 * 1024)?;
    let entries: Vec<AfdbPredictionEntry> = serde_json::from_slice(&api_bytes)
        .map_err(|e| format!("failed to parse AFDB prediction JSON for {uniprot}: {e}"))?;
    let first = entries
        .into_iter()
        .next()
        .ok_or_else(|| format!("AFDB returned empty prediction list for UniProt '{uniprot}'"))?;

    validate_afdb_https_url(&first.cif_url)?;
    let cif_fname = filename_from_afdb_url(&first.cif_url)?;
    if !cif_fname.ends_with(".cif") {
        return Err(format!(
            "unexpected AFDB coordinate filename '{cif_fname}' (expected .cif)"
        ));
    }
    let cif_path = cfg.cache_dir.join(&cif_fname);
    let cif_data = transport.get_https_bytes(&first.cif_url, cfg.max_bytes)?;
    let cif_sha256 = sha256_hex(&cif_data);
    fs::write(&cif_path, &cif_data)
        .map_err(|e| format!("failed writing '{}': {e}", cif_path.display()))?;

    let mut pae_url_out = None;
    let mut pae_path_out = None;
    let mut pae_bytes_out = None;
    let mut pae_sha_out = None;

    if let Some(ref purl) = first.pae_doc_url {
        validate_afdb_https_url(purl)?;
        let pae_fname = filename_from_afdb_url(purl)?;
        if !pae_fname.ends_with(".json") {
            return Err(format!(
                "unexpected AFDB PAE filename '{pae_fname}' (expected .json)"
            ));
        }
        let p_path = cfg.cache_dir.join(&pae_fname);
        let p_data = transport.get_https_bytes(purl, cfg.max_bytes)?;
        let p_sha = sha256_hex(&p_data);
        fs::write(&p_path, &p_data)
            .map_err(|e| format!("failed writing '{}': {e}", p_path.display()))?;
        pae_url_out = Some(purl.clone());
        pae_path_out = Some(p_path);
        pae_bytes_out = Some(p_data.len());
        pae_sha_out = Some(p_sha);
    }

    let ts_utc = now_utc_iso8601();
    let prov_path = cfg.cache_dir.join("PROVENANCE.md");
    let mut prov = String::new();
    if !prov_path.exists() {
        prov.push_str("# AlphaFold Protein Structure Database (AFDB) Download Provenance\n\n");
        prov.push_str(
            "- **Archive**: AlphaFold Protein Structure Database (EMBL-EBI / Google DeepMind)\n",
        );
        prov.push_str("- **License**: Creative Commons Attribution 4.0 International (`CC-BY-4.0`, Varadi et al., *Nucleic Acids Res.* 2022/2024; Jumper et al., *Nature* 2021)\n");
        prov.push_str(&format!(
            "- **User-Agent**: `{AFDB_USER_AGENT}` (`<= 1 req/s`)\n\n"
        ));
    } else if let Ok(existing) = fs::read_to_string(&prov_path) {
        prov.push_str(&existing);
        if !prov.ends_with('\n') {
            prov.push('\n');
        }
    }

    prov.push_str(&format!(
        "## `{}` (`{}`, AFDB `v{}` — fetched `{}`)\n",
        uniprot, first.entry_id, first.latest_version, ts_utc
    ));
    prov.push_str(&format!("- Metadata API URL: `{api_url}`\n"));
    prov.push_str(&format!(
        "- Coordinate mmCIF: `{}`\n  - Upstream URL: `{}`\n  - Size: `{}` bytes\n  - FIPS 180-4 SHA-256: `{}`\n",
        cif_fname,
        first.cif_url,
        cif_data.len(),
        cif_sha256
    ));
    if let (Some(ref purl), Some(ref ppath), Some(plen), Some(ref psha)) =
        (&pae_url_out, &pae_path_out, pae_bytes_out, &pae_sha_out)
    {
        let p_fname = ppath
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("pae.json");
        prov.push_str(&format!(
            "- Predicted Aligned Error JSON: `{p_fname}`\n  - Upstream URL: `{purl}`\n  - Size: `{plen}` bytes\n  - FIPS 180-4 SHA-256: `{psha}`\n"
        ));
    }
    prov.push('\n');

    fs::write(&prov_path, &prov)
        .map_err(|e| format!("failed writing '{}': {e}", prov_path.display()))?;

    Ok(AfdbFetchResult {
        uniprot_id: uniprot,
        entry_id: first.entry_id,
        afdb_version: first.latest_version,
        cif_url: first.cif_url,
        cif_path,
        cif_bytes: cif_data.len(),
        cif_sha256,
        pae_url: pae_url_out,
        pae_path: pae_path_out,
        pae_bytes: pae_bytes_out,
        pae_sha256: pae_sha_out,
        provenance_path: prov_path,
    })
}

fn parse_comma_tokens(raw: &str, flag: &str) -> Result<Vec<String>, String> {
    let out: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect();
    if out.is_empty() {
        return Err(format!("invalid empty {flag} value"));
    }
    Ok(out)
}

fn read_shifts_star_or_nef(path: &Path) -> Result<Reference, String> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "nef" {
        read_nef_file(path)?.to_reference()
    } else {
        read_star(path, None)
    }
}

pub fn af_validate_usage() -> &'static str {
    "Usage: ferric af-validate --structure <model.cif|model.pdb> [--pae <pae.json>] --shifts <assigned.str|assigned.nef> [--s2 <modelfree.tsv>] [--fetch-uniprot <id> --cache-dir <dir>] [--exclude-pdb <ids>] [--exclude-bmrb <ids>] [--exclude-topology <id>] [--treat-as-model | --force-bfactor-plddt] [--chain <id>] [--ph <f64>] [--temp-k <f64>] [-o <out.json|out.tsv|report.html>]"
}

/// Entry point for `ferric af-validate`.
pub fn run_af_validate_cli(args: &[String]) -> Result<(), String> {
    let mut structure_path: Option<PathBuf> = None;
    let mut pae_path: Option<PathBuf> = None;
    let mut shifts_path: Option<PathBuf> = None;
    let mut s2_path: Option<PathBuf> = None;
    let mut fetch_uniprot: Option<String> = None;
    let mut cache_dir: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut chain_id: Option<String> = None;
    let mut force_bfactor_plddt = false;
    let mut exclude_pdb: Vec<String> = Vec::new();
    let mut exclude_bmrb: Vec<String> = Vec::new();
    let mut exclude_topology: Option<String> = None;
    let mut sample_ph: Option<f64> = None;
    let mut temp_k: f64 = 298.15;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("{}", af_validate_usage());
                return Ok(());
            }
            "--structure" | "--pdb" | "--cif" => {
                i += 1;
                structure_path = Some(PathBuf::from(
                    args.get(i)
                        .ok_or("missing --structure <model.cif|model.pdb>")?,
                ));
            }
            "--pae" => {
                i += 1;
                pae_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --pae <pae.json>")?,
                ));
            }
            "--shifts" | "--star" => {
                i += 1;
                shifts_path = Some(PathBuf::from(
                    args.get(i)
                        .ok_or("missing --shifts <assigned.str|assigned.nef>")?,
                ));
            }
            "--s2" => {
                i += 1;
                s2_path = Some(PathBuf::from(
                    args.get(i).ok_or("missing --s2 <modelfree.tsv>")?,
                ));
            }
            "--fetch-uniprot" => {
                i += 1;
                fetch_uniprot = Some(
                    args.get(i)
                        .ok_or("missing --fetch-uniprot <UNIPROT_ID>")?
                        .clone(),
                );
            }
            "--cache-dir" => {
                i += 1;
                cache_dir = Some(PathBuf::from(
                    args.get(i).ok_or("missing --cache-dir <dir>")?,
                ));
            }
            "--exclude-pdb" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --exclude-pdb <ids>")?;
                exclude_pdb.extend(parse_comma_tokens(raw, "--exclude-pdb")?);
            }
            "--exclude-bmrb" => {
                i += 1;
                let raw = args.get(i).ok_or("missing --exclude-bmrb <ids>")?;
                for tok in parse_comma_tokens(raw, "--exclude-bmrb")? {
                    let norm = tok
                        .to_ascii_lowercase()
                        .trim_start_matches("bmr")
                        .to_string();
                    if norm.is_empty() {
                        return Err(format!("invalid --exclude-bmrb token {tok:?}"));
                    }
                    exclude_bmrb.push(norm);
                }
            }
            "--exclude-topology" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or("missing --exclude-topology <id>")?
                    .trim()
                    .to_string();
                if raw.is_empty() {
                    return Err("invalid empty --exclude-topology value".to_string());
                }
                exclude_topology = Some(raw);
            }
            "--force-bfactor-plddt" | "--treat-as-model" => {
                force_bfactor_plddt = true;
            }
            "--chain" => {
                i += 1;
                chain_id = Some(args.get(i).ok_or("missing --chain <id>")?.clone());
            }
            "--ph" => {
                i += 1;
                let v: f64 = args
                    .get(i)
                    .ok_or("missing --ph <f64>")?
                    .parse()
                    .map_err(|e| format!("invalid --ph: {e}"))?;
                if !v.is_finite() || !(0.0..=14.0).contains(&v) {
                    return Err(format!("--ph {v} outside [0.0, 14.0]"));
                }
                sample_ph = Some(v);
            }
            "--temp-k" => {
                i += 1;
                let v: f64 = args
                    .get(i)
                    .ok_or("missing --temp-k <f64>")?
                    .parse()
                    .map_err(|e| format!("invalid --temp-k: {e}"))?;
                if !v.is_finite() || !(200.0..=400.0).contains(&v) {
                    return Err(format!("--temp-k {v} outside [200.0, 400.0] K"));
                }
                temp_k = v;
            }
            "-o" | "--out" => {
                i += 1;
                out_path = Some(PathBuf::from(
                    args.get(i)
                        .ok_or("missing -o <out.json|out.tsv|report.html>")?,
                ));
            }
            other => {
                return Err(format!(
                    "unknown af-validate flag {other:?}\n{}",
                    af_validate_usage()
                ));
            }
        }
        i += 1;
    }

    // Optional CLI-only AFDB fetch step
    let mut fetched_uniprot_acc: Option<String> = None;
    if let Some(ref raw_unp) = fetch_uniprot {
        let cdir = cache_dir
            .as_ref()
            .ok_or("--fetch-uniprot requires --cache-dir <dir>")?;
        let unp = validate_uniprot_id(raw_unp)?;
        let mut transport =
            CurlAfdbTransport::new(Duration::from_millis(MIN_AFDB_REQUEST_INTERVAL_MS));
        let fetch_res = execute_afdb_fetch(&AfdbFetchConfig::new(&unp, cdir), &mut transport)?;
        if structure_path.is_none() {
            structure_path = Some(fetch_res.cif_path.clone());
        }
        if pae_path.is_none() {
            pae_path = fetch_res.pae_path.clone();
        }
        fetched_uniprot_acc = Some(fetch_res.uniprot_id);
    }

    let struct_p = structure_path
        .ok_or("af-validate requires --structure <model.cif|model.pdb> (or --fetch-uniprot <id> --cache-dir <dir>)")?;
    let ingest_opts = AfIngestOptions {
        target_chain_id: chain_id.clone(),
        force_bfactor_plddt,
    };
    let structure = parse_af_structure_model(&struct_p, &ingest_opts)?;
    require_predicted_model(&structure, force_bfactor_plddt)?;
    let pae = match pae_path {
        Some(ref pp) => Some(parse_pae_json_file(
            pp,
            chain_id.as_deref(),
            Some(structure.residues.len()),
        )?),
        None => None,
    };

    // If `--fetch-uniprot` was invoked standalone without `--shifts`, report verified download & structure summary
    let Some(shifts_p) = shifts_path else {
        if fetched_uniprot_acc.is_some() {
            println!(
                "Fetched & verified AlphaFold model '{}' (chain {}, L = {} aa, PAE = {:?})",
                struct_p.display(),
                structure.chain_id,
                structure.residues.len(),
                pae.as_ref().map(|m| m.n_residues)
            );
            return Ok(());
        }
        return Err("af-validate requires --shifts <assigned.str|assigned.nef>".to_string());
    };

    let shifts = read_shifts_star_or_nef(&shifts_p)?;
    let s2_entries = match s2_path {
        Some(ref sp) => Some(parse_modelfree_s2_tsv_file(sp)?),
        None => None,
    };

    let cfg = AfConcordanceConfig {
        temperature_kelvin: temp_k,
        sample_ph,
        exclude_pdb_ids: exclude_pdb,
        exclude_bmrb_ids: exclude_bmrb,
        exclude_uniprot_id: fetched_uniprot_acc,
        exclude_topology_id: exclude_topology,
        firewall_max_seq_identity: nmr_core::torsion_db::DEFAULT_FIREWALL_MAX_SEQ_IDENTITY,
    };
    let report = evaluate_af_nmr_concordance(
        &structure,
        &shifts,
        pae.as_ref(),
        s2_entries.as_deref(),
        &cfg,
    )?;

    write_or_print_af_validate_output(&report, &shifts, out_path.as_deref())
}

fn write_or_print_af_validate_output(
    report: &AfConcordanceReport,
    shifts: &Reference,
    out_path: Option<&Path>,
) -> Result<(), String> {
    if let Some(op) = out_path {
        if let Some(parent) = op.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("failed to create '{}': {e}", parent.display()))?;
            }
        }
        let ext = op
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "json" => {
                let mut js = format_af_concordance_json(report)?;
                if !js.ends_with('\n') {
                    js.push('\n');
                }
                fs::write(op, js)
                    .map_err(|e| format!("failed to write '{}': {e}", op.display()))?;
            }
            "tsv" => {
                let tsv = format_af_concordance_tsv(report);
                fs::write(op, tsv)
                    .map_err(|e| format!("failed to write '{}': {e}", op.display()))?;
            }
            "html" | "htm" => {
                let mut conf = BTreeMap::new();
                for k in 0..shifts.sequence.len() {
                    let has_shift = ["H", "N", "CA", "CB", "C"]
                        .iter()
                        .any(|a| shifts.shifts.contains_key(&(k, (*a).to_string())));
                    if has_shift {
                        conf.insert(k, 1.0);
                    }
                }
                let proj = report
                    .entry_id
                    .clone()
                    .unwrap_or_else(|| format!("Chain_{}", report.chain_id));
                let input = ReportInput::new(proj, shifts.clone(), conf)
                    .with_af_concordance_report(report.clone());
                write_quality_report_html(op, &input)?;
            }
            other => {
                return Err(format!(
                    "unsupported af-validate output extension .{other} on '{}' (expected .json, .tsv, or .html)",
                    op.display()
                ));
            }
        }
        return Ok(());
    }

    // Default stdout summary + TSV
    match &report.summary {
        ConcordanceSummary::Evaluated {
            n_eval,
            spearman_rho_s,
            rmsd_rigidity,
            ..
        } => {
            let ci_str = report
                .spearman_rho_ci95
                .map(|[lo, hi]| format!(" [{lo:.4}, {hi:.4}]"))
                .unwrap_or_default();
            let core_str = report
                .ordered_core_spearman_rho
                .map(|r| format!("{r:.4}"))
                .unwrap_or_else(|| "undefined".to_string());
            let core_ci_str = report
                .ordered_core_spearman_rho_ci95
                .map(|[lo, hi]| format!(" [{lo:.4}, {hi:.4}]"))
                .unwrap_or_default();
            println!(
                "AF-vs-NMR Concordance (chain {}, n_eval = {}, rigidity = {}): Spearman rho_s = {:.4}{}, ordered-core (pLDDT>=70, n = {}) rho_s = {}{}, RMSD_rig = {:.4}",
                report.chain_id,
                n_eval,
                report.rigidity_source.provenance_label(),
                spearman_rho_s,
                ci_str,
                report.n_ordered_core,
                core_str,
                core_ci_str,
                rmsd_rigidity
            );
        }
        ConcordanceSummary::InsufficientCoverage { n_eval, reason } => {
            println!(
                "AF-vs-NMR Concordance (chain {}): InsufficientCoverage (n_eval = {}, spearman_rho = undefined): {}",
                report.chain_id, n_eval, reason
            );
        }
    }
    print!("{}", format_af_concordance_tsv(report));
    Ok(())
}
