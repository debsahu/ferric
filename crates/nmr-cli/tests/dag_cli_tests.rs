//! End-to-end CLI integration tests for `ferric dag` (`preset`, `validate`, `compile`, `run`),
//! covering real `nmr-core` stage execution, content-addressed cache persistence, and fail-closed
//! manifest integrity rejection (`T22.11` / `22.11:P1-1` / `22.11:P1-3`).

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use nmr_core::workflow_dag::{
    compile_dag_manifest, dag_to_json, manifest_to_json, preset_backbone_assignment_dag, DagEdge,
    DagNode, NodeKind, WorkflowDag,
};
use serde_json::Value;

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(1);

struct ScopedTempDir {
    path: PathBuf,
}

impl ScopedTempDir {
    fn new(prefix: &str) -> Self {
        let id = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "ferric_dag_cli_{prefix}_{}_{}",
            std::process::id(),
            id
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        Self { path: dir }
    }
}

impl Drop for ScopedTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn ferric_bin() -> &'static str {
    env!("CARGO_BIN_EXE_ferric")
}

fn ubiquitin_star_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fixtures/synthetic_ubiquitin.str")
}

#[test]
fn test_ferric_dag_cli_help_preset_validate_compile_and_real_backbone_run_with_cache() {
    let tmp = ScopedTempDir::new("backbone_e2e");

    // 1. Top-level usage & `ferric dag --help`
    let top = Command::new(ferric_bin())
        .arg("--help")
        .output()
        .expect("run ferric --help");
    assert!(top.status.success());
    assert!(String::from_utf8_lossy(&top.stdout).contains("ferric dag"));

    let help = Command::new(ferric_bin())
        .args(["dag", "--help"])
        .output()
        .expect("run ferric dag --help");
    assert!(help.status.success());

    // 2. `ferric dag preset backbone -o dag.json`
    let dag_path = tmp.path.join("backbone_dag.json");
    let preset_out = Command::new(ferric_bin())
        .args([
            "dag",
            "preset",
            "backbone",
            "-o",
            &dag_path.to_string_lossy(),
        ])
        .output()
        .expect("run ferric dag preset");
    assert!(
        preset_out.status.success(),
        "preset failed: {}",
        String::from_utf8_lossy(&preset_out.stderr)
    );
    assert!(dag_path.is_file());

    // 3. `ferric dag validate --in dag.json --json`
    let val_out = Command::new(ferric_bin())
        .args([
            "dag",
            "validate",
            "--in",
            &dag_path.to_string_lossy(),
            "--json",
        ])
        .output()
        .expect("run ferric dag validate");
    assert!(
        val_out.status.success(),
        "validate failed: {}",
        String::from_utf8_lossy(&val_out.stderr)
    );
    let val_json: Value =
        serde_json::from_slice(&val_out.stdout).expect("parse validate JSON output");
    assert_eq!(val_json["valid"], Value::Bool(true));
    assert_eq!(val_json["node_count"], Value::from(5));

    // 4. `ferric dag compile --in dag.json -o manifest.json`
    let manifest_path = tmp.path.join("backbone_manifest.json");
    let comp_out = Command::new(ferric_bin())
        .args([
            "dag",
            "compile",
            "--in",
            &dag_path.to_string_lossy(),
            "-o",
            &manifest_path.to_string_lossy(),
        ])
        .output()
        .expect("run ferric dag compile");
    assert!(
        comp_out.status.success(),
        "compile failed: {}",
        String::from_utf8_lossy(&comp_out.stderr)
    );
    let manifest_json: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read compiled manifest"))
            .expect("parse compiled manifest");
    assert_eq!(manifest_json["schema"], Value::from("ferric-dag-v1"));

    // 5. Real `ferric dag run` on synthetic_ubiquitin.str with output dir + persistent cache
    let out_dir = tmp.path.join("out_real");
    let cache_path = tmp.path.join("dag_cache.json");
    let trace1_path = tmp.path.join("trace1.json");
    let run1 = Command::new(ferric_bin())
        .args([
            "dag",
            "run",
            "--in",
            &manifest_path.to_string_lossy(),
            "--star",
            &ubiquitin_star_path().to_string_lossy(),
            "--out-dir",
            &out_dir.to_string_lossy(),
            "--cache",
            &cache_path.to_string_lossy(),
            "--runs",
            "4",
            "--steps",
            "250",
            "--json-out",
            &trace1_path.to_string_lossy(),
        ])
        .output()
        .expect("run ferric dag run #1");
    assert!(
        run1.status.success(),
        "real dag run failed: {}",
        String::from_utf8_lossy(&run1.stderr)
    );
    let trace1: Value =
        serde_json::from_str(&fs::read_to_string(&trace1_path).expect("read trace1"))
            .expect("parse trace1 JSON");
    assert_eq!(trace1["executed_count"], Value::from(5));
    assert_eq!(trace1["cache_hit_count"], Value::from(0));
    assert_eq!(trace1["metrics"]["n_residues"], Value::from(76));
    assert!(
        trace1["metrics"]["n_assigned"].as_u64().unwrap_or(0) > 10,
        "expected real assigned residues > 10, got {}",
        trace1["metrics"]
    );
    let nef_file = out_dir.join("reports/backbone_assignments.nef");
    assert!(nef_file.is_file(), "missing real NEF output");
    let nef_text = fs::read_to_string(&nef_file).expect("read NEF output");
    assert!(nef_text.contains("data_nef_"));
    assert!(out_dir
        .join("reports/backbone_assignments.nef.report.html")
        .is_file());
    assert!(out_dir.join("dag_trace.json").is_file());

    // 6. Second `ferric dag run` with same cache -> 0 executed, 5 cache hits
    let trace2_path = tmp.path.join("trace2.json");
    let run2 = Command::new(ferric_bin())
        .args([
            "dag",
            "run",
            "--in",
            &manifest_path.to_string_lossy(),
            "--star",
            &ubiquitin_star_path().to_string_lossy(),
            "--cache",
            &cache_path.to_string_lossy(),
            "--json-out",
            &trace2_path.to_string_lossy(),
        ])
        .output()
        .expect("run ferric dag run #2");
    assert!(
        run2.status.success(),
        "cached dag run failed: {}",
        String::from_utf8_lossy(&run2.stderr)
    );
    let trace2: Value =
        serde_json::from_str(&fs::read_to_string(&trace2_path).expect("read trace2"))
            .expect("parse trace2 JSON");
    assert_eq!(trace2["executed_count"], Value::from(0));
    assert_eq!(trace2["cache_hit_count"], Value::from(5));
}

#[test]
fn test_ferric_dag_cli_metabolomics_demo_and_fail_closed_integrity_rejection() {
    let tmp = ScopedTempDir::new("metab_and_security");

    // 1. Metabolomics preset real execution via `--demo`
    let metab_dag = tmp.path.join("metab_dag.json");
    let p = Command::new(ferric_bin())
        .args([
            "dag",
            "preset",
            "metabolomics",
            "-o",
            &metab_dag.to_string_lossy(),
        ])
        .output()
        .expect("preset metabolomics");
    assert!(p.status.success());

    let metab_out = tmp.path.join("metab_out");
    let metab_trace = tmp.path.join("metab_trace.json");
    let r = Command::new(ferric_bin())
        .args([
            "dag",
            "run",
            "--in",
            &metab_dag.to_string_lossy(),
            "--demo",
            "--out-dir",
            &metab_out.to_string_lossy(),
            "--json-out",
            &metab_trace.to_string_lossy(),
        ])
        .output()
        .expect("run metabolomics dag");
    assert!(
        r.status.success(),
        "metabolomics dag run failed: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    let mt: Value =
        serde_json::from_str(&fs::read_to_string(&metab_trace).expect("read metab_trace"))
            .expect("parse metab_trace");
    assert_eq!(mt["executed_count"], Value::from(4));
    assert!(mt["metrics"]["n_samples"].as_u64().unwrap_or(0) >= 8);
    assert!(mt["metrics"]["n_buckets"].as_u64().unwrap_or(0) > 10);

    // 2. Fail-closed on missing data without --demo / --dry-run
    let missing_data = Command::new(ferric_bin())
        .args(["dag", "run", "--in", &metab_dag.to_string_lossy()])
        .output()
        .expect("run without data");
    assert!(
        !missing_data.status.success(),
        "must fail closed when no input data and neither --demo nor --dry-run is set"
    );

    // 3. Claude's exact forged manifest probe via CLI (`reversed nodes` + `node_sha256 = "forged"`)
    let mut forged = compile_dag_manifest(&preset_backbone_assignment_dag()).expect("compile");
    forged.nodes.reverse();
    for node in &mut forged.nodes {
        node.node_sha256 = "forged".to_string();
    }
    let forged_path = tmp.path.join("forged_manifest.json");
    fs::write(
        &forged_path,
        manifest_to_json(&forged).expect("serialize forged"),
    )
    .expect("write forged manifest");

    let forged_dry = Command::new(ferric_bin())
        .args([
            "dag",
            "run",
            "--in",
            &forged_path.to_string_lossy(),
            "--dry-run",
        ])
        .output()
        .expect("run forged manifest dry-run");
    assert!(
        !forged_dry.status.success(),
        "CLI must reject forged manifest fail-closed"
    );
    let stderr = String::from_utf8_lossy(&forged_dry.stderr);
    assert!(
        stderr.contains("Manifest integrity mismatch"),
        "expected integrity mismatch error in stderr, got: {stderr}"
    );

    // 4. Invalid DAG (port type mismatch) rejected by `ferric dag validate`
    let bad_dag = WorkflowDag {
        name: "Bad".to_string(),
        description: String::new(),
        nodes: vec![
            DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
            DagNode::new("chem", NodeKind::MetabolomicsChemometrics, "Chem"),
        ],
        edges: vec![DagEdge::new("ingest", "raw_fid", "chem", "spectrum_1d")],
    };
    let bad_path = tmp.path.join("bad_dag.json");
    fs::write(&bad_path, dag_to_json(&bad_dag).expect("serialize bad dag")).expect("write bad dag");
    let bad_val = Command::new(ferric_bin())
        .args(["dag", "validate", "--in", &bad_path.to_string_lossy()])
        .output()
        .expect("validate bad dag");
    assert!(!bad_val.status.success());
}
