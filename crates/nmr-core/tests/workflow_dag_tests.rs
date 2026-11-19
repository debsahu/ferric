//! Integration tests for `nmr_core::workflow_dag` (`T22.11`):
//! static typed-port validation, Kahn's deterministic topological sort, cycle detection,
//! hostile bounds & path-traversal rejection, canonical SHA-256 content-addressed manifests,
//! and incremental upstream cache reuse.

use std::collections::BTreeMap;

use nmr_core::nmrstar::read_star_str;
use nmr_core::peaks::Peak;
use nmr_core::simulate::{render_hsqc_2d, simulate, SimConfig, SpectrumRenderConfig};
use nmr_core::spectrum::write_nmrpipe_2d;
use nmr_core::workflow_dag::{
    compile_dag_manifest, dag_from_json, dag_to_json, execute_dag_manifest,
    execute_dag_manifest_dry_run, execute_dag_manifest_real, manifest_from_json, manifest_to_json,
    preset_backbone_assignment_dag, preset_by_name, preset_dynamics_relax_cest_dag,
    preset_metabolomics_1d_dag, validate_candidate_edge, validate_dag, verify_dag_manifest,
    CachedArtifact, DagArtifactCache, DagEdge, DagError, DagNode, DagRunOptions, NodeKind,
    PortDirection, PortType, StepStatus, WorkflowDag, DAG_MANIFEST_SCHEMA, MAX_DAG_EDGES,
    MAX_DAG_JSON_BYTES, MAX_DAG_NODES, SIMULATION_DEMO_PROVENANCE_TAG,
};

fn format_peaks_tsv(peaks: &[Peak]) -> String {
    let mut out = String::from("experiment\tH\tN\tC\tsign\theight\n");
    for p in peaks {
        let c_str =
            p.c.map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| ".".to_string());
        let h_str = p
            .height
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "100.0".to_string());
        out.push_str(&format!(
            "{}\t{:.4}\t{:.4}\t{c_str}\t{}\t{h_str}\n",
            p.exp, p.h, p.n, p.sign
        ));
    }
    out
}

fn is_valid_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[test]
fn test_all_three_presets_validate_compile_and_dry_run() {
    let presets = [
        (preset_backbone_assignment_dag(), 5usize, 5usize),
        (preset_dynamics_relax_cest_dag(), 6usize, 6usize),
        (preset_metabolomics_1d_dag(), 4usize, 3usize),
    ];

    for (dag, expected_nodes, expected_edges) in presets {
        assert_eq!(dag.nodes.len(), expected_nodes);
        assert_eq!(dag.edges.len(), expected_edges);

        let validated = validate_dag(&dag).expect("preset DAG must validate");
        assert_eq!(validated.topological_order.len(), expected_nodes);

        let manifest = compile_dag_manifest(&dag).expect("preset DAG must compile");
        assert_eq!(manifest.schema, DAG_MANIFEST_SCHEMA);
        assert!(is_valid_sha256_hex(&manifest.manifest_sha256));
        assert_eq!(manifest.nodes.len(), expected_nodes);
        assert_eq!(manifest.edges.len(), expected_edges);
        for node in &manifest.nodes {
            assert!(is_valid_sha256_hex(&node.node_sha256));
        }

        let json = manifest_to_json(&manifest).expect("manifest JSON serialization");
        assert!(json.contains("\"schema\": \"ferric-dag-v1\""));

        let mut cache = DagArtifactCache::new();
        let trace =
            execute_dag_manifest_dry_run(&manifest, &mut cache).expect("preset dry-run execution");
        assert_eq!(trace.executed_count, expected_nodes);
        assert_eq!(trace.cache_hit_count, 0);
        assert_eq!(trace.steps.len(), expected_nodes);
    }
}

#[test]
fn test_topological_and_manifest_sha256_determinism_under_permutations() {
    let base = preset_dynamics_relax_cest_dag();
    let base_manifest = compile_dag_manifest(&base).expect("compile base dynamics DAG");

    // Verify deterministic lexicographical tie-breaking: "exchange" < "relax"
    assert_eq!(
        base_manifest.topological_order,
        vec!["ingest", "process", "pick", "exchange", "relax", "export"]
    );

    // Reverse nodes, reverse edges, shift canvas positions, and insert nested object keys out of order
    let mut permuted = base.clone();
    permuted.nodes.reverse();
    permuted.edges.reverse();
    for (idx, node) in permuted.nodes.iter_mut().enumerate() {
        node.position = [100.0 + idx as f32 * 37.0, 220.0 - idx as f32 * 13.0];
    }

    let perm_manifest = compile_dag_manifest(&permuted).expect("compile permuted dynamics DAG");
    assert_eq!(
        perm_manifest.topological_order,
        base_manifest.topological_order
    );
    assert_eq!(perm_manifest.manifest_sha256, base_manifest.manifest_sha256);
    assert_eq!(perm_manifest.nodes, base_manifest.nodes);
    assert_eq!(perm_manifest.edges, base_manifest.edges);

    // Rotate nodes/edges by multiple offsets to verify permutation invariance
    for rot in 1..base.nodes.len() {
        let mut rotated = base.clone();
        rotated.nodes.rotate_left(rot);
        rotated.edges.rotate_right(rot % base.edges.len());
        let rot_manifest = compile_dag_manifest(&rotated).expect("compile rotated DAG");
        assert_eq!(rot_manifest.manifest_sha256, base_manifest.manifest_sha256);
        assert_eq!(
            rot_manifest.topological_order,
            base_manifest.topological_order
        );
    }

    // Verify nested JSON object parameter key-order invariance
    let mut nested_a = base.clone();
    let mut nested_b = base.clone();
    let val_a: serde_json::Value =
        serde_json::from_str(r#"{"z":1,"a":{"k2":true,"k1":3}}"#).expect("nested JSON A");
    let val_b: serde_json::Value =
        serde_json::from_str(r#"{"a":{"k1":3,"k2":true},"z":1}"#).expect("nested JSON B");
    nested_a.nodes[0]
        .params
        .insert("solver_opts".to_string(), val_a);
    nested_b.nodes[0]
        .params
        .insert("solver_opts".to_string(), val_b);
    let ma = compile_dag_manifest(&nested_a).expect("compile nested_a");
    let mb = compile_dag_manifest(&nested_b).expect("compile nested_b");
    assert_eq!(ma.manifest_sha256, mb.manifest_sha256);
    assert_eq!(ma.nodes[0].node_sha256, mb.nodes[0].node_sha256);
}

#[test]
fn test_cycle_detection_direct_indirect_and_self_loop() {
    // 1. Self-loop on SidechainAssign (assignments -> assignments)
    let self_loop = WorkflowDag {
        name: "Self Loop".to_string(),
        description: String::new(),
        nodes: vec![DagNode::new(
            "sc_a",
            NodeKind::SidechainAssign,
            "Sidechain A",
        )],
        edges: vec![DagEdge::new("sc_a", "assignments", "sc_a", "assignments")],
    };
    assert_eq!(
        validate_dag(&self_loop),
        Err(DagError::CycleDetected {
            cycle_nodes: vec!["sc_a".to_string()],
        })
    );

    // 2. Two-node mutual cycle (sc_a <-> sc_b)
    let mutual_cycle = WorkflowDag {
        name: "Mutual Cycle".to_string(),
        description: String::new(),
        nodes: vec![
            DagNode::new("sc_b", NodeKind::SidechainAssign, "Sidechain B"),
            DagNode::new("sc_a", NodeKind::SidechainAssign, "Sidechain A"),
        ],
        edges: vec![
            DagEdge::new("sc_a", "assignments", "sc_b", "assignments"),
            DagEdge::new("sc_b", "assignments", "sc_a", "assignments"),
        ],
    };
    assert_eq!(
        validate_dag(&mutual_cycle),
        Err(DagError::CycleDetected {
            cycle_nodes: vec!["sc_a".to_string(), "sc_b".to_string()],
        })
    );

    // 3. Cycle with an upstream source and a downstream sink: only cycle participants reported
    let cycle_with_tails = WorkflowDag {
        name: "Cycle With Tails".to_string(),
        description: String::new(),
        nodes: vec![
            DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
            DagNode::new("pick", NodeKind::PeakPick, "Pick"),
            DagNode::new("dec1", NodeKind::DeconvolveNd, "Deconv 1"),
            DagNode::new("dec2", NodeKind::DeconvolveNd, "Deconv 2"),
            DagNode::new("export", NodeKind::ExportNefReport, "Export"),
        ],
        edges: vec![
            DagEdge::new("ingest", "spectrum_2d", "pick", "spectrum_2d"),
            DagEdge::new("pick", "peaks", "export", "peaks"),
            DagEdge::new("dec1", "peaks", "dec2", "seed_peaks"),
            DagEdge::new("dec2", "peaks", "dec1", "seed_peaks"),
        ],
    };
    assert_eq!(
        validate_dag(&cycle_with_tails),
        Err(DagError::CycleDetected {
            cycle_nodes: vec!["dec1".to_string(), "dec2".to_string()],
        })
    );
}

#[test]
fn test_port_type_mismatch_duplicate_missing_and_dangling_required_inputs() {
    // 1. Port type mismatch: RawFid -> ProcessedSpectrum1D
    let mismatch = WorkflowDag {
        name: "Type Mismatch".to_string(),
        description: String::new(),
        nodes: vec![
            DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
            DagNode::new("chem", NodeKind::MetabolomicsChemometrics, "Chemometrics"),
        ],
        edges: vec![DagEdge::new("ingest", "raw_fid", "chem", "spectrum_1d")],
    };
    assert_eq!(
        validate_dag(&mismatch),
        Err(DagError::PortTypeMismatch {
            edge: Box::new(DagEdge::new("ingest", "raw_fid", "chem", "spectrum_1d")),
            from_type: PortType::RawFid,
            to_type: PortType::ProcessedSpectrum1D,
        })
    );

    // 2. Duplicate edge
    let mut dup_edge = preset_metabolomics_1d_dag();
    dup_edge
        .edges
        .push(DagEdge::new("ingest", "raw_fid", "process", "raw_fid"));
    assert_eq!(
        validate_dag(&dup_edge),
        Err(DagError::DuplicateEdge {
            edge: Box::new(DagEdge::new("ingest", "raw_fid", "process", "raw_fid")),
        })
    );

    // 3. Input port fan-in conflict (two different upstream nodes into same input port)
    let mut fan_in = preset_metabolomics_1d_dag();
    fan_in.nodes.push(DagNode::new(
        "ingest2",
        NodeKind::IngestBrukerVarian,
        "Ingest 2",
    ));
    fan_in
        .edges
        .push(DagEdge::new("ingest2", "raw_fid", "process", "raw_fid"));
    assert_eq!(
        validate_dag(&fan_in),
        Err(DagError::InputPortAlreadyConnected {
            node_id: "process".to_string(),
            port: "raw_fid".to_string(),
        })
    );

    // 4. Missing node referenced by edge
    let mut missing_node = preset_metabolomics_1d_dag();
    missing_node
        .edges
        .push(DagEdge::new("ghost", "raw_fid", "process", "raw_fid"));
    assert!(matches!(
        validate_dag(&missing_node),
        Err(DagError::MissingNode { missing_node_id, .. }) if missing_node_id == "ghost"
    ));

    // 5. Unknown output & input ports
    let bad_out = WorkflowDag {
        name: "Bad Out".to_string(),
        description: String::new(),
        nodes: vec![
            DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
            DagNode::new("process", NodeKind::ProcessFidNus, "Process"),
        ],
        edges: vec![DagEdge::new("ingest", "nonexistent", "process", "raw_fid")],
    };
    assert!(matches!(
        validate_dag(&bad_out),
        Err(DagError::UnknownPort {
            direction: PortDirection::Output,
            ..
        })
    ));

    let bad_in = WorkflowDag {
        name: "Bad In".to_string(),
        description: String::new(),
        nodes: vec![
            DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
            DagNode::new("process", NodeKind::ProcessFidNus, "Process"),
        ],
        edges: vec![DagEdge::new("ingest", "raw_fid", "process", "nonexistent")],
    };
    assert!(matches!(
        validate_dag(&bad_in),
        Err(DagError::UnknownPort {
            direction: PortDirection::Input,
            ..
        })
    ));

    // 6. Duplicate node ID
    let mut dup_node = preset_metabolomics_1d_dag();
    dup_node.nodes.push(DagNode::new(
        "ingest",
        NodeKind::IngestBrukerVarian,
        "Duplicate",
    ));
    assert_eq!(
        validate_dag(&dup_node),
        Err(DagError::DuplicateNodeId {
            node_id: "ingest".to_string(),
        })
    );

    // 7. Dangling required input (explicit required port)
    let dangling_explicit = WorkflowDag {
        name: "Dangling Explicit".to_string(),
        description: String::new(),
        nodes: vec![DagNode::new("process", NodeKind::ProcessFidNus, "Process")],
        edges: Vec::new(),
    };
    assert_eq!(
        validate_dag(&dangling_explicit),
        Err(DagError::MissingRequiredInput {
            node_id: "process".to_string(),
            port: "raw_fid".to_string(),
        })
    );

    // 8. Dangling required polymorphic input (`PeakPick` requires at least one spectrum)
    let dangling_any = WorkflowDag {
        name: "Dangling PeakPick".to_string(),
        description: String::new(),
        nodes: vec![DagNode::new("pick", NodeKind::PeakPick, "Pick")],
        edges: Vec::new(),
    };
    assert_eq!(
        validate_dag(&dangling_any),
        Err(DagError::MissingRequiredInput {
            node_id: "pick".to_string(),
            port: "<any_input>".to_string(),
        })
    );

    // 9. Empty DAG
    let empty = WorkflowDag {
        name: "Empty".to_string(),
        description: String::new(),
        nodes: Vec::new(),
        edges: Vec::new(),
    };
    assert_eq!(validate_dag(&empty), Err(DagError::EmptyDag));
}

#[test]
fn test_hostile_bounds_control_chars_and_path_traversal_rejected() {
    // 1. Too many nodes (> MAX_DAG_NODES)
    let too_many_nodes = WorkflowDag {
        name: "Oversized Nodes".to_string(),
        description: String::new(),
        nodes: (0..=MAX_DAG_NODES)
            .map(|i| DagNode::new(format!("n_{i}"), NodeKind::IngestBrukerVarian, "Ingest"))
            .collect(),
        edges: Vec::new(),
    };
    assert_eq!(
        validate_dag(&too_many_nodes),
        Err(DagError::TooManyNodes {
            count: MAX_DAG_NODES + 1,
            max: MAX_DAG_NODES,
        })
    );

    // 2. Too many edges (> MAX_DAG_EDGES)
    let too_many_edges = WorkflowDag {
        name: "Oversized Edges".to_string(),
        description: String::new(),
        nodes: vec![
            DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
            DagNode::new("process", NodeKind::ProcessFidNus, "Process"),
        ],
        edges: (0..=MAX_DAG_EDGES)
            .map(|_| DagEdge::new("ingest", "raw_fid", "process", "raw_fid"))
            .collect(),
    };
    assert_eq!(
        validate_dag(&too_many_edges),
        Err(DagError::TooManyEdges {
            count: MAX_DAG_EDGES + 1,
            max: MAX_DAG_EDGES,
        })
    );

    // 3. Oversized JSON payload (> MAX_DAG_JSON_BYTES)
    let huge_json = " ".repeat(MAX_DAG_JSON_BYTES + 1);
    assert_eq!(
        dag_from_json(&huge_json),
        Err(DagError::JsonTooLarge {
            bytes: MAX_DAG_JSON_BYTES + 1,
            max: MAX_DAG_JSON_BYTES,
        })
    );

    // 4. Path traversal & absolute paths in manifest params
    for hostile_path in [
        "../secret/data",
        "datasets/../../etc/passwd",
        "/etc/passwd",
        "C:\\Windows\\System32",
        "datasets\\subdir\\fid",
        "datasets/bad\0null",
    ] {
        let mut dag = preset_metabolomics_1d_dag();
        dag.nodes[0]
            .params
            .insert("dataset_path".to_string(), serde_json::json!(hostile_path));
        assert!(
            matches!(validate_dag(&dag), Err(DagError::InvalidParam { .. })),
            "expected hostile path {hostile_path:?} to be rejected"
        );
    }

    // 5. Control character in node ID and label
    let mut bad_id = preset_metabolomics_1d_dag();
    bad_id.nodes[0].id = "ingest\n1".to_string();
    assert!(matches!(
        validate_dag(&bad_id),
        Err(DagError::InvalidNodeId { .. })
    ));

    let mut bad_label = preset_metabolomics_1d_dag();
    bad_label.nodes[0].label = "Ingest\tTab".to_string();
    assert!(matches!(
        validate_dag(&bad_label),
        Err(DagError::InvalidParam { .. })
    ));

    // 6. Invalid domain numeric bounds
    let mut bad_zf = preset_metabolomics_1d_dag();
    bad_zf.nodes[1]
        .params
        .insert("zero_fill_factor".to_string(), serde_json::json!(0));
    assert!(matches!(
        validate_dag(&bad_zf),
        Err(DagError::InvalidParam { key, .. }) if key == "zero_fill_factor"
    ));

    let mut bad_snr = preset_backbone_assignment_dag();
    bad_snr.nodes[2]
        .params
        .insert("snr_threshold".to_string(), serde_json::json!(-2.5));
    assert!(matches!(
        validate_dag(&bad_snr),
        Err(DagError::InvalidParam { key, .. }) if key == "snr_threshold"
    ));

    // 7. Non-finite canvas positions (`NaN` / `Infinity`) rejected fail-closed
    for bad_pos in [
        [f32::NAN, 10.0],
        [10.0, f32::INFINITY],
        [f32::NEG_INFINITY, 0.0],
    ] {
        let mut dag = preset_metabolomics_1d_dag();
        dag.nodes[0].position = bad_pos;
        assert!(matches!(
            validate_dag(&dag),
            Err(DagError::InvalidParam { key, .. }) if key == "position"
        ));
    }
}

#[test]
fn test_adversarial_multibyte_utf8_manifest_sha_fails_closed_without_panic() {
    let mut manifest =
        compile_dag_manifest(&preset_metabolomics_1d_dag()).expect("compile metabolomics preset");
    // Multi-byte UTF-8 character straddling byte offset 12 (11 ASCII bytes + 2-byte U+00E9)
    let crafted_sha = "aaaaaaaaaaa\u{00E9}bbbb".to_string();
    assert!(!crafted_sha.is_char_boundary(12));
    manifest.manifest_sha256 = crafted_sha.clone();
    manifest.nodes[0].node_sha256 = crafted_sha;

    let mut cache = DagArtifactCache::new();
    let err = execute_dag_manifest_dry_run(&manifest, &mut cache)
        .expect_err("forged multibyte UTF-8 manifest digest must fail closed");
    assert!(
        matches!(err, DagError::ManifestIntegrityMismatch { .. }),
        "expected ManifestIntegrityMismatch, got {err:?}"
    );
    assert!(err.to_string().contains("aaaaaaaaaaa\u{00E9}"));
}

#[test]
fn test_p1_3_manifest_integrity_rejects_reversed_nodes_forged_hashes_and_reloads_cleanly() {
    let dag = preset_backbone_assignment_dag();
    let manifest = compile_dag_manifest(&dag).expect("compile backbone preset");

    // 1. Claude's exact adversarial probe:
    //    Reverse nodes AND set every node_sha256 = "forged" with a pre-poisoned cache.
    //    MUST fail closed with Err(DagError::ManifestIntegrityMismatch), NEVER Ok(executed=1, cache_hit=4).
    let mut forged = manifest.clone();
    forged.nodes.reverse();
    for node in &mut forged.nodes {
        node.node_sha256 = "forged".to_string();
    }
    let mut poisoned_cache = DagArtifactCache::new();
    poisoned_cache.insert(CachedArtifact {
        cache_key: "forged".to_string(),
        node_sha256: "forged".to_string(),
        kind: NodeKind::ExportNefReport,
        input_fingerprint: "dry_run".to_string(),
        dry_run: true,
        output_digest: "stale".to_string(),
        summary: "stale cached output".to_string(),
        payload_json: None,
    });
    assert!(
        matches!(
            execute_dag_manifest_dry_run(&forged, &mut poisoned_cache),
            Err(DagError::ManifestIntegrityMismatch { .. })
        ),
        "reversed + forged node_sha256 manifest must be rejected fail-closed"
    );

    // 2. Reversed nodes alone (even with valid node_sha256 values) must be rejected
    let mut reversed_only = manifest.clone();
    reversed_only.nodes.reverse();
    assert!(matches!(
        verify_dag_manifest(&reversed_only),
        Err(DagError::ManifestIntegrityMismatch { .. })
    ));
    assert!(matches!(
        execute_dag_manifest_dry_run(&reversed_only, &mut poisoned_cache),
        Err(DagError::ManifestIntegrityMismatch { .. })
    ));

    // 3. Forged manifest_sha256 alone must be rejected
    let mut forged_manifest_sha = manifest.clone();
    forged_manifest_sha.manifest_sha256 = "0".repeat(64);
    assert!(matches!(
        execute_dag_manifest_dry_run(&forged_manifest_sha, &mut poisoned_cache),
        Err(DagError::ManifestIntegrityMismatch { .. })
    ));

    // 4. Mutated node param without recompiling must be rejected
    let mut mutated_param = manifest.clone();
    mutated_param.nodes[2]
        .params
        .insert("snr_threshold".to_string(), serde_json::json!(99.0));
    assert!(matches!(
        execute_dag_manifest_dry_run(&mutated_param, &mut poisoned_cache),
        Err(DagError::ManifestIntegrityMismatch { .. })
    ));

    // 5. Tampered upstream_sha256 input binding must be rejected
    let mut tampered_binding = manifest.clone();
    tampered_binding.nodes[1].inputs[0].upstream_sha256 = "deadbeef".to_string();
    assert!(matches!(
        execute_dag_manifest_dry_run(&tampered_binding, &mut poisoned_cache),
        Err(DagError::ManifestIntegrityMismatch { .. })
    ));

    // 6. manifest_to_json -> manifest_from_json round-trip verifies integrity and re-executes cleanly
    let json = manifest_to_json(&manifest).expect("serialize valid manifest");
    let reloaded = manifest_from_json(&json).expect("reload and verify valid manifest");
    assert_eq!(reloaded, manifest);

    let mut clean_cache = DagArtifactCache::new();
    let t1 = execute_dag_manifest_dry_run(&reloaded, &mut clean_cache).expect("run 1");
    assert_eq!(t1.executed_count, 5);
    assert_eq!(t1.cache_hit_count, 0);
    let t2 = execute_dag_manifest_dry_run(&reloaded, &mut clean_cache).expect("run 2");
    assert_eq!(t2.executed_count, 0);
    assert_eq!(t2.cache_hit_count, 5);

    // 7. manifest_from_json rejects forged JSON and oversized JSON fail-closed
    let forged_json = manifest_to_json(&forged).expect("serialize forged manifest for probe");
    assert!(matches!(
        manifest_from_json(&forged_json),
        Err(DagError::ManifestIntegrityMismatch { .. })
    ));
    let huge_json = " ".repeat(MAX_DAG_JSON_BYTES + 1);
    assert_eq!(
        manifest_from_json(&huge_json),
        Err(DagError::JsonTooLarge {
            bytes: MAX_DAG_JSON_BYTES + 1,
            max: MAX_DAG_JSON_BYTES,
        })
    );
}

#[test]
fn test_real_stage_execution_backbone_and_metabolomics_and_fail_closed_missing_input() {
    assert!(preset_by_name("backbone").is_some());
    assert!(preset_by_name("dynamics").is_some());
    assert!(preset_by_name("metabolomics").is_some());
    assert!(preset_by_name("unknown").is_none());

    // 1. Fail-closed when no input data, no demo fixture, and not dry-run
    let bb_manifest =
        compile_dag_manifest(&preset_backbone_assignment_dag()).expect("compile backbone");
    let mut empty_cache = DagArtifactCache::new();
    let missing_err =
        execute_dag_manifest_real(&bb_manifest, &mut empty_cache, &DagRunOptions::default())
            .expect_err("must fail closed without input data or demo flag");
    assert!(matches!(missing_err, DagError::ExecutionFailed { .. }));

    // 2. Real execution on synthetic_ubiquitin.str with explicit `allow_simulation_demo: true`
    //    stamps NEF, HTML, and trace with `SIMULATION_DEMO_PROVENANCE_TAG`, and repeat run restores
    //    byte-identical artifacts from cache.
    let star_text = include_str!("../../../data/fixtures/synthetic_ubiquitin.str").to_string();
    let tmp_dir =
        std::env::temp_dir().join(format!("ferric_dag_real_core_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    let opts = DagRunOptions {
        star_text: Some(star_text),
        output_dir: Some(tmp_dir.clone()),
        allow_simulation_demo: true,
        seed: 0,
        max_runs: Some(4),
        max_steps_per_system: Some(250),
        ..DagRunOptions::default()
    };
    let mut cache = DagArtifactCache::new();
    let out1 = execute_dag_manifest_real(&bb_manifest, &mut cache, &opts)
        .expect("real backbone DAG execution with allow_simulation_demo");
    assert_eq!(out1.trace.executed_count, 5);
    assert_eq!(out1.trace.cache_hit_count, 0);
    assert!(
        out1.trace.steps[2]
            .summary
            .contains("peak_pick hsqc_picked="),
        "PeakPick must report real 2D HSQC pick count: {}",
        out1.trace.steps[2].summary
    );
    assert!(
        out1.trace.steps[3].summary.contains("backbone_assign"),
        "BackboneAssign must run real solver: {}",
        out1.trace.steps[3].summary
    );
    assert!(
        out1.trace.steps[1]
            .summary
            .contains(SIMULATION_DEMO_PROVENANCE_TAG),
        "ProcessFidNus trace must carry SIMULATION_DEMO tag: {}",
        out1.trace.steps[1].summary
    );
    let nef_txt = out1.nef_text.as_deref().expect("real NEF text generated");
    assert!(nef_txt.contains("data_nef_"));
    assert!(
        nef_txt.contains(SIMULATION_DEMO_PROVENANCE_TAG),
        "NEF provenance must carry SIMULATION_DEMO tag"
    );
    let html_txt = out1
        .report_html
        .as_deref()
        .expect("real HTML quality report generated");
    assert!(html_txt.contains("<html") || html_txt.contains("<!DOCTYPE html>"));
    assert!(
        html_txt.contains(SIMULATION_DEMO_PROVENANCE_TAG),
        "HTML report must carry SIMULATION_DEMO tag"
    );
    assert_eq!(out1.written_files.len(), 2);
    for rel in &out1.written_files {
        let fpath = tmp_dir.join(rel);
        assert!(
            fpath.is_file(),
            "expected written artifact {}",
            fpath.display()
        );
    }
    let metrics = out1.metrics_json.as_ref().expect("real metrics_json");
    assert_eq!(metrics["n_residues"].as_u64(), Some(76));
    assert!(metrics["n_assigned"].as_u64().unwrap_or(0) > 10);

    // Repeat run hits cache for all 5 stages AND restores byte-identical real outputs
    let out2 = execute_dag_manifest_real(&bb_manifest, &mut cache, &opts)
        .expect("cached repeat real backbone DAG execution");
    assert_eq!(out2.trace.executed_count, 0);
    assert_eq!(out2.trace.cache_hit_count, 5);
    assert_eq!(out2.nef_text, out1.nef_text);
    assert_eq!(out2.report_html, out1.report_html);
    assert_eq!(out2.artifacts, out1.artifacts);
    assert_eq!(out2.metrics_json, out1.metrics_json);

    // 3. Real metabolomics 1D chemometrics execution with explicit demo fixture
    let metab_manifest =
        compile_dag_manifest(&preset_metabolomics_1d_dag()).expect("compile metabolomics");
    let mut metab_cache = DagArtifactCache::new();
    let metab_out = execute_dag_manifest_real(
        &metab_manifest,
        &mut metab_cache,
        &DagRunOptions {
            use_demo_fixture: true,
            allow_simulation_demo: true,
            seed: 42,
            ..DagRunOptions::default()
        },
    )
    .expect("real metabolomics DAG execution");
    assert_eq!(metab_out.trace.executed_count, 4);
    assert!(metab_out
        .artifacts
        .contains_key("metabolomics_summary.json"));
    let m_metrics = metab_out
        .metrics_json
        .as_ref()
        .expect("metabolomics metrics_json");
    assert!(m_metrics["n_samples"].as_u64().unwrap_or(0) >= 8);
    assert!(m_metrics["n_buckets"].as_u64().unwrap_or(0) > 10);

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_incremental_content_addressed_caching_reuses_upstream_and_sibling_branches() {
    let mut dag = preset_dynamics_relax_cest_dag();
    let manifest1 = compile_dag_manifest(&dag).expect("compile initial dynamics DAG");

    let mut cache = DagArtifactCache::new();
    let mut run1_executed: Vec<String> = Vec::new();
    let trace1 = execute_dag_manifest(&manifest1, &mut cache, |node, upstream| {
        run1_executed.push(node.id.clone());
        Ok(format!("ran:{}<-{}", node.id, upstream.len()))
    })
    .expect("first run");

    assert_eq!(trace1.executed_count, 6);
    assert_eq!(trace1.cache_hit_count, 0);
    assert_eq!(
        run1_executed,
        vec!["ingest", "process", "pick", "exchange", "relax", "export"]
    );
    assert_eq!(cache.len(), 6);

    // Second run on unchanged manifest: 100% cache hits
    let mut run2_executed: Vec<String> = Vec::new();
    let trace2 = execute_dag_manifest(&manifest1, &mut cache, |node, _upstream| {
        run2_executed.push(node.id.clone());
        Ok(format!("ran:{}", node.id))
    })
    .expect("second run");

    assert_eq!(trace2.executed_count, 0);
    assert_eq!(trace2.cache_hit_count, 6);
    assert!(run2_executed.is_empty());

    // Modify ONLY `exchange` parameter `b1_hz: 25.0 -> 50.0`
    let exchange_node = dag
        .nodes
        .iter_mut()
        .find(|n| n.id == "exchange")
        .expect("exchange node");
    exchange_node
        .params
        .insert("b1_hz".to_string(), serde_json::json!(50.0));

    let manifest2 = compile_dag_manifest(&dag).expect("compile updated dynamics DAG");
    assert_ne!(manifest2.manifest_sha256, manifest1.manifest_sha256);

    // Upstream (`ingest`, `process`, `pick`) AND sibling (`relax`) must keep bit-identical node_sha256!
    let hashes1: BTreeMap<&str, &str> = manifest1
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), n.node_sha256.as_str()))
        .collect();
    let hashes2: BTreeMap<&str, &str> = manifest2
        .nodes
        .iter()
        .map(|n| (n.id.as_str(), n.node_sha256.as_str()))
        .collect();
    for unchanged in ["ingest", "process", "pick", "relax"] {
        assert_eq!(
            hashes1[unchanged], hashes2[unchanged],
            "node '{unchanged}' hash must remain unchanged"
        );
    }
    for invalidated in ["exchange", "export"] {
        assert_ne!(
            hashes1[invalidated], hashes2[invalidated],
            "node '{invalidated}' hash must change"
        );
    }

    let mut run3_executed: Vec<String> = Vec::new();
    let trace3 = execute_dag_manifest(&manifest2, &mut cache, |node, upstream| {
        run3_executed.push(node.id.clone());
        Ok(format!("ran_v2:{}<-{}", node.id, upstream.len()))
    })
    .expect("incremental third run");

    assert_eq!(trace3.cache_hit_count, 4);
    assert_eq!(trace3.executed_count, 2);
    assert_eq!(run3_executed, vec!["exchange", "export"]);

    for step in &trace3.steps {
        match step.node_id.as_str() {
            "ingest" | "process" | "pick" | "relax" => {
                assert_eq!(step.status, StepStatus::CacheHit);
            }
            "exchange" | "export" => {
                assert_eq!(step.status, StepStatus::Executed);
            }
            other => panic!("unexpected node {other}"),
        }
    }
}

#[test]
fn test_all_12_node_kinds_all_13_port_types_and_json_roundtrip() {
    assert_eq!(PortType::ALL.len(), 13);
    assert_eq!(NodeKind::ALL.len(), 12);

    // Build a comprehensive composite DAG touching every NodeKind and every PortType
    let dag = WorkflowDag {
        name: "Full NMR Suite".to_string(),
        description: "Covers all 12 NMR stage templates and all 13 typed ports.".to_string(),
        nodes: vec![
            DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
            DagNode::new("process", NodeKind::ProcessFidNus, "Process"),
            DagNode::new("pick", NodeKind::PeakPick, "Pick"),
            DagNode::new("deconv", NodeKind::DeconvolveNd, "Deconvolve"),
            DagNode::new("bb", NodeKind::BackboneAssign, "Backbone"),
            DagNode::new("sc", NodeKind::SidechainAssign, "Sidechain"),
            DagNode::new("relax", NodeKind::DynamicsRelax, "Relax"),
            DagNode::new("exch", NodeKind::ExchangeCestCpmg, "Exchange"),
            DagNode::new("titr", NodeKind::Titration2d, "Titration"),
            DagNode::new("fold", NodeKind::StructureFoldDock, "Fold/Dock"),
            DagNode::new("metab", NodeKind::MetabolomicsChemometrics, "Metabolomics"),
            DagNode::new("export", NodeKind::ExportNefReport, "Export"),
        ],
        edges: vec![
            DagEdge::new("ingest", "raw_fid", "process", "raw_fid"),
            DagEdge::new("process", "spectrum_1d", "metab", "spectrum_1d"),
            DagEdge::new("process", "spectrum_2d", "pick", "spectrum_2d"),
            DagEdge::new("process", "spectrum_3d", "deconv", "spectrum_3d"),
            DagEdge::new("pick", "peaks", "deconv", "seed_peaks"),
            DagEdge::new("deconv", "peaks", "bb", "peaks"),
            DagEdge::new("deconv", "spin_systems", "bb", "spin_systems"),
            DagEdge::new("bb", "assignments", "sc", "assignments"),
            DagEdge::new("sc", "restraints", "fold", "restraints"),
            DagEdge::new("bb", "assignments", "fold", "assignments"),
            DagEdge::new("pick", "peaks", "relax", "peaks"),
            DagEdge::new("pick", "peaks", "exch", "peaks"),
            DagEdge::new("process", "spectrum_2d", "titr", "spectrum_2d"),
            DagEdge::new("bb", "assignments", "titr", "assignments"),
            DagEdge::new("sc", "assignments", "export", "assignments"),
            DagEdge::new("fold", "structure", "export", "structure"),
            DagEdge::new("relax", "rates", "export", "rates"),
            DagEdge::new("exch", "profiles", "export", "profiles"),
            DagEdge::new("metab", "matrix", "export", "matrix"),
            DagEdge::new("titr", "restraints", "export", "restraints"),
        ],
    };

    let manifest1 = compile_dag_manifest(&dag).expect("full suite compiles");
    assert_eq!(manifest1.nodes.len(), 12);

    // JSON round-trip preserves exact manifest hash
    let json = dag_to_json(&dag).expect("serialize DAG");
    let roundtrip = dag_from_json(&json).expect("deserialize DAG");
    let manifest2 = compile_dag_manifest(&roundtrip).expect("roundtrip compiles");
    assert_eq!(manifest1.manifest_sha256, manifest2.manifest_sha256);

    // Candidate edge validation catches both type mismatch and cycle incrementally
    assert!(matches!(
        validate_candidate_edge(&dag, &DagEdge::new("fold", "structure", "bb", "peaks")),
        Err(DagError::PortTypeMismatch { .. })
    ));
    assert!(matches!(
        validate_candidate_edge(&dag, &DagEdge::new("titr", "peaks", "deconv", "seed_peaks")),
        Err(DagError::InputPortAlreadyConnected { .. })
    ));
}

#[test]
fn test_p1_1_process_fid_fail_closed_peak_pick_passthrough_and_non_wired_stages_fail_closed() {
    // 1. ProcessFidNus MUST fail closed on reference-only STAR without `allow_simulation_demo: true`.
    let bb_manifest =
        compile_dag_manifest(&preset_backbone_assignment_dag()).expect("compile backbone");
    let gb1_star = include_str!("../../../data/fixtures/gb1_bmr7280_shifts.str").to_string();
    let mut cache = DagArtifactCache::new();
    let err = execute_dag_manifest_real(
        &bb_manifest,
        &mut cache,
        &DagRunOptions {
            star_text: Some(gb1_star.clone()),
            allow_simulation_demo: false,
            ..DagRunOptions::default()
        },
    )
    .expect_err(
        "ProcessFidNus must refuse to simulate peaks from STAR without allow_simulation_demo",
    );
    assert!(
        matches!(err, DagError::ExecutionFailed { ref node_id, ref reason }
            if node_id == "process" && reason.contains("allow_simulation_demo")),
        "unexpected error: {err:?}"
    );

    // 2. Real 2D spectrum (.ft2) + measured 3D peaks (NO HSQC in peaks_tsv) + sequence,
    //    with `allow_simulation_demo: false`: PeakPick picks 2D HSQC and passes picked peaks
    //    into BackboneAssign (never discarded!).
    let gb1_ref = read_star_str(&gb1_star, None).expect("parse GB1 STAR");
    let (all_peaks, _) = simulate(&gb1_ref, &SimConfig::default());
    let hsqc_2d = render_hsqc_2d(
        &all_peaks,
        &SpectrumRenderConfig {
            nx_h: 256,
            ny_n: 128,
            ..SpectrumRenderConfig::default()
        },
    );
    let triple_only: Vec<Peak> = all_peaks
        .iter()
        .filter(|p| p.exp != "HSQC")
        .cloned()
        .collect();
    assert!(!triple_only.is_empty());
    let triple_only_tsv = format_peaks_tsv(&triple_only);

    let data_root =
        std::env::temp_dir().join(format!("ferric_dag_p1_1_data_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_root);
    let ds_dir = data_root.join("datasets/backbone_3d");
    std::fs::create_dir_all(&ds_dir).expect("create dataset dir");
    write_nmrpipe_2d(ds_dir.join("hsqc.ft2"), &hsqc_2d).expect("write real 2D NMRPipe spectrum");

    let real_opts = DagRunOptions {
        peaks_tsv: Some(triple_only_tsv.clone()),
        sequence: Some(gb1_ref.one_letter()),
        data_root: Some(data_root.clone()),
        allow_simulation_demo: false,
        seed: 0,
        max_runs: Some(4),
        max_steps_per_system: Some(250),
        ..DagRunOptions::default()
    };
    let mut real_cache = DagArtifactCache::new();
    let real_out = execute_dag_manifest_real(&bb_manifest, &mut real_cache, &real_opts)
        .expect("real 2D spectrum + 3D peaks run must succeed without simulation demo");
    let real_metrics = real_out.metrics_json.as_ref().expect("metrics");
    assert_eq!(real_metrics["n_residues"].as_u64(), Some(56));
    assert_eq!(real_metrics["simulation_demo"].as_bool(), Some(false));
    let n_ss_picked = real_metrics["n_spin_systems"].as_u64().unwrap_or(0);
    assert!(
        n_ss_picked > 20,
        "BackboneAssign must consume HSQC peaks picked by PeakPick (got {n_ss_picked} spin systems)"
    );
    assert!(
        real_metrics["n_assigned"].as_u64().unwrap_or(0) > 10,
        "BackboneAssign must assign residues from picked 2D HSQC roots"
    );
    let real_nef = real_out.nef_text.as_deref().expect("NEF text");
    assert!(!real_nef.contains(SIMULATION_DEMO_PROVENANCE_TAG));

    // Raising PeakPick `snr_threshold` to `1e6` yields 0 picked HSQC peaks -> 0 spin systems,
    // proving PeakPick output is actually passed into BackboneAssign and NEVER discarded.
    let mut high_snr_dag = preset_backbone_assignment_dag();
    let pick_node = high_snr_dag
        .nodes
        .iter_mut()
        .find(|n| n.id == "pick")
        .expect("pick node");
    pick_node
        .params
        .insert("snr_threshold".to_string(), serde_json::json!(1_000_000.0));
    let high_snr_manifest = compile_dag_manifest(&high_snr_dag).expect("compile high-snr DAG");
    let high_snr_out = execute_dag_manifest_real(&high_snr_manifest, &mut real_cache, &real_opts)
        .expect("high-snr run executes");
    assert!(
        high_snr_out.trace.steps[2]
            .summary
            .contains("hsqc_picked=0"),
        "expected 0 picked HSQC peaks at SNR=1e6: {}",
        high_snr_out.trace.steps[2].summary
    );
    assert_eq!(
        high_snr_out
            .metrics_json
            .as_ref()
            .and_then(|m| m["n_spin_systems"].as_u64()),
        Some(0),
        "0 picked HSQC peaks must yield 0 spin systems in BackboneAssign"
    );
    let _ = std::fs::remove_dir_all(&data_root);

    // 3. ALL 6 non-wired stages MUST fail closed with `DagError::StageNotExecutable`!
    let dyn_manifest =
        compile_dag_manifest(&preset_dynamics_relax_cest_dag()).expect("compile dynamics");
    let mut dyn_cache = DagArtifactCache::new();
    let dyn_err = execute_dag_manifest_real(
        &dyn_manifest,
        &mut dyn_cache,
        &DagRunOptions {
            allow_simulation_demo: true,
            use_demo_fixture: true,
            ..DagRunOptions::default()
        },
    )
    .expect_err("dynamics preset must fail closed on non-wired stages");
    assert!(
        matches!(
            dyn_err,
            DagError::StageNotExecutable {
                stage: NodeKind::ExchangeCestCpmg | NodeKind::DynamicsRelax,
                ..
            }
        ),
        "expected StageNotExecutable, got {dyn_err:?}"
    );

    let non_wired_cases: Vec<(NodeKind, WorkflowDag)> = vec![
        (
            NodeKind::DeconvolveNd,
            WorkflowDag {
                name: "Deconv".to_string(),
                description: String::new(),
                nodes: vec![
                    DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
                    DagNode::new("pick", NodeKind::PeakPick, "Pick"),
                    DagNode::new("deconv", NodeKind::DeconvolveNd, "Deconv"),
                ],
                edges: vec![
                    DagEdge::new("ingest", "spectrum_2d", "pick", "spectrum_2d"),
                    DagEdge::new("pick", "peaks", "deconv", "seed_peaks"),
                ],
            },
        ),
        (
            NodeKind::DynamicsRelax,
            WorkflowDag {
                name: "Relax".to_string(),
                description: String::new(),
                nodes: vec![
                    DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
                    DagNode::new("pick", NodeKind::PeakPick, "Pick"),
                    DagNode::new("relax", NodeKind::DynamicsRelax, "Relax"),
                ],
                edges: vec![
                    DagEdge::new("ingest", "spectrum_2d", "pick", "spectrum_2d"),
                    DagEdge::new("pick", "peaks", "relax", "peaks"),
                ],
            },
        ),
        (
            NodeKind::ExchangeCestCpmg,
            WorkflowDag {
                name: "Exchange".to_string(),
                description: String::new(),
                nodes: vec![
                    DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
                    DagNode::new("pick", NodeKind::PeakPick, "Pick"),
                    DagNode::new("exch", NodeKind::ExchangeCestCpmg, "Exch"),
                ],
                edges: vec![
                    DagEdge::new("ingest", "spectrum_2d", "pick", "spectrum_2d"),
                    DagEdge::new("pick", "peaks", "exch", "peaks"),
                ],
            },
        ),
        (
            NodeKind::SidechainAssign,
            WorkflowDag {
                name: "Sidechain".to_string(),
                description: String::new(),
                nodes: vec![
                    DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
                    DagNode::new("pick", NodeKind::PeakPick, "Pick"),
                    DagNode::new("bb", NodeKind::BackboneAssign, "BB"),
                    DagNode::new("sc", NodeKind::SidechainAssign, "SC"),
                ],
                edges: vec![
                    DagEdge::new("ingest", "spectrum_2d", "pick", "spectrum_2d"),
                    DagEdge::new("pick", "peaks", "bb", "peaks"),
                    DagEdge::new("bb", "assignments", "sc", "assignments"),
                ],
            },
        ),
        (
            NodeKind::Titration2d,
            WorkflowDag {
                name: "Titration".to_string(),
                description: String::new(),
                nodes: vec![
                    DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
                    DagNode::new("titr", NodeKind::Titration2d, "Titr"),
                ],
                edges: vec![DagEdge::new("ingest", "spectrum_2d", "titr", "spectrum_2d")],
            },
        ),
        (
            NodeKind::StructureFoldDock,
            WorkflowDag {
                name: "Fold".to_string(),
                description: String::new(),
                nodes: vec![
                    DagNode::new("ingest", NodeKind::IngestBrukerVarian, "Ingest"),
                    DagNode::new("titr", NodeKind::Titration2d, "Titr"),
                    DagNode::new("fold", NodeKind::StructureFoldDock, "Fold"),
                ],
                edges: vec![
                    DagEdge::new("ingest", "spectrum_2d", "titr", "spectrum_2d"),
                    DagEdge::new("titr", "restraints", "fold", "restraints"),
                ],
            },
        ),
    ];

    assert!(!NodeKind::StructureFoldDock.is_real_executable());
    for (expected_stage, dag) in non_wired_cases {
        assert!(!expected_stage.is_real_executable());
        let manifest = compile_dag_manifest(&dag).expect("compile non-wired stage DAG");
        let mut c = DagArtifactCache::new();
        let err = execute_dag_manifest_real(
            &manifest,
            &mut c,
            &DagRunOptions {
                allow_simulation_demo: true,
                use_demo_fixture: true,
                ..DagRunOptions::default()
            },
        )
        .expect_err("non-wired stage must fail closed in real execution");
        assert!(
            matches!(err, DagError::StageNotExecutable { .. }),
            "stage {expected_stage:?} must fail with StageNotExecutable, got {err:?}"
        );
        // Dry-run still works cleanly for graph topology validation
        let dry = execute_dag_manifest_dry_run(&manifest, &mut c)
            .expect("dry-run must still work on non-wired stages");
        assert_eq!(dry.executed_count, manifest.nodes.len());
    }
}

#[test]
fn test_p1_4_cache_keying_dry_run_isolation_input_change_and_downstream_export_change() {
    let bb_dag = preset_backbone_assignment_dag();
    let bb_manifest = compile_dag_manifest(&bb_dag).expect("compile backbone");
    let gb1_star = include_str!("../../../data/fixtures/gb1_bmr7280_shifts.str").to_string();

    let mut shared_cache = DagArtifactCache::new();

    // Probe (b): Dry run followed by real run on the SAME cache -> real run executes all 5
    // stages and emits real NEF & HTML (dry run NEVER poisons real cache).
    let dry_trace = execute_dag_manifest_dry_run(&bb_manifest, &mut shared_cache)
        .expect("dry run populates dry_run=true entries");
    assert_eq!(dry_trace.executed_count, 5);
    assert_eq!(dry_trace.cache_hit_count, 0);

    let ubq_demo_opts = DagRunOptions {
        use_demo_fixture: true,
        allow_simulation_demo: true,
        seed: 0,
        max_runs: Some(4),
        max_steps_per_system: Some(250),
        ..DagRunOptions::default()
    };
    let ubq_out = execute_dag_manifest_real(&bb_manifest, &mut shared_cache, &ubq_demo_opts)
        .expect("real run after dry run must execute all 5 stages");
    assert_eq!(ubq_out.trace.executed_count, 5);
    assert_eq!(ubq_out.trace.cache_hit_count, 0);
    assert_eq!(
        ubq_out
            .metrics_json
            .as_ref()
            .and_then(|m| m["n_residues"].as_u64()),
        Some(76)
    );
    let ubq_nef = ubq_out.nef_text.as_deref().expect("ubiquitin NEF");
    assert!(ubq_nef.contains("data_nef_"));
    assert!(ubq_out.report_html.is_some());

    // Probe (a): Warm ubiquitin demo cache followed by GB1 on SAME cache -> 100% cache miss,
    // runs GB1, emits GB1 NEF (56 residues, NOT ubiquitin's 76!) and GB1 trace.
    let gb1_demo_opts = DagRunOptions {
        star_text: Some(gb1_star.clone()),
        allow_simulation_demo: true,
        seed: 0,
        max_runs: Some(4),
        max_steps_per_system: Some(250),
        ..DagRunOptions::default()
    };
    let gb1_out = execute_dag_manifest_real(&bb_manifest, &mut shared_cache, &gb1_demo_opts)
        .expect("GB1 run on ubiquitin-warmed cache must miss and run GB1");
    assert_eq!(gb1_out.trace.executed_count, 5);
    assert_eq!(gb1_out.trace.cache_hit_count, 0);
    assert!(gb1_out.trace.steps[0].summary.contains("residues=56"));
    assert_eq!(
        gb1_out
            .metrics_json
            .as_ref()
            .and_then(|m| m["n_residues"].as_u64()),
        Some(56)
    );
    let gb1_nef = gb1_out.nef_text.as_deref().expect("GB1 NEF");
    assert_ne!(gb1_nef, ubq_nef);

    // Probe (a) fail-closed check: GB1 STAR without `--demo` / `--peaks` on SAME cache MUST miss
    // and fail closed (never return ubiquitin or GB1-demo cached hits!).
    let gb1_no_demo_err = execute_dag_manifest_real(
        &bb_manifest,
        &mut shared_cache,
        &DagRunOptions {
            star_text: Some(gb1_star),
            allow_simulation_demo: false,
            seed: 0,
            max_runs: Some(4),
            max_steps_per_system: Some(250),
            ..DagRunOptions::default()
        },
    )
    .expect_err("GB1 without peaks or allow_simulation_demo must miss cache and fail closed");
    assert!(matches!(gb1_no_demo_err, DagError::ExecutionFailed { .. }));

    // Probe (c): Warm cache + downstream-only export parameter change -> restores upstream shifts
    // & peaks from cache (4 cache hits) and re-runs ONLY export (1 executed) cleanly!
    let mut modified_dag = bb_dag.clone();
    let export_node = modified_dag
        .nodes
        .iter_mut()
        .find(|n| n.id == "export")
        .expect("export node");
    export_node.params.insert(
        "output_path".to_string(),
        serde_json::json!("reports/gb1_renamed_v2.nef"),
    );
    let modified_manifest = compile_dag_manifest(&modified_dag).expect("compile modified export");

    let tmp_out = std::env::temp_dir().join(format!(
        "ferric_dag_p1_4_export_change_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp_out);
    let mut gb1_export_opts = gb1_demo_opts.clone();
    gb1_export_opts.output_dir = Some(tmp_out.clone());

    let reexport_out =
        execute_dag_manifest_real(&modified_manifest, &mut shared_cache, &gb1_export_opts)
            .expect("downstream-only export change must restore upstream state and succeed");
    assert_eq!(reexport_out.trace.cache_hit_count, 4);
    assert_eq!(reexport_out.trace.executed_count, 1);
    assert_eq!(reexport_out.trace.steps[4].node_id, "export");
    assert_eq!(reexport_out.trace.steps[4].status, StepStatus::Executed);
    assert_eq!(reexport_out.nef_text.as_deref(), Some(gb1_nef));
    assert!(tmp_out.join("reports/gb1_renamed_v2.nef").is_file());
    assert!(tmp_out
        .join("reports/gb1_renamed_v2.nef.report.html")
        .is_file());

    // Repeat identical run on `modified_manifest` into a fresh directory -> 5 cache hits AND
    // still writes the restored NEF & HTML files to the new output_dir!
    let tmp_out2 = std::env::temp_dir().join(format!(
        "ferric_dag_p1_4_export_cached_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp_out2);
    gb1_export_opts.output_dir = Some(tmp_out2.clone());
    let cached_out =
        execute_dag_manifest_real(&modified_manifest, &mut shared_cache, &gb1_export_opts)
            .expect("full cache hit must restore and emit files");
    assert_eq!(cached_out.trace.cache_hit_count, 5);
    assert_eq!(cached_out.trace.executed_count, 0);
    assert_eq!(cached_out.nef_text, reexport_out.nef_text);
    assert_eq!(cached_out.report_html, reexport_out.report_html);
    assert_eq!(cached_out.artifacts, reexport_out.artifacts);
    assert_eq!(
        std::fs::read_to_string(tmp_out2.join("reports/gb1_renamed_v2.nef"))
            .expect("cached NEF written"),
        gb1_nef
    );

    let _ = std::fs::remove_dir_all(&tmp_out);
    let _ = std::fs::remove_dir_all(&tmp_out2);
}

#[test]
fn test_p2_1_p2_2_p2_3_traversal_numeric_canonicalization_and_symlink_rejection() {
    // P2-1: Path traversal & absolute paths rejected in ALL string params (even non-path keys
    // like `notes`, `tag`, nested arrays, and nested objects).
    for (key, val) in [
        ("notes", serde_json::json!("../secret/file")),
        ("tag", serde_json::json!("/etc/passwd")),
        ("custom", serde_json::json!("a/../../b")),
        ("win_drive", serde_json::json!("C:/Users/secret")),
        ("backslash", serde_json::json!("foo\\bar")),
        ("list_param", serde_json::json!(["safe", "../escape"])),
        (
            "nested_obj",
            serde_json::json!({"meta": {"comment": "/abs/path"}}),
        ),
    ] {
        let mut dag = preset_metabolomics_1d_dag();
        dag.nodes[0].params.insert(key.to_string(), val);
        assert!(
            matches!(validate_dag(&dag), Err(DagError::InvalidParam { .. })),
            "P2-1: expected traversal/absolute string in key '{key}' to be rejected"
        );
    }

    // P2-2: Numeric JSON canonicalization: `5` and `5.0`, `2` and `2.0`, `0` and `-0.0` hash identically.
    let mut dag_int = preset_backbone_assignment_dag();
    let mut dag_flt = preset_backbone_assignment_dag();
    dag_int.nodes[1]
        .params
        .insert("zero_fill_factor".to_string(), serde_json::json!(2));
    dag_flt.nodes[1]
        .params
        .insert("zero_fill_factor".to_string(), serde_json::json!(2.0));
    dag_int.nodes[2]
        .params
        .insert("snr_threshold".to_string(), serde_json::json!(5));
    dag_flt.nodes[2]
        .params
        .insert("snr_threshold".to_string(), serde_json::json!(5.0));
    dag_int.nodes[3]
        .params
        .insert("seeds".to_string(), serde_json::json!(5));
    dag_flt.nodes[3]
        .params
        .insert("seeds".to_string(), serde_json::json!(5.0));
    dag_int.nodes[0]
        .params
        .insert("offset".to_string(), serde_json::json!(0));
    dag_flt.nodes[0]
        .params
        .insert("offset".to_string(), serde_json::json!(-0.0));

    let m_int = compile_dag_manifest(&dag_int).expect("compile int DAG");
    let m_flt = compile_dag_manifest(&dag_flt).expect("compile float DAG");
    assert_eq!(
        m_int.manifest_sha256, m_flt.manifest_sha256,
        "P2-2: integer 5 and float 5.0 must produce identical manifest_sha256"
    );
    for (ni, nf) in m_int.nodes.iter().zip(m_flt.nodes.iter()) {
        assert_eq!(
            ni.node_sha256, nf.node_sha256,
            "P2-2: node '{}' hash differed between int and float",
            ni.id
        );
    }

    // P2-3: Symlink at final output component AND symlink at parent directory component
    // MUST be rejected BEFORE `create_dir` / `write`, leaving outside targets untouched.
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;

        let base = std::env::temp_dir().join(format!("ferric_dag_p2_3_sym_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let outside_dir = base.join("outside");
        let out_dir = base.join("out_dir");
        std::fs::create_dir_all(&outside_dir).expect("create outside");
        std::fs::create_dir_all(out_dir.join("reports")).expect("create out_dir/reports");

        // Case 1: Planted symlink at the FINAL file component
        let victim = outside_dir.join("victim.txt");
        std::fs::write(&victim, "DO_NOT_OVERWRITE").expect("write victim");
        symlink(&victim, out_dir.join("reports/backbone_assignments.nef"))
            .expect("plant final symlink");

        let manifest =
            compile_dag_manifest(&preset_backbone_assignment_dag()).expect("compile backbone");
        let mut cache = DagArtifactCache::new();
        let sym_err = execute_dag_manifest_real(
            &manifest,
            &mut cache,
            &DagRunOptions {
                use_demo_fixture: true,
                allow_simulation_demo: true,
                output_dir: Some(out_dir.clone()),
                seed: 0,
                max_runs: Some(4),
                max_steps_per_system: Some(200),
                ..DagRunOptions::default()
            },
        )
        .expect_err("P2-3: planted final-component symlink must be rejected");
        assert!(
            matches!(sym_err, DagError::ExecutionFailed { ref reason, .. } if reason.contains("symlink")),
            "expected symlink rejection error, got {sym_err:?}"
        );
        assert_eq!(
            std::fs::read_to_string(&victim).expect("read victim"),
            "DO_NOT_OVERWRITE",
            "victim file outside output_dir must NEVER be overwritten"
        );

        // Case 2: Planted symlink at an INTERMEDIATE directory component before nested subdir
        let out_dir2 = base.join("out_dir2");
        std::fs::create_dir_all(&out_dir2).expect("create out_dir2");
        symlink(&outside_dir, out_dir2.join("reports")).expect("plant dir symlink");
        let mut nested_dag = preset_backbone_assignment_dag();
        nested_dag.nodes[4].params.insert(
            "output_path".to_string(),
            serde_json::json!("reports/nested_escape/out.nef"),
        );
        let nested_manifest = compile_dag_manifest(&nested_dag).expect("compile nested DAG");
        let dir_sym_err = execute_dag_manifest_real(
            &nested_manifest,
            &mut cache,
            &DagRunOptions {
                use_demo_fixture: true,
                allow_simulation_demo: true,
                output_dir: Some(out_dir2.clone()),
                seed: 0,
                max_runs: Some(4),
                max_steps_per_system: Some(200),
                ..DagRunOptions::default()
            },
        )
        .expect_err("P2-3: planted directory symlink must be rejected before creating subdir");
        assert!(
            matches!(dir_sym_err, DagError::ExecutionFailed { ref reason, .. } if reason.contains("symlink")),
            "expected directory symlink rejection error, got {dir_sym_err:?}"
        );
        assert!(
            !outside_dir.join("nested_escape").exists(),
            "P2-3: containment check must run BEFORE creating subdirectories inside symlink target"
        );

        let _ = std::fs::remove_dir_all(&base);
    }
}
