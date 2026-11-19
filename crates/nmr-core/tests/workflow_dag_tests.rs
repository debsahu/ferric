//! Integration tests for `nmr_core::workflow_dag` (`T22.11`):
//! static typed-port validation, Kahn's deterministic topological sort, cycle detection,
//! hostile bounds & path-traversal rejection, canonical SHA-256 content-addressed manifests,
//! and incremental upstream cache reuse.

use std::collections::BTreeMap;

use nmr_core::workflow_dag::{
    compile_dag_manifest, dag_from_json, dag_to_json, execute_dag_manifest,
    execute_dag_manifest_dry_run, manifest_to_json, preset_backbone_assignment_dag,
    preset_dynamics_relax_cest_dag, preset_metabolomics_1d_dag, validate_candidate_edge,
    validate_dag, DagArtifactCache, DagEdge, DagError, DagNode, NodeKind, PortDirection, PortType,
    StepStatus, WorkflowDag, DAG_MANIFEST_SCHEMA, MAX_DAG_EDGES, MAX_DAG_JSON_BYTES, MAX_DAG_NODES,
};

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
fn test_adversarial_multibyte_utf8_manifest_sha_no_panic() {
    let mut manifest =
        compile_dag_manifest(&preset_metabolomics_1d_dag()).expect("compile metabolomics preset");
    // Multi-byte UTF-8 character straddling byte offset 12 (11 ASCII bytes + 2-byte U+00E9)
    let crafted_sha = "aaaaaaaaaaa\u{00E9}bbbb".to_string();
    assert!(!crafted_sha.is_char_boundary(12));
    manifest.manifest_sha256 = crafted_sha.clone();
    manifest.nodes[0].node_sha256 = crafted_sha;

    let mut cache = DagArtifactCache::new();
    let trace =
        execute_dag_manifest_dry_run(&manifest, &mut cache).expect("dry-run must not panic");
    assert_eq!(trace.executed_count, manifest.nodes.len());
    assert!(trace.steps[0].summary.contains("aaaaaaaaaaa\u{00E9}"));
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
