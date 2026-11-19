//! Clean-room `ferric dag` CLI subcommand (`preset | validate | compile | run`) for
//! typed NMR workflow `DAG` validation, content-addressed `ferric-dag-v1` manifest
//! compilation, fail-closed integrity verification, and deterministic stage execution (`T22.11`).
//!
//! Provenance: Original clean-room CLI driver backed by [`nmr_core::workflow_dag`].

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use nmr_core::workflow_dag::{
    compile_dag_manifest, dag_from_json, dag_to_json, execute_dag_manifest_dry_run,
    execute_dag_manifest_real, manifest_from_json, manifest_to_json, preset_by_name, validate_dag,
    DagArtifactCache, DagManifest, DagRunOptions, StepStatus, MAX_DAG_JSON_BYTES,
};

const MAX_INPUT_DATA_BYTES: usize = 64 * 1024 * 1024; // 64 MiB
const MAX_CACHE_JSON_BYTES: usize = 4 * MAX_DAG_JSON_BYTES;

fn dag_usage() -> &'static str {
    "Usage:\n  \
     ferric dag preset   <backbone|dynamics|metabolomics> [-o|--out <dag.json>]\n  \
     ferric dag validate --in <dag.json> [--json]\n  \
     ferric dag compile  --in <dag.json> [-o|--out <manifest.json>]\n  \
     ferric dag run      --in <manifest_or_dag.json> [--out-dir <dir>] [--data-dir <dir>] [--star <file.str>] [--peaks <peaks.tsv>] [--nef <file.nef>] [--seq <SEQ|fasta>] [--cache <cache.json>] [--dry-run] [--demo] [--seed <u64>] [--runs <N>] [--steps <N>] [--json-out <trace.json>]\n"
}

fn read_bounded_utf8(path: &Path, max_bytes: usize, label: &str) -> Result<String, String> {
    let meta = fs::metadata(path)
        .map_err(|e| format!("failed to stat {label} '{}': {e}", path.display()))?;
    if meta.len() > max_bytes as u64 {
        return Err(format!(
            "{label} '{}' is {} bytes, exceeding maximum allowed {max_bytes} bytes",
            path.display(),
            meta.len()
        ));
    }
    let file = File::open(path)
        .map_err(|e| format!("failed to open {label} '{}': {e}", path.display()))?;
    let mut buf = Vec::new();
    file.take((max_bytes as u64).saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|e| format!("failed to read {label} '{}': {e}", path.display()))?;
    if buf.len() > max_bytes {
        return Err(format!(
            "{label} '{}' exceeds maximum allowed {max_bytes} bytes",
            path.display()
        ));
    }
    String::from_utf8(buf)
        .map_err(|e| format!("{label} '{}' is not valid UTF-8: {e}", path.display()))
}

fn write_text_file(path: &Path, content: &str, label: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create parent directory for {label} '{}': {e}",
                    path.display()
                )
            })?;
        }
    }
    fs::write(path, content)
        .map_err(|e| format!("failed to write {label} '{}': {e}", path.display()))
}

fn next_arg<'a>(args: &'a [String], idx: &mut usize, flag: &str) -> Result<&'a str, String> {
    *idx += 1;
    args.get(*idx)
        .map(String::as_str)
        .ok_or_else(|| format!("missing value after {flag}"))
}

/// Entry point for `ferric dag <subcommand> [options]`.
pub fn run_dag_cli(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print!("{}", dag_usage());
        return Ok(());
    }
    match args[0].as_str() {
        "preset" => cmd_dag_preset(&args[1..]),
        "validate" => cmd_dag_validate(&args[1..]),
        "compile" => cmd_dag_compile(&args[1..]),
        "run" => cmd_dag_run(&args[1..]),
        other => Err(format!(
            "unknown `ferric dag` subcommand {other:?}\n\n{}",
            dag_usage()
        )),
    }
}

fn cmd_dag_preset(args: &[String]) -> Result<(), String> {
    let mut name: Option<String> = None;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{}", dag_usage());
                return Ok(());
            }
            "-o" | "--out" => {
                out_path = Some(PathBuf::from(next_arg(args, &mut i, "--out")?));
            }
            "--name" => {
                name = Some(next_arg(args, &mut i, "--name")?.to_string());
            }
            tok if !tok.starts_with('-') && name.is_none() => {
                name = Some(tok.to_string());
            }
            other => {
                return Err(format!(
                    "unknown argument {other:?} for `ferric dag preset`"
                ));
            }
        }
        i += 1;
    }

    let preset_name = name.ok_or_else(|| {
        "missing preset name; expected one of: backbone, dynamics, metabolomics".to_string()
    })?;
    let dag = preset_by_name(&preset_name).ok_or_else(|| {
        format!(
            "unknown DAG preset {preset_name:?}; expected one of: backbone, dynamics, metabolomics"
        )
    })?;
    let json = dag_to_json(&dag).map_err(|e| e.to_string())?;
    if let Some(path) = out_path {
        write_text_file(&path, &json, "preset DAG JSON")?;
        println!(
            "Wrote preset '{}' ({} nodes, {} edges) -> {}",
            dag.name,
            dag.nodes.len(),
            dag.edges.len(),
            path.display()
        );
    } else {
        println!("{json}");
    }
    Ok(())
}

fn cmd_dag_validate(args: &[String]) -> Result<(), String> {
    let mut in_path: Option<PathBuf> = None;
    let mut as_json = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{}", dag_usage());
                return Ok(());
            }
            "--in" | "-i" => {
                in_path = Some(PathBuf::from(next_arg(args, &mut i, "--in")?));
            }
            "--json" => {
                as_json = true;
            }
            tok if !tok.starts_with('-') && in_path.is_none() => {
                in_path = Some(PathBuf::from(tok));
            }
            other => {
                return Err(format!(
                    "unknown argument {other:?} for `ferric dag validate`"
                ));
            }
        }
        i += 1;
    }

    let path = in_path.ok_or_else(|| "missing required --in <dag.json>".to_string())?;
    let raw = read_bounded_utf8(&path, MAX_DAG_JSON_BYTES, "DAG JSON")?;
    let dag = dag_from_json(&raw).map_err(|e| e.to_string())?;
    let validated = validate_dag(&dag).map_err(|e| e.to_string())?;

    if as_json {
        let payload = serde_json::json!({
            "valid": true,
            "workflow_name": dag.name,
            "node_count": dag.nodes.len(),
            "edge_count": dag.edges.len(),
            "topological_order": validated.topological_order,
        });
        let encoded = serde_json::to_string_pretty(&payload)
            .map_err(|e| format!("failed to serialize validation JSON: {e}"))?;
        println!("{encoded}");
    } else {
        println!(
            "Valid DAG '{}': {} nodes, {} edges | order: {}",
            dag.name,
            dag.nodes.len(),
            dag.edges.len(),
            validated.topological_order.join(" -> ")
        );
    }
    Ok(())
}

fn cmd_dag_compile(args: &[String]) -> Result<(), String> {
    let mut in_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{}", dag_usage());
                return Ok(());
            }
            "--in" | "-i" => {
                in_path = Some(PathBuf::from(next_arg(args, &mut i, "--in")?));
            }
            "-o" | "--out" => {
                out_path = Some(PathBuf::from(next_arg(args, &mut i, "--out")?));
            }
            tok if !tok.starts_with('-') && in_path.is_none() => {
                in_path = Some(PathBuf::from(tok));
            }
            other => {
                return Err(format!(
                    "unknown argument {other:?} for `ferric dag compile`"
                ));
            }
        }
        i += 1;
    }

    let path = in_path.ok_or_else(|| "missing required --in <dag.json>".to_string())?;
    let raw = read_bounded_utf8(&path, MAX_DAG_JSON_BYTES, "DAG JSON")?;
    let dag = dag_from_json(&raw).map_err(|e| e.to_string())?;
    let manifest = compile_dag_manifest(&dag).map_err(|e| e.to_string())?;
    let json = manifest_to_json(&manifest).map_err(|e| e.to_string())?;

    if let Some(out) = out_path {
        write_text_file(&out, &json, "compiled manifest JSON")?;
        println!(
            "Compiled '{}' ({}) -> {} | manifest_sha256={}",
            manifest.workflow_name,
            manifest.schema,
            out.display(),
            manifest.manifest_sha256
        );
    } else {
        println!("{json}");
    }
    Ok(())
}

fn load_and_verify_manifest_or_dag(raw: &str) -> Result<DagManifest, String> {
    if raw.len() > MAX_DAG_JSON_BYTES {
        return Err(format!(
            "JSON payload {} bytes exceeds maximum {MAX_DAG_JSON_BYTES} bytes",
            raw.len()
        ));
    }
    let val: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| format!("invalid JSON input: {e}"))?;
    let is_manifest = val.as_object().is_some_and(|obj| {
        obj.contains_key("schema")
            || obj.contains_key("manifest_sha256")
            || obj.contains_key("topological_order")
    });
    if is_manifest {
        // Strict fail-closed manifest load: never fall back to raw DAG parsing on tampered manifests.
        manifest_from_json(raw).map_err(|e| e.to_string())
    } else {
        let dag = dag_from_json(raw).map_err(|e| e.to_string())?;
        compile_dag_manifest(&dag).map_err(|e| e.to_string())
    }
}

fn cmd_dag_run(args: &[String]) -> Result<(), String> {
    let mut in_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut data_dir: Option<PathBuf> = None;
    let mut star_path: Option<PathBuf> = None;
    let mut peaks_path: Option<PathBuf> = None;
    let mut nef_path: Option<PathBuf> = None;
    let mut seq_arg: Option<String> = None;
    let mut cache_path: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut dry_run = false;
    let mut demo = false;
    let mut seed: u64 = 0;
    let mut max_runs: Option<usize> = None;
    let mut max_steps: Option<usize> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{}", dag_usage());
                return Ok(());
            }
            "--in" | "-i" => {
                in_path = Some(PathBuf::from(next_arg(args, &mut i, "--in")?));
            }
            "--out-dir" => {
                out_dir = Some(PathBuf::from(next_arg(args, &mut i, "--out-dir")?));
            }
            "--data-dir" => {
                data_dir = Some(PathBuf::from(next_arg(args, &mut i, "--data-dir")?));
            }
            "--star" => {
                star_path = Some(PathBuf::from(next_arg(args, &mut i, "--star")?));
            }
            "--peaks" => {
                peaks_path = Some(PathBuf::from(next_arg(args, &mut i, "--peaks")?));
            }
            "--nef" => {
                nef_path = Some(PathBuf::from(next_arg(args, &mut i, "--nef")?));
            }
            "--seq" => {
                seq_arg = Some(next_arg(args, &mut i, "--seq")?.to_string());
            }
            "--cache" => {
                cache_path = Some(PathBuf::from(next_arg(args, &mut i, "--cache")?));
            }
            "--json-out" => {
                json_out = Some(PathBuf::from(next_arg(args, &mut i, "--json-out")?));
            }
            "--dry-run" => {
                dry_run = true;
            }
            "--demo" => {
                demo = true;
            }
            "--seed" => {
                let raw = next_arg(args, &mut i, "--seed")?;
                seed = raw
                    .parse::<u64>()
                    .map_err(|e| format!("invalid --seed {raw:?}: {e}"))?;
            }
            "--runs" => {
                let raw = next_arg(args, &mut i, "--runs")?;
                let parsed = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --runs {raw:?}: {e}"))?;
                if !(1..=64).contains(&parsed) {
                    return Err(format!("--runs {parsed} out of bounds (1..=64)"));
                }
                max_runs = Some(parsed);
            }
            "--steps" => {
                let raw = next_arg(args, &mut i, "--steps")?;
                let parsed = raw
                    .parse::<usize>()
                    .map_err(|e| format!("invalid --steps {raw:?}: {e}"))?;
                if !(100..=4000).contains(&parsed) {
                    return Err(format!("--steps {parsed} out of bounds (100..=4000)"));
                }
                max_steps = Some(parsed);
            }
            tok if !tok.starts_with('-') && in_path.is_none() => {
                in_path = Some(PathBuf::from(tok));
            }
            other => {
                return Err(format!("unknown argument {other:?} for `ferric dag run`"));
            }
        }
        i += 1;
    }

    let path = in_path.ok_or_else(|| "missing required --in <manifest_or_dag.json>".to_string())?;
    let raw = read_bounded_utf8(&path, MAX_DAG_JSON_BYTES, "DAG/manifest JSON")?;
    let manifest = load_and_verify_manifest_or_dag(&raw)?;

    let mut cache = if let Some(ref cp) = cache_path {
        if cp.exists() {
            let cache_raw = read_bounded_utf8(cp, MAX_CACHE_JSON_BYTES, "DAG cache JSON")?;
            serde_json::from_str::<DagArtifactCache>(&cache_raw)
                .map_err(|e| format!("invalid DAG cache JSON '{}': {e}", cp.display()))?
        } else {
            DagArtifactCache::new()
        }
    } else {
        DagArtifactCache::new()
    };

    if dry_run {
        let trace =
            execute_dag_manifest_dry_run(&manifest, &mut cache).map_err(|e| e.to_string())?;
        if let Some(ref cp) = cache_path {
            let encoded = serde_json::to_string_pretty(&cache)
                .map_err(|e| format!("failed to serialize DAG cache: {e}"))?;
            write_text_file(cp, &encoded, "DAG cache JSON")?;
        }
        if let Some(ref jp) = json_out {
            let encoded = serde_json::to_string_pretty(&trace)
                .map_err(|e| format!("failed to serialize trace JSON: {e}"))?;
            write_text_file(jp, &encoded, "trace JSON")?;
        }
        println!(
            "DAG dry-run '{}' ({}) | executed={} cached={} | manifest_sha256={}",
            manifest.workflow_name,
            manifest.schema,
            trace.executed_count,
            trace.cache_hit_count,
            trace.manifest_sha256
        );
        for step in &trace.steps {
            let status_str = match step.status {
                StepStatus::Executed => "EXECUTED",
                StepStatus::CacheHit => "CACHE_HIT",
            };
            println!(
                "  [{}] {:>9} {} ({}) :: {}",
                step.step_index,
                status_str,
                step.node_id,
                step.kind.canonical_id(),
                step.summary
            );
        }
        return Ok(());
    }

    let star_text = match star_path {
        Some(ref p) => Some(read_bounded_utf8(p, MAX_INPUT_DATA_BYTES, "NMR-STAR file")?),
        None => None,
    };
    let peaks_tsv = match peaks_path {
        Some(ref p) => Some(read_bounded_utf8(
            p,
            MAX_INPUT_DATA_BYTES,
            "peaks TSV file",
        )?),
        None => None,
    };
    let nef_text = match nef_path {
        Some(ref p) => Some(read_bounded_utf8(p, MAX_INPUT_DATA_BYTES, "NEF file")?),
        None => None,
    };
    let sequence = match seq_arg {
        Some(ref s) => {
            let p = Path::new(s);
            if p.is_file() {
                Some(read_bounded_utf8(p, MAX_INPUT_DATA_BYTES, "sequence file")?)
            } else {
                Some(s.clone())
            }
        }
        None => None,
    };

    let opts = DagRunOptions {
        star_text,
        peaks_tsv,
        nef_text,
        sequence,
        data_root: data_dir,
        output_dir: out_dir.clone(),
        use_demo_fixture: demo,
        seed,
        max_runs,
        max_steps_per_system: max_steps,
    };

    let output =
        execute_dag_manifest_real(&manifest, &mut cache, &opts).map_err(|e| e.to_string())?;

    if let Some(ref cp) = cache_path {
        let encoded = serde_json::to_string_pretty(&cache)
            .map_err(|e| format!("failed to serialize DAG cache: {e}"))?;
        write_text_file(cp, &encoded, "DAG cache JSON")?;
    }

    let trace_payload = serde_json::json!({
        "schema": output.trace.schema,
        "manifest_sha256": output.trace.manifest_sha256,
        "executed_count": output.trace.executed_count,
        "cache_hit_count": output.trace.cache_hit_count,
        "steps": output.trace.steps,
        "artifacts": output.artifacts,
        "written_files": output.written_files,
        "metrics": output.metrics_json,
    });
    let trace_json = serde_json::to_string_pretty(&trace_payload)
        .map_err(|e| format!("failed to serialize DAG run trace JSON: {e}"))?;

    if let Some(ref od) = out_dir {
        write_text_file(&od.join("dag_trace.json"), &trace_json, "dag_trace.json")?;
    }
    if let Some(ref jp) = json_out {
        write_text_file(jp, &trace_json, "trace JSON")?;
    }

    println!(
        "DAG run '{}' ({}) | executed={} cached={} | manifest_sha256={}",
        manifest.workflow_name,
        manifest.schema,
        output.trace.executed_count,
        output.trace.cache_hit_count,
        output.trace.manifest_sha256
    );
    for step in &output.trace.steps {
        let status_str = match step.status {
            StepStatus::Executed => "EXECUTED",
            StepStatus::CacheHit => "CACHE_HIT",
        };
        println!(
            "  [{}] {:>9} {} ({}) :: {}",
            step.step_index,
            status_str,
            step.node_id,
            step.kind.canonical_id(),
            step.summary
        );
    }
    if !output.written_files.is_empty() {
        println!("  Written artifacts: {}", output.written_files.join(", "));
    }
    Ok(())
}
