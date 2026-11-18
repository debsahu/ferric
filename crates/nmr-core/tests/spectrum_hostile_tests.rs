//! Deterministic hostile-file regression corpus for the processed-spectrum loaders in
//! `nmr_core::spectrum` (`v0150:P0-2`, `v0150:P1-7`, `20.W1:P2-6`).
//!
//! Every case is a few hundred bytes (or a ~1 KB file) whose header declares absurd dimensions.
//! Before the fix these aborted on multi-exabyte allocations, wrapped `nx * ny` to zero and looped
//! ~2^64 times, panicked on out-of-range tile writes, or returned `Ok` with `data.len() == 0` and
//! `2^32`-point axes. After the fix every loader must return a prompt `Err` before allocating.
//!
//! Each call runs on a watchdog thread so a regression shows up as a test failure (hang or panic
//! reported by the watchdog), not as a hung test process.

use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use nmr_core::spectrum::{
    read_bruker_pdata_2d, read_bruker_pdata_2d_from_bytes, read_bruker_pdata_3d,
    read_bruker_pdata_3d_from_bytes, read_nmrpipe_2d_from_bytes, read_nmrpipe_3d_from_bytes,
    read_ucsf_2d, read_ucsf_2d_from_bytes, read_ucsf_3d, read_ucsf_3d_from_bytes, AxisPpm,
    Spectrum2D, Spectrum3D,
};

/// A hostile header must be rejected well inside this budget (generous for loaded debug builds,
/// but orders of magnitude below any allocation or `2^64`-iteration loop).
const REJECT_BUDGET: Duration = Duration::from_millis(50);

/// Run `f` on a watchdog thread; fail if it hangs (> 5 s) or panics, and return its duration.
fn bounded<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> (T, Duration) {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let t0 = Instant::now();
        let r = f();
        let _ = tx.send((r, t0.elapsed()));
    });
    rx.recv_timeout(Duration::from_secs(5)).unwrap_or_else(|e| {
        panic!("{what}: loader hung or panicked instead of returning Err ({e})")
    })
}

fn expect_prompt_err<T: Send + std::fmt::Debug + 'static>(
    what: &str,
    f: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> String {
    let (res, dt) = bounded(what, f);
    let msg = res.expect_err(&format!("{what}: hostile header must be rejected"));
    assert!(
        dt < REJECT_BUDGET,
        "{what}: rejection took {dt:?} (budget {REJECT_BUDGET:?})"
    );
    msg
}

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nmr_spectrum_hostile_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

// ---------------------------------------------------------------------------------------------
// UCSF builders
// ---------------------------------------------------------------------------------------------

fn ucsf_header(ndim: u8, axes: &[(&str, u32, u32, f32, f32, f32)], total_len: usize) -> Vec<u8> {
    let mut b = vec![0u8; total_len.max(180 + 128 * axes.len())];
    b[..8].copy_from_slice(b"UCSF NMR");
    b[10] = ndim;
    b[11] = 1;
    b[13] = 2;
    for (i, &(nuc, n, tile, obs, sw, xmtr)) in axes.iter().enumerate() {
        let o = 180 + i * 128;
        b[o..o + nuc.len()].copy_from_slice(nuc.as_bytes());
        b[o + 8..o + 12].copy_from_slice(&n.to_be_bytes());
        b[o + 16..o + 20].copy_from_slice(&tile.to_be_bytes());
        b[o + 20..o + 24].copy_from_slice(&obs.to_be_bytes());
        b[o + 24..o + 28].copy_from_slice(&sw.to_be_bytes());
        b[o + 28..o + 32].copy_from_slice(&xmtr.to_be_bytes());
    }
    b
}

#[test]
fn ucsf_2d_692_byte_header_with_2_pow_30_axes_is_rejected_without_allocating() {
    // 692 bytes = 180 + 4 * 128: the exact hostile size from the review.
    let b = ucsf_header(
        2,
        &[
            ("15N", 1 << 30, 1 << 30, 60.8, 2400.0, 118.0),
            ("1H", 1 << 30, 1 << 30, 600.1, 9600.0, 4.7),
        ],
        692,
    );
    assert_eq!(b.len(), 692);
    let msg = expect_prompt_err("UCSF 2D 2^30 x 2^30", move || {
        read_ucsf_2d_from_bytes("hostile.ucsf", &b)
    });
    assert!(msg.contains("UCSF 2D"), "{msg}");

    // Same hostile dims with small, valid tiles: the point cap alone must reject.
    let b2 = ucsf_header(
        2,
        &[
            ("15N", 1 << 30, 64, 60.8, 2400.0, 118.0),
            ("1H", 1 << 30, 256, 600.1, 9600.0, 4.7),
        ],
        692,
    );
    expect_prompt_err("UCSF 2D 2^30 x 2^30 small tiles", move || {
        read_ucsf_2d_from_bytes("hostile2.ucsf", &b2)
    });

    // On-disk path variant of the same file.
    let dir = tmp_dir("ucsf2d_path");
    let p = dir.join("hostile.ucsf");
    fs::write(
        &p,
        ucsf_header(
            2,
            &[
                ("15N", 1 << 30, 1 << 30, 60.8, 2400.0, 118.0),
                ("1H", 1 << 30, 1 << 30, 600.1, 9600.0, 4.7),
            ],
            692,
        ),
    )
    .unwrap();
    expect_prompt_err("UCSF 2D path", move || read_ucsf_2d(&p));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn ucsf_2d_huge_tiles_on_tiny_file_are_rejected_before_tile_buffer_allocation() {
    // 16 x 16 spectrum declaring 2^30 x 2^30 tiles: `t0 * t1 * 4` would overflow/abort.
    let b = ucsf_header(
        2,
        &[
            ("15N", 16, 1 << 30, 60.8, 2400.0, 118.0),
            ("1H", 16, 1 << 30, 600.1, 9600.0, 4.7),
        ],
        1024,
    );
    expect_prompt_err("UCSF 2D huge tiles", move || {
        read_ucsf_2d_from_bytes("tiles.ucsf", &b)
    });
}

#[test]
fn ucsf_3d_wrapping_size_product_is_rejected_instead_of_zero_alloc_then_tile_panic() {
    // 2^22 * 2^21 * 2^21 == 2^64 wraps to 0 in usize. The file carries exactly one 8x8x8 tile so
    // the pre-fix reader got as far as writing that tile into a zero-length `data` (panic).
    let mut b = ucsf_header(
        3,
        &[
            ("13C", 1 << 22, 8, 150.9, 9000.0, 40.0),
            ("15N", 1 << 21, 8, 60.8, 2400.0, 118.0),
            ("1H", 1 << 21, 8, 600.1, 9600.0, 4.7),
        ],
        180 + 3 * 128,
    );
    b.extend(std::iter::repeat_n(0u8, 8 * 8 * 8 * 4));
    let msg = expect_prompt_err("UCSF 3D wrapped product", move || {
        read_ucsf_3d_from_bytes("wrap.ucsf", "HNCA", &b)
    });
    assert!(msg.contains("UCSF 3D"), "{msg}");

    // Non-wrapping but over-cap cube (2^11)^3 = 2^33 points.
    let b2 = ucsf_header(
        3,
        &[
            ("13C", 1 << 11, 8, 150.9, 9000.0, 40.0),
            ("15N", 1 << 11, 8, 60.8, 2400.0, 118.0),
            ("1H", 1 << 11, 8, 600.1, 9600.0, 4.7),
        ],
        1024,
    );
    expect_prompt_err("UCSF 3D over cap", move || {
        read_ucsf_3d_from_bytes("big.ucsf", "HNCA", &b2)
    });

    // On-disk path variant.
    let dir = tmp_dir("ucsf3d_path");
    let p = dir.join("wrap.ucsf");
    let mut f = ucsf_header(
        3,
        &[
            ("13C", 1 << 22, 8, 150.9, 9000.0, 40.0),
            ("15N", 1 << 21, 8, 60.8, 2400.0, 118.0),
            ("1H", 1 << 21, 8, 600.1, 9600.0, 4.7),
        ],
        180 + 3 * 128,
    );
    f.extend(std::iter::repeat_n(0u8, 8 * 8 * 8 * 4));
    fs::write(&p, f).unwrap();
    expect_prompt_err("UCSF 3D path", move || read_ucsf_3d(&p, "HNCA"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn ucsf_declared_payload_larger_than_file_is_rejected_before_allocation() {
    // 20000 x 20000 = 4e8 points (under the cap) but the file holds a single tile.
    let mut b = ucsf_header(
        2,
        &[
            ("15N", 20_000, 64, 60.8, 2400.0, 118.0),
            ("1H", 20_000, 256, 600.1, 9600.0, 4.7),
        ],
        180 + 2 * 128,
    );
    b.extend(std::iter::repeat_n(0u8, 64 * 256 * 4));
    let msg = expect_prompt_err("UCSF 2D payload > file", move || {
        read_ucsf_2d_from_bytes("short.ucsf", &b)
    });
    assert!(msg.contains("truncated"), "{msg}");
}

// ---------------------------------------------------------------------------------------------
// Bruker pdata
// ---------------------------------------------------------------------------------------------

fn bruker_procs(si: &str, xdim: &str) -> String {
    format!(
        "##TITLE= hostile\n##$SI= {si}\n##$XDIM= {xdim}\n##$SW_p= 5000.0\n##$SF= 500.0\n\
         ##$OFFSET= 10.0\n##$AXNUC= <1H>\n##$BYTORDP= 0\n##$DTYPP= 0\n##$NC_proc= 0\n##END=\n"
    )
}

#[test]
fn bruker_2rr_si_2_pow_32_returns_err_instead_of_wrapping_to_a_2_pow_64_loop() {
    // SI = 2^32 on both axes: `nx * ny` wraps to 0, the length check passed, and the tile loop ran
    // ~2^64 times (infinite hang). XDIM defaults to SI (single giant submatrix).
    let procs = bruker_procs("4294967296", "4294967296");
    let proc2s = procs.clone();
    let msg = expect_prompt_err("Bruker 2rr SI=2^32", move || {
        read_bruker_pdata_2d_from_bytes("hostile", &procs, &proc2s, &[0u8; 64])
    });
    assert!(msg.contains("Bruker 2rr"), "{msg}");

    // SI = 2^32 x 1 (no wrap, simply over the cap) and SI = 2^31 x 2^31.
    for (sx, sy) in [("4294967296", "1"), ("2147483648", "2147483648")] {
        let (px, py) = (bruker_procs(sx, sx), bruker_procs(sy, sy));
        expect_prompt_err("Bruker 2rr over cap", move || {
            read_bruker_pdata_2d_from_bytes("hostile", &px, &py, &[0u8; 64])
        });
    }
}

#[test]
fn bruker_2rr_on_disk_hostile_si_is_rejected_before_reading_the_file() {
    let dir = tmp_dir("bruker2d_path");
    let p = dir.join("pdata/1");
    fs::create_dir_all(&p).unwrap();
    fs::write(p.join("procs"), bruker_procs("4294967296", "4294967296")).unwrap();
    fs::write(p.join("proc2s"), bruker_procs("4294967296", "4294967296")).unwrap();
    fs::write(p.join("2rr"), [0u8; 256]).unwrap();
    let p2 = p.clone();
    expect_prompt_err("Bruker 2rr path", move || read_bruker_pdata_2d(&p2));

    // Plausible dims but a 256-byte file: truncation is caught by the stat-based preflight.
    fs::write(p.join("procs"), bruker_procs("4096", "256")).unwrap();
    fs::write(p.join("proc2s"), bruker_procs("4096", "256")).unwrap();
    let msg = expect_prompt_err("Bruker 2rr path truncated", move || {
        read_bruker_pdata_2d(&p)
    });
    assert!(msg.contains("truncated"), "{msg}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn bruker_3rrr_oversized_header_on_1kb_file_is_rejected_before_cube_allocation() {
    let kb = vec![0u8; 1024];

    // (a) 4096^3 = 2^36 points: over the cap (pre-fix: 256 GiB `vec!` -> abort).
    let big = bruker_procs("4096", "8");
    let (b1, b2, b3, k1) = (big.clone(), big.clone(), big.clone(), kb.clone());
    expect_prompt_err("Bruker 3rrr 4096^3", move || {
        read_bruker_pdata_3d_from_bytes("h", "HNCACB", &b1, &b2, &b3, &k1)
    });

    // (b) 1000^3 = 1e9 points (under the cap, ~4 GB) but only 1 KB of data: must fail on length
    //     BEFORE the 4 GB `vec!`.
    let mid = bruker_procs("1000", "10");
    let (m1, m2, m3, k2) = (mid.clone(), mid.clone(), mid.clone(), kb.clone());
    let msg = expect_prompt_err("Bruker 3rrr 1000^3 on 1KB", move || {
        read_bruker_pdata_3d_from_bytes("h", "HNCACB", &m1, &m2, &m3, &k2)
    });
    assert!(msg.contains("truncated"), "{msg}");

    // (c) wrapping product 2^22 * 2^21 * 2^21 = 2^64 -> 0.
    let (w1, w2, w3) = (
        bruker_procs("4194304", "8"),
        bruker_procs("2097152", "8"),
        bruker_procs("2097152", "8"),
    );
    expect_prompt_err("Bruker 3rrr wrapped", move || {
        read_bruker_pdata_3d_from_bytes("h", "HNCACB", &w1, &w2, &w3, &kb)
    });

    // (d) on-disk path: same 4096^3 header over a 1 KB `3rrr`.
    let dir = tmp_dir("bruker3d_path");
    let p = dir.join("pdata/1");
    fs::create_dir_all(&p).unwrap();
    for f in ["procs", "proc2s", "proc3s"] {
        fs::write(p.join(f), bruker_procs("4096", "8")).unwrap();
    }
    fs::write(p.join("3rrr"), [0u8; 1024]).unwrap();
    expect_prompt_err("Bruker 3rrr path", move || {
        read_bruker_pdata_3d(&p, "HNCACB")
    });
    let _ = fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------------------------
// NMRPipe
// ---------------------------------------------------------------------------------------------

fn pipe_header(fdsize: f32, fdspecnum: f32, fdf3size: f32, ndim: f32) -> Vec<u8> {
    let mut h = [0.0f32; 512];
    h[2] = 2.345;
    h[9] = ndim;
    h[24] = 2.0;
    h[25] = 1.0;
    h[26] = 3.0;
    h[27] = 4.0;
    h[99] = fdsize;
    h[219] = fdspecnum;
    h[15] = fdf3size;
    for slot in [100usize, 229, 11, 29] {
        h[slot] = 8000.0;
    }
    for slot in [119usize, 218, 10, 28] {
        h[slot] = 600.0;
    }
    let mut bytes = Vec::with_capacity(2048);
    for v in h {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[test]
fn nmrpipe_nan_inf_negative_and_huge_fdsize_fdspecnum_are_rejected() {
    for (label, v) in [
        ("NaN", f32::NAN),
        ("+Inf", f32::INFINITY),
        ("-Inf", f32::NEG_INFINITY),
        ("-1", -1.0),
        ("4e9", 4.0e9),
        ("2^32", 4_294_967_296.0),
    ] {
        // Hostile FDSIZE with a sane FDSPECNUM, and hostile FDSPECNUM with a sane FDSIZE.
        // Supply 2048 payload bytes (enough for 8*8*8 floats) so that pre-fix `NaN -> 1` and
        // `-1.0 -> 1` casts erroneously succeed with `Ok(...)` instead of tripping truncation.
        for (sx, sy) in [(v, 8.0f32), (8.0, v)] {
            let mut b2 = pipe_header(sx, sy, 8.0, 2.0);
            b2.extend_from_slice(&[0u8; 8 * 8 * 8 * 4]);
            expect_prompt_err(&format!("NMRPipe 2D {label}"), move || {
                read_nmrpipe_2d_from_bytes("hostile.ft2", &b2)
            });
            let mut b3 = pipe_header(sx, sy, 8.0, 3.0);
            b3.extend_from_slice(&[0u8; 8 * 8 * 8 * 4]);
            expect_prompt_err(&format!("NMRPipe 3D {label}"), move || {
                read_nmrpipe_3d_from_bytes("hostile.ft3", "HNCA", &b3)
            });
        }
        // Hostile 3rd dimension (FDF3SIZE / FDZSIZE slot).
        let mut b3 = pipe_header(8.0, 8.0, v, 3.0);
        b3.extend_from_slice(&[0u8; 8 * 8 * 8 * 4]);
        expect_prompt_err(&format!("NMRPipe 3D z {label}"), move || {
            read_nmrpipe_3d_from_bytes("hostile.ft3", "HNCA", &b3)
        });
    }
}

#[test]
fn nmrpipe_2_pow_32_squared_wrap_no_longer_returns_ok_with_empty_data() {
    // 2^32 * 2^32 wraps to 0 in usize: pre-fix returned Ok(data.len() == 0, axes of 2^32).
    let mut b = pipe_header(4_294_967_296.0, 4_294_967_296.0, 1.0, 2.0);
    b.extend_from_slice(&[0u8; 16]);
    expect_prompt_err("NMRPipe 2D 2^32 wrap", move || {
        read_nmrpipe_2d_from_bytes("wrap.ft2", &b)
    });
    let mut b3 = pipe_header(2_097_152.0, 2_097_152.0, 2_097_152.0, 3.0);
    b3.extend_from_slice(&[0u8; 16]);
    expect_prompt_err("NMRPipe 3D 2^21 cubed wrap", move || {
        read_nmrpipe_3d_from_bytes("wrap.ft3", "HNCA", &b3)
    });
}

#[test]
fn nmrpipe_declared_body_larger_than_file_is_rejected_before_allocation() {
    // 30000 x 30000 = 9e8 points (under the cap) over a 4-byte body.
    let mut b = pipe_header(30_000.0, 30_000.0, 1.0, 2.0);
    b.extend_from_slice(&[0u8; 4]);
    let msg = expect_prompt_err("NMRPipe 2D body", move || {
        read_nmrpipe_2d_from_bytes("short.ft2", &b)
    });
    assert!(msg.contains("truncated"), "{msg}");
}

// ---------------------------------------------------------------------------------------------
// Legitimate small files must still load (guards against over-eager rejection)
// ---------------------------------------------------------------------------------------------

#[test]
fn legitimate_small_files_still_round_trip_through_the_hardened_readers() {
    let dir = tmp_dir("good");

    let ax_x = AxisPpm::from_bounds("1H", 20, 10.0, 6.0, 600.0);
    let ax_y = AxisPpm::from_bounds("15N", 12, 130.0, 105.0, 60.8);
    let data2: Vec<f32> = (0..20 * 12).map(|i| i as f32 * 0.5 - 17.0).collect();
    let s2 = Spectrum2D {
        name: "ok".into(),
        axis_x: ax_x.clone(),
        axis_y: ax_y.clone(),
        data: data2.clone(),
    };
    let ucsf = dir.join("ok.ucsf");
    nmr_core::spectrum::write_ucsf_2d_tiled(&ucsf, &s2, 5, 8).unwrap();
    let r = read_ucsf_2d(&ucsf).expect("legit UCSF 2D with partial boundary tiles");
    assert_eq!(r.data, data2);
    let bytes = fs::read(&ucsf).unwrap();
    assert_eq!(read_ucsf_2d_from_bytes("ok", &bytes).unwrap().data, data2);

    let ft2 = dir.join("ok.ft2");
    nmr_core::spectrum::write_nmrpipe_2d(&ft2, &s2).unwrap();
    let fb = fs::read(&ft2).unwrap();
    assert_eq!(read_nmrpipe_2d_from_bytes("ok", &fb).unwrap().data, data2);

    let ax_z = AxisPpm::from_bounds("13C", 6, 70.0, 40.0, 150.0);
    let data3: Vec<f32> = (0..20 * 12 * 6).map(|i| (i % 97) as f32).collect();
    let s3 = Spectrum3D {
        name: "ok3".into(),
        exp: "HNCA".into(),
        axis_x: ax_x,
        axis_y: ax_y,
        axis_z: ax_z,
        data: data3.clone(),
    };
    let u3 = dir.join("ok3.ucsf");
    nmr_core::spectrum::write_ucsf_3d_tiled(&u3, &s3, 4, 5, 8).unwrap();
    assert_eq!(read_ucsf_3d(&u3, "HNCA").unwrap().data, data3);
    let ft3 = dir.join("ok.ft3");
    nmr_core::spectrum::write_nmrpipe_3d(&ft3, &s3).unwrap();
    let f3 = fs::read(&ft3).unwrap();
    assert_eq!(
        read_nmrpipe_3d_from_bytes("ok", "HNCA", &f3).unwrap().data,
        data3
    );

    let b2 = dir.join("bruker2d");
    nmr_core::spectrum::write_bruker_pdata_2d(&b2, &s2).unwrap();
    assert_eq!(read_bruker_pdata_2d(&b2).unwrap().data.len(), 20 * 12);
    let b3 = dir.join("bruker3d");
    nmr_core::spectrum::write_bruker_pdata_3d(&b3, &s3).unwrap();
    assert_eq!(
        read_bruker_pdata_3d(&b3, "HNCA").unwrap().data.len(),
        20 * 12 * 6
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn bruker_pdata_2d_axnuc_off_disambiguates_31p_1h_from_13c_15n_by_absolute_sf() {
    let procs_1h = "##TITLE= hp\n##$SI= 4\n##$XDIM= 4\n##$SW_p= 5000.0\n##$SF= 500.13\n\
                    ##$OFFSET= 10.0\n##$AXNUC= <off>\n##$BYTORDP= 0\n##$DTYPP= 0\n##$NC_proc= 0\n##END=\n";
    let proc2s_31p = "##TITLE= hp\n##$SI= 4\n##$XDIM= 4\n##$SW_p= 8000.0\n##$SF= 202.46\n\
                      ##$OFFSET= 20.0\n##$AXNUC= <off>\n##$BYTORDP= 0\n##$DTYPP= 0\n##$NC_proc= 0\n##END=\n";
    let raw = vec![0u8; 4 * 4 * 4];
    let spec_hp =
        read_bruker_pdata_2d_from_bytes("hp", procs_1h, proc2s_31p, &raw).expect("read 1H-31P 2rr");
    assert_eq!(spec_hp.axis_x.label, "1H");
    assert_eq!(spec_hp.axis_y.label, "31P");

    let procs_13c = "##TITLE= con\n##$SI= 4\n##$XDIM= 4\n##$SW_p= 3000.0\n##$SF= 150.90\n\
                     ##$OFFSET= 185.0\n##$AXNUC= <off>\n##$BYTORDP= 0\n##$DTYPP= 0\n##$NC_proc= 0\n##END=\n";
    let proc2s_15n = "##TITLE= con\n##$SI= 4\n##$XDIM= 4\n##$SW_p= 2000.0\n##$SF= 60.81\n\
                      ##$OFFSET= 138.0\n##$AXNUC= <off>\n##$BYTORDP= 0\n##$DTYPP= 0\n##$NC_proc= 0\n##END=\n";
    let spec_con = read_bruker_pdata_2d_from_bytes("con", procs_13c, proc2s_15n, &raw)
        .expect("read 13C-15N 2rr");
    assert_eq!(spec_con.axis_x.label, "13C");
    assert_eq!(spec_con.axis_y.label, "15N");
}
