//! Regression test: Snabb output must not depend on CPU load.
//!
//! On Intel CPUs with AMX (seen on a Xeon VM guest) onnxruntime's int8 AMX kernels
//! gave different, sometimes garbage, encoder output whenever the process was
//! preempted, even with one thread and no prata code involved (plain
//! onnxruntime reproduces it). prata keeps onnxruntime off AMX (snabb/noamx.rs);
//! this test checks the end result: parallel and repeated runs on a loaded CPU
//! give byte-identical transcripts.
//!
//! Needs the model and a Swedish audio file, so it is ignored by default:
//!   PRATA_SNABB_TEST_AUDIO=clip.wav cargo test --release -p prata --features snabb \
//!     --test snabb_determinism -- --ignored --nocapture
//! (`HF_HOME` as usual; the first run downloads the model.)
#![cfg(feature = "snabb")]

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn run(audio: &str, threads: Option<&str>) -> Vec<u8> {
    let mut c = Command::new(env!("CARGO_BIN_EXE_prata"));
    c.args([audio, "--model", "snabb", "--timestamps"]);
    if let Some(t) = threads {
        c.args(["--threads", t]);
    }
    let out = c.output().expect("run prata");
    assert!(out.status.success(), "prata failed: {}", String::from_utf8_lossy(&out.stderr));
    assert!(!out.stdout.is_empty(), "empty transcript");
    out.stdout
}

#[test]
#[ignore]
fn snabb_output_is_identical_under_load_serial_and_parallel() {
    let Ok(audio) = std::env::var("PRATA_SNABB_TEST_AUDIO") else {
        eprintln!("PRATA_SNABB_TEST_AUDIO not set; skipping");
        return;
    };
    let reference = run(&audio, None); // also downloads the model if needed

    // keep every core busy so the runs below get preempted a lot
    let stop = Arc::new(AtomicBool::new(false));
    let n = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let burners: Vec<_> = (0..n)
        .map(|_| {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut x = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    x = std::hint::black_box(x.wrapping_mul(6364136223846793005).wrapping_add(1));
                }
            })
        })
        .collect();

    let mut outputs = Vec::new();
    for t in [None, Some("1"), Some("4")] {
        outputs.push((format!("serial threads={t:?}"), run(&audio, t)));
    }
    let parallel: Vec<_> = (0..4)
        .map(|i| {
            let a = audio.clone();
            std::thread::spawn(move || (format!("parallel #{i}"), run(&a, None)))
        })
        .collect();
    outputs.extend(parallel.into_iter().map(|h| h.join().unwrap()));

    stop.store(true, Ordering::Relaxed);
    burners.into_iter().for_each(|b| b.join().unwrap());

    for (label, out) in &outputs {
        assert!(
            out == &reference,
            "{label}: transcript differs from the reference run\n--- reference\n{}\n--- {label}\n{}",
            String::from_utf8_lossy(&reference),
            String::from_utf8_lossy(out)
        );
    }
    eprintln!("{} runs identical ({} bytes)", outputs.len() + 1, reference.len());
}
