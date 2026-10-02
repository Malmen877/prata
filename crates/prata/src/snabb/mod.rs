//! Snabb: Klang Pianissimo (KlangAI/pianissimo-sv), a 0.6B FastConformer-TDT
//! model for Swedish, run with onnxruntime on the CPU (int8 ONNX export).
//!
//! Pipeline: ffmpeg -> 16 kHz PCM -> windows (`chunk`) -> log-mel (`mel`) ->
//! encoder -> greedy TDT decoding (`tdt`) -> words kept from each window's core
//! -> sentence segments (`text`). Only built with the `snabb` cargo feature; the
//! pure-Rust parts (features, decoding loop, merging, text) are always compiled
//! so they are unit-tested on every target.

#![cfg_attr(not(feature = "snabb"), allow(dead_code))]

pub mod chunk;
pub mod mel;
pub mod tdt;
pub mod text;
#[cfg(feature = "snabb")]
mod engine;
#[cfg(all(feature = "snabb", target_os = "linux", target_arch = "x86_64"))]
mod noamx;

use std::path::PathBuf;

/// Model repository and the revision this version of Prata was tested with.
pub const REPO: &str = "KlangAI/pianissimo-sv-onnx";
pub const REVISION: &str = "63730c6021234f26b9bbae9a07a04fec39e7a52e";
pub const ENCODER_FILE: &str = "encoder-model.int8.onnx";
pub const DECODER_FILE: &str = "decoder_joint-model.int8.onnx";
pub const VOCAB_FILE: &str = "vocab.txt";

/// Names accepted by `--model`.
pub fn is_snabb(model: &str) -> bool {
    matches!(model.to_ascii_lowercase().as_str(), "snabb" | "pianissimo" | "klangai/pianissimo-sv" | "klangai/pianissimo-sv-onnx")
}


pub struct Opts {
    pub audio: PathBuf,
    pub out: Option<PathBuf>,
    pub timestamps: bool,
    pub revision: Option<String>,
    pub vad: bool,
    pub vad_opts: crate::vad::VadOpts,
    pub chunk: chunk::ChunkOpts,
    pub threads: Option<usize>,
    pub verbose: bool,
}

#[cfg(not(feature = "snabb"))]
pub fn run(_: Opts) -> anyhow::Result<()> {
    anyhow::bail!(
        "Snabb (Klang Pianissimo) finns inte i den här versionen av prata: den kräver onnxruntime, som bara \
         byggs in för macOS på Apple Silicon och Linux x64. Välj en KB-Whisper-modell (t.ex. small) i stället."
    )
}

#[cfg(feature = "snabb")]
pub use imp::run;

#[cfg(feature = "snabb")]
mod imp {
    use super::*;
    use anyhow::{Context, Result};
    use hf_hub::progress::{DownloadEvent, ProgressEvent, ProgressHandler};
    use hf_hub::{HFClientSync, HFError};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;
    use text::{SegOpts, TimedTok};

    /// Prints download progress as `[info] downloading <file>: NN% of M MB`
    /// (the web UI turns the percentage into its progress bar).
    struct Printer {
        file: String,
        last: AtomicU64,
    }

    impl ProgressHandler for Printer {
        fn on_progress(&self, ev: &ProgressEvent) {
            let (done, total) = match ev {
                ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                    let Some(f) = files.iter().find(|f| f.filename.ends_with(&self.file)).or(files.first()) else { return };
                    (f.bytes_completed, f.total_bytes)
                }
                ProgressEvent::Download(DownloadEvent::AggregateProgress { bytes_completed, total_bytes, .. }) => {
                    (*bytes_completed, *total_bytes)
                }
                _ => return,
            };
            if total == 0 {
                return;
            }
            let pct = done * 100 / total;
            if self.last.swap(pct + 1, Ordering::Relaxed) != pct + 1 {
                eprintln!("[info] downloading {}: {pct}% of {} MB", self.file, total / 1_000_000);
            }
        }
    }

    /// Download (or reuse from the Hugging Face cache) one file of the model repo.
    fn hub_get(client: &HFClientSync, rev: &str, file: &str) -> Result<PathBuf> {
        let (owner, name) = hf_hub::split_id(REPO);
        let repo = client.model(owner, name);
        let dl = repo.download_file().filename(file).revision(rev.to_string());
        match dl.clone().local_files_only(true).send() {
            Ok(p) => Ok(p),
            Err(HFError::LocalEntryNotFound { .. }) => {
                eprintln!("[info] downloading {REPO}/{file} (Klang Pianissimo, CC BY 4.0) ...");
                let printer = Printer { file: file.to_string(), last: AtomicU64::new(0) };
                Ok(dl.progress(printer).send().with_context(|| format!("could not download {REPO}/{file}"))?)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn run(o: Opts) -> Result<()> {
        let rev = o.revision.clone().unwrap_or_else(|| REVISION.to_string());
        eprintln!("[info] model={REPO} (Snabb, Klang Pianissimo int8) revision={} device=cpu (onnxruntime)", &rev[..rev.len().min(12)]);
        // before onnxruntime initialises MLAS (see noamx.rs)
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        noamx::disable_amx();
        let t_load = Instant::now();
        let client = HFClientSync::new()?;
        let vocab_p = hub_get(&client, &rev, VOCAB_FILE)?;
        // Experimental knobs for evaluation (not part of the supported interface):
        // PRATA_SNABB_PRECISION=int8|fp16|fp32, PRATA_SNABB_BEAM=<n>, PRATA_SNABB_BEAM_NORM=1
        let precision = std::env::var("PRATA_SNABB_PRECISION").unwrap_or_else(|_| "int8".into());
        let beam: usize = std::env::var("PRATA_SNABB_BEAM").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let beam_norm = std::env::var("PRATA_SNABB_BEAM_NORM").map(|v| v == "1").unwrap_or(false);
        let (enc_f, dec_f) = match precision.as_str() {
            "fp32" => ("encoder-model.onnx".to_string(), "decoder_joint-model.onnx".to_string()),
            "fp16" | "int4" => (format!("encoder-model.{precision}.onnx"), format!("decoder_joint-model.{precision}.onnx")),
            _ => (ENCODER_FILE.to_string(), DECODER_FILE.to_string()),
        };
        if precision != "int8" || beam > 1 {
            eprintln!("[info] snabb: experimental precision={precision} beam={beam} norm={beam_norm}");
        }
        let dec_p = hub_get(&client, &rev, &dec_f)?;
        let enc_p = hub_get(&client, &rev, &enc_f)?;
        if precision == "fp32" {
            hub_get(&client, &rev, "encoder-model.onnx.data")?;
        }
        let vocab = text::Vocab::parse(&std::fs::read_to_string(&vocab_p)?)?;
        let mut encoder = engine::Encoder::load(&enc_p, o.threads)?;
        let mut joint = engine::DecoderJoint::load(&dec_p, 1024, o.threads)?;
        let load_s = t_load.elapsed().as_secs_f64();

        let pcm = crate::load_audio(&o.audio)?;
        let duration = pcm.len() as f64 / mel::SAMPLE_RATE as f64;
        let total = pcm.len() / mel::HOP;
        let db = crate::vad::frame_db(&pcm, total);
        let windows = chunk::plan(total, &db, &o.chunk);
        // Normalisation statistics over the whole recording: every window then gets
        // exactly the features of a single pass, independent of how much silence
        // happens to fall inside it.
        let stats = mel::Stats::of_signal(&pcm);
        // VAD (on by default): windows whose core has no speech are skipped.
        let speech = if o.vad {
            let (r, rep) = crate::vad::speech_regions(&db, &o.vad_opts);
            if rep.fallback_all { None } else { Some(r) }
        } else {
            None
        };
        eprintln!("[info] snabb: {} windows ({:.0} s + {:.0} s context), audio {duration:.1}s", windows.len(), o.chunk.core, o.chunk.context);
        let cfg = tdt::TdtConfig { vocab_size: vocab.len(), blank: vocab.blank, durations: vec![0, 1, 2, 3, 4], max_symbols: 10 };

        let t_asr = Instant::now();
        let mut words = vec![];
        for (i, w) in windows.iter().enumerate() {
            let t0 = Instant::now();
            if let Some(regs) = &speech {
                if !regs.iter().any(|r| r.start < w.keep_to && r.end() > w.keep_from) {
                    eprintln!("[info] window {:.1}s: no speech, skipped ({}/{})", w.keep_from as f64 / 100.0, i + 1, windows.len());
                    continue;
                }
            }
            let (feats, n_frames, valid) = mel::window_features(&pcm, w.start, w.end, &stats);
            let (enc, n, dim) = encoder.run(feats, n_frames, valid)?;
            let toks = if beam > 1 {
                tdt::beam(&mut joint, &cfg, n, |t| &enc[t * dim..(t + 1) * dim], beam, beam_norm)?
            } else {
                tdt::greedy(&mut joint, &cfg, n, |t| &enc[t * dim..(t + 1) * dim])?
            };
            let off = w.start as f64 / 100.0;
            let timed: Vec<TimedTok> = toks
                .iter()
                .map(|k| TimedTok {
                    id: k.id,
                    start: off + k.frame as f64 * text::FRAME_SECS,
                    end: off + (k.frame + k.dur.max(1)) as f64 * text::FRAME_SECS,
                })
                .collect();
            if o.verbose {
                let t: Vec<String> = toks.iter().map(|k| format!("{}@{}+{}", vocab.piece(k.id), k.frame, k.dur)).collect();
                eprintln!("[tokens] window {i} ({:.2}-{:.2}s): {}", off, w.end as f64 / 100.0, t.join(" "));
            }
            words.extend(chunk::keep_core(text::words(&vocab, &timed), w, total));
            eprintln!(
                "[info] window {:.1}s done in {:.1}s ({}/{})",
                w.keep_from as f64 / 100.0,
                t0.elapsed().as_secs_f64(),
                i + 1,
                windows.len()
            );
        }
        let asr_s = t_asr.elapsed().as_secs_f64();
        // token ends never run past the audio
        for wd in words.iter_mut() {
            for t in wd.toks.iter_mut() {
                t.end = t.end.min(duration);
                t.start = t.start.min(t.end);
            }
        }
        let segs = text::segments(&vocab, &words, &SegOpts::default());
        let output = if o.timestamps {
            segs.iter()
                .enumerate()
                .map(|(i, s)| format!("{}\n{} --> {}\n{}\n", i + 1, crate::srt_time(s.start), crate::srt_time(s.end), s.text))
                .collect::<Vec<_>>()
                .join("\n")
                .trim_end()
                .to_string()
        } else {
            segs.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ")
        };
        if let Some(out) = &o.out {
            std::fs::write(out, format!("{output}\n"))?;
            eprintln!("[info] wrote {}", out.display());
        }
        println!("{output}");
        eprintln!("[info] audio={duration:.1}s load={load_s:.1}s transcribe={asr_s:.1}s (RTF {:.3})", asr_s / duration.max(1e-9));
        Ok(())
    }
}
