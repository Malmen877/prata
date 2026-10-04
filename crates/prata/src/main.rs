//! prata – Swedish speech-to-text with KBLab kb-whisper models, using Hugging Face Candle.
//! Based on candle's whisper example (candle-examples/examples/whisper).

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod kvdec;
mod snabb;
mod vad;

use anyhow::{bail, Context, Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::{log_softmax, softmax};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use clap::Parser;
use hf_hub::{HFClientSync, HFError};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "prata", version, about = "Prata – local Swedish speech-to-text: KB-Whisper (KBLab, Candle) and Snabb (Klang Pianissimo, ONNX Runtime)", after_help = SNABB_HELP)]
struct Args {
    /// Audio file (any format ffmpeg can decode)
    audio: PathBuf,
    /// snabb (Klang Pianissimo, fastest; also "pianissimo"), tiny | base | small | medium |
    /// large (KB-Whisper), or a full Hugging Face repo id. The web UI preselects small;
    /// the CLI default is
    #[arg(long, default_value = "large")]
    model: String,
    /// Model revision/branch on the Hub (default: main; KBLab also has e.g. "strict",
    /// "subtitle"). Snabb defaults to the revision this version was tested with.
    #[arg(long)]
    revision: Option<String>,
    /// Write transcript (or SRT with --timestamps) to this file
    #[arg(long)]
    out: Option<PathBuf>,
    /// Output SRT-style timestamps
    #[arg(long)]
    timestamps: bool,
    /// Force CPU even if a GPU (metal/cuda) build is available
    #[arg(long)]
    cpu: bool,
    /// Language code to force
    #[arg(long, default_value = "sv")]
    language: String,
    /// Skip long silences before transcription (energy-based VAD; all models). Timestamps
    /// always refer to the original audio.
    #[arg(long, default_value = "on", value_parser = ["on", "off"])]
    vad: String,
    /// VAD: pauses at least this long (seconds) are removed
    #[arg(long, default_value_t = 1.0)]
    vad_min_silence: f64,
    /// VAD: audio kept before/after each speech region (seconds)
    #[arg(long, default_value_t = 0.3)]
    vad_pad: f64,
    /// VAD: speech threshold in dB above the noise floor (default: automatic)
    #[arg(long)]
    vad_threshold: Option<f64>,
    /// KB-Whisper: number of 30 s windows encoded and decoded together. 0 = automatic
    /// (4 on Metal/CUDA, 1 on CPU). Without --pack the windows of different speech
    /// regions are batched; each region is still decoded window by window exactly
    /// like the sequential decoder.
    #[arg(long, default_value_t = 0)]
    batch_size: usize,
    /// KB-Whisper, experimental: pack speech into fixed ≤30 s windows cut at quiet points, so
    /// even continuous speech can be batched. Faster, but window borders differ from
    /// the sequential decoder and some words can change (see docs/cli.md).
    #[arg(long)]
    pack: bool,
    /// KB-Whisper: cache the decoder's self-attention keys/values between steps (same result,
    /// much less work per token). `off` recomputes the whole sequence every step.
    #[arg(long, default_value = "on", value_parser = ["on", "off"])]
    kv_cache: String,
    /// Print the VAD/window plan to stderr
    #[arg(long)]
    verbose: bool,
    /// Snabb: window core length in seconds (long audio is always split into windows)
    #[arg(long, default_value_t = 30.0, hide = true)]
    snabb_window: f64,
    /// Snabb: context decoded on each side of a window, seconds
    #[arg(long, default_value_t = 5.0, hide = true)]
    snabb_context: f64,
    /// Snabb: onnxruntime threads for the encoder (default: onnxruntime's choice)
    #[arg(long, hide = true)]
    threads: Option<usize>,
}

#[cfg(feature = "snabb")]
const SNABB_HELP: &str = "Snabb (Klang Pianissimo by KlangAI, CC BY 4.0, https://huggingface.co/KlangAI/pianissimo-sv): available in this build (--model snabb). The model (~660 MB) is downloaded from Hugging Face on first use.";
#[cfg(not(feature = "snabb"))]
const SNABB_HELP: &str = "Snabb (Klang Pianissimo): not available in this build (needs macOS on Apple Silicon or Linux x64).";

enum Plan {
    /// Speech regions (frames, original timeline), each decoded with the
    /// sequential (seek) decoder
    Regions(Vec<(usize, usize)>),
    /// Packed ≤30 s windows, decoded in batches
    Windows(Vec<vad::Window>),
}

/// Largest window we pack (frames, 10 ms each): 29.5 s, leaving a little room
/// for the decoder to close the last segment with a timestamp.
const MAX_WIN_FRAMES: usize = 2950;
/// When a speech region is longer than a window, cut at the quietest point
/// within this many frames before the limit.
const SPLIT_SEARCH_FRAMES: usize = 800;

fn logsumexp(v: &[f32]) -> f32 {
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !mx.is_finite() {
        return mx;
    }
    mx + v.iter().map(|x| (x - mx).exp()).sum::<f32>().ln()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

/// The 30 s mel input for one planned window: its pieces cut out of the full-file
/// mel (so the log-mel normalisation is identical to the sequential path), padded
/// with the silent tail of the padded spectrogram.
fn window_mel(mel: &Tensor, w: &vad::Window, content_frames: usize) -> Result<Tensor> {
    let total_frames = mel.dim(2)?;
    let mut parts = Vec::with_capacity(w.pieces.len() + 1);
    for p in &w.pieces {
        parts.push(mel.narrow(2, p.start, p.len)?);
    }
    let used = w.frames();
    if used < m::N_FRAMES {
        let need = m::N_FRAMES - used;
        let from = content_frames.min(total_frames - need);
        parts.push(mel.narrow(2, from, need)?);
    }
    Ok(Tensor::cat(&parts, 2)?)
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    if candle_core::utils::cuda_is_available() {
        return Ok(Device::new_cuda(0)?);
    }
    if candle_core::utils::metal_is_available() {
        return Ok(Device::new_metal(0)?);
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    eprintln!("[info] running on CPU; build with `--features metal` to use the Apple GPU");
    Ok(Device::Cpu)
}

fn token_id(tok: &Tokenizer, s: &str) -> Result<u32> {
    tok.token_to_id(s).with_context(|| format!("no token id for {s}"))
}

/// Download (or reuse from ~/.cache/huggingface) one file of a model repo.
fn hub_get(client: &HFClientSync, repo_id: &str, rev: &str, file: &str) -> Result<PathBuf> {
    let (owner, name) = hf_hub::split_id(repo_id);
    let repo = client.model(owner, name);
    let dl = repo.download_file().filename(file).revision(rev.to_string());
    match dl.clone().local_files_only(true).send() {
        Ok(p) => Ok(p),
        Err(HFError::LocalEntryNotFound { .. }) => {
            eprintln!("[info] downloading {repo_id}/{file} ...");
            Ok(dl.send()?)
        }
        Err(e) => Err(e.into()),
    }
}

/// Decode any audio/video file to 16 kHz mono f32 PCM via ffmpeg.
fn load_audio(path: &Path) -> Result<Vec<f32>> {
    let out = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-f", "f32le", "-ac", "1", "-ar", "16000", "-"])
        .output()
        .context("failed to run ffmpeg - is it installed? (brew install ffmpeg / apt install ffmpeg)")?;
    if !out.status.success() {
        bail!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let pcm: Vec<f32> = out
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if pcm.is_empty() {
        bail!("no audio decoded from {}", path.display());
    }
    Ok(pcm)
}

fn compression_ratio(text: &str) -> f64 {
    use flate2::{write::ZlibEncoder, Compression};
    if text.is_empty() {
        return 0.0;
    }
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    enc.write_all(text.as_bytes()).ok();
    let c = enc.finish().map(|v| v.len()).unwrap_or(1).max(1);
    text.len() as f64 / c as f64
}

struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    compression_ratio: f64,
}

struct Seg {
    start: f64,
    end: f64,
    text: String,
}

struct Decoder {
    encoder: kvdec::AudioEncoder,
    decoder: kvdec::TextDecoder,
    max_pos: usize,
    /// self-attention KV cache in the decoder (--kv-cache)
    use_cache: bool,
    tokenizer: Tokenizer,
    timestamps: bool,
    suppress: Tensor,
    suppress_vec: Vec<f32>,
    begin_suppress: Vec<u32>,
    prompt: Vec<u32>,
    eot: u32,
    no_speech: u32,
    no_ts: u32,
    vocab: usize,
    rng_state: u64,
}

impl Decoder {
    fn ts_begin(&self) -> u32 {
        self.no_ts + 1
    }

    /// Whisper timestamp rules (pairs, monotonic, initial timestamp, prob-mass rule).
    fn timestamp_mask(&self, logits: &Tensor, tokens: &[u32]) -> Result<Tensor> {
        let tb = self.ts_begin();
        let v = self.vocab as u32;
        let sampled = &tokens[self.prompt.len()..];
        let mut mask = vec![0f32; self.vocab];
        let ninf = f32::NEG_INFINITY;
        let last_ts = sampled.last().map(|&t| t >= tb).unwrap_or(false);
        let pen_ts = sampled.len() < 2 || sampled[sampled.len() - 2] >= tb;
        if last_ts {
            if pen_ts {
                for i in tb..v { mask[i as usize] = ninf; }
            } else {
                for i in 0..self.eot { mask[i as usize] = ninf; }
            }
        }
        if let Some(&last) = sampled.iter().rfind(|&&t| t >= tb) {
            let lo = if last_ts && !pen_ts { last } else { last + 1 };
            for i in tb..lo.min(v) { mask[i as usize] = ninf; }
        }
        if sampled.is_empty() {
            for i in 0..tb { mask[i as usize] = ninf; }
            // max_initial_timestamp = 1.0 s (50 * 0.02)
            for i in (tb + 51)..v { mask[i as usize] = ninf; }
        }
        let logits = logits.broadcast_add(&Tensor::new(mask.as_slice(), logits.device())?)?;
        let lp: Vec<f32> = log_softmax(&logits, 0)?.to_vec1()?;
        let (text_lp, ts_lp) = lp.split_at(tb as usize);
        let mx = ts_lp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ts_sum = if mx.is_finite() {
            mx + ts_lp.iter().map(|x| (x - mx).exp()).sum::<f32>().ln()
        } else {
            mx
        };
        let text_max = text_lp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if ts_sum > text_max {
            let mut m2 = vec![0f32; self.vocab];
            for i in 0..tb { m2[i as usize] = ninf; }
            return Ok(logits.broadcast_add(&Tensor::new(m2.as_slice(), logits.device())?)?);
        }
        Ok(logits)
    }

    fn sample(&mut self, probs: &[f32]) -> u32 {
        // xorshift64* – tiny deterministic RNG for temperature fallback
        self.rng_state ^= self.rng_state >> 12;
        self.rng_state ^= self.rng_state << 25;
        self.rng_state ^= self.rng_state >> 27;
        let r = (self.rng_state.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64;
        let mut acc = 0f64;
        let total: f64 = probs.iter().map(|&p| p as f64).sum();
        for (i, &p) in probs.iter().enumerate() {
            acc += p as f64 / total;
            if acc >= r {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }

    fn decode(&mut self, mel: &Tensor, t: f64) -> Result<DecodingResult> {
        let feats = self.encoder.forward(mel)?;
        let max_len = self.max_pos / 2;
        let mut tokens = self.prompt.clone();
        let mut sum_lp = 0f64;
        let mut no_speech_prob = f64::NAN;
        for i in 0..max_len {
            let pos = if self.use_cache && i > 0 { tokens.len() - 1 } else { 0 };
            let tt = Tensor::new(&tokens[pos..], mel.device())?.unsqueeze(0)?;
            let ys = self.decoder.forward(&tt, &feats, pos, self.use_cache, i == 0)?;
            if i == 0 {
                let l = self.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&l, 0)?.i(self.no_speech as usize)?.to_scalar::<f32>()? as f64;
            }
            let (_, seq, _) = ys.dims3()?;
            let mut logits = self.decoder.final_linear(&ys.i((..1, seq - 1..))?)?.i(0)?.i(0)?;
            logits = logits.broadcast_add(&self.suppress)?;
            if i == 0 && !self.begin_suppress.is_empty() {
                let mut bm = vec![0f32; self.vocab];
                for &b in &self.begin_suppress { bm[b as usize] = f32::NEG_INFINITY; }
                logits = logits.broadcast_add(&Tensor::new(bm.as_slice(), mel.device())?)?;
            }
            if self.timestamps {
                logits = self.timestamp_mask(&logits, &tokens)?;
            }
            let next = if t > 0.0 {
                let p: Vec<f32> = softmax(&(&logits / t)?, 0)?.to_vec1()?;
                self.sample(&p)
            } else {
                let lv: Vec<f32> = logits.to_vec1()?;
                lv.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap()
            };
            tokens.push(next);
            let lp = log_softmax(&logits, 0)?.i(next as usize)?.to_scalar::<f32>()? as f64;
            if next == self.eot || tokens.len() > self.max_pos {
                break;
            }
            sum_lp += lp;
        }
        let text_tokens: Vec<u32> = tokens[self.prompt.len()..]
            .iter().cloned().filter(|&t| t < self.eot).collect();
        let text = self.tokenizer.decode(&text_tokens, true).map_err(E::msg)?;
        let n = (tokens.len() - self.prompt.len()).max(1);
        Ok(DecodingResult {
            compression_ratio: compression_ratio(&text),
            tokens: tokens[self.prompt.len()..].to_vec(),
            text,
            avg_logprob: sum_lp / n as f64,
            no_speech_prob,
        })
    }

    fn decode_with_fallback(&mut self, mel: &Tensor) -> Result<DecodingResult> {
        let mut last = None;
        for &t in m::TEMPERATURES.iter() {
            let dr = self.decode(mel, t)?;
            let needs = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                || dr.avg_logprob < m::LOGPROB_THRESHOLD;
            if !needs || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                return Ok(dr);
            }
            eprintln!("[info] fallback: temperature {t} gave avg_logprob={:.2} compression={:.2}",
                      dr.avg_logprob, dr.compression_ratio);
            last = Some(dr);
        }
        Ok(last.unwrap())
    }

    fn accept(dr: &DecodingResult) -> bool {
        let needs = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD || dr.avg_logprob < m::LOGPROB_THRESHOLD;
        !needs || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD
    }

    /// CPU implementation of the logit rules used in `decode` (suppress tokens,
    /// begin-suppress, Whisper timestamp rules incl. the probability-mass rule),
    /// applied to one row of logits in place.
    fn apply_rules(&self, logits: &mut [f32], tokens: &[u32], first: bool) {
        let ninf = f32::NEG_INFINITY;
        for (l, &s) in logits.iter_mut().zip(self.suppress_vec.iter()) {
            *l += s;
        }
        if first {
            for &b in &self.begin_suppress {
                logits[b as usize] = ninf;
            }
        }
        if !self.timestamps {
            return;
        }
        let tb = self.ts_begin() as usize;
        let v = self.vocab;
        let eot = self.eot as usize;
        let sampled = &tokens[self.prompt.len()..];
        let last_ts = sampled.last().map(|&t| t as usize >= tb).unwrap_or(false);
        let pen_ts = sampled.len() < 2 || sampled[sampled.len() - 2] as usize >= tb;
        if last_ts {
            if pen_ts {
                logits[tb..v].iter_mut().for_each(|x| *x = ninf);
            } else {
                logits[..eot].iter_mut().for_each(|x| *x = ninf);
            }
        }
        if let Some(&last) = sampled.iter().rfind(|&&t| t as usize >= tb) {
            let lo = if last_ts && !pen_ts { last } else { last + 1 } as usize;
            if lo > tb {
                logits[tb..lo.min(v)].iter_mut().for_each(|x| *x = ninf);
            }
        }
        if sampled.is_empty() {
            logits[..tb].iter_mut().for_each(|x| *x = ninf);
            // max_initial_timestamp = 1.0 s (50 * 0.02)
            if tb + 51 < v {
                logits[tb + 51..v].iter_mut().for_each(|x| *x = ninf);
            }
        }
        let total = logsumexp(logits);
        let ts_sum = logsumexp(&logits[tb..]) - total;
        let text_max = logits[..tb].iter().cloned().fold(ninf, f32::max) - total;
        if ts_sum > text_max {
            logits[..tb].iter_mut().for_each(|x| *x = ninf);
        }
    }

    /// Greedy (temperature 0) decoding of B windows at once: one encoder pass over
    /// a (B, n_mels, 3000) batch, then step-synchronous decoding of all items.
    /// Finished items are dropped from the batch (their cross-attention cache is
    /// rebuilt for the remaining rows), so long items don't pay for short ones.
    fn decode_batch(&mut self, mels: &Tensor) -> Result<Vec<DecodingResult>> {
        let b = mels.dim(0)?;
        let dev = mels.device().clone();
        let feats = self.encoder.forward(mels)?;
        let max_len = self.max_pos / 2;
        let max_pos = self.max_pos;
        let mut toks: Vec<Vec<u32>> = vec![self.prompt.clone(); b];
        let mut sum_lp = vec![0f64; b];
        let mut nsp = vec![f64::NAN; b];
        let mut rows: Vec<usize> = (0..b).collect(); // batch row -> item
        for i in 0..max_len {
            if rows.is_empty() {
                break;
            }
            let len = toks[rows[0]].len();
            let pos = if self.use_cache && i > 0 { len - 1 } else { 0 };
            let flat: Vec<u32> = rows.iter().flat_map(|&r| toks[r][pos..].iter().cloned()).collect();
            let tt = Tensor::from_vec(flat, (rows.len(), len - pos), &dev)?;
            let ys = self.decoder.forward(&tt, &feats, pos, self.use_cache, i == 0)?;
            if i == 0 {
                let l0: Vec<Vec<f32>> = self.decoder.final_linear(&ys.narrow(1, 0, 1)?)?.squeeze(1)?.to_vec2()?;
                for (k, &r) in rows.iter().enumerate() {
                    let l = &l0[k];
                    nsp[r] = ((l[self.no_speech as usize] - logsumexp(l)) as f64).exp();
                }
            }
            let seq = ys.dim(1)?;
            let last: Vec<Vec<f32>> = self.decoder.final_linear(&ys.narrow(1, seq - 1, 1)?)?.squeeze(1)?.to_vec2()?;
            let mut keep = vec![];
            for (k, &r) in rows.iter().enumerate() {
                let mut lg = last[k].clone();
                self.apply_rules(&mut lg, &toks[r], i == 0);
                let next = argmax(&lg);
                let lp = (lg[next] - logsumexp(&lg)) as f64;
                toks[r].push(next as u32);
                if next as u32 == self.eot || toks[r].len() > max_pos {
                    continue;
                }
                sum_lp[r] += lp;
                keep.push(k);
            }
            if keep.len() != rows.len() && !keep.is_empty() {
                // drop finished rows from the (self- and cross-attention) caches
                let idx: Vec<u32> = keep.iter().map(|&k| k as u32).collect();
                self.decoder.select_rows(&Tensor::new(idx.as_slice(), &dev)?)?;
            }
            rows = keep.iter().map(|&k| rows[k]).collect();
        }
        let mut out = Vec::with_capacity(b);
        for r in 0..b {
            let body = &toks[r][self.prompt.len()..];
            let text_tokens: Vec<u32> = body.iter().cloned().filter(|&t| t < self.eot).collect();
            let text = self.tokenizer.decode(&text_tokens, true).map_err(E::msg)?;
            let n = body.len().max(1);
            out.push(DecodingResult {
                compression_ratio: compression_ratio(&text),
                tokens: body.to_vec(),
                text,
                avg_logprob: sum_lp[r] / n as f64,
                no_speech_prob: nsp[r],
            });
        }
        Ok(out)
    }

    /// Turn one window's decoded tokens into segments on the original timeline.
    fn window_segments(&self, w: &vad::Window, dr: &DecodingResult) -> Result<Vec<Seg>> {
        let content = w.frames() as f64 / vad::FRAMES_PER_SEC;
        if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
            eprintln!("[info] {:.1}s: no speech, skipping window", w.orig_start());
            return Ok(vec![]);
        }
        let tb = self.ts_begin();
        let ts = |t: u32| (t - tb) as f64 * 0.02;
        let mut raw: Vec<(f64, f64, String)> = vec![];
        let mut cur: Vec<u32> = vec![];
        let mut start: Option<f64> = None;
        for &t in dr.tokens.iter().filter(|&&t| t != self.eot) {
            if t >= tb {
                if start.is_none() || cur.is_empty() {
                    start = Some(ts(t));
                } else {
                    let text = self.tokenizer.decode(&cur, true).map_err(E::msg)?;
                    raw.push((start.unwrap(), ts(t), text));
                    cur.clear();
                    start = None;
                }
            } else if t < self.eot {
                cur.push(t);
            }
        }
        if !cur.is_empty() {
            let text = self.tokenizer.decode(&cur, true).map_err(E::msg)?;
            raw.push((start.unwrap_or(0.0), content, text));
        }
        Ok(raw
            .into_iter()
            .filter(|(_, _, t)| !t.trim().is_empty())
            .map(|(s, e, text)| {
                let s0 = w.map(s.min(content), false);
                let e0 = w.map(e.min(content), true).max(s0);
                Seg { start: s0, end: e0, text: text.trim().to_string() }
            })
            .collect())
    }

    /// Decode pre-planned windows (VAD and/or batching), `batch` windows at a time.
    fn run_planned(&mut self, mel: &Tensor, wins: &[vad::Window], content_frames: usize, batch: usize) -> Result<Vec<Seg>> {
        let mut segs = vec![];
        let mut done = 0;
        for chunk in wins.chunks(batch.max(1)) {
            let t0 = Instant::now();
            let mels = chunk.iter().map(|w| window_mel(mel, w, content_frames)).collect::<Result<Vec<_>>>()?;
            let mut results = self.decode_batch(&Tensor::cat(&mels, 0)?)?;
            for (k, dr) in results.iter_mut().enumerate() {
                if Self::accept(dr) {
                    continue;
                }
                for &t in m::TEMPERATURES.iter().skip(1) {
                    eprintln!("[info] fallback: temperature {t} for window at {:.1}s (avg_logprob={:.2} compression={:.2})",
                              chunk[k].orig_start(), dr.avg_logprob, dr.compression_ratio);
                    *dr = self.decode(&mels[k], t)?;
                    if Self::accept(dr) {
                        break;
                    }
                }
            }
            for (w, dr) in chunk.iter().zip(results.iter()) {
                segs.extend(self.window_segments(w, dr)?);
            }
            let el = t0.elapsed().as_secs_f64();
            for w in chunk {
                done += 1;
                eprintln!("[info] window {:.1}s done in {:.1}s ({done}/{})", w.orig_start(), el / chunk.len() as f64, wins.len());
            }
        }
        Ok(segs)
    }

    /// 30 s sliding windows. In timestamp mode the next window starts at the last
    /// complete timestamp (like openai/whisper) so words are not cut at boundaries.
    /// The 30 s input for the sequential decoder at `seek` inside a region ending
    /// at `to`. Audio after `to` is replaced by silence, so no window looks across
    /// a removed pause; for the whole file (`to == content_frames`) this is exactly
    /// the classic window `mel[seek..seek+3000]`.
    fn seek_window(&self, mel: &Tensor, seek: usize, to: usize, content_frames: usize) -> Result<(Tensor, usize)> {
        let seg_frames = usize::min(to - seek, m::N_FRAMES);
        let window = if seek + m::N_FRAMES <= to || to == content_frames {
            mel.narrow(2, seek, m::N_FRAMES)? // mel is padded by 30 s
        } else {
            Tensor::cat(&[mel.narrow(2, seek, seg_frames)?, mel.narrow(2, content_frames, m::N_FRAMES - seg_frames)?], 2)?
        };
        Ok((window, seg_frames))
    }

    /// Turn one decoded window at `seek` into segments; returns how far to advance.
    fn consume(&self, dr: &DecodingResult, seek: usize, seg_frames: usize, segs: &mut Vec<Seg>) -> Result<usize> {
        let tb = self.ts_begin();
        let offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
        let seg_end = offset + (seg_frames * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
        if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
            eprintln!("[info] {offset:.1}s: no speech, skipping window");
            return Ok(seg_frames);
        }
        if !self.timestamps {
            segs.push(Seg { start: offset, end: seg_end, text: dr.text.trim().to_string() });
            return Ok(seg_frames);
        }
        let toks: Vec<u32> = dr.tokens.iter().cloned().filter(|&t| t != self.eot).collect();
        let ts = |t: u32| (t - tb) as f64 * 0.02;
        let mut cur: Vec<u32> = vec![];
        let mut start: Option<f64> = None;
        let mut last_complete_end: Option<f64> = None;
        for &t in &toks {
            if t >= tb {
                if start.is_none() || cur.is_empty() {
                    start = Some(ts(t));
                } else {
                    let text = self.tokenizer.decode(&cur, true).map_err(E::msg)?;
                    let (s, e) = (start.unwrap(), ts(t));
                    if !text.trim().is_empty() {
                        segs.push(Seg { start: offset + s, end: offset + e, text: text.trim().to_string() });
                    }
                    last_complete_end = Some(e);
                    cur.clear();
                    start = None;
                    // a following timestamp starts the next segment
                }
            } else if t < self.eot {
                cur.push(t);
            }
        }
        let single_ending = toks.len() >= 2 && toks[toks.len() - 1] >= tb && toks[toks.len() - 2] < tb;
        let full_window = seg_frames >= m::N_FRAMES;
        if !cur.is_empty() {
            // trailing text without closing timestamp
            let text = self.tokenizer.decode(&cur, true).map_err(E::msg)?;
            if !full_window || last_complete_end.is_none() {
                segs.push(Seg { start: offset + start.unwrap_or(0.0), end: seg_end, text: text.trim().to_string() });
                last_complete_end = None;
            }
        }
        Ok(match last_complete_end {
            Some(e) if full_window && !single_ending && e > 1.0 => ((e * m::SAMPLE_RATE as f64) as usize) / m::HOP_LENGTH,
            _ => seg_frames,
        })
    }

    /// Sequential (seek-based) decoding of each region `(from, to)` (frames on the
    /// original timeline). With `batch > 1`, the next window of up to `batch`
    /// different regions is encoded and decoded together; within a region the
    /// windows are exactly those of the sequential decoder.
    fn run_regions(&mut self, mel: &Tensor, regions: &[(usize, usize)], content_frames: usize, batch: usize) -> Result<Vec<Seg>> {
        let mut seek: Vec<usize> = regions.iter().map(|r| r.0).collect();
        let mut segs: Vec<Vec<Seg>> = regions.iter().map(|_| vec![]).collect();
        loop {
            let active: Vec<usize> = (0..regions.len()).filter(|&i| seek[i] < regions[i].1).take(batch.max(1)).collect();
            if active.is_empty() {
                break;
            }
            let t0 = Instant::now();
            let mut wins = vec![];
            for &i in &active {
                wins.push(self.seek_window(mel, seek[i], regions[i].1, content_frames)?);
            }
            let results = if active.len() == 1 {
                vec![self.decode_with_fallback(&wins[0].0)?]
            } else {
                let mels: Vec<Tensor> = wins.iter().map(|w| w.0.clone()).collect();
                let mut rs = self.decode_batch(&Tensor::cat(&mels, 0)?)?;
                for (k, dr) in rs.iter_mut().enumerate() {
                    if Self::accept(dr) {
                        continue;
                    }
                    for &t in m::TEMPERATURES.iter().skip(1) {
                        eprintln!("[info] fallback: temperature {t} gave avg_logprob={:.2} compression={:.2}",
                                  dr.avg_logprob, dr.compression_ratio);
                        *dr = self.decode(&mels[k], t)?;
                        if Self::accept(dr) {
                            break;
                        }
                    }
                }
                rs
            };
            let el = t0.elapsed().as_secs_f64() / active.len() as f64;
            for (k, &i) in active.iter().enumerate() {
                let offset = (seek[i] * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
                seek[i] += self.consume(&results[k], seek[i], wins[k].1, &mut segs[i])?;
                eprintln!("[info] window {offset:.1}s done in {el:.1}s");
            }
        }
        Ok(segs.into_iter().flatten().collect())
    }
}

fn srt_time(t: f64) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if snabb::is_snabb(&args.model) {
        return snabb::run(snabb::Opts {
            audio: args.audio.clone(),
            out: args.out.clone(),
            timestamps: args.timestamps,
            revision: args.revision.clone(),
            vad: args.vad == "on",
            vad_opts: vad::VadOpts {
                min_silence: args.vad_min_silence,
                pad: args.vad_pad,
                threshold_db: args.vad_threshold,
                ..Default::default()
            },
            chunk: snabb::chunk::ChunkOpts { core: args.snabb_window, context: args.snabb_context, ..Default::default() },
            threads: args.threads,
            verbose: args.verbose,
        });
    }
    let revision = args.revision.as_deref().unwrap_or("main");
    let repo_id = match args.model.as_str() {
        "tiny" | "base" | "small" | "medium" | "large" => format!("KBLab/kb-whisper-{}", args.model),
        other => other.to_string(),
    };
    let device = device(args.cpu)?;
    eprintln!("[info] model={repo_id} device={device:?}");

    let t_load = Instant::now();
    let client = HFClientSync::new()?;
    let get = |f: &str| hub_get(&client, &repo_id, revision, f);
    let config_path = get("config.json")?;
    let tokenizer_path = get("tokenizer.json")?;
    let gen_path = get("generation_config.json").ok();
    let weights_path = get("model.safetensors")?;

    // KBLab's config.json has "suppress_tokens": null (moved to generation_config.json).
    let mut cfg_json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    let gen: serde_json::Value = gen_path
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(serde_json::Value::Null);
    if cfg_json["suppress_tokens"].is_null() {
        cfg_json["suppress_tokens"] = gen.get("suppress_tokens").cloned().unwrap_or(serde_json::json!([]));
    }
    let begin_suppress: Vec<u32> = gen.get("begin_suppress_tokens")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();
    let config: Config = serde_json::from_value(cfg_json)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

    let mel_bytes: &[u8] = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes"),
        128 => include_bytes!("melfilters128.bytes"),
        n => bail!("unexpected num_mel_bins {n}"),
    };
    let mut filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut filters);

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], m::DTYPE, &device)? };
    // Ported encoder/decoder (kvdec.rs): same weights and ops as candle's whisper
    // model, plus a self-attention KV cache in the decoder.
    let encoder = kvdec::AudioEncoder::load(vb.pp("model.encoder"), &config)?;
    let decoder = kvdec::TextDecoder::load(vb.pp("model.decoder"), &config)?;
    let load_s = t_load.elapsed().as_secs_f64();

    let mut pcm = load_audio(&args.audio)?;
    let duration = pcm.len() as f64 / m::SAMPLE_RATE as f64;
    let content_frames = pcm.len() / m::HOP_LENGTH;
    let batch = match args.batch_size {
        0 if device.is_cpu() => 1,
        0 => 4,
        n => n,
    };
    let use_vad = args.vad == "on";
    // Decoding plan: speech regions (VAD) decoded with the sequential decoder, or
    // (--pack) regions packed into fixed windows.
    let db = vad::frame_db(&pcm, content_frames);
    let regions = if use_vad {
        let opts = vad::VadOpts {
            min_silence: args.vad_min_silence,
            pad: args.vad_pad,
            threshold_db: args.vad_threshold,
            ..Default::default()
        };
        let (r, rep) = vad::speech_regions(&db, &opts);
        let kept: usize = r.iter().map(|p| p.len).sum();
        eprintln!(
            "[info] vad: {} speech regions, kept {:.1}s of {duration:.1}s (floor {:.0} dB, peak {:.0} dB, threshold {:.0} dB{})",
            r.len(), kept as f64 / vad::FRAMES_PER_SEC, rep.floor_db, rep.peak_db, rep.threshold_db,
            if rep.fallback_all { ", low dynamic range: keeping everything" } else { "" }
        );
        r
    } else {
        vec![vad::Piece { start: 0, len: content_frames }]
    };
    let plan = if args.pack {
        let wins = vad::pack(&vad::split_long(&regions, &db, MAX_WIN_FRAMES, SPLIT_SEARCH_FRAMES), MAX_WIN_FRAMES);
        eprintln!("[info] packed into {} windows, batch size {batch}", wins.len());
        if args.verbose {
            for (i, w) in wins.iter().enumerate() {
                let p: Vec<String> = w.pieces.iter()
                    .map(|p| format!("{:.2}-{:.2}", p.start as f64 / 100.0, p.end() as f64 / 100.0)).collect();
                eprintln!("[plan] window {i}: {:.2}-{:.2}s, {:.2}s content: {}", w.orig_start(), w.orig_end(), w.frames() as f64 / 100.0, p.join(" + "));
            }
        }
        Plan::Windows(wins)
    } else {
        if args.verbose {
            for p in &regions {
                eprintln!("[plan] region {:.2}-{:.2}s", p.start as f64 / 100.0, p.end() as f64 / 100.0);
            }
        }
        Plan::Regions(regions.iter().map(|p| (p.start, p.end().min(content_frames))).collect())
    };
    pcm.extend(std::iter::repeat_n(0f32, m::N_SAMPLES)); // pad 30 s like openai/whisper
    let mel = audio::pcm_to_mel(&config, &pcm, &filters);
    let n_frames = mel.len() / config.num_mel_bins;
    let mel = Tensor::from_vec(mel, (1, config.num_mel_bins, n_frames), &device)?;

    let sot = token_id(&tokenizer, m::SOT_TOKEN)?;
    let lang = token_id(&tokenizer, &format!("<|{}|>", args.language))
        .with_context(|| format!("language {} not supported", args.language))?;
    let transcribe = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
    let no_ts = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
    let eot = token_id(&tokenizer, m::EOT_TOKEN)?;
    let no_speech = m::NO_SPEECH_TOKENS.iter().find_map(|t| tokenizer.token_to_id(t))
        .context("no no-speech token")?;
    // Always decode in timestamp mode internally: it lets each 30 s window end on a
    // complete segment and the next window resume there, so words at window borders
    // are not lost. Plain-text output simply joins the segments.
    let prompt = vec![sot, lang, transcribe];
    let vocab = config.vocab_size;
    let suppress: Vec<f32> = (0..vocab as u32)
        .map(|i| if config.suppress_tokens.contains(&i) || i == no_ts { f32::NEG_INFINITY } else { 0.0 })
        .collect();
    let mut dec = Decoder {
        suppress: Tensor::new(suppress.as_slice(), &device)?,
        suppress_vec: suppress.clone(),
        encoder, decoder, max_pos: config.max_target_positions, use_cache: args.kv_cache == "on",
        tokenizer, timestamps: true, begin_suppress, prompt,
        eot, no_speech, no_ts, vocab, rng_state: 299792458,
    };

    let t_asr = Instant::now();
    let segs = match &plan {
        Plan::Windows(wins) => dec.run_planned(&mel, wins, content_frames, batch)?,
        Plan::Regions(regions) => dec.run_regions(&mel, regions, content_frames, batch)?,
    };
    let asr_s = t_asr.elapsed().as_secs_f64();

    let output = if args.timestamps {
        segs.iter().enumerate()
            .map(|(i, s)| format!("{}\n{} --> {}\n{}\n", i + 1, srt_time(s.start), srt_time(s.end.min(duration)), s.text))
            .collect::<Vec<_>>().join("\n").trim_end().to_string()
    } else {
        segs.iter().map(|s| s.text.as_str()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ")
    };
    if let Some(out) = &args.out {
        std::fs::write(out, format!("{output}\n"))?;
        eprintln!("[info] wrote {}", out.display());
    }
    println!("{output}");
    eprintln!("[info] audio={duration:.1}s load={load_s:.1}s transcribe={asr_s:.1}s (RTF {:.2})", asr_s / duration);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srt_time_format() {
        assert_eq!(srt_time(0.0), "00:00:00,000");
        assert_eq!(srt_time(3725.5), "01:02:05,500");
    }

    #[test]
    fn mel_filters_have_expected_size() {
        // 80 and 128 mel bins × 201 FFT bins × f32
        assert_eq!(include_bytes!("melfilters.bytes").len(), 80 * 201 * 4);
        assert_eq!(include_bytes!("melfilters128.bytes").len(), 128 * 201 * 4);
    }
}
