//! prata – Swedish speech-to-text with KBLab kb-whisper models, using Hugging Face Candle.
//! Based on candle's whisper example (candle-examples/examples/whisper).

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

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
#[command(name = "prata", version, about = "Prata – Swedish speech-to-text with KBLab kb-whisper models (Candle)")]
struct Args {
    /// Audio file (any format ffmpeg can decode)
    audio: PathBuf,
    /// small | medium | large, or a full Hugging Face repo id
    #[arg(long, default_value = "large")]
    model: String,
    /// Model revision/branch on the Hub (KBLab also has e.g. "strict", "subtitle")
    #[arg(long, default_value = "main")]
    revision: String,
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
        .chunks_exact(4)
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
    model: m::model::Whisper,
    tokenizer: Tokenizer,
    timestamps: bool,
    suppress: Tensor,
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
        if let Some(&last) = sampled.iter().filter(|&&t| t >= tb).last() {
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
        let feats = self.model.encoder.forward(mel, true)?;
        let max_len = self.model.config.max_target_positions / 2;
        let mut tokens = self.prompt.clone();
        let mut sum_lp = 0f64;
        let mut no_speech_prob = f64::NAN;
        for i in 0..max_len {
            let tt = Tensor::new(tokens.as_slice(), mel.device())?.unsqueeze(0)?;
            let ys = self.model.decoder.forward(&tt, &feats, i == 0)?;
            if i == 0 {
                let l = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&l, 0)?.i(self.no_speech as usize)?.to_scalar::<f32>()? as f64;
            }
            let (_, seq, _) = ys.dims3()?;
            let mut logits = self.model.decoder.final_linear(&ys.i((..1, seq - 1..))?)?.i(0)?.i(0)?;
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
            if next == self.eot || tokens.len() > self.model.config.max_target_positions {
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

    /// 30 s sliding windows. In timestamp mode the next window starts at the last
    /// complete timestamp (like openai/whisper) so words are not cut at boundaries.
    fn run(&mut self, mel: &Tensor, content_frames: usize) -> Result<Vec<Seg>> {
        let mut seek = 0usize;
        let mut segs = vec![];
        let tb = self.ts_begin();
        while seek < content_frames {
            let t0 = Instant::now();
            let offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let seg_frames = usize::min(content_frames - seek, m::N_FRAMES);
            let window = mel.narrow(2, seek, m::N_FRAMES)?; // mel is padded by 30 s
            let dr = self.decode_with_fallback(&window)?;
            let seg_end = offset + (seg_frames * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                eprintln!("[info] {offset:.1}s: no speech, skipping window");
                seek += seg_frames;
                continue;
            }
            if !self.timestamps {
                segs.push(Seg { start: offset, end: seg_end, text: dr.text.trim().to_string() });
                seek += seg_frames;
            } else {
                let toks: Vec<u32> = dr.tokens.iter().cloned().filter(|&t| t != self.eot).collect();
                let ts = |t: u32| (t - tb) as f64 * 0.02;
                let mut cur: Vec<u32> = vec![];
                let mut start: Option<f64> = None;
                let mut last_complete_end: Option<f64> = None;
                let mut i = 0;
                while i < toks.len() {
                    let t = toks[i];
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
                    i += 1;
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
                match last_complete_end {
                    Some(e) if full_window && !single_ending && e > 1.0 => {
                        seek += ((e * m::SAMPLE_RATE as f64) as usize) / m::HOP_LENGTH;
                    }
                    _ => seek += seg_frames,
                }
            }
            eprintln!("[info] window {offset:.1}s done in {:.1}s", t0.elapsed().as_secs_f64());
        }
        Ok(segs)
    }
}

fn srt_time(t: f64) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let repo_id = match args.model.as_str() {
        "tiny" | "base" | "small" | "medium" | "large" => format!("KBLab/kb-whisper-{}", args.model),
        other => other.to_string(),
    };
    let device = device(args.cpu)?;
    eprintln!("[info] model={repo_id} device={device:?}");

    let t_load = Instant::now();
    let client = HFClientSync::new()?;
    let get = |f: &str| hub_get(&client, &repo_id, &args.revision, f);
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
    let model = m::model::Whisper::load(&vb, config.clone())?;
    let load_s = t_load.elapsed().as_secs_f64();

    let mut pcm = load_audio(&args.audio)?;
    let duration = pcm.len() as f64 / m::SAMPLE_RATE as f64;
    let content_frames = pcm.len() / m::HOP_LENGTH;
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
        model, tokenizer, timestamps: true, begin_suppress, prompt,
        eot, no_speech, no_ts, vocab, rng_state: 299792458,
    };

    let t_asr = Instant::now();
    let segs = dec.run(&mel, content_frames)?;
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
