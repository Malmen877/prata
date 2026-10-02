//! prata-web: local web UI for Prata – Swedish speech-to-text (KBLab kb-whisper models).
//!
//! Backends (chosen per job):
//!   * `prata`  – the Rust/Candle CLI (next to this binary, or $PRATA_BIN, or on $PATH)
//!   * `python` – the Python fallback transcribe.py ($PRATA_PYTHON_SCRIPT)
//!
//! Job model: POST /api/jobs (multipart) -> {id}; GET /api/jobs/{id} polls status.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, Path as AxPath, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use tokio::{
    io::{AsyncReadExt, BufReader},
    process::Command,
    sync::{Mutex, Semaphore},
};

const INDEX_HTML: &str = include_str!("index.html");

// ---------------------------------------------------------------- config

#[derive(Clone, Debug)]
struct Config {
    port: u16,
    host: String,
    backend: String, // auto | prata | python
    prata_bin: Option<PathBuf>,
    prata_args: Option<String>,
    python: String,
    python_script: Option<PathBuf>,
    default_model: String,
    work_dir: PathBuf,
    max_upload_mb: usize,
}

/// Read `PRATA_<k>`, falling back to the legacy `KBW_<k>` name.
fn env(k: &str) -> Option<String> {
    [format!("PRATA_{k}"), format!("KBW_{k}")]
        .iter()
        .find_map(|n| std::env::var(n).ok())
        .filter(|v| !v.trim().is_empty())
}

/// Find an executable on $PATH.
fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(name)).find(|p| p.is_file())
}

impl Config {
    fn from_env() -> Self {
        // Directory of the running prata-web binary. In a release bundle `prata` and
        // `transcribe.py` sit next to it; in a cargo workspace build both binaries end up
        // in target/{release,debug}/.
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.canonicalize().ok())
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        let exe_name = if cfg!(windows) { "prata.exe" } else { "prata" };

        let prata_bin = env("BIN").map(PathBuf::from).or_else(|| {
            let mut c = vec![exe_dir.join(exe_name)];
            // workspace layout: target/<profile>/prata-web -> try the other profile too
            if let Some(target) = exe_dir.parent() {
                c.push(target.join("release").join(exe_name));
                c.push(target.join("debug").join(exe_name));
            }
            c.into_iter().find(|p| p.is_file()).or_else(|| which(exe_name))
        });
        let python_script = env("PYTHON_SCRIPT").map(PathBuf::from).or_else(|| {
            let mut c = vec![exe_dir.join("transcribe.py"), exe_dir.join("python/transcribe.py")];
            // workspace layout: <root>/target/<profile>/prata-web -> <root>/python/transcribe.py
            if let Some(root) = exe_dir.parent().and_then(|t| t.parent()) {
                c.push(root.join("python/transcribe.py"));
            }
            c.into_iter().find(|p| p.is_file())
        });
        Config {
            port: env("WEB_PORT").or_else(|| env("PORT")).and_then(|p| p.parse().ok()).unwrap_or(8795),
            host: env("WEB_HOST").unwrap_or_else(|| "127.0.0.1".into()),
            backend: env("BACKEND").unwrap_or_else(|| "auto".into()).to_lowercase(),
            prata_bin,
            prata_args: env("ARGS"),
            python: env("PYTHON").unwrap_or_else(|| "python3".into()),
            python_script,
            default_model: env("MODEL").unwrap_or_else(|| "small".into()),
            work_dir: env("WORK_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir().join("prata-web")),
            max_upload_mb: env("MAX_UPLOAD_MB").and_then(|v| v.parse().ok()).unwrap_or(1024),
        }
    }
}

// ---------------------------------------------------------------- prata detection

#[derive(Clone, Debug, Serialize, Default)]
struct PrataInfo {
    path: Option<String>,
    works: bool,
    note: String,
    #[serde(skip)]
    help: String,
}

async fn probe_prata(cfg: &Config) -> PrataInfo {
    let Some(bin) = cfg.prata_bin.clone() else {
        return PrataInfo { note: "prata-binär hittades inte".into(), ..Default::default() };
    };
    let path = Some(bin.display().to_string());
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        Command::new(&bin).arg("--help").stdin(Stdio::null()).output(),
    )
    .await;
    match out {
        Ok(Ok(o)) if o.status.success() => {
            let help = String::from_utf8_lossy(&o.stdout).to_string()
                + &String::from_utf8_lossy(&o.stderr);
            PrataInfo { path, works: true, note: "prata --help OK".into(), help }
        }
        Ok(Ok(o)) => PrataInfo {
            path,
            note: format!("prata --help misslyckades ({})", o.status),
            ..Default::default()
        },
        Ok(Err(e)) => PrataInfo { path, note: format!("kunde inte starta prata: {e}"), ..Default::default() },
        Err(_) => PrataInfo { path, note: "prata --help timeout".into(), ..Default::default() },
    }
}

/// Build prata arguments. If PRATA_ARGS is set it is used as a template
/// ({input}, {model}, {out} placeholders, whitespace-split). Otherwise flags are
/// inferred from `prata --help`.
fn prata_args(cfg: &Config, info: &PrataInfo, input: &Path, model: &str, out_json: &Path) -> Vec<String> {
    let input_s = input.display().to_string();
    if let Some(t) = &cfg.prata_args {
        let mut v: Vec<String> = t
            .split_whitespace()
            .map(|a| {
                a.replace("{input}", &input_s)
                    .replace("{model}", model)
                    .replace("{out}", &out_json.display().to_string())
            })
            .collect();
        if !t.contains("{input}") {
            v.push(input_s);
        }
        return v;
    }
    let h = &info.help;
    let has = |f: &str| {
        h.split(|c: char| c.is_whitespace() || c == ',' || c == '=' || c == '[' || c == ']')
            .any(|w| w == f)
    };
    let mut v = Vec::new();
    // input: positional unless an --input flag exists
    if has("--input") {
        v.push("--input".into());
        v.push(input_s);
    } else {
        v.push(input_s);
    }
    if has("--model") {
        v.push("--model".into());
        v.push(model.into());
    }
    if has("--language") {
        v.push("--language".into());
        v.push("sv".into());
    } else if has("--lang") {
        v.push("--lang".into());
        v.push("sv".into());
    }
    // Ask for the richest output available.
    if has("--json") {
        v.push("--json".into());
    } else if has("--format") {
        let fmt = if h.contains("json") { "json" } else if h.contains("srt") { "srt" } else { "" };
        if !fmt.is_empty() {
            v.push("--format".into());
            v.push(fmt.into());
        }
    } else if has("--output-format") {
        let fmt = if h.contains("json") { "json" } else { "srt" };
        v.push("--output-format".into());
        v.push(fmt.into());
    } else if has("--srt") {
        v.push("--srt".into());
    } else if has("--timestamps") {
        v.push("--timestamps".into());
    }
    // Speed options of the Candle CLI (only passed through when set).
    for (k, flag) in [("VAD", "--vad"), ("BATCH_SIZE", "--batch-size")] {
        if let Some(val) = env(k) {
            if has(flag) {
                v.push(flag.into());
                v.push(val.trim().into());
            }
        }
    }
    v
}

// ---------------------------------------------------------------- transcript model

#[derive(Clone, Debug, Serialize)]
struct Segment {
    start: f64,
    end: f64,
    text: String,
}

fn fmt_ts(t: f64, sep: char) -> String {
    let ms_total = (t.max(0.0) * 1000.0).round() as u64;
    let (h, rem) = (ms_total / 3_600_000, ms_total % 3_600_000);
    let (m, rem) = (rem / 60_000, rem % 60_000);
    let (s, ms) = (rem / 1000, rem % 1000);
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

fn to_srt(segs: &[Segment]) -> String {
    let mut out = String::new();
    for (i, s) in segs.iter().enumerate() {
        out += &format!(
            "{}\r\n{} --> {}\r\n{}\r\n\r\n",
            i + 1,
            fmt_ts(s.start, ','),
            fmt_ts(s.end.max(s.start), ','),
            s.text.trim()
        );
    }
    out
}

fn to_txt(segs: &[Segment], with_ts: bool) -> String {
    if with_ts {
        segs.iter()
            .map(|s| format!("[{} - {}] {}\n", fmt_ts(s.start, '.'), fmt_ts(s.end, '.'), s.text.trim()))
            .collect()
    } else {
        let t: Vec<&str> = segs.iter().map(|s| s.text.trim()).filter(|s| !s.is_empty()).collect();
        t.join(" ") + "\n"
    }
}

/// Parse "HH:MM:SS,mmm", "MM:SS.mmm", "SS.mmm" or a plain float into seconds.
fn parse_ts(s: &str) -> Option<f64> {
    let s = s.trim().replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    let mut secs = 0.0;
    for p in &parts {
        secs = secs * 60.0 + p.trim().parse::<f64>().ok()?;
    }
    Some(secs)
}

fn json_f64(v: &serde_json::Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str().and_then(parse_ts))
}

/// Accepts many JSON shapes: {segments:[{start,end,text}]}, [{start,end,text}],
/// {chunks:[{timestamp:[a,b],text}]}, {text:"..."}; one object per line (JSONL) also works.
fn parse_json(s: &str, duration: f64) -> Option<Vec<Segment>> {
    let seg_from = |o: &serde_json::Value| -> Option<Segment> {
        let text = o.get("text")?.as_str()?.to_string();
        let (start, end) = if let Some(ts) = o.get("timestamp").and_then(|t| t.as_array()) {
            (ts.first().and_then(json_f64).unwrap_or(0.0), ts.get(1).and_then(json_f64).unwrap_or(duration))
        } else {
            let st = ["start", "start_s", "t0", "from"].iter().find_map(|k| o.get(*k).and_then(json_f64));
            let en = ["end", "end_s", "t1", "to"].iter().find_map(|k| o.get(*k).and_then(json_f64));
            (st.unwrap_or(0.0), en.unwrap_or(duration))
        };
        Some(Segment { start, end, text })
    };
    let from_value = |v: &serde_json::Value| -> Option<Vec<Segment>> {
        let arr = if let Some(a) = v.as_array() {
            Some(a.clone())
        } else {
            ["segments", "chunks"].iter().find_map(|k| v.get(*k).and_then(|x| x.as_array()).cloned())
        };
        if let Some(a) = arr {
            let segs: Vec<Segment> = a.iter().filter_map(seg_from).collect();
            if !segs.is_empty() {
                return Some(segs);
            }
        }
        if v.get("start").is_some() || v.get("timestamp").is_some() {
            return seg_from(v).map(|s| vec![s]);
        }
        v.get("text").and_then(|t| t.as_str()).map(|t| {
            vec![Segment { start: 0.0, end: duration, text: t.trim().to_string() }]
        })
    };
    let trimmed = s.trim();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return from_value(&v);
    }
    // JSON may be preceded by log lines; try from first '{' / '['
    if let Some(i) = trimmed.find(|c| c == '{' || c == '[') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&trimmed[i..]) {
            if let Some(r) = from_value(&v) {
                return Some(r);
            }
        }
    }
    // JSONL
    let segs: Vec<Segment> = trimmed
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l.trim()).ok())
        .filter_map(|v| seg_from(&v))
        .collect();
    (!segs.is_empty()).then_some(segs)
}

/// SRT blocks or lines like "[00:00.000 --> 00:02.000] text" / "0.00-2.00: text".
fn parse_timed_lines(s: &str) -> Option<Vec<Segment>> {
    let mut segs = Vec::new();
    let mut pending: Option<(f64, f64)> = None;
    let mut buf: Vec<String> = Vec::new();
    let flush = |segs: &mut Vec<Segment>, pending: &mut Option<(f64, f64)>, buf: &mut Vec<String>| {
        if let Some((a, b)) = pending.take() {
            let text = buf.join(" ").trim().to_string();
            if !text.is_empty() {
                segs.push(Segment { start: a, end: b, text });
            }
        }
        buf.clear();
    };
    for raw in s.lines() {
        let line = raw.trim();
        if line.starts_with("[info]") {
            continue;
        }
        // try to detect "<ts> --> <ts>" or "<ts> - <ts>" at the start, optional brackets
        let l = line.trim_start_matches('[');
        let arrow = l.find("-->").map(|i| (i, 3)).or_else(|| {
            if line.starts_with('[') { l.find(" - ").map(|i| (i, 3)) } else { None }
        });
        if let Some((i, w)) = arrow {
            let a = parse_ts(&l[..i]);
            let rest = &l[i + w..];
            let (b_str, text) = match rest.find(']') {
                Some(j) => (&rest[..j], rest[j + 1..].trim()),
                None => {
                    let rs = rest.trim_start();
                    let j = rs.find(char::is_whitespace).unwrap_or(rs.len());
                    (&rs[..j], rs[j..].trim())
                }
            };
            if let (Some(a), Some(b)) = (a, parse_ts(b_str.trim_end_matches(':'))) {
                flush(&mut segs, &mut pending, &mut buf);
                pending = Some((a, b));
                let text = text.trim_start_matches(':').trim();
                if !text.is_empty() {
                    buf.push(text.to_string());
                }
                continue;
            }
        }
        if line.is_empty() {
            flush(&mut segs, &mut pending, &mut buf);
            continue;
        }
        if pending.is_some() {
            // SRT index lines of the *next* block are numeric and come after a blank line,
            // so any non-empty line here belongs to the current block.
            buf.push(line.to_string());
        }
    }
    flush(&mut segs, &mut pending, &mut buf);
    (!segs.is_empty()).then_some(segs)
}

fn parse_output(stdout: &str, duration: f64) -> (Vec<Segment>, &'static str) {
    if let Some(s) = parse_json(stdout, duration) {
        return (s, "json");
    }
    if let Some(s) = parse_timed_lines(stdout) {
        return (s, "timestamps");
    }
    let text: Vec<&str> = stdout
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with("[info]"))
        .collect();
    (
        vec![Segment { start: 0.0, end: duration, text: text.join(" ") }],
        "plain text (inga segment-tidsstämplar från backend)",
    )
}

// ---------------------------------------------------------------- jobs

#[derive(Clone, Debug, Serialize)]
struct Job {
    id: String,
    filename: String,
    model: String,
    status: String, // queued | converting | running | done | error
    backend: Option<String>,
    backend_cmd: Option<String>,
    output_format: Option<String>,
    progress: String,
    log_tail: Vec<String>,
    error: Option<String>,
    audio_duration: Option<f64>,
    created: u64,
    elapsed_s: f64,
    segments: Vec<Segment>,
    text: Option<String>,
    #[serde(skip)]
    started: Option<Instant>,
}

struct AppState {
    cfg: Config,
    prata: Mutex<PrataInfo>,
    jobs: Mutex<HashMap<String, Job>>,
    gate: Semaphore, // one transcription at a time (memory!)
}

type St = Arc<AppState>;

async fn update(st: &St, id: &str, f: impl FnOnce(&mut Job)) {
    if let Some(j) = st.jobs.lock().await.get_mut(id) {
        f(j);
        if let Some(t) = j.started {
            j.elapsed_s = t.elapsed().as_secs_f64();
        }
    }
}

async fn ffmpeg_to_wav(input: &Path, out: &Path) -> Result<()> {
    let o = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(input)
        .args(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])
        .arg(out)
        .output()
        .await
        .context("kunde inte starta ffmpeg (är det installerat?)")?;
    if !o.status.success() {
        bail!("ffmpeg kunde inte avkoda filen: {}", String::from_utf8_lossy(&o.stderr).trim());
    }
    Ok(())
}

async fn wav_duration(p: &Path) -> f64 {
    // 16 kHz mono s16le, 44-byte header (approximately correct for ffmpeg output)
    tokio::fs::metadata(p)
        .await
        .map(|m| (m.len().saturating_sub(44)) as f64 / 32000.0)
        .unwrap_or(0.0)
}

/// Run a backend command, streaming stderr into the job's progress/log.
async fn run_cmd(st: &St, id: &str, mut cmd: Command) -> Result<String> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    let mut child = cmd.spawn().context("kunde inte starta backend")?;
    let mut stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let out_task = tokio::spawn(async move {
        let mut s = Vec::new();
        let _ = stdout.read_to_end(&mut s).await;
        String::from_utf8_lossy(&s).to_string()
    });
    let st2 = st.clone();
    let id2 = id.to_string();
    let err_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut all = String::new();
        let mut buf = Vec::new();
        // split on \n and \r so tqdm-style progress bars update live
        loop {
            buf.clear();
            let mut byte = [0u8; 1];
            let mut eof = false;
            loop {
                match reader.read(&mut byte).await {
                    Ok(0) | Err(_) => {
                        eof = true;
                        break;
                    }
                    Ok(_) if byte[0] == b'\n' || byte[0] == b'\r' => break,
                    Ok(_) => buf.push(byte[0]),
                }
            }
            let line = String::from_utf8_lossy(&buf).trim().to_string();
            if !line.is_empty() {
                all.push_str(&line);
                all.push('\n');
                update(&st2, &id2, |j| {
                    j.progress = line.chars().take(200).collect();
                    j.log_tail.push(line.chars().take(300).collect());
                    let n = j.log_tail.len();
                    if n > 30 {
                        j.log_tail.drain(..n - 30);
                    }
                })
                .await;
            }
            if eof {
                break;
            }
        }
        all
    });
    let status = child.wait().await?;
    let out = out_task.await.unwrap_or_default();
    let err = err_task.await.unwrap_or_default();
    if !status.success() {
        let tail: Vec<&str> = err.lines().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect();
        bail!("backend avslutades med {status}:\n{}", tail.join("\n"));
    }
    Ok(out)
}

fn cmd_string(prog: &str, args: &[String]) -> String {
    std::iter::once(prog.to_string()).chain(args.iter().cloned()).collect::<Vec<_>>().join(" ")
}

async fn run_prata(st: &St, id: &str, wav: &Path, model: &str, dur: f64) -> Result<(Vec<Segment>, &'static str)> {
    let info = st.prata.lock().await.clone();
    let bin = info.path.clone().ok_or_else(|| anyhow!("prata saknas"))?;
    let out_json = wav.with_extension("prata.json");
    let args = prata_args(&st.cfg, &info, wav, model, &out_json);
    update(st, id, |j| {
        j.backend = Some("prata (Rust/Candle)".into());
        j.backend_cmd = Some(cmd_string(&bin, &args));
        j.status = "running".into();
        j.progress = "Startar prata …".into();
    })
    .await;
    let mut c = Command::new(&bin);
    c.args(&args);
    let stdout = run_cmd(st, id, c).await?;
    // Prefer an output file if the template wrote one.
    let src = match tokio::fs::read_to_string(&out_json).await {
        Ok(s) if !s.trim().is_empty() => s,
        _ => stdout,
    };
    let r = parse_output(&src, dur);
    if r.0.iter().all(|s| s.text.trim().is_empty()) {
        bail!("prata gav ingen text på stdout");
    }
    Ok(r)
}

async fn run_python(st: &St, id: &str, wav: &Path, model: &str, dur: f64) -> Result<(Vec<Segment>, &'static str)> {
    let script = st
        .cfg
        .python_script
        .clone()
        .ok_or_else(|| anyhow!("transcribe.py hittades inte (sätt PRATA_PYTHON_SCRIPT)"))?;
    let args: Vec<String> = vec![
        script.display().to_string(),
        wav.display().to_string(),
        "--model".into(),
        model.into(),
        "--timestamps".into(),
    ];
    update(st, id, |j| {
        j.backend = Some("python (transformers, transcribe.py)".into());
        j.backend_cmd = Some(cmd_string(&st.cfg.python, &args));
        j.status = "running".into();
        j.progress = "Startar Python-backend (laddar modell) …".into();
    })
    .await;
    let mut c = Command::new(&st.cfg.python);
    c.args(&args).env("PYTHONUNBUFFERED", "1");
    let stdout = run_cmd(st, id, c).await?;
    Ok(parse_output(&stdout, dur))
}

async fn process_job(st: St, id: String, input: PathBuf, model: String) {
    let res: Result<()> = async {
        let _permit = st.gate.acquire().await?;
        update(&st, &id, |j| {
            j.status = "converting".into();
            j.progress = "Konverterar ljud med ffmpeg (16 kHz mono WAV) …".into();
            j.started = Some(Instant::now());
        })
        .await;
        let wav = input.with_extension("16k.wav");
        ffmpeg_to_wav(&input, &wav).await?;
        let dur = wav_duration(&wav).await;
        update(&st, &id, |j| j.audio_duration = Some(dur)).await;

        let want = st.cfg.backend.as_str();
        let prata_ok = st.prata.lock().await.works;
        let result = match want {
            "prata" => run_prata(&st, &id, &wav, &model, dur).await,
            "python" => run_python(&st, &id, &wav, &model, dur).await,
            _ if prata_ok => match run_prata(&st, &id, &wav, &model, dur).await {
                Ok(r) => Ok(r),
                Err(e) => {
                    let msg = format!("prata misslyckades, faller tillbaka på Python: {e}");
                    eprintln!("[job {id}] {msg}");
                    update(&st, &id, |j| j.log_tail.push(msg)).await;
                    run_python(&st, &id, &wav, &model, dur).await.map(|(s, f)| (s, f))
                }
            },
            _ => run_python(&st, &id, &wav, &model, dur).await,
        };
        let (segs, fmt) = result?;
        let text = to_txt(&segs, false);
        update(&st, &id, |j| {
            j.segments = segs;
            j.text = Some(text);
            j.output_format = Some(fmt.into());
            j.status = "done".into();
            j.progress = "Klar".into();
        })
        .await;
        let _ = tokio::fs::remove_file(&wav).await;
        Ok(())
    }
    .await;
    if let Err(e) = res {
        eprintln!("[job {id}] error: {e:#}");
        update(&st, &id, |j| {
            j.status = "error".into();
            j.error = Some(format!("{e:#}"));
            j.progress = "Fel".into();
        })
        .await;
    } else {
        let j = st.jobs.lock().await.get(&id).cloned();
        if let Some(j) = j {
            eprintln!(
                "[job {id}] done backend={:?} segments={} elapsed={:.1}s",
                j.backend, j.segments.len(), j.elapsed_s
            );
        }
    }
    let _ = tokio::fs::remove_file(&input).await;
}

// ---------------------------------------------------------------- handlers

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(serde_json::json!({ "error": msg.into() }))).into_response()
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn info(State(st): State<St>) -> Json<serde_json::Value> {
    // re-probe so a freshly built prata is picked up without restarting
    let mut cfg = st.cfg.clone();
    if cfg.prata_bin.is_none() {
        cfg = Config { prata_bin: Config::from_env().prata_bin, ..cfg };
    }
    let k = probe_prata(&cfg).await;
    *st.prata.lock().await = k.clone();
    let active = match st.cfg.backend.as_str() {
        "prata" => "prata",
        "python" => "python",
        _ if k.works => "prata",
        _ => "python",
    };
    Json(serde_json::json!({
        "backend_mode": st.cfg.backend,
        "active_backend": active,
        "prata": k,
        "python": {
            "interpreter": st.cfg.python,
            "script": st.cfg.python_script.as_ref().map(|p| p.display().to_string()),
        },
        "default_model": st.cfg.default_model,
    }))
}

async fn create_job(State(st): State<St>, mut mp: Multipart) -> Response {
    let mut file: Option<(String, Vec<u8>)> = None;
    let mut model = st.cfg.default_model.clone();
    loop {
        match mp.next_field().await {
            Ok(Some(f)) => {
                let name = f.name().unwrap_or("").to_string();
                if name == "file" || name == "audio" {
                    let fname = f.file_name().unwrap_or("audio").to_string();
                    match f.bytes().await {
                        Ok(b) => file = Some((fname, b.to_vec())),
                        Err(e) => return err(StatusCode::BAD_REQUEST, format!("uppladdning misslyckades: {e}")),
                    }
                } else if name == "model" {
                    if let Ok(t) = f.text().await {
                        let t = t.trim().to_string();
                        if !t.is_empty() {
                            model = t;
                        }
                    }
                }
            }
            Ok(None) => break,
            Err(e) => return err(StatusCode::BAD_REQUEST, format!("ogiltig multipart: {e}")),
        }
    }
    let Some((fname, bytes)) = file else {
        return err(StatusCode::BAD_REQUEST, "ingen fil (fält 'file')");
    };
    if bytes.is_empty() {
        return err(StatusCode::BAD_REQUEST, "filen är tom");
    }
    if !model.chars().all(|c| c.is_ascii_alphanumeric() || "-_./".contains(c)) {
        return err(StatusCode::BAD_REQUEST, "ogiltigt modellnamn");
    }
    let id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
    let ext: String = Path::new(&fname)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .take(8)
        .collect();
    let input = st.cfg.work_dir.join(format!("{id}.{ext}"));
    if let Err(e) = tokio::fs::write(&input, &bytes).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, format!("kunde inte spara filen: {e}"));
    }
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let job = Job {
        id: id.clone(),
        filename: fname.clone(),
        model: model.clone(),
        status: "queued".into(),
        backend: None,
        backend_cmd: None,
        output_format: None,
        progress: "I kö …".into(),
        log_tail: vec![],
        error: None,
        audio_duration: None,
        created,
        elapsed_s: 0.0,
        segments: vec![],
        text: None,
        started: None,
    };
    st.jobs.lock().await.insert(id.clone(), job);
    eprintln!("[job {id}] queued file={fname:?} bytes={} model={model}", bytes.len());
    tokio::spawn(process_job(st.clone(), id.clone(), input, model));
    (StatusCode::ACCEPTED, Json(serde_json::json!({ "id": id, "status_url": format!("/api/jobs/{id}") })))
        .into_response()
}

async fn get_job(State(st): State<St>, AxPath(id): AxPath<String>) -> Response {
    let mut jobs = st.jobs.lock().await;
    match jobs.get_mut(&id) {
        Some(j) => {
            if let (Some(t), false) = (j.started, matches!(j.status.as_str(), "done" | "error")) {
                j.elapsed_s = t.elapsed().as_secs_f64();
            }
            Json(j.clone()).into_response()
        }
        None => err(StatusCode::NOT_FOUND, "okänt jobb"),
    }
}

async fn download(st: &St, id: &str, kind: &str) -> Response {
    let jobs = st.jobs.lock().await;
    let Some(j) = jobs.get(id) else { return err(StatusCode::NOT_FOUND, "okänt jobb") };
    if j.status != "done" {
        return err(StatusCode::CONFLICT, "jobbet är inte klart");
    }
    let stem = Path::new(&j.filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("transkript")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect::<String>();
    let utf8_stem = Path::new(&j.filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("transkript")
        .to_string();
    let (body, ctype, ext) = match kind {
        "srt" => (to_srt(&j.segments), "application/x-subrip; charset=utf-8", "srt"),
        "json" => (
            serde_json::to_string_pretty(&serde_json::json!({"segments": j.segments, "text": j.text})).unwrap(),
            "application/json",
            "json",
        ),
        "txt-ts" => (to_txt(&j.segments, true), "text/plain; charset=utf-8", "tider.txt"),
        _ => (to_txt(&j.segments, false), "text/plain; charset=utf-8", "txt"),
    };
    (
        [
            (header::CONTENT_TYPE, ctype.to_string()),
            (header::CONTENT_DISPOSITION, {
                let name = if kind == "txt-ts" { format!("-{ext}") } else { format!(".{ext}") };
                // ASCII fallback + RFC 5987 UTF-8 name (keeps å/ä/ö in the downloaded file name)
                format!("attachment; filename=\"{stem}{name}\"; filename*=UTF-8''{}{}", pct(&utf8_stem), pct(&name))
            }),
        ],
        body,
    )
        .into_response()
}

/// Percent-encode for RFC 5987 `filename*`.
fn pct(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

async fn dl_txt(State(st): State<St>, AxPath(id): AxPath<String>) -> Response {
    download(&st, &id, "txt").await
}
async fn dl_txt_ts(State(st): State<St>, AxPath(id): AxPath<String>) -> Response {
    download(&st, &id, "txt-ts").await
}
async fn dl_srt(State(st): State<St>, AxPath(id): AxPath<String>) -> Response {
    download(&st, &id, "srt").await
}
async fn dl_json(State(st): State<St>, AxPath(id): AxPath<String>) -> Response {
    download(&st, &id, "json").await
}

// ---------------------------------------------------------------- main

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Config::from_env();
    tokio::fs::create_dir_all(&cfg.work_dir).await?;
    let prata = probe_prata(&cfg).await;
    eprintln!("prata-web config: {cfg:?}");
    eprintln!("prata probe: path={:?} works={} ({})", prata.path, prata.works, prata.note);
    let limit = cfg.max_upload_mb * 1024 * 1024;
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let st: St = Arc::new(AppState {
        cfg,
        prata: Mutex::new(prata),
        jobs: Mutex::new(HashMap::new()),
        gate: Semaphore::new(1),
    });
    let app = Router::new()
        .route("/", get(index))
        .route("/api/info", get(info))
        .route("/api/jobs", post(create_job))
        .route("/api/jobs/{id}", get(get_job))
        .route("/api/jobs/{id}/txt", get(dl_txt))
        .route("/api/jobs/{id}/txt-ts", get(dl_txt_ts))
        .route("/api/jobs/{id}/srt", get(dl_srt))
        .route("/api/jobs/{id}/json", get(dl_json))
        .layer(DefaultBodyLimit::max(limit))
        .with_state(st);
    let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("bind {addr}"))?;
    eprintln!("prata-web listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn srt_fmt() {
        assert_eq!(fmt_ts(3725.5, ','), "01:02:05,500");
        assert_eq!(fmt_ts(0.0, ','), "00:00:00,000");
    }
    #[test]
    fn parse_srt() {
        let s = "1\n00:00:00,000 --> 00:00:01,500\nHej där.\n\n2\n00:00:01,500 --> 00:00:04,000\nHur mår du?\nBra.\n";
        let (segs, f) = parse_output(s, 4.0);
        assert_eq!(f, "timestamps");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1].text, "Hur mår du? Bra.");
        assert!((segs[1].start - 1.5).abs() < 1e-9);
    }
    #[test]
    fn parse_bracket_lines() {
        let s = "[00:00.000 --> 00:02.500] Hej\n[00:02.500 --> 00:05.000] världen\n";
        let (segs, _) = parse_output(s, 5.0);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1].end, 5.0);
    }
    #[test]
    fn parse_json_shapes() {
        let (s, f) = parse_output(r#"{"segments":[{"start":0.0,"end":1.0,"text":"a"}]}"#, 1.0);
        assert_eq!((s.len(), f), (1, "json"));
        let (s, _) = parse_output(r#"[{"start":"00:00:01,000","end":2,"text":"b"}]"#, 2.0);
        assert_eq!(s[0].start, 1.0);
    }
    #[test]
    fn plain() {
        let (s, _) = parse_output("Hej hej\n", 3.0);
        assert_eq!(s[0].text, "Hej hej");
        assert_eq!(s[0].end, 3.0);
    }
}
