//! prata-web: local web UI for Prata – Swedish speech-to-text (KBLab kb-whisper models).
//!
//! Backends (chosen per job):
//!   * `prata`  – the Rust/Candle CLI (next to this binary, or $PRATA_BIN, or on $PATH)
//!   * `python` – the Python fallback transcribe.py ($PRATA_PYTHON_SCRIPT)
//!
//! Job model: POST /api/jobs (multipart) -> {id}; GET /api/jobs/{id} polls status.
//! Every finished job is saved as a note (see `notes.rs`): /api/notes…

mod fetch;
mod klang;
mod notes;

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, bail, Context, Result};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Path as AxPath, Query, Request, State},
    http::{header, HeaderValue, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use notes::{Note, Segment, Store};
use serde::{Deserialize, Serialize};
use tower::ServiceExt;
use tokio::{
    io::{AsyncReadExt, BufReader},
    process::Command,
    sync::{Mutex, Semaphore},
};

const INDEX_HTML: &str = include_str!("index.html");
const MANIFEST: &str = include_str!("assets/manifest.webmanifest");
const ICON_SVG: &str = include_str!("assets/icon.svg");
const ICON_180: &[u8] = include_bytes!("assets/apple-touch-icon.png");
const ICON_192: &[u8] = include_bytes!("assets/icon-192.png");
const ICON_512: &[u8] = include_bytes!("assets/icon-512.png");
const ICON_MASKABLE: &[u8] = include_bytes!("assets/icon-maskable-512.png");

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
    notes_dir: PathBuf,
    max_upload_mb: usize,
    /// Link downloads: size limit (MB), duration limit (s), overall timeout (s)
    url_max_mb: u64,
    url_max_duration: u64,
    url_timeout: u64,
    /// Explicit yt-dlp path (else found on $PATH)
    ytdlp: Option<PathBuf>,
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
            host: env("WEB_HOST").or_else(|| env("HOST")).unwrap_or_else(|| "127.0.0.1".into()),
            backend: env("BACKEND").unwrap_or_else(|| "auto".into()).to_lowercase(),
            prata_bin,
            prata_args: env("ARGS"),
            python: env("PYTHON").unwrap_or_else(|| "python3".into()),
            python_script,
            default_model: env("MODEL").unwrap_or_else(|| "small".into()),
            work_dir: env("WORK_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir().join("prata-web")),
            notes_dir: env("NOTES_DIR").map(PathBuf::from).unwrap_or_else(default_notes_dir),
            max_upload_mb: env("MAX_UPLOAD_MB").and_then(|v| v.parse().ok()).unwrap_or(1024),
            url_max_mb: env("URL_MAX_MB").and_then(|v| v.parse().ok()).filter(|v| *v > 0).unwrap_or(500),
            url_max_duration: env("URL_MAX_DURATION").and_then(|v| fetch::parse_duration(&v)).unwrap_or(3 * 3600),
            url_timeout: env("URL_TIMEOUT").and_then(|v| fetch::parse_duration(&v)).unwrap_or(15 * 60),
            ytdlp: env("YTDLP").map(PathBuf::from),
        }
    }
}

fn default_notes_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".prata")
        .join("notes")
}

const USAGE: &str = "prata-web – lokal webbapp för Prata (svensk tal till text)

usage: prata-web [--host ADDR] [--port PORT] [--notes-dir DIR]
                 [--url-max-mb MB] [--url-max-duration DUR] [--url-timeout DUR] [--yt-dlp PATH]

  --host ADDR       listen address (default 127.0.0.1; env PRATA_HOST / PRATA_WEB_HOST)
  --port PORT       listen port (default 8795; env PRATA_WEB_PORT)
  --notes-dir DIR   where saved notes are kept (default ~/.prata/notes; env PRATA_NOTES_DIR)
  --url-max-mb MB   size limit for links (default 500; env PRATA_URL_MAX_MB)
  --url-max-duration DUR
                    duration limit for links, e.g. 3h, 90m, 600 (default 3h; env PRATA_URL_MAX_DURATION)
  --url-timeout DUR download timeout for links (default 15m; env PRATA_URL_TIMEOUT)
  --yt-dlp PATH     yt-dlp binary (default: found on PATH; env PRATA_YTDLP)

All other settings are environment variables, see the README.";

/// Command-line flags override the environment.
fn apply_args(cfg: &mut Config, args: &[String]) -> Result<()> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_string())),
            _ => (a.as_str(), None),
        };
        let mut val = || inline.clone().or_else(|| it.next().cloned()).ok_or_else(|| anyhow!("{flag} needs a value"));
        match flag {
            "--host" => cfg.host = val()?,
            "--port" => cfg.port = val()?.parse().context("--port")?,
            "--notes-dir" => cfg.notes_dir = PathBuf::from(val()?),
            "--url-max-mb" => cfg.url_max_mb = val()?.parse().ok().filter(|v| *v > 0).context("--url-max-mb")?,
            "--url-max-duration" => cfg.url_max_duration = fetch::parse_duration(&val()?).context("--url-max-duration (t.ex. 3h, 90m)")?,
            "--url-timeout" => cfg.url_timeout = fetch::parse_duration(&val()?).context("--url-timeout (t.ex. 15m)")?,
            "--yt-dlp" => cfg.ytdlp = Some(PathBuf::from(val()?)),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => bail!("unknown argument {a}\n\n{USAGE}"),
        }
    }
    Ok(())
}

fn is_loopback(host: &str) -> bool {
    let h = host.trim_start_matches('[').trim_end_matches(']');
    h == "localhost" || h.parse::<std::net::IpAddr>().map(|ip| ip.is_loopback()).unwrap_or(false)
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
    /// Saved note (set when the job is done)
    note_id: Option<String>,
    /// Link jobs: the pasted URL, the media title (yt-dlp) and download progress 0..100
    source_url: Option<String>,
    media_title: Option<String>,
    download_pct: Option<f64>,
    /// How the link was fetched: "http" or "yt-dlp"
    fetched_via: Option<String>,
    /// Error message meant to be shown as is (Swedish)
    error_user: Option<String>,
    #[serde(skip)]
    started: Option<Instant>,
    #[serde(skip)]
    max_duration: Option<f64>,
}

impl Job {
    fn new(id: &str, filename: &str, model: &str, status: &str, progress: &str) -> Job {
        Job {
            id: id.into(),
            filename: filename.into(),
            model: model.into(),
            status: status.into(),
            backend: None,
            backend_cmd: None,
            output_format: None,
            progress: progress.into(),
            log_tail: vec![],
            error: None,
            audio_duration: None,
            created: SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
            elapsed_s: 0.0,
            segments: vec![],
            text: None,
            note_id: None,
            source_url: None,
            media_title: None,
            download_pct: None,
            fetched_via: None,
            error_user: None,
            started: None,
            max_duration: None,
        }
    }
}

struct AppState {
    cfg: Config,
    prata: Mutex<PrataInfo>,
    jobs: Mutex<HashMap<String, Job>>,
    gate: Semaphore, // one transcription at a time (memory!)
    notes: Store,
    ytdlp: Mutex<Option<fetch::YtDlp>>,
    net: fetch::NetPolicy,
    /// Klang import; `None` unless KLANG_API_KEY is set
    klang: Option<Arc<klang::Klang>>,
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
        let max_dur = st.jobs.lock().await.get(&id).and_then(|j| j.max_duration);
        if let Some(max) = max_dur {
            if dur > max + 1.0 {
                let _ = tokio::fs::remove_file(&wav).await;
                return Err(fetch::FetchError::TooLong { max_s: max as u64 }.into());
            }
        }

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
        let _ = tokio::fs::remove_file(&wav).await;
        // Save as a note (keeps the original audio for playback).
        let job = st.jobs.lock().await.get(&id).cloned();
        let note_id = match job {
            Some(j) => {
                let created = j.created as i64;
                let title = j
                    .media_title
                    .as_deref()
                    .and_then(notes::clean_title)
                    .unwrap_or_else(|| notes::default_title(created, &text));
                let note = Note {
                    id: id.clone(),
                    title,
                    source_url: j.source_url.clone(),
                    created,
                    model: j.model.clone(),
                    audio_duration: dur,
                    filename: j.filename.clone(),
                    audio: None,
                    audio_bytes: 0,
                    backend: j.backend.clone(),
                    segments: segs.clone(),
                    ..Default::default()
                };
                let st2 = st.clone();
                let input2 = input.clone();
                match tokio::task::spawn_blocking(move || st2.notes.create(note, Some(&input2))).await {
                    Ok(Ok(n)) => Some(n.id),
                    Ok(Err(e)) => {
                        eprintln!("[job {id}] could not save note: {e:#}");
                        None
                    }
                    Err(e) => {
                        eprintln!("[job {id}] could not save note: {e}");
                        None
                    }
                }
            }
            None => None,
        };
        update(&st, &id, |j| {
            j.segments = segs;
            j.text = Some(text);
            j.output_format = Some(fmt.into());
            j.note_id = note_id;
            j.status = "done".into();
            j.progress = "Klar".into();
        })
        .await;
        Ok(())
    }
    .await;
    if let Err(e) = res {
        eprintln!("[job {id}] error: {e:#}");
        let user = e.downcast_ref::<fetch::FetchError>().map(|f| f.message());
        update(&st, &id, |j| {
            j.status = "error".into();
            j.error = Some(format!("{e:#}"));
            j.error_user = user;
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
    // probe yt-dlp again while it is missing, so `brew install yt-dlp` works without a restart
    let mut y = st.ytdlp.lock().await.clone();
    if y.is_none() {
        y = fetch::probe_ytdlp(st.cfg.ytdlp.clone()).await;
        *st.ytdlp.lock().await = y.clone();
    }
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
        "url": url_info(&st.cfg, y.as_ref()),
        "klang_enabled": st.klang.is_some(),
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
    let ext = notes::clean_ext(&fname);
    let input = st.cfg.work_dir.join(format!("{id}.{ext}"));
    if let Err(e) = tokio::fs::write(&input, &bytes).await {
        return err(StatusCode::INTERNAL_SERVER_ERROR, format!("kunde inte spara filen: {e}"));
    }
    let job = Job::new(&id, &fname, &model, "queued", "I kö …");
    st.jobs.lock().await.insert(id.clone(), job);
    eprintln!("[job {id}] queued file={fname:?} bytes={} model={model}", bytes.len());
    tokio::spawn(process_job(st.clone(), id.clone(), input, model));
    (StatusCode::ACCEPTED, Json(serde_json::json!({ "id": id, "status_url": format!("/api/jobs/{id}") })))
        .into_response()
}

// ---------------------------------------------------------------- link jobs

#[derive(Deserialize)]
struct UrlReq {
    url: Option<String>,
    model: Option<String>,
}

fn url_limits(cfg: &Config) -> fetch::Limits {
    fetch::Limits {
        max_bytes: cfg.url_max_mb * 1024 * 1024,
        max_duration_s: cfg.url_max_duration,
        timeout: Duration::from_secs(cfg.url_timeout),
    }
}

/// POST /api/jobs/url  {"url": "...", "model": "small"}
async fn create_url_job(State(st): State<St>, body: Option<Json<UrlReq>>) -> Response {
    let Some(Json(req)) = body else { return err(StatusCode::BAD_REQUEST, fetch::FetchError::InvalidUrl.message()) };
    let model = req.model.map(|m| m.trim().to_string()).filter(|m| !m.is_empty()).unwrap_or_else(|| st.cfg.default_model.clone());
    if !model.chars().all(|c| c.is_ascii_alphanumeric() || "-_./".contains(c)) {
        return err(StatusCode::BAD_REQUEST, "ogiltigt modellnamn");
    }
    let url = match fetch::parse_url(req.url.as_deref().unwrap_or("")) {
        Ok(u) => u,
        Err(e) => return err(StatusCode::BAD_REQUEST, e.message()),
    };
    // Early, friendly rejection of local/private targets (checked again at connect time).
    if let Err(e) = fetch::check_host(&url, st.net).await {
        return err(StatusCode::BAD_REQUEST, e.message());
    }
    let id = uuid::Uuid::new_v4().simple().to_string()[..12].to_string();
    let label = url.host_str().unwrap_or("länk").to_string();
    let mut job = Job::new(&id, &label, &model, "downloading", "Laddar ner …");
    job.source_url = Some(url.to_string());
    job.max_duration = Some(st.cfg.url_max_duration as f64);
    job.started = Some(Instant::now());
    st.jobs.lock().await.insert(id.clone(), job);
    eprintln!("[job {id}] queued url host={label:?} model={model}");
    tokio::spawn(url_job(st.clone(), id.clone(), url, model));
    (StatusCode::ACCEPTED, Json(serde_json::json!({ "id": id, "status_url": format!("/api/jobs/{id}") })))
        .into_response()
}

async fn url_job(st: St, id: String, url: url::Url, model: String) {
    let dir = st.cfg.work_dir.join(format!("{id}-dl"));
    let limits = url_limits(&st.cfg);
    let res = tokio::time::timeout(limits.timeout, fetch_url(&st, &id, &url, &dir, &limits)).await;
    let res = match res {
        Ok(r) => r,
        Err(_) => Err(fetch::FetchError::Timeout { secs: limits.timeout.as_secs() }),
    };
    match res {
        Ok(d) => {
            let ext = notes::clean_ext(&d.path.to_string_lossy());
            let input = st.cfg.work_dir.join(format!("{id}.{ext}"));
            let moved = tokio::fs::rename(&d.path, &input).await;
            let _ = tokio::fs::remove_dir_all(&dir).await;
            if let Err(e) = moved {
                return url_failed(&st, &id, fetch::FetchError::Other(e.to_string())).await;
            }
            let bytes = tokio::fs::metadata(&input).await.map(|m| m.len()).unwrap_or(0);
            eprintln!("[job {id}] downloaded via {} bytes={bytes}", d.via);
            update(&st, &id, |j| {
                j.filename = d.filename.clone();
                j.media_title = d.title.clone();
                j.fetched_via = Some(d.via.into());
                j.download_pct = Some(100.0);
                j.status = "queued".into();
                j.progress = "I kö …".into();
            })
            .await;
            process_job(st, id, input, model).await;
        }
        Err(e) => {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            url_failed(&st, &id, e).await;
        }
    }
}

async fn url_failed(st: &St, id: &str, e: fetch::FetchError) {
    eprintln!("[job {id}] link error: {e:?}");
    update(st, id, |j| {
        j.status = "error".into();
        j.error = Some(e.message());
        j.error_user = Some(e.message());
        j.progress = "Fel".into();
    })
    .await;
}

async fn fetch_url(st: &St, id: &str, url: &url::Url, dir: &Path, limits: &fetch::Limits) -> Result<fetch::Downloaded, fetch::FetchError> {
    tokio::fs::create_dir_all(dir).await.map_err(|e| fetch::FetchError::Other(e.to_string()))?;
    // progress updates from sync callbacks: keep the latest value and publish it from here
    let pct = Arc::new(std::sync::Mutex::new(None::<f64>));
    let publisher = {
        let (st, id, pct) = (st.clone(), id.to_string(), pct.clone());
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(400)).await;
                let p = *pct.lock().unwrap();
                update(&st, &id, |j| {
                    if j.status == "downloading" {
                        j.download_pct = p;
                        j.progress = match p {
                            Some(p) => format!("Laddar ner … {p:.0} %"),
                            None => "Laddar ner …".into(),
                        };
                    }
                })
                .await;
            }
        })
    };
    let r = async {
        let p1 = pct.clone();
        let direct = fetch::direct_download(url, dir, limits, st.net, move |done, total| {
            *p1.lock().unwrap() = total.filter(|t| *t > 0).map(|t| (done as f64 / t as f64 * 100.0).min(100.0));
        })
        .await?;
        let (status, html) = match direct {
            fetch::Direct::Media(d) => return Ok(d),
            fetch::Direct::NotMedia { status, html } => (status, html),
        };
        let y = st.ytdlp.lock().await.clone();
        let Some(y) = y else {
            return Err(match (status, html) {
                (Some(code), _) => fetch::FetchError::Http(code),
                (None, true) => fetch::FetchError::ToolMissing,
                (None, false) => fetch::FetchError::NotMedia,
            });
        };
        update(st, id, |j| j.log_tail.push(format!("hämtar med yt-dlp {}", y.version))).await;
        let p2 = pct.clone();
        fetch::ytdlp_download(&y, url, dir, limits, move |p| *p2.lock().unwrap() = Some(p)).await
    }
    .await;
    publisher.abort();
    r
}

async fn health(State(st): State<St>) -> Json<serde_json::Value> {
    let y = st.ytdlp.lock().await.clone();
    Json(serde_json::json!({
        "ok": true,
        "prata": st.prata.lock().await.works,
        "ffmpeg": which("ffmpeg").is_some(),
        "yt_dlp": y.as_ref().map(|y| serde_json::json!({"available": true, "version": y.version}))
            .unwrap_or(serde_json::json!({"available": false})),
    }))
}

fn url_info(cfg: &Config, y: Option<&fetch::YtDlp>) -> serde_json::Value {
    serde_json::json!({
        "yt_dlp": y.is_some(),
        "yt_dlp_version": y.map(|y| y.version.clone()),
        "max_mb": cfg.url_max_mb,
        "max_duration_s": cfg.url_max_duration,
        "timeout_s": cfg.url_timeout,
    })
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
    let stem = Path::new(&j.filename).file_stem().and_then(|s| s.to_str()).unwrap_or("transkript").to_string();
    let json = serde_json::json!({"segments": j.segments, "text": j.text});
    transcript_file(&stem, &j.segments, json, kind)
}

/// A transcript download (`txt`, `txt-ts`, `srt`, `json`) named after `stem`.
fn transcript_file(stem: &str, segs: &[Segment], json: serde_json::Value, kind: &str) -> Response {
    // keep å/ä/ö etc. but no characters that are invalid in file names
    let utf8_stem: String = stem
        .chars()
        .map(|c| if "/\\:*?\"<>|".contains(c) || c.is_control() { '-' } else { c })
        .collect::<String>()
        .trim()
        .to_string();
    let utf8_stem = if utf8_stem.is_empty() { "transkript".to_string() } else { utf8_stem };
    let ascii: String = utf8_stem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    let (body, ctype, ext) = match kind {
        "srt" => (to_srt(segs), "application/x-subrip; charset=utf-8", "srt"),
        "json" => (serde_json::to_string_pretty(&json).unwrap(), "application/json", "json"),
        "txt-ts" => (to_txt(segs, true), "text/plain; charset=utf-8", "tider.txt"),
        _ => (to_txt(segs, false), "text/plain; charset=utf-8", "txt"),
    };
    let name = if kind == "txt-ts" { format!("-{ext}") } else { format!(".{ext}") };
    (
        [
            (header::CONTENT_TYPE, ctype.to_string()),
            // ASCII fallback + RFC 5987 UTF-8 name (keeps å/ä/ö in the downloaded file name)
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{ascii}{name}\"; filename*=UTF-8''{}{}", pct(&utf8_stem), pct(&name)),
            ),
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

// ---------------------------------------------------------------- notes API

#[derive(Deserialize)]
struct ListQ {
    q: Option<String>,
}

async fn list_notes(State(st): State<St>, Query(q): Query<ListQ>) -> Response {
    let notes = st.notes.list(q.q.as_deref());
    Json(serde_json::json!({ "notes": notes, "total": st.notes.len() })).into_response()
}

async fn get_note(State(st): State<St>, AxPath(id): AxPath<String>) -> Response {
    match st.notes.get(&id) {
        Some(n) => {
            let text = n.text();
            let mut v = serde_json::to_value(&n).unwrap();
            v["text"] = text.into();
            v["has_audio"] = n.audio.is_some().into();
            v["timed"] = n.segments.iter().any(|s| s.end > 0.0).into();
            Json(v).into_response()
        }
        None => err(StatusCode::NOT_FOUND, "okänd anteckning"),
    }
}

#[derive(Deserialize)]
struct Patch {
    title: Option<String>,
}

async fn patch_note(State(st): State<St>, AxPath(id): AxPath<String>, body: Option<Json<Patch>>) -> Response {
    let Some(Json(p)) = body else { return err(StatusCode::BAD_REQUEST, "förväntade JSON {\"title\": …}") };
    let Some(title) = p.title else { return err(StatusCode::BAD_REQUEST, "inget att ändra") };
    let st2 = st.clone();
    match tokio::task::spawn_blocking(move || st2.notes.rename(&id, &title)).await {
        Ok(Ok(Some(n))) => Json(serde_json::json!({ "id": n.id, "title": n.title })).into_response(),
        Ok(Ok(None)) => err(StatusCode::NOT_FOUND, "okänd anteckning"),
        Ok(Err(e)) if e.to_string().contains("empty title") => err(StatusCode::BAD_REQUEST, "titeln får inte vara tom"),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("kunde inte spara: {e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn delete_note(State(st): State<St>, AxPath(id): AxPath<String>) -> Response {
    let st2 = st.clone();
    match tokio::task::spawn_blocking(move || st2.notes.delete(&id)).await {
        Ok(Ok(true)) => StatusCode::NO_CONTENT.into_response(),
        Ok(Ok(false)) => err(StatusCode::NOT_FOUND, "okänd anteckning"),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("kunde inte radera: {e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// The note's original audio, with HTTP Range support (needed by iOS Safari).
async fn note_audio(State(st): State<St>, AxPath(id): AxPath<String>, req: Request) -> Response {
    let Some((path, mime)) = st.notes.audio_path(&id) else { return err(StatusCode::NOT_FOUND, "inget ljud") };
    let svc = tower_http::services::ServeFile::new_with_mime(path, &mime.parse().unwrap());
    match svc.oneshot(req).await {
        Ok(r) => {
            let mut r = r.map(Body::new);
            r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("private, max-age=3600"));
            r
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

async fn note_download(State(st): State<St>, AxPath((id, kind)): AxPath<(String, String)>) -> Response {
    if !matches!(kind.as_str(), "txt" | "txt-ts" | "srt" | "json") {
        return err(StatusCode::NOT_FOUND, "okänt format");
    }
    let Some(n) = st.notes.get(&id) else { return err(StatusCode::NOT_FOUND, "okänd anteckning") };
    let json = serde_json::json!({
        "id": n.id, "title": n.title, "created": n.created, "model": n.model,
        "audio_duration": n.audio_duration, "filename": n.filename,
        "segments": n.segments, "text": n.text(),
    });
    transcript_file(&n.title, &n.segments, json, &kind)
}

// ---------------------------------------------------------------- Klang import

fn klang_state_json(s: &klang::State) -> serde_json::Value {
    serde_json::json!({ "enabled": true, "running": s.running, "started_at": s.started_at, "last": s.last })
}

/// Start a sync (or join the one already running) and return the state right away.
async fn klang_sync(State(st): State<St>) -> Response {
    let Some(k) = st.klang.clone() else { return err(StatusCode::NOT_FOUND, "Klang är inte aktiverat (sätt KLANG_API_KEY)") };
    if k.try_begin().await {
        let st2 = st.clone();
        let k2 = k.clone();
        tokio::spawn(async move {
            let k3 = k2.clone();
            let run = tokio::spawn(async move { klang::sync(&k3.client, &st2.notes).await });
            let report = match run.await {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[klang] sync task failed: {e}");
                    klang::Report { error: Some("Synken avbröts av ett internt fel.".into()), message: "Synken avbröts av ett internt fel.".into(), finished_at: chrono::Utc::now().timestamp(), ..Default::default() }
                }
            };
            eprintln!("[klang] {}", report.message);
            k2.end(report).await;
        });
    }
    (StatusCode::ACCEPTED, Json(klang_state_json(&k.state().await))).into_response()
}

async fn klang_status(State(st): State<St>) -> Response {
    match &st.klang {
        None => Json(serde_json::json!({ "enabled": false })).into_response(),
        Some(k) => Json(klang_state_json(&k.state().await)).into_response(),
    }
}

// ---------------------------------------------------------------- static assets

fn asset(ctype: &'static str, body: impl Into<Body>) -> Response {
    ([(header::CONTENT_TYPE, ctype), (header::CACHE_CONTROL, "public, max-age=86400")], body.into()).into_response()
}

async fn manifest() -> Response {
    asset("application/manifest+json", MANIFEST)
}
async fn icon_svg() -> Response {
    asset("image/svg+xml", ICON_SVG)
}
async fn icon_180() -> Response {
    asset("image/png", ICON_180)
}
async fn icon_192() -> Response {
    asset("image/png", ICON_192)
}
async fn icon_512() -> Response {
    asset("image/png", ICON_512)
}
async fn icon_maskable() -> Response {
    asset("image/png", ICON_MASKABLE)
}

// ---------------------------------------------------------------- main

fn app(st: St) -> Router {
    let limit = st.cfg.max_upload_mb * 1024 * 1024;
    Router::new()
        .route("/", get(index))
        .route("/manifest.webmanifest", get(manifest))
        .route("/icon.svg", get(icon_svg))
        .route("/favicon.svg", get(icon_svg))
        .route("/apple-touch-icon.png", get(icon_180))
        .route("/apple-touch-icon-precomposed.png", get(icon_180))
        .route("/icon-192.png", get(icon_192))
        .route("/icon-512.png", get(icon_512))
        .route("/icon-maskable-512.png", get(icon_maskable))
        .route("/api/info", get(info))
        .route("/api/health", get(health))
        .route("/api/jobs", post(create_job))
        .route("/api/jobs/url", post(create_url_job))
        .route("/api/jobs/{id}", get(get_job))
        .route("/api/jobs/{id}/txt", get(dl_txt))
        .route("/api/jobs/{id}/txt-ts", get(dl_txt_ts))
        .route("/api/jobs/{id}/srt", get(dl_srt))
        .route("/api/jobs/{id}/json", get(dl_json))
        .route("/api/klang/sync", get(klang_status).post(klang_sync))
        .route("/api/notes", get(list_notes))
        .route("/api/notes/{id}", get(get_note).patch(patch_note).delete(delete_note))
        .route("/api/notes/{id}/audio", get(note_audio))
        .route("/api/notes/{id}/{kind}", get(note_download))
        .layer(DefaultBodyLimit::max(limit))
        .with_state(st)
}

/// Address policy for link downloads. Always strict, except in a binary built with the
/// `insecure-test-loopback` feature *and* started with PRATA_INSECURE_ALLOW_LOOPBACK=1
/// (used only by the end-to-end tests against a local mock server).
fn net_policy() -> fetch::NetPolicy {
    #[cfg(feature = "insecure-test-loopback")]
    if std::env::var("PRATA_INSECURE_ALLOW_LOOPBACK").as_deref() == Ok("1") {
        eprintln!("WARNING: test build – link downloads may reach 127.0.0.1/::1 (PRATA_INSECURE_ALLOW_LOOPBACK=1). Never use this build in production.");
        return fetch::NetPolicy::allow_loopback_for_tests();
    }
    fetch::NetPolicy::strict()
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut cfg = Config::from_env();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = apply_args(&mut cfg, &args) {
        eprintln!("prata-web: {e}");
        std::process::exit(2);
    }
    tokio::fs::create_dir_all(&cfg.work_dir).await?;
    let store = Store::open(&cfg.notes_dir)?;
    let prata = probe_prata(&cfg).await;
    eprintln!("prata-web config: {cfg:?}");
    eprintln!("prata probe: path={:?} works={} ({})", prata.path, prata.works, prata.note);
    eprintln!("notes: {} saved in {}", store.len(), store.dir().display());
    let addr = if cfg.host.contains(':') && !cfg.host.starts_with('[') {
        format!("[{}]:{}", cfg.host, cfg.port)
    } else {
        format!("{}:{}", cfg.host, cfg.port)
    };
    if !is_loopback(&cfg.host) {
        eprintln!(
            "WARNING: listening on {addr}, not only on this computer. prata-web has no login: \
             anyone who can reach this address can record, read and delete notes. \
             For Tailscale, prefer the default 127.0.0.1 with `tailscale serve`."
        );
    }
    let klang = match klang::Client::from_env() {
        Ok(Some(c)) => {
            eprintln!("klang: import enabled ({})", c.base());
            Some(Arc::new(klang::Klang::new(c)))
        }
        Ok(None) => None,
        Err(e) => {
            eprintln!("prata-web: Klang: {e}");
            std::process::exit(2);
        }
    };
    let ytdlp = fetch::probe_ytdlp(cfg.ytdlp.clone()).await;
    match &ytdlp {
        Some(y) => eprintln!("yt-dlp: {} ({})", y.version, y.path.display()),
        None => eprintln!("yt-dlp: not found – links to web pages (YouTube …) need it: brew install yt-dlp; direct audio/video links still work"),
    }
    let st: St = Arc::new(AppState {
        cfg,
        prata: Mutex::new(prata),
        jobs: Mutex::new(HashMap::new()),
        gate: Semaphore::new(1),
        notes: store,
        ytdlp: Mutex::new(ytdlp),
        net: net_policy(),
        klang,
    });
    let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("bind {addr}"))?;
    eprintln!("prata-web listening on http://{addr}");
    axum::serve(listener, app(st)).await?;
    Ok(())
}

#[cfg(test)]
mod api_tests {
    use super::*;
    use http_body_util::BodyExt;

    fn state(dir: &Path) -> St {
        state_with(dir, fetch::NetPolicy::strict(), |_| {})
    }

    fn state_with(dir: &Path, net: fetch::NetPolicy, tweak: impl FnOnce(&mut Config)) -> St {
        let mut cfg = Config::from_env();
        tweak(&mut cfg);
        cfg.notes_dir = dir.join("notes");
        cfg.work_dir = dir.join("work");
        std::fs::create_dir_all(&cfg.work_dir).unwrap();
        Arc::new(AppState {
            notes: Store::open(&cfg.notes_dir).unwrap(),
            cfg,
            prata: Mutex::new(PrataInfo::default()),
            jobs: Mutex::new(HashMap::new()),
            gate: Semaphore::new(1),
            ytdlp: Mutex::new(None),
            net,
            klang: None,
        })
    }

    #[tokio::test]
    async fn klang_endpoints_single_flight_and_no_key_leak() {
        let tmp = tempfile::tempdir().unwrap();
        // disabled
        let app0 = app(state(tmp.path()));
        let (s, _, b) = call(&app0, "GET", "/api/info", None, None).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&b).unwrap()["klang_enabled"], false);
        let (s, _, _) = call(&app0, "POST", "/api/klang/sync", None, None).await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        let convs = vec![
            klang::mock::conv("conv_1", "Möte <script>alert(1)</script>", "ready", Some("2026-09-28T08:00:00Z"), "2026-09-28T09:00:00Z", "**Viktigt**", "[00:00:00] Ada: Hej."),
            klang::mock::conv("conv_2", "Samtal", "ready", None, "2026-09-29T09:00:00Z", "", "Ada: utan tider"),
        ];
        let (m, base) = klang::mock::start(klang::mock::Mock { convs: convs.into(), page_size: 1, ..Default::default() }).await;
        m.delay_ms.store(150, std::sync::atomic::Ordering::SeqCst);
        let tmp2 = tempfile::tempdir().unwrap();
        let st = state(tmp2.path());
        let st = Arc::new(AppState {
            klang: Some(Arc::new(klang::Klang::new(klang::mock::client(&base)))),
            cfg: st.cfg.clone(),
            notes: Store::open(&st.cfg.notes_dir).unwrap(),
            prata: Mutex::new(PrataInfo::default()),
            jobs: Mutex::new(HashMap::new()),
            gate: Semaphore::new(1),
            ytdlp: Mutex::new(None),
            net: fetch::NetPolicy::strict(),
        });
        let a = app(st);
        let mut all = Vec::new();
        let (s, _, b) = call(&a, "GET", "/api/info", None, None).await;
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&b).unwrap()["klang_enabled"], true);
        all.extend(b);
        // three concurrent POSTs → one sync
        let (r1, r2, r3) = tokio::join!(
            call(&a, "POST", "/api/klang/sync", None, None),
            call(&a, "POST", "/api/klang/sync", None, None),
            call(&a, "POST", "/api/klang/sync", None, None)
        );
        for r in [&r1, &r2, &r3] {
            assert_eq!(r.0, StatusCode::ACCEPTED);
            assert_eq!(serde_json::from_slice::<serde_json::Value>(&r.2).unwrap()["running"], true);
            all.extend(r.2.clone());
        }
        let mut last = serde_json::Value::Null;
        for _ in 0..100 {
            let (_, _, b) = call(&a, "GET", "/api/klang/sync", None, None).await;
            all.extend(b.clone());
            last = serde_json::from_slice(&b).unwrap();
            if last["running"] == false {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(s, StatusCode::OK);
        assert_eq!(last["last"]["new"], 2, "{last}");
        assert_eq!(last["last"]["message"], "Klang: 2 nya, 0 uppdaterade, 0 oförändrade, 0 hoppades över.");
        assert_eq!(m.list_calls.load(std::sync::atomic::Ordering::SeqCst), 2, "single flight: 2 pages listed once");
        assert_eq!(m.detail_calls.load(std::sync::atomic::Ordering::SeqCst), 2);

        let (_, _, b) = call(&a, "GET", "/api/notes", None, None).await;
        all.extend(b.clone());
        let list: serde_json::Value = serde_json::from_slice(&b).unwrap();
        let items = list["notes"].as_array().or(list.as_array()).unwrap().clone();
        assert_eq!(items.len(), 2);
        assert!(items.iter().all(|n| n["source"] == "klang" && n["has_audio"] == false));
        let id_of = |k: &str| klang::note_id(k);
        let (_, _, b) = call(&a, "GET", &format!("/api/notes/{}", id_of("conv_1")), None, None).await;
        all.extend(b.clone());
        let n: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(n["summary"], "**Viktigt**");
        assert_eq!(n["timed"], true);
        assert_eq!(n["klang"]["id"], "conv_1");
        let (_, _, b) = call(&a, "GET", &format!("/api/notes/{}", id_of("conv_2")), None, None).await;
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&b).unwrap()["timed"], false);
        let (s, _, b) = call(&a, "GET", &format!("/api/notes/{}/txt", id_of("conv_2")), None, None).await;
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8_lossy(&b).contains("Ada: utan tider"));
        let (s, _, _) = call(&a, "GET", &format!("/api/notes/{}/audio", id_of("conv_2")), None, None).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        // delete → tombstone → re-sync skips it
        let (s, _, _) = call(&a, "DELETE", &format!("/api/notes/{}", id_of("conv_2")), None, None).await;
        assert_eq!(s, StatusCode::NO_CONTENT);
        call(&a, "POST", "/api/klang/sync", None, None).await;
        for _ in 0..100 {
            let (_, _, b) = call(&a, "GET", "/api/klang/sync", None, None).await;
            all.extend(b.clone());
            last = serde_json::from_slice(&b).unwrap();
            if last["running"] == false {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!((last["last"]["new"].as_u64(), last["last"]["unchanged"].as_u64(), last["last"]["skipped"].as_u64()), (Some(0), Some(1), Some(1)), "{last}");
        let all = String::from_utf8_lossy(&all);
        assert!(!all.contains(klang::mock::KEY), "API key leaked into a response");
        // nor into the note files
        for e in walk(&tmp2.path().join("notes")) {
            let c = std::fs::read(&e).unwrap();
            assert!(!String::from_utf8_lossy(&c).contains(klang::mock::KEY), "{}", e.display());
        }
    }

    fn walk(d: &Path) -> Vec<PathBuf> {
        let mut out = vec![];
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() { out.extend(walk(&p)) } else { out.push(p) }
        }
        out
    }

    async fn wait_job(app: &Router, id: &str) -> serde_json::Value {
        for _ in 0..300 {
            let (_, _, b) = call(app, "GET", &format!("/api/jobs/{id}"), None, None).await;
            let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
            if j["status"] == "error" || j["status"] == "done" {
                return j;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("job {id} did not finish");
    }

    #[tokio::test]
    async fn url_endpoint_validation_and_ssrf() {
        let tmp = tempfile::tempdir().unwrap();
        let app = app(state(tmp.path()));
        let post = |v: serde_json::Value| {
            let app = app.clone();
            async move {
                let (c, _, b) = call(&app, "POST", "/api/jobs/url", Some(v), None).await;
                (c, serde_json::from_slice::<serde_json::Value>(&b).unwrap_or_default()["error"].as_str().unwrap_or("").to_string())
            }
        };
        for (u, want) in [
            ("", "Ogiltig länk"),
            ("not a url", "Ogiltig länk"),
            ("ftp://example.com/a.mp3", "Ogiltig länk"),
            ("file:///etc/passwd", "Ogiltig länk"),
            ("https://user:pw@example.com/a.mp3", "Ogiltig länk"),
            ("http://127.0.0.1:8795/api/notes", "lokal eller privat"),
            ("http://localhost:8795/", "lokal eller privat"),
            ("http://[::1]/", "lokal eller privat"),
            ("http://169.254.169.254/latest/meta-data/", "lokal eller privat"),
            ("http://100.100.100.100/", "lokal eller privat"),
            ("http://192.168.1.1/a.mp3", "lokal eller privat"),
            ("http://[fd00::1]/a.mp3", "lokal eller privat"),
            ("http://2130706433/", "lokal eller privat"),
        ] {
            let (c, e) = post(serde_json::json!({ "url": u })).await;
            assert_eq!(c, StatusCode::BAD_REQUEST, "{u}");
            assert!(e.contains(want), "{u}: {e}");
        }
        let (c, _) = post(serde_json::json!({ "url": "https://example.com/a.mp3", "model": "x; rm -rf /" })).await;
        assert_eq!(c, StatusCode::BAD_REQUEST);
        let (c, _, _) = call(&app, "POST", "/api/jobs/url", None, None).await;
        assert!(c.is_client_error());
        // nothing was queued
        let (_, _, b) = call(&app, "GET", "/api/health", None, None).await;
        let h: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(h["ok"], true);
        assert!(h["yt_dlp"]["available"].is_boolean());
    }

    #[tokio::test]
    async fn url_job_downloads_from_mock_and_enforces_limits() {
        let base = fetch::tests::mock_server().await;
        let tmp = tempfile::tempdir().unwrap();
        // duration limit 1 s: the 3 s clip is downloaded and converted, then refused
        let st = state_with(tmp.path(), fetch::NetPolicy::allow_loopback_for_tests(), |c| c.url_max_duration = 1);
        let app = app(st.clone());
        let (c, _, b) = call(&app, "POST", "/api/jobs/url", Some(serde_json::json!({ "url": format!("{base}/clip.wav") })), None).await;
        assert_eq!(c, StatusCode::ACCEPTED);
        let id = serde_json::from_slice::<serde_json::Value>(&b).unwrap()["id"].as_str().unwrap().to_string();
        let j = wait_job(&app, &id).await;
        assert_eq!(j["source_url"], format!("{base}/clip.wav"));
        assert_eq!(j["fetched_via"], "http");
        assert_eq!(j["filename"], "clip.wav");
        if which("ffmpeg").is_some() {
            assert_eq!(j["error_user"], "Ljudet är för långt. Gränsen är 1 s.", "{j}");
        }
        // temp files are gone
        let left: Vec<_> = std::fs::read_dir(&st.cfg.work_dir).unwrap().flatten().map(|e| e.file_name()).collect();
        assert!(left.is_empty(), "{left:?}");

        // a web page without yt-dlp → clear hint; HTTP errors are reported
        for (path, want) in [("/page.html", "kräver yt-dlp"), ("/missing.wav", "fel 404"), ("/data.json", "inte på en ljud")] {
            let (_, _, b) = call(&app, "POST", "/api/jobs/url", Some(serde_json::json!({ "url": format!("{base}{path}") })), None).await;
            let id = serde_json::from_slice::<serde_json::Value>(&b).unwrap()["id"].as_str().unwrap().to_string();
            let j = wait_job(&app, &id).await;
            assert!(j["error_user"].as_str().unwrap().contains(want), "{path}: {j}");
        }
        // size limit
        let st = state_with(tmp.path(), fetch::NetPolicy::allow_loopback_for_tests(), |c| c.url_max_mb = 1);
        let app = super::app(st);
        let (_, _, b) = call(&app, "POST", "/api/jobs/url", Some(serde_json::json!({ "url": format!("{base}/big.wav") })), None).await;
        let id = serde_json::from_slice::<serde_json::Value>(&b).unwrap()["id"].as_str().unwrap().to_string();
        assert_eq!(wait_job(&app, &id).await["error_user"], "Filen är för stor. Gränsen är 1 MB.");
        // overall timeout
        let st = state_with(tmp.path(), fetch::NetPolicy::allow_loopback_for_tests(), |c| c.url_timeout = 1);
        let app = super::app(st);
        let (_, _, b) = call(&app, "POST", "/api/jobs/url", Some(serde_json::json!({ "url": format!("{base}/slow.wav") })), None).await;
        let id = serde_json::from_slice::<serde_json::Value>(&b).unwrap()["id"].as_str().unwrap().to_string();
        assert_eq!(wait_job(&app, &id).await["error_user"], "Nedladdningen tog för lång tid (mer än 1 s) och avbröts.");
    }

    #[tokio::test]
    async fn source_url_is_stored_and_old_notes_still_load() {
        let tmp = tempfile::tempdir().unwrap();
        let st = state(tmp.path());
        let mut n = note("cccc3333", 3000, "Från en länk");
        n.source_url = Some("https://example.com/podd.mp3".into());
        st.notes.create(n, None).unwrap();
        // an old note.json (v0.3.0) without the field
        let old = tmp.path().join("notes/dddd4444");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("note.json"), r#"{"id":"dddd4444","title":"Gammal","created":10,"model":"small","audio_duration":1.0,"filename":"a.m4a","segments":[]}"#).unwrap();
        let st = state(tmp.path());
        let app = app(st);
        let (_, _, b) = call(&app, "GET", "/api/notes/cccc3333", None, None).await;
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["source_url"], "https://example.com/podd.mp3");
        let (c, _, b) = call(&app, "GET", "/api/notes/dddd4444", None, None).await;
        assert_eq!(c, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert!(v.get("source_url").is_none() || v["source_url"].is_null());
        let json = std::fs::read_to_string(tmp.path().join("notes/dddd4444/note.json")).unwrap();
        assert!(!json.contains("source_url"));
    }

    fn note(id: &str, created: i64, text: &str) -> Note {
        Note {
            id: id.into(),
            title: notes::default_title(created, text),
            created,
            model: "small".into(),
            audio_duration: 2.0,
            filename: "inspelning-20261002.m4a".into(),
            audio: None,
            audio_bytes: 0,
            backend: None,
            source_url: None,
            segments: vec![
                Segment { start: 0.0, end: 1.0, text: text.into() },
                Segment { start: 1.0, end: 2.0, text: "Slut.".into() },
            ],
            ..Default::default()
        }
    }

    async fn call(app: &Router, method: &str, uri: &str, body: Option<serde_json::Value>, range: Option<&str>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
        let mut b = axum::http::Request::builder().method(method).uri(uri);
        if let Some(r) = range {
            b = b.header(header::RANGE, r);
        }
        let req = match body {
            Some(v) => b.header(header::CONTENT_TYPE, "application/json").body(Body::from(v.to_string())).unwrap(),
            None => b.body(Body::empty()).unwrap(),
        };
        let r = app.clone().oneshot(req).await.unwrap();
        let (parts, body) = r.into_parts();
        (parts.status, parts.headers, body.collect().await.unwrap().to_bytes().to_vec())
    }

    #[tokio::test]
    async fn notes_api_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let st = state(tmp.path());
        let audio = tmp.path().join("up.m4a");
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&audio, &bytes).unwrap();
        st.notes.create(note("aaaa1111", 1000, "Hej från mötet om budgeten"), Some(&audio)).unwrap();
        st.notes.create(note("bbbb2222", 2000, "Rödeby är en tätort"), None).unwrap();
        let app = app(st.clone());

        let (c, _, b) = call(&app, "GET", "/api/notes", None, None).await;
        assert_eq!(c, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["notes"][0]["id"], "bbbb2222", "newest first");
        assert_eq!(v["total"], 2);

        let (_, _, b) = call(&app, "GET", "/api/notes?q=BUDGETEN", None, None).await;
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["notes"].as_array().unwrap().len(), 1);
        assert!(v["notes"][0]["snippet"].as_str().unwrap().contains("budgeten"));

        let (c, _, b) = call(&app, "GET", "/api/notes/aaaa1111", None, None).await;
        assert_eq!(c, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["segments"].as_array().unwrap().len(), 2);
        assert_eq!(v["has_audio"], true);

        // rename
        let (c, _, _) = call(&app, "PATCH", "/api/notes/aaaa1111", Some(serde_json::json!({"title": "Budgetmöte"})), None).await;
        assert_eq!(c, StatusCode::OK);
        let (c, _, _) = call(&app, "PATCH", "/api/notes/aaaa1111", Some(serde_json::json!({"title": "  "})), None).await;
        assert_eq!(c, StatusCode::BAD_REQUEST);
        let (c, _, _) = call(&app, "PATCH", "/api/notes/nope", Some(serde_json::json!({"title": "x"})), None).await;
        assert_eq!(c, StatusCode::NOT_FOUND);
        assert_eq!(st.notes.get("aaaa1111").unwrap().title, "Budgetmöte");

        // audio: full, ranges, unsatisfiable
        let (c, h, b) = call(&app, "GET", "/api/notes/aaaa1111/audio", None, None).await;
        assert_eq!((c, b.len()), (StatusCode::OK, 1000));
        assert_eq!(h[header::CONTENT_TYPE], "audio/mp4");
        assert_eq!(h[header::ACCEPT_RANGES], "bytes");
        let (c, h, b) = call(&app, "GET", "/api/notes/aaaa1111/audio", None, Some("bytes=0-1")).await;
        assert_eq!((c, b.as_slice()), (StatusCode::PARTIAL_CONTENT, &bytes[0..2]));
        assert_eq!(h[header::CONTENT_RANGE], "bytes 0-1/1000");
        let (c, _, b) = call(&app, "GET", "/api/notes/aaaa1111/audio", None, Some("bytes=990-")).await;
        assert_eq!((c, b.as_slice()), (StatusCode::PARTIAL_CONTENT, &bytes[990..]));
        let (c, _, _) = call(&app, "GET", "/api/notes/aaaa1111/audio", None, Some("bytes=5000-6000")).await;
        assert_eq!(c, StatusCode::RANGE_NOT_SATISFIABLE);
        let (c, _, _) = call(&app, "GET", "/api/notes/bbbb2222/audio", None, None).await;
        assert_eq!(c, StatusCode::NOT_FOUND);

        // downloads
        let (c, h, b) = call(&app, "GET", "/api/notes/aaaa1111/srt", None, None).await;
        assert_eq!(c, StatusCode::OK);
        assert!(String::from_utf8(b).unwrap().starts_with("1\r\n00:00:00,000 --> 00:00:01,000\r\nHej från mötet"));
        assert!(h[header::CONTENT_DISPOSITION].to_str().unwrap().contains("Budgetm%C3%B6te.srt"));
        let (_, _, b) = call(&app, "GET", "/api/notes/aaaa1111/txt", None, None).await;
        assert_eq!(String::from_utf8(b).unwrap(), "Hej från mötet om budgeten Slut.\n");
        let (_, _, b) = call(&app, "GET", "/api/notes/aaaa1111/txt-ts", None, None).await;
        assert!(String::from_utf8(b).unwrap().starts_with("[00:00:00.000 - 00:00:01.000] Hej"));
        let (_, _, b) = call(&app, "GET", "/api/notes/aaaa1111/json", None, None).await;
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["title"], "Budgetmöte");
        let (c, _, _) = call(&app, "GET", "/api/notes/aaaa1111/exe", None, None).await;
        assert_eq!(c, StatusCode::NOT_FOUND);

        // path traversal never resolves
        for bad in ["/api/notes/..%2F..%2Fetc", "/api/notes/..", "/api/notes/%2e%2e/audio", "/api/notes/AAAA1111"] {
            let (c, _, _) = call(&app, "GET", bad, None, None).await;
            assert_eq!(c, StatusCode::NOT_FOUND, "{bad}");
        }

        // delete
        let (c, _, _) = call(&app, "DELETE", "/api/notes/aaaa1111", None, None).await;
        assert_eq!(c, StatusCode::NO_CONTENT);
        let (c, _, _) = call(&app, "DELETE", "/api/notes/aaaa1111", None, None).await;
        assert_eq!(c, StatusCode::NOT_FOUND);
        assert!(!tmp.path().join("notes/aaaa1111").exists());

        // restart: state comes back from disk
        let st2 = state(tmp.path());
        assert_eq!(st2.notes.len(), 1);
        assert!(st2.notes.get("bbbb2222").is_some());
    }

    #[tokio::test]
    async fn static_assets_and_index() {
        let tmp = tempfile::tempdir().unwrap();
        let app = app(state(tmp.path()));
        let (c, h, b) = call(&app, "GET", "/manifest.webmanifest", None, None).await;
        assert_eq!(c, StatusCode::OK);
        assert_eq!(h[header::CONTENT_TYPE], "application/manifest+json");
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["display"], "standalone");
        for p in ["/apple-touch-icon.png", "/icon-192.png", "/icon-512.png", "/icon-maskable-512.png"] {
            let (c, h, b) = call(&app, "GET", p, None, None).await;
            assert_eq!((c, &h[header::CONTENT_TYPE]), (StatusCode::OK, &HeaderValue::from_static("image/png")), "{p}");
            assert_eq!(&b[1..4], b"PNG");
        }
        let (_, _, b) = call(&app, "GET", "/", None, None).await;
        let html = String::from_utf8(b).unwrap();
        for needle in ["rel=\"manifest\"", "apple-touch-icon", "apple-mobile-web-app-capable", "theme-color", "viewport-fit=cover"] {
            assert!(html.contains(needle), "{needle}");
        }
        // nothing that would make the browser load something from elsewhere
        let lower = html.to_ascii_lowercase().replace(' ', "");
        for bad in ["src=\"http", "src='http", "href=\"http", "href='http", "url(http", "url(\"http", "url('http", "@import", "fetch(\"http", "fetch('http", "//cdn", "src=\"//", "href=\"//"] {
            assert!(!lower.contains(bad), "no external requests: {bad}");
        }
    }

    #[test]
    fn host_flags() {
        let mut cfg = Config::from_env();
        apply_args(&mut cfg, &["--host".into(), "0.0.0.0".into(), "--port=9000".into(), "--notes-dir".into(), "/tmp/n".into()]).unwrap();
        assert_eq!((cfg.host.as_str(), cfg.port, cfg.notes_dir.as_path()), ("0.0.0.0", 9000, Path::new("/tmp/n")));
        assert!(apply_args(&mut cfg, &["--bogus".into()]).is_err());
        assert!(is_loopback("127.0.0.1") && is_loopback("localhost") && is_loopback("::1") && is_loopback("[::1]"));
        assert!(!is_loopback("0.0.0.0") && !is_loopback("100.64.1.2"));
    }
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
