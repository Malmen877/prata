//! Read-only import of conversations from Klang (<https://docs.klang.ai>).
//!
//! Enabled only when `KLANG_API_KEY` is set. The key lives in [`ApiKey`] whose `Debug`
//! is redacted and is only ever put in the `Authorization` header – never in notes,
//! logs, error messages or API responses.
//!
//! A sync lists all conversations (`GET /conversations`, following `next_cursor` while
//! `has_more`) and imports the `ready` ones as notes:
//! * **Dedupe:** each note stores the Klang id (`note.klang.id`); the note id is derived
//!   from it (`klang-<id>`), so a re-sync never creates a second copy.
//! * **Updates:** a conversation whose `updated_at` is unchanged is skipped without
//!   fetching it again (saves API calls – the free plan allows 50 a day). Otherwise the
//!   detail is fetched and title, summary, transcript and date are compared; the note
//!   is rewritten when something changed. A title the user renamed in Prata is kept.
//! * **Deleted here:** deleting an imported note records its Klang id in
//!   `.klang-deleted.json` in the notes folder; later syncs skip it ("hoppade över").
//! * **Timestamps:** the note date is `started_at` if Klang ever sends it, else
//!   `created_at`, else `updated_at`, else the import time.
//! * **Rate limits:** HTTP 429 waits `Retry-After` (at most `max_wait`, at most
//!   `max_retries` times); 5xx and network errors are retried with a short backoff.
//!   Progress made before a failure is kept and reported.

use std::{collections::HashSet, fmt, time::Duration};

use serde::{Deserialize, Serialize};

use crate::notes::{KlangRef, Note, Segment, Store};

pub const DEFAULT_BASE: &str = "https://app.klang.ai/api/v1";

/// The API key. `Debug`/`Display` never show it.
#[derive(Clone)]
pub struct ApiKey(String);

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

#[derive(Clone, Debug)]
pub struct Client {
    base: url::Url,
    key: ApiKey,
    http: reqwest::Client,
    pub max_retries: u32,
    /// Longest Retry-After we are willing to wait inside one sync
    pub max_wait: Duration,
    /// First backoff for 5xx / network errors (doubles per retry)
    pub backoff: Duration,
    /// Safety cap on pages per sync
    pub max_pages: usize,
}

impl Client {
    /// From `KLANG_API_KEY` (or `PRATA_KLANG_API_KEY`) and optional `PRATA_KLANG_BASE_URL`.
    /// `Ok(None)` when no key is set (the feature is off).
    pub fn from_env() -> Result<Option<Self>, String> {
        let key = ["KLANG_API_KEY", "PRATA_KLANG_API_KEY"]
            .iter()
            .find_map(|n| std::env::var(n).ok())
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty());
        let Some(key) = key else { return Ok(None) };
        let base = std::env::var("PRATA_KLANG_BASE_URL").ok().filter(|b| !b.trim().is_empty());
        Self::new(base.as_deref().unwrap_or(DEFAULT_BASE), key).map(Some)
    }

    pub fn new(base: &str, key: String) -> Result<Self, String> {
        let mut base = url::Url::parse(base.trim()).map_err(|_| "PRATA_KLANG_BASE_URL är ingen giltig URL".to_string())?;
        let loopback = matches!(base.host_str(), Some("127.0.0.1" | "localhost" | "[::1]" | "::1"));
        match base.scheme() {
            "https" => {}
            "http" if loopback => {}
            _ => return Err("PRATA_KLANG_BASE_URL måste börja med https:// (http bara mot 127.0.0.1)".into()),
        }
        if !base.username().is_empty() || base.password().is_some() {
            return Err("PRATA_KLANG_BASE_URL får inte innehålla inloggningsuppgifter".into());
        }
        if !base.path().ends_with('/') {
            let p = format!("{}/", base.path());
            base.set_path(&p);
        }
        base.set_query(None);
        let http = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none()) // never forward the key elsewhere
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .user_agent(concat!("Prata/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| format!("http-klient: {e}"))?;
        Ok(Self {
            base,
            key: ApiKey(key),
            http,
            max_retries: 3,
            max_wait: Duration::from_secs(60),
            backoff: Duration::from_secs(1),
            max_pages: 1000,
        })
    }

    /// Base URL without credentials (safe to log).
    pub fn base(&self) -> &str {
        self.base.as_str()
    }

    async fn get(&self, path: &str, query: &[(&str, &str)]) -> Result<serde_json::Value, Error> {
        let mut u = self.base.join(path).map_err(|_| Error::BadResponse)?;
        if !query.is_empty() {
            u.query_pairs_mut().extend_pairs(query);
        }
        let mut attempt = 0u32;
        loop {
            let res = self
                .http
                .get(u.clone())
                .bearer_auth(&self.key.0)
                .header(reqwest::header::ACCEPT, "application/json")
                .send()
                .await;
            let retry_in = match res {
                Err(e) => {
                    let err = if e.is_timeout() { Error::Timeout } else { Error::Network };
                    eprintln!("[klang] GET {path}: {}", e.without_url());
                    if attempt >= self.max_retries {
                        return Err(err);
                    }
                    self.backoff * 2u32.pow(attempt)
                }
                Ok(r) => {
                    let status = r.status().as_u16();
                    match status {
                        200..=299 => {
                            let body = r.bytes().await.map_err(|e| if e.is_timeout() { Error::Timeout } else { Error::Network })?;
                            return serde_json::from_slice(&body).map_err(|_| Error::BadResponse);
                        }
                        401 => return Err(Error::Auth),
                        403 => {
                            let t = error_type(r).await;
                            return Err(if t.as_deref() == Some("plan_upgrade_required") { Error::Plan } else { Error::Forbidden });
                        }
                        404 => return Err(Error::NotFound),
                        429 => {
                            let after = r
                                .headers()
                                .get(reqwest::header::RETRY_AFTER)
                                .and_then(|v| v.to_str().ok())
                                .and_then(|v| v.trim().parse::<u64>().ok());
                            let wait = after.map(Duration::from_secs).unwrap_or(self.backoff * 2u32.pow(attempt));
                            if attempt >= self.max_retries || wait > self.max_wait {
                                return Err(Error::RateLimited(after));
                            }
                            wait
                        }
                        500..=599 => {
                            if attempt >= self.max_retries {
                                return Err(Error::Server(status));
                            }
                            self.backoff * 2u32.pow(attempt)
                        }
                        _ => return Err(Error::Status(status)),
                    }
                }
            };
            attempt += 1;
            tokio::time::sleep(retry_in).await;
        }
    }
}

async fn error_type(r: reqwest::Response) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(&r.bytes().await.ok()?).ok()?;
    v.pointer("/error/type")?.as_str().map(String::from)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Auth,
    Forbidden,
    Plan,
    NotFound,
    RateLimited(Option<u64>),
    Timeout,
    Network,
    Server(u16),
    Status(u16),
    BadResponse,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Auth => f.write_str("Ogiltig Klang-nyckel – kontrollera KLANG_API_KEY."),
            Error::Forbidden => f.write_str("Klang-nyckeln saknar behörighet till samtalen (403)."),
            Error::Plan => f.write_str("Din Klang-plan ger inte API-åtkomst (403) – uppgradera i Klang."),
            Error::NotFound => f.write_str("Hittades inte i Klang (404)."),
            Error::RateLimited(Some(s)) => {
                write!(f, "Klang begränsar antalet anrop just nu – försök igen om {}.", human_secs(*s))
            }
            Error::RateLimited(None) => f.write_str("Klang begränsar antalet anrop just nu – försök igen senare."),
            Error::Timeout => f.write_str("Klang svarade inte i tid – försök igen."),
            Error::Network => f.write_str("Kunde inte nå Klang – kontrollera nätverket."),
            Error::Server(c) => write!(f, "Klang har tekniska problem (HTTP {c}) – försök igen senare."),
            Error::Status(c) => write!(f, "Oväntat svar från Klang (HTTP {c})."),
            Error::BadResponse => f.write_str("Oväntat svar från Klang (kunde inte läsa JSON)."),
        }
    }
}

fn human_secs(s: u64) -> String {
    match s {
        0..=90 => format!("{s} s"),
        91..=5399 => format!("{} min", s.div_ceil(60)),
        _ => format!("{} h", s.div_ceil(3600)),
    }
}

// ---------------------------------------------------------------- API shapes (lenient)

#[derive(Deserialize, Default)]
struct Page {
    #[serde(default)]
    data: Vec<serde_json::Value>,
    #[serde(default)]
    next_cursor: Option<String>,
    #[serde(default)]
    has_more: bool,
}

#[derive(Deserialize, Default, Debug)]
struct Conv {
    id: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    started_at: Option<String>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    updated_at: Option<String>,
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    sources: Vec<Source>,
}

#[derive(Deserialize, Default, Debug)]
struct Source {
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    duration_seconds: Option<f64>,
}

// ---------------------------------------------------------------- sync

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Report {
    pub new: usize,
    pub updated: usize,
    pub unchanged: usize,
    /// Conversations that could not be imported (unreadable, gone, save failed)
    pub skipped: usize,
    /// Ready conversations deleted here earlier (tombstoned); not part of the message
    pub deleted_here: usize,
    /// Swedish error message when the sync stopped early (counts so far are kept)
    pub error: Option<String>,
    /// Swedish one-line summary
    pub message: String,
    pub finished_at: i64,
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

impl Report {
    fn finish(mut self, error: Option<Error>) -> Self {
        let mut counts = format!(
            "{}, {}, {}",
            plural(self.new, "ny", "nya"),
            plural(self.updated, "uppdaterad", "uppdaterade"),
            plural(self.unchanged, "oförändrad", "oförändrade"),
        );
        if self.skipped > 0 {
            counts.push_str(&format!(", {} hoppades över", self.skipped));
        }
        self.message = match &error {
            None => format!("Klang: {counts}."),
            Some(e) if self.new + self.updated + self.unchanged + self.skipped > 0 => format!("{e} Hittills: {counts}."),
            Some(e) => e.to_string(),
        };
        self.error = error.map(|e| e.to_string());
        self.finished_at = chrono::Utc::now().timestamp();
        self
    }
}

/// Unix seconds from an ISO 8601 timestamp.
fn parse_ts(s: Option<&str>) -> Option<i64> {
    let s = s?.trim();
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|d| d.timestamp())
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f").map(|d| d.and_utc().timestamp()))
        .ok()
}

/// `[hh:mm:ss]` / `[mm:ss]` (optionally with fractions) at the start of a line.
fn parse_stamp(line: &str) -> Option<(f64, &str)> {
    let rest = line.strip_prefix('[')?;
    let close = rest.find(']')?;
    let inner = rest[..close].split([' ', '-', '–']).next()?.trim();
    let parts: Vec<&str> = inner.split(':').collect();
    if !(2..=3).contains(&parts.len()) {
        return None;
    }
    let mut t = 0.0;
    for p in &parts {
        let v: f64 = p.replace(',', ".").parse().ok()?;
        if v < 0.0 {
            return None;
        }
        t = t * 60.0 + v;
    }
    Some((t, rest[close + 1..].trim()))
}

/// Speaker-labelled transcript → segments. Timestamped lines (`[00:01:02] Ada: …`) become
/// segments ending where the next one starts; untimed transcripts get one segment per
/// line with start = end = 0.
pub fn parse_transcript(content: &str, duration: f64) -> Vec<Segment> {
    let lines: Vec<&str> = content.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let timed = lines.iter().any(|l| parse_stamp(l).is_some());
    let mut segs: Vec<Segment> = Vec::new();
    for l in lines {
        match parse_stamp(l) {
            Some((t, text)) => segs.push(Segment { start: t, end: t, text: text.to_string() }),
            None if timed && !segs.is_empty() => {
                let s = segs.last_mut().unwrap();
                if !s.text.is_empty() {
                    s.text.push(' ');
                }
                s.text.push_str(l);
            }
            None => segs.push(Segment { start: 0.0, end: 0.0, text: l.to_string() }),
        }
    }
    segs.retain(|s| !s.text.is_empty());
    if timed {
        for i in 0..segs.len() {
            let next = segs.get(i + 1).map(|n| n.start).unwrap_or(duration);
            segs[i].end = next.max(segs[i].start);
        }
    }
    segs
}

// ---------------------------------------------------------------- titles and plain text

const MONTHS: [&str; 24] = [
    "januari", "februari", "mars", "april", "maj", "juni", "juli", "augusti", "september", "oktober", "november", "december",
    "january", "february", "march", "may", "june", "july", "august", "october", "jan", "feb", "mar", "apr",
];
const MONTHS2: [&str; 9] = ["jun", "jul", "aug", "sep", "sept", "okt", "oct", "nov", "dec"];
const DAYS: [&str; 21] = [
    "måndag", "tisdag", "onsdag", "torsdag", "fredag", "lördag", "söndag", "mån", "tis", "ons", "tors", "fre", "lör", "sön",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
];
const GENERIC_TITLES: [&str; 18] = [
    "untitled", "namnlös", "namnlöst", "utan titel", "no title", "nytt samtal", "new conversation", "samtal", "conversation",
    "möte", "meeting", "nytt möte", "new meeting", "inspelning", "recording", "ny inspelning", "new recording", "klang",
];
const GENERIC_HEADINGS: [&str; 22] = [
    "sammanfattning", "summary", "sammandrag", "översikt", "overview", "beslut", "decisions", "anteckningar", "notes",
    "mötesanteckningar", "meeting notes", "agenda", "bakgrund", "background", "att göra", "action items", "nästa steg",
    "next steps", "viktiga punkter", "key points", "deltagare", "participants",
];

/// A title that carries no information: empty, only a date/time (Klang names untitled
/// conversations like "30 sep. 10:51"), a generic word, or Prata's old "Klang-samtal <date>".
pub fn is_placeholder_title(t: &str) -> bool {
    let l = t.trim().to_lowercase();
    if l.is_empty() || GENERIC_TITLES.contains(&l.as_str()) {
        return true;
    }
    let l = l.strip_prefix("klang-samtal").unwrap_or(&l);
    l.split(|c: char| c.is_whitespace() || ".,:/-–()·".contains(c)).filter(|w| !w.is_empty()).all(|w| {
        w.chars().all(|c| c.is_ascii_digit())
            || MONTHS.contains(&w)
            || MONTHS2.contains(&w)
            || DAYS.contains(&w)
            || matches!(w, "kl" | "am" | "pm" | "idag" | "igår" | "today" | "yesterday")
    })
}

/// Markdown inline syntax → text: links/images keep their text, emphasis/code markers and tags go.
pub fn strip_inline_md(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        // [text](url) and ![alt](url)
        if ch == '[' || (ch == '!' && c.get(i + 1) == Some(&'[')) {
            let start = if ch == '!' { i + 1 } else { i };
            if let Some(close) = (start + 1..c.len()).find(|&k| c[k] == ']') {
                if c.get(close + 1) == Some(&'(') {
                    if let Some(end) = (close + 2..c.len()).find(|&k| c[k] == ')') {
                        out.extend(&c[start + 1..close]);
                        i = end + 1;
                        continue;
                    }
                }
            }
        }
        // <tag …>
        if ch == '<' {
            if let Some(end) = (i + 1..c.len()).find(|&k| c[k] == '>') {
                let inner: String = c[i + 1..end].iter().collect();
                if inner.trim_start_matches('/').chars().next().is_some_and(|x| x.is_ascii_alphabetic()) {
                    i = end + 1;
                    continue;
                }
            }
        }
        match ch {
            '*' | '`' | '~' => {}
            '_' => {
                let prev = i.checked_sub(1).map(|k| c[k]);
                let next = c.get(i + 1).copied();
                // keep snake_case / file_names, drop _emphasis_ markers
                if prev.is_some_and(char::is_alphanumeric) && next.is_some_and(char::is_alphanumeric) {
                    out.push('_');
                }
            }
            _ => out.push(ch),
        }
        i += 1;
    }
    out
}

/// One markdown line → (is_heading, text without block markers).
fn md_line(l: &str) -> (bool, String) {
    let t = l.trim();
    let hashes = t.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) && t[hashes..].starts_with(' ') {
        return (true, t[hashes..].trim().trim_end_matches('#').trim().to_string());
    }
    let mut t = t.trim_start_matches('>').trim();
    for m in ["- ", "* ", "+ ", "• "] {
        if let Some(r) = t.strip_prefix(m) {
            t = r.trim();
            break;
        }
    }
    if let Some(i) = t.find(|c: char| !c.is_ascii_digit()) {
        if i > 0 && i <= 3 && (t[i..].starts_with(". ") || t[i..].starts_with(") ")) {
            t = t[i + 2..].trim();
        }
    }
    for m in ["[ ] ", "[x] ", "[X] "] {
        if let Some(r) = t.strip_prefix(m) {
            t = r.trim();
        }
    }
    (false, t.to_string())
}

fn collapse(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Up to the first sentence end (". ", "! ", "? "); the full stop is dropped.
fn first_sentence(s: &str) -> &str {
    let b = s.as_bytes();
    for i in 0..b.len() {
        if matches!(b[i], b'.' | b'!' | b'?') && (i + 1 == b.len() || b[i + 1] == b' ') {
            // not after a single letter/number ("t.ex.", "kl. 10", "1.")
            let word_start = s[..i].rfind(' ').map(|k| k + 1).unwrap_or(0);
            if i - word_start <= 1 || s[word_start..i].contains('.') {
                continue;
            }
            return if b[i] == b'.' { &s[..i] } else { &s[..=i] };
        }
    }
    s
}

/// Shorten to at most `max` characters at a word boundary, with "…" when cut.
pub fn shorten(s: &str, max: usize) -> String {
    let s = collapse(s);
    if s.chars().count() <= max {
        return s;
    }
    let cut: String = s.chars().take(max).collect();
    let at = cut.rfind(' ').filter(|&i| i >= max / 3).unwrap_or(cut.len());
    let head = cut[..at].trim_end_matches(|c: char| c.is_whitespace() || ",;:–-(".contains(c));
    format!("{head}…")
}

const TITLE_MAX: usize = 60;

fn title_from_summary(md: &str) -> Option<String> {
    let mut first_text = None;
    for l in md.lines() {
        let (heading, t) = md_line(l);
        let t = collapse(&strip_inline_md(&t));
        let t = t.trim_end_matches(':').trim();
        if t.is_empty() || t.chars().all(|c| "-*_=|".contains(c)) {
            continue;
        }
        if heading {
            if !GENERIC_HEADINGS.contains(&t.to_lowercase().as_str()) {
                return Some(shorten(t, TITLE_MAX));
            }
        } else if first_text.is_none() {
            first_text = Some(shorten(first_sentence(t), TITLE_MAX));
        }
    }
    first_text.filter(|t| !t.is_empty())
}

/// "[00:00:12] Talare 1: Hej …" → "Hej …"
fn strip_speaker(t: &str) -> &str {
    let t = parse_stamp(t).map(|(_, r)| r).unwrap_or(t).trim();
    match t.find(": ") {
        Some(i) if i <= 40 && !t[..i].contains(['.', '!', '?']) => t[i + 2..].trim(),
        _ => t,
    }
}

fn title_from_transcript(segs: &[Segment]) -> Option<String> {
    let mut words = String::new();
    for s in segs {
        let t = strip_speaker(&s.text);
        if t.is_empty() {
            continue;
        }
        if !words.is_empty() {
            words.push(' ');
        }
        words.push_str(t);
        if words.split_whitespace().count() >= 4 {
            break;
        }
    }
    let w = collapse(&words);
    (!w.is_empty()).then(|| shorten(first_sentence(&w), TITLE_MAX))
}

/// Title for an imported conversation and where it came from:
/// Klang's own title (unless it's a placeholder) → summary (first specific heading, else
/// first sentence) → digest → first transcript words (no timestamps or speaker labels) → date.
pub fn choose_title(klang: Option<&str>, summary: Option<&str>, digest: Option<&str>, segs: &[Segment], created: i64) -> (String, &'static str) {
    if let Some(t) = klang.map(collapse).filter(|t| !is_placeholder_title(t)) {
        return (t.chars().take(200).collect(), "klang");
    }
    if let Some(t) = summary.and_then(title_from_summary) {
        return (t, "summary");
    }
    if let Some(t) = digest.and_then(title_from_summary) {
        return (t, "digest");
    }
    if let Some(t) = title_from_transcript(segs) {
        return (t, "transcript");
    }
    (format!("Klang-samtal {}", crate::notes::default_title(created, "")), "date")
}

/// Markdown summary → readable plain text for .txt exports.
pub fn md_to_plain(md: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for l in md.lines() {
        let raw = l.trim_end();
        let t = raw.trim();
        if t.is_empty() {
            if out.last().is_some_and(|x| !x.is_empty()) {
                out.push(String::new());
            }
            continue;
        }
        if t.chars().all(|c| "-*_".contains(c)) && t.len() >= 3 {
            continue; // horizontal rule
        }
        let indent = if raw.starts_with("  ") || raw.starts_with('\t') { "  " } else { "" };
        let (heading, body) = md_line(t);
        let body = collapse(&strip_inline_md(&body));
        let lt = t.trim_start_matches('>').trim_start();
        let line = if heading {
            if out.last().is_some_and(|x| !x.is_empty()) {
                out.push(String::new());
            }
            body
        } else if lt.starts_with("- [ ]") || lt.starts_with("* [ ]") {
            format!("{indent}☐ {body}")
        } else if lt.starts_with("- [x]") || lt.starts_with("- [X]") || lt.starts_with("* [x]") {
            format!("{indent}☑ {body}")
        } else if ["- ", "* ", "+ ", "• "].iter().any(|m| lt.starts_with(m)) {
            format!("{indent}• {body}")
        } else if let Some(i) = lt.find(|c: char| !c.is_ascii_digit()).filter(|&i| i > 0 && i <= 3 && (lt[i..].starts_with(". ") || lt[i..].starts_with(") "))) {
            format!("{indent}{}. {body}", &lt[..i])
        } else {
            body
        };
        out.push(line);
    }
    while out.last().is_some_and(|x| x.is_empty()) {
        out.pop();
    }
    out.join("\n")
}

/// Valid local note id for a Klang id: `klang-<id>` when possible, else a stable hash.
pub fn note_id(klang_id: &str) -> String {
    let ok = !klang_id.is_empty()
        && klang_id.len() <= 58
        && klang_id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if ok {
        return format!("klang-{klang_id}");
    }
    // FNV-1a 64
    let mut h: u64 = 0xcbf29ce484222325;
    for b in klang_id.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("klang-{h:016x}")
}

fn build_note(c: &Conv, existing: Option<&Note>) -> Note {
    let created = parse_ts(c.started_at.as_deref())
        .or_else(|| parse_ts(c.created_at.as_deref()))
        .or_else(|| parse_ts(c.updated_at.as_deref()))
        .or(existing.map(|n| n.created))
        .unwrap_or_else(|| chrono::Utc::now().timestamp());
    let transcripts: Vec<&Source> = c.sources.iter().filter(|s| s.kind.as_deref() == Some("transcript")).collect();
    let duration: f64 = transcripts.iter().filter_map(|s| s.duration_seconds).filter(|d| d.is_finite() && *d > 0.0).sum();
    let mut segments = Vec::new();
    for s in &transcripts {
        if let Some(t) = &s.content {
            segments.extend(parse_transcript(t, s.duration_seconds.unwrap_or(0.0)));
        }
    }
    let clean = |s: &Option<String>| s.as_deref().map(str::trim).filter(|s| !s.is_empty()).map(String::from);
    let summary = clean(&c.summary).or_else(|| clean(&c.digest));
    let (title, title_edited, title_source) = match existing {
        Some(n) if n.title_edited => (n.title.clone(), true, Some("user".to_string())),
        _ => {
            let (t, src) = choose_title(c.title.as_deref(), c.summary.as_deref(), c.digest.as_deref(), &segments, created);
            (t, false, Some(src.to_string()))
        }
    };
    Note {
        id: existing.map(|n| n.id.clone()).unwrap_or_else(|| note_id(&c.id)),
        title,
        title_edited,
        title_source,
        created,
        model: "klang".into(),
        audio_duration: duration,
        filename: "Klang".into(),
        backend: Some("Klang".into()),
        segments,
        source: Some("klang".into()),
        summary,
        klang: Some(KlangRef { id: c.id.clone(), updated_at: c.updated_at.clone() }),
        ..Default::default()
    }
}

fn same_content(a: &Note, b: &Note) -> bool {
    a.title == b.title && a.title_source == b.title_source && a.summary == b.summary && a.segments == b.segments && a.created == b.created && a.audio_duration == b.audio_duration
}

/// Run one full sync. Never panics on bad data; stops early on errors (keeping what was imported).
pub async fn sync(client: &Client, store: &Store) -> Report {
    let mut rep = Report::default();
    let mut cursor: Option<String> = None;
    let mut seen_cursors = HashSet::new();
    let mut seen_ids = HashSet::new();
    for _ in 0..client.max_pages {
        // only ready conversations: pending/failed ones aren't importable and aren't reported
        let mut q = vec![("limit", "100"), ("status", "ready")];
        if let Some(c) = &cursor {
            q.push(("cursor", c));
        }
        let page: Page = match client.get("conversations", &q).await {
            Ok(v) => match serde_json::from_value(v) {
                Ok(p) => p,
                Err(_) => return rep.finish(Some(Error::BadResponse)),
            },
            Err(e) => return rep.finish(Some(e)),
        };
        for item in page.data {
            let Ok(c) = serde_json::from_value::<Conv>(item) else {
                rep.skipped += 1;
                continue;
            };
            if c.id.is_empty() || !seen_ids.insert(c.id.clone()) {
                continue; // duplicate across pages
            }
            if c.status.as_deref().is_some_and(|s| s != "ready") {
                continue; // not ready yet (the API filter should already drop these)
            }
            if store.klang_deleted(&c.id) {
                rep.deleted_here += 1;
                continue;
            }
            let existing = store.find_klang(&c.id);
            if let Some(n) = &existing {
                let prev = n.klang.as_ref().and_then(|k| k.updated_at.as_deref());
                if prev.is_some() && prev == c.updated_at.as_deref() {
                    // Unchanged in Klang: no detail call. Still refresh an automatic title from the
                    // locally stored summary/transcript (e.g. notes imported before titles were derived).
                    let (title, src) = choose_title(c.title.as_deref(), n.summary.as_deref(), None, &n.segments, n.created);
                    if !n.title_edited && (title != n.title || n.title_source.as_deref() != Some(src)) {
                        let renamed = title != n.title;
                        let mut m = n.clone();
                        m.title = title;
                        m.title_source = Some(src.into());
                        match store.replace(m) {
                            // only a new title counts as an update; recording the source is silent
                            Ok(_) if renamed => rep.updated += 1,
                            Ok(_) => rep.unchanged += 1,
                            Err(e) => {
                                eprintln!("[klang] could not save conversation: {e:#}");
                                rep.skipped += 1;
                            }
                        }
                    } else {
                        rep.unchanged += 1;
                    }
                    continue;
                }
            }
            let detail: Conv = match client.get(&format!("conversations/{}", enc(&c.id)), &[]).await {
                Ok(v) => match serde_json::from_value(v) {
                    Ok(d) => d,
                    Err(_) => {
                        rep.skipped += 1;
                        continue;
                    }
                },
                Err(Error::NotFound) => {
                    rep.skipped += 1;
                    continue;
                }
                Err(e) => return rep.finish(Some(e)),
            };
            if detail.status.as_deref().is_some_and(|s| s != "ready") {
                continue;
            }
            // the list item's id is authoritative; fill gaps in the detail from it
            let merged = Conv {
                id: c.id.clone(),
                title: detail.title.or(c.title),
                status: Some("ready".into()),
                started_at: detail.started_at.or(c.started_at),
                created_at: detail.created_at.or(c.created_at),
                updated_at: detail.updated_at.or(c.updated_at),
                digest: detail.digest.or(c.digest),
                summary: detail.summary,
                sources: detail.sources,
            };
            let note = build_note(&merged, existing.as_ref());
            let res = match &existing {
                None => store.create(note, None).map(|_| rep.new += 1),
                Some(old) => {
                    let changed = !same_content(old, &note);
                    store.replace(note).map(|_| if changed { rep.updated += 1 } else { rep.unchanged += 1 })
                }
            };
            if let Err(e) = res {
                eprintln!("[klang] could not save conversation: {e:#}");
                rep.skipped += 1;
            }
        }
        if !page.has_more {
            return rep.finish(None);
        }
        match page.next_cursor {
            Some(c) if !c.is_empty() && seen_cursors.insert(c.clone()) => cursor = Some(c),
            _ => {
                eprintln!("[klang] has_more without a new next_cursor – stopping");
                return rep.finish(None);
            }
        }
    }
    eprintln!("[klang] page cap reached – stopping");
    rep.finish(None)
}

fn enc(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}

/// Single-flight wrapper: one sync at a time, last result kept for the UI.
pub struct Klang {
    pub client: Client,
    state: tokio::sync::Mutex<State>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct State {
    pub running: bool,
    pub started_at: Option<i64>,
    pub last: Option<Report>,
}

impl Klang {
    pub fn new(client: Client) -> Self {
        Self { client, state: Default::default() }
    }

    pub async fn state(&self) -> State {
        self.state.lock().await.clone()
    }

    /// Mark a sync as started. `false` when one is already running (the caller just joins it).
    pub async fn try_begin(&self) -> bool {
        let mut s = self.state.lock().await;
        if s.running {
            return false;
        }
        s.running = true;
        s.started_at = Some(chrono::Utc::now().timestamp());
        true
    }

    pub async fn end(&self, report: Report) {
        let mut s = self.state.lock().await;
        s.running = false;
        s.last = Some(report);
    }
}

/// In-process mock of the Klang API for tests (never the real one).
#[cfg(test)]
pub mod mock {
    use std::{
        collections::VecDeque,
        sync::{
            atomic::{AtomicU64, AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    use axum::{
        extract::{Path, Query, State},
        http::{HeaderMap, StatusCode},
        response::{IntoResponse, Response},
        routing::get,
        Json, Router,
    };
    use serde_json::{json, Value};

    pub const KEY: &str = "sk_test_mock_key_0123456789";

    #[derive(Default)]
    pub struct Mock {
        /// Full conversations (with summary/sources content); the list strips those
        pub convs: Mutex<Vec<Value>>,
        pub page_size: usize,
        pub list_calls: AtomicUsize,
        pub detail_calls: AtomicUsize,
        /// Retry-After values for the next requests (answered with 429)
        pub rate_limit: Mutex<VecDeque<String>>,
        /// Answer every request with this status (e.g. 500), consumed one at a time
        pub fail: Mutex<VecDeque<u16>>,
        pub delay_ms: AtomicU64,
    }

    pub fn conv(id: &str, title: &str, status: &str, created_at: Option<&str>, updated_at: &str, summary: &str, transcript: &str) -> Value {
        json!({
            "id": id, "title": title, "status": status, "created_at": created_at, "updated_at": updated_at,
            "termination_reason": null, "folder_id": null, "digest": "Kort sammanfattning", "tags": [],
            "summary": summary,
            "sources": [{ "id": format!("src_{id}"), "type": "transcript", "duration_seconds": 95, "language": "sv",
                          "title": null, "content": transcript, "participants": [], "added_at": updated_at }],
        })
    }

    fn auth(h: &HeaderMap) -> bool {
        h.get("authorization").and_then(|v| v.to_str().ok()) == Some(&format!("Bearer {KEY}"))
    }

    async fn gate(m: &Mock, h: &HeaderMap) -> Option<Response> {
        let d = m.delay_ms.load(Ordering::SeqCst);
        if d > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(d)).await;
        }
        if !auth(h) {
            return Some((StatusCode::UNAUTHORIZED, Json(json!({"error": {"type": "auth", "message": "Invalid API key"}}))).into_response());
        }
        if let Some(ra) = m.rate_limit.lock().unwrap().pop_front() {
            return Some((StatusCode::TOO_MANY_REQUESTS, [("retry-after", ra)], Json(json!({"error": {"type": "rate_limited", "message": "slow down"}}))).into_response());
        }
        if let Some(code) = m.fail.lock().unwrap().pop_front() {
            return Some((StatusCode::from_u16(code).unwrap(), Json(json!({"error": {"type": "internal_error", "message": "x"}}))).into_response());
        }
        None
    }

    async fn list(State(m): State<Arc<Mock>>, h: HeaderMap, Query(q): Query<std::collections::HashMap<String, String>>) -> Response {
        m.list_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(r) = gate(&m, &h).await {
            return r;
        }
        let all = m.convs.lock().unwrap().clone();
        let want = q.get("status").map(String::as_str).unwrap_or("ready");
        let all: Vec<Value> = all.into_iter().filter(|c| want == "all" || c["status"] == want).collect();
        let off: usize = q.get("cursor").and_then(|c| c.strip_prefix("c_")).and_then(|c| c.parse().ok()).unwrap_or(0);
        let n = m.page_size.max(1);
        let items: Vec<Value> = all
            .iter()
            .skip(off)
            .take(n)
            .map(|c| {
                let mut c = c.clone();
                c.as_object_mut().unwrap().remove("summary");
                for s in c["sources"].as_array_mut().unwrap() {
                    let o = s.as_object_mut().unwrap();
                    o.retain(|k, _| ["id", "type", "duration_seconds", "language"].contains(&k.as_str()));
                }
                c
            })
            .collect();
        let more = off + n < all.len();
        Json(json!({ "data": items, "has_more": more, "next_cursor": if more { Some(format!("c_{}", off + n)) } else { None } })).into_response()
    }

    async fn detail(State(m): State<Arc<Mock>>, h: HeaderMap, Path(id): Path<String>) -> Response {
        m.detail_calls.fetch_add(1, Ordering::SeqCst);
        if let Some(r) = gate(&m, &h).await {
            return r;
        }
        match m.convs.lock().unwrap().iter().find(|c| c["id"] == id.as_str()) {
            Some(c) => Json(c.clone()).into_response(),
            None => (StatusCode::NOT_FOUND, Json(json!({"error": {"type": "not_found", "message": "nope"}}))).into_response(),
        }
    }

    /// Start the mock on a random loopback port; returns it and its base URL.
    pub async fn start(m: Mock) -> (Arc<Mock>, String) {
        let m = Arc::new(m);
        let app = Router::new()
            .route("/api/v1/conversations", get(list))
            .route("/api/v1/conversations/{id}", get(detail))
            .with_state(m.clone());
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        (m, format!("http://{addr}/api/v1"))
    }

    pub fn client(base: &str) -> super::Client {
        let mut c = super::Client::new(base, KEY.into()).unwrap();
        c.backoff = std::time::Duration::from_millis(10);
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::Ordering;

    fn store() -> (tempfile::TempDir, Store) {
        let d = tempfile::tempdir().unwrap();
        let s = Store::open(d.path().join("notes")).unwrap();
        (d, s)
    }

    fn five() -> Vec<serde_json::Value> {
        use mock::conv;
        vec![
            conv("conv_a1", "Veckomöte", "ready", Some("2026-09-28T08:00:00Z"), "2026-09-28T09:00:00Z", "## Beslut\n- Ja", "[00:00:00] Ada: Hej.\n\n[00:00:05] Bob: Hej hej."),
            conv("conv_b2", "Kundsamtal", "ready", Some("2026-09-29T10:00:00Z"), "2026-09-29T10:30:00Z", "Bra samtal", "Ada: Utan tider.\n\nBob: Ja."),
            conv("conv_c3", "Pågår", "pending", Some("2026-09-30T10:00:00Z"), "2026-09-30T10:00:00Z", "", ""),
            conv("conv_d4", "Utan datum", "ready", None, "2026-10-01T12:00:00Z", "S", "[00:00:01] Ada: x"),
            conv("conv_e5", "Fel", "error", Some("2026-10-01T10:00:00Z"), "2026-10-01T10:00:00Z", "", ""),
        ]
    }

    #[tokio::test]
    async fn sync_paginates_dedupes_and_is_idempotent() {
        let (m, base) = mock::start(mock::Mock { convs: five().into(), page_size: 2, ..Default::default() }).await;
        let (_d, st) = store();
        let c = mock::client(&base);
        let r = sync(&c, &st).await;
        assert_eq!((r.new, r.updated, r.unchanged, r.skipped, r.error.clone()), (3, 0, 0, 0, None), "{r:?}");
        assert_eq!(m.list_calls.load(Ordering::SeqCst), 2); // 3 ready items, 2 per page
        assert_eq!(m.detail_calls.load(Ordering::SeqCst), 3); // only ready ones
        assert_eq!(r.message, "Klang: 3 nya, 0 uppdaterade, 0 oförändrade.");
        let a = st.get("klang-conv_a1").unwrap_or_else(|| st.find_klang("conv_a1").unwrap());
        assert_eq!(a.title, "Veckomöte");
        assert_eq!(a.source.as_deref(), Some("klang"));
        assert_eq!(a.summary.as_deref(), Some("## Beslut\n- Ja"));
        assert_eq!(a.created, parse_ts(Some("2026-09-28T08:00:00Z")).unwrap());
        assert_eq!(a.segments.len(), 2);
        assert_eq!(a.segments[1].text, "Bob: Hej hej.");
        assert_eq!((a.segments[0].end, a.segments[1].end), (5.0, 95.0));
        assert!(a.audio.is_none());
        let b = st.find_klang("conv_b2").unwrap();
        assert!(b.segments.iter().all(|s| s.end == 0.0) && b.segments.len() == 2);
        // missing created_at → updated_at
        let d4 = st.find_klang("conv_d4").unwrap();
        assert_eq!(d4.created, parse_ts(Some("2026-10-01T12:00:00Z")).unwrap());
        assert!(st.find_klang("conv_c3").is_none() && st.find_klang("conv_e5").is_none());

        // second run: nothing new, no detail calls
        let r2 = sync(&c, &st).await;
        assert_eq!((r2.new, r2.updated, r2.unchanged, r2.skipped), (0, 0, 3, 0), "{r2:?}");
        assert_eq!(m.detail_calls.load(Ordering::SeqCst), 3);
        assert_eq!(st.len(), 3);
        // a fresh Store (restart) still dedupes
        drop(st);
        let st = Store::open(_d.path().join("notes")).unwrap();
        let r3 = sync(&c, &st).await;
        assert_eq!((r3.new, r3.unchanged), (0, 3));
        assert_eq!(st.len(), 3);
    }

    #[tokio::test]
    async fn changed_conversations_update_but_keep_renamed_title() {
        let (m, base) = mock::start(mock::Mock { convs: five().into(), page_size: 100, ..Default::default() }).await;
        let (_d, st) = store();
        let c = mock::client(&base);
        sync(&c, &st).await;
        let a = st.find_klang("conv_a1").unwrap();
        st.rename(&a.id, "Mitt namn").unwrap();
        {
            let mut v = m.convs.lock().unwrap();
            v[0]["title"] = "Veckomöte v2".into();
            v[0]["summary"] = "Ny sammanfattning".into();
            v[0]["updated_at"] = "2026-10-02T09:00:00Z".into();
            v[1]["title"] = "Kundsamtal (ny titel)".into();
            v[1]["updated_at"] = "2026-10-02T09:00:00Z".into();
            v[3]["updated_at"] = "2026-10-02T09:00:00Z".into(); // only the timestamp moved
            v[3]["created_at"] = serde_json::Value::Null;
        }
        let r = sync(&c, &st).await;
        // d4 has no created_at, so its date follows updated_at and it counts as updated
        assert_eq!((r.new, r.updated, r.unchanged), (0, 3, 0), "{r:?}");
        let a = st.find_klang("conv_a1").unwrap();
        assert_eq!(a.title, "Mitt namn");
        assert!(a.title_edited);
        assert_eq!(a.summary.as_deref(), Some("Ny sammanfattning"));
        assert_eq!(st.find_klang("conv_b2").unwrap().title, "Kundsamtal (ny titel)");
        // only updated_at changes → fetched, but unchanged
        m.convs.lock().unwrap()[1]["updated_at"] = "2026-10-02T10:00:00Z".into();
        let r = sync(&c, &st).await;
        assert_eq!((r.updated, r.unchanged), (0, 3), "{r:?}");
        assert_eq!(st.find_klang("conv_b2").unwrap().klang.unwrap().updated_at.as_deref(), Some("2026-10-02T10:00:00Z"));
    }

    #[tokio::test]
    async fn deleted_notes_stay_deleted() {
        let (_m, base) = mock::start(mock::Mock { convs: five().into(), page_size: 100, ..Default::default() }).await;
        let (d, st) = store();
        let c = mock::client(&base);
        sync(&c, &st).await;
        let a = st.find_klang("conv_a1").unwrap();
        assert!(st.delete(&a.id).unwrap());
        assert!(st.klang_deleted("conv_a1"));
        let r = sync(&c, &st).await;
        assert_eq!((r.new, r.unchanged, r.skipped, r.deleted_here), (0, 2, 0, 1), "{r:?}");
        assert_eq!(r.message, "Klang: 0 nya, 0 uppdaterade, 2 oförändrade.");
        assert!(st.find_klang("conv_a1").is_none());
        // survives a restart; the tombstone file isn't treated as junk
        drop(st);
        let st = Store::open(d.path().join("notes")).unwrap();
        assert!(st.klang_deleted("conv_a1"));
        assert_eq!(sync(&c, &st).await.new, 0);
        assert!(d.path().join("notes").join(crate::notes::KLANG_TOMBSTONES).exists());
    }

    #[tokio::test]
    async fn rate_limits_are_retried_then_reported() {
        let (m, base) = mock::start(mock::Mock { convs: five().into(), page_size: 100, ..Default::default() }).await;
        m.rate_limit.lock().unwrap().extend(["1".to_string(), "0".to_string()]);
        let (_d, st) = store();
        let c = mock::client(&base);
        let t = std::time::Instant::now();
        let r = sync(&c, &st).await;
        assert!(t.elapsed() >= std::time::Duration::from_millis(900), "Retry-After honoured");
        assert_eq!((r.new, r.error.clone()), (3, None), "{r:?}");

        // too long a wait: stop with a Swedish message, keep partial progress
        m.convs.lock().unwrap()[0]["updated_at"] = "2026-10-02T11:00:00Z".into();
        m.rate_limit.lock().unwrap().extend(["0".into(), "3600".into()]);
        let r = sync(&c, &st).await;
        assert!(r.error.as_deref().unwrap().contains("försök igen om 60 min"), "{r:?}");
        assert_eq!(Some(r.message.clone()), r.error.clone()); // nothing done yet: no "Hittills"
        // more 429s than max_retries
        m.rate_limit.lock().unwrap().extend(std::iter::repeat_n("0".to_string(), 10));
        let r = sync(&c, &st).await;
        assert!(r.error.as_deref().unwrap().starts_with("Klang begränsar"), "{r:?}");
        m.rate_limit.lock().unwrap().clear();
        // 5xx are retried with backoff
        m.fail.lock().unwrap().extend([502, 503]);
        assert_eq!(sync(&c, &st).await.error, None);
        m.fail.lock().unwrap().extend([500; 5]);
        assert!(sync(&c, &st).await.error.unwrap().contains("HTTP 500"));
    }

    #[tokio::test]
    async fn bad_key_and_unreachable_server() {
        let (_m, base) = mock::start(mock::Mock { convs: five().into(), page_size: 100, ..Default::default() }).await;
        let (_d, st) = store();
        let mut c = Client::new(&base, "sk_wrong_key_987".into()).unwrap();
        c.backoff = Duration::from_millis(10);
        let r = sync(&c, &st).await;
        let e = r.error.clone().unwrap();
        assert!(e.starts_with("Ogiltig Klang-nyckel"), "{e}");
        assert!(!format!("{r:?}").contains("sk_wrong_key_987"));
        assert_eq!(st.len(), 0);
        // nothing listens on port 9 → network error after bounded retries
        let mut c = Client::new("http://127.0.0.1:9/api/v1", "k".into()).unwrap();
        c.backoff = Duration::from_millis(10);
        let e = sync(&c, &st).await.error.unwrap();
        assert!(e.starts_with("Kunde inte nå Klang"), "{e}");
    }

    #[test]
    fn key_is_redacted() {
        let c = Client::new("https://app.klang.ai/api/v1", "sk_secret123".into()).unwrap();
        let d = format!("{c:?}");
        assert!(!d.contains("sk_secret123"), "{d}");
        assert!(d.contains("redacted"));
    }

    #[test]
    fn base_url_rules() {
        assert!(Client::new("http://example.com/api/v1", "k".into()).is_err());
        assert!(Client::new("ftp://127.0.0.1/", "k".into()).is_err());
        assert!(Client::new("https://u:p@app.klang.ai/api/v1", "k".into()).is_err());
        let c = Client::new("http://127.0.0.1:9/api/v1", "k".into()).unwrap();
        assert_eq!(c.base(), "http://127.0.0.1:9/api/v1/");
        assert_eq!(Client::new(DEFAULT_BASE, "k".into()).unwrap().base(), "https://app.klang.ai/api/v1/");
    }

    #[test]
    fn transcript_with_timestamps() {
        let t = "[00:00:00] Ada Lovelace: Hej och välkomna.\n\n[00:00:12] Talare 2: Tack.\nfortsätter\n\n[01:02:03.5] Ada Lovelace: Slut.";
        let s = parse_transcript(t, 4000.0);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], Segment { start: 0.0, end: 12.0, text: "Ada Lovelace: Hej och välkomna.".into() });
        assert_eq!(s[1].text, "Talare 2: Tack. fortsätter");
        assert_eq!((s[2].start, s[2].end), (3723.5, 4000.0));
    }

    #[test]
    fn transcript_without_timestamps() {
        let s = parse_transcript("Ada: Hej.\n\nBob: Hej hej.\n", 0.0);
        assert_eq!(s.len(), 2);
        assert!(s.iter().all(|x| x.start == 0.0 && x.end == 0.0));
        assert_eq!(s[1].text, "Bob: Hej hej.");
        assert!(parse_transcript("", 10.0).is_empty());
        // brackets that aren't times are text
        assert_eq!(parse_transcript("[skratt] Ada: ja", 0.0)[0].text, "[skratt] Ada: ja");
    }

    #[test]
    fn ids_and_timestamps() {
        assert_eq!(note_id("kw3pq2nyax7lr9d"), "klang-kw3pq2nyax7lr9d");
        let h = note_id("Weird/ID");
        assert!(crate::notes::valid_id(&h) && h.starts_with("klang-") && h == note_id("Weird/ID"));
        assert_ne!(note_id("A"), note_id("a"));
        assert_eq!(parse_ts(Some("2026-03-12T09:15:00Z")), Some(1773306900));
        assert_eq!(parse_ts(Some("2026-03-12T10:15:00+01:00")), Some(1773306900));
        assert_eq!(parse_ts(Some("nonsense")), None);
    }

    fn seg(t: &str) -> Segment {
        Segment { start: 0.0, end: 0.0, text: t.into() }
    }

    #[test]
    fn placeholder_titles() {
        for t in ["", "  ", "30 sep. 10:51", "30 sep 10:51", "2 okt 2026 17:45", "Fredag 2 oktober kl. 09:30", "Oct 2, 2026 9:30 AM",
                  "2026-10-02 09:30", "Untitled", "Möte", "Klang-samtal 30 sep 2026 10:51", "10:51", "Idag 10:51"] {
            assert!(is_placeholder_title(t), "{t:?}");
        }
        for t in ["Veckomöte", "Möte med Ada 30 sep", "Budget 2027", "Maj-planering", "Sep 41 retro"] {
            assert!(!is_placeholder_title(t), "{t:?}");
        }
    }

    #[test]
    fn titles_from_summary_digest_transcript_date() {
        let ct = parse_ts(Some("2026-09-30T08:51:00Z")).unwrap();
        // first specific heading, markdown stripped
        let (t, src) = choose_title(Some("30 sep. 10:51"), Some("## Sammanfattning\n\n### Budget för **Q4** och [rekrytering](https://x.se)\n- punkt"), None, &[], ct);
        assert_eq!((t.as_str(), src), ("Budget för Q4 och rekrytering", "summary"));
        // only generic headings → first sentence of the first text
        let (t, _) = choose_title(None, Some("## Sammanfattning\nTeamet gick igenom *lanseringen* av appen. Beslut togs om datum.\n## Beslut\n- x"), None, &[], ct);
        assert_eq!(t, "Teamet gick igenom lanseringen av appen");
        // list item, checkbox, code, html
        let (t, _) = choose_title(None, Some("- [ ] `deploy` <b>servern</b> till produktion"), None, &[], ct);
        assert_eq!(t, "deploy servern till produktion");
        // abbreviations don't end the sentence; long text is cut at a word with …
        let (t, _) = choose_title(None, Some("Vi pratade t.ex. om hur kunderna upplever den nya onboardingen och vad som behöver förbättras innan release"), None, &[], ct);
        assert_eq!(t, "Vi pratade t.ex. om hur kunderna upplever den nya…");
        assert!(t.chars().count() <= 61);
        // a real Klang title wins
        assert_eq!(choose_title(Some("  Kundmöte  Acme "), Some("# Annat"), None, &[], ct), ("Kundmöte Acme".into(), "klang"));
        // digest, then transcript without timestamps / speaker labels, then the date
        assert_eq!(choose_title(None, None, Some("Kort genomgång av veckan."), &[], ct).1, "digest");
        let segs = [seg("[00:00:01] Talare 1: Hej och välkomna till mötet."), seg("Talare 2: Tack!")];
        assert_eq!(choose_title(Some(""), None, None, &segs, ct), ("Hej och välkomna till mötet".into(), "transcript"));
        let segs = [seg("Ada Lovelace: Ja"), seg("Bob: precis, vi kör")];
        assert_eq!(choose_title(None, Some("  "), None, &segs, ct).0, "Ja precis, vi kör");
        let (t, src) = choose_title(None, None, None, &[], ct);
        assert!(t.starts_with("Klang-samtal 30 sep 2026") && src == "date", "{t}");
    }

    #[test]
    fn summary_as_plain_text() {
        let md = "## Beslut\n- Releasen flyttas **en vecka**\n  - delpunkt\n\n## Att göra\n- [ ] Cecilia: kolla `login`\n- [x] Klart\n1. Ett\n2) Två\n\n---\nSe [planen](https://ex.se/a_b) <img src=x onerror=alert(1)> och snake_case.";
        assert_eq!(md_to_plain(md), "Beslut\n• Releasen flyttas en vecka\n  • delpunkt\n\nAtt göra\n☐ Cecilia: kolla login\n☑ Klart\n1. Ett\n2. Två\n\nSe planen och snake_case.");
        assert_eq!(md_to_plain(""), "");
        assert_eq!(strip_inline_md("a < b > c, 3<4"), "a < b > c, 3<4");
    }

    #[tokio::test]
    async fn untitled_conversations_get_titles_and_old_imports_migrate_without_refetch() {
        use mock::conv;
        let convs = vec![
            conv("u1", "30 sep. 10:51", "ready", Some("2026-09-30T08:51:00Z"), "2026-09-30T09:00:00Z", "## Sammanfattning\nGenomgång av budgeten för 2027. Mer text.", "Talare 1: Hej."),
            conv("u2", "", "ready", Some("2026-09-30T12:00:00Z"), "2026-09-30T12:30:00Z", "", "[00:00:00] Talare 1: Vi ska prata om flytten till nya kontoret i november\n\n[00:00:09] Talare 2: Ja."),
        ];
        let mut convs = convs;
        convs[1]["digest"] = serde_json::Value::Null;
        let (m, base) = mock::start(mock::Mock { convs: convs.into(), page_size: 100, ..Default::default() }).await;
        let (_d, st) = store();
        let c = mock::client(&base);
        // simulate notes imported by v0.5.0: date-like Klang title kept, no title_source
        sync(&c, &st).await;
        for id in ["u1", "u2"] {
            let mut n = st.find_klang(id).unwrap();
            n.title = if id == "u1" { "30 sep. 10:51".into() } else { "Klang-samtal 30 sep 2026 14:00".into() };
            n.title_source = None;
            st.replace(n).unwrap();
        }
        let d0 = m.detail_calls.load(Ordering::SeqCst);
        let r = sync(&c, &st).await;
        assert_eq!((r.new, r.updated, r.unchanged), (0, 2, 0), "{r:?}");
        assert_eq!(m.detail_calls.load(Ordering::SeqCst), d0, "titles migrate from local data, no refetch");
        let u1 = st.find_klang("u1").unwrap();
        assert_eq!((u1.title.as_str(), u1.title_source.as_deref()), ("Genomgång av budgeten för 2027", Some("summary")));
        let u2 = st.find_klang("u2").unwrap();
        assert_eq!((u2.title.as_str(), u2.title_source.as_deref()), ("Vi ska prata om flytten till nya kontoret i november", Some("transcript")));
        // stable afterwards
        let r = sync(&c, &st).await;
        assert_eq!((r.updated, r.unchanged), (0, 2), "{r:?}");
        // a user rename survives both the shortcut and a real Klang change
        st.rename(&u1.id, "Budgetmöte").unwrap();
        assert_eq!(sync(&c, &st).await.updated, 0);
        {
            let mut v = m.convs.lock().unwrap();
            v[0]["summary"] = "## Ny rubrik från Klang".into();
            v[0]["updated_at"] = "2026-10-02T09:00:00Z".into();
        }
        let r = sync(&c, &st).await;
        assert_eq!(r.updated, 1, "{r:?}");
        let u1 = st.find_klang("u1").unwrap();
        assert_eq!((u1.title.as_str(), u1.title_source.as_deref(), u1.summary.as_deref()), ("Budgetmöte", Some("user"), Some("## Ny rubrik från Klang")));
    }

    #[test]
    fn swedish_messages() {
        let r = Report { new: 1, updated: 2, unchanged: 0, skipped: 3, deleted_here: 4, ..Default::default() }.finish(None);
        assert_eq!(r.message, "Klang: 1 ny, 2 uppdaterade, 0 oförändrade, 3 hoppades över.");
        let r = Report { new: 0, updated: 0, unchanged: 1, deleted_here: 4, ..Default::default() }.finish(None);
        assert_eq!(r.message, "Klang: 0 nya, 0 uppdaterade, 1 oförändrad.");
        let r = Report::default().finish(Some(Error::Auth));
        assert!(r.error.as_deref().unwrap().starts_with("Ogiltig Klang-nyckel"));
        assert!(Error::RateLimited(Some(3600)).to_string().contains("60 min"));
    }
}
