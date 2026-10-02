//! Downloading media from a link: a direct HTTP(S) download with SSRF protection,
//! or yt-dlp for web pages (YouTube, SVT Play, podcasts …).
//!
//! Security model for the direct path:
//! - only `http`/`https`, no credentials in the URL, at most [`MAX_REDIRECTS`] redirects,
//!   every hop is validated again;
//! - every address a host name resolves to must be a public unicast address
//!   ([`ip_blocked`]); the check happens inside the HTTP client's DNS resolver, so the
//!   address that was checked is the address that is connected to (no DNS rebinding);
//! - IP-literal hosts are checked before the request (they bypass DNS);
//! - no proxies from the environment, size limit enforced while streaming, overall timeout.
//!
//! yt-dlp gets the URL as a single argv element after `--` (never through a shell), with
//! `--ignore-config`, no plugins, playlists off, size/duration filters and an output
//! template inside the job's own temp folder. The host of the link is checked with the
//! same classifier before yt-dlp runs; requests yt-dlp makes on its own (to the site's
//! CDN) are outside our control.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

pub const MAX_REDIRECTS: usize = 5;
pub const MAX_URL_LEN: usize = 4096;

/// Errors with a message meant for the user (Swedish).
#[derive(Debug, Clone, PartialEq)]
pub enum FetchError {
    InvalidUrl,
    Blocked,
    DnsFailed,
    TooManyRedirects,
    Http(u16),
    TooLarge { max_mb: u64 },
    TooLong { max_s: u64 },
    Timeout { secs: u64 },
    NotMedia,
    ToolMissing,
    Unsupported,
    Private,
    Unavailable,
    Live,
    Network(String),
    Other(String),
}

impl FetchError {
    pub fn message(&self) -> String {
        match self {
            FetchError::InvalidUrl => "Ogiltig länk. Klistra in en hel adress som börjar med https:// (eller http://).".into(),
            FetchError::Blocked => "Länken pekar på en lokal eller privat adress. Sådana hämtas inte av säkerhetsskäl.".into(),
            FetchError::DnsFailed => "Hittade inte servern i länken. Kontrollera adressen.".into(),
            FetchError::TooManyRedirects => "Länken skickar vidare för många gånger (fler än 5 omdirigeringar).".into(),
            FetchError::Http(c) => format!("Servern svarade med fel {c}. Kontrollera att länken fungerar i webbläsaren."),
            FetchError::TooLarge { max_mb } => format!("Filen är för stor. Gränsen är {max_mb} MB."),
            FetchError::TooLong { max_s } => format!("Ljudet är för långt. Gränsen är {}.", human_dur(*max_s)),
            FetchError::Timeout { secs } => format!("Nedladdningen tog för lång tid (mer än {}) och avbröts.", human_dur(*secs)),
            FetchError::NotMedia => "Länken pekar inte på en ljud- eller videofil. För webbsidor som YouTube behövs yt-dlp (brew install yt-dlp).".into(),
            FetchError::ToolMissing => "Den här länken är en webbsida och kräver yt-dlp, som inte är installerat. Installera med: brew install yt-dlp – eller använd en direktlänk till en ljud- eller videofil.".into(),
            FetchError::Unsupported => "Sidan stöds inte – hittade inget ljud eller video på länken.".into(),
            FetchError::Private => "Videon är privat eller kräver inloggning, så den kan inte hämtas.".into(),
            FetchError::Unavailable => "Videon eller ljudet är inte tillgängligt (borttaget eller spärrat).".into(),
            FetchError::Live => "Direktsändningar kan inte transkriberas. Vänta tills sändningen är slut.".into(),
            FetchError::Network(e) => format!("Nätverksfel vid nedladdningen: {e}"),
            FetchError::Other(e) => format!("Kunde inte hämta länken: {e}"),
        }
    }
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}
impl std::error::Error for FetchError {}

pub fn human_dur(s: u64) -> String {
    if s >= 3600 && s % 3600 == 0 {
        format!("{} h", s / 3600)
    } else if s >= 3600 {
        format!("{} h {} min", s / 3600, s % 3600 / 60)
    } else if s >= 60 && s % 60 == 0 {
        format!("{} min", s / 60)
    } else if s >= 60 {
        format!("{} min {} s", s / 60, s % 60)
    } else {
        format!("{s} s")
    }
}

/// Parse "3h", "90m", "600s", "1h30m" or plain seconds.
pub fn parse_duration(s: &str) -> Option<u64> {
    let s = s.trim().to_ascii_lowercase();
    if let Ok(n) = s.parse::<u64>() {
        return Some(n);
    }
    let (mut total, mut num) = (0u64, String::new());
    for c in s.chars() {
        if c.is_ascii_digit() {
            num.push(c);
        } else {
            let n: u64 = num.parse().ok()?;
            num.clear();
            total += n * match c {
                'h' => 3600,
                'm' => 60,
                's' => 1,
                _ => return None,
            };
        }
    }
    num.is_empty().then_some(total).filter(|t| *t > 0)
}

#[derive(Clone, Debug)]
pub struct Limits {
    pub max_bytes: u64,
    pub max_duration_s: u64,
    pub timeout: Duration,
}

impl Limits {
    pub fn max_mb(&self) -> u64 {
        self.max_bytes / (1024 * 1024)
    }
}

// ---------------------------------------------------------------- address policy

/// True for every address that must not be fetched: loopback, private (RFC 1918),
/// CGNAT/Tailscale (100.64/10), link-local, unique-local, multicast, unspecified,
/// broadcast, documentation/benchmark/reserved ranges, and IPv6 forms that embed
/// such an IPv4 address (IPv4-mapped/-compatible, NAT64, 6to4) or tunnel (Teredo).
/// For IPv6 only global unicast (2000::/3) is allowed at all.
pub fn ip_blocked(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4_blocked(v4),
        IpAddr::V6(v6) => v6_blocked(v6),
    }
}

fn v4_blocked(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    let n = u32::from(ip);
    let in_net = |base: [u8; 4], bits: u32| {
        let mask = if bits == 0 { 0 } else { u32::MAX << (32 - bits) };
        n & mask == u32::from(Ipv4Addr::from(base)) & mask
    };
    o[0] == 0                                   // 0.0.0.0/8 "this network" (incl. unspecified)
        || o[0] == 10                           // private
        || o[0] == 127                          // loopback
        || in_net([100, 64, 0, 0], 10)          // CGNAT, Tailscale
        || in_net([169, 254, 0, 0], 16)         // link-local (cloud metadata!)
        || in_net([172, 16, 0, 0], 12)          // private
        || in_net([192, 0, 0, 0], 24)           // IETF protocol assignments
        || in_net([192, 0, 2, 0], 24)           // TEST-NET-1
        || in_net([192, 88, 99, 0], 24)         // 6to4 relay anycast
        || in_net([192, 168, 0, 0], 16)         // private
        || in_net([198, 18, 0, 0], 15)          // benchmarking
        || in_net([198, 51, 100, 0], 24)        // TEST-NET-2
        || in_net([203, 0, 113, 0], 24)         // TEST-NET-3
        || o[0] >= 224                          // multicast 224/4, reserved 240/4, broadcast
}

fn v6_blocked(ip: Ipv6Addr) -> bool {
    let s = ip.segments();
    let embedded_v4 = |hi: u16, lo: u16| Ipv4Addr::new((hi >> 8) as u8, hi as u8, (lo >> 8) as u8, lo as u8);
    if let Some(v4) = ip.to_ipv4_mapped() {
        return v4_blocked(v4); // ::ffff:a.b.c.d
    }
    if s[..6] == [0; 6] {
        return true; // ::, ::1 and the deprecated IPv4-compatible ::a.b.c.d
    }
    if s[0] == 0x64 && s[1] == 0xff9b {
        // NAT64: 64:ff9b::/96 well-known (check the embedded IPv4), 64:ff9b:1::/48 local-use
        return s[2] != 0 || s[3] != 0 || s[4] != 0 || s[5] != 0 || v4_blocked(embedded_v4(s[6], s[7]));
    }
    if s[0] & 0xe000 != 0x2000 {
        return true; // not global unicast: ULA fc00::/7, link-local fe80::/10, multicast ff00::/8, discard 100::/64 …
    }
    if s[0] == 0x2002 {
        return v4_blocked(embedded_v4(s[1], s[2])); // 6to4
    }
    if s[0] == 0x2001 && s[1] == 0 {
        return true; // Teredo 2001::/32
    }
    if s[0] == 0x2001 && s[1] & 0xfff0 == 0x0010 {
        return true; // ORCHID 2001:10::/28
    }
    if s[0] == 0x2001 && s[1] == 0x0db8 {
        return true; // documentation
    }
    if s[0] == 0x2001 && s[1] < 0x0200 && s[1] != 0 {
        // 2001::/23 IETF protocol assignments (except a few anycast services; block all)
        return true;
    }
    false
}

/// Which addresses may be fetched. Production code can only obtain the strict policy.
#[derive(Clone, Copy, Debug)]
pub struct NetPolicy {
    allow_loopback: bool,
}

impl NetPolicy {
    pub const fn strict() -> Self {
        NetPolicy { allow_loopback: false }
    }

    /// For automated tests against a local mock server: additionally allows 127.0.0.0/8 and ::1
    /// (still blocks every other private range). Only exists in test builds or when the binary
    /// was compiled with the `insecure-test-loopback` feature, which release builds never enable.
    #[cfg(any(test, feature = "insecure-test-loopback"))]
    pub const fn allow_loopback_for_tests() -> Self {
        NetPolicy { allow_loopback: true }
    }

    pub fn allows_loopback(&self) -> bool {
        self.allow_loopback
    }

    pub fn ip_ok(&self, ip: IpAddr) -> bool {
        let ip = match ip {
            IpAddr::V6(v6) => v6.to_ipv4_mapped().map(IpAddr::V4).unwrap_or(ip),
            v4 => v4,
        };
        if self.allow_loopback && ip.is_loopback() {
            return true;
        }
        !ip_blocked(ip)
    }
}

/// Syntactic validation of a pasted link.
pub fn parse_url(s: &str) -> Result<url::Url, FetchError> {
    let s = s.trim();
    if s.is_empty() || s.len() > MAX_URL_LEN || s.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(FetchError::InvalidUrl);
    }
    let u = url::Url::parse(s).map_err(|_| FetchError::InvalidUrl)?;
    check_url_shape(&u)?;
    Ok(u)
}

fn check_url_shape(u: &url::Url) -> Result<(), FetchError> {
    if !matches!(u.scheme(), "http" | "https") {
        return Err(FetchError::InvalidUrl);
    }
    if !u.username().is_empty() || u.password().is_some() {
        return Err(FetchError::InvalidUrl); // user:pass@host is a classic disguise
    }
    match u.host() {
        None => Err(FetchError::InvalidUrl),
        Some(url::Host::Domain(d)) if d.is_empty() => Err(FetchError::InvalidUrl),
        _ => Ok(()),
    }
}

/// Resolve the host of `u` and check every address. IP literals are checked directly.
pub async fn check_host(u: &url::Url, policy: NetPolicy) -> Result<Vec<SocketAddr>, FetchError> {
    let port = u.port_or_known_default().ok_or(FetchError::InvalidUrl)?;
    let addrs: Vec<SocketAddr> = match u.host() {
        Some(url::Host::Ipv4(ip)) => vec![SocketAddr::new(IpAddr::V4(ip), port)],
        Some(url::Host::Ipv6(ip)) => vec![SocketAddr::new(IpAddr::V6(ip), port)],
        Some(url::Host::Domain(d)) => {
            let d = d.trim_end_matches('.');
            if d.eq_ignore_ascii_case("localhost") || d.to_ascii_lowercase().ends_with(".localhost") {
                if !policy.allows_loopback() {
                    return Err(FetchError::Blocked);
                }
            }
            tokio::time::timeout(Duration::from_secs(10), tokio::net::lookup_host((d, port)))
                .await
                .map_err(|_| FetchError::DnsFailed)?
                .map_err(|_| FetchError::DnsFailed)?
                .collect()
        }
        None => return Err(FetchError::InvalidUrl),
    };
    if addrs.is_empty() {
        return Err(FetchError::DnsFailed);
    }
    if addrs.iter().any(|a| !policy.ip_ok(a.ip())) {
        return Err(FetchError::Blocked);
    }
    Ok(addrs)
}

/// DNS resolver for the HTTP client: resolves, then refuses the whole name if any address
/// is blocked. The client connects only to the addresses returned here.
struct SafeResolver {
    policy: NetPolicy,
}

impl reqwest::dns::Resolve for SafeResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> reqwest::dns::Resolving {
        let policy = self.policy;
        let host = name.as_str().to_string();
        Box::pin(async move {
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((host.as_str(), 0)).await?.collect();
            if addrs.is_empty() {
                return Err("no addresses".into());
            }
            if addrs.iter().any(|a| !policy.ip_ok(a.ip())) {
                return Err(Box::new(FetchError::Blocked) as Box<dyn std::error::Error + Send + Sync>);
            }
            Ok(Box::new(addrs.into_iter()) as reqwest::dns::Addrs)
        })
    }
}

fn client(policy: NetPolicy) -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none()) // we follow redirects ourselves, checking each hop
        .no_proxy()
        .dns_resolver(Arc::new(SafeResolver { policy }))
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(60))
        .user_agent(concat!("Prata/", env!("CARGO_PKG_VERSION")))
        .build()
        .expect("http client")
}

fn map_reqwest(e: reqwest::Error) -> FetchError {
    // was it our resolver refusing the address?
    let mut src: Option<&(dyn std::error::Error + 'static)> = Some(&e);
    while let Some(s) = src {
        if let Some(f) = s.downcast_ref::<FetchError>() {
            return f.clone();
        }
        src = s.source();
    }
    if e.is_timeout() {
        return FetchError::Network("servern svarade inte i tid".into());
    }
    if e.is_connect() {
        let msg = format!("{e:#}");
        if msg.contains("dns error") || msg.contains("no addresses") || msg.contains("failed to lookup") {
            return FetchError::DnsFailed;
        }
        return FetchError::Network("kunde inte ansluta till servern".into());
    }
    FetchError::Network(e.to_string())
}

const MEDIA_EXTS: &[&str] = &[
    "mp3", "m4a", "mp4", "aac", "wav", "ogg", "oga", "opus", "flac", "webm", "mkv", "mov", "m4v", "wma", "aif",
    "aiff", "amr", "3gp", "mka", "caf",
];

fn ext_for_content_type(ct: &str) -> Option<&'static str> {
    Some(match ct {
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/mp4" | "audio/x-m4a" | "audio/m4a" | "audio/aac" => "m4a",
        "audio/wav" | "audio/x-wav" | "audio/wave" | "audio/vnd.wave" => "wav",
        "audio/ogg" | "application/ogg" => "ogg",
        "audio/opus" => "opus",
        "audio/flac" | "audio/x-flac" => "flac",
        "audio/webm" | "video/webm" => "webm",
        "video/mp4" | "application/mp4" => "mp4",
        "video/quicktime" => "mov",
        "video/x-matroska" | "audio/x-matroska" => "mkv",
        _ => return None,
    })
}

/// Result of a successful download.
#[derive(Debug)]
pub struct Downloaded {
    pub path: PathBuf,
    /// Media title (yt-dlp) if known.
    pub title: Option<String>,
    /// File name for display (last URL segment or title).
    pub filename: String,
    pub via: &'static str,
}

/// What the direct path found.
pub enum Direct {
    Media(Downloaded),
    /// Not a media file (HTML page etc.) or an HTTP error that yt-dlp might handle.
    NotMedia { status: Option<u16>, html: bool },
}

/// Direct HTTP(S) download into `dir`. Follows up to [`MAX_REDIRECTS`] redirects, re-checking
/// each hop. Calls `progress(done, total)` while streaming.
pub async fn direct_download(
    url: &url::Url,
    dir: &Path,
    limits: &Limits,
    policy: NetPolicy,
    mut progress: impl FnMut(u64, Option<u64>),
) -> Result<Direct, FetchError> {
    let c = client(policy);
    let mut cur = url.clone();
    let mut hops = 0;
    let mut resp = loop {
        check_url_shape(&cur)?;
        // IP literals never reach the resolver: check them here
        if let Some(ip) = match cur.host() {
            Some(url::Host::Ipv4(ip)) => Some(IpAddr::V4(ip)),
            Some(url::Host::Ipv6(ip)) => Some(IpAddr::V6(ip)),
            _ => None,
        } {
            if !policy.ip_ok(ip) {
                return Err(FetchError::Blocked);
            }
        } else if let Some(url::Host::Domain(d)) = cur.host() {
            let d = d.trim_end_matches('.').to_ascii_lowercase();
            if (d == "localhost" || d.ends_with(".localhost")) && !policy.allows_loopback() {
                return Err(FetchError::Blocked);
            }
        }
        let r = c.get(cur.clone()).header("Accept", "audio/*, video/*, */*;q=0.5").send().await.map_err(map_reqwest)?;
        if r.status().is_redirection() {
            let loc = r.headers().get(reqwest::header::LOCATION).and_then(|v| v.to_str().ok());
            let Some(loc) = loc else { return Err(FetchError::Http(r.status().as_u16())) };
            hops += 1;
            if hops > MAX_REDIRECTS {
                return Err(FetchError::TooManyRedirects);
            }
            cur = cur.join(loc).map_err(|_| FetchError::InvalidUrl)?;
            continue;
        }
        break r;
    };
    if !resp.status().is_success() {
        return Ok(Direct::NotMedia { status: Some(resp.status().as_u16()), html: false });
    }
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(';').next().unwrap_or("").trim().to_ascii_lowercase())
        .unwrap_or_default();
    let last_seg = cur
        .path_segments()
        .and_then(|mut s| s.next_back().map(|x| x.to_string()))
        .map(|s| percent_decode(&s))
        .unwrap_or_default();
    let url_ext = Path::new(&last_seg)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .filter(|e| MEDIA_EXTS.contains(&e.as_str()));
    let is_media = ct.starts_with("audio/")
        || ct.starts_with("video/")
        || ext_for_content_type(&ct).is_some()
        || (url_ext.is_some() && (ct.is_empty() || ct == "application/octet-stream" || ct == "binary/octet-stream"));
    // HLS/DASH manifests are left to yt-dlp
    if !is_media || ct.contains("mpegurl") || ct.contains("dash+xml") {
        return Ok(Direct::NotMedia { status: None, html: ct.contains("html") || ct.contains("mpegurl") || ct.contains("dash+xml") });
    }
    let resp_cd = resp.headers().get(reqwest::header::CONTENT_DISPOSITION).map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned());
    let total = resp.content_length();
    if let Some(t) = total {
        if t > limits.max_bytes {
            return Err(FetchError::TooLarge { max_mb: limits.max_mb() });
        }
    }
    let ext = url_ext
        .clone()
        .or_else(|| ext_for_content_type(&ct).map(String::from))
        .unwrap_or_else(|| "bin".into());
    let path = dir.join(format!("download.{ext}"));
    let mut f = tokio::fs::File::create(&path).await.map_err(|e| FetchError::Other(e.to_string()))?;
    let mut done = 0u64;
    progress(0, total);
    while let Some(chunk) = resp.chunk().await.map_err(map_reqwest)? {
        done += chunk.len() as u64;
        if done > limits.max_bytes {
            drop(f);
            let _ = tokio::fs::remove_file(&path).await;
            return Err(FetchError::TooLarge { max_mb: limits.max_mb() });
        }
        f.write_all(&chunk).await.map_err(|e| FetchError::Other(e.to_string()))?;
        progress(done, total);
    }
    f.flush().await.map_err(|e| FetchError::Other(e.to_string()))?;
    if done == 0 {
        return Err(FetchError::Other("filen var tom".into()));
    }
    let cd_name = resp_cd.as_deref().and_then(disposition_filename);
    let title = file_title(cd_name.as_deref().unwrap_or(&last_seg)).or_else(|| file_title(&last_seg));
    let filename = match cd_name.as_deref().map(basename).filter(|n| !n.is_empty()) {
        Some(n) => n.to_string(),
        None if last_seg.is_empty() => format!("{}.{ext}", cur.host_str().unwrap_or("länk")),
        None => last_seg,
    };
    Ok(Direct::Media(Downloaded { path, title, filename, via: "http" }))
}

/// `filename*=UTF-8''…` (preferred) or `filename="…"` from a Content-Disposition header.
fn disposition_filename(cd: &str) -> Option<String> {
    let mut plain = None;
    for part in cd.split(';').map(str::trim) {
        let Some((k, v)) = part.split_once('=') else { continue };
        let k = k.trim().to_ascii_lowercase();
        let v = v.trim();
        if k == "filename*" {
            // RFC 5987: charset'lang'percent-encoded
            let enc = v.splitn(3, '\'').nth(2).unwrap_or(v);
            let d = percent_decode(enc.trim_matches('"'));
            if !d.trim().is_empty() {
                return Some(d);
            }
        } else if k == "filename" {
            let v = v.strip_prefix('"').and_then(|x| x.strip_suffix('"')).unwrap_or(v).replace("\\\"", "\"");
            if !v.trim().is_empty() {
                plain = Some(if v.contains('%') { percent_decode(&v) } else { v });
            }
        }
    }
    plain
}

fn basename(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or("").trim()
}

/// Note title from a downloaded file's name: no directory, no extension, `_` as spaces,
/// control characters removed. `None` for empty or meaningless names (download, 1234, uuids …).
pub fn file_title(name: &str) -> Option<String> {
    let base = basename(name);
    let stem = match base.rfind('.') {
        Some(i) if base.len() - i <= 6 && base[i + 1..].chars().all(|c| c.is_ascii_alphanumeric()) => &base[..i],
        _ => base,
    };
    let t = crate::notes::clean_title(&stem.replace('_', " "))?;
    let t = t.trim_matches(|c: char| c == '-' || c == '.' || c.is_whitespace()).to_string();
    let lower = t.to_lowercase();
    const GENERIC: [&str; 16] = [
        "download", "file", "audio", "video", "media", "index", "stream", "play", "listen", "playback", "master",
        "default", "untitled", "track", "attachment", "fil",
    ];
    let hexish = t.len() >= 16 && t.chars().all(|c| c.is_ascii_hexdigit() || c == '-');
    if t.is_empty() || GENERIC.contains(&lower.as_str()) || t.chars().all(|c| c.is_ascii_digit() || c == '-' || c == ' ') || hexish {
        return None;
    }
    Some(t)
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 3 <= b.len() {
            if let Some(v) = s.get(i + 1..i + 3).and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---------------------------------------------------------------- yt-dlp

#[derive(Clone, Debug, serde::Serialize)]
pub struct YtDlp {
    pub path: PathBuf,
    pub version: String,
    #[serde(skip)]
    pub no_plugin_dirs: bool,
    /// `--print` implies `--quiet`; `--no-quiet` keeps the messages we classify errors from
    #[serde(skip)]
    pub no_quiet: bool,
}

/// Find yt-dlp (`$PRATA_YTDLP`, else `$PATH`) and check that it runs.
pub async fn probe_ytdlp(explicit: Option<PathBuf>) -> Option<YtDlp> {
    let path = explicit.or_else(|| crate::which("yt-dlp"))?;
    let run = |arg: &'static str| {
        let p = path.clone();
        async move {
            let o = tokio::time::timeout(
                Duration::from_secs(20),
                Command::new(&p).arg(arg).stdin(Stdio::null()).kill_on_drop(true).output(),
            )
            .await
            .ok()?
            .ok()?;
            o.status.success().then(|| String::from_utf8_lossy(&o.stdout).into_owned())
        }
    };
    let version = run("--version").await?.trim().to_string();
    let help = run("--help").await.unwrap_or_default();
    Some(YtDlp { no_plugin_dirs: help.contains("--no-plugin-dirs"), no_quiet: help.contains("--no-quiet"), path, version })
}

/// The yt-dlp argv (without the program). The URL comes last, after `--`.
pub fn ytdlp_args(y: &YtDlp, url: &str, dir: &Path, limits: &Limits) -> Vec<std::ffi::OsString> {
    let mut a: Vec<std::ffi::OsString> = Vec::new();
    let push = |a: &mut Vec<std::ffi::OsString>, s: &str| a.push(s.into());
    for s in [
        "--ignore-config",
        "--no-playlist",
        "--playlist-items", "1",
        "--no-exec",
        "--no-cache-dir",
        "--no-mtime",
        "--no-part",
        "--no-write-info-json",
        "--no-write-thumbnail",
        "--no-write-subs",
        "--no-color",
        "--restrict-filenames",
        "--socket-timeout", "30",
        "--retries", "3",
        "-f", "bestaudio/best",
        "--newline",
        "--progress",
        "--no-simulate",
        "--print", "before_dl:PRATA_TITLE %(extractor_key|)s\t%(playlist_title|)s\t%(title|)s",
        "--print", "after_move:PRATA_FILE %(filepath)s",
        "--progress-template", "download:PRATA_PROGRESS %(progress._percent_str)s",
    ] {
        push(&mut a, s);
    }
    if y.no_plugin_dirs {
        push(&mut a, "--no-plugin-dirs");
    }
    if y.no_quiet {
        push(&mut a, "--no-quiet");
    }
    push(&mut a, "--max-filesize");
    push(&mut a, &limits.max_bytes.to_string());
    push(&mut a, "--match-filters");
    push(&mut a, &format!("!is_live & duration <=? {}", limits.max_duration_s));
    push(&mut a, "-P");
    a.push(dir.as_os_str().to_owned());
    push(&mut a, "-o");
    push(&mut a, "media.%(ext)s");
    push(&mut a, "--");
    push(&mut a, url);
    a
}

/// "extractor\tplaylist_title\ttitle" → the best title. For a plain web page with an <audio>
/// element yt-dlp's generic/HTML5 extractors call the entry "Page title (1)"; drop that suffix.
fn pick_title(line: &str) -> Option<String> {
    let mut it = line.splitn(3, '\t');
    let (ex, pl, t) = (it.next().unwrap_or(""), it.next().unwrap_or("").trim(), it.next().unwrap_or("").trim());
    let clean = |s: &str| (!s.is_empty() && s != "NA").then(|| s.to_string());
    let mut t = t.to_string();
    if matches!(ex, "Generic" | "HTML5MediaEmbed") {
        if let Some(open) = t.rfind(" (") {
            let inner = &t[open + 2..];
            if inner.ends_with(')') && inner.len() > 1 && inner[..inner.len() - 1].chars().all(|c| c.is_ascii_digit()) {
                t.truncate(open);
            }
        }
    }
    clean(&t).or_else(|| clean(pl))
}

fn classify_ytdlp(log: &str, limits: &Limits) -> FetchError {
    let l = log.to_ascii_lowercase().replace("!is_live", "");
    if l.contains("larger than max-filesize") || l.contains("file is larger than") {
        FetchError::TooLarge { max_mb: limits.max_mb() }
    } else if l.contains("is live") || l.contains("live event") || l.contains("premieres in") || l.contains("is_live") {
        FetchError::Live
    } else if l.contains("does not pass filter") {
        FetchError::TooLong { max_s: limits.max_duration_s }
    } else if l.contains("unsupported url") {
        FetchError::Unsupported
    } else if l.contains("private video")
        || l.contains("video is private")
        || l.contains("sign in")
        || l.contains("login")
        || l.contains("log in")
        || l.contains("members-only")
        || l.contains("members only")
        || l.contains("authentication")
        || l.contains("cookies")
    {
        FetchError::Private
    } else if l.contains("video unavailable")
        || l.contains("not available")
        || l.contains("has been removed")
        || l.contains("http error 404")
        || l.contains("http error 410")
        || l.contains("geo")
    {
        FetchError::Unavailable
    } else if l.contains("timed out") {
        FetchError::Network("servern svarade inte i tid".into())
    } else if l.contains("no video formats") || l.contains("requested format is not available") {
        FetchError::Unsupported
    } else {
        let last = log.lines().rev().find(|x| x.contains("ERROR")).unwrap_or("okänt fel från yt-dlp");
        let msg: String = last.replace("ERROR:", "").trim().chars().take(240).collect();
        FetchError::Other(msg)
    }
}

/// Download with yt-dlp into `dir` (must exist). `progress(pct)` gets 0..100.
pub async fn ytdlp_download(
    y: &YtDlp,
    url: &url::Url,
    dir: &Path,
    limits: &Limits,
    mut progress: impl FnMut(f64),
) -> Result<Downloaded, FetchError> {
    let mut cmd = Command::new(&y.path);
    cmd.args(ytdlp_args(y, url.as_str(), dir, limits))
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd.spawn().map_err(|_| FetchError::ToolMissing)?;
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let err_task = tokio::spawn(async move {
        let mut buf = String::new();
        let mut lines = BufReader::new(err).lines();
        while let Ok(Some(l)) = lines.next_line().await {
            if buf.len() < 64 * 1024 {
                buf.push_str(&l);
                buf.push('\n');
            }
        }
        buf
    });
    let (mut title, mut file, mut log) = (None::<String>, None::<String>, String::new());
    let mut lines = BufReader::new(out).lines();
    while let Ok(Some(l)) = lines.next_line().await {
        if let Some(t) = l.strip_prefix("PRATA_TITLE ") {
            title = pick_title(t);
        } else if let Some(f) = l.strip_prefix("PRATA_FILE ") {
            file = Some(f.trim().to_string());
        } else if let Some(p) = l.strip_prefix("PRATA_PROGRESS ") {
            if let Ok(v) = p.trim().trim_end_matches('%').trim().parse::<f64>() {
                progress(v.clamp(0.0, 100.0));
            }
        } else if log.len() < 64 * 1024 {
            log.push_str(&l);
            log.push('\n');
        }
    }
    let status = child.wait().await.map_err(|e| FetchError::Other(e.to_string()))?;
    log.push_str(&err_task.await.unwrap_or_default());
    if !status.success() {
        return Err(classify_ytdlp(&log, limits));
    }
    let Some(file) = file else { return Err(classify_ytdlp(&log, limits)) };
    // The file must be inside the job folder (canonicalized, so ../ or symlinks can't escape).
    let canon_dir = std::fs::canonicalize(dir).map_err(|e| FetchError::Other(e.to_string()))?;
    let p = PathBuf::from(&file);
    let p = if p.is_absolute() { p } else { dir.join(p) };
    let canon = std::fs::canonicalize(&p).map_err(|_| FetchError::Other("yt-dlp gav ingen fil".into()))?;
    if !canon.starts_with(&canon_dir) || !canon.is_file() {
        return Err(FetchError::Other("yt-dlp skrev utanför jobbets mapp".into()));
    }
    let size = std::fs::metadata(&canon).map(|m| m.len()).unwrap_or(0);
    if size > limits.max_bytes {
        return Err(FetchError::TooLarge { max_mb: limits.max_mb() });
    }
    let ext = canon.extension().and_then(|e| e.to_str()).unwrap_or("bin").to_string();
    let filename = match &title {
        Some(t) => format!("{t}.{ext}"),
        None => format!("{}.{ext}", url.host_str().unwrap_or("länk")),
    };
    Ok(Downloaded { path: canon, title, filename, via: "yt-dlp" })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn b(s: &str) -> bool {
        ip_blocked(s.parse().unwrap())
    }

    #[test]
    fn blocks_non_public_addresses() {
        for s in [
            "0.0.0.0", "0.1.2.3", "127.0.0.1", "127.255.255.254", "10.0.0.1", "10.255.255.255", "172.16.0.1",
            "172.31.255.255", "192.168.1.1", "169.254.169.254", "100.64.0.1", "100.100.100.100", "100.127.255.255",
            "224.0.0.1", "239.255.255.250", "255.255.255.255", "240.0.0.1", "192.0.0.8", "192.0.2.1",
            "198.18.0.1", "198.51.100.7", "203.0.113.9", "::", "::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1",
            "::ffff:169.254.169.254", "::127.0.0.1", "fc00::1", "fd12:3456::1", "fe80::1", "fec0::1", "ff02::1",
            "64:ff9b::7f00:1", "64:ff9b::a00:1", "64:ff9b:1::1", "2002:7f00:1::1", "2002:c0a8:101::1",
            "2001::1", "2001:db8::1", "2001:10::1", "100::1",
        ] {
            assert!(b(s), "{s} must be blocked");
        }
    }

    #[test]
    fn allows_public_addresses() {
        for s in [
            "1.1.1.1", "8.8.8.8", "93.184.216.34", "172.15.255.255", "172.32.0.1", "100.63.255.255",
            "100.128.0.1", "192.169.0.1", "223.255.255.255", "2606:4700:4700::1111", "2a00:1450:4001::200e",
            "::ffff:8.8.8.8", "64:ff9b::808:808", "2002:808:808::1",
        ] {
            assert!(!b(s), "{s} must be allowed");
        }
    }

    #[test]
    fn test_policy_allows_only_loopback_extra() {
        let p = NetPolicy::allow_loopback_for_tests();
        assert!(p.ip_ok("127.0.0.1".parse().unwrap()) && p.ip_ok("::1".parse().unwrap()));
        assert!(p.ip_ok("::ffff:127.0.0.1".parse().unwrap()));
        assert!(!p.ip_ok("10.0.0.1".parse().unwrap()) && !p.ip_ok("169.254.169.254".parse().unwrap()));
        assert!(!NetPolicy::strict().ip_ok("127.0.0.1".parse().unwrap()));
    }

    #[test]
    fn url_validation() {
        for bad in [
            "", "   ", "ftp://example.com/a.mp3", "file:///etc/passwd", "javascript:alert(1)", "example.com/a.mp3",
            "https://user:pw@example.com/a.mp3", "https://exa mple.com/", "data:audio/mp3;base64,AA",
            "-o/tmp/x", "--exec=id", "http://", "gopher://example.com/",
        ] {
            assert_eq!(parse_url(bad).err(), Some(FetchError::InvalidUrl), "{bad:?}");
        }
        for good in ["https://example.com/a.mp3", "http://example.com:8080/x?y=1", " https://www.youtube.com/watch?v=abc \n"] {
            assert!(parse_url(good.trim()).is_ok(), "{good:?}");
        }
    }

    #[tokio::test]
    async fn check_host_rejects_private_targets() {
        for u in [
            "http://127.0.0.1/a.mp3", "http://localhost/a.mp3", "http://foo.localhost/a", "http://[::1]/a",
            "http://[::ffff:7f00:1]/a", "http://169.254.169.254/latest/meta-data/", "http://100.100.100.100/",
            "http://10.1.2.3:8795/", "http://0.0.0.0:8795/", "http://2130706433/", "http://0x7f.1/",
            "http://017700000001/",
        ] {
            let url = parse_url(u).unwrap();
            assert_eq!(check_host(&url, NetPolicy::strict()).await.err(), Some(FetchError::Blocked), "{u}");
        }
    }

    #[test]
    fn durations() {
        assert_eq!(parse_duration("10800"), Some(10800));
        assert_eq!(parse_duration("3h"), Some(10800));
        assert_eq!(parse_duration("1h30m"), Some(5400));
        assert_eq!(parse_duration("90m"), Some(5400));
        assert_eq!(parse_duration("15m"), Some(900));
        assert_eq!(parse_duration("abc"), None);
        assert_eq!(parse_duration("3x"), None);
        assert_eq!(human_dur(10800), "3 h");
        assert_eq!(human_dur(900), "15 min");
        assert_eq!(human_dur(5400), "1 h 30 min");
    }

    #[test]
    fn ytdlp_argv_is_safe() {
        let y = YtDlp { path: "yt-dlp".into(), version: "x".into(), no_plugin_dirs: true, no_quiet: true };
        let lim = Limits { max_bytes: 500 << 20, max_duration_s: 10800, timeout: Duration::from_secs(900) };
        let a: Vec<String> = ytdlp_args(&y, "--exec=touch /tmp/pwned", Path::new("/tmp/job"), &lim)
            .into_iter()
            .map(|s| s.into_string().unwrap())
            .collect();
        let n = a.len();
        assert_eq!(a[n - 2], "--");
        assert_eq!(a[n - 1], "--exec=touch /tmp/pwned"); // a single argv element after "--"
        for f in ["--ignore-config", "--no-playlist", "--no-exec", "--no-plugin-dirs", "--max-filesize"] {
            assert!(a.iter().any(|x| x == f), "{f}");
        }
        assert!(a.iter().any(|x| x == "!is_live & duration <=? 10800"));
        assert!(a.windows(2).any(|w| w[0] == "-P" && w[1] == "/tmp/job"));
    }

    #[test]
    fn titles_from_file_names() {
        assert_eq!(file_title("Avsnitt_12_R%C3%B6deby.mp3".replace("%C3%B6", "ö").as_str()).as_deref(), Some("Avsnitt 12 Rödeby"));
        assert_eq!(file_title("intervju med Ada.m4a").as_deref(), Some("intervju med Ada"));
        assert_eq!(file_title("../../etc/podd.v2.mp4").as_deref(), Some("podd.v2"));
        assert_eq!(file_title("C:\\x\\möte.wav").as_deref(), Some("möte"));
        assert_eq!(file_title("noext").as_deref(), Some("noext"));
        assert_eq!(file_title(".mp3"), None);
        for generic in ["download", "Download.mp3", "audio.m4a", "file", "12345.mp3", "", "  .wav", "3f2a9c1e4b5d6a7f8e9d.mp3", "550e8400-e29b-41d4-a716-446655440000.mp4"] {
            assert_eq!(file_title(generic), None, "{generic}");
        }
        assert_eq!(file_title("a\u{0}b\tc.mp3").as_deref(), Some("a b c"));
        assert_eq!(file_title(&format!("{}.mp3", "x".repeat(300))).map(|t| t.chars().count()), Some(200));
        assert_eq!(disposition_filename("attachment; filename=\"Mitt avsnitt.mp3\"").as_deref(), Some("Mitt avsnitt.mp3"));
        assert_eq!(disposition_filename("attachment; filename=plain.mp3").as_deref(), Some("plain.mp3"));
        assert_eq!(disposition_filename("attachment; filename=\"fallback.mp3\"; filename*=UTF-8''R%C3%B6deby%20podd.mp3").as_deref(), Some("Rödeby podd.mp3"));
        assert_eq!(disposition_filename("inline"), None);
    }

    #[test]
    fn titles_from_ytdlp() {
        assert_eq!(pick_title("Generic\tAvsnitt 12\tAvsnitt 12 (1)").as_deref(), Some("Avsnitt 12"));
        assert_eq!(pick_title("HTML5MediaEmbed\t\tRödeby (podd) (1)").as_deref(), Some("Rödeby (podd)"));
        assert_eq!(pick_title("Youtube\t\tMin video (1)").as_deref(), Some("Min video (1)"));
        assert_eq!(pick_title("Youtube\tNA\tEn titel").as_deref(), Some("En titel"));
        assert_eq!(pick_title("Generic\t\t").as_deref(), None);
    }

    #[test]
    fn ytdlp_errors_are_classified() {
        let lim = Limits { max_bytes: 500 << 20, max_duration_s: 10800, timeout: Duration::from_secs(900) };
        let c = |s: &str| classify_ytdlp(s, &lim);
        assert_eq!(c("ERROR: Unsupported URL: https://example.com/"), FetchError::Unsupported);
        assert_eq!(c("ERROR: [youtube] abc: Private video. Sign in if you've been granted access"), FetchError::Private);
        assert_eq!(c("ERROR: [youtube] abc: Video unavailable"), FetchError::Unavailable);
        assert_eq!(c("[info] abc: File is larger than max-filesize (600 bytes > 500 bytes). Aborting."), FetchError::TooLarge { max_mb: 500 });
        assert_eq!(c("[download] Video abc does not pass filter (!is_live & duration <=? 10800), skipping .."), FetchError::TooLong { max_s: 10800 });
        assert!(matches!(c("ERROR: something odd"), FetchError::Other(m) if m == "something odd"));
    }

    // ------------------------------------------------------------ against a local mock server

    use axum::{
        body::Body,
        extract::Path as AxPath,
        http::{header, StatusCode},
        response::{IntoResponse, Response},
        routing::get,
        Router,
    };

    /// 1 s of 16 kHz mono silence as WAV (no ffmpeg needed).
    pub(crate) fn wav_bytes(secs: f64) -> Vec<u8> {
        let n = (16000.0 * secs) as u32;
        let data = n * 2;
        let mut v = Vec::with_capacity(44 + data as usize);
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&(36 + data).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&16000u32.to_le_bytes());
        v.extend_from_slice(&32000u32.to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data.to_le_bytes());
        for i in 0..n {
            let x = ((i as f64 / 16000.0 * 440.0 * std::f64::consts::TAU).sin() * 3000.0) as i16;
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    fn redirect(to: String) -> Response {
        (StatusCode::FOUND, [(header::LOCATION, to)]).into_response()
    }

    /// Start the mock on 127.0.0.1:<random>; returns its base URL.
    pub(crate) async fn mock_server() -> String {
        let wav = wav_bytes(3.0);
        let wav2 = wav.clone();
        let (wav3, wav4) = (wav.clone(), wav.clone());
        let app = Router::new()
            .route("/dl", get(move || async move {
                ([(header::CONTENT_TYPE, "audio/wav".to_string()),
                  (header::CONTENT_DISPOSITION, "attachment; filename=\"x.wav\"; filename*=UTF-8''Intervju_med_R%C3%B6deby.wav".to_string())], wav3.clone())
            }))
            .route("/download.wav", get(move || async move { ([(header::CONTENT_TYPE, "audio/wav")], wav4.clone()) }))
            .route("/clip.wav", get(move || async move { ([(header::CONTENT_TYPE, "audio/wav")], wav.clone()) }))
            .route("/octet/klipp%20ett.wav", get(move || async move { ([(header::CONTENT_TYPE, "application/octet-stream")], wav2.clone()) }))
            .route("/r/{n}", get(|AxPath(n): AxPath<u32>| async move {
                redirect(if n == 0 { "/clip.wav".into() } else { format!("/r/{}", n - 1) })
            }))
            .route("/to-private", get(|| async { redirect("http://10.0.0.1/x.wav".into()) }))
            .route("/to-metadata", get(|| async { redirect("http://169.254.169.254/latest/meta-data/".into()) }))
            .route("/to-mapped", get(|| async { redirect("http://[::ffff:a9fe:a9fe]/x".into()) }))
            .route("/to-localhost", get(|| async { redirect("http://localhost/x.wav".into()) }))
            .route("/to-ftp", get(|| async { redirect("ftp://example.com/x.wav".into()) }))
            .route("/big.wav", get(|| async { ([(header::CONTENT_TYPE, "audio/wav")], vec![0u8; 3 << 20]) }))
            .route("/chunked.wav", get(|| async {
                let chunks = (0..48).map(|_| Ok::<_, std::io::Error>(vec![0u8; 64 * 1024]));
                ([(header::CONTENT_TYPE, "audio/wav")], Body::from_stream(futures_util::stream::iter(chunks)))
            }))
            .route("/page.html", get(|| async {
                ([(header::CONTENT_TYPE, "text/html; charset=utf-8")],
                 "<!doctype html><html><head><title>Testsida med ljud</title></head><body><audio controls src=\"/clip.wav\"></audio></body></html>")
            }))
            .route("/nomedia.html", get(|| async {
                ([(header::CONTENT_TYPE, "text/html")], "<html><head><title>Inget här</title></head><body>hej</body></html>")
            }))
            .route("/data.json", get(|| async { ([(header::CONTENT_TYPE, "application/json")], "{}") }))
            .route("/missing.wav", get(|| async { StatusCode::NOT_FOUND }))
            .route("/slow.wav", get(|| async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                ([(header::CONTENT_TYPE, "audio/wav")], wav_bytes(1.0))
            }));
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        format!("http://{addr}")
    }

    fn lim(mb: u64) -> Limits {
        Limits { max_bytes: mb << 20, max_duration_s: 10800, timeout: Duration::from_secs(60) }
    }

    async fn get_direct(url: &str, mb: u64, policy: NetPolicy) -> (Result<Direct, FetchError>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let u = parse_url(url).unwrap();
        let r = direct_download(&u, dir.path(), &lim(mb), policy, |_, _| {}).await;
        (r, dir)
    }

    #[tokio::test]
    async fn direct_download_from_mock() {
        let base = mock_server().await;
        let p = NetPolicy::allow_loopback_for_tests();
        let mut seen = vec![];
        let dir = tempfile::tempdir().unwrap();
        let u = parse_url(&format!("{base}/clip.wav")).unwrap();
        let r = direct_download(&u, dir.path(), &lim(10), p, |d, t| seen.push((d, t))).await.unwrap();
        let Direct::Media(d) = r else { panic!("not media") };
        assert_eq!(std::fs::read(&d.path).unwrap(), wav_bytes(3.0));
        assert_eq!((d.filename.as_str(), d.via), ("clip.wav", "http"));
        assert_eq!(d.title.as_deref(), Some("clip"), "title = file name without extension");
        assert!(d.path.starts_with(dir.path()));
        let total = wav_bytes(3.0).len() as u64;
        assert_eq!(seen.last(), Some(&(total, Some(total))), "progress reaches 100 %");

        // octet-stream with a media extension, percent-encoded name
        let (r, _d) = get_direct(&format!("{base}/octet/klipp%20ett.wav"), 10, p).await;
        assert!(matches!(r, Ok(Direct::Media(ref d)) if d.filename == "klipp ett.wav" && d.title.as_deref() == Some("klipp ett")));
        // Content-Disposition wins over the URL; generic names fall back to the default title
        let (r, _d) = get_direct(&format!("{base}/dl"), 10, p).await;
        assert!(matches!(r, Ok(Direct::Media(ref d)) if d.filename == "Intervju_med_Rödeby.wav" && d.title.as_deref() == Some("Intervju med Rödeby")));
        let (r, _d) = get_direct(&format!("{base}/download.wav"), 10, p).await;
        assert!(matches!(r, Ok(Direct::Media(ref d)) if d.title.is_none()));
        // redirects: the name of the final URL is used
        let (r, _d) = get_direct(&format!("{base}/r/1"), 10, p).await;
        assert!(matches!(r, Ok(Direct::Media(ref d)) if d.title.as_deref() == Some("clip")));
        // 5 redirects are fine, 6 are not
        let (r, _d) = get_direct(&format!("{base}/r/4"), 10, p).await;
        assert!(matches!(r, Ok(Direct::Media(_))));
        let (r, _d) = get_direct(&format!("{base}/r/5"), 10, p).await;
        assert_eq!(r.err(), Some(FetchError::TooManyRedirects));
        // pages and other content are not media
        let (r, _d) = get_direct(&format!("{base}/page.html"), 10, p).await;
        assert!(matches!(r, Ok(Direct::NotMedia { status: None, html: true })));
        let (r, _d) = get_direct(&format!("{base}/data.json"), 10, p).await;
        assert!(matches!(r, Ok(Direct::NotMedia { status: None, html: false })));
        let (r, _d) = get_direct(&format!("{base}/missing.wav"), 10, p).await;
        assert!(matches!(r, Ok(Direct::NotMedia { status: Some(404), .. })));
    }

    #[tokio::test]
    async fn size_limit_with_and_without_content_length() {
        let base = mock_server().await;
        let p = NetPolicy::allow_loopback_for_tests();
        let (r, d) = get_direct(&format!("{base}/big.wav"), 1, p).await;
        assert_eq!(r.err(), Some(FetchError::TooLarge { max_mb: 1 }));
        assert_eq!(std::fs::read_dir(d.path()).unwrap().count(), 0, "nothing left behind");
        let (r, d) = get_direct(&format!("{base}/chunked.wav"), 1, p).await;
        assert_eq!(r.err(), Some(FetchError::TooLarge { max_mb: 1 }));
        assert_eq!(std::fs::read_dir(d.path()).unwrap().count(), 0, "partial file removed");
        let (r, _d) = get_direct(&format!("{base}/chunked.wav"), 4, p).await;
        assert!(matches!(r, Ok(Direct::Media(_))));
    }

    #[tokio::test]
    async fn redirects_to_private_targets_are_blocked() {
        let base = mock_server().await;
        let p = NetPolicy::allow_loopback_for_tests(); // even the test policy blocks these
        for path in ["/to-private", "/to-metadata", "/to-mapped"] {
            let (r, _d) = get_direct(&format!("{base}{path}"), 10, p).await;
            assert_eq!(r.err(), Some(FetchError::Blocked), "{path}");
        }
        let (r, _d) = get_direct(&format!("{base}/to-ftp"), 10, p).await;
        assert_eq!(r.err(), Some(FetchError::InvalidUrl));
        // the strict (production) policy refuses the loopback mock itself, and a redirect to localhost
        let (r, _d) = get_direct(&format!("{base}/clip.wav"), 10, NetPolicy::strict()).await;
        assert_eq!(r.err(), Some(FetchError::Blocked));
        let port = base.rsplit(':').next().unwrap();
        let (r, _d) = get_direct(&format!("http://localhost:{port}/clip.wav"), 10, NetPolicy::strict()).await;
        assert_eq!(r.err(), Some(FetchError::Blocked));
    }

    #[tokio::test]
    async fn resolver_refuses_names_that_resolve_to_private_addresses() {
        use reqwest::dns::Resolve;
        use std::str::FromStr;
        let r = SafeResolver { policy: NetPolicy::strict() };
        let e = match r.resolve(reqwest::dns::Name::from_str("localhost").unwrap()).await {
            Ok(_) => panic!("localhost must not resolve"),
            Err(e) => e,
        };
        assert_eq!(e.downcast_ref::<FetchError>(), Some(&FetchError::Blocked));
        let ok = SafeResolver { policy: NetPolicy::allow_loopback_for_tests() };
        let addrs: Vec<_> = ok.resolve(reqwest::dns::Name::from_str("localhost").unwrap()).await.unwrap().collect();
        assert!(addrs.iter().all(|a| a.ip().is_loopback()));
    }

    /// Real yt-dlp (if installed) against the mock: an HTML page with an <audio> element.
    #[tokio::test]
    async fn ytdlp_against_mock_page() {
        let Some(y) = probe_ytdlp(None).await else {
            eprintln!("yt-dlp not installed – skipping");
            return;
        };
        let base = mock_server().await;
        let dir = tempfile::tempdir().unwrap();
        let u = parse_url(&format!("{base}/page.html")).unwrap();
        let mut pct = vec![];
        let d = ytdlp_download(&y, &u, dir.path(), &lim(10), |p| pct.push(p)).await.unwrap();
        assert_eq!(d.via, "yt-dlp");
        assert!(d.path.starts_with(std::fs::canonicalize(dir.path()).unwrap()));
        assert_eq!(std::fs::read(&d.path).unwrap(), wav_bytes(3.0));
        assert_eq!(d.title.as_deref(), Some("Testsida med ljud"), "title from the page");
        // too large for the limit
        let d2 = tempfile::tempdir().unwrap();
        let mut small = lim(1);
        small.max_bytes = 1000;
        let e = ytdlp_download(&y, &u, d2.path(), &small, |_| {}).await.unwrap_err();
        assert_eq!(e, FetchError::TooLarge { max_mb: 0 });
        // a page without media
        let u = parse_url(&format!("{base}/nomedia.html")).unwrap();
        let e = ytdlp_download(&y, &u, d2.path(), &lim(10), |_| {}).await.unwrap_err();
        assert_eq!(e, FetchError::Unsupported, "{e:?}");
    }
}
