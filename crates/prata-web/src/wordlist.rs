//! Ordlista: user-defined corrections applied to the text of new transcriptions.
//!
//! Each entry maps one correct form to known wrong forms, e.g. `OKDacke` <- `OK Dacke`.
//! Matching is case-insensitive and Unicode-aware (åäö are letters, so "Åsa" never matches
//! inside "Påsar"), a space in a wrong form matches any run of whitespace, the longest wrong
//! form wins at each position, and replaced text is never scanned again.
//!
//! Stored next to the notes directory as `wordlist.json` (default `~/.prata/wordlist.json`),
//! written atomically. API: `GET /api/wordlist`, `PUT /api/wordlist`,
//! `POST /api/notes/{id}/wordlist` (apply to an existing note).

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
};

use anyhow::{bail, Context, Result};
use axum::{
    extract::{Path as AxPath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::notes::{Segment, Store};
use crate::{err, St};

pub const MAX_ENTRIES: usize = 1000;
pub const MAX_WRONG: usize = 30;
pub const MAX_LEN: usize = 100;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    pub right: String,
    #[serde(default)]
    pub wrong: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct WordList {
    #[serde(default)]
    pub entries: Vec<Entry>,
}

/// `~/.prata/wordlist.json` for the default notes dir `~/.prata/notes`.
pub fn path_for(notes_dir: &Path) -> PathBuf {
    notes_dir.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")).join("wordlist.json")
}

/// Missing or unreadable file -> empty list (never fails a transcription).
pub fn load(path: &Path) -> WordList {
    match fs::read(path) {
        Ok(b) => serde_json::from_slice::<WordList>(&b)
            .ok()
            .and_then(|w| validate(w).ok())
            .unwrap_or_else(|| {
                eprintln!("[wordlist] could not read {} – ignoring it", path.display());
                WordList::default()
            }),
        Err(_) => WordList::default(),
    }
}

static WRITE: Mutex<()> = Mutex::new(());

pub fn save(path: &Path, w: &WordList) -> Result<()> {
    let _g = WRITE.lock().unwrap_or_else(|e| e.into_inner());
    let dir = path.parent().context("no parent dir")?;
    fs::create_dir_all(dir)?;
    let tmp = dir.join(".wordlist.json.tmp");
    {
        let mut f = fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(&serde_json::to_vec_pretty(w)?)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| format!("rename to {}", path.display()))?;
    if let Ok(d) = fs::File::open(dir) {
        let _ = d.sync_all();
    }
    Ok(())
}

fn clean(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Trim and collapse whitespace, drop empty/duplicate wrong forms, check limits.
/// Errors are Swedish (shown in the UI).
pub fn validate(w: WordList) -> Result<WordList> {
    if w.entries.len() > MAX_ENTRIES {
        bail!("Ordlistan får ha högst {MAX_ENTRIES} ord.");
    }
    let mut out = Vec::new();
    for e in w.entries {
        let right = clean(&e.right);
        let mut wrong: Vec<String> = Vec::new();
        for x in e.wrong.iter().map(|x| clean(x)) {
            if x.is_empty() || x == right || wrong.iter().any(|y| lower(y) == lower(&x)) {
                continue;
            }
            wrong.push(x);
        }
        if right.is_empty() && wrong.is_empty() {
            continue; // an empty row in the editor
        }
        if right.is_empty() {
            bail!("Ange rätt form för ”{}”.", wrong[0]);
        }
        if wrong.is_empty() {
            bail!("Ange minst en fel form för ”{right}”.");
        }
        if right.chars().count() > MAX_LEN || wrong.iter().any(|x| x.chars().count() > MAX_LEN) {
            bail!("Ett ord får vara högst {MAX_LEN} tecken.");
        }
        if wrong.len() > MAX_WRONG {
            bail!("”{right}” får ha högst {MAX_WRONG} fel former.");
        }
        if right.chars().chain(wrong.iter().flat_map(|x| x.chars())).any(|c| c.is_control()) {
            bail!("Ordlistan får inte innehålla kontrolltecken.");
        }
        out.push(Entry { right, wrong });
    }
    Ok(WordList { entries: out })
}

fn lower(s: &str) -> String {
    s.chars().flat_map(char::to_lowercase).collect()
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Compiled rules: (wrong form as lowercase chars, right form), longest first.
pub struct Rules(Vec<(Vec<char>, String)>);

impl Rules {
    pub fn new(w: &WordList) -> Rules {
        let mut v: Vec<(Vec<char>, String)> = w
            .entries
            .iter()
            .flat_map(|e| e.wrong.iter().map(move |x| (lower(&clean(x)).chars().collect::<Vec<_>>(), e.right.clone())))
            .filter(|(p, _)| !p.is_empty())
            .collect();
        v.sort_by_key(|r| std::cmp::Reverse(r.0.len()));
        Rules(v)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Length (in chars of `text`) of a match of `pat` starting at `i`, if any.
    fn match_at(text: &[char], low: &[Vec<char>], i: usize, pat: &[char]) -> Option<usize> {
        // word boundary before (only matters if the pattern starts with a word char)
        if is_word(pat[0]) && i > 0 && is_word(text[i - 1]) {
            return None;
        }
        let (mut ti, mut pi) = (i, 0);
        // compare the lowercase expansion of text chars with the (lowercase) pattern
        while pi < pat.len() {
            if ti >= text.len() {
                return None;
            }
            if pat[pi] == ' ' {
                if !text[ti].is_whitespace() {
                    return None;
                }
                while ti < text.len() && text[ti].is_whitespace() {
                    ti += 1;
                }
                pi += 1;
                continue;
            }
            let l = &low[ti];
            if pi + l.len() > pat.len() || pat[pi..pi + l.len()] != l[..] {
                return None;
            }
            pi += l.len();
            ti += 1;
        }
        // word boundary after
        if is_word(*pat.last().unwrap()) && ti < text.len() && is_word(text[ti]) {
            return None;
        }
        Some(ti - i)
    }

    /// Apply all rules to `s`. Returns the new text and the number of replacements.
    pub fn apply(&self, s: &str) -> (String, usize) {
        if self.0.is_empty() || s.is_empty() {
            return (s.to_string(), 0);
        }
        let text: Vec<char> = s.chars().collect();
        let low: Vec<Vec<char>> = text.iter().map(|c| c.to_lowercase().collect()).collect();
        let mut out = String::with_capacity(s.len());
        let (mut i, mut n) = (0, 0);
        'outer: while i < text.len() {
            for (pat, right) in &self.0 {
                if let Some(len) = Self::match_at(&text, &low, i, pat) {
                    out.push_str(right);
                    i += len;
                    n += 1;
                    continue 'outer;
                }
            }
            out.push(text[i]);
            i += 1;
        }
        (out, n)
    }

    /// Apply to every segment's text; returns the number of replacements.
    pub fn apply_segments(&self, segs: &mut [Segment]) -> usize {
        let mut n = 0;
        for s in segs.iter_mut() {
            let (t, k) = self.apply(&s.text);
            if k > 0 {
                s.text = t;
                n += k;
            }
        }
        n
    }
}

/// The job-finish hook: apply the saved word list to freshly transcribed segments.
pub fn apply_to_new(notes_dir: &Path, mut segs: Vec<Segment>) -> Vec<Segment> {
    let rules = Rules::new(&load(&path_for(notes_dir)));
    if !rules.is_empty() {
        let n = rules.apply_segments(&mut segs);
        if n > 0 {
            eprintln!("[wordlist] {n} replacement(s)");
        }
    }
    segs
}

// ---------------------------------------------------------------- HTTP handlers

pub async fn get_wordlist(State(st): State<St>) -> Response {
    let p = path_for(&st.cfg.notes_dir);
    Json(tokio::task::spawn_blocking(move || load(&p)).await.unwrap_or_default()).into_response()
}

pub async fn put_wordlist(State(st): State<St>, body: Option<Json<WordList>>) -> Response {
    let Some(Json(w)) = body else { return err(StatusCode::BAD_REQUEST, "förväntade JSON {\"entries\": [{\"right\": …, \"wrong\": […]}]}") };
    let w = match validate(w) {
        Ok(w) => w,
        Err(e) => return err(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let p = path_for(&st.cfg.notes_dir);
    let w2 = w.clone();
    match tokio::task::spawn_blocking(move || save(&p, &w2)).await {
        Ok(Ok(())) => Json(w).into_response(),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("kunde inte spara ordlistan: {e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// Apply the word list to a stored note (segments in note.json; audio untouched).
/// `None` if the note doesn't exist, else the number of replacements.
pub fn apply_to_stored(store: &Store, notes_dir: &Path, id: &str) -> Result<Option<usize>> {
    let Some(mut n) = store.get(id) else { return Ok(None) };
    let k = Rules::new(&load(&path_for(notes_dir))).apply_segments(&mut n.segments);
    if k > 0 {
        store.replace(n)?;
    }
    Ok(Some(k))
}

pub async fn apply_to_note(State(st): State<St>, AxPath(id): AxPath<String>) -> Response {
    let st2 = st.clone();
    match tokio::task::spawn_blocking(move || apply_to_stored(&st2.notes, &st2.cfg.notes_dir, &id)).await {
        Ok(Ok(Some(k))) => Json(serde_json::json!({ "replaced": k })).into_response(),
        Ok(Ok(None)) => err(StatusCode::NOT_FOUND, "okänd anteckning"),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("kunde inte spara: {e:#}")),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(e: &[(&str, &[&str])]) -> Rules {
        Rules::new(&WordList { entries: e.iter().map(|(r, w)| Entry { right: r.to_string(), wrong: w.iter().map(|s| s.to_string()).collect() }).collect() })
    }
    fn ap(r: &Rules, s: &str) -> String {
        r.apply(s).0
    }

    #[test]
    fn multi_word_case_insensitive() {
        let r = rules(&[("OKDacke", &["OK Dacke", "okej dacke"])]);
        assert_eq!(r.apply("Vi spelar mot ok dacke i kväll."), ("Vi spelar mot OKDacke i kväll.".into(), 1));
        assert_eq!(ap(&r, "OK  Dacke och Okej\nDacke"), "OKDacke och OKDacke");
        assert_eq!(ap(&r, "OK Dackes match"), "OK Dackes match"); // no match inside a longer word
        assert_eq!(ap(&r, "BOK Dacke"), "BOK Dacke");
    }

    #[test]
    fn swedish_letters_are_word_chars() {
        let r = rules(&[("Åsa", &["osa"]), ("Malmgren", &["malm gren", "malmgrén"])]);
        assert_eq!(ap(&r, "påosa osa"), "påosa Åsa"); // å is a letter: no boundary before "osa"
        assert_eq!(ap(&r, "osaå"), "osaå");
        assert_eq!(ap(&r, "Kevin Malm Gren och MALMGRÉN."), "Kevin Malmgren och Malmgren.");
        let r = rules(&[("Öckerö", &["ökerö"])]);
        assert_eq!(ap(&r, "ÖKERÖ, ökerö! Ökeröbor"), "Öckerö, Öckerö! Ökeröbor");
    }

    #[test]
    fn longest_first_and_no_re_replacement() {
        let r = rules(&[("New York", &["ny york"]), ("NY", &["ny"])]);
        assert_eq!(ap(&r, "ny york ny"), "New York NY");
        // the right form contains a wrong form of another rule: not replaced again
        let r = rules(&[("Anna Ek", &["ana ek"]), ("Annika", &["anna"])]);
        assert_eq!(ap(&r, "ana ek och anna"), "Anna Ek och Annika");
        // a rule whose right form contains its own wrong form does not loop
        let r = rules(&[("Dackeklubben", &["dacke"])]);
        assert_eq!(ap(&r, "dacke dacke"), "Dackeklubben Dackeklubben");
    }

    #[test]
    fn punctuation_and_non_word_edges() {
        let r = rules(&[("C++", &["c plus plus"]), ("SVT", &["s v t"])]);
        assert_eq!(ap(&r, "(c plus plus), s v t."), "(C++), SVT.");
        let r = rules(&[("e-post", &["e - post"])]);
        assert_eq!(ap(&r, "skicka e - post"), "skicka e-post");
    }

    #[test]
    fn empty_and_unchanged() {
        let r = rules(&[]);
        assert_eq!(r.apply("hej"), ("hej".into(), 0));
        let r = rules(&[("X", &["y"])]);
        assert_eq!(r.apply(""), ("".into(), 0));
        assert_eq!(r.apply("inget här"), ("inget här".into(), 0));
    }

    #[test]
    fn validation() {
        let w = validate(WordList { entries: vec![
            Entry { right: "  OKDacke ".into(), wrong: vec!["OK  Dacke".into(), "ok dacke".into(), "".into(), "OKDacke".into()] },
            Entry { right: "".into(), wrong: vec![" ".into()] },
        ] }).unwrap();
        assert_eq!(w.entries, vec![Entry { right: "OKDacke".into(), wrong: vec!["OK Dacke".into()] }]);
        // a case-only fix is allowed
        assert!(validate(WordList { entries: vec![Entry { right: "iPhone".into(), wrong: vec!["iphone".into()] }] }).is_ok());
        assert!(validate(WordList { entries: vec![Entry { right: "X".into(), wrong: vec![] }] }).is_err());
        assert!(validate(WordList { entries: vec![Entry { right: "".into(), wrong: vec!["y".into()] }] }).is_err());
        assert!(validate(WordList { entries: vec![Entry { right: "a\u{7}".into(), wrong: vec!["b".into()] }] }).is_err());
        assert!(validate(WordList { entries: vec![Entry { right: "x".repeat(101), wrong: vec!["b".into()] }] }).is_err());
    }

    #[test]
    fn save_load_roundtrip_and_path() {
        let d = tempfile::tempdir().unwrap();
        assert_eq!(path_for(&d.path().join("notes")), d.path().join("wordlist.json"));
        let p = path_for(&d.path().join("notes"));
        assert_eq!(load(&p), WordList::default());
        let w = WordList { entries: vec![Entry { right: "OKDacke".into(), wrong: vec!["OK Dacke".into()] }] };
        save(&p, &w).unwrap();
        assert_eq!(load(&p), w);
        fs::write(&p, b"{broken").unwrap();
        assert_eq!(load(&p), WordList::default());
        let mut segs = vec![Segment { start: 0.0, end: 1.0, text: " mot ok dacke".into() }];
        save(&p, &w).unwrap();
        segs = apply_to_new(&d.path().join("notes"), segs);
        assert_eq!(segs[0].text, " mot OKDacke");
    }

    #[test]
    fn apply_to_existing_note() {
        let d = tempfile::tempdir().unwrap();
        let notes_dir = d.path().join("notes");
        let store = Store::open(&notes_dir).unwrap();
        let n = crate::notes::Note {
            id: "abc123".into(),
            title: "t".into(),
            segments: vec![Segment { start: 0.0, end: 1.0, text: "hej ok dacke".into() }],
            ..Default::default()
        };
        store.create(n, None).unwrap();
        assert_eq!(apply_to_stored(&store, &notes_dir, "abc123").unwrap(), Some(0)); // no list yet
        save(&path_for(&notes_dir), &WordList { entries: vec![Entry { right: "OKDacke".into(), wrong: vec!["ok dacke".into()] }] }).unwrap();
        assert_eq!(apply_to_stored(&store, &notes_dir, "abc123").unwrap(), Some(1));
        assert_eq!(store.get("abc123").unwrap().segments[0].text, "hej OKDacke");
        assert_eq!(Store::open(&notes_dir).unwrap().get("abc123").unwrap().segments[0].text, "hej OKDacke"); // on disk
        assert_eq!(apply_to_stored(&store, &notes_dir, "nope00").unwrap(), None);
    }
}
