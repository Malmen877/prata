//! Vocabulary, detokenisation, words and segments for Klang Pianissimo.
//!
//! `vocab.txt` has one `<piece> <id>` per line (SentencePiece pieces, `▁` marks
//! a word start, `<blk>` is the TDT blank). Text is built like onnx-asr: pieces
//! are concatenated with `▁` as a space, the leading space is dropped and a space
//! is only kept when a word character follows it (so no space before `.` or `,`).

use anyhow::{bail, Context, Result};

/// Seconds per encoder frame: 10 ms feature hop x subsampling factor 8.
pub const FRAME_SECS: f64 = 0.08;

pub struct Vocab {
    pieces: Vec<String>,
    pub blank: u32,
}

impl Vocab {
    pub fn parse(s: &str) -> Result<Vocab> {
        let mut pairs = vec![];
        for (n, line) in s.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let (piece, id) = line.rsplit_once(' ').with_context(|| format!("vocab line {}: {line:?}", n + 1))?;
            pairs.push((id.parse::<usize>().with_context(|| format!("vocab line {}", n + 1))?, piece.replace('\u{2581}', " ")));
        }
        let size = pairs.iter().map(|p| p.0 + 1).max().unwrap_or(0);
        let mut pieces = vec![String::new(); size];
        for (id, p) in pairs {
            pieces[id] = p;
        }
        let Some(blank) = pieces.iter().position(|p| p == "<blk>") else { bail!("vocab has no <blk> token") };
        Ok(Vocab { pieces, blank: blank as u32 })
    }

    /// Number of token logits including blank.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn piece(&self, id: u32) -> &str {
        self.pieces.get(id as usize).map(String::as_str).unwrap_or("")
    }
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// onnx-asr's `re.sub(r"\A\s|\s\B|(\s)\b", ...)`: drop leading whitespace and any
/// whitespace not followed by a word character; other whitespace becomes " ".
pub fn detok_join(raw: &str) -> String {
    let chars: Vec<char> = raw.chars().collect();
    let mut out = String::with_capacity(raw.len());
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            if i == 0 {
                continue;
            }
            match chars.get(i + 1) {
                Some(&n) if is_word_char(n) => out.push(' '),
                _ => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// A token on the original timeline (seconds).
#[derive(Debug, Clone, PartialEq)]
pub struct TimedTok {
    pub id: u32,
    pub start: f64,
    pub end: f64,
}

/// A word: its tokens (first one starts with a space, except at the very start).
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub toks: Vec<TimedTok>,
}

impl Word {
    pub fn start(&self) -> f64 {
        self.toks.first().map(|t| t.start).unwrap_or(0.0)
    }
    pub fn end(&self) -> f64 {
        self.toks.last().map(|t| t.end).unwrap_or(0.0)
    }
}

/// Group tokens into words: a piece starting with a space starts a new word;
/// pieces without one (sub-words, punctuation) attach to the previous word.
pub fn words(v: &Vocab, toks: &[TimedTok]) -> Vec<Word> {
    let mut out: Vec<Word> = vec![];
    for t in toks {
        let starts = v.piece(t.id).starts_with(' ');
        match out.last_mut() {
            Some(w) if !starts => w.toks.push(t.clone()),
            _ => out.push(Word { toks: vec![t.clone()] }),
        }
    }
    out
}

pub fn text_of(v: &Vocab, toks: impl IntoIterator<Item = u32>) -> String {
    let raw: String = toks.into_iter().map(|id| v.piece(id)).collect();
    detok_join(&raw)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Segmentation limits.
pub struct SegOpts {
    /// A pause at least this long between words always ends a segment.
    pub max_gap: f64,
    /// Segments longer than this are split (at a comma if there is one, else at
    /// the longest pause).
    pub max_len: f64,
}

impl Default for SegOpts {
    fn default() -> Self {
        Self { max_gap: 1.5, max_len: 20.0 }
    }
}

fn ends_sentence(v: &Vocab, w: &Word) -> bool {
    let last = w.toks.last().map(|t| v.piece(t.id)).unwrap_or("");
    let s = last.trim_end_matches(['"', '\u{201d}', ')', '\'']);
    s.ends_with('.') || s.ends_with('?') || s.ends_with('!') || s.ends_with('\u{2026}')
}

fn ends_clause(v: &Vocab, w: &Word) -> bool {
    let last = w.toks.last().map(|t| v.piece(t.id)).unwrap_or("");
    last.ends_with(',') || last.ends_with(';') || last.ends_with(':')
}

/// Group words into segments: one per sentence (the model writes punctuation),
/// also split at long pauses and when a segment would exceed `max_len`.
pub fn segments(v: &Vocab, words: &[Word], opts: &SegOpts) -> Vec<Segment> {
    let mut groups: Vec<Vec<&Word>> = vec![];
    let mut cur: Vec<&Word> = vec![];
    for (i, w) in words.iter().enumerate() {
        if let Some(prev) = cur.last() {
            if w.start() - prev.end() >= opts.max_gap {
                groups.push(std::mem::take(&mut cur));
            }
        }
        cur.push(w);
        let next_starts_upper = words.get(i + 1).map(|n| {
            text_of(v, n.toks.iter().map(|t| t.id)).chars().next().map(|c| !c.is_lowercase()).unwrap_or(true)
        });
        if ends_sentence(v, w) && next_starts_upper.unwrap_or(true) {
            groups.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    // split overlong groups
    let mut out: Vec<Vec<&Word>> = vec![];
    let mut stack: Vec<Vec<&Word>> = groups.into_iter().rev().collect();
    while let Some(g) = stack.pop() {
        let dur = g.last().unwrap().end() - g.first().unwrap().start();
        if dur <= opts.max_len || g.len() < 2 {
            out.push(g);
            continue;
        }
        // best cut after word k (1..len-1): prefer a clause end near the middle,
        // else the longest pause
        let mid = g.first().unwrap().start() + dur / 2.0;
        let score = |k: usize| -> f64 {
            let gap = g[k].start() - g[k - 1].end();
            let clause = if ends_clause(v, g[k - 1]) { 1.0 } else { 0.0 };
            clause * 2.0 + gap - (g[k].start() - mid).abs() / dur
        };
        let k = (1..g.len()).max_by(|&a, &b| score(a).total_cmp(&score(b))).unwrap();
        let (a, b) = g.split_at(k);
        stack.push(b.to_vec());
        stack.push(a.to_vec());
    }
    out.into_iter()
        .map(|g| Segment {
            start: g.first().unwrap().start(),
            end: g.last().unwrap().end(),
            text: text_of(v, g.iter().flat_map(|w| w.toks.iter().map(|t| t.id))).trim().to_string(),
        })
        .filter(|s| !s.text.is_empty())
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn tiny_vocab() -> Vocab {
        let pieces = [
            "<unk>", "\u{2581}Hej", "\u{2581}d", "\u{e5}", ".", ",", "\u{2581}Vi", "\u{2581}ses", "\u{2581}i", "\u{2581}morgon",
            "!", "\u{2581}och", "\u{2581}sen", "<blk>",
        ];
        let s: String = pieces.iter().enumerate().map(|(i, p)| format!("{p} {i}\n")).collect();
        Vocab::parse(&s).unwrap()
    }

    fn tt(id: u32, start: f64) -> TimedTok {
        TimedTok { id, start, end: start + 0.16 }
    }

    #[test]
    fn vocab_parses_blank_and_spaces() {
        let v = tiny_vocab();
        assert_eq!(v.blank, 13);
        assert_eq!(v.len(), 14);
        assert_eq!(v.piece(1), " Hej");
    }

    #[test]
    fn detok_matches_onnx_asr_rule() {
        assert_eq!(detok_join(" Hej d\u{e5}. Vi ses , i morgon !"), "Hej d\u{e5}. Vi ses, i morgon!");
        assert_eq!(detok_join(" 12 kilometer"), "12 kilometer");
        assert_eq!(detok_join(" a - b"), "a- b");
        let v = tiny_vocab();
        assert_eq!(text_of(&v, [1, 2, 3, 4, 6, 7, 5, 8, 9, 10]), "Hej d\u{e5}. Vi ses, i morgon!");
    }

    #[test]
    fn words_group_subwords_and_punctuation() {
        let v = tiny_vocab();
        let ws = words(&v, &[tt(1, 0.0), tt(2, 0.4), tt(3, 0.5), tt(4, 0.6), tt(6, 1.0)]);
        assert_eq!(ws.len(), 3);
        assert_eq!(ws[1].toks.len(), 3);
        assert!((ws[1].start() - 0.4).abs() < 1e-9 && (ws[1].end() - 0.76).abs() < 1e-9);
    }

    #[test]
    fn segments_split_at_sentences_pauses_and_length() {
        let v = tiny_vocab();
        // "Hej då. Vi ses, i morgon!" -> two sentences
        let toks = vec![tt(1, 0.0), tt(2, 0.3), tt(3, 0.4), tt(4, 0.5), tt(6, 1.0), tt(7, 1.3), tt(5, 1.4), tt(8, 1.6), tt(9, 1.8), tt(10, 2.1)];
        let segs = segments(&v, &words(&v, &toks), &SegOpts::default());
        assert_eq!(segs.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(), vec!["Hej d\u{e5}.", "Vi ses, i morgon!"]);
        assert!((segs[1].start - 1.0).abs() < 1e-9 && (segs[1].end - 2.26).abs() < 1e-9);
        // a long pause splits without punctuation
        let toks = vec![tt(6, 0.0), tt(7, 0.3), tt(11, 3.0), tt(12, 3.3)];
        let segs = segments(&v, &words(&v, &toks), &SegOpts::default());
        assert_eq!(segs.len(), 2);
        // an overlong sentence is split at the comma
        let toks = vec![tt(6, 0.0), tt(7, 5.0), tt(5, 5.1), tt(8, 10.0), tt(9, 15.0), tt(10, 15.2)];
        let segs = segments(&v, &words(&v, &toks), &SegOpts { max_gap: 100.0, max_len: 8.0 });
        assert_eq!(segs.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(), vec!["Vi ses,", "i morgon!"]);
    }
}
