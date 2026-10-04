//! Chunking for long audio.
//!
//! The encoder's memory grows with the input length (about 21 GB for 12.5 min in
//! one pass), so audio is always transcribed in windows. Boundaries are placed
//! at the quietest point (lowest 50 ms average energy) in a search range before
//! the nominal window length, so they fall in pauses rather than inside words.
//! Each window is decoded with `context` seconds of extra audio on both sides;
//! from each window only the words that *start* inside its own core
//! `[keep_from, keep_to)` are kept. The words near a boundary therefore come from
//! a window that saw them with context on both sides, and every word is taken
//! from exactly one window.

use super::text::Word;

#[derive(Debug, Clone)]
pub struct ChunkOpts {
    /// Nominal core length (seconds) of each window.
    pub core: f64,
    /// Extra audio decoded before and after the core (seconds).
    pub context: f64,
    /// Boundary search range before the nominal end (seconds).
    pub search: f64,
}

impl Default for ChunkOpts {
    fn default() -> Self {
        Self { core: 30.0, context: 5.0, search: 8.0 }
    }
}

/// One decoding window. Frames are 10 ms frames of the original audio.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    /// Audio decoded: frames `[start, end)`.
    pub start: usize,
    pub end: usize,
    /// Core: words starting in `[keep_from, keep_to)` (frames) are kept.
    pub keep_from: usize,
    pub keep_to: usize,
}

/// 50 ms moving average of the frame energies (dB) around frame `i`.
fn smooth(db: &[f32], i: usize) -> f32 {
    let lo = i.saturating_sub(2);
    let hi = (i + 3).min(db.len());
    db[lo..hi].iter().sum::<f32>() / (hi - lo).max(1) as f32
}

/// Plan windows over `total` frames with frame energies `db` (see `vad::frame_db`).
pub fn plan(total: usize, db: &[f32], opts: &ChunkOpts) -> Vec<Window> {
    let core = (opts.core * 100.0) as usize;
    let search = ((opts.search * 100.0) as usize).min(core / 2);
    let ctx = (opts.context * 100.0) as usize;
    let mut bounds = vec![0usize];
    let mut b = 0usize;
    while total - b > core + search / 2 {
        let (lo, hi) = (b + core - search, b + core);
        let cut = (lo..hi)
            .min_by(|&x, &y| {
                let (ex, ey) = if db.is_empty() { (0.0, 0.0) } else { (smooth(db, x.min(db.len() - 1)), smooth(db, y.min(db.len() - 1))) };
                // ties: the latest frame (longest core)
                ex.total_cmp(&ey).then(y.cmp(&x))
            })
            .unwrap_or(hi);
        bounds.push(cut);
        b = cut;
    }
    bounds.push(total);
    bounds
        .windows(2)
        .map(|w| Window { start: w[0].saturating_sub(ctx), end: (w[1] + ctx).min(total), keep_from: w[0], keep_to: w[1] })
        .collect()
}

/// Keep the words of one window that start inside its core.
pub fn keep_core(words: Vec<Word>, w: &Window, total: usize) -> Vec<Word> {
    let from = if w.keep_from == 0 { f64::NEG_INFINITY } else { w.keep_from as f64 / 100.0 };
    let to = if w.keep_to >= total { f64::INFINITY } else { w.keep_to as f64 / 100.0 };
    words.into_iter().filter(|wd| wd.start() >= from && wd.start() < to).collect()
}

/// Append the words of the next window, dropping words at the seam that are the
/// same spoken word seen by both windows: the cut can fall inside a word, so each
/// window may place that word's start on its own side of the cut. Decided on
/// timestamps only: the first new word is dropped while it overlaps the last kept
/// word by more than half of the shorter one. A real repetition ("Rödeby. Rödeby
/// är") is two words one after the other and does not overlap.
pub fn append_at_seam(words: &mut Vec<Word>, new: Vec<Word>) {
    let mut new = new.into_iter().peekable();
    while let (Some(last), Some(first)) = (words.last(), new.peek()) {
        let ov = last.end().min(first.end()) - last.start().max(first.start());
        let shorter = (last.end() - last.start()).min(first.end() - first.start()).max(1e-6);
        if ov > 0.5 * shorter {
            new.next();
        } else {
            break;
        }
    }
    words.extend(new);
}

#[cfg(test)]
mod tests {
    use super::super::text::{tests::tiny_vocab, words, TimedTok};
    use super::*;

    #[test]
    fn seam_drops_the_same_word_seen_by_both_windows_but_keeps_real_repeats() {
        let w = |id: u32, a: f64, b: f64| Word { toks: vec![TimedTok { id, start: a, end: b }] };
        // window A kept "gör" 51.96-52.28, window B kept "gör" 52.03-52.35 (cut at 51.99)
        let mut words = vec![w(1, 51.0, 51.5), w(2, 51.96, 52.28)];
        append_at_seam(&mut words, vec![w(2, 52.03, 52.35), w(3, 52.4, 52.6)]);
        assert_eq!(words.iter().map(|x| x.toks[0].id).collect::<Vec<_>>(), vec![1, 2, 3]);
        // a real repetition right after the cut is kept
        let mut words = vec![w(4, 10.0, 10.5)];
        append_at_seam(&mut words, vec![w(4, 10.6, 11.1)]);
        assert_eq!(words.len(), 2);
        let mut words = vec![];
        append_at_seam(&mut words, vec![w(5, 0.0, 0.3)]);
        assert_eq!(words.len(), 1);
    }

    #[test]
    fn short_audio_is_one_window() {
        let w = plan(1500, &vec![-40.0; 1500], &ChunkOpts::default());
        assert_eq!(w, vec![Window { start: 0, end: 1500, keep_from: 0, keep_to: 1500 }]);
    }

    #[test]
    fn boundaries_fall_in_the_quietest_place_and_cover_everything() {
        let total = 10_000; // 100 s
        let mut db = vec![-20.0f32; total];
        // pauses at 27 s and 55 s
        db[2690..2720].fill(-70.0);
        db[5480..5520].fill(-70.0);
        let ws = plan(total, &db, &ChunkOpts::default());
        assert!(ws.len() >= 3);
        assert!((2690..2720).contains(&ws[0].keep_to), "{:?}", ws[0]);
        assert!((5480..5520).contains(&ws[1].keep_to), "{:?}", ws[1]);
        assert_eq!(ws[0].keep_from, 0);
        assert_eq!(ws.last().unwrap().keep_to, total);
        for p in ws.windows(2) {
            assert_eq!(p[0].keep_to, p[1].keep_from);
            assert_eq!(p[1].start, p[1].keep_from - 500);
            assert_eq!(p[0].end, p[0].keep_to + 500);
        }
        for w in &ws {
            assert!(w.end - w.start <= 4000, "window too long: {w:?}");
        }
    }

    #[test]
    fn merge_takes_each_word_from_exactly_one_window() {
        let v = tiny_vocab();
        let t = |id: u32, s: f64| TimedTok { id, start: s, end: s + 0.1 };
        let total = 6000;
        let ws = [
            Window { start: 0, end: 3500, keep_from: 0, keep_to: 3000 },
            Window { start: 2500, end: 6000, keep_from: 3000, keep_to: 6000 },
        ];
        // both windows decode the overlap 25..35 s; "d"+"å" straddles nothing
        let a = words(&v, &[t(1, 1.0), t(6, 26.0), t(2, 29.9), t(3, 30.05), t(7, 31.0)]);
        let b = words(&v, &[t(6, 26.1), t(2, 29.95), t(3, 30.1), t(7, 31.05), t(8, 50.0)]);
        let mut all = keep_core(a, &ws[0], total);
        all.extend(keep_core(b, &ws[1], total));
        let ids: Vec<u32> = all.iter().flat_map(|w| w.toks.iter().map(|t| t.id)).collect();
        assert_eq!(ids, vec![1, 6, 2, 3, 7, 8]);
    }
}
