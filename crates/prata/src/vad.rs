//! Energy-based voice activity detection and window planning.
//!
//! Works on 10 ms frames (160 samples at 16 kHz), the same grid as Whisper's mel
//! frames, so planned pieces can be cut directly out of the full-file mel
//! spectrogram (which keeps the global log-mel normalisation of the baseline).
//!
//! Pipeline: frame energy (dB) → adaptive threshold (noise floor + margin) →
//! speech runs → merge runs separated by short pauses (< `min_silence`) →
//! pad each region → pack regions into ≤ 30 s windows. Regions longer than a
//! window are split at the quietest frame near the window end. Each window keeps
//! an offset map (`pieces`) so decoded timestamps map back to the original audio.

pub const HOP: usize = 160; // samples per frame (10 ms)
pub const FRAMES_PER_SEC: f64 = 100.0;

#[derive(Debug, Clone)]
pub struct VadOpts {
    /// Pauses at least this long (seconds) are removed (after padding is added back).
    pub min_silence: f64,
    /// Audio kept before and after each speech region (seconds).
    pub pad: f64,
    /// Speech threshold in dB above the estimated noise floor; `None` = automatic.
    pub threshold_db: Option<f64>,
    /// Speech runs shorter than this (seconds) are ignored (clicks).
    pub min_speech: f64,
}

impl Default for VadOpts {
    fn default() -> Self {
        Self { min_silence: 1.0, pad: 0.3, threshold_db: None, min_speech: 0.05 }
    }
}

/// A contiguous run of original-audio frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub start: usize,
    pub len: usize,
}

impl Piece {
    pub fn end(&self) -> usize {
        self.start + self.len
    }
}

/// One decoder window: concatenation of pieces (≤ max frames in total).
#[derive(Debug, Clone, Default)]
pub struct Window {
    pub pieces: Vec<Piece>,
}

impl Window {
    pub fn frames(&self) -> usize {
        self.pieces.iter().map(|p| p.len).sum()
    }
    pub fn orig_start(&self) -> f64 {
        self.pieces.first().map(|p| p.start as f64 / FRAMES_PER_SEC).unwrap_or(0.0)
    }
    pub fn orig_end(&self) -> f64 {
        self.pieces.last().map(|p| p.end() as f64 / FRAMES_PER_SEC).unwrap_or(0.0)
    }

    /// Map a time inside the (concatenated) window to the original timeline.
    /// `is_end`: a time exactly on a piece boundary belongs to the earlier piece
    /// (segment ends), otherwise to the later piece (segment starts). Times past
    /// the content are clamped to the end of the last piece, so a segment can never
    /// start or end inside a removed gap.
    pub fn map(&self, t: f64, is_end: bool) -> f64 {
        let f = (t.max(0.0) * FRAMES_PER_SEC).round() as usize;
        let mut cum = 0usize;
        for (k, p) in self.pieces.iter().enumerate() {
            let last = k + 1 == self.pieces.len();
            let inside = if is_end { f <= cum + p.len } else { f < cum + p.len };
            if inside || last {
                let off = (f.saturating_sub(cum)).min(p.len);
                return (p.start + off) as f64 / FRAMES_PER_SEC;
            }
            cum += p.len;
        }
        0.0
    }
}

/// Per-frame energy in dB (25 ms analysis window, 10 ms hop).
pub fn frame_db(pcm: &[f32], n_frames: usize) -> Vec<f32> {
    const WIN: usize = 400;
    (0..n_frames)
        .map(|i| {
            let c = i * HOP + HOP / 2;
            let a = c.saturating_sub(WIN / 2).min(pcm.len());
            let b = (c + WIN / 2).min(pcm.len());
            if b <= a {
                return -100.0;
            }
            let e: f64 = pcm[a..b].iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / (b - a) as f64;
            (10.0 * (e + 1e-10).log10()) as f32
        })
        .collect()
}

fn percentile(v: &[f32], q: f64) -> f32 {
    if v.is_empty() {
        return -100.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[((s.len() - 1) as f64 * q).round() as usize]
}

/// Moving maximum over ±r frames (makes the detector tolerant of short dips inside words).
fn moving_max(v: &[f32], r: usize) -> Vec<f32> {
    (0..v.len())
        .map(|i| {
            let a = i.saturating_sub(r);
            let b = (i + r + 1).min(v.len());
            v[a..b].iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct VadReport {
    pub floor_db: f32,
    pub peak_db: f32,
    pub threshold_db: f32,
    pub fallback_all: bool,
}

/// Speech regions (frame ranges, padded and merged) of the first `n_frames` frames.
pub fn speech_regions(db: &[f32], opts: &VadOpts) -> (Vec<Piece>, VadReport) {
    let n = db.len();
    let floor = percentile(db, 0.10);
    let peak = percentile(db, 0.99);
    let range = peak - floor;
    let margin = opts.threshold_db.map(|t| t as f32).unwrap_or_else(|| (0.25 * range).clamp(8.0, 18.0));
    let thr = floor + margin;
    let mut report = VadReport { floor_db: floor, peak_db: peak, threshold_db: thr, fallback_all: false };
    // No usable dynamic range (constant noise / music / very short clip): keep everything.
    if n == 0 || (opts.threshold_db.is_none() && range < 10.0) {
        report.fallback_all = true;
        return (if n == 0 { vec![] } else { vec![Piece { start: 0, len: n }] }, report);
    }
    let sm = moving_max(db, 2);
    let min_speech = (opts.min_speech * FRAMES_PER_SEC).round() as usize;
    let min_sil = (opts.min_silence * FRAMES_PER_SEC).round() as usize;
    let pad = (opts.pad * FRAMES_PER_SEC).round() as usize;

    // 1. raw speech runs
    let mut runs: Vec<(usize, usize)> = vec![];
    let mut i = 0;
    while i < n {
        if sm[i] > thr {
            let s = i;
            while i < n && sm[i] > thr {
                i += 1;
            }
            if i - s >= min_speech {
                runs.push((s, i));
            }
        } else {
            i += 1;
        }
    }
    if runs.is_empty() {
        return (vec![], report);
    }
    // 2. pad, then merge regions whose remaining gap is shorter than min_silence
    let mut out: Vec<(usize, usize)> = vec![];
    for (s, e) in runs {
        let (s, e) = (s.saturating_sub(pad), (e + pad).min(n));
        if let Some(last) = out.last_mut() {
            if s <= last.1 || s - last.1 < min_sil.saturating_sub(2 * pad).max(1) {
                last.1 = last.1.max(e);
                continue;
            }
        }
        out.push((s, e));
    }
    (out.into_iter().map(|(s, e)| Piece { start: s, len: e - s }).collect(), report)
}

/// Split pieces longer than `max` frames at the quietest frame in the last
/// `search` frames before the limit (avoids cutting inside a word).
pub fn split_long(pieces: &[Piece], db: &[f32], max: usize, search: usize) -> Vec<Piece> {
    let sm: Vec<f32> = {
        // smooth over ±5 frames to find a real pause rather than a single quiet frame
        let r = 5usize;
        (0..db.len())
            .map(|i| {
                let a = i.saturating_sub(r);
                let b = (i + r + 1).min(db.len());
                db[a..b].iter().sum::<f32>() / (b - a) as f32
            })
            .collect()
    };
    let mut out = vec![];
    for p in pieces {
        let mut s = p.start;
        let e = p.end();
        while e - s > max {
            let lo = s + max - search.min(max - 1);
            let hi = s + max;
            let cut = (lo..hi)
                .min_by(|&a, &b| sm.get(a).unwrap_or(&0.0).total_cmp(sm.get(b).unwrap_or(&0.0)))
                .unwrap_or(hi);
            out.push(Piece { start: s, len: cut - s });
            s = cut;
        }
        if e > s {
            out.push(Piece { start: s, len: e - s });
        }
    }
    out
}

/// Greedily pack pieces (in time order) into windows of at most `max` frames.
/// Windows only ever break between pieces, i.e. at removed gaps or at split points.
pub fn pack(pieces: &[Piece], max: usize) -> Vec<Window> {
    let mut wins: Vec<Window> = vec![];
    let mut cur = Window::default();
    for &p in pieces {
        if !cur.pieces.is_empty() && cur.frames() + p.len > max {
            wins.push(std::mem::take(&mut cur));
        }
        cur.pieces.push(p);
    }
    if !cur.pieces.is_empty() {
        wins.push(cur);
    }
    wins
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(spec: &[(f64, bool)]) -> Vec<f32> {
        // (duration s, speech?) – speech = 200 Hz tone at -12 dBFS, silence = faint noise
        let mut v = vec![];
        let mut seed = 1u32;
        for &(d, sp) in spec {
            for i in 0..(d * 16000.0) as usize {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = ((seed >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.002;
                let tone = if sp { 0.25 * (i as f32 * 2.0 * std::f32::consts::PI * 200.0 / 16000.0).sin() } else { 0.0 };
                v.push(tone + noise);
            }
        }
        v
    }

    #[test]
    fn detects_regions_and_drops_long_pauses_only() {
        // 2 s speech, 0.4 s pause (kept), 2 s speech, 3 s pause (removed), 1 s speech
        let pcm = synth(&[(1.0, false), (2.0, true), (0.4, false), (2.0, true), (3.0, false), (1.0, true), (1.0, false)]);
        let n = pcm.len() / HOP;
        let db = frame_db(&pcm, n);
        let (r, rep) = speech_regions(&db, &VadOpts::default());
        assert!(!rep.fallback_all);
        assert_eq!(r.len(), 2, "{r:?}");
        // first region: 1.0-0.3 .. 5.4+0.3
        assert!((r[0].start as i64 - 70).abs() <= 3, "{r:?}");
        assert!((r[0].end() as i64 - 570).abs() <= 3, "{r:?}");
        assert!((r[1].start as i64 - 810).abs() <= 3, "{r:?}");
    }

    #[test]
    fn constant_signal_falls_back_to_everything() {
        let pcm = synth(&[(5.0, true)]);
        let db = frame_db(&pcm, pcm.len() / HOP);
        let (r, rep) = speech_regions(&db, &VadOpts::default());
        assert!(rep.fallback_all);
        assert_eq!(r, vec![Piece { start: 0, len: 500 }]);
    }

    #[test]
    fn packing_and_mapping() {
        let pieces = vec![Piece { start: 100, len: 1000 }, Piece { start: 1500, len: 1500 }, Piece { start: 4000, len: 1000 }];
        let w = pack(&pieces, 3000);
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].pieces.len(), 2);
        // 5.0 s into window 0 -> piece 0 -> 1.0 + 5.0 = 6.0 s
        assert!((w[0].map(5.0, false) - 6.0).abs() < 1e-9);
        // boundary at 10.0 s: as end -> end of piece 0 (11.0), as start -> start of piece 1 (15.0)
        assert!((w[0].map(10.0, true) - 11.0).abs() < 1e-9);
        assert!((w[0].map(10.0, false) - 15.0).abs() < 1e-9);
        // past content -> clamped to end of last piece (30.0)
        assert!((w[0].map(29.9, true) - 30.0).abs() < 1e-9);
        assert!((w[1].map(3.0, false) - 43.0).abs() < 1e-9);
    }

    #[test]
    fn split_long_regions_at_quiet_point() {
        let mut db = vec![-20f32; 7000];
        for d in db.iter_mut().take(2650).skip(2600) {
            *d = -60.0;
        }
        let p = split_long(&[Piece { start: 0, len: 7000 }], &db, 3000, 800);
        assert!(p.iter().all(|x| x.len <= 3000));
        assert_eq!(p.iter().map(|x| x.len).sum::<usize>(), 7000);
        assert!((2600..2650).contains(&p[0].len), "{p:?}");
    }
}
