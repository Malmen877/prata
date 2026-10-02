//! NeMo-style log-mel features for Klang Pianissimo (FastConformer).
//!
//! Matches onnx-asr's `NemoPreprocessorNumpy` (and `nemo128.onnx` in the
//! KlangAI/pianissimo-sv-onnx repo): 16 kHz input, pre-emphasis 0.97, 512-point
//! FFT over a 400-sample symmetric Hann window (zero-padded to 512), hop 160,
//! power spectrum, 128 Slaney mel bands, `ln(x + 2^-24)`, then per-feature
//! normalisation (mean, unbiased std + 1e-5) over the valid frames.

pub const SAMPLE_RATE: usize = 16_000;
pub const HOP: usize = 160;
pub const N_FFT: usize = 512;
pub const WIN: usize = 400;
pub const N_MELS: usize = 128;
const N_BINS: usize = N_FFT / 2 + 1;
const PREEMPH: f64 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24

/// Mel filterbank (257 FFT bins x 128 bands, row-major f32 LE), taken from
/// onnx-asr's `fbanks.npz["nemo128"]` (librosa Slaney mel, the same values as NeMo).
static FBANK: &[u8] = include_bytes!("nemo128fb.bytes");

/// The filterbank as one `(first_bin, weights)` run of non-zero weights per band.
fn fbank() -> Vec<(usize, Vec<f32>)> {
    let w = |k: usize, m: usize| {
        let i = (k * N_MELS + m) * 4;
        f32::from_le_bytes([FBANK[i], FBANK[i + 1], FBANK[i + 2], FBANK[i + 3]])
    };
    (0..N_MELS)
        .map(|m| {
            let lo = (0..N_BINS).find(|&k| w(k, m) != 0.0).unwrap_or(0);
            let hi = (0..N_BINS).rev().find(|&k| w(k, m) != 0.0).map(|k| k + 1).unwrap_or(lo);
            (lo, (lo..hi).map(|k| w(k, m)).collect())
        })
        .collect()
}

/// In-place iterative radix-2 FFT (n a power of two).
pub(crate) fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((ang * k as f64).cos(), (ang * k as f64).sin());
                let (a, b) = (start + k, start + k + len / 2);
                let (xr, xi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[b] = re[a] - xr;
                im[b] = im[a] - xi;
                re[a] += xr;
                im[a] += xi;
            }
        }
        len <<= 1;
    }
}

/// Raw log-mel (before normalisation) of frames `[from, to)` of `pcm`, laid out
/// as `[N_MELS][to - from]`. Frame `f` is centred on sample `f * HOP` of the whole
/// signal (zero beyond its ends), so frames cut from any range are identical to
/// the same frames of the full-signal computation.
pub fn logmel_frames(pcm: &[f32], from: usize, to: usize) -> Vec<f32> {
    let n = to.saturating_sub(from);
    let pad = (N_FFT / 2) as isize;
    // pre-emphasised sample i of the whole signal (0 outside)
    let x = |i: isize| -> f64 {
        if i < 0 || i as usize >= pcm.len() {
            return 0.0;
        }
        let i = i as usize;
        let prev = if i > 0 { pcm[i - 1] as f64 } else { 0.0 };
        pcm[i] as f64 - PREEMPH * prev
    };
    // symmetric Hann (numpy.hanning) of WIN samples, centred in N_FFT
    let off = (N_FFT - WIN) / 2;
    let mut window = vec![0f64; N_FFT];
    for k in 0..WIN {
        window[off + k] = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / (WIN - 1) as f64).cos();
    }
    let fb = fbank();
    let mut logmel = vec![0f32; N_MELS * n];
    let (mut re, mut im) = (vec![0f64; N_FFT], vec![0f64; N_FFT]);
    let mut power = vec![0f32; N_BINS];
    for j in 0..n {
        let s = ((from + j) * HOP) as isize - pad;
        for k in 0..N_FFT {
            re[k] = if window[k] == 0.0 { 0.0 } else { x(s + k as isize) * window[k] };
            im[k] = 0.0;
        }
        fft(&mut re, &mut im);
        for k in 0..N_BINS {
            let mag = (re[k] * re[k] + im[k] * im[k]).sqrt() as f32;
            power[k] = mag * mag;
        }
        for (m, (lo, w)) in fb.iter().enumerate() {
            let acc: f32 = power[*lo..*lo + w.len()].iter().zip(w).map(|(p, w)| p * w).sum();
            logmel[m * n + j] = (acc + LOG_GUARD).ln();
        }
    }
    logmel
}

/// Per-band normalisation statistics (mean and `std + 1e-5`).
#[derive(Debug, Clone)]
pub struct Stats {
    pub mean: Vec<f64>,
    pub denom: Vec<f64>,
}

impl Stats {
    /// Statistics over the first `valid` frames of a `[N_MELS][n]` log-mel block.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn of_block(logmel: &[f32], n: usize, valid: usize) -> Stats {
        let mut acc = Acc::default();
        acc.add(logmel, n, valid);
        acc.finish()
    }

    /// Statistics over all `pcm.len() / HOP` frames of a whole signal, computed in
    /// blocks so memory stays small for long recordings.
    pub fn of_signal(pcm: &[f32]) -> Stats {
        let valid = pcm.len() / HOP;
        let mut acc = Acc::default();
        let block = 6000; // 60 s
        let mut f = 0;
        while f < valid {
            let e = (f + block).min(valid);
            let lm = logmel_frames(pcm, f, e);
            acc.add(&lm, e - f, e - f);
            f = e;
        }
        acc.finish()
    }
}

/// Running per-band sums (f64; values are small, so sum of squares is exact enough).
#[derive(Default)]
struct Acc {
    n: usize,
    sum: Vec<f64>,
    sq: Vec<f64>,
}

impl Acc {
    fn add(&mut self, lm: &[f32], n: usize, valid: usize) {
        if self.sum.is_empty() {
            self.sum = vec![0.0; N_MELS];
            self.sq = vec![0.0; N_MELS];
        }
        for m in 0..N_MELS {
            for &v in &lm[m * n..m * n + valid] {
                self.sum[m] += v as f64;
                self.sq[m] += v as f64 * v as f64;
            }
        }
        self.n += valid;
    }

    fn finish(self) -> Stats {
        let cnt = self.n.max(1) as f64;
        if self.sum.is_empty() {
            return Stats { mean: vec![0.0; N_MELS], denom: vec![1e-5; N_MELS] };
        }
        let mean: Vec<f64> = self.sum.iter().map(|s| s / cnt).collect();
        let denom = (0..N_MELS)
            .map(|m| ((self.sq[m] - cnt * mean[m] * mean[m]).max(0.0) / (self.n.max(2) - 1) as f64).sqrt() + 1e-5)
            .collect();
        Stats { mean, denom }
    }
}

/// Normalise a `[N_MELS][n]` block in place; frames at or after `valid` become 0.
pub fn normalize(logmel: &mut [f32], n: usize, valid: usize, st: &Stats) {
    for m in 0..N_MELS {
        let row = &mut logmel[m * n..(m + 1) * n];
        for (i, v) in row.iter_mut().enumerate() {
            *v = if i < valid { ((*v as f64 - st.mean[m]) / st.denom[m]) as f32 } else { 0.0 };
        }
    }
}

/// Features for one whole piece of audio, exactly as onnx-asr computes them:
/// `(data, n_frames, valid_frames)` with `data` laid out as `[N_MELS][n_frames]`
/// (the encoder's `audio_signal` of shape `(1, 128, n_frames)`) and
/// `valid_frames` its `length` input. Statistics are taken over this piece.
#[cfg_attr(not(test), allow(dead_code))]
pub fn features(pcm: &[f32]) -> (Vec<f32>, usize, usize) {
    let valid = pcm.len() / HOP;
    let n_frames = valid + 1;
    let mut lm = logmel_frames(pcm, 0, n_frames);
    let st = Stats::of_block(&lm, n_frames, valid);
    normalize(&mut lm, n_frames, valid, &st);
    (lm, n_frames, valid)
}

/// Features for frames `[from, to)` of a longer signal, normalised with
/// statistics `st` of the whole signal (so each window sees exactly the features
/// a single pass over the whole recording would). When `to` reaches the end of
/// the signal one zero padding frame is added, as in `features`.
pub fn window_features(pcm: &[f32], from: usize, to: usize, st: &Stats) -> (Vec<f32>, usize, usize) {
    let total = pcm.len() / HOP;
    let to = to.min(total);
    let valid = to.saturating_sub(from);
    let n_frames = if to == total { valid + 1 } else { valid };
    let mut lm = logmel_frames(pcm, from, from + n_frames);
    normalize(&mut lm, n_frames, valid, st);
    (lm, n_frames, valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filterbank_size() {
        assert_eq!(FBANK.len(), N_BINS * N_MELS * 4);
    }

    #[test]
    fn fft_matches_dft() {
        let n = 16;
        let sig: Vec<f64> = (0..n).map(|i| ((i * 7 % 5) as f64 - 2.0) * 0.3 + (i as f64 * 0.4).sin()).collect();
        let (mut re, mut im) = (sig.clone(), vec![0.0; n]);
        fft(&mut re, &mut im);
        for k in 0..n {
            let (mut r, mut i) = (0.0, 0.0);
            for (t, &v) in sig.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                r += v * a.cos();
                i += v * a.sin();
            }
            assert!((re[k] - r).abs() < 1e-9 && (im[k] - i).abs() < 1e-9, "bin {k}");
        }
    }

    #[test]
    fn window_features_are_slices_of_the_whole_signal() {
        let pcm: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.013).sin() * 0.2 + ((i * 17 % 23) as f32 - 11.0) * 0.002).collect();
        let total = pcm.len() / HOP;
        let whole = logmel_frames(&pcm, 0, total + 1);
        let st = Stats::of_signal(&pcm);
        let st2 = Stats::of_block(&whole, total + 1, total);
        for m in 0..N_MELS {
            assert!((st.mean[m] - st2.mean[m]).abs() < 1e-9 && (st.denom[m] - st2.denom[m]).abs() < 1e-9);
        }
        let (w, n, valid) = window_features(&pcm, 100, 200, &st);
        assert_eq!((n, valid), (100, 100));
        let mut full = whole.clone();
        normalize(&mut full, total + 1, total, &st);
        for m in [0, 64, 127] {
            for j in 0..100 {
                let a = w[m * n + j];
                let b = full[m * (total + 1) + 100 + j];
                assert!((a - b).abs() < 1e-5, "mel {m} frame {j}: {a} vs {b}");
            }
        }
        // the last window gets the padding frame
        let (_, n, valid) = window_features(&pcm, 200, total, &st);
        assert_eq!((n, valid), (total - 200 + 1, total - 200));
    }

    #[test]
    fn shapes_and_normalisation() {
        let pcm: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.1 + ((i * 31 % 17) as f32 - 8.0) * 0.001).collect();
        let (f, t, valid) = features(&pcm);
        assert_eq!((t, valid), (101, 100));
        assert_eq!(f.len(), N_MELS * t);
        for m in [0, 40, 127] {
            let row = &f[m * t..m * t + valid];
            let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / valid as f64;
            assert!(mean.abs() < 1e-4, "mel {m} mean {mean}");
            assert_eq!(f[m * t + valid], 0.0); // padding frame
        }
    }
}
