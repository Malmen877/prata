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

/// Features for one piece of audio: `(data, n_frames, valid_frames)` with `data`
/// laid out as `[N_MELS][n_frames]` (the encoder's `audio_signal` of shape
/// `(1, 128, n_frames)`) and `valid_frames` its `length` input.
pub fn features(pcm: &[f32]) -> (Vec<f32>, usize, usize) {
    let n = pcm.len();
    let n_frames = n / HOP + 1;
    let valid = n / HOP;
    // pre-emphasis, centred padding of N_FFT/2 zeros on both sides
    let pad = N_FFT / 2;
    let mut x = vec![0f64; n + 2 * pad];
    for i in 0..n {
        let prev = if i > 0 { pcm[i - 1] as f64 } else { 0.0 };
        x[pad + i] = pcm[i] as f64 - PREEMPH * prev;
    }
    // symmetric Hann (numpy.hanning) of WIN samples, centred in N_FFT
    let off = (N_FFT - WIN) / 2;
    let mut window = vec![0f64; N_FFT];
    for k in 0..WIN {
        window[off + k] = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * k as f64 / (WIN - 1) as f64).cos();
    }
    let fb = fbank();
    let mut logmel = vec![0f32; N_MELS * n_frames]; // [mel][frame]
    let (mut re, mut im) = (vec![0f64; N_FFT], vec![0f64; N_FFT]);
    let mut power = vec![0f32; N_BINS];
    for f in 0..n_frames {
        let s = f * HOP;
        for k in 0..N_FFT {
            re[k] = x[s + k] * window[k];
            im[k] = 0.0;
        }
        fft(&mut re, &mut im);
        for k in 0..N_BINS {
            let mag = (re[k] * re[k] + im[k] * im[k]).sqrt() as f32;
            power[k] = mag * mag;
        }
        for (m, (lo, w)) in fb.iter().enumerate() {
            let acc: f32 = power[*lo..*lo + w.len()].iter().zip(w).map(|(p, w)| p * w).sum();
            logmel[m * n_frames + f] = (acc + LOG_GUARD).ln();
        }
    }
    // per-feature normalisation over the valid frames; padding frames are zero
    for m in 0..N_MELS {
        let row = &mut logmel[m * n_frames..(m + 1) * n_frames];
        if valid == 0 {
            row.iter_mut().for_each(|v| *v = 0.0);
            continue;
        }
        let mean = row[..valid].iter().map(|&v| v as f64).sum::<f64>() / valid as f64;
        let var = row[..valid].iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (valid.max(2) - 1) as f64;
        let denom = var.sqrt() + 1e-5;
        for (i, v) in row.iter_mut().enumerate() {
            *v = if i < valid { ((*v as f64 - mean) / denom) as f32 } else { 0.0 };
        }
    }
    (logmel, n_frames, valid)
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
