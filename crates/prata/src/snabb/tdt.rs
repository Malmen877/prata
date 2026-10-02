//! Greedy decoding for a Token-and-Duration Transducer (TDT).
//!
//! Same loop as onnx-asr's `_AsrWithTransducerDecoding._decoding` for
//! `NemoConformerTdt`: at encoder frame `t` the decoder/joint network is run with
//! the last emitted token (blank at the start) and its LSTM state. The joint
//! output holds `vocab_size` token logits (blank last) followed by duration
//! logits. The arg-max token is emitted unless it is blank (only then is the new
//! decoder state kept); the arg-max duration says how many frames to advance.
//! A duration of 0 stays on the frame, at most `max_symbols` times.

use anyhow::Result;

/// One emitted token: id, encoder frame where it was emitted, and the predicted
/// duration (frames) that followed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tok {
    pub id: u32,
    pub frame: usize,
    pub dur: usize,
}

/// The decoder + joint network for one step. `State` is the prediction
/// network's recurrent state.
pub trait Joint {
    type State: Clone;
    fn initial_state(&self) -> Self::State;
    /// Joint output for `enc` (one encoder frame) after `prev` (last emitted token,
    /// or blank). Returns the raw output (token logits then duration logits) and
    /// the prediction network's state after consuming `prev`.
    fn step(&mut self, enc: &[f32], prev: u32, state: &Self::State) -> Result<(Vec<f32>, Self::State)>;
}

pub struct TdtConfig {
    /// Number of token logits, including blank.
    pub vocab_size: usize,
    pub blank: u32,
    /// Frames advanced for each duration index (Parakeet / Pianissimo: 0..=4).
    pub durations: Vec<usize>,
    pub max_symbols: usize,
}

fn argmax(v: &[f32]) -> usize {
    // first maximum, like numpy.argmax
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Greedy TDT decoding of `n_frames` encoder frames; `frame(t)` returns frame `t`.
pub fn greedy<'a, J: Joint>(
    joint: &mut J,
    cfg: &TdtConfig,
    n_frames: usize,
    frame: impl Fn(usize) -> &'a [f32],
) -> Result<Vec<Tok>> {
    let mut state = joint.initial_state();
    let mut toks: Vec<Tok> = vec![];
    let mut t = 0usize;
    let mut emitted = 0usize;
    while t < n_frames {
        let prev = toks.last().map(|k| k.id).unwrap_or(cfg.blank);
        let (out, new_state) = joint.step(frame(t), prev, &state)?;
        let token = argmax(&out[..cfg.vocab_size]) as u32;
        let d_idx = argmax(&out[cfg.vocab_size..cfg.vocab_size + cfg.durations.len()]);
        let step = cfg.durations[d_idx];
        if token != cfg.blank {
            state = new_state;
            toks.push(Tok { id: token, frame: t, dur: step });
            emitted += 1;
        }
        if step > 0 {
            t += step;
            emitted = 0;
        } else if token == cfg.blank || emitted == cfg.max_symbols {
            t += 1;
            emitted = 0;
        }
    }
    Ok(toks)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scripted joint: frame value `f` (enc[0]) and previous token decide the output.
    struct Script {
        /// (frame, prev) -> (token, duration index)
        rules: Vec<((usize, u32), (u32, usize))>,
        calls: usize,
    }

    impl Joint for Script {
        type State = u32; // number of tokens consumed
        fn initial_state(&self) -> u32 {
            0
        }
        fn step(&mut self, enc: &[f32], prev: u32, state: &u32) -> Result<(Vec<f32>, u32)> {
            self.calls += 1;
            let f = enc[0] as usize;
            let (tok, d) = self.rules.iter().find(|(k, _)| *k == (f, prev)).map(|r| r.1).unwrap_or((4, 1));
            let mut out = vec![0f32; 5 + 5];
            out[tok as usize] = 1.0;
            out[5 + d] = 1.0;
            Ok((out, state + 1))
        }
    }

    fn cfg() -> TdtConfig {
        TdtConfig { vocab_size: 5, blank: 4, durations: vec![0, 1, 2, 3, 4], max_symbols: 3 }
    }

    #[test]
    fn durations_skip_frames_and_zero_duration_stays() {
        // frame 0: token 1 with duration 0 (stay), then token 2 with duration 2,
        // frame 2: blank with duration 3 -> frame 5: token 3 dur 1, frame 6: end
        let mut j = Script {
            rules: vec![((0, 4), (1, 0)), ((0, 1), (2, 2)), ((2, 2), (4, 3)), ((5, 2), (3, 1)), ((6, 3), (4, 1))],
            calls: 0,
        };
        let frames: Vec<Vec<f32>> = (0..7).map(|i| vec![i as f32]).collect();
        let toks = greedy(&mut j, &cfg(), 7, |t| &frames[t]).unwrap();
        assert_eq!(
            toks,
            vec![Tok { id: 1, frame: 0, dur: 0 }, Tok { id: 2, frame: 0, dur: 2 }, Tok { id: 3, frame: 5, dur: 1 }]
        );
        assert_eq!(j.calls, 5);
    }

    #[test]
    fn max_symbols_forces_advance() {
        // always token 0 with duration 0: at most 3 per frame
        struct Stuck;
        impl Joint for Stuck {
            type State = ();
            fn initial_state(&self) {}
            fn step(&mut self, _: &[f32], _: u32, _: &()) -> Result<(Vec<f32>, ())> {
                let mut o = vec![0f32; 10];
                o[0] = 1.0;
                o[5] = 1.0;
                Ok((o, ()))
            }
        }
        let frames = vec![vec![0f32]; 2];
        let toks = greedy(&mut Stuck, &cfg(), 2, |t| &frames[t]).unwrap();
        assert_eq!(toks.len(), 6);
        assert_eq!(toks.iter().filter(|t| t.frame == 1).count(), 3);
    }

    #[test]
    fn blank_with_zero_duration_advances_one() {
        struct Blank;
        impl Joint for Blank {
            type State = ();
            fn initial_state(&self) {}
            fn step(&mut self, _: &[f32], _: u32, _: &()) -> Result<(Vec<f32>, ())> {
                let mut o = vec![0f32; 10];
                o[4] = 1.0;
                o[5] = 1.0; // duration 0
                Ok((o, ()))
            }
        }
        let frames = vec![vec![0f32]; 4];
        assert!(greedy(&mut Blank, &cfg(), 4, |t| &frames[t]).unwrap().is_empty());
    }
}
