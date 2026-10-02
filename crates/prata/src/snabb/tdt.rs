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

fn log_softmax(v: &[f32]) -> Vec<f32> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
    v.iter().map(|x| x - lse).collect()
}

struct Hyp<S> {
    toks: Vec<Tok>,
    score: f32,
    /// prediction-network state before consuming the last token
    state: S,
    t: usize,
    /// symbols emitted on frame `t` so far
    emitted: usize,
}

/// Beam search for TDT (experimental, `PRATA_SNABB_BEAM`). Hypotheses carry
/// their own frame position; each round expands every active hypothesis by
/// blank (durations > 0) and the `beam` best tokens (all durations), merges
/// hypotheses with the same tokens on the same frame (log-sum-exp) and keeps
/// the `beam` best. A hypothesis is finished when it passes the last frame.
/// The result is the finished hypothesis with the best length-normalised
/// score (`norm`) or raw score.
pub fn beam<'a, J: Joint>(
    joint: &mut J,
    cfg: &TdtConfig,
    n_frames: usize,
    frame: impl Fn(usize) -> &'a [f32],
    beam: usize,
    norm: bool,
) -> Result<Vec<Tok>> {
    let beam = beam.max(1);
    let key = |h: &Hyp<J::State>| -> f32 { if norm { h.score / (h.toks.len().max(1) as f32) } else { h.score } };
    let mut active = vec![Hyp { toks: vec![], score: 0.0, state: joint.initial_state(), t: 0, emitted: 0 }];
    let mut done: Vec<Hyp<J::State>> = vec![];
    if n_frames == 0 {
        return Ok(vec![]);
    }
    while !active.is_empty() {
        let mut cand: Vec<Hyp<J::State>> = vec![];
        for h in &active {
            let prev = h.toks.last().map(|k| k.id).unwrap_or(cfg.blank);
            let (out, ns) = joint.step(frame(h.t), prev, &h.state)?;
            let lt = log_softmax(&out[..cfg.vocab_size]);
            let ld = log_softmax(&out[cfg.vocab_size..cfg.vocab_size + cfg.durations.len()]);
            // blank: only durations > 0 (renormalised over those)
            let pos: Vec<usize> = (0..cfg.durations.len()).filter(|&i| cfg.durations[i] > 0).collect();
            let lse = {
                let m = pos.iter().map(|&i| ld[i]).fold(f32::NEG_INFINITY, f32::max);
                m + pos.iter().map(|&i| (ld[i] - m).exp()).sum::<f32>().ln()
            };
            for &i in &pos {
                let d = cfg.durations[i];
                cand.push(Hyp {
                    toks: h.toks.clone(),
                    score: h.score + lt[cfg.blank as usize] + ld[i] - lse,
                    state: h.state.clone(),
                    t: h.t + d,
                    emitted: 0,
                });
            }
            let mut idx: Vec<usize> = (0..cfg.vocab_size).filter(|&k| k as u32 != cfg.blank).collect();
            idx.sort_by(|&a, &b| lt[b].partial_cmp(&lt[a]).unwrap_or(std::cmp::Ordering::Equal));
            for &k in idx.iter().take(beam) {
                for (i, &d) in cfg.durations.iter().enumerate() {
                    let stay = d == 0;
                    if stay && h.emitted + 1 >= cfg.max_symbols {
                        continue;
                    }
                    let mut toks = h.toks.clone();
                    toks.push(Tok { id: k as u32, frame: h.t, dur: d });
                    cand.push(Hyp {
                        toks,
                        score: h.score + lt[k] + ld[i],
                        state: ns.clone(),
                        t: h.t + d,
                        emitted: if stay { h.emitted + 1 } else { 0 },
                    });
                }
            }
        }
        // merge identical (tokens, frame)
        cand.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let mut merged: Vec<Hyp<J::State>> = vec![];
        for c in cand {
            let ids = |h: &Hyp<J::State>| h.toks.iter().map(|k| k.id).collect::<Vec<_>>();
            if let Some(m) = merged.iter_mut().find(|m| m.t == c.t && m.toks.len() == c.toks.len() && ids(m) == ids(&c)) {
                let (a, b) = (m.score.max(c.score), m.score.min(c.score));
                m.score = a + (b - a).exp().ln_1p();
            } else {
                merged.push(c);
            }
            if merged.len() >= beam * 4 {
                break;
            }
        }
        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(beam);
        active.clear();
        for h in merged {
            if h.t >= n_frames {
                done.push(h);
            } else {
                active.push(h);
            }
        }
        // scores only go down: stop expanding hypotheses already worse than the best
        // finished one (raw scores), keep the beam of finished ones bounded
        if !norm {
            if let Some(best) = done.iter().map(|h| h.score).reduce(f32::max) {
                active.retain(|h| h.score > best);
            }
        }
        if done.len() > beam * 4 {
            done.sort_by(|a, b| key(b).partial_cmp(&key(a)).unwrap_or(std::cmp::Ordering::Equal));
            done.truncate(beam);
        }
    }
    done.sort_by(|a, b| key(b).partial_cmp(&key(a)).unwrap_or(std::cmp::Ordering::Equal));
    Ok(done.into_iter().next().map(|h| h.toks).unwrap_or_default())
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
    fn beam_of_one_on_peaked_outputs_matches_greedy() {
        let rules = vec![((0, 4), (1, 0)), ((0, 1), (2, 2)), ((2, 2), (4, 3)), ((5, 2), (3, 1)), ((6, 3), (4, 1))];
        let frames: Vec<Vec<f32>> = (0..7).map(|i| vec![i as f32]).collect();
        let peaked = |j: Script| Script { rules: j.rules, calls: 0 };
        let mut a = peaked(Script { rules: rules.clone(), calls: 0 });
        let g = greedy(&mut a, &cfg(), 7, |t| &frames[t]).unwrap();
        // make the scripted logits sharp so log-probabilities favour the scripted path
        struct Sharp(Script);
        impl Joint for Sharp {
            type State = u32;
            fn initial_state(&self) -> u32 {
                0
            }
            fn step(&mut self, enc: &[f32], prev: u32, st: &u32) -> Result<(Vec<f32>, u32)> {
                let (o, s) = self.0.step(enc, prev, st)?;
                Ok((o.iter().map(|x| x * 20.0).collect(), s))
            }
        }
        for b in [1, 4] {
            let mut j = Sharp(Script { rules: rules.clone(), calls: 0 });
            assert_eq!(beam(&mut j, &cfg(), 7, |t| &frames[t], b, false).unwrap(), g, "beam {b}");
        }
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
