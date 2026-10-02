//! onnxruntime sessions for the Pianissimo encoder and decoder/joint network
//! (int8 ONNX export from KlangAI/pianissimo-sv-onnx).

use super::tdt::Joint;
use anyhow::{anyhow, bail, Result};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

/// Prediction network: 2 LSTM layers of 640 units.
const LSTM_LAYERS: usize = 2;
const LSTM_UNITS: usize = 640;

fn session(path: &Path, threads: Option<usize>) -> Result<Session> {
    let mut b = Session::builder().map_err(|e| anyhow!("onnxruntime: {e}"))?;
    if let Some(n) = threads {
        b = b.with_intra_threads(n).map_err(|e| anyhow!("onnxruntime: {e}"))?;
    }
    b.commit_from_file(path).map_err(|e| anyhow!("onnxruntime: could not load {}: {e}", path.display()))
}

pub struct Encoder {
    s: Session,
}

impl Encoder {
    pub fn load(path: &Path, threads: Option<usize>) -> Result<Self> {
        Ok(Self { s: session(path, threads)? })
    }

    /// `feats` is `[128][n_frames]`. Returns encoder frames `[t][dim]` (time-major).
    pub fn run(&mut self, feats: Vec<f32>, n_frames: usize, valid: usize) -> Result<(Vec<f32>, usize, usize)> {
        let x = Tensor::from_array(([1usize, super::mel::N_MELS, n_frames], feats))?;
        let len = Tensor::from_array(([1usize], vec![valid as i64]))?;
        let out = self.s.run(ort::inputs!["audio_signal" => x, "length" => len])?;
        let (shape, data) = out["outputs"].try_extract_tensor::<f32>()?;
        let (_, lens) = out["encoded_lengths"].try_extract_tensor::<i64>()?;
        if shape.len() != 3 || shape[0] != 1 {
            bail!("unexpected encoder output shape {shape:?}");
        }
        let (dim, t) = (shape[1] as usize, shape[2] as usize);
        let n = (lens[0].max(0) as usize).min(t);
        // (1, dim, t) -> (t, dim)
        let mut tm = vec![0f32; n * dim];
        for d in 0..dim {
            let row = &data[d * t..d * t + n];
            for (i, &v) in row.iter().enumerate() {
                tm[i * dim + d] = v;
            }
        }
        Ok((tm, n, dim))
    }
}

pub struct DecoderJoint {
    s: Session,
    dim: usize,
}

#[derive(Clone)]
pub struct LstmState(Vec<f32>, Vec<f32>);

impl DecoderJoint {
    pub fn load(path: &Path, dim: usize, threads: Option<usize>) -> Result<Self> {
        // tiny per-step graph: one thread avoids thread-pool overhead per call
        let _ = threads;
        Ok(Self { s: session(path, Some(1))?, dim })
    }
}

impl Joint for DecoderJoint {
    type State = LstmState;

    fn initial_state(&self) -> LstmState {
        LstmState(vec![0.0; LSTM_LAYERS * LSTM_UNITS], vec![0.0; LSTM_LAYERS * LSTM_UNITS])
    }

    fn step(&mut self, enc: &[f32], prev: u32, st: &LstmState) -> Result<(Vec<f32>, LstmState)> {
        let e = Tensor::from_array(([1usize, self.dim, 1], enc.to_vec()))?;
        let tg = Tensor::from_array(([1usize, 1], vec![prev as i32]))?;
        let tl = Tensor::from_array(([1usize], vec![1i32]))?;
        let s1 = Tensor::from_array(([LSTM_LAYERS, 1, LSTM_UNITS], st.0.clone()))?;
        let s2 = Tensor::from_array(([LSTM_LAYERS, 1, LSTM_UNITS], st.1.clone()))?;
        let out = self.s.run(ort::inputs![
            "encoder_outputs" => e,
            "targets" => tg,
            "target_length" => tl,
            "input_states_1" => s1,
            "input_states_2" => s2,
        ])?;
        let (_, o) = out["outputs"].try_extract_tensor::<f32>()?;
        let (_, a) = out["output_states_1"].try_extract_tensor::<f32>()?;
        let (_, b) = out["output_states_2"].try_extract_tensor::<f32>()?;
        Ok((o.to_vec(), LstmState(a.to_vec(), b.to_vec())))
    }
}
