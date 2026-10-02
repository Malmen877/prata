//! Whisper text decoder with a self-attention KV cache.
//!
//! A port of `candle_transformers::models::whisper::model::TextDecoder` (same
//! weights, same ops). candle's decoder only caches the cross-attention keys and
//! values; every step re-runs the whole token sequence through all layers, so a
//! window with n tokens costs O(n²) layer passes. Here each step only feeds the
//! new token(s) and appends their keys/values to a per-layer cache, which is
//! mathematically the same computation.
//!
//! With `use_cache = false` it behaves exactly like candle's decoder (full
//! recompute), which is what `--kv-cache off` uses.
//!
//! The cache is batched (dim 0 = batch row); `select_rows` drops finished rows.
//!
//! The audio encoder is ported too (unchanged ops), only so that the model can be
//! loaded without also materialising candle's decoder weights (that cost ~2 GB of
//! extra peak memory with the large model).

use candle_core::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{embedding, linear, linear_no_bias, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, VarBuilder};
use candle_transformers::models::whisper::Config;

struct Attn {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    cache: Option<(Tensor, Tensor)>,
}

impl Attn {
    fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            query: linear(n_state, n_state, vb.pp("q_proj"))?,
            value: linear(n_state, n_state, vb.pp("v_proj"))?,
            key: linear_no_bias(n_state, n_state, vb.pp("k_proj"))?,
            out: linear(n_state, n_state, vb.pp("out_proj"))?,
            n_head,
            cache: None,
        })
    }

    /// Self-attention. `pos` = number of tokens already in the cache (when caching).
    fn forward_self(&mut self, x: &Tensor, mask: &Tensor, pos: usize, use_cache: bool) -> Result<Tensor> {
        let q = self.query.forward(x)?;
        let mut k = self.key.forward(x)?;
        let mut v = self.value.forward(x)?;
        let n_new = x.dim(1)?;
        if use_cache {
            if pos == 0 {
                self.cache = None;
            }
            if let Some((ck, cv)) = &self.cache {
                k = Tensor::cat(&[ck, &k], 1)?;
                v = Tensor::cat(&[cv, &v], 1)?;
            }
            self.cache = Some((k.clone(), v.clone()));
        }
        let total = k.dim(1)?;
        let first = total - n_new;
        // causal mask rows first..total, cols 0..total (no mask needed for one new token)
        let m = if n_new > 1 { Some(mask.i((first..total, 0..total))?) } else { None };
        let wv = self.qkv_attention(&q, &k, &v, m.as_ref())?;
        self.out.forward(&wv)
    }

    /// Unmasked, uncached self-attention (encoder).
    fn forward_plain(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;
        let wv = self.qkv_attention(&q, &k, &v, None)?;
        self.out.forward(&wv)
    }

    /// Cross-attention over the encoder output; keys/values cached until `flush`.
    fn forward_cross(&mut self, x: &Tensor, xa: &Tensor, flush: bool) -> Result<Tensor> {
        let q = self.query.forward(x)?;
        if flush {
            self.cache = None;
        }
        let (k, v) = if let Some((k, v)) = &self.cache {
            (k.clone(), v.clone())
        } else {
            let k = self.key.forward(xa)?;
            let v = self.value.forward(xa)?;
            self.cache = Some((k.clone(), v.clone()));
            (k, v)
        };
        let wv = self.qkv_attention(&q, &k, &v, None)?;
        self.out.forward(&wv)
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = x.dims3()?;
        x.reshape(&[n_batch, n_ctx, self.n_head, n_state / self.n_head])?.transpose(1, 2)
    }

    fn qkv_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (_, _, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = q.matmul(&k)?;
        if let Some(mask) = mask {
            qk = qk.broadcast_add(mask)?
        }
        let w = candle_nn::ops::softmax_last_dim(&qk)?;
        w.matmul(&v)?.transpose(1, 2)?.flatten_from(2)
    }

    fn select_rows(&mut self, idx: &Tensor) -> Result<()> {
        if let Some((k, v)) = &self.cache {
            self.cache = Some((k.index_select(idx, 0)?, v.index_select(idx, 0)?));
        }
        Ok(())
    }
}

struct Block {
    attn: Attn,
    attn_ln: LayerNorm,
    cross: Attn,
    cross_ln: LayerNorm,
    mlp1: Linear,
    mlp2: Linear,
    mlp_ln: LayerNorm,
}

fn ln(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    // same construction as candle's whisper model
    Ok(LayerNorm::new(vb.get(size, "weight")?, vb.get(size, "bias")?, 1e-5))
}

fn sinusoids(length: usize, channels: usize, device: &candle_core::Device) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<_> = (0..channels / 2).map(|i| (i as f32 * (-log_timescale_increment)).exp()).collect();
    let inv_timescales = Tensor::new(inv_timescales.as_slice(), device)?.unsqueeze(0)?;
    let arange = Tensor::arange(0, length as u32, device)?.to_dtype(candle_core::DType::F32)?.unsqueeze(1)?;
    let sh = (length, channels / 2);
    let scaled_time = (arange.broadcast_as(sh)? * inv_timescales.broadcast_as(sh)?)?;
    Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)
}

struct EncBlock {
    attn: Attn,
    attn_ln: LayerNorm,
    mlp1: Linear,
    mlp2: Linear,
    mlp_ln: LayerNorm,
}

/// Port of candle's `AudioEncoder` (same weights and ops).
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<EncBlock>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let conv = |cin: usize, stride: usize, vb: VarBuilder| -> Result<Conv1d> {
            let w = vb.get((n_state, cin, 3), "weight")?;
            let b = vb.get(n_state, "bias")?;
            Ok(Conv1d::new(w, Some(b), Conv1dConfig { padding: 1, stride, groups: 1, dilation: 1, cudnn_fwd_algo: None }))
        };
        let conv1 = conv(cfg.num_mel_bins, 1, vb.pp("conv1"))?;
        let conv2 = conv(n_state, 2, vb.pp("conv2"))?;
        let positional_embedding = sinusoids(cfg.max_source_positions, n_state, vb.device())?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                let vb = vb.pp(format!("layers.{i}"));
                Ok(EncBlock {
                    attn: Attn::load(n_state, n_head, vb.pp("self_attn"))?,
                    attn_ln: ln(n_state, vb.pp("self_attn_layer_norm"))?,
                    mlp1: linear(n_state, n_state * 4, vb.pp("fc1"))?,
                    mlp2: linear(n_state * 4, n_state, vb.pp("fc2"))?,
                    mlp_ln: ln(n_state, vb.pp("final_layer_norm"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_post = ln(n_state, vb.pp("layer_norm"))?;
        Ok(Self { conv1, conv2, positional_embedding, blocks, ln_post })
    }

    /// `x`: (B, n_mels, frames) → (B, frames/2, d_model)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        let x = x.transpose(1, 2)?;
        let (_, seq_len, _) = x.dims3()?;
        let mut x = x.broadcast_add(&self.positional_embedding.narrow(0, 0, seq_len)?)?;
        for b in &self.blocks {
            x = (&x + b.attn.forward_plain(&b.attn_ln.forward(&x)?)?)?;
            let mlp = b.mlp2.forward(&b.mlp1.forward(&b.mlp_ln.forward(&x)?)?.gelu()?)?;
            x = (x + mlp)?;
        }
        self.ln_post.forward(&x)
    }
}

pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<Block>,
    ln: LayerNorm,
    mask: Tensor,
}

impl TextDecoder {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let n_ctx = cfg.max_target_positions;
        let token_embedding = embedding(cfg.vocab_size, n_state, vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get((n_ctx, n_state), "embed_positions.weight")?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                let vb = vb.pp(format!("layers.{i}"));
                Ok(Block {
                    attn: Attn::load(n_state, n_head, vb.pp("self_attn"))?,
                    attn_ln: ln(n_state, vb.pp("self_attn_layer_norm"))?,
                    cross: Attn::load(n_state, n_head, vb.pp("encoder_attn"))?,
                    cross_ln: ln(n_state, vb.pp("encoder_attn_layer_norm"))?,
                    mlp1: linear(n_state, n_state * 4, vb.pp("fc1"))?,
                    mlp2: linear(n_state * 4, n_state, vb.pp("fc2"))?,
                    mlp_ln: ln(n_state, vb.pp("final_layer_norm"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = ln(n_state, vb.pp("layer_norm"))?;
        let mask: Vec<_> = (0..n_ctx)
            .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (n_ctx, n_ctx), vb.device())?;
        Ok(Self { token_embedding, positional_embedding, blocks, ln, mask })
    }

    /// `x`: (B, n) token ids. With `use_cache`, `x` holds only the tokens after
    /// position `pos` (pos = 0 resets the self-attention cache); without, `x` is
    /// the whole sequence and `pos` must be 0. `flush` rebuilds the cross-attention
    /// cache from `xa`. Returns hidden states for the tokens in `x`.
    pub fn forward(&mut self, x: &Tensor, xa: &Tensor, pos: usize, use_cache: bool, flush: bool) -> Result<Tensor> {
        let n = x.dim(D::Minus1)?;
        let tok = self.token_embedding.forward(x)?;
        let pe = self.positional_embedding.narrow(0, pos, n)?;
        let mut x = tok.broadcast_add(&pe)?;
        for b in self.blocks.iter_mut() {
            let a = b.attn.forward_self(&b.attn_ln.forward(&x)?, &self.mask, pos, use_cache)?;
            x = (x + a)?;
            x = (&x + b.cross.forward_cross(&b.cross_ln.forward(&x)?, xa, flush)?)?;
            let mlp = b.mlp2.forward(&b.mlp1.forward(&b.mlp_ln.forward(&x)?)?.gelu()?)?;
            x = (x + mlp)?;
        }
        self.ln.forward(&x)
    }

    pub fn final_linear(&self, x: &Tensor) -> Result<Tensor> {
        let b_size = x.dim(0)?;
        let w = self.token_embedding.embeddings().broadcast_left(b_size)?;
        x.matmul(&w.t()?)
    }

    /// Keep only the given batch rows in all caches.
    pub fn select_rows(&mut self, idx: &Tensor) -> Result<()> {
        for b in self.blocks.iter_mut() {
            b.attn.select_rows(idx)?;
            b.cross.select_rows(idx)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn tiny() -> Result<(TextDecoder, Tensor)> {
        let cfg: Config = serde_json::from_value(serde_json::json!({
            "num_mel_bins": 80, "max_source_positions": 8, "d_model": 16,
            "encoder_attention_heads": 2, "encoder_layers": 1, "vocab_size": 50,
            "max_target_positions": 32, "decoder_attention_heads": 2, "decoder_layers": 2
        }))
        .unwrap();
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let dec = TextDecoder::load(vb, &cfg)?;
        let mask = dec.mask.clone();
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.5, v.shape(), &dev)?)?;
        }
        // keep the causal mask (not a var) intact
        assert_eq!(mask.dims(), dec.mask.dims());
        let xa = Tensor::randn(0f32, 1.0, (2, 8, 16), &dev)?;
        Ok((dec, xa))
    }

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
    }

    #[test]
    fn cached_steps_match_full_recompute() -> Result<()> {
        let (mut dec, xa) = tiny()?;
        let toks: Vec<u32> = vec![1, 7, 3, 9, 4, 22, 5, 11, 2];
        let dev = Device::Cpu;
        let row = |t: &[u32]| Tensor::new(t, &dev).unwrap().unsqueeze(0).unwrap().repeat((2, 1)).unwrap();
        // step 0: prompt of 3 tokens, then one token at a time
        let mut cached = vec![dec.forward(&row(&toks[..3]), &xa, 0, true, true)?.narrow(1, 2, 1)?];
        for p in 3..toks.len() {
            cached.push(dec.forward(&row(&toks[p..p + 1]), &xa, p, true, false)?);
        }
        for (k, p) in (2..toks.len()).enumerate() {
            let full = dec.forward(&row(&toks[..p + 1]), &xa, 0, false, true)?.narrow(1, p, 1)?;
            assert!(max_diff(&full, &cached[k]) < 1e-4, "pos {p}");
            assert!(max_diff(&full, &full.zeros_like()?) > 0.1, "weights not randomised");
        }
        Ok(())
    }

    #[test]
    fn select_rows_keeps_remaining_rows_consistent() -> Result<()> {
        let (mut dec, xa) = tiny()?;
        let dev = Device::Cpu;
        let a = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &dev)?;
        dec.forward(&a, &xa, 0, true, true)?;
        dec.select_rows(&Tensor::new(&[1u32], &dev)?)?;
        let y = dec.forward(&Tensor::new(&[[7u32]], &dev)?, &xa.narrow(0, 1, 1)?, 3, true, false)?;
        let full = dec.forward(&Tensor::new(&[[4u32, 5, 6, 7]], &dev)?, &xa.narrow(0, 1, 1)?, 0, false, true)?.narrow(1, 3, 1)?;
        assert!(max_diff(&y, &full) < 1e-4);
        Ok(())
    }
}
