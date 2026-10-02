# Building from source

Requirements: Rust (stable, ≥ 1.88), ffmpeg on `PATH`. On macOS: Xcode command line tools.

```bash
git clone https://github.com/Malmen877/prata.git && cd prata

# macOS, Apple Silicon (Metal GPU)
cargo build --release -p prata --features metal
cargo build --release -p prata-web

# Linux / CPU
cargo build --release          # without Snabb; add `-p prata --features snabb` (needs glibc 2.38+) for Snabb

./target/release/prata-web            # http://127.0.0.1:8795
./target/release/prata recording.m4a --model small --timestamps
```

Tests: `cargo test --release --workspace`. The link tests run against a local mock HTTP server; the real-yt-dlp
test is skipped when yt-dlp isn't installed. The `insecure-test-loopback` cargo feature (plus
`PRATA_INSECURE_ALLOW_LOOPBACK=1`) lets a *test build* download from 127.0.0.1 for browser end-to-end tests –
never build releases with it.

`prata-web` looks for the `prata` binary next to itself (so a workspace build or a release
archive just works), then `$PRATA_BIN`, then `$PATH`.

Package a release archive the same way CI does:

```bash
scripts/package.sh darwin-arm64        # → dist/prata-v<version>-darwin-arm64.tar.gz (+ .sha256)
PRATA_LOCAL_ASSET=$PWD/dist/prata-v0.6.1-darwin-arm64.tar.gz node npm/bin/prata.js
```

## Repository layout

```
crates/prata/       CLI (Candle whisper decoder, forced Swedish, 30 s windows, SRT output)
                    src/snabb/: Snabb engine (Klang Pianissimo via onnxruntime; `snabb` cargo feature)
crates/prata-web/   axum web server + single-file UI (src/index.html) and PWA icons (src/assets/), embedded in the binary
                    notes storage in src/notes.rs, link downloads (SSRF checks, yt-dlp) in src/fetch.rs,
                    Klang import in src/klang.rs, long-recording hint in src/hint.rs
python/             optional Python fallback backend (transformers), requirements.txt
npm/                `prata-app` launcher (pure Node, no dependencies)
scripts/package.sh  builds the release tar.gz
scripts/bench-models.sh, wer_sv.py  model comparison (time, peak memory, WER)
scripts/test-ui.mjs unit test for the UI's day grouping: `TZ=Europe/Stockholm node scripts/test-ui.mjs`
.github/workflows/  CI and tag-triggered release (GitHub Release assets + optional npm publish)
```

## Python fallback (optional)

If the `prata` binary is missing or fails, `prata-web` can use `python/transcribe.py`
(Hugging Face transformers). It is not needed for normal use.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r python/requirements.txt
python3 python/transcribe.py recording.m4a --model small --timestamps
```
