# Prata

> **Svenska:** Prata är en lokal app för svensk tal-till-text. Spela in direkt i webbläsaren eller släpp en ljudfil – texten kommer med tidsstämplar och kan laddas ner som .txt, .srt eller .json. Allt körs på din egen dator; inget ljud lämnar den. Starta med `npx prata-app`.

Prata is a local, private Swedish speech-to-text app. It runs the Swedish
[KB-Whisper](https://huggingface.co/KBLab/kb-whisper-large) models from the National Library of Sweden (KBLab)
with [Hugging Face Candle](https://github.com/huggingface/candle) (Metal GPU on Apple Silicon) and
serves a small web UI on `127.0.0.1`.

![Prata – finished transcript](docs/screenshot.png)

- Record from the microphone or upload a file (wav, mp3, m4a, ogg, flac, opus, webm, mp4 – anything ffmpeg reads)
- Transcript with segment timestamps, a waveform player, click-to-seek timestamps, search and copy
- Downloads: `.txt`, `.txt` with timestamps, `.srt` subtitles, `.json`
- Model choice tiny → large (default **small**), light and dark mode
- Also a command-line tool: `prata interview.m4a --timestamps`

## Quick start

```bash
# macOS
brew install ffmpeg
npx prata-app
```

On Linux install ffmpeg with `sudo apt install ffmpeg` (or your distro's package) first.

`npx prata-app` downloads the prebuilt binaries for your platform (macOS Apple Silicon, macOS Intel, Linux x64)
from this repository's GitHub Release into `~/.prata/<version>/`, starts the server on a free port
(8795 if available) and opens your browser. Press **Ctrl+C** to stop. The first transcription with a
model downloads its weights from Hugging Face into `~/.cache/huggingface/`.

Options: `npx prata-app --port 9000 --no-open --model medium`. See [npm/README.md](npm/README.md).

## Models

Measured on an Apple Silicon (M-series) Mac with the Metal build, transcribing 2 min 28 s of Swedish speech:

| Model | Time | Peak memory | Notes |
|---|---:|---:|---|
| tiny | 6 s | 0.7 GB | fastest, rough |
| base | 10 s | 1.0 GB | |
| **small** | **30 s** | **2.8 GB** | **default**: best speed/quality trade-off |
| medium | 80 s | 5.8 GB | |
| large | 137 s | 12.3 GB | best quality; needs a 16 GB+ Mac |

On CPU (Linux x64 build) everything is several times slower; `small` runs at roughly real time on an 8-core machine.
For accuracy figures (WER) of each size see the [KB-Whisper model card](https://huggingface.co/KBLab/kb-whisper-large).

## Privacy

Everything runs locally. Audio is uploaded only to the Prata server on your own machine (`127.0.0.1`),
converted with your local ffmpeg, transcribed, and the temporary files are deleted afterwards.
The only network access is downloading the model weights from Hugging Face on first use
(and the one-time binary download when you use `npx prata-app`). There is no telemetry.

## Build from source

Requirements: Rust (stable, ≥ 1.88), ffmpeg on `PATH`. On macOS: Xcode command line tools.

```bash
git clone https://github.com/Malmen877/prata.git && cd prata

# macOS, Apple Silicon (Metal GPU)
cargo build --release -p prata --features metal
cargo build --release -p prata-web

# Linux / CPU
cargo build --release

./target/release/prata-web            # http://127.0.0.1:8795
./target/release/prata recording.m4a --model small --timestamps
```

`prata-web` looks for the `prata` binary next to itself (so a workspace build or a release
archive just works), then `$PRATA_BIN`, then `$PATH`.

Package a release archive the same way CI does:

```bash
scripts/package.sh darwin-arm64        # → dist/prata-v<version>-darwin-arm64.tar.gz (+ .sha256)
PRATA_LOCAL_ASSET=$PWD/dist/prata-v0.1.0-darwin-arm64.tar.gz node npm/bin/prata.js
```

### Repository layout

```
crates/prata/       CLI (Candle whisper decoder, forced Swedish, 30 s windows, SRT output)
crates/prata-web/   axum web server + single-file UI (src/index.html, embedded in the binary)
python/             optional Python fallback backend (transformers), requirements.txt
npm/                `prata-app` launcher (pure Node, no dependencies)
scripts/package.sh  builds the release tar.gz
.github/workflows/  CI and tag-triggered release (GitHub Release assets + optional npm publish)
```

## CLI

```
prata <AUDIO> [--model tiny|base|small|medium|large|<hf repo id>] [--timestamps] [--out FILE]
              [--revision main|strict|subtitle] [--language sv] [--cpu]
              [--kv-cache on|off] [--vad on|off] [--vad-min-silence S] [--vad-pad S]
              [--vad-threshold DB] [--batch-size N] [--pack] [--verbose]
```

Without `--timestamps` it prints plain text; with it, SRT. `--cpu` forces CPU on a Metal/CUDA build.

### Speed options

| Flag | Default | What it does |
|---|---|---|
| `--kv-cache on\|off` | `on` | Caches the decoder's self-attention keys/values, so each new token costs one layer pass instead of re-running the whole sequence. Same computation, same output (byte-identical SRT in our tests), ~1.6–1.9× faster on CPU. |
| `--vad on\|off` | `on` | Energy-based voice activity detection. Pauses of at least `--vad-min-silence` seconds (default `1.0`) are skipped, keeping `--vad-pad` seconds (default `0.3`) of audio on each side of speech. Each speech region is decoded with the normal sequential decoder, and no 30 s window reaches across a skipped pause. Timestamps always refer to the original file. If no pause is long enough, the result is exactly the same as `--vad off`. `--vad-threshold DB` overrides the automatic speech threshold (dB above the noise floor). |
| `--batch-size N` | `0` = auto (4 on Metal/CUDA, 1 on CPU) | Encodes and decodes the current windows of up to N speech regions together. Within a region the windows are the same as the sequential decoder's. This only helps when VAD found several regions. |
| `--pack` | off | **Experimental.** Packs speech into fixed windows of up to 30 s, cut at quiet points, so that continuous speech can be batched too. About 1.5× faster again on CPU with `--batch-size 4`, but the window borders differ from the sequential decoder, so some words come out differently (4% word difference on a 2.5-minute test clip). |
| `--verbose` | | Prints the VAD regions or window plan. |

`scripts/bench.sh AUDIO [MODELS…]` times the configurations and reports RTF, peak memory, and WER against the baseline (`scripts/wer.py`). It works on macOS and Linux.

## Web server configuration

All settings are environment variables. `PRATA_*` is the primary name; the older `KBW_*` names still work.

| Variable | Default | Meaning |
|---|---|---|
| `PRATA_WEB_PORT` | `8795` | listen port |
| `PRATA_WEB_HOST` | `127.0.0.1` | listen address (keep it local – there is no authentication) |
| `PRATA_MODEL` | `small` | default model in the UI |
| `PRATA_BACKEND` | `auto` | `auto` (prata, falling back to Python), `prata`, or `python` |
| `PRATA_BIN` | next to `prata-web`, then `$PATH` | path to the `prata` CLI |
| `PRATA_ARGS` | inferred from `prata --help` | argument template, placeholders `{input}` `{model}` `{out}` |
| `PRATA_PYTHON` | `python3` | interpreter for the Python fallback |
| `PRATA_PYTHON_SCRIPT` | `transcribe.py` next to the binary, or `python/transcribe.py` | Python fallback script |
| `PRATA_WORK_DIR` | `$TMPDIR/prata-web` | temporary upload directory |
| `PRATA_MAX_UPLOAD_MB` | `1024` | upload size limit |
| `PRATA_VAD` | unset (CLI default `on`) | passed as `--vad` (`on`/`off`) |
| `PRATA_BATCH_SIZE` | unset (CLI default auto) | passed as `--batch-size` |

HTTP API: `POST /api/jobs` (multipart `file`, `model`) → `{id}`; `GET /api/jobs/{id}` (status, segments);
`GET /api/jobs/{id}/{txt|txt-ts|srt|json}`; `GET /api/info`.

### Python fallback (optional)

If the `prata` binary is missing or fails, `prata-web` can use `python/transcribe.py`
(Hugging Face transformers). It is not needed for normal use.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r python/requirements.txt
python3 python/transcribe.py recording.m4a --model small --timestamps
```

## Limitations

- Jobs run one at a time and live in memory only (lost when the server stops).
- Long recordings are kept in browser memory until you press stop.
- A word can occasionally be dropped or repeated at a 30-second window boundary.
- The microphone needs a secure context: use `http://127.0.0.1` or `localhost`, not a LAN IP.

## Credits and licenses

Prata's own code is released under the [MIT License](LICENSE) © 2026 Kevin Malmgren. See [NOTICE](NOTICE).

- **KB-Whisper** by [KBLab](https://huggingface.co/KBLab), National Library of Sweden – models
  `KBLab/kb-whisper-{tiny,base,small,medium,large}`, **Apache-2.0** (per the model cards). Weights are downloaded
  from Hugging Face at runtime and are not redistributed here. Please cite: Vesterbacka et al. (2025),
  *Swedish Whispers; Leveraging a Massive Speech Corpus for Swedish Speech Recognition*, Interspeech 2025.
- **Hugging Face Candle** – Apache-2.0 / MIT. The CLI is derived from Candle's whisper example.
- **OpenAI Whisper** – MIT (code); model architecture, decoding rules and mel filters.
- **ffmpeg** is used at runtime and installed separately.

Prata is an independent project and is not affiliated with KBLab, Hugging Face or OpenAI.
