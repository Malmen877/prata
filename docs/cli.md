# CLI and server configuration

## prata (CLI)

```
prata <AUDIO> [--model tiny|base|small|medium|large|snabb|<hf repo id>] [--timestamps] [--out FILE]
              [--revision main|strict|subtitle] [--language sv] [--cpu]
              [--kv-cache on|off] [--vad on|off] [--vad-min-silence S] [--vad-pad S]
              [--vad-threshold DB] [--batch-size N] [--pack] [--verbose]
```

Without `--timestamps` it prints plain text; with it, SRT. `--cpu` forces CPU on a Metal/CUDA build. The CLI's default
model is `large` (the web UI's is `small`). `prata --help` ends with a line saying whether this build includes Snabb. `--revision` has no fixed default: KB-Whisper uses `main`, and Snabb uses the
pinned revision it was tested with.
The VAD options below apply to all models; `--batch-size`, `--pack` and `--kv-cache` apply to KB-Whisper only.

## Speed options

| Flag | Default | What it does |
|---|---|---|
| `--kv-cache on\|off` | `on` | Caches the decoder's self-attention keys/values, so each new token costs one layer pass instead of re-running the whole sequence. Same computation, same output (byte-identical SRT in our tests), ~1.6–1.9× faster on CPU. |
| `--vad on\|off` | `on` | Energy-based voice activity detection. Pauses of at least `--vad-min-silence` seconds (default `1.0`) are skipped, keeping `--vad-pad` seconds (default `0.3`) of audio on each side of speech. Each speech region is decoded with the normal sequential decoder, and no 30 s window reaches across a skipped pause. Timestamps always refer to the original file. If no pause is long enough, the result is exactly the same as `--vad off`. `--vad-threshold DB` overrides the automatic speech threshold (dB above the noise floor). |
| `--batch-size N` | `0` = auto (4 on Metal/CUDA, 1 on CPU) | Encodes and decodes the current windows of up to N speech regions together. Within a region the windows are the same as the sequential decoder's. This only helps when VAD found several regions. |
| `--pack` | off | **Experimental.** Packs speech into fixed windows of up to 30 s, cut at quiet points, so that continuous speech can be batched too. About 1.5× faster again on CPU with `--batch-size 4`, but the window borders differ from the sequential decoder, so some words come out differently (4% word difference on a 2.5-minute test clip). |
| `--verbose` | | Prints the VAD regions or window plan. |

## Web server configuration

```
prata-web [--host ADDR] [--port PORT] [--notes-dir DIR]
          [--url-max-mb MB] [--url-max-duration DUR] [--url-timeout DUR] [--yt-dlp PATH]
```

Flags override the environment variables below. `PRATA_*` is the primary name; the older `KBW_*` names still work.

| Variable | Default | Meaning |
|---|---|---|
| `PRATA_WEB_PORT` | `8795` | listen port |
| `PRATA_WEB_HOST` / `PRATA_HOST` | `127.0.0.1` | listen address (`--host`). Keep it local – there is no authentication; prata-web prints a warning for any non-loopback address |
| `PRATA_NOTES_DIR` | `~/.prata/notes` | where notes are saved (`--notes-dir`) |
| `PRATA_URL_MAX_MB` | `500` | size limit for links (`--url-max-mb`) |
| `PRATA_URL_MAX_DURATION` | `3h` | duration limit for links, e.g. `3h`, `90m`, `5400` (`--url-max-duration`) |
| `PRATA_URL_TIMEOUT` | `15m` | overall download timeout for links (`--url-timeout`) |
| `PRATA_YTDLP` | `yt-dlp` on `$PATH` | yt-dlp binary (`--yt-dlp`) |
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
| `KLANG_API_KEY` | unset | enables the [Klang import](klang-import.md) (`PRATA_KLANG_API_KEY` also works) |
| `PRATA_KLANG_BASE_URL` | `https://app.klang.ai/api/v1` | Klang API base (for tests against a mock; `http` only for `127.0.0.1`) |
