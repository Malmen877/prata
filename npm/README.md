# prata-app

Launcher for **[Prata](https://github.com/Malmen877/prata)**: local, private Swedish speech-to-text in your browser.

```bash
brew install ffmpeg        # macOS  (Linux: sudo apt install ffmpeg)
npx prata-app
```

What it does:

1. Detects your platform: `darwin-arm64` (Apple Silicon, Metal GPU), `darwin-x64`, or `linux-x64`.
2. On first run, downloads `prata-v<version>-<platform>.tar.gz` from the matching GitHub Release
   (`v<version>` = this package's version), verifies its `.sha256` if present, and extracts it with
   the system `tar` into `~/.prata/<version>/`.
3. Checks that `ffmpeg` is installed, and warns (without stopping) if `yt-dlp` is missing – it is only needed
   for links to web pages such as YouTube (`brew install yt-dlp`); direct audio/video links work without it.
4. Starts `prata-web` on `127.0.0.1` (port 8795, or a free port), waits until it responds and opens your browser.
5. Ctrl+C stops the server. prata-web's own log lines appear in the same terminal (or in the LaunchAgent's
   `StandardOutPath`/`StandardErrorPath` file); earlier versions wrote them to `~/.prata/prata-web.log`.

The first transcription with a model downloads it from Hugging Face (Snabb ≈ 660 MB / 630 MiB, Standard/small ≈ 1 GB,
Large ≈ 3.1 GB) into `~/.cache/huggingface/`. Nothing else leaves your machine.

The web UI offers **Snabb** (Klang Pianissimo, fastest; macOS Apple Silicon and Linux x64), **Standard**
(KB-Whisper small, default) and **Large** (KB-Whisper large). See the [main README](https://github.com/Malmen877/prata#models).

## Options

```
npx prata-app [--port N] [--no-open] [--model snabb|small|large|tiny|base|medium] [--version] [--help]
```

`--model` sets the default model (`PRATA_MODEL`); the UI preselects it if it is one of `snabb`, `small`, `large`.

| Environment variable | Meaning |
|---|---|
| `PRATA_GITHUB_OWNER` / `PRATA_GITHUB_REPO` | where releases are downloaded from |
| `PRATA_LOCAL_ASSET=/path/prata-v….tar.gz` | install from a local archive (testing) |
| `PRATA_BIN_DIR=/path/to/dir` | run binaries from a directory, no download or cache |
| `PRATA_CACHE_DIR` | cache directory instead of `~/.prata` |
| `PRATA_PORT` | port (same as `--port`) |
| `PRATA_NO_BROWSER=1` | same as `--no-open` |

No dependencies; requires Node.js 18+. MIT License.
