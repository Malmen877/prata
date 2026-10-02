# Installation

```bash
# macOS
brew install ffmpeg
brew install yt-dlp     # optional: transcribe YouTube and other web pages
npx prata-app
```

On Linux install ffmpeg with `sudo apt install ffmpeg` (or your distro's package) first.
The Linux x64 binaries need **glibc 2.39 or newer**; see [linux.md](linux.md).

`npx prata-app` downloads the prebuilt binaries for your platform (macOS Apple Silicon, macOS Intel, Linux x64)
from this repository's GitHub Release into `~/.prata/<version>/`, starts the server on a free port
(8795 if available) and opens your browser. Press **Ctrl+C** to stop. The first transcription with a
model downloads its weights from Hugging Face into `~/.cache/huggingface/`.

Options: `npx prata-app --port 9000 --no-open --model snabb`. `--model` sets the default model (`PRATA_MODEL`): `snabb`,
`small` or `large` preselect that button in the UI; `tiny`, `base` and `medium` are only used as the default for HTTP API
jobs that don't name a model (the UI then preselects Standard). See [launcher options](#launcher-options).

## Launcher options

```
npx prata-app [--port N] [--no-open] [--model snabb|small|large|tiny|base|medium] [--version] [--help]
```

The launcher (`npm/bin/prata.js`, pure Node 18+, no dependencies) detects the platform (`darwin-arm64`, `darwin-x64`,
`linux-x64`). On first run it downloads `prata-v<version>-<platform>.tar.gz` from the GitHub Release matching the
package version, verifies its `.sha256` if one is present, and extracts it with the system `tar`. It checks that
ffmpeg is installed and warns, without stopping, if yt-dlp is missing. On Linux it checks glibc first. prata-web's
log lines appear in the same terminal, or in the LaunchAgent's `StandardOutPath`/`StandardErrorPath` file; versions
before 0.5.1 wrote them to `~/.prata/prata-web.log`.

| Environment variable | Meaning |
|---|---|
| `PRATA_GITHUB_OWNER` / `PRATA_GITHUB_REPO` | where releases are downloaded from |
| `PRATA_LOCAL_ASSET=/path/prata-v….tar.gz` | install from a local archive (testing) |
| `PRATA_BIN_DIR=/path/to/dir` | run binaries from a directory, no download or cache |
| `PRATA_CACHE_DIR` | cache directory instead of `~/.prata` |
| `PRATA_PORT` | port (same as `--port`) |
| `PRATA_NO_BROWSER=1` | same as `--no-open` |

Model downloads on first use: Snabb ≈ 660 MB (630 MiB), Standard ≈ 1 GB, Large ≈ 3.1 GB.

## Privacy

Everything runs locally. Audio is uploaded only to the Prata server on your own machine (`127.0.0.1`),
converted with your local ffmpeg and transcribed. The transcript and the original audio are saved as a note in
`~/.prata/notes/` (see [Notes](usage.md#notes)); the temporary converted files are deleted.
The only network access is downloading the model weights from Hugging Face on first use,
the one-time binary download when you use `npx prata-app`, and – only when you paste a link – downloading that
link's audio, and – only when you set `KLANG_API_KEY` and press **Synka från Klang** – reading your conversations
from Klang (see [Import from Klang](klang-import.md)). There is no telemetry.
