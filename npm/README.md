# prata-app

Launcher for **[Prata](https://github.com/Malmen877/prata)**, a local, private Swedish speech-to-text app that runs
in your browser.

```bash
npx prata-app
```

It downloads the prebuilt Prata binaries for your platform from the matching GitHub Release into `~/.prata/<version>/`
(verifying its `.sha256` if present), starts `prata-web` on `127.0.0.1` (port 8795 or a free port) and opens your browser.
Press Ctrl+C to stop. Needs Node.js 18+ and ffmpeg. yt-dlp is optional, for links to web pages.

The platforms are macOS Apple Silicon, macOS Intel (without Snabb) and Linux x64 with glibc 2.39+. On older glibc
the launcher stops with a message; use `npx prata-app@0.5.1` there.

| In the UI | Model | First download |
|---|---|---|
| **Snabb** | Klang Pianissimo (fastest) | ≈ 660 MB (630 MiB) |
| **Standard** (default) | KB-Whisper small | ≈ 1 GB |
| **Large** | KB-Whisper large | ≈ 3.1 GB |

Models are downloaded from Hugging Face into `~/.cache/huggingface/` the first time you use them. Nothing else
leaves your machine.

Options (`--port`, `--no-open`, `--model`) and environment variables: [docs/install.md](https://github.com/Malmen877/prata/blob/HEAD/docs/install.md#launcher-options).

Documentation: [README](https://github.com/Malmen877/prata#readme) and
[docs/](https://github.com/Malmen877/prata/tree/HEAD/docs).

MIT License. Snabb uses Klang Pianissimo by KlangAI (CC BY 4.0); KB-Whisper is by KBLab (Apache-2.0). See
[NOTICE](https://github.com/Malmen877/prata/blob/HEAD/NOTICE).
