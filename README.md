# Prata

> **Svenska:** Prata är en lokal app för svensk tal-till-text i webbläsaren. Allt körs på din egen dator; inget ljud lämnar den.

Prata is a local, private Swedish speech-to-text app. You can record or drop a file in the browser and get a
timestamped transcript (.txt, .srt, .json). It runs [KB-Whisper](https://huggingface.co/KBLab/kb-whisper-large)
(KBLab) and [Klang Pianissimo](https://huggingface.co/KlangAI/pianissimo-sv) (KlangAI) entirely on your own machine.

![Prata – model picker with Snabb, Standard and Large](docs/screenshot.png)

## Install

```bash
npx prata-app
```

Needs macOS (Apple Silicon; Intel works without Snabb) or Linux x64 with glibc 2.39+, plus ffmpeg (`brew install ffmpeg`
/ `sudo apt install ffmpeg`). Details: [docs/install.md](docs/install.md) · [docs/linux.md](docs/linux.md).

## Models

| In the UI | Model | Mac mini M4, 12.5 min recording | WER |
|---|---|---|---:|
| **Snabb** | Klang Pianissimo, ONNX int8 (CPU) | 23.9 s | 5.3 % |
| **Standard** (default) | KB-Whisper small (Metal) | 133.7 s | 4.3 % |
| **Large** | KB-Whisper large (Metal) | not measured (2.5 min clip: 119.5 s) | 4.6 % |

The WER is against the original reference text. On recordings over 15 minutes the UI suggests Snabb; it switches
only when you click. Method, x86 numbers and corrected-reference WER: [docs/benchmarks.md](docs/benchmarks.md).
Models in detail: [docs/models.md](docs/models.md).

## iPhone (Tailscale)

Run Prata on an always-on Mac. Enable MagicDNS and HTTPS certificates in Tailscale, then publish Prata in your
tailnet with `tailscale serve --bg 8795`. This gives an HTTPS
address (`https://<machine>.<tailnet>.ts.net`), which Safari needs for the microphone, while prata-web stays on
`127.0.0.1`. Open it on the iPhone and choose **Share → Add to Home Screen**. Use `serve`, not `funnel`. There is no
login, so everyone in your tailnet can reach it. Setup and LaunchAgent: [docs/iphone.md](docs/iphone.md).

## More

[Using Prata (notes, links)](docs/usage.md) · [Klang import](docs/klang-import.md) · [CLI and configuration](docs/cli.md) ·
[HTTP API](docs/api.md) · [Building from source](docs/development.md) · [Limitations](docs/limitations.md) ·
[Changelog](CHANGELOG.md)

## License

Prata is [MIT](LICENSE) © 2026 Kevin Malmgren. Snabb uses **Klang Pianissimo by KlangAI**, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); KB-Whisper is by KBLab (Apache-2.0). Models are downloaded
from Hugging Face, not redistributed. Full credits: [NOTICE](NOTICE). Prata is not affiliated with KBLab, KlangAI,
NVIDIA, Hugging Face or OpenAI.
