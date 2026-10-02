# Changelog

All notable changes to Prata (the `prata` CLI, `prata-web` and the `prata-app` npm launcher) are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.1] - TBD <!-- TODO(0.6.1-numbers) RELEASE-DATE -->

0.6.1 is the first published release with Snabb. Version 0.6.0 was never released: it has no git tag, no GitHub
Release and no npm package, so 0.6.1 follows 0.5.1 directly.

### Added
- **Snabb**, a fast Swedish model: [Klang Pianissimo](https://huggingface.co/KlangAI/pianissimo-sv) by KlangAI
  (CC BY 4.0). Prata runs KlangAI's official int8 ONNX export (`KlangAI/pianissimo-sv-onnx`, pinned revision),
  unmodified, with ONNX Runtime on the CPU. It decodes in 30 s windows with context on both sides, writes its own
  punctuation and casing and gives sentence-level timestamps. CLI: `prata FILE --model snabb`. Snabb is included in
  the macOS Apple Silicon and Linux x64 builds, but not in macOS Intel.
- Web UI model picker with three models: **Snabb**, **Standard** (KB-Whisper small, still the default) and
  **Large** (KB-Whisper large). `tiny`, `base` and `medium` are no longer shown in the UI but still work via the
  CLI, the HTTP API and `PRATA_MODEL`.
- Download progress in the UI the first time a model is used, and Snabb's model attribution under the picker.
- **Long-recording hint:** for files and recordings over 15 minutes the UI suggests Snabb
  (“Lång inspelning – Snabb går betydligt fortare”). It switches model only when you click it. The rule lives on the
  server: `GET /api/model-hint?duration_s=…&model=…`.
- `GET /api/info` lists the models with label, description, availability, download size, whether the model is
  already downloaded, and license.
- `scripts/bench-models.sh` and `scripts/wer_sv.py` compare models on your own recordings (wall time, peak memory,
  WER with Swedish normalisation).
- README: models, speed/WER table, Snabb, and credits. NOTICE: KlangAI Pianissimo (CC BY 4.0), its base model NVIDIA
  Parakeet TDT 0.6B v3, ONNX Runtime/ort, onnx-asr/NeMo mel front end. New CHANGELOG.

### Changed
- Release builds for macOS Apple Silicon and Linux x64 link ONNX Runtime statically (`snabb` cargo feature).
  macOS Intel is built without it.
- A Snabb job on a build without Snabb fails with a Swedish message. It never falls back to Python or to another
  model.
- `prata --help` ends with a line saying whether this build includes Snabb.

### Fixed
- Linux x64: Snabb kept ONNX Runtime from using AMX instructions (seccomp, applied to all threads). On some CPUs
  under load they produced corrupted int8 output. Set `PRATA_SNABB_AMX=1` to allow AMX. If the filter can't be
  installed, a warning is logged.
- Snabb normalises the log-mel features with statistics over the whole file instead of per window. A determinism
  regression test covers model loading and parallel runs.
<!-- TODO(0.6.1-numbers) FIXES: add Coder's final Snabb fixes here if they land (seam duplication, dropped first word, "OKDacke" detokenisation) -->

## [0.6.0] - never released

Skipped. The version number was renamed to 0.6.1 before the release, so no 0.6.0 was ever published.

## [0.5.1] - 2026-10-02
### Changed
- Klang import: notes without a real Klang title get one from the summary. The sync line is cleaner, and the summary
  is included in `.txt` exports.
- The npm launcher passes prata-web's stdout/stderr through to the terminal (the Klang key is never logged).

## [0.5.0] - 2026-10-02
### Added
- Read-only import from Klang (`KLANG_API_KEY`, **Synka från Klang**, `POST /api/klang/sync`) with a Klang label,
  a safely rendered Markdown summary and a view without a player.
- Notes from direct links are named after the file.
- The npm launcher warns (without stopping) when yt-dlp is missing.

## [0.4.0] - 2026-10-02
### Added
- Transcribe from a link (`POST /api/jobs/url`): direct audio/video links, and web pages via yt-dlp, with SSRF
  checks and limits. Notes link back to their source.

## [0.3.0] - 2026-10-02
### Added
- Saved notes grouped by day, a mobile-first UI, installing to the Home Screen (PWA) and `--host`. README: iPhone
  via `tailscale serve`, LaunchAgent, security notes.

## [0.2.0] - 2026-10-02
### Added
- KV-cached decoder (1.7–2.1× faster on Metal, identical output), energy-based silence skipping (VAD), and batched
  decoding. `scripts/bench.sh` and `scripts/wer.py`.
- prata-web passes `PRATA_VAD` / `PRATA_BATCH_SIZE` through to the CLI.

## [0.1.1] - 2026-10-02
### Fixed
- npm launcher: removed an outdated placeholder warning and included LICENSE.

## [0.1.0] - 2026-10-02
### Added
- First release: Swedish speech-to-text with KB-Whisper (CLI, web UI, `npx prata-app` launcher, release workflow).

[0.6.1]: https://github.com/Malmen877/prata/compare/v0.5.1...v0.6.1
[0.5.1]: https://github.com/Malmen877/prata/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Malmen877/prata/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Malmen877/prata/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Malmen877/prata/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Malmen877/prata/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Malmen877/prata/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Malmen877/prata/releases/tag/v0.1.0
