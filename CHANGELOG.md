# Changelog

All notable changes to Prata (the `prata` CLI, `prata-web` and the `prata-app` npm launcher) are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-10-04

### Added
- **Safer recording**: every second of a recording is also saved in the browser (IndexedDB). After a crash, reload or
  closed tab the record view offers **Återställ inspelning** (transcribe it, download it, or throw it away); the copy is
  cleared once the upload is accepted. The screen stays on while recording (Wake Lock), closing the tab during a
  recording or upload asks first, and the mic is released as soon as you stop.
- **Mic watchdog** on the record screen: a quiet warning when no sound has been heard for 10 s, no new audio data
  arrives, the mic is muted or disconnected, or audio is missing after the screen was locked ("Inspelningen pausades
  när skärmen låstes – ca 12 s saknas").
- **Queue control**: several files at once go into a visible queue under the progress card; **Avbryt** stops a running
  job (ffmpeg/prata are killed), failed or cancelled jobs get **Försök igen** and **Ta bort**. API:
  `GET /api/jobs?ids=`, `POST /api/jobs/{id}/cancel`, `POST /api/jobs/{id}/retry`, `DELETE /api/jobs/{id}`.
- Jobs that were queued or running when prata-web stopped are marked as failed at the next start and can be retried
  (the upload is kept for 7 days in the work folder).
- Web UI: estimated time left in the progress text ("ca 2 min kvar"), shown once progress is steady.
- Web UI: −15/+15 s and 1×/1,5×/2× playback speed in the player (speed is remembered), and a Dela icon that opens the
  share sheet or copies the transcript.
- **Ordlista**: map a correct form to known wrong forms (e.g. "OKDacke" ← "OK Dacke"); applied to new transcriptions
  (whole words, case-insensitive, åäö-aware, longest match first). Edited under "Ordlista" in the record view; stored
  in ~/.prata/wordlist.json (GET/PUT /api/wordlist).
- Web UI: "✓ Klar – <titel>" in the tab title when a transcription finishes, and a desktop notification if the tab is
  in the background (permission asked once, on the first transcription).

### Fixed
- Files with several audio streams (Voice Memos with Spatial Audio, some videos) are transcribed from the first audio
  stream (`-map 0:a:0`) instead of the one ffmpeg would pick.

## [0.6.1] - 2026-10-02

The first published release with Snabb. 0.6.0 was never released (no tag, GitHub Release or npm package).

### Added
- **Snabb**: [Klang Pianissimo](https://huggingface.co/KlangAI/pianissimo-sv) by KlangAI (CC BY 4.0). Prata runs the
  official int8 ONNX export, unmodified, with ONNX Runtime on the CPU (`--model snabb`). It is included in the macOS
  Apple Silicon and Linux x64 builds, not in macOS Intel.
- Web UI picker with **Snabb**, **Standard** (KB-Whisper small, the default) and **Large** (described as "Största
  modellen, långsammast"). `tiny`, `base` and `medium` still work via the CLI, the API and `PRATA_MODEL`.
- A click-only hint to switch to Snabb for recordings over 15 minutes (`GET /api/model-hint`), plus download progress
  the first time a model is used.
- `GET /api/info` reports `version` and a `models` list (availability, size, downloaded, license).
- `scripts/bench-models.sh` and `scripts/wer_sv.py` compare models; a CHANGELOG, and a NOTICE that covers Pianissimo,
  Parakeet, ONNX Runtime and onnx-asr/NeMo.

### Changed
- **Linux x64 needs glibc 2.39+** (Ubuntu 24.04+, Debian 13+, Fedora 40+). The launcher stops with a Swedish message
  on older systems; use 0.5.1 or build from source without Snabb.
- A Snabb job on a build without Snabb fails with a Swedish message and never falls back to another model.
  `prata --help` says whether the build includes Snabb.

### Fixed
- Linux x64: ONNX Runtime is kept off Intel AMX, whose int8 kernels gave corrupted output under load. This costs about
  10 % speed; `PRATA_SNABB_AMX=1` re-enables AMX.
- Snabb normalises log-mel features over the whole recording; per-window statistics broke on digital silence.
- Snabb drops a word seen by both windows at a seam (by timestamp overlap); long12 WER went from 5.49 % to 5.26 %.
  A determinism regression test was added.

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

[0.7.0]: https://github.com/Malmen877/prata/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/Malmen877/prata/compare/v0.5.1...v0.6.1
[0.5.1]: https://github.com/Malmen877/prata/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Malmen877/prata/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Malmen877/prata/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Malmen877/prata/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Malmen877/prata/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Malmen877/prata/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Malmen877/prata/releases/tag/v0.1.0
