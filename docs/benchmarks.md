# Benchmarks

Speed and accuracy of Prata 0.6.1 on three Swedish recordings: long12 (12.5 min), clip (2.5 min) and clip15 (15 s).
The measurements were made on a **Linux x86_64 machine** (KVM guest, Intel Xeon of the Sapphire Rapids class, 8 vCPU,
15 GB RAM, **CPU only, no GPU**) with default settings. Snabb ran int8 with `--threads 4` and 30 s windows with 5 s of
context; KB-Whisper ran on the CPU through Candle. The time is the wall time of one `prata` run including model
loading, with the model already downloaded. The machine was shared, so times are ±15 %. The peak memory is the peak
RSS of the `prata` process. The Mac column shows a Mac mini M4 (24 GB). Snabb runs on the CPU there too (ONNX
Runtime), while KB-Whisper uses Metal. On the Mac the memory figure is the peak memory footprint, which for
KB-Whisper includes the Metal (GPU) buffers.

WER is scored with Prata's Swedish normalisation (lowercase, punctuation removed, Swedish number words and digits
compare equal, hyphens unified). The first WER column uses the original reference text. That text comes from a 2010
article, and the reader doesn't follow it word for word in a few places. The second column uses a reference corrected
for those places.

**long12 (12.5 min)**

| Model | Time (x86 CPU) | Peak memory (x86 CPU) | WER | WER (corrected ref) | Mac mini M4 | Notes |
|---|---:|---:|---:|---:|---|---|
| **Snabb** (Klang Pianissimo, ONNX int8) | 40 s | 2.1 GB | 5.3 % | 3.0 % | 23.9 s wall time (21.7 s transcription), 2.2 GB² | fastest, about 19× faster than Standard here; recommended for long recordings |
| **Standard** (KB-Whisper small) | 576 s | 1.9 GB | 4.3 % | 2.1 % | 133.7 s, 4.9 GB peak footprint (Metal) | **default** |
| **Large** (KB-Whisper large) | not timed (about 30 min on this CPU) | 9.5 GB¹ | 4.6 % | 2.4 % | not measured (clip: 119.5 s, 11.7 GB peak footprint, Metal) | largest, slowest; needs a 16 GB+ Mac |

¹ Large's peak memory was measured on the 2.5-minute clip with the CPU build (f32). It is much lower on Metal. Large's
long12 WER comes from a Mac Metal run. Where both runs exist, KB-Whisper's output is byte-identical on the x86 CPU
build and on Metal, so the WER figures hold for both machines.

² Snabb on the Mac mini M4: the 2.5-minute clip takes 4.2 s with a 1.8 GB peak. The first run, on clip15 (15 s) and
including the download of the model, took 28.5 s. The model takes about 660 MB (630 MiB) in the Hugging Face cache.

**WER per recording** (original reference, corrected reference in brackets; time on x86 CPU)

| Recording | Snabb | Standard | Large |
|---|---|---|---|
| clip15 (15 s) | 5.56 % (5.56 %), 3.1 s | 2.78 % (2.78 %), 13.8 s | 0.00 % (0.00 %), 63.8 s |
| clip (2.5 min) | 5.14 % (2.87 %), 9.5 s | 4.29 % (2.01 %), 117.7 s | 4.00 % (1.72 %), 555.9 s |
| long12 (12.5 min) | 5.26 % (2.99 %), 40.1 s | 4.34 % (2.07 %), 576.1 s | 4.63 % (2.36 %), not timed |

Earlier, before the 0.6.1 Snabb fixes and with an older WER normalisation, a Mac mini M4 transcribed long12 in about
19 s with Snabb, 132 s with Standard and 595 s with Large. Those figures are from that earlier evaluation, not final
0.6.1 results.

One recording is not a general accuracy figure; see the model cards for benchmark WERs.
`scripts/bench-models.sh` reproduces the comparison on your own recordings.

## KB-Whisper sizes

Measured on an Apple Silicon (M-series) Mac with the Metal build, transcribing 2 min 28 s of Swedish speech:

| Model | Time | Peak memory | Notes |
|---|---:|---:|---|
| tiny | 6 s | 0.7 GB | fastest, rough |
| base | 10 s | 1.0 GB | |
| **small** | **30 s** | **2.8 GB** | **default**: best speed/quality trade-off |
| medium | 80 s | 5.8 GB | |
| large | 137 s | 12.3 GB | largest, slowest; needs a 16 GB+ Mac |

On CPU (Linux x64 build) everything is several times slower; `small` runs at roughly real time on an 8-core machine.
For accuracy figures (WER) of each size see the [KB-Whisper model card](https://huggingface.co/KBLab/kb-whisper-large).

## Reproducing
`scripts/bench.sh AUDIO [MODELS…]` times the configurations and reports RTF, peak memory, and WER against the baseline (`scripts/wer.py`). It works on macOS and Linux.

`scripts/bench-models.sh --audio DIR|FILE --refs DIR [--models "snabb small large"]` compares models against reference
transcripts: wall time, peak memory and WER (Swedish normalisation in `scripts/wer_sv.py`). It works in a fresh
`mktemp -d` folder and calls the `prata` CLI directly, so a running Prata server is not touched. Usage is at the top of the script.
