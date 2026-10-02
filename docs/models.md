# Models

Speed and accuracy figures are in [benchmarks.md](benchmarks.md).

The web UI offers three models (the CLI and HTTP API also accept `tiny`, `base` and `medium`):

| In the UI | Model | Description (as shown) |
|---|---|---|
| **Snabb** | Klang Pianissimo (`snabb`) | Klang Pianissimo – mycket bra svenska, snabbast. Rekommenderas för långa inspelningar |
| **Standard** (default) | KB-Whisper small (`small`) | Bästa balansen mellan kvalitet och tid |
| **Large** | KB-Whisper large (`large`) | Största modellen, långsammast |

`tiny`, `base` and `medium` are hidden in the UI but remain available via the CLI (`--model tiny`), the HTTP API
(`"model": "medium"`) and `PRATA_MODEL`. Standard stays preselected and Prata never switches models by itself.

**Long-recording hint.** When a file or recording is longer than 15 minutes and a model other than Snabb is selected,
the UI asks the server (`GET /api/model-hint?duration_s=…&model=…`) and shows a small hint
(“Lång inspelning – Snabb går betydligt fortare. Byt till Snabb”). It switches to Snabb only when you click it.
The rule lives on the server (`crates/prata-web/src/hint.rs`); there is no hint when this build has no Snabb or the
duration is unknown.

## Snabb (Klang Pianissimo)

[Pianissimo](https://huggingface.co/KlangAI/pianissimo-sv) is a Swedish Parakeet TDT model by KlangAI. Prata runs
KlangAI's official int8 ONNX export, [KlangAI/pianissimo-sv-onnx](https://huggingface.co/KlangAI/pianissimo-sv-onnx),
unmodified, with ONNX Runtime on the CPU. Pianissimo is a fine-tune of NVIDIA Parakeet TDT 0.6B v3. It writes punctuation
and casing itself and gives sentence-level timestamps. Long audio is decoded in 30-second windows with overlapping
context on both sides.

- **First use:** the model files (about 660 MB = 630 MiB, the same size in the cache: `encoder-model.int8.onnx`,
  `decoder_joint-model.int8.onnx`, `vocab.txt`)
  are downloaded from Hugging Face into the normal cache (`~/.cache/huggingface/`, `HF_HOME` respected). The web UI
  shows the download progress; later runs start straight away.
- **Platforms:** macOS Apple Silicon and Linux x64 (glibc 2.39+, e.g. Ubuntu 24.04+; the prebuilt ONNX Runtime needs it). The macOS Intel build has no Snabb; the option is shown greyed out there.
- **CLI:** `prata recording.m4a --model snabb --timestamps` (`pianissimo`, `KlangAI/pianissimo-sv` and
  `KlangAI/pianissimo-sv-onnx` are accepted too). The model revision is pinned to the one this version was tested with.
- **Window seams:** a word seen by both windows (a cut inside the word, e.g. "gör gör") is dropped by timestamp
  overlap; genuine repeats such as "Rödeby. Rödeby är" stay. Features are normalised over the whole recording.
- **No Python fallback:** if the build has no Snabb, a Snabb job fails with a Swedish message. It does not silently
  run another model.
- **License:** CC BY 4.0, © KlangAI – see [Credits and licenses](../NOTICE).
