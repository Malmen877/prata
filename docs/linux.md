# Linux

## Requirements

- **x86_64 with glibc 2.39 or newer**, e.g. Ubuntu 24.04+, Debian 13+, Fedora 40+ and similar. The prebuilt
  binaries are built on Ubuntu 24.04, because the prebuilt ONNX Runtime used for Snabb needs glibc 2.38+.
- ffmpeg: `sudo apt install ffmpeg` (or your distro's package).
- Optional: yt-dlp for links to web pages (see [usage.md](usage.md#transcribe-from-a-link)).

On an older glibc, `npx prata-app` stops early with a Swedish message instead of an obscure loader error. On, for
example, Ubuntu 22.04 you can:

- stay on 0.5.1 (`npx prata-app@0.5.1`; it has no Snabb), or
- build from source without Snabb (see below).

The Linux build runs on the CPU only (no GPU). Snabb and all KB-Whisper models are available in the prebuilt
binaries. For speed on CPU, see [benchmarks.md](benchmarks.md).

## Build from source

```bash
cargo build --release                          # without Snabb; works on older glibc
cargo build --release -p prata --features snabb && cargo build --release -p prata-web   # with Snabb; needs glibc 2.38+
```

Without the `snabb` feature, `prata --help` ends with *Snabb (Klang Pianissimo): not available in this build.* The
web UI then shows Snabb greyed out with a reason, and a Snabb job fails with a Swedish message. KB-Whisper is not
affected. See [development.md](development.md) for the rest of the build and test setup.

## Intel AMX

On x86_64, Snabb keeps ONNX Runtime from using Intel AMX (seccomp, all threads). ONNX Runtime's AMX int8 kernels
lost state when the thread was preempted (reproduced in a VM under load), which gave corrupted output. This makes
Snabb about 10 % slower on CPUs with AMX. `PRATA_SNABB_AMX=1` re-enables AMX. If seccomp is unavailable, a `[warn]`
line is printed and the run continues.
