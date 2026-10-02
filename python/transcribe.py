#!/usr/bin/env python3
"""Swedish speech-to-text with KBLab kb-whisper (Hugging Face transformers).

Usage:
  python3 transcribe.py <audio file> [--model KBLab/kb-whisper-large|medium|small]
                        [--out file.txt] [--timestamps]
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # hide noisy deprecation/experimental warnings

SR = 16000
SHORT = {"tiny", "base", "small", "medium", "large"}


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_audio(path):
    """Decode any ffmpeg-readable file (mp3, m4a, ogg, wav, mp4, ...) to 16 kHz mono float32."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError:
        sys.exit("ffmpeg not found - install it (apt install ffmpeg / brew install ffmpeg).")
    except subprocess.CalledProcessError as e:
        sys.exit(f"ffmpeg failed to decode {path}:\n{e.stderr.decode(errors='replace')}")
    audio = np.frombuffer(out, dtype=np.float32)
    if audio.size == 0:
        sys.exit(f"No audio decoded from {path}")
    return audio


def srt_time(t):
    t = max(0.0, float(t))
    h, rem = divmod(int(t * 1000), 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(chunks, total):
    lines = []
    for i, c in enumerate(chunks, 1):
        start, end = c["timestamp"]
        if start is None:
            start = 0.0
        if end is None:
            end = total
        lines += [str(i), f"{srt_time(start)} --> {srt_time(end)}", c["text"].strip(), ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Swedish transcription with KBLab kb-whisper")
    ap.add_argument("audio")
    ap.add_argument("--model", default="KBLab/kb-whisper-large",
                    help="HF model id, or short name: small|medium|large (default: large)")
    ap.add_argument("--out", help="write transcript (or SRT with --timestamps) to this file")
    ap.add_argument("--timestamps", action="store_true", help="output SRT-style timestamps")
    ap.add_argument("--batch-size", type=int, default=4, help="chunks decoded in parallel")
    args = ap.parse_args()

    model_id = f"KBLab/kb-whisper-{args.model}" if args.model in SHORT else args.model
    device, dtype = pick_device()
    print(f"[info] model={model_id} device={device} dtype={str(dtype).replace('torch.', '')}",
          file=sys.stderr)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=dtype,
                                                      low_cpu_mem_usage=True,
                                                      use_safetensors=True)
    model.to(device)
    asr = pipeline("automatic-speech-recognition", model=model,
                   tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
                   dtype=dtype, device=device, ignore_warning=True)
    t_load = time.time() - t0

    audio = load_audio(args.audio)
    dur = len(audio) / SR

    t1 = time.time()
    result = asr({"raw": audio, "sampling_rate": SR},
                 chunk_length_s=30, batch_size=args.batch_size,
                 return_timestamps=args.timestamps,
                 generate_kwargs={"language": "sv", "task": "transcribe"})
    t_asr = time.time() - t1

    if args.timestamps:
        text = to_srt(result.get("chunks", []), dur)
    else:
        text = result["text"].strip()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[info] wrote {args.out}", file=sys.stderr)
    print(text)
    print(f"[info] audio={dur:.1f}s load={t_load:.1f}s transcribe={t_asr:.1f}s "
          f"(RTF {t_asr / dur:.2f})", file=sys.stderr)


if __name__ == "__main__":
    main()
