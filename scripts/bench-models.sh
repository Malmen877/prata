#!/usr/bin/env bash
# Compare Prata models (Snabb / KB-Whisper) on real recordings: wall time, peak memory, WER.
#
# Usage:
#   scripts/bench-models.sh --audio PATH --refs DIR [options]
#
#   --audio PATH     an audio file, or a directory of them (wav mp3 m4a flac ogg opus webm mp4)
#   --refs DIR       reference transcripts, one per audio file, matched by file stem:
#                    ref_<stem>.txt, <stem>.txt or <stem>.srt (plain text or SRT, UTF-8)
#   --models "..."   models to run (default: "snabb small large"; any `prata --model` id works)
#   --bin PATH       prata CLI to test (default: newest ~/.prata/*/prata, then `prata` on PATH,
#                    then this repo's target/release/prata)
#   --runs N         timed runs per model and file (default 1; the table shows every run)
#   --out DIR        also copy the results here (default: results stay in the temp folder, printed at the end)
#   --equiv FILE     extra word equivalences for the WER, "variant<TAB>canonical" per line (see wer_sv.py)
#   --no-warmup      skip the untimed warm-up run (by default each model first transcribes 10 s of
#                    the first file, so model downloads and first-load costs are not timed; needs ffmpeg)
#   --isolated-cache use an empty Hugging Face cache inside the temp folder (forces fresh downloads)
#   --wait-idle      before each run wait (max 10 min) until the 1-min load average is below 3
#   --extra "ARGS"   extra arguments for every prata run, e.g. "--vad off"
#
# Example (Mac mini):
#   scripts/bench-models.sh --bin ~/.prata/0.6.0/prata --audio ~/eval/clips --refs ~/eval/refs \
#       --models "snabb small large" --out ~/eval/results-$(date +%Y%m%d)
#
# Safety: everything is written to a fresh `mktemp -d` folder (plus --out if given). The script only runs
# the prata CLI directly; it never starts, stops or talks to a Prata server, so the installed
# LaunchAgent (se.prata.server) and port 8795 are not touched. The normal model cache
# (~/.cache/huggingface) is reused unless --isolated-cache is given.
#
# Output: table.txt (pretty), results.tsv, per-run .srt/.log files, and the WER per run computed
# by scripts/wer_sv.py (Swedish normalisation, Python 3 stdlib). Peak memory is the maximum resident set
# size from /usr/bin/time (-l on macOS, -v on Linux); on macOS "peak memory footprint" is shown too.
# Works with macOS' bash 3.2 and BSD tools.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
AUDIO="" REFS="" MODELS="snabb small large" BIN="" RUNS=1 OUT="" EQUIV="" WARMUP=1 ISO=0 IDLE=0 EXTRA=""
die() { echo "bench-models: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --audio) AUDIO=${2:?}; shift 2 ;;
    --refs) REFS=${2:?}; shift 2 ;;
    --models) MODELS=${2:?}; shift 2 ;;
    --bin) BIN=${2:?}; shift 2 ;;
    --runs) RUNS=${2:?}; shift 2 ;;
    --out) OUT=${2:?}; shift 2 ;;
    --equiv) EQUIV=${2:?}; shift 2 ;;
    --no-warmup) WARMUP=0; shift ;;
    --isolated-cache) ISO=1; shift ;;
    --wait-idle) IDLE=1; shift ;;
    --extra) EXTRA=${2:-}; shift 2 ;;
    -h|--help) sed -n '2,/^set -u/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done
[ -n "$AUDIO" ] || die "--audio is required (see --help)"
[ -n "$REFS" ] || die "--refs is required (see --help)"
[ -e "$AUDIO" ] || die "no such file or directory: $AUDIO"
[ -d "$REFS" ] || die "refs directory not found: $REFS"
case "$RUNS" in ''|*[!0-9]*) die "--runs must be a number" ;; esac
[ -z "$EQUIV" ] || [ -f "$EQUIV" ] || die "no such file: $EQUIV"
command -v python3 >/dev/null 2>&1 || die "python3 is needed for the WER"
[ -x /usr/bin/time ] || die "/usr/bin/time not found"

if [ -z "$BIN" ]; then
  # shellcheck disable=SC2012
  BIN=$(ls -1d "$HOME"/.prata/*/prata 2>/dev/null | sort -V 2>/dev/null | tail -n 1)
  [ -n "$BIN" ] || BIN=$(command -v prata 2>/dev/null || true)
  [ -n "$BIN" ] || BIN=$ROOT/target/release/prata
fi
[ -x "$BIN" ] || die "prata binary not found or not executable: $BIN (use --bin)"
case "$(basename "$BIN")" in prata-web*) die "--bin must be the prata CLI, not the web server" ;; esac

# absolute paths (we cd into the temp folder)
abspath() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")"); }
AUDIO=$(abspath "$AUDIO"); REFS=$(cd "$REFS" && pwd); BIN=$(abspath "$BIN")
EQARGS=()
[ -z "$EQUIV" ] || { EQUIV=$(abspath "$EQUIV"); EQARGS=(--equiv "$EQUIV"); }
[ -z "$OUT" ] || { mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd); } || die "cannot create $OUT"

T=$(mktemp -d "${TMPDIR:-/tmp}/prata-bench.XXXXXX") || die "mktemp failed"
mkdir -p "$T/runs"
cd "$T" || exit 1
[ "$ISO" = 1 ] && export HF_HOME="$T/hf"
export HF_HUB_DISABLE_TELEMETRY=1

if [ "$(uname)" = Darwin ]; then TIMEFLAG=-l; else TIMEFLAG=-v; fi

# audio list
FILES="$T/files.txt"
if [ -d "$AUDIO" ]; then
  find "$AUDIO" -maxdepth 1 -type f \( -iname '*.wav' -o -iname '*.mp3' -o -iname '*.m4a' -o -iname '*.flac' \
    -o -iname '*.ogg' -o -iname '*.opus' -o -iname '*.webm' -o -iname '*.mp4' \) | sort > "$FILES"
else
  echo "$AUDIO" > "$FILES"
fi
[ -s "$FILES" ] || die "no audio files in $AUDIO"

ref_for() {   # stem -> reference path or empty
  for c in "$REFS/ref_$1.txt" "$REFS/$1.txt" "$REFS/$1.srt" "$REFS/ref_$1.srt"; do
    [ -f "$c" ] && { echo "$c"; return; }
  done
}
duration() {
  if command -v ffprobe >/dev/null 2>&1; then ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null && return; fi
  if command -v afinfo >/dev/null 2>&1; then afinfo "$1" 2>/dev/null | awk '/estimated duration/ {print $3; exit}' && return; fi
  echo 0
}
wall_of() {
  if [ "$TIMEFLAG" = -l ]; then awk '/ real / {print $1; exit}' "$1"
  else awk -F': ' '/Elapsed \(wall clock\)/ {n=split($2,p,":"); s=0; for(i=1;i<=n;i++) s=s*60+p[i]; print s; exit}' "$1"; fi
}
rss_of() {   # MB
  if [ "$TIMEFLAG" = -l ]; then awk '/maximum resident set size/ {printf "%.0f", $1/1048576; exit}' "$1"
  else awk -F': ' '/Maximum resident set size/ {printf "%.0f", $2/1024; exit}' "$1"; fi
}
footprint_of() {   # MB, macOS only
  awk '/peak memory footprint/ {printf "%.0f", $1/1048576; exit}' "$1"
}
wait_idle() {
  [ "$IDLE" = 1 ] || return 0
  i=0
  while [ $i -lt 60 ]; do
    if [ "$(uname)" = Darwin ]; then l=$(sysctl -n vm.loadavg | awk '{print $2}'); else l=$(cut -d' ' -f1 /proc/loadavg); fi
    awk -v l="$l" 'BEGIN{exit !(l<3)}' && return 0
    sleep 10; i=$((i + 1))
  done
}

{
  echo "date:   $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "host:   $(uname -mrs) $(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  echo "binary: $BIN"
  echo "models: $MODELS   runs: $RUNS   extra: ${EXTRA:-none}   hf cache: ${HF_HOME:-default}"
} > "$T/env.txt"
cat "$T/env.txt"; echo "work dir: $T"

if [ "$WARMUP" = 1 ]; then
  if command -v ffmpeg >/dev/null 2>&1; then
    first=$(head -n 1 "$FILES")
    ffmpeg -loglevel error -y -i "$first" -t 10 -ac 1 -ar 16000 "$T/warmup.wav" </dev/null
    for m in $MODELS; do
      echo ">> warm-up $m (untimed; downloads the model on first use)" >&2
      # shellcheck disable=SC2086
      "$BIN" "$T/warmup.wav" --model "$m" $EXTRA > /dev/null 2> "$T/runs/warmup-$m.log" \
        || echo "   warm-up for $m failed, see $T/runs/warmup-$m.log" >&2
    done
  else
    echo "ffmpeg not found: skipping warm-up (first run of each model includes model loading/downloads)" >&2
  fi
fi

TSV="$T/results.tsv"
printf 'file\tdur_s\tmodel\trun\texit\twall_s\trtf\trss_mb\tfootprint_mb\twer\tsub\tdel\tins\tref_words\n' > "$TSV"
while IFS= read -r f; do
  base=$(basename "$f"); stem=${base%.*}
  dur=$(duration "$f"); ref=$(ref_for "$stem")
  [ -n "$ref" ] || echo "   no reference for $stem in $REFS (WER skipped)" >&2
  for m in $MODELS; do
    r=1
    while [ "$r" -le "$RUNS" ]; do
      tag="$stem.$(echo "$m" | tr '/' '_').$r"
      wait_idle
      echo ">> $stem  model=$m  run=$r" >&2
      # shellcheck disable=SC2086
      /usr/bin/time $TIMEFLAG "$BIN" "$f" --model "$m" --timestamps --out "$T/runs/$tag.srt" $EXTRA \
        > "$T/runs/$tag.stdout" 2> "$T/runs/$tag.log" < /dev/null
      st=$?
      wall=$(wall_of "$T/runs/$tag.log"); rss=$(rss_of "$T/runs/$tag.log"); fp=$(footprint_of "$T/runs/$tag.log")
      rtf=$(awk -v w="${wall:-0}" -v d="${dur:-0}" 'BEGIN { if (d > 0) printf "%.3f", w/d; else print "-" }')
      werline="- - - - -"
      hyp="$T/runs/$tag.srt"; [ -s "$hyp" ] || hyp="$T/runs/$tag.stdout"
      if [ "$st" -eq 0 ] && [ -n "$ref" ] && [ -s "$hyp" ]; then
        werline=$(python3 "$HERE/wer_sv.py" "$ref" "$hyp" ${EQARGS[@]+"${EQARGS[@]}"})
        python3 "$HERE/wer_sv.py" "$ref" "$hyp" --diff 40 ${EQARGS[@]+"${EQARGS[@]}"} > /dev/null 2> "$T/runs/$tag.diff.txt"
      fi
      # shellcheck disable=SC2086
      set -- $werline
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$stem" "${dur:-0}" "$m" "$r" "$st" "${wall:--}" "$rtf" \
        "${rss:--}" "${fp:--}" "$1" "$2" "$3" "$4" "$5" >> "$TSV"
      r=$((r + 1))
    done
  done
done < "$FILES"

awk -F'\t' 'NR==1 {printf "%-16s %8s %-8s %3s %4s %9s %7s %8s %9s %7s\n", "file","audio(s)","model","run","exit","wall(s)","RTF","RSS(MB)","foot(MB)","WER(%)"; next}
  {printf "%-16s %8.1f %-8s %3s %4s %9s %7s %8s %9s %7s\n", substr($1,1,16), $2, $3, $4, $5, $6, $7, $8, $9, $10}' "$TSV" > "$T/table.txt"
echo; cat "$T/table.txt"
echo
echo "WER: scripts/wer_sv.py (lowercase, punctuation removed, Swedish abbreviations/numbers normalised)."
echo "Per-run transcripts, logs and word diffs: $T/runs/"
if [ -n "$OUT" ]; then cp -R "$T"/. "$OUT"/ && echo "results copied to $OUT"; fi
echo "results: $T"
