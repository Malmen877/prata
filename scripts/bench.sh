#!/usr/bin/env bash
# Benchmark Prata's decoding modes on one audio file.
#
#   scripts/bench.sh AUDIO [MODELS...]          (default models: small large)
#
# Environment:
#   PRATA_BIN   prata binary to test           (default: target/release/prata)
#   BASE_BIN    binary used for "baseline"     (default: PRATA_BIN with all speed-ups off, which
#               runs the original sequential decoder; set to a main-branch build to compare)
#   BATCH       batch size for batched configs (default: 4)
#   RUNS        runs per config                (default: 1)
#   CONFIGS     configs to run (default: baseline vad kv default default+pack;
#               also available: batch pack — see cfg_args below)
#   EXTRA       extra args for every run, e.g. "--cpu"
#   OUT         results directory              (default: bench-out/<timestamp>)
#
# Works with macOS' bash 3.2 and BSD tools (uses /usr/bin/time -l there, -v on Linux).
# WER needs python3 (scripts/wer.py); without it the column shows "n/a".
set -u

AUDIO=${1:?usage: scripts/bench.sh AUDIO [MODELS...]}
shift
MODELS="$*"
[ -n "$MODELS" ] || MODELS="small large"
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PRATA_BIN=${PRATA_BIN:-$ROOT/target/release/prata}
BASE_BIN=${BASE_BIN:-$PRATA_BIN}
BATCH=${BATCH:-4}
RUNS=${RUNS:-1}
CONFIGS=${CONFIGS:-"baseline vad kv default default+pack"}
EXTRA=${EXTRA:-}
OUT=${OUT:-bench-out/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

[ -x "$PRATA_BIN" ] || { echo "no binary at $PRATA_BIN (cargo build --release -p prata [--features metal])" >&2; exit 1; }
[ -f "$AUDIO" ] || { echo "no such file: $AUDIO" >&2; exit 1; }

if [ "$(uname)" = Darwin ]; then TIMEFLAG=-l; else TIMEFLAG=-v; fi
HAVE_PY=0; command -v python3 >/dev/null 2>&1 && HAVE_PY=1

duration() {
  if command -v ffprobe >/dev/null 2>&1; then
    ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" && return
  fi
  if command -v afinfo >/dev/null 2>&1; then
    afinfo "$1" | awk '/estimated duration/ {print $3; exit}' && return
  fi
  echo 0
}
DUR=$(duration "$AUDIO")

# config name -> args
#   baseline      the original sequential decoder (all speed-ups off)
#   vad           VAD only
#   batch         batching only (regions batched; without VAD there is one region)
#   kv            decoder KV cache only
#   default       the CLI defaults (KV cache + VAD, batch size auto)
#   default+pack  defaults plus experimental --pack --batch-size $BATCH
#   pack          --pack --batch-size $BATCH without VAD/KV cache
cfg_args() {
  case "$1" in
    baseline)     echo "--kv-cache off --vad off --batch-size 1" ;;
    vad)          echo "--kv-cache off --vad on --batch-size 1" ;;
    batch)        echo "--kv-cache off --vad off --batch-size $BATCH" ;;
    kv)           echo "--kv-cache on --vad off --batch-size 1" ;;
    default)      echo "" ;;
    default+pack) echo "--pack --batch-size $BATCH" ;;
    pack)         echo "--kv-cache off --vad off --pack --batch-size $BATCH" ;;
    *) echo "unknown config $1" >&2; exit 1 ;;
  esac
}

# wall seconds and peak RSS (MB) from a /usr/bin/time log
wall_of() {
  if [ "$TIMEFLAG" = -l ]; then
    awk '/ real / {print $1; exit}' "$1"
  else
    awk -F': ' '/Elapsed \(wall clock\)/ {n=split($2,p,":"); s=0; for(i=1;i<=n;i++) s=s*60+p[i]; print s; exit}' "$1"
  fi
}
rss_of() {
  if [ "$TIMEFLAG" = -l ]; then
    awk '/maximum resident set size/ {printf "%.0f", $1/1048576; exit}' "$1"
  else
    awk -F': ' '/Maximum resident set size/ {printf "%.0f", $2/1024; exit}' "$1"
  fi
}

echo "audio: $AUDIO (${DUR}s)  binary: $PRATA_BIN  batch: $BATCH  runs: $RUNS  out: $OUT"
ROWS="$OUT/table.txt"
printf "%-6s %-13s %-4s %9s %7s %8s %9s %s\n" model config run "wall(s)" RTF "RSS(MB)" "WER(%)" "same-as-baseline" > "$ROWS"

for model in $MODELS; do
  for cfg in $CONFIGS; do
    r=1
    while [ "$r" -le "$RUNS" ]; do
      bin=$PRATA_BIN; [ "$cfg" = baseline ] && bin=$BASE_BIN
      if [ "$cfg" = baseline ] && [ "$BASE_BIN" != "$PRATA_BIN" ]; then args=""; else args=$(cfg_args "$cfg"); fi
      tag="$model-$cfg-$r"
      echo ">> $tag: $bin --model $model --timestamps $args $EXTRA" >&2
      # shellcheck disable=SC2086
      /usr/bin/time $TIMEFLAG "$bin" "$AUDIO" --model "$model" --timestamps --out "$OUT/$tag.srt" $args $EXTRA \
        > /dev/null 2> "$OUT/$tag.log"
      st=$?
      wall=$(wall_of "$OUT/$tag.log"); rss=$(rss_of "$OUT/$tag.log")
      rtf=$(awk -v w="$wall" -v d="$DUR" 'BEGIN { if (d > 0) printf "%.2f", w/d; else print "n/a" }')
      werv="-"; same="-"
      if [ "$st" -ne 0 ]; then
        werv="FAIL($st)"
      elif [ "$cfg" != baseline ] && [ -f "$OUT/$model-baseline-1.srt" ]; then
        if cmp -s "$OUT/$model-baseline-1.srt" "$OUT/$tag.srt"; then same=identical; else same=DIFFERENT; fi
        if [ "$HAVE_PY" = 1 ]; then
          werv=$(python3 "$HERE/wer.py" "$OUT/$model-baseline-1.srt" "$OUT/$tag.srt" --quiet)
          python3 "$HERE/wer.py" "$OUT/$model-baseline-1.srt" "$OUT/$tag.srt" > "$OUT/$tag.diff.txt"
        else
          werv="n/a"
        fi
      fi
      printf "%-6s %-13s %-4s %9s %7s %8s %9s %s\n" "$model" "$cfg" "$r" "$wall" "$rtf" "$rss" "$werv" "$same" >> "$ROWS"
      r=$((r + 1))
    done
  done
done
echo
cat "$ROWS"
echo
echo "WER is vs the same model's baseline run 1 (normalised: lowercase, no punctuation)."
echo "same-as-baseline compares the SRT files byte for byte. Word diffs: $OUT/*.diff.txt"
