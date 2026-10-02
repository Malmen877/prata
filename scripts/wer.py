#!/usr/bin/env python3
"""Compare two transcripts (SRT or plain text): normalised WER + word diff, and for
SRT inputs a segment-start alignment check.

usage: wer.py REF HYP [--shift CUT:GAP,...] [--diff] [--tol 0.3] [--quiet]

Normalisation: lowercase, punctuation stripped, whitespace split.
--shift adds GAP seconds to REF times at/after CUT (for clips with inserted silence).
--quiet prints only the WER (as a percentage) for scripts.
"""
import re, sys, difflib, argparse

def parse(path):
    txt = open(path, encoding="utf-8").read()
    segs = []
    if "-->" in txt:
        for block in re.split(r"\n\s*\n", txt.strip()):
            lines = block.strip().splitlines()
            for i, l in enumerate(lines):
                m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", l)
                if m:
                    g = [int(x) for x in m.groups()]
                    s = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
                    e = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000
                    segs.append((s, e, " ".join(lines[i+1:])))
                    break
    else:
        segs.append((None, None, txt))
    return segs

def norm(t):
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return t.split()

def words(segs):
    out = []  # (word, seg_start, is_first)
    for s, e, t in segs:
        for i, w in enumerate(norm(t)):
            out.append((w, s, i == 0))
    return out

def wer(r, h):
    # Levenshtein on word lists
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j-1] + 1, prev + (r[i-1] != h[j-1]))
            prev = cur
    return d[len(h)] / max(1, len(r))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("hyp")
    ap.add_argument("--shift", default="")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--tol", type=float, default=0.3)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    shifts = [tuple(map(float, x.split(":"))) for x in a.shift.split(",") if x]
    def sh(t):
        return None if t is None else t + sum(g for c, g in shifts if t >= c - 1e-6)
    ref = [(sh(s), sh(e), t) for s, e, t in parse(a.ref)]
    hyp = parse(a.hyp)
    rw, hw = words(ref), words(hyp)
    r, h = [w for w, _, _ in rw], [w for w, _, _ in hw]
    w = wer(r, h)
    if a.quiet:
        print(f"{100*w:.2f}")
        return
    print(f"WER {100*w:.2f}%  ({len(r)} ref words, {len(h)} hyp words)")
    sm = difflib.SequenceMatcher(a=r, b=h, autojunk=False)
    diffs = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if diffs:
        print(f"{len(diffs)} differing spans:")
        for tag, i1, i2, j1, j2 in diffs:
            ctx_l = " ".join(r[max(0, i1-3):i1]); ctx_r = " ".join(r[i2:i2+3])
            t = rw[i1][1] if i1 < len(rw) else (rw[-1][1] if rw else None)
            ts = f"{t:7.2f}s" if t is not None else "      -"
            print(f"  {ts}  …{ctx_l} [{' '.join(r[i1:i2]) or '∅'} → {' '.join(h[j1:j2]) or '∅'}] {ctx_r}…")
    # segment-start alignment: segments in both that start at the same aligned word
    if ref and ref[0][0] is not None and hyp and hyp[0][0] is not None:
        deltas = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                continue
            for k in range(i2 - i1):
                (_, rs, rf), (_, hs, hf) = rw[i1+k], hw[j1+k]
                if rf and hf:
                    deltas.append((rs, hs - rs))
        if deltas:
            ad = sorted(abs(d) for _, d in deltas)
            bad = [(s, d) for s, d in deltas if abs(d) > a.tol]
            print(f"segment starts: {len(deltas)} common (of {len(ref)} ref / {len(hyp)} hyp segs); "
                  f"median |Δ| {ad[len(ad)//2]:.2f}s, max {ad[-1]:.2f}s, {len(bad)} over {a.tol}s")
            for s, d in bad:
                print(f"    ref start {s:7.2f}s  Δ {d:+.2f}s")

main()
