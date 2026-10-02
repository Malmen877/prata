#!/usr/bin/env python3
"""Word error rate for Swedish transcripts (stdlib only).

usage: wer_sv.py REF HYP [--json] [--raw] [--diff N] [--equiv FILE]

REF/HYP: plain text or SRT (timestamps and cue numbers are dropped).
Prints "WER% sub del ins ref_words hyp_words" (or one JSON object with --json).

Normalisation (both sides), so that only real word differences count:
  * Unicode NFC, lowercase (casefold)
  * quotes, dashes and ellipses unified; hyphens and slashes split words ("e-post" -> "e post")
  * numbers: "1 000" / "1 000" -> "1000", decimal "12.5" -> "12,5", "%" -> "procent"
  * common abbreviations expanded: ca, t.ex., bl.a., s.k., m.m., osv., d.v.s., km, kr, nr, st ...
  * all other punctuation removed; å, ä, ö and other letters are kept
Not normalised: number words vs digits ("tolv" vs "12"), spelling variants. For those, --equiv FILE
takes one "variant<TAB>canonical" pair per line (e.g. "tolvhåls<TAB>12 håls"), applied to both sides
after normalisation; lines starting with # are ignored.
--raw also prints a case- and punctuation-sensitive WER (tokens split on whitespace only).
--diff N lists the first N differing spans.
"""
import argparse, difflib, json, re, sys, unicodedata

ABBR = [  # (regex on lowercased text, replacement); applied before punctuation is stripped
    (r"\bt\.\s?ex\.?(?=\s|$)", "till exempel"), (r"\bbl\.\s?a\.?(?=\s|$)", "bland annat"),
    (r"\bs\.\s?k\.?(?=\s|$)", "så kallade"), (r"\bm\.\s?m\.?(?=\s|$)", "med mera"),
    (r"\bd\.\s?v\.\s?s\.?(?=\s|$)", "det vill säga"), (r"\bdvs\.?(?=\s|$)", "det vill säga"),
    (r"\bo\.\s?s\.\s?v\.?(?=\s|$)", "och så vidare"), (r"\bosv\.?(?=\s|$)", "och så vidare"),
    (r"\bfr\.\s?o\.\s?m\.?(?=\s|$)", "från och med"), (r"\bt\.\s?o\.\s?m\.?(?=\s|$)", "till och med"),
    (r"\bca\.?(?=\s|$)", "cirka"), (r"\bkm\b", "kilometer"), (r"\bkr\b\.?", "kronor"),
    (r"\bnr\b\.?", "nummer"), (r"\bst\b\.(?=\s|$)", "stycken"), (r"\bmilj\.(?=\s|$)", "miljoner"),
    (r"\bmdr\b\.?", "miljarder"),
]

def srt_text(s):
    if "-->" not in s:
        return s
    keep = []
    for line in s.splitlines():
        t = line.strip()
        if not t or "-->" in t or t.isdigit():
            continue
        keep.append(t)
    return " ".join(keep)

def norm(s):
    s = unicodedata.normalize("NFC", s).casefold()
    s = s.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    s = re.sub(r"[“”„\"«»‘’‚`´]", " ", s)
    s = re.sub(r"[–—‐‑−]", "-", s).replace("…", " ")
    s = re.sub(r"(?<=\d) (?=\d{3}\b)", "", s)            # 1 000 -> 1000 (repeat for 1 000 000)
    s = re.sub(r"(?<=\d) (?=\d{3}\b)", "", s)
    s = re.sub(r"(?<=\d)\.(?=\d)", ",", s)                 # 12.5 -> 12,5
    s = s.replace("%", " procent ")
    for pat, rep in ABBR:
        s = re.sub(pat, rep, s)
    s = re.sub(r"(?<=\d),(?=\d)", "\u0000", s)            # protect decimal comma
    s = re.sub(r"[-/]", " ", s)
    s = re.sub(r"[^\w\s\u0000]", " ", s).replace("_", " ")
    return s.replace("\u0000", ",").split()

def align(r, h):
    """Levenshtein with backtrace -> (sub, del, ins)."""
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): d[i][0] = i
    for j in range(m + 1): d[0][j] = j
    for i in range(1, n + 1):
        ri, row, prev = r[i - 1], d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ri != h[j - 1]))
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            S += r[i - 1] != h[j - 1]; i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I

def load_equiv(path):
    pairs = []
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line.lstrip().startswith("#") or "\t" not in line:
            continue
        a, b = line.rstrip("\n").split("\t", 1)
        pairs.append((norm(a), norm(b)))
    return pairs

def apply_equiv(words, pairs):
    if not pairs:
        return words
    s = " " + " ".join(words) + " "
    for a, b in pairs:
        if a:
            s = s.replace(" " + " ".join(a) + " ", " " + " ".join(b) + " ")
    return s.split()

def score(ref_text, hyp_text, tok=norm, equiv=()):
    r, h = apply_equiv(tok(ref_text), equiv), apply_equiv(tok(hyp_text), equiv)
    S, D, I = align(r, h)
    return {"wer": 100.0 * (S + D + I) / max(1, len(r)), "sub": S, "del": D, "ins": I, "ref_words": len(r), "hyp_words": len(h)}, r, h

def main(argv=None):
    ap = argparse.ArgumentParser(description="Swedish WER (normalised)")
    ap.add_argument("ref"); ap.add_argument("hyp")
    ap.add_argument("--json", action="store_true"); ap.add_argument("--raw", action="store_true")
    ap.add_argument("--diff", type=int, default=0, metavar="N")
    ap.add_argument("--equiv", metavar="FILE", help="variant<TAB>canonical pairs")
    a = ap.parse_args(argv)
    ref = srt_text(open(a.ref, encoding="utf-8").read())
    hyp = srt_text(open(a.hyp, encoding="utf-8").read())
    res, r, h = score(ref, hyp, equiv=load_equiv(a.equiv) if a.equiv else ())
    if a.raw:
        res["wer_raw"] = score(ref, hyp, tok=lambda s: unicodedata.normalize("NFC", s).split())[0]["wer"]
    if a.json:
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}, ensure_ascii=False))
    else:
        print(f"{res['wer']:.2f} {res['sub']} {res['del']} {res['ins']} {res['ref_words']} {res['hyp_words']}"
              + (f" raw={res['wer_raw']:.2f}" if a.raw else ""))
    if a.diff:
        shown = 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
            if tag == "equal": continue
            print(f"  …{' '.join(r[max(0, i1-3):i1])} [{' '.join(r[i1:i2]) or '∅'} → {' '.join(h[j1:j2]) or '∅'}] {' '.join(r[i2:i2+3])}…", file=sys.stderr)
            shown += 1
            if shown >= a.diff: break
    return 0

if __name__ == "__main__":
    sys.exit(main())
