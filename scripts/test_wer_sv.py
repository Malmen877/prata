#!/usr/bin/env python3
"""Self-test for scripts/wer_sv.py:  python3 scripts/test_wer_sv.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wer_sv as w

cases = {
    "Det kostar ca 1 000 kr, t.ex. i Rödeby.": "det kostar cirka 1000 kronor till exempel i rödeby",
    "Ökningen var 12.5 % – bl.a. i Malmö/Lund": "ökningen var 12,5 procent bland annat i malmö lund",
    "”Hej”, sa hon... e-post osv.": "hej sa hon e post och så vidare",
    "Han körde 12 km, dvs. långt.": "han körde 12 kilometer det vill säga långt",
    "ÅÄÖ åäö": "åäö åäö",
}
for a, b in cases.items():
    got = " ".join(w.norm(a))
    assert got == b, (a, got)
assert w.score("en två tre", "En, två – tre!")[0]["wer"] == 0
r = w.score("en två tre fyra", "en tre fyra fem")[0]
assert (r["sub"], r["del"], r["ins"], r["wer"]) == (0, 1, 1, 50.0), r
r = w.score("ett två", "ett tre")[0]
assert (r["sub"], r["wer"]) == (1, 50.0), r
assert w.srt_text("1\n00:00:00,000 --> 00:00:01,000\nHej där\n\n2\n00:00:01,000 --> 00:00:02,000\nAllihop\n") == "Hej där Allihop"
with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
    f.write("# comment\ntolvhåls\t12 håls\n")
eq = w.load_equiv(f.name); os.unlink(f.name)
assert w.score("en 12-håls bana", "en tolvhåls bana", equiv=eq)[0]["wer"] == 0
print("wer_sv tests: ok")
