// Unit tests for pure UI functions in crates/prata-web/src/index.html (day grouping, safe markdown).
// Runs the real functions (extracted between the day-grouping markers) in a
// fixed time zone:  TZ=Europe/Stockholm node scripts/test-ui.mjs
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import vm from "node:vm";

const html = readFileSync(new URL("../crates/prata-web/src/index.html", import.meta.url), "utf8");
const m = /\/\/ --- day-grouping:start[^\n]*\n([\s\S]*?)\/\/ --- day-grouping:end/.exec(html);
assert.ok(m, "day-grouping markers not found");
const ctx = {};
vm.runInNewContext(m[1] + "\nthis.dayLabel = dayLabel; this.groupByDay = groupByDay; this.clockTime = clockTime;", ctx);
const { dayLabel, groupByDay, clockTime } = ctx;

const at = (y, mo, d, h = 12, mi = 0) => new Date(y, mo - 1, d, h, mi).getTime() / 1000;
const now = new Date(2026, 9, 2, 17, 45);            // fre 2 okt 2026 17:45 local

assert.equal(dayLabel(at(2026, 10, 2, 0, 0), now), "Idag");
assert.equal(dayLabel(at(2026, 10, 2, 23, 59), now), "Idag");
assert.equal(dayLabel(at(2026, 10, 1, 23, 59), now), "Igår");
assert.equal(dayLabel(at(2026, 10, 1, 0, 1), now), "Igår");
assert.equal(dayLabel(at(2026, 9, 30, 23, 59), now), "30 september");
assert.equal(dayLabel(at(2026, 9, 28), now), "28 september");
assert.equal(dayLabel(at(2026, 1, 1), now), "1 januari");
assert.equal(dayLabel(at(2025, 9, 28), now), "28 september 2025");
assert.equal(dayLabel(at(2025, 12, 31, 23, 0), new Date(2026, 0, 1, 9)), "Igår");   // across new year: no year on "Igår"
assert.equal(dayLabel(at(2025, 12, 30), new Date(2026, 0, 1, 9)), "30 december 2025");
// DST: 25 Oct 2026 is 25 h long in Stockholm; 26 Oct morning -> 25 Oct is "Igår"
assert.equal(dayLabel(at(2026, 10, 25, 0, 30), new Date(2026, 9, 26, 0, 30)), "Igår");
assert.equal(dayLabel(at(2026, 3, 29, 0, 30), new Date(2026, 2, 30, 0, 30)), "Igår");
assert.equal(clockTime(at(2026, 10, 2, 7, 5)), "07:05");
assert.equal(clockTime(at(2026, 10, 2, 23, 59)), "23:59");

const notes = [
  { id: "a", created: at(2025, 9, 28, 9) },
  { id: "b", created: at(2026, 10, 2, 8) },
  { id: "c", created: at(2026, 10, 1, 22) },
  { id: "d", created: at(2026, 10, 2, 16) },
  { id: "e", created: at(2026, 9, 28, 10) },
  { id: "f", created: at(2026, 9, 28, 18) },
];
const g = JSON.parse(JSON.stringify(groupByDay(notes, now)));   // plain objects from the vm realm
assert.deepEqual(g.map(x => x.label), ["Idag", "Igår", "28 september", "28 september 2025"]);
assert.deepEqual(g.map(x => x.notes.map(n => n.id).join("")), ["db", "c", "fe", "a"]);
assert.equal(groupByDay([], now).length, 0);
console.log("ui day-grouping tests: ok (TZ=" + Intl.DateTimeFormat().resolvedOptions().timeZone + ")");

// ---- safe markdown subset for Klang summaries
const mm = /\/\/ --- markdown:start[^\n]*\n([\s\S]*?)\/\/ --- markdown:end/.exec(html);
assert.ok(mm, "markdown markers not found");
const mctx = {};
vm.runInNewContext(mm[1] + "\nthis.mdToHtml = mdToHtml;", mctx);
const md = mctx.mdToHtml;
assert.equal(md("## Beslut\n- **Ja** till budget\n- Nej till `rm -rf`\n\nKlart."),
  '<h4 class="md-h md-h2">Beslut</h4><ul><li><strong>Ja</strong> till budget</li><li>Nej till <code>rm -rf</code></li></ul><p>Klart.</p>');
assert.equal(md("1. Ett\n2. Två"), "<ol><li>Ett</li><li>Två</li></ol>");
assert.equal(md("- [ ] Göra\n- [x] Gjort"), "<ul><li class=\"task\">☐ Göra</li><li class=\"task\">☑ Gjort</li></ul>");
assert.equal(md("rad 1\nrad *två*"), "<p>rad 1<br>rad <em>två</em></p>");
// no raw HTML, no script, no javascript: links, attributes can't be broken out of
assert.equal(md("<script>alert(1)</script>"), "<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>");
assert.equal(md('<img src=x onerror="alert(1)">'), "<p>&lt;img src=x onerror=&quot;alert(1)&quot;&gt;</p>");
assert.equal(md("[klicka](javascript:alert(1))"), "<p>klicka)</p>");
assert.equal(md('[x](https://a.se/"onmouseover="alert(1))'), '<p><a href="https://a.se/&quot;onmouseover=&quot;alert(1" target="_blank" rel="noopener noreferrer">x</a>)</p>');
assert.equal(md("[Länk](https://ex.se/a_b_c)"), '<p><a href="https://ex.se/a_b_c" target="_blank" rel="noopener noreferrer">Länk</a></p>');
assert.equal(md("Se https://ex.se/x_y_z."), '<p>Se <a href="https://ex.se/x_y_z" target="_blank" rel="noopener noreferrer">https://ex.se/x_y_z</a>.</p>');
assert.equal(md("> citat\n\n---"), "<blockquote>citat</blockquote><hr>");
assert.equal(md(""), "");
assert.ok(!/<(?!\/?(p|br|ul|ol|li|h4|strong|em|code|a|blockquote|hr)\b)/.test(md("<b>x</b> <iframe> <svg onload=1> \u0000 x")), "only whitelisted tags");
console.log("ui markdown tests: ok");
assert.equal(md("- punkt\n  fortsätter"), "<ul><li>punkt fortsätter</li></ul>");

// ---- job progress: model download on first use (KB-Whisper and Snabb) vs transcription
const pm = /\/\/ --- progress:start[^\n]*\n([\s\S]*?)\/\/ --- progress:end/.exec(html);
assert.ok(pm, "progress markers not found");
const pctx = {};
vm.runInNewContext(pm[1] + "\nthis.jobProgress = jobProgress; this.progressPct = progressPct;", pctx);
const jp = j => JSON.parse(JSON.stringify(pctx.jobProgress(j)));
const run = (lines, extra = {}) => ({ status: "running", audio_duration: 300, log_tail: lines, progress: lines[lines.length - 1] || "", ...extra });
// KB-Whisper, unchanged: window lines and tqdm-style percentages
assert.deepEqual(jp(run(["[info] model=KBLab/kb-whisper-small device=Cpu", "[info] window 30.0s done in 9.1s"])), { pct: 20, phase: null });
assert.deepEqual(jp(run(["Transcribing: 45%|████"])), { pct: 45, phase: null });
assert.deepEqual(jp({ status: "downloading", download_pct: 12.6 }), { pct: 13, phase: null });
assert.deepEqual(jp({ status: "done" }), { pct: 100, phase: null });
assert.deepEqual(jp({ status: "queued" }), { pct: null, phase: null });
// KB-Whisper first use: download line without a percentage -> indeterminate model phase
assert.deepEqual(jp(run(["[info] downloading KBLab/kb-whisper-small/model.safetensors ..."])), { pct: null, phase: "model" });
assert.deepEqual(jp(run(["[info] downloading KBLab/kb-whisper-small/model.safetensors ...", "[info] model=KBLab/kb-whisper-small device=Cpu"])), { pct: null, phase: null });
// Snabb first use: per-file percentage from stderr, aggregate model_download_pct wins
const snabbHead = "[info] downloading KlangAI/pianissimo-sv-onnx/encoder-model.int8.onnx (Klang Pianissimo, CC BY 4.0) ...";
assert.deepEqual(jp(run([snabbHead])), { pct: null, phase: "model" });
assert.deepEqual(jp(run([snabbHead, "[info] downloading encoder-model.int8.onnx: 42% of 630 MB"])), { pct: 42, phase: "model" });
assert.deepEqual(jp(run([snabbHead, "[info] downloading encoder-model.int8.onnx: 42% of 630 MB"], { model_download_pct: 44.4 })), { pct: 44, phase: "model" });
// download done, transcription started: percentage comes from windows again, not from "100% of"
assert.deepEqual(jp(run([snabbHead, "[info] downloading encoder-model.int8.onnx: 100% of 630 MB", "[info] window 0.0s done in 1.2s (1/10)"], { model_download_pct: 100 })), { pct: 10, phase: null });
assert.deepEqual(jp(run(["[info] window 60.0s: no speech, skipped (3/10)"])), { pct: 30, phase: null });
assert.equal(pctx.progressPct(run(["[info] window 30.0s done in 9.1s"])), 20);
console.log("ui progress tests: ok");

// recording watchdog (cx: safe recording)
const rw = /\/\/ --- recwatch:start[^\n]*\n([\s\S]*?)\/\/ --- recwatch:end/.exec(html);
assert.ok(rw, "recwatch markers not found");
const rctx = {};
vm.runInNewContext(rw[1] + "\nthis.recWatchMsg = recWatchMsg; this.recGapMs = recGapMs;", rctx);
const { recWatchMsg, recGapMs } = rctx;
const T = 1_000_000, ok = { now: T, lastChunk: T - 900, lastSound: T - 200, lostMs: 0, muted: false, ended: false };
assert.equal(recWatchMsg(ok), "");
assert.equal(recWatchMsg({ ...ok, lastSound: T - 9000 }), "", "short silence is fine");
assert.match(recWatchMsg({ ...ok, lastSound: T - 12000 }), /^Hör inget – är mikrofonen på\? Inget ljud på 12 s\.$/);
assert.match(recWatchMsg({ ...ok, lastChunk: T - 7000 }), /stannat – inga nya ljuddata på 7 s/);
assert.match(recWatchMsg({ ...ok, muted: true, lastSound: T - 60000 }), /pausad av systemet/, "muted wins over silence");
assert.match(recWatchMsg({ ...ok, ended: true, muted: true }), /kopplades bort/);
assert.equal(recWatchMsg({ ...ok, lostMs: 1000 }), "", "a hiccup under 1.5 s is not reported");
assert.equal(recWatchMsg({ ...ok, lostMs: 42400 }), "Inspelningen pausades när skärmen låstes – ca 42 s saknas.");
assert.equal(recGapMs(T, T - 2000, T - 1500), 0, "chunks kept coming (desktop tab in the background)");
assert.equal(recGapMs(T, T - 30000, T - 29500), 29500, "screen locked 29.5 s after the last chunk");
assert.equal(recGapMs(T, T - 30000, T - 40000), 30000);
console.log("ui recording watchdog tests: ok");

// ---- cx: eta (needs jobProgress from the progress block)
const em = /\/\/ --- cx: eta ---\n([\s\S]*?)\/\/ --- cx: notify ---/.exec(html);
assert.ok(em, "cx: eta block not found");
const ectx = {};
vm.runInNewContext(pm[1] + em[1] + "\nconst _e = cxEta; this.cxEta = j => _e(j).replace(/\\u00a0/g, ' '); this.cxEtaText = cxEtaText;", ectx);
const ej = (pct, t, extra = {}) => ({ id: "j1", status: "running", elapsed_s: t, audio_duration: 600, progress_pct: pct, log_tail: [], ...extra });
assert.equal(ectx.cxEta(ej(10, 10)), "");          // first sample: nothing yet
assert.equal(ectx.cxEta(ej(12, 13)), "");          // too little progress: still hidden
assert.equal(ectx.cxEta(ej(30, 30)), " · ca 1 min kvar");   // 20 % in 20 s -> 70 s left
assert.match(ectx.cxEta(ej(40, 40)), /^ · (ca 1 min|under 1 min) kvar$/);
assert.equal(ectx.cxEta(ej(40, 40, { status: "done" })), "");
assert.equal(ectx.cxEta(ej(50, 5, { id: "j2", log_tail: ["[info] downloading encoder-model.int8.onnx: 50% of 630 MB"], progress_pct: undefined })), "");
assert.equal(ectx.cxEtaText(30), "under 1 min kvar");
assert.equal(ectx.cxEtaText(150), "ca 3 min kvar");
assert.equal(ectx.cxEtaText(4000), "ca 1 h 7 min kvar");
console.log("ui eta tests: ok");
