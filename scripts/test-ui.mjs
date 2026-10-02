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
