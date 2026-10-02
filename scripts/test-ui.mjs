// Unit test for the Notes day grouping in crates/prata-web/src/index.html.
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
