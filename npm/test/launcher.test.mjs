// Launcher test with a fake prata-web: its stdout/stderr must reach the launcher's own
// stdout/stderr (stdio inherit), signals must be forwarded and exit codes kept.
// Run: node --test npm/test/   (no network, no real binaries)
import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import net from "node:net";

const LAUNCHER = new URL("../bin/prata.js", import.meta.url).pathname;
const KEY = "sk_fake_launcher_test_91ab";

const FAKE_WEB = `#!/usr/bin/env node
const http = require("http");
console.log("fake-web: stdout line, klang enabled=" + (process.env.KLANG_API_KEY ? "yes" : "no"));
console.error("fake-web: stderr line");
if (process.env.FAKE_MODE === "exit3") process.exit(3);
const s = http.createServer((q, r) => { r.setHeader("content-type", "application/json"); r.end("{}"); });
s.listen(Number(process.env.PRATA_WEB_PORT), "127.0.0.1");
process.on("SIGTERM", () => { console.log("fake-web: got SIGTERM"); process.exit(0); });
`;

function fakeBinDir() {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "prata-launcher-"));
  const w = (name, body) => { fs.writeFileSync(path.join(d, name), body); fs.chmodSync(path.join(d, name), 0o755); };
  w("prata-web", FAKE_WEB);
  w("prata", "#!/bin/sh\nexit 0\n");
  w("ffmpeg", "#!/bin/sh\nexit 0\n");
  w("yt-dlp", "#!/bin/sh\necho 2026.01.01\n");
  return d;
}

function freePort() {
  return new Promise((res) => { const s = net.createServer(); s.listen(0, "127.0.0.1", () => { const p = s.address().port; s.close(() => res(p)); }); });
}

function run(extraEnv, port) {
  const dir = fakeBinDir();
  const child = spawn(process.execPath, [LAUNCHER, "--port", String(port), "--no-open"], {
    env: { ...process.env, PATH: dir + path.delimiter + process.env.PATH, PRATA_BIN_DIR: dir, PRATA_CACHE_DIR: dir, KLANG_API_KEY: KEY, ...extraEnv },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const out = { stdout: "", stderr: "" };
  child.stdout.on("data", (c) => (out.stdout += c));
  child.stderr.on("data", (c) => (out.stderr += c));
  const exited = new Promise((res) => child.on("exit", (code, signal) => res({ code, signal })));
  return { child, out, exited };
}

async function until(fn, ms = 10000) {
  const t0 = Date.now();
  while (!fn()) { if (Date.now() - t0 > ms) throw new Error("timeout"); await new Promise((r) => setTimeout(r, 50)); }
}

test("child output reaches the launcher's stdout/stderr; SIGTERM is forwarded", async () => {
  const { child, out, exited } = run({}, await freePort());
  await until(() => out.stdout.includes("Prata körs"));
  assert.match(out.stdout, /fake-web: stdout line, klang enabled=yes/);
  assert.match(out.stderr, /fake-web: stderr line/);
  child.kill("SIGTERM");
  const r = await exited;
  assert.equal(r.code, 0);
  assert.match(out.stdout, /fake-web: got SIGTERM/);
  assert.ok(!(out.stdout + out.stderr).includes(KEY), "key must not be printed");
});

test("a crashing prata-web keeps its exit code and its messages", async () => {
  const { out, exited } = run({ FAKE_MODE: "exit3" }, await freePort());
  const r = await exited;
  assert.equal(r.code, 3);
  assert.match(out.stderr, /fake-web: stderr line/);
  assert.match(out.stdout + out.stderr, /prata-web exited/);
  assert.ok(!(out.stdout + out.stderr).includes(KEY));
});

test("linux: too old glibc gives a clear Swedish error before any download", { skip: process.platform !== "linux" || process.arch !== "x64" }, async () => {
  const { spawnSync } = await import("node:child_process");
  const cache = fs.mkdtempSync(path.join(os.tmpdir(), "prata-glibc-"));
  const env = { ...process.env, PRATA_FAKE_GLIBC: "2.35", PRATA_CACHE_DIR: cache, PRATA_NO_BROWSER: "1" };
  delete env.PRATA_BIN_DIR;
  const r = spawnSync(process.execPath, [LAUNCHER, "--no-open"], { env, encoding: "utf8", timeout: 20000 });
  assert.equal(r.status, 1, r.stderr);
  assert.match(r.stderr, /kräver glibc 2\.39 eller senare/);
  assert.match(r.stderr, /glibc 2\.35/);
  assert.equal(fs.readdirSync(cache).length, 0, "nothing downloaded");
  const ok = spawnSync(process.execPath, ["-e", "process.stdout.write(String(process.report.getReport().header.glibcVersionRuntime))"], { encoding: "utf8" });
  assert.ok(ok.stdout.length > 0);
});
