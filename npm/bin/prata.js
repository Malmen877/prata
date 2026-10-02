#!/usr/bin/env node
// Prata launcher (npx prata-app): downloads the prebuilt Prata binaries for this platform
// from the GitHub Release matching this package's version, caches them under
// ~/.prata/<version>/, then starts the local web UI and opens the browser.
// Pure Node (>=18), no dependencies.
"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const http = require("http");
const https = require("https");
const net = require("net");
const crypto = require("crypto");
const { spawn, spawnSync } = require("child_process");

// TODO(Kevin): set your GitHub user/org here before publishing (or set PRATA_GITHUB_OWNER).
const DEFAULT_OWNER = "Malmen877";
const OWNER = process.env.PRATA_GITHUB_OWNER || DEFAULT_OWNER;
const REPO = process.env.PRATA_GITHUB_REPO || "prata";
const VERSION = require("../package.json").version;
const DEFAULT_PORT = 8795;

const TARGETS = { "darwin-arm64": "darwin-arm64", "darwin-x64": "darwin-x64", "linux-x64": "linux-x64" };

const argv = process.argv.slice(2);
const flag = (n) => argv.includes(n);
const opt = (n) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : undefined; };

function log(...a) { console.log("[prata]", ...a); }
function die(msg, code = 1) { console.error("[prata] " + msg); process.exit(code); }

function usage() {
  console.log(`Prata ${VERSION} – lokal svensk tal-till-text

Usage: npx prata-app [--port N] [--no-open] [--model snabb|small|large|tiny|base|medium]

Options:
  --port N       port to listen on (default ${DEFAULT_PORT}, or a free port)
  --no-open      do not open the browser
  --model M      default model in the UI (default small; the picker shows snabb, small and large)
  --version      print version
  --help         this help

Environment:
  PRATA_GITHUB_OWNER   GitHub owner of the release repo (default ${DEFAULT_OWNER})
  PRATA_LOCAL_ASSET    use this local prata-v<ver>-<target>.tar.gz instead of downloading
  PRATA_BIN_DIR        use binaries from this directory (no download, no cache)
  PRATA_CACHE_DIR      cache directory (default ~/.prata)
  PRATA_NO_BROWSER=1   same as --no-open`);
}

function target() {
  const key = `${process.platform}-${process.arch}`;
  const t = TARGETS[key];
  if (!t) die(`Unsupported platform ${key}. Prebuilt binaries exist for: ${Object.keys(TARGETS).join(", ")}.\n` +
              "        Build from source instead: https://github.com/" + OWNER + "/" + REPO);
  if (t === "linux-x64") checkGlibc();
  return t;
}

// The linux-x64 binaries are built on Ubuntu 24.04 (onnxruntime for Snabb needs it) and need
// glibc >= 2.39. Older systems would fail with an obscure loader error, so say it up front.
const MIN_GLIBC = [2, 39];
function checkGlibc() {
  if (process.env.PRATA_SKIP_GLIBC_CHECK === "1") return;
  let v = process.env.PRATA_FAKE_GLIBC; // tests only
  if (!v) { try { v = process.report.getReport().header.glibcVersionRuntime; } catch { v = undefined; } }
  if (!v) return; // not glibc (or unknown): let the loader decide
  const [maj, min] = String(v).split(".").map((x) => parseInt(x, 10));
  if (maj < MIN_GLIBC[0] || (maj === MIN_GLIBC[0] && min < MIN_GLIBC[1])) {
    die(`Prata ${VERSION} för Linux kräver glibc ${MIN_GLIBC.join(".")} eller senare (t.ex. Ubuntu 24.04, Debian 13, Fedora 40). ` +
        `Den här datorn har glibc ${v}.\n` +
        "        Uppgradera systemet, använd prata-app@0.5.1 (utan Snabb) eller bygg från källkod: https://github.com/" + OWNER + "/" + REPO);
  }
}

function checkFfmpeg() {
  const r = spawnSync("ffmpeg", ["-version"], { stdio: "ignore" });
  if (r.error || r.status !== 0) {
    const hint = process.platform === "darwin" ? "  brew install ffmpeg"
      : "  sudo apt install ffmpeg      (Debian/Ubuntu)\n  sudo dnf install ffmpeg      (Fedora)";
    die("ffmpeg was not found on PATH. Prata needs it to read audio. Install it with:\n\n" + hint + "\n");
  }
}

// yt-dlp is optional: only links to web pages (YouTube, SVT Play, podcasts …) need it.
function checkYtDlp() {
  const bin = process.env.PRATA_YTDLP || "yt-dlp";
  const r = spawnSync(bin, ["--version"], { stdio: "ignore" });
  if (r.error || r.status !== 0) {
    const hint = process.platform === "darwin" ? "brew install yt-dlp" : "pipx install yt-dlp   (or your package manager)";
    console.warn("[prata] note: yt-dlp was not found" + (process.env.PRATA_YTDLP ? ` (PRATA_YTDLP=${bin})` : " on PATH") +
      ". Transcribing links to web pages (YouTube, SVT Play, …) needs it:\n\n  " + hint +
      "\n\n        Direct links to audio/video files and uploads work without it.");
  }
}

function get(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { "User-Agent": `prata-app/${VERSION}` } }, (res) => {
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location && redirects < 10) {
        res.resume();
        resolve(get(new URL(res.headers.location, url).toString(), redirects + 1));
      } else if (res.statusCode !== 200) {
        res.resume();
        const e = new Error(`HTTP ${res.statusCode} for ${url}`); e.status = res.statusCode; reject(e);
      } else resolve(res);
    });
    req.on("error", reject);
    req.setTimeout(60000, () => req.destroy(new Error("timeout")));
  });
}

async function download(url, dest) {
  const res = await get(url);
  const total = Number(res.headers["content-length"]) || 0;
  let got = 0, lastPct = -1;
  const out = fs.createWriteStream(dest);
  await new Promise((resolve, reject) => {
    res.on("data", (c) => {
      got += c.length;
      if (total && process.stderr.isTTY) {
        const pct = Math.floor(got / total * 100);
        if (pct !== lastPct) { lastPct = pct; process.stderr.write(`\r[prata] downloading ${pct}% (${(total / 1048576).toFixed(1)} MB)`); }
      }
    });
    res.pipe(out);
    out.on("finish", resolve); out.on("error", reject); res.on("error", reject);
  });
  if (total && process.stderr.isTTY) process.stderr.write("\n");
}

async function fetchText(url) {
  const res = await get(url);
  let s = ""; for await (const c of res) s += c; return s;
}

function sha256(file) { return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex"); }

function extract(tarball, destDir) {
  const tmp = destDir + ".tmp-" + process.pid;
  fs.rmSync(tmp, { recursive: true, force: true });
  fs.mkdirSync(tmp, { recursive: true });
  const r = spawnSync("tar", ["-xzf", tarball, "-C", tmp], { stdio: "inherit" });
  if (r.error || r.status !== 0) { fs.rmSync(tmp, { recursive: true, force: true }); die("could not extract " + tarball + " (is `tar` installed?)"); }
  for (const b of ["prata", "prata-web"]) {
    const p = path.join(tmp, b);
    if (!fs.existsSync(p)) { fs.rmSync(tmp, { recursive: true, force: true }); die(`archive ${tarball} does not contain ${b}`); }
    fs.chmodSync(p, 0o755);
  }
  fs.rmSync(destDir, { recursive: true, force: true });
  fs.renameSync(tmp, destDir);
}

async function ensureBinaries() {
  if (process.env.PRATA_BIN_DIR) {
    const d = path.resolve(process.env.PRATA_BIN_DIR);
    if (!fs.existsSync(path.join(d, "prata-web"))) die(`PRATA_BIN_DIR=${d} has no prata-web binary`);
    return d;
  }
  const t = target();
  const cacheRoot = process.env.PRATA_CACHE_DIR || path.join(os.homedir(), ".prata");
  const dir = path.join(cacheRoot, VERSION);
  const ready = () => ["prata", "prata-web"].every((b) => fs.existsSync(path.join(dir, b)));
  if (ready() && !process.env.PRATA_LOCAL_ASSET) return dir;
  fs.mkdirSync(cacheRoot, { recursive: true });

  if (process.env.PRATA_LOCAL_ASSET) {
    const a = path.resolve(process.env.PRATA_LOCAL_ASSET);
    if (!fs.existsSync(a)) die("PRATA_LOCAL_ASSET not found: " + a);
    log(`using local asset ${a}`);
    extract(a, dir);
    return dir;
  }

  const asset = `prata-v${VERSION}-${t}.tar.gz`;
  const base = `https://github.com/${OWNER}/${REPO}/releases/download/v${VERSION}/`;
  log(`first run: downloading ${asset} …`);
  const tmpFile = path.join(cacheRoot, `${asset}.part-${process.pid}`);
  try {
    await download(base + asset, tmpFile);
  } catch (e) {
    fs.rmSync(tmpFile, { force: true });
    die(`download failed: ${e.message}\n        Is there a release v${VERSION} with ${asset} at github.com/${OWNER}/${REPO}?`);
  }
  try {
    const want = (await fetchText(base + asset + ".sha256")).trim().split(/\s+/)[0];
    const have = sha256(tmpFile);
    if (want && want !== have) { fs.rmSync(tmpFile, { force: true }); die(`checksum mismatch for ${asset} (expected ${want}, got ${have})`); }
  } catch (e) {
    if (e.status !== 404) log(`warning: could not verify checksum (${e.message})`);
  }
  extract(tmpFile, dir);
  fs.rmSync(tmpFile, { force: true });
  log(`installed to ${dir}`);
  return dir;
}

function portFree(port) {
  return new Promise((resolve) => {
    const s = net.createServer().once("error", () => resolve(false))
      .once("listening", () => s.close(() => resolve(true))).listen(port, "127.0.0.1");
  });
}
function randomPort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer().once("error", reject).listen(0, "127.0.0.1", () => {
      const p = s.address().port; s.close(() => resolve(p));
    });
  });
}
async function pickPort() {
  const want = Number(opt("--port") || process.env.PRATA_PORT || 0);
  if (want) { if (await portFree(want)) return want; die(`port ${want} is already in use`); }
  return (await portFree(DEFAULT_PORT)) ? DEFAULT_PORT : randomPort();
}

function waitFor(url, child, timeoutMs = 30000) {
  const t0 = Date.now();
  return new Promise((resolve, reject) => {
    const tryOnce = () => {
      if (child.exitCode !== null) return reject(new Error(`prata-web exited with code ${child.exitCode}`));
      const req = http.get(url, (res) => { res.resume(); res.statusCode === 200 ? resolve() : retry(); });
      req.on("error", retry);
      req.setTimeout(2000, () => req.destroy());
    };
    const retry = () => (Date.now() - t0 > timeoutMs ? reject(new Error("prata-web did not start in time")) : setTimeout(tryOnce, 200));
    tryOnce();
  });
}

function openBrowser(url) {
  if (flag("--no-open") || process.env.PRATA_NO_BROWSER) return;
  const cmd = process.platform === "darwin" ? "open" : "xdg-open";
  try {
    const p = spawn(cmd, [url], { stdio: "ignore", detached: true });
    p.on("error", () => log(`open ${url} in your browser`));
    p.unref();
  } catch (_) { log(`open ${url} in your browser`); }
}

async function main() {
  if (flag("--help") || flag("-h")) return usage();
  if (flag("--version") || flag("-v")) return console.log(VERSION);
  if (!process.env.PRATA_BIN_DIR) target(); // platform + glibc check before anything else
  checkFfmpeg();
  checkYtDlp();
  const dir = await ensureBinaries();
  const port = await pickPort();
  const env = { ...process.env, PRATA_WEB_PORT: String(port), PRATA_WEB_HOST: "127.0.0.1", PRATA_BIN: path.join(dir, "prata") };
  const py = path.join(dir, "transcribe.py");
  if (fs.existsSync(py) && !env.PRATA_PYTHON_SCRIPT) env.PRATA_PYTHON_SCRIPT = py;
  if (opt("--model")) env.PRATA_MODEL = opt("--model");

  // prata-web's own log lines (startup, jobs, Klang syncs …) go to the same stdout/stderr as
  // the launcher's, so a LaunchAgent's StandardOutPath/StandardErrorPath captures everything.
  const child = spawn(path.join(dir, "prata-web"), [], { env, stdio: ["ignore", "inherit", "inherit"] });
  child.on("error", (e) => die(`could not start prata-web: ${e.message}`));

  let stopping = false;
  const stop = (sig) => { if (stopping) return; stopping = true; log("stopping …"); child.kill(sig); setTimeout(() => child.kill("SIGKILL"), 5000).unref(); };
  process.on("SIGINT", () => stop("SIGINT"));
  process.on("SIGTERM", () => stop("SIGTERM"));
  process.on("SIGHUP", () => stop("SIGTERM"));
  child.on("exit", (code, signal) => {
    if (!stopping) log(`prata-web exited (${signal || code}).`);
    process.exit(stopping ? 0 : (code || 1));
  });

  const url = `http://127.0.0.1:${port}/`;
  try { await waitFor(url + "api/info", child); }
  catch (e) {
    if (child.exitCode !== null) { log(`prata-web exited (${child.exitCode}).`); process.exit(child.exitCode || 1); }
    stop("SIGTERM"); die(`${e.message} (see prata-web's messages above)`);
  }
  log(`Prata körs på ${url}  (Ctrl+C för att avsluta)`);
  log("Första transkriberingen laddar ner modellen från Hugging Face (Snabb ≈ 0,66 GB, Standard ≈ 0,97 GB, Large ≈ 3,1 GB).");
  openBrowser(url);
}

main().catch((e) => die(e.stack || String(e)));
