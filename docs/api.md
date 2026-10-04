# HTTP API

HTTP API: `POST /api/jobs` (multipart `file`, `model`) → `{id}`; `POST /api/jobs/url` (`{"url": "…", "model": "small"}`)
→ `{id}` or `400 {error}` for an invalid/blocked link; `GET /api/jobs/{id}` (status `downloading|queued|converting|running|done|error`,
`download_pct`, `model_download_pct` / `model_download_file` while a model is downloaded on first use, segments, `note_id` when done, `error_user` with a readable message);
`GET /api/jobs/{id}/{txt|txt-ts|srt|json}`; `GET /api/info` (includes `version`, `default_model` and `models`: id, label,
description, availability, download size, whether it's downloaded, license); `GET /api/model-hint?duration_s=S&model=M`
(`{suggest: "snabb"|null, reason}`); `GET /api/health` (`{ok, prata, ffmpeg, yt_dlp: {available, version}}`).

Notes API: `GET /api/notes?q=` (newest first, `{notes, total}`; every search word must occur in the title, transcript or summary),
`GET /api/notes/{id}`, `PATCH /api/notes/{id}` with `{"title": "…"}`, `DELETE /api/notes/{id}` (removes the audio too),
`GET /api/notes/{id}/audio` (original audio, supports `Range` for seeking), `GET /api/notes/{id}/{txt|txt-ts|srt|json}`.

Klang API (only when enabled, else `404`): `POST /api/klang/sync` starts a sync – or joins the one already running –
and answers `202 {running, started_at, last}`; `GET /api/klang/sync` → `{enabled, running, last: {new, updated,
unchanged, skipped, deleted_here, error, message, finished_at}}` (`skipped` = could not be imported,
`deleted_here` = deleted in Prata earlier). `GET /api/info` has `klang_enabled` (never the key).

Word list (Ordlista): `GET /api/wordlist` → `{entries: [{right, wrong: […]}]}`; `PUT /api/wordlist` with the same shape
replaces the list (validated: trimmed, empty rows dropped, max 1000 entries × 30 wrong forms, 100 characters each;
`400 {error}` in Swedish otherwise) and answers with the cleaned list. Stored in `wordlist.json` next to the notes
directory (default `~/.prata/wordlist.json`). It is applied to every new transcription: case-insensitive, whole words
(åäö count as letters), a space matches any whitespace, longest wrong form first, no re-replacement.
`POST /api/notes/{id}/wordlist` applies the current list to an existing note → `{replaced: n}`.
