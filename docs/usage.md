# Using Prata

## Features
- Record from the microphone or upload a file (wav, mp3, m4a, ogg, flac, opus, webm, mp4 – anything ffmpeg reads)
- **Transcribe from a link:** paste a YouTube/SVT Play/podcast page (needs [yt-dlp](usage.md#transcribe-from-a-link)) or a direct link to an audio/video file
- Transcript with segment timestamps, a waveform player, click-to-seek timestamps, search and copy
- Downloads: `.txt`, `.txt` with timestamps, `.srt` subtitles, `.json`
- Three models in the UI – **Snabb**, **Standard** (default) and **Large** – plus a hint to use Snabb for recordings over 15 minutes; light and dark mode
- **Notes:** every transcription is saved (text + original audio) and listed by day, with search across all notes, rename and delete
- Phone-friendly: one big record button, installable to the iPhone Home Screen, reachable from your phone over [Tailscale](iphone.md)
- Also a command-line tool: `prata interview.m4a --timestamps`

## Notes

Every finished transcription is stored as a folder per note:

```
~/.prata/notes/<id>/note.json     title, created (unix time), model, audio duration, segments with timestamps,
                                  source_url (only for notes made from a link); Klang imports also have
                                  source "klang", summary, klang {id, updated_at} and title_edited
~/.prata/notes/<id>/audio.<ext>   the original upload or recording (m4a from iPhone, webm from Chrome, …)
```

The default title is the local date and time plus the first words of the transcript; rename it in the UI.
Writes are atomic (temporary file + fsync + rename), so a crash never leaves a half-written note, and the folder is
plain files: back it up with Time Machine, copy it to another Mac, or delete a folder to remove a note.
The list is grouped by day (Idag, Igår, 28 september, …) using the browser's local time.

## Transcribe from a link

Paste a link in **Klistra in länk** under the record button. Prata downloads the audio on the Mac, then
transcribes it like an upload; the note keeps the downloaded audio, gets the video/episode title as its name and
links back to the source.

- **Direct links** to audio/video files (`.mp3`, `.m4a`, `.mp4`, `.wav`, …) work out of the box.
  The note is named after the file (from `Content-Disposition` if the server sends one, else the last part of
  the URL, without extension, `_` as spaces); generic names like `download.mp3` keep the default date title.
- **Web pages** (YouTube, SVT Play, Vimeo, most podcast pages, …) need [yt-dlp](https://github.com/yt-dlp/yt-dlp):
  `brew install yt-dlp` (keep it updated with `brew upgrade yt-dlp`; sites change often). Prata finds it on `PATH`
  (set `PATH` in the [LaunchAgent](iphone.md#keep-it-running-on-macos-launchagent)) or via `PRATA_YTDLP`, and picks it up without a restart.
  Without it the UI shows a hint and direct links still work.
- Limits: 500 MB, 3 hours of audio and a 15 minute download timeout by default (see [cli.md](cli.md#web-server-configuration)).
  Live streams, private videos and pages that need a login are refused with a message.

**Security.** Links are fetched by the Mac, so Prata refuses anything that is not a public internet address:
only `http`/`https`, no `user:password@` links, and every address the host resolves to must be public –
loopback, private (10/8, 172.16/12, 192.168/16), CGNAT/Tailscale (100.64/10), link-local (169.254/16, incl. cloud
metadata), IPv6 unique-local/link-local, multicast, unspecified and IPv4-mapped/NAT64/6to4 forms of those are blocked.
The check runs on the resolved addresses inside the HTTP client (the address that was checked is the one connected
to, so DNS rebinding does not help), again on every redirect (at most 5), and no proxy settings are taken from the
environment. yt-dlp is started without a shell, with the link as a single argument after `--`, `--ignore-config`,
no plugins, no `--exec`, playlists off, size and duration filters, and its output must stay inside the job's temp
folder. Note that yt-dlp itself fetches the media URLs the site points to; Prata checks the link you pasted
(and its redirects) but not those secondary requests. Anyone who can reach your Prata can make the Mac download
things, so keep it on `127.0.0.1` + `tailscale serve` as described in [iphone.md](iphone.md).
