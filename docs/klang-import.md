# Import from Klang

If you also record meetings with [Klang](https://klang.ai), Prata can import them as notes (read-only – nothing
is ever written to Klang, and no audio is downloaded). Create an API key in Klang and start prata-web with it:

```sh
KLANG_API_KEY=sk_… prata-web          # or add it to the LaunchAgent's EnvironmentVariables
```

The Notes list then shows **Synka från Klang**. A sync reads all ready conversations (following `next_cursor`
while `has_more`, 100 per page) and reports e.g. *Klang: 2 nya, 1 uppdaterad, 14 oförändrade.* (plus
*N hoppades över* only when something could not be imported).
Imported notes appear in the same day groups with a small **Klang** label and show Klang's summary (Markdown,
rendered as plain formatted text – raw HTML is never inserted) above the speaker-labeled transcript. They have no
player and no `.srt`; `.txt` and `.json` work, and `.txt med tider` when the transcript has timestamps.
The `.txt` exports start with the summary as plain text (headings, bullets, links as text), then `----` and the
transcript (in `.txt`, one speaker turn per paragraph).

How re-syncs behave:

- **Dedupe:** a note remembers its Klang conversation id (`klang.id` in `note.json`, folder `klang-<id>`);
  syncing again never creates a second copy, also after a restart.
- **Updates:** conversations whose `updated_at` hasn't changed are skipped without fetching them again.
  Otherwise Prata fetches it and updates the note if the title, summary, transcript or date changed.
  If you renamed the note in Prata, your title is kept; everything else follows Klang.
- **Titles:** Klang's title when it has a real one. Untitled conversations (Klang calls them e.g. *30 sep. 10:51*)
  get a title from the summary – its first specific heading (not *Sammanfattning*, *Beslut* …), else its first
  sentence – then the digest, then the first words of the transcript (without timestamps and *Talare 1:* labels),
  and only last the date. Markdown is stripped and long titles are cut at a word with *…* (about 60 characters).
  `title_source` in `note.json` records where it came from (`klang`, `summary`, `digest`, `transcript`, `date`,
  `user`). Automatic titles are refreshed on every sync from the locally stored summary/transcript – no extra
  API calls – so notes imported by older versions get the better title on the next sync. Renamed notes
  (`title_edited`) are never touched.
- **Deleted here stays deleted:** deleting an imported note adds its Klang id to `.klang-deleted.json` in the notes
  folder, and later syncs skip it (not counted in the message). To get it back, remove the id from that file (or delete the
  file) and sync again.
- Only conversations with status `ready` are listed and imported; pending/failed ones are ignored until they're ready.
- **Date:** Klang's `created_at` (`started_at` if Klang ever sends it), else `updated_at`, else the import time.
- **Transcript:** lines like `[00:01:02] Ada: …` become timestamped segments; transcripts without timestamps
  are shown one line per speaker turn without times.
- **Limits and errors:** only one sync runs at a time. Klang's free plan allows 50 API calls per day (one per page
  plus one per new or changed conversation). On HTTP 429 Prata waits `Retry-After` (up to 60 s, 3 retries);
  longer waits stop the sync with *försök igen om …*. 5xx and network errors are retried briefly. A wrong key
  shows *Ogiltig Klang-nyckel*. Whatever was imported before an error is kept.
- **The key** is only sent in the `Authorization` header to the Klang API (no redirects are followed). It is never
  written to notes, logs, error messages or API responses (an integration test runs the server with a fake key
  and checks its stdout/stderr). Sync results are logged as `[klang] Klang: …` lines.
