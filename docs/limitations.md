# Limitations

- Jobs run one at a time. A job that is still running when the server stops is lost; finished ones are saved as notes.
- Long recordings are kept in browser memory until you press stop.
- A word can occasionally be dropped or repeated at a 30-second window boundary (both KB-Whisper and Snabb decode in windows).
- Snabb is not available in the macOS Intel build.
- The microphone needs a secure context: use `http://127.0.0.1`/`localhost` on the Mac, or HTTPS (e.g. `tailscale serve`) from other devices – not a plain LAN IP.
