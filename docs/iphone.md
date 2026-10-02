# Use it from your iPhone (Tailscale)

Run Prata on an always-on Mac (e.g. a Mac mini) and reach it from your iPhone over [Tailscale](https://tailscale.com).
`tailscale serve` gives you a real HTTPS address, which iOS Safari needs for the microphone, while prata-web keeps
listening on `127.0.0.1` only.

1. Install Tailscale on the Mac (`brew install --cask tailscale` or the App Store app) and on the iPhone (App Store),
   and sign in to the **same tailnet** on both.
2. In the [admin console](https://login.tailscale.com/admin/dns) enable **MagicDNS** and **HTTPS Certificates**.
3. Start Prata on the Mac (see the LaunchAgent below to keep it running), then publish it inside your tailnet:

   ```bash
   tailscale serve --bg 8795        # https://<machine>.<tailnet>.ts.net  ->  http://127.0.0.1:8795
   tailscale serve status           # shows the URL;  `tailscale serve reset` turns it off
   ```

   (With the Mac App Store app the CLI is `/Applications/Tailscale.app/Contents/MacOS/Tailscale`.)
4. On the iPhone open `https://<machine>.<tailnet>.ts.net` in Safari, allow the microphone, then
   **Share → Add to Home Screen**. It opens full screen like an app (name *Prata*).

Use `tailscale serve`, not `tailscale funnel`: funnel publishes to the whole internet.

**Security:** prata-web has no login. Anyone who can reach it can record, read, download and delete notes.
With `tailscale serve` that means every device and user in your tailnet (restrict it with Tailscale ACLs if you share
the tailnet). Binding to another address with `--host 0.0.0.0` exposes it to your whole LAN as well, and the
microphone will not work there without HTTPS – prefer the default `127.0.0.1` plus `tailscale serve`.

## Keep it running on macOS (LaunchAgent)

Save as `~/Library/LaunchAgents/se.prata.web.plist` (adjust the paths to where you built or installed Prata;
replace `kevin` with your user name):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>se.prata.web</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/kevin/prata/target/release/prata-web</string>
    <string>--port</string><string>8795</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <!-- ffmpeg from Homebrew lives in /opt/homebrew/bin; launchd's default PATH does not include it -->
    <!-- … and so does yt-dlp (brew install yt-dlp) for links to web pages -->
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PRATA_MODEL</key><string>small</string>
    <!-- optional Klang import; the key is then stored in this file: chmod 600 it -->
    <!-- <key>KLANG_API_KEY</key><string>sk_…</string> -->
  </dict>
  <key>WorkingDirectory</key><string>/Users/kevin</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/Users/kevin/Library/Logs/prata-web.log</string>
  <key>StandardErrorPath</key><string>/Users/kevin/Library/Logs/prata-web.log</string>
</dict>
</plist>
```

```bash
plutil -lint ~/Library/LaunchAgents/se.prata.web.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/se.prata.web.plist     # start now and at every login
launchctl kickstart -k gui/$(id -u)/se.prata.web                              # restart (e.g. after rebuilding)
launchctl bootout gui/$(id -u)/se.prata.web                                   # stop and unload
tail -f ~/Library/Logs/prata-web.log
```

A LaunchAgent runs while your user is logged in; for a headless Mac mini enable automatic login
(System Settings → Users & Groups) and disable sleep (System Settings → Energy → "Prevent automatic sleeping").
