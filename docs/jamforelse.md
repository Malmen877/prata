# Prata jämfört med MacWhisper, Sagt, Aloud och Transkribera

En saklig jämförelse av appar som gör svensk tal-till-text lokalt på datorn. Uppgifterna om de andra apparna kommer
bara från deras egna officiella sidor (se [Källor](#källor)) och kontrollerades **4 oktober 2026**. Där en sida inte
säger något om en sak står det **okänt**. Det betyder att vi inte har kunnat bekräfta det, inte att funktionen saknas.
Priser och funktioner ändras; kontrollera alltid hos tillverkaren. Hittar du ett fel, [öppna ett issue](https://github.com/Malmen877/prata/issues).

| | Prata | MacWhisper | Sagt (gratisnivån) | Aloud | Transkribera |
|---|---|---|---|---|---|
| **Huvudsyfte** | Transkribera inspelningar och ljudfiler i webbläsaren | Transkribera filer, möten och diktering | Transkribera möten i realtid | Diktering: text klistras in där markören står | Skrivbordsapp: spela in och transkribera |
| **Pris** | Gratis | Gratis grundversion; Pro engångsköp €64 (macwhisper.com; Gumroad visade €65) | Gratis lokalt; Pro 199 kr/månad | 199 kr engångsköp, 7 dagars gratis provperiod | Gratis nedladdning från GitHub Releases |
| **Källkod** | Öppen, MIT | okänt | Öppen, MIT | okänt | Publik på GitHub; ingen licensfil i repot |
| **Svenska modeller** | KB-Whisper small och large, Klang Pianissimo | Whisper-modeller; egna GGML-modeller i Pro (KBLab publicerar KB-Whisper i GGML för MacWhisper) | KB-Whisper Small lokalt; KB-Whisper Large i molnet med Pro | KB-Whisper (förval), Pianissimo (beta) | KB-Whisper medium och large; även Whisper large-v3-turbo |
| **Mac** | Apple Silicon och Intel (Intel utan Snabb) | macOS 14+; M-serie rekommenderas, fungerar på Intel | macOS 14.2+, bara Apple Silicon | macOS 14+, bara Apple Silicon | Nämns (Metal), men senaste release (v1.7.0) har bara Windows-filer |
| **Windows** | Nej | okänt | Windows 10/11 | okänt | Ja |
| **Linux** | Ja, x64 (glibc 2.39+) | okänt | Nej (”No Linux build exists”) | okänt | okänt |
| **Ljudfiler** | Ja, allt ffmpeg läser, även länkar (YouTube m.fl. med yt-dlp) | Ja, dra och släpp | okänt | okänt | Ja (MP3, M4A, WAV, MP4 m.fl.) |
| **Flera filer i kö** | Ja, med Avbryt och Försök igen | Ja, batch i Pro | okänt | okänt | okänt |
| **Mötes-/systemljud** | Nej, mikrofon | Ja (Zoom, Teams m.fl.) | Ja, datorns ljud + mikrofon | okänt | Ja på Windows (WASAPI) |
| **Ordlista** | Ja (rättar kända felstavningar i nya transkriptioner) | okänt | okänt | Ja, personlig ordlista och ersättningsregler | Ja, ord skickas som ledtrådar till modellen |
| **Skydd för inspelning vid krasch** | Ja, sparas löpande i webbläsaren | okänt | okänt | okänt | Ja, streamas till disk under inspelningen |
| **Export** | .txt, .txt med tider, .srt, .json | .srt, .vtt, .txt gratis; docx, pdf, md, html m.fl. i Pro | Urklipp gratis; Word, Markdown, text i Pro | Texten klistras in i aktiv app | .txt och .wav |
| **Vad lämnar datorn** | Inget ljud; bara nedladdning av appen och modellerna (och länkar du själv klistrar in) | Lokala modeller; molntranskribering och AI-tjänster valfritt | Inget ljud på gratisnivån; uppdateringskoll, app-konfiguration och produktanalys (PostHog, EU) skickas | Inget ljud; licensaktivering och uppdateringskoll | Bara modellnedladdning |

## Kort sagt

- **Prata** passar om du vill ha något gratis med öppen källkod för intervjuer, möten och föreläsningar som ljudfiler
  eller mikrofoninspelningar, med KB-Whisper eller den snabba Pianissimo, och om du kör Linux. Den spelar inte in
  datorns ljud och har ingen diktering i andra appar.
- **MacWhisper** har flest funktioner (talarigenkänning, mötesinspelning, diktering, många exportformat), men är
  byggd för Whisper-modeller generellt; KB-Whisper läggs in som egen GGML-modell i Pro.
- **Sagt** är gjord för möten: den fångar ljudet från Teams/Zoom direkt, gratis och lokalt med KB-Whisper Small.
- **Aloud** är en dikteringsapp: håll inne en tangent, prata, och texten hamnar där du skriver.
- **Transkribera** är en gratis skrivbordsapp med KB-Whisper, i första hand för Windows.

Prata-uppgifterna kommer från det här repot ([README](../README.md), [usage.md](usage.md), [models.md](models.md),
[install.md](install.md)), version 0.7.0. Mät hastighet och träffsäkerhet själv: [benchmarks.md](benchmarks.md).

## Källor

Alla kontrollerade 4 oktober 2026.

- MacWhisper: [macwhisper.com](https://www.macwhisper.com/) (officiell sida, pris €64) och
  [goodsnooze.gumroad.com/l/macwhisper](https://goodsnooze.gumroad.com/l/macwhisper) (funktionslista, systemkrav,
  pris €65). Observera att macwhisper.org *inte* är den officiella sidan enligt macwhisper.com.
- KB-Whisper i GGML för MacWhisper: [huggingface.co/KBLab/kb-whisper-large](https://huggingface.co/KBLab/kb-whisper-large).
- Sagt: [sagt.ai/sv](https://sagt.ai/sv), [sagt.ai/sv/downloads](https://sagt.ai/sv/downloads) och
  [github.com/sagt-ai/sagt-desktop](https://github.com/sagt-ai/sagt-desktop) (Free vs Pro, plattformar, licens, vad som skickas).
- Aloud: [aloud.talk](https://aloud.talk/) och [aloud.talk/tal-till-text/mac](https://aloud.talk/tal-till-text/mac).
- Transkribera: [github.com/mrswedish/transcrptr](https://github.com/mrswedish/transcrptr) (README och
  [Releases](https://github.com/mrswedish/transcrptr/releases)).
