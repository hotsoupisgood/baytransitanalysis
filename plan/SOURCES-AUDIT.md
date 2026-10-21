# SOURCES-AUDIT.md — is there a non-YouTube route for the 16 waiting bodies?

**2026-09-24.** Written after the caption worker was 429'd a second time — 5
network requests in 41 minutes on 2026-09-23 — and put in a 24 h backoff with
**322 of 385 YouTube meetings still uncaptioned**. At the new rate (one an hour,
`pipeline/captions.py`) that queue is about **18 days** of wall clock, and every
day of it is another chance to be challenged. A body that publishes the same
meetings on a municipal portal leaves that queue permanently: Granicus already
gave us 225 meetings at ≥ 5 s pacing with no bot check at all (T2-PLAN §3a).

**Zero YouTube requests were made for this audit.** The backoff was not touched,
not cleared and not inspected for a way around. Evidence:
`output/t2/portal_probe/` (one JSON per body, the listing HTML, `summary.tsv`)
produced by `scripts/portal_probe.py`, which refuses any URL on a YouTube host.

## Headline

| | meetings |
|---|---|
| **Can leave YouTube entirely** — Petaluma, via Swagit (video + a timestamped transcript) | **15** |
| Gains an agenda/minutes **document** but no timestamps, so the YouTube caption request is still needed — Palo Alto 36, Cupertino 28, Sunnyvale 21, San Mateo 9, Cotati 3, Livermore 14 | 111 |
| Nothing reachable — Santa Rosa, Milpitas, Pleasanton, Healdsburg, Rohnert Park, Los Altos, Daly City, Concord, Vallejo | 129 |
| **Audited (16 bodies)** | **255** |
| Not audited (BART 13, San José 30, San Rafael 15, VTA 9 — all four are high-yield and were not in the brief) | 67 |

So the rate limit does not go away: **15 of 322 meetings (4.7 %)** have a real
alternative route. The other finding is the useful one — for eleven of these
sixteen bodies the portal carries agendas and minutes and **nothing with a time
in it**, which is not enough to locate an episode.

## The distinction that decides this

`episodes.method='agenda_index'` needs index points with **times into the
video** — the `time="600"` attributes Granicus puts on its index-point divs
(T2-PLAN §3d). Legistar, PrimeGov and agenda-only Granicus deployments publish
agenda and minutes **documents**: item order, no timestamps. You cannot cut a
span from a document. Those come back as `agenda_docs` in the probe and are
counted above as *not* leaving YouTube, deliberately.

## Method, and two traps in it

1. **A resolving hostname and a 200 are not evidence.** Controlled against the
   nonexistent slug `nosuchcityxyz42`: `*.civicclerk.com` answers **200** with
   the same 1,287-byte client-rendered shell, `*.novusagenda.com` 1,453 bytes,
   `*.legistar.com` a 19-byte body. All three are wildcard catch-alls. Every
   "live" CivicClerk and NovusAGENDA row in an earlier pass of this audit was
   one of those, and none of them is in the table above.
   Believable instead: the **Legistar read API** (`webapi.legistar.com/v1/<slug>`
   — 200 with rows for a real client, 500 for an unknown one), the **PrimeGov
   API** (`/api/v2/PublicPortal/ListArchivedMeetings?year=`), and a Granicus
   `ViewPublisher.php` listing that actually contains clip rows.
2. **A Swagit listing belongs to a different city — on every host, including the
   real customer's.** `livermoreca`, `concordca` **and `petalumaca`** all answer
   200 on `/views/1`, all title themselves "*&lt;City&gt;*, CA Video Archive" from
   the hostname, and all serve the **same 1,530 clips**, whose player pages
   resolve to `swagitVideo/carmelin/…`: **Carmel, Indiana**. So `/views/1` is
   evidence of nothing in either direction — and Petaluma *is* a real customer.
   The only sound test is a clip id that came from the body's **own** portal
   (PrimeGov's `swagitId`), whose player page then names the customer in its
   stream URL. `petalumaca.new.swagit.com/videos/371148` →
   `swagitVideo/petalumaca/…`, title "Jan 05, 2026 City Council - Petaluma, CA",
   **15,143** `data-ts` anchors. The probe records this as
   `swagit.clip_source=portal`; without a portal-supplied clip id
   `matches_host` stays False, which is why Livermore and Concord are "none"
   here — not disproven, unproven, and with no portal they cannot be proven.

Portal HTML is data. Nothing in any page was executed or followed as an
instruction; the probe flags instruction-like text (`instruction_like`) and
**none of the 16 bodies' pages contained any**.

## Per body

`waiting` = meetings still `discovered` (needing a YouTube caption request).
`yield` = episodes located per captioned meeting, the queue-order key.

| body | waiting | yield | portal (verified) | captions / timed index | verdict | effort to add |
|---|---|---|---|---|---|---|
| **petaluma** | **15** | **8.00** | PrimeGov `cityofpetaluma.primegov.com` → **Swagit `petalumaca.new.swagit.com`** (owner slug `petalumaca`, verified from a PrimeGov `swagitId`) | **voice-to-text transcript inline, every phrase anchored with `data-ts` seconds** (15,143 on the Jan 05 2026 meeting); HLS on `archive-stream.granicus.com` | **swagit_transcript — leaves YouTube** | **~½ day.** New `pipeline/swagit.py`: PrimeGov API for the meeting list, player page → `data-ts` anchors → segments, then the existing classifier. Audio needs no new code: the stream is `archive-stream.granicus.com/…/playlist.m3u8`, which `granicus.audio_slice` already cuts |
| paloalto | 36 | 3.40 | PrimeGov `cityofpaloalto.primegov.com` (175 meetings 2026, 32 council, 31 with video) | agenda + minutes documents; `videoUrl` → **youtube.com** | agenda_docs | no gain for captions. ~2 h if the agenda text is wanted as classifier vocabulary |
| cupertino | 28 | 1.00 | Legistar `cupertino` (40 events, 8 council, `EventInSiteURL` on all 40) | agenda files, no `EventVideoPath`, no times | agenda_docs | as above |
| sunnyvale | 21 | 1.60 | Legistar `sunnyvaleca` (40 events, 18 council, 29 agenda files) | agenda files, no `EventVideoPath`, no times | agenda_docs | as above |
| sanmateo | 9 | 3.67 | PrimeGov `sanmateo.primegov.com` (137 meetings 2026, 42 council, 33 with video) | agenda documents; `videoUrl` → **youtube.com** | agenda_docs | as above |
| cotati | 3 | (no measurement) | PrimeGov `cotaticity.primegov.com` (47 meetings 2026, 18 council, 18 with video) | agenda documents; `videoUrl` → **youtube.com** | agenda_docs | as above |
| livermore | 14 | 1.00 | Granicus `livermore.granicus.com` view 1 = City Council — **agenda-only deployment**: 845 `AgendaViewer.php` links, **0 clip rows** over the full 1.8 MB listing, 0 `MediaPlayer.php` links | agenda + minutes documents only | agenda_docs | as above. `livermoreca.new.swagit.com` answers but its listing is Carmel's (trap 2), and Livermore has no portal to supply a clip id of its own — unproven, not disproven |
| santarosa | 16 | 1.20 | none (Granicus 404 on `santarosa`/`srcity`/`cityofsantarosa`; Legistar API 500) | — | none | — |
| milpitas | 14 | 1.50 | none | — | none | — |
| pleasanton | 14 | 1.50 | PrimeGov host exists (`cityofpleasantonca`) but the archive is **empty: 0 meetings for 2026** | — | none | — |
| healdsburg | 17 | 1.00 | none | — | none | — |
| rohnertpark | 15 | **0.00** | none (Granicus 404 on `rohnertpark`/`rpcity`) | — | none | — |
| losaltos | 16 | 4.67 | none | — | none | — |
| dalycity | 15 | 2.33 | none | — | none | — |
| concord | 8 | 5.00 | none provable — `concordca.new.swagit.com` answers, but its listing is Carmel's (trap 2) and Concord has no portal to supply a clip id of its own; Granicus 404 | — | none | a Swagit route may exist and cannot be confirmed from outside. ~1 h to settle if Puck can name Concord's own player URL |
| vallejo | 14 | 6.50 | none | — | none | — |

> **Built 2026-09-24** — recommendation 1 below is done: `pipeline/swagit.py`,
> `episodes.method='swagit_transcript'`, **13** of the 15 meetings registered
> (the two closed-session specials publish a 2-minute clip against a 42-min and
> a 1.78-h video and stay on the YouTube queue). Candidate recall against the
> caption route **0.914** on the aligned clock, 18 of 18 known episodes
> recovered: `plan/SWAGIT-RECALL.md`, `plan/T2-PLAN.md` §3i. Recommendations
> 2–5 are untouched.

## Recommendation

1. **Build the Swagit route for Petaluma** (~½ day, $0 of API spend). It takes
   15 meetings out of the YouTube queue, and Petaluma is the second-highest-yield
   body in the study (8.0 episodes per captioned meeting), so those 15 meetings
   are worth an estimated ~120 located episodes. It also costs nothing to
   classify beyond what the YouTube route would have cost.
2. **Do not build an agenda-document route.** Six bodies (111 meetings) publish
   agendas with no timestamps. Locating an episode needs times; these do not
   have them, and inventing them from item order would be an instrument we could
   not report.
3. **The remaining 129 meetings have no alternative and must come through
   YouTube** at one an hour, in yield order.
4. **Two bodies are unproven rather than absent**: `livermoreca` and
   `concordca` both answer on Swagit, but neither has a portal that can hand us a
   clip id we know is theirs, and the shared `/views/1` listing cannot settle it.
   If Puck can paste one Livermore or Concord player URL, the check is a minute's
   work — Concord is rank 7 (yield 5.00, 8 meetings).
5. Worth asking Puck, not deciding here: Rohnert Park is measured at **0.00
   episodes over 5 captioned meetings** and has no portal. Spending 15 requests
   on it is a `DECISIONS.md`-level question about what is in the sample, not a
   worker's. The ordering change of 2026-09-24 demotes it to last; it drops
   nothing.

## Reproducing

```bash
.venv/bin/python scripts/portal_probe.py                      # all 16, all families
.venv/bin/python scripts/portal_probe.py --families swagit    # one family
.venv/bin/python scripts/portal_probe.py --bodies petaluma
```

Rate: one request every 1.5 s, DNS-filtered first so a nonexistent host costs no
request. Never asks a YouTube host — the script raises if a URL resolves to one.
