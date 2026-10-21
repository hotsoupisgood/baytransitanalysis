# Bay Area council / transit-board meeting video: where it lives, how to fetch it

Probed 2026-09-20 (metadata only). "Granicus" rows were confirmed by fetching the
archive page; YouTube rows come from yt-dlp search results (official channel,
recent full-length meeting uploads seen). Legistar = agenda/minutes system,
often linked to the Granicus video per agenda item (useful for segmentation).

## Download method by platform

| Platform | How | Status in pipeline |
|---|---|---|
| YouTube channel | `plan/channels.tsv` row, existing downloader | works today |
| Granicus | listing `ViewPublisher.php?view_id=N` -> `MediaPlayer.php?view_id=N&clip_id=<id>` (HLS master, sometimes a progressive MP4, `<track kind="captions" src="/videos/<id>/captions.vtt">`, agenda index points with jump times) -> `ffmpeg -ss -t -vn -c:a copy` audio slices of the HLS master, one per episode | **built 2026-09-21: `pipeline/granicus.py`** (both listing templates, player page, captions, index points, polite >= 5 s), used by `discover.py --meetings`, `captions.py`, `episodes.py`; unit-tested on fixtures from the T0 Oakland / AC Transit pages. The stream host (`archive-stream.granicus.com`, Wowza) returns 403 unless the User-Agent starts with `Mozilla/5.0`. Correction 2026-09-20: the Solana Beach Granicus archive (`solanabeach` view_id=5) only covers 2007–2015; current meetings are on YouTube "City of Solana Beach Official" `UCw7kPQ6zKXxdhkzzCP0t6uQ` (2–6 h, auto captions only, no chapters or description timestamps) |
| Legistar | agenda items + timestamps; video usually Granicus or YouTube | metadata only, for segmentation |
| SFGovTV (SF) | Granicus underneath (`sanfrancisco.granicus.com`, several view_ids) | Granicus adapter |
| Zoom recordings | yt-dlp `zoom` extractor when the share link is public | untested |

## Cities

| Body | Video | Agenda | Notes |
|---|---|---|---|
| San Francisco Board of Supervisors | Granicus `sanfrancisco` view_id=2 (517 clips, "SFGTV Board of Supervisors Archives") | Legistar `sanfrancisco` | SFMTA board is also on SFGovTV (other view_id, not yet located) |
| Oakland City Council | Granicus `oakland` view_id=2 (1,956 clips) — **pilot source** (`plan/channels.tsv`); regular meetings = "Concurrent Meeting of the Oakland Redevelopment Successor Agency and the City Council on YYYY-MM-DD" (1st/3rd Tuesday, August recess), ~2.6–4.7 h; captions.vtt served for all 3 pilot meetings (4,137–7,474 cues), 7–9 agenda index points each, which `classify.py` maps to `episodes.agenda_item` | Legistar `oakland` | duplicate "15h 00m" placeholder clips exist for some dates — `discover.select_regular` drops rows >= 12 h |
| San José City Council | YouTube "City of San Jose, CA" `UCeDiMzJEUbPgaruDcXnD4Cg` **`/streams`** (meetings 1.5–9 h; `/videos` holds boards, commissions and closed-session stubs — T0) — **pilot source**; regular = "MON D, YYYY \|  City Council" (a Spanish simulcast and a duplicate "... \| City Council Meeting" upload exist per date) | Legistar `sanjose` | |
| Berkeley | Granicus `berkeley` view_id=2 (388 clips, ZAB; council on another view_id) | Legistar `berkeley` | YouTube "City of Berkley" hits are Berkley, Michigan — ignore |
| Palo Alto City Council | YouTube "City of Palo Alto" `UC-j1sS-c6up6UvSzDNOHhHw` (4–6 h) | | |
| Mountain View | Granicus `mountainview` view_id=2 (549 clips) | Legistar `mountainview` | |
| Sunnyvale City Council | YouTube "Sunnyvale City Clerk" `UCvTBGgHGeTlrJaabIsRZvVw` | Legistar `sunnyvale` | |
| Cupertino City Council | YouTube "City of Cupertino" `UChTGNMh9Ti6cCi6gMCxhBTg` (multi-part uploads) | Legistar `cupertino` | |
| San Mateo City Council | YouTube "CityOfSanMateo" `UCtQV2ZVAgoV6smi6kWfQU8A` | Legistar `sanmateo` | |
| Redwood City | not found on YouTube search; Legistar `redwoodcity` exists — video likely Granicus/other, check Legistar links | Legistar `redwoodcity` | |
| Fremont | Granicus `fremont` view_id=1 (31 clips — thin; check other view_ids) | Legistar `fremont` | |
| Santa Clara | Granicus `santaclara` view_id=1 (587 clips) | Legistar `santaclara` | |
| Alameda | Granicus `alameda` view_id=2 (218 clips) | Legistar `alameda` | |
| Richmond | Granicus `richmond` view_id=2 (1,192 clips, "Council Meetings") | Legistar `richmond` | |
| Hayward | Granicus `hayward` view_id=1 (1,157 clips) | Legistar `hayward` | |
| San Leandro | Granicus `sanleandro` view_id=2 (1,499 clips) | Legistar `sanleandro` | |
| Walnut Creek | Granicus `walnutcreek` view_id=2 (176 clips) | | |
| Daly City | YouTube "City of Daly City" `UC6wJ6DZOo9J1ds3k1yBbZtg` | | |
| Concord | YouTube "City of Concord" `UCFQKREXWugU-5Ev5aUj6a7g` (two channels exist; verify) | | |
| San Rafael | YouTube "City of San Rafael" `UCaQDlHi5fD6sByZUw5FOg9A` — full meetings on `/streams` only, see North Bay section | | |
| Milpitas | YouTube "City of Milpitas" `UCTS_MyZvQc_c3T4-vSXKM8Q` | | |
| Sausalito | Granicus `sausalito` view_id=2 (687 clips) | | |
| Marin County | Granicus `marin` view_id=3 (329 clips; Planning Commission view — BoS on another view_id) | | |
| Napa County | Granicus `napa` view_id=2 (852 clips, Board of Supervisors) | | |
| Los Altos | YouTube "City of Los Altos" `UCwu5vgJsrSVhGX1p8aKvjjw` | | surfaced by search |
| Pleasanton, Livermore, Vallejo | not on Granicus under the obvious subdomain; not searched on YouTube yet | | Santa Rosa moved to the North Bay section |

## North Bay (SMART corridor)

Probed 2026-09-20 (metadata only, yt-dlp `--flat-playlist` + `ytsearch`). Key
finding: several cities put full council meetings on the channel's **`/streams`
tab (live-stream archive), not `/videos`** — `/videos` shows only commissions,
closed-session stubs and clips. `plan/channels.tsv` crawls must use the
`/streams` URL for those. All six meetings sampled: YouTube auto captions only
(no manual subs), no chapters, description is boilerplate (address only) — no
agenda text, so the structural cue in `segment.py --chapters` never fires here.

| Body | Video | Full meetings? | Length | Captions / chapters | Notes |
|---|---|---|---|---|---|
| San Rafael City Council | YouTube "City of San Rafael" `UCaQDlHi5fD6sByZUw5FOg9A` **`/streams`** — **pilot source**; regular = "City Council Meeting MM/DD/YYYY at ..." | yes, 2x/month, 16 in 2026 | 1.5–6.5 h | auto / no | `/videos` has only 30-s "Closed Session" stubs + commissions. Council member gives a SMART board report in "reports" (~2 min) |
| Petaluma City Council | YouTube "City of Petaluma" `UCW4N-TOX2F_U08sNVEc4Tag` `/videos` | yes, 2x/month | 3–7 h | auto / no | also uploads a 42-min "Petaluma North SMART Station Specific Plan pre-submittal meeting" (`Tv99yW5sTa4`) |
| Santa Rosa City Council | YouTube "City of Santa Rosa, California" `UChF7Zl4VNfZFm-IUdtX6B_A` **`/streams`** | yes | 1.3–10 h | auto / no | two SMART stations; TOC / CityBus items reference SMART repeatedly |
| Healdsburg City Council | YouTube "City of Healdsburg" `UCh-6ckyCBDkEsyWyDmh-5_g` **`/streams`** | yes, 2x/month | 1.3–4.5 h | auto / no | mirror channel "Go Healdsburg" `UCprYLTMdh0pcPNVIH5coLnA` (unofficial, same files). ASR spells the town "Hillsburg"/"Heelsburg" |
| Rohnert Park City Council | YouTube "City of Rohnert Park" `UClhzWns7HpJFj592c5NqZwg` **`/streams`** | yes, 2x/month | 0.5–9.5 h | not checked | |
| Cotati City Council | YouTube "City of Cotati" `UC23Rt-GKyF3PxIbJ7tznUKQ` `/videos` | yes but sparse in 2026 (Jan 27, Feb 10, Mar 24 seen; nothing after) | 1.5–4 h | not checked | may have moved platform mid-2026 — verify |
| Windsor Town Council | YouTube "Town of Windsor CA" `UCiGiTWmH4Q6ro15Bah2T3ow` | **no** — item clips only (45–75 min, e.g. "June 3, 2026 … Windsor Sales Tax") | — | not checked | full meetings not on YouTube; unofficial "Tom" channel has a few full specials. Check Granicus / town site |
| Novato City Council | YouTube "CityofNovato" `UCnR8gkjIKbw5Nmid2QPadig` | **no** — workshops, presentations, PSAs (incl. "No SMART Train Horn" PSA, "Measure Q" = Novato's own sales tax) | — | — | full meetings: NCTV `UC63-lpR2zNbVFUDgkMrAS0A` has 2017 ones only; an unofficial "Cruz Garcia" channel posted a 3-h 4/28/26 meeting. Official archive is elsewhere (Granicus?) |
| Marin County Board of Supervisors | not on YouTube. "MarinGChannel" `UCAKBLAHslCeHRgyb1IVrcjw` = county TV (commissions, webinars; no BoS) | no | — | — | Granicus `marin` (see Cities table; BoS view_id still to locate) |
| Sonoma County Board of Supervisors | not on YouTube. "County of Sonoma" `UC9BXDQlgknPh6zUgGfr8piA` has Community Advisory Council meetings, not BoS | no | — | — | 2021 BoS recordings exist on unofficial "Petaluma Civics" `UCRbgtttDiGPvDDL996ojmEw` (10 h). Official archive is Granicus/Legistar — not probed |

Segmentation test on this corridor (`plan/keywords_smart.txt`, six 2026
meetings): SMART appears as **1–5 min liaison reports and asides**, not as a
20–40 min agenda item like LOSSAN in Solana Beach. Threshold 1.5 found one
2-min fragment; **0.8** finds all 7 SMART-related passages ≥ 3 min with no
false positives in the three SMART-free meetings, and 0.5 still produced none.
Bare "smart" is reliable (41/41 hits = the agency); "Measure Q/I" were 4/4 city
sales taxes and are disabled in the keyword file. Full details in the 2026-09-20
report.

## Transit agencies

| Body | Video | Notes |
|---|---|---|
| BART Board | YouTube "BART" `UCXJOE7qbFm8mr_6ZywmzLkQ` **`/streams`** (regular meetings 2–6 h, 2x/month; `/videos` is marketing clips — T0) — **pilot source**; regular = "Board of Directors Regular Meeting, Month D, YYYY" | works (T0, T1) |
| VTA Board | YouTube "Valley Transportation Authority" `UCxjccaB5ITaq5EhsPRby9xg` | works today |
| AC Transit Board | Granicus `actransit` view_id=2 (42 clips) | adapter |
| SamTrans Board | Granicus `samtrans` view_id=1 (257 clips) | adapter; Caltrain (JPB) meetings are likely in the same archive — verify |
| MTC / ABAG | Granicus `mtc` view_id=1 (146 clips) | adapter |
| SFMTA Board | SFGovTV (Granicus `sanfrancisco`, view_id not yet located) | not on an official YouTube channel |
| Caltrain Board | no official YouTube channel found by search; check `samtrans` Granicus and caltrain.com | |
| California High-Speed Rail Authority | YouTube `UCm37ESFs9ZKyPK4ge4LW6xg` | works today; statewide, not Bay-specific |

## How a body becomes a source (built 2026-09-21, Phase T1)

1. One row in `plan/channels.tsv`: `slug<TAB>url<TAB>meeting<TAB><regular-title regex>`; the URL is the
   `/streams` tab for YouTube bodies or the Granicus `ViewPublisher.php?view_id=N` listing.
2. `pipeline/discover.py --meetings [--recent N] [--bodies a,b]` — metadata only; writes a `sources` row and
   kind=meeting rows (in_frame = 0). Then `captions.py` → `classify.py run` → `episodes.py` →
   `gpu_batch.py propose --speaker <slugs>`. Each stage is idempotent and resumable.
3. Pilot 2026-09-21 (`plan/T0-RESULTS.md` for T0; START_HERE.md for the T1 numbers): San José, San Rafael,
   BART, Oakland — 12 meetings, 44.4 meeting-h, 63 located episodes, batch b14.

Earlier notes: segmentation (keyword windows -> episodes) designed 2026-09-19 — `pipeline/segment.py` +
`plan/keywords_coaster.txt`; proven on six Solana Beach meetings (threshold 1.5 hits/100 words, 60-s
windows); the two-stage version (0.5 threshold + classifier) is what runs now. Chapter cross-check
(`--chapters`) is implemented but none of the uploads seen so far carry chapters.

## T2 listing pass, 2026-09-22 (metadata only; `output/t2/discovery/`, `output/t2/probe/`)

One `--list-only` pass over every located body (YouTube flat-playlist 200 rows, paced 45–180 s;
Granicus listings ≥ 5 s apart), then the regexes in `plan/channels.tsv` were written from the
titles and `--from-listings` inserted without a second round of requests. Corrections to the
tables above:

- **YouTube `/videos` tabs hold no council meetings in the window for Palo Alto, San Mateo, Daly
  City, Concord, Milpitas, Los Altos** — the T0 pattern again; all six have full meetings on
  **`/streams`** (`channels.tsv` updated). VTA's `/videos` last has a board regular meeting in Feb
  2025 and Petaluma's `/videos` stops at the April 20, 2026 council meeting — both re-listed on
  `/streams` (see the T2 plan for the outcome).
- **Granicus listing templates.** Seven of twelve archives use templates the T1 parser did not
  read (`listingRow` rows with the player link in an `onClick`: SamTrans, Hayward, Fremont, Napa;
  `listLine` rows with a blank name cell: SF BoS; bare `<tr>`: Alameda, Walnut Creek).
  `pipeline/granicus.parse_listing` is now template-agnostic; fixtures under
  `analysis/fixtures/granicus/*_excerpt.html`.
- **Dead or wrong-view archives (0 regular meetings in the window):** SF BoS `sanfrancisco`
  view_id=2 stops 2024-12-11 (517 clips) — current SFGovTV video is not on this archive;
  Walnut Creek view_id=2 stops 2009; Alameda view_id=2 is the Transportation Commission; Mountain
  View view_id=2 is the Environmental Planning Commission. All four join the unlocated list.
- **Fremont** view_id=1 is a live council archive after all (31 clips, all 2026, titles
  `CCMeeting9/15/2026` / `Council Meeting`), not "thin".
- Sausalito and SamTrans listings carry no duration; the caption stage fills it.
- Cupertino uploads two parts per meeting; both parts are kept as rows of one `meeting_id`.

Unlocated-body probe (≤ 30 min, `scripts/probe_sources.py`): Granicus subdomains for Redwood
City, Pleasanton and Vallejo exist but `ViewPublisher.php` → NotFound (the newer Granicus
"core" platform; no public archive at that path); Livermore's Granicus lists upcoming events
only in views 1–6; Windsor, Novato, Sonoma County, Caltrain have no Granicus host (Caltrain's
JPB meetings are not in the SamTrans archive: 143 SamTrans board + committee clips only);
Berkeley views 1–12: Rent Board (view 9), Housing Authority, BOLT — no council; Marin views
1–12: Planning Commission, telecom agency — no BoS; SF views 1–40: no SFMTA board view. Legistar:
only `sonoma-county.legistar.com` is a real calendar (Board of Supervisors; video links are
not in the HTML); the other `<city>.legistar.com` hosts serve Legistar's generic page.
`actransit.legistar.com` links `actransit.granicus.com` + Zoom (InSite video). YouTube search
results are in `output/t2/probe/youtube.json` and the T2 plan.

## T2 locate probe, 2026-09-22 (`output/t2/probe_index/`, `scripts/probe_granicus_index.py`)

One player page per body for the six caption-less archives plus Napa, ≥ 5 s apart, saved
verbatim. Three separate questions: does the page carry **agenda index points** (item title +
jump time — the free locate route of PROPOSAL §2.2), does it carry **any caption endpoint** in
any language, and is there **any media URL at all** (an ASR fallback needs audio).

| body | index points | caption endpoint | language | media | verdict |
|---|---|---|---|---|---|
| santaclara | **yes**, median 21/meeting | **yes, full English track** (4,461 cues on clip 2592) | en | HLS + MP4 | captions *and* index — was wrongly written off |
| hayward | **yes**, median 19 | **yes, English**, but only on the 3 newest meetings | en | HLS | index for the rest |
| napa | **yes**, median 18 | yes — **one track only, and it is Spanish** | es (label says `en`) | HLS + MP4 | index; no English track exists |
| sausalito | **yes**, median 11 | `<track>` exists, file is the **empty 8-byte `WEBVTT`** | — | HLS + MP4 | index only |
| samtrans | yes on 2 of 12 | same empty 8-byte file | — | HLS | index on 2; ASR for 10 |
| richmond | **none** (`JSON.php` → `[[]]`) | none | — | **ASX → MP4** | ASR only |
| sanleandro | **none** | none | — | **ASX → MP4** | ASR only |

Three findings that changed the run:

- **Santa Clara and Hayward were never caption-less.** They serve the track through a Flowplayer
  config object (`kind: "subtitles", lang: "en", label: "English", src: "/videos/<clip>/captions.vtt"`)
  and have **no `<track>` element**, which is all `granicus.parse_player` looked for — so 69
  meetings were recorded `captions='none'` and sent to `no_episodes`. `parse_player` now reads
  both forms (a real `<track>` still wins) and records `captions_lang` / `captions_label`.
  `captions.py --refetch --bodies <b>` re-asks the player pages after a parser fix; re-run
  2026-09-22, **37 of 69 meetings now have an English track** (Santa Clara 34 of 40 = 178 h,
  Hayward 3 of 29). Nothing was deleted: the player JSON is rewritten and only a 0-cue file is
  overwritten.
- **An empty caption endpoint is not a caption track.** Sausalito, SamTrans and AC Transit all
  serve an 8-byte `WEBVTT` file, which on disk is indistinguishable from a track we never asked
  for; `captions_granicus` now re-asks whenever the saved file has no cues.
- **Richmond and San Leandro have audio after all.** Their 2015-era player puts no stream on the
  page; `ASX.php` answers with `rtmp://archive-video.granicus.com/<site>/<file>.mp4`, and the
  same path is served over **HTTPS with byte ranges** (Richmond: 200, 1.79 GB for 7.5 h).
  `granicus.fetch_archive_mp4()` resolves it, which is what made the local ASR fallback possible.

Napa's track is a Spanish interpretation even though the player labels it `en`; there is no
second track on the page. Napa is located by agenda index, not by captions.
