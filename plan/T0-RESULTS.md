# T0 results — Bay Area density + transit (plan/T0-TESTPLAN.md)

Run 2026-09-20 22:07–23:18 PDT plus the test 4 re-run 2026-09-21 05:21–06:11
PDT on the supervisor Mac. Tests 1, 2, 3 (prep), 4, 6 (speaker roles, on the
b13 transcripts, 2026-09-21), 7 and 8 (partial) done.
**Test 5 and the classifier-accuracy scoring are pending** (5: needs a GPU
batch Puck approves; scoring: needs Puck's blind labels on
`output/t0/label_sheet.csv`).
Everything under `output/t0/` is gitignored; the logs cited below are there.
No GPU pod, no RunPod call, no `.env` read into any transcript. Claude spend for
the whole run: **$0.734** of the $3 cap (`output/t0/classify_ledger.json`).

## Sample as run (`output/t0/sample.json`)

Discovery finding first: **none of San José, Sunnyvale or BART has full
meetings on the `/videos` tab.** San José `/videos` holds boards, commissions
and 4-min closed-session stubs; Sunnyvale `/videos` has no council meeting after
Nov 2025; BART `/videos` is marketing clips. All three publish the meetings on
`/streams` (`output/t0/discovery/*_streams.tsv`), like the North Bay cities in
`sources_bayarea.md:52–55`. So **7 of 7 YouTube bodies in this sample are
`/streams` sources**, and `channels.tsv` rows for San José, Sunnyvale and BART
must use the `/streams` URL (PROPOSAL §3c applies to them too).

| # | video_id | Body | Meeting | Listed h | Why this one |
|---|---|---|---|---|---|
| 1 | EqZPNTTUr8A | San José | SEP 15, 2026 City Council | 5.86 | most recent regular; Spanish simulcast F5KFnbnJFa4 skipped |
| 2 | l-xT4ramOmU | San José | SEP 1, 2026 City Council | 1.53 | next regular; duplicate upload DDT7jYymXSU (5541 s) skipped |
| 3 | mpbBPvyGNMo | Sunnyvale | City Council Meeting Sept. 15, 2026 | 3.80 | most recent regular |
| 4 | Chj4lSP9je4 | Sunnyvale | City Council Meeting Aug. 25, 2026 | 6.03 | next regular |
| 5 | iyh8o9tJgVI | San Rafael | City Council 09/08/2026 | 2.86 | most recent regular; the SMART-pilot meeting (acceptance case) |
| 6 | F0hZlLcNR8o | San Rafael | City Council 08/17/2026 | 1.43 | next regular |
| 7 | WIGr3Mc7kvw | Santa Rosa | City Council 9-15-26 | 1.42 | most recent regular in English (Spanish simulcast tjSVaXJQtGw separate) |
| 8 | 9FM5406YagE | BART | Board Regular Meeting Sept 10, 2026 | 4.24 | most recent regular |
| 9 | qIlAln3uR7U | BART | Board Regular Meeting Aug 13, 2026 | 5.96 | next regular |

Total 33.1 meeting-hours, mean **3.68 h** (PROPOSAL assumed 3.8 h).

## Test 1 — discovery + captions: PASS (9/9 captioned, 0 bot-checks, 0 403s)

Source: `output/t0/test1_table.json` (built by `output/t0/test1_table.py` from
the nine `info.json` files and `fetch.log`s); wall clock in
`output/t0/fetch_captions.log` line 1 and last line.

- 9/9 fetched with `--skip-download --write-auto-subs --write-subs --sub-langs
  "en,en-orig" --write-info-json --sleep-requests 2`; one at a time, 45–180 s
  gaps; **18.5 min wall** for 9 meetings (22:20:06 → 22:38:40), i.e. about
  2 min per meeting including the gap. No "Sign in to confirm", no 429, no 403
  on any `/streams` VOD.
- Captions: **9/9 auto English**. San José additionally has an uploaded English
  track under the suffixed id `en-uYU-mmqFLq8` (`test1_table.json`
  `manual_subs`) which `--sub-langs en,en-orig` does not match; it was not
  fetched (`en.*` is the pattern that caused the Tucker 429, `PLAN.md:180–183`).
  San Rafael lists `live_chat` and BART qIlAln3uR7U an `und` track — neither is
  captions. Both `.en.vtt` and `.en-orig.vtt` are the same auto track (identical
  sizes), so `en-orig` can be dropped from the request.
- Chapters: **0/9**; description timestamps: **0/9**. The structural cue in
  `segment.py --chapters` never fires on this sample either.
- Discovery listings (`--flat-playlist --playlist-end 40`, 8 channel tabs) also
  passed without a block signal.

## Test 2 — episode share: measured

Files: `output/t0/candidates.csv` (198 rows), `output/t0/episodes.json` (45),
`output/t0/classify_summary.json`, `output/t0/classify/<video_id>.jsonl`, logs
`output/t0/classify.log` and `classify2.log` (table at `classify2.log:109–120`).

**Stage 1.** `plan/keywords_density.txt` (173 terms, 17 conditional) +
`plan/keywords_transit.txt` (161 terms, 15 conditional, marked `?`), families tagged. `segment.py` gained
`@family` / `?` parsing, `family_hits`, `window_texts`, `candidate_windows`
(`pipeline/segment.py`), all existing tests untouched. Threshold 0.5 on the
union density, 60-s windows, 3-window smoothing; maximal runs of marked windows
cut into candidates of ≤ 240 s (so a 30-min agenda item is ~8 classifier calls,
not one 10 k-token call); each candidate carries its caption text ± 30 s.
Ambiguous terms: `train`, `smart`, `freight`, `bus`, `bike`, `parking`,
`traffic`, `zoning`, `units`, `general plan`, `land use`, `ceqa` … are
conditional — they count only in a window that already has an unconditional
hit from the same family (`keywords_density.txt` header; `segment.py hits()`).

**Stage 2.** `scripts/classify_windows.py`: `claude-sonnet-4-6` (SCORING_MODEL
via `pipeline/state.setting`), `thinking={"type":"disabled"}`, no sampling
params, `output_config.format` JSON schema `{label, rationale,
contains_instructions}`, instructions + codebook (5,578 chars, 1,531 tokens) as
a cached system block, `prompt_version t0-a4a8e55760` on every row. 198 calls,
198 parsed, 0 `contains_instructions`, mean latency 2.27 s.

| video_id | body | h | cand | dens | tran | both | ment | no | eps | about min | min/h | dens/h | tran/h | $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EqZPNTTUr8A | sanjose | 5.86 | 45 | 21 | 0 | 0 | 19 | 5 | 5 | 71 | 12.1 | 12.1 | 0.0 | 0.163 |
| l-xT4ramOmU | sanjose | 1.53 | 15 | 11 | 0 | 0 | 3 | 1 | 2 | 45 | 29.3 | 29.3 | 0.0 | 0.058 |
| mpbBPvyGNMo | sunnyvale | 3.80 | 18 | 7 | 5 | 0 | 4 | 2 | 5 | 42 | 11.1 | 7.1 | 4.0 | 0.064 |
| Chj4lSP9je4 | sunnyvale | 6.03 | 23 | 1 | 16 | 3 | 2 | 1 | 5 | 75 | 12.4 | 10.9 | 11.8 | 0.091 |
| iyh8o9tJgVI | sanrafael | 2.86 | 8 | 3 | 1 | 0 | 4 | 0 | 3 | 14 | 4.9 | 3.9 | 1.1 | 0.036 |
| F0hZlLcNR8o | sanrafael | 1.43 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.003 |
| WIGr3Mc7kvw | santarosa | 1.42 | 9 | 0 | 3 | 0 | 6 | 0 | 1 | 12 | 8.5 | 0.0 | 8.5 | 0.032 |
| 9FM5406YagE | bart | 4.24 | 28 | 0 | 25 | 0 | 3 | 0 | 9 | 82 | 19.4 | 0.0 | 19.4 | 0.099 |
| qIlAln3uR7U | bart | 5.96 | 51 | 0 | 44 | 0 | 6 | 1 | 15 | 160 | 26.9 | 0.0 | 26.9 | 0.187 |
| **Total** | | **33.12** | **198** | 43 | 94 | 3 | 48 | 10 | **45** | **501** | **15.1** | 6.6 | 10.4 | **0.734** |

(`about_both` minutes are counted in both families; `classify_summary.json`
per video_id, keys `about_minutes`, `density_min_per_hour`, `transit_min_per_hour`.)

**Measured episode share** (replaces the 2–5 % ASSUMPTION in PROPOSAL §2.1):

| Group | Meetings | Meeting-h | about_* min | **Share** | density | transit | cand/h | $/meeting | $/meeting-h |
|---|---|---|---|---|---|---|---|---|---|
| All 9 | 9 | 33.12 | 501 | **25.2 %** | 11.1 % | 17.3 % | 6.0 | $0.082 | $0.0222 |
| Councils (7) | 7 | 22.93 | 259 | **18.8 %** | 16.0 % | 7.3 % | 5.2 | $0.064 | $0.0195 |
| BART (2) | 2 | 10.20 | 242 | **39.6 %** | 0.0 % | 39.6 % | 7.7 | $0.143 | $0.0281 |

Episodes: 45 (30 about_transit, 12 about_density, 3 about_both), mean 11.1 min,
median 8 min, longest 60 min (a BART item). Padding at 30 s each side is **9 %**
of episode time, not the 25 % assumed (episodes are longer than the 4-min
pilot shape). Tokens per window: 865 uncached in + 1,516 cached + 40 out.

Caveats that matter for the share: (a) the range across councils is 0–29
min/h — one San José meeting was 50 % affordable-housing finance, one San Rafael
meeting had nothing; nine meetings is not a rate estimate, it is a first
measurement; (b) a transit board is about transit by construction, so the 40 %
BART figure says the per-body cap in §2.3 is needed for boards, not that the
council rate is 25 %; (c) the classifier's precision is not yet measured
(test 6) — every about_* minute here is model-labelled.

**SMART acceptance case.** San Rafael 09/08 (iyh8o9tJgVI): stage 1 caught the
liaison report as candidate `_c002` (29:00–33:00, top keywords housing / smart /
units) and the classifier labelled it **mention_only** ("liaison report listing
many meetings; SMART ridership … mentioned briefly", `classify/iyh8o9tJgVI.jsonl`).
So it is caught and counted, not an episode. Whether a 2-min liaison report
should be about_transit is a codebook decision for Puck; the window is in the
label sheet.

## Test 3 — classifier accuracy PREP: done, not scored

- `output/t0/label_sheet.csv`: 100 windows, all 9 meetings, columns
  `id, video_id, start, end, text (≤ 1,200 chars), your_label` (blank).
  Stratified by meeting × (about / not-about); **50 are model-labelled
  no / mention_only** (≥ 25 required). Model labels are only in
  `output/t0/label_key.csv` (22 about_density, 26 about_transit, 2 about_both,
  41 mention_only, 9 no).
- `scripts/score_labels.py sheet key` → per-family precision / recall / F1
  (density = about_density ∪ about_both, transit = about_transit ∪ about_both),
  exact agreement, Cohen's κ, 5×5 confusion. Verified on a synthetic pair in
  `make test` (section 17). On the real sheet it currently reports "0 scored,
  100 unlabelled".

## Test 4 — episode audio: 40/40 fetched, one 403 (run 1), zero in run 2

Two runs. **Run 1** (2026-09-20 22:54–23:18, `output/t0/download_sections.py`,
T0 artefact, not committed; log `output/t0/sections.log`, report
`sections_report.json`): the 40 longest of the 45 about_* episodes (491 core
min, 531 min with 30 s pads; 11 + 8 BART, 5 + 5 + 5 + 2 San José / Sunnyvale,
3 San Rafael, 1 Santa Rosa), one `yt-dlp --download-sections "*<start>-<end>"
-f bestaudio -x --audio-format opus --sleep-requests 2` per episode. It fetched
4 clips of 9FM5406YagE and stopped as designed: the **6th stream open on the
same video got HTTP 403 from googlevideo** (`9FM5406YagE/e6.log`: "Server
returned 403 Forbidden" on the fresh `videoplayback` URL, no "Sign in to
confirm", no 429; `sections.log:10–11`). Not a bot-check — a per-video limit on
repeated stream opens.

**Run 2** (2026-09-21 05:21–06:11, `scripts/t0_fetch_episodes.py`, committed;
log `output/t0/fetch_episodes.log`, state `fetch_rows.json` /
`fetch_requests.json`, report `fetch_report.json`), the fix Puck approved: a
meeting with ≥ 3 planned episodes is fetched **whole, once** (`-f bestaudio -x
--audio-format opus`, → `output/t0/<video_id>/full.opus`, kept as test-5
input) and the episodes are cut locally with `ffmpeg -ss/-t -c copy`; meetings
with 1–2 episodes still use `--download-sections`. Same pacing (sequential,
`--sleep-requests 2`, 45–180 s gaps, yt-dlp retries capped at 2), stop on
bot-check / 429 / second 403. The 4 run-1 clips were kept.

| | requests | ok | 403 | bytes on wire | request wall | rate | audio obtained |
|---|---|---|---|---|---|---|---|
| whole meeting (6 videos, `fetch_episodes.log:10,17,24,29,36,55`) | 6 | 6 | 0 | **1,297 MB** (216 MB/request, **45.1 MB per meeting-h**) | **220 s** (13.5–50.4 s each) | 5.7–18.4 MiB/s, **469× real time** | 28.74 meeting-h → 33 clips, 7.02 h |
| `--download-sections` (runs 1 + 2, `sections.log`, `fetch_episodes.log:49,51,53`) | 8 | 7 | **1** | 91 MB (11.4 MB/request) | 3,512 s | 25.4 KB/s, **1.89× real time** | 7 clips, 1.84 h |

Totals (`fetch_report.json` `total`): **40/40 clips, 8.86 audio-h, 437 MB
on disk (49.3 MB per audio-h, 109 kbit/s opus, format 251 copied not
re-encoded)**; run 2 wall **2,989 s = 49.8 min** for 36 clips = 220 s of
whole-meeting pulls + 1,855 s of the 3 section pulls + 914 s of gaps (run 1:
2,140 s for 4 clips). 1.30 GB of `full.opus` is on disk as well
(`output/t0/*/full.opus`, 6 meetings). Every copy-cut landed within 20 ms of
the requested length (`CUT` lines, `dur=`); no re-encode was needed.

**Bytes per request** is the trade: a whole meeting is 216 MB per request, a
section 11 MB — 19× more bytes for the whole pull. But **wall time per request
is the constraint, not bytes**: ffmpeg's sequential read of the stream URL is
paced by YouTube at ≈ 25 KB/s, while yt-dlp's native downloader (10 MB ranged
chunks) runs at 5–18 MiB/s. A 4.2 h meeting arrived in 25 s whole; its 4-min
episode alone would have taken ≈ 120 s as a section. Break-even: whole is
faster for any episode longer than ≈ 1 min of a 3.8 h meeting. And the
section pattern (many stream opens per video) is what produced the 403; the
whole pull is the one-open-per-video pattern every yt-dlp user makes.

**Section clips are not what was asked for.** All 7 `--download-sections`
clips are exactly **10.0 s longer** than the range (`fetch_report.json`
`duration_s` vs `end_s − start_s`); cross-correlating `9FM5406YagE/e2.opus`
against `full.opus` puts clip t = 0 at 1280.02 s for a requested 1290 s start,
and clip t = 600 at 1880.02 s, so the 610 s clip ends at the requested 1890 s:
the extra 10 s is a lead-in (ffmpeg seeks to the webm cluster before `-ss`,
correlation 1.000 both ends). So section pads were
really 40 s before / 30 s after; the local copy-cuts are exact. The edge check
below uses the real boundaries (`actual_start_s`).

**Edge check** (`fetch_report.json` `clips[].edges`, caption cues from the
auto track): **38/40 clips start mid-cue and 38/40 end mid-cue** — but auto
captions are nearly gapless (BART 9FM5406YagE: 3,566 cues, mean 4.2 s, median
2.4 s, over 4.24 h), so almost any 30 s pad lands inside a cue; the 2 + 2
exceptions are t = 0 and cut points that fall on a cue edge. The useful numbers:
the nearest cue edge is **median 0.60 s, p90 1.28 s** from the boundary, 66 of
80 boundaries are within 1 s of a cue edge, 3 boundaries are > 5 s from any cue
(35–111 s: cuts inside recess / silence). A `>>` speaker change sits within 5 s
of 7 starts and 6 ends. **Pad sizing (§6):** 30 s is enough; the mid-cue
problem is solved for free by **snapping the padded boundary to the nearest cue
edge** at cut time (≤ 1.3 s of extra audio on 90 % of clips) now that cutting is
local and exact — not by a bigger pad.

**Pass line** ("all fetched, no 403s"): all 40 fetched; one 403 in run 1, none
in run 2 (10 YouTube requests, 6 of them whole meetings, all rc 0). **Passes
with the whole-meeting strategy; the per-episode section strategy fails it.**

**Recommendation for T1:** fetch every meeting **whole, once**, regardless of
its episode count, and cut episodes locally (`ffmpeg -ss/-t -c copy`, snap to
cue edges). Keep `--download-sections` only as a fallback when a meeting is
> ~10 h or bandwidth is capped; it never wins on wall time for an episode of
≥ 1 min and it is the pattern that got blocked. Consequences: (a) one yt-dlp
request per meeting can fetch info.json + captions + audio together, so the
locate pass and the audio pass collapse into one request per meeting
(11 % of the bytes are wasted on meetings with no episode — 1 of 9 here);
(b) `full.opus` retention (45 MB per meeting-h) is a disk decision for Puck:
138 GB for the 12-month window if kept, 35 GB of clips if not; (c)
`--download-sections` clips, if ever used, need `actual_start = end −
duration`, not the requested start.

## Test 6 — speaker-role labelling (dais vs public): PASS, 93.6 % of speech time

(Puck numbered this test 6 on 2026-09-21; the test plan's original test 6,
classifier-accuracy scoring, is unchanged and still pending his blind labels —
see "Pending" below.) Run 2026-09-21 on the 40 diarized b13 clips
(`output/batches/b13/out/<clip>.json`, 7.34 h of labelled speech, 273
clusters). Code: `scripts/role_label.py` (text + timing only, no model),
cues in `plan/role_cues.txt`, optional embedding step `scripts/role_enrol.py`
(venv-embed, CPU). Output: `output/t0/roles/<clip>.json` (`{label: {role,
method, confidence, talk_s, evidence}}` + per-episode summary),
`output/t0/roles/summary.json`, enrolment variant under `output/t0/roles/enrol/`.
`make test` section 18 (13 checks, synthetic transcripts). No GPU, no RunPod,
no YouTube request, nothing under `output/` overwritten.

**Method** (PROPOSAL §3g, cues in order): (1) phrase cues per cluster —
`floor` / `timer` / `vote` / `staff` / `intro` (job titles, only in
self-introduction form) / `address` → dais, `public` ("my name is", "I live
on", "I urge you") → public; (2) structure — an `opener` ("we'll now take
public comment", "I have two speaker cards") starts a region that the next
`closer` ("close public comment", "back to the council") ends; a `timer` cue
("that's your time", "you have three minutes") with no region open means the
clip began inside public comment; a cluster with ≥ 70 % of its talk inside a
region, no floor cue and ≤ 3.5 min per region is public, one that spoke
before the region or after the closer is dais, and a clip with no region at
all gives cue-less clusters a dais prior; (3) recurrence — ≥ 3 turns over
≥ 5 min → dais (halved inside public comment: serial commenters). Points
are summed, `|margin| ≥ 1` decides, otherwise `unknown`.

**Clusters per episode:** min 1, median 6, max 14 (mean 6.8); 19 of 40 clips
have a public-comment region. **Talk-time labelled** (text cues only):

| | h | dais | public | unknown |
|---|---|---|---|---|
| All 40 clips | 7.34 | 67.0 % | 26.9 % | 6.0 % |
| BART (19) | 3.42 | 59.1 % | 31.2 % | 9.8 % |
| San José (7) | 1.76 | 74.5 % | 25.5 % | 0.0 % |
| Sunnyvale (10) | 1.75 | 80.2 % | 14.1 % | 5.7 % |
| San Rafael (3) | 0.25 | 35.7 % | 64.3 % | 0.0 % |
| Santa Rosa (1) | 0.15 | 60.1 % | 33.7 % | 6.3 % |

**Hand check** (`output/t0/roles/truth_10.json`, written from the transcript
text before the labeller ran on those clips; sha1 `4cf188bc…`): 10 clips —
2 per body, 3 for BART, 1 for Santa Rosa (its only clip) — 86 clusters,
8,018 s of speech, 6 clips with a public-comment stretch. Roles: dais =
members, staff, clerk, mayor, attorney, interpreter; public = commenters,
applicants; `other` = invited presenters from outside the body (Caltrain
executives, the BART bond-oversight chair: 3 clusters, 781 s, excluded from
the denominator — the labeller called all three dais).

| run | speech-time strict (unknown = wrong) | lenient | public precision | public recall |
|---|---|---|---|---|
| first run, before any tuning | 80.8 % (779 s unknown) | 90.5 % | 0.85 | 0.79 |
| **after tuning (committed rules)** | **93.6 %** (162 s unknown) | 95.8 % | 0.94 | 1.00 (29/29) |
| + embedding enrolment | 95.1 % (54 s unknown) | 95.8 % | 0.94 | 1.00 |

I did tune after the first comparison, on the mechanics rather than the
weights: the bare "public comment" opener re-opened a region after "not
seeing any other public comment" (so director reports got the public bonus);
"next up is" / "followed by" opened regions when a mayor called
councilmembers; "general manager" / "city manager" counted as a staff cue when
a public speaker was *addressing* one; "available for questions" is said by
applicants; opener credit went to anyone who said "public comment period".
Each fix was checked on the four development clips too. Pass line ≥ 85 %:
**passes** (per clip 78–100 %, one outlier at 11 %, below).

**What did the work** (talk time of clusters carrying each evidence type):
structure 3.7 h, dais prior 2.1 h, public phrases 1.9 h, recurrence 1.9 h,
floor 1.4 h, address 1.4 h, staff 1.3 h, vote 1.1 h, intro 0.7 h. By
deciding method: dais by phrase 35 %, dais by prior 14 %, public by structure
14 %, public by phrase 13 %, dais by structure 10 %, recurrence 7 %, unknown
6 %. The public-comment region is the single most useful cue; phrases mostly
identify the chair, the clerk and staff presenters.

**Failure modes seen.** (a) Dais members who talk like residents: a
councilmember saying "I live in … as a resident" (iyh8o9tJgVI_e3, 200 s, the
11 % clip) and one saying "on behalf of my community" during a public-comment
stretch (WIGr3Mc7kvw_e1, 86 s) — both public; these are the only errors
> 10 s. (b) Diarization merges: the clerk and several directors' roll-call
votes share one label (9FM5406YagE_e1), a public speaker and the staff
presenter who followed share one (qIlAln3uR7U_e6 SPEAKER_07); a merged label
gets one role, and slips of 1–3 s (a "Good" or "income" attributed to the
wrong cluster) are everywhere but cost nothing measurable. Over-splitting was
not seen: no dais member was split into two labels in the 10 clips. (c)
Invited presenters (Caltrain briefing, museum, oversight committee) are
neither dais nor public; they come out dais. (d) Region boundaries depend on
the chair's habits — Sunnyvale and San Rafael open and close explicitly, BART
closes with "that was the last public comment", San José often has no explicit
closer, so the region runs to the clip end and the recurrence/after-closer
rules do less there. (e) Applicants get five minutes and are labelled public
by structure, which is what the truth file wants but is a codebook decision.

**Presiding-officer enrolment (step 4): tried, small help.** `role_enrol.py`
embeds every cluster ≥ 10 s once (wespeaker, cached under
`output/t0/roles/embed/`, 5.4 min for 40 clips on CPU) and (i) enrols one
anchor per body from the cluster with the most floor credit — that is the
clerk / board secretary at BART and San José, the mayor at Sunnyvale — and
tags matches at cos ≥ 0.60 (BART 8 across 19 clips, others 1–2); (ii) keeps a
per-body bank of confidently phrase-labelled voices and resolves `unknown`
clusters against it (cos ≥ 0.60, gap ≥ 0.10). Net: 4 labels changed on the 40
clips, all unknown → dais, all correct on reading; unknown share 6.0 → 5.2 %;
truth-set strict accuracy 93.6 → 95.1 %. It never overrides a text label, so
it cannot fix (a). Worth keeping as the last step, not as the method.

**Recommendation for T1.** Ship `role_label.py` as the post-batch step (it
fills `speaker_roles.role` and `role_method`, §5) with the cue file as the
tuning surface; add each new body's opener/closer phrasing from its first
meeting. Keep `unknown` as a real value (6 %) rather than forcing a label.
Add the (a) failure to the QA sheet: a `public` cluster with ≥ 3 turns or
> 3.5 min in a clip with no region should be flagged for a look. Run the
enrolment pass only if the unknown share on a body stays above ~10 % (BART).
Re-check accuracy with a second hand-labelled set that I did not tune on: the
93.6 % above is after tuning and is an upper estimate; the pre-tuning 80.8 %
is the honest lower bound for a new body with unseen phrasing.

## Test 7 — Granicus probes: 1 of 3 archives is alive

Files: `output/t0/granicus/report.json`, `probe.log`, `probe2.log`,
`scan_views.log`, `<site>/listing.html`, `<site>/player_<clip>.html`,
`oakland/captions_7755.vtt`, `oakland/chunklist.m3u8`, `oakland/ffmpeg_slice.log`.

| Site | Archive | Newest clip | Agenda index | CC track | Renditions | Slice |
|---|---|---|---|---|---|---|
| **oakland** view_id=2 | 1,956 clips, 2012 → **Sep 15 2026**, 1,542 with an Agenda link, minutes links | 7755, Concurrent … City Council 2026-09-15, 3 h 42 m | **yes**: 8 index points with `time="<s>"` and item titles on the player page (`player_7755.html`) | **yes**: `<track kind="captions" src="/videos/7755/captions.vtt">` — 366 KB, **6,215 cues over the full 3 h 42 m**, live-captioner style | one variant, 640×426 H.264 + AAC, `BANDWIDTH=1872881`; **no audio-only rendition, no EXT-X-MEDIA**; also a progressive MP4 `archive-video.granicus.com/oakland/….mp4` (1.44 GB, `Accept-Ranges: bytes`) | `ffmpeg -ss 600 -t 300 -vn -c:a copy` from the m3u8: rc 0, **34 s wall**, 300.0 s, **4.99 MB AAC 133 kbit/s**; `-ss` **did seek** — ffmpeg opened `media_0`, `media_1` (probe) then jumped to `media_299`…`media_450` (2-s segments, `ffmpeg_slice.log`); 154 segments at ≈ 224 KB each (HEAD of 60) ≈ **34 MB transferred for 5 min** |
| actransit view_id=2 | 42 clips, **2010–Jan 2011 only** (committee meetings); view_ids 1, 3–8 → 404 (`scan_views.log`) | 135, External Affairs Committee 2011-01-19 | 5 index points | endpoint exists, **empty** (8 bytes) | 320×240, `BANDWIDTH=156907` | 7.5 s wall, 4.98 MB, 27 segments |
| mtc view_id=1 | 146 clips, **2015–May 2016** (+ one 2023 webinar); view_ids 2–8 → 404 | 436, MTC 2016-04-27 | none parsed | player page has a `captionsUrl` slot, no track | **no m3u8 / mp4 on the page** (2016-era player) | not attempted |

Findings:

- **Free locate routes on the live archive (Oakland): both present.** The
  agenda index gives item titles with jump times; the caption track gives the
  whole meeting as VTT, i.e. the same input as a YouTube auto-caption and
  `segment.py` can read it unchanged (`vtt_to_segments`). If the other 12
  Granicus councils in `sources_bayarea.md` look like Oakland, the "$10–30
  ASR-only locate pass" in PROPOSAL §2.2 is **$0**. Coverage is 1 of 1 live
  archive probed; AC Transit and MTC cannot be judged from dead archives.
- **`sources_bayarea.md:87` and `:89` are wrong for the 12-month window**: the
  `actransit` and `mtc` Granicus hosts are legacy archives. Current AC Transit
  and MTC meetings live elsewhere (Legistar/InSite video or YouTube — not
  probed; 2 h of locating work per the §8 item 7 budget).
- **Transfer volume.** HLS is video+audio only: **≈ 6.9 MB/min ≈ 410 MB per
  meeting-hour** on the wire (rendition bit rate ≈ 0.9 Mbit/s actual vs 1.87
  declared); the audio kept on disk is 1.0 MB/min (60 MB/h AAC, ~12 MB/h after
  the 24 kbit/s opus transcode). The progressive MP4 has the same size
  (1.44 GB / 3.7 h = 390 MB/h) and honours byte ranges, so a seek is possible
  there too, but the audio is interleaved with video and cannot be fetched
  alone. With captions locating the episodes, only episode minutes are pulled:
  at 18.8 % share that is ~77 MB per council meeting-hour instead of 410.
- `-ss` seeking works on the HLS master, so section pulls do not download from
  the start (5-min slice = 34 s wall, not 3 h 42 m).

## Test 8 — projection (partial; GPU RTFX still the 25× ASSUMPTION)

Inputs and where they come from:

| Quantity | Value | Source |
|---|---|---|
| Meetings in window | 804 located (696 council + 108 board) | PROPOSAL §2.1 (unchanged) |
| Mean meeting length | 3.68 h measured here; 3.8 h kept for the table | `classify_summary.json` `_total.hours` / 9 |
| Meeting-hours | 3,055 h (2,645 council + 410 board) | 804 × 3.8 |
| Episode share | councils 18.8 %, boards 39.6 % | test 2 table |
| Pad overhead | 9 % | 45 episodes × 60 s / 501 min |
| Candidates | 6.0 / meeting-h (5.2 councils, 7.7 boards) | test 2 |
| Classifier $ | $0.0222 / meeting-h all, $0.0195 councils, $0.0281 boards | test 2 |
| GPU $/audio-h | $0.021 at 25× (b9, b11) | PROPOSAL §2.2; **test 5 pending** |
| YouTube whole-meeting transfer | 469× real time (7.7 s per meeting-h), 45.1 MB per meeting-h on the wire, 1 request per meeting | test 4 run 2, 6 meetings |
| YouTube section transfer (fallback only) | 1.89× real time, 49.3 MB per audio-h on disk, 1 request per episode | test 4, 7 clips |
| Granicus transfer | 410 MB / meeting-h on the wire, captions free | test 7 |

Revised PROPOSAL §2.2 table (804 meetings, 3,055 meeting-h):

| Row | Episode-h | + 9 % pad | GPU-h @25× | GPU $ @ $0.49/h | × 1.5 contingency |
|---|---|---|---|---|---|
| Councils at measured 18.8 % | 497 | 542 | 21.7 | $10.6 | $16 |
| Boards at measured 39.6 % | 162 | 177 | 7.1 | $3.5 | $5 |
| **Both, measured** | **659** | **719** | **28.8** | **$14.1** | **$21** |
| Old 5 % row (for comparison) | 153 | 191 | 7.6 | $4.0 | $6 |
| Old 10 % sensitivity row | 306 | 382 | 15.3 | $8.0 | $12 |
| Whole meetings | 3,055 | — | 122 | $64 | $96 |

At half the RTFX on short clips (the §2.2 unknown, test 5) the measured row is
$42; with the 8 unlocated bodies × 1.23. Still small: **GPU is not the
constraint.**

Other rows that change:

- **Classifier:** 3,055 h × $0.0222 = **$68** single-sample (PROPOSAL §4
  guessed $77 for 990 meetings), **$34** via the Batches API; k = 3 would be
  $204 and is not recommended. Per meeting: $0.082 measured vs $0.077 assumed.
- **Download time, YouTube (whole-meeting strategy, test 4 run 2):** 804
  meetings × 1 request; transfer 3,055 meeting-h × 7.7 s ≈ 6.5 h, plus 804
  gaps × 112.5 s (mean of 45–180) ≈ 25 h → **≈ 32 h ≈ 1.3 days** at
  concurrency 1 — the gaps are 80 % of it, the bytes are not the constraint.
  Wire volume 3,055 h × 45.1 MB ≈ **138 GB**; disk 719 padded episode-h ×
  49.3 MB ≈ **35 GB** of clips, plus 138 GB if `full.opus` is retained. If the
  same request also writes info.json + captions (one yt-dlp call per meeting),
  the separate caption pass (600 fetches ≈ **20 h**, §3d "≈ 25 h") disappears;
  otherwise add it. For comparison the per-episode section strategy would be
  719 h / 1.89 ≈ 380 h of transfer + ≈ 4,000 requests × 112.5 s ≈ 125 h of gaps
  ≈ **21 days**, with a 403 after ~5 opens per video — not viable. PROPOSAL
  §2.2 said "~3 days".
- **Granicus:** captions are free where served, so no locate pass; the audio
  for episodes is 410 MB/h × 18.8 % ≈ **77 MB per council meeting-hour** on
  the wire, ≈ 1,550 h × 0.188 × 410 MB ≈ **120 GB** for the 14 Granicus
  councils — if their archives are alive; two of three transit-board rows are
  dead and need re-locating.
- **Per-body cap (§2.3):** BART alone produces 40 % × 3.8 h × 1.5 meetings/month
  ≈ 2.3 transcribed episode-hours per month, under the proposed 3 h cap; San
  José's two meetings gave 12 and 29 min/h → 2 × 3.8 × 0.2–0.5 ≈ 1.5–3.8 h/month,
  at the cap. The cap will bind on the big bodies exactly as intended.

## Pending / uncertain

1. **Test 5 (GPU RTFX on clips)** — needs `gpu_batch.py propose` on the T0
   clips (40 clips, 8.86 audio-h, 437 MB, ≈ $0.20 at 25×; the six `full.opus`
   files, 28.7 h, are there if a whole-meeting RTFX comparison is wanted) and
   Puck's approval. Not proposed here.
2. **Classifier accuracy scoring (the test plan's original test 6)** — Puck
   labels `label_sheet.csv` blind, then `scripts/score_labels.py`. Until then every share above carries the
   classifier's unmeasured error; the codebook decision on liaison reports
   (about vs mention) moves the transit share for North Bay cities.
3. **Test 4** is complete (40/40). Open: whether Puck keeps `full.opus` in
   T1 (138 GB for the window) and whether the cue-edge snap replaces the pad
   rule in §6; both are one-line decisions, not measurements. The 45.1 MB/h
   and 469× rates are from 6 pulls on one morning from one IP — YouTube's
   per-stream pacing may differ at scale.
4. The share is from 9 meetings of 5 bodies in one month; the body-level
   spread (0–29 min/h across councils) is the thing to model, not average.
5. AC Transit and MTC sources must be re-located before the census; Oakland
   is the only Granicus evidence for the "captions are served" claim.
6. Keyword files carry no density ASR variants yet; the T0 caption dump is
   the place to mine them (the classifier's `no` labels show what the
   conditional rule still lets through: "single family" in a junk-pickup
   item, "rail" in an ALPR comment).
