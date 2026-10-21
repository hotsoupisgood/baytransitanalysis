# T2 — the full 12-month run (plan, 2026-09-22)

Window **2025-09-20 → 2026-09-20**, census of *regular* meetings of every
located body (proposed DECISIONS #19). Part 1 (everything below the line) is
built and tested; part 2 is the run and needs Puck's approvals.

## 1. What part 1 produced

| | |
|---|---|
| `gpu_batch.py reconcile <batch> [--dry-run] [--terminate]` | review M7; never creates a pod, a live pod is reported not terminated |
| RunPod auth | `Authorization: Bearer` (review L1), proved live read-only first |
| `pipeline/spend.py` | RunPod + Claude ledger, stop lines, in `propose` / `launch` / `classify` / `status` |
| `classify.py --model` / `--batch` / `collect` | per-model prices and `prompt_version`, Message Batches API (50 % off) |
| `classifier_calls` key | `(id, prompt_version)` (review L2); migration rebuilt it, 316 rows / $1.1641 intact |
| Ingest | hallucinated speaker-tag prefixes stripped; per-body `initial_prompt` vocabulary; QA compares an episode with its meeting's captions |
| Sources | 33 bodies in `plan/channels.tsv`, **622 regular meetings / 2,284 meeting-hours** discovered |
| Tests | 260 checks (`make test`) |

## 2. The classifier instrument (decision rule, applied)

The rule, written before the test: agreement ≥ 0.90 on `about_*` vs not →
Haiku 4.5 for T2; 0.80–0.90 → Haiku with a Sonnet re-check of `mention_only`
and borderline rows; < 0.80 → Sonnet 4.6 via the Batches API.

`claude-haiku-4-5-20251001` on the 198 T0 candidate windows, same codebook,
byte-identical text (`scripts/validate_classifier.py`,
`output/t2/validate/claude-haiku-4-5-20251001/report.md`):

| | |
|---|---|
| exact agreement (5 labels) | **0.712** (κ 0.599) |
| `about_*` vs not | **0.758** (κ 0.521) |
| recall / precision of Sonnet's `about_*` | 0.671 / 0.979 |
| family agreement where both say `about_*` | 92/94 = 0.979 |
| about-minutes over the 198 windows | Sonnet 482 → Haiku 350 (−27 %) |
| cost, cache | $0.52, **0 cache reads** (Haiku's minimum cacheable prefix is 4,096 tokens; the codebook is 1,531) |

Confusion, rows = Sonnet: `about_transit` 94 → 52 transit, **39 mention_only**,
2 no; `about_density` 43 → 37 density, 5 mention_only; `mention_only` 48 → 44
mention_only. Haiku is *conservative*: it almost never calls something an
episode that Sonnet did not (precision 0.98), but it demotes a third of the
transit episodes — nearly all BART board items, where a whole agenda item on
ridership reads to it as "the board discussing its own business". At 0.758
the rule says **Sonnet 4.6 via the Batches API** (proposed DECISIONS #17).
Cost below.

## 3. Expected numbers (from discovery × the T0 rates)

`scripts/t2_expected.py` (T0 shares: councils 18.8 %, boards 39.6 %; pad +9 %;
classifier $0.0222/meeting-h halved by Batches; GPU $0.021/audio-h at the
measured 25×). Meeting-hours marked * are filled with the body's mean where
the listing carries no duration.

| body | kind | platform | meetings | meeting-h | ep-h raw | ep-h capped | cap binds (months) | classifier $ (batch) | GPU $ |
|---|---|---|---|---|---|---|---|---|---|
| bart | transit_board | youtube | 16 | 67.7 | 29.2 | 26.7 | 4/12 | $0.75 | $0.56 |
| concord | council | youtube | 11 | 20.9 | 4.3 | 4.3 | 0/11 | $0.23 | $0.09 |
| cotati | council | youtube | 3 | 8.0 | 1.6 | 1.6 | 0/3 | $0.09 | $0.03 |
| cupertino | council | youtube | 31 | 80.9 | 16.6 | 16.6 | 0/10 | $0.90 | $0.35 |
| dalycity | council | youtube | 18 | 56.4 | 11.6 | 11.6 | 0/10 | $0.63 | $0.24 |
| fremont | council | granicus | 24 | 49.2 | 10.1 | 10.1 | 0/8 | $0.55 | $0.21 |
| hayward | council | granicus | 29 | 70.9 * | 14.5 | 14.5 | 0/13 | $0.79 | $0.31 |
| healdsburg | council | youtube | 20 | 56.8 | 11.6 | 11.6 | 0/11 | $0.63 | $0.24 |
| livermore | council | youtube | 17 | 22.8 | 4.7 | 4.7 | 0/11 | $0.25 | $0.10 |
| losaltos | council | youtube | 19 | 53.3 | 10.9 | 10.9 | 0/13 | $0.59 | $0.23 |
| milpitas | council | youtube | 16 | 62.2 | 12.7 | 12.7 | 0/9 | $0.69 | $0.27 |
| napa | county | granicus | 30 | 104.3 | 21.4 | 21.3 | 1/13 | $1.16 | $0.45 |
| oakland | council | granicus | 12 | 40.8 | 8.4 | 8.4 | 0/8 | $0.45 | $0.18 |
| paloalto | council | youtube | 41 | 218.4 | 44.8 | 33.9 | 10/12 | $2.42 | $0.71 |
| petaluma | council | youtube | 18 | 81.7 | 16.7 | 16.7 | 0/10 | $0.91 | $0.35 |
| pleasanton | council | youtube | 16 | 53.5 | 11.0 | 11.0 | 0/10 | $0.59 | $0.23 |
| richmond | council | granicus | 31 | 178.9 | 36.7 | 31.2 | 7/13 | $1.99 | $0.66 |
| rohnertpark | council | youtube | 20 | 50.1 | 10.3 | 10.3 | 0/13 | $0.56 | $0.22 |
| samtrans | transit_board | granicus | 12 | 45.6 * | 19.7 | 19.7 | 0/12 | $0.51 | $0.41 |
| sanjose | council | youtube | 35 | 121.5 | 24.9 | 24.4 | 1/12 | $1.35 | $0.51 |
| sanleandro | council | granicus | 29 | 89.0 * | 18.2 | 17.7 | 1/11 | $0.99 | $0.37 |
| sanmateo | council | youtube | 12 | 46.1 | 9.4 | 9.4 | 0/8 | $0.51 | $0.20 |
| sanrafael | council | youtube | 18 | 56.1 | 11.5 | 11.5 | 0/12 | $0.62 | $0.24 |
| santaclara | council | granicus | 40 | 183.0 * | 37.5 | 32.5 | 5/13 | $2.03 | $0.68 |
| santarosa | council | youtube | 21 | 98.1 | 20.1 | 19.9 | 2/13 | $1.09 | $0.42 |
| sausalito | council | granicus | 30 | 114.0 * | 23.4 | 23.2 | 1/12 | $1.27 | $0.49 |
| sunnyvale | council | youtube | 26 | 131.5 | 27.0 | 26.6 | 2/13 | $1.46 | $0.56 |
| vallejo | council | youtube | 16 | 78.0 | 16.0 | 16.0 | 0/10 | $0.87 | $0.34 |
| vta | transit_board | youtube | 11 | 44.7 | 19.3 | 18.2 | 2/9 | $0.50 | $0.38 |
| **total** | | | **622** | **2284** | **504** | **477** | | **$25.40** | **$10.00** |

(* mean-duration fill for listings without durations; cap 3 episode-h per body per month; YouTube fetch ≈ 14.9 h wall at one request per meeting; wire ≈ 134.9 GB; ≈ 10 GPU batches of ~50 episode-h)

Per-body cap sensitivity (transcribed episode-hours per body per month):

| cap | episode-h | GPU $ | batches of ~50 h |
|---|---|---|---|
| 1 h | 275 | $5.80 | 6 |
| 1.5 h | 367 | $7.70 | 8 |
| 2 h | 424 | $8.90 | 9 |
| 3 h (PROPOSAL §2.3) | 477 | $10.00 | 10 |

**Two numbers need Puck before part 2 runs:**

1. **GPU.** $6.67 of RunPod credit is left and the stop line is $9.50 total
   recorded ($3.32 already), so **$6.18 is spendable**. The 3 h cap needs
   $10.00; 1 h needs $5.80. Either drop the cap to **1 h** (recommended: it
   binds on 10 bodies but keeps every *count*, only transcripts are capped)
   or top up the credit and raise `RUNPOD_CREDIT_USD`.
2. **Claude.** The classifier over 2,284 meeting-hours is **$25.40 via
   Batches** ($50.80 direct) against `CLAUDE_CAP_USD` $10 with $1.16 spent.
   `classify.py` will refuse partway through. Raise the cap to $30, or
   classify a subset of bodies first.

Fetch and disk, unchanged by either decision: ~15 h of wall clock for the 385
YouTube meetings at one request each (the 45–180 s gaps are 80 % of it), ~135
GB on the wire, ~35 GB of clips on disk plus whole-meeting audio if kept.

## 3a. Part 2 progress (2026-09-22, first run)

Sanity passed first: `make test` 260 checks, spend RunPod $3.32 / Claude $1.69.

**Captions.** Started over all 610 `discovered` meetings. YouTube **stopped
itself on an HTTP 429 after 17 requests / 31 min** (`5Nq_-rzWqPc`, exact text
in `output/t2/captions.log`); the row stayed `discovered` and nothing was
retried. The nine Granicus-only bodies were then run separately
(`output/t2/captions_granicus.log`) — a different host, ≥ 5 s pacing — and
finished **225/225, 0 failed**. Totals: 251 captioned, 359 still `discovered`
(all YouTube).

**Granicus caption coverage is much worse than §5 feared, and also better.**
Of 234 Granicus meetings only **51 carry a caption track** — and six bodies
serve *none at all*, across every meeting in the window:

| body | meetings | meeting-h | captions |
|---|---|---|---|
| richmond | 31 | 178.9 | none |
| santaclara | 40 | 178.4 | none |
| sanleandro | 29 | 82.8 | none |
| hayward | 29 | 68.5 | none |
| sausalito | 30 | n/a | none |
| samtrans | 12 | n/a | none |
| fremont | 22 of 24 | 43.2 | **English** (new: §5 expected Oakland only) |
| napa | 20 of 30 | 75.4 | **Spanish** (see below) |
| oakland | 9 of 12 | 29.8 | English |

Those 171 no-caption meetings are now `no_episodes` with that reason. They are
two of the three largest bodies in §3 (Richmond, Santa Clara) plus SamTrans —
one of only three located transit boards. An ASR-only locate pass for them is
still the separate decision of PROPOSAL §2.2; **this run does not improvise
one**.

**Napa's caption track is Spanish.** `https://napa.granicus.com/videos/<clip>/
captions.vtt` is the only track the player page exposes and it is a Spanish
interpretation, so the English keyword matcher found **2 candidate windows over
20 meetings / 75 h** and all 20 went to `no_episodes`. That is an instrument
failure, not a finding about Napa County. Needs a decision: find an English
track, or ASR, or drop Napa.

**Classify.** Sonnet 4.6 via the Batches API, two submissions by body group:
`msgbatch_018NeiFpS6Y32XBkymASyDLg` (fremont/napa/oakland, 192 windows) and
`msgbatch_01F7rADU4TepF15khn85jHbi` (the 16 YouTube bodies captioned before the
429, 306 windows). Both collected, 0 errored. 1,010 labels: 310
`about_density`, 256 `about_transit`, 50 `about_both`, 336 `mention_only`, 58
`no`. **$1.19 actual** against a $0.97 estimate; Claude ledger $1.69 → $2.88.
Two windows of `gr-oakland-7541` come back unparseable from both the batch and
a direct call (its caption track is garbled; the third window is labelled `no`
with that rationale) — that meeting alone is still `captioned` and stops a
plain `classify.py run`, so the leftovers were finished with `--bodies`.

**Cut.** 21.75 located episode-h exist (18.14 h YouTube, 3.61 h Granicus). Only
the Granicus half can be cut while YouTube is blocked; `episodes.py --bodies
fremont,oakland` is running (`output/t2/episodes_granicus.log`), ~4 min per
clip.

**No GPU batch is proposable.** `propose --speaker fremont,oakland --min-hours
40` → 0.53 capped audio-h ready. Even the whole located pool (21.75 h) is below
the 40 h threshold, so **nothing can go to Puck until YouTube resumes**.
(Superseded by §3b: the threshold is now 15 h and the Granicus half has grown.)

Rates measured so far, against §3's T0-derived assumptions: the classifier cost
$1.19 for 498 windows over 68 caption-bearing meetings, and located episode
time is running at roughly 1.2 % of meeting time on these bodies — far under
the 18.8 % / 39.6 % shares §3 projects. Re-run `scripts/t2_expected.py` before
trusting any cost line in §3.

## 3b. Part 2, second session (2026-09-22): slow down, then stop guessing

Four changes, all of them because the first session's two blockers — YouTube's
429 and "six Granicus bodies serve no captions" — turned out to be one real
constraint and one parser bug.

**1. The caption stage is a trickle worker now.** `captions.py --loop` takes
**one meeting every 8–12 minutes** (randomised; `--sleep-requests 2` inside
yt-dlp as before), which is ~50 meetings a day and ~7 days for the 359 that
are left. On a bot check or 429 it stops, writes a **24 h row in the new
`backoffs` table**, messages Puck and **exits 0** — so launchd's `KeepAlive`
leaves it down, and a restart inside those 24 h reads the row and asks YouTube
nothing (`--backoff-status`, `--clear-backoff`). The plist
`ops/launchd/com.ytpt.captions.plist` is installed and **running** as of
2026-09-22 23:58 UTC; the first fetches were `5Nq_-rzWqPc` (cupertino — the
very video that 429'd the first run), `TkZkdilj_nA`, `Chj4lSP9je4`,
`L3CfErTI854`, all clean, gaps 493 / 530 / 682 s.

**2. Santa Clara and Hayward were never caption-less** (probe:
`plan/sources_bayarea.md` "T2 locate probe"). They serve a full English track
through a Flowplayer config object with **no `<track>` element**, which is the
only thing `parse_player` read, so 69 meetings went to `no_episodes` as
`captions='none'`. Fixed, and `captions.py --refetch` re-asked all 69 player
pages: **37 now have an English track** — Santa Clara **34 of 40 (178 h)**,
Hayward 3 of 29. Those meetings are back at `captioned`, i.e. classify-ready,
**not yet classified** (§3c). Sausalito and SamTrans really do serve nothing:
their endpoint is the empty 8-byte `WEBVTT` file.

**3. `method='agenda_index'`: locating episodes from agenda item titles**
(`pipeline/agenda_index.py`, PROPOSAL §2.2's $0 route). An index point is
`<div class="index-point" time="s">title</div>`; an item's window is
[its time, the next item's time), the density/transit keyword families are
matched against the **title**, and consecutive matching items merge into one
episode. Rows are `<meeting>_a<n>` (the classifier's are `_e<n>`), `label`,
`classifier_model` and `prompt_version` stay NULL, `agenda_item` holds the
title. **A single item is capped at 60 minutes**: an index point says where an
item starts, not where it ends, and the raw spans ran to 4.4 h ("the rest of
the evening", Napa 5713). Run over the five index-bearing bodies:

| body | meetings | with index | mtg-h | episodes | episode-h | % of meeting time |
|---|---|---|---|---|---|---|
| santaclara | 40 | 38 | 178.4 | 23 | 12.4 | 7.0 % |
| napa | 30 | 28 | 104.3 | 7 | 3.8 | 3.6 % |
| sausalito | 30 | 26 | n/a | 10 | 6.8 | — |
| samtrans | 12 | 2 | n/a | 7 | 2.0 | — |
| hayward | 29 | 27 | 68.5 | 6 | 1.6 | 2.3 % |
| **total** | **141** | **121** | **351** | **53** | **26.5** | |

62 of 2,066 agenda items matched (3.0 %). The hits read right — "Study Session
on the Santa Clara Station Area Specific Plan", "Countywide Active
Transportation Plan", "900 Calhoun Rezoning", "AC Transit Budget & Service
Update". **This instrument's n is not comparable with the classifier's**: it
reads the agenda, so it is blind to twenty minutes of housing public comment
inside an item called "Consent Calendar", and its recall is the agenda's.
Report per `method`, never pooled. Hayward's 2.3 % is the warning sign — its
index points are mostly procedural boilerplate.

**4. Local ASR works and is affordable** (`scripts/local_asr.py`, docs/MODELS.md
§4). Installed `mlx-whisper 0.4.3` into a **new** scratch venv
`~/.local/share/ytpt/venv-asr` (the pinned `.venv` and the pyannote
`venv-embed` are untouched). Proved end to end on **`gr-sanleandro-3225`**
(San Leandro council, 2026-09-08, 1.14 h), a body with neither captions nor an
index:

| | |
|---|---|
| audio | `ASX.php` → https MP4 → `ffmpeg -vn -ac 1 -ar 16000`, **17 s**, 132 MB wav |
| model | `mlx-community/whisper-large-v3-turbo`, 16 kHz mono, no alignment, no diarization |
| ASR | **5.9 min wall for 1.14 h = 11.6× realtime**, 682 cues, 6,793 words |
| locate | VTT → `segment.py` → 1 candidate window → Sonnet 4.6 → `mention_only`, **0 episodes**, $0.0084 |

The transcript is legible and correctly timed (sampled by eye at 25–28 min:
public comment on data centres and tenant relocation, verbatim enough to code).
The classifier's verdict is defensible — the whole meeting has **16 keyword
hits in 6,742 words** and the one candidate window is a commenter mentioning
rent stabilisation in passing, not a housing item. **Caveat: this is a weak
recall test.** The meeting had almost nothing to find, so it proves the chain
runs and that the instrument does not hallucinate episodes; it does not prove
the ASR finds episodes a caption track would. Running one *captioned* Granicus
meeting through both routes and comparing is the test that would.

The row it writes is `captions='asr_local'` — never `auto`/`manual`/`granicus`
— so a locate track can never be mistaken for a caption track, and the located
episode is still cut from the real audio and transcribed on the GPU under the
one pinned corpus configuration (docs/MODELS.md).

**Projection for the rest — not started, needs Puck.** 81 meetings still have
neither a usable caption track nor index points:

| body | meetings | audio-h | Mac-h @ 11.6× |
|---|---|---|---|
| richmond | 31 | 178.9 | 15.4 |
| sanleandro | 28 | 87.8 | 7.6 |
| samtrans | 10 | 38.0 * | 3.3 |
| sausalito | 4 | 15.2 * | 1.3 |
| napa / fremont / hayward / santaclara | 8 | 18.5 | 1.6 |
| **total** | **81** | **338** | **29.1** |

(* mean-filled at 3.8 h.) **≈ 29 Mac-hours ≈ 1.2 days of continuous ASR**, plus
~39 GB fetched at the measured 116 MB per audio-hour. It is free and it
competes with nothing (the GPU is idle, YouTube is the bottleneck), but it
pins this Mac for a day and it is Puck's call.

## 3c. What is open after this session

1. **Santa Clara + Hayward classification.** 37 recovered caption tracks sit at
   `captioned`, unclassified. §3's estimate for those bodies is **$2.03 +
   $0.79 = $2.82 via Batches** against `CLAUDE_CAP_USD` $10 with **$2.89**
   spent. That is over this session's $2 allowance, so it was not run.
2. **The agenda_index episodes are located, not cut.** 53 episodes / 26.5 h.
   `episodes.py --bodies santaclara,hayward,sausalito,samtrans,napa` cuts them
   as HLS slices (~4 min per clip, Granicus-paced); that is what would make a
   real ≥ 15 h GPU batch.
3. **The ASR bulk run** (29 Mac-h) — see above.
4. **YouTube.** 355 meetings still `discovered`; the trickle worker is the only
   thing asking, at ~50 a day.

## 3d. Part 2, third session (2026-09-23): the three approved runs

Puck approved all three open items of §3c on WhatsApp at 07:57 ("yes asr, yes
cut, yes classify"). No YouTube request was made in this session — the caption
trickle worker owns that host and was not touched.

**1. Santa Clara + Hayward classified — and the instruments stopped hiding each
other.** §3c priced this at $2.82 from §3's meeting-hour rate. The real cost is
**$1.18**: 455 windows over 36 meetings as one batch
(`msgbatch_01472Yu1z6nHvc3dqbVhtf6M`, **455/455 succeeded, 0 errored**, $1.105)
plus $0.07 of direct calls for 21 windows on a meeting the cutter had claimed
at submit time. The meeting-hour rate over-predicts by 2.4x on these bodies
because it assumes the T0 candidate-window density; Santa Clara's caption
tracks are long but keyword-sparse. Re-run `scripts/t2_expected.py` before
trusting §3 (still true, now with a second body-pair of evidence).

| body | meetings | windows | about_density | about_transit | about_both | mention_only | no | episodes | $ |
|---|---|---|---|---|---|---|---|---|---|
| santaclara | 29 | 442 | 181 | 47 | 30 | 126 | 58 | 103 (15.4 h) | $1.074 |
| hayward | 2 | 13 | 2 | 7 | 0 | 0 | 4 | 2 (0.55 h) | $0.031 |

Getting there needed a fix, not just a run. **19 of the 37 recovered caption
tracks were unreachable**: `agenda_index` had already moved those meetings from
'captioned' to 'classified', and `classify.py` selects `status = 'captioned'`,
so the classifier would never have read them. Two instruments on one meeting
(§3b item 3) has to survive the status machine, so:

  * `classify.py --include-classified` also selects a meeting another
    instrument advanced, as long as it still has a caption track.
  * `finish_status` never clobbers the other instrument's row: a 'captioned'
    row still moves to 'classified' / 'no_episodes'; an advanced row keeps its
    status and gets an `events` row; and a meeting the cutter had **finished**
    goes back to 'downloaded' when new episodes are located, or `episodes.py`
    would never return for them.
  * `episodes.py` selects a 'cut' meeting that still has 'located' episodes.

**2. `method` now records the ROUTE, not the model.** An episodes row from a
Mac ASR track is `method='local_asr'`; from a real caption track,
`'classifier'` (`classify.method_for`, keyed on `captions='asr_local'`). Same
codebook, same Sonnet, different recall — SCHEMA.md's rule that n is reported
per method, never pooled, now covers the ASR route too.

**3. The agenda_index cut.** `episodes.py --bodies santaclara,napa,sausalito,
samtrans,hayward` over the 41 meetings holding the 53 episodes, HLS slices at
Granicus pace, ~90 s to 4 min per clip. Running detached
(`output/t2/episodes_agenda.log`); because it re-reads each meeting's located
episodes as it reaches it, it is also cutting the Santa Clara / Hayward
classifier episodes item 1 just created.

**4. The ASR bulk run, and the buffer limit nobody had hit.**
`scripts/asr_batch.py` drives `local_asr.py` over a body's backlog one meeting
at a time, resumably (a row already saying `captions='asr_local'` is skipped),
classifying each track as soon as it exists. Richmond + San Leandro: **59
meetings, 260.6 audio-h, 22.5 Mac-h at 11.6x**.

The first long meeting killed it: `gr-richmond-6042` (7.53 h) died on
`[metal::malloc] Attempting to allocate 4346926400 bytes which is greater than
the maximum allowed buffer size` — **mlx-whisper loads the whole file as one
Metal buffer**, and Richmond's meetings average 5.8 h. §3b's proof ran on a
1.14 h meeting and never met this. `local_asr.py` now transcribes in **90-minute
stream-copied chunks** and shifts each chunk's cues back onto the meeting's
clock (`chunk_offsets`, `shift_segments`); `--chunk-min` tunes it. A meeting
short enough for one buffer still goes through whole, exactly as before.

Two San Leandro clips (3037, 3150, both listed 0 h) have an **empty
`<REF HREF="" />`** in their ASX response — the archive holds no media for
them. That is 2 of 59; the driver logs them and moves on.

The ASR classifier is **not free**, which §3c's "$0" line elided: the tracks
still go through Sonnet to locate episodes. It is *cheap* — $0.0093 for a 7.53
h meeting, because the candidate windows are so few — so the whole 59-meeting
run projects to well under $1. `--claude-stop-usd` is an absolute ceiling on
the Claude ledger, checked before each meeting; past it the ASR keeps running
free and the tracks wait on disk for a separate, paid decision.

**And the recall worry of §3b item 4 is now the main finding, not a caveat.**
Three ASR meetings (10.2 audio-h) have produced **0 episodes**. The transcript
is not the problem — it carries as many words per hour as a real caption track:

| track | route | h | words | words/h | kw hits/100w | candidate windows | per h |
|---|---|---|---|---|---|---|---|
| gr-richmond-6042 | asr_local | 7.53 | 39,701 | 5,270 | 0.05 | 2 | **0.3** |
| gr-sanleandro-3225 | asr_local | 1.13 | 6,742 | 5,949 | 0.10 | 1 | **0.9** |
| gr-richmond-6038 | asr_local | 1.57 | 1,752 | 1,118 | 0.00 | 0 | 0.0 |
| gr-santaclara-2582 | granicus | 6.33 | 42,821 | 6,761 | 0.18 | 48 | **7.6** |
| gr-santaclara-2425 | granicus | 5.20 | 36,769 | 7,071 | 0.17 | 29 | **5.6** |
| gr-santaclara-2561 | granicus | 7.35 | 41,479 | 5,643 | 0.07 | 31 | **4.2** |

Note the third row from the bottom against the last: **Santa Clara 2561 has the
same keyword-hit rate as Richmond 6042 (0.07 vs 0.05) and yields 31 candidate
windows against 2**. Equal word density, comparable hit rate, an order of
magnitude fewer windows. Either Richmond genuinely has little to find, or the
hits in an ASR track are scattered where a caption track's cluster — and stage
1's 60-s window threshold only fires on a cluster. `gr-richmond-6038`'s 1,118
words/h over 1.57 h also looks like a defective track and should be inspected
by ear before it is trusted.

**This is not settled and 22 Mac-hours are riding on it.** The test §3b named
is still the one to run: take a Santa Clara meeting that HAS a caption track,
ASR it as well, and compare the two candidate-window sets on the same audio.
Until that runs, treat `method='local_asr'` counts as a floor, never as a rate,
and never pool them with the classifier's.

## 3e. Is a caption track a transcript? (2026-09-23, fourth session)

Hayward read as ~1,592 caption words per meeting where a 2-h session should be
~18,000, and produced 1 density + 1 transit episode over 29 meetings. That
looked like a 9 % coverage artefact across the body. **It is not.** The
arithmetic is 46,126 caption words divided by 29 meetings, and 26 of those 29
meetings have **no caption track at all** — the empty 8-byte `WEBVTT`
Hayward's endpoint serves (§3b item 2 recovered 3 of 29, not 29). Divide by the
3 meetings that do have a track and Hayward reads **130 wpm**, normal speech.
Same verdict as Puck's — the classifier result is an instrument artefact, not a
finding — and the same remedy, for a different reason.

`scripts/caption_audit.py` is the meter that settles this kind of question
without guessing: for every meeting whose row carries a track it takes the
words the **classifier** sees (`segment.read_segments`, so YouTube's rolling
duplicate cues are already collapsed — counting the raw file roughly doubles an
`auto` track), divides by the meeting's minutes, and reports coverage (last cue
end / duration) and language beside it. Under 60 wpm a track cannot be a
transcript of the whole meeting. Meetings with no track are counted in their
own column and **never averaged into a wpm** — that average is the whole
mistake above. `output/analysis/caption_audit.csv`, one row per meeting.

**622 meetings, 168 tracks, 24 flagged — and 22 of the 24 are one body.**

| body | mtgs | tracks | no track | median wpm | flagged | verdict |
|---|---|---|---|---|---|---|
| santaclara | 40 | 34 | 6 | 118 | 0/34 | healthy |
| oakland | 12 | 12 | 0 | 127 | 0/12 | healthy |
| fremont | 24 | 22 | 2 | 128 | 1/22 | healthy; `gr-fremont-453` stops at 39 % |
| napa | 30 | 20 | 10 | 141 | **20/20** | **Spanish track, English keywords** |
| hayward | 29 | 3 | **26** | 130 | 1/3 | tracks fine; the body has almost none |
| richmond | 31 | 11 | 20 | 106 | 2/11 | the two known bad ASR tracks |
| sanleandro | 29 | 7 | 22 | 134 | 0/7 | healthy |
| 22 YouTube bodies | 386 | 59 | 327 | 93–162 | 0/59 | healthy; the trickle worker owes the rest |

**There is no truncation epidemic.** Every body's median sits in the 93–162 wpm
band. Four individual tracks are genuinely short — `gr-fremont-453` (54 wpm,
last cue at 39 % of a 5 h meeting), `gr-hayward-2458` (32 wpm),
`gr-richmond-6033` (7 wpm) and `gr-richmond-6038` (19 wpm) — and the last two
are the defective **ASR** tracks §3d flagged by eye, now measured.

**Napa is the finding this audit exists for.** All 20 Granicus tracks are
**Spanish end to end** ("La Junta de Supervisores del Condado de Napa…"), at
141 wpm and 100 % coverage. A wpm-only audit passes them; the density and
transit keyword families are English, so the classifier read 104 audio-h and
**could not have matched anything**. `guess_language` (function-word rates, no
dependency) is in the audit for exactly this. The damage is contained: Napa's
classifier route located **0 episodes** (2 `mention` rows), and all 7 Napa
episodes are `method='agenda_index'`, read off English agenda titles. So
nothing in the corpus is wrong — but **Napa's classifier-route recall is zero
by construction, not a measurement**, and must never be reported as one.
`captions_lang`/`captions_label` are NULL on Napa's rows: the `<track>` element
§3b item 2 reads carries them, the Flowplayer object Napa uses does not.

**Santa Clara passes and was left alone.** 34 tracks, median 118 wpm, 0
flagged, 6 meetings with no track. Its 105 episodes and its $1.18 stand.

**Two queue fixes, because Hayward's remedy would have destroyed evidence.**
`asr_batch.select` took any meeting not already `captions='asr_local'`, and
`local_asr.py --register` overwrites `captions` with `asr_local` — so
`asr_batch.py --bodies santaclara` would have replaced 34 good caption tracks
with deliberately low-quality locate tracks and silently changed those
meetings' instrument. `select` now skips a meeting whose track **passes the
audit** (`usable_caption_track`; `--redo-captioned` forces it anyway) and keeps
the ones that fail, which is what the ASR route is for. Second: `--register` on
a row another stage has advanced (`cut`, `downloaded`) used to log "NOT
registered" and write nothing, so the driver would re-fetch that meeting
forever; it now records the track, leaves the status alone, writes an `events`
row, and `asr_batch` passes `--include-classified` so `classify.py` actually
reads it. `finish_status` still owns the move back (§3d item 1).

**Hayward queued: 27 meetings, 62.9 audio-h.** The 26 with no track plus
`gr-hayward-2458`; `gr-hayward-2462` and `gr-hayward-2476` are healthy, already
cut, and excluded. Audio verified first on `gr-hayward-2326`: the player page
carries an **m3u8** (no `ASX.php` hop, 1 request), 59 MB wav in **4.6 min for
0.51 h**. That is ~9 min of wall per audio-hour against Richmond's progressive
MP4, so Hayward is **fetch-bound, not ASR-bound**: ≈ 9.4 h of fetching + 5.4
Mac-h of ASR ≈ 15 h wall, ~26 GB over the wire, ~7.3 GB of wav kept. Queued
*behind* the running Richmond/San Leandro batch, not beside it (one ffmpeg, one
Metal buffer at a time), log `output/t2/asr_batch_hayward.log`,
`--claude-stop-usd 7.50` (≈ $1.1 of classifier for the 27).

**Nothing was reset, superseded or deleted.** Hayward's 2 classifier episodes
both come from `gr-hayward-2462`, a healthy 140 wpm track, and its 6
`agenda_index` episodes are a different instrument — all of them stay. The 26
queued meetings have no classifier episodes to supersede, and `gr-hayward-2458`
has none either; when its row's `captions_path` moves to the ASR VTT the old
truncated file stays on disk.

**Napa was NOT queued and needs Puck.** 30 meetings / ~110 audio-h ≈ 9.5 Mac-h
plus HLS fetch time, on top of Hayward's 15 h and Richmond/San Leandro's
remaining ~14 h. That is past §2's "report before queueing more than ~30
meetings" line. The alternative is cheaper and better: Napa's track is a
*translation*, so a Spanish keyword list and a Spanish-aware codebook would
read the existing 104 h for classifier money and no Mac time at all. Either way
it is a decision, not a default.

**The agenda_index cut run died and was not restarted.** §3d item 3's detached
`episodes.py` crashed at 20:02 UTC after cutting 16 meetings:
`granicus.audio_slice`
lets `subprocess.run(timeout=1800)` raise, and a 17-min HLS slice of
`gr-santaclara-2497` exceeded it, so one slow slice took the whole run down
instead of failing one clip. It left `gr-santaclara-2497` stranded in
`downloading` (a restart reclaims it — `state.reclaim_stuck`). Not restarted
here: restarting it while the ASR batch is live puts two ffmpeg pulls on
Granicus at once. The fix is to catch `TimeoutExpired` in `audio_slice` and
return it as this clip's failure, the way every other slice error is handled.

## 3f. Part 2, fifth session (2026-09-23): the cut restarted, and Napa read in Spanish

Three approved items (Puck, WhatsApp 15:02). `make test` 355 checks. Nothing
was deleted; the one file moved is named below.

**1. `audio_slice` no longer takes the run down with the clip.** §3e's
diagnosis was right and the fix has three parts, because the first version of
it broke the cutter a second way and that is worth recording.

  * `subprocess.TimeoutExpired` is caught and returned as **this clip's**
    failure, like every other ffmpeg error. The meeting stays claimable and
    `state.reclaim_stuck` un-stranded `gr-santaclara-2497` on the next start,
    exactly as §3e predicted.
  * **The timeout scales with the slice.** A flat 1800 s was wrong in both
    directions. Over the 53 slices the crashed run did cut, wall clock ran at
    a median **0.33x** the slice's own length, p90 1.31x and a worst case
    **2.25x** — the archive host's throughput varies by an order of magnitude
    within one session (Santa Clara 2517's five-minute slices took 392–641 s
    each, while 2561's 61-minute slice took 1,185 s). So 1800 s is 30x
    generous for a 4-min slice and too tight for a 17-min one. It is now
    `max(900 s, 4.0 x slice)` capped at 3 h: the ratio is 1.8x the worst
    measured, the floor covers the fixed cost that dominates a short slice (a
    30 s probe slice measured **175 s**, nearly all of it manifest walk and
    seek), and the cap is a hang detector, not a throughput guard.
  * **ffmpeg writes to `<name>.part.m4a` and the file is renamed onto the real
    path only on exit 0.** The crash had left a 7.9 MB headless `.m4a` with no
    moov atom at the real path, and `episodes.py` skips a clip that "is already
    on disk" — so the invariant *a file at the clip path is a complete clip*
    had to become true rather than assumed. The extension stays **last**: the
    first attempt wrote `<name>.m4a.part`, ffmpeg picks its muxer from the
    extension, and 26 consecutive slices died with "Error opening output files:
    Invalid argument" before the run was stopped. Those 26 episodes went back
    to 'located' with `--requeue-failed`; the stranded partial was renamed to
    `gr-santaclara-2497_e1.part.m4a`, not deleted.

**2. Napa: a Spanish keyword pair, not 9.5 Mac-hours of ASR.**
`plan/keywords_density_es.txt` and `plan/keywords_transit_es.txt` are the
Spanish construct of the two families — same `@family density` / `@family
transit` tags, so a Napa episode's `tags` mean what a Fremont one's do. Three
things differ from a translation and all three are properties of *interpreted*
speech: agency names and statutes come through in English (BART, SB 9, RHNA,
ADU) and are repeated rather than translated; the loader's optional-plural rule
is English, so "-es" plurals are listed out; accents may not survive a
captioner, so load-bearing terms are listed both ways. Two conditional terms
are marked for reasons the English file has no equivalent of:
**`desplazamiento`** is displacement in housing Spanish and a *commute* in
transport Spanish, and **`frecuencia`** would otherwise fire on "con
frecuencia" — simply "often" — in every meeting. Going the other way, `tren`
needs no guard where English `train` is a verb often enough to need one.

**Language selection: no new column.** `classify.caption_language` takes the
site's own declaration (`raw_meta.captions_lang`, from the `<track>` element
`parse_player` reads) when there is one, and otherwise reads the track with
`segment.guess_language` — the same function `caption_audit.py` reports in the
`language` column of `output/analysis/caption_audit.csv`. A stored column would
be a second source of truth that is NULL in exactly the case it exists for:
Napa's player carries its track in a Flowplayer config object, which has no
lang attribute, so all 20 Spanish tracks declare nothing. The detector moved
into `segment.py` and the audit now imports it, so the two can never disagree.
All 20 Napa tracks detect `es`; every other body detects `en` and its stage 1
is byte-for-byte unchanged.

**The codebook is unchanged.** Translating or extending `INSTRUCTIONS` would
change `CODEBOOK_VERSION` and with it the `prompt_version` of every window
already measured — all of them re-billed, none comparable with the T0 labels.
A Spanish window instead carries one sentence in the **user** turn (it is an
interpretation of an English meeting; same definitions, same five labels, same
English label names; interpretation lags and paraphrases, so judge the item and
not the rendering), outside the cached prefix, and is recorded under its own
`prompt_version` **`t0-a4a8e55760/claude-sonnet-4-6/es`**. Two recalls, two
instrument ids, never pooled by accident. `--include-no-episodes` re-reads a
meeting the *English* instrument answered 'no_episodes' about; it is never a
default, and a re-read that still finds nothing writes an `events` row rather
than touching the status.

**Result: 10 of Napa's 20 caption-bearing meetings, and it stopped there on
cost.** Stage 1 alone tells the story — over the 20 tracks the English pair
finds **2 candidate windows** and the Spanish pair finds **211**.

| | English | Spanish |
|---|---|---|
| candidate windows, 20 meetings / 104 h | 2 | **211** |
| meetings classified this session | — | 10 (93 windows, `msgbatch_01M5ELEadc9drJPgMyRo8mM5`, 93/93 succeeded, 0 errored) |
| labels | — | 17 about_density, 9 about_transit, 44 mention_only, 23 no |
| episodes | 0 over 104 h | **22 (15 density / 56 min, 7 transit / 25 min)** |
| $ | — | **$0.3901 actual against a $0.1939 estimate** |

**The estimator is 2.0x low here and the reason is not Spanish alone.** Two
causes, both measured on this batch. Spanish costs **2.74 input tokens per
word** against English's 2.04 (+34 %: the same words, a worse tokenisation).
And the codebook **cache did not take** — 102,577 cache-*write* tokens over 93
requests, 1,103 per window, where Santa Clara's 455-request batch read 1,164
*cached* tokens per window. A Message Batch fans out in parallel, so a *small*
batch starts most of its requests before the first cache write lands and pays
1.25x input for the codebook on nearly every call. That is ~$0.19 of the $0.39
— half the cost of this batch was batch size, not language. `estimate_usd`
assumes an all-cache-read world and should not be trusted for a batch under a
few hundred requests; §3's $25.40 line has the same optimism in it. Fixing the
estimator was out of scope this session and is worth doing before the next big
submission.

The remaining **10 Napa meetings are 118 windows ≈ $0.49**, which would put
the Napa total at ~$0.88 against the $0.50 this session was authorised to
spend, so they were **not submitted**. They resume from the DB: the same
command with no `--max` submits exactly those 118 and nothing else.

**Spanish recall reads right, checked by eye on two episodes.**
`gr-napa-5860_e1` (about_density) is the staff presentation on the ordinance
amending Napa County Code 17.17 — urban lot splits under SB 9, removal of the
historic-district exemptions and of the "specific adverse impact" grounds for
denial, plus objective standards for unit development. `gr-napa-5753_e1`
(about_transit) is the handoff into the Napa Valley Transportation Authority's
**Countywide Active Transportation Plan** presentation, bike and pedestrian
network. Both are legible despite a rough interpreter ("una pelea de rumores
ligeros" for "edge line rumble strips"), and the rationales come back in
English naming the right subject. The strongest check is external: the
`agenda_index` titles are **English** and were read by a different instrument,
and they name the same items — "Urban Lot Split", "Countywide General Plan
Update", "Countywide Active Transportation Plan". Two independent routes
agreeing on the same spans is better evidence than either alone.

**3. `gr-fremont-453`: the TRACK is short, the meeting is not.** The listing
says 5.00 h; `ffprobe` on the HLS master (playlist only, no media fetched, one
request) says **18,001.6 s = 5.00 h**. The audio is whole and Granicus served a
caption track that stops at 7,034.5 s — **39 %** of it. Over the span it *does*
cover the track reads **139 wpm**, entirely normal; its 54 wpm is an artefact
of dividing by the whole meeting.

**How to catch this cheaply everywhere: the audit's `coverage` column already
does, and the threshold was too loose.** Coverage is **bimodal** over the 168
real tracks — median **1.000**, 10th percentile **0.993** — so a track either
runs to the end or stops dead, and there is nothing in between to make a false
positive out of. `MIN_COVERAGE` was 0.50 and caught 2 tracks (1.2 %); at
**0.95** it catches **5 of 168 (3.0 %)**. It costs $0, no network and no audio,
and it is the sharper of the two meters for this failure: a wpm floor catches
453 only because the denominator is the whole meeting, and a track stopping at
80 % would read ~110 wpm and sail past it. Tightening newly flags
**`UEV2mj4Qf-s`** (Petaluma, 7.27 h, 128 wpm, coverage **86 %** — about an hour
of meeting missing from a track that passes every other check, and the row is
'classified', so its episodes were located from 86 % of the meeting) and
`gr-santaclara-2400` (a 5-minute meeting, coverage 95 %, trivial). Petaluma's
is real and needs a decision the way Napa's did.

## 3g. Part 2, sixth session (2026-09-23): b17's real cause, the five re-reads, Napa's second half

Puck's five-item plan, in order. Two of its five premises did not survive
measurement; both are written up here rather than quietly worked around.

### 1. b17 did not fail on level. It failed on our own downmix.

The plan's diagnosis was that Fremont's clips are too quiet for the VAD (failed
clips at a median −43.2 dB mean volume against −20.9 dB for clips that passed),
and the fix was loudness normalisation at encode time. The first half of that is
right and the second half is not, because the −20.9 dB comparison does not
reproduce and the cause is one step further down.

Measuring all 83 b17 clips, **source `.m4a` against the `.opus` the GPU actually
reads**:

| | failed (30 Fremont) | passed (53) |
|---|---|---|
| source, one channel | −43.2 dB median | −38.5 dB median |
| the opus we shipped | **−76.2 dB** median | −36.1 dB median |

The source levels are 4.7 dB apart — not a difference that decides anything, and
45 of the 53 clips that PASSED sit below −30 dB, one of them at −46.7 dB. The
opus levels are 40 dB apart. So the signal was destroyed between the two, by us.

`transcode.ffmpeg_command` has always passed `-ac 1`, which averages the
channels. Fremont's Granicus MP4s are **polarity-inverted stereo** (L = −R), so
the average is cancellation:

```
gr-fremont-13_e3.m4a   left  -42.1 dB    right -42.0 dB    (L+R)/2  -77.8 dB
```

Over the whole batch the separation is total: **32 of 32 Fremont clips lose
32–38 dB to the downmix, 51 of 51 clips from the other six bodies lose 0.0 dB**
(they are real dual-mono). That is the whole of the QA failure, and it is a
property of Fremont's encoder, not of the room.

This matters beyond bookkeeping, because loudness normalisation alone would not
have fixed it. Applied before the downmix, the cancellation still happens after.
Applied after, it amplifies a −78 dB residue — noise — by 60 dB, and the clips
would have failed QA a second time at a second $0.07.

`pipeline/transcode.py` now measures before it encodes (see its "LEVEL" block):

- `downmix_filter` compares `pan=mono|c0=c0` against the summed downmix and
  takes one channel when the sum sits `POLARITY_DROP_DB` (10 dB) or more below
  it. Real dual-mono measures 0.0 dB and is left to `-ac 1` exactly as before.
- then, on the mono signal that will actually be encoded, `loudnorm` to
  `LOUDNORM_I` (−16 LUFS) when the mean is under `NORMALISE_BELOW_DBFS` (−30 dB,
  Puck's number). This is the step the plan asked for and it is kept — it is
  cheap insurance for a genuinely quiet room — but it is not what fixed b17.
- both decisions and the pre/post levels land on the clip's own `videos` row
  (`kind='episode'`, so it IS the episode's row) under
  `raw_meta.audio_levels`. An existing JSON column fitted; no column was added.
  `plan/SCHEMA.md` documents the key.

A clip with ordinary channels above the threshold gets no `-af` at all and
encodes byte-for-byte as before.

**One thing for Puck to decide.** `NORMALISE_BELOW_DBFS = -30` was chosen
against the −20.9 dB figure. The passing corpus really sits at −36.1 dB, so at
−30 dB roughly 85 % of future clips will be normalised — a change to a path that
is working, on every body, not just Fremont. Nothing already on disk was
re-encoded, so this only bites from the next cut onward. `-45` would fire only
on genuinely broken audio. It is one setting either way.

### 2. The five coverage-flagged tracks

`MIN_COVERAGE` 0.95 flags 5 of 168. Four are Granicus and were re-read from
audio with `scripts/local_asr.py --register --language auto`; the fifth is on
YouTube and was **not fetched**.

| track | h | coverage | done |
|---|---|---|---|
| `gr-santaclara-2400` | 0.08 | 0.949 | re-read, 362 words, classified → `no_episodes` ($0) |
| `gr-fremont-453` | 5.0 | 0.391 | re-read running |
| `gr-napa-5713` | 7.6 | 0.711 | re-read running |
| `gr-napa-5724` | 7.98 | 0.087 | re-read running |
| `UEV2mj4Qf-s` (Petaluma) | 7.27 | 0.862 | **queued, not fetched** |

Petaluma's audio is on YouTube and `com.ytpt.captions` is in a bot-check backoff
until 2026-09-24T04:12:47Z. AGENTS.md rule 2: no request until Puck says resume.
The row carries an events note saying so, and
`output/t2/asr/pending_youtube.json` holds the exact command to run once the
backoff clears.

`local_asr.py` gained `--language` (default `en`, unchanged; `auto` detects with
`segment.guess_language` and names both the VTT and its `Language:` header from
the result) — it hardcoded `Language: en` on every track it wrote, which would
have mislabelled a Spanish re-read and sent stage 1 to the wrong keyword files.

### 3. `estimate_usd`: there are two of them, and they are wrong in opposite directions

**The classifier one was low.** It priced a window at `words * 1.35` tokens and
credited a codebook cache read on every call. Measured over the 1,044 billed
windows, `classifier_calls.input_tokens / words` is **2.04 for English, 2.74 for
Spanish**, and the codebook is **1,531 tokens** (3.64 chars/token, not 3.2). The
cache is the other half: the share of the codebook served from cache is 28 % at
93 windows, 47 % at 190 and 76 % at 455, so on a small batch the writes at 1.25×
input never amortise. It now prices the codebook at plain input below
`CACHE_PAYS_OFF_WINDOWS` (300) and at the measured 75/25 read/write blend above,
with 40 output tokens per call (measured 39.0–39.9, not the 60 assumed).

Against every batch that has been billed:

| batch | lang | n | old est | new est | actual |
|---|---|---|---|---|---|
| `msgbatch_018N…` | en | 192 | $0.35 (−45 %) | $0.75 (+16 %) | $0.638 |
| `msgbatch_01F7…` | en | 306 | $0.62 (+12 %) | $0.83 (+50 %) | $0.552 |
| `msgbatch_0147…` | en | 455 | $0.82 (−26 %) | $1.12 (+1 %) | $1.105 |
| `msgbatch_01M5…` | es | 93 | $0.19 (−50 %) | $0.41 (+4 %) | $0.390 |
| `msgbatch_01Rz…` | es | 59 | — | $0.28 (+27 %) | $0.216 |

The systematic underestimate is gone and the estimator is never under the bill,
which is what a cap check needs. It is **not** more accurate in both directions:
the two batches it now overstates most (+50 %, +27 %) are ones that got a warm
cache from a batch submitted a minute earlier, and English caption tokenisation
itself ranges 1.69–2.24 tokens/word across the four English batches. Cache
warmth is not a function of batch size, so a size threshold cannot capture it.

`worst_case_usd` — the number `--cap-usd` and `spend.claude_refusal` are checked
against — had the same `len(text)/3.2` in it, which is **below** the measured
~2.8 chars/token. The one rail that had to be pessimistic was optimistic. Fixed.

**The GPU one was never low.** `gpu_batch.estimate_usd` runs high by design
(7-minute boot allowance × 1.5 contingency): b17 $0.50 estimated against $0.30
billed (+68 %), b13 +72 %, b15 +107 %. That is the safe direction for the one
gate on GPU spend and it was left alone. So the `estimate_usd runs 2× low` note
on the roadmap was true of the classifier only.

### 4. Napa's second half

8 of the remaining 10 went through the Spanish route as one Batches submission
(`msgbatch_01RzosaEWXQQEEFbzHHkcjGB`, 59 windows, `prompt_version
t0-a4a8e55760/claude-sonnet-4-6/es`). **Actual $0.2162**, against $0.28
estimated and the plan's ~$0.49. Claude ledger $5.35 → $5.57 of $10.

New: **11 episodes — 6 density (59 min), 3 transit (9 min), 2 both (50 min)** —
plus 15 mention rows, over 6 meetings; `gr-napa-5711` and `gr-napa-5727` yielded
no candidate window at all. Napa's Spanish route now stands at 32 located
episodes / 194 core-min over 16 meetings for $0.61.

The other 2 of the 10 are `gr-napa-5713` and `gr-napa-5724` — which are two of
the five tracks item 2 is re-reading, at 71 % and **8.7 %** coverage. Billing the
Spanish classifier to read 8.7 % of a meeting is the failure the audit exists to
catch, so they are being located from their audio instead, and will land under
`method='local_asr'`. Their classifier cost is $0 and their recall belongs to a
different instrument — never pool it with the 8.

`classify.py` gained `--ids` for this: `--bodies` is the wrong knob when a named
handful must be classified and the rest of the body must not be billed.

### 5. b18 proposed, not launched

The 30 Fremont clips were re-encoded through the fixed path
(`scripts/renormalise.py --batch b17 --status failed`), which reads the `.m4a`
already on disk, writes a new opus, and moves the previous one to
`output/opus/pre_normalise/` rather than deleting it. No network, no YouTube.

    opus mean volume   before −76.2 dB median   after −18.8 dB (range −21.3 … −16.8)

which is where the plan wanted them and 17 dB above the passing set's real
median. Rows went `failed → transcoded`, their episodes rows back to `cut`, and:

**`b18`: 30 episode clips, 3.25 episode-h, est $0.18, `awaiting_approval`.**

$0.18 is the GPU estimator's conservative figure; at b17's measured
$0.0211/audio-h the bill should be about **$0.07**, which is the plan's number.
No pod exists. Puck texts `approve b18`.

### What this session did not do

- The three large Granicus re-reads (`gr-fremont-453`, `gr-napa-5713`,
  `gr-napa-5724`, 20.6 audio-h) are running detached in
  `output/t2/asr_reread.log`, sharing the Mac with the Richmond/San Leandro ASR
  batch. They still need `classify.py run --ids … --include-classified
  --include-no-episodes` when they land.
- Petaluma `UEV2mj4Qf-s` is untouched, by rule.
- Nothing already on disk outside the 30 Fremont clips was re-encoded.

## 3h. Part 2, seventh session (2026-09-24): the second 429, and what the queue order is worth

The trickle worker resumed 2026-09-23 21:15 PDT, captioned 4 meetings and was
**429'd after 5 network requests in 41 minutes** — the 8–12 min gap of §3b was
still far too fast. It wrote the 24 h backoff (until 2026-09-24 04:56 UTC),
exited 0 and `com.ytpt.captions` stayed down, exactly as designed. **322 of 385
YouTube meetings still have no captions.** yt-dlp is current (2026.8.19), so
updating it is not the fix. Three changes, all $0, all measured; **no YouTube
request was made this session and the backoff was not touched**.

### 1. The queue is ordered by measured yield, not by date

Episodes located per captioned meeting varies **17-fold** across the 20 waiting
bodies — and at one meeting an hour the order decides what the study has in a
fortnight. Recomputed from the live DB (read-only), it agrees with Puck's table
to the digit on all 19 bodies he listed:

| rank | body | yield | eps | captioned | waiting | cum. waiting |
|---|---|---|---|---|---|---|
| 1 | bart | 16.67 | 50 | 3 | 13 | 13 |
| 2 | petaluma | 8.00 | 24 | 3 | 15 | 28 |
| 3 | vta | 8.00 | 16 | 2 | 9 | 37 |
| 4 | sanrafael | 7.33 | 22 | 3 | 15 | 52 |
| 5 | vallejo | 6.50 | 13 | 2 | 14 | 66 |
| 6 | sanjose | 5.80 | 29 | 5 | 30 | **96** |
| 7 | concord | 5.00 | 15 | 3 | 8 | 104 |
| 8 | losaltos | 4.67 | 14 | 3 | 16 | 120 |
| 9 | **cotati** | **3.92 (pooled prior)** | 0 | **0** | 3 | 123 |
| 10 | sanmateo | 3.67 | 11 | 3 | 9 | 132 |
| 11 | paloalto | 3.40 | 17 | 5 | 36 | 168 |
| 12 | dalycity | 2.33 | 7 | 3 | 15 | 183 |
| 13 | sunnyvale | 1.60 | 8 | 5 | 21 | 204 |
| 14 | milpitas | 1.50 | 3 | 2 | 14 | 218 |
| 15 | pleasanton | 1.50 | 3 | 2 | 14 | 232 |
| 16 | santarosa | 1.20 | 6 | 5 | 16 | 248 |
| 17 | cupertino | 1.00 | 3 | 3 | 28 | 276 |
| 18 | healdsburg | 1.00 | 3 | 3 | 17 | 293 |
| 19 | livermore | 1.00 | 3 | 3 | 14 | 307 |
| 20 | rohnertpark | **0.00** | 0 | 5 | 15 | 322 |

**One disagreement, and it is a 20th body Puck's table does not have: Cotati.**
It has **0 captioned meetings**, so it has no measurement at all — not a
measured zero like Rohnert Park's 0.00 over 5. It gets the **pooled rate (3.92 =
247 episodes / 63 captioned meetings)** as a prior and is marked
`measured=False` in the log, which puts it 9th instead of last. An unmeasured
body is not a zero-yield body.

`captions.body_yield` + `order_queue` (`pipeline/captions.py`): highest-yield
body first, **oldest** meeting first inside a body, recomputed on every pass —
a meeting captioned an hour ago changes its body's yield. The denominator counts
meetings that have been *through* the stage, including the ones that turned out
to have no track: those cost a request and returned nothing, which is exactly
what the order should know. Every log line carries the rank, the yield and the
denominator (`[yield rank 2/20: 8.00 ep/meeting from 24 ep over 3 captioned]`),
so the order can be audited after the fact; `--yield-table` prints it with no
network at all.

**Nothing is dropped.** Whether Rohnert Park belongs in the sample after 0
episodes over 5 meetings is a `DECISIONS.md` question for Puck; the worker only
sorts. The six highest-yield bodies hold **96 of the 322**.

### 2. One request an hour, with a hard daily cap

Defaults are now **60–70 min** between YouTube requests and **18 requests per
rolling 24 h** (`--min-gap` / `--max-gap` / `--daily-cap`, or
`CAPTIONS_MIN_GAP_S` / `CAPTIONS_MAX_GAP_S` / `CAPTIONS_DAILY_CAP` in `.env`).
The cap is deliberately below the 24 the gap alone implies: it is a backstop for
the case where the gap logic is wrong.

Both limits are enforced from a new **`worker_requests`** table (one row per
request actually sent, written *before* the call), not from a counter inside the
process. That is the hole the old code had: launchd's `KeepAlive` could restart
the worker after `ThrottleInterval` and the first request of the new process had
no gap in front of it. The new default also applies to a hand-run `--max N`,
because at the old 45–180 s it would have spent two thirds of a day's budget in
twenty minutes. A rate-limited one-shot run says why and stops; a `--loop` run
waits (it must not exit 0 — `SuccessfulExit=false` would leave it down for good).

**The 429 / bot-check path is untouched**: stop, write the 24 h backoff, notify,
exit 0, stay down. `--rate-status` prints requests-in-window versus the cap.
The plist needs no change; the defaults do the work when Puck resumes the worker.

### 3. Which bodies could leave YouTube (`plan/SOURCES-AUDIT.md`)

16 waiting bodies probed on Granicus / Legistar / PrimeGov / Swagit / CivicClerk
/ NovusAGENDA / CivicWeb, **zero YouTube requests**, evidence in
`output/t2/portal_probe/`. Only **Petaluma** has a real alternative: PrimeGov
points its 18 council meetings at **Swagit** (`petalumaca.new.swagit.com`),
whose player page carries a **voice-to-text transcript with per-phrase `data-ts`
timestamps** and an HLS stream on `archive-stream.granicus.com` —
`granicus.audio_slice` already cuts that host. **15 meetings, ~½ day of work**,
and Petaluma is rank 2.

Two traps the audit had to survive, both documented there: `*.civicclerk.com`,
`*.novusagenda.com` and `*.legistar.com` answer **200 for any slug** (controlled
against `nosuchcityxyz42`), and `livermoreca` / `concordca` on Swagit serve
**Carmel, Indiana's** 1,530 clips under a title with the local city's name in
it. Eleven bodies have agenda/minutes **documents** with no timestamps, which
cannot locate an episode; nine have nothing. **129 of the 322 must come through
YouTube.**

### What this session did not do

- No YouTube request of any kind; the backoff was not cleared and
  `com.ytpt.captions` was not kickstarted or booted out. It stays down until
  Puck resumes it.
- No spend: no Claude call, no GPU, no RunPod, no `gpu_batch.py`.
- The live DB was opened **read-only** for every measurement; the new
  `worker_requests` table appears in it the next time a worker connects.
- The Swagit adapter for Petaluma is **specified, not built**.

## 3i. The Swagit route, built (2026-09-24, second session)

`pipeline/swagit.py` is the fourth locate route, `episodes.method =
'swagit_transcript'`. Petaluma's clips come from its **own PrimeGov API**
(`ListArchivedMeetings?year=`, the only trustworthy source of a clip id), the
player page at `petalumaca.new.swagit.com/videos/<id>` gives the customer slug
(the ownership proof), the HLS master on `archive-stream.granicus.com`, a timed
agenda index with **start and end** per item, and a transcript with a timestamp
on **every word**. Words are grouped into phrase cues and written as a normal
`.vtt`, so stage 1, the cutter's cue-edge snapping and `caption_audit.py` read
it exactly as they read any other track. Audio needed **no new code**:
`platform='swagit'` + `raw_meta['m3u8']` sends the meeting down
`episodes.cut_granicus`, which already slices that host.

**13 of the 15 registered, not 15.** The specials of **2025-09-29** and
**2025-11-24** publish a ~2-minute closed-session clip (CALL TO ORDER → CLOSED
SESSION → REPORT OUT) against YouTube videos of 42 min and 1.78 h. Those clips
are complete records of the open portion, not truncations, but they are not the
same object, so `MIN_DURATION_SHARE = 0.5` refuses them and **both meetings stay
on the YouTube queue**. Whether a closed-session special belongs in the sample
is a `DECISIONS.md` question for Puck.

The superseded YouTube rows are **not deleted and not re-stated** — no status
was invented. `captions.select_rows` simply does not queue a meeting whose
non-YouTube twin carries a real transcript (`--include-routed` overrides). The
waiting queue went **322 → 309**; Petaluma's waiting count **15 → 2**.

**Recall** (`scripts/swagit_recall.py`, `plan/SWAGIT-RECALL.md`, $0): measured on
the three meetings that have both tracks. The two publications are offset by a
**constant per meeting** — −37.6 s, +163.8 s, +486.5 s, from unique shared
6-grams, with 33,937 of 33,937 and 16,551 of 16,553 landmarks inside 10 s of it.
That offset is between the two *publications* and does not touch the route: an
episode located on the Swagit transcript is cut from the Swagit stream, the
clock its own `seek(t)` anchors address. On the aligned clock, **candidate
recall 0.914** (0.624 unaligned, which is the offset and nothing else), and all
**18 of 18** episodes the paid YouTube route located on `UEV2mj4Qf-s` are
covered over at least 70 % of their span, span-weighted **0.956**. The Swagit
track is also slightly *longer* than YouTube's auto captions (58,757 vs 55,629
words on that meeting) at the same 152–162 wpm.

**Not classified.** Stage 2 over the 13 meetings (318 windows) estimates
**$1.06**, worst case $1.89, at the batch rate — over the $0.50 ceiling this
session had, so nothing was submitted. That is Puck's call:

```bash
.venv/bin/python pipeline/classify.py run --bodies petaluma --batch --dry-run
```

**Also found, not acted on:** the PrimeGov listing carries **30 more council
meetings** with Swagit clips that have no row in the DB (all of 2026-01 to
2026-04, and 2025-01 to 2025-09). They are outside what discovery found on
YouTube. Widening the frame is a study-design decision, so the adapter registers
**only clips that match a meeting already in the DB** and reports the rest.

### The one-shot resume (2026-09-24)

`ops/launchd/com.ytpt.resume-captions.plist` +
`ops/launchd/resume_captions_once.sh` fire **once**, at 22:05 local on
2026-09-24, nine minutes after the 429 backoff expires
(2026-09-25T04:56:16Z), and kickstart `com.ytpt.captions`.

It is **not** and must never become a standing auto-resume: `AGENTS.md` rule 2
says the worker parks itself on a bot check and waits for Puck, and a recurring
resume would overrule that for every future 429, invisibly. Three things keep it
single-shot — the calendar trigger is pinned to a **date**, the script **deletes
its own plist as its first action** (before any check, so no failure path leaves
a loadable job) and boots itself out as its last, and it re-reads `backoffs`
**read-only** and kickstarts nothing if the row is still live. It never clears or
shortens the backoff. Both branches were rehearsed under `RESUME_DRY_RUN=1`
against a fake label and a fake DB before it was installed.

## 4. The run, stage by stage

Each stage is idempotent and resumable; run detached and watch the log.

```bash
PY=.venv/bin/python
# 0. sanity
$PY pipeline/spend.py && $PY dashboard/status.py | head -8

# 1. captions for every discovered meeting (YouTube 45-180 s gaps; Granicus >= 5 s)
nohup $PY pipeline/captions.py > output/t2/captions.log 2>&1 &      # ~15 h, stops on a bot check

# 2. classify -- ONE batch per group of bodies, then collect (24 h max, usually minutes)
$PY pipeline/classify.py run --batch --dry-run                       # what it would submit, $0
nohup $PY pipeline/classify.py run --batch --bodies <group> > output/t2/classify_submit.log 2>&1 &
$PY pipeline/classify.py collect                                     # labels -> episodes rows

# 2b. the bodies with no usable caption track and no agenda index: Mac ASR, then the same classifier
nohup $PY scripts/asr_batch.py --bodies richmond,sanleandro --claude-stop-usd 6.09 \
    > output/t2/asr_batch.log 2>&1 &                                 # ~22 Mac-h, resumable, --status
# 3. cut the located episodes (whole-meeting audio once per YouTube meeting, HLS slices for Granicus)
nohup $PY pipeline/episodes.py > output/t2/episodes.log 2>&1 &       # ~3 days, stops on 403/429

# 4. propose a batch of >= 15 episode-h, text Puck, launch DETACHED after his "approve bN"
$PY pipeline/gpu_batch.py propose --speaker <bodies>      # BATCH_TARGET_HOURS = 15 h since 2026-09-22
#   -> Puck texts: approve bN
$PY pipeline/gpu_batch.py approve bN --via whatsapp
mkdir -p output/batches/bN && nohup $PY pipeline/gpu_batch.py launch bN > output/batches/bN/launch.log 2>&1 &
make qa QA_SRC=output/whisper                                        # after each batch
```

Order of bodies (largest first, so the biggest rates are measured earliest and
a cap decision can be revisited with data): Palo Alto, Santa Clara, Richmond,
San José, Sunnyvale, BART → Napa, Sausalito, Santa Rosa, SamTrans, VTA,
San Leandro, Petaluma, Cupertino → the remaining 19. Batch the GPU work by
body group, not by date, so one batch is one `initial_prompt` vocabulary.

## 5. Risks and what is unverified

- The per-body `initial_prompt` has **never run on a pod**; the first batch is
  where `asr.options._replace(initial_prompt=...)` is proved. A failure only
  logs and transcribes without it (`doc["initial_prompt"]` then says `null`).
- Granicus bodies serve captions on Oakland only so far; the caption stage
  records `captions='none'` for any that do not, and those meetings become
  `no_episodes` with that reason — an ASR-only locate pass for them is a
  separate decision (PROPOSAL §2.2), not part of this run.
- The 18.8 % / 39.6 % shares come from 9 meetings of 5 bodies in one month.
  Treat every projection above as an order of magnitude, and re-run
  `scripts/t2_expected.py` once 50 meetings are classified.
- Sausalito, SamTrans, Hayward, San Leandro and Santa Clara listings carry no
  durations; their meeting-hours are mean-filled until captions land.

## 6. Bodies not in the run (2026-09-22 probe, `output/t2/probe/`)

| Body | Finding |
|---|---|
| AC Transit | official YouTube has one board meeting (Dec 2023); Legistar links Granicus + Zoom (InSite). Not located |
| MTC / ABAG | Granicus archive stops 2016; YouTube search returns Charlotte's MTC. Not located |
| Caltrain (JPB) | no official channel; the SamTrans Granicus archive holds SamTrans board + committees only |
| SFMTA | **located 2026-10-18, view 55** (SFGovTV, `sanfrancisco.granicus.com`): 409 clips, regular board = the bare title `Municipal Transportation Agency`; 20 regular meetings in window, 80.6 h. The Sep probe stopped at view 40 |
| SF Board of Supervisors | **located 2026-10-18, view 10** (same site): 819 clips back to 2006, newest 2026-10-06; 37 regular meetings in window, 116.9 h. view_id=2 was a dead duplicate archive ending 2024-12-11 and is replaced in `plan/channels.tsv` |
| Berkeley council | views 1–12 are Rent Board / Housing Authority / BOLT |
| Marin County BoS | views 1–12 are Planning Commission / telecom agency |
| Mountain View, Alameda, Walnut Creek | the located view_id is a commission or stops in 2009 |
| Redwood City, Windsor, Novato, Sonoma County BoS | no live archive found; only third-party or pre-2022 uploads |

Located *by* the probe and now in the run: **Pleasanton** (`City of Pleasanton
Public Meetings`), **Livermore**, **Vallejo** — 49 meetings, 154 h.

Ten bodies remain unlocated. Recommendation: drop them from T2 and note the
gap (they are 4 of 9 Contra Costa / Solano-side councils and every one of the
four regional transit boards except BART, VTA and SamTrans — so the *board*
side of the corpus is thin, and the density side is nearly complete).

**2026-10-18 re-probe of the SF views (view ids above 40).** Both SF bodies are
on SFGovTV's Granicus after all — the September pass only tried view ids 1–40.
Registered in `plan/channels.tsv` and discovered into the state DB the same day
(57 meetings, 197.6 h, `output/sf/listings/`): `sanfrancisco` view 10 and
`sfmta` view 55. Both serve `captions.vtt` (BoS 3,767–4,563 cues per meeting on
3 of 3 sampled; SFMTA 2 of 3 — the newest clip's endpoint is the empty 8-byte
file) and agenda index points (BoS 40–47, SFMTA 17–18 per meeting), plus a
progressive MP4 on `archive-video.granicus.com` (the fast audio path). **Eight**
bodies now remain unlocated, and the regional-board side gains SFMTA.

View 10 carries the Board of Supervisors only — the **Land Use and
Transportation Committee** and the other BoS committees are on their own views
(not probed, 0 committee titles in view 10's 819 clips) and are **not**
registered; ask Puck (`sf committees`) before adding them.
