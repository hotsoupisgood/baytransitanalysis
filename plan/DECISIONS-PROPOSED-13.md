# Proposed amendment to DECISIONS #13 (locate before transcribing) — for Puck, 2026-10-18

Written by the supervisor agent. **Nothing in `plan/DECISIONS.md` has been
edited**; this file is the proposal and the evidence. Decide between the two
options at the end (or neither), and the next session writes the chosen
wording into DECISIONS.md.

## What #13 says, and what its reopen clause assumed

> **13. Caption method: locate before transcribing.** YouTube auto-captions
> (or a cheap ASR-only pass for Granicus bodies) find candidate windows;
> WhisperX with diarization runs only on the extracted, padded episodes.
> *Reopen if:* auto-captions prove too poor to locate (T0 recall < 0.9) — then
> the ASR-only pass runs on YouTube bodies too.

The reopen clause assumes the failure mode is the **transcript** (auto-captions
too rough for the keyword screen), and its remedy is a better locate track
(ASR). The measurement below says the dominant failure is the **keyword
screen's vocabulary**, on an ASR track — so the remedy the clause prescribes
would not have fixed what was measured. That is why this needs an amendment
rather than just the clause being triggered.

## Evidence

### 1. The locate-recall check (plan/RECALL-CHECK.md, 2026-10-17/18)

Six Richmond / San Leandro meetings (the `local_asr` route: Mac mlx-whisper
locate track, no captions exist), 22.6 audio-h, transcribed WHOLE on the GPU
(batch b35) and every 60-s window labelled with the pinned Sonnet codebook —
the exhaustive reference. Two Richmond meetings failed GPU QA (decoder loop,
5 % coverage) and carry no measurement; four meetings, 109 about-minutes, do.

| family | about-min | recall (pipeline, pre-expansion) | 95 % CI (6-cluster bootstrap) | screen-alone recall |
|---|---|---|---|---|
| density | 89 | **0.843** | 0.822–0.913 | 0.820 |
| transit | 23 | **0.304** | 0.000–0.600 | 0.522 |
| any about_* | 109 | **0.725** | 0.605–0.809 | 0.752 |

Of the 30 missed about-minutes, **23 (77 %) never cleared the keyword screen**
(most at exactly 0.0 hits/100w: the words were not in the list), **7 (23 %)**
were candidates the classifier called `mention_only` / `no`. The transit
misses were one Richmond sequence (Bay Trail, bike share, e-bike lending
library, protected bikeway, shoreline trail: 11 of 14 transit misses) and a
San Leandro biking-infrastructure contract — vocabulary, not transcription.

### 2. The keyword-screen expansion (same day, Puck-approved; $0 to re-screen)

57 active-transportation terms added to the transit family
(`plan/keywords_transit.txt`, list version `kw-dfd392a1fe` → `kw-24e8a4acd1`;
every `episodes` and `classifier_calls` row now records which list located it).
Re-run of the screen over all 292 caption/ASR tracks: **387 new candidate
windows in 100 meetings** (64 of them a never-read Napa ASR track, not the
expansion), estimated **$1.21** at the batch rate.

Re-scored on the same exhaustive labels (`scripts/recall_recompute.py`,
`output/validation/recall/post_kw2_screen_only/`), screen-alone recall:

| family | screen-alone recall, before → after | screen precision (marked minutes that are about anything) |
|---|---|---|
| density | 0.820 → 0.820 | before 0.500 (164 marked) |
| transit | 0.522 → **0.826** | after 0.447 (199 marked) |
| any | 0.752 → **0.817** | |

Pipeline recall after the 387 new windows were classified (**$0.75 actual**,
93 new episodes / 9.49 h corpus-wide; plan/RECALL-CHECK.md "Keyword screen
expansion"):

| family | pipeline recall before → after | 95 % CI after | misses by cause before → after |
|---|---|---|---|
| density | 0.843 → 0.843 | 0.822–0.913 | unchanged |
| transit | 0.304 → **0.435** | 0.000–1.000 | |
| any | 0.725 → **0.753** | 0.654–0.852 | screen 23 → 16, **classifier 7 → 11** |

The screen now reaches the transit minutes (0.83); the classifier then
called 4 of the 7 newly screened transit minutes `mention_only` / `no` (a
Bay Trail funding deliberation read in 60-s windows). **The residual gap has
moved from the screen to the classifier** (11 of 27 misses) and to density
vocabulary that was not in scope (feasibility, cap rates, vacancy tax). The
"any" interval still spans 0.80.

### 3. What whole-meeting transcription actually costs now (measured, b35)

| | measured | per audio-hour |
|---|---|---|
| GPU, WhisperX large-v3, **no diarization**, A40 $0.49/h | 22.64 h → **$0.2313** (28.3 GPU-min) | **$0.0102** |
| GPU with diarization (episode batches b31/b33, same pod type) | 22.35 h → $0.50; 16.26 h → $0.31 | ≈ $0.02 |
| Claude, exhaustive 60-s labelling (no screen), batch rate | 1,115 windows → $1.87 | **$0.083** |
| Claude, screened windows only (≈ 15 % of minutes marked) | | ≈ $0.012 |
| Mac mlx-whisper (Richmond/San Leandro run) | 260.6 audio-h → 22.5 Mac-h | 11.6× real time |

Corpus: 622 regular meetings, **2,284 meeting-hours** (`plan/T2-PLAN.md`);
YouTube 385 meetings / ~1,409 h (one yt-dlp request per meeting, ~45 MB per
meeting-hour ⇒ ~63 GB), Granicus 237 / ~704 h, Swagit 13 / 57 h.
RunPod credit: $6.03 of $10.00 recorded, stop line $9.50.

## Option 1 — keep locate-first; make recall a measured instrument property

Amend #13 to:

> **13 (amended).** Locate before transcribing: a **versioned keyword screen**
> (`segment.load_keywords().version` on every row) on the best locate track
> available (YouTube captions, Granicus captions, Swagit transcript, or a Mac
> ASR track), then the pinned classifier; diarized WhisperX only on the
> episodes. **Locate recall is measured per route** against an exhaustive
> whole-meeting GPU transcript + classifier labelling on a seeded sample
> (`scripts/recall_check.py`), and the measured recall (with its interval)
> is reported next to every rate and enters the rate model as a
> measurement-error term. *Reopen if:* measured pipeline recall for any
> family on any route is **< 0.80** after the vocabulary fixes it points to,
> or the screen's precision falls below 0.40 (classifier cost).

Why: the misses were vocabulary, and vocabulary is a data file that the
pipeline can re-screen for $0 and re-classify for cents (this round: $1.21 for
387 windows). The reference set is reusable for every future list change at
$0 (`recall_recompute.py`). A recall of ~0.8 on a known sample is a usable
instrument for a hierarchical rate model (it is a binomial thinning of the
count with a measured, uncertain rate), while an unmeasured 1.0 is not.

Cost: this round's $1.21 Claude + GPU for the newly located episodes (a few
$0.0x). Each future expansion: the same. The caption-route recall (Santa
Clara / Hayward / YouTube bodies) is still unmeasured — the planned
3-meeting auto-caption arm (~$2) closes that.

Risk: recall is bounded by the vocabulary a human thinks of. The 7 classifier
misses (mostly transit `mention_only`) are untouched by this option; they are
a codebook question, not a #13 question.

## Option 2 — whole-meeting ASR-only GPU pass as the universal locate track

Amend #13 to:

> **13 (alt).** Every meeting is transcribed whole, **once, without
> diarization**, on the GPU (WhisperX large-v3, the b35 configuration); that
> transcript is the locate track for every body, replacing captions, Swagit
> and Mac ASR as locate inputs. The keyword screen + classifier run on it;
> diarized WhisperX then runs on the located episodes as now.

Measured cost (b35 rates): GPU **≈ $23** for 2,284 h ASR-only (**needs a
RunPod top-up**: $3.97 of credit left) plus the diarized episode pass as now;
Claude unchanged if the screen stays (≈ $28 for the whole corpus at
$0.012/h, mostly already spent), or **≈ $190** if the screen is dropped and
every minute is labelled (the only configuration whose recall is 1.0 by
construction against the codebook — and it is over `CLAUDE_CAP_USD` $80).
Downloads: the whole audio of every YouTube meeting — ~63 GB, **one request
per meeting at the 12-per-day cap ⇒ ≥ 32 days** of trickle on this Mac (the
two 429s this month were at 14–18 requests/day). Granicus HLS is 410 MB per
meeting-hour on the wire. Mac-only alternative: ~197 Mac-hours of mlx-whisper.

What it buys: one locate instrument across all routes (today four:
`classifier`, `local_asr`, `agenda_index`, `swagit_transcript`, which
T2-PLAN forbids pooling), and the whole-meeting text as a by-product (context
for every episode, the "every meeting counted" denominator checked by ear).
What it does **not** buy: recall. On the GPU text the current screen's recall
was 0.826 (any) — the same list, the same misses. Option 2 only beats
Option 1 if the screen is dropped ($190) or the classifier is run on the
screen's neighbours.

## Recommendation

**Option 1 now.** It is the measured fix for the measured failure, it costs
~$1.21 + cents of GPU, and it leaves the exhaustive reference reusable. Run
the auto-caption recall arm (3 meetings, ~$2) so the caption route has its own
number before any rate is reported. Keep Option 2 as the fallback **only if**
post-expansion pipeline recall stays below 0.80 on either route, and then
budget it honestly: a RunPod top-up and a month of YouTube trickle.

Things Puck should decide, not the agent: the 0.80 floor; whether recall
enters the model as a fixed thinning rate or as a parameter with the bootstrap
interval as its prior; whether the 7 classifier misses (`mention_only` on
substantive transit public comment) warrant a codebook re-coding (that is
#14/#17 territory and re-bills every window).

## Implementation note, 2026-10-18 (Puck chose Option 2 for the Granicus / Swagit half, WhatsApp 10:13 PDT)

Puck approved a full-ASR locate run for the non-YouTube bodies (Granicus:
richmond, sanleandro, santaclara, hayward, napa, oakland, fremont, sausalito,
samtrans; Swagit: petaluma) under codebook **v2.1**, no keyword screen
(`scripts/fullasr.py`; results and the scope table in
`plan/FULLASR-GRANICUS.md`). What that adds to the schema and the route list,
so the wording of #13 can be settled later with the numbers in hand:

- **`episodes.method = 'fullasr'`** is a fifth locate route (SCHEMA.md): the
  whole meeting transcribed once on the GPU without diarization (the b35
  validation configuration, outputs under `output/fullasr/whisper/`, never
  `output/whisper/`), every 120-s window (± 30 s pad) read by Sonnet under
  `prompt_version v2.1-<hash>/claude-sonnet-4-6/mr120`, adjacent about_*
  windows merged exactly as `classify.merge_episodes` does. Episode ids are
  `<meeting>_f<n>` (located) and `<meeting>_fm<n>` (mention), so a meeting
  can carry `_e`, `_a` and `_f` rows side by side and nothing pools them.
- **Recall is 1.0 by construction against the codebook** on this route (no
  screen), so the measurement-error term for it is the classifier's own
  precision / window-granularity error, not vocabulary (PILOT-FULLASR, PILOT
  §9). The YouTube half keeps the screened route for now (its recall is
  measured per route, option 1), so the two halves are two instruments and
  are reported as such.
- **Transcripts already paid for are kept**: a fullasr span overlapping a
  cut / queued / transcribing / done clip of any route is trimmed to its
  largest uncovered piece ≥ 60 s or dropped (the events row names the
  covering clips); only genuinely new spans are cut, offline, from the
  whole-meeting audio on disk. No v1 row is edited.
- `classifier_calls` rows for this route sit under `video_id = 'fullasr:<meeting>'`
  so the screened route's bounds-matching cache can never reuse a label read
  from a different transcript of the same minutes.
