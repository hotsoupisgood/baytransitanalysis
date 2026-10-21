# Codebook v2 + full ASR, no keyword screen: pilot arm (2026-10-18)

**Question.** If the locate step read EVERY window of a whole-meeting WhisperX
transcript with codebook v2 (no keyword screen at all), what recall and
precision would it get against the exhaustive reference of
`plan/RECALL-CHECK.md`, and what would Sonnet cost per audio-hour at that
rate over the 2,100-h corpus? This is the "both of the above" arm: the
sibling pilot (`plan/CODEBOOK-V2-PILOT.md`, `output/validation/codebook_v2/`)
measures codebook v2 behind the keyword screen; this arm removes the screen.

Script `scripts/recall_fullasr_v2.py` (`windows` / `submit` / `collect` /
`analyze`); artefacts under `output/validation/codebook_v2_fullasr/`. Ceiling
$4 (default), actual **$0.77**. Nothing committed.

## Method

1. **Transcripts.** The six whole-meeting WhisperX `large-v3` transcripts of
   the locate-recall sample (`output/validation/recall/whisper/`, batch b35,
   22.6 audio-h). Two are excluded: `gr-richmond-6033` (2.15 h) and
   `gr-richmond-6038` (1.57 h) failed GPU QA with decoder loops (coverage 5 %
   of duration, 943 and 743 words for the whole meeting). Their text is not
   usable for labelling and the reference has 0 about-minutes on them, so
   they were not sent; the arm covers **4 meetings, 18.91 audio-h**.
2. **Windows.** Every 120-s window of each transcript (v2's granularity,
   `--max-run 120`), text padded ±30 s exactly as `classify.candidates` pads
   (`DEFAULTS['pad']`), via `classify.window_text`. 570 windows; 549 with
   ≥ 10 padded words were sent (the screen's `min_words` rule treats the rest
   as no speech). No keyword screen: `hits_per_100w` NULL, `family_rates` {}.
3. **Codebook.** `classify.INSTRUCTIONS_V2` as pinned by the sibling pilot,
   **`v2-ed4c72c9d6`** (7,331 chars, ~2,014 tokens), read through
   `classify.request_params(..., codebook='v2')` so the request is
   byte-identical to the pipeline's; `prompt_version`
   **`v2-ed4c72c9d6/claude-sonnet-4-6/mr120`** (`classify.prompt_version_for`
   with `max_run=120`). The v1 codebook, `CODEBOOK_VERSION` and every
   existing `classifier_calls` / `episodes` row are untouched. The sibling had
   already submitted its own v2 batches under this hash (15:40 UTC) before
   this arm submitted (15:41 UTC), which pins the text harder than its doc
   would; the doc did not exist yet at submission.
4. **One Message Batch**, `msgbatch_01BBtcuGSe1yz51jsXaXZfh9`, 549 requests,
   `claude-sonnet-4-6`, structured output, batch row status `pilot_fullasr`,
   note `pilot-fullasr`; call rows under `video_id = 'recallv2:<meeting>'`
   (so `classify.py report` and the per-meeting $ SELECTs ignore them;
   `spend.claude_total` counts them). 549 / 549 succeeded, 0 errored, 0
   windows flagged `contains_instructions`.
5. **Episodes.** Adjacent about_* windows merged with `classify.merge_episodes`
   (gap 90 s, core bounds, mixed labels → about_both) — the pipeline's own
   function, in memory only; no `episodes` row written. 23 episodes, 142 min.
6. **Scoring** against the exhaustive reference (codebook v1, 60-s windows,
   same transcripts, `output/validation/recall/windows.jsonl`; 109
   about-minutes). Recall = reference about-minutes inside v2 episodes over
   reference about-minutes (minute-overlap, as RECALL-CHECK). Precision =
   reference minutes with ≥ 30 s inside family-tagged v2 episodes whose
   reference label is in the family, over those minutes; also window-level
   precision (v2 about-windows with at least one reference about-minute of
   the family). 95 % intervals: meeting-cluster percentile bootstrap, 4,000
   resamples, the same 6 clusters as RECALL-CHECK (the two excluded meetings
   are zero clusters), so read them as rough.

## Results

### Recall / precision per family (v2 + full ASR)

| family | ref about-min | v2 episode-min | recall (min) | 95 % CI | precision (min) | 95 % CI | v2 about-windows | window precision | 95 % CI |
|---|---|---|---|---|---|---|---|---|---|
| density | 89 | 94 | **0.910** | 0.800–0.980 | 0.862 | 0.647–0.955 | 46 | 0.978 | 0.923–1.000 |
| transit | 23 | 62 | **1.000** | 1.000–1.000 | 0.355 | 0.150–0.556 | 28 | 0.679 | 0.429–0.889 |
| any about_* | 109 | 142 | **0.927** | 0.877–0.982 | 0.711 | 0.562–0.898 | 71 | 0.873 | 0.786–0.970 |
| about_both | 3 | 14 | 1.000 | 1.000–1.000 | 0.143 | 0.000–0.500 | 3 | 0.667 | — |

Per meeting (recall, any): `gr-richmond-6010` 0.88 (33 about-min, 11
episodes / 52 min), `gr-sanleandro-3077` 0.98 (47; 4 / 50 min),
`gr-sanleandro-3089` 1.00 (5; 2 / 8 min), `gr-sanleandro-3105` 0.88 (24;
6 / 32 min). Labels over the 549 windows: no 381, mention_only 97,
about_density 43, about_transit 25, about_both 3.

### The four arms side by side

v1+screen = the pipeline as it runs today (`plan/RECALL-CHECK.md`, post
keyword-expansion, `output/validation/recall/post_kw2/`); v2+screen and
v1@120s from the sibling's `output/validation/codebook_v2/table.md` (arms
`v2_120`, `v1_120`; its `plan/CODEBOOK-V2-PILOT.md` was not yet written when
this was compiled). Precision for v1+screen is recomputed here from the
current `episodes` rows of the 4 meetings against the same reference
(`method='local_asr'`, status ≠ mention); RECALL-CHECK reports screen
precision, not episode precision.

| family | metric | v1+screen (current) | v1 @ 120 s + screen | v2 @ 120 s + screen | **v2 + full ASR** |
|---|---|---|---|---|---|
| density | recall | 0.843 (0.822–0.913) | 0.775 (0.727–0.905) | 0.787 (0.765–0.857) | **0.910 (0.800–0.980)** |
| density | precision | 0.789 (0.561–0.889) | 0.873 (0.708–0.976) | 0.864 (0.692–0.957) | 0.862 (0.647–0.955) |
| transit | recall | 0.435 (0.000–1.000) | 0.609 (0.333–1.000) | 0.783 (0.667–1.000) | **1.000 (1.000–1.000)** |
| transit | precision | 0.476 (0.455–0.500) | 0.591 (0.500–0.625) | 0.415 (0.294–0.636) | 0.355 (0.150–0.556) |
| any | recall | 0.753 (0.654–0.852) | 0.734 (0.688–0.806) | 0.780 (0.761–0.846) | **0.927 (0.877–0.982)** |
| any | precision | 0.761 (0.587–0.871) | 0.825 (0.696–0.953) | 0.720 (0.574–0.887) | 0.711 (0.562–0.898) |
| — | located min (4 meetings) | 109 | 97 | 118 | 142 |
| — | Sonnet $ (4 meetings) | $0.218 (65 windows, cumulative) | $0.218 (111) | $0.244 (111) | **$0.770 (549)** |
| — | Sonnet $ / audio-h | $0.0115 | $0.0115 | $0.0129 | **$0.0407** |

(The sibling's v1 pipeline reconstruction, arm `v1_240`, scores recall 0.697
on the same reference because it re-merges from the 60 current candidate
windows; the 0.753 column above is RECALL-CHECK's number from the actual
`episodes` rows and is the one to quote as "current".)

**Read.** Removing the screen is what moves recall: any about_* 0.753 →
0.927, with the interval's lower end (0.877) above the 0.80 floor proposed
in `plan/DECISIONS-PROPOSED-13.md` — the first arm where that holds. Transit
reaches every one of the 23 reference minutes. The price is precision on
transit: v2 locates 62 transit-tagged minutes for 23 reference ones. The
over-call table below shows why: all 9 over-called windows are things
codebook v2 *defines* as transit that codebook v1 (the reference) called
mention_only — the e-bike lending library, bike share funding, pedestrian
bridges and bike lanes for school access — so most of the transit
"imprecision" is the definitional change of edit 1, not noise, and the
reference (v1) cannot adjudicate it. A v2-coded reference would be needed to
measure v2's transit precision properly. Density is a cleaner read: recall
0.843 → 0.910 with precision flat (0.79 → 0.86), and 0.978 of the 46
density windows contain a reference density minute.

### Over-calls: v2 about-windows whose reference minutes hold no about_* label (9, all `mention_only`)

| meeting | window | v2 label | reference (2 min) | v2 rationale |
|---|---|---|---|---|
| gr-richmond-6010 | 0:56:00 | about_transit | mention_only / mention_only | public commenter advocates e-bike lending library as affordable transportation |
| gr-richmond-6010 | 0:58:00 | about_transit | mention_only / mention_only | public comment supporting e-bike lending library as affordable transportation |
| gr-richmond-6010 | 1:22:00 | about_transit | mention_only / no | public comment supporting e-bike lending library as clean mobility / active transportation |
| gr-richmond-6010 | 1:24:00 | about_transit | mention_only / mention_only | public commenter advocates e-bike lending library as free transportation |
| gr-richmond-6010 | 1:26:00 | about_transit | mention_only / mention_only | e-bike lending library support; Safe Routes project also mentioned |
| gr-richmond-6010 | 3:10:00 | about_transit | mention_only / mention_only | council deliberating bike share funding lapse and e-bike library appropriation |
| gr-sanleandro-3077 | 3:34:00 | about_transit | mention_only / no | public comment proposing a new pedestrian bridge for school safety and access |
| gr-sanleandro-3105 | 3:04:00 | about_density | mention_only / mention_only | permanent supportive housing conversion, affordable units, zoning changes |
| gr-sanleandro-3105 | 4:14:00 | about_transit | mention_only / no | pedestrian bridges and bike lanes for active-transportation access to school |

8 of 9 are transit, and 8 of 9 are the vocabulary edit 1 added (e-bike
library, bike share, pedestrian bridges, bike lanes). The one density
over-call (supportive-housing conversion with zoning changes) reads as a
genuine about_density under either codebook's rule 2 and is arguably a
reference miss.

### Misses: reference about-minutes with < 30 s inside any v2 episode (8, all density; v2 said `mention_only` on the covering window)

`gr-richmond-6010` 0:15 and 0:22 (housing-authority operating budget / HUD
funding), 3:40–3:41 (parkland dedication fees on a development);
`gr-sanleandro-3077` 2:02 (vacancy tax); `gr-sanleandro-3105` 0:12 (rent
ordinance, ADU in a list), 0:23 (tenant protections), 2:31 (budget line for
tenant/landlord services). These are the same "density-vocabulary" minutes
RECALL-CHECK's screen missed (feasibility, vacancy tax, parkland
dedication); read in 120-s windows under v2's contiguous-stretch rule the
classifier calls them mention_only. Whether they are density at all is a
codebook question (rent/tenant policy is in v2's definition "when discussed
as housing policy"; a housing-authority budget and a parkland fee are not),
so these are the windows to put in front of Puck, not a bug. Full list with
excerpts: `output/validation/codebook_v2_fullasr/overcalls_and_misses.md`.

## Cost

| | value |
|---|---|
| estimate (classify.estimate_usd, batch rate) | $1.51 (worst case $3.27) |
| **actual** | **$0.7696** |
| windows sent / audio-h | 549 / 18.91 h = 29.0 windows per audio-h |
| input tokens (window text + wrapper) | 306,577 = 16,213 per audio-h (1.34 tokens per padded word) |
| cache-read tokens (codebook) | 1,045,845 = 55,308 per audio-h (1,905 per call: every call hit the cache) |
| cache-write tokens | 0 (the sibling's v2 batch had just warmed the same prefix) |
| output tokens | 20,380 = 1,078 per audio-h (37 per call) |
| **Sonnet $ per audio-h** | **$0.0407** (Batches rate) |
| **2,100 h corpus** | **≈ $85** at the Batches rate (≈ $171 direct) |

Why the estimate is 2x the bill: `classify.estimate_usd` assumes a 75 %
cache-hit share and 2.04 tokens per word (calibrated on YouTube captions);
this batch hit the cache on 100 % of calls and WhisperX text bills 1.34
tokens per word. Both biases are conservative, as intended. For the corpus
projection the pessimistic end (estimate rather than bill) is $170.

For comparison the pipeline's screened v1 classifier cost $0.0115 per
audio-h on the same four meetings (65 windows, cumulative over its re-reads),
so dropping the screen multiplies the Sonnet locate cost by ~3.5x — to about
$85 for the whole corpus, i.e. roughly one GPU batch. What this projection
does NOT include: the whole-meeting GPU transcript that full-ASR locate
needs for every meeting. Batch b35 transcribed 22.64 h without diarization
for $0.2313 (RECALL-CHECK), $0.0102 per audio-h → ≈ $21 for 2,100 h
without diarization, ~3x that with it; the corpus transcripts for bodies on
the `local_asr` route (the Mac mlx-whisper track) already exist and the
question of whether that track is good enough for full-ASR locate is a
separate measurement (this arm read the GPU text).

## What to decide

1. Full-ASR locate with v2 is the only arm whose "any" recall interval
   clears the proposed 0.80 floor, at ≈ $85 of Sonnet for the corpus.
2. Its transit precision cannot be read off a v1 reference: the 9 over-calls
   are exactly edit 1's new transit vocabulary. If v2's definition of transit
   is what Puck wants (active transportation counts), the over-calls are
   correct and transit precision is not 0.355; if not, edit 1 should be
   narrowed. Either way the 8 density misses are the windows to hand-code.
3. The two Richmond meetings with broken GPU transcripts are still
   unmeasured under any arm; the audio or Richmond's encode is the first
   suspect (RECALL-CHECK).

## Files

New: `scripts/recall_fullasr_v2.py` (started by the previous worker, finished
here: the two QA-failed meetings are excluded and the cost base is the 4
meetings sent); `plan/CODEBOOK-V2-PILOT-FULLASR.md` (this);
`output/validation/codebook_v2_fullasr/{windows.jsonl, estimate.json,
batch.json, results.json, table.md, overcalls_and_misses.md}`. DB: 1
`classifier_batches` row (status `collected`, note `pilot-fullasr`) and 549
`classifier_calls` rows under `recallv2:*` / `v2-ed4c72c9d6/claude-sonnet-4-6/mr120`;
no `episodes` rows. Not touched: `pipeline/classify.py` (the sibling's),
`plan/frame.csv`, `plan/DECISIONS.md`, `.env`, every v1 row. Nothing
committed.
