# Codebook v2 pilot: can the classifier close the transit gap? (2026-10-18)

**Question.** `plan/RECALL-CHECK.md` left the locate step at recall 0.753
(any about_*) / 0.435 (transit) after the keyword-screen expansion, with
the remaining gap "mostly the classifier's": 11 of 27 misses were candidate
windows the pinned codebook (`t0-a4a8e55760`) labelled `mention_only` /
`no`, nearly all of them the Richmond Bay Trail / bike-share / e-bike
sequence and a San Leandro transit-service-cuts deliberation. This pilot
asks whether three edits to the codebook recover those, at what cost in
precision, and which edit does the work. Approved by Puck, $6 ceiling.

**Status: PILOT ONLY.** The instrument is unchanged. v2 is
`classify.INSTRUCTIONS_V2`, selected with `--codebook v2`; the default is
v1 = `INSTRUCTIONS`, byte-identical, `CODEBOOK_VERSION` unchanged. No
existing `classifier_calls` or `episodes` row was touched; the pilot's
282 calls live under `video_id = 'cbv2:<meeting>'` with their own
`prompt_version`, and its three `classifier_batches` rows carry status
`codebook_v2_pilot` and note `pilot-v2: …`. Nothing committed.

## 1. The three edits (diff v1 -> v2, `output/validation/codebook_v2/codebook_diff.md`)

v2 = `v2-ed4c72c9d6` (7,331 chars; v1 5,578). Every line of v1 not listed
here is byte-identical in v2 (density definition, `about_both`,
`mention_only`, `no`, rules 2, 4-7, OUTPUT).

**Edit 1 — active transportation and transit service planning are named as
transit.** DEFINITIONS "Transit" is split into three named parts. *Public
transportation* keeps v1's list. *Transit service planning is transit*:
service cuts and restorations, budget principles for service, the equity of
service reductions, transit funding and the fiscal cliff, bus stop
placement, visibility and shelters, route and schedule policy. *Active
transportation is transit*: bike lanes, protected bikeways, bicycle
boulevards, bike share and e-bike programs (including an e-bike lending
library), e-scooters and micromobility, multi-use paths and trails *when
discussed as ways to get around* (the Bay Trail, shoreline and creek trails,
greenways), pedestrian bridges and crossings, first-mile/last-mile access,
Complete Streets, active transportation and bicycle/pedestrian plans, Safe
Routes to School, plus v1's bus lanes / BRT / road diets / Vision Zero /
traffic calming / VMT. The `about_transit` label gains seven paraphrased
positive examples, one per kind of missed window in
`output/validation/recall/missed_windows.md` (trail-segment funding
comment; council paying for a trail segment or continuing bike share; a
board setting principles and equity tests for service cuts; a resident on a
bus-stop policy change; a staff item on pedestrian bridges and bike lanes;
public comment on a biking-infrastructure consulting contract; a commenter
supporting an e-bike lending library). Rule 3's exclusions (roads, potholes,
paving, signals, speed limits, parking enforcement; congestion from housing
is density) are kept verbatim, with one sentence added: a trail or park item
is transit only when the trail is discussed as a way to get around, not
recreation, habitat or park maintenance alone.

**Edit 2 — a contiguous substantive stretch replaces "most of the
window".** Rule 1 now reads: a window is about_* when one contiguous
stretch of at least about 45 seconds (about 100 words) is discussion,
presentation, comment or deliberation on the topic, even if the rest is
other business; the stretch does not have to be most of the window; a brief
or list-like reference stays `mention_only` however often the topic words
occur. The `about_density` label text says "a substantive stretch of the
window (rule 1)" instead of "most of the window" (`about_transit` inherits
it via "the same"). The researcher's "minutes of meeting time" test is kept.

**Edit 3 — window granularity is a parameter, named in the instrument.**
`classify.py --max-run` (default 240 s, unchanged) is now part of
`prompt_version_for(...)`: any non-default value appends `/mr<seconds>`,
so a 120-s reading is `v2-ed4c72c9d6/claude-sonnet-4-6/mr120` (and v1 at
120 s is `t0-a4a8e55760/claude-sonnet-4-6/mr120`); the default adds
nothing, so no existing row's version moves. The batch rows' `note` also
records `max_run`. Nothing in the codebook text changes for edit 3.

Verified 2026-10-18 against the task list: all nine named vocabulary items
present (bike lanes, protected bikeways, bike share, e-bikes, multi-use
paths, Bay Trail as transportation, pedestrian bridges, first/last mile,
Complete Streets), service cuts / budget / equity present, rule 3
exclusions present, "most of the window" occurs in v2 only in the sentence
negating it, `/mr120` in the prompt version. No gap needed fixing; the v2
text is exactly what the prior worker wrote.

## 2. Method

Reference: the exhaustive 60-s labels of `plan/RECALL-CHECK.md` (every
minute of the whole-meeting WhisperX transcripts, pinned codebook v1, no
keyword screen; `output/validation/recall/post_kw2/windows.jsonl`), on the
4 of 6 sampled meetings whose GPU transcript passed QA: `gr-richmond-6010`,
`gr-sanleandro-3077`, `-3089`, `-3105` (18.9 audio-h of the 22.6; the two
Richmond meetings with broken audio contribute 0 about-minutes). 109
reference about-minutes: 86 density, 20 transit, 3 both. The reference is
**not** re-labelled; it is v1's reading of 60-s windows.

Four arms, all on the pipeline side only, on the Mac-ASR locate track
exactly as the pipeline reads it (expanded keyword screen `kw-24e8a4acd1`,
0.5 hits/100 w, 3-window smoothing, pad 30 s), `claude-sonnet-4-6`,
Message Batches API, `classify.request_params` byte-for-byte apart from the
system block:

| arm | codebook | max-run | prompt_version | candidates | batch |
|---|---|---|---|---|---|
| v1_240 | v1 | 240 s | `t0-a4a8e55760` | 60 | none: the pipeline's own rows, $0 |
| v1_120 | v1 | 120 s | `t0-a4a8e55760/claude-sonnet-4-6/mr120` | 111 | `msgbatch_01WKezhAWcBG5gu2owvFJusp` |
| v2_120 | v2 | 120 s | `v2-ed4c72c9d6/claude-sonnet-4-6/mr120` | 111 | `msgbatch_014s6qJ4L3qgYd2xjsfQ65DF` |
| v2_240 | v2 | 240 s | `v2-ed4c72c9d6/claude-sonnet-4-6` | 60 | `msgbatch_01UdWeKQ5cDGFSZfL68FFPBh` |

v1_120 isolates edit 3; v2_240 isolates edits 1+2; v2_120 is all three.
Episodes are merged in memory exactly as the pipeline does
(`classify.merge_episodes`, gap 90 s, mixed labels -> `about_both`), never
written to `episodes`. Recall = reference about-minutes inside the arm's
merged **core** spans / reference about-minutes, per family (density =
`about_density` ∪ `about_both`, transit likewise, any, both alone).
Precision = reference minutes (≥ 30 s) inside family-tagged spans whose
reference label is in that family. Intervals: percentile bootstrap of the
ratio of sums over meeting clusters, 4,000 resamples — **4 clusters**, so
read them as rough. Over-call = an arm window labelled about_* all of whose
reference minutes are `mention_only` / `no`. Script
`scripts/codebook_v2_pilot.py` (`estimate` / `submit` / `collect` /
`analyze`), artefacts `output/validation/codebook_v2/`.

**One correction to the baseline.** RECALL-CHECK's post-expansion numbers
(0.753 any / 0.435 transit / 0.843 density) were computed from the
`episodes` table, whose `start_s`/`end_s` are the **padded** bounds (30 s
each side), so every episode edge covered an adjacent reference minute at
exactly the 30-s threshold. Re-scoring those same rows with the pad shaved
off gives 0.696 / 0.434 / 0.774, which matches this pilot's v1_240 arm
(0.697 / 0.435 / 0.775) to the rounding. All four arms here are measured
on core bounds, so they are like-for-like with each other; compare them
with RECALL-CHECK's table only after subtracting that pad effect (~0.06 on
"any").

## 3. Cost

| arm | estimate (batch rate) | worst | actual | requests | errored | instruction flags |
|---|---|---|---|---|---|---|
| v1_120 | $0.423 | $0.554 | **$0.218** | 111 | 0 | 0 |
| v2_120 | $0.503 | $0.654 | **$0.244** | 111 | 0 | 0 |
| v2_240 | $0.312 | $0.393 | **$0.107** | 60 | 0 | 0 |
| total | $1.238 | $1.600 | **$0.569** of the $6 ceiling | 282 | 0 | 0 |

Submitted 15:40 UTC, all three ended by 15:43 UTC. The estimate is ~2x
actual, as in the kw-expansion batch: `estimate_usd` assumes the cache is
written more often than it is at this size. The v2 arms were not more
expensive per window than v1 in practice (+$0.03 on 111 windows for the
31 % longer system block, cached).

## 4. Results

Recall (minutes), precision, per family and arm. 109 about-minutes; CI =
95 % 4-cluster bootstrap.

| family | arm | about-min | located-min | **recall** | 95 % CI | covered ref-min | **precision** | 95 % CI | located total (min) |
|---|---|---|---|---|---|---|---|---|---|
| density | v1_240 | 89 | 69 | 0.775 | 0.727–0.905 | 83 | 0.831 | 0.561–0.955 | 83 |
| density | v1_120 | 89 | 69 | 0.775 | 0.727–0.905 | 79 | 0.873 | 0.708–0.976 | 79 |
| density | **v2_120** | 89 | 70 | 0.787 | 0.765–0.857 | 81 | 0.864 | 0.692–0.957 | 81 |
| density | v2_240 | 89 | 69 | 0.775 | 0.727–0.905 | 87 | 0.793 | 0.531–0.955 | 87 |
| transit | v1_240 | 23 | 10 | 0.435 | 0.000–1.000 | 18 | 0.500 | 0.455–0.571 | 18 |
| transit | v1_120 | 23 | 14 | 0.609 | 0.333–1.000 | 22 | 0.591 | 0.500–0.625 | 22 |
| transit | **v2_120** | 23 | 18 | **0.783** | 0.667–1.000 | 41 | 0.415 | 0.294–0.636 | 41 |
| transit | v2_240 | 23 | 17 | 0.739 | 0.667–1.000 | 49 | 0.327 | 0.182–0.455 | 49 |
| any | v1_240 | 109 | 76 | 0.697 | 0.632–0.769 | 94 | 0.809 | 0.600–0.952 | 94 |
| any | v1_120 | 109 | 80 | 0.734 | 0.688–0.806 | 97 | 0.825 | 0.696–0.953 | 97 |
| any | **v2_120** | 109 | 85 | **0.780** | 0.761–0.846 | 118 | 0.720 | 0.574–0.887 | 118 |
| any | v2_240 | 109 | 83 | 0.761 | 0.727–0.821 | 121 | 0.686 | 0.525–0.917 | 121 |
| both | all arms | 3 | 3 | 1.000 | — | 4–15 | 0.13–0.50 | — | — |

Labels, misses by cause, recovered misses:

| arm | labels (about_d / about_t / both / mention / no) | located spans (min) | misses: screen / classifier | over-calls | recovered of the 30 original | of the 27 post-kw2 |
|---|---|---|---|---|---|---|
| v1_240 | 21 / 3 / 2 / 28 / 6 | 15 (94) | 20 / 13 | 1 | 3 | 0 |
| v1_120 | 38 / 10 / 2 / 50 / 11 | 20 (97) | 20 / 9 | 4 | 8 | 5 |
| **v2_120** | 39 / 21 / 2 / 41 / 8 | 26 (118) | 20 / **4** | 11 | **12** | 9 |
| v2_240 | 21 / 10 / 3 / 20 / 6 | 20 (121) | 20 / 6 | 3 | 10 | 7 |

Per meeting, recall any (about-min): Richmond 6010 (33) 0.64 → 0.73 →
**0.79** → 0.79; San Leandro 3077 (47) 0.74 → 0.74 → 0.77 → 0.74; 3089 (5)
1.00 in all arms; 3105 (24) 0.62 → 0.67 → 0.75 → 0.71 (v1_240 → v1_120 →
v2_120 → v2_240). Transit windows found: Richmond 5 → 8 → 11 → 10 of 15;
3105 0 → 1 → 2 → 2 of 3.

**Reading.**
- The 20 *screen* misses are identical in every arm by construction (the
  minute never became a candidate): 18 density minutes with no
  density keyword (development feasibility, cap rates, prevailing wage,
  vacancy tax, parkland dedication, rent-program staffing) and 2 transit
  minutes of the Bay Trail deliberation with no transit word. No codebook
  reaches them; only a density-vocabulary expansion or no screen would.
- Of the 13 *classifier* misses in v1_240, v2_120 leaves **4**: two
  density minutes v2 still calls mention_only (`3105 3:02` rent-ordinance
  deliberation, `3105 3:14` interim-to-permanent housing), one density
  minute from the Richmond Housing Authority budget (`6010 0:15`,
  `mention_only` in a 120-s window v1_240 labelled about — a granularity
  loss), and `3105 0:07` (bus-stop policy complaint inside a mixed
  public-comment window; all four arms say mention_only).
- Edit 3 alone (v1_120) recovers 5 of the 13 classifier misses: transit
  0.435 → 0.609, any 0.697 → 0.734, with *higher* precision (0.809 →
  0.825). Shorter windows let the pinned codebook see the Bay Trail
  stretch (2:09–2:11) and the e-bike share deliberation (3:09–3:12) on
  their own.
- Edits 1+2 alone (v2_240) do more for transit (0.435 → 0.739) but cost
  precision (any 0.809 → 0.686): 240-s v2 windows labelled about_transit
  drag 4-minute spans over neighbouring reference minutes that are not
  about anything.
- All three (v2_120) is the best recall on every family (any **0.780**,
  transit **0.783**, density 0.787) and recovers **12 of the 30 original
  misses** (all 9 recoverable transit ones; the other 18 are screen
  misses), at precision 0.720 on any and 0.415 on transit.

## 5. Over-calls (v2_120): 11 windows, all "reference said mention_only"

None is a "reference said no": every over-called window has topic content
the reference saw and judged brief. These are the judgment calls for Puck;
the pilot cannot settle them because the reference is v1's own reading.

| meeting, window | v2 label | v2 rationale | reference (60-s, v1) |
|---|---|---|---|
| 6010 0:56–0:58 | about_transit | public comment: e-bike lending library and bike access as affordable transportation | mention_only ×2: road safety / bike-repair nonprofit; e-bike lending briefly |
| 6010 0:58–0:59 | about_transit | public comment supporting e-bike lending library | mention_only: e-bike library, personal announcements, homeless services |
| 6010 1:22–1:24 | about_transit | e-bike lending library as clean mobility / active transportation | mention_only; no |
| 6010 1:24–1:26 | about_transit | e-bike lending library as free community transportation | mention_only ×2: egress comments, e-bike library, consent calendar |
| 6010 1:26–1:27 | about_transit | e-bike library comment, then consent calendar vote | mention_only: consent calendar roll call |
| 6010 1:53–1:55 | about_transit | public comment on bike share and Bay Trail completion | mention_only ×2: budget priorities, bike share and trail briefly |
| 6010 3:06–3:07 | about_transit | trail construction, gap stabilization, bike infrastructure budget | mention_only: budget on a trail and bike rack (v1_120 made the same call) |
| 6010 3:11–3:12 | about_transit | e-bike share program ending for lack of appropriation | mention_only: brief, moves to fee schedule hearing (v1_120 same) |
| 3077 3:33–3:35 | about_transit | proposal for pedestrian bridges across the creek for school safety | mention_only ×2: council procedural reforms, bridge briefly |
| 3105 2:54–2:56 | about_density | staff on housing production, streamlining, fees, rent stabilization | mention_only ×2: economic development report, housing at the end (v1_120 same) |
| 3105 3:04–3:06 | about_density | converting interim housing to permanent supportive, zoning, unit counts | mention_only ×2: interim homeless housing operations (v1_120 same) |

Pattern: 6 of 11 are the **Richmond e-bike lending library public-comment
run** (0:56–1:27), where several residents in turn praise the library as
affordable transportation. The reference itself is split on that run: it
called 1:21 `about_transit` (one of the original 30 misses) and the
neighbouring minutes `mention_only`. v2's definition deliberately names the
e-bike lending library as transit (edit 1), so v2 reads the whole run as
about; if Puck agrees with the 1:21 call, these six are not errors and
v2_120's transit precision is understated. 4 of 11 are shared with v1_120
(the 120-s granularity, not the codebook, produced them). Only
`3077 3:33` (pedestrian bridges inside a procedural item) and `6010 1:53`
look like genuine v2 over-reach under edit 2's 45-s stretch rule.

## 6. Remaining misses (v2_120): 24 of 109 about-minutes

Screen (20, same in all arms): `6010` 2:13, 2:23, 2:24 (Bay Trail
deliberation, no transit word), 3:08 (bike share, no transit word), 3:40,
3:41 (parkland dedication); `3077` 0:51, 1:01, 1:02, 1:21, 1:22, 1:23,
1:28, 1:29 (development feasibility, cap rates, prevailing wage), 2:02,
2:06, 2:07 (vacancy tax); `3105` 3:08, 3:09 (rent-program staffing), 4:19
(construction costs, cap rates). Classifier (4): `6010` 0:15, `3105` 0:07,
3:02, 3:14 (above). Full list with excerpts:
`output/validation/codebook_v2/missed_v2_120.md`.

## 7. Recommendation

1. **Edit 3 is free and safe: adopt `--max-run 120` for the next
   measurement** regardless of the codebook decision. It recovers 5
   classifier misses, raises precision, costs 1.85× the candidate windows
   but only ~$0.22 per 19 audio-h at the batch rate (the system block is
   cached), and is already named in `prompt_version` (`/mr120`).
2. **Edits 1+2 are a codebook change, i.e. a new instrument**, and the
   measurement says they are what closes the transit gap (0.609 → 0.783 at
   120 s; 12 of the 30 original misses recovered, every recoverable transit
   one). The cost is 7 extra transit over-calls per 19 h, 6 of them one
   public-comment run the v1 reference is itself inconsistent on. Before
   adopting: Puck adjudicates the 11 rows in §5 (listen or read
   `output/validation/codebook_v2/windows_v2_120.jsonl`). If ≥ 6 of 11 are
   judged about, v2_120's precision is at least v1's and v2 should become
   the instrument for the transit measurement from the next clean start
   (never mid-measurement, docs/MODELS.md); if not, keep v1 at 120 s and
   accept transit recall ≈ 0.6.
3. Adopting v2 means every meeting is re-read under `v2-ed4c72c9d6/…/mr120`
   — all `local_asr`, `classifier` and Swagit-route rows. Extrapolating
   this pilot ($0.244 per 18.9 h of candidates), a full-corpus re-read is
   roughly $25–30 at the batch rate for the ~2,100 h; a proper estimate is
   `classify.py run --codebook v2 --max-run 120 --include-classified
   --batch --dry-run`.
4. The transit interval is still 0.667–1.000 on 23 about-minutes in 4
   clusters. A decision on v2 as *the* instrument would be better taken
   after the sibling full-ASR arm (`scripts/recall_fullasr_v2.py`, windows
   built, not submitted as of this write-up) or a second reference sample
   with more transit minutes (a BART or VTA meeting).
5. The 20 screen misses (18 density) are untouched by any codebook; the
   next recall gain is a density-vocabulary expansion (feasibility, cap
   rate, prevailing wage, vacancy tax, parkland dedication), which was
   explicitly not approved on 2026-10-18 and is proposed here, not done.

## 8. Files

Changed: `scripts/codebook_v2_pilot.py` (batch note now `pilot-v2: …`).
New: this file; `output/validation/codebook_v2/` — `pilot.json`,
`results.json`, `table.md`, `codebook_diff.md`, `overcalls.md`,
`missed_<arm>.md`, `episodes_<arm>.json`, `windows_<arm>.jsonl`,
`windows_reference_with_arms.jsonl`, `estimate.log`, `submit.log`,
`collect.log`, `analyze.log`. DB: 282 `classifier_calls` rows under
`cbv2:<meeting>`, 3 `classifier_batches` rows (status `codebook_v2_pilot`
→ `collected`); nothing else written. Untouched: `pipeline/classify.py`
v1 text and `CODEBOOK_VERSION`, all pre-existing `classifier_calls` /
`episodes` rows, `.env`, `plan/frame.csv`, `plan/DECISIONS.md`. Nothing
committed; no YouTube contact; no pod.

## 8. Adjudication (Puck, WhatsApp, 2026-10-18 09:33 PDT)

All 11 v2@120 over-call windows in §5 were judged **about** ("yes to all"):
public comment advocating e-bikes / bike share as transportation counts as
transit; active-transportation infrastructure (trails as routes, pedestrian
bridges for circulation) counts as transit; supportive-housing conversion
with zoning changes and production-policy staff reports count as density.
Consequence: the v1-coded reference under-calls on exactly the vocabulary
edit 1 added; v2's transit precision is at least v1's once the reference is
corrected (recompute in §9). v2 is the instrument going forward; this is a
DECISIONS #14 change (codebook text, new CODEBOOK_VERSION), proposed in
plan/DECISIONS-PROPOSED-14.md, old and new rows never pooled.

## 8a. Definition and adjudication rule (Puck, WhatsApp, 2026-10-18 09:36 PDT)

Transit is "anything not car related": walking, cycling, micromobility,
bus, rail, ferry, paratransit all count; car infrastructure (parking,
paving, signals for cars) does not. This goes into the codebook as one
sentence at the top of the transit definition (v2.1, new hash) before the
corpus re-read, so nothing is read twice. Future label disagreements are
adjudicated by a second pinned LLM pass carrying this definition, not by
hand; the decision rule is recorded with the result.

## 9. Recompute against the adjudicated reference ($0, 2026-10-18 09:37 PDT)

**Corrected reference.** The 11 windows of §5 cover **18 reference minutes**
(17 `mention_only`, 1 `no` — `6010 1:23`, the second minute of the 1:22–1:24
window; Puck judged the window, so it is flipped with it). They are flipped to
the label v2 assigned: 14 → `about_transit` (Richmond e-bike library run 0:56–
0:58 and 1:22–1:26, bike share / Bay Trail 1:53–1:54, 3:06, 3:11; San Leandro
3077 pedestrian bridges 3:33–3:34), 4 → `about_density` (3105 2:54–2:55,
3:04–3:05). New file
`output/validation/codebook_v2/adjudicated/reference_adjudicated.jsonl`
(first line = provenance header with no `label` key; flipped rows keep
`label_v1` and carry `adjudicated{by, at, window, v2_label, v2_rationale}`).
The v1-coded reference (`output/validation/recall/post_kw2/windows.jsonl`,
`windows_reference_with_arms.jsonl`) is untouched. About-minutes 109 → **127**
(density 89 → 93, transit 23 → 37, both 3). Script
`scripts/codebook_v2_adjudicate.py`; results
`adjudicated/results_adjudicated.json`, `adjudicated/table_adjudicated.md`.
No API call, no DB access.

**Arms.** The four pilot arms re-scored from their existing per-minute
coverage (`ep_<arm>` / `tags_<arm>`), 4-cluster bootstrap as in §4; the
full-ASR arm from its 23 merged episodes in
`output/validation/codebook_v2_fullasr/results.json`, 6-cluster bootstrap as
in its own doc (two zero clusters). The YouTube auto-caption arm
(`plan/RECALL-CHECK.md`, BART / Los Altos / Milpitas / Cupertino) shares **no
meeting** with these four and none of the 11 windows, so the adjudication
does not touch it; its numbers stand as published. "Before" reproduces §4 and
the full-ASR table to the digit, so the two columns are like-for-like.

| family | arm | recall before → **after** | 95 % CI after | precision before → **after** | 95 % CI after |
|---|---|---|---|---|---|
| density | v1_240 | 0.775 → 0.742 | 0.630–0.905 | 0.831 → 0.831 | 0.561–0.955 |
| density | v1_120 | 0.775 → 0.785 | 0.748–0.905 | 0.873 → 0.924 | 0.781–0.982 |
| density | **v2_120** | 0.787 → **0.796** | 0.771–0.857 | 0.864 → **0.914** | 0.765–0.961 |
| density | v2_240 | 0.775 → 0.742 | 0.630–0.905 | 0.793 → 0.793 | 0.531–0.955 |
| density | v2+full-ASR | 0.910 → 0.892 | 0.794–0.980 | 0.862 → 0.883 | 0.706–0.955 |
| transit | v1_240 | 0.435 → 0.297 | 0.000–0.833 | 0.500 → 0.556 | 0.455–0.714 |
| transit | v1_120 | 0.609 → 0.432 | 0.200–0.882 | 0.591 → 0.682 | 0.500–0.750 |
| transit | **v2_120** | 0.783 → **0.865** | 0.786–1.000 | 0.415 → **0.756** | 0.636–0.824 |
| transit | v2_240 | 0.739 → 0.757 | 0.400–0.882 | 0.327 → 0.551 | 0.182–0.741 |
| transit | v2+full-ASR | 1.000 → 0.919 | 0.714–1.000 | 0.355 → 0.532 | 0.206–0.664 |
| any | v1_240 | 0.697 → 0.606 | 0.507–0.741 | 0.809 → 0.819 | 0.617–0.952 |
| any | v1_120 | 0.734 → 0.677 | 0.601–0.781 | 0.825 → 0.887 | 0.761–0.973 |
| any | **v2_120** | 0.780 → **0.811** | 0.779–0.860 | 0.720 → **0.873** | 0.757–0.952 |
| any | v2_240 | 0.761 → 0.740 | 0.647–0.840 | 0.686 → 0.777 | 0.564–0.934 |
| any | v2+full-ASR | 0.927 → 0.898 | 0.836–0.966 | 0.711 → 0.803 | 0.688–0.917 |
| both | all arms | 1.000 → 1.000 | — | unchanged (0.13–0.50) | — |

Located minutes with no about_* reference minute under them, before → after:
v1_240 18 → 17, v1_120 17 → 11, **v2_120 33 → 15**, v2_240 38 → 27,
full-ASR 41 → 28 (the remainder are span edges over minutes the reference
still calls mention/no, not whole windows).

**Reading.**
- Under the reference Puck endorsed, **v2@120 is the best arm on every
  family for both recall and precision among the screened arms**: any 0.811 /
  0.873, transit 0.865 / 0.756, density 0.796 / 0.914. Its "any" recall
  interval (0.779–0.860) now sits at the 0.80 floor of
  `plan/DECISIONS-PROPOSED-13.md` — the lower end is just under it, on 4
  clusters.
- The v1 arms **lose recall** (any 0.697 → 0.606 at 240 s; 0.734 → 0.677 at
  120 s) because the 18 flipped minutes are exactly what v1 does not call
  about: the v1 reference under-called its own codebook's blind spot. v1's
  transit recall is 0.30–0.43 against a v2-coded reference.
- v2's transit precision was understated by the v1 reference by ~0.34
  (0.415 → 0.756), as §5 predicted; what is left (15 edge minutes) is the
  120-s window's granularity, not the codebook.
- The full-ASR arm stays the recall leader (0.898 any) but its advantage over
  v2@120 shrinks from 0.147 to 0.087 and its precision stays lower (0.803 vs
  0.873), at 3.2× the Sonnet cost per audio-hour (FULLASR cost table). The
  screen's 20 vocabulary misses (§6) remain the gap between the two.
- Caveats: 4 clusters, 37 transit about-minutes in 2 meetings; the 18 flips
  were adjudicated from v2's rationales and the reference excerpts, not by
  re-listening; the adjudicated minutes are, by construction, the ones v2
  called — a reference coded under v2 from scratch (a second sample, BART or
  VTA) is still the clean test. §8a's one-sentence v2.1 addition ("anything
  not car related") is **not** what was measured here; it should be checked
  on these 111 windows (≈ $0.25) before the corpus re-read.

**Quoting rule (*corrected 2026-10-18 after validation*).** The table above is
like-for-like column by column, but figures from *different* columns must not
be paired. **Never quote "0.44 → 0.87"**: 0.435 is v1@240 against the v1-coded
reference and 0.865 is v2@120 against the adjudicated one, so the pair mixes
references and roughly doubles the apparent gain. The like-for-like pairs are
**0.435 → 0.783** (v1 reference) and **0.297 → 0.865** (adjudicated). The
full-ASR arm is 0.927 on the v1 reference and **0.898** on the adjudicated one.

**Circularity, quantified (*added 2026-10-18 after validation*).** The
adjudicated reference is not independent of the arm it scores: all 18 flipped
minutes are v2@120's own over-calls, so v2@120 hits **18 of 18 by
construction** (v1@240 hits 1, v1@120 6, v2@240 11). Scored on the 109
*untouched* reference minutes only, v2@120 gets any **0.780**, transit
**18/23 = 0.783**, density **0.787** — its v1-reference row to the digit, i.e.
the whole 0.783 → 0.865 transit lift comes from minutes v2 itself defined. The
transit reference grew 23 → 37 minutes with **38 % of it from v2's output**,
and the transit precision jump 0.415 → 0.756 is the same reflip from the other
side. The mechanism was disclosed in the caveat above; what is new is that no
quoted figure carried it. Excluding the self-defined minutes, the measured case
for v2 over v1@120 on transit is **0.783 vs 0.609**, not 0.865 vs 0.432.

## 10. v2.1 frozen and checked on the 111 pilot windows ($0.37, 2026-10-18 10:30 PDT)

**Freeze.** `classify.INSTRUCTIONS_V21` = v2 with one sentence inserted at the
top of the transit definition — Puck's rule verbatim: *"Transit is anything not
car related: walking, cycling, micromobility, bus, rail, ferry and paratransit
all count; car infrastructure (parking, paving, signals for cars) does not."* —
and nothing else (asserted byte-identical after removing the sentence; 7,331 →
7,510 chars). `CODEBOOK_VERSIONS['v2.1'] = v2.1-9e052492e4`; v2 keeps
`v2-ed4c72c9d6`, v1 `t0-a4a8e55760`; `--codebook v2.1`. A Spanish block for
v2.1 exists (`classify.LANG_NOTE_V21['es']`, selected per codebook by
`classify.lang_note`): the v1 interpreter note plus one sentence saying the
transit rule applies to the interpreted Spanish; prompt_version
`v2.1-9e052492e4/<model>/es[/mr<n>]`. v1 and v2 Spanish windows keep
`LANG_NOTE` byte for byte.

**Check** (`scripts/codebook_v21_check.py`, `output/validation/codebook_v2/v21_check/`).
The window set was held fixed at the pilot's 111 v2@120 windows (ids and bounds
from `windows_v2_120.jsonl`, text rebuilt from the same Mac-ASR track with the
same ± 30 s pad): the keyword list changed after the pilot
(`keywords_density.txt`, 09:40 PDT; a fresh screen now gives 126 windows), and
the question is whether the *sentence* moves labels. One Message Batch,
`msgbatch_01FezN6sqBX56LaHr4uJDtwW`, 111/111 succeeded, **$0.3748** (the pilot's
$0.24 had a warm cache; this one wrote it). Rows under `cbv21:<meeting>`,
`prompt_version v2.1-9e052492e4/claude-sonnet-4-6/mr120`.

**Result: the labels did NOT move only toward the adjudicated reference.**

| | v2@120 (pilot) | v2.1@120 (same windows) |
|---|---|---|
| labels | mention 41 · density 39 · transit 21 · no 8 · both 2 | mention 41 · density 38 · transit 21 · no 9 · both 2 |
| changed windows | — | **6 of 111**: 5 scored *away* from the adjudicated reference, 1 neutral, 0 toward |
| recall any / transit / density | 0.811 / 0.865 / 0.796 | 0.780 (0.714–0.840) / 0.838 (0.778–1.000) / 0.763 (0.704–0.857) |
| precision any / transit / density | 0.873 / 0.756 / 0.914 | 0.861 (0.712–0.950) / 0.732 (0.636–0.800) / 0.910 (0.697–0.964) |

The six changes (`v21_check/table.md`): `6010 0:58` e-bike library
about_transit → mention_only (the reference minute Puck adjudicated *about*);
`6010 1:47` trail-gap construction mention_only → about_transit (reference
`no`, v1-coded, never adjudicated — under §8a "trails as routes" this is
arguably the sentence doing its job, and it still scores *away* because the
reference is v1 there); `3089 1:22` downtown housing mention → about_density;
`3105 2:30` and `2:54` about_density → mention_only (rent stabilisation /
housing production in a budget and economic-development report); `3089 1:12`
mention → no. **Four of the six are density flips the transit sentence cannot
have caused**, and the reference is anchored on v2's own labels wherever it
was adjudicated (REVIEW finding 2), so "away" is partly by construction.
The read is run-to-run noise of the instrument at roughly 5 % of windows, not
a systematic effect of the sentence; a replicate of **v2 itself** on the same
111 windows (`--codebook v2`, rows `cbv2rep:`) was submitted to measure that
self-agreement directly and is reported below when collected.

**Consequence.** v2.1 is the instrument for the Granicus / Swagit full-ASR
run (Puck, 10:13 PDT) — the rule had to be in the text before the corpus was
read, and the check shows no evidence that the sentence degrades either
family beyond the replicate noise. But the §9 numbers are v2's, not v2.1's,
and the recall/precision above should be quoted with the replicate's
self-agreement next to them. The one thing the check does establish is that
a label on a single 120-s window is not stable to ± 1 in 20 re-reads: any
per-window claim needs k-sample voting, which the pipeline does not do.

**Replicate (v2 on v2, same 111 windows; `msgbatch_01Hefgc4b24bZPBTLYvfhMHD`,
$0.2838, rows `cbv2rep:`, `output/validation/codebook_v2/v2_replicate/`;
the script's table labels the replicate arm `v21_120` — read it as v2-rep).**
Re-reading codebook v2 itself changed **6 of 111 labels** too (3 about_density →
mention_only, 2 mention_only → about_transit, 1 mention_only → no; 4 scored
*away*, 1 *toward*, 1 neutral); recall any 0.811 → 0.787 (0.737–0.840),
transit 0.865 → 0.892, density 0.796 → 0.753; precision any 0.873 → 0.870.
So v2.1's 6 changed windows and its −0.03 on "any" recall are exactly the
instrument's self-disagreement (≈ 5 % of 120-s windows, concentrated on
density windows near the mention/about line), not an effect of the
sentence. **v2.1 is cleared for the corpus read on that basis**, with the
caveat stated as: v2.1's recall/precision on this sample are v2's ± the
replicate spread (any recall 0.78–0.81, transit 0.84–0.89, density
0.75–0.80).
