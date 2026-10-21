# Proposed amendment to DECISIONS #14 (the classifier instrument) — for Puck, 2026-10-18

Written by the supervisor agent after Puck's adjudication of the codebook v2
pilot (WhatsApp, 2026-10-18 09:33 PDT, "yes to all"). **Nothing in
`plan/DECISIONS.md` has been edited**; this file is the proposal, the
evidence and the re-read plan. Puck fills in the two blanks at the end and the
next session writes the wording into DECISIONS.md.

## What #14 says

> **14. The episode is the unit; it is defined by two keyword families at a
> low density threshold plus a pinned Claude classifier (model + prompt
> version recorded per row), tagged density / transit / both.** *Reopen if:*
> the classifier's recall or precision on the hand-labelled set drops below
> 0.9 / 0.8 on a new body type.

#14 pins *a* classifier and records its version per row; it does not name the
codebook text. The pinned instrument since T0 has been codebook v1
(`t0-a4a8e55760`, `claude-sonnet-4-6`, 240-s max-run; DECISIONS #17). This
amendment changes the pinned text and the window granularity, so it is a new
instrument, not a tweak, and every row must say which one produced it.

## Proposed wording (replaces the instrument clause of #14; #17 stays)

> **14a. From <ADOPTION DATE> the pinned classifier instrument is codebook v2,
> `CODEBOOK_VERSION = v2-ed4c72c9d6`, `claude-sonnet-4-6`, 120-s max-run,
> `prompt_version = v2-ed4c72c9d6/claude-sonnet-4-6/mr120` (`/es`, `/swagit`
> variants as today), through the Message Batches API.** Transit means
> anything not car-related (walking, cycling, micromobility, bus, rail,
> ferry, paratransit; not parking, paving or signals for cars — PILOT §8a);
> if that sentence is added to the text before the re-read it is **v2.1
> with its own hash**, which replaces `v2-ed4c72c9d6` in this clause and is
> what the corpus is read under. Label disagreements are adjudicated by a
> second pinned LLM pass carrying that definition, decision rule recorded
> with the result (§8a), not by hand. Codebook v1 rows
> (`t0-a4a8e55760*`) stay exactly as they are, are never re-labelled in place
> and are **never pooled with v2 rows**: every count, rate and figure is
> reported per `method` and per `prompt_version` (SCHEMA.md, episodes: "n is
> reported per method / prompt_version"), and a figure that shows both shows
> them as separate facets. *Reopen if:* v2's recall or precision against a
> v2-coded reference on a new body type drops below 0.8 / 0.8, or a second
> reference sample (an operator board) contradicts the pilot's transit
> precision.

## Why (plan/CODEBOOK-V2-PILOT.md §4–§9, plan/CODEBOOK-V2-PILOT-FULLASR.md)

v2 = three edits to v1: active transportation and transit service planning
named as transit (edit 1); "one contiguous substantive stretch ≥ ~45 s"
replaces "most of the window" (edit 2); max-run named in `prompt_version`
and set to 120 s (edit 3). Measured on the four Richmond / San Leandro
meetings of the locate-recall check (18.9 audio-h), four screened arms plus a
no-screen arm, $0.57 + $0.77 of Sonnet.

Against the v1-coded reference (109 about-minutes), v2@120 had the best
recall (any 0.780, transit 0.783) but looked less precise (0.720 / 0.415),
with 11 over-call windows all "reference said mention_only". Puck adjudicated
all 11 as about: e-bike / bike-share advocacy as transportation, trails and
pedestrian bridges as circulation, supportive-housing conversion with zoning
as density. With those 18 minutes flipped (127 about-minutes; §9):

| arm | recall any | precision any | recall transit | precision transit | recall density | precision density |
|---|---|---|---|---|---|---|
| v1 @ 240 s (today's rows) | 0.606 | 0.819 | 0.297 | 0.556 | 0.742 | 0.831 |
| v1 @ 120 s (sibling re-screen) | 0.677 | 0.887 | 0.432 | 0.682 | 0.785 | 0.924 |
| **v2 @ 120 s (proposed)** | **0.811** (0.779–0.860) | **0.873** (0.757–0.952) | **0.865** (0.786–1.000) | **0.756** (0.636–0.824) | 0.796 (0.771–0.857) | 0.914 (0.765–0.961) |
| v2 + full ASR, no screen | 0.898 | 0.803 | 0.919 | 0.532 | 0.892 | 0.883 |

v2@120 is the best screened arm on every cell. Against the reference Puck
endorsed, v1's transit recall is 0.30 (0.43 at 120 s): the current instrument
misses most of what Puck counts as transit, and the gap is definitional (edit
1's vocabulary), not noise. The pilot cost $0.244 for 111 v2@120 calls
(**$0.0022 per call**, Batches rate, system block cached). Intervals are
4-cluster bootstraps on 37 transit minutes in two meetings; read them as
rough. The full-ASR arm is still 0.09 better on recall but 3.2× the Sonnet
cost per hour and needs whole-meeting GPU text; it is a separate decision
(DECISIONS-PROPOSED-13).

## What the amendment does NOT change

- `plan/frame.csv`, the body list, the keyword lists (`kw-24e8a4acd1`), the
  screen threshold, the pad, the merge gap, the GPU/WhisperX side, #15–#19.
- Any v1 row: `classifier_calls` (4,434 Sonnet rows, 268 meetings, $12.63),
  `episodes`, transcripts, clips. v1 stays queryable as its own instrument.
- The wave-1 figures already drawn are **re-stamped v1** (instrument in the
  caption / filename), not withdrawn; v2 is drawn as a separate facet when
  the re-read lands (`plan/WAVE1-RESULTS.md`).

## The re-read plan (what adopting v2 costs and does)

**1. Scope.** Every candidate window the pipeline has read under a v1 Sonnet
prompt version is re-scored under v2 via Batches. Counted 2026-10-18 09:40
PDT from `classifier_calls` (read-only; `recall*`/`cbv2:` validation rows and
the 198 Haiku T2-test rows excluded):

| prompt_version | rows | meetings |
|---|---|---|
| `t0-a4a8e55760` (English caption / ASR) | 3,922 | 241 |
| `t0-a4a8e55760/claude-sonnet-4-6/swagit` | 360 | 13 |
| `t0-a4a8e55760/claude-sonnet-4-6/es` | 152 | 16 |
| **total, distinct by span** | **4,433** (4,434 rows) | 268 |

2,917 of them are labelled about_*. Those are 240-s spans. v2 is pinned at
120 s, so the re-read does not re-score these spans; it re-screens each
meeting at 120 s (the pilot produced 1.85× the windows: 111 vs 60) and reads
the new set: **≈ 8,200 windows**. The sibling's v1@120 re-screen, running now,
produces exactly that window set; once it has landed the count is known and
the v2 batch is the same windows under the other codebook.

**2. Cost.** At the pilot's measured $0.0022 per v2@120 call: 4,433 windows
≈ **$9.8**; ≈ 8,200 windows at 120 s ≈ **$18** at the Batches rate. The
pipeline's `estimate_usd` runs ~2× the bill at this size, so expect it to
quote ≈ $36 and refuse nothing (CLAUDE_CAP_USD $10 on `classifier_calls`
**will bind**: $12.63 is already recorded — the cap needs raising for the
re-read, a #18 top-up decision). The §7 extrapolation ($25–30 for the whole
2,100 h) was for a full corpus; this is for the 268 meetings read so far. The
152 Spanish windows need a v2 Spanish instruction block; none exists yet
(`classify.INSTRUCTIONS_V2` is English), so they stay v1 until one is written
and pinned under its own hash.

**3. Episodes.** Re-derived from the v2 labels with the span-matching logic
already in `classify.write_episodes` (RECALL-CHECK "mechanics"): positional
`_c`/`_e` ids renumber when the window set changes, so matching is by core
span, never by id; an episode whose core span already has a row is that row;
a new span whose number is taken gets the next free number; located/mention
rows of the same route that no v2 span matches are marked `superseded`
(events row), not deleted. v2 episodes carry `prompt_version
v2-ed4c72c9d6/claude-sonnet-4-6/mr120` and `keywords_version kw-24e8a4acd1`.
Never re-finish a meeting mid-cut (`--ids=-<id>` for ids starting with `-`).

**4. Transcripts.** Already-transcribed clips (672 `done` episodes, 111.5 h)
are kept where the v2 episode overlaps the v1 one: the v2 row points at the
existing transcript for the overlap, and only the uncovered piece ≥ 60 s is
cut and sent to the GPU (the trim rule that exists today). New cuts only for
spans no v1 episode covers. No audio, transcript or `state.sqlite` row is
deleted (AGENTS.md rule 6). The GPU batch for the new cuts is a normal
`gpu_batch.py propose` → WhatsApp approval → `launch`; size unknown until
the labels land (the pilot's v2 located 118 min vs v1's 94 on 18.9 h, so
expect roughly +25 % episode-hours, mostly transit).

**5. Reporting.** `analysis/load_episodes.py` and the wave-1 figures group by
`prompt_version`; wave-1 outputs are re-stamped `v1` and the v2 facet is
added beside them, not over them. The v2 rate for a body is only reported
once every meeting of that body has been re-read (no mixed-instrument
body-months).

**6. Order.** (a) sibling's v1@120 re-screen finishes and its episodes are
merged; (b) `CLAUDE_CAP_USD` raised; (c) v2 batch over the 120-s window set,
`--codebook v2 --max-run 120 --include-classified --batch`, dry-run first
(the `propose` command has no dry-run; `classify.py run` does); (d) episodes
re-derived; (e) GPU batch for new cuts, approved on WhatsApp; (f) figures.

## What Puck decides

1. **Adoption date** for the wording above (today, or the date the v2 batch
   is submitted — the recommendation is the submission date, so no v2 row
   predates the decision).
2. **Whether the re-read waits for the sibling's 120-s re-screen to finish.**
   It should: the v1@120 run fixes the window set the v2 batch reads, so the
   two instruments are compared on identical spans and nothing is read twice.
3. The `CLAUDE_CAP_USD` top-up for the re-read (≈ $18 expected, ≈ $36 by the
   conservative estimator) — a #18 decision, numbers not changed here.
