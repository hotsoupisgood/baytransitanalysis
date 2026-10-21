# Locate-recall check: what the captions-first locate step misses (2026-10-17)

**Question.** DECISIONS #13 transcribes only the episodes the locate step
finds: the `segment.py` keyword screen (0.5 hits / 100 words, 60-s windows,
3-window smoothing) on a caption-equivalent track, then the pinned Sonnet
classifier. On Richmond and San Leandro that track is a Mac ASR transcript
(`captions='asr_local'`, `method='local_asr'`; T2-PLAN §3d), and §3d said its
recall was unmeasured and "22 Mac-hours are riding on it". This check
measures it directly on meetings whose whole-meeting audio is already on
disk: transcribe the WHOLE meeting on the GPU, label EVERY minute with the
same codebook, and ask what fraction of the "about" minutes the pipeline
located.

Pre-approved by Puck: ceiling $7 (GPU + Claude). Script
`scripts/recall_check.py`; artefacts under `output/validation/recall/`.
Nothing here is in the corpus: the GPU batch was launched with `WHISPER_DIR`
and `BATCH_DIR` pointed at `output/validation/recall/`, the six `videos` rows
are `kind='validation'`, `in_frame=0`, ids `rc-<meeting>`, and the classifier
rows are `video_id='recall:<meeting>'` (so `classify.py report` and every
per-meeting $ SELECT ignore them; `spend.claude_total` still counts them).

## Method

1. **Sample.** Seeded draw (seed 20261017, 13th draw satisfied the rule) of
   3 Richmond + 3 San Leandro regular meetings from the 58 `asr_local`
   meetings whose `local_asr.py` wav is on disk, constrained to 18–23 audio-h
   in total, at least one meeting with 0 located episodes and at least four
   with some. `output/validation/recall/sample.json`.
2. **Whole-meeting GPU transcript.** The same wav the locate track came from,
   transcoded exactly as the corpus is (`ffmpeg -ac 1 -ar 16000 -c:a libopus
   -b:a 16k`, no `-t` cap), through `gpu_batch.py` / `gpu_job.py` unchanged
   except for two additions: a manifest entry may say `diarize: false`
   (pyannote skipped; ASR and alignment — the text — are identical to the
   corpus configuration: WhisperX `large-v3`, `condition_on_previous_text
   =False`, `batch_size 16`, float16, the body's `initial_prompt`), and
   `build_manifest` passes a `kind='validation'` row through with its
   `raw_meta` flags. Batch **b35**, pod on the usual A40 SECURE at $0.49/h.
3. **Exhaustive labels.** Every 60-s window of each GPU transcript, text
   padded ±30 s exactly as `classify.candidates` pads a candidate, sent to
   `claude-sonnet-4-6` with `classify.request_params` (same codebook, same
   JSON schema, same `prompt_version` `t0-a4a8e55760` as the `local_asr`
   route) through the Message Batches API. Windows with fewer than 10 padded
   words are not sent (the screen's own `min_words` rule treats them as no
   speech). No keyword screen.
4. **Recall.** For each meeting and family — density (`about_density` ∪
   `about_both`), transit (`about_transit` ∪ `about_both`), any `about_*`,
   and `about_both` alone — located minutes = overlap of the pipeline's
   `episodes` rows (`method='local_asr'`, every status but `mention`, core
   bounds `start_s`–`end_s`) with the exhaustive about-windows, over
   exhaustive about-minutes. Also: tag-matched recall (only episodes whose
   `tags` fall in the family), window-level recall (a window counts as found
   when ≥ 30 s of it is inside an episode), and the keyword screen alone:
   recall = share of about-windows the screen marked on the Mac-ASR track;
   precision = share of marked windows that are about anything. The screen
   is also re-run on the GPU text itself to separate "the words were not on
   the locate track" from "the screen does not fire on them". 95 % intervals
   are a meeting-cluster percentile bootstrap (4,000 resamples of the 6
   meetings) of the ratio of sums — six clusters, so read them as rough.
5. **Misses.** Every exhaustive about-window with < 30 s inside an episode is
   listed with its cause: *screen* (the screen never marked that minute on
   the locate track), *classifier* (it was a candidate and Sonnet said
   `mention_only` / `no`), *other*.

## Results (2026-10-18)

**Cost.** GPU: batch b35, 22.64 audio-h, no diarization, **$0.2313** actual
(28.3 GPU-min) vs **$0.75** estimated — the estimate assumes diarization,
which this job skips, so it over-predicts by ~3x. Claude: 1,115 windows (of
1,359; 244 had < 10 padded words and were not sent), **$1.8737** actual vs
$2.27 estimated ($5.15 worst case), 0 errored, 0 window flagged for
instruction-like text. **Total $2.105 of the $7 ceiling.**

**Two of six meetings failed GPU transcription QA** before any labelling:
`gr-richmond-6033` and `gr-richmond-6038` — coverage 5% of duration, a
"decoder loop" (4–7 identical segments in a row spanning 400–770 s). Both are
the two meetings sampled specifically because they have **zero** located
episodes (T2-PLAN.md §3d had already flagged `gr-richmond-6038`'s Mac-ASR
track, at 1,118 words/h, as looking defective and "should be inspected by ear
before it is trusted"). That the GPU WhisperX pass *also* fails on exactly
these two audio files, independently of the Mac mlx-whisper pass that located
0 episodes on them, points at the **audio itself** (or Richmond's archive
encode), not at the locate step, as the first thing to check. Recall is
**not measurable** on either (0 exhaustive about-minutes found in the broken
transcript text — `nan` in the table below), so the zero-episode arm of the
sample did not survive intact; only 4 of 6 meetings, all with some located
episodes, carried the recall measurement.

**Recall**, located-episode-minutes over exhaustive about-minutes, 4
meetings / 22.6 locate-side audio-h with usable GPU transcripts (109 total
about-minutes):

| family | about-min | located-min | recall | 95% CI (6-cluster bootstrap) | window recall | screen-alone recall |
|---|---|---|---|---|---|---|
| density | 89 | 75.0 | **0.843** | 0.822–0.913 | 0.843 | 0.820 |
| transit | 23 | 7.0 | **0.304** | 0.000–0.600 | 0.304 | 0.522 |
| any about_* | 109 | 79.0 | **0.725** | 0.605–0.809 | 0.725 | 0.752 |
| about_both | 3 | 3.0 | 1.000 | 1.000–1.000 | 1.000 | 1.000 |

Per meeting: `gr-richmond-6010` 0.61 (33 about-min), `gr-sanleandro-3077` 0.82
(47), `gr-sanleandro-3089` 0.60 (5), `gr-sanleandro-3105` 0.73 (24).

**Screen vs classifier share of the miss.** 30 of 109 about-minutes (27.5%)
were missed. Of those, **23 (77%) never cleared the keyword screen** on the
Mac-ASR locate track (0.5 hits/100w never reached — many sit at exactly 0.0
hits/100w, meaning the ASR text used different words than the codebook's
keyword list, not that the screen's threshold was close); **7 (23%) were
screen candidates the classifier then called `mention_only` or `no`**. The
density family is carried almost entirely by the screen (density screen
recall 0.820 ≈ pipeline recall 0.843 — the classifier barely touches it).
**Transit is where the pipeline loses the most**: screen-alone recall 0.522
but pipeline recall only 0.304, so transit items are both under-screened
*and* more often reclassified down (6 of the 7 classifier misses are
transit/both; see `about_transit: n=20, found=4` and `about_both: n=3,
found=3` in `results.json`). Richmond's Bay Trail/bike-share public-comment
sequence (13 missed windows in `gr-richmond-6010`, mostly transit) reads like
a recurring agenda item whose vocabulary ("trail", "bike share", "bay trail
segment") sits just outside the screen's current keyword list rather than a
classifier problem — raising the keyword screen's transit family coverage,
not its threshold, looks like the first thing to try. Every missed window
with timestamp, label, pipeline verdict and a one-line excerpt is in
`output/validation/recall/missed_windows.md`.

**Caveat.** These are ASR transcripts (Mac mlx-whisper for the locate track,
GPU WhisperX large-v3 for the exhaustive pass), not YouTube auto-captions —
T2-PLAN.md §3d's open question was specifically about the `local_asr` route,
so that is exactly what this measures, but the number does not generalize to
the caption-based `classifier` route on captioned bodies (Santa Clara,
Hayward, etc.), which has its own, separate, still-unmeasured recall.

**Reopen check against Decision #13's rule** ("auto-captions prove too poor
to locate (T0 recall < 0.9)"): the `local_asr` route's overall recall (0.725)
is below 0.9, and transit specifically (0.304) is well below it, with a wide
bootstrap interval from only 6 meeting-clusters. This is below the
reopen threshold stated for the decision; worth flagging to Puck rather than
silently treating `method='local_asr'` counts as adequate for a rate
(T2-PLAN.md §3d's own caution already said as much — this is the measurement
that caution asked for).

**Artefacts:** `scripts/recall_check.py`; `output/validation/recall/`
(`sample.json`, `windows.jsonl`, `results.json`, `table.md`,
`missed_windows.md`, `whisper/`, `batches/b35/`, `audio/`). Not committed.

## Keyword screen expansion (2026-10-18, approved by Puck)

**Change.** 57 active-transportation terms added to the transit family in
`plan/keywords_transit.txt` (section "Active transportation vocabulary"),
derived from reading all 30 missed windows above: `bay trail`, `shoreline
trail`, `multi-use trail`, `greenway`, `bike share`/`bikeshare`, `e-bike(s)`,
`ebike(s)`, `electric bike(s)`, `bike library`, `protected bikeway(s)`,
`biking`, `bicycling`, `bike/biking/bicycle infrastructure`, `bike plan`,
`bicycle plan`, `bike/bicycle boulevard`, `active transportation plan`,
`safe routes to school`, `walking and biking` (+ 4 word-order variants),
`bike ped`, `pedestrian bridge(s)`, `micromobility`, `e-scooter(s)`, `clean
mobility`, `bus shelter(s)`, `bus rider(s)`, `bus passenger(s)`, `transit
rider(s)`; CONDITIONAL (count only beside an unconditional transit hit):
`trail`, `trails`, `lending library`, `bike rack(s)`, `bike parking`,
`pedestrian plan`, `safe routes`, `scooter(s)`. Frequencies over the 293
tracks on disk (67,944 60-s windows) are in the file's comment; bare `trail`
is 0.72 % of windows, everything unconditional is < 0.2 %. The density
family is unchanged: its 16 misses (development feasibility, cap rates,
vacancy tax, parkland dedication) are a different vocabulary question and
were not approved.

**Versioning.** `segment.load_keywords()` now hashes the parsed terms into
`KeywordList.version` (`kw-<10 hex>`; comments do not change it). Every
`classifier_calls` row and every `episodes` row written from now on carries
`keywords_version` (new nullable columns, added by `state.connect()`; backup
`~/.local/share/ytpt/backups/state-2026-10-17.sqlite` taken first). Rows
written before are NULL = **`kw-dfd392a1fe`**, the English pair as committed
(byte-identical to commit 35df7c7); the expanded pair is **`kw-24e8a4acd1`**
(391 terms, 42 conditional). The Spanish pair (`kw-ee372a87db`) is unchanged.
`prompt_version` is NOT touched: codebook and model are the same instrument,
so a window whose bounds did not move keeps its cached label. Two small
mechanics made that true: `classify.cached_calls` matches a labelled row by
bounds when the positional id shifted, and `classify.reconcile_ids` re-keys a
new window whose positional id is already taken by a different span to
`<vid>_w<start_s>` instead of overwriting the paid row. `make test`: all
checks pass (485 PASS lines).

**Re-screen, $0.** `classify.py run --include-classified --include-no-episodes
--dry-run` over the 292 meetings with a caption/ASR track (log
`output/logs/classify_dryrun_kw2.log`): **387 new candidate windows in 100
meetings**, ~499.5 k caption tokens, estimated **$2.42 direct / $1.21 via the
Batches API**. 64 of the 387 are `gr-napa-5713`, an English Mac-ASR track that
had never been read by the classifier (0 cached calls under any instrument;
the meeting was cut from its Spanish Granicus route) — a gap
`--include-classified` found, not the expansion. The remaining 323 are the
expansion; by body: Richmond 84 (15 meetings), Hayward 46, Petaluma 44
(Swagit route), Santa Clara 29, Santa Rosa 22, San Leandro 13, Fremont 13,
Healdsburg 10, BART 9, the rest ≤ 6. The terms that lifted those windows over
the threshold (top keywords of new-only windows): `bike` 151 (conditional,
now anchored by the new unconditional terms), `pedestrian` 61, `trail` 60,
`bicycle` 55, `e-bike` 50, `bike lane` 28, `biking` 24, `greenway` 24, `bay
trail` 23, `trails` 21, `bike share` 18, `safe routes to school` 14, `lending
library` 14.

**Screen-alone recall under the new list, same exhaustive labels, $0**
(`scripts/recall_recompute.py`, writes to
`output/validation/recall/post_kw2_screen_only/`; the pre-expansion artefacts
in `output/validation/recall/` are untouched):

| family | screen-alone recall before → after | screen recall on GPU text before → after |
|---|---|---|
| density | 0.820 → 0.820 | 0.831 → 0.831 |
| transit | 0.522 → **0.826** | 0.522 → **0.826** |
| any about_* | 0.752 → **0.817** | 0.761 → **0.826** |

Screen precision (marked minutes that are about anything) 0.500 → 0.447
(164 → 199 marked minutes of 1,115 labelled); on the GPU text 0.535 → 0.471.
Misses by cause before → after (screen-only stage, classifier not yet run on
the new windows): screen 23 → 16, classifier 7 → 7, now-a-candidate-but-
unlabelled 0 → 7 — those 7 are the windows the batch below labels.

**Classification of the new windows (Batches API, 2026-10-18 07:21 UTC).**
Two batches, one per instrument: `msgbatch_01K7KBCSGpcQsWUw39sa4AZY` (345
English-caption/ASR windows, est $1.07, **actual $0.6122**) and
`msgbatch_01UeiXiTXqgYvnoPMGkTxLbn` (42 Swagit-route windows, est $0.19,
**actual $0.1421**); 387 of 387 succeeded, 0 flagged instruction-like text.
**Total $0.754** vs the $1.21–1.27 estimate (the estimate's cache assumption
is conservative at this size). Result after the labels were merged into
episodes: **93 new located episodes, 9.49 h, in 52 meetings** (64 found
entirely by the new list, 29 by old + new windows merging); tags transit 60 /
density 26 / both 7; route `local_asr` 45, `classifier` 39,
`swagit_transcript` 9. By body: Napa 12 (2.97 h — the never-read ASR track),
Richmond 18 (0.97 h), BART 4 (1.05 h), Santa Rosa 4 (0.95 h), Hayward 13
(0.60 h), Petaluma 11 (0.55 h), Daly City 3 (0.48 h), Healdsburg 2 (0.45 h),
Palo Alto 2 (0.30 h), VTA 1 (0.25 h), the other 8 bodies ≤ 0.17 h each. 18
old `located` rows were superseded (their windows merged into a longer new
span), 46 new spans were trimmed to the part not already transcribed and 5
dropped as fully covered (see "mechanics" below). 64 meetings went
`cut → downloaded` for the cutter to come back to.

**Pipeline recall under the new list** (`output/validation/recall/post_kw2/`,
same exhaustive labels, the new episodes included):

| family | about-min | recall before → after | 95 % CI after | screen-alone after | misses by cause after |
|---|---|---|---|---|---|
| density | 89 | 0.843 → 0.843 | 0.822–0.913 | 0.820 | unchanged |
| transit | 23 | 0.304 → **0.435** | 0.000–1.000 | 0.826 | — |
| any about_* | 109 | 0.725 → **0.753** | 0.654–0.852 | 0.817 | screen 16, **classifier 11** (was 23 / 7) |

Per meeting: `gr-richmond-6010` 0.61 → 0.64, `gr-sanleandro-3089` 0.60 →
1.00, `gr-sanleandro-3077` and `-3105` unchanged. Read: the screen now
reaches 83 % of the transit minutes, but of the 7 transit minutes it newly
handed to the classifier, **4 came back `mention_only` / `no`** (the Bay
Trail funding deliberation at 2:09–2:24 in `gr-richmond-6010`, which the
exhaustive labelling called `about_transit` when read in 60-s windows
without a keyword context) and 3 were located. So the remaining gap is now
mostly the classifier's, not the screen's: 11 of 27 misses. The 16 screen
misses left are the density-vocabulary ones (feasibility, cap rates, vacancy
tax, parkland dedication) plus two transit minutes with no transit word at
all ("the bikes ... the library" at 3:08–3:09). With 23 transit about-minutes
in 4 meetings the transit number is still too noisy to be a rate; the "any"
recall moved from 0.725 to 0.753 and its interval still spans the 0.80 floor
proposed in `plan/DECISIONS-PROPOSED-13.md`.

**Mechanics found and fixed on the way (pipeline/classify.py, 2026-10-18).**
Re-reading a meeting that already had episodes exposed three things the
pipeline had never had to do before, because the window set of a meeting had
never changed under the same instrument:
1. `submit_batch` crashed sorting its per-instrument groups when a run mixes
   the caption route with Swagit (`None` vs `'swagit'` in the sort key);
   nothing had been billed. Fixed (sort on `''`).
2. Episode ids are positional (`_e<n>` in time order). One new about-window
   early in a meeting renumbered every later episode; `write_episodes`'s
   `ON CONFLICT ... WHERE status IN ('located','mention')` refused to touch
   the done rows, so the old last span came back as a **duplicate located row
   under a new number** and the genuinely new span was **dropped** (its
   number was taken). Seen on the first Swagit meetings re-finished
   (`sw-petalumaca-359732_e9` = `_e8`). Now an episode whose core span
   already has a row is that row; a new span whose number is taken gets the
   next free number. Ids keep their shape; they are no longer strictly
   time-ordered after a re-read.
3. A new span that **extends** a transcribed episode (one minute added at its
   edge) would re-transcribe and double count it; now trimmed to its largest
   uncovered piece ≥ 60 s or dropped. And located/mention rows of the same
   route that no span of the new reading matches (merged into a neighbour)
   are removed as superseded (events row `to_status='superseded'`), or the
   cutter would cut the same minutes twice.
One meeting slipped through between the batch landing and the fix:
`JrE255sNtfw` (Santa Rosa) was being cut by the download run while the
labels were merged, so its old `_e3`, `_e4` and the new merged `_e9` were
all cut; `_e3` and `_e4` (subsets of `_e9`) were retired by hand (episode
`failed`, clip row `dead`, `in_frame=0`, note on both) so they do not go to
the GPU. After the re-finish pass no two not-yet-transcribed episodes of any
meeting overlap in core span. `make test` passes after every change.
Also: `classify.py --ids` cannot take a YouTube id that starts with `-`
(`-12hHY34G_8`, Palo Alto) except in the `--ids=<id>` form — argparse reads
it as an option; the re-finish pass ran it separately.

## YouTube auto-caption arm (2026-10-18)

**Question.** The caveat above: the Richmond / San Leandro number is for the
`local_asr` route. The captioned bodies run the same screen + classifier on
the YouTube auto-caption VTT (`videos.captions='auto'`,
`episodes.method='classifier'`), whose recall was still unmeasured. Same
method, same codebook (`t0-a4a8e55760`, v1, `claude-sonnet-4-6`), same
scripts re-pointed (`scripts/recall_check_yt.py`), artefacts under
`output/validation/recall_yt/`; synthetic `videos` rows `rcyt-<id>`
(`kind='validation'`, `in_frame=0`), classifier rows `video_id='recallyt:<id>'`,
batch note `[validation-yt]`. Ceiling $4 (default). Nothing was fetched from
YouTube: the four meetings are the ones whose whole-meeting `output/audio/`
opus was already on disk.

**Sample.** Chosen by hand (not seeded) from the 8 `captions='auto'` meetings
with whole-meeting audio: one per body, 15.0 audio-h —

| meeting | body | date | h | located episodes (min) | tags |
|---|---|---|---|---|---|
| `9FM5406YagE` | BART board | 2026-09-10 | 4.24 | 9 (91.0) | transit |
| `jgyX7NNf7g0` | Los Altos council | 2026-09-08 | 4.82 | 12 (156.6) | density, both |
| `TkZkdilj_nA` | Milpitas council | 2026-09-01 | 3.70 | 1 (4.0) | density |
| `hGui8Fux2_g` | Cupertino council | 2026-09-15 | 2.24 | 2 (9.0) | density |

Left out: Palo Alto (6.6 h, 5.0 h) and a second BART meeting (6.0 h) for
hours, Concord as a coin-flip against the two councils already picked. No
zero-episode meeting exists on this route with audio on disk (every captioned
meeting with audio was downloaded *because* it had episodes), so this arm has
no zero-located arm at all — Milpitas (1 episode, 4 min) and Cupertino (2, 9
min) stand in for it.

**Cost.** GPU: batch **b38**, pod `hce2tpbqsje6id`, 15.0 audio-h, no
diarization, **$0.187 actual** (22.9 GPU-min, RTFX 56x) vs $0.53 estimated
(same ~3x over-prediction as b35; the measured-rate estimate was $0.21).
Claude: 709 windows of 899 (190 had < 10 padded words), **$0.766 actual** vs
$1.44 estimated ($3.27 worst), batch `msgbatch_01CqJqDutTjwiazDTzSsYXzB`, 709
of 709 succeeded, 0 flagged for instruction-like text. **Total $0.953 of the
$4 ceiling.** The pod was terminated by `gpu_batch.py` at job end.

**One QA FAIL, recovered without a new pod.** `rcyt-jgyX7NNf7g0` (Los Altos)
failed `gpu_batch` ingest QA: "2 identical segments in a row spanning 181 s at
58.7 min (decoder loop)". Unlike `gr-richmond-6033/6038` (coverage 5 %, 400–770
s of looped text) this transcript has 2,611 segments, 37,759 words and 78 %
speech coverage; the "loop" is two `Thank you.` segments bracketing a
3-minute silence (the caption track has 5 words in 58.7–61.8 min — a recess).
The JSON was kept in `batches/b38/out/` by the normal FAIL path and
`harvest` picked it up from there; windows 59–61 carry no text and were not
sent. So 4 of 4 meetings carried the measurement.

**Caption track vs WhisperX text** (`caption_quality.{json,md}`; WER is
word-level Levenshtein over the whole meeting after lower-casing, dropping
`[tags]`/`>>`, splitting hyphens, stripping punctuation):

| meeting | WhisperX words/h | caption words/h | ratio | WER | 10-min median / p90 |
|---|---|---|---|---|---|
| `9FM5406YagE` BART | 5,200 | 5,605 | 1.08 | **0.101** | 0.113 / 0.129 |
| `jgyX7NNf7g0` Los Altos | 7,839 | 8,492 | 1.08 | **0.102** | 0.123 / 0.150 |
| `TkZkdilj_nA` Milpitas | 5,241 | 5,659 | 1.08 | **0.099** | 0.109 / 0.193 |
| `hGui8Fux2_g` Cupertino | 8,024 | 8,632 | 1.08 | **0.090** | 0.103 / 0.127 |

The auto-captions carry ~8 % more tokens than WhisperX (fillers, `um`/`uh`,
restarts that WhisperX drops) and differ from it in about one word in ten,
evenly across the meeting. The errors that matter are not random: the words
the screen keys on are exactly the ones a generic acoustic model gets wrong.
On the BART track `fare` → `fair` (13 → 0 hits, `fair gate` ×5), `ridership`
→ `wrership` (16 → 6), `riders` → `writers` (6 of 12); on Los Altos `YIMBY`
(17 → 0) and `rezoning`/`rezone` (11 → 0) vanish entirely. Total keyword hits
on the caption text vs the WhisperX text: BART 303 vs 351, Los Altos 717 vs
758, Milpitas 9 vs 10, Cupertino 26 vs 28.

**Recall**, located-episode-minutes over exhaustive about-minutes, 4 meetings
/ 15.0 h, 297 about-minutes (180 `about_density`, 115 `about_transit`, 2
`about_both`; 119 `mention_only`, 293 `no`). The old keyword list
(`kw-dfd392a1fe`, git 35df7c7) and the expanded one (`kw-24e8a4acd1`) give
**byte-identical** verdicts on these four meetings (`kw_old/` and `kw_new/`;
0 of 899 windows differ): none of the 57 active-transportation terms occurs
in them, and the `episodes` rows are all pre-expansion (`keywords_version`
NULL). So "old → expanded" is one column here:

| family | about-min | located-min | recall | 95 % CI (4-cluster bootstrap) | window recall | screen-alone recall | screen on GPU text |
|---|---|---|---|---|---|---|---|
| density | 182 | 156.6 | **0.860** | 0.436–0.938 | 0.879 | 0.819 | 0.819 |
| transit | 117 | 70.3 | **0.601** | — (one meeting) | 0.650 | 0.513 | 0.701 |
| any about_* | 297 | 224.9 | **0.757** | 0.552–0.884 | 0.788 | 0.697 | 0.771 |
| about_both | 2 | 2.0 | 1.000 | — | 1.000 | 1.000 | 1.000 |

Per meeting: BART 0.60 (117 about-min, 70.3 located), Los Altos 0.88 (166,
146.5), Milpitas 0.32 (8, 2.5), Cupertino 0.92 (6, 5.5). Read the intervals
as rough: four clusters, and every transit about-minute but two is in the one
BART meeting, so the transit interval is degenerate (the bootstrap can only
resample that meeting) and the density interval's low end is Milpitas drawn
four times.

**Screen precision** on this route is much higher than on the Mac-ASR track:
0.838 (207 of 247 marked minutes are about something; 0.845 on the GPU text)
vs 0.447–0.500 on Richmond / San Leandro — these meetings have long, dense
agenda items (a 75-min Los Altos housing item, BART's own business)
rather than scattered mentions.

**Screen vs classifier share of the miss.** 63 of 297 about-minutes (21 %)
missed: **61 (97 %) never cleared the keyword screen** on the caption track,
**2 (3 %)** were candidates the classifier called `mention_only` (BART 1:07,
the general manager's farewell; Los Altos 0:30, the lottery-tier
deliberation). 40 of the 61 screen misses sit at exactly 0.0 hits/100w. Split
the screen misses by what the WhisperX text says:
- **15 are the caption track's fault**: the screen fires on the GPU text for
  the same minute (`screen_gpu_s` ≥ 30) but not on the captions — the
  `fare`/`ridership`/`riders`/`YIMBY`/`rezoning` dropouts above. 14 of the
  15 are BART (0:32, 0:47–0:49, and 1:22–1:53, the quarterly
  service-performance item: ridership, fare gates, on-time performance), 1 is
  Los Altos (4:10).
- **46 are the keyword list's fault** on either text: BART's operations and
  maintenance discussion (0:20–0:53 and 1:35–2:01: switch failures, traction
  power substations, train control / OCC, heat-related outages, delay
  metrics — "train" is conditional and nothing unconditional sits beside it),
  and three density items with no density vocabulary — Los Altos's
  below-market-rate lottery eligibility tiers (0:26–0:51, 11 minutes),
  Los Altos's ballot-measure / local-control-over-housing debate (3:36–4:14),
  and Milpitas's BMR homeownership lottery item (2:46–2:50, all 5 of its
  misses). These are the auto-caption arm's version of the Granicus arm's
  density-vocabulary gap (feasibility, cap rates, vacancy tax, parkland):
  affordable-housing *allocation* (lottery, BMR, local preference, income
  tiers) is labelled `about_density` by the codebook but is not in
  `keywords_density.txt`.

Every missed window with timestamp, label, cause, pipeline verdict and a
one-line excerpt: `output/validation/recall_yt/missed_windows.md`.

**A caveat specific to BART.** The exhaustive labeller called 115 of BART's
117 about-minutes `about_transit`: for a transit operator's own board, nearly
every substantive item *is* about transit (substation maintenance, service
performance, fare gates), so "recall" there measures how much of the
agency's business the screen reaches, not whether a transit topic surfaced in
a city council. The 9 located episodes (91 min) still catch 70 of those 117
minutes; what they miss is the operations/maintenance vocabulary, which is a
decision about what the transit family should mean on operator bodies, not
a caption-quality finding.

**Side by side with the Granicus / `local_asr` arm** (that arm's "after"
column, same codebook, same exhaustive method):

| | Granicus / `local_asr` (Richmond, San Leandro) | YouTube auto-captions (BART, Los Altos, Milpitas, Cupertino) |
|---|---|---|
| sampled / usable meetings, audio-h | 6 / 4, 22.6 h | 4 / 4, 15.0 h |
| GPU batch, actual $ | b35, $0.231 | b38, $0.187 |
| Claude windows labelled, est → actual $ | 1,115, $2.27 → $1.874 | 709, $1.44 → $0.766 |
| about-minutes (density / transit / both) | 109 (86 / 20 / 3) | 297 (180 / 115 / 2) |
| recall any, 95 % CI | 0.725 → **0.753** (0.654–0.852) | **0.757** (0.552–0.884) |
| recall density | 0.843 → 0.843 | **0.860** |
| recall transit | 0.304 → 0.435 (0–1) | **0.601** (1 meeting) |
| screen-alone recall any (old → expanded list) | 0.752 → 0.817 | 0.697 → 0.697 |
| screen-alone recall on the GPU text | 0.761 → 0.826 | 0.771 |
| screen precision (locate track) | 0.500 → 0.447 | **0.838** |
| misses: screen / classifier | 16 / 11 of 27 | **61 / 2** of 63 |
| locate track vs WhisperX, WER | not measured (Mac mlx-whisper track) | **0.090–0.102**, 1.08x the words |
| transcripts lost to GPU QA | 2 of 6 (Richmond 6033/6038, decoder loop, 5 % coverage) | 0 of 4 (Los Altos's FAIL was a 3-min silence) |

Read: the two routes land at the same overall recall (0.75–0.76), both below
the 0.9 in Decision #13's reopen rule and still spanning the 0.80 floor of
`plan/DECISIONS-PROPOSED-13.md`, but for different reasons. On `local_asr`
the expanded list moved the gap onto the classifier (11 of 27). On the
caption route the classifier is nearly blameless (2 of 63) and the screen
owns the miss: a quarter of it (15 of 61) is auto-caption vocabulary damage
on the screen's own keywords (`fare`, `ridership`, `YIMBY`, `rezoning` — a
homophone list the screen could absorb for $0), three quarters is keyword
coverage of two topics the codebook counts but the list does not name
(transit operations on operator bodies; affordable-housing allocation).
The active-transportation expansion did nothing here because none of its
terms were spoken in these four meetings.

**What more spend would buy.** The transit number rests on one meeting.
The second BART meeting (6.0 h) plus the two Palo Alto meetings (11.6 h)
left out of the sample are on disk and would cost ≈ $0.22 GPU + ≈ $0.9
Claude at the rates measured above — within the remaining $3.05 — and would
give the transit family a second operator cluster and a city with a transit
agenda. Not run: the task's gate for a new pod was "2+ usable meetings
missing", which did not apply.

**Artefacts:** `scripts/recall_check_yt.py`; `output/validation/recall_yt/`
(`sample.json`, `windows.jsonl`, `results.json`, `table.md`,
`missed_windows.md`, `caption_quality.{json,md}`, `kw_old/`, `kw_new/`,
`whisper/` (4 JSON), `batches/b38/`, `audio/`, `*.log`). Not committed.

## Codebook v2 and the adjudicated reference (2026-10-18)

The classifier gap above was taken up in `plan/CODEBOOK-V2-PILOT.md` (three
codebook edits, four arms, $0.57) and `plan/CODEBOOK-V2-PILOT-FULLASR.md`
(no screen, $0.77). Puck adjudicated all 11 v2@120 over-call windows as
"about" (PILOT §8), which flips 18 of this check's reference minutes
(17 mention_only, 1 no → 14 about_transit, 4 about_density; 109 → 127
about-minutes). The corrected reference is
`output/validation/codebook_v2/adjudicated/reference_adjudicated.jsonl`; the
files under `output/validation/recall/` here are the v1-coded originals and
are unchanged. Re-scored against it (PILOT §9), the current instrument
(v1 @ 240 s) has recall any **0.606** / transit **0.297**, and v2 @ 120 s
**0.811** / **0.865** with precision 0.873 / 0.756. The YouTube arm above
shares no meeting with the adjudication and stands as written. Adoption of
v2 is proposed in `plan/DECISIONS-PROPOSED-14.md`.

*Corrected 2026-10-18 after validation.* **Never pair a v1-reference figure
with an adjudicated-reference one.** The transit improvement must be quoted
on one reference at a time: **0.435 → 0.783** against the v1-coded reference,
or **0.297 → 0.865** against the adjudicated reference. "0.44 → 0.87" mixes
the two and roughly doubles the apparent gain. Likewise the full-ASR arm is
**0.927** on the v1 reference and **0.898** on the adjudicated one.

**And the adjudicated reference is not independent of the arm it scores.**
All 18 flipped minutes were created by adjudicating v2@120's own over-calls
as "about", so v2@120 hits **18 of 18 by construction**. On the 109
*untouched* reference minutes v2@120 scores any **0.780**, transit
**18/23 = 0.783**, density **0.787** — i.e. the whole lift from 0.783 to
0.865 on transit comes from minutes v2 itself defined. The transit reference
grew 23 → 37 minutes, **38 % of it from v2's output**, and v2@120's transit
precision jumps 0.415 → 0.756 from the same reflip. Every quoted
adjudicated-reference figure carries this caveat; a reference coded under v2
from scratch (a second sample, BART or VTA) is the clean test.

## 120-s candidate windows are the default (2026-10-18, approved by Puck)

`plan/CODEBOOK-V2-PILOT.md` §4 measured the pinned v1 codebook on the same
four Richmond / San Leandro reference meetings at two granularities, core
bounds, like-for-like: **v1 at 120 s: recall 0.734 any / 0.609 transit /
0.775 density, precision 0.825 any**; v1 at 240 s: 0.697 / 0.435 / 0.775,
precision 0.809. Edit 3 of the pilot alone recovered 5 of the 13 classifier
misses with *higher* precision, for ~1.85x the candidate windows (the system
block is cached, so ~$0.22 per 19 audio-h at the batch rate). It is adopted:
`classify.DEFAULTS['max_run']` and `segment.py --max-run` are now **120 s**.
The codebook is untouched (v1, `t0-a4a8e55760`, byte-identical; v2 is still a
pilot pending Puck's adjudication of the 11 over-calls). The granularity the
*bare* prompt_version has always meant stays 240 s (`classify.PV_MAX_RUN`),
so a default run now writes `t0-a4a8e55760/claude-sonnet-4-6/mr120` -- the
same instrument id the pilot's v1_120 arm used -- and no existing 240-s row
changes version or loses its cache. `collect` and the recall scripts pass the
granularity the rows were read at (`scripts/recall_recompute.py --max-run`).
Consequence for cost: a meeting already read at 240 s is NOT cached at 120 s
(different bounds, different instrument), so re-reading the read corpus at
120 s is a full re-bill (see the dry-run figure in the next section); the
120-s default applies to meetings read from now on and to new measurements.

## Density vocabulary + caption-homophone layer (2026-10-18, approved by Puck)

**Change 1, vocabulary.** 141 terms added to `plan/keywords_density.txt`
(section "Density vocabulary expansion"), read off the three missed-window
lists (`recall/missed_windows.md`, `recall_yt/missed_windows.md`,
`codebook_v2/missed_v2_120.md`): development economics (`cap rate(s)`,
`development/financial feasibility`, `real estate development`, `residential
development/project/units/property`, `housing market`, `pro forma`),
development fees (`parkland dedication`, `park dedication`, `quimby`),
vacancy taxes (`vacancy tax`, `vacant property/home(s) tax`, `empty homes
tax`, `ellis act`), affordable-housing allocation (`below market`,
`affordable unit(s)`, `area median income`, `very low / extremely low /
moderate income`, `income-restricted / qualified`, `income tier(s)`, `vli`,
`local preference`, `lottery`, `housing / bmr lottery`, `homeownership
(program)`, `first-time homebuyer`, `down payment assistance`, `section 8`,
`housing (choice) voucher(s)`), public and supportive housing (`housing
authority`, `public housing`, `(permanent) supportive housing`, `interim /
transitional housing`), tenant protections (`costa hawkins`, `ab 1482`,
`tenant protection act`, `rent registry / board / program / ordinance /
protection(s) / cap`, `renter protection(s)`, `rental housing / units`,
`mobile home ordinance / park(s)`, `mobilehome`); CONDITIONAL (count only
beside an unconditional density hit in the window): `feasibility`,
`feasible`, `construction cost(s)`, `cost of construction`, `prevailing
wage`, `developer(s)`, `development(s)`, `residential`, `subdivision(s)`,
`impact fee(s)`, `in(-)lieu fee`, `ami`, `low(-)income`, `affordable`, `tier
one/two`, `hud`, `operating subsidy`, `mobile home`, `rent increase(s)`,
`tenant(s)`, `renter(s)`, `landlord(s)`, `rental(s)`, `home rule`.
Frequencies over the 293 tracks on disk (59,089 60-s windows >= 10 words)
are in the file's comments; the rule was: housing-specific and under ~0.15 %
of windows -> unconditional, housing-only-in-context -> conditional.
Transit-operations vocabulary on operator bodies (BART's substations,
traction power, train control, 31 of the auto-caption arm's remaining
misses) was NOT added: that is the "what should transit mean on an operator
body" question above, not approved. English pair now **`kw-066fe5a8cc`**
(511 terms, 79 conditional; was `kw-24e8a4acd1`, 391 / 42).

**Change 2, caption homophones.** `plan/keywords_aliases_auto.txt`, applied
by `segment.apply_aliases()` through `classify.keywords_for(lang, captions)`
to `captions='auto'` tracks ONLY, to the screen's matching only (stored text,
classifier input and transcripts are untouched): `fairgate(s)` / `fair
gate(s)` -> `fare gate(s)`; `wrership`, `writership` -> `ridership`;
`transit/bus writer(s)` -> `transit/bus rider(s)`; `yimi`, `ymi` -> `yimby`;
`reszoning(s)`, `resoning(s)` -> `rezoning`, `reszone`, `resone` -> `rezone`;
CONDITIONAL aliases (own entry, counted only beside an unconditional hit of
the family): `fair` -> `fare`, `rena` -> `rhna`. Over the 107 auto tracks
`rezoning` occurs in 2 windows and `reszoning` in 23; `wrership` +
`writership` in 245 windows against `ridership`'s own count on the same
tracks. Alias list version **`al-7e4c06d52b`**; a list loaded with it is
versioned `kw-066fe5a8cc/al-7e4c06d52b` and that string is what the
`keywords_version` column carries on every row located through it (120 of
the 211 new episodes below). `make test`: all checks pass (the three checks
that pinned 240 s now read the default from `classify.DEFAULTS`).

**Re-screen, $0** (`classify.py run --include-classified
--include-no-episodes --dry-run`, 293 meetings, log
`output/logs/classify_dryrun_kw3_mr240.log` / `_mr120.log`):
- at **240 s** (the granularity every read meeting's rows are at, so only
  the genuinely new windows are unlabelled): **1,199 new windows in 209
  meetings** (3,472 cached), est **$3.81** at the batch rate. 165 of the
  1,199 exist only because of the alias layer (BART 97 windows / 12
  meetings is mostly `ridership`, `fare gate`, `fair`). By body: San Leandro
  166, Santa Clara 131, Richmond 112, Petaluma 103, San Jose 100, BART 97,
  Vallejo 83, Santa Rosa 52, Hayward 48, VTA 41, the rest <= 33. By meeting
  status: cut 656, classified 327, downloaded 149, no_episodes 67. The top
  keywords of the new-only windows are the OLD anchors (`housing` 590,
  `development` 308, `units` 175, `affordable` 125, `homes` 123,
  `residential` 122, `developer` 103 ...): the conditional terms raise the
  density of windows that already had one housing word, and the 3-window
  smoothing then carries their neighbours over 0.5 -- which is exactly the
  shape of the San Leandro feasibility and Los Altos lottery misses.
- at **120 s** (the new default): **8,579 windows, est $21.31** at the batch
  rate -- a full re-bill of the read corpus, because a 240-s label is not a
  120-s label (different bounds, instrument `…/mr120`). Not run; outside
  this part's $6 ceiling by 3.5x. The 120-s default therefore applies from
  the next meeting read onward; the read corpus stays at 240 s until Puck
  wants it re-read (`--max-run 120 --include-classified --batch --dry-run`
  prices it).

**Classification of the 1,199 (Batches API, 2026-10-18 17:31 UTC, 240 s,
v1).** `msgbatch_01WXL7u8p5zcqj67KDep5jww` (1,116 English caption/ASR
windows, `t0-a4a8e55760`) **$2.0396** and `msgbatch_01WbgBrhKjCbjvParqQtdJVg`
(83 Swagit windows, `…/swagit`) **$0.3116**; 1,199 of 1,199 succeeded, 0
flagged instruction-like text; 2 Richmond windows whose ids moved between
submit and collect were filled directly for $0.012. **Total $2.36 actual vs
$3.93 estimated** (worst-case bound $6.96 -- `--cap-usd` had to be set to 7
for the submit to pass that bound; the bill came in at 60 % of the estimate,
as on every batch since the kw2 run). Result after merging into `episodes`
(snapshot before: `output/validation/recall/episodes_snapshot_pre_kw3.json`):
**211 new located episodes, 27.08 h, in 99 meetings**; tags density 142 /
transit 44 / both 25; route `classifier` 145, `local_asr` 53,
`swagit_transcript` 13. 93 old located rows (18.18 h) were superseded (their
windows merged into longer new spans), so located-and-not-yet-transcribed
time went **65.0 h -> 74.6 h net (+9.6 h)**. By body (new rows, hours):
BART 33 (7.82 h), VTA 8 (3.23), Vallejo 14 (2.73), Santa Rosa 6 (2.55), San
Leandro 31 (1.60), San Jose 11 (1.57), Livermore 5 (1.02), Santa Clara 16
(0.90), Palo Alto 3 (0.82), Petaluma 20 (0.78), Pleasanton 8 (0.73),
Richmond 15 (0.67), Concord 5 (0.53), Los Altos 6 (0.33), Healdsburg 2
(0.32), San Mateo 5 (0.30), Cupertino 2 (0.30), the other 8 bodies <= 0.23 h.
99 meetings went `cut -> downloaded` for the cutter, 2 `no_episodes ->
classified`. The write_episodes mechanics of the kw2 run (span-matched ids,
trimming against transcribed spans, superseding) held: 0 overlapping
not-yet-transcribed pairs of this run's own rows. One 30-s overlap exists
between `gr-sanleandro-3077_e8` (this run, `located`) and `_f2` (`cut`,
`method='fullasr'`, the sibling full-ASR pilot writing to the same meeting
concurrently); left for that pilot's owner.

**Recall, same exhaustive labels, $0** (`scripts/recall_recompute.py --out
output/validation/recall/post_kw3 --max-run 240` and `scripts/recall_check_yt.py
recompute --out-new kw3`; padded episode bounds, like the "before" columns
in this file -- subtract ~0.06 on "any" to compare with the pilot's
core-bound numbers):

| arm | family | recall before -> after | screen-alone before -> after | misses by cause before -> after |
|---|---|---|---|---|
| Granicus / `local_asr` (4 mtgs, 109 about-min) | density | 0.843 -> **0.859** | 0.820 -> **0.944** | |
| | transit | 0.435 -> 0.457 | 0.826 -> 0.826 | |
| | any | 0.753 -> **0.770** (CI 0.674-0.860) | 0.817 -> **0.917** | screen 16 -> **7**, classifier 11 -> 14, other 0 -> 4 |
| YouTube auto-captions (4 mtgs, 297 about-min) | density | 0.860 -> **0.940** (CI 0.835-0.948) | 0.819 -> **0.912** | |
| | transit | 0.601 -> **0.716** (one meeting) | 0.513 -> **0.650** | |
| | any | 0.757 -> **0.851** (CI 0.720-0.945) | 0.697 -> **0.808** | screen 61 -> **38**, classifier 2 -> 1, other 0 -> 1 |

Per meeting: Richmond 6010 0.64 -> 0.65, San Leandro 3077 0.82 -> 0.83,
3105 0.73 -> 0.77, 3089 1.00; BART 0.60 -> **0.72**, Los Altos 0.88 ->
**0.95**, Milpitas 0.32 -> **0.82**, Cupertino 0.92. Read:
- On the caption arm the screen now reaches the Los Altos lottery tiers and
  the Milpitas BMR lottery (density screen-alone 0.82 -> 0.91) and the BART
  `ridership` / `fare gate` minutes (transit 0.51 -> 0.65). The 38 screen
  misses left are 31 BART operations/maintenance minutes (substations,
  traction power, train control, delay metrics -- the operator-body question,
  not approved) plus 7 Los Altos minutes of the ballot-measure / local-
  control debate with no housing word in them.
- On the Mac-ASR arm the screen's density gap is essentially closed
  (screen-alone 0.82 -> 0.94; 7 screen misses left: 3 Bay Trail minutes with
  no transit word, 2 feasibility minutes with no density word, `2:02` vacancy
  tax at 0.32 hits/100w, `1:29`), and the miss moved onto the classifier:
  14 of 25 misses are now candidate windows v1 at 240 s called
  `mention_only` -- the pilot's finding, and what the 120-s default (and v2,
  if adopted) is for. The 4 "other" misses are `3077` 1:21, 1:22, 2:06,
  2:07: labelled `about_density` by this run, but `write_episodes` trimmed
  the new spans against the sibling full-ASR pilot's `cut` rows
  (`method='fullasr'`) on that meeting, which this arm's recall does not
  count; with those four minutes the arm's "any" would read 0.81 and density
  0.90.
- Both arms are still under Decision #13's 0.9 reopen line on "any"; the
  caption arm's density family is now at it (0.94, CI 0.84-0.95).

**Artefacts:** `output/validation/recall/post_kw3/` and
`output/validation/recall_yt/kw3/` (`windows.jsonl`, `results.json`,
`table.md`, `missed_windows.md`); logs `output/logs/classify_dryrun_kw3_*.log`,
`classify_batch_kw3_mr240.log`, `classify_collect_kw3_{1,2}.log`,
`kw3_attrib.log`, `recall_recompute_kw3.log`, `recall_yt_recompute_kw3.log`.
Files changed: `pipeline/segment.py` (alias layer, 120-s default),
`pipeline/classify.py` (DEFAULTS / PV_MAX_RUN, ALIAS_FILES, keywords_for),
`plan/keywords_density.txt`, new `plan/keywords_aliases_auto.txt`,
`scripts/recall_recompute.py` (`--max-run`, `--captions`),
`scripts/recall_check_yt.py` (`--out-new`), `analysis/test_pipeline.py`
(three 240-s pins). Codebook v1 text and `CODEBOOK_VERSION` byte-identical;
`.env`, `plan/frame.csv`, `plan/DECISIONS.md` untouched. Nothing committed.
