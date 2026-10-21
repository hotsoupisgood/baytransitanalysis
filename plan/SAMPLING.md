# Sampling frame

**Design:** stratified by `speaker × month`, with a quota of transcribed
**audio-hours** per cell, drawn at random within the cell.
Decided by Puck on 2026-09-17 (`DECISIONS.md` #9). Implemented by
`plan/build_frame.py`; the drawn frame is `plan/frame.csv` and it *is* the study.

## Why month (not quarter)

The strong argument isn't balance, it's **matching on the news cycle**. Political
YouTube is reactive — everyone is talking about whatever happened that week. If
Hasan and Shapiro are sampled from the same month, they are responding to the
same events, so a difference between them is a difference in *how they talk*,
not in *what happened to be in the news* when you sampled them.

Quarterly strata leak that: a Q4 Hasan video about one event and a Q4 Shapiro
video about another are not comparable. Monthly mostly closes it. Weekly would
close it further but the cells get too thin to sample from.

## Why hours, not a video count

n videos per cell equalises the wrong thing. Ben Shapiro's uploads are ~10-minute
clips; Hasan's are 45 minutes of stream after the cap. Four videos each would
give one speaker four times the speech of the other in every cell, and word
frequency, collocates and topic proportions are all estimated from *tokens*. An
hours quota equalises the measurement, which is what the statistics use.

What it does **not** equalise is the number of videos, and videos — not tokens —
are the independent unit (`analysis/METHODS.md` §7). A 3-hour quota gives ~4
capped Hasan videos and ~18 Shapiro clips, so the between-video variance is
estimated from fewer clusters on Hasan's side. That is a real cost, and it is
why `--min-videos` exists: a cell keeps drawing until it has both the hours and
at least that many videos. Expect wider intervals for the long-video speaker;
the hierarchical model handles unequal cluster counts, it just can't invent
information that isn't there.

## Parameters

| Parameter | Default | Note |
|---|---|---|
| Strata | speaker × month | `--period quarter` collapses if too many months drop |
| Quota per cell | **3 audio-hours** (capped) | 3 speakers × ~32 months × 3 h ≈ 290 h ≈ the budget the earlier design implied |
| Per-video cap | **45 min** | one stream cannot fill a cell; also caps cost per video |
| Minimum videos per cell | **3** | so a cell is never one or two clusters |
| Duration filter | 5-90 min | sub-5-minute clips are Shorts and teasers; the 90-minute ceiling is a **format** decision, not a cost one — with the cap, longer videos could be admitted (`--max-min 0`) |
| Selection within a cell | uniform random order, per-cell seed | seed recorded in `frame_meta.json` |

`build_frame.py` writes three files: `frame.csv` (one row per sampled video),
`frame_cells.csv` (eligible / sampled / quota / shortfall per cell) and
`frame_meta.json` (every parameter, the seed, and the catalog's sha256).

## The draw, and why it is nearly equal-probability

Within cell *c* with eligible videos *i = 1…N* and capped durations
*dᵢ = min(Dᵢ, 45 min)*:

1. Put the videos in a uniformly random order (seeded by `seed:speaker:period`,
   over a pool sorted by `video_id`, so catalog row order cannot change the draw).
2. Walk the order, taking videos, until **both** the running capped total ≥ Q
   **and** at least `--min-videos` are taken.
3. Keep the video that crosses the quota.

Two properties follow.

**Bounded overshoot.** The realised hours *Hc* satisfy

```
Q ≤ H_c < Q + max_i d_i ≤ Q + 45 min
```

so a cell is never short and never more than one capped video over. The realised
value is recorded per cell; use it as the exposure/offset in any rate model
rather than assuming exactly Q.

**Near-flat inclusion probability.** Video *i* is taken iff, when the walk
reaches it, the stopping rule has not yet fired — an event determined entirely
by the videos placed *before* it:

```
π_i = P( S_before(i) < Q  or  k_before(i) < m )
```

where *S_before* and *k_before* are the capped-duration sum and count of the
videos preceding *i*. **dᵢ does not appear.** The only dependence on *i*'s own
length is indirect: a long video is absent from "the others", which makes the
prefix sums slightly smaller and tilts π upward by O(1/N).

Simulation, cell of 20 videos with capped durations 10-45 min, Q = 3 h
(200,000 replications):

| rule | π̄, shortest 5 | π̄, longest 5 |
|---|---|---|
| **take the crossing video (this design)** | 0.348 | 0.366 |
| stop before exceeding the quota | 0.447 | 0.292 |

The obvious alternative — skip any video that would overshoot — depends on *dᵢ*
directly and systematically under-samples long videos by a third. That is why
the crossing video is kept.

`build_frame.py --pi-reps N` estimates π̂ᵢ for every drawn video by re-running
the same rule N times with an independent RNG (Monte Carlo SE ≤ 0.5/√N ≈ 0.011
at the default 2,000) and writes it to `frame.csv`. Use it for design weights
(1/π̂) if you ever want a design-based estimate, or as a check that selection
really is close to equal-probability inside a cell. `frame_cells.csv` also
carries `expected_videos` = Σπ̂, the expected sample size for that cell.

## Before committing: check for thin and empty cells

```bash
make crawl                      # metadata only, no audio; writes plan/catalog.tsv
make frame HOURS=3              # dry run: the cell table
make frame HOURS=3 DRAW=1       # writes frame.csv, frame_cells.csv, frame_meta.json
```

Expect trouble in at least these places:

- **Tucker.** His output and platform changed over the last few years; there may
  be months with few or no uploads on the channel you're crawling.
- **Channel migrations.** If a person's content moved between channels, one
  channel URL is not the same as one person. A stratum is a *speaker*; list every
  channel that person used in `plan/channels.tsv`.
- **Hasan's main channel is clips**; the long streams may live elsewhere. Which
  one you crawl changes what "Hasan" means in the data.

**Rules for thin cells** — pick before you look at the results, not after:

1. Cell has ≥ Q hours of eligible video → draw to the quota.
2. Cell has some eligible video but < Q hours → take all of it, record the
   shortfall (`frame_cells.csv`, `status = short`).
3. A month where **any** speaker has fewer than `--min-videos` eligible videos
   → **drop that month for every speaker**. A month present for two speakers and
   absent for the third reintroduces exactly the confound this design removes.
4. If more than ~20% of months get dropped, collapse to quarters
   (`--period quarter`) and re-run. `build_frame.py` prints the share and says so.

Report n from `frame.csv`, never from what happens to be on disk — otherwise a
failed download silently becomes a study-design change.

## Note on `upload_date`

`--flat-playlist` does not reliably return upload dates. The crawl passes
`--extractor-args youtubetab:approximate_date`, and the dates are
**approximate** — good enough for monthly bucketing, but a video near a month
boundary may land in the wrong cell. If that matters, re-resolve exact dates for
the drawn subset only (a few hundred cheap metadata fetches) before downloading
audio, and set `date_exact = 1` on those rows.

## Open, and worth deciding before the crawl

- **The 90-minute ceiling.** With a 45-minute cap it is no longer about cost or
  dominance; it decides whether multi-hour livestreams are part of "how this
  speaker talks" at all.
- **First 45 minutes, or a random 45-minute window?** The first 45 minutes of a
  stream is often pre-show, music and chat-reading. A random window
  (`ffmpeg -ss <offset> -t 2700`) is more representative and costs one extra
  column (`window_start_s`) to record.
- **Format covariate.** Record livestream / edited / interview at discovery so
  it can enter the model later. Cheap now, impossible retroactively.
