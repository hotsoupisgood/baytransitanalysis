# Swagit transcript vs YouTube captions as a locating instrument -- petaluma

Stage 1 only (`segment.candidate_windows`, English keyword pair, 60-s windows, threshold 0.5 hits/100w, runs cut at 240 s): the same matcher over two transcripts of the same audio. Neither track is ground truth, so these are AGREEMENT figures, not accuracy.

## Tracks

| meeting | h | YouTube cues / words / wpm / cov | Swagit cues / words / wpm / cov |
|---|---|---|---|
| 2026-07-20 (`D5PLaHE9MMk` / clip 394014) | 4.63 | 6699 / 44,035 / 158.5 / 0.999 | 3293 / 44,158 / 159.3 / 0.997 |
| 2026-08-03 (`U1Pva1fkJRw` / clip 395082) | 2.26 | 3347 / 21,858 / 161.9 / 0.997 | 1656 / 22,341 / 162.2 / 1.018 |
| 2026-09-14 (`UEV2mj4Qf-s` / clip 400885) | 7.27 | 8593 / 55,629 / 152.6 / 0.862 | 4387 / 58,757 / 154.3 / 0.873 |

## Clock offset between the two publications

Add `offset` to a YouTube time to get the Swagit time. Estimated from the unique shared 6-word sequences, not from the keyword windows.

| meeting | offset s | shared unique 6-grams | within 10 s of the median | p10..p90 |
|---|---|---|---|---|
| 2026-07-20 | **-37.6** | 33,937 | 1.0 | -40.1..-35.7 |
| 2026-08-03 | **+163.8** | 16,553 | 1.0 | +161.4..+165.6 |
| 2026-09-14 | **+486.5** | 42,990 | 0.812 | +292.3..+488.9 |

## A. Candidate-window recall

| meeting | YouTube windows (s) | Swagit windows (s) | overlap s | recall (lag 0) | **recall (aligned)** | reverse (aligned) |
|---|---|---|---|---|---|---|
| 2026-07-20 | 8 (1560) | 8 (1440) | 1245 | 0.846 | **0.798** | 0.864 |
| 2026-08-03 | 9 (1620) | 9 (1680) | 1435 | 0.296 | **0.886** | 0.854 |
| 2026-09-14 | 49 (9600) | 54 (10980) | 8995 | 0.644 | **0.937** | 0.819 |
| **all 3** | (12780) | (14100) | 11675 | 0.624 | **0.914** | 0.828 |

## B. Located-episode recall (episodes the paid YouTube route already found)

18 episodes, 132 core min, all `method='classifier'` off the YouTube track. On the aligned clock, Swagit candidate windows cover **18/18** of them over at least half their span and 18/18 at all; span-weighted coverage **0.956**.

| episode | span (YouTube clock) | min | tags | Swagit coverage (aligned) | (lag 0) |
|---|---|---|---|---|---|
| `UEV2mj4Qf-s_e1` | 1:01:00-1:04:00 | 3.0 | transit | 0.70 | 0.67 |
| `UEV2mj4Qf-s_e18` | 5:04:00-5:07:00 | 3.0 | density | 0.70 | 1.00 |
| `UEV2mj4Qf-s_e6` | 2:29:00-2:34:00 | 5.0 | density | 0.80 | 0.60 |
| `UEV2mj4Qf-s_e7` | 2:42:00-3:01:00 | 19.0 | both | 0.89 | 0.47 |
| `UEV2mj4Qf-s_e2` | 1:09:00-1:11:00 | 2.0 | transit | 0.95 | 0.50 |
| `UEV2mj4Qf-s_e5` | 2:22:00-2:25:00 | 3.0 | density | 0.96 | 1.00 |
| `UEV2mj4Qf-s_e15` | 4:36:00-4:39:00 | 3.0 | density | 0.96 | 1.00 |
| `UEV2mj4Qf-s_e17` | 4:57:00-5:00:00 | 3.0 | density | 0.96 | 1.00 |
| `UEV2mj4Qf-s_e8` | 3:03:00-3:09:00 | 6.0 | both | 0.98 | 1.00 |
| `UEV2mj4Qf-s_e16` | 4:42:00-4:52:00 | 10.0 | density | 0.99 | 0.60 |
| `UEV2mj4Qf-s_e10` | 3:34:00-3:46:00 | 12.0 | both | 0.99 | 0.75 |
| `UEV2mj4Qf-s_e9` | 3:16:00-3:29:00 | 13.0 | density | 0.99 | 0.46 |
| `UEV2mj4Qf-s_e3` | 1:51:00-1:58:00 | 7.0 | density | 1.00 | 0.00 |
| `UEV2mj4Qf-s_e4` | 2:03:00-2:16:00 | 13.0 | density | 1.00 | 1.00 |
| `UEV2mj4Qf-s_e11` | 3:54:00-4:02:00 | 8.0 | density | 1.00 | 0.50 |
| `UEV2mj4Qf-s_e12` | 4:04:00-4:11:00 | 7.0 | density | 1.00 | 1.00 |
| `UEV2mj4Qf-s_e13` | 4:17:00-4:24:00 | 7.0 | density | 1.00 | 0.86 |
| `UEV2mj4Qf-s_e14` | 4:26:00-4:34:00 | 8.0 | both | 1.00 | 0.88 |

## C. Stage 2 over the 13 registered meetings (2026-09-24, $0.9015)

`msgbatch_01HwPvmjTfwNjuw2T8f5D5oo`, 318 windows, 318/318 succeeded, 0 errored,
**$0.9015 against a $1.06 estimate** (0.85x -- the first English batch whose
estimate was not low). Instrument `prompt_version
t0-a4a8e55760/claude-sonnet-4-6/swagit`: the codebook is unchanged, but a Swagit
voice-to-text transcript is a different transcriber from a caption track, and
T2-PLAN.md forbids pooling routes, so the CALLS are named apart exactly as the
Spanish ones are (`classify.ROUTE_SUFFIX`). `asr_local` is deliberately NOT in
that map -- its rows were all measured under the bare hash and naming it now
would re-bill every one of them.

| | Swagit route, 13 meetings |
|---|---|
| audio | 57.05 h |
| windows | 318 (102 about_density, 68 about_transit, 55 about_both, 69 mention_only, 24 no) |
| **located episodes** | **80, 814.0 min** -- 20 density / 142.0, 37 transit / 228.0, 23 both / 444.0 |
| mention rows | 58, 206.0 min |
| per meeting / per h | 6.15 episodes, 1.40 episodes/h, 14.3 about-min/h (23.8 % of meeting time) |
| `contains_instructions` | **0 of 318** -- no window flagged instruction-like text |

### The head-to-head this was meant to test could not be run

The 13 meetings this batch read and the 3 meetings that already have YouTube
captions are **disjoint sets**. `swagit.register` only registers a clip for a
meeting that needs one, so 2026-07-20, 2026-08-03 and 2026-09-14 have a YouTube
row and **no Swagit row** -- and of those three only `UEV2mj4Qf-s` has ever been
classified. So nothing here re-measures the 0.914 candidate recall of section A
or the 18/18 of section B; those remain measured on 3 meetings, n=1 for episodes.

What can be compared is the RATE, and it is in the right place but lower:

| | episodes/h | about-min/h | share of meeting |
|---|---|---|---|
| English caption route, `UEV2mj4Qf-s` (7.27 h) | 2.48 | 18.2 | 30.3 % |
| Swagit route, 13 meetings (57.05 h) | 1.40 | 14.3 | 23.8 % |

6.15 episodes per meeting against Petaluma's 8.00 per captioned meeting measured
on YouTube (0.77x). That is consistent with a recall a little under 1.0, but it
is **not a recall measurement**: different meetings, and the YouTube side is one
meeting. The honest test is cheap and needs **no YouTube request** --
`D5PLaHE9MMk` and `U1Pva1fkJRw` already have their caption tracks on disk and
are still `captioned`, and the Swagit clips 394014 / 395082 / 400885 were
already fetched for section A. Classifying both tracks of all three meetings
would put a real per-episode recall on this route. Registering those three
Swagit twins widens what the route covers, so it is a DECISIONS.md question,
not something this session should have done.
