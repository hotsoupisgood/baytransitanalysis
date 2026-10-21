# Full-ASR locate, codebook v2.1 — the Granicus and Swagit bodies (2026-10-18)

Written by the supervisor agent during the run; approved by Puck on WhatsApp
2026-10-18 10:13 PDT ("full ASR v2.1 on the Granicus and Swagit bodies").
Status at the bottom is the live one. Nothing committed; `plan/frame.csv`,
`plan/DECISIONS.md`, `.env` and every v1 row untouched; no YouTube contact.

## 1. What this is

DECISIONS-PROPOSED-13 option 2, run under the instrument of PROPOSED-14 for
the half of the corpus whose audio does not come from YouTube: every regular
meeting of the Granicus bodies (richmond, sanleandro, santaclara, hayward,
napa, oakland, fremont, sausalito, samtrans) and the Swagit body (petaluma),
within the study window, is

1. transcribed WHOLE on the GPU once, **without diarization** (the b35
   validation configuration: WhisperX `large-v3`, wav2vec2 alignment, the
   body's `initial_prompt` vocabulary, `diarize: False`; `videos` rows
   `fa-<meeting>` of kind `validation`; outputs `output/fullasr/whisper/<meeting>.json`,
   never `output/whisper/`);
2. read by Sonnet under **codebook v2.1** (`v2.1-9e052492e4`,
   `prompt_version v2.1-9e052492e4/claude-sonnet-4-6/mr120`) on **every 120-s
   window** with the pipeline's ± 30 s pad, **no keyword screen**, through the
   Message Batches API (batch note `fullasr-v2.1`, call rows under
   `video_id = 'fullasr:<meeting>'`);
3. merged into `episodes` rows with **`method = 'fullasr'`** (a new route value,
   SCHEMA.md; ids `_f<n>` located, `_fm<n>` mention) exactly as
   `classify.merge_episodes` merges (gap 90 s, mixed labels → about_both);
   a span already covered by a transcribed clip of any route is **kept** — the
   fullasr row is written `done` and points at that transcript (PROPOSED-14 §4)
   — and only the uncovered piece ≥ 60 s is cut;
4. cut offline from the whole-meeting audio on disk (the `local_wav` path
   generalised: core ± 30 s, snapped to the WhisperX segment edges, `-c copy`,
   `register_clip` → `output/opus/<id>.opus`, in-frame `transcoded` row,
   `raw_meta.cut_source = 'fullasr_audio'`), then a normal diarized episode
   batch (`gpu_batch.py propose --speaker <bodies>`).

Driver: `scripts/fullasr.py` (`scope · fetch · encode · reuse · prepare ·
harvest · windows · submit · collect · derive · propose · report`). State and
logs: `output/fullasr/{scope.json, gpu_batches.json, derived.json, report.md,
logs/}`.

## 2. v2.1: frozen and checked first

`plan/CODEBOOK-V2-PILOT.md` §10. One sentence (Puck's rule, 09:36 PDT) at the
top of v2's transit definition, everything else byte-identical; its own hash;
a Spanish block (`classify.LANG_NOTE_V21`) for the interpreter tracks. Checked
on the pilot's 111 windows ($0.37): 6 of 111 labels changed, 5 scored "away"
from the adjudicated reference — and a **v2-on-v2 replicate** ($0.28) changed
6 too. The sentence is indistinguishable from the instrument's own ~5 %
run-to-run churn on 120-s windows; v2.1 was cleared on that basis, with its
recall/precision quoted as v2's ± the replicate spread.

## 3. Scope

Source: `videos` where `kind = 'meeting' AND platform != 'youtube'` (the regex
in `plan/channels.tsv` already keeps regular meetings only, DECISIONS #19).
Excluded and listed: the two Richmond meetings whose b35 GPU transcript failed
QA (decoder loop, coverage 5 %: `gr-richmond-6033`, `gr-richmond-6038`), and
"meetings" shorter than 10 minutes (stub recordings of cancelled or
closed-session items: 10 of them, see `scope.json`). Sausalito and SamTrans
rows carry no duration (`listed_duration` empty on their listings); their
hours are measured from the audio once fetched.

Audio sources, in the order tried: whole-meeting wav already on disk from the
Mac-ASR run (`output/t2/asr/<id>.wav`: Richmond 31, San Leandro 27, Hayward
24, 1 Oakland, 1 Santa Clara); the four b35 whole-meeting transcripts
(reused, not re-bought); the progressive **MP4** on `archive-video.granicus.com`
(`raw_meta.mp4`; measured 85–136× real time); for Fremont, SamTrans and the
five Hayward meetings whose player lists HLS only, the **ASX lookup**
(`granicus.fetch_archive_mp4`, one extra request) returns a progressive MP4
on the same host (probed 2026-10-18 20:38 UTC on one meeting of each; the
listings carry the string `'None'` for mp4, which the first version of the
driver took for a URL); **HLS** on `archive-stream.granicus.com`
(`raw_meta.m3u8`; ~6× real time, per-segment latency) only as the fallback and
for Petaluma's Swagit streams. One ffmpeg at a time, ≥ 5 s between requests (`granicus.polite_wait`),
stop on 403/429; the master on disk is mono 64 kbps opus (a 16 kHz FLAC of
council audio measured 112 MB/h — no compression — and would not fit the
disk), the GPU copy is the 16 kHz / 16 kbps opus of DECISIONS #4.

The live scope / transcription / episode table is **generated**, not maintained
by hand: `.venv/bin/python scripts/fullasr.py report` rewrites
`output/fullasr/report.md` from the DB and `derived.json`, and that file is the
authority for §3, §5 and §6 together (one table carries scope, GPU, v2.1 and
v1 columns). Snapshot at 01:54 UTC 2026-10-19, after b41 harvest and rescue and
before the 5,989-window Sonnet batch landed — so the "v2.1 read" column still
shows only b35+b39+b40, while "transcribed (GPU)" already includes b41:

| body | meetings | hours | audio on disk | transcribed (GPU) | excluded | v2.1 read | v2.1 located (h, merged windows, pre-trim) | v2.1 episode rows (h: new cuts + covered) | v1 episodes (h, all v1 routes) | v1 transcribed (h) |
|---|---|---|---|---|---|---|---|---|---|---|
| fremont | 24 | 49.2 | 24 | 0 (0.0 h) | 0 | 0 (0.0 h) | 0 (0.00) | 0 (0.00) | 42 (3.35) | 37 (3.77) |
| hayward | 29 | 68.5 | 29 | 22 (55.6 h) | 2 | 22 (55.6 h) | 70 (7.50) | 59 (2.58) | 60 (7.00) | 56 (7.81) |
| napa | 30 | 104.3 | 25 | 10 (40.5 h) | 6 | 0 (0.0 h) | 0 (0.00) | 0 (0.00) | 55 (10.20) | 52 (10.95) |
| oakland | 12 | 40.8 | 12 | 1 (4.0 h) | 0 | 0 (0.0 h) | 0 (0.00) | 0 (0.00) | 12 (0.83) | 10 (0.95) |
| petaluma | 13 | 57.0 | 0 | 0 (0.0 h) | 0 | 0 (0.0 h) | 0 (0.00) | 0 (0.00) | 101 (14.50) | 80 (14.86) |
| richmond | 31 | 178.9 | 31 | 25 (151.8 h) | 3 | 25 (151.8 h) | 98 (11.30) | 76 (4.50) | 93 (8.07) | 77 (8.63) |
| samtrans | 12 | 32.0 | 12 | 0 (0.0 h) | 0 | 0 (0.0 h) | 0 (0.00) | 0 (0.00) | 7 (2.01) | 7 (2.13) |
| sanfrancisco | 37 | 116.9 | 0 | 0 (0.0 h) | 0 | 0 (0.0 h) | 0 (0.00) | 0 (0.00) | 0 (0.00) | 0 (0.00) |
| sanleandro | 29 | 82.8 | 27 | 5 (16.8 h) | 0 | 5 (16.8 h) | 21 (2.00) | 17 (0.80) | 100 (13.92) | 69 (13.46) |
| santaclara | 40 | 178.4 | 37 | 32 (168.1 h) | 7 | 0 (0.0 h) | 0 (0.00) | 0 (0.00) | 154 (29.26) | 130 (30.04) |
| sausalito | 30 | 52.7 | 15 | 0 (0.0 h) | 1 | 0 (0.0 h) | 0 (0.00) | 0 (0.00) | 10 (6.75) | 10 (6.92) |
| sfmta | 20 | 80.6 | 0 | 0 (0.0 h) | 0 | 0 (0.0 h) | 0 (0.00) | 0 (0.00) | 0 (0.00) | 0 (0.00) |
| **total** | 307 | 1042.3 | 212 | 95 (436.9 h) | 19 | 52 (224.2 h) | 189 (20.80) | 152 (7.88) | 634 (95.89) | 528 (99.51) |

Two notes on reading it. The **hours** column grew from 957.5 to 1042.3 as the
fetch measured SamTrans (0 → 32.0 h) and Sausalito (0 → 52.7 h) off their audio
— their listings carry no duration (§3), so their scope hours are only known
once pulled, and the total will move again as Sausalito finishes. The
**excluded** column grows with QA: 19 now, because the three b41 failures that
the dead-air rescue correctly refused are excluded from the exposure.

## 4. GPU: whole-meeting batches

`gpu_batch.estimate_usd` prices a batch at RTFX 25 (the diarized rate);
whole-meeting ASR without diarization runs at ~58× (b35, b38) and bills
**$0.0102 per audio-hour**, so the estimator overstates these batches ~3×.
Under the first stop line ($9.50 = `RUNPOD_CREDIT_USD` 10 − 0.50 margin) with
$6.57 recorded at the start, the first batch could be at most ~96 h by the
estimator; Puck then topped the credit up to $31.50 (stop line $31.00), which
let b40 take 150 h and the later batches the rest. The estimator was **not**
overridden (the stop line is Puck's, DECISIONS #18).

**Dead-air "decoder loops" (b40).** 15 of b40's 24 Richmond meetings FAILED
the gate's `identical segments spanning > 60 s` rule (`analysis/qa_transcripts.py`)
on runs of `you` / `Thank you.` / `.` / `We'll be right back.` across recesses
and closed sessions: 2–7 segments spanning 2–89 min of dead air, in transcripts
with 39–70 % coverage and 3,900–7,100 words/h otherwise. That is Whisper's
dead-air hallucination, not a decoder loop over speech; the rule's span unit
was chosen for episode clips, where a 60-s run is speech overwritten, and a
5–7 h council meeting has a recess in it as a rule. `fullasr.py rescue <bid>`
re-gates such a file with every run of ≤ 4 identical filler words collapsed
to its first segment and loads it only if (a) the loop rule was the sole FAIL,
(b) the full gate then passes (coverage ≥ 30 %, cps, compression). Real
failures stay failed: `gr-richmond-5903` (coverage 26 %) and `gr-hayward-2458`
(24 %) in b39. The rescued file carries a top-level `rescued` key listing the
runs; the `fa-` row's events row says `rescued by scripts/fullasr.py rescue`;
nothing rescued is written to `output/whisper/` or the corpus (the diarized
clips cut from these transcripts pass the gate on their own). The general QA
gate is unchanged. Rescued meetings per batch are in the GPU table.

| GPU batch | meetings | audio-h | est (RTFX 25x) | est at measured $0.0102/h | actual | GPU-min | status | loaded / rescued (dead air) / QA FAIL |
|---|---|---|---|---|---|---|---|---|
| b35 | 4 | 18.91 | $0.75 | $- | $0.2313 | - | reused | 4 / 0 / 0 |
| b39 | 24 | 59.918 | $1.85 | $0.611 | $0.6533 | 80.0 | done | 22 / 0 / 2 (gr-hayward-2458, gr-richmond-5903) |
| b40 | 26 | 149.994 | $4.5 | $1.53 | $1.2071 | 147.81 | done | 26 / 22 / 0 |
| b41 | 46 | 220.09 | $6.56 | $2.245 | $2.1490 | 263.14 | done | 43 / 3 / 3 (gr-napa-5727, gr-santaclara-2422, gr-santaclara-2534) |
| b43 | 84 | 259.68 | $7.72 | $2.65 | - | - | approved, queued behind the backlog worker's b42 | - |

**b42 is not this run's batch** — it is the backlog worker's 337-clip episode
batch (32.08 h, est $1.03), which took the pod 25 s after b41 tore down.

The estimator overstates this configuration ~3× every time, and the measured
$0.0102/h prediction has been accurate to within 5 % on all three batches
(b39 $0.611 vs $0.6533; b40 $1.53 vs $1.2071; b41 $2.245 vs $2.1490). It was
still not overridden: the stop line is Puck's (DECISIONS #18), and the only
reason 150-260 h batches fit under it is his top-up to $31.50.

## 5. Sonnet: v2.1, no screen

Generated into `output/fullasr/report.md` by `fullasr.py report` (calls, cost,
token split, label histogram, Claude total against the cap). At 01:54 UTC
2026-10-19, before the b41 windows landed: **5,564 calls, $11.8475**; labels
no 4,285 / mention_only 743 / about_density 212 / about_transit 310 /
about_both 14; Claude $69.47 of the $150 cap.

`submit` compares its `--cap-usd` against the **worst case**, not the estimate
(est $16.77 vs worst $36.35 on the b41 windows), so a cap set at the estimate
refuses. At the measured rate this route costs **$0.041 per audio-hour**, so
the remaining ~600 h of scope is ~$25 of Sonnet.

## 6. Episodes: v2.1 (fullasr) against v1 (all v1 routes), per body

The last four columns of the §3 table, per body: `v2.1 located` is merged
windows pre-trim, `v2.1 episode rows` is what survived span-match/supersede
(new cuts + spans already covered by a clip of another route, which are written
`done` and point at the existing transcript, PROPOSED-14 §4), and the two v1
columns are all four v1 routes pooled. **The v1 and v2.1 columns are two
instruments and are never pooled** (§7, finding 3); `sanfrancisco` and `sfmta`
have an empty v1 column because nothing was ever read there under v1, not
because the instrument changed.

## 7. Limitations that still apply (plan/REVIEW-2026-10-18.md, Puck: "exploratory")

- **No human gold set** (finding 1). Every number here is Sonnet against
  Sonnet; the v2.1 check and the v2 replicate are the same instrument read
  twice. Exploratory stamp on every figure until a confirmatory phase.
- **Circular v2 reference** (finding 2): the adjudicated minutes are the ones
  v2 called; v2.1's "away" moves are partly by construction.
- **Construct moved mid-study** (finding 3): v1 rows stay as a separate facet
  and are never pooled; a body read under v1 for its YouTube captions and
  under v2.1 full ASR here has two instruments, reported side by side.
- **Operator boards** (4f, 10): at SamTrans nearly every item is "about
  transit"; board and council rates are different constructs.
- **Transit precision under v2/v2.1 is low on a v1 reference** (0.53–0.76) and
  is definitional (active transportation counts); a v2.1-coded reference
  sample is still the clean test (PILOT §9 caveat).
- **Instrument churn**: ~5 % of 120-s window labels flip between identical
  re-reads (§2); located-minute totals carry that as noise on top of
  everything above. No k-sample voting in the pipeline.
- **Partial last month**, ethics sign-off not recorded, 44+ uncommitted
  files (10).
- **No redaction on this route.** Puck said on 2026-10-18 that names no
  longer need redacting; nothing produced here (whole-meeting transcripts,
  windows, clips, the diarized episode batch) has been through
  `pipeline/redact.py`, and the redaction-recall item of the review (10) no
  longer applies to new material. Earlier redacted material is unchanged.
- **Exposure**: the two excluded Richmond meetings and the stub recordings
  are out of the exposure for this route (finding 10's words/h rule); a
  meeting whose fullasr transcript fails GPU QA is listed in `gpu_batches.json`
  and stays out.

## 8. What full ASR removes, and what it does not

Removed for this half: the **keyword screen** and with it the body-specific
vocabulary bias (finding 5: active transportation at Richmond,
operations/maintenance at boards, affordable-housing allocation elsewhere) and
the caption-homophone layer; locate recall is 1.0 by construction against the
codebook, so the measurement-error term on this route is the classifier's
precision and window granularity only (PROPOSED-13 implementation note). Also
removed: the four locate instruments on these bodies (`classifier` on Granicus
captions, `agenda_index`, `local_asr`, `swagit_transcript`) — one route reads
them all, and the whole-meeting text is on disk as context.

Not removed: findings 1–3 (gold set, circular reference, construct change),
the stance κ gate, the yield-ordered caption queue on the YouTube half, and
the diarized episode pass's own QA.

## 9. What remains for the YouTube half

~385 meetings / ~1,400 h. Whole-meeting audio means one yt-dlp request per
meeting at the 12-per-day cap (≥ 32 days of trickle; two 429s this month at
14–18/day; `plan/DECISIONS-PROPOSED-13.md` option 2 costs) — or the screened
route stays there and its recall is measured per route (option 1), in which
case the corpus has two locate instruments and every rate is reported per
`method`. GPU ≈ $14 at $0.0102/h for 1,400 h; Sonnet ≈ $57 at $0.041/h.
Either way the v1 → v2.1 re-read of the already-screened YouTube windows
(PROPOSED-14 re-read plan, ≈ $18) is independent of this half.

## 10. Files

New: `scripts/fullasr.py` (second session added `rescue <bid> [--dry-run]`,
the ASX-before-HLS source order, and a `'None'`-string guard on `raw_meta.mp4`),
`scripts/codebook_v21_check.py`, this file,
`output/fullasr/` (audio/, opus/, whisper/, batches/, windows/, clips/, logs/,
scope.json, gpu_batches.json (`rescued` / `still_failed` per batch), derived.json, report.md;
logs/wait_batch.sh polls a batch to completion, logs/probe_asx.py is the MP4 probe),
`output/validation/codebook_v2/v21_check/`, `output/validation/codebook_v2/v2_replicate/`.
Changed: `pipeline/classify.py` (INSTRUCTIONS_V21, CODEBOOKS / CODEBOOK_VERSIONS
'v2.1', LANG_NOTE_V21 + `lang_note()`, `--codebook v2.1`),
`scripts/codebook_v2_adjudicate.py` (scores a v21 arm when present),
`plan/SCHEMA.md` (episodes.method 'fullasr'), `plan/DECISIONS-PROPOSED-13.md`
(implementation note), `plan/CODEBOOK-V2-PILOT.md` (§10). DB: `videos` rows
`fa-<meeting>` (kind validation), `batches` rows for the whole-meeting
batches, `classifier_batches` rows (status `fullasr_v21`, `codebook_v21_check`,
`codebook_v2_replicate`), `classifier_calls` under `fullasr:`, `cbv21:`,
`cbv2rep:`, `episodes` rows `method='fullasr'`, clip rows `<meeting>_f<n>`.
`make test`: 485 checks, all passed after the classify.py change.

## 11. Scope change: the two San Francisco bodies

`sanfrancisco` (37 meetings, 116.9 h) and `sfmta` (20, 80.6 h) are Granicus
rows that landed 2026-10-18 20:57 UTC from a sibling crawl, after the ten
bodies of §3 were scoped. They were added to this run's scope on the evening
of 2026-10-18 and sit **last** in `BODY_ORDER` (their audio is not on disk and
neither body has anything read under v1, so they contrast with nothing — their
v1 column is empty by construction, not by instrument change). `meetings()`
previously excluded them explicitly; that docstring and `BODY_ORDER` were
changed. Everything else about the route is unchanged. Total scope is now
**307 meetings / 957.5 h** across 12 bodies.

## 12. Status

Live log, newest last. Third supervisor session; the two before it died (one
session limit, one oversized report) and this section exists so their work is
not re-derived.

- **01:36 UTC 2026-10-19** (handover state): b41 RUNNING, 46 meetings /
  220.09 h, pod `my26p0dihxlcnf`. b40 done, 150 h, $1.2071, **rescue already
  applied** (26 loaded / 22 rescued, 0 still failed). b39 done, 59.9 h,
  $0.6533, 22/24. 137 whisper transcripts on disk. RunPod $8.43 of $31.50
  (stop $31.00); Claude $69.47 of $150.
- **01:40 UTC**: inherited processes identified and left running —
  `gpu_batch.py launch b41` (pid 62078, the pod monitor) and
  `fullasr.py fetch --bodies hayward,fremont,samtrans,petaluma` (pid 74408,
  mid-SamTrans, pulling MP4 at 160–210x). b41 at 37 done / 5 QA FAIL /
  4 transcribing. 52 GiB free on the data volume.
- **01:43 UTC**: `gpu_batch.py reconcile b41 --dry-run` — pod `my26p0dihxlcnf`
  (`ytpt-b41`) RUNNING, uptime 252.8 min, one live pod only, nothing orphaned.
  At $0.49/h that is ~$2.06 billed against the $6.56 estimate. b41 is on its
  last 3 (longest) meetings. There is no `pods` subcommand on `gpu_batch.py`;
  `reconcile <bid> --dry-run` is the read-only pod check (it lists the live
  pods and changes nothing — `--terminate` is the destructive form).
- **01:43 UTC**: next batch sized. `prepare --hours 230 --dry-run` →
  **72 meetings / 229.45 audio-h**, est $6.83 by the RTFX-25 estimator
  (~$2.34 at the measured $0.0102/h): richmond 3, sanleandro 22, hayward 5,
  napa 14, oakland 11, fremont 17. It passes the stop line without any
  override — Puck's top-up to $31.50 is what made the 150 h and 220 h batches
  legal, not a changed estimator (§4). Santa Clara is absent from that pack
  because its 28 meetings are *inside b41*: their `fa-` rows are already
  `done` and only the harvest into `whisper/` is outstanding, so `prepare`
  correctly skips them. A naive "opus on disk and no transcript" tally
  double-counts b41 and reads 435 h; the real ready figure is 229 h.
- **01:41 UTC**: SF audio fetch queued behind the running one
  (`output/fullasr/logs/fetch_sf.sh`, waits on the `fullasr.py fetch` pid, then
  `--bodies sanfrancisco,sfmta`) so only ever one ffmpeg runs.
- **01:54 UTC — b41 done.** 220.09 audio-h, **263.14 GPU-min, actual $2.149
  against a $6.56 estimate** (the measured-rate prediction was $2.245, within
  5 %: the RTFX-25 estimator still overstates this configuration ~3×).
  40 harvested on the first pass, 6 QA FAIL, of which **3 rescued** by the
  dead-air rule — `gr-santaclara-2448` (3 filler runs, coverage 81 %),
  `-2560` (14 runs, 76 %), `-2577` (13 runs, 77 %). **3 stay failed and are
  out of the exposure**: `gr-napa-5727` (coverage 14 %), `gr-santaclara-2422`
  (6 %), `gr-santaclara-2534` (23 %) — all three fail on coverage as well as
  the loop rule, so the rescue correctly refused them. Final b41: **43 loaded
  / 3 rescued / 3 failed**. Cumulative GPU for this route: 450 audio-h for
  **$4.24** over b35+b39+b40+b41. RunPod now $10.58 of $31.50 (stop $31.00).
- **01:56 UTC — Sonnet.** `submit` first REFUSED at `--cap-usd 30` (its own
  worst-case gate: est $16.77, worst $36.35 — the cap is compared against the
  *worst* case, not the estimate). Re-submitted at `--cap-usd 40`:
  **`msgbatch_01U15PXvEWCLPe1BSLJvYzSS`, 5,989 windows over 43 meetings**,
  est $16.77, `in_progress`. Claude $69.47 before it; ~$86 expected after,
  against the $130 warn line and the $150 cap.
- **01:54 UTC — the pod went to the backlog worker, not to me.** 25 seconds
  after b41's teardown, the sibling's **b42** (337 episode clips, 32.08 h,
  est $1.03, pod `omz6t2xwnf3kar`) started. The alternation Puck asked for
  happened on its own; this run did not contend for the pod. My next
  whole-meeting batch is therefore **b43**, not b42.
- **01:56 UTC — b43 registered and approved, launch queued behind b42.**
  84 whole meetings / **259.68 audio-h**, est $7.72 (measured-rate prediction
  $2.65), `awaiting_approval` → approved `--via "fullasr (Puck, WhatsApp
  2026-10-18 10:13 PDT)"`. Bodies: richmond 3, sanleandro 22, hayward 5,
  napa 14, oakland 11, fremont 24, sausalito 5 — i.e. everything whose audio
  is on disk and untranscribed, now that the fetch has finished SamTrans and
  part of Sausalito. The launch is **not** run directly:
  `output/fullasr/logs/launch_b43_when_free.sh` waits for b42 to leave
  `running`, re-checks that RunPod reports **no** live pod, and aborts rather
  than launching if one is still billing. One pod at a time is enforced by the
  script, not by timing.

- **02:16 UTC — the b41 Sonnet read is in, and the episodes are derived and
  cut.** `msgbatch_01U15PXvEWCLPe1BSLJvYzSS` ended ~02:04 and `collect`
  (auto-derive) ran to 02:16. This route is now at **11,553 calls / $21.0755**
  — the 5,989-window batch cost **$9.23 against a $16.77 estimate**, so the
  measured rate is holding near $0.041/audio-h. Labels over the whole route:
  no 8,670 / mention_only 1,467 / about_density 829 / about_transit 519 /
  about_both 68. Claude total ~$78.7 of the $150 cap (warn line $130).
- **02:16 UTC — episodes, method `fullasr`.** 110 cut + 79 done + 121
  transcribing + 898 mention. The span-match/supersede rule of §1.3 is doing
  exactly what PROPOSED-14 §4 asks, visibly in the log: a v2.1 span that lands
  on an already-transcribed v1 clip is **kept, not re-cut** ("nothing uncovered
  >= 60 s, kept the v1 transcript"), and where only part is uncovered it cuts
  only that part ("overlaps transcribed `gr-santaclara-2585_e2`; cut only
  0:19:00-0:20:00"). No meeting was re-finished mid-cut.
- **02:08 UTC — one cut failed its own gate.** `gr-santaclara-2447_f11`
  (5:30:00-5:32:00): "REFUSED: opus verify failed: ffprobe cannot read the
  opus file". The episode row is `failed`, no clip was registered and nothing
  downstream points at it — 1 of 111 cuts. Not retried in this session; it is
  120 s of one Santa Clara meeting and it should be re-derived rather than
  hand-cut.
- **02:05 UTC — the backlog worker is already running my clips.** b42 contains
  **121 `fullasr` episode clips** alongside its own backlog, which is the
  "whoever has the pod runs the other's ready clips too" rule working in this
  run's favour without either side having to ask. 110 more clips / 7.89 h are
  transcoded and waiting — just under `propose`'s 8 h floor, and `propose`
  refuses while a pod is live anyway.

- **02:20 UTC — the tail is automated, with the launch count bounded.**
  `output/fullasr/logs/supervise.sh` (log `supervise.log`) runs one pass a
  minute for up to ~14 h and carries the remaining stages: collect any finished
  Sonnet batch (which auto-derives and cuts), harvest + rescue any whole-meeting
  batch that finished unharvested, submit anything transcribed but unread, and
  — only when RunPod reports **no live pod** and no batch is `running` or
  `approved` — start the next batch. It is deliberately **not** an open-ended
  pod spawner: it may start **at most one further whole-meeting batch** after
  b43 and **one** diarized episode batch, then it stops launching. Every money
  and safety gate stays where it already lived (`prepare` on the stop line,
  `propose` refusing while a pod is live and passing `--speaker` with only the
  12 in-scope bodies, `rescue` refusing coverage failures, `submit` on its own
  worst-case cap); the supervisor only adds a second no-live-pod check
  immediately before each launch and aborts if a pod appeared in between.
  Two helpers it needs: `logs/q.py` (one-value SQL, so quoting in SQL literals
  cannot break the shell — the first version of the loop interpolated SQL into
  `python -c` and every `status='running'` literal raised SyntaxError; it
  fail-safed to "no launch", but it also read `live_pods=` as empty) and
  `logs/unharvested.py` (finished batches whose transcripts are not all in
  `whisper/` yet, skipping the reused b35).
- Also armed: `logs/launch_b43_when_free.sh`, which owns b43's launch alone —
  the supervisor sees b43 as `approved`, counts it as busy, and keeps away.

### What remains (02:20 UTC 2026-10-19)

1. **b42** (sibling, 337 clips incl. 121 of mine) finishes → **b43** launches
   itself (84 meetings / 259.68 h, ~4.5 h GPU by b41's measured rate) →
   `after_b43.sh` harvests, rescues the dead-air FAILs and submits the ~6,500
   new windows to Sonnet (`--cap-usd 60`).
2. **Audio still to pull**: petaluma 13 meetings / 57.0 h (Swagit HLS, ~6×
   real time, so ~10 h of wall clock — it is on its first meeting),
   sanfrancisco 37 / 116.9 h and sfmta 20 / 80.6 h (queued behind it,
   `fetch_sf.sh`), plus the Sausalito tail and 4 Santa Clara / 5 Napa /
   2 San Leandro with no audio yet. That is ~260 h of scope not yet on disk,
   needing **one further whole-meeting batch after b43** (~$8 by the
   estimator, ~$2.7 at the measured rate) and ~$11 of Sonnet.
3. **The diarized episode batch** for the remaining fullasr clips, once the
   pod is free after b43 and the clips clear 8 h: `fullasr.py propose`
   (which calls `gpu_batch.propose --speaker <the 12 in-scope bodies>` — never
   hasanabi/tucker), then `make qa`.
4. **Budgets**: RunPod $10.58 of $31.50 (stop $31.00); b43 + one more
   whole-meeting batch + the episode batch ≈ $5 at measured rates, so the stop
   line is not in play. Claude ~$78.7 of $150; the rest of the route is ~$11
   of Sonnet, landing near $90 — **under the $130 warn line**, so Puck does
   not need to be asked about Sonnet spend for this route.

