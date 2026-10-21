# PIPELINE-SCREENED — restart runbook for the captions-first (keyword-screened) locate pipeline

Written 2026-10-18 by the supervisor agent, read-only against the code and
`state.sqlite` on that day. Purpose: the Granicus/Swagit bodies are moving to
full-ASR locate (codebook v2.1, no keyword screen; `docs/FUTURE-STEPS.md` 1).
The screened pipeline below is what produced every `episodes` row so far and
must stay restartable: for the YouTube bodies, for a cheap re-screen after a
keyword change, or if the full-ASR budget is withdrawn. A future agent should
be able to bring it back up in under an hour from this page alone.

Numbers marked *live* were read from `state.sqlite` / the code on
2026-10-18; everything else cites the file it came from. Where this page
disagrees with `START_HERE.md` or `README.md`, the code wins and this page
was written later.

---

## 1. What it is

Locate before transcribing (`plan/DECISIONS.md` #13): a cheap locate track
finds candidate minutes, a pinned Claude classifier says which are *about*
density / transit, and only those minutes are cut and sent to the GPU with
diarization. The screen is the instrument's recall floor; its vocabulary is
data (`plan/keywords_*.txt`) and every row records which list and which
codebook produced it.

```mermaid
flowchart TD
    D[pipeline/discover.py --meetings<br/>plan/channels.tsv, 33 bodies<br/>videos.kind=meeting, status=discovered] --> YT[pipeline/captions.py --loop<br/>com.ytpt.captions · YouTube auto-captions<br/>1 req/h, 18 per 24 h · captions=auto|manual]
    D --> GR[pipeline/captions.py<br/>Granicus caption track incl. Flowplayer object<br/>captions=granicus]
    D --> SW[pipeline/swagit.py<br/>Petaluma word-timestamped transcript<br/>captions=swagit]
    D --> AI[pipeline/agenda_index.py<br/>Granicus index points, title match<br/>method=agenda_index, no classifier]
    D --> LA[scripts/local_asr.py / asr_batch.py<br/>Mac mlx-whisper on ASX.php MP4<br/>captions=asr_local]
    YT --> KS
    GR --> KS
    SW --> KS
    LA --> KS
    KS[pipeline/segment.py keyword screen<br/>keywords_density + keywords_transit (+ _es)<br/>+ keywords_aliases_auto.txt on captions=auto<br/>60-s windows, smooth3 >= 0.5 hits/100 w, runs cut to --max-run 120 s<br/>KeywordList.version kw-&lt;hash&gt;[/al-&lt;hash&gt;]]
    KS --> CL[pipeline/classify.py run --batch<br/>claude-sonnet-4-6 via Message Batches<br/>--codebook v1 t0-a4a8e55760 (default) | v2 | v2.1<br/>prompt_version e.g. t0-a4a8e55760/claude-sonnet-4-6/mr120]
    CL --> EP[classify.write_episodes<br/>about_* runs -> episodes _e&lt;n&gt; status=located<br/>mention_only -> _m&lt;n&gt; status=mention<br/>method = classifier | local_asr | swagit_transcript]
    AI --> EP2[episodes _a&lt;n&gt; status=located<br/>method=agenda_index]
    EP --> CUT[pipeline/episodes.py<br/>whole YouTube audio once, or HLS slice, or local wav<br/>core ± 30 s pad, cue-snapped · clip row kind=episode -> transcoded]
    EP2 --> CUT
    CUT --> GB[pipeline/gpu_batch.py propose --speaker &lt;bodies&gt; / approve / launch<br/>>= BATCH_TARGET_HOURS 15 h · Puck approves on WhatsApp]
    GB --> GPU[pipeline/gpu_job.py on RunPod A40<br/>WhisperX large-v3 + pyannote diarization]
    GPU --> QA[make qa · analysis/qa_transcripts.py<br/>a FAIL is never loaded]
    QA --> RD[pipeline/redact.py run --all -> output/redacted]
    RD --> LD[make load-episodes · analysis/load_episodes.py --qa-gate<br/>corpus.sqlite, per method / prompt_version facets]
```

Stage-by-stage, with the code that defines it:

| stage | script | reads | writes |
|---|---|---|---|
| discover | `pipeline/discover.py --meetings [--bodies b1,b2] [--since 2025-09-20 --until 2026-09-20] [--from-listings DIR]` | `plan/channels.tsv` | `videos` rows `kind='meeting'`, `status='discovered'`, `sources` |
| caption track (YouTube) | `pipeline/captions.py --loop` (launchd) or `--max N` | `videos.status='discovered'`, `backoffs`, `worker_requests` | `output/subs/<id>.en.vtt` + `info.json`; `videos.captions` (`auto`/`manual`/`none`), `captions_path`, `status='captioned'`; `worker_requests` row per request; `backoffs` row on a bot check |
| caption track (Granicus) | same worker, `--platform granicus`; `--refetch --bodies X` after a parser fix | `discovered` (or `no_episodes`/`captioned` with `--refetch`) | `output/subs/<id>.player.json`, `.en.vtt`; `captions='granicus'` or `'none'` |
| Swagit transcript | `pipeline/swagit.py register --body petaluma [--plan]` | PrimeGov listing + player | a separate `videos` row per meeting, `captions='swagit'`, `status='captioned'`; the YouTube twin stays `discovered` and is skipped by the captions queue |
| agenda index | `pipeline/agenda_index.py [--bodies X] [--plan] [--report]` | `player.json` index points, keyword families | `episodes` `_a<n>`, `method='agenda_index'`, `label` NULL; meeting -> `classified` |
| local ASR | `scripts/local_asr.py --register ...`, `scripts/asr_batch.py` | ASX.php MP4 -> `output/t2/asr/<id>.wav` | `captions='asr_local'`, `captions_path`, `status='captioned'` |
| keyword screen + classifier | `pipeline/classify.py run --batch [--codebook v1] [--max-run 120] [--bodies X | --ids ...] [--include-classified] [--include-no-episodes] [--dry-run]`; then `pipeline/classify.py collect` | `status='captioned'` (+ `classified`/`downloaded`/`cut` with `--include-classified`; + `no_episodes` with `--include-no-episodes`), `captions_path`, cached `classifier_calls` | `classifier_calls` (one row per window per `prompt_version`, `keywords_version`), `classifier_batches`; `episodes` `_e`/`_m` rows; meeting -> `classified` or `no_episodes` (or `cut` -> `downloaded` when a cut meeting gains located rows) |
| cut | `pipeline/episodes.py [--plan] [--max N] [--bodies X]`; `--requeue-failed` | meetings `classified`/`downloaded`, or `cut` with a `located` episode; `episodes.status='located'` | `output/audio/<id>.opus` (YouTube, kept), `output/opus/<id>_e<n>.opus`; clip `videos` row `kind='episode'` walked to `transcoded`, `in_frame=1`; `episodes.status='cut'`; meeting -> `cut` |
| GPU | `pipeline/gpu_batch.py propose --speaker <bodies>` -> WhatsApp `approve bN` -> `gpu_batch.py approve bN` -> `gpu_batch.py launch bN` (not on the exec allowlist; OpenClaw asks Puck) | clip rows `in_frame=1 AND status='transcoded'`, `spend`, `pods` | `batches`, `batch_videos`, clip rows `queued` -> `transcribing` -> `done`; `episodes.status` follows; `output/whisper/<episode_id>.json` |
| QA | `make qa` | `output/whisper`, `output/subs`, `episodes.start_s` for the caption window | `output/qa/latest.csv` |
| redact | `.venv/bin/python pipeline/redact.py run --all` | OK/WARN transcripts in the QA snapshot | `output/redacted/` |
| load | `make load-episodes` (`TRANSCRIPTS=output/redacted QA_GATE=output/qa/latest.csv`) | redacted JSON, `state.sqlite` episodes/roles | `corpus.sqlite` |

The screened path is **the default of every script above**. There is no
`--screen` switch because, as of 2026-10-18, there is no production
full-ASR switch either: `grep -rn fullasr pipeline/` finds nothing.
Full-ASR locate exists only as pilot tooling (`scripts/recall_fullasr_v2.py
windows|submit|collect|analyze`, which reads a whole-meeting WhisperX JSON
and sends every 120-s window with no screen, writing `classifier_calls`
rows under `video_id='recallv2:<id>'` and **no** `episodes` rows). When the
production full-ASR route lands it will presumably add an `episodes.method`
value (the sibling plan calls it `fullasr`) and a flag or script of its own;
until then "screened vs fullasr" is "which script you ran", and the
selectors below are the whole story.

---

## 2. Starting and stopping each stage

### launchd agents (installed from `ops/launchd/`, logs in `~/Library/Logs/ytpt/`)

| label | what | start / stop |
|---|---|---|
| `com.ytpt.captions` | `captions.py --loop`: the YouTube caption trickle. `KeepAlive SuccessfulExit=false`: a bot-check exit (0) leaves it **down** on purpose | install/reinstall: `./ops/launchd/install_workers.sh com.ytpt.captions`; pause: `launchctl bootout gui/$(id -u)/com.ytpt.captions`; resume: `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ytpt.captions.plist` or `launchctl kickstart -k gui/$(id -u)/com.ytpt.captions` |
| `com.ytpt.downloader` | `downloader.py --loop --concurrency 1`: **show-corpus** downloads (`in_frame=1`, `kind='show'`). Not part of the meeting path, which fetches whole audio inside `episodes.py`. Leave it down unless the frozen corpus needs something | same pattern with `com.ytpt.downloader` |
| `com.ytpt.transcoder` | `transcode.py --loop`: opus-encodes `downloaded` in-frame rows. The meeting path registers clips already `transcoded`, so this is also show-corpus only | same pattern |
| `com.ytpt.dashboard` | `dashboard/serve.py`, http://localhost:8765, read-only | `./ops/launchd/install_dashboard.sh`; `launchctl print gui/$(id -u)/com.ytpt.dashboard \| head` |
| `com.ytpt.resume-captions` | **one-shot, self-deleting** (`ops/launchd/resume_captions_once.sh`). Fired 2026-09-25; the plist in `ops/launchd/` is the template. Never make it standing | do not install without Puck's explicit per-incident approval |

*Live 2026-10-18 10:20 PDT:* `launchctl list` shows downloader, transcoder,
dashboard running and `com.ytpt.captions` loaded but not running (it exited
0 on the 2026-10-17 429; backoff row until 2026-10-18 19:06 UTC).

### One-shot stages (no launchd; run from `~/ytpt` with `.venv/bin/python`)

```bash
PY=.venv/bin/python
$PY pipeline/captions.py --backoff-status            # read first, always
$PY pipeline/captions.py --rate-status               # requests in the last 24 h vs the cap
$PY pipeline/captions.py --yield-table               # queue order, no network
$PY pipeline/captions.py --dry-run --max 5           # plan, no network
$PY pipeline/captions.py --platform granicus --bodies sausalito --max 10
$PY pipeline/captions.py --refetch --bodies santaclara,hayward --dry-run

$PY pipeline/classify.py run --dry-run --bodies bart       # estimate, no call
$PY pipeline/classify.py run --batch  --bodies bart        # submit to Message Batches
$PY pipeline/classify.py collect                           # poll + write labels, episodes
$PY pipeline/classify.py report

$PY pipeline/episodes.py --plan                            # what would be fetched/cut
$PY pipeline/episodes.py --max 3 --bodies bart             # one YouTube request per meeting

$PY pipeline/gpu_batch.py list
$PY pipeline/gpu_batch.py propose --speaker bart,losaltos,milpitas   # see §4
$PY pipeline/gpu_batch.py approve b41      # only after Puck's WhatsApp "approve b41"
$PY pipeline/gpu_batch.py launch b41 --dry-run && $PY pipeline/gpu_batch.py launch b41
make qa && $PY pipeline/redact.py run --all && make load-episodes
```

### Every flag / env name that selects the screened behaviour, with its default

Settings come from `.env` through `state.setting(key, default)`; flags
override per run. Nothing here is required: a bare run is the screened path.

| name | where | default | effect |
|---|---|---|---|
| `--codebook` | `classify.py` | `v1` (`t0-a4a8e55760`) | `v2` = `v2-ed4c72c9d6`, `v2.1` = `v2.1-9e052492e4` (*live*, uncommitted in the working tree). Any non-v1 codebook always names its model in `prompt_version` |
| `--max-run` | `classify.py`, `segment.py` | **120** (`classify.DEFAULTS`; `PV_MAX_RUN` 240 is what the *bare* v1 hash means) | candidate-run granularity; named `/mr<n>` in `prompt_version` unless 240 |
| `--window` / `--threshold` / `--pad` / `--gap` | `classify.py` | 60 s / 0.5 hits per 100 w / 30 s / 90 s | screen window, smoothed density threshold, clip pad, merge gap |
| `--model` / `SCORING_MODEL` | `classify.py` | `claude-sonnet-4-6` | must be in `classify.PRICES`; any other model is a different instrument |
| `--batch` | `classify.py run` | off (direct API) | Message Batches, 50 % off; then `collect` |
| `--cap-usd` / `CLASSIFY_CAP_USD` | `classify.py` | 5.0 | per-run ceiling |
| `CLAUDE_CAP_USD` | `pipeline/spend.py` | 10.0 | project-wide stop on `classifier_calls` (raised in `.env`; $80 per `docs/FUTURE-STEPS.md`) |
| `--include-classified`, `--include-no-episodes`, `--ids`, `--bodies`, `--max`, `--limit` | `classify.py` | off | which meetings; see §3 and §4 |
| keyword files | `classify.KEYWORD_FILES_BY_LANG` | `plan/keywords_density.txt` + `keywords_transit.txt`; `_es` pair for Spanish | list hash *live* `kw-066fe5a8cc` (English, uncommitted edits in the tree; `kw-24e8a4acd1` is the last documented), `kw-ee372a87db` (Spanish). Rows written before versioning read as `kw-dfd392a1fe` |
| alias layer | `classify.ALIAS_FILES` | `plan/keywords_aliases_auto.txt`, applied only when `videos.captions='auto'` and lang `en` | version *live* `kw-066fe5a8cc/al-7e4c06d52b`; changes what the screen matches, never stored text |
| `CAPTIONS_MIN_GAP_S` / `CAPTIONS_MAX_GAP_S` / `CAPTIONS_DAILY_CAP` (or `--min-gap --max-gap --daily-cap`) | `captions.py` | 3600 / 4200 s / 18 | the trickle; enforced from `worker_requests` so a restart cannot burst |
| `--platform` | `captions.py` | `youtube` | `granicus` for the Granicus player pass |
| `--idle-sleep` | `captions.py --loop` | 300 s | |
| `--min-gap` / `--max-gap` | `episodes.py` | 45 / 180 s | gap between whole-meeting YouTube requests |
| `BATCH_TARGET_HOURS` / `--min-hours` | `gpu_batch.py` | 15 | propose refuses below it |
| `MONTHLY_SPEND_CAP_USD`, `GPU_USD_PER_HOUR`, `GPU_RTFX_ESTIMATE`, `GPU_BOOT_MINUTES`, `GPU_COST_CONTINGENCY` | `gpu_batch.py cfg()` | 150 / 0.34 / 25 / 7 / 1.5 | estimate and cap |
| `RUNPOD_CREDIT_USD`, `RUNPOD_CREDIT_MARGIN_USD` | `spend.py` | 10 / 0.50 | stop line = credit − margin |
| `STATE_DB_PATH`, `SUBS_DIR`, `AUDIO_DIR`, `OPUS_DIR`, `WHISPER_DIR`, `LOCAL_ASR_DIR`, `BATCH_DIR` | `state.py` and workers | `~/.local/share/ytpt/state.sqlite`, `output/subs`, `output/audio`, `output/opus`, `output/whisper`, `output/t2/asr`, `output/batches` | paths; the DB must be on real local disk |
| `TRANSCRIBE_CAP_MIN` | show corpus only | 45 | not applied to episode clips (cap = clip length) |

### State the stages read and write

Meeting rows (`videos.kind='meeting'`, `in_frame=0`):
`discovered -> captioned -> classified -> downloading -> downloaded -> cut`,
with `captioned -> no_episodes` when nothing is about; `cut -> downloaded`
again when a later reading adds `located` rows. *Live:* 276 discovered, 64
classified, 20 downloaded, 146 cut, 129 no_episodes.

Episode rows: `located -> cut -> queued -> transcribing -> done | failed |
capped`; `mention` is terminal and never downloaded. Clip rows
(`videos.kind='episode'`, `in_frame=1`) mirror the clip's progress from
`transcoded` on, which is what `propose` selects.

`backoffs` (worker, until, reason): written by `captions.py` on a 429 / bot
check, read before its first request; **not read by `episodes.py`**.
`worker_requests` (worker, host, at, video_id): one row per YouTube request
by the captions worker, the rolling-24-h cap; **not written by
`episodes.py`**. `classifier_calls` keyed `(id, prompt_version)`: a window
re-read under the same instrument and bounds is served from cache, not
re-billed. `classifier_batches`: `submitted -> ended -> collected`.

---

## 3. Coexisting with full-ASR rows

- **`episodes.method` is the locate route, never the model.** Values *live*:
  `classifier` (caption track read by the classifier), `local_asr`,
  `swagit_transcript`, `agenda_index`. A production full-ASR route must get
  its own value (the plan's name is `fullasr`); do not reuse `classifier`.
- **Never pooled.** `plan/SCHEMA.md` (episodes: n per method /
  prompt_version), `analysis/facets.py` refuses to draw a pooled figure,
  `make a0-pooled` is the stamped escape hatch. v1 rows (`t0-a4a8e55760*`)
  and v2/v2.1 rows are different instruments inside the same route as well
  (`plan/DECISIONS-PROPOSED-14.md`): a body's rate is reported per
  `(method, prompt_version)` and only once every meeting of that body is read
  under that pair.
- **If both routes locate the same meeting** (`classify.write_episodes`,
  lines 855–963): matching is by **core span** (rounded start, end), never
  by id. A span that already has a row keeps that row; it is updated only
  while still `located`/`mention`. A new span overlapping a span already
  past `located` (cut/queued/done) is **trimmed** to its largest uncovered
  piece ≥ 60 s (`MIN_TRIM_S`) or dropped, so the same minutes never land
  twice. A `located`/`mention` row **of the same `method`** that no span of
  the new reading matches is deleted and an `events` row `superseded` is
  written. Rows of a *different* method are never touched: a `classifier`
  reading leaves `local_asr`, `agenda_index` and any future `fullasr` rows
  alone, and vice versa. Consequence: one meeting can carry `_e`, `_a` and
  full-ASR episodes side by side, and the cutter's trim rule, not the
  writer, keeps the audio from being transcribed twice.
- **Reporting per route.** `analysis/load_episodes.py` and
  `analysis/facet_sweep.py` facet on `episodes.method` and
  `prompt_version`; `analysis/wave1_coverage.py` gives meetings discovered
  / located / transcribed per body. Quick check:

```sql
SELECT method, prompt_version, status, COUNT(*), ROUND(SUM(end_s-start_s)/3600,1) h
FROM episodes GROUP BY 1,2,3;
```

---

## 4. Known hazards and the fix for each

| hazard | what happens | fix |
|---|---|---|
| **YouTube 429 / daily budget** | YouTube limits the IP, not the script. 18 caption requests/day held for three weeks; on 2026-10-17 the same 18 **plus 14 whole-meeting audio fetches by `episodes.py`** tripped a 429 at 19:06 UTC. `episodes.py` reads neither `backoffs` nor `worker_requests` | Treat captions + audio as **one budget of about 18 requests per day** (12 audio + up to 6 captions, or 18 captions and no audio). Before any `episodes.py` run: `captions.py --backoff-status` and `--rate-status`; never start it inside a backoff. Pace `episodes.py --max N` by hand |
| **Bot check / "Sign in to confirm" / second 403** | the worker stops, writes a 24-h `backoffs` row, notifies Puck, exits 0 so launchd leaves it down; `episodes.py` exits 3 and puts the meeting back to `classified` | Do nothing until Puck decides (`AGENTS.md` rule 2). Never loop retries, never cookies. After the backoff: `captions.py --backoff-status` says none, then `launchctl kickstart -k gui/$(id -u)/com.ytpt.captions`. `--clear-backoff` is Puck's call only |
| **`propose` has no `--dry-run`** | `--dry-run` is parsed globally but only `launch` and `reconcile` honour it; `propose --dry-run` creates a real batch and moves rows to `queued`. On 2026-10-18 a bare `propose` swept 4 hasanabi + 3 tucker rows (frozen side corpus, #16) into b36 | **Always** `propose --speaker <explicit meeting bodies>`; to preview, query `videos` by speaker/status instead. If it happens: `gpu_batch.py deny bN` restores `transcoded` as long as no pod was launched |
| **Positional `_c`/`_e` ids renumber** | `classifier_calls.id` and `episode_id` are time-ordered positions; a keyword or codebook change shifts the window set, so numbers move. `reconcile_ids` / `cached_calls` match by bounds and `write_episodes` by span, so nothing is re-billed or duplicated | Never match by id across readings. Never re-finish a meeting while the cutter is mid-cut. `--ids` values that begin with `-` need `--ids=-<id>` |
| **"decoder loop" QA FAIL that is a bracketed silence** | `qa_transcripts.py` flags identical consecutive segments spanning > n s; Los Altos `jgyX7NNf7g0` failed on two `Thank you.` lines around a 3-min recess (78 % coverage, 37k words), unlike Richmond 6033/6038 (5 % coverage, a real loop) | Check **speech coverage** and word count first; a high-coverage FAIL with a quiet caption track at that minute is a recess. The JSON is kept in `batches/bN/out/`; it can be harvested without a new pod. A low-coverage FAIL is real: never load it, tell Puck |
| **Spanish tracks (Napa)** | the only Granicus track is a live interpreter's Spanish; scanned with English keywords it located 0 and the meetings went `no_episodes`. `caption_language` now picks the `_es` lists and `prompt_version` gets `/es`. v2/v2.1 have **no Spanish instruction block** | Re-read with `classify.py run --include-no-episodes --bodies napa`; those rows stay v1 until a Spanish v2 block is pinned under its own hash |
| **Flowplayer parser recovery** | Santa Clara and Hayward serve a full English track inside a Flowplayer `subtitles` object with no `<track>`; 69 meetings were written off as caption-less until `granicus.parse_player` learned it. Sausalito / SamTrans / AC Transit serve an empty 8-byte file; Napa only Spanish | After any parser fix: `captions.py --refetch --bodies <x> --dry-run`, then without `--dry-run`; a real track moves the row back to `captioned`. Then `classify.py run --include-classified` because `agenda_index` may already have advanced the row |
| **Caption homophones** | auto-captions misspell the screen's own keys consistently: fare→fair, fare gates→fairgates, ridership→wrership/writership, YIMBY and rezoning dropped; 15 of 61 screen misses in the YouTube arm | `plan/keywords_aliases_auto.txt` (`alias => term`, `? alias => term` conditional), applied only on `captions='auto'`; `al-<hash>` is in `keywords_version`. Add aliases there, re-screen for $0, classify only the new windows |
| **Cached rows under a different granularity** | a meeting read at 240 s is not cached at 120 s: different bounds, different `prompt_version` | expect a full re-bill when changing `--max-run` or codebook; `classify.py run --dry-run` prints the estimate (it runs about 2× the bill) |

---

## 5. Measured performance, for the restart decision

All of this is the **exploratory phase**: no gold set, LLM adjudication of
disagreements, 4–6 clusters per arm, intervals are cluster bootstraps and
should be read as rough (`plan/RECALL-CHECK.md`, `plan/CODEBOOK-V2-PILOT.md`
§9, `plan/CODEBOOK-V2-PILOT-FULLASR.md`). The reference is a whole-meeting
GPU transcript labelled minute by minute.

**Screened pipeline per route, v1 codebook** (`plan/RECALL-CHECK.md`):

| route | sample | recall any (95 % CI) | recall density | recall transit | screen precision | misses screen / classifier |
|---|---|---|---|---|---|---|
| `local_asr` (Richmond, San Leandro), after keyword expansion | 4 meetings, 22.6 h | **0.753** (0.654–0.852) | 0.843 | 0.435 (0–1) | 0.447 | 16 / 11 of 27 |
| YouTube auto-captions (BART, Los Altos, Milpitas, Cupertino) | 4 meetings, 15.0 h | **0.757** (0.552–0.884) | 0.860 | 0.601 (one meeting) | 0.838 | 61 / 2 of 63 |

Caption track vs WhisperX WER on those four YouTube meetings: 0.090–0.102.

**Screened arms against the adjudicated reference** (`plan/CODEBOOK-V2-PILOT.md`
§9; 4 Richmond / San Leandro meetings, 18.9 h, 127 about-minutes):

| arm | recall any | precision any | recall transit | precision transit | recall density | precision density |
|---|---|---|---|---|---|---|
| v1 @ 240 s (most existing rows) | 0.606 | 0.819 | 0.297 | 0.556 | 0.742 | 0.831 |
| v1 @ 120 s (current default) | 0.677 | 0.887 | 0.432 | 0.682 | 0.785 | 0.924 |
| **v2 @ 120 s, screened** | **0.811** (0.779–0.860) | **0.873** | **0.865** | 0.756 | 0.796 | 0.914 |
| v2 full-ASR, no screen | 0.898 (0.836–0.966) | 0.803 | 0.919 | 0.532 | 0.892 | 0.883 |

v2.1 (the "anything not car-related" sentence) is **not** what was measured.

**Cost per audio-hour** (`docs/COSTS.md` A4, `plan/CODEBOOK-V2-PILOT-FULLASR.md`):

| | screened (v1/v2 behind the screen) | full-ASR, no screen |
|---|---|---|
| Sonnet locate, Batches rate | ≈ $0.0115 (measured on the 4 pilot meetings; ≈ $0.012 corpus-wide) | $0.0407 (3.5×) |
| whole-meeting GPU text, no diarization | not needed | $0.0102–0.0125 |
| diarized episode GPU | ≈ $0.02 per episode-hour, both | same |
| corpus (≈ 2,100 meeting-h) | ≈ $25–28 Sonnet, mostly already spent | ≈ $85 Sonnet + $21–26 GPU + boot ≈ $105 point / $130 planning |
| YouTube requests | one caption request per meeting (18/day) | one whole-audio request per meeting (12/day, ≈ 63 GB, ≈ 25–30 days) |

Reading: the screen costs about 0.09 of recall against v2 full-ASR and buys
higher precision and a 3.5× cheaper Sonnet line plus no whole-meeting GPU
pass. Restart the screened path when the budget or the YouTube request
budget is the binding constraint, or for a $0 re-screen after a keyword
change; use v2 (or v2.1 once measured) at 120 s, not v1.

---

## 6. Restart checklist

1. `cd ~/ytpt && git status --short` — know what is uncommitted in `pipeline/classify.py`, `segment.py`, `plan/keywords_*.txt`; `make test` passes.
2. `.venv/bin/python pipeline/captions.py --backoff-status` and `--rate-status` — no live backoff, fewer than 18 requests in 24 h.
3. `.venv/bin/python -c "import sys;sys.path.insert(0,'pipeline');import classify as c;print(c.CODEBOOK_VERSIONS,c.DEFAULTS,c.keywords_for('en','auto').version)"` — record the instrument you are about to run.
4. Pick the codebook and granularity (`--codebook v2 --max-run 120` recommended); check `CLAUDE_CAP_USD` headroom in `.env` against `classify.py report`.
5. Queue: `captions.py --yield-table`; Granicus/Swagit bodies that are moving to full-ASR should be excluded with `--bodies` so the two routes do not read the same meeting twice.
6. `./ops/launchd/install_workers.sh com.ytpt.captions` (or `kickstart -k`) — tail `~/Library/Logs/ytpt/captions.log` for the first captioned line.
7. `classify.py run --dry-run --bodies <x>` → estimate; then `run --batch`; later `collect`. Use `--include-classified` for meetings another instrument advanced.
8. `episodes.py --plan`, then `--max N` within the shared daily YouTube budget (12 audio + captions ≤ 18).
9. `gpu_batch.py propose --speaker <meeting bodies only>`; wait for Puck's WhatsApp approval; `approve bN`; `launch bN --dry-run`; `launch bN`.
10. `make qa` (read coverage on any FAIL), `redact.py run --all`, `make load-episodes`; report per `method` and `prompt_version`, never pooled.
