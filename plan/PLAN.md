# Transcription Pipeline — Build Plan

**Audience:** the agent session running on the orchestrator laptop (Mac #2).
**Owner:** Puck (Lucas Henry).
**Status:** written 2026-09-14; revised 2026-09-17 (stratified sample, WhatsApp,
per-batch approval, QA gate, dashboard). The pipeline itself is still to be
built — `START_HERE.md` has the current state; `docs/REVIEW-2026-09-17.md` has
what changed and why.

Read this whole file plus `DECISIONS.md` before writing code. Do not relitigate
settled decisions — if you think one is wrong, say so to Puck and wait.

---

## 1. Goal

Transcribe a **stratified sample** of political YouTube (HasanAbi, Ben Shapiro,
Tucker Carlson) with **speaker diarization**, resumably, unattended, for under
~$100 of GPU, reporting progress to **WhatsApp**.

Scope is whatever `plan/build_frame.py` draws: speaker x month, a quota of
audio-hours per cell, at most 45 minutes transcribed per video (DECISIONS #9,
SAMPLING.md). That is roughly 290 audio-hours, not the ~2,000 of a full
back-catalog.

Output per video, written into `output/whisper/<video_id>.json`:

```json
{
  "video_id": "PDTon6oss9M",
  "speaker": "hasanabi",
  "channel": "hasanabi",
  "channel_url": "https://www.youtube.com/@hasanabi/videos",
  "month": "2026-04",
  "duration_s": 1834.2,
  "transcribed_s": 1834.2,
  "window_start_s": 0,
  "model": "large-v3",
  "diarization_model": "pyannote/speaker-diarization-community-1",
  "pipeline_version": "1.0.0",
  "diarized": true,
  "host_speaker": "SPEAKER_00",
  "host_confidence": 0.78,
  "segments": [
    {"start": 12.04, "end": 15.9, "speaker": "SPEAKER_01", "text": "..."}
  ]
}
```

Every field above is read by something: `analysis/load_db.py` needs `speaker`,
`month`, `transcribed_s`, `host_speaker` and `host_confidence` (without them the
corpus loads with `month = NULL` and no host filtering);
`analysis/qa_transcripts.py` uses `transcribed_s` and the speaker labels;
`model` / `diarization_model` / `pipeline_version` are what make a result
reproducible a year later. `transcribed_s` differs from `duration_s` under the
45-minute cap, and `window_start_s` records where that window began.

The `segments` shape must stay backward-compatible with `stance_analysis.py` and
`meanness_analysis.py`, which also accept the 2026-04 format (a bare list of
`{start, end, text}`; `analysis/legacy_to_contract.py` wraps those). Adding keys
is safe; renaming or nesting is not. Host enrolment (§4.6) adds `host_method`,
`host_similarity`, `host_runner_up`, `host_speakers`, `host_ambiguous`,
`host_embed_model` after the batch; readers prefer `host_speakers` over
`host_speaker`.

---

## 2. Architecture

```
  LAPTOP (Mac #2, residential IP, always on)          RENTED GPU (RunPod, ephemeral)
  ┌─────────────────────────────────────┐             ┌──────────────────────────┐
  │  yt-dlp downloader  (slow, weeks)   │             │  WhisperX + pyannote     │
  │        ↓                            │   batch     │  large-v3 + diarization  │
  │  ffmpeg → 16kHz mono opus           │  ═══════>   │        ↓                 │
  │        ↓                            │  ~100 hrs   │  segments JSON           │
  │  state.sqlite  (source of truth)    │  <═══════   │                          │
  │        ↓                            │             └──────────────────────────┘
  │  notify() -> OpenClaw -> WhatsApp   │              spun up only when queue full,
  │  OpenClaw supervisor + dashboard    │              destroyed immediately after
  └─────────────────────────────────────┘
```

**The central rule: the GPU is not always on.** Downloading is rate-limited and
takes weeks; transcription at ~25x realtime takes hours. If you leave a 4090
running while yt-dlp trickles, you pay $0.34/hr for a box idle 95% of the time —
about $170/week wasted, several times the entire budget.

So: the laptop accumulates a queue, and the GPU is rented in **batches**, drained,
and destroyed.

**Batch sizing:** boot + model download + setup is ~5-8 min of paid time. To keep
that overhead under 5%, each batch needs ≥ ~2 hours of GPU work. At ~25x realtime
(ASR + alignment + diarization combined) that is **~50-100 audio-hours per batch**,
roughly 30-60 videos. Do not fire a batch for 3 videos.

---

## 3. State model

`state.sqlite` on the laptop is the single source of truth. No JSON files as
state, no directory-listing-as-state, no `downloaded.archive.txt` as the primary
record (keep it only as the yt-dlp-native dedupe hint).

```sql
CREATE TABLE videos (
  video_id     TEXT PRIMARY KEY,
  channel      TEXT NOT NULL,
  title        TEXT,
  upload_date  TEXT,
  duration_s   REAL,
  status       TEXT NOT NULL,   -- see state machine below
  audio_path   TEXT,
  opus_path    TEXT,
  remote_key   TEXT,
  attempts     INTEGER DEFAULT 0,
  last_error   TEXT,
  updated_at   TEXT NOT NULL
);
CREATE INDEX idx_status ON videos(status);
```

State machine — strictly one direction, every transition written immediately:

```
discovered → downloading → downloaded → transcoded → queued
           → transcribing → done
                 ↓ (any stage)
               failed  → (retry resets to prior state, attempts += 1)
```

Rules:

- A worker claims work with a single `UPDATE ... WHERE status = ? AND video_id = ?`
  so two workers can never claim the same row.
- `attempts >= 3` moves a row to `dead`, is reported to Puck on WhatsApp, and is
  never retried automatically.
- Enable WAL mode (`PRAGMA journal_mode=WAL`) — multiple workers will hit this.
- **The GPU never writes to this DB.** It writes result files; the laptop reads
  them back and updates state. The GPU is disposable and must hold no state.

---

## 4. Components

### 4.1 Discovery

`yt-dlp --flat-playlist` over each channel's `/videos` to populate `videos` with
status `discovered`. Re-runnable; insert with `ON CONFLICT DO NOTHING`.

Store `channel` explicitly at discovery time. Do **not** infer speaker from the
video later — the existing `output/speakers.json` was hand-maintained and is the
kind of thing that silently goes stale.

**Capture full metadata now, even fields you don't currently need:**
view count, like count, upload date, duration, title, description, and the
crawl timestamp. It is nearly free at discovery time and impossible to recover
later — videos get deleted and channels get pruned, and a corpus you can't
describe is a corpus you can't weight, stratify, or defend in a write-up. Add
`view_count INTEGER`, `crawled_at TEXT`, `raw_meta TEXT` (the full yt-dlp JSON)
to the schema.

### 4.2 Downloader

```bash
yt-dlp -f 'bestaudio[ext=m4a]/bestaudio' \
  --cookies-from-browser chrome \
  --sleep-requests 2 --min-sleep-interval 5 --max-sleep-interval 15 \
  --limit-rate 3M \
  --retries 5 --fragment-retries 10 \
  -o '<audio_dir>/%(id)s.%(ext)s' \
  "https://www.youtube.com/watch?v=<id>"
```

- **Concurrency 2-3 maximum.** This is not the bottleneck you think it is;
  going wide gets the IP flagged and costs you days.
- **yt-dlp needs a JavaScript runtime** (`brew install deno`) plus the
  `yt-dlp[default,curl-cffi]` extras, or YouTube fails with 403s partway through
  a crawl rather than at startup. `setup.sh` checks all three.
- **Do not fetch subtitles in the pipeline.** If you ever do, ask for
  `--sub-langs "en,en-orig"`: `"en.*"` also matches ~100 machine-translated
  tracks, which is what earned an HTTP 429 and killed the Tucker run in May
  (`output/run_tucker.log`).
- **Fetch SponsorBlock segments at discovery** and store them
  (`GET https://sponsor.ajay.app/api/skipSegments?videoID=<id>&categories=["sponsor","selfpromo"]`).
  Ad reads are the most distinctive vocabulary in the Ben Shapiro sample — `com`
  tops his word list — and they are not political speech.
- Residential IP is the asset here. Never move downloading to a datacenter box.
- **Try WITHOUT cookies first.** The residential IP is the main defence; cookies
  are the fallback for when YouTube starts challenging, not a prerequisite. Many
  residential crawls at 2-3 concurrency with sleep intervals never need them.
  Adding cookies you didn't need only raises the stakes of getting flagged.
- If cookies become necessary, a **throwaway Google account** is preferable to
  Puck's personal one -- bulk-pulling a catalog risks the account, not just the
  job. Note it needs phone verification and Google frequently rejects VoIP
  numbers (including Google Voice) for new-account verification, so this may not
  be available. If it isn't, the better trade is to slow the crawl down
  (concurrency 1, longer sleeps) rather than to log in as Puck.
- **Conflict to resolve:** `--cookies-from-browser` reads the live Chrome profile,
  which (a) needs Chrome closed on macOS because the cookie DB is locked, (b)
  needs Keychain access, and (c) **cannot be read by the separate low-privilege
  user** §4.8 asks you to run OpenClaw as. Resolve it by exporting cookies once
  to a `cookies.txt` owned by that user and passing `--cookies <file>`. Re-export
  when it expires. Do not solve it by running the agent as Puck.
- Cookies expire. On `Sign in to confirm you're not a bot`, stop the downloader,
  notify Puck, and wait for a human. Do not hammer.

### 4.3 Transcode (the important optimization)

```bash
# -t 2700 applies the 45-minute cap (SAMPLING.md) at transcode time, so no file
# longer than the cap ever reaches the GPU. Record transcribed_s (and
# window_start_s if you ever use -ss for a random window) in the output JSON.
ffmpeg -nostdin -t 2700 -i in.m4a -ac 1 -ar 16000 -c:a libopus -b:a 16k -y out.opus
```

Opus always *reports* 48 kHz in `ffprobe` output; the 16 kHz input is what
matters. Measured on this repo's cached audio: 7.4 MB per audio-hour, against
58 MB/hr for the source m4a.

~7 MB per audio-hour instead of ~60 MB. **Zero quality loss for ASR** — Whisper
resamples to 16 kHz mono internally regardless. This is what makes uploading a
back-catalog over home broadband feasible at all.

**Do not delete the .m4a at transcode time.** Delete only after the
transcription for that video has come back and validated. Transcode is not the
last step that can fail, and re-downloading costs weeks, not minutes.

And think before deleting at all. 16 kbps opus is lossless *for ASR* and
destructive for everything else. If you ever want **acoustic** measures —
prosody, pitch, loudness, vocal affect — the m4a is the only copy that supports
them, and vocal intensity is a defensible polarisation measure in its own right,
arguably better grounded than an LLM rating text.

**Measured on the 20 cached files in `output/audio/` (2026-09-17):**

| | per audio-hour | sample (~576 raw hrs) | full catalog (~2,000 hrs) |
|---|---|---|---|
| m4a (YouTube bestaudio, ~128 kbps AAC) | 58 MB | 33 GB | 116 GB |
| opus 16 kHz mono @ 16 kbps | 7 MB | 4 GB | 14 GB |
| transcript JSON | 0.14 MB | 80 MB | 280 MB |

An earlier draft of this plan said 2,000 hours of m4a was ~1.5 TB. That was
wrong by more than an order of magnitude — the real figure is ~116 GB, and it
is remarkably consistent (58-59 MB/hr across every cached file).

**So: don't delete the m4a.** The entire full-catalog archive fits on any laptop
and costs under a dollar a month to back up online. The disk-pressure argument
for deleting it does not survive contact with the actual numbers.

### 4.4 Queue transport

**v1 (default): rsync over SSH** to the RunPod box when a batch starts. Simple,
no extra accounts.

**v2 (upgrade, do this if v1 annoys you): Cloudflare R2.** Zero egress fees,
~$0.015/GB-month, S3 API. Decouples laptop from GPU entirely — laptop uploads
whenever, GPU pulls whenever, neither needs the other awake. 100 audio-hours is
~700 MB, so storage cost is rounding error.

Pick one and write the transport behind a single interface so swapping is a
one-file change.

### 4.5 GPU job

Bootstrap script, pinned versions, run on a RunPod **RTX 4090 Community tier**
(~$0.34/hr; see `DECISIONS.md` for why not Groq).

Stack: **WhisperX** — faster-whisper for ASR, wav2vec2 forced alignment for
word-level timestamps, pyannote for diarization, all batched.

Critical setup gotchas, in order of how much time they will cost you:

1. **pyannote models are gated.** WhisperX 3.8.x requires `pyannote-audio >= 4`,
   whose model is **`pyannote/speaker-diarization-community-1`** — accept its
   user conditions on Hugging Face with the same account the `HF_TOKEN` belongs
   to. (The older `speaker-diarization-3.1` + `segmentation-3.0` pair only
   applies if you pin an older WhisperX.) Failing this gives an unhelpful 401
   deep in a stack trace, on a rented GPU, with the meter running. Verify before
   renting: `make creds` checks exactly this.
2. **WhisperX dependency hell.** torch / ctranslate2 / pyannote / faster-whisper
   versions fight. WhisperX 3.8.x wants **Python >=3.10,<3.14**, `torch ~=2.8`,
   `ctranslate2 >=4.5`, `pyannote-audio >=4`; pick a pod image that matches
   before installing anything. Pin every version in the bootstrap, and when a
   fresh pod works, snapshot the exact `pip freeze` into `plan/gpu-freeze.txt`
   and use it forever.
3. Community-tier pods can be **reclaimed mid-job**. Write each video's result
   file the moment it completes and sync it out immediately — never batch the
   writeback to the end.
4. Ephemeral disk. Persist nothing on the pod you can't lose.

Per-video output must be written and synced back individually, not as one
archive at the end.

**Long files.** Under the sampling design nothing longer than the 45-minute cap
reaches the GPU (§4.3), so the split-and-stitch machinery below is **not needed
unless the cap is lifted**. If it ever is: split on silence boundaries
(`ffmpeg -af silencedetect`), process the parts, then stitch — offsetting
timestamps and **re-reconciling speaker labels across parts**, since pyannote
numbers clusters independently per file. Getting the stitch wrong silently
corrupts both timestamps and speakers, so test it on one long video first.

**Every returned transcript goes through the QA gate before it counts as done**:
`python3 analysis/qa_transcripts.py --src output/whisper --csv output/qa/latest.csv`.
A FAIL row is not loaded; a WARN row is read by a human. It catches the failure
this phase exists for: it already found a decoder loop in the 2026-04 output
(`AAiFho_u_GI` at 21:39, "Hassan piker" x45 over 30 seconds, overwriting real
speech).

### 4.6 Diarization post-processing

pyannote returns `SPEAKER_00`, `SPEAKER_01`, ... — anonymous cluster labels, not
identities, and the numbering is **not stable across videos**.

Map to roles per-video with an explicit, recorded rule — two of them, in order:

- **On the pod** (`pipeline/gpu_job.py`): the host is the speaker with the most
  total speech time. Write `host_speaker` and `host_confidence` = (host share
  of speech time). Under 0.55 the guess is unreliable — the interview-heavy
  videos.
- **On the Mac, after the batch** (`scripts/host_enroll.py --apply`, since
  2026-09-20): speaker-embedding enrolment replaces the guess. One anchor per
  channel from a video where the heuristic was certain (Hasan ← `C0LMNHE-OF4`,
  Tucker ← `q72s8zwwGg4`; `pyannote/wespeaker-voxceleb-resnet34-LM`, cached in
  `output/host_anchors/`). Every cluster is scored by cosine to its channel's
  anchor; **host = every cluster with cos ≥ 0.60**, provided the best cluster
  is ≥ 0.60 and leads the best non-host cluster by ≥ 0.15. Over-split hosts
  (Hasan's impressions) merge into one label set. Otherwise the talk-time guess
  stays and the file is marked ambiguous. Added keys: `host_method`
  (`embedding` | `talk_time`), `host_similarity`, `host_runner_up`,
  `host_speakers` (list), `host_ambiguous`, `host_embed_model`; `host_speaker`
  becomes the best-matching host cluster and `host_confidence` becomes the
  cosine. Originals are kept in `output/whisper_pre_enroll/`. Readers use
  `host_speakers` and fall back to `host_speaker` (`analysis/METHODS.md` §8).

Known failure modes, expect and spot-check them:

- **Over-merging** similar voices (two male podcast guests → one label).
- **Over-splitting** one speaker who changes register — Hasan does impressions
  and voices constantly, and will fragment into 3-4 labels.

Diarization is unsupervised clustering of voice embeddings. It is a useful prior,
not ground truth. Do not let downstream analysis treat `speaker` as reliable
without the validation in Phase 0.

### 4.7 Notifier  — built: `pipeline/notify.py`

Single `notify(level, message)`; the backend is `NOTIFY_BACKEND` in `.env`.
Default `openclaw`, which shells out to

```bash
openclaw message send --channel whatsapp --target "$OPENCLAW_NOTIFY_TARGET" --message "..."
```

Discord webhooks remain as a fallback backend (DECISIONS #7).

- **Do not notify per video.** 2,000 pings on a personal phone is worse than
  silence. The supervisor's heartbeat sends only what changed
  (`dashboard/status.py --new-alerts`), plus immediate messages for: a batch
  awaiting approval, batch finished (with cost), bot detection, any `dead` row,
  a reclaimed pod, QA failures.
- Puck asks for the full picture with `status`, or looks at the dashboard
  (§4.9). Both come from the same `status.collect()`, so they cannot disagree.
- **Delivery never blocks the pipeline.** If the gateway is down the message is
  appended to `notify_outbox.jsonl` next to `state.sqlite`, and
  `notify.py --flush` (in the heartbeat) sends it later. A notifier that can
  crash the downloader is worse than no notifier.
- Every batch-finish message must include **actual GPU minutes and dollars
  spent**. The whole point of batching is cost control; if it isn't reported,
  it isn't controlled.

### 4.8 OpenClaw supervisor

Runs on the laptop. Its job is the loop a human would otherwise do: notice the
downloader stalled, restart it, notice a batch is ready, rent the GPU, run the
job, tear it down, report.

**Approval gate (DECISIONS #10).** A batch is *proposed* by the pipeline, not by
the agent: when ready-for-GPU hours pass `BATCH_TARGET_HOURS`, a row goes into
`batches` as `awaiting_approval` and `notify("approval", ...)` sends Puck the
video count, audio-hours and estimated cost. Puck replies `approve <batch_id>`;
the agent then runs `pipeline/gpu_batch.py launch <batch_id>`, which is
deliberately **not** on OpenClaw's exec allowlist, so OpenClaw asks him to
`/approve` the command itself. The launcher independently refuses unless the row
is approved, the month's spend plus the estimate is under the cap, and no pod is
already live. Three layers, none of them a sentence in a prompt; details and the
allowlist are in `docs/OPENCLAW_SETUP.md` §5.

Security — it will hold shell access, a RunPod key, an HF token, and the
WhatsApp link:

- Run it as a **dedicated non-admin macOS user**, not Puck's account.
- Give it access to the project folder only. Not all of `~/Documents`.
- Keys in a separate env file readable only by that user, never in the repo.
- **Never give it the personal Google cookies** — throwaway account only.
- Hard spending guard: it must refuse to rent a GPU if cumulative spend this
  month exceeds a configured cap (default $150). This is a code-level check,
  not an instruction in a prompt. Track spend in a `spend` table in
  `state.sqlite` — an in-memory counter resets on restart, which defeats it.
- Pod lifecycle is RunPod's **REST/GraphQL API**, not SSH — SSH is only for
  driving a pod that already exists. The supervisor needs a RunPod API key, and
  a `finally:` that terminates the pod on *any* exit path. A create call that
  succeeds followed by a crash leaves a 4090 billing quietly until someone
  notices.

- Never allowlist a bare interpreter (`python3`, `bash`, `sh`) or `curl` for the
  agent: any of them is a way to call the RunPod API directly and walk around
  the approval gate.

Keep the downloader and transcode workers as **plain scripts under `launchd`**.
They are deterministic ETL; an LLM in that loop is slower, costlier, and
non-reproducible. OpenClaw supervises; it does not transcribe.

### 4.9 Progress dashboard — built: `dashboard/`

`make dashboard` serves http://localhost:8765 on the supervisor Mac: hours
transcribed against the frame, download rate and ETA, the speaker x month quota
heatmap, batches (including any awaiting approval), spend against the cap, the
latest QA run, and the failed/dead rows. Read-only, loopback only, no
third-party JavaScript; `dashboard/README.md` explains the rest. It reads
`state.sqlite`, `plan/frame_cells.csv` and the QA CSV and tolerates all three
being absent, so it works before the pipeline writes anything
(`make dashboard-demo` shows it on synthetic data).

---

## 5. Cost model

The corpus is the stratified sample (DECISIONS #9): 3 speakers x ~32 months x
3 capped audio-hours ≈ **290 audio-hours**. The full-catalog figures are kept in
brackets for comparison.

| Item | Rate | Sample (~290 h) | [Full catalog (~2,000 h)] |
|---|---|---|---|
| Download | free | days of wall-clock | [weeks] |
| GPU, 4090 community | $0.34/hr @ ~25x realtime (ESTIMATE) | ~12 GPU-hr → **~$4** | [~80 GPU-hr → ~$27] |
| Retries / overhead / reclaimed pods | +50% contingency | ~$6 | [~$40] |
| m4a kept on disk | 58 MB/audio-hr (measured) | ~17 GB | [~116 GB] |
| opus uploaded | 7.4 MB/audio-hr (measured) | ~2 GB | [~14 GB] |
| R2 storage (if v2) | $0.015/GB-mo | rounding error | [<$1/mo] |
| **Total** | | **~$6-10** | [~$40-60] |

Every GPU number above rides on the 25x realtime estimate. Phase 0 measures it;
rewrite this table with the measured figure.

Compare: AssemblyAI with speaker labels is $0.17/audio-hour — about **$50** for
the sample (docs/MODELS.md §5). The GPU route is ~5-10x cheaper in dollars and
costs two evenings of setup; that is the actual trade, and it is close enough
that stalling on the GPU is a good reason to switch.

**This is the finding that justifies the whole plan.** Self-hosting is not just
cheaper here, it's cheaper *and* gives speaker labels. The earlier advice to just
use Groq was correct only while diarization was out of scope. It is now in scope,
so the GPU wins on both axes.

---

## 6. Phases

Do these in order. Do not start Phase 3 before Phase 0 passes.

### Phase 0 — Validate before spending anything (half a day)

Non-negotiable. **Done by hand** — rent one pod, SSH in yourself, run things
interactively. No automation, no supervisor, no state DB. You are buying
information, not building infrastructure.

**Reference set first.** None of the 20 already-transcribed videos has a human
reference: all 20 `.en.vtt` files are YouTube's own ASR (byte-identical to
`.en-orig.vtt`, inline word timings). The only five with **manual** captions are
the Tucker videos — `GyYy-QmxttU`, `DTcfedTY0DA`, `JThlN-NX1Dw`, `K_2_7fxyIEI`,
`Js9aZSQlSQ4` — and their audio was never downloaded (that run died on an HTTP
429). So step 0 is: download those five audio files. Note that manual captions
are often *clean verbatim*, which inflates WER by a few points; hand-transcribe
two 5-minute excerpts if you want a true number.

Then, on those five plus 2-3 of the existing Hasan/Shapiro videos:

1. Run WhisperX large-v3 + community-1 diarization on one rented GPU hour.
2. `python3 analysis/qa_transcripts.py --src <out> --subs output/subs --csv output/qa/phase0.csv`
   — it computes WER against the captions in the right direction, flags decoder
   loops, coverage, speaking rate, and the minutes where the transcript and the
   captions disagree by more than 2x. Acceptance thresholds: `docs/TESTING.md` §4.
3. **Listen to the flagged timestamps**, and to two stretches of music/silence
   that were not flagged. A WER average hides hallucination because it is rare
   and concentrated.
4. Hand-check diarization on ~10 minutes of a Tucker interview (2+ speakers) and
   ~10 minutes of Hasan doing voices. Record the actual observed failure rate.
5. **A/B the transcode:** transcribe one video from the m4a and from the 16 kbps
   opus. DECISIONS #4 calls the opus "lossless for ASR" — this is the only step
   that turns that into a measurement.
6. **Measure actual throughput** (audio-seconds per GPU-second, ASR + alignment
   + diarization combined) and rewrite §5's cost model with the real number.
7. Optionally A/B `large-v3` against `large-v3-turbo` on the same files; if WER
   and hallucination match, the turbo model halves the bill.
8. Snapshot the working `pip freeze` to `plan/gpu-freeze.txt`.

**Gate:** if hallucination on silence is bad, add a VAD pre-filter before
transcribing 290 hours of it. Finding this out at hour 290 instead of hour 3 is
the most expensive mistake available here.

### Phase 1 — Laptop downloader (1 day)
State DB (including `events`, `batches`; SCHEMA.md), discovery from
`plan/frame.csv`, downloader, transcoder with the 45-min cap, launchd jobs,
`pipeline/requeue.py`. The notifier and the dashboard already exist.
**Acceptance:** runs 24h unattended, survives a laptop sleep/wake, correct
resume after `kill -9`, the dashboard and `status` agree, WhatsApp gets new
alerts and nothing else, m4a **kept** (§4.3) and every finished row has a
transcode.

### Phase 2 — GPU batch job (1 day)
`pipeline/gpu_batch.py` (propose / launch / teardown), bootstrap script, pinned
freeze, transport, per-video writeback, QA gate on return.
**Acceptance:** one 50-video batch end to end, every returned transcript through
`analysis/qa_transcripts.py` with zero FAIL loaded, cost reported and within 20%
of the model above, a deliberately killed pod loses at most one video of work,
and the `finally:` teardown leaves no pod behind.

### Phase 3 — OpenClaw supervisor (1 day)
Wire the above together (`docs/OPENCLAW_SETUP.md`). WhatsApp two-way: `status`,
`qa`, `cost`, `pause`, `resume`, `retry <id>`, `approve <batch>`.
**Acceptance:** Puck closes the laptop lid for 48h and comes back to correct
state and a coherent WhatsApp thread. Three guards verified by test, not by
reading: deny a batch and confirm no pod exists; set the cap to $1 and confirm
the launcher refuses an approved batch; send a command from a number not in
`allowFrom` and confirm it is ignored.

### Phase 4 — Backfill
Let it run. Check in on the dashboard; WhatsApp when something changes.

---

## 7. Existing-code notes

The bugs listed here on 2026-09-14 were **fixed on 2026-09-17**; the evidence and
the fixes are in `docs/REVIEW-2026-09-17.md`. What is worth carrying forward:

- Summaries are merged on write, never rewritten from the current run
  (`output/meanness/summary.json` still holds 1 of 10 from the old behaviour;
  `meanness_analysis.py --rebuild-summary` reconstructs it offline).
- An unparseable model reply is `None`, never a number.
- Claude 5 models reject `temperature` and think adaptively by default, so
  scoring passes `thinking={"type":"disabled"}` and measures noise with k
  samples instead (`docs/MODELS.md` §2).
- Any target/node matching is whole-word: the old substring search counted
  "police" as a mention of "ice".

`hasan_compare.py` is the WER/diff harness — reuse it for Phase 0's side-by-side
pages, but take the numbers from `analysis/qa_transcripts.py`, which computes
WER against the captions in the right direction and flags hallucination.

---

## 8. Open questions for Puck

Ask, don't guess:

1. ~~**Scope**~~ — **answered 2026-09-17**: stratified sample, speaker x month,
   an audio-hours quota per cell (DECISIONS #9, SAMPLING.md). What remains is
   the *value*: hours per cell (default 3) and which months the window covers.
2. **Channel URLs** (`plan/channels.tsv`) are guesses. Which Hasan channel —
   clips or full streams — decides what "Hasan" means in the results.
3. **May videos over 90 minutes enter at all?** With the 45-minute cap this is a
   format decision, not a cost one (`--max-min 0` admits them).
4. **First 45 minutes or a random 45-minute window?** The opening of a stream is
   often pre-show and music.
5. ~~External SSD~~ — moot: the measured full-catalog m4a archive is ~116 GB, and
   the sample is ~17 GB. Any laptop disk holds it.
6. Throwaway Google account for cookies — does one exist? (Only needed if
   YouTube starts challenging; §4.2.)
7. Monthly spend cap for the guard. Default $150 — but the real ceiling is the
   RunPod prepaid balance, so keep that small too.

---

## 9. Review findings

Three review passes over this plan, 2026-09-14. Items 1-6 are already fixed
inline above. Items 7-9 are unresolved and need Puck.

**Fixed inline**

1. *m4a deleted too early.* Original said delete after transcode. Transcode is
   not the last failing step, and re-download costs weeks. Now deleted only
   after transcription validates — and preferably not at all (§4.3).
2. *Cookies vs. least-privilege contradicted each other.* §4.2 said use
   `--cookies-from-browser`; §4.8 said run the agent as a separate user, which
   cannot read Puck's Chrome profile. Resolved via exported `cookies.txt`.
3. *Long files unaddressed.* 3-6h streams will OOM both WhisperX and pyannote.
   Added splitting + stitching, including cross-part speaker reconciliation,
   which is the part that silently corrupts data.
4. *Phase 0 had a circular dependency* — it needed the GPU stack that Phase 2
   builds. Now explicitly a manual, by-hand pod session.
5. *Throughput was an estimate presented as a basis for cost.* Phase 0 now
   measures it and rewrites §5.
6. *Pod lifecycle needs the RunPod API, not SSH*, and the spend counter must
   persist in SQLite or a restart silently resets the guard. An orphaned pod
   after a crash is the most likely way this project overspends.

**Unresolved — need a decision**

7. ~~**Corpus design**~~ — **resolved 2026-09-17** by Puck: stratified, hours per
   cell (DECISIONS #9). The original finding, kept because the reasoning still
   applies to any future "just download everything" impulse: The
   plan as written downloads everything, which yields a corpus dominated by
   whoever uploads most — Hasan posts near-daily, Tucker doesn't. Thousands of
   hours of one speaker and a few hundred of another is not a stronger dataset
   than a balanced sample; it is a weaker one, because the imbalance is
   confounded with format, era, and topic mix all at once. A stratified sample
   (balanced per channel per quarter, capped per channel) is better research
   *and* an order of magnitude cheaper.

   Decide this **before** the downloader runs for weeks. It is the one choice
   here that can't be fixed afterwards by spending more money.

8. **Format confound still has no mitigation.** Livestream vs edited monologue
   vs interview differs in insult density regardless of politics. Diarization
   partly addresses it (drop non-host speech), but not the scripted/unscripted
   axis. At minimum record a `format` label per video at discovery so it can
   enter the model as a covariate later. Cheap now, impossible retroactively.

9. **Ethics/ToS unaddressed.** Bulk downloading is against YouTube's ToS, and
   transcripts of identifiable people are personal data. Not a blocker for
   private research, but if this becomes a publication or a public repo, the
   corpus should not be redistributed, and a university affiliation means an
   ethics review is the norm rather than an optional extra. Worth ten minutes
   of thought before it's 2,000 hours deep.

---

## 10. Review findings — 2026-09-17 (handoff review)

A full pass over the code, the 20 existing transcripts and the run logs before
handing the project to an OpenClaw supervisor. Every finding, its evidence and
its fix: **`docs/REVIEW-2026-09-17.md`**. The four that would have cost the most:

1. The metadata crawl wrote a catalog that `build_frame.py` silently discarded
   (literal `\t`), and exited 0 while doing it.
2. A failed crawl for one channel would have produced a two-speaker frame
   reporting "0% dropped" — a changed study design, unannounced.
3. The Phase 0 WER harness measured in the wrong direction, which *reduces* the
   error a hallucinated passage produces, and every diff page mislabels
   YouTube's ASR captions as "manual".
4. Collocate analysis matched node words as substrings: 98% of the "mentions" of
   `ice` in the real transcripts were `police`, `justice`, `voice`.

All four are fixed and pinned by checks in `make test`. What the QA gate found in
the existing output (one real decoder loop, one truncated ending, sponsor reads
dominating one speaker's vocabulary) is in the same document, §E.
