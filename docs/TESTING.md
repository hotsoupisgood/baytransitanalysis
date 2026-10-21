# Testing the system

Six levels, cheapest first. Each one is worth running before the next. Levels
0-2 cost nothing and take about two minutes; level 3 sends a handful of free or
sub-cent requests; levels 4-5 involve real money and real time.

| Level | Command | Costs | Proves |
|---|---|---|---|
| 0 | `./setup.sh` | — | the machine has the tools and can install the deps |
| 1 | `make test` | — | the analysis, QA and pipeline code is correct on data with a known answer |
| 2 | `make smoke` | — | the whole toolchain runs here: ffmpeg, dashboard, notifier, credentials |
| 3 | `SMOKE_FLAGS='--network --notify --paid' make smoke` | <$0.01 | YouTube answers this IP, WhatsApp arrives, the scoring request shape is accepted |
| 4 | Phase 0 (`plan/PLAN.md` §6) | ~$1 GPU | transcription quality is good enough to buy 300 hours of it |
| 5 | Phase 1-3 acceptance (`plan/PLAN.md` §6) | ~$5-10 | the pipeline survives a batch, a reboot and a `kill -9` |

---

## Level 0 — `./setup.sh`

Idempotent. Creates `.venv`, installs `requirements.txt`, and checks:

- python ≥ 3.11 (yt-dlp deprecated 3.10 in 2026)
- ffmpeg **with the libopus encoder** — the transcode is 16 kHz mono Opus
- **deno** (or node ≥ 22): yt-dlp needs an external JS runtime for YouTube
- `yt-dlp` in the venv, plus `yt-dlp-ejs` and `curl_cffi`
- SQLite FTS5, and that `STATE_DB_PATH`'s folder can host a WAL database
- `.env` present, keys set, mode 600
- stale `.git/*.lock` files (a remote session that cannot delete files leaves them)
- the test suite

Ends with `N ok, N warnings, N failures`. Warnings are fine on a machine that is
not the supervisor Mac (no openclaw, no rclone).

## Level 1 — `make test`

81 checks in about a minute, ending in `All checks passed.` It builds a simulated
corpus with a **planted effect** and a **planted non-effect**, then asserts both
are handled correctly — an analysis that has only ever run on real data cannot
be verified. Sections 8-14 pin the bugs found in the 2026-09-17 review
(`docs/REVIEW-2026-09-17.md`), including the ones that produced plausible-looking
wrong numbers: substring KWIC matching, the asymmetric filler list, a silently
missing speaker in the sampling frame, a catalog written with literal `\t`.
Section 15 (2026-09-18) runs the laptop workers offline against a stub
`yt-dlp`: the state machine, discovery, the downloader's bot-check stop and
random pause, the transcode cap, and every refusal path of the GPU launch gate
(plus the dry-run passing once the Phase 2 artefacts exist). Section 16
(2026-09-18) covers the GPU side without a GPU: `plan/gpu-freeze.txt` parses and
pins the right majors, `gpu_bootstrap.sh` parses and exits 2 before `pip` on a
machine with no `nvidia-smi`, the contract JSON matches `load_db.resolve_host`,
`gpu_job.py --on-pod` against a fake WhisperX writes one atomic file per video
plus `timing.jsonl` and `FAILED/<id>.txt` and skips finished videos on re-run,
and the laptop-side ingest sends a QA FAIL to `failed` without loading it.
Nothing in `make test` touches RunPod, Hugging Face or a pod.

If a check fails, the line says which. The most common cause on a new machine is
a different numpy version changing the bootstrap RNG — check 7 is the only one
sensitive to that, and the message prints both intervals.

## Level 2 — `make smoke`

`scripts/smoke_test.sh`. Beyond level 1 it:

- runs the QA gate on the 20 exploratory transcripts. Reference result
  (2026-09-17): **20 videos: 18 OK, 2 WARN, 0 FAIL**. A FAIL here means the gate
  or the files changed, not that the gate is broken.
- transcodes a generated 20-second tone to 16 kHz mono Opus and checks the
  duration, codec and channel count. Opus always *reports* 48 kHz; that is the
  format, not a bug.
- renders the dashboard from demo data and prints the WhatsApp status digest
- builds (but does not send) a notifier command
- dry-runs `build_frame.py` against `plan/catalog.tsv` if there is one
- checks the credentials (see below) and `openclaw gateway status`

## Level 3 — the live checks

Each is opt-in because each has a real effect.

`--paid` → `scripts/check_credentials.py --paid` sends **one** 5-token request
with the exact shape `legacy/meanness_analysis.py` uses (`thinking: disabled`, no
temperature). It costs about $0.0001 and catches API changes that would
otherwise surface as a 400 in the middle of a scoring run.

`--network` → one metadata request to YouTube, no download. Run it **only on the
residential Mac**, never from a server (`DECISIONS.md` #1). It fails loudly if
YouTube is bot-checking this IP, and warns if the JS runtime or curl_cffi is
missing.

`--notify` → one WhatsApp message through OpenClaw. Check the phone.

### What the credential checks prove (all free)

| Check | Call | Failure means |
|---|---|---|
| Anthropic key | `GET /v1/models` | key revoked, or the models this project names aren't available to it |
| HF token | `GET /api/whoami-v2` | token invalid |
| pyannote licence | `GET .../speaker-diarization-community-1/resolve/main/config.yaml` | the conditions were not accepted **by this token's account** — the failure that otherwise appears as a 401 inside pyannote on a rented GPU |
| RunPod key | GraphQL `myself` | key invalid |
| RunPod pods | same query | **a pod is running and billing right now** |
| RunPod balance | same query | keep it small: it is the hard ceiling if everything else fails |

---

## Level 4 — Phase 0: is the transcription any good?

Full procedure in `plan/PLAN.md` §6. The part that matters here is what counts
as "good", because the gate is quality, not cost.

**Reference set.** Only five videos in this repo have human-written captions —
the Tucker five (`GyYy-QmxttU`, `DTcfedTY0DA`, `JThlN-NX1Dw`, `K_2_7fxyIEI`,
`Js9aZSQlSQ4`) — and their audio was never downloaded. Download those first.
The other 20 videos' `.en.vtt` files are YouTube's own ASR: comparing against
them measures agreement between two ASR systems, not accuracy. The QA gate
prints that warning itself.

Manual captions are often *clean verbatim* (fillers and false starts removed),
which inflates WER by a few points. For a true number, hand-transcribe two
5-minute excerpts and compare against those.

**Acceptance, per video:**

```bash
python3 analysis/qa_transcripts.py --src output/whisper --subs output/subs --csv output/qa/phase0.csv
```

| Signal | Accept | Look | Reject |
|---|---|---|---|
| FAIL rows | 0 | — | any |
| WER vs human captions | < 12% | 12-20% | > 20% |
| coverage (speech seconds / span) | > 0.85 | 0.6-0.85 | < 0.3 |
| repeated-segment runs | ≤ 2 | 3-4 | ≥ 5 |
| gzip-ratio flags (loops) | 0 | any → listen at the timestamp | — |
| minutes with >2x the captions' words | 0 | any → listen | — |
| host share (diarized) | > 0.7 | 0.55-0.7 | < 0.55 without review |
| unlabelled speech | < 5% | 5-15% | > 15% |
| subtitle-style speaker tags ("Male Speaker 1 …") | 0 | any → hallucinated words, strip at ingest | — |

Host share is not rated for meeting episodes (`<meeting>_e<n>`, or `kind =
episode`): nobody hosts a council meeting, and `pipeline/roles.py` labels the
clusters dais / public instead.

**Meeting episodes (2026-09-21):** `analysis/validate_transcription.py` runs the
whole level-4 check on every `done` episode at $0 — disagreement rate against
the meeting's caption track (per episode, per minute, per body, per role, with
the caption lag measured), Oakland against its Granicus live captions, text over
low-energy audio, loops, dropped / extra minutes, diarization merge / split
flags, and a 12-excerpt listening set with correction files that
`scripts/score_reference.py … --summary` turns into a true WER. Output under
`output/validation/` (`REPORT.md` first). The first run, on the 46 pilot
episodes: 8.7 % pooled disagreement with YouTube's ASR, 0 dropped minutes,
0 text over silence, 19 hallucinated speaker-tag segments.

Also do by hand, once, because no metric catches them:

1. Listen to 10 minutes of a Tucker interview and count segments attributed to
   the wrong speaker. Diarization is unsupervised clustering, not ground truth.
2. Listen to 10 minutes of Hasan doing voices — the classic over-splitting case.
3. Listen to two stretches of music/silence and read what the transcript claims
   is there.
4. Transcribe the same video from the `.m4a` and from the 16 kbps `.opus` and
   compare. `DECISIONS.md` #4 calls the transcode "lossless for ASR"; that is an
   argument, not a measurement.
5. Record the **measured** realtime factor and rewrite `plan/PLAN.md` §5 with it.

## Level 5 — pipeline acceptance

From `plan/PLAN.md` §6, plus:

- **Approval gate:** propose a batch, reply `deny`, confirm no pod was created.
  Then set `MONTHLY_SPEND_CAP_USD=1` and confirm the launcher refuses even after
  an approval. Both are code paths, not prompts.
- **Crash safety:** `kill -9` the downloader mid-file; on restart no row is
  stuck in `downloading` and no partial file is treated as complete.
- **Reclaimed pod:** kill a pod mid-batch; at most one video's work is lost.
- **QA in the loop:** a deliberately corrupted transcript (duplicate a segment
  six times) must make the batch's QA gate FAIL and must not be loaded.

---

## Output acceptance at every stage

| Artefact | Check | Expected |
|---|---|---|
| `plan/catalog.tsv` | `head -2`, column count | 9 tab-separated columns; every speaker present with a non-zero count |
| | dates | > 90% rows with a usable `upload_date` (they are approximate; SAMPLING.md) |
| `plan/frame.csv` | `frame_cells.csv` | ≤ 20% of months dropped, few `short` cells |
| | `frame_meta.json` | catalog sha256 recorded, seed recorded |
| `.m4a` | `ffprobe` duration vs catalog | within ~2 s; ~58 MB per audio-hour |
| `.opus` | size | ~7.4 MB per audio-hour (measured 2026-09-17), codec opus, 1 channel |
| transcript JSON | `make qa` | 0 FAIL; WARNs read and explained |
| `corpus.sqlite` | `make load` | video and segment counts match the frame; re-runnable from files |
| word frequency | `make overall` | ad-read words (`com`, sponsor names) near the top means sponsor segments are still in the corpus — see `docs/MODELS.md` |
| LLM scores | per-video JSON | parse failures < 1% of chunks; `score_sd` reported; validate against human coding before using |
