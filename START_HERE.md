# START HERE

You are an agent picking up this project on a new machine. Read this file
completely before doing anything. It takes about four minutes and will save you
from several expensive mistakes.

## What this project is

Research on political polarisation on YouTube. The method is quantitative
transcript analysis — word frequency, collocates, topic, sentiment, plus
qualitative reading. Owner: **Puck** (Lucas Henry), who has an MSc in Health
Data Science and does Bayesian hierarchical modelling. Pitch statistics at that
level; don't hand him a t-test.

The immediate job is a **transcription pipeline**: download political YouTube
audio, transcribe it with speaker diarization, and land it in a searchable
corpus. The design is finished and documented. The laptop/GPU pipeline itself is
still to be built.

## Current state (2026-09-22)

**Two corpora now.** The Hasan/Tucker frame (below) is transcribed (183 in-frame
videos done, b1–b11) and frozen as a side corpus (proposed DECISIONS #16). The
active study is the **Bay Area density + transit meeting corpus**
(`plan/PROPOSAL-transit.md`, approved by Puck 2026-09-20; T0 in
`plan/T0-RESULTS.md`; T1 pilot built and run 2026-09-21; **T2 part 1 built
2026-09-22**, `plan/T2-PLAN.md`):

| | Status |
|---|---|
| Schema: `videos.kind` + `sources`, `episodes`, `speaker_roles`, `classifier_calls` (`plan/SCHEMA.md` §1) | **Migrated**; 2026-09-22 `classifier_calls` re-keyed to `(id, prompt_version)`, `classifier_batches` + `sources.vocab` added (backups in `~/.local/share/ytpt/backups/`) |
| Meeting stages `discover --meetings` → `captions.py` → `classify.py` → `episodes.py` → `gpu_batch.py propose` | **Built and run on the T1 pilot** (12 meetings, 63 episodes, 46 transcribed, QA 0 FAIL, validation report at `output/validation/REPORT.md`) |
| Sources for the full run | **33 bodies** in `plan/channels.tsv`; discovery ran 2026-09-22 over the whole window: **622 regular meetings, 2,284 meeting-hours**. 10 bodies stay unlocated (`plan/T2-PLAN.md` §6) |
| T2 part 2 (2026-09-22 → 24) | **Running slowly on purpose.** YouTube is a launchd **trickle worker** (`com.ytpt.captions`), **429'd a second time on 2026-09-23 after 5 requests in 41 min** and now **DOWN** in a 24 h backoff with **322 of 385 meetings uncaptioned**. Since 2026-09-24 it runs at **one meeting an hour with a hard cap of 18 requests per rolling 24 h**, enforced from the new `worker_requests` table so a launchd restart cannot burst, and its queue is ordered by **measured yield** (episodes per captioned meeting, 16.7 BART → 0.00 Rohnert Park; `--yield-table`) instead of by date. Bot-check behaviour unchanged. Only **Petaluma** has a non-YouTube route — Swagit, with its own timestamped transcript (`plan/SOURCES-AUDIT.md`); **built 2026-09-24** (`pipeline/swagit.py`, `method='swagit_transcript'`): **13 of 15** meetings registered and off the YouTube queue (**322 → 309 waiting**), candidate recall against the caption route **0.914** and 18 of 18 known episodes recovered (`plan/SWAGIT-RECALL.md`); the 2 closed-session specials publish a 2-minute clip against a 42-min / 1.78-h video and stay on the queue; **not classified — stage 2 is $1.06 and needs Puck**. Detail: `plan/T2-PLAN.md` §3h, §3i. Granicus is located three ways — the classifier where a caption track exists, `method='agenda_index'` where it does not, `method='local_asr'` where there is neither. Batch threshold **50 h → 15 h**. Detail: `plan/T2-PLAN.md` §3a (first session), §3b–3c (second), **§3d (third: Puck's three approved runs)** |
| Two instruments, one meeting | A meeting carries both `_e<n>` (classifier) and `_a<n>` (agenda_index) episodes, and the status machine no longer lets one hide the other: `classify.py --include-classified` reads a caption track on a meeting `agenda_index` already advanced (**19 of Santa Clara's and Hayward's 37 recovered tracks were stranded that way**), and `episodes.py` comes back for a **'cut' meeting that gained new 'located' episodes**. `episodes.method` records the ROUTE, never the model — report per method, never pooled |
| Granicus locate routes | Not six caption-less bodies but two. **Santa Clara and Hayward serve full English tracks** through a Flowplayer config with no `<track>` element — a parser bug; `--refetch` recovered **37 meetings (Santa Clara 34 of 40, 178 h)**, all now classified (**105 episodes / 16.0 h, $1.18**). Sausalito/SamTrans serve an empty 8-byte file; Napa's only track is Spanish — those three are located by **agenda index points** (53 episodes / 26.5 h over 5 bodies). **Richmond and San Leandro have neither**; their audio is reachable only via `ASX.php` → MP4, and `scripts/asr_batch.py` is working through **59 meetings / 260.6 audio-h ≈ 22.5 Mac-h** of local `mlx-whisper` ASR (`plan/sources_bayarea.md` "T2 locate probe", `plan/T2-PLAN.md` §3d) |
| Classifier instrument | Haiku 4.5 validated against the T0 Sonnet labels and **not adopted** (0.758 about-vs-not; it demotes a third of transit episodes). T2 uses **`claude-sonnet-4-6` via the Message Batches API**; `classify.py --model / --batch / collect` |
| Analysis phases A0 / A1 (2026-09-23) | **A0 run on the pilot** (`plan/A0-RESULTS.md`), **A0 run wide** on the 99 QA-gated episodes / 10 bodies (`plan/A0-WIDE-RESULTS.md`), **A1 lexicon sentiment** on the pilot (`plan/A1-RESULTS.md`). The A1 **Claude pass is not run** — it costs money and is unapproved. Every figure is drawn **once per instrument facet** (`episodes.method`, `episodes.roles_source`); `analysis/facets.py` **refuses** to draw a pooled one, `analysis/facet_sweep.py` drives the sweep, `make a0-pooled` is the explicit escape hatch and stamps every file `POOLED_`. `make a0` still means the pinned 46-episode pilot in its own DB; `make a0-wide` is a separate target and a separate DB |
| QA window bug — **fixed 2026-10-16** (branch `showcase/qa-window-fix`, not merged) | `analysis/qa_transcripts.py` windowed an episode's caption reference by `window_start_s`, which is **0 in 28 of 29 b13 transcripts** (written before the ingest recorded the clip offset). That, not a bad transcript, is where `qIlAln3uR7U_e14`'s 115.7 % and `_e15`'s 771.1 % came from. Now 0/missing on an episode means *unknown*: the offset comes from `episode_start_s` in the JSON, else `episodes.start_s` in `state.sqlite` (`--state-db`, default `STATE_DB_PATH`), else **no caption metric at all** (empty `wer_vs_captions`, an `info` note saying why, not a WARN). `make qa` re-run 2026-10-16 (552 files, offline): `_e14` **10.1 %** (still WARN for its 6 pre-existing speaker-tag segments), `_e15` **9.2 %**; 35 b13 episodes re-windowed from the DB (5.3–33.9 %), 10 got their first caption check, 16 rows WARN→OK, **456 OK / 96 WARN → 472 OK / 80 WARN, 0 FAIL**. 5 Sunnyvale clips (`Chj4lSP9je4_e1–e5`, b13, no `episodes` row) now carry the skip note instead of a wrong number. 7 regression checks in `test_pipeline.py` section 23; `make test` 427. Before/after snapshot: `output/qa/latest_prefix_2026-10-16.csv` vs `latest.csv`. Evidence: `plan/BART-QIL-ASSESSMENT.md` |
| Money | `pipeline/spend.py`: RunPod **$3.32** of the $10 credit recorded (stop line $9.50, untouched this session), Claude **$2.89** of $10. `gpu_batch.py reconcile` settles a stuck batch (review M7); the RunPod key travels in the `Authorization` header (L1) |
| `make test` | **420 checks** (398 was stale; measured 2026-09-28, including the 5 new QA-freshness checks) |
| QA freshness (2026-09-28) | `dashboard/status.py` now dates its QA snapshot and alerts when it predates the last finished batch (`qa_stale`, serious) or is absent entirely (`qa_missing`, serious). It still does **not** run the gate. **Right now the snapshot is 2026-09-23 22:30 UTC and 299 transcripts have landed since** — `make qa` is unrun and unapproved |

**What part 2 needs from Puck** (`plan/T2-PLAN.md` §3): the per-body monthly
cap (3 h needs $10.00 of GPU against $6.18 spendable — 1 h costs $5.80), and
`CLAUDE_CAP_USD` raised to ~$30 or a smaller set of bodies, because the
classifier over 2,284 meeting-hours is $25.40 even at the batch rate.

### Hasan/Tucker corpus (2026-09-17 → done 2026-09-20)

| | Status |
|---|---|
| Architecture, decisions, sampling design | **Done** — written up, reviewed 4x |
| Analysis code (frequency, collocates, QA gate) | **Done and tested** — `make test`, 81 checks |
| Transcript QA gate (`analysis/qa_transcripts.py`) | **Done** — run it on every batch before loading |
| Progress dashboard + WhatsApp status digest | **Done** — `make dashboard`, `make status` |
| Notifier (`pipeline/notify.py`, WhatsApp via OpenClaw) | **Done** — needs `OPENCLAW_NOTIFY_TARGET` |
| Corpus metadata crawl | **Run** 2026-09-17/18 (`plan/catalog.tsv`); dates older than ~1 year are bucketed (SAMPLING.md) — frame not drawn yet |
| Download / transcode workers (`pipeline/downloader.py`, `pipeline/transcode.py`, `pipeline/state.py`, `pipeline/discover.py`) | **Built 2026-09-18**, tested (section 15), run on a first 4-video sample |
| `pipeline/gpu_batch.py` (propose/approve/deny/launch gate), `pipeline/requeue.py`, launchd worker plists | **Built 2026-09-18** — `launch b1 --dry-run` passes every check |
| GPU side: `pipeline/gpu_job.py` (pod worker + laptop driver), `pipeline/gpu_bootstrap.sh`, `pipeline/gpu_transport.py` (rsync/SSH), `plan/gpu-freeze.txt` | **Built 2026-09-18, NEVER RUN ON A POD.** Pins, pod image tag and the WhisperX 3.8 API calls are unverified until the first real launch (batch b1, est $0.10); section 16 of `make test` covers everything that can be checked offline |
| OpenClaw supervisor | **Not installed** — `docs/OPENCLAW_SETUP.md` has the whole procedure |

Under `output/` there are **20 real transcripts** (10 HasanAbi, 10 Ben Shapiro)
from April-May 2026 exploration: CPU faster-whisper medium, no diarization.
Useful for validation and QA calibration; **not** the corpus. Five Tucker videos
have captions only — that run died on an HTTP 429 before downloading audio.

## What was decided on 2026-09-17 (by Puck)

1. **Corpus = stratified sample**, speaker × month, with a quota of transcribed
   **audio-hours per cell** (default 3 h) rather than a video count, and a
   45-minute cap per video. `plan/SAMPLING.md`, `plan/build_frame.py`.
2. **Notifications and commands go to WhatsApp through OpenClaw**, not Discord.
   `pipeline/notify.py`, `plan/DECISIONS.md` #7.
3. **Every GPU batch needs Puck's approval** before a pod is created — the code
   refuses without it and OpenClaw asks him on WhatsApp. `DECISIONS.md` #10.
4. **A local progress page** on the supervisor Mac: `make dashboard` →
   http://localhost:8765 (read-only, loopback only).

## Read in this order

1. **`plan/README.md`** — index, why each component exists, glossary
2. **`plan/DECISIONS.md`** — 10 settled choices. Don't reopen without telling Puck why.
3. **`plan/PLAN.md`** — the build plan: architecture, state machine, phases, unresolved items
4. **`plan/SAMPLING.md`** — corpus design. Read before downloading anything.
5. **`plan/SCHEMA.md`** — database design
6. **`docs/TESTING.md`** — how to prove this machine works, and what "good output" means
7. **`docs/MODELS.md`** — which model for what, with prices and dates
8. **`docs/OPENCLAW_SETUP.md`** — supervisor setup, approvals, heartbeat
9. **`docs/REVIEW-2026-09-17.md`** — every bug found in the handoff review, and its fix
10. **`analysis/METHODS.md`** — the statistics, with formulas and failure modes

## First commands

```bash
./setup.sh              # environment, deps, test suite
make test               # 81 checks end to end; prints "All checks passed."
make smoke              # test of systems: offline checks + free credential checks
make dashboard-demo     # see the progress page on synthetic data
```

If `make test` and `make smoke` pass, the analysis half and the tooling work on
this machine. `docs/TESTING.md` explains every line they print.

## The things that will bite you

1. **Don't leave a rented GPU running.** An idle 4090 is ~$170/week — several
   times the project budget. Pods are rented per batch and destroyed.
   `DECISIONS.md` #2.
2. **Don't create a pod without Puck's approval.** The launcher must refuse
   unless the batch row says approved, and OpenClaw must ask him. Keep the
   RunPod prepaid balance small; it is the only hard ceiling. `DECISIONS.md` #10.
3. **Don't download from a rented server.** YouTube bot-flags datacenter IPs.
   The laptop's residential IP is the asset. `DECISIONS.md` #1.
4. **`state.sqlite` must live on real local disk.** Not a synced folder, network
   mount, iCloud or Dropbox: SQLite needs POSIX byte-range locking those don't
   provide, and fails with a bare "disk I/O error" or a 0-byte file. (There is a
   0-byte `output_sim/corpus.sqlite` in this repo from exactly that.)
5. **Do Phase 0 before spending anything.** Validate transcription quality by
   hand on a few videos. Whisper hallucinates fluent text on silence and music —
   the QA gate already found one such loop in `AAiFho_u_GI` at 21:39.
6. **The 20 existing transcripts have no human reference.** All 20 `.en.vtt`
   files are YouTube's own ASR. Only the five Tucker videos have manual
   captions, and their audio was never downloaded. Phase 0 starts by fetching
   those five. `plan/PLAN.md` §6.
7. **yt-dlp needs a JavaScript runtime** (`brew install deno`) and the
   `yt-dlp[default,curl-cffi]` extras, or YouTube downloads fail mid-crawl with
   403s. `setup.sh` checks this.
8. **Claude 5 models reject `temperature`** and think adaptively by default. Any
   scoring call needs `thinking={"type":"disabled"}` and no sampling
   parameters. `docs/MODELS.md`.
9. **Never commit `.env`.** It holds API keys, is gitignored, and is excluded
   from the transfer bundle — recreate it from `.env.example`.

## Git

The repo is under git (one commit, `Import project`). As of 2026-09-17 the
working tree has uncommitted changes — read `docs/REVIEW-2026-09-17.md`, then
commit them with a message that says what changed. If git complains about a
`.lock` file, look for `.git/stale-*.lock.*`: those are leftovers from a remote
session that could not delete files, and are safe to remove.

## What to ask Puck before starting

1. **Channel URLs.** The three in `plan/channels.tsv` are guesses. Which Hasan
   channel (clips vs full streams) materially changes what "Hasan" means.
2. **Quota hours per cell** (default 3 h → ~290 audio-hours for 32 months) and
   the **window** of months to cover.
3. **Videos longer than 90 minutes**: `--max-min 90` currently excludes them
   entirely. With the 45-minute cap they could be included — that is a decision
   about format, not about cost.
4. **First 45 minutes or a random 45-minute window?** The first 45 minutes of a
   stream is often pre-show and music.
5. Does a throwaway Google account exist for yt-dlp cookies, if they become necessary?
6. Monthly spend cap for the guard (default $150 — the real ceiling is the
   RunPod prepaid balance).

## Working style

- He wants **brief, to the point** answers, but **detailed explanations of
  complicated formulas**. `analysis/METHODS.md` and `plan/SAMPLING.md` are the
  register to match.
- He asks direct questions and wants direct answers, including "this is the
  wrong approach."
- He values being told what's uncertain. Numbers in `plan/PLAN.md` that are
  estimates (notably the 25x realtime throughput) are flagged as such — don't
  quietly launder them into facts.
