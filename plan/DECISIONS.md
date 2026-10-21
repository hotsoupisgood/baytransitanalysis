# Settled decisions

#1-#8 were argued through on 2026-09-14; #7 was revised and #9-#11 added on
2026-09-17. Don't reopen one without telling Puck why. Listed with the reasoning
so a future session can judge whether the premise still holds.

| # | Decision | Why | Reopen if |
|---|---|---|---|
| 1 | **Download from the laptop, never a rented server** | YouTube heavily bot-flags datacenter IPs; residential is the asset. Fixes (cookies, client rotation, proxies) exist but none stay fixed. | You're willing to pay for residential proxies |
| 2 | **GPU rented in batches, never left running** | Downloading takes weeks, transcription takes hours. An always-on 4090 is ~$170/wk idle — several times the whole budget. | Never |
| 3 | **RunPod + WhisperX over the Groq API** | Only because diarization is in scope. Groq is $80/2000hr with no speaker labels; self-host is ~$40 *with* them. | Diarization gets dropped — then Groq wins on simplicity |
| 4 | **16 kHz mono opus before upload** | ~7 MB/audio-hr vs ~60 (measured: 7.4 vs 58). Whisper downsamples to 16k mono anyway, so it is *effectively* lossless for ASR — Phase 0 verifies that by transcribing one video from both. | Never (but keep opus, not m4a, if you ever re-run) |
| 5 | **SQLite as the only state** | Concurrent workers, resumability, and the existing code's habit of clobbering JSON summaries. | Never |
| 6 | **Plain scripts for ETL; OpenClaw only supervises** | An LLM in a 2,000-iteration I/O loop is slower, costlier, non-reproducible. Agents are for the judgment calls. | Never |
| 7 | **WhatsApp through OpenClaw for progress and commands** *(revised 2026-09-17; was Discord)* | Puck already has WhatsApp linked to OpenClaw, so it is one place for alerts, status and batch approvals, with no extra account and no webhook to leak. `pipeline/notify.py` keeps a Discord webhook backend behind the same `notify()` call. | The gateway proves unreliable (watch the outbox), or WhatsApp blocks the linked number — then flip `NOTIFY_BACKEND=discord` |
| 8 | **No $30-40/mo rental box** | At that price it is CPU-only (GPU starts ~$200/mo), and its datacenter IP is actively bad for the one thing it would be good at. | You want an always-on agent host for reasons beyond this project |
| 9 | **Corpus = stratified sample: speaker × month, a quota of audio-HOURS per cell (3 h), 45-min cap per video** *(2026-09-17, Puck)* | A census is dominated by whoever uploads most, and that imbalance is confounded with format, era and topic at once. Hours rather than a video count because the statistics are estimated from tokens, and one speaker's videos are 4x longer than another's. ~290 audio-hours instead of ~2,000. | A specific question needs a census of one channel — that is a different study, drawn into its own frame |
| 10 | **Every GPU batch needs Puck's explicit approval before a pod is created** *(2026-09-17, Puck)* | Spend authority stays with the human. Enforced in three places, none of which is a prompt: OpenClaw's exec approval on the launch command, a code check that the batch row is approved and the estimate is under the cap, and a deliberately small RunPod prepaid balance. | Puck says so after enough batches have run cleanly — then relax to "autonomous under the cap", keeping the code check |
| 11 | **A read-only local dashboard on the supervisor Mac; approvals never happen there** *(2026-09-17)* | Puck wanted a page he can look at. It binds to loopback, opens `state.sqlite` read-only, and has no endpoint that changes anything — a button on a web page would be a second, weaker path to spending money. | Never for write access; expose it beyond loopback only through an SSH tunnel or Tailscale |

## Rejected alternatives, briefly

- **Groq API for everything** — cheapest for plain ASR, no diarization. Good
  fallback if the GPU route stalls; the pipeline should keep a Groq adapter
  behind the same interface so it's a config switch.
- **AssemblyAI ($0.15/hr + $0.02/hr for speaker labels)** — ~$50 for the
  stratified sample, ~$340 for 2,000 hours. Genuinely the least work of any
  option and its Conformer/RNN-T model hallucinates less than Whisper on noisy
  audio. **If the GPU route eats more than two evenings, switch to this and
  stop.** The time is worth more than the money.
- **Always-on GPU** — see #2.
- **Email via self-hosted mail server** — SPF/DKIM/DMARC and IP reputation to
  deliver "batch 14 of 220 done".
- **Discord webhooks** — see #7; still implemented as the fallback backend.

## Proposed 2026-09-21, pending Puck (PROPOSAL-transit.md appendix)

Puck approved the transit/density pivot and the T0 run on 2026-09-20; the
wording below is the proposal's appendix, appended verbatim for his sign-off.
Nothing above changes; #9 stays true for the Hasan/Tucker corpus.

| # | Decision | Why | Reopen if |
|---|---|---|---|
| 12 | **Second corpus: Bay Area council, county and transit-board meetings, Sep 2025 – Sep 2026, a census of regular meetings; only density/transit episodes are transcribed, with a per-body monthly cap on transcribed episode-hours** *(proposed 2026-09-21, pending Puck; cap value TBD)* | Rates per meeting-hour need every meeting counted; locating costs ~nothing on captioned bodies and transcription scales with episode-hours ($2–12 at 2–10 %), so the reason for #9's sampling does not apply. The cap is the spend valve and keeps big cities from dominating tokens | A body's cap binds every month — then raise it for that body or stratify tokens in the model |
| 13 | **Caption method: locate before transcribing.** YouTube auto-captions (or a cheap ASR-only pass for Granicus bodies) find candidate windows; WhisperX with diarization runs only on the extracted, padded episodes *(proposed 2026-09-21, pending Puck)* | Whole meetings are 3.8 h of which 2–5 % is on topic; transcribing everything costs 20–50x the GPU and, worse, 177 GB and weeks of downloads for 11 GB of use | Auto-captions prove too poor to locate (T0 recall < 0.9) — then the ASR-only pass runs on YouTube bodies too |
| 14 | **The episode is the unit; it is defined by two keyword families at a low density threshold plus a pinned Claude classifier (model + prompt version recorded per row), tagged density / transit / both** *(proposed 2026-09-21, pending Puck)* | A single density threshold cannot tell a two-minute liaison report from two stray uses of "train" (SMART pilot). The classifier is a screen with a measured error, not the measurement | The classifier's recall or precision on the hand-labelled set drops below 0.9 / 0.8 on a new body type |
| 15 | **Speaker labels stay anonymous clusters with a role (dais / staff / public); no names are extracted, stored or indexed, and episode transcripts pass a name-redaction step before loading** *(proposed 2026-09-21, pending Puck)* | Public commenters are private individuals on a public record. The research question needs roles, not identities | Never for public commenters; a body-level "who on the dais" mapping only with an explicit ethics decision |
| 16 | **The Hasan/Tucker corpus (143 videos, `output/whisper`, `corpus.sqlite`) is frozen as a finished side corpus: not re-sampled, not deleted, `kind=show` rows and their analysis code untouched** *(proposed 2026-09-21, pending Puck)* | It is complete and its frame is drawn; reusing the pipeline for meetings must not change what that study measured | A new question needs more of it — then draw a new frame, per #9 |

Implementation notes that follow from these if approved (T1, 2026-09-21):
`videos.kind`, the `episodes` / `speaker_roles` / `classifier_calls` tables
(SCHEMA.md §1); the name-redaction step in #15 is **not built yet** — `make
load` on episode transcripts should wait for it.

## Proposed 2026-09-22, pending Puck (T2 part 1, plan/T2-PLAN.md)

| # | Decision | Why | Reopen if |
|---|---|---|---|
| 17 | **T2 classifier instrument = `claude-sonnet-4-6`, prompt `t0-a4a8e55760`, through the Message Batches API** (50 % off); Haiku 4.5 is not adopted *(proposed 2026-09-22, pending Puck)* | The decision rule written before the test (≥ 0.90 Haiku; 0.80–0.90 Haiku + Sonnet re-check; < 0.80 Sonnet) came out at 0.758 about-vs-not agreement on the 198 T0 windows: Haiku demotes 39 of 94 `about_transit` windows to `mention_only`, nearly all at BART, i.e. it would cut the board transit rate by a third. Same instrument as T0/T1 keeps every row comparable | A re-coded codebook is validated on a hand-labelled set (the pending test-3 sheet) and a cheaper model reaches ≥ 0.90 on it |
| 18 | **Hard spend stops in code:** RunPod stop line = `RUNPOD_CREDIT_USD` − `RUNPOD_CREDIT_MARGIN_USD` ($10 − $0.50 = $9.50 recorded + estimate), `CLAUDE_CAP_USD` $10 on `classifier_calls`; `propose`, `launch` and `classify.py` refuse past them *(proposed 2026-09-22, pending Puck)* | The prepaid balance is the only real ceiling; the ledger matched it to the cent on 2026-09-22 ($3.32 recorded, $6.67 left). Refusing in code means a stuck supervisor or a mis-typed approval cannot spend the last dollar | Puck tops up the credit — then raise `RUNPOD_CREDIT_USD` to the new balance, nothing else |
| 19 | **Population = regular meetings only:** special meetings, work sessions, budget workshops and study sessions are excluded by the per-body title regex in `plan/channels.tsv`; multi-part uploads of one meeting are kept whole *(proposed 2026-09-22, pending Puck)* | PROPOSAL §2.1 counts regular meetings; a special session is a different exposure (one topic, often housing) and would inflate the rate for the bodies that hold many. They stay listed (in the saved listings), so adding them later is a regex change and a re-run, not a re-crawl | The episode rate per body-month is dominated by what specials contain — then add `special` as a `meeting_kind` column and count both |
