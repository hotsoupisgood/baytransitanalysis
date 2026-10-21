# plan/ — what's in here and why

This folder is the specification for the transcription pipeline. It exists so a
Claude Code session on the orchestrator laptop can pick up the work without
re-deriving the reasoning, and so Puck can audit choices months later without
trusting his own memory of a Sunday afternoon.

**Rule for anyone editing these files:** every parameter gets a stated reason.
A number without a justification is a number nobody can later change safely,
because no one knows what it was protecting against.

## Read in this order

| File | What it is | Read it when |
|---|---|---|
| `README.md` | this file — index, glossary | first |
| `DECISIONS.md` | 11 settled architectural choices + the rejected alternatives | before proposing any change |
| `PLAN.md` | the build plan: architecture, state machine, components, phases, review findings | before writing code |
| `SAMPLING.md` | corpus design — which videos get transcribed and why | before downloading anything |
| `SCHEMA.md` | database design, every column justified | before writing the loader |
| `channels.tsv` | speaker → channel-tab URLs. **Confirm these before crawling** | step 0 |
| `crawl_metadata.sh` | metadata-only crawl → `catalog.tsv` (+ `logs/`) | step 1 |
| `build_frame.py` | cell table + draws the sample → `frame.csv` | step 2 |
| `gpu-freeze.txt` | pinned pip versions of a GPU pod that worked (created in Phase 0) | every pod bootstrap |
| `catalog.tsv` | *generated* — every video on each channel, metadata only | — |
| `frame.csv` | *generated* — **the corpus definition**. The sample, fixed. | — |
| `frame_cells.csv` | *generated* — per cell: eligible, sampled, quota, shortfall | — |
| `frame_meta.json` | *generated* — parameters, seed, catalog sha256 | — |

Analysis code lives in `../analysis/`:

| File | What it is |
|---|---|
| `METHODS.md` | every statistic used, written out with its formula and failure mode |
| `simulate.py` | synthetic corpus with a planted effect and a planted non-effect |
| `load_db.py` | transcript JSON → SQLite + FTS5 index (derived, disposable) |
| `word_freq.py` | overall frequency, and collocates in a ±N-sentence window |
| `qa_transcripts.py` | the quality gate every batch of transcripts passes before loading |
| `legacy_to_contract.py` | wraps the 2026-04 transcripts in the current contract |
| `test_pipeline.py` | 45 checks end to end — run it after any change |

`../backup.sh` syncs the irreplaceable artefacts (m4a + transcripts + frame.csv) to Backblaze B2. Derived files are deliberately excluded.

`frame.csv` is the important generated file. Once drawn, it defines the study.
Any analysis reports its n from that file, not from whatever happens to be on
disk — otherwise a failed download silently becomes a study-design change.

## Why the pipeline is split the way it is

The whole architecture follows from one asymmetry: **downloading is slow and
rate-limited (weeks); transcription is fast and expensive-per-minute (hours).**

Everything else falls out of that:

- They run on different machines because they want different things — the
  downloader wants a residential IP and patience; the transcriber wants a GPU
  and to not exist for very long.
- The GPU is rented in batches because a 4090 idling while `yt-dlp` trickles
  costs ~$170/week for nothing.
- There's a durable queue between them so neither blocks the other.
- State lives in SQLite because two loosely-coupled workers plus resumability
  plus retries is exactly the problem a transactional store solves, and exactly
  the problem that JSON files have already failed at once in this repo.

## Why each component exists

| Component | Without it | Cost of including it |
|---|---|---|
| Metadata crawl (`crawl_metadata.sh`) | You design the corpus blind and discover empty Tucker months after weeks of downloading | Minutes, no audio |
| Sampling frame (`build_frame.py`) | Corpus composition decided by upload rates, not by you; imbalance confounded with format, era and topic at once | An afternoon of thought |
| Opus transcode | ~2 GB of uploads instead of ~17 GB for the sample (7.4 vs 58 MB per audio-hour, measured) | One ffmpeg call per file |
| SQLite state | The `summary.json` corruption already in this repo, but pipeline-wide and unnoticed | ~50 lines |
| Batched GPU rental | ~$170/week of idle GPU | Scheduling logic |
| Per-video writeback | A reclaimed pod loses the whole batch instead of one video | Slightly chattier sync |
| Diarization | Guest speech scored as the host's; Tucker's interviews measure his guests | ~2x GPU time, HF token setup |
| Duration cap | One 6-hour stream dominates its own stratum | A truncation flag to model |
| WhatsApp notifier + local dashboard | You discover a 3-day stall on day 4 | `pipeline/notify.py`, `dashboard/` |
| Per-batch approval | An agent rents a GPU you did not agree to | One WhatsApp reply per batch |
| Transcript QA gate | A decoder loop becomes "data" | One command per batch |
| Spend guard | An orphaned pod bills quietly until someone notices | ~10 lines + a `spend` table |
| Phase 0 validation | You find out Whisper hallucinates on silence at hour 2,000 instead of hour 3 | Half a day, ~$1 of GPU |

## Glossary

Terms used across these docs.

- **Diarization** — "who spoke when". Labels each segment with an anonymous
  speaker cluster (`SPEAKER_00`, `SPEAKER_01`). It does *not* identify people;
  mapping a cluster to "the host" is a separate heuristic step, and the
  numbering is not stable across videos.
- **DER (diarization error rate)** — fraction of audio time assigned to the
  wrong speaker. The diarization analogue of WER.
- **WER (word error rate)** — (substitutions + insertions + deletions) ÷
  reference words. Note it averages, so rare-but-severe failures like
  hallucinated passages hide inside a good-looking number.
- **Hallucination** — Whisper's decoder inventing fluent text where there was
  no speech (silence, music, crosstalk). A property of encoder-decoder ASR;
  frame-synchronous models like RNN-T fail by dropping words instead. This is
  the specific risk in stream audio and the reason Phase 0 exists.
- **VAD (voice activity detection)** — pre-filter that drops non-speech audio.
  The mitigation if Phase 0 shows hallucination is bad.
- **Forced alignment** — snapping transcript words to precise audio timestamps
  using a separate acoustic model. Whisper's own segment times are loose;
  WhisperX adds alignment to get word-level accuracy, which is what makes
  merging with diarization boundaries meaningful.
- **Stratified sample** — split the population into groups, sample a fixed
  number from each, instead of sampling at random from the whole pile. Here:
  channel × month. Precision on a between-speaker comparison is driven by the
  *smallest* stratum, not the total n.
- **Stratum / cell** — one group. Here, one (speaker, month) pair.
- **Realtime factor (RTFx)** — audio-seconds processed per wall-clock second.
  25x means one hour of audio takes ~2.4 minutes. The whole cost model is
  linear in this, which is why Phase 0 measures it rather than trusting the
  estimate in `PLAN.md`.
- **Chunk** — a ~350-word window of transcript, the unit the meanness scorer
  rates. Chunks are nested in videos, which is why they are not independent
  observations.
- **Partial pooling** — hierarchical-model behaviour where thin groups get
  shrunk toward the overall mean. Useful, but if the design is badly unbalanced
  it quietly does all the work in the thin strata — an argument for balancing
  the sample rather than leaning on the model.
