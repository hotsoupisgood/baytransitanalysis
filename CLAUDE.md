# yt-political-transcripts

Research project: political polarisation on YouTube, via transcript analysis
(word frequency, topic, sentiment, qualitative). Owner: Puck.

## New here?

**Read `START_HERE.md` first.** It covers current state, what's built and what
isn't, the reading order, the mistakes that cost real money, and the open
questions only Puck can answer. Then `plan/README.md`.

Run `./setup.sh` before building anything, then `make test` and `make smoke`
(`docs/TESTING.md`). `make` lists the common tasks.

## Repo rules

- `.env` holds API keys and is gitignored. Never commit or echo key values.
- `output/` is large and mostly gitignored. Never commit audio.
- **Transcript JSON contract.** New pipeline output is an object:
  `{video_id, speaker, month, duration_s, transcribed_s, model, diarized,
  host_speaker, host_confidence, segments: [{start, end, text, speaker?}]}`
  (`plan/PLAN.md` §1). The 2026-04 exploratory files are a bare segment list
  named `<id>.<model>.json`; `analysis/legacy_to_contract.py` wraps them.
  `load_db.py`, `qa_transcripts.py`, `legacy/meanness_analysis.py` and
  `legacy/stance_analysis.py` read both. Adding keys is fine; renaming or nesting is not.
- State lives in `state.sqlite`, not in JSON files or directory listings.
- Run `analysis/qa_transcripts.py` on every batch of transcripts **before**
  `make load`. A hallucinated passage looks like data to every statistic
  downstream.
- Never create a GPU pod without Puck's approval for that batch
  (`plan/DECISIONS.md` #10).
- Git: the repo has one commit and uncommitted changes from the 2026-09-17
  review. If git reports a stuck `.lock`, see `.git/stale-*.lock.*` (leftovers
  from a session that could not delete files).

## Calling Claude models from this repo (2026-09)

- `claude-sonnet-5` / `claude-opus-5` **reject any non-default `temperature`,
  `top_p` or `top_k`** (HTTP 400) and think adaptively unless you pass
  `thinking={"type": "disabled"}`. With `max_tokens=5` and thinking left on, the
  reply budget goes to thinking and the answer comes back empty.
- Noise is therefore measured by resampling (k samples, median), not suppressed
  with `temperature=0`. `legacy/meanness_analysis.py --samples 3` does this.
- Scoring model comes from `SCORING_MODEL` (default `claude-sonnet-4-6`, the
  model the existing `output/meanness/` scores came from). Scores are comparable
  only within one model and one `prompt_version`; both are recorded per video.
  `docs/MODELS.md` covers model choice, prices and retirement dates.

## Fixed on 2026-09-17 (don't reintroduce)

Full list with evidence in `docs/REVIEW-2026-09-17.md`:

- `meanness_analysis.py` / `stance_analysis.py` rewrote `summary.json` from the
  current run only (`output/meanness/summary.json` held 1 of 10 videos). They
  now merge on write; `--rebuild-summary` reconstructs it from the per-video files.
- `score_chunk` returned `0` for an unparseable reply ("maximally polite"), and
  read "7/10" as 10. It now parses properly and returns `None` on failure.
- `stance_analysis.py` counted substrings ("police" as a mention of "ice") and
  labelled anything unparseable NEUTRAL.
- `word_freq.py --mode kwic` matched the node word as a substring: 98% of `ice`
  "mentions" in the real transcripts were false hits. Now whole-word.
- `word_freq.py` filtered "right" as a filler while keeping "left", which made
  every left/right comparison asymmetric by construction.
- `hasan_compare.py` computed WER with Whisper as the reference, which halves
  the penalty for hallucinated text, and counted YouTube's `[&nbsp;__&nbsp;]`
  profanity mask as three error tokens.

## Statistical note

Chunks are nested in videos, which are nested in speakers. Treating chunks as
independent observations inflates significance badly — there are ~10 videos, not
~124 chunks, of real information. Any speaker comparison needs a hierarchical /
mixed model with a random intercept for video. Puck does Bayesian hierarchical
modelling; don't hand him a t-test.
