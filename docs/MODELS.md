# Which model for what

Checked 2026-09-17. Prices and retirement dates change; the *reasons* are the
part to keep. Verify with `openclaw models list --provider anthropic`,
`scripts/check_credentials.py`, and the linked docs before relying on a number.

## The rule that matters for research

**A model used to measure something is an instrument.** Pin one model ID and one
prompt version for a whole measurement, record both in every output row
(`legacy/meanness_analysis.py` and `legacy/stance_analysis.py` do), and never mix results from
two models in one comparison. Re-scoring 300 hours because a default changed
mid-run costs more than the tokens ever did.

---

## 1. The supervisor agent (OpenClaw)

| Use | Model | Why |
|---|---|---|
| Day-to-day supervision, heartbeat, status replies | `anthropic/claude-sonnet-5` | $2/$10 per Mtok; the work is running scripts and reading their output |
| Building Phases 1-3, debugging a stuck pipeline | `anthropic/claude-opus-5` | $5/$25; the strongest agentic coding model in the family — switch the session with `/model` |
| A hard design review, once | `anthropic/claude-fable-5-1` | $10/$50, adaptive thinking always on; not for routine work |

Heartbeat cost, roughly: an isolated, light-context heartbeat is a few thousand
tokens. At 30-minute intervals inside 16 active hours that is ~32 runs/day; on
Sonnet 5 that lands in the low tens of dollars a month, so `isolatedSession:
true` and `lightContext: true` are not optional (`docs/OPENCLAW_SETUP.md`). Put
the key in its own Anthropic Console workspace with a monthly spend limit.

Two Claude-5 behaviours the config has to respect:

- **Sampling parameters are rejected.** `temperature`, `top_p`, `top_k` at any
  non-default value return HTTP 400 on Sonnet 5 and Opus 5.
- **Thinking is adaptive by default**, and manual `budget_tokens` is gone. Pass
  `thinking: {"type": "disabled"}` for short structured replies; use the
  `effort` parameter (or OpenClaw's `params.thinking`) to control depth.
- Claude Sonnet 5 uses a new tokenizer: the same text is ~30% more tokens than
  on Sonnet 4.6, so a straight price comparison understates it.

## 2. The scoring model (meanness, stance, any LLM-coded variable)

Default: **`claude-sonnet-4-6`** (`SCORING_MODEL` in `.env`). Reasons, in order:

1. It produced the existing `output/meanness/` scores, so re-scoring is not
   forced;
2. it is an old-style dated-family ID whose retirement is *not sooner than
   2027-02-17*, which comfortably covers this project;
3. it accepts `temperature` if you ever want determinism, which the Claude 5
   models do not.

Switch to `claude-sonnet-5` only at the start of a measurement, never in the
middle, and re-score everything already scored. Avoid `claude-haiku-4-5` as the
instrument: its retirement window opens 2026-10-15.

**Haiku 4.5 as the episode classifier — measured 2026-09-22, not adopted.**
`claude-haiku-4-5-20251001` (the dated id this key sees; the bare alias is
not listed) re-labelled the 198 T0 windows from byte-identical text
(`scripts/validate_classifier.py`, `output/t2/validate/…/report.md`):
about-vs-not agreement with the Sonnet 4.6 labels **0.758** (κ 0.52), exact
0.712 (κ 0.60); precision on Sonnet's `about_*` 0.98 but recall 0.67 — it
demotes 39 of 94 `about_transit` windows (nearly all BART board items) to
`mention_only`, so it would cut the transit rate at boards by a third. Under
the rule in `plan/T2-PLAN.md` (< 0.80 → Sonnet) the T2 instrument stays
**`claude-sonnet-4-6` via the Message Batches API** (50 % off). Two facts to
keep: Haiku's minimum cacheable prefix is 4,096 tokens, so the 1,531-token
codebook never caches on it (198 calls cost $0.52, not the $0.10 a cached run
would); and `classify.py --model` records a different `prompt_version`
(`t0-…/<model>`) per model, so the two instruments never mix in one row.

How the scoring calls are shaped (already in the scripts):

- `thinking={"type": "disabled"}`, no sampling parameters, `max_tokens` small.
- **k = 3 samples, median kept**, with the spread recorded (`score_sd`). This
  measures the noise instead of hiding it, and is what `CLAUDE.md`'s "use
  temperature=0 and sample k=3" becomes on models that refuse temperature.
- An unparseable reply is `None`, never a number.

For the full corpus, two upgrades worth taking:

- **Message Batches API — 50% off**, up to 100k requests per batch, results
  within 24 h. Scoring is the textbook case: it is not interactive.
- **Structured outputs** (`output_config.format` with a JSON schema) removes
  reply-parsing entirely. Supported on Sonnet 4.5/4.6/5, Opus 4.5+ and Haiku
  4.5, and it works inside batches. Raise `max_tokens` to ~32 for the JSON.

**Validate before you believe it.** Have Puck hand-code ~100 chunks blind,
compare against the model's median score (weighted kappa / Spearman), and report
that agreement next to any result. An LLM rating without a human-agreement
number is an opinion with a decimal point.

## 3. Transcription and diarization (the rented GPU)

Stack: **WhisperX** (faster-whisper for ASR + wav2vec2 forced alignment +
pyannote diarization), per `DECISIONS.md` #3.

- WhisperX 3.8.x requires **Python ≥3.10 and <3.14**, `torch ~=2.8`,
  `ctranslate2 ≥4.5`, and **pyannote-audio ≥ 4.0** — which means the diarization
  model is now **`pyannote/speaker-diarization-community-1`**, not 3.1. Accept
  that model's conditions with the same HF account as `HF_TOKEN`
  (`make creds` checks it). The 3.1 + `segmentation-3.0` pair only matters if
  you pin an older WhisperX.
- ASR model: **`large-v3`** is the plan's default. Worth A/B-ing in Phase 0
  against **`large-v3-turbo`** (much faster decoder) on the same validation
  videos: if WER and hallucination are equal, the turbo model halves the bill.
  Decide with numbers, not vibes.
- Hallucination is the risk that matters here (stream audio is full of music and
  silence). Mitigations in order: VAD prefilter (WhisperX does this),
  `condition_on_previous_text=False`, then the QA gate. If Phase 0 shows it is
  bad, an RNN-T/TDT model (NVIDIA Parakeet family) drops words instead of
  inventing them — but it is a different pipeline, so treat it as a fallback,
  not a plan.
- **No file longer than the 45-minute cap reaches the GPU** under the sampling
  design (`ffmpeg -t 2700` at transcode). The split-and-stitch machinery in
  `PLAN.md` §4.5 is only needed if that cap is lifted.

## 4. Transcription on the Mac (tests only, never the corpus)

Apple Silicon: `pip install mlx-whisper`, then
`mlx_whisper --model mlx-community/whisper-large-v3-turbo audio.opus -f json`.
Fast enough to check that a downloaded file contains what it should. Intel Mac
or no MLX: `faster-whisper` on CPU, `small` or `medium` — the 2026-04 runs took
~2-3 s per audio-second with `medium` on this laptop, which is why the corpus
goes to a GPU.

Never mix locally-transcribed files into the corpus: one corpus, one ASR
configuration, recorded in every file's `model` field.

## 5. If the GPU route stalls — the hosted fallback

`DECISIONS.md` says: if the GPU route eats more than two evenings, switch and
stop. Current prices: **AssemblyAI Universal-2 $0.15/audio-hour** plus
**$0.02/hour for speaker labels**; their newer Universal-3.5 Pro is $0.21/hour.
For the stratified sample (~290 capped audio-hours) that is about **$50**,
against roughly $5-10 of GPU plus the setup evenings. For the full back-catalog
(~2,000 h) it is ~$340.

Keep the transport behind one interface so this is a config switch, and keep the
output in the same contract shape.

## 6. Downstream analysis (topic, sentiment, polarisation)

These are the project's stated methods; none of them is settled, so this is a
shortlist with reasons rather than a decision.

- **Topic.** For inference, a **structural topic model** (R `stm`) with speaker
  and month as prevalence covariates fits how this corpus is designed and how
  Puck models — covariate effects come out of the model rather than being bolted
  on afterwards. **BERTopic** (sentence-transformer embeddings + HDBSCAN) is the
  better exploratory tool. Run the topic model on host-only speech, and on
  chunks, not whole videos.
- **Sentiment.** Off-the-shelf sentiment classifiers are trained on tweets and
  reviews; a two-hour political monologue is out of domain, and "negative" is
  not the construct this project is about. Prefer (a) a transparent dictionary
  baseline, (b) LLM coding against a written codebook, validated against human
  coding as in §2, and report both.
- **Polarisation specifically.** Out-group animosity — how the speaker talks
  about the other side — is closer to the research question than valence.
  A dictionary of political out-group terms plus the LLM codebook gives a cheap
  measure and an expensive one that can be compared.
- **Do not build on Google's Perspective API**: it stopped taking new quota
  requests in February 2026 and shuts down after 31 December 2026. If a toxicity
  baseline is wanted, use a self-hosted classifier (e.g. Detoxify) so it still
  runs next year.
- **Sponsor reads contaminate word frequency.** `com` is the single most
  distinctive token for Ben Shapiro in the 20 exploratory transcripts, because
  every video contains an ad. Fetch SponsorBlock segments at discovery time
  (`GET https://sponsor.ajay.app/api/skipSegments?videoID=<id>&categories=["sponsor","selfpromo"]`,
  no key, returns `segment: [start, end]` pairs) and store them, then drop those
  spans from the analysis corpus. Coverage is community-driven, so it is a
  filter, not a guarantee.
- **Name spellings drift** across an ASR corpus ("Platner"/"Plattner",
  "Erika"/"Erica"). Keep an alias table and normalise after transcription;
  feeding a name list to Whisper as an initial prompt can *increase*
  hallucination of those names.

## Sources

- Claude models, prices, context: https://platform.claude.com/docs/en/about-claude/models/overview
- Retirement dates: https://platform.claude.com/docs/en/about-claude/model-deprecations
- Sonnet 5 migration (temperature, thinking, tokenizer): https://platform.claude.com/docs/en/models/sonnet-5/migration-guide
- Structured outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Message Batches API: https://platform.claude.com/docs/en/build-with-claude/batch-processing
- WhisperX: https://github.com/m-bain/whisperx and https://pypi.org/project/whisperx/
- pyannote community-1: https://www.pyannote.ai/blog/community-1
- yt-dlp JS runtime requirement: https://github.com/yt-dlp/yt-dlp/wiki/EJS
- yt-dlp PO tokens: https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide
- AssemblyAI pricing: https://www.assemblyai.com/blog/speech-recognition-cost
- RunPod RTX 4090 pricing: https://www.runpod.io/gpu-models/rtx-4090
- Perspective API shutdown: https://www.lassomoderation.com/blog/what-is-perspective-api/
- SponsorBlock API: https://wiki.sponsor.ajay.app/w/API_Docs
