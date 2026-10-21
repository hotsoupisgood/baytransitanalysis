# Analysis plan — density and transit talk in Bay Area public meetings

Draft 2026-09-21 for Puck. Companion to `PROPOSAL-transit.md` (data) and
`analysis/METHODS.md` (the statistics already implemented). WHY before WHAT;
every number is an estimate unless it cites a measurement.

## 0. What we are measuring, in one paragraph

Who says what about housing density and transit in Bay Area council and
transit-board meetings, how they say it, and how that varies by body, by
speaker role (dais vs public comment), and over time. The units, smallest
first: a **turn** (one speaker cluster's contiguous speech inside an episode,
with a role), an **episode** (a contiguous stretch of a meeting about density
and/or transit, from the classifier), a **meeting**, and a **body × month
cell** with meeting-hours as exposure. Inference is about rates, frames,
stance and tone as functions of role, body, family (density/transit) and
time — never about identified individuals.

## Scope note (Puck, 2026-09-21 21:20)

**No modelling yet.** Phases A0–A1 are exploratory plots only: term frequency
per speaker, and sentiment. The hierarchical models in §4 stay in the document
as *later, if wanted*; nothing below §3 is scheduled.

## 1. Exploratory plots — rates (no models, $0)

- Episode-minutes per meeting-hour by body and month, split density/transit
  (stacked bars); public-comment share of episode time by body (bars);
  speakers per episode (histogram); mention-only counts alongside.

## 2. Term frequency per speaker (existing `word_freq.py`, $0)

"Speaker" = one diarized cluster inside one episode, with its role (dais /
public / unknown) and body. Plots:

- Top-30 terms per role, per body, per family — rate per 1,000 words, with
  cluster-bootstrap CIs by episode (`METHODS.md` §7).
- Public vs dais keyness on the same episodes (G² / log ratio scatter,
  labelled), and density vs transit.
- Frame-lexicon rates (`plan/frames.yaml`: traffic/parking, neighborhood
  character, safety, environment/CEQA, affordability/displacement, property
  values, process, fiscal, state-mandate, YIMBY, transit service, transit
  safety) per 1,000 words by role × body, and by month (small multiples).
- Per-speaker view: a dot per speaker (x = talk seconds, y = frame rate or
  keyword rate), coloured by role, faceted by body — shows whether a few long
  speakers drive the rates.
- KWIC tables for the top frame terms by role, for reading.
- `make trends` / `spread` / `anim` rerun with body as the series, role as a
  facet.

## 3. Sentiment per speaker

Two passes, cheap first:
- **Lexicon pass ($0):** VADER compound score per turn and per speaker (mean
  and min), plus a hostility word-list rate. Plots: distribution by role and
  body (violins), sentiment vs talk length, sentiment by frame used,
  sentiment over the course of an episode (public comment usually late).
- **Claude pass (≤ $5 on the pilot; $15–45 full corpus):** per turn, `docs/
  MODELS.md` rules (pinned model, thinking disabled, structured output,
  prompt_version, k = 3 median + spread): sentiment −2…+2 toward the item,
  hostility 0–4 (meanness scale, so the side corpus anchors it), frames
  (multi-label), self-identification from speech only (resident / homeowner /
  renter / org / staff / elected). No names stored. Same plots as the lexicon
  pass, plus lexicon-vs-Claude agreement (scatter, κ on binned scores) — if
  they agree, the free pass carries the exploration.
- Validation when Puck wants it: 150 turns coded blind by both of us.

## 4. Later, if wanted — models (not scheduled)

 — the questions

All Bayesian (brms), episode and body random intercepts, month as a smooth,
family as a fixed effect, weakly informative priors; posterior predictive
checks; report contrasts with 90 % intervals, not p-values.

1. **Role gap.** Is public comment more opposed, more hostile, more
   anecdotal than the dais on the same item? Ordinal model for stance,
   ordinal for hostility, categorical for evidence type: `~ role * family +
   (1 | body) + (1 | episode)`.
2. **Where.** Which bodies have the most opposed / most hostile public
   comment on density? Posterior body intercepts, ranked with uncertainty.
   Same for transit (BART safety vs service frames).
3. **When.** Do frames and stance shift around events: housing-element
   deadlines, SB 79 (2025), builder's-remedy filings, BART fiscal-cliff
   votes? Event-window contrasts, not change-point fishing.
4. **Frames → stance.** Which frames predict opposition? Multinomial /
   multi-label frames as predictors of stance; interaction with `self_id`
   (do homeowners and renters use different frames?).
5. **Density vs transit.** Is opposition to density expressed in transit
   terms (traffic, parking) and support for transit in housing terms
   (station-area, TOD)? Co-occurrence of families inside `about_both`
   episodes; frame overlap.

## 5. Creative extensions (later) (pick after §1–4 exist; each is 1–3 days)

- **Repeat commenters, anonymously.** Voice embeddings (already built for
  host enrolment) cluster public speakers across meetings of the same body
  into anonymous IDs. Questions: what share of public-comment minutes comes
  from the top 10 voices? Is opposition concentrated in few recurring voices?
  Stored as embedding IDs only; never linked to names. Needs DECISIONS sign-off.
- **Template / campaign detection.** Near-duplicate public comments across
  speakers and cities (MinHash over turns) → organised campaigns vs
  spontaneous comment; share over time.
- **Dais responsiveness.** Semantic similarity between public frames and the
  dais discussion that follows in the same episode; do councils echo public
  frames, and which ones?
- **Outcome linkage.** Legistar action minutes give the item's outcome
  (approve / continue / deny). Does public-comment stance or volume predict
  continuance? (Observational; state as association.)
- **Topic model of public comment** (BERTopic on turns) as a check on the
  hand-built frame codebook — what did we miss?
- **Rhetoric features** (cheap, lexical): first-person plural rate, hedging,
  "you" directed at the dais, mentions of specific streets/blocks (a localism
  index), numbers cited per 1,000 words.
- **Time-of-night.** Public-comment hostility vs clock time and item position
  in the agenda (late items, tired rooms).
- **Cross-body diffusion.** First appearance of a phrase ("builder's
  remedy", "Sacramento mandate") by body and month — who leads, who follows.
- **Side-corpus calibration.** The Hasan/Tucker hostility scores as anchors
  for the 0–4 tone scale (a known-hot corpus), reported once, not compared
  substantively.

## 6. Deliverables and order (exploratory only for now)

| Phase | What | Needs | Effort |
|---|---|---|---|
| A0 | §1 rate plots + §2 term-frequency-per-speaker plots on the T1 pilot (46 episodes) | nothing new | 1 day |
| A1 | §3 sentiment: lexicon pass, then Claude pass on the pilot, agreement plot | ≤ $5 | 1 day |
| A2 | rerun A0–A1 on the full corpus after T2 | T2 data; $15–45 if the Claude pass is kept | 1 day + compute |
| later | §4 models, §5 extensions | Puck's call | — |

Outputs: `output/analysis/` parquet + DuckDB, one figure set per section
(`make trends`-style), a methods appendix generated from the pinned prompt
versions and model IDs, and a short written report per phase.

## 7. Open decisions for Puck

1. Anonymous repeat-commenter IDs via voice embeddings — allowed? (ethics)
2. Hostility scale: reuse the meanness 0–4 scale or a meeting-specific one?
3. Which events for §4.3 (list of dates)?
4. Coding depth: public turns only, or dais turns too (doubles scoring cost)?
5. Which two §5 extensions first?

## 8. Stance scope (Puck, 2026-10-16)

**Decided: Tier A only for the current scope.** Stance is measured per
anonymous speaker cluster within an episode (the §3 Claude pass: stance −2…+2
toward the item, hostility 0–4, frames multi-label, sub-topic, self-described
identity from speech only). Consistent with `DECISIONS.md` #15: no names
extracted, stored or indexed. This is estimand 2 of the showcase write-up.

Cost, re-estimated 2026-10-16 on the 409 QA-gated episodes (8,326 turns,
731 k words; median 22 words/turn), Message Batches rate, codebook cached,
~120 output tokens per call:

| turns scored | n | Sonnet 4.6 k=1 | k=3 | Sonnet 5 k=1 | k=3 |
|---|---|---|---|---|---|
| all | 8,326 | $13.3 | $39.9 | $8.9 | $26.6 |
| ≥ 5 words | 6,600 | $10.8 | $32.5 | $7.2 | $21.7 |
| ≥ 15 words (97 % of words) | 4,833 | $8.3 | $24.8 | $5.5 | $16.6 |

Assumptions: 1.3 tokens/word, 150 tokens per-call overhead, 2,000-token
cached codebook. Turns under 5 words ("thank you", "next speaker") carry no
stance and are not worth a call. Needs `CLAUDE_CAP_USD` raised from $10
($7.52 spent on the classifier as of 2026-10-16).

**Documented as future directions, not scheduled, not approved:**

- **Tier B — persistent anonymous speaker IDs.** Voice embeddings
  (`scripts/host_enroll.py` machinery) link clusters across meetings of the
  same body into an anonymous ID; answers the §5 repeat-commenter questions
  and lets stance be tracked per voice over months. A persistent ID is
  re-identifiable in principle, so it needs its own `DECISIONS.md` entry
  (§7 item 1) before anything is stored. Embedding IDs only, never joined to
  a name.
- **Tier C — named people.** Two different cases. *Dais members* are elected
  officials on the public record; a body-level cluster → seat mapping is the
  "explicit ethics decision" #15 allows for, and would need the name-redaction
  step (#15, not built) to exist first so the mapping lives outside the
  corpus. *Public commenters* are private individuals; #15 says never, and
  the proposal's risk table (§6, "Ethics") gives the reason: the public record
  makes the corpus defensible, not the people searchable. Not reopened.
