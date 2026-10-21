# NIMBY-PILOT — word-frequency YIMBY/NIMBY classifier, LLM-verified (2026-10-18)

**Exploratory.** A pilot of a cheap lexicon classifier of speaker turns on the
density episodes, checked against a pinned Sonnet pass on 300 turns. Nothing
here changes `plan/DECISIONS.md`, `plan/frame.csv` or the stance instrument;
`turn_stance` was read, not written. Code: `scripts/nimby_pilot.py`;
artefacts: `output/nimby_pilot/`. Cost: **$0.62** (ceiling $3).

## 0. Headline

- The lexicon is **not good enough**: hand lexicon vs LLM κ = **0.175**
  (3-way, n = 300), precision 0.31 YIMBY / 0.30 NIMBY. On turns where both
  give a side, the side agrees 73 % of the time; the failure is that the
  lexicon gives a side to turns the LLM calls neither (48 %) or not about
  housing (10 %). It is a topic detector, not a stance detector.
- The data-driven keyness lexicon is worse (κ = 0.113) and is contaminated by
  the rent-control fights (see §2b).
- The NIMBY prompt agrees with the S1 stance instrument on the same turns at
  κ = **0.631** (4-way, agreement 0.74); the two LLM instruments disagree
  mostly on neither-vs-side, which is a definitional gap (§3.3).
- Recommendation: **LLM for all turns.** The frame of this pilot (3,854 turns
  ≥ 25 words) costs about **$8** at the measured $0.00206 per turn (Batches,
  k = 1); ~$24 at the stance pass's k = 3. The lexicon's one use is as a
  topic screen (precision for "about housing" 0.90).

## 1. Unit and frame

Turn = `pipeline/stance.py episode_turns` (maximal run of one diarized cluster
in one episode of the QA-gated, redacted corpus, the stance pilot's unit).
Frame: episodes with `family` density or both (396 of 628), role dais or
public (there is no staff role; `unknown` outside the frame), **≥ 25 words**.
**3,854 turns** (dais 3,324, public 530) in 23 bodies, four locate routes
(classifier 1,700, agenda_index 839, local_asr 828, swagit_transcript 487).
Per-speaker aggregate (episode × cluster, per DECISIONS #15 — anonymous
clusters, never linked across meetings): 1,842 speakers,
`speakers_scored.csv`.

## 2. Lexicons

Score = (YIMBY hits − NIMBY hits) per 100 words; **YIMBY if ≥ +0.5, NIMBY if
≤ −0.5, else neutral** (one net hit per 200 words). Matching is
`a0_terms.term_pattern` (whole word, case-insensitive, optional 's,
space = hyphen). Versions are sha256 of the sorted lists; both in
`lexicons.json`.

**(a) hand lexicon `hand-b1f2253181`**, 80 YIMBY / 99 NIMBY terms, written
from the ordinary vocabulary of Bay Area housing hearings. YIMBY: housing
supply/shortage/crisis, density, infill, upzon*, transit-oriented, missing
middle, ADU, duplex/fourplex, multifamily, affordable housing, housing
element, RHNA, state law, builder's remedy, by right, streamline, SB 9/35/330,
AB 2011, more/new units, workforce housing, priced out, displacement,
homelessness, parking minimums, climate, sprawl, exclusionary, "support the
project", "urge you to approve", YIMBY. NIMBY: neighborhood character, out of
scale/character, too tall/big/dense, massive, overcrowding, traffic,
congestion, parking, property values, shadows, privacy, noise, crime, fire/
evacuation, water/sewer/school capacity, "not against housing", "support
housing but", wrong location, more study, CEQA/EIR, slow down, rushed, local
control, Sacramento, mandate, quality of life, preserve, single family,
height limit, stories, tower, luxury, developers, gentrification, small town,
historic, views, oppose/deny/vote no/reject, petition, referendum, lawsuit,
moratorium. Labels on the frame: YIMBY 734, NIMBY 507, neutral 2,613.

**(b) keyness lexicon `key-27f20025ae`**, 40 / 40 terms. Unigrams and bigrams
(stopwords out) over-represented in S1 pro turns (`turn_stance` stance ≥ +1,
654 turns in this frame) vs anti (≤ −1, 435), weighted log-odds with an
informative Dirichlet prior (Monroe et al. 2008, α₀ = 500), min count 8,
|z| ≥ 2.5, top 40 per side (`keyness_terms.csv`). The pro side is generic
approval language (*community, affordable, support, appreciate, forward,
happy, love*) and the anti side is **the San Leandro / Richmond rent-control
fights** (*rent control, housing providers, rental, insurance, costs*): S1
stance is toward the item, and a landlord opposing rent control is "anti"
without being NIMBY. Labels: YIMBY 1,784, NIMBY 967, neutral 1,103; agreement
with the hand lexicon 0.39 (κ 0.11).

## 3. LLM verification

**Sample:** 300 turns, 100 per hand-lexicon label, 50 dais + 50 public in
each, round-robin over bodies within a cell (23 bodies, ≤ 19 turns each;
routes classifier 213 / local_asr 38 / agenda 36 / swagit 13), seed
20261018; `sample.csv`.

**Instrument:** `claude-sonnet-4-6`, Message Batches, pinned system prompt
`nb-464460010e` (in the script; labels YIMBY / NIMBY / neither /
not_about_housing + ≤ 20-word rationale + contains_instructions), JSON-schema
output, thinking disabled, no sampling parameters, k = 1, previous/next-turn
context as in the stance pass. Inputs `llm_requests.jsonl`, outputs
`llm_labels.jsonl`, `batch.json` (msgbatch_01WC7inuGbc8YbhGHqxhYM7U, estimate
$0.76, **actual $0.618**, 300/300 parsed). LLM labels: YIMBY 50, NIMBY 59,
neither 145, not_about_housing 46. contains_instructions: 0; the script's own
instruction regex found 0 turns (one early hit was "w*as an ai*d").

### 3.1 Hand lexicon vs LLM (neither + not_about_housing → neutral)

κ = **0.175**, accuracy 0.45.

| lexicon ↓ / LLM → | YIMBY | NIMBY | neutral | n |
|---|---|---|---|---|
| YIMBY | 31 | 13 | 56 | 100 |
| NIMBY | 9 | 30 | 61 | 100 |
| neutral | 10 | 16 | 74 | 100 |

| label | precision | recall |
|---|---|---|
| YIMBY | 0.31 | 0.62 |
| NIMBY | 0.30 | 0.51 |
| neutral | 0.74 | 0.39 |

By role: dais κ 0.11 (YIMBY P/R 0.20/0.67, NIMBY 0.08/0.44), public κ 0.24
(YIMBY 0.42/0.60, NIMBY 0.52/0.52). The lexicon is twice as precise on public
comment, where people actually take sides; dais turns that mention parking,
stories and traffic are usually staff questions and motions.

Keyness lexicon: κ 0.113, accuracy 0.35, precision 0.23 / 0.34 / 0.68
(`agreement.md` has its tables).

### 3.2 What the lexicon gets wrong (hand lexicon, redacted excerpts)

1. **Objections phrased in YIMBY vocabulary** — Richmond public, lexicon
   YIMBY (+0.9), LLM NIMBY: "the Terminal 1 site, which will eventually be
   flooded by bay waters … the seawater … will push the groundwater toward
   the surface, carrying contamination" (climate, housing words; the point is
   don't build there).
2. **Support phrased in NIMBY vocabulary** — San José dais, lexicon NIMBY
   (−1.2), LLM YIMBY: "we've had numerous problems there with the parking
   issues and the blight … we welcome these units … 16 very deeply affordable
   units … I'd like to move approval."
3. **The pro-housing advocate who names the opposition** — Petaluma public,
   lexicon NIMBY (−3.2), LLM YIMBY: "make a dent in decades of redlining by
   growing vibrant housing supply … a neighbor … would fight tooth and nail
   against a three-story building next door" (stories and single family
   count against; housing supply and redlining do not outweigh them).
4. **The "more homes per capita" denier** — San José public, lexicon YIMBY
   (+0.7), LLM NIMBY: "San Jose added 16,000 dwelling units. We have more
   homes now per capita than we did 10 years ago" asking for 62 units not 80.
5. **The reform advocate who lists the limits they want removed** — San Mateo
   public, lexicon NIMBY (−0.9), LLM YIMBY: argues for higher height limits
   and dropping FAR; every named restriction is a NIMBY term.

Pattern: the vocabulary of the fight is shared by both sides; a word list
sees the topic (precision for "about housing" 0.90 on the sample), not the
direction. Negation, "but", and quoting the other side defeat it.

### 3.3 Sanity check: NIMBY prompt vs the S1 stance instrument

All 300 turns have S1 rows. Mapping stance ≥ +1 → YIMBY, ≤ −1 → NIMBY, 0 →
neither, not_applicable → not_about_housing: κ = **0.631**, agreement 0.74.

| NIMBY prompt ↓ / stance → | YIMBY | NIMBY | neither | not_about | n |
|---|---|---|---|---|---|
| YIMBY | 43 | 7 | 0 | 0 | 50 |
| NIMBY | 6 | 53 | 0 | 0 | 59 |
| neither | 24 | 10 | 98 | 13 | 145 |
| not_about | 6 | 2 | 11 | 27 | 46 |

When the NIMBY prompt gives a side, the stance instrument gives the same side
88 % of the time (96 of 109). The disagreement is one-directional: 34 turns
the stance pass scored ±1 the NIMBY prompt called neither — expected, since
its prompt says "when torn, prefer neither" and the stance codebook codes a
mild lean as ±1. The 7 + 6 sign flips are mostly items that are themselves
restrictions (a moratorium, rent control), where "stance toward the item"
and "YIMBY/NIMBY" legitimately differ. Two Sonnet instruments with different
codebooks agree at the level the SHOWCASE gate asks of human coders
(κ ≥ 0.6); this is not a validation against humans.

## 4. Figure

`fig_public_share_<route>.png`, one per locate route, never pooled
(`facets.py`): share of public turns labelled YIMBY vs NIMBY per body with
≥ 20 public turns, hand-lexicon labels **corrected by the LLM-measured
calibration** on the 150 public sample turns (P(LLM label | lexicon label)),
90 % bootstrap over episodes × calibration rows; stamped EXPLORATORY.
Because the calibration is weak (precision ~0.3–0.5), the corrected shares
are pulled toward the calibration's marginals and the bodies differ by less
than their intervals: classifier route Oakland 0.31 Y / 0.24 N, Santa Clara
0.22 / 0.32, San José 0.21 / 0.34, San Mateo 0.22 / 0.36; local_asr San
Leandro 0.20 / 0.30 (183 turns); agenda Sausalito 0.20 / 0.30; Swagit
Petaluma 0.18 / 0.34. Read: public density comment leans NIMBY by about 10
points in most bodies, Oakland the exception, nothing separable. The LLM-only
sample (300 turns over 23 bodies) is too thin for a per-body figure.

## 5. Recommendation

Lexicon: not good enough for stance (κ 0.18); keep it, if at all, as a $0
topic screen or for KWIC reading. The NIMBY prompt on every turn of this
frame: 3,854 turns × $0.00206 = **~$8** (k = 1, Batches; ~$24 at k = 3), or
~$19 for the full ≥ 5-word S1 frame of 9,358 turns. Cheaper still: S1's
`turn_stance` already covers every one of these turns and agrees with the
NIMBY prompt at κ 0.63 on sides; a YIMBY/NIMBY figure can be drawn from S1
stance on density episodes today for $0, with the definitional caveat in
§3.3 (items that are restrictions). If Puck wants the YIMBY/NIMBY framing
specifically, run the $8 pass with the pinned prompt rather than re-deriving
it from stance.

## 6. Files

New: `scripts/nimby_pilot.py`, `plan/NIMBY-PILOT.md`,
`output/nimby_pilot/{lexicons.json, keyness_terms.csv, turns_scored.csv,
speakers_scored.csv, sample.csv, llm_requests.jsonl, llm_labels.jsonl,
batch.json, agreement.json, agreement.md, public_share_by_body_<route>.csv,
fig_public_share_<route>.png (4), pilot.log, *.out, poll.sh}`. Nothing
modified; nothing committed; `.env`, `plan/frame.csv`, `plan/DECISIONS.md`,
`turn_stance` untouched.
