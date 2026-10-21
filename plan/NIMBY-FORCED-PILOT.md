# NIMBY-FORCED-PILOT — what happens when `neutral` is removed (2026-10-19)

**EXPLORATORY.** A forced-choice pilot of the cluster-level NIMBY/YIMBY
classifier, run on Puck's hypothesis: *"if you are not for something then you
are against it"* — i.e. `neutral` is a coding artefact, and removing it should
leave the role gap standing. Code `scripts/nimby_forced_pilot.py`; artefacts
`output/nimby_forced_pilot/`; rows in the **new** table `nimby_forced_pilot`.
Cost **$2.10** (ceiling estimate $2.86, budget $5). Nothing existing was
overwritten: `turn_stance`, `stance_calls`, the `nimby_view` CSVs, `.env`,
`plan/frame.csv` and `plan/DECISIONS.md` are untouched, and the existing prompt
`nb-464460010e` is unmodified. Nothing committed.

## 0. Headline

**Forcing a side does not preserve the role gap — it destroys it.**

| quantity | existing instrument | forced instrument |
|---|---|---|
| role gap δ, log-odds of NIMBY, public − dais | **+0.740** (95 % +0.501, +0.984), OR **2.10** — directional clusters, whole frame | **−0.049** (95 % **−0.571, +0.484**), OR 0.95 |
| raw NIMBY share, dais → public | 0.344 → 0.541 | **0.583 → 0.567** (frame-weighted) |
| replicate agreement, k = 3, cluster unit | Fleiss κ **0.957** (3-way) | Fleiss κ **0.968** (2-way) |
| "takes a side at all" role gap | **+2.925** (95 % +2.678, +3.175), OR 18.6 | not defined — everyone takes a side |

The mechanism is exactly the one the hypothesis denies. 69.1 % of
stance-bearing dais clusters are `neutral` against 10.5 % of public ones; the
forced rule sends **65 %** of previously-neutral dais clusters to NIMBY, which
lifts the dais NIMBY share from 0.34 to 0.58 and leaves it level with public
comment. The role gap in the existing instrument is not a gap in *who
objects*; it is a gap in *who speaks procedurally*, and the forced rule
relabels procedure as objection.

The model itself says so. On previously-neutral clusters coded NIMBY the
**median self-reported confidence is 0.05** (dais) and 0.05 (public), against
0.72 where the forced label agrees with an existing directional label. 54 % of
all forced-NIMBY clusters sit at median confidence ≤ 0.1, against 1.5 % of
forced-YIMBY ones. High replicate agreement
(κ 0.968) and near-zero confidence together say the forced rule is
*deterministic*, not *grounded*: three replicates agree because the rule is
mechanical, not because the speech contains a side.

**Read:** the hypothesis is coherent as a *definition* and empty as a
*measurement*. If "not for = against" then the NIMBY share becomes a measure of
how much procedural and staff speech a role produces, and the published role
finding (≈2.1–2.3× odds among clusters that take a side) disappears — not
because it was wrong, but because the forced denominator no longer distinguishes
the thing it was measuring. Keep `neutral`.

## 1. A correction to the brief

The brief named `nb-464460010e` as "the existing instrument" for the cluster
labels. It is not. `nb-464460010e` (`scripts/nimby_pilot.py`) is a **turn-level
4-way** prompt — YIMBY / NIMBY / neither / not_about_housing — run **k = 1 on
300 turns** as a check on the word-frequency lexicon (`plan/NIMBY-PILOT.md`).
The cluster labels carrying the 69 % / 10 % `neutral` split and the role gap are
the **derived three-way view of the S1 stance pass**, `st-7161e26ad2/claude-sonnet-4-6`
(`analysis/stance_s1_role_audit_view.py`): per cluster, the mean of its turns'
`stance_median`, YIMBY ≥ +0.5 / NIMBY ≤ −0.5 / else neutral. That is what this
pilot is compared against throughout. The forced prompt is nonetheless derived
from `nb-464460010e`'s text, as asked, because that is the repo's one explicit
YIMBY/NIMBY codebook.

The two premise figures reproduce exactly in the model shape used here, on the
whole frame (`aux_existing_gaps.json`, $0): taking a side **+2.925**, NIMBY
given a side **+0.740** (OR 2.10). The latter also reproduces
`RENTER-SHARE-3-ROLE.md` §0a's +0.839 when the fit is restricted to this
pilot's 150 directional clusters (**+0.857**, 95 % +0.164, +1.555).

## 2. Unit, frame, sample

Unit = the anonymous **speaker cluster** inside one episode (episode ×
diarized cluster, DECISIONS #15 — never linked across meetings), the unit of
`output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv`. Frame = that
file's **2,813 clusters with a stance** (`label != 'no stance'`): density /
both episodes of the QA-gated redacted corpus, `role_audit` role dais or
public. Frame composition:

| role | NIMBY | YIMBY | neutral | total | neutral % |
|---|---|---|---|---|---|
| dais | 207 | 396 | 1,349 | 1,952 | 69.1 % |
| public | 417 | 354 | 90 | 861 | 10.5 % |

**Sample: 300 clusters, seed 20261019** (`sample.csv`), half of each role's
sample currently `neutral`, the rest in that role's existing NIMBY/YIMBY
proportions, round-robin over bodies inside each stratum:

| stratum | drawn | of |
|---|---|---|
| dais neutral | 100 | 1,349 |
| dais NIMBY | 34 | 207 |
| dais YIMBY | 66 | 396 |
| public neutral | 50 | 90 |
| public NIMBY | 27 | 417 |
| public YIMBY | 23 | 354 |

26 bodies, 101,709 words, routes classifier 219 / local_asr 31 / agenda_index
21 / fullasr 17 / swagit_transcript 12. Every cluster carries an inverse
sampling weight `stratum_N / stratum_n` so the stratified sample can speak for
the frame; **unweighted sample numbers are not frame numbers** and both are
reported.

The cluster's text is its turns (≥ 5 words, the S1 frame) concatenated in
order inside `<turn n="i">` tags, capped at 700 words (`stance.MAX_TURN_WORDS`
applied to the whole cluster, flagged in the prompt); 36 of 300 clusters are
truncated. Unlike the turn-level pass there is no previous/next-turn context —
the cluster *is* the context. The instruction regex found 0 hits and the model
set `contains_instructions` on 0 of 900 calls.

## 3. The forced instrument

`nbf-f4068b4b90/claude-sonnet-4-6`, pinned in
`scripts/nimby_forced_pilot.py` (`PROMPT_VERSION` = `"nbf-" + sha256(SYSTEM)[:10]`,
the same hash scheme as `nb-`; the `nbf-` prefix marks it a different
instrument, not a revision of `nb-464460010e`). Derived from `nb-464460010e`
with four changes:

1. the unit changed from one turn to **everything one speaker said in one
   meeting**;
2. `neither` and `not_about_housing` **removed** — two labels only, stated as
   "there is no third option";
3. "when torn between YIMBY/NIMBY and neither, prefer neither" **replaced** by
   the forced-choice rule, in the brief's words: *a speaker who does not support
   the housing/density item is coded NIMBY; only clear support is YIMBY*, with
   staff presentations, procedural and clarifying questions, motions read aloud,
   balanced remarks, requests for more study and off-item speech named
   explicitly as NIMBY;
4. a **confidence** field, 0–1, defined as how well the speaker's own words
   support the label, "where values near 0 mean the label comes from the
   forced-choice rule rather than from anything the speaker said".

`claude-sonnet-4-6` (the existing instrument's model), Message Batches,
JSON-schema output, `thinking` disabled, no sampling parameters, **k = 3**
replicates. Batch `msgbatch_01S4uvWBWaUQnAWNtJ1ytFDA`, 900/900 succeeded,
900/900 parsed. Rows: table `nimby_forced_pilot` keyed
`(cluster_id, replicate, prompt_version)`, plus `forced_labels.csv` and the
per-cluster roll-up `clusters_forced.csv` (majority of 3; no cluster needed a
tie-break).

## 4. Results

### 4a. Replicate agreement — forcing does not cost reliability

| instrument | items | Fleiss κ | pairwise agreement | unanimous |
|---|---|---|---|---|
| forced, 2-way | 300 | **0.968** | 0.984 | 97.7 % |
| existing, 3-way, per-replicate cluster label | 300 | **0.957** | 0.978 | — |
| existing, 2-way (clusters directional in all 3 replicates) | 149 | 1.000 | 1.000 | — |

The existing instrument's replicate agreement is recomputed here at the cluster
unit — the cluster label built separately from each replicate `k` of
`stance_calls` — so the two κs are comparable rather than one being a
turn-level number. The 2-way row is **conditional on selection** (it keeps only
clusters every replicate called directional) and is quoted only to show that
nothing is lost by dropping `neutral` *for clusters that had a side*; it is not
evidence of a better instrument.

Reliability is therefore the wrong axis on which to prefer either prompt. A
deterministic rule is highly reliable by construction.

### 4b. Previously-neutral clusters: where they go, and how hard

| role | n | → NIMBY | → YIMBY | NIMBY share | median confidence | median conf. NIMBY | median conf. YIMBY | unanimous |
|---|---|---|---|---|---|---|---|---|
| dais | 100 | 65 | 35 | **0.65** | 0.10 | **0.05** | 0.60 | 97 % |
| public | 50 | 31 | 19 | **0.62** | 0.15 | **0.05** | 0.55 | 98 % |

Two things to read here. First, the split is **not** 100 % NIMBY: about a
third of previously-`neutral` clusters are, under a forced choice, read as
*clear support* — mostly staff and dais speech presenting a housing element or
a rezoning as something to get done (see the YIMBY examples in §4e). The
existing instrument's `neutral` is not one thing; roughly a third of it is
low-temperature advocacy that a ±1 threshold on stance did not catch.

Second, the NIMBY two-thirds arrive at confidence **0.05** — the model is
telling us it coded the rule, not the speech. That is the honest version of
"not for = against": the label is a definitional consequence, with no evidence
behind it.

### 4c. Previously-sided clusters: 84 % agreement, asymmetric

n = 150, accuracy **0.840**, Cohen κ **0.677** (dais 0.83, public 0.86).

| existing ↓ / forced → | NIMBY | YIMBY |
|---|---|---|
| NIMBY | 54 | 7 |
| YIMBY | 17 | 72 |

The disagreement runs one way: **17 previously-YIMBY clusters become NIMBY**
against 7 the other way. Median confidence on those 17 flips is **0.07** (on
the 7 reverse flips 0.35), against 0.72 where the two instruments agree — so
the forced rule also drags *mildly* supportive speech across the line whenever
support is not explicit. Forcing a side is not a neutral re-partition even on
clusters that already had one.

### 4d. The role gap under forced labels

`P(NIMBY) ~ role + route stratum + body random intercept`, cluster-level
Bernoulli logit in PyMC (non-centred body intercepts, 4 × 3,000 draws, 2,000
tune, target_accept 0.95). This is the `RENTER-SHARE-3-ROLE` §3 shape — role
fixed effect, route as a fixed-effect stratum with `classifier` the reference,
body random intercept — moved to the cluster unit; the eight ACS covariates are
dropped because at n = 300 clusters in 26 bodies they are not identified, and δ
in run 3 moved only 7–16 % under leave-one-body-out anyway. Weighted fits apply
the inverse sampling weights.

| fit | n | δ (public − dais) | 95 % | 90 % | OR | Pr > 0 | R̂ max |
|---|---|---|---|---|---|---|---|
| **forced, frame-weighted** | 300 | **−0.049** | **−0.571, +0.484** | −0.482, +0.399 | 0.95 | 0.43 | 1.0017 |
| forced, sample-unweighted | 300 | +0.236 | −0.266, +0.750 | −0.185, +0.662 | 1.27 | 0.82 | 1.0000 |
| forced, classifier route only | 219 | −0.109 | −0.734, +0.525 | — | 0.90 | — | 1.0011 |
| existing labels, directional clusters of this sample, weighted | 150 | +0.857 | +0.164, +1.555 | +0.274, +1.442 | 2.36 | 0.99 | 1.0006 |
| existing labels, directional, **whole frame** | 1,374 | **+0.740** | **+0.501, +0.984** | +0.540, +0.944 | 2.10 | 1.00 | 1.0003 |
| existing labels, "takes a side", whole frame | 2,813 | +2.925 | +2.678, +3.175 | +2.715, +3.136 | 18.6 | 1.00 | 1.0007 |

So against the two reference figures: the old **+2.9** for taking a side cannot
exist under a forced choice (everyone takes a side by construction), and the
**+0.74 / OR 2.10** for NIMBY given a side becomes **−0.05 / OR 0.95**, an
interval that contains zero and excludes +0.50 — the bottom of the existing
95 % interval. The frame-weighted forced NIMBY shares are 0.583 dais against
0.567 public: the dais is, under this instrument, marginally *more* NIMBY than
public comment.

The unweighted sample fit (+0.24) is larger than the weighted one only because
the sample over-represents `neutral` in public comment (50 of 90 public
neutrals, against 100 of 1,349 dais neutrals); it is reported for completeness
and is not a frame estimate.

**Caveats carried forward, unchanged by this run:** a cluster is not an
exchangeable trial (REGRESSION-REVIEW finding 4 — ICC 0.275 by episode, 0.156
by meeting; this fit has a body intercept and no episode or meeting level, so
its interval is if anything ~1.3–1.5× too narrow, which only strengthens a null);
`role_audit` is model output, not a gold label (finding 2, ROLE-AUDIT §8); the
stance κ ≥ 0.6 hand-coding gate (SHOWCASE 4a) is still unpassed, so neither
instrument is validated against humans.

### 4e. Ten previously-neutral **dais** clusters under the forced rule

Six lowest-confidence, four highest, redacted excerpts (first ~45 words), from
`report.json`. Votes are the three replicates.

1. `0-Wf1GC8Xnk_e5__SPEAKER_08` — vta, classifier — **NIMBY, conf 0.05**,
   3/3. *"Chair, I had a couple of is regarding 6.1e, the chairperson's report…
   I just want to highlight for the oversight committee chairperson's report, I
   just want to highlight a couple of things from the report."*
   Model's rationale: "Speaker discusses contract management and KPIs; no stance
   on housing or density expressed."
2. `4ZGPf-k5Vqw_e12__SPEAKER_02` — sanjose, classifier — **NIMBY, conf 0.05**,
   3/3, 11 turns. *"okay I think it's a great basis for discussion I um
   appreciate that um counselor [name] I know you had also…"* — rationale:
   "Speaker is a chair managing discussion; no housing stance expressed;
   forced-choice rule applies."
3. `4ZGPf-k5Vqw_e4__SPEAKER_07` — sanjose, classifier — **NIMBY, conf 0.05**,
   3/3. *"the previous direction came through a memo by then council member
   [name] that directed staff to pursue a conservation easement."* — "Staff
   procedural remarks… forced-choice applies."
4. `BUybO1DfUTE_e14__SPEAKER_03` — livermore, classifier — **NIMBY, conf 0.05**,
   3/3. *"the property owner conducted some preliminary analysis of the
   development potential on the site, taking into account the elevations and
   identified the areas that could be developed…"* — "Staff presenting
   procedural facts."
5. `Chj4lSP9je4_e2__SPEAKER_03` — sunnyvale, classifier — **NIMBY, conf 0.05**,
   3/3, 14 turns, 2,086 words. *"to look at what happens if we do not have
   external funding… we have in place two different scenarios that are pretty
   stark."* — "Staff presentation on Caltrain finances; no housing or density
   advocacy; coded NIMBY by forced-choice rule." **A transit budget briefing is
   now a NIMBY.**
6. `GB1Kt-RfyjY_e10__SPEAKER_01` — concord, classifier — **NIMBY, conf 0.05**,
   3/3. *"So I would entertain a motion to approve the consent agenda as
   presented. Okay, we have a motion and a second. Any discussion? All in
   favor…"* — "Purely procedural chair speech." **The chair's gavel is now a
   NIMBY.**
7. `gr-sausalito-1821_a1__SPEAKER_15` — sausalito, agenda_index — **YIMBY,
   conf 0.90**, 3/3, 23 turns. *"I'll provide a brief overview of the next steps
   associated with the implementation of programs 4 and 8…"* — "presents housing
   element implementation, rezoning progress, and unit targets as obligations to
   fulfill affirmatively."
8. `r05uLZd7W2Q_e1__SPEAKER_05` — santarosa, classifier — **YIMBY, conf 0.82**,
   3/3 — "Staff presentation advocates increasing housing density around transit
   hubs and by-right housing production."
9. `r05uLZd7W2Q_e7__SPEAKER_06` — santarosa, classifier — **YIMBY, conf 0.82**,
   3/3 — "Advocates removing parking minimums, pursuing ministerial housing
   ordinance, and more affordable housing volume."
10. `4ZGPf-k5Vqw_e12__SPEAKER_01` — sanjose, classifier — **YIMBY, conf 0.72**,
    3/3 — "Speaker advocates gentle density in single-family districts and
    equitable housing diversity citywide."

Items 1–6 are what "not for = against" does to staff reports and procedural
turns: a KPI report, a chair managing a queue, an easement history, a site
feasibility summary, a Caltrain funding briefing and a consent-agenda gavel all
become opposition to housing, each at confidence 0.05 and each unanimously
across replicates. Items 7–10 are the other, more interesting third: staff and
dais speech the existing threshold called `neutral` which, read whole and forced,
is legible as support — and at high confidence. A forced instrument gets item 7
right and items 1–6 catastrophically wrong, and nothing in the label itself
distinguishes the two cases. **The confidence field does**, which is the one
part of this pilot worth keeping.

## 5. Cost

| | |
|---|---|
| ceiling estimate before submit | $2.864 (budget $5, `--cap 5`) |
| actual, batch rate | **$2.1031** |
| calls | 900 (300 clusters × k = 3), 0 errors, 0 unparseable |
| tokens | 1,194,636 input, 41,488 output, 0 cache reads |
| per cluster, k = 3 | $0.0070 |

**Accounting note:** `pipeline/spend.py` sums `classifier_calls` +
`stance_calls` only, so this $2.10 is **not** in its total — it still reads
"Claude classifier $102.23 of $150.00"; the true figure is **$104.33**. Same
gap as `plan/NIMBY-PILOT.md`'s $0.62. Either `spend.py` grows a third source or
pilots keep being invisible to the guard; flagged, not changed, because the
spend accounting is Puck's.

Scaling, for reference only: the whole frame at k = 3 would be 2,813 × $0.0070 ≈
**$20**; at k = 1, ≈ $7.

## 6. Recommendation

1. **Keep `neutral`.** The hypothesis is testable and it failed: under forced
   choice the role gap is −0.05 (95 % −0.57, +0.48) instead of +0.74, and the
   reason is that two-thirds of dais `neutral` — staff reports, chairing,
   procedure, off-item transit talk — becomes NIMBY at confidence 0.05. A
   measure that counts a consent-agenda gavel as opposition cannot support the
   role claim it is being asked to support.
2. **Do not re-run this instrument at scale.** The $20 full-frame pass would buy
   a column that is a deterministic function of "how procedural is this
   speaker", which the corpus can already measure for $0.
3. **Port the confidence field into the existing three-way prompt.** It is the
   one clear win here: it separates "neutral because balanced" from "neutral
   because procedural" and would let the S1 `neutral` class be audited without a
   forced choice. That is a prompt change to `st-*`, which re-bills the read
   corpus (memory: a prompt change re-buys every call) — so it is a decision for
   Puck, not a side effect of this pilot.
4. **One substantive finding to keep from the forced pass:** about a third of
   previously-`neutral` clusters are read as *clear support* at high confidence
   (median 0.55–0.60) when the whole cluster is read at once. That is a real
   candidate finding about the existing instrument's ±0.5 threshold and its
   turn-at-a-time unit, and it is worth checking against the three-way prompt at
   the cluster unit — a different, cheaper experiment than this one.

## 7. Files

New: `scripts/nimby_forced_pilot.py`, `plan/NIMBY-FORCED-PILOT.md`,
table `nimby_forced_pilot` in `~/.local/share/ytpt/state.sqlite`,
`output/nimby_forced_pilot/{sample.csv, requests.jsonl, batch.json,
forced_labels.csv, clusters_forced.csv, report.json, aux_existing_gaps.json,
classifier_facet.json, pilot.log}`. Modified: nothing. Committed: nothing.
