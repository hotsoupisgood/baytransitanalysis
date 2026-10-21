# STANCE-S1 — stratified stance pass (estimand 2), 2026-10-18

Approved by Puck as "the stance pass, stratified sample, ~10,000 turns, $45
ceiling". This file records the design, the sample, the cost, the fits and
what a further pass would add. Nothing here changes `plan/DECISIONS.md`,
`plan/frame.csv` or the instrument. Companions: `plan/WAVE1-RESULTS.md` §6.4
(the 300-turn pilot), `analysis/models/README.md` §2 (model statement).

## 0. Headline

> **Superseded in scope 2026-10-19 by `plan/STANCE-S2.md`.** The corpus was
> reloaded from `output/whisper` (628 → 761 density/both episodes) and the
> remainder of the grown frame was scored under this same pinned instrument:
> +3,765 turns for $15.33, `turn_stance` now 13,123 rows. The *design,
> instrument, cost model and limitations* below all still stand, but **every
> count, share, interval and star in this file is the 2026-10-18 frame** and
> is smaller than the current one. The YIMBY/NIMBY numbers in §0 and §7b have
> moved in particular: the public base went 457 → 861 clusters, the split
> crossed from 46.0 / 42.9 / 11.2 to 41.1 / 48.4 / 10.4, and the starred
> bodies went from one (Oakland) to four (+ San Jose, Santa Clara, Napa).
> Read `plan/STANCE-S2.md` for the current figures; read this file for how
> they are made.

> **Reported role = `role_audit` (rule ra-1) — corrected 2026-10-18 after role
> audit.** Every public/dais number below is now computed under the audited
> role (`plan/ROLE-AUDIT-2026-10-18.md`), which overrides the diarization-cue
> role with the codebook's SELF_ID — what the speaker says about who they are —
> wherever that is informative. In the cue-labelled public bucket, 20–39 % of
> the adjudicable turns say `elected` or `staff`. The cue run is kept as a
> sensitivity under a `_cue` suffix and reproduces every previously published
> number exactly, so the differences below are the role column and nothing
> else. **One headline statement changes direction, not just magnitude**: the
> stance gap (§7). Read §7b's and ROLE-AUDIT §8's limitations before quoting
> any of it — the audit is a re-partition, not a cleanup.

- The whole frame got scored: **9,358 turns** for **$35.46** (9,058 new turns,
  27,174 batch calls, $0.0039 per turn), under the $45 ceiling, in 26 bodies.
  0 unparseable replies. Role split **1,027 public / 8,331 dais** under
  `role_audit` (1,069 / 8,289 under the cue role).
- **Hostility**: public turns are rated more hostile than dais turns in every
  family and under both role columns, and the audit **strengthens** it — the
  mislabelled dais speech was diluting the public bucket. Latent gap
  (log-odds) on density **+2.15 (90 % 1.85 to 2.46)** under `role_audit`
  against +1.80 (1.49 to 2.09) under the cue role; odds ratio about 8.6
  against 6.0. Model-free, density: **5.7 % of public turns and 0.3 % of dais
  turns score hostility ≥ 2, a gap of +5.3 points (90 % 3.4 to 7.3)** (cue:
  4.3 % / 0.5 %, +3.8 points). The model fits (PPC inside the replicated
  intervals, max R-hat **1.0042** (hostility/db), **1 divergence** in 8,000
  draws (hostility/computed) — *corrected 2026-10-18 after validation*; §0
  previously read "R-hat ≤ 1.0013, 0 divergences" and contradicted §7, which
  has the right figures).
- **Stance**: the proportional-odds "gap" is near zero overall and **should
  not be read as "no difference"** — the PPC shows the public distribution is
  bimodal where the dais sits at neutral, and a single shift cannot represent
  that. On density the latent gap is **−0.06 (90 % −0.30, +0.20)** under
  `role_audit`, against **−0.38 (−0.62, −0.13)** under the cue role: *the
  "public is more opposed on density" finding largely was dais speech sitting
  in the public bucket*, and under the audited role it is indistinguishable
  from zero. The model-free statement, which is the one to quote: among turns
  with a stance the public is **more opposed by 33 points on density (90 % 27
  to 40) and 21 on transit (15 to 28), and more supportive by 31 (25 to 37)
  and 31 (24 to 38)** — both tails *widen* under the audit (cue: +32/+16
  opposed, +23/+29 supportive), which is the same fact as the latent gap
  collapsing: taking hedged member speeches out of the public bucket sharpens
  both tails and leaves even less for one shared shift to represent.
- **Self-consistency**: the three k replicates agree on stance at weighted
  κ 0.98 (exact 0.98); all three identical on 97.6 % of turns (§6). This is
  unaffected by the role column.
- **YIMBY/NIMBY view** (density items, per anonymous speaker cluster, §7b):
  public speakers with a stance split **46.0 % YIMBY / 42.9 % NIMBY / 11.2 %
  neutral on 457 clusters** under `role_audit` (cue: 42.0 / 41.5 / 16.4 on
  414). The YIMBY−NIMBY balance barely moves (+0.5 → +3.1 net) but the neutral
  band shrinks by a third. **Oakland (classifier) keeps its star** at +0.65
  (90 % +0.11, +0.85); **Petaluma (Swagit) is withdrawn** — +0.40 (−0.04,
  +0.81), interval across zero. No new star appears. The dais is 50–90 %
  neutral everywhere, and moves very little under the audit.
- The κ ≥ 0.6 hand-coding gate (`SHOWCASE.md` 4a) is still not passed; this
  is a measurement with the pilot's instrument, not a validation of it
  (§10). Reconciled against the live DB 2026-10-18 20:50 UTC: 28,074
  `stance_calls` rows under the one prompt_version (10 smoke + 890 pilot +
  27,174 S1), 9,358 `turn_stance` rows all k = 3 / n_ok = 3, seven
  `stance_batches` rows all `collected`, nothing in flight.

## 1. Instrument (pinned, unchanged from the pilot)

`prompt_version st-7161e26ad2/claude-sonnet-4-6`: `plan/STANCE-CODEBOOK.md`
below `--- CODEBOOK ---` as a cached system block, JSON-schema output
(stance −2…+2 / not_applicable, hostility 0–4, frames, sub_topic, self_id,
contains_instructions), thinking disabled, no sampling parameters, k = 3
calls per turn, median kept, spread stored; `pipeline/stance.py run --batch`
(Message Batches API), the pilot's code path. The 300 pilot turns are the
pilot's `turn_stance` rows, not re-scored; one instrument, one table.

What the model saw differs from the pilot in one way: the pilot read
transcripts from the `state.sqlite` transcript paths; S1 read the **redacted**
copies (`output/redacted/<episode>.json`, redact-1.1), which is what
`~/.local/share/ytpt/corpus.sqlite` was loaded from (all 628 corpus episodes
point there; checked by `scripts/stance_s1_sample.py`). Redacted turns carry
`[name]` / `[address]` tokens. Turn indices and clusters are identical: all
300 pilot keys were found in the redacted pool with the same cluster and
role.

## 2. Turn unit and frame

Turn = maximal run of consecutive segments by one diarized cluster inside
one episode after `collapse_loops` (`pipeline/stance.py episode_turns`, the
pilot's function). Role per cluster from the corpus `episode_speakers` table,
`roles_source` (db 593 episodes / computed 35) carried as the role-instrument
facet; locate route from `episodes.method` as the stratum. Frame: the 628
QA-gated (OK/WARN) episodes in `corpus.sqlite`, turns ≥ 5 words, role dais or
public. There is no `staff` role in this corpus (roles are dais / public /
unknown; staff speech sits inside `dais`), so the role dimension is
two-level; 472 `unknown`-role turns ≥ 5 words are outside the frame, as in
the pilot. Pool: **9,358 turns** — the approved "~10,000" is the whole
frame.

## 3. Budget, estimate vs actual

The pilot's $2.01 for 300 turns was **already the Batches rate** (890 of 900
calls in one batch, $0.00223 per call). At that rate the frame's 9,058 new
turns cost ~$60 and `stance.py --dry-run` said $57.96, both over $45, so the
first slice was sized to keep the dry-run estimate under the ceiling
(6,400 turns, 6,100 new, estimate $44.61). The first three batches came in
at $24.14 ($0.0040 per turn: the cached codebook hit 98 % of calls against
80 % in the pilot), which left room to score the remaining 2,958 turns for
$11.33 under the same ceiling. Six batches, all `stance-s1` in
`stance_batches.note`; `output/stance_s1/batches.json`.

| | turns | calls | stance.py estimate | actual |
|---|---|---|---|---|
| S1 slice (3 batches) | 6,100 | 18,300 | $44.61 | $24.14 |
| top-up to the frame (3 batches) | 2,958 | 8,874 | $21.55 | $11.33 |
| **total new** | **9,058** | **27,174** | $66.16 | **$35.46** |

Per call: 294 input tokens, 2,556 cache-read, 13 cache-write, 61 output;
**$0.001305 per call, $0.00391 per turn at k = 3**, 0.54 of the estimate
(`stance.estimate_usd` assumes 75 % cache hits; these batches hit 98 %).
Per batch, `stance_batches` (`stance_calls` sums agree to $0.001):
`01WQWD58…` 6,150 calls $8.5152 · `01PZgieK…` 6,150 $7.9004 · `01Fzb4tr…`
6,000 $7.7221 · `017bw4qE…` 3,000 $3.6986 · `01MLKQSH…` 3,000 $3.9031 ·
`01K2hQRB…` 2,874 $3.7255. Claude ledger after the pass: `stance_calls`
$37.48 (pilot $2.01 + S1 $35.46); with `classifier_calls` $25.57 the project
sits at **$63.05 of `CLAUDE_CAP_USD` $80** (the classifier figure moved from
$17.13 while S1 ran: the 120-s re-screen). Seed 20261018. Every stratum of
`s1_strata.csv` landed: 9,358 of 9,358 frame turns scored, no stratum short.

## 4. Design and sample

Strata = body × role × route. Public is the scarce role, so **every public
turn in the frame is sampled** (1,069; 19 of 24 bodies with public comment
have fewer than 100, so this is the "≥ 100 per body" oversample taken to
its limit). The S1 dais budget (6,400 − 1,069 = 5,331) was allocated over
bodies with a floor of 150 per body (or all a smaller body has) and the
remainder proportional to body size, then proportional over routes within
the body; the 300 pilot turns were forced into their cells. The top-up then
scored the unsampled 2,958 dais turns, so the final table is a census and
the stratification only decides which turns were scored first. Artefacts:
`output/stance_s1/s1_turns.csv` (S1 slice, `pilot` flag), `s1_new_turns.csv`,
`s1_rest_turns.csv`, `s1_frame.csv` (whole frame, `sampled` flag),
`s1_strata.csv`, `s1_manifest.json`.

| body | dais S1 / frame | public (all) | pilot | | body | dais S1 / frame | public (all) | pilot |
|---|---|---|---|---|---|---|---|---|
| santaclara | 1025 / 2002 | 127 | 61 | | sanmateo | 153 / 157 | 28 | 0 |
| petaluma | 618 / 1140 | 81 | 38 | | samtrans | 119 / 119 | 13 | 11 |
| sanleandro | 484 / 858 | 228 | 25 | | dalycity | 116 / 116 | 10 | 0 |
| sausalito | 405 / 690 | 56 | 26 | | sanjose | 101 / 101 | 46 | 11 |
| richmond | 363 / 602 | 70 | 21 | | sunnyvale | 92 / 92 | 6 | 10 |
| bart | 289 / 445 | 125 | 17 | | sanrafael | 88 / 88 | 27 | 11 |
| hayward | 285 / 435 | 71 | 20 | | concord | 57 / 57 | 10 | 0 |
| napa | 241 / 342 | 19 | 16 | | oakland | 48 / 48 | 22 | 10 |
| fremont | 213 / 284 | 42 | 16 | | pleasanton | 34 / 34 | 0 | 0 |
| losaltos | 185 / 224 | 25 | 0 | | vallejo | 33 / 33 | 11 | 0 |
| vta | 176 / 205 | 19 | 0 | | santarosa | 17 / 17 | 3 | 7 |
| paloalto | 160 / 171 | 26 | 0 | | healdsburg, cupertino, livermore | 12, 9, 8 (all) | 3, 1, 0 | 0 |

Frame by route (dais / public): classifier 3,541 / 543; agenda_index 2,022 /
131; local_asr 1,712 / 339; swagit_transcript 1,014 / 56. By role instrument:
db 7,843 / 935; computed 446 / 134.

## 5. The pass

27,174 calls, 0 errored, 0 unparseable, every turn k = 3. The k samples
disagree on stance for 2.2 % of dais and 3.8 % of public turns; spread ≥ 2
points on 0.1 % / 0.3 %. `not_applicable` (no stance content): 47.9 % of
dais turns, 23.4 % of public. One turn (Richmond, `gr-richmond-6006_e1`
turn 2, a 549-word staff report on an e-bike share item) was flagged
`contains_instructions` by all three calls; it was coded on content
(stance 0, hostility 0) and is reported here as data, not acted on.
`output/stance_s1/s1_descriptives_by_role.csv`, `_by_body_role.csv`.

## 6. Agreement with the pilot, and self-consistency

**Shared turns.** The 300 pilot turns are reused, not re-scored: every one
of their 900 `stance_calls` rows comes from the pilot batch
(`msgbatch_014Ycbj5…`, 890) or the smoke (10), none from an S1 batch. κ with
the pilot on the shared turns is therefore 1.00 / exact 1.00 by construction,
and is not evidence of anything. Re-scoring them for a real test–retest
figure would cost ~$1.20 and was not spent (no new spend was authorised).

**Self-consistency across the k = 3 replicates** (the test–retest that is
available for free, `output/stance_s1/s1_k_replicate_agreement.json`;
pairwise over replicates 1–2, 1–3, 2–3, quadratic-weighted κ on the five
numeric levels, exact match, NA-vs-numeric agreement):

| turns | n | stance κ_w | stance exact | NA agreement | hostility κ_w | hostility exact | all 3 identical |
|---|---|---|---|---|---|---|---|
| all | 9,358 | 0.980 / 0.982 / 0.979 | 0.984 / 0.985 / 0.984 | 0.992–0.993 | 0.974 / 0.975 / 0.972 | 0.996 | 97.6 % |
| public | 1,069 | 0.989 / 0.990 / 0.986 | 0.973 / 0.973 / 0.968 | 0.994–0.997 | 0.971 / 0.976 / 0.968 | 0.982–0.987 | 96.2 % |
| dais | 8,289 | 0.974 / 0.976 / 0.973 | 0.986 / 0.987 / 0.986 | 0.992–0.993 | 0.972–0.975 | 0.998 | 97.8 % |
| pilot 300 | 300 | 0.985 / 0.993 / 0.993 | 0.977 / 0.989 / 0.989 | 0.983–0.993 | 0.951–0.983 | 0.990–0.997 | 97.0 % |

The replicates are the same model reading the same prompt, so this is
reliability of the instrument with itself (noise floor ≈ 2 %), not validity
(REVIEW-2026-10-18 finding 8). The one cross-instrument check on record is
the NIMBY prompt (`plan/NIMBY-PILOT.md` §3.3): κ 0.63 four-way, same side
88 % of the time when both give one, on 300 S1 turns.

**Drift pilot vs new rows** under the same instrument
(`output/stance_s1/s1_pilot_vs_new.csv`): dais not_applicable share 0.466
pilot vs 0.480 new, mean stance 0.24 vs 0.12, mean hostility 0.036 vs 0.028,
k-disagreement 2.2 % vs 2.2 %; public 0.247 vs 0.233, 0.41 vs 0.16, 0.195 vs
0.202, 5.2 % vs 3.7 %. The pilot drew from 15 bodies and the pool has since
grown to 26, so the stance means are not expected to match; the NA shares,
hostility and k-noise do.

## 7. Model and results

`analysis/stance_s1_models.py` → `analysis/models/role_gap.py` as the pilot
ran it: cumulative logit on `stance_median` (5 levels, NA dropped) and on
`hostility_median` (5 levels, all scored turns); `beta_role` (public − dais)
with family main effects and a role × family interaction; log-words control;
locate route as a fixed-effect stratum; body and episode random intercepts;
priors `analysis/models/README.md` §2. One fit per role instrument
(`roles_source`) per outcome, never pooled across instruments or routes.
4 chains × 2,000 draws after 1,000 tuning, target_accept 0.9 (frame refit at
0.95), seed 20261017. Whole-frame fit: `output/stance_s1/models_frame/`
(tables, figs, summary.json); the S1-slice fit, same code, is in
`output/stance_s1/models/` and agrees within interval widths.

**Convergence (frame, target_accept 0.95).** stance/db n = 4,845, 583
episodes: max R-hat 1.0039, min bulk ESS 916, 0 divergences. hostility/db
n = 8,778, 590 episodes: 1.0042, 965, 0. stance/computed n = 290, 35
episodes: 1.0043, 1,314, 0. hostility/computed n = 580, 35 episodes: 1.0028,
2,859, **1 divergence** in 8,000 draws (10 at target_accept 0.9; the facet
has 35 episodes and 134 public turns, and its body/episode sds are weakly
identified). All estimates below moved by less than 0.02 between the two
settings.

### 7a. The role gap under `role_audit` — the quoted fit

*Corrected 2026-10-18 after role audit.* The fit below in the big table is
the **cue-role** fit and is now the sensitivity. The quoted estimates are the
`role_audit` refit (`analysis/models/role_gap_audit.py`, same model, same
priors, same sweep, the only change being the `role` column; ~145 s for both
outcomes). Both role columns were refit **on the same input** so the pair is
like-for-like; the `_cuebaseline` tables are that baseline, and the older
stored `role_gap_gap_*_all.csv` tables come from an earlier, much smaller fit
and are not comparable to either.

| outcome | β_role cue (90 %) | **β_role `role_audit` (90 %)** |
|---|---|---|
| stance (density) | −0.376 (−0.616, −0.129) | **−0.059 (−0.303, +0.196)** |
| hostility (density) | +1.795 (+1.490, +2.091) | **+2.147 (+1.852, +2.461)** |

Under the cue role the public is measurably more opposed than the dais on the
same item; **under the audited role that gap is indistinguishable from zero**,
i.e. most of estimand 2's stance gap was members of the body sitting in the
public bucket. The hostility gap goes the other way and strengthens: public
comment really is more hostile, and mislabelled dais speech was *diluting* it.
`output/analysis/models/tables/role_gap_{gap,coef}_{stance,hostility}_all_{roleaudit,cuebaseline}.csv`;
input counts in `role_gap_input_counts_roleaudit.csv` (usable stance turns
public 819 → 840, dais 4,316 → 4,295; per body petaluma 59 → 95, bart
95 → 72, santaclara 85 → 103, concord 8 → 23, sanjose 41 → 28).

**Public − dais on the latent (log-odds) scale**, role instrument `db`,
**cue role — the sensitivity fit**, kept for the family breakdown and the PPC:

| outcome | family | n (public) | median | 90 % | 95 % | Pr(>0) |
|---|---|---|---|---|---|---|
| stance | density | 2,530 (388) | −0.36 | −0.62, −0.11 | −0.67, −0.06 | 0.01 |
| stance | transit | 1,386 (246) | +0.48 | 0.20, 0.76 | 0.14, 0.81 | 1.00 |
| stance | both | 929 (78) | +0.30 | −0.20, 0.79 | −0.31, 0.89 | 0.83 |
| stance | all families, turn-weighted | 4,845 (712) | 0.00 | −0.18, 0.20 | −0.22, 0.23 | 0.52 |
| hostility | density | 4,753 (506) | +1.70 | 1.39, 2.02 | 1.34, 2.07 | 1.00 |
| hostility | transit | 2,496 (323) | +1.48 | 1.03, 1.92 | 0.94, 2.01 | 1.00 |
| hostility | both | 1,529 (106) | +1.72 | 1.05, 2.39 | 0.91, 2.52 | 1.00 |
| hostility | all families, turn-weighted | 8,778 (935) | +1.64 | 1.39, 1.90 | 1.34, 1.94 | 1.00 |

Role instrument `computed` (35 classifier episodes): stance density −0.42
(90 % −1.15, 0.30), transit 0.00 (−0.56, 0.55); hostility density +2.62
(1.60, 3.72), transit +2.90 (1.87, 4.07). Same signs, wider.

Hostility odds ratios (db): density 5.5 (90 % 4.0 to 7.5), transit 4.4 (2.8
to 6.8); the "hostile (≥ 2)" share at the mean body/episode/length is 0.3 %
dais vs 1.8 % public on density. Length matters for hostility (`beta_len`
0.59 per log-word, 90 % 0.50 to 0.68) and little for stance (0.11). Route
strata are small and uncertain (stance: swagit +0.60, 0.10 to 1.15, the rest
within ±0.3 of zero; hostility: all within −0.4 to +0.1). Between-episode sd
≈ 1.0 on both outcomes; between-body sd 0.4 to 0.5.

**PPC.** Hostility: every observed level share for both roles lies inside
the replicated 90 % band. Stance: dais is fine (0.733 at 0 vs replicate
0.650–0.682, slightly over) but **public is a clear misfit**: observed shares
−2 0.103 / −1 0.232 / 0 0.228 / +1 0.267 / +2 0.171 against replicates
0.029 / 0.115 / 0.583 / 0.197 / 0.076. Proportional odds forces one shift of
a shared set of cutpoints; the public is spread to both tails. The latent
stance gap is therefore an average of two opposite effects and is not the
number to quote. The model-free contrast (episode-cluster bootstrap, 2,000
reps, seed 20261018), instrument `db`, **under `role_audit` — these are the
quoted shares**, *recomputed and corrected 2026-10-18 after role audit*
(`analysis/stance_s1_tail_shares_roleaudit.py` →
`output/stance_s1/s1_tail_shares_bootstrap_roleaudit.csv`):

| family | opposed share public / dais | diff (90 %) | supportive share public / dais | diff (90 %) | hostile ≥ 2 public / dais | diff (90 %) |
|---|---|---|---|---|---|---|
| density | 0.411 / 0.076 | **+0.33 (0.27, 0.40)** | 0.447 / 0.137 | **+0.31 (0.25, 0.37)** | 0.057 / 0.003 | **+0.053 (0.034, 0.073)** |
| transit | 0.284 / 0.073 | **+0.21 (0.15, 0.28)** | 0.542 / 0.230 | **+0.31 (0.24, 0.38)** | 0.032 / 0.005 | **+0.028 (0.011, 0.047)** |
| both | 0.311 / 0.099 | **+0.21 (0.06, 0.36)** | 0.500 / 0.167 | **+0.33 (0.21, 0.47)** | 0.071 / 0.004 | **+0.068 (0.024, 0.112)** |
| all (turn-weighted) | 0.361 / 0.080 | **+0.28 (0.24, 0.33)** | 0.482 / 0.170 | **+0.31 (0.27, 0.35)** | 0.051 / 0.004 | **+0.047 (0.034, 0.061)** |

Every tail gap is **larger** under the audited role, on both outcomes and in
every family. That is the same fact as the latent stance gap collapsing to
zero in §7a, not a contradiction of it: the clusters the audit takes out of
the public bucket are long, hedged member speeches that sat near neutral, and
the ones it admits from the dais side are short and opinionated. Both public
tails sharpen, and even less is left for one shared shift of the cutpoints to
represent. The PPC misfit above is therefore *worse* under the audit, which
makes §7's advice — quote the tail shares, not the gap — stronger, not weaker.

The cue-role baseline, which reproduces the originally published table exactly
(`s1_tail_shares_bootstrap_cue.csv`; the unsuffixed
`s1_tail_shares_bootstrap.csv` is the same cue run as first written, left in
place and no longer the reported one):

| family | opposed share public / dais | diff (90 %) | supportive share public / dais | diff (90 %) | hostile ≥ 2 public / dais | diff (90 %) |
|---|---|---|---|---|---|---|
| density | 0.405 / 0.085 | +0.32 (0.25, 0.39) | 0.384 / 0.155 | +0.23 (0.17, 0.29) | 0.043 / 0.005 | +0.038 (0.021, 0.056) |
| transit | 0.240 / 0.079 | +0.16 (0.10, 0.22) | 0.520 / 0.229 | +0.29 (0.22, 0.36) | 0.019 / 0.006 | +0.013 (0.000, 0.027) |
| both | 0.282 / 0.105 | +0.18 (0.04, 0.33) | 0.449 / 0.176 | +0.27 (0.17, 0.37) | 0.066 / 0.004 | +0.062 (0.017, 0.107) |

(Opposed/supportive shares are among turns with a numeric stance; the
public also has far fewer `not_applicable` turns, −26 points on density,
−24 on transit.) A category-specific or two-part model (stance taken vs not;
direction given a stance) is the right next model; the ordinal fit stands as
the pre-registered one with its PPC on record.

`output/stance_s1/models_frame/` holds the target_accept 0.95 refit (the
numbers above); `models_frame.log` has its console record. The S1-slice fit
at 0.9 (`models/`) gives the same intervals to two decimals on `db`.

## 7b. YIMBY / NIMBY derived view (EXPLORATORY — stance κ gate not passed)

`analysis/stance_s1_nimby_view.py` → `output/stance_s1/nimby_view/`. No new
spend: it re-reads `turn_stance`. Unit: the **anonymous speaker cluster
inside one episode** (episode × diarized cluster, DECISIONS #15; never
linked across meetings). Frame: S1 turns in density-family and both-family
episodes (6,500 turns; transit episodes out, YIMBY/NIMBY being a housing
construct) → 2,100 clusters. A cluster's density
stance is the mean of its turns' `stance_median` over turns with a numeric
stance; **YIMBY if ≥ +0.5, NIMBY if ≤ −0.5, else neutral** (half a codebook
step, the threshold the NIMBY pilot used on its lexicon score); a cluster
with no numeric turn is "no stance" and leaves the denominator.

*Corrected 2026-10-18 after role audit*: **the role is now `role_audit`**
(rule ra-1, `plan/ROLE-AUDIT-2026-10-18.md`), the cluster roll-up of the
codebook's SELF_ID. `analysis/stance_s1_nimby_view.py` defaults to it and
writes the unsuffixed artefacts and figures;
`analysis/stance_s1_nimby_view.py --role-col cue` reproduces the published cue
run exactly under a `_cue` suffix. Under `role_audit` the 2,100 clusters split
public 495 / dais 1,605 (cue: 465 / 1,635), of which **457 public clusters
have a stance** (cue: 414) and 38 do not (cue: 51); dais 1,140 with a stance,
465 without. **265 clusters change role**, both ways: 250 public → dais at
turn level and 208 dais → public, so the public base *grows*. Public clusters average **1.10 scored turns** (1.23 among the
clusters that have a stance) and **83 % of cluster labels rest on exactly one
scored turn** — *corrected 2026-10-18 after validation*: the 1.43 previously
quoted here is the mean of *all* turns per public cluster (1.42 on the
artefact), not of scored turns, and the caveat is stronger than it read. Shares of public clusters per body, bodies with ≥ 10 public
clusters with a stance, 90 % episode-cluster bootstrap (2,000 reps, seed
20261018), one table and one figure **per locate route, never pooled**;
dais shares in the same table for contrast.

Route-wide public split (clusters with a stance), **under `role_audit`** —
*corrected 2026-10-18 after role audit*: classifier 198 → **45.5 % YIMBY /
42.4 % NIMBY / 12.1 % neutral**; local_asr 185 → **44.9 / 46.5 / 8.6**;
agenda_index 49 → **42.9 / 40.8 / 16.3**; swagit 25 → **64.0 / 24.0 / 12.0**.
**Overall 457 public clusters: 46.0 % YIMBY / 42.9 % NIMBY / 11.2 % neutral.**
The dais is neutral in 42–86 % of clusters in every body.

The cue-role numbers this replaces (reproducible with `--role-col cue`, now in
the `_cue` artefacts): classifier 182 → 41 / 39 / 20; local_asr 174 →
42 / 47 / 11; agenda_index 41 → 34 / 41 / 24; swagit 17 → 71 / 18 / 12;
**overall 414 clusters: 42.0 / 41.5 / 16.4**. The YIMBY−NIMBY balance barely
shifts (+0.5 → +3.1 net), but **the neutral band shrinks by a third**: the
clusters arriving from the dais side are mostly short, opinionated public
turns, while the ones leaving are long, hedged member speeches.

Per body, **`role_audit`**, bodies with ≥ 10 public clusters with a stance.
★ = the 90 % interval on net YIMBY − NIMBY excludes zero.

| route | body | public clusters (ep) | YIMBY (90 %) | NIMBY (90 %) | YIMBY − NIMBY (90 %) | dais Y / N / neutral (n) |
|---|---|---|---|---|---|---|
| classifier | oakland | 17 (5) | 0.82 (0.56–0.92) | 0.18 (0.08–0.44) | **+0.65 (+0.11, +0.85) ★** | 7 clusters, under the table floor |
| classifier | concord | 14 (8) | 0.64 (0.40–0.83) | 0.36 (0.17–0.60) | +0.29 (−0.20, +0.67) | 0.25 / 0.33 / 0.42 (12) |
| classifier | sanmateo | 16 (5) | 0.50 (0.20–0.65) | 0.31 (0.10–0.67) | +0.19 (−0.44, +0.52) | 0.12 / 0.04 / 0.84 (25) |
| classifier | santaclara | 38 (26) | 0.53 (0.38–0.68) | 0.34 (0.18–0.49) | +0.18 (−0.10, +0.49) | 0.23 / 0.16 / 0.60 (225) |
| classifier | sanjose | 23 (5) | 0.48 (0.34–0.73) | 0.43 (0.20–0.61) | +0.04 (−0.24, +0.53) | 0.45 / 0.08 / 0.47 (38) |
| classifier | petaluma | 15 (6) | 0.40 (0.19–0.56) | 0.53 (0.44–0.67) | −0.13 (−0.50, +0.11) | 0.15 / 0.18 / 0.68 (34) |
| classifier | sanrafael | 13 (4) | 0.31 (0.00–0.50) | 0.62 (0.50–0.77) | −0.31 (−0.77, +0.00) | 0.13 / 0.07 / 0.80 (15) |
| classifier | paloalto | 21 (5) | 0.29 (0.00–0.55) | 0.67 (0.33–0.87) | −0.38 (−0.83, +0.17) | 0.10 / 0.05 / 0.86 (21) |
| local_asr | richmond | 19 (11) | 0.53 (0.20–0.73) | 0.32 (0.09–0.64) | +0.21 (−0.43, +0.63) | 0.20 / 0.07 / 0.73 (85) |
| local_asr | sanleandro | 163 (31) | 0.44 (0.36–0.52) | 0.48 (0.39–0.56) | −0.05 (−0.20, +0.12) | 0.16 / 0.11 / 0.73 (132) |
| agenda_index | santaclara | 14 (8) | 0.64 (0.33–0.82) | 0.14 (0.00–0.40) | +0.50 (+0.00, +0.78) | 0.21 / 0.06 / 0.72 (98) |
| agenda_index | sausalito | 24 (6) | 0.38 (0.18–0.65) | 0.46 (0.17–0.62) | −0.08 (−0.43, +0.44) | 0.12 / 0.12 / 0.76 (41) |
| swagit_transcript | petaluma | 25 (12) | 0.64 (0.44–0.86) | 0.24 (0.04–0.48) | +0.40 (−0.04, +0.81) | 0.27 / 0.11 / 0.62 (143) |

**Stars: Oakland survives, Petaluma is withdrawn.** *Corrected 2026-10-18
after role audit.* Oakland/classifier goes +0.61 (+0.25, +0.80) → **+0.65
(+0.11, +0.85)**, still starred though wider. Petaluma/Swagit goes **+0.53
(+0.22, +0.78) ★ → +0.40 (−0.04, +0.81)** and **loses its star**: four
clusters leave the public bucket and thirteen join it (n 17 → 25), and the
single largest "public" cluster in that frame (1,083 words, YIMBY) is an
elected member advocating housing at the SMART station by their own self_id
(ROLE-AUDIT §5 names all five turns). **Petaluma public-YIMBY is not a
finding.** No new star appears: Santa Clara on agenda_index comes closest at
+0.50 (+0.00, +0.78) and fails the strict `lo90 > 0` test. Petaluma's dais
bucket barely moves (net +0.152 → +0.154). Other movements: santaclara
/classifier −0.12 → +0.18 (n 26 → 38), petaluma/classifier 0.00 → −0.13,
paloalto −0.33 → −0.38, sanleandro −0.05 → −0.05 (n 158 → 163, unmoved);
losaltos drops below the 10-cluster floor, concord/classifier and
santaclara/agenda_index rise above it.

**The audit's own limitations apply to every number in this section**
(ROLE-AUDIT §8): SELF_ID is **model output, not a human label**, and Puck has
ruled out a gold set this phase — the audit says two instruments disagree and
which has the better claim, not what the truth is; **44 % of turns are
`self_id = unknown`** and keep the cue role, so the audited public bucket is
still contaminated at an unknown rate and the measured 20–39 % error is a
lower bound on the informative subset only; and the change is a
**re-partition, not a cleanup** — it adds 208 cue-dais clusters as well as
removing 250 cue-public ones, the public base *rises* 414 → 457, and a shift
in the split is not "contamination removed". 21 ties keep the cue role. The
sign-flip caveat below is unchanged and still applies to all of it.

(Episode counts and 95 % intervals in `public_dais_shares_by_body_route.csv`;
the Oakland dais row has fewer than 10 clusters and is read from
`speaker_clusters.csv`.) Read: with 3–18 episodes per body the intervals
cover most of the unit interval; only Oakland and Swagit-route Petaluma
separate from an even split, both toward YIMBY, and Petaluma's two routes
(classifier −0.13, Swagit +0.40 under the audit) are two different sets of
meetings under two locate instruments, which is exactly why routes are not
pooled — *corrected 2026-10-18 after role audit*: only **Oakland** separates
now, Petaluma's Swagit star having been withdrawn. San Leandro
(163 public clusters, the one body with a tight interval) is even, and its
top sub-topic is `affordability_displacement`: the
rent-stabilisation hearings, where a landlord opposing the ordinance is
"opposed to the item" and lands as NIMBY under this mapping while a tenant
supporting it lands as YIMBY. That is the sign-flip caveat of
`NIMBY-PILOT.md` §3.3 and it is not small for that body.

*Corrected 2026-10-18 after validation, re-quantified 2026-10-18 after role
audit*: the exposure is quantified, and it is not confined to that body's own
row. Under `role_audit`, **124 of San Leandro's 163 public clusters (76 %)**
carry `sub_topic = affordability_displacement`, the sub-topic where the mapping
inverts — and San Leandro is **163 of 457 clusters, 36 % of the whole public
base** (cue: 121 of 179 scored turns / 117 of 158 clusters, 38 % of 414).
Dropping it moves the route-wide public split from **46.0 / 42.9 / 11.2 to
47.3 / 39.8 / 12.9** (cue: 42.0 / 41.5 / 16.4 → 42.2 / 38.3 / 19.5): the
YIMBY ≈ NIMBY near-tie is partly a San Leandro artefact under either role
column, and if the mapping is wrong for rent-stabilisation turns the
direction of this view is unresolved rather than merely noisy. The role audit
does not touch this: it is a mapping problem, not a role problem.

Figures, all **under `role_audit`** as of 2026-10-18, per locate route
(classifier, local_asr, agenda_index, swagit_transcript):
`fig_public_yimby_nimby_<route>.png`, `fig_public_yimby_nimby_<route>_stars.png`
(the star marks a body whose net interval excludes zero) and
`fig_public_vs_dais_<route>.png`, each stamped "EXPLORATORY — stance κ gate not
passed" and tagged `[role_audit ra-1]`, with the mapping, the threshold, the
cluster unit and the route in the caption. The cue-role versions of all twelve
carry a `_cue` suffix and are tagged `[cue role]`.

## 8. What a further pass would add

The frame is scored, so "full pass" now means growing the frame. Measured
rate **$0.001305 per call** (Batches, 98 % cache hits): **$0.00131 per turn at
k = 1, $0.00391 at k = 3**. Turn density in the frame: 9,358 turns ≥ 5 words
(dais + public) over 121.6 episode-hours = **77 turns per episode-hour**.

| frame | turns (est.) | k = 1 | k = 3 | needs first |
|---|---|---|---|---|
| current QA-gated corpus (628 episodes) | 9,358 | done | **done ($35.46)** | — |
| + `unknown`-role turns ≥ 5 words | 472 | $0.62 | $1.85 | nothing; outside the estimand |
| + 97 `done` episodes not yet in the corpus (725 QA-gated vs 628 loaded) | ~1,450 | $1.90 | $5.70 | `make qa`, `redact.py run`, corpus reload |
| all `done` + `cut` + `located` episodes today (1,316 episodes, 218.8 episode-h) | ~16,900 | **$22** | **$66** | transcribing 498 located episodes (74 h ≈ $1.60 GPU + downloads under the YouTube day budget) |
| the whole regular-meeting frame, 2,099 meeting-h, under full ASR (`docs/FUTURE-STEPS.md` 1) | ≲ 162,000 | ≲ $211 | ≲ $633 | $105–130 locate pass first; upper bound, whole meetings are less turn-dense than located episodes |

The $100–150 / 30–45k-turn line in `docs/COSTS.md` §C is retired in favour of
this table. A re-score under a revised codebook (after the κ gate) is a new
instrument: the current frame at the measured rate is ~$12 at k = 1, ~$37
at k = 3.

## 10. Limitations

- **The κ ≥ 0.6 gate is not passed** (`SHOWCASE.md` 4a; `REVIEW-2026-10-18`
  finding 8). No human has coded any of these turns. The 0.98 replicate κ is
  the instrument agreeing with itself; the 0.63 κ with the NIMBY prompt is
  Sonnet agreeing with Sonnet under a second codebook. Puck's 2026-10-18
  decision keeps the phase exploratory: every figure here carries the stamp,
  and no number from this file is a confirmatory result.
- **Stance is toward the agenda item.** On items that are restrictions
  (rent control, moratoria, height limits) "opposed" is the pro-housing side
  and the YIMBY/NIMBY mapping in §7b flips sign (`NIMBY-PILOT.md` §3.3: 13 of
  109 sided turns). San Leandro and Richmond carry most of this.
- **Proportional odds misfits the public stance distribution** (§7 PPC): the
  latent stance gap averages two opposite tails; quote the tail shares, not
  the gap.
- **Roles:** no `staff` role exists in the corpus role column; staff reports
  sit inside `dais` and pull it toward neutral. The `computed` role instrument
  (35 episodes) is a separate, wider facet. 472 `unknown`-role turns are
  outside the frame.
- **The reported role is an audit, not a gold label** (*added 2026-10-18 after
  role audit*, `plan/ROLE-AUDIT-2026-10-18.md` §8). Since 2026-10-18 every
  public/dais number here uses `role_audit` (rule ra-1), which prefers the
  codebook's SELF_ID over the diarization-cue role. SELF_ID is model output
  and no human has checked it; **44 % of turns are `self_id = unknown`** and
  keep the cue role, so the audited public bucket remains contaminated at an
  unknown rate. The audit is a **re-partition in both directions** — 250
  cue-public clusters out, 208 cue-dais clusters in, public base 414 → 457 —
  so differences from the cue numbers are not "contamination removed". The
  cluster roll-up is majority rule, so a merged diarizer cluster carrying both
  a member and a resident still gets one label; 21 ties keep the cue role. Do
  not re-run `scripts/role_label.py` to "fix" the cues: the cue labeller has
  no access to self-identification, and the audit is deliberately a second
  instrument in a second column.
- **Frame selection:** transcribed turns only, after the per-body cap and
  whichever GPU batch ran; BART and VTA are thinly transcribed
  (`WAVE1-RESULTS.md` §3). Locate route is a stratum, not a measurement-error
  model (`REVIEW` finding 6).
- **Cluster unit:** diarization clusters, not people; a speaker split across
  clusters counts twice, two speakers merged count once; public clusters
  average 1.4 scored turns. Nothing links a cluster across meetings.
- **Redaction:** the model read `[name]` / `[address]` tokens; the pilot
  read unredacted text. No effect is visible in §6 drift, but it was not
  tested on the same turns.
- Reliability numbers are from one model, one day; a model retirement
  (`docs/MODELS.md` §2) makes the instrument unrepeatable.

## 9. Files

New: `scripts/stance_s1_sample.py`, `analysis/stance_s1_models.py`,
`analysis/stance_s1_nimby_view.py`, `plan/STANCE-S1-RESULTS.md`,
`output/stance_s1/` (s1_turns.csv, s1_new_turns.csv, s1_rest_turns.csv,
s1_frame.csv, s1_strata.csv, s1_manifest.json, batches.json, dryrun.log,
submit_*.log, collect.log, models.log, models_frame.log,
s1_descriptives_by_role.csv, s1_descriptives_by_body_role.csv,
s1_pilot_vs_new.csv, s1_tail_shares_bootstrap.csv,
s1_k_replicate_agreement.json, models/, models_frame/, models_smoke/,
nimby_view/ {speaker_clusters.csv, public_dais_shares_by_body_route.csv,
public_shares_by_route.csv, summary.json, fig_public_yimby_nimby_<route>.png
× 4}, nimby_view.log). Modified: `docs/COSTS.md` (S1 lines from PRELIMINARY
to actuals, refresh SQL kept); `~/.local/share/ytpt/state.sqlite` tables
`stance_calls` (+27,174 rows), `stance_batches` (+6 rows, `note` prefixed
`stance-s1`), `turn_stance` (+9,058 rows under the pinned prompt_version).
Nothing committed; `.env`, `plan/frame.csv`, `plan/DECISIONS.md`,
`plan/STANCE-CODEBOOK.md` untouched.

**Role-audit adoption, 2026-10-18** (`plan/ROLE-AUDIT-2026-10-18.md`,
`plan/VALIDATION-2026-10-18.md` "Role audit applied"). Modified:
`analysis/stance_s1_nimby_view.py` (now defaults to `role_audit`, writes the
unsuffixed artefacts; `--role-col cue` → `_cue`); `plan/STANCE-S1-RESULTS.md`
§0/§7/§7a/§7b/§10. New: `analysis/stance_s1_tail_shares_roleaudit.py`,
`output/stance_s1/s1_tail_shares_bootstrap_{roleaudit,cue}.csv`, and the
`_cue` copies of the twelve nimby_view figures and four tables. Regenerated
under `role_audit` (previous contents were the cue run, preserved as `_cue`):
`output/stance_s1/nimby_view/{speaker_clusters,public_dais_shares_by_body_route,public_shares_by_route}.csv`,
`summary.json`, and the twelve unsuffixed figures. `turn_stance.role`,
`speaker_roles`, `s1_frame.csv`, `s1_tail_shares_bootstrap.csv` and the
`models_frame/` fits are untouched; no new spend, nothing committed in
`~/ytpt`.
