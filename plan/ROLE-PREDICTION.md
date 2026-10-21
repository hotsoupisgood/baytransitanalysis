# ROLE-PREDICTION — can you predict dais vs public from what a speaker says?

**EXPLORATORY — the stance κ ≥ 0.6 hand-coding gate (`plan/SHOWCASE.md` 4a) is
still not passed. Nothing here is a finding about the Bay Area; it is a finding
about the instrument.** Run 2026-10-19 by the supervisor agent on Puck's
question: *what if we reframed the outcome as dais or public?*

Offline from `~/.local/share/ytpt/state.sqlite` (read-only) and
`output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv`. **No YouTube
request, no GPU, no Claude call, no API key, $0.** No DB write. `.env`,
`plan/frame.csv`, `plan/DECISIONS.md` untouched. Nothing committed. No figures.

Companions: `plan/ROLE-AUDIT-2026-10-18.md` (rule ra-1, the audited role
column), `plan/RENTER-SHARE-3-ROLE.md` (the +2.11 role effect this run
re-reads), `plan/STANCE-CODEBOOK.md` (the instrument).

## 0. Plain answer

**Yes, and that is the problem.** Role is recoverable from content at
AUC ≈ 0.78 per turn and ≈ 0.80 per cluster, held out by whole body. But
essentially all of it is a **speech act**, not a position. The single feature
`|stance|` — how far from zero the turn's stance sits, direction thrown away —
scores AUC 0.763 (turn) and 0.799 (cluster) on its own, matching the full
34-feature model. **Signed** stance, direction only, scores **0.519** per turn
and **0.435** per cluster under a linear model: chance. The dais asks
questions, moves procedure and reports; the public takes sides. Who is
speaking is legible from *whether* they have an opinion, not *which* opinion.

That re-reads RENTER-SHARE-3's role effect. Splitting its +2.11 into two nested
questions on the same rows and unit (§5): **"took a side at all"** carries
**+2.92** log-odds (OR 18.6) and **"which side, given a side"** carries only
**+0.74** (OR 2.10). The NIMBY share gap is mostly a speech-act difference
amplified by the T = 0.5 threshold. A real direction gap survives — it is just
about a quarter the size on the log-odds scale, and it was never the +2.1.

A third, useful result: a content model that **never sees `self_id`** and is
**trained on the cue role** predicts the *audited* role better (AUC 0.833) than
the cue role it was trained on (0.800). Within the cue-dais stratum it ranks
ra-1's flips at AUC 0.797, within cue-public at 0.821 (§6). That is a third
instrument, independent of SELF_ID, agreeing with ra-1 — the strongest
corroboration the audit has, and it is not a gold set either.

## 1. Design — two targets, because of circularity

| | target | source | circularity |
|---|---|---|---|
| **A** | cue role | `turn_stance.role`, `scripts/role_label.py` diarization cues | **none** — independent of the stance LLM pass |
| **B** | audited role | `role_audit.role_audit` / `cluster_role_audit`, rule ra-1 | **yes** — ra-1 is built *from* the stance model's `self_id` |

`self_id` is therefore **excluded from the feature set everywhere**, for A as
well as B, so the two targets see an identical design matrix. For B this is
necessary and still not sufficient: the target and every feature come from the
same Sonnet pass on the same text, so B measures *internal consistency of one
instrument*, not predictability. **B's AUCs are not comparable to A's and must
never be quoted as "role is predictable".** They are reported because the gap
B − A ≈ 0.06 is itself the size of that shared-pass coupling.

`rationale` text is never a feature (codebook rule 3: it must not be mined).

Units: **turn** (13,123 coded turns, one prompt version
`st-7161e26ad2/claude-sonnet-4-6`) and **episode × cluster** (2,813 with a
stance, i.e. `label != "no stance"` — DECISIONS #15's analysis unit).

Features (35): stance median, hostility median, `has_numeric_stance`,
log words, 12 frame binaries, 13 sub_topic indicators, family, route. Route is
a fixed stratum, never pooled. At cluster level frames are max over turns,
sub_topic is the share of turns, stance/hostility are medians, words is the sum.

Models: (1) L2 logistic, standardized; (2) hierarchical logistic in PyMC with a
body random intercept, for odds ratios with intervals (1,000 draws + 1,000 tune,
2 chains; the first pass left the 13-level sub_topic one-hot saturated and threw
rhat > 1.01, the reported fit uses `st_other` as reference and is clean);
(3) `HistGradientBoostingClassifier` as a flexible comparator. CV is
`GroupKFold`, 5 folds, **leaving whole bodies out** and separately **whole
episodes out**. Leave-body-out is the headline: it is the only one that answers
"does this transfer to a city the model has not seen".

## 2. Separability — AUC by target, unit and feature set

Leave-body-out / leave-episode-out. Base rate: 12.2 % public turns, 27.8 %
public clusters. `base_log_loss` is the intercept-only loss.

| feature set | turn A (cue) logit | turn A gbm | cluster A logit | cluster A gbm | turn B gbm | cluster B gbm |
|---|---|---|---|---|---|---|
| **context only** (route+family) | 0.516 / 0.563 | 0.515 / 0.579 | 0.394 / 0.562 | 0.425 / 0.565 | 0.524 / 0.570 | 0.439 / 0.543 |
| length only | 0.699 / 0.717 | 0.704 / — | 0.391 / 0.566 | — | — | — |
| hostility only | 0.603 / 0.639 | 0.606 / 0.651 | 0.538 / 0.660 | — | 0.627 / 0.648 | 0.575 / 0.661 |
| frames only | 0.730 / 0.757 | 0.707 / 0.742 | 0.577 / 0.689 | — | 0.778 / 0.791 | 0.662 / 0.724 |
| stance only | 0.632 / 0.662 | **0.762** / 0.779 | 0.531 / 0.655 | — | 0.651 / 0.673 | 0.556 / 0.643 |
| **all** | 0.746 / 0.768 | **0.775** / 0.801 | 0.691 / 0.765 | **0.802** / 0.841 | 0.834 / 0.846 | 0.865 / 0.884 |
| all, no route/family | 0.728 / 0.753 | 0.761 / 0.795 | 0.693 / 0.758 | 0.799 / 0.843 | 0.829 / 0.844 | 0.857 / 0.885 |

Route and family carry **nothing** (0.52 per turn; *below* chance per cluster
across held-out bodies). Every ablation number above is content.

The ceiling is modest and honest: at turn level the full model's leave-body-out
log-loss is 0.307 against a 0.370 intercept, Brier 0.0886, calibration slope
0.81 with intercept −0.31 — i.e. slightly over-confident out of body, and the
top decile averages p = 0.565 against 0.485 observed. Cluster level is worse
calibrated out of body (slope 0.63; top decile p = 0.830 vs 0.660 observed) and
substantially better under leave-episode-out (slope 0.72, AUC 0.841), which is
the usual sign that body is the real source of variation. Full tables:
`cv_performance_all.csv`, `calibration_*.csv`.

## 3. The ablation that matters — direction vs strength

Among turns with a numeric stance, **73 % of dais turns are exactly 0** (4,471
of 6,083) against **21 % of public turns** (260 of 1,253), while the signed
means barely differ (dais +0.060, public +0.035). So "stance" in §2 is
ambiguous between position and opinionatedness. Splitting it:

| feature set | turn A logit | turn A gbm | cluster A logit | cluster A gbm |
|---|---|---|---|---|
| `stance_dir` — signed stance only | **0.519** | 0.758 | **0.435** | 0.793 |
| `stance_strength` — \|stance\| + has_numeric | **0.763** | 0.760 | **0.799** | 0.793 |
| dir + strength | 0.767 | 0.761 | 0.812 | 0.804 |
| `speech_act2` — strength + length + process frame + sub_topic other | **0.778** | 0.768 | 0.798 | 0.795 |
| all, direction replaced by \|stance\| | 0.781 | 0.768 | 0.803 | 0.797 |
| all (35 features, reference) | 0.746 | 0.775 | 0.691 | 0.802 |

Read the first two rows together. A **linear** model given direction alone is at
chance (0.519, 0.435). A **tree** given the same column reaches 0.758 — because
it learns the U-shape, i.e. it recovers `|stance|` from the signed value. Hand
the linear model `|stance|` explicitly and it immediately matches the tree
(0.763). The information was never in the sign. Eleven features chosen as
speech-act proxies (`speech_act2`) beat the full 35-feature linear model
(0.778 vs 0.746), and dropping direction from the full set **does not hurt**
(0.781 vs 0.746 logit; 0.768 vs 0.775 gbm).

## 4. Which features carry it — odds ratios

Hierarchical logistic, body random intercept, per 1 SD, 90 % interval,
`st_other` reference. Target A, 35 features (`odds_ratios2_*.csv`).

**Turn level (n = 13,123, 27 bodies, σ_body = 0.57, 90 % 0.41–0.78):**

| feature | OR | 90 % | Pr > 0 |
|---|---|---|---|
| **\|stance\|** | **2.20** | 2.08 – 2.34 | 1.000 |
| **log words** | **1.29** | 1.19 – 1.40 | 1.000 |
| frame neighborhood_character | 1.13 | 1.08 – 1.18 | 1.000 |
| frame process_notification | 1.12 | 1.06 – 1.18 | 1.000 |
| frame affordability_displacement | 1.12 | 1.05 – 1.18 | 0.999 |
| hostility median | 1.10 | 1.06 – 1.15 | 1.000 |
| **signed stance** | **0.86** | 0.82 – 0.90 | 0.000 |
| sub_topic housing_element | 0.79 | 0.73 – 0.85 | 0.000 |
| route swagit_transcript | 0.70 | 0.61 – 0.79 | 0.000 |
| has_numeric_stance | 0.91 | 0.83 – 1.01 | 0.070 |

**Cluster level (n = 2,813, 26 bodies, σ_body = 0.76, 90 % 0.55–1.02):**

| feature | OR | 90 % | Pr > 0 |
|---|---|---|---|
| **\|stance\|** | **3.22** | 2.87 – 3.65 | 1.000 |
| has_numeric_stance | 1.41 | 1.24 – 1.61 | 1.000 |
| frame neighborhood_character | 1.19 | 1.07 – 1.32 | 0.997 |
| frame affordability_displacement | 1.18 | 1.05 – 1.33 | 0.990 |
| frame traffic_parking | 1.15 | 1.04 – 1.29 | 0.988 |
| frame process_notification | 1.12 | 1.01 – 1.24 | 0.969 |
| **signed stance** | **0.78** | 0.71 – 0.86 | 0.000 |
| **log words** | **0.82** | 0.72 – 0.93 | 0.008 |
| sub_topic housing_element | 0.83 | 0.72 – 0.96 | 0.015 |
| route swagit_transcript | 0.81 | 0.67 – 0.98 | 0.031 |

Permutation importance on the GBM agrees and is blunter: at turn level
`stance_median` drops AUC by **0.117**, `log_words` by 0.062, and **every other
feature by under 0.007**; at cluster level 0.145, 0.036, and nothing else over
0.015 (`perm_importance_*.csv`). Three facts about one turn — how far from zero,
how long, and (weakly) how hostile — are the whole model. The 12 frames and 13
sub_topics are decoration.

Two sign notes worth keeping:
- **Signed stance runs the "wrong" way** (OR 0.86 / 0.78): holding strength
  fixed, the *more supportive* a speaker is, the *more* dais-like. That is the
  dais moving approvals, and it is the one place a genuine position difference
  shows up.
- **log words flips between units** (1.29 per turn, 0.82 per cluster). Public
  comment turns are long (median 141 words vs 33) because they are capped
  monologues; dais *clusters* are long because a member speaks across a whole
  meeting (median 220 vs 206 words over 2 turns vs 1). The hypothesis that
  length carries role is right at turn level and reverses at cluster level once
  `|stance|` is held fixed — so "length" is not a portable role cue.

Procedural frames: `process_notification` is a real but **small** effect
(OR 1.12), and it points the *opposite* way from the hypothesis — the public
raises process complaints more than the dais discusses process. The procedural
character of dais speech shows up as `not_applicable`/zero stance, not as that
frame.

## 5. What this does to RENTER-SHARE-3's +2.1

`analysis/role_prediction_renter3_decomp.py`. Same unit, same rows (2,813
clusters with a stance, 26 bodies), body random intercept, route fixed stratum,
**role term only** — this is a decomposition of the role coefficient, not a
re-fit of run 3's covariate design, so it is not directly comparable to
+2.11 ± covariates.

Raw composition, under ra-1 (run 3's column):

| role | n | NIMBY | YIMBY | neutral |
|---|---|---|---|---|
| dais | 1,952 | 10.6 % | 20.3 % | **69.1 %** |
| public | 861 | 48.4 % | 41.1 % | **10.5 %** |

| outcome | β_role (public vs dais) | 90 % | OR |
|---|---|---|---|
| **took a side** (NIMBY or YIMBY \| has a stance) | **+2.92** | +2.72, +3.14 | **18.6** |
| **which side** (NIMBY \| took a side) | **+0.74** | +0.53, +0.94 | **2.10** |
| NIMBY \| has a stance *(≈ run 3's +2.11)* | +1.98 | +1.81, +2.16 | 7.28 |
| YIMBY \| has a stance *(≈ run 3's +1.03)* | +1.06 | +0.90, +1.22 | 2.89 |

The bottom two rows reproduce run 3 closely enough to trust the top two
(run 3 with covariates: NIMBY +2.11, YIMBY +1.03). **NIMBY and YIMBY both rise
by role**, which cannot be a NIMBY-ness gap — it is the signature of a
threshold (T = 0.5 on mean cluster stance) applied to two populations that
differ in how far from zero they sit. 69 % of dais clusters are neutral against
10 % of public ones; once both have taken a side, the public is 2.1× more
likely to be the NIMBY side, not 7.3×.

Under the cue role the same split is +2.44 / +0.74: the "which side" term is
**identical** (+0.742 under both columns) and only the "took a side" term moves
with the role column. ra-1 sharpens the speech-act contrast and leaves the
political one untouched, which is consistent with ROLE-AUDIT §6 finding the
stance gap collapse under the audit.

**So: if Puck re-frames the outcome as dais-vs-public, the honest headline is
that the dais and the public differ overwhelmingly in speech act and modestly
in position.** The +2.1 should be quoted as two numbers or not at all. Three
quarters of it on the log-odds scale is "public commenters state positions and
councilmembers run meetings", which is not a finding about housing politics.
`plan/RENTER-SHARE-3-ROLE.md` was not edited.

## 6. Where the content model says the cue role is wrong

Out-of-fold cluster-level predictions, target **A** (cue role), leave-body-out,
mean of the logistic and GBM probabilities — so the score never sees `self_id`
and never sees ra-1. `cluster_predictions.csv`.

The model's `p_public` is ordered by ra-1's verdict *inside* each cue stratum,
which it has no access to:

| cue role | ra-1 verdict | n | mean p_public |
|---|---|---|---|
| dais | dais (kept) | 1,870 | 0.189 |
| dais | **public (flipped)** | 161 | **0.419** |
| public | **dais (flipped)** | 82 | **0.210** |
| public | public (kept) | 700 | 0.440 |

As a ranking: within cue-dais clusters, `p_public` separates ra-1's flips at
**AUC 0.797**; within cue-public, **AUC 0.821**. And a cue-trained content model
predicts the **audited** role at **AUC 0.833** — better than the cue role it was
fitted to (0.802). Three instruments, two of them independent of each other,
and the content is more consistent with ra-1 than with the cue labeller.
(`content_model_vs_ra1_auc.csv`)

Agreement as a flip detector, treating ra-1's 243 cluster flips as the
reference (`agreement_with_ra1.csv`):

| confidence threshold | model flips | both | precision vs ra-1 | recall of ra-1 |
|---|---|---|---|---|
| 0.5 | 664 | 139 | 0.209 | 0.572 |
| 0.7 | 309 | 88 | 0.285 | 0.362 |
| 0.8 | 169 | 56 | 0.331 | 0.230 |
| 0.9 | 51 | 25 | **0.490** | 0.103 |

Agreement is real but far from identity, and it is **strongly asymmetric**. At
0.8 the model flags 152 cue-*public* clusters as dais-like (precision 0.322
against ra-1, recall 0.598) but only 17 cue-*dais* clusters as public-like
(precision 0.412, recall **0.043**). The model finds almost none of ra-1's 161
dais → public flips. That is expected and is a limitation of the content model,
not of ra-1: a short opinionated public comment misfiled as dais looks exactly
like the dais clusters it was trained on in every feature except the ones the
model does not have. **The content model can police the public bucket; it
cannot police the dais bucket.**

Ten most confident disagreements (`confident_disagreements.csv`, 40 rows; all
ten are cue-public clusters the content reads as dais):

| episode | cluster | body | route | time | words | turns | cue | ra-1 | p_public | ra-1 agrees |
|---|---|---|---|---|---|---|---|---|---|---|
| `gr-sanleandro-3105_f1` | SPEAKER_03 | sanleandro | fullasr | 01:31–02:37 | 81 | 2 | public | public | 0.018 | no |
| `gr-sanleandro-3105_f1` | SPEAKER_02 | sanleandro | fullasr | 01:37–04:01 | 151 | 4 | public | public | 0.020 | no |
| `gr-sanleandro-3105_f6` | SPEAKER_01 | sanleandro | fullasr | 00:19–02:59 | 316 | 6 | public | **dais** | 0.021 | **yes** |
| `gr-sausalito-1815_a1` | SPEAKER_00 | sausalito | agenda_index | 23:24–24:01 | 107 | 1 | public | **dais** | 0.024 | **yes** |
| `gr-sanleandro-2992_f3` | SPEAKER_02 | sanleandro | fullasr | 05:39–07:42 | 343 | 1 | public | public | 0.043 | no |
| `gr-sanleandro-3047_e6` | SPEAKER_14 | sanleandro | local_asr | 00:56–41:10 | 237 | 5 | public | public | 0.044 | no |
| `gr-sausalito-1862_a1` | SPEAKER_06 | sausalito | agenda_index | 52:50–54:51 | 288 | 1 | public | public | 0.045 | no |
| `gr-santaclara-2582_f2` | SPEAKER_00 | santaclara | fullasr | 05:37–07:59 | 272 | 1 | public | **dais** | 0.046 | **yes** |
| `MvfG35JeCwA_e6` | SPEAKER_02 | vallejo | classifier | 03:41–04:32 | 63 | 3 | public | **dais** | 0.048 | **yes** |
| `gr-sanleandro-3167_e4` | SPEAKER_02 | sanleandro | local_asr | 00:00–02:57 | 327 | 4 | public | **dais** | 0.049 | **yes** |

Five of ten are ra-1 flips. The other five all have stance median 0.0 and
hostility 0.0 and are plausibly genuine public speakers who asked a question or
read a statement without a position — which is the whole point: the model is
detecting a speech act, so a neutral member of the public is indistinguishable
from the dais to it. `gr-sanleandro-3047_e6` SPEAKER_14 spanning 00:56–41:10 is
a diarizer merge and should be read as such, not as one speaker.

## 7. Limitations

- **Not a gold set, and this does not create one.** Both targets are instrument
  output: A from the cue labeller (93.6 % is post-tuning speech-time accuracy on
  10 clips, not held out — ROLE-AUDIT §1), B from SELF_ID. Puck has ruled out a
  gold set this phase. §6 says three instruments agree; it does not say what is
  true.
- **Target B is internally circular** and its AUCs (0.83–0.88) must not be read
  as predictability. The B − A gap of ≈ 0.06 is the size of the shared-pass
  coupling, nothing more.
- **Features are LLM output too.** `|stance|` is the model's own call about
  whether a turn had a position. "The public takes sides" could in part be
  "Sonnet reads short procedural dais speech as `not_applicable` more readily".
  Nothing here distinguishes those, and the 120 s re-screen / v2 codebook
  (CODEBOOK-V2-PILOT) would change every feature.
- **Calibration is poor out of body** (slope 0.63 at cluster level, over-
  confident at the top). Per-cluster probabilities should not be used as
  weights; only the ranking in §6 is used.
- **Class imbalance**: 12.2 % public turns. AUC is robust to it, the
  probabilities are not, and leave-body-out folds vary a lot in positive rate.
- **σ_body = 0.57 (turn) / 0.76 (cluster)** — role base rates genuinely differ by body,
  which is why leave-body-out is 0.03–0.11 below leave-episode-out everywhere.
- **The model cannot audit the dais bucket** (recall 0.043 on ra-1's
  dais → public flips, §6). Any use of it as a screen is one-directional.
- §5 is a decomposition of the role coefficient with **no covariates**, not a
  re-fit of run 3's eight-covariate design. The two should not be differenced.
- The codebook's sign-flip caveat (NIMBY-PILOT 3.3: restriction items read
  "supportive" = NIMBY) applies to every stance number here.
- One prompt version only; nothing pooled across routes or instruments. Counts
  differ from ROLE-AUDIT-2026-10-18 (9,358 turns, 3,239 units) because the
  `role_audit` table has since grown to 13,123 turns / 3,701 units.

## 8. Recommendation

1. If the outcome is re-framed as dais vs public, **report it as two numbers**:
   took-a-side (+2.92) and which-side (+0.74). A single role coefficient on
   NIMBY is three-quarters speech act.
2. **Do not quote the +2.11 as a NIMBY-ness gap** without the §5 split beside
   it. `plan/RENTER-SHARE-3-ROLE.md` §0 is arithmetically right and reads as
   more than it is.
3. §6 is the best available corroboration of ra-1 and is worth one line in
   ROLE-AUDIT: a content model blind to `self_id` prefers the audited role.
   Treat it as a one-directional screen on the **public** bucket only.
4. The 169 clusters at threshold 0.8 are a cheap hand-check list if Puck ever
   reopens a gold set: 113 of them are not ra-1 flips, so they are new
   information, not a re-read of the audit.
5. Do not add role prediction to the pipeline. It is a diagnostic about the
   stance instrument and should be re-run after the v2 re-read, not before.

## Files

New:
- `analysis/role_prediction.py` — data build, ablation CV, permutation
  importance, hierarchical fits, ra-1 comparison
- `analysis/role_prediction_context.py` — route/family floor and content-only set
- `analysis/role_prediction_stance_decomp.py` — direction vs strength, and the
  identifiable hierarchical refit
- `analysis/role_prediction_renter3_decomp.py` — the §5 split of the +2.11
- `plan/ROLE-PREDICTION.md` (this file)
- `output/role_prediction/`:
  - `turns_features.parquet` (13,123 × 35), `clusters_features.parquet` (2,813 × 35)
  - `cv_performance.csv`, `cv_performance_context.csv`,
    `cv_performance_stance_decomp.csv`, `cv_performance_all.csv`
  - `calibration_{turn,cluster}_{A_cue,B_audit}_{logit_l2,gbm}.csv` (8)
  - `perm_importance_turn_A_cue.csv`, `perm_importance_cluster_A_cue.csv`
  - `odds_ratios_{turn,cluster}_{A_cue,B_audit}.csv` + `.json` (first pass,
    saturated sub_topic, kept for the record)
  - `odds_ratios2_{turn,cluster}_{A_cue,B_audit}.csv` + `.json` (**the reported
    fit**)
  - `cluster_predictions.csv`, `confident_disagreements.csv`,
    `agreement_with_ra1.csv`, `content_model_vs_ra1_auc.csv`
  - `renter3_role_decomposition.csv`, `label_by_role_{cue,ra1_audit}.csv`
- `logs/role_pred_{smoke,full,ctx,decomp,r3}.log`

Modified: none. Deleted: none. Committed: nothing. DB writes: none.
