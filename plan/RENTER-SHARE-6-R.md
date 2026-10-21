# RENTER-SHARE-6-R — the same fits in R (lme4 / glmmTMB), 2026-10-19

Puck asked for R to be installed and the frequentist mixed model redone with
the real packages. `plan/RENTER-SHARE-5-FREQ.md` §9 named the gap: "the GLMM is
a custom quadrature fit, not `glmer`". This closes it. Scripts in
`analysis/r/`, artefacts in `output/renter_share_6/`. Read-only on the DB and
on runs 2–5; no network, no spend, no API, nothing committed. `.env`,
`plan/frame.csv` and `plan/DECISIONS.md` untouched.

**Nothing in RENTER-SHARE-5-FREQ.md is superseded.** Every run-5 number that R
can check, R reproduces. Two things are *added*: profile-likelihood CIs on the
other covariates, which cost `log_density` its 95 % exclusion (§3), and a
properly nested turn-level model, which moves the role estimate rather than
just widening it (§5).

## 1. Install

| component | version | note |
|---|---|---|
| R | 4.6.1 (2026-06-24), aarch64-apple-darwin27.0.0 | `brew install r` |
| lme4 | 2.0-6 | |
| glmmTMB | 1.1.15.2 | TMB 1.9.25, RTMB 2.0 |
| DHARMa | 0.5.0 | |
| performance | 0.19.0 | |
| emmeans | 2.0.4 | |
| sandwich | 3.1.3 | |
| clubSandwich | 0.7.0 | |
| Matrix | 1.7-5 | BLAS openblas 0.3.34 |

CRAN mirror `https://cloud.r-project.org`, all from source: Homebrew's R has no
"well-defined binary type", so `install.packages(type = "both")` refuses and
`type = "source"` is required.

**One install beyond R was needed, and is worth flagging.** `lme4` needs
`nloptr`, which builds its vendored NLopt with CMake; neither CMake, pkg-config
nor a system NLopt existed on this Mac, so the first package run failed and
`lme4`, `glmmTMB` and `DHARMa` were all MISSING. Fixed with
`brew install pkgconf nlopt` — two small build dependencies (581 KB), after
which `nloptr` links the system NLopt and everything builds. That is strictly
more than "install R", so: it is build tooling for the requested packages, it
is the whole of what was added, and it can be removed with
`brew uninstall nlopt pkgconf` if you would rather not carry it.

## 2. The row sets are the run-5 row sets

`analysis/r/export_for_r.py` imports `build()`, `design()` and `turn_level()`
from `analysis/renter_share5_freq.py` and writes the design matrices out, so R
fits the same numbers rather than a re-derivation. The counts reproduce §1 of
run 5 exactly.

| | rows | bodies | trials | NIMBY |
|---|---|---|---|---|
| Primary — NIMBY/(NIMBY+YIMBY), staff dropped | 56 | 24 | 1,300 | 603 |
| Secondary — published denominator, staff kept | 64 | 25 | 2,790 | 619 |
| Turn level — NIMBY turn vs YIMBY turn | 1,892 turns | 26 | 502 episodes / 139 meetings | 856 |

## 3. `lme4::glmer`, binomial logit, body random intercept, nAGQ = 25

Primary rows. Wald CI from the `glmer` SE, profile CI from `profile.merMod`.

| parameter | est | SE | Wald 95 % | profile 95 % |
|---|---|---|---|---|
| (Intercept) | −0.826 | 0.129 | (−1.079, −0.573) | |
| **role** (public−dais) | **+0.732** | 0.127 | **(+0.482, +0.981)** | **(+0.484, +0.982)** |
| renter_share | −0.088 | 0.122 | (−0.327, +0.151) | (−0.328, +0.246) |
| role × renter | −0.159 | 0.132 | (−0.419, +0.100) | (−0.420, +0.099) |
| log_income | +0.118 | 0.124 | (−0.124, +0.361) | (−0.136, +0.369) |
| log_density | +0.309 | 0.143 | **(+0.030, +0.589)** | **(−0.140, +0.590)** |
| built_2010_plus | −0.250 | 0.138 | (−0.521, +0.021) | |
| nhwhite_share | +0.009 | 0.162 | (−0.309, +0.327) | |
| **transit_commute** | **−0.616** | 0.152 | **(−0.915, −0.317)** | **(−0.930, −0.249)** |
| vacancy_rate | +0.010 | 0.130 | (−0.244, +0.264) | |
| log_population | +0.082 | 0.116 | (−0.145, +0.309) | (−0.325, +0.306) |
| route[agenda_index] | +0.035 | 0.289 | (−0.532, +0.602) | |
| route[fullasr] | +0.866 | 0.265 | (+0.347, +1.385) | |
| route[local_asr] | +0.663 | 0.268 | (+0.138, +1.188) | |
| route[swagit_transcript] | −0.730 | 0.351 | (−1.419, −0.042) | |

σ_body = **0.000, singular fit** (`isSingular` TRUE); profile CI for σ_body
(0.000, 0.427). Secondary rows: σ_body = **0.306**, not singular, profile CI
(0.114, 0.542) — against the Bayesian posterior median 0.331.

**The one substantive change this document makes.** On the primary rows
`log_density` clears 95 % on the Wald interval (+0.030, +0.589) and **does not**
on the profile interval (−0.140, +0.590). Run 5 §2 called it "the fourth 95 %
survivor … sign-stable but not robust to the beta-binomial"; it is not a
survivor at all once the interval is profiled rather than Wald. Role and
transit commute survive both ways, on both denominators. Profile intervals are
otherwise close to Wald, as they should be.

LRTs via `anova()`:

| test | primary χ² | p | secondary χ² | p |
|---|---|---|---|---|
| role + role×renter dropped (df 2) | 32.270 | 9.8e-08 | 357.569 | 2.3e-78 |
| role dropped, no interaction either side (df 1) | 30.812 | 2.8e-08 | 356.464 | 1.7e-79 |
| role×renter dropped (df 1) | 1.459 | 0.227 | 1.105 | 0.293 |

The primary column is run 5 §4 to 3 dp (32.27 / 30.81 / 1.46). The secondary
column is 1.2 higher on the two role tests (run 5: 356.41 / 355.30), because
there σ_body > 0 and the reduced-model log-likelihoods come from 24-node
Gauss–Hermite in Python against lme4's 25-point adaptive quadrature. On a
statistic of 357 against 1 df it changes nothing.

## 4. `glmmTMB` beta-binomial

| denominator | fit | role | φ | ρ | σ_body | AIC |
|---|---|---|---|---|---|---|
| primary | beta-binomial, fixed effects | +0.740 | 33.59 | 0.029 | — | 283.63 |
| primary | beta-binomial + (1 \| body) | +0.740 | 33.59 | 0.029 | 0.000 | 285.63 |
| primary | binomial + (1 \| body) | +0.732 | — | — | 0.000 | 290.23 |
| secondary | beta-binomial, fixed effects | +2.058 | 41.73 | 0.023 | — | 330.37 |
| secondary | beta-binomial + (1 \| body) | +2.058 | 41.73 | 0.023 | 0.000 | 332.37 |
| secondary | binomial + (1 \| body) | +2.029 | — | — | 0.305 | 341.74 |

Adding a body random intercept to the beta-binomial buys **nothing**: σ_body
goes to 0 and AIC rises by exactly 2 on both denominators. The row-level
overdispersion the beta-binomial already carries absorbs everything the body
intercept was doing — including, on the secondary rows, the σ_body = 0.305 that
the pure binomial GLMM needs. The beta-binomial beats the binomial GLMM by 6.6
(primary) and 11.4 (secondary) AIC. **The clustering in these rows is row-level
overdispersion, not a body effect.**

## 5. The nested turn-level model (review finding 4)

NIMBY turn vs YIMBY turn among turns with a numeric stance, staff dropped.
Keys: `turn_stance.episode_id` → `episodes.meeting_id` → `turn_stance.body`.
Fixed effects role, renter share (z), role × renter, route.

| random structure | role | SE | p | episode SD | meeting SD | body SD | AIC |
|---|---|---|---|---|---|---|---|
| iid glm (reference) | +0.582 | 0.097 | 1.9e-09 | — | — | — | 2555.11 |
| (1 \| body) | +0.543 | 0.102 | 1.1e-07 | — | — | 0.361 | 2539.72 |
| (1 \| body) + (1 \| meeting) | +0.381 | 0.112 | 0.00066 | — | 0.803 | 0.000 | 2464.65 |
| **(1 \| body) + (1 \| meeting) + (1 \| episode)** | **+0.429** | **0.132** | **0.0011** | **0.905** | **0.709** | **0.000** | **2427.95** |

Variance components of the full model: episode 0.819 (SD 0.905), meeting 0.503
(SD 0.709), body **0.000**. Role profile-likelihood 95 % CI **(+0.161,
+0.700)**, renter share (−∞, +0.155), role × renter (−0.469, +0.071).

The three-level fit is singular in `body` — meeting is nested inside body and
absorbs it entirely — so the profile is taken from the identical non-singular
two-level reparameterisation (`analysis/r/turn_profile.R`; the two agree to
6.4e-06 in every fixed effect and to 4 dp in log-likelihood, −1202.9756).
lme4's profiler also needed `devtol = 0.01`: bobyqa stops ~0.005 deviance short
of the optimum the profiler then finds, and restarting from the fitted values
does not close it. That is a tolerance, not a validity, problem, but the role
CI is a shade wide because of it.

**This is the part run 5 could not do.** Run 5 §5 held the point estimate fixed
at +0.583 and only widened the SE (×2.6 at body level). With the nesting
actually *modelled*, the role estimate itself moves: **+0.582 → +0.429**, a
quarter of it gone, and the SE grows ×1.36 rather than ×2.6. The
cluster-robust sandwich and the random-effect model disagree about how much
precision is lost — the sandwich is more pessimistic about the SE, the mixed
model is more pessimistic about the estimate. Both still exclude zero. Renter
share goes from −0.118 (iid) to −0.089 and nowhere near significance in either.
AIC falls monotonically as nesting is added (2555 → 2540 → 2465 → 2428): the
episode and meeting levels are real and large, and **the body level is not
there at all once meeting is in the model** — which is why σ_body is zero in
the primary row-level fit too.

## 6. DHARMa diagnostics and `performance::icc`

1,000 simulations, `seed = 20261019`.

| fit | dispersion | p | KS (uniformity) | p | outlier p | ICC (adj) |
|---|---|---|---|---|---|---|
| primary glmer | 1.172 | 0.070 | 0.162 | 0.095 | 1.00 | n/a (σ=0) |
| primary beta-binomial | 1.344 | 0.146 | 0.086 | 0.769 | 1.00 | n/a |
| primary binomial + RI | 1.172 | 0.070 | 0.162 | 0.095 | 1.00 | n/a (σ=0) |
| **secondary glmer** | **0.612** | **0.000** | 0.111 | 0.413 | **0.00** | 0.554 |
| secondary beta-binomial | 1.328 | 0.148 | 0.061 | 0.962 | 1.00 | n/a |
| secondary binomial + RI | 1.108 | 0.212 | 0.089 | 0.652 | 1.00 | 0.553 |
| turn (1 \| body) | 0.999 | 0.894 | 0.012 | 0.954 | 1.00 | 0.038 |
| turn body/meeting | 1.003 | 0.718 | 0.019 | 0.477 | 1.00 | n/a |
| turn body/meeting/episode | 1.005 | 0.582 | 0.030 | 0.066 | 1.00 | n/a |

**Verdicts.** The primary glmer is clean — dispersion 1.17 at p = 0.070 and KS
p = 0.095 are both borderline but neither rejects, and the beta-binomial on the
same rows is comfortably clean (p = 0.146 / 0.769). The **secondary glmer
fails**: dispersion 0.612 at p = 0.000 is significant *under*-dispersion, with
the outlier test also at p = 0.000, and the beta-binomial on the same rows is
fine (1.328, p = 0.148). The published-denominator binomial GLMM is the one
fit here that DHARMa rejects, and the beta-binomial is the fix. All three
turn-level models pass everything; the full nested one has KS p = 0.066, which
is the weakest of the three and still not a rejection.

`performance::icc` returns the adjusted ICC only where a variance component is
non-zero: secondary glmer **0.554** (conditional 0.068), turn body-only
**0.038** (conditional 0.036). The ICC of 0.554 on the published rows is large
and is the same body clustering that σ_body = 0.306 reports; it is absent from
the primary rows entirely.

## 7. Leave-one-body-out and grouped 5-fold (primary, glmer)

**LOBO**, 24 refits (`lobo_glmer_primary.csv`). Role **+0.596** (drop San José)
to **+1.019** (drop Santa Clara), full sample +0.732 — stable in sign and
magnitude, no body decides it. Renter share **−0.433** (drop Santa Clara) to
**+0.040** (drop San José), full sample −0.088 — sign-unstable, and the
extreme is Santa Clara again. Finding 6 reproduces in `glmer` as it did in
the GEE.

**Grouped 5-fold by body**, out-of-sample binomial log-loss per trial:

| model set | log-loss | held-out log density |
|---|---|---|
| intercept only | 0.6958 | −904.5 |
| route only | 0.7170 | −932.1 |
| **+ role** | **0.6933** | **−901.3** |
| + role + renter + role×renter | 0.7199 | −935.8 |
| full (8 covariates) | 0.7569 | −984.0 |

This differs from run 5 §6 in one respect and agrees in the rest. Agrees:
adding covariates past role makes held-out prediction monotonically worse
(0.693 → 0.720 → 0.757), and the full eight-covariate model is the worst thing
on the table. Differs: **role + route (0.6933) now beats the intercept
(0.6958)**, where in run 5 every covariate model lost to the intercept. The
margin is 0.0025 log-loss per trial over 1,300 trials — small, and it rests on
a different random fold assignment (same seed, different language's RNG), so I
would not quote it as a reversal. The honest statement is that role is the
only covariate that is not actively harmful out of sample, and route alone is
harmful (0.7170), which run 5's caveat about route being near-nested in body
already predicted.

## 8. Comparison table

Log-odds. Primary = directional denominator, staff dropped. "Python GH-GLMM" is
run 5's custom Gauss–Hermite fit; Bayesian R1 is run 3, on the secondary
denominator only.

### Primary rows

| parameter | Python GH-GLMM | lme4::glmer | glmmTMB beta-binom |
|---|---|---|---|
| role | +0.7317 | +0.7317 | +0.7400 |
| renter_share | −0.0879 | −0.0879 | −0.1215 |
| role × renter | −0.1595 | −0.1595 | −0.0535 |
| transit_commute | −0.6161 | −0.6161 | −0.5111 |
| log_population | +0.0821 | +0.0821 | −0.0395 |
| log_income | +0.1184 | +0.1184 | +0.0755 |
| log_density | +0.3094 | +0.3094 | +0.1935 |

### Secondary rows

| parameter | Python GH-GLMM | lme4::glmer | glmmTMB beta-binom | Bayesian R1 |
|---|---|---|---|---|
| role | +2.0293 | +2.0292 | +2.0577 | +2.1082 |
| renter_share | +0.0288 | +0.0305 | −0.1088 | −0.0193 |
| role × renter | −0.1165 | −0.1165 | −0.0058 | −0.0071 |
| transit_commute | −0.3937 | −0.3940 | −0.4357 | −0.3371 |
| log_population | +0.0240 | +0.0230 | +0.1082 | −0.0738 |
| log_income | +0.0297 | +0.0298 | +0.0403 | −0.0094 |
| log_density | +0.1026 | +0.1017 | +0.1804 | +0.0218 |

**Where they differ by more than 0.05 log-odds, and why.**

*Python custom GLMM vs `lme4::glmer`: nowhere.* Maximum absolute difference
across all fourteen comparisons is **0.0017** log-odds (secondary
`renter_share`), and on the primary rows they agree to 4 dp on every
coefficient. σ_body agrees (0.000 / 0.000, 0.3060 / 0.3053), the LRTs agree to
3 dp, and the role profile CI agrees to 3 dp on both denominators (primary
+0.4837/+0.9822 against +0.484/+0.982). **The run-5 quadrature fit was right.**
The remaining 0.0017 is 24-node Gauss–Hermite against lme4's 25-point adaptive
quadrature, which is numerical noise, not disagreement.

*GLMM vs beta-binomial* — twelve pairs over 0.05, every one of them a covariate
rather than role. The largest are primary `transit_commute` (−0.616 vs −0.511,
0.105), primary `log_density` (+0.309 vs +0.194, 0.116), primary
`log_population` (+0.082 vs −0.040, 0.122) and `role × renter` (−0.159 vs
−0.054, 0.106). The cause is §4: the beta-binomial puts the extra variance at
row level, where it is correlated with the body-level covariates, so those
slopes shrink; role is identified *within* body × route and is untouched
(+0.732 vs +0.740, a gap of 0.008).

*Frequentist vs Bayesian R1* — eight pairs over 0.05, all on the secondary
rows. `log_population` beta-binomial +0.108 vs Bayesian −0.074 (0.182) and
`log_density` +0.180 vs +0.022 (0.159) are the largest; role is +2.029 /
+2.058 against +2.108 (0.079 / 0.051), `transit_commute` −0.394 against −0.337
(0.057). These are the horseshoe doing its job: run 3's regulariser shrinks
weakly-identified covariates toward zero and the unpenalised likelihoods do
not. Role moves least because it is the best-identified parameter in the
design. **No frequentist/Bayesian gap exceeds 0.19, and none of them changes a
sign or a conclusion.**

The one difference that *does* change a conclusion is the one run 5 already
named and this run does not touch: primary +0.73 against secondary +2.03 for
role — the denominator, not the estimator.

## 9. Plain paragraph

Installing R changed none of the answers and sharpened two of them. `lme4` and
`glmmTMB` reproduce the custom quadrature GLMM in run 5 to within two
thousandths of a log-odd on every coefficient, both denominators, including the
σ_body = 0 boundary on the directional rows and the σ_body = 0.31 on the
published ones, so the run-5 fit can be quoted as a `glmer` fit and the caveat
in its §9 can come off. What R adds is a profile-likelihood interval for every
covariate instead of just role, and that costs log density its place among the
95 % survivors: its Wald interval excludes zero, its profile interval does not.
Role and transit commute survive both ways. It also adds the turn-level model
that finding 4 asked for and run 5 could only approximate with a sandwich, and
that model says something the sandwich could not: once episodes are nested in
meetings, the role gap is not merely less precise, it is *smaller* — +0.43
rather than +0.58, with a profile interval of (+0.16, +0.70) that still
excludes zero. Almost all the clustering lives at episode and meeting level
(SDs 0.91 and 0.71); the body level, which is what every row-level fit in runs
2–5 clusters on, has a variance of exactly zero once meeting is in the model.
That is the same fact as σ_body = 0 on the primary rows and as the
beta-binomial beating the binomial GLMM by 6.6 AIC: the dependence in this
corpus is within-meeting, and the body is the wrong grouping to have been
worrying about. DHARMa agrees and adds one failure of its own — the binomial
GLMM on the published denominator is significantly *under*-dispersed
(p < 0.001) while the beta-binomial on the same rows is clean, so where runs
2–3 fit a binomial to those rows they fit the wrong likelihood. None of this
rescues renter share, which is flat, sign-unstable under leave-one-body-out
(−0.43 to +0.04, Santa Clara again), and no better than a coin in the nested
turn model either.

## 10. Files

Scripts:
- `analysis/r/export_for_r.py` — exports the run-5 row sets and design matrices
- `analysis/r/install_pkgs.R` — the CRAN install
- `analysis/r/fit_glmm.R` — glmer, glmmTMB, turn level, DHARMa, LOBO, K-fold
- `analysis/r/turn_profile.R` — the nested-model role profile CI
- `analysis/r/compare.py` — the §8 comparison and the >0.05 gap list

Artefacts under `output/renter_share_6/`:
- `data/rows_primary.csv`, `data/rows_secondary.csv`, `data/turns.csv`,
  `data/design_meta.json` — the exported inputs
- `glmer_primary_wald.csv`, `glmer_primary_profile.csv`,
  `glmer_secondary_wald.csv`, `glmer_secondary_profile.csv`
- `turn_varcomp.csv`, `turn_profile_ci.csv`, `turn_fixed_2level.csv`,
  `turn_profile.json`
- `dharma_diagnostics.csv`, `lobo_glmer_primary.csv`, `kfold_by_body.csv`
- `comparison_r.csv` (91 rows), `comparison_gaps.csv` (52 pairs > 0.05)
- `r_results.json` — everything above in one file
- `sessionInfo.txt`, `fit.log`

## 11b. The first step, same model (added 2026-10-19)

Puck: "lets keep that lme4 model then as the one we use in the dashboard. drop
the others from the final explanation but keep it in the repo." The dashboard
therefore reports the role gap from §5 and nothing else — which needs the step
*before* §5 fitted by the same estimator rather than borrowed from run 3.

New scripts, neither of which touches the ones above:
`analysis/r/export_anyside.py` (every stance-bearing turn, y = took a side at
all, same frame, same exclusions, same keys) and `analysis/r/turn_anyside.R`
(the identical nested specification).

| | rows | took a side | episodes | meetings | bodies |
|---|---|---|---|---|---|
| any-side step | 3,950 turns | 1,892 | 642 | 148 | 26 |

| random structure | role | SE | p | episode SD | meeting SD | body SD |
|---|---|---|---|---|---|---|
| (1 \| body) + (1 \| meeting) + (1 \| episode) | **+2.422** | 0.124 | 5.1e-85 | **0.898** | **0.340** | **0.305** |

Role profile-likelihood 95 % CI **(+2.183, +2.669)**, renter share
(−0.081, +0.413), role × renter (−0.503, +0.006). **This fit is not singular**:
unlike §5, the three-level structure estimates a non-zero body variance here,
so the profile is taken from the three-level fit directly and no
reparameterisation is needed (`devtol` 1e-6, lme4's own). The body interval
still covers zero (0.000, 0.655).

So the two steps, from one estimator: taking a side at all +2.42 log-odds
(odds ratio 11.27), and which side given a side +0.43 (odds ratio 1.54). Both
exclude zero. The first is mostly a speech-act difference and the second is the
one to quote for a difference in position. Artefacts:
`anyside_fixed.csv`, `anyside_profile_ci.csv`, `anyside_varcomp.csv`,
`anyside_profile.json`, `data/turns_anyside.csv`,
`data/turns_anyside_meta.json`.

## 11. Limitations

Unchanged and still binding: ecological inference, `role_audit` is a model
output and not ground truth, no gold set this phase, the κ gate is unpassed,
route is close to a body label. Added here: the nested turn-level profile ran
at `devtol = 0.01` rather than lme4's 1e-9 default, so the role CI (+0.161,
+0.700) is marginally wide; the 5-fold result in §7 uses R's fold assignment
and is not directly comparable to run 5's at the third decimal; and the
turn-level model treats a turn as the unit, which is the right nesting but
still inherits `stance_median` and `role_audit` as measurements rather than
truth.
