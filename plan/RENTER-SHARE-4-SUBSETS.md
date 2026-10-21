# RENTER-SHARE-4-SUBSETS — all-subsets comparison and permutation nulls

**EXPLORATORY — the stance κ ≥ 0.6 hand-coding gate (`plan/SHOWCASE.md` 4a)
is still not passed.** Run 2026-10-19 by the supervisor agent on Puck's
question: *"can you do the permutation test with these variables. or the
difference test? where you add different combos of covariates together"* —
read as best-subsets selection with model averaging, plus a permutation null.

Offline from `output/renter_share_3/outcomes_role_stacked.csv` (run 3's 64
stacked rows) and `output/renter_share_2/screen.json` (run 2's kept set,
asserted unchanged at load). **No YouTube request, no GPU, no Claude call, no
API key, $0.** No DB write. `.env`, `plan/frame.csv`, `plan/DECISIONS.md`
untouched. Nothing committed. No figures.

Companions: `plan/RENTER-SHARE-2.md` (covariates, VIF screen),
`plan/RENTER-SHARE-3-ROLE.md` (the stacked design and the Bayesian fits this
run re-interrogates), `plan/ROLE-AUDIT-2026-10-18.md` (the role column).

## 0. Plain answer

**Adding the covariates in every possible combination changes the answer in
one place and only one place: `transit_commute`, not renter share, is the
covariate the data actually wants — and renter share's apparent YIMBY effect
is real only in the models that leave the other seven out.**

Of 384 models per outcome, the best NIMBY model keeps **transit commute alone**
(QAICc weight 0.091) and the best YIMBY model keeps **renter share + log
density + transit commute** (0.069). No model is close to decisive: the top ten
hold 37 % (NIMBY) and 27 % (YIMBY) of the Akaike weight, so selection is
genuinely undecided and model averaging is the honest summary.

**Role survives everything.** Model-averaged across all 384 models the role gap
is **+2.110 (unconditional SE 0.172)** on NIMBY and **+1.009 (0.137)** on
YIMBY, i.e. run 3's +2.11 / +1.02 to three and two decimals — the gap does not
move whichever covariates you add or drop. ~~Permuting role by swapping the two
rows of a body × route cell, **0 of 5,000 draws** reached the observed value in
any design: p = 0.0002, the floor at 5,000 permutations. **The role gap is not
a labelling artefact of the stacked design.**~~ (It can still be an artefact of
the role *audit* — see §5.1, unchanged and still the binding limitation.)

> **Corrected 2026-10-19** by `plan/REGRESSION-REVIEW-2026-10-19.md`, findings
> 2 and 10. Two things.
>
> **The magnitude is on the wrong denominator.** +2.110 is NIMBY out of *all
> clusters with a stance*, and ~60 % of that denominator is the `neutral`
> class, which is a dais phenomenon (69.1 % of dais clusters against 10.5 % of
> public), plus ~20 % staff clusters that are 94 % `neutral`. **On the
> directional denominator — NIMBY out of NIMBY + YIMBY — the gap is +0.84
> log-odds, odds ratio 2.3; with staff excluded +0.78, odds ratio 2.2**
> (`plan/RENTER-SHARE-3-ROLE.md` §0a). Everything this run says about role
> being *stable across model space* is unaffected — it is the same number in
> all 384 models — but the number itself is ~2.2× in odds, not ~8×. The
> all-subsets sweep was never re-run on the directional denominator; the
> stability claim is therefore inherited, not re-verified.
>
> **The within-cell role permutation is withdrawn as a test of δ.** `N_i`,
> `y_i` and the row-level spread stay attached to the row when the label is
> swapped, so the test asks whether the *label* lands on the larger /
> more-NIMBY row of a cell, not whether δ is zero net of the other terms; with
> 29 swappable cells and a ±2.1 log-odds separation, the permutation
> distribution is near-degenerate and p = 0.0002 is the 5,000-draw floor rather
> than a measurement. **An across-body role permutation does not exist here**:
> every fitted body has the same role composition, so permuting role across
> bodies is the identity. **δ's evidence is the model (run 3 §3a-full) and the
> by-body cluster-robust SE (run 3 §4), and nothing else.** No permutation
> result for role should be quoted.

**Renter share stays null on NIMBY** (inclusion 0.32, model-averaged −0.023,
permutation p 0.68–0.96) **and is at best marginal on YIMBY** — and
*marginal* got weaker on 2026-10-19: against a valid partial-coefficient null
(Freedman–Lane, or permuting the **whole** body covariate vector rather than
`renter_share` alone) the public YIMBY slope is p = 0.046 / 0.095 in the
two-covariate design and p = 0.51 / 0.35 with the other six covariates in, and
it loses 72 % of its magnitude when Santa Clara alone is dropped
(`plan/RENTER-SHARE-2.md` §3a-lobo, §3d-perm). Its YIMBY
inclusion probability is 0.71, its coefficient +0.236 (SE 0.082) in the best
model but only +0.094 in the full-covariate model — so the number depends on
which covariates it is asked to share credit with. Against a **selection-aware**
null (all-subsets re-run inside every permutation, §4b) the best-model
coefficient gives p = **0.027** and the model-averaged coefficient p = **0.080**,
uncorrected for 2 outcomes × 8 covariates. **The role × renter interaction is
dead in every form**: inclusion 0.076 (NIMBY) and 0.173 (YIMBY), permutation p
0.40–0.69. Nothing here reopens `plan/DECISIONS.md` #13.

One methodological warning worth more than the coefficients: **an Akaike
inclusion probability of 0.7 means nothing at this n.** Under the permutation
null, with renter share deliberately scrambled, the median inclusion
probability is **0.38** and the 97.5th percentile is **0.995** (§4b). Inclusion
probabilities on 25 bodies are not evidence; only the permutation p is.

## 1. Method

`analysis/renter_share4_subsets.py` → `output/renter_share_4/`.

Unit, outcomes and floor are run 3's, unchanged: 64 body × route × role rows,
25 bodies, 35 cells, 2,790 clusters with a stance, floor 5 per row. Covariates
are run 2's eight kept ones, standardised over the same 64 rows (renter sd
0.0886, income sd 0.281); the loader asserts `screen.json["kept"]` still equals
that list, so this run cannot drift from run 2.

**Always in every model:** intercept, `role` (dais 0 / public 1), and the four
route dummies (reference `classifier`). **Route is a stratum in all 384 models
and is never pooled.**

**Enumerated:** every subset of the 8 covariates (2^8 = 256), times the
`role × renter_share` interaction when `renter_share` is in the subset
(+2^7 = 128) → **384 models per outcome**, all converged.

**Fit and dispersion.** Binomial logistic IRLS (the same routine as run 3's
§4 sanity check; statsmodels is not in the venv). Overdispersion is handled
quasi-binomially: ĉ = Pearson χ² / df from the **global** model (all 8 +
interaction), held fixed across the model set per Burnham & Anderson —
**ĉ = 2.580 (NIMBY), 1.989 (YIMBY)**. Then with K = n_params + 1 (the
dispersion) and n = 64,

    QAICc = -2logL/ĉ + 2K + 2K(K+1)/(n-K-1)
    QBIC  = -2logL/ĉ + K ln n

Akaike weights come from QAICc; the **Bayesian version is BMA by the BIC
approximation**, weights from QBIC (no spike-and-slab was run — the BIC
approximation was budgeted as sufficient, and §4b's 1.5 M refits spent the
compute instead). Plain AICc/BIC (ĉ = 1) are in `subsets_top.csv` for
reference; they are not used, because at ĉ = 2.58 they overstate every
covariate. Reported SEs are quasi-binomial (model SE × √ĉ); by-body
cluster-robust SEs are carried for the permuted terms.

**Model averaging** uses zero substitution (full-model averaging: a covariate
absent from a model contributes 0), with Buckland unconditional SEs
`Σ wᵢ √(varᵢ + (βᵢ − β̄)²)`. Conditional averages (over models containing the
term) are also tabulated.

## 2. Top ten models by QAICc

β (SE) are quasi-binomial. `cum w` is the running Akaike weight. `—` = term
not in the model. Covariate names short: renter, income, density, built10,
nhwhite, transit, vacancy, logpop.

### 2a. NIMBY (ĉ = 2.580)

| # | covariates kept | p | QAICc | ΔQAICc | w | cum w | role | renter | role×renter |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **transit** | 7 | 147.49 | 0.000 | **0.091** | 0.091 | +2.133 (0.169) | — | — |
| 2 | transit+vacancy | 8 | 148.68 | 1.186 | 0.050 | 0.141 | +2.126 (0.169) | — | — |
| 3 | density+transit | 8 | 149.14 | 1.649 | 0.040 | 0.180 | +2.138 (0.169) | — | — |
| 4 | transit+logpop | 8 | 149.22 | 1.728 | 0.038 | 0.219 | +2.110 (0.170) | — | — |
| 5 | built10+transit | 8 | 149.65 | 2.155 | 0.031 | 0.249 | +2.113 (0.170) | — | — |
| 6 | renter+transit | 8 | 149.79 | 2.293 | 0.029 | 0.278 | +2.125 (0.169) | −0.057 (0.088) | — |
| 7 | density+built10+transit | 9 | 149.89 | 2.398 | 0.027 | 0.306 | +2.095 (0.171) | — | — |
| 8 | nhwhite+transit | 8 | 150.09 | 2.599 | 0.025 | 0.330 | +2.135 (0.169) | — | — |
| 9 | income+transit | 8 | 150.19 | 2.697 | 0.024 | 0.354 | +2.131 (0.169) | — | — |
| 10 | renter+density+transit | 9 | 150.52 | 3.027 | 0.020 | **0.374** | +2.125 (0.169) | −0.116 (0.095) | — |

`transit` is in all ten. The interaction is in none. Renter share appears twice
with a negative sign and an interval through zero.

### 2b. YIMBY (ĉ = 1.989)

| # | covariates kept | p | QAICc | ΔQAICc | w | cum w | role | renter | role×renter |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **renter+density+transit** | 9 | 188.83 | 0.000 | **0.069** | 0.069 | +1.036 (0.132) | **+0.236 (0.082)** | — |
| 2 | renter+density+transit+logpop | 10 | 190.11 | 1.282 | 0.037 | 0.106 | +1.011 (0.133) | +0.213 (0.085) | — |
| 3 | renter+density+built10+transit | 10 | 190.49 | 1.667 | 0.030 | 0.136 | +1.060 (0.134) | +0.182 (0.092) | — |
| 4 | renter+density+transit+**role:renter** | 10 | 191.13 | 2.299 | 0.022 | 0.158 | +1.015 (0.134) | +0.202 (0.092) | +0.117 (0.148) |
| 5 | renter+income+density+transit | 10 | 191.34 | 2.511 | 0.020 | 0.178 | +1.029 (0.132) | +0.238 (0.083) | — |
| 6 | renter+density+transit+vacancy | 10 | 191.35 | 2.524 | 0.020 | 0.197 | +1.033 (0.132) | +0.249 (0.085) | — |
| 7 | income+density+built10+transit | 10 | 191.57 | 2.746 | 0.018 | 0.215 | +1.053 (0.133) | — | — |
| 8 | renter+nhwhite+transit+logpop | 10 | 191.61 | 2.785 | 0.017 | 0.232 | +0.954 (0.133) | +0.184 (0.081) | — |
| 9 | density+built10+transit | 9 | 191.73 | 2.901 | 0.016 | 0.249 | +1.055 (0.133) | — | — |
| 10 | renter+density+nhwhite+transit | 10 | 191.75 | 2.922 | 0.016 | **0.265** | +1.036 (0.132) | +0.234 (0.085) | — |

The highest-weighted model containing the interaction is 4th, Δ = 2.3, and its
interaction is +0.117 (0.148). The role coefficient sits in +0.95…+1.06 across
all ten — a range of 0.11 log-odds against a point estimate of 1.0.

## 3. Inclusion probabilities and model-averaged coefficients

Akaike = Σ w_QAICc over models containing the term; BMA = Σ w_QBIC. `β̄ full` is
zero-substituted with the Buckland unconditional SE; `β̄ cond` averages only
over models containing the term. Role is in all 384 models by construction.

### 3a. NIMBY

| term | Akaike incl | BMA(BIC) incl | β̄ full | uncond SE | β̄ cond |
|---|---|---|---|---|---|
| **transit_commute** | **0.900** | **0.835** | **−0.344** | 0.181 | −0.382 |
| log_density | 0.331 | 0.188 | +0.054 | 0.097 | +0.162 |
| renter_share | 0.316 | 0.176 | −0.023 | 0.054 | −0.073 |
| log_population | 0.311 | 0.189 | +0.038 | 0.068 | +0.122 |
| built_2010_plus | 0.291 | 0.169 | −0.039 | 0.087 | −0.135 |
| vacancy_rate | 0.283 | 0.181 | −0.034 | 0.069 | −0.119 |
| nhwhite_share | 0.228 | 0.133 | +0.008 | 0.048 | +0.034 |
| log_income | 0.205 | 0.119 | +0.004 | 0.031 | +0.022 |
| **role:renter_share** | **0.076** | 0.027 | −0.011 | 0.027 | −0.143 |
| **role** (always in) | 1.000 | 1.000 | **+2.110** | **0.172** | +2.110 |

### 3b. YIMBY

| term | Akaike incl | BMA(BIC) incl | β̄ full | uncond SE | β̄ cond |
|---|---|---|---|---|---|
| **transit_commute** | **0.802** | 0.612 | **+0.235** | 0.161 | +0.293 |
| **renter_share** | **0.714** | 0.600 | **+0.128** | 0.115 | +0.179 |
| log_density | 0.587 | 0.418 | −0.134 | 0.145 | −0.228 |
| log_population | 0.465 | 0.350 | +0.070 | 0.094 | +0.151 |
| built_2010_plus | 0.402 | 0.254 | +0.079 | 0.126 | +0.197 |
| log_income | 0.344 | 0.255 | −0.047 | 0.080 | −0.137 |
| nhwhite_share | 0.300 | 0.174 | +0.046 | 0.091 | +0.153 |
| vacancy_rate | 0.256 | 0.163 | +0.015 | 0.046 | +0.058 |
| **role:renter_share** | **0.173** | 0.097 | +0.022 | 0.050 | +0.128 |
| **role** (always in) | 1.000 | 1.000 | **+1.009** | **0.137** | +1.009 |

BMA-by-BIC orders the terms identically and shrinks every inclusion
probability (BIC penalises harder at n = 64); it never promotes a term the
Akaike weights demote, so the two agree on every qualitative call. Only
`transit_commute` exceeds 0.8 under both on both outcomes; only renter share
(YIMBY) also exceeds 0.6. **β̄ / uncond SE is below 2 for every covariate on
both outcomes** — the largest is transit on NIMBY at 1.90 — while role is at
12.3 and 7.4.

## 4. Permutation tests

### 4a. The pre-specified nulls, 5,000 permutations each

> **Corrected 2026-10-19** (review finding 10). **Neither permutation scheme in
> this subsection is a valid null for the quantity it is applied to.**
>
> 1. `renter_share` (and later `transit_commute`) is permuted **alone** while
>    seven collinear covariates stay fixed. For a *partial* coefficient that
>    breaks the joint distribution of the design, not just the
>    covariate-to-outcome link, so the null distribution is the wrong one —
>    `renter_share ~ multiunit_5plus` is r = +0.900 and the eight covariates
>    are far from independent. Two valid replacements are run and tabulated at
>    `plan/RENTER-SHARE-2.md` §3d-perm: **Freedman–Lane** (residualise on the
>    design without the target term, resample the weighted residuals in body
>    blocks, refit) and **whole-vector body permutation** (permute the entire
>    eight-covariate body vector together, preserving collinearity exactly).
>    They do not overturn the conclusion — renter share has one p below 0.05
>    in twelve rows and `transit_commute` has eight of eight — but the rows
>    below should be read as the *old* nulls, not as the current evidence.
> 2. The **role** rows are withdrawn entirely as a test of δ, for the reason
>    given in §0. The `p = 0.0002` entries below are the 5,000-draw floor of a
>    near-degenerate permutation distribution, not a measurement, and no valid
>    across-body role permutation exists in this design.
>
> The rows are kept unchanged below for the record.

Two designs: **FULL** = role + all 8 covariates + role×renter + routes;
**MIN** = role + renter share + role×renter + routes. Renter share is permuted
**across bodies**, every row of a body (both roles, all its route cells) moving
together, so within-body structure and all other covariates are preserved.
Role is permuted by **swapping the role labels of the two rows of a
body × route cell**, independently with probability ½ per cell — the 29 cells
that carry both roles, which are exactly the contrasts that identify role.
Two-sided p = (1 + #{|β_perm| ≥ |β_obs|}) / (1 + 5000); the floor is 0.0002.
`se model` treats the 64 rows as independent, `se cluster` is the by-body
sandwich over G = 25.

| outcome | design | permuted | term | β obs | #≥ | **p** | se model | se cluster |
|---|---|---|---|---|---|---|---|---|
| NIMBY | FULL | renter | renter_share | −0.044 | 4124 | 0.825 | 0.093 | 0.154 |
| NIMBY | FULL | renter | role×renter | −0.137 | 2717 | 0.544 | 0.110 | 0.250 |
| NIMBY | MIN | renter | renter_share | −0.007 | 4808 | 0.962 | 0.069 | 0.109 |
| NIMBY | MIN | renter | role×renter | −0.195 | 2004 | 0.401 | 0.108 | 0.234 |
| NIMBY | FULL | role | **role** | **+2.057** | **0** | **0.0002** | — | — |
| NIMBY | FULL | role | role×renter | −0.137 | 3456 | 0.691 | — | — |
| NIMBY | MIN | role | **role** | **+2.116** | **0** | **0.0002** | — | — |
| NIMBY | MIN | role | role×renter | −0.195 | 2826 | 0.565 | — | — |
| YIMBY | FULL | renter | renter_share | +0.094 | 1724 | 0.345 | 0.081 | 0.074 |
| YIMBY | FULL | renter | role×renter | +0.128 | 2803 | 0.561 | 0.105 | 0.146 |
| YIMBY | MIN | renter | renter_share | +0.101 | 1562 | 0.313 | 0.056 | 0.067 |
| YIMBY | MIN | renter | role×renter | +0.158 | 2327 | 0.466 | 0.101 | 0.135 |
| YIMBY | FULL | role | **role** | **+0.997** | **0** | **0.0002** | — | — |
| YIMBY | FULL | role | role×renter | +0.128 | 3367 | 0.674 | — | — |
| YIMBY | MIN | role | **role** | **+0.995** | **0** | **0.0002** | — | — |
| YIMBY | MIN | role | role×renter | +0.158 | 3308 | 0.662 | — | — |

Added, not pre-specified, because §3 made it the only covariate worth a null —
`transit_commute` permuted across bodies the same way (`MIN_T` = role +
transit + routes):

| outcome | design | term | β obs | #≥ | **p** | se model | se cluster |
|---|---|---|---|---|---|---|---|
| NIMBY | FULL | transit_commute | −0.474 | 36 | **0.0074** | 0.139 | 0.198 |
| NIMBY | MIN_T | transit_commute | −0.313 | 68 | **0.0138** | 0.073 | 0.094 |
| YIMBY | FULL | transit_commute | +0.424 | 1 | **0.0004** | 0.101 | 0.085 |
| YIMBY | MIN_T | transit_commute | +0.159 | 264 | 0.053 | 0.061 | 0.118 |

Transit's YIMBY coefficient is 2.7× larger with the other seven covariates in
than alone (+0.424 against +0.159), which is suppression by `log_density`
(inclusion 0.587, β̄ −0.134, the opposite sign): density and transit commute
move together across these 25 bodies and partly cancel. Read the FULL and
MIN_T rows as a range, not as two estimates of one number.

### 4b. Selection-aware null, 2,000 permutations

`analysis/renter_share4_selperm.py`. Quoting the best model's +0.236 against
§4a's FULL-design null would be post-selection inference: §4a nulls a
coefficient in a *fixed* design, while selection chose that coefficient partly
*because* it was large. So the **whole 384-model all-subsets selection is
re-run inside every permutation** (ĉ re-estimated per draw from the permuted
global model, so the weights are internally consistent), and the selected
quantities are nulled directly: the model-averaged β, the β in the QAICc-best
model (0 when the best model omits the term), and the Akaike inclusion
probability. renter share and transit commute are permuted across bodies in the
same draw. 1.54 M GLM fits, 265 s per outcome. The observed statistics
reproduce §3 exactly (NIMBY renter β̄ −0.0231, inclusion 0.3160), which checks
the two scripts against each other.

| outcome | term | statistic | observed | **p** | null sd | null 2.5 % | null median | null 97.5 % |
|---|---|---|---|---|---|---|---|---|
| NIMBY | renter | model-averaged β | −0.023 | 0.681 | 0.135 | −0.304 | +0.001 | +0.328 |
| NIMBY | renter | β in best model | 0.000 | 1.000 | 0.156 | −0.351 | 0.000 | +0.351 |
| NIMBY | renter | inclusion prob | 0.316 | 0.678 | 0.250 | 0.240 | **0.405** | 0.996 |
| NIMBY | transit | model-averaged β | −0.344 | **0.021** | 0.123 | −0.260 | −0.005 | +0.294 |
| NIMBY | transit | β in best model | −0.313 | 0.062 | 0.148 | −0.329 | 0.000 | +0.322 |
| NIMBY | transit | inclusion prob | 0.900 | 0.082 | 0.245 | 0.197 | **0.314** | 0.984 |
| YIMBY | renter | model-averaged β | +0.128 | 0.080 | 0.070 | −0.154 | −0.002 | +0.151 |
| YIMBY | renter | **β in best model** | **+0.236** | **0.027** | 0.081 | −0.204 | 0.000 | +0.212 |
| YIMBY | renter | inclusion prob | 0.714 | 0.196 | 0.240 | 0.238 | **0.378** | 0.995 |
| YIMBY | transit | model-averaged β | +0.235 | **0.012** | 0.071 | −0.183 | +0.000 | +0.153 |
| YIMBY | transit | β in best model | +0.272 | **0.014** | 0.091 | −0.220 | 0.000 | +0.222 |
| YIMBY | transit | inclusion prob | 0.802 | 0.064 | 0.200 | 0.194 | **0.266** | 0.899 |

Three things to take from the null columns:

1. **Inclusion probabilities are nearly uninformative at this n.** With the
   covariate deliberately scrambled, the Akaike inclusion probability has
   median 0.27–0.41 and a 97.5th percentile of 0.90–0.996. Renter share's 0.714
   on YIMBY is p = 0.196 against that; transit's 0.900 on NIMBY is p = 0.082.
   **No inclusion probability in §3 is significant at 5 %.**
2. The selection-aware p is *larger* than §4a's fixed-design p for transit
   (0.021 against 0.0074 on NIMBY; 0.012 against 0.0004 on YIMBY) — the
   selection step costs about half an order of magnitude, as it should.
3. Renter share on YIMBY is the one place the two approaches disagree:
   p = 0.027 on the best-model coefficient, p = 0.080 on the model-averaged
   one, p = 0.31–0.34 in §4a's fixed designs. **Uncorrected, and 2 outcomes ×
   8 covariates = 16 primary tests**; Bonferroni ×16 leaves nothing but role.
   The defensible statement is "marginal, direction as expected, not
   established".

## 5. Caveats

Runs 1–3's limitations all carry forward. Specific to this run:

1. **The role column is model output, not a gold label** — `role_audit`
   (`plan/ROLE-AUDIT-2026-10-18.md` ra-1), SELF_ID no human has checked. §4a
   shows the role gap is not an artefact of the *stacking* (p = 0.0002 under
   within-cell label swaps); it says nothing about whether the audit's
   dais/public split is right. A systematic audit bias still produces this
   gap. **δ remains unquotable as a finding about deliberation until the role
   column is hand-checked.** Run 3 §5.2's definitional point (dais speech
   includes procedural and staff-report material) is also untouched by any
   permutation.
2. **n ≈ 64 rows but effectively 25 bodies.** Covariates vary at body level and
   are duplicated across a body's role rows and route cells. AICc and BIC use
   n = 64, which **understates both penalties**; the permutation tests, which
   permute at body level, are the ones that respect the real n, and they are
   the reason transit and renter share are weaker here than their SE ratios
   suggest. Treat every QAICc weight as optimistic.
3. **Selection is undecided.** Top-ten weight is 0.37 / 0.27; the best model
   carries 0.091 / 0.069. There is no "the model" to report, which is why §3
   rather than §2 is the result.
4. **ĉ is fixed from the global model** (2.580 / 1.989) per Burnham & Anderson.
   Per-model dispersions (`chi2_disp` in `subsets_results.json`) range wider;
   re-estimating ĉ per model would make QAICc non-comparable across the set.
5. **No random effects.** Run 3's body random intercept and row-level
   overdispersion are replaced here by a single quasi-binomial ĉ, because 384 ×
   2 hierarchical fits is not affordable and AICc is not defined for them (run
   3 §3 declined PSIS-LOO for the same reason). This run is therefore a
   *complement* to run 3, not a replacement: run 3 has the right error
   structure, this run has the model-space sweep. They agree on role to three
   decimals, which is the cross-check available.
6. **No spike-and-slab.** BMA is the BIC approximation (§1). The compute went
   to §4b instead; a PyMC spike-and-slab over 9 terms would be the better
   Bayesian answer and is not run here.
7. **The interaction is still unpowered**, not shown absent: 29 within-body
   role contrasts, renter share spanning 0.192–0.588. Inclusion 0.076 / 0.173
   and permutation p 0.40–0.69 are consistent with "no effect" and with "an
   effect this design cannot see" (run 3 §5.3).
8. **Ecological inference**, unchanged and still central: `renter_share` and
   `transit_commute` are properties of a jurisdiction's *residents*; the
   outcome is what the people who turned up and spoke said. **Transit commute
   share is the covariate this run promotes, and it is exactly as ecological as
   renter share** — "places where more residents commute by transit have less
   NIMBY public comment" is not "transit commuters make less NIMBY comment".
9. **Multiplicity, uncorrected.** 384 models × 2 outcomes for selection; 16
   primary permutation tests. Only role clears any correction.
10. **`transit_commute` was not pre-specified.** It was chosen for §4a's extra
    rows and §4b *after* seeing §3. §4b's selection-aware null is the correct
    reading of it; §4a's FULL/MIN_T p-values for transit are not protected
    against that look.

## Corrections 2026-10-19

Applied from `plan/REGRESSION-REVIEW-2026-10-19.md`. **Every prior number is
kept above with a strikethrough or a correction note; nothing was silently
replaced.** `output/renter_share_4/` is unchanged — the corrections live in
`output/renter_share_5/` and in runs 2 and 3.

| # | what changed | where | why |
|---|---|---|---|
| 1 | role gap restated: **+0.84 / odds ratio 2.3** on the directional denominator, **+0.78 / 2.2** with staff excluded; +2.110 kept as the old denominator's number | §0 | ~60 % of the old denominator was the `neutral` class (a dais phenomenon) and ~20 % staff clusters (94 % neutral). The *stability* of role across all 384 models is unaffected; the magnitude is. |
| 2 | **the within-cell role permutation is withdrawn as a test of δ**; p = 0.0002 is not quotable | §0, §4a | `N_i` and `y_i` stay attached to the row under a label swap, so the test is about the label's placement, not δ; and no across-body role permutation exists, because every body has the same role composition. |
| 3 | §4a's single-covariate permutation flagged invalid for a **partial** coefficient; Freedman–Lane and whole-vector replacements pointed to | §4a | Permuting one of eight collinear covariates breaks the design's joint distribution rather than only the covariate-to-outcome link. |
| 4 | renter share's "marginal on YIMBY" weakened with the LOBO and the valid nulls | §0 | p = 0.046 / 0.095 in the two-covariate design, 0.51 / 0.35 with the other six in, and −72 % on dropping Santa Clara alone. |

**What did not change, and is the most useful thing in this run.**
`transit_commute` is still the covariate the data wants — highest inclusion
probability on both outcomes, in all ten top NIMBY models, and now p < 0.05
in all eight of its Freedman–Lane and whole-vector rows in run 2 §3d-perm and
clearing 95 % in run 3 §3a-full. §0's methodological warning — **an Akaike
inclusion probability of 0.7 means nothing at this n**, median 0.38 and 97.5th
percentile 0.995 under a scrambled null — stands untouched and is the single
most transferable sentence in the four documents. So does §5.2's point that
AICc and BIC use n = 64 when the effective n is 25, which the review's design
effects (ICC 0.275 by episode, 0.156 by meeting; run 3 §5.10) now quantify
from a second direction: ĉ = 1.99 and a meeting design effect of 2.34 are the
same problem counted twice.

## 6. Files

New:
- `plan/RENTER-SHARE-4-SUBSETS.md` (this)
- 2026-10-19 corrections: see `plan/RENTER-SHARE-2.md` §3d-perm and
  `plan/RENTER-SHARE-3-ROLE.md` §0a / "Corrections 2026-10-19"; scripts
  `analysis/renter_share5_{role_restated,robust,ppc_horseshoe}.py`
- `analysis/renter_share4_subsets.py` — all subsets + pre-specified permutations
- `analysis/renter_share4_selperm.py` — selection-aware permutation null
- `output/renter_share_4/subsets_top.csv` — top 25 models per outcome, QAICc /
  QBIC / plain AICc / BIC, weights, role & renter & interaction β and SE
- `output/renter_share_4/inclusion.csv` — Akaike and BIC-BMA inclusion
  probabilities, model-averaged β, unconditional and conditional
- `output/renter_share_4/permutation.csv` — §4a, all 20 pre-specified rows plus
  the 4 transit rows
- `output/renter_share_4/selperm.csv` — §4b, 12 selection-aware rows
- `output/renter_share_4/subsets_results.json` — ĉ, all 384 weights'
  summaries, top-25 full coefficient vectors, per-model dispersion, cov
  mean/sd, permutation detail
- `output/renter_share_4/selperm_results.json` — null distributions (sd,
  2.5/50/97.5 percentiles) per statistic
- `output/renter_share_4/subsets.log`, `output/renter_share_4/selperm.log`
- `logs/rs4_subsets.out`, `logs/rs4_selperm.out`

Modified: nothing. `output/renter_share_3/` and `output/renter_share_2/` are
**read only**; the kept covariate set is asserted equal to
`output/renter_share_2/screen.json` at load, nothing re-pulled, nothing
re-screened. No figures. No DB write, no episode or stance row touched, nothing
committed in `~/ytpt`. `.env`, `plan/frame.csv`, `plan/DECISIONS.md`,
`plan/RENTER-SHARE.md`, `plan/RENTER-SHARE-2.md` and
`plan/RENTER-SHARE-3-ROLE.md` untouched.
