# RENTER-SHARE-2 — the second regression: more covariates, the post-S2 outcomes

**EXPLORATORY — the stance κ ≥ 0.6 hand-coding gate (`plan/SHOWCASE.md` 4a)
is still not passed.** Run 2026-10-19 by the supervisor agent on Puck's
request ("do another regression with the rent share, median income, and what
other covariates can you consider? no need for the plot just the
regression stats"). Offline from `output/stance_s1/nimby_view/`,
`output/wave1/` and ten unauthenticated Census summary-file pulls. **No
YouTube request, no GPU, no Claude call, no API key, $0.** No DB write.
`.env`, `plan/frame.csv`, `plan/DECISIONS.md` untouched. Nothing committed.
**No figures** — Puck asked for the stats only.

Companions: `plan/RENTER-SHARE.md` (run 1, the turn-unit version),
`plan/STANCE-S2.md` (the outcome table this run uses),
`plan/VALIDATION-2026-10-18.md` §5 and "Validation applied" (the two
discrepancies and the coverage fix, both carried forward here),
`plan/ROLE-AUDIT-2026-10-18.md` (rule ra-1).

## 0. Plain-language answer

> **Corrected 2026-10-19** from `plan/REGRESSION-REVIEW-2026-10-19.md`.
> Three statements in this section were wrong or over-claimed and are struck
> through below: "nothing survives regularisation", "τ is the posterior saying
> there is nothing here", and the dismissal of `log_population` as a corpus
> artefact. The renter-share null stands; "and nor does anything else" does
> not. See "Corrections 2026-10-19".

~~**Nothing survives regularisation.**~~ **The renter-share null stands and is
weaker than advertised; the covariate that does something is
`transit_commute`, which nobody asked about.** With renter share and median
income alone, the one result that looks like something is the public **YIMBY**
share:
+0.287 log-odds per standard deviation of renter share (90 % +0.026 to
+0.529), i.e. renter-heavier bodies have slightly more pro-housing public
comment, with a 96 % posterior that the sign is positive. Add nothing but
population density and that interval re-crosses zero. Put all eight surviving
covariates in at once under a regularised horseshoe — a prior that lets a
genuinely large coefficient stay large and pulls the rest to zero — and the
renter-share coefficient collapses to **+0.023** (90 % −0.043 to +0.317). The
same happens to the NIMBY side, to the all-roles outcomes, and to median
income, whose only interval excluding zero (all-roles YIMBY, −0.185) also
vanishes. ~~The horseshoe's shrinkage factor τ lands at 0.03–0.10 on every fit,
which is the posterior saying *there is nothing here big enough to keep*.~~
**Corrected: τ₀ is fixed at 0.124 by construction (p₀ = 2, p = 8, σ = 2,
n = 29), so τ = 0.03–0.10 sits in the *prior* bulk. The honest wording is
"indistinguishable from the prior at this n", not "the posterior says there is
nothing here" — and §3b-prior shows a deliberately permuted null reproduces
τ = 0.037 and β_renter = −0.006, i.e. the same picture with the signal
destroyed.**

Renter share's own public-YIMBY near-result is also **one body**: the
unpenalised public YIMBY slope goes +0.344 → **+0.096 without Santa Clara**,
and the public NIMBY slope −0.271 → −0.043 (§3a-lobo). Twenty-one of the other
bodies move it by under 20 %.

~~The one coefficient that does hold up under the horseshoe is not a covariate
anyone asked about: **log population** on the all-roles YIMBY share, +0.266
(90 % +0.026 to +0.454, Pr(β>0) = 0.98). Read it as a corpus artefact, not a
finding: the two biggest bodies in the fitted set are San Jose (191 public
clusters, the body `plan/STANCE-S2.md` §5 warns is "the whole story of the
crossover") and Santa Clara (three route cells, 559 clusters between them), so
"bigger places are more YIMBY on the dais" is close to "bigger places supplied
most of the dais clusters".~~

**Corrected.** Two things were wrong with that paragraph. First, it understates
the result: `log_population` on all-roles YIMBY is +0.266 with a **95 %**
interval of **+0.002 to +0.492**, so it is the one coefficient in this document
that clears 95 %, which §3a claims nothing does. Second, "corpus artefact" is
not established — across the 35 all-roles cells
corr(`log_population`, `all_n`) = **+0.05**, and the biggest cluster suppliers
are Santa Clara and San Leandro, not the largest places (review finding 9).
The defensible wording is **"not separable from corpus composition at n = 25"**
— which is a confession of ignorance, not a diagnosis. And it is also the
coefficient the grouped K-fold has least use for (§3d-kfold).

**What the covariate sweep actually promotes is `transit_commute`**, on which
this document is silent: +0.163 (pub YIMBY) and +0.158 (all YIMBY) are the
largest non-`log_population` horseshoe medians here; it clears 95 % in run 3
(+0.337); it has the highest inclusion probability on both outcomes in run 4;
and it is the only covariate whose body-permutation p is below 5 % in every
design (§3d-perm). It is **exactly as ecological as renter share** and is not
quotable as a statement about transit commuters.

So run 2 does not change run 1's answer, it hardens it: **on this corpus a
body's renter share does not predict what is said at its meetings in any
direction the data can resolve**, and now that holds after controlling for
income, density, housing age, racial composition, transit commuting, vacancy
and population size. The constraint is **n**, not the covariate list — 22
bodies, 29 body × route cells. Adding covariates to 22 bodies buys precision
on nothing; it mostly demonstrates that the two-covariate result was a
two-covariate result.

## 1. Covariates — sources, definitions, what was dropped

`analysis/renter_share2_acs.py` → `output/renter_share_2/{acs_wide.json,
covariates_wide.csv}` (29 bodies, 24 columns).

**Source and vintage, unchanged from run 1.** `api.census.gov` still requires
a key this project does not have, so everything comes from the **no-key
table-based ACS 5-year Summary File**, `ACS 2019–2023 5-year`, geography
vintage 2023:

    https://www2.census.gov/programs-surveys/acs/summary_file/2023/
      table-based-SF/data/5YRData/acsdt5y2023-<table>.dat

Run 1's three tables (B25003, B19013, B01003), the geography crosswalk and the
2023 Gazetteer ALAND are **re-read from `output/renter_share/raw/`** — not
re-pulled. Ten new tables were pulled once each, 1 s apart, by
`output/renter_share_2/raw/dl.sh` (log `raw/dl.log`, all `rc=0`, 494 MB).

**Geography lookup — no new lookup was needed.** `renter_share2_acs.py`
imports `BODY_GEO` and `SUMLEVEL` from `analysis/renter_share_acs.py`, so the
two tables cannot drift: councils are census **places**, `napa` is **Napa
County** (Board of Supervisors), and the three boards are unions of **member
counties** — **BART = Alameda + Contra Costa + San Francisco**, **VTA = Santa
Clara County**, **SamTrans = San Mateo County**. Over a union, every
count-based ratio is Σnumerator / Σdenominator (exact); every ACS **median**
(income, home value, gross rent, age) is a population-weighted mean of county
medians, which is *not* a median, and those rows carry
`medians_are_weighted_means = 1`. Only `bart` is affected, and bart does not
enter the public-stance fits at all (§2).

| covariate | table | definition | run-1? |
|---|---|---|---|
| `renter_share` | B25003 | `E003 / E001` renter-occupied / occupied units | yes |
| `med_hh_income` → `log_income` | B19013 | `E001` | yes |
| `pop_density` → `log_density` | B01003 + Gazetteer | pop / ALAND sq mi | yes |
| `population` → `log_population` | B01003 | `E001` | new |
| `med_home_value` → `log_home_value` | B25077 | `E001` | new |
| `med_gross_rent` → `log_gross_rent` | B25064 | `E001` | new |
| `rent_burden_30` | B25070 | `(E007..E010) / (E001 − E011)` — renter households at ≥ 30 % of income on gross rent, "not computed" excluded from the base | new |
| `built_2010_plus` | B25034 | `(E002 + E003) / E001` units built 2010 or later | new |
| `multiunit_5plus` | B25024 | `(E005..E008) / E001` units in 5+ unit structures | new |
| `ba_plus` | B15003 | `(E022..E025) / E001` bachelor's or higher, pop 25+ | new |
| `median_age` | B01002 | `E001` | new |
| `nhwhite_share` | B03002 | `E003 / E001` non-Hispanic white | new |
| `transit_commute` | B08301 | `E010 / E001` public-transport commute share | new |
| `vacancy_rate` | B25002 | `E003 / E001` vacant housing units | new |

Dollar and count covariates enter as logs (heavy right tails over Bay Area
places — Los Altos median home value is 3.6× Vallejo's); shares and median age
enter raw. All are standardised inside each fit, so β is "log-odds per 1 sd".

**Three of the new covariates are top-coded on this sample**, which is its own
reason not to trust them: `med_home_value` hits the ACS $2,000,001 ceiling for
**cupertino, losaltos, paloalto**, `med_gross_rent` hits $3,501 for
**cupertino, losaltos**, and `med_hh_income` hits $250,001 for **losaltos**
(that last one is in run 1's table too). A top-coded covariate is censored,
not measured, and the censoring is concentrated in exactly the low-renter-share
bodies. Both top-coded housing covariates are dropped by the VIF screen below
on independent grounds.

**Spot-check against the raw files.** Two ratios recomputed by hand straight
out of the `.dat` files reproduce the built table to 6 dp: Oakland
(`1600000US0653000`) `transit_commute` = 32,964 / 227,272 = 0.145042 and
`vacancy_rate` = 14,668 / 187,847 = 0.078085.

**The two non-ACS covariates were considered and not used as covariates.**
*Locate route* is in every model, but as the **fixed-effect stratum** it has
always been (`analysis/facets.py`; run 1 §5.5) — a route is an instrument, not
a property of a body, and Petaluma, Napa, Santa Clara, Richmond and San
Leandro each contribute cells on two or three routes that disagree with each
other. *Body type* is **unidentifiable here**: of the 29 public-stance cells,
25 are `council`, 3 are the one county board (Napa, on three routes) and 1 is
a transit board (VTA). A body-type effect would be a Napa effect plus a VTA
effect, so it is reported as a cross-tab and left out of the design matrix.
BART and SamTrans — the two bodies whose geography run 1 §5.6 flagged as
least trustworthy — clear no public-stance floor at all, which removes that
worry from the primary outcome.

### Collinearity — correlation matrix and VIFs

`output/renter_share_2/{covariate_corr.csv, covariate_vif.csv}`,
`screen.json`. Computed over the **22 fitted bodies** (one row per body, not
per cell: 22 is the n that limits the design). Worst pairs:

| pair | Pearson r |
|---|---|
| `log_income` ~ `log_gross_rent` | **+0.956** |
| `log_income` ~ `ba_plus` | **+0.935** |
| `log_home_value` ~ `ba_plus` | +0.935 |
| `log_home_value` ~ `log_gross_rent` | +0.933 |
| `log_income` ~ `log_home_value` | +0.907 |
| **`renter_share` ~ `multiunit_5plus`** | **+0.900** |
| `rent_burden_30` ~ `ba_plus` | −0.823 |
| `log_income` ~ `rent_burden_30` | −0.821 |
| `median_age` ~ `nhwhite_share` | +0.741 |

In the Bay Area, income, home value, gross rent and education are one latent
variable with four names, and the renter share of a city is nearly the same
number as its multi-unit share. VIF over all 14 candidates confirms it:

| covariate | VIF, all 14 | VIF, kept set | kept |
|---|---|---|---|
| `log_income` | **188.0** | 1.74 | ✓ (forced) |
| `ba_plus` | **86.3** | — | dropped 1st |
| `log_gross_rent` | **84.2** | — | dropped 2nd |
| `log_home_value` | **39.3** | — | dropped 3rd |
| `multiunit_5plus` | **24.0** | — | dropped 4th |
| `renter_share` | **16.0** | 1.56 | ✓ (forced) |
| `median_age` | 15.4 | — | dropped 6th |
| `nhwhite_share` | 13.1 | 2.81 | ✓ |
| `rent_burden_30` | 9.1 | — | dropped 5th |
| `transit_commute` | 8.3 | 3.43 | ✓ |
| `built_2010_plus` | 5.6 | 2.92 | ✓ |
| `log_population` | 3.7 | 2.82 | ✓ |
| `log_density` | 2.7 | 2.13 | ✓ |
| `vacancy_rate` | 2.7 | 1.58 | ✓ |

The screen is a greedy drop: remove the highest-VIF non-forced covariate,
recompute, stop when every VIF < 5. `renter_share` and `log_income` are the
question and are never droppable — and note that it is *they* that make
`multiunit_5plus` and `ba_plus` redundant, not the other way round:
`renter_share`'s own VIF falls 16.0 → 1.56 once its collinear partners go.
**Kept: `renter_share`, `log_income`, `log_density`, `built_2010_plus`,
`nhwhite_share`, `transit_commute`, `vacancy_rate`, `log_population`** — eight
covariates.

**Why n forces sparsity.** Eight covariates plus an intercept plus four route
strata plus a cell-level overdispersion term, against **29 cells over 22
bodies** with a median of 16 clusters per cell. That is p ≈ 13 fixed effects
on n = 29, and the covariates vary at the *body* level, so the effective n is
22, not 29. An unregularised eight-covariate fit on 22 bodies would be a
posterior almost entirely determined by the priors. Hence the three-model
ladder in §3: the honest two-covariate model, one controlled extension, and a
regularised full model that is allowed to keep a large coefficient if one
exists.

## 2. Outcomes — the post-S2 cluster unit, per route

`analysis/renter_share2_outcomes.py` →
`output/renter_share_2/{outcomes_clusters.csv, rates_complete.csv}`.

**This is the change from run 1.** Run 1's unit was the **turn**, off
`turn_stance` as it stood on 2026-10-18 (9,358 rows), because the cluster
table was too thin. S2 has landed (`plan/STANCE-S2.md`: `turn_stance` =
13,123 rows, `role_audit` rewritten over 4,840 clusters, a fifth locate route
`fullasr`), and the per-cluster view now carries **861 public and 1,952 dais
clusters with a stance** out of 3,701. So run 2 reads the cluster table —

    output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv

the post-S2, `role_audit` (rule ra-1) roll-up, one row per episode × speaker
cluster. A cluster "has a stance" iff `label != "no stance"`. The speaker
cluster is also the unit `STANCE-S1-RESULTS.md` §7b and the site panels use,
so these outcomes line up with the published per-body figures instead of
running beside them. Family frame is inherited: **density + both** only.

Four binomial outcomes per body × route cell, floor **5 clusters** (run 1's
floor, on the new unit):

| outcome | numerator | denominator | cells | bodies | clusters |
|---|---|---|---|---|---|
| 1 public NIMBY | public NIMBY clusters | public clusters with a stance | **29** | **22** | 843 |
| 2 public YIMBY | public YIMBY clusters | same | 29 | 22 | 843 |
| 3 all-roles NIMBY | public + dais NIMBY | all clusters with a stance | **35** | **25** | 2,801 |
| 4 all-roles YIMBY | public + dais YIMBY | same | 35 | 25 | 2,801 |

843 of the 861 public clusters are inside the floor. Route composition of the
29 public cells: classifier 18, agenda_index 4, local_asr 3, fullasr 3,
swagit_transcript 1. The six extra all-roles cells are bodies with dais
clusters but under five public ones: bart (23), vallejo (22), milpitas (6),
fullasr hayward (51), fullasr napa (22), local_asr hayward (41).

The fitted cells, in renter order (public n / NIMBY / YIMBY, then all-roles):

| route | body | renter | pub n | N | Y | all n | N | Y |
|---|---|---|---|---|---|---|---|---|
| classifier | losaltos | 0.192 | 14 | 7 | 3 | 84 | 16 | 10 |
| classifier | livermore | 0.280 | 15 | 7 | 7 | 34 | 10 | 11 |
| classifier | pleasanton | 0.329 | 13 | 3 | 8 | 45 | 8 | 13 |
| agenda_index | napa | 0.356 | 5 | 1 | 3 | 13 | 1 | 4 |
| classifier | napa | 0.356 | 9 | 4 | 4 | 60 | 5 | 15 |
| local_asr | napa | 0.356 | 37 | 23 | 12 | 67 | 32 | 24 |
| classifier | petaluma | 0.361 | 16 | 9 | 6 | 58 | 16 | 11 |
| swagit | petaluma | 0.361 | 31 | 8 | 17 | 204 | 25 | 60 |
| classifier | concord | 0.386 | 22 | 9 | 13 | 37 | 14 | 18 |
| classifier | dalycity | 0.395 | 5 | 0 | 5 | 29 | 0 | 10 |
| classifier | fremont | 0.398 | 6 | 3 | 0 | 53 | 6 | 7 |
| classifier | healdsburg | 0.410 | 8 | 6 | 1 | 28 | 10 | 5 |
| fullasr | sanleandro | 0.417 | 6 | 4 | 0 | 24 | 6 | 3 |
| local_asr | sanleandro | 0.417 | 173 | 84 | 75 | 356 | 103 | 103 |
| agenda_index | sausalito | 0.421 | 24 | 11 | 9 | 65 | 16 | 14 |
| agenda_index | hayward | 0.431 | 6 | 6 | 0 | 20 | 6 | 1 |
| classifier | santarosa | 0.438 | 29 | 10 | 15 | 81 | 12 | 25 |
| classifier | sanjose | 0.441 | **191** | 125 | 59 | 293 | 142 | 87 |
| classifier | vta | 0.447 | 22 | 12 | 6 | 58 | 14 | 23 |
| classifier | paloalto | 0.458 | 37 | 19 | 11 | 108 | 20 | 25 |
| fullasr | richmond | 0.475 | 10 | 6 | 3 | 39 | 8 | 11 |
| local_asr | richmond | 0.475 | 24 | 8 | 13 | 143 | 17 | 31 |
| classifier | sanrafael | 0.482 | 13 | 8 | 4 | 28 | 9 | 6 |
| classifier | sanmateo | 0.492 | 24 | 5 | 13 | 71 | 6 | 21 |
| classifier | sunnyvale | 0.562 | 9 | 6 | 3 | 52 | 8 | 9 |
| classifier | oakland | 0.583 | 17 | 3 | 14 | 27 | 5 | 17 |
| agenda_index | santaclara | 0.588 | 14 | 2 | 9 | 112 | 8 | 30 |
| classifier | santaclara | 0.588 | 45 | 13 | 26 | 302 | 52 | 85 |
| fullasr | santaclara | 0.588 | 18 | 10 | 7 | 145 | 32 | 22 |

Two cells are 23 % and 21 % of the public base on their own (San Jose 191,
San Leandro 173) and both sit mid-range on renter share, so they anchor the
regression near its centre rather than at a tail. San Leandro's `turn_stance`
sign-flip problem on `affordability_displacement` items is diluted but not
resolved (`plan/STANCE-S2.md` §5).

**Outcome 5 — density minutes per meeting-hour — is reported but not
modelled.** After the 2026-10-18 coverage fix
(`plan/VALIDATION-2026-10-18.md` "Validation applied" 1) the complete-coverage
set is **3 bodies**, not run 1's 7, and **n = 3 is not a regression**. The
complete cells, for the record (`rates_complete.csv`):

| route | body | renter | density min/h | transit min/h | meeting-h | meetings |
|---|---|---|---|---|---|---|
| classifier | bart | 0.460 | 2.61 | 29.14 | 67.7 | 16 |
| classifier | napa | 0.356 | 1.58 | 0.81 | 104.3 | 30 |
| agenda_index | napa | 0.356 | 1.26 | 0.92 | 104.3 | 30 |
| local_asr | napa | 0.356 | 1.78 | 0.00 | 104.3 | 30 |
| classifier | vta | 0.447 | 6.02 | 27.00 | 44.7 | 11 |

Two of the three bodies are transit boards whose transit time is close to a
tautology (`plan/WAVE1-RESULTS.md` §5.1), and run 1's rate correlations were
already withdrawn from quotation on this ground. Nothing here is quotable
either.

## 3. Models

`analysis/renter_share2_models.py`, PyMC 6.3.2 / ArviZ 1.3.0, 4 chains × 2,000
draws, 1,500 tuning (4,000 for the horseshoe), `target_accept` 0.95 (0.99 for
the horseshoe), seed 20261019. `output/renter_share_2/models/`.

Unit = body × route cell. Every fit:

    y_i        ~ Binomial(N_i, p_i)
    logit p_i  = α + Σ_j β_j z_ji + γ_route(i) + u_i
    u_i        ~ Normal(0, σ_cell)      σ_cell ~ HalfNormal(0.75)
    α          ~ Normal(0, 1.5)         γ      ~ Normal(0, 1)

`u_i` is the body-level overdispersion term (one cell per body × route, so it
absorbs exactly the extra-binomial spread a pure binomial would read as
signal); route is a fixed-effect stratum, reference = `classifier`, never
pooled. Covariates standardised over the fitted cells.

Three specifications per outcome:

* **(a) base** — `renter_share + log_income`, β ~ Normal(0, 1).
* **(b) base + density** — adds `log_density`, same priors.
* **(c) full, regularised horseshoe** — all eight kept covariates under the
  regularised horseshoe of Piironen & Vehtari (2017):
  β_j = z_j·τ·λ̃_j, z_j ~ N(0,1), λ_j ~ HalfCauchy(1), τ ~ HalfCauchy(τ₀) with
  τ₀ = p₀/(p−p₀)·σ/√n (p₀ = 2 expected relevant covariates, σ = 2 the logistic
  pseudo-sd), c² ~ InvGamma(2, 2·2²), λ̃_j = √(c²λ_j²/(c²+τ²λ_j²)). The slab
  keeps a genuinely large coefficient identified while τ₀ pulls the rest to
  zero — "which of these, if any, survives?" stated as a prior.

### 3a. Renter share and median income, all twelve fits

β per 1 sd of renter share (sd = 0.0924 on the public cells, 0.0853 on the
all-roles cells). `model_table.csv` carries all 12 × p rows.

| outcome | model | β renter | 90 % | 95 % | Pr(β>0) | per +10 pt | β income | 90 % |
|---|---|---|---|---|---|---|---|---|
| public NIMBY | (a) | −0.197 | −0.456, +0.067 | −0.513, +0.124 | 0.11 | −0.213 | −0.014 | −0.304, +0.261 |
| public NIMBY | (b) | −0.138 | −0.419, +0.151 | −0.478, +0.215 | 0.22 | −0.149 | −0.000 | −0.267, +0.281 |
| public NIMBY | (c) | −0.004 | −0.199, +0.060 | −0.260, +0.087 | 0.39 | −0.004 | −0.000 | −0.108, +0.107 |
| **public YIMBY** | **(a)** | **+0.287** | **+0.026, +0.529** | −0.029, +0.580 | **0.96** | **+0.310** | −0.166 | −0.423, +0.108 |
| public YIMBY | (b) | +0.226 | −0.050, +0.509 | −0.108, +0.567 | 0.91 | +0.244 | −0.175 | −0.441, +0.096 |
| public YIMBY | (c) | +0.023 | −0.043, +0.317 | −0.069, +0.382 | 0.72 | +0.025 | −0.012 | −0.294, +0.059 |
| all-roles NIMBY | (a) | −0.110 | −0.342, +0.126 | −0.385, +0.181 | 0.22 | −0.128 | −0.019 | −0.282, +0.231 |
| all-roles NIMBY | (b) | −0.111 | −0.373, +0.153 | −0.433, +0.204 | 0.24 | −0.130 | −0.017 | −0.274, +0.243 |
| all-roles NIMBY | (c) | −0.001 | −0.115, +0.065 | −0.167, +0.097 | 0.45 | −0.001 | −0.000 | −0.096, +0.079 |
| all-roles YIMBY | (a) | +0.126 | −0.013, +0.276 | −0.041, +0.301 | 0.93 | +0.148 | **−0.185** | **−0.341, −0.027** |
| **all-roles YIMBY** | **(b)** | **+0.166** | **+0.009, +0.338** | −0.027, +0.371 | **0.96** | **+0.194** | **−0.171** | **−0.332, −0.010** |
| all-roles YIMBY | (c) | +0.026 | −0.036, +0.171 | −0.053, +0.201 | 0.74 | +0.030 | −0.028 | −0.184, +0.038 |

Diagnostics, all twelve: **max R̂ 1.0013–1.0174**, min bulk ESS **316–2,547**,
**divergences 0 except (c) public NIMBY 1 and (c) all-roles NIMBY 2** (at the
run-1 settings of 0.95/1,500 these two were 31 and 69; the funnel needed
0.99/4,000). ~~PPC is clean everywhere: **29/29 and 35/35 cells** have their
observed count inside the replicated 90 % band on every fit.~~ **Superseded
2026-10-19** (review finding 3): `u_i` is one free parameter per cell, so that
band is centred on the observed count and the check cannot fail — under a
calibrated 90 % band P(29/29) = 0.9²⁹ = 0.047 and P(35/35) = 0.025. The
replacement, with `u_i` integrated out, is tabulated for all ten fits at
`plan/RENTER-SHARE-3-ROLE.md` §3c-cal: 28–29 of 29 and 33–35 of 35, PIT
indistinguishable from uniform (KS p 0.44–0.93). The two lowest
ESS (316, 377) are both horseshoe fits, where the `tau` posterior is the slow
parameter; the β posteriors they report are the shrunk ones, so the low ESS
does not change the conclusion.

~~**Only three β's anywhere have a 90 % interval excluding zero**, and no
interval excludes zero at 95 %:~~

> **Corrected 2026-10-19** (review finding 8). The lead sentence says "three"
> and the table underneath it lists **five** — the count was wrong against this
> document's own table. And **one interval does exclude zero at 95 %**:
> `log_population` on all-roles YIMBY, 95 % **+0.002 to +0.492**. The five
> rows below are right; the sentence introducing them was not.

**Five β's have a 90 % interval excluding zero and one of them also clears
95 %:**

| outcome | model | covariate | β | 90 % | 95 % | Pr(β>0) |
|---|---|---|---|---|---|---|
| public YIMBY | (a) | `renter_share` | +0.287 | +0.026, +0.529 | −0.029, +0.580 | 0.965 |
| all-roles YIMBY | (a) | `log_income` | −0.185 | −0.341, −0.027 | −0.377, +0.006 | 0.029 |
| all-roles YIMBY | (b) | `renter_share` | +0.166 | +0.009, +0.338 | −0.028, +0.371 | 0.957 |
| all-roles YIMBY | (b) | `log_income` | −0.171 | −0.332, −0.010 | −0.364, +0.025 | 0.042 |
| all-roles YIMBY | **(c)** | **`log_population`** | **+0.266** | **+0.026, +0.454** | **+0.002, +0.492** | **0.980** |

### 3a-full. Every parameter, every fit

Added 2026-10-19 (review finding 1 — the same omission as run 3 §3a). All 52
substantive intervals from `output/renter_share_2/models/model_table.csv`,
nothing recomputed. α, σ_cell and the route strata are nuisance terms and stay
in the CSV. **Bold = 95 %.**

| outcome | model | covariate | β | 90 % | 95 % | Pr(β>0) | excl. 0 |
|---|---|---|---|---|---|---|---|
| public NIMBY | (a) | `renter_share` | −0.197 | −0.456, +0.067 | −0.513, +0.124 | 0.111 | — |
| public NIMBY | (a) | `log_income` | −0.014 | −0.304, +0.261 | −0.368, +0.312 | 0.468 | — |
| public NIMBY | (b) | `renter_share` | −0.138 | −0.419, +0.151 | −0.478, +0.215 | 0.223 | — |
| public NIMBY | (b) | `log_income` | −0.000 | −0.267, +0.281 | −0.322, +0.341 | 0.499 | — |
| public NIMBY | (b) | `log_density` | −0.146 | −0.454, +0.157 | −0.521, +0.225 | 0.208 | — |
| public NIMBY | (c) | `renter_share` | −0.004 | −0.199, +0.059 | −0.260, +0.087 | 0.387 | — |
| public NIMBY | (c) | `log_income` | −0.000 | −0.108, +0.107 | −0.159, +0.158 | 0.498 | — |
| public NIMBY | (c) | `log_density` | −0.002 | −0.176, +0.086 | −0.241, +0.133 | 0.415 | — |
| public NIMBY | (c) | `built_2010_plus` | −0.005 | −0.340, +0.058 | −0.474, +0.092 | 0.374 | — |
| public NIMBY | (c) | `nhwhite_share` | +0.000 | −0.137, +0.116 | −0.217, +0.167 | 0.503 | — |
| public NIMBY | (c) | `transit_commute` | −0.025 | −0.535, +0.031 | −0.648, +0.050 | 0.264 | — |
| public NIMBY | (c) | `vacancy_rate` | −0.000 | −0.114, +0.100 | −0.164, +0.144 | 0.497 | — |
| public NIMBY | (c) | `log_population` | +0.000 | −0.086, +0.131 | −0.125, +0.179 | 0.533 | — |
| public YIMBY | (a) | `renter_share` | +0.287 | +0.026, +0.529 | −0.029, +0.580 | 0.965 | 90 % |
| public YIMBY | (a) | `log_income` | −0.166 | −0.423, +0.107 | −0.476, +0.166 | 0.144 | — |
| public YIMBY | (b) | `renter_share` | +0.226 | −0.050, +0.509 | −0.108, +0.567 | 0.910 | — |
| public YIMBY | (b) | `log_income` | −0.175 | −0.441, +0.096 | −0.499, +0.155 | 0.138 | — |
| public YIMBY | (b) | `log_density` | +0.144 | −0.144, +0.438 | −0.213, +0.510 | 0.808 | — |
| public YIMBY | (c) | `renter_share` | +0.023 | −0.043, +0.317 | −0.069, +0.382 | 0.720 | — |
| public YIMBY | (c) | `log_income` | −0.012 | −0.294, +0.059 | −0.364, +0.089 | 0.333 | — |
| public YIMBY | (c) | `log_density` | +0.002 | −0.116, +0.187 | −0.168, +0.251 | 0.565 | — |
| public YIMBY | (c) | `built_2010_plus` | +0.019 | −0.057, +0.555 | −0.090, +0.657 | 0.687 | — |
| public YIMBY | (c) | `nhwhite_share` | +0.004 | −0.093, +0.333 | −0.142, +0.442 | 0.599 | — |
| public YIMBY | (c) | `transit_commute` | +0.163 | −0.017, +0.748 | −0.036, +0.861 | 0.838 | — |
| public YIMBY | (c) | `vacancy_rate` | −0.000 | −0.133, +0.121 | −0.186, +0.168 | 0.489 | — |
| public YIMBY | (c) | `log_population` | +0.000 | −0.119, +0.136 | −0.165, +0.181 | 0.506 | — |
| all-roles NIMBY | (a) | `renter_share` | −0.110 | −0.342, +0.126 | −0.385, +0.181 | 0.219 | — |
| all-roles NIMBY | (a) | `log_income` | −0.019 | −0.282, +0.231 | −0.336, +0.281 | 0.451 | — |
| all-roles NIMBY | (b) | `renter_share` | −0.111 | −0.373, +0.153 | −0.433, +0.204 | 0.239 | — |
| all-roles NIMBY | (b) | `log_income` | −0.017 | −0.274, +0.243 | −0.330, +0.297 | 0.458 | — |
| all-roles NIMBY | (b) | `log_density` | +0.005 | −0.272, +0.304 | −0.327, +0.363 | 0.514 | — |
| all-roles NIMBY | (c) | `renter_share` | −0.001 | −0.115, +0.065 | −0.167, +0.097 | 0.448 | — |
| all-roles NIMBY | (c) | `log_income` | −0.000 | −0.096, +0.079 | −0.147, +0.115 | 0.474 | — |
| all-roles NIMBY | (c) | `log_density` | −0.000 | −0.095, +0.097 | −0.136, +0.143 | 0.499 | — |
| all-roles NIMBY | (c) | `built_2010_plus` | −0.001 | −0.136, +0.078 | −0.208, +0.116 | 0.456 | — |
| all-roles NIMBY | (c) | `nhwhite_share` | +0.008 | −0.039, +0.307 | −0.064, +0.381 | 0.680 | — |
| all-roles NIMBY | (c) | `transit_commute` | −0.005 | −0.249, +0.044 | −0.328, +0.069 | 0.353 | — |
| all-roles NIMBY | (c) | `vacancy_rate` | −0.002 | −0.148, +0.059 | −0.209, +0.089 | 0.413 | — |
| all-roles NIMBY | (c) | `log_population` | −0.001 | −0.140, +0.069 | −0.199, +0.099 | 0.437 | — |
| all-roles YIMBY | (a) | `renter_share` | +0.126 | −0.013, +0.276 | −0.041, +0.301 | 0.933 | — |
| all-roles YIMBY | (a) | `log_income` | −0.185 | −0.341, −0.027 | −0.377, +0.006 | 0.029 | 90 % |
| all-roles YIMBY | (b) | `renter_share` | +0.166 | +0.009, +0.338 | −0.028, +0.371 | 0.957 | 90 % |
| all-roles YIMBY | (b) | `log_income` | −0.171 | −0.332, −0.010 | −0.364, +0.025 | 0.042 | 90 % |
| all-roles YIMBY | (b) | `log_density` | −0.102 | −0.276, +0.071 | −0.314, +0.105 | 0.161 | — |
| all-roles YIMBY | (c) | `renter_share` | +0.026 | −0.036, +0.171 | −0.053, +0.201 | 0.743 | — |
| all-roles YIMBY | (c) | `log_income` | −0.028 | −0.183, +0.038 | −0.219, +0.058 | 0.255 | — |
| all-roles YIMBY | (c) | `log_density` | −0.006 | −0.158, +0.083 | −0.193, +0.118 | 0.399 | — |
| all-roles YIMBY | (c) | `built_2010_plus` | −0.004 | −0.168, +0.109 | −0.215, +0.157 | 0.428 | — |
| all-roles YIMBY | (c) | `nhwhite_share` | +0.123 | −0.019, +0.397 | −0.037, +0.446 | 0.859 | — |
| all-roles YIMBY | (c) | `transit_commute` | +0.158 | −0.004, +0.357 | −0.015, +0.396 | 0.925 | — |
| all-roles YIMBY | (c) | `vacancy_rate` | +0.013 | −0.052, +0.150 | −0.076, +0.178 | 0.664 | — |
| all-roles YIMBY | (c) | **`log_population`** | **+0.266** | **+0.026, +0.454** | **+0.002, +0.492** | **0.980** | **95 %** |

### 3a-lobo. Leave-one-body-out, on every renter-share and transit estimate

Added 2026-10-19 (review finding 6). `analysis/renter_share5_robust.py` →
`output/renter_share_5/robust/lobo.csv`. Each body dropped in turn, same
design, IRLS (so comparable to §3c, not to the Bayesian medians above). Spec
(c) here is **all eight covariates unpenalised** — the horseshoe cannot be
leave-one-out-refitted cheaply, and an unpenalised fit is the conservative
version of the question.

| outcome | spec | term | full | LOBO min (body) | LOBO max (body) | sign flips / 22–25 |
|---|---|---|---|---|---|---|
| public NIMBY | (a) | `renter_share` | −0.271 | −0.332 (losaltos) | **−0.043 (santaclara)** | 0 |
| public NIMBY | (b) | `renter_share` | −0.243 | −0.319 (losaltos) | **−0.009 (santaclara)** | 0 |
| public NIMBY | (c)† | `renter_share` | −0.141 | −0.236 (sanrafael) | **+0.017 (santaclara)** | **1** |
| public NIMBY | (c)† | `transit_commute` | −0.883 | **−1.281 (oakland)** | −0.679 (sanjose) | 0 |
| public YIMBY | (a) | `renter_share` | **+0.344** | **+0.096 (santaclara)** | +0.421 (pleasanton) | 0 |
| public YIMBY | (b) | `renter_share` | +0.311 | **+0.052 (santaclara)** | +0.393 (pleasanton) | 0 |
| public YIMBY | (c)† | `renter_share` | +0.151 | **−0.067 (santaclara)** | +0.263 (dalycity) | **1** |
| public YIMBY | (c)† | `transit_commute` | +0.895 | +0.705 (dalycity) | **+1.251 (oakland)** | 0 |
| all-roles NIMBY | (a) | `renter_share` | −0.125 | −0.219 (losaltos) | −0.049 (santaclara) | 0 |
| all-roles NIMBY | (b) | `renter_share` | −0.143 | **−0.278 (losaltos)** | −0.065 (santaclara) | 0 |
| all-roles NIMBY | (c)† | `renter_share` | −0.108 | **−0.273 (losaltos)** | +0.047 (sanjose) | **1** |
| all-roles NIMBY | (c)† | `transit_commute` | −0.569 | −0.701 (santarosa) | **−0.038 (sanjose)** | 0 |
| all-roles YIMBY | (a) | `renter_share` | +0.101 | +0.043 (losaltos) | +0.128 (richmond) | 0 |
| all-roles YIMBY | (b) | `renter_share` | +0.129 | +0.067 (losaltos) | +0.162 (richmond) | 0 |
| all-roles YIMBY | (c)† | `renter_share` | +0.084 | **−0.019 (santaclara)** | +0.120 (richmond) | **1** |
| all-roles YIMBY | (c)† | `transit_commute` | +0.325 | +0.268 (oakland) | +0.380 (bart) | 0 |

† all eight, unpenalised.

**One body decides the sign of every public renter-share slope.** Santa Clara
has the highest renter share in the frame (0.588) and three route cells;
without it the public YIMBY slope falls to +0.096 — a 72 % loss on the base
model and a sign change on the all-eight model — and the public NIMBY slope to
−0.043. **Los Altos** (lowest renter share, 0.192) is the mirror image on the
all-roles outcomes. **Oakland** and **San José** are the leverage points on
`transit_commute`, where San José removes 93 % of the all-roles NIMBY
coefficient. **San Leandro is not a LOBO extreme on any renter-share or transit
estimate** — the risk it carries is the `affordability_displacement` sign flip
(`plan/STANCE-S2.md` §5), not leverage on a slope. Every other body moves
every estimate by under 20 %.

### 3b. The horseshoe fits in full

σ_cell and the global shrinkage τ, then all eight β medians:

| covariate | pub NIMBY | pub YIMBY | all NIMBY | all YIMBY |
|---|---|---|---|---|
| σ_cell | 0.583 | 0.509 | 0.711 | 0.260 |
| **τ** | **0.0431** | **0.0683** | **0.0337** | **0.0951** |
| `renter_share` | −0.004 | +0.023 | −0.001 | +0.026 |
| `log_income` | −0.000 | −0.012 | −0.000 | −0.028 |
| `log_density` | −0.002 | +0.002 | −0.000 | −0.006 |
| `built_2010_plus` | −0.005 | +0.019 | −0.001 | −0.004 |
| `nhwhite_share` | +0.000 | +0.004 | +0.008 | +0.123 |
| `transit_commute` | −0.025 | +0.163 | −0.005 | +0.158 |
| `vacancy_rate` | −0.000 | −0.000 | −0.002 | +0.013 |
| `log_population` | +0.000 | +0.000 | −0.001 | **+0.266** |

τ ≈ 0.03–0.10 against standardised covariates is near-total shrinkage. The
only β the slab lets through is `log_population` on the all-roles YIMBY share
(90 % +0.026, +0.454, **95 % +0.002, +0.492**; Pr(β>0) = 0.98);
`transit_commute` (+0.158, Pr 0.925) and `nhwhite_share` (+0.123, Pr 0.859) on
the same outcome are the next largest and both cross zero. Horseshoe 90 %
intervals are deliberately asymmetric — a median of +0.023 with an upper bound
of +0.317 means "almost certainly nothing, but the data cannot rule out
something moderate", which is the correct summary of 22 bodies.

### 3b-prior. τ₀, and what the horseshoe looks like against a true null

Added 2026-10-19 (review finding 5). `analysis/renter_share5_ppc_horseshoe.py`
→ `output/renter_share_5/ppc_horseshoe/horseshoe.csv`. All four fits are the
public YIMBY spec (c), 29 cells, 4 × 2,000 draws, 4,000 tuning,
`target_accept` 0.99.

**τ₀ is not estimated. It is fixed by construction at**

    τ₀ = p₀/(p − p₀) · σ/√n = 2/6 · 2/√29 = **0.1238**

so a posterior τ of 0.03–0.10 **sits inside the prior bulk**. It is not news.
Three comparisons make that concrete:

| fit | τ₀ used | τ (median) | β `renter_share` | β `transit_commute` | div. |
|---|---|---|---|---|---|
| **published prior** | 0.1238 | **0.0666** | **+0.021** | **+0.154** | 1 |
| **covariates permuted across bodies** (a true null) | 0.1238 | **0.0369** | **−0.006** | **+0.008** | 0 |
| τ₀ × 5 | 0.6190 | 0.1484 | +0.049 | **+0.343** | 0 |
| τ₀ / 5 | 0.0248 | 0.0210 | +0.004 | +0.015 | 0 |

(The published run reported τ = 0.0683 and β_renter = +0.023 for this fit;
0.0666 / +0.021 is the same fit at a different seed path, so these four rows
are internally comparable.)

**Read it this way.** τ is a factor of ~1.8 larger than it is under a
deliberately destroyed signal (0.067 against 0.037) — which is a hint, not a
result, and well inside what 29 cells will produce by chance. β_renter is
+0.021 observed against −0.006 permuted: **the renter-share coefficient under
the horseshoe is indistinguishable from its own null.** The one coefficient
that does separate from the null is `transit_commute`, +0.154 observed against
+0.008 permuted, and it is also the coefficient most sensitive to the prior:
τ₀ × 5 moves it to **+0.343** and τ₀ / 5 flattens it to +0.015, along with
everything else (all eight ≤ 0.015). A coefficient that moves 20-fold with the
prior scale on 29 cells is a coefficient the data is not determining.

**Wording rule adopted here and in §0, §4 and §5.** Everywhere this document
said the horseshoe showed there is "nothing here", the claim is now
**"indistinguishable from the prior at this n"**. The horseshoe is a
regularisation choice, not a measurement of absence; with p₀ = 2 built in, it
was always going to shrink six of eight coefficients to zero.

### 3c. Frequentist sanity check — HC3

No hierarchy, no priors, same cells, same standardised covariates, the
specification's own covariate set. `models/ols_hc3.csv`. HC3 is implemented in
`renter_share2_models.py` (statsmodels is not installed in this venv): OLS
HC3 is the MacKinnon–White `e_i²/(1−h_i)²` sandwich; the logistic is IRLS with
the matching score-residual sandwich, `X' diag(s_i²/(1−h_i)²) X` between two
copies of `(X'WX)⁻¹`. All twelve IRLS fits converged.

| outcome | model | logit β renter (HC3 SE) | 95 % | OLS β renter (HC3 SE) |
|---|---|---|---|---|
| public NIMBY | (a) | −0.176 (0.178) | −0.525, +0.174 | −0.017 (0.038) |
| public YIMBY | (a) | **+0.254** (0.164) | −0.068, +0.577 | +0.039 (0.035) |
| public YIMBY | (b) | +0.217 (0.177) | −0.130, +0.565 | +0.028 (0.045) |
| all-roles NIMBY | (a) | −0.104 (0.116) | −0.331, +0.124 | −0.015 (0.014) |
| all-roles YIMBY | (a) | +0.050 (0.099) | −0.144, +0.244 | +0.020 (0.023) |

Same signs as the Bayesian fits, similar magnitudes on the public outcomes,
and **every HC3 interval crosses zero** — including public YIMBY, whose
Bayesian 90 % did not. So the one near-result in §3a is at the edge of what
the priors contribute, not a robust frequentist finding either. The all-roles
income coefficient behaves the same way (logit −0.132, HC3 95 % −0.308 to
+0.045).

### 3d. LOO-ELPD comparison

`models/loo_table.csv`. **PSIS-LOO on the reported models is degenerate and
must not be used to pick a model.** `u_i` is one free parameter per
observation, so `p_loo` comes out at 18.1–27.0 against n = 29–35 and
**10–30 of the cells have Pareto k̂ > 0.7**. ArviZ also flags every pair as
`N < 100`. Reported as asked, with that warning attached:

| outcome | model | rank | ELPD (se) | Δ | dse | p_loo | k̂>0.7 |
|---|---|---|---|---|---|---|---|
| public NIMBY | (c) hs | 0 | −75.59 (4.13) | 0 | — | 18.1 | 15/29 |
| public NIMBY | (a) base | 1 | −76.05 (3.75) | −0.47 | 1.00 | 18.4 | 16/29 |
| public NIMBY | (b) base+dens | 2 | −76.58 (3.77) | −1.00 | 1.20 | 18.9 | 17/29 |
| public YIMBY | (c) hs | 0 | −76.61 (3.26) | 0 | — | 18.4 | 12/29 |
| public YIMBY | (b) base+dens | 1 | −77.13 (3.24) | −0.52 | 1.70 | 18.7 | 10/29 |
| public YIMBY | (a) base | 2 | −77.28 (3.63) | −0.66 | 1.80 | 18.4 | 13/29 |
| all-roles NIMBY | (b) base+dens | 0 | −105.16 (4.12) | 0 | — | 26.2 | 29/35 |
| all-roles NIMBY | (c) hs | 1 | −105.28 (4.41) | −0.13 | 1.10 | 26.4 | 28/35 |
| all-roles NIMBY | (a) base | 2 | −106.05 (4.19) | −0.90 | 0.71 | 27.0 | 30/35 |
| all-roles YIMBY | (c) hs | 0 | −104.84 (4.89) | 0 | — | 19.6 | 15/35 |
| all-roles YIMBY | (a) base | 1 | −108.31 (4.33) | −3.50 | 2.50 | 22.3 | 15/35 |
| all-roles YIMBY | (b) base+dens | 2 | −108.52 (4.32) | −3.70 | 2.30 | 22.9 | 20/35 |

Every Δ except the last outcome's is inside 1 dse: on three of four outcomes
the three covariate sets are **indistinguishable**, and the ranking is noise.

To get a comparison that is at least internally valid, the same three
covariate sets were re-fitted **without** `u_i` — not the reported models (the
brief requires overdispersion), but a plain binomial whose per-observation
likelihood is proper, so LOO means something. There the horseshoe ranks first
on all four outcomes, and on public NIMBY by a real margin
(Δ −14.0, dse 10.0; k̂ > 0.7 in 2–6 cells, not 16):

| outcome | rank 0 | rank 1 (Δ, dse) | rank 2 (Δ, dse) |
|---|---|---|---|
| public NIMBY | (c) hs −81.30 | (a) −95.65 (−14.0, 10.0) | (b) −98.74 (−17.0, 12.0) |
| public YIMBY | (c) hs −77.00 | (a) −83.68 (−6.7, 5.5) | (b) −84.98 (−8.0, 5.8) |
| all-roles NIMBY | (c) hs −178.55 | (a) −203.84 (−25.0, 38.0) | (b) −210.26 (−32.0, 36.0) |
| all-roles YIMBY | (c) hs −107.30 | (a) −114.81 (−7.5, 5.2) | (b) −115.98 (−8.7, 5.0) |

Read that correctly: without `u_i` the extra covariates are the only thing
available to soak up body-level overdispersion, so the horseshoe wins by
partly *doing the overdispersion term's job*. It is not evidence that the
covariates matter. Combined with §3a, the two LOO blocks say the same thing
from opposite directions: **the three covariate sets predict about equally
well, and the full model's advantage is absorption of spread, not signal.**

> **Superseded 2026-10-19** (review finding 7). The whole of §3d is retained
> above but **neither LOO block should be quoted, and "the horseshoe ranks
> first" is dropped.** PSIS-LOO is degenerate on the reported models (as §3d
> itself says), and the no-`u_i` refit that rescues it is not the reported
> model. A grouped K-fold that holds out whole *bodies* is valid for both, and
> it **reverses the ranking**: §3d-kfold.

### 3d-kfold. Grouped 5-fold by body — the replacement for §3d

`analysis/renter_share5_robust.py` → `output/renter_share_5/robust/kfold.csv`.
Bodies are assigned to 5 folds (seed 20261019); each fold's bodies are held
out entirely and the covariate set refitted by **unpenalised** binomial logit
on the rest, so a large model is rightly punished rather than rescued by a
prior. ELPD is the held-out binomial log predictive density summed over cells;
`se` is the usual √n · sd of the pointwise terms. Route strata are in every
set and are fitted on the training fold.

| outcome | covariate set | p | **K-fold ELPD (se)** |
|---|---|---|---|
| public NIMBY | **route only** | **5** | **−109.72 (24.83)** |
| public NIMBY | all eight | 13 | −110.51 (13.97) |
| public NIMBY | renter+income | 7 | −119.72 (29.06) |
| public NIMBY | +density | 8 | −121.27 (31.23) |
| public YIMBY | **route only** | **5** | **−93.76 (11.58)** |
| public YIMBY | renter+income | 7 | −107.76 (14.88) |
| public YIMBY | +density | 8 | −133.54 (18.05) |
| public YIMBY | all eight | 13 | −237.63 (80.14) |
| all-roles NIMBY | **route only** | **5** | **−247.42 (72.13)** |
| all-roles NIMBY | renter+income | 7 | −284.65 (73.62) |
| all-roles NIMBY | +density | 8 | −302.04 (78.91) |
| all-roles NIMBY | all eight | 13 | −482.66 (207.43) |
| all-roles YIMBY | **route only** | **5** | **−120.88 (11.35)** |
| all-roles YIMBY | renter+income | 7 | −121.62 (9.38) |
| all-roles YIMBY | +density | 8 | −122.21 (9.76) |
| all-roles YIMBY | all eight | 13 | −139.15 (22.52) |

**The route strata alone predict a held-out body better than any covariate set
on all four outcomes.** On public YIMBY — the outcome carrying the one
near-result in §3a — adding renter share and income costs 14 ELPD and adding
all eight costs 144. On all-roles YIMBY the three covariate sets are within
1.5 ELPD of each other and all worse than route-only, so there the ranking is
noise in the same way §3d's was, but **no covariate set ever wins.** The review
ran the same comparison with a different fold assignment and got the same
ordering at different magnitudes (route-only −116.8 against renter+income
−135.5, +density −174.5, all-eight −158.6 on public cells); the *ordering* is
the robust part, the absolute ELPDs are not.

This is the strongest single statement in the document and it is a negative
one: **out of sample, by body, these ACS covariates do not help.** It is also
consistent with §3a and §3b-prior rather than in tension with them.

### 3d-perm. Body-level permutation nulls — Freedman–Lane

Added 2026-10-19 (review finding 10). `analysis/renter_share5_robust.py` →
`output/renter_share_5/robust/permutation_fl.csv`. 5,000 permutations per row,
two-sided p = (1 + #{|β*| ≥ |β|}) / (1 + 5000).
`plan/RENTER-SHARE-4-SUBSETS.md` §4a permutes `renter_share` *alone* while
seven collinear covariates stay fixed, which is not a valid null for a partial
coefficient. Two valid versions are run instead:

* **whole-vector** — the entire body-level covariate vector is permuted across
  bodies, so the collinearity structure among the eight is preserved exactly
  and only the covariate-to-outcome link is broken;
* **Freedman–Lane** — the IRLS working response is regressed on the design
  *without* the target term, the weighted residuals are resampled in body
  blocks, and the full design is refitted.

| outcome | design | term | β | p whole-vector | p Freedman–Lane |
|---|---|---|---|---|---|
| public NIMBY | (a) base | `renter_share` | −0.271 | 0.233 | 0.192 |
| public NIMBY | (c) all eight | `renter_share` | −0.141 | 0.568 | 0.431 |
| public NIMBY | (c) all eight | **`transit_commute`** | −0.883 | **0.017** | **0.0038** |
| public YIMBY | (a) base | `renter_share` | +0.344 | **0.046** | 0.095 |
| public YIMBY | (c) all eight | `renter_share` | +0.151 | 0.509 | 0.354 |
| public YIMBY | (c) all eight | **`transit_commute`** | +0.895 | **0.017** | **0.0016** |
| all-roles NIMBY | (a) base | `renter_share` | −0.125 | 0.596 | 0.331 |
| all-roles NIMBY | (c) all eight | `renter_share` | −0.108 | 0.638 | 0.487 |
| all-roles NIMBY | (c) all eight | **`transit_commute`** | −0.569 | **0.037** | **0.024** |
| all-roles YIMBY | (a) base | `renter_share` | +0.101 | 0.201 | 0.262 |
| all-roles YIMBY | (c) all eight | `renter_share` | +0.084 | 0.461 | 0.263 |
| all-roles YIMBY | (c) all eight | **`transit_commute`** | +0.325 | **0.025** | **0.0038** |

**Renter share has one p below 0.05 in twelve** — public YIMBY in the
two-covariate model, p = 0.046 / 0.095, exactly the fit §3a-lobo shows loses 72
% of its magnitude on Santa Clara, and it is gone the moment the other six
covariates are in (p = 0.51 / 0.35). **`transit_commute` is below 0.05 in all
eight of its rows** and below 0.01 in three. The review's body-level null gave
p = 0.052 for public YIMBY renter share and 0.244 for public NIMBY,
single-test; these reproduce that for renter share and extend it.

Uncorrected, and there are 12 rows here on 2 terms × 4 outcomes × 2 designs:
Bonferroni at 12 leaves `transit_commute` on public YIMBY (0.0016 × 12 = 0.02)
and nothing else.

## 4. What survives regularisation — plain language

**Rewritten 2026-10-19** on the review's "before quoting" list. The original
three paragraphs are kept below under §4-orig, struck through where they are
wrong.

Nothing that was asked about survives — but that is a narrower claim than this
document first made, and the reasons are weaker than the ones it gave.

**Renter share.** The strongest renter-share number here is the public YIMBY
slope in the two-covariate model: **+0.287 log-odds per sd** (about +0.31 per
ten points of renter share), 90 % +0.026 to +0.529, 95 % −0.029 to +0.580. It
fails four separate ways. It crosses zero at 95 % as published. It crosses zero
at 90 % the moment density is added. Its HC3 interval crosses zero
(−0.068, +0.577). And — added 2026-10-19 — **it falls to +0.096 when Santa
Clara alone is dropped** (§3a-lobo), it is p = 0.046 / 0.095 against a valid
body-level null and p = 0.51 / 0.35 once the other six covariates are in
(§3d-perm), and the covariate set it lives in **predicts a held-out body
14 ELPD worse than the route strata alone** (§3d-kfold). "A body's renter share
does not predict what is said at its meetings in any direction this design can
resolve" is safe. **"And nor does anything else" is not** — §3a-full lists
`log_population` clearing 95 % and `transit_commute` surviving every null.

**Median income** behaves the same way and is weaker still: 90 % intervals
excluding zero only on the all-roles YIMBY share, both gone under
regularisation, and it is in the design by force (VIF 188 → 1.74 after its four
collinear partners are dropped), so **β_income is *residual* income** — income
net of education, home value and gross rent — and not "income" in the sense
anyone means when they ask the question (review finding 11). The same footnote
applies to renter share: `renter_share ~ multiunit_5plus` is r = +0.900, so
**β_renter is renter share net of multi-unit housing stock.**

**The horseshoe is not a measurement of absence.** ~~τ settles at 0.03–0.10 …
reporting that there is no coefficient here large enough to be worth keeping~~
— τ₀ is **fixed at 0.124 by construction**, so τ = 0.03–0.10 is the prior
bulk, a permuted null reproduces τ = 0.037 and β_renter = −0.006, and τ₀ / 5
flattens all eight coefficients to ≤ 0.015 while τ₀ × 5 doubles
`transit_commute` (§3b-prior). The correct wording is **"indistinguishable from
the prior at this n"**.

**`log_population` and `transit_commute`, properly stated.** `log_population`
on all-roles YIMBY is +0.266, **95 % +0.002 to +0.492** — the one coefficient
in this document that clears 95 %, which §3a wrongly said nothing does. It is
**not separable from corpus composition at n = 25**, which is weaker than the
"corpus artefact" the original text asserted: corr(`log_population`, `all_n`)
over the 35 all-roles cells is only **+0.05**, and the biggest cluster
suppliers are Santa Clara and San Leandro, not the largest places.
`transit_commute` is the covariate the data actually wants — +0.16 under the
horseshoe on both YIMBY outcomes, +0.337 clearing 95 % in run 3, highest
inclusion probability in run 4, and p < 0.05 in all eight permutation rows —
and it is **exactly as ecological as renter share**: "places where more
residents commute by transit have less NIMBY public comment" is not "transit
commuters make less NIMBY comment".

**The methodological conclusion stands, with one correction.** Run 1 said *n is
too small, not "renter share doesn't matter"*, and run 2 is what that looks
like when you spend the covariates anyway. Fourteen candidates collapse to
eight after collinearity, and the eight then collapse — not "to nothing", but
to **one coefficient nobody asked about and one that is indistinguishable from
its own permuted null**. What would change this is more bodies, not more
columns; and §3d-kfold is the sharpest form of that, because out of sample the
columns actively cost accuracy.

### 4-orig. The original §4, for the record

~~Nothing that was asked about. The two-covariate base model puts the public
YIMBY share up +0.287 log-odds per standard deviation of renter share (about
+0.31 per ten points of renter share) with a 90 % interval that just clears
zero, and that is the strongest thing in this document. It does not survive
adding one further covariate (density knocks the interval back across zero),
it does not survive the HC3 cross-check (95 % −0.068 to +0.577), and under the
horseshoe it shrinks by a factor of twelve to +0.023. Median household income
behaves identically: its only 90 % intervals excluding zero are on the
all-roles YIMBY share, where more income means slightly less pro-housing
speech, and both vanish under regularisation. The global shrinkage parameter
τ settles at 0.03–0.10 on all four outcomes, which is the model's way of
reporting that there is no coefficient here large enough to be worth keeping
against the prior that most of them are zero.~~

~~The single coefficient that does survive — log population on the all-roles
YIMBY share, +0.266 with a 90 % of +0.026 to +0.454 — is almost certainly
about the corpus rather than about politics. The all-roles outcomes are
dominated by dais clusters (1,952 of 2,801), and the bodies contributing the
most dais clusters are the largest ones: San Jose, Santa Clara on three
routes, San Leandro, Petaluma on Swagit. "Bigger jurisdictions produce more
supportive dais speech" and "bigger jurisdictions produced most of the dais
speech we have" are not separable at n = 25. It is in the table because the
horseshoe put it there, not because it should be believed.~~

~~The useful conclusion is methodological: run 1 said the answer was *n is too
small, not "renter share doesn't matter"*, and run 2 is what that looks like
when you spend the covariates anyway. Fourteen candidate covariates collapse
to eight after collinearity, eight collapse to nothing after regularisation,
and the only thing the extra columns bought was a demonstration that the
near-result in the two-covariate model was not robust to any of them. What
would change this is more bodies — not more columns.~~

## 5. Limitations

1. **n.** 22 bodies and 29 public cells; 25 bodies and 35 all-roles cells.
   Covariates vary at the body level, so the effective n is 22 / 25 against
   p ≈ 13 fixed effects in model (c). Two cells are 23 % and 21 % of the
   public base. No amount of modelling fixes this.
2. **Ecological inference.** Unchanged and central. `renter_share` is the
   renter share of a jurisdiction's **residents**; the outcome is what the
   people who came to the meeting and spoke said. Nothing in this corpus links
   the two, and homeowners are known to over-attend local meetings. A null
   here is fully consistent with a real resident-level relationship that
   meeting attendance filters out — and with the opposite. Every covariate in
   §1 inherits the same gap: `ba_plus`, `median_age`, `nhwhite_share` and the
   rest describe residents, not speakers.
3. **The stance instrument is exploratory.** κ ≥ 0.6 is not passed, no human
   has coded any of these 13,123 turns (`plan/STANCE-S2.md` §0), and the
   YIMBY/NIMBY mapping flips sign on items that are themselves restrictions —
   still 133 of San Leandro's clusters.
4. **The role is an audit, not a gold label.** The public bucket is
   `role_audit` (ra-1); SELF_ID is model output no human has checked, and the
   audit re-partitions in both directions (356 cue-dais clusters in, 327
   cue-public out at S2). Outcomes 1–2 are shares *of the public bucket* and
   move with it; outcomes 3–4 pool both roles and so are insensitive to the
   role column, which is one reason they were included.
5. **Routes are instruments, not variables.** Route is a stratum, never
   pooled. Napa contributes three cells, Santa Clara three, Petaluma /
   Richmond / San Leandro two each, and they disagree (Santa Clara public
   NIMBY share 0.14 agenda_index, 0.29 classifier, 0.56 fullasr).
6. **Median covariates over county unions are weighted means, not medians**
   (`medians_are_weighted_means`). Only bart, which enters no public-stance
   fit. BART's three-county union still excludes San Mateo, which BART serves.
7. **ACS vintage and top-coding.** 2019–2023 5-year estimates (a 2021
   midpoint) against meetings in a 12-month window ending 2026. MOEs are in
   `acs.json` for B25003/B19013 and are **not** carried into the new tables or
   propagated into any model. `med_home_value`, `med_gross_rent` and
   `med_hh_income` are top-coded for up to three low-renter-share bodies
   (§1) — censored, not measured.
8. **Rate outcome not modelled.** Complete coverage is 3 bodies
   (`plan/VALIDATION-2026-10-18.md` "Validation applied" 1). The §2 table is
   descriptive; two of three bodies are transit boards.
9. **LOO is not decision-grade here** (§3d), in either model family —
   **replaced 2026-10-19** by a grouped 5-fold by body (§3d-kfold), which is
   decision-grade and says the covariates lose.
10. ~~**Multiplicity.** Four outcomes × three specifications = twelve fits,
    uncorrected. Nothing needs correcting, because nothing clears a 95 %
    interval and the three 90 % exclusions are exactly what twelve fits of a
    null produce.~~ **Restated 2026-10-19** (review finding 8;
    `output/renter_share_5/robust/multiplicity.csv`). Two of the three facts in
    that sentence were wrong: there are **five** 90 % exclusions, not three,
    and **one does clear 95 %**.

    | run | fits | substantive intervals | excl. 0 at 90 % | expected at 90 % | excl. 0 at 95 % | expected at 95 % |
    |---|---|---|---|---|---|---|
    | **run 2** (this) | 12 | **52** | **5** | 5.2 | **1** | 2.6 |
    | run 3 | 6 | **64** | **16** | 6.4 | **11** | 3.2 |
    | **both** | 18 | **116** | **21** | **11.6** | **12** | **5.8** |

    "Substantive" = every covariate, role and interaction coefficient; α,
    σ_cell/σ_body/σ_row and the route strata are nuisance terms and are
    excluded. Expected counts are the two-sided interval's own error rate
    (10 % / 5 %) times the number of intervals, i.e. the count under a global
    null where every coefficient is zero. **Run 2 is at the null rate on its
    own: 5 against 5.2 expected, 1 against 2.6.** The whole excess is run 3,
    and 16 of its 21 90 % exclusions are **four parameters repeated across
    nested fits** (`delta_role`, `transit_commute`, `log_population`,
    `nhwhite_share`), so neither 116 nor 1 is the effective number of tests.
    The conclusion "nothing needs correcting" survives for run 2 — but it now
    rests on the right arithmetic, and it is **not** true that nothing clears
    95 % here.
11. **Nothing is meeting-clustered** (added 2026-10-19, review finding 4;
    `output/renter_share_5/robust/icc.csv`). The binomial denominator treats
    each speaker cluster as an independent trial. It is not: **86.3 % of public
    stance-bearing clusters rest on one scored turn**, cluster length spans
    64–702 words p10–p90, and the NIMBY indicator's ICC is **0.275 by episode**
    (3.82 clusters/episode) and **0.156 by meeting** (9.60), giving design
    effects of 1.77 and 2.34 — **SEs inflated ×1.33 to ×1.53**. `u_i` is a mean
    shift per cell and does not absorb this. **Every interval in this document
    is therefore about 1.4× too narrow**, and worst on the two cells (San José
    public 191, San Leandro public 173) that are 23 % and 21 % of the public
    base. Widened by ×1.4, `log_population`'s 95 % interval no longer excludes
    zero and neither 90 % renter-share exclusion survives. A turn-level model
    nesting episode in meeting in body, or a beta-binomial, is the documented
    next step and is **not** run here.
11. **Body type dropped as a covariate** (§1): 25 councils, 1 county board on
    three routes, 1 transit board. It is a Napa-and-VTA indicator, not a type
    effect.

## Corrections 2026-10-19

Applied from `plan/REGRESSION-REVIEW-2026-10-19.md`. **Every prior number is
retained above with a strikethrough or a "superseded" note; nothing was
silently replaced.** No ACS table was re-pulled, no covariate re-screened;
`output/renter_share_2/` stays read-only and the new artefacts are all under
`output/renter_share_5/`.

| # | what changed | where | why |
|---|---|---|---|
| 1 | "Nothing survives regularisation" → the renter-share null stands, `transit_commute` is what the data wants | §0, §4 | §3a-full, §3b-prior, §3d-perm and run 3 all point at `transit_commute`; the original claim over-generalised a true statement about renter share. |
| 2 | "no interval excludes zero at 95 %" struck — `log_population` is **+0.266, 95 % +0.002, +0.492** | §3a | False against this document's own `model_table.csv`. |
| 3 | "three 90 % exclusions" → **five** | §3a, §5.10 | The lead sentence contradicted the table printed directly underneath it. |
| 4 | **every parameter now printed** at 90 % and 95 % — all 52 | new §3a-full | Only renter share and income were tabulated; six other covariates were in the artefact and never shown. |
| 5 | **leave-one-body-out column** on every renter-share and transit estimate | new §3a-lobo | Santa Clara alone takes the public YIMBY slope +0.344 → +0.096 and flips the sign of both public slopes in the all-eight fit. |
| 6 | **τ₀ = 0.124 stated**, a permuted-covariate null added, τ₀ × 5 / ÷ 5 sensitivity added; "the posterior saying there is nothing here" → **"indistinguishable from the prior at this n"** everywhere | §0, new §3b-prior, §3b, §4 | τ₀ is fixed by construction, so τ = 0.03–0.10 is the prior, not a finding. Under a true null τ = 0.037 and β_renter = −0.006 — the published picture. |
| 7 | PPC counts "29/29", "35/35" struck; marginal band and PIT with `u_i` **integrated out** reported | `plan/RENTER-SHARE-3-ROLE.md` §3c-cal (all ten fits in one table) | One `u_i` per cell centres the conditional band on the observed count; the check had no power. |
| 8 | PSIS-LOO and the no-`u_i` LOO refit **withdrawn from quotation**; replaced by a grouped 5-fold by body | §3d, new §3d-kfold | Holding out whole bodies is valid for both model families. Route strata alone beat every covariate set on all four outcomes; "the horseshoe ranks first" is dropped. |
| 9 | run 4's single-covariate permutation replaced by **Freedman–Lane and whole-vector body permutations** | new §3d-perm | Permuting `renter_share` alone while seven collinear covariates stay fixed is not a null for a partial coefficient. |
| 10 | "a corpus artefact" → **"not separable from corpus composition at n = 25"** | §0, §4 | corr(`log_population`, `all_n`) = +0.05 across the 35 all-roles cells, and the biggest suppliers are Santa Clara and San Leandro, not the largest places. The dismissal was asserted, not checked. |
| 11 | β_renter and β_income relabelled as **renter share net of multi-unit stock** and **residual income** | §4 | `renter_share ~ multiunit_5plus` r = +0.900; `log_income` is forced in after its four collinear partners go (VIF 188 → 1.74). Both estimands were mislabelled. |
| 12 | **episode and meeting ICC, design effects** added; intervals declared ~1.4× too narrow | new §5.11 | ICC 0.275 by episode, 0.156 by meeting; `u_i` is a mean shift and does not absorb nesting. |

**What did not change.** The headline: on this corpus a body's renter share
does not predict what is said at its meetings in any direction the data can
resolve, after income, density, housing age, racial composition, transit
commuting, vacancy and population size. Every correction above makes that
*better* supported and the surrounding claims *worse* supported. One
unstated-at-the-time fact is also worth recording: one `u_i` per row saturates
the likelihood, so every coefficient in §3a rests on `σ_cell ~
HalfNormal(0.75)`; the review tested that on run 3's δ (2.094 at HN(0.25),
2.110 at HN(2.0)) and found it benign, but it was never stated.

## 6. Files

New:
- `plan/RENTER-SHARE-2.md` (this)
- 2026-10-19 corrections: `analysis/renter_share5_robust.py`,
  `analysis/renter_share5_ppc_horseshoe.py`,
  `analysis/renter_share5_role_restated.py`, and
  `output/renter_share_5/{robust,ppc_horseshoe,role_restated}/` plus logs
- `analysis/renter_share2_acs.py`, `analysis/renter_share2_outcomes.py`,
  `analysis/renter_share2_models.py`
- `output/renter_share_2/` — `raw/dl.sh`, `raw/dl.log`, `raw/acsdt5y2023-*.dat`
  (10 new tables), `acs_wide.json`, `covariates_wide.csv`, `acs_wide.log`,
  `outcomes_clusters.csv`, `rates_complete.csv`, `covariate_corr.csv`,
  `covariate_vif.csv`, `screen.json`, `outcomes.log`, `models.log`, and
  `models/` (`model_results.json`, `model_table.csv`, `loo_table.csv`,
  `ols_hc3.csv`)

Modified: nothing. `analysis/renter_share_acs.py` is **imported, not edited**
(`BODY_GEO`, `SUMLEVEL`), so the two covariate tables share one geography
rule. No figures were produced. No DB was written, no episode or stance row
touched, nothing committed in `~/ytpt`. `.env`, `plan/frame.csv`,
`plan/DECISIONS.md`, `plan/STANCE-CODEBOOK.md` and `plan/RENTER-SHARE.md`
untouched.
