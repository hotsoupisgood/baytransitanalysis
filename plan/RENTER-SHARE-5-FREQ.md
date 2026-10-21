# RENTER-SHARE-5-FREQ — the frequentist counterpart (2026-10-19)

Puck asked for a frequentist fit of the run-2/run-3 renter-share and role
regressions. This is it: `analysis/renter_share5_freq.py`, artefacts in
`output/renter_share_5/freq/`, no PyMC, no network, no spend, nothing
committed. It is built to the standard set by
`plan/REGRESSION-REVIEW-2026-10-19.md` findings 2, 3, 4, 6, 7 and 10, and it
does **not** restate or edit the Bayesian docs.

**Environment.** R is not installed on this Mac (`which Rscript` → not found),
so there is no `lme4::glmer` / `glmmTMB` fit. `statsmodels`
(0.15.0, installed into `.venv` for this run) has no frequentist binomial
GLMM — `BinomialBayesMixedGLM` is a variational *Bayes* fit, so it is not used.
The GLMM below is therefore a **custom marginal-maximum-likelihood** binomial
GLMM: the body random intercept is integrated out by 24-node Gauss–Hermite
quadrature and the whole thing maximised by L-BFGS-B/Nelder-Mead, with Wald
CIs from a numerical Hessian, profile-likelihood CIs for role, and LRTs.

## 1. Unit and the two denominators

| | rows | bodies | trials | NIMBY |
|---|---|---|---|---|
| **Primary** — NIMBY / (NIMBY+YIMBY), `neutral` dropped, staff clusters dropped | 56 | 24 | 1,300 | 603 |
| Secondary — published denominator (clusters with a stance, staff kept) | 64 | 25 | 2,790 | 619 |

The secondary rows reproduce run 3's artefact exactly (64 rows, 25 bodies,
2,790 clusters), which is the build check. Staff are identified as clusters
whose turns' modal non-`unknown` `turn_stance.self_id` is `staff`: 844 of the
3,701 role clusters, **843 of them dais** — the review's count. Design matched
to run 3: eight kept covariates standardised over the fitted rows, role
(dais = 0, public = 1), role × z_renter, route dummies with `classifier` as the
reference stratum, floor 5 trials per row.

## 2. Comparison table — primary (directional) denominator

Log-odds, 95 % intervals. GEE = population-averaged, exchangeable working
correlation clustered by body, robust sandwich. Bayesian column is run 3 R1
(`output/renter_share_3/models/model_table.csv`), which is on the **secondary**
denominator — it is shown for reference and repeated properly in §3.

| parameter | GEE (exch.) | GLMM (body RI) | beta-binomial | Bayesian R1 *(sec. denom.)* |
|---|---|---|---|---|
| **role** (public−dais) | **+0.741 (+0.349, +1.132)** | **+0.732 (+0.482, +0.981)** | **+0.740 (+0.403, +1.077)** | +2.108 (+1.710, +2.523) |
| renter_share | −0.106 (−0.424, +0.211) | −0.088 (−0.327, +0.151) | −0.122 (−0.434, +0.190) | −0.019 (−0.402, +0.387) |
| role × renter | −0.161 (−0.651, +0.328) | −0.159 (−0.419, +0.100) | −0.054 (−0.396, +0.289) | −0.007 (−0.376, +0.368) |
| **transit_commute** | **−0.641 (−1.004, −0.279)** | **−0.616 (−0.915, −0.317)** | **−0.511 (−0.884, −0.138)** | −0.337 (−0.792, +0.125) |
| log_population | +0.122 (−0.097, +0.341) | +0.082 (−0.145, +0.309) | −0.039 (−0.355, +0.276) | −0.074 (−0.500, +0.289) |
| log_income | +0.117 (−0.267, +0.502) | +0.118 (−0.124, +0.361) | +0.076 (−0.213, +0.364) | −0.009 (−0.371, +0.340) |

`log_density` is the fourth 95 % survivor on the primary rows: GEE +0.346
(+0.062, +0.630), GLMM +0.309 (+0.030, +0.589), beta-binomial +0.194 (−0.165,
+0.552).

**Dispersion / correlation.** GEE exchangeable within-body working correlation
α = −0.0024 (primary), +0.0095 (secondary) — essentially nil at cluster level,
which is why the independence check is almost identical (role +0.732, renter
−0.088). GLMM σ_body = **0.000, on the boundary** (primary; profile
log-likelihood is monotone decreasing in σ: −129.11 at 0, −129.35 at 0.2,
−131.28 at 0.45; boundary LRT p = 0.50). Beta-binomial φ = 33.6, **ρ = 0.029**.
Role profile-likelihood CI (primary): **(+0.484, +0.982)**, against the Wald
(+0.482, +0.981).

## 3. Same table, secondary (published) denominator

| parameter | GEE (exch.) | GLMM (body RI) | beta-binomial | Bayesian R1 |
|---|---|---|---|---|
| **role** | **+2.040 (+1.734, +2.345)** | **+2.029 (+1.810, +2.249)** | **+2.058 (+1.728, +2.387)** | **+2.108 (+1.710, +2.523)** |
| renter_share | −0.035 (−0.331, +0.262) | +0.029 (−0.232, +0.289) | −0.109 (−0.396, +0.178) | −0.019 (−0.402, +0.387) |
| role × renter | −0.102 (−0.534, +0.330) | −0.116 (−0.334, +0.101) | −0.006 (−0.325, +0.313) | −0.007 (−0.376, +0.368) |
| **transit_commute** | **−0.475 (−0.840, −0.110)** | **−0.394 (−0.749, −0.038)** | **−0.436 (−0.791, −0.080)** | −0.337 (−0.792, +0.125) |
| log_population | +0.119 (−0.197, +0.436) | +0.024 (−0.278, +0.326) | +0.108 (−0.186, +0.402) | −0.074 (−0.500, +0.289) |
| log_income | +0.064 (−0.267, +0.395) | +0.030 (−0.232, +0.291) | +0.041 (−0.217, +0.299) | −0.009 (−0.371, +0.340) |

Here σ_body = **0.305** (boundary LRT 6.09, p = 0.007), against the Bayesian
σ_body posterior median 0.331 — the two agree. Beta-binomial φ = 41.7,
ρ = 0.023. Role profile CI (+1.812, +2.251).

## 4. Likelihood-ratio tests for role (GLMM)

| test | primary | secondary |
|---|---|---|
| role + role×renter dropped (df 2) | χ² = 32.27, **p = 9.8e-08** | χ² = 356.41, p = 4.0e-78 |
| role dropped, no interaction in either model (df 1) | χ² = 30.81, **p = 2.8e-08** | χ² = 355.30, p = 3.0e-79 |
| role×renter dropped (df 1) | χ² = 1.46, p = 0.227 | χ² = 1.10, p = 0.293 |

Role is decisive on both denominators; the interaction is not detectable on
either.

## 5. Nested structure (finding 4) — turn-level logistic

NIMBY turn (`stance_median` ≤ −0.5) vs YIMBY turn (≥ +0.5), staff turns
dropped, 1,892 turns / 856 NIMBY across 502 episodes, 139 meetings, 26 bodies.
Point estimates are identical by construction; only the SE changes.

| clustering | groups | role est | role SE | role p | renter est | renter SE | renter p |
|---|---|---|---|---|---|---|---|
| none (iid) | 1,892 | +0.583 | 0.097 | 1.8e-09 | −0.158 | 0.051 | 0.002 |
| episode | 502 | +0.583 | **0.146** | 6.4e-05 | −0.158 | 0.073 | 0.030 |
| meeting | 139 | +0.583 | **0.171** | 0.0006 | −0.158 | 0.092 | 0.086 |
| body | 26 | +0.583 | **0.254** | 0.022 | −0.158 | 0.065 | 0.015 |

The role SE grows **×1.5 at episode, ×1.8 at meeting, ×2.6 at body** — the
review's design-effect 1.8–2.3 is if anything conservative, and an iid turn
model overstates precision by about a factor of two and a half. Renter share
grows ×1.4 then ×1.8 and then *falls* at body level (0.065): with 26 groups and
a covariate that varies only between bodies, the by-body sandwich is estimated
from 26 scores and is not reliable in that direction. Treat 0.092 (meeting) as
the honest SE for renter share, and the ×2.6 body figure for role.

## 6. Robustness on the primary GEE

**Leave-one-body-out** (`lobo_primary.csv`, 24 refits). Role is stable:
+0.60 (drop San Jose) to +0.94 (drop Santa Clara), full-sample +0.741 — no body
decides it. Renter share is not: full −0.106, range **+0.038** (drop Napa) to
**−0.397** (drop Santa Clara), and dropping Santa Clara **flips the
interaction** from −0.161 to +0.404. Finding 6 reproduces on the directional
denominator and in the frequentist fit.

**Grouped 5-fold by body**, out-of-sample binomial log-loss per trial
(lower is better):

| model set | log-loss | total held-out log density |
|---|---|---|
| intercept only | **0.698** | −907.0 |
| route only | 1.149 | −1,493.4 |
| + role | 1.134 | −1,474.6 |
| + role + renter + role×renter | 1.144 | −1,487.8 |
| + full (8 covariates) | 1.167 | −1,516.8 |

Every covariate model is **worse than an intercept** on held-out bodies, and
the ordering inside the covariate models is the review's: adding covariates
makes out-of-sample prediction worse. Two caveats, both real: route is close to
nested in body (Swagit is Petaluma, `fullasr`/`local_asr` are a handful of
bodies), so 78 of 1,300 held-out trials sit on a route the training fold never
saw and are predicted at the reference stratum; and the fold fits are binomial
GLMs (the working correlation changes SEs, not the fitted mean).

**Freedman–Lane permutation for renter share**, 5,000 draws: z-residualised
renter share permuted across bodies with the other seven covariates and the
design held fixed. Observed |z| = 0.544, **p = 0.678**; the permutation
distribution's 95th percentile is |z| = 2.44, i.e. the null is well calibrated
and the observed statistic is nowhere near it.

## 7. Where frequentist and Bayesian agree, and where they do not

They agree on everything that is identified by the data rather than by the
prior. On the published denominator the role gap is +2.04 (GEE), +2.03 (GLMM),
+2.06 (beta-binomial) against the Bayesian +2.11, and the GLMM's body-intercept
SD of 0.305 lands on the posterior median of 0.331; renter share and the
role × renter interaction are indistinguishable from zero in all four fits, and
all four put transit commute negative and log population and residual income
nowhere. So the null on renter share is not an artefact of PyMC, of the priors
or of the horseshoe — three different frequentist likelihoods, two of them with
no shrinkage at all, return the same flat slope, and a properly-constructed
permutation null returns p = 0.68. The disagreements are of two kinds. The
first is the denominator, not the inference: moving to NIMBY-out-of-directional
and dropping the 843 staff clusters cuts the role gap from +2.11 to +0.74 log
odds — OR 2.1 instead of 8.2 — exactly the review's finding 2, and it is the
*only* substantive movement anywhere in this document. The second is width.
The frequentist intervals are **narrower** than the Bayesian ones — uniformly
so for the GLMM, and for the GEE everywhere except residual income
(role 0.78 GEE / 0.50 GLMM wide vs 0.81; transit commute 0.73 / 0.60 vs 0.92;
log income 0.77 / 0.49 vs 0.71), because run 3 carries a per-row overdispersion term `u_i` that these
fits do not, and the frequentist analogue of that term is explicitly small:
the GEE's within-body working correlation is ≈ 0, the beta-binomial's ρ is
0.023–0.029, and on the directional denominator the GLMM's σ_body is **zero on
the boundary**. Run 3's `u_i` is therefore doing more work than the data asks
for at the row level — which is the same degeneracy finding 3 names in the PPC,
seen from the other side — while *under*-stating the clustering that does
exist, at episode and meeting level inside a row (§5). The net is that
frequentist and Bayesian intervals here are both about the right width for the
wrong reasons, and the one number that moves materially is δ, for a reason that
has nothing to do with which school fitted it. Two things survive at 95 % in
every fit on the primary rows: **role** and **transit commute** (negative — more
transit commuting, less NIMBY), with log density third and sign-stable but not
robust to the beta-binomial. Nothing here rescues renter share.

## 8. Files

- `analysis/renter_share5_freq.py` — the whole run
- `output/renter_share_5/freq.log` — run log (138 s)
- `output/renter_share_5/freq/freq_results.json` — every estimate, SE, CI, LRT, profile, fold and permutation
- `output/renter_share_5/freq/comparison.csv` — §2/§3 flat
- `output/renter_share_5/freq/lobo_primary.csv` — §6 leave-one-body-out
- `output/renter_share_5/freq/rows_primary.csv` — the 56 primary rows

## 9. Limitations

Unchanged from runs 2–3 and still the binding ones: ecological inference (ACS
covariates are body-level, the outcome is cluster-level), `role_audit` is a
model output and not ground truth, no gold set this phase, the κ gate is
unpassed, and route is a stratum that is close to a body label. Added here: the
GLMM is a custom quadrature fit, not `glmer` — it reproduces the Bayesian
σ_body on the secondary rows, which is the only external check available on
this machine; and the by-body sandwich on 24–26 groups is at the small-cluster
limit where the robust SE is itself noisy.
