# RENTER-SHARE-3-ROLE — role as a covariate, and role × renter share

**EXPLORATORY — the stance κ ≥ 0.6 hand-coding gate (`plan/SHOWCASE.md` 4a)
is still not passed.** Run 2026-10-19 by the supervisor agent on Puck's
question: *is dais/public fitted as a covariate in the renter-share
regression?* **It is not.** `plan/RENTER-SHARE-2.md` fits a **public-only**
outcome (`pub_nimby/pub_n`) and an **all-roles** outcome (`all_nimby/all_n`),
so role is a *subsetting* variable there — it selects the denominator and is
never in the design matrix, and the renter-share slope is therefore estimated
once per outcome family with no way to ask whether it differs by who is
speaking. This run builds the fit that answers it.

Offline from `output/renter_share_2/` (run 2's ACS pull and VIF screen, re-used
unchanged, nothing re-pulled) and `output/stance_s1/nimby_view/`. **No YouTube
request, no GPU, no Claude call, no API key, $0.** No DB write. `.env`,
`plan/frame.csv`, `plan/DECISIONS.md` untouched. Nothing committed. No figures.

Companions: `plan/RENTER-SHARE.md` (run 1, turn unit),
`plan/RENTER-SHARE-2.md` (run 2 — covariates, screen, the four outcomes),
`plan/ROLE-AUDIT-2026-10-18.md` (rule ra-1, the role column),
`plan/STANCE-S2.md` (the stance table).

## 0. Plain answer

> **Superseded 2026-10-19 by `plan/REGRESSION-REVIEW-2026-10-19.md` finding 2.
> The old headline is kept below with a strikethrough. See §0a and
> "Corrections 2026-10-19".**

**No — run 2 never fitted role as a covariate; and when you do, role is the
largest effect in the whole renter-share exercise, while renter share stays
null for both roles.** But the *size* of that effect as first quoted was an
artefact of the denominator. ~~Stacking the two role rows of every body ×
route cell and putting role in the design matrix gives a public-vs-dais NIMBY
gap of **+2.11 log-odds** (90 % +1.79 to +2.47, 95 % +1.73 to +2.53,
Pr > 0 = 1.000) — roughly an 8× odds ratio, public comment vs the dais, after
the eight kept covariates, route stratum and a body random intercept.~~ That
outcome is NIMBY out of *all clusters with a stance*, and ~60 % of that
denominator is the `neutral` class, which is a dais phenomenon: 69.1 % of
stance-bearing dais clusters are `neutral` against 10.5 % of public ones.
A further ~20 % is staff clusters, which ra-1 folds into the dais and which are
94 % `neutral` (5 of 667 NIMBY). So +2.11 is mostly *the dais says neutral,
procedural things*, not *the public is eight times more NIMBY*.

**On the directional denominator — NIMBY out of NIMBY + YIMBY, neutral
excluded — the gap is +0.84 log-odds, an odds ratio of 2.3; with staff
clusters dropped it is +0.78, odds ratio 2.2** (§0a). The ~~+1.03~~ YIMBY gap
is not a separate finding on this denominator — it is the same contrast with
the sign reversed, and `neutral` inflated it the same way. Raw rates, which is
what is really speaking: NIMBY share of *directional* clusters 34.4 % on the
dais (205/596) against 54.1 % in public comment (404/747), where the old
figures were 10.6 % (207/1,947) against 48.9 % (412/843).

### 0a. The role gap, restated

`analysis/renter_share5_role_restated.py` →
`output/renter_share_5/role_restated/`. Same model, priors, covariates, route
stratum, body random intercept and row overdispersion as §3; only the outcome
and the row filter change. Floor is 5 **directional** clusters per row.
`δ` is still public(1) − dais(0) in log-odds of NIMBY.

| # | outcome | rows / bodies | raw NIMBY share, dais → public | **δ** | 90 % | 95 % | **odds ratio** | OR 95 % |
|---|---|---|---|---|---|---|---|---|
| — | ~~NIMBY / all stance-bearing (published §3a)~~ | ~~64 / 25~~ | ~~0.106 → 0.489~~ | ~~+2.112~~ | ~~+1.788, +2.465~~ | ~~+1.726, +2.533~~ | ~~8.3~~ | ~~5.6, 12.6~~ |
| **(a)** | **NIMBY / (NIMBY+YIMBY), neutral excluded** | **57 / 24** | **0.344 → 0.541** | **+0.839** | **+0.499, +1.201** | **+0.437, +1.275** | **2.31** | **1.55, 3.58** |
| **(b)** | **(a), staff clusters excluded** | **56 / 24** | **0.357 → 0.541** | **+0.784** | **+0.436, +1.150** | **+0.357, +1.231** | **2.19** | **1.43, 3.42** |
| (c) | (a), `self_id` elected (0) vs resident (1) | 46 / 22 | 0.311 → 0.671 | +1.700 | +1.207, +2.236 | +1.091, +2.354 | 5.47 | 2.98, 10.53 |

**Quote (a) and (b): the role gap on direction is about 2.2–2.3× in odds, not
8×.** All three fits are clean (max R̂ 1.0040, min bulk ESS 2,168, 0–4
divergences out of 8,000). `theta_role_x_renter` is −0.05 / −0.05 / +0.05, all
90 % intervals ±0.4 wide and through zero: the interaction is as dead on the
directional denominator as it was on the old one, and `renter_share` itself is
−0.05 on all three.

Row (c) is the one that complicates the simple story and is **not** the
headline. Restricting to clusters whose `self_id` is `elected` on the dais side
and `resident` on the public side — 648 directional clusters, 22 bodies — the
gap goes *up* to +1.70, OR 5.5. Two readings, and this design cannot separate
them: self-identifying as a resident ("I live at…") is itself something people
do more when they are objecting, so (c) selects on a correlate of the outcome;
and the elected/resident contrast is the cleanest version of the
*different-speech-acts* problem (§5.2), a council member moving an item against
a neighbour objecting to it. The 2.2–2.3× of (a)/(b) is the defensible number
because its denominator is defined by the instrument, not by a self-description.

Cluster-level `self_id` is the majority non-`unknown` `self_id` over the
cluster's turns in `role_audit` (ties: elected > staff > org_rep > business >
homeowner > renter > resident), read read-only from
`~/.local/share/ytpt/state.sqlite`. It is **model output, exactly as unchecked
as the role column itself** (§5.1), so (b) and (c) inherit that limitation
twice over.

**The interaction is not resolvable.** On NIMBY, role × renter share is
**−0.006** (90 % −0.318 to +0.305, Pr > 0 = 0.49) — the renter-share slope is
the same nothing for both roles (dais −0.018, public −0.025). On YIMBY the
interaction is **+0.146** (90 % −0.103 to +0.395, Pr > 0 = 0.83) in the
direction you would guess — renter-heavier places have slightly more
pro-housing *public* comment and no dais shift — but with the dais slope at
+0.035 (90 % −0.197 to +0.254) and the public slope at +0.178 (90 % −0.076 to
+0.428), neither clears zero at 90 %, let alone 95 %. So the interesting
reading of run 2 §0 — that the near-significant renter-share effect lived in
the public bucket — is *consistent* with this fit and *not established* by it.
Income behaves the same way: `log_income` −0.122 on YIMBY (90 % −0.365 to
+0.118) and role × income −0.014 (Pr > 0 = 0.46). n is still 25 bodies.

## 1. Design

`analysis/renter_share3_role_outcomes.py` → `output/renter_share_3/
{outcomes_role_stacked.csv, cells.json}`.

Unit = **body × route × role** row, off the post-S2 `role_audit` (ra-1)
speaker-cluster table `output/stance_s1/nimby_view/
speaker_clusters_roleaudit.csv`. A cluster "has a stance" iff
`label != "no stance"`; `n` = clusters with a stance; numerators are NIMBY and
YIMBY cluster counts. Floor **5 clusters per row** (run 2's floor, on the finer
unit). Route is never pooled — fixed-effect stratum, reference `classifier`.
Covariates and the kept set are read from `output/renter_share_2/
{covariates_wide.csv, screen.json}`; **no covariate was recomputed, re-pulled
or re-screened**, so this run cannot drift from run 2.

| set | rows | bodies | cells | clusters | NIMBY | YIMBY |
|---|---|---|---|---|---|---|
| all stacked rows | **64** | **25** | 35 | 2,790 | 619 | 741 |
| dais rows | 35 | 25 | 35 | 1,947 | 207 (10.6 %) | 395 (20.3 %) |
| public rows | 29 | 22 | 29 | 843 | 412 (48.9 %) | 346 (41.0 %) |
| paired cells (both roles ≥ 5) | 58 | 22 | **29** | 2,636 | 607 | 701 |

11 role rows fall below the floor. Six body × route cells are **dais-only**
(`bart` 23, `classifier milpitas` 6, `classifier vallejo` 20, `fullasr napa`
19, `fullasr hayward` 49, `local_asr hayward` 37) — exactly run 2's six extra
all-roles cells, seen from the other side. **29 cells carry a within-body role
contrast**, and since the body random intercept is shared across a body's two
role rows, those 29 are what identifies role and role × covariate. Every fit is
re-run on the 58 paired rows alone as a sensitivity check.

## 2. Outcomes — the stacked rows

Public then dais per cell, in renter-share order. `d`/`p` = role, `pair` = both
rows clear the floor.

| route | body | renter | role | n | NIMBY | YIMBY | pair |
|---|---|---|---|---|---|---|---|
| classifier | losaltos | 0.192 | d | 70 | 9 | 7 | 1 |
| classifier | losaltos | 0.192 | p | 14 | 7 | 3 | 1 |
| classifier | livermore | 0.280 | d | 19 | 3 | 4 | 1 |
| classifier | livermore | 0.280 | p | 15 | 7 | 7 | 1 |
| classifier | pleasanton | 0.329 | d | 32 | 5 | 5 | 1 |
| classifier | pleasanton | 0.329 | p | 13 | 3 | 8 | 1 |
| agenda_index | napa | 0.356 | d | 8 | 0 | 1 | 1 |
| agenda_index | napa | 0.356 | p | 5 | 1 | 3 | 1 |
| classifier | napa | 0.356 | d | 51 | 1 | 11 | 1 |
| classifier | napa | 0.356 | p | 9 | 4 | 4 | 1 |
| fullasr | napa | 0.356 | d | 19 | 2 | 3 | 0 |
| local_asr | napa | 0.356 | d | 30 | 9 | 12 | 1 |
| local_asr | napa | 0.356 | p | 37 | 23 | 12 | 1 |
| classifier | petaluma | 0.361 | d | 42 | 7 | 5 | 1 |
| classifier | petaluma | 0.361 | p | 16 | 9 | 6 | 1 |
| swagit_transcript | petaluma | 0.361 | d | 173 | 17 | 43 | 1 |
| swagit_transcript | petaluma | 0.361 | p | 31 | 8 | 17 | 1 |
| classifier | concord | 0.386 | d | 15 | 5 | 5 | 1 |
| classifier | concord | 0.386 | p | 22 | 9 | 13 | 1 |
| classifier | dalycity | 0.395 | d | 24 | 0 | 5 | 1 |
| classifier | dalycity | 0.395 | p | 5 | 0 | 5 | 1 |
| classifier | fremont | 0.398 | d | 47 | 3 | 7 | 1 |
| classifier | fremont | 0.398 | p | 6 | 3 | 0 | 1 |
| classifier | milpitas | 0.407 | d | 6 | 1 | 0 | 0 |
| classifier | healdsburg | 0.410 | d | 20 | 4 | 4 | 1 |
| classifier | healdsburg | 0.410 | p | 8 | 6 | 1 | 1 |
| classifier | vallejo | 0.411 | d | 20 | 1 | 5 | 0 |
| fullasr | sanleandro | 0.417 | d | 18 | 2 | 3 | 1 |
| fullasr | sanleandro | 0.417 | p | 6 | 4 | 0 | 1 |
| local_asr | sanleandro | 0.417 | d | 183 | 19 | 28 | 1 |
| local_asr | sanleandro | 0.417 | p | **173** | 84 | 75 | 1 |
| agenda_index | sausalito | 0.421 | d | 41 | 5 | 5 | 1 |
| agenda_index | sausalito | 0.421 | p | 24 | 11 | 9 | 1 |
| agenda_index | hayward | 0.431 | d | 14 | 0 | 1 | 1 |
| agenda_index | hayward | 0.431 | p | 6 | 6 | 0 | 1 |
| fullasr | hayward | 0.431 | d | 49 | 6 | 8 | 0 |
| local_asr | hayward | 0.431 | d | 37 | 2 | 11 | 0 |
| classifier | santarosa | 0.438 | d | 52 | 2 | 10 | 1 |
| classifier | santarosa | 0.438 | p | 29 | 10 | 15 | 1 |
| classifier | sanjose | 0.441 | d | 102 | 17 | 28 | 1 |
| classifier | sanjose | 0.441 | p | **191** | 125 | 59 | 1 |
| classifier | vta | 0.447 | d | 36 | 2 | 17 | 1 |
| classifier | vta | 0.447 | p | 22 | 12 | 6 | 1 |
| classifier | paloalto | 0.458 | d | 71 | 1 | 14 | 1 |
| classifier | paloalto | 0.458 | p | 37 | 19 | 11 | 1 |
| classifier | bart | 0.460 | d | 23 | 0 | 13 | 0 |
| fullasr | richmond | 0.475 | d | 29 | 2 | 8 | 1 |
| fullasr | richmond | 0.475 | p | 10 | 6 | 3 | 1 |
| local_asr | richmond | 0.475 | d | 119 | 9 | 18 | 1 |
| local_asr | richmond | 0.475 | p | 24 | 8 | 13 | 1 |
| classifier | sanrafael | 0.482 | d | 15 | 1 | 2 | 1 |
| classifier | sanrafael | 0.482 | p | 13 | 8 | 4 | 1 |
| classifier | sanmateo | 0.492 | d | 47 | 1 | 8 | 1 |
| classifier | sanmateo | 0.492 | p | 24 | 5 | 13 | 1 |
| classifier | sunnyvale | 0.562 | d | 43 | 2 | 6 | 1 |
| classifier | sunnyvale | 0.562 | p | 9 | 6 | 3 | 1 |
| classifier | oakland | 0.583 | d | 10 | 2 | 3 | 1 |
| classifier | oakland | 0.583 | p | 17 | 3 | 14 | 1 |
| agenda_index | santaclara | 0.588 | d | 98 | 6 | 21 | 1 |
| agenda_index | santaclara | 0.588 | p | 14 | 2 | 9 | 1 |
| classifier | santaclara | 0.588 | d | **257** | 39 | 59 | 1 |
| classifier | santaclara | 0.588 | p | 45 | 13 | 26 | 1 |
| fullasr | santaclara | 0.588 | d | 127 | 22 | 15 | 1 |
| fullasr | santaclara | 0.588 | p | 18 | 10 | 7 | 1 |

The role gap is visible row by row and does not depend on the model: Palo Alto
dais 1/71 NIMBY against public 19/37; San Mateo 1/47 against 5/24; Santa Rosa
2/52 against 10/29. The two biggest public rows (San Jose 191, San Leandro
173) are also the two most NIMBY-heavy public rows, and both sit mid-range on
renter share.

## 3. Model

`analysis/renter_share3_role_models.py`, PyMC 6.3.2 / ArviZ 1.3.0, 4 chains ×
2,000 draws, 1,500 tuning, `target_accept` 0.95, seed 20261019.
`output/renter_share_3/models/`.

    y_i        ~ Binomial(N_i, p_i)
    logit p_i  = α + Σ_j β_j z_ji            # 8 covariates kept by run 2's VIF screen
                 + δ · role_i                 # dais = 0, public = 1
                 + θ_r · role_i · z_renter,i  # the interaction
               [ + θ_y · role_i · z_income,i ]# spec R2 only
                 + γ_route(i)                 # stratum, never pooled
                 + b_body(i)                  # shared by a body's two role rows
                 + u_i                        # row-level overdispersion

    b_body ~ Normal(0, σ_body)   σ_body ~ HalfNormal(0.75)
    u_i    ~ Normal(0, σ_row)    σ_row  ~ HalfNormal(0.75)
    α ~ Normal(0, 1.5)   γ ~ Normal(0, 1)   β, θ ~ Normal(0, 1)
    δ ~ Normal(0, 2)

Covariates: `renter_share`, `log_income`, `log_density`, `built_2010_plus`,
`nhwhite_share`, `transit_commute`, `vacancy_rate`, `log_population` —
**run 2's kept set, unchanged**, standardised over the 64 fitted rows (renter
sd 0.0886, log-income sd 0.281). δ gets a Normal(0, 2) prior because the role
gap is a priori allowed to be large; Normal(0, 1) would shrink a real +2
log-odds gap. Because β_renter multiplies z_renter at role = 0, **β_renter is
the dais slope and β_renter + θ_r is the public slope** — both are also
computed as posterior derived quantities so their intervals are exact rather
than added by hand. Two specs: **R1** = role + role × renter; **R2** = R1 +
role × income. Each outcome also gets R2 re-fitted on the 58 paired rows.

**PSIS-LOO is not reported** and still should not be: `plan/RENTER-SHARE-2.md`
§3d showed it is degenerate with one per-observation overdispersion term, and
this model has two random-effect terms. **But "no out-of-sample comparison is
possible" was wrong** — a grouped 5-fold holding out whole bodies is valid and
is reported at §5.8 (added 2026-10-19).

### 3a. The role terms and the renter/income terms

| outcome | spec | parameter | median | 90 % | 95 % | Pr > 0 |
|---|---|---|---|---|---|---|
| NIMBY | R1 | **role (δ)** | **+2.108** | **+1.783, +2.444** | **+1.710, +2.523** | **1.000** |
| NIMBY | R1 | renter share (dais) | −0.019 | −0.335, +0.306 | −0.402, +0.387 | 0.461 |
| NIMBY | R1 | **role × renter (θ_r)** | −0.007 | −0.317, +0.308 | −0.376, +0.368 | 0.483 |
| NIMBY | R1 | log income | −0.009 | −0.313, +0.275 | −0.371, +0.340 | 0.482 |
| NIMBY | R2 | **role (δ)** | **+2.112** | **+1.788, +2.465** | **+1.726, +2.533** | **1.000** |
| NIMBY | R2 | renter share (dais) | −0.018 | −0.336, +0.322 | −0.402, +0.394 | 0.461 |
| NIMBY | R2 | role × renter (θ_r) | −0.006 | −0.318, +0.305 | −0.382, +0.379 | 0.486 |
| NIMBY | R2 | log income | +0.005 | −0.333, +0.349 | −0.406, +0.419 | 0.512 |
| NIMBY | R2 | role × income (θ_y) | −0.038 | −0.364, +0.288 | −0.431, +0.350 | 0.423 |
| YIMBY | R1 | **role (δ)** | **+1.015** | **+0.763, +1.271** | **+0.708, +1.324** | **1.000** |
| YIMBY | R1 | renter share (dais) | +0.036 | −0.201, +0.257 | −0.252, +0.298 | 0.605 |
| YIMBY | R1 | role × renter (θ_r) | **+0.142** | −0.102, +0.384 | −0.149, +0.434 | **0.827** |
| YIMBY | R1 | log income | −0.131 | −0.336, +0.088 | −0.376, +0.143 | 0.160 |
| YIMBY | R2 | **role (δ)** | **+1.025** | **+0.765, +1.272** | **+0.713, +1.327** | **1.000** |
| YIMBY | R2 | renter share (dais) | +0.035 | −0.197, +0.254 | −0.239, +0.302 | 0.605 |
| YIMBY | R2 | role × renter (θ_r) | **+0.146** | −0.103, +0.395 | −0.161, +0.437 | **0.834** |
| YIMBY | R2 | log income | −0.122 | −0.365, +0.118 | −0.415, +0.167 | 0.201 |
| YIMBY | R2 | role × income (θ_y) | −0.014 | −0.257, +0.247 | −0.309, +0.297 | 0.462 |

~~**Role is the only parameter in this document whose interval excludes zero**,
and it does so by a wide margin on both outcomes. Nothing else clears 90 %.~~

> **Superseded 2026-10-19 — that sentence is false against this run's own
> artefact** (`plan/REGRESSION-REVIEW-2026-10-19.md` finding 1). §3a above
> tabulates only role, renter, income and the interactions; three other
> parameters clear 95 % and two more clear 90 %. The full table is §3a-full.
> Role is the *largest* parameter here, not the only non-zero one.

### 3a-full. Every parameter, every fit

From `output/renter_share_3/models/model_table.csv`, nothing recomputed —
these rows were always in the artefact and were simply not printed. α, σ_body,
σ_row and the route strata are omitted (they are nuisance terms, in the CSV);
every covariate, role and interaction is here. **Bold = 95 %.**

| outcome | spec | parameter | median | 90 % | 95 % | Pr > 0 | excludes 0 |
|---|---|---|---|---|---|---|---|
| NIMBY | R1 | `renter_share` | −0.019 | −0.335, +0.306 | −0.402, +0.387 | 0.461 | — |
| NIMBY | R1 | `log_income` | −0.009 | −0.312, +0.275 | −0.371, +0.340 | 0.482 | — |
| NIMBY | R1 | `log_density` | +0.022 | −0.432, +0.427 | −0.535, +0.515 | 0.536 | — |
| NIMBY | R1 | `built_2010_plus` | −0.030 | −0.425, +0.367 | −0.500, +0.454 | 0.449 | — |
| NIMBY | R1 | `nhwhite_share` | +0.105 | −0.270, +0.478 | −0.343, +0.559 | 0.692 | — |
| NIMBY | R1 | `transit_commute` | −0.337 | −0.712, +0.050 | −0.792, +0.125 | 0.077 | — |
| NIMBY | R1 | `vacancy_rate` | −0.147 | −0.476, +0.168 | −0.543, +0.232 | 0.219 | — |
| NIMBY | R1 | `log_population` | −0.074 | −0.428, +0.232 | −0.500, +0.289 | 0.354 | — |
| NIMBY | R1 | **`delta_role`** | **+2.108** | +1.783, +2.444 | +1.710, +2.523 | 1.000 | **95 %** |
| NIMBY | R1 | `theta_role_x_renter` | −0.007 | −0.317, +0.308 | −0.376, +0.368 | 0.483 | — |
| NIMBY | R2 | `renter_share` | −0.018 | −0.336, +0.323 | −0.402, +0.394 | 0.461 | — |
| NIMBY | R2 | `log_income` | +0.005 | −0.333, +0.349 | −0.406, +0.419 | 0.512 | — |
| NIMBY | R2 | `log_density` | +0.019 | −0.435, +0.448 | −0.537, +0.537 | 0.527 | — |
| NIMBY | R2 | `built_2010_plus` | −0.033 | −0.431, +0.351 | −0.506, +0.433 | 0.439 | — |
| NIMBY | R2 | `nhwhite_share` | +0.105 | −0.277, +0.482 | −0.352, +0.554 | 0.675 | — |
| NIMBY | R2 | `transit_commute` | −0.342 | −0.720, +0.038 | −0.807, +0.117 | 0.069 | — |
| NIMBY | R2 | `vacancy_rate` | −0.143 | −0.470, +0.169 | −0.548, +0.226 | 0.226 | — |
| NIMBY | R2 | `log_population` | −0.075 | −0.432, +0.240 | −0.503, +0.300 | 0.346 | — |
| NIMBY | R2 | **`delta_role`** | **+2.112** | +1.788, +2.465 | +1.726, +2.533 | 1.000 | **95 %** |
| NIMBY | R2 | `theta_role_x_renter` | −0.006 | −0.318, +0.305 | −0.382, +0.379 | 0.486 | — |
| NIMBY | R2 | `theta_role_x_income` | −0.038 | −0.363, +0.288 | −0.431, +0.350 | 0.423 | — |
| NIMBY | R2 paired | `renter_share` | −0.030 | −0.384, +0.338 | −0.454, +0.414 | 0.437 | — |
| NIMBY | R2 paired | `log_income` | −0.019 | −0.380, +0.354 | −0.453, +0.435 | 0.464 | — |
| NIMBY | R2 paired | `log_density` | +0.040 | −0.457, +0.494 | −0.564, +0.582 | 0.560 | — |
| NIMBY | R2 paired | `built_2010_plus` | −0.107 | −0.536, +0.348 | −0.624, +0.442 | 0.337 | — |
| NIMBY | R2 paired | `nhwhite_share` | +0.105 | −0.347, +0.549 | −0.432, +0.641 | 0.654 | — |
| NIMBY | R2 paired | `transit_commute` | −0.378 | −0.854, +0.101 | −0.959, +0.197 | 0.096 | — |
| NIMBY | R2 paired | `vacancy_rate` | −0.102 | −0.464, +0.242 | −0.534, +0.323 | 0.305 | — |
| NIMBY | R2 paired | `log_population` | −0.000 | −0.402, +0.359 | −0.483, +0.429 | 0.499 | — |
| NIMBY | R2 paired | **`delta_role`** | **+2.035** | +1.704, +2.384 | +1.636, +2.459 | 1.000 | **95 %** |
| NIMBY | R2 paired | `theta_role_x_renter` | −0.006 | −0.328, +0.326 | −0.389, +0.389 | 0.490 | — |
| NIMBY | R2 paired | `theta_role_x_income` | +0.016 | −0.313, +0.336 | −0.380, +0.407 | 0.533 | — |
| YIMBY | R1 | `renter_share` | +0.036 | −0.201, +0.257 | −0.252, +0.298 | 0.605 | — |
| YIMBY | R1 | `log_income` | −0.131 | −0.336, +0.087 | −0.376, +0.143 | 0.160 | — |
| YIMBY | R1 | `log_density` | −0.018 | −0.309, +0.298 | −0.366, +0.360 | 0.460 | — |
| YIMBY | R1 | `built_2010_plus` | +0.161 | −0.142, +0.429 | −0.213, +0.480 | 0.820 | — |
| YIMBY | R1 | `nhwhite_share` | +0.277 | +0.021, +0.535 | −0.029, +0.585 | 0.961 | 90 % |
| YIMBY | R1 | **`transit_commute`** | **+0.336** | +0.079, +0.583 | +0.029, +0.630 | 0.982 | **95 %** |
| YIMBY | R1 | `vacancy_rate` | +0.091 | −0.124, +0.313 | −0.165, +0.362 | 0.757 | — |
| YIMBY | R1 | **`log_population`** | **+0.303** | +0.092, +0.530 | +0.054, +0.577 | 0.991 | **95 %** |
| YIMBY | R1 | **`delta_role`** | **+1.015** | +0.763, +1.271 | +0.708, +1.324 | 1.000 | **95 %** |
| YIMBY | R1 | `theta_role_x_renter` | +0.142 | −0.102, +0.384 | −0.149, +0.434 | 0.827 | — |
| YIMBY | R2 | `renter_share` | +0.035 | −0.198, +0.254 | −0.239, +0.302 | 0.605 | — |
| YIMBY | R2 | `log_income` | −0.122 | −0.365, +0.118 | −0.415, +0.167 | 0.201 | — |
| YIMBY | R2 | `log_density` | −0.021 | −0.308, +0.290 | −0.369, +0.350 | 0.454 | — |
| YIMBY | R2 | `built_2010_plus` | +0.158 | −0.135, +0.429 | −0.193, +0.479 | 0.823 | — |
| YIMBY | R2 | `nhwhite_share` | +0.280 | +0.017, +0.539 | −0.034, +0.598 | 0.960 | 90 % |
| YIMBY | R2 | **`transit_commute`** | **+0.337** | +0.085, +0.574 | +0.028, +0.622 | 0.983 | **95 %** |
| YIMBY | R2 | `vacancy_rate` | +0.092 | −0.127, +0.311 | −0.170, +0.355 | 0.746 | — |
| YIMBY | R2 | **`log_population`** | **+0.304** | +0.098, +0.526 | +0.059, +0.573 | 0.992 | **95 %** |
| YIMBY | R2 | **`delta_role`** | **+1.025** | +0.765, +1.272 | +0.713, +1.327 | 1.000 | **95 %** |
| YIMBY | R2 | `theta_role_x_renter` | +0.146 | −0.103, +0.395 | −0.161, +0.437 | 0.834 | — |
| YIMBY | R2 | `theta_role_x_income` | −0.014 | −0.257, +0.247 | −0.309, +0.297 | 0.462 | — |
| YIMBY | R2 paired | `renter_share` | +0.018 | −0.228, +0.254 | −0.279, +0.300 | 0.552 | — |
| YIMBY | R2 paired | `log_income` | −0.122 | −0.369, +0.131 | −0.416, +0.187 | 0.210 | — |
| YIMBY | R2 paired | `log_density` | −0.067 | −0.366, +0.260 | −0.430, +0.331 | 0.361 | — |
| YIMBY | R2 paired | `built_2010_plus` | +0.304 | +0.004, +0.592 | −0.057, +0.653 | 0.952 | 90 % |
| YIMBY | R2 paired | `nhwhite_share` | +0.381 | +0.057, +0.702 | −0.012, +0.763 | 0.972 | 90 % |
| YIMBY | R2 paired | **`transit_commute`** | **+0.505** | +0.186, +0.837 | +0.117, +0.898 | 0.995 | **95 %** |
| YIMBY | R2 paired | `vacancy_rate` | +0.082 | −0.153, +0.321 | −0.202, +0.367 | 0.724 | — |
| YIMBY | R2 paired | `log_population` | +0.289 | +0.042, +0.554 | −0.009, +0.610 | 0.970 | 90 % |
| YIMBY | R2 paired | **`delta_role`** | **+1.065** | +0.813, +1.329 | +0.764, +1.380 | 1.000 | **95 %** |
| YIMBY | R2 paired | `theta_role_x_renter` | +0.155 | −0.110, +0.417 | −0.168, +0.464 | 0.833 | — |
| YIMBY | R2 paired | `theta_role_x_income` | −0.060 | −0.312, +0.207 | −0.361, +0.262 | 0.344 | — |

**The 95 % survivors, explicitly.** On the YIMBY outcome, besides role:
**`transit_commute` +0.337 (95 % +0.028, +0.622)** and **`log_population`
+0.304 (95 % +0.059, +0.573)**, both in R1, R2 and the paired-only fit;
`transit_commute` strengthens to +0.505 on the paired rows. **`nhwhite_share`
+0.280 clears 90 %** (+0.017, +0.539) but not 95 %, and `built_2010_plus`
+0.304 clears 90 % on the paired fit only. On NIMBY nothing but role clears
either level, though `transit_commute` −0.342 (Pr < 0 = 0.93) is the same
covariate pointing the other way, which is the one coherent non-role pattern
in the document — and `plan/RENTER-SHARE-4-SUBSETS.md` §3 independently
promotes `transit_commute` to the covariate with the highest inclusion
probability on both outcomes. **Every one of these is as ecological as renter
share** (§5.6): they describe residents, not speakers.

### 3b. The two implied renter-share slopes

Posterior derived quantities, log-odds per 1 sd of renter share (sd 0.0886, so
1 sd ≈ 8.9 points); `per +10 pt` rescales the median.

| outcome | spec | rows | role | slope | 90 % | 95 % | Pr > 0 | per +10 pt |
|---|---|---|---|---|---|---|---|---|
| NIMBY | R1 | 64 | dais | −0.019 | −0.335, +0.306 | −0.402, +0.387 | 0.461 | −0.022 |
| NIMBY | R1 | 64 | public | −0.027 | −0.350, +0.331 | −0.421, +0.411 | 0.450 | −0.030 |
| NIMBY | R2 | 64 | dais | −0.018 | −0.336, +0.323 | −0.402, +0.394 | 0.461 | −0.021 |
| NIMBY | R2 | 64 | public | −0.025 | −0.355, +0.330 | −0.420, +0.407 | 0.452 | −0.028 |
| NIMBY | R2 | 58 paired | dais | −0.030 | −0.384, +0.338 | −0.454, +0.414 | 0.437 | −0.033 |
| NIMBY | R2 | 58 paired | public | −0.035 | −0.395, +0.341 | −0.474, +0.425 | 0.441 | −0.038 |
| YIMBY | R1 | 64 | dais | +0.036 | −0.201, +0.257 | −0.252, +0.298 | 0.605 | +0.041 |
| YIMBY | R1 | 64 | **public** | **+0.175** | −0.085, +0.430 | −0.137, +0.482 | **0.871** | +0.198 |
| YIMBY | R2 | 64 | dais | +0.035 | −0.198, +0.254 | −0.239, +0.302 | 0.605 | +0.039 |
| YIMBY | R2 | 64 | **public** | **+0.178** | −0.076, +0.429 | −0.126, +0.478 | **0.876** | +0.201 |
| YIMBY | R2 | 58 paired | dais | +0.018 | −0.228, +0.254 | −0.279, +0.301 | 0.552 | +0.020 |
| YIMBY | R2 | 58 paired | public | +0.171 | −0.106, +0.443 | −0.167, +0.497 | 0.848 | +0.185 |

### 3b-lobo. Leave-one-body-out, attached to every slope above

Added 2026-10-19 (`plan/REGRESSION-REVIEW-2026-10-19.md` finding 6).
`analysis/renter_share5_robust.py` → `output/renter_share_5/robust/lobo.csv`.
Each body is dropped in turn and the whole design refitted; 25 refits per
estimate. These are **IRLS point estimates on the same 64 rows** (§4's routine,
no hierarchy and no priors), so they are comparable to §4 and not to the
Bayesian medians above — the Bayesian medians are the ones to quote, the LOBO
column is the fragility.

| outcome | spec | estimate | full sample | LOBO min (body) | LOBO max (body) | sign flips / 25 |
|---|---|---|---|---|---|---|
| NIMBY | R1 | renter (dais) | −0.044 | −0.279 (**santaclara**) | +0.093 (**sanjose**) | **3** |
| NIMBY | R1 | renter (public) | −0.181 | −0.248 (losaltos) | **+0.030 (santaclara)** | **1** |
| NIMBY | R1 | `transit_commute` | −0.473 | −0.610 (santarosa) | **−0.119 (sanjose)** | 0 |
| NIMBY | R1 | role (δ) | +2.057 | +1.971 (sanjose) | +2.221 (santaclara) | 0 |
| NIMBY | R2 | renter (dais) | −0.044 | −0.360 (**santaclara**) | +0.094 (**sanjose**) | **3** |
| NIMBY | R2 | renter (public) | −0.175 | −0.248 (losaltos) | **+0.036 (santaclara)** | **1** |
| NIMBY | R2 | `transit_commute` | −0.478 | −0.620 (santarosa) | **−0.126 (sanjose)** | 0 |
| NIMBY | R2 | role (δ) | +2.057 | +1.970 (sanjose) | +2.242 (santaclara) | 0 |
| YIMBY | R1 | renter (dais) | +0.094 | +0.019 (**santaclara**) | +0.132 (sanrafael) | 0 |
| YIMBY | R1 | renter (public) | +0.222 | **−0.026 (santaclara)** | +0.267 (sanrafael) | **1** |
| YIMBY | R1 | `transit_commute` | +0.424 | **+0.258 (sanjose)** | +0.470 (hayward) | 0 |
| YIMBY | R1 | role (δ) | +0.997 | +0.926 (sanleandro) | +1.153 (sanjose) | 0 |
| YIMBY | R2 | renter (dais) | +0.088 | +0.034 (sanjose) | +0.127 (sanrafael) | 0 |
| YIMBY | R2 | renter (public) | +0.252 | **−0.044 (santaclara)** | +0.311 (pleasanton) | **1** |
| YIMBY | R2 | `transit_commute` | +0.413 | **+0.255 (sanjose)** | +0.458 (hayward) | 0 |
| YIMBY | R2 | role (δ) | +0.988 | +0.914 (santaclara) | +1.145 (sanjose) | 0 |

**Three bodies decide the renter-share and transit numbers, and none of them
decides role.**

* **Santa Clara** (highest renter share, 0.588, three route cells, 559
  clusters): dropping it is the single largest move on 16 of the 32 LOBO
  estimates across runs 2 and 3. It is what makes the public slopes
  *directional* at all — the public YIMBY slope goes +0.222 → −0.026 and the
  public NIMBY slope −0.181 → +0.030 without it, i.e. **both public renter
  slopes change sign on one body.**
* **San José** (191 public clusters, 23 % of the public base): the largest
  single move on `transit_commute`, −0.473 → −0.119 on NIMBY (a 75 % loss of
  magnitude), and on the NIMBY dais slope.
* **San Leandro**, which the published text flags beside San José as a
  concentration risk, turns out **not** to be a LOBO extreme on any
  renter-share or transit estimate — its largest effect anywhere is on δ
  (+0.997 → +0.926, 7 %). The concentration problem San Leandro creates is the
  `affordability_displacement` sign flip (§5.2), not leverage on a slope.
* **δ moves 7–16 % under every single-body drop and never approaches zero.**
  Whatever is wrong with δ is the denominator (§0a) and the role audit (§5.1),
  not one body.

The public YIMBY slope is ~5× the dais slope on the point estimate and in the
expected direction, but its 90 % interval contains zero on all three fits, **it
changes sign when Santa Clara is dropped** (§3b-lobo), and
the paired-only fit (which drops the six dais-only cells) reproduces it almost
exactly — so the role terms are not an artefact of the unbalanced rows. Note
this public slope (+0.175 to +0.178) is **smaller than run 2's public-only
base-model +0.287**, whose 90 % just cleared zero; the difference is the six
extra covariates and the body random intercept, exactly as run 2 §4 predicted.

### 3c. Diagnostics and PPC

| outcome | spec | rows | max R̂ | min bulk ESS | divergences | PPC inside 90 % | σ_body | σ_row |
|---|---|---|---|---|---|---|---|---|
| NIMBY | R1 | 64 | 1.0030 | 1,848 | 1 | **64/64** | 0.331 | 0.553 |
| NIMBY | R2 | 64 | 1.0021 | 1,770 | **0** | **64/64** | 0.325 | 0.574 |
| NIMBY | R2 paired | 58 | 1.0022 | 1,140 | 1 | **58/58** | — | — |
| YIMBY | R1 | 64 | 1.0042 | 2,214 | 4 | **64/64** | 0.151 | 0.370 |
| YIMBY | R2 | 64 | 1.0026 | 2,274 | 1 | **64/64** | 0.145 | 0.379 |
| YIMBY | R2 paired | 58 | 1.0021 | 2,529 | 7 | **58/58** | — | — |

R̂ ≤ 1.004 and bulk ESS ≥ 1,140 everywhere. Divergences are 0–7 out of 8,000
draws; the paired-only YIMBY fit's 7 are the σ_body funnel on 22 bodies with a
small between-body sd (0.15) and do not move any reported quantity — that fit
is a sensitivity check, not a headline. σ_row > σ_body on both outcomes, i.e.
most of the extra-binomial spread is route/role-row specific rather than a
stable body level.

~~PPC is clean: **every row** on every fit has its observed count inside the
replicated 90 % band.~~

> **Superseded 2026-10-19** (`plan/REGRESSION-REVIEW-2026-10-19.md` finding 3).
> **That check cannot fail and 64/64 is not a pass.** `u_i` is one free
> parameter per row, so the conditional predictive is centred on the observed
> count. Under a correctly calibrated 90 % band the expected count is 57.6 of
> 64 and P(64/64) = 0.9⁶⁴ = **0.0013** — a perfect score is evidence the band
> is conditioning on the answer, not evidence of fit. The counts are kept above
> with a strikethrough; §3c-cal replaces them.

### 3c-cal. Calibration with the overdispersion term integrated out

`analysis/renter_share5_ppc_horseshoe.py` →
`output/renter_share_5/ppc_horseshoe/ppc_marginal.csv`. The fits are repeated
and, for every posterior draw, a **fresh `u* ~ Normal(0, σ_row)` replaces the
fitted `u_i`**, so the band is out-of-sample for a new row of the same body and
route. The body random intercept is *not* integrated out: 25 levels is a real
grouping, not one parameter per row. Two statistics: rows inside the marginal
90 % band, and the predictive PIT of each observed count,
P(y* < y) + ½P(y* = y), which should be Uniform(0,1) under calibration (KS
test). Cell-mean z is the observed share against the marginal predictive share.

| run / outcome | spec | rows | conditional (published) | **marginal** | expected | PIT KS | KS p | cell-mean \|z\| > 2 |
|---|---|---|---|---|---|---|---|---|
| run 3 NIMBY | R2 | 64 | ~~64/64~~ | **63/64** | 57.6 | 0.127 | 0.232 | 2 |
| run 3 YIMBY | R2 | 64 | ~~64/64~~ | **63/64** | 57.6 | 0.096 | 0.565 | 1 |
| run 2 public NIMBY | (a) | 29 | ~~29/29~~ | 28/29 | 26.1 | 0.114 | 0.806 | 1 |
| run 2 public NIMBY | (c)* | 29 | ~~29/29~~ | 29/29 | 26.1 | 0.156 | 0.436 | 0 |
| run 2 public YIMBY | (a) | 29 | ~~29/29~~ | 28/29 | 26.1 | 0.096 | 0.929 | 1 |
| run 2 public YIMBY | (c)* | 29 | ~~29/29~~ | 29/29 | 26.1 | 0.149 | 0.497 | 0 |
| run 2 all-roles NIMBY | (a) | 35 | ~~35/35~~ | 33/35 | 31.5 | 0.102 | 0.828 | 1 |
| run 2 all-roles NIMBY | (c)* | 35 | ~~35/35~~ | 35/35 | 31.5 | 0.102 | 0.825 | 0 |
| run 2 all-roles YIMBY | (a) | 35 | ~~35/35~~ | 34/35 | 31.5 | 0.087 | 0.931 | 1 |
| run 2 all-roles YIMBY | (c)* | 35 | ~~35/35~~ | 34/35 | 31.5 | 0.115 | 0.698 | 0 |

\* all eight covariates under Normal(0, 1), not the horseshoe — the horseshoe's
PIT is reported in `ppc_marginal.csv` as the (c) row of the same design.

**What this says.** With `u_i` integrated out the models are *still*
over-covered — 63/64 and 28–35 of 29–35 against an expected 90 % — and the PIT
is indistinguishable from uniform on every fit (KS p 0.23–0.93). So the fit is
adequate, but it is adequate the way a model with an extra-binomial spread of
σ_row ≈ 0.55 is adequate: the predictive intervals are wide enough to cover
almost anything, and over-coverage is the symptom. **The right reading of
§3c-cal is "nothing in these rows contradicts the model", not "the model fits
well", and the published 64/64 should never have been offered as evidence
either way.** Two run-3 NIMBY cells still have a cell-mean |z| > 2 after
marginalisation.

## 4. Frequentist check — the by-body cluster-robust SE is the primary one

> **Changed 2026-10-19** (`plan/REGRESSION-REVIEW-2026-10-19.md` findings 4
> and 5 of "before quoting"). This section was headed "sanity check" and
> treated HC3 as the main column with the by-body sandwich beside it. That is
> backwards: a cluster is not an exchangeable trial, the rows nest inside
> episodes inside meetings inside bodies, and HC3 assumes the 64 rows are
> independent. **The `cluster SE` / `cluster 95 %` columns below are the
> primary frequentist intervals; HC3 is the optimistic bound.** The by-body
> sandwich is still *not* enough — it corrects for body, not for the episode
> and meeting nesting measured in §5.10 — so read it as a floor on the width,
> not the right width.

`models/sanity_hc3.csv`, and `output/renter_share_5/robust/
run3_cluster_primary.csv` for the same fits with 90 % as well as 95 % cluster
intervals. Binomial logistic IRLS on the same 64 stacked rows,
same terms, route dummies explicit, no hierarchy and no priors; two robust
covariance estimates — HC3 (MacKinnon–White `s_i²/(1−h_i)²`) and a **by-body
cluster-robust sandwich** over G = 25 body clusters with the
`G/(G−1)·(n−1)/(n−k)` correction. All fits converged.

| outcome | spec | term | β | HC3 SE | HC3 95 % | cluster SE | cluster 95 % |
|---|---|---|---|---|---|---|---|
| NIMBY | R1 | **role** | **+2.057** | 0.253 | **+1.561, +2.553** | 0.181 | **+1.702, +2.412** |
| NIMBY | R1 | renter share | −0.044 | 0.235 | −0.505, +0.417 | 0.154 | −0.346, +0.258 |
| NIMBY | R1 | role × renter | −0.137 | 0.275 | −0.676, +0.401 | 0.250 | −0.628, +0.354 |
| NIMBY | R1 | log income | +0.040 | 0.202 | −0.356, +0.436 | 0.212 | −0.376, +0.456 |
| NIMBY | R2 | **role** | **+2.057** | 0.259 | **+1.550, +2.564** | 0.181 | **+1.703, +2.412** |
| NIMBY | R2 | role × income | −0.036 | 0.204 | −0.436, +0.364 | 0.165 | −0.358, +0.287 |
| YIMBY | R1 | **role** | **+0.997** | 0.231 | **+0.544, +1.450** | 0.197 | **+0.611, +1.383** |
| YIMBY | R1 | renter share | +0.094 | 0.117 | −0.135, +0.322 | 0.074 | −0.052, +0.240 |
| YIMBY | R1 | role × renter | +0.128 | 0.158 | −0.181, +0.437 | 0.146 | −0.158, +0.414 |
| YIMBY | R1 | log income | −0.140 | 0.112 | −0.358, +0.079 | 0.076 | −0.289, +0.009 |
| YIMBY | R2 | **role** | **+0.988** | 0.230 | **+0.538, +1.438** | 0.193 | **+0.609, +1.367** |
| YIMBY | R2 | role × renter | +0.164 | 0.182 | −0.193, +0.521 | 0.177 | −0.182, +0.510 |
| YIMBY | R2 | role × income | −0.132 | 0.169 | −0.462, +0.199 | 0.142 | −0.410, +0.146 |

Implied public renter slope (β_renter + β_interaction), frequentist point
estimates: NIMBY −0.181 (R1) / −0.175 (R2); YIMBY +0.222 (R1) / +0.252 (R2) —
and §3b-lobo shows all four change sign when Santa Clara is dropped.
**Same story from the other direction:** role survives both sandwiches with
room to spare and on both outcomes; every renter, income and interaction
interval crosses zero under both. The YIMBY income term is the closest miss
(cluster 95 % −0.289 to +0.009), which matches run 2 §3c. Note that on these
fits the cluster SE is *smaller* than HC3 on most terms (role: 0.181 against
0.253), which is not reassurance — it is what happens when G = 25 clusters
carry 64 rows and the within-body score contributions partly cancel; §5.10's
design effects are the honest correction and they go the other way.

## 5. Limitations

Runs 1 and 2's limitations all carry forward. The ones this design changes:

1. **Role is an audit, not a gold label** (`plan/ROLE-AUDIT-2026-10-18.md`
   ra-1). The role column is `role_audit`: SELF_ID is model output no human has
   checked, and the audit re-partitioned 356 cue-dais clusters in and 327
   cue-public out at S2. **δ is now the headline coefficient of this
   document, and δ is a coefficient on that audit.** A systematic audit bias
   — e.g. staff-report readings landing in `dais` while objections land in
   `public` — would produce exactly this gap. Run 2's outcomes 3–4 pooled both
   roles specifically to be insensitive to this column; this run is maximally
   sensitive to it. **δ is not quotable as a finding about deliberation until
   the role column is hand-checked.**
2. **The role gap is ~~partly~~ *mainly* definitional** (strengthened
   2026-10-19 on the review's verdict, and §0a is the measurement of it: two
   thirds of the published gap was the `neutral` class and the staff
   clusters). Dais speech includes procedural and
   staff-report material that is coded `no stance` less often than it is coded
   neutral, and the mapping from "restriction-supporting" to NIMBY flips sign
   on items that are themselves restrictions (still 133 San Leandro clusters,
   `plan/STANCE-S2.md` §5). A dais member moving an item is not making the same
   speech act as a resident objecting to it.
3. **Interaction power.** θ is identified by 29 within-body role contrasts
   across a renter-share range of 0.192–0.588. The 90 % interval on θ_r is
   ±0.3 log-odds wide — wide enough to contain both "no difference" and "the
   public slope is twice the run-2 estimate". This fit **cannot** resolve the
   interaction; it can only report that role dwarfs it.
4. **Standardisation weights bodies by row count.** Covariates vary at body
   level and are duplicated across role rows, so a body with 3 route cells and
   both roles contributes 6 rows to the mean and sd. `cov_mean`/`cov_sd` are in
   `model_results.json` per fit; coefficients can be put back on the
   covariate's own scale from there.
5. **Unbalanced rows.** 35 dais rows against 29 public rows, 6 dais-only cells.
   Handled by the paired-only sensitivity fit (§3b), which moves nothing.
6. **Ecological inference**, unchanged and still central: `renter_share` is the
   renter share of a jurisdiction's *residents*, the outcome is what people who
   came and spoke said, and nothing in this corpus links them.
7. **n.** 25 bodies, 35 cells, 64 rows; covariates vary at body level, so the
   effective n for every β is still 25. Eight covariates plus role plus one or
   two interactions plus four route strata on 25 bodies.
8. ~~**No LOO** (§3), so no out-of-sample comparison of R1 against R2.~~
   **Replaced 2026-10-19** (review finding 7): PSIS-LOO is still not reported
   and still should not be, but a **grouped 5-fold by body** is, and it is a
   valid comparison because whole bodies are held out. On these stacked rows,
   unpenalised logit, role always in:

   | outcome | covariate set | p | K-fold ELPD (se) |
   |---|---|---|---|
   | NIMBY | **route only + role** | **6** | **−220.5 (29.3)** |
   | NIMBY | renter+income + role | 8 | −244.2 (32.7) |
   | NIMBY | + role × renter | 9 | −250.9 (34.4) |
   | NIMBY | +density + role | 9 | −246.6 (34.0) |
   | NIMBY | all eight + role | 14 | −378.9 (64.9) |
   | NIMBY | all eight + role × renter + role | 15 | −380.7 (62.6) |
   | YIMBY | route only + role | 6 | −200.8 (17.7) |
   | YIMBY | **renter+income + role** | **8** | **−199.5 (18.2)** |
   | YIMBY | + role × renter | 9 | −203.1 (18.1) |
   | YIMBY | +density + role | 9 | −201.5 (17.1) |
   | YIMBY | all eight + role | 14 | −216.1 (24.8) |
   | YIMBY | all eight + role × renter + role | 15 | −217.8 (24.6) |

   **Role plus the route strata alone predicts a held-out body better than any
   covariate set on NIMBY, and ties on YIMBY** (−199.5 against −200.8 is far
   inside one se). Adding the eight covariates costs 158 ELPD on NIMBY. Adding
   the interaction costs ELPD on every outcome. Out of sample, by body, the
   covariates are not buying anything — which is the same message as §3a from a
   direction that can be lost rather than won.
9. ~~**Multiplicity.** 2 outcomes × 2 specs + 2 sensitivity fits = 6 fits,
   uncorrected. Nothing needs correcting: one parameter clears 95 % and it
   clears it by 1.7 log-odds.~~ **Restated 2026-10-19** (review finding 8;
   `output/renter_share_5/robust/multiplicity.csv`): the count was wrong in
   both directions. 6 fits carry **64 substantive intervals** (8 covariates +
   role + 1–2 interactions, per fit), of which **16 exclude zero at 90 %**
   against **6.4 expected** under a global null and **11 at 95 %** against
   **3.2 expected** — not "one parameter clears 95 %", which is what §5.9
   originally said. The excess is real but concentrated: the 11 are
   `delta_role` ×4, `transit_commute` ×3 and `log_population` ×3 (plus one
   repeat), i.e. **three distinct parameters repeated across nested fits**, so
   the effective number of independent tests is far below 64 and far above 1.
   Nothing here needs a Bonferroni to survive — δ clears it by an order of
   magnitude — but "nothing needs correcting" was argued from a false premise.
10. **Nothing is meeting-clustered** (review finding 4;
    `output/renter_share_5/robust/icc.csv`). A cluster is not an exchangeable
    trial. **86.3 % of public stance-bearing clusters rest on a single scored
    turn** (dais 53.1 %) and cluster length spans 64–702 words p10–p90. The
    clusters nest, and the NIMBY indicator's intraclass correlation is:

    | grouping | groups | mean size | ICC | design effect | SE inflation |
    |---|---|---|---|---|---|
    | episode | 737 | 3.82 | **0.275** | 1.77 | **×1.33** |
    | meeting | 293 | 9.60 | **0.156** | 2.34 | **×1.53** |
    | episode, directional rows only | 479 | 2.87 | 0.276 | 1.52 | ×1.23 |

    `u_i` is a mean shift per row and does not absorb this; run 4's
    quasi-binomial ĉ = 1.99 is the same statistic seen another way. **Every
    interval in this document is therefore ~1.3–1.5× too narrow**, and most on
    the two cells (San José public 191, San Leandro public 173) that carry the
    design. δ survives that inflation with room to spare; nothing else in
    §3a-full does — `transit_commute` +0.337 (95 % +0.028, +0.622) does not
    survive a ×1.4 widening, and nor does `log_population`. **A turn-level
    model nesting episode in meeting in body, or a beta-binomial, is the
    documented next step and is not run here.**

## Corrections 2026-10-19

Applied from `plan/REGRESSION-REVIEW-2026-10-19.md`. **Every prior number is
kept above with a strikethrough or a "superseded" note; nothing was silently
replaced.** No covariate was re-pulled or re-screened; `output/renter_share_2/`
and `output/renter_share_3/` remain read-only. New artefacts only, under
`output/renter_share_5/`.

| # | what changed | where | why |
|---|---|---|---|
| 1 | "roughly an 8× odds ratio" → **2.2–2.3× on the directional denominator**; δ = +2.11 shown superseded | §0, new §0a | ~60 % of the old denominator was the `neutral` class, which is a dais phenomenon (69.1 % dais vs 10.5 % public), and ~20 % was staff clusters (94 % neutral). The old δ mostly measured "the dais says neutral things". |
| 2 | "Role is the only parameter whose interval excludes zero" struck; **every parameter now printed** at 90 % and 95 % | §3a, new §3a-full | False against this run's own `model_table.csv`. `transit_commute` and `log_population` clear 95 % on YIMBY in all three fits; `nhwhite_share` clears 90 %. |
| 3 | PPC counts "64/64" struck; replaced with a marginal band and PIT with `u_i` **integrated out** | §3c, new §3c-cal | One `u_i` per row centres the conditional band on the observed count. P(64/64) under a calibrated band = 0.0013, so a perfect score was evidence of over-conditioning. |
| 4 | leave-one-body-out column attached to every renter-share and transit estimate | new §3b-lobo | Both public renter slopes change sign on Santa Clara alone; `transit_commute` loses 75 % of its magnitude on San José. δ moves 7–16 % and never approaches zero. |
| 5 | HC3 demoted; **by-body cluster-robust SE is primary** | §4 heading and closing paragraph | A cluster is not an exchangeable trial; HC3 assumes the 64 rows are independent. |
| 6 | PSIS-LOO absence replaced by a **grouped 5-fold by body** ELPD table | §5.8 | A valid out-of-sample comparison exists even though PSIS-LOO is degenerate. Route + role alone beats every covariate set on NIMBY and ties on YIMBY. |
| 7 | multiplicity restated: 64 intervals, **16 exclude zero at 90 %** (6.4 expected), **11 at 95 %** (3.2 expected), not "one parameter clears 95 %" | §5.9 | The original count was simply wrong; the conclusion ("nothing needs correcting") happens to survive, but it was argued from a false premise. |
| 8 | episode and meeting **ICC and design effects** added; intervals declared ~1.3–1.5× too narrow | new §5.10 | The nesting is real (ICC 0.275 by episode, 0.156 by meeting) and `u_i` does not absorb it. A turn-level model is the next step, not run here. |

**What did not change.** The renter-share null, which is the question this
document was asked: `renter_share` is −0.02 / +0.04 on the old denominator and
−0.05 on the new one, the interaction is dead on every denominator and in every
form, and §3b-lobo makes the public slopes weaker rather than stronger. Role is
still the largest effect in the exercise and still the **largest effect on a
model-output label no human has hand-checked** (§5.1) — restating its magnitude
does not make it quotable as a finding about deliberation.

## 6. Files

New:
- `plan/RENTER-SHARE-3-ROLE.md` (this)
- 2026-10-19 corrections: `analysis/renter_share5_role_restated.py`,
  `analysis/renter_share5_robust.py`,
  `analysis/renter_share5_ppc_horseshoe.py`, and
  `output/renter_share_5/{role_restated,robust,ppc_horseshoe}/` plus the three
  run logs
- `analysis/renter_share3_role_outcomes.py`
- `analysis/renter_share3_role_models.py`
- `output/renter_share_3/outcomes_role_stacked.csv` — the 64 stacked rows
- `output/renter_share_3/cells.json` — row/cell/pairing counts, below-floor log
- `output/renter_share_3/outcomes.log`, `output/renter_share_3/models.log`
- `output/renter_share_3/models/model_table.csv` — every parameter, every fit
- `output/renter_share_3/models/slopes.csv` — the two implied renter slopes
- `output/renter_share_3/models/sanity_hc3.csv` — HC3 and by-body cluster SEs
- `output/renter_share_3/models/model_results.json` — full posterior summaries,
  diagnostics, PPC, cov_mean/cov_sd, the stacking metadata

Modified: nothing. `output/renter_share_2/` is **read only** — the ACS pull,
`covariates_wide.csv` and `screen.json` are re-used byte-for-byte, nothing
re-pulled and nothing re-screened. No figures. No DB write, no episode or
stance row touched, nothing committed in `~/ytpt`. `.env`, `plan/frame.csv`,
`plan/DECISIONS.md`, `plan/RENTER-SHARE.md` and `plan/RENTER-SHARE-2.md`
untouched.
