# RENTER-SHARE — does a body's renter share predict what its meetings say?

**EXPLORATORY — the stance κ ≥ 0.6 gate (`plan/SHOWCASE.md` 4a) is not
passed.** Run 2026-10-18 by the supervisor agent, offline from
`corpus.sqlite`, `state.sqlite`, `output/wave1/`, `output/stance_s1/` and one
unauthenticated Census file pull. **No YouTube request, no GPU, no Claude
call, $0.** Nothing committed. `.env`, `plan/frame.csv`, `plan/DECISIONS.md`
untouched. Companions: `plan/STANCE-S1-RESULTS.md` §7b (the stance instrument
and the YIMBY/NIMBY mapping), `plan/WAVE1-RESULTS.md` §5 (the rates),
`analysis/facets.py` (why nothing is pooled across locate routes).

## 0. Plain-language answer

*Corrected 2026-10-18 after role audit*: the stance outcomes below are now
computed from `role_audit` (§2), not the diarization-cue role. **The answer is
unchanged** — every interval still spans "no effect" — but the point estimates
move a little further in the hypothesised direction (NIMBY slope Spearman
−0.04 → −0.15, YIMBY +0.20 → +0.27, net +0.11 → +0.20 on the classifier
route), and two bodies (bart, vallejo) drop below the turn floor while two
others enter. The sentence below stands as written.

**On this corpus, a body's renter share does not predict what is said at its
meetings in any direction the data can resolve.** Across the 18 bodies where
enough public comment has been scored, the share of public speakers taking a
NIMBY position drifts very slightly *down* as renter share rises and the
YIMBY share drifts slightly *up*, but both effects are small relative to how
uncertain they are: the 90 % interval on the NIMBY slope runs from a large
negative effect to a moderate positive one, and the same is true of every
other outcome tested. The one number that looks like a finding — more
renter-heavy bodies spend more time on density and transit items — rests on
seven bodies, two of which are transit boards that talk about transit by
definition, and its interval also crosses "no effect" once anything else is
controlled for. The honest summary is **n is too small, not "renter share
doesn't matter"**: with 15–20 bodies and a median of 18 scored public turns
each, this design could only have detected an effect several times larger
than any plausible one. The stance instrument is also exploratory, and even
if the correlation were sharp it would be an *ecological* statement — the
renter share of a city's residents is not the renter share of the people who
show up to speak.

*Corrected 2026-10-19 after `plan/REGRESSION-REVIEW-2026-10-19.md`*: **the
answer above is unchanged and is now better supported**, with three
qualifications added by runs 2–4 and the review. (i) The near-result runs 2–3
found on the public YIMBY share **rests on one body** — dropping Santa Clara
takes it from +0.344 to +0.096 log-odds per sd and flips its sign in the
eight-covariate fit (`plan/RENTER-SHARE-2.md` §3a-lobo). (ii) "Renter share
does not predict what is said" is safe; **"and nor does anything else" is
not** — `transit_commute` survives every valid permutation null and
`log_population` clears 95 % on one outcome (`plan/RENTER-SHARE-2.md`
§3a-full). (iii) Every interval in this document and in runs 2–4 is **about
1.4× too narrow**: speaker clusters nest inside episodes inside meetings, with
an ICC of 0.275 by episode and 0.156 by meeting on the NIMBY indicator, which
no per-cell overdispersion term absorbs (`plan/RENTER-SHARE-3-ROLE.md` §5.10).
Widening the intervals only strengthens this document's conclusion, which is
why it stands as written.

## 1. Covariate — data source, vintage, method

`analysis/renter_share_acs.py` → `output/renter_share/{acs.json,covariates.csv}`.

**Source.** `api.census.gov` now 302s every unkeyed request to
`/data/missing_key.html` (checked 2026-10-18 20:54 UTC); there is no Census
key in this project and obtaining one is a signup, not a light pull. The same
estimates are in the **table-based ACS 5-year Summary File**, which needs no
key:

| file | URL |
|---|---|
| B25003 tenure | `https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b25003.dat` |
| B19013 median household income | `…/acsdt5y2023-b19013.dat` |
| B01003 total population | `…/acsdt5y2023-b01003.dat` |
| geography crosswalk | `…/table-based-SF/documentation/Geos20235YR.txt` |
| land area (ALAND) | `https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_{place,counties}_national.zip` |

**Vintage: ACS 2019–2023 5-year**, the latest 5-year release; geography
vintage 2023. Raw files and their download logs are under
`output/renter_share/raw/` (`dl.log`, `gaz.log`), one pull with a 1 s gap.

**Definitions.** `renter_share` = `B25003_E003 / B25003_E001` (renter-occupied
/ occupied housing units — the denominator is occupied units, not all units).
`med_hh_income` = `B19013_E001`. `pop_density` = `B01003_E001` divided by
ALAND in square miles.

**Lookup table** (full list with GEO_IDs in `acs.json → lookup`; councils are
census *places*, Napa is the county Board of Supervisors, the three transit
boards are population-weighted unions of their **member counties**):

| body | geography | GEO_ID(s) | renter share |
|---|---|---|---|
| losaltos | Los Altos city | 1600000US0643280 | 0.192 |
| livermore | Livermore city | 1600000US0641992 | 0.280 |
| pleasanton | Pleasanton city | 1600000US0657792 | 0.329 |
| **napa** | **Napa County** (BoS) | 0500000US06055 | 0.356 |
| petaluma | Petaluma city | 1600000US0656784 | 0.361 |
| concord | Concord city | 1600000US0616000 | 0.386 |
| cupertino | Cupertino city | 1600000US0617610 | 0.387 |
| dalycity | Daly City city | 1600000US0617918 | 0.395 |
| fremont | Fremont city | 1600000US0626000 | 0.398 |
| **samtrans** | **San Mateo County** | 0500000US06081 | 0.409 |
| vallejo | Vallejo city | 1600000US0681666 | 0.411 |
| sanleandro | San Leandro city | 1600000US0668084 | 0.417 |
| sausalito | Sausalito city | 1600000US0670364 | 0.421 |
| hayward | Hayward city | 1600000US0633000 | 0.431 |
| santarosa | Santa Rosa city | 1600000US0670098 | 0.438 |
| sanjose | San Jose city | 1600000US0668000 | 0.441 |
| **vta** | **Santa Clara County** | 0500000US06085 | 0.447 |
| paloalto | Palo Alto city | 1600000US0655282 | 0.458 |
| **bart** | **Alameda + Contra Costa + San Francisco counties** | 0500000US06001; 06013; 06075 | 0.460 |
| richmond | Richmond city | 1600000US0660620 | 0.475 |
| sanrafael | San Rafael city | 1600000US0668364 | 0.482 |
| sanmateo | San Mateo city | 1600000US0668252 | 0.492 |
| sunnyvale | Sunnyvale city | 1600000US0677000 | 0.562 |
| oakland | Oakland city | 1600000US0653000 | 0.583 |
| santaclara | Santa Clara city | 1600000US0669084 | 0.588 |
| healdsburg, cotati, milpitas, rohnertpark | their places | see `acs.json` | 0.410, 0.418, 0.407, 0.494 |

**Transit-district method.** A district's renter share is the share over the
*pooled* member counties: Σ renter units / Σ occupied units, which is the
population-weighted mean by construction. **BART = Alameda + Contra Costa +
San Francisco** (the three counties in the BART District; San Mateo is served
under contract and is not a member county). **VTA = Santa Clara County.**
**SamTrans = San Mateo County.** For income this gives a weighted mean of
three county *medians*, which is not itself a median; those rows carry
`income_is_weighted_mean = 1`.

## 2. Outcomes — per body, per locate route, never pooled

`analysis/renter_share_outcomes.py` → `output/renter_share/outcomes_by_body_route.csv`
(32 body × route rows).

**(a) Public stance shares** from the S1 census of `turn_stance`
(`STANCE-S1-RESULTS.md` §7b mapping: density/both-family episodes, public
role, turns with a numeric stance; **NIMBY ≤ −0.5, YIMBY ≥ +0.5**). The unit
here is the **turn**, not the speaker cluster: §7b's cluster table clears its
≥ 10-cluster floor in only 12 body × route cells, and the turn table reaches
20 cells at a ≥ 5-turn floor. §7b's cluster shares are carried in the CSV
(`cl_YIMBY`, `cl_NIMBY`) for comparison and agree in sign everywhere.
**(a2)** public − dais NIMBY-share gap, same frame, model-free.
**(b) Rates** — density and transit located minutes per meeting-hour from
`output/wave1/rates_by_body_route.csv`. Complete-coverage bodies are the
primary set (11 bodies, 28 route × family cells); partial bodies are floors
and enter only the flagged sensitivity.
**(c)** the lexicon pilot's per-body public shares
(`output/nimby_pilot/public_share_by_body_<route>.csv`), the weaker
instrument, as a sanity comparison only.

*Corrected 2026-10-18 after role audit.* **These stance outcomes used the
diarization-cue role and have been recomputed under `role_audit`** (rule ra-1,
`plan/ROLE-AUDIT-2026-10-18.md`), now the reported role throughout the project.
Every outcome in group (a) and (a2), and the carried cluster shares, is a share
*of the public bucket*, so all of them move; the rate outcomes (b) and the
lexicon pilot (c) do not. `analysis/renter_share_outcomes.py` defaults to the
audited role and writes the unsuffixed artefacts; `--role-col cue` reproduces
the published cue run exactly under a `_cue` suffix. The 20 fitted stance
cells, in renter order, **under `role_audit`**:

| route | body | renter | public turns | NIMBY | YIMBY | net |
|---|---|---|---|---|---|---|
| classifier | losaltos | 0.192 | 11 | 0.64 | 0.18 | −0.45 |
| agenda_index | napa | 0.356 | 5 | 0.20 | 0.60 | +0.40 |
| classifier | napa | 0.356 | 11 | 0.36 | 0.36 | 0.00 |
| classifier | petaluma | 0.361 | 15 | 0.53 | 0.40 | −0.13 |
| swagit | petaluma | 0.361 | 36 | 0.17 | 0.64 | +0.47 |
| classifier | concord | 0.386 | 23 | 0.22 | 0.61 | +0.39 |
| classifier | dalycity | 0.395 | 8 | 0.00 | 1.00 | +1.00 |
| classifier | fremont | 0.398 | 6 | 0.50 | 0.00 | −0.50 |
| classifier | healdsburg | 0.410 | 6 | 0.83 | 0.00 | −0.83 |
| local_asr | sanleandro | 0.417 | 176 | 0.47 | 0.44 | −0.03 |
| agenda_index | sausalito | 0.421 | 30 | 0.40 | 0.40 | 0.00 |
| agenda_index | hayward | 0.431 | 10 | 1.00 | 0.00 | −1.00 |
| classifier | sanjose | 0.441 | 28 | 0.43 | 0.43 | 0.00 |
| classifier | paloalto | 0.458 | 26 | 0.69 | 0.23 | −0.46 |
| local_asr | richmond | 0.475 | 28 | 0.36 | 0.50 | +0.14 |
| classifier | sanrafael | 0.482 | 14 | 0.64 | 0.29 | −0.36 |
| classifier | sanmateo | 0.492 | 21 | 0.33 | 0.57 | +0.24 |
| classifier | oakland | 0.583 | 17 | 0.18 | 0.82 | +0.65 |
| agenda_index | santaclara | 0.588 | 25 | 0.12 | 0.52 | +0.40 |
| classifier | santaclara | 0.588 | 53 | 0.32 | 0.43 | +0.11 |

Still 20 cells, but not the same twenty. **bart (9 turns) and vallejo (6) fall
below the 5-turn floor**, their cue-"public" turns being dais by the speakers'
own self-identification — bart was the worst body in the audit at a 10.3 %
disagreement rate — while **healdsburg and napa/agenda_index come in**. The
cue-role table is preserved in `outcomes_by_body_route_cue.csv`: losaltos 17
turns net −0.24, napa/classifier 7 net +0.14, petaluma/classifier 13 net 0.00,
swagit petaluma 20 net +0.50, concord 8 net 0.00, dalycity 8 net +1.00, fremont
7 net −0.43, vallejo 6 net −0.17, sanleandro 179 net −0.03, sausalito 28 net
−0.04, hayward 12 net −0.83, sanjose 41 net +0.07, paloalto 19 net −0.47, bart
9 net +0.33, richmond 20 net +0.10, sanrafael 14 net −0.36, sanmateo 20 net
+0.20, oakland 18 net +0.61, santaclara/agenda 16 net +0.38,
santaclara/classifier 38 net −0.08.

Petaluma and Santa Clara appear twice because two routes read two different
sets of their meetings, and they still disagree (Petaluma net −0.13 classifier
vs +0.47 Swagit; Santa Clara +0.11 classifier vs +0.40 agenda index). That
spread is the instrument, not the body — the reason the correlations below
are computed **within** a route. Note the Swagit-Petaluma cell survives here as
a *turn-level* number while its **cluster-level star is withdrawn** (STANCE-S1
§7b): at the cluster unit the audit widens its interval across zero (+0.53 ★ →
+0.40, 90 % −0.04 to +0.81). Do not read the +0.47 above as a body-level
finding.

## 3. Correlations (`output/renter_share/correlations.csv`)

Spearman and Pearson over bodies within a route, 2,000-rep body-bootstrap
percentile CIs, seed 20261018. Only the classifier route has enough bodies;
`local_asr` (2), `agenda_index` (3) and `swagit_transcript` (1) are reported
as counts and not correlated.

*Corrected 2026-10-18 after role audit*: the stance rows below are now the
`role_audit` correlations (`correlations.csv`); the cue-role values each row
replaces are in brackets and in `correlations_cue.csv`. **No row changes its
conclusion** — every interval still runs from a large negative effect to a
moderate positive one — but the point estimates all move a little further in
the hypothesised direction and `n` shifts as bodies cross the floor.

| route | outcome | n bodies | Spearman (90 %) | Pearson (90 %) | p (Pearson) |
|---|---|---|---|---|---|
| classifier | public NIMBY share | 13 *(was 14)* | **−0.15** (−0.62, +0.34) *(was −0.04)* | −0.28 (−0.62, +0.34) | 0.36 *(was 0.87)* |
| classifier | public NIMBY share, floor 10 turns | 10 *(was 8)* | −0.38 (−0.84, +0.23) *(was −0.43)* | −0.44 | 0.20 |
| classifier | public YIMBY share | 13 | **+0.27** (−0.21, +0.70) *(was +0.20)* | +0.32 | 0.28 *(was 0.39)* |
| classifier | public YIMBY share, floor 10 | 10 | +0.54 (−0.02, +0.90) *(was +0.45)* | +0.56 | 0.09 |
| classifier | public YIMBY − NIMBY | 13 | **+0.20** (−0.27, +0.64) *(was +0.11)* | +0.31 | 0.30 |
| classifier | public − dais NIMBY gap | 12 *(was 14)* | **−0.19** (−0.66, +0.40) *(was −0.14)* | −0.31 | 0.34 |
| agenda_index | public NIMBY share | 4 *(was 3, not correlated)* | −0.20 (−0.78, +1.00) | −0.27 | 0.73 |
| classifier | density min per meeting-h (complete only) | 7 | **+0.68** (+0.06, +1.00) | +0.49 (+0.08, +1.00) | 0.27 |
| classifier | transit min per meeting-h (complete only) | 7 | **+0.64** (−0.06, +0.96) | +0.20 (−0.22, +0.96) | 0.67 |
| classifier | lexicon-pilot NIMBY share (weaker) | 9 | +0.02 (−0.67, +0.68) | −0.19 (−0.72, +0.53) | 0.63 |

The two rate correlations are the only ones whose point estimate is away from
zero, and both are driven by the same two points: BART (0.460 renter, 27.7
transit min/h) and VTA (0.447, 26.4), transit boards whose topic time is close
to a tautology (`WAVE1-RESULTS.md` §5.1). The 90 % band on Spearman reaching
+1.00 is the bootstrap hitting a perfect rank order on resamples of seven
points, not precision.

*Corrected 2026-10-18 after validation*: the sentence "dropping them leaves
five councils and no usable correlation" is **withdrawn** — it has the sign of
the effect backwards. Dropping bart and vta leaves fremont, hayward, napa,
petaluma and santaclara, and on those five the density Pearson **rises from
+0.489 to +0.667** (p 0.22) while Spearman falls from +0.679 to +0.400
(p 0.51); the transit-rate Pearson rises +0.198 → +0.592 (p 0.29). The boards
were *diluting* the Pearson correlation, not driving it. What is true is that
**nothing differs from zero at n = 5, as was already the case at n = 7**
(p 0.27). Note also that one of those "five councils" is Napa, which this file
calls a County Board of Supervisors at §1.

Both sets of figures above are computed on the **as-published** 7-body
complete-coverage set. The same validation fixed the complete-coverage test
(`analysis/wave1_coverage.py`, §1 there), which cuts the complete set from 11
bodies to 5 — so on a rerun today the two rate rows fall to **n = 3** on the
classifier route (bart, napa, vta) and the regenerated
`fig_rate_{density,transit}_classifier.png` carry n = 3, not n = 7. The
`local_asr` and `agenda_index` rate figures drop below the 3-body floor
entirely and have been moved to
`output/renter_share/_stale_pre_coverage_fix/`. **The rate correlations should
not be quoted at all until more bodies finish locating.** Every figure now
marks the transit boards with hollow markers and carries a councils-only panel
where one can be drawn.

Figures, one per outcome per route, body-labelled, point area ∝ public turns
or meeting-hours, all stamped EXPLORATORY:
`output/renter_share/fig_{stance_nimby,stance_yimby,stance_net,stance_gap,rate_density,rate_transit,lex_nimby}_<route>.png`
(16 figures).

## 4. Regression (`output/renter_share/models/`)

`analysis/renter_share_models.py`, PyMC 6.3.2, 4 chains × 2,000 draws after
1,500 tuning, `target_accept` 0.95, seed 20261018.

**Model 1 — binomial GLM on the NIMBY count.** `y_i ~ Binomial(N_i, p_i)`,
`logit p_i = α + β_renter·z_renter + γ_route(i) + u_i`, with
`u_i ~ Normal(0, σ_cell)` the body-level overdispersion term and route a
fixed-effect stratum (reference = classifier, the largest). `α ~ N(0, 1.5)`,
`β ~ N(0, 1)`, `γ ~ N(0, 1)`, `σ_cell ~ HalfNormal(0.75)`. **n = 20 cells
over 18 bodies, 500 public turns.**

**Model 2 — negative binomial with exposure.** `y_i ~ NB(μ_i, φ)`,
`log μ_i = log E_i + α + β_fam(i) + β_renter·z_renter + γ_route(i) + u_body(i)`,
`u_body ~ N(0, σ_b)`, `1/φ ~ Exponential(1)` — the same shape and the same
priors as `analysis/models/topic_time.py` (`analysis/models/README.md` §1),
with the month walk, the body × family interaction and the zero-inflation
dropped (28 cells cannot identify them alongside a body-level covariate) and
renter share added at the body level. This is a reduced re-fit for one
coefficient, **not** a replacement for `make models`.

β is per **1 sd** of renter share (sd ≈ 0.09 over the fitted cells); the
per-10-point column converts.

| model | n cells | β_renter median | 90 % | 95 % | Pr(β>0) | per +10 pt | R̂ | ESS | div | PPC |
|---|---|---|---|---|---|---|---|---|---|---|
| NIMBY share ~ renter (route stratum) | 20 | **−0.17** | −0.58, +0.24 | −0.67, +0.32 | 0.24 | −0.19 log-odds | 1.0017 | 2433 | 0 | 20/20 |
| NIMBY share ~ renter + log income + log density | 20 | −0.13 | −0.58, +0.32 | −0.68, +0.42 | 0.32 | −0.14 | 1.0024 | 2189 | 0 | 20/20 |
| NIMBY share ~ renter (floor 10 turns) | 14 | −0.26 | −0.70, +0.16 | −0.80, +0.25 | 0.15 | −0.26 | 1.0015 | 2355 | 0 | 14/14 |
| YIMBY share ~ renter (route stratum) | 20 | **+0.28** | −0.10, +0.69 | −0.18, +0.77 | 0.89 | +0.31 | 1.0012 | 1843 | 0 | 20/20 |
| topic min/h ~ renter (complete coverage) | 28 | **+0.23** | −0.46, +0.87 | −0.60, +1.01 | 0.73 | ×1.37 rate ratio (0.53, 3.32) | 1.0016 | 2016 | 0 | 28/28 |
| topic min/h ~ renter (+ partial, flagged) | 62 | −0.06 | −0.47, +0.33 | −0.56, +0.41 | 0.40 | ×0.93 (0.57, 1.48) | 1.0021 | 2222 | 0 | 61/62 |
| density min/h ~ renter (all coverage) | 31 | −0.12 | −0.47, +0.19 | −0.54, +0.26 | 0.27 | ×0.87 (0.58, 1.26) | **1.0213** | **152** | **8** | 29/31 |

Every interval covers zero. The best-supported direction anywhere is the
YIMBY one (Pr(β>0) = 0.89, i.e. an 89 % posterior that more renters means
more YIMBY public comment) and 0.89 is not a finding at this n — it is what a
flat posterior looks like when the median happens to sit off zero. The
**last row did not converge** (R̂ 1.021, 8 divergences, min ESS 152: a
single-family fit with 31 cells and a body random effect over 26 bodies) and
should not be quoted; it is in the table so the failure is on the record.
PPCs are clean on the fits that converged: every cell's observed NIMBY count
sits inside its replicated 90 % band (`models/fig_ppc_binomial.png`).

**Plain OLS sanity check** on the shares, same cells, no hierarchy: NIMBY
share slope **−0.27 per unit renter share** (95 % −1.42, +0.88), YIMBY
**+0.51** (−0.72, +1.75); with log income and log density added the renter
slope is **−0.20** (95 % −1.48, +1.08; `model_results.json →
ols_with_controls`). Same sign, same non-conclusion, so the Bayesian result is
not an artefact of the priors. *Corrected 2026-10-18 after validation*: this
read −0.29, which is in no artefact; −0.2713 is the *uncontrolled* slope
quoted in the previous sentence.

Figures: `models/fig_beta_renter_forest.png` (all seven β_renter posteriors
with 90/95 % intervals), `models/fig_ppc_binomial.png`. Tables:
`models/model_table.csv`, `models/model_results.json`.

## 5. Limitations

1. **n.** 18 bodies with scored public comment, 20 body × route cells, 500
   public turns — a median of 18 turns per cell, and six cells under 10. The
   rate side is worse: 7 complete-coverage bodies on the one route that has
   more than three. Every interval here is wide for that reason alone, and no
   amount of modelling fixes it.
2. **Ecological inference.** Renter share is the renter share of a
   jurisdiction's *residents*. The outcome is what the people who *came to
   the meeting and spoke* said. Nothing in this corpus links the two, and the
   literature on who attends local meetings says they differ sharply
   (homeowners over-attend). A null here is consistent with a real
   resident-level relationship that meeting attendance filters out.
3. **The stance instrument is exploratory.** The κ ≥ 0.6 hand-coding gate is
   not passed (`STANCE-S1-RESULTS.md` §10); the YIMBY/NIMBY mapping flips
   sign on items that are themselves restrictions (rent control, moratoria),
   which is most of San Leandro's 176 turns — the single largest cell.
3a. **The role is an audit, not a gold label** (*added 2026-10-18 after role
   audit*, `plan/ROLE-AUDIT-2026-10-18.md` §8). Since 2026-10-18 the public
   bucket here is `role_audit`, which prefers the codebook's SELF_ID over the
   diarization-cue role. SELF_ID is model output and no human has checked it;
   **44 % of turns are `self_id = unknown`** and keep the cue role, so the
   public bucket is still contaminated at an unknown rate. The audit is a
   **re-partition in both directions**, so differences from the `_cue` numbers
   are not "contamination removed". This bites harder here than in STANCE-S1
   §7b, because this file's unit is the *turn* and its per-cell n is small: two
   bodies cross the 5-turn floor in each direction on the role column alone.
4. **Coverage.** The rate outcomes are only comparable within the
   complete-coverage set; partial bodies' rates are floors and are flagged in
   the one sensitivity that uses them. Selection into the corpus is by
   platform, not by politics (`WAVE1-RESULTS.md` §7).
5. **Routes are instruments, not variables.** Correlations are within-route
   and the models carry route as a fixed-effect stratum; Petaluma and Santa
   Clara each contribute two cells that disagree with each other.
6. **Geography is approximate for the boards.** A transit district's service
   area is not the union of its member counties' residents, and the BART
   three-county union excludes San Mateo, which BART serves. The board rows
   drive the one non-null correlation and are the least trustworthy
   covariates in the table.
7. **ACS vintage.** 2019–2023 5-year estimates against meetings held in a
   12-month window ending 2026; renter share moves slowly but not none at all,
   and the 5-year estimate is a 2021 midpoint. MOEs (`B25003_M003`) are
   carried in `acs.json` and are **not** propagated into the models.
8. **Multiplicity.** Seven outcomes × routes were correlated and seven models
   fitted; nothing is corrected for that, and nothing needs to be, since
   nothing crosses a threshold.

## 6. Files

New: `plan/RENTER-SHARE.md` (this), `analysis/renter_share_acs.py`,
`analysis/renter_share_outcomes.py`, `analysis/renter_share_models.py`,
`output/renter_share/` — `acs.json`, `covariates.csv`,
`outcomes_by_body_route.csv`, `correlations.csv`, `outcomes.json`,
`fig_*.png` (16), `raw/` (5 Census files + `dl.log`, `gaz.log`),
`acs_build.log`, `outcomes.log`, `models.log`, and `models/`
(`model_results.json`, `model_table.csv`, `fig_beta_renter_forest.png`,
`fig_ppc_binomial.png`).

Modified: nothing. No database was written. Nothing committed.

**Role-audit adoption, 2026-10-18** (`plan/ROLE-AUDIT-2026-10-18.md`,
`plan/VALIDATION-2026-10-18.md` "Role audit applied"). Modified:
`analysis/renter_share_outcomes.py` (`--role-col {audit,cue}`, default audit;
the role column also selects which `nimby_view` cluster-share table the
carried `cl_YIMBY` / `cl_NIMBY` columns come from), and §0 / §2 / §3 / §5.3a
of this file. Regenerated under `role_audit` (the previous contents were the
cue run and are preserved with a `_cue` suffix):
`output/renter_share/{outcomes_by_body_route,correlations}.csv`,
`outcomes.json`, `fig_{stance_nimby,stance_yimby,stance_net,stance_gap,
rate_density,rate_transit,lex_nimby}_<route>.png`. The §4 regression
(`models/`) was **not** refit under the audited role and its β_renter
posteriors remain cue-role; nothing in §4 is quoted as a result, and the
§3 correlations it mirrors all keep their conclusion under the audit. No new
spend, no DB write, nothing committed in `~/ytpt`.

## Corrections 2026-10-19

Applied from `plan/REGRESSION-REVIEW-2026-10-19.md`, which reviewed runs 2–4
and, through them, this one. **Nothing in this document was recomputed and no
number was replaced**; §0 carries a dated qualification and this section
records what the review changed downstream.

| # | what changed | where | why |
|---|---|---|---|
| 1 | §0's answer **confirmed, with the leave-one-body-out qualification attached** | §0 | The public YIMBY near-result that runs 2–3 later found is one body (Santa Clara, highest renter share, three route cells): +0.344 → +0.096 without it, and a sign flip under eight covariates. Run 1 never claimed it; the qualification belongs here anyway, because this is the document the finding is cited from. |
| 2 | "renter share does not predict what is said" kept; **"and nor does anything else" marked unsafe** | §0 | `transit_commute` survives every valid null (`plan/RENTER-SHARE-2.md` §3d-perm) and `log_population` clears 95 % on all-roles YIMBY (§3a-full). Run 1's §0 does not make the stronger claim, but the runs that cite it did. |
| 3 | **all intervals in this family declared ~1.4× too narrow** | §0, and `plan/RENTER-SHARE-3-ROLE.md` §5.10 | The NIMBY indicator's ICC is 0.275 by episode (3.82 clusters/episode) and 0.156 by meeting (9.60); design effects 1.77 and 2.34. §4's binomial and every later overdispersion term treat a speaker cluster as an exchangeable trial, and 86.3 % of public stance-bearing clusters rest on a single scored turn. Widening strengthens this document's null. |
| 4 | §4's unrefitted cue-role regression stays **not quoted** | §4, §6 | Unchanged by the review and still correct: the §4 posteriors are cue-role and were never refit under `role_audit`. The review's findings 1, 3 and 7 (print every parameter; the PPC cannot fail; use grouped K-fold) apply to it in principle and are implemented in runs 2–3 instead, on the cluster unit that superseded the turn unit. |

**What the review did not touch.** The §1 geography and ACS provenance, the §2
role-audit adoption and the §3 Spearman correlations, all of which the review
found consistent with runs 2–4. Its "three things done right" are this
family's limitations sections naming the actual killers — ecological
inference, `role_audit` as model output, the unpassed κ gate — rather than
hedging.
