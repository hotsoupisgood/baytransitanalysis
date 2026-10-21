# analysis/models — the two showcase Bayesian models

`plan/SHOWCASE.md` §1 / `plan/SHOWCASE-EXEC.md` WP D. Everything here is
**PRELIMINARY**: the corpus freezes 2026-10-24 (more episodes will be cut and
transcribed before then) and estimand 2 runs on a 300-turn stance *pilot*
(`plan/SHOWCASE.md` 4a), not the full stance pass (4b, ~6,600+ turns, not yet
run). Re-run `make models` after both land; nothing here should be quoted as
a final number.

Run: `make models` (`MODELSOUT=` to redirect; `.venv/bin/python
analysis/models/run.py --draws --tune --chains --no-sensitivity` for manual
control). Runtime on the current corpus: ~40 s total (default
1000 draws / 1000 tune / 4 chains, PyMC's own NUTS — no GPU, no network, well
inside the 15-minute target). `make test` runs the regression checks
(`analysis/test_pipeline.py` section 32, ≥5 checks: both facet guards,
negative-binomial recovery of a planted rate, the ordinal path on a synthetic
`turn_stance` table, `cap_by_body`, `bin_standin`).

## 1. Estimand 1 — topic time

**Unit**: a cell = body × month × family × locate route. Family is
density/transit (`about_both` episodes credit minutes to **both** families by
default — the alternatives, exclude or a third `both` level, are documented
sensitivities not run by default; see `load.cell_table`). Route
(`episodes.method`) is an instrument, not a variable (`analysis/facets.py`):
a rate that mixes routes is a rate of nothing in particular.

**Exposure**: meeting-hours of every `kind='meeting'` row a route actually
read (`status` in `cut`/`no_episodes`, known duration). `captioned` /
`discovered` meetings are missingness, not a zero.

**Outcome**: minutes of every `located` episode (not `done` — `done` depends
on which GPU batch has run and would make the outcome move for a reason
unrelated to the topic).

**Model** (the primary fit is `load.cell_table_pooled(pool=True)`: routes
enter as a fixed-effect stratum, explicitly, stamped `POOLED` in every
filename/title/footnote — never pooled silently):

```
y_i        ~ ZeroInflatedNegBin(psi_i, mu_i, phi)          [default; --no-sensitivity keeps it]
logit psi_i = zi_a + zi_b * (log E_i - mean log E)
log mu_i   = log E_i + alpha + beta_f(i) + u_b(i) + v_b(i),f(i) + gamma_r(i) + w_f(i),t(i)
beta, w    sum-to-zero over family / over month within family
```

`E_i` = exposure-hours, `b` = body, `f` = family, `r` = route (reference =
largest route), `t` = month. `u_b` is the body random intercept (what the
shrinkage plot and the ranked-body figure show), `v_{b,f}` a body×family
interaction, `w_{f,t}` a month random walk per family. A zero-inflation term
distinguishes "the route read this body-month and found nothing" from "a
short one" — a body-month a route never read is not in the data at all
(that's the exposure definition), so `psi` is about active cells only.

**Priors** (one line each — `topic_time.PRIORS`):

| parameter | prior | why |
|---|---|---|
| `alpha` | Normal(1, 1.5) | e¹≈2.7 min/meeting-hour ≈ T2's ~5% topic-time expectation; sd 1.5 spans 0.2–30 min/h |
| `beta` (family) | Normal(0, 1.5), sum-to-zero | a family may differ by up to a 4.5× factor at 1 sd; no direction assumed |
| `sigma_b` | HalfNormal(1) | between-body sd; 95% mass under a 7-fold ratio spread, wider than any plausible council-to-council gap |
| `sigma_v` | HalfNormal(0.5) | body×family interaction, half the main-effect scale |
| `gamma` (route) | Normal(0, 1.5), ref = largest route | recall differences enter as a stratum, not a prior claim about direction |
| `sigma_w` (month walk) | HalfNormal(0.3) | ~35% monthly change at 1 sd, generous for a 13-month window |
| `phi` (1/phi) | Exponential(1) | most mass near Poisson, long tail — overdispersion learned, not assumed |
| `zi_a` | Normal(1, 1.5) | centred on 73% observed non-zero share; sd reaches 10–99% |
| `zi_b` | Normal(0, 1) | one log-hour of extra exposure may move active-probability by ~e at 1 sd |

**Sensitivity**: `topic_time.fit_per_route` fits each route alone (never
pooled; routes with < `MIN_CELLS`=12 cells are reported, not fitted) and
feeds the per-route rows in the contrast figure.

### Current fit (preliminary, current corpus)

220 pooled cells (360 before pooling, across `classifier` / `agenda_index` /
`local_asr` / `swagit_transcript`). `max_rhat`=1.010, 0 divergences, min
ESS≈630 (4000 post-tune draws) — converged.

5 highest posterior body rates (minutes located per meeting-hour, 90% CI):

| rank | body | median | 90% CI |
|---|---|---|---|
| 1 | bart | 11.2 | 4.9–27.8 |
| 2 | sanmateo | 8.9 | 4.2–23.8 |
| 3 | vta | 7.6 | 3.6–20.9 |
| 4 | losaltos | 7.2 | 3.2–18.3 |
| 5 | petaluma | 7.0 | 3.4–17.3 |

5 lowest:

| rank | body | median | 90% CI |
|---|---|---|---|
| 26 | oakland | 2.1 | 0.7–5.9 |
| 25 | richmond | 2.5 | 0.7–6.4 |
| 24 | fremont | 4.5 | 2.4–8.0 |
| 23 | livermore | 4.5 | 1.3–11.5 |
| 22 | healdsburg | 4.5 | 1.7–11.0 |

Density/transit contrast (rate ratio, transit / density, mean body): 0.81
(90% CI 0.52–1.25), Pr(ratio > 1) = 0.20 — no detectable difference yet at
this corpus size; most of the spread is between bodies, not between
families. Route strata (rate ratio vs. `classifier`, the largest route):
`local_asr` 0.90 (0.42–2.45), `agenda_index` 0.77 (0.45–1.39),
`swagit_transcript` 1.75 (0.66–4.17) — all consistent with 1, i.e. no route
is yet shown to over/under-recall relative to another, but the CIs are wide
(13–45 cells per route).

Outputs: `tables/POOLED_topic_time_{bodies,contrast,months,routes,hyper,ppc}.csv`,
`tables/topic_time_bodies_route-<route>.csv` (per-route sensitivity, not
pooled), `figs/POOLED_topic_time_{shrinkage,body_forest,contrast,months,ppc}.png`.

## 2. Estimand 2 — role gap

**Question**: is public comment more opposed / more hostile than the dais,
on the same episode? **Unit**: a turn (`analysis/a1_sentiment.py`'s
definition — a maximal run of segments by one speaker cluster inside one
episode). Tier A only: anonymous clusters, no names (`DECISIONS.md` #15).

**Model**: cumulative-logit (proportional-odds) ordinal regression,

```
P(y_i <= k) = logistic(c_k - eta_i)
eta_i = beta_role * public_i + beta_fam[f(i)] + beta_int[f(i)] * public_i
        + beta_len * (log words_i - mean) + beta_strat[s(i)] + a_body[b(i)] + a_ep[e(i)]
```

`y` is `stance_median` (K=5, −2…2) or `hostility_median` (K=5, 0…4). `s(i)`
is the locate route when it varies within the data passed in (a stratum, not
a facet level — the facet, `role_source`, is swept outside the model: a run
spanning two role instruments is refused, `role_gap.fit`). `a_body`,
`a_ep` are body/episode random intercepts (the item being discussed moves
everyone in the room).

**Priors** (`role_gap.PRIORS`):

| parameter | prior | why |
|---|---|---|
| `cutpoints` | Normal(0, 3), ordered | sd 3 lets a category hold anything from <1% to >95% of turns |
| `beta_role` | Normal(0, 1.5) | odds ratio of 4.5 at 1 sd; no direction assumed |
| `beta_fam` | Normal(0, 1.5), ref density | family main effect, same scale |
| `beta_int` | Normal(0, 1.0) | role×family interaction, tighter: the gap may differ by family, less than it exists |
| `beta_len` | Normal(0, 1.0) per log-word | length control the stand-in needs (A1-RESULTS 1); a check on the real pass |
| `beta_strat` | Normal(0, 1.5), ref largest level | locate-route stratum inside one role instrument |
| `sigma_body` | HalfNormal(1) | 95% under a 2-logit spread between bodies |
| `sigma_ep` | HalfNormal(1) | between-episode sd within body |

**Data, real vs. stand-in**. `load.read_turn_stance` reads the real
`turn_stance` table (`pipeline/stance.py`, WP A) when it exists.
`load.standin_turns` / `bin_standin` build the same shape from VADER
compound (turns ≥ `MIN_WORDS`=100 words, A1-RESULTS 1: VADER is mostly a
length measure below that, binned into 5 quantile levels); every stand-in
row and figure is stamped `STAND-IN: VADER, not stance`. Either source is
capped at `CAP_H`=8 h of episodes per body, drawn at random
(`load.cap_by_body`, `SHOWCASE.md` 1b), so Santa Clara/Petaluma's much larger
transcribed hours cannot dominate the estimate.

**Current state (2026-10-17)**: `turn_stance` **exists** but holds only the
300-turn **pilot** sample (`SHOWCASE.md` 4a, for Puck/Wren's blind-coding
agreement check), not the full pass (4b). The fit below ran on it because
the facet guard correctly treats "the real table" and "the stand-in" as the
only two states it knows about — it has no third state for "real but tiny".
**Treat this fit as a pipeline-readiness demonstration, not a result**: 300
raw rows → 177 usable after dropping NULL stance and non-`dais`/`public`
roles, which is below where a per-body or per-route breakdown would be
trustworthy. `role_source` is not carried on `turn_stance` rows, so the
facet sweep had one level, `all`. Re-run after 4b.

### Current fit (preliminary, pilot-sized)

`stance_all`: max_rhat=1.009, 0 divergences, min ESS≈541 — converged despite
the small n. `hostility_all`: max_rhat=1.001, 0 divergences, min ESS≈1748.

Turn-weighted gap (public − dais, latent logit scale), stance: median 0.005
(90% CI −0.64 to 0.65), Pr(gap > 0) = 0.51 — indistinguishable from zero, as
expected at n=177. Per family: density −0.03 (−0.80 to 0.74), transit 0.20
(−0.61 to 1.05), both −0.55 (−2.03 to 0.92, n=21, the "both" family cell is
tiny). No usable signal yet; the pilot was sized for coder agreement, not for
this model.

Outputs: `tables/role_gap_{gap,coef,bodies,ppc}_{stance,hostility}_all.csv`,
`figs/role_gap_{forest,ppc}_{stance,hostility}_all.png`. Filenames/titles
will say `STAND-IN: VADER, not stance` instead of `all` once that path is
exercised (it isn't, right now, because `turn_stance` exists).

## 3. What changes at the freeze / after the full stance pass

- **2026-10-24 freeze**: more located/cut/transcribed episodes land before
  then (`SHOWCASE.md` 1). Re-run `make models`; body ranks, the
  density/transit contrast, and route strata CIs should narrow.
- **Stance pass 4b**: once `turn_stance` holds the full run, `role_gap`'s
  facet sweep over `role_source` becomes meaningful again only if that
  column is added to the table; as written, `role_gap.fit` runs on whatever
  `turn_stance` holds with no facet split. If the stance pass later needs a
  role-instrument split, add `roles_source` to the `turn_stance` write (WP A)
  or join it in `load.read_turn_stance`.
- **Known facet gaps not modelled**: the `both`-family double-crediting
  sensitivity (`exclude` / third-level `both`), the `done`-outcome
  sensitivity for estimand 1, and a `role_gap` run on the uncapped (> 8 h)
  turn set are documented in the code (`load.py`, `topic_time.py`
  docstrings) but not run by default — `make models` runs the primary fits
  only, to stay well inside the 15-minute budget.
