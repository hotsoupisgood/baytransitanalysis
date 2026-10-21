# Independent statistical review — the renter-share regressions (2026-10-19)

Recomputed from `speaker_clusters_roleaudit.csv`, the published
`model_table.csv`s and the live state DB (`~/.local/share/ytpt/state.sqlite` — the
repo's two `.sqlite` files are 0 bytes). Refits in PyMC 6.3.2, 4×1,000 draws.
Read-only, no spend, nothing committed; scratch in `/tmp/rsreview/`. No
`RENTER-SHARE-4-SUBSETS.md` existed yet.

## Ranked findings

**1 (critical, reporting). "Role is the only parameter whose interval excludes
zero" is false against the run's own artefact** (`RENTER-SHARE-3-ROLE.md:215`,
`:339`). In `output/renter_share_3/models/model_table.csv`, YIMBY outcome:
`transit_commute` +0.337 (95 % +0.029, +0.630) and `log_population` +0.304 (95 %
+0.054, +0.577) clear 95 % in both R1 and R2; `nhwhite_share` +0.277 clears 90 %.
§3a tabulates only role, renter, income and the interactions. Run 4 flags the
same column. *Remedy:* print every parameter; rewrite §3a/§5.9. 1 h, $0.

**2 (critical, construct). δ = +2.11 is ~60 % the `neutral` class and ~20 %
staff.** `neutral` sits in the denominator and is a dais phenomenon: 69.1 % of
stance-bearing dais clusters vs 10.5 % public. R1 refitted unchanged on a
NIMBY-vs-YIMBY denominator: **δ = +0.839 (90 % +0.510, +1.192)** — OR 2.3, not
7.9. Staff clusters (843, folded into dais by ra-1) are 4/671 NIMBY and 94 %
neutral; dropping them moves the raw gap 2.07 → 1.61. Elected vs resident, the
**YIMBY gap is +0.11** (30.4 % vs 32.7 %) against +1.03 published. It survives
stratifying by scored-turn count (1.84/2.01/1.30), so aggregation is minor.
*Remedy:* δ on the directional denominator, staff excluded, plus the
elected-vs-resident subset. 30 min, $0.

**3 (high). The PPC cannot fail.** `u_i` is one free parameter per row, so the
band is centred on the observed count (`renter_share3_role_models.py:303-306`;
"64/64", "29/29", "35/35", "20/20"). Refit without `u_i`: **60/64**, i.e.
calibration; under a calibrated band P(64/64) = 0.9^64 = 0.0013. *Remedy:*
LOO-PIT, or cell means with `u_i` integrated out. 1 h, $0.

**4 (high). A cluster is not an exchangeable trial.** 86.3 % of public
stance-bearing clusters rest on one scored turn (dais 53.1 %); words span 64–702
(p10–p90). Clusters nest: ICC of the NIMBY indicator **0.275 by episode**
(3.8/episode), **0.156 by meeting** (9.6) → design effects 1.8–2.3, which `u_i`
(a mean shift) does not absorb; run 4's own ĉ = 1.99 agrees, and the two cells
carrying the design are the most inflated. *Remedy:* turn-level model nesting
episode in meeting in body, or beta-binomial; meanwhile the by-body SEs are
primary, not a "sanity check". 1 day, $0.

**5 (high). The horseshoe result is the prior.** τ₀ = 0.124 by construction, so
τ ≈ 0.03–0.10 sits in the prior bulk, not "the posterior saying there is nothing
here" (`RENTER-SHARE-2.md:33`, `:464`). Refitting (c) public YIMBY with covariate
vectors **permuted across bodies** — a true null — reproduces the published
picture: τ 0.039 vs 0.061, β_renter −0.010 vs +0.016. τ₀×5 moves
`transit_commute` 0.119 → 0.281; τ₀/5 flattens all to ≤0.013. *Remedy:* report τ₀
and the permuted null; drop "nothing survives regularisation". Done.

**6 (high). One body decides the sign.** Leave-one-body-out, IRLS, same design:
run 2 base public YIMBY β_renter +0.344 → **+0.096** without Santa Clara (top
renter share, three route cells); public NIMBY −0.271 → **−0.043**. In the role
model the public slopes flip: YIMBY +0.222 → −0.026, NIMBY −0.181 → +0.030. The
other 21 bodies move it <20 %. *Remedy:* publish the LOBO column. Done.

**7 (med-high). K-fold by body reverses §3d.** Grouped 5-fold, 29 public cells,
unpenalised logit (so big models are rightly punished): route-only ELPD −116.8
beats renter+income −135.5, +density −174.5, all-eight −158.6; same order on
YIMBY. *Remedy:* grouped K-fold replaces PSIS-LOO; drop "horseshoe ranks first".
2 h, $0.

**8 (medium). Multiplicity, counted.** 52 substantive intervals in run 2, 64 in
run 3: **21 exclude zero at 90 %** against ~12 expected under a global null, 16 of
them four covariates repeated across nested fits. "Nothing needs correcting"
survives that, but run 2 §5.10's "three exclusions" is five and run 3's "one
parameter clears 95 %" is three. A body-level permutation null gives **p = 0.052**
(public YIMBY), 0.244 (public NIMBY), single-test.

**9 (medium). The `log_population` dismissal is unverified:** across the 35
all-roles cells corr(log_population, all_n) = **+0.05**, and the biggest
suppliers are Santa Clara and San Leandro, not the largest places. "Not
separable", then — not "a corpus artefact".

**10 (medium). Run 4's permutation is not a valid null for a partial
coefficient:** it permutes `renter_share` alone while seven collinear covariates
stay fixed. Use Freedman–Lane, or permute the whole body covariate vector.
Permuting role within body is also invalid — it breaks the cells identifying δ.

**11 (low-med). Two estimands are mislabelled.** `renter_share ~
multiunit_5plus` r = +0.900, so β_renter is renter share *net of multi-unit
stock*; `log_income` is forced in after its four collinear partners go (VIF
188 → 1.74), so β_income is *residual* income. Separately, one `u_i` per row
saturates the likelihood, so every coefficient rests on `σ_row ~
HalfNormal(0.75)` — benign on test (δ 2.094 / 2.110 at HN(0.25) / HN(2.0)) but
unstated.

## Verdict

**The null on renter share stands, and is weaker than advertised** — the one
near-result rests on a single body (6), a prior-dominated regulariser (5) and a
check with no power (3). "Renter share does not predict what is said" is safe
here; "and nor does anything else" is **not** (1). **The role effect is right in
direction and wrong in magnitude as quoted:** "public NIMBY odds ~8× dais" is
mostly the `neutral` class plus staff reports; among clusters that take a side it
is ~2.3×. "Partly definitional" understates it — it is *mainly* definitional.

## Three things done right

1. The limitations name the actual killers — ecological inference, `role_audit`
   as model output, the unpassed κ gate — instead of hedging.
2. Run 3 answered the question asked, with a paired-only sensitivity fit and
   by-body cluster-robust SEs beside HC3.
3. No drift between runs, and the LOO degeneracy was diagnosed against interest.

## Before quoting any of this

1. Drop "8×" and "role is the only parameter"; quote δ on the directional
   denominator without staff, and list the 95 % survivors.
2. Attach the leave-one-body-out column to every renter-share number.
3. Replace the PPC counts and PSIS-LOO with LOO-PIT and grouped K-fold.
4. Where the horseshoe is called evidence of absence, add "τ₀ = 0.124, and a
   permuted null looks the same".
5. Nothing is meeting-clustered: intervals are ~1.4–1.5× too narrow on the two
   cells that carry the design.

## Review applied

**2026-10-19, same day, by the supervisor agent.** All eleven findings and all
five "before quoting" items are implemented. Read-only on the DB and on
`output/renter_share_{2,3,4}/`; no YouTube request, no GPU, no Claude call, no
API key, **$0**; no DB write; `.env`, `plan/frame.csv` and `plan/DECISIONS.md`
untouched; **nothing committed** in `~/ytpt` or `~/ytpt-site`. Every prior
number in the four documents is kept with a strikethrough or a "superseded"
note — nothing was silently replaced.

New scripts: `analysis/renter_share5_role_restated.py`,
`analysis/renter_share5_robust.py`, `analysis/renter_share5_ppc_horseshoe.py`.
New artefacts: `output/renter_share_5/{role_restated,robust,ppc_horseshoe}/`
and the three run logs.

| finding | implemented as | headline result |
|---|---|---|
| 1 print every parameter | `RENTER-SHARE-2.md` §3a-full (52 intervals), `RENTER-SHARE-3-ROLE.md` §3a-full (64) | 95 % survivors: role, `transit_commute` +0.337, `log_population` +0.304 (run 3 YIMBY); `log_population` +0.266, **95 % +0.002, +0.492**, in run 2 — so run 2's "no interval excludes zero at 95 %" was also false. `nhwhite_share` +0.280 at 90 %. |
| 2 δ on the directional denominator | `RENTER-SHARE-3-ROLE.md` §0a, three refits | **δ = +0.839 (90 % +0.499, +1.201), OR 2.31**; staff excluded **+0.784, OR 2.19**. Reproduces the review's +0.839 to 3 dp. Elected-vs-resident on the same denominator is *larger* (+1.700, OR 5.47) and is reported but not quoted, because `self_id = resident` selects on a correlate of the outcome. |
| 3 PPC replaced | `RENTER-SHARE-3-ROLE.md` §3c-cal, all ten fits | With `u_i` integrated out: 63/64, 63/64 (run 3) and 28–35 of 29–35 (run 2) against 90 % expected; PIT indistinguishable from uniform, KS p 0.23–0.93. Still over-covered — "nothing contradicts the model", not "the model fits". |
| 4 clustering | ICCs added (`RENTER-SHARE-3-ROLE.md` §5.10, `RENTER-SHARE-2.md` §5.11); by-body cluster SE made primary (run 3 §4) | ICC **0.275** by episode (m̄ 3.82), **0.156** by meeting (m̄ 9.60) — reproduces the review exactly; design effects 1.77 / 2.34, SEs **×1.33–×1.53** too narrow. Turn-level nested model documented as the next step, not run. |
| 5 horseshoe is the prior | `RENTER-SHARE-2.md` §3b-prior | **τ₀ = 0.1238 stated.** Permuted-across-bodies null: τ 0.0369, β_renter −0.006 against observed 0.0666 / +0.021. τ₀ ×5 → `transit_commute` +0.343; τ₀ ÷5 → all eight ≤ 0.015. Every "the posterior says there is nothing here" reworded to **"indistinguishable from the prior at this n"**. |
| 6 LOBO | `RENTER-SHARE-2.md` §3a-lobo (16 estimates), `RENTER-SHARE-3-ROLE.md` §3b-lobo (16) | **Santa Clara** is the extreme on 16 of 32; public YIMBY +0.344 → +0.096, public NIMBY −0.271 → −0.043, both sign-flipping under eight covariates. **San José** owns `transit_commute` (−0.473 → −0.119). **Los Altos** owns the all-roles renter slopes. **San Leandro is not an extreme anywhere** — correcting the expectation that it would be. δ moves 7–16 % and never nears zero. |
| 7 grouped K-fold | `RENTER-SHARE-2.md` §3d-kfold, `RENTER-SHARE-3-ROLE.md` §5.8 | **Route strata alone beat every covariate set on all four run-2 outcomes** and on run 3 NIMBY; public YIMBY −93.8 (route only) against −107.8, −133.5, −237.6. Ordering reproduces the review at different magnitudes (different fold assignment). "The horseshoe ranks first" dropped. |
| 8 multiplicity | `RENTER-SHARE-2.md` §5.10 table | 116 substantive intervals over 18 fits: **21 exclude zero at 90 %** against 11.6 expected, **12 at 95 %** against 5.8. Run 2 alone is at the null rate (5 vs 5.2; 1 vs 2.6); the excess is all run 3 and is four parameters repeated across nested fits. Run 2's "three exclusions" is five; run 3's "one parameter clears 95 %" is three. |
| 9 `log_population` | `RENTER-SHARE-2.md` §0, §4 | "A corpus artefact" → **"not separable from corpus composition at n = 25"**. |
| 10 permutation | `RENTER-SHARE-2.md` §3d-perm; `RENTER-SHARE-4-SUBSETS.md` §0, §4a | Freedman–Lane **and** whole-vector body permutation, 5,000 draws each. Renter share: one p < 0.05 in twelve rows (public YIMBY base, 0.046 / 0.095), gone with the other six covariates in (0.51 / 0.35). `transit_commute`: **p < 0.05 in all eight rows**, < 0.01 in three. The within-body role swap is **withdrawn** as a test of δ, and no across-body role permutation exists (every body has the same role composition), so **δ rests on the model and the cluster-robust SE**. |
| 11 mislabelled estimands | `RENTER-SHARE-2.md` §4, Corrections | β_renter relabelled **renter share net of multi-unit stock** (r = +0.900); β_income relabelled **residual income** (VIF 188 → 1.74). The `σ_row ~ HalfNormal(0.75)` dependence is now stated. |

Also updated: `docs/RESUME-BRIEF.md` (the role-gap findings line — hostility
keeps "odds ratio about 5 to 8"; the NIMBY share gap becomes ~2× among
clusters that take a side). **`plan/STANCE-S2.md` and `~/ytpt-site` quote no
"8×" or "eight times" anywhere** — grepped, no match — so neither was changed
and nothing was committed to the site.

**The verdict is accepted in full**, including the part that is not a
correction: "renter share does not predict what is said" is safe here, "and
nor does anything else" is not, and the role effect is right in direction and
was wrong in magnitude as quoted.
