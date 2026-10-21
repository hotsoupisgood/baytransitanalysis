# Independent validation, 2026-10-18

Sceptical reproduction of the claims in `WAVE1-RESULTS.md`, `STANCE-S1-RESULTS.md`,
`NIMBY-PILOT.md`, `CLUSTER-PILOT.md`, `RENTER-SHARE.md`, `RECALL-CHECK.md`,
`CODEBOOK-V2-PILOT.md` and `docs/COSTS.md`. Everything recomputed from raw rows;
no claim accepted on the write-up's word. **$0: no API call, no YouTube, no pod,
no GPU.** Read-only except this file. Nothing committed.

Method note: the live `state.sqlite` is being written continuously (a meeting
changed status mid-query during this check). All numbers below come from a
frozen snapshot taken 2026-10-19T01:42:54Z (`sqlite3 .backup`). Where a doc
figure is compared to its own saved artefact the comparison is drift-free and
labelled as such.

## 1. Rates — VERIFIED against artefact, three DISCREPANCIES

All 14 cells of the §5 table match `output/wave1/rates_by_body_route.csv`
**exactly** (BART 27.72, VTA 26.35, San Leandro density 8.00, …). The doc
reports its own run faithfully. Denominators reproduce exactly too (BART 16
meetings / 67.7 h, VTA 11 / 44.7).

- **"Density outweighs transit on every council" is false on the doc's own
  table** (`WAVE1-RESULTS.md:184`). Hayward classifier is 0.1 density vs 0.4
  transit; Petaluma-Swagit is 7.2 vs 8.2. The doc's supporting enumeration
  lists only the four councils that do fit. → "on every council on the
  classifier route except Hayward, and not on Petaluma's Swagit route".
- **The 11-body complete-coverage set is stale and the test is latently
  buggy.** `wave1_coverage.py:99` sets `complete` when `pending == 0`, where
  pending counts only `discovered` + `captioned`. Any other status is counted
  as *neither* located nor pending. 42 meeting-groups now sit at `downloaded`
  (all stamped 2026-10-18, after the run) plus 1 `failed`, so **6 of the 11
  "complete" bodies are no longer complete**: fremont (3), hayward (5),
  petaluma (2 `discovered`), richmond (9), sanleandro (5), santaclara (13).
  Their §5 rates are now floors — exactly what the doc says a partial body's
  rate is. Only bart, napa, samtrans, sausalito, vta are still clean.
- **Pad-bounds: 24 % of on-topic episodes carry an unremoved pad.** Core
  bounds are never NULL (the fallback in `wave1_rates.py:85` never fires), but
  334 of 1,390 on-topic episodes have `core_*` *identical* to the padded
  bounds — 37 % of the classifier route, 20 % of fullasr, 0 % of local_asr /
  agenda_index / swagit (which all strip ~59 s correctly). Those numerators are
  wide by ~60 s each. Corpus-wide this is 2.7 %, but it lands **precisely on
  the headline cells**: BART 29.14 → 27.31 (−6.3 %), VTA 27.00 → 25.47
  (−5.6 %), Petaluma-classifier −3.0 %; Santa Clara, Fremont, Hayward and
  San Leandro are unaffected.

Claims that survive: BART/VTA at ≈ a quarter of meeting time with intervals
clear of every council (VTA lower bound 23.7 > highest council upper bound
17.1) — **VERIFIED**. "Route is a first-order effect" — **VERIFIED and now
stronger** (Hayward 0.65 / 1.73 / 4.17 by classifier / agenda / local ASR;
Santa Clara 6.92 classifier vs 2.81 agenda).

Drift since the run (denominators unchanged, numerators grew): San Leandro
density 8.0 → 10.0, Santa Clara classifier 5.2 → 6.9, BART 27.7 → 29.1, and
Santa Clara agenda_index 4.2 → **2.8** — the one cell that fell, worth a look.

## 2. Role gap — VERIFIED

The model-free bootstrap reproduces to within Monte Carlo noise on every cell.
Recomputed from `turn_stance` (9,358 rows, one `prompt_version`, all k=3/n_ok=3),
episode-cluster bootstrap, 2,000 reps:

| family | opposed diff | doc | supportive diff | doc |
|---|---|---|---|---|
| density | +0.32 (0.25, 0.38) | +0.32 (0.25, 0.39) | +0.23 (0.17, 0.29) | +0.23 (0.17, 0.29) |
| transit | +0.16 (0.11, 0.22) | +0.16 (0.10, 0.22) | +0.29 (0.23, 0.35) | +0.29 (0.22, 0.36) |

Model inputs match the frame **exactly**: stance/db n=4,845 / 583 episodes /
24 bodies / 712 public; hostility/db 8,778 / 590 / 935; computed 290 / 35 and
580 / 35. Length control is in the fit as described (`role_gap.py:60,100`,
centred log-words, `beta_len`) and locate route enters as `beta_strat`
(`:45`, 4 strata for db). Hostility ≥ 2 on instrument db: **3.74 % public /
0.52 % dais**, matching the doc's 3.7 / 0.5. Odds ratios ≈ 4–6 confirmed.

The "1.8 % vs 0.3 % on density" is the *model-implied* share at mean
body/episode/length, correctly labelled at `:258`; the observed raw shares are
4.3 % / 0.5 %. Both are right — they are different quantities.

**DISCREPANCY (internal contradiction).** §0 claims "R-hat ≤ 1.0013, 0
divergences". `models_frame/summary.json` gives max R-hat **1.0042**
(hostility/db) and **1 divergence** (hostility/computed). §7 reports both
correctly; §0 contradicts §7. "R-hat < 1.01" holds; "0 divergences" does not.
The proportional-odds caveat (bimodal public, PPC misfit) is stated and
correct.

## 3. NIMBY view — VERIFIED, one understated caveat

Reproduces exactly: 6,500 density+both turns → 2,100 clusters (public 465,
dais 1,635), 414 public clusters with a stance, split **42.0 / 41.5 / 16.4 %**,
per-route splits 41/39/20, 42/47/11, 34/41/24, 71/18/12 all matching.
Threshold ±0.5 confirmed. The cluster key is `(episode_id, cluster)` — per
episode, never across meetings, **DECISIONS #15 compliant**. Only Oakland
(+0.61) and Swagit-Petaluma (+0.53) exclude zero — **VERIFIED**.

- **Minor DISCREPANCY**: "Public clusters average 1.43 scored turns" (`:271`).
  1.42 is the mean of *all* turns per public cluster; the mean *scored* is
  **1.10** (1.23 among clusters with a stance), and **83 % of labels rest on
  exactly one scored turn**. The caveat is stronger than stated.
- **Sign-flip exposure, unquantified in the doc**: 121 of 179 (**68 %**) of
  San Leandro's scored public turns and 117 of its 158 public clusters
  (**74 %**) carry `sub_topic = affordability_displacement`, where the mapping
  inverts (a landlord opposing rent stabilisation codes as NIMBY). San Leandro
  is 158 of 414 clusters — **38 % of the whole public base**. Dropping it moves
  the split to 42.2 / 38.3 / 19.5: the Y ≈ N tie is partly a San Leandro
  artefact.

## 4. Lexicon and clustering — VERIFIED

κ recomputed from `llm_labels.jsonl` (300 rows, 0 errors) joined to
`sample.csv` on `turn_id`: 300 pairs, 0 dropped; confusion matrix
`[31,13,56] [9,30,61] [10,16,74]` character-for-character as published;
p_o 0.4500, p_e 0.3333, **κ = 0.1750** ✓. Keyness arm κ 0.1130 ✓.
Silhouettes rebuilt from the corpus snapshot against the stored labels:
style k=2 **0.1034** (doc 0.103), lex k=2 **0.0286** (doc 0.029); ARI vs the
Sonnet 3-way −0.061 / −0.029 ✓.

Two caveats the docs do not carry: p_e is exactly 1/3 because the sample is
stratified 100/100/100 on the lexicon label, so this κ is a within-sample
statistic and **not transportable** to the 3,854-turn frame. And "ARI ≈ 0 vs
stance" depends on folding `not_applicable` into neutral — kept as a fourth
class, lex k=2 scores ARI **0.216** / NMI 0.213. The clusters do track
procedural-vs-substantive, just not stance. The conclusion survives; the
framing is narrower than written.

## 5. Renter share — two DISCREPANCIES

ACS **2019–2023 5-year** vintage confirmed from raw file *contents*, not
filenames (US median HH income 78,538; B25003 US 127,482,865 = 82,892,037 +
44,590,828, internally consistent). Five-body spot-check against
`raw/acsdt5y2023-b25003.dat` reproduces to 6 dp: losaltos 0.1916, sanleandro
0.4169, samtrans 0.4091, bart 0.4597, santaclara 0.5880. Transit-board
construction correct (`geo_kind = county_union` for exactly bart/samtrans/vta;
BART = 06001/06013/06075). All §3 correlations and the binomial β (−0.169 per
sd by independent IRLS vs −0.1713 PyMC) reproduce.

- **DISCREPANCY (`:233`)**: "with log income and log density added the renter
  slope is **−0.29**". The artefact says `ols_with_controls.slope =
  **−0.2002**`. −0.29 appears nowhere; −0.2713 is the *uncontrolled* slope.
- **DISCREPANCY — the key claim fails (`:176–177`)**: "both are driven by the
  same two points… Dropping them leaves five councils and no usable
  correlation." Recomputed on the 7 complete-coverage classifier bodies,
  dropping bart+vta: density **Pearson rises +0.489 → +0.667** (p 0.219);
  Spearman falls 0.679 → 0.400. The boards were *diluting* the Pearson
  correlation, not driving it. Nothing is distinguishable from zero at n=5 —
  but nothing was at n=7 either (p = 0.27). Also "five **councils**" includes
  Napa, which the same file calls a County Board of Supervisors (`:69`).
- Housekeeping: `raw/geo-geography.csv` is an HTML error page logged `rc=0`
  in `raw/dl.log:5`. Unused downstream, but that is a silent-failure pattern.

## 6. Recall — reference mixing, and a circularity

§9's adjudicated table reproduces **to the digit** from
`reference_adjudicated.jsonl` (1,091 minutes, 18 flips: 17 `mention_only` +
1 `no` → 14 transit / 4 density; 127 about-minutes = 93 density / 37 transit):
v1_240 transit 0.297, v1_120 0.432, **v2_120 0.865**, any 0.811, density
0.796, precision_any 0.873. Full-ASR any **0.927** confirmed from
`codebook_v2_fullasr/results.json`.

- **DISCREPANCY: "0.44 → 0.87" is not like-for-like.** 0.44 (= 0.435) is
  v1_240 scored against the **v1-coded** reference; 0.87 (= 0.865) is v2_120
  against the **adjudicated** reference. Like-for-like is **0.297 → 0.865**
  (adjudicated) or **0.435 → 0.783** (v1). Likewise "full ASR 0.92" is the
  v1-reference figure; on the adjudicated reference it is **0.898**.
- **The bigger problem: the adjudicated reference is not independent of the
  arm it scores.** All 18 flipped minutes were created by adjudicating v2's
  own over-calls as "about", so v2_120 hits **18 of 18 by construction**. On
  the 109 *untouched* reference minutes v2_120 scores any **0.780**, transit
  **18/23 = 0.783**, density 0.787 — i.e. the entire lift from 0.78 to 0.87
  transit comes from minutes v2 itself defined. Transit's reference grew
  23 → 37 minutes, **38 % of it from v2's output**; v2_120's transit
  precision jumps 0.415 → 0.756 from the same reflip. The mechanism is
  disclosed at §9 and flagged in the project notes as "transit over-calls are
  definitional, need a v2-coded reference", but no quoted figure carries it.

## 7. Spend ledger — DISCREPANCY (stale in both directions)

Snapshot totals: RunPod **$8.4337** (22 `spend` rows), Claude **$69.4660**
(`classifier_calls` $31.9897 + `stance_calls` $37.4763), 31
`classifier_batches`, 28,074 `stance_calls` rows. `docs/COSTS.md` says $6.57
RunPod and $63.05 Claude ($19.15 in the headline) — short by **$1.86** and
**$6.42**, outside the $0.50 tolerance. It omits pods b39 ($0.6533) and b40
($1.2071) and 15 classifier batches ($9.1239), all after its last stamp.
Every *as-of-stamped* line ties out exactly, so this is staleness, not
miscomputation. Minor: pod `6a8i5phufeukie` is attributed to b5 at `:150`;
`batches` says b6. No dollar impact.

**The doc's caps are stale too, and that reverses the alarming reading.**
`pipeline/spend.py --json` — the authority per AGENTS.md — reads settings and
reports credit **$31.50**, stop line $31.00, **$23.07 left**, and
`claude_cap_usd` **$150.00**. So the true position is $8.43 of $31.50 (27 %)
and $69.47 of $150 (46 %), `runpod_blocked` and `claude_blocked` both false.
COSTS.md's "$3.43 left" and "$60.85 headroom" are both wrong, in opposite
directions from the truth. **The 80 %-of-cap escalation is not triggered.**

## What would most change the conclusions

1. **The adjudicated reference's circularity (§6).** Exclude the 18
   self-defined minutes and v2@120's transit recall is 0.783, not 0.865 — the
   measured case for codebook v2 largely evaporates. v2 is already adopted as
   the instrument with a re-read of the read corpus pending, so this is the one
   finding that would change a *decision* rather than a number. An
   independently coded reference, or any reference not derived from v2's
   output, would settle it.
2. **The San Leandro sign flip (§3).** 38 % of the public cluster base, 74 % of
   San Leandro's clusters, on a sub-topic where the YIMBY/NIMBY mapping
   inverts. The 42/41 near-tie is the NIMBY view's headline and it moves to
   42/38 without that body. If the mapping is wrong for rent-stabilisation
   turns, the direction of the whole view is unresolved, not merely noisy.
3. **The complete-coverage definition (§1).** It gates every cross-body rate
   comparison in the project. `pending == 0` ignores any status outside its
   `RANK` dict, so 42 meetings downloaded on 2026-10-18 are invisible to it and
   6 of the 11 "complete" bodies silently stopped being complete. The fix is a
   whitelist that raises on an unknown status rather than defaulting it to −1.

## Numbers that should be withdrawn or corrected

**Withdraw:**
- `RENTER-SHARE.md:176–177` "Dropping them leaves five councils and no usable
  correlation" — the councils-only Pearson is *higher* (+0.667). Replace with
  "nothing is distinguishable from zero at n=5, as was already true at n=7".
- `RENTER-SHARE.md:233` renter slope "−0.29" → **−0.20**.
- "0.44 → 0.87 transit with adjudication" should not be quoted as a pair.
  Quote **0.297 → 0.865** (adjudicated, like-for-like) with the circularity
  caveat, or 0.435 → 0.783 (v1 reference).

**Correct:**
- `WAVE1-RESULTS.md:184` "density outweighs transit on every council" → except
  Hayward, and not on Petaluma's Swagit route.
- `STANCE-S1-RESULTS.md` §0 "R-hat ≤ 1.0013, 0 divergences" → max R-hat
  1.0042, 1 divergence (as §7 already says).
- `STANCE-S1-RESULTS.md:271` "1.43 scored turns" → 1.10 scored (1.43 is all
  turns); 83 % of labels rest on one scored turn.
- `docs/COSTS.md` — refresh totals **and** caps; the current caps are $31.50
  credit / $150 Claude.

**Qualify, do not withdraw:** BART 27.7 and VTA 26.4 are correct as published
against their artefact, but are pad-inflated by ~6 % and have since drifted.
Regenerate, or quote as ≈ 25–29 min/h.

**Reproduced cleanly, quote with confidence:** the whole §2 role-gap model-free
result and its convergence (bar the §0 slip), the NIMBY split arithmetic and
its DECISIONS #15 cluster unit, κ 0.175, the silhouettes and ARIs, and the ACS
vintage and renter-share lookups.

---

## Validation applied (2026-10-18, supervisor pass)

Everything below was applied from the findings above. **$0: no API call, no
YouTube, no pod, no GPU.** `.env`, `plan/frame.csv` and `plan/DECISIONS.md`
untouched; **no `episodes` row and no other DB row was written** (see item 2);
nothing committed. Recomputes ran against a `sqlite3 .backup` snapshot
(`/tmp/val/snap.sqlite`, taken 2026-10-19T01:53Z) so the sibling workers'
writes were never blocked. Previous artefacts kept as `*.backup` beside each.

### 1. Coverage test — code fixed, finding CONFIRMED

`analysis/wave1_coverage.py:32-51` — `RANK.get(status, -1)` and
`pending = uncaptioned + captioned` replaced with an exhaustive
`LOCATED` / `PENDING` whitelist and `status_rank()`, which **raises
`UnknownStatus`** on a status in neither; `compute()` asserts
`located + pending == meetings` per body. `PENDING` now carries `downloaded`
and `failed`. New columns `downloaded_pending`, `locate_failed`.
`analysis/wave1_rates.py:42-45` imports the same whitelist instead of keeping
a second private copy of `RANK`.

| | before | after |
|---|---|---|
| complete-coverage set | **11** — bart, fremont, hayward, napa, petaluma, richmond, samtrans, sanleandro, santaclara, sausalito, vta | **5** — bart, napa, samtrans, sausalito, vta |
| now partial (rates are floors) | — | fremont, hayward, petaluma, richmond, sanleandro, santaclara |
| corpus | 614 meetings / 2,099 h / 29 bodies, 358 located | 671 meetings / 2,296.5 h / 31 bodies, 315 located, 43 `downloaded`, 1 `failed` |

### 2. Pad bounds — finding WITHDRAWN, false positive, no row changed

§1's third bullet does not hold and **no `core_*` value was recomputed**;
recomputing them from the window spans would have corrupted correct data.

`core_*` are the merged **unpadded** window bounds
(`classify.merge_episodes:766-797`). `episodes.start_s/end_s` are *the same
bounds* until the cutter replaces them with `clip_bounds()`'s padded,
clamped, cue-snapped span (`pipeline/episodes.py:107-114,292,347`). So
`core == start/end` on every row that has **not been cut** — by design, not a
leaked pad. Split by status, all 334 (328 on the snapshot) "zero-pad" rows are
exactly those rows: 294 classifier `located`, 2 classifier + 1 local_asr
`mention`, and 31 fullasr `done` rows written at `scripts/fullasr.py:824-829`
pointing at an already-transcribed clip's transcript without a cut of their
own. **Every cut row on every route carries its pad.** Verified directly
against `classifier_calls`: e.g. `0-Wf1GC8Xnk_e1` core 5940–6960 is exactly
`_c001.start_s` → `_c005.end_s`, while the padded span is 5910–6990;
`gr-richmond-6010_f1` core 480–1320 matches the unpadded fullasr windows.
The rate numerator uses `core` only, so it was never inflated:
**BART and VTA are not pad-inflated by ~6 %, and §1's 27.31 / 25.47 and the
"Qualify" line about a ~6 % pad should be struck.**

The invariant is now documented where the misreading happened
(`pipeline/classify.py:969-973`, `analysis/wave1_rates.py:81-87`).

Re-run of `wave1_rates.py` on the snapshot (**drift only**, same denominators,
numerators grew; no row was edited):

| cell | published artefact | snapshot re-run |
|---|---|---|
| bart classifier, any | 27.72 (24.67–30.74), 1,877.0 min / 67.7 h, 170 eps | **29.14** (26.20–32.15), 1,973.0 min / 67.7 h, 173 eps |
| vta classifier, any | 26.35 (22.79–29.82), 1,177.0 min / 44.7 h, 98 eps | **27.00** (23.71–30.25), 1,206.0 min / 44.7 h, 96 eps |

One small real defect found in passing, **not** fixed (it changes episode
spans, which is Puck's call): 7 rows have `core_end_s` past the end of the
media because the window grid overruns the recording — `gr-santaclara-2561_a3`,
`gr-santaclara-2398_e5`, `gr-santaclara-2517_e11`, `sw-petalumaca-361182_e12`,
`sw-petalumaca-392762_e9`, `4ZGPf-k5Vqw_e13`, `gr-santaclara-2585_e7`. Worth
at most ~60 s each.

### 3. Document corrections

| file:line | before | after |
|---|---|---|
| `WAVE1-RESULTS.md:111` | complete set stated as 11 bodies | staleness note + the 5-body set and the 6 bodies now partial |
| `WAVE1-RESULTS.md:184` (now :193-202) | "density outweighs transit on **every** council" | "…on every council on the classifier route **except Hayward**, and not on Petaluma's Swagit route"; the two exceptions enumerated (Hayward 0.1 vs 0.4, Petaluma Swagit 7.2 vs 8.2) |
| `STANCE-S1-RESULTS.md:17` | "R-hat ≤ 1.0013, 0 divergences" | "max R-hat **1.0042** (hostility/db), **1 divergence** (hostility/computed)" — now agrees with §7 |
| `STANCE-S1-RESULTS.md:271` (now :278-281) | "Public clusters average 1.43 scored turns" | "**1.10 scored turns** (1.23 among clusters with a stance); **83 %** of labels rest on exactly one scored turn"; 1.43 was all turns (1.42 on the artefact) |
| `STANCE-S1-RESULTS.md:320-331` (NIMBY view) | sign flip described, not sized | **117 of 158** San Leandro public clusters (**74 %**) and 121 of 179 scored public turns (68 %) on `affordability_displacement`; San Leandro is **38 %** of the 414-cluster public base; split **42.0 / 41.5 / 16.4 → 42.2 / 38.3 / 19.5** without it (reproduced from `speaker_clusters.csv`) |
| `RENTER-SHARE.md:176-177` (now :180-190) | "Dropping them leaves five councils and no usable correlation" | **withdrawn**: density Pearson **+0.489 → +0.667** (p 0.22), Spearman 0.679 → 0.400; transit-rate Pearson +0.198 → +0.592; "nothing differs from zero at n = 5, as was already true at n = 7"; Napa named as a County Board |
| `RENTER-SHARE.md:233` (now :256-260) | controlled renter slope "−0.29" | **−0.20** (95 % −1.48, +1.08; `ols_with_controls`); −0.2713 is the *uncontrolled* slope |
| `RENTER-SHARE.md:186-196` | — | new: the coverage fix drops the rate rows to n = 3; do not quote the rate correlations |
| `RECALL-CHECK.md:488-506` | 0.44 → 0.87 quotable as a pair | **never pair across references**: 0.435 → 0.783 (v1) or 0.297 → 0.865 (adjudicated); full ASR 0.927 (v1) / **0.898** (adjudicated); + circularity paragraph |
| `CODEBOOK-V2-PILOT.md:399-418` (§9) | caveat present, no figure carried it | quoting rule + **circularity quantified**: v2@120 hits **18/18** flipped minutes by construction (v1@240 1, v1@120 6, v2@240 11); on the 109 untouched minutes v2@120 scores any **0.780** / transit **18/23 = 0.783** / density **0.787**; transit reference grew 23 → 37, 38 % from v2's own output |
| `docs/COSTS.md:23-47` | "$6.57 RunPod of $10.00 credit, $3.43 left"; "$63.05 Claude of $80, $16.95 headroom" | snapshot totals **$8.4337** RunPod (22 rows) and **$69.4660** Claude (= classifier **$31.9897** + stance **$37.4763**); live caps **$31.50** credit / **$31.00** stop / **$150** Claude; position 27 % and 46 %, **80 %-of-cap escalation not triggered**; the $1.86 / $6.42 deltas labelled as staleness (pods b39, b40; 15 classifier batches); b5 → **b6** pod attribution noted |

All circularity and split figures above were recomputed independently before
being written, not copied from §3 / §6.

### 4. Figures regenerated

- `output/wave1/{coverage,rates_by_body_route}.csv`, `coverage.md`,
  `rates_complete_bodies.md` — re-run on the snapshot; the complete-bodies
  table now has 5 bodies.
- `output/renter_share/` — 12 figures regenerated by
  `analysis/renter_share_outcomes.py`, which now draws **transit boards as
  hollow markers** and adds a **councils-only right-hand panel** wherever ≥ 3
  councils remain (`scatter()` / new `_panel()`, `BOARD_GEO = county_union`);
  `correlations.csv` gains `councils_only_{n,spearman,pearson,pearson_p}`.
- `output/renter_share/_stale_pre_coverage_fix/` — the four `local_asr` /
  `agenda_index` rate figures, now below the 3-body floor. Do not quote.
- `output/stance_s1/nimby_view/*_stars.png` — **unchanged and correct**: no DB
  row and no stance code was touched, and `public_shares_by_route.csv` still
  reproduces 42.0 / 41.5 / 16.4 on 414 public clusters.

### Files changed

`analysis/wave1_coverage.py`, `analysis/wave1_rates.py`,
`analysis/renter_share_outcomes.py`, `pipeline/classify.py` (comment only),
`plan/WAVE1-RESULTS.md`, `plan/STANCE-S1-RESULTS.md`, `plan/RENTER-SHARE.md`,
`plan/RECALL-CHECK.md`, `plan/CODEBOOK-V2-PILOT.md`, `docs/COSTS.md`, this
file, and the regenerated artefacts under `output/wave1/` and
`output/renter_share/`.

---

## Role audit applied — 2026-10-18

`plan/ROLE-AUDIT-2026-10-18.md` recommended switching the reported public/dais
role from the diarization cue (`scripts/role_label.py`) to the audited role
(`scripts/role_audit.py`, rule `ra-1`, `state.sqlite.role_audit`), and
withdrawing the Petaluma Swagit star. **That recommendation has been adopted
across the results and the site.** Still EXPLORATORY: the stance κ ≥ 0.6 gate
(`SHOWCASE.md` 4a) is not passed, and the audit does not pass it — SELF_ID is
a second model reading the speaker's own words, not a human label.

No DB row was written. No money was spent, no API was called, no model refit
was run for this adoption (the `_roleaudit` / `_cuebaseline` fits already
existed). Nothing in `~/ytpt` is committed.

### Before / after

Every cue-role run below was **re-executed**, not quoted, and reproduces the
previously published number exactly — so each difference is the role column
and nothing else.

| quantity | cue role (before) | `role_audit` (after) |
|---|---|---|
| public turns in the frame | 1,069 | **1,027** |
| public clusters with a stance | 414 | **457** |
| frame-wide public split Y / neutral / N | 42.0 / 16.4 / 41.5 | **46.0 / 11.2 / 42.9** |
| starred bodies (net interval excludes 0) | oakland (classifier), **petaluma (swagit)** | **oakland only** |
| oakland net YIMBY−NIMBY (90 %) | +0.61 (+0.25, +0.80) ★ | **+0.65 (+0.11, +0.85) ★** |
| petaluma/swagit net (90 %) | +0.53 (+0.22, +0.78) ★ | **+0.40 (−0.04, +0.81)** — withdrawn |
| β_role stance, density (90 %) | −0.376 (−0.616, −0.129) | **−0.059 (−0.303, +0.196)** |
| β_role hostility, density (90 %) | +1.795 (+1.490, +2.091) | **+2.147 (+1.852, +2.461)** |
| tail: public−dais opposed, density | +0.32 (0.25, 0.39) | **+0.33 (0.27, 0.40)** |
| tail: public−dais supportive, density | +0.23 (0.17, 0.29) | **+0.31 (0.25, 0.37)** |
| tail: public−dais opposed, transit | +0.16 (0.10, 0.22) | **+0.21 (0.15, 0.28)** |
| tail: public−dais supportive, transit | +0.29 (0.22, 0.36) | **+0.31 (0.24, 0.38)** |
| hostile ≥ 2 share, density (public / dais) | 0.043 / 0.005 | **0.057 / 0.003** |
| renter-share public NIMBY ρ, classifier | −0.04 (−0.48, +0.40), n = 14 | **−0.15 (−0.62, +0.34), n = 13** |
| renter-share public YIMBY ρ, classifier | +0.20 (−0.26, +0.62), n = 14 | **+0.27 (−0.21, +0.70), n = 13** |
| renter-share net ρ, classifier | +0.11 (−0.35, +0.57) | **+0.20 (−0.27, +0.64)** |
| San Leandro share of the public base | 158 of 414 (38 %) | **163 of 457 (36 %)** |
| split without San Leandro | 42.2 / 19.5 / 38.3 | **47.3 / 12.9 / 39.8** |

**The two statements that changed in kind, not degree.**

1. **"The public is more opposed on density" does not survive.** The latent
   stance gap goes from −0.38 (90 % −0.62, −0.13) to −0.06 (−0.30, +0.20).
   Most of estimand 2's stance gap was members of the body sitting in the
   public bucket.
2. **"Petaluma's public runs YIMBY" does not survive.** Its 90 % interval
   crosses zero once the roles are audited. The largest "public" cluster in
   that frame — 1,083 words, YIMBY — is an elected member advocating housing
   near the SMART station, by their own self_id.

**Two that got stronger.** The hostility gap grows (+1.80 → +2.15 latent;
tail +3.8 → +5.3 points on density) because mislabelled dais speech had been
*diluting* the public bucket. And the model-free stance tails all widen while
the single latent shift collapses — the same fact, since the clusters leaving
the public bucket are long hedged member speeches near neutral and the ones
arriving are short and opinionated. §7's advice (quote the tails, not the gap)
is therefore stronger than it was, and the proportional-odds PPC misfit is
worse under the audit.

**What did not move.** The κ gate, the replicate κ 0.98, the sign-flip caveat
(`NIMBY-PILOT.md` §3.3 — a mapping problem, not a role problem; it still
carries San Leandro, still a third of the public base), the "83 % of cluster
labels rest on one scored turn" caveat, every rate and coverage number, the
`models_frame/` fits, and `turn_stance.role` / `speaker_roles`, which were not
overwritten. RENTER-SHARE's §4 regression was **not** refit under the audited
role; nothing in §4 is quoted as a result and its §3 counterparts all keep
their conclusion.

**Limitations of the audit itself, carried into every corrected statement**
(ROLE-AUDIT §8): SELF_ID is model output, not a gold label, and 2 of 11
Petaluma calls rest on a single phrase; **44 % of turns are
`self_id = unknown`** and keep the cue role, so the audited public bucket is
still contaminated at an unknown rate and the measured 20–39 % cue error is a
lower bound on the informative subset only; the change is a **re-partition in
both directions** (250 cue-public clusters out, 208 cue-dais in, public base
up 414 → 457), so differences are not "contamination removed"; the cluster
roll-up is majority rule and 21 ties keep the cue role; one prompt version,
nothing pooled.

### Section 4 correction

The §4 line above — "`output/stance_s1/nimby_view/*_stars.png` — **unchanged
and correct**" — is **superseded, not wrong**. It was accurate as a validation
statement (that pass touched no stance code and no DB row), and
`public_shares_by_route_cue.csv` still reproduces 42.0 / 41.5 / 16.4 on 414
clusters today. But those figures are no longer the reported ones: the
unsuffixed `*_stars.png` are now the `role_audit` versions and the cue
versions carry a `_cue` suffix.

### Files changed / new

**Modified (code).** `analysis/stance_s1_nimby_view.py` — now the reported
view: defaults to `role_audit` and writes the unsuffixed artefacts,
`--role-col cue` reproduces the cue run under `_cue`; the computation is
delegated to `analysis/stance_s1_role_audit_view.py` so the two role columns
cannot drift apart. `analysis/renter_share_outcomes.py` — `--role-col
{audit,cue}`, default audit; the role column also selects which `nimby_view`
cluster-share table feeds its carried `cl_YIMBY` / `cl_NIMBY` columns.
`~/ytpt-site/export/export_panels.py` — reads the audit, replaces its own
heuristic "Petaluma role misattribution" caveat with the real "star withdrawn"
one, adds the role-gap refit table and a role-column caveat to both panels.
`~/ytpt-site/export/export_site_data.py` — the reader's `role` is now the
audited role, with `role_cue`, `role_audit_turn`, `role_changed`,
`role_self_id`, `role_rule`, `role_audit_version` beside it.
`~/ytpt-site/index.html`, `site.css` — panel TL;DRs, the audit note, and the
reader showing both roles with the self_id.

**Modified (docs), each statement stamped "corrected 2026-10-18 after role
audit".** `plan/STANCE-S1-RESULTS.md` §0, §7 (new §7a), §7b, §9, §10;
`plan/RENTER-SHARE.md` §0, §2, §3, §5.3a, §6; `plan/WAVE1-RESULTS.md` §6.4
item 4; `docs/REPORT.md` §5.2, §7 items 5–6; `docs/post-lucashenry.md`
(hostility, stance, YIMBY/NIMBY and the Petaluma caveat); this file.

**New.** `analysis/stance_s1_tail_shares_roleaudit.py`;
`output/stance_s1/s1_tail_shares_bootstrap_{roleaudit,cue}.csv`;
`output/stance_s1/nimby_view/` `_cue` copies of 12 figures and 4 tables;
`output/renter_share/{outcomes_by_body_route,correlations}_cue.csv`,
`outcomes_cue.json` and 12 `_cue` figures;
`logs/roleaudit_adopt/{nimby_view,tail_shares,renter_share}.log`.

**Regenerated under `role_audit`** (previous contents were the cue run and are
preserved with a `_cue` suffix):
`output/stance_s1/nimby_view/{speaker_clusters,public_dais_shares_by_body_route,public_shares_by_route}.csv`,
`summary.json`, 12 figures;
`output/renter_share/{outcomes_by_body_route,correlations}.csv`,
`outcomes.json`, 12 figures.

**Untouched.** `.env`, `plan/frame.csv`, `plan/DECISIONS.md`,
`plan/STANCE-CODEBOOK.md`, `turn_stance.role`, `speaker_roles`,
`output/stance_s1/s1_tail_shares_bootstrap.csv`, `output/stance_s1/models_frame/`,
`output/role_audit/`. Committed: `~/ytpt-site` only; nothing in `~/ytpt`.
