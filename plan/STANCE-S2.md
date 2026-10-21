# STANCE-S2 — the top-up to the reloaded corpus, 2026-10-19

A census top-up, not a new design. `plan/STANCE-S1-RESULTS.md` scored the
whole frame **as it stood on 2026-10-18** — 9,358 turns over 628 QA-gated
corpus episodes. The corpus has since been reloaded from `output/whisper`
(1,209 episodes, 761 density/both), so the frame grew. S2 scores the
remainder under the same frame rule, the same pinned instrument and the same
k, then recomputes everything downstream. Nothing in `plan/DECISIONS.md`,
`plan/frame.csv`, `plan/STANCE-CODEBOOK.md` or `.env` was touched, and
nothing is committed in `~/ytpt`.

Trigger: Puck, 2026-10-19 08:16, "any more significant bars?" — which cannot
be answered off the old frame, because the overnight transcripts were not
scored.

## 0. Headline

- **3,765 new turns for $15.33** (11,295 Batches calls at k = 3, 0 errored,
  0 unparseable). `turn_stance` is now **13,123 rows**, all k = 3 / n_ok = 3.
  Claude ledger **$102.23 of `CLAUDE_CAP_USD` $150**.
- **Yes, there are more significant bars — three.** Starred public results
  (90 % interval on YIMBY − NIMBY excluding zero) went from **one to four**:

  | route | body | before | after |
  |---|---|---|---|
  | classifier | **sanjose** | n=23, +0.04 (−0.24, +0.53) | **n=191, −0.35 (−0.48, −0.15) ★** |
  | classifier | **santaclara** | n=38, +0.18 (−0.10, +0.49) | **n=45, +0.29 (+0.03, +0.56) ★** |
  | local_asr | **napa** | under min_clusters (10) | **n=37, −0.30 (−0.55, −0.10) ★** |
  | classifier | oakland | n=17, +0.65 (+0.11, +0.85) ★ | n=17, +0.65 (+0.11, +0.85) ★ |

  **No body lost a star.** Petaluma (Swagit) stays withdrawn: +0.40
  (−0.04, +0.81) → **+0.29 (−0.09, +0.68)**, interval still across zero.
- **The frame-wide public split crossed over.** Public clusters with a
  stance: **457 → 861**. The split went **46.0 YIMBY / 42.9 NIMBY / 11.2
  neutral → 41.1 / 48.4 / 10.4**. The slight YIMBY lean S1 reported is gone;
  the frame now leans *against* the housing item in front of the speaker.
  Under the cue role: 42.0 / 41.5 / 16.4 on 414 → **38.4 / 48.2 / 13.4 on
  782**, so this is the frame growing, not the role column.
- **San Jose is the whole story of the crossover.** It went from 23 public
  clusters to **191** — a sixth of the new frame's whole public base — and
  it is NIMBY-leaning. Read §5 before quoting the overall split.
- **A fifth locate route appears in this view: `fullasr`** (797 turns), which
  did not exist when S1 ran. Two bodies clear `min_clusters` on it; neither
  stars.
- Everything in `plan/STANCE-S1-RESULTS.md` §10 still applies, unchanged: the
  **κ ≥ 0.6 hand-coding gate is still not passed**, no human has coded any of
  these turns, and every figure carries the EXPLORATORY stamp. S2 makes the
  numbers bigger, not more validated.

## 1. Instrument — pinned, unchanged

`prompt_version st-7161e26ad2/claude-sonnet-4-6`, `claude-sonnet-4-6`,
k = 3, the cached-codebook system block, Message Batches. Byte-identical to
the pilot and S1, so S1 and S2 rows pool. **No existing `turn_stance` row was
modified**: every S2 row is a key that had none.

## 2. Frame — the same rule, a bigger corpus

Same five clauses as S1 §2, re-applied:

* episode in `corpus.sqlite` (built from the QA gate `output/qa/latest.csv`)
* family **density or both** — transit is a different estimand, YIMBY/NIMBY
  is a housing construct (S1 §7b, `NIMBY-PILOT.md` §5). 448 transit episodes
  in the reloaded corpus are out of scope here.
* a **redacted** transcript exists (`output/redacted/`)
* turn = `pipeline/stance.py episode_turns`, **turns ≥ 5 words**
* cue role **dais or public**; `unknown` stays outside the estimand

**The per-body 8 h cap is not re-applied, and here is why.** It is a
*corpus-construction* rule: it decided which meetings got transcribed,
upstream of the stance pass. S2 adds no transcription — it scores transcripts
the cap already admitted. Re-applying it at scoring time would drop turns that
are in the corpus S1 scored from, which is a different frame, not the same
one. Stated here so the rule stays auditable.

**Redaction was the actual blocker.** On first pass the frame yielded only 114
new turns: 361 of the 761 density/both episodes had no `output/redacted/*.json`,
and the frame is built from the redacted corpus by design (S1 §10,
"Redaction"). `scripts/redact_missing.py` ran `pipeline/redact.py` over
exactly those 361 ids — **360 written, 388,989 words, 2,700 replacements**
(person 1,280, cue_honorific 863, cue_speaker 160, cue_name_is 125, given_name
82, cue_im 71, cue_thank_you 56, address 54, cue_this_is 5, cue_resident 4),
63 episodes with zero replacements. `redact.py run --all` was **not** used: it
rewrites the whole directory, and the S1 frame is joined against the files it
wrote, so a top-up must only add.

**Positional-id safety.** A keyword-list change renumbers `turn_idx`
(`ytpt-positional-ids-rereads`). A turn counted as already scored only where
the existing row agrees on `cluster` at that index; any episode where a scored
index now named a different cluster would be excluded wholesale. **0 such
episodes** — 6,500 of 6,500 in-frame scored turns matched by key + cluster.

### Counts

| | episodes | turns ≥ 5 w | dais | public |
|---|---|---|---|---|
| S2 frame (density/both) | 758 | 10,265 | 9,076 | 1,189 |
| already scored (S1) | 393 | 6,500 | — | — |
| **new — the S2 pass** | | **3,765** | **3,237** | **528** |

New turns by body: santaclara 568, sanjose 524, paloalto 335, santarosa 324,
sanleandro 272, richmond 245, sunnyvale 236, napa 224, vta 156, pleasanton
155, hayward 131, petaluma 106, livermore 105, losaltos 103, sanmateo 76,
healdsburg 48, fremont 33, oakland 33, milpitas 31, concord 27, vallejo 12,
dalycity 10, bart 6, cupertino 5. By route: classifier 2,365, **fullasr 797**,
local_asr 521, swagit_transcript 82.

The S1 frame (9,358) and the S2 frame (10,265) overlap but neither contains
the other: S1 included transit-family episodes that S2's density/both rule
excludes. `output/stance_s2/frame_all.csv` is the deduped union, 13,123 rows —
exactly the `turn_stance` row count — and is the join table every downstream
view now uses.

## 3. Cost — estimate vs actual

Submitted as four chunks, `stance_batches.note` prefixed `stance-s2`. Chunking
was not cosmetic: `spend.claude_refusal` tests *worst case*, and the whole pass
at once was $77.99 worst case against a $86.90 ledger and a $150 cap, which
refuses. Four chunks are ~$19.5 worst case each.

| | turns | calls | `stance.py` estimate | actual |
|---|---|---|---|---|
| c1 `…UzQhQr2` | 942 | 2,826 | $6.85 | $4.6077 |
| c2 `…WEDQ1Pe` | 942 | 2,826 | $6.80 | $3.6080 |
| c3 `…Ha1Lrtt` | 942 | 2,826 | $6.78 | $3.5153 |
| c4 `…RcVdKEK` | 939 | 2,817 | $6.85 | $3.6021 |
| **total** | **3,765** | **11,295** | $27.28 | **$15.3331** |

**$0.001357 per call, $0.00407 per turn at k = 3** — 0.56 of the estimate,
within 4 % of S1's measured $0.001305/call. c1 is the dearer chunk because it
carries the cold cache. Projection before the run was $14.74 at S1's rate;
actual $15.33. Claude ledger **$86.90 → $102.23 of $150**, nowhere near the
$130 line.

## 4. The pass

11,295 calls, **0 errored, 0 unparseable**, every turn k = 3 / n_ok = 3.
`not_applicable` over the pooled frame: 47.2 % of dais turns, 21.5 % of public
(S1: 47.9 / 23.4). One turn in the whole table carries
`contains_instructions` — the same Richmond staff report S1 §5 recorded; no
new flag. Turn text is data: nothing in it was executed or followed.

## 5. What moved, and the one thing to be careful about

Recomputed under `role_audit` (rule ra-1) over all 13,123 scored turns:
**13,123 of 13,123 matched a frame row**, 0 blank routes. The audit now covers
4,840 clusters and moves 380 of them (S1: 265 of 3,239); informative-turn
disagreement 7.0 %, and it is still a **re-partition in both directions** —
356 cue-dais clusters in, 327 cue-public clusters out.

Per-cluster view: **3,701 clusters** (S1: 2,100); public with a stance
**861** (457).

> **San Jose dominates the new public base.** 191 of 861 public clusters —
> 22 % — and it is the one body whose interval is far from zero on the NIMBY
> side. The overall split crossing from +3.1 net YIMBY to −7.3 net is largely
> San Jose arriving, not every body moving. Quote the per-body table, not the
> overall split. This is the same warning S1 §7b carried about San Leandro,
> which has now *shrunk* to 21 % of the base from 36 %: dropping San Leandro
> used to move the overall split visibly and now barely does (41.1 / 48.4 /
> 10.4 → 40.9 / 48.2 / 10.8). The sign-flip problem on
> `affordability_displacement` items (133 of San Leandro's 179 clusters) is
> **diluted, not resolved**.

Model-free tail shares, density, public − dais, under `role_audit`: opposed
**+0.371 (90 % +0.312, +0.427)**, supportive **+0.274 (+0.228, +0.321)**,
hostile ≥ 2 **+0.044 (+0.032, +0.056)**. Both tails stay wide and the
hostility gap holds — the S1 direction, on a bigger frame.

## 6. Files

New: `scripts/stance_s2_sample.py`, `scripts/redact_missing.py`,
`scripts/stance_s2_recompute.sh`, `analysis/stance_s2_site_numbers.py`,
`plan/STANCE-S2.md`, `output/stance_s2/` (s2_frame.csv, s2_new_turns.csv +
four chunks, frame_all.csv, s2_strata.csv, s2_manifest.json, to_redact.txt,
site_numbers.json, before_* snapshots, and the pool / redact / sample /
dryrun / submit / collect / recompute / site_export logs), and 360 files in
`output/redacted/`.

Modified: `scripts/role_audit.py` (prefers `frame_all.csv`, takes `--frame`,
reports unmatched rows; **rule ra-1 unchanged**). Regenerated in
`output/stance_s1/nimby_view/`: the unsuffixed, `_roleaudit` and `_cue`
tables, summaries and **all 45 figures** (fifteen per suffix: five routes ×
{bars, bars + stars, public-vs-dais}, `fullasr` being the new route);
`output/stance_s1/s1_tail_shares_bootstrap_{roleaudit,cue}.csv`;
`output/role_audit/*.csv`. DB: `stance_calls` +11,295, `stance_batches` +4
(all `collected`), `turn_stance` +3,765, `role_audit` rewritten for the pinned
prompt_version (13,123 rows, ra-1).

`~/ytpt-site` (committed there, not here): `export/export_panels.py` and
`export/export_methods.py` — the route list and six prose counts that were
2026-10-18 literals now read `output/stance_s2/site_numbers.json`, and the
NIMBY TLDR and headline bullets are rewritten for the new result;
`index.html` one stale clause; `data/nimby_points.json` 457 → **861 points**,
`data/panels.json`, `data/methods.json`, `data/panels/*.png`.

Untouched: `.env`, `plan/frame.csv`, `plan/DECISIONS.md`,
`plan/STANCE-CODEBOOK.md`, `plan/STANCE-S1-RESULTS.md`, `output/whisper/`,
every pre-existing `output/redacted/` file, and every pre-existing
`turn_stance` row.

## 7. Limitations

Everything in `plan/STANCE-S1-RESULTS.md` §10 carries over unchanged. Added
by this pass:

- **The overall split is a composition effect as much as a measurement.** The
  public base doubled and one body supplied a fifth of it. "The frame leans
  NIMBY" is a statement about which meetings have been transcribed, not about
  the Bay Area.
- **`fullasr` enters this view for the first time** and is the one route with
  no caption screen in front of it. Its public base is thin (40 clusters, two
  reportable bodies) and the S1 route-as-stratum caveat (`REVIEW` finding 6:
  a stratum, not a measurement-error model) applies with less evidence behind
  it than for the older four.
- **The three new stars are 90 % intervals on an unvalidated instrument.**
  Four reportable cells of twenty-one clear zero; at 90 % and twenty-one
  cells, roughly two would be expected to clear it by chance alone. San Jose
  (n=191, interval well clear) does not rest on that; Santa Clara (+0.03 lower
  bound) and Napa (−0.10) are marginal and should be re-read after the next
  top-up before anyone leans on them.
- **Redaction coverage changed mid-corpus.** The 360 episodes redacted today
  used the current rule set; the S1 files were written 2026-10-17/18. The
  rules have not changed in between as far as this pass can tell, but it was
  not tested on the same episodes, which is the same caveat S1 §10 records
  about the pilot's unredacted text.
