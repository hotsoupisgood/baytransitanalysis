# Role-label audit — dais speech in the public bucket

2026-10-18. EXPLORATORY, like everything downstream of the S1 stance pass: the
stance κ gate (SHOWCASE.md 4a) is not passed. Nothing here is committed and no
existing role column, figure or table was overwritten; every new artefact
carries a `_roleaudit` suffix or lives under `output/role_audit/`.

## 1. What triggered this

On Petaluma (Swagit route) the public bucket ran strongly YIMBY — net
YIMBY − NIMBY **+0.53 (90 % +0.22, +0.78)**, the only starred public result
outside Oakland — while two of the strongest "public" clusters are, by the
stance model's own rationales, members of the body ("Elected member supports
housing near SMART station…"). `scripts/role_label.py` names this miss in its
own docstring: *"a councilmember who says 'I live in …' or 'on behalf of my
community' reads as public"*. The 93.6 % figure quoted for the cue labeller is
post-tuning speech-time accuracy on the same 10 hand-checked clips (T0 test 6;
REVIEW-2026-10-18 finding 4c) — it is not a held-out estimate, and it is a
speech-time number, not a per-turn one.

The independent read available on every turn is the codebook's **SELF_ID**:
what the speaker says about who they are, *from this turn's words only, never
inferred from role, tone, context or a name* (STANCE-CODEBOOK.md). Values:
`resident, homeowner, renter, business, org_rep, staff, elected, unknown`.

## 2. Disagreement between the cue role and SELF_ID

9,358 turns, one prompt version (`st-7161e26ad2/claude-sonnet-4-6`). 5,269
turns (56 %) carry an informative self_id; 4,089 are `unknown` and cannot
adjudicate anything.

| route | turns | informative | disagree | rate | cue public → self dais | cue dais → self public | rationale-flagged |
|---|---|---|---|---|---|---|---|
| agenda_index | 2,153 | 1,299 | 62 | **4.8 %** | 33 | 29 | 9 |
| classifier | 4,084 | 2,334 | 185 | **7.9 %** | 105 | 80 | 37 |
| local_asr | 2,051 | 1,053 | 77 | **7.3 %** | 40 | 37 | 17 |
| swagit_transcript | 1,070 | 583 | 38 | **6.5 %** | 7 | 31 | 6 |
| **all** | **9,358** | **5,269** | **362** | **6.9 %** | **185** | **177** | **69** |

The overall rate hides where the damage is. Split by the cue label, the public
bucket is far dirtier than the dais bucket, because it is ~50× smaller:

| route | dais turns: disagree rate | public turns: disagree rate |
|---|---|---|
| agenda_index | 2.4 % (29/1,214) | **38.8 %** (33/85) |
| classifier | 4.0 % (80/2,027) | **34.2 %** (105/307) |
| local_asr | 4.3 % (37/853) | **20.0 %** (40/200) |
| swagit_transcript | 5.6 % (31/552) | **22.6 %** (7/31) |

So roughly **one in five to two in five** cue-"public" turns whose speaker says
who they are say *elected* or *staff*. Worst bodies by turn count: bart 10.3 %,
sanjose 10.7 %, napa 9.2 %, richmond 7.9 %, hayward 7.7 %; sausalito is the
cleanest at 2.2 % (`output/role_audit/disagreement_by_body.csv`).

69 cue-"public" turns have a rationale naming an elected office
(`elected|council ?member|councilmember|mayor|supervisor|director`); 63 of them
also have a dais self_id, so the rationale flag is almost entirely redundant
with the rule below. The remaining 6 (self_id `unknown`) are mostly a speaker
*describing* a councilmember, not being one — which is why the flag is recorded
but does not drive the rule.

## 3. The rule (`ra-1`, `scripts/role_audit.py`)

Per turn:

```
self_id in {elected, staff}                                  -> role_audit = dais
self_id in {resident, homeowner, renter, business, org_rep}  -> role_audit = public
self_id = unknown                                            -> keep the cue role
```

Rationale matches are recorded as `rationale_flag` and are **not** used: the
rationale is free text about the item, and "the director said" is someone being
described. Staff are folded into dais, as STANCE-S1 §2 already does.

The analysis unit is the episode × cluster (DECISIONS #15), so the turn rule
rolls up: `cluster_role_audit` = majority of that cluster's informative turns;
ties and clusters with no informative turn keep the cue role. Of 3,239
episode × cluster units: 2,546 decided by majority, 21 ties kept, 672 with no
signal kept. **265 clusters change role** (250 public → dais, 208 dais → public
at turn level; 8.2 % of clusters).

Both directions matter. The cue labeller's `NO_REGION_PRIOR` ("a clip with no
public-comment stretch is dais") sweeps invited presenters and speakers in
clips that began mid-comment into the dais bucket; those come back as public.

Written to a **new** table `state.sqlite.role_audit` (one row per turn:
`role_cue, self_id, role_audit, rule, rationale_flag, cluster_role_audit,
cluster_rule, audit_version, audited_at`). `turn_stance.role` and
`speaker_roles` are untouched.

## 4. The NIMBY/YIMBY view under the audit

`analysis/stance_s1_role_audit_view.py --role-col {cue,audit}` is the same
frame, unit, threshold (T = 0.5) and episode bootstrap as
`analysis/stance_s1_nimby_view.py`, with the role column switched. Run with
`--role-col cue` it reproduces the published split exactly —
**42.0 / 16.4 / 41.5 on 414 public clusters** — so the difference below is the
role column and nothing else.

Frame-wide public split (YIMBY / neutral / NIMBY), clusters with a stance:

| route | cue | audit |
|---|---|---|
| ALL | 42.0 / 16.4 / 41.5 (n=414) | **46.0 / 11.2 / 42.9 (n=457)** |
| agenda_index | 34.1 / 24.4 / 41.5 (41) | 42.9 / 16.3 / 40.8 (49) |
| classifier | 41.2 / 19.8 / 39.0 (182) | 45.5 / 12.1 / 42.4 (198) |
| local_asr | 42.0 / 11.5 / 46.6 (174) | 44.9 / 8.6 / 46.5 (185) |
| swagit_transcript | **70.6 / 11.8 / 17.6 (17)** | **64.0 / 12.0 / 24.0 (25)** |

The headline 42/41/16 moves to 46/43/11: the YIMBY−NIMBY balance barely shifts
(+0.5 → +3.1 net), but the neutral band shrinks by a third — the clusters
arriving from the dais side are mostly short, opinionated public turns, while
the ones leaving are long, hedged member speeches.

**Stars.** Starred = the 90 % interval on net YIMBY−NIMBY excludes zero.

| route | body | cue net (90 %) | audit net (90 %) | star survives |
|---|---|---|---|---|
| classifier | oakland | +0.61 (+0.25, +0.80) ★ | +0.65 (+0.11, +0.85) ★ | **yes** |
| swagit_transcript | **petaluma** | **+0.53 (+0.22, +0.78) ★** | **+0.40 (−0.04, +0.81)** | **no** |

No new star appears. Santa Clara (agenda_index, public n=14) comes closest at
+0.50 (+0.00, +0.78) and is not starred under the strict `lo90 > 0` test.
Other notable movements: santaclara/classifier −0.12 → +0.18 (n 26 → 38),
petaluma/classifier +0.00 → −0.13, paloalto −0.33 → −0.38, sanleandro
−0.05 → −0.05 (n 158 → 163, unmoved), losaltos drops below the 10-cluster
floor and concord (classifier, n=14) and santaclara (agenda_index, n=14) rise
above it. Dais shares move very little anywhere (largest: concord classifier
+0.17 → −0.08 on 12 clusters).

## 5. Petaluma walk-through

Petaluma, Swagit route, density/both: **17 public clusters with a stance → 25**.
Four clusters leave the public bucket, thirteen join it, one is a tie.
The named turns (episode, cluster, timestamp, self_id, rationale):

| episode | cluster | turn | time | words | self_id | stance | rationale |
|---|---|---|---|---|---|---|---|
| `sw-petalumaca-392762_e5` | SPEAKER_00 | 12 | 02:45–10:01 | 1,083 | elected | +1 | "Elected member supports housing near SMART station with density, green space, safety, and mobility improvements." |
| `sw-petalumaca-384810_e6` | SPEAKER_04 | 17 | 04:42–06:02 | 251 | elected | −1 | "Elected member opposes housing on fairgrounds while supporting density at transit-oriented sites elsewhere." |
| `sw-petalumaca-388303_e9` | SPEAKER_05 | 33 | 18:03–20:59 | 510 | elected | −1 | "Elected member prefers lower 65-foot height limit over 8 stories, citing congestion and character." |
| `sw-petalumaca-392762_e9` | SPEAKER_01 | 5 | 00:04–00:07 | 9 | elected | n/a | "Procedural motion to extend meeting time; no substantive content about any item." |
| `sw-petalumaca-364602_e3` | SPEAKER_00 | 6 | 11:50–11:58 | 27 | staff | n/a | "Clerk announces received comments and begins process; no substantive position on item." |

The first of these is the single largest "public" cluster in the Petaluma
Swagit frame at 1,083 words, and it is YIMBY: a member advocating housing at
the SMART station was being counted as a member of the public. The second and
third are NIMBY-side members. Removing them *and* admitting thirteen genuine
public clusters from the dais side takes the net from +0.53 to +0.40 and widens
the interval across zero (n 17 → 25).

Two further cases worth seeing:

- `sw-petalumaca-388303_e9` SPEAKER_06 — turn 35 (21:04–22:53) self_id
  `resident`, turn 40 (23:07–25:00) self_id `elected`, same cluster. A 1–1 tie,
  so the rule keeps the cue role (public). Either the diarizer merged two
  people or a member said "I live in …"; the rule deliberately does not guess.
- `sw-petalumaca-361182_e8` SPEAKER_02 (00:00–01:02) — rationale names a
  councilmember's recusal, self_id `unknown`. Stays public, correctly: the
  speaker is describing a member, not being one. This is why the rationale flag
  is diagnostic only.

Petaluma's dais bucket (143–151 clusters) barely moves: net +0.152 → +0.154.

## 6. Role-gap model inputs

`analysis/models/role_gap_audit.py`. Usable stance turns per the model's own
`prepare()`: public **819 → 840** under the cluster roll-up (847 at turn level),
dais 4,316 → 4,295. Per body, the public count moves most at petaluma
**59 → 95 (+36)**, bart **95 → 72 (−23)**, santaclara 85 → 103 (+18), concord
8 → 23 (+15), sanjose 41 → 28 (−13). Counts for both outcomes are in
`output/analysis/models/tables/role_gap_input_counts_roleaudit.csv`.

A refit was cheap (~145 s for both outcomes), so it was run **on the same
input** under both role columns — the stored tables in
`output/analysis/models/tables/` come from an earlier, much smaller fit and are
not comparable:

| outcome | β_role cue (90 %) | β_role role_audit (90 %) |
|---|---|---|
| stance | **−0.376 (−0.616, −0.129)** | **−0.059 (−0.303, +0.196)** |
| hostility | +1.795 (+1.490, +2.091) | **+2.147 (+1.852, +2.461)** |

This is the sharpest result of the audit. Under the cue role, the public is
measurably more opposed than the dais on the same item; under the audited role
that gap is **indistinguishable from zero**, i.e. most of estimand 2's stance
gap was members of the body sitting in the public bucket. The hostility gap
goes the other way and strengthens: public comment really is more hostile, and
mislabelled dais speech was *diluting* it. `_roleaudit` and `_cuebaseline`
coefficient and gap tables are written beside the originals.

## 7. Recommendation

**Switch the public/dais view to `role_audit`, and withdraw the Petaluma
Swagit star.** The cue labeller and SELF_ID are independent instruments and
SELF_ID is the one that reads the speaker's own words; where they disagree in
the public bucket, the cue label is wrong often enough (20–39 % of adjudicable
turns) to carry a starred result on its own. Concretely:

1. Quote nothing from `fig_public_yimby_nimby_swagit_transcript_stars.png`
   without the audit beside it; Petaluma public-YIMBY is not a finding.
2. Re-state the role gap (estimand 2) with the audited role, or stop quoting
   the stance gap. The hostility gap stands and strengthens.
3. The published 42.0 / 41.5 / 16.4 split should be reported as
   46.0 / 42.9 / 11.2 under the audit, with both shown.
4. Do not re-run `scripts/role_label.py` cues to "fix" this — the cue labeller
   has no access to self-identification. The audit is a second instrument, kept
   in a second column.

## 8. Limitations

- **Not a gold set.** SELF_ID is model output, not a human label, and Puck has
  ruled out a gold set this phase (exploratory posture). This audit says two
  instruments disagree and which one has the better claim, not what the truth
  is. 2 of 11 Petaluma SELF_ID calls rest on a single phrase.
- **44 % of turns cannot be adjudicated** (`self_id = unknown`) and keep the cue
  role, so the audited public bucket is still contaminated at an unknown rate —
  the measured 20–39 % is a lower bound on the informative subset only.
- **Composition changes both ways.** The audit does not only remove members
  from the public bucket; it adds 208 cue-dais clusters. The net public count
  *rises* (414 → 457 clusters with a stance). Changes in the split are not
  "contamination removed", they are a re-partition.
- **Cluster roll-up is majority rule**: a merged diarizer cluster carrying a
  member and a resident gets one label, as before. 21 ties keep the cue role.
- The codebook's sign-flip caveat (NIMBY-PILOT 3.3, restriction items read
  "supportive" = NIMBY) is unchanged and still applies to every number here.
- One prompt version only; nothing is pooled across routes or instruments.

## Files

New:
- `scripts/role_audit.py`
- `analysis/stance_s1_role_audit_view.py`
- `analysis/models/role_gap_audit.py`
- `plan/ROLE-AUDIT-2026-10-18.md` (this file)
- `state.sqlite` table **`role_audit`** (9,358 rows, `audit_version = ra-1`)
- `output/role_audit/disagreement_{overall,by_route,by_body,by_route_role}.csv`,
  `output/role_audit/flagged_public_turns.csv`,
  `output/role_audit/baseline_cue/` (the cue-role reproduction run)
- `output/stance_s1/nimby_view/{speaker_clusters,public_dais_shares_by_body_route,public_shares_by_route}_roleaudit.csv`,
  `summary_roleaudit.json`, and 12 figures
  `fig_public_yimby_nimby_<route>{,_stars}_roleaudit.png`,
  `fig_public_vs_dais_<route>_roleaudit.png`
- `output/analysis/models/tables/role_gap_input_counts_roleaudit.csv`,
  `role_gap_{gap,coef}_{stance,hostility}_all_{roleaudit,cuebaseline}.csv`

Modified: none. Deleted: none. Committed: nothing.
