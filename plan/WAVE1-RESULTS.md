# Wave 1 results — the transit/density meeting corpus as transcribed on 2026-10-17

Run 2026-10-17 19:50 → 20:05 PDT (2026-10-18 02:50 → 03:05 UTC) by the
supervisor agent, day 2 of the wave-1 push. Everything here was produced
offline from `state.sqlite`, the transcripts already on disk and the venv:
**no YouTube request, no GPU, no Claude call, $0.** Nothing is committed;
the working tree is for Puck to review (file list at the end). Numbers are
as of the run; the state database moved slightly during it (a classifier
fill run by the main agent at 02:51 UTC located 5 more episodes), so the
coverage table was recomputed last and is the authority where two numbers
disagree.

Reproduce:

```
make test                                   # 427 checks, passed before anything ran
.venv/bin/python pipeline/redact.py run --all          # -> output/redacted/  (redact-1.1)
.venv/bin/python pipeline/redact.py run --residual-scan
.venv/bin/python analysis/qa_transcripts.py --src output/redacted --subs output/subs --csv output/qa/wave1_redacted.csv
make load-episodes                          # corpus.sqlite   <- output/redacted, gated on output/qa/latest.csv
make a0-wide                                # corpus_a0_wide.sqlite, same gate, figures per facet
make a1-wide                                # VADER + hostility lexicon, per facet
make wave1-coverage                         # output/wave1/coverage.{csv,md}
.venv/bin/python analysis/wave1_rates.py    # output/wave1/rates_*.{csv,md}
make models MODELSOUT=output/wave1/models   # estimand 1 + 2 refit, ~65 s
```

`make load-episodes` and `make a0-wide` now default to `TRANSCRIPTS=output/redacted
QA_GATE=output/qa/latest.csv` (Makefile); `TRANSCRIPTS=` empty restores the
unredacted read, `QA_GATE=output/qa/a0_episodes.csv` restores the 2026-09-23
99-episode A0-WIDE scope. **`output/analysis/a0_wide/` and
`corpus_a0_wide.sqlite` were overwritten by this run**; `plan/A0-WIDE-RESULTS.md`
describes the old 99-episode tree, which the second form regenerates.

---

## 1. What is in

| | |
|---|---|
| bodies with transcribed episodes | **26** of 29 with regular meetings in the window (33 in `plan/channels.tsv`) |
| episodes loaded | **628** (633 passed QA; 5 Sunnyvale clips `Chj4lSP9je4_e1–e5` have no `episodes` row and are not loadable) |
| episode hours | **121.6 h** (padded cut bounds) |
| words after loop-collapse | **1,029,427** in 72,106 segments; 3,437 diarized clusters |
| meetings with ≥1 transcribed episode | 157 |
| GPU batches | b13, b15, b17, b18, b22, b31, b32, b33, b34 (9 done batches, **$2.74** actual; every other b-id failed before billing or was denied) |
| locate routes (`episodes.method`) | classifier 334 ep / 53.9 h · local_asr 161 / 25.4 h · agenda_index 53 / 27.4 h · swagit_transcript 80 / 14.9 h |
| role instrument (`roles_source`) | db 593 · computed 35 (all on the classifier route, the b13 pilot) |
| family tags | density 328 · transit 232 · both 68 episodes |

Spend recorded in the ledgers: RunPod **$5.64** total across both corpora
(the $10 credit); `classifier_calls` sums to **$12.39** (the classifier's own
log line this evening read $14.40 — `pipeline/spend.py` is the authority, not
this note).

### 1a. Redaction (DECISIONS #15) — `pipeline/redact.py run --all`, version `redact-1.1`

Run over every `kind=episode` transcript with QA status OK/WARN in the
02:44 UTC snapshot:

| | |
|---|---|
| episodes redacted | **633** |
| words seen | 1,047,532 |
| replacements | **6,977** (6.7 per 1,000 words) |
| episodes with 0 replacements | 46 (short consent-calendar / closed-session clips, same pattern as the calibration note) |

By rule: person (spaCy) 3,399 · cue_honorific 2,280 · cue_speaker 322 ·
given_name 267 (the new layer from `output/redacted/CALIBRATION.md` #2) ·
cue_name_is 244 · cue_thank_you 211 · address 144 · cue_im 90 · cue_this_is 9 ·
contact 5 · cue_record 4 · cue_resident 2.

`--residual-scan` on the output: **1 probable miss**, `gr-santaclara-2540_e1`
turn 29, "President **Abraham** Lincoln issued the Emancipation
Proclamation" — a historical namesake on the allowlist, not a participant;
no action. The redacted set is what was loaded: `analysis/load_episodes.py
--transcripts output/redacted` reads `<episode_id>.json` from there and
refuses to fall back to `output/whisper` (628 read, 0 missing, all
`redact-1.1`). The hand calibration (200 rows, 90 % correct, 6.5 %
over-redaction) is in `output/redacted/CALIBRATION.md` and predates this run;
the code is unchanged since except that it ran over 46 more episodes.

### 1b. QA gate

- `output/qa/latest.csv` (02:44 UTC, `output/whisper`, 776 files): **681 OK / 95 WARN / 0 FAIL**; the 633 episode rows: 556 OK / 77 WARN.
- Re-run on `output/redacted` (`output/qa/wave1_redacted.csv`): **556 OK / 77 WARN / 1 FAIL**. The FAIL is `output/redacted/summary.json`, the redactor's own run summary, which is not a transcript ("neither a segment list nor an object with 'segments'"). No transcript failed. Two status flips against the unredacted run (`gr-santaclara-2517_e6` WARN→OK, `sw-petalumaca-392762_e9` OK→WARN), both on the caption-agreement threshold; WER vs captions moved by a median +0.16 pp (max +3.2 pp) because `[name]` tokens do not match the caption text. Nothing was excluded on QA grounds.
- Instruction-like text scan over all 72k redacted segments (prompt-injection phrasing, role-play markers, "ignore previous instructions" and variants): **0 hits**.

## 2. What is out

| | | |
|---|---|---|
| uncaptioned regular meetings | **255 distinct** (276 `discovered` rows: 13 are Petaluma YouTube rows already covered by Swagit, 8 are Cupertino multi-part duplicates) | numerator unknown; YouTube trickle worker, 1/h, 18 per 24 h |
| located, not cut | **403 episodes / 73.9 h** (398 / 72.7 h earlier today) | awaiting YouTube audio downloads; 25 of them are BART (131 ep / 24.1 h) and VTA (82 / 17.7 h) |
| captioned, not classified | 1 meeting (`gr-oakland-7541`, API errors in the 02:51 UTC classifier run) | $0.01 to finish |
| bodies with meetings, nothing transcribed | cotati, milpitas, rohnertpark (all partial coverage) | |
| in `channels.tsv`, no regular meeting in window | alameda, mountainview, sanfrancisco, walnutcreek | archive is a commission or stale |
| unlocated bodies | **10** — AC Transit, MTC/ABAG, Caltrain, SFMTA, Berkeley, Marin County BoS, Redwood City, Windsor, Novato, Sonoma County BoS (`plan/T2-PLAN.md` §6) | the regional-board side of the corpus is thin |
| Hasan/Tucker side corpus | 143 `kind=show` videos, frozen (DECISIONS #16) | untouched, not redacted, not in these figures |
| paid passes | full stance pass (only the 300-turn pilot exists), A1 Claude pass | not run |

## 3. Coverage (`output/wave1/coverage.md`, full table there)

614 distinct regular meetings / 2,099 meeting-hours over 29 bodies; 358
located (58 %). **Complete locate coverage (11 bodies):** bart, fremont,
hayward, napa, petaluma, richmond, samtrans, sanleandro, santaclara,
sausalito, vta. **Partial (18):** every other YouTube council, where 50–95 %
of meetings still have no caption track. Only the complete set supports a
rate comparison between bodies; a partial body's rate is a floor.

*Corrected 2026-10-18 after validation*: the 11-body set above is as of this
run and the test that produced it was buggy — `wave1_coverage.py` counted only
`discovered` + `captioned` as pending, so any other status was neither located
nor pending. With the whitelist fix (unknown status now raises) and the
2026-10-18 snapshot, **the complete set is 5 bodies: bart, napa, samtrans,
sausalito, vta** (671 meetings / 2,296.5 h over 31 bodies, 315 located, 43 at
`downloaded` and 1 `failed`). fremont, hayward, petaluma, richmond, sanleandro
and santaclara are now partial, so their §5 rates are floors.

Within the complete set, *locate* coverage is not *transcript* coverage: BART
has 25 % of its located hours transcribed and VTA 11 % (the rest is in the
YouTube download queue); the nine others are at 98–100 %. Rates in §4 use
located minutes and are unaffected; the text-based figures (§5–6) for BART
and VTA rest on the transcribed quarter.

## 4. The facet rule

`episodes.method` (locate route) and `roles_source` (role instrument) are
instrument ids, not corpus variables (`analysis/facets.py`,
`plan/T2-PLAN.md`). Every A0/A1 figure in this run is drawn once per level —
`output/analysis/{a0_wide,a1_wide}/by_method/<route>/` and
`by_role_source/<source>/` — and the manifests list each cell's size. The
only pooled objects are the stratified model outputs stamped `POOLED_`, where
route enters as a fixed-effect stratum. Hayward shows why: the same body
measured by three routes gives 0.5, 1.4 and 3.6 on-topic min per meeting-h
(§5). That spread is the instrument, not Hayward.

Facet cells (episodes / words): classifier 334 / 448,574 (22 bodies) ·
local_asr 161 / 215,269 (3) · agenda_index 53 / 233,784 (5) ·
swagit_transcript 80 / 131,800 (1) · roles db 593 / 966,724 (24) · roles
computed 35 / 62,703 (5). `computed` roles exist only on the classifier
route, so the role-instrument comparison is testable there and nowhere else.

## 5. Headline findings — topic time

**Estimand:** located on-topic minutes per regular-meeting hour, by body ×
route × family (`analysis/wave1_rates.py`). Every located episode counts,
transcribed or not; `mention_only` and failed rows do not. Denominator = the
body's located meeting-hours (the `meeting_exposure` rule: a meeting is
exposure for every route that read it; the route-read denominator is in the
CSV as a sensitivity and moves Petaluma-Swagit 10.0 → 12.5, Hayward-agenda
1.4 → 8.4 on 6 meetings). Intervals are 95 % percentile cluster bootstraps
that resample whole **meetings** within body (2,000 replicates): the meeting
is the sampling unit, so a body with 11–16 meetings has honest, wide
intervals. No pooling across bodies in this table.

Complete-coverage bodies only, min per meeting-h (95 % bootstrap):

| route | body | any | density | transit |
|---|---|---|---|---|
| classifier | **bart** | **27.7** (24.7–30.7) | 2.6 (0.6–5.2) | 27.7 (24.9–30.9) |
| classifier | **vta** | **26.4** (22.8–29.8) | 5.7 (2.9–8.6) | 26.4 (22.8–29.8) |
| classifier | santaclara | 5.2 (3.1–7.4) | 4.0 (2.0–6.3) | 2.1 (1.0–3.5) |
| classifier | fremont | 3.7 (1.7–6.0) | 2.1 (0.7–3.9) | 2.1 (0.7–3.8) |
| classifier | napa | 1.9 (0.4–4.2) | 1.6 (0.2–3.6) | 0.8 (0–2.1) |
| classifier | petaluma (5 YouTube meetings) | 1.8 (0–5.2) | 1.8 (0–5.0) | 0.6 (0–1.7) |
| classifier | hayward | 0.5 (0–1.6) | 0.1 (0–0.3) | 0.4 (0–1.3) |
| local_asr | sanleandro | 8.8 (4.6–14.0) | 8.0 (3.9–13.0) | 1.9 (0.4–3.8) |
| local_asr | hayward | 3.6 (1.3–6.6) | 0.7 (0.2–1.5) | 3.1 (0.7–6.2) |
| local_asr | richmond | 2.1 (1.3–3.1) | 1.4 (0.6–2.3) | 0.9 (0.4–1.5) |
| agenda_index | santaclara | 4.2 (2.3–6.1) | 2.8 (1.3–4.5) | 1.6 (0.6–2.8) |
| agenda_index | napa | 2.2 (0.5–4.2) | 1.3 (0–2.9) | 0.9 (0–2.4) |
| agenda_index | hayward | 1.4 (0.1–3.6) | 1.0 (0–3.1) | 0.4 (0–1.3) |
| swagit_transcript | petaluma | 10.0 (4.8–16.0) | 7.2 (1.9–13.4) | 8.2 (3.7–13.5) |

(samtrans and sausalito have no non-zero numerator cell with a bootstrap
interval; they are in the CSV.)

What it says, with the uncertainty attached:

1. **The two transit boards spend about a quarter of their meeting time on
   transit** (BART 27.7, VTA 26.4 min/h; intervals do not overlap any
   council's). That is close to a tautology for a transit agency, and it
   is the reason the board side needs its own comparison set — which the
   corpus does not yet have (SamTrans is the only other board, located by
   agenda index with 7 episodes; AC Transit, MTC, Caltrain, SFMTA are
   unlocated).
2. **Among councils on the classifier route, Santa Clara (5.2) and Fremont
   (3.7) sit above Napa (1.9), Petaluma (1.8) and Hayward (0.5), but the
   intervals of the bottom three overlap each other and Fremont's.** The
   only pairwise statement the bootstrap supports is Santa Clara > Hayward
   and Santa Clara > Napa on this route.
3. **Density outweighs transit on every council on the classifier route
   except Hayward, and not on Petaluma's Swagit route; transit outweighs
   density on every board.** Santa Clara 4.0 vs 2.1, San Leandro 8.0 vs 1.9,
   Richmond 1.4 vs 0.9, Napa 1.6 vs 0.8; BART 2.6 vs 27.7, VTA 5.7 vs 26.4.
   The two exceptions on this same table: Hayward classifier 0.1 density vs
   0.4 transit, Petaluma Swagit 7.2 vs 8.2. San Leandro's density figure
   (rent stabilisation, §6) is the largest council rate in the corpus on any
   route. *Corrected 2026-10-18 after validation*: the original read "on
   every council" with an enumeration of only the four councils that fit
   (`plan/VALIDATION-2026-10-18.md` §1).
4. **Route is a first-order effect.** Hayward 0.5 / 1.4 / 3.6 by classifier /
   agenda / local ASR; Santa Clara 5.2 classifier vs 4.2 agenda; Petaluma 1.8
   classifier (5 meetings) vs 10.0 Swagit (13 meetings). The stratified model
   below puts local_asr and agenda_index at roughly 0.77× the classifier and
   Swagit at 1.75×, each with an interval spanning 1.

**Partial-pooling refit (`make models`, estimand 1, `output/wave1/models/`,
stamped `POOLED_`).** Negative-binomial on rounded located minutes per body ×
month × family cell with log-exposure offset, body random intercepts
(σ_b 0.55, 90 % 0.14–0.97 on the log scale), a month smooth, zero-inflation,
and **route as a fixed-effect stratum**; 4 chains × 1,000 draws after 1,000
tuning, R̂ ≤ 1.009, 0 divergences, min bulk ESS 613, PPC in `figs/`.
Mean-body rates: density **6.1** min/h (90 % 3.9–9.6), transit **5.0**
(3.0–8.2); transit/density ratio 0.83 (0.53–1.27), Pr(ratio > 1) = 0.22.
Route ratios vs classifier: local_asr 0.77 (0.37–1.80), agenda_index 0.77
(0.44–1.36), swagit 1.75 (0.67–4.27). Two differences from the table above,
both deliberate in `analysis/models/load.py`: its exposure is **cut**
meetings only (so BART's classifier cell has 15.3 of 67.7 located hours and
its posterior median is shrunk to 23.7 with a 9.0–68.0 interval on 6 cells),
and it includes the 18 partial-coverage bodies. Read it as the shrinkage
estimate of the body hierarchy, not as the coverage-restricted comparison;
the per-route fits (`topic_time_bodies_route-*.csv`) are the unpooled
sensitivity.

## 6. Headline findings — who speaks how (lexicon, $0)

`make a1-wide`: VADER compound per turn and the hostility lexicon
(`plan/hostility.txt`), per route, roles from the facet's own instrument.

1. **Public turns score more positive than dais turns on every route**
   (median compound, turns ≥ 15 words: 0.88 vs 0.68 classifier; 0.90 vs 0.64
   local_asr; 0.96 vs 0.64 agenda; 0.97 vs 0.61 Swagit). This is mostly
   length: the median compound climbs monotonically from 0.0 at 1–5 words to
   0.996 at 600+ words (`a1_sentiment_vs_length.csv`), and a prepared
   two-minute public comment is long while a dais turn is often procedural.
   VADER is not a stance instrument here; treat this as a description of
   turn shape.
2. **Hostility-lexicon hits are rare and slightly more frequent in public
   speech on three routes**: per 1,000 words public vs dais 0.15 vs 0.08
   (classifier, 13 vs 28 hits), 0.18 vs 0.09 (local_asr), 0.33 vs 0.05
   (agenda), and reversed on Swagit (0.29 vs 0.45, 4 vs 51 hits, one body).
   With double-digit hit counts the cluster-bootstrap intervals overlap
   everywhere; this is a screen for where to read, not a result.
3. **Keyness (public vs dais, log-ratio, `a0_keyness_public_vs_dais.csv`)**
   on the classifier route: public-leaning *resident, people's, political,
   subway, underground, engineer, licensed, stanford*; dais-leaning
   *recommends, easement, elevations, concludes, followed, maps, card*. On
   the local-ASR route (Richmond, San Leandro, Hayward): public *mom, pop,
   landlord, expenses, retirement, petition, parents, corporate, oppose,
   forced, caps*; dais *zoning, tdm, clipper, pavement, agreement, minimums*.
   The rent-stabilisation fight in Richmond and San Leandro is visible in the
   public lexicon and absent from the dais one.
4. **Stance-pass pilot (estimand 2, 300 turns, 77 public, Sonnet instrument,
   ordinal model with episode and body intercepts; refit tonight):** stance
   gap public − dais, latent scale, **0.01 (90 % −0.64, 0.65)**, Pr(public
   more opposed) = 0.50 — no evidence either way; hostility gap **0.81
   (−0.04, 1.64)**, odds ratio 2.26 (0.96–5.16), Pr(gap > 0) = 0.94 —
   moderate evidence that public turns are rated more hostile, consistent in
   sign with the lexicon screen. Pilot only: 300 turns, the κ ≥ 0.6 gate
   against hand coding (`plan/SHOWCASE.md` 4a) is not yet passed, and no
   money was spent extending it.

   *Corrected 2026-10-18 after role audit.* Superseded twice over: the whole
   9,358-turn frame has since been scored (`plan/STANCE-S1-RESULTS.md`) and
   the public/dais role column has since been audited
   (`plan/ROLE-AUDIT-2026-10-18.md`), which is now the reported role. The
   numbers above — like every role-gap number written before today — are
   **cue-role** numbers. On the frame under `role_audit`: **stance gap on
   density −0.06 (90 % −0.30, +0.20)** against −0.38 (−0.62, −0.13) under the
   cue role, so what read as "the public is more opposed on density" was
   largely dais speech sitting in the public bucket; **hostility gap +2.15
   (1.85, 2.46)**, *larger* than the cue-role +1.80 (1.49, 2.09), because the
   mislabelled dais speech had been diluting it. The hostility direction this
   pilot saw survives and strengthens; the stance direction does not survive.
   For stance, quote the model-free tail shares, not the latent gap (STANCE-S1
   §7). The audit reads a second model's self-identification, **not** a human
   gold label, and it does not pass the κ gate.

Thin cells are listed per facet in `a0_cells_thin.csv` (classifier route:
livermore/dais/transit 1 episode, cupertino/public/density 104 words, …);
the figures draw them but they are not reportable.

## 7. Caveats that bound every number above

- **Selection is by platform, not by politics.** The complete-coverage set is
  Granicus bodies plus two YouTube boards whose captions landed first; the 18
  partial bodies are the YouTube councils behind the caption rate limit.
  Nothing says the uncaptioned 255 meetings resemble the located 358.
- **Routes are instruments with different recall.** `local_asr` counts are a
  floor (`T2-PLAN.md`), agenda indexes find only what the clerk titled, the
  classifier reads only caption windows above the keyword threshold.
- **Roles are diarization clusters labelled by cue phrases**, 5–6 % of words
  are `unknown` in every facet, and `computed` roles were never validated
  against `db` roles outside the pilot.
- **WER vs YouTube captions measures ASR agreement, not accuracy** (QA note);
  the 8.7 % figure from T0 stands for the one hand-checked set.
- The redaction pass is calibrated at 90 % correct / 0 cue misses on a 200-row
  sample; the residual-scan is a regex check, and PERSON recall of spaCy
  `en_core_web_sm` on ASR text is unmeasured. Publishing text still needs
  Puck's 30-turn hand check (`SHOWCASE.md` §3.4).

## 8. Files

New: `plan/WAVE1-RESULTS.md` (this), `analysis/wave1_coverage.py`,
`analysis/wave1_rates.py`, `output/wave1/` (coverage.{csv,md},
rates_by_body_route.csv, rates_complete_bodies.md, a1_wide_facet_manifest.csv,
a0_wide_facet_manifest.csv, figures/, models/), `output/analysis/a1_wide/`,
`output/qa/wave1_redacted.csv`, `output/logs/{redact,qa_redacted,load,a0_wide,
a1_wide,models}_wave1_*.log`.

Modified: `Makefile` (TRANSCRIPTS/QA_GATE defaults for `load-episodes` and
`a0-wide-load`, new `a1-wide`, `wave1-coverage`), `analysis/load_episodes.py`
(`--transcripts DIR`), `output/redacted/` (633 files rewritten, same version),
`output/analysis/a0_wide/` (now the 628-episode run), `~/.local/share/ytpt/
corpus.sqlite` and `corpus_a0_wide.sqlite` (episode tables rebuilt from the
redacted set; the side-corpus tables in `corpus.sqlite` untouched).

Already uncommitted before this run (day 1): `pipeline/redact.py` (given_name
layer, redact-1.1), `analysis/test_pipeline.py` (+7 checks),
`plan/given_names.txt`, `plan/build_given_names.py`, `ops/retry_until_lands.py`,
`ops/launchd/fired/`, `output/{reference,t1,trends,trends_sim_demo}/`.

Not done: `make package` (it builds a code-only transfer tarball that excludes
`output/`, so it cannot carry this deliverable; `output/wave1/` is the package).
Nothing deleted; `.env`, `plan/frame.csv`, `plan/DECISIONS.md` untouched.
