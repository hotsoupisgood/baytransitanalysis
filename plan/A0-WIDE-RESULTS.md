# A0-WIDE results — A0 on every QA-gated episode, faceted by instrument

Run 2026-09-23. Scope: `plan/ANALYSIS-transit.md` §1 + §2, the A0 figure set,
on **99 episodes / 10 bodies / 161,072 words** — every episode that is through
the QA gate (`output/qa/a0_episodes.csv`, 63 OK / 36 WARN / 0 FAIL), not just
the T1 pilot's 46. Exploratory plots only, **no model was fitted**, nothing was
spent ($0: no Claude call, no GPU, no RunPod), and no YouTube request was made
(the caption bot-check backoff is live until 2026-09-24 04:12 UTC).

Reproduce with `make a0-wide`. `make a0` still means the pinned 46-episode
pilot, in its own database, and is unchanged.

```
make a0        -> ~/.local/share/ytpt/corpus_a0_pilot.sqlite -> output/analysis/a0/
make a0-wide   -> ~/.local/share/ytpt/corpus_a0_wide.sqlite  -> output/analysis/a0_wide/
```

---

## 1. The facets, and why a pooled figure is now an error

A0-RESULTS §4 and §6 flagged that running A0 wide would pool things
`plan/T2-PLAN.md` says never to pool. Two columns of `episodes` are instrument
ids, not corpus variables:

| facet | levels in the wide corpus | episodes |
|---|---|---|
| `method` — the **locate route** | `classifier` (Claude over caption windows) | 80 |
| | `agenda_index` (English agenda titles) | 19 |
| `role_source` — the **role instrument** | `db` (labeller wrote `speaker_roles` at ingest) | 70 |
| | `computed` (cue phrases, recomputed at load time) | 29 |

`analysis/facets.py` holds both definitions and one guard,
`require_single(con)`. `a0_plots.py`, `a0_terms.py`, `a0_bridge.py` and
`a1_sentiment.py` call it before they draw anything. A corpus spanning a facet,
with no facet stamp, is **refused** with the command to run instead:

```
$ .venv/bin/python analysis/a0_plots.py --db .../corpus_a0_pilot.sqlite
refusing to draw a pooled figure: this corpus spans role_source (computed/db).
  These are instruments, not variables (analysis/facets.py, plan/T2-PLAN.md).
  Draw it per facet:   .venv/bin/python analysis/facet_sweep.py --db <db>
  Or pool on purpose:  .venv/bin/python analysis/a0_plots.py --db <db> --pool
```

`analysis/facet_sweep.py` slices the corpus once per level, stamps each slice
with an `a0_facet` table, and runs the whole pipeline into
`by_<facet>/<level>/`. The slice database sits in the same folder as the figures
drawn from it. A pooled set is still reachable — `--pool`, or `make a0-pooled` —
but every file it writes is prefixed `POOLED_` and captioned `POOLED`, so it
cannot be mistaken for a per-facet result six weeks from now. Every figure now
carries a `Scope:` line naming its instrument.

**Only three of the four method × role-source cells exist**: `agenda_index` is
100 % `db`-roles. The two facets are therefore partly confounded, and
`agenda_index` vs `computed` is not a clean comparison. `classifier` splits
51 db / 29 computed, which is where the role instrument can actually be tested.

### One correction to A0-RESULTS §6

A0-RESULTS §6 says running wide "mixes three `episodes.method` routes —
`classifier`, `agenda_index` and `local_asr`". Today it mixes **two**: there is
no `local_asr` episode in `state.sqlite` yet. That route is real but still
upstream — `scripts/asr_batch.py` is working through Richmond and San Leandro
(59 meetings / 260.6 audio-h, START_HERE "Granicus locate routes"), and nothing
has landed. The facet takes a third level the moment one appears, and
`plan/T2-PLAN.md` §400 already says to treat `local_asr` counts as a floor and
never to pool them. Nothing to change in the plan; the count in A0-RESULTS §6
was ahead of the data.

---

## 2. The corpus, and the skew

| | pilot | wide |
|---|---|---|
| episodes | 46 | **99** |
| bodies | 4 | **10** |
| words | 69,467 | **161,072** |
| clusters | 285 | **573** |
| meetings holding an episode | 8 | 28 |

**The skew is real and it is Santa Clara: 34 of 99 episodes (36 %) and 58,584
of 161,072 words (36 %).** The pilot traded a BART skew (24 of 46 episodes) for
a Santa Clara one. Nothing below should be read as a Bay Area average.

One consolation: **Santa Clara is the only body that spans both locate routes**
(24 classifier, 10 agenda_index), so it is the only place where the routes can
be compared within a body — the role San Rafael plays for density-vs-transit in
A0-RESULTS §3.1.

---

## 3. Does the wide run change an A0 conclusion?

### 3.1 The public-vs-dais frame gap — holds on the classifier route, reverses on agenda_index

A0-RESULTS §2.3 is the headline: public comment leads the dais by **6.5×** on
neighbourhood character and **3.3×** on traffic/parking. Frame-term rates per
1,000 words, public ÷ dais:

| frame | pilot POOLED (A0-RESULTS) | pilot role=computed | pilot role=db | wide `classifier` | wide `agenda_index` |
|---|---|---|---|---|---|
| neighbourhood character | **6.5×** | 3.8× | ∞ (dais 0 hits / 9,062 w) | **5.2×** | **0.8×** |
| traffic / parking | **3.3×** | 1.9× | 5.0× | **2.5×** | **0.6×** |
| safety / crime | 2.0× | 1.3× | 4.1× | 2.5× | 0.9× |
| state mandate | 2.7× | 2.1× | — (0 hits) | 2.5× | 0.0× |
| affordability / displacement | 0.6× | 0.7× | 0.3× | **2.2×** | 0.2× |
| fiscal | 0.4× | 0.6× | 0.1× | 0.4× | 0.4× |

Three things, plainly:

1. **The direction survives faceting and survives going wide — on the
   classifier route.** Neighbourhood character 6.5× → 5.2×, traffic/parking
   3.3× → 2.5×. Smaller, same sign, still the largest role effect in A0. The
   two *role* instruments also agree on direction wherever both have hits, so
   the gap is not an artefact of the b13 cue labeller.
2. **On the agenda_index route the gap vanishes or reverses** (0.8× and 0.6×).
   That is not a contradiction, it is the instrument: the agenda_index route
   locates *agenda items*, which are staff and dais business.
   **Public comment is 12.5 % of agenda_index words against 28.2 % of
   classifier words** — the route captures 2.3× less public speech per word,
   and what it does capture sits next to far more procedural talk (dais fiscal
   rate 7.95 vs 4.36 per 1,000 words). This is exactly the recall difference
   `plan/T2-PLAN.md` said to expect, now measured.
3. **Within Santa Clara, the two routes disagree on the same body**: public ÷
   dais on neighbourhood character is 2.9× by classifier and 1.0× by
   agenda_index; on safety/crime 4.2× vs 0.8×; on affordability 6.6× vs 0.4×.
   Public share is similar in both (13.1 % vs 11.5 %), so this is not only a
   public-share artefact — the routes select different *content*. Only `fiscal`
   agrees (0.3× both). **Pooling the routes was the right thing to forbid.**

**Affordability/displacement flips sign.** In the pilot the dais led it
(0.6×); on the wide classifier route the public leads it (2.2×, and 5.7× on the
`db` role slice). This is a Santa Clara effect and it is the one substantive
change the wide run makes to an A0 conclusion.

### 3.2 "Oakland is not a data point" — no longer true

A0-RESULTS §3.3: Oakland had 3 located episodes, 11 dais clusters, **2 public
clusters / 373 words**. Wide: **9 episodes, 23 dais and 19 public clusters,
3,932 public words**, and the highest public share of talk time of any body
(**53.7 %**). Oakland is now a body with data. **7 of 10 bodies clear the
500-word public-comment floor** (pilot: 3 of 4); Sausalito (269 w), Fremont
(253 w) and Napa (0 public words) do not.

### 3.3 Keyword-list power — better, still thin

Of 334 keyword terms, the pilot had **200 never firing** and 30 firing ≥ 10
times. Wide classifier route: **168 never, 41 at ≥ 10**. More than twice the
words bought 32 more live terms. The `agenda_index` route is worse in isolation
(229 never fire) because it is a smaller, more procedural slice. The A0
conclusion stands: **do not read the zeros as absence of the concept.**

### 3.4 What did not change

The `fiscal` frame is dais-led in every cell (0.1×–0.6×) — the most robust
result in A0 and still, as A0-RESULTS §2.3 caveat (b) says, institutional
register rather than a framing choice. BART remains the highest-exposure body.

---

## 4. Data problems the wide run exposed

- **Two of ten bodies have no meeting-hours denominator.** SamTrans and
  Sausalito are Granicus meetings with `videos.duration_s` NULL (42 meetings in
  scope). The rate panel used to divide by zero and print `0.0`, which reads as
  "they never discuss it". It now prints **"no meeting duration"** and writes an
  empty cell to the CSV. The per-meeting-hour rate cannot be computed for the
  agenda_index route's two granicus-only bodies until those durations exist.
- **The wide exposure denominator is dominated by bodies that were read and
  barely transcribed.** Napa is 30 meetings / 104.3 h for 1 transcribed
  episode; Hayward 29 / 68.5 h for 5. The rate panel is honest about this (it
  is exposure, not salience) but it is not comparable to the pilot's panel,
  where every body had 3 meetings.
- **The role instruments are NOT 44× apart, and A0-RESULTS §4 over-read that
  number.** On the pilot, unknown-role share was 6.63 % (computed) vs 0.15 %
  (db). On the wide corpus it is **6.63 % vs 6.56 % — indistinguishable.** What
  actually varies is the **body**: the `db` labeller leaves 0.00 % unknown at
  BART, Oakland, Hayward and SamTrans, 8.5 % at Santa Clara, 15.5 % at Napa and
  **71.3 % at Fremont** (on 1,319 words, 2 episodes). The pilot's 44× was a
  body effect wearing an instrument label. Facet by `role_source` because the
  two instruments are not *known* to be equal — not because this gap is
  established. `analysis/facets.py` says so in its docstring.
- **30 more episodes are transcribed but have no QA row** (batch b18,
  `b16,b17,b18`), so they are out of scope here. `make qa` would bring the
  corpus to 129. The wide scope is defined by the QA report rather than by
  batch id on purpose: batch ids overlap (those 30 also carry b17), so
  `--batches b13..b17` would silently pull in a never-QA'd batch.
- **Trends / spread / anim are skipped on cells with more than four bodies.**
  `freq_trends`, `freq_spread` and `freq_anim` colour one series per body from
  a 4-colour validated palette and index off the end of it past four. These are
  line and bar charts with no second visual channel, so cycling would put two
  bodies on one colour. Each skipped cell gets a `SKIPPED.md` saying so.
  A0-RESULTS §2.5 already rated these the weakest A0 outputs. The `a0_plots`
  figures keep every body by encoding it as **(colour, marker)** — 4 × 5 = 20
  distinguishable combinations, legend included.

## 5. Untrusted text

Transcripts, titles and captions are data. Nothing in the 99 episodes was acted
on as an instruction. The classifier's `contains_instructions` flag is set on
exactly **1** window in the whole state database (`gr-santaclara-2582`), whose
rationale reads as an ordinary housing-accountability appeal; it is now inside
the wide scope and is noted, not acted on. No speaker names are extracted or
stored: clusters are anonymous ids with a role.

## 6. What this run did not deliver

- **No model.** Same scope note as A0: §4 of `ANALYSIS-transit.md` is not
  scheduled.
- **No cross-route recall estimate.** We can now say the routes differ; we
  cannot say which is right. That needs the same meetings read both ways —
  Santa Clara is the only body where that is even partly possible today.
- **Month is still not a time axis.** Bodies have 1–3 meetings each in scope.
- **The 30 b18 episodes** are excluded until they have a QA row.
