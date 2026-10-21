# A1 results — sentiment, lexicon pass only

Run 2026-09-23. Scope: `plan/ANALYSIS-transit.md` §3, **first bullet only**.
VADER compound per turn and per speaker (mean and min), plus a hostility
word-list rate, on the **T1 pilot** (46 episodes, 827 turns, 285 clusters,
69,467 words, 4 bodies).

**The Claude pass was NOT run.** It costs money and Puck has not approved it.
$0 spent: no Claude call, no GPU, no RunPod, no YouTube request. §7 below says
what the lexicon pass suggests the Claude pass would add.

Reproduce with `make a1`. Faceted by role source (`analysis/facets.py`), so the
two role instruments are never pooled — A1 is a role contrast, which is exactly
the comparison that pooling would corrupt.

```
output/analysis/a1/by_method/classifier/       46 ep   (the whole pilot: one route)
output/analysis/a1/by_role_source/computed/    29 ep   b13, cue phrases at load time
output/analysis/a1/by_role_source/db/          17 ep   b15, labeller at ingest
```

Figures per cell: `a1_violins.png`, `a1_sentiment_vs_length.png`,
`a1_sentiment_by_frame.png`, `a1_sentiment_over_episode.png`, each with a CSV
twin, plus `a1_turns.csv` (one row per turn) and `a1_speakers.csv`.

A **turn** is a maximal run of consecutive segments by one diarized cluster
inside one episode — the same object `episode_speakers.turns` counts, verified
identical (177 turns on the `db` cell from both paths).

---

## 1. The headline is an artefact, and the artefact is turn length

Raw, on the whole pilot:

| role | turns | mean compound | 95 % CI | median | negative turns |
|---|---|---|---|---|---|
| dais | 621 | **+0.407** | [+0.366, +0.451] | +0.361 | 3.4 % |
| public | 173 | **+0.529** | [+0.464, +0.601] | +0.648 | 8.7 % |
| unknown | 33 | +0.374 | [+0.181, +0.475] | +0.273 | 0.0 % |

Read naively: *public comment is more positive than the dais.* That is wrong,
and `a1_sentiment_vs_length.png` shows why. **VADER's compound score saturates
with length**:

| words in turn | turns | median compound | \|compound\| > 0.9 | exactly 0.00 |
|---|---|---|---|---|
| 1–5 | 204 | 0.000 | 0.0 % | **73.0 %** |
| 5–15 | 163 | +0.296 | 0.6 % | 38.7 % |
| 15–40 | 135 | +0.421 | 3.7 % | 12.6 % |
| 40–100 | 112 | +0.712 | 16.1 % | 0.0 % |
| 100–250 | 124 | +0.925 | 55.6 % | 0.0 % |
| 250+ | 89 | +0.984 | **84.3 %** | 0.0 % |

The compound is a normalised sum over matched lexicon words, so a long turn
almost cannot score near zero. In this corpus it is **mostly a length measure**.
And the roles differ in length by construction: the dais interjects, the public
gets a fixed slot (A0-RESULTS §2.4: median public cluster 180 words vs dais 139).

**Control for length and the sign flips.** Turns of ≥ 100 words only:

| cell | dais | public | public negative-turn share |
|---|---|---|---|
| whole pilot (`classifier`) | +0.857 [+0.785, +0.909] (n=118) | **+0.729** [+0.615, +0.827] (n=86) | 11.6 % vs dais 3.4 % |
| `role_source = computed` | +0.876 [+0.776, +0.933] (n=90) | **+0.758** [+0.636, +0.870] (n=67) | 9.0 % vs dais 3.3 % |
| `role_source = db` | +0.799 [+0.705, +0.900] (n=28) | **+0.626** [+0.441, +0.832] (n=19) | 21.1 % vs dais 3.6 % |

**All three cells agree**: at comparable length, public comment is less positive
than the dais and 2.7–5.9× more likely to produce a negative turn. The two role
instruments give the same answer, so this is not the b13 cue labeller. The CIs
(cluster bootstrap resampling whole episodes, METHODS.md §7) overlap in every
cell, so this is a direction, not an estimate.

**Do not report the uncontrolled role means.** Any A1 number quoted without a
length control is quoting turn length.

## 2. Per-speaker mean and min — mostly not available for public comment

The plan asks for mean and min of the compound per speaker. The min is only a
speaker statistic when a speaker has several turns, and in this corpus:

- dais: 90 of 173 clusters have ≥ 3 turns (speaker-min mean **+0.037**)
- public: **15 of 100** clusters have ≥ 3 turns (speaker-min mean +0.020)
- on the `db` cell, **0 of 22** public clusters reach 3 turns

A public commenter speaks once. Their per-speaker min *is* their turn, and the
per-speaker mean is the same number again. Both are written to
`a1_speakers.csv` (with `weighted_mean_compound` alongside, because a speaker
with one 400-word turn and six "thank you"s has two defensible means and they
disagree) but **the per-speaker summary carries no information beyond the turn
for public comment.** It is a real statistic for the dais only.

## 3. Sentiment by frame — the lexicons agree with each other

Mean compound of the turns that use each frame (`a1_sentiment_by_frame.csv`,
whole pilot; everything is positive because of §1, so read the *gap*):

| frame | dais | public | gap |
|---|---|---|---|
| state mandate / Sacramento | +0.859 (n=7) | **+0.473** (n=11) | −0.39 |
| YIMBY / pro-housing | +0.969 (n=12) | **+0.579** (n=10) | −0.39 |
| neighbourhood character | +0.858 (n=13) | **+0.611** (n=30) | −0.25 |
| safety / crime | +0.845 (n=30) | **+0.636** (n=28) | −0.21 |
| traffic / parking | +0.847 (n=22) | **+0.697** (n=22) | −0.15 |
| process / notification | +0.796 (n=40) | +0.827 (n=28) | +0.03 |
| environment / CEQA / fire | +0.814 (n=15) | +0.855 (n=7) | +0.04 |

**The frames the public brings most (A0-RESULTS §2.3: neighbourhood character
6.5×, traffic/parking 3.3×, state mandate 2.7×) are the frames the public is
least positive about.** Two independent lexicons — the A0 frame list and VADER —
line up on the same five frames. That is the most useful thing A1 produces, and
it is a coherence check on A0 rather than a new finding.

Caveat that matters: this is the sentiment of turns that **mention** a frame,
not sentiment **about** the frame. VADER does not know what the negativity is
aimed at. Cell sizes are 7–40 turns; the CIs in the CSV are wide.

## 4. Sentiment over the course of an episode — no effect, and the plan's
hypothesis was about the wrong unit

`a1_sentiment_over_episode.png`. Median position through the episode: dais
0.450, public 0.491. Essentially identical, and the binned medians are flat for
both roles.

The plan's parenthesis — "public comment usually late" — is about a **meeting**.
An episode is already one agenda item, and inside an item the order is staff →
public → dais, so public comment sits in the middle by construction. To test the
plan's actual hypothesis the x-axis has to be position in the **meeting**
(`episodes.start_s / videos.duration_s`), which A1 does not do. **The figure as
specified cannot answer the question it was specified for.** Noted for A2.

## 5. Hostility — the pilot has almost none, and the first word list was wrong

`plan/hostility.txt`, seeded for this run: 75 terms in 5 categories (accusation,
contempt, bad faith, threat/pressure, slur/dismissal), matched whole-word by the
same reader as `frames.yaml`.

**The first draft fired 13 times and a KWIC read found 11 of the 13 were not
hostility:**

- `entitled` — 3 dais hits, **all land-use** ("the units that have already been
  entitled", "a report entitled *BART at a Crossroads*"). The technical
  vocabulary of the corpus collides head-on with the hostility word.
- `lie` — 1 dais hit, the verb ("where the tenant protection should lie").
- `accountable` — 2 public hits, ordinary civic register, one of them *quoting a
  grand jury report*.
- `lawsuit` — 2 public hits, both describing third-party litigation ("they have
  eight active lawsuits"), neither a threat.
- `don't care` — 1 public hit, not aimed at anyone ("I don't care whether I'm
  travelling in Europe").

Those terms are now in `ambiguous_*` categories: still counted, still written to
`a1_hostility.csv`, but **out of the headline rate** (`a1_sentiment.py`
`AMBIGUOUS`). They stay in the file rather than being deleted because the same
word *is* hostile in another mouth, and a list that quietly dropped them would
hide the failure mode from the next corpus.

After the split, on the whole pilot:

| role | hostility hits | words | per 1,000 w | turns with any | ambiguous (not counted) |
|---|---|---|---|---|---|
| dais | **0** | 43,551 | 0.000 | 0 of 621 | 4 |
| public | **4** | 22,261 | 0.180 | **2 of 173** | 6 |
| unknown | 0 | 3,655 | 0.000 | 0 of 33 | 0 |

The two genuinely hostile public turns are an "absurd assumption" critique of an
EIR (San Jose) and "an insult to public participation … a cynical manoeuvre"
(San Rafael). **Four hostile words in 69,467 words of pilot speech.** Report
this as a floor, not a level: single words cannot see sarcasm, tone, or a civil
sentence with a hostile point, and civic meetings are procedurally polite.

The honest summary is that **the hostility word list is not yet an instrument**.
Precision on the first draft was 2/13; after the fix it is 2/2 on a sample of
two, which is not a measurement. It is a screen for turns worth reading.

## 6. Facet effects

The role instruments do not disagree on anything in A1:

| | dais mean | public mean | public negative share |
|---|---|---|---|
| `role_source = computed` (29 ep) | +0.408 | +0.508 | 7.4 % |
| `role_source = db` (17 ep) | +0.403 | +0.652 | 16.0 % |

Dais means are identical to three decimals. The public means differ, but the
`db` cell has **25 public turns** and its CI [+0.472, +0.845] contains the
`computed` estimate. The length-controlled comparison (§1) agrees in both.
`method` has one level in the pilot (`classifier`), so that facet is a
single cell here; it matters on the wide corpus (see `plan/A0-WIDE-RESULTS.md`).

## 7. What the Claude pass would add — and whether it is worth $5

The lexicon pass answers "how positive is the wording". The three questions
`ANALYSIS-transit.md` §4 actually asks need "what is this person arguing, and
against what", and VADER cannot reach any of them:

1. **Stance, which is the whole point.** VADER scores valence. "I strongly
   support this excellent project" and "I am terrified of the traffic" both
   score near +1 and −0.7 respectively regardless of *what item* they are about
   — the second is opposition to housing, the first could be opposition to a
   fare rise. Nothing in §1–§6 above measures support or opposition. The role
   gap A1 can show is "public comment is worded less warmly at equal length",
   which is a much weaker claim than "public comment is more opposed".
2. **A hostility scale that works.** §5 is the evidence: a word list gets
   2 of 13 right on first contact with this register, and after repair yields
   4 hits in 69k words. An ordinal 0–4 judgement per turn, anchored on the
   Hasan/Tucker side corpus, would separate "civil and strongly opposed" from
   "contemptuous" — which is the distinction the research question needs and the
   one a lexicon provably cannot make here.
3. **The length confound disappears.** §1 is the largest single problem in A1:
   the headline statistic is mostly turn length. A per-turn judgement on a fixed
   −2…+2 scale is length-invariant by construction, so the role contrast becomes
   readable instead of needing a ≥100-word subset that costs 76 % of the turns.
4. **Self-identification and multi-label frames** (resident / homeowner /
   renter / org / staff / elected) are not lexical at all and are simply absent
   from A1. §4's question 4 — do homeowners and renters use different frames? —
   is unanswerable without them.

**What A1 has already bought:** the frame-sentiment alignment in §3 is a real
coherence check on A0's frame result, obtained for $0, and it is evidence the
frame lexicon is picking up something. The pilot is 827 turns, so the Claude
pass on the pilot is small and the plan's ≤ $5 estimate looks right.

The plan says the agreement plot decides whether "the free pass carries the
exploration". On this evidence it does not: the free pass measures length and
politeness, the paid pass measures stance. But that is Puck's call and his
money, and the numbers to make it are above.

## 8. Untrusted text

The 4,812 pilot segments were scanned for instruction-like content in A0:
**0 hits**. Nothing in any turn was acted on. No speaker names are extracted or
stored; `a1_turns.csv` and `a1_speakers.csv` carry anonymous cluster ids and a
role, and the two hostile turns are cited above by episode and body only.

## 9. Environment

`vaderSentiment>=3.3.2` added to `requirements.txt`: pure Python plus a
7,506-entry lexicon, no model download, no network at run time, $0 per run.
