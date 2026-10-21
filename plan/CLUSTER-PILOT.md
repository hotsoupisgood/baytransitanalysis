# CLUSTER-PILOT — do density/transit speaker turns cluster into stances without labels?

**EXPLORATORY. 2026-10-18. $0 (local compute only). Nothing here is an instrument result.**

Question: without any labels, do speaker turns on density/transit items fall into
anything like NIMBY / YIMBY, or into something else? Labels (the sibling NIMBY
pilot's 300 Sonnet reads, plan/NIMBY-PILOT.md) were used **only after the fact**
to measure what the clusters track. Nothing was trained on them.

## Method

- **Unit**: the NIMBY pilot's turn — one diarized cluster's maximal run inside one
  episode, redacted text (`output/redacted/`, names → `[name]`), ≥ 25 words,
  family density | both, role dais | public. Anonymous per-episode speaker
  clusters only (DECISIONS #15); nothing links a speaker across meetings.
  3,854 turns from 628 episodes.
- **Facets, never pooled** (analysis/facets.py): every clustering and figure is
  inside one locate route (`method`) and one role instrument (`roles_source = db`;
  the 148 `computed` turns were left out, not pooled). Facets: classifier 1,552
  turns / 201 episodes / 19 bodies; agenda_index 839 / 31 / 4; local_asr 828 /
  107 / 3; swagit_transcript 487 / 43 / 1 (Petaluma only).
- **Views**: (a) *lex* — TF-IDF unigrams+bigrams, stopwords, fillers and `[name]`
  removed, min_df 5, LSA 100-d, l2; (b) *style* — 22 register features (log
  words, words/min, hedges, we/I/you/they rates, we−I, questions, modals,
  negation, procedure words, numbers, sentence length, type–token ratio, VADER
  compound/pos/neg, plan/hostility.txt rate, long-word rate, A0 frame count),
  standardised; (c) *embed* — **skipped**: no sentence-embedding model is
  installed in the venv and the pilot is offline/$0 (installing one would be a
  ~90–400 MB download; say the word); (d) *nmf* — NMF topics k = 6..12 on the
  TF-IDF matrix, turn → argmax topic.
- **Clusterers**: k-means k = 2..8 (10 restarts); HDBSCAN (scikit-learn 1.9) on
  lex and style. k chosen by rank-sum of silhouette and bootstrap stability
  (mean ARI of 10 refits on 80 % subsamples against the full fit).
- **Validation**: ARI / NMI / purity of each clustering against the Sonnet
  3-way label (YIMBY / NIMBY / neither+not_about_housing; 185 labelled turns in
  the classifier facet, 36 / 38 / 13 in the others — too few to read), against
  role, and against body.
- New this run: `scikit-learn`, `umap-learn` installed into `.venv` (~30 MB).

## Best clustering per view (classifier facet; others in results.json)

| view | k | silhouette | stability ARI | HDBSCAN |
|---|---|---|---|---|
| lex (TF-IDF/LSA) | 2 | 0.029 | 0.89 | 3 clusters, 72 % noise |
| style (register) | 2 | 0.103 | 0.91 | 0 clusters, 100 % noise |
| nmf topics | 6 | 0.39 (in topic space) | 0.65 | — |
| embed | skipped | | | |

Silhouettes of 0.03–0.10 mean there is **no cluster structure** in the lexical or
style spaces: k-means is cutting a continuum, and the cut is stable only because
it is the obvious one (short procedural turns vs long substantive turns). HDBSCAN
agrees: it calls 66–100 % of turns noise in every facet. NMF topics are the only
view with interpretable separation, and it is topical, not stance.

## What the clusters track (classifier facet, n = 1,552; LLM labels n = 185)

| view (k) | vs Sonnet stance ARI / NMI / purity | vs role ARI / NMI | vs body ARI / NMI / purity |
|---|---|---|---|
| lex (2) | −0.061 / 0.072 / 0.638 | −0.016 / 0.002 | −0.005 / 0.016 / 0.35 |
| style (2) | −0.029 / 0.073 / 0.638 | 0.024 / 0.026 | 0.004 / 0.015 / 0.35 |
| nmf (6) | −0.033 / 0.103 / 0.638 | −0.004 / 0.026 | 0.032 / 0.093 / 0.39 |
| lex, any k 2..8 | ARI ≤ 0.043, NMI ≤ 0.16 | | NMI ≤ 0.10 |
| reference: Sonnet label itself | — | 0.083 / 0.091 | 0.010 / 0.093 |

Purity 0.638 against stance is exactly the share of "neither" in the labelled
turns: the clusters predict nothing beyond the majority class. Other facets
(small label counts, read as direction only): local_asr nmf k=7 is the only
cell above noise — ARI 0.02, NMI 0.22 vs stance, and NMI 0.13 vs role, 0.11 vs
body — because one topic is "San Leandro rent control", 86 % public, which is
*where* the public spoke, not *what side*. swagit (one body) NMI 0.17–0.22 vs
stance on 13 labels is not evidence.

**Hypothesis check.** The clusters do not track stance. They also do not track
role (NMI ≤ 0.03 except local_asr) and only weakly track body in the lexical
views (NMI 0.02–0.19, strongest in NMF and in HDBSCAN's non-noise core). What
they track is **register and agenda item**: procedural chair/clerk speech versus
substantive argument, and within substance, the topic of the item (housing
element / HCD, objective design standards, fees, rent stabilization, parking).

## Clusters, one line each (classifier facet)

*lex k=2*: c0 (1,051; 14 % public; mean 205 words) substantive housing talk — "housing, units, building, affordable"; c1 (501; 11 % public; 72 words) procedure — "council member, motion, item, public hearing".
*style k=2*: c0 (831; 8 % public; 58 words; we−I −1.1; 11 questions/100 sentences) short chair/clerk turns; c1 (721; 19 % public; 283 words; VADER +0.80; we−I +0.65) long presentations and testimony. Hostility rate is ≤ 0.015 / 100 words in every cluster of every view; nothing clusters on hostility.
*nmf k=6*: t0 (505) colloquial deliberation "it's, because, don't, people"; t1 (184) opening/closing public comment; t2 (259) objective design standards / zoning / general plan, 14 bodies; t3 (158) motions and votes; t4 (352; 27 % public, highest) affordable-housing programmes and support — the only topic where public share rises; t5 (94; 69 of them Los Altos) housing element / HCD / Measure D compliance.

## Example turns (redacted, trimmed; episode, turn index)

- lex c0 / nmf t0 — dais, Napa, gr-napa-5691_e5 t3: "Okay, hard act to follow. So thank you for all the thought and public participation that's gone into getting us to this point in this process. It's huge. And I think I want us to remember…"
- lex c0 — dais, Santa Clara, gr-santaclara-2452_e1 t23: "…taking from that economic trend that we talked about, we had multiple conversations with task force in the community in terms of what are the kind of heights and…"
- lex c1 / nmf t3 — dais, Santa Clara, gr-santaclara-2577_e4 t15: "So a motion was made by council member [name] and seconded by council member [name] to agendize a review of public participation options such as…"
- lex c1 / nmf t1 — dais, Fremont, gr-fremont-21_e1 t1: "Next we have oral communications. This is an opportunity for the public to speak to the City Council on any issue that is not on the agenda. The Council cannot take action nor respond…"
- style c1 — public, Palo Alto, ag678vV_olc_e4 t41: "I am speaking in opposition of the rezoning application. My husband and I live in the single family home directly adjacent to the proposed development should the rezoning be allowed…"
- style c0 — dais, Daly City, PWfDq1v3WlU_e2 t67: "So I think we round up. So five units. So the top people who have Daly City preference, the five of them get to pick, right? And then for the other remainder…"
- nmf t2 — dais, Santa Clara, gr-santaclara-2468_e2 t16: "Objective design standards? They are predictable, user-friendly, enforceable, and it's enabled by the state, to preserve local control and to provide standards that…"
- nmf t4 — dais, Santa Clara, gr-santaclara-2398_e1 t6: "Good evening, Mayor, council members. Let's begin the presentation. So first I'll do introductions. We have a number of staff that have been working on this…"
- nmf t5 — dais, Los Altos, jgyX7NNf7g0_e6 t11: "So when we talk about the housing element, one of the items we hear a lot about is the buffer. We'll say, well, the city adopted more units than is necessary…"
- nmf t5 — dais, Los Altos, jgyX7NNf7g0_e9 t12: "Measure D is the death of expertise, and we saw it here tonight. We had comments from residents who don't know anything about HCD…"

Note the Palo Alto turn: a textbook NIMBY statement sits in the same style
cluster as a staff housing-element presentation, because both are long and
first-person. The lexical view puts it with "housing, units" talk. No view
separates it from a YIMBY turn of the same length on the same item.

## Conclusion

**Stance does not emerge unsupervised** from word frequency, register features,
or NMF topics on this corpus. ARI against the Sonnet stance label is at or below
zero in every view and every k; NMI never exceeds 0.16 on the one facet with
enough labels, and purity never beats the majority class. What the unsupervised
structure encodes is (1) procedural vs substantive register, (2) the agenda item
and the body's vocabulary, (3) turn length. Stance in a hearing is expressed by
the *same* vocabulary on both sides (housing, units, density, traffic, parking),
with the sign carried by syntax, negation and context that bag-of-words and
simple style counts do not see. The NIMBY pilot's lexicon result (hand vs
keyness agreement 0.39, κ 0.11) says the same thing from the supervised side.

Honest limits: no embedding view (an installed sentence model might separate
stance better; it also might just find topic again); labels only in the
classifier facet in useful numbers; k-means on a continuum.

## Proposed cluster labels for Puck to confirm (descriptive, not stance)

| view | cluster | proposed label |
|---|---|---|
| lex | c0 / c1 | substantive housing talk / procedure & chairing |
| style | c0 / c1 | short chair-clerk turns / long presentation-testimony |
| nmf | t0 | colloquial deliberation |
| nmf | t1 | public-comment procedure |
| nmf | t2 | design standards, zoning, general plan |
| nmf | t3 | motions and votes |
| nmf | t4 | affordable-housing programmes (public-heavy) |
| nmf | t5 | housing element / HCD compliance (Los Altos) |

## What is needed next

1. Decide whether to keep pursuing unsupervised stance at all. Evidence says no;
   if yes, the one untried view is sentence embeddings (local model, ~90 MB
   all-MiniLM or ~400 MB bge-small; needs an OK to download), and even then a
   fairer test is clustering *within one agenda item* where topic is held fixed.
2. The useful unsupervised product is the topic layer (nmf t2/t4/t5), which could
   stratify the supervised stance read by item type.
3. Redaction leak: two example turns in `results.json` (public self-introductions
   in Petaluma and Oakland) contain full names that `[name]` redaction missed —
   speakers introducing themselves. Worth a look at pipeline/redact.py before
   any transcript text leaves the machine. Not reproduced here.
4. If the stance read is to be scaled, NIMBY-PILOT's supervised route is the
   only one with signal; this pilot should be cited as the reason no
   "unsupervised stance" figure exists.

## Artefacts

- `scripts/cluster_pilot.py` — new; `run` reproduces everything.
- `output/cluster_pilot/results.json` — sweeps, validation at every k, per-cluster
  stats, examples, for all four facets.
- `output/cluster_pilot/assign_{lex,style}_<facet>.csv` — turn → cluster, HDBSCAN,
  2-D coordinates, LLM label.
- `output/cluster_pilot/proj_{lex,style,nmf}_<facet>.png` — UMAP, coloured by
  cluster / role / Sonnet label (stamped EXPLORATORY).
- `output/cluster_pilot/heat_{lex,style,nmf}_<facet>.png` — cluster × body.
- `output/cluster_pilot/pilot.log`, `run.out`.
- Nothing committed; sibling files in `output/nimby_pilot/` untouched.
