# A0 results — exploratory rates and term frequency on the T1 pilot

> **Two corrections, 2026-09-23 (later the same day).** This file is the record
> of the A0 run and is left as it was written, with two exceptions noted here:
>
> 1. **§4, "treat any number from `qIlAln3uR7U_e13/e14/e15` as provisional" —
>    LIFTED.** The 115.7 % and 771.1 % caption disagreements were a bug in
>    `qa_transcripts.py`'s caption windowing, not damaged transcripts. Against
>    the correct caption window those episodes score 8.5 %, 10.1 % and 9.2 %.
>    The 475-second unknown-role turn is a genuine BART staff presentation
>    (7.1 % against captions, TTR 0.38, no repeated n-grams). Evidence and
>    recommendation: `plan/BART-QIL-ASSESSMENT.md`.
> 2. **§4, "their unknown-rates differ by ~44×" — OVER-READ.** On the wide
>    corpus the two role instruments are 6.63 % vs 6.56 %, indistinguishable.
>    The 44× is a body effect: the `db` labeller leaves 0.00 % unknown at BART
>    and Oakland and 71.3 % at Fremont. See `plan/A0-WIDE-RESULTS.md` §4.
>
> §6's "run A0 wide" is done: `plan/A0-WIDE-RESULTS.md` (99 episodes, faceted).
> The figure paths below have moved: `make a0` now draws once per instrument
> facet into `output/analysis/a0/by_<facet>/<level>/`, and the pooled set quoted
> in this file is `make a0-pooled` (files prefixed `POOLED_`).

Run 2026-09-23. Scope: `plan/ANALYSIS-transit.md` §1 + §2, phase A0 of the §6
table. Exploratory plots only — **no model was fitted**, nothing was spent
($0: no Claude call, no GPU, no RunPod), and no YouTube request was made (the
caption bot-check backoff is live until 2026-09-24 04:12 UTC).

Reproduce with `make a0`. Everything below comes from
`output/analysis/a0/` (13 PNGs + CSV twins, `parquet/*.parquet`, `a0.duckdb`)
and `output/analysis/a0_kwic.md`.

## 0. What "the pilot" means here, and why it is pinned

The corpus has grown since T1: **99** episodes are transcribed today, not 46.
A0 is specified on the pilot, so the pilot is now pinned in code
(`analysis/load_episodes.py` `PILOT`) rather than meaning "whatever is done":

| | |
|---|---|
| Bodies | BART, Oakland, San Jose, San Rafael |
| Meetings | **12** (3 per body), all classified 2026-09-21, **44.4 meeting-hours** |
| Located episodes | **63** about density/transit, + 56 mention-only |
| Transcribed | **46** episodes, **69,467 words**, **285 speaker-clusters** |
| GPU batches | b13 + b14/b15 |

`make a0` builds its own `corpus_a0_pilot.sqlite`; the shared `corpus.sqlite`
is not touched, so a later wide run cannot silently redefine the pilot.

**Exposure rule changed.** The denominator is now every *classified* meeting,
including the one San Jose meeting whose audio download failed. It was read by
the classifier, so its hours are real exposure and its located minutes are real
numerator. Excluding it (the previous behaviour) inflated San Jose from 3
meetings / 9.7 h to 2 / 7.4 h, and its density rate from 12.1 to 15.7
episode-min per meeting-hour — a 30 % overstatement.

## 1. Rates (`a0_rates.png`, `a0_rates_by_month.png`)

**How much meeting time is about this at all.** Episode-minutes per
meeting-hour, located episodes, transcribed or not:

| body | meetings | meeting-h | density | transit | both | **total** |
|---|---|---|---|---|---|---|
| BART | 3 | 15.3 | 0.0 | 23.1 | 5.2 | **28.3** |
| San Jose | 3 | 9.7 | 12.1 | 0.0 | 1.2 | **13.4** |
| San Rafael | 3 | 8.5 | 4.1 | 3.7 | 1.7 | **9.4** |
| Oakland | 3 | 11.0 | 0.8 | 0.0 | 0.0 | **0.8** |

BART spends **35 times** the share of its meeting time on this subject that
Oakland does (28.3 vs 0.8 min/h). That is the single largest effect in A0, and
it is mostly definitional: BART is a transit agency, so nearly all of its
business is in-frame, while a general-purpose city council spends most of a
meeting on everything else. **This is exposure, not salience** — it should not
be read as "Oakland does not care about housing".

**Mention-only.** Oakland is 3 located vs **15** mention-only episodes and San
Jose 9 vs 20, against BART's 39 vs 11. Density talk in city councils is
dispersed across a meeting in passing references; transit talk at BART is
concentrated in agenda items. Any per-episode rate therefore measures a
different object in a council than in a transit board.

**Public-comment share of episode talk time**: San Rafael **47.5 %**, BART
31.1 %, San Jose 25.5 %, Oakland 23.3 %. Overall 100 public clusters hold
8,075 s (2.2 h) against the dais's 15,468 s (4.3 h).

**Speakers per episode**: median 5.5, mean 6.2, range 1–14 over 46 episodes.

**By month is not a trend.** Each body-month bar is *one meeting* (San Jose
2026-09 is the only one with two). Month and meeting are perfectly confounded
in the pilot, so `a0_rates_by_month.png` shows between-meeting spread and
nothing else. BART's three meetings run 37.4 → 26.8 → 19.3 min/h; that is
meeting-to-meeting variation, not a decline.

## 2. Term frequency per speaker (§2)

### 2.1 The keyword lists barely fire

Of the **334** terms in `keywords_density.txt` + `keywords_transit.txt`:
**200 never occur**, 104 occur 1–9 times, **30 occur 10 or more times**.

Everything that would carry a policy story is at zero: `sb 79`, `sb 9`,
`sb 330`, `rhna` (2), `abag`, `yimby`/`nimby`, `upzoning`, `parking minimums`,
`station area`, `transit-oriented`, `rent control`, `housing crisis`,
`fiscal cliff`, `adus`, `builder remedy` (the possessive `builder's remedy`
does fire, 17). The lists were written for the full 12-month, 33-body corpus;
on 44 meeting-hours from 4 bodies most of them cannot fire. **Do not read the
zeros as absence of the concept** — read them as an underpowered sample.

### 2.2 Public comment and the dais speak different languages

`a0_keyness_public_vs_dais.png`, on the 30 episodes carrying both roles
(public 10,023 content tokens, dais 19,318).

Most distinctive of **public comment** (G² > 10.83): `york` 18/0, `subway`
19/1, `neighborhood` 35/9, `engineer` 13/0, `life` 19/2, `letter` 12/0,
`resident` 12/0, `hello` 11/0, `professional` 11/0, `years` 50/23,
`because` 67/38, `safety` 32/12, `name` 30/11.

Most distinctive of the **dais**: `director` 10/84, `speaker` 3/42, `aye` 1/29,
`followed` 0/22, `comment` 12/56, `motion` 1/24, `financing` 2/26, `item` 6/38,
`presentation` 6/33, `regional` 1/19.

Two things follow. First, the role labels are working: `hello` / `name` /
`resident` / `professional` / `engineer` is the public-comment
self-introduction formula, and `aye` / `motion` / `item` / `presentation` is
procedure. Second, public comment argues from **personal standing and
experience** (`years`, `life`, `because`, `resident`) where the dais argues
from **process and money** (`motion`, `item`, `financing`).

`york`/`subway` is not one talkative person: it appears in **9 distinct public
clusters across both BART meetings**, comparing BART to the New York subway.
One BART episode (`9FM5406YagE_e1`) is a transit *museum* item — heritage
streetcars, not service policy — which is a classifier precision issue, not a
finding.

### 2.3 Frames: public comment is local and sensory, the dais is fiscal

Frame-term rates per 1,000 words, pooled over bodies
(`a0_frames_role_body.png`, `a0_frames_by_family.csv`):

| frame | public | dais | public ÷ dais |
|---|---|---|---|
| neighborhood character | 2.70 | 0.41 | **6.5×** |
| traffic / parking | 2.79 | 0.85 | **3.3×** |
| state mandate / Sacramento | 0.99 | 0.37 | 2.7× |
| safety / crime | 2.83 | 1.40 | 2.0× |
| process / notification | 1.93 | 1.54 | 1.3× |
| YIMBY / pro-housing | 0.67 | 0.67 | 1.0× |
| transit service | 2.20 | 2.57 | 0.9× |
| environment / CEQA / fire | 0.67 | 0.87 | 0.8× |
| affordability / displacement | 1.53 | 2.64 | **0.6×** |
| transit safety | 0.40 | 0.71 | 0.6× |
| fiscal | 1.62 | 3.74 | **0.4×** |

This is the clearest result in A0 and it is the expected shape: the public
brings neighbourhood character, traffic and parking, and safety; the dais
brings money and affordability/displacement. The gap is large enough
(6.5× and 3.3×) to survive a good deal of measurement slop.

**Two caveats on the lexicon.** (a) The `neighborhood character` frame is
**53 of its 78 hits from the bare word `neighborhood`** — a generic locative,
not necessarily a character claim. The 6.5× is partly "the public says
*neighborhood*". (b) The `fiscal` frame is generic budget vocabulary (`cost`
44, `funding` 31, `fiscal` 27, `revenue` 22, `grant` 20); the dais leading it
is institutional register, not a framing choice. Both need a KWIC read before
they carry weight — `a0_kwic.md` has the lines.

`property_values` collected **12 hits** in the whole pilot and is dropped from
the heat map. Of 196 frame terms, 60 never fire.

### 2.4 No small group of speakers drives the rates

`a0_speakers.png`, one dot per cluster. Across 100 public clusters, the
**top 5 hold 16.8 %** of public words, the **top 10 hold 26.0 %** and 24.4 % of
public frame hits. The median public cluster is 180 words / 65 s; the longest
is 1,168 words / 8.9 min (`sU-D8Y6cC7w_e5`). So the public/dais frame gap is a
broad pattern over many short comments, not an artefact of two long speeches.
Public clusters are *longer* than dais clusters at the median (180 vs 139
words) because the dais interjects and the public gets a fixed slot.

### 2.5 Trends / spread / anim by body, faceted by role

`output/analysis/a0/trends/{dais,public}/` — `trends.html`, `trends.png`,
`top_words_spread.png`, `top_words.gif`, one set per role, body as the series
(`analysis/a0_bridge.py` maps episode→"video", body→"speaker", role→`is_host`
so the three existing scripts run unmodified).

These are **the weakest outputs in A0 and should not be shown to anyone**: the
cell is body × month and there are only **7 non-empty cells for the dais and 6
for public**, each holding one meeting. The animation interpolates between two
points. They exist because §2 asks for them; they will become meaningful at A2.

## 3. Where the framing of A0 does not fit the data

Three things worth saying plainly.

1. **"Density vs transit" is, in the pilot, "cities vs BART."** 87.9 % of
   transit words come from BART and 69.3 % of density words from San Jose.
   Only San Rafael carries all three families. So
   `a0_density_vs_transit.png` is mostly a body contrast wearing a topic
   label, and the same is true of any density/transit comparison at A0.
   San Rafael is the only body where the comparison is within-body, and it has
   6,464 density / 4,995 transit words — enough to look at, not enough to rest on.

2. **"Role × body × family" has 11 usable cells out of 24.** 12 of the
   remaining 13 are *structurally* empty, not merely thin: BART has no density
   episode, San Jose and Oakland have no transit episode. The 13th is Oakland
   public comment at **373 words**, under the 500-word floor. The three-way
   grid the plan asks for is really a two-way grid in this pilot.

3. **Oakland is not a data point.** 3 located episodes, 12 episode-minutes,
   11 dais clusters, **2 public clusters / 373 words** from 11 meeting-hours.
   Every Oakland number in every figure is an anecdote. Its `fiscal` rate of
   14.9 per 1,000 words — the largest cell in the heat map — rests on 1,141
   dais words.

## 4. Data problems rather than findings

- **Two role instruments are pooled.** 29 pilot episodes (b13) have roles
  *computed* at load time from cue phrases; 17 (b15) have roles written by the
  labeller into `speaker_roles`. Their unknown-rates differ by ~44×:
  **6.6 % of computed-source words are `unknown` vs 0.15 % of db-source**.
  That is an instrument difference, not a corpus difference, and it is exactly
  the kind of pooling `plan/T2-PLAN.md` forbids for `episodes.method`. A0
  reports it; A1 should decide whether to relabel b13 or facet by source.
- **3,655 unknown-role words are nearly all in two BART episodes.** Three
  clusters (1,414 / 986 / 967 words) in `qIlAln3uR7U_e13` and `_e14` carry
  92 % of all unknown speech. The largest is a single 475-second turn — almost
  certainly a staff presentation the cue labeller could not place. Unknown
  speech is excluded from the frame heat map, so ~5 % of the corpus is silently
  dropped there.
- **Those same two episodes fail the QA gate's caption check**:
  `qIlAln3uR7U_e14` reads 115.7 % and `_e15` 771.1 % disagreement against the
  auto captions, with 15 minutes of ">2× the captions' words". Both are WARN,
  not FAIL, but the long unknown-role monologues sit inside them. Treat any
  number from `qIlAln3uR7U_e13/e14/e15` as provisional.
- **One BART episode is a transit-museum item** (`9FM5406YagE_e1`), in-frame
  by keyword and out-of-frame by intent. Single instance; noted for the
  classifier, not fixed here.
- **Keyword-list power.** 200 of 334 terms and 60 of 196 frame terms never
  fire. The lists are not wrong; the pilot is small.

## 5. Untrusted text

Transcripts, titles and captions are data. The 4,812 pilot segments were
scanned for instruction-like content (prompt-injection markers, "ignore
previous", imperatives addressed to a model): **0 hits**. The classifier's own
`contains_instructions` flag is set on exactly **1** window in the entire state
database — `gr-santaclara-2582`, a T2 Santa Clara meeting **outside this
pilot**, whose rationale reads as an ordinary housing-accountability appeal.
Nothing was acted on from transcript content.

Speaker names: none extracted or stored. Clusters are anonymous ids with a
role; `a0_terms.redact_names` reduces person-like tokens in KWIC to initials
(e.g. "a presentation … by M. C."), and name-like tokens are never used as
plot labels. Street and place names (`gerstle`, `marydale`, `safeway`) survive
and are wanted — they are the localism signal.

## 6. Corpus available for A2

Not part of A0; recorded so the A2 decision is informed.

**99** episodes are transcribed and through the QA gate today: **63 OK, 36
WARN, 0 FAIL**. That is the pilot's 46 (28 OK / 18 WARN) plus **53 new** from
b17 (35 OK / 18 WARN). Bodies go from 4 to **10** (adds Fremont, Hayward, Napa,
SamTrans, Santa Clara, Sausalito).

The new material is **heavily concentrated: 34 of 53 episodes are Santa
Clara.** Running A0 wide today would trade one skew (BART) for another (Santa
Clara), and it mixes three `episodes.method` routes — `classifier`,
`agenda_index` and `local_asr` — which `plan/T2-PLAN.md` says to report
separately, never pooled. It also mixes the two role instruments from §4.
A2 is worth pulling forward for the **cross-body** questions, but the method
and role-source facets have to be built first.

## 7. What A0 did not deliver

- **No model.** Per the scope note in `ANALYSIS-transit.md` §19–23, §4 is not
  scheduled. Nothing hierarchical, no negative binomial, no brms.
- **No sentiment.** §3 is phase A1.
- **Month as a time axis is not interpretable** in the pilot (one meeting per
  body-month). The figures are drawn and labelled as spread; they are not
  trends, and `make anim`'s interpolation between two points is decorative.
- **`property_values`** has too few hits (12) to plot.
- **Oakland public comment** (373 words) is below the floor for a term table.
