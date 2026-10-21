# Word frequency analysis — methods

Every statistic here, why it's used, and what it would hide if used alone.

```
simulate.py ──► output_sim/whisper/*.json ──► load_db.py ──► corpus.sqlite
                                                                  │
                            test_pipeline.py ◄──── word_freq.py ◄──┘
```

The real pipeline replaces `simulate.py` with the transcription pipeline. Every
other stage is identical, which is the point: the analysis is exercised on data
whose answer is known before it is trusted on data whose answer isn't.

---

## 1. Tokenisation and filtering

Lowercase; keep internal apostrophes and hyphens (`don't`, `so-called`); drop
tokens under 2 characters.

Then two filters, both defeatable by a flag because both are judgement calls:

**Stopwords** (`--keep-stop` to disable). Function words — *the, of, and, is* —
are the highest-frequency tokens in any corpus and say nothing about content.

**Disfluencies and discourse glue** (`--keep-fillers` to disable). *um, uh,
like, you know, basically, literally, actually.* Whisper transcribes these
faithfully and they sit near the top of any spoken word list.

*"right" is not in that list* (removed 2026-09-17). It is a discourse marker —
and it is also "the right", "right-wing", "rights". Filtering it while keeping
"left" made every left/right comparison asymmetric by construction, which is a
bias in exactly the dimension this project measures. Any word with a political
sense stays in; `test_pipeline.py` §9 asserts that both survive tokenisation.

They are filtered by default but **worth studying on their own**. Hedging and
hesitation rates differ systematically between scripted and unscripted speech,
and that's exactly the format confound this project has to deal with — so
running `--keep-fillers --mode overall` is a legitimate measurement of format,
not a mistake.

---

## 2. Normalisation: rate per million

Raw counts can't be compared across speakers with different corpus sizes.

```
rate_per_million = count / total_tokens_for_that_speaker * 1_000_000
```

Reported alongside the raw `n`, because a rate computed from 3 occurrences looks
identical to one computed from 3,000 and is not the same evidence.

---

## 3. Log-likelihood (G²) — is the difference real?

Dunning's G², the standard keyness test in corpus linguistics.

For a word, build a 2×2 table:

| | target corpus | reference corpus |
|---|---|---|
| this word | `a` | `b` |
| all words | `c` | `d` |

Under the null hypothesis *"this word is equally likely in both corpora"*, the
word's `a + b` total occurrences should be split in proportion to corpus size:

```
E1 = c * (a + b) / (c + d)        expected count in target
E2 = d * (a + b) / (c + d)        expected count in reference
```

G² measures how far the observed counts sit from those expectations, in log
space:

```
G² = 2 * [ a * ln(a / E1) + b * ln(b / E2) ]
```

Each term is *observed × log(observed/expected)* — zero when observed equals
expected, growing as they diverge. A zero count contributes zero by convention.
The factor of 2 makes G² chi-square distributed with 1 degree of freedom, so:

- G² > 3.84 → p < 0.05
- G² > 10.83 → p < 0.001

**Why G² rather than chi-square:** it behaves far better on the sparse, skewed
counts typical of word frequencies, where chi-square's normal approximation
breaks down for rare words.

**Why it is never reported alone.** G² scales with corpus size. Double both
corpora and G² doubles for the same *proportional* difference. In a
million-word corpus nearly every word clears p<0.001, so sorting by G² just
sorts by frequency. `test_pipeline.py` asserts this scaling behaviour explicitly
so the property stays documented and visible.

---

## 4. Log Ratio — how big is the difference?

Hardie's Log Ratio is the effect size, and it's what the output is **ranked by**.

```
LR = log2( (a / c) / (b / d) )
```

`a/c` and `b/d` are the relative frequencies in the two corpora, so the ratio is
"how many times more often". `log2` makes it symmetric around zero and readable
in doublings:

| LR | meaning |
|---|---|
| 0 | equally frequent |
| +1 | twice as frequent |
| +2 | four times |
| +7 | 128 times |

Zero counts would send this to infinity, so both counts get `+0.5` before the
ratio. This is deliberately conservative: a word occurring 20 times in one
corpus and never in the other is reported as ~5.3 doublings, not infinity.

**Read G² and LR together.** High LR + low G² = a big difference on too little
data. High G² + low LR = a tiny difference measured very precisely. Only both
high is interesting.

---

## 5. LogDice — collocation strength

For collocates, the question is different: not "is this word more common here
than there", but "how strongly does this word bind to the node word".

```
logDice = 14 + log2( 2 * f(node, collocate) / (f(node) + f(collocate)) )
```

The inner fraction is the Dice coefficient — co-occurrences over the *average*
of the two marginal frequencies. "Of all the times either word appears, how
often together?"

**Why not PMI**, the more familiar choice:

```
PMI = log2( P(node, collocate) / (P(node) * P(collocate)) )
```

PMI divides by the *product* of the marginals, so it explodes for rare words. A
transcription error occurring twice, both times near the node, outranks a
genuine collocate occurring 500 times. Dice uses the **sum**, so a rare word
can't outrank a frequent one on rarity alone.

The `+14` is convention, putting typical scores in a readable 0–14 range (14 =
the two words only ever occur together). No statistical content.

**LogDice has its own blind spot**, and the test run shows it plainly. Ranked by
logDice on the simulated corpus:

```
  collocate     n    logDice   LogRatio     G²
  actually    1048     13.66      -0.02     0.1      <- highest logDice
  october      107     12.50      +7.82   153.9      <- the actual signal
```

`actually` was generated at a *uniform* rate everywhere, near and far from the
node. It tops the logDice ranking purely because it's common. LogDice measures
association strength but not *distinctiveness* — so collocates are ranked by
Log Ratio (in-window vs out-of-window) with logDice shown alongside.

---

## 6. Range — is it a habit or one rant?

```
range = (videos by this speaker containing the word) / (their total videos)
```

A word said 300 times in one video and never again is not a speaker
characteristic. `--min-range` (default 0.20) drops anything appearing in under a
fifth of a speaker's videos.

This is a crude dispersion measure. Juilland's D or Gries's DP are more
principled if it ever matters; range is transparent and sufficient at this n.

---

## 7. Cluster bootstrap confidence intervals

**The important one, and the mistake this project has already made once.**

Tokens within a video are not independent. One video on one subject supplies
hundreds of correlated occurrences. A standard interval on token counts assumes
independent draws and is therefore far too narrow — the same error as pooling
meanness chunk scores across videos as though there were 124 independent
observations instead of ~10.

So the interval resamples **whole videos** with replacement:

1. Draw *n* videos with replacement from the speaker's *n* videos.
2. Recompute the pooled rate: `Σ counts / Σ totals × 10⁶`.
3. Repeat 2,000 times; take the 2.5th and 97.5th percentiles.

Each replicate takes or omits entire videos, so the spread reflects
**between-video** variation — the real uncertainty when generalising to "how
this speaker talks."

`test_pipeline.py` check 7 demonstrates the gap: for a word concentrated in 3 of
10 videos, the clustered interval comes out several times wider than the naive
one. Both describe the same 300 occurrences; only one is honest.

---

## 8. Host filtering

`--host-only` (default on) drops segments whose speaker isn't the resolved host,
so an interviewee's vocabulary isn't attributed to the interviewer. In the
simulation Tucker is deliberately interview-heavy and loses ~35% of segments,
which check 6 asserts.

On real data the host is identified by **speaker-embedding enrolment**
(`scripts/host_enroll.py`, since 2026-09-20): one anchor voice per channel
(Hasan ← `C0LMNHE-OF4`, Tucker ← `q72s8zwwGg4`, 180 s of the host's speech
through `pyannote/wespeaker-voxceleb-resnet34-LM`), then every diarization
cluster in every video is scored by cosine to its channel's anchor. Clusters
with cos ≥ 0.60 are the host — all of them, so Hasan's impressions, which split
him into several labels, are merged — provided the best cluster is ≥ 0.60 and
beats the best non-host cluster by ≥ 0.15. Otherwise the file keeps the old
"host = most talk time" guess and is marked `host_method = "talk_time"`,
`host_ambiguous = true`. The transcript JSON records `host_speakers` (the label
set), `host_method`, `host_similarity` and `host_runner_up`; `is_host` in
`corpus.sqlite` is "speaker ∈ host_speakers". Files without `host_speakers`
fall back to the single `host_speaker`.

`host_confidence` therefore means two things: under `embedding` it is the
cosine similarity (typically 0.7–0.9); under `talk_time` it is the host's share
of speech, as before. `--min-host-conf 0.55` still skips the shaky videos under
both readings. Diarization remains unsupervised clustering — a useful prior,
not ground truth; enrolment fixes *which* cluster is the host, not the
cluster boundaries.

---

## 9. Known limitations

- **No lemmatisation.** *attack / attacks / attacking* count separately. The
  FTS5 index uses Porter stemming for *search*, but frequency counting is on
  surface forms. Add lemmatisation if the word lists look fragmented.
- **No multi-word units.** "October 7th" is two tokens. Add n-gram support if
  phrases matter.
- **Sentence splitting is regex on punctuation**, and punctuation in an ASR
  transcript is the model's guess. Windows are therefore approximate. A segment-
  based window is more faithful to the audio; a sentence-based one is more
  faithful to the language. Both are defensible — this uses sentences.
- **The reference corpus is the other speakers**, so "distinctive" means
  distinctive *within this corpus*, not against English generally. That's the
  right comparison for this question, but it means results shift when the
  corpus composition changes — one more reason `frame.csv` is fixed before
  analysis begins.

---

## 10. KWIC node matching

The node word is matched as a **whole word or whole phrase**, not a substring:

```
(?<![a-z0-9]) node (?![a-z0-9])        case-insensitive; spaces match any whitespace
```

So `ICE,` and `Israel's` match; `police`, `justice` and `Israeli` do not. Until
2026-09-17 this was `node in sentence.lower()`, and on the 20 exploratory
transcripts that made **98%** of the "mentions" of `ice` false hits (209 vs 5),
76% of `trans`, 46% of `war` and 12% of `israel`. Every collocate table built on
that was mostly noise about policing.

Use `--node-regex` when the variants are the point: `--node-regex 'israel(i|is)?'`
matches the family deliberately, and the boundary rule still applies around it.
The node text is removed before collocates are counted, so a node can never be
its own collocate.

## 11. Multiple comparisons

Each row of the frequency table is its own significance test. Screening 500
words at p<0.05 puts ~25 null words in the "significant" column — the simulated
corpus demonstrates it: `deductible`, generated at an identical rate for every
speaker, comes out starred. The output prints that caveat under every table.
Rank and read by **Log Ratio**, use the stars as a floor rather than a finding,
and if a p-value must be defended, apply Bonferroni (α/number of words screened)
or report the effect with its cluster-bootstrap interval instead.

## 12. Transcript quality metrics (`analysis/qa_transcripts.py`)

The gate every batch passes before it becomes data. Each number, and what it
catches:

**Word error rate.** `WER = (S + D + I) / N_reference`, reference **first**:

```
WER = edit_distance(reference_words, hypothesis_words) / len(reference_words)
```

Direction is not cosmetic. With the hypothesis as the denominator, 180
hallucinated words appended to a 180-word reference score 0.50 instead of 1.00 —
the metric rewards the failure it is supposed to detect. Text is normalised
first: HTML unescaped, YouTube's `[ __ ]` profanity mask and bracketed
non-speech tags dropped, case and punctuation removed. Digits are left as
digits, so "7" vs "seven" still counts as an error — a known, roughly constant
upward bias.

The edit distance is computed with the **Myers/Hyyrö bit-parallel algorithm**:
each DP column is packed into the bits of one Python integer, so a word of the
hypothesis costs a handful of big-integer operations instead of a loop over the
reference. O(n·m/64) instead of O(n·m) — seconds instead of hours on a 3-hour
transcript. `--self-test` checks it against the textbook DP on 300 random pairs.

**Repetition / decoder loops.** Whisper's own fallback heuristic: gzip the
segment text and flag a compression ratio above **2.4** (only for segments ≥ 60
characters, because short strings compress badly). Plus runs of identical
consecutive segments: 3 or more is a warning; a failure only when one run
**spans more than 60 s of audio** (since 2026-09-20 — a count was the wrong
unit: a five-line crowd chant over 4.8 s failed a 28-minute transcript).
`load_db.py` collapses any such run to one row on load and records the number
dropped per video (`videos.loop_segments_dropped`). The gzip check is what
found the "Hassan piker" ×45 loop in `AAiFho_u_GI`.

**Coverage** = union of segment intervals ÷ transcribed span. Below 0.6 warn,
below 0.3 on anything over two minutes fail — most of the audio produced no text.

**Speaking rate** = words ÷ covered speech minutes, flagged outside 60-300 wpm
(fast political speech is ~200). **Characters per second** inside a segment,
flagged above 35 for segments ≥ 3 s: above that the audio cannot contain the
text. Short segments are exempt because Whisper's timestamps quantise.

**Caption divergence.** Per minute, the transcript's word count against the
captions'. More than 2× (plus 20 words of slack) in either direction is not
ordinary ASR disagreement — it is a hallucinated passage, a dropped passage, or
an uncaptioned one. It localises the problem to a timestamp, which is what a
human needs to check it.

**Diarization sanity.** Host share of speech — the share of `host_speakers`
when the host was enrolled by embedding (§8), else of the longest cluster — and
the share of speech with no speaker label (flag > 5%). A share under 0.55 is a
WARN only when the file still relies on the talk-time heuristic
(`host_method = "talk_time"`); an enrolled host who talks 30% of the time is a
fact about the video, not a doubt about who the host is.
