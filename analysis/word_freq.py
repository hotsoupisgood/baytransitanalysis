#!/usr/bin/env python3
"""
word_freq.py — word frequency analysis. Two modes: `overall` and `kwic`.

See analysis/METHODS.md for the statistics written out in full. Short version of
why each piece is here:

RAW COUNTS ARE MEANINGLESS ACROSS SPEAKERS. One speaker has more words than
another, so every count is normalised to a rate per million words.

BIG CORPORA MAKE EVERYTHING "SIGNIFICANT". With a million words, a trivial
difference clears p<0.001. So significance (log-likelihood) and effect size
(log ratio) are reported separately and ranked by effect size.

A WORD SAID 300 TIMES IN ONE VIDEO IS NOT A SPEAKER HABIT. `range` = the
percentage of that speaker's videos containing the word. A high-frequency,
low-range word is one rant, not a pattern, so there's a minimum-range filter.

VIDEOS ARE THE UNIT OF INDEPENDENCE, NOT WORDS. Tokens within a video are
massively correlated. The confidence interval is a CLUSTER bootstrap that
resamples whole videos, which is the honest interval; a token-level interval
would be roughly an order of magnitude too narrow.

GUESTS ARE NOT THE HOST. `--host-only` (default on) drops non-host speech, so an
interviewee's vocabulary isn't attributed to the interviewer.
"""
import argparse, math, re, shutil, sqlite3, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------- filtering ---
# Function words. These dominate any word list and say nothing about content.
STOP = set("""a an the and or but if then than that this these those there here
of to in on at by for with from into onto about over under again further is are
was were be been being am do does did doing have has had having i me my we our
you your he him his she her it its they them their what which who whom whose
when where why how all any both each few more most other some such no nor not
only own same so too very can will just should now as up down out off above
below s t don ve ll re d m o y ain aren couldn didn doesn hadn hasn haven isn
ma mightn mustn needn shan shouldn wasn weren won wouldn""".split())

# Speech disfluencies and discourse glue. Whisper transcribes these faithfully
# and they are among the highest-frequency tokens in any spoken corpus. They are
# a real object of study (hedging, hesitation) but they swamp content analysis,
# so they are filtered here and countable separately via --keep-fillers.
#
# "right" is deliberately NOT in this list (removed 2026-09-17). In a
# polarisation corpus it is also "the right" / "right-wing", and filtering it
# while keeping "left" made every left/right comparison asymmetric by
# construction. Any word with a political sense stays in; test_pipeline.py
# asserts that "left" and "right" both survive tokenisation.
FILLERS = set("""um uh umm uhh er erm ah oh hmm mhm yeah yep nope okay ok like
literally basically actually kinda sorta gonna wanna gotta dunno well
anyway anyways look mean know thing things stuff really pretty much lot get got
go going goes went say says said see saw think thought want wanted make made
take took come came give gave one two first also even still back way""".split())

TOKEN_RE = re.compile(r"[a-z][a-z'\-]*[a-z]|[a-z]")
SENT_RE  = re.compile(r"(?<=[.!?])\s+")


def tokens(text, keep_fillers=False, keep_stop=False):
    out = []
    for w in TOKEN_RE.findall(text.lower()):
        if not keep_stop and w in STOP:      continue
        # "it's", "don't", "that's": the token keeps its apostrophe, so STOP
        # (which lists it/s/don/t/that separately) never matched it and
        # contractions topped every content-word list (freq_trends,
        # 2026-09-19). A contraction is a stop word when every piece is one.
        if not keep_stop and "'" in w and all(p in STOP for p in w.split("'") if p):
            continue
        if not keep_fillers and w in FILLERS: continue
        if len(w) < 2:                        continue
        out.append(w)
    return out


# ------------------------------------------------------------- statistics ----
def log_likelihood(a, b, c, d):
    """Dunning's G^2 for a 2x2 contingency table. Is this difference real?

    a = count of the word in the target corpus
    b = count of the word in the reference corpus
    c = total words in target,  d = total words in reference

    Expected counts under the null "the word is equally likely in both":
        E1 = c * (a+b) / (c+d)        E2 = d * (a+b) / (c+d)
    i.e. share out the word's total occurrences in proportion to corpus size.

        G^2 = 2 * [ a*ln(a/E1) + b*ln(b/E2) ]

    Each term measures how far an observed count sits from its expectation, in
    log space; a zero count contributes zero by convention. G^2 is chi-square
    distributed with 1 df, so 3.84 is p<0.05 and 10.83 is p<0.001.

    The catch, and why this is never reported alone: G^2 scales with corpus
    size. Double the corpus and G^2 doubles for the same proportional gap. In a
    million-word corpus almost everything is "significant", which is why the
    ranking below uses the effect size instead.
    """
    if c <= 0 or d <= 0 or (a + b) == 0:
        return 0.0
    e1 = c * (a + b) / (c + d)
    e2 = d * (a + b) / (c + d)
    g = 0.0
    if a > 0: g += a * math.log(a / e1)
    if b > 0: g += b * math.log(b / e2)
    return 2 * g


def log_ratio(a, b, c, d, smooth=0.5):
    """Hardie's Log Ratio -- the EFFECT SIZE. How much more common, in doublings?

        LR = log2( (a/c) / (b/d) )

    (a/c) and (b/d) are the word's relative frequencies in the two corpora, so
    the ratio is "how many times more often". Taking log2 makes it symmetric
    around zero and readable: +1 = twice as frequent, +2 = four times, -1 = half.

    Zero counts would send this to infinity, so both counts get +smooth (0.5).
    That is deliberately conservative: a word appearing 20 times in one corpus
    and 0 in the other is reported as ~5.3 doublings rather than infinity.
    """
    ra = (a + smooth) / c
    rb = (b + smooth) / d
    return math.log2(ra / rb)


def log_dice(f_node_coll, f_node, f_coll):
    """LogDice (Rychly 2008) -- collocation strength.

        logDice = 14 + log2( 2 * f(node, collocate) / (f(node) + f(collocate)) )

    The inner term is the Dice coefficient: the co-occurrence count over the
    average of the two marginal counts. It asks "of all the times either word
    appears, how often do they appear together?"

    Why not PMI, the usual choice: PMI divides by the product of the two
    marginals, which explodes for rare words -- a typo occurring twice, both
    times near the node, scores higher than a genuine collocate occurring 500
    times. Dice uses the SUM, so a rare word cannot outrank a frequent one on
    rarity alone.

    The +14 is a convention that puts typical scores in a readable 0-14 range
    (14 is the theoretical maximum, where the two words only ever co-occur). It
    has no statistical content. Scores are comparable across corpora of
    different sizes, which is the property that matters here.
    """
    if f_node <= 0 or f_coll <= 0 or f_node_coll <= 0:
        return float("-inf")
    return 14 + math.log2(2 * f_node_coll / (f_node + f_coll))


def cluster_bootstrap_ci(per_video_counts, per_video_totals, n_boot=2000, seed=7):
    """95% CI for a rate per million, resampling WHOLE VIDEOS with replacement.

    Why not a normal-approximation interval on token counts: tokens inside one
    video are not independent draws. One video about one subject supplies
    hundreds of correlated occurrences. Treating them as independent understates
    the standard error badly -- the same mistake as pooling chunk scores across
    videos in the meanness analysis.

    Resampling at the video level respects that clustering: each bootstrap
    replicate draws n videos with replacement and recomputes the pooled rate, so
    the spread reflects between-video variation, which is the real uncertainty.
    """
    c = np.asarray(per_video_counts, float)
    t = np.asarray(per_video_totals, float)
    n = len(c)
    if n == 0 or t.sum() == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    rates = c[idx].sum(1) / np.maximum(t[idx].sum(1), 1) * 1e6
    return tuple(np.percentile(rates, [2.5, 97.5]))


# ------------------------------------------------------------------ corpus ---
def _open_readonly(db):
    """Open the corpus DB, copying it to local disk first if needed.

    WHY: a network/FUSE-mounted project folder may not provide the POSIX
    byte-range locking SQLite requires, and every query fails with a bare
    "disk I/O error". Copying the single file to local scratch and reading it
    there is reliable everywhere. On a normal local disk the copy is a few
    milliseconds and changes nothing.
    """
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.execute("SELECT count(*) FROM videos").fetchone()
        return con
    except sqlite3.OperationalError:
        tmp = Path(tempfile.mkdtemp(prefix="corpusrd-")) / Path(db).name
        shutil.copy(db, tmp)
        return sqlite3.connect(f"file:{tmp}?mode=ro", uri=True)


def load(db, host_only=True, min_host_conf=0.0):
    """Return {speaker: {video_id: [segment texts in order]}}."""
    con = _open_readonly(db)
    q = ("SELECT v.speaker, s.video_id, s.seg_idx, s.text, s.is_host, v.host_confidence "
         "FROM segments s JOIN videos v USING(video_id) ORDER BY s.video_id, s.seg_idx")
    out = defaultdict(lambda: defaultdict(list))
    dropped = 0
    for sp, vid, _i, text, is_host, conf in con.execute(q):
        if min_host_conf and (conf or 0) < min_host_conf:
            continue
        if host_only and is_host is not None and not is_host:
            dropped += 1
            continue
        out[sp][vid].append(text)
    con.close()
    return out, dropped


def sentences(texts):
    joined = " ".join(t.strip() for t in texts)
    return [s for s in SENT_RE.split(joined) if s.strip()]


def node_pattern(node, is_regex=False):
    """Compile the KWIC node as a WHOLE-WORD (or whole-phrase) match.

    WHY: the original test was `node in sentence.lower()`, a substring match. On
    the 20 real transcripts that made 98% of the "mentions" of `ice` false hits
    (police, justice, price, voice...), 46% of `war` (award, toward) and 76% of
    `trans` (transcript, transfer). Every collocate table built on that was
    mostly noise. Now: letters/digits may not touch either end of the match, so
    "ICE," and "Israel's" match but "police" and "Israeli" do not. Spaces inside
    a phrase match any whitespace. For deliberate variants pass is_regex=True,
    e.g. `israel(i|is)?` -- boundaries are still enforced around the pattern.
    """
    body = node if is_regex else r"\s+".join(re.escape(w) for w in node.lower().split())
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])", re.IGNORECASE)


def kwic_counts(videos, node_re, window, keep_fillers=False, keep_stop=False):
    """Collocate counts for one speaker: tokens inside +/-window sentences of a
    node match vs everywhere else. `videos` = {video_id: [segment texts]}.
    Returns (near, far, near_by_video, hits). The node itself is removed from
    the text before counting, so it never counts as its own collocate. This is
    the one implementation used by the CLI and by test_pipeline.py."""
    near, far = Counter(), Counter()
    near_vid = defaultdict(Counter)
    hits = 0
    for vid, texts in videos.items():
        sents = sentences(texts)
        idx = [i for i, s in enumerate(sents) if node_re.search(s)]
        hits += len(idx)
        # Union of windows, so a node word appearing twice close together
        # doesn't double-count the sentences between them.
        keep = set()
        for i in idx:
            keep.update(range(max(0, i - window), min(len(sents), i + window + 1)))
        for i, s in enumerate(sents):
            toks = tokens(node_re.sub(" ", s), keep_fillers, keep_stop)
            if i in keep:
                near.update(toks)
                near_vid[vid].update(toks)
            else:
                far.update(toks)
    return near, far, near_vid, hits


# ------------------------------------------------------------------- modes ---
def mode_overall(corpus, args):
    per_video = {sp: {v: Counter(tokens(" ".join(t), args.keep_fillers, args.keep_stop))
                      for v, t in vids.items()} for sp, vids in corpus.items()}
    totals   = {sp: {v: sum(c.values()) for v, c in d.items()} for sp, d in per_video.items()}
    pooled   = {sp: sum(d.values(), Counter()) for sp, d in per_video.items()}
    grand    = {sp: sum(t.values()) for sp, t in totals.items()}

    print(f"\n{'='*78}\nOVERALL WORD FREQUENCY")
    print(f"host_only={not args.all_speakers}  fillers={'kept' if args.keep_fillers else 'filtered'}"
          f"  stopwords={'kept' if args.keep_stop else 'filtered'}")
    print(f"{'='*78}")
    for sp in sorted(pooled):
        print(f"  {sp:<12} {grand[sp]:>9,} content tokens   {len(per_video[sp]):>3} videos")

    for sp in sorted(pooled):
        ref = sum((pooled[o] for o in pooled if o != sp), Counter())
        ref_n = sum(grand[o] for o in grand if o != sp)
        rows = []
        nvid = len(per_video[sp])
        for w, a in pooled[sp].items():
            if a < args.min_count:
                continue
            rng_ = sum(1 for v in per_video[sp] if per_video[sp][v][w] > 0) / nvid
            if rng_ < args.min_range:      # one rant, not a habit
                continue
            b = ref[w]
            rows.append((w, a, a / grand[sp] * 1e6,
                         log_ratio(a, b, grand[sp], ref_n),
                         log_likelihood(a, b, grand[sp], ref_n), rng_))
        rows.sort(key=lambda r: -r[3])     # rank by EFFECT SIZE, not p-value
        print(f"\n--- {sp}: most distinctive vs the other speakers "
              f"(min count {args.min_count}, min range {args.min_range:.0%}) ---")
        print(f"  {'word':<16}{'n':>7}{'per M':>10}{'LogRatio':>10}{'G2':>10}{'range':>8}   95% CI (per M)")
        for w, a, pm, lr, g2, rg in rows[:args.top]:
            lo, hi = cluster_bootstrap_ci([per_video[sp][v][w] for v in per_video[sp]],
                                          [totals[sp][v] for v in per_video[sp]])
            sig = "***" if g2 > 10.83 else "*" if g2 > 3.84 else "ns"
            print(f"  {w:<16}{a:>7}{pm:>10.0f}{lr:>+10.2f}{g2:>10.1f}{rg:>7.0%} {sig:<4}[{lo:.0f}, {hi:.0f}]")


def mode_kwic(corpus, args):
    regex = getattr(args, "node_regex", None)
    node = regex or args.node.lower()
    node_re = node_pattern(node, is_regex=bool(regex))
    win  = args.window
    print(f"\n{'='*78}\nCOLLOCATES OF '{node}'  (whole-word match; window: +/-{win} sentences)")
    print(f"host_only={not args.all_speakers}  fillers={'kept' if args.keep_fillers else 'filtered'}")
    print(f"{'='*78}")

    for sp in sorted(corpus):
        nvid = len(corpus[sp])
        near, far, near_vid, hits = kwic_counts(corpus[sp], node_re, win,
                                                args.keep_fillers, args.keep_stop)
        n_near, n_far = sum(near.values()), sum(far.values())
        print(f"\n--- {sp}: {hits} mentions in {nvid} videos; "
              f"{n_near:,} tokens in-window vs {n_far:,} outside ---")
        if not hits or not n_near:
            print("    no mentions"); continue

        rows = []
        for w, a in near.items():
            if a < args.min_count:
                continue
            rg = sum(1 for v in near_vid if near_vid[v][w] > 0) / nvid
            if rg < args.min_range:
                continue
            b = far[w]
            rows.append((w, a,
                         log_dice(a, hits, a + b),
                         log_ratio(a, b, n_near, n_far),
                         log_likelihood(a, b, n_near, n_far), rg))
        rows.sort(key=lambda r: -r[3])
        print(f"  {'collocate':<16}{'n':>6}{'logDice':>9}{'LogRatio':>10}{'G2':>9}{'range':>8}")
        for w, a, ld, lr, g2, rg in rows[:args.top]:
            sig = "***" if g2 > 10.83 else "*" if g2 > 3.84 else "ns"
            print(f"  {w:<16}{a:>6}{ld:>9.2f}{lr:>+10.2f}{g2:>9.1f}{rg:>7.0%} {sig}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="output_sim/corpus.sqlite")
    ap.add_argument("--mode", choices=["overall", "kwic"], default="overall")
    ap.add_argument("--node", default="gaza", help="kwic: node word or phrase (whole-word match)")
    ap.add_argument("--node-regex", default=None,
                    help="kwic: regex instead of --node, e.g. 'israel(i|is)?'; word boundaries still enforced")
    ap.add_argument("--window", type=int, default=2, help="kwic: +/- N sentences")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-count", type=int, default=10,
                    help="ignore words rarer than this (noise floor)")
    ap.add_argument("--min-range", type=float, default=0.20,
                    help="word must appear in at least this fraction of videos")
    ap.add_argument("--all-speakers", action="store_true",
                    help="include guest speech (default: host only)")
    ap.add_argument("--min-host-conf", type=float, default=0.0,
                    help="skip videos whose host share is below this")
    ap.add_argument("--keep-fillers", action="store_true")
    ap.add_argument("--keep-stop", action="store_true")
    a = ap.parse_args()

    db = ROOT / a.db
    if not db.exists():
        sys.exit(f"no database at {db} -- run analysis/load_db.py first")
    corpus, dropped = load(db, host_only=not a.all_speakers,
                           min_host_conf=a.min_host_conf)
    if dropped:
        print(f"(dropped {dropped:,} non-host segments)")
    (mode_overall if a.mode == "overall" else mode_kwic)(corpus, a)
    # Every row is its own test. With hundreds of words screened, about 1 in 20
    # null words earns a '*' by chance; rank and read by LogRatio (METHODS.md 3-4).
    print("\n* p<0.05, *** p<0.001 per word, uncorrected for the number of words "
          "screened -- read LogRatio and range first.")


if __name__ == "__main__":
    main()
