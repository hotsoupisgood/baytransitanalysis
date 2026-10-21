#!/usr/bin/env python3
"""
pipeline/segment.py -- find the part of a long meeting that is about one topic
(sources_bayarea.md "Next steps" 3; designed 2026-09-19).

WHY: a council meeting is 2-6 h and the rail item is 20-40 min of it. Transcribing
the whole thing costs 5-10x the GPU minutes and buries the item in a transcript
nobody will read. We want [start, end] for the topic BEFORE audio is downloaded
or transcribed, so the downloader can fetch just that section.

Two independent cues, cross-checked against each other:

  1. structural -- YouTube chapters, or "h:mm:ss title" lines in the description,
     when the uploader bothered (`--chapters <info.json>`). Solana Beach does not.
  2. keyword density -- the free YouTube auto captions (or an existing transcript)
     cut into fixed windows; hits per 100 words of a small keyword list, smoothed
     over 3 windows; windows above a threshold merged into "episodes", padded,
     and short ones dropped. Auto captions are rough (Salana Beach, Lossan) but
     a density measure survives that; a single exact match would not.

The keyword file is data (plan/keywords_coaster.txt): one term per line, '#'
comments, whole-word, case-insensitive, optional plural. Caption text is
untrusted input and is only ever counted, never executed or shown to an LLM here.

2026-09-20 (PROPOSAL-transit.md 4, T0 test 2): several keyword files may be given
and each may tag its terms with a family ('@family density' line) and mark a
term conditional ('? parking'): a conditional term counts in a window only when
that window also has an unconditional hit from the same family. `--candidates`
writes every 60-s window above the threshold with its per-family rate, for the
stage-2 classifier (scripts/classify_windows.py); `find_episodes` is unchanged.

    python3 pipeline/segment.py output/segtest/<id>/<id>.en.vtt plan/keywords_coaster.txt \
        [--window 60] [--threshold 1.5] [--min-episode 120] [--pad 30] \
        [--chapters output/segtest/<id>/<id>.info.json] [--out episodes.json]

stdout (or --out): [{start_s, end_s, duration_s, hits_per_100w, top_keywords,
first_words, chapters}]; stderr: a human-readable summary. Exit 0 even with zero
episodes -- "no transit item in this meeting" is a correct answer.
"""
import argparse, html, json, re, sys
from collections import Counter
from pathlib import Path

_TAG = re.compile(r"<[^>]+>")
# WebVTT lets the hours field be omitted (MM:SS.mmm); YouTube always writes
# it, a Granicus captioner may not. A track whose cues did not match here
# read as ZERO segments and the meeting went to 'no_episodes' with no error.
_CUE = re.compile(r"(?:(\d+):)?(\d\d):(\d\d)\.(\d{3})\s+-->\s+(?:(\d+):)?(\d\d):(\d\d)\.(\d{3})")
_DESC_TS = re.compile(r"(?m)^\s*(?:(\d{1,2}):)?(\d{1,2}):(\d\d)\s*[-–—:]?\s*(\S.*?)\s*$")


def vtt_time(h, m, s, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def vtt_to_segments(path):
    """YouTube auto-sub VTT -> [{start, end, text}]. YouTube emits rolling cues:
    every line appears twice (as line 2 of one cue, then line 1 of the next) plus
    per-word <c> timing tags. Keep a line the first time it shows up."""
    segs, last, cur = [], None, None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _CUE.match(raw)
        if m:
            g = m.groups()
            cur = (vtt_time(*g[:4]), vtt_time(*g[4:]))
            continue
        if cur is None or raw.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        line = html.unescape(_TAG.sub("", raw)).strip()         # &gt;&gt; is YouTube's speaker-change mark
        if not line or line == last:
            continue
        segs.append({"start": cur[0], "end": cur[1], "text": line})
        last = line
    return segs


def read_segments(path):
    p = Path(path)
    if p.suffix.lower() == ".vtt":
        return vtt_to_segments(p)
    doc = json.loads(p.read_text())
    return doc["segments"] if isinstance(doc, dict) else doc      # contract object or legacy bare list


# --- what language is this track? (T2 2026-09-23, plan/T2-PLAN.md 3e) ---------
#
# Stage 1 is a keyword matcher, so a fluent, complete caption track in the wrong
# language scores zero and the meeting reads as "nothing to find". Napa's 20
# Granicus tracks are a full Spanish interpretation at 141 wpm and 100 %
# coverage: 104 audio-h that the English families could not have matched.
# The detector is function-word rates -- deliberately not a dependency, and
# good enough for the only question asked of it (which keyword file to load).
# scripts/caption_audit.py reports it per meeting; classify.py selects the
# keyword files with it.
EN_MARKERS = {"the", "and", "of", "to", "is", "that", "this", "we", "for", "with",
              "have", "are", "from", "they", "be", "it", "on", "our", "you", "council"}
ES_MARKERS = {"que", "de", "la", "el", "los", "las", "del", "una", "por", "para",
              "con", "se", "su", "más", "pero", "como", "está", "esta", "señor", "gracias"}
MARKER_MARGIN = 1.5              # the winner must be this many times the loser
DEFAULT_LANG = "en"
LANG_SAMPLE_SEGMENTS = 4000


def guess_language(text, margin=MARKER_MARGIN):
    """'en' | 'es' | 'unknown' from function-word rates. 'unknown' when the
    text is too short or the two rates are too close to call -- an audit that
    guesses confidently on 30 words would condemn a track on noise."""
    toks = [t.strip(".,;:!?¿¡\"'()[]").lower() for t in str(text).split()]
    toks = [t for t in toks if t]
    if len(toks) < 50:
        return "unknown"
    en = sum(1 for t in toks if t in EN_MARKERS)
    es = sum(1 for t in toks if t in ES_MARKERS)
    if en == 0 and es == 0:
        return "unknown"
    if en >= max(1, es * margin):
        return "en"
    if es >= max(1, en * margin):
        return "es"
    return "unknown"


def segments_language(segments, max_segments=LANG_SAMPLE_SEGMENTS):
    """guess_language over a track's cues (the first `max_segments` of them --
    an interpretation does not change language halfway through a meeting)."""
    return guess_language(" ".join(str(s.get("text", "")) for s in segments[:max_segments]))


class KeywordList(list):
    """[(term, compiled regex)] plus .family[term] and .conditional (set of terms).
    A plain list of pairs still works everywhere; the extras default to one
    unnamed family and no conditional terms. `.version` (2026-10-18) names the
    list: 'kw-' + a hash of (term, family, conditional) over every file loaded,
    in order -- comments and blank lines do not change it, one term does.
    classify.py records it on every classifier_calls and episodes row so a
    located span can be traced to the list that located it (RECALL-CHECK.md)."""
    def __init__(self, items=(), family=None, conditional=None, version=None):
        super().__init__(items)
        self.family = dict(family or {})
        self.conditional = set(conditional or ())
        self.version = version
        self.aliases_version = None                 # 'al-<hash>' once apply_aliases() ran

    def families(self):
        return sorted(set(self.family.get(t, "all") for t, _ in self))


def keywords_version(keywords):
    """'kw-<10 hex>' over the parsed terms of a KeywordList (see KeywordList)."""
    import hashlib
    fam, cond = getattr(keywords, "family", {}), getattr(keywords, "conditional", set())
    body = "\n".join(f"{t}\t{fam.get(t, '')}\t{int(t in cond)}" for t, _ in keywords)
    return "kw-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:10]


def term_body(term):
    """The regex body for one term: words joined across whitespace or a hyphen."""
    return r"[\s\-]+".join(re.escape(w) for w in term.split())


def term_regex(*bodies):
    """Whole-word, case-insensitive, optional 's / s, over one or more bodies."""
    return re.compile(rf"(?<!\w)(?:{'|'.join(bodies)})(?:'?s)?(?!\w)", re.I)


def load_keywords(*paths, aliases=None):
    """KeywordList. Whole-word like word_freq.node_pattern, plus an optional 's / s
    so 'bluff' also counts 'bluffs'. '@family x' tags the terms that follow;
    a leading '?' marks a conditional term (see hits()).

    `aliases` (2026-10-18, plan/RECALL-CHECK.md "YouTube auto-caption arm"):
    a path to an alias file, applied with apply_aliases() -- the list's
    .version then also names the alias list ('kw-<hash>/al-<hash>')."""
    out = KeywordList()
    for path in paths:
        fam = None
        for line in Path(path).read_text().splitlines():
            term = line.split("#")[0].strip().lower()
            if not term:
                continue
            if term.startswith("@family"):
                fam = term.split(None, 1)[1].strip() if " " in term else None
                continue
            cond = term.startswith("?")
            term = term.lstrip("?").strip()
            out.append((term, term_regex(term_body(term))))
            if fam:
                out.family[term] = fam
            if cond:
                out.conditional.add(term)
    out.version = keywords_version(out)
    if aliases:
        apply_aliases(out, aliases)
    return out


# --- caption homophones / misspellings (2026-10-18) ------------------------------
#
# YouTube's auto-captions get exactly the words the screen keys on wrong, and
# consistently: on the BART track 'fare' -> 'fair' (13 -> 0), 'fare gates' ->
# 'fairgates', 'ridership' -> 'wrership' / 'writership' (16 -> 6); on Los Altos
# 'YIMBY' -> 'yimi' (17 -> 0) and 'rezoning' -> 'reszoning' / 'resoning' (11 ->
# 0). Over the 107 auto-caption tracks on disk 'rezoning' occurs in 2 60-s
# windows and 'reszoning' in 23. An alias file maps each misspelling to the
# term it stands for:
#
#     alias => term        counted AS the term (its regex gains the alias)
#     ? alias => term      CONDITIONAL alias: its own entry, named after the
#                          alias, in the term's family, counted only beside an
#                          unconditional hit of that family ('fair' is a
#                          common word; 'fairgate' is not)
#
# The layer applies ONLY to the screen's matching (hits / candidate_windows):
# the stored caption text, the window text the classifier reads and the
# transcripts are never rewritten. classify.keywords_for() loads it for
# captions='auto' tracks only -- a Granicus captioner, Swagit or the Mac ASR
# make different mistakes, and the Mac-ASR arm's misses were vocabulary, not
# spelling (plan/RECALL-CHECK.md).

def load_aliases(path):
    """[(alias, term, conditional)] from an alias file; '#' comments."""
    out = []
    for line in Path(path).read_text().splitlines():
        s = line.split("#")[0].strip().lower()
        if not s or "=>" not in s:
            continue
        cond = s.startswith("?")
        alias, term = (x.strip() for x in s.lstrip("?").split("=>", 1))
        if alias and term:
            out.append((alias, term, cond))
    return out


def aliases_version(aliases):
    """'al-<10 hex>' over the parsed (alias, term, conditional) triples."""
    import hashlib
    body = "\n".join(f"{a}\t{t}\t{int(c)}" for a, t, c in aliases)
    return "al-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:10]


def apply_aliases(keywords, path):
    """Fold an alias file into `keywords` IN PLACE (see above). An alias whose
    term is not in the list is skipped and reported on stderr -- a typo in the
    alias file must not silently invent a term. Returns the alias version."""
    aliases = load_aliases(path)
    terms = {t: i for i, (t, _) in enumerate(keywords)}
    extra = {}
    for alias, term, cond in aliases:
        if term not in terms:
            print(f"alias file {path}: {alias!r} => {term!r}: no such term; skipped", file=sys.stderr)
            continue
        if cond:
            if alias in terms:
                continue                                    # already a term in its own right
            keywords.append((alias, term_regex(term_body(alias))))
            terms[alias] = len(keywords) - 1
            if term in keywords.family:
                keywords.family[alias] = keywords.family[term]
            keywords.conditional.add(alias)
        else:
            extra.setdefault(term, []).append(term_body(alias))
    for term, bodies in extra.items():
        keywords[terms[term]] = (term, term_regex(term_body(term), *bodies))
    av = aliases_version(aliases)
    keywords.aliases_version = av
    keywords.version = f"{keywords.version}/{av}"
    return av


def hits(text, keywords):
    """Counter{term: n} for one piece of text. Conditional terms are dropped
    unless the same text has an unconditional hit from their family."""
    c = Counter({t: len(rx.findall(text)) for t, rx in keywords if rx.search(text)})
    cond = getattr(keywords, "conditional", ())
    if cond:
        fam = getattr(keywords, "family", {})
        anchored = {fam.get(t, "all") for t in c if t not in cond}
        for t in [t for t in c if t in cond and fam.get(t, "all") not in anchored]:
            del c[t]
    return c


def family_hits(text, keywords):
    """{family: hits} for one piece of text, after the conditional rule."""
    fam = getattr(keywords, "family", {})
    out = Counter()
    for t, n in hits(text, keywords).items():
        out[fam.get(t, "all")] += n
    return out


def window_texts(segments, window):
    """[(words, text)] per window, indexed by floor(start / window). Cue texts
    are joined so a phrase split across two cues still matches."""
    if not segments:
        return []
    n = int(max(s["start"] for s in segments) // window) + 1
    parts = [[] for _ in range(n)]
    for s in segments:
        parts[int(s["start"] // window)].append(s["text"])
    texts = [" ".join(p) for p in parts]
    return [(len(t.split()), t) for t in texts]


def window_density(segments, keywords, window):
    """[(words, hits)] per window, indexed by floor(start / window)."""
    return [(w, sum(hits(t, keywords).values())) for w, t in window_texts(segments, window)]


def candidate_windows(segments, keywords, window=60, threshold=0.5, max_run=120):
    """Stage 1 of PROPOSAL-transit.md 4: every maximal run of consecutive windows
    whose smoothed union density >= threshold, cut into pieces of at most
    max_run seconds so a 30-min agenda item becomes several classifier calls.
    [{start_s, end_s, words, hits_per_100w, family_rates{fam: hits/100w},
    top_keywords}] in time order. No padding here -- the classifier pads."""
    wt = window_texts(segments, window)
    dens = [density_per_100w(w, sum(hits(t, keywords).values())) for w, t in wt]
    marked = [d >= threshold for d in smooth3(dens)]
    runs, start = [], None
    for i, m in enumerate(marked + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i)); start = None
    per = max(1, int(max_run // window))
    out = []
    for a, b in runs:
        for i in range(a, b, per):
            j = min(b, i + per)
            text = " ".join(t for _, t in wt[i:j])
            words = len(text.split())
            c = hits(text, keywords)
            fh = family_hits(text, keywords)
            out.append({"start_s": float(i * window), "end_s": float(j * window), "words": words,
                        "hits_per_100w": round(density_per_100w(words, sum(c.values())), 2),
                        "family_rates": {f: round(density_per_100w(words, n), 2) for f, n in sorted(fh.items())},
                        "top_keywords": [t for t, _ in c.most_common(5)]})
    return out


def density_per_100w(words, hit, min_words=10):
    return 100.0 * hit / words if words >= min_words else 0.0


def smooth3(xs):
    return [sum(xs[max(0, i - 1):i + 2]) / len(xs[max(0, i - 1):i + 2]) for i in range(len(xs))]


def find_episodes(segments, keywords, window=60, threshold=1.5, min_episode=120, pad=30, gap=90):
    """The method described at the top. Returns [{start_s, end_s, ...}] in time order."""
    dens = window_density(segments, keywords, window)
    marked = [d >= threshold for d in smooth3([density_per_100w(w, h) for w, h in dens])]
    runs, start = [], None
    for i, m in enumerate(marked + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append([start * window, i * window]); start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    end_of_talk = max(s["end"] for s in segments) if segments else 0
    episodes = []
    for a, b in merged:
        a, b = max(0.0, a - pad), min(end_of_talk, b + pad)
        if b - a < min_episode:
            continue
        inside = [s for s in segments if a <= s["start"] < b]
        text = " ".join(s["text"] for s in inside)
        words = len(text.split())
        c = hits(text, keywords)
        episodes.append({"start_s": round(a, 1), "end_s": round(b, 1), "duration_s": round(b - a, 1),
                         "hits_per_100w": round(density_per_100w(words, sum(c.values())), 2),
                         "top_keywords": [t for t, _ in c.most_common(5)],
                         "first_words": " ".join(text.split()[:12])})
    return episodes


def structural_chapters(info_path):
    """[(start, end, title)] from info-json chapters, else from description timestamps."""
    info = json.loads(Path(info_path).read_text())
    ch = [(c["start_time"], c["end_time"], c.get("title", "")) for c in info.get("chapters") or []]
    if ch:
        return ch
    ts = [(vtt_time(h or 0, m, s, 0), title) for h, m, s, title in _DESC_TS.findall(info.get("description") or "")]
    ts.sort()
    dur = float(info.get("duration") or (ts[-1][0] + 1 if ts else 0))
    return [(t, ts[i + 1][0] if i + 1 < len(ts) else dur, title) for i, (t, title) in enumerate(ts)]


def overlapping(episode, chapters):
    return [t for a, b, t in chapters if a < episode["end_s"] and b > episode["start_s"]]


def hms(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("transcript", help=".vtt (YouTube captions) or contract/legacy transcript JSON")
    ap.add_argument("keywords", nargs="+", help="one term per line, # comments; several files are unioned")
    ap.add_argument("--window", type=float, default=60, help="seconds per window (60)")
    ap.add_argument("--threshold", type=float, default=1.5, help="smoothed hits per 100 words (1.5)")
    ap.add_argument("--min-episode", type=float, default=120, help="drop episodes shorter than this, s (120)")
    ap.add_argument("--pad", type=float, default=30, help="extend each episode by this much on both sides, s (30)")
    ap.add_argument("--gap", type=float, default=90, help="merge marked runs closer than this, s (90)")
    ap.add_argument("--chapters", help="yt-dlp info.json; cross-check episodes against chapters / description timestamps")
    ap.add_argument("--out", help="write the episode JSON here instead of stdout")
    ap.add_argument("--candidates", help="also write stage-1 candidate windows (unpadded, <= --max-run s each) here")
    ap.add_argument("--max-run", type=float, default=120, help="split a marked run into candidates of at most this many s "
                    "(120 since 2026-10-18, plan/CODEBOOK-V2-PILOT.md 7.1; the T0 instrument read 240)")
    a = ap.parse_args(argv)

    segments = read_segments(a.transcript)
    keywords = load_keywords(*a.keywords)
    if a.candidates:
        cands = candidate_windows(segments, keywords, a.window, a.threshold, a.max_run)
        Path(a.candidates).write_text(json.dumps(cands, indent=1))
        print(f"{len(cands)} candidate window(s) -> {a.candidates}", file=sys.stderr)
    eps = find_episodes(segments, keywords, a.window, a.threshold, a.min_episode, a.pad, a.gap)
    chapters = structural_chapters(a.chapters) if a.chapters else []
    for e in eps:
        e["chapters"] = overlapping(e, chapters)

    err = sys.stderr
    total = max((s["end"] for s in segments), default=0)
    print(f"{a.transcript}: {len(segments)} segments, {hms(total)} of speech, {len(keywords)} keywords, "
          f"window {a.window:.0f} s, threshold {a.threshold} /100w", file=err)
    if a.chapters:
        print(f"structural cues: {len(chapters)} chapter(s)" + ("" if chapters else " -- none in info.json (no chapters, no description timestamps)"), file=err)
        for s, e, t in chapters:
            print(f"    {hms(s)}-{hms(e)}  {t}", file=err)
    print(f"{len(eps)} episode(s)" + ("" if eps else " -- no window reached the threshold"), file=err)
    for i, e in enumerate(eps, 1):
        print(f"  {i}. {hms(e['start_s'])}-{hms(e['end_s'])} ({e['duration_s'] / 60:.1f} min) {e['hits_per_100w']} hits/100w "
              f"{e['top_keywords']}" + (f"  chapters={e['chapters']}" if chapters else "") + f"\n     \"{e['first_words']}...\"", file=err)
    out = json.dumps(eps, indent=1)
    if a.out:
        Path(a.out).write_text(out)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
