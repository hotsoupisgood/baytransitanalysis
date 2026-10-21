#!/usr/bin/env python3
"""
scripts/swagit_recall.py -- is Swagit's transcript as good a LOCATING instrument
as the YouTube caption track it replaces? (plan/SOURCES-AUDIT.md, 2026-09-24.)

pipeline/swagit.py takes Petaluma's 15 waiting meetings off the YouTube queue by
reading Swagit's own voice-to-text transcript instead. That is only a win if the
new transcript finds the same topic spans. The only meetings where the two can be
compared are the ones that ALREADY have a YouTube caption track, so this measures
exactly those and nothing else -- three meetings (2026-07-20, 08-03, 09-14).

WHAT IS COMPARED, and why it is stage 1 and not the classifier. The locate
pipeline is two stages: `segment.candidate_windows` (keyword density over the
transcript, $0, deterministic) then a Claude pass over the candidates ($, and the
part that assigns a label). The instrument difference between two transcripts of
the same meeting is entirely in stage 1: same keyword files, same 60-s windows,
same 0.5 hits/100w threshold, different transcriber. So:

  A. CANDIDATE RECALL. Seconds of the YouTube route's candidate windows that the
     Swagit route's candidate windows also cover, over the YouTube total. This is
     the head-to-head number: what the English keyword route found on the
     captions, and how much of it survives on the Swagit text.
  B. LOCATED-EPISODE RECALL. The meeting of 2026-09-14 has 18 episodes the FULL
     YouTube route located and paid for (method='classifier', $-already-spent).
     For each, the share of its span covered by Swagit candidate windows, and how
     many clear 50 %. This is the operational question -- would the Swagit route
     have put these episodes in front of the classifier at all -- and it costs
     nothing, because the YouTube side's labels are already in the DB.
  C. TRACK FACTS. words, words per minute, and coverage (speech span / meeting
     duration), the same three numbers scripts/caption_audit.py reports, so a
     recall figure can be read against the track that produced it.

Recall is directional and reported both ways: A over the YouTube total is recall,
A over the Swagit total says whether Swagit merely flags MORE of the meeting
(cheap recall bought with precision). Neither transcript is ground truth --
they are two ASR passes over the same audio -- so this is agreement, not
accuracy, and the report says so.

THE CLOCK OFFSET, which dominates the raw figure. The two publications of the
same meeting do not start at the same instant: measured by matching UNIQUE 6-word
sequences between the two tracks (`estimate_offset` -- independent of the keyword
list, so not circular), Swagit's clock runs -37.6 s, +163.8 s and +486.5 s
against YouTube's on the three meetings, and 33,937 of 33,937 / 16,551 of 16,553
shared 6-grams agree with that shift to within 10 s. It is a constant per
meeting, not drift. So every span is compared twice: at lag 0, and with the
Swagit spans shifted onto the YouTube clock.

**The offset is not a defect in either track and does not affect the route.** An
episode located on the Swagit transcript is cut from the Swagit HLS stream, whose
timeline is the one the transcript's own `seek(t)` anchors address -- the same
clock, by construction. The offset only exists when the two publications are
compared with each other, which is what this script does and nothing else does.

ZERO YouTube requests: the .vtt files are already on disk from before the
backoff, and the Swagit pages are read from output/t2/swagit/<body>/. Zero spend:
no Claude call. The live DB is opened READ-ONLY.

    .venv/bin/python scripts/swagit_recall.py                       # the report
    .venv/bin/python scripts/swagit_recall.py --out plan/SWAGIT-RECALL.md
"""
import argparse, json, re, sqlite3, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import segment as sg  # noqa: E402
import classify  # noqa: E402
import swagit  # noqa: E402
import state  # noqa: E402

BODY = "petaluma"
P = classify.DEFAULTS                     # window / threshold / max_run / pad / gap -- the real stage 1


def ro(path=None):
    """The live state DB, READ-ONLY (AGENTS.md rule 6 / the brief). Never a
    relative 'state.sqlite' path."""
    p = Path(path) if path else state.db_path()
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def spans(windows, key=("start_s", "end_s")):
    return [(float(w[key[0]]), float(w[key[1]])) for w in windows]


def merge(sp):
    out = []
    for a, b in sorted(sp):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def total(sp):
    return sum(b - a for a, b in merge(sp))


def overlap(sp_a, sp_b):
    """Seconds of merge(sp_a) also inside merge(sp_b)."""
    a_list, b_list = merge(sp_a), merge(sp_b)
    s, j = 0.0, 0
    for a0, a1 in a_list:
        while j < len(b_list) and b_list[j][1] <= a0:
            j += 1
        k = j
        while k < len(b_list) and b_list[k][0] < a1:
            s += max(0.0, min(a1, b_list[k][1]) - max(a0, b_list[k][0]))
            k += 1
    return s


def track_facts(segs, duration_s=None):
    words = sum(len(str(s["text"]).split()) for s in segs)
    speech = max((float(s["end"]) for s in segs), default=0.0)
    span = speech - min((float(s["start"]) for s in segs), default=0.0)
    return {"cues": len(segs), "words": words, "speech_s": round(speech, 1),
            "wpm": round(words / (span / 60), 1) if span > 60 else None,
            "coverage": round(speech / duration_s, 3) if duration_s else None,
            "lang": sg.segments_language(segs)}


# --- aligning two publications of one meeting -----------------------------------

NGRAM_N = 6
OFFSET_TOL_S = 10.0
_WORD = re.compile(r"[^a-z']")


def word_times(segs):
    """[(cue_start_s, word)] -- every word of a track, timed by its cue. Good
    enough to align two tracks: a cue is <= 8 s either side."""
    out = []
    for s in segs:
        t = float(s["start"])
        for raw in str(s.get("text", "")).split():
            w = _WORD.sub("", raw.lower())
            if w:
                out.append((t, w))
    return out


def unique_ngrams(words, n=NGRAM_N):
    """{n-gram: time} for the n-grams that occur EXACTLY ONCE in the track.
    Uniqueness is what makes the match a landmark rather than a coincidence."""
    d = {}
    for i in range(len(words) - n + 1):
        d.setdefault(" ".join(w for _, w in words[i:i + n]), []).append(words[i][0])
    return {k: v[0] for k, v in d.items() if len(v) == 1}


def estimate_offset(yt_segs, sw_segs, n=NGRAM_N, tol=OFFSET_TOL_S):
    """Seconds to ADD to a YouTube time to get the Swagit time, from the unique
    n-grams the two tracks share. Returns {offset_s, n_common, inlier_share,
    p10, p90}: `inlier_share` near 1.0 means a constant shift, a low one means
    the two publications are edited differently and no single lag aligns them."""
    a, b = unique_ngrams(word_times(yt_segs), n), unique_ngrams(word_times(sw_segs), n)
    common = set(a) & set(b)
    if not common:
        return {"offset_s": 0.0, "n_common": 0, "inlier_share": None, "p10": None, "p90": None}
    diffs = sorted(b[k] - a[k] for k in common)
    med = statistics.median(diffs)
    inl = sum(1 for d in diffs if abs(d - med) <= tol)
    return {"offset_s": round(med, 1), "n_common": len(diffs), "inlier_share": round(inl / len(diffs), 3),
            "p10": round(diffs[len(diffs) // 10], 1), "p90": round(diffs[-max(1, len(diffs) // 10)], 1)}


def shift(sp, by):
    return [(a + by, b + by) for a, b in sp]


def candidates_for(segs, lang="en"):
    kw = classify.keywords_for(lang)
    return sg.candidate_windows(segs, kw, P["window"], P["threshold"], P["max_run"])


def swagit_clip_for(body, date, cache):
    """The cached Swagit player page for one meeting date, via the player JSONs
    already written, else by scanning the cache for a page whose title carries
    the date. No network."""
    for pj in sorted(swagit.subs_dir().glob("sw-*.player.json")):
        try:
            meta = json.loads(pj.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if meta.get("meeting_date") == date:
            return meta.get("clip_id"), cache / f"player_{meta.get('clip_id')}.html"
    for html in sorted(cache.glob("player_*.html")):
        head = html.read_text(encoding="utf-8", errors="replace")[:400000]
        p = swagit.parse_player(head, None)
        if p["title"] and swagit.parse_meeting_date(p["title"].split(" City Council")[0].strip()) == date:
            return html.stem.replace("player_", ""), html
    return None, None


def compare(con, body=BODY, cache=None):
    cache = cache or ROOT / "output" / "t2" / "swagit" / body
    rows = con.execute("SELECT * FROM videos WHERE kind = 'meeting' AND speaker = ? AND platform = 'youtube' "
                       "AND captions IS NOT NULL AND captions != 'none' ORDER BY meeting_date", (body,)).fetchall()
    out = []
    for r in rows:
        vtt = r["captions_path"]
        if not vtt or not Path(vtt).exists():
            continue
        clip, html = swagit_clip_for(body, r["meeting_date"], cache)
        if not html or not html.exists():
            out.append({"date": r["meeting_date"], "video_id": r["video_id"], "skip": "no cached Swagit page"})
            continue
        yt_segs = sg.read_segments(vtt)
        sw = swagit.parse_player(html.read_text(encoding="utf-8", errors="replace"), None)
        sw_segs = sw["cues"]
        yt_c, sw_c = candidates_for(yt_segs), candidates_for(sw_segs)
        yt_sp, sw_sp = spans(yt_c), spans(sw_c)
        al = estimate_offset(yt_segs, sw_segs)
        # Swagit spans moved onto the YouTube clock, so the two publications of
        # the same meeting are compared second-for-second.
        sw_al = shift(sw_sp, -al["offset_s"])
        ov, ov_al = overlap(yt_sp, sw_sp), overlap(yt_sp, sw_al)
        eps = con.execute("SELECT episode_id, core_start_s, core_end_s, tags, method FROM episodes "
                          "WHERE video_id = ? AND status = 'located' ORDER BY core_start_s", (r["video_id"],)).fetchall()
        ep_rows = []
        for e in eps:
            a, b = float(e["core_start_s"]), float(e["core_end_s"])
            ep_rows.append({"episode_id": e["episode_id"], "start_s": a, "end_s": b, "min": round((b - a) / 60, 1),
                            "tags": e["tags"], "method": e["method"],
                            "covered": round(overlap([(a, b)], sw_sp) / max(1e-9, b - a), 3),
                            "covered_aligned": round(overlap([(a, b)], sw_al) / max(1e-9, b - a), 3)})
        out.append({
            "date": r["meeting_date"], "video_id": r["video_id"], "clip_id": clip,
            "duration_s": r["duration_s"], "status": r["status"], "align": al,
            "yt": track_facts(yt_segs, r["duration_s"]), "sw": track_facts(sw_segs, r["duration_s"]),
            "yt_windows": len(yt_c), "sw_windows": len(sw_c),
            "yt_cand_s": round(total(yt_sp), 1), "sw_cand_s": round(total(sw_sp), 1),
            "overlap_s": round(ov, 1), "overlap_aligned_s": round(ov_al, 1),
            "recall": round(ov / total(yt_sp), 3) if total(yt_sp) else None,
            "recall_aligned": round(ov_al / total(yt_sp), 3) if total(yt_sp) else None,
            "reverse": round(ov / total(sw_sp), 3) if total(sw_sp) else None,
            "reverse_aligned": round(ov_al / total(sw_sp), 3) if total(sw_sp) else None,
            "episodes": ep_rows,
        })
    return out


def render(res, body=BODY):
    L = []
    w = L.append
    w(f"# Swagit transcript vs YouTube captions as a locating instrument -- {body}\n")
    w("Stage 1 only (`segment.candidate_windows`, English keyword pair, 60-s windows, "
      f"threshold {P['threshold']} hits/100w, runs cut at {P['max_run']:.0f} s): the same matcher over two "
      "transcripts of the same audio. Neither track is ground truth, so these are AGREEMENT figures, not accuracy.\n")
    w("## Tracks\n")
    w("| meeting | h | YouTube cues / words / wpm / cov | Swagit cues / words / wpm / cov |")
    w("|---|---|---|---|")
    for r in res:
        if r.get("skip"):
            w(f"| {r['date']} | | — | {r['skip']} |")
            continue
        y, s = r["yt"], r["sw"]
        w(f"| {r['date']} (`{r['video_id']}` / clip {r['clip_id']}) | {r['duration_s'] / 3600:.2f} "
          f"| {y['cues']} / {y['words']:,} / {y['wpm']} / {y['coverage']} "
          f"| {s['cues']} / {s['words']:,} / {s['wpm']} / {s['coverage']} |")
    ok = [r for r in res if not r.get("skip")]
    w("\n## Clock offset between the two publications\n")
    w("Add `offset` to a YouTube time to get the Swagit time. Estimated from the unique shared 6-word sequences, "
      "not from the keyword windows.\n")
    w("| meeting | offset s | shared unique 6-grams | within 10 s of the median | p10..p90 |")
    w("|---|---|---|---|---|")
    for r in ok:
        a = r["align"]
        w(f"| {r['date']} | **{a['offset_s']:+.1f}** | {a['n_common']:,} | {a['inlier_share']} "
          f"| {a['p10']:+.1f}..{a['p90']:+.1f} |")
    w("\n## A. Candidate-window recall\n")
    w("| meeting | YouTube windows (s) | Swagit windows (s) | overlap s | recall (lag 0) | **recall (aligned)** | reverse (aligned) |")
    w("|---|---|---|---|---|---|---|")
    for r in ok:
        w(f"| {r['date']} | {r['yt_windows']} ({r['yt_cand_s']:.0f}) | {r['sw_windows']} ({r['sw_cand_s']:.0f}) "
          f"| {r['overlap_aligned_s']:.0f} | {r['recall']:.3f} | **{r['recall_aligned']:.3f}** "
          f"| {r['reverse_aligned']:.3f} |")
    ty = sum(r["yt_cand_s"] for r in ok)
    ts = sum(r["sw_cand_s"] for r in ok)
    to = sum(r["overlap_s"] for r in ok)
    ta = sum(r["overlap_aligned_s"] for r in ok)
    if ty and ts:
        w(f"| **all {len(ok)}** | ({ty:.0f}) | ({ts:.0f}) | {ta:.0f} | {to / ty:.3f} | **{ta / ty:.3f}** "
          f"| {ta / ts:.3f} |")
    eps = [e for r in ok for e in r.get("episodes", [])]
    if eps:
        w("\n## B. Located-episode recall (episodes the paid YouTube route already found)\n")
        mins = sum(e["min"] for e in eps)
        hit = [e for e in eps if e["covered_aligned"] >= 0.5]
        any_ = [e for e in eps if e["covered_aligned"] > 0]
        w(f"{len(eps)} episodes, {mins:.0f} core min, all `method='classifier'` off the YouTube track. On the aligned "
          f"clock, Swagit candidate windows cover **{len(hit)}/{len(eps)}** of them over at least half their span and "
          f"{len(any_)}/{len(eps)} at all; span-weighted coverage "
          f"**{sum(e['covered_aligned'] * e['min'] for e in eps) / mins:.3f}**.\n")
        w("| episode | span (YouTube clock) | min | tags | Swagit coverage (aligned) | (lag 0) |")
        w("|---|---|---|---|---|---|")
        for e in sorted(eps, key=lambda e: e["covered_aligned"]):
            w(f"| `{e['episode_id']}` | {sg.hms(e['start_s'])}-{sg.hms(e['end_s'])} | {e['min']} | {e['tags']} "
              f"| {e['covered_aligned']:.2f} | {e['covered']:.2f} |")
    return "\n".join(x for x in L if x is not None) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", default=BODY)
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", default=None, help="write the markdown report here as well as to stdout")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    con = ro(a.state)
    try:
        res = compare(con, a.body)
    finally:
        con.close()
    md = render(res, a.body)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
