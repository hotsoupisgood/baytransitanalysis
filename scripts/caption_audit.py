#!/usr/bin/env python3
"""
scripts/caption_audit.py -- is a caption track actually a transcript of the
meeting? (plan/T2-PLAN.md 3e; built 2026-09-23 after Hayward's classifier
result looked like an instrument artefact rather than a finding.)

Stage 1 reads a caption track and never asks whether the track covers the
meeting. A track that holds ten minutes of a three-hour session produces a
defensible-looking "1 density episode over 29 meetings" and nothing in the
pipeline says otherwise. This is the meter for that: for every meeting whose
row carries a caption track, how many words does the track hold against how
long the meeting ran?

  words     what the CLASSIFIER sees -- segment.read_segments, so YouTube's
            rolling duplicate cues are already collapsed. Counting the raw
            file would roughly double an 'auto' track and flatter it.
  wpm       words / meeting minutes. Unhurried public speech is 120-160 wpm
            (a council meeting has gavel bangs and silences, so the floor is
            lower than conversation). Under --min-wpm (60) the track cannot
            be a transcript of the whole meeting: flagged `truncated`.
  coverage  last cue end / meeting duration. Separates the two failures --
            a track that stops after 20 minutes (coverage 0.1) from one that
            is sparse the whole way through (coverage 1.0, low wpm). It is
            also the CHEAPER of the two: it needs no model, no network and no
            audio, and it is the only one that catches a track which is dense
            and correct right up to the point where it stops (`gr-fremont-453`,
            139 wpm over the 39 % of the meeting it covers). See MIN_COVERAGE.
  language  a wpm-only audit passes a track that is fluent and complete and
            in the WRONG LANGUAGE. Napa's Granicus track is Spanish end to
            end at 141 wpm; the density/transit keywords are English, so the
            classifier read 20 meetings and could not have matched. Flagged
            `non_english`.
  no track  a body whose meetings mostly have captions='none' is not a
            truncation problem and must not be averaged with one: the
            per-body table counts those separately (`no_track`).

Nothing here touches the network or writes to state.sqlite -- it reads the DB
and the caption files on disk.

    .venv/bin/python scripts/caption_audit.py
    .venv/bin/python scripts/caption_audit.py --bodies hayward,santaclara,napa
    .venv/bin/python scripts/caption_audit.py --csv output/analysis/caption_audit.csv
    .venv/bin/python scripts/caption_audit.py --flagged        # just the bad rows
"""
import argparse, csv, json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import state          # noqa: E402
import segment as sg  # noqa: E402

DEFAULT_CSV = ROOT / "output" / "analysis" / "caption_audit.csv"
MIN_WPM = 60.0                   # below this a track is not the whole meeting
# Coverage is BIMODAL, which is what makes it a cheap detector rather than a
# judgement call. Over the 168 tracks of the 2026-09-23 audit the median is
# 1.000 and the 10th percentile 0.993: a track either runs to the end of the
# meeting or stops dead. The threshold was 0.50 and caught 2 tracks (1.2 %);
# at 0.95 it catches 5 (3.0 %) and there is nothing in between to make a false
# positive out of. Tightened 2026-09-23 after gr-fremont-453 (T2-PLAN 3f):
# 5 h of meeting, 1,720 cues, last one at 39 %, and 139 wpm over the span it
# DOES cover -- a wpm floor only caught it because the denominator was the
# whole meeting, and a track that stopped at 80 % would have read 110 wpm and
# passed. Coverage is the sharper of the two instruments for this failure.
MIN_COVERAGE = 0.95              # last cue this far into the meeting or the track stops early
NO_TRACK = (None, "", "none")

# Language markers: function words that are common in one language and absent
# from the other. Deliberately NOT a dependency -- a Spanish council caption
# track is dense in these and an English one has none of them. The detector
# lives in pipeline/segment.py since 2026-09-23, because the SEGMENT stage now
# picks its keyword files with it (classify.keyword_files_for); this audit and
# the classifier must never disagree about what language a track is in.
EN_MARKERS, ES_MARKERS, MARKER_MARGIN = sg.EN_MARKERS, sg.ES_MARKERS, sg.MARKER_MARGIN
guess_language = sg.guess_language


def words_per_minute(words, duration_s):
    """None when the meeting length is unknown -- an unknown denominator must
    not become a verdict."""
    if not duration_s or duration_s <= 0:
        return None
    return float(words) / (float(duration_s) / 60.0)


def coverage(track_end_s, duration_s):
    """How far into the meeting the last cue lands, capped at 1.0 (a caption
    clock can overrun the listing's duration by a few seconds)."""
    if not duration_s or duration_s <= 0:
        return None
    return min(1.0, float(track_end_s) / float(duration_s))


def audit_track(segments, duration_s):
    """The measurements for one caption track."""
    words = sum(len(str(s.get("text", "")).split()) for s in segments)
    end = max((float(s.get("end", 0.0)) for s in segments), default=0.0)
    return {"cues": len(segments), "words": words, "track_end_s": round(end, 1),
            "wpm": words_per_minute(words, duration_s), "coverage": coverage(end, duration_s),
            "language": sg.segments_language(segments)}


def flags(rec, min_wpm=MIN_WPM, min_coverage=MIN_COVERAGE):
    """The problems with one track, worst first. Empty list = it reads as a
    transcript of the meeting."""
    out = []
    if rec["cues"] == 0:
        return ["empty"]
    if rec["wpm"] is not None and rec["wpm"] < min_wpm:
        out.append("truncated")
    if rec["coverage"] is not None and rec["coverage"] < min_coverage:
        out.append("stops_early")
    if rec["language"] == "es":
        out.append("non_english")
    if rec["wpm"] is None:
        out.append("no_duration")
    return out


def body_verdict(recs, no_track, min_wpm=MIN_WPM):
    """One line per body. `recs` are the audited tracks; `no_track` is how many
    of the body's meetings carry no caption track at all -- reported, never
    averaged in, because it is a different failure with a different remedy."""
    if not recs:
        return "NO CAPTION TRACK on any meeting" if no_track else "no meetings"
    n_es = sum(1 for r in recs if r["language"] == "es")
    if n_es >= max(1, len(recs) * 0.5):
        return "WRONG LANGUAGE (Spanish track, English keywords) -- classifier result is void"
    bad = [r for r in recs if flags(r, min_wpm)]
    share = len(bad) / len(recs)
    if share >= 0.5:
        return "TRUNCATED on most meetings -- re-locate by ASR"
    if bad:
        verdict = f"healthy overall, {len(bad)} of {len(recs)} track(s) flagged"
    else:
        verdict = "healthy"
    if no_track >= max(1, len(recs)):
        verdict += f"; but {no_track} meeting(s) have NO track"
    return verdict


def select(con, bodies=None):
    where = "kind = 'meeting'"
    params = []
    if bodies:
        where += f" AND speaker IN ({','.join('?' * len(bodies))})"
        params += bodies
    return [dict(r) for r in con.execute(
        f"SELECT video_id, speaker, platform, status, captions, captions_path, duration_s, meeting_date, raw_meta "
        f"FROM videos WHERE {where} ORDER BY speaker, meeting_date DESC, video_id", params)]


def audit(con, bodies=None, min_wpm=MIN_WPM):
    """([per-meeting record], {body: {...}}). A row whose captions_path is
    missing from disk is recorded as `missing_file`, not skipped: a track the
    DB believes in and the disk does not is exactly the kind of thing this is
    supposed to surface."""
    rows, per_body = [], {}
    for v in select(con, bodies):
        body = v["speaker"]
        b = per_body.setdefault(body, {"body": body, "meetings": 0, "no_track": 0, "tracks": []})
        b["meetings"] += 1
        rec = {"video_id": v["video_id"], "body": body, "platform": v["platform"], "status": v["status"],
               "meeting_date": v["meeting_date"], "captions": v["captions"],
               "duration_s": v["duration_s"], "duration_h": round((v["duration_s"] or 0) / 3600.0, 2)}
        if (v["captions"] or "none") in NO_TRACK or not v["captions_path"]:
            b["no_track"] += 1
            rows.append({**rec, "cues": 0, "words": 0, "track_end_s": 0.0, "wpm": None,
                         "coverage": None, "language": "unknown", "flags": "no_track"})
            continue
        p = Path(v["captions_path"])
        if not p.exists():
            rows.append({**rec, "cues": 0, "words": 0, "track_end_s": 0.0, "wpm": None,
                         "coverage": None, "language": "unknown", "flags": "missing_file"})
            continue
        m = audit_track(sg.read_segments(p), v["duration_s"])
        f = flags(m, min_wpm)
        rows.append({**rec, **m, "flags": "|".join(f)})
        b["tracks"].append(m)
    for b in per_body.values():
        t = b["tracks"]
        wpms = [r["wpm"] for r in t if r["wpm"] is not None]
        b["n_tracks"] = len(t)
        b["median_wpm"] = round(statistics.median(wpms), 1) if wpms else None
        b["flagged"] = sum(1 for r in t if flags(r, min_wpm))
        b["pct_flagged"] = round(100.0 * b["flagged"] / len(t), 1) if t else None
        b["track_h"] = round(sum(r["words"] for r in t) / 9000.0, 1)   # at 150 wpm, for scale only
        b["verdict"] = body_verdict(t, b["no_track"], min_wpm)
    return rows, per_body


def write_csv(rows, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cols = ["video_id", "body", "platform", "status", "meeting_date", "captions", "duration_s", "duration_h",
            "cues", "words", "track_end_s", "wpm", "coverage", "language", "flags"]
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "wpm": "" if r["wpm"] is None else round(r["wpm"], 1),
                        "coverage": "" if r["coverage"] is None else round(r["coverage"], 3)})
    return dest


def print_table(per_body, min_wpm):
    print(f"{'body':<14}{'mtgs':>5}{'track':>6}{'none':>6}{'med wpm':>9}{'flagged':>12}   verdict")
    for b in sorted(per_body.values(), key=lambda x: x["body"]):
        med = "-" if b["median_wpm"] is None else f"{b['median_wpm']:.0f}"
        pct = "-" if b["pct_flagged"] is None else f"{b['flagged']}/{b['n_tracks']} {b['pct_flagged']:.0f}%"
        print(f"{b['body']:<14}{b['meetings']:>5}{b['n_tracks']:>6}{b['no_track']:>6}{med:>9}{pct:>12}   {b['verdict']}")
    print(f"\nnormal speech 120-160 wpm; flagged below --min-wpm {min_wpm:.0f}, "
          f"coverage < {MIN_COVERAGE:.0%}, or a non-English track")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", default=None, help="comma-separated; default every body")
    ap.add_argument("--min-wpm", type=float, default=MIN_WPM)
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="'' to skip writing")
    ap.add_argument("--flagged", action="store_true", help="also print every flagged meeting")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)

    con = state.connect(a.state, create=False)
    bodies = [b.strip() for b in a.bodies.split(",") if b.strip()] if a.bodies else None
    rows, per_body = audit(con, bodies, a.min_wpm)
    con.close()

    if a.json:
        print(json.dumps({"bodies": [{k: v for k, v in b.items() if k != "tracks"} for b in per_body.values()],
                          "meetings": rows}, indent=1, default=str))
        return 0
    print_table(per_body, a.min_wpm)
    bad = [r for r in rows if r["flags"] and r["flags"] != "no_track"]
    print(f"\n{len(rows)} meeting(s), {sum(1 for r in rows if r['flags'] != 'no_track' and r['cues'])} with a track, "
          f"{len(bad)} flagged")
    if a.flagged and bad:
        print()
        for r in sorted(bad, key=lambda x: (x["body"], x["video_id"])):
            wpm = "-" if r["wpm"] is None else f"{r['wpm']:.0f}"
            cov = "-" if r["coverage"] is None else f"{r['coverage']:.0%}"
            print(f"  {r['video_id']:<22}{r['body']:<13}{r['duration_h']:>6.2f} h{r['words']:>8} w"
                  f"{wpm:>7} wpm  cov {cov:>5}  {r['language']:<8}{r['flags']}  [{r['status']}]")
    if a.csv:
        print(f"\n-> {write_csv(rows, a.csv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
