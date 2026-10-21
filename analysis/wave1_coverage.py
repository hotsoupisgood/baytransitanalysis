#!/usr/bin/env python3
"""
wave1_coverage.py -- per-body coverage of the meeting corpus: regular meetings
discovered vs located vs transcribed, from state.sqlite alone ($0, no network).

WHY: a rate per meeting-hour (plan/PROPOSAL-transit.md 2.1) needs every regular
meeting of a body counted AND located. A body whose YouTube meetings are still
waiting for a caption track (status 'discovered') has an unknown numerator for
those hours, so its rate is a floor, not an estimate. This table says, per body,
which it is, and the results note restricts cross-body comparisons to the
bodies flagged complete (plan/WAVE1-RESULTS.md).

Definitions (videos.kind = 'meeting', one row per (meeting, platform)):
  meetings        distinct videos.meeting_id -- Petaluma carries the same meeting
                  twice, once on YouTube (uncaptioned) and once on Swagit (cut),
                  and counts it once, by its best status
  located         best status in classified / cut / no_episodes: a locate route
                  has read the meeting, whether or not it found an episode
  pending         best status discovered (no caption track yet), downloaded
                  (audio fetched, not read), captioned (track fetched,
                  classifier not run) or failed (locate failed) -- in every
                  case the numerator for those hours is unknown
  episodes        rows of `episodes` by status; hours are end_s - start_s
  complete        pending == 0, i.e. every meeting-group of the body has been
                  read by some locate route

  Both sets are whitelists and every meeting status must be in one of them: a
  status in neither used to be counted as *neither* located nor pending, so a
  body could silently stop being complete (2026-10-18 validation: 43 meetings
  at 'downloaded' plus 1 'failed' were invisible to the test and 6 of the 11
  "complete" bodies were stale). An unknown status now raises.

    .venv/bin/python analysis/wave1_coverage.py --out-dir output/wave1
"""
import argparse, csv, os, sqlite3, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LOCATED = ("classified", "cut", "no_episodes")
# Every meeting status a locate route has NOT finished reading. Kept exhaustive
# with LOCATED: status_rank() raises on anything in neither, so a new state in
# pipeline/state.py cannot quietly default to "not pending" and leave a body
# flagged complete (2026-10-18 validation finding 1).
PENDING = ("discovered", "downloaded", "captioned", "failed")
RANK = {"failed": 0, "discovered": 1, "downloaded": 2, "captioned": 3,
        "classified": 4, "no_episodes": 5, "cut": 6}


class UnknownStatus(RuntimeError):
    pass


def status_rank(status):
    """Rank for folding the platform rows of one meeting to its best status.
    Raises on a status that is in neither LOCATED nor PENDING."""
    if status not in RANK:
        raise UnknownStatus(
            f"meeting status {status!r} is in neither LOCATED {LOCATED} nor PENDING {PENDING}; "
            "add it to one of them (and to RANK) before trusting any coverage flag")
    return RANK[status]

# plan/T2-PLAN.md 6 (2026-09-22 probe): bodies with no located archive, not in
# plan/channels.tsv and so absent from state.sqlite. Listed here so the table
# shows the denominator the study does NOT have, not just the one it has.
UNLOCATED = ["AC Transit", "MTC / ABAG", "Caltrain (JPB)", "SFMTA", "Berkeley council",
             "Marin County BoS", "Redwood City", "Windsor", "Novato", "Sonoma County BoS"]


def state_db_path(explicit=None):
    return Path(os.path.expanduser(explicit or os.environ.get("STATE_DB_PATH")
                                   or "~/.local/share/ytpt/state.sqlite"))


def compute(state):
    st = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
    st.row_factory = sqlite3.Row
    kinds = {r["body"]: r["body_kind"] for r in st.execute("SELECT body, body_kind FROM sources")}
    all_bodies = sorted({r["body"] for r in st.execute("SELECT body FROM sources WHERE active = 1")})

    # one record per meeting_id: best status, longest duration, platforms seen
    meet = {}
    for r in st.execute("SELECT video_id, speaker, platform, status, duration_s, meeting_id "
                        "FROM videos WHERE kind = 'meeting'"):
        key = r["meeting_id"] or r["video_id"]
        m = meet.setdefault(key, {"body": r["speaker"], "status": r["status"], "dur": r["duration_s"] or 0.0,
                                  "platforms": set(), "rows": 0})
        m["platforms"].add(r["platform"]); m["rows"] += 1
        if status_rank(r["status"]) > status_rank(m["status"]):
            m["status"] = r["status"]
        m["dur"] = max(m["dur"], r["duration_s"] or 0.0)

    eps = defaultdict(lambda: defaultdict(float))
    routes = defaultdict(set)
    for r in st.execute("SELECT v.speaker AS body, e.status, e.method, e.end_s - e.start_s AS dur "
                        "FROM episodes e JOIN videos v ON v.video_id = e.video_id"):
        b = eps[r["body"]]
        b[f"n_{r['status']}"] += 1
        b[f"h_{r['status']}"] += (r["dur"] or 0.0) / 3600
        if r["status"] == "done":
            routes[r["body"]].add(r["method"] or "unknown")
    st.close()

    rows = []
    for body in all_bodies:
        ms = [m for m in meet.values() if m["body"] == body]
        n = len(ms)
        hours = sum(m["dur"] for m in ms) / 3600
        located = sum(m["status"] in LOCATED for m in ms)
        uncaptioned = sum(m["status"] == "discovered" for m in ms)
        downloaded = sum(m["status"] == "downloaded" for m in ms)
        captioned = sum(m["status"] == "captioned" for m in ms)
        locate_failed = sum(m["status"] == "failed" for m in ms)
        no_eps = sum(m["status"] == "no_episodes" for m in ms)
        dup_rows = sum(m["rows"] - 1 for m in ms)
        plats = sorted({p for m in ms for p in m["platforms"]})
        e = eps.get(body, {})
        pending = sum(m["status"] in PENDING for m in ms)
        assert located + pending == n, f"{body}: {located} located + {pending} pending != {n} meetings"
        rows.append({
            "body": body, "kind": kinds.get(body, ""), "platforms": "+".join(plats),
            "meetings": n, "meeting_hours": round(hours, 1),
            "located": located, "uncaptioned": uncaptioned, "downloaded_pending": downloaded,
            "captioned_pending": captioned, "locate_failed": locate_failed,
            "pending": pending, "located_share": round(located / n, 3) if n else "",
            "meetings_no_episodes": no_eps, "duplicate_rows": dup_rows,
            "episodes_located": int(e.get("n_located", 0)), "episodes_done": int(e.get("n_done", 0)),
            "episodes_failed": int(e.get("n_failed", 0)), "episodes_mention": int(e.get("n_mention", 0)),
            "hours_done": round(e.get("h_done", 0.0), 2), "hours_awaiting": round(e.get("h_located", 0.0), 2),
            "routes_done": "+".join(sorted(routes.get(body, ()))),
            "coverage": ("none" if n == 0 else "complete" if pending == 0 else "partial"),
        })
    for name in UNLOCATED:
        rows.append({"body": name, "kind": "", "platforms": "", "meetings": "", "meeting_hours": "",
                     "located": "", "uncaptioned": "", "downloaded_pending": "", "captioned_pending": "",
                     "locate_failed": "", "pending": "",
                     "located_share": "", "meetings_no_episodes": "", "duplicate_rows": "",
                     "episodes_located": "", "episodes_done": "", "episodes_failed": "", "episodes_mention": "",
                     "hours_done": "", "hours_awaiting": "", "routes_done": "", "coverage": "unlocated"})
    return rows


def totals(rows):
    num = [r for r in rows if r["meetings"] != ""]
    t = {"body": "TOTAL", "kind": "", "platforms": "", "coverage": "", "located_share": "", "routes_done": ""}
    for k in ("meetings", "meeting_hours", "located", "uncaptioned", "downloaded_pending",
              "captioned_pending", "locate_failed", "pending", "meetings_no_episodes", "duplicate_rows", "episodes_located", "episodes_done",
              "episodes_failed", "episodes_mention", "hours_done", "hours_awaiting"):
        v = sum(r[k] for r in num)
        t[k] = round(v, 1) if isinstance(v, float) else v
    t["located_share"] = round(t["located"] / t["meetings"], 3) if t["meetings"] else ""
    return t


def markdown(rows, tot):
    cols = [("body", "body"), ("kind", "kind"), ("platforms", "platform"), ("meetings", "regular meetings"),
            ("meeting_hours", "meeting-h"), ("located", "located"), ("uncaptioned", "uncaptioned"),
            ("downloaded_pending", "downloaded, unread"), ("captioned_pending", "captioned, unclassified"),
            ("locate_failed", "locate failed"), ("episodes_located", "episodes awaiting cut"),
            ("hours_awaiting", "h awaiting"), ("episodes_done", "episodes transcribed"), ("hours_done", "h transcribed"),
            ("routes_done", "routes"), ("coverage", "locate coverage")]
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows + [tot]:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if k == "body" and r.get("coverage") == "complete":
                v = f"**{v}**"
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "wave1"))
    a = ap.parse_args(argv)
    state = state_db_path(a.state)
    if not state.exists():
        sys.exit(f"no state database at {state}")
    rows = compute(state)
    tot = totals(rows)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows + [tot])
    (out / "coverage.md").write_text(markdown(rows, tot), encoding="utf-8")
    comp = [r["body"] for r in rows if r["coverage"] == "complete"]
    part = [r["body"] for r in rows if r["coverage"] == "partial"]
    none = [r["body"] for r in rows if r["coverage"] == "none"]
    print(f"{tot['meetings']} regular meetings / {tot['meeting_hours']} h over {len(comp) + len(part)} bodies; "
          f"located {tot['located']} ({tot['located_share']:.1%}), uncaptioned {tot['uncaptioned']}, "
          f"downloaded-unread {tot['downloaded_pending']}, captioned-unclassified {tot['captioned_pending']}, "
          f"locate-failed {tot['locate_failed']}; "
          f"{tot['duplicate_rows']} duplicate rows (same meeting on two platforms) folded")
    print(f"episodes: {tot['episodes_done']} transcribed / {tot['hours_done']} h, "
          f"{tot['episodes_located']} awaiting cut / {tot['hours_awaiting']} h, {tot['episodes_failed']} failed")
    print(f"complete locate coverage ({len(comp)}): {', '.join(comp)}")
    print(f"partial ({len(part)}): {', '.join(part)}")
    print(f"in channels.tsv, no regular meeting in window ({len(none)}): {', '.join(none)}")
    print(f"unlocated (plan/T2-PLAN.md 6, {len(UNLOCATED)}): {', '.join(UNLOCATED)}")
    print(f"wrote {out / 'coverage.csv'}, {out / 'coverage.md'}")


if __name__ == "__main__":
    main()
