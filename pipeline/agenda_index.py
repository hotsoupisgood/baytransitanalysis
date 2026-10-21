#!/usr/bin/env python3
"""
pipeline/agenda_index.py -- locate episodes from Granicus agenda index points
(PROPOSAL-transit.md 2.2 "Granicus agenda index / Legistar item timestamps",
$0; built 2026-09-22 after the T2 probe, plan/T2-PLAN.md 3b).

The caption route cannot run on a body that serves no usable English caption
track. Most of those archives still publish the agenda as index points on the
player page:

    <div class="index-point" time="2615">5.A) Marin County Special Presentation
    on El Nino: King Tide Preparation</div>

That is an agenda item with a jump time, so an item's span is
[its time, the next item's time) and the whole meeting is partitioned. This
module matches the density / transit keyword families (plan/keywords_*.txt,
the same lists stage 1 of the classifier uses) against the ITEM TITLE and
writes the matching spans as `episodes` rows with **method = 'agenda_index'**.

What this instrument is and is not:

  * It reads the agenda, not the speech. It finds items that are ABOUT the
    topic by their title -- "6. Housing Element Annual Progress Report" -- and
    is blind to the twenty minutes of public comment on housing inside an item
    called "Consent Calendar". Recall is the agenda's recall; precision is
    high because an item title is written to describe the item.
  * Its n is therefore NOT comparable with the classifier's n. Report per
    `method` (SCHEMA.md `episodes`), never pooled -- the same rule that keeps
    keyword and classifier counts apart.
  * There is no label: `label` stays NULL and `classifier_model` /
    `prompt_version` stay NULL on these rows. `agenda_item` carries the title
    verbatim, `score` the title's keyword hits per 100 words.

Episode ids are `<video_id>_a<n>` (the classifier's are `_e<n>`), so a meeting
can carry both instruments without either overwriting the other.

Index-point titles are untrusted text (AGENTS.md rule 5): matched with
regexes, stored, never executed.

    python3 pipeline/agenda_index.py --plan                  # what would be written, no DB change
    python3 pipeline/agenda_index.py --bodies santaclara,hayward
    python3 pipeline/agenda_index.py --report                # rows per body already written
"""
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402
import segment as sg  # noqa: E402

ROOT = state.ROOT
KEYWORD_FILES = [ROOT / "plan" / "keywords_density.txt", ROOT / "plan" / "keywords_transit.txt"]
METHOD = "agenda_index"
MIN_ITEM_S = 30.0          # an item shorter than this is a mis-timed index point, not a window
DEFAULT_TAIL_S = 600.0     # the last item of a meeting whose duration is unknown
MAX_TAIL_S = 3600.0        # ... and never longer than this
# An index point says where an item STARTS. Its end is only inferred from the
# next point, and Granicus agendas are sparse at the end of a long meeting: the
# raw spans here ran to 4.4 h (Napa 5713), which is not an episode, it is "the
# rest of the evening". Cap a single item at an hour -- the item is still
# located and counted, its window is just bounded by what the agenda actually
# supports. Matched-item spans: median 19 min, p90 68 min, 8 of 62 over an hour.
MAX_ITEM_S = 3600.0
TRUNCATED = " [span truncated to 60 min: the next agenda index point is later than that]"
TAGS = {("density",): "density", ("transit",): "transit", ("density", "transit"): "both"}


def subs_dir():
    import os
    return Path(os.path.expanduser(state.setting("SUBS_DIR", str(ROOT / "output" / "subs"))))


def index_points_for(video_id, raw_meta=None):
    """[(time_s, title)] from output/subs/<id>.player.json, sorted, de-duplicated
    on (time, title). Returns [] when the player page was never saved."""
    pj = subs_dir() / f"{video_id}.player.json"
    if not pj.exists():
        return []
    try:
        pts = json.loads(pj.read_text(encoding="utf-8")).get("index_points") or []
    except ValueError:
        return []
    seen, out = set(), []
    for t, title in pts:
        title = " ".join(str(title).split())
        key = (float(t), title)
        if title and key not in seen:
            seen.add(key)
            out.append((float(t), title))
    out.sort(key=lambda p: p[0])
    return out


def item_windows(points, duration_s=None):
    """[(start, end, title)]: each index point up to the NEXT point's time.

    Points sharing a time (Santa Clara prints a section header and its first
    item at the same second) all get the span up to the next distinct time, so
    a matching header and a matching item do not produce two zero-length rows.
    The last item runs to the meeting duration; where the listing carries no
    duration (Sausalito, SamTrans) it gets the median of the other items,
    bounded by DEFAULT_TAIL_S / MAX_TAIL_S -- recorded, and the cutter clamps
    to whatever the audio actually is."""
    if not points:
        return []
    times = sorted({t for t, _ in points})
    nxt = {t: times[i + 1] for i, t in enumerate(times[:-1])}
    spans = [nxt[t] - t for t in times[:-1]] if len(times) > 1 else []
    if duration_s and duration_s > times[-1]:
        tail = float(duration_s) - times[-1]
    else:
        med = sorted(spans)[len(spans) // 2] if spans else DEFAULT_TAIL_S
        tail = min(max(med, MIN_ITEM_S), MAX_TAIL_S)
    out = []
    for t, title in points:
        end = min(float(nxt.get(t, t + tail)), t + MAX_ITEM_S)
        out.append((t, end, title))
    return out


def match_items(windows, keywords):
    """The windows whose TITLE hits a keyword family, with tags and score."""
    out = []
    for start, end, title in windows:
        fam = sg.family_hits(title, keywords)
        fam = Counter({f: n for f, n in fam.items() if n > 0})
        if not fam:
            continue
        words = max(1, len(title.split()))
        if end - start >= MAX_ITEM_S - 0.5:
            title += TRUNCATED
        out.append({"start_s": start, "end_s": end, "title": title,
                    "families": dict(fam), "hits": sum(fam.values()),
                    "tags": TAGS.get(tuple(sorted(fam)), "both"),
                    "score": round(sum(fam.values()) / words * 100, 2)})
    return out


def merge_runs(matched):
    """Consecutive matched items (one ending where the next begins) become one
    episode, exactly as adjacent about_* windows do in classify.merge_episodes.
    Titles are joined with ' | ' so the agenda_item shows what the span covers."""
    eps = []
    for m in sorted(matched, key=lambda m: (m["start_s"], m["end_s"])):
        cur = eps[-1] if eps else None
        if cur and m["start_s"] <= cur["end_s"] + 1.0:
            cur["end_s"] = max(cur["end_s"], m["end_s"])
            cur["title"] += " | " + m["title"]
            cur["hits"] += m["hits"]
            for f, n in m["families"].items():
                cur["families"][f] = cur["families"].get(f, 0) + n
            cur["score"] = max(cur["score"], m["score"])
            cur["n_items"] += 1
        else:
            eps.append({**m, "families": dict(m["families"]), "n_items": 1})
    for e in eps:
        e["tags"] = TAGS.get(tuple(sorted(e["families"])), "both")
        e["duration_s"] = round(e["end_s"] - e["start_s"], 1)
    return [e for e in eps if e["duration_s"] >= MIN_ITEM_S]


def write_episodes(con, row, eps, dry_run=False):
    """episodes rows, method='agenda_index', ids <video_id>_a<n>. A row that is
    already past 'located' (cut, queued, done) is left alone, like
    classify.write_episodes."""
    vid = row["video_id"]
    now = state.now_iso()
    n_rows = 0
    if dry_run:
        return len(eps)
    con.execute("BEGIN IMMEDIATE")
    for i, e in enumerate(eps, 1):
        cur = con.execute(
            "INSERT INTO episodes (episode_id, meeting_id, video_id, start_s, end_s, core_start_s, core_end_s, tags, method, "
            "score, label, classifier_model, prompt_version, agenda_item, status, attempts, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,?,'located',0,?) "
            "ON CONFLICT(episode_id) DO UPDATE SET start_s = excluded.start_s, end_s = excluded.end_s, "
            "core_start_s = excluded.core_start_s, core_end_s = excluded.core_end_s, tags = excluded.tags, "
            "score = excluded.score, agenda_item = excluded.agenda_item, updated_at = excluded.updated_at "
            "WHERE episodes.status = 'located'",
            (f"{vid}_a{i}", row["meeting_id"], vid, e["start_s"], e["end_s"], e["start_s"], e["end_s"], e["tags"],
             METHOD, e["score"], e["title"][:500], now))
        n_rows += cur.rowcount
    con.execute("COMMIT")
    return n_rows


def locate_meeting(con, row, keywords, dry_run=False):
    """(summary dict) for one Granicus meeting."""
    pts = index_points_for(row["video_id"])
    windows = item_windows(pts, row["duration_s"])
    matched = match_items(windows, keywords)
    eps = merge_runs(matched)
    summ = {"video_id": row["video_id"], "body": row["speaker"], "date": row["meeting_date"],
            "index_points": len(pts), "items": len(windows), "matched": len(matched), "episodes": len(eps),
            "episode_min": round(sum(e["duration_s"] for e in eps) / 60, 1),
            "meeting_h": round((row["duration_s"] or 0) / 3600, 2), "eps": eps}
    if not pts:
        summ["reason"] = "no index points saved for this meeting (output/subs/<id>.player.json)"
        return summ
    summ["rows_written"] = write_episodes(con, row, eps, dry_run)
    return summ


def select_meetings(con, bodies=None, statuses=("no_episodes", "captioned", "classified"), limit=0):
    where = "kind = 'meeting' AND platform = 'granicus' AND status IN (" + ",".join("?" * len(statuses)) + ")"
    params = list(statuses)
    if bodies:
        where += f" AND speaker IN ({','.join('?' * len(bodies))})"
        params += bodies
    return con.execute(f"SELECT * FROM videos WHERE {where} ORDER BY speaker, meeting_date DESC"
                       + (f" LIMIT {int(limit)}" if limit else ""), params).fetchall()


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def report(con):
    print(f"{'body':14} {'meetings':>8} {'episodes':>8} {'episode-h':>9} {'mean min':>8}")
    tot = Counter()
    for r in con.execute("SELECT v.speaker body, COUNT(DISTINCT e.video_id) m, COUNT(*) n, "
                         "COALESCE(SUM(e.end_s - e.start_s), 0) s FROM episodes e JOIN videos v USING (video_id) "
                         "WHERE e.method = ? GROUP BY v.speaker ORDER BY v.speaker", (METHOD,)):
        print(f"{r['body']:14} {r['m']:>8} {r['n']:>8} {r['s'] / 3600:>9.1f} {r['s'] / 60 / max(1, r['n']):>8.1f}")
        tot["m"] += r["m"]; tot["n"] += r["n"]; tot["s"] += r["s"]
    print(f"{'TOTAL':14} {tot['m']:>8} {tot['n']:>8} {tot['s'] / 3600:>9.1f} {tot['s'] / 60 / max(1, tot['n']):>8.1f}")
    return 0


def run(args):
    con = state.connect(args.state)
    if args.report:
        rc = report(con)
        con.close()
        return rc
    keywords = sg.load_keywords(*KEYWORD_FILES)
    bodies = [b.strip() for b in args.bodies.split(",") if b.strip()] if args.bodies else None
    rows = select_meetings(con, bodies, limit=args.max)
    log(f"{len(rows)} Granicus meeting(s) to locate from agenda index points; "
        f"{len(keywords)} keywords ({', '.join(keywords.families())})" + (" (plan only)" if args.plan else ""))
    per_body = defaultdict(lambda: Counter())
    n_no_index = 0
    for r in rows:
        s = locate_meeting(con, r, keywords, args.plan)
        b = per_body[s["body"]]
        b["meetings"] += 1
        b["index_points"] += s["index_points"]
        b["episodes"] += s["episodes"]
        b["episode_min"] += s["episode_min"]
        b["meeting_h"] += s["meeting_h"]
        if s.get("reason"):
            n_no_index += 1
            b["no_index"] += 1
            continue
        b["rows"] += s.get("rows_written", 0)
        if s["episodes"] and not args.quiet:
            log(f"  {s['video_id']} ({s['body']}, {s['date']}): {s['index_points']} index points, "
                f"{s['matched']} matching items -> {s['episodes']} episodes, {s['episode_min']} min")
            for e in s["eps"][:args.show]:
                log(f"      {sg.hms(e['start_s'])}-{sg.hms(e['end_s'])} [{e['tags']}, {e['n_items']} item(s)] {e['title'][:110]}")
        if s["episodes"] and not args.plan:
            # the meeting now has located episodes: episodes.py cuts rows in
            # 'classified' / 'downloaded', so move it out of 'no_episodes'
            if r["status"] in ("no_episodes", "captioned"):
                state.transition(con, r["video_id"], r["status"], "classified",
                                 note=f"{METHOD}: {s['episodes']} episodes from {s['index_points']} agenda index points")
    print(f"\n{'body':14} {'mtgs':>5} {'no idx':>6} {'mtg-h':>7} {'eps':>5} {'ep-h':>6} {'% of mtg':>8}")
    tot = Counter()
    for body, b in sorted(per_body.items()):
        print(f"{body:14} {b['meetings']:>5} {b['no_index']:>6} {b['meeting_h']:>7.1f} {b['episodes']:>5} "
              f"{b['episode_min'] / 60:>6.1f} {100 * b['episode_min'] / 60 / b['meeting_h'] if b['meeting_h'] else 0:>7.1f}%")
        for k in ("meetings", "no_index", "meeting_h", "episodes", "episode_min", "rows"):
            tot[k] += b[k]
    print(f"{'TOTAL':14} {tot['meetings']:>5} {tot['no_index']:>6} {tot['meeting_h']:>7.1f} {tot['episodes']:>5} "
          f"{tot['episode_min'] / 60:>6.1f} {100 * tot['episode_min'] / 60 / tot['meeting_h'] if tot['meeting_h'] else 0:>7.1f}%")
    log(f"done: {tot['episodes']} episodes ({tot['episode_min'] / 60:.1f} h) over {tot['meetings']} meetings, "
        f"{tot['rows']} rows written, {n_no_index} meeting(s) with no saved index points"
        + (" (plan only, nothing written)" if args.plan else ""))
    con.close()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", default=None)
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--plan", action="store_true", help="print what would be written, change nothing")
    ap.add_argument("--report", action="store_true", help="rows already written, per body")
    ap.add_argument("--show", type=int, default=3, help="episodes to print per meeting")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--state", default=None)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
