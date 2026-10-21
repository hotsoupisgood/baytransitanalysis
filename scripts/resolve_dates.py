#!/usr/bin/env python3
"""
scripts/resolve_dates.py -- replace approximate upload dates in plan/catalog.tsv
with exact ones, one metadata request per video (no download).

WHY: the channel-tab crawl (--flat-playlist, youtubetab:approximate_date) only
knows "1 year ago" for anything older than ~11 months, so every such video
lands on the crawl's anniversary (20250919, 20240919, ...) and a whole year
collapses into one fake month. Monthly strata need the real date
(plan/SAMPLING.md, "Note on upload_date").

    .venv/bin/python scripts/resolve_dates.py --bucket 20250919          # the "1 year ago" rows
    .venv/bin/python scripts/resolve_dates.py --bucket 20250919 --limit 5 # try a few first

Rules (same as the downloader, PLAN.md / DECISIONS #1): from this Mac only,
one request at a time, a random 3-8 s pause between requests, and a hard stop
on "Sign in to confirm you're not a bot" or HTTP 429. Progress is written to
plan/catalog_dates.tsv after every video, so a stopped run resumes where it
left off; the catalog is rewritten at the end (and on stop) with a new
`date_exact` column (1 = resolved here, 0 = approximate).
"""
import argparse, csv, random, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "plan" / "catalog.tsv"
DATES = ROOT / "plan" / "catalog_dates.tsv"
YTDLP = ROOT / ".venv" / "bin" / "yt-dlp"
# "Sign in to confirm your AGE" is an age-restricted video (a per-video
# failure, recorded and skipped); only the bot check and 429 are a stop.
STOP = re.compile(r"confirm you.re not a bot|not a bot|HTTP Error 429", re.I)


def load_done():
    done = {}
    if DATES.exists():
        for r in csv.DictReader(open(DATES, encoding="utf-8"), delimiter="\t"):
            done[r["video_id"]] = r
    return done


def fetch(video_id):
    cmd = [str(YTDLP), "--skip-download", "--no-warnings", "--print",
           "%(id)s\t%(upload_date)s\t%(duration)s\t%(live_status)s\t%(was_live)s",
           f"https://www.youtube.com/watch?v={video_id}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if STOP.search(r.stderr or ""):
        raise RuntimeError("YOUTUBE_CHALLENGE: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    if r.returncode != 0:
        return None, (r.stderr.strip().splitlines() or ["?"])[-1]
    parts = r.stdout.strip().split("\t")
    if len(parts) < 5 or not re.fullmatch(r"\d{8}", parts[1]):
        return None, "no upload_date in metadata"
    return {"video_id": parts[0], "upload_date": parts[1], "duration_s": parts[2],
            "live_status": parts[3], "was_live": parts[4]}, None


def rewrite_catalog(done):
    rows = list(csv.DictReader(open(CATALOG, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    fields = list(rows[0].keys())
    if "date_exact" not in fields:
        fields.append("date_exact")
    n = 0
    for r in rows:
        d = done.get(r["video_id"])
        if d and d.get("upload_date"):
            r["upload_date"] = d["upload_date"]
            if d.get("live_status") and d["live_status"] != "NA":
                r["live_status"] = d["live_status"]
            r["date_exact"] = "1"
            n += 1
        else:
            r.setdefault("date_exact", "0")
    # Plain tab-joined lines, like the crawl writes them: no csv quoting or
    # escaping (a first version used csv.QUOTE_NONE + escapechar and sprayed
    # backslashes over 72 rows). Tabs/newlines inside a title become spaces.
    tmp = CATALOG.with_suffix(".tsv.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
        for r in rows:
            f.write("\t".join(re.sub(r"[\t\r\n]+", " ", r.get(k) or "") for k in fields) + "\n")
    tmp.replace(CATALOG)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default="20250919", help="approximate upload_date value to resolve")
    ap.add_argument("--since", default="", metavar="YYYYMMDD",
                    help="instead of one bucket: every row with date_exact != 1 and upload_date >= this. "
                         "The channel tab's 'N months ago' dates are crawl-date minus N months, so even "
                         "recent months are only accurate to +/- 2 weeks; resolve the whole window before drawing")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-sleep", type=float, default=3.0)
    ap.add_argument("--max-sleep", type=float, default=8.0)
    a = ap.parse_args(argv)

    rows = list(csv.DictReader(open(CATALOG, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    done = load_done()
    if a.since:
        todo = [r["video_id"] for r in rows
                if (r.get("date_exact") or "0") != "1" and r["upload_date"] >= a.since and r["video_id"] not in done]
    else:
        todo = [r["video_id"] for r in rows if r["upload_date"] == a.bucket and r["video_id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} videos to resolve (bucket {a.bucket}; {len(done)} already done)", flush=True)
    new_file = not DATES.exists()
    out = open(DATES, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(out, fieldnames=["video_id", "upload_date", "duration_s", "live_status", "was_live", "error"],
                       delimiter="\t", lineterminator="\n")
    if new_file:
        w.writeheader()
    ok = fail = 0
    try:
        for i, vid in enumerate(todo, 1):
            rec, err = fetch(vid)
            if rec:
                rec["error"] = ""
                done[vid] = rec
                ok += 1
            else:
                rec = {"video_id": vid, "upload_date": "", "duration_s": "", "live_status": "", "was_live": "", "error": err}
                fail += 1
            w.writerow(rec)
            out.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}  ok={ok} fail={fail}", flush=True)
            if i < len(todo):
                time.sleep(random.uniform(a.min_sleep, a.max_sleep))
    except RuntimeError as e:
        print(f"STOPPED: {e}  (ok={ok} fail={fail}); resume later by re-running", flush=True)
        n = rewrite_catalog(done)
        print(f"catalog: {n} rows now have exact dates", flush=True)
        return 3
    n = rewrite_catalog(done)
    print(f"done: ok={ok} fail={fail}; catalog: {n} rows now have exact dates", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
