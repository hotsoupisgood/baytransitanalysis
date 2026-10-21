#!/usr/bin/env python3
"""
pipeline/discover.py -- load plan/catalog.tsv into state.sqlite as 'discovered'
rows, and mark which of them are in the sample (in_frame = 1).

Re-runnable: inserts are ON CONFLICT DO NOTHING (PLAN.md 4.1), so a re-crawl
adds new videos and never touches rows the workers own. The speaker and channel
are stored at discovery and never inferred later.

Who is in the frame:
  --frame plan/frame.csv   (default when it exists) the drawn sample -- THE
                           corpus definition (SAMPLING.md). Rows listed there
                           get in_frame = 1; nothing else changes.
  --pick-recent N          for a first run before the frame is drawn: mark the N
                           most recent eligible videos PER SPEAKER (5-90 min,
                           not live). This is a smoke sample, not the study;
                           when frame.csv is drawn it takes over.
  --mark ID [ID ...]       mark specific videos.

    python3 pipeline/discover.py                       # load catalog (+ frame.csv if present)
    python3 pipeline/discover.py --pick-recent 2       # first-run sample

Meetings (2026-09-21, PROPOSAL-transit.md 3a/3c, T1). `--meetings` reads the
kind=meeting rows of plan/channels.tsv (speaker, url, kind, regular-title
regex), lists each source -- a YouTube /streams tab with `yt-dlp
--flat-playlist` (metadata only, --sleep-requests 2, one channel at a time),
or a Granicus ViewPublisher listing (pipeline/granicus.py) -- keeps the
REGULAR meetings whose date (parsed from the title, not the upload date) is on
or after --since, and inserts them as kind='meeting' rows with in_frame = 0:
the launchd downloader and transcoder select in_frame = 1 only, so a meeting
row is never picked up by them; pipeline/captions.py, classify.py and
episodes.py are the meeting stages. A `sources` row is written per channel.

    python3 pipeline/discover.py --meetings --recent 3         # newest 3 regular meetings per body
    python3 pipeline/discover.py --meetings --since 2025-09-20 # the whole window
"""
import argparse, csv, json, os, re, shutil, subprocess, sys, time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402

ROOT = state.ROOT
MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
MONTHS.update({"sept": 9, "june": 6, "july": 7})
_DATE_PATTERNS = [
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),                                   # 2026-09-15
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),                               # 09/08/2026
    re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})\b"),                # SEP 15, 2026 / September 10, 2026 / Sept. 15, 2026
    re.compile(r"\b(\d{1,2})-(\d{1,2})-(\d{2})\b"),                               # 9-15-26
    re.compile(r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b"),                               # 9-8-2026 (San Mateo, T2)
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2})\b"),                               # 9/15/26 (Pleasanton, T2)
]


def month_of(upload_date):
    d = (upload_date or "").strip()
    return f"{d[:4]}-{d[4:6]}" if len(d) >= 6 and d[:6].isdigit() else "unknown"


def parse_meeting_date(title):
    """'YYYY-MM-DD' from a meeting title in any of the forms seen in T0, or None.
    Titles are untrusted text; only these four shapes are read."""
    t = title or ""
    for i, rx in enumerate(_DATE_PATTERNS):
        m = rx.search(t)
        if not m:
            continue
        try:
            if i == 0:
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif i == 1:
                mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            elif i == 2:
                mo = MONTHS.get(m.group(1).lower()) or MONTHS.get(m.group(1).lower()[:3])
                if not mo:
                    continue
                d, y = int(m.group(2)), int(m.group(3))
            elif i in (3, 5):
                mo, d, y = int(m.group(1)), int(m.group(2)), 2000 + int(m.group(3))
            else:
                mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return date(y, mo, d).isoformat()
        except ValueError:
            continue
    return None


def read_channels(path=None):
    """Every non-comment row of channels.tsv: {speaker, url, kind, regular_re}."""
    rows = []
    for line in Path(path or ROOT / "plan" / "channels.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        rows.append({"speaker": parts[0].strip(), "url": parts[1].strip() if len(parts) > 1 else "",
                     "kind": (parts[2].strip() if len(parts) > 2 and parts[2].strip() else "show"),
                     "regular_re": parts[3].strip() if len(parts) > 3 else ""})
    return rows


def ytdlp_bin():
    if os.environ.get("YTDLP_BIN"):
        return os.environ["YTDLP_BIN"]
    p = ROOT / ".venv" / "bin" / "yt-dlp"
    return str(p) if p.exists() else (shutil.which("yt-dlp") or "yt-dlp")


def list_youtube(url, playlist_end=40):
    """Flat-playlist listing, metadata only (T0 test 1 pattern). Returns
    (rows, stderr): rows = [{id, upload_date, duration_s, live_status, title}]."""
    cmd = [ytdlp_bin(), "--flat-playlist", "--playlist-end", str(playlist_end), "--sleep-requests", "2",
           "--print", "%(id)s\t%(upload_date,release_date)s\t%(duration)s\t%(live_status)s\t%(title)s", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    rows = parse_listing_tsv(r.stdout)
    return rows, r.stderr, r.returncode


def parse_listing_tsv(text):
    rows = []
    for line in text.splitlines():
        # a listing saved with a literal backslash-t template (T0's list2.sh) is still readable
        parts = line.split("\t") if "\t" in line else line.split("\\t")
        if len(parts) < 5:
            continue
        vid, up, dur, live, title = parts[0], parts[1], parts[2], parts[3], "\t".join(parts[4:])
        try:
            dur_s = float(dur) if dur not in ("NA", "") else None
        except ValueError:
            dur_s = None
        rows.append({"id": vid, "upload_date": None if up in ("NA", "") else up, "duration_s": dur_s,
                     "live_status": live, "title": title})
    return rows


BLOCK_SIGNAL = re.compile(r"sign in to confirm|not a bot|HTTP Error 429|Too Many Requests|HTTP Error 403", re.I)


PART = re.compile(r"\bpart\s*(\d+)\b", re.I)     # Cupertino uploads "... (Part 1)" / "(Part 2)" per meeting


def select_regular(rows, regular_re, since=None, recent=0, max_duration_s=12 * 3600, until=None):
    """Regular meetings by title regex, live/upcoming rows dropped, one row per
    meeting date (the first listed wins: duplicate uploads and the Granicus
    '15h 00m' placeholder clips lose) -- except multi-part uploads, where each
    "Part N" of a date is kept (same meeting_id, its own video row: T2,
    2026-09-22) -- dated >= since, newest `recent` (0 = all)."""
    rx = re.compile(regular_re) if regular_re else None
    seen, out = set(), []
    for r in rows:
        title = r.get("title") or r.get("name") or ""
        if rx and not rx.search(title):
            continue
        if r.get("live_status") in ("is_live", "is_upcoming", "post_live"):
            continue
        if r.get("duration_s") and r["duration_s"] >= max_duration_s:
            continue
        d = parse_meeting_date(title) or parse_meeting_date(r.get("date") or "")
        part = PART.search(title)
        key = (d, part.group(1) if part else None)
        if not d or (since and d < since) or (until and d > until) or key in seen:
            continue
        seen.add(key)
        out.append(dict(r, meeting_date=d, title=title))
    out.sort(key=lambda r: (r["meeting_date"], r["title"]), reverse=True)
    return out[:recent] if recent else out


def upsert_source(con, body, url, kind, regular_re, platform, platform_id, body_kind):
    sid = f"{body}:{platform}:{platform_id}"
    con.execute("INSERT INTO sources (source_id, body, body_kind, platform, platform_id, url, regular_re, region, active, updated_at) "
                "VALUES (?,?,?,?,?,?,?,'bayarea',1,?) ON CONFLICT(source_id) DO UPDATE SET url = excluded.url, "
                "regular_re = excluded.regular_re, updated_at = excluded.updated_at",
                (sid, body, body_kind, platform, platform_id, url, regular_re, state.now_iso()))
    return sid


def insert_meeting(con, body, ch_url, platform, video_id, source_url, title, meeting_date, duration_s, raw):
    """One kind='meeting' row, in_frame = 0, status 'discovered'. ON CONFLICT DO
    NOTHING: a re-run never touches a row a stage owns. Returns 1 if inserted."""
    now = state.now_iso()
    cur = con.execute(
        "INSERT INTO videos (video_id, speaker, channel_url, month, title, upload_date, date_exact, duration_s, raw_meta, "
        "crawled_at, in_frame, status, attempts, updated_at, kind, platform, source_url, meeting_id, meeting_date) "
        "VALUES (?,?,?,?,?,?,0,?,?,?,0,'discovered',0,?,'meeting',?,?,?,?) ON CONFLICT(video_id) DO NOTHING",
        (video_id, body, ch_url, meeting_date[:7], title, None, duration_s, json.dumps(raw, ensure_ascii=False),
         now, now, platform, source_url, f"{body}:{meeting_date}", meeting_date))
    if cur.rowcount:
        con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                    (video_id, None, "discovered", now, f"meeting discovered ({platform})"))
    return cur.rowcount


TRANSIT_BOARDS = ("bart", "vta", "actransit", "samtrans", "mtc", "sfmta", "caltrain")
COUNTY_BOARDS = ("napa", "marin", "sonomacounty")


def body_kind_of(body):
    return "transit_board" if body in TRANSIT_BOARDS else ("county" if body in COUNTY_BOARDS else "council")


def save_listing(save_dir, body, platform, rows):
    """One file per body under save_dir: YouTube rows as the flat-playlist TSV
    (parse_listing_tsv reads it back), Granicus rows as JSON. T2 (2026-09-22):
    the full-run listing pass is made ONCE, the regular-title regexes are
    written from what it shows, and --from-listings inserts without a second
    round of YouTube requests."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if platform == "youtube":
        (Path(save_dir) / f"{body}.tsv").write_text("".join(
            f"{r['id']}\t{r.get('upload_date') or 'NA'}\t{r['duration_s'] if r.get('duration_s') is not None else 'NA'}\t"
            f"{r.get('live_status') or 'NA'}\t{r.get('title') or ''}\n" for r in rows), encoding="utf-8")
    else:
        (Path(save_dir) / f"{body}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")


def load_listing(from_dir, body, platform):
    """The saved listing for a body, or None when there is none."""
    p = Path(from_dir) / (f"{body}.tsv" if platform == "youtube" else f"{body}.json")
    if not p.exists():
        return None
    return parse_listing_tsv(p.read_text(encoding="utf-8")) if platform == "youtube" else json.loads(p.read_text(encoding="utf-8"))


def discover_meetings(con, channels, since=None, recent=0, playlist_end=40, bodies=None, log=print, sleep=None,
                      list_only=None, from_listings=None, until=None):
    """Run every kind=meeting source. Returns {body: (listed, regular, new)}.
    Stops (and reports) on a YouTube block signal; sources are one at a time
    with a random 45-180 s pause between YouTube listings.
    list_only=DIR: fetch and save the listings, insert nothing (the regexes
    are not needed yet). from_listings=DIR: read the saved listings instead
    of the network and insert (no YouTube request at all)."""
    import granicus                                        # noqa: E402
    import random
    out, first = {}, True
    for ch in channels:
        if ch["kind"] != "meeting" or (bodies and ch["speaker"] not in bodies):
            continue
        body, url = ch["speaker"], ch["url"]
        body_kind = body_kind_of(body)
        if "granicus.com" in url:
            site, view = granicus.parse_source_url(url)
            rows = load_listing(from_listings, body, "granicus") if from_listings else None
            if from_listings and rows is None:
                log(f"{body}: no saved Granicus listing in {from_listings}; skipped")
                out[body] = ("missing", 0, 0)
                continue
            if rows is None:
                try:
                    rows, info = granicus.fetch_listing(site, view)
                except Exception as e:                      # noqa: BLE001 -- one dead archive must not stop the pass
                    log(f"{body}: granicus {site} view {view}: listing failed ({type(e).__name__}: {str(e)[:120]})")
                    out[body] = ("error", 0, 0)
                    continue
            if list_only:
                save_listing(list_only, body, "granicus", rows)
                log(f"{body}: granicus {site} view {view}: {len(rows)} clips listed -> saved")
                out[body] = (len(rows), 0, 0)
                continue
            listed = [dict(r, title=r["name"], id=granicus.video_id_for(site, r["clip_id"]), live_status="was_live") for r in rows]
            regular = select_regular(listed, ch["regular_re"], since, recent, until=until)
            upsert_source(con, body, url, "meeting", ch["regular_re"], "granicus", f"{site}:{view}", body_kind)
            n_new = 0
            con.execute("BEGIN IMMEDIATE")
            for r in regular:
                n_new += insert_meeting(con, body, url, "granicus", r["id"], granicus.player_url(site, view, r["clip_id"]),
                                        r["title"], r["meeting_date"], r.get("duration_s"),
                                        {"clip_id": r["clip_id"], "site": site, "view_id": view, "listed_date": r.get("date"),
                                         "listed_duration": r.get("duration"), "agenda": r.get("agenda"), "minutes": r.get("minutes")})
            con.execute("COMMIT")
            log(f"{body}: granicus {site} view {view}: {len(rows)} clips listed, {len(regular)} regular since {since or 'ever'}, {n_new} new")
        else:
            rows = load_listing(from_listings, body, "youtube") if from_listings else None
            if from_listings and rows is None:
                log(f"{body}: no saved YouTube listing in {from_listings}; skipped (no request made)")
                out[body] = ("missing", 0, 0)
                continue
            if rows is None:
                if not first:
                    gap = (sleep if sleep is not None else random.uniform(45, 180))
                    log(f"  sleeping {gap:.0f} s before the next YouTube listing")
                    time.sleep(gap)
                first = False
                rows, err, rc = list_youtube(url, playlist_end)
                if BLOCK_SIGNAL.search(err or ""):
                    log(f"{body}: STOP -- YouTube block signal while listing {url}: {err.strip().splitlines()[-1][:200]}")
                    out[body] = ("blocked", 0, 0)
                    return out
                if rc != 0 and not rows:
                    log(f"{body}: yt-dlp exit {rc}, no rows: {(err or '').strip().splitlines()[-1:] }")
                    out[body] = ("error", 0, 0)
                    continue
            tab = url.rstrip("/").rsplit("/", 1)[-1]
            if list_only:
                save_listing(list_only, body, "youtube", rows)
                log(f"{body}: youtube {tab}: {len(rows)} listed -> saved")
                out[body] = (len(rows), 0, 0)
                continue
            regular = select_regular(rows, ch["regular_re"], since, recent, until=until)
            cid = url.split("/channel/")[1].split("/")[0] if "/channel/" in url else url
            upsert_source(con, body, url, "meeting", ch["regular_re"], "youtube", f"{cid}:{tab}", body_kind)
            n_new = 0
            con.execute("BEGIN IMMEDIATE")
            for r in regular:
                n_new += insert_meeting(con, body, url, "youtube", r["id"], f"https://www.youtube.com/watch?v={r['id']}",
                                        r["title"], r["meeting_date"], r.get("duration_s"),
                                        {"live_status": r.get("live_status"), "listed_upload_date": r.get("upload_date"), "tab": tab})
            con.execute("COMMIT")
            log(f"{body}: youtube {tab}: {len(rows)} listed, {len(regular)} regular since {since or 'ever'}, {n_new} new")
        out[body] = (len(rows), len(regular), n_new)
    return out


def load_catalog(con, catalog):
    # QUOTE_NONE: titles contain literal double quotes; the crawl never quotes fields.
    rows = list(csv.DictReader(open(catalog, newline="", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    n_new = 0
    con.execute("BEGIN IMMEDIATE")
    for r in rows:
        try:
            dur = float(r["duration_s"]) if r.get("duration_s") not in (None, "", "NA") else None
        except ValueError:
            dur = None
        try:
            views = int(r["view_count"]) if r.get("view_count") not in (None, "", "NA") else None
        except ValueError:
            views = None
        cur = con.execute(
            "INSERT INTO videos (video_id, speaker, channel_url, month, title, upload_date, date_exact, "
            "duration_s, view_count, raw_meta, crawled_at, in_frame, status, attempts, updated_at) "
            "VALUES (?,?,?,?,?,?,0,?,?,?,?,0,'discovered',0,?) ON CONFLICT(video_id) DO NOTHING",
            (r["video_id"], r["speaker"], r["channel_url"], month_of(r.get("upload_date")),
             r.get("title"), r.get("upload_date"), dur, views,
             json.dumps({k: r[k] for k in r if k not in ("title",)}), r.get("crawled_at") or state.now_iso(),
             state.now_iso()))
        n_new += cur.rowcount
    con.execute("COMMIT")
    return len(rows), n_new


def mark_frame(con, ids, note):
    """in_frame = 1 for these ids. Never un-marks: a row that left the frame is
    a study-design question, not something a script decides."""
    n = 0
    con.execute("BEGIN IMMEDIATE")
    for vid in ids:
        cur = con.execute("UPDATE videos SET in_frame = 1, updated_at = ? WHERE video_id = ? AND in_frame = 0",
                          (state.now_iso(), vid))
        if cur.rowcount:
            con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                        (vid, None, "discovered", state.now_iso(), f"in_frame: {note}"))
            n += 1
    con.execute("COMMIT")
    return n


def pick_recent(con, per_speaker, min_min=5, max_min=90):
    """Most recent eligible videos per speaker. Recent rows have exact-enough
    dates even in a flat-playlist catalog; older ones are bucketed (SAMPLING.md)."""
    picked = []
    speakers = [r[0] for r in con.execute("SELECT DISTINCT speaker FROM videos ORDER BY speaker")]
    for sp in speakers:
        rows = con.execute(
            "SELECT video_id FROM videos WHERE speaker = ? AND duration_s >= ? AND duration_s <= ? "
            "AND status = 'discovered' AND in_frame = 0 "
            "AND (raw_meta IS NULL OR raw_meta NOT LIKE '%\"live_status\": \"is_live\"%') "
            "ORDER BY upload_date DESC, video_id LIMIT ?",
            (sp, min_min * 60, max_min * 60, per_speaker)).fetchall()
        picked += [r["video_id"] for r in rows]
    return picked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=str(ROOT / "plan" / "catalog.tsv"))
    ap.add_argument("--frame", default=None, help="frame.csv (default plan/frame.csv if it exists)")
    ap.add_argument("--pick-recent", type=int, default=0, metavar="N")
    ap.add_argument("--mark", nargs="*", default=[], metavar="VIDEO_ID")
    ap.add_argument("--state", default=None)
    ap.add_argument("--meetings", action="store_true", help="list the kind=meeting sources in channels.tsv (network)")
    ap.add_argument("--channels", default=None, help="channels.tsv (default plan/channels.tsv)")
    ap.add_argument("--since", default="2025-09-20", help="meetings: keep meetings dated on/after this (PROPOSAL window)")
    ap.add_argument("--until", default="2026-09-20", help="meetings: keep meetings dated on/before this (PROPOSAL window)")
    ap.add_argument("--recent", type=int, default=0, help="meetings: keep only the newest N regular meetings per body")
    ap.add_argument("--bodies", default=None, help="meetings: comma list of speakers to run (default all)")
    ap.add_argument("--playlist-end", type=int, default=40)
    ap.add_argument("--list-only", default=None, metavar="DIR", help="meetings: fetch + save the listings to DIR, insert nothing")
    ap.add_argument("--from-listings", default=None, metavar="DIR", help="meetings: insert from listings saved by --list-only (no network)")
    a = ap.parse_args(argv)

    con = state.connect(a.state)
    if a.meetings:
        chans = read_channels(a.channels)
        res = discover_meetings(con, chans, a.since, a.recent, a.playlist_end,
                                bodies=[b.strip() for b in a.bodies.split(",")] if a.bodies else None,
                                list_only=a.list_only, from_listings=a.from_listings, until=a.until)
        n = con.execute("SELECT COUNT(*) FROM videos WHERE kind = 'meeting'").fetchone()[0]
        print(f"meetings in state DB: {n}; this run: " + ", ".join(f"{b}={v}" for b, v in res.items()))
        return 3 if any(v[0] == "blocked" for v in res.values()) else 0
    if Path(a.catalog).exists():
        total, new = load_catalog(con, a.catalog)
        print(f"catalog: {total} rows, {new} new")
    else:
        print(f"no catalog at {a.catalog} (run make crawl); marking only")

    frame = Path(a.frame) if a.frame else ROOT / "plan" / "frame.csv"
    if frame.exists():
        ids = [r["video_id"] for r in csv.DictReader(open(frame, newline="", encoding="utf-8"))]
        print(f"frame.csv: {len(ids)} videos, {mark_frame(con, ids, 'frame.csv')} newly marked")
    if a.pick_recent:
        ids = pick_recent(con, a.pick_recent)
        print(f"pick-recent {a.pick_recent}/speaker: {mark_frame(con, ids, 'pick-recent')} newly marked: {' '.join(ids)}")
    if a.mark:
        print(f"mark: {mark_frame(con, a.mark, 'manual')} newly marked")
    print("in-frame by status: " + ", ".join(f"{k}={v}" for k, v in state.counts(con).items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
