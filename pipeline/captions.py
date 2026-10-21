#!/usr/bin/env python3
"""
pipeline/captions.py -- meeting stage 1: 'discovered' -> 'captioned'
(PROPOSAL-transit.md 3d, T0 test 1; built 2026-09-21).

For every kind='meeting' row in 'discovered':

  youtube   yt-dlp --skip-download --write-auto-subs --write-subs
            --sub-langs "en,en-orig" --write-info-json --sleep-requests 2
            -> output/subs/<id>.en.vtt (+ <id>.info.json). Never "en.*" (the
            pattern behind the Tucker 429, PLAN.md 4.2). One request per
            meeting, sequential, a random 45-180 s pause between requests;
            STOP (exit 3, row back to 'discovered') on "Sign in to confirm",
            HTTP 429, or a second 403.
            A meeting fetched in T0 (output/t0/<id>/<id>.en.vtt + info.json)
            is COPIED from there instead of asked for again.
  granicus  player page (m3u8 / mp4 / captions track / agenda index points,
            saved to output/subs/<id>.player.json) then the caption track ->
            output/subs/<id>.en.vtt; pipeline/granicus.py paces requests >= 5 s.

The row records captions = auto | manual | granicus | none, captions_path, the
exact duration and (YouTube) upload date. A meeting with no captions still
moves to 'captioned' (captions='none'); classify.py then marks it
'no_episodes' with that reason -- the locate pass for such bodies is a T2
decision (PROPOSAL 2.2), not something this stage improvises.

TRICKLE MODE (--loop, 2026-09-22; slowed 2026-09-24). The first T2 caption run
asked YouTube 17 times in 31 min and was 429'd (T2-PLAN 3a). --loop is the slow
resumable worker that replaced it, and on 2026-09-23 it was 429'd again after
only FIVE requests in 41 minutes -- so the 8-12 min gap was still far too fast.
It now defaults to ONE meeting an hour with a hard cap on requests per rolling
24 h (--min-gap / --max-gap / --daily-cap, or CAPTIONS_MIN_GAP_S,
CAPTIONS_MAX_GAP_S, CAPTIONS_DAILY_CAP in .env). Both limits are enforced from
the `worker_requests` table, not from a counter inside the process, so a
launchd restart cannot walk straight back into the rate limit.

On a bot check or 429 it stops, writes a 24 h backoff into the `backoffs` table
and exits 0 -- so launchd's KeepAlive leaves it down, and any restart inside
those 24 h asks YouTube nothing (it reads the row and exits 0 again). Rows stay
'discovered'; nothing is retried, nothing is charged an attempt. UNCHANGED on
2026-09-24: that is the one behaviour this file must not get cleverer about.

QUEUE ORDER (2026-09-24). Not a flat oldest-first sweep any more: measured
yield -- episodes located per captioned meeting -- varies 17-fold across the
waiting bodies (BART 16.7, Rohnert Park 0.00 over 5 captioned meetings), so at
one meeting an hour the order decides what the study has in a fortnight. The
worker takes the highest-yield body's OLDEST uncaptioned meeting next. Yield is
recomputed from the database on every pass (body_yield); nothing is dropped --
which bodies belong in the sample is a DECISIONS.md question for Puck, not a
worker's -- and the rank, yield and denominator go in the log line so the
ordering can be audited after the fact.

    python3 pipeline/captions.py --max 12                # this many meetings, then exit
    python3 pipeline/captions.py --dry-run               # print the plan, no network
    python3 pipeline/captions.py --bodies oakland,bart
    python3 pipeline/captions.py --loop                  # trickle worker (launchd)
    python3 pipeline/captions.py --yield-table           # the measured order, no network
    python3 pipeline/captions.py --backoff-status        # when may YouTube be asked again?
    python3 pipeline/captions.py --rate-status           # requests in the last 24 h vs the cap
    python3 pipeline/captions.py --clear-backoff         # a human ends the backoff early
"""
import argparse, json, os, random, re, shutil, subprocess, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402
import granicus  # noqa: E402
from downloader import ytdlp_bin, BOT_CHECK, FORBIDDEN, error_line, random_gap  # noqa: E402
from notify import notify  # noqa: E402

ROOT = state.ROOT
T0 = ROOT / "output" / "t0"
INLINE_TS = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>")
EXIT_BOT_CHECK = 3
WORKER = "captions"
BACKOFF_HOURS = 24.0
# 2026-09-24: 5 requests in 41 min was 429'd, so the trickle is now one meeting
# an hour (plus up to 10 min of jitter, so the pattern is not a metronome) with
# a hard cap per rolling 24 h. 18 < 24 on purpose: the cap is a backstop that
# bites even if the gap logic is wrong or the worker is restarted repeatedly.
LOOP_MIN_GAP = 60 * 60.0
LOOP_MAX_GAP = 70 * 60.0
LOOP_DAILY_CAP = 18
RATE_WINDOW_H = 24.0
YT_HOST = "youtube"


def subs_dir():
    return Path(os.path.expanduser(state.setting("SUBS_DIR", str(ROOT / "output" / "subs"))))


def caption_kind(vtt_path):
    """'auto' (YouTube ASR: inline word timings), 'manual' (a human track or a
    Granicus live-captioner track) or 'none' (missing / no cues)."""
    p = Path(vtt_path)
    if not p.exists():
        return "none"
    raw = p.read_text(encoding="utf-8", errors="replace")
    if raw.count("-->") < 5:
        return "none"
    return "auto" if INLINE_TS.search(raw) else "manual"


def ytdlp_captions_command(video_id, out_dir):
    return [ytdlp_bin(), "--skip-download", "--write-auto-subs", "--write-subs", "--sub-langs", "en,en-orig",
            "--write-info-json", "--no-write-playlist-metafiles", "--no-playlist", "--no-progress",
            "--sleep-requests", "2", "--retries", "2",
            "-o", str(Path(out_dir) / "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={video_id}"]


def t0_cached(video_id, out_dir):
    """Copy the T0 caption + info.json for this meeting into out_dir if T0
    fetched them (output/t0/<id>/). Returns True when a vtt was copied."""
    d = T0 / video_id
    if not d.is_dir():
        return False
    got = False
    for name in (f"{video_id}.en.vtt", f"{video_id}.info.json"):
        src, dst = d / name, Path(out_dir) / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        got = got or (name.endswith(".vtt") and dst.exists())
    return got


def info_fields(info_path):
    """The exact metadata a yt-dlp info.json carries (same keys downloader.record_metadata keeps)."""
    p = Path(info_path)
    if not p.exists():
        return {}
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    keep = {k: info.get(k) for k in ("id", "title", "upload_date", "timestamp", "duration", "view_count", "like_count",
                                      "channel", "channel_id", "uploader", "live_status", "was_live", "availability",
                                      "language", "chapters", "release_timestamp", "subtitles_langs")}
    keep["subtitles_langs"] = sorted((info.get("subtitles") or {}).keys())[:20]
    keep["auto_langs_n"] = len(info.get("automatic_captions") or {})
    f = {"raw_meta": json.dumps(keep, ensure_ascii=False)}
    if info.get("upload_date"):
        f["upload_date"], f["date_exact"] = info["upload_date"], 1
    if info.get("duration"):
        f["duration_s"] = float(info["duration"])
    if info.get("title"):
        f["title"] = info["title"]
    for k in ("view_count", "like_count"):
        if info.get(k) is not None:
            f[k] = int(info[k])
    return f


def captions_youtube(video_id, out_dir, dry_run=False, consecutive_403=0):
    """('ok'|'error'|'bot', fields, message, made_request)."""
    out_dir = Path(out_dir)
    vtt = out_dir / f"{video_id}.en.vtt"
    made_request = False
    if not vtt.exists() and t0_cached(video_id, out_dir):
        src = "T0 cache (output/t0)"
    elif vtt.exists():
        src = "already on disk"
    else:
        cmd = ytdlp_captions_command(video_id, out_dir)
        if dry_run:
            return "ok", {}, "DRY RUN: " + " ".join(cmd), False
        made_request = True
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return "error", {}, "yt-dlp timed out after 900 s", True
        if r.returncode != 0:
            err = r.stderr or ""
            if BOT_CHECK.search(err) or (FORBIDDEN.search(err) and consecutive_403 + 1 >= 2):
                return "bot", {}, error_line(err), True
            return "error", {}, error_line(err), True
        src = "yt-dlp"
    kind = caption_kind(vtt)
    fields = info_fields(out_dir / f"{video_id}.info.json")
    fields.update(captions=kind, captions_path=str(vtt) if kind != "none" else None)
    return "ok", fields, f"captions {kind} via {src}" + (f", {vtt.stat().st_size / 1e3:.0f} kB" if vtt.exists() else ""), made_request


def captions_granicus(video_id, raw_meta, out_dir, dry_run=False, force=False):
    """Player page -> stream URLs + index points (saved as <id>.player.json), then the caption track.

    force=True re-asks for the player page and re-parses it even though
    <id>.player.json is already on disk (--refetch: a parser fix changed what
    the page means, 2026-09-22). Nothing is deleted: the player JSON is
    rewritten and a caption file with no cues -- the empty 8-byte track some
    sites serve -- is overwritten by a real one if there now is one."""
    out_dir = Path(out_dir)
    meta = json.loads(raw_meta or "{}")
    site, view, clip = meta.get("site"), meta.get("view_id"), meta.get("clip_id")
    if not (site and view and clip):
        return "error", {}, "granicus row without site/view_id/clip_id in raw_meta", False
    pj = out_dir / f"{video_id}.player.json"
    vtt = out_dir / f"{video_id}.en.vtt"
    if dry_run:
        return "ok", {}, f"DRY RUN: GET {granicus.player_url(site, view, clip)} then its captions track", False
    t0_player, t0_vtt = T0 / "granicus" / site / f"player_{clip}.html", T0 / "granicus" / site / f"captions_{clip}.vtt"
    if pj.exists() and not force:
        player = json.loads(pj.read_text(encoding="utf-8"))
    elif t0_player.exists() and not force:                  # the T0 probe saved this page: parse it, do not re-request
        player = granicus.parse_player(t0_player.read_text(encoding="utf-8", errors="replace"), site)
        pj.write_text(json.dumps(player, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        try:
            player, info = granicus.fetch_player(site, view, clip)
        except Exception as e:                                  # noqa: BLE001 -- a 4xx/5xx/timeout is this row's failure, not a crash of the run
            return "error", {}, f"player page: {type(e).__name__}: {str(e)[:160]}", True
        pj.write_text(json.dumps(player, ensure_ascii=False, indent=1), encoding="utf-8")
    n_bytes = n_cues = 0
    if t0_vtt.exists() and not vtt.exists():
        shutil.copy2(t0_vtt, vtt)
        n_bytes, n_cues = vtt.stat().st_size, vtt.read_text(encoding="utf-8", errors="replace").count("-->")
    # A track with no cues is not a fetched track: the 8-byte "WEBVTT" file
    # Sausalito / SamTrans / AC Transit serve is indistinguishable on disk from
    # a track we never asked for, so ask again when the page now names one.
    if player.get("captions_url") and caption_kind(vtt) == "none":
        try:
            n_bytes, n_cues = granicus.fetch_captions(player["captions_url"], vtt)
        except Exception as e:                                  # noqa: BLE001 -- 404 = no track; recorded, not fatal
            return "error", {}, f"captions track: {type(e).__name__}: {str(e)[:160]}", True
    kind = "granicus" if caption_kind(vtt) != "none" else "none"
    meta.update({"m3u8": player.get("m3u8"), "mp4": player.get("mp4"), "captions_url": player.get("captions_url"),
                 "captions_lang": player.get("captions_lang"), "captions_label": player.get("captions_label"),
                 "index_points": len(player.get("index_points") or []), "captions_cues": n_cues or None})
    fields = {"raw_meta": json.dumps(meta, ensure_ascii=False), "captions": kind,
              "captions_path": str(vtt) if kind != "none" else None}
    return "ok", fields, (f"captions {kind} ({n_cues} cues, {n_bytes / 1e3:.0f} kB), m3u8 {'yes' if player.get('m3u8') else 'NO'}, "
                          f"{len(player.get('index_points') or [])} index points"), True


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


# --- the 24 h backoff (T2, 2026-09-22) ------------------------------------------

def backoff_left(con, worker=WORKER, now=None):
    """(seconds still to wait, row) for this worker's backoff; (0.0, row|None)
    when there is none or it has expired. An unparseable `until` is treated as
    expired -- a broken row must never wedge the worker shut for good."""
    r = con.execute("SELECT * FROM backoffs WHERE worker = ?", (worker,)).fetchone()
    if not r:
        return 0.0, None
    try:
        until = datetime.fromisoformat(r["until"])
    except (TypeError, ValueError):
        return 0.0, r
    return max(0.0, (until - (now or datetime.now(timezone.utc))).total_seconds()), r


def set_backoff(con, reason, hours=BACKOFF_HOURS, worker=WORKER, now=None):
    """Record that YouTube challenged us: no requests from this worker until
    `hours` from now. Returns the ISO timestamp written."""
    now = now or datetime.now(timezone.utc)
    until = (now + timedelta(hours=hours)).isoformat(timespec="seconds")
    con.execute("INSERT INTO backoffs (worker, until, reason, set_at) VALUES (?,?,?,?) "
                "ON CONFLICT(worker) DO UPDATE SET until = excluded.until, reason = excluded.reason, set_at = excluded.set_at",
                (worker, until, (reason or "")[:300], now.isoformat(timespec="seconds")))
    return until


def clear_backoff(con, worker=WORKER):
    return con.execute("DELETE FROM backoffs WHERE worker = ?", (worker,)).rowcount


# --- the rate limit: one an hour, with a hard daily cap (2026-09-24) -------------
# Both halves read `worker_requests`, so they hold across a restart. The 429 of
# 2026-09-23 happened at 5 requests / 41 min; a process-local counter cannot
# stop launchd from handing that back after ThrottleInterval.

def record_request(con, video_id, worker=WORKER, host=YT_HOST, now=None):
    """One row per request about to be sent; returns its id. Written BEFORE the
    call, so a killed process still counts the request it made."""
    return con.execute("INSERT INTO worker_requests (worker, host, at, video_id) VALUES (?,?,?,?)",
                       (worker, host, (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
                        video_id)).lastrowid


def recent_requests(con, hours=RATE_WINDOW_H, worker=WORKER, host=YT_HOST, now=None):
    """The request timestamps inside the rolling window, oldest first."""
    since = ((now or datetime.now(timezone.utc)) - timedelta(hours=hours)).isoformat(timespec="seconds")
    rows = con.execute("SELECT at FROM worker_requests WHERE worker = ? AND host = ? AND at >= ? ORDER BY at",
                       (worker, host, since)).fetchall()
    out = []
    for r in rows:
        try:
            out.append(datetime.fromisoformat(r["at"]))
        except (TypeError, ValueError):
            continue                                  # a corrupt row must not wedge the worker
    return out


def rate_wait(con, min_gap, daily_cap, hours=RATE_WINDOW_H, worker=WORKER, host=YT_HOST, now=None):
    """(seconds to wait before the next request, why). 0.0 when it may go now.

    Two independent limits: at least `min_gap` since the last request, and no
    more than `daily_cap` inside the rolling `hours` window. When the cap is
    full the wait is until the OLDEST request in the window ages out of it --
    never a guess, and never longer than the window."""
    now = now or datetime.now(timezone.utc)
    ats = recent_requests(con, hours, worker, host, now)
    waits = []
    if ats:
        gap_left = min_gap - (now - ats[-1]).total_seconds()
        if gap_left > 0:
            waits.append((gap_left, f"{min_gap / 60:.0f} min gap: {gap_left / 60:.1f} min left since {ats[-1].isoformat()}"))
    if daily_cap and len(ats) >= daily_cap:
        # ats[len(ats) - daily_cap] is the request whose expiry frees a slot.
        free_at = ats[len(ats) - daily_cap] + timedelta(hours=hours)
        cap_left = (free_at - now).total_seconds()
        if cap_left > 0:
            waits.append((cap_left, f"daily cap {daily_cap}/{hours:.0f} h reached ({len(ats)} requests); "
                                    f"a slot frees at {free_at.isoformat(timespec='seconds')}"))
    if not waits:
        return 0.0, f"{len(ats)} request(s) in the last {hours:.0f} h of {daily_cap}"
    return max(waits)


# --- queue order: measured yield, highest first (2026-09-24) --------------------

def body_yield(con, platform=None):
    """{body: {'eps', 'captioned', 'yield', 'measured'}} -- episodes located per
    captioned meeting, measured from the database, never hard-coded.

    Denominator = meetings this body has already been THROUGH the captions stage
    for (captions IS NOT NULL), including the ones that turned out to have no
    track: a body whose meetings carry no captions costs a request and returns
    nothing, and that is exactly what the ordering should know. Numerator = every
    episodes row located from those meetings, any method.

    A body with no captioned meeting yet has no measurement. It gets the pooled
    rate as a prior and is marked measured=False, so it sits mid-table instead of
    last -- an unmeasured body is not a zero-yield body."""
    where = "kind = 'meeting'" + (" AND platform = ?" if platform else "")
    params = [platform] if platform else []
    rows = con.execute(
        f"""SELECT v.speaker AS body,
                   SUM(CASE WHEN v.captions IS NOT NULL THEN 1 ELSE 0 END) AS captioned,
                   (SELECT COUNT(*) FROM episodes e JOIN videos v2 ON v2.video_id = e.video_id
                      WHERE v2.speaker = v.speaker AND v2.kind = 'meeting'
                      {"AND v2.platform = ?" if platform else ""}) AS eps
              FROM videos v WHERE {where} GROUP BY v.speaker""",
        (params + params) if platform else params).fetchall()
    out = {r["body"]: {"eps": int(r["eps"] or 0), "captioned": int(r["captioned"] or 0)} for r in rows}
    tot_eps = sum(v["eps"] for v in out.values())
    tot_cap = sum(v["captioned"] for v in out.values())
    pooled = (tot_eps / tot_cap) if tot_cap else 0.0
    for v in out.values():
        v["measured"] = v["captioned"] > 0
        v["yield"] = (v["eps"] / v["captioned"]) if v["captioned"] else pooled
    return out, pooled


def order_queue(rows, yields, pooled=0.0):
    """The waiting meetings, highest-yield body first and oldest meeting first
    inside a body. Returns [(row, rank, n_bodies)] -- rank is the body's place in
    the yield order, carried into the log line so the order can be audited.

    Nothing is dropped and nothing is filtered: this is an ORDERING. Whether a
    zero-yield body belongs in the sample at all is a DECISIONS.md question."""
    def y(body):
        return yields.get(body, {}).get("yield", pooled)
    bodies = sorted({r["speaker"] for r in rows}, key=lambda b: (-y(b), b))
    rank = {b: i + 1 for i, b in enumerate(bodies)}
    ordered = sorted(rows, key=lambda r: (rank.get(r["speaker"], len(bodies) + 1),
                                          r["meeting_date"] or "", r["video_id"]))
    return [(r, rank.get(r["speaker"], len(bodies) + 1), len(bodies)) for r in ordered]


def yield_line(body, yields, rank, n_bodies, pooled=0.0):
    y = yields.get(body, {})
    v, eps, cap = y.get("yield", pooled), y.get("eps", 0), y.get("captioned", 0)
    return (f"[yield rank {rank}/{n_bodies}: {v:.2f} ep/meeting"
            + (f" from {eps} ep over {cap} captioned]" if y.get("measured") else f", POOLED PRIOR ({cap} captioned)]"))


def select_rows(con, bodies=None, skip_routed=True):
    """Meetings still waiting for captions. No in-progress state: a killed run
    leaves the row 'discovered' and the next run re-checks the disk. The SQL
    order is oldest-first WITHIN a body; order_queue puts the bodies in measured
    yield order.

    skip_routed (2026-09-24): a meeting that ALREADY HAS A NON-YOUTUBE ROUTE is
    not in this queue. Petaluma's 15 waiting meetings are each published on
    Swagit as well (pipeline/swagit.py, plan/SOURCES-AUDIT.md), the Swagit row
    for the same `meeting_id` carries its own transcript, and asking YouTube for
    a second one would spend a request from an 18/day cap on a meeting already
    located. The YouTube row is NOT touched, deleted or re-stated: it stays
    'discovered' and comes straight back into the queue if its Swagit twin is
    ever removed. `--include-routed` asks anyway."""
    where = "kind = 'meeting' AND status = 'discovered'"
    params = []
    if skip_routed:
        where += (" AND NOT EXISTS (SELECT 1 FROM videos o WHERE o.kind = 'meeting' AND o.platform != videos.platform "
                  "AND o.platform != 'youtube' AND o.meeting_id = videos.meeting_id AND o.meeting_id IS NOT NULL "
                  "AND o.captions IS NOT NULL AND o.captions != 'none')")
    if bodies:
        where += f" AND speaker IN ({','.join('?' * len(bodies))})"
        params = bodies
    return con.execute(f"SELECT video_id, speaker, platform, raw_meta, title, meeting_date FROM videos WHERE {where} "
                       "ORDER BY meeting_date, video_id", params).fetchall()


def refetch_granicus(con, bodies, out_dir, dry_run=False, limit=0):
    """Ask a Granicus body's player pages again for meetings recorded as having
    no captions, and re-parse them (--refetch). A human's explicit request after
    a parser fix -- 2026-09-22: parse_player learned the Flowplayer subtitles
    object, which is how Santa Clara and Hayward serve a full English track,
    and 69 of their meetings had been written off as caption-less.

    A meeting that turns out to have a real track goes back to 'captioned' so
    classify.py can pick it up; one that still has none keeps its status.
    Nothing is deleted (AGENTS.md rule 6): the player JSON is rewritten and an
    empty track is overwritten only if a real one comes back."""
    where = ("kind = 'meeting' AND platform = 'granicus' AND (captions IS NULL OR captions = 'none') "
             "AND status IN ('no_episodes', 'captioned', 'discovered')")
    params = []
    if bodies:
        where += f" AND speaker IN ({','.join('?' * len(bodies))})"
        params = bodies
    rows = con.execute(f"SELECT * FROM videos WHERE {where} ORDER BY speaker, meeting_date DESC"
                       + (f" LIMIT {int(limit)}" if limit else ""), params).fetchall()
    log(f"--refetch: {len(rows)} Granicus meeting(s) recorded without captions -> {out_dir}" + (" (dry run)" if dry_run else ""))
    stats = {"ok": 0, "found": 0, "failed": 0, "requests": 0}
    for r in rows:
        vid = r["video_id"]
        kind, fields, msg, made = captions_granicus(vid, r["raw_meta"], out_dir, dry_run, force=True)
        stats["requests"] += made
        if dry_run:
            log(f"  {vid} ({r['speaker']}): {msg}")
            continue
        if kind != "ok":
            stats["failed"] += 1
            log(f"FAILED {vid} ({r['speaker']}): {msg}")
            continue
        stats["ok"] += 1
        got = fields.get("captions") not in (None, "none")
        stats["found"] += got
        if got and r["status"] != "captioned":
            state.transition(con, vid, r["status"], "captioned", note=f"--refetch: {msg}"[:200], **fields)
        else:
            con.execute("UPDATE videos SET raw_meta = ?, captions = ?, captions_path = ?, updated_at = ? WHERE video_id = ?",
                        (fields["raw_meta"], fields["captions"], fields["captions_path"], state.now_iso(), vid))
        log(f"{'CAPTIONS ' if got else 'still none'} {vid} ({r['speaker']}): {msg}")
    log(f"--refetch done: {stats['ok']} re-checked, {stats['found']} now have a caption track, "
        f"{stats['failed']} failed, {stats['requests']} requests")
    return 0


def run(args):
    con = state.connect(args.state)
    bodies = [b.strip() for b in args.bodies.split(",") if b.strip()] if args.bodies else None
    if args.clear_backoff:
        n = clear_backoff(con)
        log(f"backoff cleared ({n} row(s)); the next run may ask YouTube again")
        con.close()
        return 0
    if args.backoff_status:
        left, row = backoff_left(con)
        log(f"backoff: {left / 3600:.2f} h left (until {row['until']}, set {row['set_at']}: {row['reason']})" if left > 0
            else ("no active backoff" + (f" (last one expired at {row['until']})" if row else "")))
        con.close()
        return 0
    if args.rate_status:
        ats = recent_requests(con)
        wait, why = rate_wait(con, args.min_gap, args.daily_cap)
        log(f"rate: {len(ats)} YouTube request(s) in the last {RATE_WINDOW_H:.0f} h, cap {args.daily_cap}; "
            f"gap {args.min_gap / 60:.0f}-{args.max_gap / 60:.0f} min"
            + (f"; next request may go in {wait / 60:.1f} min ({why})" if wait > 0 else f"; may request now ({why})"))
        if ats:
            log(f"  last request {ats[-1].isoformat()}, first in window {ats[0].isoformat()}")
        con.close()
        return 0
    if args.yield_table:
        yields, pooled = body_yield(con, args.platform)
        rows = select_rows(con, bodies, not args.include_routed)
        waiting = {}
        for r in rows:
            waiting[r["speaker"]] = waiting.get(r["speaker"], 0) + 1
        ordered = order_queue(rows, yields, pooled)
        seen, cum = set(), 0
        log(f"measured yield (episodes located per captioned meeting), pooled prior {pooled:.2f}; "
            f"{len(rows)} meeting(s) waiting")
        print(f"{'rank':>4} {'body':14}{'yield':>7}{'eps':>6}{'capt':>6}{'waiting':>8}{'cum':>7}  measured")
        for r, rank, n in ordered:
            b = r["speaker"]
            if b in seen:
                continue
            seen.add(b)
            y = yields.get(b, {})
            cum += waiting.get(b, 0)
            print(f"{rank:>4} {b:14}{y.get('yield', pooled):>7.2f}{y.get('eps', 0):>6}{y.get('captioned', 0):>6}"
                  f"{waiting.get(b, 0):>8}{cum:>7}  {'yes' if y.get('measured') else 'NO (prior)'}")
        con.close()
        return 0
    out_dir = subs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.refetch:
        rc = refetch_granicus(con, bodies, out_dir, args.dry_run, args.max)
        con.close()
        return rc
    left, row = backoff_left(con)
    if left > 0 and not args.dry_run:
        # A restart inside the backoff window asks YouTube nothing at all.
        log(f"BACKING OFF: {left / 3600:.1f} h of {BACKOFF_HOURS:.0f} h left (until {row['until']}, set {row['set_at']}: "
            f"{row['reason']}); no requests made. `--clear-backoff` ends it early.")
        con.close()
        return 0
    rng = random.Random()
    stats = {"ok": 0, "failed": 0, "bot": None, "requests": 0, "capped": None}
    consecutive_403, done = 0, 0

    def refill():
        """The waiting meetings in measured-yield order, recomputed every pass:
        a meeting captioned an hour ago changes its body's yield."""
        yields, pooled = body_yield(con, args.platform)
        return order_queue(list(select_rows(con, bodies, not args.include_routed)), yields, pooled), yields, pooled

    queue, yields, pooled = refill()
    top = ", ".join(f"{b} {yields.get(b, {}).get('yield', pooled):.1f}"
                    for b in dict.fromkeys(r["speaker"] for r, _, _ in queue[:60]))[:200]
    log(f"{len(queue)} meeting(s) to caption -> {out_dir}" + (" (dry run)" if args.dry_run else "")
        + f"; order = measured yield (pooled prior {pooled:.2f}), highest first: {top}"
        + f"; one meeting every {args.min_gap / 60:.0f}-{args.max_gap / 60:.0f} min, cap {args.daily_cap}"
          f"/{RATE_WINDOW_H:.0f} h, {BACKOFF_HOURS:.0f} h backoff on a bot check")
    while True:
        if not queue:
            if not args.loop or args.dry_run:
                break
            time.sleep(args.idle_sleep)
            queue, yields, pooled = refill()
            continue
        r, rank, n_bodies = queue.pop(0)
        vid = r["video_id"]
        order = yield_line(r["speaker"], yields, rank, n_bodies, pooled)
        if r["platform"] == "youtube":
            need_request = not (out_dir / f"{vid}.en.vtt").exists() and not (T0 / vid / f"{vid}.en.vtt").exists()
            if need_request and not args.dry_run:
                # Both limits come out of the DB, so a restart cannot skip them.
                gap = random_gap(args.min_gap, args.max_gap, rng)
                wait, why = rate_wait(con, gap, args.daily_cap)
                if wait > 0 and not args.loop:
                    # A one-shot run does not sit on the clock for hours: it says
                    # why and stops. Nothing is retried, no row changes.
                    stats["capped"] = why
                    log(f"RATE LIMIT: not asking YouTube for {vid} -- {why}. Stopping; re-run later "
                        f"or use --loop. (`--rate-status`)")
                    break
                if wait > 0:
                    log(f"  waiting {wait / 60:.1f} min before the next YouTube request -- {why}")
                    time.sleep(wait)
                rq_id = record_request(con, vid)
            kind, fields, msg, made = captions_youtube(vid, out_dir, args.dry_run, consecutive_403)
            if need_request and not args.dry_run and not made:
                # Recorded before the call on purpose (a killed process must not
                # lose the fact that it asked); give the slot back if nothing was
                # actually sent -- a file that appeared on disk meanwhile.
                con.execute("DELETE FROM worker_requests WHERE id = ?", (rq_id,))
        else:
            kind, fields, msg, made = captions_granicus(vid, r["raw_meta"], out_dir, args.dry_run)
        stats["requests"] += made
        if args.dry_run:
            log(f"  {vid} ({r['speaker']}): {msg} {order}")
            done += 1
            if args.max and done >= args.max:
                break
            continue
        if kind == "ok":
            state.transition(con, vid, "discovered", "captioned", note=msg[:200], **fields)
            stats["ok"] += 1
            consecutive_403 = 0
            log(f"captioned {vid} ({r['speaker']}): {msg} {order}")
        elif kind == "bot":
            stats["bot"] = msg
            until = set_backoff(con, f"{vid}: {msg}")
            log(f"STOP: YouTube is challenging this IP on {vid}: {msg}")
            log(f"backing off until {until} ({BACKOFF_HOURS:.0f} h); the row stays 'discovered' and nothing is retried")
            notify("error", f"captions worker STOPPED on {vid}: {msg}. Backing off until {until} "
                            f"({stats['ok']} meetings captioned this run). Not retrying; waiting for you.")
            break
        else:
            consecutive_403 += 1 if FORBIDDEN.search(msg) else 0
            new = state.fail(con, vid, "discovered", msg)
            stats["failed"] += 1
            log(f"{new.upper()} {vid}: {msg} {order}")
        done += 1
        if args.max and done >= args.max:
            break
    n_recent = len(recent_requests(con))
    con.close()
    log(f"done: {stats['ok']} captioned, {stats['failed']} failed, {stats['requests']} network requests"
        f", {n_recent}/{args.daily_cap} YouTube requests in the last {RATE_WINDOW_H:.0f} h"
        + (f", RATE LIMITED: {stats['capped']}" if stats["capped"] else "")
        + (f", STOPPED: {stats['bot']}" if stats["bot"] else ""))
    if stats["bot"] and args.loop:
        # Exactly the downloader's rule (AGENTS.md 2): a non-zero exit would let
        # launchd's KeepAlive retry YouTube every ThrottleInterval. Exit 0 so the
        # worker stays DOWN; the backoff row above makes a manual restart inside
        # the next 24 h a no-op as well.
        log("STOPPED by a YouTube challenge in --loop mode: exiting 0 so launchd does not restart the worker; "
            "resume with `launchctl kickstart -k gui/$(id -u)/com.ytpt.captions` after the backoff (--backoff-status)")
        return 0
    return EXIT_BOT_CHECK if stats["bot"] else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=0, help="stop after this many meetings (0 = no limit)")
    ap.add_argument("--bodies", default=None)
    ap.add_argument("--loop", action="store_true", help="keep polling when the queue is empty (launchd trickle worker)")
    ap.add_argument("--min-gap", type=float, default=None,
                    help=f"minimum seconds between YouTube requests (default CAPTIONS_MIN_GAP_S or {LOOP_MIN_GAP:.0f} = one an hour)")
    ap.add_argument("--max-gap", type=float, default=None,
                    help=f"upper end of the randomised gap (default CAPTIONS_MAX_GAP_S or {LOOP_MAX_GAP:.0f})")
    ap.add_argument("--daily-cap", type=int, default=None,
                    help=f"hard cap on YouTube requests per rolling {RATE_WINDOW_H:.0f} h "
                         f"(default CAPTIONS_DAILY_CAP or {LOOP_DAILY_CAP}; 0 = no cap)")
    ap.add_argument("--platform", default="youtube",
                    help="platform the yield table is measured over (default youtube; '' = every platform)")
    ap.add_argument("--idle-sleep", type=float, default=300, help="--loop: seconds to wait when the queue is empty")
    ap.add_argument("--refetch", action="store_true",
                    help="Granicus only: ask the player pages of meetings recorded without captions again (after a parser fix)")
    ap.add_argument("--backoff-status", action="store_true", help="print the backoff and exit")
    ap.add_argument("--rate-status", action="store_true", help="print requests in the last 24 h vs the cap, and exit")
    ap.add_argument("--yield-table", action="store_true",
                    help="print the measured yield order the worker would use (no network) and exit")
    ap.add_argument("--include-routed", action="store_true",
                    help="also queue meetings that already have a non-YouTube route (Petaluma via Swagit); "
                         "off by default -- see select_rows")
    ap.add_argument("--clear-backoff", action="store_true", help="end the backoff early (a human's decision) and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state", default=None)
    args = ap.parse_args(argv)
    # One an hour is the default in EVERY mode now, not just --loop: a hand-run
    # `--max 12` at the old 45-180 s gaps would spend two thirds of the daily cap
    # in twenty minutes, which is the shape of both 429s (T2-PLAN 3a, 3g).
    if args.min_gap is None:
        args.min_gap = float(state.setting("CAPTIONS_MIN_GAP_S", LOOP_MIN_GAP))
    if args.max_gap is None:
        args.max_gap = float(state.setting("CAPTIONS_MAX_GAP_S", max(LOOP_MAX_GAP, args.min_gap)))
    if args.daily_cap is None:
        args.daily_cap = int(state.setting("CAPTIONS_DAILY_CAP", LOOP_DAILY_CAP))
    args.platform = args.platform or None
    if args.min_gap < 10 or args.max_gap < args.min_gap:
        ap.error("--min-gap must be >= 10 and <= --max-gap")
    if args.daily_cap < 0:
        ap.error("--daily-cap must be >= 0 (0 = no cap)")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
