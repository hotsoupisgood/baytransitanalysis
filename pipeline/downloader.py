#!/usr/bin/env python3
"""
pipeline/downloader.py -- the laptop download worker (PLAN.md 4.2).

Takes in-frame 'discovered' rows one at a time, fetches audio only with yt-dlp
into output/audio/<id>.<ext>, verifies the file with ffprobe, stores the exact
metadata (upload date, counts, full info JSON, SponsorBlock spans) and moves the
row to 'downloaded'. Plain script; runs under launchd or by hand. OpenClaw
supervises it, it never runs inside OpenClaw (DECISIONS.md 6).

The rules it enforces, all from Puck (2026-09-17/18):
  * Residential IP only, no cookies (DECISIONS.md 1; PLAN.md 4.2 "try without
    cookies first"). There is no --cookies flag here on purpose.
  * Concurrency 1 by default, hard maximum 2.
  * A RANDOM pause between downloads, uniform 45-180 s plus jitter, so the
    request pattern does not look like a script. --min-gap/--max-gap change it;
    the pause is never skipped except by --max 1.
  * On "Sign in to confirm you're not a bot", HTTP 429, or three 403s in a row:
    STOP the worker at once (exit code 3; exit 0 under --loop so launchd's
    KeepAlive does not restart it every ThrottleInterval), put the video back
    to 'discovered' without charging it an attempt, and notify. No retry loop.
    A human decides.
  * Any other failure: 'failed' with the yt-dlp error, attempts += 1; after 3
    attempts 'dead' (state.MAX_ATTEMPTS) and Puck is told.

Crash safety: a killed worker leaves rows in 'downloading'. At start-up every
such row goes back to 'discovered' (only one downloader runs per machine). yt-dlp
writes '<id>.<ext>.part' and renames on completion, so a partial file is never
the final path; ffprobe confirms the duration before the row is marked.

    python3 pipeline/downloader.py --max 4            # this many downloads, then exit
    python3 pipeline/downloader.py --loop             # run until killed (launchd)
    python3 pipeline/downloader.py --dry-run --max 1  # print the command only
"""
import argparse, json, os, random, re, shutil, subprocess, sys, threading, time, urllib.parse, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402
from notify import notify  # noqa: E402

ROOT = state.ROOT
BOT_CHECK = re.compile(r"sign in to confirm|not a bot|HTTP Error 429|Too Many Requests", re.I)
FORBIDDEN = re.compile(r"HTTP Error 403", re.I)
MAX_CONCURRENCY = 2
EXIT_BOT_CHECK = 3


def ytdlp_bin():
    if os.environ.get("YTDLP_BIN"):                      # tests substitute a stub
        return os.environ["YTDLP_BIN"]
    p = ROOT / ".venv" / "bin" / "yt-dlp"
    return str(p) if p.exists() else (shutil.which("yt-dlp") or "yt-dlp")


def audio_dir():
    return Path(os.path.expanduser(state.setting("AUDIO_DIR", str(ROOT / "output" / "audio"))))


def ytdlp_command(video_id, out_dir, limit_rate="3M"):
    # PLAN.md 4.2 minus cookies. --write-info-json gives the exact upload date
    # and every field we might want later (PLAN.md 4.1) for free.
    return [ytdlp_bin(),
            "-f", "bestaudio[ext=m4a]/bestaudio",
            "--no-playlist", "--no-progress",
            "--sleep-requests", "2", "--min-sleep-interval", "5", "--max-sleep-interval", "15",
            "--limit-rate", limit_rate,
            "--retries", "5", "--fragment-retries", "10",
            "--write-info-json", "--no-write-playlist-metafiles",
            "-o", str(Path(out_dir) / "%(id)s.%(ext)s"),
            f"https://www.youtube.com/watch?v={video_id}"]


def random_gap(lo, hi, rng=random):
    """Uniform on [lo, hi] plus Gaussian jitter (sd = 5% of the range), floored
    at half of lo so the jitter can never make it negligible."""
    g = rng.uniform(lo, hi) + rng.gauss(0, 0.05 * (hi - lo))
    return max(lo / 2, g)


def classify_error(stderr, consecutive_403):
    """'bot' means stop the worker; 'error' is a per-video failure."""
    if BOT_CHECK.search(stderr or ""):
        return "bot"
    if FORBIDDEN.search(stderr or "") and consecutive_403 + 1 >= 3:
        return "bot"
    return "error"


def error_line(stderr):
    """The last ERROR line yt-dlp printed, verbatim (Puck wants the exact text)."""
    lines = (stderr or "").strip().splitlines()
    errors = [l for l in lines if l.startswith("ERROR") or "Error" in l]
    return (errors or lines or ["unknown error"])[-1].strip()


def ffprobe_duration(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def find_audio(out_dir, video_id):
    for p in sorted(Path(out_dir).glob(f"{video_id}.*")):
        if p.suffix not in (".json", ".part", ".ytdl") and not p.name.endswith(".part"):
            return p
    return None


def sponsorblock(video_id, timeout=10):
    """Free API, no key (PLAN.md 4.2). Returns [[start, end], ...], [] when the
    video has no entries (404), None when the call failed."""
    if os.environ.get("YTPT_OFFLINE"):
        return None
    q = urllib.parse.urlencode({"videoID": video_id, "categories": json.dumps(["sponsor", "selfpromo"])})
    req = urllib.request.Request(f"https://sponsor.ajay.app/api/skipSegments?{q}",
                                 headers={"User-Agent": "ytpt-downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return [s["segment"] for s in json.loads(r.read()) if "segment" in s]
    except urllib.error.HTTPError as e:
        return [] if e.code == 404 else None
    except Exception:  # noqa: BLE001 -- metadata nicety, never blocks a download
        return None


def record_metadata(con, video_id, out_dir, audio_path):
    """Exact fields from <id>.info.json; the flat-playlist catalog was approximate."""
    info_path = Path(out_dir) / f"{video_id}.info.json"
    fields = {"audio_path": str(audio_path)}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except ValueError:
            info = {}
        keep = {k: info.get(k) for k in ("id", "title", "upload_date", "timestamp", "duration", "view_count",
                                          "like_count", "comment_count", "channel", "channel_id", "uploader",
                                          "categories", "tags", "live_status", "was_live", "description",
                                          "availability", "age_limit", "language", "chapters", "release_timestamp")}
        fields["raw_meta"] = json.dumps(keep)
        if info.get("upload_date"):
            fields["upload_date"] = info["upload_date"]
            fields["month"] = f"{info['upload_date'][:4]}-{info['upload_date'][4:6]}"
            fields["date_exact"] = 1
        if info.get("duration"):
            fields["duration_s"] = float(info["duration"])
        if info.get("title"):
            fields["title"] = info["title"]
        for k in ("view_count", "like_count"):
            if info.get(k) is not None:
                fields[k] = int(info[k])
    sb = sponsorblock(video_id)
    if sb is not None:
        fields["sponsor_segments"] = json.dumps(sb)
    return fields


def download_one(con, video_id, out_dir, dry_run=False, limit_rate="3M", consecutive_403=0):
    """Returns ('ok'|'error'|'bot', message)."""
    cmd = ytdlp_command(video_id, out_dir, limit_rate)
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return "ok", "dry run"
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return "error", "yt-dlp timed out after 3600 s"
    if r.returncode != 0:
        kind = classify_error(r.stderr, consecutive_403)
        return kind, error_line(r.stderr)
    path = find_audio(out_dir, video_id)
    if not path:
        return "error", "yt-dlp exited 0 but no audio file was written"
    dur = ffprobe_duration(path)
    row = con.execute("SELECT duration_s FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    expect = float(row["duration_s"]) if row and row["duration_s"] else None
    if dur is None:
        return "error", f"ffprobe cannot read {path.name}"
    if expect and dur < 0.9 * expect:
        return "error", f"file is {dur:.0f} s but the catalog says {expect:.0f} s (truncated download?)"
    fields = record_metadata(con, video_id, out_dir, path)
    if not state.transition(con, video_id, "downloading", "downloaded",
                            note=f"{path.stat().st_size / 1e6:.1f} MB, {dur:.0f} s, {time.time() - t0:.0f} s wall",
                            **fields):
        return "error", "row was not in 'downloading' when the download finished"
    return "ok", f"{path.name} {path.stat().st_size / 1e6:.1f} MB, {dur / 60:.1f} min, {time.time() - t0:.0f} s"


def next_video(con):
    return con.execute("SELECT video_id, speaker FROM videos WHERE in_frame = 1 AND status = 'discovered' "
                       "ORDER BY upload_date DESC, video_id LIMIT 1").fetchone()


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def run_worker(args, stop, stats, lock, worker_id=0):
    con = state.connect(args.state)
    out_dir = audio_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random()
    consecutive_403 = 0
    while not stop.is_set():
        with lock:
            if args.max and stats["started"] >= args.max:
                break
            row = next_video(con)
            if row and not state.claim(con, row["video_id"], "discovered", "downloading"):
                continue                      # another worker took it; try the next one
            if row:
                stats["started"] += 1
        if not row:
            if not args.loop:
                break
            time.sleep(args.idle_sleep)
            continue
        vid = row["video_id"]
        log(f"[w{worker_id}] downloading {vid} ({row['speaker']})")
        kind, msg = download_one(con, vid, out_dir, args.dry_run, args.limit_rate, consecutive_403)
        if kind == "ok":
            consecutive_403 = 0
            with lock:
                stats["ok"] += 1
            log(f"[w{worker_id}] downloaded  {vid}: {msg}")
            if args.dry_run:
                state.transition(con, vid, "downloading", "discovered", note="dry run")
        elif kind == "bot":
            stop.set()
            with lock:
                stats["bot"] = msg
            attempts = con.execute("SELECT attempts FROM videos WHERE video_id = ?", (vid,)).fetchone()["attempts"]
            state.transition(con, vid, "downloading", "discovered", note="bot check / rate limit; not charged",
                             attempts=max(0, int(attempts) - 1), last_error=msg)
            log(f"[w{worker_id}] STOP: YouTube is challenging this IP on {vid}: {msg}")
            notify("error", f"downloader STOPPED on {vid}: {msg}. Not retrying; waiting for you. "
                            f"({stats['ok']} downloads succeeded this run)")
            break
        else:
            consecutive_403 += 1 if FORBIDDEN.search(msg) else 0
            new = state.fail(con, vid, "downloading", msg)
            with lock:
                stats["failed"] += 1
            log(f"[w{worker_id}] {new.upper()}  {vid}: {msg}")
            if new == "dead":
                notify("error", f"{vid} is dead after {state.MAX_ATTEMPTS} download attempts: {msg}")
        if stop.is_set() or (args.max and stats["started"] >= args.max):
            break
        gap = random_gap(args.min_gap, args.max_gap, rng)
        log(f"[w{worker_id}] sleeping {gap:.0f} s before the next download")
        if stop.wait(gap):
            break
    con.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=0, help="stop after this many downloads (0 = no limit)")
    ap.add_argument("--loop", action="store_true", help="keep polling when the queue is empty (launchd mode)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--min-gap", type=float, default=45, help="seconds; random pause lower bound")
    ap.add_argument("--max-gap", type=float, default=180, help="seconds; random pause upper bound")
    ap.add_argument("--idle-sleep", type=float, default=60)
    ap.add_argument("--limit-rate", default="3M")
    ap.add_argument("--state", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        ap.error(f"--concurrency must be 1..{MAX_CONCURRENCY} (PLAN.md 4.2: going wide gets the IP flagged)")
    if args.min_gap < 10 or args.max_gap < args.min_gap:
        ap.error("--min-gap must be >= 10 and <= --max-gap")

    con = state.connect(args.state)
    # in_frame = 1 only, mirroring next_video(): a kind='meeting' row that
    # episodes.py is fetching right now is also 'downloading' and is not ours.
    n = state.reclaim_stuck(con, "downloading", "discovered", in_frame=1)
    if n:
        log(f"reclaimed {n} row(s) left in 'downloading' by a previous run")
    con.close()

    stop, lock = threading.Event(), threading.Lock()
    stats = {"started": 0, "ok": 0, "failed": 0, "bot": None}
    threads = [threading.Thread(target=run_worker, args=(args, stop, stats, lock, i), daemon=True)
               for i in range(args.concurrency)]
    for t in threads:
        t.start()
        time.sleep(random.uniform(2, 8))       # workers do not start in lockstep
    try:
        for t in threads:
            while t.is_alive():
                t.join(1)
    except KeyboardInterrupt:
        stop.set()
        log("interrupted; letting yt-dlp finish the current file")
        for t in threads:
            t.join()
    log(f"done: {stats['ok']} downloaded, {stats['failed']} failed" + (f", STOPPED: {stats['bot']}" if stats["bot"] else ""))
    if stats["bot"] and args.loop:
        # launchd mode. The plist's KeepAlive (SuccessfulExit=false) restarts a
        # non-zero exit after ThrottleInterval, i.e. a bot check would make
        # launchd retry YouTube every 10 min and page Puck each time -- the
        # retry loop AGENTS.md rule 2 forbids. Exit 0 so launchd leaves the
        # worker DOWN until a human brings it back (`resume` / launchctl
        # kickstart); the WhatsApp message above already went out.
        log("STOPPED by a YouTube challenge in --loop mode: exiting 0 so launchd does not restart the worker; "
            "resume with `launchctl kickstart -k gui/$(id -u)/com.ytpt.downloader` when Puck says so")
        return 0
    return EXIT_BOT_CHECK if stats["bot"] else 0


if __name__ == "__main__":
    sys.exit(main())
