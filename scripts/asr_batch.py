#!/usr/bin/env python3
"""
scripts/asr_batch.py -- the bulk local-ASR locate run (plan/T2-PLAN.md 3b item 4
and 3c item 3; built 2026-09-23 after Puck approved the 29 Mac-hour run).

scripts/local_asr.py does ONE meeting. This drives it over a body's whole
backlog, one meeting at a time, resumably, and classifies each track as soon as
it exists so episodes appear while the run is still going:

    meeting -> local_asr.py --register   (Granicus audio -> wav -> VTT,
                                          captions='asr_local', row -> 'captioned')
            -> classify.py run --bodies  (segment.py windows -> Sonnet -> episodes
                                          rows with method='local_asr')

It is a DRIVER, not a second instrument: every measurement decision lives in
those two scripts. What it owns is the loop, the budget and the stop rules.

  resumable   a meeting whose row already says captions='asr_local' is skipped,
              so a killed run resumes by being restarted with the same command.
  free ASR    the transcription costs nothing but Mac time (~11.6x realtime).
              The CLASSIFIER is not free: --claude-stop-usd is an absolute
              ceiling on the whole Claude ledger (pipeline/spend.py), checked
              before each meeting. Past it the ASR keeps running and the
              classify step is skipped -- the tracks are on disk and
              `classify.py run --bodies <bodies>` finishes them later for real
              money, deliberately as a separate decision.
  politeness  Granicus rules (AGENTS.md): >= 5 s between requests -- each
              meeting is 2 requests (ASX.php, then the MP4) and 20 minutes
              apart in practice -- audio only, one ffmpeg at a time. A 403 or
              429 from the archive host STOPS the run; it never retries.
  disk        stops when free space falls below --min-free-gb. ~116 MB per
              audio-hour of wav is kept (nothing under output/ is deleted);
              pass --drop-audio to keep only the VTT.

    .venv/bin/python scripts/asr_batch.py --bodies richmond,sanleandro --plan
    nohup .venv/bin/python scripts/asr_batch.py --bodies richmond,sanleandro \\
        --claude-stop-usd 6.00 > output/t2/asr_batch.log 2>&1 &
    .venv/bin/python scripts/asr_batch.py --status

Runs in the PINNED venv; it shells out to the scratch ASR venv
(~/.local/share/ytpt/venv-asr) for mlx-whisper and back to the pinned one for
the classifier, so neither environment has to hold both.
"""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))
import state    # noqa: E402
import spend    # noqa: E402
import segment as sg       # noqa: E402
import caption_audit       # noqa: E402

ASR_PYTHON = os.environ.get("ASR_PYTHON", str(Path.home() / ".local/share/ytpt/venv-asr/bin/python"))
PINNED_PYTHON = os.environ.get("PINNED_PYTHON", str(ROOT / ".venv" / "bin" / "python"))
OUT = ROOT / "output" / "t2" / "asr"
PROGRESS = OUT / "progress.json"
ASR_CAPTIONS = "asr_local"
# the archive host refusing: stop the run, exactly like the YouTube bot check
BLOCKED = re.compile(r"\b(403|429)\b|Forbidden|Too Many Requests", re.I)


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def free_gb(path=ROOT):
    return shutil.disk_usage(path).free / 1e9


def dir_gb(path=OUT):
    if not Path(path).exists():
        return 0.0
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e9


def usable_caption_track(row):
    """Does this meeting already have a REAL caption track that reads as a
    transcript of the whole meeting (scripts/caption_audit.py)?

    ASR-ing such a meeting is not merely wasted Mac time: local_asr.py
    --register overwrites `captions` with 'asr_local', so a good Granicus or
    YouTube track would be replaced by a deliberately low-quality locate track
    and the meeting's method would silently change instrument. A track the
    audit FAILS -- truncated, stops early, Spanish where the keywords are
    English -- is exactly what the ASR route is for, so those are kept in."""
    caps = row.get("captions")
    if not caps or caps in ("none", ASR_CAPTIONS):
        return False
    p = row.get("captions_path")
    if not p or not Path(p).exists():
        return False
    return not caption_audit.flags(caption_audit.audit_track(sg.read_segments(p), row.get("duration_s")))


def select(con, bodies, order="date", limit=0, redo_captioned=False):
    """Meetings of these bodies that have no local ASR track yet, newest first
    (--order short: shortest first, which finishes more meetings per hour).
    Meetings that already have a caption track the audit passes are skipped
    (usable_caption_track); --redo-captioned takes them anyway."""
    where = "kind = 'meeting'"
    params = []
    if bodies:
        where += f" AND speaker IN ({','.join('?' * len(bodies))})"
        params += bodies
    where += " AND (captions IS NULL OR captions != ?)"
    params.append(ASR_CAPTIONS)
    by = "COALESCE(duration_s, 1e9) ASC, video_id" if order == "short" else "meeting_date DESC, video_id"
    rows = [dict(r) for r in con.execute(
        f"SELECT * FROM videos WHERE {where} ORDER BY {by}", params)]
    if not redo_captioned:
        rows = [r for r in rows if not usable_caption_track(r)]
    return rows[:limit] if limit else rows


def already_done(con, bodies):
    q = f"SELECT COUNT(*) n, COALESCE(SUM(duration_s), 0) s FROM videos WHERE kind = 'meeting' AND captions = ? AND speaker IN ({','.join('?' * len(bodies))})"
    r = con.execute(q, [ASR_CAPTIONS, *bodies]).fetchone()
    return r["n"], r["s"] / 3600.0


def run_asr(video_id, drop_audio, timeout):
    """local_asr.py in the scratch venv. Returns (ok, blocked, tail)."""
    cmd = [ASR_PYTHON, str(ROOT / "scripts" / "local_asr.py"), video_id, "--register"]
    if drop_audio:
        cmd.append("--drop-audio")
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, False, f"local_asr.py timed out after {timeout} s"
    out = (r.stdout or "") + (r.stderr or "")
    tail = " / ".join(out.strip().splitlines()[-3:])[:300]
    return r.returncode == 0, bool(BLOCKED.search(out)), tail


def run_classify(bodies, cap_usd, timeout):
    """classify.py in the pinned venv over these bodies: every 'captioned' row,
    which at this point is the ASR tracks registered so far. Cached windows
    cost nothing, so re-running per meeting is cheap.

    --include-classified because a meeting another instrument has advanced
    ('classified' by agenda_index, 'cut' by the cutter) keeps its status when
    local_asr.py registers the track on it; without this the track would sit
    on disk unread. finish_status still decides what the row becomes."""
    cmd = [PINNED_PYTHON, str(ROOT / "pipeline" / "classify.py"), "run", "--bodies", ",".join(bodies),
           "--include-classified", "--cap-usd", f"{cap_usd:.4f}"]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"classify.py timed out after {timeout} s"
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0, " / ".join(out.strip().splitlines()[-2:])[:300]


def write_progress(d):
    OUT.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(d, indent=1), encoding="utf-8")


def show_status(con):
    if not PROGRESS.exists():
        print("no run recorded yet")
        return 0
    d = json.loads(PROGRESS.read_text(encoding="utf-8"))
    print(json.dumps(d, indent=1))
    print(f"\nClaude ledger now ${spend.claude_total(con):.4f}; asr dir {dir_gb():.1f} GB; free {free_gb():.0f} GB")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", default="richmond,sanleandro")
    ap.add_argument("--max", type=int, default=0, help="at most this many meetings, then exit")
    ap.add_argument("--order", choices=["date", "short"], default="date")
    ap.add_argument("--redo-captioned", action="store_true",
                    help="also ASR meetings that already have a caption track the audit passes (it will be REPLACED)")
    ap.add_argument("--claude-stop-usd", type=float, default=None,
                    help="absolute ceiling on the Claude ledger; past it the ASR continues and classification is skipped")
    ap.add_argument("--cap-usd", type=float, default=0.50, help="classify.py --cap-usd for ONE meeting")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--drop-audio", action="store_true", help="keep only the VTT, not the working wav")
    ap.add_argument("--gap", type=float, default=10.0, help="seconds between meetings (Granicus politeness)")
    ap.add_argument("--asr-timeout", type=float, default=6 * 3600)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)

    con = state.connect(a.state)
    if a.status:
        return show_status(con)
    bodies = [b.strip() for b in a.bodies.split(",") if b.strip()]
    rows = select(con, bodies, a.order, a.max, a.redo_captioned)
    done_n, done_h = already_done(con, bodies)
    total_h = sum((r["duration_s"] or 0) for r in rows) / 3600.0
    log(f"{len(rows)} meeting(s) to ASR over {', '.join(bodies)}: {total_h:.1f} audio-h "
        f"= {total_h / 11.6:.1f} Mac-h at the measured 11.6x; {done_n} already done ({done_h:.1f} h); "
        f"free disk {free_gb():.0f} GB; Claude ledger ${spend.claude_total(con):.4f}"
        + (f", stop at ${a.claude_stop_usd:.2f}" if a.claude_stop_usd else ""))
    if a.plan:
        for r in rows:
            log(f"  {r['video_id']} ({r['speaker']}, {r['meeting_date']}) {(r['duration_s'] or 0) / 3600:.2f} h  [{r['status']}]")
        return 0

    t_start = time.time()
    st = {"started": state.now_iso(), "bodies": bodies, "total": len(rows), "done": 0, "failed": 0,
          "audio_h": 0.0, "asr_min": 0.0, "classified": 0, "skipped_classify": 0, "stopped": None,
          "resume": f"nohup {PINNED_PYTHON} scripts/asr_batch.py --bodies {a.bodies}"
                    + (f" --claude-stop-usd {a.claude_stop_usd}" if a.claude_stop_usd else "")
                    + " > output/t2/asr_batch.log 2>&1 &"}
    write_progress(st)

    for i, m in enumerate(rows, 1):
        vid, hrs = m["video_id"], (m["duration_s"] or 0) / 3600.0
        if free_gb() < a.min_free_gb:
            st["stopped"] = f"free disk {free_gb():.0f} GB below --min-free-gb {a.min_free_gb:.0f}"
            log(f"STOP: {st['stopped']}")
            break
        log(f"[{i}/{len(rows)}] {vid} ({m['speaker']}, {m['meeting_date']}, {hrs:.2f} h) -> ASR")
        t0 = time.time()
        ok, blocked, tail = run_asr(vid, a.drop_audio, a.asr_timeout)
        wall = (time.time() - t0) / 60
        if blocked:
            st["stopped"] = f"archive host refused {vid}: {tail}"
            log(f"STOP: the Granicus archive host is refusing audio on {vid}: {tail}")
            break
        if not ok:
            st["failed"] += 1
            log(f"  FAILED {vid} after {wall:.1f} min: {tail}")
            write_progress(st)
            time.sleep(a.gap)
            continue
        st["done"] += 1; st["audio_h"] += hrs; st["asr_min"] += wall
        log(f"  ASR done in {wall:.1f} min ({hrs / (wall / 60) if wall else 0:.1f}x realtime); {tail}")

        ledger = spend.claude_total(con)
        if a.claude_stop_usd is not None and ledger >= a.claude_stop_usd:
            st["skipped_classify"] += 1
            log(f"  classify SKIPPED: Claude ledger ${ledger:.4f} has reached --claude-stop-usd ${a.claude_stop_usd:.2f}; "
                f"the track is on disk, `classify.py run --bodies {a.bodies}` finishes it later")
        else:
            cok, ctail = run_classify(bodies, a.cap_usd, timeout=3600)
            st["classified"] += 1 if cok else 0
            log(f"  classify {'ok' if cok else 'FAILED'}: {ctail}")

        eps = con.execute("SELECT COUNT(*) n, COALESCE(SUM(end_s - start_s), 0) s FROM episodes e JOIN videos v USING (video_id) "
                          f"WHERE e.method = 'local_asr' AND v.speaker IN ({','.join('?' * len(bodies))})", bodies).fetchone()
        st.update({"episodes": eps["n"], "episode_h": round(eps["s"] / 3600.0, 2),
                   "claude_usd": spend.claude_total(con), "asr_dir_gb": round(dir_gb(), 1),
                   "free_gb": round(free_gb(), 1), "elapsed_h": round((time.time() - t_start) / 3600, 2),
                   "last": vid, "at": state.now_iso()})
        write_progress(st)
        if i % 10 == 0 or i == len(rows):
            log(f"PROGRESS {st['done']}/{len(rows)} done, {st['failed']} failed, {st['audio_h']:.1f} audio-h transcribed, "
                f"{st['episodes']} local_asr episodes ({st['episode_h']} h), ${st['claude_usd']:.4f} Claude, "
                f"{st['asr_dir_gb']} GB in {OUT.name}/, {st['elapsed_h']:.1f} h elapsed")
        time.sleep(a.gap)

    st["finished"] = state.now_iso()
    write_progress(st)
    log(f"done: {st['done']}/{len(rows)} meetings, {st['failed']} failed, {st['audio_h']:.1f} audio-h, "
        f"{st.get('episodes', 0)} local_asr episodes, ${spend.claude_total(con):.4f} Claude ledger, "
        f"{dir_gb():.1f} GB on disk" + (f"; STOPPED: {st['stopped']}" if st["stopped"] else ""))
    con.close()
    return 3 if st["stopped"] else 0


if __name__ == "__main__":
    sys.exit(main())
