#!/usr/bin/env python3
"""
scripts/renormalise.py -- re-encode clips that are already cut, through the
level-aware encode path (pipeline/transcode.py "LEVEL"), and put them back in
the queue. Built 2026-09-23 for b17's 30 Fremont QA failures.

It does NOT re-cut and it does NOT touch YouTube or Granicus: it reads the .m4a
already under output/audio/ and writes a new output/opus/<id>.opus. The previous
opus is moved to output/opus/pre_normalise/ first, so the change is reversible
and nothing that took network time is destroyed. The .m4a is never written.

    ... --batch b17 --status failed --plan     # measure and print, write nothing
    ... --batch b17 --status failed            # re-encode, then requeue
    ... --ids gr-fremont-13_e1 gr-fremont-13_e2

A row is only requeued when the new file passes transcode.verify, and only from
'failed' -- a 'done' clip is left alone unless --redo-done is given, because
re-encoding one means throwing away a transcript that was already paid for.
"""
import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import state       # noqa: E402
import transcode   # noqa: E402

BACKUP_DIR = "pre_normalise"
NOTE = "re-encoded through the level-aware path (scripts/renormalise.py)"


def log(msg):
    print(f"{state.now_iso()} {msg}", flush=True)


def select(con, args):
    where, params = ["kind = 'episode'"], []
    if args.batch:
        where.append("video_id IN (SELECT video_id FROM batch_videos WHERE batch_id = ?)")
        params.append(args.batch)
    if args.ids:
        where.append(f"video_id IN ({','.join('?' * len(args.ids))})")
        params += args.ids
    if args.bodies:
        bodies = [b.strip() for b in args.bodies.split(",") if b.strip()]
        where.append(f"speaker IN ({','.join('?' * len(bodies))})")
        params += bodies
    if args.status:
        where.append("status = ?")
        params.append(args.status)
    rows = [dict(r) for r in con.execute(
        f"SELECT video_id, speaker, status, audio_path, opus_path, transcribed_s, episode_id, raw_meta "
        f"FROM videos WHERE {' AND '.join(where)} ORDER BY video_id", params)]
    if not args.redo_done:
        rows = [r for r in rows if r["status"] != "done"]
    return rows


def record_levels(con, row, levels):
    """The measurement goes on the clip's own videos row -- kind='episode', so
    this IS the episode's row -- under raw_meta.audio_levels. No new column
    (plan/SCHEMA.md 'audio_levels')."""
    try:
        meta = json.loads(row["raw_meta"] or "{}")
    except ValueError:
        meta = {}
    meta["audio_levels"] = levels
    con.execute("UPDATE videos SET raw_meta = ? WHERE video_id = ?",
                (json.dumps(meta, ensure_ascii=False), row["video_id"]))


def one(con, row, plan_only):
    vid = row["video_id"]
    src = Path(row["audio_path"] or "")
    if not src.exists():
        return "SKIP", f"{vid}: source audio missing ({src})"
    dur = float(row["transcribed_s"] or 0) or transcode.probe(src)["duration"]
    if plan_only:
        p = transcode.encode_plan(src, float("inf"))
        act = [p["downmix"]] if p["af"] and "pan=" in p["af"] else []
        if p["normalised"]:
            act.append(f"loudnorm I={p['target_lufs']:.0f}")
        return "PLAN", (f"{vid}: mean {p['pre_mean_dbfs']:.1f} dB max {p['pre_max_dbfs']:.1f} dB -> "
                        f"{'; '.join(act) or 'unchanged'}")
    dst = Path(row["opus_path"] or (transcode.opus_dir() / f"{vid}.opus"))
    tmp = dst.with_suffix(".tmp.opus")
    ok, detail, levels = transcode.encode(src, tmp, float("inf"))
    if not ok:
        tmp.unlink(missing_ok=True)
        return "FAIL", f"{vid}: ffmpeg: {detail}"
    ok, vdetail = transcode.verify(tmp, dur)
    if not ok:
        tmp.unlink(missing_ok=True)
        return "FAIL", f"{vid}: verify: {vdetail}"
    if dst.exists():                       # keep the old encode; nothing is deleted
        keep = dst.parent / BACKUP_DIR
        keep.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst), str(keep / dst.name))
    tmp.replace(dst)
    record_levels(con, row, levels)
    return "OK", f"{vid}: {vdetail}, {detail}"


def requeue(con, row):
    """failed -> transcoded (the new opus is on disk), and the episodes row with
    it, so gpu_batch.propose can see the clip again."""
    if row["status"] != "failed":
        return None
    if not state.transition(con, row["video_id"], "failed", "transcoded", note=NOTE,
                            attempts=0, last_error=None):
        return f"{row['video_id']}: state changed underneath; not requeued"
    ep = row["episode_id"] or row["video_id"]
    if con.execute("SELECT 1 FROM episodes WHERE episode_id = ?", (ep,)).fetchone():
        state.set_episode(con, ep, "cut", note=NOTE, last_error=None)
    return f"{row['video_id']}: failed -> transcoded"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", help="clips of this GPU batch")
    ap.add_argument("--bodies", help="comma-separated speakers")
    ap.add_argument("--ids", nargs="+", help="explicit clip ids")
    ap.add_argument("--status", help="only rows in this status (e.g. failed)")
    ap.add_argument("--redo-done", action="store_true", help="include 'done' clips (throws away a paid transcript)")
    ap.add_argument("--plan", action="store_true", help="measure and print; write nothing")
    ap.add_argument("--no-requeue", action="store_true", help="re-encode but leave the row 'failed'")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    if not (a.batch or a.ids or a.bodies):
        ap.error("give --batch, --bodies or --ids")
    con = state.connect(a.state)
    rows = select(con, a)
    log(f"{len(rows)} clip(s) selected" + (" (plan only)" if a.plan else ""))
    counts, requeued = {}, 0
    for r in rows:
        kind, msg = one(con, r, a.plan)
        counts[kind] = counts.get(kind, 0) + 1
        log(f"  {kind} {msg}")
        if kind == "OK" and not a.no_requeue:
            m = requeue(con, r)
            if m:
                requeued += 1
                log(f"       {m}")
    con.close()
    log("done: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        + (f", {requeued} requeued" if requeued else ""))
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
