#!/usr/bin/env python3
"""
scripts/t0_register_clips.py -- T0 test 5 prep (plan/T0-TESTPLAN.md test 5):
register the 40 episode clips that test 4 fetched into output/t0/<video_id>/
e<n>.opus as in-frame 'transcoded' rows in state.sqlite, so that

    pipeline/gpu_batch.py propose --min-hours 0 --speaker sanjose,sunnyvale,sanrafael,santarosa,bart

can propose ONE batch over them (RTFX + diarization on 4-60 min clips). This
script never proposes, approves or launches anything; it only prepares rows.

Per clip (output/t0/fetch_report.json `clips[]`, joined with episodes.json
for the label, sample.json for the body and meeting title, and the meeting's
info.json for upload_date / channel):

  1. transcode to the pipeline's standard 16 kHz mono 16 kbps opus with the
     SAME ffmpeg command transcode.py uses (transcode.ffmpeg_command), into
     OPUS_DIR/<yt_id>_e<n>.opus; verified with transcode.verify. The `-t` cap
     is the clip's own length: an episode is the unit here, TRANSCRIBE_CAP_MIN
     (a per-video sampling rule) is not applied. The original clip under
     output/t0/ is kept.
  2. one `videos` row, video_id = <yt_id>_e<n>, speaker = body slug, walked
     through the real state machine with state.transition():
     discovered -> downloaded (audio_path = the t0 clip) -> transcoded
     (opus_path, transcribed_s). The row is inserted with in_frame = 0 and
     flipped to 1 by discover.mark_frame() only once it is 'transcoded': the
     launchd downloader/transcoder pick in-frame rows only, so they can never
     grab a half-registered one.

Idempotent: a video_id that already exists is left alone (reported as
'exists'). An opus file at the target path WITHOUT a row is a collision and
stops the script. Nothing is deleted.

    .venv/bin/python scripts/t0_register_clips.py --dry-run   # plan only, no ffmpeg, no DB writes
    .venv/bin/python scripts/t0_register_clips.py             # do it (~1-2 min of ffmpeg)
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import state, transcode, discover                       # noqa: E402

T0 = ROOT / "output" / "t0"
NOTE = "T0 test 5 episode clip"
BODY_TAB = "streams"                                     # every T0 pick came from the /streams tab (sample.json)


def hms(s):
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_plan():
    report = json.loads((T0 / "fetch_report.json").read_text(encoding="utf-8"))
    episodes = {e["episode_id"]: e for e in json.loads((T0 / "episodes.json").read_text(encoding="utf-8"))}
    sample = {s["video_id"]: s for s in json.loads((T0 / "sample.json").read_text(encoding="utf-8"))}
    infos = {}
    rows = []
    for c in report["clips"]:
        yt = c["video_id"]
        if yt not in infos:
            infos[yt] = json.loads((T0 / yt / f"{yt}.info.json").read_text(encoding="utf-8"))
        info, pick, ep = infos[yt], sample[yt], episodes[c["episode_id"]]
        n = int(c["episode_id"].rsplit("_e", 1)[1])
        start = float(c.get("actual_start_s", c["start_s"]))       # section clips carry a 10 s lead-in
        dur = float(c["duration_s"])
        end = start + dur
        rows.append({
            "video_id": c["episode_id"], "n": n, "yt_id": yt, "speaker": pick["body"],
            "channel_url": f"https://www.youtube.com/channel/{info['channel_id']}/{pick.get('tab', BODY_TAB)}",
            "month": discover.month_of(info.get("upload_date")),
            "title": f"{pick['title']} — episode {n} ({ep['label']}) {hms(start)}–{hms(end)}",
            "upload_date": info.get("upload_date"),
            "duration_s": round(dur, 2), "src": Path(c["path"]),
            "raw_meta": json.dumps({
                "t0": "test 5 episode clip", "source_video_id": yt, "episode_id": c["episode_id"],
                "label": ep["label"], "family_rates": ep.get("family_rates"),
                "window_start_s": round(start, 2), "window_end_s": round(end, 2),
                "core_start_s": c.get("core_start_s"), "core_end_s": c.get("core_end_s"),
                "strategy": c.get("strategy"), "channel": info.get("channel"),
                "channel_id": info.get("channel_id"), "meeting_duration_s": info.get("duration"),
                "meeting_title": info.get("title")}, ensure_ascii=False),
        })
    return rows


def register(con, r, out_dir, dry_run):
    dst = out_dir / f"{r['video_id']}.opus"
    row = con.execute("SELECT status, in_frame FROM videos WHERE video_id = ?", (r["video_id"],)).fetchone()
    if row:
        return f"exists ({row['status']}, in_frame={row['in_frame']})"
    if dst.exists():
        sys.exit(f"refusing: {dst} exists but no videos row {r['video_id']} -- collision, sort it out by hand")
    if not r["src"].exists():
        sys.exit(f"refusing: clip missing {r['src']}")
    if dry_run:
        return f"would transcode {r['src'].name} -> {dst.name} and register as {r['speaker']} {r['month']}"
    # 1. opus, exactly as transcode.py does it, cap = the clip itself
    tmp = out_dir / f"{r['video_id']}.tmp.opus"
    p = subprocess.run(transcode.ffmpeg_command(r["src"], tmp, r["duration_s"]), capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        sys.exit(f"ffmpeg failed on {r['src']}: {(p.stderr or '').strip()[-300:]}")
    ok, detail = transcode.verify(tmp, r["duration_s"])
    if not ok:
        tmp.unlink(missing_ok=True)
        sys.exit(f"verify failed on {tmp}: {detail}")
    tmp.replace(dst)
    # 2. the row, in_frame = 0 until it is safely 'transcoded'
    now = state.now_iso()
    con.execute("BEGIN IMMEDIATE")
    con.execute(
        "INSERT INTO videos (video_id, speaker, channel_url, month, title, upload_date, date_exact, duration_s, "
        "raw_meta, crawled_at, in_frame, status, attempts, updated_at, sponsor_segments) "
        "VALUES (?,?,?,?,?,?,1,?,?,?,0,'discovered',0,?,'[]')",
        (r["video_id"], r["speaker"], r["channel_url"], r["month"], r["title"], r["upload_date"],
         r["duration_s"], r["raw_meta"], now, now))
    con.execute("INSERT INTO events (video_id, from_status, to_status, at, note) VALUES (?,?,?,?,?)",
                (r["video_id"], None, "discovered", now, NOTE))
    con.execute("COMMIT")
    if not state.transition(con, r["video_id"], "discovered", "downloaded", note="T0 test 5",
                            audio_path=str(r["src"])):
        sys.exit(f"{r['video_id']}: could not move discovered -> downloaded")
    if not state.transition(con, r["video_id"], "downloaded", "transcoded", note=f"T0 test 5: {detail}",
                            opus_path=str(dst), transcribed_s=r["duration_s"]):
        sys.exit(f"{r['video_id']}: could not move downloaded -> transcoded")
    discover.mark_frame(con, [r["video_id"]], NOTE)
    return f"transcoded {detail}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    rows = load_plan()
    con = state.connect(a.state)
    out_dir = transcode.opus_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} clips, {sum(r['duration_s'] for r in rows) / 3600:.2f} audio-h -> {out_dir}, DB {state.db_path()}"
          + (" (dry run)" if a.dry_run else ""))
    for r in rows:
        print(f"  {r['video_id']:<18} {r['speaker']:<9} {r['month']} {r['duration_s']:7.1f} s  {register(con, r, out_dir, a.dry_run)}",
              flush=True)
    n = con.execute("SELECT COUNT(*) FROM videos WHERE in_frame = 1 AND status = 'transcoded' AND speaker IN "
                    "('sanjose','sunnyvale','sanrafael','santarosa','bart')").fetchone()[0]
    print(f"in-frame 'transcoded' rows for the five bodies: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
