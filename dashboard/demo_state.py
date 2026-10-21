#!/usr/bin/env python3
"""
dashboard/demo_state.py -- build a synthetic state.sqlite shaped like plan/SCHEMA.md.

WHY: the dashboard, the WhatsApp status digest and their tests have to work
before the real pipeline exists, and a monitoring page nobody has seen with
realistic data is a page nobody trusts. Everything here is fake and labelled
DEMO; nothing reads it except --demo and the tests.

    python3 dashboard/demo_state.py /tmp/demo_state.sqlite
"""
import random, sqlite3, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Keep in step with plan/SCHEMA.md 1 (videos, spend, pods) plus the 2026-09-17
# additions (events, batches). The dashboard tolerates missing tables/columns.
DDL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS videos (
  video_id TEXT PRIMARY KEY, speaker TEXT NOT NULL, channel_url TEXT NOT NULL,
  month TEXT NOT NULL, title TEXT, upload_date TEXT, date_exact INTEGER DEFAULT 0,
  duration_s REAL, view_count INTEGER, like_count INTEGER, raw_meta TEXT,
  crawled_at TEXT NOT NULL, in_frame INTEGER DEFAULT 0, status TEXT NOT NULL,
  audio_path TEXT, opus_path TEXT, transcribed_s REAL, attempts INTEGER DEFAULT 0,
  last_error TEXT, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY, video_id TEXT NOT NULL, from_status TEXT,
  to_status TEXT NOT NULL, at TEXT NOT NULL, note TEXT);
CREATE TABLE IF NOT EXISTS spend (
  id INTEGER PRIMARY KEY, pod_id TEXT, started_at TEXT NOT NULL, ended_at TEXT,
  gpu_hours REAL, usd REAL, n_videos INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS pods (
  pod_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, terminated_at TEXT, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS batches (
  batch_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
  n_videos INTEGER, audio_hours REAL, est_usd REAL, approval_requested_at TEXT,
  approved_at TEXT, approved_via TEXT, pod_id TEXT, started_at TEXT,
  finished_at TEXT, gpu_minutes REAL, actual_usd REAL, note TEXT);
"""

SPEAKERS = {"hasanabi": (900, 5400), "benshapiro": (300, 1500), "tucker": (1800, 6000)}
CAP_S = 45 * 60


def iso(t):
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def build(path, seed=20260917, now=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()
    now = now or datetime.now(timezone.utc).replace(microsecond=0)
    rng = random.Random(seed)
    con = sqlite3.connect(path)
    con.executescript(DDL)
    start = now - timedelta(days=24)
    months = [f"2025-{m:02d}" for m in range(7, 13)] + [f"2026-{m:02d}" for m in range(1, 7)]
    order = ["discovered", "downloading", "downloaded", "transcoded", "queued", "transcribing", "done"]
    vids, n = [], 0
    for sp, (lo, hi) in SPEAKERS.items():
        for mi, month in enumerate(months):
            got = 0.0
            while got < 3 * 3600:
                dur = rng.uniform(lo, hi)
                got += min(dur, CAP_S)
                vid = f"DEMO{sp[:2]}{mi:02d}{n:04d}"
                n += 1
                # older months have progressed further through the pipeline
                progress = (len(months) - mi) / len(months) + rng.uniform(-0.25, 0.25)
                stage = max(0, min(6, int(progress * 7.4)))
                if stage == 5:            # nothing is on a GPU in the demo: b3 awaits approval
                    stage = 4
                status = order[stage]
                attempts, err = 0, None
                if rng.random() < 0.03:
                    status, attempts, err = "failed", 1, "HTTP Error 403: Forbidden (retrying)"
                elif rng.random() < 0.01:
                    status, attempts, err = "dead", 3, "Video unavailable: removed by uploader"
                vids.append((vid, sp, month, dur, status, attempts, err, stage))
    t_dl = {}
    for vid, sp, month, dur, status, attempts, err, stage in vids:
        dl_at = start + timedelta(seconds=rng.uniform(0, 24 * 86400 - 3600))
        t_dl[vid] = dl_at
        done = status == "done"
        con.execute(
            "INSERT INTO videos VALUES (?,?,?,?,?,?,0,?,?,NULL,NULL,?,1,?,NULL,NULL,?,?,?,?)",
            (vid, sp, f"https://www.youtube.com/@{sp}/videos", month, f"demo video {vid}",
             month.replace("-", "") + "15", dur, rng.randint(1000, 900000), iso(start),
             status, min(dur, CAP_S) if done else None, attempts, err,
             iso(dl_at + timedelta(hours=rng.uniform(0, 30)) if stage >= 2 else start)))
        if stage >= 2 and status not in ("failed", "dead"):
            con.execute("INSERT INTO events (video_id, from_status, to_status, at) VALUES (?,?,?,?)",
                        (vid, "downloading", "downloaded", iso(dl_at)))
    done = [v for v in vids if v[4] == "done"]
    rng.shuffle(done)
    b_times = [start + timedelta(days=9), start + timedelta(days=17)]
    per = len(done) // 2
    for bi, (bt, chunk) in enumerate(zip(b_times, (done[:per], done[per:])), 1):
        hours = sum(min(v[3], CAP_S) for v in chunk) / 3600
        gpu_min = hours * 60 / 27.5
        usd = gpu_min / 60 * 0.34 + 0.05
        pod = f"demo-pod-{bi}"
        con.execute("INSERT INTO batches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"b{bi}", iso(bt - timedelta(hours=1)), "done", len(chunk), round(hours, 2),
                     round(hours / 25 * 0.34, 2), iso(bt - timedelta(minutes=50)),
                     iso(bt - timedelta(minutes=41)), "whatsapp", pod, iso(bt),
                     iso(bt + timedelta(minutes=gpu_min + 7)), round(gpu_min + 7, 1), round(usd, 2),
                     "demo"))
        con.execute("INSERT INTO pods VALUES (?,?,?,?)",
                    (pod, iso(bt), iso(bt + timedelta(minutes=gpu_min + 8)), "terminated"))
        con.execute("INSERT INTO spend (pod_id, started_at, ended_at, gpu_hours, usd, n_videos, note) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (pod, iso(bt), iso(bt + timedelta(minutes=gpu_min + 8)),
                     round((gpu_min + 8) / 60, 3), round(usd, 2), len(chunk), "demo"))
        for v in chunk:
            at = bt + timedelta(minutes=rng.uniform(5, gpu_min + 6))
            con.execute("INSERT INTO events (video_id, from_status, to_status, at) VALUES (?,?,?,?)",
                        (v[0], "transcribing", "done", iso(at)))
    ready = [v for v in vids if v[4] in ("transcoded", "queued")]
    hours = sum(min(v[3], CAP_S) for v in ready) / 3600
    con.execute("INSERT INTO batches (batch_id, created_at, status, n_videos, audio_hours, est_usd, "
                "approval_requested_at, note) VALUES (?,?,?,?,?,?,?,?)",
                ("b3", iso(now - timedelta(minutes=35)), "awaiting_approval", len(ready),
                 round(hours, 2), round(hours / 25 * 0.34, 2), iso(now - timedelta(minutes=34)), "demo"))
    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_state.sqlite")
    build(out)
    print(f"wrote demo state DB -> {out}")
