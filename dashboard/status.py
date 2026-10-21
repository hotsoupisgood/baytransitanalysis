#!/usr/bin/env python3
"""
dashboard/status.py -- ONE function computes pipeline status; the web dashboard
and the WhatsApp "status" reply both use it, so they can never disagree.

Reads, never writes:
  state.sqlite          (STATE_DB_PATH; plan/SCHEMA.md 1) opened read-only
  plan/frame_cells.csv  per-cell quota targets written by build_frame.py
  QA CSV                latest analysis/qa_transcripts.py --csv output (QA_CSV);
                        its mtime is when the gate last ran, and is reported

Every table and column is optional: a missing one shows up as "no data", never
as a crash, because the dashboard must work on day 1 (empty DB) and while the
schema is still growing.

    python3 dashboard/status.py            # WhatsApp-sized text digest
    python3 dashboard/status.py --json     # the full structure the page renders
"""
import argparse, csv, json, os, re, sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STAGE_GROUPS = [   # ordered; the page draws them as one stacked bar
    ("to download", ("discovered", "downloading")),
    ("on laptop", ("downloaded", "transcoded", "queued")),
    ("on GPU", ("transcribing",)),
    ("done", ("done",)),
]
DOWNLOADED_OR_LATER = ("downloaded", "transcoded", "queued", "transcribing", "done")
READY_FOR_GPU = ("transcoded", "queued")


def load_env(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), re.sub(r"\s+#.*$", "", v.strip()).strip('"').strip("'"))


def settings():
    load_env()
    f = lambda k, d: float(os.environ.get(k) or d)
    return {
        "state_db": os.path.expanduser(os.environ.get("STATE_DB_PATH") or "~/.local/share/ytpt/state.sqlite"),
        "frame_cells": os.environ.get("FRAME_CELLS_CSV") or str(ROOT / "plan" / "frame_cells.csv"),
        "qa_csv": os.environ.get("QA_CSV") or str(ROOT / "output" / "qa" / "latest.csv"),
        "spend_cap_usd": f("MONTHLY_SPEND_CAP_USD", 150),
        "batch_target_hours": f("BATCH_TARGET_HOURS", 15),
        "cap_min": f("TRANSCRIBE_CAP_MIN", 45),
        "stall_hours": f("STALL_HOURS", 6),
        "stale_batch_hours": f("STALE_BATCH_HOURS", 13),   # 12-h job cap (gpu_job.job_timeout_min) + boot
        "silent_days": f("BODY_SILENT_DAYS", 60),
        "port": int(f("DASHBOARD_PORT", 8765)),
    }


def _parse_ts(s):
    if not s:
        return None
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _read_csv(path):
    p = Path(path)
    if not p.exists():
        return None
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _stamp(t):
    """Absolute, to the minute. Deliberately NOT an elapsed age: this string
    goes into alert text, and alert_key() only normalises away elapsed time --
    anything else that moves between heartbeats re-fires the alert."""
    return t.strftime("%Y-%m-%d %H:%M UTC")


def collect(state_db=None, frame_cells=None, qa_csv=None, now=None, cfg=None):
    cfg = dict(cfg or settings())
    state_db = str(state_db or cfg["state_db"])
    frame_cells = frame_cells or cfg["frame_cells"]
    qa_csv = qa_csv or cfg["qa_csv"]
    now = now or datetime.now(timezone.utc)
    cap_s = cfg["cap_min"] * 60
    out = {
        "generated_at": now.isoformat(timespec="seconds"),
        "settings": {k: cfg[k] for k in ("spend_cap_usd", "batch_target_hours", "cap_min", "stall_hours")},
        "state_db": {"path": state_db, "exists": Path(state_db).exists(), "tables": []},
        "alerts": [], "totals": {}, "stages": [], "status_counts": {}, "cells": [],
        "timeline": {"days": [], "downloaded_h": [], "done_h": [], "approximate": False},
        "batches": [], "errors": [], "qa": None, "spend": {}, "pods_active": [],
        "meetings": {"bodies": [], "totals": {}},          # 2026-09-21: the meeting corpus (PROPOSAL-transit.md 3i)
    }
    alert = lambda level, kind, text: out["alerts"].append({"level": level, "kind": kind, "text": text})

    if not out["state_db"]["exists"]:
        alert("info", "no_state", f"No state database yet at {state_db}. It appears when discovery first runs.")
        _add_qa(out, qa_csv, alert, now)
        return out

    con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    out["state_db"]["tables"] = sorted(tables)
    cols = lambda t: {r[1] for r in con.execute(f"PRAGMA table_info({t})")} if t in tables else set()

    videos = []
    if "videos" in tables:
        vc = cols("videos")
        where = "WHERE in_frame = 1" if "in_frame" in vc else ""
        sel = ", ".join(c if c in vc else f"NULL AS {c}" for c in
                        ("video_id", "speaker", "month", "duration_s", "transcribed_s",
                         "status", "attempts", "last_error", "updated_at"))
        videos = [dict(r) for r in con.execute(f"SELECT {sel} FROM videos {where}")]

    capped = lambda v: min(float(v["duration_s"] or 0), cap_s)
    counts = defaultdict(int)
    for v in videos:
        counts[v["status"]] += 1
    out["status_counts"] = dict(sorted(counts.items()))
    for label, members in STAGE_GROUPS:
        sel = [v for v in videos if v["status"] in members]
        out["stages"].append({"label": label, "videos": len(sel),
                              "hours": round(sum(capped(v) for v in sel) / 3600, 2)})

    done_h = sum(float(v["transcribed_s"] or capped(v)) for v in videos if v["status"] == "done") / 3600
    dl_h = sum(capped(v) for v in videos if v["status"] in DOWNLOADED_OR_LATER) / 3600
    ready_h = sum(capped(v) for v in videos if v["status"] in READY_FOR_GPU) / 3600
    frame_h = sum(capped(v) for v in videos) / 3600

    # Quota cells: planned hours per speaker x period from frame_cells.csv when it
    # exists (the design), otherwise the sum over in-frame videos (what is known).
    plan = {}
    for row in _read_csv(frame_cells) or []:
        if row.get("status") != "dropped_period":
            plan[(row["speaker"], row["period"])] = float(row.get("sampled_capped_h") or 0)
    agg = defaultdict(lambda: {"videos": 0, "target_h": 0.0, "downloaded_h": 0.0, "done_h": 0.0})
    for v in videos:
        c = agg[(v["speaker"], v["month"])]
        c["videos"] += 1
        c["target_h"] += capped(v) / 3600
        if v["status"] in DOWNLOADED_OR_LATER:
            c["downloaded_h"] += capped(v) / 3600
        if v["status"] == "done":
            c["done_h"] += float(v["transcribed_s"] or capped(v)) / 3600
    for key in set(plan) | set(agg):
        c = agg[key]
        target = plan.get(key, c["target_h"])
        out["cells"].append({"speaker": key[0], "period": key[1], "videos": c["videos"],
                             "target_h": round(target, 2), "downloaded_h": round(c["downloaded_h"], 2),
                             "done_h": round(c["done_h"], 2),
                             "done_frac": round(min(c["done_h"] / target, 1.0), 3) if target else None})
    out["cells"].sort(key=lambda c: (c["speaker"], c["period"]))
    if plan:
        frame_h = sum(plan.values())

    out["totals"] = {
        "frame_videos": len(videos), "frame_hours": round(frame_h, 2),
        "downloaded_hours": round(dl_h, 2), "done_hours": round(done_h, 2),
        "done_frac": round(done_h / frame_h, 4) if frame_h else 0.0,
        "ready_for_gpu_hours": round(ready_h, 2),
        "cells_total": len(out["cells"]),
        "cells_full": sum(1 for c in out["cells"] if c["done_frac"] is not None and c["done_frac"] >= 0.999),
        "failed": counts.get("failed", 0), "dead": counts.get("dead", 0),
    }

    # Timeline: cumulative hours by day from the events table (exact), else from
    # updated_at of rows in their current state (approximate, and labelled so).
    by_id = {v["video_id"]: v for v in videos}
    dl_ev, done_ev = [], []
    done_at = {}          # video_id -> when it last reached 'done'; counts transcripts a QA snapshot cannot cover
    if "events" in tables:
        for r in con.execute("SELECT video_id, to_status, at FROM events ORDER BY at"):
            v = by_id.get(r["video_id"])
            t = _parse_ts(r["at"])
            if not v or not t:
                continue
            if r["to_status"] == "downloaded":
                dl_ev.append((t, capped(v)))
            elif r["to_status"] == "done":
                done_ev.append((t, float(v["transcribed_s"] or capped(v))))
                done_at[v["video_id"]] = t          # rows arrive in 'at' order, so the last write is the latest
    else:
        out["timeline"]["approximate"] = True
        for v in videos:
            t = _parse_ts(v["updated_at"])
            if t and v["status"] in DOWNLOADED_OR_LATER:
                dl_ev.append((t, capped(v)))
            if t and v["status"] == "done":
                done_ev.append((t, float(v["transcribed_s"] or capped(v))))
                done_at[v["video_id"]] = t
    if dl_ev or done_ev:
        first = min(t for t, _ in dl_ev + done_ev).date()
        days = [(first + timedelta(days=i)) for i in range((now.date() - first).days + 1)]
        def cumulative(ev):
            per_day = defaultdict(float)
            for t, s in ev:
                per_day[t.date()] += s / 3600
            run, series = 0.0, []
            for d in days:
                run += per_day.get(d, 0.0)
                series.append(round(run, 2))
            return series
        out["timeline"].update(days=[d.isoformat() for d in days],
                               downloaded_h=cumulative(dl_ev), done_h=cumulative(done_ev))
    week = now - timedelta(days=7)
    rate = sum(s for t, s in dl_ev if t >= week) / 3600 / 7
    remaining = max(frame_h - dl_h, 0.0)
    last_dl = max((t for t, _ in dl_ev), default=None)
    out["totals"].update(
        download_rate_h_per_day=round(rate, 2),
        download_eta_days=round(remaining / rate, 1) if rate > 0 and remaining > 0 else None,
        last_download_at=last_dl.isoformat(timespec="seconds") if last_dl else None,
        last_download_age_h=round((now - last_dl).total_seconds() / 3600, 1) if last_dl else None)

    # GPU batches, pods, spend ------------------------------------------------
    if "batches" in tables:
        out["batches"] = [dict(r) for r in con.execute(
            "SELECT * FROM batches ORDER BY created_at DESC LIMIT 25")]
        if "kind" in cols("videos"):
            # a batch made of meeting episodes is described in episode-hours (PROPOSAL 3i)
            for b in out["batches"]:
                k = con.execute("SELECT COUNT(*), SUM(v.kind = 'episode') FROM batch_videos bv JOIN videos v USING (video_id) "
                                "WHERE bv.batch_id = ?", (b["batch_id"],)).fetchone()
                b["episode_batch"] = bool(k[0]) and k[0] == k[1]
    # The reference point for QA freshness: transcripts exist from this moment
    # that no gate run has seen. SQLite gives the batch_id of the MAX() row.
    last_done_batch = None
    if "batches" in tables and {"status", "finished_at"} <= cols("batches"):
        r = con.execute("SELECT batch_id, MAX(finished_at) AS f FROM batches "
                        "WHERE status = 'done' AND finished_at IS NOT NULL").fetchone()
        if r and r["f"] and _parse_ts(r["f"]):
            last_done_batch = (r["batch_id"], _parse_ts(r["f"]))
    _add_meetings(out, con, tables, cols, now)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    spent = 0.0
    if "spend" in tables:
        for r in con.execute("SELECT started_at, usd FROM spend"):
            t = _parse_ts(r["started_at"])
            if t and t >= month_start and r["usd"] is not None:
                spent += float(r["usd"])
    out["spend"] = {"month_usd": round(spent, 2), "cap_usd": cfg["spend_cap_usd"],
                    "frac": round(spent / cfg["spend_cap_usd"], 3) if cfg["spend_cap_usd"] else None}
    try:                                                     # T2 (2026-09-22): the credit ledger + Claude cap, pipeline/spend.py
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "pipeline"))
        import spend as _spend                               # noqa: E402
        out["spend"]["ledger"] = _spend.summary(con)
    except Exception as e:                                   # noqa: BLE001 -- status must render on any DB
        out["spend"]["ledger"] = {"error": repr(e)}
    if "pods" in tables:
        out["pods_active"] = [dict(r) for r in con.execute(
            "SELECT * FROM pods WHERE status IN ('creating', 'running', 'ORPHAN')")]
    if "videos" in tables:
        out["errors"] = [dict(r) for r in con.execute(
            "SELECT video_id, speaker, status, attempts, last_error, updated_at FROM videos "
            "WHERE status IN ('failed', 'dead') ORDER BY updated_at DESC LIMIT 20")]
    con.close()

    # Alerts, most severe first -------------------------------------------------
    # A pod is legitimate only while a batch is approved, launching or running:
    # the pod exists for a few minutes before the batch flips to 'running'.
    # ... and only for as long as a launch can actually run: gpu_job caps a job
    # at 12 h, so a batch still 'running' after STALE_BATCH_HOURS means the
    # launch process died (SIGKILL, Mac asleep) with its pod possibly alive --
    # exactly the case the finally: cannot cover (b10, 2026-09-20).
    stale_h = cfg.get("stale_batch_hours", 13)
    running, stale = [], []
    for b in out["batches"]:
        if b.get("status") not in ("approved", "launching", "running"):
            continue
        t = _parse_ts(b.get("started_at") or b.get("approved_at") or b.get("created_at"))
        (stale if t and (now - t).total_seconds() > stale_h * 3600 else running).append(b)
    for p in out["pods_active"]:
        if p.get("status") == "ORPHAN" or not running:
            why = (f"its batch {stale[0]['batch_id']} has been '{stale[0]['status']}' for over {stale_h:.0f} h (launch died?)"
                   if stale and not running else "no running batch")
            alert("critical", "orphan_pod",
                  f"GPU pod {p.get('pod_id')} is {p.get('status')} with {why} -- it is billing. "
                  "Terminate it in the RunPod console now.")
    if out["spend"]["frac"] is not None and out["spend"]["frac"] >= 1:
        alert("critical", "spend_cap", f"GPU spend ${spent:.2f} has reached the ${cfg['spend_cap_usd']:.0f} monthly cap.")
    elif out["spend"]["frac"] is not None and out["spend"]["frac"] >= 0.8:
        alert("warning", "spend_cap", f"GPU spend ${spent:.2f} is over 80% of the ${cfg['spend_cap_usd']:.0f} cap.")
    for b in out["batches"]:
        if b.get("status") == "awaiting_approval":
            asked = _parse_ts(b.get("approval_requested_at") or b.get("created_at"))
            age = f", asked {int((now - asked).total_seconds() // 60)} min ago" if asked else ""
            unit, hrs = ("episode clips", "episode-h") if b.get("episode_batch") else ("videos", "audio-h")
            alert("warning", "approval",
                  f"Batch {b['batch_id']} awaits approval: {b.get('n_videos')} {unit}, "
                  f"{b.get('audio_hours')} {hrs}, est ${b.get('est_usd')}{age}.")
    for body in out["meetings"]["bodies"]:
        if body.get("silent_days") is not None and body["silent_days"] > cfg["silent_days"]:
            alert("warning", "body_silent",
                  f"{body['body']}: no meeting found in {body['silent_days']} days (last {body['last_meeting'] or 'never'}) "
                  "-- source dried up or moved?")
    if out["totals"]["dead"]:
        alert("serious", "dead", f"{out['totals']['dead']} videos are dead (3 failed attempts) and need a human.")
    stalled = out["totals"]["last_download_age_h"]
    if remaining > 0 and stalled is not None and stalled > cfg["stall_hours"]:
        alert("warning", "stalled", f"No download has finished for {stalled:.1f} h.")
    _add_qa(out, qa_csv, alert, now, last_done_batch, done_at)
    order = {"critical": 0, "serious": 1, "warning": 2, "info": 3}
    out["alerts"].sort(key=lambda a: order.get(a["level"], 9))
    return out


def _add_meetings(out, con, tables, cols, now):
    """Per body: meetings by stage, episodes located / cut / transcribed, mention
    rows, episode-hours, days since the last meeting (for the silence alert)."""
    if "videos" not in tables or "kind" not in cols("videos"):
        return
    bodies = {}
    for r in con.execute("SELECT speaker, status, COUNT(*) n, MAX(meeting_date) last, COALESCE(SUM(duration_s), 0) s "
                         "FROM videos WHERE kind = 'meeting' GROUP BY speaker, status"):
        b = bodies.setdefault(r["speaker"], {"body": r["speaker"], "meetings": 0, "by_status": {}, "meeting_hours": 0.0,
                                             "last_meeting": None, "located": 0, "cut": 0, "in_batch": 0, "transcribed": 0,
                                             "mention": 0, "failed": 0, "located_h": 0.0, "transcribed_h": 0.0})
        b["meetings"] += r["n"]; b["by_status"][r["status"]] = r["n"]; b["meeting_hours"] += r["s"] / 3600
        if r["last"] and (b["last_meeting"] is None or r["last"] > b["last_meeting"]):
            b["last_meeting"] = r["last"]
    if "sources" in tables:
        for r in con.execute("SELECT body FROM sources WHERE active = 1"):
            bodies.setdefault(r["body"], {"body": r["body"], "meetings": 0, "by_status": {}, "meeting_hours": 0.0,
                                          "last_meeting": None, "located": 0, "cut": 0, "in_batch": 0, "transcribed": 0,
                                          "mention": 0, "failed": 0, "located_h": 0.0, "transcribed_h": 0.0})
    if "episodes" in tables:
        for r in con.execute("SELECT v.speaker, e.status, COUNT(*) n, COALESCE(SUM(e.end_s - e.start_s), 0) s "
                             "FROM episodes e JOIN videos v ON v.video_id = e.video_id GROUP BY v.speaker, e.status"):
            b = bodies.get(r["speaker"])
            if not b:
                continue
            st = r["status"]
            if st == "mention":
                b["mention"] += r["n"]
            elif st == "failed":
                b["failed"] += r["n"]
            else:
                b["located"] += r["n"]; b["located_h"] += r["s"] / 3600
                if st == "cut":
                    b["cut"] += r["n"]
                elif st in ("queued", "transcribing"):
                    b["in_batch"] += r["n"]
                elif st == "done":
                    b["transcribed"] += r["n"]; b["transcribed_h"] += r["s"] / 3600
    for b in bodies.values():
        if b["last_meeting"]:
            try:
                b["silent_days"] = (now.date() - datetime.fromisoformat(b["last_meeting"]).date()).days
            except ValueError:
                b["silent_days"] = None
        else:
            b["silent_days"] = None
        for k in ("meeting_hours", "located_h", "transcribed_h"):
            b[k] = round(b[k], 2)
    out["meetings"]["bodies"] = sorted(bodies.values(), key=lambda b: b["body"])
    tot = {"bodies": len(bodies)}
    for k in ("meetings", "located", "cut", "in_batch", "transcribed", "mention", "failed", "meeting_hours", "located_h", "transcribed_h"):
        tot[k] = round(sum(b[k] for b in bodies.values()), 2)
    out["meetings"]["totals"] = tot


def _add_qa(out, qa_csv, alert, now=None, last_done_batch=None, done_at=None):
    """QA counts, AND how old the snapshot they came from is.

    status.py does not run the gate -- it reads the CSV that
    analysis/qa_transcripts.py left behind. So "0 FAIL" means one of two very
    different things: the corpus is clean, or nobody has looked. Before
    2026-09-28 those were indistinguishable in the output, and the second one
    is what was actually true (192/61/0 from 2026-09-23 while 299 transcripts
    landed after it). The snapshot's mtime therefore goes next to the counts,
    and a snapshot older than the last finished batch is an alert in its own
    right -- a stale gate is exactly the state in which a real FAIL hides, so
    it is 'serious', the same level as a known qa_fail.

    An absent snapshot used to produce no QA line and no alert at all, which
    read as "nothing to report" for the case with the least coverage of all.
    It is now the loudest of the three."""
    now = now or datetime.now(timezone.utc)
    path = Path(qa_csv)
    rows = _read_csv(path)
    if rows is None:
        out["qa"] = {"path": str(qa_csv), "exists": False, "ok": 0, "warn": 0, "fail": 0, "rows": [],
                     "ran_at": None, "age_h": None, "stale": True, "covers_through": None, "ungated": None}
        alert("serious", "qa_missing",
              f"QA gate has never run here: no snapshot at {qa_csv}. No transcript is gated -- "
              "run `make qa` before `make load`.")
        return
    n = defaultdict(int)
    for r in rows:
        n[r.get("status", "?")] += 1
    ran = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    bid, finished = last_done_batch or (None, None)
    out["qa"] = {"path": str(qa_csv), "exists": True, "ok": n["OK"], "warn": n["WARN"], "fail": n["FAIL"],
                 "ran_at": ran.isoformat(timespec="seconds"),
                 "age_h": round((now - ran).total_seconds() / 3600, 1),
                 "stale": bool(finished and finished > ran), "ungated": None,
                 "covers_through": finished.isoformat(timespec="seconds") if finished else None,
                 "rows": [{"status": r.get("status"), "video_id": r.get("video_id"),
                           "issues": r.get("issues", "")} for r in rows if r.get("status") != "OK"][:50]}
    if n["FAIL"]:
        alert("serious", "qa_fail", f"QA gate: {n['FAIL']} transcripts FAIL -- do not load them.")
    if out["qa"]["stale"]:
        ungated = sum(1 for t in (done_at or {}).values() if t > ran)
        out["qa"]["ungated"] = ungated
        alert("serious", "qa_stale",
              f"QA snapshot is stale: the gate last ran {_stamp(ran)}, batch {bid} finished "
              f"{_stamp(finished)}, and {ungated} transcripts have never been through it. "
              f"'{n['FAIL']} FAIL' does not cover them -- run `make qa`.")


def as_text(s, url=None):
    """A WhatsApp-sized digest (well under 1,500 characters)."""
    t = s["totals"]
    lines = [f"ytpt status {s['generated_at'][:16].replace('T', ' ')} UTC"]
    if not s["state_db"]["exists"]:
        lines.append("No state database yet -- the pipeline has not started.")
    else:
        lines.append(f"Transcribed {t['done_hours']:.1f} of {t['frame_hours']:.1f} h "
                     f"({t['done_frac']:.0%}); {t['cells_full']} of {t['cells_total']} cells complete")
        eta = f", ETA {t['download_eta_days']:.0f} d" if t.get("download_eta_days") else ""
        lines.append(f"Downloaded {t['downloaded_hours']:.1f} h ({t['download_rate_h_per_day']:.1f} h/day{eta})")
        lines.append(f"Ready for GPU {t['ready_for_gpu_hours']:.1f} h "
                     f"(batch threshold {s['settings']['batch_target_hours']:.0f} h)")
        lines.append(f"Failed {t['failed']}, dead {t['dead']}")
        sp = s["spend"]
        lines.append(f"GPU spend this month ${sp['month_usd']:.2f} of ${sp['cap_usd']:.0f}")
        led = sp.get("ledger") or {}
        if "runpod_usd" in led:
            lines.append(f"RunPod ${led['runpod_usd']:.2f} of ${led['runpod_credit_usd']:.2f} credit (stop at ${led['runpod_stop_usd']:.2f}"
                         f"{', BLOCKED' if led.get('runpod_blocked') else ''}); Claude classifier ${led['claude_usd']:.2f} of "
                         f"${led['claude_cap_usd']:.2f}{' BLOCKED' if led.get('claude_blocked') else ''}")
        mt = s.get("meetings", {}).get("totals") or {}
        if mt.get("bodies"):
            lines.append(f"Meetings: {mt['meetings']} ({mt['meeting_hours']:.0f} h) over {mt['bodies']} bodies; episodes "
                         f"{mt['located']} located ({mt['located_h']:.1f} h), {mt['cut']} cut, {mt['in_batch']} in batch, "
                         f"{mt['transcribed']} transcribed ({mt['transcribed_h']:.1f} h), {mt['mention']} mention-only")
            for b in s["meetings"]["bodies"]:
                lines.append(f"  {b['body']}: {b['meetings']} mtg, ep {b['located']}/{b['cut']}/{b['transcribed']} "
                             f"located/cut/done, {b['mention']} mention, last {b['last_meeting'] or '-'}")
    q = s.get("qa")
    if q and not q.get("exists", True):
        lines.append("QA: NO SNAPSHOT -- the gate has never run here")
    elif q:
        h = q.get("age_h")
        age = "" if h is None else (f", {h:.0f} h old" if h < 48 else f", {h / 24:.1f} d old")
        lines.append(f"QA {q['ran_at'][:16].replace('T', ' ')} UTC{age}"
                     f"{' STALE' if q.get('stale') else ''}: "
                     f"{q['ok']} OK, {q['warn']} WARN, {q['fail']} FAIL")
    for a in s["alerts"][:5]:
        lines.append(f"! {a['text']}")
    if url:
        lines.append(url)
    return "\n".join(lines)


# Only the parts of an alert that are pure elapsed time are normalised away, so
# "asked 34 min ago" -> "asked 64 min ago" is not news, but "4 dead" -> "5 dead"
# is. Everything else (counts, dollars, batch ids) stays in the key.
_AGE = re.compile(r"(asked \d+ min ago|for [\d.]+ h\b|\b[\d.]+ h ago)")


def alert_key(a):
    return a["kind"] + ":" + _AGE.sub("<age>", a["text"])


def new_alerts(s, seen_path):
    """Alerts that are new since the last call, plus kinds that have cleared.
    The heartbeat relays this instead of the full digest, so a problem reaches
    WhatsApp once rather than every 30 minutes."""
    seen_path = Path(seen_path)
    try:
        before = json.loads(seen_path.read_text())
    except (OSError, ValueError):
        before = []
    before_keys = {k for k, _ in before}
    before_kinds = {k.split(":", 1)[0] for k, _ in before}
    now = [(alert_key(a), a["level"]) for a in s["alerts"]]
    now_keys = {k for k, _ in now}
    lines = [f"NEW {a['level'].upper()}: {a['text']}"
             for a in s["alerts"] if alert_key(a) not in before_keys]
    gone = sorted(before_kinds - {k.split(":", 1)[0] for k in now_keys})
    lines += [f"CLEARED: {kind}" for kind in gone]
    try:
        seen_path.parent.mkdir(parents=True, exist_ok=True)
        seen_path.write_text(json.dumps(sorted(now)))
    except OSError:
        pass
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default=None, help="state.sqlite (default STATE_DB_PATH)")
    ap.add_argument("--frame-cells", default=None)
    ap.add_argument("--qa", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--new-alerts", action="store_true",
                    help="print only alerts that are new or cleared since the last --new-alerts call "
                         "(empty output = nothing to report)")
    a = ap.parse_args(argv)
    cfg = settings()
    s = collect(a.state, a.frame_cells, a.qa, cfg=cfg)
    if a.new_alerts:
        seen = Path(s["state_db"]["path"]).parent / "alerts_seen.json"
        out = new_alerts(s, seen)
        if out:
            print(out)
        return
    print(json.dumps(s, indent=1, default=str) if a.json
          else as_text(s, f"Dashboard: http://localhost:{cfg['port']}"))


if __name__ == "__main__":
    main()
