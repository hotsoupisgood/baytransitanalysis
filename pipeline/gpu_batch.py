#!/usr/bin/env python3
"""
pipeline/gpu_batch.py -- GPU batches and the approval gate (DECISIONS.md 10,
PLAN.md 4.8, SCHEMA.md `batches`).

A batch is PROPOSED by this script from the transcoded queue, never by the
agent. Puck approves it on WhatsApp. Then, and only then, `launch` may rent a
pod -- and `launch` is deliberately not on OpenClaw's exec allowlist, so
OpenClaw asks Puck to /approve the command itself. Three layers; this file is
the middle one and it does not trust the other two.

  propose [--min-hours H] [--limit N] [--speaker S[,S...]]
      Take in-frame 'transcoded' rows (oldest first) into a new batch row
      'awaiting_approval', move the videos to 'queued', message Puck with the
      count, capped audio-hours and the ESTIMATED cost. Refuses below
      BATCH_TARGET_HOURS unless --min-hours says otherwise (a 3-video batch pays
      more for boot than for work; PLAN.md 2). --speaker keeps only rows whose
      speaker is in the list, so a test batch (T0 test 5: the five transit
      bodies) does not sweep up rows parked in 'transcoded' for another reason.
  approve <batch_id> [--via whatsapp]
      Record that Puck said "approve <id>". Bookkeeping only; rents nothing.
  deny <batch_id>
      Status 'denied'; its videos go back to 'transcoded'.
  launch <batch_id>
      Refuses unless: the row is 'approved'; this month's recorded spend plus
      the estimate is under MONTHLY_SPEND_CAP_USD; no pod is creating/running/
      ORPHAN; RUNPOD_API_KEY is set; and the Phase 0/2 artefacts exist
      (plan/gpu-freeze.txt, pipeline/gpu_bootstrap.sh, pipeline/gpu_job.py).
      Every refusal is printed and exits 2. --dry-run stops after the checks.
      When it does run: the `pods` row is written BEFORE the create call and a
      finally: terminates the pod on every exit path (SCHEMA.md `pods`).
  teardown <pod_id>
      Terminate a pod (orphan runbook) and record the spend.
  reconcile <batch_id> [--dry-run] [--terminate]
      A batch stuck in launching / running / failed with a creating / running
      pods row (2026-09-21 review M7): ask RunPod (by id and by name
      ytpt-<batch>) and settle. A live pod is REPORTED, never terminated
      without --terminate; a gone pod -> pods row terminated, batch failed,
      videos released. Never creates a pod.
  list
      All batches.

Spend caps (2026-09-22, plan/T2-PLAN.md): propose and launch also refuse when
the recorded RunPod total plus the estimate would pass the stop line
(RUNPOD_CREDIT_USD - RUNPOD_CREDIT_MARGIN_USD, default $10 - $0.50); see
pipeline/spend.py.

Cost estimate: (audio_hours / RTFX + boot) * $/hr * 1.5 contingency, with
RTFX = 25 an ESTIMATE until Phase 0 measures it (PLAN.md 5). The number is
stored per batch so a later, measured RTFX does not rewrite history.
"""
import argparse, json, os, sys, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402
from notify import notify  # noqa: E402

ROOT = state.ROOT
RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"
EXIT_REFUSED = 2
PHASE0_ARTEFACTS = ("plan/gpu-freeze.txt", "pipeline/gpu_bootstrap.sh", "pipeline/gpu_job.py")


class Interrupted(Exception):
    """SIGTERM / SIGHUP during launch, raised so the failure path runs (see launch)."""


def cfg():
    f = lambda k, d: float(state.setting(k, d))
    return {"cap_usd": f("MONTHLY_SPEND_CAP_USD", 150), "target_h": f("BATCH_TARGET_HOURS", 15),
            "rtfx": f("GPU_RTFX_ESTIMATE", 25), "usd_hr": f("GPU_USD_PER_HOUR", 0.34),
            "boot_min": f("GPU_BOOT_MINUTES", 7), "contingency": f("GPU_COST_CONTINGENCY", 1.5)}


def estimate_usd(audio_hours, c=None):
    c = c or cfg()
    return round((audio_hours / c["rtfx"] + c["boot_min"] / 60) * c["usd_hr"] * c["contingency"], 2)


def month_spend(con, now=None):
    now = now or datetime.now(timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")
    r = con.execute("SELECT COALESCE(SUM(usd), 0) AS s FROM spend WHERE started_at >= ?", (start,)).fetchone()
    return float(r["s"] or 0)


def active_pods(con):
    return [dict(r) for r in con.execute("SELECT * FROM pods WHERE status IN ('creating', 'running', 'ORPHAN')")]


def next_batch_id(con):
    n = con.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
    while True:
        n += 1
        if not con.execute("SELECT 1 FROM batches WHERE batch_id = ?", (f"b{n}",)).fetchone():
            return f"b{n}"


def get_batch(con, batch_id):
    r = con.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
    if not r:
        sys.exit(f"no batch {batch_id}")
    return dict(r)


def set_batch(con, batch_id, status, **fields):
    sets = ", ".join(["status = ?"] + [f"{k} = ?" for k in fields])
    con.execute(f"UPDATE batches SET {sets} WHERE batch_id = ?", [status, *fields.values(), batch_id])


def batch_videos(con, batch_id):
    return [r["video_id"] for r in con.execute(
        "SELECT video_id FROM batch_videos WHERE batch_id = ? ORDER BY video_id", (batch_id,))]


# --- propose / approve / deny -------------------------------------------------

def episodes_follow(con, video_ids, status, note):
    """Mirror a clip's batch state onto its `episodes` row (2026-09-21). Only
    rows currently in the batch flow (cut / queued / transcribing) move."""
    if not video_ids or not con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'episodes'").fetchone():
        return 0
    n = 0
    for vid in video_ids:
        r = con.execute("SELECT episode_id FROM videos WHERE video_id = ?", (vid,)).fetchone()
        ep = (r["episode_id"] if r and "episode_id" in r.keys() else None) or vid
        cur = con.execute("SELECT status FROM episodes WHERE episode_id = ?", (ep,)).fetchone()
        if cur and cur["status"] in ("cut", "queued", "transcribing") and cur["status"] != status:
            n += state.set_episode(con, ep, status, note=note)
    return n

def parse_speakers(spec):
    """'a,b, c' -> ['a', 'b', 'c']; None/'' -> [] (no filter)."""
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def propose(con, min_hours=None, limit=0, quiet=False, speakers=None):
    c = cfg()
    speakers = parse_speakers(speakers) if isinstance(speakers, str) else list(speakers or [])
    where = "in_frame = 1 AND status = 'transcoded'"
    params = []
    if speakers:
        where += f" AND speaker IN ({','.join('?' * len(speakers))})"
        params = speakers
    kind_col = "kind" if "kind" in {r[1] for r in con.execute("PRAGMA table_info(videos)")} else "'show' AS kind"
    rows = [dict(r) for r in con.execute(
        f"SELECT video_id, speaker, duration_s, transcribed_s, opus_path, {kind_col} FROM videos "
        f"WHERE {where} ORDER BY updated_at, video_id" + (f" LIMIT {int(limit)}" if limit else ""), params)]
    missing = [r["video_id"] for r in rows if not (r["opus_path"] and Path(r["opus_path"]).exists())]
    if missing:
        sys.exit(f"refusing: opus file missing for {' '.join(missing)} -- run pipeline/requeue.py on them")
    hours = sum(float(r["transcribed_s"] or 0) for r in rows) / 3600
    threshold = c["target_h"] if min_hours is None else min_hours
    if not rows:
        print("nothing transcoded and waiting" + (f" for speaker {', '.join(speakers)}" if speakers else "")
              + "; no batch proposed")
        return None
    if hours < threshold:
        print(f"{len(rows)} videos, {hours:.2f} capped audio-h ready; below the {threshold:.0f} h threshold "
              f"(BATCH_TARGET_HOURS) -- no batch proposed. --min-hours 0 to propose anyway.")
        return None
    est = estimate_usd(hours, c)
    import spend                                          # noqa: E402 -- the credit stop line (T2)
    why = spend.runpod_refusal(con, est)
    if why:
        print(f"refusing to propose: {why}")
        return None
    bid = next_batch_id(con)
    now = state.now_iso()
    con.execute("BEGIN IMMEDIATE")
    con.execute("INSERT INTO batches (batch_id, created_at, status, n_videos, audio_hours, est_usd, "
                "approval_requested_at, note) VALUES (?,?,?,?,?,?,?,?)",
                (bid, now, "awaiting_approval", len(rows), round(hours, 3), est, now,
                 f"est = ({hours:.2f} h / {c['rtfx']:.0f}x RTFX ESTIMATE + {c['boot_min']:.0f} min boot) "
                 f"* ${c['usd_hr']:.2f}/h * {c['contingency']:.1f} contingency"))
    con.executemany("INSERT INTO batch_videos (batch_id, video_id) VALUES (?, ?)",
                    [(bid, r["video_id"]) for r in rows])
    con.execute("COMMIT")
    for r in rows:
        state.transition(con, r["video_id"], "transcoded", "queued", note=f"batch {bid}")
    episodes_follow(con, [r["video_id"] for r in rows if r.get("kind") == "episode"], "queued", f"batch {bid}")
    by_sp = {}
    for r in rows:
        by_sp[r["speaker"]] = by_sp.get(r["speaker"], 0) + 1
    # 2026-09-21: a batch of meeting episodes is described in episode-hours and
    # clips per body, not "videos" (PROPOSAL 3i)
    all_episodes = all(r.get("kind") == "episode" for r in rows)
    unit, hours_label = ("episode clips", "episode-h") if all_episodes else ("videos", "capped audio-h")
    msg = (f"batch {bid} awaits your approval: {len(rows)} {unit} "
           f"({', '.join(f'{v} {k}' for k, v in sorted(by_sp.items()))}), {hours:.2f} {hours_label}, "
           f"est ${est:.2f} (RTFX {c['rtfx']:.0f}x is an estimate; cap ${c['cap_usd']:.0f}, "
           f"spent this month ${month_spend(con):.2f}). Reply 'approve {bid}' or 'deny {bid}'.")
    print(msg)
    if not quiet:
        notify("approval", msg)
    return bid


def approve(con, batch_id, via="whatsapp"):
    b = get_batch(con, batch_id)
    if b["status"] != "awaiting_approval":
        sys.exit(f"{batch_id} is '{b['status']}', not awaiting_approval; nothing recorded")
    set_batch(con, batch_id, "approved", approved_at=state.now_iso(), approved_via=via)
    print(f"{batch_id} approved via {via} at {state.now_iso()}. Next: pipeline/gpu_batch.py launch {batch_id} "
          "(OpenClaw will ask Puck to /approve that command).")


def deny(con, batch_id, note="denied"):
    b = get_batch(con, batch_id)
    if b["status"] not in ("proposed", "awaiting_approval", "approved"):
        sys.exit(f"{batch_id} is '{b['status']}'; cannot deny")
    set_batch(con, batch_id, "denied", note=note, finished_at=state.now_iso())
    n = sum(state.transition(con, v, "queued", "transcoded", note=f"batch {batch_id} denied")
            for v in batch_videos(con, batch_id))
    episodes_follow(con, batch_videos(con, batch_id), "cut", f"batch {batch_id} denied")
    print(f"{batch_id} denied; {n} videos back to 'transcoded'. No pod exists.")


# --- launch -------------------------------------------------------------------

def launch_checks(con, batch_id, now=None):
    """Every reason not to rent a GPU, as a list of strings. Empty = go."""
    c = cfg()
    b = get_batch(con, batch_id)
    problems = []
    if b["status"] != "approved":
        problems.append(f"batch {batch_id} is '{b['status']}', not 'approved' (Puck must reply 'approve {batch_id}')")
    spent = month_spend(con, now)
    est = float(b["est_usd"] or 0)
    if spent + est > c["cap_usd"]:
        problems.append(f"spend ${spent:.2f} + estimate ${est:.2f} exceeds MONTHLY_SPEND_CAP_USD ${c['cap_usd']:.2f}")
    import spend                                          # noqa: E402 -- the credit stop line (T2, 2026-09-22)
    why = spend.runpod_refusal(con, est)
    if why:
        problems.append(why)
    pods = active_pods(con)
    if pods:
        problems.append("a pod is already live: " + ", ".join(f"{p['pod_id']} ({p['status']})" for p in pods))
    if not state.setting("RUNPOD_API_KEY", ""):
        problems.append("RUNPOD_API_KEY is not set")
    if not state.setting("HF_TOKEN", ""):
        problems.append("HF_TOKEN is not set (pyannote would 401 on the pod)")
    missing = [rel for rel in PHASE0_ARTEFACTS if not (ROOT / rel).exists()]
    for rel in missing:
        problems.append(f"{rel} does not exist -- Phase 0/2 (PLAN.md 6) has not produced it; "
                        "nothing can run on a pod yet")
    if not batch_videos(con, batch_id):
        problems.append("batch has no videos")
    if not missing:
        # Everything that would fail AFTER money was spent -- SSH key, rsync,
        # opus files, output dirs, bootstrap syntax, freeze pins -- for $0.
        import importlib
        try:
            problems += importlib.import_module("gpu_job").preflight(con, batch_id)
        except Exception as e:                        # noqa: BLE001 -- a broken job module is a refusal, not a crash
            problems.append(f"pipeline/gpu_job.py preflight crashed: {e!r}")
    return problems


def launch(con, batch_id, dry_run=False):
    problems = launch_checks(con, batch_id)
    if problems:
        print(f"REFUSED to launch {batch_id}:")
        for p in problems:
            print(f"  - {p}")
        return EXIT_REFUSED
    b = get_batch(con, batch_id)
    print(f"{batch_id}: all checks pass ({b['n_videos']} videos, {b['audio_hours']} h, est ${b['est_usd']}).")
    import importlib
    job = importlib.import_module("gpu_job")          # pipeline/gpu_job.py (Phase 2)
    if dry_run:
        print(job.describe_plan(con, batch_id))
        print("dry run: stopping before any RunPod call.")
        return 0
    # Reached only with every artefact in place. The pods row is written BEFORE
    # the create call; the finally: terminates on every exit path.
    # SIGTERM / SIGHUP (launchctl, a closed terminal, an exec host's timeout)
    # are turned into an exception so that path is the SAME as a failure:
    # batch 'failed', videos released, pod torn down. Only SIGKILL escapes,
    # which is why the runbook says to run launch detached (nohup) and why
    # status.py flags a pod whose batch has been 'running' too long.
    set_batch(con, batch_id, "launching", started_at=state.now_iso())
    pod_id = None
    import signal

    def interrupted(signum, frame):
        raise Interrupted(f"launch received signal {signum}")
    previous = {}
    try:
        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP)}
    except ValueError:                                # not the main thread: no handlers, the finally: still runs
        pass
    try:
        pod_id = job.create_pod(con, batch_id)          # inserts the pods row first
        set_batch(con, batch_id, "running", pod_id=pod_id)
        result = job.run(con, batch_id, pod_id)
        set_batch(con, batch_id, "done", finished_at=state.now_iso(), **result)
        return 0
    except (Exception, KeyboardInterrupt) as e:       # noqa: BLE001 -- any failure is recorded, never swallowed
        set_batch(con, batch_id, "failed", note=(str(e) or type(e).__name__)[:500], finished_at=state.now_iso())
        # Release whatever this batch still holds: 'queued' when the create
        # call failed (SUPPLY_CONSTRAINT), 'transcribing' when the pod died or
        # the job never reported (b6). Videos already 'done'/'failed' from a
        # partial run keep their state; their transcripts were ingested.
        n = 0
        for v in batch_videos(con, batch_id):
            for frm in ("queued", "transcribing"):
                if state.transition(con, v, frm, "transcoded", note=f"batch {batch_id} failed: {str(e)[:120]}"):
                    n += 1
                    episodes_follow(con, [v], "cut", f"batch {batch_id} failed")
                    break
        print(f"{batch_id} failed; {n} unfinished videos back to 'transcoded'.")
        raise
    finally:
        if pod_id:
            teardown(con, pod_id, n_videos=b["n_videos"])
        for sig, handler in previous.items():
            signal.signal(sig, handler)


# --- teardown -----------------------------------------------------------------

def runpod_graphql(query, variables=None, timeout=30):
    key = state.setting("RUNPOD_API_KEY", "")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY is not set")
    # The key travels in the Authorization header, not the URL query string
    # (2026-09-21 review L1). Switched 2026-09-22 after scripts/check_credentials.py
    # proved the header path live on the same endpoint (read-only `myself`).
    req = urllib.request.Request(RUNPOD_GRAPHQL,
                                 data=json.dumps({"query": query, "variables": variables or {}}).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}",
                                          # Cloudflare rejects the default Python-urllib UA (error 1010)
                                          "User-Agent": "ytpt-gpu-batch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    if out.get("errors"):
        raise RuntimeError(f"RunPod: {out['errors']}")
    return out.get("data", {})


def teardown(con, pod_id, n_videos=None, note=None):
    """Terminate a pod and record what it cost. Idempotent: a pod this DB
    already has as 'terminated' is left alone (a second call would otherwise
    bill its uptime into `spend` twice, and RunPod's error for an unknown pod
    would flip the row to ORPHAN and block every later launch)."""
    c = cfg()
    row = con.execute("SELECT * FROM pods WHERE pod_id = ?", (pod_id,)).fetchone()
    if row and row["status"] == "terminated":
        print(f"pod {pod_id} is already recorded as terminated ({row['terminated_at']}); nothing to do")
        return 0
    try:
        runpod_graphql('mutation ($id: String!) { podTerminate(input: {podId: $id}) }', {"id": pod_id})
        print(f"pod {pod_id} terminated")
    except Exception as e:                            # noqa: BLE001 -- ANY failure here is an orphan until proven otherwise
        # Not a narrow list: a Cloudflare HTML page (json ValueError) or an
        # http.client.HTTPException escaped the old tuple and left the pods
        # row 'running' with no message to Puck (2026-09-21 review).
        print(f"pod {pod_id}: terminate call failed: {type(e).__name__}: {e} -- check the RunPod console now", file=sys.stderr)
        notify("error", f"could not terminate pod {pod_id}: {type(e).__name__}: {str(e)[:200]}. Check the RunPod console NOW.")
        if row:
            con.execute("UPDATE pods SET status = 'ORPHAN' WHERE pod_id = ?", (pod_id,))
        return 1
    ended = state.now_iso()
    started = row["created_at"] if row else ended
    hours = max((datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds(), 0) / 3600
    usd = round(hours * c["usd_hr"], 4)
    if row:
        con.execute("UPDATE pods SET status = 'terminated', terminated_at = ? WHERE pod_id = ?", (ended, pod_id))
    else:
        con.execute("INSERT INTO pods VALUES (?,?,?,?)", (pod_id, started, ended, "terminated"))
    con.execute("INSERT INTO spend (pod_id, started_at, ended_at, gpu_hours, usd, n_videos, note) VALUES (?,?,?,?,?,?,?)",
                (pod_id, started, ended, round(hours, 4), usd, n_videos, note or "teardown"))
    print(f"spend recorded: {hours * 60:.1f} GPU-min, ${usd:.2f}")
    return 0


# --- reconcile (2026-09-21 review M7, approved by Puck 2026-09-22) -------------

RECONCILABLE = ("launching", "running", "failed")
POD_LIVE = ("RUNNING", "CREATED", "RESTARTING", "PAUSED")
POD_BY_ID_QUERY = """
query ($id: String!) {
  pod(input: {podId: $id}) { id name desiredStatus costPerHr runtime { uptimeInSeconds } }
}"""
POD_BY_NAME_QUERY = "query { myself { pods { id name desiredStatus costPerHr runtime { uptimeInSeconds } } } }"


def reconcile_candidates(con, batch_id):
    """The pods rows this batch may still be holding: its recorded pod_id, or a
    provisional 'pending-<batch>-*' row, in creating / running / ORPHAN."""
    b = get_batch(con, batch_id)
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM pods WHERE status IN ('creating', 'running', 'ORPHAN') AND (pod_id = ? OR pod_id LIKE ?) ORDER BY created_at",
        (b.get("pod_id") or "", f"pending-{batch_id}-%"))]
    return b, rows


def lookup_pod(batch_id, pod_id=None):
    """What RunPod knows: by id when it is a real id, and by name ytpt-<batch>
    among my pods (a provisional id names nothing). Read-only. Returns
    {id, name, desiredStatus, uptime_s} or None when no such pod exists."""
    found = None
    if pod_id and not pod_id.startswith("pending-"):
        try:
            found = (runpod_graphql(POD_BY_ID_QUERY, {"id": pod_id}, timeout=60) or {}).get("pod")
        except RuntimeError as e:
            if "not found" not in str(e).lower() and "does not exist" not in str(e).lower():
                raise
    if not found:
        mine = ((runpod_graphql(POD_BY_NAME_QUERY, timeout=60) or {}).get("myself") or {}).get("pods") or []
        found = next((p for p in mine if p.get("name") == f"ytpt-{batch_id}"), None)
    if not found:
        return None
    return {"id": found.get("id"), "name": found.get("name"), "desiredStatus": found.get("desiredStatus"),
            "uptime_s": ((found.get("runtime") or {}).get("uptimeInSeconds") or 0)}


def release_batch(con, batch_id, why):
    """The failure path's bookkeeping, shared with launch: queued / transcribing
    videos back to 'transcoded', their episodes rows to 'cut'. Returns n."""
    n = 0
    for v in batch_videos(con, batch_id):
        for frm in ("queued", "transcribing"):
            if state.transition(con, v, frm, "transcoded", note=f"batch {batch_id} {why}"[:200]):
                n += 1
                episodes_follow(con, [v], "cut", f"batch {batch_id} {why}")
                break
    return n


def reconcile(con, batch_id, dry_run=False, terminate=False):
    """Settle a batch left in launching / running / failed with a pods row still
    creating / running / ORPHAN (a kill between set_batch('launching') and the
    create call returning, a launch failure, a SIGKILL). Asks RunPod what is
    actually there and makes the DB agree:

      pod exists and is live   -> REPORT ONLY (nothing changes) unless
                                  --terminate, which runs teardown (spend
                                  recorded) and then settles as below
      pod gone / exited        -> pods row 'terminated', batch 'failed' with a
                                  note, queued / transcribing videos back to
                                  'transcoded', episodes rows 'cut'

    Never creates a pod; every RunPod call here is a read except the
    explicit --terminate. --dry-run does the lookups and prints the plan.
    Exit 2 when the batch is not reconcilable, 0 otherwise."""
    b, rows = reconcile_candidates(con, batch_id)
    if b["status"] not in RECONCILABLE:
        print(f"{batch_id} is '{b['status']}'; reconcile handles {', '.join(RECONCILABLE)} only. Nothing changed.")
        return EXIT_REFUSED
    print(f"{batch_id}: status '{b['status']}', pod_id {b.get('pod_id') or '-'}; "
          f"{len(rows)} live pods row(s) in the DB: " + (", ".join(f"{r['pod_id']} ({r['status']})" for r in rows) or "none"))
    pod = lookup_pod(batch_id, b.get("pod_id") or (rows[0]["pod_id"] if rows else None))
    if pod:
        print(f"RunPod: pod {pod['id']} ({pod['name']}) is {pod['desiredStatus']}, uptime {pod['uptime_s'] / 60:.1f} min")
    else:
        print(f"RunPod: no pod named ytpt-{batch_id}" + (f" and no pod {b['pod_id']}" if b.get("pod_id") and not str(b["pod_id"]).startswith("pending-") else ""))
    if not rows and not pod:
        print("nothing to reconcile: no live pods row and no pod on RunPod." + ("" if b["status"] == "failed" else
              f" (batch stays '{b['status']}'; use deny/requeue paths for its videos)"))
        return 0
    live = pod is not None and pod["desiredStatus"] in POD_LIVE
    if live and not terminate:
        print(f"pod {pod['id']} is LIVE and BILLING. Nothing changed. Re-run with --terminate to tear it down "
              f"and settle the batch, or terminate it in the RunPod console.")
        return 0
    if dry_run:
        acts = []
        if live:
            acts.append(f"teardown {pod['id']} (terminate + spend row)")
        for r in rows:
            acts.append(f"pods row {r['pod_id']}: {r['status']} -> terminated")
        acts.append(f"batch {batch_id}: {b['status']} -> failed (note: reconciled)")
        acts.append(f"release queued/transcribing videos of {batch_id} to 'transcoded' (episodes -> 'cut')")
        print("dry run -- would: " + "; ".join(acts))
        return 0
    now = state.now_iso()
    why = "reconciled: pod not on RunPod" if not pod else (f"reconciled: pod {pod['id']} {pod['desiredStatus']}"
                                                          if not live else f"reconciled: pod {pod['id']} terminated by --terminate")
    if live:
        rc = teardown(con, pod["id"], n_videos=b.get("n_videos"), note=f"reconcile {batch_id}")
        if rc != 0:
            print(f"teardown of {pod['id']} failed; batch left as is. Check the RunPod console NOW.")
            return 1
    for r in rows:
        if live and r["pod_id"] == pod["id"]:
            continue                                        # teardown just settled that row and its spend
        con.execute("UPDATE pods SET status = 'terminated', terminated_at = ? WHERE pod_id = ? AND status != 'terminated'", (now, r["pod_id"]))
        print(f"pods row {r['pod_id']}: {r['status']} -> terminated (no spend row: uptime unknown -- check RunPod billing)")
    note = f"{why} at {now}" + (f"; was: {b['note']}" if b.get("note") else "")
    set_batch(con, batch_id, "failed", note=note[:500], finished_at=b.get("finished_at") or now)
    n = release_batch(con, batch_id, why)
    print(f"{batch_id}: {b['status']} -> failed; {n} video(s) back to 'transcoded'. No pod was created.")
    notify("info", f"{batch_id} reconciled: {why}; {n} videos released. No pod is billing for it.")
    return 0


def list_batches(con):
    rows = con.execute("SELECT batch_id, status, n_videos, audio_hours, est_usd, actual_usd, created_at, "
                       "approved_at FROM batches ORDER BY created_at").fetchall()
    if not rows:
        print("no batches")
    for r in rows:
        print(f"{r['batch_id']:<5} {r['status']:<18} {r['n_videos'] or 0:>3} videos {float(r['audio_hours'] or 0):6.2f} h "
              f"est ${float(r['est_usd'] or 0):.2f} actual {('$%.2f' % r['actual_usd']) if r['actual_usd'] is not None else '-':<6} "
              f"created {r['created_at']} approved {r['approved_at'] or '-'}")
    c = cfg()
    print(f"spend this month ${month_spend(con):.2f} of ${c['cap_usd']:.0f}; live pods: "
          + (", ".join(p["pod_id"] for p in active_pods(con)) or "none"))
    import spend                                          # noqa: E402
    print("\n".join(spend.lines(spend.summary(con))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["propose", "approve", "deny", "launch", "teardown", "reconcile", "list"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("--min-hours", type=float, default=None, help="propose: override BATCH_TARGET_HOURS")
    ap.add_argument("--limit", type=int, default=0, help="propose: at most this many videos")
    ap.add_argument("--speaker", default=None, help="propose: only rows whose speaker is in this comma list")
    ap.add_argument("--via", default="whatsapp")
    ap.add_argument("--dry-run", action="store_true", help="launch / reconcile: run the checks and lookups, change nothing")
    ap.add_argument("--terminate", action="store_true", help="reconcile: tear down a pod RunPod still reports live")
    ap.add_argument("--quiet", action="store_true", help="propose: do not send the WhatsApp message")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    con = state.connect(a.state)
    if a.cmd in ("approve", "deny", "launch", "teardown", "reconcile") and not a.id:
        ap.error(f"{a.cmd} needs an id")
    if a.cmd == "reconcile":
        return reconcile(con, a.id, a.dry_run, a.terminate)
    if a.cmd == "propose":
        return 0 if propose(con, a.min_hours, a.limit, a.quiet, a.speaker) or True else 1
    if a.cmd == "approve":
        return approve(con, a.id, a.via) or 0
    if a.cmd == "deny":
        return deny(con, a.id) or 0
    if a.cmd == "launch":
        return launch(con, a.id, a.dry_run)
    if a.cmd == "teardown":
        return teardown(con, a.id)
    list_batches(con)
    return 0


if __name__ == "__main__":
    sys.exit(main())
