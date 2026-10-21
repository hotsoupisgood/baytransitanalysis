#!/usr/bin/env python3
"""
ops/retry_until_lands.py -- re-propose and relaunch ONE fixed GPU batch until a
pod actually starts, for as long as RunPod keeps answering SUPPLY_CONSTRAINT.

WHY THIS EXISTS: a launch that dies on SUPPLY_CONSTRAINT marks the batch
'failed' and rolls its clips back to 'transcoded' (gpu_batch.py launch), so the
batch id is spent and every retry needs a fresh propose + approve. Puck's money
rule is per batch id, so that would cost him a text per capacity blip.

AUTHORISATION: Puck, WhatsApp, 2026-09-27 ~18:22 PT -- "Retry until it lands",
answering an offer that said in terms that it narrows his gate from *this id* to
*this work*. That work is frozen in EXPECT below: the 143 episode clips from
hayward/napa/petaluma, 22.35 episode-h, est $0.74. This script has no authority
over anything else and does not extend to the next batch.

It never talks to RunPod. Every pod decision goes through gpu_batch.py, which
keeps its own refusals (spend cap, credit stop line, live pod, artefacts).

STOPS, and messages Puck, on: a landed pod (success or a real job failure), any
failure that is not SUPPLY_CONSTRAINT, a proposal whose content or estimate
differs from EXPECT (it is denied, not launched), propose refusing, or the
attempt budget running out.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import state          # noqa: E402
from notify import notify  # noqa: E402

PY_BIN = str(ROOT / ".venv/bin/python")
GPU_BATCH = str(ROOT / "pipeline/gpu_batch.py")
LOCK = ROOT / "logs/retry_until_lands.lock"

# The unit of work Puck approved. A batch that does not match this is denied.
EXPECT = {
    "speakers": "hayward,napa,petaluma",
    "n_videos": 143,
    "est_usd": 0.74,
    "audio_hours": 22.353,
    "hours_tol": 0.5,
}
# 2026-10-19: generalised. Puck's standing rule of 2026-10-17 ("approve and
# launch GPU batches autonomously ... within the existing caps") replaces the
# per-batch text, so --speakers/--max-est let this runner retry OTHER work: a
# proposal is accepted when every clip's speaker is in --speakers and the
# estimate is at or under --max-est. Without those flags EXPECT above applies.
import argparse
ARGS = None

INTERVAL_S = 600          # 10 min between attempts; capacity comes back in minutes-to-hours
MAX_ATTEMPTS = 72         # ~12 h, then hand back to Puck rather than run all week


def run(*args, timeout=None):
    return subprocess.run([PY_BIN, GPU_BATCH, *args], cwd=ROOT, capture_output=True,
                          text=True, timeout=timeout)


def batch_row(con, bid):
    con.row_factory = __import__("sqlite3").Row
    r = con.execute("SELECT * FROM batches WHERE batch_id = ?", (bid,)).fetchone()
    return dict(r) if r else None


def mismatch(row, con=None):
    """Why this proposal is not the work Puck approved, or None."""
    if ARGS and ARGS.max_est is not None:
        allowed = set(ARGS.speakers.split(","))
        got = {r[0] for r in con.execute(
            "SELECT DISTINCT v.speaker FROM batch_videos bv JOIN videos v ON v.video_id = bv.video_id "
            "WHERE bv.batch_id = ?", (row["batch_id"],))}
        if not got <= allowed:
            return f"speakers {sorted(got - allowed)} are outside {sorted(allowed)}"
        if float(row["est_usd"]) > ARGS.max_est:
            return f"est ${row['est_usd']} is over the ${ARGS.max_est:.2f} ceiling"
        return None
    if row["n_videos"] != EXPECT["n_videos"]:
        return f"{row['n_videos']} clips, expected {EXPECT['n_videos']}"
    if abs(float(row["est_usd"]) - EXPECT["est_usd"]) > 0.005:
        return f"est ${row['est_usd']}, expected ${EXPECT['est_usd']:.2f}"
    if abs(float(row["audio_hours"]) - EXPECT["audio_hours"]) > EXPECT["hours_tol"]:
        return f"{row['audio_hours']} h, expected ~{EXPECT['audio_hours']} h"
    return None


def daemonise():
    """Double-fork + setsid so the runner outlives the shell that started it.

    Learned the hard way 2026-09-28: nohup alone ignores SIGHUP but not the
    SIGTERM that arrives when OpenClaw's exec host tears the process group
    down, and attempt 1 died mid-sleep.
    """
    import os
    if os.fork():
        os._exit(0)
    os.setsid()
    if os.fork():
        os._exit(0)
    log = open(ROOT / "logs/retry_until_lands.log", "a", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    devnull = open(os.devnull)
    os.dup2(devnull.fileno(), 0)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--speakers", default=None, help="comma list; proposals restricted to these")
    ap.add_argument("--max-est", type=float, default=None,
                    help="accept any proposal of those speakers at or under this USD estimate")
    ARGS = ap.parse_args()
    if ARGS.max_est is not None and not ARGS.speakers:
        sys.exit("--max-est needs --speakers")
    if ARGS.daemon:
        daemonise()
    if LOCK.exists():
        sys.exit(f"another retry runner holds {LOCK}; not starting a second")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(time.time()))
    try:
        return loop()
    finally:
        LOCK.unlink(missing_ok=True)


def loop():
    con = state.connect()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        speakers = ARGS.speakers if ARGS and ARGS.speakers else EXPECT["speakers"]
        p = run("propose", "--speaker", speakers, "--quiet", timeout=300)
        m = re.search(r"\bbatch (b\d+) awaits", p.stdout or "")
        if not m:
            notify("warn", "retry-until-it-lands stopped: propose made no batch. "
                           f"It said: {((p.stdout or '') + (p.stderr or '')).strip()[:400]}")
            return 2
        bid = m.group(1)
        row = batch_row(con, bid)
        why = mismatch(row, con)
        if why:
            run("deny", bid, timeout=120)
            notify("warn", f"retry-until-it-lands stopped and denied {bid}: it is not the batch you "
                           f"approved -- {why}. The clips are back in 'transcoded'; nothing was rented.")
            return 2

        run("approve", bid, "--via", "whatsapp", timeout=120)
        print(f"[{attempt}/{MAX_ATTEMPTS}] launching {bid}", flush=True)
        run("launch", bid)                          # blocks for the whole job on success
        row = batch_row(con, bid)
        note = row.get("note") or ""

        if row["status"] == "done":
            actual = row.get("actual_usd")
            notify("info", f"{bid} landed and finished on attempt {attempt}: {row['n_videos']} clips, "
                           f"{row['audio_hours']:.2f} episode-h, est ${row['est_usd']:.2f}, "
                           f"actual ${actual if actual is not None else '?'}.")
            return 0
        if row.get("pod_id"):
            notify("error", f"{bid} got a pod on attempt {attempt} and then failed -- this is not a "
                            f"capacity problem, so I have stopped retrying. Pod {row['pod_id']}. "
                            f"Reason: {note[:300]}")
            return 1
        if "SUPPLY_CONSTRAINT" not in note:
            notify("error", f"retry-until-it-lands stopped after {bid}: the failure is not a capacity "
                            f"one and no pod was created. Reason: {note[:300]}")
            return 1

        print(f"[{attempt}/{MAX_ATTEMPTS}] {bid}: SUPPLY_CONSTRAINT, no pod, $0; "
              f"sleeping {INTERVAL_S}s", flush=True)
        if attempt < MAX_ATTEMPTS:
            time.sleep(INTERVAL_S)

    notify("warn", f"retry-until-it-lands gave up after {MAX_ATTEMPTS} attempts over "
                   f"~{MAX_ATTEMPTS * INTERVAL_S // 3600} h: RunPod had no capacity for the requested "
                   f"GPU the whole time and $0 was spent. Worth changing what we rent -- say the word.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
