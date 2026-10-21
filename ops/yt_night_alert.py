#!/usr/bin/env python3
"""
ops/yt_night_alert.py -- tell Puck on WhatsApp if a night's YouTube shift
aborts. Sends at most ONE message per night, then exits. Generalised from
ops/yt_1020_alert.py, which is the version that ran on 2026-10-20; the three
nights of 2026-10-21..23 differ only in log, window and expected count, so
they share this file rather than three hand-edited copies.

    ops/yt_night_alert.py --label 1021 --log output/logs/yt_1021_shift.log \\
        --start-epoch 1792566000 --stop-epoch 1792591200 --expected 12

WHY: ops/yt_night.sh guards itself properly -- on a bot check, a 429/403, a
nonzero episodes.py exit or low disk it writes a "STOP: ... Tell Puck." line,
aborts, and runs no batch chain. But nothing reads that line.
dashboard/status.py --new-alerts does not look at the shift log, and
yt_night.sh never calls pipeline/notify.py, so the abort is silent until
someone opens the log. AGENTS.md requires a bot check or 429 to reach Puck
promptly.

This is the reader. It is NOT a retrier and NOT a supervisor: it never signals,
restarts or inspects the shift's processes, and it makes no YouTube request. It
reads one log file and sends one message.

WHY IT IGNORES "HOLD:": when an earlier night ends on an unresolved STOP,
yt_night.sh holds the later nights and logs "HOLD: ... makes NO YouTube
request". Puck already has the message about the actual refusal; a second and
third copy of the same news on the following nights would be noise, so only
"STOP:" alerts. The HOLD is in the log for whoever reads it.

WHY n MEETINGS IS A SUM, NOT A LINE COUNT: a "done:" line is one *run*, not one
meeting -- `done: {meetings} meetings, ...` (pipeline/episodes.py:569), appended
to the shift log by yt_night.sh. The clean 2026-10-19 shift logged 6 "done:"
lines for its 12 meetings (1+3+4+2+1+1), so counting lines would report "3 of
12" after a complete three-run shift. The meeting counts are summed instead,
and the line count is only a fallback if the format ever changes.

WHY THE STOP MATCH TOLERATES A PREFIX: yt_night.sh's say() writes
"<ISO-8601 UTC> STOP: ...", so the line does not literally begin with "STOP:".
The timestamp is also the stop's own clock, which beats "now" by up to one poll
interval, so it is what gets reported in PDT.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
NOTIFY = ROOT / "pipeline/notify.py"
PY = ROOT / ".venv/bin/python"

PACIFIC = ZoneInfo("America/Los_Angeles")
POLL_S = 60
MAX_STOP_CHARS = 200

TS = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00"
STOP_RE = re.compile(rf"^(?:({TS})\s+)?(STOP:.*)$")
DONE_RE = re.compile(r"\bdone:\s*(\d+)\s+meetings\b")
COMPLETE_MARK = "shift complete"
HOLD_MARK = "HOLD:"

ARGS = None
ALERT_LOG = None


def say(msg):
    with open(ALERT_LOG, "a", buffering=1) as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime())} {msg}\n")


def read_lines():
    try:
        return Path(ARGS.log).read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    except OSError as e:
        say(f"cannot read {ARGS.log}: {e}")
        return []


def scan(lines):
    """-> (stop_ts, stop_text, meetings) for the first STOP that no later
    'shift complete' supersedes, else (None, None, meetings)."""
    stop_i = stop_ts = stop_text = None
    for i, line in enumerate(lines):
        m = STOP_RE.match(line.strip())
        if m:
            stop_i, stop_ts, stop_text = i, m.group(1), m.group(2).strip()
            break
    if stop_i is None:
        return None, None, count_meetings(lines)
    # A completion after the stop means the shift got there anyway -- no alarm.
    if any(COMPLETE_MARK in l for l in lines[stop_i + 1:]):
        say("a 'shift complete' line follows the STOP -- superseded, no alert.")
        return None, None, count_meetings(lines)
    return stop_ts, stop_text, count_meetings(lines[: stop_i + 1])


def count_meetings(lines):
    hits = [int(m.group(1)) for m in (DONE_RE.search(l) for l in lines) if m]
    if hits:
        return sum(hits)
    # Format changed: fall back to the line count rather than claiming zero.
    return sum(1 for l in lines if "done:" in l)


def pdt_hhmm(stop_ts):
    if stop_ts:
        try:
            return datetime.strptime(stop_ts, "%Y-%m-%dT%H:%M:%S+00:00").replace(
                tzinfo=timezone.utc).astimezone(PACIFIC).strftime("%H:%M")
        except ValueError:
            pass
    return datetime.now(PACIFIC).strftime("%H:%M")


def compose(stop_ts, stop_text, meetings):
    text = stop_text[:MAX_STOP_CHARS].rstrip()
    sep = "" if text.endswith((".", "!", "?")) else "."
    return (f"YouTube shift (night {ARGS.label}) stopped at {pdt_hhmm(stop_ts)} PDT: {text}{sep} "
            f"{meetings} of {ARGS.expected} meetings downloaded. "
            "No retries, no GPU batch tonight, and the later nights will hold "
            "themselves until you say go.")


def send(message):
    """One message through pipeline/notify.py. notify.py never raises: on a
    delivery failure it queues the message in notify_outbox.jsonl and exits 1,
    so a nonzero exit means queued, not lost."""
    cmd = [str(PY), str(NOTIFY), "error", message]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        say(f"notify.py could not run: {e}")
        return False
    out = (r.stdout + r.stderr).strip()
    say(f"notify.py exit {r.returncode}{': ' + out[:300] if out else ''}")
    return r.returncode == 0


def flush():
    """Resend the queued copy -- the same one message, not a second one."""
    try:
        r = subprocess.run([str(PY), str(NOTIFY), "--flush"], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        say(f"--flush could not run: {e}")
        return False
    say(f"--flush exit {r.returncode}: {(r.stdout + r.stderr).strip()[:300]}")
    return r.returncode == 0


def daemonise():
    """Double-fork + setsid: plain nohup survives SIGHUP but not the SIGTERM an
    exec host sends when it tears a process group down (ops/retry_until_lands.py,
    2026-09-28). macOS has no setsid(1), so it is done here."""
    if os.fork():
        os._exit(0)
    os.setsid()
    if os.fork():
        os._exit(0)
    # Bind both handles to names. open(...).fileno() inline lets the temporary
    # file object be freed -- closing the fd -- before dup2 runs, which is EBADF
    # and is what killed ops/yt_1020_watchdog.py's first start on 2026-10-19.
    log = open(ALERT_LOG, "a", buffering=1)
    devnull = open(os.devnull)
    os.dup2(devnull.fileno(), 0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    return log, devnull


def main():
    global ARGS, ALERT_LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--log", required=True, help="the shift log to watch")
    ap.add_argument("--start-epoch", type=int, required=True)
    ap.add_argument("--stop-epoch", type=int, required=True)
    ap.add_argument("--expected", type=int, required=True)
    ap.add_argument("--no-daemon", action="store_true", help="stay in the foreground (tests)")
    ARGS = ap.parse_args()
    ALERT_LOG = ROOT / f"output/logs/yt_{ARGS.label}_alert.log"
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)

    if ARGS.stop_epoch <= ARGS.start_epoch:
        sys.exit("--stop-epoch must be after --start-epoch")

    handles = None
    if not ARGS.no_daemon:
        handles = daemonise()                  # noqa: F841 -- keeps fds 0-2 open
    (ROOT / f"output/logs/yt_{ARGS.label}_alert.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    say(f"alert armed (pid {os.getpid()}, setsid); watching {ARGS.log} "
        f"{datetime.fromtimestamp(ARGS.start_epoch, timezone.utc):%Y-%m-%dT%H:%MZ}"
        f"-{datetime.fromtimestamp(ARGS.stop_epoch, timezone.utc):%H:%MZ}, "
        f"expecting {ARGS.expected} meetings")

    while time.time() < ARGS.start_epoch:
        time.sleep(min(300, max(1, ARGS.start_epoch - time.time())))

    say("start reached; polling every 60 s")
    while time.time() < ARGS.stop_epoch:
        lines = read_lines()
        stop_ts, stop_text, meetings = scan(lines)
        if stop_text:
            message = compose(stop_ts, stop_text, meetings)
            say(f"STOP seen -> {message}")
            if not send(message):
                for _ in range(3):             # queued; try to get it out tonight
                    if time.time() >= ARGS.stop_epoch or flush():
                        break
                    time.sleep(POLL_S)
            say("alert sent; exiting.")
            return 0
        if any(HOLD_MARK in l for l in lines):
            say("night held because an earlier night aborted; Puck already knows. Exiting silently.")
            return 0
        if any(COMPLETE_MARK in l for l in lines):
            say(f"shift complete ({meetings} meetings); nothing to report, exiting.")
            return 0
        # Never sleep past the window: a plain sleep(POLL_S) leaves the process
        # alive for up to a minute after its stop epoch, which is harmless for
        # an 8 h window and confusing for a short one.
        time.sleep(max(1.0, min(POLL_S, ARGS.stop_epoch - time.time())))

    say("window closed with no STOP line. Exiting silently.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
