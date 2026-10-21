#!/bin/bash
# ops/launchd/resume_captions_once.sh -- fire ONCE, tonight, after the YouTube
# backoff expires, and kickstart the caption trickle worker.
#
# =============================================================================
# THIS IS A ONE-SHOT. IT MUST NEVER BECOME A STANDING AUTO-RESUME.
#
# AGENTS.md rule 2: when YouTube challenges this IP the worker STOPS, writes a
# 24 h backoff, notifies Puck and waits FOR HIM. That is a human decision, made
# once per incident, with the rate limit and the evidence in front of him. A
# recurring resume -- a cron entry, a StartInterval, a KeepAlive that outlives
# the backoff -- would silently make that decision for every future 429, which
# is exactly the behaviour the rule exists to prevent, and it would do it
# invisibly because the worker's own logs would look like a normal restart.
#
# So: Puck approved resuming ONCE, after the backoff that expires at
# 2026-09-25T04:56:16Z (21:56 PDT on 2026-09-24). This script implements that
# single approved resume and then DELETES ITS OWN LAUNCH AGENT, before it does
# anything else, so that no failure mode anywhere below can leave a repeating
# job behind. If the worker is 429'd again, it stops again and waits for Puck
# again -- and resuming it is a new decision, needing a new artefact.
# =============================================================================
#
# What it does, in order:
#   1. Removes its own launchd plist from ~/Library/LaunchAgents (kept as
#      evidence under ops/launchd/fired/), so it cannot load again.
#   2. VERIFIES the backoff is actually clear by reading the `backoffs` table
#      READ-ONLY. It never clears, shortens or edits the backoff: if the row is
#      still live (a fresh 429, a clock difference, an early fire) it kickstarts
#      NOTHING, says so, and stops -- the worker stays down and Puck decides.
#   3. Only then `launchctl kickstart -k gui/<uid>/com.ytpt.captions`.
#   4. Logs every step to ~/Library/Logs/ytpt/resume_captions_once.log and
#      texts Puck the outcome.
#   5. Boots itself out of launchd.
#
# It makes no YouTube request of its own: the worker it starts does that, under
# its own one-an-hour gap and 18-per-24-h cap (pipeline/captions.py).
#
# Every path and label below can be overridden from the environment, and
# RESUME_DRY_RUN=1 skips the three side effects (kickstart, notify, bootout)
# while still exercising the disarm and the backoff check. That is how this
# artefact was rehearsed before it was installed -- a job that fires once,
# unattended, at night is not a thing to leave untested.
set -uo pipefail

REPO="${RESUME_REPO:-$HOME/ytpt}"
PY="${RESUME_PY:-$REPO/.venv/bin/python}"
LABEL="${RESUME_LABEL:-com.ytpt.resume-captions}"
WORKER="${RESUME_WORKER:-com.ytpt.captions}"
PLIST="${RESUME_PLIST:-$HOME/Library/LaunchAgents/$LABEL.plist}"
FIRED="${RESUME_FIRED:-$REPO/ops/launchd/fired}"
LOG="${RESUME_LOG:-$HOME/Library/Logs/ytpt/resume_captions_once.log}"
DRY="${RESUME_DRY_RUN:-0}"
UID_N="$(id -u)"

mkdir -p "$(dirname "$LOG")" "$FIRED"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$LOG"; }

log "=== one-shot resume fired (pid $$) ==="

# --- 1. disarm FIRST: whatever happens below, this cannot run again ----------
if [ -f "$PLIST" ]; then
  if mv "$PLIST" "$FIRED/$LABEL.plist.fired-$(date -u +%Y%m%dT%H%M%SZ)"; then
    log "disarmed: moved $PLIST into $FIRED (it can no longer load)"
  else
    log "WARNING: could not move $PLIST -- continuing, and booting out at the end"
  fi
else
  log "no plist at $PLIST (already disarmed?); continuing"
fi

# --- 2. is the backoff actually clear? (READ-ONLY; never cleared here) -------
BACKOFF="$(RESUME_PIPELINE="$REPO/pipeline" "$PY" - <<'PYEOF' 2>&1
import os, sqlite3, sys
from datetime import datetime, timezone
sys.path.insert(0, os.environ["RESUME_PIPELINE"])
import state                                   # for db_path() / the .env setting
p = state.db_path()
try:
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)     # READ-ONLY, always
    con.row_factory = sqlite3.Row
    r = con.execute("SELECT * FROM backoffs WHERE worker = 'captions'").fetchone()
    con.close()
except Exception as e:                                        # noqa: BLE001
    print(f"UNKNOWN cannot read {p}: {type(e).__name__}: {e}")
    raise SystemExit(0)
if not r:
    print("CLEAR no backoff row")
    raise SystemExit(0)
try:
    left = (datetime.fromisoformat(r["until"]) - datetime.now(timezone.utc)).total_seconds()
except (TypeError, ValueError):
    print(f"UNKNOWN unparseable until={r['until']!r}")
    raise SystemExit(0)
print((f"LIVE {left/3600:.2f} h left, until {r['until']} (set {r['set_at']}: {str(r['reason'])[:120]})")
      if left > 0 else f"CLEAR expired at {r['until']}")
PYEOF
)"
log "backoff check: $BACKOFF"

STATE="${BACKOFF%% *}"
if [ "$STATE" != "CLEAR" ]; then
  log "NOT kickstarting $WORKER: the backoff is not clear. The worker stays down; this is Puck's call (AGENTS.md rule 2)."
  if [ "$DRY" != "1" ]; then
    "$PY" "$REPO/pipeline/notify.py" error \
      "One-shot caption resume fired but did NOT start the worker: $BACKOFF. It has removed itself; the worker is still down. Resume by hand when you want it: launchctl kickstart -k gui/$UID_N/$WORKER" >>"$LOG" 2>&1
    launchctl bootout "gui/$UID_N/$LABEL" >>"$LOG" 2>&1
  else
    log "DRY RUN: would notify Puck and boot out $LABEL"
  fi
  exit 0
fi

# --- 3. the single approved resume ------------------------------------------
log "backoff is clear; kickstarting $WORKER"
if [ "$DRY" = "1" ]; then
  log "DRY RUN: would run: launchctl kickstart -k gui/$UID_N/$WORKER, then notify Puck and boot out $LABEL"
  exit 0
fi
if launchctl kickstart -k "gui/$UID_N/$WORKER" >>"$LOG" 2>&1; then
  sleep 5
  STATUS="$(launchctl list | awk -v l="$WORKER" '$3 == l {print "pid=" $1 " last_exit=" $2}')"
  log "kickstart ok; launchctl says ${STATUS:-<not listed>}"
  "$PY" "$REPO/pipeline/notify.py" info \
    "Caption worker resumed at the approved time (backoff expired: $BACKOFF). ${STATUS:-not listed}. One meeting an hour, 18 per 24 h. The one-shot has removed itself -- a future 429 will stop the worker and wait for you, as before." >>"$LOG" 2>&1
else
  log "ERROR: kickstart failed"
  "$PY" "$REPO/pipeline/notify.py" error \
    "One-shot caption resume FAILED to kickstart $WORKER (backoff was clear). It has removed itself; start the worker by hand: launchctl kickstart -k gui/$UID_N/$WORKER" >>"$LOG" 2>&1
fi

# --- 4. and out ---------------------------------------------------------------
log "=== done; booting out $LABEL ==="
launchctl bootout "gui/$UID_N/$LABEL" >>"$LOG" 2>&1
exit 0
