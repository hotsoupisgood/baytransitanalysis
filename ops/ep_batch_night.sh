#!/bin/zsh
# ops/ep_batch_night.sh -- the episode-clip GPU batch chain that follows one
# night's YouTube shift. Generalised from ops/ep_batch_1020.sh, which ran clean
# on 2026-10-20 (b50, done 07:22 UTC). Started by ops/yt_night.sh ONLY on a
# clean shift: any 429 / 403 / bot-check aborts the shift and this never runs.
#
# ONE DELIBERATE DIFFERENCE FROM ep_batch_1020.sh, AND IT IS THE POINT OF THIS
# FILE: it stops after QA. Puck, 2026-10-20 09:22 PDT -- "start the yt
# transcription in the background we just wont put it into production". So
# there is no `make load-episodes`, no pipeline/redact.py, no stance pass and
# no site export anywhere below, and no recount of public turns either (that
# reads corpus.sqlite, which only a load would change). Today's frozen dataset
# stays the published one; these transcripts land in output/whisper and wait.
# QA still runs, because a FAIL is the one thing Puck must hear about tonight
# rather than next week (AGENTS.md: "Any QA FAIL (do not load those
# transcripts)") -- and not loading is already the default here.
#
# Authority to spend: Puck's standing rule of 2026-10-17, "approve and launch
# GPU batches autonomously ... within the existing caps". Live headroom read
# 2026-10-20 16:30 UTC: RunPod $16.12 recorded of the $31.00 stop line
# ($14.88 left); month $11.12 of the $40.00 MONTHLY_SPEND_CAP_USD. The three
# nights together are ~$1.23 estimated / ~$0.85 expected actual, so ~6 % of
# what is left on the stop line. gpu_batch.py keeps its own refusals on top of
# that (spend cap, credit stop line, live pod, artefacts) and this script
# cannot talk to RunPod any other way.
#
# Config from the environment (set once, in ops/nights_1021_1023.sh):
#   EPB_LABEL    names the logs, e.g. 1021
#   EPB_MIN_H    floor in episode-hours: the chain's own pre-check AND, via
#                BATCH_TARGET_HOURS below, propose's threshold
#   EPB_MAX_EST  USD ceiling on the proposal
#   EPB_DRY_RUN  1 = run every guard, report what would be proposed, rent
#                nothing. ALWAYS use this to test a change to this file.
#
# WHY EPB_DRY_RUN EXISTS, learned 2026-10-20 16:37 UTC at a cost of $0.40:
# this chain was smoke-tested by running it with a low floor, on the reasoning
# that the propose queue was empty and it would stop at guard 3. The queue had
# been empty seven minutes earlier. In between, another worker's full-ASR run
# finished and put 74 episode clips (10.55 h, fremont + samtrans) into
# 'transcoded', so every guard passed and the chain did exactly what it is for:
# it rented a GPU (b51, est $0.40, inside every cap, legitimate work -- just
# hours earlier than anyone intended). A script whose purpose is to spend money
# has no safe no-op path, so it needs an explicit one. Guard state is live and
# a reading minutes old is not a prediction.
#
# WHY retry_until_lands.py DOES THE WORK: a launch that dies on RunPod
# SUPPLY_CONSTRAINT marks the batch 'failed' and rolls its clips back to
# 'transcoded', so the batch id is spent and every retry needs a fresh
# propose + approve -- b45 and b47 both died that way on 2026-10-19.
# ops/retry_until_lands.py is the tested runner for exactly that: propose ->
# check -> approve -> launch, re-proposing every 10 min for up to 12 h while
# the failure stays SUPPLY_CONSTRAINT, stopping the moment it is anything else.
#
# WHY BATCH_TARGET_HOURS IS EXPORTED: retry_until_lands.py calls
# `propose --speaker <roster> --quiet` with no --min-hours, so propose's floor
# is BATCH_TARGET_HOURS, which is 15.0 in the live settings. Nights 2 and 3 are
# ~4.6 h and ~1.4 h of clips, so an un-lowered floor would make propose refuse
# and the runner notify "propose made no batch" -- a false alarm, not a batch.
# state.setting reads os.environ first and state.load_env uses
# os.environ.setdefault, documented as "never overrides the real env"
# (pipeline/state.py:223-237), so exporting it here wins and `.env` is never
# read for it, written or touched.
#
# THREE GUARDS AGAINST THE FROZEN SIDE CORPUS. The propose queue permanently
# holds frozen Hasan/Tucker 'show' rows -- right now hasanabi 4 (1.15 h) and
# tucker 3 (2.25 h) in status='transcoded' with in_frame=1, and they are the
# ONLY such rows at arming time. An unrestricted `propose` would sweep them
# into a batch, which is how b36 nearly went out (fixed then with `deny`). So:
#   1. --speakers is the meeting-body roster, derived live below. hasanabi and
#      tucker are kind='show' only and own no kind='meeting' row, so they
#      cannot appear in a roster built that way -- the exclusion is structural
#      rather than a list someone has to keep correct.
#   2. retry_until_lands.py independently re-checks the proposal's speakers
#      against that same roster and DENIES any batch that strays outside it,
#      renting nothing (ops/retry_until_lands.py:71-82).
#   3. --max-est: a proposal over the ceiling is denied too, by the same check.
# A live derivation has its own failure mode -- an empty roster would mean an
# unrestricted propose -- so it is sanity-checked before use.
#
# Never deletes anything. Never writes to plan/, .env or the site.
set -u
cd "$HOME/ytpt" || exit 1
PY=.venv/bin/python

: ${EPB_LABEL:?set EPB_LABEL}
: ${EPB_MIN_H:?set EPB_MIN_H}
: ${EPB_MAX_EST:?set EPB_MAX_EST}
EPB_DRY_RUN=${EPB_DRY_RUN:-0}

LOG=output/logs/ep_batch_${EPB_LABEL}.log
LOCK=logs/retry_until_lands.lock
STATE="$HOME/.local/share/ytpt/state.sqlite"
mkdir -p output/logs

export BATCH_TARGET_HOURS=$EPB_MIN_H

say() { print -r -- "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> "$LOG" }

# One scalar out of the live DB. The SQL goes in as argv, NOT interpolated into
# the Python source.
#
# WHY, and this one had already cost a night: ops/ep_batch_1020.sh built the
# statement by substituting into a '''...''' literal. Every query that ENDS in
# a quoted SQL literal then closes that literal and the Python triple-quote
# with four consecutive quotes --
#     '''SELECT status FROM batches WHERE batch_id='b50''''
# -- which Python reads as a string ending at the first three, followed by a
# stray quote: SyntaxError, no output, and `2>/dev/null` hid it. So
# `q "SELECT status FROM batches WHERE batch_id='b50'"` did not fail
# occasionally, it failed ALWAYS. In last night's log that is exactly what
# happened: "b50 final status: " then "b50 did not finish ('') -- no QA, no
# corpus load. Tell Puck." while b50 had in fact completed, 49 clips, actual
# $0.18. The same read on b51 reproduced it at 17:07 UTC today.
#
# Passing the statement as sys.argv[1] removes the quoting question entirely.
# The retry is kept on top of that for genuine WAL contention: state.sqlite is
# in WAL mode and a mode=ro reader can transiently fail against a writer
# mid-checkpoint, which is plausible precisely when a launch is committing its
# last updates. An empty answer is "ask again", and callers that still get one
# treat it as "not safe to proceed" rather than as a value.
q() {
  local out=""
  for _ in 1 2 3 4 5; do
    out=$($PY -I -c "
import sqlite3, sys
con = sqlite3.connect('file:$STATE?mode=ro', uri=True)
print(next(iter(con.execute(sys.argv[1])))[0])
" "$1" 2>/dev/null)
    [[ -n $out ]] && break
    sleep 2
  done
  print -r -- "$out"
}

# In a dry run the two polling guards take one pass instead of hours: the point
# is to see what they report, not to wait out a sibling worker.
if [[ ${EPB_DRY_RUN} == 1 ]]; then
  SETTLE_TRIES=1; POD_TRIES=1
else
  SETTLE_TRIES=1080; POD_TRIES=60
fi

say "=== night ${EPB_LABEL} batch chain starting (pid $$), floor ${EPB_MIN_H} h, ceiling \$${EPB_MAX_EST}$([[ ${EPB_DRY_RUN} == 1 ]] && print -n ', DRY RUN: rents nothing') ==="

# --- the roster, derived and then checked -------------------------------
BODIES=$($PY -I -c "
import sqlite3
con = sqlite3.connect('file:$STATE?mode=ro', uri=True)
print(','.join(r[0] for r in con.execute(
    \"SELECT DISTINCT speaker FROM videos WHERE kind='meeting' ORDER BY speaker\")))
" 2>/dev/null)
nbodies=$(print -r -- "${BODIES}" | tr ',' '\n' | grep -c . )
if [[ ${nbodies:-0} -lt 20 ]]; then
  say "STOP: the meeting-body roster came back with ${nbodies:-0} bodies (expected ~31). An empty or short roster would let propose run unrestricted over the frozen show rows. Nothing proposed -- tell Puck."
  exit 1
fi
if print -r -- ",${BODIES}," | grep -qE ",(hasanabi|tucker),"; then
  say "STOP: the roster contains a frozen show body (${BODIES}). Nothing proposed -- tell Puck."
  exit 1
fi
say "roster: ${nbodies} meeting bodies, no show bodies"

# --- guard 1: no batch running anywhere (one pod at a time) -------------
# Checking "the last batch is done" was not enough on 2026-10-19: b42 and b43
# were launched 2 min apart and two pods billed side by side for 1.4 h.
say "waiting for every batch to settle"
for i in $(seq 1 ${SETTLE_TRIES}); do          # 20 s apart, up to 6 h
  [[ "$(q "SELECT COUNT(*) FROM batches WHERE status IN ('running','launching')")" == "0" ]] && break
  [[ $i -lt ${SETTLE_TRIES} ]] && sleep 20
done
nrun=$(q "SELECT COUNT(*) FROM batches WHERE status IN ('running','launching')")
if [[ "$nrun" != "0" ]]; then
  say "GIVING UP: ${nrun} batch(es) still running. Nothing proposed."; exit 1
fi

# --- guard 2: RunPod really has no live pod ----------------------------
# Polled, not single-shot, and this is the guard that matters most tonight.
# Guard 1 only sees `batches`; active_pods() counts pods in
# 'creating','running','ORPHAN' (pipeline/gpu_batch.py:89-90), a wider set, so
# a sibling pod can still be creating or tearing down after its batch row has
# left 'running'. Another worker is running a full-ASR -> episode batch across
# these same nights, and the three chains test "busy" differently, so a
# one-shot check here would hand the night to whichever process looked first.
podcount() { $PY dashboard/status.py --json 2>/dev/null | $PY -I -c "
import json,sys; print(len(json.load(sys.stdin).get('pods_active') or []))" 2>/dev/null }
pods=""
for i in $(seq 1 ${POD_TRIES}); do  # 60 x 3 min = 3 h
  pods=$(podcount)
  [[ "${pods:-}" == "0" ]] && break
  say "guard 2: ${pods:-?} active pod(s) (or status.py unreadable) -- re-checking in 3 min ($i/${POD_TRIES})"
  [[ $i -lt ${POD_TRIES} ]] && sleep 180
done
if [[ "${pods:-1}" != "0" ]]; then
  say "STOP: still ${pods:-?} active pod(s) -- another worker holds the GPU. Nothing proposed. The clips keep waiting; the next night's chain will sweep them in."
  exit 1
fi
say "pod free: no running batch, pods_active empty"

# --- guard 3: is there enough work? ------------------------------------
# Measured exactly the way propose measures it: in-frame 'transcoded' rows,
# restricted to the meeting bodies, SUM(transcribed_s).
ready=$($PY -I -c "
import sqlite3
b = '''$BODIES'''.split(',')
con = sqlite3.connect('file:$STATE?mode=ro', uri=True)
qs = ','.join('?' * len(b))
h = con.execute(f\"SELECT COALESCE(SUM(transcribed_s),0)/3600.0 FROM videos \"
                f\"WHERE status='transcoded' AND in_frame=1 AND speaker IN ({qs})\", b).fetchone()[0]
n = con.execute(f\"SELECT COUNT(*) FROM videos \"
                f\"WHERE status='transcoded' AND in_frame=1 AND speaker IN ({qs})\", b).fetchone()[0]
print(f'{h:.2f} {n}')
" 2>/dev/null)
rh=${ready%% *}; rn=${ready##* }
say "episode clips ready: ${rn:-?} clips, ${rh:-?} h (floor ${EPB_MIN_H} h)"
if [[ -z "${rh:-}" ]] || [[ $(print -r -- "$rh < $EPB_MIN_H" | bc -l 2>/dev/null) == "1" ]]; then
  say "under the ${EPB_MIN_H} h floor -- nothing proposed, exiting quietly. The clips keep waiting."
  exit 0
fi

# --- guard 3b: everything in the queue must be an episode clip ---------
# propose filters on `in_frame = 1 AND status = 'transcoded'` and the speaker
# list ONLY -- it does NOT filter on kind (pipeline/gpu_batch.py:142-149). So a
# 'show' or 'validation' row parked under a listed body would join an episode
# batch. Several bodies (bart, petaluma, richmond ...) do own 'validation'
# rows, so this turns a silent sweep into a loud stop.
stray=$($PY -I -c "
import sqlite3
b = '''$BODIES'''.split(',')
con = sqlite3.connect('file:$STATE?mode=ro', uri=True)
qs = ','.join('?' * len(b))
r = con.execute(f\"SELECT kind, COUNT(*) FROM videos WHERE status='transcoded' AND in_frame=1 \"
                f\"AND speaker IN ({qs}) AND kind <> 'episode' GROUP BY kind\", b).fetchall()
print('; '.join(f'{k}:{n}' for k, n in r))
" 2>/dev/null)
if [[ -n "${stray:-}" ]]; then
  say "STOP: non-episode rows in the propose queue under the meeting bodies (${stray}). A batch would sweep them in. Nothing proposed -- tell Puck."
  exit 1
fi
say "queue is all kind='episode'"

# --- guard 4: a retry runner must not already be live ------------------
# One global lock, so last night's runner still retrying SUPPLY_CONSTRAINT
# blocks tonight rather than racing it.
if [[ -e $LOCK ]]; then
  say "STOP: $LOCK exists -- a retry runner is already live. Not starting a second."
  exit 1
fi

# --- dry run stops here, one step short of spending ---------------------
if [[ ${EPB_DRY_RUN} == 1 ]]; then
  say "DRY RUN: every guard passed. It would now run retry_until_lands.py --daemon --speakers <${nbodies} bodies> --max-est ${EPB_MAX_EST} (BATCH_TARGET_HOURS=${BATCH_TARGET_HOURS}) over ${rn} clips / ${rh} h. Renting nothing. === night ${EPB_LABEL} dry run done ==="
  exit 0
fi

# --- propose / approve / launch, retrying only on SUPPLY_CONSTRAINT ----
say "starting retry_until_lands.py --daemon --speakers <${nbodies} bodies> --max-est ${EPB_MAX_EST} (BATCH_TARGET_HOURS=${BATCH_TARGET_HOURS})"
before=$(q "SELECT COALESCE(MAX(CAST(SUBSTR(batch_id,2) AS INTEGER)),0) FROM batches")
say "highest batch id before: b${before}"
$PY ops/retry_until_lands.py --daemon --speakers "$BODIES" --max-est "$EPB_MAX_EST"
say "runner detached; its own log is logs/retry_until_lands.log"

# --- watch for the batch to settle -------------------------------------
# The runner is detached, so this watches the DB rather than a child process.
BID=""
for i in $(seq 1 2400); do          # up to ~13.3 h: 12 h of retries + the job
  nb=$(q "SELECT COALESCE(MAX(CAST(SUBSTR(batch_id,2) AS INTEGER)),0) FROM batches")
  if [[ "${nb:-0}" -gt "${before:-0}" ]]; then
    BID="b${nb}"
    st=$(q "SELECT status FROM batches WHERE batch_id='${BID}'")
    [[ "$st" == "done" || "$st" == "denied" ]] && { say "$BID reached '$st'"; break; }
    # 'failed' on SUPPLY_CONSTRAINT just means the runner will propose a new id
    if [[ "$st" == "failed" ]]; then
      note=$(q "SELECT COALESCE(note,'') FROM batches WHERE batch_id='${BID}'")
      case "$note" in
        *SUPPLY_CONSTRAINT*) : ;;                       # runner retries; keep watching
        *) say "$BID failed and it is NOT a capacity problem: ${note}"; break ;;
      esac
    fi
  fi
  [[ ! -e $LOCK ]] && [[ -n "$BID" ]] && { say "runner has exited; last batch $BID"; break; }
  sleep 20
done

if [[ -z "$BID" ]]; then
  say "no batch appeared. Check logs/retry_until_lands.log -- propose may have refused. Nothing rented."
  exit 0
fi

st=$(q "SELECT status FROM batches WHERE batch_id='${BID}'")
say "$BID final status: ${st:-<unreadable>}"
$PY pipeline/gpu_batch.py list 2>&1 | tail -4 >> "$LOG"
if [[ -z "$st" ]]; then
  # Distinguished from a real failure on purpose: the batch may well have
  # finished, so do not say it did not. `gpu_batch.py list` above is in the log
  # and shows the truth.
  say "could not read $BID's status after 5 tries -- no QA run. See the list output above for what it actually did. Tell Puck."
  exit 1
fi
if [[ "$st" != "done" ]]; then
  say "$BID did not finish ('$st') -- no QA. Tell Puck."
  exit 1
fi

# --- QA, and then STOP. No load, no redact, no stance, no export -------
say "running make qa"
make qa > "output/logs/qa_after_${BID}.log" 2>&1
grep -E "OK|WARN|FAIL" "output/logs/qa_after_${BID}.log" | tail -3 >> "$LOG"
if grep -qE " [1-9][0-9]* FAIL" "output/logs/qa_after_${BID}.log"; then
  say "QA FAIL after $BID -- escalate to Puck. (Nothing was going to be loaded anyway: not for production.)"
  exit 1
fi
say "0 QA FAIL. ${BID}'s transcripts are in output/whisper and stay there: not loaded, not redacted, not stance-coded, not exported, by instruction."
say "=== night ${EPB_LABEL} chain done for ${BID} ==="
