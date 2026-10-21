#!/bin/zsh
# ops/yt_night.sh -- one night's YouTube shift, generalised from ops/yt_1020.sh
# so the three nights of 2026-10-21..23 run ONE tested code path instead of
# three hand-edited copies. Armed by ops/nights_1021_1023.sh, which holds the
# per-night numbers and the reasoning behind them.
#
# Puck, 2026-10-20 09:22 PDT: "start the yt transcription in the background we
# just wont put it into production". So each night downloads and cuts clips and
# its chain transcribes them on the GPU -- and nothing is loaded into the
# corpus, redacted, stance-coded or exported. Today's frozen dataset stays the
# published one. ops/ep_batch_night.sh is the chain, and it stops after QA.
#
# Config, all from the environment (no flags, so a typo is a missing-variable
# error under `set -u` rather than a silently different shift):
#   YT_LABEL      e.g. 1021 -- names every log this night writes
#   YT_TARGET     epoch seconds; no YouTube request is made before it
#   YT_RUNS       "A:bart:4 B:bart:4 C:bart:4" -- run-label:body:max, in order.
#                 Several entries may share a run label, which is how a run
#                 covers more than one body ("A:dalycity:1 A:sanrafael:2"); the
#                 25-minute gap falls where the label CHANGES, exactly as
#                 ops/yt_1020.sh's run C did vallejo 3 then bart 1 back to back.
#   YT_CHAIN      the post-shift chain to hand off to, or "" for none
#   YT_PRIOR_LOGS prior nights' shift logs; an unresolved STOP in any of them
#                 holds this night (see below)
#
# THE REFUSAL GATE IS COPIED VERBATIM from ops/yt_1020.sh, which ran clean, and
# from yt_1019b.sh before it -- exit code FIRST, text grep (incl. 403) second:
#   episodes.py declares a bot check either on BOT_CHECK text (sign in to
#   confirm | not a bot | HTTP Error 429 | Too Many Requests) OR on two
#   consecutive HTTP 403s (pipeline/episodes.py:144). In the 403 case the log
#   carries only "HTTP Error 403", which a 429-only grep misses, so the exit
#   code is the primary gate: episodes.py returns EXIT_BOT_CHECK = 3
#   (pipeline/episodes.py:73,572) for both shapes.
#
# WHAT IS NEW HERE, AND WHY: three nights are armed at once, so for the first
# time a night can start AFTER an earlier night was refused. AGENTS.md rule 2 is
# "stop the downloader, tell Puck, wait for him" -- not "wait 24 h and try 12
# more". A bot check is this Mac being challenged, and that condition does not
# expire on a calendar. So before touching YouTube this script reads the earlier
# nights' logs and, on a STOP that no later "shift complete" supersedes, writes
# a HOLD line and exits having made no request. HOLD rather than STOP because
# ops/yt_night_alert.py alerts on STOP: Puck already has the message about the
# real refusal and does not need a second one per night. Clearing it is manual
# and deliberate: once he says go, re-arm the night.
#
# Each run is <= 4 meetings and runs are >= 25 min apart (AGENTS.md rule 2:
# "at most 2-3 at a time" is about concurrency; episodes.py is serial within a
# run, one meeting at a time, with yt-dlp's own 5-15 s inter-request sleeps).
# A run only requests a meeting whose whole audio is absent, and `--max N` is
# LIMIT N over meetings ordered meeting_date DESC (pipeline/episodes.py:476),
# so the per-night total is a ceiling that cannot be exceeded, only under-used.
# Every run names a body: never a bare --max.
#
# Hard stop: if any run exits nonzero, or its log shows 429 / 403 / bot-check,
# the shift aborts, leaves a STOP line for the supervisor, and runs NO batch
# chain. It never retries a refused download and never uses cookies.
set -u
cd "$HOME/ytpt" || exit 1
PY=.venv/bin/python

: ${YT_LABEL:?set YT_LABEL}
: ${YT_TARGET:?set YT_TARGET}
: ${YT_RUNS:?set YT_RUNS}
YT_CHAIN=${YT_CHAIN:-}
YT_PRIOR_LOGS=${YT_PRIOR_LOGS:-}

LOG=output/logs/yt_${YT_LABEL}_shift.log
mkdir -p output/logs

say() { print -r -- "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> "$LOG" }

# How many meetings this night may request, summed from the run spec. Reported
# so the log says what the ceiling was, and used by nothing else.
planned=0
for spec in ${=YT_RUNS}; do planned=$(( planned + ${spec##*:} )); done

say "=== night ${YT_LABEL} armed (pid $$): ${planned} meetings max, runs: ${YT_RUNS} ==="

# --- wait for the start boundary ----------------------------------------
# YT_TARGET is an epoch computed once, at arming time, by nights_1021_1023.sh
# with TZ=America/Los_Angeles -- so DST and "which midnight" were settled then,
# not re-derived here in whatever timezone the process wakes up in.
#
# THE WAIT COMES FIRST, BEFORE EVERY GATE BELOW, and that ordering is the whole
# point: all three nights are armed in one go, so night 3's process is alive
# ~62 h before it fires. A gate evaluated at arming time answers a question
# nobody asked -- the first version of this script checked the earlier nights'
# logs up here and had night 1022 cheerfully certify night 1021 "clean" two
# hours before night 1021 existed. Disk is the same: 28 Gi free tonight says
# nothing about Thursday. Every gate below reads the world as it is when the
# night actually starts.
say "start at $(date -u -r ${YT_TARGET} +%Y-%m-%dT%H:%M:%SZ) ($(TZ=America/Los_Angeles date -r ${YT_TARGET} +'%Y-%m-%d %H:%M %Z'))"
while :; do
  now=$(date -u +%s)
  [[ $now -ge ${YT_TARGET} ]] && break
  left=$(( YT_TARGET - now ))
  [[ $left -gt 300 ]] && sleep 300 || sleep $left
done
say "start time reached"

# --- hold if an earlier night was refused -------------------------------
# awk, not grep: the question is whether a "shift complete" FOLLOWS the first
# STOP, which a line-matcher cannot answer.
for plog in ${=YT_PRIOR_LOGS}; do
  [[ -f $plog ]] || continue
  if ! awk 'f==0 && /STOP:/ {f=NR} f && NR>f && /shift complete/ {ok=1}
            END {exit (f && !ok) ? 1 : 0}' "$plog"; then
    stopline=$(grep -m1 "STOP:" "$plog")
    say "HOLD: ${plog} ends on an unresolved STOP (${stopline}). AGENTS.md rule 2 says wait for Puck, so night ${YT_LABEL} makes NO YouTube request and runs no chain. Re-arm it once he says go."
    exit 0
  fi
done
[[ -n ${YT_PRIOR_LOGS} ]] && say "prior nights clean: ${YT_PRIOR_LOGS}"

# --- disk guard ---------------------------------------------------------
free_gb=$(df -g . | awk 'NR==2 {print $4}')
if [[ ${free_gb:-0} -lt 5 ]]; then
  say "STOP: only ${free_gb} Gi free on the volume; refusing to fetch audio. Tell Puck."
  exit 1
fi
say "disk ok: ${free_gb} Gi free"

refused() {
  grep -qiE "429|Too Many Requests|Sign in to confirm|not a bot|HTTP Error 403" "$1" 2>/dev/null
}

run() {  # run <label> <body> <max>
  local label=$1 body=$2 max=$3 rc=0
  local l="output/logs/yt_${YT_LABEL}_${label}_${body}.log"
  say "run ${label}: ${body} --max ${max} -> ${l}"
  $PY pipeline/episodes.py --bodies "$body" --max "$max" > "$l" 2>&1
  rc=$?
  grep -E "done:" "$l" >> "$LOG"
  # EXIT_BOT_CHECK = 3; any nonzero exit stops the shift rather than guessing.
  if [[ $rc -ne 0 ]]; then
    say "STOP: episodes.py exited ${rc} on ${body} (3 = bot check / rate limit) -- shift aborted, no retry, no batch chain. Tell Puck."
    return 1
  fi
  if refused "$l"; then
    say "STOP: YouTube refused in ${l} -- shift aborted, no retry, no batch chain. Tell Puck."
    return 1
  fi
  return 0
}

# --- the runs -----------------------------------------------------------
# The 25-minute gap falls where the run label changes, not between every
# episodes.py call: bodies sharing a label are one run.
prev=""
for spec in ${=YT_RUNS}; do
  label=${spec%%:*}
  rest=${spec#*:}
  body=${rest%%:*}
  max=${rest##*:}
  if [[ -n $prev && $label != $prev ]]; then
    say "run ${prev} done; sleeping 25 min before run ${label}"
    sleep 1500
  fi
  prev=$label
  run "$label" "$body" "$max" || exit 1
done

say "shift complete: up to ${planned} meetings requested for night ${YT_LABEL}. No further YouTube tonight."

# --- hand off to the GPU batch chain ------------------------------------
# Only on a clean shift. The chain does its own pod-idle, spend and queue
# checks and exits quietly if another worker holds the GPU or the queue is
# small. It stops after QA: no load, no redact, no stance, no export.
if [[ -z ${YT_CHAIN} ]]; then
  say "no chain configured for this night; stopping here"
elif [[ -x ${YT_CHAIN} ]]; then
  # The chain reads EPB_LABEL / EPB_MIN_H / EPB_MAX_EST from the environment
  # this script inherited from nights_1021_1023.sh and passes on unchanged;
  # they are not re-derived here, so there is one place they are set.
  say "handing off to ${YT_CHAIN} (EPB_LABEL=${EPB_LABEL:-unset} EPB_MIN_H=${EPB_MIN_H:-unset} EPB_MAX_EST=${EPB_MAX_EST:-unset})"
  chain_pid=$($PY ops/spawn_detached.py --log "output/logs/ep_batch_${YT_LABEL}_spawn.log" \
                  -- "${YT_CHAIN}" 2>> "$LOG")
  say "chain detached, pid ${chain_pid:-?}; its log is output/logs/ep_batch_${YT_LABEL}.log"
else
  say "no executable ${YT_CHAIN} on disk; stopping here, no batch proposed"
fi
