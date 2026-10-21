#!/bin/zsh
# ops/yt_1019b.sh -- the 2026-10-19 YouTube shift for plan/BACKLOG-2026-10-18.md.
#
# Supersedes ops/yt_1019.sh (stopped 2026-10-19 ~03:40 UTC, before it had made
# any YouTube request). Same schedule and same coverage order; the only change
# is the refusal gate, which had a hole:
#
#   episodes.py declares a bot check either on BOT_CHECK text
#   (sign in to confirm | not a bot | HTTP Error 429 | Too Many Requests) OR on
#   two consecutive HTTP 403s (pipeline/episodes.py:144). In the 403 case the
#   log contains only "HTTP Error 403", which the old grep did not match, so the
#   shift would have carried on to the next run after YouTube had already
#   started refusing. episodes.py returns EXIT_BOT_CHECK = 3 in both cases, so
#   this version aborts on the exit code first and keeps the text grep (widened
#   with 403) as a second line of defence.
#
# Pacific 2026-10-18 spent 16 YouTube meeting requests against a 12/day cap
# (BACKLOG-2026-10-18.md 3b), so nothing may touch YouTube until 00:00 PDT
# 2026-10-19 = 07:00 UTC. This script waits for that boundary and then takes
# exactly 12 meetings in three runs of 4, >= 20 min apart, in the coverage
# order of plan/BACKLOG-2026-10-18.md 2 (fewest public density turns first,
# recomputed after the b37-b40 load): pleasanton 0, vta 2, cupertino 4,
# sunnyvale 6. Each episodes.py call is unchanged -- its own 403/429 stop
# rule, padding, cue snapping and one-ffmpeg politeness apply. --max N is
# LIMIT N on meetings (date DESC) and a request is only made when the whole
# audio is not already on disk, so 12 is a ceiling, never exceeded.
#
# Hard stop: if any run exits 3, or its log shows 429 / 403 / bot-check, the
# shift aborts and leaves a STOP line for the supervisor. It never retries a
# refused download.
set -u
cd "$HOME/ytpt" || exit 1
PY=.venv/bin/python
LOG=output/logs/yt_1019_shift.log

say() { print -r -- "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> "$LOG" }

# --- wait for 00:00 PDT 2026-10-19 (07:00 UTC) -------------------------
# date -j -f parses in LOCAL time, so the literal is Pacific, not UTC.
TARGET=$(TZ=America/Los_Angeles date -j -f "%Y-%m-%d %H:%M:%S" "2026-10-19 00:00:00" +%s 2>/dev/null)
say "shift re-armed as yt_1019b.sh (exit-code gate); waiting for 00:00 PDT 2026-10-19"
while :; do
  now=$(date -u +%s)
  [[ $now -ge $TARGET ]] && break
  left=$(( TARGET - now ))
  [[ $left -gt 300 ]] && sleep 300 || sleep $left
done
say "00:00 PDT reached; starting run A"

refused() {
  grep -qiE "429|Too Many Requests|Sign in to confirm|not a bot|HTTP Error 403" "$1" 2>/dev/null
}

run() {  # run <label> <body> <max>
  local label=$1 body=$2 max=$3 rc=0
  local l="output/logs/yt_1019_${label}_${body}.log"
  say "run $label: $body --max $max -> $l"
  $PY pipeline/episodes.py --bodies "$body" --max "$max" > "$l" 2>&1
  rc=$?
  grep -E "done:" "$l" >> "$LOG"
  # EXIT_BOT_CHECK = 3; any nonzero exit stops the shift rather than guessing.
  if [[ $rc -ne 0 ]]; then
    say "STOP: episodes.py exited $rc on $body (3 = bot check / rate limit) -- shift aborted, no retry. Tell Puck."
    return 1
  fi
  if refused "$l"; then
    say "STOP: YouTube refused in $l -- shift aborted, no retry. Tell Puck."
    return 1
  fi
  return 0
}

# run A: 4 meetings -- pleasanton 1 + vta 3
run A pleasanton 1 || exit 1
run A vta 3        || exit 1
say "run A done; sleeping 25 min"
sleep 1500

# run B: 4 meetings -- vta
run B vta 4 || exit 1
say "run B done; sleeping 25 min"
sleep 1500

# run C: 4 meetings -- vta 2 + cupertino 1 + sunnyvale 1
run C vta 2       || exit 1
run C cupertino 1 || exit 1
run C sunnyvale 1 || exit 1

say "shift complete: 12 meetings requested, daily cap reached. No further YouTube today."
