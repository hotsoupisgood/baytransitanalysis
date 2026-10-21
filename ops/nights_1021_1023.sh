#!/bin/zsh
# ops/nights_1021_1023.sh -- arm the remaining YouTube backlog as three nightly
# background shifts, 2026-10-21/22/23, each at 00:00 PDT. This file is the
# per-night numbers and the reasoning; ops/yt_night.sh, ops/yt_night_alert.py
# and ops/ep_batch_night.sh are the three mechanisms, shared by all of them.
#
#   ops/nights_1021_1023.sh arm      # spawn shift + alert for each night
#   ops/nights_1021_1023.sh status   # are they alive, and when do they fire
#   ops/nights_1021_1023.sh plan     # what each night would request, no requests
#
# WHAT PUCK ASKED FOR, 2026-10-20 09:22 PDT: "start the yt transcription in the
# background we just wont put it into production". Download, cut and GPU-
# transcribe; do not load, redact, stance-code or export. Today's frozen
# dataset stays the published one. ops/ep_batch_night.sh stops after QA for
# exactly that reason -- see its header.
#
# WHY NOTHING TOUCHES YOUTUBE TODAY: the 2026-10-20 budget was spent at
# 00:00-00:22 PDT on 12 meetings (ops/yt_1020.sh, clean, b50 done 07:22 UTC).
# 12/12 used, so the earliest permitted request is 00:00 PDT on 2026-10-21 =
# 1792566000 = 2026-10-21T07:00:00Z. Each night's shift sits in a sleep loop
# until its own epoch and makes no request before it.
#
# PRE-ARM CHECKS, run live 2026-10-20 ~16:30 UTC:
#   * backoff:  `captions.py --backoff-status` -> "no active backoff" (the last
#               expired 2026-10-18T19:06:58Z).
#   * refusals: no unresolved 429 / 403 / bot-check; last night's shift log
#               ends on "shift complete".
#   * disk:     29 Gi free. Measured 47.1 MB per audio-hour over 66 whole
#               meetings already on disk, and the three nights fetch 100.9
#               audio-h -> ~4.8 GB of whole-meeting opus plus ~1.4 GB of
#               clips. Night 1 alone is 50.4 audio-h = ~2.4 GB. The 5 Gi guard
#               in yt_night.sh aborts well before the volume is in trouble.
#   * spend:    RunPod $16.12 of the $31.00 stop line; month $11.12 of the
#               $40.00 cap. The three batches are ~$1.23 estimated and ~$0.85
#               expected actual (2026-10-20's b50 came in at 0.69x estimate).
#   * queue:    the ONLY in_frame=1 status='transcoded' rows right now are the
#               frozen side corpus -- tucker 3 (2.25 h), hasanabi 4 (1.15 h).
#               Both are kind='show' with no kind='meeting' row, so the
#               derived roster in ep_batch_night.sh cannot name them.
#
# THE BACKLOG, read live from state.sqlite, not copied from the backlog table:
# 27 YouTube-gated meetings carrying 162 located episodes, 29.55 core episode-h,
# and NONE of the 27 has whole audio on disk, so 27 meetings == 27 requests.
# (The task sheet said ~28 / ~164 and "dalycity 2"; live it is 27 / 162 and
# dalycity 1. The 28th row is a rohnertpark meeting with zero located episodes,
# which is why it never appears below.) Per body, with core episode-hours:
#   bart 12 mtgs / 119 eps / 24.30 h      sanjose 6 / 15 / 2.35
#   petaluma 2 / 5 / 0.32                 sanrafael 2 / 4 / 0.40
#   sanmateo 1 / 7 / 1.03                 losaltos 1 / 6 / 0.33
#   dalycity 1 / 2 / 0.42                 santarosa 1 / 2 / 0.27
#   concord 1 / 2 / 0.13
#
# COVERAGE ORDER, recomputed live rather than reused: public density turns per
# body from scripts/backlog_public_turns.py against corpus.sqlite, 2026-10-20
# 16:28 UTC -> output/backlog/public_turns_1020_pre_arm.csv. Gated bodies,
# fewest public turns first:
#   bart 8, dalycity 8, sanrafael 15, concord 19, santarosa 22, losaltos 23,
#   sanmateo 25, petaluma 49, sanjose 214.
# bart and dalycity tie at 8. bart goes first on the same tie-break
# ops/yt_1020.sh used: it is the more under-covered body (4 density episodes
# loaded against dalycity's 5) and it is 24.30 of the 29.55 remaining gated
# episode-hours -- 82 % -- so it has to start some time. Taking bart first also
# means the one body that cannot fit in a single night gets the whole of night
# 1 rather than being split across two.
#
# SO: night 1 is bart's 12, which clears the body entirely. Night 2 is the next
# seven bodies in coverage order (9 meetings) plus sanjose's 3 most recent to
# fill the 12. Night 3 is sanjose's remaining 3.
#
# WHY `--max N` IS SAFE TO REPEAT WITHIN A NIGHT: --max N is LIMIT N over
# meetings ordered meeting_date DESC (pipeline/episodes.py:476), and a meeting
# leaves that queue once it is cut (SELECT_WHERE keeps status='cut' only while
# an episode is still 'located', pipeline/episodes.py:82). So "bart 4" three
# times takes 12 distinct meetings, and night 3's "sanjose 3" takes whatever
# night 2 did not. A request is only made when the whole audio is absent, so a
# re-picked meeting costs a re-cut, never a second download. Every call names a
# body: never a bare --max.
set -u
cd "$HOME/ytpt" || exit 1
PY=.venv/bin/python
SPAWN="ops/spawn_detached.py"
mkdir -p output/logs

# --- the three nights ---------------------------------------------------
# label | PDT date | runs (run-label:body:max) | meetings | batch floor h | batch ceiling USD
#
# Floors: night 1 expects ~26.3 episode-h once PAD_S (30 s each side, 119
# clips) is on the clips, night 2 ~4.6 h, night 3 ~1.4 h. The floor is set well
# under each expectation so a partly-refused night still batches what it got,
# and it doubles as propose's BATCH_TARGET_HOURS, which is 15.0 in the live
# settings and would otherwise refuse nights 2 and 3 outright.
#
# Ceiling: $3.00 for all three, and this is deliberately much looser than the
# nights themselves need. Their own estimates are $0.86 / $0.22 / $0.13 at the
# live (h/25 + 7 min boot) * $0.49/h * 1.5 contingency, and even the catch-up
# case -- night 1's batch never landing, so all 29.55 h roll into night 2 --
# estimates $0.96. $1.50 would cover all of that.
#
# It is $3.00 because the roster is every meeting body, not just the night's,
# and a sibling worker is feeding the same queue. At 16:47 UTC today another
# worker's full-ASR run had already cut 39 fresh samtrans episode clips
# (12.25 h) into 'transcoded', and samtrans is a meeting body, so night 1's
# proposal will be bart's ~26 h PLUS whatever that worker has produced by
# 00:00 PDT. bart + 12 h estimates $1.22; bart + 34 h estimates $1.85. A $1.50
# ceiling would therefore have a good chance of DENYING night 1 -- which costs
# the night's GPU work and sends Puck a "stopped and denied" warning at 1 a.m.
# for something that is not wrong.
#
# What the ceiling is actually for is catching a runaway: every in-frame
# episode clip under a meeting body is 1552 clips / 237.62 h, which estimates
# $7.07, so $3.00 still refuses that by a wide margin. Keeping the frozen
# Hasan/Tucker corpus out is the ROSTER's job, not this number's (see
# ep_batch_night.sh), and the real money guards are inside gpu_batch.py: the
# $31.00 RunPod stop line, with $14.88 left, and MONTHLY_SPEND_CAP_USD $40.00,
# with $11.12 spent. Three nights at the full $3.00 ceiling would be $9.00,
# still inside that headroom.
NIGHTS=(
  "1021|2026-10-21|A:bart:4 B:bart:4 C:bart:4|12|8.0|3.00"
  "1022|2026-10-22|A:dalycity:1 A:sanrafael:2 A:concord:1 B:santarosa:1 B:losaltos:1 B:petaluma:2 C:sanmateo:1 C:sanjose:3|12|1.5|3.00"
  "1023|2026-10-23|A:sanjose:3|3|0.5|3.00"
)
ALERT_WINDOW_S=28800     # 8 h from the start; the alert exits early on
                         # "shift complete", so this is margin, not a deadline

epoch_for() {  # epoch_for <YYYY-MM-DD> -> 00:00 Pacific that day, as epoch
  # date -j -f parses in LOCAL time, so TZ is set explicitly and the literal is
  # Pacific. Computed once, here, at arming time: DST and "which midnight" are
  # settled now rather than re-derived in whatever timezone a process wakes up.
  TZ=America/Los_Angeles date -j -f "%Y-%m-%d %H:%M:%S" "$1 00:00:00" +%s
}

alive() {  # alive <pidfile>
  [[ -f $1 ]] || return 1
  local p=$(cat "$1" 2>/dev/null)
  [[ -n $p ]] && kill -0 "$p" 2>/dev/null
}

prior_logs_for() {  # the shift logs of every night before this one
  local upto=$1 out=""
  for row in $NIGHTS; do
    local l=${row%%|*}
    [[ $l == "$upto" ]] && break
    out="${out} output/logs/yt_${l}_shift.log"
  done
  print -r -- "${out# }"
}

cmd=${1:-status}

case $cmd in
arm)
  for row in $NIGHTS; do
    label=${${(s:|:)row}[1]}
    day=${${(s:|:)row}[2]}
    runs=${${(s:|:)row}[3]}
    expected=${${(s:|:)row}[4]}
    minh=${${(s:|:)row}[5]}
    maxest=${${(s:|:)row}[6]}
    target=$(epoch_for "$day")
    stop=$(( target + ALERT_WINDOW_S ))
    shift_pid=output/logs/yt_${label}_shift.pid
    alert_pid=output/logs/yt_${label}_alert.pid

    print -r -- "--- night ${label} (${day} 00:00 PDT = $(date -u -r $target +%Y-%m-%dT%H:%M:%SZ), epoch ${target})"

    if alive "$shift_pid"; then
      print -r -- "    shift already alive (pid $(cat $shift_pid)); not arming a second"
    else
      YT_LABEL="$label" YT_TARGET="$target" YT_RUNS="$runs" YT_CHAIN=ops/ep_batch_night.sh \
      YT_PRIOR_LOGS="$(prior_logs_for "$label")" \
      EPB_LABEL="$label" EPB_MIN_H="$minh" EPB_MAX_EST="$maxest" \
        $PY $SPAWN --log "output/logs/yt_${label}_spawn.log" --pidfile "$shift_pid" \
                   -- /bin/zsh ops/yt_night.sh > /dev/null
      print -r -- "    shift  pid $(cat $shift_pid)  ${expected} meetings  runs: ${runs}"
    fi

    if alive "$alert_pid"; then
      print -r -- "    alert already alive (pid $(cat $alert_pid)); not arming a second"
    else
      $PY ops/yt_night_alert.py --label "$label" --log "output/logs/yt_${label}_shift.log" \
          --start-epoch "$target" --stop-epoch "$stop" --expected "$expected"
      sleep 1
      print -r -- "    alert  pid $(cat $alert_pid 2>/dev/null)  window $(date -u -r $target +%H:%M)-$(date -u -r $stop +%H:%M) UTC"
    fi
  done
  ;;

status)
  print -r -- "now: $(date -u +%Y-%m-%dT%H:%M:%SZ) / $(TZ=America/Los_Angeles date +'%Y-%m-%d %H:%M %Z')"
  for row in $NIGHTS; do
    label=${${(s:|:)row}[1]}
    day=${${(s:|:)row}[2]}
    expected=${${(s:|:)row}[4]}
    target=$(epoch_for "$day")
    sp=output/logs/yt_${label}_shift.pid
    ap=output/logs/yt_${label}_alert.pid
    s="dead"; alive "$sp" && s="alive pid $(cat $sp)"
    a="dead"; alive "$ap" && a="alive pid $(cat $ap)"
    print -r -- "night ${label}  fires $(date -u -r $target +%Y-%m-%dT%H:%M:%SZ) (epoch ${target}, in $(( (target - $(date -u +%s)) / 3600 )) h)  ${expected} mtgs  shift: ${s}  alert: ${a}"
  done
  ;;

plan)
  # --plan is episodes.py's dry run: it lists the meetings and episodes it
  # would take and makes NO request of any kind -- the `if args.plan:` branch at
  # pipeline/episodes.py:485 `continue`s before every download path, and the
  # run's own summary line confirms it with "YouTube 0 requests". Verified
  # 2026-10-20 16:34 UTC against all nine bodies. Safe to run at any hour.
  for row in $NIGHTS; do
    label=${${(s:|:)row}[1]}
    runs=${${(s:|:)row}[3]}
    print -r -- "=== night ${label} ==="
    for spec in ${=runs}; do
      rest=${spec#*:}
      print -r -- "--- ${spec}"
      # say() prefixes every line with a UTC timestamp, so these patterns match
      # the body of the line, not its start.
      $PY pipeline/episodes.py --bodies "${rest%%:*}" --max "${rest##*:}" --plan 2>&1 \
        | grep -E "meeting\(s\) with located|episodes, [0-9]+ core min|YouTube [0-9]+ requests"
    done
  done
  ;;

*)
  print -r -- "usage: ops/nights_1021_1023.sh {arm|status|plan}"; exit 2
  ;;
esac
