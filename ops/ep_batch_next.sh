#!/bin/bash
# ops/ep_batch_next.sh -- the NEXT episode-clip batch, after b42
# (plan/BACKLOG-2026-10-18.md 4). Waits until NO batch is 'running' at all,
# so the pod really is free: on 2026-10-19 the full-ASR worker launched b43
# at 01:56 while this side's b42 was launched at 01:54, and two pods billed
# side by side for 1.4 h. Checking "b41 is done" was not enough; checking
# "nothing is running" is.
#
# --min-hours 8 is the agreed re-batch threshold. propose has no --dry-run.
# The --speaker list is meeting bodies ONLY: frozen tucker/hasanabi 'show'
# rows must never be swept in.
set -u
cd "$HOME/ytpt" || exit 1
PY=.venv/bin/python
LOG=output/logs/ep_batch_next.log
BODIES=richmond,santarosa,paloalto,hayward,sanleandro,vta,sunnyvale,petaluma,santaclara,livermore,healdsburg,concord,sanjose,losaltos,bart,fremont,sanmateo,napa,vallejo,pleasanton,milpitas,oakland,sanrafael,dalycity

say() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> "$LOG"; }
nrunning() { $PY -c "
import sqlite3;con=sqlite3.connect('$HOME/.local/share/ytpt/state.sqlite')
print(len(list(con.execute(\"SELECT 1 FROM batches WHERE status='running'\"))))" 2>/dev/null; }

say "waiting for every batch to settle (one pod at a time)"
for i in $(seq 1 1080); do           # up to 6 h
  [ "$(nrunning)" = "0" ] && break
  sleep 20
done
if [ "$(nrunning)" != "0" ]; then
  say "GIVING UP: a batch is still running after 6 h. Nothing proposed."; exit 1
fi
say "pod free; proposing with --min-hours 8"
$PY pipeline/gpu_batch.py propose --speaker "$BODIES" --min-hours 8 >> "$LOG" 2>&1
BID=$(grep -oE "batch b[0-9]+ awaits" "$LOG" | tail -1 | grep -oE "b[0-9]+")
if [ -z "${BID:-}" ]; then
  say "nothing proposed (under 8 h ready, or refused on the spend line). Stopping."; exit 0
fi
say "proposed $BID"
$PY pipeline/gpu_batch.py approve "$BID" >> "$LOG" 2>&1 || { say "approve $BID failed"; exit 1; }
say "approved $BID; launching"
nohup $PY pipeline/gpu_batch.py launch "$BID" > "output/logs/launch_${BID}.log" 2>&1
say "launch $BID returned $?"
make qa > "output/logs/qa_after_${BID}.log" 2>&1
grep -E "videos: .* OK" "output/logs/qa_after_${BID}.log" >> "$LOG"
if grep -qE " [1-9][0-9]* FAIL" "output/logs/qa_after_${BID}.log"; then
  say "QA FAIL after $BID -- NOT loading those transcripts. Tell Puck."
else
  make load-episodes TRANSCRIPTS=output/whisper > "output/logs/load_ep_after_${BID}.log" 2>&1
  say "corpus reloaded (unredacted, explicit TRANSCRIPTS)"
fi
$PY scripts/backlog_public_turns.py --out "output/nimby_pilot/public_turns_after_${BID}.csv" >> "$LOG" 2>&1
$PY pipeline/gpu_batch.py list 2>&1 | tail -3 >> "$LOG"
say "done for $BID"
