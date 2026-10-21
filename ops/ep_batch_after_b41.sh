#!/bin/bash
# ops/ep_batch_after_b41.sh -- launch the episode-clip batch as soon as the
# sibling full-ASR batch b41 releases the GPU pod (plan/BACKLOG-2026-10-18.md
# 4). One pod at a time, shared with the full-ASR worker; b41's own chain
# (output/fullasr/logs/after_b41.sh) deliberately takes no pod action.
#
# The --speaker list is explicit and contains meeting bodies ONLY: the
# 'transcoded' queue also holds frozen tucker/hasanabi 'show' rows that must
# never be swept into a meeting batch (plan/… 2026-10-18, batch b36).
# propose has no --dry-run -- passing one would run for real.
set -u
cd "$HOME/ytpt" || exit 1
PY=.venv/bin/python
LOG=output/logs/ep_batch_after_b41.log
BODIES=richmond,santarosa,paloalto,hayward,sanleandro,vta,sunnyvale,petaluma,santaclara,livermore,healdsburg,concord,sanjose,losaltos,bart,fremont,sanmateo,napa,vallejo,pleasanton,milpitas,oakland,sanrafael,dalycity

say() { echo "$(date -u +%Y-%m-%dT%H:%M:%S+00:00) $*" >> "$LOG"; }

# --- 1. wait for b41 to settle and for no batch to be running ----------
say "waiting for b41 to release the pod"
for i in $(seq 1 720); do   # up to 4 h
  st=$($PY -c "
import sqlite3;con=sqlite3.connect('$HOME/.local/share/ytpt/state.sqlite')
print(list(con.execute(\"SELECT status FROM batches WHERE batch_id='b41'\"))[0][0])" 2>/dev/null)
  [ "$st" != "running" ] && break
  sleep 20
done
say "b41 status=$st"
if [ "$st" = "running" ]; then
  say "GIVING UP: b41 still running after 4 h. No batch proposed. Tell Puck."
  exit 1
fi

# any other batch still running means the pod is not ours
other=$($PY -c "
import sqlite3;con=sqlite3.connect('$HOME/.local/share/ytpt/state.sqlite')
print(len(list(con.execute(\"SELECT 1 FROM batches WHERE status='running'\"))))" 2>/dev/null)
if [ "$other" != "0" ]; then
  say "ABORT: another batch is still 'running' -- pod not free. No batch proposed."
  exit 1
fi

# --- 2. propose --------------------------------------------------------
say "proposing episode batch (--speaker $BODIES)"
$PY pipeline/gpu_batch.py propose --speaker "$BODIES" >> "$LOG" 2>&1
BID=$(grep -oE "batch b[0-9]+ awaits" "$LOG" | tail -1 | grep -oE "b[0-9]+")
if [ -z "${BID:-}" ]; then
  say "no batch proposed (below threshold, or refused on the spend line). Stopping."
  exit 1
fi
say "proposed $BID"

# --- 3. approve, then launch under nohup ------------------------------
$PY pipeline/gpu_batch.py approve "$BID" >> "$LOG" 2>&1 || { say "approve $BID failed"; exit 1; }
say "approved $BID; launching"
nohup $PY pipeline/gpu_batch.py launch "$BID" > "output/logs/launch_${BID}.log" 2>&1
say "launch $BID returned $?"

# --- 4. QA, then load the corpus, then recount public turns -----------
make qa > "output/logs/qa_after_${BID}.log" 2>&1
grep -E "videos: .* OK" "output/logs/qa_after_${BID}.log" >> "$LOG"
if grep -qE " [1-9][0-9]* FAIL" "output/logs/qa_after_${BID}.log"; then
  say "QA FAIL present after $BID -- NOT loading those transcripts. Tell Puck."
else
  make load-episodes TRANSCRIPTS=output/whisper > "output/logs/load_ep_after_${BID}.log" 2>&1
  say "corpus reloaded (unredacted, explicit TRANSCRIPTS)"
fi
$PY scripts/backlog_public_turns.py --out "output/nimby_pilot/public_turns_after_${BID}.csv" >> "$LOG" 2>&1
$PY pipeline/gpu_batch.py list 2>&1 | tail -3 >> "$LOG"
say "done for $BID"
