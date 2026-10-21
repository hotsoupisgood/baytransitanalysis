#!/bin/bash
# One-off (Puck, 2026-10-20: "b2 free"): remove the audio that was uploaded to B2 before the
# switch to free-tier mode, then run one free-mode backup (transcripts + DB snapshots).
# The audio itself stays on the Mac and in the Linux node's ~/ytpt-mirror. B2 refuses every
# call while the daily transaction cap is spent, so this retries every 30 min, 8 times.
set -uo pipefail
cd "$(dirname "$0")/.."
while IFS='=' read -r k v; do
  v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"; export "$k=$v"
done < <(grep -E '^(export +)?B2_[A-Z_]+=' .env | sed -E 's/^export +//')
export B2_APPLICATION_KEY_ID="${B2_KEY_ID:?}" B2_APPLICATION_KEY="${B2_APPLICATION_KEY:?}"
base="b2://${B2_BUCKET:?}/${B2_PREFIX:?}"

for attempt in 1 2 3 4 5 6 7 8; do
  if b2 ls "$base/" >/dev/null 2>&1; then
    for p in output/audio output/fullasr/audio output/t2 meetings _probe; do
      echo ">>> rm $p $(date -u +%H:%M:%SZ)"
      b2 rm -r --versions --no-progress --threads 8 "$base/$p" || echo "rm $p failed"
    done
    echo ">>> free-mode backup $(date -u +%H:%M:%SZ)"
    ./backup.sh
    exit $?
  fi
  echo "attempt $attempt: B2 not reachable (daily cap?), retrying in 30 min $(date -u +%H:%M:%SZ)"
  sleep 1800
done
echo "gave up after 8 attempts"; exit 1
