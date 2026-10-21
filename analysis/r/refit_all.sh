#!/bin/bash
# analysis/r/refit_all.sh -- step 6 of the 2026-10-20 freeze: the nested lme4
# refits on the frozen frame, one row set per outcome, plus the drop-BART
# sensitivity for each. Route is a fixed effect in every fit, so the by-route
# estimates come out of *_fixed.csv and need no separate model.
#
# Export is read-only on the DB and costs nothing; the fits are CPU only.
# Run from the repo root with its venv, after the stance top-up has been
# collected and scripts/stance_s2_recompute.sh has rebuilt
# output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv (which
# analysis/renter_share5_freq.py reads for role and route).
#
#   nohup analysis/r/refit_all.sh > output/renter_share_6/refit.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python

for OUTCOME in anyside withinside hostility; do
  for BART in "" "--drop-bart"; do
    echo "=== export $OUTCOME $BART"
    $PY analysis/r/export_nested.py --outcome "$OUTCOME" $BART || echo "EXPORT FAILED: $OUTCOME $BART"
  done
done

for STEM in anyside withinside hostility anyside_nobart withinside_nobart hostility_nobart; do
  echo "=== fit $STEM  ($(date +%H:%M:%S))"
  Rscript analysis/r/turn_nested.R "$STEM" || echo "FIT FAILED: $STEM"
done

echo "=== refit done ($(date +%H:%M:%S))"
