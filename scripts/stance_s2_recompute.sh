#!/bin/bash
# scripts/stance_s2_recompute.sh -- everything downstream of the S2 stance
# top-up, in order, against the combined frame (plan/STANCE-S2.md step 2-3).
# No scoring, no spend: these all read turn_stance and rewrite derived
# artefacts. Run from the repo root with its venv.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FRAME=output/stance_s2/frame_all.csv

echo "=== 1. role_audit (rule ra-1) over all scored turns, combined frame"
$PY scripts/role_audit.py --frame "$FRAME"

echo "=== 2. NIMBY/YIMBY view, role_audit -> _roleaudit (what the site export reads)"
$PY analysis/stance_s1_role_audit_view.py --role-col audit --frame "$FRAME"

echo "=== 3. NIMBY/YIMBY view, role_audit -> unsuffixed (the reported artefacts)"
$PY analysis/stance_s1_nimby_view.py --frame "$FRAME"

echo "=== 4. NIMBY/YIMBY view, cue role -> _cue (the sensitivity)"
$PY analysis/stance_s1_nimby_view.py --role-col cue --frame "$FRAME"

echo "=== 5. tail shares under both role columns"
$PY analysis/stance_s1_tail_shares_roleaudit.py --role-col audit --frame "$FRAME"
$PY analysis/stance_s1_tail_shares_roleaudit.py --role-col cue --frame "$FRAME"

echo "=== 6. the numbers the site prose quotes as literals"
$PY analysis/stance_s2_site_numbers.py --json | tee output/stance_s2/site_numbers.json

echo "=== done"
