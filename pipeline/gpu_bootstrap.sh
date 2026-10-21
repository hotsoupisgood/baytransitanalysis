#!/usr/bin/env bash
# pipeline/gpu_bootstrap.sh -- runs ON THE RENTED POD (PLAN.md 4.5, Phase 2).
#
#   bash gpu_bootstrap.sh <batch_id>        (from the job dir; see below)
#
# What it does, in order, and each step is safe to re-run:
#   1. sanity: a GPU is visible, Python is in WhisperX's supported range
#   2. deps: `pip install -r gpu-freeze.txt` -- skipped when the freeze file's
#      checksum matches the last successful install (.deps-ok)
#   3. inputs: transport "ssh" (v1, default) means this Mac already pushed
#      in/manifest.json and in/*.opus with rsync; "r2" (v2, PLAN.md 4.4) is
#      where an `rclone copy` would go -- it is NOT built and exits 3
#   4. run gpu_job.py --on-pod: one JSON per video into out/, written atomically
#      the moment it completes (community pods can be reclaimed mid-job), plus
#      out/timing.jsonl and out/FAILED/<id>.txt for per-video failures
#   5. touch out/DONE. Results ship back by this Mac's rsync pull loop
#      (gpu_job.run), per file, not as one archive at the end.
#
# The layout it expects (gpu_job.run pushes exactly this):
#   $JOB_DIR/gpu_bootstrap.sh  gpu_job.py  gpu-freeze.txt
#   $JOB_DIR/in/manifest.json  in/<video_id>.opus ...
#   $JOB_DIR/out/              (created here)
#
# Secrets: HF_TOKEN arrives in the pod's environment (set at pod create). It
# is never echoed, never written to a file, never put on a command line.
# Shell tracing is deliberately off for that reason -- do not turn it on.
set -euo pipefail

BATCH_ID="${1:?usage: gpu_bootstrap.sh <batch_id>}"
JOB_DIR="${YTPT_JOB_DIR:-/workspace/ytpt}"
TRANSPORT="${YTPT_TRANSPORT:-ssh}"
PY="${YTPT_PYTHON:-python3}"
export HF_HOME="${HF_HOME:-$JOB_DIR/hf}"           # model cache; ephemeral like the rest
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
mkdir -p "$JOB_DIR"
cd "$JOB_DIR"
mkdir -p out in "$HF_HOME"
LOG="$JOB_DIR/bootstrap.log"
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }
log "bootstrap start batch=$BATCH_ID dir=$JOB_DIR transport=$TRANSPORT host=$(hostname)"

# 1. sanity ------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
  log "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  log "FATAL: nvidia-smi not found -- this is not a GPU pod"; exit 2
fi
"$PY" - <<'EOF' || { log "FATAL: python version outside WhisperX 3.8.x range (>=3.10,<3.14)"; exit 2; }
import sys; v = sys.version_info[:2]; assert (3, 10) <= v < (3, 14), v; print(f"python {v[0]}.{v[1]}")
EOF
if [ -z "${HF_TOKEN:-}" ] && [ -r /proc/1/environ ]; then
  # An SSH session does not inherit the container's env (b3, 2026-09-18: the
  # token was on PID 1 but not in the sshd session). Take it from PID 1's
  # environment -- still never echoed, never written, never on a command line.
  HF_TOKEN="$(tr '\0' '\n' </proc/1/environ | sed -n 's/^HF_TOKEN=//p' | head -1)"
  export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
  log "FATAL: HF_TOKEN is not in the environment (pyannote would 401); create the pod with it"; exit 2
fi
log "hf token: present (length not logged)"
for f in gpu_job.py gpu-freeze.txt in/manifest.json; do
  [ -f "$f" ] || { log "FATAL: missing $f -- the push from the Mac did not complete"; exit 2; }
done

# 2. deps, idempotent ----------------------------------------------------------
# WhisperX decodes audio through the ffmpeg binary; the runpod/pytorch image
# does not ship it (b3, 2026-09-18: every video FAILED with "No such file or
# directory: 'ffmpeg'" after the models had loaded). Install it before pip.
if ! command -v ffmpeg >/dev/null 2>&1; then
  log "deps: apt-get install ffmpeg ..."
  export DEBIAN_FRONTEND=noninteractive
  (apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg) >/dev/null 2>&1 \
    || { log "FATAL: apt-get install ffmpeg failed"; exit 2; }
fi
log "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f1-3)"
if command -v sha256sum >/dev/null 2>&1; then FREEZE_SUM=$(sha256sum gpu-freeze.txt | cut -c1-16)
else FREEZE_SUM=$(shasum -a 256 gpu-freeze.txt | cut -c1-16); fi
if [ -f .deps-ok ] && [ "$(cat .deps-ok)" = "$FREEZE_SUM" ]; then
  log "deps: already installed for freeze $FREEZE_SUM"
else
  log "deps: pip install -r gpu-freeze.txt (freeze $FREEZE_SUM) ..."
  "$PY" -m pip install -q -r gpu-freeze.txt 2>&1 | tail -20 | tee -a "$LOG"
  # An unpinned transitive package can still break the import; prove it now,
  # before the meter has run for an hour.
  "$PY" - <<'EOF' 2>&1 | tee -a "$LOG"
import torch, whisperx, faster_whisper, ctranslate2, pyannote.audio
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} whisperx {getattr(whisperx, '__version__', '?')} "
      f"faster_whisper {faster_whisper.__version__} ctranslate2 {ctranslate2.__version__} "
      f"pyannote.audio {pyannote.audio.__version__}")
assert torch.cuda.is_available(), "torch sees no CUDA device"
EOF
  printf '%s' "$FREEZE_SUM" > .deps-ok
  log "deps: ok"
fi
if [ ! -f "$JOB_DIR/.pip-freeze.txt" ]; then
  # The real gpu-freeze.txt for the repo (PLAN.md 6, Phase 0 step 8). The Mac
  # pulls this back with the results.
  "$PY" -m pip freeze > "$JOB_DIR/out/pip-freeze.txt" 2>/dev/null || true
  cp "$JOB_DIR/out/pip-freeze.txt" "$JOB_DIR/.pip-freeze.txt" 2>/dev/null || true
fi

# 3. inputs --------------------------------------------------------------------
case "$TRANSPORT" in
  ssh) n=$(ls in/*.opus 2>/dev/null | wc -l | tr -d ' ')
       [ "$n" -gt 0 ] || { log "FATAL: no in/*.opus -- nothing was pushed"; exit 2; }
       log "inputs: $n opus files pushed by the Mac" ;;
  r2)  log "FATAL: transport r2 is not built (PLAN.md 4.4 v2); use ssh"; exit 3 ;;
  *)   log "FATAL: unknown YTPT_TRANSPORT=$TRANSPORT"; exit 3 ;;
esac

# 4. transcribe ----------------------------------------------------------------
# Re-run safe: gpu_job.py skips any video whose out/<id>.json already exists.
log "job: start"
set +e
"$PY" gpu_job.py --on-pod --batch "$BATCH_ID" --manifest in/manifest.json --in in --out out 2>&1 | tee -a "$JOB_DIR/job.log"
rc=${PIPESTATUS[0]}
set -e
log "job: exit $rc"

# 5. done marker ---------------------------------------------------------------
# Written even on a non-zero exit so the Mac stops polling and reads
# out/FAILED/ instead of waiting for a timeout. The exit code is inside.
printf '%s\n' "$rc" > out/DONE
log "bootstrap end rc=$rc"
exit "$rc"
