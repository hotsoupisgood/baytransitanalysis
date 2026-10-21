#!/usr/bin/env bash
# Install the downloader, transcoder and caption trickle worker as per-user
# LaunchAgents (PLAN.md 4.8: plain scripts under launchd, supervised -- not run
# -- by OpenClaw).
#   ./ops/launchd/install_workers.sh             install / reinstall all three
#   ./ops/launchd/install_workers.sh --remove    uninstall all three
#   ./ops/launchd/install_workers.sh com.ytpt.captions   just this one
# Pause/resume one of them (docs/OPENCLAW_SETUP.md 7):
#   launchctl bootout   gui/$(id -u)/com.ytpt.downloader
#   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ytpt.downloader.plist
# See install_dashboard.sh for the macOS privacy (TCC) caveat about ~/Documents.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
DOMAIN="gui/$(id -u)"
LOGDIR="$HOME/Library/Logs/ytpt"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY=$(command -v python3)
# ffmpeg/ffprobe/deno come from Homebrew; launchd does not read the shell rc.
WPATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

LABELS="com.ytpt.downloader com.ytpt.transcoder com.ytpt.captions"
case "${1:-}" in com.ytpt.*) LABELS="$1"; shift ;; esac

for LABEL in $LABELS; do
  DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
  if [ "${1:-}" = "--remove" ]; then
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$DEST"; echo "removed $LABEL"; continue
  fi
  mkdir -p "$LOGDIR" "$(dirname "$DEST")"
  sed -e "s#__REPO__#$REPO#g" -e "s#__PYTHON__#$PY#g" -e "s#__LOGDIR__#$LOGDIR#g" -e "s#__PATH__#$WPATH#g" \
      "ops/launchd/$LABEL.plist" > "$DEST"
  plutil -lint "$DEST" > /dev/null
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  launchctl bootstrap "$DOMAIN" "$DEST"
  echo "loaded $LABEL   (log: $LOGDIR/${LABEL#com.ytpt.}.log)"
done
