#!/usr/bin/env bash
# Install the progress dashboard as a per-user LaunchAgent on this Mac.
#   ./ops/launchd/install_dashboard.sh            install / reinstall
#   ./ops/launchd/install_dashboard.sh --remove   uninstall
#
# macOS privacy (TCC): a LaunchAgent does NOT inherit Terminal's permission to
# read ~/Documents, ~/Desktop or ~/Downloads. If the repo lives in one of those,
# the agent starts and then fails with "Operation not permitted". Keep the repo
# the supervisor runs from outside them (docs/OPENCLAW_SETUP.md suggests ~/ytpt).
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
LABEL=com.ytpt.dashboard
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

if [ "${1:-}" = "--remove" ]; then
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  rm -f "$DEST"
  echo "removed $LABEL"
  exit 0
fi

case "$REPO" in
  "$HOME/Documents"*|"$HOME/Desktop"*|"$HOME/Downloads"*)
    echo "WARNING: $REPO is a privacy-protected folder; launchd jobs may be denied access."
    echo "         Move the working copy (e.g. to ~/ytpt) or grant python Full Disk Access." ;;
esac

PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY=$(command -v python3)
LOGDIR="$HOME/Library/Logs/ytpt"
mkdir -p "$LOGDIR" "$(dirname "$DEST")"
sed -e "s#__REPO__#$REPO#g" -e "s#__PYTHON__#$PY#g" -e "s#__LOGDIR__#$LOGDIR#g" \
    ops/launchd/com.ytpt.dashboard.plist > "$DEST"
plutil -lint "$DEST" > /dev/null
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$DEST"
sleep 2
port=$(sed -n 's/^DASHBOARD_PORT=\([0-9]*\).*/\1/p' .env 2>/dev/null | tail -1)
port=${port:-8765}
if curl -fsS "http://127.0.0.1:$port/healthz" > /dev/null 2>&1; then
  echo "dashboard running: http://localhost:$port   (logs: $LOGDIR/dashboard.log)"
else
  echo "loaded, but http://127.0.0.1:$port/healthz does not answer yet -- check $LOGDIR/dashboard.log"
fi
