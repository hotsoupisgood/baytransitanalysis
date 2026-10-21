#!/usr/bin/env bash
# Bootstrap this project on a new machine. Idempotent — safe to re-run.
set -uo pipefail
cd "$(dirname "$0")"
ok=0; warn=0; fail=0
say() { printf '%s\n' "$*"; }
pass() { say "  [ok]   $*"; ok=$((ok+1)); }
note() { say "  [warn] $*"; warn=$((warn+1)); }
bad()  { say "  [FAIL] $*"; fail=$((fail+1)); }

say "== 1. interpreters and tools =="
if command -v python3 >/dev/null; then
  v=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
  # 3.11+: yt-dlp deprecated Python 3.10 in 2026 (it warns on every call).
  if python3 -c 'import sys;sys.exit(0 if sys.version_info>=(3,11) else 1)'; then
    pass "python3 $v"
  else bad "python3 $v — need >= 3.11 (brew install python@3.13)"; fi
else bad "python3 not found"; fi

if command -v ffmpeg >/dev/null; then
  # The transcode is 16 kHz mono Opus (PLAN.md 4.3). An ffmpeg built without
  # libopus fails on the first file, not here -- unless we look now.
  # Capture first: `ffmpeg ... | grep -q` under pipefail reports failure when
  # grep exits early and ffmpeg dies of SIGPIPE.
  encoders=$(ffmpeg -hide_banner -encoders 2>/dev/null)
  case "$encoders" in
    *libopus*) pass "ffmpeg (libopus encoder present)" ;;
    *) bad "ffmpeg has no libopus encoder — brew reinstall ffmpeg" ;;
  esac
else note "ffmpeg not found — needed for the pipeline (brew install ffmpeg)"; fi

# yt-dlp needs an external JavaScript runtime to solve YouTube's JS challenges
# (yt-dlp "EJS", required since Nov 2025). Deno >= 2.3 is found automatically;
# Node >= 22 only if yt-dlp is told (--js-runtimes node). Without one, downloads
# fail with 403s mid-crawl rather than at startup.
if command -v deno >/dev/null; then
  pass "deno $(deno --version 2>/dev/null | head -1 | awk '{print $2}') (yt-dlp JS runtime)"
elif command -v node >/dev/null && node -e 'process.exit(+process.versions.node.split(".")[0] >= 22 ? 0 : 1)' 2>/dev/null; then
  note "no deno; node $(node --version) works only with yt-dlp --js-runtimes node (or: brew install deno)"
else note "no JavaScript runtime for yt-dlp — brew install deno"; fi

if command -v git >/dev/null; then pass "git"; else note "git not found (xcode-select --install)"; fi
if command -v rclone >/dev/null; then pass "rclone (for backup.sh)"
else note "rclone not found — only needed for backup.sh (brew install rclone)"; fi
if command -v openclaw >/dev/null; then pass "openclaw CLI (supervisor)"
else note "openclaw not found — only needed on the supervisor Mac (docs/OPENCLAW_SETUP.md)"; fi

say ""
say "== 2. python packages =="
# WHY A VENV: modern Pythons (Homebrew, Debian/Ubuntu system python) mark
# themselves "externally managed" (PEP 668) and refuse `pip install`, with or
# without --user, to stop pip fighting the OS package manager. A project-local
# venv sidesteps that entirely, needs no sudo, no --break-system-packages, and
# keeps this project's deps off the system python. It is also the only version
# of this that behaves the same on macOS and Linux.
VENV=".venv"
if [ ! -d "$VENV" ]; then
  say "  creating $VENV ..."
  python3 -m venv "$VENV" 2>/dev/null || {
    note "python3 -m venv failed — trying 'pip install --user'"
    VENV=""
  }
fi

if [ -n "$VENV" ] && [ -x "$VENV/bin/python" ]; then
  PY="$VENV/bin/python"
  "$PY" -m pip install --quiet --upgrade pip >/dev/null 2>&1
  if "$PY" -m pip install --quiet -r requirements.txt; then
    pass "deps installed in $VENV"
    say "  activate with:  source $VENV/bin/activate"
  else bad "pip install into $VENV failed"; fi
  # yt-dlp lives in the venv, so `command -v yt-dlp` on PATH says nothing about
  # it. Check the two optional parts that YouTube actually needs.
  if [ -x "$VENV/bin/yt-dlp" ]; then
    pass "yt-dlp $("$VENV/bin/yt-dlp" --version 2>/dev/null) in $VENV"
    "$PY" -c 'import yt_dlp_ejs' 2>/dev/null && pass "yt-dlp-ejs (JS challenge solver scripts)" \
      || bad "yt-dlp-ejs missing — requirements.txt must say yt-dlp[default,...]"
    "$PY" -c 'import curl_cffi' 2>/dev/null && pass "curl_cffi (browser impersonation)" \
      || note "curl_cffi missing — yt-dlp will warn 'no impersonate target is available'"
  else note "yt-dlp not in $VENV"; fi
else
  PY="python3"
  if "$PY" -c 'import numpy, pandas' 2>/dev/null; then
    pass "numpy/pandas already available"
  elif "$PY" -m pip install --user --quiet -r requirements.txt 2>/dev/null; then
    pass "installed with --user"
  elif "$PY" -m pip install --user --quiet --break-system-packages -r requirements.txt 2>/dev/null; then
    note "installed with --break-system-packages (venv would be cleaner)"
  else
    bad "could not install dependencies — see the note below"
    say "       try:  python3 -m venv .venv && source .venv/bin/activate \\"
    say "               && pip install -r requirements.txt"
  fi
fi

say ""
say "== 3. sqlite fts5 =="
if "$PY" -c "
import sqlite3
c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts5(x)')
" 2>/dev/null; then pass "FTS5 available"
else bad "FTS5 missing — the search index won't build"; fi

say ""
say "== 4. state db location =="
# WHY THIS CHECK: SQLite needs POSIX byte-range locking. Synced and network
# folders often don't provide it, and SQLite fails with a bare 'disk I/O error'
# or silently writes a 0-byte file. Catch it now, not mid-run.
# STATE_DB_PATH may be set in .env; read it without executing .env.
if [ -z "${STATE_DB_PATH:-}" ] && [ -f .env ]; then
  STATE_DB_PATH=$(sed -n 's/^STATE_DB_PATH=//p' .env | tail -1 | sed 's/[[:space:]]*#.*$//; s/^["'\'']//; s/["'\'']$//')
fi
state_dir="${STATE_DB_PATH:-$HOME/.local/share/ytpt/state.sqlite}"
state_dir=$(dirname "${state_dir/#\~/$HOME}")
mkdir -p "$state_dir" 2>/dev/null
if "$PY" - "$state_dir" <<'PY'
import sqlite3, sys, os
# Unique name: a leftover test file (another user, a crashed run, a folder that
# refuses deletes) must not turn this check into a false failure.
p = os.path.join(sys.argv[1], f".locktest-{os.getpid()}.sqlite")
try:
    c = sqlite3.connect(p); c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS t(x)"); c.execute("INSERT INTO t VALUES (1)")
    c.commit(); c.close()
    ok = os.path.getsize(p) > 0
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(p + suffix)
        except OSError:
            pass
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
then pass "WAL-capable local disk at $state_dir"
else bad "$state_dir cannot host a SQLite WAL database — pick a real local path"; fi

say ""
say "== 5. secrets =="
if [ -f .env ]; then
  miss=""
  # Notifications go to WhatsApp through OpenClaw (DECISIONS.md #7), so the
  # target number is needed; Discord variables are optional.
  for k in ANTHROPIC_API_KEY HF_TOKEN RUNPOD_API_KEY OPENCLAW_NOTIFY_TARGET; do
    grep -qE "^${k}=[^[:space:]#]+" .env || miss="$miss $k"
  done
  [ -z "$miss" ] && pass ".env present and populated" \
                 || note ".env present; not yet set:$miss"
  perms=$("$PY" -c 'import os;print(oct(os.stat(".env").st_mode & 0o077))' 2>/dev/null)
  [ "$perms" = "0o0" ] && pass ".env readable by owner only" \
    || note ".env is readable by group/others — chmod 600 .env"
else note ".env missing — cp .env.example .env and fill it in"; fi

say ""
say "== 6. version control =="
if [ -d .git ]; then
  pass "git repo initialised"
  # A remote/VM session that cannot delete files leaves git lock files behind,
  # and every later commit then fails with "Unable to create ... .lock".
  locks=$(find .git -maxdepth 3 -name '*.lock' 2>/dev/null | tr '\n' ' ')
  [ -z "$locks" ] && pass "no stale git lock files" \
    || note "git lock files present: $locks— if no git command is running: rm $locks"
else
  note "not a git repo — initialising"
  git init -q && git add -A && git -c user.email=noreply@localhost \
    -c user.name=setup commit -qm "Import project" && pass "initial commit created" \
    || note "git init failed; continuing"
fi

say ""
say "== 7. test suite =="
# Per-run log file: a fixed /tmp path already owned by another macOS user (the
# dedicated OpenClaw account, PLAN.md 4.8) made this step fail with
# "Permission denied" even though the tests pass.
log=$(mktemp "${TMPDIR:-/tmp}/ytpt_test.XXXXXX")
if "$PY" analysis/test_pipeline.py >"$log" 2>&1; then
  pass "$(tail -1 "$log")"
else bad "test suite failed — see $log"; fi

say ""
say "=============================================="
say " $ok ok, $warn warnings, $fail failures"
if [ "$fail" -gt 0 ]; then
  say " Fix the failures before building anything."
  exit 1
fi
if [ -x .venv/bin/python ]; then
  say " Activate the venv before running anything by hand:"
  say "     source .venv/bin/activate"
  say " (make targets pick it up automatically.)"
fi
say " Ready. Next: make smoke (zero-cost connectivity checks, docs/TESTING.md),"
say "        then read START_HERE.md and plan/README.md"
say "=============================================="
