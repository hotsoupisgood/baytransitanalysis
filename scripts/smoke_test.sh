#!/usr/bin/env bash
# scripts/smoke_test.sh -- level 2 of the test ladder in docs/TESTING.md.
#
# Proves the machine can run the pipeline, without renting, downloading or
# spending anything. Run after ./setup.sh, after any upgrade (brew, yt-dlp,
# OpenClaw), and before approving a GPU batch.
#
#   ./scripts/smoke_test.sh              offline checks + free credential checks
#   ./scripts/smoke_test.sh --network    + ONE metadata request to YouTube (residential Mac only)
#   ./scripts/smoke_test.sh --notify     + ONE WhatsApp test message via OpenClaw
#   ./scripts/smoke_test.sh --paid       + ONE ~$0.0001 Anthropic request (scoring request shape)
#   ./scripts/smoke_test.sh --offline    skip everything that touches the network
#
# Exit code 0 = no failures. Warnings are things to read, not blockers.
set -uo pipefail
cd "$(dirname "$0")/.."

NETWORK=0; NOTIFY=0; PAID=0; OFFLINE=0
for arg in "$@"; do
  case "$arg" in
    --network) NETWORK=1 ;; --notify) NOTIFY=1 ;; --paid) PAID=1 ;; --offline) OFFLINE=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

PY=python3;   [ -x .venv/bin/python ] && PY=.venv/bin/python
YTDLP=yt-dlp; [ -x .venv/bin/yt-dlp ] && YTDLP=.venv/bin/yt-dlp
ok=0; warn=0; fail=0
pass() { printf '  [ok]   %s\n' "$*"; ok=$((ok+1)); }
note() { printf '  [warn] %s\n' "$*"; warn=$((warn+1)); }
bad()  { printf '  [FAIL] %s\n' "$*"; fail=$((fail+1)); }
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ytpt_smoke.XXXXXX")

echo "== 1. analysis and QA code =="
if "$PY" analysis/test_pipeline.py > "$tmp/test.log" 2>&1; then
  pass "test_pipeline: $(tail -1 "$tmp/test.log")"
else
  bad "test_pipeline failed -- $(grep -c '\[FAIL\]' "$tmp/test.log") check(s); see $tmp/test.log"
fi
"$PY" analysis/qa_transcripts.py --self-test > /dev/null 2>&1 && pass "QA gate self-test" || bad "QA gate self-test"
if ls output/whisper/*.medium.json > /dev/null 2>&1; then
  mkdir -p output/qa
  "$PY" analysis/qa_transcripts.py --src output/whisper --glob '*.medium.json' --subs output/subs \
      --quiet --csv output/qa/legacy.csv > "$tmp/qa.log" 2>&1
  rc=$?
  line=$(grep -E '^[0-9]+ videos:' "$tmp/qa.log")
  # Reference result on 2026-09-17: 20 videos: 18 OK, 2 WARN, 0 FAIL.
  [ $rc -eq 0 ] && pass "QA on the 20 exploratory transcripts: $line" || bad "QA gate FAILs on exploratory transcripts: $line"
else
  note "no output/whisper/*.medium.json -- skipped the real-transcript QA run"
fi

echo; echo "== 2. audio toolchain =="
if command -v ffmpeg > /dev/null; then
  if ffmpeg -nostdin -v error -f lavfi -i "sine=frequency=440:duration=20" \
       -ac 1 -ar 16000 -c:a libopus -b:a 16k -y "$tmp/t.opus" 2> "$tmp/ff.log"; then
    info=$(ffprobe -v error -show_entries format=duration:stream=codec_name,channels -of default=nw=1 "$tmp/t.opus")
    dur=$(printf '%s\n' "$info" | sed -n 's/^duration=//p')
    # Opus always REPORTS 48000 Hz; the 16 kHz input is what matters. Do not "fix" that.
    codec_ok=0
    case "$info" in *codec_name=opus*) case "$info" in *channels=1*) codec_ok=1 ;; esac ;; esac
    if [ $codec_ok -eq 1 ] && "$PY" -c "import sys; sys.exit(0 if abs(float('${dur:-0}') - 20) < 0.2 else 1)"; then
      pass "ffmpeg -> 16 kHz mono Opus 16 kbps (PLAN.md 4.3 transcode) works"
    else bad "transcode produced unexpected output: $info"; fi
  else bad "ffmpeg transcode failed: $(tail -1 "$tmp/ff.log")"; fi
else bad "ffmpeg missing"; fi
if command -v deno > /dev/null; then pass "deno present (yt-dlp JS runtime)"
else note "deno missing -- yt-dlp YouTube downloads need it (brew install deno)"; fi

echo; echo "== 3. dashboard, status digest, notifier =="
"$PY" dashboard/serve.py --demo --once "$tmp/dash.html" > /dev/null 2>&1 && grep -q "Pipeline monitor" "$tmp/dash.html" \
  && pass "dashboard renders (demo snapshot)" || bad "dashboard failed to render"
if "$PY" dashboard/status.py > "$tmp/status.txt" 2>&1; then
  pass "status digest: $(sed -n 2p "$tmp/status.txt")"
else bad "dashboard/status.py crashed -- see $tmp/status.txt"; fi
if "$PY" pipeline/notify.py --dry-run info "smoke test" > "$tmp/notify.txt" 2>&1; then
  pass "notifier builds: $(head -c 90 "$tmp/notify.txt")"
else note "notifier not configured: $(tail -1 "$tmp/notify.txt")"; fi
if [ -f plan/catalog.tsv ]; then
  "$PY" plan/build_frame.py > "$tmp/frame.txt" 2>&1 && pass "build_frame dry run on plan/catalog.tsv" \
    || bad "build_frame rejects plan/catalog.tsv: $(tail -2 "$tmp/frame.txt" | head -1)"
fi

if [ $OFFLINE -eq 0 ]; then
  echo; echo "== 4. credentials (free API calls) =="
  if "$PY" scripts/check_credentials.py $([ $PAID -eq 1 ] && echo --paid) > "$tmp/cred.txt" 2>&1; then
    pass "$(tail -1 "$tmp/cred.txt")"
  else bad "$(tail -1 "$tmp/cred.txt")"; fi
  grep -E '\[(FAIL|warn)' "$tmp/cred.txt" | sed 's/^/        /'

  echo; echo "== 5. OpenClaw gateway =="
  if command -v openclaw > /dev/null; then
    if openclaw gateway status > "$tmp/oc.txt" 2>&1; then pass "gateway: $(head -1 "$tmp/oc.txt")"
    else note "openclaw gateway not running: $(head -1 "$tmp/oc.txt") -- see docs/OPENCLAW_SETUP.md"; fi
  else note "openclaw CLI not installed on this Mac (fine unless this is the supervisor Mac)"; fi
fi

if [ $NETWORK -eq 1 ] && [ $OFFLINE -eq 0 ]; then
  echo; echo "== 6. YouTube reachability (one metadata request, no download) =="
  if "$YTDLP" --skip-download --no-playlist --print "%(id)s %(duration)s" \
       "https://www.youtube.com/watch?v=PDTon6oss9M" > "$tmp/yt.out" 2> "$tmp/yt.err"; then
    pass "yt-dlp metadata fetch: $(cat "$tmp/yt.out")"
  else bad "yt-dlp metadata fetch failed: $(tail -1 "$tmp/yt.err")"; fi
  grep -qiE "sign in to confirm|not a bot" "$tmp/yt.err" && bad "YouTube is bot-checking this IP -- stop, do not retry in a loop"
  grep -qiE "js runtime|javascript runtime|n challenge|signature" "$tmp/yt.err" \
    && note "yt-dlp JS challenge warnings: $(grep -iE 'runtime|challenge|signature' "$tmp/yt.err" | head -1)"
  grep -qi "impersonat" "$tmp/yt.err" && note "curl_cffi missing (impersonation warning)"
fi

if [ $NOTIFY -eq 1 ] && [ $OFFLINE -eq 0 ]; then
  echo; echo "== 7. WhatsApp =="
  "$PY" pipeline/notify.py info "smoke test from $(hostname -s) at $(date '+%H:%M'). Reply-free; ignore." \
    && pass "test message handed to OpenClaw -- check it arrived on the phone" \
    || bad "notify failed (queued in outbox; see message above)"
fi

echo
echo "=============================================="
echo " smoke test: $ok ok, $warn warnings, $fail failures   (logs: $tmp)"
echo "=============================================="
[ $fail -eq 0 ]
