#!/usr/bin/env bash
# Metadata-only crawl. Downloads NO audio. Produces plan/catalog.tsv
#
# Channels come from plan/channels.tsv (speaker <TAB> channel-tab URL). A speaker
# may have several rows (a clips channel and a VOD channel, say): the stratum is
# the SPEAKER, not the channel (SAMPLING.md).
#
# Output columns, tab-separated, title LAST because titles are free text:
#   speaker channel_url video_id upload_date duration_s view_count live_status crawled_at title
#
# Exit code is non-zero if any channel failed or returned zero rows. Do not draw
# the frame from an incomplete catalog: a missing speaker silently becomes a
# study-design change.
#
# Fixed 2026-09-17. The previous version passed "%(id)s\t..." inside ordinary
# double quotes, so yt-dlp printed a literal backslash-t, every row collapsed into
# one column, and build_frame.py discarded all of them as "no date". It also sent
# yt-dlp's stderr to /dev/null, hiding bot checks and rate limits.
set -euo pipefail
cd "$(dirname "$0")/.."

YTDLP=yt-dlp
[ -x .venv/bin/yt-dlp ] && YTDLP=.venv/bin/yt-dlp
command -v "$YTDLP" >/dev/null 2>&1 || { echo "yt-dlp not found -- run ./setup.sh" >&2; exit 1; }

CHANNELS=plan/channels.tsv
OUT=plan/catalog.tsv
LOGDIR=plan/logs
[ -f "$CHANNELS" ] || { echo "missing $CHANNELS" >&2; exit 1; }
mkdir -p "$LOGDIR"

TAB=$'\t'
# $'...' produces REAL tab characters.
FMT=$'%(id)s\t%(upload_date)s\t%(duration)s\t%(view_count)s\t%(live_status)s\t%(title)s'
stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

tmp="$OUT.partial"
printf 'speaker\tchannel_url\tvideo_id\tupload_date\tduration_s\tview_count\tlive_status\tcrawled_at\ttitle\n' > "$tmp"

failed=0
while IFS="$TAB" read -r speaker url kind _rest || [ -n "${speaker:-}" ]; do
  case "$speaker" in ''|\#*) continue ;; esac
  # kind=meeting rows (2026-09-21) belong to pipeline/discover.py --meetings, not this crawl
  case "${kind:-show}" in meeting) continue ;; esac
  if [ -z "${url:-}" ]; then
    echo "!!! $CHANNELS: row for '$speaker' has no URL (is the separator a TAB?)" >&2
    failed=1; continue
  fi
  tag=$(printf '%s' "$url" | tr -c 'A-Za-z0-9' '_' | tail -c 48)
  log="$LOGDIR/crawl_${speaker}_${tag}.log"
  echo ">>> $speaker  $url" >&2
  before=$(wc -l < "$tmp")
  # awk, not sed, to prefix columns: BSD sed on macOS does not expand \t.
  # Titles containing tabs arrive as extra fields; rejoin them with spaces.
  if "$YTDLP" --flat-playlist \
       --extractor-args "youtubetab:approximate_date" \
       --print "$FMT" \
       --sleep-requests 1 \
       "$url" 2> "$log" \
     | awk -F "$TAB" -v OFS="$TAB" -v sp="$speaker" -v url="$url" -v ts="$stamp" '
         NF >= 6 { t = $6; for (i = 7; i <= NF; i++) t = t " " $i
                   print sp, url, $1, $2, $3, $4, $5, ts, t }' >> "$tmp"; then
    :
  else
    echo "!!! $speaker: yt-dlp exited non-zero -- see $log" >&2
    failed=1
  fi
  n=$(( $(wc -l < "$tmp") - before ))
  echo "    $n videos   (log: $log)" >&2
  if [ "$n" -eq 0 ]; then
    echo "!!! $speaker: ZERO rows from $url" >&2
    failed=1
  fi
  if grep -qiE "sign in to confirm|not a bot|HTTP Error 429" "$log"; then
    echo "!!! $speaker: YouTube is challenging this IP (see $log). Stop and wait; do not retry in a loop." >&2
    failed=1
  fi
done < "$CHANNELS"

mv -f "$tmp" "$OUT"
echo >&2
echo "Wrote $OUT ($(( $(wc -l < "$OUT") - 1 )) videos)" >&2
tail -n +2 "$OUT" | cut -f1 | sort | uniq -c >&2
if [ "$failed" -ne 0 ]; then
  echo "CRAWL INCOMPLETE -- fix the failures above before running build_frame.py --draw." >&2
  exit 1
fi
