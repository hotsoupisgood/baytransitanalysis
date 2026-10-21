#!/usr/bin/env bash
# Back up the irreplaceable artefacts to Backblaze B2 via the b2 CLI (brew install b2-tools).
#
# WHAT IS BACKED UP, AND WHY ONLY THIS:
#   output/audio/     m4a originals   -- re-downloading costs WEEKS of
#                                        rate-limited crawling, and videos get
#                                        deleted in the meantime
#   output/fullasr/audio/, output/t2/, output/subs/
#                     whole-meeting audio and captions -- the same downloads
#   output/whisper/, output/fullasr/whisper/, output/batches/
#                     transcript JSON -- re-transcribing costs real GPU money
#   plan/frame.csv    the corpus definition -- irreproducible once drawn
#   state.sqlite      the pipeline state -- see below; added 2026-09-24
#
# COPY, NEVER SYNC. The bucket keeps only the latest version of a file and
# deletes hidden versions after one day, so `rclone sync` would erase a backup
# one day after the local file went missing -- the case a backup exists for.
#
# DELIBERATELY NOT BACKED UP (all cheaply regenerable):
#   output/opus/    one ffmpeg command from the m4a, seconds per file
#   output_sim/     deterministic from a fixed seed
#   .env            secrets never leave the machine
#
# WHY state.sqlite IS NOW BACKED UP. This file used to say `make load` rebuilds
# it from the JSON. That stopped being true. `make load` rebuilds the *analysis*
# corpus from transcripts; it does not rebuild the source catalogue, the episode
# locations found by four different locate routes, the spend ledger, the batch
# and pod history, the backoff rows, or `classifier_calls` -- which on
# 2026-09-24 held 2,376 rows of labels, rationales and prompt versions
# representing $7.20 of purchased Claude output and the provenance the analysis
# depends on. None of that is in a transcript.
#
# The snapshot uses VACUUM INTO, not cp: the database runs in WAL mode with
# several writers, so a plain copy can catch a torn state and will miss a
# write-ahead log that has been seen at 4 MB.
#
# The rule: back up what you cannot regenerate cheaply. Everything else is a
# build artefact.
set -euo pipefail
cd "$(dirname "$0")"

# One run at a time: the nightly launchd job must not start while a long upload
# is still going. A lock left by a dead run is cleared.
lock="$HOME/.cache/ytpt-backup.lock"; mkdir -p "$HOME/.cache"
if ! mkdir "$lock" 2>/dev/null; then
  if [ -f "$lock/pid" ] && kill -0 "$(cat "$lock/pid")" 2>/dev/null; then
    echo "backup already running (pid $(cat "$lock/pid")); exiting"; exit 0
  fi
  rm -rf "$lock"; mkdir "$lock"
fi
echo $$ > "$lock/pid"; trap 'rm -rf "$lock"' EXIT

# Load .env so credentials live in ONE place.
# Read only the B2_* and STATE_DB_PATH lines rather than sourcing the file: .env
# is written for python-dotenv, and unquoted values with spaces break `. ./.env`.
if [ -f .env ]; then
  while IFS='=' read -r k v; do
    v="${v%\"}"; v="${v#\"}"; v="${v%\'}"; v="${v#\'}"
    export "$k=$v"
  done < <(grep -E '^(export +)?(B2_[A-Z_]+|STATE_DB_PATH)=' .env | sed -E 's/^export +//')
fi

BUCKET="${B2_BUCKET:?set B2_BUCKET in .env}"
DEST="b2://$BUCKET${B2_PREFIX:+/$B2_PREFIX}"

# Uploads use Backblaze's own CLI (`brew install b2-tools`), not rclone. rclone's
# B2 backend calls b2_list_buckets on API v1, which returns 500 internal_error
# for a key with a file-name prefix restriction (tested 2026-10-20: v1 500, v2
# and v3 fine); rclone then tries to create the bucket and fails with
# duplicate_bucket_name. The b2 CLI uses v3.
# B2 calls the key id "keyID" in the console and "account" in the API, so accept
# any of the spellings rather than making anyone rename their .env.
B2_ID="${B2_ACCOUNT_ID:-${B2_KEY_ID:-${B2_KEYID:-}}}"
export B2_APPLICATION_KEY_ID="${B2_ID:?set B2_KEY_ID in .env}"
export B2_APPLICATION_KEY="${B2_APPLICATION_KEY:-${B2_APP_KEY:?set B2_APPLICATION_KEY in .env}}"

DRY=""
[[ "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

command -v b2 >/dev/null || { echo "b2 not installed: brew install b2-tools"; exit 1; }

# `b2 sync` never deletes on the destination unless given --delete or
# --keep-days, so this is a copy. In-flight downloads (*.part, yt-dlp temp
# files) are skipped; the next run takes them.
common=(--threads 8 --exclude-regex '.*\.part(\..*)?$' --exclude-regex '.*\.ytdl$' --exclude-regex '.*\.wav$')
[ -t 1 ] || common+=(--no-progress)
[ -n "$DRY" ] && common+=("$DRY")

# WAV goes up as Opus (Puck, 2026-10-20: "opus"): 16 kHz mono speech at 24 kbps,
# about a tenth of the size. Encoded on the fly into a temp file; the local WAV is
# never touched. A ledger of path|size|mtime skips files already done.
wav_ledger="$HOME/.cache/ytpt-backup/wav2opus.done"
mkdir -p "$(dirname "$wav_ledger")"; touch "$wav_ledger"
wav_to_opus() {  # wav_to_opus <dir>
  local tmpd; tmpd=$(mktemp -d)
  find "$1" -type f -iname '*.wav' | sort | while read -r f; do
    key="$f|$(stat -f '%z|%m' "$f")"
    grep -qxF "$key" "$wav_ledger" && continue
    rel="${f%.*}.opus"; [ -e "$rel" ] && rel="$f.opus"   # never overwrite a real .opus
    if [ -n "$DRY" ]; then echo "(dry run: would upload $rel as opus)"; continue; fi
    ffmpeg -nostdin -v error -y -i "$f" -c:a libopus -b:a 24k -application voip -ac 1 "$tmpd/x.opus" \
      && b2 file upload --no-progress "$BUCKET" "$tmpd/x.opus" "${B2_PREFIX:+$B2_PREFIX/}$rel" >/dev/null \
      && echo "$key" >> "$wav_ledger" && echo "opus $rel"
    rm -f "$tmpd/x.opus"
  done
  rm -rf "$tmpd"
}

# Audio is uploaded by default (Puck raised the B2 caps, 2026-10-20 12:13). B2_AUDIO=0 is
# the free-tier mode: transcripts and DB snapshots only, inside the free 10 GB, with the
# audio kept on the Mac and the Linux node's mirror (~/ytpt-mirror).
srcs="output/whisper output/fullasr/whisper output/batches output/redacted output/subs"
[ "${B2_AUDIO:-1}" = 1 ] && srcs="$srcs output/audio output/fullasr/audio output/t2"
# One file that fails (a DNS blip on 2026-10-20 failed one upload and `set -e` then skipped
# fullasr/audio, t2, the DB snapshots and the meeting names) must not stop the rest: carry
# on, and exit non-zero at the end. The next run re-sends only what is missing.
incomplete=""
for src in $srcs; do
  [ -d "$src" ] || continue
  echo ">>> $src  $(date -u +%H:%M:%SZ)"
  b2 sync "${common[@]}" "$src" "$DEST/$src" || { echo "!!! $src incomplete, continuing"; incomplete="$incomplete $src"; }
  wav_to_opus "$src"
done

up_file() {  # up_file <local file> <remote dir>
  local tmp; tmp=$(mktemp -d)
  cp -p "$1" "$tmp/"
  b2 sync "${common[@]}" "$tmp" "$DEST/$2"
  rm -rf "$tmp"
}

if [ -f plan/frame.csv ]; then
  echo ">>> plan/frame.csv"
  up_file plan/frame.csv plan
fi

# state.sqlite, via a consistent snapshot. VACUUM INTO reads through WAL and
# refuses to write a torn file, so this is safe while the workers are running;
# `cp` is not. The snapshot is kept locally too -- a restore you cannot do
# without network access is not much of a restore.
STATE_DB="${STATE_DB_PATH:-$HOME/.local/share/ytpt/state.sqlite}"
STATE_DB="${STATE_DB/#\~/$HOME}"
# corpus.sqlite (the loaded analysis corpus, transcript text included) rides along;
# it is rebuildable with `make load`, but it is small and the rebuild is not.
CORPUS_DB="$(dirname "$STATE_DB")/corpus.sqlite"
if [ -f "$CORPUS_DB" ] && [ -z "$DRY" ]; then
  csnap="$HOME/ytpt-backups/corpus-$(date -u +%Y%m%dT%H%M%SZ).sqlite"; mkdir -p "$HOME/ytpt-backups"
  echo ">>> $CORPUS_DB"
  sqlite3 "file:$CORPUS_DB?mode=ro" "VACUUM INTO '$csnap'"
  [ "$(sqlite3 "$csnap" 'PRAGMA integrity_check')" = ok ] && up_file "$csnap" state
  ls -1t "$HOME/ytpt-backups"/corpus-*.sqlite | tail -n +4 | xargs -r rm --
fi
if [ -f "$STATE_DB" ]; then
  snap_dir="$HOME/ytpt-backups"
  mkdir -p "$snap_dir"
  snap="$snap_dir/state-$(date -u +%Y%m%dT%H%M%SZ).sqlite"
  echo ">>> $STATE_DB"
  if [ -n "$DRY" ]; then
    echo "(dry run: would snapshot to $snap and upload)"
  else
    sqlite3 "file:$STATE_DB?mode=ro" "VACUUM INTO '$snap'"
    [ "$(sqlite3 "$snap" 'PRAGMA integrity_check')" = ok ] || {
      echo "snapshot failed integrity_check: $snap"; exit 1; }
    up_file "$snap" state
    # Keep the 14 most recent local snapshots; B2 keeps the rest.
    ls -1t "$snap_dir"/state-*.sqlite | tail -n +15 | xargs -r rm --
  fi
fi

# Descriptive names for the whole meetings (2026-10-20, Puck on WhatsApp).
# The raw trees above are keyed by opaque video ids. ops/meeting_copier.py adds a
# second name per whole meeting -- meetings/<Body>/<date> <Body> - <title>.<ext> --
# with a B2 SERVER-SIDE copy of the raw object just uploaded, so it costs no
# upload bandwidth, and refreshes meetings/INDEX.csv. Runs after the uploads so
# tonight's new meetings are already in the bucket. Non-fatal: a failure here
# must never fail the backup.
if [ "${B2_AUDIO:-1}" = 1 ] && [ -z "$DRY" ] && [ -x "$(command -v python3)" ] && [ -f ops/meeting_copier.py ]; then
  echo ">>> descriptive meeting names  $(date -u +%H:%M:%SZ)"
  python3 -I ops/meeting_copier.py --once || echo "meeting_copier failed (backup itself is fine)"
fi

# Keep the newest 14 state and 3 corpus snapshots in B2 (each ~0.05-0.07 GB);
# older ones are deleted so the free 10 GB is never reached.
if [ -z "$DRY" ]; then
  for kind_keep in state:14 corpus:3; do
    kind=${kind_keep%%:*}; keep=${kind_keep##*:}
    b2 ls "$DEST/state/" 2>/dev/null | grep -E "/${kind}-[0-9TZ]+\.sqlite$" | sort -r | tail -n +$((keep + 1)) |
      while read -r old; do b2 rm --versions --no-progress "b2://$BUCKET/$old" >/dev/null && echo "pruned $old"; done
  done
fi

echo
echo "done $(date -u +%FT%TZ) -> $DEST"
echo "(B2 is \$6.95/TB/month — divide GB by 144 for dollars/month)"
if [ -n "$incomplete" ]; then echo "!!! incomplete:$incomplete (re-run to finish)"; exit 1; fi
