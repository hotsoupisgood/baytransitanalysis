#!/usr/bin/env bash
# Convenience wrapper for the exploratory WER/diff harness (legacy/hasan_compare.py).
# Usage: ./run.sh                 # 3 videos, 10-30 min
#        ./run.sh --n 5           # 5 videos
#        ./run.sh --max-min 60    # widen the length window
#
# Fixed 2026-09-17: installs into the project venv instead of `pip --user`,
# which Homebrew and system Pythons refuse (PEP 668, "externally managed"), and
# puts the venv on PATH so hasan_compare.py's `yt-dlp` subprocess is the pinned
# one rather than whatever is on the system.
set -euo pipefail
cd "$(dirname "$0")"

command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found — brew install ffmpeg"; exit 1; }
[ -x .venv/bin/python ] || ./setup.sh
# faster-whisper is not in requirements.txt: it is only needed by this
# exploratory harness (CPU transcription), not by the pipeline or the analysis.
.venv/bin/python -c 'import faster_whisper' 2>/dev/null \
  || .venv/bin/python -m pip install --quiet faster-whisper
exec env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python legacy/hasan_compare.py "$@"
