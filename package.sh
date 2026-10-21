#!/usr/bin/env bash
# Build a clean tarball for moving this project to another machine.
#
# INCLUDED: all code, all documentation, the sampling frame if drawn.
# EXCLUDED, deliberately:
#   .env            secrets — recreate from .env.example on the new machine
#   output/         ~550MB of audio and transcripts; re-derivable, and audio
#                   should not travel around on USB sticks anyway
#   output_sim/     regenerate with `make sim` — it is deterministic (fixed seed)
#   *.sqlite        derived indexes; rebuild with `make load`
#   .git/           optional: pass --with-git to include history
set -euo pipefail
cd "$(dirname "$0")"
stamp=$(date +%Y%m%d)
name="ytpt-handoff-$stamp"
# Written inside the project (into dist/, which is gitignored) rather than to
# the parent directory: the parent may be outside whatever folder the agent has
# been granted access to, and writing there fails.
mkdir -p dist
out="dist/$name.tar.gz"
excl=(--exclude='.env' --exclude='output' --exclude='output_sim'
      --exclude='__pycache__' --exclude='*.pyc' --exclude='*.sqlite*'
      --exclude='.DS_Store' --exclude='.claude' --exclude='dist'
      --exclude='.venv' --exclude='plan/logs')
[[ "${1:-}" == "--with-git" ]] || excl+=(--exclude='.git')

# macOS ships bsdtar, which rejects GNU's --transform ("Option --transform is
# not supported"); its -s flag does the same path rewrite. Both honour the
# --exclude list above identically (checked 2026-09-17, incl. that .env is
# dropped and .env.example kept). .venv is excluded: it is machine-specific.
tar_version=$(tar --version 2>/dev/null || true)
if [[ "$tar_version" == *"GNU tar"* ]]; then
  tar czf "$out" "${excl[@]}" --transform "s,^\.,$name," .
else
  tar czf "$out" "${excl[@]}" -s ",^\.,$name," .
fi
echo "wrote $out  ($(du -h "$out" | cut -f1))"
echo
echo "On the new machine:"
echo "  tar xzf $name.tar.gz && cd $name"
echo "  cp .env.example .env    # then fill it in"
echo "  ./setup.sh"
echo "  cat START_HERE.md"
