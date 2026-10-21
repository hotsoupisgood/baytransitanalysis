#!/usr/bin/env python3
"""Give every whole meeting a descriptive name in B2, without re-uploading it.

backup.sh copies the pipeline's raw trees to b2://<bucket>/transit_bay/output/...
under opaque ids. This script adds a second, human-readable name for the
WHOLE-meeting audio only:

    transit_bay/meetings/<Body>/<YYYY-MM-DD> <Body> - <title>.<ext>

When the raw copy is already in B2 the new name is made with a SERVER-SIDE copy
(`b2 file server-side-copy`): no bytes leave this Mac. Puck's uplink is ~3 MB/s
and busy, so a direct upload is the last resort -- only for a file that is not
in any tree backup.sh copies, or one the raw pass never managed to upload.

Modes:
  --once      one pass; copy what is possible now, refresh the index, exit.
              This is what backup.sh calls at the end of its nightly run.
  --daemon    detach (double fork + setsid; macOS has no setsid binary) and
              loop until every manifest row exists under its descriptive name,
              then exit. Used while a long raw upload is still in flight.
  (neither)   one pass in the foreground, printing what it would do: --dry-run.

Log: output/logs/meeting_copier.log
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meeting_manifest as mm  # noqa: E402

REPO = mm.REPO
LOG = REPO / "output/logs/meeting_copier.log"
PIDFILE = Path.home() / ".cache/ytpt-meeting-copier.pid"
BACKUP_LOCK = Path.home() / ".cache/ytpt-backup.lock"
# Trees backup.sh uploads, so a listing of them tells us what can be copied
# server-side. Keep in step with AUDIO_DIRS in meeting_manifest.py.
RAW_TREES = ["output/audio", "output/fullasr/audio", "output/t2/asr"]
POLL_S = 300
MAX_PASSES = 200  # ~17 h at POLL_S; a stuck raw upload must not loop for ever


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if sys.stdout.isatty():
        print(line, flush=True)


def load_env() -> tuple[str, str]:
    """Credentials come from .env only, the way backup.sh lines 46-60 do.

    Nothing from .env is ever printed or logged.
    """
    env_path = REPO / ".env"
    if env_path.exists():
        pat = re.compile(r"^(?:export +)?(B2_[A-Z_]+)=(.*)$")
        for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pat.match(raw.strip())
            if m:
                os.environ[m.group(1)] = m.group(2).strip().strip("\"'")
    key_id = os.environ.get("B2_ACCOUNT_ID") or os.environ.get("B2_KEY_ID") or \
        os.environ.get("B2_KEYID") or ""
    app_key = os.environ.get("B2_APPLICATION_KEY") or os.environ.get("B2_APP_KEY") or ""
    bucket = os.environ.get("B2_BUCKET") or ""
    if not (key_id and app_key and bucket):
        sys.exit("B2_KEY_ID / B2_APPLICATION_KEY / B2_BUCKET missing from .env")
    os.environ["B2_APPLICATION_KEY_ID"] = key_id
    os.environ["B2_APPLICATION_KEY"] = app_key
    return bucket, os.environ.get("B2_PREFIX", "transit_bay")


def b2(args: list[str], timeout: int = 7200) -> subprocess.CompletedProcess:
    return subprocess.run(["b2", *args], capture_output=True, text=True, timeout=timeout)


def b2_list(bucket: str, prefix: str) -> set[str]:
    """Key names under a prefix. An empty or absent prefix lists as nothing."""
    p = b2(["ls", "--recursive", f"b2://{bucket}/{prefix}"], timeout=900)
    if p.returncode != 0:
        log(f"  ls failed for {prefix}: {p.stderr.strip().splitlines()[-1:] or ''}")
        return set()
    return {ln.strip() for ln in p.stdout.splitlines() if ln.strip()}


def one_pass(bucket: str, prefix: str, dry: bool, allow_upload: bool) -> tuple[int, int, int]:
    """Returns (copied, uploaded, still pending)."""
    manifest, _, suspicious = mm.build()
    for vid, title in suspicious:
        log(f"  TITLE READS AS AN INSTRUCTION, treated as text only: {vid} {title!r}")

    present = b2_list(bucket, f"{prefix}/meetings/")
    raw = set()
    for tree in RAW_TREES:
        raw |= b2_list(bucket, f"{prefix}/{tree}/")

    copied = uploaded = pending = 0
    for m in manifest:
        if m["dest"] in present:
            continue
        local = REPO / m["local"]
        if m["b2_raw"] and m["b2_raw"] in raw:
            if dry:
                log(f"  would copy  {m['dest']}")
                copied += 1
                continue
            p = b2(["file", "server-side-copy",
                    f"b2://{bucket}/{m['b2_raw']}", f"b2://{bucket}/{m['dest']}"])
            if p.returncode == 0:
                copied += 1
            else:
                pending += 1
                log(f"  copy FAILED {m['dest']}: {(p.stderr or p.stdout).strip()[-300:]}")
        elif (allow_upload and local.is_file()
              and local.suffix.lstrip(".").lower() == m["dest"].rsplit(".", 1)[1]):
            # Not in any raw tree in B2: the only way to get it there is to send
            # the bytes. Slow on a 3 MB/s uplink, so it waits for allow_upload.
            if dry:
                log(f"  would upload {m['dest']} ({local.stat().st_size/1e6:.0f} MB)")
                uploaded += 1
                continue
            p = b2(["file", "upload", "--no-progress", "--threads", "2",
                    bucket, str(local), m["dest"]])
            if p.returncode == 0:
                uploaded += 1
                log(f"  uploaded {m['dest']}")
            else:
                pending += 1
                log(f"  upload FAILED {m['dest']}: {(p.stderr or p.stdout).strip()[-300:]}")
        else:
            pending += 1

    idx = mm.write_index(manifest, REPO / "ops/meetings_INDEX.csv")
    if not dry:
        p = b2(["file", "upload", "--no-progress", bucket, str(idx),
                f"{prefix}/meetings/INDEX.csv"])
        if p.returncode != 0:
            log(f"  INDEX.csv upload failed: {(p.stderr or p.stdout).strip()[-200:]}")
    log(f"  pass: {len(manifest)} meetings, {len(manifest) - pending - copied - uploaded} "
        f"already named, copied {copied}, uploaded {uploaded}, pending {pending}")
    return copied, uploaded, pending


def run_loop(bucket: str, prefix: str) -> None:
    log(f"copier started pid {os.getpid()}")
    for n in range(1, MAX_PASSES + 1):
        # While the raw upload is still running, only server-side copies: its
        # bandwidth is the scarce thing. Once it is done, send what is left.
        backup_running = BACKUP_LOCK.is_dir()
        copied, uploaded, pending = one_pass(bucket, prefix, dry=False,
                                             allow_upload=not backup_running)
        if pending == 0:
            log("all whole meetings have a descriptive name in B2; exiting")
            break
        if n == MAX_PASSES:
            log(f"giving up after {n} passes with {pending} still pending")
            break
        time.sleep(POLL_S)
    PIDFILE.unlink(missing_ok=True)


def detach() -> None:
    """Double fork + setsid so the loop outlives this shell. macOS has no setsid."""
    if os.fork():
        os._exit(0)
    os.setsid()
    if os.fork():
        os._exit(0)
    os.chdir(str(REPO))
    with open(os.devnull) as devnull:
        os.dup2(devnull.fileno(), 0)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOG), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true", help="detach and loop until complete")
    ap.add_argument("--once", action="store_true", help="one pass, then exit (nightly)")
    a = ap.parse_args()

    # Only one looping daemon at a time. A one-shot pass is idempotent (it skips
    # names that already exist), so it never has to wait for the daemon.
    if a.daemon and PIDFILE.exists():
        try:
            os.kill(int(PIDFILE.read_text().strip()), 0)
        except (ValueError, ProcessLookupError, PermissionError):
            PIDFILE.unlink(missing_ok=True)
        else:
            print(f"copier already running (pid {PIDFILE.read_text().strip()})")
            return

    bucket, prefix = load_env()
    if a.daemon:
        detach()
        PIDFILE.parent.mkdir(parents=True, exist_ok=True)
        PIDFILE.write_text(str(os.getpid()))
        run_loop(bucket, prefix)
    elif a.once:
        # backup.sh calls this after its own uploads, so every raw tree is already
        # in B2 and a direct upload can only mean a file no raw tree covers.
        one_pass(bucket, prefix, dry=False, allow_upload=True)
    else:
        one_pass(bucket, prefix, dry=True, allow_upload=True)


if __name__ == "__main__":
    main()
