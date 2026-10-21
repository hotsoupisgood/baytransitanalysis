#!/usr/bin/env python3
"""
ops/spawn_detached.py -- start a long-lived command so it survives the shell,
the exec host and this OpenClaw session. Prints the child's pid and exits 0.

    .venv/bin/python ops/spawn_detached.py --log output/logs/x.log -- ops/yt_night.sh

WHY NOT nohup, AND WHY NOT setsid: nohup ignores SIGHUP but NOT the SIGTERM an
exec host sends when it tears its process group down -- that is what killed the
first retry runner on 2026-09-28 (ops/retry_until_lands.py:92-98). The usual
answer, setsid(1), does not exist on macOS, so the escape has to be done in
process: fork, setsid(), fork again. The grandchild is a session leader with no
controlling terminal, in its own process group, reparented to launchd (ppid 1),
so neither SIGHUP nor a group SIGTERM reaches it.

WHY THE HANDLES ARE NAMED: `os.dup2(open(p, "a").fileno(), 1)` lets CPython free
the temporary file object -- closing the fd -- before or during dup2, which is
EBADF. That is the bug that killed ops/yt_1020_watchdog.py's first start on
2026-10-19 (ops/yt_1020_alert.py:78-80). Both handles are bound to names here
and kept alive until exec replaces the image.

The child inherits this process's cwd and environment, so callers that need
BATCH_TARGET_HOURS or a run spec set it before calling. os.execvp never
returns, so a failure to exec is reported through the log and exit code 127.
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="append stdout+stderr here")
    ap.add_argument("--pidfile", default=None, help="write the detached pid here")
    ap.add_argument("--chdir", default=None, help="cwd for the child (default: inherit)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- then the command and its arguments")
    a = ap.parse_args()

    cmd = [c for c in a.cmd if c != "--"]
    if not cmd:
        sys.exit("nothing to run: pass the command after --")

    log_path = Path(a.log).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # --- first fork: the parent returns to the caller immediately ----------
    r, w = os.pipe()                      # the grandchild reports its pid back
    if os.fork():
        os.close(w)
        with os.fdopen(r) as fh:
            pid = fh.read().strip()
        os.waitpid(-1, 0)                 # reap the middle child, not the grandchild
        if not pid:
            sys.exit("the detached child never reported a pid; check the log")
        print(pid)
        if a.pidfile:
            Path(a.pidfile).expanduser().write_text(f"{pid}\n", encoding="utf-8")
        return 0

    # --- middle child: new session, then fork again ------------------------
    os.close(r)
    os.setsid()                           # no controlling terminal from here on
    if os.fork():
        os._exit(0)                       # the session leader exits; the grandchild
                                          # can never reacquire a tty

    # --- grandchild: the real process --------------------------------------
    with os.fdopen(w, "w") as fh:
        fh.write(f"{os.getpid()}\n")

    if a.chdir:
        os.chdir(os.path.expanduser(a.chdir))

    log = open(log_path, "a", buffering=1)     # bound to a name on purpose
    devnull = open(os.devnull, "rb")           # ditto
    os.dup2(devnull.fileno(), 0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    try:
        os.execvp(cmd[0], cmd)
    except OSError as e:
        log.write(f"spawn_detached: cannot exec {cmd[0]}: {e}\n")
        log.flush()
        os._exit(127)


if __name__ == "__main__":
    sys.exit(main())
