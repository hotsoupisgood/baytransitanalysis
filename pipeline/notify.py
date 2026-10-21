#!/usr/bin/env python3
"""
pipeline/notify.py -- the pipeline's one notify(level, message).

Puck reads the pipeline on WhatsApp through OpenClaw (plan/DECISIONS.md #7,
2026-09-17). Every script calls notify(); nothing else talks to a chat app.

Backends, chosen by NOTIFY_BACKEND in .env:
  openclaw (default)  openclaw message send --channel $OPENCLAW_NOTIFY_CHANNEL
                        --target $OPENCLAW_NOTIFY_TARGET --message <text>
                      Channel defaults to whatsapp; target is Puck's number in
                      E.164 (+1...). Needs the OpenClaw gateway running here.
  discord             POST to DISCORD_WEBHOOK_URL (errors to DISCORD_ERROR_WEBHOOK_URL)
  none                print only

WHY A FAILURE NEVER RAISES: a notifier that can crash the downloader is worse
than no notifier. An undelivered message goes to an outbox file next to
state.sqlite and `notify.py --flush` resends it (the supervisor's heartbeat runs
that). WHY NO PER-VIDEO MESSAGES: 2,000 pings is noise; callers send rolling
summaries plus immediate alerts (PLAN.md 4.7).

Levels: info | warn | error | approval

CLI:
  python3 pipeline/notify.py info "batch 3 done: 41 videos, 52.0 audio-h, $0.71"
  python3 pipeline/notify.py --dry-run warn "test"     # print the command only
  python3 pipeline/notify.py --flush                   # resend the outbox
"""
import argparse, json, os, re, shlex, subprocess, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVELS = ("info", "warn", "error", "approval")
MAX_CHARS = 1500           # long enough for a status digest, short enough to read on a phone


def load_env(path=ROOT / ".env"):
    """KEY=value lines; inline '# comments' stripped; never overrides the real env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = re.sub(r"\s+#.*$", "", v.strip()).strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


def outbox_path():
    state = os.environ.get("STATE_DB_PATH") or "~/.local/share/ytpt/state.sqlite"
    return Path(os.path.expanduser(state)).parent / "notify_outbox.jsonl"


def format_message(level, message):
    tag = {"info": "", "warn": " WARN", "error": " ERROR", "approval": " APPROVAL NEEDED"}[level]
    text = f"[ytpt{tag}] {message.strip()}"
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS - 15] + " ...(truncated)"


def openclaw_bin():
    """OPENCLAW_BIN, else `openclaw` on PATH, else the copy OpenClaw installs
    under ~/.openclaw/tools (launchd jobs and the pod launcher run with a bare
    PATH: b3, 2026-09-18, 'No such file or directory: openclaw')."""
    explicit = os.environ.get("OPENCLAW_BIN")
    if explicit:
        return explicit
    import glob, shutil
    found = shutil.which("openclaw")
    if found:
        return found
    cands = sorted(glob.glob(os.path.expanduser("~/.openclaw/tools/node-*/bin/openclaw")))
    return cands[-1] if cands else "openclaw"


def openclaw_command(text):
    target = os.environ.get("OPENCLAW_NOTIFY_TARGET", "").strip()
    if not target:
        raise RuntimeError("OPENCLAW_NOTIFY_TARGET is not set (E.164 number, e.g. +15551234567)")
    return [openclaw_bin(), "message", "send",
            "--channel", os.environ.get("OPENCLAW_NOTIFY_CHANNEL", "whatsapp"),
            "--target", target, "--message", text]


def _send(level, text, dry_run=False):
    backend = (os.environ.get("NOTIFY_BACKEND") or "openclaw").strip().lower()
    if backend == "none":
        print(text)
        return True
    if backend == "openclaw":
        cmd = openclaw_command(text)
        if dry_run:
            print(shlex.join(cmd))
            return True
        # The openclaw script is `#!/usr/bin/env node`; make its own node visible
        # even from a bare launchd/launcher PATH ("env: node: No such file").
        env = dict(os.environ)
        bindir = os.path.dirname(cmd[0])                  # empty for a bare name: do not put "" (= cwd) on PATH
        env["PATH"] = (bindir + os.pathsep if bindir else "") + env.get("PATH", "/usr/bin:/bin")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"openclaw exit {r.returncode}: {(r.stderr or r.stdout).strip()[:300]}")
        return True
    if backend == "discord":
        url = os.environ.get("DISCORD_ERROR_WEBHOOK_URL") if level in ("error", "approval") else None
        url = url or os.environ.get("DISCORD_WEBHOOK_URL")
        if not url:
            raise RuntimeError("DISCORD_WEBHOOK_URL is not set")
        if dry_run:
            print(f"POST <discord webhook> {text!r}")
            return True
        req = urllib.request.Request(url, data=json.dumps({"content": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
        return True
    raise RuntimeError(f"unknown NOTIFY_BACKEND={backend!r}")


def notify(level, message, dry_run=False):
    """Send one message. Returns True if delivered; on failure queues it in the
    outbox and returns False. Never raises."""
    load_env()
    level = level if level in LEVELS else "info"
    text = format_message(level, message)
    try:
        return _send(level, text, dry_run)
    except Exception as e:                                   # noqa: BLE001 -- by design
        if dry_run:
            print(f"notify (dry run): would fail: {e}", file=sys.stderr)
            return False
        try:
            box = outbox_path()
            box.parent.mkdir(parents=True, exist_ok=True)
            with box.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "level": level, "text": text, "error": str(e)}) + "\n")
        except OSError:
            pass
        print(f"notify: not delivered ({e}); queued in outbox", file=sys.stderr)
        return False


def flush():
    """Resend queued messages oldest first; keep the ones that still fail."""
    load_env()
    box = outbox_path()
    if not box.exists():
        print("outbox empty")
        return 0
    pending, kept = [json.loads(l) for l in box.read_text().splitlines() if l.strip()], []
    for m in pending:
        try:
            _send(m["level"], m["text"])
        except Exception as e:                               # noqa: BLE001
            m["error"] = str(e)
            kept.append(m)
    box.write_text("".join(json.dumps(m) + "\n" for m in kept), encoding="utf-8")
    print(f"flushed {len(pending) - len(kept)}, still queued {len(kept)}")
    return 0 if not kept else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level", nargs="?", choices=LEVELS)
    ap.add_argument("message", nargs="?")
    ap.add_argument("--dry-run", action="store_true", help="print what would be sent")
    ap.add_argument("--flush", action="store_true", help="resend queued messages")
    a = ap.parse_args(argv)
    if a.flush:
        return flush()
    if not a.level or a.message is None:
        ap.error("give a level and a message, or --flush")
    return 0 if notify(a.level, a.message, dry_run=a.dry_run) else 1


if __name__ == "__main__":
    sys.exit(main())
