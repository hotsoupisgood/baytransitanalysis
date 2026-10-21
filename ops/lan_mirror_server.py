#!/usr/bin/env python3
"""Serve the audio and transcript folders, read-only, to the Linux node for its mirror.

Started daily by launchd (com.ytpt.lanmirror) and stops itself after --hours. Only paths
under /<token>/ are served; everything else is 404, including the root listing, so the
token cannot be discovered by browsing. The served tree is a directory of symlinks to
the folders below and nothing else: .env, the repo and the databases are not reachable.
The Linux node pulls with `wget -m` (copy-only; nothing on the mirror is ever deleted).
"""
import argparse
import functools
import http.server
import os
import secrets
import threading
from pathlib import Path

ROOT = Path.home() / ".local/share/ytpt/lanmirror"
TOKEN_FILE = Path.home() / ".local/share/ytpt/lanmirror.token"
OUT = Path.home() / "ytpt/output"
SERVED = {  # path under <token>/output/  ->  real folder
    "audio": OUT / "audio",
    "whisper": OUT / "whisper",
    "batches": OUT / "batches",
    "subs": OUT / "subs",
    "redacted": OUT / "redacted",
    "t2": OUT / "t2",
    "fullasr/audio": OUT / "fullasr/audio",
    "fullasr/whisper": OUT / "fullasr/whisper",
}


def token() -> str:
    if not TOKEN_FILE.exists():
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(secrets.token_urlsafe(24) + "\n")
        TOKEN_FILE.chmod(0o600)
    return TOKEN_FILE.read_text().strip()


def build_tree(tok: str) -> None:
    base = ROOT / tok / "output"
    for rel, real in SERVED.items():
        link = base / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and os.readlink(link) == str(real):
            continue
        if link.is_symlink():
            link.unlink()
        if real.is_dir():
            link.symlink_to(real)
    ROOT.chmod(0o700)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8897)
    ap.add_argument("--hours", type=float, default=3.0)
    a = ap.parse_args()
    tok = token()
    build_tree(tok)
    prefix = f"/{tok}/"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def _allowed(self) -> bool:
            return self.path.startswith(prefix) and ".." not in self.path

        def do_GET(self):
            if not self._allowed():
                self.send_error(404)
                return
            super().do_GET()

        def do_HEAD(self):
            if not self._allowed():
                self.send_error(404)
                return
            super().do_HEAD()

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(
        ("0.0.0.0", a.port), functools.partial(Handler, directory=str(ROOT)))
    threading.Timer(a.hours * 3600, srv.shutdown).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
