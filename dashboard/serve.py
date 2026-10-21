#!/usr/bin/env python3
"""
dashboard/serve.py -- local progress page for the pipeline. Standard library only.

    python3 dashboard/serve.py                 # http://127.0.0.1:8765  (make dashboard)
    python3 dashboard/serve.py --demo          # synthetic data, to see it before the pipeline exists
    python3 dashboard/serve.py --once out.html # write a static snapshot and exit

WHY IT IS BUILT THIS WAY
  * Read-only. state.sqlite is opened with mode=ro, and the server has no
    endpoint that changes anything. Approvals happen on WhatsApp, not here: a
    button on a web page is a second, weaker path to spending money.
  * Loopback only by default. The page shows spend, errors and the corpus
    design; binding 0.0.0.0 publishes that to the local network. --host with a
    non-loopback address needs --i-know-this-is-exposed.
  * One status function (dashboard/status.py) feeds this page and the WhatsApp
    "status" reply, so the two can never disagree.
  * No third-party JS or CSS. A monitoring page that breaks when a CDN changes,
    or that phones home, is a liability. Every value is inserted with
    textContent; a strict Content-Security-Policy with a per-response nonce
    blocks anything else.

Routes: /  (page)   /api/status  (JSON)   /api/status.txt  (the WhatsApp digest)   /healthz
"""
import argparse, html, json, secrets, sys, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dashboard import status as st          # noqa: E402

PAGE = Path(__file__).resolve().parent / "page.html"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def render_page(data, poll_ms, demo, nonce):
    # JSON inside <script>: escape "<" so no string in the data (a video title, an
    # error message) can close the script element.
    blob = json.dumps(data, default=str).replace("<", "\\u003c")
    return (PAGE.read_text(encoding="utf-8")
            .replace("__NONCE__", nonce)
            .replace("__POLL_MS__", str(int(poll_ms)))
            .replace("__DEMO__", "true" if demo else "false")
            .replace("__INITIAL_JSON__", blob))


def make_handler(get_status, poll_ms, demo, port):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ytpt-dashboard"

        def _send(self, code, body, ctype, nonce=None):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            if nonce:
                self.send_header("Content-Security-Policy",
                                 f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                                 "connect-src 'self'; img-src data:; base-uri 'none'; "
                                 "form-action 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):                                        # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    nonce = secrets.token_urlsafe(16)
                    self._send(200, render_page(get_status(), poll_ms, demo, nonce), "text/html; charset=utf-8", nonce)
                elif path == "/api/status":
                    self._send(200, json.dumps(get_status(), default=str), "application/json")
                elif path == "/api/status.txt":
                    self._send(200, st.as_text(get_status(), f"http://localhost:{port}"), "text/plain; charset=utf-8")
                elif path == "/healthz":
                    self._send(200, "ok", "text/plain")
                else:
                    self._send(404, "not found", "text/plain")
            except Exception as e:                               # show the error, keep serving
                self._send(500, f"<pre>{html.escape(repr(e))}</pre>", "text/html; charset=utf-8")

        def log_message(self, fmt, *args):                       # quiet: one line per non-API request
            if "/api/" not in (args[0] if args else ""):
                sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))
    return Handler


def main(argv=None):
    cfg = st.settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=cfg["port"])
    ap.add_argument("--state", default=None, help="state.sqlite (default: STATE_DB_PATH from .env)")
    ap.add_argument("--frame-cells", default=None)
    ap.add_argument("--qa", default=None)
    ap.add_argument("--poll", type=float, default=30, help="browser refresh interval, seconds")
    ap.add_argument("--demo", action="store_true", help="serve a synthetic state DB")
    ap.add_argument("--once", metavar="OUT.html", help="write a static snapshot and exit")
    ap.add_argument("--i-know-this-is-exposed", action="store_true", dest="exposed")
    a = ap.parse_args(argv)

    state, frame_cells, qa = a.state, a.frame_cells, a.qa
    if a.demo:
        from dashboard import demo_state
        state = demo_state.build(Path(tempfile.mkdtemp(prefix="ytpt-demo-")) / "state.sqlite")
        frame_cells = frame_cells or str(Path(state).with_name("no_frame_cells.csv"))
    get_status = lambda: st.collect(state, frame_cells, qa, cfg=cfg)

    if a.once:
        Path(a.once).write_text(render_page(get_status(), 0, a.demo, secrets.token_urlsafe(16)), encoding="utf-8")
        print(f"wrote {a.once}")
        return 0

    if a.host not in LOOPBACK and not a.exposed:
        sys.exit(f"refusing to bind {a.host}: this page shows spend and errors to anyone on the network. "
                 "Use an SSH tunnel or Tailscale instead, or pass --i-know-this-is-exposed.")
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(get_status, a.poll * 1000, a.demo, a.port))
    print(f"dashboard on http://{'localhost' if a.host in LOOPBACK else a.host}:{a.port}"
          f"{'  (DEMO DATA)' if a.demo else ''}  -- Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
