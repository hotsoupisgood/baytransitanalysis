# dashboard/ — the local progress page

```bash
make dashboard        # http://localhost:8765   (reads STATE_DB_PATH from .env)
make dashboard-demo   # the same page on synthetic data, before the pipeline exists
make status           # the same numbers as a WhatsApp-sized text digest
python3 dashboard/serve.py --once /tmp/snapshot.html      # static snapshot
python3 dashboard/status.py --new-alerts                  # what changed since last call
```

Install it as a LaunchAgent so it is always there:
`./ops/launchd/install_dashboard.sh` (see `docs/OPENCLAW_SETUP.md` §7 for the
macOS privacy caveat about `~/Documents`).

## What it shows

Hero: transcribed audio-hours against the sampling frame. Then downloaded hours
and the download rate with an ETA, hours ready for the next GPU batch, spend
against the monthly cap, failed/dead counts. Below: cumulative hours over time
against the target, a speaker × month heatmap of quota completion, where the
sampled videos sit in the pipeline, GPU batches (including any awaiting your
approval), the latest QA gate run, and the failed/dead rows with their errors.

Alerts sit at the top, most severe first: an orphan pod that is billing, the
spend cap, a batch waiting for approval, dead videos, a stalled downloader, QA
failures.

## How it is built

- `status.py` — `collect()` reads `state.sqlite` (read-only), `plan/frame_cells.csv`
  and the QA CSV, and returns one structure. `as_text()` renders the WhatsApp
  digest from that same structure, so the page and the phone can never disagree.
  `--new-alerts` prints only what is new since the last call; the heartbeat uses
  it to stay quiet when nothing has changed.
- `serve.py` — a stdlib HTTP server, loopback only, no write endpoints, a strict
  per-response CSP. Approvals happen on WhatsApp, not here: a button on a web
  page would be a second, weaker path to spending money.
- `page.html` — one file, no CDN, no framework. Values are inserted with
  `textContent`; charts are hand-drawn SVG with a crosshair tooltip, a table
  view under every chart, and a light/dark palette checked for colour-vision
  separation.
- `demo_state.py` — a synthetic `state.sqlite` shaped like `plan/SCHEMA.md`, so
  the page and its tests work before the pipeline writes a single row.

Every table and column is optional: a missing `events` table degrades the
timeline to "approximate", a missing state DB shows "the pipeline has not
started". The dashboard must not be the thing that breaks on day one.
