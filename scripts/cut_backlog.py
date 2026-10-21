#!/usr/bin/env python3
"""
scripts/cut_backlog.py -- cut the 'located' backlog WITHOUT touching YouTube
(plan/BACKLOG-2026-10-18.md). A thin wrapper over pipeline/episodes.py:

  * selects only meetings that still have a 'located' episode (no no-op
    claims on classified meetings with nothing to cut);
  * skips every YouTube-platform meeting whose whole audio is NOT already on
    disk (output/audio/<id>.opus or output/t0/<id>/full.opus) -- a sibling
    worker owns YouTube downloads at 12/day;
  * runs the bodies in the stated order (coverage rule: fewest public density
    turns first; within a body episodes.py's own most-recent-first), one
    episodes.py run per body, so Granicus politeness (>= 5 s, one ffmpeg) and
    the 403/429 stop rule are episodes.py's, unchanged.

    .venv/bin/python scripts/cut_backlog.py --bodies milpitas,fremont,... [--plan]
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import episodes, state  # noqa: E402


def youtube_on_disk(con):
    ids = [r[0] for r in con.execute("SELECT video_id FROM videos WHERE kind='meeting' AND platform='youtube'")]
    out = episodes.audio_dir()
    return [v for v in ids if (out / f"{v}.opus").exists() or (episodes.T0 / v / "full.opus").exists()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", required=True, help="comma list, in the order to run")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    con = state.connect(a.state)
    ondisk = youtube_on_disk(con)
    con.close()
    quoted = ",".join("'" + v.replace("'", "''") + "'" for v in ondisk) or "''"
    episodes.SELECT_WHERE = (
        "kind = 'meeting' AND status IN ('classified', 'downloaded', 'cut') "
        "AND EXISTS (SELECT 1 FROM episodes e WHERE e.video_id = videos.video_id AND e.status = 'located') "
        f"AND (platform != 'youtube' OR video_id IN ({quoted}))")
    rc = 0
    for body in [b.strip() for b in a.bodies.split(",") if b.strip()]:
        episodes.log(f"=== body {body} ===")
        args = argparse.Namespace(max=0, bodies=body, min_gap=45, max_gap=180, plan=a.plan,
                                  requeue_failed=False, state=a.state)
        rc = episodes.run(args)
        if rc:
            episodes.log(f"STOPPED on {body} (exit {rc}); not continuing to the next body")
            return rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
