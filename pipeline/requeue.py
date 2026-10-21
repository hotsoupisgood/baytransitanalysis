#!/usr/bin/env python3
"""
pipeline/requeue.py -- put a 'failed' or 'dead' video back to work (PLAN.md 3;
the WhatsApp `retry <video_id>` command).

The row returns to the LAST GOOD state the files on disk support, so nothing is
redone unnecessarily and nothing is trusted that is not there:

    opus file present  -> transcoded
    audio file present -> downloaded
    otherwise          -> discovered

attempts resets to 0 and last_error is cleared; the events row keeps the history.
A dead row is only ever revived here, by a human's explicit request -- the
workers never do it (PLAN.md 3).

    python3 pipeline/requeue.py <video_id> [<video_id> ...]
    python3 pipeline/requeue.py --all-failed          # failed rows only, never dead
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402


def target_state(row):
    if row["opus_path"] and Path(row["opus_path"]).exists():
        return "transcoded"
    if row["audio_path"] and Path(row["audio_path"]).exists():
        return "downloaded"
    return "discovered"


def requeue(con, video_id):
    row = con.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    if not row:
        return f"{video_id}: no such video"
    if row["status"] not in ("failed", "dead"):
        return f"{video_id}: is '{row['status']}', not failed/dead; nothing done"
    to = target_state(row)
    fields = {"attempts": 0, "last_error": None}
    if to == "discovered":
        fields.update(audio_path=None, opus_path=None, transcribed_s=None)
    elif to == "downloaded":
        fields.update(opus_path=None, transcribed_s=None)
    ok = state.transition(con, video_id, row["status"], to, note="requeued by human", **fields)
    if ok and to == "transcoded" and "episode_id" in row.keys() and row["kind"] == "episode":
        # the clip's episodes row follows (it was 'failed' with the clip); gpu_batch.propose
        # only moves episodes rows that are cut / queued / transcribing
        ep = row["episode_id"] or video_id
        if con.execute("SELECT 1 FROM episodes WHERE episode_id = ?", (ep,)).fetchone():
            state.set_episode(con, ep, "cut", note="requeued by human", last_error=None)
    return f"{video_id}: {row['status']} -> {to}" if ok else f"{video_id}: state changed underneath; nothing done"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id", nargs="*")
    ap.add_argument("--all-failed", action="store_true")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    con = state.connect(a.state)
    ids = list(a.video_id)
    if a.all_failed:
        ids += [r["video_id"] for r in con.execute("SELECT video_id FROM videos WHERE status = 'failed'")]
    if not ids:
        ap.error("give at least one video_id or --all-failed")
    for vid in ids:
        print(requeue(con, vid))
    return 0


if __name__ == "__main__":
    sys.exit(main())
