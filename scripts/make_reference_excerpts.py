#!/usr/bin/env python3
"""
scripts/make_reference_excerpts.py -- cut editable excerpts out of transcript
JSON so a human can correct them into a reference (Phase 0, PLAN.md 6).

    .venv/bin/python scripts/make_reference_excerpts.py [--out DIR] VIDEO_ID:START_S:DURATION_S ...

Writes output/reference/<video_id>.<start>-<end>.txt (or DIR/...): one line per segment,
"[mm:ss] SPEAKER_XX: text". Edit the text (and the speaker label if wrong),
keep the timestamps. The corrected file becomes the WER reference for that
window; the YouTube URL in the header opens the video at the right second.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "reference"


def mmss(s):
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"


def excerpt(video_id, start, dur, out_dir=OUT, quiet=False):
    """Write the editable excerpt; returns its path. out_dir lets the
    validation listening set (analysis/validate_transcription.py) keep its
    files next to the audio clips. Episode clips (<meeting>_e<n>) have no
    YouTube URL of their own: the header names the clip instead."""
    doc = json.loads((ROOT / "output" / "whisper" / f"{video_id}.json").read_text(encoding="utf-8"))
    end = start + dur
    segs = [s for s in doc["segments"] if s["end"] > start and s["start"] < end]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{video_id}.{int(start)}-{int(end)}.txt"
    link = (f"# clip {video_id} (times are clip-relative)" if "_e" in video_id
            else f"# https://www.youtube.com/watch?v={video_id}&t={int(start)}s")
    lines = [
        f"# {doc['speaker']}  {video_id}  window {mmss(start)}-{mmss(end)}  model {doc['model']}",
        link,
        "# Correct the TEXT to what is actually said (keep timestamps). Fix the speaker",
        "# label only if it is clearly a different person. Delete a line if nobody speaks.",
        "",
    ]
    for s in segs:
        lines.append(f"[{mmss(s['start'])}] {s.get('speaker') or 'UNKNOWN'}: {s['text'].strip()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    words = sum(len(s["text"].split()) for s in segs)
    if not quiet:
        print(f"{path.relative_to(ROOT)}  {len(segs)} segments, {words} words")
    return path


def main(argv):
    out = OUT
    if argv and argv[0] == "--out":                 # --out DIR VIDEO_ID:START:DUR ...
        out, argv = Path(argv[1]), argv[2:]
    for spec in argv:
        vid, st, du = spec.split(":")
        excerpt(vid, float(st), float(du), out_dir=out)


if __name__ == "__main__":
    main(sys.argv[1:])
