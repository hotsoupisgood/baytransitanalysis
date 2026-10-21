#!/usr/bin/env python3
"""
scripts/score_reference.py -- score a machine transcript against a human-corrected
excerpt made with scripts/make_reference_excerpts.py.

    .venv/bin/python scripts/score_reference.py output/reference/<id>.<start>-<end>.txt ...

The corrected file is the REFERENCE. The hypothesis is rebuilt from
output/whisper/<id>.json for the same window, so the original excerpt need not
be kept. Timestamps in the corrected file are ignored (only the text and the
speaker labels matter); lines starting with # are ignored.

Prints WER (edits / reference words), the largest differing runs, and the
speaker-label changes the human made.
"""
import difflib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from qa_transcripts import edit_distance, norm_words  # noqa: E402

LINE = re.compile(r"^\[(\d+):(\d+)\]\s*(\S+):\s*(.*)$")


def parse_ref(path):
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        m = LINE.match(raw)
        if m:
            rows.append((int(m.group(1)) * 60 + int(m.group(2)), m.group(3), m.group(4)))
        else:                                   # a line the human retyped without the prefix
            rows.append((None, None, raw))
    return rows


def hyp_rows(video_id, start, end):
    doc = json.loads((ROOT / "output" / "whisper" / f"{video_id}.json").read_text(encoding="utf-8"))
    return [(s["start"], s.get("speaker") or "UNKNOWN", s["text"].strip())
            for s in doc["segments"] if s["end"] > start and s["start"] < end]


def runs(ref, hyp, min_words=4):
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal" and max(i2 - i1, j2 - j1) >= min_words:
            out.append((max(i2 - i1, j2 - j1), " ".join(ref[i1:i2]), " ".join(hyp[j1:j2])))
    return sorted(out, reverse=True)


def main(paths):
    summary = "--summary" in paths                  # one pooled line at the end (edits / reference words over all files)
    paths = [p for p in paths if p != "--summary"]
    tot_edits = tot_ref = 0
    for p in map(Path, paths):
        vid, span = p.name.split(".")[0], p.name.split(".")[1]
        start, end = map(float, span.split("-"))
        ref_rows, hyp = parse_ref(p), hyp_rows(vid, start, end)
        ref = [w for _, _, t in ref_rows for w in norm_words(t)]
        hypw = [w for _, _, t in hyp for w in norm_words(t)]
        edits = edit_distance(ref, hypw)
        tot_edits, tot_ref = tot_edits + edits, tot_ref + len(ref)
        wer = edits / max(len(ref), 1)
        print(f"== {p.name}: WER {wer:.1%}  ({len(ref)} ref words, {len(hypw)} hyp words)")
        for n, r, h in runs(ref, hypw)[:5]:
            print(f"   {n:>3} words   REF: {r[:110]}")
            print(f"               HYP: {h[:110]}")
        # speaker labels: compare line-by-line where the human kept the prefix
        changed = [(t, a, b) for (t, a, _), (_, b, _) in zip([r for r in ref_rows if r[0] is not None], hyp)
                   if a != b]
        if changed:
            print(f"   speaker labels changed on {len(changed)} of {len(hyp)} lines: "
                  + ", ".join(f"{t//60:02d}:{t%60:02d} {b}->{a}" for t, a, b in changed[:8]))
        else:
            print(f"   speaker labels: no changes on {len(hyp)} lines")
    if summary and tot_ref:
        print(f"\n== pooled over {len(paths)} excerpts: WER {tot_edits / tot_ref:.1%}  ({tot_ref} reference words)")


if __name__ == "__main__":
    main(sys.argv[1:])
