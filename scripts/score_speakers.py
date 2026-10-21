#!/usr/bin/env python3
"""
scripts/score_speakers.py -- word-level speaker agreement between a human-
corrected excerpt (scripts/make_reference_excerpts.py format) and a transcript
JSON, with the label mismatch handled: diarization IDs (SPEAKER_03 ...) are
arbitrary per run, so hypothesis labels are mapped onto reference labels by
majority overlap before scoring.

    .venv/bin/python scripts/score_speakers.py REF.txt HYP.json [HYP2.json ...]

Prints, per hypothesis: words aligned, speaker agreement after mapping, and
the confusion by reference speaker. Timestamps in REF are ignored.
"""
import difflib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from qa_transcripts import norm_words  # noqa: E402

LINE = re.compile(r"^\[(\d+):(\d+)\]\s*(\S+):\s*(.*)$")


def ref_words(path):
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        m = LINE.match(raw)
        spk, text = (m.group(3), m.group(4)) if m else ("?", raw)
        out += [(w, spk) for w in norm_words(text)]
    return out


def hyp_words(path, start, end):
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for s in doc["segments"]:
        if s["end"] > start and s["start"] < end:
            out += [(w, s.get("speaker") or "?") for w in norm_words(s["text"])]
    return out, doc.get("pipeline_version")


def score(ref, hyp):
    sm = difflib.SequenceMatcher(a=[w for w, _ in ref], b=[w for w, _ in hyp], autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pairs += [(ref[i + i1][1], hyp[j1 + i][1]) for i in range(i2 - i1)]
    co = Counter(pairs)
    # greedy one-to-one mapping hyp label -> ref label by overlap
    mapping, used = {}, set()
    for (r, h), n in co.most_common():
        if h not in mapping and r not in used:
            mapping[h] = r
            used.add(r)
    agree = sum(n for (r, h), n in co.items() if mapping.get(h) == r)
    by_ref = defaultdict(lambda: [0, 0])
    for (r, h), n in co.items():
        by_ref[r][1] += n
        if mapping.get(h) == r:
            by_ref[r][0] += n
    return len(pairs), agree, mapping, by_ref


def main(argv):
    ref_path, hyps = Path(argv[0]), [Path(p) for p in argv[1:]]
    span = ref_path.name.split(".")[1]
    start, end = map(float, span.split("-"))
    ref = ref_words(ref_path)
    print(f"reference {ref_path.name}: {len(ref)} words, speakers {dict(Counter(s for _, s in ref))}")
    for hp in hyps:
        hyp, ver = hyp_words(hp, start, end)
        n, agree, mapping, by_ref = score(ref, hyp)
        print(f"  {hp} (pipeline {ver}): {n} aligned words, speaker agreement {agree / n:.1%}")
        for r, (a, t) in sorted(by_ref.items(), key=lambda kv: -kv[1][1]):
            print(f"      ref {r}: {a}/{t} words right ({a / t:.0%})")


if __name__ == "__main__":
    main(sys.argv[1:])
