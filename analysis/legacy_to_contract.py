#!/usr/bin/env python3
"""
legacy_to_contract.py -- wrap the exploratory transcripts in the PLAN.md 1 contract.

WHY: hasan_compare.py (April-May 2026) wrote bare segment lists named
<id>.<model>.json. load_db.py expects the contract object ({"video_id",
"segments", ...}). This converts them WITHOUT touching the originals, so the 20
real transcripts can exercise the loader, QA gate and word_freq.py on real speech
before the GPU pipeline exists.

These are NOT the corpus: no diarization, no upload month, CPU faster-whisper
medium (int8), picked as "recent videos in a length window". Smoke tests and QA
calibration only.

    python3 analysis/legacy_to_contract.py            # -> output/contract_legacy/
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="output/whisper")
    ap.add_argument("--model", default="medium", help="which <id>.<model>.json files to wrap")
    ap.add_argument("--speakers", default="output/speakers.json")
    ap.add_argument("--out", default="output/contract_legacy")
    a = ap.parse_args(argv)

    src, out = ROOT / a.src, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    spk_path = ROOT / a.speakers
    speakers = json.loads(spk_path.read_text()) if spk_path.exists() else {}

    n, missing = 0, []
    for f in sorted(src.glob(f"*.{a.model}.json")):
        segs = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(segs, list):
            continue                      # already a contract object
        vid = f.name.split(".")[0]
        if vid not in speakers:
            missing.append(vid)
        end = round(segs[-1]["end"], 2) if segs else 0.0
        doc = {
            "video_id": vid,
            "channel": speakers.get(vid),
            "speaker": speakers.get(vid),
            "month": None,                # hasan_compare.py never recorded upload dates
            "duration_s": end,
            "transcribed_s": end,
            "model": f"faster-whisper-{a.model} (cpu int8, vad_filter; hasan_compare.py)",
            "diarized": False,
            "source": "legacy",
            "segments": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs],
        }
        (out / f"{vid}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
        n += 1
    print(f"wrapped {n} legacy transcripts -> {out}")
    if missing:
        print(f"WARNING: no speaker in {a.speakers} for: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
