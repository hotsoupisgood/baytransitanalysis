#!/usr/bin/env python3
"""
meanness_analysis.py
=====================
For each Whisper transcript, chunk the text into ~350-word windows, send each
chunk to Claude, and get a 0-10 "meanness" score (personal attacks, insults,
mockery, contempt -- not just strong disagreement on ideas).

EXPLORATORY. The study-grade version belongs in the DuckDB meanness_chunks table
(plan/SCHEMA.md 3) and a hierarchical model (chunks nested in videos nested in
speakers). Nothing printed here is an inferential result.

Fixed 2026-09-17 (docs/REVIEW-2026-09-17.md):
  * An unparseable reply is None, never 0. A 0 reads as "maximally polite".
  * Replies like "7/10" or "Score: 3 out of 10" parse as 7 and 3. The old digit
    scrape read both as 10.
  * k samples per chunk (default 3), median kept, all k and their SD recorded.
    Temperature is NOT set: Claude Sonnet 5 / Opus 5 reject any non-default
    temperature with HTTP 400, so noise is measured by resampling instead.
  * thinking is disabled explicitly. Claude 5 models think adaptively by default,
    and thinking tokens would consume the 5-token reply budget.
  * summary.json is MERGED on write. It used to be rewritten from the current
    run only, which is how it came to hold 1 of 10 videos.
  * Model comes from SCORING_MODEL (default claude-sonnet-4-6, the model that
    produced the existing output). Results are comparable only within one
    model AND one prompt_version; both are recorded per video.
  * Reads both transcript formats; --host-only (default) drops non-host speech
    when the transcript is diarized.

Usage:
    python3 legacy/meanness_analysis.py
    python3 legacy/meanness_analysis.py --video PDTon6oss9M
    python3 legacy/meanness_analysis.py --chunk-words 350 --workers 8 --samples 3

Output:
    output/meanness/<video_id>.json   per-chunk scores for that video
    output/meanness/summary.json      per-video summary, merged across runs
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:                     # the SDK is only needed to call the API; --rebuild-summary is offline
    import anthropic
except ImportError:      # pragma: no cover
    anthropic = None


# Lives in legacy/ since 2026-10-16; the repo root is one level up.
_REPO = Path(__file__).resolve().parent.parent


def _load_env(path: Path = _REPO / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = re.sub(r"\s+#.*$", "", v.strip())          # inline comments
        os.environ.setdefault(k.strip(), v.strip('"').strip("'"))


_load_env()

ROOT         = _REPO
WHISPER_DIR  = ROOT / "output" / "whisper"
MEANNESS_DIR = ROOT / "output" / "meanness"

MODEL = os.environ.get("SCORING_MODEL") or "claude-sonnet-4-6"


def _load_speakers() -> dict[str, str]:
    p = ROOT / "output" / "speakers.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------- transcripts --------------------------------------------------------

def load_doc(video_id: str, model_name: str = "medium"):
    """Return (segments, doc_meta). Legacy files are <id>.<model>.json holding a
    bare list; pipeline files are <id>.json holding {"segments": [...], ...}."""
    for path in (WHISPER_DIR / f"{video_id}.{model_name}.json", WHISPER_DIR / f"{video_id}.json"):
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(doc, list):
                return doc, {}
            return doc["segments"], {k: v for k, v in doc.items() if k != "segments"}
    raise FileNotFoundError(WHISPER_DIR / f"{video_id}.{model_name}.json")


def host_segments(segs: list[dict], meta: dict) -> tuple[list[dict], bool]:
    """Keep only the host's speech when the transcript is diarized. The host is
    every label in meta['host_speakers'] (embedding enrolment, PLAN.md 4.6;
    Hasan's impressions split him into several labels), else meta['host_speaker'],
    else the labelled speaker with the most speech time. Undiarized transcripts
    pass through."""
    talk = {}
    for s in segs:
        if s.get("speaker") is not None:
            talk[s["speaker"]] = talk.get(s["speaker"], 0.0) + s["end"] - s["start"]
    if not talk:
        return segs, False
    hosts = set(meta.get("host_speakers") or [meta.get("host_speaker") or max(talk, key=talk.get)])
    return [s for s in segs if s.get("speaker") in hosts], True


def chunk_segments(segs: list[dict], chunk_words: int = 350) -> list[dict]:
    """Group consecutive segments into ~chunk_words-word windows."""
    chunks = []
    cur_segs: list[dict] = []
    cur_words = 0
    for seg in segs:
        cur_segs.append(seg)
        cur_words += len(seg["text"].split())
        if cur_words >= chunk_words:
            chunks.append(_flush(cur_segs))
            cur_segs, cur_words = [], 0
    if cur_segs:
        chunks.append(_flush(cur_segs))
    return chunks


def _flush(segs: list[dict]) -> dict:
    return {
        "start": segs[0]["start"],
        "end": segs[-1]["end"],
        "text": " ".join(s["text"].strip() for s in segs),
    }


# ---------- LLM meanness scoring ----------------------------------------------

PROMPT_TEMPLATE = """\
Rate how mean the speaker is being in this transcript excerpt, on a scale of 0-10.

Focus on personal attacks, insults, name-calling, mockery, contempt, or \
dehumanizing language directed at specific people or groups. Strong criticism \
of ideas, policies, or passionate disagreement is NOT by itself mean — only \
score it up if the speaker is being derisive or hostile toward people rather \
than just arguing against a position.

0  = calm, respectful, no personal attacks
3  = sharp but substantive criticism, maybe one dismissive jab
5  = sarcastic, condescending, mocking a person or group
8  = harsh insults, name-calling, open contempt
10 = vicious, dehumanizing, sustained personal attacks

Transcript:
{text}

Reply with only a single integer from 0 to 10."""

# Changes whenever the prompt text changes; scores are comparable only within one.
PROMPT_VERSION = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:10]

_SCALE_NOISE = re.compile(r"(/\s*10\b|\bout of\s+10\b|\b0\s*(-|–|to)\s*10\b)", re.I)
_SCORE = re.compile(r"(?<![\d.])(10|[0-9])(?![\d.])")


def parse_score(raw: str):
    """First integer 0-10 in the reply, after removing scale mentions ("/10",
    "out of 10", "0-10"). None if there is none -- never a default number."""
    m = _SCORE.search(_SCALE_NOISE.sub(" ", raw or ""))
    return int(m.group(1)) if m else None


def _reply_text(msg) -> str:
    return next((b.text for b in msg.content if getattr(b, "type", "text") == "text"), "")


def score_once(client, text: str):
    msg = client.messages.create(
        model=MODEL,
        max_tokens=5,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}],
    )
    return parse_score(_reply_text(msg))


def score_chunk(client, text: str, samples: int = 3) -> dict:
    """k independent samples; the median is the score. Failed or unparseable
    samples are counted, not replaced."""
    got, failed = [], 0
    for _ in range(samples):
        try:
            s = score_once(client, text)
        except Exception as e:                    # rate limits, 5xx, 400s
            print(f"      API error: {e}")
            s = None
        if s is None:
            failed += 1
        else:
            got.append(s)
    return {
        "score": statistics.median(got) if got else None,
        "scores": got,
        "score_sd": round(statistics.stdev(got), 3) if len(got) > 1 else None,
        "n_samples": samples,
        "n_failed": failed,
    }


# ---------- summary ------------------------------------------------------------

def rebuild_summary() -> dict:
    """Rebuild summary.json from the per-video files. No API calls.

    WHY IT EXISTS: the pre-2026-09-17 script rewrote summary.json from the
    videos of the current run only, so output/meanness/summary.json held 1 of
    the 10 videos that had actually been scored. The per-video files were
    always intact, so the summary is derivable."""
    summary = {}
    for f in sorted(MEANNESS_DIR.glob("*.json")):
        if f.name == "summary.json":
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        chunks = d.get("chunks", [])
        scored = [c for c in chunks if c.get("score") is not None]
        scores = [c["score"] for c in scored]
        meanest = max(scored, key=lambda c: c["score"]) if scored else None
        summary[d["video_id"]] = {
            "speaker": d.get("speaker", "unknown"),
            "mean": sum(scores) / len(scores) if scores else None,
            "n_chunks": len(scored), "n_unscored": len(chunks) - len(scored),
            # Legacy runs recorded neither; and their parse failures were saved
            # as 0, so a legacy mean is a floor, not a measurement.
            "model": d.get("model", "legacy (unrecorded; parse failures scored 0)"),
            "prompt_version": d.get("prompt_version", "legacy"),
            "host_only": d.get("host_only", False),
            "meanest_score": meanest["score"] if meanest else None,
            "meanest_excerpt": meanest["text"] if meanest else None,
            "meanest_start": meanest["start"] if meanest else None,
        }
    (MEANNESS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------- main ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-words", type=int, default=350,
                    help="Approx words per scored chunk (default 350)")
    ap.add_argument("--model", default="medium",
                    help="Whisper model suffix of legacy transcripts (<id>.<model>.json)")
    ap.add_argument("--video", default=None,
                    help="Process only this video ID (default: all cached)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent chunks (default 8)")
    ap.add_argument("--samples", type=int, default=3,
                    help="LLM samples per chunk; the median is kept (default 3)")
    ap.add_argument("--all-speakers", action="store_true",
                    help="Score guest speech too (default: host only, when diarized)")
    ap.add_argument("--rebuild-summary", action="store_true",
                    help="rebuild summary.json from the per-video files and exit (no API calls)")
    args = ap.parse_args()

    MEANNESS_DIR.mkdir(parents=True, exist_ok=True)
    if args.rebuild_summary:
        s = rebuild_summary()
        print(f"rebuilt {MEANNESS_DIR / 'summary.json'} from {len(s)} per-video files")
        for vid, v in sorted(s.items(), key=lambda kv: -(kv[1]["mean"] or -1)):
            print(f"  {v['speaker']:<12} {vid:<14} mean={v['mean']:.2f} n={v['n_chunks']}")
        return

    if anthropic is None:
        sys.exit("the anthropic SDK is not installed: .venv/bin/pip install -r requirements.txt")
    client = anthropic.Anthropic()
    speakers = _load_speakers()

    if args.video:
        video_ids = [args.video]
    else:
        video_ids = sorted({p.name.split(".")[0] for p in WHISPER_DIR.glob(f"*.{args.model}.json")}
                           | {p.stem for p in WHISPER_DIR.glob("*.json") if "." not in p.stem})

    if not video_ids:
        print(f"No transcripts found in {WHISPER_DIR}")
        sys.exit(1)

    summary_path = MEANNESS_DIR / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    print(f"model={MODEL} prompt_version={PROMPT_VERSION} samples={args.samples}")

    for vid in video_ids:
        try:
            segs, meta = load_doc(vid, args.model)
        except FileNotFoundError as e:
            print(f"  Skipping — {e}")
            continue
        sp = meta.get("speaker") or speakers.get(vid, "unknown")
        print(f"\n{'='*60}")
        print(f"Video: {vid} ({sp})")
        host_only = False
        if not args.all_speakers:
            segs, host_only = host_segments(segs, meta)

        chunks = chunk_segments(segs, args.chunk_words)
        print(f"  {len(chunks)} chunk(s){'  [host speech only]' if host_only else ''}")

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(score_chunk, client, c["text"], args.samples): i
                       for i, c in enumerate(chunks)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    chunks[i].update(fut.result())
                except Exception as e:
                    print(f"    chunk {i} failed: {e}")
                    chunks[i].update({"score": None, "scores": [], "n_samples": args.samples,
                                      "n_failed": args.samples, "score_sd": None})

        scored = [c for c in chunks if c["score"] is not None]
        scores = [c["score"] for c in scored]
        mean = sum(scores) / len(scores) if scores else None
        meanest = max(scored, key=lambda c: c["score"]) if scored else None

        fmt = lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}:{int(s)%60:02d}"
        print(f"  => mean score: {mean:.2f} / 10" if mean is not None else "  => no parseable scores")
        print(f"  unscored chunks: {len(chunks) - len(scored)}")
        if meanest:
            print(f"  meanest moment [{fmt(meanest['start'])}] "
                  f"score={meanest['score']}: {meanest['text'][:100]}…")

        (MEANNESS_DIR / f"{vid}.json").write_text(
            json.dumps({"video_id": vid, "speaker": sp, "chunk_words": args.chunk_words,
                        "model": MODEL, "prompt_version": PROMPT_VERSION,
                        "samples": args.samples, "host_only": host_only,
                        "chunks": chunks, "mean": mean}, indent=2),
            encoding="utf-8")

        summary[vid] = {
            "speaker": sp, "mean": mean, "n_chunks": len(scored),
            "n_unscored": len(chunks) - len(scored),
            "model": MODEL, "prompt_version": PROMPT_VERSION, "host_only": host_only,
            "meanest_score": meanest["score"] if meanest else None,
            "meanest_excerpt": meanest["text"] if meanest else None,
            "meanest_start": meanest["start"] if meanest else None,
        }
        # Merge-on-write after every video: an interrupted run keeps what it scored.
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print("PER VIDEO (all runs; compare only rows with the same model + prompt_version)")
    print(f"{'='*60}")
    rows = sorted(summary.items(), key=lambda kv: -(kv[1].get("mean") or -1))
    for vid, s in rows:
        m = s.get("mean")
        print(f"  {s.get('speaker', '?'):<15} {vid:<14} mean={'%.2f' % m if m is not None else '  —  '}"
              f"  n={s.get('n_chunks')}  {s.get('model', 'legacy')}/{s.get('prompt_version', '?')}")
    print("\nDescriptive only. Chunks are nested in videos; a speaker comparison needs a "
          "hierarchical model with a video random intercept (CLAUDE.md).")


if __name__ == "__main__":
    main()
