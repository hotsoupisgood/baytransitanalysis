#!/usr/bin/env python3
"""
stance_analysis.py
==================
For each Whisper transcript, find every mention of a target word, extract a
context window of surrounding segments, send to Claude, and get a stance label.

Replicates the direct-prompting methodology from:
  "A Systematic Comparison of Prompting and Multi-Agent Methods for LLM-based
   Stance Detection" (arXiv:2604.26319)

EXPLORATORY. The study-grade store is the stance_mentions table (plan/SCHEMA.md 3).

Fixed 2026-09-17 (docs/REVIEW-2026-09-17.md):
  * Mentions are WHOLE-WORD matches. The old re.escape(target) substring search
    counted "police" and "justice" as mentions of "ice".
  * A reply that is not exactly one valid label is UNPARSEABLE -- it used to be
    silently counted as NEUTRAL, inflating the neutral share.
  * k samples per mention (default 3), majority label kept; a three-way split
    is NO_MAJORITY. Temperature is not set (Claude 5 models reject non-default
    temperature); thinking is disabled so it cannot eat the reply budget.
  * summary.json is merged across runs instead of rewritten; model and
    prompt_version are recorded because labels are comparable only within both.

Usage:
    python3 legacy/stance_analysis.py --targets israel vaccines --window 3
    python3 legacy/stance_analysis.py --targets israel --video PDTon6oss9M

Output:
    output/stance/<video_id>.<target>.json   per-mention results
    output/stance/summary.json               per-video counts, merged across runs
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
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
        v = re.sub(r"\s+#.*$", "", v.strip())
        os.environ.setdefault(k.strip(), v.strip('"').strip("'"))


_load_env()

ROOT        = _REPO
WHISPER_DIR = ROOT / "output" / "whisper"
STANCE_DIR  = ROOT / "output" / "stance"

MODEL = os.environ.get("SCORING_MODEL") or "claude-sonnet-4-6"
LABELS = ("FAVOR", "AGAINST", "NEUTRAL")


def _load_speakers() -> dict[str, str]:
    p = ROOT / "output" / "speakers.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------- context window extraction ----------------------------------------

def load_segments(video_id: str, model_name: str = "medium") -> list[dict]:
    """Legacy <id>.<model>.json (bare list) or pipeline <id>.json ({"segments"})."""
    for path in (WHISPER_DIR / f"{video_id}.{model_name}.json", WHISPER_DIR / f"{video_id}.json"):
        if path.exists():
            doc = json.loads(path.read_text(encoding="utf-8"))
            return doc if isinstance(doc, list) else doc["segments"]
    raise FileNotFoundError(WHISPER_DIR / f"{video_id}.{model_name}.json")


def target_pattern(target: str) -> re.Pattern:
    """Whole-word / whole-phrase, case-insensitive (same rule as
    analysis/word_freq.node_pattern): "ICE," and "Israel's" match; "police" and
    "Israeli" do not."""
    body = r"\s+".join(re.escape(w) for w in target.split())
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", re.IGNORECASE)


def find_mentions(segs: list[dict], target: str, window: int = 3) -> list[dict]:
    """
    Return one entry per mention of `target` in the transcript.
    Each entry has:
      - mention_idx: segment index of the hit
      - start/end: timestamps of the hit segment
      - context: concatenated text of [idx-window .. idx+window] segments
      - hit_text: just the segment that contains the mention
    """
    target_re = target_pattern(target)
    mentions = []
    for i, seg in enumerate(segs):
        if target_re.search(seg["text"]):
            lo = max(0, i - window)
            hi = min(len(segs) - 1, i + window)
            context_segs = segs[lo : hi + 1]
            context = " ".join(s["text"].strip() for s in context_segs)
            mentions.append({
                "mention_idx": i,
                "start": seg["start"],
                "end":   seg["end"],
                "hit_text": seg["text"],
                "context": context,
            })
    return mentions


# ---------- LLM stance classification ----------------------------------------

PROMPT_TEMPLATE = """\
Determine the stance of the speaker toward "{target}" based on the following transcript excerpt.

Transcript:
{context}

What is the speaker's stance toward "{target}"?
Choose exactly one of: FAVOR, AGAINST, NEUTRAL

- FAVOR: the speaker supports, defends, or speaks positively about {target}
- AGAINST: the speaker criticizes, opposes, or speaks negatively about {target}
- NEUTRAL: the speaker mentions {target} without clear positive or negative stance

Reply with only the single word label."""

PROMPT_VERSION = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:10]


def parse_label(raw: str) -> str:
    """Exactly one valid label in the reply -> that label; otherwise UNPARSEABLE."""
    found = {lab for lab in LABELS if re.search(rf"\b{lab}\b", (raw or "").upper())}
    return found.pop() if len(found) == 1 else "UNPARSEABLE"


def classify_once(client, context: str, target: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=10,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(target=target, context=context)}],
    )
    text = next((b.text for b in msg.content if getattr(b, "type", "text") == "text"), "")
    return parse_label(text)


def classify_stance(client, context: str, target: str, samples: int = 3) -> dict:
    votes = []
    for _ in range(samples):
        try:
            votes.append(classify_once(client, context, target))
        except Exception as e:                    # rate limits, 5xx, 400s
            print(f"      API error: {e}")
            votes.append("UNPARSEABLE")
    valid = Counter(v for v in votes if v in LABELS)
    if not valid:
        label = "UNPARSEABLE"
    else:
        (top, n), *rest = valid.most_common()
        label = top if not rest or n > rest[0][1] else "NO_MAJORITY"
    return {"stance": label, "votes": votes}


# ---------- aggregate --------------------------------------------------------

def aggregate(mentions: list[dict]) -> dict:
    counts = {k: 0 for k in (*LABELS, "NO_MAJORITY", "UNPARSEABLE")}
    for m in mentions:
        counts[m["stance"]] = counts.get(m["stance"], 0) + 1
    total = sum(counts.values())
    valid = {k: counts[k] for k in LABELS}
    dominant = max(valid, key=valid.__getitem__) if sum(valid.values()) else None
    return {"counts": counts, "total": total, "dominant": dominant}


# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True,
                    help="Target words/phrases, e.g. --targets israel vaccines")
    ap.add_argument("--window",  type=int, default=3,
                    help="Segments of context on each side of the mention (default 3 ≈ ±45s)")
    ap.add_argument("--model",   default="medium",
                    help="Whisper model suffix of legacy transcripts (<id>.<model>.json)")
    ap.add_argument("--video",   default=None,
                    help="Process only this video ID (default: all cached)")
    ap.add_argument("--samples", type=int, default=3,
                    help="LLM samples per mention; majority label kept (default 3)")
    args = ap.parse_args()

    STANCE_DIR.mkdir(parents=True, exist_ok=True)

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

    summary_path = STANCE_DIR / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    print(f"model={MODEL} prompt_version={PROMPT_VERSION} samples={args.samples}")

    for vid in video_ids:
        print(f"\n{'='*60}")
        print(f"Video: {vid}")
        try:
            segs = load_segments(vid, args.model)
        except FileNotFoundError as e:
            print(f"  Skipping — {e}")
            continue

        summary.setdefault(vid, {})

        for target in args.targets:
            mentions = find_mentions(segs, target, args.window)
            print(f"  Target '{target}': {len(mentions)} mention(s)")
            for m in mentions:
                m.update(classify_stance(client, m["context"], target, args.samples))
                fmt = lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}:{int(s)%60:02d}"
                print(f"    [{fmt(m['start'])}] {m['stance']:>11}  …{m['hit_text'][:80]}…")

            agg = aggregate(mentions)
            agg.update({"model": MODEL, "prompt_version": PROMPT_VERSION, "window": args.window})
            summary[vid][target] = agg
            print(f"  => {agg['dominant']}  {agg['counts']}")

            out = STANCE_DIR / f"{vid}.{target.replace(' ', '_')}.json"
            out.write_text(json.dumps({"video_id": vid, "target": target,
                                       "window": args.window, "model": MODEL,
                                       "prompt_version": PROMPT_VERSION,
                                       "samples": args.samples, "mentions": mentions,
                                       "aggregate": agg}, indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")   # merge-on-write

    by_speaker: dict[str, list[str]] = {}
    for vid in summary:
        by_speaker.setdefault(speakers.get(vid, "unknown"), []).append(vid)

    print(f"\n{'='*60}")
    print("STANCE SUMMARY (all runs; compare only rows with the same model + prompt_version)")
    print(f"{'='*60}")
    for target in args.targets:
        print(f"\nTarget: {target.upper()}")
        print(f"  {'Speaker':<14} {'Video':<14} {'Total':>5} {'FAV':>4} {'AGN':>4} {'NEU':>4} {'?':>3}  Dominant")
        print(f"  {'-'*70}")
        for sp, vids in sorted(by_speaker.items()):
            for vid in sorted(vids):
                a = summary.get(vid, {}).get(target)
                if not a:
                    continue
                c = a["counts"]
                unclear = c.get("UNPARSEABLE", 0) + c.get("NO_MAJORITY", 0)
                print(f"  {sp:<14} {vid:<14} {a['total']:>5} {c.get('FAVOR', 0):>4} "
                      f"{c.get('AGAINST', 0):>4} {c.get('NEUTRAL', 0):>4} {unclear:>3}  {a['dominant']}")


if __name__ == "__main__":
    main()
