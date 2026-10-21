#!/usr/bin/env python3
"""
simulate.py — generate a synthetic corpus for testing the pipeline end to end.

WHY THIS EXISTS
---------------
Every downstream component (loader, FTS index, word frequency, collocates,
scoring) needs to be exercised before 300 real videos exist. More importantly,
an analysis you have never run against data with a KNOWN answer is an analysis
you cannot trust: if the collocate finder returns something plausible on real
data you have no way to tell whether it found a real pattern or a bug.

So this generator plants a known effect and the test asserts it is recovered:

  near the node word "gaza":
      hasanabi  over-uses: genocide, occupation, siege, apartheid, famine
      benshapiro over-uses: hamas, terror, hostages, tunnels, october
      tucker     over-uses: neocons, forever, funding, borders, donors

If `word_freq.py --mode kwic --node gaza` does not put those words at the top
for the right speaker, something is broken -- and we learn that now, for free,
rather than after renting a GPU.

It also plants a NULL: the word "actually" is emitted at the same rate by all
three speakers and near/far from the node. Anything that reports it as
distinctive is over-fitting, and the test checks it stays non-significant.

Output mirrors the real contract exactly (PLAN.md §1) so nothing downstream can
tell the difference:  output_sim/whisper/<video_id>.json
"""
import json, random, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- vocabularies
# Shared filler/function words. Deliberately the overwhelming majority of tokens,
# because that is true of real speech and it is what the stopword filter has to
# survive. If the analysis can't see past these, it can't see past real speech.
FILLER = ("um uh like you know i mean kinda sorta basically literally right so "
          "the a an and but or of to in on that this it is was be are were").split()

NEUTRAL = ("people think going said really thing time way lot part点 point talk "
           "watch clip video show today week country state media").replace("点", "").split()

# Speaker-specific background style (present everywhere, not just near the node)
STYLE = {
    "hasanabi":   "chat stream dude bro insane wild crazy literally".split(),
    "benshapiro": "look frankly premise fact leftist argument obviously".split(),
    "tucker":     "interesting extraordinary nobody question truth honestly".split(),
}

# THE PLANTED EFFECT: words over-used within the window around the node word.
NODE = "gaza"
NEAR_NODE = {
    "hasanabi":   "genocide occupation siege apartheid famine".split(),
    "benshapiro": "hamas terror hostages tunnels october".split(),
    "tucker":     "neocons forever funding borders donors".split(),
}
# Distractor topic so the node window isn't the only place content words appear.
OTHER_TOPIC = ("healthcare premiums deductible insurer hospital").split()

NULL_WORD = "actually"   # planted non-effect; must NOT come out as distinctive

SPEAKERS = list(STYLE)


def sentence(rng, speaker, near_node: bool) -> str:
    """Build one sentence. ~70% filler by design -- real speech is mostly glue."""
    words = []
    for _ in range(rng.randint(8, 20)):
        r = rng.random()
        if r < 0.62:
            words.append(rng.choice(FILLER))
        elif r < 0.78:
            words.append(rng.choice(NEUTRAL))
        elif r < 0.88:
            words.append(rng.choice(STYLE[speaker]))
        elif r < 0.92:
            words.append(NULL_WORD)                      # flat across everything
        elif near_node and r < 0.99:
            words.append(rng.choice(NEAR_NODE[speaker]))  # <-- the planted signal
        else:
            words.append(rng.choice(OTHER_TOPIC))
    return " ".join(words).capitalize() + "."


def make_video(rng, video_id, speaker, month, n_segments, guest_share=0.0):
    """One transcript. `guest_share` of segments are attributed to a second
    speaker, so host-filtering can be tested: a guest who talks about the node
    topic with the OPPOSITE vocabulary should be excluded by is_host."""
    segs, t = [], 0.0
    guest_style = rng.choice([s for s in SPEAKERS if s != speaker])
    for i in range(n_segments):
        is_guest = rng.random() < guest_share
        spk = "SPEAKER_01" if is_guest else "SPEAKER_00"
        voice = guest_style if is_guest else speaker
        # ~12% of sentences are "about" the node topic
        near = rng.random() < 0.12
        text = sentence(rng, voice, near)
        if near:
            # place the node word inside the sentence
            parts = text.split()
            parts.insert(rng.randint(1, max(1, len(parts) - 1)), NODE.capitalize())
            text = " ".join(parts)
        dur = rng.uniform(2.5, 6.0)
        segs.append({"start": round(t, 2), "end": round(t + dur, 2),
                     "speaker": spk, "text": text})
        t += dur + rng.uniform(0.0, 0.4)
    return {
        "video_id": video_id, "channel": speaker, "speaker": speaker,
        "month": month, "duration_s": round(t, 2), "model": "SIMULATED",
        "diarized": True, "host_speaker": "SPEAKER_00",
        "host_confidence": round(1.0 - guest_share, 2),
        "transcribed_s": round(t, 2), "segments": segs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output_sim")
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--months", type=int, default=8)
    ap.add_argument("--segments", type=int, default=160)
    ap.add_argument("--seed", type=int, default=20260914)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    out = ROOT / a.out / "whisper"
    out.mkdir(parents=True, exist_ok=True)
    # Overwrite rather than delete-then-write: the output folder may be a mount
    # where unlink is not permitted, and a fixed seed regenerates identical
    # filenames anyway. Stale files from a *different* seed are the caller's
    # problem -- use a fresh --out directory when changing the generator.
    for f in out.glob("*.json"):
        try:
            f.unlink()
        except OSError:
            pass

    months = [f"2026-{m:02d}" for m in range(1, a.months + 1)]
    n = 0
    for month in months:
        for sp in SPEAKERS:
            for k in range(a.per_cell):
                vid = f"SIM{sp[:3]}{month.replace('-','')}{k}"
                # tucker gets interview-heavy videos: exercises host filtering
                gs = 0.35 if sp == "tucker" else (0.08 if rng.random() < 0.3 else 0.0)
                doc = make_video(rng, vid, sp, month,
                                 rng.randint(int(a.segments * 0.6), a.segments), gs)
                (out / f"{vid}.json").write_text(json.dumps(doc, indent=1))
                n += 1
    print(f"wrote {n} simulated transcripts to {out}")
    print(f"planted node='{NODE}'; expected collocates per speaker:")
    for s, w in NEAR_NODE.items():
        print(f"  {s:<12} {' '.join(w)}")
    print(f"planted null (must NOT be distinctive): '{NULL_WORD}'")


if __name__ == "__main__":
    main()
