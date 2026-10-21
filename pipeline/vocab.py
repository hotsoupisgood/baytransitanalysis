#!/usr/bin/env python3
"""
pipeline/vocab.py -- the per-body WhisperX initial_prompt vocabulary (T2,
2026-09-22; output/validation/REPORT.md 6.1: proper nouns -- agency names,
titles, station and place names, bill numbers -- are where WhisperX and the
caption track disagreed most).

Where a body's vocabulary comes from, first hit wins:
  1. sources.vocab in the state DB (set with `vocab.py apply`, or by hand)
  2. plan/vocab.tsv (body<TAB>text, hand-kept)
  3. nothing (the manifest carries initial_prompt = None)
Every value is clipped to MAX_CHARS at a word boundary: Whisper only keeps the
last ~224 tokens of a prompt, and a long name list raises hallucination of
those names (docs/MODELS.md 6), so short and specific beats long.

`suggest <body>` derives candidates from what the body actually says: the
capitalised tokens that recur across its CASED caption tracks (Granicus
live-captioner tracks; YouTube auto captions are lower-case and yield nothing)
plus the body's meeting titles, minus a stoplist. It prints, never writes --
the human picks, because a frequent capitalised token is often a person's
name, which never goes in (DECISIONS #15).

    python3 pipeline/vocab.py show [body]        # what the manifest would carry
    python3 pipeline/vocab.py suggest oakland    # candidate tokens from cased captions
    python3 pipeline/vocab.py apply              # plan/vocab.tsv -> sources.vocab (bodies with a sources row)
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state  # noqa: E402

ROOT = state.ROOT
VOCAB_TSV = ROOT / "plan" / "vocab.tsv"
MAX_CHARS = 200
CAP_TOKEN = re.compile(r"\b[A-Z][a-zA-Z'\-]{2,}\b")
STOP = set("""The This That These Those There Here And But Or Not For With From Into Over Under About After Before Because
So If Then When Where What Which Who Whom Whose Why How All Any Each Every Some Many Much More Most Other Another Such
Yes No Okay Thank Thanks Please Good Morning Afternoon Evening Hello Welcome Item Items Motion Second Aye Ayes Nay Nays Roll
Call Vote Public Comment Comments Speaker Speakers Council Councilmember Councilmembers Mayor Vice Member Members Staff City
County Board Director Directors President Chair Madam Mister Miss Ms Mr Mrs Dr Sir Ma'am Next Last First Two Three Four Five
Minutes Minute Time Today Tonight Meeting Agenda Consent Calendar Report Reports Question Questions Motion Approve Approved
Move Moved Seconded Adopted Adopt Resolution Ordinance Reading Closed Session Open Adjourn Adjourned Recess Also Just Well Right
Now Let Like Would Could Should Will Can May Might Must Have Has Had Been Being Was Were Are Is Am Do Does Did Done Going Get
Got Going Want Need Know Think Thought Say Said See Seen Look Come Came Take Took Make Made Give Gave Really Very Okay OK
January February March April June July August September October November December Monday Tuesday Wednesday Thursday Friday
Saturday Sunday""".split())
# Titles and institutions that are useful regardless of frequency
KEEP_ALWAYS = {"BART", "Caltrain", "SMART", "VTA", "Muni", "SFMTA", "SamTrans", "MTC", "ABAG", "HCD", "RHNA", "ADU", "CEQA"}


def read_tsv(path=VOCAB_TSV):
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or "\t" not in line:
            continue
        body, text = line.split("\t", 1)
        out[body.strip()] = " ".join(text.split())
    return out


def clip(text, max_chars=MAX_CHARS):
    """<= max_chars, cut at a word boundary, no dangling comma."""
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    cut = cut[:cut.rfind(" ")] if " " in cut else cut
    return cut.rstrip(" ,;.") + "."


def for_body(con, body):
    """The initial_prompt for a body (clipped) or None."""
    if con is not None and con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sources'").fetchone():
        cols = {r[1] for r in con.execute("PRAGMA table_info(sources)")}
        if "vocab" in cols:
            r = con.execute("SELECT vocab FROM sources WHERE body = ? AND vocab IS NOT NULL AND TRIM(vocab) != '' "
                            "ORDER BY active DESC, updated_at DESC LIMIT 1", (body,)).fetchone()
            if r and r["vocab"]:
                return clip(r["vocab"])
    text = read_tsv().get(body)
    return clip(text) if text else None


def suggest(con, body, top=25):
    """Capitalised tokens recurring in the body's cased caption tracks and
    titles: [(token, count)], stoplist and sentence-initial-only tokens removed."""
    counts, files = Counter(), 0
    rows = con.execute("SELECT captions_path, captions, title FROM videos WHERE kind = 'meeting' AND speaker = ?", (body,)).fetchall()
    for r in rows:
        for tok in CAP_TOKEN.findall(r["title"] or ""):
            counts[tok] += 1
        p = Path(r["captions_path"]) if r["captions_path"] else None
        if not p or not p.exists() or r["captions"] == "auto":
            continue                                      # auto tracks are lower-case: nothing to learn
        files += 1
        for block in re.split(r"\n{2,}", p.read_text(encoding="utf-8", errors="replace")):
            lines = [l for l in block.split("\n") if l.strip() and "-->" not in l and not l.startswith("WEBVTT")]
            text = re.sub(r"<[^>]+>", "", " ".join(lines))
            toks = CAP_TOKEN.findall(text)
            # a token that only ever starts a cue is a sentence start, not a name
            for i, tok in enumerate(toks):
                if i == 0 and text.lstrip().startswith(tok):
                    continue
                counts[tok] += 1
    out = [(t, n) for t, n in counts.most_common() if t in KEEP_ALWAYS or (n >= 3 and t not in STOP)]
    return out[:top], files, len(rows)


def apply(con, tsv=VOCAB_TSV):
    """plan/vocab.tsv -> sources.vocab for every body that has a sources row. Returns n."""
    vals = read_tsv(tsv)
    n = 0
    for body, text in vals.items():
        cur = con.execute("UPDATE sources SET vocab = ?, updated_at = ? WHERE body = ?", (clip(text), state.now_iso(), body))
        n += cur.rowcount
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["show", "suggest", "apply"])
    ap.add_argument("body", nargs="?")
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)
    con = state.connect(a.state)
    if a.cmd == "show":
        bodies = [a.body] if a.body else sorted(read_tsv())
        for b in bodies:
            v = for_body(con, b)
            print(f"{b:14} {len(v or ''):3d} chars  {v}")
        return 0
    if a.cmd == "suggest":
        if not a.body:
            ap.error("suggest needs a body")
        toks, files, rows = suggest(con, a.body)
        print(f"{a.body}: {rows} meeting rows, {files} cased caption tracks; candidates (token count) -- pick, never paste names:")
        print("  " + ", ".join(f"{t} ({n})" for t, n in toks))
        return 0
    n = apply(con)
    print(f"sources.vocab set for {n} source row(s) from {VOCAB_TSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
