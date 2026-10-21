#!/usr/bin/env python3
"""
scripts/backlog_public_turns.py -- per-body dais/public turn counts on the
NIMBY-pilot frame (scripts/nimby_pilot.py: density|both episodes, turns =
maximal runs of one diarization cluster, >= 25 words, role dais or public),
rebuilt from corpus.sqlite's episode_segments (already loop-collapsed by
analysis/load_episodes.py) so the count can be taken before and after a
`make load-episodes` without re-reading transcript files.

    .venv/bin/python scripts/backlog_public_turns.py --out output/backlog/public_turns_before.csv
"""
import argparse, csv, os, re, sqlite3
from collections import Counter, defaultdict
from pathlib import Path

WORD_RE = re.compile(r"[a-z0-9']+")
MIN_WORDS = 25


def counts(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    eps = {r["episode_id"]: (r["body"], r["method"]) for r in con.execute(
        "SELECT episode_id, body, method FROM episodes WHERE family IN ('density', 'both')")}
    c = Counter()
    n_eps = Counter()
    for eid in eps:
        n_eps[eps[eid][0]] += 1
    cur_spk = cur_role = cur_ep = None
    buf = []

    def flush():
        if cur_ep is None:
            return
        words = len(WORD_RE.findall(" ".join(buf).lower()))
        if words >= MIN_WORDS and cur_role in ("dais", "public"):
            c[(eps[cur_ep][0], cur_role)] += 1

    for r in con.execute("SELECT episode_id, speaker, role, text FROM episode_segments ORDER BY episode_id, seg_idx"):
        if r["episode_id"] not in eps:
            continue
        spk = r["speaker"] or "UNLABELLED"
        if r["episode_id"] != cur_ep or spk != cur_spk:
            flush()
            cur_ep, cur_spk, cur_role, buf = r["episode_id"], spk, r["role"], []
        buf.append(r["text"])
    flush()
    con.close()
    return c, n_eps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/ytpt/corpus.sqlite"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    c, n_eps = counts(a.db)
    bodies = sorted(n_eps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["body", "density_episodes_loaded", "dais_turns", "public_turns"])
        for b in bodies:
            w.writerow([b, n_eps[b], c[(b, "dais")], c[(b, "public")]])
    for b in sorted(bodies, key=lambda b: c[(b, "public")]):
        print(f"{b:12s} eps={n_eps[b]:4d} dais={c[(b, 'dais')]:5d} public={c[(b, 'public')]:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
