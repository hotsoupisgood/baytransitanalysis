#!/usr/bin/env python3
"""
a0_bridge.py -- run the existing trends / spread / anim figures on the MEETING
corpus, with BODY as the series and ROLE as the facet (plan/ANALYSIS-transit.md
2, last bullet).

WHY A BRIDGE AND NOT A REWRITE: freq_trends.py, freq_spread.py and freq_anim.py
all read one shape -- `segments JOIN videos USING(video_id)`, with a cell of
(videos.speaker x videos.month) and a cluster bootstrap that resamples whole
video_ids. The meeting corpus has exactly that shape under different names:

    videos.video_id  <-  episodes.episode_id   (the bootstrap cluster)
    videos.speaker   <-  episodes.body         (the series)
    videos.month     <-  episodes.month        (the x axis)
    segments.is_host <-  1 where the segment's role is the facet role

so this writes a small throwaway SQLite per role in the side-corpus shape and
calls the three scripts unchanged. `is_host` carries the ROLE, which is what
makes role a facet: freq_trends' default host_only=True then keeps exactly the
speech of that role, and one run produces one facet.

Nothing is re-tokenised or re-counted here: the words that reach the charts are
word_freq.tokens() of the same episode_segments rows every other A0 number uses.

    .venv/bin/python analysis/a0_bridge.py --db ~/.local/share/ytpt/corpus_a0_pilot.sqlite

Outputs (output/analysis/a0/trends/<role>/): trends.html, trends.csv, months.csv,
trends.png, top_words_spread.png, top_words.gif -- one set per role.
"""
import argparse, os, shutil, sqlite3, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import facets  # noqa: E402

ROLES = ("dais", "public")
MAX_SERIES = 4   # len(freq_trends.PALETTE["light"]["series"]), the validated categorical slots

DDL = """
CREATE TABLE videos (
  video_id TEXT PRIMARY KEY, speaker TEXT NOT NULL, month TEXT,
  title TEXT, duration_s REAL, host_confidence REAL
);
CREATE TABLE segments (
  video_id TEXT NOT NULL, seg_idx INTEGER NOT NULL,
  start_s REAL, end_s REAL, speaker TEXT, is_host INTEGER, text TEXT NOT NULL,
  PRIMARY KEY (video_id, seg_idx)
);
CREATE INDEX idx_seg_vid ON segments(video_id);
"""


def build(src, dst, role):
    """One episode = one 'video' of speaker=body; is_host marks the facet role."""
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    eps = [dict(r) for r in con.execute(
        "SELECT episode_id, body, month, family, duration_s FROM episodes")]
    segs = [dict(r) for r in con.execute(
        "SELECT episode_id, seg_idx, start_s, end_s, speaker, role, text FROM episode_segments "
        "ORDER BY episode_id, seg_idx")]
    con.close()

    out = sqlite3.connect(dst)
    out.executescript(DDL)
    out.executemany("INSERT INTO videos VALUES (?,?,?,?,?,?)",
                    [(e["episode_id"], e["body"], e["month"], e["family"], e["duration_s"], 1.0) for e in eps])
    out.executemany("INSERT INTO segments VALUES (?,?,?,?,?,?,?)",
                    [(s["episode_id"], s["seg_idx"], s["start_s"], s["end_s"], s["speaker"],
                      1 if s["role"] == role else 0, s["text"]) for s in segs])
    out.commit()
    n = out.execute("SELECT COUNT(*) FROM segments WHERE is_host = 1").fetchone()[0]
    eps_with = out.execute("SELECT COUNT(DISTINCT video_id) FROM segments WHERE is_host = 1").fetchone()[0]
    bodies = [r[0] for r in out.execute(
        "SELECT DISTINCT v.speaker FROM videos v JOIN segments s USING(video_id) WHERE s.is_host = 1 ORDER BY 1")]
    out.close()
    return n, eps_with, bodies


def run(cmd):
    print("  $ " + " ".join(Path(c).name if c.endswith(".py") else c for c in cmd[1:]))
    r = subprocess.run(cmd, capture_output=True, text=True)
    for line in (r.stdout or "").splitlines():
        if line.strip():
            print("    " + line)
    if r.returncode:
        print("    FAILED:\n" + "\n".join("    " + l for l in (r.stderr or "").splitlines()[-12:]))
    return r.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/ytpt/corpus_a0_pilot.sqlite"))
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "analysis" / "a0" / "trends"))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--roles", nargs="*", default=list(ROLES))
    ap.add_argument("--keep-db", action="store_true", help="leave the bridge SQLite files in --out-dir")
    ap.add_argument("--pool", action="store_true",
                    help="draw over every instrument at once; refused by default (analysis/facets.py)")
    a = ap.parse_args(argv)
    con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    scope = facets.require_single(con, pool=a.pool, script="analysis/a0_bridge.py")
    con.close()
    py = sys.executable
    out_root = Path(a.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="a0bridge-"))
    rc = 0
    for role in a.roles:
        bridge = tmpdir / f"corpus_{role}.sqlite"
        n, eps, bodies = build(a.db, bridge, role)
        out = out_root / role; out.mkdir(parents=True, exist_ok=True)
        print(f"\n== {role}: {n:,} segments in {eps} episodes, bodies {', '.join(bodies)} -> {out}")
        if n == 0:
            print("  no speech with this role; skipped"); continue
        # freq_trends / freq_spread / freq_anim colour one series per body from a
        # 4-colour categorical palette validated all-pairs (freq_trends.PALETTE).
        # Past four bodies they index off the end of it. Cycling would put two
        # bodies on one colour with nothing else to tell them apart -- these are
        # line and bar charts with no second channel -- so the cell is skipped
        # with a written reason instead of drawn wrong. The wide corpus has 10
        # bodies; A0-RESULTS 2.5 already rates these the weakest A0 outputs.
        if len(bodies) > MAX_SERIES:
            (out / "SKIPPED.md").write_text(
                f"# not drawn\n\n{len(bodies)} bodies ({', '.join(bodies)}) in this facet cell; "
                f"freq_trends/freq_spread/freq_anim have {MAX_SERIES} validated series colours and no second "
                f"visual channel, so two bodies would share a colour. The per-body numbers are in the "
                f"a0_plots figures and their CSV twins in the parent folder.\n")
            print(f"  {len(bodies)} bodies > {MAX_SERIES} series colours; skipped, see {out / 'SKIPPED.md'}")
            continue
        rc |= run([py, str(ROOT / "analysis" / "freq_trends.py"), "--db", str(bridge), "--out-dir", str(out),
                   "--top", str(a.top), "--title", f"Content-word frequency by body -- {role} speech [{scope}]"])
        rc |= run([py, str(ROOT / "analysis" / "freq_spread.py"), "--db", str(bridge), "--out-dir", str(out),
                   "--top", str(a.top), "--title", f"Top content words by body -- {role} speech [{scope}]"])
        rc |= run([py, str(ROOT / "analysis" / "freq_anim.py"), "--db", str(bridge), "--out-dir", str(out),
                   "--top", min(a.top, 8) and str(min(a.top, 8)),
                   "--title", f"Top content words by month -- {role} speech [{scope}]"])
        if a.keep_db:
            shutil.copy(bridge, out / f"bridge_{role}.sqlite")
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"\nwrote {out_root}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
