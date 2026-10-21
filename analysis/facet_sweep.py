#!/usr/bin/env python3
"""
facet_sweep.py -- run the A0 / A1 figure pipeline once per facet level.

The corpus has two instrument facets (analysis/facets.py): `method`, the route
that located the episode, and `role_source`, the instrument that assigned dais /
public. Neither is a corpus variable, so no figure may average over them. This
driver is how "per facet" is actually produced:

  for each facet, for each level
      slice the corpus  -> a stamped corpus_<facet>_<level>.sqlite
      a0_terms.py       -> term counts, frame counts, KWIC for that level
      a0_plots.py       -> every A0 figure for that level
      a0_bridge.py      -> trends / spread / anim for that level
      a0_export.py      -> parquet + DuckDB for that level
      a1_sentiment.py   -> the A1 lexicon pass for that level

Outputs land in <out>/by_<facet>/<level>/, plus <out>/facet_manifest.csv with
one row per cell (episodes, words, clusters, bodies) so a thin cell is visible
before anyone reads a figure drawn on it.

The slices are kept, not deleted: a figure and the exact corpus it came from
sit in the same folder.

    .venv/bin/python analysis/facet_sweep.py --db ~/.local/share/ytpt/corpus_a0_pilot.sqlite
    .venv/bin/python analysis/facet_sweep.py --db <db> --steps a1 --out-dir output/analysis/a1
"""
import argparse, csv, os, sqlite3, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import facets  # noqa: E402

STEPS = ("terms", "plots", "bridge", "export", "a1")
MIN_WORDS = 500   # a0_plots.MIN_CELL_WORDS: below this a per-1,000-word rate is one utterance


def run(cmd, label):
    print(f"  $ {label}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-4:]
    for line in tail:
        print("    " + line)
    if r.returncode:
        print("    FAILED:\n" + "\n".join("    " + l for l in (r.stderr or "").splitlines()[-12:]))
    return r.returncode


def cell_stats(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    n_ep, n_words, n_bodies = con.execute(
        "SELECT COUNT(*), COALESCE(SUM(words), 0), COUNT(DISTINCT body) FROM episodes").fetchone()
    n_spk = con.execute("SELECT COUNT(*) FROM episode_speakers").fetchone()[0]
    roles = dict(con.execute("SELECT role, COALESCE(SUM(words), 0) FROM episode_speakers GROUP BY 1"))
    bodies = ",".join(r[0] for r in con.execute("SELECT DISTINCT body FROM episodes ORDER BY 1"))
    con.close()
    tot = sum(roles.values()) or 1
    return {"episodes": n_ep, "words": n_words, "clusters": n_spk, "n_bodies": n_bodies, "bodies": bodies,
            "dais_words": roles.get("dais", 0), "public_words": roles.get("public", 0),
            "unknown_words": roles.get("unknown", 0), "unknown_share": round(roles.get("unknown", 0) / tot, 5)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/ytpt/corpus_a0_pilot.sqlite"))
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "analysis" / "a0"))
    ap.add_argument("--frames", default=str(ROOT / "plan" / "frames.yaml"))
    ap.add_argument("--facets", nargs="*", default=list(facets.FACETS), help=f"subset of {', '.join(facets.FACETS)}")
    ap.add_argument("--steps", nargs="*", default=[s for s in STEPS if s != "a1"],
                    help=f"subset of {', '.join(STEPS)} (default: the A0 steps; 'a1' runs the sentiment pass)")
    ap.add_argument("--title", default="", help="scope name for figure titles, e.g. 'A0 wide'")
    a = ap.parse_args(argv)

    py = sys.executable
    out_root = Path(a.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    src = Path(a.db)
    if not src.exists():
        sys.exit(f"no corpus at {src} -- run analysis/load_episodes.py first")

    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    avail = facets.present(con)
    con.close()

    rc, manifest = 0, []
    for facet in a.facets:
        levels = avail.get(facet) or []
        if not levels:
            print(f"\n== {facet}: no levels in this corpus; skipped")
            continue
        print(f"\n== facet {facet}: {len(levels)} level(s) -- {', '.join(levels)}")
        for level in levels:
            out = out_root / f"by_{facet}" / level
            out.mkdir(parents=True, exist_ok=True)
            db = out / f"corpus_{facet}_{level}.sqlite"
            if db.exists():
                db.unlink()
            n_ep, n_words = facets.slice_db(src, db, facet, level)
            st = cell_stats(db)
            manifest.append({"facet": facet, "level": level, "label": facets.LABELS.get(level, level),
                             "out_dir": os.path.relpath(out.resolve(), ROOT), **st,
                             "thin": int(st["words"] < MIN_WORDS)})
            print(f"\n-- {facet}={level}: {n_ep} episodes, {n_words:,} words, "
                  f"{st['n_bodies']} bodies, unknown-role {st['unknown_share']:.2%} -> {out}")
            if st["words"] < MIN_WORDS:
                print(f"   THIN CELL ({st['words']} words < {MIN_WORDS}); figures drawn but not reportable")
            if "terms" in a.steps:
                rc |= run([py, str(ROOT / "analysis" / "a0_terms.py"), "--db", str(db),
                           "--out-dir", str(out), "--frames", a.frames], f"a0_terms  {facet}={level}")
            if "plots" in a.steps:
                rc |= run([py, str(ROOT / "analysis" / "a0_plots.py"), "--db", str(db),
                           "--out-dir", str(out), "--frames", a.frames], f"a0_plots  {facet}={level}")
            if "bridge" in a.steps:
                rc |= run([py, str(ROOT / "analysis" / "a0_bridge.py"), "--db", str(db),
                           "--out-dir", str(out / "trends")], f"a0_bridge {facet}={level}")
            if "a1" in a.steps:
                rc |= run([py, str(ROOT / "analysis" / "a1_sentiment.py"), "--db", str(db),
                           "--out-dir", str(out), "--frames", a.frames], f"a1_sent  {facet}={level}")
            if "export" in a.steps:
                rc |= run([py, str(ROOT / "analysis" / "a0_export.py"), "--db", str(db),
                           "--csv-dirs", str(out), "--out-dir", str(out)], f"a0_export {facet}={level}")

    mpath = out_root / "facet_manifest.csv"
    if manifest:
        with open(mpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)
        print(f"\n{len(manifest)} facet cells -> {mpath}")
        w1 = max(len(m["facet"]) for m in manifest); w2 = max(len(m["level"]) for m in manifest)
        for m in manifest:
            print(f"  {m['facet']:<{w1}} {m['level']:<{w2}}  {m['episodes']:>3} ep  {m['words']:>8,} w  "
                  f"{m['n_bodies']:>2} bodies  unknown {m['unknown_share']:>7.2%}"
                  + ("   THIN" if m["thin"] else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
