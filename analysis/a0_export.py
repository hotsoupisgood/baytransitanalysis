#!/usr/bin/env python3
"""
a0_export.py -- Phase A0 deliverable format (plan/ANALYSIS-transit.md 6:
"Outputs: output/analysis/ parquet + DuckDB").

Two kinds of table go in, and the distinction matters when reading them:

  BASE tables are the corpus itself, straight out of the pilot corpus.sqlite:
  meetings, episodes, episode_speakers, episode_segments. They carry the unit of
  analysis (a diarized cluster inside an episode, with a role and a body) and are
  what a later phase re-aggregates.

  DERIVED tables are the numbers behind each A0 figure, loaded from the CSV twin
  the figure already writes, so the parquet and the PNG cannot disagree: there is
  one computation, not two. Column types come from the CSV via pandas, with the
  rate/count columns coerced to numeric.

Everything lands twice: one parquet per table (portable, reads in R via arrow)
and one DuckDB file with the same tables (fast aggregation, reads in R via
duckdb). No transcript TEXT leaves episode_segments, and no speaker NAME is
written anywhere: clusters stay anonymous ids with a role, per the standing rule.

    .venv/bin/python analysis/a0_export.py --db ~/.local/share/ytpt/corpus_a0_pilot.sqlite
"""
import argparse, os, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE = ["meetings", "episodes", "episode_speakers", "episode_segments"]
# Present once load_episodes.py / analysis/facets.py have run: the per-route
# exposure numerators, and the stamp naming the facet slice this export is of.
# Exported when they exist so a reader in R can see the facet without the PNGs.
OPTIONAL = ["meeting_exposure", "a0_facet"]

# Derived CSVs -> table name. Written by a0_plots.py / a0_terms.py.
DERIVED = {
    "a0_rates_episode_min.csv": "rates_episode_min",
    "a0_rates_public_share.csv": "rates_public_share",
    "a0_rates_by_month.csv": "rates_by_month",
    "a0_top_terms_dais.csv": "top_terms_dais",
    "a0_top_terms_public.csv": "top_terms_public",
    "a0_top_terms_role_body_family.csv": "top_terms_role_body_family",
    "a0_cells_thin.csv": "cells_thin",
    "a0_keyness_public_vs_dais.csv": "keyness_public_vs_dais",
    "a0_keyness_density_vs_transit.csv": "keyness_density_vs_transit",
    "a0_frames_role_body.csv": "frames_role_body",
    "a0_frames_by_date.csv": "frames_by_date",
    "a0_frames_by_month.csv": "frames_by_month",
    "a0_frames_by_family.csv": "frames_by_family",
    "a0_speakers.csv": "speakers",
    "a0_term_counts.csv": "term_counts",
    "a0_frame_counts.csv": "frame_counts",
}


def numeric(df):
    """CSV twins are written by csv.DictWriter, so every column arrives as text.
    Anything that is fully numeric becomes numeric; the rest is left alone."""
    import pandas as pd
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() and conv.isna().sum() == df[c].isna().sum():
                df[c] = conv
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.expanduser("~/.local/share/ytpt/corpus_a0_pilot.sqlite"))
    ap.add_argument("--csv-dirs", nargs="*", default=[str(ROOT / "output" / "analysis" / "a0"),
                                                      str(ROOT / "output" / "analysis")])
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "analysis" / "a0"))
    a = ap.parse_args(argv)
    try:
        import pandas as pd
        import duckdb
    except ImportError as e:
        sys.exit(f"{e}. Phase A0's parquet/DuckDB deliverable needs them: "
                 f".venv/bin/pip install duckdb pyarrow")

    out = Path(a.out_dir); pq = out / "parquet"; pq.mkdir(parents=True, exist_ok=True)
    ddb = out / "a0.duckdb"
    if ddb.exists():
        ddb.unlink()                       # a rebuild replaces, never appends
    con = duckdb.connect(str(ddb))
    src = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)

    have = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    written = []
    for t in BASE + [t for t in OPTIONAL if t in have]:
        df = numeric(pd.read_sql_query(f"SELECT * FROM {t}", src))
        df.to_parquet(pq / f"{t}.parquet", index=False)
        con.register("_df", df); con.execute(f"CREATE TABLE {t} AS SELECT * FROM _df"); con.unregister("_df")
        written.append((t, len(df), len(df.columns), "base"))
    src.close()

    seen = set()
    for csv_name, table in DERIVED.items():
        path = next((Path(d) / csv_name for d in a.csv_dirs if (Path(d) / csv_name).exists()), None)
        if path is None:
            print(f"  {csv_name}: not found; skipped (run a0_plots.py / a0_terms.py first)")
            continue
        if table in seen:
            continue
        seen.add(table)
        df = numeric(pd.read_csv(path))
        df.to_parquet(pq / f"{table}.parquet", index=False)
        con.register("_df", df); con.execute(f"CREATE TABLE {table} AS SELECT * FROM _df"); con.unregister("_df")
        written.append((table, len(df), len(df.columns), "derived"))

    con.execute("CREATE TABLE _manifest (table_name TEXT, rows BIGINT, cols BIGINT, kind TEXT)")
    con.executemany("INSERT INTO _manifest VALUES (?,?,?,?)", written)
    con.close()

    w = max(len(t) for t, *_ in written)
    print(f"{len(written)} tables -> {pq}/*.parquet and {ddb}\n")
    for t, n, c, kind in written:
        print(f"  {kind:<7} {t:<{w}}  {n:>6,} rows  {c:>2} cols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
