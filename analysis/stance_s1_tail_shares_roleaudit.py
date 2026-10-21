#!/usr/bin/env python3
"""
analysis/stance_s1_tail_shares_roleaudit.py -- the model-free public-dais tail
shares of STANCE-S1-RESULTS 7 (opposed / supportive / not_applicable / hostile,
episode-cluster bootstrap) recomputed under a switchable role column, so the
cue role and the audited role (`role_audit`, rule ra-1) can be read side by
side. plan/ROLE-AUDIT-2026-10-18.md 7: the audited role is the reported one.

The proportional-odds fit misfits the public stance distribution (7 PPC: the
public is bimodal where the dais sits at neutral), so these shares -- not the
latent gap -- are the number STANCE-S1-RESULTS quotes. They have to move with
the role column, and before this script they did not.

Exactly analysis/stance_s1_models.py's `tail_bootstrap` on exactly its input
(output/stance_s1/s1_frame.csv joined to turn_stance), with `role` swapped and
nothing else: same reps, same seed, same per-episode weighted-sum bootstrap.
The role instrument facet (`role_source`, db / computed) is kept, since the
audit does not change which episodes carry db roles.

Writes output/stance_s1/s1_tail_shares_bootstrap{_roleaudit,_cue}.csv. No DB
write, no model refit, no spend.

    .venv/bin/python analysis/stance_s1_tail_shares_roleaudit.py                 # audit (reported)
    .venv/bin/python analysis/stance_s1_tail_shares_roleaudit.py --role-col cue  # _cue baseline
"""
import argparse, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "pipeline"))

from analysis import stance_s1_models as M  # noqa: E402
import stance, state  # noqa: E402

OUT = ROOT / "output" / "stance_s1"
SUFFIX = {"audit": "_roleaudit", "cue": "_cue"}
TAILS = (("opposed (stance <= -1 | numeric)", "opposed"),
         ("supportive (stance >= 1 | numeric)", "supportive"),
         ("hostile (hostility >= 2)", "hostile"))


def load(pv, frame_csv, role_col):
    """turn_stance joined to the S1 frame, `role` from the chosen column."""
    con = state.connect(None, create=False)
    ts = pd.read_sql_query(
        "SELECT episode_id, cluster, turn_idx, stance, stance_median, stance_spread, "
        "hostility_median, disagree, words, role, body, family, sub_topic, "
        "contains_instructions, cost_usd FROM turn_stance WHERE prompt_version = ?",
        con, params=(pv,))
    n_cue_public = int((ts.role == "public").sum())
    if role_col == "audit":
        ra = pd.read_sql_query(
            "SELECT episode_id, cluster, turn_idx, cluster_role_audit FROM role_audit "
            "WHERE prompt_version = ?", con, params=(pv,))
        ts = ts.merge(ra, on=["episode_id", "cluster", "turn_idx"], how="left", validate="one_to_one")
        miss = int(ts.cluster_role_audit.isna().sum())
        if miss:
            print(f"warning: {miss} turns with no role_audit row; cue role kept", flush=True)
        ts["role"] = ts.cluster_role_audit.fillna(ts.role)
    con.close()
    fr = pd.read_csv(frame_csv, usecols=["episode_id", "cluster", "turn_idx", "method", "roles_source"])
    df = ts.merge(fr, on=["episode_id", "cluster", "turn_idx"], how="inner", validate="one_to_one")
    df["scored"] = df["stance"].notna()
    df["role_source"] = df["roles_source"]
    print(f"{len(df)} frame turns, public {n_cue_public} (cue) -> {int((df.role == 'public').sum())} "
          f"({role_col}), dais {int((df.role == 'dais').sum())}", flush=True)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role-col", choices=tuple(SUFFIX), default="audit")
    ap.add_argument("--frame", default=str(OUT / "s1_frame.csv"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261018)
    a = ap.parse_args(argv)
    sfx = a.suffix if a.suffix is not None else SUFFIX[a.role_col]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pv = stance.prompt_version_for(state.setting("SCORING_MODEL", stance.DEFAULT_MODEL))
    df = load(pv, a.frame, a.role_col)
    t = M.tail_bootstrap(df, out, reps=a.reps, seed=a.seed)
    # tail_bootstrap writes the unsuffixed name; move it to the role-column one
    src = out / "s1_tail_shares_bootstrap.csv"
    dst = out / f"s1_tail_shares_bootstrap{sfx}.csv"
    if sfx and src.exists():
        src.replace(dst)
    t.insert(0, "role_col", a.role_col)
    t.to_csv(dst, index=False)
    show = t[(t.role_source == "db") & t.statistic.isin([s for s, _ in TAILS])]
    print(f"\n[{a.role_col}] role instrument db, public - dais (90 %):", flush=True)
    for _, r in show.iterrows():
        print(f"  {r.family:<22} {r.statistic:<36} public {r.public:.3f} / dais {r.dais:.3f}  "
              f"diff {r.diff_public_minus_dais:+.3f} ({r.diff_lo90:+.3f}, {r.diff_hi90:+.3f})  "
              f"n={r.n_turns} (public {r.n_public})", flush=True)
    print(f"wrote {dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
