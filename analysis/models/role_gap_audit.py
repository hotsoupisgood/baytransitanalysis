#!/usr/bin/env python3
"""
analysis/models/role_gap_audit.py -- the estimand-2 role gap recomputed with
the audited role (state.sqlite `role_audit`, scripts/role_audit.py) in place of
the diarization-cue role, for comparison only. Same model, same priors, same
sweep as analysis/models/role_gap.py; the ONLY change is the `role` column.

Writes, beside the originals in output/analysis/models/tables/:
  role_gap_input_counts_roleaudit.csv   public/dais turn counts per body, both roles
  role_gap_gap_<outcome>_all_roleaudit.csv
  role_gap_coef_<outcome>_all_roleaudit.csv
Nothing in output/analysis/models/summary.json or the original tables is touched.

    .venv/bin/python analysis/models/role_gap_audit.py [--counts-only]
"""
import argparse, sys, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from analysis.models import load as L, role_gap as RG  # noqa: E402

TABLES = ROOT / "output" / "analysis" / "models" / "tables"


def turns_both():
    st = L.open_state(None)
    try:
        df = L.read_turn_stance(st)
        ra = pd.read_sql_query("SELECT episode_id, cluster, turn_idx, role_audit, cluster_role_audit FROM role_audit", st)
    finally:
        st.close()
    d = df.merge(ra, on=["episode_id", "cluster", "turn_idx"], how="left")
    d["role_cue"] = d["role"]
    d["role_aud"] = d["role_audit"].fillna(d["role"])          # per-turn rule
    d["role_clu"] = d["cluster_role_audit"].fillna(d["role"])  # cluster roll-up
    return d


def counts(d):
    rows = []
    for which, col in (("cue", "role_cue"), ("audit_turn", "role_aud"), ("audit_cluster", "role_clu")):
        t = d.assign(role=d[col])
        for oc in ("stance", "hostility"):
            p = RG.prepare(t, oc)
            g = p.groupby(["body", "role"]).size().unstack(fill_value=0)
            for body, r in g.iterrows():
                rows.append({"role_col": which, "outcome": oc, "body": body,
                             "dais": int(r.get("dais", 0)), "public": int(r.get("public", 0))})
    c = pd.DataFrame(rows)
    w = c.pivot_table(index=["outcome", "body"], columns="role_col", values=["public", "dais"], fill_value=0)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    w["public_delta_cluster"] = w["public_audit_cluster"] - w["public_cue"]
    w.to_csv(TABLES / "role_gap_input_counts_roleaudit.csv", index=False)
    s = w[w.outcome == "stance"].sort_values("public_cue", ascending=False)
    print(s[["body", "dais_cue", "public_cue", "public_audit_turn", "public_audit_cluster", "public_delta_cluster"]].to_string(index=False))
    tot = s[["dais_cue", "public_cue", "public_audit_turn", "public_audit_cluster"]].sum()
    print("TOTAL(stance):", tot.to_dict())
    return w


def refit(d, draws, tune, chains, role_col="role_clu", sfx="_roleaudit"):
    t = d.assign(role=d[role_col])
    t0 = time.time()
    fits, skipped = RG.sweep(t, facet="role_source", outcomes=("stance", "hostility"),
                             draws=draws, tune=tune, chains=chains)
    for (lev, oc), f in fits.items():
        s = RG.summarise(f)
        s["gap"].to_csv(TABLES / f"role_gap_gap_{oc}_{lev}{sfx}.csv", index=False)
        s["coef"].to_csv(TABLES / f"role_gap_coef_{oc}_{lev}{sfx}.csv", index=False)
        print(f"--- {oc}/{lev} ({role_col})")
        print(s["coef"].to_string(index=False))
    print("skipped:", skipped, "| seconds", round(time.time() - t0, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts-only", action="store_true")
    ap.add_argument("--role-col", choices=("role_clu", "role_cue", "role_aud"), default="role_clu",
                    help="role_clu = audited cluster roll-up (default); role_cue = like-for-like baseline on the same input")
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args(argv)
    TABLES.mkdir(parents=True, exist_ok=True)
    d = turns_both()
    counts(d)
    if not a.counts_only:
        sfx = a.suffix if a.suffix is not None else ("_cuebaseline" if a.role_col == "role_cue" else "_roleaudit")
        refit(d, a.draws, a.tune, a.chains, role_col=a.role_col, sfx=sfx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
