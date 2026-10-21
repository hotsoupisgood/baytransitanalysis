#!/usr/bin/env python3
"""
analysis/stance_s2_site_numbers.py -- the handful of numbers ~/ytpt-site's
panel prose quotes as literals (the frame-wide public split, the public cluster
base cue vs audit, how many clusters the audit moved, the San Leandro sign-flip
share and the split with San Leandro dropped, the one-scored-turn share). They
were written by hand on 2026-10-18 and go stale the moment the frame grows, so
S2 prints them from the view's own artefacts instead of re-deriving them.

    .venv/bin/python analysis/stance_s2_site_numbers.py
    .venv/bin/python analysis/stance_s2_site_numbers.py --json
"""
import argparse, json, sqlite3, sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import state  # noqa: E402

VIEW = ROOT / "output" / "stance_s1" / "nimby_view"


def split_of(cl):
    """YIMBY / neutral / NIMBY shares of the public clusters that carry a stance."""
    g = cl[(cl.role == "public") & (cl.label != "no stance")]
    c, n = Counter(g.label), len(g)
    return {"n": n, **{k: round(c[k] / n, 4) if n else None for k in ("YIMBY", "neutral", "NIMBY")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = {}

    audit = pd.read_csv(VIEW / "speaker_clusters.csv")        # role_audit, the reported view
    cue = pd.read_csv(VIEW / "speaker_clusters_cue.csv")
    out["split_audit"] = split_of(audit)
    out["split_cue"] = split_of(cue)
    out["public_clusters_with_stance"] = {"cue": out["split_cue"]["n"], "audit": out["split_audit"]["n"]}

    con = sqlite3.connect(f"file:{state.db_path()}?mode=ro", uri=True)
    ra = pd.read_sql_query("SELECT episode_id, cluster, role_cue, cluster_role_audit FROM role_audit", con)
    pub = pd.read_sql_query(
        "SELECT episode_id, cluster, turn_idx, stance_median, sub_topic FROM turn_stance", con)
    con.close()
    cl_ra = ra.drop_duplicates(["episode_id", "cluster"])
    out["clusters_changed"] = int((cl_ra.role_cue != cl_ra.cluster_role_audit).sum())
    out["clusters_total"] = int(len(cl_ra))

    # San Leandro sign flip, under the reported (audited) role
    sl = audit[(audit.role == "public") & (audit.body == "sanleandro") & (audit.label != "no stance")]
    key = set(zip(sl.episode_id, sl.cluster))
    flip_cl = sum(1 for e, c in key
                  if "affordability_displacement" in str(
                      ";".join(pub[(pub.episode_id == e) & (pub.cluster == c)].sub_topic.fillna(""))))
    out["sanleandro"] = {"clusters": int(len(sl)), "flip_clusters": int(flip_cl),
                         "flip_share": round(flip_cl / len(sl), 3) if len(sl) else None,
                         "share_of_public_base": round(len(sl) / out["split_audit"]["n"], 3)}
    out["split_audit_drop_sanleandro"] = split_of(audit[audit.body != "sanleandro"])
    out["split_cue_drop_sanleandro"] = split_of(cue[cue.body != "sanleandro"])

    # how thin a cluster label is
    pubcl = audit[audit.role == "public"]
    withst = pubcl[pubcl.label != "no stance"]
    out["public_cluster_scored_turns"] = {
        "mean_all": round(float(pubcl.n_numeric.mean()), 2),
        "mean_with_stance": round(float(withst.n_numeric.mean()), 2),
        "share_one_numeric": round(float((withst.n_numeric == 1).mean()), 3)}

    if a.json:
        print(json.dumps(out, indent=2))
        return 0
    for k, v in out.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
