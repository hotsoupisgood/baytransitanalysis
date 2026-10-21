#!/usr/bin/env python3
"""
export_anyside.py -- the first step of the role gap, on the same rows and the
same nesting as the second.

`analysis/r/export_for_r.py` exports the turns that already took a side, so the
glmer over it answers "which side, given a side". This script exports the step
before it: every stance-bearing turn, with y = 1 if the turn took a side at all
(|stance_median| >= 0.5) and 0 if it sits in the neutral band. Same frame, same
exclusions (staff dropped, dais/public only, bodies with covariates), same
keys -- episode inside meeting inside body -- so `analysis/r/turn_anyside.R`
can fit the identical nested model and the two steps are comparable.

Nothing is re-derived: the cluster/role/route tables and the covariate set come
from `analysis/renter_share5_freq.py`, exactly as run 6 does.

Writes output/renter_share_6/data/turns_anyside.csv and
turns_anyside_meta.json. Read-only on the DB; no network, no spend.
"""
import csv
import json
import pathlib
import sqlite3
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import renter_share5_freq as F  # noqa: E402

OUT = ROOT / "output" / "renter_share_6" / "data"


def turn_level_anyside(con, staff, cov):
    """Every stance-bearing turn; y = took a side rather than which side."""
    role_of, route_of = {}, {}
    for r in csv.DictReader(open(F.CLUSTERS)):
        role_of[(r["episode_id"], r["cluster"])] = r["role"]
        route_of[(r["episode_id"], r["cluster"])] = r["method"]
    meeting = dict(con.execute("select episode_id, meeting_id from episodes"))
    rows = []
    q = ("select episode_id, cluster, body, stance_median from turn_stance "
         "where stance_median is not null")
    for ep, cl, body, sm_ in con.execute(q):
        key = (ep, cl)
        if key in staff or key not in role_of or body not in cov:
            continue
        if role_of[key] not in ("dais", "public"):
            continue
        rows.append(dict(y=1.0 if abs(sm_) >= 0.5 else 0.0, body=body,
                         episode_id=ep, meeting_id=meeting.get(ep, ep),
                         route=route_of[key], role=role_of[key],
                         role_code=0 if role_of[key] == "dais" else 1,
                         renter_share=cov[body]["renter_share"]))
    return rows


def main():
    con = sqlite3.connect(f"file:{F.DB}?mode=ro", uri=True)
    cov = F.covariates()
    staff = F.staff_clusters(con)
    rows = turn_level_anyside(con, staff, cov)
    ren = np.array([r["renter_share"] for r in rows], float)
    zren = (ren - ren.mean()) / ren.std(ddof=0)
    with open(OUT / "turns_anyside.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y", "body", "meeting_id", "episode_id", "route",
                    "role", "role_code", "renter_share", "z_renter"])
        for i, r in enumerate(rows):
            w.writerow([int(r["y"]), r["body"], r["meeting_id"],
                        r["episode_id"], r["route"], r["role"], r["role_code"],
                        f"{r['renter_share']:.10g}", f"{zren[i]:.10g}"])
    meta = dict(
        n=len(rows), took_a_side=int(sum(r["y"] for r in rows)),
        episodes=len({r["episode_id"] for r in rows}),
        meetings=len({r["meeting_id"] for r in rows}),
        bodies=len({r["body"] for r in rows}),
        by_role={
            role: dict(
                n=sum(1 for r in rows if r["role"] == role),
                took_a_side=int(sum(r["y"] for r in rows
                                    if r["role"] == role)))
            for role in ("public", "dais")},
        z_renter_mean=float(ren.mean()), z_renter_sd=float(ren.std(ddof=0)),
        outcome="took a side (|stance_median| >= 0.5) vs the neutral band",
        source="analysis/r/export_anyside.py")
    (OUT / "turns_anyside_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("->", OUT)


if __name__ == "__main__":
    main()
