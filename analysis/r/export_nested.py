#!/usr/bin/env python3
"""
export_nested.py -- the run-6 nested row sets, one outcome per call, so the
hostility model and the drop-BART sensitivity run on the same rows, the same
exclusions and the same keys as the two headline steps (2026-10-20).

`analysis/r/export_for_r.py` writes the within-side rows (NIMBY vs YIMBY given
a side) and `analysis/r/export_anyside.py` the any-side rows (took a side at
all). Both are fixed scripts with one outcome each. Step 6 of the freeze also
needs the same specification fitted with a third outcome (hostility) and each
outcome refitted with BART dropped, so this script parameterises only the
outcome and the body filter and changes nothing else: staff dropped, dais or
public only, bodies with covariates, episode inside meeting inside body, route
a fixed effect, renter share z-scored over the rows that remain.

    --outcome anyside      y = |stance_median| >= 0.5     (= export_anyside.py)
    --outcome withinside   y = stance_median <= -0.5      (= export_for_r.py turns.csv)
    --outcome hostility    y = hostility_median >= 1
    --drop-bart            exclude body 'bart'

Writes output/renter_share_6/data/turns_<outcome>[_nobart].csv and the matching
_meta.json, for analysis/r/turn_nested.R. Read-only on the DB; no spend.
"""
import argparse
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

OUTCOMES = {
    "anyside":    ("stance_median",    lambda v: 1.0 if abs(v) >= 0.5 else 0.0, None),
    "withinside": ("stance_median",    lambda v: 1.0 if v <= -0.5 else 0.0,
                   lambda v: abs(v) >= 0.5),
    "hostility":  ("hostility_median", lambda v: 1.0 if v >= 1 else 0.0, None),
}


def rows_for(con, staff, cov, outcome, drop_bart):
    """Every turn carrying this outcome, with the run-6 exclusions."""
    col, to_y, keep = OUTCOMES[outcome]
    role_of, route_of = {}, {}
    for r in csv.DictReader(open(F.CLUSTERS)):
        role_of[(r["episode_id"], r["cluster"])] = r["role"]
        route_of[(r["episode_id"], r["cluster"])] = r["method"]
    meeting = dict(con.execute("select episode_id, meeting_id from episodes"))
    rows = []
    q = (f"select episode_id, cluster, body, {col} from turn_stance "
         f"where {col} is not null")
    for ep, cl, body, v in con.execute(q):
        key = (ep, cl)
        if key in staff or key not in role_of or body not in cov:
            continue
        if role_of[key] not in ("dais", "public"):
            continue
        if drop_bart and body == "bart":
            continue
        if keep is not None and not keep(v):
            continue
        rows.append(dict(y=to_y(v), body=body, episode_id=ep,
                         meeting_id=meeting.get(ep, ep),
                         route=route_of[key], role=role_of[key],
                         role_code=0 if role_of[key] == "dais" else 1,
                         renter_share=cov[body]["renter_share"]))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", required=True, choices=sorted(OUTCOMES))
    ap.add_argument("--drop-bart", action="store_true")
    a = ap.parse_args(argv)

    con = sqlite3.connect(f"file:{F.DB}?mode=ro", uri=True)
    cov = F.covariates()
    staff = F.staff_clusters(con)
    rows = rows_for(con, staff, cov, a.outcome, a.drop_bart)
    if not rows:
        sys.exit(f"no rows for outcome {a.outcome}")
    ren = np.array([r["renter_share"] for r in rows], float)
    zren = (ren - ren.mean()) / ren.std(ddof=0)

    stem = a.outcome + ("_nobart" if a.drop_bart else "")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"turns_{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y", "body", "meeting_id", "episode_id", "route",
                    "role", "role_code", "renter_share", "z_renter"])
        for i, r in enumerate(rows):
            w.writerow([int(r["y"]), r["body"], r["meeting_id"],
                        r["episode_id"], r["route"], r["role"], r["role_code"],
                        f"{r['renter_share']:.10g}", f"{zren[i]:.10g}"])
    meta = dict(
        outcome=a.outcome, drop_bart=bool(a.drop_bart), stem=stem,
        n=len(rows), n_y=int(sum(r["y"] for r in rows)),
        episodes=len({r["episode_id"] for r in rows}),
        meetings=len({r["meeting_id"] for r in rows}),
        bodies=len({r["body"] for r in rows}),
        routes=sorted({r["route"] for r in rows}),
        z_renter_mean=float(ren.mean()), z_renter_sd=float(ren.std(ddof=0)),
        clusters_csv=str(F.CLUSTERS), db=str(F.DB))
    (OUT / f"turns_{stem}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{stem}: {meta['n']} turns, y={meta['n_y']}, "
          f"{meta['episodes']} episodes, {meta['meetings']} meetings, "
          f"{meta['bodies']} bodies, routes {','.join(meta['routes'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
