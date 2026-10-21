#!/usr/bin/env python3
"""
export_for_r.py -- emit the EXACT row sets and design matrices used by
analysis/renter_share5_freq.py so that R (lme4 / glmmTMB) fits the same model
on the same numbers. Nothing is re-derived here: the build/design/turn_level
functions are imported from the run-5 script.

Writes output/renter_share_6/data/:
  rows_primary.csv     56 rows, body x route x role, NIMBY/(NIMBY+YIMBY),
                       staff dropped, + the standardised design columns
  rows_secondary.csv   64 rows, published denominator, staff kept
  turns.csv            turn-level: NIMBY vs YIMBY turn, body/meeting/episode
  design_meta.json     standardisation means/sds, column names

Read-only on the DB and on runs 2-5. No network, no spend, nothing committed.
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
OUT.mkdir(parents=True, exist_ok=True)


def safe(nm):
    nm = (nm.replace("[", "_").replace("]", "")
            .replace("-", "_").replace(".", "_"))
    # `role` would collide with the role label column; the design column is
    # the numeric 0/1 contrast.
    return "role_num" if nm == "role" else nm


def write_rows(path, rows, X, names):
    cols = ["route", "body", "role", "role_code", "y", "N", "episodes"]
    xn = [safe(n) for n in names if n != "const"]
    keep = [j for j, n in enumerate(names) if n != "const"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + xn)
        for i, r in enumerate(rows):
            w.writerow([r[c] for c in cols] + [f"{X[i, j]:.10g}" for j in keep])
    return xn


def main():
    con = sqlite3.connect(f"file:{F.DB}?mode=ro", uri=True)
    cov = F.covariates()
    staff = F.staff_clusters(con)
    cl = F.clusters(staff)

    meta = {"staff_clusters": len(staff), "kept_covariates": F.KEPT,
            "min_n": F.MIN_N, "ref_route": F.REF_ROUTE}

    for tag, directional, drop_staff in (("primary", True, True),
                                         ("secondary", False, False)):
        rows = F.build(cl, cov, directional=directional, drop_staff=drop_staff)
        X, names, dmeta = F.design(rows)
        xn = write_rows(OUT / f"rows_{tag}.csv", rows, X, names)
        meta[tag] = dict(
            rows=len(rows), bodies=len({r["body"] for r in rows}),
            trials=int(sum(r["N"] for r in rows)),
            nimby=int(sum(r["y"] for r in rows)),
            design_columns=xn, **dmeta)
        print(f"{tag}: {len(rows)} rows, {meta[tag]['bodies']} bodies, "
              f"{meta[tag]['trials']} trials, {meta[tag]['nimby']} NIMBY")

    trows = F.turn_level(con, staff, cov)
    ren = np.array([r["renter_share"] for r in trows], float)
    zren = (ren - ren.mean()) / ren.std(ddof=0)
    with open(OUT / "turns.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["y", "body", "meeting_id", "episode_id", "route",
                    "role", "role_code", "renter_share", "z_renter"])
        for i, r in enumerate(trows):
            w.writerow([int(r["y"]), r["body"], r["meeting_id"], r["episode_id"],
                        r["route"], r["role"], r["role_code"],
                        f"{r['renter_share']:.10g}", f"{zren[i]:.10g}"])
    meta["turns"] = dict(
        n=len(trows), nimby=int(sum(r["y"] for r in trows)),
        episodes=len({r["episode_id"] for r in trows}),
        meetings=len({r["meeting_id"] for r in trows}),
        bodies=len({r["body"] for r in trows}),
        z_renter_mean=float(ren.mean()), z_renter_sd=float(ren.std(ddof=0)))
    print("turns:", meta["turns"])

    (OUT / "design_meta.json").write_text(json.dumps(meta, indent=2))
    print("->", OUT)


if __name__ == "__main__":
    main()
