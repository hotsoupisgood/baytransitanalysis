#!/usr/bin/env python3
"""
renter_share3_role_outcomes.py -- the STACKED role outcomes for
plan/RENTER-SHARE-3-ROLE.md.

THE QUESTION. plan/RENTER-SHARE-2.md fits two outcome families -- public-only
(pub_nimby/pub_n) and all-roles (all_nimby/all_n) -- so `role` is a *subsetting
variable* there, never a covariate. Puck asked whether dais/public is fitted as
a covariate. It is not. This file builds the unit that makes it one: the
body x route x ROLE row, two rows per body x route cell (public, dais), so role
enters the design matrix and role x renter_share is estimable.

UNIT. One row per (route, body, role) with role in {dais, public}, off the
post-S2 role_audit speaker-cluster table

    output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv

A cluster "has a stance" iff label != "no stance"; n = clusters with a stance,
numerators are NIMBY and YIMBY cluster counts. Floor MIN_N = 5 clusters PER ROW
(plan/RENTER-SHARE-2.md's floor, applied to the finer unit). Route is never
pooled -- it is a fixed-effect stratum in the models, as in runs 1 and 2.

Covariates are read unchanged from output/renter_share_2/covariates_wide.csv
and the kept set is read unchanged from output/renter_share_2/screen.json, so
this run re-uses run 2's ACS pull and its VIF screen and pulls nothing.

Writes output/renter_share_3/{outcomes_role_stacked.csv, cells.json}.
No DB, no network, no spend, nothing committed.
"""
import csv
import json
import pathlib
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT2 = ROOT / "output" / "renter_share_2"
OUT = ROOT / "output" / "renter_share_3"
CLUSTERS = (ROOT / "output" / "stance_s1" / "nimby_view" /
            "speaker_clusters_roleaudit.csv")

MIN_N = 5
ROLES = ("dais", "public")
ROLE_CODE = {"dais": 0, "public": 1}     # the brief's coding

LOGCOLS = {"log_income": "med_hh_income", "log_density": "pop_density",
           "log_home_value": "med_home_value",
           "log_gross_rent": "med_gross_rent", "log_population": "population"}


def covariates():
    out = {}
    for r in csv.DictReader(open(OUT2 / "covariates_wide.csv")):
        d = dict(body=r["body"], body_type=r["body_type"], geo_kind=r["geo_kind"],
                 medians_are_weighted_means=int(r["medians_are_weighted_means"]))
        d["pop_density"] = float(r["pop_density_per_sq_mi"])
        for k in ("renter_share", "med_hh_income", "population",
                  "med_home_value", "med_gross_rent", "rent_burden_30",
                  "built_2010_plus", "multiunit_5plus", "ba_plus",
                  "median_age", "nhwhite_share", "transit_commute",
                  "vacancy_rate"):
            d[k] = float(r[k])
        for lk, raw in LOGCOLS.items():
            d[lk] = float(np.log(d[raw]))
        out[r["body"]] = d
    return out


def role_counts():
    """(route, body, role) -> label counts over clusters with a stance."""
    cnt = defaultdict(lambda: dict(NIMBY=0, YIMBY=0, neutral=0, n=0,
                                   no_stance=0, episodes=set()))
    fams, methods = set(), set()
    for r in csv.DictReader(open(CLUSTERS)):
        role = r["role"]
        if role not in ROLES:
            continue
        fams.add(r["family"])
        methods.add(r["method"])
        c = cnt[(r["method"], r["body"], role)]
        c["episodes"].add(r["episode_id"])
        lab = r["label"]
        if lab == "no stance":
            c["no_stance"] += 1
            continue
        c["n"] += 1
        if lab in c:
            c[lab] += 1
    for c in cnt.values():
        c["episodes"] = len(c["episodes"])
    return cnt, sorted(fams), sorted(methods)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cov = covariates()
    cnt, fams, methods = role_counts()
    kept = json.loads((OUT2 / "screen.json").read_text())["kept"]

    rows, skipped_body, below_floor = [], set(), []
    for (route, body, role), c in sorted(cnt.items()):
        if body not in cov:
            skipped_body.add(body)
            continue
        rec = dict(route=route, role=role, role_code=ROLE_CODE[role],
                   **cov[body])
        rec.update(n=c["n"], nimby=c["NIMBY"], yimby=c["YIMBY"],
                   neutral=c["neutral"], no_stance=c["no_stance"],
                   episodes=c["episodes"])
        rec["nimby_share"] = round(c["NIMBY"] / c["n"], 6) if c["n"] else None
        rec["yimby_share"] = round(c["YIMBY"] / c["n"], 6) if c["n"] else None
        if c["n"] < MIN_N:
            below_floor.append(dict(route=route, body=body, role=role, n=c["n"]))
            continue
        rows.append(rec)

    # pairing: body x route cells where BOTH role rows clear the floor. The
    # body random intercept is shared across a body's role rows, so these cells
    # are the ones that carry a WITHIN-body role contrast.
    bycell = defaultdict(set)
    for r in rows:
        bycell[(r["route"], r["body"])].add(r["role"])
    paired = {k for k, v in bycell.items() if v == set(ROLES)}
    for r in rows:
        r["paired_cell"] = int((r["route"], r["body"]) in paired)

    cols = list(rows[0])
    with open(OUT / "outcomes_role_stacked.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def block(sel):
        return dict(
            rows=len(sel), bodies=len({r["body"] for r in sel}),
            cells=len({(r["route"], r["body"]) for r in sel}),
            clusters=sum(r["n"] for r in sel),
            nimby=sum(r["nimby"] for r in sel),
            yimby=sum(r["yimby"] for r in sel),
            routes={k: sum(1 for r in sel if r["route"] == k)
                    for k in sorted({r["route"] for r in sel})})

    meta = dict(
        min_n=MIN_N, role_code=ROLE_CODE, families=fams, methods=methods,
        kept_covariates=kept,
        all_rows=block(rows),
        dais_rows=block([r for r in rows if r["role"] == "dais"]),
        public_rows=block([r for r in rows if r["role"] == "public"]),
        paired_rows=block([r for r in rows if r["paired_cell"]]),
        n_paired_cells=len(paired),
        paired_cells=sorted(f"{a}/{b}" for a, b in paired),
        below_floor=below_floor,
        bodies_without_acs=sorted(skipped_body),
        source=str(CLUSTERS.relative_to(ROOT)))
    (OUT / "cells.json").write_text(json.dumps(meta, indent=2))

    print(f"stacked rows (floor {MIN_N}/row): {len(rows)} over "
          f"{meta['all_rows']['bodies']} bodies, "
          f"{meta['all_rows']['cells']} body x route cells, "
          f"{meta['all_rows']['clusters']} clusters")
    for nm in ("dais_rows", "public_rows", "paired_rows"):
        b = meta[nm]
        print(f"  {nm:13s} rows {b['rows']:3d} bodies {b['bodies']:3d} "
              f"clusters {b['clusters']:5d} N {b['nimby']:4d} Y {b['yimby']:4d}")
    print(f"  paired body x route cells (both roles >= {MIN_N}): {len(paired)}")
    print(f"  rows below floor: {len(below_floor)}")
    print("\nroute  body            role    n  NIMBY YIMBY  paired")
    for r in sorted(rows, key=lambda r: (r["renter_share"], r["body"],
                                         r["route"], r["role"])):
        print(f"{r['route'][:13]:14s} {r['body'][:13]:14s} {r['role']:6s} "
              f"{r['n']:4d} {r['nimby']:5d} {r['yimby']:5d} "
              f"{r['renter_share']:.3f} {r['paired_cell']}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
