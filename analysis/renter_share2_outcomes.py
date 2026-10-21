#!/usr/bin/env python3
"""
renter_share2_outcomes.py -- outcomes and the collinearity screen for
plan/RENTER-SHARE-2.md.

WHAT CHANGED FROM v1. v1's stance unit was the TURN, off `turn_stance` as it
stood on 2026-10-18 (9,358 rows). S2 has landed (plan/STANCE-S2.md:
`turn_stance` = 13,123 rows, role_audit rewritten, a fifth locate route
`fullasr`), and the per-cluster view is now big enough to be the unit:
861 public and 1,952 dais clusters with a stance. This file therefore reads
the **cluster** table

    output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv

which is the post-S2, role_audit (rule ra-1) roll-up, one row per
episode x speaker-cluster with a `label` in {YIMBY, NIMBY, neutral, no stance}.
A cluster "has a stance" iff label != "no stance". The speaker cluster is also
the unit STANCE-S1 7b and the site figures use, so these outcomes line up with
the published per-body panels instead of running beside them.

Outcomes, per body x locate route (= `method`), never pooled across routes
(analysis/facets.py; route enters the models as a fixed-effect stratum):

  1. pub_nimby  / pub_n   public NIMBY clusters out of public clusters with a stance
  2. pub_yimby  / pub_n   public YIMBY clusters, same denominator
  3. all_nimby  / all_n   public + dais NIMBY clusters out of all clusters with a stance
  4. all_yimby  / all_n   public + dais YIMBY clusters, same denominator
  5. rate_density         located density minutes per meeting-hour from
                          output/wave1/rates_by_body_route.csv, COMPLETE
                          coverage only. After the 2026-10-18 coverage fix
                          (plan/VALIDATION-2026-10-18.md "Validation applied"
                          1) the complete set is 3 bodies, so this is reported
                          descriptively and NOT modelled.

Family frame is inherited from the cluster table: density + both only (transit
is a different estimand -- STANCE-S1 7b).

Also computes the covariate screen the sparsity argument needs: the Pearson
correlation matrix and the VIF of every candidate covariate over the fitted
bodies, and a greedy drop list (drop the highest-VIF covariate until all
VIF < VIF_MAX), so plan/RENTER-SHARE-2.md can say which covariates were
dropped and why.

Reads only: output/stance_s1/nimby_view/speaker_clusters_roleaudit.csv,
output/wave1/rates_by_body_route.csv, output/renter_share_2/covariates_wide.csv.
Writes output/renter_share_2/{outcomes_clusters.csv, rates_complete.csv,
covariate_corr.csv, covariate_vif.csv, screen.json}. No DB, no spend,
nothing committed.
"""
import csv
import json
import pathlib
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "renter_share_2"
CLUSTERS = (ROOT / "output" / "stance_s1" / "nimby_view" /
            "speaker_clusters_roleaudit.csv")
RATES = ROOT / "output" / "wave1" / "rates_by_body_route.csv"

MIN_N = 5            # floor on clusters with a stance, as v1's turn floor
VIF_MAX = 5.0

#: Every ACS candidate the brief lists. `renter_share` and `med_hh_income` are
#: the two the question is about and are never dropped by the screen.
CANDIDATES = [
    "renter_share", "log_income", "log_density", "log_home_value",
    "log_gross_rent", "rent_burden_30", "built_2010_plus", "multiunit_5plus",
    "ba_plus", "median_age", "nhwhite_share", "transit_commute",
    "vacancy_rate", "log_population",
]
FORCED = ["renter_share", "log_income"]


def covariates():
    out = {}
    for r in csv.DictReader(open(OUT / "covariates_wide.csv")):
        d = dict(body=r["body"], body_type=r["body_type"],
                 geo_kind=r["geo_kind"],
                 medians_are_weighted_means=int(r["medians_are_weighted_means"]))
        d["renter_share"] = float(r["renter_share"])
        d["med_hh_income"] = float(r["med_hh_income"])
        d["pop_density"] = float(r["pop_density_per_sq_mi"])
        d["population"] = float(r["population"])
        d["med_home_value"] = float(r["med_home_value"])
        d["med_gross_rent"] = float(r["med_gross_rent"])
        for k in ("rent_burden_30", "built_2010_plus", "multiunit_5plus",
                  "ba_plus", "median_age", "nhwhite_share", "transit_commute",
                  "vacancy_rate"):
            d[k] = float(r[k])
        # logs for the heavy-tailed dollar and count covariates; shares stay raw
        d["log_income"] = float(np.log(d["med_hh_income"]))
        d["log_density"] = float(np.log(d["pop_density"]))
        d["log_home_value"] = float(np.log(d["med_home_value"]))
        d["log_gross_rent"] = float(np.log(d["med_gross_rent"]))
        d["log_population"] = float(np.log(d["population"]))
        out[r["body"]] = d
    return out


def cluster_counts():
    """(route, body) -> per-role label counts over clusters WITH a stance."""
    cnt = defaultdict(lambda: defaultdict(lambda: dict(
        NIMBY=0, YIMBY=0, neutral=0, n=0, no_stance=0, episodes=set())))
    fams = set()
    for r in csv.DictReader(open(CLUSTERS)):
        role = r["role"]
        if role not in ("public", "dais"):
            continue
        fams.add(r["family"])
        c = cnt[(r["method"], r["body"])][role]
        c["episodes"].add(r["episode_id"])
        lab = r["label"]
        if lab == "no stance":
            c["no_stance"] += 1
            continue
        c["n"] += 1
        if lab in c:
            c[lab] += 1
    for v in cnt.values():
        for c in v.values():
            c["episodes"] = len(c["episodes"])
    return cnt, sorted(fams)


def rates_complete():
    out = []
    for r in csv.DictReader(open(RATES)):
        if r["coverage"] != "complete" or r["family"] not in ("density", "transit"):
            continue
        out.append(dict(body=r["body"], route=r["route"], family=r["family"],
                        rate_min_per_h=float(r["rate_min_per_h"] or 0),
                        meeting_hours=float(r["meeting_hours"] or 0),
                        meetings=int(r["meetings"] or 0),
                        episode_minutes=float(r["episode_minutes"] or 0)))
    return out


def vif_table(rows, cols):
    """VIF_j = 1/(1 - R2_j) from regressing covariate j on the others.
    Computed over the FITTED BODIES (one row per body, not per cell), which is
    the n that actually limits the design."""
    X = np.column_stack([[r[c] for r in rows] for c in cols])
    X = (X - X.mean(0)) / X.std(0, ddof=0)
    out = {}
    for j, c in enumerate(cols):
        others = np.column_stack([X[:, k] for k in range(len(cols)) if k != j]
                                 + [np.ones(len(X))])
        beta, *_ = np.linalg.lstsq(others, X[:, j], rcond=None)
        res = X[:, j] - others @ beta
        sst = float(((X[:, j] - X[:, j].mean()) ** 2).sum())
        r2 = 1 - float(res @ res) / sst if sst else 0.0
        out[c] = float(1 / (1 - r2)) if r2 < 1 - 1e-12 else float("inf")
    return out


def greedy_drop(rows, cols, forced, vif_max=VIF_MAX):
    """Drop the highest-VIF non-forced covariate until every VIF < vif_max or
    only the forced ones are left. Returns (kept, drop log)."""
    keep = list(cols)
    log = []
    while True:
        if len(keep) <= len(forced) + 0:
            break
        v = vif_table(rows, keep)
        droppable = {k: x for k, x in v.items() if k not in forced}
        if not droppable:
            break
        worst, wv = max(droppable.items(), key=lambda kv: kv[1])
        if wv < vif_max:
            break
        keep.remove(worst)
        log.append(dict(dropped=worst, vif=wv,
                        remaining=len(keep),
                        note=f"VIF {wv:.1f} >= {vif_max} among {len(keep)+1} covariates"))
    return keep, log, vif_table(rows, keep)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cov = covariates()
    cnt, fams = cluster_counts()

    table = []
    for (route, body), roles in sorted(cnt.items()):
        if body not in cov:
            print(f"note: no ACS row for body {body!r} ({route}); skipped")
            continue
        pub = roles.get("public", dict(NIMBY=0, YIMBY=0, neutral=0, n=0,
                                       no_stance=0, episodes=0))
        dai = roles.get("dais", dict(NIMBY=0, YIMBY=0, neutral=0, n=0,
                                     no_stance=0, episodes=0))
        row = dict(route=route, **{k: v for k, v in cov[body].items()})
        row.update(
            pub_n=pub["n"], pub_nimby=pub["NIMBY"], pub_yimby=pub["YIMBY"],
            pub_neutral=pub["neutral"], pub_no_stance=pub["no_stance"],
            pub_episodes=pub["episodes"],
            dais_n=dai["n"], dais_nimby=dai["NIMBY"], dais_yimby=dai["YIMBY"],
            all_n=pub["n"] + dai["n"],
            all_nimby=pub["NIMBY"] + dai["NIMBY"],
            all_yimby=pub["YIMBY"] + dai["YIMBY"])
        for pre in ("pub", "all"):
            n = row[f"{pre}_n"]
            row[f"{pre}_nimby_share"] = round(row[f"{pre}_nimby"] / n, 6) if n else None
            row[f"{pre}_yimby_share"] = round(row[f"{pre}_yimby"] / n, 6) if n else None
            row[f"{pre}_net"] = (round((row[f"{pre}_yimby"] - row[f"{pre}_nimby"]) / n, 6)
                                 if n else None)
        table.append(row)

    cols = [c for c in table[0]]
    with open(OUT / "outcomes_clusters.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in table:
            w.writerow(r)

    rc = rates_complete()
    with open(OUT / "rates_complete.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rc[0]))
        w.writeheader()
        for r in rc:
            w.writerow(r)

    # ---- the covariate screen, over the bodies the primary outcome fits
    fitted = [r for r in table if r["pub_n"] >= MIN_N]
    bodies = {}
    for r in fitted:
        bodies.setdefault(r["body"], r)
    brows = sorted(bodies.values(), key=lambda r: r["body"])
    print(f"public-stance cells (floor {MIN_N} clusters): {len(fitted)} over "
          f"{len(brows)} bodies, {sum(r['pub_n'] for r in fitted)} public clusters")
    all_fitted = [r for r in table if r["all_n"] >= MIN_N]
    print(f"all-roles cells (floor {MIN_N}): {len(all_fitted)} over "
          f"{len(set(r['body'] for r in all_fitted))} bodies, "
          f"{sum(r['all_n'] for r in all_fitted)} clusters")

    C = np.corrcoef(np.column_stack([[r[c] for r in brows]
                                     for c in CANDIDATES]).T)
    with open(OUT / "covariate_corr.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([""] + CANDIDATES)
        for i, c in enumerate(CANDIDATES):
            w.writerow([c] + [f"{C[i, j]:+.3f}" for j in range(len(CANDIDATES))])

    vif_all = vif_table(brows, CANDIDATES)
    keep, droplog, vif_keep = greedy_drop(brows, CANDIDATES, FORCED)
    with open(OUT / "covariate_vif.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["covariate", "vif_all_candidates", "vif_kept_set", "kept"])
        for c in CANDIDATES:
            w.writerow([c, f"{vif_all[c]:.2f}",
                        f"{vif_keep[c]:.2f}" if c in vif_keep else "",
                        int(c in keep)])

    screen = dict(
        n_bodies_public=len(brows), n_cells_public=len(fitted),
        n_bodies_all=len(set(r["body"] for r in all_fitted)),
        n_cells_all=len(all_fitted),
        families=fams, min_n=MIN_N, vif_max=VIF_MAX,
        candidates=CANDIDATES, forced=FORCED, kept=keep, drop_log=droplog,
        vif_all=vif_all, vif_kept=vif_keep,
        top_abs_corr=sorted(
            [dict(a=CANDIDATES[i], b=CANDIDATES[j], r=float(C[i, j]))
             for i in range(len(CANDIDATES)) for j in range(i + 1, len(CANDIDATES))],
            key=lambda d: -abs(d["r"]))[:15],
        rates_complete_bodies=sorted({r["body"] for r in rc}),
        rates_complete_note=("complete-coverage set is 3 bodies after the "
                             "2026-10-18 coverage fix; descriptive only, not modelled"))
    (OUT / "screen.json").write_text(json.dumps(screen, indent=2))

    print("\nVIF over all candidates (n bodies = %d):" % len(brows))
    for c in CANDIDATES:
        print(f"  {c:18s} {vif_all[c]:9.2f}" +
              ("" if c in keep else "   DROPPED"))
    print("\ndrop order:")
    for d in droplog:
        print(f"  {d['dropped']:18s} VIF {d['vif']:.1f}")
    print("\nkept:", keep)
    print("VIF on the kept set:", {k: round(v, 2) for k, v in vif_keep.items()})
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
