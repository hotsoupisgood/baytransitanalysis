#!/usr/bin/env python3
"""
renter_share5_role_restated.py -- finding 2 of plan/REGRESSION-REVIEW-2026-10-19.md.

plan/RENTER-SHARE-3-ROLE.md quotes the role gap as delta = +2.11 log-odds
("~8x odds") on the outcome NIMBY / (clusters with a stance). The review shows
that denominator is ~60 % the `neutral` class and ~20 % staff clusters, and
that `neutral` is a dais phenomenon (69.1 % of stance-bearing dais clusters
against 10.5 % public). So +2.11 is largely "the dais says neutral things",
not "the public is 8x more NIMBY".

This file refits run 3's stacked role model on three restated outcomes:

  (a) DIRECTIONAL   y = NIMBY, N = NIMBY + YIMBY   (neutral excluded)
  (b) DIR_NOSTAFF   (a) with staff clusters dropped from the dais
  (c) ELECT_RESID   (a) restricted to self_id elected (0) vs resident (1)

Cluster-level self_id is the majority non-`unknown` self_id over the cluster's
turns in role_audit (ties broken elected > staff > org_rep > business >
homeowner > renter > resident), read read-only from the live state DB.

Model, priors, covariates, route stratum, body random intercept and row-level
overdispersion are run 3's, unchanged (analysis/renter_share3_role_models.py);
only the outcome and the row filter change. Floor is run 3's 5 per row, now 5
*directional* clusters per row.

Writes output/renter_share_5/role_restated/{results.json, delta_table.csv,
rows_*.csv}. Read-only on the DB, no network, no spend, nothing committed.

    nohup .venv/bin/python analysis/renter_share5_role_restated.py \
        > output/renter_share_5/role_restated.log 2>&1 &
"""
import collections
import csv
import json
import pathlib
import sqlite3
import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT2 = ROOT / "output" / "renter_share_2"
OUT3 = ROOT / "output" / "renter_share_3"
OUT = ROOT / "output" / "renter_share_5" / "role_restated"
DB = "file:" + str(pathlib.Path.home() / ".local/share/ytpt/state.sqlite") + "?mode=ro"
CLUSTERS = (ROOT / "output" / "stance_s1" / "nimby_view" /
            "speaker_clusters_roleaudit.csv")
STAMP = ("EXPLORATORY — stance κ ≥ 0.6 hand-coding gate not passed; role is an "
         "AUDIT (ra-1), self_id is model output no human has checked")
SEED = 20261019
DRAWS, TUNE, CHAINS = 2000, 1500, 4
MIN_N = 5
ROUTE_REF = "classifier"
SELF_PRI = {"elected": 0, "staff": 1, "org_rep": 2, "business": 3,
            "homeowner": 4, "renter": 5, "resident": 6, "unknown": 9}
LOGCOLS = {"log_income": "med_hh_income", "log_density": "pop_density",
           "log_population": "population"}


def covariates():
    out = {}
    for r in csv.DictReader(open(OUT2 / "covariates_wide.csv")):
        d = {"pop_density": float(r["pop_density_per_sq_mi"])}
        for k in ("renter_share", "med_hh_income", "population",
                  "built_2010_plus", "nhwhite_share", "transit_commute",
                  "vacancy_rate"):
            d[k] = float(r[k])
        for lk, raw in LOGCOLS.items():
            d[lk] = float(np.log(d[raw]))
        out[r["body"]] = d
    return out


def cluster_self_id():
    db = sqlite3.connect(DB, uri=True)
    turns = collections.defaultdict(list)
    for ep, cl, sid in db.execute("select episode_id, cluster, self_id "
                                  "from role_audit"):
        turns[(ep, cl)].append(sid)
    db.close()
    out = {}
    for k, sids in turns.items():
        c = collections.Counter(s for s in sids if s and s != "unknown")
        if not c:
            out[k] = "unknown"
            continue
        top = max(c.values())
        out[k] = sorted([s for s, v in c.items() if v == top],
                        key=lambda s: SELF_PRI.get(s, 8))[0]
    return out


def clusters():
    cself = cluster_self_id()
    rows = []
    for r in csv.DictReader(open(CLUSTERS)):
        if r["label"] == "no stance":
            continue
        r["self_id"] = cself.get((r["episode_id"], r["cluster"]), "unknown")
        rows.append(r)
    return rows


def build_rows(variant, cov):
    """body x route x contrast-group rows on the directional denominator."""
    cl = clusters()
    if variant == "DIRECTIONAL":
        keep, group = cl, (lambda r: r["role"])      # dais / public
    elif variant == "DIR_NOSTAFF":
        keep = [r for r in cl if r["self_id"] != "staff"]
        group = lambda r: r["role"]
    elif variant == "ELECT_RESID":
        keep = [r for r in cl if r["self_id"] in ("elected", "resident")]
        group = lambda r: ("dais" if r["self_id"] == "elected" else "public")
    else:
        raise ValueError(variant)
    cnt = collections.defaultdict(lambda: dict(NIMBY=0, YIMBY=0, neutral=0,
                                               episodes=set()))
    for r in keep:
        if r["role"] not in ("dais", "public"):
            continue
        c = cnt[(r["method"], r["body"], group(r))]
        c[r["label"]] = c.get(r["label"], 0) + 1
        c["episodes"].add(r["episode_id"])
    rows, below = [], []
    for (route, body, grp), c in sorted(cnt.items()):
        if body not in cov:
            continue
        nd = c["NIMBY"] + c["YIMBY"]
        rec = dict(route=route, body=body, group=grp,
                   role_code=0 if grp == "dais" else 1,
                   n=nd, nimby=c["NIMBY"], yimby=c["YIMBY"],
                   neutral=c["neutral"], episodes=len(c["episodes"]),
                   **cov[body])
        if nd < MIN_N:
            below.append(dict(route=route, body=body, group=grp, n=nd))
            continue
        rows.append(rec)
    bycell = collections.defaultdict(set)
    for r in rows:
        bycell[(r["route"], r["body"])].add(r["group"])
    paired = {k for k, v in bycell.items() if len(v) == 2}
    for r in rows:
        r["paired_cell"] = int((r["route"], r["body"]) in paired)
    return rows, below, len(paired)


def psum(d):
    d = np.asarray(d).ravel()
    return dict(median=float(np.median(d)), mean=float(d.mean()),
                sd=float(d.std(ddof=1)),
                ci90=[float(np.percentile(d, 5)), float(np.percentile(d, 95))],
                ci95=[float(np.percentile(d, 2.5)),
                      float(np.percentile(d, 97.5))],
                pr_gt_0=float((d > 0).mean()),
                excl90=bool(np.percentile(d, 5) > 0 or np.percentile(d, 95) < 0),
                excl95=bool(np.percentile(d, 2.5) > 0
                            or np.percentile(d, 97.5) < 0))


def fit(rows, cols, label, variant):
    y = np.array([r["nimby"] for r in rows], int)
    N = np.array([r["n"] for r in rows], int)
    role = np.array([r["role_code"] for r in rows], float)
    X = np.column_stack([[r[c] for r in rows] for c in cols]).astype(float)
    mu, sd = X.mean(0), X.std(0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    jr = cols.index("renter_share")
    ix_r = role * Z[:, jr]
    routes = sorted({r["route"] for r in rows},
                    key=lambda s: (s != ROUTE_REF, s))
    ridx = np.array([routes.index(r["route"]) for r in rows])
    bodies = sorted({r["body"] for r in rows})
    bidx = np.array([bodies.index(r["body"]) for r in rows])
    n, p = len(rows), len(cols)

    with pm.Model():
        alpha = pm.Normal("alpha", 0, 1.5)
        beta = pm.Normal("beta", 0, 1.0, shape=p)
        delta = pm.Normal("delta_role", 0, 2.0)
        theta_r = pm.Normal("theta_role_x_renter", 0, 1.0)
        eta = alpha + pt.dot(Z, beta) + delta * role + theta_r * ix_r
        sigma_body = pm.HalfNormal("sigma_body", 0.75)
        b_raw = pm.Normal("b_body_raw", 0, 1, shape=len(bodies))
        b_body = pm.Deterministic("b_body", b_raw * sigma_body)
        sigma_row = pm.HalfNormal("sigma_row", 0.75)
        u_raw = pm.Normal("u_raw", 0, 1, shape=n)
        u = pm.Deterministic("u", u_raw * sigma_row)
        eta = eta + b_body[bidx] + u
        if len(routes) > 1:
            g_raw = pm.Normal("gamma_route_raw", 0, 1.0, shape=len(routes) - 1)
            gamma = pm.Deterministic("gamma_route",
                                     pt.concatenate([[0.0], g_raw]))
            eta = eta + gamma[ridx]
        pm.Deterministic("odds_ratio_role", pt.exp(delta))
        pm.Deterministic("slope_renter_dais", beta[jr])
        pm.Deterministic("slope_renter_public", beta[jr] + theta_r)
        pm.Binomial("y", n=N, p=pm.math.sigmoid(eta), observed=y)
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS,
                          target_accept=0.95, random_seed=SEED,
                          progressbar=False)
    post = idata.posterior
    params = {c: psum(np.asarray(post["beta"])[..., j])
              for j, c in enumerate(cols)}
    for nm in ("alpha", "delta_role", "odds_ratio_role",
               "theta_role_x_renter", "sigma_body", "sigma_row",
               "slope_renter_dais", "slope_renter_public"):
        params[nm] = psum(np.asarray(post[nm]))
    if "gamma_route" in post.data_vars:
        gm = np.asarray(post["gamma_route"])
        for i, rt in enumerate(routes):
            if i:
                params[f"gamma[{rt}]"] = psum(gm[..., i])
    skip = {"b_body", "u", "gamma_route", "odds_ratio_role",
            "slope_renter_dais", "slope_renter_public"}
    names = [v for v in post.data_vars if v not in skip]
    rhat, ess = az.rhat(idata, var_names=names), az.ess(idata, var_names=names)
    diag = dict(
        max_r_hat=float(max(float(np.nanmax(rhat[v].to_numpy())) for v in names)),
        min_ess_bulk=float(min(float(np.nanmin(ess[v].to_numpy())) for v in names)),
        divergences=int(idata.sample_stats["diverging"].sum()))
    d0 = np.array([r for r in rows if r["role_code"] == 0])
    raw = dict(
        dais_nimby=sum(r["nimby"] for r in rows if r["role_code"] == 0),
        dais_n=sum(r["n"] for r in rows if r["role_code"] == 0),
        pub_nimby=sum(r["nimby"] for r in rows if r["role_code"] == 1),
        pub_n=sum(r["n"] for r in rows if r["role_code"] == 1))
    raw["dais_share"] = raw["dais_nimby"] / raw["dais_n"]
    raw["pub_share"] = raw["pub_nimby"] / raw["pub_n"]
    return dict(variant=variant, label=label, covariates=cols, n_rows=n,
                n_bodies=len(bodies), n_dir_clusters=int(N.sum()),
                n_dais_rows=int((role == 0).sum()),
                n_public_rows=int((role == 1).sum()),
                routes=routes, cov_sd=dict(zip(cols, map(float, sd))),
                raw=raw, params=params, diagnostics=diag)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cols = json.loads((OUT2 / "screen.json").read_text())["kept"]
    cov = covariates()
    print(STAMP)
    print(f"covariates (run-2 VIF screen, unchanged): {cols}\n")
    results = []
    for variant, label in (
            ("DIRECTIONAL", "(a) NIMBY / (NIMBY+YIMBY), neutral excluded"),
            ("DIR_NOSTAFF", "(b) (a) with staff clusters excluded"),
            ("ELECT_RESID", "(c) elected (0) vs resident (1), directional")):
        rows, below, npair = build_rows(variant, cov)
        with open(OUT / f"rows_{variant}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        r = fit(rows, cols, label, variant)
        r["n_below_floor"] = len(below)
        r["below_floor"] = below
        r["n_paired_cells"] = npair
        results.append(r)
        P, D = r["params"], r["diagnostics"]
        print(f"{variant}: {label}")
        print(f"  rows {r['n_rows']} ({r['n_dais_rows']} grp0 / "
              f"{r['n_public_rows']} grp1), bodies {r['n_bodies']}, "
              f"paired cells {npair}, directional clusters "
              f"{r['n_dir_clusters']}, below floor {len(below)}")
        print(f"  raw NIMBY share grp0 {r['raw']['dais_share']:.3f} "
              f"({r['raw']['dais_nimby']}/{r['raw']['dais_n']})  grp1 "
              f"{r['raw']['pub_share']:.3f} "
              f"({r['raw']['pub_nimby']}/{r['raw']['pub_n']})")
        print(f"  delta  {P['delta_role']['median']:+.3f}  "
              f"90% [{P['delta_role']['ci90'][0]:+.3f}, "
              f"{P['delta_role']['ci90'][1]:+.3f}]  "
              f"95% [{P['delta_role']['ci95'][0]:+.3f}, "
              f"{P['delta_role']['ci95'][1]:+.3f}]  "
              f"Pr>0 {P['delta_role']['pr_gt_0']:.3f}")
        o = P["odds_ratio_role"]
        print(f"  OR     {o['median']:.2f}  90% [{o['ci90'][0]:.2f}, "
              f"{o['ci90'][1]:.2f}]  95% [{o['ci95'][0]:.2f}, "
              f"{o['ci95'][1]:.2f}]")
        print(f"  theta_role_x_renter {P['theta_role_x_renter']['median']:+.3f} "
              f"90% [{P['theta_role_x_renter']['ci90'][0]:+.3f}, "
              f"{P['theta_role_x_renter']['ci90'][1]:+.3f}]")
        print(f"  renter_share {P['renter_share']['median']:+.3f} "
              f"90% [{P['renter_share']['ci90'][0]:+.3f}, "
              f"{P['renter_share']['ci90'][1]:+.3f}]")
        print(f"  R-hat {D['max_r_hat']:.4f} ESS {D['min_ess_bulk']:.0f} "
              f"div {D['divergences']} sigma_body "
              f"{P['sigma_body']['median']:.3f} sigma_row "
              f"{P['sigma_row']['median']:.3f}\n")

    with open(OUT / "delta_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "label", "n_rows", "n_bodies",
                    "n_dir_clusters", "raw_grp0_share", "raw_grp1_share",
                    "delta_median", "ci90_lo", "ci90_hi", "ci95_lo", "ci95_hi",
                    "pr_gt_0", "or_median", "or_ci90_lo", "or_ci90_hi",
                    "or_ci95_lo", "or_ci95_hi", "max_r_hat", "divergences"])
        for r in results:
            d, o = r["params"]["delta_role"], r["params"]["odds_ratio_role"]
            w.writerow([r["variant"], r["label"], r["n_rows"], r["n_bodies"],
                        r["n_dir_clusters"], f"{r['raw']['dais_share']:.4f}",
                        f"{r['raw']['pub_share']:.4f}",
                        f"{d['median']:+.4f}", f"{d['ci90'][0]:+.4f}",
                        f"{d['ci90'][1]:+.4f}", f"{d['ci95'][0]:+.4f}",
                        f"{d['ci95'][1]:+.4f}", f"{d['pr_gt_0']:.3f}",
                        f"{o['median']:.3f}", f"{o['ci90'][0]:.3f}",
                        f"{o['ci90'][1]:.3f}", f"{o['ci95'][0]:.3f}",
                        f"{o['ci95'][1]:.3f}",
                        f"{r['diagnostics']['max_r_hat']:.4f}",
                        r["diagnostics"]["divergences"]])

    # every parameter of every restated fit, for the "print every parameter" rule
    with open(OUT / "param_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "parameter", "median", "ci90_lo", "ci90_hi",
                    "ci95_lo", "ci95_hi", "pr_gt_0", "excl90", "excl95"])
        for r in results:
            for nm, v in r["params"].items():
                w.writerow([r["variant"], nm, f"{v['median']:+.4f}",
                            f"{v['ci90'][0]:+.4f}", f"{v['ci90'][1]:+.4f}",
                            f"{v['ci95'][0]:+.4f}", f"{v['ci95'][1]:+.4f}",
                            f"{v['pr_gt_0']:.3f}", int(v["excl90"]),
                            int(v["excl95"])])

    (OUT / "results.json").write_text(json.dumps(dict(
        stamp=STAMP, seed=SEED, draws=DRAWS, tune=TUNE, chains=CHAINS,
        min_n=MIN_N, route_reference=ROUTE_REF, covariates=cols,
        self_id_priority=SELF_PRI, results=results), indent=2, default=str))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
