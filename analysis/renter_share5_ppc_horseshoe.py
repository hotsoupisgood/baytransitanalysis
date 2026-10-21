#!/usr/bin/env python3
"""
renter_share5_ppc_horseshoe.py -- findings 3 and 5 of
plan/REGRESSION-REVIEW-2026-10-19.md.

FINDING 3 (the PPC cannot fail). Runs 2 and 3 report "29/29", "35/35",
"64/64" rows inside a 90 % posterior-predictive band. `u_i` is one free
parameter per row, so the conditional predictive is centred on the observed
count and the check has no power: under a calibrated 90 % band
P(64/64) = 0.9^64 = 0.0013, so 64/64 is itself evidence of over-fitting.
This file replaces that count with two checks that the overdispersion term
cannot absorb:

  MARGINAL PPC  u_i is INTEGRATED OUT -- for every posterior draw a fresh
                u* ~ Normal(0, sigma_row) is drawn instead of using the fitted
                u_i, so the band is a genuine out-of-sample band for a new row
                of the same body and route.
  LOO-PIT-like  the marginal predictive PIT value of each observed count,
                P(y* < y) + 0.5 P(y* = y), with a Kolmogorov-Smirnov statistic
                against Uniform(0,1). A calibrated model gives ~uniform PIT;
                u_i-conditional PIT piles up at 0.5.

The body random intercept is NOT integrated out: it has 22-25 levels and is a
real grouping, not one parameter per row.

FINDING 5 (the horseshoe result is the prior). tau0 = p0/(p-p0) * sigma/sqrt(n)
is fixed by construction at 0.124 for p0 = 2, p = 8, sigma = 2, n = 29, so the
reported tau of 0.03-0.10 sits in the PRIOR bulk and is not the posterior
"saying there is nothing here". Three comparisons are run on run 2's spec (c):

  HS_TAU0      the published prior, tau0 = 0.124
  HS_PERM      the same fit with the eight covariate vectors PERMUTED ACROSS
               BODIES (a true null) -- if the published picture reappears, the
               picture is the prior
  HS_TAU0x5    tau0 * 5
  HS_TAU0d5    tau0 / 5

Writes output/renter_share_5/ppc_horseshoe/. No network, no spend, nothing
committed.

    nohup .venv/bin/python analysis/renter_share5_ppc_horseshoe.py \
        > output/renter_share_5/ppc_horseshoe.log 2>&1 &
"""
import csv
import json
import pathlib
import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT2 = ROOT / "output" / "renter_share_2"
OUT3 = ROOT / "output" / "renter_share_3"
OUT = ROOT / "output" / "renter_share_5" / "ppc_horseshoe"
SEED = 20261019
DRAWS, TUNE, CHAINS = 2000, 1500, 4
HS_TUNE = 4000
ROUTE_REF = "classifier"
KEPT8 = ["renter_share", "log_income", "log_density", "built_2010_plus",
         "nhwhite_share", "transit_commute", "vacancy_rate", "log_population"]
P0, SIGMA_PSEUDO = 2.0, 2.0


def load_run2():
    rows = []
    for r in csv.DictReader(open(OUT2 / "outcomes_clusters.csv")):
        d = dict(route=r["route"], body=r["body"])
        for c in KEPT8:
            d[c] = float(r[c])
        for c in ("pub_n", "pub_nimby", "pub_yimby", "all_n", "all_nimby",
                  "all_yimby"):
            d[c] = int(r[c])
        rows.append(d)
    return rows


def load_run3():
    rows = []
    for r in csv.DictReader(open(OUT3 / "outcomes_role_stacked.csv")):
        d = dict(route=r["route"], body=r["body"], role=r["role"],
                 role_code=int(r["role_code"]))
        for c in KEPT8:
            d[c] = float(r[c])
        for c in ("n", "nimby", "yimby"):
            d[c] = int(r[c])
        rows.append(d)
    return rows


def std(rows, cols, permute_bodies=False, seed=SEED):
    X = np.column_stack([[r[c] for r in rows] for c in cols]).astype(float)
    if permute_bodies:
        bodies = sorted({r["body"] for r in rows})
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(bodies))
        mapping = {bodies[i]: bodies[perm[i]] for i in range(len(bodies))}
        first = {}
        for i, r in enumerate(rows):
            first.setdefault(r["body"], X[i].copy())
        X = np.array([first[mapping[r["body"]]] for r in rows])
    mu, sd = X.mean(0), X.std(0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def marginal_checks(y, N, eta_nou, sigma_row, rng):
    """Posterior predictive with u_i integrated out.

    eta_nou: (ndraw, n) linear predictor WITHOUT u_i.
    sigma_row: (ndraw,) draws of sigma_row.
    """
    nd, n = eta_nou.shape
    u = rng.normal(0.0, 1.0, size=(nd, n)) * sigma_row[:, None]
    p = 1.0 / (1.0 + np.exp(-np.clip(eta_nou + u, -30, 30)))
    rep = rng.binomial(np.broadcast_to(N, (nd, n)), p)
    lo = np.percentile(rep, 5, axis=0)
    hi = np.percentile(rep, 95, axis=0)
    inside = int(((y >= lo) & (y <= hi)).sum())
    pit = ((rep < y).mean(0) + 0.5 * (rep == y).mean(0))
    ks = stats.kstest(pit, "uniform")
    # cell-mean check: observed share against the marginal predictive share
    obs_sh = y / N
    rep_sh = rep / N
    z = (obs_sh - rep_sh.mean(0)) / np.maximum(rep_sh.std(0, ddof=1), 1e-9)
    return dict(rows=n, inside_90=inside,
                inside_90_frac=round(inside / n, 3),
                expected_inside=round(0.9 * n, 1),
                binom_p_all_inside=float(0.9 ** n),
                pit_ks_stat=float(ks.statistic), pit_ks_p=float(ks.pvalue),
                pit_mean=float(pit.mean()), pit_sd=float(pit.std(ddof=1)),
                pit_q=[float(q) for q in np.percentile(pit, [5, 25, 50, 75, 95])],
                max_abs_cellmean_z=float(np.abs(z).max()),
                n_cellmean_z_gt_2=int((np.abs(z) > 2).sum()),
                pit=[round(float(v), 4) for v in pit])


def fit_run3(rows, ycol, label, with_income_ix=True):
    y = np.array([r[ycol] for r in rows], int)
    N = np.array([r["n"] for r in rows], int)
    role = np.array([r["role_code"] for r in rows], float)
    Z, mu, sd = std(rows, KEPT8)
    jr, ji = KEPT8.index("renter_share"), KEPT8.index("log_income")
    routes = sorted({r["route"] for r in rows},
                    key=lambda s: (s != ROUTE_REF, s))
    ridx = np.array([routes.index(r["route"]) for r in rows])
    bodies = sorted({r["body"] for r in rows})
    bidx = np.array([bodies.index(r["body"]) for r in rows])
    n = len(rows)
    with pm.Model():
        alpha = pm.Normal("alpha", 0, 1.5)
        beta = pm.Normal("beta", 0, 1.0, shape=len(KEPT8))
        delta = pm.Normal("delta_role", 0, 2.0)
        th = pm.Normal("theta_role_x_renter", 0, 1.0)
        eta = (alpha + pt.dot(Z, beta) + delta * role
               + th * (role * Z[:, jr]))
        if with_income_ix:
            thy = pm.Normal("theta_role_x_income", 0, 1.0)
            eta = eta + thy * (role * Z[:, ji])
        sb = pm.HalfNormal("sigma_body", 0.75)
        b = pm.Deterministic("b_body",
                             pm.Normal("b_raw", 0, 1, shape=len(bodies)) * sb)
        sr = pm.HalfNormal("sigma_row", 0.75)
        u = pm.Deterministic("u", pm.Normal("u_raw", 0, 1, shape=n) * sr)
        g = pm.Deterministic("gamma_route", pt.concatenate(
            [[0.0], pm.Normal("g_raw", 0, 1.0, shape=len(routes) - 1)]))
        pm.Deterministic("eta_nou", eta + b[bidx] + g[ridx])
        pm.Binomial("y", n=N, p=pm.math.sigmoid(eta + b[bidx] + u + g[ridx]),
                    observed=y)
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS,
                          target_accept=0.95, random_seed=SEED,
                          progressbar=False)
    post = idata.posterior
    eta_nou = np.asarray(post["eta_nou"]).reshape(-1, n)
    sr_d = np.asarray(post["sigma_row"]).ravel()
    rng = np.random.default_rng(SEED)
    chk = marginal_checks(y, N, eta_nou, sr_d, rng)
    # the conditional (published) band, for the side-by-side
    p_cond = 1 / (1 + np.exp(-np.clip(
        eta_nou + np.asarray(post["u"]).reshape(-1, n), -30, 30)))
    rep_c = rng.binomial(np.broadcast_to(N, p_cond.shape), p_cond)
    lo, hi = np.percentile(rep_c, 5, axis=0), np.percentile(rep_c, 95, axis=0)
    chk["conditional_inside_90"] = int(((y >= lo) & (y <= hi)).sum())
    pit_c = (rep_c < y).mean(0) + 0.5 * (rep_c == y).mean(0)
    chk["conditional_pit_ks_stat"] = float(stats.kstest(pit_c, "uniform").statistic)
    chk["conditional_pit_sd"] = float(pit_c.std(ddof=1))
    chk.update(run="run3", outcome=ycol, spec=label, sigma_row_median=
               float(np.median(sr_d)))
    return chk


def fit_run2(rows, ycol, ncol, cols, label, outcome_name):
    y = np.array([r[ycol] for r in rows], int)
    N = np.array([r[ncol] for r in rows], int)
    Z, mu, sd = std(rows, cols)
    routes = sorted({r["route"] for r in rows},
                    key=lambda s: (s != ROUTE_REF, s))
    ridx = np.array([routes.index(r["route"]) for r in rows])
    n = len(rows)
    with pm.Model():
        alpha = pm.Normal("alpha", 0, 1.5)
        beta = pm.Normal("beta", 0, 1.0, shape=len(cols))
        sc = pm.HalfNormal("sigma_row", 0.75)
        u = pm.Deterministic("u", pm.Normal("u_raw", 0, 1, shape=n) * sc)
        g = pm.Deterministic("gamma_route", pt.concatenate(
            [[0.0], pm.Normal("g_raw", 0, 1.0, shape=len(routes) - 1)]))
        eta = alpha + pt.dot(Z, beta) + g[ridx]
        pm.Deterministic("eta_nou", eta)
        pm.Binomial("y", n=N, p=pm.math.sigmoid(eta + u), observed=y)
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS,
                          target_accept=0.95, random_seed=SEED,
                          progressbar=False)
    post = idata.posterior
    eta_nou = np.asarray(post["eta_nou"]).reshape(-1, n)
    sr_d = np.asarray(post["sigma_row"]).ravel()
    rng = np.random.default_rng(SEED)
    chk = marginal_checks(y, N, eta_nou, sr_d, rng)
    p_cond = 1 / (1 + np.exp(-np.clip(
        eta_nou + np.asarray(post["u"]).reshape(-1, n), -30, 30)))
    rep_c = rng.binomial(np.broadcast_to(N, p_cond.shape), p_cond)
    lo, hi = np.percentile(rep_c, 5, axis=0), np.percentile(rep_c, 95, axis=0)
    chk["conditional_inside_90"] = int(((y >= lo) & (y <= hi)).sum())
    pit_c = (rep_c < y).mean(0) + 0.5 * (rep_c == y).mean(0)
    chk["conditional_pit_ks_stat"] = float(stats.kstest(pit_c, "uniform").statistic)
    chk["conditional_pit_sd"] = float(pit_c.std(ddof=1))
    chk.update(run="run2", outcome=outcome_name, spec=label,
               sigma_row_median=float(np.median(sr_d)))
    return chk


# --------------------------------------------------------------- horseshoe
def horseshoe(rows, ycol, ncol, tau_mult=1.0, permute=False, tag=""):
    y = np.array([r[ycol] for r in rows], int)
    N = np.array([r[ncol] for r in rows], int)
    Z, mu, sd = std(rows, KEPT8, permute_bodies=permute)
    routes = sorted({r["route"] for r in rows},
                    key=lambda s: (s != ROUTE_REF, s))
    ridx = np.array([routes.index(r["route"]) for r in rows])
    n, p = len(rows), len(KEPT8)
    tau0 = (P0 / (p - P0)) * SIGMA_PSEUDO / np.sqrt(n)
    with pm.Model():
        alpha = pm.Normal("alpha", 0, 1.5)
        tau = pm.HalfCauchy("tau", tau0 * tau_mult)
        lam = pm.HalfCauchy("lam", 1.0, shape=p)
        c2 = pm.InverseGamma("c2", 2.0, 2.0 * SIGMA_PSEUDO ** 2)
        lam_t = pt.sqrt(c2 * lam ** 2 / (c2 + tau ** 2 * lam ** 2))
        z = pm.Normal("z", 0, 1, shape=p)
        beta = pm.Deterministic("beta", z * tau * lam_t)
        sc = pm.HalfNormal("sigma_cell", 0.75)
        u = pm.Normal("u_raw", 0, 1, shape=n) * sc
        g = pt.concatenate([[0.0],
                            pm.Normal("g_raw", 0, 1.0, shape=len(routes) - 1)])
        pm.Binomial("y", n=N,
                    p=pm.math.sigmoid(alpha + pt.dot(Z, beta) + g[ridx] + u),
                    observed=y)
        idata = pm.sample(draws=DRAWS, tune=HS_TUNE, chains=CHAINS,
                          target_accept=0.99, random_seed=SEED,
                          progressbar=False)
    post = idata.posterior
    bd = np.asarray(post["beta"]).reshape(-1, p)
    td = np.asarray(post["tau"]).ravel()
    out = dict(tag=tag, tau0=float(tau0), tau0_multiplier=tau_mult,
               tau0_used=float(tau0 * tau_mult), permuted=bool(permute),
               n_rows=n, tau_median=float(np.median(td)),
               tau_ci90=[float(np.percentile(td, 5)),
                         float(np.percentile(td, 95))],
               divergences=int(idata.sample_stats["diverging"].sum()),
               sigma_cell_median=float(np.median(
                   np.asarray(post["sigma_cell"]))))
    for j, c in enumerate(KEPT8):
        d = bd[:, j]
        out[c] = float(np.median(d))
        out[c + "_ci90"] = [float(np.percentile(d, 5)),
                            float(np.percentile(d, 95))]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r2, r3 = load_run2(), load_run3()
    checks = []

    print("== finding 3: PPC with u_i integrated out")
    for oname, ycol, ncol in (("public NIMBY", "pub_nimby", "pub_n"),
                              ("public YIMBY", "pub_yimby", "pub_n"),
                              ("all-roles NIMBY", "all_nimby", "all_n"),
                              ("all-roles YIMBY", "all_yimby", "all_n")):
        rows = [r for r in r2 if r[ncol] >= 5]
        for lab, cols in (("(a) base", ["renter_share", "log_income"]),
                          ("(c) all eight, Normal(0,1)", KEPT8)):
            c = fit_run2(rows, ycol, ncol, cols, lab, oname)
            checks.append(c)
            print(f"  run2 {oname:16s} {lab:26s} conditional "
                  f"{c['conditional_inside_90']}/{c['rows']} -> marginal "
                  f"{c['inside_90']}/{c['rows']} (expect "
                  f"{c['expected_inside']}) PIT KS {c['pit_ks_stat']:.3f} "
                  f"p {c['pit_ks_p']:.3f} cellmean |z|>2 "
                  f"{c['n_cellmean_z_gt_2']}")
    for ycol, lab in (("nimby", "R2"), ("yimby", "R2")):
        c = fit_run3(r3, ycol, lab)
        checks.append(c)
        print(f"  run3 {ycol:16s} {lab:26s} conditional "
              f"{c['conditional_inside_90']}/{c['rows']} -> marginal "
              f"{c['inside_90']}/{c['rows']} (expect {c['expected_inside']}) "
              f"PIT KS {c['pit_ks_stat']:.3f} p {c['pit_ks_p']:.3f} "
              f"cellmean |z|>2 {c['n_cellmean_z_gt_2']}")

    with open(OUT / "ppc_marginal.csv", "w", newline="") as fh:
        keys = [k for k in checks[0] if k != "pit"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for c in checks:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v)
                        for k, v in c.items() if k != "pit"})

    print("\n== finding 5: the horseshoe against its own prior")
    pub = [r for r in r2 if r["pub_n"] >= 5]
    hs = []
    for tag, mult, perm in (("HS_TAU0 (published)", 1.0, False),
                            ("HS_PERM (covariates permuted across bodies)",
                             1.0, True),
                            ("HS_TAU0 x5", 5.0, False),
                            ("HS_TAU0 /5", 0.2, False)):
        h = horseshoe(pub, "pub_yimby", "pub_n", mult, perm, tag)
        hs.append(h)
        print(f"  {tag:46s} tau0 {h['tau0_used']:.4f} tau "
              f"{h['tau_median']:.4f} renter {h['renter_share']:+.4f} "
              f"transit {h['transit_commute']:+.4f} div {h['divergences']}")

    with open(OUT / "horseshoe.csv", "w", newline="") as fh:
        keys = [k for k in hs[0] if not k.endswith("_ci90")]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(hs)

    (OUT / "results.json").write_text(json.dumps(dict(
        seed=SEED, draws=DRAWS, tune=TUNE, hs_tune=HS_TUNE, chains=CHAINS,
        tau0_formula="p0/(p-p0) * sigma/sqrt(n), p0=2, p=8, sigma=2",
        ppc_note=("u_i is integrated out: for every posterior draw a fresh "
                  "u* ~ Normal(0, sigma_row) replaces the fitted u_i, so the "
                  "band is out-of-sample for a new row of the same body and "
                  "route. The body random intercept is NOT integrated out."),
        ppc=checks, horseshoe=hs), indent=2, default=str))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
