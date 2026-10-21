#!/usr/bin/env python3
"""
renter_share3_role_models.py -- the role-interaction regression for
plan/RENTER-SHARE-3-ROLE.md.

WHY THIS RUN EXISTS. plan/RENTER-SHARE-2.md never fits dais/public as a
covariate: it fits a public-only outcome and an all-roles outcome, so role is a
subsetting variable and the renter-share slope is estimated once per outcome
family with no way to ask whether it differs by who is speaking. This file
stacks the two role rows of every body x route cell
(analysis/renter_share3_role_outcomes.py) and puts role IN the design matrix
together with role x renter_share, which is the fit that answers the question.

UNIT. body x route x role row, floor 5 clusters with a stance per row.
64 rows / 35 cells / 25 bodies; 29 of the 35 cells have both role rows.

MODEL (per outcome, NIMBY and YIMBY):

    y_i        ~ Binomial(N_i, p_i)
    logit p_i  = alpha + sum_j beta_j z_ji              # 8 kept covariates
                 + delta * role_i                       # dais=0, public=1
                 + theta_r * role_i * z_renter,i        # the interaction
                [+ theta_y * role_i * z_income,i]       # spec R2 only
                 + gamma_route(i)                       # stratum, never pooled
                 + b_body(i)                            # shared by a body's
                                                        # two role rows
                 + u_i                                  # row overdispersion

    b_body ~ Normal(0, sigma_body)   sigma_body ~ HalfNormal(0.75)
    u_i    ~ Normal(0, sigma_row)    sigma_row  ~ HalfNormal(0.75)
    alpha  ~ Normal(0, 1.5)   gamma ~ Normal(0, 1)
    beta_j ~ Normal(0, 1)     theta ~ Normal(0, 1)
    delta  ~ Normal(0, 2)     # the role gap is a priori allowed to be large;
                              # Normal(0,1) would shrink a real +2 log-odds gap

Covariates are the eight kept by run 2's VIF screen (output/renter_share_2/
screen.json), standardised over the fitted rows. Because covariates vary at the
BODY level and are duplicated across a body's role rows, standardisation
weights a body by its row count; cov_mean/cov_sd are recorded per fit so any
coefficient can be put back on the covariate's own scale.

IDENTIFICATION. role and role x covariate are identified within body by the 29
paired cells; b_body removes the between-body level, so delta and theta are
within-body contrasts for those cells. Spec R2 is re-fitted on the paired rows
only as a sensitivity check -- if the 6 dais-only rows (bart, milpitas,
vallejo, fullasr napa, fullasr hayward, local_asr hayward) were driving the
role terms, that fit would move.

REPORTED. Posterior median, 90 % and 95 % equal-tailed intervals and
Pr(beta > 0) for every parameter; the two IMPLIED renter-share slopes
(dais = beta_renter, public = beta_renter + theta_r) as posterior derived
quantities with their own intervals; max R-hat, min bulk ESS, divergences; a
PPC count of rows whose observed count sits inside the replicated 90 % band.
PSIS-LOO is deliberately NOT reported: plan/RENTER-SHARE-2.md 3d showed it is
degenerate once a per-observation overdispersion term is in the model, and this
model has two such terms.

Frequentist sanity check on the same stacked rows: binomial logistic IRLS with
(i) an HC3-type sandwich and (ii) a by-body cluster-robust sandwich, both
implemented here (statsmodels is not installed in this venv).

Writes output/renter_share_3/models/{model_results.json, model_table.csv,
slopes.csv, sanity_hc3.csv}. No figures, no DB write, no spend, nothing
committed.

    nohup .venv/bin/python analysis/renter_share3_role_models.py \
        > output/renter_share_3/models.log 2>&1 &
"""
import csv
import json
import pathlib
import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT2 = ROOT / "output" / "renter_share_2"
OUT = ROOT / "output" / "renter_share_3"
MOUT = OUT / "models"
STAMP = ("EXPLORATORY — stance κ ≥ 0.6 hand-coding gate not passed "
         "(plan/SHOWCASE.md 4a); post-S2 cluster unit, role_audit roles "
         "(rule ra-1), role is an AUDIT not a gold label")
SEED = 20261019
DRAWS, TUNE, CHAINS = 2000, 1500, 4
ROUTE_REF = "classifier"

OUTCOMES = [("nimby", "NIMBY share"), ("yimby", "YIMBY share")]


def load(paired_only=False):
    rows = []
    for r in csv.DictReader(open(OUT / "outcomes_role_stacked.csv")):
        if paired_only and int(r["paired_cell"]) != 1:
            continue
        d = dict(route=r["route"], body=r["body"], role=r["role"],
                 role_code=int(r["role_code"]), paired=int(r["paired_cell"]))
        for k in ("n", "nimby", "yimby"):
            d[k] = int(r[k])
        for k in ("renter_share", "log_income", "log_density",
                  "built_2010_plus", "nhwhite_share", "transit_commute",
                  "vacancy_rate", "log_population"):
            d[k] = float(r[k])
        rows.append(d)
    return rows


def design(rows, cols):
    X = np.column_stack([[r[c] for r in rows] for c in cols]).astype(float)
    mu, sd = X.mean(0), X.std(0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def psum(d):
    d = np.asarray(d).ravel()
    return dict(mean=float(d.mean()), median=float(np.median(d)),
                sd=float(d.std(ddof=1)),
                ci90=[float(np.percentile(d, 5)), float(np.percentile(d, 95))],
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                pr_gt_0=float((d > 0).mean()),
                excludes_zero_90=bool(np.percentile(d, 5) > 0
                                      or np.percentile(d, 95) < 0),
                excludes_zero_95=bool(np.percentile(d, 2.5) > 0
                                      or np.percentile(d, 97.5) < 0))


def diag(idata):
    skip = {"mu", "p", "eta", "gamma_route", "b_body", "u"}
    names = [v for v in idata.posterior.data_vars if v not in skip]
    rhat, ess = az.rhat(idata, var_names=names), az.ess(idata, var_names=names)
    return dict(
        max_r_hat=float(max(float(np.nanmax(rhat[v].to_numpy())) for v in names)),
        min_ess_bulk=float(min(float(np.nanmin(ess[v].to_numpy())) for v in names)),
        divergences=(int(idata.sample_stats["diverging"].sum())
                     if "diverging" in idata.sample_stats else -1),
        diag_params=names)


# ------------------------------------------------- frequentist sanity checks
def logit_sandwich(X, ycount, N, groups, maxit=300, tol=1e-11):
    """Binomial logistic IRLS with two robust covariance estimates.

    X already contains every column including the intercept. Score of cell i is
    x_i * s_i with s_i = y_i - N_i p_i, bread A = X' diag(N p (1-p)) X.
      HC3:      meat = X' diag(s_i^2 / (1 - h_i)^2) X, h_i the IRLS leverage.
      cluster:  meat = sum_g (X_g' s_g)(X_g' s_g)', with the usual
                G/(G-1) * (n-1)/(n-k) finite-sample correction.
    Returns (beta, se_hc3, se_cluster, converged, n_groups).
    """
    X = np.asarray(X, float)
    ycount, N = np.asarray(ycount, float), np.asarray(N, float)
    n, k = X.shape
    beta, ok = np.zeros(k), False
    for _ in range(maxit):
        eta = X @ beta
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = np.maximum(N * p * (1 - p), 1e-9)
        z = eta + (ycount - N * p) / W
        new = np.linalg.pinv((X.T * W) @ X) @ ((X.T * W) @ z)
        if np.max(np.abs(new - beta)) < tol:
            beta, ok = new, True
            break
        beta = new
    eta = X @ beta
    p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
    W = np.maximum(N * p * (1 - p), 1e-9)
    Ai = np.linalg.pinv((X.T * W) @ X)
    Xw = X * np.sqrt(W)[:, None]
    h = np.einsum("ij,jk,ik->i", Xw, np.linalg.pinv(Xw.T @ Xw), Xw)
    s = ycount - N * p
    meat_hc3 = (X.T * (s ** 2 / np.maximum((1 - h) ** 2, 1e-12))) @ X
    V3 = Ai @ meat_hc3 @ Ai
    gs = sorted(set(groups))
    meat_cl = np.zeros((k, k))
    for g in gs:
        m = np.array([x == g for x in groups])
        sg = X[m].T @ s[m]
        meat_cl += np.outer(sg, sg)
    G = len(gs)
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    Vc = corr * (Ai @ meat_cl @ Ai)
    return (beta, np.sqrt(np.maximum(np.diag(V3), 0)),
            np.sqrt(np.maximum(np.diag(Vc), 0)), bool(ok), G)


# ----------------------------------------------------------------- the fit
def fit(rows, ycol, cols, with_income_ix, label):
    y = np.array([r[ycol] for r in rows], int)
    N = np.array([r["n"] for r in rows], int)
    role = np.array([r["role_code"] for r in rows], float)
    Z, mu, sd = design(rows, cols)
    jr, ji = cols.index("renter_share"), cols.index("log_income")
    ix_r = role * Z[:, jr]
    ix_y = role * Z[:, ji]

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
        if with_income_ix:
            theta_y = pm.Normal("theta_role_x_income", 0, 1.0)
            eta = eta + theta_y * ix_y
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
        # the two implied renter-share slopes, as derived quantities
        pm.Deterministic("slope_renter_dais", beta[jr])
        pm.Deterministic("slope_renter_public", beta[jr] + theta_r)
        pm.Binomial("y", n=N, p=pm.math.sigmoid(eta), observed=y)
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS,
                          target_accept=0.95, random_seed=SEED,
                          progressbar=False)
        pm.sample_posterior_predictive(idata, var_names=["y"],
                                       extend_inferencedata=True,
                                       random_seed=SEED, progressbar=False)

    post = idata.posterior
    params = {}
    for j, c in enumerate(cols):
        params[c] = psum(np.asarray(post["beta"])[..., j])
    for nm in ("alpha", "delta_role", "theta_role_x_renter",
               "theta_role_x_income", "sigma_body", "sigma_row",
               "slope_renter_dais", "slope_renter_public"):
        if nm in post.data_vars:
            params[nm] = psum(np.asarray(post[nm]))
    if "gamma_route" in post.data_vars:
        gm = np.asarray(post["gamma_route"])
        for i, rt in enumerate(routes):
            if i == 0:
                continue
            params[f"gamma[{rt}]"] = psum(gm[..., i])

    # the implied slope DIFFERENCE is theta_role_x_renter by construction; also
    # report the slope on the covariate's own scale (per +10 points of renter
    # share) for both roles
    sdr = float(sd[jr])
    for nm, key in (("slope_renter_dais", "dais"),
                    ("slope_renter_public", "public")):
        params[nm]["per_10pt_renter"] = params[nm]["median"] * 0.10 / sdr

    ppc = np.asarray(idata.posterior_predictive["y"]).reshape(-1, n)
    lo, hi = np.percentile(ppc, 5, axis=0), np.percentile(ppc, 95, axis=0)
    inside = int(((y >= lo) & (y <= hi)).sum())

    # ---- frequentist: same rows, same terms, route dummies explicit
    Xcols = list(cols) + ["role", "role_x_renter"] + (
        ["role_x_income"] if with_income_ix else [])
    Xf = [Z, role[:, None], ix_r[:, None]] + (
        [ix_y[:, None]] if with_income_ix else [])
    for i, rt in enumerate(routes):
        if i == 0:
            continue
        Xf.append((ridx == i).astype(float)[:, None])
        Xcols.append(f"route[{rt}]")
    Xf.append(np.ones((n, 1)))
    Xcols.append("intercept")
    bf, se3, sec, conv, G = logit_sandwich(np.column_stack(Xf), y, N,
                                           [r["body"] for r in rows])
    sanity = {}
    for j, c in enumerate(Xcols):
        sanity[c] = dict(beta=float(bf[j]), se_hc3=float(se3[j]),
                         hc3_lo95=float(bf[j] - 1.96 * se3[j]),
                         hc3_hi95=float(bf[j] + 1.96 * se3[j]),
                         se_cluster_body=float(sec[j]),
                         cl_lo95=float(bf[j] - 1.96 * sec[j]),
                         cl_hi95=float(bf[j] + 1.96 * sec[j]))
    jr_f, jix = Xcols.index("renter_share"), Xcols.index("role_x_renter")
    sanity["slope_renter_public (freq)"] = dict(
        beta=float(bf[jr_f] + bf[jix]), se_hc3=float("nan"),
        hc3_lo95=float("nan"), hc3_hi95=float("nan"),
        se_cluster_body=float("nan"), cl_lo95=float("nan"),
        cl_hi95=float("nan"))

    return dict(
        label=label, outcome=ycol, spec="R2" if with_income_ix else "R1",
        paired_only=all(r["paired"] for r in rows) and len(rows) < 64,
        covariates=cols, n_rows=n, n_bodies=len(bodies),
        n_cells=len({(r["route"], r["body"]) for r in rows}),
        n_clusters=int(N.sum()), n_events=int(y.sum()), routes=routes,
        n_dais_rows=int((role == 0).sum()), n_public_rows=int((role == 1).sum()),
        cov_mean=dict(zip(cols, map(float, mu))),
        cov_sd=dict(zip(cols, map(float, sd))),
        params=params, diagnostics=diag(idata),
        ppc_rows_inside_90=inside, sanity=sanity,
        sanity_converged=conv, sanity_n_body_clusters=G), idata


def main():
    MOUT.mkdir(parents=True, exist_ok=True)
    cols = json.loads((OUT2 / "screen.json").read_text())["kept"]
    meta = json.loads((OUT / "cells.json").read_text())
    rows = load()
    rows_p = load(paired_only=True)
    print(f"{STAMP}\ncovariates (run-2 VIF screen, unchanged): {cols}")
    print(f"stacked rows {len(rows)} ({meta['all_rows']['bodies']} bodies, "
          f"{meta['all_rows']['cells']} cells); paired rows {len(rows_p)}")

    results = []
    for ycol, nice in OUTCOMES:
        for spec, wi in (("R1", False), ("R2", True)):
            lab = (f"{nice} — {spec}: role + role x renter"
                   + (" + role x income" if wi else ""))
            r, _ = fit(rows, ycol, cols, wi, lab)
            results.append(r)
            P = r["params"]
            print(f"\n{lab}\n  delta_role          "
                  f"{P['delta_role']['median']:+.3f} "
                  f"90% [{P['delta_role']['ci90'][0]:+.3f},"
                  f"{P['delta_role']['ci90'][1]:+.3f}] "
                  f"Pr>0 {P['delta_role']['pr_gt_0']:.3f}")
            for nm in ("renter_share", "theta_role_x_renter",
                       "slope_renter_dais", "slope_renter_public",
                       "log_income", "theta_role_x_income"):
                if nm not in P:
                    continue
                print(f"  {nm:20s}{P[nm]['median']:+.3f} "
                      f"90% [{P[nm]['ci90'][0]:+.3f},{P[nm]['ci90'][1]:+.3f}] "
                      f"95% [{P[nm]['ci95'][0]:+.3f},{P[nm]['ci95'][1]:+.3f}] "
                      f"Pr>0 {P[nm]['pr_gt_0']:.3f}")
            d = r["diagnostics"]
            print(f"  R-hat {d['max_r_hat']:.4f} ESS {d['min_ess_bulk']:.0f} "
                  f"div {d['divergences']} PPC "
                  f"{r['ppc_rows_inside_90']}/{r['n_rows']} "
                  f"sigma_body {P['sigma_body']['median']:.3f} "
                  f"sigma_row {P['sigma_row']['median']:.3f}")
        # paired-only sensitivity on the richer spec
        r, _ = fit(rows_p, ycol, cols, True,
                   f"{nice} — R2 paired cells only (sensitivity)")
        r["paired_only"] = True
        results.append(r)
        P = r["params"]
        print(f"\n{r['label']}  ({r['n_rows']} rows)")
        for nm in ("delta_role", "theta_role_x_renter", "slope_renter_dais",
                   "slope_renter_public"):
            print(f"  {nm:20s}{P[nm]['median']:+.3f} "
                  f"90% [{P[nm]['ci90'][0]:+.3f},{P[nm]['ci90'][1]:+.3f}] "
                  f"Pr>0 {P[nm]['pr_gt_0']:.3f}")
        print(f"  R-hat {r['diagnostics']['max_r_hat']:.4f} "
              f"div {r['diagnostics']['divergences']}")

    with open(MOUT / "model_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["outcome", "spec", "paired_only", "parameter", "n_rows",
                    "n_bodies", "n_clusters", "median", "ci90_lo", "ci90_hi",
                    "ci95_lo", "ci95_hi", "pr_gt_0", "excl_zero_90",
                    "excl_zero_95", "max_r_hat", "min_ess", "divergences",
                    "ppc_inside_90"])
        for r in results:
            for nm, v in r["params"].items():
                w.writerow([r["outcome"], r["spec"], int(r["paired_only"]), nm,
                            r["n_rows"], r["n_bodies"], r["n_clusters"],
                            f"{v['median']:+.4f}", f"{v['ci90'][0]:+.4f}",
                            f"{v['ci90'][1]:+.4f}", f"{v['ci95'][0]:+.4f}",
                            f"{v['ci95'][1]:+.4f}", f"{v['pr_gt_0']:.3f}",
                            int(v["excludes_zero_90"]),
                            int(v["excludes_zero_95"]),
                            f"{r['diagnostics']['max_r_hat']:.4f}",
                            f"{r['diagnostics']['min_ess_bulk']:.0f}",
                            r["diagnostics"]["divergences"],
                            f"{r['ppc_rows_inside_90']}/{r['n_rows']}"])

    with open(MOUT / "slopes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["outcome", "spec", "paired_only", "role", "slope_median",
                    "ci90_lo", "ci90_hi", "ci95_lo", "ci95_hi", "pr_gt_0",
                    "per_10pt_renter"])
        for r in results:
            for nm, role in (("slope_renter_dais", "dais"),
                             ("slope_renter_public", "public")):
                v = r["params"][nm]
                w.writerow([r["outcome"], r["spec"], int(r["paired_only"]),
                            role, f"{v['median']:+.4f}", f"{v['ci90'][0]:+.4f}",
                            f"{v['ci90'][1]:+.4f}", f"{v['ci95'][0]:+.4f}",
                            f"{v['ci95'][1]:+.4f}", f"{v['pr_gt_0']:.3f}",
                            f"{v['per_10pt_renter']:+.4f}"])

    with open(MOUT / "sanity_hc3.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["outcome", "spec", "paired_only", "term", "logit_beta",
                    "se_hc3", "hc3_lo95", "hc3_hi95", "se_cluster_body",
                    "cl_lo95", "cl_hi95", "converged", "n_body_clusters"])
        for r in results:
            for c, v in r["sanity"].items():
                w.writerow([r["outcome"], r["spec"], int(r["paired_only"]), c,
                            f"{v['beta']:+.4f}", f"{v['se_hc3']:.4f}",
                            f"{v['hc3_lo95']:+.4f}", f"{v['hc3_hi95']:+.4f}",
                            f"{v['se_cluster_body']:.4f}",
                            f"{v['cl_lo95']:+.4f}", f"{v['cl_hi95']:+.4f}",
                            int(r["sanity_converged"]),
                            r["sanity_n_body_clusters"]])

    (MOUT / "model_results.json").write_text(json.dumps(dict(
        stamp=STAMP, seed=SEED, draws=DRAWS, tune=TUNE, chains=CHAINS,
        route_reference=ROUTE_REF, covariates=cols, cells_meta=meta,
        role_code={"dais": 0, "public": 1},
        loo_note=("PSIS-LOO deliberately not reported: plan/RENTER-SHARE-2.md "
                  "3d showed it is degenerate with one per-observation "
                  "overdispersion term, and this model has two random-effect "
                  "terms (body, row)."),
        results=results), indent=2, default=str))
    print(f"\n-> {MOUT}")


if __name__ == "__main__":
    main()
